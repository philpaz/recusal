"""
What a gate can judge: the data-only invocation, held in one place.

A gate in front of a tool call makes one promise: the arguments the policy judged are the
arguments the tool ran with. Keeping it takes the same few decisions in every framework,
and they are decisions about data, not about any framework:

- which parameter annotations a gate can judge by value (:func:`supported_annotation`);
- which values are JSON data, so a default can be completed and judged
  (:func:`json_data`);
- whether every parameter of a callable can be judged by name
  (:func:`unsupported_signature`), counting every callable a call reaches
  (:func:`reached_parameters`), and refusing two shapes the walk cannot read: a wrapper
  that holds a second ``functools.wraps`` wrapper of the same function
  (:class:`HiddenWrapper`), and a single-dispatch function that can run anything but its
  base (:class:`DispatchedAtCallTime`). Both checks are exact, run nothing, and state
  what they do not find. A callable ``inspect.signature`` reads as a
  ``functools.partialmethod`` shows parameters that are not its own and is refused too
  (``TypeError``);
- how to hand a policy a copy it can write to without reaching what executes
  (:func:`detached`).

The framework gates under ``examples/`` each carried their own copy of these. They import
them from here instead, so the supported set is defined once and a fix reaches every gate
that uses it. Nothing here imports a framework, reads a file or calls a policy; every
function is a pure function of its argument. Standard library only.

What stays in each gate is the part that is the framework's: where its hook sits, how it
prepares an invocation, and which of its own parameters it injects.
"""

import functools
import inspect
import math
import types
import typing
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

#: The scalar types a judged parameter may be annotated with.
SCALARS: Tuple[type, ...] = (str, int, float, bool)

# The PEP 604 union (``int | None``) has its own type from Python 3.10; 3.9 has none.
_UNION_TYPE = getattr(types, "UnionType", None)


def supported_annotation(annotation: Any) -> bool:
    """True for an annotation whose preparation is a pure function of the value: ``str``,
    ``int``, ``float``, ``bool``, ``Optional`` of one, ``list[T]`` and ``dict[str, T]`` with
    ``T`` supported in turn. Everything else (no annotation, ``Any``, a model, an ``Enum``, an
    ``Annotated``, a dataclass, a bare ``list``/``dict``) is unsupported.

    By identity: an object that only compares equal to ``str`` (a metaclass with its own
    ``__eq__``) can carry a schema of its own, which would prepare the value differently."""
    if any(annotation is scalar for scalar in SCALARS):
        return True
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Union or (_UNION_TYPE is not None and origin is _UNION_TYPE):
        rest = [a for a in args if a is not type(None)]
        return len(rest) == 1 and supported_annotation(rest[0])
    if origin is list:
        return len(args) == 1 and supported_annotation(args[0])
    if origin is dict:
        return len(args) == 2 and args[0] is str and supported_annotation(args[1])
    return False


def json_data(value: Any, _depth: int = 0) -> bool:
    """True for a value in the JSON domain (what a default may be), built only of the exact
    JSON types: ``None``, ``str``, ``int``, ``float`` (finite), ``bool``, ``list`` and
    ``dict`` with string keys.

    A subclass of one of them is not JSON data here: its own ``__deepcopy__``, ``__eq__`` or
    serialisation can make the copy a policy judges differ from the value a tool receives.
    A non-finite float is not JSON, and is not equal to itself."""
    if _depth > 64:
        return False
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(json_data(v, _depth + 1) for v in value)
    if type(value) is dict:
        return all(type(k) is str and json_data(v, _depth + 1) for k, v in value.items())
    return False


def unsupported_signature(func: Callable[..., Any], skip: Tuple[str, ...] = ()) -> Optional[str]:
    """Why this callable's parameters cannot all be judged, or None if they can.

    ``skip`` names the parameters the framework injects itself (a context object, a run
    manager); they are not judged and not checked. Name them explicitly: a parameter that
    is merely absent from a schema is not one of them."""
    try:
        parameters = inspect.signature(func).parameters
        # resolves string annotations (PEP 563) and keeps Annotated so it can be refused
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception:  # noqa: BLE001, an uninspectable callable cannot be judged
        return "its signature cannot be inspected"
    for name, parameter in parameters.items():
        if name in skip:
            continue
        annotation = hints.get(name, parameter.empty)
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            return f"`{name}` is variadic; every argument the tool receives must be judged by name"
        if parameter.kind is parameter.POSITIONAL_ONLY:
            return f"`{name}` is positional-only and cannot be judged by name"
        if not supported_annotation(annotation):
            return f"`{name}` is not annotated with a supported data type"
        if parameter.default is not parameter.empty and not json_data(parameter.default):
            return f"`{name}` has a default outside the JSON domain"
    # ``inspect.signature`` reads a ``functools.wraps`` wrapper as the function it wraps, and
    # honours a declared ``__signature__``: a wrapper's own parameter (``force=True``) would
    # run with its default while the gate judged a call without it
    try:
        reached = _reached(func, skip, parameters)
    except DeclaredSignature:
        return "it declares a signature that is not its own"
    except HiddenWrapper as refused:
        return str(refused)
    except DispatchedAtCallTime:
        return _DISPATCHED_AT_CALL_TIME
    except Exception:  # noqa: BLE001, an uninspectable callable cannot be judged
        return "its signature cannot be inspected"
    return _argument_of_its_own([_read_level(level) for level in reached], parameters, skip)


class DeclaredSignature(Exception):
    """A callable on the way to the code that runs carries a ``__signature__`` of its own."""


class HiddenWrapper(ValueError):
    """A callable on the way holds a second wrapper of a callable the walk reads: one that
    names what it wraps as ``__wrapped__``, that the walk does not reach, and that has a
    parameter of its own, so the walk cannot read every parameter the call reaches.

    Raised for exactly what :func:`_refuse_hidden_wrapper` describes, which also lists what
    is not found: among others a second wrapper that names nothing as ``__wrapped__``, and
    one that is or is held by an object, a proxy or a wrapper a C extension builds. The
    message is the reason a wrapper the walk reads would get for the same parameter; it
    names that parameter and no value."""


_HIDDEN_WRAPPER = (
    "a wrapper holds another wrapper of the function it wraps, which the walk does not follow"
)


class DispatchedAtCallTime(ValueError):
    """A callable on the way is the function ``functools.singledispatch`` returned, or the
    callable a ``functools.singledispatchmethod`` hands out, and can run something other
    than the base function it names as ``__wrapped__``: an implementation is registered
    beside the base, or in its place. Which runs is chosen at call time by the type of an
    argument, so the walk cannot read the parameters the call reaches.
    :func:`_refuse_call_time_dispatch` lists what is not found."""


_DISPATCHED_AT_CALL_TIME = (
    "the function that runs is chosen at call time by the type of an argument, which the "
    "walk cannot read"
)

#: Wrappers that pass every argument through unchanged and declare no parameters of their
#: own, so the code that runs is the function they wrap: ``functools.lru_cache`` and
#: ``functools.cache``.
_PASS_THROUGH = (type(functools.lru_cache(maxsize=None)(lambda: None)),)

#: A ``__call__`` implemented in C (a slot wrapper or a bound one): it carries no
#: ``__signature__`` of its own and has nothing further in to read. A bound method is one
#: such callable: its own ``__dict__`` shows its function's attributes, so a signature
#: declared there is found, and its ``__call__`` is where the walk stops.
_C_CALL = (types.WrapperDescriptorType, types.MethodWrapperType)


def own_parameters(func: Any, _depth: int = 0) -> Any:
    """The parameters Python binds when ``func`` is called: the callable's own, never a
    signature it shows for something else.

    ``inspect.signature`` follows ``__wrapped__`` and honours a declared ``__signature__``,
    on the callable or one level in (a callable object's ``__call__``, a
    ``functools.partial``'s function; a method shows its function's attributes as its
    own). The first is turned off; the second raises :class:`DeclaredSignature` wherever it
    is found. Wrappers that only pass arguments through (``lru_cache``, ``cache``, a
    ``staticmethod``, and ``lru_cache`` over a method) are looked through. A class is not a
    callable this reads: calling it runs ``__new__`` and ``__init__``. Neither is the method
    ``functools.partialmethod`` builds over a callable that does not bind as one, nor
    anything else ``inspect.signature`` would read as a ``functools.partialmethod``
    (:func:`_refuse_partialmethod`): that raises ``TypeError``.
    """
    if _depth > 8:
        raise ValueError("too many wrappers to inspect")
    if isinstance(func, type):
        raise TypeError("a class is not inspected as a callable")
    if isinstance(func, staticmethod):
        return own_parameters(func.__func__, _depth + 1)
    if isinstance(func, _PASS_THROUGH):
        return own_parameters(func.__wrapped__, _depth + 1)
    if inspect.ismethod(func) and isinstance(func.__func__, _PASS_THROUGH):
        inner = func.__func__.__wrapped__
        bound = types.MethodType(inner, func.__self__) if inspect.isfunction(inner) else inner
        return own_parameters(bound, _depth + 1)
    _refuse_declared_signature(func, _depth)
    return inspect.signature(func, follow_wrapped=False).parameters


def _refuse_declared_signature(func: Any, depth: int) -> None:
    """Raise :class:`DeclaredSignature` if ``func``, or what ``inspect.signature`` would
    read in its place, declares ``__signature__``, and ``TypeError`` if either would be
    read as a ``functools.partialmethod`` (:func:`_refuse_partialmethod`)."""
    if depth > 8:
        raise ValueError("too many wrappers to inspect")
    if "__signature__" in getattr(func, "__dict__", {}) or hasattr(type(func), "__signature__"):
        raise DeclaredSignature
    _refuse_partialmethod(func)
    if isinstance(func, functools.partial):
        _refuse_declared_signature(func.func, depth + 1)
    elif not (inspect.isfunction(func) or inspect.isbuiltin(func) or isinstance(func, type)):
        call = getattr(type(func), "__call__", None)
        if call is not None and not isinstance(call, _C_CALL):
            _refuse_declared_signature(call, depth + 1)


#: The attribute under which ``functools.partialmethod`` leaves itself on the function it
#: builds, and under which ``inspect.signature`` looks for it: ``_partialmethod`` up to
#: Python 3.12, ``__partialmethod__`` from 3.13. Both are looked for on every Python.
_PARTIALMETHOD_MARKS = ("_partialmethod", "__partialmethod__")

_PARTIALMETHOD = (
    "`inspect.signature` reads it as a `functools.partialmethod`, which shows parameters "
    "that are not the callable's own"
)


def _refuse_partialmethod(func: Any) -> None:
    """Raise ``TypeError`` if ``inspect.signature`` would read ``func`` as a
    ``functools.partialmethod``: ``func`` shows, under one of the two names above, an
    instance of ``functools.partialmethod``.

    ``functools.partialmethod(f, ...)`` hands out one of two things. Where ``f`` binds as a
    method (a plain function taken from an instance, a ``classmethod``, a
    ``staticmethod``), a ``functools.partial`` of the bound ``f``: the walk follows that,
    and it is not refused here. Where ``f`` does not (a callable object, a class-based
    decorator; a plain function when the method is taken from the class), a method of a
    function ``functools`` builds, which calls ``f`` from its closure, names nothing as
    ``__wrapped__`` and carries the mark. ``inspect.signature`` reads the mark in place of
    that function's own parameters and shows the parameters of ``f`` less what is bound, so
    the walk would read one level that is no callable's own and nothing of ``f``: not its
    ``__call__``, not a wrapper under it, not a default any of them fills.

    ``functools.update_wrapper`` copies the mark onto every wrapper of such a method, and
    ``inspect.signature`` then reads the mark in place of that wrapper's parameters too. So
    the mark is refused wherever ``inspect.signature`` looks for it: on the callable, on a
    method's function (a method shows its function's attributes as its own), on a
    ``functools.partial``'s function and on a callable object's ``__call__``, which are the
    places :func:`_refuse_declared_signature` visits. The test is the one
    ``inspect.signature`` makes, an attribute lookup on a callable of the walk and an
    ``isinstance``; a mark that is anything but a ``functools.partialmethod`` is not read by
    it and is not refused. Nothing is prepared for it and nothing found is called.

    Not found, by design: a partial application written by hand, and a callable that
    reaches such a method through a helper that names nothing as ``__wrapped__`` and does
    not carry the mark."""
    for mark in _PARTIALMETHOD_MARKS:
        if isinstance(getattr(func, mark, None), functools.partialmethod):
            raise TypeError(_PARTIALMETHOD)


# The two checks below read what a callable on the walk HOLDS. A held value is not a
# callable the walk was asked to read: it can be a lazy proxy that connects when touched, a
# remote stub that answers every attribute, a settings object. So the checks are exact and
# run nothing. A held value is looked at only when its type, read with ``type()``, is one
# of a closed set CPython itself builds, and then only through that type's own C-level
# accessors: no attribute lookup, no descriptor a class declares, no comparison but
# identity, no hashing. A value of any other type is skipped without being read at all.
#
# Two rules hold for every dictionary read here that is not this module's own (a function's
# ``__dict__``, an instance's, a class's, a registry). It is listed in one C-level step
# (``list(d.items())`` of a plain ``dict``) before any loop, so another thread changing it
# cannot break the loop. And no key is looked up in it: a key is compared only when it is
# exactly a ``str``, so no key's ``__hash__`` or ``__eq__`` runs.

#
# Nothing of this is built when the module is imported: importing it does what it did before
# these checks existed and no more, whatever stands in ``functools``, in ``sys.modules`` or
# in the audit hooks. The accessors and the standard-library objects the checks rely on are
# found at the first walk (:func:`_prepare`), each one identified as the standard library's
# own before it is used, and none of them called. If any cannot be, both checks are off:
# the walk is then what it was before them, and nothing is raised.

_MISSING: Any = object()
_DEPTH = 32
_VARIADIC = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS
_HEAP_TYPE, _IMMUTABLE_TYPE = 1 << 9, 1 << 8  # ``Py_TPFLAGS_HEAPTYPE``, ``..._IMMUTABLETYPE``

#: None until the first walk; then True if the checks are on, False if they are off.
_READY: Optional[bool] = None
_function_dict: Any = None
_function_code: Any = None
_method_function: Any = None
_static_function: Any = None
_class_function: Any = None
_type_mro: Any = None
_type_dict: Any = None
_module_dict: Any = None
#: The standard library's ``functools.partial`` and ``functools.lru_cache`` wrapper types,
#: and the accessors of what they hold.
_PARTIAL: Any = None
_partial_function: Any = None
_partial_args: Any = None
_CACHE_WRAPPER: Any = None
_cache_dict: Any = None
#: ``functools.singledispatch`` is written in Python: the dispatcher it returns is the
#: plain function whose code is that of its inner ``wrapper``, and the registry lives in
#: the closure of its inner ``dispatch``. None where they cannot be identified, and the
#: dispatch check is then off: it refuses nothing and reads nothing.
_WRAPPER_CODE: Any = None
_DISPATCH_CODE: Any = None
#: Where the callable ``functools.singledispatchmethod`` hands out keeps the way to its
#: registry (:func:`_method_access`), or None.
_METHOD_ACCESS: Any = None


def _prepare() -> bool:
    """Find, once, what the two checks read with; True if they are on.

    Run at the first walk, not at import. Everything is read through the C-level accessors
    of built-in types and from dictionaries listed as :func:`_entry` lists them; nothing
    found is called, and no attribute is looked up on ``functools`` or on anything in it.
    The checks are on only if ``functools`` is a plain module whose ``partial`` and whose
    ``lru_cache`` wrapper are types C code defines, with the accessors those have. What
    stands in ``functools`` is read as it is then: a ``functools`` reloaded afterwards
    builds dispatchers from new code objects, which are not recognised. Any ``Exception`` on
    the way (an audit hook that refuses a read, an interpreter built another way) turns
    them off. What the first walk finds stands for the process: a ``functools`` patched
    when it is made and restored afterwards leaves the checks as that walk found them."""
    global _READY, _function_dict, _function_code, _method_function, _static_function
    global _class_function, _type_mro, _type_dict, _module_dict
    global _PARTIAL, _partial_function, _partial_args, _CACHE_WRAPPER, _cache_dict
    global _WRAPPER_CODE, _DISPATCH_CODE, _METHOD_ACCESS
    if _READY is not None:
        return _READY
    try:
        # the accessors of built-in types: the same in every thread that gets here
        _type_mro = type.__dict__["__mro__"].__get__
        _type_dict = type.__dict__["__dict__"].__get__
        _function_dict = _type_dict(types.FunctionType)["__dict__"].__get__
        _function_code = _type_dict(types.FunctionType)["__code__"].__get__
        _method_function = _type_dict(types.MethodType)["__func__"].__get__
        _static_function = _type_dict(staticmethod)["__func__"].__get__
        _class_function = _type_dict(classmethod)["__func__"].__get__
        _module_dict = _type_dict(types.ModuleType)["__dict__"].__get__
        found = _standard_library()
    except Exception:  # noqa: BLE001, whatever cannot be read turns the checks off
        found = None
    # threads that make their first walk at once each get here with what they found; the
    # first to arrive decides for the process, in one step, and every thread sets the same
    # values, so no walk ever sees a part of one finding beside a part of another
    found = _DECIDED.setdefault("found", (found,))[0]
    if found is not None:
        _PARTIAL, _partial_function, _partial_args, _CACHE_WRAPPER, _cache_dict = found[:5]
        _WRAPPER_CODE, _DISPATCH_CODE, _METHOD_ACCESS = found[5:]
    _READY = found is not None  # last: once it is set, everything above is
    return _READY


#: What the first walk of the process found (:func:`_prepare`), under the one key "found".
_DECIDED: Dict[str, Any] = {}


def _standard_library() -> Any:
    """What the checks read ``functools`` with, if all of it is the standard library's own
    (:func:`_prepare`), else None: the ``partial`` type and the accessors of its function
    and arguments, the ``lru_cache`` wrapper type and the accessor of its dictionary, the
    two code objects of a dispatcher and where a single-dispatch method keeps its
    registry (those three None where the dispatcher is not identified)."""
    type_flags = _type_dict(type)["__flags__"].__get__

    def compiled(kind: Any) -> bool:
        # a type C code defines: static, or declared immutable (a flag no class written in
        # Python can have), so not a look-alike put in its place
        return type(kind) is type and (
            not type_flags(kind) & _HEAP_TYPE or bool(type_flags(kind) & _IMMUTABLE_TYPE)
        )

    if type(functools) is not types.ModuleType:
        return None
    names = _module_dict(functools)
    partial, cache = _entry(names, "partial"), _PASS_THROUGH[0]
    # the wrapper type is the one ``functools`` itself names, not whatever a replaced
    # ``lru_cache`` returned when this module was imported
    its_own = cache is _entry(names, "_lru_cache_wrapper")
    if not (compiled(partial) and compiled(cache) and its_own):
        return None
    wrapper = _nested_code(names, "singledispatch", "wrapper")
    dispatch = _nested_code(names, "singledispatch", "dispatch")
    method = _method_access(names)
    if wrapper is None or dispatch is None:
        wrapper = dispatch = method = None
    return (
        partial,
        _type_dict(partial)["func"].__get__,
        _type_dict(partial)["args"].__get__,
        cache,
        _type_dict(cache)["__dict__"].__get__,
        wrapper,
        dispatch,
        method,
    )


def _entry(attributes: Any, name: str) -> Any:
    """The value under the string ``name`` in a plain ``dict`` (or in the ``mappingproxy``
    of a class), or ``_MISSING``: the items are listed in one step, and no key is hashed or
    compared unless it is exactly a ``str``."""
    if type(attributes) is not dict and type(attributes) is not types.MappingProxyType:
        return _MISSING
    for key, value in list(attributes.items()):
        if type(key) is str and key == name:
            return value
    return _MISSING


def _own(value: Any, name: str) -> Any:
    """The entry ``name`` in the own ``__dict__`` of a plain function or of a
    ``functools.lru_cache`` wrapper, or ``_MISSING``; ``_MISSING`` for any other value."""
    if type(value) is types.FunctionType:
        return _entry(_function_dict(value), name)
    if type(value) is _CACHE_WRAPPER:
        return _entry(_cache_dict(value), name)
    return _MISSING


def _cell(function: Any, name: str) -> Any:
    """What a plain function's closure holds for the free variable ``name``, or
    ``_MISSING``."""
    for index, free in enumerate(function.__code__.co_freevars):
        if free == name:
            try:
                return function.__closure__[index].cell_contents
            except ValueError:  # an empty cell holds nothing
                return _MISSING
    return _MISSING


def _instance_attributes(value: Any) -> Any:
    """The instance dictionary of ``value``, or an empty one: read through the C-level
    ``__dict__`` accessor of its class, found by listing the dictionaries of that class. A
    class that answers ``__dict__`` any other way (a property) has none this reads, and
    neither has an instance whose dictionary is not a plain ``dict``."""
    for klass in _type_mro(type(value)):
        accessor = _entry(_type_dict(klass), "__dict__")
        if accessor is not _MISSING:
            break
    else:
        return {}
    if type(accessor) is not types.GetSetDescriptorType:
        return {}
    try:
        found = accessor.__get__(value, type(value))
    except Exception:  # noqa: BLE001, no instance dictionary
        return {}
    return found if type(found) is dict else {}


def _cells(function: Any) -> List[Any]:
    held = []
    for cell in function.__closure__ or ():
        try:
            held.append(cell.cell_contents)
        except ValueError:  # an empty cell holds nothing
            continue
    return held


def _held(func: Any, wrapped: Any) -> List[Any]:
    """What the walked ``func`` holds: the closure cells of a plain function, the instance
    attributes of any other callable. An instance attribute that is, by identity, also an
    attribute of the plain function ``func`` names as ``__wrapped__`` is left out:
    ``functools.update_wrapper`` copied it over, and a function's own attributes are not
    read."""
    if type(func) is types.FunctionType:
        return _cells(func)
    copied = _function_dict(wrapped) if type(wrapped) is types.FunctionType else {}
    copied = list(copied.values()) if type(copied) is dict else []
    return [
        value
        for value in list(_instance_attributes(func).values())
        if not any(value is other for other in copied)
    ]


def _passes_on_to(value: Any) -> Any:
    """What a callable of the closed set passes a call on to, or ``_MISSING``.

    The closed set, each read by its exact type: a plain function and a
    ``functools.lru_cache`` wrapper (what its own ``__dict__`` names as ``__wrapped__``), a
    bound method (its function), a ``functools.partial`` (its function), a ``staticmethod``
    and a ``classmethod`` (its function). Anything else, a subclass of one of these
    included, is ``_MISSING``: it is not read. What one of these passes the call on to is
    read in its turn only if it is of the closed set too."""
    kind = type(value)
    if kind is types.FunctionType or kind is _CACHE_WRAPPER:
        return _own(value, "__wrapped__")
    if kind is types.MethodType:
        return _method_function(value)
    if kind is _PARTIAL:
        return _partial_function(value)
    if kind is staticmethod:
        return _static_function(value)
    if kind is classmethod:
        return _class_function(value)
    return _MISSING


def _level(named: Any, forwards: bool) -> Tuple[Tuple[str, ...], bool]:
    """One callable's parameters as the rule below reads them: the names it takes, and
    whether it also takes ``*args`` or ``**kwargs`` and so can pass a call on."""
    return tuple(named), forwards


def _read_level(parameters: Any) -> Tuple[Tuple[str, ...], bool]:
    """The level of a callable the walk read (:func:`own_parameters`)."""
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    named = [name for name, parameter in parameters.items() if parameter.kind not in variadic]
    return _level(named, len(named) < len(parameters))


def _held_level(function: Any, bound: int) -> Tuple[Tuple[str, ...], bool]:
    """The level of a held plain function, after the ``bound`` leading arguments it
    receives as a method: read from the code object Python binds a call with, never from a
    signature it shows."""
    code = function.__code__
    names, positional = code.co_varnames, code.co_argcount
    named = names[bound:positional]
    named += names[positional : positional + code.co_kwonlyargcount]
    return _level(named, bool(code.co_flags & _VARIADIC))


def _argument_of_its_own(
    levels: List[Any], declared: Any, skip: Any, judged: Optional[Any] = None
) -> Optional[str]:
    """Why some callable a call reaches would run with a value no policy judged, or None.

    The one rule for a parameter a wrapper has of its own, for a wrapper the walk reads and
    for one a callable holds alike. ``levels`` are the callables' parameters
    (:func:`_level`), ``declared`` the names the tool declares, ``skip`` the names the
    framework injects. A name some level takes that the tool does not declare would be
    filled from that level's default. A name a forwarding level takes (one with ``*args``
    or ``**kwargs``) that another level takes too is kept by the wrapper, and the function
    runs with its own default. With ``judged`` (indexes into ``levels``), only a finding
    that involves one of those levels counts."""
    for index, (named, _) in enumerate(levels):
        if judged is not None and index not in judged:
            continue
        for name in named:
            if name in skip or name in declared:
                continue
            return (
                f"`{name}` is a parameter of a function the call reaches that the tool does "
                "not declare, so it would run with a value no policy judged"
            )
    # a forwarding wrapper (``*args``/``**kwargs``) that also names a parameter takes that
    # argument itself; a function further in that names it too then runs with its own
    # default, not the value the policy judged
    for index, (named, forwards) in enumerate(levels):
        if not forwards:
            continue
        for name in sorted(set(named) - set(skip)):
            for other, (names, _) in enumerate(levels):
                if other == index or name not in names:
                    continue
                if judged is None or index in judged or other in judged:
                    return (
                        f"`{name}` is taken by a wrapper and by the function it wraps, which "
                        "can then run with a value no policy judged"
                    )
    return None


def _walk() -> Any:
    """What one walk has read (``read``) and found held (``held``), for the judgement made
    at its end, and whether the two checks are on (``checked``, :func:`_prepare`: where it
    is False the walk collects nothing held). A plain namespace, not a class of this
    module: importing the module defines no class but the two exceptions it adds."""
    return types.SimpleNamespace(read=[], held=[], checked=_prepare())


def _hidden_levels(value: Any, read: List[Any], bound: int) -> List[Any]:
    """The levels of the wrappers the walk did not read between a held ``value`` and a
    callable it did read, or none when ``value`` reaches no such callable.

    ``value`` is followed through the closed set (:func:`_passes_on_to`) until a link is,
    by identity, one of ``read``. Each plain function passed on the way is a wrapper the
    walk never read, and its parameters are returned for the same judgement the walk's own
    levels get. ``bound`` is the number of leading arguments the callable that holds
    ``value`` receives bound (its receiver, when the walk reached it as a method): the
    held wrapper is passed them too, so they are not parameters of its own, exactly as
    they are not for the level the walk read. Each bound method, ``classmethod`` and bound
    positional argument of a ``functools.partial`` on the way adds to it. Raises
    :class:`HiddenWrapper` when the way cannot be judged by parameters: it is still going
    after 32 links, or a link is a single-dispatch function or method that can run
    anything but its base."""
    links: List[Any] = []
    levels: List[Any] = []
    dispatches = False
    for _ in range(_DEPTH):
        if any(value is seen for seen in read):
            if dispatches:
                raise HiddenWrapper(_HIDDEN_WRAPPER)
            return levels
        if any(value is link for link in links):
            return []  # a cycle: it reaches nothing the walk reads
        inner = _passes_on_to(value)
        if inner is _MISSING:
            return []
        if type(value) is types.FunctionType:
            levels.append(_held_level(value, bound))
            dispatches = dispatches or _chosen_at_call_time(value)
        links.append(value)
        if type(value) is _PARTIAL:
            args = _partial_args(value)
            bound += len(args) if type(args) is tuple else 0
        elif type(value) is types.MethodType or type(value) is classmethod:
            bound += 1
        value = inner
    raise HiddenWrapper(_HIDDEN_WRAPPER)  # still going: what it reaches is not known


def _refuse_hidden_wrapper(
    func: Any, found: List[Any], walk: Any, skip: Any, declared: Any = None
) -> None:
    """Raise :class:`HiddenWrapper` if a callable on the walk holds a wrapper the walk did
    not read, with a parameter of its own. ``declared`` is what the tool declares
    (``inspect.signature(func).parameters``) when the caller has read it already, so that
    the tool is not read for it a second time.

    A decorator can apply another wrapper inside itself and call the result, while
    ``functools.wraps`` sets ``__wrapped__`` to the undecorated function: the walk then
    never reads the wrapper in between, which can keep an argument for itself or fill a
    default of its own. This finds exactly one shape of that, and runs nothing to do it.

    A callable on the walk that names a ``__wrapped__`` holds what is in its closure cells
    (a plain function) or its instance attributes (any other callable). A held value is a
    hidden wrapper when all of these hold:

    - it is of the closed set of :func:`_passes_on_to`;
    - what it passes a call on to, followed through that same set, reaches by identity a
      callable the walk read: any callable this walk took the parameters of, or the
      function of a method or a ``functools.partial`` it took them through;
    - a plain function passed on the way has a parameter of its own, by the same rule and
      the same code that judge a wrapper the walk reads (:func:`_argument_of_its_own`:
      a name the tool does not declare, or a name a forwarding wrapper keeps that another
      level takes too). A wrapper that only passes ``(*args, **kwargs)`` on has none, and
      neither has one that copies the tool's own parameters. The leading arguments the
      holder receives bound (its receiver, when the walk reached it as a method) are not
      parameters of the held wrapper either. A way still going after 32 links, and one
      through a single-dispatch function or method that can run more than its base, are
      refused as they stand.

    Only parameters are judged: a held wrapper is not read for the other things the walk
    refuses a callable it reads for (a declared ``__signature__``). And what a callable
    object holds is judged whether or not its ``__call__`` uses it: an extra entry point
    kept on a class-based decorator (``self.batch = batched(self.function)``) is refused
    if it has a parameter of its own.

    Not found, by design: a second wrapper that does not itself name what it wraps (a plain
    inner function, a lambda, a helper, a bound method of an object that holds the
    function); one that is, or is held by, an object of any other type (a callable object,
    a proxy, a wrapper a C extension builds, a subclass of a type of the closed set); one
    held anywhere but the closure cells or instance attributes of a callable on the walk (a
    class attribute, a slot, a module global, a container, a function's own attribute); a
    class-based decorator whose own method keeps the argument; a wrapper that takes an
    argument in its body alone; the arguments a ``functools.partial`` binds; a held wrapper
    that cannot take the bound arguments at all and has no parameter
    of its own (the call fails when it is made); and everything, where ``functools.partial``
    or the ``functools.lru_cache`` wrapper is not the standard library's own type at the
    first walk (:func:`_prepare`: this check is then off).

    One shape gets a different verdict held than read by the walk: under a tool that is a
    ``functools.partial`` binding a keyword (``partial(tool, limit=3)``), a held wrapper
    that takes that name and ``*args`` but no ``**kwargs`` is refused, while the same
    wrapper read by the walk is accepted, because ``inspect.signature`` of such a partial
    no longer shows the ``*args``. The call fails either way.
    """
    hidden: List[Any] = []
    for value, bound in walk.held:
        hidden.extend(_hidden_levels(value, walk.read, bound))
    if not hidden:
        return
    if declared is None:  # not handed in by a caller that has read it: read once, here
        try:
            declared = set(inspect.signature(func).parameters)
        except Exception:  # noqa: BLE001, what the tool declares cannot be read
            raise HiddenWrapper(_HIDDEN_WRAPPER) from None
    levels = [_read_level(parameters) for parameters in found] + hidden
    why = _argument_of_its_own(levels, declared, skip, range(len(found), len(levels)))
    if why is not None:
        raise HiddenWrapper(why)


def _own_code(names: Any, function: Any) -> Any:
    """The code object of ``function`` if it is a plain function compiled from the file
    ``functools.total_ordering`` was compiled from (one the standard library wrote, not one
    put in its place), else None. ``names`` is the dictionary of the ``functools`` module."""
    anchor = _entry(names, "total_ordering")
    if type(function) is not types.FunctionType or type(anchor) is not types.FunctionType:
        return None
    code = _function_code(function)
    return code if code.co_filename == _function_code(anchor).co_filename else None


def _code_inside(code: Any, name: str) -> Any:
    """The code object of the function ``name`` defined inside ``code``, or None."""
    for constant in code.co_consts if code is not None else ():
        if type(constant) is types.CodeType and constant.co_name == name:
            return constant
    return None


def _nested_code(names: Any, outer: str, name: str) -> Any:
    """The code object of the function ``name`` defined inside ``functools.<outer>``, or
    None when what stands there is not the standard library's own plain function (it was
    wrapped or replaced) or defines no such function. Nothing is called and no attribute
    is looked up to find out."""
    return _code_inside(_own_code(names, _entry(names, outer)), name)


def _method_access(names: Any) -> Any:
    """Where the callable ``functools.singledispatchmethod`` hands out keeps the way to its
    registry, read from the standard library's own code without building or calling
    anything: the callable's type, its code object if it is a plain function, the name of
    the closure cell or instance attribute, and whether that holds the descriptor or the
    ``dispatch`` function.

    ``singledispatchmethod.__get__`` either defines the function it hands out, ``_method``,
    whose free variables name what it keeps (``self``, the descriptor, on Python 3.9 to
    3.12; ``dispatch`` on 3.13), or hands out an instance of
    ``functools._singledispatchmethod_get``, which keeps the descriptor as ``_unbound``
    (3.14, 3.15). None where ``functools.singledispatchmethod`` or that class is not the
    standard library's own, and where a Python does it another way."""
    method = _entry(names, "singledispatchmethod")
    if type(method) is not type:
        return None
    handing = _own_code(names, _entry(_type_dict(method), "__get__"))
    inner = _code_inside(handing, "_method")
    if inner is not None:
        if "self" in inner.co_freevars:
            return types.FunctionType, inner, "self", True
        if "dispatch" in inner.co_freevars:
            return types.FunctionType, inner, "dispatch", False
        return None
    handed = _entry(names, "_singledispatchmethod_get")
    if handing is None or type(handed) is not type:
        return None
    building = _own_code(names, _entry(_type_dict(handed), "__init__"))
    calling = _own_code(names, _entry(_type_dict(handed), "__call__"))
    if building is None or calling is None or "_unbound" not in building.co_names:
        return None
    return handed, None, "_unbound", True


def _only_runs(dispatch: Any, base: Any) -> bool:
    """True if ``dispatch`` is the ``dispatch`` function of a single-dispatch registry
    that is exactly ``{object: base}``, the key and the entry by identity. The registry is
    the plain ``dict`` in that function's closure; anything else there is not read, and
    the answer is False."""
    if type(dispatch) is not types.FunctionType or dispatch.__code__ is not _DISPATCH_CODE:
        return False
    registry = _cell(dispatch, "registry")
    if type(registry) is not dict:
        return False
    items = list(registry.items())
    return len(items) == 1 and items[0][0] is object and items[0][1] is base


def _dispatcher(func: Any, wrapped: Any) -> Any:
    """The ``dispatch`` function and the base of ``func``, if ``func`` is exactly the
    function ``functools.singledispatch`` returns or the callable
    ``functools.singledispatchmethod`` hands out; None for anything else, and for
    everything when the dispatcher's code was not identified (:func:`_prepare`; no function has
    None for a code object, and :data:`_METHOD_ACCESS` is then None too). ``wrapped`` is
    what ``func`` names as ``__wrapped__``."""
    if type(func) is types.FunctionType and func.__code__ is _WRAPPER_CODE:
        return _cell(func, "dispatch"), wrapped
    if _METHOD_ACCESS is None or type(func) is not _METHOD_ACCESS[0]:
        return None
    _, code, name, is_descriptor = _METHOD_ACCESS
    if code is None:
        dispatch = _entry(_instance_attributes(func), name)
    elif func.__code__ is code:
        dispatch = _cell(func, name)
    else:
        return None
    if is_descriptor:
        # the base is the function the descriptor was built on: what the callable handed
        # out names as ``__wrapped__`` is not always that object (Python 3.9 names the
        # function inside a ``classmethod`` there)
        attributes = _instance_attributes(dispatch)
        dispatcher, wrapped = _entry(attributes, "dispatcher"), _entry(attributes, "func")
        if type(dispatcher) is types.FunctionType and dispatcher.__code__ is _WRAPPER_CODE:
            dispatch = _cell(dispatcher, "dispatch")
    return dispatch, wrapped


def _chosen_at_call_time(function: Any) -> bool:
    """True for a held plain function that is exactly a single-dispatch function, or the
    function a single-dispatch method hands out (:func:`_dispatcher`), and can run
    anything but the base its own ``__dict__`` names as ``__wrapped__``."""
    found = _dispatcher(function, _own(function, "__wrapped__"))
    return found is not None and not _only_runs(*found)


def _refuse_call_time_dispatch(func: Any, wrapped: Any) -> None:
    """Raise :class:`DispatchedAtCallTime` if a call to ``func`` picks what runs by type.

    A ``functools.singledispatch`` function names its base implementation as
    ``__wrapped__``, which is all the walk follows; an implementation registered for a
    type runs in its place when the first argument is of that type, with defaults of its
    own that no policy saw. ``func`` is the callable the walk stands on and ``wrapped``
    what it names as ``__wrapped__``. Two callables are recognised, exactly, and nothing
    is run:

    - the function ``functools.singledispatch`` returns: a plain function whose code
      object is that of the ``wrapper`` defined inside it. Attributes are not what
      identifies it: ``functools.update_wrapper`` copies ``registry``, ``dispatch`` and
      ``register`` onto every wrapper of a dispatcher, and overwrites them on a dispatcher
      whose base carries its own;
    - the callable ``functools.singledispatchmethod`` hands out at each access, known by
      its type and code object (:func:`_method_access`): a function that keeps the
      descriptor (Python 3.9 to 3.12) or the ``dispatch`` function (3.13) in a closure
      cell, or an object that keeps the descriptor in an instance attribute (3.14, 3.15).

    For either, the registry read is the real one, the plain ``dict`` in the closure of
    the ``dispatch`` function, not the ``registry`` attribute. It is refused unless that
    registry is exactly ``{object: base}``, the key and the entry by identity:
    ``register(object, ...)`` replaces the base and keeps the one key. The base is
    ``wrapped``, or, where a single-dispatch method keeps its descriptor, the function that
    descriptor was built on. A registry that cannot be read this way is refused unread.

    Not found, by design: a dispatch written by hand in a function's body; a dispatcher
    another library builds; an implementation registered after the walk; a dispatcher
    reached through a wrapper or a helper that names nothing as ``__wrapped__``; a
    dispatcher whose closure was rewritten by hand; a registered single-dispatch method a
    wrapper holds, where what is handed out is an object (Python 3.14 and later); a
    dispatcher built by a ``functools`` that was reloaded after the first walk, or before
    it by the module as it was before a reload (its code objects are other objects); and
    everything, when ``functools.singledispatch`` was not the standard library's own
    function at the first walk or a Python builds it another way than 3.9 to 3.15 do
    (:data:`_WRAPPER_CODE` is then None and this check is off: it refuses nothing), or the
    method half when ``functools.singledispatchmethod`` was not (:data:`_METHOD_ACCESS`).
    """
    found = _dispatcher(func, wrapped)
    if found is not None and not _only_runs(*found):
        raise DispatchedAtCallTime(_DISPATCHED_AT_CALL_TIME)


def _next_callables(func: Any, walk: Any, bound: Optional[int] = 0) -> List[Any]:
    """The callables ``func`` may hand the call to, each bound as ``func`` binds it.
    ``bound`` counts the leading arguments ``func`` receives bound: one for each method it
    was reached through, and the positional arguments of each ``functools.partial``, a
    subclass included, read through the C-level accessor of the standard library's type.
    It is None once that cannot be read (behind an object that is a ``functools.partial``
    for ``isinstance`` and not an instance of that type, where another stands in
    ``functools`` since the first walk): what is held behind that point is not judged.

    A ``functools.wraps`` wrapper that takes ``(*args, **kwargs)`` passes the call to the
    function it wraps, which fills its own defaults; a wrapper can also add a parameter of
    its own. Such a wrapper can stand anywhere on the way: on the callable, behind a
    method, behind a ``functools.partial`` or as a callable object's ``__call__``. Each step
    is followed as Python takes it: a method's function bound to its object, a partial's
    function with the partial's own arguments, ``__wrapped__``, and an object's ``__call__``
    bound to the object. An object can have both (a class-based decorator built with
    ``functools.update_wrapper``): both are followed, because its ``__call__`` is what runs.
    A ``__call__`` that Python would call without the object (a ``staticmethod``, a
    ``partial``, any other non-function) raises ``TypeError``: what it binds cannot be read
    as a method's parameters.

    ``__wrapped__`` is the only link from a wrapper to what it calls that is followed. Two
    shapes that defeat it are refused, wherever one stands on the way. A single-dispatch
    function or method that can run anything but its base implementation raises
    :class:`DispatchedAtCallTime` here (:func:`_refuse_call_time_dispatch`). And ``walk``
    is handed every callable read and everything a callable that names a ``__wrapped__``
    holds, for the judgement :func:`reached_parameters` makes at the end
    (:func:`_refuse_hidden_wrapper`). Each of those two states exactly what it reads and
    what it does not find, and neither runs anything.
    """
    # a ``staticmethod`` (Python 3.10+) shows its function as ``__wrapped__``, taken below
    if inspect.ismethod(func):
        inners = _next_callables(func.__func__, walk, None if bound is None else bound + 1)
        return [types.MethodType(inner, func.__self__) for inner in inners]
    if isinstance(func, functools.partial):
        # the C-level accessor of the standard library's type, which a subclass cannot
        # change: no attribute of ``func`` is looked up for the count
        counted = any(base is _PARTIAL for base in _type_mro(type(func)))
        args = _partial_args(func) if counted else None
        bound = bound + len(args) if bound is not None and type(args) is tuple else None
        return [
            functools.partial(inner, *func.args, **func.keywords)
            for inner in _next_callables(func.func, walk, bound)
        ]
    found = []
    wrapped = getattr(func, "__wrapped__", None)
    _refuse_call_time_dispatch(func, wrapped)
    walk.read.append(func)
    if wrapped is not None:
        if walk.checked and bound is not None:
            walk.held.extend((value, bound) for value in _held(func, wrapped))
        found.append(wrapped)
    if not (inspect.isfunction(func) or inspect.isbuiltin(func) or isinstance(func, type)):
        call = inspect.getattr_static(type(func), "__call__", None)
        if inspect.isfunction(call):
            found.append(types.MethodType(call, func))
        elif call is not None and not isinstance(call, _C_CALL):
            raise TypeError("its callable object's `__call__` is not a plain method")
    return found


def reached_parameters(func: Any, skip: Tuple[str, ...] = ()) -> List[Any]:
    """The parameters of every callable a call to ``func`` reaches, ``func``'s own first.

    A gate that completes and judges a call by the parameters a framework declares must
    also find every named parameter some callable on the way would fill from a default of
    its own; this is that walk. It follows a method's function, a ``functools.partial``'s
    function, every ``__wrapped__`` and a callable object's ``__call__``. ``skip`` names
    the parameters the framework injects itself, as for :func:`unsupported_signature`.
    Raises :class:`DeclaredSignature`, :class:`HiddenWrapper` (a ``ValueError``: a callable
    on the way holds a second ``functools.wraps`` wrapper of a callable the walk reads,
    with a parameter of its own), :class:`DispatchedAtCallTime` (a ``ValueError``: a
    ``functools.singledispatch`` function or ``singledispatchmethod`` on the way can run
    something other than its base implementation), ``TypeError`` or ``ValueError`` for a
    callable it cannot read, which a gate turns into a refusal.

    The two refusals are exact and narrow: each finds one shape and states the rest as
    limits (:func:`_refuse_hidden_wrapper`, :func:`_refuse_call_time_dispatch`). They read
    what a callable holds without running any of it. A wrapper a callable holds is judged
    by the rule and the code that judge the levels returned here
    (:func:`_argument_of_its_own`), so it gets the verdict it would get read directly. The
    callables on the way themselves are read as before, by attribute lookup.
    """
    return _reached(func, skip, None)


def _reached(func: Any, skip: Any, declared: Any) -> List[Any]:
    """:func:`reached_parameters`, for a caller that has read what ``func`` declares."""
    walk = _walk()
    found = [own_parameters(func)]
    pending = _next_callables(func, walk)
    for _ in range(_DEPTH):
        if not pending:
            _refuse_hidden_wrapper(func, found, walk, skip, declared)
            return found
        current = pending.pop()
        found.append(own_parameters(current))
        pending.extend(_next_callables(current, walk))
    raise ValueError("too many wrappers to inspect")


def detached(value: Any, ancestors: Tuple[int, ...] = ()) -> Any:
    """Detach the supported data domain without invoking application copying/serialization.

    Returns a copy the policy can write to without reaching the original, or raises
    ``TypeError`` for a value that is not data. A caller turns that into a refusal.

    Exact container/scalar types avoid subclasses with custom iteration or copy behavior.
    Every Enum member is refused, whatever its value: a member is one shared object, and
    attributes its class sets on it (an ``__init__`` that stores a dict, say) would be the
    same object at policy and at execution. Passing only ``.value`` would hide that state
    from the policy while the tool still used it, so that is no repair either.
    """
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("non-finite numbers are not JSON data")
        return value
    if isinstance(value, Enum):
        raise TypeError(
            "Enum arguments are unsupported; declare data-only parameters "
            "and resolve application types inside the tool body"
        )
    if type(value) not in (dict, list):
        raise TypeError("arguments must be data-only")
    if id(value) in ancestors:
        raise TypeError("cyclic arguments are not JSON data")
    ancestors = (*ancestors, id(value))
    if type(value) is list:
        return [detached(item, ancestors) for item in value]
    if any(type(key) is not str for key in value):
        raise TypeError("argument keys must be strings")
    return {key: detached(item, ancestors) for key, item in value.items()}
