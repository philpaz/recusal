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
  what they do not find;
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
import gc
import inspect
import math
import types
import typing
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

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
        reached = reached_parameters(func)
    except DeclaredSignature:
        return "it declares a signature that is not its own"
    except HiddenWrapper:
        return _HIDDEN_WRAPPER
    except DispatchedAtCallTime:
        return _DISPATCHED_AT_CALL_TIME
    except Exception:  # noqa: BLE001, an uninspectable callable cannot be judged
        return "its signature cannot be inspected"
    for level in reached:
        for name, parameter in level.items():
            if name in skip or name in parameters:
                continue
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            return (
                f"`{name}` is a parameter of a function the call reaches that the tool does "
                "not declare, so it would run with a value no policy judged"
            )
    # a forwarding wrapper (``*args``/``**kwargs``) that also names a parameter takes that
    # argument itself; a function further in that names it too then runs with its own
    # default, not the value the policy judged
    named = [
        {n for n, p in level.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        for level in reached
    ]
    for index, level in enumerate(reached):
        forwards = any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in level.values())
        if not forwards:
            continue
        for name in sorted(named[index] - set(skip)):
            if any(name in other for i, other in enumerate(named) if i != index):
                return (
                    f"`{name}` is taken by a wrapper and by the function it wraps, which can "
                    "then run with a value no policy judged"
                )
    return None


class DeclaredSignature(Exception):
    """A callable on the way to the code that runs carries a ``__signature__`` of its own."""


class HiddenWrapper(ValueError):
    """A callable on the way holds a second wrapper of a function the walk reads: one that
    names what it wraps as ``__wrapped__``, that ``__wrapped__`` on the way does not lead
    to, and that takes an argument of its own, so the walk cannot read every parameter the
    call reaches.

    Raised for exactly what :func:`_refuse_hidden_wrapper` describes, which also lists what
    is not found: among others a second wrapper that names nothing as ``__wrapped__``, and
    one that is or is held by an object, a proxy or a wrapper a C extension builds."""


_HIDDEN_WRAPPER = (
    "a wrapper holds another wrapper of the function it wraps, which the walk does not follow"
)


class DispatchedAtCallTime(ValueError):
    """A callable on the way is a ``functools.singledispatch`` function, or the function a
    ``functools.singledispatchmethod`` hands out, that can run something other than the
    base function it names as ``__wrapped__``: an implementation is registered beside the
    base, or in its place. Which runs is chosen at call time by the type of an argument, so
    the walk cannot read the parameters the call reaches.
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
    callable this reads: calling it runs ``__new__`` and ``__init__``.
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
    read in its place, declares ``__signature__``."""
    if depth > 8:
        raise ValueError("too many wrappers to inspect")
    if "__signature__" in getattr(func, "__dict__", {}) or hasattr(type(func), "__signature__"):
        raise DeclaredSignature
    if isinstance(func, functools.partial):
        _refuse_declared_signature(func.func, depth + 1)
    elif not (inspect.isfunction(func) or inspect.isbuiltin(func) or isinstance(func, type)):
        call = getattr(type(func), "__call__", None)
        if call is not None and not isinstance(call, _C_CALL):
            _refuse_declared_signature(call, depth + 1)


# The two checks below read what a callable on the walk HOLDS. A held value is not a
# callable the walk was asked to read: it can be a lazy proxy that connects when touched, a
# remote stub that answers every attribute, a settings object. So the checks are exact and
# run nothing. A held value is looked at only when its type, read with ``type()``, is one
# of a closed set CPython itself builds, and then only through that type's own C-level
# accessors: no attribute lookup, no descriptor a class declares, no comparison but
# identity, no hashing. A value of any other type is skipped without being read at all.

_MISSING: Any = object()
_DEPTH = 32
_function_dict = types.FunctionType.__dict__["__dict__"].__get__
_CACHE_WRAPPER = _PASS_THROUGH[0]
_cache_dict = _CACHE_WRAPPER.__dict__["__dict__"].__get__
_method_function = types.MethodType.__dict__["__func__"].__get__
_partial_function = functools.partial.__dict__["func"].__get__
_partial_args = functools.partial.__dict__["args"].__get__
_type_mro = type.__dict__["__mro__"].__get__
_type_dict = type.__dict__["__dict__"].__get__


def _entry(attributes: Any, name: str) -> Any:
    """The value under the string ``name`` in a plain ``dict``, or ``_MISSING``, found by
    listing the items: no key is hashed or compared unless it is a ``str``."""
    if type(attributes) is not dict:
        return _MISSING
    for key, value in attributes.items():
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


def _instance_attributes(value: Any) -> Any:
    """The instance dictionary of ``value``, or an empty one: read through the C-level
    ``__dict__`` accessor of its class, found in the dictionaries of that class. A class
    that answers ``__dict__`` any other way (a property) has none this reads."""
    for klass in _type_mro(type(value)):
        accessor = _type_dict(klass).get("__dict__", _MISSING)
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
        for value in _instance_attributes(func).values()
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
    if kind is functools.partial:
        return _partial_function(value)
    if kind is staticmethod or kind is classmethod:
        return kind.__dict__["__func__"].__get__(value)
    return _MISSING


def _takes_an_argument(value: Any, bound: int) -> bool:
    """True if a link of the closed set can keep or fill an argument of a call passed on
    to it: a plain function with a named parameter after the ``bound`` leading ones it
    receives (read from the code object Python binds a call with, never from a signature
    it shows), and a single-dispatch function that can run anything but its base. A
    function whose only parameters are ``*args`` and ``**kwargs`` cannot, and no other type
    of the closed set counts itself: what it passes the call on to is the next link."""
    if type(value) is not types.FunctionType:
        return False
    code = value.__code__
    named = max(code.co_argcount - bound, 0) + code.co_kwonlyargcount
    return named > 0 or _chosen_at_call_time(value)


def _hides(value: Any, func: Any, followed: List[Any]) -> bool:
    """True if ``value``, held by the walked ``func``, is a wrapper the walk does not read
    of a callable it does read (one of ``followed``): its chain of the closed set reaches
    one by identity, and a link on the way can take an argument of its own or the chain is
    still going at the walk's bound."""
    if value is func:
        return False
    links: List[Any] = []
    takes = False
    bound = 0
    for _ in range(_DEPTH):
        if any(value is seen for seen in followed):
            return takes
        if any(value is link for link in links):
            return False  # a cycle: it reaches nothing the walk reads
        inner = _passes_on_to(value)
        if inner is _MISSING:
            return False
        takes = takes or _takes_an_argument(value, bound)
        links.append(value)
        if type(value) is functools.partial:
            args = _partial_args(value)
            bound = len(args) if type(args) is tuple else 0
        else:
            bound = 1 if type(value) is types.MethodType or type(value) is classmethod else 0
        value = inner
    return True  # still going at the bound: what it reaches is not known


def _refuse_hidden_wrapper(func: Any, wrapped: Any) -> None:
    """Raise :class:`HiddenWrapper` if ``func`` holds a wrapper the walk does not read.

    A decorator can apply another wrapper inside itself and call the result, while
    ``functools.wraps`` sets ``__wrapped__`` to the undecorated function: the walk then
    never reads the wrapper in between, which can keep an argument for itself or fill a
    default of its own. This finds exactly one shape of that, and runs nothing to do it.

    ``func`` is a callable on the walk that names ``wrapped`` as ``__wrapped__``. It holds
    what is in its closure cells (a plain function) or its instance attributes (any other
    callable). The callables the walk reads from here are ``wrapped`` and what it passes a
    call on to in turn, as far as the closed set of :func:`_passes_on_to` goes. A held
    value is a hidden wrapper when all of these hold:

    - it is of that closed set, and is not ``func``;
    - the chain it passes a call on to, read the same way, reaches by identity a callable
      the walk reads;
    - a link of that chain before the one reached can take an argument of its own
      (:func:`_takes_an_argument`), or the chain is still going after 32 links.

    Not found, by design: a second wrapper that does not itself name what it wraps (a plain
    inner function, a lambda, a helper, a bound method of an object that holds the
    function); one that is, or is held by, an object of any other type (a callable object,
    a proxy, a wrapper a C extension builds, a subclass of a type of the closed set); one
    held anywhere but the closure cells or instance attributes of a callable on the walk (a
    class attribute, a slot, a module global, a container, a function's own attribute); a
    class-based decorator whose own method keeps the argument; a wrapper that takes an
    argument in its body alone; and the arguments a ``functools.partial`` binds.
    """
    followed: List[Any] = []
    link = wrapped
    for _ in range(_DEPTH):
        if link is _MISSING or any(link is seen for seen in followed):
            break
        followed.append(link)
        link = _passes_on_to(link)
    if any(_hides(value, func, followed) for value in _held(func, wrapped)):
        raise HiddenWrapper(_HIDDEN_WRAPPER)


def _chosen_at_call_time(function: Any) -> bool:
    """True for exactly what ``functools.singledispatch`` builds, when it can run anything
    but the base function it names as ``__wrapped__``.

    What it builds is a plain function whose own ``__dict__`` holds ``registry`` (a
    ``mappingproxy``), a callable ``dispatch`` and ``register``. The base alone runs when
    the registry is ``{object: base}``, the key and the entry by identity:
    ``register(object, ...)`` replaces the base and keeps the one key. The registry is read
    only when the mapping behind the ``mappingproxy`` is a plain ``dict`` (found without
    touching it); behind anything else it counts as chosen at call time, unread. A wrapper
    that only carries the copy of these attributes ``functools.update_wrapper`` makes
    (the same ``registry`` object as the function it names as ``__wrapped__``, or as the
    function of the bound method it names) is not the dispatcher: the walk reads that
    function next. A function whose attributes were rewritten by hand to pass this is not
    found."""
    registry = _own(function, "registry")
    if (
        type(function) is not types.FunctionType
        or type(registry) is not types.MappingProxyType
        or not callable(_own(function, "dispatch"))
        or _own(function, "register") is _MISSING
    ):
        return False
    wrapped = _own(function, "__wrapped__")
    source = _method_function(wrapped) if type(wrapped) is types.MethodType else wrapped
    if _own(source, "registry") is registry:
        return False
    behind = gc.get_referents(registry)
    if len(behind) != 1 or type(behind[0]) is not dict:
        return True
    items = list(behind[0].items())
    return not (len(items) == 1 and items[0][0] is object and items[0][1] is wrapped)


def _refuse_call_time_dispatch(func: Any) -> None:
    """Raise :class:`DispatchedAtCallTime` if a call to ``func`` picks what runs by type.

    A ``functools.singledispatch`` function names its base implementation as
    ``__wrapped__``, which is all the walk follows; an implementation registered for a
    type runs in its place when the first argument is of that type, with defaults of its
    own that no policy saw. Two things are checked, and nothing is run:

    - ``func`` itself, the callable the walk stands on (:func:`_chosen_at_call_time`);
    - a ``functools.singledispatchmethod``, or a subclass of it, in the closure cells of
      ``func`` when ``func`` is a plain function: that is where the function
      ``singledispatchmethod.__get__`` hands out at each access keeps the descriptor. Its
      dispatcher, read from its instance dictionary, is checked the same way.

    Not found, by design: a dispatch written by hand in a function's body; a dispatcher
    another library builds; an implementation registered after the walk; a dispatcher
    reached through a wrapper that names nothing as ``__wrapped__``; a
    ``singledispatchmethod`` kept anywhere but that closure (a later Python may hand it out
    another way; checked on Python 3.9 to 3.12); and a dispatcher whose attributes were
    rewritten by hand.
    """
    dispatchers = [func]
    if type(func) is types.FunctionType:
        for value in _cells(func):
            if issubclass(type(value), functools.singledispatchmethod):
                dispatchers.append(_entry(_instance_attributes(value), "dispatcher"))
    if any(_chosen_at_call_time(dispatcher) for dispatcher in dispatchers):
        raise DispatchedAtCallTime(_DISPATCHED_AT_CALL_TIME)


def _next_callables(func: Any) -> List[Any]:
    """The callables ``func`` may hand the call to, each bound as ``func`` binds it.

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
    shapes that defeat it are refused here, on every callable this is given, so wherever
    one stands on the way. A callable that names a ``__wrapped__`` and holds a second
    ``functools.wraps`` wrapper of a function the walk reads, one with a parameter of its
    own, raises :class:`HiddenWrapper` (:func:`_refuse_hidden_wrapper`). A single-dispatch
    function or method that can run anything but its base implementation raises
    :class:`DispatchedAtCallTime` (:func:`_refuse_call_time_dispatch`). Each of those two
    states exactly what it reads and what it does not find, and neither runs anything.
    """
    # a ``staticmethod`` (Python 3.10+) shows its function as ``__wrapped__``, taken below
    if inspect.ismethod(func):
        return [types.MethodType(inner, func.__self__) for inner in _next_callables(func.__func__)]
    if isinstance(func, functools.partial):
        return [
            functools.partial(inner, *func.args, **func.keywords)
            for inner in _next_callables(func.func)
        ]
    _refuse_call_time_dispatch(func)
    found = []
    wrapped = getattr(func, "__wrapped__", None)
    if wrapped is not None:
        _refuse_hidden_wrapper(func, wrapped)
        found.append(wrapped)
    if not (inspect.isfunction(func) or inspect.isbuiltin(func) or isinstance(func, type)):
        call = inspect.getattr_static(type(func), "__call__", None)
        if inspect.isfunction(call):
            found.append(types.MethodType(call, func))
        elif call is not None and not isinstance(call, _C_CALL):
            raise TypeError("its callable object's `__call__` is not a plain method")
    return found


def reached_parameters(func: Any) -> List[Any]:
    """The parameters of every callable a call to ``func`` reaches, ``func``'s own first.

    A gate that completes and judges a call by the parameters a framework declares must
    also find every named parameter some callable on the way would fill from a default of
    its own; this is that walk. It follows a method's function, a ``functools.partial``'s
    function, every ``__wrapped__`` and a callable object's ``__call__``. Raises
    :class:`DeclaredSignature`, :class:`HiddenWrapper` (a ``ValueError``: a callable on
    the way holds a second ``functools.wraps`` wrapper of a function the walk reads, with
    a parameter of its own), :class:`DispatchedAtCallTime` (a ``ValueError``: a
    ``functools.singledispatch`` function or ``singledispatchmethod`` on the way can run
    something other than its base implementation), ``TypeError`` or ``ValueError`` for a
    callable it cannot read, which a gate turns into a refusal.

    The two refusals are exact and narrow: each finds one shape and states the rest as
    limits (:func:`_refuse_hidden_wrapper`, :func:`_refuse_call_time_dispatch`). They read
    what a callable holds without running any of it. The callables on the way themselves
    are read as before, by attribute lookup.
    """
    found = [own_parameters(func)]
    pending = _next_callables(func)
    for _ in range(32):
        if not pending:
            return found
        current = pending.pop()
        found.append(own_parameters(current))
        pending.extend(_next_callables(current))
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
