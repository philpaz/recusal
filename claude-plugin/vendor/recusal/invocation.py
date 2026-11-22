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
  (:func:`reached_parameters`), and refusing a wrapper that holds another wrapper of the
  same function which the walk does not read (:class:`HiddenWrapper`) and a
  single-dispatch function that can run anything but its base
  (:class:`DispatchedAtCallTime`);
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
from typing import Any, Callable, List, Optional, Set, Tuple

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
    """A callable on the way holds another wrapper of a function the walk reads, one that
    ``__wrapped__`` does not lead to and that can take an argument of its own, so the walk
    cannot read every parameter the call reaches.

    Raised for what :func:`_refuse_hidden_wrapper` finds, which states what is looked at
    (a function's closure cells; a callable object's instance attributes and its own
    class) and what is not found. A held wrapper that only passes ``(*args, **kwargs)`` on
    is not one."""


_HIDDEN_WRAPPER = (
    "a wrapper holds another wrapper of the function it wraps, which the walk does not follow"
)


class DispatchedAtCallTime(ValueError):
    """A callable on the way is a ``functools.singledispatch`` function, or holds a
    ``functools.singledispatchmethod``, that can run something other than the base function
    it names as ``__wrapped__``: an implementation is registered beside the base, or in its
    place. Which runs is chosen at call time by the type of an argument, so the walk cannot
    read the parameters the call reaches. :func:`_refuse_call_time_dispatch` states what is
    not found."""


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


def _wrapped_chain(func: Any) -> List[Any]:
    """What the walked ``func`` names as ``__wrapped__``, what that names in turn, and so
    on, ``func`` itself left out: the callables the walk goes on to read, each link read
    as the walk reads it, by a plain attribute lookup. Bounded like the walk, so a cycle
    ends. An ``Exception`` raised by a lookup ends the chain; nothing else is caught."""
    chain: List[Any] = []
    for _ in range(_DEPTH):
        try:
            func = getattr(func, "__wrapped__", None)
        except Exception:  # noqa: BLE001, an attribute that cannot be read names nothing
            break
        if func is None:
            break
        chain.append(func)
    return chain


# What a callable HOLDS is not a callable the walk was asked to read. It can be anything a
# program keeps beside a function: a lazy proxy that connects when touched, a remote stub
# that answers every attribute, a settings object. So nothing below runs code of a held
# value. Its class is ``type(value)``, never ``isinstance`` (which asks for ``__class__``).
# Its attributes are read from the dictionaries of that class and from the instance
# dictionary the class's own C-level ``__dict__`` accessor returns, never by an attribute
# lookup, and values are compared by identity only. Two things do run: a descriptor a held
# value's class declares for ``__wrapped__`` itself (a property), and the ``items`` of the
# mapping behind a single-dispatch ``registry``.

_MISSING: Any = object()
_DEPTH = 32
_mro_of = type.__dict__["__mro__"].__get__
_dict_of = type.__dict__["__dict__"].__get__
_function_dict = types.FunctionType.__dict__["__dict__"].__get__
_method_function = types.MethodType.__dict__["__func__"].__get__
_method_receiver = types.MethodType.__dict__["__self__"].__get__
_PARTIAL = functools.partial.__dict__


def _is_a(value: Any, kind: Any) -> bool:
    return issubclass(type(value), kind)


def _class_attribute(value: Any, name: str) -> Any:
    """The entry ``name`` in the dictionaries of ``value``'s class, as stored, or
    ``_MISSING``."""
    for klass in _mro_of(type(value)):
        entry = _dict_of(klass).get(name, _MISSING)
        if entry is not _MISSING:
            return entry
    return _MISSING


def _instance_attributes(value: Any) -> Any:
    """``value``'s instance dictionary, or an empty one: read through the C-level
    ``__dict__`` accessor of its class. A class that answers ``__dict__`` any other way (a
    property) has none this reads."""
    accessor = _class_attribute(value, "__dict__")
    if type(accessor) is not types.GetSetDescriptorType:
        return {}
    try:
        found = accessor.__get__(value, type(value))
    except Exception:  # noqa: BLE001, no instance dictionary
        return {}
    return found if type(found) is dict else {}


def _static_attribute(value: Any, name: str) -> Any:
    """``value``'s instance attribute ``name``, else its class's entry as stored, else
    ``_MISSING``."""
    own = _instance_attributes(value)
    return own[name] if name in own else _class_attribute(value, name)


def _static_wrapped(value: Any) -> Any:
    """What ``value`` names as ``__wrapped__``, or None, found where Python would find it
    but without an attribute lookup on ``value``: its ``__getattr__`` and
    ``__getattribute__`` never run. A descriptor the class declares for the name is called
    directly: for a slot or a built-in accessor that runs no code of the instance, and for
    any other (a property) it runs that descriptor's own code and nothing else. An
    ``Exception`` it raises reads as None; nothing else is caught."""
    if type(value) is types.FunctionType:
        return _function_dict(value).get("__wrapped__")
    entry = _class_attribute(value, "__wrapped__")
    getter = _MISSING if entry is _MISSING else _class_attribute(entry, "__get__")
    overrides = entry is not _MISSING and (
        _class_attribute(entry, "__set__") is not _MISSING
        or _class_attribute(entry, "__delete__") is not _MISSING
    )
    if not overrides:
        own = _instance_attributes(value)
        if "__wrapped__" in own:
            return own["__wrapped__"]
    if getter is _MISSING:
        return None if entry is _MISSING else entry
    try:
        return getter(entry, value, type(value))
    except Exception:  # noqa: BLE001, a descriptor that cannot be read names nothing
        return None


def _binds(value: Any) -> bool:
    """True for a ``functools.partial`` that carries arguments of its own."""
    args, keywords = _PARTIAL["args"].__get__(value), _PARTIAL["keywords"].__get__(value)
    return not (type(args) is tuple and type(keywords) is dict) or bool(args or keywords)


def _viewed(value: Any) -> Any:
    """The callable ``value`` only presents, or ``_MISSING``: the function of a bound
    method, of a ``staticmethod`` or a ``classmethod``, a ``property``'s getter, and the
    function of a ``functools.partial`` that binds no argument."""
    if type(value) is types.MethodType:
        return _method_function(value)
    for kind, name in ((staticmethod, "__func__"), (classmethod, "__func__"), (property, "fget")):
        if _is_a(value, kind):
            return kind.__dict__[name].__get__(value)
    if _is_a(value, functools.partial) and not _binds(value):
        return _PARTIAL["func"].__get__(value)
    return _MISSING


def _presents(value: Any, followed: List[Any]) -> bool:
    """True if ``value`` is, by identity, a callable the walk reads (one of ``followed``)
    or only presents one (:func:`_viewed`)."""
    for _ in range(_DEPTH):
        if any(value is seen for seen in followed):
            return True
        if type(value) is types.FunctionType:
            return False
        value = _viewed(value)
        if value is _MISSING:
            return False
    return False


def _cells(function: Any) -> List[Any]:
    held = []
    for cell in function.__closure__ or ():
        try:
            held.append(cell.cell_contents)
        except ValueError:  # an empty cell holds nothing
            continue
    return held


def _holds(value: Any) -> List[Any]:
    """What a held callable holds itself, one level: the closure cells of a function; for
    a bound method, those of its function and the instance attributes of its object; the
    arguments of a ``functools.partial`` (its function is what it wraps, :func:`_wraps`);
    the instance attributes of any other object."""
    if type(value) is types.FunctionType:
        return _cells(value)
    if type(value) is types.MethodType:
        function = _method_function(value)
        inner = _cells(function) if type(function) is types.FunctionType else []
        return inner + list(_instance_attributes(_method_receiver(value)).values())
    held = list(_instance_attributes(value).values())
    if _is_a(value, functools.partial):
        args, keywords = _PARTIAL["args"].__get__(value), _PARTIAL["keywords"].__get__(value)
        held.extend(args if type(args) is tuple else ())
        held.extend(keywords.values() if type(keywords) is dict else ())
    return held


def _named_parameters(function: Any, bound: int = 0) -> Tuple[str, ...]:
    """The named parameters of a plain function after the ``bound`` leading ones, read
    from the code object Python binds a call with, never from a signature it shows."""
    code = function.__code__
    positional = code.co_varnames[min(bound, code.co_argcount) : code.co_argcount]
    return positional + code.co_varnames[code.co_argcount :][: code.co_kwonlyargcount]


def _can_take_an_argument(
    value: Any, bound: int = 0, taken: Optional[Set[str]] = None, _depth: int = 0
) -> bool:
    """True unless ``value`` is known to be unable to keep or fill an argument of a call
    passed on to it, by its own parameters.

    A callable whose only parameters are ``*args`` and ``**kwargs`` (after the ``bound``
    leading ones a method or a ``classmethod`` receives) can do neither: that is all a
    signature walk reads of any wrapper. With ``taken`` None any named parameter counts.
    With ``taken`` the names the functions the walk reads take, a named parameter counts
    when it has one of those names (the callable keeps the argument) or when the callable
    declares any default (it fills one); a helper such as ``on_error(error)`` then does
    not. ``functools.lru_cache`` passes every argument through. A callable that cannot be
    read this way (a class, a built-in, a ``__call__`` that is not a plain function, a
    ``functools.partial`` with arguments) counts as able to."""
    kind = type(value)
    if kind in _PASS_THROUGH:
        return False
    if _depth > 8 or issubclass(kind, type):
        return True
    if kind is not types.MethodType and _viewed(value) is not _MISSING:
        return False  # a view takes nothing itself: what it presents is read in its turn
    if kind is types.MethodType:
        return _can_take_an_argument(_method_function(value), bound + 1, taken, _depth + 1)
    if kind is not types.FunctionType:
        call = _class_attribute(value, "__call__")
        if type(call) is not types.FunctionType:
            return True
        value, bound = call, bound + 1
    named = _named_parameters(value, bound)
    if taken is None:
        return bool(named)
    return bool(value.__defaults__ or value.__kwdefaults__) or any(n in taken for n in named)


def _wraps(value: Any) -> Any:
    """What a held callable passes a call on to by its own word, or None: what the function
    of a bound method names as ``__wrapped__``, the function of a ``functools.partial``,
    what a view presents (:func:`_viewed`), else what ``value`` names as ``__wrapped__``."""
    if type(value) is types.FunctionType:
        return _function_dict(value).get("__wrapped__")
    if type(value) is types.MethodType:
        return _static_wrapped(_method_function(value))
    if _is_a(value, functools.partial):
        return _PARTIAL["func"].__get__(value)
    inner = _viewed(value)
    return _static_wrapped(value) if inner is _MISSING else inner


def _hides(value: Any, func: Any, followed: List[Any]) -> bool:
    """True if ``value``, held by the walked ``func``, is or may be a wrapper the walk does
    not read of a callable it does read (one of ``followed``)."""
    bound = 0
    for _ in range(_DEPTH):
        if value is func or _presents(value, followed):
            return False
        inner = _MISSING if type(value) is types.MethodType else _viewed(value)
        if inner is _MISSING:
            break
        bound = 0 if _is_a(value, (staticmethod, functools.partial)) else 1
        value = inner
    else:
        return True
    if not callable(value):
        return False
    # its own ``__wrapped__`` chain, to the first callable the walk reads
    links = [value]
    for _ in range(_DEPTH):
        inner = _wraps(links[-1])
        if inner is None or any(inner is link for link in links):
            break
        if _presents(inner, followed):
            # a wrapper by its own word: a parameter of its own on any link counts
            return any(
                _can_take_an_argument(link, bound if link is value else 0)
                or _chosen_at_call_time(link)
                for link in links
            )
        links.append(inner)
    else:
        return True  # still going at the bound: what it reaches is not known
    # it names no callable the walk reads: does it hold one itself?
    if not any(_presents(held, followed) for held in _holds(value)):
        return False
    functions = [seen for seen in followed if type(seen) is types.FunctionType]
    taken = {name for seen in functions for name in _named_parameters(seen)} if functions else None
    return _can_take_an_argument(value, bound, taken)


def _held(func: Any) -> List[Any]:
    """What the walked ``func`` holds: the closure cells of a function; for a callable
    object its instance attributes, and the entries of its own class other than
    ``__call__`` (which the walk reads), a slot read for its value. A function's own
    attributes are not read, and neither is an instance attribute that is the same object
    under the same name on the callable ``func`` names as ``__wrapped__``:
    ``functools.update_wrapper`` copies those over."""
    if type(func) is types.FunctionType:
        return _cells(func)
    wrapped = _static_wrapped(func)
    copied = {} if wrapped is None else _instance_attributes(wrapped)
    held = [
        value
        for name, value in _instance_attributes(func).items()
        if copied.get(name, _MISSING) is not value
    ]
    for name, entry in _dict_of(type(func)).items():
        if name == "__call__":
            continue
        if type(entry) is types.MemberDescriptorType:
            try:
                entry = entry.__get__(func, type(func))
            except Exception:  # noqa: BLE001, an empty slot holds nothing
                continue
        held.append(entry)
    return held


def _refuse_hidden_wrapper(func: Any) -> None:
    """Raise :class:`HiddenWrapper` if ``func`` holds a wrapper the walk does not read.

    A decorator can apply another wrapper inside itself and call the result, while
    ``functools.wraps`` sets ``__wrapped__`` to the undecorated function: the walk then
    never reads the wrapper in between, which can keep an argument for itself or fill a
    default of its own. What ``func`` holds is read where such a wrapper lives
    (:func:`_held`). A held callable is a hidden wrapper when all of these hold:

    - it is not ``func``, and neither is nor only presents a callable the walk reads
      through ``__wrapped__`` (:func:`_presents`);
    - it reaches one: its own ``__wrapped__`` chain leads to one, or, naming nothing that
      does, it holds one itself, one level in (:func:`_holds`);
    - it can take an argument of its own (:func:`_can_take_an_argument`), or a link of its
      chain can, or a link of its chain chooses what runs at call time
      (:func:`_chosen_at_call_time`). A chain still going at the walk's bound counts.

    So a logger, a clock, a callback, configuration, and a held wrapper that only passes
    ``(*args, **kwargs)`` on are not hidden wrappers. Not found: a wrapper held anywhere
    else (a module global, a container, a function's own attribute, a class other than the
    callable object's own, a slot of a base class), one reached only through two callables
    that each name nothing as ``__wrapped__``, and one held by a callable that itself names
    nothing as ``__wrapped__``. Nothing held is asked for an attribute: see the note above
    ``_MISSING``.
    """
    followed = []
    for link in _wrapped_chain(func):
        for _ in range(_DEPTH):
            followed.append(link)
            link = _viewed(link)
            if link is _MISSING:
                break
    if any(_hides(value, func, followed) for value in _held(func)):
        raise HiddenWrapper(_HIDDEN_WRAPPER)


def _chosen_at_call_time(func: Any) -> bool:
    """True for a single-dispatch function, as ``functools.singledispatch`` builds one,
    that can run anything but the base function it names as ``__wrapped__``.

    Such a function carries a ``registry`` (a ``mappingproxy`` with an entry for
    ``object``) and a callable ``dispatch``. It is read by structure: the attributes are
    found without an attribute lookup, ``dispatch`` is not called, and only the registry's
    items are listed. The base alone is what runs when ``object`` is the only key and its
    entry is, by identity, the function named as ``__wrapped__``: ``register(object, ...)``
    replaces the base and keeps the one key. A registry that cannot be listed counts as
    chosen at call time. A wrapper that only carries a copy of these attributes
    (``functools.update_wrapper`` copies them from the function it wraps) is not the
    dispatcher: the walk reads that function next.
    """
    registry = _static_attribute(func, "registry")
    if type(registry) is not types.MappingProxyType:
        return False
    if not callable(_static_attribute(func, "dispatch")):
        return False
    wrapped = _static_wrapped(func)
    if wrapped is not None and _static_attribute(wrapped, "registry") is registry:
        return False
    try:
        items = list(registry.items())
    except Exception:  # noqa: BLE001, a registry that cannot be read is not the base alone
        return True
    if not any(key is object for key, _ in items):
        return False
    return not (len(items) == 1 and items[0][1] is wrapped)


def _refuse_call_time_dispatch(func: Any) -> None:
    """Raise :class:`DispatchedAtCallTime` if a call to ``func`` picks what runs by type.

    A ``functools.singledispatch`` function names its base implementation as
    ``__wrapped__``, which is all the walk follows; an implementation registered for a
    type runs in its place when the first argument is of that type, with defaults of its
    own that no policy saw. With only the base registered the base is what runs, and the
    function is walked like any wrapper (:func:`_chosen_at_call_time`). A
    ``functools.singledispatchmethod`` hands out a new callable at each access with no
    registry of its own; it holds the descriptor (:func:`_held`), and the descriptor's
    dispatcher is read the same way. The descriptor is recognised by its type, a subclass
    included, and its dispatcher is read from its instance dictionary: nothing ``func``
    holds is asked for an attribute. Not found: an implementation registered after the
    walk, a dispatch written by hand in a function's body, a dispatcher built by another
    library, and a single-dispatch function a wrapper calls without naming it as
    ``__wrapped__`` (unless :func:`_refuse_hidden_wrapper` finds it).
    """
    dispatchers = [func]
    for value in _held(func):
        if _is_a(value, functools.singledispatchmethod):
            dispatchers.append(_instance_attributes(value).get("dispatcher"))
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

    ``__wrapped__`` is the only link from a wrapper to what it calls that is followed. A
    callable that names a ``__wrapped__`` and also holds another wrapper of a function the
    walk reads, one that can take an argument of its own, raises :class:`HiddenWrapper`
    (:func:`_refuse_hidden_wrapper`, which states where it looks and what it does not
    find): what the call reaches cannot be read from it. A single-dispatch function or
    method that can run anything but its base implementation raises
    :class:`DispatchedAtCallTime` (:func:`_refuse_call_time_dispatch`): ``__wrapped__``
    names the base, and which implementation runs is not known until the call. Both are
    checked on every callable this is given, so wherever one stands on the way.
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
        _refuse_hidden_wrapper(func)
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
    the way holds another wrapper of a function the walk reads, which ``__wrapped__`` does
    not lead to and which can take an argument of its own), :class:`DispatchedAtCallTime`
    (a ``ValueError``: a ``functools.singledispatch`` function or ``singledispatchmethod``
    on the way can run something other than its base implementation, chosen at call time),
    ``TypeError`` or ``ValueError`` for a callable it cannot read, which a gate turns into
    a refusal. Neither check finds everything: each states what it does not find. What a
    callable holds is read without running it; the callables on the way themselves are
    read as before, by attribute lookup.
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
