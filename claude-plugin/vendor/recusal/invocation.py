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
  (:func:`reached_parameters`), and refusing five shapes the walk cannot read to the
  end: a wrapper that holds a second ``functools.wraps`` wrapper of the same function
  (:class:`HiddenWrapper`); a ``functools.lru_cache`` wrapper that names one function
  and calls another, where which one cannot be told (:class:`UnreadCache`); a
  single-dispatch function that can run anything but its base
  (:class:`DispatchedAtCallTime`); a bound argument a wrapper takes under another name
  and may keep (:class:`BoundArgumentKept`); and a callable ``inspect.signature`` reads
  as a ``functools.partialmethod``, which shows parameters that are not its own
  (:class:`PartialMethodSignature`). Each check is exact and states what it does not
  find, and the two that read what a callable holds run nothing;
- how to hand a policy a copy it can write to without reaching what executes
  (:func:`detached`).

The framework gates under ``examples/`` each carried their own copy of these. They import
them from here instead, so the supported set is defined once and a fix reaches every gate
that uses it. Nothing here imports a framework, reads a file or calls a policy. The
answer of every function but the walk depends on its argument alone. The walk's depends
also on what the first walk of the process found the standard library to be
(:func:`_prepare`), and it keeps for the next walk two things that cannot change while
an object lives (:func:`_keep`). Standard library only.

What stays in each gate is the part that is the framework's: where its hook sits, how it
prepares an invocation, and which of its own parameters it injects.
"""

import functools
import inspect
import math
import types
import typing
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# ``gc`` and ``weakref`` are two modules 0.15.0 did not import, and one check reads with
# each (:func:`_cache_calls`, :func:`_kept`). A process can refuse either import (None put
# in ``sys.modules``, an import hook or an audit hook that raises): the module is then
# imported as 0.15.0 is, with None in its place and that one reading off. So whatever the
# import raises is caught, but for an interrupt from the keyboard, which is no answer about
# the module. The import is made here, where this module is itself being imported, and
# never at a walk: a walk can be the first thing a worker thread does, and an import there
# would wait on the import lock of whatever thread holds it. What was imported is not
# used, and nothing of it is read, until the first walk has shown it to be the
# interpreter's own (:func:`_collector`, :func:`_reference_type`).
_gc: Any = None
_weakref: Any = None
try:
    import gc as _gc_module

    _gc = _gc_module
except KeyboardInterrupt:
    raise
except BaseException:  # noqa: BLE001, S110, a refused import leaves one reading off
    pass
try:
    import weakref as _weakref_module

    _weakref = _weakref_module
except KeyboardInterrupt:
    raise
except BaseException:  # noqa: BLE001, S110, a refused import leaves one reading off
    pass

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
    is merely absent from a schema is not one of them.

    A callable ``inspect.signature`` reads as a ``functools.partialmethod`` is refused
    for that before its parameters are looked at: the parameters it shows are those of
    what the partialmethod binds, and the function ``functools`` builds for it carries no
    annotations, so any other reason given for it would be about a signature that is not
    its own. So the walk is made first and asked for that one refusal; every other
    outcome of it is reported after the parameters, where it was reported before."""
    try:
        parameters = inspect.signature(func).parameters
        # resolves string annotations (PEP 563) and keeps Annotated so it can be refused
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception:  # noqa: BLE001, an uninspectable callable cannot be judged
        try:
            own_parameters(func)
        except PartialMethodSignature as refused:
            return str(refused)
        except Exception as raised:  # noqa: BLE001, uninspectable either way
            if _a_descriptor(raised):
                return _DESCRIPTOR
        return "its signature cannot be inspected"
    # ``inspect.signature`` reads a ``functools.wraps`` wrapper as the function it wraps, and
    # honours a declared ``__signature__``: a wrapper's own parameter (``force=True``) would
    # run with its default while the gate judged a call without it
    reached: Any = None
    failed: Optional[BaseException] = None
    try:
        reached = _reached(func, skip, parameters)
    except PartialMethodSignature as refused:
        return str(refused)
    except Exception as raised:  # noqa: BLE001, reported below, after the parameters
        failed = raised
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
    if isinstance(failed, DeclaredSignature):
        return "it declares a signature that is not its own"
    named = (HiddenWrapper, DispatchedAtCallTime, BoundArgumentKept)
    if isinstance(failed, named) or _a_descriptor(failed):
        return str(failed)  # each of these says its own cause
    if failed is not None:  # an uninspectable callable cannot be judged
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


class UnreadCache(HiddenWrapper):
    """A ``functools.lru_cache`` wrapper on the way names one function as ``__wrapped__``
    and calls another, and which one it calls cannot be told from what the cache refers
    to (:func:`_cache_calls`). A subclass of :class:`HiddenWrapper`, so a gate that
    refuses that one refuses this; its message is the cause and what to do, and names no
    value. A cache whose function can be told is not refused for this: it is read."""


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


class PartialMethodSignature(TypeError):
    """A callable on the way is one ``inspect.signature`` reads as a
    ``functools.partialmethod``: it shows the parameters of what the partialmethod binds,
    less what is bound, in place of its own, so the walk cannot read the callable or
    anything under it. ``functools.partialmethod`` over a callable that does not bind as a
    method (a callable object, a class-based decorator) hands out such a callable, and
    ``functools.update_wrapper`` copies its mark onto every wrapper of one.
    :func:`_refuse_partialmethod` says exactly what is refused and what is not: a
    ``partialmethod`` over a plain function, read from an instance, is a
    ``functools.partial`` and is walked. The message names the cause and what to do, and
    no value."""


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

    ``inspect.signature`` follows ``__wrapped__`` and honours a declared signature
    (``__signature__``, a ``__text_signature__`` that is not the interpreter's own, or
    the code object of an object that looks like a function),
    on the callable or on what it reads the callable through: the class of a callable
    object and its ``__call__``, a ``functools.partial``'s function, and what a method
    binds, which is its function or an object bound as one (by its own ``__get__``, by
    ``types.MethodType``, by a ``classmethod``), with that object's class and
    ``__call__`` in turn. The first is turned off; the second raises
    :class:`DeclaredSignature` at each of those places, on every callable the walk
    reads, wherever the lookup ``inspect.signature`` makes finds one
    (:func:`_refuse_declared_signature`, :func:`_declares_text`), the wrappers looked
    through below included. It is not looked for on the receiver of a method, nor on a
    wrapper a callable only holds (:func:`_refuse_hidden_wrapper`). Wrappers that only pass arguments through (``lru_cache``, ``cache``, a
    ``staticmethod``, and ``lru_cache`` over a method) are looked through, a cache
    wrapper to what it calls, which need not be the function it names
    (:func:`_cache_target`, which raises :class:`UnreadCache`, a :class:`HiddenWrapper`,
    where that cannot be known). On a plain function and under a method alike: a method
    over a cache named for a plain function is read as what the cache calls, with the
    object bound, whether that is a plain function, a ``functools.partial``, a callable
    object or a second cache. One reading is kept as 0.15.0 made it: a method over a
    cache NAMED for something that is not a plain function is read by the names alone.
    A class is not a
    callable this reads: calling it runs ``__new__`` and ``__init__``. Neither is the method
    ``functools.partialmethod`` builds over a callable that does not bind as one, nor
    anything else ``inspect.signature`` would read as a ``functools.partialmethod``
    (:func:`_refuse_partialmethod`): that raises :class:`PartialMethodSignature`, a
    ``TypeError``.
    """
    if _depth > 8:
        raise ValueError("too many wrappers to inspect")
    if isinstance(func, type):
        raise TypeError("a class is not inspected as a callable")
    if isinstance(func, staticmethod):
        _refuse_shown_signature(func)
        return own_parameters(func.__func__, _depth + 1)
    if isinstance(func, _PASS_THROUGH):
        _refuse_shown_signature(func)
        return own_parameters(_cache_target(func), _depth + 1)
    if inspect.ismethod(func) and isinstance(func.__func__, _PASS_THROUGH):
        _refuse_shown_signature(func)  # a method shows what its cache declares
        inner, named = _cache_target(func.__func__), func.__func__.__wrapped__
        if inspect.isfunction(named):
            # what the cache calls, with the object bound, whatever it is: the function
            # it names, a wrapper, a ``functools.partial``, a callable object, a second
            # cache. The function it names is read as well, bound, as a callable of its
            # own (:func:`_next_callables`), so nothing 0.15.0 refused for it is accepted
            if isinstance(inner, type):
                raise TypeError("a class is not inspected as a callable")
            return own_parameters(types.MethodType(inner, func.__self__), _depth + 1)
        # a cache NAMED for something that is not a plain function (a second cache, a
        # callable object) is read here as 0.15.0 read it, by the names alone and without
        # the object bound: what that refused stays refused. The callables behind it are
        # read, as what they are, when the walk reaches them
        for _ in range(8):
            if not isinstance(named, _PASS_THROUGH):
                break
            named = named.__wrapped__
        return own_parameters(named, _depth + 1)
    _refuse_declared_signature(func, _depth)
    try:
        return inspect.signature(func, follow_wrapped=False).parameters
    except ValueError:
        # ``inspect`` shows no signature for a callable object whose class defines
        # ``__get__``: the same ``ValueError`` as before, saying the cause and what to do
        if _read_as_a_descriptor(func):
            raise ValueError(_DESCRIPTOR) from None
        raise


_DESCRIPTOR = (
    "a callable on the way is an object whose class defines `__get__` (a class-based "
    "decorator that binds as a method, or a `staticmethod` object called as a method): "
    "`inspect.signature` reads such an object as a built-in and shows no parameters for "
    "it, so what it runs cannot be read; register a plain function that calls it, or "
    "give the class no `__get__` and bind the object another way"
)


def _a_descriptor(raised: Any) -> bool:
    """True for the ``ValueError`` :func:`own_parameters` raises with that reason."""
    return type(raised) is ValueError and raised.args == (_DESCRIPTOR,)


def _read_as_a_descriptor(func: Any) -> bool:
    """True if what ``inspect.signature`` reads for ``func`` (``func``, or the function of
    the methods it is bound through) is an object ``inspect`` takes for a method
    descriptor, because its class defines ``__get__`` and no ``__set__``, while it is
    called through a ``__call__`` written in Python, or is a ``staticmethod`` object.
    ``inspect.signature`` shows no signature for those ("no signature found for
    builtin"). Asked only after it has raised, with the test ``inspect`` itself makes."""
    for _ in range(8):
        if not inspect.ismethod(func):
            break
        func = func.__func__
    if not inspect.ismethoddescriptor(func):
        return False
    call = inspect.getattr_static(type(func), "__call__", None)
    return isinstance(func, staticmethod) or inspect.isfunction(call)


_REWRAPPED_CACHE = (
    "a `functools.lru_cache` wrapper names one function as `__wrapped__` and calls another, "
    "which cannot be told from what the cache refers to; apply the cache to the function "
    "itself, or call the cached function from a plain function and register that one"
)


def _called_by(cache: Any, referred: Any) -> Any:
    """The function a ``functools.lru_cache`` wrapper calls, picked out of ``referred``,
    what ``gc.get_referents`` lists for it; ``_MISSING`` where the list is not laid out
    as one of the two ways below.

    The wrapper's C type lists what it refers to in a fixed order, the same for a cache
    that is unbounded, bounded, typed or of size zero, empty or full, and
    ``tests/test_invocation.py`` holds every supported interpreter to it:

    - Python 3.10 to 3.15 (3.14 free-threaded included): the wrapper's type; for each
      entry of a BOUNDED cache its key, its result and the type of the entry; then the
      cache's dictionary, THE FUNCTION, an ``object()`` that marks keywords, the type of an
      entry, the ``CacheInfo`` class, and the wrapper's own ``__dict__``;
    - Python 3.9: for each entry of a bounded cache its key and its result; then THE
      FUNCTION, the cache's dictionary, the ``CacheInfo`` class, and the wrapper's own
      ``__dict__``.

    So the function is the fifth from the end, or the fourth, and never one of the keys
    and results a bounded cache holds, which all come before the cache's dictionary: a
    cached result that is itself a function, or the very function the wrapper names,
    cannot be taken for it. The layout is recognised by what stands around the function,
    each by exact type or by identity, and anything else is ``_MISSING``. Nothing listed
    is called, compared but by identity, or asked for an attribute."""
    if type(referred) is not list:
        return _MISSING
    own, size = _cache_dict(cache), len(referred)
    if (
        size >= 7
        and referred[0] is _CACHE_WRAPPER
        and referred[-1] is own
        and type(referred[-2]) is type
        and type(referred[-3]) is type
        and type(referred[-4]) is object
        and type(referred[-6]) is dict
    ):
        return referred[-5]
    if (
        size >= 4
        and referred[-1] is own
        and type(referred[-2]) is type
        and type(referred[-3]) is dict
    ):
        return referred[-4]
    return _MISSING


def _cache_calls(cache: Any, named: Any) -> Any:
    """What a ``functools.lru_cache`` wrapper calls: ``named``, the function its own
    ``__dict__`` names as ``__wrapped__``, or another; ``_MISSING`` where that cannot be
    known.

    ``functools.lru_cache`` names the function it caches as ``__wrapped__``, and the walk
    read the wrapper as that function. ``functools.wraps(f)`` applied over the cache
    (``@functools.wraps(f)`` above ``@functools.lru_cache`` above a wrapper, an ordinary
    way to write a caching decorator, or ``functools.update_wrapper(cached, f)``)
    overwrites the name with ``f``, while the function the cache calls, the wrapper, is
    held only in C. ``gc.get_referents`` lists what the cache refers to by asking the
    wrapper's own C type and nothing else: no attribute is looked up and nothing the
    application wrote is called (the interpreter raises its ``gc.get_referents`` audit
    event, as it raises one for other reads here). The function is picked out of that
    list by its place (:func:`_called_by`) and read as what the cache calls: a wrapper
    that only passes its arguments on is accepted, and one with a parameter of its own
    gets the refusal a wrapper written without the cache gets.

    Where the list is not laid out as either known way, ``_MISSING``: what the cache
    calls cannot be told, whatever is listed (no supported interpreter lays it out
    another way, and a test holds each to that). And where the listing is not to be had,
    nothing is known and the name stands, as it does in 0.15.0: the checks are off
    (:func:`_prepare`); ``gc`` could not be imported, or is not the interpreter's own
    module, or has another function under ``get_referents``, at the first walk
    (:func:`_collector`); the listing raises, an audit hook's doing, whatever it raises
    but an interrupt from the keyboard, which is not caught; or the wrapper is an
    instance of a subclass of the cache type, which only Python 3.9 lets a class be.

    Why a listing that is stopped is not passed on: 0.15.0 asked no such question, so a
    hook that stops it would stop every tool with a cache that 0.15.0 reads, and the
    listing is the one read of the walk it can do without. A ``BaseException`` an audit
    hook raises at any other read of the walk (the ``__code__`` a signature is read
    from) is passed on, here as in 0.15.0, because nothing can be read without it."""
    if not _prepare() or _referents is None or type(cache) is not _CACHE_WRAPPER:
        return named
    called = _told(cache)
    if called is not _MISSING:
        return called
    try:
        referred = _referents(cache)
    except KeyboardInterrupt:
        raise
    except BaseException:  # noqa: BLE001, a listing that is refused proves nothing
        return named
    called = _called_by(cache, referred)
    _tell(cache, called)  # nothing is told of ``_MISSING``: it cannot be weakly referenced
    return called


#: What was found once about an object and cannot change while it lives, by the object's
#: ``id()``: a weak reference to the object, which tells it from a later object at the
#: same address, and what was found. A plain ``dict`` with ``int`` keys: no code written
#: in Python runs to read or write it, the standard library's included.
_KEPT: Dict[int, Any] = {}
#: The size at which entries of objects that are gone are dropped (:func:`_keep`).
_SWEEP_AT = 256


def _kept(owner: Any) -> Any:
    """What :func:`_keep` kept for ``owner``, or ``_MISSING``. One read of a ``dict`` under
    an ``int`` and one call of a weak reference, both in C; nothing of ``owner`` is asked."""
    entry = _KEPT.get(id(owner))
    if entry is None or entry[0]() is not owner:
        return _MISSING  # nothing kept, or kept for an object that stood at this address
    return entry[1]


def _keep(owner: Any, value: Any) -> None:
    """Keep ``value`` for as long as ``owner`` lives, without keeping ``owner`` alive: under
    its ``id()``, beside a weak reference to it made by the interpreter's own type
    (:func:`_reference_type`), with no callback. Nothing is kept where that type is not to
    be had, or for an object that cannot be weakly referenced. An entry whose object is
    gone is never read as another's (:func:`_kept`) and is dropped at the next sweep,
    not before: when the dictionary has reached twice the entries that were live at the
    last one (256 at least). So the dead can number as many as were live then, and each
    costs its key, a two-item tuple and a dead reference, with what it held (the names of
    a code object): about 300 bytes for a cache on Python 3.12. Review round eight
    measured there: 100,000 caches named again, alive and walked, held 35.6 MiB; dropped
    at once, their entries held 29.3 MiB until 12,116 more objects had been walked."""
    global _KEPT, _SWEEP_AT
    try:
        entry = (_reference(owner), value)
    except TypeError:  # no reference type to be had (None), or not an object it can refer to
        return
    if len(_KEPT) >= _SWEEP_AT:
        # a new dictionary of the entries whose objects live, put in place in one step;
        # an entry another thread adds to the old one meanwhile is found again later
        live = {}
        for key in list(_KEPT):
            kept = _KEPT.get(key)
            if kept is not None and kept[0]() is not None:
                live[key] = kept
        _SWEEP_AT = max(256, 2 * len(live))
        _KEPT = live
    _KEPT[id(owner)] = entry


def _told(cache: Any) -> Any:
    """The function ``cache`` was found to call at an earlier walk, or ``_MISSING``.

    A cache calls the same function for as long as it lives (the C type sets it once and
    offers no way to change it), and a bounded cache lists every key and result it holds
    along with it: tens of thousands of objects for a large cache that is full, on every
    walk, with every thread stopped for it on a free-threaded build. So what was told
    once is kept (:func:`_keep`), weakly both ways, so that neither the cache nor the
    function lives longer for it. A function that cannot be weakly referenced is told
    again each time.

    The FIRST walk of each cache still lists it, and on a free-threaded build the
    listing stops every thread, whatever the size of the cache. Review round eight
    measured, on 3.14 free-threaded with 16 threads, each walking a new cache named
    again every time: 570 walks a second, where 0.15.0, which lists nothing, made
    103,061. A cache walked before costs no listing: 85,186 a second against 113,458."""
    reference = _kept(cache)
    called = reference() if reference is not _MISSING else None
    return _MISSING if called is None else called


def _tell(cache: Any, called: Any) -> None:
    """Keep what ``cache`` was found to call (:func:`_told`), weakly. Nothing is kept
    where there is no reference type, or for what cannot be weakly referenced: a
    built-in function, and ``_MISSING``, which is a plain ``object()``."""
    try:
        _keep(cache, _reference(called))
    except TypeError:  # no reference type to be had, or a callable it cannot refer to
        pass


def _cache_target(cache: Any) -> Any:
    """The function a ``functools.lru_cache`` wrapper on the walk calls
    (:func:`_cache_calls`). Raises :class:`UnreadCache`, a :class:`HiddenWrapper`, where
    it cannot be known."""
    called = _cache_calls(cache, cache.__wrapped__)
    if called is _MISSING:
        raise UnreadCache(_REWRAPPED_CACHE)
    return called


def _declares_text(func: Any) -> bool:
    """Whether ``func`` shows a ``__text_signature__`` that something declared for it.

    ``inspect.signature`` reads a signature from that text, in place of the parameters
    the callable binds, for a plain function that carries the attribute (and so for a
    method, a ``functools.partial``, a cache or a ``staticmethod`` over one, and for a
    wrapper ``functools.wraps`` copied it onto), for an object whose class defines
    ``__get__``, and, from Python 3.13, for any callable object. Which callables it reads
    so has changed between versions, so the text is refused wherever it is declared,
    on every Python, also where this one would not read it.

    Not declared, and not refused: the text of a callable implemented in C (a built-in
    function, a method of a built-in type), which the interpreter derives from the
    callable itself. That one is told by where the lookup finds the name: on the
    callable's type, as the accessor the interpreter defines there, with nothing in the
    callable's own ``__dict__``. Anything else that answers with a text is a
    declaration: an attribute of the callable or of its class, a property, a hook. A
    method shows the text of what it binds, so it is asked of that: a built-in bound as
    a method (``types.MethodType(len, obj)``, a ``classmethod`` of one) declares
    nothing."""
    for _ in range(_DEPTH):
        if not inspect.ismethod(func):
            break
        func = func.__func__
    if not getattr(func, "__text_signature__", None):
        return False
    if "__text_signature__" in getattr(func, "__dict__", {}):
        return True
    for owner in type(func).__mro__:
        if "__text_signature__" in owner.__dict__:
            return type(owner.__dict__["__text_signature__"]) is not types.GetSetDescriptorType
    return True


def _refuse_declared_signature(func: Any, depth: int) -> None:
    """Raise :class:`DeclaredSignature` if ``func``, or what ``inspect.signature`` would
    read in its place, declares ``__signature__``, and :class:`PartialMethodSignature` if
    either would be read as a ``functools.partialmethod`` (:func:`_refuse_partialmethod`).

    The test for ``__signature__`` is the lookup ``inspect.signature`` makes: the
    attribute, asked of the callable. So it is found wherever that lookup finds it: in
    the callable's ``__dict__`` or a slot, on its class (an attribute, a property), and
    where the class answers for the name in ``__getattr__`` or ``__getattribute__`` (a
    forwarding proxy over a function that declares one). One the METACLASS of its class
    declares is not found, as ``inspect.signature`` does not find it on the instance.
    A name present with the value None is refused too: ``inspect.signature`` reads
    through it, and stops following ``__wrapped__`` at it.

    Two more declarations are refused with it: a ``__text_signature__`` that is not the
    interpreter's own (:func:`_declares_text`), and a code object shown by an object
    that is called through a ``__call__`` written in Python and names nothing as
    ``__wrapped__``, which ``inspect.signature`` reads as a function where the object
    shows a name and defaults too.

    Asking runs what the class defines for the lookup of each of the three names.
    Where that raises anything but ``AttributeError`` this raises ``ValueError``: the
    callable cannot be inspected. ``inspect.signature`` raises there too for
    ``__signature__``, and for ``__text_signature__`` from Python 3.13; before that it
    does not ask an ordinary callable object for its text, and reads one this refuses."""
    if depth > 8:
        raise ValueError("too many wrappers to inspect")
    _refuse_shown_signature(func)
    _refuse_partialmethod(func)
    if inspect.ismethod(func):
        # a method shows the attributes of what it binds as its own, but not what the
        # CLASS of that declares: an object bound as a method (by its own ``__get__``,
        # or under a cache) is read by ``inspect.signature`` as the object is. Not a
        # wrapper more: a method cannot bind itself, so the same depth
        _refuse_declared_signature(func.__func__, depth)
    elif isinstance(func, functools.partial):
        _refuse_declared_signature(func.func, depth + 1)
    elif not (inspect.isfunction(func) or inspect.isbuiltin(func) or isinstance(func, type)):
        call = getattr(type(func), "__call__", None)
        if call is not None and not isinstance(call, _C_CALL):
            # an object called through a ``__call__`` written in Python that shows a
            # code object: ``inspect.signature`` reads one that also shows a name and
            # defaults as a function, from that code and those defaults, in place of
            # its ``__call__``. Not one that names a ``__wrapped__`` (a transparent
            # proxy): ``inspect.signature`` follows that first, and so does the walk
            try:
                shows_code = isinstance(
                    getattr(func, "__code__", None), types.CodeType
                ) and not hasattr(func, "__wrapped__")
            except Exception as raised:  # noqa: BLE001, ``inspect.signature`` fails too
                raise ValueError("a declared signature cannot be looked for") from raised
            if shows_code:
                raise DeclaredSignature
            _refuse_declared_signature(call, depth + 1)


def _refuse_shown_signature(func: Any) -> None:
    """Raise :class:`DeclaredSignature` if ``func`` itself shows a declared signature:
    the test of :func:`_refuse_declared_signature`, for this one callable. Asked also of
    the wrappers :func:`own_parameters` looks through (a cache, a method over one, a
    ``staticmethod`` object): ``inspect.signature`` stops at a ``__signature__`` one of
    them carries and reads it, in place of the function under it."""
    try:
        declares = (
            hasattr(func, "__signature__")
            or "__signature__" in getattr(func, "__dict__", {})
            or _declares_text(func)
        )
    except Exception as raised:  # noqa: BLE001, ``inspect.signature`` fails on it as well
        raise ValueError("a declared signature cannot be looked for") from raised
    if declares:
        raise DeclaredSignature


#: The attribute under which ``functools.partialmethod`` leaves itself on the function it
#: builds, and under which ``inspect.signature`` looks for it: ``_partialmethod`` up to
#: Python 3.12, ``__partialmethod__`` from 3.13. Both are looked for on every Python.
_PARTIALMETHOD_MARKS = ("_partialmethod", "__partialmethod__")

_PARTIALMETHOD = (
    "it is a `functools.partialmethod` over a callable that does not bind as a method (a "
    "callable object, a class-based decorator), or a wrapper of one: its signature shows "
    "the parameters of what the partialmethod binds, not its own, so what runs under it "
    "cannot be read; register a plain function that calls it, or put the partialmethod "
    "over a plain function"
)


def _refuse_partialmethod(func: Any) -> None:
    """Raise :class:`PartialMethodSignature` if ``inspect.signature`` would read ``func`` as a
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
    ``isinstance``, made only for a callable that shows something under one of the names
    and only while ``functools.partialmethod`` is a plain class (its type is exactly
    ``type``, as the standard library's is). A mark that is anything but an instance of it
    is not refused. Where something else stands in ``functools`` under that name, or
    nothing does, the check asks nothing and refuses nothing: a callable without a mark is
    read as 0.15.0 reads it, and one with a mark is left to ``inspect.signature``. Nothing
    is prepared for the check and nothing found is called.

    Not found, by design: a partial application written by hand, and a callable that
    reaches such a method through a helper that names nothing as ``__wrapped__`` and does
    not carry the mark."""
    for mark in _PARTIALMETHOD_MARKS:
        found = getattr(func, mark, None)
        if found is None:
            continue
        # asked only once a mark is there, and only of the standard library's class: where
        # another object stands in ``functools`` under that name (a tracing function, a
        # class with a metaclass of its own) nothing is asked, and ``inspect.signature``
        # does with the mark what it did before this check existed
        kind = getattr(functools, "partialmethod", None)
        if type(kind) is type and isinstance(found, kind):
            raise PartialMethodSignature(_PARTIALMETHOD)


# The two checks below read what a callable on the walk HOLDS. A held value is not a
# callable the walk was asked to read: it can be a lazy proxy that connects when touched, a
# remote stub that answers every attribute, a settings object. So the checks are exact and
# run nothing. A held value is looked at only when its type, read with ``type()``, is one
# of a closed set CPython itself builds, and then only through what that type itself
# defines in C: its accessors, and for a plain function its ``__code__``, its
# ``__closure__`` and the contents of its cells, which no class can stand in for (a plain
# function's type cannot be subclassed). No attribute an application could answer is
# looked up, no descriptor a class declares is called, nothing is compared but by
# identity, nothing is hashed. A value of any other type is skipped without being read.
#
# Two rules hold for every dictionary read here that is not this module's own (a function's
# ``__dict__``, an instance's, a class's, a registry). It is listed in one call into C
# (``list(d.items())`` of a plain ``dict``, :func:`_listed`) before any loop, so another
# thread changing it cannot break the loop; on Python 3.9 to 3.11 a collection that call
# itself sets off can change the dictionary and make it raise, and it is then made again.
# And no key is looked up in it: a key is compared only when it is exactly a ``str``, so
# no key's ``__hash__`` or ``__eq__`` runs. One dictionary is read another way: the module
# globals a wrapper names are looked up by name, as the wrapper's own code looks them up
# (:func:`_named_globals`, which says what that can compare).
#
# What is kept between walks is kept in a plain ``dict`` of this module's own, under the
# ``id()`` of an object and beside a weak reference to it (:func:`_keep`): reading and
# writing it runs no code written in Python, the standard library's included.
#
# Nothing of this is built when the module is imported. Importing it reads nothing of
# ``functools`` that 0.15.0 did not read, and raises nothing that importing 0.15.0 does
# not raise, whatever stands in ``functools``, in ``sys.modules`` or in the audit hooks,
# with one exception: it does import two modules 0.15.0 did not, ``gc`` and ``weakref``,
# each inside a guard (see the top of the module), and a ``KeyboardInterrupt`` raised
# while one of the two is imported (by a hook, or from the keyboard) is passed on.
# The accessors and the standard-library objects the checks rely on are found at the first
# walk (:func:`_prepare`), each one identified as the standard library's own before it is
# used, and none of them called. If any cannot be, both checks are off: the walk is then
# what it was before them, and nothing is raised.

_MISSING: Any = object()
_DEPTH = 32
_VARIADIC = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS
_HEAP_TYPE, _IMMUTABLE_TYPE = 1 << 9, 1 << 8  # ``Py_TPFLAGS_HEAPTYPE``, ``..._IMMUTABLETYPE``

#: None until the first walk; then True if the checks are on, False if they are off.
_READY: Optional[bool] = None
_function_dict: Any = None
_function_code: Any = None
_function_globals: Any = None
#: ``gc.get_referents``, or None where ``gc`` is not the interpreter's own module.
_referents: Any = None
#: The interpreter's weak reference type, or None where it is not to be had.
_reference: Any = None
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

    Run at the first walk, not at import, and importing nothing: ``gc`` and ``weakref``
    were imported with this module or are left out. Everything is read through the
    C-level accessors of built-in types and from dictionaries listed as :func:`_entry`
    lists them; nothing found is called, and no attribute is looked up on ``functools``
    or on anything in it.
    The checks are on only if ``functools`` is a plain module whose ``partial`` and whose
    ``lru_cache`` wrapper are types C code defines, with the accessors those have. What
    stands in ``functools`` is read as it is then: a ``functools`` reloaded afterwards
    builds dispatchers from new code objects, which are not recognised. Any ``Exception`` on
    the way (an audit hook that refuses a read, an interpreter built another way) turns
    them off. What the first walk finds stands for the process: a ``functools`` patched
    when it is made and restored afterwards leaves the checks as that walk found them."""
    global _READY, _function_dict, _function_code, _method_function, _static_function
    global _function_globals, _referents, _reference
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
        _function_globals = _type_dict(types.FunctionType)["__globals__"].__get__
        _method_function = _type_dict(types.MethodType)["__func__"].__get__
        _static_function = _type_dict(staticmethod)["__func__"].__get__
        _class_function = _type_dict(classmethod)["__func__"].__get__
        _module_dict = _type_dict(types.ModuleType)["__dict__"].__get__
        found = _standard_library()
    except Exception:  # noqa: BLE001, whatever cannot be read turns the checks off
        found = None
    try:
        listing = _collector()
    except Exception:  # noqa: BLE001, without it one check is off, the rest stand
        listing = None
    try:
        reference = _reference_type()
    except Exception:  # noqa: BLE001, without it nothing is kept between walks
        reference = None
    # threads that make their first walk at once each get here with what they found; the
    # first to arrive decides for the process, in one step, and every thread sets the same
    # values, so no walk ever sees a part of one finding beside a part of another
    found, _referents, _reference = _DECIDED.setdefault("found", (found, listing, reference))
    if found is not None:
        _PARTIAL, _partial_function, _partial_args, _CACHE_WRAPPER, _cache_dict = found[:5]
        _WRAPPER_CODE, _DISPATCH_CODE, _METHOD_ACCESS = found[5:]
    _READY = found is not None  # last: once it is set, everything above is
    return _READY


#: What the first walk of the process found (:func:`_prepare`), under "found": what the
#: checks read ``functools`` with, ``gc.get_referents`` and the weak reference type.
_DECIDED: Dict[str, Any] = {}


def _collector() -> Any:
    """``gc.get_referents``, if this module imported ``gc``, that is a plain module, and
    what it has under that name is the built-in function of that module and of that name,
    by its type and by what the function itself says in C; else None. So a ``gc`` put in
    ``sys.modules`` by the application is not used, and neither is another function of
    ``gc`` put under the name (``gc.get_referrers``, which would list something else).
    Asked at the first walk, not when this module is imported; nothing found is called
    here."""
    if type(_gc) is not types.ModuleType:
        return None
    listing = _entry(_module_dict(_gc), "get_referents")
    if type(listing) is not types.BuiltinFunctionType:
        return None
    if listing.__self__ is not _gc or listing.__name__ != "get_referents":
        return None
    return listing


def _reference_type() -> Any:
    """The interpreter's weak reference type, ``weakref.ref``, if this module imported
    ``weakref``, that is a plain module, and what it has under that name is that type;
    else None, and nothing is kept between walks (:func:`_keep`). It is known as the one
    type C code defines whose own dictionary has ``__callback__`` as a member the type
    itself lays out: a class written in Python is not a type C code defines, and another
    built-in type put under the name (``type``, ``str``, ``bool``, the proxy types) has
    no such member. Calling that type makes a weak reference in C: no code written in
    Python runs, and no callback is given."""
    if type(_weakref) is not types.ModuleType:
        return None
    kind = _entry(_module_dict(_weakref), "ref")
    if not _compiled(kind):
        return None
    callback = _entry(_type_dict(kind), "__callback__")
    return kind if type(callback) is types.MemberDescriptorType else None


def _compiled(kind: Any) -> bool:
    """True for a type C code defines: static, or declared immutable (a flag no class
    written in Python can have), so not a look-alike put in its place."""
    flags = _type_dict(type)["__flags__"].__get__
    return type(kind) is type and (
        not flags(kind) & _HEAP_TYPE or bool(flags(kind) & _IMMUTABLE_TYPE)
    )


def _standard_library() -> Any:
    """What the checks read ``functools`` with, if all of it is the standard library's own
    (:func:`_prepare`), else None: the ``partial`` type and the accessors of its function
    and arguments, the ``lru_cache`` wrapper type and the accessor of its dictionary, the
    two code objects of a dispatcher and where a single-dispatch method keeps its
    registry (those three None where the dispatcher is not identified)."""
    compiled = _compiled
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
    of a class), or ``_MISSING``: the items are listed in one call (:func:`_listed`), and
    no key is hashed or compared unless it is exactly a ``str``."""
    if type(attributes) is not dict and type(attributes) is not types.MappingProxyType:
        return _MISSING
    for key, value in _listed(attributes.items):
        if type(key) is str and key == name:
            return value
    return _MISSING


_TRIES = 8
_UNLISTED = "a dictionary the walk reads was changed each time it was listed"


def _listed(view: Any) -> List[Any]:
    """``list(view())``, where ``view`` is the ``items`` or ``values`` of a plain ``dict``
    or of the ``mappingproxy`` of a class: the whole listing is one call into C, so a
    thread that changes the dictionary cannot change it halfway through.

    On Python 3.9 to 3.11 that call is not always one step. Listing the items makes a
    tuple for each, a collection can start at any of them, and a finalizer or a
    weak-reference callback it runs can change the size of the very dictionary being
    listed: the call then raises ``RuntimeError`` ("dictionary changed size during
    iteration"). Nothing was read, so the listing is made again, up to eight times; a
    dictionary that changed at every one of them is not read at all, and the walk raises
    ``ValueError``, which a gate turns into a refusal. From 3.12 a collection does not
    start inside the call, and it does not raise."""
    for _ in range(_TRIES):
        try:
            return list(view())
        except RuntimeError:
            continue
    raise ValueError(_UNLISTED)


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


def _names(code: Any, depth: int = 0) -> List[str]:
    """The names the code of a function refers to that are not its locals or its closure
    (``co_names``: the globals it reads, and the attributes it reads off anything), with
    those of every function, lambda and comprehension defined inside it. Read from the
    code objects alone; nothing is run. A function nested more than 32 deep is not read
    further."""
    found = [name for name in code.co_names if type(name) is str]
    if depth < _DEPTH:
        for constant in code.co_consts:
            if type(constant) is types.CodeType:
                found.extend(_names(constant, depth + 1))
    return found


def _named_globals(function: Any) -> List[Any]:
    """The values of the module globals the code of a plain function names.

    A wrapper written beside the function it wraps can call a second wrapper of that
    function by its global name (``guarded = with_timeout(fetch)`` at module level, and a
    ``functools.wraps(fetch)`` wrapper that returns ``guarded(...)``): the second wrapper
    is in no closure cell, and is held all the same. The namespace is the function's own
    ``__globals__``, read through the C-level accessor and only when it is exactly a
    ``dict``. A name the code uses only as an attribute (``self.guarded``) and that is
    also a global is included: the code object does not tell the two apart. Builtins are
    not read. What is returned is then judged like anything else held: only a value of
    the closed set that reaches, by identity, a callable the walk read is a hidden
    wrapper, so a helper, a logger or a formatter the wrapper calls is not.

    Each name is looked up in the namespace, which is what the function's own code does
    with it each time it runs (a global is read by this lookup): a handful of reads,
    whatever the size of the module. This is the one dictionary here that is read by
    lookup and not by listing. Every name is exactly a ``str``, and so is every key of an
    ordinary module namespace, and the lookup then runs no code but the dictionary's own.
    A key that is not exactly a ``str`` is compared, by its own ``__eq__``, only if it
    hashes as one of the names looked up, as it is whenever the function reads that
    global. The names a code object refers to cannot change, so they are found once for
    each code object and kept (:func:`_keep`)."""
    namespace = _function_globals(function)
    if type(namespace) is not dict:
        return []
    code = _function_code(function)
    names = _kept(code)
    if names is _MISSING:
        names = tuple(dict.fromkeys(_names(code)))  # each once, in the order of the code
        _keep(code, names)
    held = []
    for name in names:
        try:
            value = namespace.get(name, _MISSING)
        except Exception:  # noqa: BLE001, a key of the namespace that raises when compared
            raise ValueError("the module globals a wrapper names cannot be read") from None
        if value is not _MISSING:
            held.append(value)
    return held


def _held(func: Any, wrapped: Any) -> List[Any]:
    """What the walked ``func`` holds: the closure cells of a plain function and the
    module globals its code names (:func:`_named_globals`); the instance attributes of any
    other callable, with the closure cells and the named module globals of the plain
    function its class has as ``__call__``. An instance attribute that is, by identity, also an attribute of the
    plain function ``func`` names as ``__wrapped__`` is left out:
    ``functools.update_wrapper`` copied it over, and a function's own attributes are not
    read."""
    if type(func) is types.FunctionType:
        return _cells(func) + _named_globals(func)
    copied = _function_dict(wrapped) if type(wrapped) is types.FunctionType else {}
    copied = _listed(copied.values) if type(copied) is dict else []
    held = [
        value
        for value in _listed(_instance_attributes(func).values)
        if not any(value is other for other in copied)
    ]
    # the ``__call__`` of its class is what runs: a class written inside a decorator closes
    # over what the decorator built, and one written at module level names it as a global
    call = _class_attribute(type(func), "__call__")
    if type(call) is types.FunctionType:
        held += _cells(call) + _named_globals(call)
    return held


def _class_attribute(kind: Any, name: str) -> Any:
    """What the class ``kind`` or a base of it has under ``name`` in its own dictionary,
    the first in the order Python looks them up, or ``_MISSING``: read from the
    dictionaries, so no descriptor is called."""
    for klass in _type_mro(kind):
        found = _entry(_type_dict(klass), name)
        if found is not _MISSING:
            return found
    return _MISSING


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


def _read_level(parameters: Any) -> Tuple[Tuple[str, ...], bool]:
    """The level of a callable the walk read (:func:`own_parameters`): its parameters as
    the rule below reads them, which is the names it takes, and whether it also takes
    ``*args`` or ``**kwargs`` and so can pass a call on."""
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    named = [name for name, parameter in parameters.items() if parameter.kind not in variadic]
    return tuple(named), len(named) < len(parameters)


def _held_level(function: Any, bound: int) -> Tuple[Tuple[str, ...], bool]:
    """The level of a held plain function, after the ``bound`` leading arguments it
    receives as a method: read from the code object Python binds a call with, never from a
    signature it shows."""
    code = function.__code__
    names, positional = code.co_varnames, code.co_argcount
    named = names[bound:positional]
    named += names[positional : positional + code.co_kwonlyargcount]
    return tuple(named), bool(code.co_flags & _VARIADIC)


_HELD_IN_BOUND_PLACE = (
    "`{name}` is the leading parameter of a wrapper a decorator holds, which passes the "
    "call on, in the place of an argument bound on the way (the object of a method): "
    "where that argument is handed on by keyword the wrapper takes the judged `{name}` "
    "itself, and the function can then run with a value no policy judged; register a "
    "plain function that names every parameter the tool takes"
)


def _held_in_bound_places(function: Any, bound: int) -> Tuple[str, ...]:
    """The leading names of a held plain function that the ``bound`` arguments are taken
    to fill (:func:`_held_level` leaves them out), where the function passes the call on
    (``*args`` or ``**kwargs``); none where it does not."""
    code = function.__code__
    if not code.co_flags & _VARIADIC:
        return ()
    return tuple(code.co_varnames[: min(bound, code.co_argcount)])


def _argument_of_its_own(
    levels: List[Any], declared: Any, skip: Any, judged: Optional[Any] = None
) -> Optional[str]:
    """Why some callable a call reaches would run with a value no policy judged, or None.

    The one rule for a parameter a wrapper has of its own, for a wrapper the walk reads and
    for one a callable holds alike. ``levels`` are the callables' parameters
    (:func:`_read_level`), ``declared`` the names the tool declares, ``skip`` the names the
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


def _walk(tool: Any, declared: Any) -> Any:
    """What one walk of ``tool`` has found (``found``, the parameters of each callable it
    read, which the walk returns), read (``read``) and found held (``held``), for the
    judgement made at its end, what is known of its bound arguments (``taken``, and
    ``given`` for :func:`_refuse_kept_by_name`; :func:`_rebound`), what the tool
    declares (``declared``, :func:`_declared`), and whether the two checks are on
    (``checked``, :func:`_prepare`: where it is False the walk collects nothing held).
    A plain namespace, not a class of this module: importing the module defines no
    class but the exceptions it adds."""
    declared = _MISSING if declared is None else declared
    return types.SimpleNamespace(
        tool=tool,
        declared=declared,
        found=[],
        read=[],
        held=[],
        taken=[],
        given=[],
        checked=_prepare(),
    )


def _declared(walk: Any) -> Any:
    """The parameters the tool of this walk declares (``inspect.signature``), or None
    where they cannot be read. Handed in by a caller that has read them; otherwise read
    here, once for the walk, and only when a check asks."""
    if walk.declared is _MISSING:
        try:
            walk.declared = inspect.signature(walk.tool).parameters
        except Exception:  # noqa: BLE001, what the tool declares cannot be read
            walk.declared = None
    return walk.declared


def _refuse_kept_by_name(walk: Any, levels: List[Any]) -> None:
    """Raise :class:`BoundArgumentKept` for a parameter :func:`_rebound` took as given by
    name (the tool declares it, so a call hands it its judged value as a keyword) where
    a wrapper keeps that keyword: a callable the walk read, or a wrapper one of them
    holds, that passes the call on (``*args`` or ``**kwargs``) and takes the name
    itself. The function's own parameter of that name is one the walk bound an argument
    to, so the rule that refuses a kept name (:func:`_argument_of_its_own`) does not see
    that two levels take it. ``levels`` are those of the whole walk, read and held,
    which is why this is asked at its end."""
    for taken, last in walk.given:
        kept = any(forwards and last.name in named for named, forwards in levels)
        if kept and last.default is not last.empty:
            raise BoundArgumentKept(_UNHANDED.format(taken=taken, filled=last.name))


def _hidden_levels(
    value: Any, read: List[Any], bound: int, in_bound_places: Optional[List[str]] = None
) -> List[Any]:
    """The levels of the wrappers the walk did not read between a held ``value`` and a
    callable it did read, or none when ``value`` reaches no such callable.

    ``value`` is followed through the closed set (:func:`_passes_on_to`) until a link is,
    by identity, one of ``read``. Each plain function passed on the way is a wrapper the
    walk never read, and its parameters are returned for the same judgement the walk's own
    levels get. ``bound`` is the number of leading arguments the callable that holds
    ``value`` receives bound (its receiver, when the walk reached it as a method): the
    held wrapper is passed them too, so they are not parameters of its own, exactly as
    they are not for the level the walk read. Each bound method, ``classmethod`` and bound
    positional argument of a ``functools.partial`` on the way adds to it. That holds
    where the holder hands them on by position. Where it hands one on by keyword, the
    leading name of a held wrapper that passes the call on is free, and takes a judged
    keyword of that name: those names are added to ``in_bound_places``
    (:func:`_held_in_bound_places`), for the caller to refuse the ones the tool
    declares. Raises
    :class:`HiddenWrapper` when the way cannot be judged by parameters: it is still going
    after 32 links, or a link is a single-dispatch function or method that can run
    anything but its base."""
    links: List[Any] = []
    levels: List[Any] = []
    places: List[str] = []
    dispatches = False
    for _ in range(_DEPTH):
        if any(value is seen for seen in read):
            if dispatches:
                raise HiddenWrapper(_HIDDEN_WRAPPER)
            if in_bound_places is not None:
                in_bound_places.extend(places)
            return levels
        if any(value is link for link in links):
            return []  # a cycle: it reaches nothing the walk reads
        inner = _passes_on_to(value)
        if inner is _MISSING:
            return []
        if type(value) is types.FunctionType:
            levels.append(_held_level(value, bound))
            places.extend(_held_in_bound_places(value, bound))
            dispatches = dispatches or _chosen_at_call_time(value)
        elif type(value) is _CACHE_WRAPPER:
            # a held cache that calls another function than the one it names: that
            # function is one more wrapper on the way, read like the others; where it
            # cannot be known, or is not a plain function, the way cannot be judged
            called = _cache_calls(value, inner)
            if called is _MISSING or (
                called is not inner and type(called) is not types.FunctionType
            ):
                dispatches = True
            elif called is not inner:
                levels.append(_held_level(called, bound))
                places.extend(_held_in_bound_places(called, bound))
                dispatches = dispatches or _chosen_at_call_time(called)
        links.append(value)
        if type(value) is _PARTIAL:
            args = _partial_args(value)
            bound += len(args) if type(args) is tuple else 0
        elif type(value) is types.MethodType or type(value) is classmethod:
            bound += 1
        value = inner
    raise HiddenWrapper(_HIDDEN_WRAPPER)  # still going: what it reaches is not known


def _refuse_hidden_wrapper(walk: Any, skip: Any) -> List[Any]:
    """Raise :class:`HiddenWrapper` if a callable on the walk holds a wrapper the walk did
    not read, with a parameter of its own; otherwise return the levels of the wrappers
    held (:func:`_hidden_levels`). ``walk.found`` are the parameters of the callables the
    walk read; what the tool declares is asked of the walk (:func:`_declared`) once
    something held is to be judged, and where it cannot be read then, the tool is
    refused as it stands.

    A decorator can apply another wrapper inside itself and call the result, while
    ``functools.wraps`` sets ``__wrapped__`` to the undecorated function: the walk then
    never reads the wrapper in between, which can keep an argument for itself or fill a
    default of its own. This finds exactly one shape of that, and runs nothing to do it.

    A callable on the walk that names a ``__wrapped__`` holds what is in its closure cells
    and the module globals its code names (a plain function, :func:`_named_globals`) or
    its instance attributes (any other callable). A held value is a hidden wrapper when
    all of these hold:

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

    Refused with it: a held wrapper that passes the call on and whose leading
    parameter, in the place of an argument the holder receives bound, is one the tool
    declares. The holder may hand that argument on by keyword (``held(*args,
    tenant=self.tenant, **kwargs)`` over ``def wrapper(limit=100, *args, **kwargs)``):
    the judged keyword of that name then stops at the held wrapper, and whether that
    hands it on cannot be read. A held ``def wrapper(self, *args, **kwargs)`` is not
    refused for this: no tool declares ``self``.

    Only parameters are judged: a held wrapper is not read for the other things the walk
    refuses a callable it reads for (a declared ``__signature__``). And what a callable
    object holds is judged whether or not its ``__call__`` uses it: an extra entry point
    kept on a class-based decorator (``self.batch = batched(self.function)``) is refused
    if it has a parameter of its own.

    Not found, by design: a second wrapper that does not itself name what it wraps (a plain
    inner function, a lambda, a helper, a bound method of an object that holds the
    function); one that is, or is held by, an object of any other type (a callable object,
    a proxy, a wrapper a C extension builds, a subclass of a type of the closed set), or
    whose way to the function it wraps passes through such an object (a second wrapper
    reached through a proxy, as a method a ``wrapt`` decorator binds); one
    held anywhere but where a callable on the walk THAT NAMES A ``__wrapped__`` holds
    things, which is its closure cells and named module globals (a plain function), or its
    instance attributes and the closure cells and named module globals of its class's
    ``__call__`` (any other callable): so not one held by a callable on the walk that
    names nothing as ``__wrapped__``, nor in a class attribute, a slot, a container, a
    function's own attribute, a global of another module reached through an attribute, or
    a global of a function whose ``__globals__`` is not exactly a ``dict``; a
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
    in_bound_places: List[str] = []
    for value, bound in walk.held:
        hidden.extend(_hidden_levels(value, walk.read, bound, in_bound_places))
    if not hidden:
        return hidden
    declared = _declared(walk)
    if declared is None:
        raise HiddenWrapper(_HIDDEN_WRAPPER)
    for name in in_bound_places:
        if name in declared and name not in skip:
            # the holder may hand its bound argument on by keyword (``held(*args,
            # tenant=self.tenant, **kwargs)``): the judged keyword of this name then
            # stops at the held wrapper, and whether that hands it on cannot be read
            raise HiddenWrapper(_HELD_IN_BOUND_PLACE.format(name=name))
    levels = [_read_level(parameters) for parameters in walk.found] + hidden
    why = _argument_of_its_own(levels, declared, skip, range(len(walk.found), len(levels)))
    if why is not None:
        raise HiddenWrapper(why)
    return hidden


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
    items = _listed(registry.items)
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


class BoundArgumentKept(ValueError):
    """An argument bound on the way (the object of a method, a positional argument of a
    ``functools.partial``) is taken by a wrapper under one name, and on a function that
    wrapper reaches (the one it names as ``__wrapped__``, or one under any number of
    layers that only pass arguments on) it would fill a parameter of another name.
    Whether the wrapper hands the argument on cannot be read from a signature. If it
    keeps it, every argument bound after it fills the parameter before its own, and the
    last parameter the bound arguments would fill is left to its default: raised where
    that parameter has one and nothing gives it a judged value by name (the tool does
    not declare it, or a wrapper on the way takes it by name itself), so that no policy
    judged the value it runs with.

    The same signature is refused although it runs correctly, because the walk reads
    parameters and not bodies and the two cannot be told apart: a wrapper that does hand
    the argument on, and a facade that passes a value of its own in its place (``def
    search(self, *args, **kwargs): return search(self.tenant, *args, **kwargs)`` under
    ``functools.wraps(search)``, where ``tenant`` has a default). The remedy is the same
    for all of them: a plain function that names every parameter the tool takes and
    calls the facade. :func:`_rebound` says exactly what is refused and what is not. The
    message names the two parameters and what to do, and no value."""


_UNHANDED = (
    "an argument bound on the way (the object of a method, a positional argument of a "
    "`functools.partial`) is taken by a wrapper as `{taken}`, another name than the "
    "parameter it would fill on a function the wrapper reaches through `__wrapped__`. "
    "Whether the wrapper hands it on cannot be read from a signature: if it keeps it, each "
    "bound argument after it fills the parameter before its own, and `{filled}`, the last "
    "they would fill, runs at a default no policy judged. A "
    "wrapper that hands the argument on, or passes a value of its own in its place, has "
    "the same signature and is refused too, although it runs correctly. Call it from a "
    "plain function that names every parameter the tool takes, and register that one"
)


def _leading(parameters: Any) -> List[Any]:
    """The parameters a positional argument can fill, in order: those before the first
    that is ``*args``, keyword-only or ``**kwargs``."""
    leading = []
    for parameter in parameters.values():
        if parameter.kind not in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            break
        leading.append(parameter)
    return leading


def _taken_under(walk: Any, bound: Any, count: int) -> Tuple[Tuple[Optional[str], ...], Any, Any]:
    """What this walk knows of the arguments the callable ``bound`` binds
    (:func:`_rebound`): the name each of the ``count`` it is asked about was last taken
    under, None for one no wrapper has named; the doubt raised where the walk bound
    them (the name a wrapper may have kept one under), or None, which is for the
    arguments bound after them; and the doubt that is open for these same arguments
    further in, or None. Nothing is known of the tool itself or of any callable the walk
    did not bind. The names are given only where they are as many as are asked about: a method
    the walk bound has one, and is asked for none when a ``functools.partial`` of it
    binds arguments after its own."""
    for known, names, doubt, still in walk.taken:
        if known is bound:
            return (names if len(names) == count else (None,) * count), doubt, still
    return (None,) * count, None, None


def _bound_count(partial: Any) -> Optional[int]:
    """How many positional arguments a ``functools.partial`` binds, a subclass included:
    read through the C-level accessor of the standard library's type, which a subclass
    cannot change, so no attribute of the partial is looked up for it. None for an
    object that is a ``functools.partial`` for ``isinstance`` and not an instance of
    that type (another stands in ``functools`` since the first walk)."""
    if not any(base is _PARTIAL for base in _type_mro(type(partial))):
        return None
    return len(_partial_args(partial))


def _rebound(outer: Any, inner: Any, walk: Any) -> Any:
    """The callable ``inner``, which the function of ``outer`` (a method, or a
    ``functools.partial``) hands the call to, bound with the leading arguments ``outer``
    binds. Raises :class:`BoundArgumentKept` where binding them to ``inner`` assumes
    what cannot be read.

    The walk binds the arguments a method or a ``functools.partial`` holds to what its
    function hands the call to, as if every wrapper on the way handed them on. A wrapper
    that takes ``(*args, **kwargs)`` does, and so does one that takes the argument under
    the name the function under it has for it. A wrapper that takes it under ANOTHER
    name need not: ``def search(self, *args, **kwargs): return free_search(*args,
    **kwargs)`` under ``functools.wraps(free_search)`` keeps the object. Every argument
    bound after a kept one then fills the parameter before its own, so the parameter
    left without a value is the LAST one the bound arguments would fill, not the one the
    kept argument would: with only the object bound, the first parameter of
    ``free_search``; under ``functools.partial(tools.search, "acme")``, its second,
    while ``"acme"`` fills the first. Where that last parameter has no default the call
    fails. Where it has one, nothing fails, the signature the tool shows does not have
    the parameter, and no policy judged the value.

    So two things go with the bound arguments down the walk (``walk.taken``, by the
    identity of each callable the walk binds), through every callable that takes them in
    ``*args`` and names nothing for them (a ``(*args, **kwargs)`` wrapper, a class-based
    decorator whose ``__call__`` passes everything on, a cache, a ``functools.partial``,
    however many there are). One is the name each was last taken under. The other is
    the doubt: the name a wrapper took one under, where the function it hands the call
    to has another name for the parameter that argument would fill. Refused: bound
    arguments under a doubt, where the last parameter they would fill on ``inner`` has a
    default and is not given by name: the tool does not declare it, or a wrapper on the
    way that passes the call on takes it by name itself (:func:`_refuse_kept_by_name`,
    at the end of the walk). The free function of the example is refused bare, under any
    number of such layers, and with more arguments bound after the object.

    Refused with them, whatever its default: bound arguments under a doubt, where the
    last parameter they would fill is one the tool declares and ``inner`` itself passes
    the call on (``*args`` or ``**kwargs``). There ``inner`` is the wrapper that takes
    the name. A facade that hands its argument on BY KEYWORD (``return
    capped(*args, tenant=self.tenant, **kwargs)`` over ``def wrapper(limit=100, *args,
    **kwargs)``) leaves that parameter free, the judged keyword of that name stops at
    ``inner``, and whether ``inner`` hands it on cannot be read. The walk binds the
    argument to the parameter and so reads ``inner`` as taking nothing by that name,
    which is why the rule at the end of the walk does not see it.

    The doubt goes two ways. It goes to the arguments bound AFTER the doubted one (a
    ``functools.partial`` of the method): those are the ones a kept argument moves. And
    it stays with the same arguments on each callable further in, the ones bound after
    included, until one settles it: a callable that has a parameter for each of them,
    the last of which has no default and is not declared by the tool, so a call that
    had kept an argument fails there for want of it, and a call that runs kept none. A
    parameter the tool declares settles nothing: it is given by name, and, on a
    callable that passes the call on to nothing, judged whatever a wrapper above did
    with the bound arguments (Python 3.9 shows the free function's own parameters for a
    decorator over the bound method, where 3.10 and later show them less the first).

    Refused although it runs correctly, because it has the very signature of the
    example and the walk reads parameters, not bodies: a wrapper that hands the argument
    on to a function that names it otherwise (``def wrapper(client, *args, **kwargs):
    return f(client, *args, **kwargs)`` over ``def f(session=None, limit=10)``, bound by
    a ``functools.partial``); a facade that passes a value of its own in its place
    (``return free_search(self.tenant, *args, **kwargs)``, the first parameter
    defaulted); a ``functools.partial`` that binds a defaulted parameter of a method
    whose decorator names the receiver otherwise than the method does; and bound
    arguments under a doubt that are more than ``inner`` has parameters for, where its
    last one has a default. A plain function that names every parameter the tool takes,
    and calls the facade, is read to the end.

    An ordinary bound method, decorated or not, is not refused: its function hands the
    call to nothing, or no wrapper names the object, or the function under the wrapper
    takes the object under the same name or as a parameter without a default
    (``self``). So ``def wrapper(self, *args, **kwargs)`` over a method is read as
    before, and so is a facade that hands its receiver on to a method.

    Not found, by design: a wrapper that takes the bound argument in ``*args`` and drops
    it in its body, and one that takes it under the same name and does not hand it on."""
    rebound: Any
    if inspect.ismethod(outer):
        level, count, rebound = outer.__func__, 1, types.MethodType(inner, outer.__self__)
    else:
        level, count = outer.func, _bound_count(outer) or 0
        rebound = functools.partial(inner, *outer.args, **outer.keywords)
    names, _, still = _taken_under(walk, outer, count)
    taken = list(names)
    passes_on = False
    try:
        inner_own = own_parameters(inner)
        above, below = _leading(own_parameters(level)), _leading(inner_own)
        passes_on = _read_level(inner_own)[1]
    except Exception:  # noqa: BLE001, what cannot be read here is read, or refused, bound
        above, below = [], []
    for index in range(min(count, len(above))):
        taken[index] = above[index].name  # the name the nearest wrapper has for it
    # what ``rebound`` binds: these arguments, after those of a ``functools.partial`` it
    # was merged with (``partial(partial(f, a), b)`` is ``partial(f, a, b)``)
    merged = len(_partial_args(rebound)) - count if type(rebound) is _PARTIAL else 0
    before, ahead, _ = _taken_under(walk, inner, merged)
    filled = below[:count]  # the parameters these arguments would fill on ``inner``
    here = [name for name, to in zip(taken, filled) if name not in (None, to.name)]
    doubt = here[0] if here else ahead  # of these arguments, or of one bound ahead
    still = doubt or still
    if still is not None and filled:
        last, declared = filled[-1], _declared(walk)
        if declared is not None and last.name in declared:
            if passes_on:
                # ``inner`` takes the name itself and passes the call on: where the
                # bound argument did not reach it by position (kept above, or handed
                # on by keyword), the judged keyword stops here, and whether ``inner``
                # hands it on cannot be read
                raise BoundArgumentKept(_UNHANDED.format(taken=still, filled=last.name))
            walk.given.append((still, last))  # given by name, unless a wrapper keeps it
        elif last.default is not last.empty:
            raise BoundArgumentKept(_UNHANDED.format(taken=still, filled=last.name))
        elif len(filled) == count:
            still = None  # settled: had one been kept, the call would fail here
    walk.taken.append((rebound, before + tuple(taken), doubt, still))
    return rebound


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
        return [_rebound(func, inner, walk) for inner in inners]
    if isinstance(func, functools.partial):
        count = _bound_count(func)
        under = None if bound is None or count is None else bound + count
        found = [_rebound(func, inner, walk) for inner in _next_callables(func.func, walk, under)]
        # a partial can carry a ``__wrapped__`` of its own
        # (``functools.wraps(f)(functools.partial(helper, f))``), which is what
        # ``inspect.signature`` shows for it: followed like any other, as it is named
        wrapped = getattr(func, "__wrapped__", None)
        if wrapped is not None:
            walk.read.append(func)
            if walk.checked and bound is not None:
                walk.held.extend((value, bound) for value in _held(func, wrapped))
            found.append(wrapped)
        return found
    found = []
    wrapped = getattr(func, "__wrapped__", None)
    _refuse_call_time_dispatch(func, wrapped)
    walk.read.append(func)
    if wrapped is not None:
        if walk.checked and bound is not None:
            walk.held.extend((value, bound) for value in _held(func, wrapped))
        found.append(wrapped)
        if isinstance(func, _PASS_THROUGH):
            # a cache named again for another function (``functools.wraps(f)`` over
            # ``functools.lru_cache`` over a wrapper): the wrapper it calls is the one
            # that stands for ``f``. Its parameters are the cache's own level
            # (:func:`own_parameters`); what it holds and what it hands the call to are
            # read here, as for a wrapper that names ``f`` itself
            called = _cache_target(func)
            if called is not wrapped:
                if type(called) is types.FunctionType and walk.checked and bound is not None:
                    walk.held.extend((value, bound) for value in _held(called, wrapped))
                found.extend(_next_callables(called, walk, bound))
                if not inspect.isfunction(wrapped):
                    # a cache NAMED for something that is not a plain function is read,
                    # under a method, by the names alone (:func:`own_parameters`): the
                    # wrapper it calls is then read here, as a callable of its own
                    found.append(called)
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
    function, every ``__wrapped__`` (a function's, a callable object's, a
    ``functools.lru_cache`` wrapper's when that is the function it calls, and the one a
    ``functools.partial`` itself carries) and a callable object's ``__call__``. ``skip`` names
    the parameters the framework injects itself, as for :func:`unsupported_signature`.
    Raises :class:`DeclaredSignature`, :class:`HiddenWrapper` (a ``ValueError``: a callable
    on the way holds a second ``functools.wraps`` wrapper of a callable the walk reads,
    with a parameter of its own), :class:`DispatchedAtCallTime` (a ``ValueError``: a
    ``functools.singledispatch`` function or ``singledispatchmethod`` on the way can run
    something other than its base implementation), :class:`PartialMethodSignature` (a
    ``TypeError``: a callable on the way is read by ``inspect.signature`` as a
    ``functools.partialmethod``), :class:`BoundArgumentKept` (a ``ValueError``: a bound
    argument is taken by a wrapper under another name than the parameter it would fill
    on a function that wrapper reaches, and the last parameter the bound arguments would
    fill there has a default and is not given by name; a wrapper of that signature that
    hands the argument on is refused too), ``TypeError`` or ``ValueError`` for a callable
    it cannot read, which a gate turns into a refusal.

    :class:`UnreadCache` is a :class:`HiddenWrapper`. The ``ValueError`` for a callable
    object whose class defines ``__get__`` (``inspect.signature`` shows no signature for
    one) says that cause and what to do.

    Each of these refusals is exact and narrow: it finds one shape and states the rest
    as limits (:func:`_refuse_hidden_wrapper`, :func:`_refuse_call_time_dispatch`,
    :func:`_refuse_partialmethod`, :func:`_rebound`, :func:`_cache_calls`). The two that
    read what a callable HOLDS (a hidden wrapper, a dispatcher) read it without running
    any of it. A wrapper a callable holds is judged by the rule and the code that judge
    the levels returned here (:func:`_argument_of_its_own`), so it gets the verdict it
    would get read directly. The callables on the way themselves are read as before, by
    attribute lookup.
    """
    return _reached(func, skip, None)


def _reached(func: Any, skip: Any, declared: Any) -> List[Any]:
    """:func:`reached_parameters`, for a caller that has read what ``func`` declares."""
    walk = _walk(func, declared)
    found = walk.found
    found.append(own_parameters(func))
    pending = _next_callables(func, walk)
    for _ in range(_DEPTH):
        if not pending:
            hidden = _refuse_hidden_wrapper(walk, skip)
            _refuse_kept_by_name(walk, [_read_level(level) for level in found] + hidden)
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

    It does not call itself: it keeps its place in each list and object it is copying,
    and calls no function written in Python for a value, with one exception it has
    always had: telling a value's type compares that type with each supported one, and
    a class whose METACLASS defines ``__eq__`` answers that comparison itself (one that
    answers True for ``str`` is handed back as it is, not copied; a model's arguments
    are parsed JSON and are never of such a class). So data nested deeper than the
    stack has room for is copied all the same (the version that called itself once for
    every value raised ``RecursionError`` there, at a depth that moved with the caller's
    own), and what a copy costs does not move with where it is called from (measured on
    Python 3.12, which keeps frames in blocks, that version cost up to thirty times
    more from some depths of the caller's stack than from the next; the changelog has
    the figures). Every value is tested as before, in the same order, so the copy and
    each ``TypeError`` are the same.
    ``ancestors`` are the ``id()`` of containers ``value`` is to be taken as inside of:
    meeting one of them again is a cycle.
    """
    made: List[Any] = []
    # for each list or object being copied: what is left of it, its copy, whether its
    # members are named, and its ``id()`` (0 for the holder of the copy itself)
    levels: List[Tuple[Any, Any, bool, int]] = [(iter((value,)), made, False, 0)]
    above: Set[int] = set()  # the ``id()`` of every list and object being copied
    scalars, containers = (str, int, bool, type(None)), (dict, list)
    while levels:
        members, target, named, identity = levels[-1]
        for each in members:
            if named:
                name, each = each
            if type(each) in scalars:
                pass
            elif type(each) is float:
                if not math.isfinite(each):
                    raise TypeError("non-finite numbers are not JSON data")
            elif isinstance(each, Enum):
                raise TypeError(
                    "Enum arguments are unsupported; declare data-only parameters "
                    "and resolve application types inside the tool body"
                )
            elif type(each) not in containers:
                raise TypeError("arguments must be data-only")
            else:
                if id(each) in ancestors or id(each) in above:
                    raise TypeError("cyclic arguments are not JSON data")
                if not above:
                    ancestors = (*ancestors,)  # read once, as before, at the first of them
                inner: Any
                if type(each) is list:
                    inner = []
                    levels.append((iter(each), inner, False, id(each)))
                else:
                    for key in each:
                        if type(key) is not str:
                            raise TypeError("argument keys must be strings")
                    inner = {}
                    levels.append((iter(each.items()), inner, True, id(each)))
                above.add(id(each))
                if named:
                    target[name] = inner
                else:
                    target.append(inner)
                break  # its copy is in place: its own members are copied next
            if named:
                target[name] = each
            else:
                target.append(each)
        else:
            levels.pop()
            above.discard(identity)
    return made[0]
