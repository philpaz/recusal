"""``recusal.invocation``: the data-only invocation, defined once.

The framework gates under ``examples/`` used to carry their own copies of these helpers.
The last block here is the lock that keeps it one definition: no file under ``examples/``
may define a function by one of these names again.
"""

import ast
import enum
import pathlib
import sys
import typing
from typing import Any, Dict, List, Optional

import pytest

from recusal.invocation import (
    SCALARS,
    DeclaredSignature,
    DispatchedAtCallTime,
    HiddenWrapper,
    detached,
    json_data,
    reached_parameters,
    supported_annotation,
    unsupported_signature,
)

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


# --- supported_annotation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation",
    [
        str,
        int,
        float,
        bool,
        Optional[int],
        List[str],
        Dict[str, int],
        List[Optional[int]],
        Dict[str, List[float]],
        Optional[Dict[str, List[bool]]],
    ],
)
def test_the_supported_set(annotation):
    assert supported_annotation(annotation) is True


class _Color(enum.Enum):
    RED = "red"


@pytest.mark.parametrize(
    "annotation",
    [
        Any,
        None,
        type(None),
        list,
        dict,
        bytes,
        object,
        _Color,
        Dict[int, str],
        typing.Union[int, str],
        typing.Tuple[int, ...],
        List[Any],
        Dict[str, Any],
        typing.Set[int],
        "str",
    ],
)
def test_everything_else_is_unsupported(annotation):
    assert supported_annotation(annotation) is False


@pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 604 unions need Python 3.10")
def test_pep_604_unions_read_as_optional():
    assert supported_annotation(eval("int | None")) is True
    assert supported_annotation(eval("list[str] | None")) is True
    assert supported_annotation(eval("int | str")) is False


def test_the_scalar_set_is_the_four_json_scalars():
    assert SCALARS == (str, int, float, bool)


# --- json_data ----------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "a", 1, 1.5, True, [], {}, [1, [2]], {"k": {"j": [None]}}])
def test_json_values(value):
    assert json_data(value) is True


@pytest.mark.parametrize(
    "value", [b"x", (1, 2), {1, 2}, object(), {1: "a"}, {"k": (1,)}, [object()], _Color.RED]
)
def test_values_outside_the_json_domain(value):
    assert json_data(value) is False


class _SneakyStr(str):
    def __deepcopy__(self, memo):
        return "SELECT 1"


class _SneakyList(list):
    pass


@pytest.mark.parametrize(
    "value",
    [
        _SneakyStr("DROP TABLE t"),
        _SneakyList([1]),
        ["ok", _SneakyStr("x")],
        {"k": _SneakyList()},
        float("inf"),
        float("nan"),
        [float("-inf")],
    ],
    ids=["str-subclass", "list-subclass", "nested-str", "nested-list", "inf", "nan", "nested-inf"],
)
def test_a_subclass_or_a_non_finite_float_is_not_json_data(value):
    """A subclass of a JSON type can copy, compare or serialise as something else, so the
    copy a policy judges could differ from the value the tool gets; a non-finite float is
    not JSON and is not equal to itself."""
    assert json_data(value) is False


def test_an_annotation_that_only_compares_equal_to_a_scalar_is_unsupported():
    class Lookalike(type):
        def __eq__(cls, other):
            return other is str

        __hash__ = type.__hash__

    class Cmd(metaclass=Lookalike):
        pass

    assert Cmd == str  # noqa: E721, the equality the annotation check must not trust
    assert supported_annotation(Cmd) is False
    assert supported_annotation(Optional[Cmd]) is False


# --- unsupported_signature ----------------------------------------------------------------


def test_a_fully_annotated_callable_can_be_judged():
    def tool(a: str, b: int = 3, c: Optional[List[str]] = None) -> str:
        return a

    assert unsupported_signature(tool) is None


def test_each_refusal_names_the_parameter_and_the_reason():
    def variadic(a: str, *rest: str) -> None: ...

    def keywords(a: str, **rest: str) -> None: ...

    def bare(a) -> None: ...

    def bad_default(a: str = None, b: typing.List[str] = ("x",)) -> None: ...  # type: ignore[assignment]

    assert unsupported_signature(variadic) == (
        "`rest` is variadic; every argument the tool receives must be judged by name"
    )
    assert unsupported_signature(keywords) == (
        "`rest` is variadic; every argument the tool receives must be judged by name"
    )
    assert unsupported_signature(bare) == "`a` is not annotated with a supported data type"
    assert unsupported_signature(bad_default) == "`b` has a default outside the JSON domain"


def test_a_positional_only_parameter_is_refused():
    namespace: Dict[str, Any] = {}
    exec("def tool(a: str, /, b: int = 1) -> None: ...", namespace)
    assert unsupported_signature(namespace["tool"]) == (
        "`a` is positional-only and cannot be judged by name"
    )


def test_skip_names_the_parameters_the_framework_injects_and_only_those():
    def tool(ctx, a: str, other) -> None: ...

    assert unsupported_signature(tool, ("ctx", "other")) is None
    assert unsupported_signature(tool, ("ctx",)) == (
        "`other` is not annotated with a supported data type"
    )


def test_an_uninspectable_callable_is_refused_not_raised():
    def tool(a: "NoSuchName") -> None: ...  # noqa: F821

    assert unsupported_signature(tool) == "its signature cannot be inspected"
    assert unsupported_signature(42) == "its signature cannot be inspected"  # type: ignore[arg-type]


# --- every callable a call reaches ---------------------------------------------------------

_HIDDEN = (
    "`force` is a parameter of a function the call reaches that the tool does not declare, "
    "so it would run with a value no policy judged"
)


def _traced(function):
    """A tracer as instrumentation writes one: ``functools.wraps``, everything passed on."""
    import functools

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def test_a_wrapper_that_adds_a_parameter_of_its_own_is_refused():
    """``inspect.signature`` reads a ``functools.wraps`` wrapper as the function it wraps,
    so a framework declares the wrapped parameters; the wrapper's own ``force=True`` would
    run with its default, never judged."""
    import functools

    def wipe(table: str) -> str:
        return table

    @functools.wraps(wipe)
    def forcing(*args, force: bool = True, **kwargs):
        return wipe(*args, **kwargs)

    assert list(inspect_signature(forcing)) == ["table"]
    assert unsupported_signature(forcing) == _HIDDEN
    # inside another tracer the outer wrapper shows nothing of its own: the walk must follow
    # ``__wrapped__`` to reach the wrapper that adds ``force``
    outer = _traced(forcing)
    assert list(inspect_signature(outer)) == ["table"]
    assert unsupported_signature(outer) == _HIDDEN


def test_a_wrapper_that_takes_a_parameter_the_function_also_takes_is_refused():
    """Review round 1: ``wrapper(*args, sandbox=False, **kwargs)`` around
    ``transfer(amount, sandbox=False)`` keeps ``sandbox`` for itself, so the policy judged
    ``sandbox=True`` and the transfer ran with ``False``."""
    import functools

    def transfer(amount: int, sandbox: bool = False) -> str:
        return "sent"

    @functools.wraps(transfer)
    def wrapper(*args, sandbox: bool = False, **kwargs):
        return transfer(*args, **kwargs)

    assert list(inspect_signature(wrapper)) == ["amount", "sandbox"]
    assert unsupported_signature(wrapper) == (
        "`sandbox` is taken by a wrapper and by the function it wraps, which can then run "
        "with a value no policy judged"
    )

    # a name the framework injects (``skip``) may be taken by both
    @functools.wraps(transfer)
    def with_context(*args, tool_context=None, **kwargs):
        return transfer(*args, **kwargs)

    def transfer_in_context(amount: int, tool_context=None) -> str:
        return "sent"

    with_context.__wrapped__ = transfer_in_context
    assert unsupported_signature(with_context, skip=("tool_context",)) is None


def test_every_place_a_wrapper_can_stand_is_followed():
    """A wrapper that adds ``force`` hides it wherever it stands: behind a
    ``functools.partial``, and in a class-based decorator (``functools.update_wrapper``),
    whose ``__call__`` is what runs while its signature reads as the wrapped function's."""
    import functools

    def wipe(table: str) -> str:
        return table

    @functools.wraps(wipe)
    def forcing(*args, force: bool = True, **kwargs):
        return wipe(*args, **kwargs)

    class Decorator:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, table: str, force: bool = True) -> str:
            return table

    assert list(inspect_signature(Decorator(wipe))) == ["table"]
    assert unsupported_signature(Decorator(wipe)) == _HIDDEN

    # the walk itself reaches ``force`` behind a partial; ``unsupported_signature`` refuses a
    # partial before it gets there, as before, because its annotations cannot be read
    # (older Pythons raise, 3.14 returns none, so the reason differs), so the walk is
    # pinned directly
    reached = reached_parameters(functools.partial(forcing))
    assert list(inspect_signature(functools.partial(forcing))) == ["table"]
    assert any("force" in level for level in reached)
    assert unsupported_signature(functools.partial(forcing)) is not None


def test_ordinary_wrappers_are_read_through_and_accepted():
    import functools

    def search(query: str, limit: int = 5) -> str:
        return query

    def with_client(client: object, query: str) -> str:
        return query

    class Holder:
        @functools.lru_cache(maxsize=None)  # noqa: B019, the tool style under test
        def lookup(self, query: str) -> str:
            return query

    for func in (
        _traced(search),
        functools.lru_cache(maxsize=None)(search),
        Holder().lookup,
    ):
        assert unsupported_signature(func) is None, func

    # a partial that binds a dependency is read with that binding kept: ``client`` is bound,
    # so no callable on the way shows it as a parameter the call must supply
    reached = reached_parameters(functools.partial(_traced(with_client), object()))
    assert all("client" not in level for level in reached)

    # an injected parameter named in ``skip`` may be added by a wrapper too
    @functools.wraps(search)
    def with_context(*args, tool_context=None, **kwargs):
        return search(*args, **kwargs)

    assert unsupported_signature(with_context, skip=("tool_context",)) is None
    assert unsupported_signature(with_context) is not None


def test_a_declared_signature_anywhere_on_the_way_is_refused():
    import functools
    import inspect

    def shown(table: str) -> str:
        return table

    def purge(table: str, force: bool = True) -> str:
        return table

    purge.__signature__ = inspect.signature(shown)  # type: ignore[attr-defined]
    declared = "it declares a signature that is not its own"
    assert unsupported_signature(purge) == declared
    for func in (purge, functools.partial(purge)):
        with pytest.raises(DeclaredSignature):
            reached_parameters(func)


def test_a_call_that_is_not_a_plain_method_and_a_class_are_refused():
    import functools

    def impl(force=True, *, table: str) -> str:
        return table

    class StaticCall:
        __call__ = staticmethod(impl)

    class PartialCall:
        __call__ = functools.partial(impl)

    class Runner:
        def __init__(self, table: str) -> None:
            self.table = table

    for func in (StaticCall(), PartialCall()):
        # refused before the walk on every Python, for a reason that differs by version
        # (``get_type_hints`` raises on older ones and returns none on 3.14)
        assert unsupported_signature(func) is not None, func
        # the walk refuses them itself, whatever refuses them first
        with pytest.raises(TypeError, match="is not a plain method"):
            reached_parameters(func)
    # a class is refused whichever check reaches it first, and the walk refuses it itself
    assert unsupported_signature(Runner) is not None
    with pytest.raises(TypeError):
        reached_parameters(Runner)


# --- a wrapper the walk cannot follow ------------------------------------------------------

_UNFOLLOWED = (
    "a wrapper holds another wrapper of the function it wraps, which the walk does not follow"
)


def _with_timeout(default):
    """A decorator whose wrapper keeps ``timeout`` for itself."""
    import functools

    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, timeout=default, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    return decorate


def _forcing(function):
    """A decorator whose wrapper adds ``force``, filled from its own default."""
    import functools

    @functools.wraps(function)
    def wrapper(*args, force=True, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def _composed(inner):
    """A decorator that applies ``inner`` inside itself: its wrapper calls the result, held
    only in its closure, while ``functools.wraps`` names the undecorated function."""
    import functools

    def decorate(function):
        guarded = inner(function)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return guarded(*args, **kwargs)

        return wrapper

    return decorate


def _levels(func):
    return [list(level) for level in reached_parameters(func)]


def test_a_wrapper_applied_inside_another_keeps_an_argument_and_is_refused():
    """Three adapter reviews, one gap: ``resilient`` applies ``with_timeout`` inside itself
    and calls the result, but its ``__wrapped__`` is the undecorated function, so the walk
    never read the wrapper in between. The policy judged ``timeout=5``; that wrapper kept
    it and the function ran with its own default."""

    def fetch(url: str, timeout: int = 999999) -> int:
        return timeout

    tool = _composed(_with_timeout(30))(fetch)
    assert tool.__wrapped__ is fetch
    assert list(inspect_signature(tool)) == ["url", "timeout"]
    assert tool("u", timeout=5) == 999999  # the value judged is not the value run
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    assert str(refused.value) == _UNFOLLOWED  # names no function, parameter or value
    assert unsupported_signature(tool) == _UNFOLLOWED


def test_the_refusal_is_a_value_error_so_a_gate_that_catches_one_refuses():
    assert issubclass(HiddenWrapper, ValueError)
    assert not issubclass(HiddenWrapper, DeclaredSignature)


def test_a_default_of_a_wrapper_applied_inside_another_is_refused():
    """The hidden wrapper's own ``force=True`` ran unseen the same way."""

    def wipe(table: str) -> str:
        return table

    tool = _composed(_forcing)(wipe)
    assert list(inspect_signature(tool)) == ["table"]
    with pytest.raises(HiddenWrapper):
        reached_parameters(tool)
    assert unsupported_signature(tool) == _UNFOLLOWED


def test_a_hidden_wrapper_is_refused_wherever_the_wrapper_that_holds_it_stands():
    """Further in behind a tracer, behind a ``functools.partial``, as a method, and as a
    coroutine function."""
    import functools

    def fetch(url: str, timeout: int = 999999) -> int:
        return timeout

    tool = _composed(_with_timeout(30))(fetch)

    class Holder:
        @_composed(_with_timeout(30))
        def fetch(self, url: str, timeout: int = 999999) -> int:
            return timeout

    def composed_async(function):
        @functools.wraps(function)
        async def guarded(*args, timeout=30, **kwargs):
            return await function(*args, **kwargs)

        @functools.wraps(function)
        async def wrapper(*args, **kwargs):
            return await guarded(*args, **kwargs)

        return wrapper

    @composed_async
    async def fetch_async(url: str, timeout: int = 999999) -> int:
        return timeout

    for func in (
        _traced(tool),
        _traced(_traced(tool)),
        functools.partial(tool),
        functools.partial(_traced(tool), "u"),
        Holder().fetch,
        fetch_async,
    ):
        with pytest.raises(HiddenWrapper):
            reached_parameters(func)
    for func in (_traced(tool), Holder().fetch, fetch_async):
        assert unsupported_signature(func) == _UNFOLLOWED, func


def test_a_hidden_wrapper_in_an_instance_attribute_is_refused():
    """A class-based decorator holds the composed wrapper on the instance, not in a cell."""
    import functools

    def fetch(url: str, timeout: int = 999999) -> int:
        return timeout

    class Resilient:
        def __init__(self, function):
            functools.update_wrapper(self, function)
            self.guarded = _with_timeout(30)(function)

        def __call__(self, *args, **kwargs):
            return self.guarded(*args, **kwargs)

    tool = Resilient(fetch)
    assert tool("u", timeout=5) == 999999
    for func in (tool, _traced(tool), functools.partial(tool)):
        with pytest.raises(HiddenWrapper):
            reached_parameters(func)


def test_a_hidden_wrapper_is_found_at_any_depth_of_either_chain():
    """What is held may wrap the function through other wrappers, and may wrap a function
    further in than the one ``__wrapped__`` names."""
    import functools
    import inspect

    def fetch(url: str, timeout: int = 999999) -> int:
        return timeout

    # held: tracer -> timeout wrapper -> the function
    through = _composed(lambda function: _traced(_with_timeout(30)(function)))(fetch)
    assert through("u", timeout=5) == 999999

    # held: a wrapper of the innermost function, while ``__wrapped__`` names a tracer of it
    def around_the_innermost(function):
        guarded = _with_timeout(30)(inspect.unwrap(function))

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return guarded(*args, **kwargs)

        return wrapper

    innermost = around_the_innermost(_traced(_traced(fetch)))
    assert innermost("u", timeout=5) == 999999
    for func in (through, innermost):
        with pytest.raises(HiddenWrapper):
            reached_parameters(func)


def test_ordinary_decorators_are_walked_as_before():
    """The refusal is for a held wrapper of the same function only. What a decorator
    ordinarily holds (the function, configuration, helpers that wrap nothing) is not one."""
    import contextlib
    import functools
    import logging
    import time

    def search(query: str, limit: int = 5) -> str:
        return query

    forward, own = ["args", "kwargs"], ["query", "limit"]

    def retrying(attempts, message="retrying"):
        def decorate(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                for _ in range(attempts):
                    return function(*args, **kwargs)
                return message

            return wrapper

        return decorate

    def observed(on_error):
        log, clock = logging.getLogger(__name__).info, time.monotonic

        def decorate(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                started = clock()
                try:
                    return function(*args, **kwargs)
                except Exception as error:  # noqa: BLE001, the decorator style under test
                    on_error(error)
                finally:
                    log("%s", clock() - started)

            return wrapper

        return decorate

    def traced_async(function):
        @functools.wraps(function)
        async def wrapper(*args, **kwargs):
            return await function(*args, **kwargs)

        return wrapper

    @traced_async
    async def search_async(query: str, limit: int = 5) -> str:
        return query

    class Decorator:
        def __init__(self, function):
            functools.update_wrapper(self, function)
            self.function = function

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    class Holder:
        @_traced
        def search(self, query: str, limit: int = 5) -> str:
            return query

    @contextlib.contextmanager
    def span():
        yield

    # with only its base implementation, which is what then runs
    dispatched = functools.singledispatch(search)

    for func, expected in (
        (_traced(search), [forward, own]),
        (_traced(_traced(_traced(search))), [forward, forward, forward, own]),
        (retrying(3)(search), [forward, own]),
        (observed(print)(search), [forward, own]),
        (functools.lru_cache(maxsize=None)(search), [own, own]),
        (functools.cache(search), [own, own]),
        (_traced(functools.lru_cache(maxsize=None)(search)), [forward, own, own]),
        (functools.partial(_traced(search), "q"), [forward, ["limit"]]),
        (dispatched, [["args", "kw"], own]),
        (span()(search), [["args", "kwds"], own]),
        (Decorator(search), [forward, forward, own]),
        (Decorator(_traced(search)), [forward, forward, forward, own]),
        (Holder().search, [forward, own]),
        (search_async, [forward, own]),
    ):
        assert _levels(func) == expected, func
    # ``contextlib.contextmanager`` is itself a ``functools.wraps`` decorator
    assert _levels(span) == [["args", "kwds"], []]
    for func in (_traced(search), retrying(3)(search), observed(print)(search), search_async):
        assert unsupported_signature(func) is None, func


def test_what_a_wrapper_holds_that_is_not_a_hidden_wrapper_is_not_refused():
    """Each exemption of the check: the wrapper itself, a wrapper the walk reaches anyway,
    a value that is not callable, an empty cell, and attributes that cannot be read."""
    import functools
    import types

    def search(query: str, limit: int = 5) -> str:
        return query

    forward, own = ["args", "kwargs"], ["query", "limit"]

    # a wrapper that calls itself again (a retry) holds itself, and it wraps the function;
    # its own parameter is read by the walk, as a level, not refused as a hidden wrapper's
    def again(function):
        @functools.wraps(function)
        def wrapper(*args, attempt=0, **kwargs):
            return wrapper if not args else function(*args, **kwargs)

        return wrapper

    assert _levels(again(search)) == [["args", "attempt", "kwargs"], own]

    # a wrapper may hold the function it wraps and what that wraps in turn: the walk reads
    # both through ``__wrapped__``
    def unwrapping(function):
        inner = function.__wrapped__

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return (function if args else inner)(*args, **kwargs)

        return wrapper

    assert _levels(unwrapping(_traced(_traced(search)))) == [forward, forward, forward, own]

    # a record that names the function but cannot be called is not a wrapper
    def recorded(function):
        record = types.SimpleNamespace(__wrapped__=function)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs), record

        return wrapper

    assert _levels(recorded(search)) == [forward, own]

    # a cell that was never filled
    def unfilled(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs), later

        if not function:
            later = None
        return wrapper

    assert _levels(unfilled(search)) == [forward, own]

    # a held callable whose attributes raise on being read
    class Opaque:
        def __call__(self):
            return None

        def __getattr__(self, name):
            raise RuntimeError(name)

    def opaquely(function, helper=Opaque()):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs), helper

        return wrapper

    assert _levels(opaquely(search)) == [forward, own]

    # a callable object with no instance attributes to read
    class Slotted:
        __slots__ = ("__wrapped__",)

        def __init__(self, function):
            self.__wrapped__ = function

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    assert _levels(Slotted(search)) == [forward, forward, own]


def test_a_cycle_of_wrapped_ends_and_is_never_a_hidden_wrapper():
    import functools

    from recusal import invocation

    def search(query: str, limit: int = 5) -> str:
        return query

    def first():
        return None

    def second():
        return None

    first.__wrapped__, second.__wrapped__ = second, first  # type: ignore[attr-defined]
    # what a callable names is read as deep as the walk goes and no further, cycle or not
    assert len(invocation._wrapped_chain(first)) == 32

    def holding(function, helper=first):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs), helper

        return wrapper

    assert _levels(holding(search)) == [["args", "kwargs"], ["query", "limit"]]
    # a callable that names itself is refused as too deep, as before, not as a hidden wrapper
    looped = _traced(search)
    looped.__wrapped__ = looped
    with pytest.raises(ValueError, match="too many wrappers") as refused:
        reached_parameters(looped)
    assert not isinstance(refused.value, HiddenWrapper)


# --- a function chosen at call time --------------------------------------------------------

_AT_CALL_TIME = (
    "the function that runs is chosen at call time by the type of an argument, which the "
    "walk cannot read"
)


def _dispatching():
    """A single-dispatch function whose ``int`` implementation has a default of its own."""
    import functools

    @functools.singledispatch
    def lookup(key: str, deep: bool = False) -> str:
        return "base"

    @lookup.register(int)
    def lookup_int(key: int, deep: bool = False, skip_review: bool = True) -> str:
        return f"int, skip_review={skip_review}"

    return lookup


def test_a_single_dispatch_function_with_a_registered_implementation_is_refused():
    """``functools.singledispatch`` names its base function as ``__wrapped__``, so the walk
    read the base only. The implementation registered for ``int`` ran in its place with its
    own ``skip_review=True``, which no policy judged."""
    lookup = _dispatching()
    assert list(inspect_signature(lookup)) == ["key", "deep"]
    assert lookup(1) == "int, skip_review=True"  # a parameter nothing declared
    with pytest.raises(DispatchedAtCallTime) as refused:
        reached_parameters(lookup)
    assert str(refused.value) == _AT_CALL_TIME  # names no function, type or parameter
    assert unsupported_signature(lookup) == _AT_CALL_TIME


def test_the_dispatch_refusal_is_a_value_error_so_a_gate_that_catches_one_refuses():
    assert issubclass(DispatchedAtCallTime, ValueError)
    assert not issubclass(DispatchedAtCallTime, HiddenWrapper)


def test_a_dispatching_function_is_refused_wherever_it_stands_on_the_walk():
    """Behind tracers, a ``functools.partial``, ``functools.lru_cache``, as a method, and
    as a coroutine function."""
    import functools

    lookup = _dispatching()

    class Holder:
        find = lookup

    @functools.singledispatch
    async def lookup_async(key: str, deep: bool = False) -> str:
        return "base"

    @lookup_async.register(int)
    async def lookup_async_int(key: int, deep: bool = False, skip_review: bool = True) -> str:
        return "int"

    for func in (
        _traced(lookup),
        _traced(_traced(lookup)),
        functools.partial(lookup),
        functools.partial(_traced(lookup), 1),
        functools.lru_cache(maxsize=None)(lookup),
        Holder().find,
        lookup_async,
        _traced(lookup_async),
    ):
        with pytest.raises(DispatchedAtCallTime):
            reached_parameters(func)
    for func in (_traced(lookup), lookup_async):
        assert unsupported_signature(func) == _AT_CALL_TIME, func


def test_a_single_dispatch_method_with_a_registered_implementation_is_refused():
    """``functools.singledispatchmethod`` hands out a new function at each access. It has
    no registry of its own; it holds the descriptor, whose dispatcher has one."""
    import functools

    class Holder:
        @functools.singledispatchmethod
        def lookup(self, key: str, deep: bool = False) -> str:
            return "base"

        @lookup.register(int)
        def lookup_int(self, key: int, deep: bool = False, skip_review: bool = True) -> str:
            return f"int, skip_review={skip_review}"

        @functools.singledispatchmethod
        def plain(self, key: str, deep: bool = False) -> str:
            return "base"

    assert Holder().lookup(1) == "int, skip_review=True"
    assert not hasattr(Holder().lookup, "registry")
    for func in (
        Holder().lookup,
        Holder.lookup,
        _traced(Holder().lookup),
        functools.partial(Holder().lookup),
    ):
        with pytest.raises(DispatchedAtCallTime):
            reached_parameters(func)
    assert unsupported_signature(Holder().lookup, skip=("self",)) == _AT_CALL_TIME
    # with only the base registered the base is what runs, and it is walked as before
    assert _levels(Holder().plain) == [["args", "kwargs"], ["self", "key", "deep"]]
    assert unsupported_signature(Holder().plain, skip=("self",)) is None

    # a callable object that holds the descriptor in an instance attribute, not a cell
    class Access:
        def __init__(self, descriptor, function):
            functools.update_wrapper(self, function)
            self.descriptor = descriptor

        def __call__(self, *args, **kwargs):
            return self.descriptor.dispatcher.dispatch(type(args[0]))(*args, **kwargs)

    def base(key: str, deep: bool = False) -> str:
        return "base"

    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(Access(vars(Holder)["lookup"], base))
    assert _levels(Access(vars(Holder)["plain"], base)) == [
        ["args", "kwargs"],
        ["args", "kwargs"],
        ["key", "deep"],
    ]


def test_only_a_registry_with_more_than_the_base_is_dispatch_at_call_time():
    """The base alone is what runs, so it is walked. A function that merely carries a
    ``registry`` or a ``dispatch`` of some other kind is not a single-dispatch function."""
    import functools
    import types

    def search(query: str, limit: int = 5) -> str:
        return query

    levels = [["args", "kw"], ["query", "limit"]]
    base_only = functools.singledispatch(search)
    assert list(base_only.registry) == [object]
    assert _levels(base_only) == levels
    assert _levels(_traced(base_only)) == [["args", "kwargs"], *levels]
    assert unsupported_signature(base_only) is None

    plain = [["args", "kwargs"], ["query", "limit"]]
    plugins = _traced(search)
    plugins.registry = types.MappingProxyType({"csv": print})  # a registry, no dispatch
    assert _levels(plugins) == plain

    routed = _traced(search)
    routed.registry, routed.dispatch = ["csv"], print  # a dispatch, no registry of types
    assert _levels(routed) == plain


# --- what a callable holds is read without running it ---------------------------------------

_TOUCHED: List[str] = []


class _RaisingClass:
    """``__class__`` is a property that raises: a lazy object whose setup fails."""

    @property
    def __class__(self):
        _TOUCHED.append("__class__")
        raise RuntimeError("not set up")


class _LazyProxy:
    """A stand-in for a lazy proxy (a request's user, unconfigured settings): asking for
    its class or for any attribute runs its factory."""

    @property
    def __class__(self):
        _TOUCHED.append("__class__")
        return int

    def __getattr__(self, name):
        _TOUCHED.append(name)
        raise RuntimeError(name)


class _Stub:
    """A callable that answers every attribute with another of itself, as a remote stub
    does, and whose every comparison runs code."""

    def __call__(self):
        return None

    def __getattr__(self, name):
        _TOUCHED.append(name)
        return _Stub()

    def __eq__(self, other):
        _TOUCHED.append("__eq__")
        return True

    def __hash__(self):
        _TOUCHED.append("__hash__")
        return 0

    def __bool__(self):
        _TOUCHED.append("__bool__")
        return True


class _Interrupting:
    def __call__(self):
        return None

    def __getattr__(self, name):
        _TOUCHED.append(name)
        raise KeyboardInterrupt(name)


def _holding(value, function):
    """A tracer that also holds ``value`` in its closure."""
    import functools

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs), value

    return wrapper


def _calling(make_guarded):
    """A decorator whose wrapper calls ``make_guarded(function)``, held in its closure."""
    import functools

    def decorate(function):
        guarded = make_guarded(function)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return guarded(*args, **kwargs)

        return wrapper

    return decorate


class _DictProperty:
    """A callable whose ``__dict__`` is a property that runs code."""

    __slots__ = ()

    @property
    def __dict__(self):
        _TOUCHED.append("__dict__")
        raise RuntimeError("no dict")

    def __call__(self):
        return None


@pytest.mark.parametrize("held", [_RaisingClass, _LazyProxy, _Stub, _Interrupting, _DictProperty])
def test_nothing_a_callable_holds_is_asked_for_its_class_or_an_attribute(held):
    """A tool with no decorator at all can close over a lazy object. The walk reads what a
    callable holds, so it must not ask a held value for ``__class__`` (``isinstance``
    does) or for any attribute: that runs the object's code at judgment time, or raises
    out of the walk."""
    value = held()

    def search(query: str, limit: int = 5) -> str:
        return query

    def tool(query: str, limit: int = 5) -> str:
        return query + str(type(value))

    class Tool:
        def __init__(self):
            self.user = value

        def __call__(self, query: str, limit: int = 5) -> str:
            return query

    own = ["query", "limit"]
    del _TOUCHED[:]
    assert _levels(tool) == [own]
    assert unsupported_signature(tool) is None
    assert _levels(_holding(value, search)) == [["args", "kwargs"], own]
    assert unsupported_signature(_holding(value, search)) is None
    assert _levels(Tool()) == [own, own]
    assert _TOUCHED == []


def test_a_held_callable_names_what_it_wraps_without_an_attribute_lookup():
    """A held proxy names what it wraps through a slot, a property, or a class attribute.
    Each is found where it is declared; ``__getattr__`` is never the way."""
    import functools

    def fetch(url: str, timeout: int = 999999) -> int:
        return timeout

    guarded = _with_timeout(30)(fetch)

    def call(self, *args, **kwargs):
        return None

    def missing(self, name):
        _TOUCHED.append(name)
        raise AttributeError(name)

    class Slotted:
        __slots__ = ("__wrapped__",)
        __call__, __getattr__ = call, missing

        def __init__(self):
            self.__wrapped__ = guarded

    class Declared:
        __call__, __getattr__ = call, missing

        @property
        def __wrapped__(self):
            return guarded

    declared = Declared()
    vars(declared)["__wrapped__"] = fetch  # Python reads the property first, and so must this
    by_method = type("ByMethod", (), {"__call__": call, "__getattr__": missing})
    by_method.__wrapped__ = guarded  # a function in a class is bound on the way out
    by_static = type("ByStatic", (), {"__call__": call, "__getattr__": missing})
    by_static.__wrapped__ = staticmethod(guarded)
    by_value = type("ByValue", (), {"__call__": call, "__getattr__": missing})
    by_value.__wrapped__ = functools.partial(guarded)  # not a descriptor: read as stored

    class Failing:
        __call__, __getattr__ = call, missing

        @property
        def __wrapped__(self):
            raise AttributeError("__wrapped__")

    del _TOUCHED[:]
    for proxy in (Slotted(), declared, by_method(), by_static(), by_value()):
        with pytest.raises(HiddenWrapper):
            reached_parameters(_holding(proxy, fetch))
    # a property that fails names nothing, and Python's fallback to ``__getattr__`` is not taken
    assert _levels(_holding(Failing(), fetch)) == [["args", "kwargs"], ["url", "timeout"]]
    assert _TOUCHED == []


# --- a hidden wrapper that does not name what it wraps -------------------------------------


def test_a_hidden_wrapper_that_names_nothing_as_wrapped_is_refused():
    """The natural way to write one: a plain inner function, a lambda, a callable object or
    a bound method, none of them built with ``functools.wraps``. Each holds the function
    the walk reads, and each can keep or fill an argument."""
    import functools
    import types

    def fetch(url: str, timeout: int = 999999) -> int:
        return timeout

    def inner(function):
        def guarded(*args, timeout=30, **kwargs):
            return function(*args, **kwargs)

        return guarded

    def by_hand(function):
        def guarded(url, timeout=30):
            return function(url)

        return guarded

    def kept_by_name(function):
        def guarded(url, timeout):  # no default of its own: it takes the name the function takes
            return function(url)

        return guarded

    class Guard:
        def __init__(self, function):
            self.function = function

        def __call__(self, *args, timeout=30, **kwargs):
            return self.function(*args, **kwargs)

        def run(self, *args, timeout=30, **kwargs):
            return self.function(*args, **kwargs)

    def runner(function, *args, timeout=30, **kwargs):
        return function(*args, **kwargs)

    def relay(*args, function, timeout=30, **kwargs):
        return function(*args, **kwargs)

    def closing_method(function):
        def run(self, *args, timeout=30, **kwargs):
            return function(*args, **kwargs)

        return types.MethodType(run, object())

    for make_guarded in (
        inner,
        lambda function: lambda *args, timeout=30, **kwargs: function(*args, **kwargs),
        by_hand,
        kept_by_name,
        Guard,
        lambda function: Guard(function).run,
        closing_method,
        lambda function: functools.partial(runner, function),
        lambda function: functools.partial(relay, function=function),
    ):
        tool = _calling(make_guarded)(fetch)
        assert tool("u", timeout=5) == 999999  # the value judged is not the value run
        with pytest.raises(HiddenWrapper):
            reached_parameters(tool)
        assert unsupported_signature(tool) == _UNFOLLOWED
    # a partial that binds an argument fills it: the function's own default is never used
    preset = _calling(lambda function: functools.partial(function, timeout=30))(fetch)
    assert preset("u") == 30
    with pytest.raises(HiddenWrapper):
        reached_parameters(preset)


def test_a_held_callable_that_cannot_take_an_argument_is_not_a_hidden_wrapper():
    """What a wrapper holds may well hold the function too. It hides nothing when its own
    parameters can neither keep an argument (no name the function takes) nor fill one (no
    default): a helper, and a wrapper that only passes ``(*args, **kwargs)`` on."""
    import functools

    def search(query: str, limit: int = 5) -> str:
        return query

    def forwarding(function):
        def attempt(*args, **kwargs):
            return function(*args, **kwargs)

        return attempt

    def on_error(function):
        def report(error):
            return function.__name__, error

        return report

    expected = [["args", "kwargs"], ["query", "limit"]]
    for make_guarded in (
        forwarding,
        _traced,
        lambda function: _traced(_traced(function)),
        lambda function: functools.lru_cache(maxsize=None)(function),
        lambda function: functools.lru_cache(maxsize=None)(_traced(function)),
        lambda function: functools.partial(function),
        lambda function: _traced(functools.partial(_traced(function))),
        lambda function: lambda: function.__name__,
        on_error,
    ):
        tool = _calling(make_guarded)(search)
        assert _levels(tool) == expected
        assert unsupported_signature(tool) is None

    # the same helper is refused once it declares a default, or takes a name the function takes
    def with_default(function):
        def report(error, retries=3):
            return function.__name__, error

        return report

    def with_keyword_default(function):
        def report(error, *, retries=3):
            return function.__name__, error

        return report

    def with_a_taken_name(function):
        def report(query):
            return function.__name__, query

        return report

    for make_guarded in (
        with_default,
        with_keyword_default,
        with_a_taken_name,
        lambda function: functools.lru_cache(maxsize=None)(_with_timeout(30)(function)),
        lambda function: _traced(_with_timeout(30)(function)),
    ):
        with pytest.raises(HiddenWrapper):
            reached_parameters(_calling(make_guarded)(search))


def test_a_function_s_own_attributes_are_not_read_as_held():
    """A wrapper commonly carries a second entry point as an attribute (``tool.aio``). The
    walk reads what the call reaches, and a call to the wrapper does not reach it."""

    def search(query: str, limit: int = 5) -> str:
        return query

    tool = _traced(search)
    tool.aio = _with_timeout(30)(search)
    assert _levels(tool) == [["args", "kwargs"], ["query", "limit"]]


def test_a_hidden_wrapper_behind_a_view_is_refused():
    """A held ``staticmethod``, ``classmethod`` or ``property`` object, a held
    ``functools.partial`` and a held bound method only present a callable; what they
    present is read. So is the class a callable object is built from, and its slots."""
    import functools

    def fetch(url: str, timeout: int = 999999) -> int:
        return timeout

    class Service:
        def fetch(self, url: str, timeout: int = 999999) -> int:
            return timeout

    for make_guarded in (
        lambda function: staticmethod(_with_timeout(30)(function)).__func__,
        lambda function: functools.partial(_with_timeout(30)(function)),
        lambda function: _with_timeout(30)(functools.partial(function)),
    ):
        with pytest.raises(HiddenWrapper):
            reached_parameters(_calling(make_guarded)(fetch))

    def holding_a(view):
        def decorate(function):
            held = view(_with_timeout(30)(function))

            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                return held, function(*args, **kwargs)

            return wrapper

        return decorate

    for view in (staticmethod, classmethod, property, functools.partial):
        with pytest.raises(HiddenWrapper):
            reached_parameters(holding_a(view)(fetch))
        # the same view of the function itself, or of a tracer of it, hides nothing
        assert _levels(_holding(view(fetch), fetch)) == [["args", "kwargs"], ["url", "timeout"]]
        assert _levels(_holding(view(_traced(fetch)), fetch)) == [
            ["args", "kwargs"],
            ["url", "timeout"],
        ]

    # what a classmethod presents receives the class first: that is not a parameter of its own
    def receiving(function):
        @functools.wraps(function)
        def forward(cls, *args, **kwargs):
            return function(*args, **kwargs)

        return forward

    assert _levels(_holding(classmethod(receiving(fetch)), fetch)) == [
        ["args", "kwargs"],
        ["url", "timeout"],
    ]
    with pytest.raises(HiddenWrapper):
        reached_parameters(_holding(staticmethod(receiving(fetch)), fetch))

    # each access to a method builds a new object: the two are the same function
    service = Service()
    twice = functools.wraps(service.fetch)(
        _calling(lambda function: _with_timeout(30)(service.fetch))(service.fetch)
    )
    assert twice("u", timeout=5) == 999999
    with pytest.raises(HiddenWrapper):
        reached_parameters(twice)
    assert _levels(_holding(service.fetch, _traced(service.fetch))) == [
        ["args", "kwargs"],
        ["args", "kwargs"],
        ["url", "timeout"],
    ]

    def in_the_class(function):
        class Resilient:
            guarded = staticmethod(_with_timeout(30)(function))

            def __init__(self):
                functools.update_wrapper(self, function)

            def __call__(self, *args, **kwargs):
                return self.guarded(*args, **kwargs)

        return Resilient()

    def in_a_slot(function):
        class Resilient:
            __slots__ = ("__wrapped__", "guarded")

            def __init__(self):
                self.__wrapped__, self.guarded = function, _with_timeout(30)(function)

            def __call__(self, *args, **kwargs):
                return self.guarded(*args, **kwargs)

        return Resilient()

    def in_neither(function):
        class Plain:
            __slots__ = ("__wrapped__", "label")

            def __init__(self):
                self.__wrapped__ = function

            def describe(self):
                return function.__name__

            def __call__(self, *args, verbose=False, **kwargs):
                return function(*args, **kwargs)

        return Plain()

    for build in (in_the_class, in_a_slot):
        assert build(fetch)("u", timeout=5) == 999999
        with pytest.raises(HiddenWrapper):
            reached_parameters(build(fetch))
    # ``__call__`` is read by the walk itself, and an empty slot holds nothing
    assert _levels(in_neither(fetch)) == [
        ["args", "verbose", "kwargs"],
        ["args", "verbose", "kwargs"],
        ["url", "timeout"],
    ]


def test_a_held_chain_longer_than_the_walk_reads_is_refused_not_passed():
    """What a held wrapper reaches is read as deep as the walk goes. Past that it is not
    known, and not known is not the same as wrapping nothing."""
    import inspect

    def fetch(url: str, timeout: int = 999999) -> int:
        return timeout

    def tracers(count, function):
        for _ in range(count):
            function = _traced(function)
        return function

    within = _calling(lambda function: tracers(32, function))(fetch)
    assert _levels(within) == [["args", "kwargs"], ["url", "timeout"]]
    with pytest.raises(HiddenWrapper):
        reached_parameters(_calling(lambda function: tracers(33, function))(fetch))
    with pytest.raises(HiddenWrapper):
        reached_parameters(
            _calling(lambda function: tracers(33, _with_timeout(30)(function)))(fetch)
        )
    # and on the walk's own side: a wrapper of a function twenty tracers in is still found
    deep = tracers(20, fetch)
    hidden = _calling(lambda function: _with_timeout(30)(inspect.unwrap(function)))(deep)
    assert hidden("u", timeout=5) == 999999
    with pytest.raises(HiddenWrapper):
        reached_parameters(hidden)


def test_a_wrapper_that_keeps_the_descriptor_it_came_from_is_walked():
    """A caching method decorator hands out one wrapper for each instance, and that wrapper
    keeps the descriptor it came from. Both name the method as ``__wrapped__``, but the
    descriptor's own call takes nothing by name, so nothing can be hidden in it. The same
    shape is refused the moment what is kept can take an argument."""
    import functools
    import types

    class Descriptor:
        def __init__(self, method):
            functools.update_wrapper(self, method)

        def __get__(self, obj, objtype=None):
            return Bound(obj, self, self.__wrapped__)

        def __call__(self, *args, **kwargs):
            raise TypeError("not called directly")

    class Bound:
        def __init__(self, obj, descriptor, method):
            functools.update_wrapper(self, method)
            self.descriptor = descriptor
            self.method = types.MethodType(method, obj)

        def __call__(self, *args, **kwargs):
            return self.method(*args, **kwargs)

    class Keeping(Descriptor):
        def __call__(self, *args, timeout=30, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    class Holder:
        @Descriptor
        def search(self, query: str, limit: int = 5) -> str:
            return query

        @Keeping
        def fetch(self, url: str, timeout: int = 999999) -> int:
            return timeout

    assert Holder().search("q") == "q"
    assert _levels(Holder().search) == [
        ["args", "kwargs"],
        ["args", "kwargs"],
        ["self", "query", "limit"],
    ]
    assert unsupported_signature(Holder().search, skip=("self",)) is None
    with pytest.raises(HiddenWrapper):
        reached_parameters(Holder().fetch)


def test_attributes_copied_from_the_wrapped_function_are_not_read_as_held():
    """``functools.update_wrapper`` copies the wrapped function's attributes onto a
    callable object. They are that function's attributes, which the walk does not read:
    a single-dispatch method's ``register`` is one, and it holds the base function."""
    import functools

    class Holder:
        @functools.singledispatchmethod
        def plain(self, key: str, deep: bool = False) -> str:
            return "base"

    class Decorator:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    access = Holder().plain
    assert access.register.__self__.func is vars(Holder)["plain"].func
    own = ["self", "key", "deep"]
    assert _levels(Decorator(access)) == [
        ["args", "kwargs"],
        ["args", "kwargs"],
        ["args", "kwargs"],
        own,
    ]
    assert _levels(functools.lru_cache(maxsize=None)(access)) == [
        ["args", "kwargs"],
        ["args", "kwargs"],
        own,
    ]


# --- a dispatcher read by structure ---------------------------------------------------------


def test_a_registry_that_replaces_the_base_or_cannot_be_read_is_refused():
    """``register(object, ...)`` replaces the base implementation and leaves ``object`` the
    only key: the base alone runs only when that entry is the function ``__wrapped__``
    names. A registry that cannot be listed is refused, not raised."""
    import functools
    import types

    @functools.singledispatch
    def lookup(key: str, deep: bool = False) -> str:
        return "base"

    @lookup.register(object)
    def replacement(key: str, deep: bool = False, skip_review: bool = True) -> str:
        return f"replacement, skip_review={skip_review}"

    class Holder:
        @functools.singledispatchmethod
        def lookup(self, key: str, deep: bool = False) -> str:
            return "base"

        @lookup.register(object)
        def replacement(self, key: str, deep: bool = False, skip_review: bool = True) -> str:
            return f"replacement, skip_review={skip_review}"

    assert list(lookup.registry) == [object]
    assert lookup("k") == Holder().lookup("k") == "replacement, skip_review=True"
    for func in (lookup, _traced(lookup), Holder().lookup):
        with pytest.raises(DispatchedAtCallTime):
            reached_parameters(func)
    assert unsupported_signature(lookup) == _AT_CALL_TIME

    def search(query: str, limit: int = 5) -> str:
        return query

    class Unlistable(dict):
        def items(self):
            raise RuntimeError("not listed")

    odd = _traced(search)
    odd.registry, odd.dispatch = types.MappingProxyType(Unlistable({object: search})), print
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(odd)
    assert unsupported_signature(odd) == _AT_CALL_TIME


def test_what_only_looks_like_a_dispatcher_is_walked():
    """A single-dispatch function is known by what ``functools.singledispatch`` builds: a
    ``mappingproxy`` registry with an entry for ``object``, and a callable ``dispatch``."""
    import functools
    import types

    def search(query: str, limit: int = 5) -> str:
        return query

    entries = {object: search, int: print}
    plain = [["args", "kwargs"], ["query", "limit"]]

    def carrying(registry, dispatch):
        tool = _traced(search)
        tool.registry, tool.dispatch = registry, dispatch
        return tool

    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(carrying(types.MappingProxyType(entries), print))
    for registry, dispatch in (
        (dict(entries), print),  # a plain mapping is not what singledispatch builds
        (types.MappingProxyType(entries), "by name"),  # a dispatch that cannot be called
        (types.MappingProxyType({}), print),  # no entry for ``object``
        (types.MappingProxyType({"csv": print}), print),  # handlers by name, not by type
    ):
        assert _levels(carrying(registry, dispatch)) == plain

    # a wrapper only carries a copy of the attributes of the dispatcher it wraps
    base_only = functools.singledispatch(search)
    assert _levels(functools.lru_cache(maxsize=None)(base_only)) == [
        ["args", "kw"],
        ["args", "kw"],
        ["query", "limit"],
    ]

    # a helper that dispatches, held beside the function, is not on the way of the call
    assert _levels(_holding(_dispatching(), search)) == plain

    # a dispatcher that no longer names its base cannot be shown to run the base alone
    nameless = functools.singledispatch(search)
    del nameless.__wrapped__
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(nameless)


def test_a_dispatcher_a_wrapper_calls_in_place_of_its_base_is_refused():
    """A wrapper that names the base function as ``__wrapped__`` and calls the dispatcher
    it holds: the dispatcher is a hidden wrapper of the base, and it chooses at call time."""
    import functools

    lookup = _dispatching()

    @functools.wraps(lookup.__wrapped__)
    def tool(*args, **kwargs):
        return lookup(*args, **kwargs)

    assert tool(1) == "int, skip_review=True"
    with pytest.raises(HiddenWrapper):
        reached_parameters(tool)
    # the same wrapper around a dispatcher with only its base hides nothing
    base_only = functools.singledispatch(lookup.__wrapped__)

    @functools.wraps(lookup.__wrapped__)
    def plain(*args, **kwargs):
        return base_only(*args, **kwargs)

    assert _levels(plain) == [["args", "kwargs"], ["key", "deep"]]


def test_a_dispatcher_is_found_without_an_attribute_lookup():
    """``registry`` and ``dispatch`` are read where they are stored, and a
    ``singledispatchmethod`` is known by its type, a subclass included."""
    import functools
    import types

    def search(query: str, limit: int = 5) -> str:
        return query

    class Dynamic:
        """Answers ``registry`` and ``dispatch`` from ``__getattr__`` only."""

        def __call__(self, query: str, limit: int = 5) -> str:
            return query

        def __getattr__(self, name):
            _TOUCHED.append(name)
            if name == "registry":
                return types.MappingProxyType({object: search, int: print})
            if name == "dispatch":
                return print
            raise AttributeError(name)

    class Half(Dynamic):
        """Stores a registry, and answers ``dispatch`` from ``__getattr__`` only."""

        def __init__(self):
            self.registry = types.MappingProxyType({object: search, int: print})

    del _TOUCHED[:]
    for tool in (Dynamic(), Half()):
        assert _levels(tool) == [["query", "limit"], ["query", "limit"]]
    assert "registry" not in _TOUCHED and "dispatch" not in _TOUCHED

    class Recording(functools.singledispatchmethod):
        def __getattribute__(self, name):
            _TOUCHED.append(name)
            return super().__getattribute__(name)

    class Holder:
        @Recording
        def lookup(self, key: str, deep: bool = False) -> str:
            return "base"

        @lookup.register(int)
        def lookup_int(self, key: int, deep: bool = False, skip_review: bool = True) -> str:
            return "int"

    tool = Holder().lookup
    del _TOUCHED[:]
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(tool)
    assert _TOUCHED == []


def inspect_signature(func):
    import inspect

    return inspect.signature(func).parameters


# --- detached -----------------------------------------------------------------------------


def test_a_policys_write_to_the_detached_copy_does_not_reach_the_original():
    original = {"k": [1, {"j": "v"}], "n": None}
    copy = detached(original)
    assert copy == original
    copy["k"][1]["j"] = "changed"
    copy["k"].append(2)
    copy["new"] = True
    assert original == {"k": [1, {"j": "v"}], "n": None}


def test_a_container_reachable_twice_is_copied_and_a_cycle_is_refused():
    shared = [1]
    assert detached({"a": shared, "b": shared}) == {"a": [1], "b": [1]}
    cyclic: Dict[str, Any] = {}
    cyclic["self"] = cyclic
    with pytest.raises(TypeError, match="cyclic"):
        detached(cyclic)


class _IntKind(enum.IntEnum):
    ONE = 1


class _DictSub(dict):
    pass


class _StrSub(str):
    pass


@pytest.mark.parametrize(
    "value",
    [
        _Color.RED,
        _IntKind.ONE,
        float("nan"),
        float("inf"),
        (1, 2),
        {1, 2},
        b"x",
        object(),
        _DictSub(a=1),
        _StrSub("x"),
        {1: "a"},
        {"k": [_Color.RED]},
    ],
)
def test_what_is_not_plain_data_is_refused(value):
    with pytest.raises(TypeError):
        detached({"v": value})


# --- one definition -----------------------------------------------------------------------

SHARED = {
    "_supported_annotation",
    "_json_data",
    "_unsupported_signature",
    "_detached",
    "supported_annotation",
    "json_data",
    "unsupported_signature",
    "detached",
}

#: The gates that import the shared helpers, and the names each one uses.
USERS = {
    "adapters/crewai/src/recusal_crewai/_gate.py": {"json_data", "supported_annotation"},
    "adapters/google-adk/src/recusal_google_adk/_gate.py": {
        "detached",
        "json_data",
        "reached_parameters",
        "unsupported_signature",
    },
    "adapters/openai/src/recusal_openai/_gate.py": {"unsupported_signature"},
    "adapters/pydantic-ai/src/recusal_pydantic_ai/_gate.py": {"detached"},
}


class _Source:
    """A gate source file and the name the lock reports it by: the file name for an
    example, the repository path for a module of an adapter package."""

    def __init__(self, path, name):
        self.path, self.name = path, name


def _trees():
    for path in sorted(EXAMPLES.glob("*.py")):
        yield _Source(path, path.name), ast.parse(path.read_text(encoding="utf-8"))
    for path in sorted((EXAMPLES.parent / "adapters").glob("*/src/**/*.py")):
        name = path.relative_to(EXAMPLES.parent).as_posix()
        yield _Source(path, name), ast.parse(path.read_text(encoding="utf-8"))


def test_no_example_defines_its_own_copy_of_a_shared_helper():
    copies = [
        f"{path.name}:{node.lineno} {node.name}"
        for path, tree in _trees()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in SHARED
    ]
    assert copies == []


def test_the_gates_that_used_a_copy_import_the_shared_helper():
    found = {}
    for path, tree in _trees():
        names = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "recusal.invocation"
            for alias in node.names
        }
        if names:
            found[path.name] = names
    assert found == USERS


def test_negative_control_the_lock_sees_a_copy_when_one_exists():
    tree = ast.parse("def _json_data(value):\n    return True\n")
    assert [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in SHARED]
