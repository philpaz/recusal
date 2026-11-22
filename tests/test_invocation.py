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

    # a wrapper that calls itself again (a retry) holds itself, and it wraps the function
    def again(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return wrapper if not args else function(*args, **kwargs)

        return wrapper

    assert _levels(again(search)) == [forward, own]

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
