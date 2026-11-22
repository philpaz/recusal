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

from recusal import invocation
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


# --- a wrapper the walk does not read ------------------------------------------------------

_UNFOLLOWED = (
    "a wrapper holds another wrapper of the function it wraps, which the walk does not follow"
)
_KEPT = (
    "`timeout` is taken by a wrapper and by the function it wraps, which can then run with a "
    "value no policy judged"
)
_ADDED = (
    "`force` is a parameter of a function the call reaches that the tool does not declare, so "
    "it would run with a value no policy judged"
)
_FORWARD, _FETCH = ["args", "kwargs"], ["url", "timeout"]


def _with_timeout(function):
    """A ``functools.wraps`` wrapper that keeps ``timeout`` for itself."""
    import functools

    @functools.wraps(function)
    def guarded(*args, timeout=30, **kwargs):
        return function(*args, **kwargs)

    return guarded


def _forcing(function):
    """A ``functools.wraps`` wrapper that adds ``force``, filled from its own default."""
    import functools

    @functools.wraps(function)
    def wrapper(*args, force=True, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def _calling(make_guarded):
    """A decorator that applies ``make_guarded`` inside itself: its wrapper calls the
    result, held only in its closure, while ``functools.wraps`` names the function."""
    import functools

    def decorate(function):
        guarded = make_guarded(function)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return guarded(*args, **kwargs)

        return wrapper

    return decorate


def _holding(value, function):
    """A tracer of ``function`` that also holds ``value`` in its closure."""
    import functools

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs), value

    return wrapper


def _levels(func):
    return [list(level) for level in reached_parameters(func)]


def _fetch():
    def fetch(url: str, timeout: int = 999999) -> int:
        return timeout

    return fetch


def test_a_wrapper_applied_inside_another_keeps_an_argument_and_is_refused():
    """Three adapter reviews, one gap: ``resilient`` applies ``with_timeout`` inside itself
    and calls the result, but its ``__wrapped__`` is the undecorated function, so the walk
    never read the wrapper in between. The policy judged ``timeout=5``; that wrapper kept
    it and the function ran with its own default."""
    fetch = _fetch()
    tool = _calling(_with_timeout)(fetch)
    assert tool.__wrapped__ is fetch
    assert list(inspect_signature(tool)) == ["url", "timeout"]
    assert tool("u", timeout=5) == 999999  # the value judged is not the value run
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    # the reason is the one a wrapper the walk reads gets: it names the parameter, no value
    assert str(refused.value) == _KEPT
    assert unsupported_signature(tool) == _KEPT
    assert unsupported_signature(_with_timeout(fetch)) == _KEPT  # the same wrapper, read


def test_the_refusal_is_a_value_error_so_a_gate_that_catches_one_refuses():
    assert issubclass(HiddenWrapper, ValueError)
    assert not issubclass(HiddenWrapper, DeclaredSignature)


def test_a_default_of_a_wrapper_applied_inside_another_is_refused():
    """The hidden wrapper's own ``force=True`` ran unseen the same way."""

    def wipe(table: str) -> str:
        return table

    tool = _calling(_forcing)(wipe)
    assert list(inspect_signature(tool)) == ["table"]
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    assert str(refused.value) == _ADDED
    assert unsupported_signature(tool) == _ADDED
    assert unsupported_signature(_forcing(wipe)) == _ADDED  # the same wrapper, read


def test_a_hidden_wrapper_is_refused_wherever_the_wrapper_that_holds_it_stands():
    """Further in behind tracers, behind a ``functools.partial`` and ``functools.lru_cache``,
    as a method, as a coroutine function, and in an instance attribute of a class-based
    decorator."""
    import functools

    tool = _calling(_with_timeout)(_fetch())

    class Holder:
        @_calling(_with_timeout)
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

    class Resilient:
        def __init__(self, function):
            functools.update_wrapper(self, function)
            self.guarded = _with_timeout(function)

        def __call__(self, *args, **kwargs):
            return self.guarded(*args, **kwargs)

    in_an_attribute = Resilient(_fetch())
    assert in_an_attribute("u", timeout=5) == 999999
    for func in (
        _traced(tool),
        _traced(_traced(tool)),
        functools.partial(tool),
        functools.partial(_traced(tool), "u"),
        functools.lru_cache(maxsize=None)(tool),
        Holder().fetch,
        fetch_async,
        in_an_attribute,
        _traced(in_an_attribute),
        functools.partial(in_an_attribute),
    ):
        with pytest.raises(HiddenWrapper):
            reached_parameters(func)
    for func in (_traced(tool), Holder().fetch, fetch_async):
        assert unsupported_signature(func, skip=("self",)) == _KEPT, func


def _receiving(function):
    """A ``functools.wraps`` wrapper whose one named parameter is the receiver it is bound
    to: nothing of its own once bound."""
    import functools

    @functools.wraps(function)
    def forward(receiver, *args, **kwargs):
        return function(*args, **kwargs)

    return forward


def _keeping_method(function):
    import functools

    @functools.wraps(function)
    def guarded(receiver, *args, timeout=30, **kwargs):
        return function(*args, **kwargs)

    return guarded


def _closed_set():
    """Each type of held value the check reads, as (name, a hidden wrapper of that type,
    the same type hiding nothing)."""
    import functools
    import types

    cached = functools.lru_cache(maxsize=None)
    return [
        ("function", _with_timeout, _traced),
        ("function, two links", lambda f: _traced(_with_timeout(f)), lambda f: _traced(_traced(f))),
        (
            "bound method of a function",
            lambda f: types.MethodType(_keeping_method(f), object()),
            lambda f: types.MethodType(_receiving(f), object()),
        ),
        ("bound method of the function itself", None, lambda f: types.MethodType(_receiving(f), 0)),
        (
            "partial",
            lambda f: functools.partial(_with_timeout(f)),
            lambda f: functools.partial(_traced(f)),
        ),
        ("partial of the function itself", None, lambda f: functools.partial(f)),
        (
            "partial that binds a receiver",
            lambda f: functools.partial(_keeping_method(f), object()),
            lambda f: functools.partial(_receiving(f), object()),
        ),
        (
            "partial that binds nothing for a receiver",
            lambda f: functools.partial(_receiving(f)),
            None,
        ),
        (
            "staticmethod",
            lambda f: staticmethod(_with_timeout(f)),
            lambda f: staticmethod(_traced(f)),
        ),
        ("staticmethod of a function with a receiver", lambda f: staticmethod(_receiving(f)), None),
        (
            "classmethod",
            lambda f: classmethod(_keeping_method(f)),
            lambda f: classmethod(_receiving(f)),
        ),
        ("lru_cache", lambda f: cached(_with_timeout(f)), lambda f: cached(f)),
        ("lru_cache over a tracer", None, lambda f: cached(_traced(f))),
    ]


@pytest.mark.parametrize(
    "name, hidden, harmless", _closed_set(), ids=lambda v: v if type(v) is str else ""
)
def test_each_type_of_the_closed_set_is_read_for_what_it_passes_a_call_on_to(
    name, hidden, harmless
):
    """A held value of one of these types is followed, link by link, to a callable the walk
    reads; it is refused only when a link on the way can take an argument of its own."""
    fetch = _fetch()
    if hidden is not None:
        with pytest.raises(HiddenWrapper):
            reached_parameters(_holding(hidden(fetch), fetch))
    if harmless is not None:
        assert _levels(_holding(harmless(fetch), fetch)) == [_FORWARD, _FETCH]


def test_a_hidden_wrapper_is_found_as_deep_as_the_walk_reads_and_no_deeper():
    """Either chain may be long: what is held may wrap the function through other wrappers,
    and may wrap a function further in than the one ``__wrapped__`` names. A held chain
    still going after 32 links is refused: not known is not the same as wrapping nothing."""
    import functools
    import inspect

    fetch = _fetch()

    def tracers(count, function):
        for _ in range(count):
            function = _traced(function)
        return function

    assert _levels(_calling(lambda f: tracers(31, f))(fetch)) == [_FORWARD, _FETCH]
    for make_guarded in (
        lambda f: tracers(30, _with_timeout(f)),
        lambda f: tracers(32, f),
        lambda f: tracers(40, _with_timeout(f)),
    ):
        with pytest.raises(HiddenWrapper):
            reached_parameters(_calling(make_guarded)(fetch))

    # the walk's side: a wrapper of a function twenty tracers in is found
    deep = tracers(20, fetch)
    hidden = _calling(lambda f: _with_timeout(inspect.unwrap(f)))(deep)
    assert hidden("u", timeout=5) == 999999
    with pytest.raises(HiddenWrapper):
        reached_parameters(hidden)

    # the walk's side is read through the same closed set: a method names its function
    class Service:
        def fetch(self, url: str, timeout: int = 999999) -> int:
            return timeout

    service = Service()
    twice = _calling(lambda f: _with_timeout(service.fetch))(service.fetch)
    assert twice("u", timeout=5) == 999999
    with pytest.raises(HiddenWrapper):
        reached_parameters(twice)
    through_partial = _calling(_with_timeout)(fetch)
    through_partial.__wrapped__ = functools.partial(fetch)
    with pytest.raises(HiddenWrapper):
        reached_parameters(through_partial)


def test_what_a_wrapper_holds_that_is_not_a_hidden_wrapper_is_walked():
    """The wrapper itself, a wrapper the walk reads anyway, a cycle, an empty cell, and
    attributes ``functools.update_wrapper`` copied from the function."""
    import functools

    def search(query: str, limit: int = 5) -> str:
        return query

    own = ["query", "limit"]

    # a wrapper that calls itself again holds itself; its parameter is a level of the walk
    def again(function):
        @functools.wraps(function)
        def wrapper(*args, attempt=0, **kwargs):
            return wrapper if not args else function(*args, **kwargs)

        return wrapper

    assert _levels(again(search)) == [["args", "attempt", "kwargs"], own]

    # a wrapper may hold what it wraps and what that wraps in turn, parameters and all
    def unwrapping(function):
        inner = function.__wrapped__

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return (function if args else inner)(*args, **kwargs)

        return wrapper

    assert _levels(unwrapping(_traced(_forcing(search)))) == [
        _FORWARD,
        _FORWARD,
        ["args", "force", "kwargs"],
        own,
    ]

    # a cycle of ``__wrapped__`` on the held side reaches nothing the walk reads
    def first(*args, flag=True):
        return None

    def second(*args, flag=True):
        return None

    first.__wrapped__, second.__wrapped__ = second, first  # type: ignore[attr-defined]
    assert _levels(_holding(first, search)) == [_FORWARD, own]

    # a cell that was never filled
    def unfilled(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs), later

        if not function:
            later = None
        return wrapper

    assert _levels(unfilled(search)) == [_FORWARD, own]

    # a class-based decorator carries the function's own attributes, copied: not read
    class Decorator:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    search.raw = _with_timeout(search)  # type: ignore[attr-defined]
    assert _levels(Decorator(search)) == [_FORWARD, _FORWARD, own]
    assert _levels(functools.lru_cache(maxsize=None)(search)) == [own, own]

    # an instance dictionary that is not a plain dict is not listed: listing it runs code
    class Attributes(dict):
        values = _touch("values", raises=RuntimeError)
        items = _touch("items", raises=RuntimeError)

    odd = Decorator(search)
    odd.__dict__ = Attributes(vars(odd), guarded=_with_timeout(search))
    del _TOUCHED[:]
    assert _levels(odd) == [_FORWARD, _FORWARD, own]
    assert _TOUCHED == []

    # a callable that names itself is refused as too deep, as before, not as a hidden wrapper
    looped = _traced(search)
    looped.__wrapped__ = looped
    with pytest.raises(ValueError, match="too many wrappers") as refused:
        reached_parameters(looped)
    assert not isinstance(refused.value, HiddenWrapper)


def test_ordinary_decorators_are_walked_as_before():
    """The refusal is for one shape only. What a decorator ordinarily holds (the function,
    configuration, helpers) is not it."""
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
        # a second wrapper of the same function that takes nothing of its own
        (_calling(_traced)(search), [forward, own]),
        (_calling(functools.lru_cache(maxsize=None))(search), [forward, own]),
    ):
        assert _levels(func) == expected, func
    # ``contextlib.contextmanager`` is itself a ``functools.wraps`` decorator
    assert _levels(span) == [["args", "kwds"], []]
    for func in (_traced(search), retrying(3)(search), observed(print)(search), search_async):
        assert unsupported_signature(func) is None, func


# --- one rule for a wrapper the walk reads and for one a callable holds --------------------


def _copying(function):
    """A wrapper that copies the function's own parameters, as the signature-preserving
    decorators of several libraries generate one."""
    import functools

    def search(query, limit=5):
        return function(query, limit)

    return functools.update_wrapper(search, function)


def _renaming(function):
    import functools

    def search(query, limit=5, verbose=False):
        return function(query, limit)

    return functools.update_wrapper(search, function)


def test_a_held_wrapper_is_judged_by_the_rule_and_the_code_that_judge_one_the_walk_reads():
    """A wrapper gets the same verdict, for the same reason, held inside a decorator and
    applied directly: one function decides both."""
    import functools

    def search(query: str, limit: int = 5) -> str:
        return query

    def with_context(function):
        @functools.wraps(function)
        def wrapper(*args, tool_context=None, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    def keeping_limit(function):
        @functools.wraps(function)
        def wrapper(*args, limit=1, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    def keeping_through_args(function):
        @functools.wraps(function)
        def wrapper(*args, limit=1):  # passes on through ``*args`` alone
            return function(*args)

        return wrapper

    def keeping_through_kwargs(function):
        @functools.wraps(function)
        def wrapper(query, limit=1, **kwargs):  # passes on through ``**kwargs`` alone
            return function(query, **kwargs)

        return wrapper

    for make, skip in (
        (_traced, ()),
        (_copying, ()),  # names only what the tool declares, and passes nothing else on
        (_renaming, ()),  # a name the tool does not declare
        (_forcing, ()),
        (keeping_limit, ()),  # a forwarding wrapper that keeps a name the function takes
        (keeping_through_args, ()),
        (keeping_through_kwargs, ()),
        (with_context, ()),
        (with_context, ("tool_context",)),  # a name the framework injects
        (lambda f: _traced(_copying(f)), ()),
        (lambda f: _traced(keeping_limit(f)), ()),
    ):
        read = unsupported_signature(make(search), skip)
        held = unsupported_signature(_calling(make)(search), skip)
        assert held == read, (make, read, held)
        if read is None:
            levels = reached_parameters(_calling(make)(search), skip)
            assert [list(level) for level in levels] == [_FORWARD, ["query", "limit"]]
        else:
            with pytest.raises(HiddenWrapper) as refused:
                reached_parameters(_calling(make)(search), skip)
            assert str(refused.value) == read
    # a wrapper the walk reads is not judged as a held one: it stays a level of the walk,
    # also when the same callable holds a harmless second wrapper
    for visible, level, why in (
        (_forcing(search), ["args", "force", "kwargs"], _ADDED),
        (keeping_limit(search), ["args", "limit", "kwargs"], "`limit` is taken by a wrapper"),
    ):
        tool = _calling(_traced)(visible)
        assert _levels(tool) == [_FORWARD, level, ["query", "limit"]]
        assert why in unsupported_signature(tool)
    assert unsupported_signature(_copying(search)) is None
    assert "`verbose` is a parameter" in unsupported_signature(_calling(_renaming)(search))
    assert "`limit` is taken by a wrapper" in unsupported_signature(_calling(keeping_limit)(search))
    assert unsupported_signature(_calling(with_context)(search)) is not None
    assert unsupported_signature(_calling(with_context)(search), ("tool_context",)) is None


def test_a_held_wrapper_of_a_method_receives_the_receiver_the_method_does():
    """A method's decorator is usually written ``def wrapper(self, *args, **kwargs)``. When
    the tool is the bound method, the walk reads its levels without the receiver; a wrapper
    its decorator holds is passed the same receiver, so its first positional parameter is
    not one of its own either, whatever it is called. Held and read agree here too."""
    import functools
    import types

    def held(make):
        return _calling(make)

    def cases(make):
        class Service:
            @make
            def search(self, query: str, limit: int = 5) -> str:
                return query

            @classmethod
            @make
            def build(cls, query: str, limit: int = 5) -> str:
                return query

        def search(receiver, query: str, limit: int = 5) -> str:
            return query

        bound = functools.partial(make(search), object())
        return Service().search, Service.build, Service().build, bound

    for make, keeps in (
        (_receiving, False),
        (_traced, False),
        (_copying_method, False),
        (_keeping_method, True),
    ):
        for read, kept in zip(cases(make), cases(held(make))):
            reached_parameters(read)  # read on the walk, it is a level, never a hidden wrapper
            if not isinstance(read, functools.partial):
                why = unsupported_signature(read)
                assert (why is not None) is keeps, (make, read, why)
                assert unsupported_signature(kept) == why, (make, read)
            if keeps:
                with pytest.raises(HiddenWrapper):
                    reached_parameters(kept)
            else:
                reached_parameters(kept)
    service_tool = cases(held(_receiving))[0]
    assert unsupported_signature(service_tool) is None
    assert _levels(service_tool) == [_FORWARD, ["query", "limit"]]
    assert "`timeout` is a parameter" in unsupported_signature(cases(held(_keeping_method))[0])

    # the receiver is counted once for every binding on the way, and never forgotten: a
    # bound method of an ``lru_cache`` wrapper still binds what the function receives
    fetch = _fetch()
    cached = functools.lru_cache(maxsize=None)
    assert _levels(_holding(types.MethodType(cached(_receiving(fetch)), object()), fetch)) == [
        _FORWARD,
        _FETCH,
    ]
    with pytest.raises(HiddenWrapper):
        reached_parameters(_holding(types.MethodType(cached(_keeping_method(fetch)), 0), fetch))


def _copying_method(function):
    """A wrapper that copies a method's own parameters, receiver included."""
    import functools

    def search(self, query, limit=5):
        return function(self, query, limit)

    return functools.update_wrapper(search, function)


def test_a_bound_receiver_is_not_a_parameter_and_a_keyword_behind_it_is():
    import types

    fetch = _fetch()
    # the wrapper takes its receiver through ``*args``: nothing is bound by name
    keeping = types.MethodType(_with_timeout(fetch), object())
    with pytest.raises(HiddenWrapper):
        reached_parameters(_holding(keeping, fetch))
    assert _levels(_holding(types.MethodType(_traced(fetch), object()), fetch)) == [
        _FORWARD,
        _FETCH,
    ]


def test_a_held_wrapper_of_anything_the_walk_reads_is_found():
    """What the walk reads is every callable it took the parameters of: the function a
    callable object names as ``__wrapped__``, that object's own ``__call__``, the function
    of a method or a ``functools.partial`` on the way."""
    import functools

    def search(query: str, limit: int = 5) -> str:
        return query

    def keeping_limit(function):
        @functools.wraps(function)
        def wrapper(*args, limit=1, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    class Decorator:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    class Tool:
        def __call__(self, query: str, limit: int = 5) -> str:
            return query

    def holding(guarded, wrapped):
        @functools.wraps(wrapped)
        def wrapper(*args, **kwargs):
            return guarded(*args, **kwargs)

        return wrapper

    through_an_object = Decorator(search)
    tool = Tool()
    for guarded, wrapped in (
        (keeping_limit(search), through_an_object),  # read through the object's __wrapped__
        (keeping_limit(through_an_object), through_an_object),
        (keeping_limit(tool.__call__), tool),  # the object's own __call__
        (keeping_limit(search), functools.partial(search)),
        (keeping_limit(functools.partial(search)), search),
    ):
        with pytest.raises(HiddenWrapper):
            reached_parameters(holding(guarded, wrapped))

    # an equal callable is not the one the walk reads
    class Alike(Tool):
        __eq__ = _touch("__eq__", True)
        __hash__ = _touch("__hash__", 0)

    one, other = Tool(), Alike()
    del _TOUCHED[:]
    reached_parameters(holding(keeping_limit(other), one))
    assert _TOUCHED == []


def test_a_dictionary_another_thread_changes_does_not_break_the_walk():
    """An attribute set or removed on another thread while the walk lists a dictionary must
    not raise out of it. Every live dictionary is listed in one step before any loop."""
    import functools
    import sys as system
    import threading
    import time

    def search(query: str, limit: int = 5) -> str:
        return query

    class Decorator:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    inner = _traced(search)
    cached = functools.lru_cache(maxsize=None)(search)
    instance = Decorator(search)
    shapes = (
        (instance, instance, [_FORWARD, _FORWARD, ["query", "limit"]]),
        (_holding(inner, search), inner, [_FORWARD, ["query", "limit"]]),
        (_holding(cached, search), cached, [_FORWARD, ["query", "limit"]]),
    )
    stop = threading.Event()

    def churn(target):
        names = [f"extra_{index}" for index in range(64)]
        while not stop.is_set():
            for name in names:
                setattr(target, name, None)
            for name in names:
                delattr(target, name)

    interval = system.getswitchinterval()
    system.setswitchinterval(1e-6)
    workers = [
        threading.Thread(target=churn, args=(target,), daemon=True) for _, target, _ in shapes
    ]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 2.0
        walks = 0
        while walks < 400 and time.monotonic() < deadline:
            for tool, _, expected in shapes:
                assert _levels(tool) == expected
            walks += 1
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=5)
        system.setswitchinterval(interval)
    assert walks > 0


def test_a_dictionary_that_is_not_a_plain_dict_is_not_listed():
    """A function's ``__dict__`` can be replaced by a subclass of ``dict``, whose listing
    runs code. Such a function names nothing the check reads."""
    fetch = _fetch()

    class Attributes(dict):
        items = _touch("items", raises=RuntimeError)
        values = _touch("values", raises=RuntimeError)
        __iter__ = _touch("__iter__", raises=RuntimeError)

    guarded = _with_timeout(fetch)
    guarded.__dict__ = Attributes(guarded.__dict__)
    del _TOUCHED[:]
    assert _levels(_holding(guarded, fetch)) == [_FORWARD, _FETCH]
    assert _TOUCHED == []


# --- the stated limits: accepted, on purpose ------------------------------------------------

_SLOT = []


def _limits():
    """Every hidden wrapper the check does not find, as (name, a decorator that builds it).
    Each really hides the argument; each is accepted. Widening the check to one of these is
    a decision, and this list is where it shows."""
    import functools

    def named_nothing(function):
        def guarded(*args, timeout=30, **kwargs):  # a plain inner function, no functools.wraps
            return function(*args, **kwargs)

        return guarded

    class Guard:
        def __init__(self, function):
            self.function = function

        def __call__(self, *args, timeout=30, **kwargs):
            return self.function(*args, **kwargs)

        def run(self, *args, timeout=30, **kwargs):
            return self.function(*args, **kwargs)

    class Proxy:
        """An object that names what it wraps, as a transparent proxy does."""

        def __init__(self, inner):
            self.__wrapped__ = inner

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    class Partial(functools.partial):
        pass

    def in_a_container(function):
        guards = [_with_timeout(function)]

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return guards[0](*args, **kwargs)

        return wrapper

    def in_a_global(function):
        _SLOT[:] = [_with_timeout(function)]

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return _SLOT[0](*args, **kwargs)

        return wrapper

    def in_a_function_attribute(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return wrapper.guarded(*args, **kwargs)

        wrapper.guarded = _with_timeout(function)
        return wrapper

    def in_a_class_attribute(function):
        class Resilient:
            guarded = staticmethod(_with_timeout(function))

            def __init__(self):
                functools.update_wrapper(self, function)

            def __call__(self, *args, **kwargs):
                return self.guarded(*args, **kwargs)

        return Resilient()

    def in_a_slot(function):
        class Resilient:
            __slots__ = ("__wrapped__", "guarded")

            def __init__(self):
                self.__wrapped__, self.guarded = function, _with_timeout(function)

            def __call__(self, *args, **kwargs):
                return self.guarded(*args, **kwargs)

        return Resilient()

    def in_its_own_method(function):
        class Resilient:
            def __init__(self):
                functools.update_wrapper(self, function)

            def guarded(self, *args, timeout=30, **kwargs):
                return function(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                return self.guarded(*args, **kwargs)

        return Resilient()

    def by_hand(function):
        guarded = _with_timeout(function)

        def wrapper(url: str, timeout: int = 999999) -> int:  # names nothing as __wrapped__
            return guarded(url, timeout=timeout)

        return wrapper

    def in_the_body(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            kwargs.pop("timeout", None)
            return function(*args, **kwargs)

        return wrapper

    def through_a_helper(function):
        guarded = _with_timeout(function)

        def helper(*args, **kwargs):
            return guarded(*args, **kwargs)

        return helper

    def of_a_tool_that_names_nothing(function):
        def tool(url: str, timeout: int = 999999) -> int:  # no __wrapped__: it holds nothing read
            return function(url) if timeout else again(url)

        again = _with_timeout(tool)
        return tool

    return [
        ("a wrapper of a tool that names nothing as __wrapped__", of_a_tool_that_names_nothing),
        ("a plain inner function", _calling(named_nothing)),
        ("a lambda", _calling(lambda f: lambda *a, timeout=30, **k: f(*a, **k))),
        ("a callable object", _calling(Guard)),
        ("a bound method of an object that holds the function", _calling(lambda f: Guard(f).run)),
        ("a helper that calls the wrapper", _calling(through_a_helper)),
        (
            "held by an object that names it as __wrapped__",
            _calling(lambda f: Proxy(_with_timeout(f))),
        ),
        ("a subclass of a type of the closed set", _calling(lambda f: Partial(_with_timeout(f)))),
        ("in a container", in_a_container),
        ("in a module global", in_a_global),
        ("in a function attribute", in_a_function_attribute),
        ("in a class attribute", in_a_class_attribute),
        ("in a slot", in_a_slot),
        ("a class-based decorator whose own method keeps the argument", in_its_own_method),
        ("a wrapper that names nothing as __wrapped__", by_hand),
        ("an argument taken in the body alone", in_the_body),
    ]


@pytest.mark.parametrize("name, decorate", _limits(), ids=lambda v: v if type(v) is str else "")
def test_a_stated_limit_is_accepted(name, decorate):
    tool = decorate(_fetch())
    assert tool("u", timeout=5) == 999999  # the argument really is kept from the function
    reached_parameters(tool)  # and the walk does not find it


def test_the_arguments_a_partial_binds_are_a_stated_limit():
    """A ``functools.partial`` is read for the function it passes the call on to, not for
    what it binds: a keyword it fills, and a function it carries as an argument."""
    import functools

    def relay(function, *args, timeout=30, **kwargs):
        return function(*args, **kwargs)

    carried = _calling(lambda f: functools.partial(relay, f))(_fetch())
    assert carried("u", timeout=5) == 999999
    assert _levels(carried) == [_FORWARD, _FETCH]
    filled = _calling(lambda f: functools.partial(f, timeout=30))(_fetch())
    assert filled("u") == 30  # the function's own default, which a policy judged, is not used
    assert _levels(filled) == [_FORWARD, _FETCH]


# --- nothing held is run -------------------------------------------------------------------

_TOUCHED: List[str] = []


def _touch(name, result=None, raises=None):
    def hook(*args, **kwargs):
        _TOUCHED.append(name)
        if raises is not None:
            raise raises(name)
        return result

    return hook


def _hostile():
    """Held values of every hook kind: each records any code of its own that runs."""

    class Meta(type):
        __eq__ = _touch("metaclass __eq__", True)
        __hash__ = _touch("metaclass __hash__", 0)
        __instancecheck__ = _touch("__instancecheck__", True)
        __subclasscheck__ = _touch("__subclasscheck__", True)
        __getattribute__ = _touch("metaclass __getattribute__", raises=RuntimeError)

    hooks = {
        "__class__ property": {"__class__": property(_touch("__class__", raises=RuntimeError))},
        "__class__ property that answers": {"__class__": property(_touch("__class__", int))},
        "__getattr__": {"__getattr__": _touch("__getattr__", raises=RuntimeError)},
        "__getattr__ that answers": {"__getattr__": _touch("__getattr__", print)},
        "__getattr__ interrupted": {"__getattr__": _touch("__getattr__", raises=KeyboardInterrupt)},
        "__getattribute__": {"__getattribute__": _touch("__getattribute__", raises=RuntimeError)},
        "__dict__ property": {"__dict__": property(_touch("__dict__", raises=RuntimeError))},
        "__wrapped__ property": {"__wrapped__": property(_touch("__wrapped__", print))},
        "__wrapped__ property that fails": {
            "__wrapped__": property(_touch("__wrapped__", raises=AttributeError))
        },
        "__eq__ and __hash__": {
            "__eq__": _touch("__eq__", True),
            "__hash__": _touch("__hash__", 0),
        },
        "__bool__, __len__, __iter__": {
            "__bool__": _touch("__bool__", True),
            "__len__": _touch("__len__", 1),
            "__iter__": _touch("__iter__", raises=RuntimeError),
        },
        "__get__": {"__get__": _touch("__get__", print), "__set__": _touch("__set__")},
        "__call__": {"__call__": _touch("__call__", print)},
    }
    values = [(name, type("Held", (), dict(body))()) for name, body in hooks.items()]
    values.append(("metaclass hooks", Meta("Held", (), {})()))
    values.append(("class with metaclass hooks", Meta("Held", (), {})))

    def helper():
        return None

    class Key:
        __hash__ = _touch("key __hash__", hash("__wrapped__"))
        __eq__ = _touch("key __eq__", True)

    vars(helper)[Key()] = _with_timeout(helper)
    values.append(("a function whose __dict__ has a key that compares equal to any name", helper))
    return values


@pytest.mark.parametrize("name, value", _hostile(), ids=lambda v: v if type(v) is str else "")
def test_nothing_a_callable_holds_is_run(name, value):
    """A tool can hold anything: a lazy proxy that connects when touched, a remote stub
    that answers every attribute. The checks read a held value's type and nothing else of
    it, so none of its code runs: not ``__class__``, not an attribute hook, not a
    descriptor its class declares, not a comparison, not its metaclass."""
    import functools

    def search(query: str, limit: int = 5) -> str:
        return query

    search.label = object()  # an attribute for a class-based decorator to copy

    def tool(query: str, limit: int = 5) -> str:  # no decorator at all
        return query + str(type(value))

    class Tool:
        def __init__(self):
            self.held = value

        def __call__(self, query: str, limit: int = 5) -> str:
            return query

    class Decorator:
        def __init__(self, function):
            functools.update_wrapper(self, function)
            self.held = value

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    own = ["query", "limit"]
    del _TOUCHED[:]
    assert _levels(tool) == [own]
    assert unsupported_signature(tool) is None
    assert _levels(_holding(value, search)) == [_FORWARD, own]
    assert unsupported_signature(_holding(value, search)) is None
    assert _levels(functools.partial(_holding(value, search))) == [_FORWARD, own]
    assert _levels(Tool()) == [own, own]
    assert _levels(Decorator(search)) == [_FORWARD, _FORWARD, own]
    assert _TOUCHED == []


def test_the_verdict_is_the_same_with_and_without_the_c_extension_of_wrapt():
    """A wrapper a C extension builds is outside the closed set and is not read, so the
    same decorator gets the same verdict whichever way the library was built."""
    import importlib.util
    import os
    import subprocess

    if importlib.util.find_spec("wrapt") is None:
        pytest.skip("wrapt is not installed")
    script = (
        "import functools, wrapt\n"
        "from recusal.invocation import reached_parameters\n"
        "@wrapt.decorator\n"
        "def passing(wrapped, instance, args, kwargs):\n"
        "    return wrapped(*args, **kwargs)\n"
        "def calling(make):\n"
        "    def decorate(function):\n"
        "        guarded = make(function)\n"
        "        @functools.wraps(function)\n"
        "        def wrapper(*args, **kwargs):\n"
        "            return guarded(*args, **kwargs)\n"
        "        return wrapper\n"
        "    return decorate\n"
        "def keeping(function):\n"
        "    @functools.wraps(function)\n"
        "    def guarded(*args, timeout=30, **kwargs):\n"
        "        return function(*args, **kwargs)\n"
        "    return guarded\n"
        "def fetch(url: str, timeout: int = 5) -> int:\n"
        "    return timeout\n"
        "class Holder:\n"
        "    @passing\n"
        "    def fetch(self, url: str, timeout: int = 5) -> int:\n"
        "        return timeout\n"
        "shapes = [passing(fetch), calling(passing)(fetch), passing(calling(passing)(fetch)),\n"
        "          calling(lambda f: passing(keeping(f)))(fetch), passing(calling(keeping)(fetch)),\n"
        "          Holder().fetch]\n"
        "for shape in shapes:\n"
        "    try:\n"
        "        print([list(level) for level in reached_parameters(shape)])\n"
        "    except Exception as error:\n"
        "        print(type(error).__name__)\n"
    )
    root = str(pathlib.Path(__file__).resolve().parent.parent)

    def verdicts(pure):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
        env.pop("WRAPT_DISABLE_EXTENSIONS", None)
        if pure:
            env["WRAPT_DISABLE_EXTENSIONS"] = "true"
        done = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.splitlines()

    built, pure = verdicts(False), verdicts(True)
    assert built == pure
    assert len(built) == 6 and built[4] == "HiddenWrapper"


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
        _traced(Holder().find),
        lookup_async,
        _traced(lookup_async),
    ):
        with pytest.raises(DispatchedAtCallTime):
            reached_parameters(func)
    for func in (_traced(lookup), lookup_async):
        assert unsupported_signature(func) == _AT_CALL_TIME, func


def test_a_single_dispatch_method_with_a_registered_implementation_is_refused():
    """``functools.singledispatchmethod`` hands out a new function at each access. It has
    no registry of its own; where it keeps the way to one is found once, at import. What
    it hands out differs by Python (a function, or from 3.14 an object that answers
    ``__wrapped__`` through a property), so what is pinned here is relative to the walk
    without this check: a registered one is refused, a base-only one is walked exactly as
    it was, and the check runs no code the walk did not already run."""
    import functools

    class Recording(functools.singledispatchmethod):
        """A subclass, which records every attribute asked of it."""

        def __getattribute__(self, name):
            _TOUCHED.append(name)
            return super().__getattribute__(name)

    class Holder:
        @functools.singledispatchmethod
        def lookup(self, key: str, deep: bool = False) -> str:
            return "base"

        @lookup.register(int)
        def lookup_int(self, key: int, deep: bool = False, skip_review: bool = True) -> str:
            return f"int, skip_review={skip_review}"

        @functools.singledispatchmethod
        @classmethod
        def make(cls, key: str, deep: bool = False) -> str:
            return "base"

        @make.register(int)
        @classmethod
        def make_int(cls, key: int, deep: bool = False, skip_review: bool = True) -> str:
            return "int"

        @Recording
        def recorded(self, key: str, deep: bool = False) -> str:
            return "base"

        @recorded.register(int)
        def recorded_int(self, key: int, deep: bool = False, skip_review: bool = True) -> str:
            return "int"

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
        functools.lru_cache(maxsize=None)(Holder().lookup),
        Holder.make,
        Holder().recorded,
    ):
        _, before = _without_the_dispatch_check(func)
        del _TOUCHED[:]
        with pytest.raises(DispatchedAtCallTime):
            reached_parameters(func)
        # the descriptor is known by its type and read where it is stored: whatever of it
        # runs, the walk ran before this check existed, in the same order
        assert _TOUCHED == before[: len(_TOUCHED)], func
    assert unsupported_signature(Holder().lookup, skip=("self",)) == _AT_CALL_TIME

    # a subclass that answers ``__dict__`` itself cannot be read, so it is refused; its
    # property is not run, and neither is the ``__eq__`` of a key of its class
    class Key:
        __hash__ = _touch("key __hash__", hash("__dict__"))
        __eq__ = _touch("key __eq__", True)

    class Opaque(functools.singledispatchmethod):
        __dict__ = property(_touch("__dict__", {}))  # type: ignore[assignment]

    keyed = type("Keyed", (functools.singledispatchmethod,), {Key(): None})

    class Other:
        @Opaque
        def lookup(self, key: str, deep: bool = False) -> str:
            return "base"

        @lookup.register(int)
        def lookup_int(self, key: int, deep: bool = False, skip_review: bool = True) -> str:
            return "int"

        @keyed
        def find(self, key: str, deep: bool = False) -> str:
            return "base"

    opaque, find = Other().lookup, Other().find
    _, before = _without_the_dispatch_check(opaque)
    del _TOUCHED[:]
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(opaque)
    assert _TOUCHED == before[: len(_TOUCHED)]
    # with only the base registered the base is what runs: the walk, the verdict and the
    # code that runs are what they were without the check
    for func in (
        find,
        Holder().plain,
        Holder.plain,
        _traced(Holder().plain),
        functools.lru_cache(maxsize=None)(Holder().plain),
    ):
        walked, before = _without_the_dispatch_check(func)
        del _TOUCHED[:]
        assert _levels(func) == walked, func
        assert _TOUCHED == before, func
        assert walked[-1] == ["self", "key", "deep"]
    for skip in ((), ("self",)):
        real = invocation._refuse_call_time_dispatch
        invocation._refuse_call_time_dispatch = lambda func, wrapped: None
        try:
            judged = unsupported_signature(Holder().plain, skip)
        finally:
            invocation._refuse_call_time_dispatch = real
        assert unsupported_signature(Holder().plain, skip) == judged


def _without_the_dispatch_check(func):
    """The walk of ``func`` with the call-time-dispatch check left out, as the walk was
    before that check existed, and the names of the hooks of this file it ran."""
    real = invocation._refuse_call_time_dispatch
    invocation._refuse_call_time_dispatch = lambda func, wrapped: None
    del _TOUCHED[:]
    try:
        walked = _levels(func)
    except (TypeError, ValueError) as error:
        walked = type(error)
    finally:
        invocation._refuse_call_time_dispatch = real
    return walked, list(_TOUCHED)


def _registry_cell(dispatcher):
    """The closure cell that holds the real registry of a single-dispatch function."""
    dispatch = dispatcher.dispatch
    return dict(zip(dispatch.__code__.co_freevars, dispatch.__closure__))["registry"]


def test_a_replaced_base_is_refused_by_identity():
    """``register(object, ...)`` replaces the base implementation and leaves ``object`` the
    only key: the base alone runs only when that one entry is, by identity, the function
    ``__wrapped__`` names, under the key that is, by identity, ``object``."""
    import functools

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

    def dispatcher_with(entries, wrapped=search):
        built = functools.singledispatch(search)
        registry = _registry_cell(built).cell_contents
        registry.clear()
        registry.update(entries)
        built.__wrapped__ = wrapped
        return built

    class Alike(type):
        """A class that compares equal to ``object``, and anything else."""

        __eq__ = _touch("__eq__", True)
        __hash__ = _touch("__hash__", hash(object))

    assert _levels(dispatcher_with({object: search})) == [["args", "kw"], ["query", "limit"]]
    refused = (
        dispatcher_with({Alike("Alike", (), {}): search}),  # a key equal to object is not object
        dispatcher_with({object: search}, wrapped=Alike("Alike", (), {})),  # nor an equal entry
        dispatcher_with({object: search, int: print}),
        dispatcher_with({}),
    )
    del _TOUCHED[:]
    for func in refused:
        with pytest.raises(DispatchedAtCallTime):
            reached_parameters(func)
    assert _TOUCHED == []


def test_the_registry_read_is_the_real_one_not_the_attribute():
    """``functools.update_wrapper`` copies the base function's attributes onto the
    dispatcher, so a base that carries a ``registry`` or a ``dispatch`` of its own
    overwrites the dispatcher's. What is read is the registry in the closure, which nothing
    overwrites. One that is not a plain ``dict`` is refused, and none of its code runs."""
    import functools
    import types

    def base(key: str, deep: bool = False) -> str:
        return "base"

    base.registry, base.dispatch = {}, "sync"  # type: ignore[attr-defined]
    masked = functools.singledispatch(base)
    masked.register(int, lambda key, deep=False, skip_review=True: "int")
    assert masked.registry == {} and masked(1) == "int"
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(masked)

    # and a registry attribute that claims the base alone changes nothing
    claimed = _dispatching()
    claimed.registry = types.MappingProxyType({object: claimed.__wrapped__})
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(claimed)

    # nor does a ``dispatch`` attribute: the base carries the ``dispatch`` of another
    # dispatcher over the same base, which has only that base
    def shared(key: str, deep: bool = False) -> str:
        return "base"

    shared.dispatch = functools.singledispatch(shared).dispatch  # type: ignore[attr-defined]
    borrowed = functools.singledispatch(shared)
    borrowed.register(int, lambda key, deep=False, skip_review=True: "int")
    assert borrowed.dispatch is shared.dispatch and borrowed(1) == "int"  # type: ignore[attr-defined]
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(borrowed)

    def search(query: str, limit: int = 5) -> str:
        return query

    class Mapping(dict):
        items = _touch("items", raises=RuntimeError)
        keys = _touch("keys", raises=RuntimeError)
        __iter__ = _touch("__iter__", raises=RuntimeError)
        __len__ = _touch("__len__", 1)
        __getitem__ = _touch("__getitem__", search)
        __contains__ = _touch("__contains__", True)

    odd = functools.singledispatch(search)
    _registry_cell(odd).cell_contents = Mapping({object: search})
    del _TOUCHED[:]
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(odd)
    assert unsupported_signature(odd) == _AT_CALL_TIME
    assert _TOUCHED == []


_IMPORTED_AFTER = """
import functools, json, sys
REAL, REAL_METHOD, RAN = functools.singledispatch, functools.singledispatchmethod, []
scenario = sys.argv[1]
if scenario == "wrapped":
    def trace(function):
        @functools.wraps(function)
        def traced(*args, **kwargs):
            RAN.append("traced")
            return function(*args, **kwargs)
        return traced
    functools.singledispatch = trace(REAL)
elif scenario == "object":
    class Odd:
        @property
        def __code__(self):
            RAN.append("__code__")
            raise RuntimeError("no code here")
        def __getattr__(self, name):
            RAN.append(name)
            raise RuntimeError(name)
        def __call__(self, function):
            return REAL(function)
    functools.singledispatch = Odd()
elif scenario == "method":
    class Recording(REAL_METHOD):
        def __getattribute__(self, name):
            RAN.append(name)
            return super().__getattribute__(name)
        def __get__(self, obj, cls=None):
            RAN.append("__get__")
            raise KeyboardInterrupt("no")
    functools.singledispatchmethod = Recording
elif scenario == "method-subclass":
    class Inheriting(REAL_METHOD):
        def __getattribute__(self, name):
            RAN.append(name)
            return super().__getattribute__(name)
    functools.singledispatchmethod = Inheriting
elif scenario == "reimplemented":
    def singledispatch(func):
        RAN.append("singledispatch")
        registry = {object: func}
        def dispatch(cls):
            return registry[object]
        def register(cls, func=None):
            if func is None:
                return lambda function: register(cls, function)
            registry[cls] = func
            return func
        def wrapper(*args, **kw):
            return dispatch(args[0].__class__)(*args, **kw)
        wrapper.register, wrapper.dispatch, wrapper.registry = register, dispatch, registry
        return functools.update_wrapper(wrapper, func)
    functools.singledispatch = singledispatch
from recusal import invocation
ran_at_import = list(RAN)
def base(key: str, deep: bool = False) -> str:
    return "base"
def verdict(func):
    try:
        invocation.reached_parameters(func)
        return "walked"
    except Exception as error:
        return type(error).__name__
registered = REAL(base)
registered.register(int, lambda key, deep=False, skip_review=True: "int")
class Holder:
    @REAL_METHOD
    def plain(self, key: str, deep: bool = False) -> str:
        return "base"
    @REAL_METHOD
    def lookup(self, key: str, deep: bool = False) -> str:
        return "base"
    @lookup.register(int)
    def lookup_int(self, key: int, deep: bool = False, skip_review: bool = True) -> str:
        return "int"
print(json.dumps({
    "ran": ran_at_import,
    "found": invocation._WRAPPER_CODE is not None,
    "method": invocation._METHOD_ACCESS is not None,
    "registered": verdict(registered),
    "base_only": verdict(REAL(base)),
    "method_registered": verdict(Holder().lookup),
    "method_base_only": verdict(Holder().plain),
}))
"""


@pytest.mark.parametrize(
    "scenario", ["none", "wrapped", "object", "reimplemented", "method", "method-subclass"]
)
def test_importing_the_module_runs_nothing_whatever_stands_in_functools(scenario):
    """A process can wrap or replace ``functools.singledispatch`` before this module is
    imported. The import then runs none of that code and does not raise; the dispatcher's
    code is not identified, and the dispatch check is off: it refuses nothing, so a
    base-only single-dispatch method is walked as before, and a registered dispatcher is
    not found (a stated limit). A replaced ``singledispatchmethod`` turns off only the
    recognition of single-dispatch methods."""
    import json
    import os
    import subprocess

    root = str(pathlib.Path(__file__).resolve().parent.parent)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run(
        [sys.executable, "-c", _IMPORTED_AFTER, scenario],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    assert seen["ran"] == []
    assert seen["base_only"] == seen["method_base_only"] == "walked"
    expected = {
        "none": (True, True, "DispatchedAtCallTime", "DispatchedAtCallTime"),
        "wrapped": (False, False, "walked", "walked"),
        "object": (False, False, "walked", "walked"),
        "reimplemented": (False, False, "walked", "walked"),
        "method": (True, False, "DispatchedAtCallTime", "walked"),
        "method-subclass": (True, False, "DispatchedAtCallTime", "walked"),
    }[scenario]
    assert (
        seen["found"],
        seen["method"],
        seen["registered"],
        seen["method_registered"],
    ) == expected


def test_the_single_dispatch_this_python_builds_is_recognised():
    """The dispatcher is known by the code object of the ``wrapper`` that
    ``functools.singledispatch`` defines, and its registry by the closure of its
    ``dispatch``. If a Python builds it another way the check finds nothing there: this
    test fails first, loudly, so that is a decision and not a silence."""
    import functools

    def search(query: str, limit: int = 5) -> str:
        return query

    built = functools.singledispatch(search)
    assert invocation._WRAPPER_CODE is not None and built.__code__ is invocation._WRAPPER_CODE
    assert built.dispatch.__code__ is invocation._DISPATCH_CODE
    assert invocation._cell(built, "dispatch") is built.dispatch
    registry = invocation._cell(built.dispatch, "registry")
    assert type(registry) is dict and list(registry.items()) == [(object, search)]
    assert invocation._only_runs(built.dispatch, search)
    assert not invocation._only_runs(built.dispatch, built)
    assert not invocation._only_runs(search, search)


def test_where_a_single_dispatch_method_keeps_its_registry_is_found():
    """Found at import by looking at one built there. It differs by Python: a closure cell
    holding the descriptor (3.9 to 3.12) or the ``dispatch`` function (3.13), an instance
    attribute holding the descriptor (3.14, 3.15). Not finding it is a failure on every
    Python, never a skip: recognising a single-dispatch method would be lost silently."""
    import functools

    assert invocation._METHOD_ACCESS is not None, "single-dispatch methods are no longer read"
    kind, code, name, is_descriptor = invocation._METHOD_ACCESS

    class Holder:
        @functools.singledispatchmethod
        def lookup(self, key: str, deep: bool = False) -> str:
            return "base"

        @functools.singledispatchmethod
        @classmethod
        def make(cls, key: str, deep: bool = False) -> str:
            return "base"

        @functools.singledispatchmethod
        @staticmethod
        def plain(key: str, deep: bool = False) -> str:
            return "base"

    handed, descriptor = Holder().lookup, vars(Holder)["lookup"]
    assert type(handed) is kind
    if code is None:
        kept = invocation._entry(invocation._instance_attributes(handed), name)
    else:
        assert handed.__code__ is code
        kept = invocation._cell(handed, name)
    assert kept is (descriptor if is_descriptor else descriptor.dispatcher.dispatch)
    found = invocation._dispatcher(handed, getattr(handed, "__wrapped__", None))
    assert found is not None and found[0] is descriptor.dispatcher.dispatch
    assert invocation._only_runs(*found)
    # with only its base, each flavour is read as before, which is not always a walk:
    # never a refusal as dispatch (on Python 3.9 the function handed out for a classmethod
    # names the function inside it as ``__wrapped__``, not the classmethod the registry has)
    for func in (Holder().lookup, Holder.make, Holder().make, Holder.plain, Holder().plain):
        try:
            reached_parameters(func)
        except DispatchedAtCallTime:
            raise
        except (TypeError, ValueError):
            pass


def test_what_only_looks_like_a_dispatcher_is_walked():
    """Attributes do not make a dispatcher: ``functools.update_wrapper`` copies
    ``registry``, ``dispatch`` and ``register`` onto every wrapper of one. The check is made
    on the function ``functools.singledispatch`` itself returned, and on nothing held."""
    import functools
    import types

    def search(query: str, limit: int = 5) -> str:
        return query

    own = ["query", "limit"]
    entries = {object: search, int: print}
    everything = dict(registry=types.MappingProxyType(entries), dispatch=print, register=print)

    carrying = _traced(search)
    vars(carrying).update(everything)
    assert _levels(carrying) == [_FORWARD, own]
    cached = functools.lru_cache(maxsize=None)(search)
    vars(cached).update(everything)
    assert _levels(cached) == [own, own]

    class Router:
        """A callable object with all three."""

        def __init__(self):
            vars(self).update(everything)

        def __call__(self, query: str, limit: int = 5) -> str:
            return query

        def __getattr__(self, name):
            _TOUCHED.append(name)
            raise AttributeError(name)

    del _TOUCHED[:]
    assert _levels(Router()) == [own, own]
    assert "registry" not in _TOUCHED and "dispatch" not in _TOUCHED

    # every wrapper of a dispatcher with only its base carries the copy, and is walked
    base_only = functools.singledispatch(search)
    dispatch = [["args", "kw"], own]

    class Decorator:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    class Service:
        find = base_only

    bound = functools.update_wrapper(functools.partial(base_only), base_only)
    assert _levels(_traced(base_only)) == [_FORWARD, *dispatch]
    assert _levels(functools.lru_cache(maxsize=None)(base_only)) == [["args", "kw"], *dispatch]
    assert _levels(_traced(functools.lru_cache(maxsize=None)(base_only))) == [
        _FORWARD,
        ["args", "kw"],
        *dispatch,
    ]
    assert _levels(_traced(Decorator(base_only))) == [_FORWARD, _FORWARD, _FORWARD, *dispatch]
    assert _levels(_traced(bound)) == [_FORWARD, *dispatch]
    assert _levels(_traced(Service().find)) == [_FORWARD, ["args", "kw"], ["limit"]]
    # and the same wrappers of one with an implementation registered are refused, when the
    # walk reaches the dispatcher itself
    lookup = _dispatching()
    for wrapper in (_traced(Decorator(lookup)), _traced(functools.partial(lookup))):
        with pytest.raises(DispatchedAtCallTime):
            reached_parameters(wrapper)

    # a dispatcher, or a single-dispatch method, that is merely held is not on the way
    def closing(query: str, limit: int = 5) -> str:
        return query + lookup(1)

    class Tool:
        @functools.singledispatchmethod
        def render(self, value):
            return "base"

        @render.register(int)
        def render_int(self, value, wide=True):
            return "int"

        def __init__(self):
            self.lookup = lookup
            self.dispatch = lookup.dispatch
            self.descriptor = vars(Tool)["render"]

        def __call__(self, query: str, limit: int = 5) -> str:
            return query

    class Decorated(Tool):
        def __init__(self, function):
            Tool.__init__(self)
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    descriptor = vars(Tool)["render"]

    def holding_the_descriptor(query: str, limit: int = 5) -> str:
        return query + str(descriptor) + str(lookup.dispatch)

    assert _levels(closing) == [own]
    assert _levels(holding_the_descriptor) == [own]
    assert _levels(_holding(lookup, search)) == [_FORWARD, own]
    assert _levels(_holding(descriptor, search)) == [_FORWARD, own]
    assert _levels(Tool()) == [own, own]
    assert _levels(Decorated(search)) == [_FORWARD, _FORWARD, own]

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

    assert _levels(plain) == [_FORWARD, ["key", "deep"]]

    # a dispatcher anywhere on the held way counts, not only the last wrapper on it
    def base(key: str, deep: bool = False) -> str:
        return "base"

    over_a_tracer = functools.singledispatch(_traced(base))
    over_a_tracer.register(int, lambda key, deep=False, skip_review=True: "int")

    @functools.wraps(base)
    def deep(*args, **kwargs):
        return over_a_tracer(*args, **kwargs)

    assert deep(1) == "int"
    with pytest.raises(HiddenWrapper):
        reached_parameters(deep)

    # and so does the function a registered single-dispatch method hands out, where it is
    # a plain function (from Python 3.14 it is an object, which a callable is not read for)
    class Holder:
        @functools.singledispatchmethod
        def lookup(self, key: str, deep: bool = False) -> str:
            return "base"

        @lookup.register(int)
        def lookup_int(self, key: int, deep: bool = False, skip_review: bool = True) -> str:
            return "int"

        @functools.singledispatchmethod
        def plain(self, key: str, deep: bool = False) -> str:
            return "base"

    def holding_the_method(handed, descriptor):
        @functools.wraps(descriptor.func)
        def through(*args, **kwargs):
            return handed(*args, **kwargs)

        return through

    holder = Holder()
    registered = holding_the_method(holder.lookup, vars(Holder)["lookup"])
    only_base = holding_the_method(holder.plain, vars(Holder)["plain"])
    assert registered(1) == "int"
    reached_parameters(only_base)
    if invocation._METHOD_ACCESS[1] is not None:
        with pytest.raises(HiddenWrapper):
            reached_parameters(registered)
    else:
        reached_parameters(registered)  # a stated limit on this Python


def test_dispatch_the_walk_does_not_find_is_accepted():
    """The stated limits of the dispatch check, each accepted on purpose."""
    import functools

    lookup = _dispatching()

    def by_hand(key: str, deep: bool = False) -> str:  # a dispatch written in the body
        return lookup(int(key), deep) if key.isdigit() else lookup(key, deep)

    def relaying(dispatcher):
        relay = lambda *args, **kwargs: dispatcher(*args, **kwargs)  # noqa: E731

        @functools.wraps(dispatcher.__wrapped__)
        def through_a_lambda(*args, **kwargs):
            return relay(*args, **kwargs)

        return through_a_lambda

    through_a_lambda = relaying(lookup)
    assert through_a_lambda(1) == "int, skip_review=True"
    later = functools.singledispatch(lookup.__wrapped__)
    for func in (by_hand, through_a_lambda, later):
        reached_parameters(func)
    later.register(int, lambda key, deep=False, skip_review=True: "int")  # after the walk
    assert later(1) == "int"


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
