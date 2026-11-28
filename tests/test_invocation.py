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
    PartialMethodSignature,
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


_DECLARED = "it declares a signature that is not its own"


class _Shows:
    """A class-based decorator that declares, on the CLASS, the signature of the function
    it holds less ``skip_review``, and names nothing as ``__wrapped__``."""

    def __init__(self, function):
        self.function = function
        self.__annotations__ = {"query": str, "limit": int}

    @property
    def __signature__(self):
        import inspect

        shown = inspect.signature(self.function)
        kept = [p for p in shown.parameters.values() if p.name != "skip_review"]
        return shown.replace(parameters=kept)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _ShowsBound(_Shows):
    def __get__(self, obj, cls=None):
        import types

        return self if obj is None else types.MethodType(self, obj)


def _service(attribute):
    return type("Service", (), {"fetch": attribute, "__hash__": lambda self: 1})()


def test_a_signature_declared_on_the_class_of_what_a_method_binds_is_refused():
    """A bound method shows the attributes of what it binds as its own, but not what the
    class of that declares, and ``inspect.signature`` reads a method of a callable
    object as the object: its class's ``__signature__``, less the receiver. So an object
    that declares a signature and is bound as a method (by its own ``__get__``, by
    ``types.MethodType``, by a ``classmethod``, under a ``functools.wraps`` wrapper) was
    read as the signature it declared: judged ``{'query', 'limit'}``, ran the
    ``skip_review=True`` of the function it holds. 0.15.0 accepted these."""
    import functools
    import types

    ran = []

    def method(self, query: str, limit: int = 10, skip_review: bool = True) -> str:
        ran.append(skip_review)
        return "rows"

    def twice(first, second, query: str, limit: int = 10, skip_review: bool = True) -> str:
        ran.append(skip_review)
        return "rows"

    class OnCall:  # the signature is declared on its ``__call__``, not on the class
        def __init__(self, function):
            self.function = function
            self.__annotations__ = {"query": str, "limit": int}

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    def shown(self, receiver, query: str, limit: int = 10) -> str:
        return "rows"

    OnCall.__call__.__signature__ = __import__("inspect").signature(shown)  # type: ignore[attr-defined]
    wrapper = _traced(_Shows(method))
    # ``functools.wraps`` copies ``__annotations__`` up to Python 3.13 and ``__annotate__``
    # from 3.14, which an object has not: given here, so the reason is the same on all
    wrapper.__annotations__ = {"query": str, "limit": int}
    tools = {
        "its own __get__": _service(_ShowsBound(method)).fetch,
        "types.MethodType": types.MethodType(_Shows(method), _service(None)),
        "a classmethod": _service(classmethod(_Shows(method))).fetch,
        "under functools.wraps": _service(wrapper).fetch,
        "a method of a method": types.MethodType(types.MethodType(_Shows(twice), "a"), "b"),
    }
    for name, tool in tools.items():
        del ran[:]
        assert list(inspect_signature(tool)) == ["query", "limit"], name
        assert tool(query="q", limit=5) == "rows" and ran == [True], name  # no policy saw it
        assert unsupported_signature(tool) == _DECLARED, name
        with pytest.raises(DeclaredSignature):
            reached_parameters(tool)
    # behind a ``functools.partial``, where the walk refused the object itself before
    with pytest.raises(DeclaredSignature):
        reached_parameters(functools.partial(tools["its own __get__"]))
    # a signature declared on the object's ``__call__`` was refused before too, when the
    # walk reached that method: it is refused at the object now
    with pytest.raises(DeclaredSignature):
        invocation.own_parameters(types.MethodType(OnCall(method), _service(None)))
    # under a cache, which 0.15.0 refused when the cache is named for the object, and
    # accepted when it is named for the function: that one runs clean, declares
    # ``skip_review`` and is refused all the same, because the object it calls declares
    # a signature that is not its own
    cached = functools.lru_cache(maxsize=None)(_Shows(method))
    cached.__annotations__ = {"query": str, "limit": int}
    assert unsupported_signature(_service(cached).fetch) == _DECLARED
    named_again = functools.update_wrapper(
        functools.lru_cache(maxsize=None)(_Shows(method)), method
    )
    named_again.__annotations__ = dict(method.__annotations__)  # on every Python
    clean = _service(named_again).fetch
    assert list(inspect_signature(clean)) == ["query", "limit", "skip_review"]
    assert unsupported_signature(clean) == _DECLARED


def test_a_method_that_binds_nothing_which_declares_a_signature_is_read_as_before():
    """The controls of the test above: what a method binds is looked at, and only for a
    signature it declares. Not the receiver, and not a wrapper more on the way."""
    import functools
    import inspect
    import types

    def fetch(self, query: str, limit: int = 10) -> str:
        return "rows"

    def named(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            return function(self, *args, **kwargs)

        return wrapper

    class Passing:  # a class-based decorator that declares nothing
        def __init__(self, function):
            functools.update_wrapper(self, function)
            self.function = function

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    # the RECEIVER's class declares a signature, as a model class does
    Model = type(
        "Model",
        (),
        {
            "__signature__": inspect.Signature(),
            "__hash__": lambda self: 1,
            "fetch": fetch,
            "traced": _traced(fetch),
        },
    )
    own = ["query", "limit"]
    for tool, levels in (
        (_service(fetch).fetch, [own]),
        (_service(_traced(fetch)).fetch, [_FORWARD, own]),
        (_service(named(fetch)).fetch, [_FORWARD, own]),
        (_service(classmethod(fetch)).fetch, [own]),
        (_service(functools.lru_cache(maxsize=None)(fetch)).fetch, [own, own]),
        (types.MethodType(Passing(fetch), _service(None)), [_FORWARD, own, _FORWARD]),
        (Model().fetch, [own]),
        (Model().traced, [_FORWARD, own]),
    ):
        assert tool("q") == "rows"
        assert unsupported_signature(tool) is None, tool
        assert sorted(_levels(tool)) == sorted(levels), tool
    # a method of a method is not a wrapper more: ten deep is read as one is

    def many(*receivers, query: str = "q") -> str:
        return "rows"

    deep = many
    for _ in range(10):
        deep = types.MethodType(deep, "receiver")
    assert deep() == "rows"
    assert list(invocation.own_parameters(deep)) == ["receivers", "query"]
    assert _levels(deep) == [["receivers", "query"]]


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

    class Derived(Resilient):
        """Its instance dictionary is the accessor of the class it inherits from."""

    in_an_attribute = Resilient(_fetch())
    assert in_an_attribute("u", timeout=5) == 999999
    with pytest.raises(HiddenWrapper):
        reached_parameters(Derived(_fetch()))
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
    # a single-dispatch function is not one of them, with nothing registered either
    with pytest.raises(invocation.OpenDispatch):
        reached_parameters(dispatched)
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

    # a name a wrapper the walk reads keeps counts against a held wrapper that takes it too:
    # the finding involves the held one, whichever of the two is the forwarding wrapper
    def keeping_and_holding(function):
        copied = _copying(function)

        @functools.wraps(function)
        def wrapper(*args, limit=7, **kwargs):
            return copied(*args, **kwargs)

        return wrapper

    with pytest.raises(HiddenWrapper, match="`limit` is taken by a wrapper"):
        reached_parameters(keeping_and_holding(search))
    assert unsupported_signature(_copying(search)) is None
    assert "`verbose` is a parameter" in unsupported_signature(_calling(_renaming)(search))
    assert "`limit` is taken by a wrapper" in unsupported_signature(_calling(keeping_limit)(search))
    assert unsupported_signature(_calling(with_context)(search)) is not None
    assert unsupported_signature(_calling(with_context)(search), ("tool_context",)) is None


def test_a_held_wrapper_of_a_method_receives_the_receiver_the_method_does():
    """A method's decorator is usually written ``def wrapper(self, *args, **kwargs)``. When
    the tool is the bound method, the walk reads its levels without the receiver; a wrapper
    its decorator holds is passed the same receiver, so its first positional parameter is
    not one of its own either. Held and read agree here too. Since 0.17.1 a wrapper that
    takes the receiver under another name than the method has for it (``receiver`` over
    ``self``) is refused, read or held: ``_receiving`` keeps it, and a call that passes the
    arguments by position runs the method with each one place from where it was judged."""
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

    # the name each case's function has for its receiver, and the name each wrapper takes
    # it under (None: in ``*args``)
    receivers = ("self", "cls", "cls", "receiver")
    for make, keeps, taken in (
        (_receiving, False, "receiver"),
        (_traced, False, None),
        (_copying_method, False, "self"),
        (_keeping_method, True, "receiver"),
    ):
        for read, kept, receiver in zip(cases(make), cases(held(make)), receivers):
            if taken not in (None, receiver):
                # taken under another name than the function has for it: refused either
                # way, read or held, whether the wrapper keeps it or hands it on
                with pytest.raises(invocation.BoundArgumentKept, match=f"`{taken}`"):
                    reached_parameters(read)
                with pytest.raises(HiddenWrapper):
                    reached_parameters(kept)
                continue
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
    assert service_tool("q", 5) == 5  # by position: ``self='q'``, ``query=5``
    assert "`receiver` is a leading parameter" in unsupported_signature(service_tool)
    assert _levels(cases(held(_traced))[0]) == [_FORWARD, ["query", "limit"]]
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


def test_what_only_passes_for_a_partial_is_not_counted(monkeypatch):
    """The walk takes for a ``functools.partial`` whatever ``isinstance`` says is one, as it
    always did. The arguments bound are counted only for an instance of the standard
    library's type, found at the first walk. Where another class was put in ``functools``
    afterwards, an instance of it is walked as before and its bound arguments are not
    known: what is held behind it is not judged, through a method too. A stated limit."""
    import functools
    import types

    assert invocation._prepare()
    real = functools.partial

    class Lookalike:
        def __init__(self, func, *args, **keywords):
            self.func, self.args, self.keywords = func, args, keywords

        def __call__(self, *args, **keywords):
            return self.func(*self.args, *args, **dict(self.keywords, **keywords))

    def search(query: str, limit: int = 5) -> str:
        return query

    def method(self, query: str, limit: int = 5) -> str:
        return query

    hidden = _calling(_with_timeout)(search)
    kept = _calling_held(method, _wrapper_taking(method, "me, force=True, *args, **kwargs"))
    for tool in (real(hidden), real(kept, object()), real(types.MethodType(kept, object()))):
        with pytest.raises(HiddenWrapper):
            reached_parameters(tool)
    monkeypatch.setattr(functools, "partial", Lookalike)
    reached_parameters(Lookalike(hidden))
    reached_parameters(Lookalike(kept, object()))
    reached_parameters(Lookalike(types.MethodType(kept, object())))


def _wrapper_taking(function, text):
    """A ``functools.wraps`` wrapper of ``function`` whose parameters are ``text``."""
    import functools

    scope: Dict[str, Any] = {}
    exec("def inner(" + text + "):\n    return None\n", scope)
    return functools.wraps(function)(scope["inner"])


def _calling_held(function, held):
    """A wrapper of ``function`` that passes everything on to ``held``, kept in its closure."""
    import functools

    @functools.wraps(function)
    def outer(*args, **kwargs):
        return held(*args, **kwargs)

    return outer


def _receiver_forms():
    """Every way a held wrapper comes to receive two leading arguments bound, or one, as
    (name, the number bound, a builder taking the held wrapper's parameters)."""
    import functools
    import types

    def two(self, other, query: str, limit: int = 5) -> str:
        return query

    def one(self, query: str, limit: int = 5) -> str:
        return query

    def plain(query: str, limit: int = 5) -> str:
        return query

    def held_by(function, text):
        return _calling_held(function, _wrapper_taking(function, text))

    first, second = object(), object()
    return [
        (
            "a method of a method",
            2,
            lambda text: types.MethodType(types.MethodType(held_by(two, text), first), second),
        ),
        (
            "a partial of a bound method",
            2,
            lambda text: functools.partial(types.MethodType(held_by(two, text), first), 1),
        ),
        (
            "a bound method of a partial",
            2,
            lambda text: types.MethodType(functools.partial(held_by(two, text), 1), first),
        ),
        (
            "a partial with two arguments",
            2,
            lambda text: functools.partial(held_by(two, text), first, 1),
        ),
        (
            "a held partial with two arguments",
            2,
            lambda text: _calling_held(
                plain, functools.partial(_wrapper_taking(plain, text), 1, 2)
            ),
        ),
        (
            "a held partial of a held partial",
            2,
            lambda text: _calling_held(
                plain,
                functools.partial(functools.partial(_wrapper_taking(plain, text), 1), 2),
            ),
        ),
        (
            "a held bound method",
            1,
            lambda text: _calling_held(
                plain, types.MethodType(_wrapper_taking(plain, text), first)
            ),
        ),
        (
            "a held bound method under a bound method",
            2,
            lambda text: types.MethodType(
                _calling_held(one, types.MethodType(_wrapper_taking(one, text), first)), second
            ),
        ),
        ("a bound method", 1, lambda text: types.MethodType(held_by(one, text), first)),
    ]


@pytest.mark.parametrize(
    "name, bound, build", _receiver_forms(), ids=lambda v: v if type(v) is str else ""
)
def test_a_held_wrapper_is_passed_every_argument_bound_on_the_way(name, bound, build):
    """A held wrapper receives every leading argument bound before it: one for each method
    the walk went through and each bound method that holds it, and each positional argument
    of each ``functools.partial``, on the walk and held. Exactly that many of its leading
    parameters are not its own: one fewer and its next parameter would be taken for a
    receiver and a wrapper with a default of its own accepted, one more and a wrapper that
    only names its receivers refused. The receivers are named as the function the held
    wrapper wraps names them: since 0.17.1 a held wrapper that names one otherwise is
    refused where a call by position would run shifted
    (``test_x_a_held_wrapper_that_names_a_receiver_otherwise_is_refused``)."""
    receivers = ", ".join(["self", "other"][:bound])
    reached_parameters(build(receivers + ", *args, **kwargs"))
    # the parameter right after the receivers is the wrapper's own
    with pytest.raises(HiddenWrapper, match="`force` is a parameter"):
        reached_parameters(build(receivers + ", force=True, *args, **kwargs"))
    with pytest.raises(HiddenWrapper, match="`query` is taken by a wrapper"):
        reached_parameters(build(receivers + ", query, *args, **kwargs"))
    # and so is a keyword-only one, wherever the receivers went
    with pytest.raises(HiddenWrapper, match="`force` is a parameter"):
        reached_parameters(build("*args, force=True, **kwargs"))
    # one receiver fewer named: what follows is still not the wrapper's own
    if bound == 2:
        reached_parameters(build("self, *args, **kwargs"))


def _receiver_witnesses():
    """Hand-written cases from an independent review, each with what really runs: (name,
    whether it is accepted, a builder of the tool)."""
    import functools
    import types

    def method(self, query: str, limit: int = 5) -> str:
        return query

    def two(self, other, query: str, limit: int = 5) -> str:
        return query

    def plain(query: str, limit: int = 5) -> str:
        return query

    def held_by(function, text):
        return _calling_held(function, _wrapper_taking(function, text))

    bind, part, this = types.MethodType, functools.partial, object()
    return [
        (
            "a bound method; held (me, force=True, *a, **k): force is its own",
            False,
            lambda: bind(held_by(method, "me, force=True, *a, **k"), this),
        ),
        (
            "partial(g, a, b); held (self, other, *a, **k): both are bound",
            True,
            lambda: part(held_by(two, "self, other, *a, **k"), this, 1),
        ),
        (
            "partial(g) with no arguments; held (force=True, *a, **k)",
            False,
            lambda: part(held_by(plain, "force=True, *a, **k")),
        ),
        (
            "partial(g, limit=3); held (force=True, *a, **k): a keyword binds no position",
            False,
            lambda: part(held_by(plain, "force=True, *a, **k"), limit=3),
        ),
        (
            "a method of a method; held (self, other, *a, **k): two receivers",
            True,
            lambda: bind(bind(held_by(two, "self, other, *a, **k"), this), this),
        ),
        (
            "a plain tool; held bound method of (me, force=True, *a, **k)",
            False,
            lambda: _calling_held(
                plain, bind(_wrapper_taking(plain, "me, force=True, *a, **k"), this)
            ),
        ),
        (
            "partial(bound method, 1); held (self, other, *a, **k)",
            True,
            lambda: part(bind(held_by(two, "self, other, *a, **k"), this), 1),
        ),
        (
            "a bound method of partial(g, 1); held (self, other, *a, **k)",
            True,
            lambda: bind(part(held_by(two, "self, other, *a, **k"), 1), this),
        ),
        (
            "a plain tool; held partial((x, y, *a, **k), 1, 2)",
            True,
            lambda: _calling_held(plain, part(_wrapper_taking(plain, "x, y, *a, **k"), 1, 2)),
        ),
        (
            "a plain tool; held partial((x, y, force=True, *a, **k), 1, 2)",
            False,
            lambda: _calling_held(
                plain, part(_wrapper_taking(plain, "x, y, force=True, *a, **k"), 1, 2)
            ),
        ),
        (
            "a bound method; held partial((self, *a, **k)) with no arguments",
            True,
            lambda: bind(
                _calling_held(method, part(_wrapper_taking(method, "self, *a, **k"))), this
            ),
        ),
        (
            "a bound method; held (*a, force=True, **k): keyword-only is its own",
            False,
            lambda: bind(held_by(method, "*a, force=True, **k"), this),
        ),
        (
            "a bound method; held (self, *a, **k)",
            True,
            lambda: bind(held_by(method, "self, *a, **k"), this),
        ),
    ]


@pytest.mark.parametrize(
    "name, accepted, build", _receiver_witnesses(), ids=lambda v: v if type(v) is str else ""
)
def test_the_receivers_of_a_held_wrapper_are_counted_exactly(name, accepted, build):
    """The bound arguments are counted exactly at every step: one for a method, the number
    of positional arguments for a partial (none for its keywords), added to what was bound
    before, on the walk and along what is held."""
    if accepted:
        reached_parameters(build())
    else:
        with pytest.raises(HiddenWrapper, match="`force` is a parameter"):
            reached_parameters(build())


def test_the_arguments_of_a_partial_subclass_are_read_as_before():
    """The arguments a ``functools.partial`` binds are counted through the C-level accessor
    of the standard library's type, for a subclass too: that is what the partial really
    passes, and no attribute of the object is looked up for it. So a subclass is read by
    attribute exactly as the walk always read it, once for each callable behind it: a
    property there is not called once more, what it answers need not have a length, and
    it does not decide the count. What is held behind a subclass is judged as behind a
    ``functools.partial``."""
    import functools
    import types

    def search(query: str, limit: int = 5) -> str:
        return query

    def method(self, query: str, limit: int = 5) -> str:
        return query

    read = []

    class Counting(functools.partial):
        @property
        def args(self):
            read.append("args")
            return iter(())

    class Claiming(functools.partial):
        args = ("one",)  # it binds nothing

    class Plain(functools.partial):
        pass

    assert _levels(Counting(search)) == [["query", "limit"]]
    assert read == ["args"]  # by ``inspect.signature``, for the partial's own parameters
    del read[:]
    assert _levels(Counting(_traced(search))) == [_FORWARD, ["query", "limit"]]
    assert read == ["args", "args"]  # and once by the walk, for the function behind it

    hidden = _calling(_with_timeout)(search)
    kept = _calling_held(method, _wrapper_taking(method, "me, force=True, *args, **kwargs"))
    passing = _calling_held(method, _wrapper_taking(method, "self, *args, **kwargs"))
    for kind in (functools.partial, Plain, Counting):
        del read[:]
        with pytest.raises(HiddenWrapper, match="`timeout` is a parameter"):
            reached_parameters(kind(hidden))
        with pytest.raises(HiddenWrapper, match="`force` is a parameter"):
            reached_parameters(kind(kept, object()))
        with pytest.raises(HiddenWrapper, match="`force` is a parameter"):
            reached_parameters(kind(types.MethodType(kept, object())))
        reached_parameters(kind(types.MethodType(passing, object())))
        assert len(read) == (12 if kind is Counting else 0)
    # nothing is really bound, so the wrapper's first parameter is its own: what ``args``
    # claims is not what is counted
    with pytest.raises(HiddenWrapper, match="`me` is a parameter"):
        reached_parameters(Claiming(kept))
    reached_parameters(Plain(passing, object()))


def test_a_held_wrapper_that_cannot_take_what_is_bound_is_a_stated_limit():
    """A held wrapper with fewer positional parameters than the arguments bound on the way
    cannot be called with them at all: the call fails when it is made. With no parameter of
    its own it is accepted, as before the check."""
    import types

    def method(self, query: str, limit: int = 5) -> str:
        return query

    for text in ("", "**kwargs", "*, query=None, limit=5"):
        tool = types.MethodType(_calling_held(method, _wrapper_taking(method, text)), object())
        reached_parameters(tool)
        with pytest.raises(TypeError):
            tool("x")
    with pytest.raises(HiddenWrapper, match="`force` is a parameter"):
        reached_parameters(
            types.MethodType(
                _calling_held(method, _wrapper_taking(method, "*, force=True")), object()
            )
        )


def test_the_one_shape_judged_differently_held_and_read():
    """Under a tool that is a ``functools.partial`` binding a keyword, a held wrapper that
    takes that name and ``*args`` but no ``**kwargs`` is refused, and the same wrapper read
    by the walk is accepted: ``inspect.signature`` of such a partial no longer shows the
    ``*args``, so the walk does not see a wrapper that passes arguments on. The call fails
    either way. Stated where the rule is stated; with ``**kwargs`` the two agree."""
    import functools
    import inspect

    def search(query: str, limit: int = 5) -> str:
        return query

    def tools(text):
        wrapper = _wrapper_taking(search, text)
        return (
            functools.partial(wrapper, limit=3),
            functools.partial(_calling_held(search, wrapper), limit=3),
        )

    def verdict(tool):
        levels = reached_parameters(tool)
        declared = set(inspect.signature(tool).parameters)
        return invocation._argument_of_its_own(
            [invocation._read_level(level) for level in levels], declared, ()
        )

    read, held = tools("limit, *args")
    assert verdict(read) is None
    with pytest.raises(HiddenWrapper, match="`limit` is taken by a wrapper"):
        reached_parameters(held)
    with pytest.raises(TypeError):
        read("x")
    with pytest.raises(TypeError):
        held("x")
    read, held = tools("limit, *args, **kwargs")
    assert "`limit` is taken by a wrapper" in verdict(read)
    with pytest.raises(HiddenWrapper, match="`limit` is taken by a wrapper"):
        reached_parameters(held)


def test_finding_a_held_wrapper_does_not_read_the_tool_again():
    """To judge a held wrapper the check needs what the tool declares. ``unsupported_signature``
    has read that already and hands it on, so finding a held wrapper costs no second reading
    of the tool: a property of the tool runs as often with a held wrapper as without one.
    ``reached_parameters`` called alone reads it once, and only when a held wrapper is
    found."""

    def search(query: str, limit: int = 5) -> str:
        return query

    read = []

    class Decorator:
        def __init__(self, function, held):
            self._function, self.held = function, held
            self.__annotations__ = dict(function.__annotations__)

        @property
        def __wrapped__(self):
            read.append("__wrapped__")
            return self._function

        def __call__(self, *args, **kwargs):
            return self.held(*args, **kwargs)

    def reads(check, held):
        del read[:]
        check(Decorator(search, held))
        return len(read)

    passing = _wrapper_taking(search, "*args, **kwargs")
    assert unsupported_signature(Decorator(search, passing)) is None
    assert "`force` is a parameter" in unsupported_signature(
        Decorator(search, _wrapper_taking(search, "*args, force=True, **kwargs"))
    )
    assert reads(unsupported_signature, passing) == reads(unsupported_signature, None) > 0
    assert reads(reached_parameters, passing) > reads(reached_parameters, None) > 0


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


def test_an_instance_dictionary_is_found_without_a_lookup_or_a_property():
    """The instance dictionary of an object is reached through the C-level accessor of its
    class, found by listing the class dictionaries: a key of a class is never compared, and
    a ``__dict__`` the class answers itself (a property) is never called. Pinned on the
    helper itself, since which objects it is used on differs by Python."""
    assert invocation._prepare()

    class Key:
        """Hashes like ``"__dict__"`` and compares unequal to it: any lookup of that name
        in a dictionary that holds this key runs ``__eq__``."""

        __hash__ = _touch("key __hash__", hash("__dict__"))
        __eq__ = _touch("key __eq__", False)

    keyed = type("Keyed", (), {Key(): None})()
    keyed.held = 1
    answering = type("Answering", (), {"__dict__": property(_touch("__dict__", {"held": 2}))})()

    class Base:
        pass

    class Derived(Base):
        pass

    derived = Derived()
    derived.held = 3
    del _TOUCHED[:]
    assert invocation._instance_attributes(keyed) == {"held": 1}
    assert invocation._instance_attributes(answering) == {}
    assert invocation._instance_attributes(derived) == {"held": 3}
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

    def built_on_first_use(function):
        state = {}

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if "guarded" not in state:  # it does not exist when the walk is made
                state["guarded"] = _with_timeout(function)
            return state["guarded"](*args, **kwargs)

        return wrapper

    def in_a_namespace(function):
        import types

        held = types.SimpleNamespace(guarded=_with_timeout(function))

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return held.guarded(*args, **kwargs)

        return wrapper

    def in_an_object_that_is_not_callable(function):
        class Settings:
            def __init__(self):
                self.guarded = _with_timeout(function)

        settings = Settings()

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return settings.guarded(*args, **kwargs)

        return wrapper

    def behind_a_weak_reference(function):
        import weakref

        guarded = _with_timeout(function)
        _SLOT.append(guarded)  # kept alive, as the application would keep it
        reference = weakref.ref(guarded)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return reference()(*args, **kwargs)

        return wrapper

    return [
        ("a wrapper built on first use", built_on_first_use),
        ("in a SimpleNamespace", in_a_namespace),
        ("in an object that is not callable", in_an_object_that_is_not_callable),
        ("behind a weakref.ref the wrapper calls", behind_a_weak_reference),
        ("a wrapper of a tool that names nothing as __wrapped__", of_a_tool_that_names_nothing),
        ("a plain inner function", _calling(named_nothing)),
        ("a lambda", _calling(lambda f: lambda *a, timeout=30, **k: f(*a, **k))),
        ("a callable object", _calling(Guard)),
        ("a bound method of an object that holds the function", _calling(lambda f: Guard(f).run)),
        ("a helper that calls the wrapper", _calling(through_a_helper)),
        ("a subclass of a type of the closed set", _calling(lambda f: Partial(_with_timeout(f)))),
        ("in a container", in_a_container),
        ("in a container in a module global", in_a_global),
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
    """A wrapper a C extension builds is outside the closed set and is not read, and
    neither is the one wrapt builds in Python (read in wrapt 1.17.2), which keeps what
    it wraps in slots and answers ``__dict__`` with a property: so the same decorator
    gets the same verdict whichever way the library was built."""
    import importlib.util
    import os
    import subprocess

    if importlib.util.find_spec("wrapt") is None:
        # the project's test environments do not install wrapt, so its own runs skip this;
        # ``test_a_proxy_is_read_where_its_instance_dictionary_shows_what_it_wraps`` runs
        # everywhere
        pytest.skip("wrapt is not installed (never in the project's own test runs)")
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


def _proxies():
    """Three ways to build the same transparent proxy decorator, as (name, the decorator):
    one keeps what it wraps in its instance dictionary; one in a slot named
    ``__wrapped__``, with no instance dictionary at all, as an object a C extension builds
    and as the proxies of wrapt do; and one in a slot of another name, answering for
    ``__wrapped__`` from ``__getattr__`` alone. None is of the closed set."""

    class Kept:
        def __init__(self, wrapped):
            self.__dict__["__wrapped__"] = wrapped

        def __getattr__(self, name):
            return getattr(self.__dict__["__wrapped__"], name)

        def __get__(self, instance, owner=None):
            return type(self)(self.__dict__["__wrapped__"].__get__(instance, owner))

        def __call__(self, *args, **kwargs):
            return self.__dict__["__wrapped__"](*args, **kwargs)

    class Slotted:
        __slots__ = ("__wrapped__",)

        def __init__(self, wrapped):
            self.__wrapped__ = wrapped

        def __getattr__(self, name):
            return getattr(self.__wrapped__, name)

        def __get__(self, instance, owner=None):
            return type(self)(self.__wrapped__.__get__(instance, owner))

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    class Answering:
        __slots__ = ("_wrapped",)

        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            wrapped = object.__getattribute__(self, "_wrapped")
            return wrapped if name == "__wrapped__" else getattr(wrapped, name)

        def __get__(self, instance, owner=None):
            return type(self)(self._wrapped.__get__(instance, owner))

        def __call__(self, *args, **kwargs):
            return self._wrapped(*args, **kwargs)

    return [
        ("kept in the instance dictionary", Kept),
        ("kept in a slot", Slotted),
        ("answered by __getattr__", Answering),
    ]


def _proxy_verdicts(passing):
    """The verdicts of the shapes a proxy decorator ``passing`` appears in: read by the
    walk, held by a wrapper, holding a hidden wrapper, and on a method."""

    def fetch(url: str, timeout: int = 5) -> int:
        return timeout

    class Holder:
        @passing
        def fetch(self, url: str, timeout: int = 5) -> int:
            return timeout

    shapes = [
        passing(fetch),
        _calling(passing)(fetch),
        passing(_calling(passing)(fetch)),
        _calling(lambda function: passing(_with_timeout(function)))(fetch),
        passing(_calling(_with_timeout)(fetch)),
        Holder().fetch,
    ]
    verdicts = []
    for shape in shapes:
        try:
            verdicts.append(_levels(shape))
        except Exception as error:  # noqa: BLE001, the verdict is the type of the refusal
            verdicts.append(type(error).__name__)
    return verdicts


def test_a_proxy_is_read_where_its_instance_dictionary_shows_what_it_wraps():
    """The test above needs wrapt and both of its builds, which the project's own test
    environments do not install. This pins what a held proxy is read by, everywhere. An
    object outside the closed set is read, when held, through its instance dictionary and
    nothing else, so a proxy that keeps what it wraps there is followed, and a hidden
    wrapper held inside it is refused (0.16.0 read neither). One that keeps what it wraps
    where only its own code can reach (a slot, as a C extension's object does) is not
    followed, and a wrapper held inside it is the stated limit. In every other shape the
    two get the same verdict: a hidden wrapper behind a proxy the walk reads is refused
    for both. A proxy that answers for ``__wrapped__`` without holding one is refused
    wherever the walk stands on it (0.16.0 read it), and read as before where it is only
    held."""
    (_, kept), (_, slotted), (_, answering) = _proxies()
    own, called = ["url", "timeout"], ["args", "kwargs"]
    expected = [
        [own, called, own],
        [_FORWARD, own],
        [_FORWARD, called, _FORWARD, own],
        [_FORWARD, own],  # the stated limit: a wrapper held inside a proxy is not read
        "HiddenWrapper",
        [["self"] + own, called, own],
    ]
    assert _proxy_verdicts(slotted) == expected
    refused = ["ProxiedCallable", expected[1], "ProxiedCallable", expected[3]]
    refused += ["HiddenWrapper", "ProxiedCallable"]
    assert _proxy_verdicts(answering) == refused
    expected[3] = "HiddenWrapper"  # read: the instance dictionary names the wrapper
    assert _proxy_verdicts(kept) == expected


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
    # with only the base registered it is refused all the same, where 0.16.0 walked it:
    # an implementation can be registered between the walk and the call. The check runs
    # nothing more of what is held than the walk without it ran
    for func in (
        find,
        Holder().plain,
        Holder.plain,
        _traced(Holder().plain),
        functools.lru_cache(maxsize=None)(Holder().plain),
    ):
        walked, before = _without_the_dispatch_check(func)
        assert walked[-1] == ["self", "key", "deep"]
        del _TOUCHED[:]
        with pytest.raises(invocation.OpenDispatch):
            reached_parameters(func)
        assert _TOUCHED == before[: len(_TOUCHED)], func
    for skip in ((), ("self",)):
        # ``self`` is shown or not by the Python, so the reason is the dispatch or the
        # parameter, and never None
        assert unsupported_signature(Holder().plain, skip) is not None
    assert unsupported_signature(Holder().plain, ("self",)) == invocation._OPEN_DISPATCH


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
    ``__wrapped__`` names, under the key that is, by identity, ``object``. That one is
    refused as open to registration (0.16.0 walked it), every other as dispatched."""
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

    with pytest.raises(invocation.OpenDispatch):
        reached_parameters(dispatcher_with({object: search}))
    assert not issubclass(invocation.OpenDispatch, DispatchedAtCallTime)
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
import enum, functools, importlib.util, inspect, json, math, sys, types, typing, weakref
REAL, REAL_METHOD, REAL_PARTIAL = functools.singledispatch, functools.singledispatchmethod, functools.partial
REAL_MODULE, RAN, EVENTS, ARMED, SAVED = functools, [], [], [True], {}
scenario = sys.argv[1]
def note(what):
    if ARMED[0]:
        RAN.append(what)
if scenario == "wrapped":
    def trace(function):
        @functools.wraps(function)
        def traced(*args, **kwargs):
            note("traced")
            return function(*args, **kwargs)
        return traced
    functools.singledispatch = trace(REAL)
elif scenario == "object":
    class Odd:
        @property
        def __code__(self):
            note("__code__")
            raise RuntimeError("no code here")
        def __getattr__(self, name):
            note(name)
            raise RuntimeError(name)
        def __call__(self, function):
            return REAL(function)
    functools.singledispatch = Odd()
elif scenario == "reimplemented":
    def singledispatch(func):
        note("singledispatch")
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
elif scenario == "total-ordering-missing":
    del functools.total_ordering
elif scenario == "method":
    class Recording(REAL_METHOD):
        def __getattribute__(self, name):
            note(name)
            return super().__getattribute__(name)
        def __get__(self, obj, cls=None):
            note("__get__")
            raise KeyboardInterrupt("no")
    functools.singledispatchmethod = Recording
elif scenario == "method-subclass":
    class Inheriting(REAL_METHOD):
        def __getattribute__(self, name):
            note(name)
            return super().__getattribute__(name)
    functools.singledispatchmethod = Inheriting
elif scenario == "method-get-patched":
    real_get = REAL_METHOD.__get__
    def patched_get(self, obj, cls=None):
        note("patched __get__")
        return real_get(self, obj, cls)
    REAL_METHOD.__get__ = patched_get
elif scenario == "method-init-patched":
    real_init = REAL_METHOD.__init__
    def patched_init(self, func):
        note("patched __init__")
        real_init(self, func)
    REAL_METHOD.__init__ = patched_init
elif scenario == "handed-class-replaced" and hasattr(functools, "_singledispatchmethod_get"):
    class Handed(functools._singledispatchmethod_get):
        def __init__(self, *args, **kwargs):
            note("handed __init__")
            super().__init__(*args, **kwargs)
    functools._singledispatchmethod_get = Handed
elif scenario in ("handed-init-patched", "handed-call-patched") and hasattr(functools, "_singledispatchmethod_get"):
    name = "__init__" if scenario == "handed-init-patched" else "__call__"
    real_handed = getattr(functools._singledispatchmethod_get, name)
    def patched_handed(self, *args, **kwargs):
        note("patched " + name)
        return real_handed(self, *args, **kwargs)
    setattr(functools._singledispatchmethod_get, name, patched_handed)
elif scenario == "partial-lookalike":
    class Slotted:
        __slots__ = ("func", "args", "keywords")
        def __init__(self, func, *args, **keywords):
            self.func, self.args, self.keywords = func, args, keywords
        def __call__(self, *args, **keywords):
            return self.func(*self.args, *args, **dict(self.keywords, **keywords))
        def __getattribute__(self, name):
            note("lookalike." + name)
            return object.__getattribute__(self, name)
    functools.partial = Slotted
elif scenario == "cache-lookalike":
    class Cached:
        def __init__(self, function):
            self.__wrapped__ = function
        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)
    functools.lru_cache = lambda maxsize=128, typed=False: Cached
    functools._lru_cache_wrapper = Cached
elif scenario == "cache-is-partial":
    functools.lru_cache = lambda maxsize=128, typed=False: REAL_PARTIAL
elif scenario == "module-getattr":
    for name in ("total_ordering", "singledispatch", "singledispatchmethod", "_singledispatchmethod_get"):
        if name in vars(functools):
            SAVED[name] = vars(functools).pop(name)
    def answering(name):
        note("module __getattr__ " + name)
        raise AttributeError(name)
    functools.__getattr__ = answering
elif scenario == "module-subclass":
    class Module(types.ModuleType):
        pass
    replaced = Module("functools")
    replaced.__dict__.update(vars(functools))
    sys.modules["functools"] = replaced
elif scenario == "partial-subclass":
    class TracedPartial(REAL_PARTIAL):
        pass
    functools.partial = TracedPartial
elif scenario == "partial-function":
    def partial(func, *args, **kwargs):
        note("partial factory")
        return REAL_PARTIAL(func, *args, **kwargs)
    functools.partial = partial
elif scenario == "module-namespace":
    sys.modules["functools"] = types.SimpleNamespace(**vars(functools))
elif scenario == "module-proxy":
    class Proxy:
        def __getattr__(self, name):
            note("proxy." + name)
            return getattr(REAL_MODULE, name)
    sys.modules["functools"] = Proxy()
elif scenario == "weak-dictionary-replaced":
    class Weak(weakref.WeakKeyDictionary):
        def __init__(self, *args, **kwargs):
            note("WeakKeyDictionary()")
            super().__init__(*args, **kwargs)
    weakref.WeakKeyDictionary = Weak
elif scenario in ("audit-raises", "audit-observes"):
    def audit(event, args):
        if ARMED[0] and event.startswith("object.") and len(args) > 1:
            EVENTS.append(event + ":" + str(args[1]))
            if scenario == "audit-raises" and args[1] == "__code__":
                raise RuntimeError("__code__ is not read here")
    sys.addaudithook(audit)
try:
    spec = importlib.util.spec_from_file_location("invocation_alone", sys.argv[2])
    invocation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(invocation)  # this module alone: what its own import does
    imported = "ok"
except BaseException as error:
    imported = type(error).__name__
ran_at_import, events_at_import = list(RAN), list(EVENTS)
ready = invocation._prepare()
ran_preparing = RAN[len(ran_at_import):]
ARMED[0] = False
vars(REAL_MODULE).update(SAVED)  # what a scenario took out of the module is put back
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
def resilient(function):
    @functools.wraps(function)
    def guarded(*args, deep=True, **kwargs):
        return function(*args, **kwargs)
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return guarded(*args, **kwargs)
    return wrapper
def verdicts():
    holder = Holder()
    return {
        "registered": verdict(registered),
        "base_only": verdict(REAL(base)),
        "method_registered": verdict(holder.lookup),
        "method_base_only": verdict(holder.plain),
        "hidden": verdict(resilient(base)),
        "partial": verdict(REAL_PARTIAL(resilient(base))),
    }
found, method, real = invocation._WRAPPER_CODE is not None, invocation._METHOD_ACCESS is not None, verdicts()
invocation._READY = False  # the walk as it was before the two checks
invocation._WRAPPER_CODE = invocation._DISPATCH_CODE = invocation._METHOD_ACCESS = None
print(json.dumps({
    "imported": imported,
    "ran": ran_at_import,
    "events": events_at_import,
    "ran_preparing": ran_preparing,
    "ready": ready,
    "found": found,
    "method": method,
    "real": real,
    "before": verdicts(),
}))
"""

_IMPORT_SCENARIOS = {
    # scenario: (checks on, dispatcher found, single-dispatch method found)
    "none": (True, True, True),
    "wrapped": (True, False, False),
    "object": (True, False, False),
    "reimplemented": (True, False, False),
    "total-ordering-missing": (True, False, False),
    "method": (True, True, False),
    "method-subclass": (True, True, False),
    "method-get-patched": (True, True, False),
    "method-init-patched": (True, True, True),
    # the class a single-dispatch method hands out, which exists from Python 3.14
    "handed-class-replaced": (True, True, sys.version_info < (3, 14)),
    "handed-init-patched": (True, True, sys.version_info < (3, 14)),
    "handed-call-patched": (True, True, sys.version_info < (3, 14)),
    "partial-lookalike": (False, False, False),
    "cache-lookalike": (False, False, False),
    "cache-is-partial": (False, False, False),
    "module-getattr": (True, False, False),
    "module-subclass": (False, False, False),
    "partial-subclass": (False, False, False),
    "partial-function": (False, False, False),
    "module-namespace": (False, False, False),
    "module-proxy": (False, False, False),
    "weak-dictionary-replaced": (True, True, True),
    "audit-raises": (False, False, False),
    "audit-observes": (True, True, True),
}


@pytest.mark.parametrize("scenario", sorted(_IMPORT_SCENARIOS))
def test_importing_the_module_runs_nothing_whatever_stands_in_functools(scenario):
    """A process can wrap or replace what stands in ``functools`` or in ``sys.modules``, and
    can install audit hooks, before this module is imported. Importing it then does what it
    did before the two checks existed and nothing more: it does not raise, runs none of
    that code, and reads no code object (an audit hook sees none read). What the checks
    read with is found at the first walk, running none of that code either. Where a part is
    not the standard library's own, the checks that need it are off and the walk is what it
    was before them: nothing is refused because of them, so a base-only single-dispatch
    function or method always gets the verdict it had."""
    import json
    import os
    import subprocess

    on, found, method = _IMPORT_SCENARIOS[scenario]
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    alone = os.path.join(root, "recusal", "invocation.py")
    done = subprocess.run(
        [sys.executable, "-c", _IMPORTED_AFTER, scenario, alone],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    assert seen["imported"] == "ok"
    # the import machinery reads dunder names of what stands in ``sys.modules``, and the
    # module always looked ``lru_cache`` up: nothing else of that object is touched
    assert [name for name in seen["ran"] if not name.startswith("proxy.__")] == (
        ["proxy.lru_cache"] if scenario == "module-proxy" else []
    )
    assert seen["ran_preparing"] == []
    assert not [event for event in seen["events"] if event.endswith(":__code__")]
    assert (seen["ready"], seen["found"], seen["method"]) == (on, found, method)
    real, before = seen["real"], seen["before"]
    expected = dict(before)
    if on:
        for case, refusal, applies in (
            ("registered", "DispatchedAtCallTime", found),
            ("base_only", "OpenDispatch", found),
            ("method_registered", "DispatchedAtCallTime", method),
            ("method_base_only", "OpenDispatch", method),
            ("hidden", "HiddenWrapper", True),
            ("partial", "HiddenWrapper", True),
        ):
            if applies and before[case] == "walked":
                expected[case] = refusal
    assert real == expected
    # with the checks off the walk is what it was before them: everything here is walked,
    # but for what that walk itself cannot read in the scenario
    unread = set()
    if scenario == "partial-function":  # ``isinstance`` with a function raises for every tool
        unread = set(before)
    elif scenario in ("partial-lookalike", "partial-subclass", "cache-is-partial"):
        unread = {"partial"}  # a real partial is then read as a callable object, or a cache
    elif scenario == "handed-class-replaced" and not method:  # the replaced class is read
        unread = {"method_registered", "method_base_only"}
    assert {case: verdict for case, verdict in before.items() if case not in unread} == {
        case: "walked" for case in before if case not in unread
    }
    if scenario == "none":
        assert real["registered"] == real["method_registered"] == "DispatchedAtCallTime"
        assert real["hidden"] == real["partial"] == "HiddenWrapper"


def test_the_single_dispatch_this_python_builds_is_recognised():
    """The dispatcher is known by the code object of the ``wrapper`` that
    ``functools.singledispatch`` defines, and its registry by the closure of its
    ``dispatch``. If a Python builds it another way the check finds nothing there: this
    test fails first, loudly, so that is a decision and not a silence."""
    import functools

    assert invocation._prepare()

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

    # a function is a ``dispatch`` by its code object, not by a free variable called registry
    def lookalike(base):
        registry = {object: base}
        return lambda cls: registry[object]

    assert not invocation._only_runs(lookalike(search), search)
    with pytest.raises(invocation.OpenDispatch):  # with nothing registered beside its base
        reached_parameters(built)


def test_where_a_single_dispatch_method_keeps_its_registry_is_found():
    """Read from the standard library's own code at the first walk, without building one,
    and checked here against one really built. It differs by Python: a closure cell
    holding the descriptor (3.9 to 3.12) or the ``dispatch`` function (3.13), an instance
    attribute holding the descriptor (3.14, 3.15). Not finding it is a failure on every
    Python, never a skip: recognising a single-dispatch method would be lost silently."""
    import functools

    assert invocation._prepare()

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
    # with only its base, each flavour is refused as open to registration, or before that
    # by what the walk cannot read of it on this Python: never as dispatched (on Python
    # 3.9 the function handed out for a classmethod names the function inside it as
    # ``__wrapped__``, not the classmethod the registry has), and never walked
    for func in (Holder().lookup, Holder.make, Holder().make, Holder.plain, Holder().plain):
        try:
            reached_parameters(func)
        except DispatchedAtCallTime:
            raise
        except (TypeError, ValueError):
            continue
        raise AssertionError("a single-dispatch method with only its base was walked")
    with pytest.raises(invocation.OpenDispatch):
        reached_parameters(Holder().lookup)


def test_what_only_looks_like_a_dispatcher_is_walked():
    """Attributes do not make a dispatcher: ``functools.update_wrapper`` copies
    ``registry``, ``dispatch`` and ``register`` onto every wrapper of one. The check is made
    on the function ``functools.singledispatch`` itself returned, and on nothing held. A
    wrapper of a dispatcher is refused when the walk reaches the dispatcher, whatever is
    registered on it, and not for the attributes it carries."""
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

    # every wrapper of a dispatcher with only its base carries the copy. 0.16.0 walked
    # each of them; each is refused, by the dispatcher the walk reaches behind it
    base_only = functools.singledispatch(search)

    class Decorator:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    class Service:
        find = base_only

    bound = functools.update_wrapper(functools.partial(base_only), base_only)
    for wrapper in (
        _traced(base_only),
        functools.lru_cache(maxsize=None)(base_only),
        _traced(functools.lru_cache(maxsize=None)(base_only)),
        _traced(Decorator(base_only)),
        _traced(bound),
        _traced(Service().find),
    ):
        with pytest.raises(invocation.OpenDispatch):
            reached_parameters(wrapper)
    # and so are the same wrappers of one with an implementation registered
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

    # nor can one whose closure was emptied: it is refused as dispatch, not raised out of
    emptied = functools.singledispatch(search)
    cells = dict(zip(emptied.__code__.co_freevars, emptied.__closure__))
    del cells["dispatch"].cell_contents
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(emptied)


_RELOADED = """
import functools, importlib, json
from recusal import invocation
def verdict(build):
    def base(key: str, deep: bool = False) -> str:
        return "base"
    built = build(base)
    built.register(int, lambda key, deep=False, skip_review=True: "int")
    try:
        invocation.reached_parameters(built)
        return "walked"
    except Exception as error:
        return type(error).__name__
before = functools.singledispatch
first = verdict(before)
importlib.reload(functools)
print(json.dumps([first, verdict(before), verdict(functools.singledispatch)]))
"""


def test_a_dispatcher_a_reloaded_functools_builds_is_a_stated_limit():
    """The dispatcher is known by the code object of the standard library's function as it
    stood at the first walk. ``importlib.reload(functools)`` compiles new code objects: a
    dispatcher built by the reloaded module is not recognised and is walked as before the
    check, registered implementations and all. One built by the function held from before
    the reload still is."""
    import json
    import os
    import subprocess

    root = str(pathlib.Path(__file__).resolve().parent.parent)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run(
        [sys.executable, "-c", _RELOADED], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["DispatchedAtCallTime", "DispatchedAtCallTime", "walked"]


_FIRST_WALK = """
import functools, json, sys, threading
from recusal import invocation
REAL = functools.singledispatch
def base(key: str, deep: bool = False) -> str:
    return "base"
def resilient(function):
    @functools.wraps(function)
    def guarded(*args, deep=True, **kwargs):
        return function(*args, **kwargs)
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return guarded(*args, **kwargs)
    return wrapper
class Holder:
    @functools.singledispatchmethod
    def plain(self, key: str, deep: bool = False) -> str:
        return "base"
    @functools.singledispatchmethod
    def lookup(self, key: str, deep: bool = False) -> str:
        return "base"
    @lookup.register(int)
    def lookup_int(self, key: int, deep: bool = False, skip_review: bool = True) -> str:
        return "int"
registered = REAL(base)
registered.register(int, lambda key, deep=False, skip_review=True: "int")
holder = Holder()
tools = [registered, REAL(base), resilient(base), functools.partial(resilient(base)), holder.lookup, holder.plain]
def verdicts():
    out = []
    for tool in tools:
        try:
            invocation.reached_parameters(tool)
            out.append("walked")
        except Exception as error:
            out.append(type(error).__name__)
    return out
assert invocation._READY is None and not invocation._DECIDED  # nothing was found at import
if sys.argv[1] == "threads":
    sys.setswitchinterval(1e-6)
    count = 32
    barrier, seen = threading.Barrier(count), [None] * count
    def first_walk(index):
        barrier.wait()
        seen[index] = verdicts()
    threads = [threading.Thread(target=first_walk, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(json.dumps({"seen": sorted(set(map(tuple, seen))), "ready": invocation._READY, "after": verdicts()}))
elif sys.argv[1] == "decided":
    invocation._DECIDED["found"] = (None, None, None)  # as if another thread arrived first and found nothing
    print(json.dumps({"seen": [verdicts()], "ready": invocation._READY, "after": verdicts()}))
else:
    functools.singledispatch = lambda function: REAL(function)  # patched for the first walk
    first = verdicts()
    functools.singledispatch = REAL
    print(json.dumps({"seen": [first], "ready": invocation._READY, "after": verdicts()}))
"""


@pytest.mark.parametrize("run", range(4))
def test_threads_that_make_their_first_walk_at_once_get_the_same_verdicts(run):
    """What the checks read with is found at the first walk. When many threads make their
    first walk at the same moment, each finds it for itself and the first to arrive decides
    for the process in one step: every thread gets the right verdict on its very first
    walk, and so does every walk after. A fresh interpreter each run, since a first walk
    happens once."""
    import json
    import os
    import subprocess

    root = str(pathlib.Path(__file__).resolve().parent.parent)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run(
        [sys.executable, "-c", _FIRST_WALK, "threads"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    right = ["DispatchedAtCallTime", "OpenDispatch", "HiddenWrapper", "HiddenWrapper"]
    right += ["DispatchedAtCallTime", "OpenDispatch"]
    assert json.loads(done.stdout) == {"seen": [right], "ready": True, "after": right}


def test_what_the_first_walk_finds_stands_for_the_process():
    """A ``functools.singledispatch`` that is patched when the first walk is made is not
    the standard library's function, so the dispatch check is off; it stays off when the
    function is put back, because what the first walk finds is not looked for again. The
    hidden-wrapper check, which does not depend on it, is on throughout. Stated as it is."""
    import json
    import os
    import subprocess

    root = str(pathlib.Path(__file__).resolve().parent.parent)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run(
        [sys.executable, "-c", _FIRST_WALK, "patched"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    off = ["walked", "walked", "HiddenWrapper", "HiddenWrapper", "walked", "walked"]
    assert json.loads(done.stdout) == {"seen": [off], "ready": True, "after": off}
    # and a thread that arrives after another has decided takes that decision, whatever it
    # would have found itself: no walk sees a part of one finding beside a part of another
    done = subprocess.run(
        [sys.executable, "-c", _FIRST_WALK, "decided"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {
        "seen": [["walked"] * 6],
        "ready": False,
        "after": ["walked"] * 6,
    }


def test_where_a_single_dispatch_method_keeps_its_registry_is_read_from_its_code():
    """Where the callable a single-dispatch method hands out keeps the way to its registry
    is read from the code of ``singledispatchmethod.__get__``, without building or calling
    anything. Every way a Python from 3.9 to 3.15 does it is pinned here on a stand-in for
    the ``functools`` namespace, whichever Python runs this: a ``_method`` closing over the
    descriptor (``self``) or over ``dispatch``, or an object of the class ``functools``
    names, which keeps the descriptor as ``_unbound``. Anything else, and code that is not
    from the file the namespace's own functions are from, gives None: the method half of
    the check is then off, never guessed."""
    import functools
    import types

    assert invocation._prepare()

    def total_ordering(cls):  # the function whose file says what is the namespace's own
        return cls

    def closes_self(self, obj, cls=None):
        def _method(*args, **kwargs):
            return self

        return _method

    def closes_dispatch(self, obj, cls=None):
        dispatch = None

        def _method(*args, **kwargs):
            return dispatch

        return _method

    def closes_another(self, obj, cls=None):
        another = None

        def _method(*args, **kwargs):
            return another

        return _method

    def hands_an_object(self, obj, cls=None):
        return None

    class Handed:
        def __init__(self, unbound):
            self._unbound = unbound

        def __call__(self, *args, **kwargs):
            return None

    class KeptElsewhere:
        def __init__(self, unbound):
            self._kept = unbound

        def __call__(self, *args, **kwargs):
            return None

    class NeverCalled:
        def __init__(self, unbound):
            self._unbound = unbound

    class BuiltElsewhere:
        __init__ = functools.partialmethod.__init__  # code from another file

        def __call__(self, *args, **kwargs):
            return self._unbound

    class CalledElsewhere:
        def __init__(self, unbound):
            self._unbound = unbound

        __call__ = functools.partialmethod.__get__  # code from another file

    def access(get, handed=None, anchor=total_ordering):
        names = {
            "total_ordering": anchor,
            "singledispatchmethod": type("singledispatchmethod", (), {"__get__": get}),
        }
        if handed is not None:
            names["_singledispatchmethod_get"] = handed
        return invocation._method_access(names)

    inner = closes_self(None, None).__code__
    assert access(closes_self) == (types.FunctionType, inner, "self", True)
    inner = closes_dispatch(None, None).__code__
    assert access(closes_dispatch) == (types.FunctionType, inner, "dispatch", False)
    assert access(closes_another) is None
    assert access(closes_another, Handed) is None
    assert access(hands_an_object, Handed) == (Handed, None, "_unbound", True)
    assert access(hands_an_object) is None
    assert access(hands_an_object, KeptElsewhere) is None
    assert access(hands_an_object, NeverCalled) is None
    assert access(hands_an_object, BuiltElsewhere) is None
    assert access(hands_an_object, CalledElsewhere) is None
    # a ``__get__`` that is not the namespace's own is not read, whatever it defines
    assert access(functools.partialmethod.__get__, Handed) is None
    assert access(closes_self, anchor=functools.total_ordering) is None
    assert access(closes_self, anchor=None) is None
    # and neither is anything but a class under the names read
    names = {"total_ordering": total_ordering, "singledispatchmethod": closes_self}
    assert invocation._method_access(names) is None
    names = {
        "total_ordering": total_ordering,
        "singledispatchmethod": type("singledispatchmethod", (), {"__get__": hands_an_object}),
        "_singledispatchmethod_get": Handed(None),
    }
    assert invocation._method_access(names) is None


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
    # the same wrapper around a dispatcher with only its base: 0.16.0 read it as hiding
    # nothing, and an implementation registered after that walk ran unjudged
    base_only = functools.singledispatch(lookup.__wrapped__)

    @functools.wraps(lookup.__wrapped__)
    def plain(*args, **kwargs):
        return base_only(*args, **kwargs)

    with pytest.raises(HiddenWrapper):
        reached_parameters(plain)

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
    if invocation._METHOD_ACCESS[1] is not None:
        for held in (registered, only_base):  # 0.16.0 read the one with only its base
            with pytest.raises(HiddenWrapper):
                reached_parameters(held)
    else:
        reached_parameters(registered)  # a stated limit on this Python
        reached_parameters(only_base)


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
    for func in (by_hand, through_a_lambda):
        reached_parameters(func)
    # an implementation registered after the walk is no longer one of these: a
    # dispatcher is refused before anything is registered on it
    later = functools.singledispatch(lookup.__wrapped__)
    with pytest.raises(invocation.OpenDispatch):
        reached_parameters(later)
    later.register(int, lambda key, deep=False, skip_review=True: "int")
    assert later(1) == "int"


def inspect_signature(func):
    import inspect

    return inspect.signature(func).parameters


# --- functools.partialmethod over a callable that does not bind as a method ---------------
#
# ``functools.partialmethod(f, ...)`` over a plain function hands out a ``functools.partial``
# of the bound function, which the walk follows. Over a callable object it hands out a
# method of a function ``functools`` builds, which calls ``f`` from its closure and names
# nothing as ``__wrapped__``; ``inspect.signature`` reads a mark on it and shows the
# parameters of ``f`` less what is bound. The walk read that one level and nothing under
# the object.

_RAN: List[Any] = []


def _shows(func) -> List[str]:
    import inspect

    return [str(inspect.Signature(list(level.values()))) for level in reached_parameters(func)]


def _refused_as_a_partialmethod(func) -> None:
    with pytest.raises(TypeError, match="functools.partialmethod") as refused:
        reached_parameters(func)
    # a ``TypeError``, which every gate already refuses on, with a name of its own
    assert type(refused.value) is PartialMethodSignature
    assert issubclass(PartialMethodSignature, TypeError)
    assert not issubclass(PartialMethodSignature, ValueError)
    assert str(refused.value) == invocation._PARTIALMETHOD
    assert "plain function that calls it" not in str(refused.value)  # 0.17.0's advice
    assert "put the partialmethod over a plain function" in str(refused.value)
    # the kernel's own check gives that reason, before any other about the signature shown
    assert unsupported_signature(func) == invocation._PARTIALMETHOD


class _Fetcher:
    """Q1: the tool's code is a callable object; ``table`` is positional-only beside
    ``**extra``."""

    def __call__(self, repo, table, /, limit=10, **extra):
        _RAN.append({"table": table, "limit": limit, **{"extra." + k: v for k, v in extra.items()}})
        return "ok"


def _fetch_rows(repo, table, /, limit=10, **extra):
    _RAN.append({"table": table, "limit": limit, **{"extra." + k: v for k, v in extra.items()}})
    return "ok"


class _Fetcher2:
    """Q2: ``limit`` is positional-only, so a keyword of that name lands in ``**extra``."""

    def __call__(self, repo, table, limit=10, /, **extra):
        _RAN.append({"table": table, "limit": limit, **{"extra." + k: v for k, v in extra.items()}})
        return "ok"


def _fetch_rows_capped(repo, table, limit=10, /, **extra):
    _RAN.append({"table": table, "limit": limit, **{"extra." + k: v for k, v in extra.items()}})
    return "ok"


def _scoped(fn):
    """D1: the tenant is the decorator's parameter, not the function's."""
    import functools

    @functools.wraps(fn)
    def wrapper(repo, tenant, *args, **kwargs):
        return fn(repo, *args, **kwargs)

    return wrapper


def _default_table(fn):
    """D1: the table defaults to the archive."""
    import functools

    @functools.wraps(fn)
    def wrapper(repo, table="archive", *, limit=100):
        return fn(repo, table, limit=limit)

    return wrapper


class _Audited:
    """D1: a class-based decorator that states what it takes and passes all of it on."""

    def __init__(self, fn):
        import functools

        functools.update_wrapper(self, fn)
        self.fn = fn

    def __call__(self, repo, tenant, *, limit=100):
        return self.fn(repo, tenant, limit=limit)


def _purge(repo, table, *, limit=100):
    _RAN.append({"table": table, "limit": limit})
    return "ok"


def _repo():
    import functools

    class Repo:
        q1 = functools.partialmethod(_Fetcher(), "customers")
        q1_function = functools.partialmethod(_fetch_rows, "customers")
        q2 = functools.partialmethod(_Fetcher2(), limit=77)
        q2_function = functools.partialmethod(_fetch_rows_capped, limit=77)
        d1 = functools.partialmethod(_Audited(_scoped(_default_table(_purge))), "acme")
        d1_function = functools.partialmethod(_scoped(_default_table(_purge)), "acme")
        nothing_bound = functools.partialmethod(_Fetcher())
        twice = functools.partialmethod(functools.partialmethod(_Fetcher(), "customers"), 3)
        as_static = functools.partialmethod(staticmethod(_fetch_rows), None, "customers")
        as_class = functools.partialmethod(classmethod(_fetch_rows), "customers")

    return Repo


def test_q1_a_partialmethod_over_a_callable_object_binding_by_position_is_refused():
    """The model sends ``table='orders'`` and a policy that allows only that table passes
    it; the object runs with the bound ``'customers'`` and the model's value lands in
    ``**extra``. The walk showed one level, ``(limit=10, **extra)``, and read nothing of
    the object."""
    tool = _repo()().q1
    _RAN.clear()
    tool(table="orders", limit=5)
    assert _RAN == [{"table": "customers", "limit": 5, "extra.table": "orders"}]
    assert inspect_signature(tool).keys() == {"limit", "extra"}
    _refused_as_a_partialmethod(tool)
    assert unsupported_signature(tool) is not None


def test_q2_a_partialmethod_over_a_callable_object_binding_a_keyword_is_refused():
    """``limit=77`` is bound by keyword to a positional-only parameter: it never reaches
    it, and the object runs with its own default, which nothing read."""
    tool = _repo()().q2
    _RAN.clear()
    tool("orders")
    assert _RAN == [{"table": "orders", "limit": 10, "extra.limit": 77}]
    _refused_as_a_partialmethod(tool)
    assert unsupported_signature(tool) is not None


def test_d1_a_partialmethod_over_a_class_based_decorator_is_refused():
    """A value bound by position hides the leading parameter, and its default, of every
    wrapper under the object: judged ``{'limit': 5}``, ran ``table='archive'``."""
    tool = _repo()().d1
    _RAN.clear()
    tool(limit=5)
    assert _RAN == [{"table": "archive", "limit": 5}]
    assert list(inspect_signature(tool)) == ["limit"]
    _refused_as_a_partialmethod(tool)
    assert unsupported_signature(tool) is not None


def test_a_partialmethod_over_a_plain_function_is_still_read_in_full():
    """Over a function the same ``partialmethod`` is a ``functools.partial`` of the bound
    function: it carries no mark, and every callable under it is read as before."""
    import functools

    repo = _repo()()
    for name in ("q1_function", "q2_function", "d1_function", "as_static", "as_class"):
        tool = getattr(repo, name)
        assert type(tool) is functools.partial, name
        assert not any(hasattr(tool, mark) for mark in invocation._PARTIALMETHOD_MARKS)
    assert _shows(repo.q1_function) == ["(limit=10, **extra)"]
    # D1 over a plain function is a partial the walk reads, and it is the same defect: the
    # bound "acme" is the wrapper's ``tenant``, and ``table`` runs at its default. Refused
    # by the rule for a bound argument a wrapper takes under another name.
    with pytest.raises(invocation.BoundArgumentKept, match="`tenant`.*`table`"):
        reached_parameters(repo.d1_function)
    assert _shows(repo.as_static) == ["(limit=10, **extra)"]
    assert _shows(repo.as_class) == ["(limit=10, **extra)"]
    _RAN.clear()
    assert repo.q1_function(limit=5) == "ok" and _RAN == [{"table": "customers", "limit": 5}]


def test_a_partialmethod_is_refused_whatever_it_binds_and_wherever_it_stands():
    """With nothing bound, bound twice, taken from the class (where a plain function gets
    the same built method), and behind a ``functools.partial``: the level the walk would
    read is in every case the one ``inspect.signature`` makes up from the mark."""
    import functools

    Repo = _repo()
    repo = Repo()
    for tool in (
        repo.nothing_bound,
        repo.twice,
        Repo.q1,
        Repo.q1_function,
        functools.partial(repo.q1, limit=2),
        functools.partial(functools.partial(repo.d1), limit=2),
    ):
        _refused_as_a_partialmethod(tool)

    class Calls:
        __call__ = functools.partialmethod(_Fetcher(), "customers")

    _refused_as_a_partialmethod(Calls())
    # everything bound: no parameter is left to judge, and the object under it is still
    # not read. ``unsupported_signature`` accepted this one.
    assert list(inspect_signature(repo.twice)) == ["extra"]

    class Quiet:
        def __call__(self, repo, table, limit=10):
            return "ok"

    class Bound:
        all_bound = functools.partialmethod(Quiet(), "customers", 3)

    assert list(inspect_signature(Bound().all_bound)) == []
    _refused_as_a_partialmethod(Bound().all_bound)
    assert unsupported_signature(Bound().all_bound) == invocation._PARTIALMETHOD


def test_a_partialmethod_is_the_reason_given_before_any_other():
    """``unsupported_signature`` read the signature a partialmethod shows first, and the
    function ``functools`` builds carries no annotations: a tool whose parameters are all
    annotated was refused as "`limit` is not annotated with a supported data type". The
    partialmethod is asked for first, on the callable and further in."""
    import functools

    class Typed:
        def __call__(self, repo, table: str, limit: int = 10) -> str:
            return table

    class Repo:
        fetch = functools.partialmethod(Typed(), "customers")
        variadic = functools.partialmethod(_Fetcher(), "customers")

    for tool in (Repo().fetch, Repo().variadic, functools.partial(Repo().fetch, limit=2)):
        assert unsupported_signature(tool) == invocation._PARTIALMETHOD
        assert unsupported_signature(tool, ("limit", "extra")) == invocation._PARTIALMETHOD

    @functools.wraps(Repo().fetch)
    def forwards(*args, **kwargs):
        return Repo().fetch(*args, **kwargs)

    del forwards.__dict__[next(m for m in invocation._PARTIALMETHOD_MARKS if m in vars(forwards))]
    # the wrapper no longer carries the mark; what it names as ``__wrapped__`` does
    assert unsupported_signature(forwards) == invocation._PARTIALMETHOD

    # what is not a partialmethod is judged as before, in the order it was
    def plain(table, limit: int = 10) -> str:
        return table

    assert unsupported_signature(plain) == "`table` is not annotated with a supported data type"

    def declared(table, limit: int = 10) -> str:
        return table

    declared.__signature__ = __import__("inspect").signature(plain)
    with pytest.raises(DeclaredSignature):
        reached_parameters(declared)
    assert unsupported_signature(declared) == (
        "`table` is not annotated with a supported data type"
    )


def test_a_wrapper_that_carries_a_partialmethods_mark_is_refused():
    """``functools.update_wrapper`` copies the mark onto a wrapper of such a method, and
    ``inspect.signature`` then shows the partialmethod's parameters in place of the
    wrapper's own: a parameter the wrapper adds would not be read."""
    import functools

    method = _repo()().q1

    def forcing(fn):
        @functools.wraps(fn)
        def wrapper(*args, force=True, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    wrapper = forcing(method)
    assert any(hasattr(wrapper, mark) for mark in invocation._PARTIALMETHOD_MARKS)
    assert "force" not in inspect_signature(wrapper)
    import inspect

    assert "force" not in inspect.signature(wrapper, follow_wrapped=False).parameters
    _refused_as_a_partialmethod(wrapper)
    _refused_as_a_partialmethod(functools.lru_cache(maxsize=None)(method))

    class Traced:
        def __init__(self, fn):
            functools.update_wrapper(self, fn)

        def __call__(self, *args, force=True, **kwargs):
            return "ok"

    _refused_as_a_partialmethod(Traced(method))


@pytest.mark.parametrize("mark", ["_partialmethod", "__partialmethod__"])
def test_the_mark_is_refused_under_either_name_and_only_for_a_partialmethod(mark):
    """Both names are looked for on every Python (``functools`` writes one of them, by
    version). The test is the one ``inspect.signature`` makes: an attribute that is not a
    ``functools.partialmethod`` is not a mark, and the callable is read as before."""
    import functools

    def plain(table: str, limit: int = 10) -> str:
        return table

    setattr(plain, mark, "a setting of the application's")
    assert _shows(plain) == ["(table: str, limit: int = 10)"]
    assert unsupported_signature(plain) is None

    def marked(table: str, limit: int = 10) -> str:
        return table

    setattr(marked, mark, functools.partialmethod(_Fetcher(), "customers"))
    _refused_as_a_partialmethod(marked)

    class Sub(functools.partialmethod):
        pass

    def by_subclass(table: str, limit: int = 10) -> str:
        return table

    setattr(by_subclass, mark, Sub(_Fetcher(), "customers"))
    _refused_as_a_partialmethod(by_subclass)

    # a ``functools.partialmethod`` set on a class under one of the names is a method of
    # that class, not a mark: read from an object it is what the descriptor hands out,
    # which ``inspect.signature`` does not take for one either
    class Carries:
        def __call__(self, table: str, limit: int = 10) -> str:
            return table

    setattr(Carries, mark, functools.partialmethod(_Fetcher(), "customers"))
    assert _shows(Carries()) == ["(table: str, limit: int = 10)"] * 2
    assert list(inspect_signature(Carries())) == ["table", "limit"]


def test_the_partialmethod_check_prepares_nothing_and_runs_before_any_walk_state():
    """The refusal needs none of what the first walk prepares: it holds where both of the
    other checks are off."""
    tool = _repo()().d1
    ready = invocation._READY
    try:
        invocation._READY = False
        _refused_as_a_partialmethod(tool)
    finally:
        invocation._READY = ready


# --- a second wrapper held in a module global the wrapper names ---------------------------
#
# A wrapper written beside the function it wraps can call a second wrapper of that function
# by its global name. It is in no closure cell and is held all the same: the module globals
# the code of a wrapper names are read like its closure cells, and judged by the same rule,
# so only a wrapper or a dispatcher OF A CALLABLE THE WALK READ is refused.


def _global_fetch(url: str, timeout: int = 999999) -> int:
    return timeout


def _global_other(url: str, timeout: int = 999999) -> int:
    return timeout


_GLOBAL_GUARDED = _with_timeout(_global_fetch)
_GLOBAL_GUARDED_OTHER = _with_timeout(_global_other)
_GLOBAL_HELPER = lambda url: url  # noqa: E731
_GLOBAL_LOG = __import__("logging").getLogger("recusal.tests.invocation")
_GLOBAL_PARTIAL_OF_OTHER = __import__("functools").partial(_global_other, timeout=30)
_GLOBAL_FORWARDS = __import__("functools").wraps(_global_fetch)(
    lambda *args, **kwargs: _global_fetch(*args, **kwargs)
)


def _global_lookup(key: str, deep: bool = False) -> str:
    return "base"


def _global_lookup_other(key: str, deep: bool = False) -> str:
    return "other"


_GLOBAL_DISPATCHER = __import__("functools").singledispatch(_global_lookup)
_GLOBAL_DISPATCHER.register(int, lambda key, deep=False, skip_review=True: f"int, {skip_review}")
_GLOBAL_DISPATCHER_OTHER = __import__("functools").singledispatch(_global_lookup_other)
_GLOBAL_DISPATCHER_OTHER.register(int, lambda key, deep=False, skip_review=True: "other int")
_GLOBAL_DISPATCHER_BASE_ONLY = __import__("functools").singledispatch(_global_lookup)


def _wraps(function):
    import functools

    return functools.wraps(function)


@_wraps(_global_fetch)
def _calls_a_global_wrapper(*args, **kwargs):
    return _GLOBAL_GUARDED(*args, **kwargs)


@_wraps(_global_fetch)
def _calls_a_global_wrapper_in_a_lambda(*args, **kwargs):
    return (lambda: [_GLOBAL_GUARDED(*args, **kwargs) for _ in (0,)][0])()


@_wraps(_global_lookup)
def _calls_a_global_dispatcher(*args, **kwargs):
    return _GLOBAL_DISPATCHER(*args, **kwargs)


@_wraps(_global_fetch)
def _calls_global_helpers(*args, **kwargs):
    _GLOBAL_LOG.debug("fetching")
    _GLOBAL_HELPER(args)
    _GLOBAL_FORWARDS("u")  # a second wrapper of the same function with no parameter of its own
    return _global_fetch(*args, **kwargs)


@_wraps(_global_fetch)
def _calls_globals_of_another_function(*args, **kwargs):
    _GLOBAL_GUARDED_OTHER("u")
    _GLOBAL_PARTIAL_OF_OTHER("u")
    _GLOBAL_DISPATCHER_OTHER(1)
    _GLOBAL_DISPATCHER_BASE_ONLY("k")
    return _global_fetch(*args, **kwargs)


def test_a_second_wrapper_in_a_module_global_the_wrapper_names_is_refused():
    """``_GLOBAL_GUARDED = _with_timeout(_global_fetch)`` keeps ``timeout`` for itself, and
    the wrapper that names ``_global_fetch`` as ``__wrapped__`` calls it by its global
    name: judged ``timeout=5``, ran the function's own default. Refused with the reason a
    wrapper the walk reads would get, also when the name is used inside a lambda or a
    comprehension of the wrapper."""
    for tool in (_calls_a_global_wrapper, _calls_a_global_wrapper_in_a_lambda):
        assert tool("u", timeout=5) == 999999
        with pytest.raises(HiddenWrapper) as refused:
            reached_parameters(tool)
        assert str(refused.value).startswith("`timeout` is taken by a wrapper and by the function")
        assert unsupported_signature(tool) == str(refused.value)


def test_a_dispatcher_in_a_module_global_the_wrapper_names_is_refused():
    """The wrapper names the base as ``__wrapped__`` and calls the dispatcher of that base
    by its global name: the registered implementation ran its own ``skip_review=True``."""
    assert _calls_a_global_dispatcher(1) == "int, True"
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(_calls_a_global_dispatcher)
    assert str(refused.value) == invocation._HIDDEN_WRAPPER
    assert unsupported_signature(_calls_a_global_dispatcher) == invocation._HIDDEN_WRAPPER


def test_what_a_wrapper_calls_by_a_global_name_is_otherwise_not_refused():
    """Only a second wrapper or a dispatcher of a callable the walk read counts, and only
    with a parameter of its own (or more than its base to run). A logger, a plain function,
    a forwarding wrapper of the same function, and a wrapper, a ``functools.partial`` and a
    registered dispatcher of ANOTHER function are what an ordinary decorator calls."""
    for tool in (_calls_global_helpers, _calls_globals_of_another_function):
        assert tool("u", timeout=5) == 5
        assert _levels(tool) == [_FORWARD, _FETCH]
        assert unsupported_signature(tool) is None


def test_the_globals_of_a_wrapper_are_read_without_running_anything():
    """The namespace is the function's own ``__globals__``, read only when it is exactly a
    ``dict``, listed in one step, and no key in it is hashed or compared unless it is
    exactly a ``str``. A namespace of any other type is not read: a stated limit."""
    import types

    class Key:
        hashed = True  # until the namespace holds it

        def __hash__(self):
            if not Key.hashed:
                raise AssertionError("a key of the namespace was hashed")
            return 7

        def __eq__(self, other):
            raise AssertionError("a key of the namespace was compared")

    code = _calls_a_global_wrapper.__code__
    assert "_GLOBAL_GUARDED" in invocation._names(code)
    namespace = {"_GLOBAL_GUARDED": _GLOBAL_GUARDED}
    namespace[Key()] = _GLOBAL_GUARDED
    Key.hashed = False
    hostile = types.FunctionType(code, namespace)
    hostile.__wrapped__ = _global_fetch
    with pytest.raises(HiddenWrapper):
        reached_parameters(hostile)

    class Namespace(dict):
        def items(self):
            raise AssertionError("a namespace that is not a dict was read")

    other = types.FunctionType(code, Namespace(_GLOBAL_GUARDED=_GLOBAL_GUARDED))
    other.__wrapped__ = _global_fetch
    assert other("u", timeout=5) == 999999
    assert _levels(other) == [_FORWARD, _FETCH]  # not found: the stated limit

    # a name the code does not use is not read, whatever the namespace holds
    unused = types.FunctionType(_calls_global_helpers.__code__, dict(globals()))
    unused.__wrapped__ = _global_fetch
    assert "_GLOBAL_GUARDED" not in invocation._names(unused.__code__)
    assert _levels(unused) == [_FORWARD, _FETCH]

    # with the checks off nothing held is collected, as for a closure cell
    ready = invocation._READY
    try:
        invocation._READY = False
        assert _levels(_calls_a_global_wrapper) == [_FORWARD, _FETCH]
    finally:
        invocation._READY = ready
    with pytest.raises(HiddenWrapper):
        reached_parameters(_calls_a_global_wrapper)


def test_a_global_named_only_as_an_attribute_is_read_too():
    """The code object lists the names a function reads as globals and as attributes in one
    tuple. A wrapper that reads an attribute called like a global of its module is judged
    by that global as well: stated, and only ever a wrapper of the same function."""

    class Settings:
        _GLOBAL_GUARDED = None

    @_wraps(_global_fetch)
    def wrapper(*args, **kwargs):
        Settings._GLOBAL_GUARDED  # noqa: B018, an attribute, not the global
        return _global_fetch(*args, **kwargs)

    assert wrapper("u", timeout=5) == 5
    with pytest.raises(HiddenWrapper):
        reached_parameters(wrapper)


# --- review round six ----------------------------------------------------------------------
#
# E: a ``__wrapped__`` a ``functools.partial`` carries. A: a ``functools.lru_cache`` wrapper
# that names one function and calls another. B: what the ``__call__`` of a class-based
# decorator holds. C: a bound argument a wrapper takes under another name. D: the
# partialmethod check where ``functools.partialmethod`` is not the standard library's class.

_SIX: List[Any] = []


def _six_fetch(query: str, limit: int = 10) -> str:
    _SIX.append(("fetch", query, limit))
    return "rows"


def _six_audited(function):
    import functools

    @functools.wraps(function)
    def wrapper(*args, skip_review: bool = True, **kwargs):
        _SIX.append(("audited", skip_review))
        return function(*args, **kwargs)

    return wrapper


def _six_capped(function):
    import functools

    @functools.wraps(function)
    def wrapper(*args, limit: int = 1000, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def _six_call_with_retry(function, *args, **kwargs):
    return function(*args, **kwargs)


def _six_retry(function):
    """A decorator with no nested function: the helper is bound with ``functools.partial``,
    which then carries the ``__wrapped__``."""
    import functools

    return functools.wraps(function)(functools.partial(_six_call_with_retry, function))


_SIX_UNDECLARED = (
    "`skip_review` is a parameter of a function the call reaches that the tool does not"
)
_SIX_KEPT = "`limit` is taken by a wrapper and by the function it wraps"


def test_e_a_wrapped_carried_by_a_partial_is_followed():
    """``functools.wraps(f)(functools.partial(helper, f))`` shows the signature of ``f``,
    and the walk followed the partial's function (the helper) alone: the wrapper under it
    was not read. Judged ``limit=5``, ran ``limit=10``; or ran ``skip_review=True``."""
    import functools

    hidden = _six_retry(_six_audited(_six_fetch))
    _SIX.clear()
    hidden(query="q", limit=5)
    assert _SIX == [("audited", True), ("fetch", "q", 5)]
    assert unsupported_signature(hidden).startswith(_SIX_UNDECLARED)
    # the partial (the helper, less the function it binds), then what it names
    assert _levels(hidden) == [_FORWARD, ["args", "skip_review", "kwargs"], ["query", "limit"]]

    keeps = _six_retry(_six_capped(_six_fetch))
    _SIX.clear()
    keeps(query="q", limit=5)
    assert _SIX == [("fetch", "q", 10)]
    assert unsupported_signature(keeps).startswith(_SIX_KEPT)

    # the decorator alone is read, and accepted: the partial, then the function it names
    alone = _six_retry(_six_fetch)
    assert _levels(alone) == [_FORWARD, ["query", "limit"]]
    assert unsupported_signature(alone) is None
    # so is a partial that carries nothing, as before
    plain = functools.partial(_six_fetch)
    assert _levels(plain) == [["query", "limit"]]
    # and one behind a method, and under a forwarding wrapper
    assert unsupported_signature(_traced(alone)) is None
    assert unsupported_signature(_traced(hidden)).startswith(_SIX_UNDECLARED)

    # what a partial that carries a ``__wrapped__`` holds is judged like any other holder's
    def holding(function):
        guarded = _with_timeout(function)
        relay = functools.partial(_six_call_with_retry, function)
        relay.guarded = guarded  # an instance attribute of the partial
        return functools.wraps(function)(relay)

    with pytest.raises(HiddenWrapper):
        reached_parameters(holding(_fetch()))


def _six_cached(make_wrapper, **options):
    """A caching decorator written as ``functools.wraps(f)`` over ``functools.lru_cache``
    over a wrapper: the cache is named for ``f`` and calls the wrapper."""
    import functools

    def decorate(function):
        wrapper = functools.lru_cache(**options)(make_wrapper(function))
        return functools.update_wrapper(wrapper, function)

    return decorate


def _six_forwarder(function):
    return lambda *args, **kwargs: function(*args, **kwargs)


_CACHES = (
    {"maxsize": 128},
    {"maxsize": None},
    {"maxsize": 0},
    {"maxsize": 2, "typed": True},
)


def test_a_what_a_cache_refers_to_is_laid_out_as_the_walk_reads_it():
    """The function a ``functools.lru_cache`` wrapper calls is picked out of
    ``gc.get_referents`` by its place. This holds every interpreter to the two layouts
    the kernel knows, for every kind of cache, empty and used, with functions and the
    named function itself among the cached keys and results: a Python that lists them
    another way fails here first."""
    import functools
    import gc

    reached_parameters(_six_fetch)  # the first walk has been made
    results: dict = {}

    def called(key, limit: int = 10):
        return results.get(key, key)

    def named(key, limit: int = 10):
        return key

    for options in _CACHES:
        cache = functools.lru_cache(**options)(called)
        states = [gc.get_referents(cache)]
        cache(1), cache("a")
        states.append(gc.get_referents(cache))
        results["f"], results["n"] = _six_fetch, named
        cache("f"), cache("n"), cache(named), cache(_six_fetch)  # functions cached
        states.append(gc.get_referents(cache))
        functools.update_wrapper(cache, named)  # named again for another function
        states.append(gc.get_referents(cache))
        for referred in states:
            assert invocation._called_by(cache, referred) is called, (options, referred)
            assert referred[-1] is cache.__dict__ and type(referred[-2]) is type
            if sys.version_info >= (3, 10):
                assert referred[0] is type(cache) and referred[-5] is called
                assert type(referred[-4]) is object and type(referred[-6]) is dict
                assert type(referred[-3]) is type
            else:
                assert referred[-4] is called and type(referred[-3]) is dict
        assert invocation._cache_calls(cache, named) is called
        assert invocation._cache_target(cache) is called
    # a bounded cache lists its keys and results before its dictionary, never after
    bounded = functools.lru_cache(maxsize=8)(called)
    bounded("f")
    listed = gc.get_referents(bounded)
    tail = 6 if sys.version_info >= (3, 10) else 4
    assert any(value is _six_fetch for value in listed[:-tail])
    assert not any(value is _six_fetch for value in listed[-tail:])
    # what is not laid out either way is not read: each thing that stands around the
    # function is held to, by exact type or by identity
    for odd in ([], [called], listed[:-1], [*listed, 1], tuple(listed), None):
        assert invocation._called_by(bounded, odd) is invocation._MISSING
    around = (0, -1, -2, -3, -4, -6) if sys.version_info >= (3, 10) else (-1, -2, -3)
    for index in around:
        altered = list(listed)
        altered[index] = "not what stands there"
        assert invocation._called_by(bounded, altered) is invocation._MISSING, index


def test_a_cache_named_again_is_read_as_the_wrapper_it_calls():
    """``@functools.wraps(f)`` over ``@functools.lru_cache`` over a wrapper overwrites the
    cache's ``__wrapped__`` with ``f``; the walk read the cache as ``f``, and the wrapper
    it really calls, held only in C, was read by nothing. The wrapper is found among what
    the cache refers to and read: one that only passes its arguments on is accepted, and
    one with a parameter of its own gets the refusal it would get without the cache."""
    own = ["query", "limit"]
    for options in _CACHES:
        passing = _six_cached(_six_forwarder, **options)(_six_fetch)
        assert passing("q", 5) == "rows"
        assert _levels(passing) == [_FORWARD, own]
        assert unsupported_signature(passing) is None
        assert _levels(_traced(passing)) == [_FORWARD, _FORWARD, own]

        audited = _six_cached(_six_audited, **options)(_six_fetch)
        assert _levels(audited) == [["args", "skip_review", "kwargs"], own, own]
        assert unsupported_signature(audited).startswith(_SIX_UNDECLARED)
        assert unsupported_signature(_traced(audited)).startswith(_SIX_UNDECLARED)
        capped = _six_cached(_six_capped, **options)(_six_fetch)
        assert unsupported_signature(capped).startswith(_SIX_KEPT)
    _SIX.clear()
    _six_cached(_six_capped, maxsize=128)(_six_fetch)(query="q", limit=5)
    assert _SIX == [("fetch", "q", 10)]

    class Service:
        @_six_cached(_six_audited, maxsize=128)
        def fetch(self, query: str, limit: int = 10) -> str:
            return "rows"

        @_six_cached(_six_forwarder, maxsize=128)
        def passing(self, query: str, limit: int = 10) -> str:
            return "rows"

        def __hash__(self):
            return 1

    assert unsupported_signature(Service().fetch).startswith(_SIX_UNDECLARED)
    assert _levels(Service().passing) == [_FORWARD, own]
    assert unsupported_signature(Service().passing) is None

    # the wrapper the cache calls is read like one that names the function itself: a
    # second wrapper it holds in its closure is found, and a dispatcher it is, refused
    def holding(function):
        guarded = _with_timeout(function)
        return lambda *args, **kwargs: guarded(*args, **kwargs)

    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(_six_cached(holding)(_fetch()))
    assert str(refused.value).startswith("`timeout` is taken by a wrapper and by the function")
    lookup = _dispatching()
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(_six_cached(lambda function: lookup)(lookup.__wrapped__))

    # a cache a wrapper holds, named for the function and calling a wrapper with a
    # parameter of its own, is a hidden wrapper; one that only passes on is not
    with pytest.raises(HiddenWrapper):
        reached_parameters(_calling(_six_cached(_with_timeout))(_fetch()))
    assert _levels(_calling(_six_cached(_six_forwarder))(_fetch())) == [_FORWARD, _FETCH]


def test_a_cache_that_calls_what_it_names_is_read_as_before():
    """The ordinary uses: a cache on the function, used and full, bounded and not, on a
    method, under a wrapper, and named again for the same function."""
    import functools

    own = ["query", "limit"]

    def fetch(query: str, limit: int = 10) -> str:
        return "rows"

    for options in _CACHES:
        cached = functools.lru_cache(**options)(fetch)
        for query in ("a", "b", "c", fetch):  # links, evictions, and the function as a key
            cached(query)
        assert _levels(cached) == [own, own]
        assert unsupported_signature(cached) is None
        assert _levels(_traced(cached)) == [_FORWARD, own, own]
        assert _levels(functools.update_wrapper(cached, fetch)) == [own, own]

    class Service:
        @functools.lru_cache(maxsize=None)  # noqa: B019, the shape under test
        def fetch(self, query: str, limit: int = 10) -> str:
            return "rows"

    assert _levels(Service().fetch) == [own, own]
    assert invocation._cache_calls(fetch, fetch) is fetch  # not a cache: the name stands


def test_a_method_over_a_cache_named_for_a_cache_is_read_as_0_15_0_read_it():
    """0.15.0 read a method over a cache whose ``__wrapped__`` is not a plain function (a
    cache of a cache) by the names alone and without the object bound, which shows the
    method's ``self`` as a parameter the tool does not declare, and refused it. The call
    is clean, and the reading is kept all the same: nothing 0.15.0 refused is accepted
    now. A cache named for a plain function is read as the wrapper it calls, bound."""
    import functools

    def method(self, query: str, limit: int = 10) -> str:
        return "rows"

    undeclared = "`self` is a parameter of a function the call reaches that the tool does not"

    def holder(attribute):
        return type("Service", (), {"fetch": attribute, "__hash__": lambda self: 1})()

    twice = functools.lru_cache(maxsize=None)(functools.lru_cache(maxsize=None)(method))
    over_a_cache = _six_cached(_six_forwarder)(functools.lru_cache(maxsize=None)(method))
    for attribute in (twice, over_a_cache, functools.lru_cache(maxsize=None)(over_a_cache)):
        tool = holder(attribute).fetch
        assert tool("q") == "rows"
        assert unsupported_signature(tool).startswith(undeclared), attribute

    # the wrapper such a cache calls is still read: a parameter of its own is refused
    # by name, also where the names alone would have been accepted
    class Passing:
        def __init__(self, function):
            functools.update_wrapper(self, function)
            self.function = function

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    over_an_object = holder(_six_cached(_six_audited)(Passing(method))).fetch
    assert unsupported_signature(over_an_object).startswith(_SIX_UNDECLARED)
    assert unsupported_signature(holder(_six_cached(_six_forwarder)(Passing(method))).fetch) is None
    # named for the plain function: the wrapper the cache calls, with the object bound
    named_for_it = holder(_six_cached(_six_forwarder)(method)).fetch
    assert _levels(named_for_it) == [_FORWARD, ["query", "limit"]]
    assert unsupported_signature(named_for_it) is None


def test_what_a_cache_calls_is_told_once_and_kept_weakly(monkeypatch):
    """A cache calls the same function for as long as it lives, and a full bounded cache
    lists every key and result along with it. So the listing is made once per cache and
    what it told is kept, without keeping the cache or the function alive."""
    import functools
    import gc
    import weakref

    reached_parameters(_six_fetch)  # the first walk has been made
    real, listed = invocation._referents, []

    def counting(value):
        listed.append(value)
        return real(value)

    monkeypatch.setattr(invocation, "_referents", counting)

    def called(query: str, limit: int = 10) -> str:
        return "rows"

    plain = functools.lru_cache(maxsize=4)(called)
    renamed = _six_cached(_six_forwarder, maxsize=4)(_six_fetch)
    for cache in (plain, renamed):
        del listed[:]
        for _ in range(3):
            reached_parameters(cache)
            unsupported_signature(cache)
            cache("q")
        assert len(listed) == 1 and listed[0] is cache
    assert invocation._told(plain) is called
    assert invocation._told(called) is invocation._MISSING  # nothing was told of it

    # a callable that cannot be weakly referenced is told again each time, and read
    class Slotted:
        __slots__ = ()

        def __call__(self, query: str, limit: int = 10) -> str:
            return "rows"

    slotted = Slotted()
    unkept = functools.lru_cache(maxsize=None)(slotted)
    del listed[:]
    assert invocation._cache_calls(unkept, slotted) is slotted
    assert invocation._cache_calls(unkept, slotted) is slotted
    assert len(listed) == 2 and invocation._told(unkept) is invocation._MISSING

    # neither the cache nor the function lives longer for having been told
    cache_gone, function_gone = weakref.ref(plain), weakref.ref(called)
    del plain, called, cache, listed[:]
    gc.collect()
    assert cache_gone() is None and function_gone() is None


def test_a_cache_whose_function_cannot_be_told_is_refused_or_read_as_before(monkeypatch):
    """A bounded cache that holds the named function as a cached result is told apart by
    place, not by presence. Where the list is laid out another way, the cache is refused
    whatever is listed, a cache on the function itself included: what it calls cannot be
    told. And where the listing is not to be had, or raises, nothing is known and the
    cache is read as 0.15.0 reads it."""
    import gc
    import types

    own = ["query", "limit"]
    returning = _six_cached(lambda f: lambda *a, **k: _six_fetch, maxsize=8)(_six_fetch)
    assert returning("q") is _six_fetch  # the named function, now a cached result
    assert _levels(returning) == [["a", "k"], own]

    tool = _six_cached(_six_audited)(_six_fetch)
    reached_parameters(_six_fetch)  # the first walk has been made
    real = invocation._referents
    monkeypatch.setattr(invocation, "_referents", lambda value: real(value)[:-1])
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    assert type(refused.value) is invocation.UnreadCache
    assert issubclass(invocation.UnreadCache, HiddenWrapper)
    assert str(refused.value) == invocation._REWRAPPED_CACHE
    assert "register the function itself, without the cache" in str(refused.value)
    assert unsupported_signature(tool) == invocation._REWRAPPED_CACHE
    with pytest.raises(HiddenWrapper):
        invocation.own_parameters(tool)
    with pytest.raises(HiddenWrapper):  # held by a wrapper: the way cannot be judged
        reached_parameters(_calling(_six_cached(_six_forwarder))(_fetch()))
    plain = __import__("functools").lru_cache(maxsize=None)(_six_fetch)
    with pytest.raises(invocation.UnreadCache):  # the named function is listed: no proof
        reached_parameters(plain)

    monkeypatch.setattr(invocation, "_referents", None)
    assert _levels(tool) == [own, own]

    def refusing(value):
        raise RuntimeError("an audit hook refused the listing")

    monkeypatch.setattr(invocation, "_referents", refusing)
    assert _levels(tool) == [own, own]
    monkeypatch.undo()
    assert _levels(tool) == [["args", "skip_review", "kwargs"], own, own]

    assert invocation._referents is gc.get_referents
    assert invocation._collector() is gc.get_referents
    monkeypatch.setattr(gc, "get_referents", lambda value: [])
    assert invocation._collector() is None  # not the interpreter's own function
    monkeypatch.undo()
    stand_in = types.SimpleNamespace(get_referents=gc.get_referents)
    monkeypatch.setattr(invocation, "_gc", stand_in)
    assert invocation._collector() is None  # not a module


_SIX_GUARDED = _with_timeout(_six_fetch)


def test_b_what_the_call_of_a_class_based_decorator_holds_is_read():
    """A class written inside the decorator, whose ``__call__`` closes over a second
    wrapper: the object's instance attributes were read and the cells of its ``__call__``
    were not. Judged ``timeout`` never; ran ``timeout=30``. The same for a class at module
    level whose ``__call__`` names the wrapper as a global."""
    import functools
    import logging

    def tool_decorator(function, held):
        class Tool:
            def __init__(self):
                functools.update_wrapper(self, function)

            def __call__(self, *args, **kwargs):
                return held(*args, **kwargs)

        return Tool()

    fetch = _fetch()
    hidden = tool_decorator(fetch, _with_timeout(fetch))
    assert hidden("u", timeout=5) == 999999
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(hidden)
    assert str(refused.value).startswith("`timeout` is taken by a wrapper and by the function")

    class ToolGlobal:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return _SIX_GUARDED(*args, **kwargs)

    with pytest.raises(HiddenWrapper):
        reached_parameters(ToolGlobal(_six_fetch))

    class Base:
        def __call__(self, *args, **kwargs):
            return _SIX_GUARDED(*args, **kwargs)

    class Inherits(Base):  # the ``__call__`` Python finds, on a base class
        def __init__(self, function):
            functools.update_wrapper(self, function)

    with pytest.raises(HiddenWrapper):
        reached_parameters(Inherits(_six_fetch))

    # what an ordinary ``__call__`` closes over or names is not refused
    log = logging.getLogger("recusal.tests.invocation")
    for held in (fetch, _traced(fetch), lambda *a, **k: fetch(*a, **k), log.debug):
        tool = tool_decorator(fetch, held)
        assert _levels(tool) == [_FORWARD, _FORWARD, _FETCH]
    assert _levels(ToolGlobal(_global_other)) == [_FORWARD, _FORWARD, _FETCH]
    assert invocation._class_attribute(Inherits, "__call__") is Base.__dict__["__call__"]
    assert invocation._class_attribute(Inherits, "no_such_name") is invocation._MISSING


def _six_search(query: str = "*", limit: int = 10) -> str:
    _SIX.append(("search", query, limit))
    return "rows"


def test_c_a_bound_argument_a_wrapper_takes_under_another_name_is_refused():
    """A method that delegates to a free function and carries ``functools.wraps`` of it:
    the walk bound the object to the free function's first parameter, as if the wrapper
    handed it on. It keeps it: declared ``(limit=10)``, judged ``{'limit': 5}``, ran
    ``query='*'``, a parameter no signature showed any more."""
    import functools

    class Tools:
        @functools.wraps(_six_search)
        def search(self, *args, **kwargs):
            return _six_search(*args, **kwargs)

        @classmethod
        @functools.wraps(_six_search)
        def search_cm(cls, *args, **kwargs):
            return _six_search(*args, **kwargs)

    class CallableTool:
        @functools.wraps(_six_search)
        def __call__(self, *args, **kwargs):
            return _six_search(*args, **kwargs)

    def with_session(function):
        @functools.wraps(function)
        def wrapper(session, *args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    method = Tools().search
    _SIX.clear()
    method(limit=5)
    assert _SIX == [("search", "*", 5)] and list(inspect_signature(method)) == ["limit"]
    cases = [
        (method, "self"),
        (Tools.search_cm, "cls"),
        (CallableTool(), "self"),
        (functools.partial(with_session(_six_search), object()), "session"),
    ]
    if "query" not in inspect_signature(_traced(method)):  # 3.9 declares it: see FR8 below
        cases.append((_traced(method), "self"))
    for tool, taken in cases:
        with pytest.raises(invocation.BoundArgumentKept) as refused:
            reached_parameters(tool)
        text = str(refused.value)
        assert (
            f"is taken by a wrapper as `{taken}`" in text
            and "`query`, the last they would fill" in text
        )
        assert "Register the function the wrapper calls itself" in text
        assert "*" not in text.replace("`*", "")
        assert issubclass(invocation.BoundArgumentKept, ValueError)
    with pytest.raises(invocation.BoundArgumentKept) as refused:
        reached_parameters(method)
    assert unsupported_signature(method) == str(refused.value)

    # the second of two bound arguments, where the first is handed on under its own name
    def target(first, query: str = "*", limit: int = 10) -> str:
        return query

    @functools.wraps(target)
    def keeps_the_second(first, session, *args, **kwargs):
        return target(first, *args, **kwargs)

    assert functools.partial(keeps_the_second, 1, object())() == "*"
    with pytest.raises(invocation.BoundArgumentKept, match="`session`.*`query`"):
        reached_parameters(functools.partial(keeps_the_second, 1, object()))
    reached_parameters(functools.partial(keeps_the_second, 1))  # the first alone is its own


def test_c_an_ordinary_bound_method_or_partial_is_not_refused():
    """The commonest tool shapes: a method of a class, decorated or not, cached, a
    ``classmethod``, a wrapper that names ``self``, a partial that binds a parameter with
    a default through a forwarding wrapper or through one that restates it."""
    import functools

    def naming_self(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            return function(self, *args, **kwargs)

        return wrapper

    def another_name(function):
        @functools.wraps(function)
        def wrapper(this, *args, **kwargs):
            return function(this, *args, **kwargs)

        return wrapper

    def restating(function):
        @functools.wraps(function)
        def wrapper(table: str = "users", limit: int = 10) -> str:
            return function(table, limit)

        return wrapper

    def required(query: str, limit: int = 10) -> str:
        return "rows"

    class Service:
        def plain(self, query: str, limit: int = 10) -> str:
            return "rows"

        @_traced
        def traced(self, query: str, limit: int = 10) -> str:
            return "rows"

        @naming_self
        def named(self, query: str, limit: int = 10) -> str:
            return "rows"

        @another_name
        def other(self, query: str, limit: int = 10) -> str:
            return "rows"

        @functools.lru_cache(maxsize=None)  # noqa: B019, the shape under test
        def cached(self, query: str, limit: int = 10) -> str:
            return "rows"

        @classmethod
        @_traced
        def on_class(cls, query: str, limit: int = 10) -> str:
            return "rows"

        @functools.wraps(required)
        def delegating(self, *args, **kwargs):  # the first parameter has no default
            return required(*args, **kwargs)

    service = Service()
    for name in ("plain", "traced", "named", "cached", "on_class"):
        tool = getattr(service, name)
        reached_parameters(tool)
        assert unsupported_signature(tool) is None, name
    assert _levels(service.plain) == [["query", "limit"]]
    # since 0.17.1: a wrapper that takes the receiver under another name than the method
    # has for it, which a call by keyword that kept it fails on and one by position does
    # not. ``other`` hands it on and runs correctly; ``delegating`` keeps it, and the tool
    # declares only ``limit``, so by position the judged ``limit`` runs as ``query``
    assert service.other("q", 5) == "rows"
    with pytest.raises(invocation.BoundArgumentKept, match="`this`.*`self`, the last"):
        reached_parameters(service.other)
    if list(inspect_signature(service.delegating)) == ["limit"]:  # 3.9 shows ``query`` too
        with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`query`, the last"):
            reached_parameters(service.delegating)

    def table(table: str = "users", limit: int = 10) -> str:
        return table

    for tool in (
        functools.partial(_traced(table), "orders"),
        functools.partial(restating(table), "orders"),
        functools.partial(table, "orders"),
        functools.partial(_traced(table), limit=3),
    ):
        assert tool() == "orders" or tool() == "users"
        reached_parameters(tool)

    # a partial that binds nothing asks nothing of what its function names, which need
    # not be a callable the walk reads unbound (a class, as the ``decorator`` library
    # names one): read as before
    class Manager:
        def __init__(self, table: str = "users"):
            self.table = table

    def make(table: str = "users"):
        return Manager(table)

    make.__wrapped__ = Manager
    before = _levels(functools.partial(make))
    assert before == [["table"], ["table"]]


def test_d_a_replaced_partialmethod_leaves_every_walk_as_it_was(monkeypatch):
    """The partialmethod check asked ``isinstance(..., functools.partialmethod)`` of every
    callable: with a tracing function, None, nothing, or a class with a metaclass of its
    own under that name, every ordinary tool was refused, where 0.15.0 reads them. It asks
    only once a mark is there, and only of a plain class."""
    import functools

    real = functools.partialmethod
    asked: list = []

    class Meta(type):
        def __instancecheck__(cls, instance):
            asked.append(instance)
            raise RuntimeError("an instance check of the application's")

    def tracing(func, *args, **kwargs):
        return real(func, *args, **kwargs)

    class Service:
        def fetch(self, query: str, limit: int = 10) -> str:
            return "rows"

    class Repo:
        marked = real(_Fetcher(), "customers")

    marked = Repo().marked
    tools = (_six_fetch, _traced(_six_fetch), Service().fetch, functools.lru_cache(_six_fetch))
    expected = [_levels(tool) for tool in tools]
    for stand_in in (tracing, None, object(), Meta("partialmethod", (), {}), "missing"):
        if stand_in == "missing":
            monkeypatch.delattr(functools, "partialmethod")
        else:
            monkeypatch.setattr(functools, "partialmethod", stand_in)
        assert [_levels(tool) for tool in tools] == expected
        assert unsupported_signature(_six_fetch) is None
        # a callable with a mark is left to ``inspect.signature``, which raises there as
        # it did in 0.15.0, or reads the mark: either way this check refuses nothing
        try:
            reached_parameters(marked)
        except PartialMethodSignature:  # pragma: no cover, the defect
            raise AssertionError("the check asked a stand-in") from None
        except Exception:  # noqa: BLE001, S110, ``inspect.signature`` itself, as before
            pass
        monkeypatch.undo()
    # ``inspect.signature`` asks the stand-in class about a marked callable, as it did in
    # 0.15.0; this check asked nothing of it, and nothing at all for an unmarked callable
    assert len(asked) == 1 and type(asked[0]) is real
    with pytest.raises(PartialMethodSignature):
        reached_parameters(marked)

    # a stand-in class that answers quietly is asked once, by ``inspect.signature``, and
    # never by this check
    del asked[:]

    class Quiet(type):
        def __instancecheck__(cls, instance):
            asked.append(instance)
            return False

    monkeypatch.setattr(functools, "partialmethod", Quiet("partialmethod", (), {}))
    reached_parameters(marked)
    assert len(asked) == 1
    monkeypatch.undo()

    class Sub(real):
        pass

    monkeypatch.setattr(functools, "partialmethod", Sub)  # still a plain class
    assert _levels(_six_fetch) == [["query", "limit"]]


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


_NOT_FINITE = "non-finite numbers are not JSON data"
_ENUM = (
    "Enum arguments are unsupported; declare data-only parameters "
    "and resolve application types inside the tool body"
)
_NOT_DATA = "arguments must be data-only"
_CYCLIC = "cyclic arguments are not JSON data"
_KEYS = "argument keys must be strings"


def _cycle_in_a_list():
    loop: List[Any] = [1]
    loop.append(loop)
    return loop


def _cycle_below():
    holder: Dict[str, Any] = {"a": [1, {"b": []}]}
    holder["a"][1]["b"].append(holder["a"])  # not the outermost: a list inside itself
    return holder


@pytest.mark.parametrize(
    ("make", "message"),
    [
        # each refusal, alone and at the top
        (lambda: float("nan"), _NOT_FINITE),
        (lambda: [float("-inf")], _NOT_FINITE),
        (lambda: _Color.RED, _ENUM),
        (lambda: {"k": _IntKind.ONE}, _ENUM),
        (lambda: (1, 2), _NOT_DATA),
        (lambda: [_StrSub("x")], _NOT_DATA),
        (lambda: [type("FloatSub", (float,), {})(1.5)], _NOT_DATA),
        (lambda: [type("IntSub", (int,), {})(3)], _NOT_DATA),
        (lambda: {"k": _DictSub(a=1)}, _NOT_DATA),
        (lambda: {1: "a"}, _KEYS),
        (lambda: [{"a": 1, _StrSub("b"): 2}], _KEYS),
        (_cycle_in_a_list, _CYCLIC),
        (_cycle_below, _CYCLIC),
        # the first one met, depth first, in the order of the members
        (lambda: [float("nan"), _Color.RED], _NOT_FINITE),
        (lambda: [_Color.RED, float("nan")], _ENUM),
        (lambda: [[1, (1,)], float("nan")], _NOT_DATA),
        (lambda: {"a": {"b": [_Color.RED]}, "c": (1,)}, _ENUM),
        (lambda: [_cycle_in_a_list(), (1,)], _CYCLIC),
        (lambda: [(1,), _cycle_in_a_list()], _NOT_DATA),
        # every key of an object is looked at before any of its members is copied
        (lambda: {"a": float("nan"), 1: 2}, _KEYS),
        (lambda: {"a": [float("nan")], None: 1}, _KEYS),
        # and the members of the objects before it come first
        (lambda: [{"a": float("nan")}, {1: 2}], _NOT_FINITE),
    ],
)
def test_detached_refuses_with_the_first_reason_it_meets(make, message):
    with pytest.raises(TypeError) as refused:
        detached(make())
    assert str(refused.value) == message


def test_detached_copies_every_container_once_for_each_place_it_stands_in():
    """A list or object reachable twice is not a cycle: it is copied at each place, and
    the copies share nothing. Only one that holds itself is refused."""
    shared: List[Any] = [1, {"x": []}]
    original = {"a": shared, "b": shared, "c": [shared, shared, {"d": shared}]}
    copy = detached(original)
    assert copy == original and list(copy) == ["a", "b", "c"]
    copies = [copy["a"], copy["b"], copy["c"][0], copy["c"][1], copy["c"][2]["d"]]
    assert len({id(one) for one in copies}) == 5 and all(one is not shared for one in copies)
    assert (
        len({id(one[1]) for one in copies}) == 5 and len({id(one[1]["x"]) for one in copies}) == 5
    )
    assert detached([[], [], {}, {}]) == [[], [], {}, {}]
    # a scalar is handed back as it is, and a float with it
    for scalar in ("text", 7, True, None, 1.5):
        assert detached(scalar) is scalar
    # ``ancestors``: the containers the value is to be taken as inside of
    inner = [1]
    assert detached(inner, (id(original),)) == [1]
    with pytest.raises(TypeError, match=_CYCLIC):
        detached(inner, (id(inner),))
    with pytest.raises(TypeError, match=_CYCLIC):
        detached({"a": [inner]}, (1, id(inner)))
    assert detached(5, (id(inner),)) == 5  # nothing is asked of them for a scalar

    # they are asked ``in`` at the first container and listed once there, as before: a
    # collection of the caller's own that answers one way and lists another is read so
    class Odd:
        def __contains__(self, item):
            return False

        def __iter__(self):
            return iter((id(inner),))

    assert detached(inner, Odd()) == [1]
    with pytest.raises(TypeError, match=_CYCLIC):
        detached({"a": inner}, Odd())  # type: ignore[arg-type]


def _nested(depth):
    """Lists and objects in turn, ``depth`` of them, around ``"leaf"``: built in a loop."""
    value: Any = "leaf"
    for index in range(depth):
        value = [0, value] if index % 2 else {"k": value, "n": None}
    return value


def _depth_of(value):
    """How deep ``_nested`` data goes, and its innermost value: read in a loop."""
    depth = 0
    while type(value) in (list, dict):
        value = value[1] if type(value) is list else value["k"]
        depth += 1
    return depth, value


def _from_the_bottom_of_the_stack(call):
    """Call ``call`` from as far down the stack as Python lets a function be called, and
    return its result with how many frames had to be given back before it ran."""
    given_back = [0]

    def down():
        try:
            return down()
        except RecursionError:
            given_back[0] += 1
            return call()

    return down(), given_back[0]


def test_detached_copies_data_nested_deeper_than_the_stack():
    """The copy does not call itself, so the depth of the data is not a depth of the
    stack: 100 times the recursion limit is copied (the version that called itself
    raised ``RecursionError`` before the limit)."""
    depth = 100 * sys.getrecursionlimit()
    original = _nested(depth)
    copy = detached(original)
    assert copy is not original and _depth_of(copy) == (depth, "leaf")
    # the copy shares no container with the original, at any depth
    ours, theirs = copy, original
    while type(ours) in (list, dict):
        assert ours is not theirs and type(ours) is type(theirs) and len(ours) == 2
        ours = ours[1] if type(ours) is list else ours["k"]
        theirs = theirs[1] if type(theirs) is list else theirs["k"]
    # a cycle and a value that is not data are still found at the bottom of it
    for leaf, message in ((float("nan"), _NOT_FINITE), ((1,), _NOT_DATA), ({1: 2}, _KEYS)):
        deep: Any = [leaf]
        for _ in range(depth):
            deep = [deep]
        with pytest.raises(TypeError) as refused:
            detached(deep)
        assert str(refused.value) == message
    inner: List[Any] = []
    deep = inner
    for _ in range(depth):
        deep = {"k": deep}
    inner.append(deep)
    with pytest.raises(TypeError) as refused:
        detached(deep)
    assert str(refused.value) == _CYCLIC


def test_detached_runs_from_the_bottom_of_the_stack():
    """Called where almost no frame is left (a deep application stack), the copy needs
    its own frame and no more, whatever the depth of the data. The version that called
    itself needed one frame for every level and raised ``RecursionError`` here."""
    for depth in (0, 3, 200, 20 * sys.getrecursionlimit()):
        original = _nested(depth)
        copy, given_back = _from_the_bottom_of_the_stack(lambda: detached(original))
        assert _depth_of(copy) == (depth, "leaf") and (depth == 0 or copy is not original)
        # a frame for ``call`` and one for ``detached``, with what Python keeps in reserve
        assert given_back <= 8, (depth, given_back)
    # the refusals are the same from there
    for make, message in ((_cycle_below, _CYCLIC), (lambda: [_nested(50), (1,)], _NOT_DATA)):
        with pytest.raises(TypeError) as refused:
            _from_the_bottom_of_the_stack(lambda: detached(make()))
        assert str(refused.value) == message


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
    "adapters/openai-agents-mcp/src/recusal_openai_agents_mcp/_gate.py": {"detached"},
    "adapters/openai-agents-mcp/src/recusal_openai_agents_mcp/_schema.py": {"detached"},
    "adapters/openai-agents/src/recusal_openai_agents/_gate.py": {
        "BoundArgumentKept",
        "DeclaredSignature",
        "DispatchedAtCallTime",
        "HiddenWrapper",
        "PartialMethodSignature",
        "UnreadCache",
        "detached",
        "reached_parameters",
    },
    "adapters/langgraph/src/recusal_langgraph/_gate.py": {
        "DeclaredSignature",
        "reached_parameters",
    },
    "adapters/openai/src/recusal_openai/_gate.py": {
        "detached",
        "json_data",
        "unsupported_signature",
    },
    "adapters/pydantic-ai/src/recusal_pydantic_ai/_gate.py": {
        "DeclaredSignature",
        "detached",
        "json_data",
        "reached_parameters",
    },
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


# --- review round seven --------------------------------------------------------------------
#
# F1: a bound argument a wrapper takes under another name, with layers that only pass
# arguments on between the wrapper and the function. F2: a cache named again, over a method,
# that calls something other than a plain function. R1 to R4: the import of ``gc``, a
# listing an audit hook stops, a replaced ``gc.get_referents``, a listing a collection
# interrupts. And what is kept between walks, the globals a wrapper names, and a reason of
# its own for a callable object that defines ``__get__``.


class _SevenTraced:
    """An ordinary class-based decorator that passes every argument on."""

    def __init__(self, function):
        import functools

        functools.update_wrapper(self, function)
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def _seven_memo(function):
    """An ordinary caching decorator: ``functools.wraps`` over ``functools.lru_cache`` over
    a wrapper that passes every argument on."""
    import functools

    @functools.wraps(function)
    @functools.lru_cache(maxsize=64)
    def inner(*args, **kwargs):
        return function(*args, **kwargs)

    return inner


def _seven_cache_over(function):
    import functools

    return functools.lru_cache(maxsize=None)(_traced(function))


def _seven_partial(function):
    import functools

    return functools.partial(function)


def _seven_with_session(function):
    import functools

    @functools.wraps(function)
    def wrapper(session, *args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


_SEVEN_LAYERS = {
    "a forwarding wrapper": _traced,
    "a class-based decorator": _SevenTraced,
    "a caching decorator": _seven_memo,
    "a cache over a forwarding wrapper": _seven_cache_over,
    "a partial that binds nothing": _seven_partial,
    "a partial that carries the name": _six_retry,
    "two forwarding wrappers": lambda f: _traced(_traced(f)),
    "three layers": lambda f: _SevenTraced(_seven_memo(_traced(f))),
    "four layers": lambda f: _traced(_six_retry(_SevenTraced(_seven_cache_over(f)))),
}


@pytest.mark.parametrize("layer", sorted(_SEVEN_LAYERS))
def test_f1_a_bound_argument_kept_is_refused_through_layers_that_pass_arguments_on(layer):
    """Round six's C came back when the free function wore one ordinary decorator: the
    method takes the object as ``self`` and wraps the decorated function, whose
    ``(*args, **kwargs)`` level has no leading parameter to compare with, and the walk
    bound the object to the free function's first parameter under it. Declared
    ``(limit=10)``, judged ``{'limit': 5}``, ran ``query='*'``. The name the argument was
    taken under now goes with it through every layer that only passes arguments on."""
    import functools

    free = _SEVEN_LAYERS[layer](_six_search)

    class Tools:
        @functools.wraps(free)
        def search(self, *args, **kwargs):
            return free(*args, **kwargs)

        @classmethod
        @functools.wraps(free)
        def search_cm(cls, *args, **kwargs):
            return free(*args, **kwargs)

    method = Tools().search
    _SIX.clear()
    method(limit=5)
    assert _SIX == [("search", "*", 5)] and list(inspect_signature(method)) == ["limit"]
    cases = [(method, "self"), (Tools.search_cm, "cls")]
    if "query" not in inspect_signature(_traced(method)):  # 3.9 declares it: see FR8 below
        cases.append((_traced(method), "self"))
    for tool, taken in cases:
        with pytest.raises(invocation.BoundArgumentKept) as refused:
            reached_parameters(tool)
        text = str(refused.value)
        assert (
            f"is taken by a wrapper as `{taken}`" in text
            and "`query`, the last they would fill" in text
        )
        assert unsupported_signature(tool) is not None
    if layer == "a forwarding wrapper":  # the layers that keep the annotations
        with pytest.raises(invocation.BoundArgumentKept) as refused:
            reached_parameters(method)
        assert unsupported_signature(method) == str(refused.value)
    # behind a ``functools.partial`` the same: one that binds nothing carries the name on,
    # and one that binds the argument has it taken as ``session``. A partial of a cache
    # cannot be read by ``inspect.signature`` and was refused for that before
    session = functools.partial(_seven_with_session(free), object())
    _SIX.clear()
    session(limit=6)
    assert _SIX == [("search", "*", 6)]
    for tool, taken in ((functools.partial(method), "self"), (session, "session")):
        with pytest.raises(ValueError) as refused:
            reached_parameters(tool)
        if not ("cach" in layer or "layers" in layer):
            assert type(refused.value) is invocation.BoundArgumentKept
            assert f"is taken by a wrapper as `{taken}`" in str(refused.value)


def test_f1_the_name_goes_with_each_bound_argument_also_where_partials_merge():
    """``functools.partial(functools.partial(f, a), b)`` is ``functools.partial(f, a, b)``:
    the walk's own rebinding merges two partials, and the name each argument was taken
    under stays with that argument."""
    import functools

    def target(first, query: str = "*", limit: int = 10) -> str:
        return query

    inner = functools.partial(_traced(target), 1)
    tool = functools.partial(_seven_with_session(inner), object())
    assert tool() == "*"  # the session was kept: ``query`` ran at its default
    with pytest.raises(invocation.BoundArgumentKept, match="`session`.*`query`"):
        reached_parameters(tool)
    # a partial the walk bound, merged into the next: its name is kept ahead of the rest
    holds = functools.partial(_seven_with_session(_traced(_six_search)), object())
    holds.note = "a partial with an attribute is not merged when it is built"
    for tool in (functools.partial(holds), functools.partial(_traced(holds))):
        assert type(tool.func) is not functools.partial or tool.func is holds
        with pytest.raises(invocation.BoundArgumentKept, match="`session`.*`query`"):
            reached_parameters(tool)
    # the bound argument a partial on the way holds itself was taken by no wrapper
    reached_parameters(functools.partial(_traced(functools.partial(_traced(target), 1)), "q"))


def test_f1_an_ordinary_method_under_layers_is_read_as_before():
    """What must not be refused: a method decorator that names the receiver and hands it
    on, over any layers; a decorated ``classmethod``. Until 0.17.0 also a decorator that
    names the receiver otherwise than the method, a facade that hands its receiver on
    under another name, and a delegating method whose function takes no default first:
    a call by keyword that kept the receiver fails on each, and a call by position does
    not, so since 0.17.1 each is refused, the first two although they run correctly."""
    import functools

    def naming_self(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            return function(self, *args, **kwargs)

        return wrapper

    def another_name(function):
        @functools.wraps(function)
        def wrapper(instance, *args, **kwargs):
            return function(instance, *args, **kwargs)

        return wrapper

    def required(query: str, limit: int = 10) -> str:
        return "rows"

    class Service:
        @naming_self
        @_traced
        def one(self, query: str, limit: int = 10) -> str:
            return "rows"

        @_traced
        @naming_self
        @_traced
        def two(self, query: str, limit: int = 10) -> str:
            return "rows"

        @another_name
        @_seven_memo
        @_traced
        def three(self, query: str, limit: int = 10) -> str:
            return "rows"

        @naming_self
        @_seven_cache_over
        def four(self, query: str, limit: int = 10) -> str:
            return "rows"

        @classmethod
        @naming_self
        @_traced
        def five(cls, query: str, limit: int = 10) -> str:
            return "rows"

        @another_name
        @another_name
        def six(self, query: str, limit: int = 10) -> str:
            return "rows"

        @functools.wraps(_traced(_traced(required)))
        def delegating(self, *args, **kwargs):  # the first parameter has no default
            return required(*args, **kwargs)

        def __hash__(self):
            return 1

    # the name that counts is the one the nearest wrapper has for the argument: a facade
    # hands its receiver on to a wrapper that names it as the function under it does.
    # 0.17.0 read it, the receiver's ``owner`` on the wrapper having no default; since
    # 0.17.1 that settles nothing, because a call by position that kept the receiver
    # would run ``owner`` with the value judged for ``limit``. Refused although it runs
    # correctly, and read once the facade takes its receiver as ``owner``
    def owned(owner: Any = None, limit: int = 10) -> str:
        return "rows"

    @functools.wraps(owned)
    def naming_owner(owner, *args, **kwargs):
        return owned(owner, *args, **kwargs)

    class Facade:
        @functools.wraps(naming_owner)
        def call(self, *args, **kwargs):
            return naming_owner(self, *args, **kwargs)

        @functools.wraps(naming_owner)
        def named(owner, *args, **kwargs):
            return naming_owner(owner, *args, **kwargs)

    assert Facade().call(limit=5) == "rows" and Facade().call(5) == "rows"
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`owner`, the last"):
        reached_parameters(Facade().call)
    reached_parameters(Facade().named)

    service = Service()
    for name in ("one", "two", "three", "four", "five", "six"):
        tool = getattr(service, name)
        assert tool("q", 5) == "rows"
        if name in ("three", "five", "six"):
            # ``instance`` where the method has ``self``, ``self`` where it has ``cls``:
            # read by 0.17.0, refused since 0.17.1 although it runs correctly (a call by
            # position that kept the receiver would not)
            taken = "self" if name == "five" else "instance"
            with pytest.raises(invocation.BoundArgumentKept, match=f"`{taken}`"):
                reached_parameters(tool)
            continue
        assert unsupported_signature(tool) is None, name
        if name != "four":  # a partial of a cache was never read
            reached_parameters(functools.partial(tool, "q"))
    if list(inspect_signature(service.delegating)) == ["limit"]:  # 3.9 shows ``query`` too
        # it keeps its receiver: by position the judged ``limit`` runs as ``query``
        with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`query`"):
            reached_parameters(service.delegating)

    def table(table: str = "users", limit: int = 10) -> str:
        return table

    for tool in (
        functools.partial(_traced(_traced(table)), "orders"),
        functools.partial(_SevenTraced(_traced(table)), "orders"),
        functools.partial(_six_retry(_SevenTraced(table)), "orders"),
    ):
        assert tool() == "orders"
        reached_parameters(tool)


def test_f2_a_cache_named_again_over_a_method_is_read_as_what_it_calls():
    """A method whose function is a cache named for the method and calling something that
    is not a plain function (a ``functools.partial`` of a wrapper, a callable object, a
    second cache named again) was read by the name: judged ``{'query', 'limit'}``, ran a
    wrapper's ``skip_review=True``. It is read as what it calls, as on a plain function."""
    import functools

    def method(self, query: str, limit: int = 10) -> str:
        _SIX.append(("fetch", query, limit))
        return "rows"

    def over_a_partial(function):
        def wrapper(mode, self, query, limit=10, skip_review=True):
            _SIX.append(("audited", skip_review))
            return function(self, query, limit)

        cached = functools.lru_cache(maxsize=64)(functools.partial(wrapper, "strict"))
        return functools.wraps(function)(cached)

    class Audited:
        def __init__(self, function):
            self.function = function

        def __call__(self, receiver, query, limit=10, skip_review=True):
            _SIX.append(("audited", skip_review))
            return self.function(receiver, query, limit)

    def over_an_object(function):
        return functools.wraps(function)(functools.lru_cache(maxsize=64)(Audited(function)))

    def twice(function):  # wraps(f)(cache(wraps(f)(cache(wrapper))))
        return _six_cached(lambda f: _six_cached(_six_audited)(function))(function)

    def holder(attribute):
        return type("Service", (), {"fetch": attribute, "__hash__": lambda self: 1})()

    for decorate in (over_a_partial, over_an_object, twice):
        tool = holder(decorate(method)).fetch
        _SIX.clear()
        assert tool("q", 5) == "rows" and _SIX == [("audited", True), ("fetch", "q", 5)]
        assert list(inspect_signature(tool)) == ["query", "limit"]
        assert unsupported_signature(decorate(_six_fetch)) is not None  # as on a function
        if decorate is over_an_object:
            # its ``__call__`` takes the receiver as ``receiver`` where the method has
            # ``self``: since 0.17.1 the walk refuses that first, with another reason
            with pytest.raises(invocation.BoundArgumentKept, match="`receiver`"):
                reached_parameters(tool)
            continue
        assert unsupported_signature(tool).startswith(_SIX_UNDECLARED), decorate.__name__
        assert "skip_review" in _levels(tool)[0], decorate.__name__

    # a cache that calls something that only passes arguments on is read, and accepted
    def passing(function):
        relay = functools.partial(_six_call_with_retry, function)
        return functools.wraps(function)(functools.lru_cache(maxsize=64)(relay))

    for decorate in (passing, lambda f: _six_cached(_SevenTraced)(f)):
        tool = holder(decorate(method)).fetch
        assert tool("q", 5) == "rows"
        assert unsupported_signature(tool) is None
    # and a class is not a callable the walk reads, here as anywhere
    renamed = functools.wraps(method)(functools.lru_cache(maxsize=None)(dict))
    with pytest.raises(TypeError, match="a class is not inspected"):
        invocation.own_parameters(holder(renamed).fetch)


_SEVEN_ENVIRONMENT = """
import enum, functools, importlib.util, inspect, json, math, sys, types, typing
scenario, alone = sys.argv[1], sys.argv[2]
RAN = []
class Stop(BaseException):
    pass
if scenario in ("gc-none", "both-none"):
    sys.modules["gc"] = None
if scenario in ("weakref-none", "both-none"):
    sys.modules["weakref"] = None
if scenario == "gc-namespace":
    import gc, types
    sys.modules["gc"] = types.SimpleNamespace(get_referents=gc.get_referents)
elif scenario.startswith("import-"):
    sys.modules.pop("gc", None)
    sys.modules.pop("weakref", None)
    RAISED = {"import-refused": RuntimeError, "import-stopped": Stop}
    NAMES = {"import-interrupted-gc": ("gc",), "import-interrupted-weakref": ("weakref",)}
    def hook(event, args):
        if event == "import" and args[0] in NAMES.get(scenario, ("gc", "weakref")):
            RAN.append("import " + args[0])
            raise RAISED.get(scenario, KeyboardInterrupt)(args[0])
    sys.addaudithook(hook)
elif scenario == "referents-is-referrers":
    import gc
    gc.get_referents = gc.get_referrers
elif scenario == "referents-is-tracked":
    import gc
    gc.get_referents = gc.is_tracked
elif scenario == "referents-python":
    import gc
    REAL = gc.get_referents
    def get_referents(*values):
        RAN.append("get_referents")
        return REAL(*values)
    gc.get_referents = get_referents
elif scenario in ("listing-refused", "listing-stopped"):
    def hook(event, args):
        if event == "gc.get_referents":
            RAN.append("hook")
            raise (RuntimeError if scenario == "listing-refused" else Stop)("no listing")
    sys.addaudithook(hook)
elif scenario.startswith("ref-is-"):
    import weakref
    OTHERS = {"type": type, "str": str, "bool": bool, "proxy": weakref.ProxyType}
    weakref.ref = OTHERS[scenario[len("ref-is-"):]]
elif scenario == "weak-dictionary-replaced":
    import weakref
    def refusing(*args, **kwargs):
        RAN.append("WeakKeyDictionary")
        raise RuntimeError("no weak dictionary")
    weakref.WeakKeyDictionary = weakref.WeakValueDictionary = refusing
try:
    spec = importlib.util.spec_from_file_location("recusal_invocation_alone", alone)
    invocation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(invocation)  # this module alone: what its own import does
    imported = "ok"
except BaseException as error:
    print(json.dumps({"imported": type(error).__name__, "ran": RAN}))
    sys.exit(0)
def fetch(query: str, limit: int = 10):
    return "rows"
def traced(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)
    return wrapper
def memo(function):
    @functools.wraps(function)
    @functools.lru_cache(maxsize=16)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)
    return wrapper
def memo_own(function):
    @functools.wraps(function)
    @functools.lru_cache(maxsize=16)
    def wrapper(*args, skip_review: bool = True, **kwargs):
        return function(*args, **kwargs)
    return wrapper
class Service:
    def fetch(self, query: str, limit: int = 10):
        return "rows"
    @functools.lru_cache(maxsize=None)
    def cached(self, query: str, limit: int = 10):
        return "rows"
# before any walk: a cache read on its own finds what it reads with
first = list(invocation.own_parameters(memo_own(fetch)))
tools = {
    "plain": fetch, "traced": traced(fetch), "method": Service().fetch,
    "lru": functools.lru_cache(maxsize=None)(fetch), "lru-method": Service().cached,
    "memo": memo(fetch), "memo-own": memo_own(fetch), "partial": functools.partial(traced(fetch)),
}
def verdict(tool):
    try:
        invocation.reached_parameters(tool)
        # a partial has no hints to read: its reason is not the walk's, on any kernel
        partial = isinstance(tool, functools.partial)
        why = invocation.unsupported_signature(tool) if not partial else None
    except BaseException as error:
        return "RAISES " + type(error).__name__
    return "accept" if why is None else why[:13]
verdicts = {name: verdict(tool) for name, tool in tools.items()}
again = {name: verdict(tool) for name, tool in tools.items()}
print(json.dumps({
    "imported": imported, "first": first, "verdicts": verdicts, "again": again,
    "ran": sorted(set(RAN)), "referents": invocation._referents is not None,
    "reference": invocation._reference is not None, "kept": len(invocation._KEPT),
}))
"""

_SEVEN_SCENARIOS = {
    # scenario: (the cache is listed, something is kept between walks, what ran)
    "none": (True, True, []),
    "gc-none": (False, True, []),
    "weakref-none": (True, False, []),
    "both-none": (False, False, []),
    "gc-namespace": (False, True, []),
    "import-refused": (False, False, ["import gc", "import weakref"]),
    "import-stopped": (False, False, ["import gc", "import weakref"]),
    # an interrupt from the keyboard is not caught, at either import
    "import-interrupted-gc": "import gc",
    "import-interrupted-weakref": "import weakref",
    "referents-is-referrers": (False, True, []),
    "referents-is-tracked": (False, True, []),
    "referents-python": (False, True, []),
    "listing-refused": (False, True, ["hook"]),
    "listing-stopped": (False, True, ["hook"]),
    "weak-dictionary-replaced": (True, True, []),
    # another built-in type under ``weakref.ref`` is not used: nothing is kept
    "ref-is-type": (True, False, []),
    "ref-is-str": (True, False, []),
    "ref-is-bool": (True, False, []),
    "ref-is-proxy": (True, False, []),
}


@pytest.mark.parametrize("scenario", sorted(_SEVEN_SCENARIOS))
def test_r_the_module_imports_and_walks_wherever_0_15_0_does(scenario):
    """R1 to R3. 0.15.0 imported neither ``gc`` nor ``weakref`` and listed nothing. So where
    a process refuses either import (None in ``sys.modules``, an import or audit hook that
    raises, whatever it raises), has another function under ``gc.get_referents``, or
    stops the listing with an audit hook (whatever that raises), this module is imported
    and every tool is read as 0.15.0 reads it: nothing is raised and nothing is refused.
    Nothing written in Python is called for it: not a function put under
    ``gc.get_referents``, not a stand-in for ``weakref.WeakKeyDictionary``."""
    import json
    import os
    import subprocess

    root = str(pathlib.Path(__file__).resolve().parent.parent)
    alone = os.path.join(root, "recusal", "invocation.py")
    done = subprocess.run(
        [sys.executable, "-c", _SEVEN_ENVIRONMENT, scenario, alone],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    if scenario.startswith("import-interrupted"):
        assert seen == {"imported": "KeyboardInterrupt", "ran": [_SEVEN_SCENARIOS[scenario]]}
        return
    listed, keeps, ran = _SEVEN_SCENARIOS[scenario]
    assert seen["imported"] == "ok"
    assert seen["ran"] == ran
    told = listed or scenario in ("listing-refused", "listing-stopped")
    assert (seen["referents"], seen["reference"]) == (told, keeps)
    # the cache named again is read as what it calls where it can be listed, and by its
    # name, as in 0.15.0, where it cannot; everything else is accepted either way
    expected = {name: "accept" for name in seen["verdicts"]}
    if listed:
        expected["memo-own"] = "`skip_review`"
    assert seen["verdicts"] == expected and seen["again"] == expected
    assert seen["first"] == (["args", "skip_review", "kwargs"] if listed else ["query", "limit"])
    assert (seen["kept"] > 0) == keeps


def test_r_a_listing_that_is_stopped_proves_nothing_and_an_interrupt_is_not_caught(monkeypatch):
    """R2 and R3 on the helpers: whatever stops the listing but an interrupt from the
    keyboard leaves the name standing, and the listing is the interpreter's own function
    by its type, its module and its name."""
    import gc
    import types

    own = ["query", "limit"]
    reached_parameters(_six_fetch)  # the first walk has been made

    class Stop(BaseException):
        pass

    def stopped(value):
        raise Stop("an audit hook stopped the listing")

    monkeypatch.setattr(invocation, "_referents", stopped)
    assert _levels(_six_cached(_six_audited)(_six_fetch)) == [own, own]

    def interrupted(value):
        raise KeyboardInterrupt

    monkeypatch.setattr(invocation, "_referents", interrupted)
    with pytest.raises(KeyboardInterrupt):
        reached_parameters(_six_cached(_six_audited)(_six_fetch))
    monkeypatch.undo()

    assert invocation._collector() is gc.get_referents
    # the built-in under another module object of that name is not that module's own
    another = types.ModuleType("gc")
    another.get_referents = gc.get_referents
    monkeypatch.setattr(invocation, "_gc", another)
    assert invocation._collector() is None
    monkeypatch.undo()
    for other in (gc.get_referrers, gc.is_tracked, gc.get_objects, len):
        monkeypatch.setattr(gc, "get_referents", other)
        assert invocation._collector() is None, other
    monkeypatch.undo()
    assert invocation._collector() is gc.get_referents


def test_r4_a_listing_a_collection_interrupts_is_made_again():
    """On Python 3.9 to 3.11 ``list(d.items())`` is not one step: a collection can start
    at any of the tuples it makes, and a weak-reference callback it runs can change the
    dictionary, which then raises ``RuntimeError``. The listing is made again."""
    import gc
    import weakref

    calls: list = []

    def twice_interrupted():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("dictionary changed size during iteration")
        return iter([("a", 1)])

    assert invocation._listed(twice_interrupted) == [("a", 1)] and len(calls) == 3

    def always_interrupted():
        calls.append(1)
        raise RuntimeError("dictionary changed size during iteration")

    del calls[:]
    with pytest.raises(ValueError, match="was changed each time it was listed"):
        invocation._listed(always_interrupted)
    assert len(calls) == invocation._TRIES == 8

    def another_failure():
        calls.append(1)
        raise TypeError("not what a collection does to a listing")

    del calls[:]
    with pytest.raises(TypeError):
        invocation._listed(another_failure)
    assert len(calls) == 1  # only the interrupted listing is made again

    # the reviewer's reproduction: a callback changes the dictionary being listed
    def tool():
        pass

    tool.__dict__.update({"key%d" % index: index for index in range(5000)})
    attributes, kept = tool.__dict__, []

    class Node:
        pass

    def spoil(_reference=None):
        attributes["added-by-a-callback"] = 1
        attributes.pop("added-by-a-callback", None)
        attributes["added-%d" % len(attributes)] = 1

    def garbage():
        one, other = Node(), Node()
        one.other, other.other = other, one  # a cycle: freed only by the collector
        kept.append(weakref.ref(one, spoil))

    thresholds = gc.get_threshold()
    gc.collect()
    gc.set_threshold(50, 1, 1)
    try:
        for _ in range(60):
            garbage()
            assert invocation._entry(tool.__dict__, "key4999") == 4999
            garbage()
            assert 4999 in invocation._listed(tool.__dict__.values)
    finally:
        gc.set_threshold(*thresholds)
    if sys.version_info < (3, 12):
        assert len(attributes) > 5000  # the callback ran


def test_what_is_kept_between_walks_is_kept_by_identity_and_weakly(monkeypatch):
    """What a cache calls, and the names a code object refers to, are kept in a plain
    ``dict`` under the ``id()`` of the object, beside a weak reference to it: an entry
    whose object is gone is not read as another's, nothing is kept alive, and entries of
    objects that are gone are dropped."""
    import gc
    import weakref

    reached_parameters(_six_fetch)  # the first walk has been made
    assert invocation._reference is weakref.ref
    assert type(invocation._KEPT) is dict
    monkeypatch.setattr(invocation, "_KEPT", {})

    class Owner:
        pass

    first, second = Owner(), Owner()
    invocation._keep(first, "one")
    assert invocation._kept(first) == "one" and invocation._kept(second) is invocation._MISSING
    # an entry left under the address of another object is not that object's
    invocation._KEPT[id(second)] = invocation._KEPT[id(first)]
    assert invocation._kept(second) is invocation._MISSING
    invocation._keep(second, "two")
    assert invocation._kept(second) == "two" and invocation._kept(first) == "one"
    gone = weakref.ref(first)
    del first
    gc.collect()
    assert gone() is None  # not kept alive
    # an object that cannot be weakly referenced is not kept
    invocation._keep(7, "seven")
    assert invocation._kept(7) is invocation._MISSING and id(7) not in invocation._KEPT

    # entries of objects that are gone are dropped once the dictionary has grown
    owners = [Owner() for _ in range(40)]
    for owner in owners:
        invocation._keep(owner, "kept")
    survivor = owners[0]
    del owners, owner
    gc.collect()
    assert len(invocation._KEPT) >= 41
    monkeypatch.setattr(invocation, "_SWEEP_AT", 1)
    third = Owner()
    invocation._keep(third, "three")
    assert sorted(invocation._KEPT) == sorted({id(second), id(survivor), id(third)})
    assert invocation._SWEEP_AT == 256
    assert invocation._kept(third) == "three" and invocation._kept(survivor) == "kept"

    # where the interpreter's weak reference type is not to be had, nothing is kept
    monkeypatch.setattr(invocation, "_reference", None)
    fourth = Owner()
    invocation._keep(fourth, "four")
    invocation._tell(fourth, _six_fetch)
    assert invocation._kept(fourth) is invocation._MISSING
    assert invocation._told(fourth) is invocation._MISSING
    monkeypatch.undo()

    assert invocation._reference_type() is weakref.ref

    class Lookalike:
        __slots__ = ("__callback__", "__weakref__")  # a member of that name, as the type has

        def __init__(self, *args):  # a stand-in: never called by the kernel
            pass

    monkeypatch.setattr(weakref, "ref", Lookalike)
    assert invocation._reference_type() is None  # not a type C code defines
    monkeypatch.undo()
    # another type C code defines is not it either: calling ``type`` or ``str`` in its
    # place would keep a strong reference and raise at the next walk
    others = (type, str, bool, int, object, weakref.ProxyType, weakref.CallableProxyType)
    for other in others + (weakref.KeyedRef, weakref.WeakMethod):
        monkeypatch.setattr(weakref, "ref", other)
        assert invocation._reference_type() is None, other
        monkeypatch.undo()
    monkeypatch.setattr(invocation, "_weakref", None)
    assert invocation._reference_type() is None


_SEVEN_GUARDED: Any = None


def test_the_globals_a_wrapper_names_are_looked_up_as_they_stand_and_named_once(monkeypatch):
    """The names a code object refers to cannot change: they are found once and kept.
    What stands under them in the module is looked up at each walk, a handful of reads
    whatever the size of the module, so a global bound again is read as it now stands. A
    key of the namespace that is not a ``str`` is neither hashed nor compared unless it
    hashes as one of those names."""
    import functools
    import types

    @functools.wraps(_six_fetch)
    def wrapper(*args, **kwargs):
        return _SEVEN_GUARDED(*args, **kwargs)

    real, named = invocation._names, []

    def counting(code, depth=0):
        named.append(code)
        return real(code, depth)

    monkeypatch.setattr(invocation, "_names", counting)
    monkeypatch.setitem(globals(), "_SEVEN_GUARDED", _traced(_six_fetch))
    for _ in range(3):
        assert unsupported_signature(wrapper) is None
    assert named == [wrapper.__code__]
    assert invocation._kept(wrapper.__code__) == ("_SEVEN_GUARDED",)
    monkeypatch.setitem(globals(), "_SEVEN_GUARDED", _six_audited(_six_fetch))
    assert unsupported_signature(wrapper).startswith(_SIX_UNDECLARED)
    assert named == [wrapper.__code__]
    assert invocation._named_globals(wrapper) == [globals()["_SEVEN_GUARDED"]]
    monkeypatch.delitem(globals(), "_SEVEN_GUARDED")
    assert invocation._named_globals(wrapper) == []  # a name that is not there is no value

    class Key:
        __hash__ = _touch("key __hash__", 20261123)
        __eq__ = _touch("key __eq__", False)

    namespace = dict(globals(), _SEVEN_GUARDED=_six_audited(_six_fetch))
    namespace[Key()] = None
    elsewhere = types.FunctionType(wrapper.__code__, namespace, "wrapper")
    functools.update_wrapper(elsewhere, _six_fetch)
    del _TOUCHED[:]
    assert unsupported_signature(elsewhere).startswith(_SIX_UNDECLARED)
    assert _TOUCHED == []

    # a key that hashes as a name the wrapper reads, and raises when it is compared: the
    # wrapper's own lookup of that global raises too. The walk refuses with a ValueError
    class Raising:
        __hash__ = _touch("raising __hash__", hash("_SEVEN_GUARDED"))
        __eq__ = _touch("raising __eq__", raises=RuntimeError)

    namespace = {name: value for name, value in globals().items() if name != "_SEVEN_GUARDED"}
    namespace[Raising()] = None
    unreadable = types.FunctionType(wrapper.__code__, namespace, "wrapper")
    functools.update_wrapper(unreadable, _six_fetch)
    with pytest.raises(ValueError, match="the module globals a wrapper names cannot be read"):
        invocation._named_globals(unreadable)
    with pytest.raises(ValueError, match="the module globals a wrapper names cannot be read"):
        reached_parameters(unreadable)
    assert unsupported_signature(unreadable) == "its signature cannot be inspected"


def test_a_callable_object_that_defines_get_is_refused_with_a_reason_of_its_own():
    """``inspect.signature`` shows no signature for a callable object whose class defines
    ``__get__`` (it takes it for a method descriptor of a built-in type). The walk raised
    ``ValueError`` for it and still does: with the cause and what to do in its message."""
    import functools
    import types

    class Binding:
        def __init__(self, function):
            functools.update_wrapper(self, function)
            self.function = function

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

        def __get__(self, instance, owner=None):
            return self if instance is None else types.MethodType(self, instance)

    for tool in (_six_retry(Binding(_six_fetch)), _traced(_six_retry(Binding(_six_fetch)))):
        assert tool("q", 5) == "rows"
        with pytest.raises(ValueError) as refused:
            reached_parameters(tool)
        text = str(refused.value)
        assert type(refused.value) is ValueError  # the class every gate already names
        assert unsupported_signature(tool) == text == invocation._DESCRIPTOR
        assert "defines `__get__`" in text and "give the class no `__get__`" in text
        assert "plain function" not in text  # 0.17.0 advised one that calls it

    class Service:
        fetch = Binding(lambda self, query, limit=10: "rows")

    bound = _six_retry(Service().fetch)  # the object, bound as a method
    assert bound("q") == "rows"
    with pytest.raises(ValueError, match="defines `__get__`"):
        reached_parameters(bound)
    if sys.version_info >= (3, 10):  # a ``staticmethod`` object is callable from 3.10
        static = types.MethodType(staticmethod(lambda self, query: "rows"), Service())
        with pytest.raises(ValueError, match="defines `__get__`"):
            invocation.own_parameters(static)
    # what ``inspect`` cannot read for another reason keeps the reason it had
    unread = functools.partial(functools.lru_cache(maxsize=None)(_six_fetch), "q")
    with pytest.raises(ValueError) as refused:
        invocation.own_parameters(unread)
    assert str(refused.value) != invocation._DESCRIPTOR
    assert not invocation._read_as_a_descriptor(str.upper)
    assert not invocation._read_as_a_descriptor(_six_fetch)


def test_a_partial_that_carries_a_wrapped_is_a_callable_the_walk_read():
    """A held wrapper is followed until it reaches a callable the walk read, for 32 links.
    A ``functools.partial`` that carries a ``__wrapped__`` is such a callable: a chain of
    31 wrappers that ends on it is judged, where the 32nd link would refuse it unread."""
    import functools

    relay = _six_retry(_six_fetch)
    held = relay
    for _ in range(31):
        held = _traced(held)

    @functools.wraps(relay)
    def outer(*args, **kwargs):
        return held(*args, **kwargs)

    assert outer("q", 5) == "rows"
    assert unsupported_signature(outer) is None
    held = _traced(held)  # one more: still going at the 32nd link
    with pytest.raises(HiddenWrapper):
        reached_parameters(outer)


def test_the_partialmethod_check_asks_nothing_of_functools_without_a_mark(monkeypatch):
    """A callable that shows nothing under either name of the mark is not asked about:
    ``functools.partialmethod`` is looked up only once a mark is there."""
    import functools

    asked: list = []

    class Recording:
        def __getattr__(self, name):
            asked.append(name)
            return getattr(functools, name)

    class Repo:
        marked = functools.partialmethod(_Fetcher(), "customers")

    marked = Repo().marked
    monkeypatch.setattr(invocation, "functools", Recording())
    for unmarked in (_six_fetch, _traced(_six_fetch), Repo(), len):
        invocation._refuse_partialmethod(unmarked)
    assert asked == []
    with pytest.raises(PartialMethodSignature):
        invocation._refuse_partialmethod(marked)
    assert asked == ["partialmethod"]


# --- review round eight: a kept bound argument moves the ones bound after it --------------

_EIGHT: list = []


def _eight_search(tenant: str, region: str = "eu", limit: int = 10) -> str:
    _EIGHT.append((tenant, region, limit))
    return "rows"


def _eight_tools(free):
    """A facade whose method delegates to ``free`` and keeps its own object."""
    import functools

    class Tools:
        @functools.wraps(free)
        def search(self, *args, **kwargs):
            return free(*args, **kwargs)

        @classmethod
        @functools.wraps(free)
        def search_cm(cls, *args, **kwargs):
            return free(*args, **kwargs)

        acme = functools.partialmethod(search, "acme")

        @functools.wraps(free)
        def __call__(self, *args, **kwargs):
            return free(*args, **kwargs)

    return Tools


def _eight_scoped(function):
    """Takes the first argument under the function's own name for it, and hands it on."""
    import functools

    @functools.wraps(function)
    def scoped(tenant, *args, **kwargs):
        return function(tenant, *args, **kwargs)

    return scoped


def _eight_shapes():
    import functools
    import types

    Tools = _eight_tools(_eight_search)
    tools = Tools()
    yield "a partial of the delegating method", functools.partial(tools.search, "acme"), "self"
    yield "a partialmethod of it", tools.acme, "self"
    yield "one partial of both", functools.partial(Tools.search, tools, "acme"), "self"
    both = functools.partial(functools.partial(Tools.search, tools), "acme")
    yield "two partials, merged", both, "self"
    yield "a classmethod", functools.partial(Tools.search_cm, "acme"), "cls"
    yield "a callable object", functools.partial(tools, "acme"), "self"
    yield "a method of a method", types.MethodType(tools.search, "acme"), "self"
    # the count of names is asked for: the method the walk bound has one, the partial of
    # it binds one more, and the name of the first is not taken for the second's
    traced = _eight_tools(_traced(_eight_search))()
    yield "over a traced function", functools.partial(traced.search, "acme"), "self"
    # the doubt stays with the same arguments further in: a wrapper under the facade
    # takes the next argument under the function's own name and hands it on
    scoped = _eight_tools(_eight_scoped(_eight_search))
    yield (
        "one partial, a wrapper between",
        functools.partial(scoped.search, scoped(), "acme"),
        "self",
    )
    yield (
        "a partial of the method, a wrapper between",
        functools.partial(scoped().search, "acme"),
        "self",
    )
    under = functools.partial(_traced(scoped.search), scoped(), "acme")
    yield "one partial of a traced method, a wrapper between", under, "self"

    def with_ctx(function):
        @functools.wraps(function)
        def wrapper(ctx, *args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    yield "a context a wrapper keeps", functools.partial(with_ctx(_eight_search), 1, "acme"), "ctx"


_EIGHT_SHAPES = [label for label, _, _ in _eight_shapes()]


@pytest.mark.parametrize("shape", _EIGHT_SHAPES)
def test_s_a_kept_bound_argument_moves_every_argument_bound_after_it(shape):
    """Round eight's S. ``def search(self, *args, **kwargs): return search(*args,
    **kwargs)`` keeps the object, so under ``functools.partial(tools.search, "acme")``
    the bound ``"acme"`` fills the first parameter of the free function and its SECOND
    runs at its default. The walk tested the first (it has no default, so nothing was
    refused): declared ``(limit=10)``, judged ``{'limit': 5}``, ran ``region='eu'``,
    which no signature showed. The parameter left without a value is the last one the
    bound arguments would fill."""
    tool, taken = next((tool, taken) for label, tool, taken in _eight_shapes() if label == shape)
    del _EIGHT[:]
    assert tool(limit=5) == "rows"
    assert _EIGHT == [("acme", "eu", 5)] and list(inspect_signature(tool)) == ["limit"]
    with pytest.raises(invocation.BoundArgumentKept) as refused:
        reached_parameters(tool)
    assert type(refused.value) is invocation.BoundArgumentKept
    text = str(refused.value)
    assert f"is taken by a wrapper as `{taken}`" in text
    assert "`region`, the last they would fill" in text and "acme" not in text


def test_s_what_a_kept_argument_cannot_move_is_read_as_before():
    """Beside S: where no wrapper takes a bound argument under another name there is no
    doubt. Where the last parameter the bound arguments fill has no default, a call BY
    KEYWORD that kept one fails, and 0.17.0 took that for the doubt settled; a call by
    position runs, so since 0.17.1 it settles the doubt only for a tool that declares
    no parameter an argument passed by position can fill (``test_x_``)."""
    import functools

    assert _levels(functools.partial(_eight_search, "acme")) == [["region", "limit"]]
    tools = _eight_tools(_eight_search)()
    with pytest.raises(TypeError, match="tenant"):
        tools.search(limit=5)  # the object was kept, and nothing fills ``tenant``
    # by position it runs: ``tenant`` gets the value judged for ``region``
    assert tools.search("us", 5) == "rows" and _EIGHT[-1] == ("us", 5, 10)
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`tenant`, the last"):
        reached_parameters(tools.search)

    # two required parameters, two arguments bound after the object: the third is moved
    def query(tenant: str, user: str, region: str = "eu", limit: int = 10) -> str:
        return region

    facade = _eight_tools(query)()
    assert functools.partial(facade.search, "acme", "bob")(limit=5) == "eu"
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`region`"):
        reached_parameters(functools.partial(facade.search, "acme", "bob"))
    # ``user`` has no default: by keyword the call fails, by position it runs moved
    assert list(inspect_signature(functools.partial(facade.search, "acme"))) == [
        "region",
        "limit",
    ]
    assert functools.partial(facade.search, "acme")("us", 5) == 5  # ``user='us'``, ``region=5``
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`user`, the last"):
        reached_parameters(functools.partial(facade.search, "acme"))

    # an ordinary method, a decorator that names the receiver as the method does, and
    # arguments bound after it
    class Repo:
        def fetch(self, query: str = "*", limit: int = 10) -> str:
            return query

        @_eight_named_self
        def find(self, query: str = "*", limit: int = 10) -> str:
            return query

        @_eight_named_self
        @_traced
        def under(self, query: str = "*", limit: int = 10) -> str:
            return query

    for tool in (functools.partial(Repo().fetch, "q"), functools.partial(Repo.fetch, Repo(), "q")):
        assert _levels(tool) == [["limit"]]
    for tool in (functools.partial(Repo().find, "q"), functools.partial(Repo.find, Repo(), "q")):
        assert tool(limit=5) == "q"
        assert _levels(tool) == [["args", "kwargs"], ["limit"]]
    # the name the receiver was taken under is the receiver's alone: with a layer between
    # that names nothing, it is not read as the name of the argument a partial binds
    # after it (the walk asks for as many names as the partial binds, and is given none)
    passing = [["args", "kwargs"]]
    assert functools.partial(Repo().under, "q")(limit=5) == "q"
    assert _levels(functools.partial(Repo().under, "q")) == passing * 2 + [["limit"]]

    # a facade that hands its receiver on to a wrapper with required parameters, under a
    # partial: had the receiver been kept, a call by keyword would go without the
    # wrapper's ``user`` and fail there. 0.17.0 took that for the doubt settled; a call
    # by position that kept it runs ``owner='bob'`` and ``user`` with the judged
    # ``limit``, so it is refused now, although this facade hands its receiver on
    def owned(owner: Any = None, user: str = "u", limit: int = 10) -> str:
        return user

    @functools.wraps(owned)
    def naming(owner, user, *args, **kwargs):
        return owned(owner, user, *args, **kwargs)

    class Hands:
        @functools.wraps(naming)
        def call(self, *args, **kwargs):
            return naming(self, *args, **kwargs)

    handed = functools.partial(Hands().call, "bob")
    assert handed(limit=5) == "bob" and list(inspect_signature(handed)) == ["limit"]
    assert handed(5) == "bob"  # it hands the receiver on: correct by position too
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`user`, the last"):
        reached_parameters(handed)

    # the keywords a partial binds are bound again to what its function hands the call
    # to: one that function does not take fails the call, and the walk refuses it
    unknown = functools.partial(_traced(_six_search), force=True)
    with pytest.raises(TypeError, match="force"):
        unknown()
    with pytest.raises(ValueError, match="incorrect arguments"):
        reached_parameters(unknown)


def _eight_named_self(method):
    """A method decorator that takes the receiver as ``self`` and hands it on."""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return method(self, *args, **kwargs)

    return wrapper


def _eight_named_instance(method):
    """The same, taking the receiver under another name than the method has for it."""
    import functools

    @functools.wraps(method)
    def wrapper(instance, *args, **kwargs):
        return method(instance, *args, **kwargs)

    return wrapper


def test_fr9_a_facade_that_passes_a_value_of_its_own_is_refused_and_says_so():
    """Stated, not fixed (the maintainer's decision, review round eight). A facade that
    passes a FIELD of its object in the first position runs correctly, and so does a
    wrapper that hands the bound argument on to a function with another name for it.
    Each has, parameter for parameter, the signature of the wrapper that keeps the
    argument, and the walk reads parameters and not bodies: it cannot tell them apart,
    so all are refused, in the safe direction. 0.15.0 accepts them. The message says
    that a wrapper which runs correctly is refused too, and what to write instead."""
    import functools

    def free_search(tenant: str = "public", limit: int = 10) -> str:
        return f"{tenant}:{limit}"

    class TenantTools:
        def __init__(self, tenant):
            self.tenant = tenant

        @functools.wraps(free_search)
        def search(self, *args, **kwargs):
            return free_search(self.tenant, *args, **kwargs)

    facade = TenantTools("acme")
    assert facade.search(limit=5) == "acme:5"  # nothing unjudged runs
    with pytest.raises(invocation.BoundArgumentKept) as refused:
        reached_parameters(facade.search)
    text = str(refused.value)
    assert "is taken by a wrapper as `self`" in text and "`tenant`, the last they would" in text
    assert "or passes a value of its own in its place" in text
    assert "is refused too, although it runs correctly" in text
    assert "Register the function the wrapper calls itself" in text
    assert "plain function" not in text  # 0.17.0 advised one, which is read for its name alone
    assert unsupported_signature(facade.search) == text
    assert "although it runs correctly" in invocation.BoundArgumentKept.__doc__

    # the remedy the message gives: the function itself is read to the end, and so is a
    # partial that binds its tenant by keyword, which runs with what the reading shows
    # (``test_x_`` follows every remedy of every message)
    assert unsupported_signature(free_search) is None
    search_acme = functools.partial(free_search, tenant="acme")
    assert search_acme(limit=5) == "acme:5"
    assert _levels(search_acme) == [["tenant", "limit"]]

    # the same cost elsewhere: a wrapper that hands the argument on under another name,
    # and a partial that binds a defaulted parameter of a method whose decorator calls
    # the receiver otherwise than the method does
    def lookup(session: Any = None, limit: int = 10) -> bool:
        return session is not None

    @functools.wraps(lookup)
    def with_client(client, *args, **kwargs):
        return lookup(client, *args, **kwargs)

    handed = functools.partial(with_client, object())
    assert handed(limit=5) is True
    with pytest.raises(invocation.BoundArgumentKept, match="`client`.*`session`"):
        reached_parameters(handed)

    class Repo:
        @_eight_named_instance
        def fetch(self, query: str = "*", limit: int = 10) -> str:
            return query

    # bare: read by 0.17.0, refused since 0.17.1, where a call by position that kept the
    # receiver would run ``self`` with the value judged for ``query``
    assert Repo().fetch("q", 5) == "q"
    with pytest.raises(invocation.BoundArgumentKept, match="`instance`.*`self`, the last"):
        reached_parameters(Repo().fetch)
    assert functools.partial(Repo().fetch, "q")(limit=5) == "q"
    with pytest.raises(invocation.BoundArgumentKept, match="`instance`.*`query`"):
        reached_parameters(functools.partial(Repo().fetch, "q"))

    # and more arguments under a doubt than the function has parameters for: one kept
    # would leave none of them without a value, more than one would
    def few(tenant: str = "public", *rest: str) -> str:
        return tenant

    spare = _eight_tools(few)
    assert functools.partial(spare.search, spare(), "acme", "x")() == "acme"
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`tenant`"):
        reached_parameters(functools.partial(spare.search, spare(), "acme", "x"))


def test_fr8_a_parameter_the_tool_declares_is_not_refused_for_a_kept_argument(monkeypatch):
    """Round eight's FR8. For a pass-through decorator over the BOUND delegating method,
    Python 3.9's ``inspect.signature`` goes on through the method to the free function
    and shows every parameter of it; 3.10 and later stop at the method and show them
    less the first. A parameter the tool declares is given by name and judged whatever
    the wrapper keeps, so it is refused only where the tool does not declare it."""
    import functools

    class Tools:
        @functools.wraps(_six_search)
        def search(self, *args, **kwargs):
            return _six_search(*args, **kwargs)

    tool = _traced(Tools().search)
    declared = list(inspect_signature(tool))
    levels = [["args", "kwargs"], ["args", "kwargs"], ["limit"]]
    if sys.version_info < (3, 10):
        assert declared == ["query", "limit"]
        assert _levels(tool) == levels and unsupported_signature(tool) is None
        _SIX.clear()
        tool(query="q", limit=5)
        assert _SIX == [("search", "q", 5)]  # judged by the names it declares, and ran so
    else:
        assert declared == ["limit"]
        with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`query`"):
            reached_parameters(tool)
    # the same on every Python, with what the tool declares handed in
    both = inspect_signature(_six_search)
    assert [list(level) for level in invocation._reached(tool, (), both)] == levels
    with pytest.raises(invocation.BoundArgumentKept):
        invocation._reached(tool, (), {"limit": both["limit"]})

    # a wrapper that copies the tool's parameters and passes nothing else on does not
    # keep them: under it the parameter is given by name still
    @functools.wraps(tool)
    def copying(query: str = "*", limit: int = 10) -> str:
        return tool(query=query, limit=limit)

    assert [list(level) for level in invocation._reached(copying, (), both)] == [
        ["query", "limit"],
        *levels,
    ]

    # not given by name: a wrapper on the way that passes the call on and takes the
    # parameter by name itself keeps the keyword. Two delegating methods under one: the
    # walk binds the second receiver to the function's own ``limit``, so the rule that
    # refuses a name a wrapper keeps does not see that the function takes it too
    delegating = Tools().search

    class Outer:
        @functools.wraps(delegating)
        def search(self, *args, **kwargs):
            return delegating(*args, **kwargs)

    stacked = Outer().search

    @functools.wraps(stacked)
    def keeping(*args, limit: int = 3, **kwargs):
        return stacked(*args, **kwargs)

    _SIX.clear()
    keeping(query="q", limit=5)
    assert _SIX == [("search", "q", 10)]  # judged ``limit=5``, ran the function's default
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`limit`, the last"):
        invocation._reached(keeping, (), both)
    with pytest.raises(invocation.BoundArgumentKept):
        reached_parameters(keeping)
    assert "default no policy judged" in unsupported_signature(keeping)
    if sys.version_info < (3, 10):  # where the tool itself declares both
        assert list(inspect_signature(keeping)) == ["query", "limit"]

    # the same where the wrapper that keeps the name is one a decorator HOLDS, which is
    # known only at the end of the walk
    def keeps_limit(function):
        @functools.wraps(function)
        def held_keeping(*args, limit: int = 3, **kwargs):
            return function(*args, **kwargs)

        return held_keeping

    holding = _calling(keeps_limit)(stacked)
    _SIX.clear()
    holding(query="q", limit=5)
    assert _SIX == [("search", "q", 10)]
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`limit`, the last"):
        invocation._reached(holding, (), both)
    with pytest.raises(ValueError):  # a BoundArgumentKept or, from 3.10, the wrapper held
        reached_parameters(holding)

    # a parameter without a default that a wrapper keeps: the call fails, nothing is run
    def needs(query: str, limit: int) -> str:
        return query

    class Needs:
        @functools.wraps(needs)
        def search(self, *args, **kwargs):
            return needs(*args, **kwargs)

    needing = Needs().search

    class OuterNeeds:
        @functools.wraps(needing)
        def search(self, *args, **kwargs):
            return needing(*args, **kwargs)

    twice = OuterNeeds().search

    @functools.wraps(twice)
    def keeps_required(*args, limit: int = 3, **kwargs):
        return twice(*args, **kwargs)

    with pytest.raises(TypeError, match="limit"):
        keeps_required(query="q", limit=5)
    assert len(invocation._reached(keeps_required, (), inspect_signature(needs))) == 4

    # what the tool declares is read once for a walk, when a check asks, and a tool whose
    # declaration cannot be read is refused where one is needed
    read: list = []
    real = invocation.inspect.signature

    def counting(func, **named):
        if not named:
            read.append(func)
        return real(func, **named)

    monkeypatch.setattr(invocation.inspect, "signature", counting)
    walk = invocation._walk(tool, None)
    assert invocation._declared(walk) is invocation._declared(walk) and read == [tool]
    reached_parameters(_six_search)
    assert read == [tool]  # nothing asked: the tool was not read for it
    monkeypatch.undo()
    assert invocation._declared(invocation._walk(object(), None)) is None
    monkeypatch.setattr(invocation, "_declared", lambda walk: None)
    with pytest.raises(invocation.BoundArgumentKept):
        reached_parameters(tool)
    with pytest.raises(HiddenWrapper):
        reached_parameters(_calling(_traced)(_six_fetch))


# --- review round nine: a bound argument handed on by keyword, and signatures declared ----
# --- where only the lookup ``inspect.signature`` makes finds them ------------------------

_NINE: List[Any] = []


def _nine_search(tenant: str, limit: int = 10) -> str:
    _NINE.append(("search", tenant, limit))
    return "rows"


def _nine_capped(function):
    """A wrapper with a leading parameter of its own, which it keeps."""
    import functools

    @functools.wraps(function)
    def wrapper(limit: int = 100, *args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


_NINE_CAPPED = _nine_capped(_nine_search)
_NINE_TRACED = _traced(_NINE_CAPPED)


def _nine_tools():
    import functools

    class Tools:
        TENANT = "acme"

        def __init__(self, tenant):
            self.tenant = tenant

        @functools.wraps(_NINE_CAPPED)
        def search(self, *args, **kwargs):
            return _NINE_CAPPED(*args, tenant=self.tenant, **kwargs)

        @classmethod
        @functools.wraps(_NINE_CAPPED)
        def search_of_class(cls, *args, **kwargs):
            return _NINE_CAPPED(*args, tenant=cls.TENANT, **kwargs)

        @functools.wraps(_NINE_TRACED)
        def search_traced(self, *args, **kwargs):
            return _NINE_TRACED(*args, tenant=self.tenant, **kwargs)

        @functools.wraps(_nine_search)
        def search_plain(self, *args, **kwargs):
            return _nine_search(self.tenant, *args, **kwargs)

        @functools.wraps(_nine_search)
        def search_plain_by_keyword(self, *args, **kwargs):
            return _nine_search(*args, tenant=self.tenant, **kwargs)

    return Tools


@pytest.mark.parametrize(
    "build",
    [
        lambda tools: tools("acme").search,
        lambda tools: tools.search_of_class,
        lambda tools: tools("acme").search_traced,
        lambda tools: _traced(tools("acme").search),
    ],
    ids=["method", "classmethod", "a pass-through between", "a pass-through over"],
)
def test_k_a_bound_argument_handed_on_by_keyword_to_a_wrapper_that_keeps_a_name(build):
    """Round nine's K. A facade hands its tenant on BY KEYWORD, so the receiver the walk
    binds to the wrapper's leading ``limit`` never fills it: the judged ``limit`` goes to
    the wrapper, which keeps it, and the function runs its own default. The walk read that
    level as taking nothing, so no rule saw the wrapper keep the name. Judged ``limit=5``,
    ran ``limit=10``; 0.15.0 accepted it as well. (For a decorator over the BOUND method,
    Python 3.9 shows the free function's own parameters, ``tenant`` with them.)"""
    tool = build(_nine_tools())
    declared = list(inspect_signature(tool))
    assert declared == ["limit"] or (sys.version_info < (3, 10) and declared == ["tenant", "limit"])
    _NINE.clear()
    tool(limit=5)
    assert _NINE == [("search", "acme", 10)]  # what a call runs: not the judged value
    with pytest.raises(invocation.BoundArgumentKept, match="`limit`"):
        reached_parameters(tool)
    assert "`limit`" in (unsupported_signature(tool) or "")


def test_k_the_leading_parameter_the_wrapper_keeps_need_not_have_a_default():
    """The same where the wrapper's leading parameter has no default: the judged value
    fills it, and the function further in runs its own."""
    import functools

    ran: list = []

    def search(tenant: str, region: str = "eu", limit: int = 10) -> str:
        ran.append((tenant, region, limit))
        return "rows"

    def routed(function):
        @functools.wraps(function)
        def wrapper(region: str, *args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    inner = routed(search)

    class Tools:
        @functools.wraps(inner)
        def search(self, *args, **kwargs):
            return inner(*args, tenant="acme", **kwargs)

    tool = Tools().search
    assert list(inspect_signature(tool)) == ["region", "limit"]
    tool(region="us", limit=5)
    assert ran == [("acme", "eu", 5)]
    with pytest.raises(invocation.BoundArgumentKept, match="`region`"):
        reached_parameters(tool)


def test_k_a_facade_over_a_function_that_passes_nothing_on_is_refused_since_0_17_1():
    """The facade hands its tenant on to the plain function, by position or by keyword,
    and runs correctly when called by keyword. 0.17.0 read both: the function's
    ``tenant`` has no default, so a facade that kept its receiver would fail a call by
    keyword. It would not fail a call by position: ``search(5)`` of a facade that kept
    it runs ``tenant=5``. The two have the signature of that facade, so both are refused
    now, although they run correctly; binding the tenant by keyword is read."""
    import functools

    tools = _nine_tools()
    for tool in (tools("acme").search_plain, tools("acme").search_plain_by_keyword):
        _NINE.clear()
        tool(limit=5)
        assert _NINE == [("search", "acme", 5)]
        with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`tenant`, the last"):
            reached_parameters(tool)
    bound = functools.partial(_nine_search, tenant="acme")
    assert _levels(bound) == [["tenant", "limit"]]
    _NINE.clear()
    bound(limit=5)
    assert _NINE == [("search", "acme", 5)]


def _nine_real(query: str, limit: int = 10, skip_review: bool = True) -> str:
    _NINE.append(("real", query, limit, skip_review))
    return "rows"


def _nine_shown(query: str, limit: int = 10) -> str:
    return "rows"


class _NineCarrier:
    """A class-based decorator that names nothing as ``__wrapped__``."""

    def __call__(self, *args, **kwargs):
        return _nine_real(*args, **kwargs)


def _nine_carriers():
    import inspect
    import types

    shown = inspect.signature(_nine_shown)

    class ByGetattr(_NineCarrier):
        def __getattr__(self, name):
            if name == "__signature__":
                return shown
            raise AttributeError(name)

    class ByGetattribute(_NineCarrier):
        def __getattribute__(self, name):
            if name == "__signature__":
                return shown
            return object.__getattribute__(self, name)

    class Proxy:
        """A forwarding proxy over a function that declares a signature."""

        def __init__(self, target):
            object.__setattr__(self, "_target", target)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_target"), name)

        def __call__(self, *args, **kwargs):
            return object.__getattribute__(self, "_target")(*args, **kwargs)

    def declaring(*args, **kwargs):
        return _nine_real(*args, **kwargs)

    declaring.__signature__ = shown  # type: ignore[attr-defined]

    class TextOnClass(_NineCarrier):
        __text_signature__ = "($self, query, limit=10)"

        def __get__(self, obj, cls=None):
            return self if obj is None else types.MethodType(self, obj)

    class TextOnInstance(_NineCarrier):
        def __init__(self):
            self.__text_signature__ = "(query, limit=10)"

        def __get__(self, obj, cls=None):
            return self if obj is None else types.MethodType(self, obj)

    return [
        ("__getattr__", ByGetattr()),
        ("__getattribute__", ByGetattribute()),
        ("a proxy over a function that declares one", Proxy(declaring)),
        ("__text_signature__ on the class", TextOnClass()),
        ("__text_signature__ on the instance", TextOnInstance()),
    ]


@pytest.mark.parametrize(
    "name, carrier", _nine_carriers(), ids=lambda v: v if type(v) is str else ""
)
def test_d_a_signature_only_the_lookup_finds_is_refused(name, carrier):
    """Round nine's D1 and D2. ``inspect.signature`` asks the callable for
    ``__signature__``, so a class that answers for the name in ``__getattr__`` or
    ``__getattribute__`` (a forwarding proxy) declares one, and it reads an object whose
    class defines ``__get__`` as a built-in, from ``__text_signature__`` (from Python
    3.13, any callable object). The check looked in the callable's ``__dict__`` and on
    its class, and saw neither. Shown ``(query, limit=10)``; ran ``skip_review=True``."""
    import functools
    import types

    assert "skip_review" not in inspect_signature(carrier)
    _NINE.clear()
    carrier(query="q", limit=5)
    assert _NINE == [("real", "q", 5, True)]
    bound = types.MethodType(carrier, object())
    for tool in (carrier, functools.partial(carrier), _traced(carrier), bound):
        with pytest.raises(DeclaredSignature):
            reached_parameters(tool)
        assert unsupported_signature(tool) is not None


def test_d_a_signature_the_metaclass_declares_is_not_the_instances():
    """Round nine's D3. A ``__signature__`` on the METACLASS is the class's: an instance
    shows none and ``inspect.signature`` reads its ``__call__``. Refused before with "it
    declares a signature that is not its own", which was untrue of it."""
    import inspect
    import types

    class Meta(type):
        @property
        def __signature__(cls):
            return inspect.signature(_nine_shown)

    class Carrier(metaclass=Meta):
        def __call__(self, query: str, limit: int = 10, skip_review: bool = True) -> str:
            return _nine_real(query, limit, skip_review)

    tool = Carrier()
    assert list(inspect_signature(tool)) == ["query", "limit", "skip_review"]
    assert _levels(tool) == [["query", "limit", "skip_review"]] * 2  # it, and its call
    assert _levels(types.MethodType(tool, "q")) == [["limit", "skip_review"]] * 2
    _NINE.clear()
    tool(query="q", limit=5, skip_review=False)
    assert _NINE == [("real", "q", 5, False)]  # every parameter judged, and ran so


def test_d_what_the_lookup_raises_and_a_text_that_is_the_callables_own():
    """A lookup that raises anything but ``AttributeError`` fails ``inspect.signature``
    too: the callable cannot be inspected, and is refused as that. A callable implemented
    in C is not refused for its ``__text_signature__``, which is its own. One declared on
    a callable object is refused on every Python, also where this one reads the
    ``__call__`` (before 3.13, for a class without ``__get__``)."""

    class Raises(_NineCarrier):
        def __getattr__(self, name):
            raise RuntimeError("no connection")

    with pytest.raises(RuntimeError):
        inspect_signature(Raises())
    with pytest.raises(ValueError, match="cannot be looked for"):
        reached_parameters(Raises())
    assert unsupported_signature(Raises()) == "its signature cannot be inspected"

    assert str.upper.__text_signature__
    invocation._refuse_declared_signature(str.upper, 0)
    invocation._refuse_declared_signature(len, 0)

    class Texted:
        __text_signature__ = "(query)"

        def __call__(self, query: str, limit: int = 10) -> str:
            return query

    read_from_text = sys.version_info >= (3, 13)
    shown = ["query"] if read_from_text else ["query", "limit"]
    assert list(inspect_signature(Texted())) == shown
    with pytest.raises(DeclaredSignature):
        reached_parameters(Texted())
    assert not invocation._declares_text(_NineCarrier())


def _nine_texted():
    """A plain function that carries ``__text_signature__``: ``inspect.signature`` reads
    its parameters from the text, on every Python."""

    def real(query: str, limit: int = 10, skip_review: bool = True) -> str:
        return _nine_real(query, limit, skip_review)

    real.__text_signature__ = "(query, limit=10)"  # type: ignore[attr-defined]
    return real


def _nine_texted_method():
    import types

    def real(self, query: str, limit: int = 10, skip_review: bool = True) -> str:
        return _nine_real(query, limit, skip_review)

    real.__text_signature__ = "($self, query, limit=10)"  # type: ignore[attr-defined]
    return types.MethodType(real, object())


def _nine_texted_call():
    class Carrier:
        def __call__(self, query: str, limit: int = 10, skip_review: bool = True) -> str:
            return _nine_real(query, limit, skip_review)

        __call__.__text_signature__ = "($self, query, limit=10)"  # type: ignore[attr-defined]

    return Carrier()


def _nine_partial(function):
    import functools

    return functools.partial(function)


def _nine_cached(function):
    import functools

    return functools.lru_cache(maxsize=None)(function)


@pytest.mark.parametrize(
    "build",
    [
        _nine_texted,
        lambda: _traced(_nine_texted()),
        lambda: _nine_partial(_nine_texted()),
        lambda: _nine_cached(_nine_texted()),
        lambda: staticmethod(_nine_texted()),
        _nine_texted_method,
        _nine_texted_call,
    ],
    ids=[
        "the function",
        "a wrapper of it",
        "a partial of it",
        "a cache over it",
        "a staticmethod of it",
        "a method of one",
        "the __call__ of an object",
    ],
)
def test_d_a_text_signature_declared_on_a_plain_function_is_refused(build):
    """Found while fixing round nine's D2. ``inspect.signature`` reads a plain function
    that carries ``__text_signature__`` from that text, on Python 3.9 to 3.15, and so
    reads everything it reads through such a function. The check did not look for the
    attribute at all. Shown ``(query, limit=10)``; ran ``skip_review=True``."""
    tool = build()
    if not (isinstance(tool, staticmethod) and sys.version_info < (3, 10)):
        assert list(inspect_signature(tool)) == ["query", "limit"]
        _NINE.clear()
        tool(query="q", limit=5)
        assert _NINE == [("real", "q", 5, True)]
    with pytest.raises(DeclaredSignature):
        reached_parameters(tool)
    if not isinstance(tool, staticmethod):
        assert unsupported_signature(tool) is not None


def _nine_cache_that_declares(shown):
    import functools

    def real(query: str, limit: int = 999999) -> str:
        return _nine_real(query, limit, False)

    cached = functools.lru_cache(maxsize=None)(real)
    cached.__signature__ = shown  # type: ignore[attr-defined]
    return cached


def test_d_a_signature_declared_on_a_wrapper_the_walk_looks_through_is_refused():
    """Found while fixing round nine's D1. ``inspect.signature`` stops at a
    ``__signature__`` a cache wrapper carries and reads it, in place of the function
    under the cache; the walk looked through the cache, a method over one and a
    ``staticmethod`` object without asking them. Every parameter of the function was
    still read, so none could be hidden, but a default could: shown ``limit=10``, ran
    ``limit=999999`` when the call left it out."""
    import functools
    import inspect

    shown = inspect.signature(_nine_shown)
    cached = _nine_cache_that_declares(shown)
    assert str(inspect.signature(cached)) == str(shown)
    _NINE.clear()
    cached(query="q")
    assert _NINE == [("real", "q", 999999, False)]
    for tool in (cached, functools.partial(cached), _traced(cached)):
        with pytest.raises(DeclaredSignature):
            reached_parameters(tool)
    assert unsupported_signature(cached) == _DECLARED

    def of_a_method(self, query: str, limit: int = 999999) -> str:
        return query

    over_a_method = functools.lru_cache(maxsize=None)(of_a_method)
    over_a_method.__signature__ = inspect.signature(  # type: ignore[attr-defined]
        lambda self, query, limit=10: None
    )

    class Holder:
        search = over_a_method

    assert list(inspect_signature(Holder().search)) == ["query", "limit"]
    assert inspect_signature(Holder().search)["limit"].default == 10
    with pytest.raises(DeclaredSignature):
        reached_parameters(Holder().search)

    if sys.version_info >= (3, 10):  # a ``staticmethod`` object takes attributes from 3.10
        static = staticmethod(lambda query, limit=999999: query)
        static.__signature__ = shown  # type: ignore[attr-defined]
        assert inspect_signature(static)["limit"].default == 10
        with pytest.raises(DeclaredSignature):
            reached_parameters(static)

    # an ordinary cache, over a function and over a method, is read as before
    def plain(query: str, limit: int = 10) -> str:
        return query

    assert _levels(functools.lru_cache(maxsize=None)(plain)) == [["query", "limit"]] * 2


# --- review round ten: the keyword hand-on to a HELD wrapper, a built-in bound as a ------
# --- method, and an object that looks like a function ------------------------------------


def _ten_scoped(function):
    """A method decorator that applies the keeping wrapper inside itself, hands its
    tenant on to it by keyword, and names the function as ``__wrapped__``."""
    import functools

    held = _nine_capped(function)

    @functools.wraps(function)
    def method(self, *args, **kwargs):
        return held(*args, tenant=self.tenant, **kwargs)

    return method


def _ten_scoped_global(function):
    import functools

    @functools.wraps(function)
    def method(self, *args, **kwargs):
        return _NINE_CAPPED(*args, tenant=self.tenant, **kwargs)

    return method


@pytest.mark.parametrize(
    "decorator", [_ten_scoped, _ten_scoped_global], ids=["in its closure", "a module global"]
)
def test_h_a_bound_argument_handed_on_by_keyword_to_a_held_wrapper(decorator):
    """Round ten's H: round nine's K, with the keeping wrapper HELD by the facade and the
    function named as ``__wrapped__``. The held wrapper's leading ``limit`` was left out
    as the place of the receiver, so no rule saw the wrapper keep the name. Judged
    ``limit=5``, ran ``limit=10``; 0.15.0 accepted it as well."""

    class Tools:
        def __init__(self, tenant):
            self.tenant = tenant

        search = decorator(_nine_search)

    tool = Tools("acme").search
    assert list(inspect_signature(tool)) == ["limit"]
    _NINE.clear()
    tool(limit=5)
    assert _NINE == [("search", "acme", 10)]  # what a call runs: not the judged value
    with pytest.raises(HiddenWrapper, match="`limit` is the leading parameter of a wrapper"):
        reached_parameters(tool)
    assert "`limit`" in (unsupported_signature(tool) or "")


def test_h_a_held_wrapper_that_takes_the_receiver_is_read_as_before():
    """Not refused: a held wrapper whose leading parameter is the receiver's own
    (``self``, which the tool does not declare), and one that passes the call on to
    nothing."""
    import functools

    ran: list = []

    def passing(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            return function(self, *args, **kwargs)

        return wrapper

    def twice(function):
        held = passing(function)

        @functools.wraps(function)
        def method(self, *args, **kwargs):
            return held(self, *args, **kwargs)

        return method

    class Tools:
        @twice
        def search(self, query: str, limit: int = 10) -> str:
            ran.append((query, limit))
            return query

    tool = Tools().search
    assert unsupported_signature(tool) is None
    tool(query="q", limit=5)
    assert ran == [("q", 5)]


def test_t_a_callable_written_in_c_bound_as_a_method_declares_nothing():
    """Round ten's T1, in the first form of the text-signature rule and never released: a
    method shows the ``__text_signature__`` of the built-in it binds, and the method type
    has no accessor of its own for the name, so ``types.MethodType(len, obj)`` and a
    ``classmethod`` of a built-in were refused as declaring one. The text is asked of
    what the method binds."""
    import math
    import types

    class Holder:
        root = classmethod(math.sqrt)

    for bound in (types.MethodType(len, []), Holder.root, types.MethodType(str.upper, "q")):
        assert bound.__text_signature__
        assert not invocation._declares_text(bound)
        invocation._refuse_declared_signature(bound, 0)
    assert not invocation._declares_text(len)

    # and a method of a function that does declare one is refused still
    assert invocation._declares_text(_nine_texted_method())


def test_m_an_object_that_looks_like_a_function_is_refused():
    """Round ten's M3. ``inspect.signature`` reads a callable object that shows a code
    object, a name and defaults as a function, from those, in place of its ``__call__``.
    The walk read the ``__call__`` as well, so no parameter could be hidden, but a
    default could: shown ``limit=10``, ran ``limit=999999`` when the call left it out.
    0.15.0 accepted it as well."""

    def shown(query: str, limit: int = 10) -> str:
        return query

    class Lookalike:
        def __init__(self):
            self.__name__ = "search"
            self.__code__ = shown.__code__
            self.__defaults__ = shown.__defaults__
            self.__kwdefaults__ = None
            self.__annotations__ = dict(shown.__annotations__)

        def __call__(self, query: str, limit: int = 999999) -> str:
            _NINE.append(("lookalike", query, limit))
            return query

    tool = Lookalike()
    assert inspect_signature(tool)["limit"].default == 10
    _NINE.clear()
    tool(query="q")
    assert _NINE == [("lookalike", "q", 999999)]
    with pytest.raises(DeclaredSignature):
        reached_parameters(tool)
    with pytest.raises(DeclaredSignature):
        reached_parameters(_traced(tool))
    assert unsupported_signature(tool) == _DECLARED
    # a transparent proxy, which also shows the code of what it wraps, names it as
    # ``__wrapped__``: read as before (the proxy tests above)


# --- review round eleven: a held wrapper's own defaulted leading parameter, and a --------
# --- lookalike whose ``__wrapped__`` is None ---------------------------------------------


def test_h_a_held_wrappers_own_defaulted_parameter_in_a_bound_place_is_refused():
    """Round eleven's H4. The held wrapper's leading parameter is one of its OWN, with a
    default, that the tool does not declare; the holder hands its tenant on by keyword,
    so nothing fills it. It was left out as the place of the receiver. Judged
    ``limit=5``; ran the wrapper with ``skip_review=True``. The same wrapper read by the
    walk is refused by the bound-argument rule; 0.15.0 accepted the held one as well."""
    import functools

    ran: list = []

    def reviewed(function):
        @functools.wraps(function)
        def wrapper(skip_review: bool = True, *args, **kwargs):
            ran.append(("reviewed", skip_review))
            return function(*args, **kwargs)

        return wrapper

    def scoped(function):
        held = reviewed(function)

        @functools.wraps(function)
        def method(self, *args, **kwargs):
            return held(*args, tenant=self.tenant, **kwargs)

        return method

    class Tools:
        tenant = "acme"
        search = scoped(_nine_search)

    tool = Tools().search
    assert list(inspect_signature(tool)) == ["limit"]
    _NINE.clear()
    tool(limit=5)
    assert ran == [("reviewed", True)] and _NINE == [("search", "acme", 5)]
    with pytest.raises(HiddenWrapper, match="`skip_review` is the leading parameter.*default"):
        reached_parameters(tool)
    assert "`skip_review`" in (unsupported_signature(tool) or "")


def test_m_a_lookalike_whose_wrapped_is_none_names_nothing():
    """Round eleven's E2. The lookalike rule left out an object that shows a
    ``__wrapped__``, by ``hasattr``, which is true of one set to None: nothing is
    followed from that, by ``inspect.signature`` or by the walk."""

    def shown(query: str, limit: int = 10) -> str:
        return query

    class Lookalike:
        __wrapped__ = None

        def __init__(self):
            self.__name__ = "search"
            self.__code__ = shown.__code__
            self.__defaults__ = shown.__defaults__
            self.__kwdefaults__ = None
            self.__annotations__ = dict(shown.__annotations__)

        def __call__(self, query: str, limit: int = 999999) -> str:
            return query

    with pytest.raises(DeclaredSignature):
        reached_parameters(Lookalike())


# --- review round twelve: a held partial fills the places it binds for certain ------------


def test_h_a_place_a_held_partial_binds_itself_is_filled_for_certain():
    """Round twelve's P1, in the round-eleven rule and never released. A decorator on a
    plain function holds ``functools.partial(retrying(function), 5)``, and the wrapper
    takes its configuration as a defaulted leading parameter. The partial hands ``5`` on
    by position, always, so that place is not one a holder could hand on by keyword;
    the rule counted every bound place as in doubt and refused it, naming a method that
    is not there. Read now, as 0.15.0 reads it. The place AFTER the partial's own is
    the holder's, and is refused as before."""
    import functools

    ran: list = []

    def retrying(function):
        @functools.wraps(function)
        def wrapper(attempts=3, *args, **kwargs):
            ran.append(("retrying", attempts))
            return function(*args, **kwargs)

        return wrapper

    def configured(function):
        held = functools.partial(retrying(function), 5)

        @functools.wraps(function)
        def outer(*args, **kwargs):
            return held(*args, **kwargs)

        return outer

    @configured
    def fetch(query: str, limit: int = 10) -> str:
        ran.append(("fetch", query, limit))
        return query

    assert unsupported_signature(fetch) is None
    fetch(query="q", limit=5)
    assert ran == [("retrying", 5), ("fetch", "q", 5)]

    # under a method, the same partial over the keeping wrapper: the partial's place is
    # certain, the receiver's place after it is not
    def capped_after(function):
        @functools.wraps(function)
        def wrapper(attempts=3, limit: int = 100, *args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    def scoped(function):
        held = functools.partial(capped_after(function), 5)

        @functools.wraps(function)
        def method(self, *args, **kwargs):
            return held(*args, tenant=self.tenant, **kwargs)

        return method

    class Tools:
        tenant = "acme"
        search = scoped(_nine_search)

    _NINE.clear()
    Tools().search(limit=5)
    assert _NINE == [("search", "acme", 10)]
    with pytest.raises(HiddenWrapper, match="`limit` is the leading parameter"):
        reached_parameters(Tools().search)


def _twelve_tools(inner):
    """A facade that takes its receiver in ``*args`` and hands a tenant on by keyword."""
    import functools

    class Tools:
        tenant = "acme"

        @functools.wraps(inner)
        def search(*args, **kwargs):
            self, rest = args[0], args[1:]
            return inner(*rest, tenant=self.tenant, **kwargs)

    return Tools()


def test_n_a_bound_argument_no_wrapper_names_handed_on_by_keyword():
    """Round twelve's N2: round nine's K with a facade that takes its receiver in
    ``*args`` (``self = args[0]``), so that no wrapper names the bound argument and
    there is no name doubt for the rule to act on. The wrapper under it passes the call
    on and keeps a leading parameter: the judged ``limit``, or its own defaulted
    ``skip_review``. A held wrapper of the same shape was refused already; a wrapper
    the walk reads gets the same rule now. 0.15.0 accepted both."""
    import functools

    tool = _twelve_tools(_NINE_CAPPED).search
    assert list(inspect_signature(tool)) == ["limit"]
    _NINE.clear()
    tool(limit=5)
    assert _NINE == [("search", "acme", 10)]  # what a call runs: not the judged value
    with pytest.raises(invocation.BoundArgumentKept, match="would fill `limit`, by position"):
        reached_parameters(tool)
    assert "`limit`" in (unsupported_signature(tool) or "")

    ran: list = []

    def reviewed(function):
        @functools.wraps(function)
        def wrapper(skip_review: bool = True, *args, **kwargs):
            ran.append(skip_review)
            return function(*args, **kwargs)

        return wrapper

    tool = _twelve_tools(reviewed(_nine_search)).search
    assert list(inspect_signature(tool)) == ["limit"]
    tool(limit=5)
    assert ran == [True]  # a default the tool does not show
    with pytest.raises(invocation.BoundArgumentKept, match="would fill `skip_review`"):
        reached_parameters(tool)


def test_n_what_passes_the_call_on_to_nothing_is_read_as_before():
    """Not refused by that rule: bound arguments that fill the parameters of a callable
    that passes the call on to nothing (a ``functools.partial`` over a decorated
    function, a decorated method), and a wrapper whose leading parameter has no default
    and is not declared (``self``): had it not been handed on by position, the call
    would fail."""
    import functools

    def table(table: str = "users", limit: int = 10) -> str:
        return table

    assert _levels(functools.partial(_traced(table), "orders")) == [["args", "kwargs"], ["limit"]]

    tool = _twelve_tools(_nine_search).search  # the same facade over the plain function
    assert _levels(tool) == [["args", "kwargs"], ["limit"]]
    _NINE.clear()
    tool(limit=5)
    assert _NINE == [("search", "acme", 5)]

    def with_self(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            return function(self, *args, **kwargs)

        return wrapper

    class Service:
        @_traced
        @with_self
        def search(self, query: str = "*", limit: int = 10) -> str:
            return query

    assert unsupported_signature(Service().search) is None


# --- review round thirteen: every bound place of a wrapper that passes the call on -------


@pytest.mark.parametrize("names", [False, True], ids=["taken in *args", "taken by name"])
def test_n_every_bound_place_of_a_wrapper_that_passes_the_call_on_is_tested(names):
    """Round thirteen's N15 and N16. One ``functools.partial`` binds two arguments, the
    facade hands both on by keyword, and the wrapper under it has the judged ``limit``
    first and a required name of its own second. The rule tested the LAST place the
    bound arguments would fill, which is required and not declared, so nothing was
    refused, with a name doubt or without one. A held wrapper was tested place by
    place already. Judged ``limit=5``; ran ``limit=10``. 0.15.0 accepted both."""
    import functools

    ran: list = []

    def search(tenant: str, context: str, limit: int = 10) -> str:
        ran.append((tenant, context, limit))
        return "rows"

    def capped(function):
        @functools.wraps(function)
        def wrapper(limit, context, *args, **kwargs):
            return function(*args, context=context, **kwargs)

        return wrapper

    inner = capped(search)
    if names:

        @functools.wraps(inner)
        def facade(tenant_of, context_of, *args, **kwargs):
            return inner(*args, tenant=tenant_of, context=context_of, **kwargs)

    else:

        @functools.wraps(inner)
        def facade(*args, **kwargs):
            return inner(*args[2:], tenant=args[0], context=args[1], **kwargs)

    tool = functools.partial(facade, "acme", "ctx")
    assert list(inspect_signature(tool)) == ["limit"]
    tool(limit=5)
    assert ran == [("acme", "ctx", 10)]  # what a call runs: not the judged value
    with pytest.raises(invocation.BoundArgumentKept, match="`limit`"):
        reached_parameters(tool)


# --- review round fourteen: a held wrapper that passes the call on to nothing ------------


def _fourteen_restating(function):
    """A wrapper that names every parameter again, with a default of its own for the
    first, and passes the call on to nothing."""
    import functools

    @functools.wraps(function)
    def wrapper(tenant: str = "public", limit: int = 10):
        return function(tenant, limit)

    return wrapper


@pytest.mark.parametrize("named", [True, False], ids=["as self", "in *args"])
def test_h_a_held_wrapper_that_passes_nothing_on_with_a_defaulted_bound_place(named):
    """Round fourteen's H8. A method decorator holds a wrapper that passes the call on to
    nothing and has a default of its own for its first parameter; the holder keeps its
    receiver. That parameter was left out as the receiver's place, and the held rule
    looked only at wrappers that pass the call on. Judged ``limit=5``; ran
    ``tenant='public'``, a default the tool does not show. The same wrapper read by the
    walk under a holder that names its receiver was refused already; 0.15.0 accepted the
    held one."""
    import functools

    def scoped(function):
        held = _fourteen_restating(function)
        if named:

            @functools.wraps(function)
            def method(self, *args, **kwargs):
                return held(*args, **kwargs)

        else:

            @functools.wraps(function)
            def method(*args, **kwargs):
                return held(*args[1:], **kwargs)

        return method

    class Tools:
        search = scoped(_nine_search)

    tool = Tools().search
    assert list(inspect_signature(tool)) == ["limit"]
    _NINE.clear()
    tool(limit=5)
    assert _NINE == [("search", "public", 5)]
    with pytest.raises(HiddenWrapper, match="`tenant` is the leading parameter.*has a default"):
        reached_parameters(tool)
    assert "`tenant`" in (unsupported_signature(tool) or "")


def test_h_what_the_held_rule_reads_and_what_it_refuses_though_it_runs():
    """Read: a held wrapper that passes nothing on whose leading parameter in the bound
    place is the receiver's own (``self``, no default), and one the tool declares, which
    takes the judged keyword and runs with it. Refused although it runs correctly: a
    ``functools.partial`` that binds, by position, the defaulted first parameter of a
    function whose decorator holds such a wrapper. And a name the framework injects is
    not excepted: the rule for a wrapper the walk reads has no such exception either."""
    import functools

    def holding_method(function):
        @functools.wraps(function)
        def restating(self, query: str = "*", limit: int = 10):
            return function(self, query, limit)

        @functools.wraps(function)
        def outer(self, *args, **kwargs):
            return restating(self, *args, **kwargs)

        return outer

    class Service:
        @holding_method
        def find(self, query: str = "*", limit: int = 10) -> str:
            return query

    assert unsupported_signature(Service().find) is None
    assert Service().find(query="q") == "q"

    def holding(function):
        held = _fourteen_restating(function)

        @functools.wraps(function)
        def outer(*args, **kwargs):
            return held(*args, **kwargs)

        return outer

    plain = holding(_nine_search)
    assert _levels(plain) == [["args", "kwargs"], ["tenant", "limit"]]  # nothing bound
    bound = functools.partial(plain, "acme")
    _NINE.clear()
    bound(limit=5)
    assert _NINE == [("search", "acme", 5)]  # clean
    with pytest.raises(HiddenWrapper, match="`tenant`"):
        reached_parameters(bound)

    def injected(function):
        @functools.wraps(function)
        def wrapper(ctx=None, *args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    def scoped(function):
        held = injected(function)

        @functools.wraps(function)
        def method(self, *args, **kwargs):
            return held(*args, tenant="acme", **kwargs)

        return method

    class Tools:
        search = scoped(_nine_search)

    declared = inspect_signature(Tools().search)
    for skip in ((), ("ctx",)):
        with pytest.raises(HiddenWrapper, match="`ctx`"):
            invocation._reached(Tools().search, skip, declared)


# --- the review of recusal-openai-agents 0.2.2 ---------------------------------------------
#
# Four items that review left open in the kernel, each reproduced on 0.16.0 before it was
# changed: what a cache named again calls had no public reading, and a ``weakref.proxy``
# on the way was read through as the object it stands for; a held wrapper that takes the
# tool's parameters in another order was read as passing them on; a held wrapper was not
# found where the way to the function passed anything but the closed set; and one reason
# said a parameter "runs at a default" where it has none.


def _again(called, named):
    """A ``functools.lru_cache`` wrapper over ``called``, named again for ``named``."""
    import functools

    return functools.update_wrapper(functools.lru_cache(maxsize=None)(called), named)


def test_o_what_a_cache_calls_is_a_public_step(monkeypatch):
    """``cache_target`` is the step from a cache to what runs under it, for a gate that
    makes a walk of its own: the function the cache calls, which is not the one it names
    where ``functools.wraps`` was written above the cache."""
    import functools
    import weakref

    cache_target = invocation.cache_target
    fetch, other = _fetch(), _fetch()
    assert cache_target(functools.lru_cache(maxsize=None)(fetch)) is fetch
    forcing = _forcing(fetch)
    named_again = _again(forcing, other)
    assert named_again.__wrapped__ is other  # what a walk of ``__wrapped__`` alone reads
    assert cache_target(named_again) is forcing  # what runs
    for options in _CACHES:
        cache = functools.update_wrapper(functools.lru_cache(**options)(forcing), other)
        cache("u")  # a result held, where the cache holds any
        assert cache_target(cache) is forcing, options
    partial = functools.partial(fetch)
    assert cache_target(_again(partial, other)) is partial  # whatever the cache calls
    unnamed = _again(forcing, other)
    del unnamed.__wrapped__
    assert cache_target(unnamed) is forcing

    proxy = weakref.proxy(named_again)
    assert isinstance(proxy, type(named_again))  # a proxy answers ``isinstance`` for it
    for not_a_cache in (fetch, partial, proxy, object(), None, type(named_again)):
        with pytest.raises(TypeError, match="not a `functools.lru_cache` wrapper"):
            cache_target(not_a_cache)

    fresh, bare = _again(_forcing(fetch), other), _again(_forcing(fetch), other)
    del bare.__wrapped__
    real = invocation._referents
    monkeypatch.setattr(invocation, "_referents", lambda value: real(value)[:-1])
    with pytest.raises(invocation.UnreadCache) as refused:  # laid out another way
        cache_target(fresh)
    assert str(refused.value) == invocation._REWRAPPED_CACHE
    monkeypatch.setattr(invocation, "_referents", None)
    assert cache_target(fresh) is other  # no listing to be had: the name stands
    with pytest.raises(invocation.UnreadCache) as refused:  # and where there is no name
        cache_target(bare)
    assert str(refused.value) == invocation._UNNAMED_CACHE
    monkeypatch.undo()
    assert cache_target(fresh) is not other
    assert "what this returns" in cache_target.__doc__


def test_o_the_two_types_of_a_weak_proxy_are_found_on_this_python(monkeypatch):
    """A ``weakref.proxy`` is told by its type, and the two types are found at the first
    walk as the interpreter's own. This holds every supported interpreter to it."""
    import types
    import weakref

    reached_parameters(_fetch())  # the first walk has been made
    assert invocation._PROXIES == (weakref.ProxyType, weakref.CallableProxyType)
    assert invocation._weak_references is weakref.getweakrefs
    assert invocation._proxy_types() == (invocation._PROXIES, weakref.getweakrefs)
    fetch = _fetch()
    assert invocation._a_weak_proxy(weakref.proxy(fetch)) and not invocation._a_weak_proxy(fetch)

    monkeypatch.setattr(weakref, "ProxyType", str)  # another built-in type under the name
    assert invocation._proxy_types() == ((), None)
    monkeypatch.undo()
    monkeypatch.setattr(weakref, "CallableProxyType", weakref.ProxyType)
    assert invocation._proxy_types() == ((), None)
    monkeypatch.undo()
    monkeypatch.setattr(weakref, "getweakrefs", lambda value: [])
    assert invocation._proxy_types() == (invocation._PROXIES, None)
    monkeypatch.undo()
    stand_in = types.SimpleNamespace(
        ProxyType=weakref.ProxyType, CallableProxyType=weakref.CallableProxyType
    )
    monkeypatch.setattr(invocation, "_weakref", stand_in)
    assert invocation._proxy_types() == ((), None)  # not a module


_O_PROXIED = ["the tool", "under a partial", "named by a wrapper", "called by a cache named again"]


@pytest.mark.parametrize("where", _O_PROXIED)
def test_o_a_weak_proxy_on_the_way_is_refused(where, monkeypatch):
    """A cache named again for ``fetch`` calls a wrapper that fills ``force`` from a
    default of its own. Read directly it is read as that wrapper. Behind a
    ``weakref.proxy`` 0.16.0 read it by its name: the proxy answers ``isinstance`` for the
    cache and is not of its type, so what the cache calls was not asked. Accepted, with
    ``force`` run at a default no policy judged."""
    import functools
    import weakref

    seen = []

    def fetch(url: str, timeout: int = 999999) -> int:
        return timeout

    @functools.wraps(fetch)
    def forcing(*args, force=True, **kwargs):
        seen.append(force)
        return fetch(*args, **kwargs)

    cache = _again(forcing, fetch)
    assert _levels(cache) == [["args", "force", "kwargs"], _FETCH, _FETCH]  # read, direct
    proxy = weakref.proxy(cache)
    tool = {
        "the tool": proxy,
        "under a partial": functools.partial(proxy),
        "named by a wrapper": _traced(proxy),
        "called by a cache named again": _again(proxy, fetch),
    }[where]
    assert tool("u") == 999999 and seen == [True]  # ``force`` ran, and no level shows it
    if where == "under a partial":
        # on some Pythons (3.11 is one) ``inspect.signature`` shows no signature for a
        # partial of a proxy of a cache, and the tool is refused for that before the
        # proxy is
        with pytest.raises((invocation.WeakProxy, ValueError)):
            reached_parameters(tool)
        assert unsupported_signature(tool) is not None
        return
    with pytest.raises(invocation.WeakProxy) as refused:
        reached_parameters(tool)
    assert str(refused.value) == invocation._WEAK_PROXY
    assert unsupported_signature(tool) == invocation._WEAK_PROXY

    monkeypatch.setattr(invocation, "_READY", False)  # the checks off: read as before
    assert all(level == _FETCH or level == _FORWARD for level in _levels(tool))


def test_o_a_weak_proxy_is_refused_whatever_it_stands_for():
    """A dispatcher with an implementation registered, behind a proxy, was read as its
    base: the dispatcher is known by its type and its code, and a proxy has neither. And
    a proxy of a plain function is refused although it runs correctly. A tool 0.16.0
    refused keeps its reason: the proxy is the last thing the walk refuses for."""
    import weakref

    lookup = _dispatching()
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(lookup)
    proxy = weakref.proxy(lookup)
    assert proxy(3) == "int, skip_review=True"  # a default no policy judged
    with pytest.raises(invocation.WeakProxy):
        reached_parameters(proxy)
    assert unsupported_signature(proxy) == invocation._WEAK_PROXY

    assert issubclass(invocation.WeakProxy, TypeError)
    assert not issubclass(invocation.WeakProxy, HiddenWrapper)
    fetch = _fetch()
    plain = weakref.proxy(fetch)
    assert plain("u", timeout=5) == 5  # it runs correctly
    with pytest.raises(invocation.WeakProxy):
        reached_parameters(plain)
    for text in (invocation._WEAK_PROXY, invocation._HELD_WEAK_PROXY):
        assert "itself, not a proxy of it" in text and "999999" not in text

    traced = _traced(lookup)
    with pytest.raises(DispatchedAtCallTime):  # behind the proxy, and found all the same
        reached_parameters(_traced(weakref.proxy(traced)))
    method = _repo()().q1  # kept alive: a proxy of a method does not keep it
    with pytest.raises(PartialMethodSignature):
        reached_parameters(_traced(weakref.proxy(method)))
    # a parameter of a wrapper's own behind the proxy is still the reason given
    forcing = _forcing(fetch)
    behind = _traced(weakref.proxy(forcing))
    assert "`force` is a parameter of a function the call reaches" in (
        unsupported_signature(behind) or ""
    )
    with pytest.raises(invocation.WeakProxy):  # the walk alone returns no levels for it
        reached_parameters(behind)


def test_o_a_held_wrapper_that_takes_the_parameters_in_another_order_is_refused():
    """A decorator holds a ``functools.wraps`` wrapper that takes the tool's two
    parameters the other way round. It names nothing the tool does not declare, so
    0.16.0 read it as passing them on: judged ``url="u"``, ``timeout=5``, and the
    function ran ``timeout="u"``. The levels the walk reads are returned to the caller,
    who can hold their order to its own; a held level is returned to nobody."""
    import functools

    fetch = _fetch()

    @functools.wraps(fetch)
    def flipped(timeout, url):
        return fetch(url=url, timeout=timeout)

    tool = _calling(lambda function: flipped)(fetch)
    assert list(inspect_signature(tool)) == ["url", "timeout"]
    assert tool("u", 5) == "u"  # the value judged for ``url`` ran as ``timeout``
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    expected = invocation._HELD_IN_ANOTHER_PLACE.format(name="timeout", declared="url")
    assert str(refused.value) == expected and "`timeout`" in expected
    assert unsupported_signature(tool) == expected
    # the same wrapper read by the walk: its order is in what the walk returns
    assert _levels(flipped) == [["timeout", "url"], _FETCH]

    class Flipping:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, timeout, url):
            return self.__wrapped__(url, timeout)

    with pytest.raises(HiddenWrapper) as refused:  # the ``__call__`` of a held object
        reached_parameters(_calling(Flipping)(fetch))
    assert str(refused.value) == expected

    def method_holding(held):
        def decorate(function):
            @functools.wraps(function)
            def method(self, *args, **kwargs):
                return held(self, *args, **kwargs)

            return method

        return decorate

    def find(self, tenant: str, limit: int = 10) -> str:
        return tenant

    @functools.wraps(find)
    def find_flipped(self, limit, tenant):
        return find(self, tenant, limit)

    @functools.wraps(find)
    def find_again(self, tenant, limit=10):
        return find(self, tenant, limit)

    class Tools:
        flipped = method_holding(find_flipped)(find)
        again = method_holding(find_again)(find)

    assert Tools().flipped("acme", 5) == 5  # the receiver is bound; the two are swapped
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(Tools().flipped)
    assert str(refused.value) == invocation._HELD_IN_ANOTHER_PLACE.format(
        name="limit", declared="tenant"
    )
    assert _levels(Tools().again) == [_FORWARD, ["tenant", "limit"]]


def test_o_what_the_order_rule_reads_and_what_it_refuses_though_it_runs():
    """Read: a held wrapper that takes the tool's parameters in the tool's order, the
    first of them alone, or by keyword only in any order. Refused although it runs
    correctly, because the walk reads parameters and not bodies: a held wrapper in
    another order that its decorator calls by keyword, and one that takes only a later
    parameter, in the first place."""
    import functools

    fetch = _fetch()

    def holding(held, call=None):
        @functools.wraps(fetch)
        def wrapper(*args, **kwargs):
            return (call or held)(*args, **kwargs)

        wrapper.held = held  # not what holds it: the closure does
        return wrapper

    @functools.wraps(fetch)
    def same(url, timeout=1):
        return fetch(url, timeout)

    @functools.wraps(fetch)
    def first(url):
        return fetch(url)

    @functools.wraps(fetch)
    def by_keyword(*, timeout=1, url):
        return fetch(url, timeout)

    for held in (same, first, by_keyword):
        assert _levels(holding(held))[0] == _FORWARD, held
        assert unsupported_signature(holding(held)) is None, held

    @functools.wraps(fetch)
    def flipped(timeout=1, url=""):
        return fetch(url, timeout)

    def calls_by_keyword(url, timeout=1):
        return flipped(url=url, timeout=timeout)

    tool = holding(flipped, calls_by_keyword)
    assert tool("u", 5) == 5  # it runs correctly
    with pytest.raises(HiddenWrapper, match="`timeout` is taken by a wrapper a decorator"):
        reached_parameters(tool)

    @functools.wraps(fetch)
    def later(timeout=1):
        return timeout

    def calls_the_later(url, timeout=1):
        return later(timeout)

    tool = holding(later, calls_the_later)
    assert tool("u", 5) == 5
    with pytest.raises(HiddenWrapper, match="in the place where the tool declares `url`"):
        reached_parameters(tool)


def _o_ways():
    """The callables 0.16.0 did not follow on the way from a held wrapper to the function,
    each as (name, what stands for the function there)."""
    import functools
    import weakref

    class Timed:
        """A class-based decorator: it names what it wraps."""

        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    class Calling:
        """A callable object: it holds the function and names nothing."""

        def __init__(self, function):
            self.function = function

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    def relay(function, *args, **kwargs):
        return function(*args, **kwargs)

    def carried(function):
        return functools.update_wrapper(functools.partial(relay, function), function)

    def another(url: str, timeout: int = 999999) -> int:
        return -1

    return [
        ("a class-based decorator", Timed),
        ("a callable object", Calling),
        ("a partial that carries a __wrapped__", carried),
        ("a weak proxy of the function", weakref.proxy),
        ("a cache named again for another function", lambda f: _again(_traced(f), another)),
    ]


@pytest.mark.parametrize("name, link", _o_ways(), ids=lambda v: v if type(v) is str else "")
def test_o_a_held_wrapper_is_found_whatever_its_way_to_the_function_passes(name, link):
    """A decorator holds a ``functools.wraps`` wrapper that keeps ``timeout``; what that
    wrapper wraps is not the function but something that stands for it. 0.16.0 followed
    a held wrapper through a closed set of types and gave up at anything else, so the
    wrapper was not found: judged ``timeout=5``, ran the function's own default."""
    fetch = _fetch()
    tool = _calling(lambda function: _with_timeout(link(function)))(fetch)
    assert tool("u", timeout=5) == 999999  # the argument is kept from the function
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    assert str(refused.value) == _KEPT
    assert unsupported_signature(tool) == _KEPT
    # the same way under a wrapper that only passes the call on is read, not refused
    passing = _calling(lambda function: _traced(link(function)))(fetch)
    assert passing("u", timeout=5) == 5
    assert _levels(passing) == [_FORWARD, _FETCH]
    assert unsupported_signature(passing) is None


def _o_behind():
    """The same callables, standing between the decorator and the wrapper it holds."""
    ways = dict(_o_ways())
    proxied = []

    def a_proxy(function):
        import weakref

        proxied.append(function)  # kept alive, as the application would keep it
        return weakref.proxy(function)

    ways["a weak proxy of the function"] = a_proxy
    return sorted(ways.items())


@pytest.mark.parametrize("name, link", _o_behind(), ids=lambda v: v if type(v) is str else "")
def test_o_a_wrapper_held_behind_another_callable_is_found(name, link):
    """The decorator's closure holds a class-based decorator, a callable object, a
    partial, a proxy or a cache named again for another function, and that holds the
    wrapper which keeps ``timeout``. Each was a stated limit of 0.16.0."""
    fetch = _fetch()
    tool = _calling(lambda function: link(_with_timeout(function)))(fetch)
    assert tool("u", timeout=5) == 999999
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    if name == "a weak proxy of the function":
        # not read behind: this proxy stands for a callable the walk did not read
        assert str(refused.value) == invocation._HELD_WEAK_PROXY
    else:
        assert str(refused.value) == _KEPT
    assert unsupported_signature(tool) == str(refused.value)


def test_o_a_wrapper_a_held_wrapper_holds_is_found():
    """A second held level: the decorator holds a wrapper that only passes the call on,
    and that one holds, in its own closure, the wrapper which keeps ``timeout``. What a
    wrapper found holds is searched like what a callable of the walk holds."""
    import functools

    fetch = _fetch()
    tool = _calling(_calling(_with_timeout))(fetch)
    assert tool("u", timeout=5) == 999999
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    assert str(refused.value) == _KEPT
    assert unsupported_signature(tool) == _KEPT
    assert _levels(_calling(_calling(_traced))(fetch)) == [_FORWARD, _FETCH]

    class Keeping:
        """A class-based decorator whose own ``__call__`` keeps the argument."""

        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, timeout=30, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    held = _calling(Keeping)(fetch)
    assert held("u", timeout=5) == 999999
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(held)
    assert str(refused.value) == _KEPT

    def helper(*args, **kwargs):
        return None

    def recursing(function):
        @functools.wraps(function)
        def inner(*args, **kwargs):
            return (inner, outer, helper) and function(*args, **kwargs)

        @functools.wraps(function)
        def outer(*args, **kwargs):
            return inner(*args, **kwargs)

        return outer

    assert _levels(recursing(fetch)) == [_FORWARD, _FETCH]  # wrappers that hold each other


def test_o_a_weak_proxy_a_wrapper_holds():
    """A proxy a wrapper holds is not read. One of a callable the walk read is known for
    it without being asked, from the weak references the interpreter lists for that
    callable, and is followed. One of any other callable is refused, although a helper
    held that way runs correctly; one of an object that is not callable passes no call
    on."""
    import weakref

    fetch = _fetch()
    read = _calling(weakref.proxy)(fetch)
    assert read("u", timeout=5) == 5
    assert _levels(read) == [_FORWARD, _FETCH]

    def helper(*args, **kwargs):
        return 5

    tool = _holding(weakref.proxy(helper), fetch)
    assert tool("u", timeout=5)[0] == 5
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    assert str(refused.value) == invocation._HELD_WEAK_PROXY
    assert unsupported_signature(tool) == invocation._HELD_WEAK_PROXY

    class Settings:
        pass

    settings = Settings()
    assert _levels(_holding(weakref.proxy(settings), fetch)) == [_FORWARD, _FETCH]


def test_o_a_held_proxy_is_refused_where_what_it_stands_for_cannot_be_listed(monkeypatch):
    import weakref

    fetch = _fetch()
    tool = _calling(weakref.proxy)(fetch)
    reached_parameters(tool)
    monkeypatch.setattr(invocation, "_weak_references", None)
    with pytest.raises(HiddenWrapper, match="holds a `weakref.proxy` of a callable"):
        reached_parameters(tool)

    def refusing(value):
        raise RuntimeError("an audit hook refused the listing")

    monkeypatch.setattr(invocation, "_weak_references", refusing)
    with pytest.raises(HiddenWrapper, match="holds a `weakref.proxy` of a callable"):
        reached_parameters(tool)


def test_o_the_search_of_what_a_wrapper_holds_ends():
    """Objects that hold each other are each stood on once. A way that is still going 32
    callables from what the wrapper holds is refused as it stands, whatever it reaches."""

    class Link:
        def __init__(self, inner=None):
            self.inner = inner

        def __call__(self, *args, **kwargs):
            return self.inner(*args, **kwargs)

    fetch = _fetch()
    first, second = Link(), Link()
    first.inner, second.inner = second, first
    assert _levels(_holding(first, fetch)) == [_FORWARD, _FETCH]

    def chain(length, end):
        for _ in range(length):
            end = Link(end)
        return end

    assert _levels(_holding(chain(31, None), fetch)) == [_FORWARD, _FETCH]
    assert _levels(_holding(chain(31, fetch), fetch)) == [_FORWARD, _FETCH]
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(_holding(chain(40, None), fetch))
    assert str(refused.value) == invocation._HIDDEN_WRAPPER


def _o_find(tenant, query: str, limit: int = 10) -> str:
    return f"{tenant} {query} {limit}"


def test_o_the_reason_for_a_bound_argument_says_only_what_holds_for_the_parameter():
    """The reason for a bound argument a wrapper may not hand on said the parameter left
    open "runs at a default no policy judged". Where the function that takes it passes
    the call on, the parameter can be one the tool declares that has no default: it is
    left to the judged keyword, which stops there. The reason now says of the parameter
    only what holds for it."""
    import functools

    def facade(inner, named=True):
        if named:

            @functools.wraps(inner)
            def search(self, *args, **kwargs):
                return inner(*args, **kwargs)

        else:

            @functools.wraps(inner)
            def search(*args, **kwargs):
                return inner(*args[1:], **kwargs)

        return type("Tools", (), {"search": search})().search

    def reason(tool):
        with pytest.raises(invocation.BoundArgumentKept) as refused:
            reached_parameters(tool)
        return str(refused.value)

    @functools.wraps(_o_find)
    def required(query, *args, **kwargs):  # declared by the tool, and no default anywhere
        return _o_find("acme", query, *args, **kwargs)

    tool = facade(required)
    assert list(inspect_signature(tool)) == ["query", "limit"]
    assert tool("q") == "acme q 10"  # it runs correctly, and is refused all the same
    text = reason(tool)
    assert "default" not in text  # 0.16.0: "`query`, the last they would fill, runs at a default"
    assert "`query` is left to a keyword of that name" in text
    remedy = invocation._UNPLACED_REMEDY.format(filled="query")
    assert text == invocation._UNPLACED_UNDER_A_NAME.format(
        taken="self", filled="query", left=invocation._LEFT[True, False], remedy=remedy
    )
    text = reason(facade(required, named=False))
    assert text == invocation._UNPLACED.format(
        filled="query", left=invocation._LEFT[True, False], remedy=remedy
    )
    assert "default" not in text

    @functools.wraps(_o_find)
    def defaulted(limit=100, *args, **kwargs):  # declared, with a default of its own
        return _o_find("acme", "q", *args, **kwargs)

    text = reason(facade(defaulted))
    assert text == invocation._UNPLACED_UNDER_A_NAME.format(
        taken="self",
        filled="limit",
        left=invocation._LEFT[True, True],
        remedy=invocation._UNPLACED_REMEDY.format(filled="limit"),
    )
    assert "or runs at a default no policy judged" in text

    @functools.wraps(_o_find)
    def undeclared(skip_review=True, *args, **kwargs):  # a default the tool does not show
        return _o_find("acme", *args, **kwargs)

    text = reason(facade(undeclared, named=False))
    assert text == invocation._UNPLACED.format(
        filled="skip_review",
        left=invocation._LEFT[False, True],
        remedy=invocation._UNPLACED_REMEDY.format(filled="skip_review"),
    )
    assert "`skip_review` runs at a default no policy judged" in text
    assert "keyword of that name" not in text
    assert set(invocation._LEFT) == {(True, True), (True, False), (False, True)}


# --- the review of recusal-pydantic-ai on 0.16.0 -------------------------------------------
#
# The same gaps by other routes, and three more: a dispatcher that gets its implementation
# between the walk and the call, a proxy written in Python that answers for
# ``__wrapped__``, and a reason that named ``__signature__`` for a ``__text_signature__``.


def test_p_a_dispatcher_with_nothing_registered_is_refused():
    """A wrapper registers an implementation on its first use and then calls the
    dispatcher. 0.16.0 read the registry at the walk, found the base alone and walked
    it: the first call was judged without ``purge`` and ran ``purge=True``. Refused now
    before anything is registered: on the ``__wrapped__`` chain, held by a wrapper named
    for the base, and behind a cache named again for the base."""
    import functools

    def build():
        def base(table: str, limit: int = 10) -> str:
            return "base"

        dispatcher = functools.singledispatch(base)
        done = []

        def register():
            if not done:
                done.append(1)
                dispatcher.register(str, lambda table, limit=10, purge=True: f"purge={purge}")

        return base, dispatcher, register

    base, dispatcher, register = build()

    @functools.wraps(dispatcher)
    def chained(*args, **kwargs):
        register()
        return dispatcher(*args, **kwargs)

    with pytest.raises(invocation.OpenDispatch) as open_to:  # the walk before the first call
        reached_parameters(chained)
    assert str(open_to.value) == invocation._OPEN_DISPATCH
    assert unsupported_signature(chained) == invocation._OPEN_DISPATCH
    assert issubclass(invocation.OpenDispatch, ValueError)
    assert chained("t") == "purge=True"  # what that call would have run

    base, dispatcher, register = build()

    @functools.wraps(base)
    def holding(*args, **kwargs):
        register()
        return dispatcher(*args, **kwargs)

    with pytest.raises(HiddenWrapper):
        reached_parameters(holding)
    assert holding("t") == "purge=True"

    base, dispatcher, register = build()

    @functools.wraps(dispatcher, updated=())
    def lazy(*args, **kwargs):
        register()
        return dispatcher(*args, **kwargs)

    cached = _again(lazy, base)
    with pytest.raises(invocation.OpenDispatch):
        reached_parameters(cached)
    assert cached("t") == "purge=True"
    with pytest.raises(DispatchedAtCallTime):  # and once one is registered, as before
        reached_parameters(cached)
    for carrier in (functools.partial(functools.singledispatch(base)), _traced(dispatcher)):
        with pytest.raises((invocation.OpenDispatch, DispatchedAtCallTime)):
            reached_parameters(_again(carrier, base))


def _p_forwarding():
    """A transparent proxy written in Python, as a lazy object or a tracing shim is: it
    keeps its target in a slot and forwards every attribute to it, ``__wrapped__`` too."""

    class Forwarding:
        __slots__ = ("_target",)

        def __init__(self, target):
            object.__setattr__(self, "_target", target)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_target"), name)

        def __call__(self, *args, **kwargs):
            return object.__getattribute__(self, "_target")(*args, **kwargs)

    return Forwarding


def test_p_a_proxy_that_answers_for_wrapped_without_holding_one_is_refused():
    """What such a proxy names as ``__wrapped__`` is what its TARGET wraps, so the walk
    stepped from the proxy past the target: past the ``__call__`` of a class-based
    decorator that keeps ``timeout``, past a cache named again for another function.
    0.16.0 read each as a pass-through. The same holds for ``weakref.proxy``, which is
    refused by its type; this one is told by where the name is answered from."""
    import functools

    forwarding = _p_forwarding()
    fetch = _fetch()

    class Keeping:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, timeout=30, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    direct = Keeping(fetch)
    assert unsupported_signature(direct) == _KEPT  # read directly, it is refused
    tool = forwarding(direct)
    assert tool("u", timeout=5) == 999999  # the argument is kept from the function
    with pytest.raises(invocation.ProxiedCallable) as refused:
        reached_parameters(tool)
    assert type(refused.value) is invocation.ProxiedCallable
    assert str(refused.value) == invocation._ANSWERS_FOR_WRAPPED
    assert unsupported_signature(tool) == invocation._ANSWERS_FOR_WRAPPED
    cache = _again(_with_timeout(fetch), fetch)
    for behind in (forwarding(cache), _traced(forwarding(cache)), forwarding(_traced(fetch))):
        with pytest.raises(invocation.ProxiedCallable):
            reached_parameters(behind)

    assert issubclass(invocation.WeakProxy, invocation.ProxiedCallable)
    assert issubclass(invocation.ProxiedCallable, TypeError)

    # a proxy that holds a ``__wrapped__`` of its own is read: an attribute, a slot, a
    # property. And one that names nothing is the callable object it is
    class Named(forwarding):
        __slots__ = ("__wrapped__",)

        def __init__(self, target):
            super().__init__(target)
            object.__setattr__(self, "__wrapped__", target)

    class Shown(forwarding):
        __slots__ = ()

        @property
        def __wrapped__(self):
            return object.__getattribute__(self, "_target")

    class Silent:
        def __init__(self, target):
            self._target = target

        def __call__(self, *args, **kwargs):
            return self._target(*args, **kwargs)

    for proxy in (Named, Shown):
        assert _levels(proxy(fetch))[-1] == _FETCH, proxy
        assert unsupported_signature(proxy(fetch)) is None, proxy
        assert unsupported_signature(proxy(direct)) == _KEPT, proxy
    assert _levels(Silent(fetch)) == [_FORWARD, _FORWARD]
    assert not invocation._answers_for_wrapped(_traced(fetch), fetch)
    assert not invocation._answers_for_wrapped(Named(fetch), fetch)
    assert invocation._answers_for_wrapped(forwarding(_traced(fetch)), fetch)

    # a placeholder of the name on the class does not make the answer the object's own
    class Placeholder(forwarding):
        __slots__ = ()
        __wrapped__ = None

        def __getattribute__(self, name):
            if name in ("_target", "__class__"):
                return object.__getattribute__(self, name)
            return getattr(object.__getattribute__(self, "_target"), name)

    placed = Placeholder(direct)
    assert placed("u", timeout=5) == 999999
    with pytest.raises(invocation.ProxiedCallable):
        reached_parameters(placed)
    # a proxy of a function that names nothing answers nothing for ``__wrapped__``: it is
    # not this refusal (it shows the function's code, and is refused for that)
    with pytest.raises(DeclaredSignature):
        reached_parameters(forwarding(fetch))


def test_p_a_declared_signature_says_which_attribute():
    """A function with a ``__text_signature__`` was refused with an exception that said
    nothing, and gates wrote "a ``__signature__``" for it. The exception names the
    attribute found, in ``declared`` and in its message."""
    import inspect

    def texted(table: str, limit: int = 10, purge: bool = True) -> str:
        return table

    texted.__text_signature__ = "(table, limit=10)"  # type: ignore[attr-defined]

    def signed(table: str, limit: int = 10, purge: bool = True) -> str:
        return table

    signed.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter("table", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)]
    )

    class Coded:
        __name__ = "coded"
        __defaults__ = None
        __kwdefaults__ = None
        __code__ = texted.__code__

        def __call__(self, table):
            return table

    for tool, attribute in (
        (texted, "__text_signature__"),
        (signed, "__signature__"),
        (Coded(), "__code__"),
        (_traced(texted), "__text_signature__"),
    ):
        with pytest.raises(DeclaredSignature) as refused:
            reached_parameters(tool)
        assert refused.value.declared == attribute, attribute
        assert str(refused.value) == invocation._DECLARED[attribute]
        assert f"`{attribute}`" in str(refused.value)
    assert "`__signature__`" not in invocation._DECLARED["__text_signature__"]
    # the reason ``unsupported_signature`` gives holds for each, and is as it was
    for tool in (_traced(texted), _traced(signed)):
        assert unsupported_signature(tool) == "it declares a signature that is not its own"
    bare = DeclaredSignature()  # as code written for 0.16.0 raises it
    assert bare.declared is None and str(bare) == "" and bare.args == ()
    import copy
    import pickle

    told = DeclaredSignature("__text_signature__")
    for again in (copy.copy(told), pickle.loads(pickle.dumps(told))):
        assert again.declared == "__text_signature__" and str(again) == str(told)


def test_p_the_callables_of_the_walk_stand_beside_its_levels():
    """``reached_callables`` is the walk's own list of what it read each level from, for
    a gate that has a question for each callable or rules of its own beside the levels:
    one callable for each level, in the same order, and the same refusals."""
    import functools

    reached_callables = invocation.reached_callables
    fetch = _fetch()

    class Decorator:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    class Service:
        @_traced
        def find(self, url: str, timeout: int = 999999) -> int:
            return timeout

    forcing = _forcing(fetch)
    named = functools.update_wrapper(functools.partial(fetch), fetch)
    tools = [
        fetch,
        _traced(fetch),
        _traced(_traced(fetch)),
        forcing,
        functools.lru_cache(maxsize=None)(fetch),
        _again(forcing, fetch),
        Decorator(_traced(fetch)),
        Service().find,
        functools.partial(_traced(fetch), "u"),
        _traced(named),
        _calling(_traced)(fetch),
    ]
    for tool in tools:
        callables = reached_callables(tool)
        assert callables[0] is tool
        own = [list(invocation.own_parameters(each)) for each in callables]
        assert own == _levels(tool), tool
    assert reached_callables(_traced(fetch))[1] is fetch
    assert reached_callables(_again(forcing, fetch))[1] is fetch  # the cache is the wrapper's level
    # the round-thirteen shape of ``recusal-pydantic-ai``: a partial that binds nothing,
    # named with ``update_wrapper``, under another wrapper. The kernel read it in 0.16.0
    # too; a gate that counted its own callables against these levels refused it
    assert _levels(_traced(named)) == [_FORWARD, _FETCH, _FETCH]
    assert unsupported_signature(_traced(named)) is None
    assert [type(each).__name__ for each in reached_callables(_traced(named))] == [
        "function",
        "partial",
        "function",
    ]
    for refused, kind in (
        (_calling(_with_timeout)(fetch), HiddenWrapper),
        (functools.singledispatch(fetch), invocation.OpenDispatch),
        (__import__("weakref").proxy(fetch), invocation.WeakProxy),
    ):
        with pytest.raises(kind):
            reached_callables(refused)
    assert "in the same" in reached_callables.__doc__


def test_p_a_class_based_decorator_is_two_callables_of_the_walk():
    """The walk reads the tool and at most 31 callables after it; a tool with a 32nd is
    refused. A function decorator is one of them. A class-based decorator built with
    ``functools.update_wrapper`` is two: what it names as ``__wrapped__`` and its own
    ``__call__``, both of which the walk reads. So a stack of them is read half as deep,
    and a deeper one is refused, not gated."""
    import functools

    class Logged:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    def deepest(layer):
        deep = 0
        for count in range(1, 40):
            tool = _fetch()
            for _ in range(count):
                tool = layer(tool)
            try:
                reached_parameters(_traced(tool))
            except ValueError as refused:
                assert str(refused) == "too many wrappers to inspect"
                return deep
            deep = count
        return deep

    assert deepest(_traced) == 30
    assert deepest(Logged) == 15


# --- the review of the change after 0.16.0 -------------------------------------------------
#
# A review of the change the two blocks above pin found the order rule wrong under a bound
# argument the holder keeps, one held wrapper judged for one of two ways to it, objects
# that show another class than their own, and exceptions that were no refusal.


def _v_search(self, sql: str, force: bool = False) -> str:
    return f"sql={sql!r} force={force!r}"


def _v_holding(second, hands_on):
    """A method decorator that holds ``second`` and keeps its receiver, or hands it on."""
    import functools

    def decorate(method):
        if hands_on:

            @functools.wraps(method)
            def first(self, *args, **kwargs):
                return second(self, *args, **kwargs)

        else:

            @functools.wraps(method)
            def first(self, *args, **kwargs):
                return second(*args, **kwargs)

        return first

    return decorate


def test_v_the_order_of_a_held_wrapper_is_judged_where_the_holder_keeps_its_receiver():
    """A method decorator holds ``def second(force, sql)`` and its own wrapper keeps the
    receiver: ``second(*args, **kwargs)``. The order rule took the first place of a held
    wrapper under a method for the receiver's whatever its name, so it compared ``sql``
    alone, read the tool, and a call by position ran the two swapped. And it refused
    the same decorator over ``def second(sql, force)``, the tool's own order, for a
    place that wrapper does not take. A place named as the tool names a parameter is
    that parameter's."""
    import functools
    import types

    @functools.wraps(_v_search)
    def swapped(force, sql):
        return _v_search(None, sql=sql, force=force)

    @functools.wraps(_v_search)
    def in_order(sql, force=False):
        return _v_search(None, sql=sql, force=force)

    @functools.wraps(_v_search)
    def swapped_with_self(self, force, sql):
        return _v_search(self, sql=sql, force=force)

    @functools.wraps(_v_search)
    def in_order_with_self(self, sql, force=False):
        return _v_search(self, sql=sql, force=force)

    held = {
        "kept, swapped": _v_holding(swapped, False)(_v_search),
        "kept, in order": _v_holding(in_order, False)(_v_search),
        "handed on, swapped": _v_holding(swapped_with_self, True)(_v_search),
        "handed on, in order": _v_holding(in_order_with_self, True)(_v_search),
    }
    Tools = type("Tools", (), {f"tool_{index}": each for index, each in enumerate(held.values())})
    tools = types.SimpleNamespace(
        swapped=Tools().tool_0,
        in_order=Tools().tool_1,
        swapped_with_self=Tools().tool_2,
        in_order_with_self=Tools().tool_3,
    )
    expected = invocation._HELD_IN_ANOTHER_PLACE.format(name="force", declared="sql")
    assert tools.swapped("DROP", False) == "sql=False force='DROP'"  # judged the other way
    for tool in (tools.swapped, tools.swapped_with_self):
        with pytest.raises(HiddenWrapper) as refused:
            reached_parameters(tool)
        assert str(refused.value) == expected
        assert unsupported_signature(tool) == expected
    assert tools.in_order("DROP", False) == "sql='DROP' force=False"
    for tool in (tools.in_order, tools.in_order_with_self):
        assert _levels(tool) == [_FORWARD, ["sql", "force"]], tool  # as 0.16.0 read both
        assert unsupported_signature(tool) is None

    # the same under a ``functools.partial`` that binds the first parameter
    def fetch(tenant: str, sql: str, force: bool = False) -> str:
        return f"sql={sql!r} force={force!r}"

    def bound(held):
        @functools.wraps(fetch)
        def wrapper(tenant, *args, **kwargs):
            return held(*args, **kwargs)

        return functools.partial(wrapper, "acme")

    @functools.wraps(fetch)
    def flipped(force, sql):
        return fetch("T", sql=sql, force=force)

    @functools.wraps(fetch)
    def straight(sql, force=False):
        return fetch("T", sql=sql, force=force)

    assert bound(flipped)("DROP", False) == "sql=False force='DROP'"
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(bound(flipped))
    assert str(refused.value) == expected
    assert _levels(bound(straight))[0] == _FORWARD
    assert invocation._bound_taken(swapped, 1, 0, {"sql", "force"}) == 0
    assert invocation._bound_taken(swapped_with_self, 1, 0, {"sql", "force"}) == 1
    assert invocation._bound_taken(swapped, 1, 1, {"sql", "force"}) == 1  # filled for certain


def test_v_a_held_wrapper_reached_two_ways_is_judged_for_each():
    """A held callable object keeps ``functools.partial(hw)``, where ``hw(skip_review=True,
    *args, **kwargs)`` has a leading parameter of its own: refused. With an attribute
    listed first that reaches the same ``hw`` as a method of the object, which nothing
    calls, the search stood on ``hw`` once, with the method's bound count, took
    ``skip_review`` for the receiver's place and read the tool."""
    import functools
    import types

    fetch = _fetch()
    seen = []

    @functools.wraps(fetch)
    def held(skip_review=True, *args, **kwargs):
        seen.append(skip_review)
        return fetch(*args, **kwargs)

    def tool(method_first):
        class Holder:
            def __init__(self):
                if method_first:
                    self.alias = types.MethodType(held, self)  # never called
                self.run = functools.partial(held)

            def __call__(self, *args, **kwargs):
                return self.run(*args, **kwargs)

        holder = Holder()

        @functools.wraps(fetch)
        def wrapper(*args, **kwargs):
            return holder(*args, **kwargs)

        return wrapper

    for method_first in (False, True):
        del seen[:]
        tool(method_first)(url="u", timeout=5)
        assert seen == [True]  # a default no policy judged
        with pytest.raises(HiddenWrapper, match="`skip_review`"):
            reached_parameters(tool(method_first))
        assert "`skip_review`" in (unsupported_signature(tool(method_first)) or "")


def _v_liars():
    """Callable objects whose ``__class__`` property shows a type they are not."""
    import functools
    import types

    def liar(shown_class, **faces):
        class Liar:
            def __init__(self, real, **attributes):
                self._real = real
                vars(self).update(attributes)

            @property
            def __class__(self):
                return shown_class

            def __call__(self, *args, **kwargs):
                return self._real(*args, **kwargs)

        return Liar

    def as_function(shown, real):
        face = {
            name: getattr(shown, name)
            for name in ("__code__", "__defaults__", "__kwdefaults__", "__annotations__")
        }
        return liar(types.FunctionType)(real, __name__=shown.__name__, **face)

    def as_method(shown, real):
        return liar(types.MethodType)(real, __func__=lambda self, *a, **k: None, __self__="S")

    def as_partial(shown, real):
        return liar(functools.partial)(real, func=shown, args=(), keywords={})

    return [("a function", as_function), ("a method", as_method), ("a partial", as_partial)]


@pytest.mark.parametrize("name, build", _v_liars(), ids=lambda v: v if type(v) is str else "")
def test_v_an_object_that_shows_another_class_than_its_own_is_refused(name, build):
    """The walk tells a function, a method and a ``functools.partial`` by ``isinstance``
    and ``inspect``, which ask ``__class__``, and a property can answer that. The object
    was read as the face it showed while a call ran its own ``__call__``, which passes
    on to a wrapper with a parameter of its own. In 0.16.0 too."""
    fetch = _fetch()
    tool = build(fetch, _forcing(fetch))
    assert tool("u") == 999999
    with pytest.raises(invocation.ProxiedCallable) as refused:
        reached_parameters(tool)
    assert str(refused.value) == invocation._SHOWS_ANOTHER_CLASS
    assert type(refused.value) is invocation.ProxiedCallable


def test_v_what_shows_another_class_and_holds_a_wrapped_of_its_own():
    """A transparent proxy of ``wrapt`` shows the class of what it wraps and holds a
    ``__wrapped__`` of its own, in a slot or in C: it is read as before, through that.
    Refused, of such objects, is one whose own ``__call__`` takes a parameter by name:
    behind the class it shows, that ``__call__`` is not read."""
    import types

    fetch = _fetch()

    class Transparent:
        __slots__ = ("__wrapped__",)

        def __init__(self, wrapped):
            self.__wrapped__ = wrapped

        @property
        def __class__(self):
            return type(self.__wrapped__)

        def __getattr__(self, name):
            return getattr(self.__wrapped__, name)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    class Keeping(Transparent):
        __slots__ = ()

        def __call__(self, *args, timeout=30, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    tool = Transparent(fetch)
    assert isinstance(tool, types.FunctionType) and type(tool) is Transparent
    assert _levels(tool)[-1] == _FETCH
    assert unsupported_signature(tool) is None
    assert invocation._shows_another_class(tool) is None
    assert invocation._shows_another_class(fetch) is None
    keeping = Keeping(fetch)
    assert keeping("u", timeout=5) == 999999
    with pytest.raises(invocation.ProxiedCallable) as refused:
        reached_parameters(keeping)
    assert str(refused.value) == invocation._SHOWS_ANOTHER_CLASS_AND_TAKES


def test_v_a_subclass_of_partial_with_a_call_of_its_own_is_read_as_a_callable_object():
    """``reached_parameters`` read a ``functools.partial`` for its function alone. A
    subclass with a ``__call__`` of its own runs that, with whatever parameters it has,
    and need not call the function at all. Its ``__call__`` is a level now."""
    import functools

    fetch = _fetch()

    class Guarded(functools.partial):
        def __call__(self, *args, timeout=30, **kwargs):
            return self.func(*args, **kwargs)

    tool = Guarded(fetch)
    assert tool("u", timeout=5) == 999999
    assert _levels(tool) == [_FETCH, ["args", "timeout", "kwargs"]]
    assert _levels(_traced(tool)) == [_FORWARD, _FETCH, ["args", "timeout", "kwargs"]]
    assert _levels(functools.partial(fetch)) == [_FETCH]  # a plain one, as before

    class Plain(functools.partial):
        pass

    assert _levels(Plain(fetch)) == [_FETCH]


def test_v_a_cache_that_names_nothing_is_read_as_what_it_calls():
    """``functools.lru_cache`` with its ``__wrapped__`` set to None, or deleted. The walk
    followed what the cache names and read nothing under such a cache, and the deleted
    one raised ``AttributeError``, which is no refusal."""
    import functools

    fetch = _fetch()

    def cache(delete):
        made = functools.lru_cache(maxsize=None)(_traced(_forcing(fetch)))
        if delete:
            del made.__wrapped__
        else:
            made.__wrapped__ = None
        return made

    for delete in (False, True):
        tool = cache(delete)
        assert tool("u") == 999999
        assert _levels(tool) == [_FORWARD, ["args", "force", "kwargs"], _FETCH], delete
        assert unsupported_signature(tool) is not None, delete
        assert _levels(_traced(tool))[2] == ["args", "force", "kwargs"]
        assert invocation.cache_target(tool).__wrapped__.__wrapped__ is fetch

    class Tools:
        find = cache(True)

    under_a_method = _levels(Tools().find)  # the object is bound to what the cache calls
    assert under_a_method[0] == _FORWARD and ["args", "force", "kwargs"] in under_a_method
    held = _calling(lambda function: cache(True))(fetch)
    with pytest.raises(HiddenWrapper):  # held, it is followed to what it calls as well
        reached_parameters(held)


def test_v_what_the_code_of_a_callable_raises_is_a_refusal():
    """The public functions raise ``DeclaredSignature``, ``TypeError`` or ``ValueError``,
    which a gate turns into a refusal. A ``weakref.proxy`` whose object is gone raised
    ``ReferenceError`` from all of them, and a ``__getattr__`` that raises for
    ``__wrapped__`` raised whatever it raised. An exception that is not an ``Exception``
    is still passed on."""
    import weakref

    def dead():
        return weakref.proxy(_forcing(_fetch()))

    for read in (reached_parameters, invocation.reached_callables, invocation.own_parameters):
        with pytest.raises(invocation.WeakProxy) as refused:
            read(dead())
        assert str(refused.value) == invocation._GONE
    assert unsupported_signature(dead()) is not None

    def raising(error):
        class Tool:
            def __call__(self, url: str, timeout: int = 5) -> int:
                return timeout

            def __getattr__(self, name):
                if name == "__wrapped__":
                    raise error("boom")
                raise AttributeError(name)

        return Tool()

    for error in (RuntimeError, KeyError, ReferenceError):
        with pytest.raises(ValueError) as refused:
            reached_parameters(raising(error))
        assert str(refused.value) == invocation._RAISED_WHEN_READ
        assert "boom" not in str(refused.value)
        assert type(refused.value.__cause__) is error
        assert unsupported_signature(raising(error)) == "its signature cannot be inspected"
    with pytest.raises(KeyboardInterrupt):
        reached_parameters(raising(KeyboardInterrupt))
    assert _levels(raising(AttributeError)) == [["url", "timeout"], ["url", "timeout"]]


def test_v_a_held_proxy_of_a_partial_or_a_method_the_walk_stood_on_is_followed():
    """A decorator over a ``functools.partial``, or over a bound method, that holds a
    ``weakref.proxy`` of what it wraps and calls that. The proxy was known only for a
    function or an object the walk read, and a partial or a method is read through its
    function, so these two were refused although they run correctly."""
    import functools
    import weakref

    fetch = _fetch()
    partial = functools.partial(fetch)

    class Tools:
        def find(self, url: str, timeout: int = 999999) -> int:
            return timeout

    method = Tools().find
    for wrapped in (fetch, partial, method):
        tool = _calling(weakref.proxy)(wrapped)
        assert tool("u", timeout=5) == 5
        assert _levels(tool)[0] == _FORWARD, wrapped
    helper = functools.partial(print)
    with pytest.raises(HiddenWrapper, match="holds a `weakref.proxy` of a callable"):
        reached_parameters(_holding(weakref.proxy(helper), fetch))


def test_v_only_callables_count_towards_the_bound_of_the_search():
    """The bound on one search counted every value a wrapper found holds, so a
    class-based decorator with some hundreds of plain attributes and one harmless held
    wrapper was refused, and so was a wrapper that holds 600 harmless callable objects.
    Plain values are not counted, and the bound is far above what a decorator holds."""
    import functools

    fetch = _fetch()

    class Decorated:
        def __init__(self, function):
            functools.update_wrapper(self, function)
            for index in range(600):
                setattr(self, f"setting_{index}", index)
            self.traced = _traced(function)

        def __call__(self, *args, **kwargs):
            return self.traced(*args, **kwargs)

    assert _levels(Decorated(fetch)) == [_FORWARD, _FORWARD, _FETCH]

    class Hook:
        def __call__(self, event):
            return None

    hooks = [Hook() for _ in range(600)]

    class Hooked:
        def __init__(self):
            vars(self).update({f"hook_{index}": hook for index, hook in enumerate(hooks)})

        def __call__(self, *args, **kwargs):
            return None

    assert _levels(_holding(Hooked(), fetch)) == [_FORWARD, _FETCH]
    assert invocation._HELD_NODES >= 16384


def test_v_a_declared_signature_takes_whatever_arguments_0_16_0_took():
    assert DeclaredSignature("a", "b").args == ("a", "b")
    assert DeclaredSignature("a", "b").declared is None
    assert DeclaredSignature("__signature__").declared == "__signature__"
    assert DeclaredSignature("__signature__", "x").declared is None
    assert str(DeclaredSignature("a message")) == "a message"


def test_v_the_lookup_of_wrapped_is_made_once_for_each_callable():
    """A ``__wrapped__`` property that answers differently each time: the proxy test
    made the lookup a second time, so the walk read another answer than 0.16.0 did."""
    fetch = _fetch()
    asked = []

    class Once:
        @property
        def __wrapped__(self):
            asked.append(1)
            return fetch if len(asked) == 1 else None

        def __call__(self, *args, **kwargs):
            return fetch(*args, **kwargs)

    assert _levels(Once()) == [_FORWARD, _FORWARD, _FETCH]
    assert len(asked) == 1


def test_v_an_object_with_many_attributes_copied_from_its_function_is_read_in_step():
    """What ``functools.update_wrapper`` copied from the function onto a class-based
    decorator is left out of what the decorator holds. Each attribute was compared with
    every copied one; they are told by ``id()`` now, and the reading is the same."""
    import functools

    fetch = _fetch()
    for index in range(3000):
        setattr(fetch, f"note_{index}", object())
    fetch.guarded = _with_timeout(fetch)  # a function's own attribute: copied, not read

    class Decorated:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    tool = Decorated(fetch)
    assert _levels(tool) == [_FORWARD, _FORWARD, _FETCH]
    tool.kept = _with_timeout(fetch)  # the decorator's own: read, and refused
    with pytest.raises(HiddenWrapper):
        reached_parameters(tool)


def test_v_the_decorator_that_hands_its_attributes_on_and_what_makes_it_read():
    """Refused although it runs correctly, and stated: a class-based decorator that keeps
    the function in an attribute and hands every other attribute on to it, over a
    function that names a ``__wrapped__``. With ``functools.update_wrapper(self, fn)``
    in its ``__init__`` it holds the name itself and is read to the end."""
    import functools

    fetch = _fetch()

    class Counted:
        def __init__(self, function, named):
            if named:
                functools.update_wrapper(self, function)
            self.function = function
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return self.function(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.function, name)

    for inner in (_traced(fetch), functools.lru_cache(maxsize=None)(fetch)):
        as_written = Counted(inner, named=False)
        assert as_written("u", timeout=5) == 5  # it runs correctly
        with pytest.raises(invocation.ProxiedCallable) as refused:
            reached_parameters(as_written)
        assert str(refused.value) == invocation._ANSWERS_FOR_WRAPPED
        named = Counted(inner, named=True)
        assert _levels(named)[-1] == _FETCH
        assert unsupported_signature(named) is None
    # and over a wrapper with a parameter of its own, named, it is refused for that
    assert unsupported_signature(Counted(_with_timeout(fetch), named=True)) == _KEPT


def test_v_a_dispatcher_built_over_a_dispatcher_is_refused_as_open():
    """A tool that was refused can be refused by another class. ``functools`` copies the
    inner dispatcher's ``register`` onto the outer one, so an implementation registered
    through the outer lands on the inner: the outer has nothing registered, and the walk
    meets it first. 0.16.0 walked on to the inner and raised ``DispatchedAtCallTime``."""
    import functools

    def base(key: str, deep: bool = False) -> str:
        return "base"

    inner = functools.singledispatch(base)
    outer = functools.singledispatch(inner)
    outer.register(int, lambda key, deep=False, skip_review=True: "int")
    assert outer(1) == "int" and int in inner.registry
    with pytest.raises(DispatchedAtCallTime):
        reached_parameters(inner)
    with pytest.raises(invocation.OpenDispatch):
        reached_parameters(outer)


def test_v_what_an_object_a_cache_named_again_calls_holds_is_searched():
    """A stated limit, pinned from both sides. A cache named again for the function
    calls a callable object that names nothing and keeps a wrapper with a parameter of
    its own. Where the object keeps it in its instance dictionary it is found and
    refused, as in 0.16.0. Where it keeps it in a slot it is not read, and ``timeout``
    runs at the wrapper's default: the limit the review's generated stacks still show."""
    fetch = _fetch()

    class Calling:
        def __init__(self, function):
            self.function = function

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    class Slotted:
        __slots__ = ("function",)

        def __init__(self, function):
            self.function = function

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    tool = _again(Calling(_with_timeout(fetch)), fetch)
    assert tool("u", timeout=5) == 999999
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    assert str(refused.value) == _KEPT
    assert unsupported_signature(tool) == _KEPT
    assert unsupported_signature(_again(Calling(_traced(fetch)), fetch)) is None
    assert unsupported_signature(_again(Slotted(_with_timeout(fetch)), fetch)) is None  # stated


def test_v_an_instance_dictionary_that_cannot_be_read_tells_nothing():
    """A class that answers ``__dict__`` with a property hides the instance dictionary
    from every reading that runs nothing. The proxy ``wrapt`` 2 builds in Python is such
    a class: it keeps ``__wrapped__`` in that dictionary, shows the class of what it
    wraps, and hands every other attribute on. So nothing is told of such an object, and
    it is read as 0.16.0 read it: the proxy of ``wrapt`` correctly, in both builds, and,
    as a stated limit, a proxy of that shape that keeps nothing of its own and answers
    ``__wrapped__`` for its target."""
    import functools

    fetch = _fetch()

    class Hidden:
        """As the Python build of ``wrapt`` 2 keeps what it wraps."""

        def __init__(self, wrapped):
            object.__setattr__(self, "__wrapped__", wrapped)

        @property
        def __dict__(self):
            return self.__wrapped__.__dict__

        @property
        def __class__(self):
            return type(self.__wrapped__)

        def __getattr__(self, name):
            return getattr(self.__wrapped__, name)

        def __call__(*args, **kwargs):
            self, *rest = args
            return self.__wrapped__(*rest, **kwargs)

    tool = Hidden(_traced(fetch))
    assert tool("u", timeout=5) == 5
    assert _levels(tool)[-1] == _FETCH
    assert unsupported_signature(tool) is None

    class Keeping:
        def __init__(self, function):
            functools.update_wrapper(self, function)

        def __call__(self, *args, timeout=30, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    class Answering:
        """Keeps its target where nothing reads it and answers ``__wrapped__`` for it."""

        def __init__(self, target):
            object.__setattr__(self, "_target", target)

        @property
        def __dict__(self):
            return {}

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_target"), name)

        def __call__(self, *args, **kwargs):
            return object.__getattribute__(self, "_target")(*args, **kwargs)

    limit = Answering(Keeping(fetch))
    assert limit("u", timeout=5) == 999999  # the argument is kept from the function
    reached_parameters(limit)  # and it is read: the stated limit


# --- stated behaviour, pinned after the second pass of that review -------------------------
#
# Nothing below is a fix: each test holds a sentence of the changelog to what happens.


def test_w_what_the_name_rule_reads_that_runs_otherwise():
    """Stated limits, in 0.16.0 too. Under a bound argument the places of a held wrapper
    are counted by their names and the holder is not read. A holder that hands its
    receiver on to a held wrapper whose leading place is named as the tool names a
    parameter is read: a call with every declared parameter fails, and one that leaves
    the defaulted one out runs with the receiver in the first parameter's place and the
    arguments shifted. And a held wrapper with a required parameter of its own in the
    leading place, where the holder keeps the receiver, was read: the place is taken
    for the receiver's. Since 0.17.1 that one is refused where the function the held
    wrapper names has another name for the place."""
    import functools

    ran = []

    def run(self, sql: str = "x", force: bool = False) -> str:
        ran.append((sql, force))
        return "ran"

    @functools.wraps(run)
    def in_order(sql, force=False):
        return run(None, sql, force)

    @functools.wraps(run)
    def with_its_own(mode, sql="x", force=False):
        return run(None, sql, force)

    class Tools:
        handed_on = _v_holding(in_order, True)(run)
        kept = _v_holding(with_its_own, False)(run)

    tools = Tools()
    assert _levels(tools.handed_on) == [_FORWARD, ["sql", "force"]]
    assert unsupported_signature(tools.handed_on) is None
    with pytest.raises(TypeError):
        tools.handed_on("S", True)  # three positional arguments for two places
    tools.handed_on("S")
    assert ran.pop() == (tools, "S")  # judged sql="S", force=False

    tools.kept("S", True)
    assert ran.pop() == (True, False)  # judged sql="S", force=True
    # read by 0.17.0, and refused since 0.17.1: the function the held wrapper names has
    # ``self`` in the place the wrapper takes as ``mode``, so had the wrapper kept the
    # receiver a call by position runs shifted (``_held_renamed``). Where the held
    # wrapper names that place as the function does, it is read still, as stated
    with pytest.raises(HiddenWrapper, match="`mode` is a leading parameter"):
        reached_parameters(tools.kept)
    assert "`mode`" in (unsupported_signature(tools.kept) or "")


def test_w_an_object_that_shows_another_class_and_restates_its_parameters():
    """Refused although it runs correctly, and stated: an object that shows another class,
    holds a ``__wrapped__`` of its own, and whose own ``__call__`` names the parameters
    and hands each on in order. A subclass of ``wrapt.CallableObjectProxy`` is one, in
    the builds of ``wrapt`` that keep ``__wrapped__`` where it can be read; this is that
    shape without ``wrapt``. And stated as read, in 0.16.0 too: the same object with a
    ``staticmethod`` as ``__call__``, which without the lie is refused for it."""
    fetch = _fetch()

    class Transparent:
        __slots__ = ("__wrapped__",)

        def __init__(self, wrapped):
            self.__wrapped__ = wrapped

        @property
        def __class__(self):
            return type(self.__wrapped__)

        def __getattr__(self, name):
            return getattr(self.__wrapped__, name)

    class Restating(Transparent):
        __slots__ = ()

        def __call__(self, url, timeout=5):
            return self.__wrapped__(url, timeout)

    tool = Restating(fetch)
    assert tool("u", 5) == 5 and tool("u", timeout=7) == 7  # it runs correctly
    with pytest.raises(invocation.ProxiedCallable) as refused:
        reached_parameters(tool)
    assert str(refused.value) == invocation._SHOWS_ANOTHER_CLASS_AND_TAKES

    seen = []

    def keeping(*args, timeout=30, **kwargs):
        seen.append(timeout)
        return 999999

    class Static(Transparent):
        __slots__ = ()
        __call__ = staticmethod(keeping)

    class Honest:
        __slots__ = ("__wrapped__",)
        __call__ = staticmethod(keeping)

        def __init__(self, wrapped):
            self.__wrapped__ = wrapped

    lying = Static(fetch)
    assert lying("u", timeout=5) == 999999 and seen == [5]  # its own parameter took it
    assert _levels(lying)[-1] == _FETCH  # read: the stated limit
    with pytest.raises(TypeError, match="is not a plain method"):
        reached_parameters(Honest(fetch))


def test_w_a_dead_proxy_is_a_weak_proxy_where_the_walk_stands_on_it():
    """As the tool and as what a wrapper names, a ``weakref.proxy`` whose object is gone
    is refused as a ``WeakProxy``. As the function of a ``functools.partial`` or of a
    method it is a plain ``ValueError``: a refusal either way, and stated as it is."""
    import functools
    import types
    import weakref

    def dead():
        return weakref.proxy(_forcing(_fetch()))

    named = _traced(_fetch())
    named.__wrapped__ = dead()
    for tool in (dead(), named):
        with pytest.raises(invocation.WeakProxy):
            reached_parameters(tool)
    for tool in (functools.partial(dead()), types.MethodType(dead(), object())):
        with pytest.raises(ValueError) as refused:
            reached_parameters(tool)
        assert type(refused.value) is ValueError
        assert unsupported_signature(tool) is not None


def test_w_whatever_is_raised_while_the_walk_reads_is_one_refusal(monkeypatch):
    """The consequence of the exception contract, stated: an ``Exception`` raised while
    the walk reads is a ``ValueError`` that names the callable's hooks, a fault of the
    kernel's own included, and the original is its cause."""

    def fault(walk, skip):
        raise KeyError("a fault of the kernel's own")

    monkeypatch.setattr(invocation, "_refuse_hidden_wrapper", fault)
    with pytest.raises(ValueError) as refused:
        reached_parameters(_fetch())
    assert str(refused.value) == invocation._RAISED_WHEN_READ
    assert "`__getattr__`" in str(refused.value)
    assert type(refused.value.__cause__) is KeyError


# --- 0.17.1: a doubt settled only for a call by keyword, and advice that reopened a gap ---
#
# Found by the review of ``recusal-openai-agents`` 0.2.3 and by that of
# ``recusal-pydantic-ai`` 0.2.1, both beside 0.17.0, and reproduced here without either
# framework. A framework can hand the judged values in by position, every declared
# parameter in its order, and the kernel cannot know whether it does: ``_x_calls`` makes
# both calls. Each ``test_x_`` test fails on 0.17.0.

_X: List[Any] = []


def _x_calls(tool, sent):
    """The judged values, ``sent`` completed with the defaults ``tool`` declares, as a
    gate completes them; and what a call runs with when a framework hands them in by
    position (every positional-or-keyword parameter in its order, the keyword-only ones
    by keyword), then when one hands them in by keyword. Each is what the functions
    under the tool recorded in ``_X``, or ``TypeError`` where the call raised it."""
    import inspect

    parameters = inspect.signature(tool).parameters
    judged = {}
    for name, parameter in parameters.items():
        if name in sent:
            judged[name] = sent[name]
        elif parameter.default is not parameter.empty:
            judged[name] = parameter.default
    positional = [judged[n] for n, p in parameters.items() if p.kind is p.POSITIONAL_OR_KEYWORD]
    keywords = {n: judged[n] for n, p in parameters.items() if p.kind is p.KEYWORD_ONLY}
    runs: List[Any] = []
    for args, named in ((positional, keywords), ((), judged)):
        del _X[:]
        try:
            tool(*args, **named)
        except TypeError:
            runs.append(TypeError)
            continue
        runs.append(list(_X))
    return judged, runs


def _x_read_and_run(tool, sent, ran, signature=True):
    """``tool`` is read to the end, and every call of it that runs, by position or by
    keyword, runs with ``ran``: what the reading shows. With ``signature``,
    ``unsupported_signature`` accepts it too; without, the tool is a
    ``functools.partial``, which ``unsupported_signature`` refuses whatever it binds
    (``typing.get_type_hints`` raises for a partial before Python 3.14 and finds no
    annotation on one from 3.14), and ``reached_parameters`` reads it."""
    if signature:
        assert unsupported_signature(tool) is None
    reached_parameters(tool)
    _, runs = _x_calls(tool, sent)
    assert [ran] in runs, runs
    for run in runs:
        assert run is TypeError or run == [ran], runs


def _x_execute(database: str, sql: str = "SELECT 1", note: str = "") -> str:
    _X.append({"database": database, "sql": sql, "note": note})
    return "recorded"


def _x_fetch(sql: str, force: bool = False) -> str:
    _X.append({"sql": sql, "force": force})
    return "ok"


def _x_facade(function, kind="other name"):
    """A class whose ``run`` exposes ``function`` as a method and drops its object, the
    object taken as ``kind`` says: under another name than ``function`` has for that
    place, in ``*args``, under the very name, or under another name below a cache."""
    import functools

    if kind == "in *args":

        @functools.wraps(function)
        def method(*args, **kwargs):
            return function(*args[1:], **kwargs)

    elif kind == "same name":
        first = next(iter(inspect_signature(function)))
        scope: Dict[str, Any] = {"functools": functools, "function": function}
        exec(
            f"@functools.wraps(function)\ndef method({first}, *args, **kwargs):\n"
            "    return function(*args, **kwargs)\n",
            scope,
        )
        method = scope["method"]
    else:

        @functools.wraps(function)
        def method(self, *args, **kwargs):
            return function(*args, **kwargs)

    if kind == "a cache":
        method = functools.lru_cache(maxsize=None)(method)
    return type("Warehouse", (), {"run": method})


@pytest.mark.parametrize("kind", ["other name", "a cache"])
def test_x_a_doubt_settled_only_for_a_call_by_keyword_is_refused(kind):
    """The review of ``recusal-openai-agents`` 0.2.3. The walk binds the facade's object
    to ``database`` and reads ``execute`` as taking ``(sql, note)``; the wrapper takes
    the object as ``self``, so whether it hands it on is in doubt. 0.17.0 took the doubt
    for settled because ``database`` has no default, so a call by keyword that kept the
    object fails. A call by position does not fail: judged ``sql='SELECT 1'``,
    ``note='DELETE FROM audit_log'``; ran ``database='SELECT 1'``,
    ``sql='DELETE FROM audit_log'``."""
    tool = _x_facade(_x_execute, kind)().run
    judged, (by_position, by_keyword) = _x_calls(
        tool, {"sql": "SELECT 1", "note": "DELETE FROM audit_log"}
    )
    assert judged == {"sql": "SELECT 1", "note": "DELETE FROM audit_log"}
    assert by_position == [{"database": "SELECT 1", "sql": "DELETE FROM audit_log", "note": ""}]
    assert by_keyword is TypeError
    with pytest.raises(invocation.BoundArgumentKept) as refused:
        reached_parameters(tool)
    text = str(refused.value)
    assert text == invocation._shifted("self", "database")
    assert "`database`, the last they would fill, has no default" in text
    assert "by position runs with `database` given the first of them" in text
    assert unsupported_signature(tool) == text and "DELETE" not in text and "SELECT" not in text

    # the two-parameter form: judged ``force=True``, ran ``fetch(sql=True, force=False)``
    tool = _x_facade(_x_fetch, kind)().run
    judged, (by_position, by_keyword) = _x_calls(tool, {"force": True})
    assert judged == {"force": True} and by_position == [{"sql": True, "force": False}]
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`sql`, the last"):
        reached_parameters(tool)


@pytest.mark.parametrize("kind", ["in *args", "same name"])
def test_x_a_facade_that_names_its_object_as_the_function_does_is_still_read(kind):
    """Stated, not closed. The same facade, taking its object in ``*args`` or under the
    name ``execute`` has for that place, runs shifted by position just the same; but no
    name is in doubt, and the shape is that of every method decorator that hands its
    receiver on (``def wrapper(*args, **kwargs)``, ``def wrapper(self, *args,
    **kwargs)`` over ``def method(self, ...)``). Nothing in a signature tells the two
    apart, so it is read, as 0.17.0 read it."""
    tool = _x_facade(_x_execute, kind)().run
    _, (by_position, by_keyword) = _x_calls(tool, {"sql": "SELECT 1", "note": "x"})
    assert by_position == [{"database": "SELECT 1", "sql": "x", "note": ""}]
    assert by_keyword is TypeError
    assert _levels(tool) == [_FORWARD, ["sql", "note"]]
    assert unsupported_signature(tool) is None


def test_x_the_doubt_is_settled_where_the_tool_declares_nothing_by_position():
    """Where the tool declares no parameter an argument passed by position can fill,
    every call that kept the object fails for want of the first parameter, by keyword
    and by position alike: the doubt is settled there, as in 0.17.0, and the tool read."""

    def execute(database: str, *, sql: str = "SELECT 1", note: str = "") -> str:
        _X.append({"database": database, "sql": sql, "note": note})
        return "recorded"

    tool = _x_facade(execute)().run
    assert [p.kind for p in inspect_signature(tool).values()] == [
        inspect_signature(execute)["sql"].KEYWORD_ONLY
    ] * 2
    assert _x_calls(tool, {"sql": "x"})[1] == [TypeError, TypeError]
    assert _levels(tool) == [_FORWARD, ["sql", "note"]]


def test_x_arguments_bound_after_the_object_are_judged_by_the_partial_that_binds_them():
    """Under a ``functools.partial`` of the facade, the argument the partial binds is
    what would fill the first parameter, and the partial is judged by the same rule
    with the doubt it is handed: refused where its last place has no default and the
    tool declares a parameter by position, and where it has one, as before. Where an
    object only passes for a partial, what it binds cannot be counted, and the doubt is
    not settled under it."""
    import functools

    def query(tenant: str, user: str, limit: int = 10) -> str:
        _X.append({"tenant": tenant, "user": user, "limit": limit})
        return "rows"

    facade = _x_facade(query)()
    tool = functools.partial(facade.run, "acme")
    assert _x_calls(tool, {"limit": 5})[1] == [
        [{"tenant": "acme", "user": 5, "limit": 10}],
        TypeError,
    ]
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`user`, the last"):
        reached_parameters(tool)


def test_x_what_only_passes_for_a_partial_does_not_settle_the_doubt(monkeypatch):
    """The arguments an object binds are counted only for the standard library's
    ``functools.partial``. Under another class put in ``functools``, the arguments bound
    outside the facade's method cannot be counted, so the doubt the method raises is not
    settled there: refused. 0.17.0 settled it."""
    import functools

    assert invocation._prepare()

    class Lookalike:
        def __init__(self, func, *args, **keywords):
            self.func, self.args, self.keywords = func, args, keywords

        def __call__(self, *args, **keywords):
            return self.func(*self.args, *args, **dict(self.keywords, **keywords))

    facade = _x_facade(_x_execute)()
    monkeypatch.setattr(functools, "partial", Lookalike)
    with pytest.raises(invocation.BoundArgumentKept, match="`self`.*`database`"):
        reached_parameters(Lookalike(facade.run))


def _x_receiving(function):
    """A method decorator whose wrapper takes the receiver as ``receiver`` and keeps it."""
    import functools

    @functools.wraps(function)
    def forward(receiver, *args, **kwargs):
        return function(*args, **kwargs)

    return forward


def _x_handing(function):
    """The same, taking the receiver as ``self``, the name the method has for it, and
    handing it on by position: what the refusal says to write."""
    import functools

    @functools.wraps(function)
    def forward(self, *args, **kwargs):
        return function(self, *args, **kwargs)

    return forward


def test_x_a_held_wrapper_that_names_a_receiver_otherwise_is_refused():
    """The sibling of the doubt, in a wrapper a decorator holds: the held wrapper takes
    the receiver as ``receiver``, and the method it names as ``__wrapped__`` has
    ``self`` in that place. 0.17.0 read it: no tool declares ``receiver``, and it has no
    default, so a call by keyword that the held wrapper kept the receiver from fails.
    By position it runs the method with ``self='q'`` and ``query=5``. Refused now,
    after every refusal 0.17.0 makes; and read once the held wrapper takes the receiver
    as ``self`` and hands it on."""

    class Service:
        @_calling(_x_receiving)
        def search(self, query: str, limit: int = 5) -> str:
            _X.append({"self": self if isinstance(self, str) else "S", "query": query})
            return "rows"

        @_calling(_x_handing)
        def handed(self, query: str, limit: int = 5) -> str:
            _X.append({"self": self if isinstance(self, str) else "S", "query": query})
            return "rows"

    tool = Service().search
    _, (by_position, by_keyword) = _x_calls(tool, {"query": "q", "limit": 5})
    assert by_position == [{"self": "q", "query": 5}] and by_keyword is TypeError
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    assert str(refused.value) == invocation._HELD_RENAMED.format(
        taken="receiver",
        filled="self",
        left=invocation._HELD_LEFT[False].format(filled="self"),
    )
    assert unsupported_signature(tool) == str(refused.value)
    _x_read_and_run(Service().handed, {"query": "q"}, {"self": "S", "query": "q"})

    # where the place has a default the held wrapper runs it even by keyword: the walk
    # reads the holder, which takes the receiver in ``*args``, as handing it on
    def free(tenant: str = "public", limit: int = 10) -> str:
        _X.append({"tenant": tenant if isinstance(tenant, str) else "T", "limit": limit})
        return "rows"

    Tools = type("Tools", (), {"run": _calling(_x_receiving)(free)})
    _, runs = _x_calls(Tools().run, {"limit": 5})
    assert runs[1] == [{"tenant": "public", "limit": 5}]  # judged ``limit`` alone
    with pytest.raises(HiddenWrapper, match="`receiver`.*`tenant`, the last they would fill"):
        reached_parameters(Tools().run)


def _x_layers():
    """What a held wrapper can name as ``__wrapped__`` between it and the method: each
    takes the arguments in ``*args`` and passes them on, or binds one before them."""
    import functools

    def class_based(function):
        class Logged:
            def __init__(self, function):
                functools.update_wrapper(self, function)
                self.function = function

            def __call__(self, *args, **kwargs):
                return self.function(*args, **kwargs)

        return Logged(function)

    def helper(function, *args, **kwargs):
        return function(*args, **kwargs)

    return [
        ("nothing", lambda f: f),
        ("a cache", functools.lru_cache(maxsize=None)),
        ("a cache named again", _seven_memo),
        ("a class-based decorator", class_based),
        ("a partial named for it", lambda f: functools.wraps(f)(functools.partial(helper, f))),
        ("a wrapper that names the receiver as the method does", _x_handing),
    ]


@pytest.mark.parametrize("name, layer", _x_layers(), ids=lambda v: v if type(v) is str else "")
def test_x_a_renamed_receiver_is_found_through_what_passes_it_on(name, layer):
    """The held wrapper's ``__wrapped__`` need not be the method: the name it is compared
    with is the one the first callable with named places has, found through what takes
    the arguments in ``*args`` and passes them on, as the walk reads it."""

    def decorate(method):
        return _calling(_x_receiving)(layer(method))

    class Service:
        @decorate
        def search(self, query: str, limit: int = 5) -> str:
            _X.append({"self": self if isinstance(self, str) else "S", "query": query})
            return "rows"

        def __hash__(self):
            return 1

    tool = Service().search
    _, (by_position, _) = _x_calls(tool, {"query": "q", "limit": 5})
    assert by_position == [{"self": "q", "query": 5}]
    with pytest.raises(HiddenWrapper, match="`receiver` is a leading parameter"):
        reached_parameters(tool)


def test_x_a_renamed_receiver_is_found_past_a_value_a_partial_binds():
    """A partial between the held wrapper and the method binds the method's receiver
    place itself, so the held wrapper's bound place is compared with the place after
    it: ``query`` where it takes ``receiver``. Kept, ``limit`` runs as ``query``."""
    import functools

    def query_of(tenant: str, query: str, limit: int = 5) -> str:
        _X.append({"tenant": tenant, "query": query, "limit": limit})
        return "rows"

    def decorate(function):
        return _calling(_x_receiving)(_traced(functools.partial(function, "acme")))

    Tools = type("Tools", (), {"run": decorate(query_of)})
    tool = Tools().run
    assert list(inspect_signature(tool)) == ["limit"]
    _, (by_position, _) = _x_calls(tool, {"limit": 7})
    assert by_position == [{"tenant": "acme", "query": 7, "limit": 5}]
    with pytest.raises(HiddenWrapper, match="`receiver`.*`query`, the last"):
        reached_parameters(tool)


# What every refusal with advice says to write, built and checked: read soundly, which is
# to say read to the end and run with what the reading shows, by position and by keyword,
# or refused for a cause the reading names. 0.17.0's advice to call the refused tool from
# a plain function is followed too, to show what it opened: the gate reads a plain
# function's parameters and never its body.


def _x_recorder(name="table"):
    def purge(table: str, limit: int = 10) -> str:
        _X.append({"table": table, "limit": limit})
        return "ok"

    return purge


def _x_keeping(keyword):
    """A ``functools.wraps`` decorator with a keyword of its own, which it records."""
    import functools

    def decorate(function):
        scope: Dict[str, Any] = {"functools": functools, "function": function, "_X": _X}
        exec(
            f"@functools.wraps(function)\ndef wrapper(*args, {keyword}=True, **kwargs):\n"
            f"    _X.append({{{keyword!r}: {keyword}}})\n"
            "    return function(*args, **kwargs)\n",
            scope,
        )
        return scope["wrapper"]

    return decorate


def test_x_advice_for_a_dispatcher_with_nothing_registered():
    import functools

    @functools.singledispatch
    def process(item: str, mode: str = "safe") -> str:
        _X.append({"item": item, "mode": mode})
        return "base"

    with pytest.raises(invocation.OpenDispatch) as refused:
        reached_parameters(process)
    text = str(refused.value)
    assert text == invocation._OPEN_DISPATCH and invocation._PLAIN_BODY in text
    assert "register the base function itself" in text and "branches on the type" in text

    def calls_it(item: str, mode: str = "safe") -> str:  # 0.17.0's advice
        return process(item, mode)

    def branching(item: str, mode: str = "safe") -> str:  # the advice now
        if isinstance(item, str):
            _X.append({"item": item, "mode": mode})
        return "done"

    base = process.__wrapped__
    for tool in (base, branching):
        _x_read_and_run(tool, {"item": "t"}, {"item": "t", "mode": "safe"})
    assert unsupported_signature(calls_it) is None
    process.register(str, lambda item, mode="safe", purge=True: _X.append({"purge": purge}))
    assert _x_calls(calls_it, {"item": "t"})[1] == [[{"purge": True}]] * 2  # unjudged
    for tool in (base, branching):  # a registration after the walk reaches neither
        _x_read_and_run(tool, {"item": "t"}, {"item": "t", "mode": "safe"})


def test_x_advice_for_an_object_that_answers_for_wrapped():
    import functools

    class Forwarding:
        def __init__(self, function):
            self.function = function

        def __getattr__(self, name):
            return getattr(self.function, name)

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    class Named(Forwarding):
        def __init__(self, function):
            functools.update_wrapper(self, function)
            self.function = function

    forcing = _x_keeping("force")(_x_recorder())
    with pytest.raises(invocation.ProxiedCallable) as refused:
        reached_parameters(Forwarding(_traced(_x_recorder())))
    assert str(refused.value) == invocation._ANSWERS_FOR_WRAPPED
    assert "plain function" not in str(refused.value)
    # with a ``__wrapped__`` of its own, or as the object it calls, the parameter the
    # object hid is read, and refused; where there is none, the tool is read and runs so
    for tool in (Named(forcing), forcing):
        assert "`force` is a parameter" in unsupported_signature(tool)
    for tool in (Named(_traced(_x_recorder())), _traced(_x_recorder())):
        _x_read_and_run(tool, {"table": "t"}, {"table": "t", "limit": 10})


def test_x_advice_for_a_weak_proxy():
    import weakref

    mode = _x_keeping("mode")(_x_recorder())
    plain = _x_recorder()
    for target in (mode, plain):
        with pytest.raises(invocation.WeakProxy) as refused:
            reached_parameters(weakref.proxy(target))
        assert str(refused.value) == invocation._WEAK_PROXY
    for text in (invocation._WEAK_PROXY, invocation._GONE):
        assert text.endswith("register the function itself, not a proxy of it")
    assert "`mode` is a parameter" in unsupported_signature(mode)
    _x_read_and_run(plain, {"table": "t"}, {"table": "t", "limit": 10})


def test_x_advice_for_a_callable_object_whose_class_defines_get():
    import functools

    def audited(binds):
        class Audited:
            def __init__(self, function):
                functools.update_wrapper(self, function)
                self.function = function

            def __call__(self, *args, dry_run=False, **kwargs):
                _X.append({"dry_run": dry_run})
                return self.function(*args, **kwargs)

        class Logged:
            def __init__(self, function):
                functools.update_wrapper(self, function)
                self.function = function

            def __call__(self, *args, **kwargs):
                return self.function(*args, **kwargs)

        if binds:
            for kind in (Audited, Logged):
                kind.__get__ = lambda self, obj, owner=None: (  # type: ignore[attr-defined]
                    self if obj is None else functools.partial(self, obj)
                )
        return Audited, Logged

    purge = _x_recorder()
    Audited, _ = audited(True)
    with pytest.raises(ValueError) as refused:
        reached_parameters(Audited(purge))
    assert str(refused.value) == invocation._DESCRIPTOR
    assert "functools.update_wrapper(self, fn)" in str(refused.value)

    def calls_it(table: str, limit: int = 10) -> str:  # 0.17.0's advice
        return Audited(purge)(table, limit)

    assert unsupported_signature(calls_it) is None
    assert _x_calls(calls_it, {"table": "t"})[1][0][0] == {"dry_run": False}  # unjudged

    Audited, Logged = audited(False)  # the advice now: no ``__get__``
    assert "`dry_run` is a parameter" in unsupported_signature(Audited(purge))

    class Repo:
        def search(self, table: str, limit: int = 10) -> str:
            _X.append({"table": table, "limit": limit})
            return "ok"

    for tool in (Logged(purge), Logged(Repo().search), purge):
        _x_read_and_run(tool, {"table": "t"}, {"table": "t", "limit": 10})


def test_x_advice_for_a_cache_whose_function_cannot_be_told(monkeypatch):
    purge = _x_recorder()
    tool = _six_cached(_six_audited)(purge)
    reached_parameters(purge)  # the first walk has been made
    real = invocation._referents
    monkeypatch.setattr(invocation, "_referents", lambda value: real(value)[:-1])
    with pytest.raises(invocation.UnreadCache) as refused:
        reached_parameters(tool)
    assert str(refused.value) == invocation._REWRAPPED_CACHE
    assert "plain function" not in str(refused.value)
    _x_read_and_run(purge, {"table": "t"}, {"table": "t", "limit": 10})


def test_x_advice_for_a_partialmethod_over_a_callable_object():
    import functools

    class Logged:
        def __init__(self, function):
            functools.update_wrapper(self, function)
            self.function = function

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    def scoped(prefix: str, table: str, limit: int = 10) -> str:
        _X.append({"prefix": prefix, "table": table, "limit": limit})
        return "ok"

    def method(self, prefix: str, table: str, limit: int = 10) -> str:
        _X.append({"prefix": prefix, "table": table, "limit": limit})
        return "ok"

    class Repo:
        over_an_object = functools.partialmethod(Logged(scoped), "p")
        over_a_function = functools.partialmethod(method, "p")

    with pytest.raises(PartialMethodSignature) as refused:
        reached_parameters(Repo().over_an_object)
    assert str(refused.value) == invocation._PARTIALMETHOD
    assert invocation._PLAIN_BODY in str(refused.value)
    ran = {"prefix": "p", "table": "t", "limit": 10}
    _x_read_and_run(Repo().over_a_function, {"table": "t"}, ran, signature=False)
    _x_read_and_run(Logged(scoped), {"prefix": "p", "table": "t"}, ran)


def test_x_advice_for_an_object_that_shows_another_class():
    purge = _x_recorder()
    as_function = dict(_v_liars())["a function"]  # shows a function's face, calls another

    class Transparent:
        __slots__ = ("__wrapped__",)

        def __init__(self, wrapped):
            self.__wrapped__ = wrapped

        @property
        def __class__(self):
            return type(self.__wrapped__)

        def __getattr__(self, name):
            return getattr(self.__wrapped__, name)

        def __call__(self, *args, **kwargs):  # the advice: hand everything on
            return self.__wrapped__(*args, **kwargs)

    class Naming(Transparent):
        __slots__ = ()

        def __call__(self, table, limit=10):
            return self.__wrapped__(table, limit)

    for tool, text in (
        (as_function(purge, purge), invocation._SHOWS_ANOTHER_CLASS),
        (Naming(purge), invocation._SHOWS_ANOTHER_CLASS_AND_TAKES),
    ):
        with pytest.raises(invocation.ProxiedCallable) as refused:
            reached_parameters(tool)
        assert str(refused.value) == text and "plain function" not in text
    for tool in (Transparent(purge), purge):
        _x_read_and_run(tool, {"table": "t"}, {"table": "t", "limit": 10})


def _x_held_holder(held_of, receiver_by_keyword=True):
    """A method decorator that holds ``held_of(method)`` and hands it the receiver by
    keyword, as ``self``."""
    import functools

    def decorate(method):
        held = held_of(method)

        @functools.wraps(method)
        def first(self, *args, **kwargs):
            return held(*args, self=self, **kwargs)

        return first

    return decorate


def test_x_advice_for_a_held_wrapper_with_a_parameter_in_a_bound_place():
    import functools

    def capped(function):
        @functools.wraps(function)
        def wrapper(limit: int = 100, *args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    def reviewed(function):
        @functools.wraps(function)
        def wrapper(skip_review: bool = True, *args, **kwargs):
            _X.append({"skip_review": skip_review})
            return function(*args, **kwargs)

        return wrapper

    def dropped(function):  # the advice: the parameter dropped from the held wrapper
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    def build(held_of):
        def search(self, limit: int = 10) -> str:
            _X.append({"limit": limit})
            return "rows"

        return type("Tools", (), {"search": _x_held_holder(held_of)(search)})().search

    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(build(capped))
    assert str(refused.value) == invocation._HELD_IN_BOUND_PLACE.format(name="limit")
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(build(reviewed))
    assert str(refused.value) == invocation._HELD_DEFAULT_IN_BOUND_PLACE.format(name="skip_review")
    for text in (invocation._HELD_IN_BOUND_PLACE, invocation._HELD_DEFAULT_IN_BOUND_PLACE):
        assert "drop `{name}` from the held wrapper" in text and "plain function" not in text
    _x_read_and_run(build(dropped), {"limit": 5}, {"limit": 5})


def test_x_advice_for_a_held_wrapper_in_another_order():
    import functools

    def holding(second):
        def decorate(function):
            @functools.wraps(function)
            def first(*args, **kwargs):
                return second(*args, **kwargs)

            return first

        return decorate

    @functools.wraps(_x_fetch)
    def swapped(force, sql):
        return _x_fetch(sql, force)

    @functools.wraps(_x_fetch)
    def in_order(sql, force=False):
        return _x_fetch(sql, force)

    @functools.wraps(_x_fetch)
    def by_keyword(*, sql, force=False):
        return _x_fetch(sql, force)

    tool = holding(swapped)(_x_fetch)
    assert _x_calls(tool, {"sql": "S", "force": True})[1][0] == [{"sql": True, "force": "S"}]
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(tool)
    assert str(refused.value) == invocation._HELD_IN_ANOTHER_PLACE.format(
        name="force", declared="sql"
    )
    assert "plain function" not in str(refused.value)
    for second in (in_order, by_keyword):
        _x_read_and_run(
            holding(second)(_x_fetch), {"sql": "S", "force": True}, {"sql": "S", "force": True}
        )


def test_x_advice_for_a_held_weak_proxy():
    import weakref

    def helper(*args, **kwargs):
        return "helped"

    purge = _x_recorder()
    with pytest.raises(HiddenWrapper) as refused:
        reached_parameters(_holding(weakref.proxy(helper), purge))
    assert str(refused.value) == invocation._HELD_WEAK_PROXY
    assert str(refused.value).endswith("hold the callable itself, not a proxy of it")
    _x_read_and_run(_holding(helper, purge), {"table": "t"}, {"table": "t", "limit": 10})


def test_x_advice_for_a_bound_argument_a_wrapper_may_keep():
    """The four reasons of ``BoundArgumentKept`` and the reason of a held wrapper that
    names a receiver otherwise, each with its advice followed. 0.17.0's advice, a plain
    function that names every parameter and calls the facade, is read for its parameters
    alone and runs the facade's shift unjudged: judged ``query='Q'``, ``limit=5``; ran
    ``search(tenant='Q', query=5, limit=10)``."""
    import functools

    def search(tenant: str = "public", query: str = "*", limit: int = 10) -> str:
        _X.append({"tenant": tenant, "query": query, "limit": limit})
        return "rows"

    facade = _x_facade(search)()
    with pytest.raises(invocation.BoundArgumentKept) as refused:
        reached_parameters(facade.run)
    assert str(refused.value) == invocation._unhanded("self", "tenant")

    def calls_it(query: str = "*", limit: int = 10) -> str:  # 0.17.0's advice
        return facade.run(query, limit)

    assert unsupported_signature(calls_it) is None
    assert _x_calls(calls_it, {"query": "Q", "limit": 5})[1][0] == [
        {"tenant": "Q", "query": 5, "limit": 10}
    ]
    # the advice now: the function itself, or with the application's value bound by
    # keyword, or a wrapper that takes the argument under the function's own name for it
    # and hands it on by position
    _x_read_and_run(
        search, {"tenant": "acme", "query": "Q"}, {"tenant": "acme", "query": "Q", "limit": 10}
    )
    _x_read_and_run(
        functools.partial(search, tenant="acme"),
        {"query": "Q"},
        {"tenant": "acme", "query": "Q", "limit": 10},
        signature=False,
    )
    for text in (
        invocation._unhanded("self", "tenant"),
        invocation._shifted("self", "database"),
        invocation._unplaced(None, "limit", True, True),
        invocation._unplaced("self", "limit", True, True),
    ):
        assert "plain function" not in text and "register that one" not in text

    class Repo:
        @_eight_named_instance
        def renamed(self, query: str = "*", limit: int = 10) -> str:
            _X.append({"query": query, "limit": limit})
            return "rows"

        @_eight_named_self
        def named(self, query: str = "*", limit: int = 10) -> str:
            _X.append({"query": query, "limit": limit})
            return "rows"

    with pytest.raises(invocation.BoundArgumentKept, match="`instance`"):
        reached_parameters(Repo().renamed)
    _x_read_and_run(Repo().named, {"query": "Q", "limit": 5}, {"query": "Q", "limit": 5})

    # a bound place of a wrapper that passes the call on: the advice drops it
    def capped(function):
        @functools.wraps(function)
        def wrapper(limit: int = 100, *args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    def tenant_search(tenant: str, limit: int = 10) -> str:
        _X.append({"tenant": tenant, "limit": limit})
        return "rows"

    with pytest.raises(invocation.BoundArgumentKept) as refused:
        reached_parameters(_twelve_tools(capped(tenant_search)).search)
    assert str(refused.value) == invocation._unplaced(None, "limit", True, True)
    assert "Drop `limit` from the function that takes it in that place" in str(refused.value)
    _x_read_and_run(
        _twelve_tools(_traced(tenant_search)).search,
        {"limit": 5},
        {"tenant": "acme", "limit": 5},
    )


def test_x_no_message_advises_a_plain_function_without_saying_what_is_read_of_it():
    """Every reason the kernel gives that names a plain function says that the gate
    reads its parameters and never its body, so anything it calls is not judged; and
    none advises calling the refused tool from one, the advice both reviews followed to
    a tool that ran the refused hazard."""
    texts = [
        value
        for name, value in vars(invocation).items()
        if name.startswith("_") and name.isupper() and type(value) is str
    ]
    texts += list(invocation._LEFT.values()) + list(invocation._HELD_LEFT.values())
    texts += [
        invocation._unhanded("t", "f"),
        invocation._shifted("t", "f"),
        invocation._unplaced(None, "f", True, True),
        invocation._unplaced("t", "f", False, True),
    ]
    assert len(texts) > 30
    for text in texts:
        assert "that calls it" not in text and "names every parameter" not in text, text
        assert "register that one" not in text, text
        if "plain function" in text:
            assert invocation._PLAIN_BODY in text, text
