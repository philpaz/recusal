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
    "adapters/pydantic-ai/src/recusal_pydantic_ai/_gate.py": {"detached", "reached_parameters"},
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
