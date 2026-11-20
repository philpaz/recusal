"""What the LangGraph gate can see, pinned so a new LangGraph release cannot change it quietly.

`tests/test_langgraph_gate.py` proves the example's behavior. This file pins the LangGraph
surface that behavior depends on: every way a graph can run a tool call must pass through
the `wrap_tool_call` / `awrap_tool_call` hook, the hook must see the arguments the tool
runs with, and a hook that raises must stop the call. An ungated node is the negative
control, so a probe that stopped observing anything would fail rather than pass.

It runs in the pinned CI jobs (the exact versions the README names) and weekly against
the newest LangGraph in `.github/workflows/compat.yml`. A failure there means a LangGraph
release changed what the gate can see; STABILITY.md says what happens next.
"""

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "langgraph_gate.py"

SELECT = "SELECT * FROM t"
DELETE = "DELETE FROM t"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+"
)


@pytest.fixture(scope="module")
def lg():
    pytest.importorskip("langgraph")
    import importlib.util

    spec = importlib.util.spec_from_file_location("recusal_langgraph_compat_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _call(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


CALLS = [
    _call("echo", {"text": "hi"}, "1"),
    _call("run_sql", {"sql": DELETE}, "2"),
    _call("run_sql", {"sql": SELECT}, "3"),
]


class Seen:
    """A policy wrapper recording every call the gate judged."""

    def __init__(self, policy):
        self.policy = policy
        self.calls = []

    def __call__(self, name, args):
        self.calls.append((name, dict(args)))
        return self.policy(name, args)


def _graph(tool_node, calls, *, sub=False):
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import tools_condition

    def agent(state):
        if any(isinstance(m, ToolMessage) for m in state["messages"]):
            return {"messages": [AIMessage(content="done")]}
        return {"messages": [AIMessage(content="", tool_calls=calls)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    if sub:
        inner = StateGraph(MessagesState)
        inner.add_node("tools", tool_node)
        inner.add_edge(START, "tools")
        builder.add_node("tools", inner.compile())
    else:
        builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


def _drain(stream):
    for _ in stream:
        pass


async def _adrain(stream):
    async for _ in stream:
        pass


def _routes():
    """Every way to run a graph: (label, uses the async hook, nested in a subgraph, runner)."""
    inp = {"messages": []}
    return [
        ("invoke", False, False, lambda g: g.invoke(inp)),
        ("invoke in a subgraph", False, True, lambda g: g.invoke(inp)),
        *[
            (f"stream {mode}", False, False, lambda g, m=mode: _drain(g.stream(inp, stream_mode=m)))
            for mode in ("values", "updates", "messages", "debug")
        ],
        (
            "stream with subgraphs=True",
            False,
            True,
            lambda g: _drain(g.stream(inp, stream_mode="updates", subgraphs=True)),
        ),
        ("ainvoke, async hook", True, False, lambda g: asyncio.run(g.ainvoke(inp))),
        (
            "astream, async hook",
            True,
            False,
            lambda g: asyncio.run(_adrain(g.astream(inp, stream_mode="updates"))),
        ),
        ("ainvoke, sync hook only", False, False, lambda g: asyncio.run(g.ainvoke(inp))),
    ]


def test_toolnode_still_offers_both_hooks(lg):
    params = inspect.signature(lg.ToolNode).parameters
    assert "wrap_tool_call" in params
    assert "awrap_tool_call" in params


@pytest.mark.parametrize("route", _routes(), ids=lambda r: r[0])
def test_every_route_passes_every_call_through_the_gate(lg, route):
    label, use_async, sub, run = route
    seen = Seen(lg.sql_scope_policy)
    # an async route is served by both hooks together: the async hook is never attached
    # alone, because a ToolNode's sync route would then run tools with no hook
    kwargs = lg.recusal_gates(seen) if use_async else {"wrap_tool_call": lg.recusal_gate(seen)}
    lg.EXECUTED.clear()
    run(_graph(lg.ToolNode([lg.echo, lg.run_sql], **kwargs), CALLS, sub=sub))
    assert sorted(seen.calls, key=str) == sorted(
        [(c["name"], c["args"]) for c in CALLS], key=str
    ), f"{label}: the gate did not judge every proposed call"
    assert ("run_sql", {"sql": DELETE}) not in lg.EXECUTED, f"{label}: a refused call ran"
    assert sorted(lg.EXECUTED, key=str) == sorted(
        [("echo", {"text": "hi"}), ("run_sql", {"sql": SELECT})], key=str
    ), f"{label}: an allowed call did not run"


def test_negative_control_an_ungated_node_runs_the_refused_call(lg):
    lg.EXECUTED.clear()
    _graph(lg.ToolNode([lg.echo, lg.run_sql]), CALLS).invoke({"messages": []})
    assert ("run_sql", {"sql": DELETE}) in lg.EXECUTED


def test_the_gate_sees_the_arguments_the_tool_runs_with(lg):
    seen = Seen(lambda name, args: [])
    lg.EXECUTED.clear()
    node = lg.ToolNode([lg.echo, lg.run_sql], wrap_tool_call=lg.recusal_gate(seen))
    _graph(node, CALLS).invoke({"messages": []})
    assert sorted(seen.calls, key=str) == sorted(lg.EXECUTED, key=str)


def test_a_hook_that_raises_stops_the_call(lg):
    def broken(request, execute):
        raise RuntimeError("gate crashed")

    lg.EXECUTED.clear()
    node = lg.ToolNode([lg.echo, lg.run_sql], wrap_tool_call=broken)
    try:
        _graph(node, CALLS).invoke({"messages": []})
    except Exception:
        pass
    assert lg.EXECUTED == [], "a crashing hook let a tool run"


def test_injected_arguments_reach_the_gate_but_not_the_tool(lg):
    """A model can put extra keys in a call. LangGraph strips every injected argument and
    adds back only the host's values before the tool runs; the gate judges at that point, so
    the policy sees the model's arguments without the forgeries, and this pins that
    LangGraph still strips them."""
    from typing import Annotated

    from langchain_core.tools import InjectedToolArg, tool
    from langgraph.prebuilt import InjectedState

    received = []

    @tool
    def read_state(sql: str, state: Annotated[dict, InjectedState]) -> str:
        """Run SQL with graph state injected."""
        received.append(("state", sorted(state)))
        return "ok"

    @tool
    def with_override(sql: str, override: Annotated[str, InjectedToolArg] = "") -> str:
        """Run SQL; override is supplied by the host, never by the model."""
        received.append(("override", override))
        return "ok"

    seen = Seen(lambda name, args: [])
    node = lg.ToolNode([read_state, with_override], wrap_tool_call=lg.recusal_gate(seen))
    calls = [
        _call("read_state", {"sql": SELECT, "state": {"forged": True}}, "s1"),
        _call("with_override", {"sql": SELECT, "override": DELETE}, "o1"),
    ]
    _graph(node, calls).invoke({"messages": []})
    assert ("read_state", {"sql": SELECT}) in seen.calls
    assert ("with_override", {"sql": SELECT}) in seen.calls
    assert ("state", ["messages"]) in received, "the forged state reached the tool"
    assert ("override", "") in received, "the model-supplied override reached the tool"


# --- both hook slots: no way of running a node skips the gate (0.2.0) ----------------------
#
# A ToolNode consults its hooks by route. Its sync route looks only at ``wrap_tool_call``,
# so a node given only ``awrap_tool_call`` runs every tool there with no hook at all. The
# package therefore attaches both, cannot build the async hook alone for a ToolNode, and
# ``is_gated`` says False for a node whose sync slot is not the gate.

_EVERY_ROUTE = [
    ("invoke", lambda g: g.invoke({"messages": []})),
    ("stream", lambda g: _drain(g.stream({"messages": []}, stream_mode="updates"))),
    ("ainvoke", lambda g: asyncio.run(g.ainvoke({"messages": []}))),
    ("astream", lambda g: asyncio.run(_adrain(g.astream({"messages": []}, stream_mode="updates")))),
    ("batch", lambda g: g.batch([{"messages": []}, {"messages": []}])),
    ("abatch", lambda g: asyncio.run(g.abatch([{"messages": []}, {"messages": []}]))),
]


def _audit_log(tmp_path):
    from recusal.audit import AuditLog

    return AuditLog(path=str(tmp_path / "audit.jsonl"))


def _shapes(lg, policy, tmp_path):
    """Every supported way to put the gate on a node: (label, node, its audit log or None)."""
    import recusal_langgraph as package

    tools = [lg.echo, lg.run_sql]
    log_a, log_b = _audit_log(tmp_path / "a"), _audit_log(tmp_path / "b")
    return [
        ("gated_tool_node", package.gated_tool_node(tools, policy), None),
        ("ToolNode(**recusal_gates)", lg.ToolNode(tools, **package.recusal_gates(policy)), None),
        ("sync hook only", lg.ToolNode(tools, wrap_tool_call=package.recusal_gate(policy)), None),
        ("audited_tool_node", package.audited_tool_node(tools, policy, log_a), log_a),
        (
            "ToolNode(**recusal_audited_gates)",
            lg.ToolNode(tools, **package.recusal_audited_gates(policy, log_b)),
            log_b,
        ),
    ]


@pytest.mark.parametrize("route", _EVERY_ROUTE, ids=lambda r: r[0])
@pytest.mark.parametrize("sub", [False, True], ids=["node", "subgraph"])
def test_every_supported_shape_refuses_on_every_route(lg, tmp_path, route, sub):
    import recusal_langgraph as package

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    label, run = route
    for shape, node, log in _shapes(lg, Seen(lg.sql_scope_policy), tmp_path):
        assert package.is_gated(node), shape
        lg.EXECUTED.clear()
        run(_graph(node, CALLS, sub=sub))
        where = f"{shape} on {label}"
        assert ("run_sql", {"sql": DELETE}) not in lg.EXECUTED, f"{where}: a refused call ran"
        runs = 2 if "batch" in label else 1
        assert sorted(lg.EXECUTED, key=str) == sorted(
            [("echo", {"text": "hi"}), ("run_sql", {"sql": SELECT})] * runs, key=str
        ), f"{where}: an allowed call did not run exactly once per run"
        if log is not None:
            decisions = sorted((e["action"]["tool_use_id"], e["decision"]) for e in log.entries)
            assert decisions == sorted([("1", "PASS"), ("2", "FAIL"), ("3", "PASS")] * runs), where


@pytest.mark.parametrize("route", _EVERY_ROUTE, ids=lambda r: r[0])
def test_every_route_judges_every_call_exactly_once_with_both_hooks(lg, route):
    import recusal_langgraph as package

    seen = Seen(lg.sql_scope_policy)
    lg.EXECUTED.clear()
    route[1](_graph(package.gated_tool_node([lg.echo, lg.run_sql], seen), CALLS))
    runs = 2 if "batch" in route[0] else 1
    assert sorted(seen.calls, key=str) == sorted(
        [(c["name"], c["args"]) for c in CALLS] * runs, key=str
    )


def test_the_async_hook_cannot_be_built_alone_for_a_toolnode(lg, tmp_path):
    import recusal_langgraph as package

    log = _audit_log(tmp_path)
    for name, build in (
        ("recusal_agate", lambda **k: package.recusal_agate(lg.sql_scope_policy, **k)),
        (
            "recusal_audited_agate",
            lambda **k: package.recusal_audited_agate(lg.sql_scope_policy, log, **k),
        ),
    ):
        for not_a_statement in (
            {},
            {"host_refuses_sync_runs": False},
            {"host_refuses_sync_runs": 1},
        ):
            with pytest.raises(TypeError) as caught:
                build(**not_a_statement)
            message = str(caught.value)
            assert message.startswith(f"{name} on its own gates only the async route")
            assert "recusal_gates" in message and "gated_tool_node" in message
        assert callable(build(host_refuses_sync_runs=True))


def test_why_a_toolnode_given_only_the_async_hook_runs_the_refused_call_on_its_sync_route(lg):
    """The framework behaviour the rest of this section exists for, pinned so that a
    LangGraph release which changes it shows up here. The hook is built with the statement
    that is false for a ToolNode, which is the only way left to build it."""
    import recusal_langgraph as package

    seen = Seen(lg.sql_scope_policy)
    hook = package.recusal_agate(seen, host_refuses_sync_runs=True)
    node = lg.ToolNode([lg.echo, lg.run_sql], awrap_tool_call=hook)
    assert package.is_gated(node) is False

    lg.EXECUTED.clear()
    _graph(node, CALLS).invoke({"messages": []})
    assert ("run_sql", {"sql": DELETE}) in lg.EXECUTED and seen.calls == []

    lg.EXECUTED.clear()
    asyncio.run(_graph(node, CALLS).ainvoke({"messages": []}))
    assert ("run_sql", {"sql": DELETE}) not in lg.EXECUTED and len(seen.calls) == 3


def test_is_gated_reads_both_slots(lg, tmp_path):
    import recusal_langgraph as package

    policy, tools = lg.sql_scope_policy, [lg.echo]
    log = _audit_log(tmp_path)

    def foreign(request, execute):
        return execute(request)

    async def aforeign(request, execute):
        return await execute(request)

    alone = package.recusal_agate(policy, host_refuses_sync_runs=True)
    cases = [
        (True, package.gated_tool_node(tools, policy)),
        (True, package.audited_tool_node(tools, policy, log)),
        (True, lg.ToolNode(tools, **package.recusal_gates(policy))),
        (True, lg.ToolNode(tools, wrap_tool_call=package.recusal_gate(policy))),
        (True, lg.ToolNode(tools, wrap_tool_call=package.recusal_audited_gate(policy, log))),
        (False, lg.ToolNode(tools)),
        (False, lg.ToolNode(tools, awrap_tool_call=alone)),
        (False, lg.ToolNode(tools, wrap_tool_call=foreign)),
        (False, lg.ToolNode(tools, wrap_tool_call=foreign, awrap_tool_call=alone)),
        (
            False,
            lg.ToolNode(
                tools, wrap_tool_call=package.recusal_gate(policy), awrap_tool_call=aforeign
            ),
        ),
        (False, package.recusal_gate(policy)),
        (False, None),
        (False, object()),
    ]
    assert [package.is_gated(node) for _, node in cases] == [want for want, _ in cases]


def test_toolnode_keeps_its_hooks_where_is_gated_reads_them(lg):
    """``is_gated`` reads two private attributes. If a release moves them it answers False
    for every node (never True for an ungated one), and this says why."""

    def hook(request, execute):
        return execute(request)

    async def ahook(request, execute):
        return await execute(request)

    node = lg.ToolNode([lg.echo], wrap_tool_call=hook, awrap_tool_call=ahook)
    assert node._wrap_tool_call is hook and node._awrap_tool_call is ahook
    bare = lg.ToolNode([lg.echo])
    assert bare._wrap_tool_call is None and bare._awrap_tool_call is None


def test_the_node_helpers_take_toolnode_options_but_never_a_hook_slot(lg, tmp_path):
    import recusal_langgraph as package

    log = _audit_log(tmp_path)
    node = package.gated_tool_node([lg.echo], lg.sql_scope_policy, name="actions", tags=["t"])
    assert isinstance(node, lg.ToolNode) and node.name == "actions"
    for slot in ("wrap_tool_call", "awrap_tool_call"):
        for build in (
            lambda **k: package.gated_tool_node([lg.echo], lg.sql_scope_policy, **k),
            lambda **k: package.audited_tool_node([lg.echo], lg.sql_scope_policy, log, **k),
        ):
            with pytest.raises(TypeError, match="the gate's own slot"):
                build(**{slot: lambda request, execute: execute(request)})


# --- the tool body runs only for an invocation the parse step just allowed (0.2.0) ---------


def _judging_copy(tool, policy, record=None):
    from recusal_langgraph import _gate

    call = _call(tool.name, {"sql": SELECT}, "c1")
    return _gate._judging(tool, call, policy, record or {}, False), call


def test_a_tool_body_reached_without_the_parse_step_is_refused_not_run(lg, tmp_path):
    """If a framework release ever ran the tool's callable without the parse step this gate
    judges in, the call would run unjudged. The callable on the judging copy refuses unless
    the parse step allowed this very invocation, and one allowance runs one body."""
    from recusal_langgraph import RecusalRefusal

    log = _audit_log(tmp_path)
    record = {"audit": log, "surface": "langgraph.toolnode", "tool_use_id": "c1"}
    seen = Seen(lambda name, args: [])
    copy, call = _judging_copy(lg.run_sql, seen, record)

    lg.EXECUTED.clear()
    with pytest.raises(RecusalRefusal, match="without passing the gate's judging step"):
        copy.func(sql=DELETE)
    assert lg.EXECUTED == [] and seen.calls == []
    assert [e["decision"] for e in log.entries] == ["FAIL"]
    assert log.entries[0]["failures"][0]["check"] == "recusal_unjudgeable_input"

    # through the lifecycle the parse step allows it, and the body runs once
    message = copy.invoke({**call, "type": "tool_call"})
    assert message.status != "error" and lg.EXECUTED == [("run_sql", {"sql": SELECT})]
    assert seen.calls == [("run_sql", {"sql": SELECT})]

    # the allowance was spent by that run: the body cannot be entered again without judgment
    with pytest.raises(RecusalRefusal):
        copy.func(sql=DELETE)
    assert lg.EXECUTED == [("run_sql", {"sql": SELECT})]


def test_the_guard_covers_a_coroutine_and_a_basetool_subclass(lg):
    from langchain_core.tools import BaseTool, StructuredTool
    from recusal_langgraph import RecusalRefusal

    ran = []

    async def afetch(sql: str) -> str:
        ran.append(("coroutine", sql))
        return sql

    def fetch(sql: str) -> str:
        ran.append(("func", sql))
        return sql

    both = StructuredTool.from_function(func=fetch, coroutine=afetch, name="fetch", description="d")

    class Classic(BaseTool):
        name: str = "classic"
        description: str = "d"

        def _run(self, sql: str) -> str:
            ran.append(("_run", sql))
            return sql

        async def _arun(self, sql: str) -> str:
            ran.append(("_arun", sql))
            return sql

    copy, _ = _judging_copy(both, lambda n, a: [])
    with pytest.raises(RecusalRefusal):
        copy.func(sql=DELETE)
    with pytest.raises(RecusalRefusal):
        asyncio.run(copy.coroutine(sql=DELETE))

    classic, _ = _judging_copy(Classic(), lambda n, a: [])
    with pytest.raises(RecusalRefusal):
        classic._run(sql=DELETE)
    with pytest.raises(RecusalRefusal):
        asyncio.run(classic._arun(sql=DELETE))
    assert ran == []
    # the guard keeps the callable's own signature, which the framework inspects
    assert list(inspect.signature(copy.func).parameters) == ["sql"]
    assert inspect.iscoroutinefunction(copy.coroutine)


# --- the second review's cases: every other way back to the unsafe shape (0.2.0) -----------


def test_the_sync_gate_in_the_async_slot_raises_instead_of_gating_one_route(lg, tmp_path):
    """``ToolNode(tools, awrap_tool_call=recusal_gate(policy))`` is the unsafe shape built
    with the other function: the sync slot is empty, so ``invoke`` runs tools with no hook.
    No hook can act on a route that never calls it, so the misuse is made loud on the route
    that does: the sync gate refuses to run from the async slot."""
    import recusal_langgraph as package

    log = _audit_log(tmp_path)
    for name, hook in (
        ("recusal_gate", package.recusal_gate(lg.sql_scope_policy)),
        ("recusal_audited_gate", package.recusal_audited_gate(lg.sql_scope_policy, log)),
    ):
        node = lg.ToolNode([lg.echo, lg.run_sql], awrap_tool_call=hook)
        assert package.is_gated(node) is False
        lg.EXECUTED.clear()
        with pytest.raises(TypeError) as caught:
            asyncio.run(_graph(node, CALLS).ainvoke({"messages": []}))
        assert str(caught.value).startswith(f"{name} is in a ToolNode's async slot")
        assert lg.EXECUTED == [] and log.entries == []


def test_the_sync_gate_in_both_slots_is_not_gated_and_raises_on_the_async_route(lg):
    """The sync hook judges the sync callable. In the async slot it would judge that while
    the route runs the coroutine, which can declare a default the policy never sees."""
    import recusal_langgraph as package
    from langchain_core.tools import StructuredTool

    ran = []

    def fetch(sql: str) -> str:
        ran.append(("func", sql))
        return sql

    async def afetch(sql: str, mode: str = "DROP-EVERYTHING") -> str:
        ran.append(("coroutine", sql, mode))
        return sql

    tool = StructuredTool.from_function(
        func=fetch, coroutine=afetch, name="run_sql", description="d"
    )
    gate = package.recusal_gate(lambda n, a: [])
    node = lg.ToolNode([tool], wrap_tool_call=gate, awrap_tool_call=gate)
    assert package.is_gated(node) is False
    call = [_call("run_sql", {"sql": SELECT}, "1")]
    with pytest.raises(TypeError, match="async slot"):
        asyncio.run(_graph(node, call).ainvoke({"messages": []}))
    assert ran == []
    # with the pair, the async route judges the coroutine and refuses its unjudgeable default
    ran.clear()
    paired = lg.ToolNode([tool], **package.recusal_gates(lambda n, a: []))
    out = asyncio.run(_graph(paired, call).ainvoke({"messages": []}))
    refused = [m for m in out["messages"] if getattr(m, "status", None) == "error"]
    assert ran == [] and "parameter `mode` is absent from the argument schema" in refused[0].content


def test_is_gated_cannot_be_fooled_by_a_copied_marker_or_a_subclass(lg):
    """``functools.wraps`` copies a function's attributes, so a marker attribute would be
    copied onto a wrapper that never calls the gate. A hook is recognised only by being one
    this package built, and a node only if its methods for running a call are ToolNode's."""
    import functools

    import recusal_langgraph as package

    gate = package.recusal_gate(lg.sql_scope_policy)

    @functools.wraps(gate)
    def skips_the_gate(request, execute):
        return execute(request)

    def claims_to_be(request, execute):
        return execute(request)

    claims_to_be.recusal_gate = True  # type: ignore[attr-defined]

    class Replaced(lg.ToolNode):
        def _run_one(self, *args, **kwargs):
            return super()._run_one(*args, **kwargs)

    class Unchanged(lg.ToolNode):
        """Adds nothing that runs a call."""

    tools = [lg.echo, lg.run_sql]
    forged = [
        lg.ToolNode(tools, wrap_tool_call=skips_the_gate),
        lg.ToolNode(tools, wrap_tool_call=claims_to_be),
        lg.ToolNode(tools, wrap_tool_call=functools.partial(gate)),
        Replaced(tools, **package.recusal_gates(lg.sql_scope_policy)),
    ]
    assert [package.is_gated(node) for node in forged] == [False, False, False, False]
    # a subclass is not vouched for even when it changes nothing: it could change anything
    assert not package.is_gated(Unchanged(tools, **package.recusal_gates(lg.sql_scope_policy)))
    # the two forged hooks really do run the refused call, which is why the answer matters
    for node in forged[:2]:
        lg.EXECUTED.clear()
        _graph(node, CALLS).invoke({"messages": []})
        assert ("run_sql", {"sql": DELETE}) in lg.EXECUTED


def test_toolnode_is_laid_out_as_is_gated_reads_it(lg):
    """``is_gated`` reads the two hook slots and the two functions a ToolNode runs from the
    instance, and treats any other instance attribute that shadows a class method as a
    replacement. If a release lays a ToolNode out differently it answers False for every
    node, never True for an ungated one, and this test says what moved."""
    import recusal_langgraph as package

    node = package.gated_tool_node([lg.echo], lg.sql_scope_policy)
    held = vars(node)
    assert {"_wrap_tool_call", "_awrap_tool_call", "func", "afunc"} <= set(held)
    assert held["func"].__func__ is lg.ToolNode._func and held["func"].__self__ is node
    assert held["afunc"].__func__ is lg.ToolNode._afunc and held["afunc"].__self__ is node
    assert [name for name in held if callable(getattr(lg.ToolNode, name, None))] == []
    assert package.is_gated(node)


def test_is_gated_is_false_for_each_way_a_gated_node_can_be_altered(lg, tmp_path):
    """Every case here was reported as answering True while a call ran refused, unjudged
    or unrecorded. Each starts from a correctly gated node and alters one thing."""
    import recusal_langgraph as package

    tools = [lg.echo, lg.run_sql]
    policy = lg.sql_scope_policy
    log = _audit_log(tmp_path)

    def fresh():
        return package.gated_tool_node(tools, policy)

    def passthrough(*args, **kwargs):
        raise AssertionError("not called")

    class Impostor:
        """Hashes and compares like a real hook; is not one."""

        def __init__(self, real):
            self.real = real

        def __hash__(self):
            return hash(self.real)

        def __eq__(self, other):
            return True

        def __call__(self, request, execute):
            return execute(request)

    altered = {}

    node = fresh()
    node._run_one = passthrough
    altered["an instance method replaced"] = node

    node = fresh()
    node._execute_tool_sync = passthrough
    altered["the sync executor replaced on the instance"] = node

    node = fresh()
    node.func = passthrough
    altered["the function the node runs replaced"] = node

    node = fresh()
    node.afunc = passthrough
    altered["the async function the node runs replaced"] = node

    class OverridesInvoke(lg.ToolNode):
        def invoke(self, *args, **kwargs):
            return super().invoke(*args, **kwargs)

    altered["a subclass overriding invoke"] = OverridesInvoke(
        tools, **package.recusal_gates(policy)
    )

    class NoStrip(lg.ToolNode):
        def _inject_tool_args(self, *args, **kwargs):
            return super()._inject_tool_args(*args, **kwargs)

    altered["a subclass overriding injection"] = NoStrip(tools, **package.recusal_gates(policy))

    other = package.recusal_gates(lambda name, args: [])
    altered["hooks built for two different policies"] = lg.ToolNode(
        tools, wrap_tool_call=package.recusal_gate(policy), awrap_tool_call=other["awrap_tool_call"]
    )
    audited = package.recusal_audited_gates(policy, log)
    altered["a plain sync hook beside the audited async one"] = lg.ToolNode(
        tools,
        wrap_tool_call=package.recusal_gates(policy)["wrap_tool_call"],
        awrap_tool_call=audited["awrap_tool_call"],
    )
    altered["audited hooks for two different logs"] = lg.ToolNode(
        tools,
        wrap_tool_call=package.recusal_audited_gate(policy, _audit_log(tmp_path / "x")),
        awrap_tool_call=audited["awrap_tool_call"],
    )
    real = package.recusal_gate(policy)
    altered["an object that compares equal to a real hook"] = lg.ToolNode(
        tools, wrap_tool_call=Impostor(real)
    )
    node = fresh()
    node._wrap_tool_call = None
    altered["the sync hook removed after construction"] = node

    assert {label: package.is_gated(node) for label, node in altered.items()} == dict.fromkeys(
        altered, False
    )
    assert package.is_gated(fresh())
    assert package.is_gated(lg.ToolNode(tools, **audited))
    assert real is not None


# --- what the model sent is judged, whatever the tool's annotations say (0.2.0) ------------


def _disagreeing_tool(ran):
    """A tool whose function marks ``secret`` as host-injected while its argument schema
    declares it an ordinary argument. ToolNode reads the schema, so it does not strip the
    model's value; the function's annotation says the host supplies it."""
    from typing import Annotated

    from langchain_core.tools import InjectedToolArg, StructuredTool
    from pydantic import BaseModel

    def f(x: int, secret: Annotated[str, InjectedToolArg] = "safe") -> str:
        ran.append((x, secret))
        return "ok"

    class Args(BaseModel):
        x: int
        secret: str = "safe"

    return StructuredTool.from_function(func=f, name="f", description="d", args_schema=Args)


def _no_evil(name, args):
    from recusal import Finding

    if "evil" in str(args):
        return [Finding.fail("no_evil", severity="CRITICAL", message="evil is not allowed")]
    return []


@pytest.mark.parametrize("route", _EVERY_ROUTE[:4], ids=lambda r: r[0])
def test_an_argument_the_model_sent_is_judged_even_if_the_function_calls_it_injected(lg, route):
    import recusal_langgraph as package

    ran = []
    seen = Seen(_no_evil)
    node = package.gated_tool_node([_disagreeing_tool(ran)], seen)
    route[1](_graph(node, [_call("f", {"x": 1, "secret": "evil"}, "1")]))
    assert seen.calls == [("f", {"x": 1, "secret": "evil"})]
    assert ran == [], "the value the policy refused must not reach the tool"

    ran.clear()
    seen.calls.clear()
    route[1](_graph(node, [_call("f", {"x": 1, "secret": "fine"}, "1")]))
    assert seen.calls == [("f", {"x": 1, "secret": "fine"})] and ran == [(1, "fine")]


def test_negative_control_the_ungated_node_runs_the_model_supplied_value(lg):
    ran = []
    node = lg.ToolNode([_disagreeing_tool(ran)])
    _graph(node, [_call("f", {"x": 1, "secret": "evil"}, "1")]).invoke({"messages": []})
    assert ran == [(1, "evil")]


def test_a_tool_swapped_into_a_built_node_is_still_judged_on_what_the_model_sent(lg):
    """ToolNode caches which arguments it strips when it is built. A tool put into
    ``tools_by_name`` afterwards is stripped by the old tool's list, so a value the new
    tool marks as host-injected arrives from the model. It is judged."""
    from typing import Annotated

    import recusal_langgraph as package
    from langchain_core.tools import InjectedToolArg, tool

    ran = []

    @tool
    def plain(x: int, secret: str = "safe") -> str:
        """d"""
        ran.append(("plain", x, secret))
        return "ok"

    @tool("plain")
    def replacement(x: int, secret: Annotated[str, InjectedToolArg] = "safe") -> str:
        """d"""
        ran.append(("replacement", x, secret))
        return "ok"

    seen = Seen(_no_evil)
    node = package.gated_tool_node([plain], seen)
    node.tools_by_name["plain"] = replacement
    _graph(node, [_call("plain", {"x": 1, "secret": "evil"}, "1")]).invoke({"messages": []})
    assert ran == []
    assert all("evil" in str(args) for _, args in seen.calls) or seen.calls == []


def test_a_host_supplied_argument_is_still_left_out_of_judgment(lg):
    """The other direction, held: state the host injects is never judged, and a value the
    model forges under that name is replaced by the host's before the tool sees it."""
    from typing import Annotated

    import recusal_langgraph as package
    from langchain_core.tools import tool
    from langgraph.prebuilt import InjectedState

    ran = []

    @tool
    def needs_state(x: int, state: Annotated[dict, InjectedState]) -> str:
        """d"""
        ran.append((x, "forged" in state))  # the host state has no such key
        return "ok"

    seen = Seen(lambda name, args: [])
    node = package.gated_tool_node([needs_state], seen)
    _graph(node, [_call("needs_state", {"x": 1, "state": {"forged": True}}, "1")]).invoke(
        {"messages": []}
    )
    assert seen.calls == [("needs_state", {"x": 1})]
    assert ran == [(1, False)]


def test_the_allowance_is_for_the_judged_arguments_only(lg):
    """One allowance runs one body, and only with the objects the policy judged."""
    from recusal_langgraph import RecusalRefusal

    copy, call = _judging_copy(lg.run_sql, lambda name, args: [])
    args, kwargs = copy._to_args_and_kwargs({"sql": SELECT}, "c1")
    assert args == () and kwargs == {"sql": SELECT}
    lg.EXECUTED.clear()
    with pytest.raises(RecusalRefusal, match="arguments other than the ones the gate judged"):
        copy.func(sql=DELETE)
    with pytest.raises(RecusalRefusal, match="without passing the gate's judging step"):
        copy.func(**kwargs)  # the allowance above was spent by the refused attempt
    assert lg.EXECUTED == []

    args, kwargs = copy._to_args_and_kwargs({"sql": SELECT}, "c1")
    with pytest.raises(RecusalRefusal, match="arguments other than"):
        copy.func(**kwargs, extra="x")
    args, kwargs = copy._to_args_and_kwargs({"sql": SELECT}, "c1")
    assert copy.func(**kwargs) is not None and lg.EXECUTED == [("run_sql", {"sql": SELECT})]


def test_a_function_holding_tool_that_replaces_its_runner_is_refused(lg):
    """A StructuredTool subclass with its own ``_run`` executes that, not the function the
    gate held the invocation to, so the call is refused before the policy runs."""
    from langchain_core.tools import StructuredTool

    ran = []

    class OwnRunner(StructuredTool):
        def _run(self, *args, **kwargs):
            ran.append(kwargs)
            return "ran"

    def fetch(sql: str) -> str:
        return sql

    tool = OwnRunner.from_function(func=fetch, name="run_sql", description="d")
    seen = Seen(lambda name, args: [])
    import recusal_langgraph as package

    node = package.gated_tool_node([tool], seen)
    out = _graph(node, [_call("run_sql", {"sql": SELECT}, "1")]).invoke({"messages": []})
    refused = [m for m in out["messages"] if getattr(m, "status", None) == "error"]
    assert "OwnRunner overrides `_run`" in refused[0].content
    assert ran == [] and seen.calls == []


def test_the_guard_holds_through_a_real_node_for_a_basetool_subclass(lg):
    import recusal_langgraph as package
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel

    ran = []

    class SqlArgs(BaseModel):
        sql: str

    class Classic(BaseTool):
        name: str = "run_sql"
        description: str = "d"
        args_schema: type = SqlArgs

        def _run(self, sql: str) -> str:
            ran.append(sql)
            return sql

    for route in _EVERY_ROUTE[:4]:
        ran.clear()
        node = package.gated_tool_node([Classic()], lg.sql_scope_policy)
        route[1](
            _graph(
                node,
                [_call("run_sql", {"sql": DELETE}, "1"), _call("run_sql", {"sql": SELECT}, "2")],
            )
        )
        assert ran == [SELECT], route[0]


# --- the two private helpers the gate imports, each named by its own probe ----------------
#
# `recusal_langgraph._gate` imports `_get_runnable_config_param`
# (`langchain_core.tools.base`, used by `_framework_keys`) and `_get_all_injected_args`
# (`langgraph.prebuilt.tool_node`, used by `_host_keys`) to find the arguments the host
# injects. Both are private: a release may rename, move or change them without notice, and
# the adapter's framework requirements carry an upper bound because of it. Each probe below
# imports its helper from its exact private path and pins only what the gate reads from it,
# so a release that changes either fails on a test that says which helper and how, instead
# of on a gate test somewhere downstream or on none. `FILTERED_ARGS`, the other name the
# gate imports from `langchain_core.tools.base`, is not private and is already pinned
# against the gate's `_RESERVED` in `tests/test_langgraph_gate.py`.


def _private_helper(module_path, name):
    """The private helper at ``module_path.name``, or a failure that names it.

    Whether the framework is installed at all is the caller's ``importorskip``; once it
    is, a helper that was renamed, moved or removed must fail here, not skip.
    """
    import importlib

    where = f"{module_path}.{name}"
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise AssertionError(
            f"{where} changed: the module the gate imports it from can no longer "
            f"be imported ({exc})"
        ) from exc
    helper = getattr(module, name, None)
    assert callable(helper), (
        f"{where} changed: it no longer exists at that path as a callable; "
        "the gate imports it from there"
    )
    return helper


def _accepts_one_positional(helper, where):
    """The helper still accepts the one positional argument the gate passes it."""
    params = list(inspect.signature(helper).parameters.values())
    assert len(params) == 1 and params[0].kind in (
        params[0].POSITIONAL_ONLY,
        params[0].POSITIONAL_OR_KEYWORD,
    ), f"{where} changed: it no longer accepts the one positional argument the gate passes it"


def test_private_helper_get_runnable_config_param_still_names_the_config_parameter():
    """``_framework_keys`` passes the gate a tool's execution callable and trusts the
    name this helper returns for its ``RunnableConfig`` parameter (or ``None``). The
    callable with no such parameter is the negative control: a helper that had stopped
    inspecting its argument and answered a name anyway would fail it, not pass."""
    pytest.importorskip("langchain_core")
    from langchain_core.runnables import RunnableConfig

    where = "langchain_core.tools.base._get_runnable_config_param"
    helper = _private_helper("langchain_core.tools.base", "_get_runnable_config_param")
    _accepts_one_positional(helper, where)

    def with_config(sql: str, config: RunnableConfig) -> str:
        return sql

    def with_renamed_config(sql: str, cfg: RunnableConfig) -> str:
        return sql

    def without_config(sql: str) -> str:
        return sql

    assert helper(with_config) == "config", (
        f"{where} changed: a callable with a `RunnableConfig` parameter no longer "
        "gives that parameter's name, so `_framework_keys` would judge it as the model's"
    )
    assert helper(with_renamed_config) == "cfg", (
        f"{where} changed: it no longer gives the parameter's own name (`cfg`), so a "
        "tool that renames its `RunnableConfig` parameter would be misjudged"
    )
    assert helper(without_config) is None, (
        f"{where} changed (negative control): a callable with no `RunnableConfig` "
        "parameter no longer gives `None`, so a name the model controls would be "
        "treated as filled by the framework"
    )


def test_private_helper_get_all_injected_args_still_names_every_injected_argument():
    """``_host_keys`` passes the gate a tool and trusts ``all_injected_keys`` on what
    this helper returns to leave host-supplied arguments out of judgment. The tool
    with no injected arguments is the negative control: a helper that had stopped
    observing anything and answered a key anyway would fail it, not pass. Only
    ``all_injected_keys`` is pinned, because that is all the gate reads."""
    pytest.importorskip("langgraph")
    from typing import Annotated

    from langchain_core.tools import InjectedToolArg, tool
    from langgraph.prebuilt import InjectedState, InjectedStore, ToolRuntime

    where = "langgraph.prebuilt.tool_node._get_all_injected_args"
    helper = _private_helper("langgraph.prebuilt.tool_node", "_get_all_injected_args")
    _accepts_one_positional(helper, where)

    @tool
    def with_injected(
        sql: str,
        state: Annotated[dict, InjectedState],
        store: Annotated[object, InjectedStore],
        runtime: ToolRuntime,
        override: Annotated[str, InjectedToolArg] = "",
    ) -> str:
        """Run SQL with state, store, runtime and a generic injected argument."""
        return sql

    @tool
    def without_injected(sql: str) -> str:
        """Run SQL with no host-injected arguments."""
        return sql

    injected = helper(with_injected)
    assert hasattr(injected, "all_injected_keys"), (
        f"{where} changed: what it returns no longer has `all_injected_keys`, "
        "which is what `_host_keys` reads"
    )
    assert set(injected.all_injected_keys) == {"state", "store", "runtime", "override"}, (
        f"{where} changed: a tool with state, store, runtime and a generic injected "
        f"argument no longer gives all of their names, got {injected.all_injected_keys!r}; "
        "an argument it misses would be judged as the model's"
    )
    plain = helper(without_injected)
    assert hasattr(plain, "all_injected_keys"), (
        f"{where} changed: what it returns no longer has `all_injected_keys`, "
        "which is what `_host_keys` reads"
    )
    assert set(plain.all_injected_keys) == set(), (
        f"{where} changed (negative control): a tool with no injected arguments no "
        f"longer gives the empty answer, got {plain.all_injected_keys!r}; a probe "
        "that stopped observing anything would fail here, not pass"
    )
