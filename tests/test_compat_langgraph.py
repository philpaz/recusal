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
    hook = lg.recusal_agate(seen) if use_async else lg.recusal_gate(seen)
    kwargs = {"awrap_tool_call" if use_async else "wrap_tool_call": hook}
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
    """A model can put extra keys in a call. The gate sees them; LangGraph replaces an
    injected argument before the tool runs. A policy must judge only the arguments the
    tool's schema declares, and this pins that LangGraph still strips the rest."""
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
    assert ("read_state", {"sql": SELECT, "state": {"forged": True}}) in seen.calls
    assert ("with_override", {"sql": SELECT, "override": DELETE}) in seen.calls
    assert ("state", ["messages"]) in received, "the forged state reached the tool"
    assert ("override", "") in received, "the model-supplied override reached the tool"
