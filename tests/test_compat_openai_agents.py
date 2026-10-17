"""What the OpenAI Agents SDK guardrail can see, pinned so a new SDK release cannot change it quietly.

`tests/test_openai_agents_gate.py` proves the example's behavior. This file pins the SDK
surface that behavior depends on: every way the SDK can run a function tool must pass through
its tool input guardrails, the guardrail must see the arguments the tool runs with, a
rejection from any guardrail in the list must stop the call, and a guardrail that raises must
stop it too. It also pins the facts `gated_agent` relies on to refuse what the guardrail
cannot see: the SDK's tool types, how agent tools and Codex tools are marked, and where an
agent's tools come from. An ungated agent is the negative control, so a probe that stopped
observing anything would fail rather than pass.

It runs in the pinned CI jobs (the exact versions the README names) and weekly against the
newest SDK in `.github/workflows/compat.yml`. A failure there means an SDK release changed
what the gate can see; STABILITY.md says what happens next.
"""

import asyncio
import copy
import json
import sys
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "openai_agents_gate.py"

SELECT = "SELECT * FROM t"
DELETE = "DELETE FROM t"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)


@pytest.fixture(scope="module")
def oa():
    pytest.importorskip("agents.testing", reason="openai-agents is not installed")
    import importlib.util

    spec = importlib.util.spec_from_file_location("recusal_openai_agents_compat_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CALLS = [
    ("echo", {"text": "hi"}, "c1"),
    ("run_sql", {"sql": DELETE}, "c2"),
    ("run_sql", {"sql": SELECT}, "c3"),
]


class Seen:
    """A policy wrapper recording every call the guardrail judged."""

    def __init__(self, policy):
        self.policy = policy
        self.calls = []

    def __call__(self, name, args):
        self.calls.append((name, json.loads(json.dumps(args))))
        return self.policy(name, args)


def _agent(oa, calls, tools, model=None):
    return oa.Agent(name="a", model=model or oa.scripted_model(calls), tools=tools)


def _run_route(oa, agent, route):
    config = oa.RunConfig(tracing_disabled=True)
    if route == "run":
        asyncio.run(oa.Runner.run(agent, "go", run_config=config))
    elif route == "run_streamed":
        asyncio.run(oa.run_agent(agent, streamed=True))
    elif route == "run_sync":
        oa.Runner.run_sync(agent, "go", run_config=config)
    else:
        raise AssertionError(route)


ROUTES = ["run", "run_streamed", "run_sync"]


@pytest.mark.parametrize("route", ROUTES)
def test_every_run_route_passes_each_call_through_the_guardrail(oa, route):
    seen = Seen(oa.sql_scope_policy)
    agent = oa.gated_agent(_agent(oa, CALLS, [oa.echo, oa.run_sql]), seen)
    oa.EXECUTED.clear()
    _run_route(oa, agent, route)
    assert sorted(seen.calls, key=json.dumps) == sorted(
        ((name, args) for name, args, _ in CALLS), key=json.dumps
    )
    assert ("run_sql", {"sql": DELETE}) not in oa.EXECUTED
    assert ("run_sql", {"sql": SELECT}) in oa.EXECUTED


@pytest.mark.parametrize("route", ROUTES)
def test_negative_control_every_route_runs_delete_ungated(oa, route):
    oa.EXECUTED.clear()
    _run_route(oa, _agent(oa, CALLS, [oa.echo, oa.run_sql]), route)
    assert ("run_sql", {"sql": DELETE}) in oa.EXECUTED


@pytest.mark.parametrize(
    "arguments",
    [
        {"sql": SELECT},
        {"sql": "SELECT 'é ☃ \u0000 \U0001f600'"},
        '{"sql": "SELECT 1", "sql": "SELECT 2"}',
        '{"sql": "SELECT 1", "extra": {"nested": [1, 2.5, null]}}',
    ],
    ids=["plain", "unicode", "duplicate-keys", "extra-fields"],
)
def test_the_guardrail_sees_the_arguments_the_tool_runs_with(oa, arguments):
    ran = []

    @oa.function_tool(strict_mode=False)
    def run_sql(sql: str, extra: typing.Optional[dict] = None) -> str:
        """Record what the tool received."""
        ran.append(sql)
        return sql

    seen = Seen(lambda name, args: [])  # allows everything: this pins parity, not SQL rules
    agent = oa.gated_agent(_agent(oa, [("run_sql", arguments, "c1")], [run_sql]), seen)
    asyncio.run(oa.run_agent(agent))
    assert len(seen.calls) == 1 and len(ran) == 1
    assert seen.calls[0][1]["sql"] == ran[0]


def test_a_rejection_anywhere_in_the_guardrail_list_stops_the_call(oa):
    @oa.tool_input_guardrail
    def allows(data):
        return oa.ToolGuardrailFunctionOutput.allow()

    for position in ("after", "before"):
        tool = copy.copy(oa.run_sql)
        tool.tool_input_guardrails = [allows]
        (gated,) = oa.gate_tools([tool], oa.sql_scope_policy)
        if position == "before":
            gated.tool_input_guardrails.reverse()
        oa.EXECUTED.clear()
        asyncio.run(oa.run_agent(_agent(oa, [("run_sql", {"sql": DELETE}, "c1")], [gated])))
        assert oa.EXECUTED == [], position


def test_a_guardrail_that_raises_stops_the_call(oa):
    @oa.tool_input_guardrail
    def crashes(data):
        raise RuntimeError("guardrail crashed")

    tool = copy.copy(oa.run_sql)
    tool.tool_input_guardrails = [crashes]
    oa.EXECUTED.clear()
    with pytest.raises(Exception):
        asyncio.run(oa.run_agent(_agent(oa, [("run_sql", {"sql": SELECT}, "c1")], [tool])))
    assert oa.EXECUTED == []


@pytest.mark.parametrize("pre_approval", [False, True])
def test_an_approval_never_skips_the_guardrail(oa, pre_approval):
    tool = copy.copy(oa.run_sql)
    tool.needs_approval = True
    seen = Seen(oa.sql_scope_policy)
    agent = oa.gated_agent(_agent(oa, [("run_sql", {"sql": DELETE}, "c1")], [tool]), seen)
    config = oa.RunConfig(
        tracing_disabled=True,
        tool_execution={"pre_approval_tool_input_guardrails": pre_approval},
    )

    async def flow():
        result = await oa.Runner.run(agent, "go", run_config=config)
        if pre_approval:
            # judged before a person is asked; a refusal never reaches them
            assert result.interruptions == [] and len(seen.calls) == 1
            return
        assert len(result.interruptions) == 1 and seen.calls == []
        state = result.to_state()
        state.approve(result.interruptions[0])
        await oa.Runner.run(agent, state, run_config=config)
        assert len(seen.calls) == 1

    oa.EXECUTED.clear()
    asyncio.run(flow())
    assert oa.EXECUTED == []


def test_only_function_tools_carry_tool_input_guardrails(oa):
    import dataclasses

    import agents.tool

    members = [typing.get_origin(m) or m for m in typing.get_args(agents.tool.Tool)]
    with_guardrails = {
        m.__name__
        for m in members
        if dataclasses.is_dataclass(m)
        and "tool_input_guardrails" in {f.name for f in dataclasses.fields(m)}
    }
    # if another type gains guardrails, gate_tools could accept it; review it here first
    assert with_guardrails == {"FunctionTool"}


def test_agent_tools_and_codex_tools_are_marked_the_way_gate_tools_reads_them(oa):
    sub = oa.Agent(name="sub", tools=[oa.echo])
    tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask the sub-agent.")
    assert isinstance(tool, oa.FunctionTool)
    assert tool._is_agent_tool is True
    assert tool._agent_instance is sub
    assert oa.echo._is_agent_tool is False and oa.echo._is_codex_tool is False
    from agents.extensions.experimental.codex import codex_tool

    codex = codex_tool()  # built offline; nothing starts until it is called
    assert isinstance(codex, oa.FunctionTool) and codex._is_codex_tool is True


def test_a_plain_agent_runs_only_its_listed_tools(oa):
    # gated_agent accepts only a plain Agent with no mcp_servers; such an agent must expose
    # exactly its own tool list. Subclasses (SandboxAgent) override this, which is why they
    # are refused.
    from agents.run_context import RunContextWrapper
    from agents.sandbox import SandboxAgent

    agent = oa.gated_agent(oa.Agent(name="a", tools=[oa.echo, oa.run_sql]), oa.sql_scope_policy)
    tools = asyncio.run(agent.get_all_tools(RunContextWrapper(context=None)))
    assert tools == agent.tools
    assert SandboxAgent.get_all_tools is not oa.Agent.get_all_tools
