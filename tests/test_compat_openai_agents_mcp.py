"""What the OpenAI Agents SDK's MCP paths let a gate see, pinned so a new SDK release cannot change it quietly.

`tests/test_openai_agents_mcp_gate.py` proves the example's behavior. This file pins the SDK
surface that behavior depends on. For local MCP servers: `MCPServer` takes and keeps
`tool_input_guardrails`, every tool a server lists becomes a function tool carrying those
guardrails, the guardrail sees the public tool name and the JSON arguments the server is
then called with, a rejection stops the call and becomes its output, a raising guardrail
stops the call, and a person's approval never skips the guardrail. For hosted MCP: the
`HostedMCPTool` fields the gate sets, the approval callback's request (name, arguments,
server label) and the response the SDK sends back for a rejection with a reason, and that
without a callback an approval request becomes an interruption rather than a call. An
ungated server and tool are the negative controls, so a probe that stopped observing
anything would fail rather than pass.

It runs in the pinned CI jobs (the exact versions the README names) and weekly against the
newest SDK in `.github/workflows/compat.yml`. A failure there means an SDK release changed
what the gate can see; STABILITY.md says what happens next.
"""

import asyncio
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "openai_agents_mcp_gate.py"

SELECT = "SELECT * FROM t"
DELETE = "DELETE FROM t"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)


@pytest.fixture(scope="module")
def ex():
    pytest.importorskip("agents.mcp", reason="openai-agents is not installed")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recusal_openai_agents_mcp_compat_example", EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _guardrail(func, name="probe"):
    from agents import tool_input_guardrail

    return tool_input_guardrail(func, name=name)


def _allow():
    from agents import ToolGuardrailFunctionOutput

    return ToolGuardrailFunctionOutput.allow()


def _run_server(ex, calls, server, **agent_kwargs):
    from agents import Agent

    ex.CALLED.clear()
    model = ex.scripted_model(calls)
    agent = Agent(name="a", model=model, mcp_servers=[server], **agent_kwargs)
    asyncio.run(ex.run_agent(agent))
    return ex.tool_outputs(model), list(ex.CALLED)


# --- local MCP servers ---------------------------------------------------------------------


def test_mcp_server_takes_and_keeps_tool_input_guardrails(ex):
    from agents.mcp import MCPServer

    params = inspect.signature(MCPServer.__init__).parameters
    assert "tool_input_guardrails" in params
    assert params["tool_input_guardrails"].kind is inspect.Parameter.KEYWORD_ONLY
    guardrail = _guardrail(lambda data: _allow())
    server = ex.InProcessMCPServer(tool_input_guardrails=[guardrail])
    assert server.tool_input_guardrails == [guardrail]
    assert ex.InProcessMCPServer().tool_input_guardrails is None


def test_every_listed_tool_becomes_a_function_tool_carrying_the_servers_guardrails(ex):
    from agents import Agent, FunctionTool, RunContextWrapper

    guardrail = _guardrail(lambda data: _allow())
    server = ex.InProcessMCPServer(tool_input_guardrails=[guardrail])
    agent = Agent(name="a", model=ex.scripted_model([]), mcp_servers=[server])
    tools = asyncio.run(agent.get_mcp_tools(RunContextWrapper(context=None)))
    assert sorted(tool.name for tool in tools) == ["echo", "run_sql"]
    for tool in tools:
        assert isinstance(tool, FunctionTool)
        assert tool.tool_input_guardrails == [guardrail]
        assert tool.tool_input_guardrails is not server.tool_input_guardrails  # a copy per tool


def test_the_guardrail_sees_the_public_name_and_the_json_the_server_is_called_with(ex):
    seen = []

    def probe(data):
        seen.append((data.context.tool_name, data.context.tool_arguments))
        return _allow()

    server = ex.InProcessMCPServer(tool_input_guardrails=[_guardrail(probe)])
    arguments = json.dumps({"sql": SELECT})
    _, called = _run_server(ex, [("run_sql", arguments, "c1")], server)
    assert seen == [("run_sql", arguments)]
    assert called == [("db", "run_sql", json.loads(arguments))]


def test_with_server_prefixed_names_the_guardrail_sees_the_prefixed_name(ex):
    from agents.mcp.util import MCPUtil

    assert MCPUtil._build_prefixed_tool_base_name("db", "run_sql") == "mcp_db__run_sql"
    seen = []

    def probe(data):
        seen.append(data.context.tool_name)
        return _allow()

    server = ex.InProcessMCPServer(tool_input_guardrails=[_guardrail(probe)])
    _, called = _run_server(
        ex,
        [("mcp_db__run_sql", {"sql": SELECT}, "c1")],
        server,
        mcp_config={"include_server_in_tool_names": True},
    )
    assert seen == ["mcp_db__run_sql"]
    assert called == [("db", "run_sql", {"sql": SELECT})]  # the server still gets its own name


def test_a_rejection_stops_the_call_and_becomes_its_output(ex):
    from agents import ToolGuardrailFunctionOutput

    server = ex.InProcessMCPServer(
        tool_input_guardrails=[
            _guardrail(lambda data: ToolGuardrailFunctionOutput.reject_content("stopped here"))
        ]
    )
    outputs, called = _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], server)
    assert called == []
    assert outputs["c1"] == "stopped here"


def test_a_guardrail_that_raises_stops_the_call(ex):
    def boom(data):
        raise RuntimeError("guardrail exploded")

    server = ex.InProcessMCPServer(tool_input_guardrails=[_guardrail(boom)])
    with pytest.raises(Exception, match="guardrail exploded"):
        _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], server)
    assert ex.CALLED == []


@pytest.mark.parametrize("pre_approval", [False, True])
def test_an_approval_never_skips_the_guardrail(ex, pre_approval):
    from agents import Agent, RunConfig, Runner

    seen = []

    def probe(data):
        seen.append(data.context.tool_name)
        return _allow()

    server = ex.InProcessMCPServer(
        require_approval="always", tool_input_guardrails=[_guardrail(probe)]
    )
    model = ex.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
    agent = Agent(name="a", model=model, mcp_servers=[server])
    config = RunConfig(
        tracing_disabled=True, tool_execution={"pre_approval_tool_input_guardrails": pre_approval}
    )

    async def flow():
        result = await Runner.run(agent, "go", run_config=config)
        if pre_approval:
            assert seen == ["run_sql"] and len(result.interruptions) == 1
        else:
            assert seen == [] and len(result.interruptions) == 1
        state = result.to_state()
        state.approve(result.interruptions[0])
        await Runner.run(agent, state, run_config=config)

    ex.CALLED.clear()
    asyncio.run(flow())
    # judged when it runs; with pre-approval guardrails on, also before the person is asked
    assert seen == ["run_sql", "run_sql"] if pre_approval else ["run_sql"]
    assert ex.CALLED == [("db", "run_sql", {"sql": SELECT})]


def test_negative_control_an_ungated_server_runs_delete(ex):
    outputs, called = _run_server(ex, [("run_sql", {"sql": DELETE}, "c1")], ex.InProcessMCPServer())
    assert called == [("db", "run_sql", {"sql": DELETE})]
    assert ex._text(outputs["c1"]) == f"recorded SQL: {DELETE}"


# --- hosted MCP -----------------------------------------------------------------------------


def test_hosted_mcp_tool_has_the_fields_the_gate_sets(ex):
    import dataclasses

    from agents import HostedMCPTool

    names = {f.name for f in dataclasses.fields(HostedMCPTool)}
    assert {"tool_config", "on_approval_request"} <= names
    tool = HostedMCPTool(
        tool_config={"type": "mcp", "server_label": "db", "server_url": "https://x.invalid"}
    )
    assert tool.on_approval_request is None and tool.name == "hosted_mcp"
    copied = dataclasses.replace(
        tool, tool_config={**tool.tool_config, "require_approval": "always"}
    )
    assert copied.tool_config["require_approval"] == "always"
    assert "require_approval" not in tool.tool_config


def test_the_approval_callback_receives_name_arguments_and_server_label(ex):
    from agents import HostedMCPTool, MCPToolApprovalRequest

    seen = []

    def probe(request):
        assert isinstance(request, MCPToolApprovalRequest)
        seen.append((request.data.name, request.data.arguments, request.data.server_label))
        return {"approve": True}

    tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "db",
            "server_url": "https://x.invalid",
            "require_approval": "always",
        },
        on_approval_request=probe,
    )
    arguments = json.dumps({"sql": SELECT})
    responses = ex.run_hosted([("run_sql", arguments, "r1")], gated=False, tool=tool)
    assert seen == [("run_sql", arguments, "db")]
    assert responses["r1"] == {
        "approval_request_id": "r1",
        "approve": True,
        "type": "mcp_approval_response",
    }


def test_a_rejection_with_a_reason_is_sent_back_as_the_approval_response(ex):
    from agents import HostedMCPTool

    tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "db",
            "server_url": "https://x.invalid",
            "require_approval": "always",
        },
        on_approval_request=lambda request: {"approve": False, "reason": "because"},
    )
    responses = ex.run_hosted([("run_sql", {"sql": SELECT}, "r1")], gated=False, tool=tool)
    assert responses["r1"] == {
        "approval_request_id": "r1",
        "approve": False,
        "type": "mcp_approval_response",
        "reason": "because",
    }


def test_without_a_callback_an_approval_request_is_an_interruption_not_a_call(ex):
    from agents import Agent, HostedMCPTool, RunConfig, Runner
    from agents.testing import ScriptedModel, assistant_message

    tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "db",
            "server_url": "https://x.invalid",
            "require_approval": "always",
        }
    )
    model = ScriptedModel(
        [[ex.approval_request("run_sql", {"sql": SELECT}, "r1")], [assistant_message("done")]]
    )
    result = asyncio.run(
        Runner.run(
            Agent(name="a", model=model, tools=[tool]),
            "go",
            run_config=RunConfig(tracing_disabled=True),
        )
    )
    assert len(result.interruptions) == 1
    assert model.last_call is not None and not any(
        isinstance(item, dict) and item.get("type") == "mcp_approval_response"
        for item in model.last_call.input
    )


def test_negative_control_a_tool_that_approves_everything_approves_delete(ex):
    responses = ex.run_hosted([("run_sql", {"sql": DELETE}, "r1")], gated=False)
    assert responses["r1"]["approve"] is True
