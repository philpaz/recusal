"""What the OpenAI Agents SDK's MCP paths let a gate see, pinned so a new SDK release cannot change it quietly.

`tests/test_openai_agents_mcp_gate.py` proves the example's behavior. This file pins the SDK
surface that behavior depends on. For local MCP servers: the SDK invokes the server's
`call_tool` with the server's own tool name and the parsed argument dict, whatever public
name the model called (prefixed, sanitized, hashed or shortened), it honors an instance-level
`call_tool`, it passes `meta` only when a resolver produced one, an error result's text is
what the model reads, and a person's approval happens before `call_tool`. For hosted MCP: the
`HostedMCPTool` fields the gate sets, the approval callback's request (name, arguments, server
label) and the response the SDK sends back for a rejection with a reason, and that without a
callback an approval request becomes an interruption rather than a call. An ungated server and
tool are the negative controls, so a probe that stopped observing anything would fail rather
than pass.

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


def _run_server(ex, calls, servers, **agent_kwargs):
    from agents import Agent

    ex.CALLED.clear()
    model = ex.scripted_model(calls)
    agent = Agent(name="a", model=model, mcp_servers=list(servers), **agent_kwargs)
    asyncio.run(ex.run_agent(agent))
    return ex.tool_outputs(model), list(ex.CALLED)


def _public_names(ex, servers, **agent_kwargs):
    from agents import Agent, RunContextWrapper

    agent = Agent(name="a", model=ex.scripted_model([]), mcp_servers=list(servers), **agent_kwargs)
    return [t.name for t in asyncio.run(agent.get_mcp_tools(RunContextWrapper(context=None)))]


def _record_calls(server):
    """Replace the server's call_tool with a recorder, the way the gate wraps it."""
    from mcp.types import CallToolResult, TextContent

    seen = []

    async def recorder(tool_name, arguments, meta=None):
        seen.append((tool_name, arguments, meta))
        return CallToolResult(content=[TextContent(type="text", text="recorded")])

    server.call_tool = recorder
    return seen


# --- local MCP servers ---------------------------------------------------------------------


def test_call_tool_takes_the_tool_name_the_arguments_and_an_optional_meta(ex):
    from agents.mcp import MCPServer

    params = inspect.signature(MCPServer.call_tool).parameters
    assert list(params) == ["self", "tool_name", "arguments", "meta"]
    assert params["meta"].default is None


def test_the_sdk_calls_an_instance_level_call_tool_with_the_parsed_arguments(ex):
    server = ex.InProcessMCPServer()
    seen = _record_calls(server)
    arguments = json.dumps({"sql": SELECT})
    outputs, called = _run_server(ex, [("run_sql", arguments, "c1")], [server])
    assert seen == [("run_sql", json.loads(arguments), None)]
    assert called == []  # the class method was not reached: the instance attribute won
    assert ex._text(outputs["c1"]) == "recorded"


@pytest.mark.parametrize(
    "label, server_specs, mcp_config, raw_names",
    [
        ("prefixed", [("db", ("run_sql",))], {"include_server_in_tool_names": True}, ["run_sql"]),
        (
            "collision",
            [("db.a", ("run_sql",)), ("db_a", ("run_sql",))],
            {"include_server_in_tool_names": True},
            ["run_sql", "run_sql"],
        ),
        ("sanitized", [("db", ("run.sql",))], {"include_server_in_tool_names": True}, ["run.sql"]),
        ("raw looks prefixed", [("db", ("mcp_db__run_sql",))], None, ["mcp_db__run_sql"]),
        (
            "shortened",
            [("db", ("run_sql_" + "x" * 70,))],
            {"include_server_in_tool_names": True},
            ["run_sql_" + "x" * 70],
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_call_tool_receives_the_servers_own_name_whatever_the_public_name(
    ex, label, server_specs, mcp_config, raw_names
):
    servers = [ex.InProcessMCPServer(name, tools) for name, tools in server_specs]
    recorders = [_record_calls(server) for server in servers]
    kwargs = {"mcp_config": mcp_config} if mcp_config else {}
    public = _public_names(ex, servers, **kwargs)
    assert len(public) == len(raw_names), label
    if mcp_config:
        assert all(name.startswith("mcp_") for name in public)  # the public names differ
    calls = [(name, {"sql": SELECT}, f"c{i}") for i, name in enumerate(public, 1)]
    _run_server(ex, calls, servers, **kwargs)
    raw_seen = sorted(name for seen in recorders for name, _, _ in seen)
    assert raw_seen == sorted(raw_names), label
    assert all(args == {"sql": SELECT} for seen in recorders for _, args, _ in seen)


def test_meta_is_passed_only_when_a_resolver_produced_one(ex):
    plain = ex.InProcessMCPServer()
    seen_plain = _record_calls(plain)
    _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], [plain])
    assert seen_plain[0][2] is None

    with_meta = ex.InProcessMCPServer(tool_meta_resolver=lambda *a, **k: {"trace": "t1"})
    seen_meta = _record_calls(with_meta)
    _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], [with_meta])
    assert seen_meta[0][2] == {"trace": "t1"}


def test_an_error_results_text_is_what_the_model_reads_and_the_run_goes_on(ex):
    from mcp.types import CallToolResult, TextContent

    server = ex.InProcessMCPServer()

    async def refuse(tool_name, arguments, meta=None):
        return CallToolResult(content=[TextContent(type="text", text="stopped here")], isError=True)

    server.call_tool = refuse
    outputs, called = _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], [server])
    assert called == []
    assert ex._text(outputs["c1"]) == "stopped here"


def test_a_call_tool_that_raises_becomes_an_error_the_model_reads_not_a_tool_run(ex):
    """The SDK turns an exception from call_tool into the call's error output (its default
    failure handler) and the run goes on; nothing ran. The gate never relies on raising:
    it returns an error result, so this is pinned only as the surrounding behavior."""
    server = ex.InProcessMCPServer()

    async def boom(tool_name, arguments, meta=None):
        raise RuntimeError("call_tool exploded")

    server.call_tool = boom
    outputs, called = _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], [server])
    assert called == []
    assert "error" in ex._text(outputs["c1"]).lower()


def test_a_persons_approval_happens_before_call_tool(ex):
    from agents import Agent, RunConfig, Runner

    server = ex.InProcessMCPServer(require_approval="always")
    seen = _record_calls(server)
    model = ex.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
    agent = Agent(name="a", model=model, mcp_servers=[server])
    config = RunConfig(tracing_disabled=True)

    async def flow():
        result = await Runner.run(agent, "go", run_config=config)
        assert len(result.interruptions) == 1 and seen == []
        state = result.to_state()
        state.approve(result.interruptions[0])
        await Runner.run(agent, state, run_config=config)

    asyncio.run(flow())
    assert seen == [("run_sql", {"sql": SELECT}, None)]


def test_negative_control_an_ungated_server_runs_delete(ex):
    outputs, called = _run_server(
        ex, [("run_sql", {"sql": DELETE}, "c1")], [ex.InProcessMCPServer()]
    )
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
