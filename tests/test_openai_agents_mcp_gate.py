"""Regression tests for the offline OpenAI Agents SDK + Recusal MCP example."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "openai_agents_mcp_gate.py"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"
REFUSED = "Recusal refused `run_sql` [FAIL]"


def test_python_version_refusal_is_clear_on_every_python():
    code = (
        "import runpy, sys; "
        "sys.version_info = (3, 9, 0, 'final', 0); "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "openai_agents_mcp_gate.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_the_example_shares_the_function_tool_examples_pins_and_jobs():
    """One runtime, one set of pins: this example imports the function-tool example, names
    the same versions, and runs in the same two CI jobs rather than new ones."""
    source = EXAMPLE.read_text(encoding="utf-8")
    assert "from examples.openai_agents_gate import" in source
    base = (ROOT / "examples" / "openai_agents_gate.py").read_text(encoding="utf-8")
    for pin in ("openai-agents==0.24.0", "openai==3.34.0"):
        assert pin in source and pin in base, pin
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert ci.count("tests/test_openai_agents_mcp_gate.py") == 2
    assert ci.count("tests/test_compat_openai_agents_mcp.py") == 2


def _text(output):
    if isinstance(output, list):
        return "".join(item.get("text", "") for item in output if isinstance(item, dict))
    return str(output)


def _refuse_deletes(name, args):
    """A policy keyed on arguments alone, so what it saw as the tool NAME is the evidence."""
    from recusal import Finding

    sql = args.get("sql", "") if isinstance(args, dict) else ""
    if sql.upper().startswith("DELETE"):
        return [Finding.fail("sql_scope", severity="CRITICAL", detail="destructive SQL")]
    return []


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestOpenAIAgentsMcpGate:
    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.mcp", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "recusal_openai_agents_mcp_gate_example", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def three_calls():
        return [
            ("echo", {"text": "hello"}, "c1"),
            ("run_sql", {"sql": DELETE}, "c2"),
            ("run_sql", {"sql": SELECT}, "c3"),
        ]

    # --- local MCP servers ---------------------------------------------------------------

    def test_allowed_call_reaches_the_server(self, example):
        outputs, called = example.run_calls([("run_sql", {"sql": SELECT}, "c1")])
        assert called == [("db", "run_sql", {"sql": SELECT})]
        assert _text(outputs["c1"]) == f"recorded SQL: {SELECT}"

    def test_refused_call_never_reaches_the_server_and_reason_reaches_the_model(self, example):
        outputs, called = example.run_calls([("run_sql", {"sql": DELETE}, "c1")])
        assert called == []
        text = _text(outputs["c1"])
        assert text.startswith(REFUSED) and "top-level WHERE" in text

    def test_several_calls_in_one_turn_are_gated_separately(self, example):
        outputs, called = example.run_calls(self.three_calls())
        assert _text(outputs["c1"]) == "hello"
        assert _text(outputs["c2"]).startswith(REFUSED)
        assert _text(outputs["c3"]) == f"recorded SQL: {SELECT}"
        assert called == [("db", "echo", {"text": "hello"}), ("db", "run_sql", {"sql": SELECT})]

    def test_policy_exception_fails_closed(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        outputs, called = example.run_calls([("echo", {"text": "x"}, "c1")], policy=broken_policy)
        assert called == []
        assert _text(outputs["c1"]) == "Recusal failed closed (policy error): policy exploded"

    def test_negative_control_the_server_runs_delete_without_the_gate(self, example):
        outputs, called = example.run_calls([("run_sql", {"sql": DELETE}, "c1")], gated=False)
        assert called == [("db", "run_sql", {"sql": DELETE})]
        assert _text(outputs["c1"]) == f"recorded SQL: {DELETE}"

    def test_gate_mcp_server_gates_in_place_once_and_leaves_guardrails_alone(self, example):
        from agents import ToolGuardrailFunctionOutput, tool_input_guardrail

        order = []

        @tool_input_guardrail
        def first(data):
            order.append("first")
            return ToolGuardrailFunctionOutput.allow()

        server = example.InProcessMCPServer(tool_input_guardrails=[first])
        assert not example.is_gated_mcp_server(server)
        assert example.gate_mcp_server(server, example.sql_scope_policy) is server
        assert example.is_gated_mcp_server(server)
        gated_once = server.call_tool
        assert example.gate_mcp_server(server, example.sql_scope_policy) is server
        assert server.call_tool is gated_once  # idempotent: one gate, not a chain
        assert server.tool_input_guardrails == [first]  # the app's guardrails stay as written
        outputs, called = example.run_calls(
            [("run_sql", {"sql": DELETE}, "c1")], gated=False, server=server
        )
        assert order == ["first"] and called == [] and _text(outputs["c1"]).startswith(REFUSED)

    def test_a_server_carries_one_policy_the_same_again_is_a_no_op_a_different_one_is_refused(
        self, example
    ):
        from agents import Agent

        server = example.InProcessMCPServer()
        assert example.gate_mcp_server(server, example.sql_scope_policy) is server
        gated_once = server.call_tool
        assert example.gate_mcp_server(server, example.sql_scope_policy) is server
        assert server.call_tool is gated_once  # the same policy again: nothing changed
        with pytest.raises(ValueError, match="already gated with a different policy"):
            example.gate_mcp_server(server, _refuse_deletes)
        assert server.call_tool is gated_once  # neither kept silently nor replaced
        # the same refusal through gated_agent on a second agent that shares the connection
        other = Agent(name="b", model=example.scripted_model([]), mcp_servers=[server])
        with pytest.raises(ValueError, match="one policy for every agent that shares it"):
            example.gated_agent(other, _refuse_deletes)

    def test_negative_control_the_reviewers_sequence_no_longer_runs_under_the_first_policy(
        self, example
    ):
        """1. gate a server with an allowing policy; 2. reuse the connection in another agent;
        3. ask gated_agent for a policy that refuses everything. Before the fix the agent
        reported itself gated, only the first policy ran, and the tool executed."""
        from agents import Agent

        def allow_all(name, args):
            return []

        def refuse_all(name, args):
            from recusal import Finding

            return [Finding.fail("refuse_all", severity="CRITICAL", message="no")]

        server = example.gate_mcp_server(example.InProcessMCPServer(), allow_all)
        second = Agent(
            name="b",
            model=example.scripted_model([("run_sql", {"sql": SELECT}, "c1")]),
            mcp_servers=[server],
        )
        with pytest.raises(ValueError, match="different policy"):
            example.gated_agent(second, refuse_all)
        # and with its own server object, the stricter policy is the one that runs
        own = example.InProcessMCPServer()
        model = example.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
        strict = example.gated_agent(Agent(name="c", model=model, mcp_servers=[own]), refuse_all)
        example.CALLED.clear()
        asyncio.run(example.run_agent(strict))
        assert example.CALLED == []
        assert "Recusal refused" in _text(example.tool_outputs(model)["c1"])

    def test_gate_mcp_server_refuses_what_is_not_a_server(self, example):
        with pytest.raises(TypeError, match="not an agents.mcp.MCPServer"):
            example.gate_mcp_server(object(), example.sql_scope_policy)

    @pytest.mark.parametrize(
        "label, server_specs, mcp_config, raw_names",
        [
            ("plain", [("db", ("run_sql",))], None, ["run_sql"]),
            (
                "prefixed",
                [("db", ("run_sql",))],
                {"include_server_in_tool_names": True},
                ["run_sql"],
            ),
            (
                "collision",
                [("db.a", ("run_sql",)), ("db_a", ("run_sql",))],
                {"include_server_in_tool_names": True},
                ["run_sql", "run_sql"],
            ),
            (
                "raw name looks prefixed",
                [("db", ("mcp_db__run_sql",))],
                None,
                ["mcp_db__run_sql"],
            ),
            (
                "sanitized",
                [("db", ("run.sql",))],
                {"include_server_in_tool_names": True},
                ["run.sql"],
            ),
            (
                "shortened",
                [("db", ("run_sql_" + "x" * 70,))],
                {"include_server_in_tool_names": True},
                ["run_sql_" + "x" * 70],
            ),
        ],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_the_policy_judges_the_servers_own_tool_name_under_every_public_naming(
        self, example, label, server_specs, mcp_config, raw_names
    ):
        """The SDK prefixes, sanitizes, hashes and shortens public names; the gate sits at
        call_tool, so the policy sees the raw name the server runs and nothing runs when
        it refuses. The policy here keys on the arguments alone, so the names it recorded
        are the evidence of what it was shown."""
        from agents import Agent, RunContextWrapper

        seen = []

        def recording(name, args):
            seen.append(name)
            return _refuse_deletes(name, args)

        servers = [
            example.gate_mcp_server(example.InProcessMCPServer(name, tools), recording)
            for name, tools in server_specs
        ]
        kwargs = {"mcp_config": mcp_config} if mcp_config else {}
        probe = Agent(name="a", model=example.scripted_model([]), mcp_servers=servers, **kwargs)
        public = [t.name for t in asyncio.run(probe.get_mcp_tools(RunContextWrapper(context=None)))]
        assert len(public) == len(raw_names)
        calls = [(name, {"sql": DELETE}, f"c{i}") for i, name in enumerate(public, 1)]
        example.CALLED.clear()
        model = example.scripted_model(calls)
        asyncio.run(example.run_agent(Agent(name="a", model=model, mcp_servers=servers, **kwargs)))
        assert sorted(seen) == sorted(raw_names), label
        assert example.CALLED == []
        for _, _, call_id in calls:
            assert "[FAIL]: sql_scope" in _text(example.tool_outputs(model)[call_id])

    def test_a_function_tool_holding_the_name_hides_the_mcp_tool_without_prefixing(self, example):
        """Pinned as observed on the pinned SDK: an unprefixed MCP tool whose name a function
        tool already holds is shadowed, so the model's call runs the function tool. The MCP
        gate never saw a call because the server was never called; the function tool is
        gated by the function-tool example, which gated_agent applies."""
        from agents import Agent, RunContextWrapper

        from examples.openai_agents_gate import run_sql

        seen = []

        def recording(name, args):
            seen.append(name)
            return _refuse_deletes(name, args)

        server = example.gate_mcp_server(example.InProcessMCPServer(tools=("run_sql",)), recording)
        agent = Agent(
            name="a", model=example.scripted_model([]), tools=[run_sql], mcp_servers=[server]
        )
        assert [
            t.name for t in asyncio.run(agent.get_mcp_tools(RunContextWrapper(context=None)))
        ] == ["run_sql"]
        example.CALLED.clear()
        model = example.scripted_model([("run_sql", {"sql": DELETE}, "c1")])
        gated = example.gated_agent(
            Agent(name="a", model=model, tools=[run_sql], mcp_servers=[server]), recording
        )
        asyncio.run(example.run_agent(gated))
        assert example.CALLED == []
        assert "[FAIL]: sql_scope" in _text(example.tool_outputs(model)["c1"])
        assert seen == ["run_sql"]  # judged once, by the function-tool guardrail

    def test_an_approval_never_skips_the_gate(self, example):
        from agents import Agent, RunConfig, Runner

        model = example.scripted_model([("run_sql", {"sql": DELETE}, "c1")])
        server = example.gate_mcp_server(
            example.InProcessMCPServer(require_approval="always"), example.sql_scope_policy
        )
        agent = Agent(name="a", model=model, mcp_servers=[server])
        config = RunConfig(tracing_disabled=True)

        async def flow():
            result = await Runner.run(agent, "go", run_config=config)
            assert len(result.interruptions) == 1
            state = result.to_state()
            state.approve(result.interruptions[0])
            await Runner.run(agent, state, run_config=config)

        example.CALLED.clear()
        asyncio.run(flow())
        assert example.CALLED == []
        assert _text(example.tool_outputs(model)["c1"]).startswith(REFUSED)

    # --- hosted MCP ----------------------------------------------------------------------

    def test_hosted_refusal_is_sent_back_as_a_rejection_with_the_reason(self, example):
        responses = example.run_hosted([("run_sql", {"sql": DELETE}, "r1")])
        assert responses["r1"]["approve"] is False
        assert responses["r1"]["reason"].startswith(REFUSED)

    def test_hosted_allowed_call_is_approved(self, example):
        responses = example.run_hosted([("run_sql", {"sql": SELECT}, "r1")])
        assert responses["r1"]["approve"] is True and "reason" not in responses["r1"]

    def test_hosted_requests_in_one_turn_are_judged_separately(self, example):
        responses = example.run_hosted(
            [
                ("echo", {"text": "hi"}, "r1"),
                ("run_sql", {"sql": DELETE}, "r2"),
                ("run_sql", {"sql": SELECT}, "r3"),
            ]
        )
        assert [responses[r]["approve"] for r in ("r1", "r2", "r3")] == [True, False, True]

    def test_hosted_policy_exception_fails_closed(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        responses = example.run_hosted([("echo", {"text": "x"}, "r1")], policy=broken_policy)
        assert responses["r1"] == {
            "approval_request_id": "r1",
            "approve": False,
            "type": "mcp_approval_response",
            "reason": "Recusal failed closed (policy error): policy exploded",
        }

    def test_negative_control_the_tools_own_callback_approves_delete_without_the_gate(
        self, example
    ):
        responses = example.run_hosted([("run_sql", {"sql": DELETE}, "r1")], gated=False)
        assert responses["r1"]["approve"] is True

    def test_gate_hosted_mcp_requires_approval_for_every_call_and_keeps_the_callback(self, example):
        from agents import HostedMCPTool

        asked = []

        def existing(request):
            asked.append(request.data.name)
            return {"approve": False, "reason": "the app said no"}

        original = HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "db",
                "server_url": "https://example.invalid/mcp",
            },
            on_approval_request=existing,
        )
        assert not example.is_gated_hosted_mcp(original)
        gated = example.gate_hosted_mcp(original, example.sql_scope_policy)
        assert example.is_gated_hosted_mcp(gated) and not example.is_gated_hosted_mcp(original)
        assert gated is not original and original.tool_config.get("require_approval") is None
        assert gated.tool_config["require_approval"] == "always"
        responses = example.run_hosted(
            [("run_sql", {"sql": DELETE}, "r1"), ("run_sql", {"sql": SELECT}, "r2")],
            gated=False,
            tool=gated,
        )
        # the gate refused r1 before the app's callback ran; r2 passed the gate and the app refused it
        assert asked == ["run_sql"]
        assert responses["r1"]["reason"].startswith(REFUSED)
        assert responses["r2"] == {
            "approval_request_id": "r2",
            "approve": False,
            "type": "mcp_approval_response",
            "reason": "the app said no",
        }

    def test_a_hosted_tool_that_does_not_require_approval_is_not_gated(self, example):
        """The documented limit: a call OpenAI does not ask approval for is never seen."""
        from agents import Agent, HostedMCPTool

        tool = HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "db",
                "server_url": "https://example.invalid/mcp",
                "require_approval": "never",
            },
            on_approval_request=example.recusal_approval(example.sql_scope_policy),
        )
        assert not example.is_gated_hosted_mcp(tool)
        assert not example.is_gated(Agent(name="a", model=example.scripted_model([]), tools=[tool]))

    # --- the whole agent -------------------------------------------------------------------

    def test_gated_agent_gates_function_tools_servers_and_hosted_tools_together(self, example):
        from agents import Agent, HostedMCPTool

        from examples.openai_agents_gate import run_sql

        hosted = HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "db",
                "server_url": "https://example.invalid/mcp",
            }
        )
        server = example.InProcessMCPServer()
        agent = Agent(
            name="a",
            model=example.scripted_model([]),
            tools=[run_sql, hosted],
            mcp_servers=[server],
        )
        assert not example.is_gated(agent)
        gated = example.gated_agent(agent, example.sql_scope_policy)
        assert gated is not agent and example.is_gated(gated)
        assert gated.mcp_servers == [server] and example.is_gated_mcp_server(server)  # in place
        assert [type(t).__name__ for t in gated.tools] == ["FunctionTool", "HostedMCPTool"]
        assert gated.tools[0] is not run_sql and gated.tools[1] is not hosted
        assert not example.is_gated(
            agent.clone(mcp_servers=[])
        )  # the original's tools are untouched

    def test_gated_agent_refuses_handoffs_and_subclasses(self, example):
        from agents import Agent

        class Special(Agent):
            pass

        with pytest.raises(ValueError, match="not a plain Agent"):
            example.gated_agent(
                Special(name="s", model=example.scripted_model([])), example.sql_scope_policy
            )
        other = Agent(name="other", model=example.scripted_model([]))
        with pytest.raises(ValueError, match="handoffs"):
            example.gated_agent(
                Agent(name="a", model=example.scripted_model([]), handoffs=[other]),
                example.sql_scope_policy,
            )

    def test_gated_agent_refuses_an_ungated_sub_agent_behind_as_tool(self, example):
        from agents import Agent

        from examples.openai_agents_gate import run_sql

        sub = Agent(name="sub", model=example.scripted_model([]), tools=[run_sql])
        parent = Agent(
            name="a", model=example.scripted_model([]), tools=[sub.as_tool("sub", "sub")]
        )
        with pytest.raises(TypeError, match="agent tool"):
            example.gated_agent(parent, example.sql_scope_policy)

    # --- the script --------------------------------------------------------------------------

    def test_example_runs_end_to_end(self, example):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "tools the server ran: 2" in result.stdout
        assert result.stdout.count(REFUSED) == 2
        assert (
            "local MCP server:" in result.stdout
            and "hosted MCP, approval requests:" in result.stdout
        )
        assert result.stdout.count("approved") == 2 and "rejected:" in result.stdout
