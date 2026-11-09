"""Regression tests for the offline OpenAI Agents SDK + Recusal MCP example."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from recusal.claude_code import decide

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
        assert (
            _text(outputs["c1"])
            == "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        )

    def test_a_policy_that_writes_to_what_it_judged_does_not_change_what_the_server_runs(
        self, example
    ):
        """Containment of a buggy trusted policy: it judges a deep copy of the arguments,
        the server receives the dict itself. Before this guard, the write reached the
        server through the real Runner."""
        import copy

        seen = []

        def mutating(tool_name, args):
            seen.append(copy.deepcopy(args))
            findings = example.sql_scope_policy(tool_name, args)
            args["sql"] = DELETE  # an accidental write after the evidence was gathered
            return findings

        outputs, called = example.run_calls([("run_sql", {"sql": SELECT}, "c1")], policy=mutating)
        assert seen == [{"sql": SELECT}]
        assert called == [("db", "run_sql", {"sql": SELECT})]
        assert _text(outputs["c1"]) == f"recorded SQL: {SELECT}"

    def test_the_policy_observes_nested_arguments_detached_from_the_call(self, example):
        arguments = {"sql": SELECT, "meta": {"tags": ["read"], "limits": [1, 2]}}
        before = {"sql": SELECT, "meta": {"tags": ["read"], "limits": [1, 2]}}
        observed = []

        def mutating(tool_name, args):
            observed.append(args)
            args["meta"]["tags"].append("write")
            args["meta"]["limits"][0] = 999
            args["sql"] = DELETE
            return []

        assert example._refusal_for_call("run_sql", arguments, mutating) is None
        assert arguments == before, "the arguments are what the server receives; untouched"
        assert observed[0] is not arguments and observed[0]["meta"] is not arguments["meta"]
        assert observed[0]["sql"] == DELETE  # the policy changed its own copy only

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
            "reason": "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)",
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


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestNameDomainBinding:
    """Review 5, configuration gap: ``manifest_policy`` judges Claude Code's
    ``mcp__<server>__<tool>`` names; this gate hands policies the server's own tool name.
    Plugged together directly, a missing manifest let a call run. The pairing is refused at
    setup now, and ``claude_named`` binds the policy to the connection."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_openai_mcp_domain_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def manifest_for_db(tmp_path, tools=("run_sql",)):
        from recusal.mcp import build_manifest, manifest_policy, manifest_to_text

        path = tmp_path / "mcp-manifest.json"
        path.write_text(
            manifest_to_text(build_manifest({"db": [{"name": t} for t in tools]})),
            encoding="utf-8",
        )
        return manifest_policy(str(path))

    def test_negative_control_the_reviewers_pairing_is_refused_at_setup(self, example, tmp_path):
        """Before: manifest_policy(missing) on the local gate judged `run_sql`, saw no MCP name,
        raised no objection, and the server ran the call. Now the pairing never installs."""
        from recusal.mcp import manifest_policy

        missing = manifest_policy(str(tmp_path / "deliberately-missing.json"))
        with pytest.raises(ValueError, match="claude_named"):
            example.gate_mcp_server(example.InProcessMCPServer(), missing)
        with pytest.raises(ValueError, match="claude_named"):
            example.run_calls([("run_sql", {"sql": DELETE}, "c1")], policy=missing)
        assert example.CALLED == []
        # bound, the same missing manifest fails closed through the gate, and nothing runs
        outputs, called = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")], policy=example.claude_named("db", missing)
        )
        assert called == []
        assert "no usable MCP manifest" in _text(outputs["c1"])

    def test_a_bound_manifest_policy_pins_the_servers_tools(self, example, tmp_path):
        bound = example.claude_named("db", self.manifest_for_db(tmp_path, ("run_sql",)))
        outputs, called = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1"), ("echo", {"text": "hi"}, "c2")], policy=bound
        )
        assert called == [("db", "run_sql", {"sql": SELECT})]
        assert "Recusal refused `echo`" in _text(outputs["c2"])
        # the manifest digest reaches the audit record through the bound policy's identity
        from recusal.claude_code import decide

        decide("run_sql", {"sql": SELECT}, bound)
        decide(
            "run_sql", {"sql": SELECT}, bound
        )  # a ContextVar set inside the run loop stays there
        assert bound.get_control_identity()["manifest_sha256"].startswith("sha256:")
        bound.reset_control_identity()
        assert bound.get_control_identity()["manifest_sha256"] is None

    def test_the_hosted_gate_holds_the_same_rule(self, example, tmp_path):
        from recusal.mcp import manifest_policy

        with pytest.raises(ValueError, match="gate_hosted_mcp"):
            example.run_hosted(
                [("run_sql", {"sql": SELECT}, "r1")],
                policy=manifest_policy(str(tmp_path / "missing.json")),
            )
        bound = example.claude_named("db", self.manifest_for_db(tmp_path, ("run_sql",)))
        responses = example.run_hosted(
            [("run_sql", {"sql": SELECT}, "r1"), ("echo", {"text": "hi"}, "r2")], policy=bound
        )
        assert responses["r1"]["approve"] is True
        assert responses["r2"]["approve"] is False and "echo" in responses["r2"]["reason"]

    def test_a_policy_bound_to_another_server_is_refused(self, example):
        other = example.claude_named("other", example.sql_scope_policy)
        with pytest.raises(ValueError, match="bound to server 'other'"):
            example.gate_mcp_server(example.InProcessMCPServer(name="db"), other)
        assert example.gate_mcp_server(example.InProcessMCPServer(name="other"), other) is not None

    def test_a_composition_carrying_a_manifest_policy_is_in_its_domain(self, example, tmp_path):
        from examples.composed_policies import compose_policies

        composed = compose_policies(example.sql_scope_policy, self.manifest_for_db(tmp_path))
        assert composed.recusal_name_domain == "claude_code"
        with pytest.raises(ValueError, match="claude_named"):
            example.gate_mcp_server(example.InProcessMCPServer(), composed)
        bound = example.claude_named("db", composed)
        outputs, called = example.run_calls([("run_sql", {"sql": SELECT}, "c1")], policy=bound)
        assert called == [("db", "run_sql", {"sql": SELECT})]
        decide(
            "run_sql", {"sql": SELECT}, bound
        )  # a ContextVar set inside the run loop stays there
        assert bound.get_control_identity()["manifest_sha256"].startswith("sha256:")

    def test_raw_name_policies_and_plain_compositions_are_unaffected(self, example):
        from examples.composed_policies import compose_policies

        assert compose_policies(example.sql_scope_policy).recusal_name_domain is None
        outputs, called = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")], policy=compose_policies(example.sql_scope_policy)
        )
        assert called == [("db", "run_sql", {"sql": SELECT})]

    def test_the_combined_gated_agent_refuses_function_tools_that_share_a_name(self, example):
        """The combined entry point gated function tools one at a time, so a batch-only
        duplicate check would never see the pair; it checks the whole list first."""
        from examples.openai_agents_gate import echo, transforming_sql_tool

        sub = example.Agent(
            name="sub",
            model=example.scripted_model([("echo", {"text": "x"}, "s1")]),
            tools=[echo],
        )
        agent_tool = example.gated_agent(sub, example.sql_scope_policy).as_tool(
            tool_name="run_sql", tool_description="same name"
        )
        model = example.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
        agent = example.Agent(
            name="dup",
            model=model,
            tools=[agent_tool, transforming_sql_tool()],
            mcp_servers=[example.InProcessMCPServer()],
        )
        with pytest.raises(ValueError, match="share a name"):
            example.gated_agent(agent, example.sql_scope_policy)
        assert not example.is_gated(agent)
        assert example.CALLED == []

    def test_claude_named_needs_a_server_name(self, example):
        with pytest.raises(ValueError, match="name of the server"):
            example.claude_named("", example.sql_scope_policy)

    # --- review 6, P1: composing a bound policy keeps its server ----------------------------

    @staticmethod
    def noop(_name, _args):
        return []

    def test_negative_control_a_composition_of_a_bound_policy_is_refused_on_another_server(
        self, example, tmp_path
    ):
        """Before: compose_policies(claude_named("db", manifest), noop) carried no binding,
        installed on a server named `other`, and the Runner executed other/run_sql against
        the approval for db/run_sql. The composition is bound to db now."""
        from examples.composed_policies import compose_policies

        composed = compose_policies(
            example.claude_named("db", self.manifest_for_db(tmp_path)), self.noop
        )
        assert composed.recusal_bound_server == "db"
        assert composed.recusal_name_domain is None
        with pytest.raises(ValueError, match="bound to server 'db'"):
            example.gate_mcp_server(example.InProcessMCPServer(name="other"), composed)
        example.CALLED.clear()
        with pytest.raises(ValueError, match="bound to server 'db'"):
            example.run_calls(
                [("run_sql", {"sql": SELECT}, "c1")],
                policy=composed,
                server=example.InProcessMCPServer(name="other"),
            )
        assert example.CALLED == []

    def test_both_wrapper_orders_run_on_the_bound_server_and_refuse_any_other(
        self, example, tmp_path
    ):
        from examples.composed_policies import compose_policies

        shapes = {
            "compose(claude_named(manifest), noop)": compose_policies(
                example.claude_named("db", self.manifest_for_db(tmp_path)), self.noop
            ),
            "claude_named(compose(manifest, noop))": example.claude_named(
                "db", compose_policies(self.manifest_for_db(tmp_path), self.noop)
            ),
        }
        calls = [("run_sql", {"sql": SELECT}, "c1"), ("echo", {"text": "hi"}, "c2")]
        for label, policy in shapes.items():
            assert policy.recusal_bound_server == "db", label
            outputs, called = example.run_calls(
                calls, policy=policy, server=example.InProcessMCPServer(name="db")
            )
            assert called == [("db", "run_sql", {"sql": SELECT})], label
            assert "Recusal refused `echo`" in _text(outputs["c2"]), label
            # the same raw tool name on another connection is not this policy's to approve
            with pytest.raises(ValueError, match="bound to server 'db'"):
                example.run_calls(
                    calls[:1], policy=policy, server=example.InProcessMCPServer(name="other")
                )

    def test_two_connections_with_the_same_raw_tool_name_are_judged_by_their_own_policies(
        self, example, tmp_path
    ):
        from examples.composed_policies import compose_policies

        for_db = compose_policies(
            example.claude_named("db", self.manifest_for_db(tmp_path)), self.noop
        )
        for_other = example.claude_named("other", _refuse_deletes)
        db = example.gate_mcp_server(example.InProcessMCPServer(name="db"), for_db)
        other = example.gate_mcp_server(example.InProcessMCPServer(name="other"), for_other)
        with pytest.raises(ValueError, match="bound to server 'db'"):
            example.gate_mcp_server(example.InProcessMCPServer(name="other"), for_db)
        assert example.is_gated_mcp_server(db) and example.is_gated_mcp_server(other)

    def test_contradictory_bindings_are_refused_at_composition(self, example):
        from examples.composed_policies import compose_policies

        with pytest.raises(ValueError, match="different servers \\['db', 'other'\\]"):
            compose_policies(
                example.claude_named("db", self.noop), example.claude_named("other", self.noop)
            )

    def test_nested_compositions_and_kept_subclasses_forward_the_binding(self, example, tmp_path):
        from examples.composed_policies import ComposedPolicy, compose_policies

        bound = example.claude_named("db", self.manifest_for_db(tmp_path))
        nested = compose_policies(compose_policies(bound, self.noop), self.noop)
        assert nested.recusal_bound_server == "db"

        class Kept(ComposedPolicy):
            pass

        assert compose_policies(Kept(bound), self.noop).recusal_bound_server == "db"
        with pytest.raises(ValueError, match="different servers"):
            compose_policies(Kept(bound), example.claude_named("other", self.noop))

    def test_binding_the_same_server_twice_is_the_same_policy_and_never_prefixes_twice(
        self, example
    ):
        seen = []

        def raw(name, _args):
            seen.append(name)
            return []

        bound = example.claude_named("db", raw)
        assert example.claude_named("db", bound) is bound
        example.claude_named("db", bound)("run_sql", {})
        assert seen == ["mcp__db__run_sql"]
        with pytest.raises(ValueError, match="already bound to server 'db'"):
            example.claude_named("other", bound)

    def test_a_bound_policy_beside_an_unbound_manifest_policy_is_refused_at_composition(
        self, example, tmp_path
    ):
        from examples.composed_policies import compose_policies

        # two manifest policies are already two identity carriers, refused on that rule
        with pytest.raises(ValueError, match="one manifest_policy per gate"):
            compose_policies(
                example.claude_named("db", self.manifest_for_db(tmp_path)),
                self.manifest_for_db(tmp_path),
            )

        # an unbound Claude-name policy without an identity of its own beside a bound one
        def claude_named_check(_name, _args):
            return []

        claude_named_check.recusal_name_domain = "claude_code"
        with pytest.raises(ValueError, match="bind that policy with claude_named too"):
            compose_policies(
                example.claude_named("db", self.manifest_for_db(tmp_path)), claude_named_check
            )

    def test_an_unmet_naming_requirement_is_refused_even_on_the_bound_server(self, example):
        """compose_policies refuses the mixed shape, but a hand-built or subclassed policy can
        still present both attributes: the gates check the naming requirement first, so a
        matching binding never makes an unbound Claude-name check installable."""

        def both(_name, _args):
            return []

        both.recusal_bound_server = "db"
        both.recusal_name_domain = "claude_code"
        with pytest.raises(ValueError, match="claude_named"):
            example.gate_mcp_server(example.InProcessMCPServer(name="db"), both)
        with pytest.raises(ValueError, match="claude_named"):
            example.run_hosted([("run_sql", {"sql": SELECT}, "r1")], policy=both)

    def test_the_hosted_gate_honours_a_compositions_binding(self, example, tmp_path):
        from examples.composed_policies import compose_policies

        for_db = compose_policies(
            example.claude_named("db", self.manifest_for_db(tmp_path)), self.noop
        )
        responses = example.run_hosted(
            [("run_sql", {"sql": SELECT}, "r1"), ("echo", {"text": "hi"}, "r2")], policy=for_db
        )
        assert responses["r1"]["approve"] is True and responses["r2"]["approve"] is False
        for_other = compose_policies(example.claude_named("other", self.noop), self.noop)
        with pytest.raises(ValueError, match="bound to server 'other'"):
            example.run_hosted([("run_sql", {"sql": SELECT}, "r1")], policy=for_other)

    def test_a_bound_composition_keeps_the_digest_and_the_reset(self, example, tmp_path):
        from examples.composed_policies import compose_policies

        composed = compose_policies(
            example.claude_named("db", self.manifest_for_db(tmp_path)), self.noop
        )
        decide("run_sql", {"sql": SELECT}, composed)
        assert composed.get_control_identity()["manifest_sha256"].startswith("sha256:")
        composed.reset_control_identity()
        assert composed.get_control_identity()["manifest_sha256"] is None
