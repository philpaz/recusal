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


def _adapter():
    """The package module the gate's code lives in. Every test that calls this has loaded
    the example first, which puts the in-tree adapter on the path, so this is the code
    under review and not an installed copy."""
    import importlib
    from pathlib import Path

    module = importlib.import_module("recusal_openai_agents_mcp._gate")
    assert "adapters" in Path(module.__file__).parts, module.__file__
    return module


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
    for pin in ("openai-agents==0.24.0", "openai==3.49.0"):
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

    def test_a_refusal_is_an_mcp_error_result_and_an_allowed_call_is_not(self, example):
        """The gated ``call_tool`` answers a refused call with an MCP result flagged as an
        error, carrying the reason; an allowed call's result comes back from the server,
        unflagged. Read by the wire name, ``isError``, which is what a client sees."""
        import asyncio

        example.CALLED.clear()
        server = example.gate_mcp_server(example.InProcessMCPServer(), example.sql_scope_policy)
        refused = asyncio.run(server.call_tool("run_sql", {"sql": DELETE}))
        allowed = asyncio.run(server.call_tool("run_sql", {"sql": SELECT}))
        assert refused.model_dump(by_alias=True)["isError"] is True
        assert refused.content[0].text.startswith("Recusal refused `run_sql` [FAIL]")
        assert not allowed.model_dump(by_alias=True).get("isError")
        assert example.CALLED == [("db", "run_sql", {"sql": SELECT})]

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

        sent, reason = _adapter()._judged_call("run_sql", arguments, {}, mutating)
        assert reason is None
        assert sent == before, "what is sent to the server is what was proposed; untouched"
        assert arguments == before, "and so is the dict the SDK handed over"
        assert observed[0] is not sent and observed[0]["meta"] is not sent["meta"]
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


#: The input schema a server lists for ``purge``: defaults the server fills when the call
#: leaves a parameter out, and types it converts a value to.
PURGE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "default": "/"},
        "recursive": {"type": "boolean", "default": True},
        "dry_run": {"type": "boolean", "default": False},
    },
}


def _purge_server(gate, schema=None, name="files"):
    """An in-process server that lists one tool, ``purge``, under ``schema``, counts its
    listings and records what it is sent."""
    import copy

    from mcp.types import Tool

    listed = PURGE_SCHEMA if schema is None else schema

    class PurgeServer(gate.InProcessMCPServer):
        listings = 0

        async def list_tools(self, run_context=None, agent=None):
            self.listings += 1
            return [
                Tool(
                    name="purge",
                    description="Delete files under a path.",
                    inputSchema=copy.deepcopy(listed),
                )
            ]

    return PurgeServer(name=name, tools=("purge",))


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestWhatIsSentIsWhatWasJudged:
    """The 2026-11-21 review, F1. The gate judged the dict as sent and never read the tool's
    listed schema, and a server then ran values no policy judged: ``"false"`` converted to
    ``False``, ``"500"`` to ``500``, and a parameter left out filled from its default. The
    gate now writes every listed default the call leaves out into the arguments, which the
    policy judges and the server is sent, and refuses arguments that are not valid against
    the listed schema as they stand. ``TestTheWholeListedSchema`` holds the second review,
    which found the first fix read only a top-level property with a plain ``type``."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.mcp", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "recusal_openai_agents_mcp_sent_example", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    class Recording:
        def __init__(self, policy=None):
            self.policy, self.seen = policy or (lambda name, args: []), []

        def __call__(self, name, args):
            import copy

            self.seen.append((name, copy.deepcopy(args)))
            return self.policy(name, args)

    @staticmethod
    def dry_run_only(name, args):
        """An ordinary policy written against the tool's published parameters."""
        from recusal import Finding

        if name == "purge" and (args.get("recursive") is True or not args.get("dry_run")):
            return [Finding.fail("dry_run_only", severity="CRITICAL", message="dry runs only")]
        return []

    # --- defaults the server would fill are written out, judged and sent ----------------------

    def test_a_parameter_left_out_is_judged_and_sent_at_its_listed_default(self, example):
        policy = self.Recording()
        outputs, called = example.run_calls(
            [("purge", {}, "c1"), ("purge", {"path": "/data", "dry_run": True}, "c2")],
            policy=policy,
            server=_purge_server(example),
        )
        first = {"path": "/", "recursive": True, "dry_run": False}
        second = {"path": "/data", "dry_run": True, "recursive": True}
        assert policy.seen == [("purge", first), ("purge", second)]
        assert called == [("files", "purge", first), ("files", "purge", second)]

    def test_a_policy_refuses_the_default_it_now_sees(self, example):
        policy = self.Recording(self.dry_run_only)
        outputs, called = example.run_calls(
            [("purge", {}, "c1")], policy=policy, server=_purge_server(example)
        )
        assert called == []
        assert _text(outputs["c1"]) == "Recusal refused `purge` [FAIL]: dry runs only"

    def test_negative_control_ungated_the_server_is_sent_the_call_without_its_defaults(
        self, example
    ):
        _, called = example.run_calls(
            [("purge", {}, "c1")], gated=False, server=_purge_server(example)
        )
        assert called == [("files", "purge", {})]

    def test_a_default_outside_the_checked_types_is_still_written_out(self, example):
        schema = {
            "type": "object",
            "properties": {
                "label": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                "tags": {"type": "array", "default": ["a"]},
                "note": {"type": "string"},
            },
        }
        policy = self.Recording()
        _, called = example.run_calls(
            [("purge", {}, "c1")], policy=policy, server=_purge_server(example, schema)
        )
        # ``note`` has no listed default: it is the server's to fill, and is not seen
        assert policy.seen == [("purge", {"label": None, "tags": ["a"]})]
        assert called == [("files", "purge", {"label": None, "tags": ["a"]})]

    def test_a_written_default_is_a_copy_the_policy_and_the_server_cannot_share(self, example):
        schema = {"type": "object", "properties": {"tags": {"type": "array", "default": ["a"]}}}
        server = example.gate_mcp_server(_purge_server(example, schema), self.spoiling)
        example.CALLED.clear()
        for _ in range(2):
            asyncio.run(server.call_tool("purge", {}))
        assert example.CALLED == [("files", "purge", {"tags": ["a"]})] * 2

    @staticmethod
    def spoiling(name, args):
        args["tags"].append("WRITTEN BY THE POLICY")
        return []

    def test_a_listed_default_its_own_schema_does_not_allow_is_judged_and_not_sent(self, example):
        """The server runs with such a default when the parameter is left out, and would
        reject it if it were sent: it is what the policy judges, and it is not sent."""
        schema = {
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean", "default": "no"},
                "path": {"type": "string", "default": "/"},
            },
        }
        policy = self.Recording()
        _, called = example.run_calls(
            [("purge", {}, "c1"), ("purge", {"dry_run": True}, "c2")],
            policy=policy,
            server=_purge_server(example, schema),
        )
        assert policy.seen == [
            ("purge", {"dry_run": "no", "path": "/"}),
            ("purge", {"dry_run": True, "path": "/"}),
        ]
        assert called == [
            ("files", "purge", {"path": "/"}),
            ("files", "purge", {"dry_run": True, "path": "/"}),
        ]

    def test_a_listed_schema_that_is_not_json_data_is_refused(self, example):
        schema = {"type": "object", "properties": {"ratio": {"default": float("nan")}}}
        policy = self.Recording()
        assert _adapter()._judged_call("purge", {}, schema, policy) == (
            None,
            "Recusal refused `purge`: the server's listed schema for this tool is not JSON data",
        )
        assert policy.seen == []

    # --- an argument must already be of the type the schema declares --------------------------

    TYPED = {
        "type": "object",
        "properties": {
            "dry_run": {"type": "boolean"},
            "depth": {"type": "integer"},
            "ratio": {"type": ["number", "null"]},
            "path": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "options": {"type": "object", "properties": {"force": {"type": "boolean"}}},
            "mode": {"enum": ["soft", "hard", 1]},
            "kind": {"const": "files"},
            "label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "ref": {"$ref": "#/$defs/Ref"},
        },
        "$defs": {"Ref": {"type": "object", "properties": {"deep": {"type": "boolean"}}}},
    }

    @pytest.mark.parametrize(
        "arguments, detail",
        [
            (
                {"dry_run": "false"},
                "argument `dry_run` is of JSON type string where the server's schema declares "
                "boolean",
            ),
            (
                {"dry_run": 0},
                "argument `dry_run` is of JSON type integer where the server's schema "
                "declares boolean",
            ),
            (
                {"depth": "MARKER"},
                "argument `depth` is of JSON type string where the server's schema declares "
                "integer",
            ),
            (
                {"depth": 5.0},
                "argument `depth` is of JSON type number where the server's schema declares "
                "integer",
            ),
            (
                {"depth": True},
                "argument `depth` is of JSON type boolean where the server's schema declares "
                "integer",
            ),
            (
                {"ratio": "MARKER"},
                "argument `ratio` is of JSON type string where the server's schema declares "
                "number or null",
            ),
            (
                {"ratio": False},
                "argument `ratio` is of JSON type boolean where the server's schema declares "
                "number or null",
            ),
            (
                {"path": None},
                "argument `path` is of JSON type null where the server's schema declares string",
            ),
            (
                {"path": ["MARKER"]},
                "argument `path` is of JSON type array where the server's schema declares string",
            ),
            (
                {"tags": {"MARKER": 1}},
                "argument `tags` is of JSON type object where the server's schema declares array",
            ),
            (
                {"tags": ["a", 1]},
                "argument `tags[1]` is of JSON type integer where the server's schema "
                "declares string",
            ),
            (
                {"options": {"force": "MARKER"}},
                "argument `options.force` is of JSON type string where the server's schema "
                "declares boolean",
            ),
            (
                {"label": 5},
                "argument `label` is of JSON type integer where the server's schema declares "
                "string or null",
            ),
            (
                {"ref": {"deep": "MARKER"}},
                "argument `ref.deep` is of JSON type string where the server's schema "
                "declares boolean",
            ),
            (
                {"ref": "MARKER"},
                "argument `ref` is of JSON type string where the server's schema declares object",
            ),
            (
                {"mode": "MARKER"},
                "argument `mode` is not one of the values the server's schema lists",
            ),
            ({"mode": 1.0}, "argument `mode` is not one of the values the server's schema lists"),
            ({"mode": True}, "argument `mode` is not one of the values the server's schema lists"),
            ({"kind": "MARKER"}, "argument `kind` is not the value the server's schema fixes"),
        ],
        ids=[
            "string-for-boolean",
            "integer-for-boolean",
            "string-for-integer",
            "float-for-integer",
            "boolean-for-integer",
            "string-for-number",
            "boolean-for-number",
            "null-for-string",
            "array-for-string",
            "object-for-array",
            "item-of-another-type",
            "member-of-another-type",
            "none-of-the-alternatives",
            "member-behind-a-ref",
            "string-for-an-object-behind-a-ref",
            "not-in-enum",
            "enum-by-type",
            "enum-boolean-is-not-a-number",
            "not-the-const",
        ],
    )
    def test_an_argument_of_another_type_is_refused_before_the_policy_runs(
        self, example, arguments, detail
    ):
        policy = self.Recording()
        outputs, called = example.run_calls(
            [("purge", arguments, "c1")], policy=policy, server=_purge_server(example, self.TYPED)
        )
        assert called == [] and policy.seen == []
        # only a value of another JSON type is told to send the declared type
        advice = (
            "the server would not run with the value the policy judged. Send the declared "
            "JSON type."
            if "is of JSON type" in detail
            else "the gate sends only arguments that are valid against it as they stand"
        )
        assert _text(outputs["c1"]) == f"Recusal refused `purge`: {detail}; {advice}"
        assert "MARKER" not in _text(outputs["c1"]), "the reason carries no argument text"

    def test_negative_control_ungated_the_server_is_sent_the_string_it_would_convert(self, example):
        _, called = example.run_calls(
            [("purge", {"dry_run": "false"}, "c1")],
            gated=False,
            server=_purge_server(example, self.TYPED),
        )
        assert called == [("files", "purge", {"dry_run": "false"})]

    def test_arguments_of_the_declared_types_are_judged_and_sent_as_they_are(self, example):
        arguments = {
            "dry_run": True,
            "depth": 3,
            "ratio": 1,  # an integer is a number
            "path": "/data",
            "tags": ["a", "b"],
            "options": {"force": False, "unlisted": "kept"},
            "mode": 1,
            "kind": "files",
            "label": None,
            "ref": {"deep": True, "unlisted": "kept"},
            "unlisted": "kept",  # an argument the schema does not name, and does not forbid
        }
        policy = self.Recording()
        _, called = example.run_calls(
            [("purge", arguments, "c1"), ("purge", {"ratio": None, "depth": -1}, "c2")],
            policy=policy,
            server=_purge_server(example, self.TYPED),
        )
        assert policy.seen == [("purge", arguments), ("purge", {"ratio": None, "depth": -1})]
        assert called == [
            ("files", "purge", arguments),
            ("files", "purge", {"ratio": None, "depth": -1}),
        ]

    @pytest.mark.parametrize("schema", [{}, {"type": "object"}], ids=["empty", "bare"])
    def test_a_schema_that_constrains_nothing_judges_and_sends_the_call_as_it_is(self, schema):
        pytest.importorskip("agents.mcp", reason="openai-agents is not installed")
        policy = self.Recording()
        sent, reason = _adapter()._judged_call("purge", {"dry_run": "false"}, schema, policy)
        assert (sent, reason) == ({"dry_run": "false"}, None)
        assert policy.seen == [("purge", {"dry_run": "false"})]

    def test_a_schema_without_a_type_is_read_all_the_same(self, example):
        schema = {"properties": {"dry_run": {"type": "boolean", "default": False}}}
        policy = self.Recording()
        sent, reason = _adapter()._judged_call("purge", {"dry_run": "false"}, schema, policy)
        assert sent is None and "argument `dry_run` is of JSON type string" in reason
        assert _adapter()._judged_call("purge", {}, schema, policy)[0] == {"dry_run": False}

    # --- the schema is the one the server listed ----------------------------------------------

    def test_the_listing_is_returned_as_the_server_gave_it_and_read_once(self, example):
        server = example.gate_mcp_server(_purge_server(example), self.Recording())
        tools = asyncio.run(server.list_tools())
        assert [tool.name for tool in tools] == ["purge"]
        assert server.listings == 1
        # a caller that rewrites the listing it was handed does not rewrite the gate's copy
        schema = getattr(tools[0], "input_schema", None) or tools[0].inputSchema
        schema["properties"]["dry_run"]["type"] = "string"
        example.CALLED.clear()
        result = asyncio.run(server.call_tool("purge", {"dry_run": "false"}))
        assert result.model_dump(by_alias=True)["isError"] is True and example.CALLED == []
        assert server.listings == 1, "the schema came from the listing the gate had seen"

    def test_a_call_before_any_listing_makes_the_gate_list_the_tools_once(self, example):
        server = example.gate_mcp_server(_purge_server(example), self.Recording())
        example.CALLED.clear()
        for _ in range(2):
            asyncio.run(server.call_tool("purge", {}))
        assert server.listings == 1
        assert (
            example.CALLED
            == [("files", "purge", {"path": "/", "recursive": True, "dry_run": False})] * 2
        )

    def test_the_completed_arguments_are_sent_beside_request_metadata(self, example):
        """The SDK passes ``meta`` when a resolver produced one; the gate passes it on,
        unjudged, beside the completed arguments."""
        server = _purge_server(example)
        inner, received = server.call_tool, []

        async def recorder(tool_name, arguments, meta=None):
            received.append((arguments, meta))
            return await inner(tool_name, arguments)

        server.call_tool = recorder
        example.gate_mcp_server(server, self.Recording())
        asyncio.run(server.call_tool("purge", {"path": "/data"}, meta={"trace": "t1"}))
        asyncio.run(server.call_tool("purge", {"path": "/data"}))
        completed = {"path": "/data", "recursive": True, "dry_run": False}
        assert received == [(completed, {"trace": "t1"}), (completed, None)]

    def test_a_tool_the_server_does_not_list_is_refused(self, example):
        policy = self.Recording()
        server = example.gate_mcp_server(_purge_server(example), policy)
        example.CALLED.clear()
        result = asyncio.run(server.call_tool("drop_all", {}))
        assert result.model_dump(by_alias=True)["isError"] is True
        assert result.content[0].text == (
            "Recusal refused `drop_all`: the server lists no tool of this name, or its tools "
            "could not be listed, so the gate cannot read the schema its arguments would be "
            "checked against"
        )
        assert example.CALLED == [] and policy.seen == []

    def test_a_server_that_cannot_list_its_tools_refuses_the_call(self, example):
        class Unlistable(example.InProcessMCPServer):
            async def list_tools(self, run_context=None, agent=None):
                raise RuntimeError("not connected")

        policy = self.Recording()
        server = example.gate_mcp_server(Unlistable(), policy)
        example.CALLED.clear()
        result = asyncio.run(server.call_tool("run_sql", {"sql": SELECT}))
        assert result.model_dump(by_alias=True)["isError"] is True
        assert "its tools could not be listed" in result.content[0].text
        assert example.CALLED == [] and policy.seen == []

    # --- a sub-agent with MCP behind as_tool ----------------------------------------------------

    def test_a_gated_sub_agent_with_a_server_is_accepted_behind_as_tool(self, example):
        """F2: the parent asked the function-tool package whether the sub-agent was gated,
        and that package says no to any agent with ``mcp_servers``."""
        from agents import Agent

        policy = self.Recording(example.sql_scope_policy)
        sub = example.gated_agent(
            Agent(
                name="sub",
                model=example.scripted_model([("run_sql", {"sql": DELETE}, "s1")]),
                mcp_servers=[example.InProcessMCPServer()],
            ),
            policy,
        )
        assert example.is_gated(sub)
        parent_model = example.scripted_model([("ask_sub", {"input": "tidy the log"}, "p1")])
        parent = Agent(name="parent", model=parent_model, tools=[sub.as_tool("ask_sub", "ask")])
        assert not example.is_gated(parent)
        gated = example.gated_agent(parent, policy)
        assert example.is_gated(gated)
        example.CALLED.clear()
        asyncio.run(example.run_agent(gated))
        # the parent's call to the sub-agent and the sub-agent's server call are both judged
        assert [name for name, _ in policy.seen] == ["ask_sub", "run_sql"]
        assert example.CALLED == [], "the sub-agent's DELETE never reached its server"

    def test_a_gated_sub_agent_with_a_hosted_tool_is_accepted_behind_as_tool(self, example):
        from agents import Agent, HostedMCPTool

        hosted = HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "db",
                "server_url": "https://example.invalid/mcp",
                "require_approval": "never",
            }
        )
        model = example.scripted_model([])
        sub = example.gated_agent(
            Agent(name="sub", model=model, tools=[hosted]), example.sql_scope_policy
        )
        parent = Agent(name="parent", model=model, tools=[sub.as_tool("ask_sub", "ask")])
        assert example.is_gated(example.gated_agent(parent, example.sql_scope_policy))

    def test_a_sub_agent_whose_server_is_not_gated_is_still_refused(self, example):
        from agents import Agent

        model = example.scripted_model([])
        sub = Agent(name="sub", model=model, mcp_servers=[example.InProcessMCPServer()])
        parent = Agent(name="parent", model=model, tools=[sub.as_tool("ask_sub", "ask")])
        with pytest.raises(TypeError, match="agent tool 'ask_sub': its agent's own tool calls"):
            example.gated_agent(parent, example.sql_scope_policy)
        # gated, then a server appended afterwards: is_gated says so, for sub and parent
        sub = example.gated_agent(sub, example.sql_scope_policy)
        parent = example.gated_agent(
            Agent(name="parent", model=model, tools=[sub.as_tool("ask_sub", "ask")]),
            example.sql_scope_policy,
        )
        assert example.is_gated(parent)
        sub.mcp_servers.append(example.InProcessMCPServer(name="late"))
        assert not example.is_gated(sub) and not example.is_gated(parent)

    # --- the gated copy, and what no gate sees --------------------------------------------------

    def test_the_gated_copy_has_its_own_handoffs_list(self, example):
        from agents import Agent

        model = example.scripted_model([])
        agent = Agent(name="a", model=model, mcp_servers=[example.InProcessMCPServer()])
        gated = example.gated_agent(agent, example.sql_scope_policy)
        assert gated.handoffs is not agent.handoffs
        assert gated.mcp_servers is not agent.mcp_servers and gated.tools is not agent.tools
        agent.handoffs.append(Agent(name="other", model=model))
        assert gated.handoffs == [] and example.is_gated(gated)

    def test_an_agent_with_a_prompt_is_refused(self, example):
        from agents import Agent

        model = example.scripted_model([])
        agent = Agent(name="a", model=model, prompt={"id": "pmpt_stored"})
        with pytest.raises(ValueError, match="cannot gate agent 'a': it has a prompt"):
            example.gated_agent(agent, example.sql_scope_policy)
        gated = example.gated_agent(Agent(name="a", model=model), example.sql_scope_policy)
        assert example.is_gated(gated)
        gated.prompt = {"id": "pmpt_stored"}
        assert not example.is_gated(gated)

    def test_names_shared_across_namespaces_are_refused_for_the_true_reason(self, example):
        from agents import Agent, tool_namespace

        from examples.openai_agents_gate import run_sql

        crm = tool_namespace(name="crm", description="crm tools", tools=[run_sql])
        erp = tool_namespace(name="erp", description="erp tools", tools=[run_sql])
        agent = Agent(name="a", model=example.scripted_model([]), tools=[*crm, *erp])
        with pytest.raises(ValueError, match="its function tools in different namespaces share"):
            example.gated_agent(agent, example.sql_scope_policy)

    # --- hosted MCP: the argument text is all the gate holds -----------------------------------

    @pytest.mark.parametrize("arguments", ['{"limit": NaN}', '{"limit": 1e999}'])
    def test_a_hosted_request_with_a_non_finite_number_is_rejected(self, example, arguments):
        policy = self.Recording()
        responses = example.run_hosted([("run_sql", arguments, "r1")], policy=policy)
        assert responses["r1"]["approve"] is False
        assert responses["r1"]["reason"] == (
            "Recusal refused `run_sql`: arguments hold a non-finite number, which is not JSON data"
        )
        assert policy.seen == []


#: A member with a listed default, the shape a nested model or an inline object publishes.
OPTIONS = {
    "type": "object",
    "properties": {
        "recursive": {"type": "boolean", "default": True},
        "dry_run": {"type": "boolean", "default": False},
    },
}
FILLED = {"recursive": True, "dry_run": False}

_SCHEMA = "the server's listed schema for this tool"
_AS_THEY_STAND = "the gate sends only arguments that are valid against it as they stand"
_SEND_THE_TYPE = (
    "the server would not run with the value the policy judged. Send the declared JSON type."
)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestTheWholeListedSchema:
    """The second review of 2026-11-21. The first fix read a top-level property with a
    plain ``type`` and completed top-level defaults only, so the first review's exploit
    still ran for ``anyOf`` (every ``Optional``), ``$ref`` (every nested model),
    ``additionalProperties`` (every ``dict``), ``prefixItems`` (every tuple) and nested
    defaults. The arguments must now be valid against the whole listed schema as they
    stand, every default it lists at any depth is written out, and what the gate cannot
    check or complete is refused. ``TestARealServerRunsWhatWasJudged`` shows the same
    against a server built with the MCP Python SDK."""

    example = TestWhatIsSentIsWhatWasJudged.example
    Recording = TestWhatIsSentIsWhatWasJudged.Recording

    def judged(self, arguments, schema):
        """``(what the policy saw or None, what is sent or None, the reason or None)``."""
        policy = self.Recording()
        sent, reason = _adapter()._judged_call("purge", arguments, schema, policy)
        return (policy.seen[0][1] if policy.seen else None), sent, reason

    # --- defaults at every depth -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "schema, arguments, completed",
        [
            ({"properties": {"opts": OPTIONS}}, {"opts": {}}, {"opts": FILLED}),
            (
                {"properties": {"opts": OPTIONS}},
                {"opts": {"dry_run": True}},
                {"opts": {"dry_run": True, "recursive": True}},
            ),
            (
                {"properties": {"targets": {"type": "array", "items": OPTIONS}}},
                {"targets": [{}, {"recursive": False}]},
                {"targets": [FILLED, {"recursive": False, "dry_run": False}]},
            ),
            (
                {
                    "properties": {"opts": {"$ref": "#/$defs/Options"}},
                    "$defs": {"Options": OPTIONS},
                },
                {"opts": {}},
                {"opts": FILLED},
            ),
            (
                {
                    "properties": {"opts": {"$ref": "#/$defs/Options", "default": {}}},
                    "$defs": {"Options": OPTIONS},
                },
                {},
                {"opts": FILLED},
            ),
            (
                {
                    "properties": {"mode": {"$ref": "#/$defs/Mode"}},
                    "$defs": {"Mode": {"default": 1}},
                },
                {},
                {"mode": 1},
            ),
            (
                {"properties": {"by": {"type": "object", "additionalProperties": OPTIONS}}},
                {"by": {"a": {}}},
                {"by": {"a": FILLED}},
            ),
            (
                {
                    "properties": {
                        "t": {"type": "array", "prefixItems": [{"type": "integer"}, OPTIONS]}
                    }
                },
                {"t": [1, {}]},
                {"t": [1, FILLED]},
            ),
            (
                {
                    "properties": {
                        "opts": {"allOf": [OPTIONS, {"properties": {"depth": {"default": 3}}}]}
                    }
                },
                {"opts": {}},
                {"opts": {**FILLED, "depth": 3}},
            ),
            (
                {"properties": {"opts": {"anyOf": [OPTIONS, {"type": "null"}]}}},
                {"opts": {}},
                {"opts": FILLED},
            ),
            (
                {"properties": {"opts": {"anyOf": [OPTIONS, {"type": "null"}]}}},
                {"opts": None},
                {"opts": None},
            ),
            (
                {
                    "properties": {
                        "shape": {
                            "oneOf": [
                                {
                                    "properties": {"kind": {"const": "a"}, "x": {"default": 1}},
                                    "required": ["kind"],
                                },
                                {
                                    "properties": {"kind": {"const": "b"}, "y": {"default": 2}},
                                    "required": ["kind"],
                                },
                            ]
                        }
                    }
                },
                {"shape": {"kind": "b"}},
                {"shape": {"kind": "b", "y": 2}},
            ),
            (
                {
                    "properties": {"node": {"$ref": "#/$defs/Node"}},
                    "$defs": {
                        "Node": {
                            "type": "object",
                            "properties": {
                                "leaf": {"type": "boolean", "default": True},
                                "child": {"$ref": "#/$defs/Node"},
                            },
                        }
                    },
                },
                {"node": {"child": {"child": {}}}},
                {"node": {"leaf": True, "child": {"leaf": True, "child": {"leaf": True}}}},
            ),
        ],
        ids=[
            "nested-object",
            "nested-object-in-part",
            "items-of-a-list",
            "behind-a-ref",
            "a-written-default-is-completed-too",
            "a-default-the-ref-lists",
            "additional-properties",
            "prefix-items",
            "all-of",
            "the-one-alternative-that-validates",
            "the-other-alternative",
            "one-of",
            "a-schema-that-refers-to-itself",
        ],
    )
    def test_every_listed_default_is_written_out_judged_and_sent(
        self, example, schema, arguments, completed
    ):
        import copy

        before = copy.deepcopy(arguments)
        assert self.judged(arguments, schema) == (completed, completed, None)
        assert arguments == before, "the caller's own dict is not written to"

    def test_the_policy_and_the_server_share_no_object(self, example):
        """A policy that writes to what it was handed changes its copy, not the call, and
        not the schema's own default for the next call."""
        schema = {"properties": {"opts": {"default": {"tags": ["a"]}}}}
        server = example.gate_mcp_server(_purge_server(example, schema), self.spoiling)
        example.CALLED.clear()
        asyncio.run(server.call_tool("purge", {}))
        assert example.CALLED == [("files", "purge", {"opts": {"tags": ["a"]}})]
        # nor does a server that writes to what it received reach the listed default
        example.CALLED[0][2]["opts"]["tags"].append("WRITTEN BY THE SERVER")
        asyncio.run(server.call_tool("purge", {}))
        assert example.CALLED[1] == ("files", "purge", {"opts": {"tags": ["a"]}})

    @staticmethod
    def spoiling(name, args):
        args["opts"]["tags"].append("WRITTEN BY THE POLICY")
        args["extra"] = True
        return []

    def test_alternatives_that_would_write_different_defaults_are_refused(self, example):
        other = {
            "type": "object",
            "properties": {"recursive": {"type": "boolean", "default": False}},
        }
        schema = {"properties": {"opts": {"anyOf": [OPTIONS, other]}}}
        assert self.judged({"opts": {}}, schema) == (
            None,
            None,
            "Recusal refused `purge`: argument `opts` matches more than one of the alternatives "
            "the server's schema lists, and they list different defaults, so the gate cannot "
            "tell what the server would run with",
        )
        # nothing is left out, so nothing is ambiguous
        given = {"opts": {"recursive": True, "dry_run": True}}
        assert self.judged(given, schema) == (given, given, None)
        # two alternatives that write the same defaults are one answer
        same = {"properties": {"opts": {"anyOf": [OPTIONS, {**OPTIONS, "title": "again"}]}}}
        assert self.judged({"opts": {}}, same) == ({"opts": FILLED}, {"opts": FILLED}, None)
        # alternatives that differ only in a default that cannot be sent differ all the same
        unsendable = [
            {"properties": {"q": {"type": "string", "default": None}}},
            {"properties": {"q": {"type": "string", "default": 5}}},
        ]
        schema = {"properties": {"opts": {"type": "object", "anyOf": unsendable}}}
        assert "they list different defaults" in self.judged({"opts": {}}, schema)[2]

    def test_a_default_that_cannot_be_sent_is_judged_at_any_depth_and_not_sent(self, example):
        schema = {
            "properties": {
                "opts": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "default": None},
                        "limit": {"type": "integer", "default": 10},
                    },
                }
            }
        }
        assert self.judged({"opts": {}}, schema) == (
            {"opts": {"q": None, "limit": 10}},
            {"opts": {"limit": 10}},
            None,
        )
        assert self.judged({"opts": {"q": "x"}}, schema) == (
            {"opts": {"q": "x", "limit": 10}},
            {"opts": {"q": "x", "limit": 10}},
            None,
        )

    def test_an_item_left_out_that_has_a_listed_default_is_refused(self, example):
        schema = {"properties": {"t": {"prefixItems": [{"type": "integer"}, {"default": 1}]}}}
        assert self.judged({"t": [5]}, schema) == (
            None,
            None,
            "Recusal refused `purge`: argument `t` leaves out an item the server's schema lists "
            "a default for, which the gate cannot write in place",
        )
        assert self.judged({"t": [5, 2]}, schema) == ({"t": [5, 2]}, {"t": [5, 2]}, None)

    # --- strict validity against the whole schema ------------------------------------------------

    @pytest.mark.parametrize(
        "schema, arguments, detail",
        [
            (
                {"properties": {"dry_run": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}}},
                {"dry_run": "false"},
                "argument `dry_run` is of JSON type string where the server's schema declares "
                f"boolean or null; {_SEND_THE_TYPE}",
            ),
            (
                {"properties": {"dry_run": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}}},
                {"dry_run": 1},
                "argument `dry_run` is of JSON type integer where the server's schema declares "
                f"boolean or null; {_SEND_THE_TYPE}",
            ),
            (
                {
                    "properties": {"opts": {"$ref": "#/$defs/Options"}},
                    "$defs": {"Options": OPTIONS},
                },
                {"opts": {"recursive": "false"}},
                "argument `opts.recursive` is of JSON type string where the server's schema "
                f"declares boolean; {_SEND_THE_TYPE}",
            ),
            (
                {
                    "properties": {"opts": {"$ref": "#/$defs/Options"}},
                    "$defs": {"Options": OPTIONS},
                },
                {"opts": '{"recursive": true}'},
                "argument `opts` is of JSON type string where the server's schema declares "
                f"object; {_SEND_THE_TYPE}",
            ),
            (
                {"properties": {"flags": {"additionalProperties": {"type": "boolean"}}}},
                {"flags": {"MARKER": "true"}},
                "argument `flags.*` is of JSON type string where the server's schema declares "
                f"boolean; {_SEND_THE_TYPE}",
            ),
            (
                {"properties": {"t": {"prefixItems": [{"type": "integer"}, {"type": "boolean"}]}}},
                {"t": [1, "true"]},
                "argument `t[1]` is of JSON type string where the server's schema declares "
                f"boolean; {_SEND_THE_TYPE}",
            ),
            (
                {"properties": {"items": {"type": "array", "items": OPTIONS}}},
                {"items": [{}, {"recursive": "no"}]},
                "argument `items[1].recursive` is of JSON type string where the server's schema "
                f"declares boolean; {_SEND_THE_TYPE}",
            ),
            (
                {"properties": {"n": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}},
                {"n": 5.0},
                "argument `n` is of JSON type number where the server's schema declares integer "
                f"or null; {_SEND_THE_TYPE}",
            ),
            (
                {"properties": {"level": {"enum": ["low", "high"], "type": "string"}}},
                {"level": "MARKER"},
                "argument `level` is not one of the values the server's schema lists; "
                f"{_AS_THEY_STAND}",
            ),
            (
                {"properties": {"path": {"type": "string"}}, "required": ["path"]},
                {},
                f"the arguments leave out `path`, which the server's schema requires; "
                f"{_AS_THEY_STAND}",
            ),
            (
                {"properties": {"opts": {"required": ["recursive", "dry_run"]}}},
                {"opts": {"recursive": True}},
                "argument `opts` leaves out `dry_run`, which the server's schema requires; "
                f"{_AS_THEY_STAND}",
            ),
            (
                {"properties": {"path": {"type": "string"}}, "additionalProperties": False},
                {"MARKER": 1},
                f"the arguments hold a member the server's schema does not allow; {_AS_THEY_STAND}",
            ),
            (
                {"properties": {"n": {"type": "integer", "minimum": 10}}},
                {"n": 5},
                f"argument `n` does not meet `minimum` in the server's schema; {_AS_THEY_STAND}",
            ),
            (
                {
                    "properties": {
                        "n": {"anyOf": [{"type": "string", "maxLength": 2}, {"type": "null"}]}
                    }
                },
                {"n": "MARKER"},
                "argument `n` matches none of the alternatives the server's schema lists there; "
                f"{_AS_THEY_STAND}",
            ),
            (
                {"type": "string"},
                {"path": "MARKER"},
                "the arguments are of JSON type object where the server's schema declares "
                f"string; {_SEND_THE_TYPE}",
            ),
            (
                {
                    "properties": {
                        "n": {"anyOf": [{"type": "number", "minimum": 10}, {"type": "null"}]}
                    }
                },
                {"n": 5},
                "argument `n` matches none of the alternatives the server's schema lists there; "
                f"{_AS_THEY_STAND}",
            ),
            (
                {"properties": {"a": {"type": "integer", "default": 1}}, "required": ["a"]},
                {},
                f"the arguments leave out `a`, which the server's schema requires; "
                f"{_AS_THEY_STAND}",
            ),
        ],
        ids=[
            "string-for-an-optional-boolean",
            "integer-for-an-optional-boolean",
            "member-of-a-nested-model",
            "json-text-for-an-object",
            "member-the-schema-does-not-name",
            "item-of-a-tuple",
            "member-of-a-list-item",
            "float-for-an-optional-integer",
            "not-in-enum",
            "required-left-out",
            "nested-required-left-out",
            "member-not-allowed",
            "another-constraint",
            "no-alternative-and-the-type-is-listed",
            "the-arguments-themselves",
            "an-integer-is-a-number-so-the-type-is-listed",
            "required-is-checked-before-any-default-is-written",
        ],
    )
    def test_arguments_not_valid_against_the_schema_as_they_stand_are_refused(
        self, example, schema, arguments, detail
    ):
        seen, sent, reason = self.judged(arguments, schema)
        assert seen is None and sent is None, "refused before the policy ran"
        assert reason == f"Recusal refused `purge`: {detail}"
        assert "MARKER" not in reason, "the reason carries no text of the model's"

    def test_a_number_that_is_an_integer_or_a_float_is_judged_as_sent(self, example):
        schema = {
            "properties": {
                "ratio": {"type": "number"},
                "either": {"type": ["integer", "number"]},
                "count": {"type": "integer"},
            }
        }
        arguments = {"ratio": 1, "either": 5.0, "count": 5}
        assert self.judged(arguments, schema) == (arguments, arguments, None)
        assert self.judged({"ratio": True}, schema)[2] is not None, "a boolean is not a number"

    def test_a_format_is_not_asserted(self, example):
        schema = {"properties": {"mail": {"type": "string", "format": "email"}}}
        assert self.judged({"mail": "not an address"}, schema)[2] is None

    def test_a_written_default_must_leave_the_arguments_valid(self, example):
        schema = {
            "properties": {"a": {"type": "integer", "default": 1}, "b": {"type": "integer"}},
            "maxProperties": 1,
        }
        assert self.judged({"b": 2}, schema) == (
            None,
            None,
            "Recusal refused `purge`: the arguments do not meet `maxProperties` in the server's "
            f"schema; {_AS_THEY_STAND}",
        )

    # --- the older drafts a server may declare ---------------------------------------------------

    def test_a_draft_07_schema_is_validated_and_completed_by_its_own_keywords(self, example):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "opts": {"$ref": "#/definitions/Options"},
                "t": {
                    "type": "array",
                    "items": [{"type": "integer"}, OPTIONS],
                    "additionalItems": OPTIONS,
                },
            },
            "definitions": {"Options": OPTIONS},
        }
        completed = {"opts": FILLED, "t": [1, FILLED, FILLED]}
        assert self.judged({"opts": {}, "t": [1, {}, {}]}, schema) == (completed, completed, None)
        reason = self.judged({"opts": {}, "t": ["1"]}, schema)[2]
        assert "argument `t[0]` is of JSON type string" in reason

    def test_a_2019_09_schema_is_validated_and_completed(self, example):
        schema = {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "properties": {"t": {"items": OPTIONS}},
        }
        assert self.judged({"t": [{}]}, schema) == ({"t": [FILLED]}, {"t": [FILLED]}, None)

    # --- a schema the gate cannot check or complete against is refused --------------------------

    @pytest.mark.parametrize(
        "schema, detail",
        [
            (None, "is not a JSON object"),
            (["object"], "is not a JSON object"),
            (True, "is not a JSON object"),
            ({"type": "object", "properties": ["dry_run"]}, "is not a valid JSON Schema"),
            ({"type": 5}, "is not a valid JSON Schema"),
            ({"properties": {"a": {"default": object()}}}, "is not JSON data"),
            (
                {"$schema": "http://json-schema.org/draft-04/schema#"},
                "declares a JSON Schema draft the gate does not read (it reads draft-07, "
                "2019-09 and 2020-12)",
            ),
            (
                {"$schema": "https://example.invalid/dialect"},
                "declares a JSON Schema draft the gate does not read (it reads draft-07, "
                "2019-09 and 2020-12)",
            ),
            (
                {"$schema": 7},
                "declares a JSON Schema draft the gate does not read (it reads draft-07, "
                "2019-09 and 2020-12)",
            ),
            (
                {"properties": {"a": {"$ref": "#/$defs/Missing"}}},
                "has a `$ref` that does not resolve inside the schema itself, so the gate cannot "
                "check the arguments against it",
            ),
            (
                {"properties": {"a": {"$ref": "https://example.invalid/schema.json"}}},
                "has a `$ref` that does not resolve inside the schema itself, so the gate cannot "
                "check the arguments against it",
            ),
            (
                {"properties": {"a": {"$ref": "other.json#/$defs/A"}}, "$defs": {"A": {}}},
                "has a `$ref` that does not resolve inside the schema itself, so the gate cannot "
                "check the arguments against it",
            ),
            (
                {"properties": {"a": {"$ref": "x/$defs/A"}}, "$defs": {"A": {}}},
                "has a `$ref` that does not resolve inside the schema itself, so the gate cannot "
                "check the arguments against it",
            ),
            (
                {"properties": {"a": {"$ref": "#name"}}, "$defs": {"A": {"$anchor": "name"}}},
                "has a `$ref` that does not resolve inside the schema itself, so the gate cannot "
                "check the arguments against it",
            ),
            (
                {"$defs": {"unused": {"$ref": "#/$defs/Missing"}}},
                "has a `$ref` that does not resolve inside the schema itself, so the gate cannot "
                "check the arguments against it",
            ),
            (
                {"properties": {"a": {"$dynamicRef": "#node"}}},
                "uses `$dynamicRef`, which the gate does not resolve",
            ),
            (
                {
                    "$schema": "https://json-schema.org/draft/2019-09/schema",
                    "properties": {"a": {"$recursiveRef": "#"}},
                },
                "uses `$recursiveRef`, which the gate does not resolve",
            ),
            (
                {"properties": {"a": {"$id": "https://example.invalid/a", "type": "string"}}},
                "embeds a schema with an `$id` of its own, which the gate does not resolve "
                "references against",
            ),
            (
                {"if": {"required": ["a"]}, "then": {"properties": {"b": {"default": 1}}}},
                "lists a default under `then`, where the gate cannot tell whether the server "
                "would apply it",
            ),
            (
                {"patternProperties": {"^x": {"default": 1}}},
                "lists a default under `patternProperties`, where the gate cannot tell whether "
                "the server would apply it",
            ),
            (
                {"dependentSchemas": {"a": {"properties": {"b": {"default": 1}}}}},
                "lists a default under `dependentSchemas`, where the gate cannot tell whether "
                "the server would apply it",
            ),
            ({"$ref": "#"}, "refers to itself without end"),
            (
                {
                    "properties": {"a": {"$ref": "#/$defs/A"}},
                    "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}},
                },
                "refers to itself without end",
            ),
        ],
        ids=[
            "none",
            "a-list",
            "a-boolean",
            "properties-not-a-mapping",
            "type-not-a-name",
            "not-json-data",
            "draft-04",
            "an-unknown-dialect",
            "a-dialect-that-is-not-a-string",
            "a-ref-to-nowhere",
            "a-remote-ref",
            "a-ref-into-another-document",
            "a-relative-ref",
            "a-ref-to-an-anchor",
            "a-ref-to-nowhere-nothing-uses",
            "a-dynamic-ref",
            "a-recursive-ref",
            "an-embedded-id",
            "a-default-under-then",
            "a-default-under-pattern-properties",
            "a-default-under-dependent-schemas",
            "a-ref-to-itself",
            "refs-in-a-loop",
        ],
    )
    def test_a_schema_the_gate_cannot_check_or_complete_against_is_refused(
        self, example, schema, detail
    ):
        seen, sent, reason = self.judged({"a": "MARKER"}, schema)
        assert seen is None and sent is None
        assert reason == f"Recusal refused `purge`: {_SCHEMA} {detail}"

    def test_defaults_that_never_settle_are_refused(self, example):
        """A default that is an object which lists the same default again, without end."""
        again = {"default": {}, "$ref": "#/$defs/A"}
        schema = {"properties": {"a": again}, "$defs": {"A": {"properties": {"a": again}}}}
        assert self.judged({}, schema) == (
            None,
            None,
            f"Recusal refused `purge`: {_SCHEMA} lists defaults that never settle",
        )

    def test_a_remote_ref_is_never_fetched(self, example, monkeypatch):
        """Refused from the text of the schema, before any validator could ask for it."""
        import urllib.request

        asked: list = []

        def fetched(request, *args, **kwargs):
            asked.append(getattr(request, "full_url", request))
            raise OSError("no network in a test")

        monkeypatch.setattr(urllib.request, "urlopen", fetched)
        schema = {"properties": {"a": {"$ref": "https://example.invalid/schema.json"}}}
        assert "does not resolve inside the schema itself" in self.judged({"a": 1}, schema)[2]
        # and the validator itself holds an empty registry, so it has nowhere to ask
        import importlib

        listed = importlib.import_module("recusal_openai_agents_mcp._schema").Listed({})
        with pytest.raises(Exception, match="Unresolvable"):
            list(listed._prepared().descend(1, schema["properties"]["a"]))
        assert asked == [], "the validator asked the network for a schema"

    def test_what_the_walk_does_not_follow_is_still_validated(self, example):
        """A keyword the walk does not follow is refused only when a default sits under it."""
        schema = {
            "if": {"properties": {"kind": {"const": "file"}}, "required": ["kind"]},
            "then": {"required": ["path"]},
            "patternProperties": {"^x_": {"type": "integer"}},
        }
        given = {"kind": "file", "path": "/data", "x_n": 1}
        assert self.judged(given, schema) == (given, given, None)
        assert "leave out `path`" in self.judged({"kind": "file"}, schema)[2]
        assert "argument `*` is of JSON type string" in self.judged({"x_n": "1"}, schema)[2]

    # --- the most recent listing is the one that counts -----------------------------------------

    @staticmethod
    def changing_server(example, listings):
        """A server whose successive listings are ``listings``: each a map of tool name to
        input schema; the last one repeats."""
        from mcp.types import Tool

        class Changing(example.InProcessMCPServer):
            listed = 0

            async def list_tools(self, run_context=None, agent=None):
                now = listings[min(self.listed, len(listings) - 1)]
                self.listed += 1
                return [Tool(name=n, description=n, inputSchema=dict(s)) for n, s in now.items()]

        return Changing(name="files", tools=("purge", "legacy"))

    def test_a_later_listing_replaces_the_schema_of_an_earlier_one(self, example):
        """First listing wins was a mutation no test caught."""
        first = {"properties": {"recursive": {"type": "boolean", "default": False}}}
        second = {
            "properties": {
                "recursive": {"type": "boolean", "default": True},
                "force": {"type": "boolean", "default": True},
            }
        }
        policy = self.Recording()
        server = example.gate_mcp_server(
            self.changing_server(example, [{"purge": first}, {"purge": second}]), policy
        )
        example.CALLED.clear()
        asyncio.run(server.list_tools())
        asyncio.run(server.call_tool("purge", {}))
        asyncio.run(server.list_tools())
        asyncio.run(server.call_tool("purge", {}))
        assert [args for _, args in policy.seen] == [
            {"recursive": False},
            {"recursive": True, "force": True},
        ]
        assert [args for _, _, args in example.CALLED] == [args for _, args in policy.seen]

    def test_a_tool_the_server_withdrew_is_refused(self, example):
        policy = self.Recording()
        server = example.gate_mcp_server(
            self.changing_server(example, [{"purge": {}, "legacy": {}}, {"purge": {}}]), policy
        )
        example.CALLED.clear()
        asyncio.run(server.list_tools())
        assert (
            not asyncio.run(server.call_tool("legacy", {})).model_dump(by_alias=True).get("isError")
        )
        asyncio.run(server.list_tools())
        result = asyncio.run(server.call_tool("legacy", {}))
        assert result.model_dump(by_alias=True)["isError"] is True
        assert result.content[0].text.startswith(
            "Recusal refused `legacy`: the server lists no tool of this name"
        )
        assert example.CALLED == [("files", "legacy", {})], "only the call made while it was listed"
        assert [name for name, _ in policy.seen] == ["legacy"]

    def test_a_call_to_a_tool_not_in_the_latest_listing_lists_once_for_that_call(self, example):
        server = self.changing_server(example, [{"purge": {}}])
        gated = example.gate_mcp_server(server, self.Recording())
        asyncio.run(gated.list_tools())
        assert server.listed == 1
        for expected in (2, 3):
            assert asyncio.run(gated.call_tool("legacy", {})).model_dump(by_alias=True)["isError"]
            assert server.listed == expected
        asyncio.run(gated.call_tool("purge", {}))
        assert server.listed == 3, "a listed tool is not listed for again"

    # --- arguments nested too deeply, and model settings ----------------------------------------

    def test_arguments_nested_past_the_depth_limit_are_refused_before_the_schema_is_read(
        self, example
    ):
        def nested(depth):
            value: list = []
            for _ in range(depth - 1):
                value = [value]
            return {"meta": value}

        assert self.judged(nested(64), {}) == (nested(64), nested(64), None)
        assert self.judged(nested(65), {}) == (
            None,
            None,
            "Recusal refused `purge`: arguments are nested too deeply",
        )

    @pytest.mark.parametrize("field", ["extra_body", "extra_args"])
    def test_model_settings_that_carry_tools_are_refused(self, example, field):
        from agents import Agent, ModelSettings

        settings = ModelSettings(**{field: {"tools": [{"type": "web_search"}]}})
        agent = Agent(
            name="a",
            model=example.scripted_model([]),
            mcp_servers=[example.InProcessMCPServer()],
            model_settings=settings,
        )
        with pytest.raises(
            ValueError, match=f"cannot gate agent 'a': its model_settings.{field} sets `tools`"
        ):
            example.gated_agent(agent, example.sql_scope_policy)
        gated = example.gated_agent(
            Agent(name="a", model=example.scripted_model([])), example.sql_scope_policy
        )
        assert example.is_gated(gated)
        gated.model_settings = settings
        assert not example.is_gated(gated)


def _in_order(calls):
    """Calls in one order whatever order they ran in and whatever order their keys are in."""
    import json

    return sorted(calls, key=lambda call: json.dumps(call, sort_keys=True))


#: A server built with the MCP Python SDK's own server class, run over stdio. It writes
#: what each tool function received, after the SDK's own validation, to a file.
STDIO_SERVER = """
import json
import os
import threading
from typing import Literal, Optional, Union

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

app = MCPServer("files")
LOG = os.environ["RECUSAL_TEST_SERVER_LOG"]
WRITING = threading.Lock()  # the calls of one turn run side by side


def plain(value):
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return value


def record(tool, **received):
    with WRITING, open(LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps([tool, plain(received)]) + "\\n")
    return "ran " + tool


class Options(BaseModel):
    recursive: bool = True
    dry_run: bool = False


@app.tool()
def purge_opt(
    path: Optional[str] = "/", recursive: Optional[bool] = True, dry_run: Optional[bool] = None
) -> str:
    return record("purge_opt", path=path, recursive=recursive, dry_run=dry_run)


@app.tool()
def purge_nested(path: str, opts: Options) -> str:
    return record("purge_nested", path=path, opts=opts)


@app.tool()
def purge_nested_default(path: str, opts: Options = Options()) -> str:
    return record("purge_nested_default", path=path, opts=opts)


@app.tool()
def boxes(items: list[Options]) -> str:
    return record("boxes", items=items)


@app.tool()
def set_flags(flags: dict[str, bool]) -> str:
    return record("set_flags", flags=flags)


@app.tool()
def set_limits(limits: dict[str, int]) -> str:
    return record("set_limits", limits=limits)


@app.tool()
def pair(t: tuple[int, bool]) -> str:
    return record("pair", t=t)


@app.tool()
def search(q: str = None, limit: int = None, tags: list[str] = None) -> str:
    return record("search", q=q, limit=limit, tags=tags)


@app.tool()
def run_mode(level: Literal["low", "high"] = "low") -> str:
    return record("run_mode", level=level)


@app.tool()
def count(n: int = 1, ratio: float = 1) -> str:
    return record("count", n=n, ratio=ratio)


@app.tool()
def wipe(prefix) -> str:
    return record("wipe", prefix=prefix)


@app.tool()
def either(v: Union[dict, str]) -> str:
    return record("either", v=v)


@app.tool()
def pay(amount: float) -> str:
    return record("pay", amount=amount)


@app.tool()
def tagged(src: str, tags: list[str] = Field(default_factory=lambda: ["all"])) -> str:
    return record("tagged", src=src, tags=tags)


def swap_v1(path: str, recursive: bool = False) -> str:
    return record("swap", version=1, path=path, recursive=recursive)


def swap_v2(path: str, recursive: bool = True, force: bool = True) -> str:
    return record("swap", version=2, path=path, recursive=recursive, force=force)


def legacy(x: int = 1) -> str:
    return record("legacy", x=x)


app.add_tool(swap_v1, name="swap")
app.add_tool(legacy, name="legacy")


@app.tool()
def reconfigure() -> str:
    app.remove_tool("swap")
    app.add_tool(swap_v2, name="swap")
    app.remove_tool("legacy")
    return "reconfigured"


if __name__ == "__main__":
    app.run("stdio")
"""


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestARealServerRunsWhatWasJudged:
    """The second review's probes, kept as tests: a server built with the MCP Python SDK,
    reached over stdio, with the parameter shapes Pydantic publishes. For every call the
    server runs, what its function received equals what the policy judged."""

    example = TestWhatIsSentIsWhatWasJudged.example
    Recording = TestWhatIsSentIsWhatWasJudged.Recording

    @pytest.fixture
    def stdio(self, example, tmp_path):
        pytest.importorskip("mcp.server.mcpserver", reason="the MCP SDK's server is not installed")
        import json
        import os
        from types import SimpleNamespace

        from agents import Agent, RunConfig, Runner
        from agents.mcp import MCPServerStdio
        from agents.testing import ScriptedModel, assistant_message, function_call

        script = tmp_path / "stdio_server.py"
        script.write_text(STDIO_SERVER, encoding="utf-8")
        server_log = tmp_path / "server.jsonl"
        env = {"RECUSAL_TEST_SERVER_LOG": str(server_log)}
        if "SYSTEMROOT" in os.environ:
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

        def run(turns, policy=None, *, gated=True, direct=(), schemas=()):
            """Script ``turns`` (each a list of ``(tool, arguments)``) through an agent
            whose one server is the stdio server; then make the ``direct`` calls on the
            gated server object."""
            policy = policy or self.Recording()
            script_turns, ids = [], iter(range(1, 1000))
            for turn in turns:
                script_turns.append(
                    [function_call(n, json.dumps(a), call_id=f"c{next(ids)}") for n, a in turn]
                )
            model = ScriptedModel([*script_turns, [assistant_message("done")]])
            found = SimpleNamespace(wire=[], direct=[], schemas={}, judged=policy.seen)

            async def go():
                params = {"command": sys.executable, "args": [str(script)], "env": env}
                async with MCPServerStdio(params=params, client_session_timeout_seconds=60) as srv:
                    send = srv.session.call_tool

                    async def spy(name, arguments=None, *args, **kwargs):
                        found.wire.append((name, json.loads(json.dumps(arguments))))
                        return await send(name, arguments, *args, **kwargs)

                    srv.session.call_tool = spy
                    for tool in await srv.list_tools() if schemas else ():
                        if tool.name in schemas:
                            schema = getattr(tool, "input_schema", None) or tool.inputSchema
                            found.schemas[tool.name] = schema["properties"]
                    agent = Agent(name="assistant", model=model, mcp_servers=[srv])
                    if gated:
                        agent = example.gated_agent(agent, policy)
                    await Runner.run(agent, "go", run_config=RunConfig(tracing_disabled=True))
                    for name, arguments in direct:
                        result = await srv.call_tool(name, arguments)
                        found.direct.append(result.content[0].text)

            asyncio.run(asyncio.wait_for(go(), 180))
            found.outputs = {k: _text(v) for k, v in example.tool_outputs(model).items()}
            lines = (
                server_log.read_text(encoding="utf-8").splitlines() if server_log.exists() else []
            )
            found.ran = [tuple(json.loads(line)) for line in lines]
            return found

        return run

    def test_what_the_sdk_publishes_for_the_shapes_under_test(self, stdio):
        """The premise: these are the schema shapes the first fix did not read."""
        found = stdio([], schemas=("purge_opt", "purge_nested", "set_flags", "pair", "search"))
        assert found.schemas["purge_opt"]["dry_run"] == {
            "anyOf": [{"type": "boolean"}, {"type": "null"}],
            "default": None,
            "title": "Dry Run",
        }
        assert found.schemas["purge_nested"]["opts"] == {"$ref": "#/$defs/Options"}
        assert found.schemas["set_flags"]["flags"]["additionalProperties"] == {"type": "boolean"}
        assert found.schemas["pair"]["t"]["prefixItems"] == [
            {"type": "integer"},
            {"type": "boolean"},
        ]
        assert found.schemas["search"]["q"] == {"default": None, "title": "Q", "type": "string"}

    def test_negative_control_ungated_the_server_converts_and_fills_what_no_policy_saw(self, stdio):
        found = stdio(
            [
                [
                    ("purge_opt", {"path": "/data", "dry_run": "false"}),
                    ("purge_nested", {"path": "/d", "opts": {}}),
                    ("purge_nested", {"path": "/d", "opts": '{"dry_run": true}'}),
                    ("set_flags", {"flags": {"force": "true"}}),
                    ("pair", {"t": ["7", 1]}),
                ]
            ],
            gated=False,
        )
        assert _in_order(found.ran) == _in_order(
            [
                ("purge_opt", {"path": "/data", "recursive": True, "dry_run": False}),
                ("purge_nested", {"path": "/d", "opts": {"recursive": True, "dry_run": False}}),
                ("purge_nested", {"path": "/d", "opts": {"recursive": True, "dry_run": True}}),
                ("set_flags", {"flags": {"force": True}}),
                ("pair", {"t": [7, True]}),
            ]
        )

    def test_a_value_the_server_would_convert_is_refused_and_never_reaches_it(self, stdio):
        calls = [
            ("purge_opt", {"path": "/data", "dry_run": "false"}),
            ("purge_opt", {"path": "/d", "recursive": "true"}),
            ("purge_opt", {"path": "/d", "recursive": 1}),
            ("purge_nested", {"path": "/d", "opts": {"recursive": "false", "dry_run": "true"}}),
            ("purge_nested", {"path": "/d", "opts": '{"recursive": true}'}),
            ("boxes", {"items": [{}, {"recursive": "no"}]}),
            ("set_flags", {"flags": {"force": "true"}}),
            ("set_limits", {"limits": {"max": "500"}}),
            ("pair", {"t": [1, "true"]}),
            ("pair", {"t": ["7", 1]}),
            ("count", {"n": 2.0}),
            ("run_mode", {"level": "HIGH"}),
        ]
        found = stdio([calls])
        assert found.ran == [] and found.wire == [] and found.judged == []
        expected = {
            "c1": "argument `dry_run` is of JSON type string where the server's schema declares "
            "boolean or null",
            "c2": "argument `recursive` is of JSON type string where the server's schema "
            "declares boolean or null",
            "c3": "argument `recursive` is of JSON type integer where the server's schema "
            "declares boolean or null",
            "c4": "argument `opts.dry_run` is of JSON type string where the server's schema "
            "declares boolean",
            "c5": "argument `opts` is of JSON type string where the server's schema declares "
            "object",
            "c6": "argument `items[1].recursive` is of JSON type string where the server's "
            "schema declares boolean",
            "c7": "argument `flags.*` is of JSON type string where the server's schema declares "
            "boolean",
            "c8": "argument `limits.*` is of JSON type string where the server's schema "
            "declares integer",
            "c9": "argument `t[1]` is of JSON type string where the server's schema declares "
            "boolean",
            "c10": "argument `t[0]` is of JSON type string where the server's schema declares "
            "integer",
            "c11": "argument `n` is of JSON type number where the server's schema declares integer",
        }
        for call_id, detail in expected.items():
            tool = calls[int(call_id[1:]) - 1][0]
            assert found.outputs[call_id] == f"Recusal refused `{tool}`: {detail}; {_SEND_THE_TYPE}"
        assert found.outputs["c12"] == (
            "Recusal refused `run_mode`: argument `level` is not one of the values the "
            f"server's schema lists; {_AS_THEY_STAND}"
        )

    def test_what_the_server_runs_is_what_the_policy_judged(self, stdio):
        calls = [
            ("purge_opt", {}),
            ("purge_opt", {"path": None, "recursive": None, "dry_run": None}),
            ("purge_nested", {"path": "/d", "opts": {}}),
            ("purge_nested", {"path": "/d", "opts": {"dry_run": True}}),
            ("purge_nested_default", {"path": "/d"}),
            ("purge_nested_default", {"path": "/d", "opts": {"dry_run": True}}),
            ("boxes", {"items": [{}, {"recursive": False}]}),
            ("set_flags", {"flags": {"force": True}}),
            ("set_limits", {"limits": {"max": 500}}),
            ("pair", {"t": [7, True]}),
            ("run_mode", {}),
            ("count", {"n": 3, "ratio": 2}),
        ]
        found = stdio([calls])
        judged = [
            ("purge_opt", {"path": "/", "recursive": True, "dry_run": None}),
            ("purge_opt", {"path": None, "recursive": None, "dry_run": None}),
            ("purge_nested", {"path": "/d", "opts": {"recursive": True, "dry_run": False}}),
            ("purge_nested", {"path": "/d", "opts": {"recursive": True, "dry_run": True}}),
            ("purge_nested_default", {"path": "/d", "opts": {"recursive": True, "dry_run": False}}),
            ("purge_nested_default", {"path": "/d", "opts": {"recursive": True, "dry_run": True}}),
            (
                "boxes",
                {
                    "items": [
                        {"recursive": True, "dry_run": False},
                        {"recursive": False, "dry_run": False},
                    ]
                },
            ),
            ("set_flags", {"flags": {"force": True}}),
            ("set_limits", {"limits": {"max": 500}}),
            ("pair", {"t": [7, True]}),
            ("run_mode", {"level": "low"}),
            ("count", {"n": 3, "ratio": 2}),
        ]
        assert _in_order(found.judged) == _in_order(judged)
        assert _in_order(found.wire) == _in_order(judged), "sent what was judged"
        assert _in_order(found.ran) == _in_order(judged), "and ran what was judged"

    def test_a_policy_refuses_the_nested_default_it_now_sees(self, stdio):
        from recusal import Finding

        def dry_runs_only(name, args):
            opts = args.get("opts", {})
            if opts.get("recursive") is True or not opts.get("dry_run"):
                return [Finding.fail("dry_run_only", severity="CRITICAL", message="dry runs only")]
            return []

        found = stdio(
            [
                [
                    ("purge_nested", {"path": "/d", "opts": {}}),
                    ("purge_nested", {"path": "/d", "opts": {"recursive": False, "dry_run": True}}),
                ]
            ],
            self.Recording(dry_runs_only),
        )
        assert found.outputs["c1"] == "Recusal refused `purge_nested` [FAIL]: dry runs only"
        assert found.ran == [
            ("purge_nested", {"path": "/d", "opts": {"recursive": False, "dry_run": True}})
        ]

    def test_a_default_the_server_would_reject_if_sent_is_judged_and_not_sent(self, stdio):
        """``q: str = None``: Pydantic publishes ``{"default": null, "type": "string"}``. The
        first fix wrote the ``null``, then refused its own written default, so every call
        that left the parameter out failed. The server runs with ``None``: that is judged,
        and it is not sent."""
        found = stdio(
            [
                [
                    ("search", {}),
                    ("search", {"q": "x"}),
                    ("search", {"q": "x", "limit": 3, "tags": ["a"]}),
                ]
            ]
        )
        judged = [
            ("search", {"q": None, "limit": None, "tags": None}),
            ("search", {"q": "x", "limit": None, "tags": None}),
            ("search", {"q": "x", "limit": 3, "tags": ["a"]}),
        ]
        assert _in_order(found.judged) == _in_order(judged)
        assert _in_order(found.ran) == _in_order(judged), "the server ran with None"
        assert _in_order(found.wire) == _in_order(
            [
                ("search", {}),
                ("search", {"q": "x"}),
                ("search", {"q": "x", "limit": 3, "tags": ["a"]}),
            ]
        )

    def test_what_a_server_does_beyond_its_listed_schema_is_not_seen(self, stdio):
        """The README's list of what the gate does not see, kept true by a test. The MCP
        Python SDK's server lists ``type: string`` for a parameter without an annotation
        and then parses the string; it parses JSON text for a parameter whose schema allows
        a string beside an object; a float parameter rounds an integer above 2**53; and a
        ``default_factory`` default is in no schema."""
        found = stdio(
            [
                [
                    ("wipe", {"prefix": "null"}),
                    ("either", {"v": '{"a": 1}'}),
                    ("pay", {"amount": 9007199254740993}),
                    ("tagged", {"src": "a"}),
                ]
            ],
            schemas=("wipe", "tagged"),
        )
        assert found.schemas["wipe"] == {"prefix": {"title": "prefix", "type": "string"}}
        assert "default" not in found.schemas["tagged"]["tags"]
        sent = [
            ("wipe", {"prefix": "null"}),
            ("either", {"v": '{"a": 1}'}),
            ("pay", {"amount": 9007199254740993}),
            ("tagged", {"src": "a"}),
        ]
        assert _in_order(found.judged) == _in_order(found.wire) == _in_order(sent)
        assert _in_order(found.ran) == _in_order(
            [
                ("wipe", {"prefix": None}),
                ("either", {"v": {"a": 1}}),
                ("pay", {"amount": 9007199254740992.0}),
                ("tagged", {"src": "a", "tags": ["all"]}),
            ]
        )

    def test_the_most_recent_listing_governs_and_a_withdrawn_tool_is_refused(self, stdio):
        """The server replaces ``swap`` and withdraws ``legacy`` in the first turn; the SDK
        lists again for the second."""
        found = stdio(
            [
                [("swap", {"path": "/x"}), ("legacy", {})],
                [("reconfigure", {})],
                [("swap", {"path": "/x"})],
            ],
            direct=[("legacy", {}), ("swap", {"path": "/y"})],
        )
        assert _in_order(found.judged[:2]) == _in_order(
            [("swap", {"path": "/x", "recursive": False}), ("legacy", {"x": 1})]
        )
        assert found.judged[2:] == [
            ("reconfigure", {}),
            ("swap", {"path": "/x", "recursive": True, "force": True}),
            ("swap", {"path": "/y", "recursive": True, "force": True}),
        ]
        assert found.ran[2:] == [
            ("swap", {"version": 2, "path": "/x", "recursive": True, "force": True}),
            ("swap", {"version": 2, "path": "/y", "recursive": True, "force": True}),
        ]
        assert found.direct[0].startswith(
            "Recusal refused `legacy`: the server lists no tool of this name"
        )
        assert [name for name, _ in found.wire].count("legacy") == 1, "not sent for again"
