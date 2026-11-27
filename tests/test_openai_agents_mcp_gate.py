"""Regression tests for the offline OpenAI Agents SDK + Recusal MCP example."""

import asyncio
import contextlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Union

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
_TOO_COSTLY = f"the arguments are too costly to check against {_SCHEMA}"
#: the reason when the ceiling on one call stopped the check, at the default ceiling
_OVER_THE_CEILING = (
    f"{_TOO_COSTLY}: the check was stopped at 200,000 steps, the ceiling the gate sets on "
    "the steps of one call. Send fewer values in one call. Where calls of this size are "
    "expected, the application can raise the ceiling: `step_ceiling=` on `gate_mcp_server`, "
    "`gated_agent` or `audited_agent`. Raising it does not help a call that costs more than "
    "10,000 steps and 1,000 more for every value in its arguments, which the gate refuses "
    "under any ceiling."
)
#: the reason when the arguments are over the ceiling on their size, at the default one
_TOO_LARGE = (
    "the arguments are more than 1,000,000 characters as JSON text, the ceiling the gate "
    "sets on the size of one call's arguments so that the time a call takes does not grow "
    "with what is sent; the gate has not read them. Send less in one call. Where calls of "
    "this size are expected, the application can raise the ceiling: `size_ceiling=` on "
    "`gate_mcp_server`, `gated_agent` or `audited_agent`."
)


@contextlib.contextmanager
def _digit_limit(limit):
    """Run under another limit on the digits of an integer read from text (0: none), as
    a server's interpreter may; an interpreter that has no such limit runs as it is."""
    if not hasattr(sys, "set_int_max_str_digits"):
        yield
        return
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(limit)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(before)


def _over_each_value(steps=10_000, more=1_000):
    """The reason when the steps allowed for the values of the arguments stopped it."""
    return (
        f"{_TOO_COSTLY}: the check costs more than the {steps:,} steps and {more:,} more for "
        "every value in the arguments that the gate allows a call. That is what the schema "
        "costs for each value here, which raising the gate's ceiling on a call "
        "(`step_ceiling=`) does not change."
    )


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

    def judged(self, arguments, schema, **ceilings):
        """``(what the policy saw or None, what is sent or None, the reason or None)``."""
        policy = self.Recording()
        sent, reason = _adapter()._judged_call("purge", arguments, schema, policy, **ceilings)
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

    def test_an_object_the_schema_offers_two_forms_for_is_refused(self, example):
        """Refused once only where the two forms would write different defaults; a server
        picks the form by a rule of its own whatever the object holds (the fourth
        review), so the object is refused whatever it holds."""
        other = {
            "type": "object",
            "properties": {"recursive": {"type": "boolean", "default": False}},
        }
        schema = {"properties": {"opts": {"anyOf": [OPTIONS, other]}}}
        refused = (None, None, f"Recusal refused `purge`: argument `opts` {_TWO_OBJECT_FORMS}")
        assert self.judged({"opts": {}}, schema) == refused
        # with nothing left out too
        assert self.judged({"opts": {"recursive": True, "dry_run": True}}, schema) == refused
        # and where the two forms are word for word the same
        same = {"properties": {"opts": {"anyOf": [OPTIONS, {**OPTIONS, "title": "again"}]}}}
        assert self.judged({"opts": {}}, same) == refused
        # one object form beside what is not an object is one form
        one = {"properties": {"opts": {"anyOf": [OPTIONS, {"type": "null"}, {"type": "string"}]}}}
        assert self.judged({"opts": {}}, one) == ({"opts": FILLED}, {"opts": FILLED}, None)
        assert self.judged({"opts": None}, one) == ({"opts": None}, {"opts": None}, None)

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
            "dependentRequired": {"x_n": ["kind"]},
        }
        given = {"kind": "file", "path": "/data", "x_n": 1}
        assert self.judged(given, schema) == (given, given, None)
        assert "leave out `path`" in self.judged({"kind": "file"}, schema)[2]
        assert "do not meet `dependentRequired`" in self.judged({"x_n": 1}, schema)[2]

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
import dataclasses
import datetime
import json
import os
import threading
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from mcp.server.mcpserver import MCPServer
from pydantic import AliasPath, AnyUrl, BaseModel, Field, StringConstraints

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


@app.tool()
def scoped(scope: Optional[str] = "tmp", confirm: bool = False) -> str:
    return record("scoped", scope=scope, confirm=confirm)


@app.tool()
def scoped_required(scope: Optional[str]) -> str:
    return record("scoped_required", scope=scope)


@app.tool()
def limited(limit: Union[int, str, None] = 10) -> str:
    return record("limited", limit=limit)


@app.tool()
def moded(mode: Optional[str] = "null") -> str:
    return record("moded", mode=mode)


@app.tool()
def since(day: datetime.date, at: Optional[datetime.datetime] = None) -> str:
    return record("since", day=day.isoformat(), at=None if at is None else at.isoformat())


@app.tool()
def located(q: Optional[Path] = None) -> str:
    return record("located", q=None if q is None else q.as_posix())


@app.tool()
def named(
    name: Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True)] = "X ",
) -> str:
    return record("named", name=name)


class ByPath(BaseModel):
    path: str
    recursive: bool = True


class ByPattern(BaseModel):
    path: str
    pattern: str = "*"
    dry_run: bool = False


@app.tool()
def remove(target: Union[ByPath, ByPattern]) -> str:
    return record("remove", target=target)


class Keep(BaseModel):
    kind: Literal["keep"] = "keep"
    days: int = 30


class Drop(BaseModel):
    kind: Literal["drop"] = "drop"
    force: bool = False


@app.tool()
def retain(rule: Union[Keep, Drop]) -> str:
    return record("retain", rule=rule)


@app.tool()
def after(when: Union[int, datetime.date], strict: Union[bool, datetime.date] = False) -> str:
    return record("after", when=repr(when), strict=repr(strict))


class Aliased(BaseModel):
    model_config = {"extra": "forbid"}
    force: bool = Field(True, validation_alias="forceIt")


@app.tool()
def aliased(opts: Aliased = Aliased()) -> str:
    return record("aliased", opts=opts)


@app.tool()
def keyed(limits: dict[Annotated[str, Field(pattern="^[a-z]+$")], int]) -> str:
    return record("keyed", limits=limits)


@app.tool()
def set_quotas(quotas: dict[int, int]) -> str:
    return record("set_quotas", quotas=quotas)


@app.tool()
def set_marks(marks: dict[bool, str]) -> str:
    return record("set_marks", marks=marks)


@dataclasses.dataclass
class Job:
    path: str = "/tmp/scratch"
    dry_run: bool = dataclasses.field(default=False, init=False)


@app.tool()
def clean(job: Job) -> str:
    return record("clean", path=job.path, dry_run=job.dry_run)


class Target(BaseModel):
    dry_run: bool = Field(True, validation_alias=AliasPath("options", "dry_run"))
    path: str = "/tmp/scratch"


@app.tool()
def remove_at(target: Target) -> str:
    return record("remove_at", target=target)


@app.tool()
def expire(
    when: Annotated[Union[datetime.date, float], Field(union_mode="left_to_right")],
) -> str:
    return record("expire", when=str(when))


@app.tool()
def find(ident: Union[int, str], name: str = "") -> str:
    return record("find", ident=ident, name=name)


def told(value):
    return type(value).__name__ + " " + str(value)


@app.tool()
def stamp(at: Union[datetime.datetime, Decimal]) -> str:
    return record("stamp", at=told(at))


@app.tool()
def forced(force: Union[bool, Decimal] = False) -> str:
    return record("forced", force=told(force))


@app.tool()
def levelled(n: Union[Literal[1, 2], bool]) -> str:
    return record("levelled", n=told(n))


@app.tool()
def visit(url: AnyUrl) -> str:
    return record("visit", url=str(url))


@app.tool()
def opened(at: Path) -> str:
    return record("opened", at=at.as_posix())


@app.tool()
def ident(key: uuid.UUID) -> str:
    return record("ident", key=str(key))


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
        and then parses the string (a schema that allows only strings keeps every string,
        so this one is not refused); a float parameter rounds an integer above 2**53; and
        a ``default_factory`` default is in no schema. ``TestThirdReview`` holds the JSON
        text the gate does refuse: for a parameter whose schema allows another type."""
        found = stdio(
            [
                [
                    ("wipe", {"prefix": "null"}),
                    ("wipe", {"prefix": "[]"}),
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
            ("wipe", {"prefix": "[]"}),
            ("pay", {"amount": 9007199254740993}),
            ("tagged", {"src": "a"}),
        ]
        assert _in_order(found.judged) == _in_order(found.wire) == _in_order(sent)
        assert _in_order(found.ran) == _in_order(
            [
                ("wipe", {"prefix": None}),
                ("wipe", {"prefix": []}),
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


DRAFT_07 = "http://json-schema.org/draft-07/schema#"

#: What ``zod-to-json-schema`` publishes for ``z.boolean().default(true)``.
ZOD_FLAG = {"type": "boolean", "default": True}

#: What pydantic publishes for ``where: Union[And, Or, Leaf]``, three models of which two
#: (``And`` and ``Or``) hold a list of the same union.
EXPRESSION = {
    "type": "object",
    "properties": {"where": {"$ref": "#/$defs/Expr"}},
    "required": ["where"],
    "$defs": {
        "Expr": {
            "anyOf": [
                {"$ref": "#/$defs/And"},
                {"$ref": "#/$defs/Or"},
                {"$ref": "#/$defs/Leaf"},
            ]
        },
        "And": {
            "type": "object",
            "properties": {
                "op": {"const": "and", "default": "and"},
                "args": {"type": "array", "items": {"$ref": "#/$defs/Expr"}},
                "strict": {"type": "boolean", "default": True},
            },
            "required": ["args"],
        },
        "Or": {
            "type": "object",
            "properties": {
                "op": {"const": "or", "default": "or"},
                "args": {"type": "array", "items": {"$ref": "#/$defs/Expr"}},
                "strict": {"type": "boolean", "default": True},
            },
            "required": ["args"],
        },
        "Leaf": {
            "type": "object",
            "properties": {
                "op": {"const": "leaf", "default": "leaf"},
                "field": {"type": "string"},
                "eq": {"type": "string"},
            },
            "required": ["field", "eq"],
        },
    },
}

#: What pydantic publishes for a model with ``child: Optional[Node] = None``.
TREE = {
    "type": "object",
    "properties": {"node": {"$ref": "#/$defs/Node"}},
    "required": ["node"],
    "$defs": {
        "Node": {
            "type": "object",
            "properties": {
                "keep": {"type": "boolean", "default": True},
                "child": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}], "default": None},
            },
        }
    },
}

_OBJECT_FORMS = (
    "is an object, and the server's schema offers more than one object form there "
    "(alternatives of an `anyOf` or `oneOf`); the gate cannot tell which of them the server "
    "will run, so it cannot show the policy what runs."
)
_PUBLISH_A_TAG = (
    " The server can publish a discriminated union (every form fixing one same-named "
    "property to a `const` of its own, which the call then carries), or separate tools."
)
_TWO_OBJECT_FORMS = _OBJECT_FORMS + _PUBLISH_A_TAG

_DEFAULT_OF_TWO_FORMS = (
    "is left out, and the default the server's schema lists for it holds an object or a "
    "list where the schema offers more than one form (alternatives of an `anyOf` or "
    "`oneOf`), so the gate cannot tell what the server would run with"
)

_TWO_LIST_FORMS = (
    "is a list, and the server's schema offers more than one list form there (alternatives "
    "of an `anyOf` or `oneOf`); the gate cannot tell which of them the server will run, so "
    "it cannot show the policy what runs. The server can publish one list form, or separate "
    "tools."
)


def _reads_as(held, allows, read, meant):
    """The one reason for a string a server may read as another type: what to send where
    the other type was meant, and what the server's author can do where the text was."""
    return (
        f"is a string that holds {held} where the server's schema allows {allows}; a server "
        f"may read the text as {read} and run with that value, not with the string the policy "
        f"judged. If {meant}, not text. If the text itself is meant (an identifier, a code, a "
        "word), the gate cannot send it while the schema allows both: the server's author can "
        "declare this parameter a string only (`str`, or `z.string()` with zod), and the gate then "
        "takes every string there as the string it is."
    )


_NUMBER_TEXT = _reads_as(
    "the text of a number",
    "a number as well as a string",
    "a number",
    "a number is meant, send a JSON number",
)
_JAVASCRIPT_NUMBER_TEXT = _reads_as(
    "text a server written in JavaScript reads as a number (an empty string or white space "
    "as 0, the digits after `0x`, `0o` or `0b` as an integer, a number after a byte order "
    "mark)",
    "a number as well as a string",
    "a number",
    "a number is meant, send a JSON number",
)
_BOOLEAN_TEXT = _reads_as(
    "a word for a boolean",
    "a boolean as well as a string",
    "a boolean",
    "a boolean is meant, send JSON `true` or `false`",
)
_JSON_TEXT = _reads_as(
    "JSON text (null, a list or an object)",
    "another JSON type than string",
    "JSON",
    "null, a list or an object is meant, send that JSON value",
)


def _nested(depth, leaf):
    """``leaf`` under ``depth`` levels of ``{"op": "and", "args": [...]}``."""
    value = {"op": "and", **leaf}
    for _ in range(depth):
        value = {"op": "and", "args": [value]}
    return {"where": value}


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestThirdReview:
    """The third review of 2026-11-21. A default listed inside an alternative of a
    property's own ``anyOf`` or ``allOf`` (what zod publishes for
    ``z.boolean().default(true).nullable()``) was neither written nor refused; JSON text
    for a parameter that may also be null ran as null on the MCP Python SDK's server; a
    model could hang the agent's process with a few hundred bytes of nested arguments; and
    ordinary servers were refused for a legal chain of a recursive model, a pattern Python
    cannot compile, an ``https`` spelling of a draft and a ``tool_filter`` shared by two
    agents."""

    example = TestWhatIsSentIsWhatWasJudged.example
    Recording = TestWhatIsSentIsWhatWasJudged.Recording
    stdio = TestARealServerRunsWhatWasJudged.stdio
    judged = TestTheWholeListedSchema.judged

    @staticmethod
    def schema_module():
        import importlib

        _adapter()
        return importlib.import_module("recusal_openai_agents_mcp._schema")

    # --- a default listed inside an alternative of the property's own schema ------------------

    @pytest.mark.parametrize(
        "listed, written",
        [
            ({"anyOf": [ZOD_FLAG, {"type": "null"}]}, True),
            (
                {
                    "anyOf": [{**ZOD_FLAG, "description": "d"}, {"type": "null"}],
                    "description": "d",
                },
                True,
            ),
            ({"allOf": [ZOD_FLAG, {"type": "boolean"}]}, True),
            ({"anyOf": [ZOD_FLAG, {"type": "string"}]}, True),
            ({"anyOf": [{"type": "string"}, ZOD_FLAG]}, True),
            ({"oneOf": [ZOD_FLAG, {"type": "null"}]}, True),
            ({"anyOf": [{"anyOf": [ZOD_FLAG, {"type": "null"}]}, {"type": "string"}]}, True),
            ({"anyOf": [{"$ref": "#/$defs/Flag"}, {"type": "null"}]}, True),
            ({"allOf": [{"$ref": "#/$defs/Nullable"}]}, True),
            ({"anyOf": [ZOD_FLAG, {"type": ["boolean", "null"], "default": True}]}, True),
            (
                {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"force": ZOD_FLAG},
                            "additionalProperties": False,
                            "default": {},
                        },
                        {"type": "null"},
                    ]
                },
                {"force": True},
            ),
            (
                {"anyOf": [{"type": "array", "default": [1, {"a": 1}]}, {"type": "null"}]},
                [1, {"a": 1}],
            ),
        ],
        ids=[
            "nullable",
            "described-then-nullable",
            "piped",
            "a-union",
            "a-union-the-other-way",
            "one-of",
            "an-alternative-of-an-alternative",
            "an-alternative-that-is-a-ref",
            "a-member-that-is-a-ref-to-alternatives",
            "two-alternatives-one-default",
            "an-object-default-completed-in-turn",
            "a-list-default",
        ],
    )
    def test_a_default_listed_by_an_alternative_is_written_judged_and_sent(
        self, example, listed, written
    ):
        schema = {
            "$schema": DRAFT_07,
            "type": "object",
            "properties": {"path": {"type": "string"}, "recursive": listed},
            "required": ["path"],
            "additionalProperties": False,
            "$defs": {"Flag": ZOD_FLAG, "Nullable": {"anyOf": [ZOD_FLAG, {"type": "null"}]}},
        }
        completed = {"path": "/data", "recursive": written}
        assert self.judged({"path": "/data"}, schema) == (completed, completed, None)
        # a value that is sent is judged and sent as it is
        given = {"path": "/data", "recursive": None}
        if "null" in str(listed):
            assert self.judged(given, schema) == (given, given, None)

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda leaf: {"anyOf": [leaf, {"type": "null"}]},
            lambda leaf: {"anyOf": [{"type": "null"}, leaf]},
            lambda leaf: {"oneOf": [leaf, {"type": "null"}]},
            lambda leaf: {"allOf": [leaf, {}]},
            lambda leaf: {"allOf": [{}, leaf]},
            lambda leaf: {"anyOf": [{"anyOf": [leaf, {"type": "null"}]}, {"type": "null"}]},
            lambda leaf: {"allOf": [{"anyOf": [leaf, {"type": "null"}]}]},
            lambda leaf: {"anyOf": [{"allOf": [leaf, {}]}, {"type": "null"}]},
            lambda leaf: {"anyOf": [leaf, {**leaf, "title": "again"}]},
        ],
        ids=[
            "nullable",
            "null-first",
            "one-of",
            "all-of",
            "all-of-last",
            "nullable-twice",
            "all-of-over-any-of",
            "any-of-over-all-of",
            "listed-twice",
        ],
    )
    @pytest.mark.parametrize(
        "leaf",
        [
            {"type": "boolean", "default": True},
            {"type": "boolean", "default": False},
            {"type": "string", "default": "s"},
            {"type": "number", "default": 7},
            {"type": "integer", "default": 0},
            {"type": "array", "items": {"type": "string"}, "default": ["x"]},
            {"type": "string", "enum": ["a", "b"], "default": "a"},
        ],
        ids=["true", "false", "a-string", "a-number", "zero", "a-list", "an-enum"],
    )
    def test_every_spelling_of_one_default_writes_that_default(self, example, wrap, leaf):
        """A differential over the ways a schema can carry one default for a property: for
        each, the gate writes the value a server reading the default would run with."""
        schema = {"type": "object", "properties": {"p": wrap(leaf)}}
        completed = {"p": leaf["default"]}
        seen, sent, reason = self.judged({}, schema)
        if str(schema).count("'array'") == 2:
            # a list where the schema offers two list forms: which one a server takes its
            # default for is not in the schema
            assert reason == f"Recusal refused `purge`: argument `p` {_DEFAULT_OF_TWO_FORMS}"
            return
        assert reason is None
        assert seen == completed and sent == completed
        assert type(seen["p"]) is type(leaf["default"])

    @pytest.mark.parametrize(
        "listed",
        [
            {"anyOf": [ZOD_FLAG, {"type": "string", "default": "other"}]},
            {"anyOf": [ZOD_FLAG, {"type": "boolean", "default": False}]},
            {"anyOf": [{"type": "integer", "default": 1}, ZOD_FLAG]},
            {"anyOf": [{"type": "integer", "default": 1}, {"type": "number", "default": 1.0}]},
            {"anyOf": [{"default": [1]}, {"default": [True]}]},
            {"anyOf": [{"default": {"a": 1}}, {"default": {"a": True}}]},
            {"allOf": [{"default": 1}, {"default": 2}]},
            {"anyOf": [{"$ref": "#/$defs/Flag"}, {"type": "boolean", "default": False}]},
        ],
        ids=[
            "two-types",
            "true-and-false",
            "one-and-true",
            "one-and-one-point-zero",
            "inside-a-list",
            "inside-an-object",
            "all-of",
            "a-ref-and-another",
        ],
    )
    def test_alternatives_that_list_different_defaults_for_a_left_out_property_are_refused(
        self, example, listed
    ):
        schema = {"properties": {"recursive": listed}, "$defs": {"Flag": ZOD_FLAG}}
        assert self.judged({}, schema) == (
            None,
            None,
            "Recusal refused `purge`: argument `recursive` is left out, and the alternatives "
            "the server's schema lists for it list different defaults, so the gate cannot tell "
            "what the server would run with",
        )
        # sent, nothing is left to tell
        for value in (True, 1, "x", [1], {"a": 1}):
            seen, sent, reason = self.judged({"recursive": value}, schema)
            assert reason is None or "left out" not in reason

    def test_the_propertys_own_default_is_the_one(self, example):
        """Over the default the schema its ``$ref`` names lists, and over the defaults of
        its alternatives."""
        schema = {
            "properties": {
                "a": {"$ref": "#/$defs/Flag", "default": False},
                "b": {"anyOf": [ZOD_FLAG, {"type": "null"}], "default": None},
                "c": {"allOf": [{"default": 1}, {"default": 2}], "default": 3},
            },
            "$defs": {"Flag": ZOD_FLAG},
        }
        completed = {"a": False, "b": None, "c": 3}
        assert self.judged({}, schema) == (completed, completed, None)

    def test_a_default_of_an_alternative_that_cannot_be_sent_is_judged_and_not_sent(self, example):
        schema = {"properties": {"q": {"anyOf": [{"type": "string", "default": 5}]}}}
        assert self.judged({}, schema) == ({"q": 5}, {}, None)

    def test_an_item_left_out_whose_alternative_lists_a_default_is_refused(self, example):
        entry = {"anyOf": [ZOD_FLAG, {"type": "null"}]}
        schema = {"properties": {"t": {"prefixItems": [{"type": "integer"}, entry]}}}
        assert (
            "leaves out an item the server's schema lists a default for"
            in (self.judged({"t": [5]}, schema)[2])
        )

    def test_two_parts_of_the_schema_that_list_different_defaults_for_one_place_are_refused(
        self, example
    ):
        one = {"properties": {"depth": {"type": "integer", "default": 1}}}
        other = {"properties": {"depth": {"type": "integer", "default": 2}}}
        schema = {"properties": {"opts": {"type": "object", "allOf": [one, other]}}}
        assert self.judged({"opts": {}}, schema) == (
            None,
            None,
            "Recusal refused `purge`: argument `opts.depth` is left out, and the server's "
            "schema lists different defaults for it, so the gate cannot tell what the server "
            "would run with",
        )
        # alternatives that list one default, which only one of them could be sent, differ
        typed = {"properties": {"q": {"type": "string", "default": 5}}}
        untyped = {"properties": {"q": {"default": 5}}}
        schema = {"properties": {"opts": {"type": "object", "allOf": [typed, untyped]}}}
        assert self.judged({"opts": {}}, schema)[2] == (
            "Recusal refused `purge`: argument `opts.q` is left out, and the server's schema "
            "lists different defaults for it, so the gate cannot tell what the server would "
            "run with"
        )
        for both in ([typed, dict(typed)], [untyped, dict(untyped)]):
            schema = {"properties": {"opts": {"type": "object", "allOf": both}}}
            assert self.judged({"opts": {}}, schema)[2] is None
        # compared by the JSON type: `1` and `1.0` are two defaults
        exact = {"properties": {"depth": {"default": 1.0}}}
        schema = {"properties": {"opts": {"type": "object", "allOf": [one, exact]}}}
        assert "lists different defaults for it" in self.judged({"opts": {}}, schema)[2]
        same = {"properties": {"opts": {"type": "object", "allOf": [one, dict(one)]}}}
        assert self.judged({"opts": {}}, same) == (
            {"opts": {"depth": 1}},
            {"opts": {"depth": 1}},
            None,
        )

    @staticmethod
    def zod_like_server(example, ran):
        """A server that lists what zod publishes for
        ``z.object({path: z.string(), recursive: z.boolean().default(true).nullable()})``
        and runs as ``schema.parse`` would: a ``recursive`` left out is ``true``."""
        from mcp.types import Tool

        schema = {
            "$schema": DRAFT_07,
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"anyOf": [ZOD_FLAG, {"type": "null"}]},
            },
            "required": ["path"],
            "additionalProperties": False,
        }

        class ZodLike(example.InProcessMCPServer):
            async def list_tools(self, run_context=None, agent=None):
                return [Tool(name="z_delete", description="Delete.", inputSchema=dict(schema))]

            async def call_tool(self, tool_name, arguments, meta=None):
                ran.append({"recursive": True, **(arguments or {})})
                return await super().call_tool(tool_name, arguments, meta)

        return ZodLike(name="files", tools=("z_delete",))

    def test_a_server_that_fills_the_alternatives_default_runs_what_the_policy_judged(
        self, example
    ):
        from recusal import Finding

        def no_recursive(name, args):
            if args.get("recursive") is not False:
                return [Finding.fail("no_recursive", severity="CRITICAL", message="not that")]
            return []

        ran: list = []
        policy = self.Recording(no_recursive)
        server = example.gate_mcp_server(self.zod_like_server(example, ran), policy)
        refused = asyncio.run(server.call_tool("z_delete", {"path": "/data"}))
        assert refused.content[0].text == "Recusal refused `z_delete` [FAIL]: not that"
        assert ran == [], "the server never ran with the default no policy would have seen"
        asyncio.run(server.call_tool("z_delete", {"path": "/data", "recursive": False}))
        assert policy.seen == [
            ("z_delete", {"path": "/data", "recursive": True}),
            ("z_delete", {"path": "/data", "recursive": False}),
        ]
        assert ran == [{"path": "/data", "recursive": False}]

    def test_negative_control_ungated_the_server_fills_the_default_no_policy_saw(self, example):
        ran: list = []
        server = self.zod_like_server(example, ran)
        asyncio.run(server.call_tool("z_delete", {"path": "/data"}))
        assert ran == [{"path": "/data", "recursive": True}]

    # --- a default where the gate does not read one refuses the schema ------------------------

    @pytest.mark.parametrize(
        "schema, keyword",
        [
            (
                {"$schema": DRAFT_07, "dependencies": {"a": {"properties": {"b": ZOD_FLAG}}}},
                "dependencies",
            ),
            ({"not": {"properties": {"b": ZOD_FLAG}}}, "not"),
            ({"if": {"properties": {"b": ZOD_FLAG}}}, "if"),
            ({"if": {"required": ["zz"]}, "else": {"properties": {"b": ZOD_FLAG}}}, "else"),
            ({"properties": {"a": {"contains": ZOD_FLAG}}}, "contains"),
            ({"propertyNames": {"default": "x"}}, "propertyNames"),
            ({"unevaluatedProperties": ZOD_FLAG}, "unevaluatedProperties"),
            ({"properties": {"a": {"unevaluatedItems": ZOD_FLAG}}}, "unevaluatedItems"),
            (
                {"properties": {"a": {"contentSchema": {"properties": {"b": ZOD_FLAG}}}}},
                "contentSchema",
            ),
        ],
        ids=[
            "dependencies",
            "not",
            "if",
            "else",
            "contains",
            "property-names",
            "unevaluated-properties",
            "unevaluated-items",
            "content-schema",
        ],
    )
    def test_a_default_under_a_keyword_the_walk_does_not_follow_is_refused(
        self, example, schema, keyword
    ):
        assert self.judged({"x": 1}, schema) == (
            None,
            None,
            f"Recusal refused `purge`: {_SCHEMA} lists a default under `{keyword}`, where the "
            "gate cannot tell whether the server would apply it",
        )

    @pytest.mark.parametrize(
        "schema",
        [
            {"x-shape": ZOD_FLAG},
            {"properties": {"a": {"x-shape": {"properties": {"b": ZOD_FLAG}}}}},
            {"properties": {"a": {"futureOf": [{"type": "string"}, ZOD_FLAG]}}},
            {"properties": {"a": {"type": "string", "x-ui": {"widget": {"default": "big"}}}}},
        ],
        ids=["at-the-root", "under-a-property", "in-a-list", "deep-in-data"],
    )
    def test_a_default_under_a_keyword_the_gate_does_not_know_is_refused(self, example, schema):
        assert self.judged({"a": "x"}, schema) == (
            None,
            None,
            f"Recusal refused `purge`: {_SCHEMA} lists a default under a keyword the gate "
            "does not read, where it cannot tell whether the server would apply it",
        )

    def test_a_member_named_default_in_data_refuses_nothing(self, example):
        """``const``, ``enum``, ``examples`` and a ``default`` hold data, which may have a
        member named ``default``; so may the arguments, and a property may be named so."""
        schema = {
            "properties": {
                "a": {"const": {"default": 1}},
                "b": {"enum": [{"default": 1}, {"default": 2}]},
                "c": {"type": "object", "examples": [{"default": 1}], "example": {"default": 1}},
                "d": {"type": "object", "default": {"default": 1}},
                "default": {"type": "integer", "default": 3},
                "e": {"type": "string", "x-ui": {"widget": "big"}, "discriminator": {"k": "v"}},
            }
        }
        given = {"a": {"default": 1}, "b": {"default": 2}, "c": {"default": 5}}
        completed = {**given, "d": {"default": 1}, "default": 3}
        assert self.judged(given, schema) == (completed, completed, None)

    def test_a_default_where_a_value_is_always_present_is_never_applied(self, example):
        """On the schema of the arguments themselves, of the items of a list and of
        ``additionalProperties``: JSON cannot send a member or an item without a value."""
        schema = {
            "type": "object",
            "default": {"path": "/"},
            "properties": {
                "flags": {"type": "array", "items": ZOD_FLAG},
                "by": {"type": "object", "additionalProperties": ZOD_FLAG},
            },
        }
        given = {"flags": [False], "by": {"a": False}}
        assert self.judged(given, schema) == (given, given, None)
        assert self.judged({}, schema) == ({}, {}, None)
        assert self.judged(None, schema) == ({}, {}, None)

    # --- JSON text a server may read as another value -----------------------------------------

    NULLABLE = [
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        {"type": ["string", "null"]},
        {"oneOf": [{"type": "string"}, {"type": "null"}]},
        {"anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}], "default": 10},
        {"anyOf": [{"type": "string", "format": "date-time"}, {"type": "null"}]},
        {"anyOf": [{"type": "object"}, {"type": "string"}]},
        {"$ref": "#/$defs/Nullable"},
        {"allOf": [{"$ref": "#/$defs/Nullable"}]},
        {},
        {"title": "Anything"},
    ]

    @pytest.mark.parametrize("text", ["null", "[]", "{}", '{"a": 1}', "[true]", " null\n"])
    @pytest.mark.parametrize("listed", NULLABLE, ids=[str(n) for n in range(len(NULLABLE))])
    def test_json_text_for_a_property_that_allows_another_type_is_refused(
        self, example, listed, text
    ):
        schema = {
            "properties": {"scope": listed},
            "$defs": {"Nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        }
        assert self.judged({"scope": text}, schema) == (
            None,
            None,
            f"Recusal refused `purge`: argument `scope` {_JSON_TEXT}",
        )

    @pytest.mark.parametrize("text", ["true", "false", " true\n"])
    @pytest.mark.parametrize("listed", NULLABLE, ids=[str(n) for n in range(len(NULLABLE))])
    def test_text_for_a_boolean_is_judged_and_sent_as_the_string_it_is(self, example, listed, text):
        """The MCP Python SDK's server keeps the text of a boolean a string (to its check
        a ``bool`` is an ``int``, which it skips), so what runs is the string judged."""
        schema = {
            "properties": {"scope": listed},
            "$defs": {"Nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        }
        given = {"scope": text}
        assert self.judged(given, schema) == (given, given, None)

    @pytest.mark.parametrize(
        "text", ["tmp", "", "500", "1e3", "-0.5", '"quoted"', "nul", "{", "[1,", "NaN", "None"]
    )
    def test_text_that_is_a_number_a_string_or_not_json_is_judged_as_sent(self, example, text):
        schema = {"properties": {"scope": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}
        given = {"scope": text}
        assert self.judged(given, schema) == (given, given, None)

    STRINGS_ONLY = [
        {"type": "string"},
        {"type": "string", "format": "date-time"},
        {"type": "string", "minLength": 1, "title": "Scope"},
        {"$ref": "#/$defs/Text"},
        {"allOf": [{"$ref": "#/$defs/Text"}]},
        {"anyOf": [{"type": "string"}, {"$ref": "#/$defs/Text"}]},
        {"enum": ["null", "[]", "{}", "true"]},
        {"const": "null"},
        {"type": ["string"]},
        {"allOf": [{"type": ["string", "null"]}, {"type": "string"}]},
    ]

    @pytest.mark.parametrize("listed", STRINGS_ONLY, ids=[str(n) for n in range(len(STRINGS_ONLY))])
    def test_a_property_that_allows_only_strings_keeps_every_string(self, example, listed):
        schema = {"properties": {"scope": listed}, "$defs": {"Text": {"type": "string"}}}
        for text in ("null", "[]", "{}", "true"):
            if "const" in listed and text != "null":
                continue
            given = {"scope": text}
            assert self.judged(given, schema) == (given, given, None)

    def test_in_draft_07_a_ref_is_read_alone(self, example):
        """Draft-07 does not apply the keywords beside a ``$ref``, so they do not make a
        property a string-only one."""
        schema = {
            "$schema": DRAFT_07,
            "properties": {"scope": {"$ref": "#/definitions/Nullable", "type": "string"}},
            "definitions": {"Nullable": {"type": ["string", "null"]}},
        }
        assert _JSON_TEXT in self.judged({"scope": "null"}, schema)[2]
        modern = {**schema, "$schema": "https://json-schema.org/draft/2020-12/schema"}
        assert self.judged({"scope": "null"}, modern) == (
            {"scope": "null"},
            {"scope": "null"},
            None,
        )

    def test_json_text_is_looked_for_in_the_named_arguments_only(self, example):
        """The MCP Python SDK's server decodes the text of the arguments its function
        names, at the top; a string further in, or under a name the schema does not list,
        is judged and sent as it is."""
        schema = {
            "properties": {
                "opts": {
                    "type": "object",
                    "properties": {"scope": {"type": ["string", "null"]}},
                },
                "tags": {"type": "array", "items": {"type": ["string", "null"]}},
            }
        }
        given = {"opts": {"scope": "null"}, "tags": ["null", "[]"], "note": "null"}
        assert self.judged(given, schema) == (given, given, None)

    def test_a_value_of_another_type_keeps_its_own_reason(self, example):
        """Text for a property that allows no string is refused for its type, as before."""
        schema = {"properties": {"opts": {"anyOf": [{"type": "object"}, {"type": "null"}]}}}
        assert self.judged({"opts": "{}"}, schema)[2] == (
            "Recusal refused `purge`: argument `opts` is of JSON type string where the "
            f"server's schema declares object or null; {_SEND_THE_TYPE}"
        )

    def test_a_default_that_is_json_text_a_server_would_decode_is_judged_and_not_sent(
        self, example
    ):
        """The server runs with its own default as the string it is, and would decode it
        if the gate sent it: so it is judged, and it is not sent. A default the server
        would not decode (text for a boolean, one for a property that allows only
        strings, one further in) is written and sent."""
        nullable = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        schema = {
            "properties": {
                "mode": {**nullable, "default": "null"},
                "flag": {**nullable, "default": "true"},
                "tags": {"type": ["string", "array"], "default": "[]"},
                "plain": {"type": "string", "default": "null"},
                "name": {**nullable, "default": "tmp"},
                "opts": {
                    "type": "object",
                    "properties": {"mode": {**nullable, "default": "null"}},
                    "default": {},
                },
            }
        }
        assert self.judged({}, schema) == (
            {
                "mode": "null",
                "flag": "true",
                "tags": "[]",
                "plain": "null",
                "name": "tmp",
                "opts": {"mode": "null"},
            },
            {"flag": "true", "plain": "null", "name": "tmp", "opts": {"mode": "null"}},
            None,
        )

    def test_on_a_real_server_a_default_that_is_json_text_runs_as_the_string_judged(self, stdio):
        found = stdio([[("moded", {}), ("moded", {"mode": "fast"})]], schemas=("moded",))
        assert found.schemas["moded"]["mode"]["default"] == "null"
        judged = [("moded", {"mode": "null"}), ("moded", {"mode": "fast"})]
        assert _in_order(found.judged) == _in_order(judged)
        assert _in_order(found.ran) == _in_order(judged), "the server ran the string, not None"
        assert _in_order(found.wire) == _in_order([("moded", {}), ("moded", {"mode": "fast"})])

    def test_negative_control_the_real_server_decodes_that_default_when_it_is_sent(self, stdio):
        found = stdio([[("moded", {"mode": "null"})]], gated=False)
        assert found.ran == [("moded", {"mode": None})]

    def test_on_a_real_server_json_text_for_a_nullable_parameter_never_runs(self, stdio):
        calls = [
            ("scoped", {"scope": "null"}),
            ("limited", {"limit": "null"}),
            ("since", {"day": "2024-01-01", "at": "null"}),
            ("located", {"q": "null"}),
            ("scoped_required", {"scope": "null"}),
            ("either", {"v": '{"a": 1}'}),
            ("scoped", {"scope": "[]"}),
        ]
        found = stdio([calls])
        assert found.ran == [] and found.wire == [] and found.judged == []
        names = ["scope", "limit", "at", "q", "scope", "v", "scope"]
        for number, ((tool, _), name) in enumerate(zip(calls, names), 1):
            assert found.outputs[f"c{number}"] == (
                f"Recusal refused `{tool}`: argument `{name}` {_JSON_TEXT}"
            )

    def test_on_a_real_server_an_ordinary_string_runs_as_the_string_that_was_judged(self, stdio):
        calls = [
            ("scoped", {"scope": "archive"}),
            ("scoped", {"scope": None}),
            ("limited", {"limit": "five hundred"}),
            ("limited", {"limit": 5}),
            ("scoped_required", {"scope": "nul"}),
            ("either", {"v": "plain text"}),
            ("either", {"v": {"a": 1}}),
            ("scoped_required", {"scope": "true"}),
            ("limited", {"limit": "false"}),
        ]
        found = stdio([calls])
        judged = [
            ("scoped_required", {"scope": "true"}),
            ("limited", {"limit": "false"}),
            ("scoped", {"scope": "archive", "confirm": False}),
            ("scoped", {"scope": None, "confirm": False}),
            ("limited", {"limit": "five hundred"}),
            ("limited", {"limit": 5}),
            ("scoped_required", {"scope": "nul"}),
            ("either", {"v": "plain text"}),
            ("either", {"v": {"a": 1}}),
        ]
        assert _in_order(found.judged) == _in_order(judged)
        assert _in_order(found.wire) == _in_order(judged)
        assert _in_order(found.ran) == _in_order(judged)

    def test_negative_control_ungated_the_real_server_runs_json_text_as_null(self, stdio):
        found = stdio(
            [[("scoped", {"scope": "null"}), ("scoped_required", {"scope": "null"})]], gated=False
        )
        assert _in_order(found.ran) == _in_order(
            [("scoped", {"scope": None, "confirm": False}), ("scoped_required", {"scope": None})]
        )

    def test_what_a_real_servers_validators_do_to_a_string_is_not_seen(self, stdio):
        """The README's examples of what a schema does not say, kept true by a test: a
        numeric string runs as a date, and a constraint that strips and lowers a string
        changes the value that is sent, a default the gate wrote included (the server
        itself would not have validated its own default)."""
        found = stdio(
            [
                [
                    ("since", {"day": "0"}),
                    ("named", {"name": "  /ETC "}),
                    ("named", {}),
                ]
            ],
            schemas=("named",),
        )
        assert found.schemas["named"] == {
            "name": {"default": "X ", "title": "Name", "type": "string"}
        }
        sent = [
            ("since", {"day": "0", "at": None}),
            ("named", {"name": "  /ETC "}),
            ("named", {"name": "X "}),
        ]
        assert _in_order(found.judged) == _in_order(found.wire) == _in_order(sent)
        assert _in_order(found.ran) == _in_order(
            [
                ("since", {"day": "1970-01-01", "at": None}),
                ("named", {"name": "/etc"}),
                ("named", {"name": "x"}),
            ]
        )

    def test_negative_control_ungated_the_real_server_does_not_validate_its_own_default(
        self, stdio
    ):
        found = stdio([[("named", {})]], gated=False)
        assert found.ran == [("named", {"name": "X "})]

    # --- the work is bounded --------------------------------------------------------------------

    def spent(self, arguments, schema, ceiling=10**9):
        """``(steps spent, steps allowed, the result or the reason it was refused)``,
        under a ceiling no call here reaches unless the test names one: these tests
        are of what is counted for each value."""
        module = self.schema_module()
        made: list = []
        original = module._Work.__init__

        def remembering(work, allowed):
            original(work, allowed)
            made.append(work)

        module._Work.__init__ = remembering
        try:
            try:
                result = module.Listed(schema, ceiling).complete(arguments)
            except module.Refused as refused:
                result = refused.reason
        finally:
            module._Work.__init__ = original
        return made[-1].steps, made[-1].allowed, result

    def test_a_schema_that_refers_to_itself_costs_in_proportion_to_the_arguments(self, example):
        """The reviewer's case: two alternatives of a recursive ``anyOf`` both validate.
        The cost more than doubled with every level: twelve took over a minute."""
        costs = {}
        for depth in (4, 8, 16, 31):
            arguments = _nested(depth, {"args": []})
            steps, allowed, result = self.spent(arguments, EXPRESSION)
            assert type(result) is tuple, result
            assert result[0]["where"]["strict"] is True
            costs[depth] = steps
            assert steps * 20 < allowed
        assert costs[16] - costs[8] == 2 * (costs[8] - costs[4]), "each level costs the same"
        assert costs[31] < 4 * costs[8]

    def test_arguments_that_fail_deep_inside_cost_in_proportion_too(self, example):
        costs = {}
        for depth in (4, 8, 16, 31):
            steps, allowed, result = self.spent(_nested(depth, {"args": 5}), EXPRESSION)
            assert result == (
                "argument `where` matches none of the alternatives the server's schema lists "
                f"there; {_AS_THEY_STAND}"
            )
            costs[depth] = steps
            assert steps * 20 < allowed
        assert costs[31] < 4 * costs[8]

    def test_the_arguments_of_the_deepest_legal_recursive_call_are_completed(self, example):
        arguments = _nested(31, {"args": []})
        assert self.schema_module().measure(arguments, 10**9) == ""
        seen, sent, reason = self.judged(arguments, EXPRESSION)
        assert reason is None and seen == sent
        level = seen["where"]
        for _ in range(32):
            assert level["strict"] is True
            level = level["args"][0] if level["args"] else None
        assert level is None

    def test_the_allowance_grows_with_the_arguments(self, example):
        module = self.schema_module()
        schema = {
            "type": "object",
            "properties": {"files": {"type": "array", "items": OPTIONS}},
        }
        for count in (1, 500, 5000):
            arguments = {"files": [{} for _ in range(count)]}
            steps, allowed, result = self.spent(arguments, schema)
            assert allowed == module._STEPS + module._STEPS_PER_VALUE * (count + 2)
            assert type(result) is tuple and result[0]["files"][-1] == FILLED
            assert steps * 20 < allowed

    def test_arguments_too_costly_to_check_are_refused_before_the_policy(
        self, example, monkeypatch
    ):
        module = self.schema_module()
        monkeypatch.setattr(module, "_STEPS", 150)
        monkeypatch.setattr(module, "_STEPS_PER_VALUE", 1)
        reason = f"Recusal refused `purge`: {_over_each_value(150, 1)}"
        assert self.judged(_nested(16, {"args": []}), EXPRESSION) == (None, None, reason)
        assert self.judged(_nested(16, {"args": 5}), EXPRESSION) == (None, None, reason)
        small = {"where": {"op": "leaf", "field": "a", "eq": "b"}}
        assert self.judged(small, EXPRESSION) == (small, small, None)

    def test_every_step_is_counted_and_the_count_is_left_behind(self, example, monkeypatch):
        """A step past the allowance stops the work at once, whatever is being checked,
        and the next call starts from nothing."""
        module = self.schema_module()
        steps, allowed, _ = self.spent(_nested(8, {"args": []}), EXPRESSION)
        monkeypatch.setattr(module, "_STEPS_PER_VALUE", 0)
        monkeypatch.setattr(module, "_STEPS", steps - 1)
        over, limit, result = self.spent(_nested(8, {"args": []}), EXPRESSION)
        assert (over, limit) == (steps, steps - 1)
        assert result == _over_each_value(steps - 1, 0)
        monkeypatch.setattr(module, "_STEPS", steps)
        again, _, result = self.spent(_nested(8, {"args": []}), EXPRESSION)
        assert again == steps and type(result) is tuple
        assert module._WORK.get() is None

    def test_no_keyword_is_run_twice_for_one_schema_and_one_value(self, example):
        """A schema that reaches itself twice at every level, through two members of an
        ``allOf``: checked once for each value, a valid chain costs in proportion to its
        length, where checking it afresh through each member would double at every level."""
        again = {"properties": {"child": {"$ref": "#/$defs/Node"}}}
        schema = {
            "properties": {"node": {"$ref": "#/$defs/Node"}},
            "$defs": {"Node": {"type": "object", "allOf": [again, dict(again)]}},
        }
        costs = {}
        for length in (5, 10, 20, 40):
            steps, allowed, result = self.spent(self.chain(length), schema)
            assert type(result) is tuple, result
            costs[length] = steps
            assert steps * 5 < allowed
        assert costs[40] < 5 * costs[10]

    # --- a recursive model nested to the limit of the arguments -------------------------------

    @staticmethod
    def chain(length):
        value: dict = {}
        for _ in range(length):
            value = {"child": value}
        return {"node": value}

    def test_a_chain_of_a_recursive_model_is_completed_to_the_depth_limit(self, example):
        """Refused at 43 levels as a schema that "refers to itself without end": the walk
        counted schema steps, about three for each level of the arguments."""
        deepest = self.chain(63)
        measure = self.schema_module().measure
        assert measure(deepest, 10**9) == "" and measure(self.chain(64), 10**9) == "depth"
        seen, sent, reason = self.judged(deepest, TREE)
        assert reason is None
        level, levels = seen["node"], 0
        while level is not None:
            assert level["keep"] is True
            level, levels = level["child"], levels + 1
        assert levels == 64
        assert self.judged(self.chain(64), TREE)[2] == (
            "Recusal refused `purge`: arguments are nested too deeply"
        )

    def test_the_chain_runs_through_a_gated_server_inside_an_event_loop(self, example):
        from mcp.types import Tool

        class Trees(example.InProcessMCPServer):
            async def list_tools(self, run_context=None, agent=None):
                return [Tool(name="purge", description="Purge.", inputSchema=dict(TREE))]

        policy = self.Recording()
        server = example.gate_mcp_server(Trees(name="files", tools=("purge",)), policy)
        example.CALLED.clear()
        result = asyncio.run(server.call_tool("purge", self.chain(63)))
        assert not result.model_dump(by_alias=True).get("isError")
        assert len(example.CALLED) == 1 and len(policy.seen) == 1

    @pytest.mark.parametrize(
        "schema",
        [
            {"$ref": "#"},
            {"anyOf": [{"type": "null"}, {"$ref": "#"}]},
            {"properties": {"a": {"allOf": [{"$ref": "#/properties/a"}]}}},
            {"properties": {"a": {"not": {"$ref": "#/properties/a"}}}},
            {"properties": {"a": {"if": {"$ref": "#/properties/a"}}}},
            {
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "$defs": {"A": {"oneOf": [{"$ref": "#/$defs/B"}]}, "B": {"$ref": "#/$defs/A"}},
            },
        ],
        ids=["a-ref", "any-of", "all-of", "not", "if", "through-two-definitions"],
    )
    def test_a_schema_that_leads_back_to_itself_with_no_value_between_is_refused(
        self, example, schema
    ):
        assert self.judged({"a": 1}, schema) == (
            None,
            None,
            f"Recusal refused `purge`: {_SCHEMA} refers to itself without end",
        )

    def test_a_recursion_error_while_checking_names_the_interpreters_limit(
        self, example, monkeypatch
    ):
        module = self.schema_module()

        def out_of_stack(self, validator, instance):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(module.Listed, "_validate", out_of_stack)
        assert self.judged({"node": {}}, TREE) == (
            None,
            None,
            "Recusal refused `purge`: the arguments are nested too deeply to check against "
            f"{_SCHEMA} within what is left of the interpreter's recursion limit where the "
            "gate was called",
        )

    # --- a pattern Python cannot read -----------------------------------------------------------

    def test_a_pattern_is_not_asserted(self, example):
        """Whether Python can read it or not (the fourth review: one it can read ran on
        the model's text for as long as the pattern took)."""
        schema = {
            "properties": {
                "name": {"type": "string", "pattern": "^\\p{L}+$"},
                "code": {"type": "string", "pattern": "^[a-z]+$"},
            }
        }
        for given in ({"name": "anything 123", "code": "abc"}, {"code": "ABC"}):
            assert self.judged(given, schema) == (given, given, None)
        # the type beside a pattern still is
        assert "is of JSON type integer" in self.judged({"name": 5}, schema)[2]

    def test_a_pattern_python_cannot_compile_is_not_asserted_for_member_names(self, example):
        schema = {"propertyNames": {"pattern": "^\\p{L}+$"}}
        given = {"any name": 1}
        assert self.judged(given, schema) == (given, given, None)

    def test_members_named_by_a_pattern_python_cannot_compile_are_checked_all_the_same(
        self, example
    ):
        """Refused as a schema once: no pattern is compiled now, so it does not matter
        whether Python could."""
        schema = {"patternProperties": {"^\\p{L}+$": {"type": "integer"}}}
        assert self.judged({"a": 1}, schema) == ({"a": 1}, {"a": 1}, None)
        assert "argument `*` is of JSON type string" in self.judged({"a": "1"}, schema)[2]

    # --- the spellings of the drafts the gate reads ---------------------------------------------

    @pytest.mark.parametrize("fragment", ["", "#"], ids=["bare", "with-a-fragment"])
    @pytest.mark.parametrize("scheme", ["http", "https"])
    @pytest.mark.parametrize(
        "dialect", ["draft-07/schema", "draft/2019-09/schema", "draft/2020-12/schema"]
    )
    def test_every_common_spelling_of_a_supported_draft_is_read(
        self, example, dialect, scheme, fragment
    ):
        declared = f"{scheme}://json-schema.org/{dialect}{fragment}"
        tuple_keyword = "prefixItems" if "2020-12" in dialect else "items"
        schema = {
            "$schema": declared,
            "properties": {
                "t": {"type": "array", tuple_keyword: [{"type": "integer"}, OPTIONS]},
                "n": {"type": "integer", "default": 1},
            },
        }
        completed = {"t": [1, FILLED], "n": 1}
        assert self.judged({"t": [1, {}]}, schema) == (completed, completed, None)
        assert "is of JSON type string" in self.judged({"t": ["1", {}]}, schema)[2]

    @pytest.mark.parametrize(
        "declared",
        [
            "ftp://json-schema.org/draft-07/schema#",
            "http://json-schema.org/draft-07/schema##",
            "http://json-schema.org/draft-07/schema/",
            "json-schema.org/draft-07/schema",
            "http://json-schema.org/draft-06/schema#",
            "https://json-schema.org/draft/next/schema",
        ],
    )
    def test_any_other_spelling_is_still_a_draft_the_gate_does_not_read(self, example, declared):
        assert (
            "declares a JSON Schema draft the gate does not read"
            in (self.judged({}, {"$schema": declared})[2])
        )

    # --- what the README says of ordinary schemas -----------------------------------------------

    def test_an_openapi_nullable_keyword_does_not_allow_null(self, example):
        """``nullable: true`` is OpenAPI, not JSON Schema: the gate reads the declared
        type, so a ``null`` is refused. The README says so."""
        schema = {"properties": {"a": {"type": "string", "nullable": True}}}
        assert self.judged({"a": None}, schema)[2] == (
            "Recusal refused `purge`: argument `a` is of JSON type null where the server's "
            f"schema declares string; {_SEND_THE_TYPE}"
        )
        assert self.judged({"a": "x"}, schema) == ({"a": "x"}, {"a": "x"}, None)

    def test_a_union_of_two_models_with_defaults_refuses_what_both_accept(self, example):
        """Two ordinary Pydantic models: which of them a server runs is its own rule, so
        an object is refused whatever it holds, and ``additionalProperties: false`` does
        not tell them apart (a zod union drops what the first member does not know). A
        tag does."""
        by_path = {"type": "object", "properties": {"path": {"type": "string", "default": "/"}}}
        by_name = {"type": "object", "properties": {"name": {"type": "string", "default": "n"}}}
        loose = {"properties": {"target": {"anyOf": [by_path, by_name]}}}
        refused = f"Recusal refused `purge`: argument `target` {_TWO_OBJECT_FORMS}"
        for target in ({}, {"path": "/x"}, {"path": "/x", "name": "y"}):
            assert self.judged({"target": target}, loose) == (None, None, refused)
        strict = {
            "properties": {
                "target": {
                    "anyOf": [
                        {**by_path, "additionalProperties": False},
                        {**by_name, "additionalProperties": False},
                    ]
                }
            }
        }
        for target in ({}, {"path": "/x"}):
            assert self.judged({"target": target}, strict) == (None, None, refused)
        tagged = {
            "properties": {
                "target": {
                    "oneOf": [
                        {
                            **by_path,
                            "properties": {**by_path["properties"], "kind": {"const": "p"}},
                            "required": ["kind"],
                        },
                        {
                            **by_name,
                            "properties": {**by_name["properties"], "kind": {"const": "n"}},
                            "required": ["kind"],
                        },
                    ]
                }
            }
        }
        completed = {"target": {"kind": "n", "name": "n"}}
        assert self.judged({"target": {"kind": "n"}}, tagged) == (completed, completed, None)

    def test_a_tuple_with_a_defaulted_member_refuses_a_call_that_leaves_it_out(self, example):
        """What Pydantic publishes for a ``NamedTuple`` whose last member has a default."""
        schema = {
            "properties": {
                "pt": {
                    "type": "array",
                    "prefixItems": [{"type": "integer"}, {"type": "integer", "default": 0}],
                    "minItems": 1,
                    "maxItems": 2,
                }
            }
        }
        assert (
            "leaves out an item the server's schema lists a default for"
            in (self.judged({"pt": [1]}, schema)[2])
        )
        assert self.judged({"pt": [1, 2]}, schema) == ({"pt": [1, 2]}, {"pt": [1, 2]}, None)

    # --- the listing the gate keeps -------------------------------------------------------------

    @staticmethod
    def filtering_server(example, *, keeps_listing, fails_without_agent=False):
        """A server that lists ``purge`` to the agent named ``writer`` and ``legacy`` to
        any other, as a dynamic ``tool_filter`` would; with ``keeps_listing`` it shows the
        whole listing as ``cached_tools``, as the SDK's own servers do."""
        from mcp.types import Tool

        whole = [
            Tool(name="purge", description="Purge.", inputSchema=dict(PURGE_SCHEMA)),
            Tool(name="legacy", description="Legacy.", inputSchema={"type": "object"}),
        ]

        class Filtering(example.InProcessMCPServer):
            fetched = None

            @property
            def cached_tools(self):
                return self.fetched if keeps_listing else None

            async def list_tools(self, run_context=None, agent=None):
                self.fetched = list(whole)
                if agent is None and fails_without_agent:
                    raise RuntimeError("a dynamic tool filter needs an agent")
                wanted = "purge" if getattr(agent, "name", None) == "writer" else "legacy"
                return [tool for tool in whole if tool.name == wanted]

        return Filtering(name="files", tools=("purge", "legacy"))

    def test_two_agents_with_two_filters_share_the_servers_whole_listing(self, example):
        from types import SimpleNamespace

        policy = self.Recording()
        server = example.gate_mcp_server(self.filtering_server(example, keeps_listing=True), policy)
        writer, reader = SimpleNamespace(name="writer"), SimpleNamespace(name="reader")
        listed = asyncio.run(server.list_tools(None, writer))
        assert [tool.name for tool in listed] == ["purge"], "the listing is returned as filtered"
        asyncio.run(server.list_tools(None, reader))
        example.CALLED.clear()
        for name in ("purge", "legacy"):
            result = asyncio.run(server.call_tool(name, {}))
            assert not result.model_dump(by_alias=True).get("isError"), name
        assert [name for name, _ in policy.seen] == ["purge", "legacy"]
        assert policy.seen[0][1] == {"path": "/", "recursive": True, "dry_run": False}

    def test_on_a_real_server_two_agents_with_a_dynamic_filter_both_reach_their_tools(
        self, example, tmp_path
    ):
        """The reviewer's sequence on the SDK's own stdio server: a dynamic ``tool_filter``
        lists one tool to each agent, each listing replaced the other's, and the gate's
        own listing (with no agent) raised inside the filter."""
        pytest.importorskip("mcp.server.mcpserver", reason="the MCP SDK's server is not installed")
        import json
        import os

        from agents import RunContextWrapper
        from agents.mcp import MCPServerStdio

        script = tmp_path / "stdio_server.py"
        script.write_text(STDIO_SERVER, encoding="utf-8")
        server_log = tmp_path / "server.jsonl"
        env = {"RECUSAL_TEST_SERVER_LOG": str(server_log)}
        if "SYSTEMROOT" in os.environ:
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

        def per_agent(context, tool):
            return tool.name == ("count" if context.agent.name == "reader" else "scoped")

        policy = self.Recording()
        listed: dict = {}

        async def go():
            params = {"command": sys.executable, "args": [str(script)], "env": env}
            async with MCPServerStdio(
                params=params, client_session_timeout_seconds=60, tool_filter=per_agent
            ) as server:
                example.gate_mcp_server(server, policy)
                run_context = RunContextWrapper(context=None)
                for name in ("reader", "writer"):
                    agent = example.Agent(name=name)
                    tools = await server.list_tools(run_context, agent)
                    listed[name] = [tool.name for tool in tools]
                return [
                    await server.call_tool("count", {"n": 3}),
                    await server.call_tool("scoped", {"scope": "archive"}),
                    await server.call_tool("purge_opt", {}),
                ]

        results = asyncio.run(asyncio.wait_for(go(), 180))
        assert listed == {"reader": ["count"], "writer": ["scoped"]}
        for result in results:
            assert not result.model_dump(by_alias=True).get("isError"), result.content[0].text
        judged = [
            ("count", {"n": 3, "ratio": 1}),
            ("scoped", {"scope": "archive", "confirm": False}),
            ("purge_opt", {"path": "/", "recursive": True, "dry_run": None}),
        ]
        assert policy.seen == judged
        ran = [tuple(json.loads(line)) for line in server_log.read_text("utf-8").splitlines()]
        assert ran == judged

    def test_a_filter_that_cannot_run_without_an_agent_leaves_the_fetched_listing(self, example):
        """The gate asks for a listing itself, with no agent, when a tool is not in the
        one it holds. A dynamic filter raises then, after the SDK fetched the tools."""
        policy = self.Recording()
        server = example.gate_mcp_server(
            self.filtering_server(example, keeps_listing=True, fails_without_agent=True), policy
        )
        result = asyncio.run(server.call_tool("purge", {}))
        assert not result.model_dump(by_alias=True).get("isError")
        assert policy.seen == [("purge", {"path": "/", "recursive": True, "dry_run": False})]
        with pytest.raises(RuntimeError, match="needs an agent"):
            asyncio.run(server.list_tools())

    def test_a_server_object_that_keeps_no_listing_holds_the_latest_one_as_filtered(self, example):
        """The limit the README states: without ``cached_tools`` the latest listing, as
        filtered for one agent, replaces the one before."""
        from types import SimpleNamespace

        policy = self.Recording()
        server = example.gate_mcp_server(
            self.filtering_server(example, keeps_listing=False, fails_without_agent=True), policy
        )
        asyncio.run(server.list_tools(None, SimpleNamespace(name="writer")))
        asyncio.run(server.list_tools(None, SimpleNamespace(name="reader")))
        refused = asyncio.run(server.call_tool("purge", {}))
        assert refused.content[0].text.startswith(
            "Recusal refused `purge`: the server lists no tool of this name"
        )
        assert (
            not asyncio.run(server.call_tool("legacy", {})).model_dump(by_alias=True).get("isError")
        )

    def test_a_listing_that_fails_leaves_the_one_before(self, example):
        from mcp.types import Tool

        class Flaky(example.InProcessMCPServer):
            listings = 0

            async def list_tools(self, run_context=None, agent=None):
                self.listings += 1
                if self.listings > 1:
                    raise RuntimeError("connection lost")
                return [Tool(name="purge", description="Purge.", inputSchema=dict(PURGE_SCHEMA))]

        policy = self.Recording()
        server = example.gate_mcp_server(Flaky(name="files", tools=("purge",)), policy)
        asyncio.run(server.list_tools())
        with pytest.raises(RuntimeError, match="connection lost"):
            asyncio.run(server.list_tools())
        result = asyncio.run(server.call_tool("purge", {"dry_run": True}))
        assert not result.model_dump(by_alias=True).get("isError")
        assert policy.seen == [("purge", {"dry_run": True, "path": "/", "recursive": True})]

    @pytest.mark.parametrize("attribute", ["input_schema", "inputSchema"])
    def test_the_schema_is_read_under_either_name_the_sdk_has_given_it(self, example, attribute):
        from types import SimpleNamespace

        class Listing(example.InProcessMCPServer):
            async def list_tools(self, run_context=None, agent=None):
                return [SimpleNamespace(name="purge", **{attribute: dict(PURGE_SCHEMA)})]

        policy = self.Recording()
        server = example.gate_mcp_server(Listing(name="files", tools=("purge",)), policy)
        asyncio.run(server.call_tool("purge", {}))
        assert policy.seen == [("purge", {"path": "/", "recursive": True, "dry_run": False})]

    # --- hosted MCP: the tool's own callback may be a coroutine function -----------------------

    def test_a_hosted_callback_that_is_a_coroutine_function_is_awaited(self, example):
        from agents import HostedMCPTool

        async def existing(request):
            await asyncio.sleep(0)
            return {"approve": False, "reason": "the app said no"}

        hosted = HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "db",
                "server_url": "https://example.invalid/mcp",
            },
            on_approval_request=existing,
        )
        gated = example.gate_hosted_mcp(hosted, example.sql_scope_policy)
        request = example.approval_request("run_sql", {"sql": SELECT}, "r1")
        answer = asyncio.run(gated.on_approval_request(SimpleRequest(request)))
        assert answer == {"approve": False, "reason": "the app said no"}

    # --- what earlier reviews fixed and no test held --------------------------------------------

    @pytest.mark.parametrize(
        "listed, value",
        [
            ({"const": 1}, True),
            ({"const": 1}, 1.0),
            ({"const": True}, 1),
            ({"const": [1]}, [True]),
            ({"const": {"a": 1}}, {"a": 1.0}),
            ({"enum": [[1, 2]]}, [1, True]),
            ({"enum": [[1, 2]]}, [1.0, 2]),
            ({"enum": [{"a": 1}]}, {"a": True}),
            ({"enum": [{"a": [0]}]}, {"a": [False]}),
        ],
        ids=[
            "const-true",
            "const-float",
            "const-one",
            "const-in-a-list",
            "const-in-an-object",
            "enum-in-a-list",
            "enum-float-in-a-list",
            "enum-in-an-object",
            "enum-deep",
        ],
    )
    def test_a_fixed_or_listed_value_is_matched_by_json_type_all_the_way_down(
        self, example, listed, value
    ):
        schema = {"properties": {"mode": listed}}
        keyword = "the value the server's schema fixes" if "const" in listed else "one of"
        assert keyword in self.judged({"mode": value}, schema)[2]
        exact = listed.get("const", (listed.get("enum") or [None])[0])
        assert self.judged({"mode": exact}, schema) == ({"mode": exact}, {"mode": exact}, None)

    def test_a_list_under_alternatives_is_followed(self, example):
        schema = {
            "properties": {
                "targets": {"anyOf": [{"type": "array", "items": OPTIONS}, {"type": "null"}]}
            }
        }
        completed = {"targets": [FILLED, {"recursive": False, "dry_run": False}]}
        assert self.judged({"targets": [{}, {"recursive": False}]}, schema) == (
            completed,
            completed,
            None,
        )

    def test_additional_properties_complete_only_the_members_properties_does_not_name(
        self, example
    ):
        schema = {
            "properties": {
                "by": {
                    "type": "object",
                    "properties": {"plain": {"type": "object"}},
                    "additionalProperties": OPTIONS,
                }
            }
        }
        given = {"by": {"plain": {}, "other": {}}}
        completed = {"by": {"plain": {}, "other": FILLED}}
        assert self.judged(given, schema) == (completed, completed, None)
        # beside `patternProperties` the gate cannot tell which members are additional
        # without evaluating the patterns: a default there refuses the schema
        schema["properties"]["by"]["patternProperties"] = {"^x_": {"type": "object"}}
        assert self.judged(given, schema) == (
            None,
            None,
            f"Recusal refused `purge`: {_SCHEMA} lists a default under `additionalProperties` "
            "beside `patternProperties`, where the gate cannot tell whether the server would "
            "apply it",
        )

    @pytest.mark.parametrize(
        "name, ref",
        [
            ("a b", "#/$defs/a%20b"),
            ("a/b", "#/$defs/a~1b"),
            ("a~b", "#/$defs/a~0b"),
            ("a~1b", "#/$defs/a~01b"),
        ],
        ids=["percent-escaped", "a-slash", "a-tilde", "a-tilde-then-one"],
    )
    def test_a_ref_token_is_decoded_as_a_json_pointer(self, example, name, ref):
        schema = {"properties": {"opts": {"$ref": ref}}, "$defs": {name: OPTIONS}}
        assert self.judged({"opts": {}}, schema) == ({"opts": FILLED}, {"opts": FILLED}, None)
        assert "is of JSON type string" in self.judged({"opts": {"dry_run": "no"}}, schema)[2]

    def test_a_ref_can_step_through_a_list(self, example):
        schema = {
            "properties": {
                "first": {"anyOf": [OPTIONS, {"type": "null"}]},
                "second": {"$ref": "#/properties/first/anyOf/0"},
            }
        }
        completed = {"second": FILLED}
        assert self.judged({"second": {}}, schema) == (completed, completed, None)
        beyond = {
            "properties": {"a": {"$ref": "#/properties/first/anyOf/2"}, **schema["properties"]}
        }
        assert "has a `$ref` that does not resolve" in self.judged({}, beyond)[2]


class TestFourthReview:
    """The fourth review of 2026-11-22. The gate worked out from JSON Schema which of
    several object forms a server would run, and the SDKs do not follow JSON Schema there:
    Pydantic takes the member with more of the fields that are set and converts as it
    goes, zod the first member that parses, dropping what that member does not know. A
    server's regular expression ran on text the model chose, for as long as it took, and
    ``uniqueItems``, a long ``enum`` and a schema that reaches one subschema by many ways
    cost far more than the steps they were counted as."""

    example = TestWhatIsSentIsWhatWasJudged.example
    Recording = TestWhatIsSentIsWhatWasJudged.Recording
    stdio = TestARealServerRunsWhatWasJudged.stdio
    judged = TestTheWholeListedSchema.judged
    schema_module = staticmethod(TestThirdReview.schema_module)
    spent = TestThirdReview.spent

    BY_PATH = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "recursive": ZOD_FLAG},
        "required": ["path"],
        "additionalProperties": False,
    }
    BY_PATTERN = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string"},
            "dry_run": {"type": "boolean", "default": False},
        },
        "required": ["path", "pattern"],
        "additionalProperties": False,
    }

    # --- more than one object form, more than one list form -------------------------------------

    @pytest.mark.parametrize(
        "listed",
        [
            {"anyOf": [BY_PATH, BY_PATTERN]},
            {"oneOf": [BY_PATH, BY_PATTERN]},
            {"anyOf": [{"$ref": "#/$defs/ByPath"}, {"$ref": "#/$defs/ByPattern"}]},
            {"$ref": "#/$defs/Either"},
            {"allOf": [{"$ref": "#/$defs/Either"}]},
            {"anyOf": [{"$ref": "#/$defs/Either"}, {"type": "null"}]},
            {"anyOf": [BY_PATH, BY_PATTERN, {"type": "null"}, {"type": "string"}]},
            {"anyOf": [BY_PATTERN, {}]},
            {"anyOf": [BY_PATTERN, {"type": ["object", "null"]}]},
            {"anyOf": [BY_PATTERN, {"description": "anything"}]},
        ],
        ids=[
            "any-of",
            "one-of",
            "refs",
            "a-ref-to-alternatives",
            "all-of-over-a-ref",
            "alternatives-of-an-alternative",
            "beside-scalars",
            "beside-a-schema-that-says-nothing",
            "beside-a-typed-object",
            "beside-a-description",
        ],
    )
    def test_an_object_is_refused_where_the_schema_offers_more_than_one_object_form(
        self, example, listed
    ):
        """The reviewer's calls: each was judged with one member's values and ran with
        the other member's defaults."""
        schema = {
            "type": "object",
            "properties": {"target": listed},
            "$defs": {
                "ByPath": self.BY_PATH,
                "ByPattern": self.BY_PATTERN,
                "Either": {"anyOf": [self.BY_PATH, self.BY_PATTERN]},
            },
        }
        refused = (None, None, f"Recusal refused `purge`: argument `target` {_TWO_OBJECT_FORMS}")
        for target in (
            {"path": "/data", "pattern": "*.tmp", "dry_run": True},
            {"path": "/data", "pattern": "*.tmp"},
            {"path": "/data"},
        ):
            assert self.judged({"target": target}, schema) == refused

    def test_at_any_depth_and_for_the_arguments_themselves(self, example):
        either = {"anyOf": [self.BY_PATH, self.BY_PATTERN]}
        target = {"path": "/data", "pattern": "*"}
        for schema, arguments, place in (
            ({"properties": {"l": {"type": "array", "items": either}}}, {"l": [target]}, "l[0]"),
            (
                {"properties": {"m": {"type": "object", "additionalProperties": either}}},
                {"m": {"a": target}},
                "m.*",
            ),
            (
                {"properties": {"o": {"properties": {"t": either}}}},
                {"o": {"t": target}},
                "o.t",
            ),
            ({"properties": {"t": {"prefixItems": [{}, either]}}}, {"t": [1, target]}, "t[1]"),
        ):
            assert self.judged(arguments, schema)[2] == (
                f"Recusal refused `purge`: argument `{place}` {_TWO_OBJECT_FORMS}"
            )
        root = {"type": "object", "anyOf": [self.BY_PATH, self.BY_PATTERN]}
        assert self.judged(target, root)[2] == (
            "Recusal refused `purge`: the arguments are" + _TWO_OBJECT_FORMS[2:]
        )

    @pytest.mark.parametrize(
        "keyword, wrap",
        [
            ("then", lambda either: {"if": {}, "then": {"properties": {"t": either}}}),
            ("else", lambda either: {"if": False, "else": {"properties": {"t": either}}}),
            (
                "dependentSchemas",
                lambda either: {"dependentSchemas": {"t": {"properties": {"t": either}}}},
            ),
            (
                "dependencies",
                lambda either: {
                    "$schema": DRAFT_07,
                    "dependencies": {"t": {"properties": {"t": either}}},
                },
            ),
            ("patternProperties", lambda either: {"patternProperties": {"^t$": either}}),
            ("unevaluatedProperties", lambda either: {"unevaluatedProperties": either}),
        ],
        ids=lambda value: value if isinstance(value, str) else "",
    )
    def test_the_refusals_are_made_under_the_keywords_no_default_is_written_through(
        self, example, keyword, wrap
    ):
        def listing(inner):
            # a draft is declared by the schema of the arguments, never by a subschema
            wrapped = wrap(inner)
            declared = {"$schema": wrapped.pop("$schema")} if "$schema" in wrapped else {}
            return {**declared, "properties": {"o": {"type": "object", **wrapped}}}

        either = {"anyOf": [{"type": "object", "title": "a"}, {"type": "object"}]}
        schema = listing(either)
        assert self.judged({"o": {"t": {}}}, schema)[2] == (
            f"Recusal refused `purge`: argument `o.{'*' if keyword.endswith('Properties') else 't'}` "
            f"{_TWO_OBJECT_FORMS}"
        ), keyword
        texts = {"anyOf": [{"type": "integer"}, {"type": "string", "format": "date"}]}
        schema = listing(texts)
        assert _NUMBER_TEXT in self.judged({"o": {"t": "5"}}, schema)[2], keyword
        given = {"o": {"t": "2024-01-01"}}
        assert self.judged(given, schema) == (given, given, None)

    @pytest.mark.parametrize("keyword", ["contains", "unevaluatedItems"])
    def test_the_refusals_are_made_under_the_keywords_of_a_list_too(self, example, keyword):
        either = {"anyOf": [{"type": "object", "title": "a"}, {"type": "object"}]}
        schema = {"properties": {"l": {"type": "array", keyword: either}}}
        assert self.judged({"l": [{}]}, schema)[2] == (
            f"Recusal refused `purge`: argument `l[0]` {_TWO_OBJECT_FORMS}"
        )

    @staticmethod
    def tagged(first, second, keyword="anyOf", name="kind"):
        """A union of two forms, each with the property ``name`` as given."""
        return {
            "type": "object",
            "properties": {
                "rule": {
                    keyword: [
                        {
                            "type": "object",
                            "properties": {name: first, "days": {"type": "integer", "default": 30}},
                        },
                        {
                            "type": "object",
                            "properties": {name: second, "force": ZOD_FLAG},
                        },
                    ]
                }
            },
        }

    @pytest.mark.parametrize(
        "first, second",
        [
            ({"const": "keep"}, {"const": "drop"}),
            ({"const": "keep", "default": "keep"}, {"const": "drop", "default": "drop"}),
            ({"enum": ["keep"]}, {"enum": ["drop"]}),
            ({"const": 1}, {"const": 2}),
            ({"$ref": "#/$defs/Keep"}, {"allOf": [{"$ref": "#/$defs/Drop"}]}),
        ],
        ids=["const", "const-with-a-default", "an-enum-of-one", "numbers", "through-a-ref"],
    )
    @pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
    def test_a_tag_tells_the_forms_apart_and_only_its_form_is_completed(
        self, example, first, second, keyword
    ):
        schema = self.tagged(first, second, keyword)
        schema["$defs"] = {"Keep": {"const": "keep"}, "Drop": {"enum": ["drop"]}}
        keep, drop = (1, 2) if "const" in first and first["const"] == 1 else ("keep", "drop")
        completed = {"rule": {"kind": drop, "force": True}}
        assert self.judged({"rule": {"kind": drop}}, schema) == (completed, completed, None)
        completed = {"rule": {"kind": keep, "days": 30}}
        assert self.judged({"rule": {"kind": keep}}, schema) == (completed, completed, None)
        # without the tag the forms cannot be told apart, whatever else the object holds
        # (under `oneOf` such an object is valid against both, and refused for that first)
        for rule in ({}, {"force": False}, {"days": 1}) if keyword == "anyOf" else ():
            assert self.judged({"rule": rule}, schema)[2] == (
                f"Recusal refused `purge`: argument `rule` {_OBJECT_FORMS} The forms are "
                "told apart by `kind`, and the object carries the value of none of them "
                f"there.{_PUBLISH_A_TAG}"
            )

    @pytest.mark.parametrize(
        "first, second",
        [
            ({"const": "keep"}, {"const": "keep"}),
            ({"const": 1}, {"const": True}),
            ({"const": 1}, {"const": 1.0}),
            ({"const": "keep"}, {"type": "string"}),
            ({"const": "keep"}, {"enum": ["drop", "wipe"]}),
            ({"enum": []}, {"const": "drop"}),
        ],
        ids=[
            "one-tag",
            "one-and-true",
            "one-and-one-point-zero",
            "one-untagged",
            "two-values",
            "none",
        ],
    )
    def test_forms_that_no_property_fixes_apart_are_refused_with_the_tag_too(
        self, example, first, second
    ):
        schema = self.tagged(first, second)
        reasons = [self.judged({"rule": {"kind": k}}, schema)[2] for k in ("keep", "drop", 1, True)]
        refused = f"Recusal refused `purge`: argument `rule` {_TWO_OBJECT_FORMS}"
        # what neither form admits is not valid; what one does is still not told apart
        assert refused in reasons
        assert all(reason == refused or "matches none of" in reason for reason in reasons)

    def test_a_property_fixed_to_two_values_in_one_form_is_no_tag(self, example):
        schema = self.tagged({"const": "keep"}, {"const": "drop"})
        forms = schema["properties"]["rule"]["anyOf"]
        forms[0] = {"allOf": [forms[0], {"properties": {"kind": {"const": "other"}}}]}
        reason = self.judged({"rule": {"kind": "drop"}}, schema)[2]
        assert reason == f"Recusal refused `purge`: argument `rule` {_TWO_OBJECT_FORMS}"
        # fixed twice to one value it still is
        forms[0]["allOf"][1]["properties"]["kind"]["const"] = "keep"
        assert self.judged({"rule": {"kind": "drop"}}, schema)[2] is None

    def test_a_second_tag_serves_where_the_first_is_not_carried(self, example):
        schema = self.tagged({"const": "keep"}, {"const": "drop"})
        forms = schema["properties"]["rule"]["anyOf"]
        forms[0]["properties"]["v"] = {"const": 1}
        forms[1]["properties"]["v"] = {"const": 2}
        completed = {"rule": {"v": 2, "force": True}}
        assert self.judged({"rule": {"v": 2}}, schema) == (completed, completed, None)
        assert "told apart by `kind`, `v`" in self.judged({"rule": {"days": 3}}, schema)[2]

    def test_in_draft_07_a_const_beside_a_ref_is_not_a_tag(self, example):
        """Draft-07 reads a ``$ref`` alone: the ``const`` beside it is not applied, so it
        tells nothing apart."""
        schema = self.tagged(
            {"$ref": "#/definitions/Any", "const": "keep"},
            {"$ref": "#/definitions/Any", "const": "drop"},
        )
        schema["definitions"] = {"Any": {}}
        completed = {"rule": {"kind": "drop", "force": True}}
        assert self.judged({"rule": {"kind": "drop"}}, schema) == (completed, completed, None)
        schema["$schema"] = DRAFT_07
        assert _TWO_OBJECT_FORMS in self.judged({"rule": {"kind": "drop"}}, schema)[2]
        # nor are the properties beside the `$ref` of a form
        schema = self.tagged({"const": "keep"}, {"const": "drop"})
        schema["$schema"] = DRAFT_07
        schema["definitions"] = {"Object": {"type": "object"}}
        for form in schema["properties"]["rule"]["anyOf"]:
            form["$ref"] = "#/definitions/Object"
        assert _TWO_OBJECT_FORMS in self.judged({"rule": {"kind": "drop"}}, schema)[2]

    @pytest.mark.parametrize(
        "listed",
        [
            {"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "array"}]},
            {"oneOf": [{"type": "array", "maxItems": 1}, {"type": "array", "minItems": 2}]},
            {"anyOf": [{"type": "array"}, {}]},
            {"anyOf": [{"$ref": "#/$defs/L"}, {"$ref": "#/$defs/L"}, {"type": "null"}]},
        ],
        ids=["two-lists", "one-of", "beside-anything", "refs"],
    )
    def test_a_list_is_refused_where_the_schema_offers_more_than_one_list_form(
        self, example, listed
    ):
        schema = {"properties": {"l": listed}, "$defs": {"L": {"type": "array"}}}
        for value in ([], [1], [1, 2, 3]):
            assert self.judged({"l": value}, schema) == (
                None,
                None,
                f"Recusal refused `purge`: argument `l` {_TWO_LIST_FORMS}",
            )

    @pytest.mark.parametrize("nothing", [{"not": {}}, {"not": True}, False])
    def test_an_alternative_that_admits_nothing_is_not_a_form(self, example, nothing):
        """What zod publishes for ``z.object({...}).optional().default({})``: the object
        beside ``{"not": {}}``, the schema of a member that may be undefined."""
        inner = {"type": "object", "properties": {"f": ZOD_FLAG}, "additionalProperties": False}
        listed = {"anyOf": [{"anyOf": [nothing, inner], "default": {}}, {"type": "null"}]}
        schema = {"type": "object", "properties": {"p": listed, "l": {"anyOf": [nothing, {}]}}}
        completed = {"p": {"f": True}}
        assert self.judged({}, schema) == (completed, completed, None)
        assert self.judged({"p": {}}, schema) == (completed, completed, None)
        given = {"p": None, "l": [1]}
        assert self.judged(given, schema) == (given, given, None)
        # a `not` that says something admits what the gate does not work out: a form
        other = {"anyOf": [{"not": {"type": "string"}}, inner]}
        reason = self.judged({"p": {}}, {"properties": {"p": other}})[2]
        assert reason == f"Recusal refused `purge`: argument `p` {_TWO_OBJECT_FORMS}"

    def test_one_form_beside_what_is_not_one_is_walked_as_before(self, example):
        schema = {
            "properties": {
                "o": {"anyOf": [OPTIONS, {"type": "array", "items": OPTIONS}, {"type": "null"}]},
                "l": {"anyOf": [OPTIONS, {"type": "array", "items": OPTIONS}, {"type": "string"}]},
            }
        }
        completed = {"o": FILLED, "l": [FILLED]}
        assert self.judged({"o": {}, "l": [{}]}, schema) == (completed, completed, None)

    def test_a_default_where_the_schema_offers_two_forms_refuses_the_call(self, example):
        """Pydantic runs with such a default as listed; zod parses it again through the
        union, and its first member fills or drops members. The schema does not say."""
        either = {"anyOf": [self.BY_PATH, self.BY_PATTERN], "default": {"path": "/"}}
        lists = {"anyOf": [{"type": "array"}, {"type": "array", "maxItems": 3}], "default": []}
        deep = {"type": "object", "properties": {"t": either}, "default": {"t": {"path": "/"}}}
        for listed in (either, lists, deep):
            schema = {"properties": {"target": listed, "n": {"type": "integer", "default": 1}}}
            assert self.judged({}, schema) == (
                None,
                None,
                f"Recusal refused `purge`: argument `target` {_DEFAULT_OF_TWO_FORMS}",
            )
            assert self.judged({"target": None}, schema)[2] != self.judged({}, schema)[2]
        # a default that carries the tag of one form is completed as that form
        tagged = self.tagged({"const": "keep"}, {"const": "drop"})
        tagged["properties"]["rule"]["default"] = {"kind": "drop"}
        completed = {"rule": {"kind": "drop", "force": True}}
        assert self.judged({}, tagged) == (completed, completed, None)

    def test_on_a_real_server_an_object_for_a_union_of_two_models_never_runs(self, stdio):
        calls = [
            ("remove", {"target": {"path": "/data", "pattern": "*.tmp", "dry_run": True}}),
            ("remove", {"target": {"path": "/data"}}),
            ("retain", {"rule": {"days": 1}}),
            ("retain", {"rule": {"kind": "drop"}}),
            ("retain", {"rule": {"kind": "keep", "days": 7}}),
        ]
        found = stdio([calls])
        ran = [
            ("retain", {"rule": {"kind": "drop", "force": False}}),
            ("retain", {"rule": {"kind": "keep", "days": 7}}),
        ]
        assert _in_order(found.judged) == _in_order(found.wire) == _in_order(ran)
        assert _in_order(found.ran) == _in_order(ran)
        for number, name in ((1, "target"), (2, "target"), (3, "rule")):
            assert found.outputs[f"c{number}"].startswith(
                f"Recusal refused `{calls[number - 1][0]}`: argument `{name}` is an object, and "
                "the server's schema offers more than one object form there"
            )

    def test_negative_control_ungated_the_real_server_picks_a_member_by_its_own_rule(self, stdio):
        """The member with more of its fields set, whose default then runs unjudged."""
        calls = [
            ("remove", {"target": {"path": "/a", "recursive": False}}),
            ("remove", {"target": {"path": "/b", "pattern": "*.tmp"}}),
            ("remove", {"target": {"path": "/c"}}),
        ]
        found = stdio([calls], gated=False)
        assert _in_order(found.ran) == _in_order(
            [
                ("remove", {"target": {"path": "/a", "recursive": False}}),
                ("remove", {"target": {"path": "/b", "pattern": "*.tmp", "dry_run": False}}),
                ("remove", {"target": {"path": "/c", "recursive": True}}),
            ]
        )

    # --- a string a server reads as another type the schema admits ------------------------------

    INT_OR_DATE = {"anyOf": [{"type": "integer"}, {"type": "string", "format": "date"}]}

    @pytest.mark.parametrize(
        "text",
        [
            "5",
            " 5 ",
            "\t5\n",
            "+5",
            "-5",
            "05",
            "5.0",
            "5.5",
            ".5",
            "5.",
            "1e3",
            "1E-3",
            "1_000",
            "1_0e1_0",
            "1._5",
            "inf",
            "-Infinity",
            "NaN",
            "1" * 400,
            "\u00a05",
            "1__0",
            "_1",
            "0-5",
            "+0_-5",
            "--5",
        ],
    )
    def test_the_text_of_a_number_is_refused_where_a_number_is_listed_beside_string(
        self, example, text
    ):
        for listed in (
            self.INT_OR_DATE,
            {"type": ["number", "string"]},
            {"anyOf": [{"type": "number"}, {"type": "string", "pattern": "^x"}]},
            {"enum": [1, "a", text]},
            {"anyOf": [{"const": 1}, {"type": "string"}]},
            {"$ref": "#/$defs/Either"},
            {"allOf": [{"$ref": "#/$defs/Either"}]},
            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        ):
            schema = {"properties": {"when": listed}, "$defs": {"Either": self.INT_OR_DATE}}
            assert self.judged({"when": text}, schema) == (
                None,
                None,
                f"Recusal refused `purge`: argument `when` {_NUMBER_TEXT}",
            ), listed

    @pytest.mark.parametrize(
        "text",
        ["true", "False", "TRUE", "1", "0", "yes", "No", "on", "OFF", "t", "F", "y", "n", " on "],
    )
    def test_a_word_for_a_boolean_is_refused_where_a_boolean_is_listed_beside_string(
        self, example, text
    ):
        for listed in (
            {"anyOf": [{"type": "boolean"}, {"type": "string", "format": "path"}]},
            {"type": ["boolean", "string"]},
            {"enum": [True, text]},
        ):
            assert self.judged({"flag": text}, {"properties": {"flag": listed}}) == (
                None,
                None,
                f"Recusal refused `purge`: argument `flag` {_BOOLEAN_TEXT}",
            )

    @pytest.mark.parametrize(
        "text",
        [
            "five",
            "2024-01-01",
            "0x",
            "x10",
            "1,000",
            "5 5",
            "5-5",
            "e3",
            "tru",
            "of",
            "nul",
        ],
    )
    def test_any_other_text_is_judged_and_sent(self, example, text):
        schema = {
            "properties": {
                "when": self.INT_OR_DATE,
                "flag": {"type": ["boolean", "string", "number"]},
            }
        }
        given = {"when": text, "flag": text}
        assert self.judged(given, schema) == (given, given, None)

    def test_text_is_kept_where_the_schema_admits_only_strings_or_names_no_type(self, example):
        """A string parameter keeps ``"5"`` and ``"true"``, and so does one without an
        annotation, which a server does not convert."""
        schema = {
            "properties": {
                "s": {"type": "string"},
                "e": {"enum": ["1", "true"]},
                "n": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "any": {},
                "described": {"description": "anything"},
                "narrowed": {"allOf": [{"type": ["string", "integer"]}, {"type": "string"}]},
                "o": {"type": "object", "additionalProperties": True},
            }
        }
        for text in ("5", "true", "1", "yes"):
            given = {name: text for name in ("s", "n", "any", "described", "narrowed")}
            given.update(e="1" if text in "15" else "true", o={"k": text}, unlisted=text)
            assert self.judged(given, schema) == (given, given, None)
        # the number is named, and the boolean is not
        mixed = {"properties": {"v": {"type": ["integer", "string"]}}}
        assert self.judged({"v": "yes"}, mixed) == ({"v": "yes"}, {"v": "yes"}, None)
        mixed = {"properties": {"v": {"type": ["boolean", "string"]}}}
        assert self.judged({"v": "2"}, mixed) == ({"v": "2"}, {"v": "2"}, None)
        # `number` admits an integer, so a schema that narrows one by the other names both
        narrowed = {"allOf": [{"type": ["number", "string"]}, {"type": ["integer", "string"]}]}
        assert _NUMBER_TEXT in self.judged({"v": "5"}, {"properties": {"v": narrowed}})[2]

    def test_the_text_of_a_number_is_refused_at_every_depth(self, example):
        either = self.INT_OR_DATE
        for schema, arguments, place in (
            ({"properties": {"o": {"properties": {"w": either}}}}, {"o": {"w": "5"}}, "o.w"),
            ({"properties": {"l": {"items": either}}}, {"l": ["2024-01-01", "5"]}, "l[1]"),
            ({"properties": {"m": {"additionalProperties": either}}}, {"m": {"k": "5"}}, "m.*"),
            ({"properties": {"t": {"prefixItems": [{}, either]}}}, {"t": ["5", "5"]}, "t[1]"),
            (
                {"properties": {"o": {"anyOf": [{"properties": {"w": either}}, {"type": "null"}]}}},
                {"o": {"w": "5"}},
                "o.w",
            ),
            ({"additionalProperties": either}, {"k": "5"}, "*"),
            ({"allOf": [{"properties": {"w": either}}]}, {"w": "5"}, "w"),
        ):
            assert self.judged(arguments, schema)[2] == (
                f"Recusal refused `purge`: argument `{place}` {_NUMBER_TEXT}"
            )

    def test_json_text_is_refused_for_an_argument_however_the_schema_names_it(self, example):
        """Through an ``allOf`` or a ``$ref`` of the arguments' own schema too, which the
        third review's rule did not reach."""
        nullable = {"type": ["string", "null"]}
        for schema in (
            {"properties": {"s": nullable}},
            {"allOf": [{"properties": {"s": nullable}}]},
            {"$ref": "#/$defs/Args", "$defs": {"Args": {"properties": {"s": nullable}}}},
            {"additionalProperties": nullable},
            {"properties": {"s": {"oneOf": [{"type": "string"}, {"type": "null"}]}}},
        ):
            reason = self.judged({"s": "null"}, schema)[2]
            assert reason is not None and reason.endswith(_JSON_TEXT), schema
        # and further in it is the string it is, which no server decodes
        deep = {"properties": {"o": {"properties": {"s": nullable}}}}
        given = {"o": {"s": "null"}}
        assert self.judged(given, deep) == (given, given, None)

    def test_a_default_a_call_could_not_send_as_text_is_judged_and_not_sent(self, example):
        schema = {
            "properties": {
                "when": {**self.INT_OR_DATE, "default": "0"},
                "flag": {"type": ["boolean", "string"], "default": "yes"},
                "o": {
                    "type": "object",
                    "properties": {"when": {**self.INT_OR_DATE, "default": "7"}},
                    "default": {},
                },
                "day": {**self.INT_OR_DATE, "default": "2024-01-01"},
            }
        }
        assert self.judged({}, schema) == (
            {"when": "0", "flag": "yes", "o": {"when": "7"}, "day": "2024-01-01"},
            {"o": {}, "day": "2024-01-01"},
            None,
        )

    def test_on_a_real_server_text_that_would_run_as_a_number_or_a_boolean_never_runs(self, stdio):
        calls = [
            ("after", {"when": "1700000000"}),
            ("after", {"when": 5, "strict": "true"}),
            ("limited", {"limit": "500"}),
            ("after", {"when": "2024-01-01"}),
            ("after", {"when": 5, "strict": True}),
        ]
        found = stdio([calls])
        ran = [
            ("after", {"when": "2024-01-01", "strict": False}),
            ("after", {"when": 5, "strict": True}),
        ]
        assert _in_order(found.judged) == _in_order(found.wire) == _in_order(ran)
        assert _in_order(found.ran) == _in_order(
            [
                ("after", {"when": "datetime.date(2024, 1, 1)", "strict": "False"}),
                ("after", {"when": "5", "strict": "True"}),
            ]
        )
        assert found.outputs["c1"] == f"Recusal refused `after`: argument `when` {_NUMBER_TEXT}"
        assert found.outputs["c2"] == f"Recusal refused `after`: argument `strict` {_BOOLEAN_TEXT}"
        assert found.outputs["c3"] == f"Recusal refused `limited`: argument `limit` {_NUMBER_TEXT}"

    def test_negative_control_ungated_the_real_server_runs_that_text_as_a_number(self, stdio):
        found = stdio(
            [[("after", {"when": "1700000000", "strict": "true"}), ("limited", {"limit": "500"})]],
            gated=False,
        )
        assert _in_order(found.ran) == _in_order(
            [("after", {"when": "1700000000", "strict": "True"}), ("limited", {"limit": "500"})]
        ), "the integer and the boolean ran, not the strings `'1700000000'` and `'true'`"
        assert found.schemas == {} and found.judged == []

    # --- a default listed under names its own schema does not have -----------------------------

    def test_a_default_with_a_member_its_own_schema_does_not_allow_is_refused(self, example):
        model = {
            "type": "object",
            "properties": {"forceIt": {"type": "boolean", "default": True}},
            "additionalProperties": False,
        }
        for listed in (
            {**model, "default": {"force": True}},
            {"$ref": "#/$defs/M", "default": {"force": True}},
            {"anyOf": [{"$ref": "#/$defs/M"}, {"type": "null"}], "default": {"force": True}},
            {"oneOf": [{"$ref": "#/$defs/M"}, {"type": "null"}], "default": {"force": True}},
            {"type": "array", "items": model, "default": [{"force": True}]},
        ):
            schema = {"properties": {"opts": listed}, "$defs": {"M": model}}
            assert self.judged({}, schema) == (
                None,
                None,
                "Recusal refused `purge`: argument `opts` is left out, and the default the "
                "server's schema lists for it has a member its own schema does not allow "
                "there (a Pydantic model whose fields have aliases lists its default under "
                "other names than its schema), so the gate cannot tell what the server would "
                "run with",
            )
            given = {"opts": [] if "items" in listed else {"forceIt": False}}
            assert self.judged(given, schema)[2] is None
        # a default that fails its own schema another way is judged as listed, as before
        schema = {"properties": {"opts": {"anyOf": [model], "default": 5}}}
        assert self.judged({}, schema) == ({"opts": 5}, {}, None)

    def test_on_a_real_server_a_default_listed_under_an_alias_never_runs(self, stdio):
        found = stdio(
            [[("aliased", {}), ("aliased", {"opts": {"forceIt": False}})]], schemas=("aliased",)
        )
        assert found.schemas["aliased"]["opts"]["default"] == {"force": True}
        assert "has a member its own schema does not allow" in found.outputs["c1"]
        assert [name for name, _ in found.ran] == ["aliased"]
        assert found.judged == found.wire == [("aliased", {"opts": {"forceIt": False}})]

    # --- a server's regular expression is never evaluated ---------------------------------------

    def test_no_regular_expression_is_evaluated_for_a_call(self, example, monkeypatch):
        """The reviewer's pattern took eleven seconds on forty bytes. Nothing in ``re`` is
        called once the listing is read, whatever the schema's patterns."""
        import re

        slow = "^(\\w+\\s?)+$"
        schema = {
            "properties": {
                "name": {"type": "string", "pattern": slow},
                "by": {"type": "object", "patternProperties": {slow: {"type": "integer"}}},
                "keys": {"type": "object", "propertyNames": {"pattern": slow}},
                "rest": {
                    "type": "object",
                    "properties": {"a": {}},
                    "patternProperties": {slow: {}},
                    "additionalProperties": False,
                },
            }
        }
        text = "a" * 40 + "!"
        given = {"name": text, "by": {text: 1}, "keys": {text: 1}, "rest": {"a": 1, text: 2}}
        module = self.schema_module()
        listed = module.Listed(schema)
        listed.complete({})

        def evaluated(*args, **kwargs):
            raise AssertionError("a regular expression was evaluated")

        refused = None
        with monkeypatch.context() as patched:
            for name in ("compile", "search", "match", "fullmatch", "finditer", "findall", "sub"):
                patched.setattr(re, name, evaluated)
            completed = listed.complete(given)
            try:
                listed.complete({"by": {text: "1"}})
            except module.Refused as cause:
                refused = cause.reason
        assert completed == (given, given)
        assert refused is not None and refused.startswith("argument `by.*` is of JSON type string")

    def test_a_member_beside_patterns_must_meet_the_schema_of_every_pattern(self, example):
        """The gate does not work out which pattern a name matches: a member that meets
        every pattern's schema is valid whichever govern it."""
        schema = {
            "properties": {
                "limits": {
                    "type": "object",
                    "properties": {"max": {"type": "integer"}},
                    "patternProperties": {"^a": {"type": "integer"}, "^b": {"minimum": 0}},
                    "additionalProperties": {"maximum": 10},
                }
            }
        }
        given = {"limits": {"max": 1, "a": 2, "zz": 3}}
        assert self.judged(given, schema) == (given, given, None)
        for limits, why in (
            ({"a": "2"}, "is of JSON type string"),
            ({"zz": "2"}, "is of JSON type string"),
            ({"zz": -1}, "does not meet `minimum`"),
            ({"max": -1}, "does not meet `minimum`"),
            ({"zz": 11}, "does not meet `maximum`"),
        ):
            assert why in self.judged({"limits": limits}, schema)[2], limits
        # `properties` names `max`, so what governs additional members does not govern it
        assert self.judged({"limits": {"max": 11}}, schema)[2] is None
        # beside patterns `additionalProperties: false` asserts nothing: the server rejects
        # a member no pattern matches
        closed = {"patternProperties": {"^a": {"type": "integer"}}, "additionalProperties": False}
        assert self.judged({"zz": 1}, closed) == ({"zz": 1}, {"zz": 1}, None)
        assert "is of JSON type string" in self.judged({"zz": "1"}, closed)[2]
        assert "hold a member" in self.judged({"zz": 1}, {"additionalProperties": False})[2]
        # an empty `patternProperties` is none
        empty = {"patternProperties": {}, "additionalProperties": False}
        assert "hold a member" in self.judged({"zz": 1}, empty)[2]

    def test_unevaluated_properties_beside_patterns_refuse_the_schema(self, example):
        schema = {
            "properties": {"o": {"patternProperties": {"^a": {}}}},
            "unevaluatedProperties": False,
        }
        assert self.judged({}, schema) == (
            None,
            None,
            f"Recusal refused `purge`: {_SCHEMA} uses `unevaluatedProperties` together with "
            "`patternProperties`, which the gate could apply only by evaluating the server's "
            "patterns",
        )
        for alone in ({"unevaluatedProperties": False}, {"patternProperties": {"^a": {}}}):
            assert self.judged({}, alone) == ({}, {}, None)
        named = {"properties": {"patternProperties": {}}, "unevaluatedProperties": False}
        assert self.judged({}, named) == ({}, {}, None), "a property of that name is no keyword"
        empty = {"patternProperties": {}, "unevaluatedProperties": False}
        assert self.judged({}, empty) == ({}, {}, None), "no pattern is listed"

    def test_on_a_real_server_a_member_named_by_a_pattern_is_checked_without_the_pattern(
        self, stdio
    ):
        calls = [
            ("keyed", {"limits": {"cpu": 2}}),
            ("keyed", {"limits": {"cpu": "2"}}),
            ("keyed", {"limits": {"CPU": 2}}),
        ]
        found = stdio([calls], schemas=("keyed",))
        assert "patternProperties" in found.schemas["keyed"]["limits"]
        assert found.ran == [("keyed", {"limits": {"cpu": 2}})]
        assert _in_order(found.judged) == _in_order([calls[0], calls[2]]), "the server rejects one"
        assert "is of JSON type string" in found.outputs["c2"]

    # --- the work is counted --------------------------------------------------------------------

    @pytest.mark.parametrize(
        "items",
        [
            [1, 1],
            [1, 1.0],
            [1, True],
            [0, False],
            ["a", "a"],
            [None, None],
            [[1, 2], [1.0, 2]],
            [{"a": 1, "b": [True]}, {"b": [1], "a": 1.0}],
        ],
    )
    def test_unique_items_counts_equal_numbers_of_any_type_as_one_item(self, example, items):
        """A server that makes a set of them keeps one: ``{1, 1.0, True}`` holds one."""
        schema = {"properties": {"tags": {"type": "array", "uniqueItems": True}}}
        assert self.judged({"tags": items}, schema)[2] == (
            "Recusal refused `purge`: argument `tags` does not meet `uniqueItems` in the "
            f"server's schema; {_AS_THEY_STAND}"
        )
        loose = {"properties": {"tags": {"type": "array", "uniqueItems": False}}}
        assert self.judged({"tags": items}, loose)[2] is None

    def test_unique_items_keeps_what_differs(self, example):
        schema = {"properties": {"tags": {"type": "array", "uniqueItems": True}}}
        given = {"tags": [1, "1", 2, None, [1], [1, 1], {"a": 1}, {"a": 2}, {"b": 1}, [], {}]}
        assert self.judged(given, schema) == (given, given, None)

    def test_unique_items_and_a_long_enum_cost_in_proportion_to_the_arguments(self, example):
        """The reviewer's cases: 12,000 items took fifteen seconds, and 20,000 values
        against 600 choices two and a half, each counted as a few steps."""
        unique = {"properties": {"tags": {"type": "array", "uniqueItems": True}}}
        models = {
            "properties": {
                "rules": {"type": "array", "uniqueItems": True, "items": {"type": "object"}}
            }
        }
        choices = [f"zone-{number}" for number in range(600)]
        zones = {"properties": {"tz": {"type": "array", "items": {"enum": choices}}}}
        for schema, make in (
            (unique, lambda count: {"tags": [str(number) for number in range(count)]}),
            (
                models,
                lambda count: {"rules": [{"id": number, "on": True} for number in range(count)]},
            ),
            (zones, lambda count: {"tz": [choices[number % 600] for number in range(count)]}),
        ):
            costs = {}
            for count in (1000, 2000, 4000):
                steps, allowed, result = self.spent(make(count), schema)
                assert type(result) is tuple, result
                assert steps * 20 < allowed
                costs[count] = steps
            assert costs[4000] - costs[2000] == 2 * (costs[2000] - costs[1000])
            assert costs[4000] < 20 * 4000

    def test_the_choices_of_an_enum_are_read_once_for_a_call(self, example):
        choices = list(range(600))
        schema = {"properties": {"l": {"type": "array", "items": {"enum": choices}}}}
        one, _, _ = self.spent({"l": [1]}, schema)
        two, _, _ = self.spent({"l": [1, 2]}, schema)
        none, _, _ = self.spent({"l": []}, schema)
        assert one - none > 600 and two - one < 20

    def test_a_subschema_reached_by_many_ways_is_read_once(self, example):
        """The reviewer's schema: every level an ``allOf`` of the next level twice. The
        types it admits were read once for every way down: nine seconds at twenty levels."""
        module = self.schema_module()
        schema = {"type": "object", "properties": {"v": {"$ref": "#/$defs/d0"}}, "$defs": {}}
        for level in range(60):
            ref = {"$ref": f"#/$defs/d{level + 1}"}
            schema["$defs"][f"d{level}"] = {"allOf": [ref, dict(ref)]}
        schema["$defs"]["d60"] = {"type": ["string", "integer"], "default": "x"}
        read = []
        original = module.Listed._types

        def counting(listed, node):
            read.append(1)
            assert len(read) < 2000, "a subschema is read again for every way to it"
            return original(listed, node)

        module.Listed._types = counting
        try:
            listed = module.Listed(schema)
            assert listed.complete({}) == ({"v": "x"}, {"v": "x"})
            prepared = len(read)
            with pytest.raises(module.Refused, match="holds the text of a number"):
                listed.complete({"v": "5"})
            assert len(read) - prepared < 10
        finally:
            module.Listed._types = original
        steps, allowed, result = self.spent({"v": "ok"}, schema)
        assert result == ({"v": "ok"}, {"v": "ok"}) and steps < 1000

    def test_an_untagged_recursive_union_is_refused_for_its_forms_at_a_cost_in_proportion(
        self, example
    ):
        """The third review's schema, ``where: Union[And, Or, Leaf]`` with no tag: two
        alternatives accept every level, so validating it cost twice as much with each
        level until an alternative's answer was kept. It is validated before it is
        refused, and still costs in proportion."""
        import copy

        untagged = copy.deepcopy(EXPRESSION)
        for form in ("And", "Or", "Leaf"):
            del untagged["$defs"][form]["properties"]["op"]
        costs = {}
        for depth in (4, 8, 16, 31):
            value: dict = {"args": []}
            for _ in range(depth):
                value = {"args": [value]}
            steps, allowed, result = self.spent({"where": value}, untagged)
            assert result == f"argument `where` {_TWO_OBJECT_FORMS}"
            assert steps * 20 < allowed
            costs[depth] = steps
        assert costs[16] - costs[8] == 2 * (costs[8] - costs[4]), "each level costs the same"
        # and one that fails at the bottom, which two alternatives find out at every level
        failing = {}
        for depth in (4, 8, 16, 31):
            value = {"args": 5}
            for _ in range(depth):
                value = {"args": [value]}
            steps, allowed, result = self.spent({"where": value}, untagged)
            assert result == (
                "argument `where` matches none of the alternatives the server's schema lists "
                f"there; {_AS_THEY_STAND}"
            )
            assert steps * 20 < allowed
            failing[depth] = steps
        assert failing[16] - failing[8] == 2 * (failing[8] - failing[4])

    def test_a_tag_behind_many_ways_is_looked_for_once(self, example):
        """Reading the listing ends for a property that reaches one subschema by a
        million ways and fixes no value there."""
        module = self.schema_module()
        schema = self.tagged({"$ref": "#/$defs/d0"}, {"const": "drop"})
        schema["$defs"] = {"d40": {"type": "string"}}
        for level in range(40):
            ref = {"$ref": f"#/$defs/d{level + 1}"}
            schema["$defs"][f"d{level}"] = {"allOf": [ref, dict(ref)]}
        read = []
        original = module.Listed._fixed

        def counting(listed, node, seen):
            read.append(1)
            assert len(read) < 2000, "a subschema is read again for every way to it"
            return original(listed, node, seen)

        module.Listed._fixed = counting
        try:
            reason = self.judged({"rule": {"kind": "drop"}}, schema)[2]
            assert reason == f"Recusal refused `purge`: argument `rule` {_TWO_OBJECT_FORMS}"
            schema["$defs"]["d40"] = {"const": "keep"}
            completed = {"rule": {"kind": "drop", "force": True}}
            assert self.judged({"rule": {"kind": "drop"}}, schema) == (completed, completed, None)
        finally:
            module.Listed._fixed = original

    def test_a_list_or_an_object_reached_by_many_ways_is_walked_once(self, example):
        schema = {"type": "object", "properties": {"v": {"$ref": "#/$defs/d0"}}, "$defs": {}}
        for level in range(30):
            ref = {"$ref": f"#/$defs/d{level + 1}"}
            schema["$defs"][f"d{level}"] = {"allOf": [ref, dict(ref)]}
        schema["$defs"]["d30"] = {"type": "object", "properties": {"keep": ZOD_FLAG}}
        steps, allowed, result = self.spent({"v": {}}, schema)
        assert result == ({"v": {"keep": True}}, {"v": {"keep": True}}) and steps < 2000

    def test_what_the_schema_alone_says_is_read_with_the_listing_and_not_for_a_call(self, example):
        """So the same call costs the same steps whether it is the first or the hundredth."""
        module = self.schema_module()
        made: list = []
        original = module._Work.__init__

        def remembering(work, allowed):
            original(work, allowed)
            made.append(work)

        module._Work.__init__ = remembering
        try:
            listed = module.Listed(EXPRESSION)
            for _ in range(3):
                listed.complete(_nested(6, {"args": []}))
        finally:
            module._Work.__init__ = original
        assert len({work.steps for work in made}) == 1
        fresh, _, _ = self.spent(_nested(6, {"args": []}), EXPRESSION)
        assert made[0].steps == fresh

    def test_nothing_is_read_from_the_schema_alone_once_the_listing_is_read(self, example):
        module = self.schema_module()
        schema = {
            "properties": {
                "o": {"anyOf": [OPTIONS, {"type": "null"}]},
                "l": {"anyOf": [{"type": "array", "items": OPTIONS}, {"type": "null"}]},
                "s": {"type": ["string", "integer"]},
            }
        }
        listed = module.Listed(schema)
        listed._prepared()
        read = (len(listed._typed), len(listed._defaults), len(listed._unions))
        assert read == (10, 10, 4)
        listed.complete({"o": {}, "l": [{}], "s": "x"})
        assert (len(listed._typed), len(listed._defaults), len(listed._unions)) == read

    def test_a_keyword_costs_a_step_more_for_each_member_or_item_of_its_value(self, example):
        schema = {"type": "object", "properties": {"l": {"type": "array"}}}
        costs = [self.spent({"l": [0] * count}, schema)[0] for count in (0, 1, 10, 110)]
        # the two keywords of the list (the walk goes through it once, each item one step)
        assert costs[1] - costs[0] == costs[2] - costs[1] - 8 * 2 == 2
        assert costs[3] - costs[2] == 100 * 2
        wide = {"type": "object", "additionalProperties": True}
        costs = [
            self.spent({str(number): 0 for number in range(count)}, wide)[0] for count in (0, 10)
        ]
        assert costs[1] - costs[0] == 10 * 2 * 2, "two keywords, each applied twice"

    def test_the_place_of_every_validation_error_is_counted(self, example):
        schema = {"properties": {"l": {"type": "array", "items": {"type": "integer"}}}}
        one, _, _ = self.spent({"l": ["x"] + [1] * 49}, schema)
        fifty, _, reason = self.spent({"l": ["x"] * 50}, schema)
        assert reason.startswith("argument `l[0]` is of JSON type string")
        assert fifty - one == 49 * 2, "two parts of the place of each further error"

    def test_reading_a_value_for_a_comparison_is_counted(self, example):
        fixed = {"properties": {"c": {"const": [0] * 5}}}
        small, _, _ = self.spent({"c": [0] * 5}, fixed)
        large, _, reason = self.spent({"c": [0] * 505}, fixed)
        assert "not the value the server's schema fixes" in reason
        assert large - small >= 500

    def test_what_was_worked_out_is_forgotten_once_a_default_is_written(self, example):
        """Whether a list is valid against an alternative is kept while the arguments
        stand: a default written into one of its items is a new question."""
        item = {
            "type": "object",
            "properties": {"k": {"type": "integer", "default": 5}},
            "not": {"required": ["k"]},
        }
        schema = {
            "properties": {
                "l": {"oneOf": [{"type": "array", "items": item}, {"type": "string"}]},
            }
        }
        assert self.judged({"l": [{}]}, schema)[2] == (
            "Recusal refused `purge`: argument `l` matches not exactly one of the alternatives "
            f"the server's schema lists there; {_AS_THEY_STAND}"
        )
        # and what an object is compared by is read again
        fixed = {
            "properties": {
                "o": {
                    "type": "object",
                    "properties": {"k": {"type": "integer", "default": 5}},
                    "enum": [{}, {"k": 6}],
                }
            }
        }
        assert "is not one of the values" in self.judged({"o": {}}, fixed)[2]
        unique = {
            "properties": {
                "l": {
                    "type": "array",
                    "uniqueItems": True,
                    "prefixItems": [{"properties": {"k": {"default": 5}}}, {}],
                }
            }
        }
        assert "does not meet `uniqueItems`" in self.judged({"l": [{}, {"k": 5}]}, unique)[2]

    # --- JSON Schema where no server's own rule is in play --------------------------------------

    def test_one_of_is_exactly_one(self, example):
        schema = {"properties": {"v": {"oneOf": [{"type": "integer"}, {"type": "number"}]}}}
        assert self.judged({"v": 1.5}, schema) == ({"v": 1.5}, {"v": 1.5}, None)
        reason = (
            "Recusal refused `purge`: argument `v` matches not exactly one of the alternatives "
            f"the server's schema lists there; {_AS_THEY_STAND}"
        )
        assert self.judged({"v": 1}, schema)[2] == reason, "valid against both"
        bounded = {"properties": {"v": {"oneOf": [{"maximum": 1}, {"minimum": 3}]}}}
        assert self.judged({"v": 2}, bounded)[2] == reason, "valid against neither"

    def test_three_alternatives_two_of_one_default_and_a_third_of_another_are_refused(
        self, example
    ):
        listed = {"anyOf": [{"default": 1}, {"default": 1}, {"default": 2}]}
        assert "list different defaults" in self.judged({}, {"properties": {"p": listed}})[2]
        nested = {"anyOf": [{"default": 1}, {"anyOf": [{"default": 1}, {"default": 2}]}]}
        assert "list different defaults" in self.judged({}, {"properties": {"p": nested}})[2]
        assert self.judged({"p": 3}, {"properties": {"p": nested}})[2] is None

    @pytest.mark.parametrize(
        "schema",
        [
            {"properties": {"a": {"if": {}, "then": {"$ref": "#/properties/a"}}}},
            {"properties": {"a": {"if": False, "else": {"$ref": "#/properties/a"}}}},
            {
                "$schema": DRAFT_07,
                "properties": {"a": {"dependencies": {"x": {"$ref": "#/properties/a"}}}},
            },
            {"properties": {"a": {"dependentSchemas": {"x": {"$ref": "#/properties/a"}}}}},
        ],
        ids=["then", "else", "draft-07-dependencies", "dependentSchemas"],
    )
    def test_a_schema_that_leads_back_to_itself_through_a_condition_is_refused(
        self, example, schema
    ):
        assert self.judged({"a": {"x": 1}}, schema) == (
            None,
            None,
            f"Recusal refused `purge`: {_SCHEMA} refers to itself without end",
        )

    def test_a_default_list_that_leaves_out_an_item_with_a_default_refuses_the_call(self, example):
        """Written, it would be a list the next pass cannot complete: not one to hold."""
        schema = {
            "properties": {
                "t": {
                    "type": "array",
                    "prefixItems": [{"type": "integer"}, {"type": "integer", "default": 0}],
                    "default": [1],
                }
            }
        }
        assert (
            "leaves out an item the server's schema lists a default for"
            in (self.judged({}, schema)[2])
        )

    def test_one_object_in_two_places_is_completed_in_both(self, example):
        """The walk knows a place by the object that is there, so the arguments are
        copied into ones in which no object stands twice."""
        module = self.schema_module()
        shared: dict = {}
        schema = {"properties": {"a": OPTIONS, "b": OPTIONS}}
        judged, sent = module.Listed(schema).complete({"a": shared, "b": shared})
        assert judged == sent == {"a": FILLED, "b": FILLED}
        assert sent["a"] is not sent["b"] and shared == {}

    def test_what_is_judged_and_not_sent_is_not_left_in_what_is_sent(self, example):
        module = self.schema_module()
        schema = {
            "properties": {
                "q": {"type": "string", "default": None},
                "o": {"type": "object", "properties": {"q": {"type": "string", "default": None}}},
            }
        }
        judged, sent = module.Listed(schema).complete({"o": {}})
        assert judged == {"q": None, "o": {"q": None}} and sent == {"o": {}}
        judged["o"]["extra"] = 1
        assert sent == {"o": {}}, "the two share nothing"

    def test_a_server_object_whose_listing_cannot_be_read_keeps_none(self, example):
        """``cached_tools`` that raises is a server object that keeps no listing."""
        from mcp.types import Tool

        class Raising(example.InProcessMCPServer):
            @property
            def cached_tools(self):
                raise RuntimeError("not connected")

            async def list_tools(self, run_context=None, agent=None):
                return [Tool(name="purge", description="Purge.", inputSchema=dict(PURGE_SCHEMA))]

        policy = self.Recording()
        server = example.gate_mcp_server(Raising(name="files", tools=("purge",)), policy)
        result = asyncio.run(server.call_tool("purge", {}))
        assert not result.model_dump(by_alias=True).get("isError")
        assert policy.seen == [("purge", {"path": "/", "recursive": True, "dry_run": False})]


def _leaves(key):
    """Everything a comparison key holds that is not a tuple or a frozenset."""
    if type(key) in (tuple, frozenset):
        for part in key:
            yield from _leaves(part)
    else:
        yield key


class TestFifthReview:
    """The fifth review, of the 0.2.0 release. Integers chosen to share one hash turned the set
    ``uniqueItems`` is checked with into a scan the allowance did not count, and so did
    lists of small integers; ``unevaluatedProperties`` and ``unevaluatedItems`` compared
    every member with every other, counted as one step each; the reason for a string a
    server may read as another type told the model to send JSON where the text was not
    that value; and a zod ``z.coerce.number()`` beside a string reads text by
    JavaScript's rules, which the text rule did not know."""

    example = TestWhatIsSentIsWhatWasJudged.example
    Recording = TestWhatIsSentIsWhatWasJudged.Recording
    stdio = TestARealServerRunsWhatWasJudged.stdio
    judged = TestTheWholeListedSchema.judged
    schema_module = staticmethod(TestThirdReview.schema_module)
    spent = TestThirdReview.spent

    #: Every multiple of this has the hash 0: Python reduces an integer by it.
    ONE_HASH = 2**61 - 1
    UNIQUE = {"properties": {"ids": {"type": "array", "uniqueItems": True}}}
    NOT_UNIQUE = (
        "Recusal refused `purge`: argument `ids` does not meet `uniqueItems` in the server's "
        f"schema; {_AS_THEY_STAND}"
    )
    TOO_COSTLY = _over_each_value()
    UNEVALUATED = {
        "properties": {
            "o": {
                "type": "object",
                "allOf": [{"additionalProperties": True}],
                "unevaluatedProperties": False,
            },
            "l": {
                "type": "array",
                "allOf": [{"items": {"type": "number"}}],
                "unevaluatedItems": False,
            },
        }
    }

    # --- uniqueItems: no value the sender chooses makes the set a scan --------------------------

    def test_integers_of_one_hash_are_compared_by_keys_of_different_hashes(self, example):
        """The reviewer's case: 10,000 integers of one hash held the event loop for
        seconds. What they are compared by now hashes apart, alone and inside a list."""
        module = self.schema_module()
        integers = [number * self.ONE_HASH for number in range(1, 2001)]
        assert len({hash(number) for number in integers}) == 1
        assert len({hash(module._key(None, number, False)) for number in integers}) == 2000
        assert len({hash(module._key(None, [number, 1], False)) for number in integers}) == 2000
        given = {"ids": integers}
        assert self.judged(given, self.UNIQUE) == (given, given, None)
        steps, _, _ = self.spent(given, self.UNIQUE)
        ordinary, _, _ = self.spent({"ids": list(range(2000))}, self.UNIQUE)
        assert steps == ordinary

    def test_a_key_that_is_gathered_into_a_set_holds_no_number(self, example):
        """A small integer hashes to itself and the hash of a tuple has no secret in it,
        so lists of small integers can be chosen to share one hash too. Every number is
        held as text, whose hash the interpreter keys."""
        module = self.schema_module()
        value = [0, -1, 1.5, True, 2**70, -0.0, 1e300, {"a": [3, 4.0], "b": {"c": 7}}, "5", None]
        leaves = list(_leaves(module._key(None, value, False)))
        assert [leaf for leaf in leaves if type(leaf) in (int, float, bool)] == []
        assert module._key(None, [1, 1.0, True, 0.5], False) == (
            list,
            ((float, "0x1"), (float, "0x1"), (float, "0x1"), (float, "0x1.0000000000000p-1")),
        )
        # an exact key is never gathered: it keeps the number, and its JSON type
        assert module._key(None, [1, 1.0, True], True) == (
            list,
            ((int, 1), (float, 1.0), (bool, True)),
        )

    @pytest.mark.parametrize(
        "items",
        [
            [2**70, float(2**70)],
            [0, -0.0],
            [False, -0.0],
            [0.5, 0.5],
            [1e300, int(1e300)],
            [-(2**64), -float(2**64)],
            [[2**70, 1], [float(2**70), True]],
            [{"a": 2**61 - 1}, {"a": 2**61 - 1}],
        ],
    )
    def test_equal_numbers_of_any_size_are_still_one_item(self, example, items):
        assert self.judged({"ids": items}, self.UNIQUE)[2] == self.NOT_UNIQUE

    @pytest.mark.parametrize(
        "items",
        [
            [2**70, 2**70 + 1],
            [float(2**70), 2**70 + 1],
            [2**61 - 1, 2 * (2**61 - 1), 0],
            [-1, -2],
            [0.5, 1.5, -0.5],
            [0.1, 0.1 + 2**-55],
            [2**53, 2**53 + 1],
            [1, "0x1", "1"],
            [0.5, "0x1.0000000000000p-1"],
            [[1, 2], [2, 1], [1], [1, 2, 0]],
        ],
    )
    def test_numbers_that_differ_are_still_different_items(self, example, items):
        given = {"ids": items}
        assert self.judged(given, self.UNIQUE) == (given, given, None)

    def test_a_list_is_read_once_for_a_comparison_while_the_arguments_stand(self, example):
        """A list inside an item is read for the item's key, and again where the item's
        own ``uniqueItems`` compares it: the second reading is one step, not its length."""
        once = {"properties": {"l": {"type": "array", "uniqueItems": True}}}
        inner = {"type": "array", "uniqueItems": True}
        twice = {"properties": {"l": {"type": "array", "uniqueItems": True, "items": inner}}}

        def grows(schema):
            """The steps one hundred more numbers in the innermost list cost."""
            longer, _, _ = self.spent({"l": [[list(range(200))]]}, schema)
            shorter, _, _ = self.spent({"l": [[list(range(100))]]}, schema)
            return longer - shorter

        assert grows(twice) == grows(once) == 100

    # --- unevaluatedProperties and unevaluatedItems cost the square of what they read -----------

    @pytest.mark.parametrize(
        "draft",
        [
            "https://json-schema.org/draft/2020-12/schema",
            "https://json-schema.org/draft/2019-09/schema",
        ],
    )
    @pytest.mark.parametrize(
        "make",
        [
            lambda count: {"o": {f"k{number}": 1 for number in range(count)}},
            lambda count: {"l": [1] * count},
        ],
        ids=["unevaluatedProperties", "unevaluatedItems"],
    )
    def test_an_unevaluated_keyword_is_counted_by_the_pairs_it_compares(self, example, draft, make):
        """The reviewer's case: 30,000 members took seconds and were counted as a few
        steps each. ``jsonschema`` looks every member up in a list of the members other
        keywords evaluated, so the square of their number is counted before it runs."""
        schema = {"$schema": draft, **self.UNEVALUATED}
        for count in (100, 400):
            steps, allowed, result = self.spent(make(count), schema)
            assert type(result) is tuple, result
            assert count**2 < steps < count**2 + 30 * count
        steps, allowed, result = self.spent(make(2000), schema)
        assert result == self.TOO_COSTLY and steps > allowed

    def test_an_unevaluated_keyword_costs_nothing_more_for_a_value_it_does_not_read(self, example):
        members = {f"k{number}": 1 for number in range(2000)}
        for listed, value in (
            ({"unevaluatedProperties": False}, list(range(2000))),
            ({"unevaluatedItems": False}, members),
            ({"unevaluatedProperties": False, "unevaluatedItems": False}, "x" * 2000),
        ):
            steps, _, result = self.spent({"v": value}, {"properties": {"v": listed}})
            assert type(result) is tuple and steps < 30 * 2000
        # draft-07 has neither keyword: it is not applied, and not counted
        old = {"$schema": DRAFT_07, **self.UNEVALUATED}
        steps, _, result = self.spent({"o": {f"k{number}": 1 for number in range(2000)}}, old)
        assert type(result) is tuple and steps < 30 * 2000

    def test_the_allowance_is_the_one_the_readme_states_and_no_larger(self, example):
        """1,000 steps for every value: under ``unevaluatedProperties`` members cost
        their square, and from 142 members that is more, whatever the ceiling: the call
        is refused before the policy runs."""
        module = self.schema_module()
        assert (module._STEPS, module._STEPS_PER_VALUE) == (10_000, 1_000)
        readme = (ROOT / "adapters" / "openai-agents-mcp" / "README.md").read_text(encoding="utf-8")
        assert (
            f"an allowance of {module._STEPS:,} steps and {module._STEPS_PER_VALUE:,} more for "
            "every value"
        ) in " ".join(readme.split())
        given = {"o": {f"k{number}": 1 for number in range(1500)}}
        steps, allowed, result = self.spent(given, self.UNEVALUATED)
        assert allowed == 10_000 + 1_000 * 1502
        assert steps > allowed and result == self.TOO_COSTLY
        # at the default ceiling 150 members are allowed 162,000 steps, under it
        few = {"o": {f"k{number}": 1 for number in range(150)}}
        assert self.spent(few, self.UNEVALUATED, module.STEP_CEILING)[1] == 162_000
        assert self.judged(few, self.UNEVALUATED) == (few, few, None)
        many = {"o": {f"k{number}": 1 for number in range(420)}}
        steps, allowed, result = self.spent(many, self.UNEVALUATED, module.STEP_CEILING)
        assert allowed == 200_000 and 420**2 < steps < allowed
        assert self.judged(many, self.UNEVALUATED) == (many, many, None)
        assert self.judged(given, self.UNEVALUATED) == (
            None,
            None,
            f"Recusal refused `purge`: {_OVER_THE_CEILING}",
        )

    # --- the reason for text a server may read as another type ----------------------------------

    @pytest.mark.parametrize(
        "listed, text, reason",
        [
            ({"type": ["integer", "string"]}, "007", _NUMBER_TEXT),
            ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, "01234", _NUMBER_TEXT),
            ({"anyOf": [{"type": "string"}, {"type": "boolean"}]}, "no", _BOOLEAN_TEXT),
            ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "null", _JSON_TEXT),
            ({"type": ["number", "string"]}, "0x1F", _JAVASCRIPT_NUMBER_TEXT),
        ],
        ids=["an-identifier", "a-postal-code", "a-word", "json-text", "javascript"],
    )
    def test_the_reason_for_text_names_both_ways_out(self, example, listed, text, reason):
        """The text is not always the other type written wrongly: ``"007"`` is not 7. The
        reason says how to send the other type, and what lets the text itself through:
        the parameter declared a string only, which this test then does."""
        assert self.judged({"p": text}, {"properties": {"p": listed}}) == (
            None,
            None,
            f"Recusal refused `purge`: argument `p` {reason}",
        )
        assert "Send the value as JSON" not in reason
        assert (
            "the server's author can declare this parameter a string only (`str`, or "
            "`z.string()` with zod)"
        ) in reason
        given = {"p": text}
        assert self.judged(given, {"properties": {"p": {"type": "string"}}}) == (given, given, None)

    def test_on_a_real_server_the_text_runs_once_the_parameter_is_a_string_only(self, stdio):
        """``ident: Union[int, str]`` cannot be sent ``"01234"``; ``name: str`` can."""
        found = stdio(
            [[("find", {"ident": "01234"}), ("find", {"ident": 5, "name": "01234"})]],
            schemas=("find",),
        )
        assert found.schemas["find"]["ident"]["anyOf"] == [{"type": "integer"}, {"type": "string"}]
        assert found.schemas["find"]["name"]["type"] == "string"
        assert found.outputs["c1"] == f"Recusal refused `find`: argument `ident` {_NUMBER_TEXT}"
        sent = [("find", {"ident": 5, "name": "01234"})]
        assert found.judged == found.wire == found.ran == sent

    # --- text JavaScript reads as a number ------------------------------------------------------

    @pytest.mark.parametrize(
        "text",
        [
            "",
            " ",
            "\t\n",
            chr(0xFEFF),
            "0x10",
            "0X1f",
            " 0b101 ",
            "0o17",
            "0O17",
            "-0x1F",
            "0x1_0",
            chr(0xFEFF) + "5",
            "5" + chr(0xFEFF) + " ",
        ],
    )
    def test_text_javascript_reads_as_a_number_is_refused_beside_a_number(self, example, text):
        """A zod ``z.union([z.coerce.number(), z.string()])`` lists ``number`` and
        ``string`` and runs ``Number()`` on the text: ``"0x1F"`` ran as 31 and ``""`` as 0."""
        for listed in (
            {"type": ["number", "string"]},
            {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            TestFourthReview.INT_OR_DATE,
        ):
            assert self.judged({"p": text}, {"properties": {"p": listed}}) == (
                None,
                None,
                f"Recusal refused `purge`: argument `p` {_JAVASCRIPT_NUMBER_TEXT}",
            )
        for kept in ({"type": "string"}, {}, {"type": ["boolean", "string"]}):
            given = {"p": text}
            assert self.judged(given, {"properties": {"p": kept}}) == (given, given, None)

    def test_the_white_space_dropped_is_pythons_and_the_byte_order_mark(self, example):
        module = self.schema_module()
        python = {chr(code) for code in range(sys.maxunicode + 1) if chr(code).isspace()}
        assert set(module._SPACE) == python | {chr(0xFEFF)}
        assert len(module._SPACE) == len(python) + 1

    def test_a_listed_default_javascript_reads_as_a_number_is_judged_and_not_sent(self, example):
        schema = {"properties": {"port": {"type": ["integer", "string"], "default": ""}}}
        assert self.judged({}, schema) == ({"port": ""}, {}, None)

    # --- what the findings asked to be pinned ---------------------------------------------------

    def test_a_default_with_a_member_unevaluated_properties_forbids_refuses_the_call(self, example):
        """As for ``additionalProperties: false``: the default is listed under other names
        than its schema, so what the server would run with cannot be told."""
        model = {
            "type": "object",
            "properties": {"forceIt": {"type": "boolean"}},
            "unevaluatedProperties": False,
        }
        for listed in (
            {**model, "default": {"force": True}},
            {"anyOf": [model, {"type": "null"}], "default": {"force": True}},
        ):
            assert self.judged({}, {"properties": {"opts": listed}}) == (
                None,
                None,
                "Recusal refused `purge`: argument `opts` is left out, and the default the "
                "server's schema lists for it has a member its own schema does not allow "
                "there (a Pydantic model whose fields have aliases lists its default under "
                "other names than its schema), so the gate cannot tell what the server would "
                "run with",
            )
            given = {"opts": {"forceIt": False}}
            assert self.judged(given, {"properties": {"opts": listed}}) == (given, given, None)

    @pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
    def test_a_hand_written_choice_of_required_lists_is_two_object_forms(self, example, keyword):
        """That one of ``a`` and ``b`` is required, written as alternatives that each
        name one, is two alternatives that admit an object: the object is refused
        whatever it holds. Written as a condition it is one form, and the call is judged."""
        members = {"a": {"type": "string"}, "b": {"type": "string"}}
        either = {
            "type": "object",
            "properties": members,
            keyword: [{"required": ["a"]}, {"required": ["b"]}],
        }
        for arguments in ({"a": "x"}, {"b": "x"}):
            assert self.judged(arguments, either) == (
                None,
                None,
                "Recusal refused `purge`: the arguments are" + _TWO_OBJECT_FORMS[2:],
            )
            assert self.judged({"t": arguments}, {"properties": {"t": either}}) == (
                None,
                None,
                f"Recusal refused `purge`: argument `t` {_TWO_OBJECT_FORMS}",
            )
        condition = {
            "type": "object",
            "properties": members,
            "if": {"not": {"required": ["a"]}},
            "then": {"required": ["b"]},
        }
        for arguments in ({"a": "x"}, {"b": "x"}, {"a": "x", "b": "y"}):
            assert self.judged(arguments, condition) == (arguments, arguments, None)
        assert self.judged({}, condition) == (
            None,
            None,
            "Recusal refused `purge`: the arguments leave out `b`, which the server's schema "
            f"requires; {_AS_THEY_STAND}",
        )

    # --- what the README lists as not seen, on a real server ------------------------------------

    def test_what_a_pydantic_server_makes_of_keys_ignored_members_and_ordered_unions(self, stdio):
        """The README's entries for the fifth review, kept true by a test: the policy
        judged and the gate sent exactly these arguments, and the server ran others."""
        calls = [
            ("set_quotas", {"quotas": {"1": 5, "01": 1000000}}),
            ("set_marks", {"marks": {"true": "a", "yes": "b", "0": "c"}}),
            ("clean", {"job": {"path": "/data", "dry_run": True}}),
            (
                "remove_at",
                {"target": {"dry_run": True, "path": "/data", "options": {"dry_run": False}}},
            ),
            ("expire", {"when": 0}),
        ]
        found = stdio([calls], schemas=("set_quotas", "clean", "remove_at", "expire"))
        assert found.schemas["set_quotas"]["quotas"] == {
            "additionalProperties": {"type": "integer"},
            "title": "Quotas",
            "type": "object",
        }, "the key type is not listed"
        assert found.schemas["expire"]["when"]["anyOf"] == [
            {"format": "date", "type": "string"},
            {"type": "number"},
        ]
        assert _in_order(found.judged) == _in_order(found.wire) == _in_order(calls)
        assert _in_order(found.ran) == _in_order(
            [
                ("set_quotas", {"quotas": {"1": 1000000}}),
                ("set_marks", {"marks": {"true": "b", "false": "c"}}),
                ("clean", {"path": "/data", "dry_run": False}),
                ("remove_at", {"target": {"dry_run": False, "path": "/data"}}),
                ("expire", {"when": "1970-01-01"}),
            ]
        )


class TestSixthReview:
    """The sixth review, of the fixes to the fifth. JSON text nested deeper than the gate's
    own interpreter decodes, or holding an integer of more digits than it reads, was
    judged a string, and a server under another interpreter decoded it and ran a list
    or an object; the steps allowed for each value did not bound the time a call takes,
    which grew with what the model sent; the reason for an integer too large for a
    float blamed the server's schema; and a Pydantic server converts a number into a
    date, a time or a boolean in a default-mode union, which the page did not say."""

    example = TestWhatIsSentIsWhatWasJudged.example
    Recording = TestWhatIsSentIsWhatWasJudged.Recording
    stdio = TestARealServerRunsWhatWasJudged.stdio
    judged = TestTheWholeListedSchema.judged
    schema_module = staticmethod(TestThirdReview.schema_module)
    spent = TestThirdReview.spent

    #: More levels than any released interpreter decodes (3.15 reads about 13,850).
    DEEPEST = 200_000
    #: Schemas of a parameter that admits another JSON type than string.
    OPEN = [
        {},
        {"anyOf": [{"type": "string"}, {"type": "array"}]},
        {"anyOf": [{"type": "string"}, {"type": "object"}]},
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
    ]
    INTEGERS = {"properties": {"p": {"type": "array", "items": {"type": "integer"}}}}

    # --- JSON text this interpreter gives up on, which another decodes --------------------------

    def test_the_premise_this_interpreter_gives_up_on_json_text_for_its_own_limits(self):
        """Neither error says the text is not JSON: ``json.loads`` raises them for JSON a
        later interpreter, or one with the digit limit switched off, decodes."""
        with pytest.raises(RecursionError):
            json.loads("[" * self.DEEPEST + "]" * self.DEEPEST)
        with _digit_limit(640):
            with pytest.raises(ValueError) as raised:
                json.loads("[" + "9" * 641 + "]")
            assert not isinstance(raised.value, json.JSONDecodeError)
            assert json.loads("[" + "9" * 640 + "]") == [int("9" * 640)]

    @pytest.mark.parametrize(
        "depth", [5, 600, 900, 992, 1200, 2500, 2990, 2997, 3100, 11_326, 13_850, 20_000, DEEPEST]
    )
    @pytest.mark.parametrize("listed", OPEN, ids=[str(n) for n in range(len(OPEN))])
    def test_json_text_nested_to_any_depth_is_refused(self, example, listed, depth):
        """The reviewer's case: nested 1,200 deep, the text was judged a string by a gate
        on Python 3.10 and ran as a list on a server on 3.12. Whatever this interpreter's
        limit is, the text is refused below it (it decodes) and above it (it does not).
        The deepest of these are longer than the ceiling on the size of one call, which
        refuses them first, so that ceiling is raised here."""
        for text in (
            "[" * depth + '"rm -rf /"' + "]" * depth,
            '{"a":' * depth + "1" + "}" * depth,
            " \n" + "[" * depth + "]" * depth,
        ):
            schema = {"properties": {"payload": listed}}
            assert self.judged({"payload": text}, schema, size_ceiling=10**9) == (
                None,
                None,
                f"Recusal refused `purge`: argument `payload` {_JSON_TEXT}",
            )
            if len(text) > 1_000_000:
                assert self.judged({"payload": text}, schema) == (
                    None,
                    None,
                    f"Recusal refused `purge`: {_TOO_LARGE}",
                )

    @pytest.mark.parametrize("digits", [100, 640, 641, 4300, 4301, 6000, 100_000])
    @pytest.mark.parametrize("limit", [640, 4300, 0])
    def test_json_text_that_holds_an_integer_of_any_length_is_refused(self, example, limit, digits):
        """The second limit: a list holding an integer of 4,301 digits was judged a string,
        and a server without the digit limit ran a list. Refused under every limit this
        interpreter can be given, and under none."""
        number = "9" * digits
        with _digit_limit(limit):
            for text in (f"[{number}]", f'{{"a": {number}}}', f"\t[1, [2, -{number}]]"):
                for listed in self.OPEN:
                    schema = {"properties": {"payload": listed}}
                    assert self.judged({"payload": text}, schema) == (
                        None,
                        None,
                        f"Recusal refused `purge`: argument `payload` {_JSON_TEXT}",
                    )

    @pytest.mark.parametrize("digits", [100, 4300, 4301, 6000, 100_000])
    @pytest.mark.parametrize("limit", [640, 4300, 0])
    def test_a_long_run_of_digits_alone_is_judged_and_sent_as_the_string_it_is(
        self, example, limit, digits
    ):
        """Text that opens neither a list nor an object could decode only to a number,
        which the MCP Python SDK's server keeps a string, so its length refuses nothing."""
        number = "9" * digits
        with _digit_limit(limit):
            for text in (number, f"-{number}", f" {number}\n", f'"{number}"', f"{number}]"):
                for listed in (self.OPEN[0], self.OPEN[3]):
                    given = {"payload": text}
                    schema = {"properties": {"payload": listed}}
                    assert self.judged(given, schema) == (given, given, None)

    def test_text_that_is_not_json_is_still_judged_as_sent(self, example):
        for text in ("[draft] notes", "{braces} are fine", "[1, 2", "{", "[9" + "x" * 6000):
            given = {"payload": text}
            assert self.judged(given, {"properties": {"payload": {}}}) == (given, given, None)

    def test_negative_control_the_sdks_server_decodes_what_this_interpreter_would_not(self):
        """The server's side, under the other limit: with no digit limit the MCP Python
        SDK decodes the text into a list, which is what ran."""
        from mcp.server.mcpserver.tools import Tool

        def store(payload: Union[str, list, None]) -> str:
            return ""

        text = "[" + "9" * 641 + "]"
        with _digit_limit(640):
            kept = Tool.from_function(store).fn_metadata.pre_parse_json({"payload": text})
            assert kept == {"payload": text}
            assert self.schema_module()._json_text(text) is True
        with _digit_limit(0):
            ran = Tool.from_function(store).fn_metadata.pre_parse_json({"payload": text})
        assert ran == {"payload": [int("9" * 641)]}

    def test_on_a_real_server_json_text_past_the_limits_never_runs(self, stdio):
        deep = 20_000
        calls = [
            ("either", {"v": '{"a":' * deep + "1" + "}" * deep}),
            ("scoped", {"scope": "[" * deep + '"rm -rf /"' + "]" * deep}),
            ("either", {"v": '{"a": ' + "9" * 6000 + "}"}),
            ("scoped", {"scope": "[" + "9" * 6000 + "]"}),
        ]
        found = stdio([calls])
        assert found.ran == [] and found.wire == [] and found.judged == []
        for number, (_, arguments) in enumerate(calls, 1):
            (name,) = arguments
            assert found.outputs[f"c{number}"] == (
                f"Recusal refused `{calls[number - 1][0]}`: argument `{name}` {_JSON_TEXT}"
            )
        # the digits alone are a string for both, and run as the string judged
        sent = [("scoped", {"scope": "9" * 6000, "confirm": False})]
        found = stdio([[("scoped", {"scope": "9" * 6000})]])
        assert found.judged == found.wire == found.ran == sent

    # --- a ceiling on the steps of one call -----------------------------------------------------

    def test_the_ceiling_is_the_one_the_readme_states(self, example):
        module = self.schema_module()
        package = importlib.import_module("recusal_openai_agents_mcp")
        assert module.STEP_CEILING == package.STEP_CEILING == 200_000
        assert "STEP_CEILING" in package.__all__
        readme = (ROOT / "adapters" / "openai-agents-mcp" / "README.md").read_text(encoding="utf-8")
        assert (
            f"up to a ceiling of {module.STEP_CEILING:,} steps for one call "
            "(`recusal_openai_agents_mcp.STEP_CEILING`)"
        ) in " ".join(readme.split())

    def test_the_allowance_grows_with_the_values_up_to_the_ceiling_and_no_further(self, example):
        module = self.schema_module()
        for count, allowed in ((1, 13_000), (100, 112_000), (188, 200_000), (50_000, 200_000)):
            given = {"p": list(range(count))}
            assert self.spent(given, self.INTEGERS, module.STEP_CEILING)[1] == allowed
        assert self.spent({"p": [1] * 500}, self.INTEGERS, 300_000)[1] == 300_000
        assert self.spent({"p": [1] * 500}, self.INTEGERS, 600_000)[1] == 512_000

    def test_a_call_over_the_ceiling_is_refused_before_the_policy_and_says_how_to_raise_it(
        self, example
    ):
        """Four steps for each integer: 49,998 of them are the last the default ceiling
        completes."""
        module = self.schema_module()
        most = {"p": list(range(49_998))}
        steps, allowed, result = self.spent(most, self.INTEGERS, module.STEP_CEILING)
        assert (steps, allowed) == (200_000, 200_000) and type(result) is tuple
        assert self.judged(most, self.INTEGERS) == (most, most, None)
        over = {"p": list(range(49_999))}
        steps, allowed, result = self.spent(over, self.INTEGERS, module.STEP_CEILING)
        assert allowed == 200_000 < steps <= 200_004 and result == _OVER_THE_CEILING
        assert self.judged(over, self.INTEGERS) == (
            None,
            None,
            f"Recusal refused `purge`: {_OVER_THE_CEILING}",
        )
        for name in ("step_ceiling=", "gate_mcp_server", "gated_agent", "audited_agent"):
            assert name in _OVER_THE_CEILING

    def test_the_ceiling_stops_a_schema_that_costs_a_thousand_steps_for_a_value(self, example):
        """The reviewer's case: a hand-written schema that reaches one subschema 4,096
        ways. 1,000 short lists (6.9 KB) ran six seconds, to 2,012,002 steps, before
        they were refused; they are now stopped at the ceiling."""
        schema = {
            "properties": {"p": {"type": "array", "items": {"$ref": "#/$defs/a0"}}},
            "$defs": {"a12": {"type": "integer"}},
        }
        for level in range(12):
            onward = {"$ref": f"#/$defs/a{level + 1}"}
            schema["$defs"][f"a{level}"] = {"allOf": [onward, dict(onward)]}
        given = {"p": [[number] for number in range(1000)]}
        steps, allowed, result = self.spent(given, schema, self.schema_module().STEP_CEILING)
        assert allowed == 200_000 < steps <= 200_002 and result == _OVER_THE_CEILING
        # without one, the steps allowed for each value are what stops it, ten times later
        steps, allowed, result = self.spent(given, schema)
        assert allowed == 2_012_000 and steps > allowed and result == _over_each_value()

    def test_the_reason_says_which_of_the_two_limits_stopped_the_check(self, example):
        """Raising the ceiling helps only where the ceiling was the limit, so only that
        reason says to."""
        module = self.schema_module()
        assert module._too_costly(200_000, 200_000) == _OVER_THE_CEILING
        assert module._too_costly(250_000, 200_000) == _OVER_THE_CEILING
        assert module._too_costly(199_999, 200_000) == _over_each_value()
        assert module._too_costly(700, 700) == _OVER_THE_CEILING.replace("200,000", "700")
        assert "can raise the ceiling" not in _over_each_value()

    def _storing(self, example, schema):
        """An in-process server that lists one tool, ``store``, and keeps what it ran."""
        import copy

        from mcp.types import CallToolResult, TextContent, Tool

        class Storing(example.InProcessMCPServer):
            async def list_tools(self, run_context=None, agent=None):
                return [Tool(name="store", description="d", inputSchema=copy.deepcopy(schema))]

            async def call_tool(self, tool_name, arguments, meta=None):
                self.ran.append((tool_name, arguments))
                return CallToolResult(content=[TextContent(type="text", text="ran")])

        server = Storing(name="files")
        server.ran = []
        return server

    @staticmethod
    def _call(server, arguments):
        return asyncio.run(server.call_tool("store", arguments)).content[0].text

    def test_an_application_raises_the_ceiling_for_a_connection(self, example):
        over = {"p": list(range(60_000))}
        seen = []

        def policy(name, arguments):
            seen.append(len(arguments["p"]))
            return []

        default = example.gate_mcp_server(self._storing(example, self.INTEGERS), policy)
        assert default.call_tool.recusal_step_ceiling == 200_000
        assert self._call(default, over) == f"Recusal refused `store`: {_OVER_THE_CEILING}"
        assert default.ran == [] and seen == []
        raised = example.gate_mcp_server(
            self._storing(example, self.INTEGERS), policy, step_ceiling=400_000
        )
        assert raised.call_tool.recusal_step_ceiling == 400_000
        assert self._call(raised, over) == "ran"
        assert raised.ran == [("store", over)] and seen == [60_000]
        # and lowers it: the reason then names the ceiling that was set
        lowered = example.gate_mcp_server(
            self._storing(example, self.INTEGERS), policy, step_ceiling=100
        )
        assert self._call(lowered, {"p": list(range(30))}) == (
            f"Recusal refused `store`: {_OVER_THE_CEILING.replace('200,000', '100')}"
        )
        assert self._call(lowered, {"p": [1, 2]}) == "ran"
        assert lowered.ran == [("store", {"p": [1, 2]})]

    def test_gated_agent_and_audited_agent_hand_the_ceiling_to_every_server(
        self, example, tmp_path
    ):
        from agents import Agent

        from recusal.audit import AuditLog

        package = importlib.import_module("recusal_openai_agents_mcp")
        first = self._storing(example, self.INTEGERS)
        second = self._storing(example, self.INTEGERS)
        agent = Agent(name="a", mcp_servers=[first, second])
        gated = package.gated_agent(agent, lambda *_: [], step_ceiling=300_000)
        assert package.is_gated(gated)
        assert [s.call_tool.recusal_step_ceiling for s in gated.mcp_servers] == [300_000] * 2
        assert self._call(first, {"p": list(range(60_000))}) == "ran"
        third = self._storing(example, self.INTEGERS)
        log = AuditLog(path=str(tmp_path / "decisions.jsonl"))
        package.audited_agent(
            Agent(name="b", mcp_servers=[third]), lambda *_: [], log, step_ceiling=7
        )
        assert third.call_tool.recusal_step_ceiling == 7
        plain = self._storing(example, self.INTEGERS)
        package.audited_agent(Agent(name="c", mcp_servers=[plain]), lambda *_: [], log)
        assert plain.call_tool.recusal_step_ceiling == package.STEP_CEILING

    def test_a_connection_carries_one_ceiling(self, example):
        from agents import Agent

        def policy(name, arguments):
            return []

        server = example.gate_mcp_server(self._storing(example, self.INTEGERS), policy)
        gated_once = server.call_tool
        assert example.gate_mcp_server(server, policy, step_ceiling=200_000) is server
        assert server.call_tool is gated_once
        with pytest.raises(ValueError, match="already gated with a different step_ceiling"):
            example.gate_mcp_server(server, policy, step_ceiling=400_000)
        with pytest.raises(ValueError, match="already gated with a different step_ceiling"):
            example.gated_agent(Agent(name="a", mcp_servers=[server]), policy, step_ceiling=1)
        assert server.call_tool is gated_once
        assert server.call_tool.recusal_step_ceiling == 200_000

    @pytest.mark.parametrize("ceiling", [0, -1, True, 1.5, 200_000.0, "200000", None])
    def test_a_ceiling_that_is_not_a_whole_number_of_steps_is_refused(self, example, ceiling):
        from agents import Agent

        module = self.schema_module()
        server = self._storing(example, self.INTEGERS)
        told = "step_ceiling is the most steps the gate spends checking the arguments of one call"
        with pytest.raises(ValueError, match=told):
            example.gate_mcp_server(server, lambda *_: [], step_ceiling=ceiling)
        assert not example.is_gated_mcp_server(server)
        with pytest.raises(ValueError, match=told):
            example.gated_agent(Agent(name="a"), lambda *_: [], step_ceiling=ceiling)
        with pytest.raises(ValueError, match=told):
            module.Listed({}, ceiling)

    # --- an integer too large for a float -------------------------------------------------------

    NO_FLOAT_SENT = (
        "the arguments hold an integer too large to be a float (more than about 308 digits) "
        "where the server's schema lists a `multipleOf` that is a fraction, which is checked "
        "by dividing the two as floats, so the gate cannot check it. Send a number a float "
        "can hold."
    )

    @pytest.mark.parametrize("listed", [{"type": "number", "multipleOf": 0.5}, {"multipleOf": 0.1}])
    def test_an_integer_too_large_for_a_float_is_refused_for_what_it_is(self, example, listed):
        """The reviewer's case: the reason said the server's schema could not be applied,
        for an argument the model chose."""
        schema = {"properties": {"amount": listed, "rows": {"type": "array", "items": listed}}}
        for given in ({"amount": 10**309}, {"amount": -(10**400)}, {"rows": [1, 2**1024]}):
            assert self.judged(given, schema) == (
                None,
                None,
                f"Recusal refused `purge`: {self.NO_FLOAT_SENT}",
            )
        for given in ({"amount": 10**308}, {"amount": 2**1023}, {"rows": [1.5, 3]}):
            assert self.judged(given, schema)[2] != f"Recusal refused `purge`: {self.NO_FLOAT_SENT}"
        if listed["multipleOf"] == 0.5:
            whole = {"amount": 10**308}
            assert self.judged(whole, schema) == (whole, whole, None)

    def test_a_multiple_too_large_for_a_float_is_the_schemas_and_the_reason_says_so(self, example):
        schema = {"properties": {"amount": {"type": "number", "multipleOf": 10**400}}}
        assert self.judged({"amount": 1.5}, schema) == (
            None,
            None,
            f"Recusal refused `purge`: {_SCHEMA} lists a `multipleOf` too large to be a float, "
            "which the gate cannot check a number that is not an integer against",
        )
        whole = {"amount": 10**400}
        assert self.judged(whole, schema) == (whole, whole, None)

    @pytest.mark.parametrize(
        "listed",
        [
            {"type": "number", "multipleOf": 0.5, "default": 10**309},
            {"type": "number", "multipleOf": 10**400, "default": 1.5},
            {
                "anyOf": [{"type": "number", "multipleOf": 0.5}, {"type": "null"}],
                "default": 10**309,
            },
        ],
    )
    def test_a_listed_default_too_large_for_a_float_is_not_blamed_on_the_arguments(
        self, example, listed
    ):
        schema = {"properties": {"amount": listed, "path": {"type": "string"}}}
        assert self.judged({"path": "/"}, schema) == (
            None,
            None,
            "Recusal refused `purge`: argument `amount` is left out, and the default the "
            "server's schema lists for it cannot be checked against a `multipleOf` listed "
            "there, one of the two being too large to be a float, so the gate cannot tell "
            "what the server would run with",
        )
        given = {"path": "/", "amount": 3 if listed.get("multipleOf") != 10**400 else 10**400}
        assert self.judged(given, schema) == (given, given, None)

    def test_an_error_the_gate_does_not_know_blames_neither_side(self, example, monkeypatch):
        module = self.schema_module()

        def broken(*_):
            raise KeyError("MARKER")

        monkeypatch.setattr(module.Listed, "_complete", broken)
        assert self.judged({"path": "/"}, {}) == (
            None,
            None,
            f"Recusal refused `purge`: the arguments could not be checked against {_SCHEMA} "
            "(KeyError)",
        )

    # --- what a Pydantic server makes of a number, which the page now says ----------------------

    def test_on_a_real_server_a_number_runs_as_a_date_a_time_or_a_boolean(self, stdio):
        """Not closed, stated: in a default-mode union Pydantic converts a number into
        the first member that takes it. The policy judges the number it was given."""
        calls = [
            ("stamp", {"at": 5}),
            ("stamp", {"at": 1.5}),
            ("forced", {"force": 1}),
            ("forced", {"force": 0}),
            ("forced", {"force": 2}),
            ("levelled", {"n": True}),
        ]
        found = stdio([calls], schemas=("stamp", "forced", "levelled"))
        assert found.schemas["stamp"]["at"]["anyOf"] == [
            {"format": "date-time", "type": "string"},
            {"type": "number"},
            {"type": "string"},
        ]
        assert found.schemas["forced"]["force"]["anyOf"] == [
            {"type": "boolean"},
            {"type": "number"},
            {"type": "string"},
        ]
        assert found.schemas["levelled"]["n"]["anyOf"] == [
            {"enum": [1, 2], "type": "integer"},
            {"type": "boolean"},
        ]
        assert _in_order(found.judged) == _in_order(found.wire) == _in_order(calls)
        assert _in_order(found.ran) == _in_order(
            [
                ("stamp", {"at": "datetime 1970-01-01 00:00:05+00:00"}),
                ("stamp", {"at": "datetime 1970-01-01 00:00:01.500000+00:00"}),
                ("forced", {"force": "bool True"}),
                ("forced", {"force": "bool False"}),
                ("forced", {"force": "Decimal 2"}),
                ("levelled", {"n": "int 1"}),
            ]
        )

    def test_on_a_real_server_a_string_with_a_format_runs_normalised(self, stdio):
        """Not closed, stated: a ``format`` is not asserted, and the type behind it
        rewrites the text."""
        calls = [
            ("visit", {"url": "HTTP://X/a/../admin"}),
            ("opened", {"at": ""}),
            ("ident", {"key": "{12345678-1234-5678-1234-56781234ABCD}"}),
        ]
        found = stdio([calls], schemas=("visit", "opened", "ident"))
        formats = {
            name: next(iter(found.schemas[name].values()))["format"] for name in found.schemas
        }
        assert formats == {"visit": "uri", "opened": "path", "ident": "uuid"}
        assert _in_order(found.judged) == _in_order(found.wire) == _in_order(calls)
        assert _in_order(found.ran) == _in_order(
            [
                ("visit", {"url": "http://x/admin"}),
                ("opened", {"at": "."}),
                ("ident", {"key": "12345678-1234-5678-1234-56781234abcd"}),
            ]
        )

    def test_the_other_conversions_the_page_names_are_pydantics_own(self):
        """The cases the page names beside the ones the stdio server runs."""
        import datetime
        from decimal import Decimal
        from enum import IntEnum
        from typing import Literal

        from pydantic import TypeAdapter

        class Level(IntEnum):
            low = 1
            high = 2

        def ran(kind, value):
            result = TypeAdapter(kind).validate_python(value)
            return type(result), result

        assert ran(Union[datetime.date, Decimal], 0) == (datetime.date, datetime.date(1970, 1, 1))
        assert ran(Union[datetime.time, Decimal], 5)[0] is datetime.time
        assert ran(Union[datetime.datetime, Level], 1)[0] is datetime.datetime
        assert ran(Union[Decimal, datetime.datetime], 5) == (Decimal, Decimal(5))
        assert ran(Union[bool, Level], 1) == (bool, True)
        assert ran(Union[bool, Level], 2) == (Level, Level.high)
        assert ran(Union[Literal[True], int], 1) == (bool, True)
        assert ran(Union[Literal[True], float], 1) == (bool, True)
        # a number that is exactly the type of a member runs as that member
        assert ran(Union[datetime.date, float], 0) == (float, 0.0)
        assert ran(Union[bool, int], 1) == (int, 1)

    def test_the_page_says_what_the_gate_does_not_see_of_a_number(self):
        readme = " ".join(
            (ROOT / "adapters" / "openai-agents-mcp" / "README.md")
            .read_text(encoding="utf-8")
            .split()
        )
        for stated in (
            "**A number a Pydantic server reads as a date, a time, an address, a complex "
            "number or an enum member.**",
            "**`1`, `0`, `1.0` and `0.0` that run as booleans, and `true` that runs as `1`.**",
            '`"HTTP://X/a/../admin"` for a parameter typed `AnyUrl` runs as `http://x/admin`',
        ):
            assert stated in readme, stated


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestSeventhReview:
    """The seventh review (2026-11-23). ``jsonschema`` writes the value that failed into
    the message of every error it makes, the ones an ``anyOf`` asks for and drops among
    them, so a schema that nests wrote out everything below each level: 750 KB of
    arguments held the event loop 2.1 seconds for 497 counted steps. No keyword the gate
    runs writes a message from a value now. Found beside it: a subschema that declares
    a draft, and a ``$ref`` back to a root that declares one, were checked by the
    keywords of ``jsonschema`` and none of the gate's. And the steps do not bound the
    size of a call, so the arguments of one call have a ceiling on their size too."""

    example = TestWhatIsSentIsWhatWasJudged.example
    Recording = TestWhatIsSentIsWhatWasJudged.Recording
    judged = TestTheWholeListedSchema.judged
    schema_module = staticmethod(TestThirdReview.schema_module)
    spent = TestThirdReview.spent
    _storing = TestSixthReview._storing
    _call = staticmethod(TestSixthReview._call)

    INTEGERS = TestSixthReview.INTEGERS
    DRAFT_2019 = "https://json-schema.org/draft/2019-09/schema"
    DRAFT_2020 = "https://json-schema.org/draft/2020-12/schema"

    #: A value of every JSON type that holds a marker a message would carry.
    MARK = "MARKER-" * 40
    BIG_OBJECT = {"k": [MARK], "zz": MARK}
    BIG_LIST = [[MARK], MARK, MARK]
    BIG_NUMBER = 10**300 + 7

    #: keyword -> (the schema of ``p``, a value that fails it), for every keyword of
    #: every draft the gate reads; ``None`` for the two it never asserts. A list of
    #: ``(drafts, schema, value)`` where the drafts differ.
    FAILING = {
        "$ref": ({"$ref": "#/$defs/never"}, BIG_OBJECT),
        "additionalItems": ({"items": [{}], "additionalItems": False}, BIG_LIST),
        "additionalProperties": ({"additionalProperties": False}, BIG_OBJECT),
        "allOf": ({"allOf": [{"type": "integer"}]}, BIG_OBJECT),
        "anyOf": ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, BIG_OBJECT),
        "const": ({"const": 1}, BIG_OBJECT),
        "contains": ({"contains": {"type": "integer"}}, BIG_LIST),
        "dependencies": (
            {"dependencies": {"k": ["absent"], "zz": {"type": "integer"}}},
            BIG_OBJECT,
        ),
        "dependentRequired": ({"dependentRequired": {"k": ["absent"]}}, BIG_OBJECT),
        "dependentSchemas": ({"dependentSchemas": {"k": {"type": "integer"}}}, BIG_OBJECT),
        "enum": ({"enum": [1, "a"]}, BIG_OBJECT),
        "exclusiveMaximum": ({"exclusiveMaximum": 1}, BIG_NUMBER),
        "exclusiveMinimum": ({"exclusiveMinimum": 0}, -BIG_NUMBER),
        "format": None,
        "if": ({"if": {}, "then": {"type": "integer"}}, BIG_OBJECT),
        "items": ({"items": {"type": "integer"}}, BIG_LIST),
        "maxItems": ({"maxItems": 1}, BIG_LIST),
        "maxLength": ({"maxLength": 1}, MARK),
        "maxProperties": ({"maxProperties": 1}, BIG_OBJECT),
        "maximum": ({"maximum": 1}, BIG_NUMBER),
        "minItems": ({"minItems": 9}, BIG_LIST),
        "minLength": ({"minLength": 10_000}, MARK),
        "minProperties": ({"minProperties": 9}, BIG_OBJECT),
        "minimum": ({"minimum": 1}, -BIG_NUMBER),
        "multipleOf": ({"multipleOf": 10}, BIG_NUMBER),
        "not": ({"not": {}}, BIG_OBJECT),
        "oneOf": ({"oneOf": [{}, {}]}, BIG_OBJECT),
        "pattern": None,
        "patternProperties": ({"patternProperties": {"^k": {"type": "integer"}}}, BIG_OBJECT),
        "prefixItems": ({"prefixItems": [{"type": "integer"}]}, BIG_LIST),
        "properties": ({"properties": {"k": {"type": "integer"}}}, BIG_OBJECT),
        "propertyNames": ({"propertyNames": {"maxLength": 1}}, BIG_OBJECT),
        "required": ({"required": ["absent"]}, BIG_OBJECT),
        "type": ({"type": "integer"}, BIG_OBJECT),
        "unevaluatedItems": ({"unevaluatedItems": False}, BIG_LIST),
        "unevaluatedProperties": ({"unevaluatedProperties": False}, BIG_OBJECT),
        "uniqueItems": ({"uniqueItems": True}, BIG_LIST),
    }
    #: More ways to a message: the ``false`` schema wherever a schema stands, the keywords
    #: that ask whether a subschema is valid, and the errors one keyword sets on another.
    MORE = [
        ({"properties": {"k": False}}, BIG_OBJECT),
        ({"additionalProperties": {"not": {}}}, BIG_OBJECT),
        ({"propertyNames": False}, BIG_OBJECT),
        ({"if": {}, "then": False}, BIG_OBJECT),
        ({"if": False, "else": False}, BIG_OBJECT),
        ({"allOf": [False]}, BIG_OBJECT),
        ({"anyOf": [False, {"type": "integer"}]}, BIG_OBJECT),
        ({"oneOf": [False]}, BIG_OBJECT),
        ({"not": {"not": False}}, BIG_OBJECT),
        ({"dependentSchemas": {"k": False}}, BIG_OBJECT),
        ({"unevaluatedProperties": {"type": "integer"}}, BIG_OBJECT),
        ({"properties": {"k": {}}, "unevaluatedProperties": False}, BIG_OBJECT),
        (
            {"allOf": [{"properties": {"k": False}}], "unevaluatedProperties": {"not": {}}},
            BIG_OBJECT,
        ),
        ({"contains": False}, BIG_LIST),
        ({"contains": {}, "minContains": 9}, BIG_LIST),
        ({"contains": {}, "maxContains": 1}, BIG_LIST),
        ({"items": False}, BIG_LIST),
        ({"prefixItems": [False]}, BIG_LIST),
        ({"prefixItems": [{}], "items": False}, BIG_LIST),
        ({"prefixItems": [{}], "unevaluatedItems": False}, BIG_LIST),
        ({"contains": {"type": "string"}, "unevaluatedItems": False}, BIG_LIST),
        ({"unevaluatedItems": {"type": "integer"}}, BIG_LIST),
        (
            {"anyOf": [{"items": False}, {"items": {"type": "string"}}], "unevaluatedItems": False},
            BIG_LIST,
        ),
        ({"items": {"anyOf": [{"type": "integer"}, {"maxLength": 1}, {"maxItems": 0}]}}, BIG_LIST),
        ({"maximum": 1.5}, BIG_NUMBER),
        ({"multipleOf": 0.5}, 0.25),
    ]

    def drafts(self):
        module = self.schema_module()
        declared = {"7": DRAFT_07, "2019": self.DRAFT_2019, "2020": self.DRAFT_2020}
        named = {
            "7": "Draft7Validator",
            "2019": "Draft201909Validator",
            "2020": "Draft202012Validator",
        }
        by_name = {base.__name__: (base, strict) for base, strict in module._DRAFTS.items()}
        assert set(by_name) == set(named.values())
        return [(declared[key], *by_name[name]) for key, name in named.items()]

    @contextlib.contextmanager
    def messages(self):
        """Every message of every validation error made while this is open, the ones
        that are dropped among them."""
        from jsonschema.exceptions import ValidationError

        made: list = []
        original = ValidationError.__init__
        assert "__init__" not in vars(ValidationError)

        def recording(error, message, *args, **kwargs):
            made.append(message)
            original(error, message, *args, **kwargs)

        ValidationError.__init__ = recording
        try:
            yield made
        finally:
            del ValidationError.__init__

    @staticmethod
    def listing(declared, inner, shape):
        """A tool's schema with ``inner`` as the schema of ``p``: directly, as an
        alternative beside null, or as the items of a list."""
        inner = json.loads(json.dumps(inner))
        definitions = {"$defs": {"never": False}} if inner.get("$ref") else {}
        if shape == "beside null":
            inner = {"anyOf": [{"type": "null"}, inner]}
        elif shape == "in a list":
            inner = {"type": "array", "items": {"anyOf": [inner, {"type": "null"}]}}
        return {"$schema": declared, "type": "object", "properties": {"p": inner}, **definitions}

    # --- B1: no message is written from a value ------------------------------------------------

    def test_the_premise_jsonschema_writes_the_value_into_the_message(self):
        from jsonschema import validators

        with self.messages() as made:
            list(validators.Draft202012Validator({"type": "integer"}).iter_errors(self.MARK))
            list(validators.Draft202012Validator(False).iter_errors(self.MARK))
        assert len(made) == 2 and all(self.MARK in message for message in made)

    def test_every_keyword_is_the_gates_own_function_but_ref(self):
        """A function of ``jsonschema`` among them would write its message again. A
        ``jsonschema`` that adds a keyword fails here: the keyword has no function of
        the gate's, which refuses a schema that uses it rather than run one of theirs."""
        module = self.schema_module()
        for _, base, strict in self.drafts():
            checks = module._checks(base)
            assert set(checks) == set(base.VALIDATORS) == set(strict.VALIDATORS)
            assert set(checks) <= set(self.FAILING) | {"$dynamicRef", "$recursiveRef"}, base
            for keyword, check in checks.items():
                if keyword == "$ref":
                    assert check is base.VALIDATORS["$ref"]
                    continue
                assert check.__module__ == module.__name__, (base, keyword)
                assert check is not base.VALIDATORS[keyword]
                unread = check.__name__ == "unread"
                assert unread == (keyword in ("$dynamicRef", "$recursiveRef")), (base, keyword)
        # the functions it takes from `jsonschema` for what the other keywords evaluated
        from jsonschema import _legacy_keywords, _utils

        assert module._EVALUATED_2020 == (
            _utils.find_evaluated_item_indexes_by_schema,
            _utils.find_evaluated_property_keys_by_schema,
        )
        assert module._EVALUATED_2019 == (
            _legacy_keywords.find_evaluated_item_indexes_by_schema,
            _legacy_keywords.find_evaluated_property_keys_by_schema,
        )

    def test_no_error_made_while_a_call_is_checked_holds_the_value(self):
        """Every keyword of every draft, failing on a value that holds a marker, alone,
        beside null and inside a list: every error made, kept or dropped, has the
        gate's one message."""
        module = self.schema_module()
        for declared, base, strict in self.drafts():
            cases = []
            for keyword in strict.VALIDATORS:
                if keyword in ("$dynamicRef", "$recursiveRef") or self.FAILING[keyword] is None:
                    continue
                cases.append((keyword, *self.FAILING[keyword]))
            cases += [(None, schema, value) for schema, value in self.MORE]
            for keyword, inner, value in cases:
                known = set(strict.VALIDATORS) | {"then", "else"}
                if "dependentRequired" in known:  # 2019-09 and 2020-12
                    known |= {"minContains", "maxContains"}
                if not set(inner) <= known:
                    continue  # a keyword this draft does not have
                for shape in ("alone", "beside null", "in a list"):
                    schema = self.listing(declared, inner, shape)
                    given = {"p": [value, value] if shape == "in a list" else value}
                    listed = module.Listed(schema)
                    # the listing itself is checked once, against the draft's own
                    # schema, by `jsonschema`: its messages are of the listing
                    listed._prepared()
                    with self.messages() as made:
                        with pytest.raises(module.Refused) as refused:
                            listed.complete(given)
                    where = (base.__name__, keyword, inner, shape)
                    assert made, where
                    assert set(made) == {module._UNMET}, where
                    assert "MARKER" not in refused.value.reason, where
        for keyword in ("pattern", "format"):
            schema = {"properties": {"p": {keyword: "^a+$" if keyword == "pattern" else "date"}}}
            listed = module.Listed(schema)
            listed._prepared()
            with self.messages() as made:
                assert listed.complete({"p": self.MARK})[1] == {"p": self.MARK}
            assert made == []

    def test_a_refusal_reads_as_it_did(self):
        """What the gate reads of an error is unchanged: the keyword, its value, the
        JSON type of the value and the place. (The ``false`` schema, which has no
        keyword, is named for what it is since the eighth review.)"""
        for inner, value, reason in (
            (
                {"type": "integer"},
                "x",
                "argument `p` is of JSON type string where the server's schema declares integer",
            ),
            (
                {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                [],
                "argument `p` is of JSON type array where the server's schema declares integer or null",
            ),
            (
                {"required": ["a", "b"]},
                {"a": 1},
                "argument `p` leaves out `b`, which the server's schema requires",
            ),
            (
                {"minLength": 3},
                "x",
                "argument `p` does not meet `minLength` in the server's schema",
            ),
            ({"maximum": 3}, 4, "argument `p` does not meet `maximum` in the server's schema"),
            (
                {"contains": {}, "maxContains": 1},
                [1, 2],
                "argument `p` does not meet `maxContains` in the server's schema",
            ),
            (
                {"contains": {}, "minContains": 3},
                [1, 2],
                "argument `p` does not meet `minContains` in the server's schema",
            ),
            ({"items": False}, [1], "argument `p` does not meet `items` in the server's schema"),
            ({"not": {}}, 1, "argument `p` does not meet `not` in the server's schema"),
            (
                {"properties": {"x": False}},
                {"x": 1},
                "argument `p.x` stands where the server's schema allows no value (`false`)",
            ),
            (
                {"unevaluatedProperties": False},
                {"x": 1},
                "argument `p` does not meet `unevaluatedProperties` in the server's schema",
            ),
        ):
            assert self.judged({"p": value}, {"properties": {"p": inner}})[2] == (
                f"Recusal refused `purge`: {reason}"
                + (
                    "; the server would not run with the value the policy judged. Send the "
                    "declared JSON type."
                    if "JSON type" in reason
                    else f"; {_AS_THEY_STAND}"
                )
            ), inner

    #: The recursive model `tools/measure_mcp_cost.py` measures, as Pydantic lists it.
    NODE = {
        "type": "object",
        "properties": {"p": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "properties": {
                    "v": {"default": 1, "type": "integer"},
                    "kids": {"default": [], "items": {"$ref": "#/$defs/Node"}, "type": "array"},
                    "kid": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}], "default": None},
                },
                "type": "object",
            }
        },
    }

    def chain(self, count):
        """A chain of `kid` 60 deep whose last node is not valid and holds ``count`` kids."""
        node = {"v": "x", "kids": [{} for _ in range(count)]}
        for _ in range(60):
            node = {"kid": node}
        return {"p": node}

    def test_the_reviewers_chain_costs_the_same_whatever_its_last_node_holds(self):
        """The reviewer's case: every level of the chain asked its alternative for an
        error, and each error wrote out everything below it. Counted here without a
        clock: the errors made and the characters of their messages are the same for
        100 kids and for 20,000, and so are the steps."""
        module = self.schema_module()
        listed = module.Listed(self.NODE)
        listed._prepared()
        found = []
        for count in (100, 20_000):
            works: list = []
            original = module._Work.__init__

            def remembering(work, allowed):
                original(work, allowed)
                works.append(work)

            module._Work.__init__ = remembering
            try:
                with self.messages() as made:
                    with pytest.raises(module.Refused) as refused:
                        listed.complete(self.chain(count))
            finally:
                module._Work.__init__ = original
            assert refused.value.reason == (
                "argument `p.kid` matches none of the alternatives the server's schema "
                f"lists there; {_AS_THEY_STAND}"
            )
            found.append((works[-1].steps, len(made), sum(map(len, made))))
        assert found[0] == found[1]
        steps, errors, characters = found[0]
        assert steps == 497 and characters == errors * len(module._UNMET) and errors < 500

    def test_a_keyword_jsonschema_adds_refuses_the_schema_that_uses_it(self, monkeypatch):
        """What a later ``jsonschema`` with one more keyword meets: no function of the
        gate's for it, so a schema that uses it is refused when a call reaches it, and
        every other schema is checked as before."""
        from jsonschema import validators

        module = self.schema_module()
        base = validators.Draft202012Validator
        monkeypatch.delitem(module._KEYWORDS, "minLength")
        monkeypatch.setitem(module._DRAFTS, base, module._strict(base))
        assert module.Listed({"properties": {"p": {"type": "string"}}}).complete({"p": "a"})
        with pytest.raises(module.Refused) as refused:
            module.Listed({"properties": {"p": {"minLength": 2}}}).complete({"p": "abc"})
        assert (
            refused.value.reason == f"{_SCHEMA} uses `minLength`, which the gate has no check for"
        )

    # --- a draft declared anywhere but on the schema of the arguments ---------------------------

    def test_the_premise_jsonschema_picks_its_own_validator_for_a_schema_that_declares_a_draft(
        self,
    ):
        from jsonschema import validators

        strict = self.schema_module()._DRAFTS[validators.Draft202012Validator]
        chosen = validators.validator_for({"$schema": DRAFT_07}, default=strict)
        assert chosen is validators.Draft7Validator

    @pytest.mark.parametrize("declared", [DRAFT_07, DRAFT_2019, DRAFT_2020])
    def test_a_ref_back_to_a_root_that_declares_a_draft_is_checked_by_the_gates_keywords(
        self, declared, monkeypatch
    ):
        """A float for an integer, a boolean for an ``enum`` of numbers and a pattern,
        one level under a ``$ref`` to the schema of the arguments: each was checked by
        ``jsonschema``'s own keywords, which take ``5.0`` for an integer and run the
        pattern."""
        import re

        schema = {
            "$schema": declared,
            "type": "object",
            "properties": {
                "n": {"type": "integer"},
                "level": {"enum": [1, 2]},
                "name": {"type": "string", "pattern": "^(a+)+$"},
                "kid": {"$ref": "#"},
            },
        }
        searched = []
        search = re.search

        def counting(pattern, string, *args, **kwargs):
            searched.append(pattern)
            return search(pattern, string, *args, **kwargs)

        monkeypatch.setattr(re, "search", counting)
        for given, reason in (
            ({"n": 5.0}, "is of JSON type number where the server's schema declares integer"),
            ({"level": 1.0}, "is not one of the values the server's schema lists"),
        ):
            name = next(iter(given))
            for depth in (0, 1, 3):
                nested = given
                for _ in range(depth):
                    nested = {"kid": nested}
                place = "kid." * depth + name
                assert self.judged(nested, schema)[2].startswith(
                    f"Recusal refused `purge`: argument `{place}` {reason}"
                ), (declared, nested)
        ok = {"n": 5, "kid": {"n": 6, "name": "aaaa!", "kid": {"level": 2}}}
        assert self.judged(ok, schema) == (ok, ok, None)
        assert searched == []

    def test_a_subschema_that_declares_a_draft_is_refused(self):
        told = (
            f"Recusal refused `purge`: {_SCHEMA} embeds a schema that declares a draft of its "
            "own (`$schema`), which the gate does not read"
        )
        for schema in (
            {"properties": {"p": {"$schema": DRAFT_07, "type": "integer"}}},
            {"properties": {"p": {"$ref": "#/$defs/n"}}, "$defs": {"n": {"$schema": DRAFT_07}}},
            {"properties": {"p": {"items": {"anyOf": [{"$schema": self.DRAFT_2020}]}}}},
        ):
            assert self.judged({"p": 5.0}, schema) == (None, None, told)
        # a property of that name is a property
        named = {"properties": {"$schema": {"type": "string"}}}
        given = {"$schema": "x"}
        assert self.judged(given, named) == (given, given, None)

    def test_a_schema_the_scan_does_not_read_that_declares_a_draft_is_refused_at_the_call(self):
        """A ``$ref`` may name a place under a keyword the gate does not read, which the
        scan does not look into. Whatever leaves the gate's validator is refused."""
        schema = {
            "properties": {"p": {"$ref": "#/x-shared/n"}},
            "x-shared": {"n": {"$schema": DRAFT_07, "type": "integer"}},
        }
        assert self.judged({"p": 5.0}, schema) == (
            None,
            None,
            f"Recusal refused `purge`: {_SCHEMA} declares a JSON Schema draft on a subschema, "
            "which the gate does not read",
        )

    # --- a ceiling on the size of one call's arguments -----------------------------------------

    def test_the_size_ceiling_is_the_one_the_readme_states(self, example):
        module = self.schema_module()
        package = importlib.import_module("recusal_openai_agents_mcp")
        assert module.SIZE_CEILING == package.SIZE_CEILING == 1_000_000
        assert "SIZE_CEILING" in package.__all__
        readme = (ROOT / "adapters" / "openai-agents-mcp" / "README.md").read_text(encoding="utf-8")
        assert (
            f"more than {module.SIZE_CEILING:,} characters as JSON text "
            "(`recusal_openai_agents_mcp.SIZE_CEILING`)"
        ) in " ".join(readme.split())

    SIZED = [
        {},
        {"a": 1},
        {"a": [], "b": {}, "c": [[], {}], "d": None, "e": True, "f": False},
        {"text": 'a "quoted" \\ back\nline\ttab\r\x00\x1f\x7f end', "é": "ü \U0001f600"},
        {"numbers": [0, -1, 10**30, -(10**30), 1.5, -0.0, 1e300, 5e-324, 1e16, 123456789.125]},
        {"deep": [[[[[[[[[[{"k": [1, [2, [3]]]}]]]]]]]]]]},
        {"many": list(range(500)), "names": {f"k{n}": str(n) for n in range(200)}},
        {"": "", " ": [" "], "\\": ["\\"]},
    ]

    @pytest.mark.parametrize("value", SIZED, ids=[str(n) for n in range(len(SIZED))])
    def test_the_size_is_the_length_of_the_most_compact_json_text(self, value):
        module = self.schema_module()
        size = len(json.dumps(value, separators=(",", ":"), ensure_ascii=False))
        assert not module.too_large(value, size)
        assert not module.too_large(value, size + 1)
        assert module.too_large(value, size - 1)
        assert module.too_large(value, 1)

    def test_the_size_is_read_no_further_than_the_ceiling(self, monkeypatch):
        """Without a clock: a string longer than what is left is not read, and data
        that holds itself is given up at the ceiling or at the depth the gate reads."""
        module = self.schema_module()
        read = []
        as_json = module._as_json

        def counting(text):
            read.append(len(text))
            return as_json(text)

        monkeypatch.setattr(module, "_as_json", counting)
        assert module.too_large({"a": "x" * 50, "b": "y" * 5_000_000, "c": "z" * 50}, 1_000)
        assert read == [1, 1, 1, 50]
        endless: list = []
        endless.append(endless)
        assert not module.too_large(endless, 1_000)
        wide: list = []
        wide.extend([wide] * 3)
        assert module.too_large(wide, 100_000)
        # more members than characters are left: they are not looked at
        assert module.too_large([[0] * 2_000_000], 1_000_000)
        # what is not JSON data is one character, and the gate refuses it next
        assert not module.too_large({"a": object(), "b": (1, 2, 3), "c": b"x" * 9}, 20)

    def test_a_call_over_the_size_ceiling_is_refused_before_anything_reads_it(self, example):
        module = self.schema_module()
        text = {"type": "object", "properties": {"p": {"type": "string"}}}
        most = {"p": "a" * (module.SIZE_CEILING - len('{"p":""}'))}
        assert self.judged(most, text) == (most, most, None)
        over = {"p": most["p"] + "a"}
        assert self.judged(over, text) == (None, None, f"Recusal refused `purge`: {_TOO_LARGE}")
        for name in ("size_ceiling=", "gate_mcp_server", "gated_agent", "audited_agent"):
            assert name in _TOO_LARGE
        # before the other refusals: none of these is read to find out what it is
        long = "a" * module.SIZE_CEILING
        for arguments in (
            [long],
            {"p": long, "q": object()},
            {"zz": [long]},
            {"p": long, "q": float("nan")},
        ):
            assert self.judged(arguments, text) == (
                None,
                None,
                f"Recusal refused `purge`: {_TOO_LARGE}",
            )
        # an argument the schema does not name costs no step, and is counted here
        unnamed = {"zz": [0] * 600_000}
        assert self.spent({"zz": [0] * 1000}, text, module.STEP_CEILING)[0] < 10
        assert self.judged(unnamed, text)[2] == f"Recusal refused `purge`: {_TOO_LARGE}"
        assert self.judged(unnamed, text, size_ceiling=2_000_000) == (unnamed, unnamed, None)

    def test_the_largest_calls_the_step_ceiling_completes_are_under_the_size_ceiling(self, example):
        """The page's largest row: 39,997 members of `dict[str, Optional[str]]`, decoded
        from JSON text as the SDK hands them over. One member more is over the ceiling
        on the steps."""
        module = self.schema_module()
        schema = {
            "type": "object",
            "properties": {
                "p": {
                    "type": "object",
                    "additionalProperties": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                }
            },
        }

        def members(count):
            made = {"p": {f"x{number}": f"x{number}" for number in range(count)}}
            return json.loads(json.dumps(made))

        given = members(39_997)
        assert len(json.dumps(given, separators=(",", ":"))) == 697_733 < module.SIZE_CEILING
        sent, reason = _adapter()._judged_call("purge", given, schema, lambda *_: [])
        assert reason is None and sent == given
        sent, reason = _adapter()._judged_call("purge", members(39_998), schema, lambda *_: [])
        assert sent is None and reason == f"Recusal refused `purge`: {_OVER_THE_CEILING}"

    def test_an_application_raises_the_size_ceiling_for_a_connection(self, example):
        over = {"p": [1, 2], "note": "a" * 1_100_000}
        seen = []

        def policy(name, arguments):
            seen.append(len(arguments.get("note", "")))
            return []

        default = example.gate_mcp_server(self._storing(example, self.INTEGERS), policy)
        assert default.call_tool.recusal_size_ceiling == 1_000_000
        assert self._call(default, over) == f"Recusal refused `store`: {_TOO_LARGE}"
        assert default.ran == [] and seen == []
        raised = example.gate_mcp_server(
            self._storing(example, self.INTEGERS), policy, size_ceiling=1_300_000
        )
        assert raised.call_tool.recusal_size_ceiling == 1_300_000
        assert self._call(raised, over) == "ran"
        assert raised.ran == [("store", over)] and seen == [1_100_000]
        lowered = example.gate_mcp_server(
            self._storing(example, self.INTEGERS), policy, size_ceiling=20
        )
        assert self._call(lowered, {"p": list(range(30))}) == (
            f"Recusal refused `store`: {_TOO_LARGE.replace('1,000,000', '20')}"
        )
        assert self._call(lowered, {"p": [1, 2]}) == "ran"
        # a call with no arguments is `{}`
        assert self._call(lowered, None) == "ran"

    def test_gated_agent_and_audited_agent_hand_the_size_ceiling_to_every_server(
        self, example, tmp_path
    ):
        from agents import Agent

        from recusal.audit import AuditLog

        package = importlib.import_module("recusal_openai_agents_mcp")
        first = self._storing(example, self.INTEGERS)
        second = self._storing(example, self.INTEGERS)
        agent = Agent(name="a", mcp_servers=[first, second])
        gated = package.gated_agent(agent, lambda *_: [], size_ceiling=50)
        assert package.is_gated(gated)
        assert [s.call_tool.recusal_size_ceiling for s in gated.mcp_servers] == [50, 50]
        assert [s.call_tool.recusal_step_ceiling for s in gated.mcp_servers] == [200_000] * 2
        assert self._call(first, {"p": list(range(30))}).startswith(
            "Recusal refused `store`: the arguments are more than 50 characters as JSON text"
        )
        third = self._storing(example, self.INTEGERS)
        log = AuditLog(path=str(tmp_path / "decisions.jsonl"))
        package.audited_agent(
            Agent(name="b", mcp_servers=[third]), lambda *_: [], log, size_ceiling=7
        )
        assert third.call_tool.recusal_size_ceiling == 7
        plain = self._storing(example, self.INTEGERS)
        package.audited_agent(Agent(name="c", mcp_servers=[plain]), lambda *_: [], log)
        assert plain.call_tool.recusal_size_ceiling == package.SIZE_CEILING

    def test_a_connection_carries_one_size_ceiling(self, example):
        from agents import Agent

        def policy(name, arguments):
            return []

        server = example.gate_mcp_server(self._storing(example, self.INTEGERS), policy)
        gated_once = server.call_tool
        assert example.gate_mcp_server(server, policy, size_ceiling=1_000_000) is server
        assert server.call_tool is gated_once
        with pytest.raises(ValueError, match="already gated with a different size_ceiling"):
            example.gate_mcp_server(server, policy, size_ceiling=2_000_000)
        with pytest.raises(ValueError, match="already gated with a different size_ceiling"):
            example.gated_agent(Agent(name="a", mcp_servers=[server]), policy, size_ceiling=1)
        assert server.call_tool is gated_once
        assert server.call_tool.recusal_size_ceiling == 1_000_000

    @pytest.mark.parametrize("ceiling", [0, -1, True, 1.5, 1_000_000.0, "1000000", None])
    def test_a_size_ceiling_that_is_not_a_whole_number_is_refused(self, example, ceiling):
        from agents import Agent

        server = self._storing(example, self.INTEGERS)
        told = "size_ceiling is the most characters the arguments of one call may be as JSON text"
        with pytest.raises(ValueError, match=told):
            example.gate_mcp_server(server, lambda *_: [], size_ceiling=ceiling)
        assert not example.is_gated_mcp_server(server)
        with pytest.raises(ValueError, match=told):
            example.gated_agent(Agent(name="a"), lambda *_: [], size_ceiling=ceiling)

    # --- a copy handed over is not copied again -------------------------------------------------

    def test_arguments_handed_over_are_written_to_and_others_are_not(self):
        module = self.schema_module()
        schema = {"properties": {"o": {"properties": {"flag": {"default": True}}}}}
        given = {"o": {}}
        judged, sent = module.Listed(schema).complete(given)
        assert given == {"o": {}} and sent == {"o": {"flag": True}} and sent is not given
        judged, sent = module.Listed(schema).complete(given, owned=True)
        assert sent is given and given == {"o": {"flag": True}} and judged is not given

    def test_a_refusal_after_the_copy_is_bound_to_the_arguments_and_reads_them_once(
        self, example, monkeypatch
    ):
        gate = _adapter()
        read = []
        measure, copied = gate._measure, gate._detached
        monkeypatch.setattr(gate, "_measure", lambda *a: read.append("measured") or measure(*a))
        monkeypatch.setattr(gate, "_detached", lambda value: read.append("copied") or copied(value))
        monkeypatch.setattr(gate, "_rendered", lambda value: read.append("written") or "")
        given = {"p": ["x"]}
        assert self.judged(given, self.INTEGERS)[2].startswith(
            "Recusal refused `purge`: argument `p[0]` is of JSON type string"
        )
        assert read == ["measured", "copied"]

    # --- gated_agent gates every server or none -------------------------------------------------

    @pytest.mark.parametrize("how", ["step_ceiling", "size_ceiling", "policy", "recording", "type"])
    def test_gated_agent_leaves_no_server_gated_when_it_refuses_one(self, example, how):
        """The reviewer's case: the second server already carried another ceiling, and
        the first was left gated in place though ``gated_agent`` raised."""
        from agents import Agent

        def policy(name, arguments):
            return []

        def other(name, arguments):
            return []

        first = self._storing(example, self.INTEGERS)
        second = self._storing(example, self.INTEGERS)
        before = (first.call_tool, first.list_tools)
        asked = {}
        if how == "policy":
            example.gate_mcp_server(second, other)
        elif how == "recording":
            example.gate_mcp_server(second, policy, lambda seam, context: {})
        elif how == "type":
            second = object()
        else:
            example.gate_mcp_server(second, policy)
            asked = {how: 123_456}
        agent = Agent(name="a", mcp_servers=[first, second])
        with pytest.raises(TypeError if how == "type" else ValueError):
            example.gated_agent(agent, policy, **asked)
        assert not example.is_gated_mcp_server(first)
        assert (first.call_tool, first.list_tools) == before
        assert self._call(first, {"p": ["not an integer"]}) == "ran"
        # with the second server as asked, both are gated
        fresh = self._storing(example, self.INTEGERS)
        gated = example.gated_agent(Agent(name="b", mcp_servers=[first, fresh]), policy, **asked)
        package = importlib.import_module("recusal_openai_agents_mcp")
        assert package.is_gated(gated) and example.is_gated_mcp_server(first)
        assert self._call(first, {"p": ["not an integer"]}).startswith("Recusal refused")

    # --- the reasons and the page ---------------------------------------------------------------

    def test_the_ceiling_reason_says_when_raising_it_does_not_help(self, example):
        """The reviewer's case: a call over both limits was told to raise the ceiling,
        and with it raised was refused for the steps allowed for its values."""
        module = self.schema_module()
        schema = {
            "properties": {"p": {"type": "array", "items": {"$ref": "#/$defs/a0"}}},
            "$defs": {"a12": {"type": "integer"}},
        }
        for level in range(12):
            onward = {"$ref": f"#/$defs/a{level + 1}"}
            schema["$defs"][f"a{level}"] = {"allOf": [onward, dict(onward)]}
        given = {"p": [[number] for number in range(100)]}
        assert self.spent(given, schema, module.STEP_CEILING)[2] == _OVER_THE_CEILING
        assert self.spent(given, schema, 10**12)[2] == _over_each_value()
        assert (
            "Raising it does not help a call that costs more than 10,000 steps and 1,000 more "
            "for every value in its arguments, which the gate refuses under any ceiling."
        ) in _OVER_THE_CEILING

    def test_a_number_runs_as_what_the_page_now_names(self):
        """The cases the page's two headings did not name: a float that is `1.0` or
        `0.0` runs as a boolean, a number as an address or a complex number, and beside
        a `Decimal` only a whole number of days runs as a date."""
        import datetime
        import ipaddress
        from decimal import Decimal
        from enum import IntEnum
        from typing import Literal

        from pydantic import TypeAdapter

        class Level(IntEnum):
            low = 1
            high = 2

        def ran(kind, value):
            result = TypeAdapter(kind).validate_python(value)
            return type(result), result

        assert ran(Union[bool, Decimal], 1.0) == (bool, True)
        assert ran(Union[bool, Decimal], 0.0) == (bool, False)
        assert ran(Union[bool, Decimal], -0.0) == (bool, False)
        assert ran(Union[bool, Decimal], 2.0)[0] is Decimal
        assert ran(Union[bool, Level], 1.0) == (bool, True)
        assert ran(Union[Literal[True], float], 1.0) == (bool, True)
        assert ran(Union[ipaddress.IPv4Address, Decimal], 5) == (
            ipaddress.IPv4Address,
            ipaddress.IPv4Address("0.0.0.5"),
        )
        assert ran(Union[ipaddress.IPv4Address, Level], 1)[0] is ipaddress.IPv4Address
        assert ran(Union[complex, Decimal], 5) == (complex, 5 + 0j)
        assert ran(Union[complex, Level], 1) == (complex, 1 + 0j)
        assert ran(Union[Decimal, complex], 5)[0] is Decimal
        # a date beside a Decimal: only a whole number of days
        assert ran(Union[datetime.date, Decimal], 0)[0] is datetime.date
        assert ran(Union[datetime.date, Decimal], 86_400) == (
            datetime.date,
            datetime.date(1970, 1, 2),
        )
        assert ran(Union[datetime.date, Decimal], 5) == (Decimal, Decimal(5))
        assert ran(Union[datetime.date, Decimal], 1.5)[0] is Decimal
        assert ran(Union[datetime.time, Decimal], 1.5)[0] is datetime.time
        assert ran(Union[datetime.time, Decimal], 86_400)[0] is Decimal
        # and the gate sends each as it stands
        for kind, value in ((Union[bool, Decimal], 1.0), (Union[complex, Decimal], 5)):
            schema = {"properties": {"p": TypeAdapter(kind).json_schema()}}
            given = {"p": value}
            assert self.judged(given, schema) == (given, given, None)

    def test_the_page_says_what_bounds_the_time_of_a_call(self):
        readme = " ".join(
            (ROOT / "adapters" / "openai-agents-mcp" / "README.md")
            .read_text(encoding="utf-8")
            .split()
        )
        for stated in (
            "**A number a Pydantic server reads as a date, a time, an address, a complex "
            "number or an enum member.**",
            "**`1`, `0`, `1.0` and `0.0` that run as booleans, and `true` that runs as `1`.**",
            "only a whole number of days runs as a `date`",
            "Neither ceiling alone bounds the time",
            "an argument the schema does not name cost no steps",
        ):
            assert stated in readme, stated
        for gone in ("The ceiling does bound it.", "is 12 steps and took 14 ms"):
            assert gone not in readme, gone


_LONG_INTEGER = (
    "the arguments hold an integer of more than 4,300 digits, the most an integer in one "
    "call's arguments may have (the limit Python sets by default on writing an integer as "
    "text, which the gate applies on every interpreter); the gate has read no further. Send "
    "a number of fewer digits, or the digits as a string where the server's schema takes a "
    "string."
)


class _Panic(BaseException):
    """What a panic in a compiled dependency arrives as: not an ``Exception``."""


class TestEighthReview:
    """The eighth review (2026-11-23). On an interpreter without the limit on the digits
    of an integer, the size check wrote every integer out as text, which is quadratic
    before Python 3.12: one integer under the size ceiling held the event loop 11
    seconds. The gate's own readings of the arguments recursed, so on Python 3.11 and
    later their time moved up to four times with the depth the sender nested at. A call
    a little over the size ceiling was refused only after most of it was read, and one
    that is not JSON data was read three more times for its refusal. The gate read a
    ``$ref`` its own way, which differed from the validator's for ``%2F``; three
    exceptions left the gate uncaught; and a ``false`` subschema was reported as the
    keyword ``None``."""

    example = TestWhatIsSentIsWhatWasJudged.example
    Recording = TestWhatIsSentIsWhatWasJudged.Recording
    judged = TestTheWholeListedSchema.judged
    schema_module = staticmethod(TestThirdReview.schema_module)
    spent = TestThirdReview.spent

    ANY = {"type": "object", "properties": {"p": {}}}

    @staticmethod
    def lists(depth, leaf):
        """``leaf`` inside ``depth`` lists."""
        for _ in range(depth):
            leaf = [leaf]
        return leaf

    # --- an integer is never written out, and a long one is refused ------------------------------

    @pytest.mark.parametrize("limit", [640, 4300, 0])
    def test_the_digits_of_an_integer_are_counted_from_its_bits_exactly(self, limit):
        """Under a limit of 640 digits Python refuses to write a longer integer as text
        at all, so a count that wrote one would fail or fall back on a guess. The count
        is exact for the least and the greatest integer of every length, where a guess
        from the bits alone is one too many (``10**d - 1`` has the bits of ``10**d``)."""
        module = self.schema_module()
        with _digit_limit(limit):
            for digits in [*range(1, 330), 640, 641, 1000, 2000, 4299, 4300]:
                least, most = 10 ** (digits - 1), 10**digits - 1
                for number in (least, most, least + 1, most - 1):
                    if number < least:
                        continue
                    for value, size in ((number, digits), (-number, digits + 1)):
                        assert module.too_large(value, size - 1), (digits, value < 0)
                        assert not module.too_large(value, size), (digits, value < 0)
                        assert module.measure([value], size + 2) == module.FITS
                        assert module.measure([value], size + 1) == module.OVER
            assert module.measure(0, 1) == module.FITS and module.measure(-0, 1) == module.FITS
            assert module.measure({"a": [7, -7, 10, -10]}, len('{"a":[7,-7,10,-10]}')) == ""

    def test_the_bounds_the_count_rests_on(self):
        """An integer of ``b`` bits has at least ``(b - 1) * 30102 // 100000 + 1`` digits
        and at most ``b * 30103 // 100000 + 1``, and they differ by one at the most for
        every integer the gate reads. Checked against the digits themselves. (An
        integer of up to 18 digits is looked up among the powers of ten instead.)"""
        module = self.schema_module()
        assert module._POWERS == tuple(10**n for n in range(19)) and module._BEYOND == 10**18
        assert module.LONGEST_INTEGER == 4_300
        # the limit Python sets by default, where this Python has one
        assert getattr(sys.int_info, "default_max_str_digits", 4_300) == 4_300
        assert module._TOO_LONG == 10**4300 and module._TOO_LONG_BITS == 14_285
        power, digits = 1, 1
        for bits in range(1, module._TOO_LONG_BITS + 2):
            least, most = 1 << (bits - 1), (1 << bits) - 1
            while power * 10 <= least:
                power, digits = power * 10, digits + 1
            fewest = digits  # of ``least``
            greatest = digits + (most >= power * 10)
            assert (bits - 1) * 30102 // 100000 + 1 <= fewest
            assert greatest <= bits * 30103 // 100000 + 1 <= fewest + 1
        # the bits of 10**4300: no integer of fewer bits has more than 4,300 digits
        assert (10**4300 - 1).bit_length() == 14_285 == (10**4300).bit_length()
        assert ((1 << 14_284) - 1) < 10**4300 <= (1 << 14_285) - 1

    LONG = [10**4300, -(10**4300), 10**4300 + 1, 10**5000, 1 << 14_285, 1 << 3_000_000]
    NOT_LONG = [10**4300 - 1, -(10**4300 - 1), 10**4299, (1 << 14_284) + 1, 0, -1]

    @pytest.mark.parametrize("limit", [640, 4300, 0])
    def test_an_integer_of_more_than_4300_digits_is_refused_on_every_interpreter(
        self, example, limit
    ):
        """The reviewer's case: with no limit on the digits, ``json.loads`` hands the gate
        an integer of any length, and 999,000 digits held the event loop 11 seconds
        (23 with an audit log). Refused from its bits, whatever the limit is set to."""
        module = self.schema_module()
        told = f"Recusal refused `purge`: {_LONG_INTEGER}"
        schemas = (
            self.ANY,
            {"type": "object", "properties": {"p": {"type": "integer"}}},
            {"type": "object", "properties": {"q": {"type": "string"}}},
        )
        with _digit_limit(limit):
            for number in self.LONG:
                assert module.measure(number, 10**9) == module.LONG
                assert not module.too_large(number, 10**9)
                for arguments in (
                    {"p": number},
                    {"p": [1, {"deep": [number]}]},
                    {"zz": number},
                    {number: 1},
                    [number],
                    number,
                ):
                    for schema in schemas:
                        assert self.judged(arguments, schema) == (None, None, told)
            for number in self.NOT_LONG:
                given = {"p": number}
                assert self.judged(given, self.ANY) == (given, given, None)
        for name in ("4,300 digits", "on every interpreter", "as a string"):
            assert name in _LONG_INTEGER

    def test_whichever_of_the_size_and_a_long_integer_is_reached_first_refuses(self, example):
        module = self.schema_module()
        long = 10**5000
        assert module.measure(["a" * 100, long], 50) == module.OVER
        assert module.measure([long, "a" * 100], 50) == module.LONG
        assert self.judged({"p": long, "q": "a" * 2_000_000}, self.ANY)[2] == (
            f"Recusal refused `purge`: {_LONG_INTEGER}"
        )
        assert self.judged({"q": "a" * 2_000_000, "p": long}, self.ANY)[2] == (
            f"Recusal refused `purge`: {_TOO_LARGE}"
        )
        # below the depth the gate reads nothing is read, and the depth refuses the call
        deep = self.lists(70, [long])
        assert module.measure(deep, 10**9) == module.DEEP
        assert self.judged({"p": deep}, self.ANY)[2] == (
            "Recusal refused `purge`: arguments are nested too deeply"
        )

    def test_nothing_the_gate_does_with_an_integer_writes_it_out(self, example):
        """Without a clock. Under a limit of 640 digits Python raises for every integer
        of more digits that anything writes as text, so a call whose integers have 641
        to 4,300 digits, checked against every keyword that reads a number, completes
        only if the gate writes none of them (the policy here writes none either)."""
        schema = {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "minimum": 0, "multipleOf": 3},
                "e": {"enum": [3 * 10**3000, "x"]},
                "c": {"const": 3 * 10**4000},
                "u": {"type": "array", "uniqueItems": True},
                "x": {"type": "number", "exclusiveMaximum": 10**4300},
            },
        }
        big = 3 * 10**4000
        given = {
            "n": big,
            "e": 3 * 10**3000,
            "c": big,
            "u": [big, big + 1, 3 * 10**700, 1.5, [big]],
            "x": 10**4299,
            "zz": {"unnamed": [10**641, -(10**4299)]},
        }
        with _digit_limit(640):
            with pytest.raises(ValueError):
                str(10**641)
            assert self.judged(given, schema) == (given, given, None)
            # and refused for what each is, the integer still not written
            assert "does not meet `uniqueItems`" in self.judged({"u": [big, big]}, schema)[2]
            assert "does not meet `multipleOf`" in self.judged({"n": big + 1}, schema)[2]
            assert "is not the value" in self.judged({"c": big + 1}, schema)[2]
            # arguments that are not JSON data are written as text for the record: the
            # integers in them are the ones measured, of 640 digits here
            assert _adapter()._rendered({"n": 10**639, "x": float("inf")}) == (
                '{"n":1' + "0" * 639 + ',"x":inf}'
            )

    def test_json_text_is_read_without_making_an_integer_from_its_digits(self, monkeypatch):
        """Where the interpreter has no limit, decoding a long run of digits is quadratic
        too. The decoder is handed a function that makes no integer and gives up on more
        than 4,300 digits, as the decoder itself does under the default limit."""
        module = self.schema_module()
        made = []
        digits = module._digits
        assert module._DECODER.parse_int is digits

        def counting(text):
            made.append(len(text))
            return digits(text)

        monkeypatch.setattr(module, "_DECODER", json.JSONDecoder(parse_int=counting))
        for limit in (640, 4300, 0):
            with _digit_limit(limit):
                made.clear()
                assert module._json_text("[" + "9" * 4300 + ", -" + "9" * 4300 + "]") is True
                assert made == [4300, 4301]
                assert module._json_text("[" + "9" * 4301 + "]") is True
                assert module._json_text('{"a": -' + "9" * 300_000 + "}") is True
                assert module._json_text("9" * 4301) is False
                assert module._json_text("9" * 300_000) is False
                assert module._json_text("[" + "9" * 4301 + " oops") is True
                assert module._json_text("[" + "9" * 4300 + " oops") is False
                assert module._json_text("[1, 2.5, -3e7, 12]") is True
        assert module._digits("-" + "9" * 4300) == 0 == module._digits("9" * 4300)
        with pytest.raises(ValueError):
            module._digits("9" * 4301)
        with pytest.raises(ValueError):
            module._digits("-" + "9" * 4301)

    # --- the gate's own readings do not recurse --------------------------------------------------

    DATA = [
        None,
        True,
        0,
        -7,
        10**30,
        1.5,
        -0.0,
        "",
        "text",
        [],
        {},
        [1, [2, [3, {"a": [None, True, 1.5, "x"]}]]],
        {"a": {"b": {"c": [1, 2, {"d": {}}]}}, "e": [[], [[]], {}]},
        {"": "", "é": "ü", "list": [1.0, 1, True]},
    ]

    #: What the copy says of arguments that are not JSON data, which a refusal quotes.
    NOT_DATA = (
        "non-finite numbers are not JSON data",
        "Enum arguments are unsupported; declare data-only parameters "
        "and resolve application types inside the tool body",
        "arguments must be data-only",
        "cyclic arguments are not JSON data",
        "argument keys must be strings",
    )

    def test_the_copy_is_the_kernels_and_the_package_requires_the_kernel_that_has_it(self):
        """The gate copies the arguments with the kernel's ``detached`` and has no copy of
        its own: what is JSON data is said in one place. The kernel's does not call
        itself from 0.16.0, which is therefore the oldest kernel the package installs
        with (the test below copies data nested 5,000 deep), and the page says the same."""
        import ast
        from enum import Enum, IntEnum

        from recusal.invocation import detached

        module, gate = self.schema_module(), _adapter()
        assert module.detached is detached and gate._detached is detached
        assert not hasattr(module, "copied") and not hasattr(gate, "_copied")
        # no function of the package but the kernel's builds the copy: none of its own
        # modules defines one under the old name or under the kernel's
        for source in sorted(Path(module.__file__).parent.glob("*.py")):
            defined = {
                node.name
                for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert not defined & {"copied", "_copied", "detached", "_detached"}, source.name
        package = ROOT / "adapters" / "openai-agents-mcp"
        pyproject = (package / "pyproject.toml").read_text(encoding="utf-8")
        readme = (package / "README.md").read_text(encoding="utf-8")
        assert '"recusal>=0.16.0,<1"' in pyproject and "`recusal>=0.16.0`" in readme
        assert "recusal>=0.15" not in pyproject and "recusal>=0.15" not in readme

        class Color(Enum):
            RED = 1

        class Level(IntEnum):
            LOW = 1

        class Text(str):
            pass

        class Items(list):
            pass

        def outcome(function, value):
            try:
                return function(value)
            except TypeError as refused:
                return ("TypeError", str(refused))

        holds_itself: list = [1]
        holds_itself.append({"again": holds_itself})
        twice = [1, 2]
        not_data = [
            float("inf"),
            float("nan"),
            (1, 2),
            b"bytes",
            {1, 2},
            object(),
            Color.RED,
            Level.LOW,
            Text("x"),
            Items(),
            {1: "a"},
            {None: "a"},
            {"a": 1, 2: float("inf")},
            {"a": [float("inf")], 1: 2},
            holds_itself,
        ]
        for value in (*self.DATA, {"a": twice, "b": twice, "c": [twice]}):
            copy = module.detached(value)
            assert copy == value
            assert json.dumps(copy) == json.dumps(value)
        for bad in not_data:
            for value in (bad, [bad], {"a": bad}, [1, {"a": [2, bad]}, float("nan")], {"z": [bad]}):
                refused = outcome(module.detached, value)
                assert type(refused) is tuple and refused[0] == "TypeError", (value, refused)
                assert refused[1] in self.NOT_DATA, (value, refused)
        # nothing is shared with what was handed over, or within the copy
        given = {"a": twice, "b": twice, "c": {"d": [twice]}}
        copy = module.detached(given)
        assert copy is not given and copy["a"] is not twice and copy["a"] is not copy["b"]
        assert copy["c"] is not given["c"] and copy["c"]["d"][0] is not copy["a"]

    def test_no_reading_of_the_arguments_that_is_the_gates_own_recurses(self):
        """Without a clock: data nested 5,000 deep, five times the interpreter's recursion
        limit, is measured, copied, counted, compared and written with no
        ``RecursionError``. (The validation recurses with the schema, and is refused at
        the limit; the gate itself reads no deeper than 64 levels.)"""
        module, gate = self.schema_module(), _adapter()
        assert sys.getrecursionlimit() < 5_000
        for kind in ("list", "object"):
            deep = 7
            for _ in range(5_000):
                deep = [deep, 1] if kind == "list" else {"k": deep, "n": 1}
            assert module.measure(deep, 10**9) == module.DEEP
            assert module.measure(deep, 30) == module.OVER
            copy = module.detached(deep)
            level, other, levels = deep, copy, 0
            while type(level) in (list, dict):
                assert type(other) is type(level) and other is not level
                level, other = (level[0], other[0]) if kind == "list" else (level["k"], other["k"])
                levels += 1
            assert levels == 5_000 and other == 7
            assert module._count(deep) == 10_001
            assert module._same(deep, copy) is True
            unlike = module.detached(deep)
            inner = unlike
            for _ in range(4_999):
                inner = inner[0] if kind == "list" else inner["k"]
            if kind == "list":
                inner[1] = 1.0
            else:
                inner["n"] = 1.0
            assert module._same(deep, unlike) is False
            for exact in (True, False):
                work = module._Work(10**9)
                module._key(work, deep, exact)
                assert work.steps == 10_001
                module._key(work, deep, exact)
                assert work.steps == 10_002  # read once while the arguments stand
            text = gate._rendered(deep)
            if kind == "list":
                assert text == "[" * 65 + "...,...]" + ",1]" * 64
            else:
                assert text == '{"k":' * 64 + "{...:...,...:...}" + ',"n":1}' * 64

    @pytest.mark.parametrize("value", DATA, ids=[str(n) for n in range(len(DATA))])
    def test_what_the_readings_find_is_what_a_recursion_finds(self, value):
        module = self.schema_module()

        def count(each):
            if type(each) is dict:
                return 1 + sum(count(member) for member in each.values())
            if type(each) is list:
                return 1 + sum(count(item) for item in each)
            return 1

        def key(each, exact):
            if type(each) is list:
                return (list, tuple(key(item, exact) for item in each))
            if type(each) is dict:
                return (dict, frozenset((name, key(item, exact)) for name, item in each.items()))
            if exact or type(each) is str or each is None:
                return (type(each), each)
            return (float, module._number(each))

        assert module._count(value) == count(value)
        for exact in (True, False):
            work = module._Work(10**9)
            assert (
                module._key(work, value, exact)
                == key(value, exact)
                == module._key(None, value, exact)
            )
            assert work.steps == count(value)
            # stopped where the allowance ends, as a step for every value read asks
            for allowed in range(count(value)):
                short = module._Work(allowed)
                with pytest.raises(module._TooCostly):
                    module._key(short, value, exact)
                assert short.steps == allowed + 1
        assert module._same(value, json.loads(json.dumps(value)))
        for other in self.DATA:
            assert module._same(value, other) is (value is other), (value, other)

    def test_two_objects_are_the_same_only_under_the_same_names(self):
        module = self.schema_module()
        assert module._same({"a": 1, "b": [2]}, {"b": [2], "a": 1})
        assert not module._same({"a": 1}, {"b": 1})
        assert not module._same({"a": 1, "b": 2}, {"a": 1, "c": 2})
        assert not module._same([{"a": 1}], [{"b": 1}])
        assert not module._same({"a": 1}, {"a": 1, "b": 2})
        assert not module._same([1, 2], [1, 2, 3]) and not module._same([1, 2, 3], [1, 2])

    def test_a_ref_is_resolved_once_for_a_listing(self, example):
        """The walk asks for the schema a ``$ref`` names at every value it visits: each
        is asked of the library once, and read from what was kept after that."""
        module = self.schema_module()
        listed = module.Listed(TREE)
        chain = {"node": {"child": {"child": {"child": {}}}}}
        assert listed.complete(chain)[1]["node"]["child"]["child"]["child"] == {
            "keep": True,
            "child": None,
        }
        asked = []
        resolver = listed._resolve._resolver

        class Counting:
            def lookup(self, ref):
                asked.append(ref)
                return resolver.lookup(ref)

        listed._resolve._resolver = Counting()
        listed.complete(chain)
        assert asked == []
        assert listed._resolve("#/$defs/Node") is TREE["$defs"]["Node"]
        assert asked == []
        assert listed._resolve("#/properties/node") is TREE["properties"]["node"]
        assert asked == ["#/properties/node"]

    def test_a_value_65_levels_down_is_too_deep_and_64_is_not(self, example):
        module = self.schema_module()
        for leaf in (0, "x", None, [], {}):
            for kind in ("list", "object"):
                value = leaf
                for depth in range(1, 70):
                    value = [value] if kind == "list" else {"k": value}
                    found = module.measure(value, 10**9)
                    assert found == (module.DEEP if depth > 64 else module.FITS), (depth, leaf)
        # an empty list or object 64 levels down holds nothing deeper
        assert module.measure(self.lists(64, []), 10**9) == module.FITS
        assert module.measure(self.lists(64, [[]]), 10**9) == module.DEEP

    def test_a_call_is_read_once_to_measure_it_and_once_to_copy_it(self, example, monkeypatch):
        """And a refusal reads nothing again: one that is bound to the arguments knows
        from the copy that they are JSON data, and one that is not writes them once."""
        gate = _adapter()
        read: list = []
        measure, copied, rendered = gate._measure, gate._detached, gate._rendered
        monkeypatch.setattr(gate, "_measure", lambda *a: read.append("measured") or measure(*a))
        monkeypatch.setattr(gate, "_detached", lambda value: read.append("copied") or copied(value))
        monkeypatch.setattr(
            gate, "_rendered", lambda value: read.append("written") or rendered(value)
        )
        integers = TestSixthReview.INTEGERS
        cases = [
            ({"p": [1]}, None, ["measured", "copied"]),
            ({"p": ["x"]}, "argument `p[0]` is of JSON type string", ["measured", "copied"]),
            ([1, 2], "arguments are not a mapping", ["measured", "copied"]),
            (
                {"p": [1], "x": float("inf")},
                "arguments are not JSON data (non-finite numbers are not JSON data)",
                ["measured", "copied", "written"],
            ),
            ([object()], "arguments are not a mapping", ["measured", "copied", "written"]),
            (
                {"p": self.lists(70, 1)},
                "arguments are nested too deeply",
                ["measured", "written"],
            ),
            (self.lists(70, 1), "arguments are not a mapping", ["measured", "written"]),
            ({"p": "a" * 1_000_001}, "the arguments are more than 1,000,000", ["measured"]),
            ({"p": [10**4300]}, "the arguments hold an integer of more than", ["measured"]),
        ]
        for arguments, reason, readings in cases:
            read.clear()
            sent, told = gate._judged_call("purge", arguments, integers, lambda *_: [])
            assert read == readings, arguments
            if reason is None:
                assert sent == arguments and told is None
            else:
                assert sent is None and told.startswith(f"Recusal refused `purge`: {reason}")

    # --- the size is read no further than the ceiling, and cheaply -------------------------------

    def test_a_string_is_counted_a_part_at_a_time_and_no_further_than_the_ceiling(
        self, monkeypatch
    ):
        """Without a clock. A string of escapes a little over the ceiling was read to its
        end, a Python step for each escape (999,000 of them: 70 ms). It is written out
        as JSON text 65,536 characters at a time, and the reading stops at the part that
        goes over."""
        module = self.schema_module()
        written: list = []
        as_json = module._as_json

        def counting(text):
            written.append(len(text))
            return as_json(text)

        monkeypatch.setattr(module, "_as_json", counting)
        window = module._WINDOW
        assert window == 65_536
        # longer than what is left: not read at all
        assert module.too_large({"a": "x" * 50, "b": "y" * 5_000_000, "c": "z" * 50}, 1_000)
        assert max(written) <= 50 and 5_000_000 not in written
        for escape, each in (("\x00", 6), ('"', 2), ("\n", 2), ("\\", 2), ("\x1f", 6)):
            text = escape * 999_000
            written.clear()
            assert module.too_large({"p": text}, 1_000_000)
            assert written == [1, window], escape  # the name, and one part
            size = len(json.dumps(text))
            assert size == each * 999_000 + 2
            written.clear()
            assert not module.too_large(text, size) and module.too_large(text, size - 1)
            assert set(written) <= {window, 999_000 % window} and sum(written) == 2 * 999_000
        # the parts add up to the string, whatever falls on their edges
        for text in (
            'a"' * 100_000,
            "\x00" * (window + 1),
            "é\n" * window,
            "\U0001f600\\" * 70_000,
        ):
            size = len(json.dumps(text, ensure_ascii=False))
            assert not module.too_large(text, size) and module.too_large(text, size - 1)

    def test_what_a_list_below_the_depth_holds_counts_one_character_each(self):
        """What a list or an object more than 64 levels down holds is not read. Each
        member is one character at the least, so one that holds more of them than
        characters are left is over the ceiling all the same."""
        module = self.schema_module()
        below = self.lists(65, [0] * 1_000)
        written = 2 * 65 + 2 + 999  # the brackets of each list, and the commas
        assert not module.too_large(below, written + 1_000)
        assert module.too_large(below, written + 999)
        named = self.lists(65, {str(n): 0 for n in range(1_000)})
        written += 1_000  # a colon in each member
        assert not module.too_large(named, written + 1_000)
        assert module.too_large(named, written + 999)
        # one level higher the members are read, and counted for what they are
        read = self.lists(64, [0] * 1_000)
        size = len(json.dumps(read, separators=(",", ":")))
        assert not module.too_large(read, size) and module.too_large(read, size - 1)
        # and above it, a list of more members than characters are left is not read
        assert module.too_large([[0] * 2_000_000], 1_000_000)

    # --- one reading of a `$ref` ------------------------------------------------------------------

    def test_a_ref_is_read_as_the_validator_reads_it(self, example):
        """The reviewer's case: ``#/$defs/a%2Fb`` beside a definition named ``a/b`` and a
        path ``a`` then ``b``. The validator checked against the path, while the types,
        the default and the walk were read from the name."""
        schema = {
            "type": "object",
            "properties": {"p": {"$ref": "#/$defs/a%2Fb"}, "q": {"$ref": "#/$defs/a~1b"}},
            "$defs": {
                "a/b": {"type": ["integer", "string"], "default": 7},
                "a": {"b": {"type": "string"}},
            },
        }
        # `p` is the string schema, by the validation and by everything else: a number in
        # text is kept (the schema names no number there), and no default is written
        # for it (the gate's own reading took the 7 of the other definition)
        given = {"p": "5", "q": 5}
        assert self.judged(given, schema) == (given, given, None)
        assert self.judged({}, schema)[1] == {"q": 7}
        assert self.judged({"p": 5}, schema)[2].startswith(
            "Recusal refused `purge`: argument `p` is of JSON type integer where the server's "
            "schema declares string"
        )
        # `q` is the definition named `a/b`, where the text of a number is refused
        assert "holds the text of a number" in self.judged({"q": "5"}, schema)[2]

    AT = "#/$defs/x/anyOf/"
    REFS = [
        AT + "0",
        AT + "1",
        AT + "-1",
        AT + "+1",
        AT + "01",
        AT + " 1",
        AT + "1_0",
        AT + "\u0663",
    ]
    NOT_REFS = [AT + "\u00b2", AT + "9", AT + "a", AT, "#/y", "#x", "x", "", 5, None, ["#"]]

    def test_the_position_of_an_item_is_read_as_the_validator_reads_it(self, example):
        """``referencing`` reads a position with ``int``: a sign, zeros and a digit that is
        not ASCII are positions. The value each definition takes says which one a
        ``$ref`` names, for the validation and for the gate alike."""
        from jsonschema import validators
        from referencing import Registry

        module = self.schema_module()
        items = [{"const": f"item {n}", "default": f"item {n}"} for n in range(11)]
        for ref in self.REFS:
            schema = {"properties": {"p": {"$ref": ref}}, "$defs": {"x": {"anyOf": items}}}
            stock = validators.Draft202012Validator(schema, registry=Registry())
            taken = [n for n in range(11) if stock.is_valid({"p": f"item {n}"})]
            assert len(taken) == 1, ref
            listed = module.Listed(schema)
            assert listed.complete({})[0] == {"p": f"item {taken[0]}"}, ref
            assert listed._resolve(ref) is items[taken[0]]
            for n in range(11):
                given = {"p": f"item {n}"}
                if n == taken[0]:
                    assert self.judged(given, schema) == (given, given, None)
                else:
                    assert "is not the value" in self.judged(given, schema)[2]

    @pytest.mark.parametrize("ref", NOT_REFS, ids=[str(n) for n in range(len(NOT_REFS))])
    def test_a_ref_that_names_no_schema_is_refused_and_never_raised(self, example, ref):
        """The reviewer's case: ``²`` is a digit to ``str.isdigit`` and no number to
        ``int``, and the ``ValueError`` left the gate."""
        told = (
            f"Recusal refused `purge`: {_SCHEMA} has a `$ref` that does not resolve inside the "
            "schema itself, so the gate cannot check the arguments against it"
        )
        two = {"x": {"anyOf": [{}, {}]}}
        for schema in (
            {"properties": {"p": {"$ref": ref}}, "$defs": two},
            {"properties": {"p": {"anyOf": [{"type": "null"}, {"$ref": ref}]}}, "$defs": two},
        ):
            assert self.judged({"p": 1}, schema) == (None, None, told)
            assert self.judged({}, schema) == (None, None, told)

    def test_an_id_that_is_not_text_is_not_a_valid_schema(self, example):
        for schema in ({"$id": 5}, {"$id": ["x"], "properties": {"p": {"$ref": "#"}}}):
            assert self.judged({}, schema) == (
                None,
                None,
                f"Recusal refused `purge`: {_SCHEMA} is not a valid JSON Schema",
            )
        # an `$id` that is text changes nothing a `$ref` into the schema names
        schema = {
            "$id": "https://example.com/schemas/purge",
            "properties": {"p": {"$ref": "#/$defs/n"}},
            "$defs": {"n": {"type": "integer", "default": 3}},
        }
        assert self.judged({}, schema)[1] == {"p": 3}

    # --- nothing leaves the gate but a refusal ---------------------------------------------------

    def test_an_exception_that_is_not_an_exception_is_a_refusal(self, example, monkeypatch):
        """On Python 3.10 a panic in ``referencing``'s compiled map arrived as a
        ``BaseException`` and left the gate."""
        module = self.schema_module()

        def panics(self, validator, instance):
            raise _Panic("a panic in a compiled dependency")

        monkeypatch.setattr(module.Listed, "_validate", panics)
        assert self.judged({"p": 1}, self.ANY) == (
            None,
            None,
            f"Recusal refused `purge`: the arguments could not be checked against {_SCHEMA} "
            "(_Panic)",
        )
        with pytest.raises(module.Refused, match="could not be checked"):
            module.Listed(self.ANY).complete({"p": 1})

    @pytest.mark.parametrize(
        "stops", [KeyboardInterrupt, SystemExit, GeneratorExit, asyncio.CancelledError]
    )
    def test_what_stops_the_interpreter_or_the_loop_is_passed_on(self, example, monkeypatch, stops):
        module = self.schema_module()
        assert module.PASSED_ON == (
            KeyboardInterrupt,
            SystemExit,
            GeneratorExit,
            asyncio.CancelledError,
        )

        def stopped(self, validator, instance):
            raise stops()

        monkeypatch.setattr(module.Listed, "_validate", stopped)
        with pytest.raises(stops):
            self.judged({"p": 1}, self.ANY)

        def policy(name, arguments):
            raise stops()

        monkeypatch.undo()
        with pytest.raises(stops):
            _adapter()._judged_call("purge", {"p": 1}, self.ANY, policy)

    def test_a_policy_that_raises_what_is_not_an_exception_fails_closed(self, example):
        def policy(name, arguments):
            raise _Panic

        assert _adapter()._judged_call("purge", {"p": 1}, self.ANY, policy) == (
            None,
            "Recusal refused `purge`: the gate could not judge this call (_Panic)",
        )

    def test_whatever_leaves_the_gates_own_reading_is_a_refusal(self, example, monkeypatch):
        gate = _adapter()

        def no_room(*args):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(gate, "_measure", no_room)
        assert gate._judged_call("purge", {"p": 1}, self.ANY, lambda *_: []) == (
            None,
            "Recusal refused `purge`: the gate could not judge this call within what is left "
            "of the interpreter's recursion limit where it was called",
        )

        def broken(*args):
            raise ZeroDivisionError

        monkeypatch.setattr(gate, "_measure", broken)
        assert gate._judged_call("purge", {"p": 1}, self.ANY, lambda *_: []) == (
            None,
            "Recusal refused `purge`: the gate could not judge this call (ZeroDivisionError)",
        )
        # and where the refusal cannot be put on the record either, it stands
        monkeypatch.setattr(gate, "_unjudgeable", no_room)
        assert gate._judged_call("purge", {"p": 1}, self.ANY, lambda *_: []) == (
            None,
            "Recusal failed closed: the gate could not judge this call (RecursionError)",
        )

        def panics(*args):
            raise _Panic

        monkeypatch.setattr(gate, "_unjudgeable", panics)
        assert gate._judged_call("purge", {"p": 1}, self.ANY, lambda *_: []) == (
            None,
            "Recusal failed closed: the gate could not judge this call (_Panic)",
        )

    def test_a_schema_the_recursion_limit_stopped_the_reading_of_is_read_again(
        self, example, monkeypatch
    ):
        """Where the limit is reached depends on how deep in the application's stack the
        gate was called, so that refusal is of the call and is not kept for the listing."""
        module = self.schema_module()
        scan = module._scan
        calls = []

        def out_of_stack(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RecursionError("maximum recursion depth exceeded")
            return scan(*args, **kwargs)

        monkeypatch.setattr(module, "_scan", out_of_stack)
        listed = module.Listed(self.ANY)
        told = (
            f"{_SCHEMA} is nested too deeply to read within what is left of the interpreter's "
            "recursion limit where the gate was called"
        )
        with pytest.raises(module.Refused) as refused:
            listed.complete({"p": 1})
        assert refused.value.reason == told
        assert listed.complete({"p": 1}) == ({"p": 1}, {"p": 1})
        # a schema that really is nested deeper than the interpreter reads says the same
        monkeypatch.undo()
        deep: dict = {"type": "integer"}
        for _ in range(3 * sys.getrecursionlimit()):
            deep = {"properties": {"k": deep}}
        assert self.judged({"k": 1}, deep) == (None, None, f"Recusal refused `purge`: {told}")

    def test_from_any_depth_of_the_applications_stack_the_gate_answers(self, example):
        """The reviewer's probe: the gate called from every depth of the stack up to the
        interpreter's recursion limit, with arguments a recursive schema recurses on.
        Each call completes or is refused; none raises, a ``BaseException`` included."""
        gate, module = _adapter(), self.schema_module()
        listed = module.Listed(TREE)
        chain: dict = {}
        for _ in range(40):
            chain = {"child": chain}
        arguments = {"node": chain}
        outcomes: dict = {}

        def room():
            return None

        def judge():
            try:
                room()
            except RecursionError:
                return "no room to call the gate at all"
            sent, reason = gate._judged_call("purge", arguments, listed, lambda *_: [])
            if sent is not None:
                return "completed"
            return reason

        def under(frames):
            return under(frames - 1) if frames else judge()

        for frames in range(0, sys.getrecursionlimit(), 3):
            try:
                outcome = under(frames)
            except RecursionError:
                outcome = "the test's own frames reached the limit"
            outcomes.setdefault(outcome, []).append(frames)
        limit = (
            "Recusal refused `purge`: the arguments are nested too deeply to check against "
            f"{_SCHEMA} within what is left of the interpreter's recursion limit where the "
            "gate was called"
        )
        allowed = {
            "completed",
            limit,
            "no room to call the gate at all",
            "the test's own frames reached the limit",
            "Recusal refused `purge`: the gate could not judge this call within what is left "
            "of the interpreter's recursion limit where it was called",
            "Recusal failed closed: the gate could not judge this call (RecursionError)",
            f"Recusal refused `purge`: the arguments could not be checked against {_SCHEMA} "
            "(PanicException)",
        }
        assert set(outcomes) <= allowed, sorted(set(outcomes) - allowed)
        assert "completed" in outcomes and limit in outcomes
        # the schema was read at the first call, so no later refusal is of the schema
        assert listed._failure is None

    # --- the `false` schema --------------------------------------------------------------------

    FALSE = [
        ({"properties": {"p": False}}, {"p": 1}, "argument `p`"),
        ({"properties": {"o": {"properties": {"p": False}}}}, {"o": {"p": 1}}, "argument `o.p`"),
        (
            {"$schema": DRAFT_07, "properties": {"p": {"items": False}}},
            {"p": [1, 2]},
            "argument `p[0]`",
        ),
        (
            {"properties": {"p": {"prefixItems": [{}, False]}}},
            {"p": [1, 2]},
            "argument `p[1]`",
        ),
        (
            {
                "properties": {"p": {"additionalProperties": {"$ref": "#/$defs/never"}}},
                "$defs": {"never": False},
            },
            {"p": {"k": 1}},
            "argument `p.*`",
        ),
        ({"properties": {"p": {"allOf": [False]}}}, {"p": 1}, "argument `p`"),
        ({"properties": {"p": {"if": {}, "then": False}}}, {"p": 1}, "argument `p`"),
        ({"$ref": "#/$defs/never", "$defs": {"never": False}}, {"p": 1}, "the arguments"),
    ]

    @pytest.mark.parametrize("schema, given, place", FALSE, ids=[str(n) for n in range(len(FALSE))])
    def test_a_false_schema_is_named_for_what_it_is_at_the_value_it_refuses(
        self, example, schema, given, place
    ):
        """It read "does not meet `None`", at the list or the object holding the value."""
        verb = "stand" if place == "the arguments" else "stands"
        assert self.judged(given, schema) == (
            None,
            None,
            f"Recusal refused `purge`: {place} {verb} where the server's schema allows no "
            f"value (`false`); {_AS_THEY_STAND}",
        )

    def test_member_names_no_schema_allows_are_named_by_their_keyword(self, example):
        schema = {"properties": {"p": {"propertyNames": False}}}
        assert self.judged({"p": {}}, schema) == ({"p": {}}, {"p": {}}, None)
        assert self.judged({"p": {"a": 1, "b": 2}}, schema) == (
            None,
            None,
            "Recusal refused `purge`: argument `p` does not meet `propertyNames` in the "
            f"server's schema; {_AS_THEY_STAND}",
        )

    def test_no_reason_names_a_keyword_none(self, example):
        for schema, given, _ in self.FALSE:
            assert "`None`" not in self.judged(given, schema)[2]


class SimpleRequest:
    """The approval request the SDK hands ``on_approval_request``: the item under ``data``."""

    def __init__(self, data):
        self.data = data
