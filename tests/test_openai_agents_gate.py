"""Regression tests for the offline OpenAI Agents SDK + Recusal guardrail example."""

import asyncio
import copy
import subprocess
import sys
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "openai_agents_gate.py"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"


def test_python_version_refusal_is_clear_on_every_python():
    code = (
        "import runpy, sys; "
        "sys.version_info = (3, 9, 0, 'final', 0); "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "openai_agents_gate.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_the_sdk_stays_out_of_the_recusal_package_and_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "openai" not in pyproject
    for path in (ROOT / "recusal").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "openai" not in source.lower(), path
        assert "from agents" not in source and "import agents" not in source, path


def test_pinned_versions_are_explicit_in_example_docs_and_ci():
    expected = ("openai-agents==0.24.0", "openai==3.34.0")
    for path in (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        text = path.read_text(encoding="utf-8")
        for pin in expected:
            assert pin in text, f"{path.name}: {pin}"
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for pin in expected:
        assert ci.count(pin) == 2, pin


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestOpenAIAgentsGate:
    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_openai_agents_gate_example", EXAMPLE)
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

    def test_allowed_call_runs_its_tool(self, example):
        outputs, executed = example.run_calls([("run_sql", {"sql": SELECT}, "c1")])
        assert executed == [("run_sql", {"sql": SELECT})]
        assert outputs == {"c1": f"recorded SQL: {SELECT}"}

    def test_refused_call_does_not_run_and_reason_reaches_the_model(self, example):
        outputs, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")])
        assert executed == []
        assert outputs["c1"].startswith("Recusal refused `run_sql` [FAIL]")
        assert "top-level WHERE" in outputs["c1"]

    @pytest.mark.parametrize("streamed", [False, True], ids=["run", "run_streamed"])
    def test_several_calls_in_one_turn_are_gated_separately(self, example, streamed):
        outputs, executed = example.run_calls(self.three_calls(), streamed=streamed)
        assert outputs["c1"] == "hello"
        assert "Recusal refused" in outputs["c2"]
        assert outputs["c3"] == f"recorded SQL: {SELECT}"
        assert sorted(executed) == [("echo", {"text": "hello"}), ("run_sql", {"sql": SELECT})]

    def test_policy_exception_fails_closed(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        outputs, executed = example.run_calls(
            [("echo", {"text": "unsafe"}, "c1")], policy=broken_policy
        )
        assert executed == []
        assert outputs["c1"] == "Recusal failed closed (policy error): policy exploded"

    def test_a_crash_inside_decide_also_fails_closed(self, example, monkeypatch):
        # decide() already denies when a policy raises; the guardrail's own except covers
        # a failure anywhere else in the decision path
        def crashing_decide(*args, **kwargs):
            raise RuntimeError("decide crashed")

        monkeypatch.setattr(example, "decide", crashing_decide)
        outputs, executed = example.run_calls([("echo", {"text": "unsafe"}, "c1")])
        assert executed == []
        assert outputs["c1"] == "Recusal failed closed (policy error): decide crashed"

    @pytest.mark.parametrize(
        "arguments, reason",
        [
            ('{"sql": ', "arguments are not valid JSON"),
            ('["DELETE FROM audit_log"]', "arguments are not a JSON object"),
            ('"DELETE FROM audit_log"', "arguments are not a JSON object"),
        ],
    )
    def test_malformed_and_non_object_arguments_are_refused(self, example, arguments, reason):
        outputs, executed = example.run_calls([("run_sql", arguments, "c1")])
        assert executed == []
        assert outputs["c1"] == f"Recusal refused `run_sql`: {reason}"

    @pytest.mark.parametrize(
        "arguments, runs",
        [
            ('{"sql": "SELECT 1", "sql": "DELETE FROM t"}', None),
            ('{"sql": "DELETE FROM t", "sql": "SELECT 1"}', "SELECT 1"),
        ],
    )
    def test_duplicate_keys_are_judged_as_the_tool_reads_them(self, example, arguments, runs):
        # the SDK and the guardrail both decode with json.loads, so the last key wins in both
        outputs, executed = example.run_calls([("run_sql", arguments, "c1")])
        assert executed == ([] if runs is None else [("run_sql", {"sql": runs})])

    def test_negative_control_runs_delete_without_the_guardrail(self, example):
        outputs, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")], gated=False)
        assert executed == [("run_sql", {"sql": DELETE})]
        assert outputs["c1"] == f"recorded SQL: {DELETE}"

    def test_gate_tools_refuses_every_non_function_tool_in_the_sdk_union(self, example):
        import agents.tool

        members = [typing.get_origin(m) or m for m in typing.get_args(agents.tool.Tool)]
        assert example.FunctionTool in members
        others = [m for m in members if m is not example.FunctionTool]
        assert len(others) >= 12  # guards the iteration: the 0.24.0 union has 12 others
        for cls in others:
            with pytest.raises(TypeError, match=f"cannot gate {cls.__name__} "):
                example.gate_tools([cls.__new__(cls)], example.sql_scope_policy)

    def test_gate_tools_leaves_the_originals_unchanged_and_keeps_their_guardrails(self, example):
        @example.tool_input_guardrail
        def existing(data):
            return example.ToolGuardrailFunctionOutput.allow()

        tool = copy.copy(example.run_sql)
        tool.tool_input_guardrails = [existing]
        (gated,) = example.gate_tools([tool], example.sql_scope_policy)
        assert tool.tool_input_guardrails == [existing]
        assert gated.tool_input_guardrails[0] is existing
        assert [g.name for g in gated.tool_input_guardrails] == ["existing", "recusal"]
        assert not example.run_sql.tool_input_guardrails

    def test_gate_tools_refuses_the_codex_tool(self, example):
        from agents.extensions.experimental.codex import codex_tool

        codex = codex_tool()
        with pytest.raises(TypeError, match="Codex tool"):
            example.gate_tools([codex], example.sql_scope_policy)

    def test_gated_agent_refuses_mcp_servers_handoffs_and_agent_subclasses(self, example):
        from agents.mcp import MCPServerStdio
        from agents.sandbox import SandboxAgent

        Agent = example.Agent
        server = MCPServerStdio(params={"command": "never-started"})
        cases = [
            (Agent(name="a", mcp_servers=[server]), "mcp_servers"),
            (Agent(name="a", handoffs=[Agent(name="b")]), "handoffs"),
            (SandboxAgent(name="a"), "SandboxAgent is not a plain Agent"),
        ]
        for agent, reason in cases:
            with pytest.raises(ValueError, match=reason):
                example.gated_agent(agent, example.sql_scope_policy)

    def test_gated_agent_returns_a_copy(self, example):
        agent = example.Agent(name="a", tools=[example.echo, example.run_sql])
        gated = example.gated_agent(agent, example.sql_scope_policy)
        assert gated is not agent
        assert agent.tools == [example.echo, example.run_sql]
        assert example.is_gated(gated) and not example.is_gated(agent)

    def _parent_and_sub(self, example, *, gate_sub):
        """A parent whose only tool is a sub-agent that proposes DELETE."""
        sub = example.Agent(
            name="sub",
            model=example.scripted_model([("run_sql", {"sql": DELETE}, "s1")]),
            tools=[example.run_sql],
        )
        if gate_sub:
            sub = example.gated_agent(sub, example.sql_scope_policy)
        parent_model = example.scripted_model([("ask_sub", {"input": "tidy the log"}, "p1")])
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask the sub-agent.")
        return parent_model, example.Agent(name="parent", model=parent_model, tools=[tool])

    def test_an_agent_tool_whose_sub_agent_is_ungated_is_refused(self, example):
        _, parent = self._parent_and_sub(example, gate_sub=False)
        with pytest.raises(TypeError, match="its agent's own tool calls are not gated"):
            example.gated_agent(parent, example.sql_scope_policy)

    def test_a_gated_sub_agent_refuses_its_own_calls(self, example):
        _, parent = self._parent_and_sub(example, gate_sub=True)
        example.EXECUTED.clear()
        asyncio.run(example.run_agent(example.gated_agent(parent, example.sql_scope_policy)))
        assert example.EXECUTED == []

    def test_negative_control_a_gate_on_the_agent_tool_alone_misses_the_sub_agent(self, example):
        # what gate_tools refuses to build: the parent's guardrail sees only the sub-agent's
        # input, and the ungated sub-agent runs the DELETE
        _, parent = self._parent_and_sub(example, gate_sub=False)
        (tool,) = parent.tools
        tool.tool_input_guardrails = [example.recusal_guardrail(example.sql_scope_policy)]
        example.EXECUTED.clear()
        asyncio.run(example.run_agent(parent))
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]

    @pytest.mark.parametrize("sql, ran", [(SELECT, True), (DELETE, False)])
    def test_an_approved_call_is_still_judged_when_it_runs(self, example, sql, ran):
        tool = copy.copy(example.run_sql)
        tool.needs_approval = True
        model = example.scripted_model([("run_sql", {"sql": sql}, "c1")])
        agent = example.gated_agent(
            example.Agent(name="a", model=model, tools=[tool]), example.sql_scope_policy
        )
        config = example.RunConfig(tracing_disabled=True)

        async def approve_and_resume():
            result = await example.Runner.run(agent, "go", run_config=config)
            assert len(result.interruptions) == 1
            assert example.EXECUTED == []
            state = result.to_state()
            state.approve(result.interruptions[0])
            await example.Runner.run(agent, state, run_config=config)

        example.EXECUTED.clear()
        asyncio.run(approve_and_resume())
        assert example.EXECUTED == ([("run_sql", {"sql": sql})] if ran else [])

    def test_example_runs_end_to_end(self, example):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "run_sql (c2): Recusal refused `run_sql` [FAIL]" in result.stdout
        assert "tools that ran: 2" in result.stdout


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestEffectiveArguments:
    """Review 5, P1: the guardrail judged the model's raw JSON; the SDK then validated it
    through the tool's parameter model, and a validator, a default or a coercion could make
    the function receive something else. The gate now judges the invocation the SDK prepared
    and pins it, so what was judged is what executes; a schema the SDK cannot prepare (user
    code in the model) is refused."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_openai_effective_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    class Recording:
        def __init__(self, policy):
            self.policy, self.seen = policy, []

        def __call__(self, name, args):
            self.seen.append((name, dict(args)))
            return self.policy(name, args)

    def test_a_schema_that_rewrites_an_allowed_proposal_is_refused_and_nothing_runs(self, example):
        policy = self.Recording(example.sql_scope_policy)
        outputs, executed = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")],
            policy=policy,
            tools=[example.transforming_sql_tool()],
        )
        assert executed == []
        assert policy.seen == []  # nothing to judge: the value that would run is unknowable
        assert "cannot be judged" in outputs["c1"] and "custom validation" in outputs["c1"]

    def test_negative_control_the_same_rewriting_tool_runs_the_rewritten_sql_ungated(self, example):
        outputs, executed = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")],
            gated=False,
            tools=[example.transforming_sql_tool()],
        )
        assert executed == [("run_sql", {"sql": DELETE})]  # the recorder is live; the gate is why

    def test_defaults_and_coercion_are_judged_as_the_tool_receives_them(self, example):
        from recusal import Finding

        ran = []

        @example.function_tool(strict_mode=False)
        def charge(amount: int = 100) -> str:
            """Record a charge; contacts nothing."""
            ran.append((amount, type(amount).__name__))
            return "recorded"

        def budget(name, args):
            amount = args.get("amount", 0)
            if isinstance(amount, int) and amount > 10:
                return [Finding.fail("amount_limit", severity="CRITICAL", message="over 10")]
            return []

        for proposed, expect_ran, expect_seen in [
            ({}, [], {"amount": 100}),  # the default, 100, is what would run: refused
            ({"amount": "50"}, [], {"amount": 50}),  # the string the raw check missed: refused
            ({"amount": "5"}, [(5, "int")], {"amount": 5}),  # coerced, allowed, executed as judged
            ({"amount": 7}, [(7, "int")], {"amount": 7}),
        ]:
            ran.clear()
            policy = self.Recording(budget)
            outputs, _ = example.run_calls(
                [("charge", proposed, "c1")], policy=policy, tools=[charge]
            )
            assert policy.seen == [("charge", expect_seen)], proposed
            assert ran == expect_ran, (proposed, outputs)
            assert (outputs["c1"] == "recorded") is bool(expect_ran), outputs

    def test_a_context_taking_tool_is_judged_by_its_data_parameters(self, example):
        ran = []

        @example.function_tool
        def ctx_sql(ctx: example.RunContextWrapper, sql: str) -> str:
            """Record SQL with a context parameter the SDK fills."""
            ran.append(sql)
            return "recorded"

        policy = self.Recording(example.sql_scope_policy)
        outputs, _ = example.run_calls(
            [("ctx_sql", {"sql": SELECT}, "c1")], policy=policy, tools=[ctx_sql]
        )
        assert policy.seen == [("ctx_sql", {"sql": SELECT})] and ran == [SELECT]

    def test_a_nested_application_model_is_refused_because_the_sdk_will_not_prepare_it(
        self, example
    ):
        """The SDK prepares only data-only parameter models; an application model inside
        the parameters could run code, so nothing is prepared and the gate refuses. A
        dict parameter carries the same data and is judged."""
        from pydantic import BaseModel

        class Query(BaseModel):
            sql: str
            limit: int = 10

        ran = []

        @example.function_tool
        def query(q: Query) -> str:
            """Run a structured query; records only."""
            ran.append(q.model_dump())
            return "recorded"

        @example.function_tool(strict_mode=False)
        def query_dict(q: dict) -> str:
            """The same query as plain data."""
            ran.append(dict(q))
            return "recorded"

        policy = self.Recording(lambda n, a: [])
        outputs, _ = example.run_calls(
            [("query", {"q": {"sql": SELECT}}, "c1")], policy=policy, tools=[query, query_dict]
        )
        assert ran == [] and policy.seen == [] and "cannot be judged" in outputs["c1"]
        outputs, _ = example.run_calls(
            [("query_dict", {"q": {"sql": SELECT}}, "c2")], policy=policy, tools=[query, query_dict]
        )
        assert policy.seen == [("query_dict", {"q": {"sql": SELECT}})]
        assert ran == [{"sql": SELECT}] and outputs["c2"] == "recorded"

    def test_arguments_that_do_not_validate_are_refused_by_the_gate(self, example):
        policy = self.Recording(example.sql_scope_policy)
        outputs, executed = example.run_calls([("run_sql", {"sql": 5}, "c1")], policy=policy)
        assert executed == [] and policy.seen == []
        assert "cannot be judged" in outputs["c1"]

    def test_a_hand_built_function_tool_is_refused_at_setup(self, example):
        async def invoke(ctx, arguments: str) -> str:
            return "ran"

        hand_built = example.FunctionTool(
            name="raw",
            description="a tool built by hand",
            params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
            on_invoke_tool=invoke,
        )
        with pytest.raises(TypeError, match="function_tool"):
            example.gate_tools([hand_built], example.sql_scope_policy)

    def test_the_ordinary_path_is_unchanged_and_still_judges_the_exact_sql(self, example):
        policy = self.Recording(example.sql_scope_policy)
        outputs, executed = example.run_calls([("run_sql", {"sql": SELECT}, "c1")], policy=policy)
        assert policy.seen == [("run_sql", {"sql": SELECT})]
        assert executed == [("run_sql", {"sql": SELECT})]
