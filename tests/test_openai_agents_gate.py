"""Regression tests for the offline OpenAI Agents SDK + Recusal guardrail example."""

import asyncio
import copy
import re
import subprocess
import sys
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "openai_agents_gate.py"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"


def _adapter():
    """The package module the gate's code lives in. Every test that calls this has loaded
    the example first, which puts the in-tree adapter on the path, so this is the code
    under review and not an installed copy."""
    import importlib
    from pathlib import Path

    module = importlib.import_module("recusal_openai_agents._gate")
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
    expected = ("openai-agents==0.24.0", "openai==3.49.0")
    for path in (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        text = path.read_text(encoding="utf-8")
        for pin in expected:
            assert pin in text, f"{path.name}: {pin}"
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # both are roots of the hash-locked files the pull request lanes install: the two
    # example jobs and the recusal-openai-agents wheel job install one, the
    # recusal-openai-agents-mcp wheel job another, and the recusal-openai wheel job's
    # carries the SDK alone
    for lock, pins in (
        ("adapter-openai-agents", expected),
        ("adapter-openai-agents-mcp", expected),
        ("adapter-openai", ("openai==3.49.0",)),
    ):
        roots = (ROOT / "requirements" / f"{lock}.txt").read_text(encoding="utf-8").splitlines()[0]
        for pin in pins:
            assert pin in roots.split(), (lock, pin)
    assert ci.count("requirements/adapter-openai-agents.txt") == 3
    assert ci.count("requirements/adapter-openai-agents-mcp.txt") == 1
    assert ci.count("requirements/adapter-openai.txt") == 1


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
        assert (
            outputs["c1"]
            == "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        )

    def test_a_crash_inside_decide_also_fails_closed(self, example, monkeypatch):
        # decide() already denies when a policy raises; the guardrail's own except covers
        # a failure anywhere else in the decision path
        def crashing_decide(*args, **kwargs):
            raise RuntimeError("decide crashed")

        monkeypatch.setattr(_adapter(), "decide", crashing_decide)
        outputs, executed = example.run_calls([("echo", {"text": "unsafe"}, "c1")])
        assert executed == []
        assert (
            outputs["c1"]
            == "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        )

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
        @_adapter().tool_input_guardrail
        def existing(data):
            return _adapter().ToolGuardrailFunctionOutput.allow()

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
        # what gate_tools refuses to build: the parent's guardrail (bound by hand to the
        # agent tool) sees only the sub-agent's input, and the ungated sub-agent runs the DELETE
        _, parent = self._parent_and_sub(example, gate_sub=False)
        (tool,) = parent.tools
        holder = [tool]
        tool.tool_input_guardrails = [_adapter()._guardrail_for(example.sql_scope_policy, holder)]
        example.EXECUTED.clear()
        asyncio.run(example.run_agent(parent))
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]

    def test_a_standalone_recusal_guardrail_is_bound_to_nothing_and_refuses(self, example):
        tool = copy.copy(example.run_sql)
        tool.tool_input_guardrails = [example.recusal_guardrail(example.sql_scope_policy)]
        model = example.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
        example.EXECUTED.clear()
        asyncio.run(example.run_agent(example.Agent(name="a", model=model, tools=[tool])))
        assert example.EXECUTED == []
        assert "bound to no tool" in example.tool_outputs(model)["c1"]

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
        """Refused twice over: when the agent is built, so the developer learns then, and by
        the guardrail's own answer for a call, should the tool reach one another way."""
        import json

        policy = self.Recording(example.sql_scope_policy)
        tool = example.transforming_sql_tool()
        for build in (
            lambda: example.gate_tools([tool], policy),
            lambda: example.run_calls([("run_sql", {"sql": SELECT}, "c1")], tools=[tool]),
        ):
            with pytest.raises(TypeError, match="cannot gate 'run_sql': parameter `sql` has"):
                build()
        example.EXECUTED.clear()
        reason = _adapter()._function_tool_refusal(
            "run_sql", json.dumps({"sql": SELECT}), policy, tool
        )
        assert reason.startswith("Recusal refused `run_sql`: parameter `sql` has a type or a")
        assert "a custom validator" in reason and "no call could be judged" in reason
        assert policy.seen == []  # nothing to judge: the value that would run is unknowable
        assert example.EXECUTED == []

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
        from agents import RunContextWrapper

        ran = []

        @example.function_tool
        def ctx_sql(ctx: RunContextWrapper, sql: str) -> str:
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
        the parameters could run code, so nothing is prepared and the gate refuses the tool
        when the agent is built. A dict parameter carries the same data and is judged."""
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
        with pytest.raises(TypeError, match="cannot gate 'query': parameter `q` has a type"):
            example.run_calls(
                [("query", {"q": {"sql": SELECT}}, "c1")], policy=policy, tools=[query, query_dict]
            )
        assert ran == [] and policy.seen == []
        outputs, _ = example.run_calls(
            [("query_dict", {"q": {"sql": SELECT}}, "c2")], policy=policy, tools=[query_dict]
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


class TestCapabilityIdentity:
    """Adapters review, finding 1: the guardrail looked its tool up by name (first match)
    while the SDK executes the last tool of that name. With a gated sub-agent exposed as
    `run_sql` listed before a function tool of the same name whose schema rewrites SELECT
    to DELETE, the guardrail judged raw SELECT and the function tool ran DELETE. Each
    guardrail is now bound to its own tool, and shared names are refused at setup."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_openai_identity_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def sub_agent_as_run_sql(example):
        sub = example.Agent(
            name="sub",
            model=example.scripted_model([("echo", {"text": "hi"}, "s1")]),
            tools=[example.echo],
        )
        sub = example.gated_agent(sub, example.sql_scope_policy)
        return sub.as_tool(tool_name="run_sql", tool_description="A sub-agent named like the tool.")

    @pytest.mark.parametrize(
        "agent_tool_first", [True, False], ids=["agent-tool-first", "function-first"]
    )
    @pytest.mark.parametrize("streamed", [False, True], ids=["run", "streamed"])
    def test_the_reviewers_duplicate_configuration_is_refused_at_setup(
        self, example, agent_tool_first, streamed
    ):
        agent_tool = self.sub_agent_as_run_sql(example)
        function = example.transforming_sql_tool()
        tools = [agent_tool, function] if agent_tool_first else [function, agent_tool]
        with pytest.raises(TypeError, match="share a name \\['run_sql'\\]"):
            example.gate_tools(tools, example.sql_scope_policy)
        model = example.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
        agent = example.Agent(name="dup", model=model, tools=tools)
        with pytest.raises(ValueError, match="share a name"):
            example.gated_agent(agent, example.sql_scope_policy)
        example.EXECUTED.clear()
        assert example.EXECUTED == [], "nothing ran: the agent was never built"

    @staticmethod
    def limited_sql_tool(example):
        """A second, data-only tool named ``run_sql``: its prepared invocation carries a
        default the raw arguments do not."""

        @example.function_tool(name_override="run_sql", strict_mode=False)
        def run_sql(sql: str, limit: int = 10) -> str:
            """Record SQL text and a row limit without connecting to a database."""
            example.EXECUTED.append(("run_sql", {"sql": sql, "limit": limit}))
            return f"recorded SQL: {sql}"

        return run_sql

    def test_is_gated_is_false_for_an_agent_whose_gated_tools_share_a_name(self, example):
        (a,) = example.gate_tools([example.run_sql], example.sql_scope_policy)
        (b,) = example.gate_tools([self.limited_sql_tool(example)], example.sql_scope_policy)
        model = example.scripted_model([])
        agent = example.Agent(name="dup", model=model, tools=[a, b])
        assert not example.is_gated(agent)
        assert example.is_gated(example.Agent(name="one", model=model, tools=[a]))

    def test_a_bound_guardrail_judges_its_own_tool_whatever_the_agent_lists(self, example):
        """Kills the lookup mutant on its own: the agent's metadata lists a same-named
        agent tool FIRST, which a name lookup would find and judge as the raw arguments;
        the bound guardrail judges the function tool it was built for, as the SDK prepares
        it, with the default the raw arguments leave out."""
        import json
        from types import SimpleNamespace

        judged = []

        def recording_policy(name, tool_input):
            judged.append(dict(tool_input))
            return example.sql_scope_policy(name, tool_input)

        (gated,) = example.gate_tools([self.limited_sql_tool(example)], recording_policy)
        decoy = self.sub_agent_as_run_sql(example)
        misleading_agent = SimpleNamespace(name="misleading", tools=[decoy, gated])
        guardrail = gated.tool_input_guardrails[-1]
        ctx = SimpleNamespace(tool_name="run_sql", tool_arguments=json.dumps({"sql": SELECT}))
        guardrail.guardrail_function(SimpleNamespace(context=ctx, agent=misleading_agent))
        assert judged == [{"sql": SELECT, "limit": 10}]
        assert ctx._function_tool_arguments.args == [SELECT, 10]

    def test_an_allowed_call_pins_the_invocation_that_was_judged(self, example):
        """The gate judges the invocation the SDK prepared and pins that same object on the
        tool context, so the SDK executes it without preparing a second time. A refused
        call pins nothing."""
        import json
        from types import SimpleNamespace

        from recusal import Finding

        judged = []

        def recording_policy(name, tool_input):
            judged.append(dict(tool_input))
            return example.sql_scope_policy(name, tool_input)

        (gated,) = example.gate_tools([example.charge_tool()], recording_policy)
        guardrail = gated.tool_input_guardrails[-1]
        prepare = gated.on_invoke_tool.prepare_arguments

        ctx = SimpleNamespace(tool_name="charge", tool_arguments=json.dumps({}))
        guardrail.guardrail_function(SimpleNamespace(context=ctx, agent=None))
        assert judged == [{"amount": 100}], "the omitted default is judged at its value"
        pinned = ctx._function_tool_arguments
        assert pinned is not None
        assert pinned.kwargs == prepare(json.dumps({}), "charge").kwargs

        def refuse(name, tool_input):
            return [Finding.fail("no", severity="CRITICAL", message="no")]

        (refusing,) = example.gate_tools([example.charge_tool()], refuse)
        ctx = SimpleNamespace(tool_name="charge", tool_arguments=json.dumps({}))
        refusing.tool_input_guardrails[-1].guardrail_function(
            SimpleNamespace(context=ctx, agent=None)
        )
        assert not getattr(ctx, "_function_tool_arguments", None)

    def test_a_bound_guardrail_refuses_a_call_for_another_name(self, example):
        import json
        from types import SimpleNamespace

        (gated,) = example.gate_tools([example.run_sql], example.sql_scope_policy)
        guardrail = gated.tool_input_guardrails[-1]
        ctx = SimpleNamespace(tool_name="echo", tool_arguments=json.dumps({"text": "x"}))
        output = guardrail.guardrail_function(SimpleNamespace(context=ctx, agent=None))
        assert "bound to 'run_sql'" in str(output)

    def test_unique_names_run_exactly_as_before(self, example):
        outputs, executed = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1"), ("echo", {"text": "hi"}, "c2")]
        )
        assert executed == [("run_sql", {"sql": SELECT}), ("echo", {"text": "hi"})]


#: A callable-object tool whose context parameter is annotated through an alias, in a
#: module with string annotations: neither the text ``RunContextWrapper`` nor the class is
#: in the annotation as written, and only resolving it the SDK's way finds the context.
_ALIASED_CONTEXT_SOURCE = '''
from __future__ import annotations

from typing import Any

from agents import RunContextWrapper as RCW

Ctx = RCW[Any]


class SqlRunner:
    def __init__(self, ran):
        self.ran = ran

    def __call__(self, ctx: Ctx, sql: str) -> str:
        """Record SQL text without connecting to a database."""
        self.ran.append(sql)
        return "recorded"
'''


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestJudgedIsWhatRuns:
    """The 2026-11-21 review. What the policy judged was not always what ran: a context
    parameter the gate detected differently from the SDK shifted every argument one name
    over; a wrapper's own parameter ran at a default no policy saw; the gated copy shared
    its ``handoffs`` list with the original. And a tool the gate could never judge was
    accepted when the agent was built and refused on every call."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_openai_judged_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    Recording = TestEffectiveArguments.Recording

    # --- the context parameter is the SDK's decision, never a second reading ----------------

    @pytest.mark.parametrize(
        "sql, runs", [(DELETE, False), (SELECT, True)], ids=["delete", "select"]
    )
    def test_an_aliased_context_on_a_callable_object_does_not_shift_the_names(
        self, example, sql, runs
    ):
        namespace: dict = {}
        exec(compile(_ALIASED_CONTEXT_SOURCE, "<aliased context>", "exec"), namespace)
        ran: list = []
        tool = example.function_tool(namespace["SqlRunner"](ran), name_override="run_sql")
        policy = self.Recording(example.sql_scope_policy)
        outputs, _ = example.run_calls(
            [("run_sql", {"sql": sql}, "c1")], policy=policy, tools=[tool]
        )
        # judged under the name the function receives it by, not under the context's
        assert policy.seen == [("run_sql", {"sql": sql})]
        assert ran == ([sql] if runs else [])
        assert (outputs["c1"] == "recorded") is runs, outputs

    def test_negative_control_the_aliased_context_tool_runs_delete_ungated(self, example):
        namespace: dict = {}
        exec(compile(_ALIASED_CONTEXT_SOURCE, "<aliased context>", "exec"), namespace)
        ran: list = []
        tool = example.function_tool(namespace["SqlRunner"](ran), name_override="run_sql")
        example.run_calls([("run_sql", {"sql": DELETE}, "c1")], gated=False, tools=[tool])
        assert ran == [DELETE]

    def test_prepared_values_that_are_not_one_per_parameter_are_refused(self, example):
        from types import SimpleNamespace

        @example.function_tool(strict_mode=False)
        def search(sql: str, *, dry: bool = False) -> str:
            """Record a search; contacts nothing."""
            return "recorded"

        gate = _adapter()
        schema = gate._sdk_schema(search)
        named = gate._effective_arguments(
            schema, SimpleNamespace(args=[SELECT], kwargs={"dry": True})
        )
        assert named == {"sql": SELECT, "dry": True}
        for args, kwargs in (
            ([], {"dry": True}),  # a value short: the names would shift
            ([SELECT, "one too many"], {"dry": True}),
            ([SELECT], {}),  # a keyword parameter left to the function's own default
            ([SELECT], {"dry": True, "force": True}),  # a keyword the tool does not declare
        ):
            with pytest.raises(TypeError, match="not one for each of the tool's parameters"):
                gate._effective_arguments(schema, SimpleNamespace(args=args, kwargs=kwargs))

    # --- a wrapper cannot run a value no policy judged ----------------------------------------

    @staticmethod
    def wrapped_tools(example, ran):
        import functools

        def adds_a_parameter(fn):
            @functools.wraps(fn)
            def wrapper(*args, dry_run=False, **kwargs):
                ran.append(("wrapper", dry_run))
                return fn(*args, **kwargs)

            return wrapper

        def takes_the_functions_parameter(fn):
            @functools.wraps(fn)
            def wrapper(sql, *args, **kwargs):
                return fn(*args, **kwargs)  # the function then runs on its own default

            return wrapper

        def reorders(fn):
            @functools.wraps(fn)
            def wrapper(table, sql):
                return fn(sql, table)

            return wrapper

        def forwards(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def run_sql(sql: str = DELETE) -> str:
            """Record SQL text without connecting to a database."""
            ran.append(("run_sql", sql))
            return "recorded"

        def copy_rows(sql: str, table: str) -> str:
            """Record a copy; contacts nothing."""
            ran.append(("copy_rows", sql, table))
            return "recorded"

        def tool(decorator, func):
            return example.function_tool(
                decorator(func), name_override=func.__name__, strict_mode=False
            )

        return {
            "adds": (tool(adds_a_parameter, run_sql), "dry_run"),
            "takes": (tool(takes_the_functions_parameter, run_sql), "sql"),
            "reorders": (tool(reorders, copy_rows), "table"),
            "forwards": (tool(forwards, run_sql), None),
        }

    @pytest.mark.parametrize("case", ["adds", "takes", "reorders"])
    def test_a_wrapper_with_a_parameter_of_its_own_is_refused_when_the_agent_is_built(
        self, example, case
    ):
        import json

        ran: list = []
        tool, parameter = self.wrapped_tools(example, ran)[case]
        expected = f"cannot gate {tool.name!r}: a function the call reaches takes `{parameter}`"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)
        with pytest.raises(TypeError, match=expected):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)
        # and the guardrail's own answer for a call, should the tool reach one another way
        policy = self.Recording(example.sql_scope_policy)
        reason = _adapter()._function_tool_refusal(
            tool.name, json.dumps({"sql": SELECT, "table": "t"}), policy, tool
        )
        assert reason.startswith(
            f"Recusal refused `{tool.name}`: a function the call reaches takes `{parameter}`"
        )
        assert policy.seen == [] and ran == []

    def test_negative_control_the_wrapper_that_takes_the_parameter_runs_delete_ungated(
        self, example
    ):
        ran: list = []
        tool, _ = self.wrapped_tools(example, ran)["takes"]
        example.run_calls([("run_sql", {"sql": SELECT}, "c1")], gated=False, tools=[tool])
        assert ran == [("run_sql", DELETE)]  # proposed SELECT, ran the function's default

    def test_a_wrapper_that_only_forwards_is_gated_and_judged_as_the_function(self, example):
        ran: list = []
        tool, _ = self.wrapped_tools(example, ran)["forwards"]
        policy = self.Recording(example.sql_scope_policy)
        outputs, _ = example.run_calls(
            [("run_sql", {}, "c1"), ("run_sql", {"sql": SELECT}, "c2")],
            policy=policy,
            tools=[tool],
        )
        # the function's default, DELETE, is written out by the SDK, judged and refused
        assert sorted(policy.seen, key=str) == sorted(
            [("run_sql", {"sql": DELETE}), ("run_sql", {"sql": SELECT})], key=str
        )
        assert ran == [("run_sql", SELECT)]
        assert outputs["c1"].startswith("Recusal refused `run_sql` [FAIL]")

    def test_a_function_that_declares_another_signature_is_refused(self, example):
        import inspect

        def run_sql(sql: str, limit: int = 10) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        run_sql.__signature__ = inspect.Signature(
            [inspect.Parameter("sql", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)]
        )
        tool = example.function_tool(run_sql)
        with pytest.raises(TypeError, match="cannot be inspected \\(DeclaredSignature\\)"):
            example.gate_tools([tool], example.sql_scope_policy)

    # --- the gated copy shares no list with the original --------------------------------------

    def test_the_gated_copy_has_its_own_lists(self, example):
        other = example.Agent(name="billing", tools=[example.run_sql])
        agent = example.Agent(name="triage", tools=[example.echo])
        gated = example.gated_agent(agent, example.sql_scope_policy)
        assert gated.handoffs is not agent.handoffs
        assert gated.mcp_servers is not agent.mcp_servers
        assert gated.tools is not agent.tools
        agent.handoffs.append(other)  # the original changes after gating
        agent.mcp_servers.append(object())
        assert gated.handoffs == [] and gated.mcp_servers == []
        assert example.is_gated(gated)

    def test_a_handoff_added_to_the_original_afterwards_is_not_the_gated_copys(self, example):
        from agents.exceptions import ModelBehaviorError

        billing = example.Agent(
            name="billing",
            model=example.scripted_model([("run_sql", {"sql": DELETE}, "b1")]),
            tools=[example.run_sql],
        )
        model = example.scripted_model([("transfer_to_billing", {}, "c1")])
        agent = example.Agent(name="triage", model=model, tools=[example.echo])
        gated = example.gated_agent(agent, example.sql_scope_policy)
        agent.handoffs.append(billing)
        example.EXECUTED.clear()
        with pytest.raises(ModelBehaviorError, match="transfer_to_billing"):
            asyncio.run(example.run_agent(gated))
        assert example.EXECUTED == []

    def test_an_agent_with_a_prompt_is_refused(self, example):
        prompt = {"id": "pmpt_stored"}
        agent = example.Agent(name="a", tools=[example.echo], prompt=prompt)
        with pytest.raises(ValueError, match="cannot gate agent 'a': it has a prompt"):
            example.gated_agent(agent, example.sql_scope_policy)
        gated = example.gated_agent(
            example.Agent(name="a", tools=[example.echo]), example.sql_scope_policy
        )
        assert example.is_gated(gated)
        gated.prompt = prompt
        assert not example.is_gated(gated)

    # --- what the gate could never judge is refused when the agent is built ------------------

    @staticmethod
    def typed_tool(example, annotation, name="typed"):
        def typed(value) -> str:
            """Record a value; contacts nothing."""
            example.EXECUTED.append((name, {"value": value}))
            return "recorded"

        typed.__annotations__ = {"value": annotation, "return": str}
        return example.function_tool(typed, name_override=name, strict_mode=False)

    @staticmethod
    def refused_annotations():
        import dataclasses
        import datetime
        import enum

        from pydantic import BaseModel, Field

        class Color(enum.Enum):
            RED = "red"

        from typing_extensions import TypedDict  # pydantic wants it before Python 3.12

        class Location(TypedDict):
            lat: float

        class Model(BaseModel):
            x: int

        @dataclasses.dataclass
        class Data:
            x: int

        return {
            "enum": Color,
            "typed-dict": Location,
            "model": Model,
            "dataclass": Data,
            "datetime": datetime.datetime,
            "date": datetime.date,
            "tuple": typing.Tuple[int, int],
            "set": typing.Set[str],
            "bytes": bytes,
            "list-of-typed-dict": typing.List[Location],
            "pattern": typing.Annotated[str, Field(pattern="^a")],
        }

    @pytest.mark.parametrize(
        "case",
        [
            "enum",
            "typed-dict",
            "model",
            "dataclass",
            "datetime",
            "date",
            "tuple",
            "set",
            "bytes",
            "list-of-typed-dict",
            "pattern",
        ],
    )
    def test_a_parameter_type_the_sdk_will_not_prepare_is_refused_at_setup(self, example, case):
        tool = self.typed_tool(example, self.refused_annotations()[case])
        expected = "cannot gate 'typed': parameter `value` has a type or a validation"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)
        agent = example.Agent(name="a", tools=[example.echo, tool])
        with pytest.raises(TypeError, match=expected):
            example.gated_agent(agent, example.sql_scope_policy)

    def test_the_refused_parameter_is_named_among_the_others(self, example):
        import enum

        class Mode(enum.Enum):
            FAST = "fast"

        @example.function_tool
        def search(sql: str, mode: Mode, limit: int) -> str:
            """Record a search; contacts nothing."""
            return "recorded"

        with pytest.raises(TypeError, match="cannot gate 'search': parameter `mode` has a type"):
            example.gate_tools([search], example.sql_scope_policy)

    @pytest.mark.parametrize(
        "annotation, value",
        [
            (str, "a"),
            (int, 1),
            (float, 1.5),
            (bool, True),
            (typing.Optional[int], None),
            (typing.List[str], ["a"]),
            (typing.Dict[str, int], {"a": 1}),
            (typing.Any, {"k": [1]}),
            (typing.Literal["a", "b"], "a"),
            (typing.Union[int, str], "a"),
        ],
        ids=["str", "int", "float", "bool", "optional", "list", "dict", "any", "literal", "union"],
    )
    def test_the_data_only_parameter_types_are_gated_and_judged(self, example, annotation, value):
        tool = self.typed_tool(example, annotation)
        policy = self.Recording(lambda name, args: [])
        outputs, executed = example.run_calls(
            [("typed", {"value": value}, "c1")], policy=policy, tools=[tool]
        )
        assert policy.seen == [("typed", {"value": value})]
        assert executed == [("typed", {"value": value})] and outputs["c1"] == "recorded"

    @pytest.mark.parametrize("stars", ["*", "**"])
    def test_a_variadic_parameter_is_refused_at_setup(self, example, stars):
        def many(sql: str, *parts: str) -> str:
            """Record SQL parts; contacts nothing."""
            return "recorded"

        def keyed(sql: str, **options: int) -> str:
            """Record SQL options; contacts nothing."""
            return "recorded"

        func, name = (many, "parts") if stars == "*" else (keyed, "options")
        tool = example.function_tool(func, strict_mode=False)
        expected = (
            f"cannot gate '{func.__name__}': its parameter `{re.escape(stars)}{name}` is variadic"
        )
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_a_tool_not_built_by_function_tool_has_no_schema_to_read(self, example):
        tool = copy.copy(example.run_sql)
        tool.on_invoke_tool = lambda ctx, arguments: "ran"
        with pytest.raises(TypeError, match="Build the tool with function_tool"):
            _adapter()._sdk_schema(tool)
        assert "function_tool" in _adapter()._tool_refusal(tool)

    # --- names shared across namespaces: refused, for the reason that is true ---------------

    def test_a_name_shared_across_namespaces_is_refused_for_what_the_policy_is_given(self, example):
        from agents import tool_namespace

        crm = tool_namespace(name="crm", description="crm tools", tools=[example.run_sql])
        erp = tool_namespace(name="erp", description="erp tools", tools=[example.run_sql])
        expected = (
            "function tools in different namespaces share a name \\['run_sql'\\]; a policy "
            "and the audit record are given a tool's own name without its namespace"
        )
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([*crm, *erp], example.sql_scope_policy)
        agent = example.Agent(name="a", tools=[*crm, *erp])
        with pytest.raises(ValueError, match=expected):
            example.gated_agent(agent, example.sql_scope_policy)
        # one namespace alone is gated, and the policy is given the tool's own name
        from agents.testing import ScriptedModel, assistant_message, function_call

        call = function_call("run_sql", {"sql": SELECT}, call_id="c1", namespace="crm")
        model = ScriptedModel([[call], [assistant_message("done")]])
        policy = self.Recording(example.sql_scope_policy)
        gated = example.gated_agent(example.Agent(name="a", model=model, tools=crm), policy)
        example.EXECUTED.clear()
        asyncio.run(example.run_agent(gated))
        assert policy.seen == [("run_sql", {"sql": SELECT})]
        assert example.EXECUTED == [("run_sql", {"sql": SELECT})]

    def test_a_name_shared_in_one_namespace_keeps_the_last_wins_reason(self, example):
        from agents import tool_namespace

        crm = tool_namespace(
            name="crm", description="crm tools", tools=[example.run_sql, example.run_sql]
        )
        for tools in (crm, [example.run_sql, example.run_sql]):
            with pytest.raises(TypeError, match="share a name \\['run_sql'\\]; the SDK resolves"):
                example.gate_tools(tools, example.sql_scope_policy)

    # --- a refusal says what happened ---------------------------------------------------------

    def test_a_tool_without_parameters_refuses_arguments_and_says_so(self, example):
        ran: list = []

        @example.function_tool(strict_mode=False)
        def ping() -> str:
            """Record a ping; contacts nothing."""
            ran.append("ping")
            return "pong"

        policy = self.Recording(lambda name, args: [])
        outputs, _ = example.run_calls(
            [("ping", {"extra": 1}, "c1"), ("ping", {}, "c2")], policy=policy, tools=[ping]
        )
        assert outputs["c1"] == (
            "Recusal refused `ping`: the arguments the tool would execute cannot be judged: "
            "the tool takes no arguments and the call supplies some"
        )
        assert outputs["c2"] == "pong"
        assert policy.seen == [("ping", {})] and ran == ["ping"]

    def test_arguments_that_do_not_validate_say_so(self, example):
        outputs, executed = example.run_calls([("run_sql", {"sql": ["MARKER"]}, "c1")])
        assert executed == []
        assert outputs["c1"] == (
            "Recusal refused `run_sql`: the arguments the tool would execute cannot be "
            "judged: they do not validate against the tool's parameters"
        )

    @pytest.mark.parametrize(
        "arguments, detail",
        [
            ('{"amount": NaN}', "arguments hold a non-finite number, which is not JSON data"),
            ('{"amount": -Infinity}', "arguments hold a non-finite number, which is not JSON data"),
            ('{"amount": 1e999}', "arguments hold a non-finite number, which is not JSON data"),
            ('{"amount": [1, NaN]}', "arguments hold a non-finite number, which is not JSON data"),
            (
                '{"amount": "nan"}',
                "the arguments the tool would execute cannot be judged (non-finite numbers "
                "are not JSON data)",
            ),
        ],
        ids=["nan", "minus-infinity", "overflow", "nested", "string-read-as-nan"],
    )
    def test_a_non_finite_number_is_refused_before_the_policy_runs(
        self, example, arguments, detail
    ):
        from recusal import Finding

        ran: list = []

        @example.function_tool(strict_mode=False)
        def pay(amount: float) -> str:
            """Record a payment; contacts nothing."""
            ran.append(amount)
            return "paid"

        def ceiling(name, args):  # NaN > 1000 is False: this policy alone lets NaN through
            if args.get("amount", 0) > 1000:
                return [Finding.fail("limit", severity="CRITICAL", message="over limit")]
            return []

        policy = self.Recording(ceiling)
        outputs, _ = example.run_calls([("pay", arguments, "c1")], policy=policy, tools=[pay])
        assert outputs["c1"] == f"Recusal refused `pay`: {detail}"
        assert policy.seen == [] and ran == []

    @pytest.mark.parametrize("depth", [65, 1_500, 100_000])
    def test_arguments_nested_too_deeply_are_refused_not_raised(self, example, depth):
        """Three depths: one past the gate's own limit, one the JSON parser reads and the
        gate's limit refuses, and one the JSON parser itself gives up on. In a real run the
        SDK raises on the larger two before the guardrail is asked (the README says so);
        ``TestSecondReview`` runs the limit itself through a real run."""
        deep = "[" * depth + "]" * depth
        policy = self.Recording(lambda name, args: [])
        reason = _adapter()._function_tool_refusal(
            "run_sql", '{"sql": ' + deep + "}", policy, example.run_sql
        )
        assert reason == "Recusal refused `run_sql`: arguments are nested too deeply"
        assert policy.seen == []

    # --- the pin, in a real run ----------------------------------------------------------------

    @pytest.mark.parametrize("gated", [True, False], ids=["gated", "ungated-control"])
    def test_a_real_run_executes_the_invocation_the_gate_pinned(self, example, monkeypatch, gated):
        """The SDK prepares an invocation by parsing the arguments once. Gated, the one
        parse is the gate's, the pin is on the tool context when the tool starts, and the
        SDK runs it without parsing again. Ungated, there is no pin and the one parse is
        the SDK's own."""
        import agents.tool as sdk_tool
        from agents import RunHooks

        parses: list = []
        original = sdk_tool._parse_function_tool_json_input

        def counting(**kwargs):
            parses.append(kwargs["tool_name"])
            return original(**kwargs)

        monkeypatch.setattr(sdk_tool, "_parse_function_tool_json_input", counting)
        pins: list = []

        class Hooks(RunHooks):
            async def on_tool_start(self, context, agent, tool):
                pin = getattr(context, "_function_tool_arguments", None)
                pins.append(None if pin is None else list(pin.args))

        policy = self.Recording(lambda name, args: [])
        agent = example.Agent(
            name="a",
            model=example.scripted_model([("charge", {}, "c1")]),
            tools=[example.charge_tool()],
        )
        if gated:
            agent = example.gated_agent(agent, policy)
        example.EXECUTED.clear()
        asyncio.run(
            example.Runner.run(
                agent, "go", run_config=example.RunConfig(tracing_disabled=True), hooks=Hooks()
            )
        )
        assert example.EXECUTED == [("charge", {"amount": 100})]
        assert parses == ["charge"], "prepared once; the SDK did not prepare a second time"
        assert pins == ([[100]] if gated else [None])
        assert policy.seen == ([("charge", {"amount": 100})] if gated else [])


class _DescriptorDecorator:
    """A class-based decorator that is also a descriptor, the shape of ``cachetools.cached``:
    ``inspect`` reads such an object as a built-in and finds no signature for it."""

    def __init__(self, fn):
        import functools

        functools.update_wrapper(self, fn)
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __get__(self, instance, owner=None):
        return self


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestSecondReview:
    """The second review of 2026-11-21. The walk over the callables a call reaches was
    pinned only for a single wrapper; a ``functools.singledispatch`` function ran an
    implementation whose default no policy saw; the plain and the audited gate refused
    deeply nested arguments at different depths and for different stated causes; and some
    tools were accepted when the agent was built and then refused on every call."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording

    # --- the walk goes past the outermost callable --------------------------------------------

    @staticmethod
    def stack(example, ran, order):
        """``run_sql(sql, force=False)`` under a stack of decorators, outermost first:
        ``f`` forwards everything, ``t`` takes the function's ``force`` and drops it."""
        import functools

        def forwards(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def takes_force(fn):
            @functools.wraps(fn)
            def wrapper(*args, force=False, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def run_sql(sql: str, *, force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            ran.append({"sql": sql, "force": force})
            return "recorded"

        func = run_sql
        for letter in reversed(order):
            func = (forwards if letter == "f" else takes_force)(func)
        return example.function_tool(func, name_override="run_sql", strict_mode=False)

    @pytest.mark.parametrize("order", ["ft", "fft", "ffft", "ftf", "fftf"])
    def test_a_wrapper_under_other_wrappers_is_found_at_any_level(self, example, order):
        ran: list = []
        tool = self.stack(example, ran, order)
        expected = "cannot gate 'run_sql': a function the call reaches takes `force`"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_negative_control_ungated_the_buried_wrapper_drops_the_argument(self, example):
        ran: list = []
        tool = self.stack(example, ran, "ffft")
        example.run_calls(
            [("run_sql", {"sql": SELECT, "force": True}, "c1")], gated=False, tools=[tool]
        )
        assert ran == [{"sql": SELECT, "force": False}]  # proposed force=True

    def test_a_stack_of_wrappers_that_only_forward_is_gated_and_judged(self, example):
        ran: list = []
        tool = self.stack(example, ran, "ffff")
        policy = self.Recording(lambda name, args: [])
        example.run_calls(
            [("run_sql", {"sql": SELECT, "force": True}, "c1")], policy=policy, tools=[tool]
        )
        assert policy.seen == [("run_sql", {"sql": SELECT, "force": True})]
        assert ran == [{"sql": SELECT, "force": True}]

    # --- a function that picks its code by the type of an argument ---------------------------

    @staticmethod
    def dispatching(ran):
        import functools

        @functools.singledispatch
        def lookup(key, force: bool = False) -> str:
            """Record a lookup; contacts nothing."""
            ran.append({"key": key, "force": force})
            return "recorded"

        @lookup.register
        def _(key: str, force: bool = False, include_deleted: bool = True) -> str:
            ran.append({"key": key, "force": force, "include_deleted": include_deleted})
            return "recorded"

        return lookup

    @pytest.mark.parametrize(
        "wrapped", ["no", "wrapper", "partial"], ids=["bare", "under-a-wrapper", "under-a-partial"]
    )
    def test_a_singledispatch_function_is_refused_when_the_agent_is_built(self, example, wrapped):
        import functools

        func = self.dispatching([])
        if wrapped != "no":
            inner = func if wrapped == "wrapper" else functools.partial(func, force=True)

            @functools.wraps(inner)
            def func(*args, **kwargs):
                return inner(*args, **kwargs)

        tool = example.function_tool(func, name_override="lookup", strict_mode=False)
        expected = "cannot gate 'lookup': it runs a `functools.singledispatch` function"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)
        with pytest.raises(TypeError, match=expected):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)

    def test_a_singledispatchmethod_is_refused_too(self, example):
        import functools

        class Service:
            @functools.singledispatchmethod
            def lookup(self, key, force: bool = False) -> str:
                """Record a lookup; contacts nothing."""
                return "recorded"

        tool = example.function_tool(Service().lookup, name_override="lookup", strict_mode=False)
        with pytest.raises(TypeError, match="it runs a `functools.singledispatch` function"):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_negative_control_ungated_the_registered_implementation_runs_its_own_default(
        self, example
    ):
        ran: list = []
        tool = example.function_tool(
            self.dispatching(ran), name_override="lookup", strict_mode=False
        )
        assert "include_deleted" not in tool.params_json_schema["properties"]
        example.run_calls([("lookup", {"key": "k"}, "c1")], gated=False, tools=[tool])
        assert ran == [{"key": "k", "force": False, "include_deleted": True}]

    # --- a decorator whose signature cannot be read --------------------------------------------

    @pytest.mark.parametrize("wrapped", [False, True], ids=["outermost", "under-a-wrapper"])
    def test_a_descriptor_class_decorator_is_refused_with_what_to_do(self, example, wrapped):
        import functools

        def run_sql(sql: str) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        func = _DescriptorDecorator(run_sql)
        if wrapped:
            inner = func

            @functools.wraps(inner)
            def func(*args, **kwargs):
                return inner(*args, **kwargs)

        tool = example.function_tool(func, name_override="run_sql")
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == (
            "Recusal cannot gate 'run_sql': a decorator on the function cannot be inspected "
            "(a class-based decorator that is also a descriptor, such as `cachetools.cached`), "
            "so the functions under it cannot be read. Call the decorated function from a "
            "plain function and register that."
        )

    def test_a_decorator_that_calls_the_decorated_function_from_a_plain_one_is_gated(self, example):
        ran: list = []

        @_DescriptorDecorator
        def cached(sql: str) -> str:
            ran.append(sql)
            return "recorded"

        @example.function_tool
        def run_sql(sql: str) -> str:
            """Record SQL text without connecting to a database."""
            return cached(sql)

        policy = self.Recording(example.sql_scope_policy)
        example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1"), ("run_sql", {"sql": DELETE}, "c2")],
            policy=policy,
            tools=[run_sql],
        )
        assert ran == [SELECT]

    @staticmethod
    def forwarding_stack(depth, decorate=None):
        """``run_sql(sql)`` under ``depth`` wrappers that only forward, or under ``depth``
        layers of ``decorate``."""
        import functools

        def run_sql(sql: str) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        def forwards(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        func = run_sql
        for _ in range(depth):
            func = (decorate or forwards)(func)
        return func

    def test_too_many_wrappers_are_refused_for_that_reason(self, example):
        """Too many wrappers is a ``ValueError`` of the walk's own, with no decorator object
        on the chain: neither the reason about a class-based decorator nor "cannot be
        inspected (ValueError)" would say what is wrong with it."""
        import functools

        expected = (
            "Recusal cannot gate 'run_sql': the function it runs is under more wrappers than "
            "the gate reads (a stack of more than 31 wrappers, or more than eight "
            "`functools.lru_cache` layers one on another), so the functions under them cannot "
            "be read. Call the wrapped function from a plain function and register that."
        )
        cache = functools.lru_cache(maxsize=None)
        refused = [
            self.forwarding_stack(32),
            cache(self.forwarding_stack(40)),
            self.forwarding_stack(9, cache),
        ]
        for candidate in refused:
            tool = example.function_tool(candidate, name_override="run_sql")
            with pytest.raises(TypeError) as refusal:
                example.gate_tools([tool], example.sql_scope_policy)
            assert str(refusal.value) == expected
        for candidate in (self.forwarding_stack(31), self.forwarding_stack(8, cache)):
            tool = example.function_tool(candidate, name_override="run_sql")
            assert example.gate_tools([tool], example.sql_scope_policy)

    def test_another_unreadable_callable_keeps_the_plain_reason(self, example, monkeypatch):
        """A ``ValueError`` that is not the walk's own limit is still named for what it is."""

        def unreadable(func):
            raise ValueError("no signature found")

        monkeypatch.setattr(_adapter(), "_reached_parameters", unreadable)
        with pytest.raises(TypeError, match="cannot be inspected \\(ValueError\\)"):
            example.gate_tools([example.run_sql], example.sql_scope_policy)

    # --- a mapping keyed by anything but str ----------------------------------------------------

    @pytest.mark.parametrize(
        "annotation",
        [
            typing.Dict[int, str],
            typing.Dict[bool, str],
            typing.Optional[typing.Dict[int, str]],
            typing.List[typing.Dict[int, str]],
            typing.Dict[str, typing.Dict[float, str]],
            typing.Union[str, typing.Dict[int, str]],
        ],
        ids=["int", "bool", "optional", "in-a-list", "nested", "in-a-union"],
    )
    def test_a_mapping_keyed_by_another_type_than_str_is_refused_at_setup(
        self, example, annotation
    ):
        tool = TestJudgedIsWhatRuns.typed_tool(example, annotation)
        expected = "cannot gate 'typed': parameter `value` is a mapping whose key type is not `str`"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)
        with pytest.raises(TypeError, match=expected):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)

    def test_negative_control_ungated_the_sdk_converts_the_keys(self, example):
        tool = TestJudgedIsWhatRuns.typed_tool(example, typing.Dict[int, str])
        _, executed = example.run_calls(
            [("typed", {"value": {"1": "a"}}, "c1")], gated=False, tools=[tool]
        )
        assert executed == [("typed", {"value": {1: "a"}})]

    @pytest.mark.parametrize(
        "annotation",
        [typing.Dict[str, int], dict, typing.Dict[typing.Any, int], typing.Mapping[str, int]],
        ids=["str", "bare", "any", "mapping"],
    )
    def test_a_mapping_keyed_by_str_is_still_gated_and_judged(self, example, annotation):
        tool = TestJudgedIsWhatRuns.typed_tool(example, annotation)
        policy = self.Recording(lambda name, args: [])
        _, executed = example.run_calls(
            [("typed", {"value": {"1": 2}}, "c1")], policy=policy, tools=[tool]
        )
        assert policy.seen == [("typed", {"value": {"1": 2}})]
        assert executed == [("typed", {"value": {"1": 2}})]

    # --- defaults: which are prepared, and the true reason for the ones that are not --------

    def test_a_default_that_is_not_json_data_is_refused_for_that_reason(self, example):
        import math

        def tagged(sql: str, tags: typing.List[str] = ()) -> str:  # type: ignore[assignment]
            """Record SQL text without connecting to a database."""
            return "recorded"

        def timed(sql: str, timeout: float = math.inf) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        for func, parameter in ((tagged, "tags"), (timed, "timeout")):
            tool = example.function_tool(func, strict_mode=False)
            with pytest.raises(TypeError) as refusal:
                example.gate_tools([tool], example.sql_scope_policy)
            assert str(refusal.value).startswith(
                f"Recusal cannot gate {func.__name__!r}: parameter `{parameter}` has a default "
                "that is not plain JSON data"
            )
            assert "has a type or a validation" not in str(refusal.value)

    def test_a_default_factory_of_the_tools_own_is_refused_for_that_reason(self, example):
        from pydantic import Field

        def tagged(
            sql: str,
            tags: typing.Annotated[typing.List[str], Field(default_factory=lambda: ["all"])],
        ) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        tool = example.function_tool(tagged, strict_mode=False)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value).startswith(
            "Recusal cannot gate 'tagged': parameter `tags` has a default factory other than "
            "`list` or `dict`"
        )

    @pytest.mark.parametrize("factory, empty", [(list, []), (dict, {})], ids=["list", "dict"])
    def test_a_list_or_dict_default_factory_is_gated_and_judged(self, example, factory, empty):
        from pydantic import Field

        ran: list = []
        annotation = typing.List[str] if factory is list else typing.Dict[str, str]

        def tagged(sql, tags) -> str:
            """Record SQL text without connecting to a database."""
            ran.append({"sql": sql, "tags": tags})
            return "recorded"

        tagged.__annotations__ = {
            "sql": str,
            "tags": typing.Annotated[annotation, Field(default_factory=factory)],
            "return": str,
        }
        tool = example.function_tool(tagged, strict_mode=False)
        policy = self.Recording(lambda name, args: [])
        example.run_calls([("tagged", {"sql": SELECT}, "c1")], policy=policy, tools=[tool])
        assert policy.seen == [("tagged", {"sql": SELECT, "tags": empty})]
        assert ran == [{"sql": SELECT, "tags": empty}]

    def test_a_refused_type_beside_a_refused_default_is_named_for_the_type(self, example):
        import datetime

        def both(when: datetime.date, tags: typing.List[str] = ()) -> str:  # type: ignore[assignment]
            """Record a search; contacts nothing."""
            return "recorded"

        tool = example.function_tool(both, strict_mode=False)
        with pytest.raises(TypeError, match="parameter `when` has a type or a validation"):
            example.gate_tools([tool], example.sql_scope_policy)

    # --- the parent's policy judges a call to an agent tool ----------------------------------

    def test_the_parent_policy_can_refuse_a_call_to_an_agent_tool(self, example):
        from recusal import Finding

        def no_deletes_to_the_sub_agent(name, args):
            if name == "ask_sub" and "DELETE" in str(args.get("input", "")).upper():
                return [Finding.fail("no_deletes", severity="CRITICAL", message="not to the sub")]
            return []

        def run(text):
            sub_model = example.scripted_model([("run_sql", {"sql": SELECT}, "s1")])
            sub = example.gated_agent(
                example.Agent(name="sub", model=sub_model, tools=[example.run_sql]),
                lambda name, args: [],
            )
            parent_model = example.scripted_model([("ask_sub", {"input": text}, "p1")])
            tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask the sub-agent.")
            parent = example.gated_agent(
                example.Agent(name="parent", model=parent_model, tools=[tool]),
                no_deletes_to_the_sub_agent,
            )
            example.EXECUTED.clear()
            asyncio.run(example.run_agent(parent))
            return example.tool_outputs(parent_model)["p1"], list(example.EXECUTED)

        output, executed = run("please DELETE everything")
        assert output == "Recusal refused `ask_sub` [FAIL]: not to the sub"
        assert executed == [], "the sub-agent was never started"
        output, executed = run("tidy the log")
        assert executed == [("run_sql", {"sql": SELECT})], "an allowed input reaches the sub-agent"

    # --- one fixed depth limit, in a real run --------------------------------------------------

    @staticmethod
    def nested(depth, shape):
        """JSON text whose deepest value sits ``depth`` levels below the value it is."""
        if shape == "list":
            return "[" * depth + "]" * depth
        return '{"a":' * (depth - 1) + "1" + "}" * (depth - 1)

    @pytest.mark.parametrize("shape", ["list", "dict"])
    def test_arguments_at_the_depth_limit_run_and_one_level_deeper_are_refused(
        self, example, shape
    ):
        limit = _adapter()._MAX_DEPTH
        assert limit == 64
        tool = TestJudgedIsWhatRuns.typed_tool(example, typing.Any)
        policy = self.Recording(lambda name, args: [])
        calls = [
            ("typed", '{"value": ' + self.nested(limit, shape) + "}", "c1"),
            ("typed", '{"value": ' + self.nested(limit + 1, shape) + "}", "c2"),
        ]
        outputs, executed = example.run_calls(calls, policy=policy, tools=[tool])
        assert outputs["c1"] == "recorded"
        assert outputs["c2"] == "Recusal refused `typed`: arguments are nested too deeply"
        assert len(executed) == 1 and len(policy.seen) == 1

    def test_the_depth_limit_does_not_move_with_the_recursion_limit(self, example):
        tool = TestJudgedIsWhatRuns.typed_tool(example, typing.Any)
        policy = self.Recording(lambda name, args: [])
        previous = sys.getrecursionlimit()
        sys.setrecursionlimit(previous * 4)
        try:
            reason = _adapter()._function_tool_refusal(
                "typed", '{"value": ' + self.nested(65, "list") + "}", policy, tool
            )
        finally:
            sys.setrecursionlimit(previous)
        assert reason == "Recusal refused `typed`: arguments are nested too deeply"
        assert policy.seen == []

    # --- model settings that put tools into the request ---------------------------------------

    @pytest.mark.parametrize("field", ["extra_body", "extra_args"])
    @pytest.mark.parametrize(
        "key", ["tools", "prompt", "plugins", "web_search_options", "mcp_servers"]
    )
    def test_model_settings_that_carry_tools_are_refused(self, example, field, key):
        from agents import ModelSettings

        settings = ModelSettings(**{field: {key: [{"type": "web_search"}]}})
        agent = example.Agent(name="a", tools=[example.run_sql], model_settings=settings)
        expected = f"cannot gate agent 'a': its model_settings.{field} sets `{key}`"
        with pytest.raises(ValueError, match=expected):
            example.gated_agent(agent, example.sql_scope_policy)
        gated = example.gated_agent(
            example.Agent(name="a", tools=[example.run_sql]), example.sql_scope_policy
        )
        assert example.is_gated(gated)
        gated.model_settings = settings
        assert not example.is_gated(gated)

    def test_model_settings_that_carry_no_tools_are_gated(self, example):
        from agents import ModelSettings

        settings = ModelSettings(
            temperature=0.1, extra_body={"service_tier": "flex"}, extra_args={"user": "u1"}
        )
        agent = example.Agent(name="a", tools=[example.run_sql], model_settings=settings)
        assert example.is_gated(example.gated_agent(agent, example.sql_scope_policy))


class Router:
    """A callable service object that happens to have a ``dispatch`` method and a
    ``registry`` attribute, as a ``functools.singledispatch`` function does."""

    def __init__(self, ran):
        self.ran = ran
        self.registry = {"default": "db1"}

    def dispatch(self, event):
        return None

    def __call__(self, sql: str, limit: int = 10) -> str:
        """Record SQL text without connecting to a database."""
        self.ran.append({"sql": sql, "limit": limit})
        return "recorded"


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestThirdReview:
    """The third review of 2026-11-21. With ``agent.as_tool(parameters=Model)`` the parent's
    policy judged the model's raw JSON while the sub-agent was handed the validated,
    converted and defaulted input; three checks refused what they had no reason to (a
    service object taken for a ``singledispatch`` function, a mapping keyed by a
    ``Literal`` of strings, a stack of wrappers named as "cannot be inspected"); and three
    mutations of the gate passed the suite."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording

    # --- an agent tool with a parameters model ------------------------------------------------

    @staticmethod
    def ask_model():
        from pydantic import BaseModel

        class Ask(BaseModel):
            task: str
            confirm: bool = False
            rows: int = 1

        return Ask

    @staticmethod
    def no_confirmed_or_large(name, args):
        """Refuses a confirmed request and one for more than five rows."""
        from recusal import Finding

        if args.get("confirm") is True:
            return [Finding.fail("no_confirm", severity="CRITICAL", message="not confirmed")]
        if type(args.get("rows", 1)) is int and args.get("rows", 1) > 5:
            return [Finding.fail("too_many", severity="CRITICAL", message="at most five rows")]
        return []

    def ask(self, example, arguments, *, parameters, policy=None, gated=True):
        """A parent that calls ``ask_sub`` with ``arguments``: ``(what the parent's policy
        was handed, the call's output, what the sub-agent was given or None)``."""
        from agents.testing import ScriptedModel, assistant_message

        sub_model = ScriptedModel([[assistant_message("sub done")]])
        sub = example.gated_agent(
            example.Agent(name="sub", model=sub_model, tools=[example.run_sql]),
            lambda name, args: [],
        )
        keywords = {} if parameters is None else {"parameters": parameters}
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask the sub-agent.", **keywords)
        parent_model = example.scripted_model([("ask_sub", arguments, "p1")])
        parent = example.Agent(name="parent", model=parent_model, tools=[tool])
        recording = self.Recording(policy or self.no_confirmed_or_large)
        if gated:
            parent = example.gated_agent(parent, recording)
        asyncio.run(example.run_agent(parent))
        given = None
        if sub_model.first_call is not None:
            given = [item["content"] for item in sub_model.first_call.input]
        return recording.seen, example.tool_outputs(parent_model)["p1"], given

    @pytest.mark.parametrize(
        "arguments, judged",
        [
            (
                {"task": "list", "confirm": False, "rows": 2},
                {"task": "list", "confirm": False, "rows": 2},
            ),
            (
                {"task": "list", "confirm": "false", "rows": "2"},
                {"task": "list", "confirm": False, "rows": 2},
            ),
            ({"task": "list"}, {"task": "list", "confirm": False, "rows": 1}),
            (
                {"task": "list", "rows": 3, "force": True},
                {"task": "list", "confirm": False, "rows": 3},
            ),
        ],
        ids=["typed", "strings-the-sdk-converts", "defaults-left-out", "a-key-the-model-drops"],
    )
    def test_the_parent_policy_is_handed_what_the_sub_agent_is_given(
        self, example, arguments, judged
    ):
        import json

        seen, output, given = self.ask(example, arguments, parameters=self.ask_model())
        assert seen == [("ask_sub", judged)]
        assert output == "sub done"
        assert [json.loads(text) for text in given] == [judged]

    @pytest.mark.parametrize(
        "arguments",
        [{"task": "list", "confirm": "true"}, {"task": "list", "rows": "9"}],
        ids=["confirm-as-a-string", "rows-as-a-string"],
    )
    def test_the_parent_policy_refuses_the_converted_value_it_now_sees(self, example, arguments):
        seen, output, given = self.ask(example, arguments, parameters=self.ask_model())
        assert output.startswith("Recusal refused `ask_sub` [FAIL]: ")
        assert given is None, "the sub-agent was never started"

    def test_negative_control_ungated_the_sub_agent_is_given_the_converted_values(self, example):
        import json

        arguments = {"task": "list", "confirm": "true", "rows": "9", "force": True}
        _, _, given = self.ask(example, arguments, parameters=self.ask_model(), gated=False)
        assert [json.loads(text) for text in given] == [
            {"task": "list", "confirm": True, "rows": 9}
        ]

    @pytest.mark.parametrize(
        "arguments",
        [{"task": "list", "rows": "many"}, {"confirm": True}, {}],
        ids=["a-value-of-another-type", "a-required-field-left-out", "nothing"],
    )
    def test_arguments_the_parameters_model_does_not_accept_are_refused_before_the_policy(
        self, example, arguments
    ):
        seen, output, given = self.ask(example, arguments, parameters=self.ask_model())
        assert output == (
            "Recusal refused `ask_sub`: the input the sub-agent would be given cannot be "
            "judged (the arguments do not validate against its parameters)"
        )
        assert seen == [] and given is None

    def test_the_default_as_tool_is_judged_by_the_arguments_as_proposed(self, example):
        arguments = {"input": "tidy the log", "note": "kept"}
        seen, output, given = self.ask(example, arguments, parameters=None)
        assert seen == [("ask_sub", arguments)]
        assert given == ["tidy the log"]

    @staticmethod
    def unreadable_parameters():
        """``parameters`` types whose validated input the gate cannot work out without
        running code of the type's own, by the name of what makes them so."""
        import dataclasses
        import datetime

        from pydantic import BaseModel, Field, computed_field, field_serializer, field_validator

        class Validated(BaseModel):
            task: str

            @field_validator("task")
            @classmethod
            def upper(cls, value):
                return value.upper()

        class Serialized(BaseModel):
            task: str

            @field_serializer("task")
            def upper(self, value):
                return value.upper()

        class Computed(BaseModel):
            task: str

            @computed_field  # type: ignore[prop-decorator]
            @property
            def shout(self) -> str:
                return self.task.upper()

        class Inner(BaseModel):
            task: str

        class Nested(BaseModel):
            inner: Inner

        class Dated(BaseModel):
            when: datetime.date

        class Built(BaseModel):
            tags: typing.List[str] = Field(default_factory=lambda: ["all"])

        @dataclasses.dataclass
        class Plain:
            task: str

        return {
            "a-validator": Validated,
            "a-serializer": Serialized,
            "a-computed-field": Computed,
            "a-nested-model": Nested,
            "a-date": Dated,
            "a-default-factory": Built,
            "a-dataclass": Plain,
        }

    @pytest.mark.parametrize(
        "case, named",
        [
            (
                "a-validator",
                "model has a field outside the data-only set: `task` (a validator, pydantic "
                "core type `function-after`). The gate works out",
            ),
            ("a-serializer", "model writes `task` out through code of its own"),
            ("a-computed-field", "model writes `shout` out through code of its own"),
            (
                "a-nested-model",
                "model has a field outside the data-only set: `inner` (pydantic core type "
                "`model`). ",
            ),
            (
                "a-date",
                "model has a field outside the data-only set: `when` (pydantic core type `date`). ",
            ),
            (
                "a-default-factory",
                "model has a field outside the data-only set: `tags` (a default that is not "
                "JSON data, or a default factory other than `list` or `dict`). ",
            ),
            (
                "a-dataclass",
                "type is not a Pydantic model the gate reads: its pydantic core type is "
                "`dataclass` ",
            ),
        ],
    )
    def test_a_parameters_type_that_runs_its_own_code_is_refused_at_setup(
        self, example, case, named
    ):
        parameters = self.unreadable_parameters()[case]
        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=parameters)
        expected = f"Recusal cannot gate agent tool 'ask_sub': its `parameters` {named}"
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value).startswith(expected)
        with pytest.raises(TypeError) as refusal:
            example.gated_agent(example.Agent(name="p", tools=[tool]), example.sql_scope_policy)
        assert str(refusal.value).startswith(expected)

    def test_a_refused_parameters_type_is_refused_on_a_call_too(self, example):
        """Should the tool reach a call another way than through ``gate_tools``."""
        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(
            tool_name="ask_sub",
            tool_description="Ask.",
            parameters=self.unreadable_parameters()["a-validator"],
        )
        policy = self.Recording(lambda name, args: [])
        reason = _adapter()._function_tool_refusal("ask_sub", '{"task": "x"}', policy, tool)
        assert reason.startswith(
            "Recusal refused `ask_sub`: its `parameters` model has a field outside the "
            "data-only set: `task` (a validator, pydantic core type `function-after`)"
        )
        assert policy.seen == []

    def test_negative_control_a_validator_gives_the_sub_agent_what_no_policy_saw(self, example):
        import json

        parameters = self.unreadable_parameters()["a-validator"]
        _, _, given = self.ask(example, {"task": "list"}, parameters=parameters, gated=False)
        assert [json.loads(text) for text in given] == [{"task": "LIST"}]

    def test_an_agent_tool_without_the_sdks_adapter_is_refused(self, example):
        """An agent tool the gate cannot read the SDK's adapter from is refused, never
        judged by its raw arguments."""
        import dataclasses

        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.")

        async def invoke(ctx, arguments):
            return "ran"

        rebuilt = dataclasses.replace(tool, on_invoke_tool=invoke)
        rebuilt._is_agent_tool = True
        rebuilt._agent_instance = sub
        with pytest.raises(TypeError, match="does not expose the adapter the SDK validates"):
            example.gate_tools([rebuilt], example.sql_scope_policy)

    def test_a_crash_inside_decide_fails_closed_for_an_agent_tool(self, example, monkeypatch):
        def crashing_decide(*args, **kwargs):
            raise RuntimeError("decide crashed")

        monkeypatch.setattr(_adapter(), "decide", crashing_decide)
        for parameters in (None, self.ask_model()):
            arguments = {"input": "go"} if parameters is None else {"task": "list"}
            _, output, given = self.ask(example, arguments, parameters=parameters)
            assert (
                output
                == "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
            )
            assert given is None, "the sub-agent was never started"

    # --- a sub-agent is gated only by Recusal's own guardrail ---------------------------------

    def test_a_sub_agent_whose_tool_carries_only_another_guardrail_is_not_gated(self, example):
        import dataclasses

        @_adapter().tool_input_guardrail
        def logging_only(data):
            return _adapter().ToolGuardrailFunctionOutput.allow()

        foreign = dataclasses.replace(example.run_sql, tool_input_guardrails=[logging_only])
        sub_model = example.scripted_model([("run_sql", {"sql": DELETE}, "s1")])
        sub = example.Agent(name="sub", model=sub_model, tools=[foreign])
        assert not example.is_gated(sub)
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask the sub-agent.")
        with pytest.raises(TypeError, match="its agent's own tool calls are not gated"):
            example.gated_agent(example.Agent(name="p", tools=[tool]), example.sql_scope_policy)
        (gated,) = example.gate_tools([foreign], example.sql_scope_policy)
        assert example.is_gated(example.Agent(name="sub", tools=[gated]))

    # --- empty argument text -------------------------------------------------------------------

    def test_empty_argument_text_is_judged_as_no_arguments(self, example):
        """The SDK reads empty argument text as no arguments, and so does the gate."""
        ran: list = []

        @example.function_tool
        def tidy(limit: int = 10) -> str:
            """Record a tidy; contacts nothing."""
            ran.append(limit)
            return "recorded"

        policy = self.Recording(lambda name, args: [])
        outputs, _ = example.run_calls([("tidy", "", "c1")], policy=policy, tools=[tidy])
        assert outputs == {"c1": "recorded"}
        assert policy.seen == [("tidy", {"limit": 10})]
        assert ran == [10]

    # --- what is and is not a singledispatch function -----------------------------------------

    def test_a_service_object_with_a_dispatch_method_and_a_registry_is_gated(self, example):
        ran: list = []
        tool = example.function_tool(Router(ran), name_override="run_sql")
        policy = self.Recording(example.sql_scope_policy)
        outputs, _ = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1"), ("run_sql", {"sql": DELETE}, "c2")],
            policy=policy,
            tools=[tool],
        )
        assert policy.seen[0] == ("run_sql", {"sql": SELECT, "limit": 10})
        assert ran == [{"sql": SELECT, "limit": 10}]
        assert outputs["c2"].startswith("Recusal refused `run_sql` [FAIL]")

    def test_reading_for_a_singledispatch_function_runs_no_code_of_the_object(self, example):
        """``registry`` and ``dispatch`` are read without calling a property or a
        ``__getattr__`` of the tool's own."""
        import types

        touched: list = []

        class Lazy:
            @property
            def registry(self):
                touched.append("registry")
                return {}

            @property
            def dispatch(self):
                touched.append("dispatch")
                return lambda event: None

            def __call__(self, sql: str) -> str:
                """Record SQL text without connecting to a database."""
                return "recorded"

        tool = example.function_tool(Lazy(), name_override="lazy")
        assert example.gate_tools([tool], example.sql_scope_policy)
        assert touched == []

        class Registered(Lazy):
            # what ``functools.singledispatch`` sets, with a ``dispatch`` that is not one
            registry = types.MappingProxyType({})

        tool = example.function_tool(Registered(), name_override="lazy")
        assert example.gate_tools([tool], example.sql_scope_policy)
        assert touched == []

    def test_a_singledispatch_function_with_nothing_registered_is_refused(self, example):
        """It was gated until the fifth review: an implementation can be registered
        after a call was judged and before it runs (``TestFifthReview``)."""
        import functools

        @functools.singledispatch
        def lookup(key: str, force: bool = False) -> str:
            """Record a lookup; contacts nothing."""
            return "recorded"

        assert len(lookup.registry) == 1
        tool = example.function_tool(lookup, strict_mode=False)
        with pytest.raises(TypeError, match="it runs a `functools.singledispatch` function"):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_a_registry_whose_base_is_another_function_is_refused(self, example):
        """``registry[object]`` that is not the function ``__wrapped__`` names counts as
        an implementation registered."""
        import functools

        @functools.singledispatch
        def lookup(key: str, force: bool = False) -> str:
            """Record a lookup; contacts nothing."""
            return "recorded"

        @lookup.register(object)
        def _(key, force: bool = False, include_deleted: bool = True) -> str:
            return "recorded"

        assert len(lookup.registry) == 1
        tool = example.function_tool(lookup, strict_mode=False)
        with pytest.raises(TypeError, match="it runs a `functools.singledispatch` function"):
            example.gate_tools([tool], example.sql_scope_policy)

    # --- a mapping keyed by a Literal of strings -------------------------------------------------

    def test_a_mapping_keyed_by_a_literal_of_strings_is_gated_and_judged(self, example):
        annotation = typing.Dict[typing.Literal["a", "b"], int]
        tool = TestJudgedIsWhatRuns.typed_tool(example, annotation)
        policy = self.Recording(lambda name, args: [])
        calls = [
            ("typed", {"value": {"a": 1}}, "c1"),
            ("typed", {"value": {"c": 1}}, "c2"),
        ]
        outputs, executed = example.run_calls(calls, policy=policy, tools=[tool])
        assert policy.seen == [("typed", {"value": {"a": 1}})]
        assert executed == [("typed", {"value": {"a": 1}})]
        assert outputs["c2"] == (
            "Recusal refused `typed`: the arguments the tool would execute cannot be judged: "
            "they do not validate against the tool's parameters"
        )

    @pytest.mark.parametrize(
        "annotation",
        [
            typing.Dict[typing.Literal[1, 2], int],
            typing.Dict[typing.Literal["a", 1], int],
            typing.Dict[typing.Literal[True], int],
        ],
        ids=["integers", "a-string-and-an-integer", "a-boolean"],
    )
    def test_a_mapping_keyed_by_any_other_literal_is_still_refused(self, example, annotation):
        tool = TestJudgedIsWhatRuns.typed_tool(example, annotation)
        expected = (
            "cannot gate 'typed': parameter `value` is a mapping whose key type is not `str` "
            "\\(or a `Literal` of strings\\)"
        )
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)

    # --- the refusal names the abstract collections ---------------------------------------------

    @pytest.mark.parametrize(
        "annotation",
        [typing.Sequence[str], typing.Iterable[str]],
        ids=["sequence", "iterable"],
    )
    def test_a_sequence_or_an_iterable_is_refused_and_named(self, example, annotation):
        tool = TestJudgedIsWhatRuns.typed_tool(example, annotation)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert "parameter `value` has a type or a validation" in str(refusal.value)
        assert "a Sequence or an Iterable" in str(refusal.value)

    # --- an error of the interpreter inside the guardrail ---------------------------------------

    def test_a_recursion_error_inside_the_guardrail_stops_the_run_and_nothing_runs(
        self, example, monkeypatch
    ):
        """Under a recursion limit too small for the walk over the arguments the guardrail
        raises. The SDK then stops the run: the tool does not run."""

        def out_of_stack(value, depth=0):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(_adapter(), "_too_deep", out_of_stack)
        policy = self.Recording(lambda name, args: [])
        with pytest.raises(Exception, match="maximum recursion depth exceeded"):
            example.run_calls([("run_sql", {"sql": SELECT}, "c1")], policy=policy)
        assert example.EXECUTED == [] and policy.seen == []


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestFourthReview:
    """The fourth review of 2026-11-22. A ``functools.singledispatch`` function used as a
    method was no longer recognised (the round-three check did not see through a bound
    method), so a registered implementation ran with a default no policy saw; one with
    nothing registered was refused under a forwarding wrapper, which carries a copy of the
    dispatcher's attributes; and the refusal of an agent tool's ``parameters`` type named
    no field and gave a cause that was not true of most of the types it refused."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording

    # --- the dispatcher is found exactly, wherever it stands on the chain ----------------------

    @staticmethod
    def service(ran, register):
        """The reviewer's shape: ``singledispatch`` on a method, an implementation
        registered for the class itself, which adds ``dry_run``."""
        import functools

        class Service:
            @functools.singledispatch
            def handle(self, sql: str = "", force: bool = False) -> str:
                """Record SQL text without connecting to a database."""
                ran.append(("base", sql, force))
                return "recorded"

            on_class = classmethod(
                functools.singledispatch(
                    lambda cls, sql="", force=False: ran.append(("base", sql, force)) or "recorded"
                )
            )

        if register:

            @Service.handle.register(Service)
            def _(self, sql: str = "", force: bool = False, dry_run: bool = False):
                ran.append(("registered", sql, force, dry_run))
                return "recorded"

            @Service.__dict__["on_class"].__func__.register(type)
            def _(cls, sql: str = "", force: bool = False, dry_run: bool = False):
                ran.append(("registered", sql, force, dry_run))
                return "recorded"

        return Service()

    @staticmethod
    def forwarding(inner):
        import functools

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            return inner(*args, **kwargs)

        return wrapper

    @pytest.mark.parametrize(
        "how", ["bound", "classmethod", "wrapped", "wrapped-twice"], ids=lambda how: how
    )
    def test_a_singledispatch_method_with_an_implementation_registered_is_refused(
        self, example, how
    ):
        service = self.service([], register=True)
        func = {
            "bound": lambda: service.handle,
            "classmethod": lambda: service.on_class,
            "wrapped": lambda: self.forwarding(service.handle),
            "wrapped-twice": lambda: self.forwarding(self.forwarding(service.handle)),
        }[how]()
        tool = example.function_tool(func, name_override="handle", strict_mode=False)
        expected = "cannot gate 'handle': it runs a `functools.singledispatch` function"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)
        with pytest.raises(TypeError, match=expected):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)

    def test_negative_control_ungated_the_registered_method_runs_its_own_default(self, example):
        ran: list = []
        tool = example.function_tool(
            self.service(ran, register=True).handle, name_override="handle", strict_mode=False
        )
        assert "dry_run" not in tool.params_json_schema["properties"]
        example.run_calls([("handle", {"sql": "x"}, "c1")], gated=False, tools=[tool])
        assert ran == [("registered", "x", False, False)]

    def test_where_the_dispatchers_code_is_not_known_it_is_still_found(self, example, monkeypatch):
        """With the code object not found, at import or now, the dispatcher is known by
        its closure and by its attributes, and what the third review pinned holds."""
        adapter = _adapter()
        monkeypatch.setattr(adapter, "_SINGLEDISPATCH_WRAPPER", None)
        monkeypatch.setattr(adapter, "_singledispatch_wrapper_code", lambda: None)
        refused = example.function_tool(
            TestSecondReview.dispatching([]), name_override="lookup", strict_mode=False
        )
        with pytest.raises(TypeError, match="it runs a `functools.singledispatch` function"):
            example.gate_tools([refused], example.sql_scope_policy)
        third = TestThirdReview()
        third.test_a_service_object_with_a_dispatch_method_and_a_registry_is_gated(example)
        third.test_reading_for_a_singledispatch_function_runs_no_code_of_the_object(example)
        third.test_a_singledispatch_function_with_nothing_registered_is_refused(example)
        third.test_a_registry_whose_base_is_another_function_is_refused(example)
        # a service object whose ``registry`` is a plain dict is not one
        router = example.function_tool(Router([]), name_override="run_sql")
        assert example.gate_tools([router], example.sql_scope_policy)

    # --- an agent tool's parameters type: the field, its type and the true reason --------------

    @staticmethod
    def refused_fields():
        import datetime
        import decimal
        import pathlib
        import uuid

        from pydantic import AnyUrl

        return {
            "time": (datetime.time, "time"),
            "timedelta": (datetime.timedelta, "timedelta"),
            "decimal": (decimal.Decimal, "decimal"),
            "uuid": (uuid.UUID, "uuid"),
            "path": (pathlib.Path, "lax-or-strict"),
            "url": (AnyUrl, "function-wrap"),  # a validator of pydantic's own
            "frozenset": (typing.FrozenSet[str], "frozenset"),
            "sequence": (typing.Sequence[str], "json-or-python"),
            "tuple": (typing.Tuple[int, int], "tuple"),
            "set": (typing.Set[str], "set"),
            "bytes": (bytes, "bytes"),
        }

    @pytest.mark.parametrize(
        "case",
        [
            "time",
            "timedelta",
            "decimal",
            "uuid",
            "path",
            "url",
            "frozenset",
            "sequence",
            "tuple",
            "set",
            "bytes",
        ],
    )
    def test_a_refused_parameters_field_is_named_with_its_type(self, example, case):
        from pydantic import create_model

        annotation, kind = self.refused_fields()[case]
        parameters = create_model("Ask", task=(str, ...), value=(annotation, ...))
        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=parameters)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        cause = f"pydantic core type `{kind}`"
        if kind.startswith("function-"):
            cause = f"a validator, {cause}"
        assert str(refusal.value) == (
            "Recusal cannot gate agent tool 'ask_sub': its `parameters` model has a field "
            f"outside the data-only set: `value` ({cause}). The gate works "
            "out what the sub-agent is given only for a Pydantic model the SDK's own rule for "
            "data-only validation admits (`can_prepare_without_user_code`) and that is written "
            "out with no code of its own. Declare fields that are str, int, float, bool, None, "
            "Literal and lists of them, with JSON defaults, and build other types inside the "
            "sub-agent's own tools."
        )

    @staticmethod
    def refused_models():
        from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PrivateAttr

        class Private(BaseModel):
            task: str
            _seen: int = PrivateAttr(default=0)

        class Deferred(BaseModel):
            model_config = ConfigDict(defer_build=True)
            task: str

        class Recursive(BaseModel):
            task: str
            next: typing.Optional["Recursive"] = None

        Recursive.model_rebuild(_types_namespace={"Recursive": Recursive})

        class NotJson(BaseModel):
            task: str
            tags: typing.List[str] = ()  # type: ignore[assignment]

        class Excluded(BaseModel):
            task: str
            note: str = Field("", exclude_if=lambda value: value == "")

        class InAUnion(BaseModel):
            task: str
            rows: typing.Union[
                typing.Annotated[int, PlainSerializer(lambda value: value + 100)], str
            ] = 0

        model = "model is not one the SDK's rule for data-only validation admits"
        return {
            "a-private-attribute": (Private, model),
            "defer-build": (Deferred, model),
            "a-model-that-refers-to-itself": (
                Recursive,
                "type is not a Pydantic model the gate reads: its pydantic core type is "
                "`definitions` ",
            ),
            "a-default-that-is-not-json": (
                NotJson,
                "model has a field outside the data-only set: `tags` (a default that is not "
                "JSON data",
            ),
            "exclude-if": (Excluded, "model writes `note` out through code of its own"),
            "a-serializer-in-a-union-member": (
                InAUnion,
                "model writes `rows` out through code of its own",
            ),
        }

    @pytest.mark.parametrize(
        "case",
        [
            "a-private-attribute",
            "defer-build",
            "a-model-that-refers-to-itself",
            "a-default-that-is-not-json",
            "exclude-if",
            "a-serializer-in-a-union-member",
        ],
    )
    def test_a_refused_parameters_model_says_what_puts_it_outside(self, example, case):
        parameters, named = self.refused_models()[case]
        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=parameters)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value).startswith(
            f"Recusal cannot gate agent tool 'ask_sub': its `parameters` {named}"
        )

    def test_negative_control_ungated_exclude_if_runs_the_models_own_code(self, example):
        from agents.testing import ScriptedModel, assistant_message
        from pydantic import BaseModel, Field

        ran: list = []

        class Excluded(BaseModel):
            task: str
            note: str = Field("", exclude_if=lambda value: ran.append(value) or value == "")

        sub_model = ScriptedModel([[assistant_message("sub done")]])
        sub = example.Agent(name="sub", model=sub_model, tools=[example.run_sql])
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=Excluded)
        parent_model = example.scripted_model([("ask_sub", {"task": "list"}, "p1")])
        asyncio.run(example.run_agent(example.Agent(name="p", model=parent_model, tools=[tool])))
        assert ran, "the callable ran when the SDK wrote the input out"

    def test_text_a_float_field_reads_as_a_non_finite_number_is_refused(self, example):
        """``"nan"`` validates as a float and is not JSON data: refused before the policy,
        for that reason."""
        from pydantic import BaseModel

        class Ask(BaseModel):
            task: str
            ratio: float = 0.0

        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        (tool,) = example.gate_tools(
            [sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=Ask)],
            lambda name, args: [],
        )
        policy = self.Recording(lambda name, args: [])
        for text in ("nan", "inf", "-inf", "1e999"):
            arguments = '{"task": "list", "ratio": "%s"}' % text
            reason = _adapter()._function_tool_refusal("ask_sub", arguments, policy, tool)
            assert reason == (
                "Recusal refused `ask_sub`: the input the sub-agent would be given cannot be "
                "judged (the validated arguments are not JSON data: non-finite numbers are "
                "not JSON data)"
            )
        assert policy.seen == []
        assert (
            _adapter()._function_tool_refusal(
                "ask_sub", '{"task": "list", "ratio": "0.5"}', policy, tool
            )
            is None
        )
        assert policy.seen == [("ask_sub", {"task": "list", "ratio": 0.5})]


_REPLACED_AND_RELOADED = '''
import functools
import importlib

pristine = functools.singledispatch


class Traced:
    """A class-based instrumentation wrapper: it has no ``__code__``."""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)


functools.singledispatch = Traced(pristine)


@pristine
def early(value: str, force: bool = False) -> str:
    """Made by the function as it was, before the package is imported."""
    return "base"


import recusal_openai_agents  # noqa: E402
from agents import function_tool  # noqa: E402
from recusal_openai_agents import _gate, gate_tools  # noqa: E402

print("IMPORTED", _gate._SINGLEDISPATCH_WRAPPER)


def attempt(label, func):
    try:
        gate_tools([function_tool(func)], lambda name, args: [])
    except TypeError as exc:
        print(label, "REFUSED" if "it runs a `functools.singledispatch` function" in str(exc) else exc)
    else:
        print(label, "GATED")


attempt("early", early)
importlib.reload(functools)
assert functools.singledispatch is not pristine


@functools.singledispatch
def late(value: str, force: bool = False) -> str:
    """Made after the reload, by a function with another code object."""
    return "base"


attempt("late", late)
late.register(str)(lambda value, force=False, recursive=True: "registered")
attempt("late-registered", late)
'''


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestFifthReview:
    """The fifth review of 2026-11-22, closed on 2026-11-23. A ``functools.singledispatch``
    tool with nothing registered was gated, and the SDK judges every call of a model turn
    before any tool body runs, so an implementation registered during a turn (by another
    call of it, or by the tool's own first use) ran with a parameter no policy saw. A
    dispatcher that is a callable object's ``__call__`` was not found under a forwarding
    wrapper. The dispatcher was known by one code object read at import. And the refusal
    of an agent tool's ``parameters`` type named a type the data-only set admits."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording
    REFUSED = "it runs a `functools.singledispatch` function"

    # --- registered in the same turn: every dispatcher is refused when the agent is built ----

    @staticmethod
    def plugin_tools(example, events):
        """The reviewer's shape: ``export`` has nothing registered, and ``load_plugin``
        registers an implementation with a parameter of its own."""
        import functools

        @functools.singledispatch
        def export(target: str, overwrite: bool = False) -> str:
            """Record an export; contacts nothing."""
            events.append(("base", target, overwrite))
            return "exported"

        def load_plugin(name: str) -> str:
            """Record a plugin load; contacts nothing."""

            @export.register(str)
            def _(target, overwrite=False, recursive=True):
                events.append(("registered", target, overwrite, recursive))
                return "exported by the plugin"

            return "loaded"

        return [
            example.function_tool(export, strict_mode=False),
            example.function_tool(load_plugin),
        ]

    @staticmethod
    def lazy_tool(example, events):
        """One tool whose own first use registers an implementation."""
        import functools

        @functools.singledispatch
        def render(doc: str, strict: bool = True) -> str:
            """Record a rendering; contacts nothing."""
            events.append(("base", doc, strict))
            if len(render.registry) == 1:

                @render.register(str)
                def _(doc, strict=True, unsafe_html=True):
                    events.append(("registered", doc, strict, unsafe_html))
                    return "text"

            return "base"

        return example.function_tool(render, strict_mode=False)

    @pytest.mark.parametrize("order", ["load-then-export", "export-then-load"])
    def test_a_dispatcher_another_call_of_the_turn_would_register_on_is_refused(
        self, example, order
    ):
        events: list = []
        tools = self.plugin_tools(example, events)
        if order == "export-then-load":
            tools.reverse()
        policy = self.Recording(lambda name, args: [])
        with pytest.raises(TypeError, match=f"cannot gate 'export': {self.REFUSED}"):
            example.gated_agent(example.Agent(name="a", tools=tools), policy)
        with pytest.raises(TypeError, match=f"cannot gate 'export': {self.REFUSED}"):
            example.gate_tools(tools, policy)
        assert events == [] and policy.seen == []

    def test_negative_control_ungated_an_implementation_registered_in_the_turn_runs(self, example):
        """Why registered or not makes no difference: the call to ``export`` is checked
        while nothing is registered, and runs the implementation the earlier call of the
        same turn registered, with ``recursive`` at its own default."""
        events: list = []
        tools = self.plugin_tools(example, events)
        assert "recursive" not in tools[0].params_json_schema["properties"]
        outputs, _ = example.run_calls(
            [("load_plugin", {"name": "s3"}, "c1"), ("export", {"target": "bucket"}, "c2")],
            gated=False,
            tools=tools,
        )
        assert outputs == {"c1": "loaded", "c2": "exported by the plugin"}
        assert events == [("registered", "bucket", False, True)]

    def test_a_dispatcher_that_registers_on_its_own_first_use_is_refused(self, example):
        events: list = []
        tool = self.lazy_tool(example, events)
        with pytest.raises(TypeError, match=f"cannot gate 'render': {self.REFUSED}"):
            example.gated_agent(example.Agent(name="a", tools=[tool]), lambda name, args: [])
        assert events == []

    def test_negative_control_ungated_the_second_call_of_the_turn_runs_the_new_one(self, example):
        events: list = []
        tool = self.lazy_tool(example, events)
        example.run_calls(
            [("render", {"doc": "a"}, "c1"), ("render", {"doc": "b"}, "c2")],
            gated=False,
            tools=[tool],
        )
        assert events == [("base", "a", True), ("registered", "b", True, True)]

    @staticmethod
    def shapes(register):
        """A dispatcher in every place a call reaches one, by name."""
        import functools

        forwarding = TestFourthReview.forwarding
        service = TestFourthReview.service([], register=register)

        def plain():
            @functools.singledispatch
            def handle(sql: str = "", force: bool = False) -> str:
                """Record SQL text without connecting to a database."""
                return "recorded"

            if register:
                handle.register(str)(lambda sql="", force=False, dry_run=False: "recorded")
            return handle

        class Callable:
            __call__ = functools.singledispatch(lambda self, sql="", force=False: "recorded")

        class Static:
            handle = staticmethod(plain())

        class Method:
            @functools.singledispatchmethod
            def handle(self, sql: str = "", force: bool = False) -> str:
                """Record SQL text without connecting to a database."""
                return "recorded"

        if register:
            Callable.__call__.register(Callable)(
                lambda self, sql="", force=False, dry_run=False: "recorded"
            )
            Method.handle.register(str)(lambda self, sql="", force=False, dry_run=False: "recorded")
        return {
            "the-tool": plain,
            "wrapped": lambda: forwarding(plain()),
            "wrapped-twice": lambda: forwarding(forwarding(plain())),
            "cached": lambda: functools.lru_cache(maxsize=None)(plain()),
            "wrapped-over-cached": lambda: forwarding(functools.lru_cache(maxsize=None)(plain())),
            "bound": lambda: service.handle,
            "bound-wrapped": lambda: forwarding(service.handle),
            "classmethod": lambda: service.on_class,
            "staticmethod": lambda: Static().handle,
            "the-staticmethod-object": lambda: Static.__dict__["handle"],
            "singledispatchmethod": lambda: Method().handle,
            "singledispatchmethod-wrapped": lambda: forwarding(Method().handle),
            "bound-call": lambda: Callable().__call__,
            "wrapped-over-a-callable-object": lambda: forwarding(Callable()),
            "wrapped-twice-over-a-callable-object": lambda: forwarding(forwarding(Callable())),
        }

    SHAPES = [
        "the-tool",
        "wrapped",
        "wrapped-twice",
        "cached",
        "wrapped-over-cached",
        "bound",
        "bound-wrapped",
        "classmethod",
        "staticmethod",
        "the-staticmethod-object",
        "singledispatchmethod",
        "singledispatchmethod-wrapped",
        "bound-call",
        "wrapped-over-a-callable-object",
        "wrapped-twice-over-a-callable-object",
    ]

    @pytest.mark.parametrize("register", [False, True], ids=["nothing-registered", "registered"])
    @pytest.mark.parametrize("how", SHAPES)
    def test_a_dispatcher_is_refused_wherever_it_stands_registered_or_not(
        self, example, how, register
    ):
        shapes = self.shapes(register)
        assert sorted(shapes) == sorted(self.SHAPES)
        tool = example.function_tool(shapes[how](), name_override="handle", strict_mode=False)
        with pytest.raises(TypeError, match=f"cannot gate 'handle': {self.REFUSED}"):
            example.gate_tools([tool], example.sql_scope_policy)
        with pytest.raises(TypeError, match=f"cannot gate 'handle': {self.REFUSED}"):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)

    def test_negative_control_ungated_a_wrapped_callable_objects_dispatcher_runs_its_own(
        self, example
    ):
        """The reviewer's second shape: the dispatcher is a callable object's ``__call__``
        under a forwarding wrapper, and the registered implementation adds ``dry_run``."""
        import functools

        ran: list = []

        class Runner:
            __call__ = functools.singledispatch(
                lambda self, sql="", force=False: ran.append(("base", sql, force)) or "recorded"
            )

        @Runner.__call__.register(Runner)
        def _(self, sql: str = "", force: bool = False, dry_run: bool = False):
            ran.append(("registered", sql, force, dry_run))
            return "recorded"

        tool = example.function_tool(
            TestFourthReview.forwarding(Runner()), name_override="handle", strict_mode=False
        )
        assert "dry_run" not in tool.params_json_schema["properties"]
        example.run_calls([("handle", {"sql": "x"}, "c1")], gated=False, tools=[tool])
        assert ran == [("registered", "x", False, False)]

    def test_the_refusal_says_the_cause_and_what_to_do(self, example):
        tool = example.function_tool(
            self.shapes(False)["the-tool"](), name_override="handle", strict_mode=False
        )
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == (
            "Recusal cannot gate 'handle': it runs a `functools.singledispatch` function: the "
            "implementation that runs is chosen at call time by the type of an argument, one "
            "can be registered at any moment, and each has its own parameters and defaults, "
            "so the gate cannot tie a call to the function that will run. Register one plain "
            "function as the tool and dispatch inside its body."
        )

    def test_the_guardrail_asks_again_on_every_call(self, example):
        """A tool that was gated and whose function is then changed so that it reaches a
        dispatcher is refused on its next call, before the policy and the body."""
        import functools

        ran: list = []

        def run_sql(sql: str) -> str:
            """Record SQL text without connecting to a database."""
            ran.append(sql)
            return "recorded"

        policy = self.Recording(lambda name, args: [])
        (gated,) = example.gate_tools([example.function_tool(run_sql)], policy)
        outputs, _ = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")], gated=False, tools=[gated]
        )
        assert outputs == {"c1": "recorded"} and ran == [SELECT]
        run_sql.__wrapped__ = functools.singledispatch(lambda sql: "other")  # type: ignore[attr-defined]
        outputs, _ = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c2")], gated=False, tools=[gated]
        )
        assert outputs["c2"].startswith(f"Recusal refused `run_sql`: {self.REFUSED}")
        assert ran == [SELECT] and len(policy.seen) == 1

    # --- every callable the kernel's walk visits is asked ---------------------------------------

    def test_the_gates_walk_takes_every_step_the_kernels_walk_takes(self, example):
        """The dispatcher question is asked of the callables ``_reached_callables`` finds.
        For each shape, whatever the kernel's own step leads to from a callable the gate
        reached is reached by the gate too, as the function under it."""
        import functools
        import inspect

        from recusal import invocation

        adapter = _adapter()

        def under(link):
            while inspect.ismethod(link) or isinstance(link, functools.partial):
                link = link.__func__ if inspect.ismethod(link) else link.func
            return link

        def plain(sql: str = "", force: bool = False) -> str:
            return "recorded"

        class Callable:
            def __call__(self, sql: str = "", force: bool = False) -> str:
                return "recorded"

        class Decorator:
            def __init__(self, wrapped):
                functools.update_wrapper(self, wrapped)

            def __call__(self, *args, **kwargs):
                return self.__wrapped__(*args, **kwargs)

        forwarding = TestFourthReview.forwarding
        shapes = [
            plain,
            forwarding(forwarding(plain)),
            functools.lru_cache(maxsize=None)(forwarding(plain)),
            functools.partial(forwarding(plain), force=True),
            Callable(),
            Callable().__call__,
            forwarding(Callable()),
            Decorator(plain),
            forwarding(Decorator(forwarding(plain))),
            staticmethod(plain),
            *(make() for make in self.shapes(True).values()),
        ]
        step = getattr(invocation, "_next_callables", None)
        one_argument = step is not None and len(inspect.signature(step).parameters) == 1
        for func in shapes:
            reached, complete = adapter._reached_callables(func)
            assert complete and reached[0] is func
            assert all(sum(link is other for other in reached) == 1 for link in reached)
            if not one_argument:
                continue  # a kernel whose step takes more: it refuses a dispatcher itself
            for link in reached:
                try:
                    following = step(link)
                except TypeError:
                    continue  # a ``__call__`` the kernel refuses to read
                for nxt in following:
                    assert any(under(nxt) is other for other in reached), (func, link, nxt)

    def test_a_walk_that_does_not_finish_is_refused(self, example, monkeypatch):
        tool = example.function_tool(TestFourthReview.forwarding(self.plain_function()))
        assert example.gate_tools([tool], example.sql_scope_policy)
        monkeypatch.setattr(_adapter(), "_MAX_REACHED", 1)
        with pytest.raises(TypeError, match="under more wrappers than the gate reads"):
            example.gate_tools([tool], example.sql_scope_policy)

    @staticmethod
    def plain_function():
        def run_sql(sql: str) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        return run_sql

    def test_a_callable_the_question_cannot_be_asked_of_is_refused(self, example, monkeypatch):
        tool = example.function_tool(self.plain_function())

        def broken(link):
            raise RuntimeError("no answer")

        monkeypatch.setattr(_adapter(), "_is_dispatcher", broken)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == (
            "Recusal cannot gate 'run_sql': the gate could not tell whether a callable it "
            "reaches is a `functools.singledispatch` function (RuntimeError), and one is "
            "refused. Register one plain function as the tool."
        )

    def test_a_kernel_that_refuses_a_dispatcher_itself_gets_the_same_reason(
        self, example, monkeypatch
    ):
        """A later kernel's walk raises ``DispatchedAtCallTime`` for a dispatcher it finds.
        The released kernel has no such class, so it is known by its name."""
        adapter = _adapter()
        tool = example.function_tool(self.plain_function())

        class DispatchedAtCallTime(ValueError):
            pass

        def walk(func):
            raise DispatchedAtCallTime("chosen at call time")

        monkeypatch.setattr(adapter, "_reached_parameters", walk)
        with pytest.raises(TypeError, match=f"cannot gate 'run_sql': {self.REFUSED}"):
            example.gate_tools([tool], example.sql_scope_policy)

        def other(func):
            raise ValueError("something else")

        monkeypatch.setattr(adapter, "_reached_parameters", other)
        with pytest.raises(TypeError, match=r"cannot be inspected \(ValueError\)"):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_a_kernel_that_refuses_a_hidden_wrapper_gets_a_reason_that_names_it(
        self, example, monkeypatch
    ):
        """A later kernel's walk raises ``HiddenWrapper`` (a ``ValueError``) for a wrapper that
        holds a second wrapper of the function it does not follow. The refusal says so and
        gives the remedy, with the kernel's own text, which names the parameter."""
        adapter = _adapter()
        tool = example.function_tool(self.plain_function())

        class HiddenWrapper(ValueError):
            pass

        def walk(func):
            raise HiddenWrapper("`force` is a parameter of a wrapper the walk does not follow")

        monkeypatch.setattr(adapter, "_reached_parameters", walk)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        reason = str(refusal.value)
        assert "holds a second wrapper of it that the gate does not follow" in reason
        assert "`force` is a parameter of a wrapper the walk does not follow" in reason
        assert "cannot be inspected" not in reason

    # --- how a dispatcher is known ---------------------------------------------------------------

    @staticmethod
    def dispatcher():
        import functools

        @functools.singledispatch
        def lookup(key: str) -> str:
            return key

        return lookup

    def test_each_of_the_three_ways_knows_a_dispatcher_on_its_own(self, example, monkeypatch):
        import types

        adapter = _adapter()
        lookup = self.dispatcher()
        assert type(lookup) is types.FunctionType
        assert type(adapter._SINGLEDISPATCH_WRAPPER) is types.CodeType
        assert lookup.__code__ is adapter._SINGLEDISPATCH_WRAPPER
        assert lookup.__code__ is adapter._singledispatch_wrapper_code()
        dispatch = adapter._closure_value(lookup, "dispatch")
        assert dispatch is lookup.dispatch
        assert type(adapter._closure_value(dispatch, "registry")) is dict
        assert adapter._closure_value(lookup, "no_such_name") is None
        for plain in (print, len, Router([]), None, self.plain_function(), "text"):
            assert adapter._dispatch_function(plain) is False
            assert adapter._is_dispatcher(plain) is False

        # the attributes alone: no code known, a closure that cannot be read
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_SINGLEDISPATCH_WRAPPER", None)
            patch.setattr(adapter, "_singledispatch_wrapper_code", lambda: None)
            patch.setattr(adapter, "_closure_value", lambda func, name: None)
            assert adapter._dispatch_function(lookup) is True
            # a wrapper that carries a copy of them is taken for one
            assert adapter._dispatch_function(TestFourthReview.forwarding(lookup)) is True

        bare = self.dispatcher()
        for name in ("registry", "dispatch", "register"):
            delattr(bare, name)
        assert adapter._dispatch_function(bare) is True
        # the closure alone
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_SINGLEDISPATCH_WRAPPER", None)
            patch.setattr(adapter, "_singledispatch_wrapper_code", lambda: None)
            assert adapter._dispatch_function(bare) is True
            patch.setattr(adapter, "_closure_value", lambda func, name: None)
            assert adapter._dispatch_function(bare) is False
        # the code as it was at import alone, and as functools has it now alone
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_closure_value", lambda func, name: None)
            assert adapter._dispatch_function(bare) is True
            patch.setattr(adapter, "_singledispatch_wrapper_code", lambda: None)
            assert adapter._dispatch_function(bare) is True
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_closure_value", lambda func, name: None)
            patch.setattr(adapter, "_SINGLEDISPATCH_WRAPPER", None)
            assert adapter._dispatch_function(bare) is True

    @pytest.mark.parametrize("missing", ["registry", "dispatch", "register", "nothing"])
    def test_the_attributes_are_a_mapping_proxy_beside_two_callables(self, example, missing):
        """An object that is no function is known by all three attributes together; a
        ``registry`` of another type, or one of the two callables missing, is not one."""
        import types

        class Service:
            def __call__(self, sql: str) -> str:
                """Record SQL text without connecting to a database."""
                return "recorded"

        attributes = {
            "registry": types.MappingProxyType({object: None}),
            "dispatch": lambda self, cls: None,
            "register": lambda self, cls: None,
        }
        if missing == "registry":
            attributes["registry"] = {object: None}  # a plain dict
        elif missing != "nothing":
            del attributes[missing]
        for name, value in attributes.items():
            setattr(Service, name, value)
        tool = example.function_tool(Service(), name_override="run_sql")
        if missing == "nothing":
            assert _adapter()._dispatch_function(Service()) is True
            with pytest.raises(TypeError, match=self.REFUSED):
                example.gate_tools([tool], example.sql_scope_policy)
        else:
            assert _adapter()._dispatch_function(Service()) is False
            assert example.gate_tools([tool], example.sql_scope_policy)

    def test_asking_runs_no_code_of_the_object(self, example):
        touched: list = []

        class Lazy:
            def __getattr__(self, name):
                touched.append(name)
                raise AttributeError(name)

            @property
            def register(self):
                touched.append("register")
                return lambda cls: None

            @property
            def _unbound(self):
                touched.append("_unbound")
                return None

            @property
            def dispatcher(self):
                touched.append("dispatcher")
                return None

        assert _adapter()._is_dispatcher(Lazy()) is False
        assert touched == []

    def test_a_dispatcher_built_by_another_copy_of_functools_is_refused(self, example):
        """What ``importlib.reload(functools)`` leaves behind: a ``singledispatch`` with
        another code object and a ``singledispatchmethod`` of another class. Built here
        from a second copy of the module, so the interpreter's own is left alone."""
        import functools
        import importlib.util

        adapter = _adapter()
        spec = importlib.util.find_spec("functools")
        assert spec is not None and spec.loader is not None
        other = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(other)
        assert other.singledispatch is not functools.singledispatch
        assert other.singledispatchmethod is not functools.singledispatchmethod

        @other.singledispatch
        def lookup(key: str, force: bool = False) -> str:
            """Record a lookup; contacts nothing."""
            return "recorded"

        assert lookup.__code__ is not adapter._SINGLEDISPATCH_WRAPPER
        assert lookup.__code__ is not adapter._singledispatch_wrapper_code()

        class Service:
            @other.singledispatchmethod
            def lookup(self, key: str, force: bool = False) -> str:
                """Record a lookup; contacts nothing."""
                return "recorded"

        assert not isinstance(Service.__dict__["lookup"], functools.singledispatchmethod)
        for func in (lookup, TestFourthReview.forwarding(lookup), Service().lookup):
            tool = example.function_tool(func, name_override="lookup", strict_mode=False)
            with pytest.raises(TypeError, match=f"cannot gate 'lookup': {self.REFUSED}"):
                example.gate_tools([tool], example.sql_scope_policy)

    def test_the_code_is_read_without_raising_whatever_stands_at_singledispatch(
        self, example, monkeypatch
    ):
        import functools
        import types

        adapter = _adapter()
        pristine = functools.singledispatch
        assert adapter._singledispatch_wrapper_code() is self.dispatcher().__code__

        class NoCode:
            def __call__(self, func):
                return pristine(func)

        class Raises:
            @property
            def __code__(self):
                raise RuntimeError("instrumented")

        class NotCode:
            __code__ = "text"

        def two_of_them(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            first = wrapper

            def wrapper(*args, **kwargs):  # noqa: F811
                return first(*args, **kwargs)

            return wrapper

        def one_of_them(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        def none_of_them(func):
            return func

        class LooksLikeCode:
            # not a code object, though it holds one nested function named ``wrapper``
            __code__ = types.SimpleNamespace(co_consts=one_of_them.__code__.co_consts)

        assert [c.co_name for c in LooksLikeCode.__code__.co_consts if hasattr(c, "co_name")] == [
            "wrapper"
        ]
        replacements = (
            NoCode(),
            Raises(),
            NotCode(),
            LooksLikeCode(),
            two_of_them,
            none_of_them,
            None,
        )
        for replacement in replacements:
            monkeypatch.setattr(functools, "singledispatch", replacement)
            assert adapter._singledispatch_wrapper_code() is None, replacement
        monkeypatch.setattr(functools, "singledispatch", one_of_them)
        found = adapter._singledispatch_wrapper_code()
        assert type(found) is types.CodeType and found is one_of_them(len).__code__
        monkeypatch.setattr(functools, "singledispatch", pristine)

    def test_how_a_singledispatchmethod_is_known(self, example):
        """By its class (a subclass too), or by the class's name together with a
        ``dispatcher`` that is a dispatch function; what it hands out, by the bound
        ``register`` it carries or by the descriptor it holds as ``_unbound`` (the shape
        Python 3.14 gives it, built by hand here so every Python runs it)."""
        import functools

        adapter = _adapter()

        class Renamed(functools.singledispatchmethod):
            pass

        class Service:
            @Renamed
            def lookup(self, key: str, force: bool = False) -> str:
                """Record a lookup; contacts nothing."""
                return "recorded"

            @functools.singledispatchmethod
            def plain(self, key: str = "") -> str:
                return key

        descriptor = Service.__dict__["lookup"]
        assert type(descriptor).__name__ == "Renamed"
        assert adapter._dispatch_method_descriptor(descriptor) is True
        assert adapter._dispatch_method(Service().lookup) is True
        tool = example.function_tool(Service().lookup, strict_mode=False)
        with pytest.raises(TypeError, match=f"cannot gate 'lookup': {self.REFUSED}"):
            example.gate_tools([tool], example.sql_scope_policy)

        # the name of the class says nothing without a dispatcher under it
        class singledispatchmethod:  # noqa: N801, the name is the point
            dispatcher = None

        assert adapter._dispatch_method_descriptor(singledispatchmethod()) is False
        singledispatchmethod.dispatcher = self.dispatcher()
        assert adapter._dispatch_method_descriptor(singledispatchmethod()) is True

        class HandedOut:
            def __init__(self, unbound):
                self._unbound = unbound

        assert adapter._dispatch_method(HandedOut(Service.__dict__["plain"])) is True
        assert adapter._dispatch_method(HandedOut(None)) is False
        assert adapter._dispatch_method(HandedOut(self.plain_function())) is False

    def test_importing_the_package_under_a_replaced_singledispatch_and_a_reload(self, example):
        """In a process of its own: ``functools.singledispatch`` replaced by an object
        with no ``__code__`` before the import, which made the import raise, then
        ``functools`` reloaded, after which a new dispatcher was gated."""
        import os

        src = ROOT / "adapters" / "openai-agents" / "src"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(src)]))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", _REPLACED_AND_RELOADED],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            "IMPORTED None",
            "early REFUSED",
            "late REFUSED",
            "late-registered REFUSED",
        ]

    # --- an agent tool's parameters type: what puts a field outside ----------------------------

    @staticmethod
    def outside():
        import datetime

        from pydantic import BaseModel, Field, field_validator, model_validator

        class Inner(BaseModel):
            name: str

        class Pattern(BaseModel):
            task: str
            value: str = Field(pattern=r"^[a-z]+$")

        class OptionalPattern(BaseModel):
            task: str
            value: typing.Optional[str] = Field(None, pattern=r"^[a-z]+$")

        class NestedModels(BaseModel):
            task: str
            value: typing.List[Inner]

        class ListOfDates(BaseModel):
            task: str
            value: typing.List[typing.List[datetime.date]] = []

        class InAUnion(BaseModel):
            task: str
            value: typing.Union[int, datetime.date] = 0

        class OptionalDate(BaseModel):
            task: str
            value: typing.Optional[datetime.date] = None

        class FloatLiteral(BaseModel):
            task: str
            value: typing.Literal[0.5, 1.5] = 0.5

        class FieldValidator(BaseModel):
            task: str
            value: str = ""

            @field_validator("value")
            @classmethod
            def check(cls, value):
                return value

        class ModelValidator(BaseModel):
            task: str

            @model_validator(mode="after")
            def check(self):
                return self

        field = "model has a field outside the data-only set: `value` "
        return {
            "a-pattern": (Pattern, field + "(a string pattern). "),
            "an-optional-pattern": (OptionalPattern, field + "(a string pattern). "),
            "a-list-of-models": (NestedModels, field + "(a list of pydantic core type `model`). "),
            "a-list-of-lists-of-dates": (
                ListOfDates,
                field + "(a list of a list of pydantic core type `date`). ",
            ),
            "a-date-in-a-union": (InAUnion, field + "(a union with pydantic core type `date`). "),
            "an-optional-date": (OptionalDate, field + "(pydantic core type `date`). "),
            "a-literal-of-floats": (
                FloatLiteral,
                field + "(a `Literal` of values other than None, a bool, an int or a str). ",
            ),
            "a-field-validator": (
                FieldValidator,
                field + "(a validator, pydantic core type `function-after`). ",
            ),
            "a-model-validator": (
                ModelValidator,
                "type has a validator of its own on the type as a whole (a `model_validator`, "
                "for one; its pydantic core type is `function-after`). ",
            ),
        }

    @pytest.mark.parametrize(
        "case",
        [
            "a-pattern",
            "an-optional-pattern",
            "a-list-of-models",
            "a-list-of-lists-of-dates",
            "a-date-in-a-union",
            "an-optional-date",
            "a-literal-of-floats",
            "a-field-validator",
            "a-model-validator",
        ],
    )
    def test_a_refused_parameters_type_names_what_puts_it_outside(self, example, case):
        """``str`` and ``list`` are in the data-only set: a pattern field was reported as
        "pydantic core type `str`", a list of models as "`list`", and a model with a
        validator as a type that "is not a Pydantic model"."""
        parameters, named = self.outside()[case]
        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=parameters)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == (
            f"Recusal cannot gate agent tool 'ask_sub': its `parameters` {named}"
            "The gate works out what the sub-agent is given only for a Pydantic model the "
            "SDK's own rule for data-only validation admits (`can_prepare_without_user_code`) "
            "and that is written out with no code of its own. Declare fields that are str, "
            "int, float, bool, None, Literal and lists of them, with JSON defaults, and build "
            "other types inside the sub-agent's own tools."
        )

    def test_the_cause_of_a_schema_the_gate_cannot_read_is_still_a_sentence(self, example):
        cause = _adapter()._outside_cause
        assert cause(None) == "pydantic core type unknown"
        assert cause({}) == "pydantic core type unknown"
        assert cause({"type": "dict", "values_schema": {"type": "date"}}) == (
            "a mapping with values of pydantic core type `date`"
        )
        assert cause({"type": "dict", "keys_schema": {"type": "date"}}) == (
            "a mapping with keys of pydantic core type `date`"
        )
        nested = {"type": "default", "schema": {"type": "nullable", "schema": {"type": "set"}}}
        assert cause(nested) == "pydantic core type `set`"
