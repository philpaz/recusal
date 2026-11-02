"""Regression tests for the offline Pydantic AI + Recusal capability example."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pydantic_ai_gate.py"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"
PIN = "pydantic-ai-slim==2.64.0"


@pytest.mark.parametrize("minor", [9, 10])
def test_python_version_refusal_is_clear_on_every_python(minor):
    code = (
        "import runpy, sys; "
        f"sys.version_info = (3, {minor}, 0, 'final', 0); "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "pydantic_ai_gate.py requires Python 3.11+" in result.stderr
    assert f"this is Python 3.{minor}." in result.stderr


def test_pydantic_ai_stays_out_of_the_recusal_package_and_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "pydantic-ai" not in pyproject and "pydantic_ai" not in pyproject
    for path in (ROOT / "recusal").rglob("*.py"):
        assert "pydantic_ai" not in path.read_text(encoding="utf-8"), path


def test_pinned_version_is_explicit_in_example_docs_and_ci():
    for path in (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        assert PIN in path.read_text(encoding="utf-8"), path.name
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert ci.count(PIN) == 2


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Pydantic AI requires Python 3.11+")
class TestPydanticAiGate:
    @pytest.fixture
    def example(self):
        pytest.importorskip("pydantic_ai", reason="pydantic-ai-slim is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_pydantic_ai_gate_example", EXAMPLE)
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

    # --- the capability route ------------------------------------------------------------

    def test_allowed_call_runs_its_tool(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": SELECT}, "c1")])
        assert executed == [("run_sql", {"sql": SELECT})]
        assert responses == {"c1": ("success", f"recorded SQL: {SELECT}")}

    def test_refused_call_does_not_run_and_reason_reaches_the_model(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")])
        assert executed == []
        outcome, content = responses["c1"]
        assert outcome == "failed"
        assert content.startswith("Recusal refused `run_sql` [FAIL]")
        assert "top-level WHERE" in content

    @pytest.mark.parametrize("via_hooks", [False, True], ids=["capability", "hooks"])
    def test_several_calls_in_one_turn_are_gated_separately(self, example, via_hooks):
        responses, executed = example.run_calls(self.three_calls(), via_hooks=via_hooks)
        assert responses["c1"] == ("success", "hello")
        assert responses["c2"][0] == "failed" and "Recusal refused" in responses["c2"][1]
        assert responses["c3"] == ("success", f"recorded SQL: {SELECT}")
        assert sorted(executed) == [("echo", {"text": "hello"}), ("run_sql", {"sql": SELECT})]

    @pytest.mark.parametrize("via_hooks", [False, True], ids=["capability", "hooks"])
    def test_policy_exception_fails_closed(self, example, via_hooks):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        responses, executed = example.run_calls(
            [("echo", {"text": "unsafe"}, "c1")], policy=broken_policy, via_hooks=via_hooks
        )
        assert executed == []
        assert responses["c1"] == (
            "failed",
            "Recusal failed closed (policy error): policy exploded",
        )

    def test_a_crash_inside_decide_also_fails_closed(self, example, monkeypatch):
        def crashing_decide(*args, **kwargs):
            raise RuntimeError("decide crashed")

        monkeypatch.setattr(example, "decide", crashing_decide)
        responses, executed = example.run_calls([("echo", {"text": "unsafe"}, "c1")])
        assert executed == []
        assert responses["c1"] == ("failed", "Recusal failed closed (policy error): decide crashed")

    def test_non_mapping_arguments_are_refused_by_the_gate_itself(self, example):
        assert example._refusal("run_sql", ["DELETE"], example.sql_scope_policy) == (
            "Recusal refused `run_sql`: arguments are not a mapping"
        )
        assert example._refusal("run_sql", "DELETE", example.sql_scope_policy) is not None

    def test_negative_control_runs_delete_without_the_gate(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")], gated=False)
        assert executed == [("run_sql", {"sql": DELETE})]
        assert responses["c1"] == ("success", f"recorded SQL: {DELETE}")

    def test_a_refusal_spends_no_retry_budget(self, example):
        """A refused call is not a model mistake to correct: the run goes on, with the
        reason in front of the model, even when the agent allows no retries at all."""
        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import FunctionModel

        turns = {"n": 0}

        def insists(messages, info):
            turns["n"] += 1
            if turns["n"] <= 2:
                call = ToolCallPart(
                    tool_name="run_sql", args={"sql": DELETE}, tool_call_id=f"c{turns['n']}"
                )
                return ModelResponse(parts=[call])
            return ModelResponse(parts=[TextPart(content="done")])

        example.EXECUTED.clear()
        agent = Agent(
            FunctionModel(insists),
            tools=[example.run_sql],
            capabilities=[example.RecusalGate(example.sql_scope_policy)],
            retries=0,
        )
        result = agent.run_sync("go")
        responses = example.responses_of(result)
        assert result.output == "done"
        assert example.EXECUTED == []
        assert responses["c1"][0] == "failed" and responses["c2"][0] == "failed"

    # --- where the gate attaches ---------------------------------------------------------

    def test_per_run_capabilities_gate_an_ungated_agent_for_that_run(self, example):
        agent = example.scripted_agent([("run_sql", {"sql": DELETE}, "c1")])
        assert not example.is_gated(agent)
        example.EXECUTED.clear()
        result = agent.run_sync("go", capabilities=[example.RecusalGate(example.sql_scope_policy)])
        assert example.EXECUTED == []
        assert example.responses_of(result)["c1"][0] == "failed"

    def test_a_toolset_registered_with_toolsets_is_gated_the_same_way(self, example):
        from pydantic_ai import Agent
        from pydantic_ai.toolsets import FunctionToolset

        example.EXECUTED.clear()
        agent = Agent(
            example.scripted_model([("run_sql", {"sql": DELETE}, "c1")]),
            toolsets=[FunctionToolset([example.run_sql])],
            capabilities=[example.RecusalGate(example.sql_scope_policy)],
        )
        result = agent.run_sync("go")
        assert example.EXECUTED == []
        assert "Recusal refused" in example.responses_of(result)["c1"][1]

    def test_is_gated_reads_the_agents_own_capabilities(self, example):
        from pydantic_ai.capabilities import CombinedCapability, Hooks

        gate = example.RecusalGate(example.sql_scope_policy)
        assert example.is_gated(example.scripted_agent([], capabilities=[gate]))
        assert example.is_gated(
            example.scripted_agent([], capabilities=[CombinedCapability([gate])])
        )
        assert not example.is_gated(example.scripted_agent([]))
        # the plain hook form is a function inside Hooks, which is_gated does not read
        hook = Hooks(
            before_tool_execute=example.recusal_before_tool_execute(example.sql_scope_policy)
        )
        assert not example.is_gated(example.scripted_agent([], capabilities=[hook]))

    # --- ordering: why the gate is innermost ----------------------------------------------

    @staticmethod
    def _rewrite_to_delete():
        async def rewrite(ctx, *, call, tool_def, args):
            if call.tool_name == "run_sql":
                return {**args, "sql": DELETE}
            return args

        return rewrite

    def test_why_the_gate_is_innermost_a_later_hook_can_change_what_runs(self, example):
        from pydantic_ai.capabilities import Hooks

        rewrite = Hooks(before_tool_execute=self._rewrite_to_delete())
        select = [("run_sql", {"sql": SELECT}, "c1")]
        # the plain hook form listed FIRST judges the SELECT; the rewrite after it turns the
        # call into a DELETE that runs
        plain = Hooks(
            before_tool_execute=example.recusal_before_tool_execute(example.sql_scope_policy)
        )
        example.EXECUTED.clear()
        result = example.scripted_agent(select, capabilities=[plain, rewrite]).run_sync("go")
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]
        assert example.responses_of(result)["c1"][0] == "success"
        # RecusalGate listed first is still sorted innermost: it judges the rewritten DELETE
        example.EXECUTED.clear()
        gate = example.RecusalGate(example.sql_scope_policy)
        result = example.scripted_agent(select, capabilities=[gate, rewrite]).run_sync("go")
        assert example.EXECUTED == []
        assert "Recusal refused" in example.responses_of(result)["c1"][1]

    def test_the_limit_another_innermost_capability_listed_after_the_gate_runs_after_it(
        self, example
    ):
        """Pinned so the docstring's "list the gate last" stays the accurate advice."""
        from dataclasses import dataclass

        from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering

        rewrite = self._rewrite_to_delete()

        @dataclass
        class InnerRewrite(AbstractCapability):
            def get_ordering(self):
                return CapabilityOrdering(position="innermost")

            async def before_tool_execute(self, ctx, *, call, tool_def, args):
                return await rewrite(ctx, call=call, tool_def=tool_def, args=args)

        select = [("run_sql", {"sql": SELECT}, "c1")]
        gate = example.RecusalGate(example.sql_scope_policy)
        example.EXECUTED.clear()
        example.scripted_agent(select, capabilities=[gate, InnerRewrite()]).run_sync("go")
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]  # the bypass
        example.EXECUTED.clear()
        result = example.scripted_agent(select, capabilities=[InnerRewrite(), gate]).run_sync("go")
        assert example.EXECUTED == []  # the gate last, as documented
        assert "Recusal refused" in example.responses_of(result)["c1"][1]

    # --- a delegated agent: the trap, and both ways out -------------------------------------

    def _delegation(self, example, child_capabilities=None, per_run=None):
        from pydantic_ai import Agent

        child = example.scripted_agent(
            [("run_sql", {"sql": DELETE}, "s1")],
            tools=[example.run_sql],
            capabilities=child_capabilities,
        )
        answers = {}

        async def ask_child(request: str) -> str:
            """Delegate the request to the child agent."""
            result = await child.run(request, capabilities=per_run)
            answers.update(example.responses_of(result))
            return result.output

        parent = Agent(
            example.scripted_model([("ask_child", {"request": "clean up"}, "p1")]),
            tools=[ask_child],
            capabilities=[example.RecusalGate(example.sql_scope_policy)],
        )
        example.EXECUTED.clear()
        return parent, child, answers

    def test_negative_control_a_gate_on_the_parent_alone_misses_the_delegated_agent(self, example):
        parent, child, answers = self._delegation(example)
        parent.run_sync("go")
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]
        assert answers["s1"] == ("success", f"recorded SQL: {DELETE}")
        assert example.is_gated(parent) and not example.is_gated(child)

    def test_a_gate_on_the_child_itself_refuses_its_call(self, example):
        parent, child, answers = self._delegation(
            example, child_capabilities=[example.RecusalGate(example.sql_scope_policy)]
        )
        parent.run_sync("go")
        assert example.EXECUTED == []
        assert "Recusal refused" in answers["s1"][1]
        assert example.is_gated(child)

    def test_a_per_run_gate_inside_the_delegating_tool_refuses_its_call(self, example):
        parent, child, answers = self._delegation(
            example, per_run=[example.RecusalGate(example.sql_scope_policy)]
        )
        parent.run_sync("go")
        assert example.EXECUTED == []
        assert "Recusal refused" in answers["s1"][1]

    # --- the script ------------------------------------------------------------------------

    def test_example_runs_end_to_end(self, example):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("tools that ran: 2") == 2
        assert result.stdout.count("Recusal refused `run_sql` [FAIL]") == 2
        assert "RecusalGate:" in result.stdout and "Hooks(before_tool_execute):" in result.stdout
