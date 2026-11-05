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

    @pytest.mark.parametrize("via_hooks", [False, True], ids=["capability", "hooks"])
    def test_a_policy_that_writes_to_what_it_judged_does_not_change_what_runs(
        self, example, via_hooks
    ):
        """Containment of a buggy trusted policy: it judges a detached copy, the tool runs
        the validated dict itself. Before this guard, the write reached the tool."""
        import copy

        seen = []

        def mutating(tool_name, args):
            seen.append(copy.deepcopy(args))
            findings = example.sql_scope_policy(tool_name, args)
            args["sql"] = DELETE  # an accidental write after the evidence was gathered
            return findings

        responses, executed = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")], policy=mutating, via_hooks=via_hooks
        )
        assert seen == [{"sql": SELECT}]
        assert executed == [("run_sql", {"sql": SELECT})]
        assert responses == {"c1": ("success", f"recorded SQL: {SELECT}")}

    def test_the_policy_observes_nested_containers_detached_from_the_invocation(self, example):
        invocation = {"sql": SELECT, "meta": {"tags": ["read"], "limits": [1, 2]}}
        before = {"sql": SELECT, "meta": {"tags": ["read"], "limits": [1, 2]}}
        observed = []

        def mutating(tool_name, args):
            observed.append(args)
            args["meta"]["tags"].append("write")
            args["meta"]["limits"][0] = 999
            args["sql"] = DELETE
            return []

        assert example._refusal("run_sql", invocation, mutating) is None
        assert invocation == before, "the invocation is what the tool runs; it is untouched"
        assert observed[0] is not invocation and observed[0]["meta"] is not invocation["meta"]
        assert observed[0]["sql"] == DELETE  # the policy changed its own copy only

    def test_unsupported_objects_are_refused_without_calling_their_copy_method(self, example):
        called = []

        class Undetachable:
            def __deepcopy__(self, memo):
                called.append("copy")
                raise TypeError("PRIVATE_CREDENTIAL_SENTINEL")

        reason = example._refusal("run_sql", {"sql": SELECT, "h": Undetachable()}, lambda *_: [])
        assert reason == (
            "Recusal refused `run_sql`: arguments cannot be detached for judging (TypeError); "
            "declare data-only parameters and validate inside the tool body"
        )
        assert called == [] and "PRIVATE_CREDENTIAL_SENTINEL" not in reason

    @pytest.mark.parametrize("via_hooks", [False, True], ids=["capability", "hooks"])
    @pytest.mark.parametrize("mutating", [False, True], ids=["observational", "mutating"])
    def test_a_mutable_valued_enum_is_refused_before_policy_or_execution(
        self, example, via_hooks, mutating
    ):
        from enum import Enum

        class Action(Enum):
            READ = {"sql": SELECT}

        seen, executed = [], []

        def record(action: Action) -> str:
            """Record the enum's SQL without executing it."""
            executed.append(action.value["sql"])
            return action.value["sql"]

        def policy(name, args):
            seen.append(args["action"].value["sql"])
            if mutating:
                args["action"].value["sql"] = DELETE
            return []

        capabilities = (
            [example.Hooks(before_tool_execute=example.recusal_before_tool_execute(policy))]
            if via_hooks
            else [example.RecusalGate(policy)]
        )
        agent = example.scripted_agent(
            [("record", {"action": {"sql": SELECT}}, "c1")],
            tools=[record],
            capabilities=capabilities,
        )
        result = agent.run_sync("go")
        outcome, reason = example.responses_of(result)["c1"]
        assert outcome == "failed" and "arguments cannot be detached" in reason
        assert seen == executed == []
        assert Action.READ.value == {"sql": SELECT}

        # A live recorder and a valid schema: the same proposal runs without the gate.
        example.scripted_agent(
            [("record", {"action": {"sql": SELECT}}, "c1")], tools=[record]
        ).run_sync("go")
        assert executed == [SELECT]

    @pytest.mark.parametrize("via_hooks", [False, True], ids=["capability", "hooks"])
    def test_an_immutable_scalar_valued_enum_is_judged_and_run_as_the_same_member(
        self, example, via_hooks
    ):
        from enum import Enum

        class Action(Enum):
            READ = "read"

        seen, executed = [], []

        def record(action: Action) -> str:
            """Record the selected action."""
            executed.append(action)
            return action.value

        def policy(name, args):
            seen.append(args["action"])
            args["action"] = "write"  # assignment changes only the detached dict
            return []

        capabilities = (
            [example.Hooks(before_tool_execute=example.recusal_before_tool_execute(policy))]
            if via_hooks
            else [example.RecusalGate(policy)]
        )
        result = example.scripted_agent(
            [("record", {"action": "read"}, "c1")], tools=[record], capabilities=capabilities
        ).run_sync("go")
        assert seen == executed == [Action.READ]
        assert example.responses_of(result)["c1"] == ("success", "read")

    @pytest.mark.parametrize("via_hooks", [False, True], ids=["capability", "hooks"])
    @pytest.mark.parametrize("kind", ["model", "dataclass"])
    def test_rich_models_and_dataclasses_are_refused_without_projection(
        self, example, via_hooks, kind
    ):
        from dataclasses import dataclass

        from pydantic import BaseModel, model_serializer

        invoked, executed, judged = [], [], []

        class Model(BaseModel):
            sql: str

            def __deepcopy__(self, memo=None):
                invoked.append("copy")
                return self

            @model_serializer
            def serialize(self):
                invoked.append("serialize")
                return {"sql": SELECT}

        @dataclass
        class Data:
            sql: str

            def __deepcopy__(self, memo=None):
                invoked.append("copy")
                return self

        def record(action) -> str:
            """Record SQL without executing it."""
            executed.append(action.sql)
            return action.sql

        record.__annotations__["action"] = Model if kind == "model" else Data

        def policy(name, args):
            judged.append(args)
            return []

        capabilities = (
            [example.Hooks(before_tool_execute=example.recusal_before_tool_execute(policy))]
            if via_hooks
            else [example.RecusalGate(policy)]
        )
        result = example.scripted_agent(
            [("record", {"action": {"sql": DELETE}}, "c1")],
            tools=[record],
            capabilities=capabilities,
        ).run_sync("go")
        outcome, reason = example.responses_of(result)["c1"]
        assert outcome == "failed" and "declare data-only parameters" in reason
        assert judged == executed == invoked == []

    @pytest.mark.parametrize("via_hooks", [False, True], ids=["capability", "hooks"])
    def test_detachment_errors_name_the_class_without_disclosing_exception_text(
        self, example, monkeypatch, via_hooks
    ):
        def fail_detachment(*args):
            raise RuntimeError("PRIVATE_CREDENTIAL_SENTINEL")

        monkeypatch.setattr(example, "_detached", fail_detachment)
        responses, executed = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")], via_hooks=via_hooks
        )
        assert executed == [] and responses["c1"][0] == "failed"
        assert "(RuntimeError)" in responses["c1"][1]
        assert "PRIVATE_CREDENTIAL_SENTINEL" not in responses["c1"][1]

    @pytest.mark.parametrize("bad", [(1, 2), {1, 2}, b"bytes", {1: "value"}, float("nan")])
    def test_non_json_values_refuse_before_the_policy_runs(self, example, bad):
        judged = []
        reason = example._refusal("record", {"value": bad}, lambda *a: judged.append(a))
        assert reason is not None and "(TypeError)" in reason
        assert judged == []

    def test_cyclic_arguments_refuse_and_shared_json_containers_are_detached(self, example):
        cyclic = []
        cyclic.append(cyclic)
        assert "(TypeError)" in example._refusal("record", {"value": cyclic}, lambda *_: [])
        shared = {"tags": ["read"]}
        invocation = {"left": shared, "right": shared}

        def mutate(name, observed):
            observed["left"]["tags"].append("write")
            return []

        assert example._refusal("record", invocation, mutate) is None
        assert shared == {"tags": ["read"]}

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
