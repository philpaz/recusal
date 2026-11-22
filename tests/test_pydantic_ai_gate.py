"""Regression tests for the offline Pydantic AI + Recusal capability example."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pydantic_ai_gate.py"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"
PIN = "pydantic-ai-slim==2.67.0"


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
    # the pin is a root of the hash-locked file the pull request lanes install: the two
    # example jobs and the job that smoke-tests the installed adapter wheel
    assert (
        PIN
        in (ROOT / "requirements" / "adapter-pydantic-ai.txt")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .split()
    )
    assert ci.count("requirements/adapter-pydantic-ai.txt") == 3


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

    @pytest.fixture
    def adapter(self, example):
        """The package module the gate's decision lives in. Loading the example first puts
        the in-tree adapter on the path, so this is the code under review, not an install."""
        import importlib

        module = importlib.import_module("recusal_pydantic_ai._gate")
        assert "adapters" in Path(module.__file__).parts, module.__file__
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
            "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)",
        )

    def test_a_crash_inside_decide_also_fails_closed(self, example, adapter, monkeypatch):
        def crashing_decide(*args, **kwargs):
            raise RuntimeError("decide crashed")

        monkeypatch.setattr(adapter, "decide", crashing_decide)
        responses, executed = example.run_calls([("echo", {"text": "unsafe"}, "c1")])
        assert executed == []
        assert responses["c1"] == (
            "failed",
            "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)",
        )

    def test_non_mapping_arguments_are_refused_by_the_gate_itself(self, example, adapter):
        assert adapter._refusal("run_sql", ["DELETE"], example.sql_scope_policy) == (
            "Recusal refused `run_sql`: arguments are not a mapping"
        )
        assert adapter._refusal("run_sql", "DELETE", example.sql_scope_policy) is not None

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

    def test_the_policy_observes_nested_containers_detached_from_the_invocation(
        self, example, adapter
    ):
        invocation = {"sql": SELECT, "meta": {"tags": ["read"], "limits": [1, 2]}}
        before = {"sql": SELECT, "meta": {"tags": ["read"], "limits": [1, 2]}}
        observed = []

        def mutating(tool_name, args):
            observed.append(args)
            args["meta"]["tags"].append("write")
            args["meta"]["limits"][0] = 999
            args["sql"] = DELETE
            return []

        assert adapter._refusal("run_sql", invocation, mutating) is None
        assert invocation == before, "the invocation is what the tool runs; it is untouched"
        assert observed[0] is not invocation and observed[0]["meta"] is not invocation["meta"]
        assert observed[0]["sql"] == DELETE  # the policy changed its own copy only

    def test_unsupported_objects_are_refused_without_calling_their_copy_method(
        self, example, adapter
    ):
        called = []

        class Undetachable:
            def __deepcopy__(self, memo):
                called.append("copy")
                raise TypeError("PRIVATE_CREDENTIAL_SENTINEL")

        reason = adapter._refusal("run_sql", {"sql": SELECT, "h": Undetachable()}, lambda *_: [])
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
    @pytest.mark.parametrize("kind", ["enum", "int_enum", "str_enum"])
    def test_a_scalar_valued_enum_is_refused_before_policy_or_execution(
        self, example, via_hooks, kind
    ):
        import enum

        base = {"enum": enum.Enum, "int_enum": enum.IntEnum, "str_enum": enum.StrEnum}[kind]
        proposed = 1 if kind == "int_enum" else "read"
        Action = base("Action", {"READ": proposed})

        seen, executed = [], []

        def record(action) -> str:
            """Record the selected action."""
            executed.append(action)
            return str(action.value)

        record.__annotations__["action"] = Action

        def policy(name, args):
            seen.append(args["action"])
            return []

        capabilities = (
            [example.Hooks(before_tool_execute=example.recusal_before_tool_execute(policy))]
            if via_hooks
            else [example.RecusalGate(policy)]
        )
        result = example.scripted_agent(
            [("record", {"action": proposed}, "c1")], tools=[record], capabilities=capabilities
        ).run_sync("go")
        outcome, reason = example.responses_of(result)["c1"]
        assert outcome == "failed"
        assert reason == (
            "Recusal refused `record`: arguments cannot be detached for judging (TypeError); "
            "declare data-only parameters and validate inside the tool body"
        )
        assert seen == executed == []

        # A live recorder and a valid schema: the same proposal runs without the gate.
        example.scripted_agent([("record", {"action": proposed}, "c1")], tools=[record]).run_sync(
            "go"
        )
        assert executed == [Action.READ]

    @pytest.mark.parametrize("via_hooks", [False, True], ids=["capability", "hooks"])
    def test_a_scalar_enum_with_mutable_associated_state_is_refused(self, example, via_hooks):
        """The review's case: the value is a scalar, the member carries a shared dict."""
        from enum import Enum

        class Action(Enum):
            READ = "read"

            def __init__(self, _value):
                self.payload = {"sql": SELECT}

        seen, executed = [], []

        def record(action: Action) -> str:
            """Record the SQL the member carries without executing it."""
            executed.append(action.payload["sql"])
            return action.payload["sql"]

        def policy(name, args):
            seen.append(args["action"].payload["sql"])
            args["action"].payload["sql"] = DELETE  # the alias the refusal exists to deny
            return []

        capabilities = (
            [example.Hooks(before_tool_execute=example.recusal_before_tool_execute(policy))]
            if via_hooks
            else [example.RecusalGate(policy)]
        )
        result = example.scripted_agent(
            [("record", {"action": "read"}, "c1")], tools=[record], capabilities=capabilities
        ).run_sync("go")
        outcome, reason = example.responses_of(result)["c1"]
        assert outcome == "failed" and "arguments cannot be detached" in reason
        assert seen == executed == []
        assert Action.READ.payload == {"sql": SELECT}

        # The associated state is live: ungated, the tool runs on the member's payload.
        example.scripted_agent([("record", {"action": "read"}, "c1")], tools=[record]).run_sync(
            "go"
        )
        assert executed == [SELECT]

    @pytest.mark.parametrize("via_hooks", [False, True], ids=["capability", "hooks"])
    @pytest.mark.parametrize("shape", ["list", "dict"])
    def test_an_enum_nested_in_a_container_is_refused(self, example, via_hooks, shape):
        from enum import Enum

        class Action(Enum):
            READ = "read"

        seen, executed = [], []

        def record(actions) -> str:
            """Record the selected actions."""
            executed.append(actions)
            return "recorded"

        record.__annotations__["actions"] = list[Action] if shape == "list" else dict[str, Action]
        proposed = ["read"] if shape == "list" else {"first": "read"}

        def policy(name, args):
            seen.append(args)
            return []

        capabilities = (
            [example.Hooks(before_tool_execute=example.recusal_before_tool_execute(policy))]
            if via_hooks
            else [example.RecusalGate(policy)]
        )
        result = example.scripted_agent(
            [("record", {"actions": proposed}, "c1")], tools=[record], capabilities=capabilities
        ).run_sync("go")
        outcome, reason = example.responses_of(result)["c1"]
        assert outcome == "failed" and "arguments cannot be detached" in reason
        assert seen == executed == []

        example.scripted_agent([("record", {"actions": proposed}, "c1")], tools=[record]).run_sync(
            "go"
        )
        assert executed == [[Action.READ] if shape == "list" else {"first": Action.READ}]

    def test_no_enum_member_is_ever_returned_by_the_detachment(self, example, adapter):
        from enum import Enum, IntEnum

        class Plain(Enum):
            A = "a"

        class Number(IntEnum):
            ONE = 1

        for value in (Plain.A, Number.ONE, [Plain.A], {"k": Number.ONE}, {"k": [Plain.A]}):
            with pytest.raises(TypeError, match="Enum arguments are unsupported"):
                adapter._detached(value)
        assert adapter._detached({"k": [1, "a", None, True, 1.5]}) == {
            "k": [1, "a", None, True, 1.5]
        }

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
        self, example, adapter, monkeypatch, via_hooks
    ):
        def fail_detachment(*args):
            raise RuntimeError("PRIVATE_CREDENTIAL_SENTINEL")

        monkeypatch.setattr(adapter, "_detached", fail_detachment)
        responses, executed = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")], via_hooks=via_hooks
        )
        assert executed == [] and responses["c1"][0] == "failed"
        assert "(RuntimeError)" in responses["c1"][1]
        assert "PRIVATE_CREDENTIAL_SENTINEL" not in responses["c1"][1]

    @pytest.mark.parametrize("bad", [(1, 2), {1, 2}, b"bytes", {1: "value"}, float("nan")])
    def test_non_json_values_refuse_before_the_policy_runs(self, example, adapter, bad):
        judged = []
        reason = adapter._refusal("record", {"value": bad}, lambda *a: judged.append(a))
        assert reason is not None and "(TypeError)" in reason
        assert judged == []

    def test_cyclic_arguments_refuse_and_shared_json_containers_are_detached(
        self, example, adapter
    ):
        cyclic = []
        cyclic.append(cyclic)
        assert "(TypeError)" in adapter._refusal("record", {"value": cyclic}, lambda *_: [])
        shared = {"tags": ["read"]}
        invocation = {"left": shared, "right": shared}

        def mutate(name, observed):
            observed["left"]["tags"].append("write")
            return []

        assert adapter._refusal("record", invocation, mutate) is None
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

    def test_another_innermost_capability_listed_after_the_gate_is_refused(self, example):
        """Another ``innermost`` capability listed after the gate runs after it, and one that
        rewrote the call ran a DELETE the gate had judged as a SELECT. The gate now reads
        the order the run dispatches in and refuses a call when a capability with a
        ``before_tool_execute`` of the application's comes after it."""
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
        result = example.scripted_agent(select, capabilities=[gate, InnerRewrite()]).run_sync("go")
        assert example.EXECUTED == []
        assert "runs after the gate" in example.responses_of(result)["c1"][1]
        result = example.scripted_agent(select, capabilities=[InnerRewrite(), gate]).run_sync("go")
        assert example.EXECUTED == []  # the gate last: it judges the rewritten call
        assert "Recusal refused" in example.responses_of(result)["c1"][1]

    @pytest.mark.parametrize("form", ["wrapped", "wrapper_rewrites", "function", "plain_wrapper"])
    def test_a_gate_inside_a_wrapper_is_held_to_the_same_order(self, example, form):
        """A ``WrapperCapability`` does not pass the gate's ``innermost`` position on, so a
        hook listed after a wrapped gate ran after it: judged ``force=False``, ran
        ``force=True``. That is refused, as is a wrapper around the gate with a
        ``before_tool_execute`` of its own. A gate a capability function returns keeps its
        position, and a wrapper with nothing after it changes nothing."""
        from pydantic_ai.capabilities import Hooks, WrapperCapability

        ran, seen = [], []

        async def flip(ctx, *, call, tool_def, args):
            return {**args, "force": True}

        class Flipping(WrapperCapability):
            async def before_tool_execute(self, ctx, *, call, tool_def, args):
                args = await super().before_tool_execute(
                    ctx, call=call, tool_def=tool_def, args=args
                )
                return {**args, "force": True}

        gate = example.RecusalGate(self._deny_force(seen))
        capabilities = {
            "wrapped": [WrapperCapability(gate), Hooks(before_tool_execute=flip)],
            "wrapper_rewrites": [Flipping(gate)],
            "function": [lambda ctx: gate, Hooks(before_tool_execute=flip)],
            "plain_wrapper": [Hooks(before_tool_execute=flip), WrapperCapability(gate)],
        }[form]
        agent = example.Agent(
            self._model_calling(example, "wipe", {"table": "t"}),
            tools=[self._wipe(ran)],
            capabilities=capabilities,
        )
        answers = example.responses_of(agent.run_sync("go"))
        assert ran == []
        if form in ("wrapped", "wrapper_rewrites"):
            assert seen == [] and "could change what the gate judged" in answers["c1"][1]
        else:
            assert seen == [("wipe", {"table": "t", "force": True})]
            assert "force refused" in answers["c1"][1]

    def test_capabilities_after_the_gate_that_change_nothing_are_not_refused(
        self, example, adapter
    ):
        """Only a ``before_tool_execute`` of the application's counts: a ``Hooks`` with other
        hooks, a wrapper that passes the hook on and Pydantic AI's own capabilities do not."""
        from pydantic_ai.capabilities import (
            AbstractCapability,
            CombinedCapability,
            Hooks,
            Instrumentation,
            WrapperCapability,
        )

        async def after(ctx, *, call, tool_def, args, result):
            return result

        class Quiet(AbstractCapability):
            pass

        class Unreadable(Hooks):
            def _get(self, key):
                raise RuntimeError("no registry")

        gate = example.RecusalGate(lambda n, a: [])
        harmless = [Hooks(after_tool_execute=after), WrapperCapability(Quiet()), Instrumentation()]
        assert adapter._rewriter_after(CombinedCapability([gate, *harmless]), gate) is None
        hook = "before_tool_execute"
        assert [adapter._changes_arguments(each, gate, hook) for each in harmless] == [None] * 3
        assert adapter._rewriter_after(gate, gate) is None
        assert adapter._rewriter_after(None, gate) is None
        assert adapter._changes_arguments(Unreadable(), gate, hook) == "Hooks"
        other = example.RecusalGate(lambda n, a: [])
        assert adapter._rewriter_after(CombinedCapability([gate, other]), gate) is None
        assert "place is unknown" in adapter._rewriter_after(CombinedCapability([Quiet()]), gate)
        assert adapter._deferred_handlers_elsewhere(CombinedCapability([Unreadable()])) == ["Hooks"]

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

    # --- what ran outside the tool hook (adapter review of 2026-11-19) -----------------------

    @staticmethod
    def _model_calling(example, name, args):
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import FunctionModel

        turns = {"n": 0}

        def fn(messages, info):
            turns["n"] += 1
            if turns["n"] == 1:
                part = ToolCallPart(tool_name=name, args=args, tool_call_id="c1")
                return ModelResponse(parts=[part])
            return ModelResponse(parts=[TextPart(content="done")])

        return FunctionModel(fn)

    @staticmethod
    def _deny_force(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append((name, dict(args)))
            if args.get("force") or args.get("output", {}) == "force":
                return [Finding.fail("no_force", severity="CRITICAL", message="force refused")]
            return []

        return policy

    def test_an_output_function_is_judged_before_it_runs(self, example):
        """``output_type=fn`` runs the application's function with values the model chose;
        it ran with no policy asked. It is now judged under the function's name, and a
        refusal keeps it from running."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        ran, seen = [], []

        def publish(table: str, force: bool = False) -> str:
            """Publish."""
            ran.append((table, force))
            return "published"

        agent = example.Agent(
            self._model_calling(example, "final_result", {"table": "t", "force": True}),
            output_type=publish,
            capabilities=[example.RecusalGate(self._deny_force(seen))],
        )
        with pytest.raises(UnexpectedModelBehavior):
            agent.run_sync("go")
        assert ran == [] and ("publish", {"table": "t", "force": True}) in seen

        # an allowed call runs, judged with the arguments the function receives
        ran.clear(), seen.clear()
        agent = example.Agent(
            self._model_calling(example, "final_result", {"table": "t"}),
            output_type=publish,
            capabilities=[example.RecusalGate(self._deny_force(seen))],
        )
        assert agent.run_sync("go").output == "published"
        assert seen == [("publish", {"table": "t", "force": False})] and ran == [("t", False)]

    def test_an_output_function_whose_name_is_unknown_is_refused(self, example):
        """Pydantic AI leaves ``function_name`` as ``None`` when a union processor picks the
        function by the output's type; the gate cannot say what would run, so it refuses
        before any policy runs."""
        import asyncio

        from pydantic_ai.exceptions import ModelRetry
        from pydantic_ai.output import OutputContext

        seen = []
        gate = example.RecusalGate(lambda name, args: seen.append(name) or [])
        context = OutputContext(mode="tool", output_type=None, object_def=None, has_function=True)
        with pytest.raises(ModelRetry, match="the output function that would run is not known"):
            asyncio.run(gate.before_output_process(None, output_context=context, output={"a": 1}))
        assert seen == []
        # and plain structured output, with no function to run, is not judged
        plain = OutputContext(mode="tool", output_type=None, object_def=None, has_function=False)
        assert asyncio.run(gate.before_output_process(None, output_context=plain, output=5)) == 5
        assert seen == []

    def test_a_wrapper_toolset_that_runs_calls_itself_is_refused(self, example):
        """A ``WrapperToolset`` whose ``call_tool`` rewrote the arguments ran after the gate
        judged them: judged ``force=False``, ran ``force=True``. Every call of a run with
        such a wrapper in its toolsets is now refused; Pydantic AI's own wrappers pass."""
        from pydantic_ai import FunctionToolset
        from pydantic_ai.toolsets.prefixed import PrefixedToolset
        from pydantic_ai.toolsets.wrapper import WrapperToolset

        ran, seen = [], []

        def wipe(table: str, force: bool = False) -> str:
            """Wipe."""
            ran.append((table, force))
            return "wiped"

        class Forcing(WrapperToolset):
            async def call_tool(self, name, tool_args, ctx, tool):
                return await super().call_tool(name, {**tool_args, "force": True}, ctx, tool)

        agent = example.Agent(
            self._model_calling(example, "wipe", {"table": "t"}),
            toolsets=[Forcing(FunctionToolset([wipe]))],
            capabilities=[example.RecusalGate(self._deny_force(seen))],
        )
        answers = example.responses_of(agent.run_sync("go"))
        assert ran == [] and seen == []
        assert "runs their calls itself" in answers["c1"][1]

        # Pydantic AI's own wrappers route the call and are not refused
        agent = example.Agent(
            self._model_calling(example, "db_wipe", {"table": "t"}),
            toolsets=[PrefixedToolset(FunctionToolset([wipe]), "db")],
            capabilities=[example.RecusalGate(self._deny_force(seen))],
        )
        agent.run_sync("go")
        assert ran == [("t", False)] and seen == [("db_wipe", {"table": "t", "force": False})]

    def test_is_gated_reads_the_capabilities_an_override_puts_in_effect(self, example):
        """``agent.override(spec={"capabilities": [...]})`` replaces the agent's capabilities,
        the gate with them, and ``is_gated`` said ``True`` while the tool ran unjudged."""
        agent = example.scripted_agent(
            [("run_sql", {"sql": "SELECT 1"}, "c1")],
            capabilities=[example.RecusalGate(example.sql_scope_policy)],
        )
        assert example.is_gated(agent)
        with agent.override(spec={"capabilities": ["Thinking"]}):
            assert not example.is_gated(agent)
        assert example.is_gated(agent)

    def test_a_call_handed_to_the_application_is_judged_first(self, example):
        """An ``ExternalToolset`` call is returned to the application to run, and was handed
        on with no policy asked. A refused one is now answered with the refusal; an allowed
        one is handed on with the arguments judged."""
        from pydantic_ai import DeferredToolRequests
        from pydantic_ai.tools import ToolDefinition
        from pydantic_ai.toolsets.external import ExternalToolset

        schema = {"type": "object", "properties": {"table": {"type": "string"}}}
        external = ExternalToolset(
            [ToolDefinition(name="wipe_remote", parameters_json_schema=schema)]
        )
        for args, refused in (({"table": "t", "force": True}, True), ({"table": "t"}, False)):
            seen = []
            agent = example.Agent(
                self._model_calling(example, "wipe_remote", args),
                toolsets=[external],
                output_type=[str, DeferredToolRequests],
                capabilities=[example.RecusalGate(self._deny_force(seen))],
            )
            output = agent.run_sync("go").output
            assert seen == [("wipe_remote", args)]
            if refused:
                assert output == "done"
            else:
                assert isinstance(output, DeferredToolRequests)
                assert [call.args_as_dict() for call in output.calls] == [args]

    # --- first review round of the fix (2026-11-21) -------------------------------------------

    @staticmethod
    def _wipe(ran):
        def wipe(table: str, force: bool = False) -> str:
            """Wipe."""
            ran.append((table, force))
            return "wiped"

        return wipe

    @staticmethod
    def _forcing():
        from pydantic_ai.toolsets.wrapper import WrapperToolset

        class Forcing(WrapperToolset):
            async def call_tool(self, name, tool_args, ctx, tool):
                return await super().call_tool(name, {**tool_args, "force": True}, ctx, tool)

        return Forcing

    def test_a_choices_action_is_judged_before_it_runs(self, example):
        """A ``Choices`` output whose value is callable runs that action when the model
        picks it, and ran with no policy asked. Every final pick is judged as ``choice``."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior
        from pydantic_ai.output import Choice, Choices

        from recusal import Finding

        ran, seen = [], []

        def drop_all() -> str:
            ran.append("drop_all")
            return "dropped"

        def policy(name, args):
            seen.append((name, dict(args)))
            if name == "choice" and args["choice"] == "drop":
                return [Finding.fail("no_drop", severity="CRITICAL", message="drop refused")]
            return []

        def agent(pick, gate_policy):
            choices = Choices(
                {
                    "noop": Choice(description="nothing", value="nothing"),
                    "drop": Choice(description="drop", value=drop_all),
                }
            )
            return example.Agent(
                self._model_calling(example, "final_result", {"response": pick}),
                output_type=choices,
                capabilities=[example.RecusalGate(gate_policy)],
            )

        with pytest.raises(UnexpectedModelBehavior):
            agent("drop", policy).run_sync("go")
        assert ran == [] and seen[0] == ("choice", {"choice": "drop", "set": "Choices"})

        ran.clear(), seen.clear()
        assert (
            agent("drop", lambda name, args: seen.append((name, dict(args))) or [])
            .run_sync("go")
            .output
            == "dropped"
        )
        assert ran == ["drop_all"] and seen == [("choice", {"choice": "drop", "set": "Choices"})]

    def test_a_structured_dict_output_is_not_taken_for_a_choice(self, example, adapter):
        from pydantic_ai import StructuredDict
        from pydantic_ai.output import Choices

        shape = StructuredDict({"type": "object", "properties": {"a": {"type": "string"}}})
        assert not adapter._is_choices(shape)
        assert not adapter._is_choices(str)
        assert adapter._is_choices(Choices(["a", "b"]))

    @pytest.mark.parametrize("route", ["capability", "dynamic", "process_tool_call"])
    def test_a_rewriting_wrapper_the_gate_did_not_assemble_is_refused(self, example, route):
        """A wrapper another capability adds (``get_wrapper_toolset``) sits outside the
        gate's own, and one a dynamic toolset returns is built after the run is assembled:
        both rewrote the arguments after the gate judged them. The gate now reads the run's
        toolset at the time of the call. A toolset with a ``process_tool_call`` function
        (an MCP server's) runs its calls itself the same way."""
        from pydantic_ai import FunctionToolset
        from pydantic_ai.capabilities import AbstractCapability

        ran, seen = [], []
        wipe, forcing = self._wipe(ran), self._forcing()
        gate = example.RecusalGate(self._deny_force(seen))
        model = self._model_calling(example, "wipe", {"table": "t"})
        if route == "capability":

            class Wrapping(AbstractCapability):
                def get_wrapper_toolset(self, toolset):
                    return forcing(toolset)

            agent = example.Agent(model, tools=[wipe], capabilities=[Wrapping(), gate])
        elif route == "dynamic":
            agent = example.Agent(model, capabilities=[gate])

            @agent.toolset
            def dynamic(ctx):
                return forcing(FunctionToolset([wipe]))

        else:
            toolset = FunctionToolset([wipe])
            toolset.process_tool_call = lambda ctx, call_tool, name, args: call_tool(name, args)
            agent = example.Agent(model, toolsets=[toolset], capabilities=[gate])
        answers = example.responses_of(agent.run_sync("go"))
        assert ran == [] and seen == []
        expected = (
            "process_tool_call" if route == "process_tool_call" else "runs their calls itself"
        )
        assert expected in answers["c1"][1]

    @staticmethod
    def _external(name="wipe_remote"):
        from pydantic_ai.tools import ToolDefinition
        from pydantic_ai.toolsets.external import ExternalToolset

        schema = {"type": "object", "properties": {"table": {"type": "string"}}}
        return ExternalToolset([ToolDefinition(name=name, parameters_json_schema=schema)])

    @pytest.mark.parametrize("how", ["capability", "hooks", "wrapped", "per_run"])
    def test_another_capability_that_answers_deferred_calls_is_refused(self, example, how):
        """Pydantic AI offers deferred calls to each capability in turn, the innermost gate
        last, so ``HandleDeferredToolCalls`` executed an external call no policy had judged.
        A run with such a capability is refused before any request is sent, wherever the
        gate is listed, and the error says where the handler goes instead."""
        from pydantic_ai import DeferredToolRequests, DeferredToolResults
        from pydantic_ai.capabilities import HandleDeferredToolCalls, Hooks, WrapperCapability
        from pydantic_ai.exceptions import UserError

        ran, seen = [], []

        def handler(ctx, requests):
            results = DeferredToolResults()
            for call in requests.calls:
                ran.append(call.args_as_dict())
                results.calls[call.tool_call_id] = "wiped"
            return results

        async def hook(ctx, *, requests):
            return handler(ctx, requests)

        other = {
            "capability": HandleDeferredToolCalls(handler),
            "hooks": Hooks(deferred_tool_calls=hook),
            "wrapped": WrapperCapability(HandleDeferredToolCalls(handler)),
            "per_run": HandleDeferredToolCalls(handler),
        }[how]
        gate = example.RecusalGate(self._deny_force(seen))
        for listed in ([other, gate], [gate, other]):
            on_agent = [gate] if how == "per_run" else listed
            agent = example.Agent(
                self._model_calling(example, "wipe_remote", {"table": "t", "force": True}),
                toolsets=[self._external()],
                output_type=[str, DeferredToolRequests],
                capabilities=on_agent,
            )
            per_run = {"capabilities": [other]} if how == "per_run" else {}
            with pytest.raises(UserError, match=r"deferred=handler") as refused:
                agent.run_sync("go", **per_run)
            assert "answers deferred tool calls" in str(refused.value)
            assert ran == [] and seen == []

    def test_capabilities_that_answer_no_deferred_calls_are_not_refused(self, example):
        """A ``Hooks`` with no deferred hook, a capability function and a wrapper around
        either pass the question on and answer nothing; an ordinary run with them is judged
        once, with the validated arguments, as without them."""
        from pydantic_ai.capabilities import Hooks, WrapperCapability

        ran, seen = [], []
        agent = example.Agent(
            self._model_calling(example, "wipe", {"table": "t"}),
            tools=[self._wipe(ran)],
            capabilities=[
                Hooks(),
                lambda ctx: Hooks(),
                WrapperCapability(Hooks()),
                example.RecusalGate(self._deny_force(seen)),
            ],
        )
        agent.run_sync("go")
        assert seen == [("wipe", {"table": "t", "force": False})] and ran == [("t", False)]

    def test_the_gate_judges_deferred_calls_then_hands_the_allowed_ones_to_its_handler(
        self, example
    ):
        """``RecusalGate(policy, deferred=handler)``: the handler is the application's, called
        by the gate after it judged. A refused call never reaches it, and a result it
        returns for one anyway is dropped."""
        from pydantic_ai import DeferredToolRequests, DeferredToolResults
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import FunctionModel

        ran, seen, turns = [], [], {"n": 0}

        def model(messages, info):
            turns["n"] += 1
            if turns["n"] > 1:
                return ModelResponse(parts=[TextPart(content="done")])
            return ModelResponse(
                parts=[
                    ToolCallPart("wipe_remote", {"table": "a"}, tool_call_id="c1"),
                    ToolCallPart("wipe_remote", {"table": "b", "force": True}, tool_call_id="c2"),
                ]
            )

        async def handler(ctx, requests):
            results = DeferredToolResults()
            for call in requests.calls:
                ran.append(call.args_as_dict())
                results.calls[call.tool_call_id] = "wiped"
            results.calls["c2"] = "the handler does not get to answer a refused call"
            return results

        agent = example.Agent(
            FunctionModel(model),
            toolsets=[self._external()],
            output_type=[str, DeferredToolRequests],
            capabilities=[example.RecusalGate(self._deny_force(seen), deferred=handler)],
        )
        answers = example.responses_of(agent.run_sync("go"))
        assert seen == [
            ("wipe_remote", {"table": "a"}),
            ("wipe_remote", {"table": "b", "force": True}),
        ]
        assert ran == [{"table": "a"}]
        assert answers["c1"][1] == "wiped" and "Recusal refused" in answers["c2"][1]

    def test_an_approval_from_the_gates_handler_is_judged_once_when_the_call_runs(self, example):
        """A call waiting for approval is not judged in the deferred hook: it is judged when
        it runs, with the arguments it runs with, the approver's overrides included."""
        from pydantic_ai import DeferredToolRequests, DeferredToolResults, Tool, ToolApproved

        for override, refused in ((None, False), ({"table": "t", "force": True}, True)):
            ran, seen = [], []

            def approve(ctx, requests, override=override):
                results = DeferredToolResults()
                for call in requests.approvals:
                    results.approvals[call.tool_call_id] = ToolApproved(override_args=override)
                return results

            agent = example.Agent(
                self._model_calling(example, "wipe", {"table": "t"}),
                tools=[Tool(self._wipe(ran), requires_approval=True)],
                output_type=[str, DeferredToolRequests],
                capabilities=[example.RecusalGate(self._deny_force(seen), deferred=approve)],
            )
            agent.run_sync("go")
            assert seen == [("wipe", {"table": "t", "force": refused})]
            assert ran == ([] if refused else [("t", False)])

    def test_a_run_resumed_from_history_is_refused_before_another_handler_runs(self, example):
        """A run resumed from a history that ends in a tool call executes it before any
        request to the model, so the refusal made before each request came after another
        capability's handler had run the call. It is also made before the run starts."""
        from pydantic_ai import DeferredToolRequests, DeferredToolResults
        from pydantic_ai.capabilities import HandleDeferredToolCalls
        from pydantic_ai.exceptions import UserError
        from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart

        ran, seen = [], []

        def handler(ctx, requests):
            results = DeferredToolResults()
            for call in requests.calls:
                ran.append(call.args_as_dict())
                results.calls[call.tool_call_id] = "wiped"
            return results

        history = [
            ModelRequest(parts=[UserPromptPart(content="go")]),
            ModelResponse(
                parts=[
                    ToolCallPart("wipe_remote", {"table": "t", "force": True}, tool_call_id="c1")
                ]
            ),
        ]
        agent = example.Agent(
            self._model_calling(example, "nothing", {}),
            toolsets=[self._external()],
            output_type=[str, DeferredToolRequests],
            capabilities=[
                HandleDeferredToolCalls(handler),
                example.RecusalGate(self._deny_force(seen)),
            ],
        )
        with pytest.raises(UserError, match="Recusal refused the run"):
            agent.run_sync(message_history=history)
        assert ran == [] and seen == []

    def test_the_gates_handler_is_not_called_with_nothing_and_may_decline(self, example):
        """When every call was refused there is nothing left to hand over, so the handler is
        not called; and a handler that returns ``None`` declines, as with Pydantic AI's own."""
        import asyncio

        from pydantic_ai import DeferredToolRequests
        from pydantic_ai.messages import ToolCallPart

        called = []

        def handler(ctx, requests):
            called.append([call.tool_call_id for call in requests.calls])
            return None

        gate = example.RecusalGate(self._deny_force([]), deferred=handler)
        refused = DeferredToolRequests(
            calls=[ToolCallPart("wipe_remote", {"force": True}, tool_call_id="c1")]
        )
        results = asyncio.run(gate.handle_deferred_tool_calls(None, requests=refused))
        assert list(results.calls) == ["c1"] and called == []
        allowed = DeferredToolRequests(
            calls=[ToolCallPart("wipe_remote", {"table": "t"}, tool_call_id="c2")]
        )
        assert asyncio.run(gate.handle_deferred_tool_calls(None, requests=allowed)) is None
        assert called == [["c2"]]

    def test_a_realtime_session_is_refused(self, example):
        """A realtime session calls none of the request and output hooks the gate judges in."""
        import asyncio
        from types import SimpleNamespace

        from pydantic_ai.exceptions import UserError

        gate = example.RecusalGate(lambda n, a: [])
        with pytest.raises(UserError, match="realtime session"):
            asyncio.run(gate.before_run(SimpleNamespace(realtime=True, root_capability=None)))
        assert (
            asyncio.run(gate.before_run(SimpleNamespace(realtime=False, root_capability=None)))
            is None
        )

    def test_a_deferred_call_whose_arguments_are_not_an_object_is_refused(self, example):
        import asyncio

        from pydantic_ai import DeferredToolRequests
        from pydantic_ai.exceptions import ToolFailed
        from pydantic_ai.messages import ToolCallPart

        seen = []
        gate = example.RecusalGate(lambda n, a: seen.append(n) or [])
        for bad in ("not json", "[1, 2]"):
            requests = DeferredToolRequests(
                calls=[ToolCallPart("wipe_remote", bad, tool_call_id="c1")]
            )
            results = asyncio.run(gate.handle_deferred_tool_calls(None, requests=requests))
            assert isinstance(results.calls["c1"], ToolFailed) and seen == []

    def test_a_gate_cannot_be_loaded_on_demand_and_one_in_such_a_group_does_not_count(
        self, example
    ):
        """Pydantic AI runs no hook of a capability that is not loaded: a gate marked
        ``defer_loading`` judged nothing while ``is_gated`` said ``True``."""
        from pydantic_ai.capabilities import CombinedCapability
        from pydantic_ai.exceptions import UserError

        policy = lambda name, args: []  # noqa: E731
        with pytest.raises(UserError, match="cannot be loaded on demand"):
            example.RecusalGate(policy, id="gate", defer_loading=True, description="gate")
        group = CombinedCapability(
            [example.RecusalGate(policy)], id="safety", defer_loading=True, description="safety"
        )
        model = self._model_calling(example, "nothing", {})
        assert not example.is_gated(example.Agent(model, capabilities=[group]))
        plain = CombinedCapability([example.RecusalGate(policy)])
        assert example.is_gated(example.Agent(model, capabilities=[plain]))

    @pytest.mark.parametrize("shape", ["override", "container", "hidden", "instance"])
    def test_an_application_toolset_that_runs_other_tools_calls_is_refused(self, example, shape):
        """The toolset check is not a list of wrapper classes: any toolset whose
        ``call_tool`` is the application's is refused when it holds other toolsets (a
        pass-through container would hide a rewriting one) or replaces the ``call_tool`` of
        the Pydantic AI toolset it extends."""
        from pydantic_ai import FunctionToolset
        from pydantic_ai.toolsets.abstract import AbstractToolset

        ran, seen = [], []
        wipe, forcing = self._wipe(ran), self._forcing()

        class Overriding(FunctionToolset):
            async def call_tool(self, name, tool_args, ctx, tool):
                return await super().call_tool(name, {**tool_args, "force": True}, ctx, tool)

        class Container(AbstractToolset):
            def __init__(self, inner, rewrite):
                self.inner, self.rewrite = inner, rewrite

            @property
            def id(self):
                return None

            async def get_tools(self, ctx):
                return await self.inner.get_tools(ctx)

            async def call_tool(self, name, tool_args, ctx, tool):
                if self.rewrite:
                    tool_args = {**tool_args, "force": True}
                return await self.inner.call_tool(name, tool_args, ctx, tool)

        if shape == "override":
            toolset = Overriding([wipe])
        elif shape == "container":
            toolset = Container(FunctionToolset([wipe]), rewrite=True)
        elif shape == "hidden":
            toolset = Container(forcing(FunctionToolset([wipe])), rewrite=False)
        else:
            toolset = FunctionToolset([wipe])
            stock = toolset.call_tool

            async def call_tool(name, tool_args, ctx, tool):
                return await stock(name, {**tool_args, "force": True}, ctx, tool)

            toolset.call_tool = call_tool
        agent = example.Agent(
            self._model_calling(example, "wipe", {"table": "t"}),
            toolsets=[toolset],
            capabilities=[example.RecusalGate(self._deny_force(seen))],
        )
        answers = example.responses_of(agent.run_sync("go"))
        assert ran == [] and seen == []
        assert "could change the arguments after the gate judged them" in answers["c1"][1]

    @pytest.mark.parametrize(
        "shape", ["tuples", "dict_of_lists", "records", "slots", "dict", "set", "nested_tuple"]
    )
    def test_a_container_is_found_however_it_holds_its_toolsets(self, example, adapter, shape):
        """A container written from scratch keeps its toolsets however it likes: in pairs, in
        a dict of lists, in small records, in ``__slots__``. The check looks through all of
        these, several levels down, so none of them hides a toolset it runs calls for."""
        from types import SimpleNamespace

        from pydantic_ai import FunctionToolset
        from pydantic_ai.toolsets.abstract import AbstractToolset

        ran, seen = [], []
        inner = FunctionToolset([self._wipe(ran)])

        class Container(AbstractToolset):
            def __init__(self, held):
                self.held = held

            @property
            def id(self):
                return None

            async def get_tools(self, ctx):
                return await inner.get_tools(ctx)

            async def call_tool(self, name, tool_args, ctx, tool):
                return await inner.call_tool(name, {**tool_args, "force": True}, ctx, tool)

        class Slotted(Container):
            __slots__ = ("kept",)

            def __init__(self, held):
                self.kept = held

        held = {
            "tuples": [("db", inner)],
            "dict_of_lists": {"db": [inner]},
            "records": [SimpleNamespace(prefix="db", toolset=inner)],
            "slots": inner,
            "dict": {"db": inner},
            "set": {inner},
            "nested_tuple": (("db", (inner,)),),
        }[shape]
        toolset = Slotted(held) if shape == "slots" else Container(held)
        assert adapter._held_toolsets(toolset) == ([inner], True)
        agent = example.Agent(
            self._model_calling(example, "wipe", {"table": "t"}),
            toolsets=[toolset],
            capabilities=[example.RecusalGate(self._deny_force(seen))],
        )
        answers = example.responses_of(agent.run_sync("go"))
        assert ran == [] and seen == []
        assert "runs their calls itself" in answers["c1"][1]

    def test_ordinary_toolsets_of_the_applications_own_are_not_refused(self, example, adapter):
        """A subclass of a Pydantic AI wrapper that leaves ``call_tool`` alone runs as the
        stock one does. A toolset written from scratch that holds no other toolset is the
        tool itself. A toolset reached twice, or one nested past the limit, ends the walk."""
        from pydantic_ai import FunctionToolset
        from pydantic_ai.toolsets.abstract import AbstractToolset
        from pydantic_ai.toolsets.combined import CombinedToolset
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        ran, seen = [], []

        class MyPrefixed(PrefixedToolset):
            pass

        agent = example.Agent(
            self._model_calling(example, "db_wipe", {"table": "t"}),
            toolsets=[MyPrefixed(FunctionToolset([self._wipe(ran)]), "db")],
            capabilities=[example.RecusalGate(self._deny_force(seen))],
        )
        agent.run_sync("go")
        assert ran == [("t", False)] and seen == [("db_wipe", {"table": "t", "force": False})]

        class Leaf(AbstractToolset):
            @property
            def id(self):
                return None

            async def get_tools(self, ctx):
                return {}

            async def call_tool(self, name, tool_args, ctx, tool):
                return "ran"

        leaf, plain = Leaf(), FunctionToolset([])
        assert adapter._rewriting_wrapper(leaf) is None

        class Elsewhere:
            __module__ = "pydantic_ai_extras.toolsets"

        assert adapter._stock(PrefixedToolset) and not adapter._stock(Elsewhere)
        assert adapter._rewriting_wrapper(CombinedToolset([plain, plain, leaf])) is None
        deep = plain
        for _ in range(40):
            deep = PrefixedToolset(deep, "p")
        assert "too deep" in adapter._rewriting_wrapper(deep)

    def test_the_toolsets_are_also_checked_when_the_run_is_assembled(self, example, adapter):
        """Where the run context carries no tool manager to read the toolset from at the
        time of the call, the mark set when the run was assembled still refuses it."""
        import asyncio
        from types import SimpleNamespace

        from pydantic_ai import FunctionToolset
        from pydantic_ai.exceptions import ToolFailed
        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai.tools import ToolDefinition
        from pydantic_ai.toolsets.set_metadata import SetMetadataToolset

        seen = []
        gate = example.RecusalGate(lambda n, a: seen.append(n) or [])
        plain = FunctionToolset([self._wipe([])])
        assert gate.get_wrapper_toolset(plain) is None
        marked = gate.get_wrapper_toolset(self._forcing()(plain))
        assert isinstance(marked, SetMetadataToolset)
        (reason,) = marked.metadata.values()
        call = ToolCallPart(tool_name="wipe", args={"table": "t"}, tool_call_id="c1")
        tool_def = ToolDefinition(name="wipe", metadata=dict(marked.metadata))
        with pytest.raises(ToolFailed, match="runs their calls itself"):
            asyncio.run(
                gate.before_tool_execute(
                    SimpleNamespace(tool_manager=None),
                    call=call,
                    tool_def=tool_def,
                    args={"table": "t"},
                )
            )
        assert seen == [] and "Forcing" in reason

    def test_a_partial_choice_runs_nothing_and_is_not_judged(self, example, adapter):
        import asyncio
        from types import SimpleNamespace

        from pydantic_ai.output import Choices, OutputContext

        seen = []
        gate = example.RecusalGate(lambda n, a: seen.append(n) or [])
        context = OutputContext(
            mode="tool", output_type=Choices(["a", "b"]), object_def=None, has_function=False
        )
        partial = SimpleNamespace(partial_output=True)
        assert (
            asyncio.run(gate.before_output_process(partial, output_context=context, output="a"))
            == "a"
        )
        assert seen == []

        class Elsewhere:
            @classmethod
            def __get_pydantic_core_schema__(cls, source_type, handler):
                raise NotImplementedError

        assert not adapter._is_choices(Elsewhere)

    def test_a_tool_the_provider_runs_is_refused_with_the_request(self, example):
        """Web search, code execution and the like run on the provider with arguments the
        model chose, where no hook sees them; a gated agent offering one is refused before
        the request is sent. Provider-side tool search only finds tools and is allowed."""
        import asyncio
        from types import SimpleNamespace

        from pydantic_ai import CodeExecutionTool
        from pydantic_ai.capabilities import NativeTool
        from pydantic_ai.exceptions import UserError
        from pydantic_ai.native_tools._tool_search import ToolSearchTool

        seen = []
        gate = example.RecusalGate(lambda n, a: seen.append(n) or [])
        agent = example.Agent(
            self._model_calling(example, "nothing", {}),
            capabilities=[NativeTool(CodeExecutionTool()), gate],
        )
        with pytest.raises(UserError, match="tools the provider runs .CodeExecutionTool."):
            agent.run_sync("go")
        assert seen == []
        request = SimpleNamespace(
            model=None,
            model_settings=None,
            model_request_parameters=SimpleNamespace(native_tools=[ToolSearchTool()]),
        )
        assert asyncio.run(gate.prepare_model_request(None, request)) is request

    def test_a_provider_tool_the_model_will_not_be_sent_is_not_refused(self, example):
        """A native tool with a local version, on a model whose provider runs none: Pydantic
        AI sends the local tool only, and its calls are judged like any other. The gate asks
        the model what the request will carry instead of reading what was asked for."""
        from pydantic_ai.capabilities import NativeOrLocalTool
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.native_tools import WebSearchTool

        ran, seen, sent = [], [], []

        def local_search(query: str) -> str:
            """Search."""
            ran.append(query)
            return "results"

        class NoNativeModel(FunctionModel):
            @classmethod
            def supported_native_tools(cls):
                return frozenset()

        def fn(messages, info):
            sent.append([type(t).__name__ for t in info.model_request_parameters.native_tools])
            if len(sent) == 1:
                call = ToolCallPart("local_search", {"query": "q"}, tool_call_id="c1")
                return ModelResponse(parts=[call])
            return ModelResponse(parts=[TextPart(content="done")])

        agent = example.Agent(
            NoNativeModel(fn),
            capabilities=[
                NativeOrLocalTool(native=WebSearchTool(), local=local_search),
                example.RecusalGate(lambda n, a: seen.append(n) or []),
            ],
        )
        assert agent.run_sync("go").output == "done"
        assert ran == ["q"] and seen == ["local_search"] and sent == [[], []]

    def test_settings_that_switch_a_provider_tool_on_are_refused(self, example, adapter):
        from types import SimpleNamespace

        def found(settings):
            request = SimpleNamespace(
                model=None,
                model_settings=settings,
                model_request_parameters=SimpleNamespace(native_tools=[]),
            )
            return adapter._provider_run_tools(request)

        assert found(None) == [] and found({"temperature": 0.0, "extra_body": {"user": "u"}}) == []
        assert found({"openai_native_tools": [{"type": "web_search"}]}) == ["`openai_native_tools`"]
        assert found({"extra_body": {"tools": [{"type": "web_search"}]}}) == ["`extra_body.tools`"]
        assert found({"extra_body": {"plugins": [{"id": "web"}]}}) == ["`extra_body.plugins`"]
        for key in ("web_search_options", "mcp_servers", "search_parameters"):
            assert found({"extra_body": {key: {"on": True}}}) == [f"`extra_body.{key}`"]

    def test_a_models_own_settings_and_each_fallback_candidate_are_read(self, example):
        """Settings can sit on the model, not the run, and a ``FallbackModel`` resolves
        nothing itself: each model it may hand the request to merges its own settings and
        drops or swaps native tools when its turn comes. Each is asked."""
        from pydantic_ai.capabilities import NativeOrLocalTool
        from pydantic_ai.exceptions import UserError
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.fallback import FallbackModel
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.native_tools import WebSearchTool

        gate = example.RecusalGate(lambda n, a: [])
        switched_on = {"extra_body": {"tools": [{"type": "web_search"}]}}

        def text(messages, info):
            return ModelResponse(parts=[TextPart(content="done")])

        for model in (
            FunctionModel(text, settings=switched_on),
            FallbackModel(FunctionModel(text), FunctionModel(text, settings=switched_on)),
        ):
            with pytest.raises(UserError, match="extra_body.tools"):
                example.Agent(model, capabilities=[gate]).run_sync("go")

        ran, turns = [], {"n": 0}

        def local_search(query: str) -> str:
            """Search."""
            ran.append(query)
            return "results"

        class NoNativeModel(FunctionModel):
            @classmethod
            def supported_native_tools(cls):
                return frozenset()

        def calls_local(messages, info):
            turns["n"] += 1
            if turns["n"] == 1:
                call = ToolCallPart("local_search", {"query": "q"}, tool_call_id="c1")
                return ModelResponse(parts=[call])
            return ModelResponse(parts=[TextPart(content="done")])

        agent = example.Agent(
            FallbackModel(NoNativeModel(calls_local)),
            capabilities=[NativeOrLocalTool(native=WebSearchTool(), local=local_search), gate],
        )
        assert agent.run_sync("go").output == "done" and ran == ["q"]

    @pytest.mark.parametrize("wrapper", ["NativeOutput", "PromptedOutput"])
    def test_a_native_union_of_output_functions_is_refused_and_says_what_to_use(
        self, example, wrapper
    ):
        """Pydantic AI does not tell the hook which function of a native or prompted union
        it will call, so the gate refuses it, and the functions do not run."""
        import pydantic_ai
        from pydantic_ai.exceptions import UnexpectedModelBehavior
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel

        ran, retries = [], []

        def publish(table: str) -> str:
            ran.append("publish")
            return "published"

        def archive(table: str, force: bool = False) -> str:
            ran.append("archive")
            return "archived"

        def fn(messages, info):
            retries.append(str(messages[-1]))
            body = '{"result": {"kind": "archive", "data": {"table": "t"}}}'
            return ModelResponse(parts=[TextPart(content=body)])

        agent = example.Agent(
            FunctionModel(fn),
            output_type=getattr(pydantic_ai, wrapper)([publish, archive]),
            capabilities=[example.RecusalGate(lambda n, a: [])],
        )
        with pytest.raises(UnexpectedModelBehavior):
            agent.run_sync("go")
        assert ran == [] and "use tool output" in retries[-1]

    def test_an_output_function_taking_a_model_is_refused_like_a_tool_taking_one(self, example):
        """The data-only rule holds for output functions too: a parameter that is a Pydantic
        model is refused before the policy runs, and the function does not run."""
        from pydantic import BaseModel
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        ran, seen = [], []

        class Order(BaseModel):
            sku: str

        def place(order: Order) -> str:
            ran.append(order)
            return "placed"

        agent = example.Agent(
            self._model_calling(example, "final_result", {"sku": "a"}),
            output_type=place,
            capabilities=[example.RecusalGate(lambda n, a: seen.append(n) or [])],
        )
        with pytest.raises(UnexpectedModelBehavior):
            agent.run_sync("go")
        assert ran == [] and seen == []

    def test_a_single_argument_output_function_is_judged_under_output(self, example):
        seen = []

        def shout(text: str) -> str:
            return text.upper()

        agent = example.Agent(
            self._model_calling(example, "final_result", {"text": "hi"}),
            output_type=shout,
            capabilities=[example.RecusalGate(lambda n, a: seen.append((n, dict(a))) or [])],
        )
        assert agent.run_sync("go").output == "HI"
        assert seen == [("shout", {"output": "hi"})]

    # --- fourth review round (2026-11-22) -----------------------------------------------------

    @pytest.mark.parametrize("prefixed", [False, True])
    def test_a_default_the_function_fills_in_itself_is_judged(self, example, prefixed):
        """A tool given its schema by hand (``Tool.from_schema``) is not validated, so a
        parameter the model leaves out reached the hook absent and Python filled the default
        when the function was called: judged ``{"table": "t"}``, ran ``mode="hard"``. The
        gate completes the arguments with the defaults the function would fill, judges
        those, and the function is called with them written out."""
        from pydantic_ai import FunctionToolset, Tool
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        from recusal import Finding

        ran, seen = [], []

        def delete_rows(table: str, mode: str = "hard") -> str:
            ran.append((table, mode))
            return f"deleted ({mode})"

        def policy(name, args):
            seen.append((name, dict(args)))
            if args.get("mode") == "hard":
                return [Finding.fail("no_hard", severity="CRITICAL", message="hard refused")]
            return []

        schema = {
            "type": "object",
            "properties": {"table": {"type": "string"}, "mode": {"type": "string"}},
            "required": ["table"],
        }
        tool = Tool.from_schema(
            delete_rows, name="delete_rows", description="d", json_schema=schema
        )
        name = "db_delete_rows" if prefixed else "delete_rows"
        toolset = FunctionToolset([tool])
        for sent, expected in (
            ({"table": "t"}, {"table": "t", "mode": "hard"}),
            ({"table": "t", "mode": "soft"}, {"table": "t", "mode": "soft"}),
        ):
            ran.clear(), seen.clear()
            agent = example.Agent(
                self._model_calling(example, name, sent),
                toolsets=[PrefixedToolset(toolset, "db") if prefixed else toolset],
                capabilities=[example.RecusalGate(policy)],
            )
            agent.run_sync("go")
            assert seen == [(name, expected)]
            assert ran == ([] if expected["mode"] == "hard" else [("t", "soft")])

    def test_the_hook_form_judges_the_completed_arguments_too(self, example):
        from pydantic_ai import Tool
        from pydantic_ai.capabilities import Hooks

        ran, seen = [], []

        def delete_rows(ctx, table: str, mode: str = "hard") -> str:
            ran.append((table, mode))
            return "deleted"

        schema = {"type": "object", "properties": {"table": {"type": "string"}}}
        tool = Tool.from_schema(
            delete_rows, name="delete_rows", description="d", json_schema=schema, takes_ctx=True
        )
        hook = example.recusal_before_tool_execute(lambda n, a: seen.append((n, dict(a))) or [])
        agent = example.Agent(
            self._model_calling(example, "delete_rows", {"table": "t"}),
            tools=[tool],
            capabilities=[Hooks(before_tool_execute=hook)],
        )
        agent.run_sync("go")
        assert seen == [("delete_rows", {"table": "t", "mode": "hard"})] and ran == [("t", "hard")]

    @staticmethod
    def _ctx_for(function, *, takes_ctx=False, validated=True):
        """A run context whose tool ``t`` runs ``function``, as the tool manager holds it."""
        from types import SimpleNamespace

        def validate_python(value):
            if validated:
                raise ValueError("not an object")
            return value

        schema = SimpleNamespace(
            function=function,
            takes_ctx=takes_ctx,
            validator=SimpleNamespace(validate_python=validate_python),
        )
        bound = SimpleNamespace(__self__=schema)
        record = SimpleNamespace(call_func=None, source_tool=SimpleNamespace(call_func=bound))
        return SimpleNamespace(tool_manager=SimpleNamespace(tools={"t": record}))

    def test_what_is_completed_and_what_is_left_alone(self, example, adapter):
        """Only the outermost callable's own defaults are completed: Python fills those in
        when the function is called. Where there is no local function, nothing is done."""
        import functools
        from types import SimpleNamespace

        def inner(table: str, mode: str = "hard", limit: int = 5) -> str:
            return "ok"

        @functools.wraps(inner)
        def retrying(*args, _retries: int = 3, **kwargs):
            return inner(*args, **kwargs)

        validated = {"table": "x", "mode": "hard", "limit": 1}
        completed, reason = adapter._with_defaults(self._ctx_for(retrying), "t", validated)
        assert reason is None and completed == {**validated, "_retries": 3}
        sent = {"table": "x", "_retries": 9}
        assert adapter._with_defaults(self._ctx_for(retrying), "t", sent) == (sent, None)
        same = {"table": "x", "mode": "soft", "limit": 2}
        assert adapter._with_defaults(self._ctx_for(inner), "t", same) == (same, None)
        assert adapter._with_defaults(self._ctx_for(inner), "other", {}) == ({}, None)
        assert adapter._with_defaults(SimpleNamespace(), "t", {}) == ({}, None)
        assert adapter._with_defaults(self._ctx_for(inner), "t", "text") == ("text", None)
        assert adapter._function_of(self._ctx_for(None), "t") == (None, False, True)

        def with_context(ctx=None, table: str = "x") -> str:
            return "ok"

        completed, _ = adapter._with_defaults(self._ctx_for(with_context, takes_ctx=True), "t", {})
        assert completed == {"table": "x"}
        completed, _ = adapter._with_defaults(self._ctx_for(with_context), "t", {})
        assert completed == {"ctx": None, "table": "x"}

    def test_a_default_that_cannot_be_judged_is_refused_by_name(self, example, adapter):
        """A decorator's own keyword with an object for a default is filled in by Python
        where no policy sees it, and cannot be shown to one; the refusal names it."""
        import functools
        import logging

        log = logging.getLogger("x")

        def inner(table: str) -> str:
            return "ok"

        @functools.wraps(inner)
        def logged(*args, _log=log, **kwargs):
            return inner(*args, **kwargs)

        args = {"table": "x"}
        same, reason = adapter._with_defaults(self._ctx_for(logged), "t", args)
        assert same is args and "parameter `_log`" in reason and "not plain data" in reason

    def test_an_unvalidated_tool_whose_defaults_cannot_be_read_is_refused(self, example, adapter):
        """For a tool given its schema by hand, what the function does with a parameter the
        model left out is all that decides its value. A function taking ``**kwargs`` can
        fill one in in its body, as Pydantic AI's LangChain bridge does, and a wrapper may
        or may not pass on a parameter that has a default further in."""
        import functools

        def inner(table: str, mode: str = "hard") -> str:
            return "ok"

        def proxy(*args, **kwargs):
            return inner(**({"mode": "hard"} | kwargs))

        @functools.wraps(inner)
        def narrow(table):
            return inner(table, mode="soft")

        @functools.wraps(inner)
        def naming(table, mode="soft"):
            return inner(table, mode)

        def by_hand(function):
            return adapter._with_defaults(
                self._ctx_for(function, validated=False), "t", {"table": "x"}
            )

        assert "variadic parameter" in by_hand(proxy)[1]
        assert "hides `mode`" in by_hand(narrow)[1]
        sent = {"table": "x", "mode": "soft"}  # the model sent it: nothing is left to a default
        unvalidated = self._ctx_for(narrow, validated=False)
        assert adapter._with_defaults(unvalidated, "t", sent) == (sent, None)
        assert by_hand(naming) == ({"table": "x", "mode": "soft"}, None)
        assert by_hand(inner) == ({"table": "x", "mode": "hard"}, None)
        # the same functions behind a validated tool are Pydantic AI's to complete
        for function in (proxy, narrow):
            assert adapter._with_defaults(self._ctx_for(function), "t", {"table": "x"})[1] is None
        assert adapter._validates(object()) is True

    def test_a_function_that_cannot_be_inspected_is_refused(self, example, adapter, monkeypatch):
        def tool(table: str, mode: str = "hard") -> str:
            return "ok"

        def unreadable(function, **kwargs):
            raise ValueError("no signature")

        monkeypatch.setattr(adapter, "_reached_parameters", unreadable)
        completed, reason = adapter._with_defaults(self._ctx_for(tool), "t", {"table": "x"})
        assert reason is None and completed == {"table": "x", "mode": "hard"}  # the fallback
        monkeypatch.setattr(adapter.inspect, "signature", unreadable)
        same, reason = adapter._with_defaults(self._ctx_for(tool), "t", {"table": "x"})
        assert same == {"table": "x"} and "cannot be inspected" in reason

    def test_the_completed_arguments_are_what_the_hook_hands_back(self, example):
        """The hook returns the arguments it judged, so the function is called with the
        defaults written out, and a refusal for a function it could not read is a refusal."""
        import asyncio

        from pydantic_ai.exceptions import ToolFailed
        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai.tools import ToolDefinition

        def tool(table: str, mode: str = "hard") -> str:
            return "ok"

        def proxy(**kwargs):
            return "ok"

        seen = []
        gate = example.RecusalGate(lambda n, a: seen.append(dict(a)) or [])
        call = ToolCallPart("t", {"table": "x"}, tool_call_id="c1")

        def run(function, hook):
            ctx = self._ctx_for(function, validated=False)
            ctx.root_capability = None
            ctx.tool_manager.toolset = None
            return asyncio.run(
                hook(ctx, call=call, tool_def=ToolDefinition(name="t"), args={"table": "x"})
            )

        plain = example.recusal_before_tool_execute(lambda n, a: seen.append(dict(a)) or [])
        for hook in (gate.before_tool_execute, plain):
            seen.clear()
            assert run(tool, hook) == {"table": "x", "mode": "hard"}
            assert seen == [{"table": "x", "mode": "hard"}]
            with pytest.raises(ToolFailed, match="variadic parameter"):
                run(proxy, hook)
            assert len(seen) == 1

    def test_tools_from_pydantic_ais_langchain_bridge_are_refused(self, example):
        """``pydantic_ai.ext.langchain`` builds each tool by hand around a function that takes
        ``**kwargs`` and merges the LangChain tool's defaults in its body: the policy was
        shown ``{"table": "t"}`` and the tool ran with ``mode="hard"``. Such a tool is now
        refused."""
        from types import SimpleNamespace

        from pydantic_ai.ext.langchain import tool_from_langchain

        ran, seen = [], []
        langchain_tool = SimpleNamespace(
            name="delete_rows",
            description="d",
            args={"table": {"type": "string"}, "mode": {"type": "string", "default": "hard"}},
            get_input_jsonschema=lambda: {
                "type": "object",
                "properties": {"table": {"type": "string"}, "mode": {"type": "string"}},
            },
            run=lambda kwargs: ran.append(kwargs) or "ok",
        )
        agent = example.Agent(
            self._model_calling(example, "delete_rows", {"table": "t"}),
            tools=[tool_from_langchain(langchain_tool)],
            capabilities=[example.RecusalGate(lambda n, a: seen.append(n) or [])],
        )
        answers = example.responses_of(agent.run_sync("go"))
        assert ran == [] and seen == [] and "variadic parameter" in answers["c1"][1]

    @pytest.mark.parametrize("form", ["innermost", "wrapped_gate", "hooks_subclass"])
    def test_an_output_changed_after_the_gate_is_refused(self, example, form):
        """The order check holds for output functions as for tool calls: a
        ``before_output_process`` of the application's dispatched after the gate rewrote
        the value the output function ran with (judged ``force=False``, ran ``True``)."""
        from dataclasses import dataclass

        from pydantic_ai.capabilities import (
            AbstractCapability,
            CapabilityOrdering,
            Hooks,
            WrapperCapability,
        )
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        ran, seen = [], []

        def publish(table: str, force: bool = False) -> str:
            ran.append((table, force))
            return "published"

        async def flip(ctx, *, output_context, output):
            return {**output, "force": True}

        @dataclass
        class InnerFlip(AbstractCapability):
            def get_ordering(self):
                return CapabilityOrdering(position="innermost")

            async def before_output_process(self, ctx, *, output_context, output):
                return {**output, "force": True}

        class HooksFlip(Hooks):
            async def before_output_process(self, ctx, *, output_context, output):
                return {**output, "force": True}

            def get_ordering(self):
                return CapabilityOrdering(position="innermost")

        gate = example.RecusalGate(self._deny_force(seen))
        capabilities = {
            "innermost": [gate, InnerFlip()],
            "wrapped_gate": [WrapperCapability(gate), Hooks(before_output_process=flip)],
            "hooks_subclass": [gate, HooksFlip()],
        }[form]
        agent = example.Agent(
            self._model_calling(example, "final_result", {"table": "t"}),
            output_type=publish,
            capabilities=capabilities,
        )
        with pytest.raises(UnexpectedModelBehavior):
            agent.run_sync("go")
        assert ran == [] and seen == []

    def test_a_choice_changed_after_the_gate_is_refused(self, example):
        from dataclasses import dataclass

        from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
        from pydantic_ai.exceptions import UnexpectedModelBehavior
        from pydantic_ai.output import Choice, Choices

        ran, seen = [], []

        def drop_all() -> str:
            ran.append("drop_all")
            return "dropped"

        @dataclass
        class Repick(AbstractCapability):
            def get_ordering(self):
                return CapabilityOrdering(position="innermost")

            async def before_output_process(self, ctx, *, output_context, output):
                return "drop"

        choices = Choices({"noop": Choice(value="nothing"), "drop": Choice(value=drop_all)})
        agent = example.Agent(
            self._model_calling(example, "final_result", {"response": "noop"}),
            output_type=choices,
            capabilities=[example.RecusalGate(lambda n, a: seen.append(n) or []), Repick()],
        )
        with pytest.raises(UnexpectedModelBehavior):
            agent.run_sync("go")
        assert ran == [] and seen == []

    def test_a_hooks_subclass_that_writes_the_hook_as_a_method_counts(self, example, adapter):
        """A ``Hooks`` is read by its registry, and a subclass can write a hook as a method
        instead: that one registered nothing and was taken for silent."""
        from pydantic_ai.capabilities import CombinedCapability, Hooks

        class Rewrites(Hooks):
            async def before_tool_execute(self, ctx, *, call, tool_def, args):
                return {**args, "force": True}

        class Answers(Hooks):
            async def handle_deferred_tool_calls(self, ctx, *, requests):
                return None

        gate = example.RecusalGate(lambda n, a: [])
        assert adapter._changes_arguments(Rewrites(), gate, "before_tool_execute") == "Rewrites"
        assert adapter._changes_arguments(Rewrites(), gate, "before_output_process") is None
        assert adapter._deferred_handlers_elsewhere(Answers()) == ["Answers"]
        nested = CombinedCapability([CombinedCapability([Answers()])])
        assert adapter._deferred_handlers_elsewhere(nested) == ["Answers"]

    def test_the_order_is_read_level_by_level(self, example, adapter):
        """A hook bundled before the gate, behind a wrapper or a capability function, runs
        before it and is not refused; one dispatched after the member that holds the gate
        is, through a pass-through wrapper and inside a nested group too."""
        from pydantic_ai.capabilities import (
            AbstractCapability,
            CombinedCapability,
            Hooks,
            WrapperCapability,
        )

        async def before(ctx, *, call, tool_def, args):
            return args

        class Solo(AbstractCapability):
            pass

        gate = example.RecusalGate(lambda n, a: [])
        hooks = Hooks(before_tool_execute=before)
        bundle = WrapperCapability(CombinedCapability([hooks, gate]))
        assert adapter._rewriter_after(CombinedCapability([Solo(), bundle]), gate) is None
        assert adapter._rewriter_after(bundle, gate) is None
        assert adapter._holds(bundle, gate) and not adapter._holds(Solo(), gate)

        wrapped_gate = WrapperCapability(gate)
        later = WrapperCapability(hooks)
        root = CombinedCapability([wrapped_gate, later])
        members = list(root._dispatch)
        assert members.index(wrapped_gate) < members.index(later)
        assert "runs after the gate" in adapter._rewriter_after(root, gate)
        group = CombinedCapability([Solo(), hooks])
        assert adapter._changes_arguments(group, gate, "before_tool_execute") == "Hooks"

        class Rewriting(WrapperCapability):
            async def before_tool_execute(self, ctx, *, call, tool_def, args):
                return {**args, "force": True}

        rewriting = Rewriting(Solo())
        assert adapter._changes_arguments(rewriting, gate, "before_tool_execute") == "Rewriting"
        assert adapter._changes_arguments(rewriting, gate, "before_output_process") is None
        assert "place is unknown" in adapter._rewriter_after(Solo(), gate)
        assert "place is unknown" in adapter._rewriter_after(WrapperCapability(Solo()), gate)

    def test_a_container_too_large_to_search_is_refused_not_passed(self, example, adapter):
        """The search for held toolsets gives up after a fixed number of values. It used to
        report "holds nothing" then, so a container with a large table beside its inner
        toolset passed as a leaf. The search reads the nearest values first, and an
        application toolset it could not finish searching is refused."""
        from pydantic_ai import FunctionToolset
        from pydantic_ai.toolsets.abstract import AbstractToolset

        inner = FunctionToolset([])

        class Router(AbstractToolset):
            def __init__(self, table, inner=None, deep=None):
                self.inner, self.table, self.deep = inner, table, deep

            @property
            def id(self):
                return None

            async def get_tools(self, ctx):
                return {}

            async def call_tool(self, name, tool_args, ctx, tool):
                return None

        big = {f"k{i}": f"v{i}" for i in range(adapter._HELD_LIMIT)}
        held, complete = adapter._held_toolsets(Router(big, inner))
        assert held == [inner] and not complete
        assert "runs their calls itself" in adapter._rewriting_wrapper(Router(big, inner))
        assert "holds more than the gate searches" in adapter._rewriting_wrapper(Router(big))
        assert adapter._rewriting_wrapper(Router({"k": "v"})) is None
        assert adapter._held_toolsets(Router({inner: "as a key"})) == ([inner], True)
        assert adapter._held_toolsets(Router({}, deep=[[[inner]]])) == ([inner], True)
        assert adapter._held_toolsets(Router({}, deep=[[[[inner]]]])) == ([], True)
        stock_holder = FunctionToolset([])
        stock_holder.note = inner
        assert adapter._held_toolsets(Router({}, deep=[stock_holder])) == ([stock_holder], True)
        # an object of Pydantic AI's own is not opened: a toolset that keeps its agent is not
        # taken to hold the agent's toolsets
        agent = example.Agent(self._model_calling(example, "nothing", {}), toolsets=[inner])
        assert adapter._held_toolsets(Router({}, deep=agent)) == ([], True)

    def test_another_deferred_handler_is_refused_before_each_request_too(self, example):
        import asyncio
        from types import SimpleNamespace

        from pydantic_ai.capabilities import CombinedCapability, HandleDeferredToolCalls
        from pydantic_ai.exceptions import UserError

        gate = example.RecusalGate(lambda n, a: [])
        root = CombinedCapability([HandleDeferredToolCalls(lambda ctx, requests: None), gate])
        with pytest.raises(UserError, match="answers deferred tool calls"):
            asyncio.run(gate.prepare_model_request(SimpleNamespace(root_capability=root), None))

    def test_the_policy_is_given_the_name_the_model_called(self, example):
        import asyncio
        from types import SimpleNamespace

        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai.tools import ToolDefinition

        seen = []
        gate = example.RecusalGate(lambda n, a: seen.append(n) or [])
        asyncio.run(
            gate.before_tool_execute(
                SimpleNamespace(tool_manager=None, root_capability=None),
                call=ToolCallPart("db_wipe", {}, tool_call_id="c1"),
                tool_def=ToolDefinition(name="wipe"),
                args={},
            )
        )
        assert seen == ["db_wipe"]

    # --- the script ------------------------------------------------------------------------

    def test_example_runs_end_to_end(self, example):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("tools that ran: 2") == 2
        assert result.stdout.count("Recusal refused `run_sql` [FAIL]") == 2
        assert "RecusalGate:" in result.stdout and "Hooks(before_tool_execute):" in result.stdout
