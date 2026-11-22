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

    @staticmethod
    def _decorators():
        """Wrappers of the shapes the rule tells apart."""
        import functools
        from types import SimpleNamespace

        def forwards(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)

            return wrapper

        def retries(default=3):
            def decorate(function):
                @functools.wraps(function)
                def wrapper(*args, retries=default, **kwargs):
                    return function(*args, **kwargs)

                return wrapper

            return decorate

        return SimpleNamespace(forwards=forwards, retries=retries)

    def test_a_transparent_wrapper_is_read_through(self, example, adapter):
        """What the innermost function runs with can be read from the arguments when every
        wrapper above it either takes only ``*args`` and ``**kwargs``, restates the
        function's parameters with a fixed signature, or forwards and names a parameter that
        has no default on either side. Then the function's own defaults that the arguments
        leave out are added."""
        import functools
        from types import SimpleNamespace

        d = self._decorators()

        def inner(table: str, mode: str = "hard", limit: int = 5) -> str:
            return "ok"

        @functools.wraps(inner)
        def restated(table, mode="hard", limit=5):
            return inner(table, mode, limit)

        @functools.wraps(inner)
        def all_required(table, mode, limit):
            return inner(table, mode, limit)

        @functools.wraps(inner)
        def names_a_required_one(table, **kwargs):
            return inner(table, **kwargs)

        forwarded = d.forwards(inner)

        # each wrapper is held against the innermost function, not the wrapper under it
        @functools.wraps(forwarded)
        def names_above_a_forwarder(table, **kwargs):
            return forwarded(table, **kwargs)

        validated = {"table": "x", "mode": "soft", "limit": 1}
        completed = {"table": "x", "mode": "hard", "limit": 5}
        for function in (
            inner,
            d.forwards(inner),
            d.forwards(d.forwards(d.forwards(inner))),
            restated,
            all_required,
            names_a_required_one,
            names_above_a_forwarder,
            d.forwards(names_above_a_forwarder),
            d.forwards(restated),
        ):
            ctx = self._ctx_for(function)
            assert adapter._with_defaults(ctx, "t", dict(validated)) == (validated, None)
            # arguments that reached the hook without validation: the defaults are added
            assert adapter._with_defaults(ctx, "t", {"table": "x"}) == (completed, None)
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

    def test_a_wrapper_with_a_parameter_of_its_own_is_refused(self, example, adapter):
        """Whether a value a wrapper takes reaches the function under it depends on the
        wrapper's body. A keyword default of the wrapper's own ran unjudged; a keyword the
        wrapper shares with the function let the model send ``refresh=False`` while the
        function ran on its own ``refresh=True``; two wrappers of the same shape hid the
        collision. Each is refused, naming the parameter, wherever it sits in a stack and
        whatever was sent."""
        import functools

        d = self._decorators()

        def inner(table: str, mode: str = "hard") -> str:
            return "ok"

        def purge(table: str, refresh: bool = True, **extra: bool) -> str:
            return "ok"

        @functools.wraps(purge)
        def keeps_refresh(*args, refresh=True, **kwargs):
            return purge(*args, **kwargs)

        @functools.wraps(inner)
        def hides_mode(table):
            return inner(table)

        @functools.wraps(inner)
        def other_default(table, mode="soft"):
            return inner(table, mode)

        @functools.wraps(inner)
        def reordered(mode, table):
            return inner(table, mode)

        @functools.wraps(inner)
        def keyword_only(table, *, mode="hard"):
            return inner(table, mode)

        @functools.wraps(inner)
        def options_named_like_nothing(*args, options="skip", **kwargs):
            return inner(*args, **kwargs)

        @functools.wraps(inner)
        def names_a_defaulted_one(*args, mode="hard", **kwargs):
            return inner(*args, mode=mode, **kwargs)

        # a wrapper that forwards and names a parameter with a default, on either side
        @functools.wraps(inner)
        def both_default(table, mode="hard", **kwargs):
            return inner(table, mode, **kwargs)

        @functools.wraps(inner)
        def function_defaults(table, mode, **kwargs):
            return inner(table, **kwargs)

        def required(table: str, mode: str) -> str:
            return "ok"

        @functools.wraps(required)
        def wrapper_defaults(table, mode="hard", **kwargs):
            return required(table, mode, **kwargs)

        def flag(table: str, force: bool = True) -> str:
            return "ok"

        @functools.wraps(flag)
        def one_for_true(table, force=1):
            return flag(table, force)

        def with_extra(table: str, **extra: bool) -> str:
            return "ok"

        @functools.wraps(with_extra)
        def names_the_variadic(*args, extra, **kwargs):
            return with_extra(*args, **kwargs)

        own = "takes `{}`, which is not a named parameter of the function"
        has_default = "names `{0}` and also forwards, and `{0}` has a default"
        other_kind = (
            "takes `{}` as a keyword-only parameter where the function takes it as a "
            "positional or keyword one"
        )
        not_its_default = "gives `{}` a default that is not the function's own"
        cases = [
            (one_for_true, not_its_default.format("force")),
            (both_default, has_default.format("mode")),
            (function_defaults, has_default.format("mode")),
            (wrapper_defaults, has_default.format("mode")),
            (d.retries()(inner), own.format("retries")),
            (d.forwards(d.retries()(inner)), own.format("retries")),
            (d.retries()(d.forwards(inner)), own.format("retries")),
            (d.forwards(d.forwards(d.forwards(d.retries()(inner)))), own.format("retries")),
            (d.retries(1)(d.retries(50)(inner)), own.format("retries")),
            (d.retries()(d.retries()(inner)), own.format("retries")),
            (keeps_refresh, other_kind.format("refresh")),
            (hides_mode, "has a fixed signature that leaves out the function's parameter `mode`"),
            (other_default, not_its_default.format("mode")),
            (reordered, "in another order (`mode` where the function has `table`)"),
            (keyword_only, other_kind.format("mode")),
            (options_named_like_nothing, own.format("options")),
            (names_a_defaulted_one, other_kind.format("mode")),
            # the function's `**extra` is not a parameter a wrapper can name
            (names_the_variadic, own.format("extra")),
        ]
        # a keyword of its own however many forwarding wrappers stand above it
        for above in (4, 5, 6, 12):
            stack = d.retries()(inner)
            for _ in range(above):
                stack = d.forwards(stack)
            cases.append((stack, own.format("retries")))
        for function, cause in cases:
            for sent in ({"table": "x"}, {"table": "x", "mode": "soft", "refresh": False}):
                for validated in (True, False):
                    ctx = self._ctx_for(function, validated=validated)
                    same, reason = adapter._with_defaults(ctx, "t", dict(sent))
                    assert same == sent, function
                    assert reason.startswith("a wrapper around the function this tool runs ")
                    assert cause in reason, (function, reason)
                    assert "in a way of its own" not in reason

        # a tool that is not validated: a fixed wrapper with a default where the function has
        # none ran on it when the model sent nothing, and one with a parameter of its own
        # ran on that
        @functools.wraps(inner)
        def defaults_a_required_one(table="prod", mode="hard"):
            return inner(table, mode)

        @functools.wraps(inner)
        def forces(table, mode="hard", force=True):
            return inner(table, mode)

        for function, cause, sent in (
            (defaults_a_required_one, not_its_default.format("table"), {}),
            (forces, own.format("force"), {"table": "x"}),
        ):
            ctx = self._ctx_for(function, validated=False)
            assert cause in adapter._with_defaults(ctx, "t", dict(sent))[1]

    def test_a_decorated_tool_with_a_keyword_of_its_own_never_reaches_the_policy(self, example):
        """End to end, for both forms of the gate: the decorator in the middle of the stack
        has ``skip_review=True``; the call is refused before the policy and nothing runs."""
        import functools

        ran, seen = [], []
        d = self._decorators()

        def fast_path(function):
            @functools.wraps(function)
            def wrapper(*args, skip_review=True, **kwargs):
                ran.append(("skip_review", skip_review))
                return function(*args, **kwargs)

            return wrapper

        @d.forwards
        @fast_path
        def transfer(account: str, amount: int) -> str:
            """Transfer."""
            ran.append(("transfer", account, amount))
            return "sent"

        @d.forwards
        @fast_path
        def transfer_with_options(account: str, amount: int, **options: bool) -> str:
            """Transfer."""
            ran.append(("transfer", account, amount))
            return "sent"

        def policy(name, args):
            seen.append(dict(args))
            return []

        for tool, sent in (
            (transfer, {"account": "x", "amount": 5}),
            (transfer_with_options, {"account": "x", "amount": 5}),
            (transfer_with_options, {"account": "x", "amount": 5, "skip_review": False}),
        ):
            for capabilities in (
                [example.RecusalGate(policy)],
                [example.Hooks(before_tool_execute=example.recusal_before_tool_execute(policy))],
            ):
                ran.clear(), seen.clear()
                agent = example.Agent(
                    self._model_calling(example, tool.__name__, sent),
                    tools=[tool],
                    capabilities=capabilities,
                )
                answers = example.responses_of(agent.run_sync("go"))
                assert ran == [] and seen == []
                assert (
                    "takes `skip_review`, which is not a named parameter of the function"
                    in answers["c1"][1]
                )

    def test_a_forwarding_decorator_changes_nothing(self, example):
        """The ordinary case: a logging or timing decorator that takes ``*args, **kwargs``.
        The tool is judged with the validated arguments and runs with them."""
        ran, seen = [], []
        d = self._decorators()

        @d.forwards
        @d.forwards
        def wipe(table: str, force: bool = False) -> str:
            """Wipe."""
            ran.append((table, force))
            return "wiped"

        agent = example.Agent(
            self._model_calling(example, "wipe", {"table": "t"}),
            tools=[wipe],
            capabilities=[example.RecusalGate(lambda n, a: seen.append(dict(a)) or [])],
        )
        agent.run_sync("go")
        assert seen == [{"table": "t", "force": False}] and ran == [("t", False)]

    def test_a_default_that_cannot_be_judged_is_refused_by_name(self, example, adapter):
        """A default that is not plain data cannot be shown to a policy, and one on a
        positional-only parameter cannot be written out by name."""
        import logging

        log = logging.getLogger("x")

        def with_object(table: str, log=log) -> str:
            return "ok"

        def positional(table, mode="hard", /) -> str:
            return "ok"

        args = {"table": "x"}
        same, reason = adapter._with_defaults(self._ctx_for(with_object), "t", args)
        assert same is args and "parameter `log`" in reason and "not plain data" in reason
        same, reason = adapter._with_defaults(self._ctx_for(positional), "t", args)
        assert same is args and "parameter `mode`" in reason and "positional-only" in reason
        sent = {"table": "x", "mode": "soft", "log": None}
        assert adapter._with_defaults(self._ctx_for(with_object), "t", sent) == (sent, None)

    def test_a_tool_that_is_not_validated_is_held_to_named_parameters(self, example, adapter):
        """For a tool given its schema by hand, what the function does with a parameter the
        model left out is all that decides its value. A function taking ``*args`` or
        ``**kwargs`` can fill one in in its body, as Pydantic AI's LangChain bridge does, so
        it is refused, at any level; a function with named parameters has its defaults
        added; a ``functools.partial`` says in its own signature what a call binds."""
        import functools

        d = self._decorators()

        def inner(table: str, mode: str = "hard") -> str:
            return "ok"

        def proxy(*args, **kwargs):
            return inner(**({"mode": "hard"} | kwargs))

        def only_positional(*args):
            return "ok"

        def takes_options(table, **options):
            return "ok"

        @functools.wraps(takes_options)
        def named(table):
            return takes_options(table)

        def by_hand(function, sent=None):
            ctx = self._ctx_for(function, validated=False)
            return adapter._with_defaults(ctx, "t", {"table": "x"} if sent is None else sent)

        # the reason says whose parameter it is: the function's, or a wrapper's
        assert "and its function takes `args` as a variadic parameter" in by_hand(proxy)[1]
        assert "give the function named parameters" in by_hand(proxy)[1]
        assert "its function takes `args` as a variadic parameter" in by_hand(only_positional)[1]
        assert "and its function takes `options` as a variadic parameter" in by_hand(named)[1]
        reason = by_hand(d.forwards(inner))[1]
        assert "a wrapper around its function takes `args` as a variadic parameter" in reason
        assert "restate the function's parameters in the wrapper" in reason
        assert by_hand(inner) == ({"table": "x", "mode": "hard"}, None)
        sent = {"table": "x", "mode": "soft"}
        assert by_hand(inner, sent) == (sent, None)
        # behind a validated tool the same variadic functions are read like any other
        for function in (proxy, d.forwards(inner)):
            assert adapter._with_defaults(self._ctx_for(function), "t", dict(sent))[1] is None
        assert adapter._validates(object()) is True

        # a partial: its keywords are the defaults of its own signature, its positionals are gone
        soft = functools.partial(inner, mode="soft")
        assert by_hand(soft) == ({"table": "x", "mode": "soft"}, None)
        assert adapter._with_defaults(self._ctx_for(soft), "t", {"table": "x"})[0]["mode"] == "soft"
        bound = functools.partial(inner, "acme")
        assert by_hand(bound, {}) == ({"mode": "hard"}, None)
        # what the partial wraps is read like any other function
        over_opaque = functools.partial(d.forwards(d.retries()(inner)), "acme")
        same, reason = adapter._with_defaults(self._ctx_for(over_opaque), "t", {})
        assert same == {} and "takes `retries`, which is not a named parameter" in reason
        variadic = functools.partial(proxy, table="x")
        assert "variadic parameter" in by_hand(variadic, {})[1]

    def test_a_decorator_the_kernel_cannot_read_is_refused_wherever_it_sits(self, example):
        """A class-based decorator that is also a descriptor (it defines ``__get__`` so it
        works on methods; ``cachetools.cached`` is one) has no signature Python will give,
        and the walk fails as a whole. Read by its outermost callable alone, a stack with
        one in the middle left the default under it unjudged: the transfer ran with
        ``skip_review=True``. Such a stack is refused, outermost, in the middle, or under a
        ``functools.partial``."""
        import functools

        ran, seen = [], []
        d = self._decorators()

        class Counted:
            def __init__(self, function):
                functools.update_wrapper(self, function)
                self.function = function

            def __call__(self, *args, **kwargs):
                return self.function(*args, **kwargs)

            def __get__(self, instance, owner=None):
                return self if instance is None else functools.partial(self, instance)

        def fast_path(function):
            @functools.wraps(function)
            def wrapper(*args, skip_review=True, **kwargs):
                ran.append(("skip_review", skip_review))
                return function(*args, **kwargs)

            return wrapper

        def transfer(account: str, amount: int) -> str:
            """Transfer."""
            ran.append(("transfer", account, amount))
            return "sent"

        def bound_first(function):
            named = functools.partial(function, "x")
            named.__name__ = named.__qualname__ = "transfer"
            named.__doc__ = "Transfer."
            return named

        for build, sent in (
            (lambda: d.forwards(Counted(fast_path(transfer))), {"account": "x", "amount": 5}),
            (lambda: Counted(fast_path(transfer)), {"account": "x", "amount": 5}),
            (lambda: bound_first(d.forwards(Counted(fast_path(transfer)))), {"amount": 5}),
        ):
            ran.clear(), seen.clear()
            agent = example.Agent(
                self._model_calling(example, "transfer", sent),
                tools=[build()],
                capabilities=[example.RecusalGate(lambda n, a: seen.append(dict(a)) or [])],
            )
            answers = example.responses_of(agent.run_sync("go"))
            assert ran == [] and seen == []
            assert "or a decorator on it, cannot be inspected" in answers["c1"][1]

    def test_a_function_that_declares_its_own_signature_is_refused(self, example, adapter):
        """A callable with a ``__signature__`` shows parameters other than the ones Python
        binds, so a keyword of its own with a default cannot be found: refused, for a
        validated tool and one given its schema by hand alike."""
        import functools
        import inspect

        def inner(table: str) -> str:
            return "ok"

        @functools.wraps(inner)
        def traced(*args, sample=True, **kwargs):
            return inner(*args, **kwargs)

        traced.__signature__ = inspect.signature(inner)
        for validated in (True, False):
            ctx = self._ctx_for(traced, validated=validated)
            same, reason = adapter._with_defaults(ctx, "t", {"table": "x"})
            assert same == {"table": "x"} and "declares a signature of its own" in reason

    def test_a_function_that_cannot_be_walked_is_refused_unless_nothing_can_wrap_it(
        self, example, adapter, monkeypatch
    ):
        """When the kernel cannot walk the callable, nothing under it has been read. Only a
        class or a builtin, which wraps nothing and whose own signature Pydantic AI
        validated, is let through, and only for a validated tool."""

        class Report:
            def __init__(self, title: str, pages: int = 1):
                self.title = title

        args = {"title": "x"}
        assert adapter._with_defaults(self._ctx_for(Report), "t", args) == (args, None)
        by_hand = self._ctx_for(Report, validated=False)
        same, reason = adapter._with_defaults(by_hand, "t", dict(args))
        assert same == args and "cannot be inspected" in reason

        def tool(table: str, mode: str = "hard") -> str:
            return "ok"

        def unreadable(function, **kwargs):
            raise ValueError("no signature")

        monkeypatch.setattr(adapter, "_reached_parameters", unreadable)
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

    def test_a_naming_wrapper_is_refused_where_name_and_position_can_differ(self, example, adapter):
        """Pydantic AI passes a function's values by position when the function takes
        ``*args`` or a positional-only parameter, and a wrapper can pass on by position what
        it was given by name. A wrapper that forwards and also names a parameter matched it
        by name alone: ``wrapper(dst, *args, **kwargs)`` over ``copy_file(src, dst,
        *flags)`` was judged with ``dst`` under ``/tmp`` and wrote to the path sent as
        ``src``. Such a wrapper is transparent only where the two cannot differ."""
        import functools

        def copy_file(src: str, dst: str, *flags: str) -> str:
            return "ok"

        @functools.wraps(copy_file)
        def names_the_second(dst, *args, **kwargs):
            return copy_file(*args, dst=dst, **kwargs)

        def delete_rows(table: str, *ids: str, dry_run: bool) -> str:
            return "ok"

        @functools.wraps(delete_rows)
        def keeps_table(table, *args, **kwargs):
            return delete_rows(*args, **kwargs)

        @functools.wraps(delete_rows)
        def by_keyword(table, **kwargs):
            return delete_rows(table, **kwargs)

        @functools.wraps(delete_rows)
        def names_the_keyword_only_one(*args, dry_run, **kwargs):
            return delete_rows(*args, dry_run=dry_run, **kwargs)

        def positional_only(table: str, /, limit: int) -> str:
            return "ok"

        @functools.wraps(positional_only)
        def names_limit(limit, **kwargs):
            return positional_only(limit=limit, **kwargs)

        def plain(table: str, limit: int, *, flag: bool) -> str:
            return "ok"

        @functools.wraps(plain)
        def forwards_positionals(table, *args, **kwargs):
            return plain(table, *args, **kwargs)

        @functools.wraps(plain)
        def names_the_second_one(limit, **kwargs):
            return plain(limit=limit, **kwargs)

        @functools.wraps(plain)
        def swapped(limit, table, **kwargs):
            return plain(table, limit, **kwargs)

        @functools.wraps(plain)
        def positional_only_name(table, /, **kwargs):
            return plain(table, **kwargs)

        @functools.wraps(plain)
        def flag_by_position(table, limit, flag, **kwargs):
            return plain(table, limit, flag=flag, **kwargs)

        # an honest wrapper that passes on by position, over one that keeps what it names:
        # the value sent as `limit` ran as `table`
        @functools.wraps(forwards_positionals)
        def restates(table, limit, *, flag):
            return forwards_positionals(table, limit, flag=flag)

        @functools.wraps(forwards_positionals)
        def names_two_above(table, limit, **kwargs):
            return forwards_positionals(table, limit, **kwargs)

        # read whatever the order: the wrapper that names more may stand under it as well
        @functools.wraps(plain)
        def names_two(table, limit, **kwargs):
            return plain(table, limit, **kwargs)

        @functools.wraps(names_two)
        def forwards_args_above(table, *args, **kwargs):
            return names_two(table, *args, **kwargs)

        # no wrapper names more positional parameters than the one that forwards `*args`
        @functools.wraps(forwards_positionals)
        def forwards_positionals_twice(table, *args, **kwargs):
            return forwards_positionals(table, *args, **kwargs)

        @functools.wraps(forwards_positionals)
        def names_as_many(table, **kwargs):
            return forwards_positionals(table, **kwargs)

        by_position = (
            "names `{}` and also forwards, over a function that takes `*args` or a "
            "positional-only parameter"
        )
        not_alone = (
            "names `table` and also forwards positional arguments (`*args`), in a stack "
            "where another wrapper names more positional parameters"
        )
        for function, cause in (
            (names_the_second, by_position.format("dst")),
            (keeps_table, by_position.format("table")),
            (by_keyword, by_position.format("table")),
            (names_the_keyword_only_one, by_position.format("dry_run")),
            (names_limit, by_position.format("limit")),
            (restates, not_alone),
            (names_two_above, not_alone),
            (forwards_args_above, not_alone),
            (names_the_second_one, "`limit` is not where the function has it (the function has"),
            (swapped, "`limit` is not where the function has it (the function has `table`"),
            (positional_only_name, "takes `table` as a positional-only parameter where"),
            (flag_by_position, "takes `flag` as a positional or keyword parameter where"),
        ):
            for validated in (True, False):
                ctx = self._ctx_for(function, validated=validated)
                same, reason = adapter._with_defaults(ctx, "t", {"table": "x"})
                assert same == {"table": "x"}, function
                assert cause in reason, (function, reason)

        # transparent: the leading required parameters, in order, and a required
        # keyword-only one, over a function Pydantic AI calls by keyword
        @functools.wraps(plain)
        def first(table, **kwargs):
            return plain(table, **kwargs)

        @functools.wraps(plain)
        def first_two(table, limit, **kwargs):
            return plain(table, limit, **kwargs)

        @functools.wraps(plain)
        def names_flag(*args, flag, **kwargs):
            return plain(*args, flag=flag, **kwargs)

        @functools.wraps(plain)
        def first_and_flag(table, *, flag, **kwargs):
            return plain(table, flag=flag, **kwargs)

        # and one that forwards `*args` too, while no wrapper names more positional parameters
        # than it does: then nothing reaches its `*args` to move into the place it keeps
        d = self._decorators()
        sent = {"table": "x", "limit": 1, "flag": True}
        for function in (
            first,
            first_two,
            names_flag,
            first_and_flag,
            forwards_positionals,
            d.forwards(forwards_positionals),
            forwards_positionals_twice,
            names_as_many,
        ):
            assert adapter._with_defaults(self._ctx_for(function), "t", dict(sent)) == (sent, None)

    def test_a_value_pydantic_ai_would_spread_is_refused(self, example, adapter):
        """Pydantic AI hands a function's ``*rest`` its items one by one. Validation makes
        the value a list; where a hook of the application's skips validation, a string or a
        dict reaches the gate, the policy was shown ``"ab"`` and the function ran with
        ``"a", "b"``. Refused, for both forms of the gate."""
        from pydantic_ai.exceptions import SkipToolValidation

        ran, seen = [], []

        def tag(table: str, *labels: str) -> str:
            """Tag."""
            ran.append((table, labels))
            return "tagged"

        spread = "the value for `*labels` of the function this tool runs is not a list"
        for sent, refused in (
            ({"table": "x", "labels": "ab"}, True),
            ({"table": "x", "labels": {"a": 1}}, True),
            ({"table": "x", "labels": ["a", "b"]}, False),
            ({"table": "x", "labels": []}, False),
            ({"table": "x"}, False),
        ):
            same, reason = adapter._with_defaults(self._ctx_for(tag), "t", dict(sent))
            assert same == sent
            assert (spread in reason) if refused else (reason is None), (sent, reason)

        async def skip_validation(ctx, *, call, tool_def, args):
            raise SkipToolValidation(dict(args))

        def policy(name, args):
            seen.append(dict(args))
            return []

        for gate in (
            example.RecusalGate(policy),
            example.Hooks(before_tool_execute=example.recusal_before_tool_execute(policy)),
        ):
            for sent, refused in (
                ({"table": "t", "labels": "ab"}, True),
                ({"table": "t", "labels": ["a", "b"]}, False),
            ):
                ran.clear(), seen.clear()
                agent = example.Agent(
                    self._model_calling(example, "tag", sent),
                    tools=[tag],
                    capabilities=[example.Hooks(before_tool_validate=skip_validation), gate],
                )
                answers = example.responses_of(agent.run_sync("go"))
                if refused:
                    assert ran == [] and seen == [] and spread in answers["c1"][1]
                else:
                    assert seen == [sent] and ran == [("t", ("a", "b"))]

    def test_a_wrapper_that_binds_by_position_never_reaches_the_policy(self, example):
        """End to end, for both forms of the gate, the two stacks of the review: the path
        the model sent as ``src`` was written to, and the rows of the table sent among the
        ids were deleted. Neither is judged or run now."""
        import functools

        ran, seen = [], []

        def checks_destination(function):
            @functools.wraps(function)
            def wrapper(dst, *args, **kwargs):
                return function(*args, dst=dst, **kwargs)

            return wrapper

        def with_table_lock(function):
            @functools.wraps(function)
            def wrapper(table, *args, **kwargs):
                return function(*args, **kwargs)

            return wrapper

        @checks_destination
        def copy_file(src: str, dst: str, *flags: str) -> str:
            """Copy a file."""
            ran.append(("copy_file", src, dst))
            return "copied"

        @with_table_lock
        def delete_rows(table: str, *ids: str, dry_run: bool = False) -> str:
            """Delete rows."""
            ran.append(("delete_rows", table, ids))
            return "deleted"

        def policy(name, args):
            seen.append(dict(args))
            return []

        for tool, sent, named in (
            (copy_file, {"src": "/etc/cron.d/x", "dst": "/tmp/payload", "flags": []}, "dst"),
            (delete_rows, {"table": "scratch", "ids": ["customers", "1"]}, "table"),
        ):
            for capabilities in (
                [example.RecusalGate(policy)],
                [example.Hooks(before_tool_execute=example.recusal_before_tool_execute(policy))],
            ):
                ran.clear(), seen.clear()
                agent = example.Agent(
                    self._model_calling(example, tool.__name__, sent),
                    tools=[tool],
                    capabilities=capabilities,
                )
                answers = example.responses_of(agent.run_sync("go"))
                assert ran == [] and seen == []
                assert f"names `{named}` and also forwards, over a function" in answers["c1"][1]
                assert "customers" not in answers["c1"][1] and "/etc" not in answers["c1"][1]

    def test_the_run_context_is_the_first_positional_parameter_whatever_it_is_called(
        self, example, adapter
    ):
        """Pydantic AI passes the run context positionally and first, so for a
        context-taking tool the first positional parameter of every level is the context:
        a wrapper that calls it ``context`` or ``run_ctx``, or takes it positional-only, was
        refused as taking a parameter of its own."""
        import functools

        from pydantic_ai import Tool

        ran, seen = [], []

        def lookup(ctx, table: str, limit: int = 10) -> str:
            """Lookup."""
            ran.append((table, limit))
            return "ok"

        @functools.wraps(lookup)
        def another_name(context, *args, **kwargs):
            return lookup(context, *args, **kwargs)

        @functools.wraps(lookup)
        def restated(run_ctx, table, limit=10):
            return lookup(run_ctx, table, limit)

        @functools.wraps(lookup)
        def positional_only(ctx, /, *args, **kwargs):
            return lookup(ctx, *args, **kwargs)

        @functools.wraps(lookup)
        def names_one_after_it(context, table, **kwargs):
            return lookup(context, table, **kwargs)

        wrappers = (another_name, restated, positional_only, names_one_after_it)
        for function in wrappers:
            ctx = self._ctx_for(function, takes_ctx=True)
            completed = {"table": "x", "limit": 10}
            assert adapter._with_defaults(ctx, "t", {"table": "x"}) == (completed, None)
            # only for a context-taking tool: elsewhere the same parameter is the wrapper's own
            reason = adapter._with_defaults(self._ctx_for(function), "t", {"table": "x"})[1]
            assert reason is not None and "a wrapper around the function" in reason

        # the context is the first positional parameter, not any first parameter
        @functools.wraps(lookup)
        def keyword_first(*args, retries=3, **kwargs):
            return lookup(*args, **kwargs)

        ctx = self._ctx_for(keyword_first, takes_ctx=True)
        assert "takes `retries`" in adapter._with_defaults(ctx, "t", {"table": "x"})[1]

        # a first positional parameter that carries the name of another parameter of the
        # function is that parameter: a wrapper above can hand it the value by keyword, and
        # this one kept it while the function ran on its own default
        @functools.wraps(lookup)
        def names_limit_first(limit, **kwargs):
            return lookup(**kwargs)

        @functools.wraps(names_limit_first)
        def by_keyword(ctx, *args, **kwargs):
            return names_limit_first(*args, ctx=ctx, **kwargs)

        @functools.wraps(lookup)
        def names_table_first(table, *args, **kwargs):
            return lookup(*args, **kwargs)

        # the context is held to the rules as the function's first parameter: over a
        # function Pydantic AI calls by position, a wrapper that names it and forwards
        # could keep it and let the next value move into its place
        def listed(ctx, table: str, *ids: str) -> str:
            return "ok"

        @functools.wraps(listed)
        def names_the_context(context, *args, **kwargs):
            return listed(*args, **kwargs)

        @functools.wraps(listed)
        def forwards_everything(*args, **kwargs):
            return listed(*args, **kwargs)

        @functools.wraps(another_name)
        def restates_above(run_ctx, table, limit=10):
            return another_name(run_ctx, table, limit)

        @functools.wraps(another_name)
        def names_it_again(ctx, *args, **kwargs):
            return another_name(ctx, *args, **kwargs)

        for function, cause in (
            (names_limit_first, "names `limit` and also forwards, and `limit` has a default"),
            (by_keyword, "names `limit` and also forwards, and `limit` has a default"),
            (names_table_first, "`table` is not where the function has it (the function has `ctx`"),
            (
                names_the_context,
                "names `ctx` and also forwards, over a function that takes `*args`",
            ),
            (restates_above, "names `ctx` and also forwards positional arguments (`*args`), in a"),
        ):
            reason = adapter._with_defaults(
                self._ctx_for(function, takes_ctx=True), "t", {"table": "x"}
            )[1]
            assert reason is not None and cause in reason, (function, reason)
        for function in (forwards_everything, names_it_again):
            ctx = self._ctx_for(function, takes_ctx=True)
            assert adapter._with_defaults(ctx, "t", {"table": "x", "limit": 1})[1] is None

        for function in wrappers:
            ran.clear(), seen.clear()
            agent = example.Agent(
                self._model_calling(example, "lookup", {"table": "t"}),
                tools=[Tool(function, name="lookup", takes_ctx=True)],
                capabilities=[example.RecusalGate(lambda n, a: seen.append(dict(a)) or [])],
            )
            agent.run_sync("go")
            assert seen == [{"table": "t", "limit": 10}] and ran == [("t", 10)], function

    def test_a_wrapper_that_restates_the_keyword_variadic_too_is_transparent(
        self, example, adapter
    ):
        """A wrapper that restates a function exactly, its ``**extra`` included, is handed
        every value like a fixed one. Not so with another name for the variadic (then it
        only forwards, and names a defaulted parameter), with ``*rest`` (positional
        arguments can move into the place of one it keeps), over a positional-only
        parameter, or with a default of its own."""
        import functools

        ran, seen = [], []

        def tagged(table: str, limit: int = 10, **extra: int) -> str:
            """Tagged."""
            ran.append((table, limit, extra))
            return "ok"

        @functools.wraps(tagged)
        def restated(table, limit=10, **extra):
            return tagged(table, limit, **extra)

        @functools.wraps(tagged)
        def no_defaults(table, limit, **extra):
            return tagged(table, limit, **extra)

        @functools.wraps(tagged)
        def other_name(table, limit=10, **kwargs):
            return tagged(table, limit, **kwargs)

        @functools.wraps(tagged)
        def other_default(table, limit=11, **extra):
            return tagged(table, limit, **extra)

        def listed(table: str, limit: int = 10, *rest: int) -> str:
            return "ok"

        @functools.wraps(listed)
        def restates_rest(table, limit=10, *rest):
            return listed(table, limit, *rest)

        def positional(table: str, /, limit: int = 10, **extra: int) -> str:
            return "ok"

        @functools.wraps(positional)
        def restates_positional(table, /, limit=10, **extra):
            return positional(table, limit, **extra)

        sent = {"table": "x", "limit": 5, "depth": 2}
        for function in (restated, no_defaults):
            assert adapter._with_defaults(self._ctx_for(function), "t", dict(sent)) == (sent, None)
        for function, cause in (
            (other_name, "names `limit` and also forwards, and `limit` has a default"),
            (other_default, "gives `limit` a default that is not the function's own"),
            (restates_rest, "names `table` and also forwards, over a function that takes `*args`"),
            (restates_positional, "over a function that takes `*args` or a positional-only"),
        ):
            reason = adapter._with_defaults(self._ctx_for(function), "t", dict(sent))[1]
            assert cause in reason, (function, reason)

        agent = example.Agent(
            self._model_calling(example, "tagged", {"table": "t", "depth": 2}),
            tools=[restated],
            capabilities=[example.RecusalGate(lambda n, a: seen.append(dict(a)) or [])],
        )
        agent.run_sync("go")
        assert seen == [{"table": "t", "limit": 10, "depth": 2}]
        assert ran == [("t", 10, {"depth": 2})]

    def test_a_single_dispatch_function_with_a_registered_implementation_is_refused(
        self, example, adapter, monkeypatch
    ):
        """``functools.singledispatch`` picks the implementation that runs by the type of
        the first argument, at call time, and the walk ends at the base function: a
        registered implementation's own ``skip_review=True`` ran unjudged. Refused wherever
        it stands in a stack, for the function and for ``singledispatchmethod``; with only
        its base implementation the function that runs is the one that was read."""
        import functools
        import types

        ran, seen = [], []
        d = self._decorators()

        @functools.singledispatch
        def lookup(key: int | str, /, deep: bool = False) -> str:
            """Lookup."""
            ran.append(("base", key, deep))
            return "base"

        @functools.singledispatch
        def only_base(key: int, /, deep: bool = False) -> str:
            """Lookup."""
            ran.append(("only", key, deep))
            return "base"

        class Service:
            @functools.singledispatchmethod
            def lookup(self, key, deep=False):
                return "base"

            @functools.singledispatchmethod
            def only_base(self, key, deep=False):
                return "base"

        chosen = "the implementation that runs is chosen at call time"
        sent = {"key": "k"}
        assert adapter._with_defaults(self._ctx_for(lookup), "t", dict(sent))[1] is None

        @lookup.register(str)
        def _lookup_text(key: str, /, deep: bool = False, skip_review: bool = True) -> str:
            ran.append(("text", key, deep, skip_review))
            return "text"

        @Service.lookup.register(str)
        def _method_text(self, key, deep=False, skip_review=True):
            return "text"

        partial = functools.partial(lookup, deep=True)

        # a wrapper that says what it wraps and copies nothing else from it, as a method
        def passes_on(*args, **kwargs):
            return lookup(*args, **kwargs)

        passes_on.__wrapped__ = lookup
        as_method = types.MethodType(passes_on, "k")
        for function in (lookup, d.forwards(lookup), partial, as_method, Service().lookup):
            for validated in (True, False):
                ctx = self._ctx_for(function, validated=validated)
                same, reason = adapter._with_defaults(ctx, "t", dict(sent))
                assert same == sent and chosen in reason, (function, reason)
                assert "single-dispatch function" in reason
        for function in (only_base, d.forwards(only_base), Service().only_base):
            reason = adapter._with_defaults(self._ctx_for(function), "t", dict(sent))[1]
            assert reason is None or chosen not in reason, (function, reason)
        assert adapter._with_defaults(self._ctx_for(only_base), "t", dict(sent))[1] is None

        def policy(name, args):
            seen.append(dict(args))
            return []

        for capabilities in (
            [example.RecusalGate(policy)],
            [example.Hooks(before_tool_execute=example.recusal_before_tool_execute(policy))],
        ):
            ran.clear(), seen.clear()
            agent = example.Agent(
                self._model_calling(example, "lookup", {"key": "k"}),
                tools=[lookup],
                capabilities=capabilities,
            )
            answers = example.responses_of(agent.run_sync("go"))
            assert ran == [] and seen == [] and chosen in answers["c1"][1]
        ran.clear(), seen.clear()
        agent = example.Agent(
            self._model_calling(example, "only_base", {"key": 1}),
            tools=[only_base],
            capabilities=[example.RecusalGate(policy)],
        )
        agent.run_sync("go")
        assert seen == [{"key": 1, "deep": False}] and ran == [("only", 1, False)]

        # a stack with more callables than the gate reads, or one it cannot read, is refused
        monkeypatch.setattr(adapter, "_STEP_LIMIT", 2)
        stacked = d.forwards(d.forwards(only_base))
        reason = adapter._with_defaults(self._ctx_for(stacked), "t", dict(sent))[1]
        assert "cannot be inspected" in reason
        assert adapter._with_defaults(self._ctx_for(only_base), "t", dict(sent))[1] is None

    def test_a_value_bound_by_position_past_a_naming_wrapper_is_refused(self, example, adapter):
        """A ``functools.partial`` with a positional argument, and a bound method, hand a
        value in by position, and it is read as bound to the innermost function's first
        parameter. A wrapper on the way that takes it and forwards the rest need not hand it
        on: ``partial(scoped(purge), "acme")`` was judged with ``{"limit": 5}`` and ran
        ``table="prod"``, the default of the parameter the partial was read as filling.
        Refused unless a wrapper that keeps the value fails the call."""
        import functools

        ran, seen = [], []
        d = self._decorators()

        def scoped(function):
            @functools.wraps(function)
            def wrapper(tenant, *args, **kwargs):
                return function(*args, **kwargs)

            return wrapper

        def with_self(function):
            @functools.wraps(function)
            def wrapper(self, *args, **kwargs):
                return function(self, *args, **kwargs)

            return wrapper

        def purge(table: str = "prod", limit: int = 10) -> str:
            """Purge."""
            ran.append((table, limit))
            return "purged"

        def required(table: str, limit: int = 10) -> str:
            return "ok"

        def listed(table: str, *ids: str) -> str:
            return "ok"

        def only_listed(*ids: str) -> str:
            return "ok"

        class Service:
            @with_self
            def query(self, table: str, limit: int = 10) -> str:
                return "ok"

            @with_self
            def remove(self, table: str, *ids: str) -> str:
                return "ok"

            @with_self
            def positional(self, table: str, /, limit: int = 10) -> str:
                return "ok"

            adapted = with_self(purge)

        class Wrapping:
            def __init__(self, function):
                functools.update_wrapper(self, function)
                self.function = function

            def __call__(self, *args, **kwargs):
                return self.function(*args, **kwargs)

        class Called:
            # the object is handed in by position, like a method's `self`
            __call__ = with_self(purge)

        def three(tenant: str, table: str = "prod", limit: int = 10) -> str:
            return "ok"

        @functools.wraps(purge)
        def restated(table="prod", limit=10):
            return purge(table, limit)

        @functools.wraps(required)
        def restated_required(table, limit=10):
            return required(table, limit)

        past = "by position past a wrapper that names a positional parameter: whether"
        service = Service()
        for function, name in (
            (functools.partial(scoped(purge), "acme"), "table"),
            (functools.partial(restated, "acme"), "table"),
            (functools.partial(d.forwards(scoped(purge)), "acme"), "table"),
            (functools.partial(scoped(listed), "acme"), "ids"),
            (functools.partial(scoped(only_listed), "acme"), "ids"),
            (d.forwards(functools.partial(scoped(purge), "acme")), "table"),
            (service.remove, "ids"),
            (service.positional, "table"),
            (service.adapted, "table"),
            (Called(), "table"),
            # every value a partial binds is read, not the first alone
            (functools.partial(scoped(three), "acme", "users"), "table"),
        ):
            for validated in (True, False):
                ctx = self._ctx_for(function, validated=validated)
                same, reason = adapter._with_defaults(ctx, "t", {"limit": 5})
                assert same == {"limit": 5}, function
                assert past in reason and f"the function's `{name}`" in reason, (function, reason)
        # not refused: keeping the value fails the call, or no wrapper names anything
        for function in (
            functools.partial(scoped(required), "acme"),
            functools.partial(restated_required, "acme"),
            functools.partial(d.forwards(purge), "acme"),
            functools.partial(scoped(purge), limit=5),
            functools.partial(purge, "acme"),
            service.query,
            Wrapping(purge),
            functools.partial(Wrapping(purge), "acme"),
        ):
            reason = adapter._with_defaults(self._ctx_for(function), "t", {"limit": 5})[1]
            assert reason is None or past not in reason, (function, reason)
        completed = adapter._with_defaults(self._ctx_for(service.query), "t", {"table": "x"})
        assert completed == ({"table": "x", "limit": 10}, None)

        tool = functools.partial(scoped(purge), "acme")
        tool.__name__ = tool.__qualname__ = "purge"
        tool.__doc__ = "Purge."
        for capabilities in (
            [example.RecusalGate(lambda n, a: seen.append(dict(a)) or [])],
            [
                example.Hooks(
                    before_tool_execute=example.recusal_before_tool_execute(
                        lambda n, a: seen.append(dict(a)) or []
                    )
                )
            ],
        ):
            ran.clear(), seen.clear()
            agent = example.Agent(
                self._model_calling(example, "purge", {"limit": 5}),
                tools=[tool],
                capabilities=capabilities,
            )
            answers = example.responses_of(agent.run_sync("go"))
            assert ran == [] and seen == [] and past in answers["c1"][1]

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
