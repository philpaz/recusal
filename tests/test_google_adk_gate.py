"""Regression tests for the offline Google ADK + Recusal before-tool-callback example."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "google_adk_gate.py"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"


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
    assert "google_adk_gate.py requires Python 3.11+" in result.stderr
    assert f"this is Python 3.{minor}." in result.stderr


def test_the_adk_stays_out_of_the_recusal_package_and_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "google-adk" not in pyproject
    assert "google-genai" not in pyproject
    for path in (ROOT / "recusal").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "google.adk" not in source and "google.genai" not in source, path


def test_pinned_versions_are_explicit_in_example_docs_and_ci():
    expected = ("google-adk==2.13.0", "google-genai==2.34.0")
    for path in (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        text = path.read_text(encoding="utf-8")
        for pin in expected:
            assert pin in text, f"{path.name}: {pin}"
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for pin in expected:
        assert ci.count(pin) == 2, pin


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the Google ADK requires Python 3.11+")
class TestGoogleAdkGate:
    @pytest.fixture
    def example(self):
        pytest.importorskip("google.adk", reason="google-adk is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_google_adk_gate_example", EXAMPLE)
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

    # --- the callback route ----------------------------------------------------------------

    def test_allowed_call_runs_its_tool(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": SELECT}, "c1")])
        assert executed == [("run_sql", {"sql": SELECT})]
        assert responses == {"c1": {"result": f"recorded SQL: {SELECT}"}}

    def test_refused_call_does_not_run_and_reason_reaches_the_model(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")])
        assert executed == []
        assert responses["c1"]["error"].startswith("Recusal refused `run_sql` [FAIL]")
        assert "top-level WHERE" in responses["c1"]["error"]

    @pytest.mark.parametrize("via_plugin", [False, True], ids=["callback", "plugin"])
    def test_several_calls_in_one_turn_are_gated_separately(self, example, via_plugin):
        responses, executed = example.run_calls(self.three_calls(), via_plugin=via_plugin)
        assert responses["c1"] == {"result": "hello"}
        assert "Recusal refused" in responses["c2"]["error"]
        assert responses["c3"] == {"result": f"recorded SQL: {SELECT}"}
        assert sorted(executed) == [("echo", {"text": "hello"}), ("run_sql", {"sql": SELECT})]

    @pytest.mark.parametrize("via_plugin", [False, True], ids=["callback", "plugin"])
    def test_policy_exception_fails_closed(self, example, via_plugin):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        responses, executed = example.run_calls(
            [("echo", {"text": "unsafe"}, "c1")], policy=broken_policy, via_plugin=via_plugin
        )
        assert executed == []
        assert responses["c1"] == {"error": "Recusal failed closed (policy error): policy exploded"}

    def test_a_crash_inside_decide_also_fails_closed(self, example, monkeypatch):
        def crashing_decide(*args, **kwargs):
            raise RuntimeError("decide crashed")

        monkeypatch.setattr(example, "decide", crashing_decide)
        responses, executed = example.run_calls([("echo", {"text": "unsafe"}, "c1")])
        assert executed == []
        assert responses["c1"] == {"error": "Recusal failed closed (policy error): decide crashed"}

    def test_missing_arguments_reach_the_policy_as_an_empty_mapping(self, example):
        seen = []

        def recording(tool_name, tool_input):
            seen.append((tool_name, tool_input))
            return []

        responses, executed = example.run_calls([("echo", None, "c1")], policy=recording)
        # the ADK normalizes a missing args field to {}; the gate completes defaults before
        # judgment and `text` has none, so the call is refused before the policy runs
        assert seen == []
        assert executed == []
        assert "required parameter `text` was not supplied" in responses["c1"]["error"]

    def test_non_mapping_arguments_are_refused_by_the_gate_itself(self, example):
        tool = example.FunctionTool(example.run_sql)
        assert example._refusal(tool, ["DELETE"], example.sql_scope_policy) == (
            "Recusal refused `run_sql`: arguments are not a mapping"
        )
        assert example._refusal(tool, "DELETE", example.sql_scope_policy) is not None

    def test_negative_control_runs_delete_without_the_gate(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")], gated=False)
        assert executed == [("run_sql", {"sql": DELETE})]
        assert responses["c1"] == {"result": f"recorded SQL: {DELETE}"}

    # --- gated_agent: a copy, the gate last, everything under it gated ---------------------

    def test_gated_agent_returns_a_copy_and_leaves_the_original_ungated(self, example):
        agent = example.scripted_agent(self.three_calls())
        gated = example.gated_agent(agent, example.sql_scope_policy)
        assert gated is not agent
        assert agent.before_tool_callback is None
        assert example.is_gated(gated) and not example.is_gated(agent)
        assert [t.name for t in gated.tools] == [t.name for t in agent.tools]

    def test_gated_agent_keeps_existing_callbacks_and_appends_the_gate_last(self, example):
        order = []

        def first(tool, args, tool_context):
            order.append("first")
            return None

        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
        agent = agent.clone(update={"before_tool_callback": first})
        gated = example.gated_agent(agent, example.sql_scope_policy)
        callbacks = gated.before_tool_callback
        assert isinstance(callbacks, list) and len(callbacks) == 2
        assert callbacks[0] is first
        assert getattr(callbacks[-1], "recusal_gate", False)
        asyncio.run(example.run_agent(gated))
        assert order == ["first"]

    def test_why_the_gate_is_last_a_later_callback_can_change_what_runs(self, example):
        def rewrite(tool, args, tool_context):
            if tool.name == "run_sql":
                args["sql"] = DELETE  # mutates the dict the tool will receive
            return None

        example.EXECUTED.clear()
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
        # gate FIRST: it judges the SELECT, then the rewrite turns it into a DELETE that runs
        wrong = agent.clone(
            update={
                "before_tool_callback": [
                    example.recusal_before_tool(example.sql_scope_policy),
                    rewrite,
                ]
            }
        )
        responses = asyncio.run(example.run_agent(wrong))
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]
        assert "result" in responses["c1"]
        # gate LAST, as gated_agent places it: the rewritten DELETE is what gets judged
        # (a fresh scripted agent: the model's turn counter is per instance)
        example.EXECUTED.clear()
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
        right = example.gated_agent(
            agent.clone(update={"before_tool_callback": rewrite}), example.sql_scope_policy
        )
        responses = asyncio.run(example.run_agent(right))
        assert example.EXECUTED == []
        assert "Recusal refused" in responses["c1"]["error"]

    def test_gated_agent_refuses_a_code_executor(self, example):
        from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor

        agent = example.scripted_agent([]).clone(
            update={"code_executor": UnsafeLocalCodeExecutor()}
        )
        with pytest.raises(ValueError, match="code_executor"):
            example.gated_agent(agent, example.sql_scope_policy)
        assert not example.is_gated(agent)

    def test_gated_agent_refuses_an_llm_agent_subclass(self, example):
        class Special(example.LlmAgent):
            pass

        agent = Special(name="special", model=example.ScriptedLlm(model="s"), tools=[])
        with pytest.raises(ValueError, match="not a plain LlmAgent"):
            example.gated_agent(agent, example.sql_scope_policy)

    # --- sub-agents: the trap, and both ways out --------------------------------------------

    def _transfer_setup(self, example):
        """A parent whose scripted model transfers to a child that proposes a DELETE."""
        from google.adk.models import LlmResponse
        from google.genai import types

        class Transfer(example.ScriptedLlm):
            async def generate_content_async(self, llm_request, stream=False):
                self.turns += 1
                if self.turns == 1:
                    parts = [
                        types.Part(
                            function_call=types.FunctionCall(
                                id="t1", name="transfer_to_agent", args={"agent_name": "child"}
                            )
                        )
                    ]
                else:
                    parts = [types.Part(text="done")]
                yield LlmResponse(content=types.Content(role="model", parts=parts))

        child = example.LlmAgent(
            name="child",
            model=example.ScriptedLlm(model="s", calls=[("run_sql", {"sql": DELETE}, "s1")]),
            tools=[example.FunctionTool(example.run_sql)],
        )
        parent = example.LlmAgent(name="parent", model=Transfer(model="s"), sub_agents=[child])
        example.EXECUTED.clear()
        return parent

    def test_negative_control_a_gate_on_the_parent_alone_misses_the_sub_agent(self, example):
        parent = self._transfer_setup(example)
        only_parent = parent.clone(
            update={"before_tool_callback": example.recusal_before_tool(example.sql_scope_policy)}
        )
        responses = asyncio.run(example.run_agent(only_parent))
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]
        assert responses["s1"] == {"result": f"recorded SQL: {DELETE}"}
        assert not example.is_gated(only_parent)

    def test_gated_agent_gates_the_sub_agent_reached_by_transfer(self, example):
        parent = self._transfer_setup(example)
        gated = example.gated_agent(parent, example.sql_scope_policy)
        responses = asyncio.run(example.run_agent(gated))
        assert example.EXECUTED == []
        assert "Recusal refused" in responses["s1"]["error"]
        assert example.is_gated(gated)
        assert not example.is_gated(parent)  # the original is untouched

    def test_the_plugin_gates_the_sub_agent_without_touching_any_agent(self, example):
        parent = self._transfer_setup(example)
        responses = asyncio.run(
            example.run_agent(parent, plugins=[example.RecusalPlugin(example.sql_scope_policy)])
        )
        assert example.EXECUTED == []
        assert "Recusal refused" in responses["s1"]["error"]

    def test_gated_agent_gates_the_agent_behind_an_agent_tool(self, example):
        child = example.LlmAgent(
            name="child",
            model=example.ScriptedLlm(model="s", calls=[("run_sql", {"sql": DELETE}, "s1")]),
            tools=[example.FunctionTool(example.run_sql)],
        )
        parent = example.scripted_agent(
            [("child", {"request": "clean up"}, "a1")], tools=[example.AgentTool(child)]
        )
        assert not example.is_gated(
            parent.clone(
                update={
                    "before_tool_callback": example.recusal_before_tool(example.sql_scope_policy)
                }
            )
        )
        gated = example.gated_agent(parent, example.sql_scope_policy)
        assert example.is_gated(gated)
        inner = [t for t in gated.tools if isinstance(t, example.AgentTool)][0].agent
        assert example.is_gated(inner)
        assert inner is not child

    def test_gated_agent_gates_the_sub_agents_of_a_workflow_agent(self, example):
        child = example.scripted_agent([("run_sql", {"sql": DELETE}, "s1")])
        flow = example.SequentialAgent(name="flow", sub_agents=[child])
        gated = example.gated_agent(flow, example.sql_scope_policy)
        assert isinstance(gated, example.SequentialAgent)
        assert example.is_gated(gated) and not example.is_gated(flow)
        example.EXECUTED.clear()
        responses = asyncio.run(example.run_agent(gated))
        assert example.EXECUTED == []
        assert "Recusal refused" in responses["s1"]["error"]

    # --- the script ------------------------------------------------------------------------

    def test_example_runs_end_to_end(self, example):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("tools that ran: 2") == 2
        assert result.stdout.count("Recusal refused `run_sql` [FAIL]") == 2
        assert "before_tool_callback:" in result.stdout and "RecusalPlugin:" in result.stdout


def _same(a, b) -> bool:
    """Exact agreement: the same type at every node, then equal."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_the_exact_comparator_refuses_look_alikes():
    assert not _same(True, 1) and not _same(1, 1.0) and not _same({"a": 1}, {"a": "1"})


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the ADK requires Python 3.11+")
class TestPreparedInvocation:
    """Adapters review, finding 3 (ADK): the callback judged the raw args dict; the ADK then
    prepared (coerced) the arguments and Python filled omitted defaults, so the policy saw
    ``{}`` and the function ran 100, and ``amount: int = "100"`` was judged as a str and
    received as an int. Defaults are completed into the raw arguments before the tool's own
    preparation runs once; the converted values are written back and judged; only shapes
    whose preparation is pure are supported."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("google.adk", reason="google-adk is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_adk_prepared_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.fixture(params=[True, False], ids=["validation-on", "validation-off"])
    def validation(self, request):
        from google.adk.features import FeatureName, override_feature_enabled

        override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, request.param)
        yield request.param
        override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True)

    @staticmethod
    def budget(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append(args)
            if isinstance(args.get("amount"), int) and args["amount"] > 10:
                return [Finding.fail("budget", severity="CRITICAL", message="over 10")]
            return []

        return policy

    @pytest.mark.parametrize("via_plugin", [False, True], ids=["callback", "plugin"])
    def test_an_omitted_default_is_judged_at_the_value_the_function_uses(self, example, via_plugin):
        seen = []
        responses, executed = example.run_calls(
            [("charge", {}, "c1")], policy=self.budget(seen), via_plugin=via_plugin
        )
        assert seen == [{"amount": 100}] and type(seen[0]["amount"]) is int
        assert executed == [] and "over 10" in responses["c1"]["error"]

    def test_a_non_canonical_default_is_prepared_before_judgment(self, example, validation):
        received, seen = [], []

        def charge(amount: int = "100") -> str:  # type: ignore[assignment]
            received.append(amount)
            return "recorded"

        example.run_calls(
            [("charge", {}, "c1")],
            policy=lambda n, a: seen.append(a) or [],
            tools=[example.FunctionTool(charge)],
        )
        assert _same(seen[0]["amount"], received[0]), (seen, received)
        assert type(received[0]) is (int if validation else str)

    def test_an_explicit_string_is_prepared_once_and_judged_as_received(self, example, validation):
        received, seen = [], []

        def charge(amount: int = 100) -> str:
            received.append(amount)
            return "recorded"

        example.run_calls(
            [("charge", {"amount": "7"}, "c1")],
            policy=lambda n, a: seen.append(a) or [],
            tools=[example.FunctionTool(charge)],
        )
        assert _same(seen[0]["amount"], received[0])
        assert type(received[0]) is (int if validation else str)

    def test_a_nested_container_default_is_judged_and_received_exactly(self, example):
        received, seen = [], []

        def tag(labels: list[str] = ["a", "b"], weights: dict[str, float] = {"x": 1.5}) -> str:  # noqa: B006
            received.append((labels, weights))
            return "recorded"

        example.run_calls(
            [("tag", {}, "c1")],
            policy=lambda n, a: seen.append(a) or [],
            tools=[example.FunctionTool(tag)],
        )
        assert _same(seen[0], {"labels": ["a", "b"], "weights": {"x": 1.5}})
        assert _same(list(received[0]), [["a", "b"], {"x": 1.5}])

    def test_a_mutating_policy_cannot_change_the_prepared_invocation(self, example):
        received = []

        def tag(labels: list[str] = ["a"]) -> str:  # noqa: B006
            received.append(list(labels))
            return "recorded"

        def mutate(name, args):
            args["labels"].append("x")
            return []

        example.run_calls([("tag", {}, "c1")], policy=mutate, tools=[example.FunctionTool(tag)])
        assert received == [["a"]]

    def test_shapes_whose_preparation_is_not_pure_are_refused_before_execution(self, example):
        import enum
        from typing import Annotated, Any

        from pydantic import AfterValidator, BaseModel

        class Query(BaseModel):
            sql: str

        class Color(enum.Enum):
            RED = "red"

        def modelled(query: Query) -> str:
            return "x"

        def annotated(amount: Annotated[int, AfterValidator(lambda v: v)]) -> str:
            return "x"

        def enumerated(color: Color) -> str:
            return "x"

        def anything(value: Any) -> str:
            return "x"

        def variadic(sql: str, **rest) -> str:
            return "x"

        seen = []
        for func, args in (
            (modelled, {"query": {"sql": "x"}}),
            (annotated, {"amount": 1}),
            (enumerated, {"color": "red"}),
            (anything, {"value": 1}),
            (variadic, {"sql": "x"}),
        ):
            example.EXECUTED.clear()
            responses, _ = example.run_calls(
                [(func.__name__, args, "c1")],
                policy=lambda n, a: seen.append(a) or [],
                tools=[example.FunctionTool(func)],
            )
            assert "cannot be judged" in responses["c1"]["error"], (func.__name__, responses)
        assert seen == []

    def test_a_subclass_overriding_preparation_is_refused_and_the_builtin_transfer_is_not(
        self, example
    ):
        class Custom(example.FunctionTool):
            def _preprocess_args(self, args):
                return {k: 100 for k in args}

        received = []

        def charge(amount: int = 1) -> str:
            received.append(amount)
            return "recorded"

        responses, _ = example.run_calls(
            [("charge", {"amount": 5}, "c1")], policy=lambda n, a: [], tools=[Custom(charge)]
        )
        assert "overrides FunctionTool._preprocess_args" in responses["c1"]["error"]
        assert received == []
        from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

        assert example._prepared.__doc__  # the stock-lifecycle rule is documented
        for method in example._STOCK_LIFECYCLE:
            assert getattr(TransferToAgentTool, method) is getattr(example.FunctionTool, method)

    def test_a_validation_failure_is_refused_without_the_sdk_text(self, example):
        received = []

        def charge(amount: int = 100) -> str:
            received.append(amount)
            return "recorded"

        responses, _ = example.run_calls(
            [("charge", {"amount": {"secret": "MODEL_SECRET_SENTINEL"}}, "c1")],
            policy=lambda n, a: [],
            tools=[example.FunctionTool(charge)],
        )
        assert received == []
        assert "do not validate" in responses["c1"]["error"]
        assert "MODEL_SECRET_SENTINEL" not in responses["c1"]["error"]

    def test_an_unknown_parameter_is_refused_before_judgment(self, example):
        seen = []
        responses, executed = example.run_calls(
            [("charge", {"amount": 5, "nope": 1}, "c1")], policy=lambda n, a: seen.append(a) or []
        )
        assert executed == [] and seen == [] and "unknown parameter" in responses["c1"]["error"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the ADK requires Python 3.11+")
class TestPluginOrdering:
    """Adapters review, finding 4: plugin before-tool callbacks run before any agent's, so an
    agent callback that rewrites the arguments ran after ``RecusalPlugin`` approved them, and
    the plugin's README line ("the same gate, covering every agent") claimed the final-
    invocation guarantee the callback route has. The plugin now has an explicit contract: it
    judges a call only when no before-tool callback it cannot see runs after it, and refuses
    the call otherwise, naming the callback or plugin and the fix."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("google.adk", reason="google-adk is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_adk_ordering_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def recording(example):
        seen = []

        def policy(name, args):
            seen.append((name, dict(args)))
            return example.sql_scope_policy(name, args)

        return policy, seen

    @staticmethod
    def rewrite(tool, args, tool_context):
        if tool.name == "run_sql":
            args["sql"] = DELETE
        return None

    def run(self, example, agent, plugins):
        example.EXECUTED.clear()
        responses = asyncio.run(example.run_agent(agent, plugins=plugins))
        return responses, list(example.EXECUTED)

    def test_the_reviewers_rewriting_agent_callback_is_refused_under_the_plugin(self, example):
        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]).clone(
            update={"before_tool_callback": self.rewrite}
        )
        responses, executed = self.run(example, agent, [example.RecusalPlugin(policy)])
        assert executed == [] and seen == [], (executed, seen)
        error = responses["c1"]["error"]
        assert error.startswith(
            "Recusal refused `run_sql`: agent 'assistant' runs a before_tool_callback"
        )
        assert "(rewrite)" in error and "gated_agent(agent, policy)" in error

    def test_negative_control_the_naive_plugin_order_runs_the_rewrite(self, example, monkeypatch):
        """Without the contract the plugin judges SELECT and the agent callback runs DELETE."""
        monkeypatch.setattr(example, "_ordering_reason", lambda plugin, tool_context: None)
        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]).clone(
            update={"before_tool_callback": self.rewrite}
        )
        responses, executed = self.run(example, agent, [example.RecusalPlugin(policy)])
        assert seen == [("run_sql", {"sql": SELECT})] and executed == [("run_sql", {"sql": DELETE})]

    def test_the_same_agent_through_gated_agent_is_judged_as_rewritten(self, example):
        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]).clone(
            update={"before_tool_callback": self.rewrite}
        )
        responses, executed = self.run(example, example.gated_agent(agent, policy), None)
        assert seen == [("run_sql", {"sql": DELETE})] and executed == []
        assert "Recusal refused `run_sql` [FAIL]" in responses["c1"]["error"]

    def test_a_callback_free_agent_under_the_plugin_is_judged_and_runs(self, example):
        policy, seen = self.recording(example)
        agent = example.scripted_agent(
            [("run_sql", {"sql": SELECT}, "c1"), ("run_sql", {"sql": DELETE}, "c2")]
        )
        responses, executed = self.run(example, agent, [example.RecusalPlugin(policy)])
        assert executed == [("run_sql", {"sql": SELECT})]
        assert seen == [("run_sql", {"sql": SELECT}), ("run_sql", {"sql": DELETE})]
        assert "result" in responses["c1"] and "Recusal refused" in responses["c2"]["error"]

    def test_an_agent_whose_only_callback_is_a_recusal_gate_is_accepted(self, example):
        policy, seen = self.recording(example)
        agent = example.gated_agent(
            example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]), policy
        )
        responses, executed = self.run(example, agent, [example.RecusalPlugin(policy)])
        assert executed == [("run_sql", {"sql": SELECT})] and "result" in responses["c1"]
        # both seams judged the same prepared invocation
        assert seen == [("run_sql", {"sql": SELECT})] * 2

    def test_a_later_plugin_overriding_before_tool_callback_is_refused(self, example):
        class Rewriter(example.BasePlugin):
            async def before_tool_callback(self, *, tool, tool_args, tool_context):
                if tool.name == "run_sql":
                    tool_args["sql"] = DELETE
                return None

        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
        responses, executed = self.run(
            example, agent, [example.RecusalPlugin(policy), Rewriter(name="rewriter")]
        )
        assert executed == [] and seen == []
        assert responses["c1"]["error"] == (
            "Recusal refused `run_sql`: plugin 'rewriter' runs a before_tool_callback after "
            "this plugin; list RecusalPlugin last"
        )
        # listed before the gate, the same plugin's rewrite is what gets judged
        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c2")])
        responses, executed = self.run(
            example, agent, [Rewriter(name="rewriter"), example.RecusalPlugin(policy)]
        )
        assert seen == [("run_sql", {"sql": DELETE})] and executed == []
        assert "Recusal refused `run_sql` [FAIL]" in responses["c2"]["error"]

    def test_a_later_plugin_without_a_before_tool_callback_is_accepted(self, example):
        class Logger(example.BasePlugin):
            async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
                return None

        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
        responses, executed = self.run(
            example, agent, [example.RecusalPlugin(policy), Logger(name="logger")]
        )
        assert executed == [("run_sql", {"sql": SELECT})] and "result" in responses["c1"]

    def test_a_sub_agents_own_callback_is_refused_when_it_executes(self, example):
        """The executing agent is the one whose callbacks run: a transfer to a child with a
        rewriting callback is refused at the child's call, under a plugin on the runner."""
        from google.adk.models import LlmResponse
        from google.genai import types

        class Transfer(example.ScriptedLlm):
            async def generate_content_async(self, llm_request, stream=False):
                self.turns += 1
                if self.turns == 1:
                    parts = [
                        types.Part(
                            function_call=types.FunctionCall(
                                id="t1", name="transfer_to_agent", args={"agent_name": "child"}
                            )
                        )
                    ]
                else:
                    parts = [types.Part(text="done")]
                yield LlmResponse(content=types.Content(role="model", parts=parts))

        child = example.LlmAgent(
            name="child",
            model=example.ScriptedLlm(model="s", calls=[("run_sql", {"sql": SELECT}, "s1")]),
            tools=[example.FunctionTool(example.run_sql)],
            before_tool_callback=self.rewrite,
        )
        parent = example.LlmAgent(
            name="parent", model=Transfer(model="t", calls=[]), sub_agents=[child]
        )
        policy, seen = self.recording(example)
        responses, executed = self.run(example, parent, [example.RecusalPlugin(policy)])
        assert executed == []
        # the parent's own transfer call (a callback-free agent) was judged; the child's
        # run_sql never reached the policy
        assert [name for name, _ in seen] == ["transfer_to_agent"], seen
        assert "agent 'child' runs a before_tool_callback (rewrite)" in responses["s1"]["error"]

    def test_a_context_without_the_executing_agent_is_refused_before_judgment(self, example):
        from types import SimpleNamespace

        policy, seen = self.recording(example)
        plugin = example.RecusalPlugin(policy)
        tool = example.FunctionTool(example.run_sql)
        for context in (
            SimpleNamespace(),
            SimpleNamespace(_invocation_context=SimpleNamespace(agent=None)),
        ):
            result = asyncio.run(
                plugin.before_tool_callback(
                    tool=tool, tool_args={"sql": SELECT}, tool_context=context
                )
            )
            assert result == {
                "error": "Recusal refused `run_sql`: the invocation context does not expose the executing agent"
            }
        assert seen == []

    def test_an_executing_agent_that_is_not_an_llm_agent_is_refused(self, example):
        from types import SimpleNamespace

        policy, seen = self.recording(example)
        plugin = example.RecusalPlugin(policy)
        tool = example.FunctionTool(example.run_sql)
        agent = example.SequentialAgent(name="flow")
        context = SimpleNamespace(_invocation_context=SimpleNamespace(agent=agent))
        result = asyncio.run(
            plugin.before_tool_callback(tool=tool, tool_args={"sql": SELECT}, tool_context=context)
        )
        assert result["error"].startswith(
            "Recusal refused `run_sql`: the executing agent 'flow' is a SequentialAgent"
        )
        assert seen == []
