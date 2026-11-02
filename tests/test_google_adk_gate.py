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
        # the ADK normalizes a missing args field to {}, judges it, then the tool itself
        # refuses the missing parameter; nothing ran
        assert seen == [("echo", {})]
        assert executed == []
        assert "error" in responses["c1"]

    def test_non_mapping_arguments_are_refused_by_the_gate_itself(self, example):
        assert example._refusal("run_sql", ["DELETE"], example.sql_scope_policy) == (
            "Recusal refused `run_sql`: arguments are not a mapping"
        )
        assert example._refusal("run_sql", "DELETE", example.sql_scope_policy) is not None

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
