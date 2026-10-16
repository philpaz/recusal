"""Offline regressions for LangChain create_agent using the shared Recusal gate."""

import asyncio
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "langchain_agent_gate.py"


def test_version_refusal_before_langchain_import():
    code = (
        "import runpy,sys; sys.version_info=(3,9,0,'final',0); "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "langchain_agent_gate.py requires Python 3.10+" in result.stderr


def test_langchain_stays_out_of_runtime_package():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "langchain" not in project
    for path in (ROOT / "recusal").rglob("*.py"):
        assert "langchain" not in path.read_text(encoding="utf-8").lower(), path


def test_langchain_pins_are_documented_and_in_existing_ci_jobs():
    paths = (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md")
    for path in paths:
        assert "langchain==1.4.3" in path.read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert ci.count("langchain==1.4.3") == 2


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangChain requires Python 3.10+")
class TestLangChainAgentGate:
    @pytest.fixture
    def example(self):
        langchain = pytest.importorskip("langchain")
        if int(langchain.__version__.split(".", 1)[0]) < 1:
            pytest.skip("create_agent needs LangChain 1.x")
        spec = importlib.util.spec_from_file_location("recusal_langchain_gate_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def call(name, args, ident="1"):
        return {"name": name, "args": args, "id": ident, "type": "tool_call"}

    def test_allowed_call_executes(self, example):
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": "SELECT * FROM audit_log"})]
        )
        assert executed == [("run_sql", {"sql": "SELECT * FROM audit_log"})]
        assert messages[0].status == "success"

    def test_refused_call_returns_error_without_execution(self, example):
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": "DELETE FROM audit_log"})]
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "Recusal refused" in str(messages[0].content)
        assert "top-level WHERE" in str(messages[0].content)

    def test_multiple_calls_are_gated_independently(self, example):
        calls = [
            self.call("echo", {"text": "hi"}, "e"),
            self.call("run_sql", {"sql": "DELETE FROM audit_log"}, "d"),
            self.call("run_sql", {"sql": "SELECT * FROM audit_log"}, "s"),
        ]
        messages, executed = example.run_calls(calls)
        assert {(m.tool_call_id, m.status) for m in messages} == {
            ("e", "success"),
            ("d", "error"),
            ("s", "success"),
        }
        assert sorted(name for name, _ in executed) == ["echo", "run_sql"]

    def test_broken_policy_fails_closed(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        messages, executed = example.run_calls(
            [self.call("echo", {"text": "unsafe"})], policy=broken_policy
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "failed closed (policy error)" in str(messages[0].content)

    def test_negative_control_executes_delete_without_gate(self, example):
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": "DELETE FROM audit_log"})], gated=False
        )
        assert executed == [("run_sql", {"sql": "DELETE FROM audit_log"})]
        assert messages[0].status == "success"

    # --- middleware order: the gate must be listed LAST ---------------------------------
    # In create_agent the first middleware is the outermost, so whatever is listed after
    # the gate acts on the call between the gate's check and the tool. Each case below
    # runs both orders: the gate listed first is the hazard, listed last is the fix.

    SELECT = {"sql": "SELECT * FROM audit_log"}
    DELETE = {"sql": "DELETE FROM audit_log"}

    @staticmethod
    def rewriter(args):
        from langchain.agents.middleware import wrap_tool_call

        @wrap_tool_call
        def rewrite(request, handler):
            call = dict(request.tool_call)
            call["args"] = args
            return handler(request.override(tool_call=call))

        return rewrite

    def test_rewriting_middleware_after_the_gate_is_unchecked(self, example):
        calls = [self.call("run_sql", self.SELECT)]
        rewrite = self.rewriter(self.DELETE)
        messages, executed = example.run_calls(
            calls, middleware=[example.gate_middleware(), rewrite]
        )
        assert executed == [("run_sql", self.DELETE)]
        assert messages[0].status == "success"

    def test_gate_listed_last_judges_the_rewritten_call(self, example):
        calls = [self.call("run_sql", self.SELECT)]
        rewrite = self.rewriter(self.DELETE)
        messages, executed = example.run_calls(
            calls, middleware=[rewrite, example.gate_middleware()]
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "Recusal refused" in str(messages[0].content)

    @pytest.mark.parametrize(("gate_last", "ran"), [(False, True), (True, False)])
    def test_human_in_the_loop_edit(self, example, gate_last, ran):
        from langchain.agents.middleware import HumanInTheLoopMiddleware
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        example.EXECUTED.clear()
        hitl = HumanInTheLoopMiddleware(interrupt_on={"run_sql": True})
        gate = example.gate_middleware()
        middleware = [hitl, gate] if gate_last else [gate, hitl]
        agent = example.scripted_agent(
            [self.call("run_sql", self.SELECT)], middleware, checkpointer=InMemorySaver()
        )
        config = {"configurable": {"thread_id": f"hitl-{gate_last}"}}
        first = agent.invoke({"messages": [HumanMessage(content="go")]}, config)
        assert "__interrupt__" in first, "the reviewer must be asked before the tool runs"
        decision = {"type": "edit", "edited_action": {"name": "run_sql", "args": self.DELETE}}
        result = agent.invoke(Command(resume={"decisions": [decision]}), config)
        messages = example.tool_messages(result)
        if ran:
            assert example.EXECUTED == [("run_sql", self.DELETE)]
            assert messages[-1].status == "success"
        else:
            assert example.EXECUTED == []
            assert messages[-1].status == "error"
            assert "Recusal refused" in str(messages[-1].content)

    def test_human_in_the_loop_edit_to_another_tool_is_judged_when_gate_is_last(self, example):
        from langchain.agents.middleware import HumanInTheLoopMiddleware
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        for gate_last, expect_ran in ((False, True), (True, False)):
            example.EXECUTED.clear()
            hitl = HumanInTheLoopMiddleware(interrupt_on={"echo": True})
            gate = example.gate_middleware()
            middleware = [hitl, gate] if gate_last else [gate, hitl]
            agent = example.scripted_agent(
                [self.call("echo", {"text": "hi"})], middleware, checkpointer=InMemorySaver()
            )
            config = {"configurable": {"thread_id": f"rename-{gate_last}"}}
            agent.invoke({"messages": [HumanMessage(content="go")]}, config)
            decision = {
                "type": "edit",
                "edited_action": {"name": "run_sql", "args": self.DELETE},
            }
            agent.invoke(Command(resume={"decisions": [decision]}), config)
            ran = ("run_sql", self.DELETE) in example.EXECUTED
            assert ran is expect_ran, f"gate_last={gate_last}"

    def test_tool_emulator_before_the_gate_skips_it(self, example):
        from langchain.agents.middleware import LLMToolEmulator
        from langchain_core.messages import AIMessage

        def emulator():
            replies = iter([AIMessage(content="3 rows deleted")] * 4)
            return LLMToolEmulator(
                tools=["run_sql"], model=example.ScriptedChatModel(messages=replies)
            )

        calls = [self.call("run_sql", self.DELETE)]
        messages, executed = example.run_calls(
            calls, middleware=[emulator(), example.gate_middleware()]
        )
        assert executed == []
        assert messages[0].status == "success"
        assert "3 rows deleted" in str(messages[0].content), "the model is told it worked"

        messages, executed = example.run_calls(
            calls, middleware=[example.gate_middleware(), emulator()]
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "Recusal refused" in str(messages[0].content)

    # --- async: ainvoke needs the async gate --------------------------------------------

    def test_ainvoke_with_only_the_sync_gate_raises_and_runs_nothing(self, example):
        calls = [self.call("run_sql", self.SELECT)]
        with pytest.raises(NotImplementedError):
            asyncio.run(example.arun_calls(calls, middleware=[example.gate_middleware()]))
        assert example.EXECUTED == []

    def test_ainvoke_with_the_async_gate_refuses_and_allows(self, example):
        calls = [
            self.call("run_sql", self.DELETE, "d"),
            self.call("run_sql", self.SELECT, "s"),
        ]
        messages, executed = asyncio.run(example.arun_calls(calls))
        assert {(m.tool_call_id, m.status) for m in messages} == {
            ("d", "error"),
            ("s", "success"),
        }
        assert executed == [("run_sql", self.SELECT)]
