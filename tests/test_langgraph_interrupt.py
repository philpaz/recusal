"""LangGraph pause, resume and replay: every resumed or replayed call is judged again.

The gate in ``examples/langgraph_gate.py`` runs at ToolNode execution time. A LangGraph agent
can stop before that point (a review node calls ``interrupt()``), stop inside it (the tool
itself calls ``interrupt()``), and come back later through ``Command(resume=...)``; a
checkpointed thread can also be replayed or forked from an earlier state. These tests pin that
a decision made before a pause never becomes permission that outlives it: the gate judges the
call as it will run, under the policy in force at that moment, and a replay is a new execution
judged again. One pattern is outside a pre-tool gate and is pinned as a known limitation: a
value supplied on resume and consumed inside the tool.

Every refusal below is shown with zero executions, not only an error message.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "langgraph_gate.py"

SELECT = "SELECT * FROM t"
DELETE = "DELETE FROM t"


def test_interrupt_tests_are_in_the_langgraph_ci_jobs():
    """This file joins the pinned LangGraph jobs and the weekly latest lane; no new job."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert ci.count("tests/test_langgraph_interrupt.py") == 2
    compat = (ROOT / ".github" / "workflows" / "compat.yml").read_text(encoding="utf-8")
    assert compat.count("tests/test_langgraph_interrupt.py") == 2


def test_docs_explain_pause_resume_and_replay():
    for path in (ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        text = path.read_text(encoding="utf-8")
        assert "judged again" in text, path
        assert "resume" in text, path


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestPauseResumeReplay:
    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        spec = importlib.util.spec_from_file_location(
            "recusal_langgraph_interrupt_example", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.EXECUTED.clear()
        return module

    @pytest.fixture
    def lg(self):
        """The LangGraph pieces the example does not import: checkpoints and interrupts."""
        pytest.importorskip("langgraph")
        from types import SimpleNamespace

        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition
        from langgraph.types import Command, interrupt

        return SimpleNamespace(
            AIMessage=AIMessage,
            tool=tool,
            InMemorySaver=InMemorySaver,
            START=START,
            MessagesState=MessagesState,
            StateGraph=StateGraph,
            ToolNode=ToolNode,
            tools_condition=tools_condition,
            Command=Command,
            interrupt=interrupt,
        )

    # --- fixtures: a policy that records every judgement and can be swapped or expired ---

    class Judge:
        """Wraps the policy the gate consults; records every judgement in order."""

        def __init__(self, example):
            self.example = example
            self.policy = example.sql_scope_policy
            self.judged: list[tuple[str, dict[str, Any]]] = []

        def __call__(self, tool_name: str, tool_input: dict) -> list:
            self.judged.append((tool_name, dict(tool_input)))
            return self.policy(tool_name, tool_input)

        def revoke(self) -> None:
            from recusal.evidence import Finding

            def deny_all(tool_name: str, tool_input: dict) -> list:
                return [Finding("grant", "critical", False, "grant revoked")]

            self.policy = deny_all

        def sqls(self) -> list[str]:
            return [args["sql"] for _, args in self.judged]

    @staticmethod
    def call(sql: str, name: str = "run_sql", ident: str = "c1") -> dict[str, Any]:
        return {"name": name, "args": {"sql": sql}, "id": ident, "type": "tool_call"}

    @staticmethod
    def thread(name: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": name}}

    @staticmethod
    def tool_messages(result: dict[str, Any]) -> list:
        from langchain_core.messages import ToolMessage

        return [m for m in result["messages"] if isinstance(m, ToolMessage)]

    def review_graph(self, example, lg, judge, *, gated: bool = True, tools=None):
        """agent -> review (interrupt before the ToolNode) -> tools -> agent.

        The reviewer's resume value is ``"approve"`` or an edited tool_calls list, which the
        review node writes back over the model message, so the ToolNode runs the edited call.
        """
        script = iter([lg.AIMessage(content="", tool_calls=[self.call(SELECT)])])

        def agent(state):
            return {"messages": [next(script, lg.AIMessage(content="done"))]}

        def review(state):
            last = state["messages"][-1]
            verdict = lg.interrupt({"tool_calls": last.tool_calls})
            if verdict == "approve":
                return {}
            return {"messages": [lg.AIMessage(content="", tool_calls=verdict, id=last.id)]}

        tools = [example.echo, example.run_sql] if tools is None else tools
        node = (
            lg.ToolNode(tools, wrap_tool_call=example.recusal_gate(judge))
            if gated
            else lg.ToolNode(tools)
        )
        builder = lg.StateGraph(lg.MessagesState)
        builder.add_node("agent", agent)
        builder.add_node("review", review)
        builder.add_node("tools", node)
        builder.add_edge(lg.START, "agent")
        builder.add_conditional_edges(
            "agent", lg.tools_condition, {"tools": "review", "__end__": "__end__"}
        )
        builder.add_edge("review", "tools")
        builder.add_edge("tools", "agent")
        return builder.compile(checkpointer=lg.InMemorySaver())

    def tool_interrupt_graph(self, example, lg, judge, tool_obj, *, gated: bool = True):
        """agent -> tools -> agent, where the tool itself pauses with ``interrupt()``."""
        script = iter([lg.AIMessage(content="", tool_calls=[self.call(SELECT, tool_obj.name)])])

        def agent(state):
            return {"messages": [next(script, lg.AIMessage(content="done"))]}

        node = (
            lg.ToolNode([tool_obj], wrap_tool_call=example.recusal_gate(judge))
            if gated
            else lg.ToolNode([tool_obj])
        )
        builder = lg.StateGraph(lg.MessagesState)
        builder.add_node("agent", agent)
        builder.add_node("tools", node)
        builder.add_edge(lg.START, "agent")
        builder.add_conditional_edges("agent", lg.tools_condition)
        builder.add_edge("tools", "agent")
        return builder.compile(checkpointer=lg.InMemorySaver())

    # --- A: a review node pauses before the ToolNode -------------------------------------

    def test_paused_run_has_judged_nothing_and_run_nothing(self, example, lg):
        judge = self.Judge(example)
        graph = self.review_graph(example, lg, judge)
        paused = graph.invoke({"messages": []}, self.thread("a0"))
        assert "__interrupt__" in paused
        assert judge.judged == []
        assert example.EXECUTED == []

    def test_approved_call_is_judged_once_on_resume_and_runs(self, example, lg):
        judge = self.Judge(example)
        graph = self.review_graph(example, lg, judge)
        graph.invoke({"messages": []}, self.thread("a1"))
        result = graph.invoke(lg.Command(resume="approve"), self.thread("a1"))
        assert judge.sqls() == [SELECT]
        assert example.EXECUTED == [("run_sql", {"sql": SELECT})]
        assert [m.status for m in self.tool_messages(result)] == ["success"]

    def test_reviewer_edit_to_delete_is_judged_as_edited_and_refused(self, example, lg):
        judge = self.Judge(example)
        graph = self.review_graph(example, lg, judge)
        graph.invoke({"messages": []}, self.thread("a2"))
        result = graph.invoke(lg.Command(resume=[self.call(DELETE)]), self.thread("a2"))
        assert judge.sqls() == [DELETE], "the gate must see the call as it will run"
        assert example.EXECUTED == []
        (message,) = self.tool_messages(result)
        assert message.status == "error"
        assert message.tool_call_id == "c1"
        assert "Recusal refused" in str(message.content)

    def test_policy_swapped_between_pause_and_resume_is_honoured(self, example, lg):
        judge = self.Judge(example)
        graph = self.review_graph(example, lg, judge)
        graph.invoke({"messages": []}, self.thread("a3"))
        judge.revoke()
        result = graph.invoke(lg.Command(resume="approve"), self.thread("a3"))
        assert judge.sqls() == [SELECT]
        assert example.EXECUTED == []
        assert [m.status for m in self.tool_messages(result)] == ["error"]

    def test_grant_that_expires_before_resume_refuses(self, example, lg):
        """A time-boxed grant: valid when the run paused, expired when it resumed."""
        from recusal.evidence import Finding

        clock = {"now": 100}
        judge = self.Judge(example)
        base = judge.policy

        def timed(tool_name: str, tool_input: dict) -> list:
            if clock["now"] > 120:
                return [Finding("grant", "critical", False, "grant expired")]
            return base(tool_name, tool_input)

        judge.policy = timed
        graph = self.review_graph(example, lg, judge)
        graph.invoke({"messages": []}, self.thread("a5"))
        clock["now"] = 200
        result = graph.invoke(lg.Command(resume="approve"), self.thread("a5"))
        assert judge.sqls() == [SELECT]
        assert example.EXECUTED == []
        (message,) = self.tool_messages(result)
        assert message.status == "error"
        assert "grant expired" in str(message.content)

    def test_negative_control_ungated_review_graph_runs_the_edited_delete(self, example, lg):
        judge = self.Judge(example)
        graph = self.review_graph(example, lg, judge, gated=False)
        graph.invoke({"messages": []}, self.thread("a4"))
        graph.invoke(lg.Command(resume=[self.call(DELETE)]), self.thread("a4"))
        assert judge.judged == []
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]

    # --- B: the tool itself pauses -------------------------------------------------------

    def reviewed_tool(self, example, lg):
        @lg.tool
        def run_sql_reviewed(sql: str) -> str:
            """Record SQL once a human approves it from inside the tool."""
            if lg.interrupt({"sql": sql}) != "approve":
                return "rejected by reviewer"
            example.EXECUTED.append(("run_sql_reviewed", {"sql": sql}))
            return f"recorded SQL: {sql}"

        return run_sql_reviewed

    def test_tool_interrupt_is_judged_before_the_pause_and_again_on_resume(self, example, lg):
        judge = self.Judge(example)
        graph = self.tool_interrupt_graph(example, lg, judge, self.reviewed_tool(example, lg))
        paused = graph.invoke({"messages": []}, self.thread("b1"))
        assert "__interrupt__" in paused
        assert judge.sqls() == [SELECT], "judged before the pause"
        assert example.EXECUTED == []
        result = graph.invoke(lg.Command(resume="approve"), self.thread("b1"))
        assert judge.sqls() == [SELECT, SELECT], "judged again on resume"
        assert example.EXECUTED == [("run_sql_reviewed", {"sql": SELECT})], "one execution"
        assert [m.status for m in self.tool_messages(result)] == ["success"]

    def test_tool_interrupt_honours_a_policy_swapped_during_the_pause(self, example, lg):
        judge = self.Judge(example)
        graph = self.tool_interrupt_graph(example, lg, judge, self.reviewed_tool(example, lg))
        graph.invoke({"messages": []}, self.thread("b2"))
        judge.revoke()
        result = graph.invoke(lg.Command(resume="approve"), self.thread("b2"))
        assert judge.sqls() == [SELECT, SELECT]
        assert example.EXECUTED == []
        (message,) = self.tool_messages(result)
        assert message.status == "error"
        assert "grant revoked" in str(message.content)

    def test_known_limitation_resume_value_consumed_inside_the_tool_is_not_judged(
        self, example, lg
    ):
        """KNOWN LIMITATION, pinned on purpose.

        When a tool takes its effective input from the resume value, the gate only ever judged
        the model's arguments: the value arrives inside the tool, after the gate has run. This
        test asserts the DELETE runs. If it ever fails, LangGraph has started routing resume
        values through the ToolNode hook, and the documentation must change with it. Route
        reviewer edits through a review node before the ToolNode instead (the A tests).
        """

        @lg.tool
        def run_sql_from_resume(sql: str) -> str:
            """Record the SQL the reviewer supplies on resume, else the model's."""
            supplied = lg.interrupt({"sql": sql})
            effective = supplied if isinstance(supplied, str) else sql
            example.EXECUTED.append(("run_sql_from_resume", {"sql": effective}))
            return f"recorded SQL: {effective}"

        judge = self.Judge(example)
        graph = self.tool_interrupt_graph(example, lg, judge, run_sql_from_resume)
        graph.invoke({"messages": []}, self.thread("b3"))
        graph.invoke(lg.Command(resume=DELETE), self.thread("b3"))
        assert judge.sqls() == [SELECT, SELECT], "the gate saw only the model's arguments"
        assert example.EXECUTED == [("run_sql_from_resume", {"sql": DELETE})]

    # --- replay and fork from a checkpoint ------------------------------------------------

    def approved_run(self, example, lg, judge, name: str, *, gated: bool = True):
        """Run the review graph to completion with approval; return graph and the
        checkpoint whose next node is the ToolNode."""
        graph = self.review_graph(example, lg, judge, gated=gated)
        graph.invoke({"messages": []}, self.thread(name))
        first = graph.invoke(lg.Command(resume="approve"), self.thread(name))
        before_tools = next(
            s for s in graph.get_state_history(self.thread(name)) if s.next == ("tools",)
        )
        return graph, first, before_tools

    def test_replay_from_a_checkpoint_is_a_new_execution_judged_again(self, example, lg):
        judge = self.Judge(example)
        graph, first, before_tools = self.approved_run(example, lg, judge, "r1")
        assert judge.sqls() == [SELECT]
        assert example.EXECUTED == [("run_sql", {"sql": SELECT})]
        replayed = graph.invoke(None, before_tools.config)
        assert judge.sqls() == [SELECT, SELECT], "judged again"
        assert example.EXECUTED == [("run_sql", {"sql": SELECT})] * 2, "a second execution"
        (a,) = self.tool_messages(first)
        (b,) = self.tool_messages(replayed)
        assert a.tool_call_id == b.tool_call_id == "c1", "same call id, new execution"

    def test_replay_under_a_revoked_policy_is_refused(self, example, lg):
        judge = self.Judge(example)
        graph, _, before_tools = self.approved_run(example, lg, judge, "r2")
        example.EXECUTED.clear()
        judge.revoke()
        replayed = graph.invoke(None, before_tools.config)
        assert judge.sqls() == [SELECT, SELECT]
        assert example.EXECUTED == []
        assert [m.status for m in self.tool_messages(replayed)] == ["error"]

    def fork_to_delete(self, lg, graph, before_tools):
        state = graph.get_state(before_tools.config)
        last_ai = [m for m in state.values["messages"] if isinstance(m, lg.AIMessage)][-1]
        edited = lg.AIMessage(content="", tool_calls=[self.call(DELETE)], id=last_ai.id)
        forked = graph.update_state(before_tools.config, {"messages": [edited]}, as_node="review")
        return graph.invoke(None, forked)

    def test_fork_that_edits_the_call_to_delete_is_refused(self, example, lg):
        judge = self.Judge(example)
        graph, _, before_tools = self.approved_run(example, lg, judge, "r3")
        example.EXECUTED.clear()
        forked = self.fork_to_delete(lg, graph, before_tools)
        assert judge.sqls() == [SELECT, DELETE]
        assert example.EXECUTED == []
        (message,) = self.tool_messages(forked)
        assert message.status == "error"
        assert "Recusal refused" in str(message.content)

    def test_negative_control_ungated_fork_runs_the_delete(self, example, lg):
        judge = self.Judge(example)
        graph, _, before_tools = self.approved_run(example, lg, judge, "r4", gated=False)
        example.EXECUTED.clear()
        self.fork_to_delete(lg, graph, before_tools)
        assert judge.judged == []
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]
