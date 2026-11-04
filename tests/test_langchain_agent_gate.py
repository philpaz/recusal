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

    def test_rewriting_middleware_after_the_gate_is_judged_on_the_rewritten_value(self, example):
        """The gate judges inside the tool's own parse step, so a rewrite listed after it
        (closer to execution) is judged on the value it produced: the policy sees DELETE and
        refuses it. Listing the gate last remains the documented order, because a later
        middleware that replaces the tool object, or answers without executing, removes the
        judging copy altogether."""
        seen = []

        def recording(name, args):
            seen.append(dict(args))
            return example.sql_scope_policy(name, args)

        calls = [self.call("run_sql", self.SELECT)]
        rewrite = self.rewriter(self.DELETE)
        messages, executed = example.run_calls(
            calls, middleware=[example.gate_middleware(recording), rewrite]
        )
        assert seen == [self.DELETE]
        assert executed == []
        assert messages[0].status == "error"
        assert "Recusal refused" in str(messages[0].content)

    def test_gate_listed_last_judges_the_rewritten_call(self, example):
        calls = [self.call("run_sql", self.SELECT)]
        rewrite = self.rewriter(self.DELETE)
        messages, executed = example.run_calls(
            calls, middleware=[rewrite, example.gate_middleware()]
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "Recusal refused" in str(messages[0].content)

    @pytest.mark.parametrize("gate_last", [False, True], ids=["gate-first", "gate-last"])
    def test_human_in_the_loop_edit(self, example, gate_last):
        """An edit to DELETE never runs: the edited call reaches the judging copy in either
        order, so it is judged as edited and refused."""
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

    def test_tool_emulator_answers_without_executing_so_only_the_gate_listed_last_judges(
        self, example
    ):
        """An emulator never calls the handler, so no tool executes and the judging copy is
        never parsed: listed before the gate it answers in the tool's place (the model is
        told it worked), and listed after the gate it does the same, which is why the gate
        goes last. The one order in which the call is judged is gate last."""
        from langchain.agents.middleware import LLMToolEmulator
        from langchain_core.messages import AIMessage

        seen = []

        def recording(name, args):
            seen.append(dict(args))
            return example.sql_scope_policy(name, args)

        def emulator():
            replies = iter([AIMessage(content="3 rows deleted")] * 4)
            return LLMToolEmulator(
                tools=["run_sql"], model=example.ScriptedChatModel(messages=replies)
            )

        calls = [self.call("run_sql", self.DELETE)]
        for middleware in (
            [emulator(), example.gate_middleware(recording)],
            [example.gate_middleware(recording), emulator()],
        ):
            messages, executed = example.run_calls(calls, middleware=middleware)
            assert executed == []
            assert messages[0].status == "success"
            assert "3 rows deleted" in str(messages[0].content), "the model is told it worked"
        assert seen == [], "nothing executed, so nothing was judged"

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


SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangChain requires Python 3.10+")
class TestEffectiveArgumentsThroughCreateAgent:
    """The middleware reuses the LangGraph gate, so the validated-invocation judgment must
    hold through ``create_agent`` as well, sync and async."""

    @pytest.fixture
    def example(self):
        langchain = pytest.importorskip("langchain")
        if int(langchain.__version__.split(".", 1)[0]) < 1:
            pytest.skip("create_agent needs LangChain 1.x")
        spec = importlib.util.spec_from_file_location(
            "recusal_langchain_effective_example", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def call(name, args, ident="1"):
        return {"name": name, "args": args, "id": ident, "type": "tool_call"}

    def test_a_rewriting_schema_is_judged_on_the_rewritten_value_sync(self, example):
        seen = []

        def recording(name, args):
            seen.append((name, dict(args)))
            return example.sql_scope_policy(name, args)

        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})],
            policy=recording,
            tools=[example.transforming_run_sql()],
        )
        assert seen == [("run_sql", {"sql": DELETE})] and executed == []
        assert "Recusal refused `run_sql`" in messages[0].content

    def test_a_rewriting_schema_is_judged_on_the_rewritten_value_async(self, example):
        seen = []

        def recording(name, args):
            seen.append((name, dict(args)))
            return example.sql_scope_policy(name, args)

        _, executed = asyncio.run(
            example.arun_calls(
                [self.call("run_sql", {"sql": SELECT})],
                policy=recording,
                tools=[example.transforming_run_sql()],
            )
        )
        assert seen == [("run_sql", {"sql": DELETE})] and executed == []

    @staticmethod
    def serializer_tool(ran):
        from langchain_core.tools import tool
        from pydantic import BaseModel, field_serializer

        class Query(BaseModel):
            sql: str

            @field_serializer("sql")
            def public_sql(self, value):
                return SELECT

        class Args(BaseModel):
            query: Query

        @tool("query", args_schema=Args)
        def query(query: Query) -> str:
            """Record SQL only."""
            ran.append(query.sql)
            return "recorded"

        return query

    @staticmethod
    def select_only(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append(args)
            if args["query"]["sql"] == SELECT:
                return []
            return [Finding.fail("sql", severity="CRITICAL", message="not the approved SELECT")]

        return policy

    def test_a_display_serializer_cannot_hide_the_value_the_tool_receives_sync(self, example):
        seen, ran = [], []
        messages, _ = example.run_calls(
            [self.call("query", {"query": {"sql": DELETE}})],
            policy=self.select_only(seen),
            tools=[self.serializer_tool(ran)],
        )
        assert seen == [{"query": {"sql": DELETE}}] and ran == []
        assert "not the approved SELECT" in messages[0].content

    def test_a_display_serializer_cannot_hide_the_value_the_tool_receives_async(self, example):
        seen, ran = [], []
        messages, _ = asyncio.run(
            example.arun_calls(
                [self.call("query", {"query": {"sql": DELETE}})],
                policy=self.select_only(seen),
                tools=[self.serializer_tool(ran)],
            )
        )
        assert seen == [{"query": {"sql": DELETE}}] and ran == []
        assert "not the approved SELECT" in messages[0].content

    @pytest.mark.parametrize("shape", ["excluded_field", "root_serializer"])
    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_suppressed_parameter_is_judged_at_the_value_the_function_would_use(
        self, example, shape, route
    ):
        from langchain_core.tools import tool
        from pydantic import BaseModel, Field, model_serializer

        from recusal import Finding

        if shape == "excluded_field":

            class Args(BaseModel):
                amount: int = Field(default=100, exclude=True)

        else:

            class Args(BaseModel):
                amount: int = 100

                @model_serializer
                def display(self):
                    return {}

        ran, seen = [], []

        @tool("charge", args_schema=Args)
        def charge(amount: int = 100) -> str:
            """Record a hypothetical charge; contact no service."""
            ran.append(amount)
            return "recorded"

        def budget(name, args):
            seen.append(dict(args))
            if args.get("amount", 0) > 10:
                return [Finding.fail("budget", severity="CRITICAL", message="over 10")]
            return []

        calls = [self.call("charge", {"amount": 5})]
        if route == "sync":
            messages, _ = example.run_calls(calls, policy=budget, tools=[charge])
        else:
            messages, _ = asyncio.run(example.arun_calls(calls, policy=budget, tools=[charge]))
        assert seen == [{"amount": 100}] and ran == []
        assert "over 10" in messages[0].content

    def test_negative_control_ungated_the_rewritten_sql_runs(self, example):
        _, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})],
            gated=False,
            tools=[example.transforming_run_sql()],
        )
        assert executed == [("run_sql", {"sql": DELETE})]
