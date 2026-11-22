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
        assert "langchain==1.4.10" in path.read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # the pin is a root of the hash-locked file the pull request lanes install: the two
    # example jobs and the recusal-langchain wheel job
    assert (
        "langchain==1.4.10"
        in (ROOT / "requirements" / "adapter-langchain.txt")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .split()
    )
    assert ci.count("requirements/adapter-langchain.txt") == 3


def test_the_package_requires_the_langgraph_gate_it_wraps():
    """The middleware only wraps ``recusal-langgraph``'s hooks. What it promises (a call
    that names another tool than the one that runs is refused, an answer that never passed
    the judging step is withheld) is that package's from 0.3.0, so that is the floor, stated
    in the package and in its README; the gate in this tree is at or past it."""
    import re

    floor = "0.3.0"
    pyproject = (ROOT / "adapters" / "langchain" / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "adapters" / "langchain" / "README.md").read_text(encoding="utf-8")
    assert f'"recusal-langgraph>={floor},<1"' in pyproject
    assert f"`recusal-langgraph>={floor}`" in readme
    assert '"recusal>=0.15.0,<1"' in pyproject and "`recusal>=0.15.0`" in readme
    source = (
        ROOT / "adapters" / "langgraph" / "src" / "recusal_langgraph" / "__init__.py"
    ).read_text(encoding="utf-8")
    version = re.search(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"$', source, re.MULTILINE)
    assert version is not None
    assert tuple(int(part) for part in version.groups()) >= (0, 3, 0)


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
            result = agent.invoke(Command(resume={"decisions": [decision]}), config)
            ran = ("run_sql", self.DELETE) in example.EXECUTED
            assert ran is expect_ran, f"gate_last={gate_last}"
            # listed first, the gate's judging copy was swapped out and the other tool ran;
            # the gate sees the answer come back unjudged and withholds it
            answer = example.tool_messages(result)[-1]
            assert answer.status == "error", f"gate_last={gate_last}"
            if not gate_last:
                assert "without passing the gate's judging step" in str(answer.content)

    def test_tool_emulator_answers_without_executing_so_only_the_gate_listed_last_judges(
        self, example
    ):
        """An emulator never calls the handler, so no tool executes and the judging copy is
        never parsed: listed before the gate it answers in the tool's place (the model is
        told it worked) and the gate is never reached. Listed after the gate it answers the
        same way, and the gate, whose judging step never ran, withholds that answer and
        refuses. Neither order judges the call; no tool runs in either."""
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
        messages, executed = example.run_calls(
            calls, middleware=[emulator(), example.gate_middleware(recording)]
        )
        assert executed == []
        assert messages[0].status == "success"
        assert "3 rows deleted" in str(messages[0].content), "the model is told it worked"

        messages, executed = example.run_calls(
            calls, middleware=[example.gate_middleware(recording), emulator()]
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "3 rows deleted" not in str(messages[0].content), "the answer is withheld"
        assert "without passing the gate's judging step" in str(messages[0].content)
        assert seen == [], "nothing executed, so nothing was judged"

    # --- one middleware object serves invoke and ainvoke ---------------------------------

    @pytest.mark.parametrize("build", ["gate_middleware", "agate_middleware"])
    def test_one_middleware_gates_the_same_agent_under_invoke_and_under_ainvoke(
        self, example, build
    ):
        """Review round 1: the sync and the async middleware were two objects with one name,
        so an agent could not list both, and each raised on the other route. Either name now
        returns one object that implements both halves."""
        from langchain_core.messages import HumanMessage

        calls = [
            self.call("run_sql", self.DELETE, "d"),
            self.call("run_sql", self.SELECT, "s"),
        ]
        start = {"messages": [HumanMessage(content="go")]}
        middleware = [getattr(example, build)()]
        expected = {("d", "error"), ("s", "success")}

        example.EXECUTED.clear()
        result = example.scripted_agent(calls, middleware).invoke(start)
        assert {(m.tool_call_id, m.status) for m in example.tool_messages(result)} == expected
        assert example.EXECUTED == [("run_sql", self.SELECT)]

        example.EXECUTED.clear()
        result = asyncio.run(example.scripted_agent(calls, middleware).ainvoke(start))
        assert {(m.tool_call_id, m.status) for m in example.tool_messages(result)} == expected
        assert example.EXECUTED == [("run_sql", self.SELECT)]

    def test_each_half_of_the_middleware_is_the_gate_for_its_route(self, example):
        """Neither half is LangChain's default, which raises ``NotImplementedError``; and
        neither is the other route's hook: the sync half holds the sync ToolNode hook and
        the async half the async one, so each judges the callable its route runs."""
        import inspect

        from langchain.agents.middleware import AgentMiddleware

        middleware = example.gate_middleware()
        cls = type(middleware)
        assert cls.wrap_tool_call is not AgentMiddleware.wrap_tool_call
        assert cls.awrap_tool_call is not AgentMiddleware.awrap_tool_call
        assert not inspect.iscoroutinefunction(middleware._sync_hook)
        assert inspect.iscoroutinefunction(middleware._async_hook)

    def test_two_gates_cannot_be_listed_together(self, example, tmp_path):
        """Every form has one name, so ``create_agent`` refuses a list that holds two: a
        call would otherwise be judged twice, or recorded by one and not the other."""
        from langchain.agents import create_agent
        from recusal_langchain import audited_gate_middleware

        from recusal.audit import AuditLog

        gate = example.gate_middleware()
        assert gate.name == example.agate_middleware().name == "recusal_gate"
        audited = audited_gate_middleware(lambda name, args: [], AuditLog(path=str(tmp_path / "a")))
        assert audited.name == "recusal_gate"
        model = example.ScriptedChatModel(messages=iter([]))
        for pair in ([gate, example.agate_middleware()], [gate, audited]):
            with pytest.raises(AssertionError, match="duplicate middleware"):
                create_agent(model=model, tools=[example.run_sql], middleware=pair)

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

    def test_a_coroutine_only_tool_runs_gated_under_ainvoke_and_is_refused_under_invoke(
        self, example
    ):
        """The async half judges the coroutine; the sync half has no callable to judge on
        its route, and refuses rather than run the tool unjudged."""
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import StructuredTool

        ran = []

        async def fetch(url: str) -> str:
            ran.append(url)
            return "ok"

        tool = StructuredTool.from_function(coroutine=fetch, name="fetch", description="Fetch.")
        calls = [self.call("fetch", {"url": "u"})]
        start = {"messages": [HumanMessage(content="go")]}
        middleware = [example.gate_middleware(lambda name, args: [])]
        result = asyncio.run(example.scripted_agent(calls, middleware, [tool]).ainvoke(start))
        assert ran == ["u"] and example.tool_messages(result)[0].status == "success"
        ran.clear()
        result = example.scripted_agent(calls, middleware, [tool]).invoke(start)
        assert ran == [] and example.tool_messages(result)[0].status == "error"
        assert "no callable for this route" in str(example.tool_messages(result)[0].content)

    # --- the name the policy judges is the tool that runs ---------------------------------

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_call_whose_name_is_not_the_tool_that_runs_is_refused_before_the_policy(
        self, example, route
    ):
        """Review round 1, 2a: the model called ``Run_SQL``; an earlier middleware attached
        the registered ``run_sql`` (LangChain's ``request.override(tool=...)`` pattern); a
        policy keyed on ``run_sql`` judged ``Run_SQL`` and the DELETE ran."""
        from langchain.agents.middleware import AgentMiddleware
        from langchain_core.messages import HumanMessage

        class Repair(AgentMiddleware):
            tools = ()

            @staticmethod
            def fixed(request):
                if request.tool is None and request.tool_call["name"].lower() == "run_sql":
                    return request.override(tool=example.run_sql)
                return request

            def wrap_tool_call(self, request, handler):
                return handler(self.fixed(request))

            async def awrap_tool_call(self, request, handler):
                return await handler(self.fixed(request))

        seen = []

        def recording(name, args):
            seen.append((name, dict(args)))
            return example.sql_scope_policy(name, args)

        calls = [self.call("Run_SQL", self.DELETE)]
        start = {"messages": [HumanMessage(content="go")]}
        example.EXECUTED.clear()
        agent = example.scripted_agent(calls, [Repair(), example.gate_middleware(recording)])
        result = asyncio.run(agent.ainvoke(start)) if route == "async" else agent.invoke(start)
        message = example.tool_messages(result)[0]
        assert example.EXECUTED == [] and seen == []
        assert message.status == "error" and message.name == "run_sql"
        assert "the call names another tool than this one" in str(message.content)
        assert "Run_SQL" not in str(message.content), "no model-chosen text in the reason"

        example.EXECUTED.clear()
        ungated = example.scripted_agent(calls, [Repair()])
        ungated.invoke(start)
        assert example.EXECUTED == [("run_sql", self.DELETE)], "negative control"

    # --- what the model provider would run is refused before the model is called ----------

    PROVIDER_TOOL = {"type": "web_search_20250305", "name": "web_search"}

    @staticmethod
    def binding_model(example, bound, called):
        from langchain_core.messages import AIMessage

        class Binding(example.ScriptedChatModel):
            def bind_tools(self, tools, **kwargs):
                bound.extend(t if isinstance(t, dict) else t.name for t in tools)
                return self

            def _generate(self, *args, **kwargs):
                called.append(True)
                return super()._generate(*args, **kwargs)

        return Binding(messages=iter([AIMessage(content="done")]))

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_provider_run_tool_dict_refuses_the_model_call(self, example, route):
        """Review round 1, 2b: ``create_agent`` binds a dict in ``tools=`` to the model and
        the provider runs it; no tool-call middleware is ever called for it."""
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage

        bound, called = [], []
        agent = create_agent(
            model=self.binding_model(example, bound, called),
            tools=[example.run_sql, self.PROVIDER_TOOL],
            middleware=[example.gate_middleware()],
        )
        start = {"messages": [HumanMessage(content="go")]}
        result = asyncio.run(agent.ainvoke(start)) if route == "async" else agent.invoke(start)
        assert bound == [] and called == [], "the model was never bound or called"
        answer = str(result["messages"][-1].content)
        assert answer.startswith("Recusal refused this model call")
        assert "1 tool(s) given as a dict" in answer
        assert "web_search" not in answer, "nothing from inside the dict is echoed"

    def test_negative_control_ungated_the_provider_run_tool_is_bound_to_the_model(self, example):
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage

        bound, called = [], []
        agent = create_agent(
            model=self.binding_model(example, bound, called),
            tools=[example.run_sql, self.PROVIDER_TOOL],
        )
        agent.invoke({"messages": [HumanMessage(content="go")]})
        assert self.PROVIDER_TOOL in bound and called == [True]

    def test_an_agent_with_only_its_own_tools_calls_the_model(self, example):
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage

        bound, called = [], []
        agent = create_agent(
            model=self.binding_model(example, bound, called),
            tools=[example.run_sql],
            middleware=[example.gate_middleware()],
        )
        result = agent.invoke({"messages": [HumanMessage(content="go")]})
        assert bound == ["run_sql"] and called == [True]
        assert str(result["messages"][-1].content) == "done"

    # --- what the gate does not see, asserted on purpose ----------------------------------

    def test_limitation_an_earlier_middleware_that_runs_a_tool_itself_is_never_judged(
        self, example
    ):
        """Documented in the package README: a middleware listed before the gate that runs
        a tool itself and answers never reaches the gate, wherever the gate is listed."""
        from langchain.agents.middleware import wrap_tool_call
        from langchain_core.messages import ToolMessage

        @wrap_tool_call
        def runs_it_itself(request, handler):
            output = example.run_sql.invoke(request.tool_call["args"])
            return ToolMessage(content=output, tool_call_id=request.tool_call["id"])

        seen = []

        def recording(name, args):
            seen.append(name)
            return example.sql_scope_policy(name, args)

        _, executed = example.run_calls(
            [self.call("run_sql", self.DELETE)],
            middleware=[runs_it_itself, example.gate_middleware(recording)],
        )
        assert executed == [("run_sql", self.DELETE)] and seen == []

    # --- the tool shapes the LangGraph gate takes, through the middleware -----------------

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_legacy_single_input_tool_is_judged_under_its_parameter_name(self, example, route):
        from langchain_core.tools import Tool

        ran, seen = [], []

        def search(query):
            ran.append(query)
            return "found"

        def recording(name, args):
            seen.append((name, dict(args)))
            return example.sql_scope_policy("run_sql", {"sql": args.get("query", "")})

        tool = Tool(name="search", func=search, description="Search.")
        calls = [
            self.call("search", {"__arg1": "SELECT * FROM audit_log"}, "s"),
            self.call("search", {"__arg1": "DELETE FROM audit_log"}, "d"),
        ]
        if route == "sync":
            messages, _ = example.run_calls(calls, policy=recording, tools=[tool])
        else:
            messages, _ = asyncio.run(example.arun_calls(calls, policy=recording, tools=[tool]))
        assert sorted(seen, key=str) == [
            ("search", {"query": "DELETE FROM audit_log"}),
            ("search", {"query": "SELECT * FROM audit_log"}),
        ]
        assert ran == ["SELECT * FROM audit_log"]
        assert {(m.tool_call_id, m.status) for m in messages} == {("s", "success"), ("d", "error")}

    def test_a_basetool_subclass_without_a_schema_is_judged_on_its_inferred_one(self, example):
        from langchain_core.tools import BaseTool

        ran, seen = [], []

        class Lookup(BaseTool):
            name: str = "lookup"
            description: str = "Look up."

            def _run(self, q: str, limit: int = 10, run_manager=None) -> str:
                ran.append((q, limit))
                return "ok"

        def recording(name, args):
            seen.append((name, dict(args)))
            return []

        messages, _ = example.run_calls(
            [self.call("lookup", {"q": "x"})], policy=recording, tools=[Lookup()]
        )
        assert messages[0].status == "success", messages[0].content
        assert seen == [("lookup", {"q": "x", "limit": 10})] and ran == [("x", 10)]


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

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_callable_parameter_the_schema_omits_refuses_through_the_middleware(
        self, example, route
    ):
        """The 2026-11-06 review's P1 on the middleware routes: ``charge(label, amount=100)``
        behind a schema declaring only ``label`` was judged without ``amount`` and ran with
        100."""
        from recusal import Finding

        seen = []

        def budget(name, args):
            seen.append(dict(args))
            if args.get("amount", 0) > 10:
                return [Finding.fail("budget", severity="CRITICAL", message="over 10")]
            return []

        calls = [self.call("charge", {"label": "review"})]
        tools = [example.narrow_charge_tool()]
        if route == "sync":
            messages, executed = example.run_calls(calls, policy=budget, tools=tools)
        else:
            messages, executed = asyncio.run(example.arun_calls(calls, policy=budget, tools=tools))
        assert executed == [] and seen == []
        assert messages[0].status == "error"
        text = messages[0].content
        assert "the callable's parameter `amount` is absent from the argument schema" in text
        assert "100" not in text, "the default is never disclosed"

    def test_negative_control_ungated_the_schema_absent_default_runs(self, example):
        _, executed = example.run_calls(
            [self.call("charge", {"label": "review"})],
            gated=False,
            tools=[example.narrow_charge_tool()],
        )
        assert executed == [("charge", {"label": "review", "amount": 100})]
