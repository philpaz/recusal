"""Regression tests for the offline LangGraph + Recusal ToolNode example."""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "langgraph_gate.py"


def test_python_version_refusal_is_clear_on_every_python():
    code = (
        "import runpy, sys; "
        "sys.version_info = (3, 9, 0, 'final', 0); "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "langgraph_gate.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_langgraph_stays_out_of_the_recusal_package_and_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "langgraph" not in pyproject
    assert "langchain-core" not in pyproject
    assert "langchain" not in pyproject
    for path in (ROOT / "recusal").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "langgraph" not in source, path
        assert "langchain" not in source, path


def test_pinned_versions_are_explicit_in_example_docs_and_ci():
    expected = ("langgraph==1.2.12", "langchain-core==1.6.5")
    for path in (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        text = path.read_text(encoding="utf-8")
        for pin in expected:
            assert pin in text
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for pin in expected:
        assert ci.count(pin) == 2


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestLangGraphGate:
    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_langgraph_gate_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def call(name, args, ident="1"):
        return {"name": name, "args": args, "id": ident, "type": "tool_call"}

    def test_allowed_call_runs_its_tool(self, example):
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": "SELECT * FROM audit_log"})]
        )
        assert executed == [("run_sql", {"sql": "SELECT * FROM audit_log"})]
        assert messages[0].status == "success"

    def test_refused_call_does_not_run_and_reason_reaches_agent(self, example):
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": "DELETE FROM audit_log"})]
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "Recusal refused" in str(messages[0].content)
        assert "top-level WHERE" in str(messages[0].content)

    def test_several_calls_are_gated_separately(self, example):
        calls = [
            self.call("echo", {"text": "hello"}, "1"),
            self.call("run_sql", {"sql": "DELETE FROM audit_log"}, "2"),
            self.call("run_sql", {"sql": "SELECT * FROM audit_log"}, "3"),
        ]
        messages, executed = example.run_calls(calls)
        assert [m.status for m in messages] == ["success", "error", "success"]
        assert len(executed) == 2

    def test_gate_is_plug_and_play_for_an_unseen_tool_list(self, example):
        @example.tool
        def query(sql: str) -> str:
            """A policy-covered tool unknown to the example's built-in list."""
            raise AssertionError("refused tool must not run")

        node = example.ToolNode(
            [query], wrap_tool_call=example.recusal_gate(example.sql_scope_policy)
        )
        messages = example.run_in_graph(node, [self.call("query", {"sql": "DELETE FROM x"}, "x")])
        assert messages[0].status == "error"

    def test_policy_exception_fails_closed(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        messages, executed = example.run_calls(
            [self.call("echo", {"text": "unsafe"})], policy=broken_policy
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "failed closed (policy error): policy exploded" in str(messages[0].content)

    def test_negative_control_runs_delete_without_hook(self, example):
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": "DELETE FROM audit_log"})], gated=False
        )
        assert executed == [("run_sql", {"sql": "DELETE FROM audit_log"})]
        assert messages[0].status == "success"

    def test_async_allowed_call_runs_its_tool(self, example):
        messages, executed = asyncio.run(
            example.run_calls_async([self.call("run_sql", {"sql": "SELECT * FROM audit_log"})])
        )
        assert executed == [("run_sql", {"sql": "SELECT * FROM audit_log"})]
        assert messages[0].status == "success"

    def test_async_refusal_reaches_agent_without_running_tool(self, example):
        messages, executed = asyncio.run(
            example.run_calls_async([self.call("run_sql", {"sql": "DELETE FROM audit_log"})])
        )
        assert executed == []
        assert messages[0].status == "error"
        assert messages[0].tool_call_id == "1"
        assert messages[0].name == "run_sql"
        assert "Recusal refused" in str(messages[0].content)
        assert "top-level WHERE" in str(messages[0].content)

    def test_async_several_calls_are_gated_separately(self, example):
        messages, executed = asyncio.run(
            example.run_calls_async(
                [
                    self.call("echo", {"text": "hello"}, "1"),
                    self.call("run_sql", {"sql": "DELETE FROM audit_log"}, "2"),
                    self.call("run_sql", {"sql": "SELECT * FROM audit_log"}, "3"),
                ]
            )
        )
        assert [(m.tool_call_id, m.status) for m in messages] == [
            ("1", "success"),
            ("2", "error"),
            ("3", "success"),
        ]
        assert sorted(name for name, _ in executed) == ["echo", "run_sql"]
        assert ("run_sql", {"sql": "DELETE FROM audit_log"}) not in executed

    def test_async_gate_awaits_an_unseen_async_tool(self, example):
        executed = []

        @example.tool
        async def query(sql: str) -> str:
            """Record an async SQL call without a database."""
            await asyncio.sleep(0)
            executed.append(sql)
            return sql

        node = example.ToolNode(
            [query], awrap_tool_call=example.recusal_agate(example.sql_scope_policy)
        )
        messages = asyncio.run(
            example.run_in_graph_async(
                node,
                [
                    self.call("query", {"sql": "SELECT * FROM x"}, "allowed"),
                    self.call("query", {"sql": "DELETE FROM x"}, "refused"),
                ],
            )
        )
        assert executed == ["SELECT * FROM x"]
        assert [m.status for m in messages] == ["success", "error"]

    def test_async_policy_exception_fails_closed(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        messages, executed = asyncio.run(
            example.run_calls_async([self.call("echo", {"text": "unsafe"})], policy=broken_policy)
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "failed closed (policy error): policy exploded" in str(messages[0].content)

    def test_async_negative_control_runs_delete_without_hook(self, example):
        messages, executed = asyncio.run(
            example.run_calls_async(
                [self.call("run_sql", {"sql": "DELETE FROM audit_log"})], gated=False
            )
        )
        assert executed == [("run_sql", {"sql": "DELETE FROM audit_log"})]
        assert messages[0].status == "success"


SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestEffectiveArguments:
    """Review 5, P1: the gate judged ``tool_call["args"]`` as the model wrote them; the tool
    then validated them through its ``args_schema``, and a field validator, a default or a
    coercion could make the function receive something else. The gate now judges the
    validated invocation and executes exactly that, once."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "recusal_langgraph_effective_example", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    class Recording:
        def __init__(self, policy):
            self.policy, self.seen = policy, []

        def __call__(self, name, args):
            self.seen.append((name, dict(args)))
            return self.policy(name, args)

    @staticmethod
    def call(name, args, ident="1"):
        return {"name": name, "args": args, "id": ident, "type": "tool_call"}

    def test_a_schema_that_rewrites_an_allowed_proposal_is_judged_on_the_rewritten_value(
        self, example
    ):
        policy = self.Recording(example.sql_scope_policy)
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})],
            policy=policy,
            tools=[example.transforming_run_sql()],
        )
        assert policy.seen == [("run_sql", {"sql": DELETE})]  # what the tool would receive
        assert executed == []
        assert messages[0].status == "error" and "Recusal refused `run_sql`" in messages[0].content

    def test_negative_control_the_rewriting_tool_runs_the_rewritten_sql_ungated(self, example):
        _, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})],
            gated=False,
            tools=[example.transforming_run_sql()],
        )
        assert executed == [("run_sql", {"sql": DELETE})]

    def test_the_async_gate_judges_the_rewritten_value_too(self, example):
        policy = self.Recording(example.sql_scope_policy)
        _, executed = asyncio.run(
            example.run_calls_async(
                [self.call("run_sql", {"sql": SELECT})],
                policy=policy,
                tools=[example.transforming_run_sql()],
            )
        )
        assert policy.seen == [("run_sql", {"sql": DELETE})] and executed == []

    def test_defaults_and_coercion_are_judged_as_the_tool_receives_them(self, example):
        from langchain_core.tools import tool
        from pydantic import BaseModel

        from recusal import Finding

        class Args(BaseModel):
            sql: str
            limit: int = 10

        ran = []

        @tool("query", args_schema=Args)
        def query(sql: str, limit: int = 10) -> str:
            """Record a bounded query."""
            ran.append({"sql": sql, "limit": limit, "type": type(limit).__name__})
            return "recorded"

        def cap(name, args):
            if args.get("limit", 0) > 50:
                return [Finding.fail("limit", severity="CRITICAL", message="over 50")]
            return []

        policy = self.Recording(cap)
        example.run_calls([self.call("query", {"sql": SELECT})], policy=policy, tools=[query])
        assert policy.seen[-1] == ("query", {"sql": SELECT, "limit": 10})
        assert ran[-1] == {"sql": SELECT, "limit": 10, "type": "int"}
        example.run_calls(
            [self.call("query", {"sql": SELECT, "limit": "7"})], policy=policy, tools=[query]
        )
        assert policy.seen[-1] == ("query", {"sql": SELECT, "limit": 7})
        assert ran[-1]["limit"] == 7 and ran[-1]["type"] == "int"
        ran.clear()
        messages, _ = example.run_calls(
            [self.call("query", {"sql": SELECT, "limit": "99"})], policy=policy, tools=[query]
        )
        assert policy.seen[-1] == ("query", {"sql": SELECT, "limit": 99}) and ran == []
        assert "Recusal refused `query`" in messages[0].content

    def test_arguments_the_tool_rejects_are_refused_with_the_reason(self, example):
        policy = self.Recording(example.sql_scope_policy)
        messages, executed = example.run_calls([self.call("run_sql", {"sql": 5})], policy=policy)
        assert executed == [] and policy.seen == []
        assert "do not validate" in messages[0].content

    def test_a_tool_the_node_does_not_know_is_refused_not_errored(self, example):
        messages, executed = example.run_calls([self.call("nope", {"x": 1})])
        assert executed == [] and "no tool by that name" in messages[0].content

    def test_the_judging_copy_runs_only_what_it_judged(self, example):
        call = {"name": "run_sql", "args": {"sql": SELECT}, "id": "c1", "type": "tool_call"}
        copy = example._judging(example.run_sql, call, example.sql_scope_policy, {})
        assert copy._to_args_and_kwargs({"sql": SELECT}, "c1") == ((), {"sql": SELECT})
        with pytest.raises(example.RecusalRefusal, match="Recusal refused `run_sql`"):
            copy._to_args_and_kwargs({"sql": DELETE}, "c1")
        with pytest.raises(RuntimeError, match="different call"):
            copy._to_args_and_kwargs({"sql": SELECT}, "c2")
        # the original tool is untouched
        assert example.run_sql._to_args_and_kwargs({"sql": "x"}, "c9") == ((), {"sql": "x"})
        assert example.run_sql.handle_tool_error is False

    def test_unjudgeable_inputs_are_refused_through_decide_and_reach_the_audit_log(
        self, example, tmp_path
    ):
        import json

        from recusal.audit import AuditLog

        log = AuditLog(path=str(tmp_path / "audit.jsonl"))
        call = {"name": "run_sql", "args": {"sql": 5}, "id": "c1", "type": "tool_call"}
        copy = example._judging(
            example.run_sql, call, example.sql_scope_policy, {"audit": log, "surface": "t"}
        )
        with pytest.raises(example.RecusalRefusal) as refused:
            copy._to_args_and_kwargs({"sql": 5}, "c1")
        assert str(refused.value).startswith("Recusal refused `run_sql`")
        assert "do not validate" in str(
            refused.value
        ) and "sql: Input should be a valid string" in str(refused.value)
        entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert entry["action"]["decision"] == "deny" and entry["action"]["tool"] == "run_sql"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestFinalInvocation:
    """Review 6, P1 and P2: the gate validated once in the hook (before ToolNode injected the
    host's state), judged a ``to_jsonable_python`` projection of the result, then validated
    again inside a pinned copy and compared projections. A nested model's serializer showed
    the policy SELECT while the field held DELETE; a stateful validator returned ``1`` then
    ``True`` and ``1 == True`` passed the comparison; a tool with a required
    ``InjectedState`` failed validation before the state existed. The gate now judges once,
    inside the tool's own parse step after injection, through a serializer-free projection,
    and executes the very objects it parsed."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_langgraph_final_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def call(name, args, ident="c1"):
        return {"name": name, "args": args, "id": ident, "type": "tool_call"}

    @staticmethod
    def serializer_tool(ran):
        """The review's probe: a nested model whose serializer displays SELECT over DELETE."""
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
            ran.append({"sql": query.sql, "type": type(query).__name__})
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

    # --- the serializer probe, both routes --------------------------------------------------

    def test_a_display_serializer_cannot_hide_the_value_the_tool_receives(self, example):
        seen, ran = [], []
        messages, _ = example.run_calls(
            [self.call("query", {"query": {"sql": DELETE}})],
            policy=self.select_only(seen),
            tools=[self.serializer_tool(ran)],
        )
        assert seen == [{"query": {"sql": DELETE}}], "the policy sees the field, not the serializer"
        assert ran == []
        assert messages[0].status == "error" and "not the approved SELECT" in messages[0].content

    def test_the_async_gate_sees_through_the_serializer_too(self, example):
        seen, ran = [], []
        messages, _ = asyncio.run(
            example.run_calls_async(
                [self.call("query", {"query": {"sql": DELETE}})],
                policy=self.select_only(seen),
                tools=[self.serializer_tool(ran)],
            )
        )
        assert seen == [{"query": {"sql": DELETE}}] and ran == []
        assert messages[0].status == "error"

    def test_negative_control_ungated_the_serializer_tool_runs_the_delete(self, example):
        ran = []
        example.run_calls(
            [self.call("query", {"query": {"sql": DELETE}})],
            gated=False,
            tools=[self.serializer_tool(ran)],
        )
        assert ran == [{"sql": DELETE, "type": "Query"}]

    def test_an_approved_nested_value_runs_as_the_model_object(self, example):
        seen, ran = [], []
        messages, _ = example.run_calls(
            [self.call("query", {"query": {"sql": SELECT}})],
            policy=self.select_only(seen),
            tools=[self.serializer_tool(ran)],
        )
        assert messages[0].status == "success"
        assert ran == [{"sql": SELECT, "type": "Query"}], "the tool still receives its model"

    # --- one validation: the policy's object is the tool's object --------------------------

    @staticmethod
    def stateful_amount(counter, second):
        from langchain_core.tools import tool
        from pydantic import BaseModel, field_validator

        class Args(BaseModel):
            amount: Any

            @field_validator("amount")
            @classmethod
            def convert(cls, value):
                counter["n"] += 1
                return 1 if counter["n"] == 1 else second

        ran = []

        @tool("charge", args_schema=Args)
        def charge(amount: Any) -> str:
            """Record a typed value."""
            ran.append((amount, type(amount).__name__))
            return "recorded"

        return charge, ran

    @staticmethod
    def int_only(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append((args["amount"], type(args["amount"]).__name__))
            if type(args["amount"]) is int:
                return []
            return [Finding.fail("type", severity="CRITICAL", message="not an int")]

        return policy

    def test_a_stateful_validator_runs_once_so_the_tool_gets_the_judged_value(self, example):
        counter, seen = {"n": 0}, []
        charge, ran = self.stateful_amount(counter, second=True)
        messages, _ = example.run_calls(
            [self.call("charge", {"amount": 1})], policy=self.int_only(seen), tools=[charge]
        )
        assert counter["n"] == 1
        assert seen == [(1, "int")] and ran == [(1, "int")]
        assert messages[0].status == "success"

    def test_a_validator_that_yields_a_bool_on_its_one_run_is_judged_as_a_bool(self, example):
        from langchain_core.tools import tool
        from pydantic import BaseModel, field_validator

        class Args(BaseModel):
            amount: Any

            @field_validator("amount")
            @classmethod
            def convert(cls, value):
                return True

        ran = []

        @tool("charge", args_schema=Args)
        def charge(amount: Any) -> str:
            """Record a typed value."""
            ran.append(amount)
            return "recorded"

        seen = []
        messages, _ = example.run_calls(
            [self.call("charge", {"amount": 1})], policy=self.int_only(seen), tools=[charge]
        )
        assert seen == [(True, "bool")] and ran == []
        assert "not an int" in messages[0].content

    # --- host injection: present at the seam, excluded from judgment, never the model's ----

    @staticmethod
    def state_tool(ran):
        from typing import Annotated

        from langchain_core.tools import tool
        from langgraph.prebuilt import InjectedState

        @tool
        def with_state(sql: str, state: Annotated[dict, InjectedState]) -> str:
            """Record SQL with the host's state."""
            ran.append({"sql": sql, "state_keys": sorted(state)})
            return "recorded"

        return with_state

    @staticmethod
    def recording(seen, inner):
        def policy(name, args):
            seen.append(dict(args))
            return inner(name, args)

        return policy

    def test_a_required_injected_state_runs_gated_and_the_policy_sees_only_the_model(self, example):
        ran, seen = [], []
        messages, _ = example.run_calls(
            [self.call("with_state", {"sql": SELECT})],
            policy=self.recording(seen, example.sql_scope_policy),
            tools=[self.state_tool(ran)],
        )
        assert messages[0].status == "success", messages[0].content
        assert ran == [{"sql": SELECT, "state_keys": ["messages"]}]
        assert seen == [{"sql": SELECT}], "host state is not the model's proposal"

    def test_the_async_gate_passes_injected_state_too(self, example):
        ran, seen = [], []
        messages, _ = asyncio.run(
            example.run_calls_async(
                [self.call("with_state", {"sql": SELECT})],
                policy=self.recording(seen, example.sql_scope_policy),
                tools=[self.state_tool(ran)],
            )
        )
        assert messages[0].status == "success" and seen == [{"sql": SELECT}]
        assert ran == [{"sql": SELECT, "state_keys": ["messages"]}]

    def test_a_forged_state_in_the_proposal_is_replaced_by_the_hosts(self, example):
        ran, seen = [], []
        example.run_calls(
            [self.call("with_state", {"sql": SELECT, "state": {"forged": True}})],
            policy=self.recording(seen, example.sql_scope_policy),
            tools=[self.state_tool(ran)],
        )
        assert ran == [{"sql": SELECT, "state_keys": ["messages"]}]
        assert seen == [{"sql": SELECT}]

    def test_the_tool_runtime_is_injected_and_a_generic_injected_arg_is_never_the_models(
        self, example
    ):
        from typing import Annotated

        from langchain_core.tools import InjectedToolArg, tool
        from langgraph.prebuilt import ToolRuntime

        ran, seen = [], []

        @tool
        def with_runtime(
            sql: str, runtime: ToolRuntime, note: Annotated[str, InjectedToolArg] = "host"
        ) -> str:
            """Record SQL with the runtime and a host-only note."""
            ran.append({"sql": sql, "runtime": type(runtime).__name__, "note": note})
            return "recorded"

        messages, _ = example.run_calls(
            [self.call("with_runtime", {"sql": SELECT, "note": "forged", "runtime": "x"})],
            policy=self.recording(seen, example.sql_scope_policy),
            tools=[with_runtime],
        )
        assert messages[0].status == "success", messages[0].content
        assert ran == [{"sql": SELECT, "runtime": "ToolRuntime", "note": "host"}]
        assert seen == [{"sql": SELECT}], "the model cannot fill a host argument"

    def test_the_store_is_injected_when_the_graph_has_one(self, example):
        from typing import Annotated

        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.tools import tool
        from langgraph.graph import START, MessagesState, StateGraph
        from langgraph.prebuilt import InjectedStore, ToolNode, tools_condition
        from langgraph.store.base import BaseStore
        from langgraph.store.memory import InMemoryStore

        ran, seen = [], []

        @tool
        def with_store(sql: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
            """Record SQL with the host's store."""
            ran.append({"sql": sql, "store": type(store).__name__})
            return "recorded"

        calls = [self.call("with_store", {"sql": SELECT, "store": "forged"})]
        script = iter([AIMessage(content="", tool_calls=calls)])
        builder = StateGraph(MessagesState)
        builder.add_node(
            "agent", lambda state: {"messages": [next(script, AIMessage(content="done"))]}
        )
        node = ToolNode(
            [with_store],
            wrap_tool_call=example.recusal_gate(self.recording(seen, example.sql_scope_policy)),
        )
        builder.add_node("tools", node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        graph = builder.compile(store=InMemoryStore())
        messages = [
            m for m in graph.invoke({"messages": []})["messages"] if isinstance(m, ToolMessage)
        ]
        assert messages[0].status == "success", messages[0].content
        assert ran == [{"sql": SELECT, "store": "InMemoryStore"}] and seen == [{"sql": SELECT}]

    # --- every model-controlled argument is judged, not only the published properties ------

    @staticmethod
    def extra_tool(ran):
        import json

        from langchain_core.tools import tool
        from pydantic import BaseModel, ConfigDict

        class Args(BaseModel):
            model_config = ConfigDict(extra="allow")
            sql: str

        @tool("query", args_schema=Args)
        def query(sql: str, **kwargs) -> str:
            """Record a query and whatever extra options the schema accepted."""
            ran.append(json.loads(json.dumps({"sql": sql, **kwargs})))
            return "recorded"

        return query

    @staticmethod
    def own_subject_only(name, args):
        from recusal import Finding

        if args.get("subject", "me") != "me":
            return [Finding.fail("subject", severity="CRITICAL", message="another subject")]
        return []

    def test_an_extra_field_the_schema_accepts_is_judged(self, example):
        ran, seen = [], []
        query = self.extra_tool(ran)
        assert list(query.args) == ["sql"], "the published schema does not list the extra"
        messages, _ = example.run_calls(
            [self.call("query", {"sql": SELECT, "subject": "other-customer"})],
            policy=self.recording(seen, self.own_subject_only),
            tools=[query],
        )
        assert seen == [{"sql": SELECT, "subject": "other-customer"}]
        assert ran == [] and "another subject" in messages[0].content
        messages, _ = example.run_calls(
            [self.call("query", {"sql": SELECT})],
            policy=self.recording(seen, self.own_subject_only),
            tools=[query],
        )
        assert messages[0].status == "success" and ran == [{"sql": SELECT}]

    # --- the lifecycle the seam relies on ----------------------------------------------------

    def test_a_tool_whose_class_bypasses_the_parse_step_is_refused_not_executed(self, example):
        from langchain_core.messages import ToolMessage
        from langchain_core.tools import BaseTool

        ran = []

        class DirectTool(BaseTool):
            name: str = "direct_sql"
            description: str = "A custom invocation path, recording only."

            def _run(self, sql: str) -> str:
                ran.append(sql)
                return "recorded"

            def invoke(self, input, config=None, **kwargs):
                return ToolMessage(
                    content=self._run(**input["args"]), name=self.name, tool_call_id=input["id"]
                )

        messages, _ = example.run_calls(
            [self.call("direct_sql", {"sql": DELETE})], tools=[DirectTool()]
        )
        assert ran == []
        assert messages[0].status == "error"
        assert "cannot be judged (DirectTool overrides `invoke`)" in messages[0].content
        # the same tool, ungated, runs: the refusal is the gate's doing
        example.run_calls(
            [self.call("direct_sql", {"sql": DELETE})], gated=False, tools=[DirectTool()]
        )
        assert ran == [DELETE]

    def test_an_async_only_override_is_refused_on_both_routes(self, example):
        from langchain_core.tools import BaseTool

        ran = []

        class AsyncDirect(BaseTool):
            name: str = "adirect"
            description: str = "Overrides only the async entry point."

            def _run(self, sql: str) -> str:
                ran.append(sql)
                return "recorded"

            async def arun(self, tool_input, *args, **kwargs):
                return self._run(**tool_input)

        messages, _ = example.run_calls(
            [self.call("adirect", {"sql": SELECT})], tools=[AsyncDirect()]
        )
        assert ran == [] and "AsyncDirect overrides `arun`" in messages[0].content
        messages, _ = asyncio.run(
            example.run_calls_async([self.call("adirect", {"sql": SELECT})], tools=[AsyncDirect()])
        )
        assert ran == [] and "AsyncDirect overrides `arun`" in messages[0].content

    def test_an_instance_level_override_is_refused(self, example):
        copy = example.run_sql.model_copy()
        object.__setattr__(copy, "run", lambda *a, **k: "bypassed")
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})], tools=[copy]
        )
        assert executed == []
        assert "`run` is overridden on the tool instance" in messages[0].content

    # --- the projection -------------------------------------------------------------------------

    def test_a_mutating_policy_cannot_change_the_invocation(self, example):
        ran = []

        def mutate(name, args):
            args["query"]["sql"] = DELETE
            args["extra"] = 1
            return []

        messages, _ = example.run_calls(
            [self.call("query", {"query": {"sql": SELECT}})],
            policy=mutate,
            tools=[self.serializer_tool(ran)],
        )
        assert messages[0].status == "success"
        assert ran == [{"sql": SELECT, "type": "Query"}]

    def test_a_value_that_cannot_be_projected_is_refused_without_executing(self, example):
        from langchain_core.tools import tool
        from pydantic import BaseModel, field_validator

        class Args(BaseModel):
            names: Any

            @field_validator("names")
            @classmethod
            def to_set(cls, value):
                return set(value)

        ran = []

        @tool("tag", args_schema=Args)
        def tag(names: Any) -> str:
            """Record names."""
            ran.append(names)
            return "recorded"

        seen = []
        messages, _ = example.run_calls(
            [self.call("tag", {"names": ["a"]})],
            policy=self.recording(seen, lambda n, a: []),
            tools=[tag],
        )
        assert ran == [] and seen == []
        assert "a set value cannot be judged faithfully" in messages[0].content

    def test_the_projection_is_explicit_about_what_it_supports(self, example):
        import dataclasses
        import datetime
        import decimal
        import enum
        import uuid

        import pydantic.v1
        from pydantic import BaseModel

        class Color(enum.Enum):
            RED = "red"

        class Level(enum.IntEnum):
            HIGH = 3

        @dataclasses.dataclass
        class Point:
            x: int
            y: float

        class Inner(BaseModel):
            flag: bool

        class Outer(BaseModel):
            inner: Inner
            when: datetime.date

        faithful = example._faithful
        assert faithful((1, True, "s", None)) == [1, True, "s", None]
        projected = faithful([1, True])
        assert [type(v) for v in projected] == [int, bool], "1 and True stay distinct"
        assert faithful(Color.RED) == "red" and faithful(Level.HIGH) == 3
        assert faithful(datetime.datetime(2026, 11, 4, 12, 0)) == "2026-11-04T12:00:00"
        assert faithful(decimal.Decimal("1.50")) == "1.50"
        ident = uuid.UUID(int=1)
        assert faithful(ident) == str(ident)
        assert faithful(Point(1, 2.5)) == {"x": 1, "y": 2.5}
        assert faithful(Outer(inner=Inner(flag=False), when=datetime.date(2026, 1, 2))) == {
            "inner": {"flag": False},
            "when": "2026-01-02",
        }
        for bad in (
            float("nan"),
            float("inf"),
            decimal.Decimal("NaN"),
            {1: "int key"},
            {"a"},
            object(),
            pydantic.v1.create_model("V1", x=(int, 1))(),
        ):
            with pytest.raises(TypeError):
                faithful(bad)
        cyclic = []
        cyclic.append(cyclic)
        with pytest.raises(TypeError, match="cyclic"):
            faithful(cyclic)
        source = {"q": [1]}
        copy = faithful(source)
        copy["q"].append(2)
        assert source == {"q": [1]}, "the projection is detached"

    # --- refusal handling: narrow, and ordinary errors keep their path ---------------------

    @pytest.mark.parametrize(
        "handler", [True, lambda exc: "replaced by the node"], ids=["bool", "callable"]
    )
    def test_a_refusal_survives_the_nodes_error_handler(self, example, handler):
        from langgraph.prebuilt import ToolNode

        node = ToolNode(
            [example.run_sql],
            wrap_tool_call=example.recusal_gate(example.sql_scope_policy),
            handle_tool_errors=handler,
        )
        example.EXECUTED.clear()
        messages = example.run_in_graph(node, [self.call("run_sql", {"sql": DELETE})])
        assert example.EXECUTED == []
        assert messages[0].status == "error"
        assert messages[0].content.startswith("Recusal refused `run_sql`"), messages[0].content

    @staticmethod
    def erroring_tool(**update):
        from langchain_core.tools import ToolException, tool

        @tool
        def erroring(sql: str) -> str:
            """Raise an ordinary tool error."""
            raise ToolException("ordinary tool failure")

        return erroring.model_copy(update=update) if update else erroring

    def test_an_ordinary_tool_error_still_propagates_when_the_tool_does_not_handle_it(
        self, example
    ):
        from langchain_core.tools import ToolException
        from langgraph.prebuilt import ToolNode

        node = ToolNode(
            [self.erroring_tool()],
            wrap_tool_call=example.recusal_gate(lambda n, a: []),
            handle_tool_errors=False,
        )
        with pytest.raises(ToolException, match="ordinary tool failure"):
            example.run_in_graph(node, [self.call("erroring", {"sql": SELECT})])

    @pytest.mark.parametrize(
        ("setting", "expected"),
        [
            (True, "ordinary tool failure"),
            ("the tool's own text", "the tool's own text"),
            (lambda exc: f"handled: {exc}", "handled: ordinary tool failure"),
        ],
        ids=["bool", "str", "callable"],
    )
    def test_an_ordinary_tool_error_keeps_the_tools_own_handling(self, example, setting, expected):
        messages, _ = example.run_calls(
            [self.call("erroring", {"sql": SELECT})],
            policy=lambda n, a: [],
            tools=[self.erroring_tool(handle_tool_error=setting)],
        )
        assert messages[0].status == "error" and messages[0].content == expected

    # --- redaction after injection ------------------------------------------------------------

    def test_a_validation_error_on_host_state_never_shows_the_state(self, example):
        from typing import Annotated

        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langgraph.prebuilt import InjectedState, ToolNode

        @tool
        def wants_int_state(sql: str, state: Annotated[int, InjectedState]) -> str:
            """Declare a host-state shape the graph does not provide."""
            return "recorded"

        node = ToolNode([wants_int_state], wrap_tool_call=example.recusal_gate(lambda n, a: []))
        graph = example._graph(node, [self.call("wants_int_state", {"sql": SELECT})])
        result = graph.invoke({"messages": [HumanMessage(content="HOST_SECRET_SENTINEL")]})
        message = [m for m in result["messages"] if getattr(m, "tool_call_id", None) == "c1"][0]
        assert message.status == "error"
        assert "HOST_SECRET_SENTINEL" not in message.content
        assert "state: int_type" in message.content, message.content

    def test_a_validation_error_on_a_model_argument_names_the_location_not_the_input(self, example):
        messages, _ = example.run_calls(
            [self.call("run_sql", {"sql": {"nested": "MODEL_SECRET_SENTINEL"}})]
        )
        assert "sql: Input should be a valid string" in messages[0].content
        assert "MODEL_SECRET_SENTINEL" not in messages[0].content

    # --- the old comparison machinery is gone -------------------------------------------------

    def test_no_second_validation_and_no_comparison_remain(self):
        source = EXAMPLE.read_text(encoding="utf-8")
        for gone in ("_pinned", "ChangedAfterJudgment", "refuse_changed", "to_jsonable_python"):
            assert gone not in source, gone
