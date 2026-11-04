"""Regression tests for the offline LangGraph + Recusal ToolNode example."""

import asyncio
import subprocess
import sys
from pathlib import Path

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

    def test_the_pinned_copy_runs_only_the_judged_invocation(self, example):
        pinned = example._pinned(example.run_sql, {"sql": SELECT}, "c1")
        assert pinned._to_args_and_kwargs({"sql": SELECT}, "c1") == ((), {"sql": SELECT})
        with pytest.raises(example.ChangedAfterJudgment, match="changed after it was judged"):
            pinned._to_args_and_kwargs({"sql": DELETE}, "c1")
        with pytest.raises(RuntimeError, match="different call"):
            pinned._to_args_and_kwargs({"sql": SELECT}, "c2")
        # the original tool is untouched
        assert example.run_sql._to_args_and_kwargs({"sql": "x"}, "c9") == ((), {"sql": "x"})

    def test_unjudgeable_inputs_are_refused_through_decide_and_reach_the_audit_log(
        self, example, tmp_path
    ):
        import json

        from recusal.audit import AuditLog

        log = AuditLog(path=str(tmp_path / "audit.jsonl"))
        request = type("R", (), {})()
        request.tool_call = {"name": "run_sql", "args": {"sql": 5}, "id": "c1"}
        request.tool = example.run_sql
        refusal, pinned = example._judge(request, example.sql_scope_policy, audit=log, surface="t")
        assert pinned is None and refusal.status == "error"
        assert refusal.content.startswith("Recusal refused `run_sql`")
        assert "do not validate" in refusal.content
        entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert entry["action"]["decision"] == "deny" and entry["action"]["tool"] == "run_sql"
