"""Regression tests for the offline LangGraph + Recusal ToolNode example."""

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
    for path in (ROOT / "recusal").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "langgraph" not in source, path


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
