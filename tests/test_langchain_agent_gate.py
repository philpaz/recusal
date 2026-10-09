"""Offline regressions for LangChain create_agent using the shared Recusal gate."""
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
        pytest.importorskip("langchain")
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
            ("e", "success"), ("d", "error"), ("s", "success")
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
