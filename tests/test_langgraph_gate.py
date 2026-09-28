"""Regression tests for the offline LangGraph + Recusal example."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "langgraph_gate.py"


def load_example():
    spec = importlib.util.spec_from_file_location("recusal_langgraph_gate_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_python_version_refusal_is_clear_on_every_python(capsys):
    example = load_example()
    with pytest.raises(SystemExit) as caught:
        example.main(version_info=(3, 9))
    assert caught.value.code != 0
    assert (
        "langgraph_gate.py requires Python 3.10+ "
        "(LangGraph 1.2.12); this is Python 3.9." in capsys.readouterr().err
    )


def test_langgraph_stays_out_of_the_recusal_package_and_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "langgraph" not in pyproject
    assert "langchain-core" not in pyproject
    for path in (ROOT / "recusal").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "langgraph" not in source, path


def test_pinned_versions_are_explicit_in_example_docs_and_ci():
    expected = ("langgraph==1.2.12", "langchain-core==1.6.5")
    files = [
        ROOT / "examples" / "langgraph_gate.py",
        ROOT / "README.md",
        ROOT / "docs" / "COOKBOOK.md",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for pin in expected:
            assert pin in text, f"{path} does not state {pin}"

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for pin in expected:
        assert ci.count(pin) == 2, f"expected one {pin} install in each LangGraph CI job"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestLangGraphGate:
    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        return load_example()

    def test_allowed_call_runs_its_tool(self, example):
        state, executed = example.run_scripted(
            [{"name": "run_sql", "args": {"sql": "SELECT * FROM audit_log"}}]
        )
        assert executed == [("run_sql", {"sql": "SELECT * FROM audit_log"})]
        assert state["results"] == ["recorded SQL: SELECT * FROM audit_log"]

    def test_refused_call_does_not_run_and_reason_reaches_agent(self, example):
        state, executed = example.run_scripted(
            [{"name": "run_sql", "args": {"sql": "DELETE FROM audit_log"}}]
        )
        assert executed == []
        assert state["results"][0].startswith("REFUSED: Recusal refused")
        assert "top-level WHERE" in state["results"][0]

    def test_policy_exception_fails_closed_without_running_tool(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        state, executed = example.run_scripted(
            [{"name": "echo", "args": {"text": "unsafe without a verdict"}}],
            policy=broken_policy,
        )
        assert executed == []
        assert "failed closed (policy error): policy exploded" in state["results"][0]

    def test_negative_control_runs_refused_call_when_gate_is_removed(self, example):
        state, executed = example.run_scripted(
            [{"name": "run_sql", "args": {"sql": "DELETE FROM audit_log"}}],
            gated=False,
        )
        assert executed == [("run_sql", {"sql": "DELETE FROM audit_log"})]
        assert state["results"] == ["recorded SQL: DELETE FROM audit_log"]
