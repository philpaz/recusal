"""Regression tests for the offline CrewAI + Recusal tool-hook example."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "crewai_gate.py"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"
REFUSED = "Recusal refused `run_sql` [FAIL]"


def test_python_version_refusal_is_clear_on_every_python():
    code = (
        "import runpy, sys; "
        "sys.version_info = (3, 9, 0, 'final', 0); "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "crewai_gate.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_crewai_stays_out_of_the_recusal_package_and_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "crewai" not in pyproject
    for path in (ROOT / "recusal").rglob("*.py"):
        assert "crewai" not in path.read_text(encoding="utf-8").lower(), path


def test_pinned_version_is_explicit_in_example_docs_and_ci():
    for path in (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        assert "crewai==1.15.34" in path.read_text(encoding="utf-8"), path.name
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert ci.count("crewai==1.15.34") == 2


@pytest.mark.skipif(sys.version_info < (3, 10), reason="CrewAI requires Python 3.10+")
class TestCrewAIGate:
    @pytest.fixture
    def example(self):
        pytest.importorskip("crewai", reason="crewai is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_crewai_gate_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        # the registry is process-global: start every test from a clean one
        from crewai.hooks import clear_after_tool_call_hooks, clear_before_tool_call_hooks

        clear_before_tool_call_hooks()
        clear_after_tool_call_hooks()
        yield module
        clear_before_tool_call_hooks()
        clear_after_tool_call_hooks()

    @staticmethod
    def three_calls():
        return [
            ("echo", {"text": "hello"}, "c1"),
            ("run_sql", {"sql": DELETE}, "c2"),
            ("run_sql", {"sql": SELECT}, "c3"),
        ]

    # --- the gate ------------------------------------------------------------------------

    def test_allowed_call_runs_its_tool(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": SELECT}, "c1")])
        assert executed == [("run_sql", {"sql": SELECT})]
        assert responses["c1"] == f"recorded SQL: {SELECT}"

    def test_refused_call_does_not_run_and_the_reason_reaches_the_model(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")])
        assert executed == []
        assert responses["c1"].startswith(REFUSED)
        assert "top-level WHERE" in responses["c1"]
        assert not responses["c1"].startswith(example.BLOCKED_PREFIX)

    def test_several_calls_in_one_turn_are_gated_separately(self, example):
        responses, executed = example.run_calls(self.three_calls())
        assert responses["c1"] == "hello"
        assert responses["c2"].startswith(REFUSED)
        assert responses["c3"] == f"recorded SQL: {SELECT}"
        assert sorted(executed) == [("echo", {"text": "hello"}), ("run_sql", {"sql": SELECT})]

    def test_policy_exception_fails_closed(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        responses, executed = example.run_calls(
            [("echo", {"text": "unsafe"}, "c1")], policy=broken_policy
        )
        assert executed == []
        assert responses["c1"] == "Recusal failed closed (policy error): policy exploded"

    def test_a_crash_inside_decide_also_fails_closed(self, example, monkeypatch):
        def crashing_decide(*args, **kwargs):
            raise RuntimeError("decide crashed")

        monkeypatch.setattr(example, "decide", crashing_decide)
        responses, executed = example.run_calls([("echo", {"text": "unsafe"}, "c1")])
        assert executed == []
        assert responses["c1"] == "Recusal failed closed (policy error): decide crashed"

    def test_non_mapping_arguments_are_refused_by_the_gate_itself(self, example):
        assert example._refusal("run_sql", ["x"], example.sql_scope_policy) == (
            "Recusal refused `run_sql`: arguments are not a mapping"
        )

    def test_negative_control_runs_delete_without_the_hooks(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")], gated=False)
        assert executed == [("run_sql", {"sql": DELETE})]
        assert responses["c1"] == f"recorded SQL: {DELETE}"

    # --- install and uninstall: a global registry, handled honestly -------------------------

    def test_uninstall_removes_both_hooks_and_the_delete_runs_again(self, example):
        from crewai.hooks import get_after_tool_call_hooks, get_before_tool_call_hooks

        uninstall = example.install_recusal_hooks(example.sql_scope_policy)
        assert any(getattr(h, "recusal_gate", False) for h in get_before_tool_call_hooks())
        assert any(getattr(h, "recusal_gate", False) for h in get_after_tool_call_hooks())
        uninstall()
        assert not any(getattr(h, "recusal_gate", False) for h in get_before_tool_call_hooks())
        assert not any(getattr(h, "recusal_gate", False) for h in get_after_tool_call_hooks())
        example.EXECUTED.clear()
        example.run_crew(example.scripted_agent([("run_sql", {"sql": DELETE}, "c1")]))
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]

    def test_one_install_gates_a_second_agent_and_crew_in_the_same_process(self, example):
        uninstall = example.install_recusal_hooks(example.sql_scope_policy)
        try:
            example.EXECUTED.clear()
            first = example.run_crew(example.scripted_agent([("run_sql", {"sql": DELETE}, "a1")]))
            second = example.run_crew(example.scripted_agent([("run_sql", {"sql": DELETE}, "b1")]))
        finally:
            uninstall()
        assert example.EXECUTED == []
        assert first["a1"].startswith(REFUSED) and second["b1"].startswith(REFUSED)

    def test_the_after_hook_leaves_another_hooks_block_alone(self, example):
        from crewai.hooks import register_before_tool_call_hook

        def someone_elses_block(context):
            return False if context.tool_name == "echo" else None

        register_before_tool_call_hook(someone_elses_block)
        uninstall = example.install_recusal_hooks(example.sql_scope_policy)
        try:
            example.EXECUTED.clear()
            responses = example.run_crew(example.scripted_agent([("echo", {"text": "x"}, "c1")]))
        finally:
            uninstall()
        assert example.EXECUTED == []
        assert responses["c1"].startswith(example.BLOCKED_PREFIX)  # CrewAI's own line, not ours

    # --- order: install the gate after your other hooks --------------------------------------

    def test_why_the_gate_installs_last_a_later_hook_can_change_what_runs(self, example):
        from crewai.hooks import register_before_tool_call_hook

        def rewrite(context):
            if context.tool_name == "run_sql":
                context.tool_input["sql"] = DELETE  # the dict the tool will receive
            return None

        # gate FIRST, rewrite after it: the SELECT is judged, the DELETE runs
        uninstall = example.install_recusal_hooks(example.sql_scope_policy)
        register_before_tool_call_hook(rewrite)
        try:
            example.EXECUTED.clear()
            responses = example.run_crew(
                example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
            )
        finally:
            uninstall()
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]
        assert responses["c1"] == f"recorded SQL: {DELETE}"
        # rewrite first, gate LAST: the rewritten DELETE is what gets judged
        from crewai.hooks import clear_before_tool_call_hooks

        clear_before_tool_call_hooks()
        register_before_tool_call_hook(rewrite)
        uninstall = example.install_recusal_hooks(example.sql_scope_policy)
        try:
            example.EXECUTED.clear()
            responses = example.run_crew(
                example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
            )
        finally:
            uninstall()
        assert example.EXECUTED == []
        assert responses["c1"].startswith(REFUSED)

    # --- the script ------------------------------------------------------------------------

    def test_example_runs_end_to_end(self, example):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "tools that ran: 2" in result.stdout
        assert REFUSED in result.stdout
        assert "Execution Traces" not in result.stdout
