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


@pytest.mark.skipif(sys.version_info < (3, 10), reason="CrewAI requires Python 3.10+")
class TestDenialReasonLifecycle:
    """Adapters review: the after hook re-judged a blocked call to explain it, so a stateful
    policy was invoked twice and the model lost the original reason. The reason is kept per
    invocation now, correlated by the argument object, per thread, bounded."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("crewai", reason="crewai is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_crewai_reason_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def stateful():
        calls = {"n": 0}

        def policy(name, args):
            from recusal import Finding

            calls["n"] += 1
            if calls["n"] == 1:
                return [Finding.fail("deny", severity="CRITICAL", message="original refusal")]
            return []

        return policy, calls

    @staticmethod
    def context(example, name, tool_input, result=None):
        kwargs = {
            "tool_name": name,
            "tool_input": tool_input,
            "tool": None,
            "agent": None,
            "task": None,
            "crew": None,
        }
        if result is not None:
            kwargs["tool_result"] = result
        try:
            return example.ToolCallHookContext(**kwargs)
        except TypeError:
            from types import SimpleNamespace

            return SimpleNamespace(**kwargs)

    # --- the integration witness: the reviewer's stateful policy through the real runner --

    def test_the_model_receives_the_reason_the_before_hook_blocked_on(self, example):
        policy, calls = self.stateful()
        responses, executed = example.run_calls(
            [("run_sql", {"sql": "DELETE FROM audit_log"}, "c1")], policy=policy
        )
        assert executed == []
        assert calls["n"] == 1, "the call was judged once"
        assert "original refusal" in responses["c1"], responses

    # --- hook-dispatch tests: labelled synthetic, they drive the two hooks directly ---------

    def test_same_thread_distinct_equal_dicts_each_get_their_own_reason(self, example):
        """Kills a content-keyed map and an equality match alike: d1 == d2, d1 is not d2,
        both denials pending, after hooks in reversed order."""
        from recusal import Finding

        reasons = iter(["reason A", "reason B"])

        def policy(name, args):
            return [Finding.fail("deny", severity="CRITICAL", message=next(reasons))]

        denials = example._Denials()
        before = example.recusal_before_tool_call(policy, denials)
        after = example.recusal_after_tool_call(policy, denials)
        d1, d2 = {"sql": "SELECT 1"}, {"sql": "SELECT 1"}
        assert d1 == d2 and d1 is not d2
        assert before(self.context(example, "run_sql", d1)) is False
        assert before(self.context(example, "run_sql", d2)) is False
        blocked = example.BLOCKED_PREFIX + " Tool: run_sql"
        assert "reason B" in after(self.context(example, "run_sql", d2, blocked))
        assert "reason A" in after(self.context(example, "run_sql", d1, blocked))

    def test_an_abandoned_entry_is_never_attached_to_a_later_call_reusing_the_dict(self, example):
        from recusal import Finding

        decisions = iter([[Finding.fail("deny", severity="CRITICAL", message="stale A")], []])
        denials = example._Denials()
        before = example.recusal_before_tool_call(lambda n, a: next(decisions), denials)
        after = example.recusal_after_tool_call(lambda n, a: [], denials)
        shared = {"sql": "SELECT 1"}
        assert before(self.context(example, "run_sql", shared)) is False  # A, after hook skipped
        assert before(self.context(example, "run_sql", shared)) is None  # B reuses the dict
        blocked = example.BLOCKED_PREFIX + " Tool: run_sql"  # another hook blocked B
        assert after(self.context(example, "run_sql", shared, blocked)) is None

    def test_a_call_another_hook_blocked_keeps_crewais_line(self, example):
        denials = example._Denials()
        after = example.recusal_after_tool_call(lambda n, a: [], denials)
        blocked = example.BLOCKED_PREFIX + " Tool: run_sql"
        assert after(self.context(example, "run_sql", {"sql": "x"}, blocked)) is None

    def test_another_thread_never_sees_this_threads_reason(self, example):
        import threading

        from recusal import Finding

        denials = example._Denials()
        before = example.recusal_before_tool_call(
            lambda n, a: [Finding.fail("deny", severity="CRITICAL", message="mine")], denials
        )
        after = example.recusal_after_tool_call(lambda n, a: [], denials)
        d = {"sql": "x"}
        assert before(self.context(example, "run_sql", d)) is False
        seen = []
        blocked = example.BLOCKED_PREFIX + " Tool: run_sql"
        t = threading.Thread(
            target=lambda: seen.append(after(self.context(example, "run_sql", d, blocked)))
        )
        t.start()
        t.join()
        assert seen == [None]
        assert "mine" in after(self.context(example, "run_sql", d, blocked))

    def test_retention_is_bounded_under_abandoned_calls(self, example):
        from recusal import Finding

        denials = example._Denials()
        before = example.recusal_before_tool_call(
            lambda n, a: [Finding.fail("deny", severity="CRITICAL", message="r")], denials
        )
        held = [{"i": i} for i in range(example.PENDING_LIMIT + 20)]
        for d in held:
            assert before(self.context(example, "run_sql", d)) is False
        assert len(denials._entries()) == example.PENDING_LIMIT
        after = example.recusal_after_tool_call(lambda n, a: [], denials)
        blocked = example.BLOCKED_PREFIX + " Tool: run_sql"
        assert after(self.context(example, "run_sql", held[0], blocked)) is None, "dropped: generic"
        assert (
            after(self.context(example, "run_sql", held[-1], blocked))
            == "Recusal refused `run_sql` [FAIL]: r"
        )

    def test_a_failure_to_keep_the_reason_cannot_unblock(self, example):
        from recusal import Finding

        class Broken(example._Denials):
            def keep(self, tool_input, reason):
                raise RuntimeError("store broke")

            def forget(self, tool_input):
                raise RuntimeError("store broke")

        before = example.recusal_before_tool_call(
            lambda n, a: [Finding.fail("deny", severity="CRITICAL", message="r")], Broken()
        )
        assert before(self.context(example, "run_sql", {"sql": "x"})) is False

    def test_each_sync_runner_invocation_gets_a_fresh_argument_object(self, example):
        """The freshness the correlation relies on, with the objects retained while compared."""
        kept = []

        def retaining(name, args):
            kept.append(args)
            return []

        example.run_calls(
            [("run_sql", {"sql": "SELECT 1"}, "c1"), ("run_sql", {"sql": "SELECT 1"}, "c2")],
            policy=retaining,
        )
        assert len(kept) == 2 and kept[0] is not kept[1]

    def test_the_after_hook_never_judges(self, example):
        calls = {"n": 0}

        def counting(name, args):
            calls["n"] += 1
            return []

        denials = example._Denials()
        after = example.recusal_after_tool_call(counting, denials)
        blocked = example.BLOCKED_PREFIX + " Tool: run_sql"
        after(self.context(example, "run_sql", {"sql": "x"}, blocked))
        assert calls["n"] == 0
