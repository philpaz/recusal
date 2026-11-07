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
        tool = example.run_sql.to_structured_tool()
        assert example._refusal(tool, "run_sql", ["x"], example.sql_scope_policy) == (
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
        # the real structured tool for the name: the gate judges the invocation the tool
        # receives, and refuses a call that carries no tool object
        kwargs = {
            "tool_name": name,
            "tool_input": tool_input,
            "tool": getattr(example, name).to_structured_tool(),
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
        held = [{"sql": f"q{i}"} for i in range(example.PENDING_LIMIT + 20)]
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


def _same(a, b) -> bool:
    """Exact agreement: the same type at every node, then equal."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_the_exact_comparator_refuses_look_alikes():
    assert not _same(True, 1) and not _same(1, 1.0) and not _same({"a": 1}, {"a": "1"})
    assert not _same(0, False) and _same({"a": [1, 2.0]}, {"a": [1, 2.0]})


@pytest.mark.skipif(sys.version_info < (3, 10), reason="CrewAI requires Python 3.10+")
class TestPreparedInvocation:
    """Adapters review, finding 3 (CrewAI): the before hook judged the dict the model
    proposed; CrewAI then validated it against the tool's ``args_schema`` at execution,
    filling omitted defaults and coercing values, so ``charge(amount: int = 100)`` proposed
    with ``{}`` reached the policy as ``{}`` and ran 100. The hook now completes defaults
    into the raw input, validates once, writes the validated values back in place and judges
    a copy; it supports only a stock tool and a stock schema, with every field compared
    against the canonical ``FieldInfo`` its declaration produces (the v4 contract)."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("crewai", reason="crewai is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_crewai_prepared_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        from crewai.hooks import clear_after_tool_call_hooks, clear_before_tool_call_hooks

        clear_before_tool_call_hooks()
        clear_after_tool_call_hooks()
        yield module
        clear_before_tool_call_hooks()
        clear_after_tool_call_hooks()

    # --- helpers ---------------------------------------------------------------------------

    @staticmethod
    def recording(inner=None):
        import copy

        class Recording:
            def __init__(self):
                self.seen = []

            def __call__(self, name, args):
                self.seen.append((name, copy.deepcopy(args)))
                return inner(name, args) if inner is not None else []

        return Recording()

    @staticmethod
    def budget(name, args):
        from recusal import Finding

        if args.get("amount", 0) > 10:
            return [Finding.fail("budget", severity="CRITICAL", message="over 10")]
        return []

    @staticmethod
    def tool_with(example, schema, name="t"):
        """A BaseTool with the given argument schema whose ``_run`` records what it got."""

        class Recorder(example.BaseTool):
            name: str = "t"
            description: str = "records its arguments"
            args_schema: type[example.BaseModel] = schema

            def _run(self, **kwargs):
                example.EXECUTED.append((self.name, kwargs))
                return "ok"

        return Recorder(name=name)

    @staticmethod
    def context(example, tool, name, tool_input):
        return example.ToolCallHookContext(
            tool_name=name, tool_input=tool_input, tool=tool, agent=None, task=None, crew=None
        )

    def reason(self, example, tool, tool_input, name="t"):
        """The gate's refusal for this call at the hook, asserting the policy never ran."""
        policy = self.recording()
        before = example.recusal_before_tool_call(policy)
        assert before(self.context(example, tool, name, tool_input)) is False
        assert policy.seen == []
        reason = example._refusal(tool, name, dict(tool_input), policy)
        assert reason is not None and reason.startswith(f"Recusal refused `{name}`"), reason
        return reason

    # --- through the real Crew runner: judged equals received, exactly ---------------------

    def test_an_omitted_default_is_judged_at_the_value_the_function_uses(self, example):
        policy = self.recording(self.budget)
        responses, executed = example.run_calls([("charge", {}, "c1")], policy=policy)
        assert policy.seen == [("charge", {"amount": 100})]
        assert type(policy.seen[0][1]["amount"]) is int
        assert executed == []
        assert responses["c1"] == "Recusal refused `charge` [FAIL]: over 10"
        policy = self.recording(self.budget)
        responses, executed = example.run_calls([("charge", {"amount": 5}, "c2")], policy=policy)
        assert policy.seen == [("charge", {"amount": 5})]
        assert executed == [("charge", {"amount": 5})]
        assert responses["c2"] == "recorded"

    def test_a_callable_parameter_the_schema_omits_refuses_before_the_policy(self, example):
        """The 2026-11-06 review's P1 through the real Crew runner: ``_run(label, amount=100)``
        behind an ``args_schema`` declaring only ``label`` was judged as ``{"label": ...}``
        and ran with ``amount=100``."""
        policy = self.recording(self.budget)
        responses, executed = example.run_calls(
            [("charge", {"label": "review"}, "c1")],
            policy=policy,
            tools=[example.narrow_charge_tool()],
        )
        assert executed == [] and policy.seen == []
        assert responses["c1"] == (
            "Recusal refused `charge`: its arguments cannot be judged "
            "(its callable's parameter `amount` is absent from the argument schema)"
        )

    def test_negative_control_ungated_the_schema_absent_default_runs(self, example):
        _, executed = example.run_calls(
            [("charge", {"label": "review"}, "c1")],
            gated=False,
            tools=[example.narrow_charge_tool()],
        )
        assert executed == [("charge", {"label": "review", "amount": 100})]

    def test_a_stock_tool_wrapper_is_inspected_at_the_function_it_wraps(self, example):
        """``Tool._run(*args, **kwargs)`` forwards to ``func``: the wrapper's signature says
        nothing, the function's is the one Python completes."""
        from pydantic import BaseModel

        class LabelOnly(BaseModel):
            label: str

        class Full(BaseModel):
            label: str
            amount: int = 100

        def charge(label: str, amount: int = 100) -> str:
            return "recorded"

        narrow = example.Tool(
            name="charge", description="d", func=charge, args_schema=LabelOnly
        ).to_structured_tool()
        assert example._unsupported_tool(narrow) == (
            "its callable's parameter `amount` is absent from the argument schema"
        )
        assert "cannot be judged" in self.reason(
            example, narrow, {"label": "review"}, name="charge"
        )
        complete = example.Tool(
            name="charge", description="d", func=charge, args_schema=Full
        ).to_structured_tool()
        assert example._execution_callable(complete) is charge
        assert example._unsupported_tool(complete) is None

    def test_a_structured_tool_whose_callable_is_not_the_originals_run_is_refused(self, example):
        tool = example.charge.to_structured_tool()
        assert example._unsupported_tool(tool) is None
        tool.func = lambda **kwargs: "elsewhere"
        assert example._unsupported_tool(tool) == "its callable is not the original tool's `_run`"

    def test_a_positional_only_callable_parameter_is_refused(self, example):
        from pydantic import BaseModel

        class LabelOnly(BaseModel):
            label: str

        def charge(label: str, /) -> str:
            return "recorded"

        tool = example.Tool(
            name="charge", description="d", func=charge, args_schema=LabelOnly
        ).to_structured_tool()
        assert (
            example._unsupported_tool(tool) == "its callable's parameter `label` is positional-only"
        )

    def test_a_non_canonical_default_is_prepared_before_judgment(self, example):
        @example.tool("loose")
        def loose(amount: int = "100", ratio: float = 1, flag: bool = "true") -> str:  # type: ignore[assignment]
            """Defaults in the JSON domain but not the annotation's canonical value."""
            example.EXECUTED.append(("loose", {"amount": amount, "ratio": ratio, "flag": flag}))
            return "ok"

        policy = self.recording()
        _, executed = example.run_calls([("loose", {}, "c1")], policy=policy, tools=[loose])
        assert executed == [("loose", {"amount": 100, "ratio": 1.0, "flag": True})]
        assert _same(policy.seen[0][1], executed[0][1]), (policy.seen, executed)
        assert type(executed[0][1]["amount"]) is int and type(executed[0][1]["ratio"]) is float
        # the naive hook would have judged the raw defaults
        assert not _same({"amount": "100", "ratio": 1, "flag": "true"}, executed[0][1])

    def test_an_explicit_string_is_prepared_once_and_judged_as_received(self, example):
        policy = self.recording(self.budget)
        _, executed = example.run_calls([("charge", {"amount": "7"}, "c1")], policy=policy)
        assert executed == [("charge", {"amount": 7})]
        assert _same(policy.seen[0][1], {"amount": 7})
        policy = self.recording(self.budget)
        _, executed = example.run_calls([("charge", {"amount": "70"}, "c2")], policy=policy)
        assert executed == [] and _same(policy.seen[0][1], {"amount": 70})

    def test_a_nested_container_default_is_judged_and_received_exactly(self, example):
        @example.tool("nested")
        def nested(tags: list[int] = [1, "2"], meta: dict[str, float] = {"k": 1}) -> str:  # type: ignore[list-item]  # noqa: B006
            """Container defaults the schema coerces at validation."""
            example.EXECUTED.append(("nested", {"tags": tags, "meta": meta}))
            return "ok"

        policy = self.recording()
        _, executed = example.run_calls([("nested", {}, "c1")], policy=policy, tools=[nested])
        assert executed == [("nested", {"tags": [1, 2], "meta": {"k": 1.0}})]
        assert _same(policy.seen[0][1], executed[0][1])
        assert type(executed[0][1]["meta"]["k"]) is float

    def test_a_described_default_on_a_user_schema_is_supported_and_judged_exactly(self, example):
        class Args(example.BaseModel):
            amount: int = example.Field(default=100, description="the amount", title="Amount")

        tool = self.tool_with(example, Args, name="charge")
        policy = self.recording(self.budget)
        responses, executed = example.run_calls([("charge", {}, "c1")], policy=policy, tools=[tool])
        assert policy.seen == [("charge", {"amount": 100})] and executed == []
        assert responses["c1"] == "Recusal refused `charge` [FAIL]: over 10"
        policy = self.recording(self.budget)
        _, executed = example.run_calls(
            [("charge", {"amount": "5"}, "c2")], policy=policy, tools=[tool]
        )
        assert executed == [("charge", {"amount": 5})] and _same(policy.seen[0][1], {"amount": 5})

    def test_a_mutating_policy_cannot_change_the_prepared_invocation(self, example):
        def mutating(name, args):
            args["amount"] = 1
            args["extra"] = "x"
            return []

        _, executed = example.run_calls([("charge", {}, "c1")], policy=mutating)
        assert executed == [("charge", {"amount": 100})]
        _, executed = example.run_calls([("charge", {"amount": 50}, "c2")], policy=mutating)
        assert executed == [("charge", {"amount": 50})]

    def test_negative_control_the_ungated_runner_runs_the_omitted_default(self, example):
        policy = self.recording(self.budget)
        responses, executed = example.run_calls([("charge", {}, "c1")], policy=policy, gated=False)
        assert executed == [("charge", {"amount": 100})] and policy.seen == []
        assert responses["c1"] == "recorded"

    def test_a_field_with_a_stateful_exclude_if_is_refused_before_execution(self, example):
        state = {"n": 0}

        def flip(value):
            state["n"] += 1
            return state["n"] % 2 == 0

        class Args(example.BaseModel):
            amount: int = example.Field(default=100, exclude_if=flip)

        policy = self.recording()
        responses, executed = example.run_calls(
            [("charge", {}, "c1")], policy=policy, tools=[self.tool_with(example, Args, "charge")]
        )
        assert executed == [] and policy.seen == []
        assert responses["c1"].startswith(
            "Recusal refused `charge`: its arguments cannot be judged"
        )
        assert "field `amount` carries a setting" in responses["c1"]

    def test_the_rewriting_schema_fixture_is_refused_and_the_rewrite_never_runs(self, example):
        policy = self.recording(example.sql_scope_policy)
        responses, executed = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")],
            policy=policy,
            tools=[example.transforming_run_sql()],
        )
        assert executed == [] and policy.seen == []
        assert "Recusal refused `run_sql`" in responses["c1"] and "_rewrite" in responses["c1"]

    # --- the write-back and the single preparation ------------------------------------------

    def test_the_hook_replaces_the_dicts_contents_in_place_with_the_prepared_invocation(
        self, example
    ):
        tool = example.charge.to_structured_tool()
        proposed = {"amount": "7"}
        policy = self.recording()
        before = example.recusal_before_tool_call(policy)
        context = self.context(example, tool, "charge", proposed)
        assert before(context) is None
        assert context.tool_input is proposed, "CrewAI executes this very object"
        assert _same(proposed, {"amount": 7}) and _same(policy.seen[0][1], {"amount": 7})

    def test_re_preparation_at_execution_cannot_change_the_judged_value(self, example):
        """CrewAI validates again when it runs the tool (``_parse_args`` on the structured
        tool, ``_validate_kwargs`` on the original): on the supported set both are identity
        on the prepared invocation."""
        from typing import Optional

        @example.tool("wide")
        def wide(
            amount: int = "100",  # type: ignore[assignment]
            ratio: float = 1,
            flag: bool = "true",  # type: ignore[assignment]
            tags: list[int] = [],  # noqa: B006
            meta: dict[str, float] = {},  # noqa: B006
            note: Optional[str] = None,
        ) -> str:
            """Every supported shape."""
            return "ok"

        tool = wide.to_structured_tool()
        for proposed in ({}, {"amount": "7", "ratio": 2, "flag": "false", "tags": ["3"]}):
            judged, why = example._prepared(tool, dict(proposed))
            assert why is None, why
            assert _same(tool._parse_args(dict(judged)), judged)
            assert _same(tool._original_tool._validate_kwargs(dict(judged)), judged)

    # --- refused before judgment: schema, field, tool, input ---------------------------------

    def test_crewais_own_generated_schemas_are_stock(self, example):
        assert example._unsupported_schema(example.charge.args_schema) is None
        assert example._unsupported_schema(example.run_sql.args_schema) is None

        class Plain(example.BaseTool):
            name: str = "plain"
            description: str = "no args_schema of its own"

            def _run(self, amount: int = 100, note: str = "x") -> str:
                return "ok"

        assert example._unsupported_tool(Plain().to_structured_tool()) is None

    def test_field_settings_beyond_the_contract_are_refused(self, example):
        from pydantic import Field, create_model

        rows = {
            "exclude": Field(default=100, exclude=True),
            "exclude_if": Field(default=100, exclude_if=lambda v: v > 5),
            "default_factory": Field(default_factory=lambda: 100),
            "alias": Field(default=100, alias="amt"),
            "gt": Field(default=100, gt=0),
            "validate_default": Field(default=100, validate_default=True),
            "frozen": Field(default=100, frozen=True),
            "json_schema_extra": Field(default=100, json_schema_extra={"x": 1}),
            "deprecated": Field(default=100, deprecated=True),
        }
        for label, field in rows.items():
            schema = create_model("S", amount=(int, field))
            why = example._unsupported_schema(schema)
            # deprecated=True also puts a descriptor named `amount` on the class, which the
            # class check refuses first; every other row fails the canonical comparison
            assert why is not None and "`amount`" in why, (label, why)
            if label != "deprecated":
                assert "field `amount` carries a setting" in why, (label, why)
            tool = self.tool_with(example, schema, name="charge").to_structured_tool()
            assert "cannot be judged" in self.reason(example, tool, {}, name="charge"), label

    def test_fields_within_the_contract_are_supported_and_completed(self, example):
        from typing import Optional

        from pydantic import ConfigDict, Field, create_model

        schema = create_model(
            "S",
            __config__=ConfigDict(extra="forbid", title="Charge"),
            amount=(int, Field(default=100, description="the amount")),
            note=(str, Field(default="x", title="Note")),
            tags=(list[str], []),
            meta=(dict[str, int], {}),
            maybe=(Optional[int], None),
            required=(str, ...),
        )
        assert example._unsupported_schema(schema) is None
        tool = self.tool_with(example, schema).to_structured_tool()
        policy = self.recording()
        before = example.recusal_before_tool_call(policy)
        assert before(self.context(example, tool, "t", {"required": "r"})) is None
        expected = {
            "amount": 100,
            "note": "x",
            "tags": [],
            "meta": {},
            "maybe": None,
            "required": "r",
        }
        assert _same(policy.seen[0][1], expected), policy.seen

    def test_unsupported_annotations_are_refused(self, example):
        import enum
        from typing import Annotated, Any

        from pydantic import BaseModel, Field, create_model

        class Color(enum.Enum):
            RED = "red"

        class Inner(BaseModel):
            x: int = 1

        rows = {
            "Annotated": (Annotated[int, Field(gt=0)], 100),
            "Enum": (Color, Color.RED),
            "Any": (Any, 100),
            "model": (Inner, Inner()),
            "bare list": (list, []),
            "list of Any": (list[Any], []),
            "dict with int keys": (dict[int, int], {}),
        }
        for label, declaration in rows.items():
            why = example._unsupported_schema(create_model("S", amount=declaration))
            assert why is not None, label
            assert "field `amount`" in why, (label, why)

        @example.tool("unannotated")
        def unannotated(amount=100) -> str:
            """CrewAI gives an unannotated parameter the Any annotation."""
            return "ok"

        why = self.reason(example, unannotated.to_structured_tool(), {}, name="unannotated")
        assert "field `amount` is not annotated with a supported data type" in why

    def test_a_default_outside_the_json_domain_is_refused(self, example):
        import decimal

        from pydantic import create_model

        why = example._unsupported_schema(create_model("S", amount=(int, decimal.Decimal(1))))
        assert why == "field `amount` has a default outside the JSON domain"

    def test_customised_model_classes_are_refused(self, example):
        from pydantic import BaseModel, ConfigDict, create_model, field_validator

        class OverridesValidate(BaseModel):
            amount: int = 100

            @classmethod
            def model_validate(cls, *args, **kwargs):
                return super().model_validate(*args, **kwargs)

        class Base(BaseModel):
            @field_validator("*")
            @classmethod
            def _touch(cls, value):
                return value

        class IntermediateBase(Base):
            amount: int = 100

        class PostInit(BaseModel):
            amount: int = 100

            def model_post_init(self, context):
                self.amount = 1

        class CoreSchema(BaseModel):
            amount: int = 100

            @classmethod
            def __get_pydantic_core_schema__(cls, source, handler):
                return handler(source)

        class Decorated(BaseModel):
            amount: int = 100

            @field_validator("amount")
            @classmethod
            def _rewrite(cls, value):
                return 1

        class Configured(BaseModel):
            model_config = ConfigDict(str_strip_whitespace=True)
            amount: int = 100

        class WithProperty(BaseModel):
            amount: int = 100

            @property
            def doubled(self):
                return self.amount * 2

        class WithAttribute(BaseModel):
            amount: int = 100
            limit: int = 5  # a plain annotated attribute is a field; this is not
            __slots__ = ()

        rows = {
            "model_validate": (OverridesValidate, "defines `model_validate`"),
            "intermediate base": (IntermediateBase, "base class of its own"),
            "model_post_init": (PostInit, "defines `model_post_init`"),
            "core schema": (CoreSchema, "defines `__get_pydantic_core_schema__`"),
            "field_validator": (Decorated, "defines `_rewrite`"),
            "configuration": (Configured, "sets configuration `str_strip_whitespace`"),
            "property": (WithProperty, "defines `doubled`"),
            "slots": (WithAttribute, "defines `__slots__`"),
            "create_model validators": (
                create_model(
                    "S",
                    amount=(int, 100),
                    __validators__={"v": field_validator("amount")(classmethod(lambda c, v: 1))},
                ),
                "defines `v`",
            ),
        }
        for label, (schema, expected) in rows.items():
            why = example._unsupported_schema(schema)
            assert why is not None and expected in why, (label, why)
            tool = self.tool_with(example, schema, name="charge").to_structured_tool()
            assert expected in self.reason(example, tool, {}, name="charge"), label
        assert example._unsupported_schema(dict) == "its argument schema is not a Pydantic model"

    def test_tool_classes_that_change_validation_or_running_are_refused(self, example):
        class Args(example.BaseModel):
            amount: int = 100

        class LooseStructured(example.CrewStructuredTool):
            def _parse_args(self, raw_args):
                return raw_args if isinstance(raw_args, dict) else {}

        loose = LooseStructured(
            name="charge", description="d", args_schema=Args, func=lambda amount=100: "ok"
        )
        assert "overrides `_parse_args`" in self.reason(example, loose, {}, name="charge")

        class OwnRun(example.BaseTool):
            name: str = "charge"
            description: str = "d"
            args_schema: type[example.BaseModel] = Args

            def run(self, *args, **kwargs):
                return self._run(**kwargs)

            def _run(self, **kwargs):
                return "ok"

        assert "overrides `run`" in self.reason(
            example, OwnRun().to_structured_tool(), {}, name="charge"
        )

        class OwnValidation(example.BaseTool):
            name: str = "charge"
            description: str = "d"
            args_schema: type[example.BaseModel] = Args

            def _validate_kwargs(self, kwargs):
                return kwargs

            def _run(self, **kwargs):
                return "ok"

        assert "overrides `_validate_kwargs`" in self.reason(
            example, OwnValidation().to_structured_tool(), {}, name="charge"
        )
        assert "no CrewAI tool object" in self.reason(example, None, {}, name="charge")

    def test_unknown_missing_and_invalid_arguments_are_refused_before_judgment(self, example):
        tool = example.charge.to_structured_tool()
        assert self.reason(example, tool, {"amount": 5, "zzz": 1}, "charge") == (
            "Recusal refused `charge`: `zzz` is not a parameter of the tool"
        )
        required = self.tool_with(
            example, example.run_sql.args_schema, "run_sql"
        ).to_structured_tool()
        assert self.reason(example, required, {}, "run_sql") == (
            "Recusal refused `run_sql`: required parameter `sql` is missing"
        )
        why = self.reason(example, tool, {"amount": "secret-value"}, "charge")
        assert why == "Recusal refused `charge`: validation failed for `amount`"
        assert "secret-value" not in why

    def test_a_crash_inside_preparation_blocks_the_call(self, example, monkeypatch):
        def boom(tool, tool_input):
            raise RuntimeError("bookkeeping bug")

        monkeypatch.setattr(example, "_prepared", boom)
        policy = self.recording()
        before = example.recusal_before_tool_call(policy)
        tool = example.charge.to_structured_tool()
        assert before(self.context(example, tool, "charge", {})) is False
        assert policy.seen == []

    # --- the pins that keep the whitelist a whitelist ----------------------------------------

    def test_the_generated_names_of_a_stock_model_stay_within_the_allowed_set(self, example):
        from pydantic import BaseModel, create_model

        class Statement(BaseModel):
            amount: int = 100

        allowed = lambda n: n in example._GENERATED_NAMES or example._PYDANTIC_FAMILY.match(n)  # noqa: E731
        for schema in (create_model("S", amount=(int, 100)), Statement, example.charge.args_schema):
            own = sorted(n for n in vars(schema) if not allowed(n))
            assert own == [], (schema, own)

    def test_the_canonical_form_of_a_bare_declaration_sets_only_annotation_and_default(
        self, example
    ):
        from pydantic import Field, create_model

        fields = create_model(
            "S", a=(int, 100), b=(int, ...), c=(int, Field(default=1, description="d"))
        ).model_fields
        assert set(example._canonical_field(fields["a"])._attributes_set) == {
            "annotation",
            "default",
        }
        assert set(example._canonical_field(fields["b"])._attributes_set) == {"annotation"}
        assert set(example._canonical_field(fields["c"])._attributes_set) == {
            "annotation",
            "default",
            "description",
        }
        for name in fields:
            assert example._unsupported_field(name, fields[name]) is None, name
