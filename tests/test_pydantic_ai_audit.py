"""Regression tests for the offline Pydantic AI audit-trail example.

Every call the audited gate sees, through the capability or the plain ``Hooks`` function,
lands on a hash-chained log exactly once, labeled ``pydantic_ai.before_tool_execute`` and
keyed by Pydantic AI's own tool call id, and the audited gate decides exactly as the plain
``RecusalGate`` does. One model turn's calls can complete in any order, so entries are
matched by id, never by position.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pydantic_ai_audit.py"
SURFACE = "pydantic_ai.before_tool_execute"
PIN = "pydantic-ai-slim==2.64.0"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"

ROUTES = pytest.mark.parametrize("via_hooks", [False, True], ids=["capability", "hooks"])


@pytest.mark.parametrize("minor", [9, 10])
def test_python_version_refusal_is_clear_on_every_python(minor):
    code = (
        "import runpy, sys; "
        f"sys.version_info = (3, {minor}, 0, 'final', 0); "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "pydantic_ai_audit.py requires Python 3.11+" in result.stderr
    assert f"this is Python 3.{minor}." in result.stderr


def test_pinned_version_is_explicit_in_the_example():
    assert PIN in EXAMPLE.read_text(encoding="utf-8")


def test_both_ci_lanes_run_this_file_with_the_pydantic_ai_example():
    # no new jobs: the two Pydantic AI jobs and the weekly lane run it
    for workflow in ("ci.yml", "compat.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert text.count("tests/test_pydantic_ai_audit.py") == 2, workflow


def test_the_works_with_table_marks_the_audit_trail_tested():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "| Pydantic AI audit trail | Tested | [`examples/pydantic_ai_audit.py`]" in readme


def _sha(args):
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


THREE = [
    ("echo", {"text": "hello"}, "c1"),
    ("run_sql", {"sql": DELETE}, "c2"),
    ("run_sql", {"sql": SELECT}, "c3"),
]


def _corpus():
    def raising(tool_name, tool_input):
        raise RuntimeError("policy exploded")

    def silent(tool_name, tool_input):
        return None  # no objection: PASS

    def ambiguous(tool_name, tool_input):
        return [{"check": "vague"}]  # neither status nor passed: strict fails closed

    def warning_only(tool_name, tool_input):
        from recusal import Finding

        return [Finding.fail("soft", severity="WARNING", message="noted")]

    def error_level(tool_name, tool_input):
        from recusal import Finding

        return [Finding.fail("hard", severity="ERROR", message="retry this")]

    long_sql = "SELECT " + "x" * 50_000 + " FROM t"
    return {
        "sql-scope-three": (None, THREE),
        "sql-scope-long": (None, [("run_sql", {"sql": long_sql}, "c1")]),
        "sql-scope-unicode": (None, [("echo", {"text": "héllo ☃"}, "c1")]),
        "raising": (raising, THREE),
        "silent": (silent, THREE),
        "ambiguous": (ambiguous, THREE),
        "warning-only": (warning_only, THREE),
        "error-level": (error_level, THREE),
    }


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Pydantic AI requires Python 3.11+")
class TestPydanticAiAudit:
    @pytest.fixture
    def example(self):
        pytest.importorskip("pydantic_ai", reason="pydantic-ai-slim is not installed")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location("recusal_pydantic_ai_audit_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def gate(self, example):
        """The plain gate example, the SAME module object the audited one imports from."""
        return importlib.import_module("examples.pydantic_ai_gate")

    @pytest.fixture
    def log_path(self, tmp_path):
        return tmp_path / "pydantic-ai-audit.jsonl"

    @pytest.fixture
    def log(self, example, log_path):
        return example.AuditLog(path=str(log_path))

    @staticmethod
    def by_id(log):
        entries = {}
        for entry in log.entries:
            ident = entry["action"]["tool_use_id"]
            assert ident not in entries, f"{ident} recorded twice"
            entries[ident] = entry
        return entries

    # --- every call recorded once, allowed and refused alike -------------------------------

    @ROUTES
    def test_every_call_is_recorded_once_with_surface_and_id(self, example, log, via_hooks):
        responses, executed = example.run_calls(THREE, log, actor="demo-agent", via_hooks=via_hooks)
        assert {k: v[0] for k, v in responses.items()} == {
            "c1": "success",
            "c2": "failed",
            "c3": "success",
        }
        assert sorted(name for name, _ in executed) == ["echo", "run_sql"]
        entries = self.by_id(log)
        assert set(entries) == {"c1", "c2", "c3"}
        for entry in entries.values():
            assert entry["action"]["surface"] == SURFACE
            assert entry["actor"] == "demo-agent"
        assert {k: v["action"]["decision"] for k, v in entries.items()} == {
            "c1": "defer",
            "c2": "deny",
            "c3": "defer",
        }
        assert {k: v["decision"] for k, v in entries.items()} == {
            "c1": "PASS",
            "c2": "FAIL",
            "c3": "PASS",
        }
        assert entries["c2"]["action"]["tool"] == "run_sql"
        assert "top-level WHERE" in entries["c2"]["reasons"]
        # the reason the model read is the reason on the record
        assert entries["c2"]["action"]["reason"] == responses["c2"][1]

    @ROUTES
    def test_entries_form_one_chain_whatever_the_completion_order(self, example, log, via_hooks):
        many = [("echo", {"text": f"t{i}"}, f"id{i}") for i in range(12)]
        many.insert(5, ("run_sql", {"sql": "DELETE FROM t"}, "refused"))
        example.run_calls(many, log, via_hooks=via_hooks)
        assert len(self.by_id(log)) == 13
        assert [e["seq"] for e in log.entries] == list(range(13))
        for previous, entry in zip(log.entries, log.entries[1:]):
            assert entry["prev_hash"] == previous["hash"]
        from recusal.audit import verify

        ok, problems = verify(log.entries)
        assert ok, problems

    # --- decision equivalence with the plain gate -----------------------------------------

    @ROUTES
    @pytest.mark.parametrize("case", sorted(_corpus()))
    def test_decisions_match_the_plain_gate(self, example, gate, log, case, via_hooks):
        policy, calls = _corpus()[case]
        kwargs = {} if policy is None else {"policy": policy}
        plain = gate.run_calls(calls, via_hooks=via_hooks, **kwargs)
        audited = example.run_calls(calls, log, via_hooks=via_hooks, **kwargs)
        assert audited == plain
        assert len(self.by_id(log)) == len(calls)

    def test_the_audited_capability_is_a_recusal_gate(self, example, gate, log):
        audited = example.RecusalAuditedGate(gate.sql_scope_policy, log=log)
        assert isinstance(audited, gate.RecusalGate)
        assert audited.get_ordering().position == "innermost"
        assert gate.is_gated(gate.scripted_agent([], capabilities=[audited]))

    @ROUTES
    def test_a_refused_call_does_not_run(self, example, log, via_hooks):
        responses, executed = example.run_calls(
            [("run_sql", {"sql": DELETE}, "c1")], log, via_hooks=via_hooks
        )
        assert executed == []
        assert responses["c1"][0] == "failed" and "Recusal refused" in responses["c1"][1]
        assert self.by_id(log)["c1"]["action"]["decision"] == "deny"

    # --- the record is part of the control ---------------------------------------------

    @ROUTES
    def test_policy_that_raises_is_denied_nothing_runs_and_the_denial_is_recorded(
        self, example, gate, log, log_path, via_hooks
    ):
        def broken(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        calls = [("echo", {"text": "x"}, "c1")]
        responses, executed = example.run_calls(calls, log, policy=broken, via_hooks=via_hooks)
        assert executed == []
        plain, _ = gate.run_calls(calls, policy=broken, via_hooks=via_hooks)
        assert responses == plain
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["decision"] == "FAIL"
        assert [f["message"] for f in entry["failures"]] == ["RECUSAL_POLICY_ERROR (RuntimeError)"]
        assert "policy exploded" not in responses["c1"][1]
        assert "policy exploded" not in log_path.read_text(encoding="utf-8")

    @ROUTES
    def test_unwritable_log_refuses_every_call_and_no_tool_runs(
        self, example, log, log_path, via_hooks
    ):
        example.run_calls([("echo", {"text": "warm-up"}, "w")], log, via_hooks=via_hooks)
        assert log_path.is_file()
        log_path.chmod(stat.S_IREAD)
        try:
            with open(log_path, "a", encoding="utf-8"):
                pass
        except PermissionError:
            pass
        else:
            pytest.skip("this filesystem does not enforce a read-only file")
        try:
            responses, executed = example.run_calls(THREE, log, via_hooks=via_hooks)
        finally:
            log_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        assert executed == []
        assert {k: v[0] for k, v in responses.items()} == dict.fromkeys(
            ("c1", "c2", "c3"), "failed"
        )
        for _, content in responses.values():
            assert "audit log unavailable" in content
            assert "the record is part of the control" in content
        # the file holds only the warm-up entry
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1

    @ROUTES
    def test_failing_sink_refuses_the_call_and_no_tool_runs(self, example, log_path, via_hooks):
        class Offline:
            def write(self, entry):
                raise OSError("anchor store unreachable")

        log = example.AuditLog(path=str(log_path), sinks=[Offline()])
        responses, executed = example.run_calls(THREE, log, via_hooks=via_hooks)
        assert executed == []
        assert all(outcome == "failed" for outcome, _ in responses.values())
        assert all("RECUSAL_AUDIT_UNAVAILABLE" in content for _, content in responses.values())
        assert not any("anchor store unreachable" in content for _, content in responses.values())

    # --- the record binds what executed ---------------------------------------------------

    @ROUTES
    def test_no_tool_argument_text_appears_in_the_log(self, example, log, log_path, via_hooks):
        secret = "DELETE FROM customers WHERE ssn = 'MARKER-7f3e9'"
        calls = [
            ("run_sql", {"sql": secret}, "s1"),
            ("echo", {"text": "MARKER-echo-text"}, "s2"),
        ]
        example.run_calls(calls, log, via_hooks=via_hooks)
        text = log_path.read_text(encoding="utf-8")
        assert "MARKER" not in text
        assert "customers" not in text
        entries = self.by_id(log)
        for _, args, call_id in calls:
            assert entries[call_id]["action"]["input_sha256"] == _sha(args)
            assert entries[call_id]["action"]["control"]["recusal_version"]

    @ROUTES
    def test_the_record_binds_what_executed_whatever_the_policy_did_to_its_copy(
        self, example, log, via_hooks
    ):
        def mutating(tool_name, tool_input):
            tool_input["text"] = "REWRITTEN BY THE POLICY"
            tool_input["extra"] = [1, 2, 3]
            return []

        responses, executed = example.run_calls(
            [("echo", {"text": "as proposed"}, "c1")], log, policy=mutating, via_hooks=via_hooks
        )
        assert executed == [("echo", {"text": "as proposed"})]
        entry = self.by_id(log)["c1"]
        assert entry["action"]["input_sha256"] == _sha(executed[0][1])
        assert entry["action"]["input_sha256"] != _sha(
            {"text": "REWRITTEN BY THE POLICY", "extra": [1, 2, 3]}
        )

    def test_the_record_binds_the_arguments_an_earlier_capability_rewrote(self, example, gate, log):
        """The gate is innermost, so it judges, and records, the arguments as they reach the
        tool: a rewrite by an ordinary capability is what ran, and what the entry binds."""

        async def rewrite(ctx, *, call, tool_def, args):
            return {"text": "rewritten before the gate"}

        agent = gate.scripted_agent(
            [("echo", {"text": "as proposed"}, "c1")],
            capabilities=[
                example.RecusalAuditedGate(gate.sql_scope_policy, log=log),
                example.Hooks(before_tool_execute=rewrite),
            ],
        )
        gate.EXECUTED.clear()
        agent.run_sync("go")
        assert gate.EXECUTED == [("echo", {"text": "rewritten before the gate"})]
        entry = self.by_id(log)["c1"]
        assert entry["action"]["input_sha256"] == _sha({"text": "rewritten before the gate"})
        assert entry["action"]["input_sha256"] != _sha({"text": "as proposed"})

    @ROUTES
    def test_a_nested_invocation_is_recorded_as_executed(self, example, log, via_hooks):
        ran = []

        def store(payload: dict, tags: list[str]) -> str:
            """Record a nested payload."""
            ran.append({"payload": payload, "tags": tags})
            return "stored"

        args = {"payload": {"a": [1, {"b": None, "c": 2.5}], "ok": True}, "tags": ["x", "y"]}
        responses, _ = example.run_calls(
            [("store", args, "c1")],
            log,
            policy=lambda *_: [],
            tools=[store],
            via_hooks=via_hooks,
        )
        assert responses["c1"][0] == "success"
        assert ran == [args]
        assert self.by_id(log)["c1"]["action"]["input_sha256"] == _sha(ran[0])

    # --- refused before judgment: still one entry -----------------------------------------

    @ROUTES
    def test_arguments_that_cannot_be_detached_are_one_recorded_deny_bound_to_the_proposal(
        self, example, log, log_path, via_hooks
    ):
        from enum import Enum

        class Action(Enum):
            READ = "MARKER-enum-value"

        seen, ran = [], []

        def record(action: Action) -> str:
            """Record the action."""
            ran.append(action)
            return "recorded"

        def policy(name, args):
            seen.append(args)
            return []

        proposed = {"action": "MARKER-enum-value"}
        responses, _ = example.run_calls(
            [("record", proposed, "c1")], log, policy=policy, tools=[record], via_hooks=via_hooks
        )
        assert seen == ran == []
        outcome, reason = responses["c1"]
        assert outcome == "failed"
        assert reason.startswith("Recusal refused `record` [FAIL]")
        assert "arguments cannot be detached for judging (TypeError)" in reason
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny" and entry["decision"] == "FAIL"
        assert entry["action"]["reason"] == reason
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["input_sha256"] == _sha(proposed)
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    def test_non_mapping_arguments_are_one_recorded_deny(self, example, gate, log):
        record = {"audit": log, "surface": SURFACE, "tool_use_id": "c1"}
        reason = gate._refusal("run_sql", ["DELETE"], gate.sql_scope_policy, **record)
        assert reason is not None and "arguments are not a mapping" in reason
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["input_sha256"] == _sha({"input": ["DELETE"]})
        # without the recording arguments the plain text is unchanged and nothing is written
        assert gate._refusal("run_sql", ["DELETE"], gate.sql_scope_policy) == (
            "Recusal refused `run_sql`: arguments are not a mapping"
        )
        assert len(log.entries) == 1

    def test_a_proposal_the_record_cannot_bind_is_still_one_recorded_deny(self, example, gate, log):
        class Opaque:
            pass

        record = {"audit": log, "surface": SURFACE, "tool_use_id": "c1"}
        reason = gate._refusal(
            "run_sql", {"h": Opaque()}, lambda *_: [], proposal={"h": Opaque()}, **record
        )
        assert reason is not None and reason.startswith("Recusal failed closed")
        assert "RECUSAL_INPUT_BINDING_ERROR" in reason
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert "input_sha256" not in entry["action"]

    def test_a_blank_call_id_is_recorded_as_no_id(self, example, gate, log):
        from pydantic_ai.messages import ToolCallPart

        call = ToolCallPart(tool_name="echo", args={"text": "x"}, tool_call_id="")
        recording = example._recording(call, None, None, log)
        assert recording["tool_use_id"] is None
        assert gate._refusal("echo", {"text": "x"}, lambda *_: [], **recording) is None
        assert len(log.entries) == 1
        assert "tool_use_id" not in log.entries[0]["action"]
        assert log.entries[0]["action"]["input_sha256"] == _sha({"text": "x"})

    def test_the_proposal_of_a_call_never_raises(self, example):
        from pydantic_ai.messages import ToolCallPart

        assert example._proposal(ToolCallPart(tool_name="t", args='{"a": 1}')) == {"a": 1}
        assert example._proposal(ToolCallPart(tool_name="t", args=None)) == {}
        assert isinstance(example._proposal(ToolCallPart(tool_name="t", args="[1, 2")), dict)

    # --- what the hook never sees is not on the log, as the example says -------------------

    @ROUTES
    def test_a_call_that_fails_validation_does_not_run_and_is_not_recorded(
        self, example, log, via_hooks
    ):
        responses, executed = example.run_calls(
            [("run_sql", {"sql": None}, "c1"), ("nope", {"a": 1}, "c2")], log, via_hooks=via_hooks
        )
        assert executed == []
        # Pydantic AI answers both itself, with a retry prompt, before any execute hook
        assert {k: v[0] for k, v in responses.items()} == {"c1": "retried", "c2": "retried"}
        assert log.entries == []

    # --- control identity, verification, tamper detection ---------------------------------

    def test_caller_control_is_recorded_but_cannot_forge_the_version(self, example, log):
        control = {"policy_id": "sql-scope", "policy_version": "3", "recusal_version": "9.9.9"}
        example.run_calls([("echo", {"text": "x"}, "c1")], log, control=control)
        recorded = self.by_id(log)["c1"]["action"]["control"]
        assert recorded["policy_id"] == "sql-scope"
        assert recorded["policy_version"] == "3"
        assert recorded["recusal_version"] != "9.9.9"

    def test_log_verifies_intact_and_editing_any_entry_breaks_it(self, example, log, log_path):
        example.run_calls(THREE, log, actor="demo-agent")
        code, text = example.verify_with_cli(str(log_path))
        assert code == 0, text
        assert "intact - 3 entries" in text
        lines = log_path.read_text(encoding="utf-8").splitlines()
        for index in range(len(lines)):
            edited = list(lines)
            entry = json.loads(edited[index])
            flipped = "allow" if entry["action"]["decision"] != "allow" else "deny"
            entry["action"]["decision"] = flipped
            edited[index] = json.dumps(entry, sort_keys=True)
            copy = log_path.with_name(f"tampered-{index}.jsonl")
            copy.write_text("\n".join(edited) + "\n", encoding="utf-8")
            code, text = example.verify_with_cli(str(copy))
            assert code == 1, (index, text)
            assert "BROKEN" in text

    def test_the_example_tamper_helper_targets_the_refused_call(self, example, log, log_path):
        example.run_calls(THREE, log)
        example.tamper(str(log_path), "c2")
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        edited = [e for e in entries if e["action"]["tool_use_id"] == "c2"]
        assert edited and edited[0]["action"]["decision"] == "allow"
        code, _ = example.verify_with_cli(str(log_path))
        assert code == 1

    # --- negative controls --------------------------------------------------------------

    @ROUTES
    def test_negative_control_plain_gate_writes_no_entries(self, gate, log, log_path, via_hooks):
        responses, _ = gate.run_calls(THREE, via_hooks=via_hooks)
        assert responses["c2"][0] == "failed"
        assert log.entries == []
        assert not log_path.exists()

    def test_negative_control_ungated_agent_records_nothing_and_runs_the_delete(self, gate, log):
        _, executed = gate.run_calls(THREE, gated=False)
        assert ("run_sql", {"sql": DELETE}) in executed
        assert log.entries == []

    # --- the script itself --------------------------------------------------------------

    def test_main_runs_end_to_end_and_shows_the_verifier_refusing(self, example):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)],
            capture_output=True,
            text=True,
            check=False,
            env=dict(os.environ),
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "c2 run_sql  failed" in out
        assert f"c2 {SURFACE} run_sql  deny" in out
        assert "verify intact:   exit 0" in out
        assert "verify tampered: exit 1" in out
        assert "DELETE" not in out

    def test_main_leaves_no_log_behind(self, example, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setenv("TEMP", str(tmp_path))
        monkeypatch.setenv("TMP", str(tmp_path))
        before = set(tmp_path.iterdir())
        example.main()
        assert set(tmp_path.iterdir()) == before
