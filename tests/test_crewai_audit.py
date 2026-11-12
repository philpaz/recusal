"""Regression tests for the offline CrewAI audit-trail example.

Every call the audited before hook sees lands on a hash-chained log exactly once, labeled
``crewai.before_tool_call``, and the audited hooks decide exactly as the plain ones do.
CrewAI's hook context carries no call id, so entries are matched by tool and fingerprint.
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
EXAMPLE = ROOT / "examples" / "crewai_audit.py"
SURFACE = "crewai.before_tool_call"
PIN = "crewai==1.15.34"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"


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
    assert "crewai_audit.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_pinned_version_is_explicit_in_the_example():
    assert PIN in EXAMPLE.read_text(encoding="utf-8")


def test_both_ci_lanes_run_this_file_with_the_crewai_example():
    # no new jobs: the two CrewAI jobs and the weekly lane run it
    for workflow in ("ci.yml", "compat.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert text.count("tests/test_crewai_audit.py") == 2, workflow


def test_the_works_with_table_marks_the_audit_trail_tested():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    assert "| CrewAI audit trail | Tested | [`examples/crewai_audit.py`]" in table


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

    def error_level(tool_name, tool_input):
        from recusal import Finding

        return [Finding.fail("hard", severity="ERROR", message="retry this")]

    return {
        "sql-scope-three": (None, THREE),
        "sql-scope-unicode": (None, [("echo", {"text": "héllo ☃"}, "c1")]),
        "default-omitted": (None, [("charge", {}, "c1")]),
        "raising": (raising, THREE),
        "silent": (silent, THREE),
        "ambiguous": (ambiguous, THREE),
        "error-level": (error_level, THREE),
    }


@pytest.mark.skipif(sys.version_info < (3, 10), reason="CrewAI requires Python 3.10+")
class TestCrewAiAudit:
    @pytest.fixture
    def example(self):
        pytest.importorskip("crewai", reason="crewai is not installed")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location("recusal_crewai_audit_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def gate(self, example):
        """The plain gate example, the SAME module object the audited one imports from."""
        return importlib.import_module("examples.crewai_gate")

    @pytest.fixture
    def log_path(self, tmp_path):
        return tmp_path / "crewai-audit.jsonl"

    @pytest.fixture
    def log(self, example, log_path):
        return example.AuditLog(path=str(log_path))

    @staticmethod
    def by_input(log):
        entries = {}
        for entry in log.entries:
            key = (entry["action"]["tool"], entry["action"]["input_sha256"])
            assert key not in entries, f"{key} recorded twice"
            entries[key] = entry
        return entries

    # --- every call recorded once, allowed and refused alike -------------------------------

    def test_every_call_is_recorded_once_with_surface_and_agent(self, example, log):
        responses, executed = example.run_calls(THREE, log, actor="demo-agent")
        assert sorted(name for name, _ in executed) == ["echo", "run_sql"]
        entries = self.by_input(log)
        assert set(entries) == {(name, _sha(args)) for name, args, _ in THREE}
        for entry in entries.values():
            assert entry["action"]["surface"] == SURFACE
            assert entry["actor"] == "demo-agent"
            assert entry["action"]["control"]["crewai_agent"] == "assistant"
            # CrewAI's hook context has no call id, and the record does not invent one
            assert "tool_use_id" not in entry["action"]
        refused = entries[("run_sql", _sha({"sql": DELETE}))]
        assert refused["action"]["decision"] == "deny" and refused["decision"] == "FAIL"
        assert "top-level WHERE" in refused["reasons"]
        # the reason the model read is the reason on the record
        assert refused["action"]["reason"] == responses["c2"]
        assert entries[("run_sql", _sha({"sql": SELECT}))]["action"]["decision"] == "defer"

    def test_entries_form_one_chain(self, example, log):
        many = [("echo", {"text": f"t{i}"}, f"id{i}") for i in range(8)]
        many.insert(3, ("run_sql", {"sql": "DELETE FROM t"}, "refused"))
        example.run_calls(many, log)
        assert len(self.by_input(log)) == 9
        assert [e["seq"] for e in log.entries] == list(range(9))
        from recusal.audit import verify

        ok, problems = verify(log.entries)
        assert ok, problems

    # --- decision equivalence with the plain gate -----------------------------------------

    @pytest.mark.parametrize("case", sorted(_corpus()))
    def test_decisions_match_the_plain_gate(self, example, gate, log, case):
        policy, calls = _corpus()[case]
        kwargs = {} if policy is None else {"policy": policy}
        plain_responses, plain_executed = gate.run_calls(calls, **kwargs)
        responses, executed = example.run_calls(calls, log, **kwargs)
        assert responses == plain_responses
        assert sorted(map(repr, executed)) == sorted(map(repr, plain_executed))
        assert len(log.entries) == len(calls)

    def test_a_refused_call_does_not_run(self, example, log):
        responses, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")], log)
        assert executed == []
        assert responses["c1"].startswith("Recusal refused `run_sql` [FAIL]")
        assert log.entries[0]["action"]["decision"] == "deny"

    # --- the record is part of the control ---------------------------------------------

    def test_policy_that_raises_is_denied_and_the_denial_is_recorded_without_its_text(
        self, example, gate, log, log_path
    ):
        def broken(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        calls = [("echo", {"text": "x"}, "c1")]
        responses, executed = example.run_calls(calls, log, policy=broken)
        assert executed == []
        assert responses == gate.run_calls(calls, policy=broken)[0]
        entry = log.entries[0]
        assert entry["action"]["decision"] == "deny" and entry["decision"] == "FAIL"
        assert [f["message"] for f in entry["failures"]] == ["RECUSAL_POLICY_ERROR (RuntimeError)"]
        assert "policy exploded" not in responses["c1"]
        assert "policy exploded" not in log_path.read_text(encoding="utf-8")

    def test_unwritable_log_refuses_every_call_and_no_tool_runs(self, example, log, log_path):
        example.run_calls([("echo", {"text": "warm-up"}, "w")], log)
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
            responses, executed = example.run_calls(THREE, log)
        finally:
            log_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        assert executed == []
        assert set(responses) == {"c1", "c2", "c3"}
        for content in responses.values():
            assert "audit log unavailable" in content
            assert "the record is part of the control" in content
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1

    def test_failing_sink_refuses_the_call_and_no_tool_runs(self, example, log_path):
        class Offline:
            def write(self, entry):
                raise OSError("anchor store unreachable")

        log = example.AuditLog(path=str(log_path), sinks=[Offline()])
        responses, executed = example.run_calls(THREE, log)
        assert executed == []
        assert all("RECUSAL_AUDIT_UNAVAILABLE" in content for content in responses.values())
        assert not any("anchor store unreachable" in content for content in responses.values())

    def test_a_recording_that_raises_blocks_the_call(self, example, gate):
        def broken(context):
            raise RuntimeError("RECORDING_SECRET_SENTINEL")

        gate.EXECUTED.clear()
        uninstall = gate.install_recusal_hooks(lambda *_: [], broken)
        try:
            responses = gate.run_crew(gate.scripted_agent([("echo", {"text": "x"}, "c1")]))
        finally:
            uninstall()
        assert gate.EXECUTED == []
        assert responses["c1"].startswith(gate.BLOCKED_PREFIX)
        assert "RECORDING_SECRET_SENTINEL" not in responses["c1"]

    # --- the record binds what executed ---------------------------------------------------

    def test_no_tool_argument_text_appears_in_the_log(self, example, log, log_path):
        secret = "DELETE FROM customers WHERE ssn = 'MARKER-7f3e9'"
        calls = [("run_sql", {"sql": secret}, "s1"), ("echo", {"text": "MARKER-echo"}, "s2")]
        example.run_calls(calls, log)
        text = log_path.read_text(encoding="utf-8")
        assert "MARKER" not in text and "customers" not in text
        assert set(self.by_input(log)) == {(name, _sha(args)) for name, args, _ in calls}

    def test_a_default_the_model_omitted_is_recorded_at_the_value_that_runs(self, example, log):
        _, executed = example.run_calls([("charge", {}, "c1")], log, policy=lambda *_: [])
        assert executed == [("charge", {"amount": 100})]
        assert log.entries[0]["action"]["input_sha256"] == _sha({"amount": 100})
        assert log.entries[0]["action"]["input_sha256"] != _sha({})

    def test_a_coerced_value_is_recorded_as_the_tool_receives_it(self, example, log):
        _, executed = example.run_calls(
            [("charge", {"amount": "7"}, "c1")], log, policy=lambda *_: []
        )
        assert executed == [("charge", {"amount": 7})]
        assert log.entries[0]["action"]["input_sha256"] == _sha({"amount": 7})

    def test_the_record_binds_what_executed_whatever_the_policy_did_to_its_copy(self, example, log):
        def mutating(tool_name, tool_input):
            tool_input["text"] = "REWRITTEN BY THE POLICY"
            return []

        _, executed = example.run_calls(
            [("echo", {"text": "as proposed"}, "c1")], log, policy=mutating
        )
        assert executed == [("echo", {"text": "as proposed"})]
        assert log.entries[0]["action"]["input_sha256"] == _sha(executed[0][1])

    # --- refused before judgment: still one entry, bound to the proposal --------------------

    def test_every_refusal_before_judgment_is_one_recorded_deny(self, example, log, log_path):
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        calls = [
            ("run_sql", {"sql": SELECT, "surprise": "MARKER-extra"}, "unknown_param"),
            ("run_sql", {}, "missing"),
            ("charge", {"amount": "MARKER-not-an-int"}, "invalid"),
        ]
        responses, executed = example.run_calls(calls, log, policy=policy)
        assert executed == [] and seen == []
        details = {
            "unknown_param": "`surprise` is not a parameter of the tool",
            "missing": "required parameter `sql` is missing",
            "invalid": "validation failed for `amount`",
        }
        entries = self.by_input(log)
        assert len(entries) == 3
        for name, args, call_id in calls:
            entry = entries[(name, _sha(args))]
            reason = responses[call_id]
            assert reason.startswith(f"Recusal refused `{name}` [FAIL]")
            assert details[call_id] in reason
            assert entry["action"]["decision"] == "deny" and entry["decision"] == "FAIL"
            assert entry["action"]["reason"] == reason
            assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    def test_non_mapping_arguments_are_one_recorded_deny(self, example, gate, log):
        record = {"audit": log, "surface": SURFACE}
        reason = gate._refusal(None, "run_sql", ["DELETE"], gate.sql_scope_policy, **record)
        assert reason is not None and "arguments are not a mapping" in reason
        entry = log.entries[0]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["input_sha256"] == _sha({"input": ["DELETE"]})
        # without the recording arguments the plain text is unchanged and nothing is written
        assert gate._refusal(None, "run_sql", ["DELETE"], gate.sql_scope_policy) == (
            "Recusal refused `run_sql`: arguments are not a mapping"
        )
        assert len(log.entries) == 1

    # --- what the gate never sees is not on the log, as the example says -------------------

    def test_a_call_an_earlier_hook_blocked_does_not_run_and_is_not_recorded(
        self, example, gate, log
    ):
        from crewai.hooks import (
            register_before_tool_call_hook,
            unregister_before_tool_call_hook,
        )

        def block_everything(context):
            return False

        register_before_tool_call_hook(block_everything)
        try:
            responses, executed = example.run_calls([("echo", {"text": "x"}, "c1")], log)
        finally:
            unregister_before_tool_call_hook(block_everything)
        assert executed == []
        assert responses["c1"].startswith(gate.BLOCKED_PREFIX)
        assert log.entries == []

    # --- control identity, verification, tamper detection ---------------------------------

    def test_caller_control_is_recorded_but_cannot_forge_the_agent_or_the_version(
        self, example, log
    ):
        control = {"policy_id": "sql-scope", "crewai_agent": "forged", "recusal_version": "9.9.9"}
        example.run_calls([("echo", {"text": "x"}, "c1")], log, control=control)
        recorded = log.entries[0]["action"]["control"]
        assert recorded["policy_id"] == "sql-scope"
        assert recorded["crewai_agent"] == "assistant"
        assert recorded["recusal_version"] != "9.9.9"

    def test_a_context_without_an_agent_records_no_agent(self, example, log):
        class Context:
            agent = None

        record = example.recording(log, control={"crewai_agent": "forged"})(Context())
        assert record["control"] is None
        assert "tool_use_id" not in record

    def test_log_verifies_intact_and_editing_any_entry_breaks_it(self, example, log, log_path):
        example.run_calls(THREE, log, actor="demo-agent")
        code, text = example.verify_with_cli(str(log_path))
        assert code == 0, text
        assert "intact - 3 entries" in text
        for seq in range(3):
            tampered = log_path.with_name(f"tampered-{seq}.jsonl")
            tampered.write_bytes(log_path.read_bytes())
            flipped = "allow" if log.entries[seq]["action"]["decision"] != "allow" else "deny"
            example.tamper(str(tampered), seq, flipped)
            code, text = example.verify_with_cli(str(tampered))
            assert code == 1, (seq, text)
            assert "BROKEN" in text

    # --- negative controls --------------------------------------------------------------

    def test_negative_control_plain_gate_writes_no_entries(self, gate, log, log_path):
        responses, _ = gate.run_calls(THREE)
        assert responses["c2"].startswith("Recusal refused")
        assert log.entries == []
        assert not log_path.exists()

    def test_negative_control_ungated_crew_records_nothing_and_runs_the_delete(self, gate, log):
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
        assert "c2 run_sql  refused" in out
        assert f"1 {SURFACE} run_sql  deny" in out
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
