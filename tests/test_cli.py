"""The CI surface: ``verdict`` / ``audit verify`` / ``doctor`` must map decisions to
blocking exit codes (PASS 0 / RETRY 1 / FAIL 2) and fail CLOSED on every operational
error - an unreadable file, invalid JSON, a missing log, an empty evidence set. A CI
gate that errors into a green job is the silent-pass failure mode this library exists
to prevent, one seam further out.

Also drift-locks ``action.yml``: the composite action must route every adjudication
through ``python -m recusal``, pass inputs via env (never interpolated into the shell
body), and refuse to pass vacuously when given nothing to adjudicate.
"""

import io
import json
import os
import re

from recusal import AuditLog, Finding, compute_verdict
from recusal.__main__ import (
    audit_verify_command,
    doctor_command,
    init,
    main,
    verdict_command,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_findings(tmp_path, findings):
    path = tmp_path / "findings.json"
    path.write_text(json.dumps(findings), encoding="utf-8")
    return str(path)


def _verdict(path, **kwargs):
    buf = io.StringIO()
    rc = verdict_command(path, stdout=buf, **kwargs)
    return rc, buf.getvalue()


def _audit_verify(path, **kwargs):
    buf = io.StringIO()
    rc = audit_verify_command(path, stdout=buf, **kwargs)
    return rc, buf.getvalue()


def _doctor(project_dir, **kwargs):
    buf = io.StringIO()
    rc = doctor_command(project_dir, stdout=buf, **kwargs)
    return rc, buf.getvalue()


# --- verdict: decisions map to blocking exit codes ----------------------------------------


def test_verdict_pass_exits_0(tmp_path):
    path = _write_findings(tmp_path, [{"check": "suite", "severity": "CRITICAL", "status": "pass"}])
    rc, out = _verdict(path)
    assert rc == 0
    assert out.startswith("PASS")


def test_verdict_retry_exits_1(tmp_path):
    path = _write_findings(
        tmp_path, [{"check": "lint", "severity": "ERROR", "status": "fail", "message": "2 errors"}]
    )
    rc, out = _verdict(path)
    assert rc == 1
    assert out.startswith("RETRY")
    assert "2 errors" in out


def test_verdict_fail_exits_2(tmp_path):
    path = _write_findings(
        tmp_path, [{"check": "schema", "severity": "CRITICAL", "status": "fail"}]
    )
    rc, out = _verdict(path)
    assert rc == 2
    assert out.startswith("FAIL")


def test_verdict_accepts_object_with_findings_key(tmp_path):
    path = _write_findings(
        tmp_path, {"findings": [{"check": "s", "severity": "CRITICAL", "status": "pass"}]}
    )
    rc, _ = _verdict(path)
    assert rc == 0


def test_verdict_reads_stdin_dash():
    stdin = io.StringIO(json.dumps([{"check": "s", "severity": "CRITICAL", "status": "pass"}]))
    buf = io.StringIO()
    assert verdict_command("-", stdout=buf, stdin=stdin) == 0


def test_verdict_json_shape_is_stable(tmp_path):
    path = _write_findings(
        tmp_path, [{"check": "lint", "severity": "ERROR", "status": "fail", "message": "boom"}]
    )
    rc, out = _verdict(path, as_json=True)
    payload = json.loads(out)
    assert rc == payload["exit_code"] == 1
    assert payload["decision"] == "RETRY"
    assert payload["highest_severity"] == "ERROR"
    assert payload["failures"][0]["check"] == "lint"
    assert "boom" in payload["reasons"]


# --- verdict: every operational error fails closed -----------------------------------------


def test_verdict_missing_file_fails_closed(tmp_path):
    rc, out = _verdict(str(tmp_path / "absent.json"))
    assert rc == 2
    assert "failed closed" in out


def test_verdict_invalid_json_fails_closed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    rc, out = _verdict(str(path))
    assert rc == 2
    assert "failed closed" in out


def test_verdict_non_array_fails_closed(tmp_path):
    path = tmp_path / "scalar.json"
    path.write_text('"just a string"', encoding="utf-8")
    rc, _ = _verdict(str(path))
    assert rc == 2


def test_verdict_empty_evidence_fails_closed(tmp_path):
    """An empty findings array certifies nothing - same rule as GateAdjudicator."""
    rc, out = _verdict(_write_findings(tmp_path, []))
    assert rc == 2
    assert "certifies nothing" in out


def test_verdict_strict_by_default_rejects_ambiguous_evidence(tmp_path):
    """A finding with no status/passed must not read as a silent pass in CI."""
    path = _write_findings(tmp_path, [{"check": "x", "severity": "CRITICAL"}])
    rc, out = _verdict(path)
    assert rc == 2
    assert "failed closed" in out


def test_verdict_lenient_opts_out_of_strict(tmp_path):
    path = _write_findings(tmp_path, [{"check": "x", "severity": "CRITICAL"}])
    rc, _ = _verdict(path, lenient=True)
    assert rc == 0


def test_verdict_failed_closed_json_carries_the_flag(tmp_path):
    rc, out = _verdict(str(tmp_path / "absent.json"), as_json=True)
    payload = json.loads(out)
    assert rc == payload["exit_code"] == 2
    assert payload["decision"] == "FAIL"
    assert payload["failed_closed"] is True


# --- audit verify ---------------------------------------------------------------------------


def _make_log(tmp_path, n=3):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path)
    for i in range(n):
        verdict = compute_verdict([Finding.ok("probe", severity="CRITICAL")])
        log.append(verdict, action={"i": i})
    return path, log


def test_audit_verify_intact_exits_0(tmp_path):
    path, log = _make_log(tmp_path)
    rc, out = _audit_verify(path)
    assert rc == 0
    assert "intact" in out
    assert log.last_hash in out


def test_audit_verify_tamper_exits_1(tmp_path):
    path, _ = _make_log(tmp_path)
    text = open(path, encoding="utf-8").read()
    open(path, "w", encoding="utf-8").write(text.replace('"i": 1', '"i": 9'))
    rc, out = _audit_verify(path)
    assert rc == 1
    assert "tampered" in out


def test_audit_verify_missing_log_fails_closed(tmp_path):
    rc, out = _audit_verify(str(tmp_path / "absent.jsonl"))
    assert rc == 2
    assert "missing log is not an intact log" in out


def test_audit_verify_unparseable_line_is_a_break_not_a_skip(tmp_path):
    """recusal.audit.load skips a corrupt line so a reader survives; the VERIFIER must
    surface it, or a log whose recent entries are garbage would verify clean."""
    path, _ = _make_log(tmp_path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("{this line is not json\n")
    rc, out = _audit_verify(path)
    assert rc == 1
    assert "not valid JSON" in out


def test_audit_verify_expect_head_good_and_bad(tmp_path):
    path, log = _make_log(tmp_path)
    rc, _ = _audit_verify(path, expect_head=f"3:{log.last_hash}")
    assert rc == 0
    rc, out = _audit_verify(path, expect_head=f"2:{log.last_hash}")
    assert rc == 1
    assert "truncation" in out or "length mismatch" in out


def test_audit_verify_malformed_expect_head_fails_closed(tmp_path):
    path, _ = _make_log(tmp_path)
    for bogus in ("nonsense", "3", ":abc", "x:abc"):
        rc, _ = _audit_verify(path, expect_head=bogus)
        assert rc == 2, bogus


def test_audit_verify_json_shape(tmp_path):
    path, log = _make_log(tmp_path)
    rc, out = _audit_verify(path, as_json=True)
    payload = json.loads(out)
    assert rc == payload["exit_code"] == 0
    assert payload["intact"] is True
    assert payload["entries"] == 3
    assert payload["head"] == log.last_hash
    assert payload["problems"] == []


def test_audit_verify_empty_existing_file_is_intact_with_zero_entries(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    rc, out = _audit_verify(str(path), as_json=True)
    assert rc == 0
    assert json.loads(out)["entries"] == 0


# --- doctor ---------------------------------------------------------------------------------


def test_doctor_passes_on_a_scaffolded_project(tmp_path):
    assert init(str(tmp_path), stdout=io.StringIO()) == 0
    rc, out = _doctor(str(tmp_path))
    assert rc == 0
    assert "PASS" in out


def test_doctor_fails_on_an_empty_project(tmp_path):
    rc, out = _doctor(str(tmp_path))
    assert rc == 2
    assert "FAIL" in out
    assert "recusal init" in out


def test_doctor_fails_when_hook_is_not_registered(tmp_path):
    init(str(tmp_path), stdout=io.StringIO())
    settings = tmp_path / ".claude" / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    rc, out = _doctor(str(tmp_path))
    assert rc == 2
    assert "hook_registered" in out


def test_doctor_fails_when_settings_do_not_parse(tmp_path):
    init(str(tmp_path), stdout=io.StringIO())
    (tmp_path / ".claude" / "settings.json").write_text("{broken", encoding="utf-8")
    rc, _ = _doctor(str(tmp_path))
    assert rc == 2


def test_doctor_degraded_when_gate_does_not_compile(tmp_path):
    init(str(tmp_path), stdout=io.StringIO())
    gate = tmp_path / ".claude" / "hooks" / "recusal_gate.py"
    gate.write_text("def broken(:\n", encoding="utf-8")
    rc, out = _doctor(str(tmp_path))
    assert rc == 1  # ERROR -> RETRY: refusing every call, recoverable by fixing the script
    assert "does not compile" in out


def test_doctor_warns_on_a_launcher_that_may_fail_open(tmp_path):
    init(str(tmp_path), stdout=io.StringIO())
    settings_path = tmp_path / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    hook = settings["hooks"]["PreToolUse"][0]["hooks"][0]
    hook["command"] = "python3 .claude/hooks/recusal_gate.py"  # bare launcher, no exit 2
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    rc, out = _doctor(str(tmp_path))
    assert rc == 0  # a warning does not block, but it must be said out loud
    assert "fail OPEN" in out


def test_doctor_json_carries_every_check(tmp_path):
    init(str(tmp_path), stdout=io.StringIO())
    rc, out = _doctor(str(tmp_path), as_json=True)
    payload = json.loads(out)
    assert rc == payload["exit_code"] == 0
    checks = {c["check"] for c in payload["checks"]}
    assert {"gate_script", "hook_registered", "launcher_fails_closed"} <= checks


# --- main() dispatch ------------------------------------------------------------------------


def test_main_dispatches_verdict(tmp_path, capsys):
    path = _write_findings(tmp_path, [{"check": "s", "severity": "CRITICAL", "status": "pass"}])
    assert main(["verdict", path]) == 0
    assert "PASS" in capsys.readouterr().out


def test_main_dispatches_audit_verify(tmp_path, capsys):
    path, _ = _make_log(tmp_path)
    assert main(["audit", "verify", path]) == 0
    assert "intact" in capsys.readouterr().out


def test_main_dispatches_doctor(tmp_path, capsys):
    init(str(tmp_path), stdout=io.StringIO())
    assert main(["doctor", "--dir", str(tmp_path)]) == 0


def test_main_bare_audit_prints_help_and_exits_2(capsys):
    assert main(["audit"]) == 2


# --- action.yml drift locks -----------------------------------------------------------------


def _action_text():
    with open(os.path.join(REPO_ROOT, "action.yml"), encoding="utf-8") as fh:
        return fh.read()


def test_action_is_composite_and_routes_through_the_cli():
    text = _action_text()
    assert 'using: "composite"' in text
    assert "python -m recusal doctor" in text
    assert "python -m recusal audit verify" in text
    assert "python -m recusal verdict" in text


def test_action_refuses_to_pass_vacuously():
    """An action given nothing to adjudicate must exit nonzero, not read as green."""
    assert "refusing to pass vacuously" in _action_text()


def test_action_inputs_flow_through_env_never_into_the_shell_body():
    """Every ${{ ... }} interpolation must be an env assignment (KEY: ${{ inputs...}}),
    so a crafted input value cannot inject shell into a run block."""
    for line in _action_text().splitlines():
        if "${{" in line:
            assert re.match(r"^\s+[A-Z_]+: \$\{\{ inputs(\.|\[)", line), line


def test_ci_dogfoods_the_action():
    """The repo's own CI must run the action from this checkout (uses: ./), including
    the negative case: a tampered audit log has to make the gate refuse."""
    with open(os.path.join(REPO_ROOT, ".github", "workflows", "ci.yml"), encoding="utf-8") as fh:
        ci = fh.read()
    assert "uses: ./" in ci
    assert "tampered" in ci
