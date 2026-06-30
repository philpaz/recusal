"""Tests for the tamper-evident audit log — chain integrity and tamper detection."""

import json
from datetime import datetime, timezone

from recusal import compute_verdict
from recusal.audit import GENESIS, AuditLog, load, verify, verify_file

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _log(path=None):
    return AuditLog(path=path, clock=lambda: FIXED)


def _fail():
    return compute_verdict([{"severity": "CRITICAL", "status": "fail", "message": "boom"}])


def _pass():
    return compute_verdict([])


def test_first_entry_links_to_genesis():
    log = _log()
    e = log.append(_pass(), action={"tool": "Read"})
    assert e["prev_hash"] == GENESIS
    assert e["seq"] == 0
    assert e["hash"] == log.last_hash


def test_entries_chain():
    log = _log()
    a = log.append(_fail(), action={"tool": "Bash"})
    b = log.append(_pass())
    assert b["prev_hash"] == a["hash"]
    assert b["seq"] == 1


def test_verify_intact():
    log = _log()
    log.append(_fail())
    log.append(_pass())
    log.append(_fail())
    ok, problems = verify(log.entries)
    assert ok and problems == []


def test_verify_detects_content_tamper():
    log = _log()
    log.append(_fail())
    log.append(_pass())
    log.entries[0]["reasons"] = "nothing to see here"  # edit without re-hashing
    ok, problems = verify(log.entries)
    assert not ok
    assert any("tampered" in p for p in problems)


def test_verify_detects_deletion():
    log = _log()
    log.append(_fail())
    log.append(_pass())
    log.append(_fail())
    del log.entries[1]
    ok, problems = verify(log.entries)
    assert not ok
    assert any("broken link" in p for p in problems)


def test_verify_detects_reorder():
    log = _log()
    log.append(_fail())
    log.append(_pass())
    log.entries.reverse()
    ok, _ = verify(log.entries)
    assert not ok


def test_records_the_verdict():
    log = _log()
    e = log.append(_fail(), action={"tool": "Bash"}, actor="session-1")
    assert e["decision"] == "FAIL"
    assert e["actor"] == "session-1"
    assert e["failures"][0]["message"] == "boom"


def test_deterministic_hash():
    a = _log().append(_fail(), action={"tool": "X"}, timestamp="t")
    b = _log().append(_fail(), action={"tool": "X"}, timestamp="t")
    assert a["hash"] == b["hash"]


def test_file_roundtrip_and_resume(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    log = _log(path=p)
    log.append(_fail())
    log.append(_pass())
    loaded = load(p)
    assert len(loaded) == 2
    assert verify_file(p)[0]

    log2 = _log(path=p)  # resume from the existing file
    e = log2.append(_fail())
    assert e["seq"] == 2 and e["prev_hash"] == loaded[-1]["hash"]
    assert verify_file(p)[0]


def test_tampered_file_is_detected(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    log = _log(path=p)
    log.append(_fail())
    log.append(_pass())
    with open(p, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    rec = json.loads(lines[0])
    rec["reasons"] = "edited"
    lines[0] = json.dumps(rec)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    ok, problems = verify_file(p)
    assert not ok and problems
