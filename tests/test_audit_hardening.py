"""Hardening tests for the audit log: the documented limits (truncation/rewrite need an
external anchor) and the robustness fixes (corrupt-resume, non-serializable action)."""

from datetime import datetime, timezone

import pytest

from recusal import compute_verdict
from recusal.audit import AuditFramingError, AuditLog, _digest, load, verify

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _log(path=None):
    return AuditLog(path=path, clock=lambda: FIXED)


def _fail():
    return compute_verdict([{"severity": "CRITICAL", "status": "fail", "message": "boom"}])


def test_trailing_truncation_needs_an_anchor_to_detect():
    log = _log()
    log.append(_fail())
    log.append(_fail())
    log.append(_fail())
    head = (len(log.entries), log.last_hash)
    truncated = log.entries[:-1]  # attacker deletes the last entry

    assert verify(truncated)[0] is True  # plain verify cannot see truncation (documented)
    ok, problems = verify(truncated, expected_head=head)
    assert not ok
    assert any("truncation" in p or "length mismatch" in p for p in problems)


def test_editing_the_last_entry_needs_an_anchor_to_detect():
    # The tightest case of the documented tail-suffix limit: an attacker edits ONLY the
    # newest entry (the most recently recorded action) and recomputes its single hash.
    # There is no successor whose prev_hash would break, so plain verify passes; the docs
    # must not claim "any in-place edit is caught". The head anchor closes it.
    log = _log()
    log.append(_fail())
    log.append(_fail())
    good_head = (len(log.entries), log.last_hash)

    entries = [dict(e) for e in log.entries]
    last = entries[-1]
    last["verdict"] = {"decision": "PASS", "message": "quietly flipped"}  # falsify the record
    last["hash"] = _digest({k: v for k, v in last.items() if k != "hash"})  # recompute its own hash

    assert verify(entries)[0] is True  # unanchored verify cannot see a last-entry rewrite
    ok, problems = verify(entries, expected_head=good_head)
    assert not ok
    assert any("head mismatch" in p for p in problems)


def test_full_rewrite_needs_an_anchor_to_detect():
    log = _log()
    log.append(_fail())
    log.append(_fail())
    good_head = (len(log.entries), log.last_hash)

    forged = _log()  # attacker rewrites entry 0 and recomputes the whole chain
    forged.append(compute_verdict([{"severity": "INFO", "status": "pass", "message": "all fine"}]))
    forged.append(_fail())

    assert verify(forged.entries)[0] is True  # internally consistent, passes plain verify
    ok, problems = verify(forged.entries, expected_head=good_head)
    assert not ok
    assert any("head mismatch" in p for p in problems)


def test_incomplete_trailing_record_resumes_but_refuses_the_append_until_removed(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    log = _log(path=p)
    log.append(_fail())
    log.append(_fail())
    with open(p, "a", encoding="utf-8") as fh:  # process killed mid-append
        fh.write('{"seq": 2, "partial"')
    with open(p, "rb") as fh:
        damaged = fh.read()

    assert len(load(p)) == 2  # corrupt line skipped, not raised
    log2 = _log(path=p)  # resume succeeds rather than crashing
    # ...but the append refuses: written at EOF, it would fuse with the incomplete
    # record into one unreadable line. The damaged bytes are preserved for the operator.
    with pytest.raises(AuditFramingError, match="incomplete record") as info:
        log2.append(_fail())
    with open(p, "rb") as fh:
        assert fh.read() == damaged
    assert info.value.offset == len(damaged) - len('{"seq": 2, "partial"')
    # The documented recovery: truncate the incomplete bytes, then the chain continues.
    with open(p, "r+b") as fh:
        fh.truncate(info.value.offset)
    assert log2.append(_fail())["seq"] == 2
    assert verify(load(p))[0]


def test_non_serializable_action_is_recorded_not_dropped():
    log = _log()
    # a set is not JSON-serializable; default=str must keep the verdict on the record
    e = log.append(_fail(), action={"weird": {1, 2, 3}})
    assert e["decision"] == "FAIL"
    assert verify(log.entries)[0]
