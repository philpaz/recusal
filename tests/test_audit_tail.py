"""``AuditLog(resume="tail")``: resume the chain head without holding the log in memory.

The contract: tail resume continues the exact same hash chain (seq and prev_hash) as a
full resume would, retains nothing in memory before or after, and the file stays
verifiable end to end. Memory-bounded is a property of the mode, not a different log.
"""

import pytest

from recusal import AuditLog, compute_verdict, verify
from recusal.audit import load, verify_file


def _verdict(msg="ok"):
    return compute_verdict([{"check": "c", "severity": "INFO", "status": "pass", "message": msg}])


def test_tail_resume_continues_the_same_chain(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    full = AuditLog(path=path)
    for i in range(5):
        full.append(_verdict(f"entry {i}"), action={"i": i})

    tail = AuditLog(path=path, resume="tail")
    assert tail.entries == []  # nothing retained from the resume
    entry = tail.append(_verdict("after tail resume"), action={"i": 5})
    assert entry["seq"] == 5
    assert entry["prev_hash"] == full.last_hash
    assert tail.entries == []  # nothing retained after the append either

    ok, problems = verify_file(path)
    assert ok, problems
    assert len(load(path)) == 6


def test_tail_resume_matches_full_resume_head(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    writer = AuditLog(path=path)
    for i in range(3):
        writer.append(_verdict(str(i)))
    assert AuditLog(path=path, resume="tail").last_hash == AuditLog(path=path).last_hash


def test_tail_resume_skips_a_corrupt_trailing_line(tmp_path):
    # A partial final line (a process killed mid-append) must not brick the resume; the
    # chain continues from the last intact entry, same as a full resume. Reading is
    # tolerant; VERIFYING is not: the strict verifier still flags the unreadable line.
    path = tmp_path / "audit.jsonl"
    full = AuditLog(path=str(path))
    full.append(_verdict("intact"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"seq": 1, "truncat\n')  # not JSON
    tail = AuditLog(path=str(path), resume="tail")
    tail.append(_verdict("continues"))
    ok, problems = verify(load(str(path)))  # the parsed chain itself is unbroken
    assert ok, problems
    ok, problems = verify_file(str(path))  # ...but a strict verify names the bad line
    assert not ok
    assert any("not valid JSON" in p for p in problems)


def test_tail_resume_requires_a_path():
    with pytest.raises(ValueError, match="requires a path"):
        AuditLog(resume="tail")


def test_an_unknown_resume_mode_is_rejected():
    with pytest.raises(ValueError, match="resume"):
        AuditLog(path="x.jsonl", resume="bogus")


def test_full_resume_is_unchanged(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    a = AuditLog(path=path)
    a.append(_verdict("one"))
    b = AuditLog(path=path)  # default: full
    b.append(_verdict("two"))
    assert [e["seq"] for e in b.entries] == [0, 1]
    ok, problems = verify(b.entries)
    assert ok, problems
