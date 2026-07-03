"""
Adversarial suite: try to tamper with the audit record without getting caught.

`recusal.audit` is **tamper-evident, not tamper-proof** - and that distinction is the
whole point of these tests. They prove two things at once:

  1. What the chain DOES catch on its own: in-place edits and reordering of existing
     entries (the ``verify`` must go red).
  2. What it does NOT catch without an external anchor: tail truncation and a full
     recompute by someone with write access. Those tests assert the *documented* limit
     (unanchored ``verify`` still passes) AND that ``verify(expected_head=...)`` closes it.

If a change ever silently strengthened or weakened either property, a test here flips.
"""

import json

from recusal.audit import AuditLog, verify
from recusal.evidence import Finding, compute_verdict


def _log_with(n, path=None):
    log = AuditLog(path=path, clock=lambda: __import__("datetime").datetime(2026, 1, 1))
    for i in range(n):
        v = compute_verdict([Finding.fail(f"c{i}", severity="CRITICAL", message=f"m{i}")])
        log.append(v, action={"tool": "Bash", "i": i}, actor="agent-1")
    return log


# --- what the chain catches unaided ----------------------------------------------------


class TestTamperIsDetected:
    def test_in_place_edit_of_a_message_breaks_the_chain(self):
        log = _log_with(4)
        entries = [dict(e) for e in log.entries]
        entries[1]["reasons"] = "quietly changed the recorded reason"
        intact, problems = verify(entries)
        assert not intact
        assert any("tampered" in p for p in problems)

    def test_flipping_a_decision_is_caught(self):
        log = _log_with(3)
        entries = [dict(e) for e in log.entries]
        entries[0]["decision"] = "PASS"  # rewrite a refusal as a pass
        intact, _ = verify(entries)
        assert not intact

    def test_reordering_entries_breaks_the_chain(self):
        log = _log_with(4)
        entries = [dict(e) for e in log.entries]
        entries[1], entries[2] = entries[2], entries[1]
        intact, problems = verify(entries)
        assert not intact
        assert any("order" in p or "link" in p for p in problems)

    def test_deleting_a_middle_entry_is_caught(self):
        log = _log_with(5)
        entries = [dict(e) for e in log.entries]
        del entries[2]
        intact, _ = verify(entries)
        assert not intact

    def test_inserting_a_forged_entry_is_caught(self):
        log = _log_with(3)
        entries = [dict(e) for e in log.entries]
        forged = dict(entries[-1])
        forged["reasons"] = "forged"
        entries.insert(1, forged)
        assert not verify(entries)[0]


# --- the honest limits: unanchored verify cannot see these -----------------------------


class TestDocumentedLimitsHold:
    def test_tail_truncation_passes_unanchored_but_fails_with_anchor(self):
        log = _log_with(5)
        anchor = (len(log.entries), log.last_hash)  # committed off-box
        truncated = [dict(e) for e in log.entries[:3]]  # attacker drops the last two
        # Unaided, a truncated-but-consistent prefix still verifies - the documented gap.
        assert verify(truncated)[0] is True
        # With the external head anchor, truncation is caught.
        intact, problems = verify(truncated, expected_head=anchor)
        assert not intact
        assert any("truncation" in p or "length" in p for p in problems)

    def test_full_rehash_forgery_passes_unanchored_but_fails_with_anchor(self):
        log = _log_with(4)
        anchor = (len(log.entries), log.last_hash)
        # Attacker with write access rewrites every entry and recomputes the whole chain.
        forged = AuditLog(clock=lambda: __import__("datetime").datetime(2026, 1, 1))
        for i in range(4):
            forged.append(
                compute_verdict([Finding.ok(f"clean{i}")]),  # all passes, no refusals
                action={"tool": "Bash", "i": i},
            )
        forged_entries = forged.entries
        # A self-consistent forged chain passes an unanchored verify - the documented gap.
        assert verify(forged_entries)[0] is True
        # The anchor (real count is same, but head hash differs) catches the swap.
        intact, problems = verify(forged_entries, expected_head=anchor)
        assert not intact
        assert any("head" in p for p in problems)


# --- resilience: a half-written tail must not brick the log -----------------------------


class TestPartialWriteResilience:
    def test_corrupt_final_line_is_skipped_not_fatal(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        _log_with(3, path=path)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"seq": 3, "partial": ')  # process killed mid-append
        # Resuming the log must not crash on the corrupt tail.
        resumed = AuditLog(path=path, clock=lambda: __import__("datetime").datetime(2026, 1, 1))
        assert len(resumed.entries) == 3  # the good prefix survives
        assert verify(resumed.entries)[0] is True

    def test_blank_lines_are_ignored(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        _log_with(2, path=path)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n\n   \n")
        resumed = AuditLog(path=path)
        assert len(resumed.entries) == 2


# --- append integrity ------------------------------------------------------------------


class TestAppendIsWellFormed:
    def test_every_entry_hashes_itself_and_links_back(self):
        log = _log_with(6)
        assert verify(log.entries)[0] is True

    def test_persisted_and_in_memory_chains_match(self, tmp_path):
        path = str(tmp_path / "a.jsonl")
        log = _log_with(4, path=path)
        with open(path, encoding="utf-8") as fh:
            on_disk = [json.loads(line) for line in fh if line.strip()]
        assert on_disk == log.entries
