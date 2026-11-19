"""Hardening tests for the audit log: the documented limits (truncation/rewrite need an
external anchor) and the robustness fixes (corrupt-resume, non-serializable action)."""

from datetime import datetime, timezone

import pytest

from recusal import compute_verdict
from recusal.audit import AuditFramingError, AuditLog, _digest, load, verify, verify_file

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


# --- the hash covers the STORED form of an entry ---------------------------------------
#
# A verifier reads the file, so the hash has to be taken over what the file holds. Before
# this was held, an entry whose action carried integer keys hashed in one key order and
# verified in another, and an untouched log reported "content tampered".

INT_KEYS = {"tool": "Bash", "codes": {1: "a", 2: "b", 10: "c"}}

#: What the v0.13.1 tag's own ``recusal/audit.py`` wrote for the plain entry below.
PLAIN_HASH_0_13_1 = "1ccab620df9f634bafea9f5124f2a106ae2975b0072a767231bbd23056bca3be"


def test_an_untouched_log_with_integer_keys_verifies_from_the_file(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    log = _log(p)
    log.append(_fail(), action=INT_KEYS)
    log.append(_fail(), action=INT_KEYS)
    assert verify(log.entries) == (True, [])
    assert verify_file(p) == (True, [])


def test_decide_with_integer_keys_in_control_verifies_from_the_file(tmp_path):
    from recusal.claude_code import decide

    p = str(tmp_path / "audit.jsonl")
    decision, _ = decide(
        "Bash",
        {"command": "ls"},
        lambda name, tool_input: [],
        audit=_log(p),
        surface="test.gate",
        control={"policy_id": "p", "tiers": {1: "a", 2: "b", 10: "c"}},
    )
    assert decision == "defer"
    assert verify_file(p) == (True, [])


def test_the_entry_returned_mirrored_and_sent_to_sinks_is_the_entry_on_disk(tmp_path):
    class Sink:
        def __init__(self):
            self.seen = []

        def write(self, entry):
            self.seen.append(entry)

    p = str(tmp_path / "audit.jsonl")
    sink = Sink()
    log = AuditLog(path=p, clock=lambda: FIXED, sinks=[sink])
    returned = log.append(_fail(), action={"codes": {2: "b", 10: "c"}, "pair": (1, 2), "s": {7}})
    on_disk = load(p)
    assert on_disk == [returned] == log.entries == sink.seen
    assert returned["action"] == {"codes": {"2": "b", "10": "c"}, "pair": [1, 2], "s": "{7}"}


def test_mixed_integer_and_string_keys_are_recorded_and_verify(tmp_path):
    # sort_keys cannot order 1 against "a"; hashing the raw dict raised and the append failed
    p = str(tmp_path / "audit.jsonl")
    _log(p).append(_fail(), action={1: "x", "a": "y"})
    assert verify_file(p) == (True, [])


def test_a_plain_json_entry_hashes_as_it_did_before_the_stored_form(tmp_path):
    # 0.13.1 wrote this hash for this entry; a fix that moved it would break every log
    log = _log()
    entry = log.append(_fail(), action={"tool": "Bash", "n": 1, "f": 0.1, "ok": True}, actor="a")
    assert entry["hash"] == PLAIN_HASH_0_13_1


def test_negative_control_an_edit_to_an_integer_keyed_entry_is_still_caught(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    log = _log(p)
    log.append(_fail(), action=INT_KEYS)
    log.append(_fail(), action=INT_KEYS)
    with open(p, encoding="utf-8") as fh:
        text = fh.read()
    assert text.count('"10": "c"') == 2
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text.replace('"10": "c"', '"10": "z"', 1))
    intact, problems = verify_file(p)
    assert intact is False
    assert problems == ["entry 0 (seq 0): content tampered - hash does not match"]


# --- the tolerant reader survives a line that is not UTF-8 (found by fuzzing) --------------
#
# ``_scan`` promised that a corrupt line is skipped "so a half-written tail can never brick
# the log", and ``_tail_state`` already treated a line that is not UTF-8 that way. ``_scan``
# opened the file in text mode, so one such byte raised ``UnicodeDecodeError`` out of
# ``load``, out of a resume, and out of every append after it.


def _two_entries(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=path)
    verdict = compute_verdict([{"severity": "CRITICAL", "status": "fail", "check": "c"}])
    log.append(verdict, action={"tool": "t"})
    log.append(verdict, action={"tool": "t"})
    return path, verdict


@pytest.mark.parametrize(
    "damage, readable",
    [
        (lambda good: good + b"\xf9\n", 2),
        (lambda good: good + b"\xff\xfe", 2),
        (lambda good: b"\x80\n" + good, 2),
        (lambda good: good.replace(b'"tool"', b'"to\xffl"', 1), 1),
    ],
    ids=["bad line appended", "bad unterminated tail", "bad first line", "bad byte in entry 1"],
)
def test_a_line_that_is_not_utf8_is_skipped_by_every_tolerant_reader(tmp_path, damage, readable):
    path, verdict = _two_entries(tmp_path)
    with open(path, "rb") as fh:
        good = fh.read()
    with open(path, "wb") as fh:
        fh.write(damage(good))

    assert len(load(path)) == readable
    assert len(AuditLog(path=path).entries) == readable
    # the strict reader does not share the tolerance: the file is reported, not blessed
    intact, problems = verify_file(path)
    assert intact is False and any("cannot read" in problem for problem in problems)


def test_an_append_after_a_line_that_is_not_utf8_chains_onto_the_last_good_entry(tmp_path):
    path, verdict = _two_entries(tmp_path)
    head = load(path)[-1]["hash"]
    with open(path, "ab") as fh:
        fh.write(b"\xf9\xf9\n")
    AuditLog(path=path).append(verdict, action={"tool": "t"})
    entries = load(path)
    assert len(entries) == 3 and entries[-1]["prev_hash"] == head and entries[-1]["seq"] == 2
    assert verify(entries) == (True, [])


def test_an_undecodable_unterminated_tail_still_refuses_the_append(tmp_path):
    """Skipping a bad line when reading is not licence to write after a torn one."""
    path, verdict = _two_entries(tmp_path)
    with open(path, "rb") as fh:
        before = fh.read()
    with open(path, "ab") as fh:
        fh.write(b"\xff\xfe")
    with pytest.raises(AuditFramingError):
        AuditLog(path=path).append(verdict, action={"tool": "t"})
    with open(path, "rb") as fh:
        assert fh.read() == before + b"\xff\xfe"


def test_reading_as_bytes_keeps_every_line_ending_the_text_reader_accepted(tmp_path):
    path, _ = _two_entries(tmp_path)
    with open(path, "rb") as fh:
        first, second = fh.read().splitlines()
    for joined in (
        first + b"\n" + second + b"\n",
        first + b"\r\n" + second + b"\r\n",
        first + b"\r" + second + b"\r",
        b"\n\n  " + first + b"  \n\t\n" + second,
    ):
        with open(path, "wb") as fh:
            fh.write(joined)
        assert [entry["seq"] for entry in load(path)] == [0, 1], joined[:20]


def test_the_cli_still_calls_a_file_that_is_not_utf8_an_operational_error(tmp_path, capsys):
    """``recusal audit verify`` exits 2 when it cannot read the log and 1 when the chain is
    broken. A file that is not UTF-8 stays the first kind, as it was before the tolerant
    reader learned to skip such a line."""
    from recusal.__main__ import main

    path, _ = _two_entries(tmp_path)
    assert main(["audit", "verify", path]) == 0
    capsys.readouterr()
    with open(path, "ab") as fh:
        fh.write(b"\xf9\n")
    assert main(["audit", "verify", path]) == 2
    assert "cannot read audit log" in capsys.readouterr().out
