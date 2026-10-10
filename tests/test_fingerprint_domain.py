"""The audit fingerprint binds to exactly one input: the strict JSON domain, nothing folded.

Found by an outside review: the fingerprint serialized with ``default=str`` and allowed
non-string keys and non-finite numbers, so ``{1: "x"}`` and ``{"1": "x"}`` shared a
fingerprint, an object shared one with its ``str()``, and NaN was recorded as if it were
JSON. The fingerprint now uses the one strict serializer the authorization module
already uses. The contract pinned here:

- every valid JSON input hashes exactly as before, so existing logs keep their meaning;
- an input outside the JSON domain is never fingerprinted: a recorded call is denied
  (fail closed) or, under ``fail_closed=False``, keeps its decision with no fingerprint;
- without an audit log nothing is hashed, so no unaudited decision changes.
"""

import hashlib
import io
import json
import math
import random

import pytest

from recusal import AuditLog, Finding
from recusal.audit import load, verify_file
from recusal.claude_code import _input_fingerprint, decide, run_pretooluse_hook

SURFACE = "my_runtime.tool_gate"


def _legacy(tool_input):
    """The pre-fix formula, kept here only to prove valid inputs hash identically."""
    canonical = json.dumps(
        tool_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pass(tool_name, tool_input):
    return []


def _deny_delete(tool_name, tool_input):
    if "DELETE" in str(tool_input.get("sql", "")):
        return [Finding.fail("delete", severity="CRITICAL", message="no deletes")]
    return []


class _Obj:
    def __str__(self):
        return "abc"


OUTSIDE_THE_DOMAIN = [
    pytest.param({1: "x"}, id="int-key"),
    pytest.param({(1,): "x"}, id="tuple-key"),
    pytest.param({"v": math.nan}, id="nan"),
    pytest.param({"v": math.inf}, id="inf"),
    pytest.param({"v": -math.inf}, id="-inf"),
    pytest.param({"v": _Obj()}, id="object-via-str"),
    pytest.param({"v": {1, 2}}, id="set"),
    pytest.param({"v": b"x"}, id="bytes"),
    pytest.param({"a": [{"b": math.nan}]}, id="nested-nan"),
    pytest.param({"a": {"b": {2: "y"}}}, id="nested-int-key"),
]


@pytest.mark.parametrize("bad", OUTSIDE_THE_DOMAIN)
def test_an_input_outside_the_json_domain_is_never_fingerprinted(bad):
    with pytest.raises(ValueError):
        _input_fingerprint(bad)


def test_the_look_alikes_no_longer_share_a_fingerprint():
    assert _legacy({1: "x"}) == _legacy({"1": "x"})  # the defect, as it was
    assert _legacy({"v": _Obj()}) == _legacy({"v": "abc"})
    with pytest.raises(ValueError):
        _input_fingerprint({1: "x"})
    with pytest.raises(ValueError):
        _input_fingerprint({"v": _Obj()})


def _random_json(rnd, depth=0):
    atoms = [None, True, False, 0, -1, 2**63, 1.5, -0.0, 1e300, "", "é", "日本", 'a"b\\c', "\n"]
    r = rnd.random()
    if depth > 4 or r < 0.4:
        return rnd.choice(atoms)
    if r < 0.7:
        return [_random_json(rnd, depth + 1) for _ in range(rnd.randint(0, 4))]
    return {f"k{rnd.randint(0, 9)}": _random_json(rnd, depth + 1) for _ in range(rnd.randint(0, 4))}


def test_every_valid_json_input_hashes_exactly_as_before():
    rnd = random.Random(20261010)
    for _ in range(5000):
        value = {"v": _random_json(rnd)}
        assert _input_fingerprint(value) == _legacy(value)
        parsed = json.loads(json.dumps(value))  # what the hook sees after json.load
        assert _input_fingerprint(parsed) == _legacy(parsed)


def test_moderately_deep_valid_input_hashes_as_before():
    value = {"v": 1}
    for _ in range(100):
        value = {"n": [value]}
    assert _input_fingerprint(value) == _legacy(value)


# --- decide(audit=) -------------------------------------------------------------------


@pytest.mark.parametrize("bad", OUTSIDE_THE_DOMAIN)
def test_audited_decide_denies_and_records_an_unbindable_input(tmp_path, bad):
    path = tmp_path / "a.jsonl"
    decision, reason = decide("t", bad, _pass, audit=AuditLog(str(path)), surface=SURFACE)
    assert decision == "deny"
    assert "cannot be fingerprinted" in reason
    (entry,) = load(str(path))
    assert entry["action"]["decision"] == "deny"
    assert "input_sha256" not in entry["action"]
    assert verify_file(str(path))[0]


@pytest.mark.parametrize("bad", OUTSIDE_THE_DOMAIN)
def test_fail_open_keeps_the_decision_and_records_no_fingerprint(tmp_path, bad):
    path = tmp_path / "a.jsonl"
    log = AuditLog(str(path))
    decision, _ = decide("t", bad, _pass, audit=log, surface=SURFACE, fail_closed=False)
    assert decision == "defer"
    (entry,) = load(str(path))
    assert "input_sha256" not in entry["action"]


@pytest.mark.parametrize("bad", OUTSIDE_THE_DOMAIN)
def test_unaudited_decide_is_unchanged_for_any_input(bad):
    assert decide("t", bad, _pass)[0] == "defer"
    assert decide("t", bad, _pass, fail_closed=False)[0] == "defer"


def test_a_refused_call_stays_refused_whatever_its_input(tmp_path):
    bad = {"sql": "DELETE FROM t", "x": math.nan}
    log = AuditLog(str(tmp_path / "a.jsonl"))
    assert decide("run_sql", bad, _deny_delete, audit=log, surface=SURFACE)[0] == "deny"
    log2 = AuditLog(str(tmp_path / "b.jsonl"))
    assert (
        decide("run_sql", bad, _deny_delete, audit=log2, surface=SURFACE, fail_closed=False)[0]
        == "deny"
    )


def test_a_valid_audited_input_records_the_same_fingerprint_as_before(tmp_path):
    path = tmp_path / "a.jsonl"
    value = {"sql": "SELECT 1", "n": [1, 2.5, None, {"é": True}]}
    decide("run_sql", value, _pass, audit=AuditLog(str(path)), surface=SURFACE)
    (entry,) = load(str(path))
    assert entry["action"]["input_sha256"] == _legacy(value)


# --- the hook ---------------------------------------------------------------------------

NAN_EVENT = '{"hook_event_name":"PreToolUse","tool_name":"t","tool_input":{"v":NaN}}'


def test_the_unaudited_hook_never_hashes_so_its_decision_is_unchanged():
    out = io.StringIO()
    assert run_pretooluse_hook(_pass, stdin=io.StringIO(NAN_EVENT), stdout=out) is None
    assert out.getvalue() == ""


def test_the_audited_hook_denies_and_records_an_unbindable_input(tmp_path):
    path = tmp_path / "a.jsonl"
    out = io.StringIO()
    result = run_pretooluse_hook(
        _pass, audit=AuditLog(str(path)), stdin=io.StringIO(NAN_EVENT), stdout=out
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "cannot be fingerprinted" in result["hookSpecificOutput"]["permissionDecisionReason"]
    (entry,) = load(str(path))
    assert entry["action"]["surface"] == "claude_code.pretooluse"
    assert entry["action"]["decision"] == "deny"
    assert "input_sha256" not in entry["action"]


def test_the_audited_fail_open_hook_defers_with_no_fingerprint(tmp_path):
    path = tmp_path / "a.jsonl"
    result = run_pretooluse_hook(
        _pass,
        audit=AuditLog(str(path)),
        fail_closed=False,
        stdin=io.StringIO(NAN_EVENT),
        stdout=io.StringIO(),
    )
    assert result is None
    (entry,) = load(str(path))
    assert entry["action"]["decision"] == "defer"
    assert "input_sha256" not in entry["action"]


def test_the_audited_hook_records_valid_input_exactly_as_before(tmp_path):
    path = tmp_path / "a.jsonl"
    tool_input = {"command": "ls -la", "description": "list"}
    event = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": tool_input}
    run_pretooluse_hook(
        _pass, audit=AuditLog(str(path)), stdin=io.StringIO(json.dumps(event)), stdout=io.StringIO()
    )
    (entry,) = load(str(path))
    assert entry["action"]["input_sha256"] == _legacy(tool_input)


def test_a_malformed_event_still_records_no_fingerprint(tmp_path):
    path = tmp_path / "a.jsonl"
    result = run_pretooluse_hook(
        _pass, audit=AuditLog(str(path)), stdin=io.StringIO("not json"), stdout=io.StringIO()
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    (entry,) = load(str(path))
    assert "input_sha256" not in entry["action"]
