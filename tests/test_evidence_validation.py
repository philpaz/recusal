"""A malformed Finding can never fold as a pass.

Before 0.12.4 the fold compared the severity enum by identity and read ``passed`` by
truthiness, so a Finding built with a string severity, an unknown level, or
``passed="no"`` was simply not counted: a failed CRITICAL finding folded to PASS. These
tests pin the fix at every boundary: construction, the fold, the loose-dict path, and the
adapters that turn a verdict into a refusal (``decide`` and the Claude Code hook).
"""

import io
import json
import sys

import pytest

from recusal import Finding, Severity, compute_verdict
from recusal.claude_code import decide, run_pretooluse_hook
from recusal.evidence import certify_evidence, evaluate_policy

# --- construction -----------------------------------------------------------------------


def test_string_severity_is_normalized_to_the_enum():
    f = Finding("x", "CRITICAL", False)
    assert f.severity is Severity.CRITICAL
    assert Finding("x", "critical", False).severity is Severity.CRITICAL
    assert Finding("x", Severity.ERROR, True).severity is Severity.ERROR


@pytest.mark.parametrize("severity", ["NOT_A_LEVEL", "", "fatal", 3, None, ["CRITICAL"]])
def test_unknown_severity_is_refused_at_construction(severity):
    with pytest.raises(ValueError):
        Finding("x", severity, False)


@pytest.mark.parametrize(
    "passed", ["no", "false", "yes", "", object(), [True], {"ok": True}, None, 2, -1, 1.0]
)
def test_non_bool_passed_is_refused_at_construction(passed):
    with pytest.raises(TypeError, match="Finding.passed must be a bool"):
        Finding("x", Severity.CRITICAL, passed)


def test_int_zero_and_one_are_accepted_as_outcomes():
    assert Finding("x", Severity.CRITICAL, 0).passed is False
    assert Finding("x", Severity.CRITICAL, 1).passed is True


def test_ok_and_fail_constructors_still_work_with_strings():
    assert Finding.fail("x", severity="critical").severity is Severity.CRITICAL
    assert Finding.ok("x").passed is True


# --- the fold: a failed finding is counted whatever path built it -----------------------


def test_failed_finding_with_string_severity_refuses():
    verdict = compute_verdict([Finding("x", "CRITICAL", False)])
    assert verdict.refused
    assert verdict.failures[0].severity is Severity.CRITICAL


def test_failed_error_with_string_severity_retries():
    assert compute_verdict([Finding("x", "ERROR", False)]).retryable


def test_a_finding_altered_after_construction_is_caught_at_the_fold():
    f = Finding.fail("x", severity="CRITICAL")
    object.__setattr__(f, "severity", "BOGUS")
    with pytest.raises(ValueError):
        compute_verdict([f])
    g = Finding.fail("x", severity="CRITICAL")
    object.__setattr__(g, "passed", "no")
    with pytest.raises(TypeError):
        compute_verdict([g])


def test_a_finding_altered_to_a_string_severity_still_counts():
    f = Finding.fail("x", severity="CRITICAL")
    object.__setattr__(f, "severity", "CRITICAL")  # a str, equal to but not the enum
    assert compute_verdict([f]).refused


def test_an_equal_but_not_identical_enum_member_still_counts():
    class Shadow(str):
        """Equal to the enum value as a string, never identical to the member."""

    f = Finding.fail("x", severity="CRITICAL")
    object.__setattr__(f, "severity", Shadow("CRITICAL"))
    assert compute_verdict([f]).refused


@pytest.mark.parametrize("strict", [False, True])
def test_every_entry_point_refuses_the_same_malformed_finding(strict):
    f = Finding.fail("x", severity="CRITICAL")
    object.__setattr__(f, "severity", "BOGUS")
    with pytest.raises(ValueError):
        compute_verdict([f], strict=strict)
    with pytest.raises(ValueError):
        evaluate_policy([f], strict=strict)
    with pytest.raises(ValueError):
        certify_evidence([f], strict=strict)


# --- loose dicts: a truthy object is not an outcome -------------------------------------


@pytest.mark.parametrize("passed", [object(), [True], {"ok": True}, (1,), 2, 1.5, b"yes"])
def test_dict_with_an_object_as_passed_is_refused_in_strict_mode(passed):
    with pytest.raises(TypeError, match="'passed' must be a bool"):
        compute_verdict([{"check": "x", "severity": "CRITICAL", "passed": passed}], strict=True)


@pytest.mark.parametrize("passed", [object(), [True], {"ok": True}, (1,), 2, 1.5, b"yes"])
def test_dict_with_an_object_as_passed_reads_as_a_failure_in_lenient_mode(passed):
    verdict = compute_verdict([{"check": "x", "severity": "CRITICAL", "passed": passed}])
    assert verdict.refused


def test_dict_none_passed_is_a_failure_not_a_pass():
    assert compute_verdict([{"severity": "CRITICAL", "passed": None}]).refused
    with pytest.raises(TypeError):
        compute_verdict([{"severity": "CRITICAL", "passed": None}], strict=True)


def test_dict_bool_and_token_passed_are_unchanged():
    assert compute_verdict([{"severity": "CRITICAL", "passed": True}]).passed
    assert compute_verdict([{"severity": "CRITICAL", "passed": 1}]).passed
    assert compute_verdict([{"severity": "CRITICAL", "passed": "yes"}]).passed
    assert compute_verdict([{"severity": "CRITICAL", "passed": "false"}]).refused
    assert compute_verdict([{"severity": "CRITICAL", "passed": 0}]).refused


def test_dict_unknown_severity_is_refused_in_both_modes():
    for strict in (False, True):
        with pytest.raises(ValueError):
            compute_verdict([{"severity": "BOGUS", "passed": False}], strict=strict)


# --- through the adapters: the gate refuses, it does not pass -----------------------------


def _string_severity_policy(tool_name, tool_input):
    return [Finding("sql_scope", "CRITICAL", False, "destructive SQL")]


def _bogus_severity_policy(tool_name, tool_input):
    f = Finding.fail("sql_scope", severity="CRITICAL", message="destructive SQL")
    object.__setattr__(f, "severity", "BOGUS")
    return [f]


def _string_passed_policy(tool_name, tool_input):
    f = Finding.fail("sql_scope", severity="CRITICAL", message="destructive SQL")
    object.__setattr__(f, "passed", "no")
    return [f]


def _object_passed_dict_policy(tool_name, tool_input):
    return [{"check": "sql_scope", "severity": "CRITICAL", "passed": object()}]


POLICIES = [
    _string_severity_policy,
    _bogus_severity_policy,
    _string_passed_policy,
    _object_passed_dict_policy,
]


@pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.__name__)
def test_decide_denies_on_every_malformed_finding(policy):
    decision, reason = decide("run_sql", {"sql": "DELETE FROM t"}, policy)
    assert decision == "deny", reason
    assert "Recusal" in reason


def test_decide_denies_with_the_real_reason_for_a_string_severity():
    decision, reason = decide("run_sql", {"sql": "DELETE FROM t"}, _string_severity_policy)
    assert decision == "deny"
    assert "destructive SQL" in reason


def test_decide_fail_open_still_cannot_turn_a_malformed_failure_into_allow():
    # fail_closed=False ignores a *policy error*, it never promotes a malformed failure
    decision, _ = decide(
        "run_sql", {"sql": "DELETE FROM t"}, _string_severity_policy, fail_closed=False
    )
    assert decision == "deny"


@pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.__name__)
def test_claude_code_hook_blocks_on_every_malformed_finding(policy):
    event = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "rm -rf /"},
        "session_id": "s",
        "tool_use_id": "t1",
    }
    out = io.StringIO()
    run_pretooluse_hook(policy, stdin=io.StringIO(json.dumps(event)), stdout=out)
    payload = json.loads(out.getvalue())
    assert payload["hookSpecificOutput"]["permissionDecision"] == "deny", payload


def test_negative_control_a_well_formed_pass_still_passes_through_decide():
    def clean(tool_name, tool_input):
        return [Finding.ok("sql_scope", severity="CRITICAL")]

    assert decide("run_sql", {"sql": "SELECT 1"}, clean)[0] == "defer"


def test_the_old_behaviour_is_the_thing_this_file_refuses():
    # the pre-0.12.4 fold, reproduced: identity comparison plus truthiness
    def old_fold(findings):
        critical = [f for f in findings if not f.passed and f.severity is Severity.CRITICAL]
        return "FAIL" if critical else "PASS"

    f = Finding.fail("x", severity="CRITICAL")
    object.__setattr__(f, "severity", "CRITICAL")
    assert old_fold([f]) == "PASS"  # the bug
    assert compute_verdict([f]).refused  # the fix


if sys.version_info >= (3, 10):

    def test_dataclass_replace_runs_validation():
        import dataclasses

        f = Finding.ok("x")
        with pytest.raises(TypeError):
            dataclasses.replace(f, passed="yes")
