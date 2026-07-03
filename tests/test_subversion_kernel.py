"""
Adversarial suite: try to talk the *verdict kernel* into passing work it should refuse.

The kernel (``compute_verdict`` over ``Finding``s) is the one decision function the whole
library funnels through. If it can be coerced into a silent PASS, every adapter above it
inherits the hole. These tests attack the coercion surface, the severity axis, and the
determinism guarantee. The through-line: a *failure must never read as a pass*, and the
only sanctioned way to pass is the absence of any failing finding.

Grouped by attacker objective. Every test asserts the *current* behavior, so a regression
that reopens one of these holes turns the suite red.
"""

import copy

import pytest

from recusal.evidence import (
    Decision,
    Finding,
    compute_verdict,
)

# --- objective: make a failing finding read as a pass -----------------------------------


class TestFailureMustNotReadAsPass:
    def test_stringified_false_passed_is_a_failure_not_a_silent_pass(self):
        # bool("false") is True in raw Python; a naive coerce would PASS this. The kernel
        # reads the intent instead, so a loosely serialized failure cannot slip through.
        for token in ("false", "False", "FALSE", "no", "n", "0", "off", "fail", "error"):
            v = compute_verdict([{"severity": "CRITICAL", "passed": token}], strict=True)
            assert v.decision is Decision.FAIL, token

    def test_genuine_true_still_passes(self):
        for truthy in (True, 1, "true", "yes", "ok", "passed"):
            v = compute_verdict([{"severity": "CRITICAL", "passed": truthy}])
            assert v.decision is Decision.PASS, truthy

    def test_empty_string_passed_is_a_failure(self):
        # An empty/whitespace value is not an affirmative pass.
        assert compute_verdict([{"severity": "CRITICAL", "passed": ""}]).refused
        assert compute_verdict([{"severity": "CRITICAL", "passed": "   "}]).refused

    def test_status_fail_family_all_block(self):
        for status in ("fail", "error", "warn", "FAIL", "Error"):
            v = compute_verdict([{"severity": "CRITICAL", "status": status}])
            # warn maps to a failed finding too; at CRITICAL that is a refusal.
            assert not v.passed, status

    def test_ambiguous_evidence_cannot_pass_under_strict(self):
        # No status, no passed -> lenient default is PASS (documented footgun); strict must
        # refuse to adjudicate rather than let a CRITICAL-shaped dict pass silently.
        loose = [{"severity": "CRITICAL", "message": "database dropped"}]
        assert compute_verdict(loose).passed  # the footgun, documented
        with pytest.raises(ValueError):
            compute_verdict(loose, strict=True)  # the safe posture the adapters use


# --- objective: smuggle an unknown/invalid severity past the gate -----------------------


class TestSeverityAxisCannotBeGamed:
    def test_unknown_severity_raises_rather_than_defaulting_low(self):
        # An attacker can't invent "CATASTROPHIC" to route around the CRITICAL branch; an
        # unrecognized severity is rejected, not silently downgraded to INFO/WARNING.
        with pytest.raises(ValueError):
            compute_verdict([{"severity": "CATASTROPHIC", "status": "fail"}])

    def test_severity_is_case_insensitive_but_closed_set(self):
        assert compute_verdict([Finding.fail("x", severity="critical")]).refused
        assert compute_verdict([Finding.fail("x", severity="Critical")]).refused

    def test_passing_critical_does_not_block(self):
        # severity is "how bad IF it failed" - a passed CRITICAL check is fine.
        assert compute_verdict([Finding.ok("x", severity="CRITICAL")]).passed

    def test_failed_info_is_a_contradiction_that_never_blocks(self):
        # A failed INFO is kept as a metric, not escalated. INFO is calibration only.
        v = compute_verdict([Finding.fail("x", severity="INFO")])
        assert v.passed
        assert v.metrics


# --- objective: exploit precedence / mixed findings -------------------------------------


class TestDecisionPrecedence:
    def test_one_critical_outranks_many_passes(self):
        findings = [Finding.ok(f"ok{i}") for i in range(50)]
        findings.append(Finding.fail("boom", severity="CRITICAL"))
        assert compute_verdict(findings).decision is Decision.FAIL

    def test_critical_beats_error_no_retry_escape(self):
        # A CRITICAL failure must be terminal FAIL, never demoted to a recoverable RETRY.
        v = compute_verdict(
            [Finding.fail("e", severity="ERROR"), Finding.fail("c", severity="CRITICAL")]
        )
        assert v.decision is Decision.FAIL
        assert not v.retryable

    def test_error_without_critical_is_retry_not_pass(self):
        assert compute_verdict([Finding.fail("e", severity="ERROR")]).decision is Decision.RETRY

    def test_warning_only_still_passes_but_is_recorded(self):
        v = compute_verdict([Finding.fail("w", severity="WARNING")])
        assert v.passed
        assert v.warnings  # surfaced, not swallowed


# --- objective: exploit "absence of evidence" ------------------------------------------


class TestEmptyEvidence:
    def test_empty_findings_pass_is_by_design_but_documented(self):
        # No findings == "no opinion" == PASS at the kernel. Callers that need affirmative
        # evidence must require it (gates do; see test_subversion_gates-style rollups).
        assert compute_verdict([]).passed

    def test_none_and_falsey_iterables(self):
        assert compute_verdict(iter(())).passed


# --- objective: break determinism / replayability --------------------------------------


class TestDeterminismCannotBeBroken:
    def test_same_findings_same_verdict_every_time(self):
        findings = [
            Finding.fail("a", severity="ERROR", message="x"),
            Finding.ok("b"),
            Finding.fail("c", severity="CRITICAL", message="y"),
        ]
        first = compute_verdict(copy.deepcopy(findings))
        # Findings carry a dict ``context`` (unhashable), so compare by value equality: the
        # frozen dataclasses must be byte-for-byte equal across every replay.
        for _ in range(25):
            assert compute_verdict(copy.deepcopy(findings)) == first

    def test_ordering_of_findings_does_not_change_decision(self):
        a = Finding.fail("a", severity="CRITICAL")
        b = Finding.fail("b", severity="ERROR")
        assert compute_verdict([a, b]).decision is compute_verdict([b, a]).decision

    def test_findings_are_immutable(self):
        f = Finding.ok("x")
        with pytest.raises(Exception):
            f.passed = False  # frozen dataclass; evidence can't be mutated post-hoc

    def test_verdict_is_immutable(self):
        v = compute_verdict([Finding.ok("x")])
        with pytest.raises(Exception):
            v.decision = Decision.FAIL


# --- objective: crash the kernel with hostile input (DoS / type confusion) --------------


class TestKernelIsRobustToHostileInput:
    def test_non_mapping_non_finding_raises_typeerror_not_pass(self):
        for junk in (42, 3.14, object()):
            with pytest.raises(TypeError):
                compute_verdict([junk])

    def test_deeply_nested_context_does_not_crash(self):
        nested = {
            "severity": "CRITICAL",
            "status": "fail",
            "context_blob": {"a": {"b": [1] * 1000}},
        }
        assert compute_verdict([nested]).refused

    def test_huge_finding_set_is_linear_and_correct(self):
        big = [Finding.ok(f"n{i}") for i in range(20_000)]
        big.append(Finding.fail("boom", severity="CRITICAL"))
        assert compute_verdict(big).refused
