"""The evaluate_policy / certify_evidence split (0.6.0 evidence-semantics decision).

Empty evidence means two different things and the kernel now names both:
``evaluate_policy`` reads absence as "no objection" (PASS), ``certify_evidence``
reads it as "nothing proven" (refuse). On non-empty input both are the same fold
as ``compute_verdict``; these tests lock the parity, the empty-input semantics,
the strict defaults, and generator safety.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import recusal
from recusal.evidence import (
    Decision,
    Finding,
    Severity,
    certify_evidence,
    compute_verdict,
    evaluate_policy,
)

DETERMINISTIC = settings(derandomize=True, max_examples=150)

findings = st.builds(
    Finding,
    check=st.text(min_size=1, max_size=8),
    severity=st.sampled_from(list(Severity)),
    passed=st.booleans(),
    message=st.text(max_size=16),
)
finding_lists = st.lists(findings, max_size=12)
nonempty_finding_lists = st.lists(findings, min_size=1, max_size=12)


def test_both_names_are_exported_from_the_package_root():
    assert recusal.evaluate_policy is evaluate_policy
    assert recusal.certify_evidence is certify_evidence
    assert "evaluate_policy" in recusal.__all__
    assert "certify_evidence" in recusal.__all__


def test_empty_policy_evaluation_passes():
    verdict = evaluate_policy([])
    assert verdict.decision is Decision.PASS
    assert verdict.passed


def test_empty_certification_refuses_and_explains_itself():
    verdict = certify_evidence([])
    assert verdict.decision is Decision.FAIL
    assert verdict.refused
    assert verdict.highest_severity is Severity.CRITICAL
    assert [f.check for f in verdict.failures] == ["no_evidence"]
    assert "no evidence" in verdict.reasons()


def test_certification_is_strict_by_default_about_outcomeless_evidence():
    ambiguous = {"severity": "CRITICAL", "message": "looks fine"}
    with pytest.raises(ValueError):
        certify_evidence([ambiguous])
    # The policy surface keeps compute_verdict's documented lenient default.
    assert evaluate_policy([ambiguous]).decision is Decision.PASS
    # And the escape hatch stays explicit, never implicit.
    assert certify_evidence([ambiguous], strict=False).decision is Decision.PASS


def test_certification_materializes_generators_before_deciding_emptiness():
    made = (f for f in [Finding.fail("g", severity="CRITICAL")])
    assert certify_evidence(made).decision is Decision.FAIL
    assert certify_evidence(f for f in []).failures[0].check == "no_evidence"


@DETERMINISTIC
@given(finding_lists)
def test_evaluate_policy_is_the_raw_fold_for_all_inputs(items):
    assert evaluate_policy(items) == compute_verdict(items)


@DETERMINISTIC
@given(nonempty_finding_lists)
def test_certify_evidence_matches_the_raw_fold_on_nonempty_input(items):
    assert certify_evidence(items) == compute_verdict(items, strict=True)


@DETERMINISTIC
@given(nonempty_finding_lists)
def test_certification_never_weakens_the_raw_verdict(items):
    rank = {Decision.PASS: 0, Decision.RETRY: 1, Decision.FAIL: 2}
    assert rank[certify_evidence(items).decision] >= rank[compute_verdict(items).decision]


@DETERMINISTIC
@given(nonempty_finding_lists)
def test_both_surfaces_preserve_warnings_and_metrics(items):
    raw = compute_verdict(items)
    for split_verdict in (evaluate_policy(items), certify_evidence(items)):
        assert split_verdict.warnings == raw.warnings
        assert split_verdict.metrics == raw.metrics


loose_dicts = st.lists(
    st.fixed_dictionaries(
        {
            "check": st.text(min_size=1, max_size=8),
            "severity": st.sampled_from([s.value for s in Severity]),
            "status": st.sampled_from(["pass", "passed", "ok", "fail", "failed", "denied"]),
        }
    ),
    max_size=8,
)


@DETERMINISTIC
@given(loose_dicts)
def test_loose_dict_evidence_folds_identically_through_both_names(dicts):
    assert evaluate_policy(dicts) == compute_verdict(dicts)
    if dicts:
        assert certify_evidence(dicts) == compute_verdict(dicts, strict=True)


def test_empty_certification_verdict_shape_is_complete():
    verdict = certify_evidence([])
    assert not verdict.passed
    assert not verdict.retryable
    assert verdict.warnings == ()
    assert verdict.metrics == ()
    assert len(verdict.failures) == 1
    assert verdict.failures[0].severity is Severity.CRITICAL
    assert not verdict.failures[0].passed
    assert certify_evidence([]) == verdict  # deterministic, same verdict every time


def test_certification_of_only_passing_or_info_evidence_passes():
    assert certify_evidence([Finding.ok("built", severity="CRITICAL")]).passed
    only_metrics = certify_evidence([Finding.ok("timing", severity="INFO", message="2s")])
    assert only_metrics.passed
    assert len(only_metrics.metrics) == 1


def test_strict_mode_parity_with_compute_verdict_in_both_directions():
    ambiguous = {"severity": "ERROR", "message": "no outcome stated"}
    with pytest.raises(ValueError):
        evaluate_policy([ambiguous], strict=True)
    with pytest.raises(ValueError):
        compute_verdict([ambiguous], strict=True)


def test_gate_adjudicator_released_behavior_is_untouched_by_the_split():
    from recusal.gates import GateAdjudicator

    result = GateAdjudicator().adjudicate("G0", [])
    assert result.verdict.decision is Decision.FAIL
    assert [f.check for f in result.verdict.failures] == ["evidence_error"]
