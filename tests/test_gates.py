"""Hermetic tests for the staged-gate adjudicator (built on the verdict kernel)."""

from recusal import Finding
from recusal.checks import referential_integrity, row_count
from recusal.gates import DEFAULT_GATES, GateAdjudicator, GateResult, ReleaseEvidence


def test_gate_folds_findings_into_a_typed_verdict():
    adj = GateAdjudicator()
    result = adj.adjudicate(
        "G2",
        [
            Finding.fail("null_rate", severity="ERROR", message="null rate 0.30 > 0.15"),
            Finding.fail("drift", severity="WARNING", message="distribution drift on product_code"),
        ],
    )
    assert isinstance(result, GateResult)
    assert result.verdict.retryable  # ERROR → RETRY
    assert result.decision.value == "RETRY"
    assert "null rate" in result.reasons()


def test_gate_critical_finding_refuses():
    adj = GateAdjudicator()
    result = adj.adjudicate(
        "G2",
        [referential_integrity([{"user_id": "M9"}], [{"id": "M1"}], fk="user_id", pk="id")],
    )
    assert not result.passed
    assert result.decision.value == "FAIL"
    assert "orphan" in result.reasons().lower()


def test_gate_accepts_loose_evidence_dicts():
    adj = GateAdjudicator()
    result = adj.adjudicate(
        "G5", [{"severity": "CRITICAL", "status": "fail", "message": "coverage 50% < 75%"}]
    )
    assert result.decision.value == "FAIL"
    assert "coverage" in result.reasons()


def test_clean_gate_passes():
    adj = GateAdjudicator()
    result = adj.adjudicate("G1", [row_count([{"id": 1}], min_rows=1)])
    assert result.passed
    assert result.decision.value == "PASS"


def test_release_requires_all_gates_pass():
    adj = GateAdjudicator()
    g1 = adj.adjudicate("G1", [row_count([{"id": 1}], min_rows=1)])
    g5_pass = adj.adjudicate("G5", [Finding.ok("tests", message="12 passed, coverage 90%")])
    release = adj.release("mission-1", [g1, g5_pass])
    assert isinstance(release, ReleaseEvidence)
    assert release.release_ready is True
    assert release.blocking == ()

    g5_fail = adj.adjudicate(
        "G5", [Finding.fail("tests", severity="CRITICAL", message="coverage 10% < 75%")]
    )
    blocked = adj.release("mission-1", [g1, g5_fail])
    assert blocked.release_ready is False
    assert blocked.blocking[0].gate_id == "G5"


def test_adjudicate_all_runs_in_gate_order():
    adj = GateAdjudicator()
    release = adj.adjudicate_all(
        "mission-2",
        {
            "G5": [Finding.fail("tests", severity="CRITICAL", message="coverage 10% < 75%")],
            "G1": [row_count([{"id": 1}], min_rows=1)],
        },
    )
    # emitted in adjudicator gate order (G1 before G5), not dict order
    assert [r.gate_id for r in release.results] == ["G1", "G5"]
    assert release.release_ready is False


def test_release_is_a_pure_function_of_findings():
    adj = GateAdjudicator()
    findings = [row_count([{"id": 1}], min_rows=1)]
    a = adj.release("m", [adj.adjudicate("G1", findings)]).to_dict()
    b = adj.release("m", [adj.adjudicate("G1", findings)]).to_dict()
    assert a == b  # no timestamps, no nondeterminism, replayable


def test_default_gates_are_domain_neutral():
    labels = " ".join(desc for _, desc in DEFAULT_GATES).lower()
    for vendor_word in ("salesforce", "postgres", "target_org", "event_bus", "migration"):
        assert vendor_word not in labels


def test_custom_gate_staging():
    adj = GateAdjudicator(gates=(("CHECK", "my one gate"),))
    assert adj.describe("CHECK") == "my one gate"
    # A gate adjudicated with real passing evidence passes.
    assert adj.adjudicate("CHECK", [Finding.ok("c", severity="INFO")]).passed


def test_empty_findings_gate_fails_closed():
    # A gate given NO evidence proved nothing: it must fail closed, not pass
    # vacuously (absence of evidence is not a pass). Regression for the empty-
    # findings vacuous-pass footgun that let an empty gate ship release_ready.
    adj = GateAdjudicator(gates=(("CHECK", "my one gate"),))
    result = adj.adjudicate("CHECK", [])
    assert not result.passed
    assert result.decision.value == "FAIL"
    rel = adj.release("m", [result], required=("CHECK",))
    assert rel.release_ready is False
