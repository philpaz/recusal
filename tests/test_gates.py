"""Hermetic tests for the gate adjudication engine."""

from recusal.gates import GateAdjudicator


def test_g2_tiered_verdict_passes_through():
    adj = GateAdjudicator()
    result = adj.adjudicate_gate(
        "G2",
        {
            "verdict": "RETRY",
            "failures": [{"type": "null_rate_high", "message": "[ERROR] null rate 0.30 > 0.15"}],
            "warnings": [{"message": "[WARNING] distribution drift on product_code"}],
        },
    )
    assert result["verdict"] == "RETRY"
    assert any("null rate" in f for f in result["failures"])
    assert any("distribution drift" in f for f in result["failures"])


def test_g2_legacy_orphans_fail():
    adj = GateAdjudicator()
    result = adj.adjudicate_gate(
        "G2",
        {
            "validations": [
                {
                    "type": "referential_integrity",
                    "relationship": "account->member",
                    "orphan_count": 4,
                },
            ]
        },
    )
    assert result["verdict"] == "FAIL"
    assert "Orphaned records" in result["failures"][0]


def test_g5_coverage_floor():
    adj = GateAdjudicator()
    passing = adj.adjudicate_gate("G5", {"test_results": {"coverage": 82, "failed": 0}})
    assert passing["verdict"] == "PASS"

    failing = adj.adjudicate_gate("G5", {"test_results": {"coverage": 50, "failed": 2}})
    assert failing["verdict"] == "FAIL"
    assert any("Coverage 50%" in f for f in failing["failures"])
    assert any("2 tests failed" in f for f in failing["failures"])


def test_g1_schema_fail():
    adj = GateAdjudicator()
    result = adj.adjudicate_gate("G1", {"validations": [{"status": "fail", "table": "members"}]})
    assert result["verdict"] == "FAIL"


def test_release_evidence_requires_all_gates_pass():
    adj = GateAdjudicator()
    g0 = adj.adjudicate_gate("G0", {})
    g5_pass = adj.adjudicate_gate(
        "G5", {"test_results": {"coverage": 90, "failed": 0, "total_tests": 12}}
    )
    release = adj.generate_release_evidence("mission-1", [g0, g5_pass])
    assert release["release_ready"] is True

    g5_fail = adj.adjudicate_gate("G5", {"test_results": {"coverage": 10, "failed": 3}})
    blocked = adj.generate_release_evidence("mission-1", [g0, g5_fail])
    assert blocked["release_ready"] is False


def test_data_system_label_reflected_in_g0_criteria():
    adj = GateAdjudicator(data_system="databricks")
    assert "databricks" in adj.GATE_CRITERIA["G0"]["required_systems"]
