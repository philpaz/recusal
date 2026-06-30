"""
Quickstart: collect evidence, then let the gate decide — and refuse.

    python examples/quickstart.py
"""

from recusal import GateAdjudicator, compute_verdict

# 1. The tiered-severity kernel. Evidence in, deterministic verdict out.
#    CRITICAL refuses outright; ERROR earns exactly one retry; the rest pass.
findings = [
    {"severity": "CRITICAL", "status": "fail", "message": "0 rows synthesized in `members`"},
    {"severity": "WARNING", "status": "warn", "message": "product-mix drift on `account_type`"},
    {"severity": "INFO", "status": "pass", "metric": "row_count", "value": 0},
]
verdict = compute_verdict(findings)
print(f"verdict          : {verdict.decision.value}")        # FAIL
print(f"highest_severity : {verdict.highest_severity.value}")  # CRITICAL
print(f"message          : {verdict.message}")               # refused, no retry

# 2. Staged gates roll up into a single release decision. The gate can say no.
gate = GateAdjudicator()
g5 = gate.adjudicate_gate("G5", {"test_results": {"coverage": 61, "failed": 1, "total_tests": 9}})
print(f"\nG5 verdict       : {g5['verdict']}  ({g5['failures']})")  # FAIL

release = gate.generate_release_evidence("UC-001", [g5])
print(f"release_ready    : {release['release_ready']}")     # False — it refuses to ship
