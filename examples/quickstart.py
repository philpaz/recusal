"""
Quickstart: collect evidence, then let the gate decide — and refuse.

    python examples/quickstart.py
"""

import os
import sys

# Run straight from a clone, no install needed: put the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding, GateAdjudicator, compute_verdict  # noqa: E402

# 1. The tiered-severity kernel. Evidence in, deterministic verdict out.
#    CRITICAL refuses outright; ERROR earns exactly one retry; the rest pass.
findings = [
    {"severity": "CRITICAL", "status": "fail", "message": "0 rows synthesized in `members`"},
    {"severity": "WARNING", "status": "warn", "message": "product-mix drift on `account_type`"},
    {"severity": "INFO", "status": "pass", "metric": "row_count", "value": 0},
]
verdict = compute_verdict(findings)
print(f"verdict          : {verdict.decision.value}")  # FAIL
print(f"highest_severity : {verdict.highest_severity.value}")  # CRITICAL
print(f"message          : {verdict.message}")  # refused, no retry

# 2. Staged gates roll up into a single release decision. Each gate is just
#    compute_verdict at a checkpoint — the same kernel, applied to a pipeline.
gate = GateAdjudicator()
g5 = gate.adjudicate(
    "G5",
    [Finding.fail("coverage_floor", severity="CRITICAL", message="coverage 61% < 75%")],
)
print(f"\nG5 verdict       : {g5.decision.value}  ({g5.reasons()})")  # FAIL

release = gate.release("UC-001", [g5])
print(f"release_ready    : {release.release_ready}")  # False — it refuses to ship
print(f"blocking gates   : {[r.gate_id for r in release.blocking]}")  # ['G5']
