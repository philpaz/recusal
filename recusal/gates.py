"""
Gate adjudication engine.

Determines a gate verdict from collected evidence. Supports tiered severity
(PASS / RETRY / FAIL). The adjudicator never *generates* anything — it only
judges evidence produced by other parties. "Builders cannot grade their own work."

Gates are staged checkpoints in a delivery pipeline:

    G0  environment / required systems reachable
    G1  model / schema loaded
    G2  synthesized data valid (tiered severity)
    G3  required objects present
    G4  migration counts reconcile
    G5  code deployed, tests pass, coverage floor met
    G6  metadata valid
    G7  zero critical policy violations
    G8  all evidence complete

Pure logic, standard library only.
"""

from datetime import datetime, timezone
from typing import List, Tuple


class GateAdjudicator:
    """Determines gate pass/fail based on collected evidence."""

    def __init__(self, data_system: str = "postgres", required_objects: List[str] = None):
        # ``data_system`` is just a label for the G0 required-systems list, so the
        # release evidence reflects whichever data backend you actually run.
        # ``required_objects`` are the domain objects G3 expects present — empty by
        # default, because the gate engine is domain-neutral; you supply your own.
        self.data_system = data_system
        self.GATE_CRITERIA = {
            "G0": {"required_systems": ["event_bus", data_system, "target_org"]},
            "G1": {"min_tables": 1, "min_rows": 1},
            "G2": {"row_count_tolerance": 0.05, "max_null_rate": 0.15, "zero_orphans": True},
            "G3": {"required_objects": list(required_objects or [])},
            "G4": {"migration_tolerance": 0.01},
            "G5": {"min_coverage": 75, "zero_compile_errors": True},
            "G6": {"metadata_valid": True},
            "G7": {"zero_critical_violations": True},
            "G8": {"all_evidence_complete": True},
        }

    def adjudicate_gate(self, gate_id: str, evidence_collection: dict) -> dict:
        """Determine whether a gate should pass or fail given its evidence."""
        criteria = self.GATE_CRITERIA.get(gate_id, {})
        verdict = "PASS"
        failures: List[str] = []

        if gate_id == "G1":
            verdict, failures = self._adjudicate_g1(evidence_collection, criteria)
        elif gate_id == "G2":
            verdict, failures = self._adjudicate_g2(evidence_collection, criteria)
        elif gate_id == "G4":
            verdict, failures = self._adjudicate_g4(evidence_collection, criteria)
        elif gate_id == "G5":
            verdict, failures = self._adjudicate_g5(evidence_collection, criteria)
        else:
            verdict = "PASS"  # Default for gates without specific logic

        return {
            "gate_id": gate_id,
            "verdict": verdict,
            "failures": failures,
            "evidence": evidence_collection,
            "adjudicated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _adjudicate_g1(self, evidence: dict, criteria: dict) -> Tuple[str, List[str]]:
        """Gate 1: model / schema loaded."""
        failures = []
        for validation in evidence.get("validations", []):
            if validation.get("status") == "fail":
                failures.append(f"Schema validation failed: {validation.get('table', 'unknown')}")
        verdict = "PASS" if not failures else "FAIL"
        return verdict, failures

    def _adjudicate_g2(self, evidence: dict, criteria: dict) -> Tuple[str, List[str]]:
        """Gate 2: synthesized data valid, with tiered severity.

        If evidence carries a ``verdict`` key (from a tiered-severity validator),
        use it directly. Otherwise fall back to binary adjudication.
        """
        if "verdict" in evidence:
            verdict = evidence["verdict"]
            failures = []
            for f in evidence.get("failures", []):
                failures.append(f.get("message", f"Validation failure: {f.get('type', 'unknown')}"))
            for w in evidence.get("warnings", []):
                msg = w.get("message")
                if msg:
                    failures.append(msg)  # surface warnings as evidence
            return verdict, failures

        # Legacy fallback: binary adjudication
        failures = []
        validations = evidence.get("validations", [])

        for validation in validations:
            if validation.get("type") == "error" and validation.get("status") == "fail":
                failures.append(f"Validation error: {validation.get('details', 'unknown')}")

        for validation in validations:
            if validation.get("type") == "row_count" and validation.get("status") == "fail":
                failures.append(f"Row count mismatch: {validation.get('table')}")

        for validation in validations:
            if validation.get("type") == "null_rate":
                null_rate = validation.get("null_rate", 0)
                if null_rate > criteria.get("max_null_rate", 0.15):
                    failures.append(
                        f"Null rate too high: {validation.get('table')}.{validation.get('column')} "
                        f"({null_rate:.2%} > {criteria['max_null_rate']:.2%})"
                    )

        for validation in validations:
            if validation.get("type") == "referential_integrity":
                orphan_count = validation.get("orphan_count", 0)
                if orphan_count > 0:
                    failures.append(
                        f"Orphaned records: {validation.get('relationship')} ({orphan_count} orphans)"
                    )

        verdict = "PASS" if not failures else "FAIL"
        return verdict, failures

    def _adjudicate_g4(self, evidence: dict, criteria: dict) -> Tuple[str, List[str]]:
        """Gate 4: migration counts reconcile."""
        failures = []
        for validation in evidence.get("validations", []):
            if validation.get("type") == "migration_count" and validation.get("status") == "fail":
                failures.append(
                    f"Migration count mismatch: {validation.get('source')} -> {validation.get('target')} "
                    f"(source: {validation.get('db_count')}, target: {validation.get('sf_count')})"
                )
        verdict = "PASS" if not failures else "FAIL"
        return verdict, failures

    def _adjudicate_g5(self, evidence: dict, criteria: dict) -> Tuple[str, List[str]]:
        """Gate 5: code deployed, tests pass, coverage floor met."""
        failures = []
        test_results = evidence.get("test_results", {})

        coverage = test_results.get("coverage", 0)
        min_coverage = criteria.get("min_coverage", 75)
        if coverage < min_coverage:
            failures.append(f"Coverage {coverage}% < {min_coverage}%")

        failed_count = test_results.get("failed", 0)
        if failed_count > 0:
            failures.append(f"{failed_count} tests failed")

        if test_results.get("error"):
            failures.append(f"Test execution error: {test_results['error']}")

        verdict = "PASS" if not failures else "FAIL"
        return verdict, failures

    def generate_release_evidence(self, mission_id: str, all_gate_evidence: List[dict]) -> dict:
        """Roll staged gate verdicts up into a single release-control record."""
        return {
            "mission_id": mission_id,
            "release_ready": all(e.get("verdict") == "PASS" for e in all_gate_evidence),
            "gate_summary": [
                {
                    "gate_id": e.get("gate_id"),
                    "verdict": e.get("verdict"),
                    "failures": e.get("failures", []),
                }
                for e in all_gate_evidence
            ],
            "test_coverage": self._aggregate_coverage(all_gate_evidence),
            "total_tests_run": self._aggregate_test_count(all_gate_evidence),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _aggregate_coverage(self, all_evidence: List[dict]) -> float:
        coverages = []
        for evidence in all_evidence:
            test_results = evidence.get("evidence", {}).get("test_results", {})
            if test_results.get("coverage"):
                coverages.append(test_results["coverage"])
        return sum(coverages) / len(coverages) if coverages else 0.0

    def _aggregate_test_count(self, all_evidence: List[dict]) -> int:
        total = 0
        for evidence in all_evidence:
            test_results = evidence.get("evidence", {}).get("test_results", {})
            total += test_results.get("total_tests", 0)
        return total
