"""Recipe 8: conservative quality gate before a merge or deploy.

Run from the repository root: python examples/quality_gate.py

No CI or deployment is triggered; this module only evaluates supplied measurements.
Unusable measurements produce an ERROR finding (RETRY), never an empty PASS.
"""

import math

from recusal import Finding, compute_verdict


def _valid_percentage(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and 0 <= value <= 100
    except (TypeError, ValueError, OverflowError):
        return False


def quality_gate(coverage, failed, min_coverage=75):
    """Refuse missing, malformed and nonfinite quality evidence, without raising."""
    findings = []
    coverage_ok = _valid_percentage(coverage)
    minimum_ok = _valid_percentage(min_coverage)
    failures_ok = type(failed) is int and failed >= 0

    if not coverage_ok:
        findings.append(
            Finding.fail(
                "coverage_input",
                severity="ERROR",
                message="coverage must be a finite percentage between 0 and 100",
            )
        )
    if not minimum_ok:
        findings.append(
            Finding.fail(
                "minimum_coverage_input",
                severity="ERROR",
                message="minimum coverage must be a finite percentage between 0 and 100",
            )
        )
    if not failures_ok:
        findings.append(
            Finding.fail(
                "tests_input",
                severity="ERROR",
                message="failed tests must be a nonnegative integer",
            )
        )

    if coverage_ok and minimum_ok and coverage < min_coverage:
        findings.append(
            Finding.fail(
                "coverage_floor",
                severity="ERROR",
                message=f"coverage {coverage}% < required {min_coverage}%",
            )
        )
    if failures_ok and failed > 0:
        findings.append(
            Finding.fail("tests", severity="ERROR", message=f"{failed} test(s) failing")
        )
    return findings


def main():
    for label, coverage, failed in (
        ("below floor", 61, 2),
        ("clean", 80, 0),
        ("unknown coverage", float("nan"), 0),
        ("invalid failed count", 80, -1),
    ):
        result = compute_verdict(quality_gate(coverage, failed))
        print(f"{label}: {result.decision.value}")


if __name__ == "__main__":
    main()
