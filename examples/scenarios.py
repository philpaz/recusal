"""
Reusable demo policies, each returns Recusal Findings for a proposed agent action.

These cover the common autonomous-agent failure modes (wrong target, destructive
action, unscoped mutation, data exfiltration, coverage floor, runaway volume).
Imported by both examples/gallery.py (to display) and tests/test_scenarios.py
(to assert) so the same policies are demonstrated *and* proven.

Severity is chosen by consequence: a wrong-target write or exfiltration is CRITICAL
(refuse, terminal); a missed coverage floor is ERROR (retry); high-but-not-runaway
volume is WARNING (allow, but record it).
"""

import os

from recusal import Finding

SAFE_PATH = os.path.abspath("/workspace/tmp")
EMAIL_ALLOWLIST = {"acme.com"}


def wrong_subject(tool_input: dict, active_id: str) -> list:
    """A write must target the session's active customer (the right-subject invariant)."""
    target = tool_input.get("customer_id")
    if target != active_id:
        return [
            Finding.fail(
                "subject_match",
                severity="CRITICAL",
                message=f"write targets {target}, not the active customer {active_id}",
                target=target,
                active=active_id,
            )
        ]
    return [Finding.ok("subject_match", severity="CRITICAL", target=target)]


def destructive_path(tool_input: dict) -> list:
    """A file delete/write must stay inside the safe path."""
    path = tool_input.get("path", "")
    target = os.path.abspath(path)
    try:  # commonpath, not startswith ("/workspace/tmp_evil" would slip past startswith)
        inside = os.path.commonpath([SAFE_PATH, target]) == SAFE_PATH
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        return [
            Finding.fail(
                "path_allowlist",
                severity="CRITICAL",
                message=f"{path} is outside the safe path {SAFE_PATH}",
                path=path,
            )
        ]
    return [Finding.ok("path_allowlist", severity="CRITICAL", path=path)]


def unscoped_sql(tool_input: dict) -> list:
    """Destructive SQL must be scoped by a WHERE clause."""
    sql = str(tool_input.get("sql", "")).lower()
    if ("delete" in sql or "update" in sql) and "where" not in sql:
        return [
            Finding.fail(
                "sql_scope",
                severity="CRITICAL",
                message="destructive SQL without a WHERE clause",
                sql=tool_input.get("sql"),
            )
        ]
    return [Finding.ok("sql_scope", severity="CRITICAL")]


def data_exfiltration(tool_input: dict) -> list:
    """Outbound email must go to an allowlisted domain (guards prompt-injected exfiltration)."""
    to = str(tool_input.get("to", ""))
    domain = to.split("@")[-1].lower() if "@" in to else ""
    if domain not in EMAIL_ALLOWLIST:
        return [
            Finding.fail(
                "egress_allowlist",
                severity="CRITICAL",
                message=f"recipient domain '{domain}' is not on the allowlist",
                to=to,
            )
        ]
    return [Finding.ok("egress_allowlist", severity="CRITICAL", to=to)]


def coverage_floor(coverage: float, floor: float = 75) -> list:
    """Below the coverage floor is recoverable, RETRY, not a terminal refusal."""
    if coverage < floor:
        return [
            Finding.fail(
                "coverage_floor",
                severity="ERROR",
                message=f"coverage {coverage}% < required {floor}%",
                coverage=coverage,
            )
        ]
    return [Finding.ok("coverage_floor", severity="ERROR", coverage=coverage)]


def action_budget(count: int, soft: int = 25, hard: int = 100) -> list:
    """Tiered runaway control: over soft → warn (allow); over hard → RETRY (stop the loop)."""
    if count > hard:
        return [
            Finding.fail(
                "action_budget",
                severity="ERROR",
                message=f"{count} actions exceeds the hard cap {hard}",
                count=count,
            )
        ]
    if count > soft:
        return [
            Finding.fail(
                "action_budget",
                severity="WARNING",
                message=f"{count} actions over the soft budget {soft}",
                count=count,
            )
        ]
    return [Finding.ok("action_budget", severity="INFO", count=count)]
