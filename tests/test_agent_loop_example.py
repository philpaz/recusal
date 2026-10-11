"""The framework-neutral agent loop must not pass a DELETE whose WHERE is only in a comment.

Regression for philpaz/recusal#79: a substring check for "where" let
"DELETE FROM orders -- where id=7" through.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from examples.agent_loop import evaluate
from recusal import compute_verdict

ROOT = Path(__file__).resolve().parents[1]


def _old_substring_check(sql: str) -> bool:
    """The check this regression replaced; kept here as the negative control."""
    s = sql.lower()
    return "delete" in s and "where" not in s


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DELETE FROM orders -- where id=7",
        "DELETE FROM orders /* where id=7 */",
    ],
)
def test_unscoped_delete_is_refused(sql):
    assert compute_verdict(evaluate("sql", {"sql": sql})).refused


def test_negative_control_old_check_passed_a_where_in_a_comment():
    # Proves the test above would have caught the bug: the old check let this through.
    assert not _old_substring_check("DELETE FROM orders -- where id=7")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders WHERE id = 7",
        "SELECT id FROM orders",
    ],
)
def test_scoped_delete_and_plain_select_are_allowed(sql):
    assert not evaluate("sql", {"sql": sql})


def test_example_runs_end_to_end():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "agent_loop.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "REFUSE  sql({'sql': 'DELETE FROM orders'})" in result.stdout
    assert "ALLOW   sql({'sql': 'DELETE FROM orders WHERE id = 7'})" in result.stdout
