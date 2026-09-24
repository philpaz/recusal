"""The cookbook budget counts across calls and process lifetimes."""

import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from examples.action_budget import SQLiteCounter, make_action_budget


def test_budget_boundaries_and_refused_attempts(tmp_path):
    policy = make_action_budget(SQLiteCounter(tmp_path / "budget.db"), soft=2, hard=4)
    findings = [policy("Bash", {})[0] for _ in range(6)]
    assert [f.context["count"] for f in findings] == [1, 2, 3, 4, 5, 6]
    assert [f.passed for f in findings] == [True, True, False, False, False, False]
    assert [f.severity.value for f in findings] == [
        "INFO",
        "INFO",
        "WARNING",
        "WARNING",
        "ERROR",
        "ERROR",
    ]


def test_fresh_counter_and_reopened_counter(tmp_path):
    path = tmp_path / "budget.db"
    assert SQLiteCounter(path)() == 1
    assert SQLiteCounter(path)() == 2
    assert SQLiteCounter(tmp_path / "other.db")() == 1


def test_counter_survives_fresh_processes(tmp_path):
    path = tmp_path / "budget.db"
    code = (
        "import sys; from examples.action_budget import SQLiteCounter; "
        "print(SQLiteCounter(sys.argv[1])())"
    )
    results = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(path)],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
        for _ in range(3)
    ]
    assert results == ["1", "2", "3"]


def test_concurrent_increments_are_not_lost(tmp_path):
    path = tmp_path / "budget.db"
    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(lambda _: SQLiteCounter(path)(), range(12)))
    assert sorted(counts) == list(range(1, 13))


def test_storage_failure_is_not_silently_reset(tmp_path):
    path = tmp_path / "budget.db"
    path.write_text("not a database")
    with pytest.raises(sqlite3.DatabaseError):
        make_action_budget(SQLiteCounter(path))("Bash", {})


def test_counter_storage_is_replaceable():
    counts = iter([1, 26, 101])
    policy = make_action_budget(lambda: next(counts))
    assert [policy("tool", {})[0].context["count"] for _ in range(3)] == [1, 26, 101]


@pytest.mark.parametrize("soft,hard", [(-1, 2), (3, 2), (True, 2), (1, 2.5)])
def test_invalid_budgets_are_rejected(soft, hard):
    with pytest.raises(ValueError):
        make_action_budget(lambda: 1, soft=soft, hard=hard)


@pytest.mark.parametrize("path", ["", ":memory:"])
def test_counter_requires_persistent_storage(path):
    with pytest.raises(ValueError):
        SQLiteCounter(path)
