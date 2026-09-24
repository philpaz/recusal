"""Recipe 7: count attempted actions across fresh hook processes.

Run: python examples/action_budget.py /path/to/session-budget.sqlite

The adopter supplies a counter callable to ``make_action_budget``. The included
SQLite counter commits each increment before returning it; new objects and fresh
processes using the same file continue the count. A new file starts at zero.
Choose a separate, protected file per session/budget and retain it across hook
restarts. Deleting it resets the budget. This is local state, not tamper-proof
storage; do not let the agent select or modify the file. The parent directory
must already exist. Storage errors propagate so the hook can refuse safely.

Every policy invocation counts, including refused attempts and retries, not just
successful tool calls. SQLite serializes concurrent increments. The counter owns
I/O; the existing action_budget finding function remains pure. A WARNING is a
soft-budget signal, not a stop; beyond the hard cap it produces ERROR/RETRY.
"""

import os
import sqlite3
import sys
from contextlib import closing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.scenarios import action_budget


class SQLiteCounter:
    """A durable, one-budget counter at an explicitly chosen filesystem path."""

    def __init__(self, path):
        self.path = os.fspath(path)
        if self.path in ("", ":memory:"):
            raise ValueError("choose a persistent counter file")

    def __call__(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS action_count "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), n INTEGER NOT NULL CHECK (n >= 0))"
            )
            db.execute("INSERT OR IGNORE INTO action_count VALUES (1, 0)")
            db.execute("UPDATE action_count SET n = n + 1 WHERE id = 1")
            count = db.execute("SELECT n FROM action_count WHERE id = 1").fetchone()[0]
        return count


def make_action_budget(counter, soft=25, hard=100):
    """Bind a caller-owned increment-and-return counter to a hook policy."""
    if type(soft) is not int or type(hard) is not int or not 0 <= soft <= hard:
        raise ValueError("budgets must be integers with 0 <= soft <= hard")

    def policy(tool_name, tool_input):
        return action_budget(counter(), soft=soft, hard=hard)

    return policy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counter_file")
    args = parser.parse_args()
    print(make_action_budget(SQLiteCounter(args.counter_file))("demo", {}))
