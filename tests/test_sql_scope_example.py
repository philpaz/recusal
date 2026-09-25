"""The SQL recipe distinguishes lexical scope from a data-safety guarantee."""

import pytest

from examples.sql_scope_policy import policy
from recusal import compute_verdict


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM users -- where id = 1",
        "DELETE FROM users /* where */",
        "SELECT 1; DELETE FROM t",
        "UPDATE t SET note = 'where'",
        'UPDATE t SET "where" = 1',
        "UPDATE t SET [where] = 1",
        "UPDATE t SET `where` = 1",
        "UPDATE t SET note = 'it''s where'",
        "UPDATE t SET n = (SELECT n FROM s WHERE id=1)",
        "DELETE FROM t /* outer /* where */ still comment */",
        "dElEtE FROM users",
        "DROP TABLE t; SELECT 1 WHERE 1=1",
        "TRUNCATE t",
        "SELECT 1 WHERE 1=1; UPDATE t SET n=0",
        "UPDATE t SET note=$$where$$",
        "UPDATE t SET note=$tag$where$tag$",
        "UPDATE t SET note='unterminated",
        "DELETE FROM t /* unclosed",
        "UPDATE t SET n=(1",
        "UPDATE t SET n=1)",
        "UPDATE t SET note='back\\slash' WHERE id=1",
    ],
)
@pytest.mark.parametrize("tool", ["run_sql", "query"])
def test_unscoped_or_unsupported_statements_are_refused(sql, tool):
    assert compute_verdict(policy(tool, {"sql": sql})).refused


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT updated_at FROM orders",
        "SELECT * FROM deleted_items",
        "SELECT 'DELETE; DROP TABLE users'",
        "UPDATE accounts SET balance = 0 WHERE 1=1",
        "uPdAtE t SET n=1 wHeRe id=2",
        "UPDATE t SET note='; where' WHERE id=1; DELETE FROM s WHERE id=2",
        "-- DELETE FROM t\nSELECT 1",
        "UPDATE/* separator */t SET n=1 WHERE id=1",
        "DELETE FROM t WHERE id IN (SELECT id FROM s)",
    ],
)
def test_scoped_or_non_destructive_sql_defers(sql):
    assert policy("run_sql", {"sql": sql}) == []


@pytest.mark.parametrize("value", [None, "", "  ", 42])
def test_missing_or_invalid_sql_is_refused(value):
    assert compute_verdict(policy("run_sql", {"sql": value})).refused


def test_shell_sql_is_explicitly_outside_scope():
    assert policy("Bash", {"command": "psql -c 'DELETE FROM users'"}) == []
