"""Recipe 2: require a top-level WHERE on explicit DELETE/UPDATE statements.

Run: python examples/sql_scope_policy.py

This is a lexical teaching policy, not a SQL parser or proof of bounded writes.
Comments, quoted strings/identifiers and nested expressions cannot supply WHERE
for an outer statement. Each semicolon-delimited statement is checked separately.
WHERE 1=1 still defers: predicates are not evaluated. Dynamic SQL, SQL constructed
inside a shell command and tools other than run_sql/query are not inspected.
Data-modifying statements inside CTEs are outside this top-level screening too.
Dialect-specific dollar quoting, unquoted # and backslash escapes in quotes are refused
rather than guessed. Extend the grammar deliberately for your database, and use
database permissions/transactions for the actual data-protection boundary.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding


def _statements(sql):
    """Return top-level word lists, or None for ambiguous/unclosed syntax."""
    statements, words = [], []
    i, depth = 0, 0
    while i < len(sql):
        if sql.startswith("--", i):
            end = sql.find("\n", i + 2)
            i = len(sql) if end < 0 else end + 1
        elif sql.startswith("/*", i):
            nesting = 1
            i += 2
            while i < len(sql) and nesting:
                if sql.startswith("/*", i):
                    nesting += 1
                    i += 2
                elif sql.startswith("*/", i):
                    nesting -= 1
                    i += 2
                else:
                    i += 1
            if nesting:
                return None
        elif sql[i] in "'\"`[":
            end_quote = "]" if sql[i] == "[" else sql[i]
            i += 1
            while i < len(sql):
                if sql[i] == "\\":
                    return None
                if sql[i] == end_quote:
                    i += 1
                    if i < len(sql) and sql[i] == end_quote:
                        i += 1
                        continue
                    break
                i += 1
            else:
                return None
        elif sql[i] == "$" and re.match(r"\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$", sql[i:]):
            return None
        elif sql[i] == "#":
            # MySQL/MariaDB comment syntax conflicts with PostgreSQL operators.
            return None
        elif sql[i] == "(":
            depth += 1
            i += 1
        elif sql[i] == ")":
            depth -= 1
            if depth < 0:
                return None
            i += 1
        elif sql[i] == ";":
            if depth:
                return None
            statements.append(words)
            words = []
            i += 1
        elif sql[i].isalnum() or sql[i] == "_":
            start = i
            while i < len(sql) and (sql[i].isalnum() or sql[i] in "_$"):
                i += 1
            if depth == 0:
                words.append(sql[start:i].upper())
        else:
            i += 1
    return None if depth else statements + [words]


def policy(tool_name, tool_input):
    """Inspect only explicit SQL arguments; an empty finding list means defer."""
    if tool_name not in ("run_sql", "query"):
        return []
    sql = tool_input.get("sql")
    statements = _statements(sql) if isinstance(sql, str) and sql.strip() else None
    reason = None
    if statements is None:
        reason = "missing, malformed or unsupported SQL syntax"
    else:
        for words in statements:
            if "DROP" in words or "TRUNCATE" in words:
                reason = "schema-destructive SQL (DROP/TRUNCATE)"
                break
            if ("DELETE" in words or "UPDATE" in words) and "WHERE" not in words:
                reason = "destructive SQL without a top-level WHERE clause"
                break
    return [Finding.fail("sql_scope", severity="CRITICAL", message=reason)] if reason else []


def main():
    for sql in (
        "SELECT updated_at FROM orders",
        "DELETE FROM users -- where id=1",
        "UPDATE t SET n=0 WHERE id=1",
    ):
        print(sql, "DENY" if policy("run_sql", {"sql": sql}) else "DEFER")


if __name__ == "__main__":
    main()
