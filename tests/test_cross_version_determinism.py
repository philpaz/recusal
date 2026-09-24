"""The same evidence must give the same verdict on every supported Python (3.9 to 3.14).

Some standard-library parsers accept different input on different Python versions:
``date.fromisoformat``, ``datetime.fromisoformat`` and ``time.fromisoformat`` accept far
more formats from 3.11 on (``20260615``, ``2026-06-15T10:00:00.5``, ``2026-W24-1``), so
a check built on them passes on 3.12 and fails on 3.9 for the same row. That breaks the
determinism the verdict rests on, so the package does not call them; parse against one
explicit grammar instead (CONTRIBUTING.md, "Principles").

The guard reads the syntax tree, not the text, so a comment or docstring that names a
parser (like this one) is not a violation, while any call through any alias is.
"""

import ast
import os

PACKAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recusal")

#: Attribute names whose accepted input differs across the supported Python versions.
VERSION_DEPENDENT_PARSERS = frozenset({"fromisoformat"})


def _violations(source, filename):
    found = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Attribute) and node.attr in VERSION_DEPENDENT_PARSERS:
            found.append(f"{os.path.basename(filename)}:{node.lineno} uses .{node.attr}")
    return found


def test_the_package_uses_no_version_dependent_parser():
    violations = []
    for name in sorted(os.listdir(PACKAGE)):
        if name.endswith(".py"):
            path = os.path.join(PACKAGE, name)
            with open(path, encoding="utf-8") as fh:
                violations += _violations(fh.read(), path)
    assert not violations, (
        "a version-dependent parser gives different verdicts on different Pythons; parse "
        "against one explicit grammar instead (CONTRIBUTING.md): " + "; ".join(violations)
    )


def test_the_guard_catches_a_call_and_ignores_prose():
    # the guard itself must work: a real call is caught, a mention in text is not
    assert _violations("from datetime import date\ndate.fromisoformat(s)\n", "x.py")
    assert _violations("import datetime as d\nd.datetime.fromisoformat(s)\n", "x.py")
    assert not _violations('"""date.fromisoformat is banned"""\n# fromisoformat\n', "x.py")
