"""Fuzz the reference deny-list with arbitrary text in place of a shell command.

The deny-list reads a command a model wrote. Whatever the text:

- the policy never raises (a policy that raises is refused by the gate, which is safe,
  but the deny-list is meant to answer, not to crash on odd input);
- it returns a list of findings, the same list for the same command;
- padding a command with whitespace does not turn a refusal into no objection.
"""

from __future__ import annotations

import contextlib
import sys

try:  # the fuzzing engine; absent when the harness is exercised by the test suite
    import atheris
except ImportError:  # pragma: no cover
    atheris = None

# Coverage guidance comes from instrumenting the code under test as it is imported.
# Without the engine the same imports run plain.
with atheris.instrument_imports() if atheris else contextlib.nullcontext():
    from recusal.deny_list import deny_list_policy

_POLICY = deny_list_policy()


def _refused(findings) -> bool:
    return any(not finding.passed for finding in findings)


def TestOneInput(data: bytes) -> None:
    command = data.decode("utf-8", errors="replace")
    findings = _POLICY("Bash", {"command": command})
    assert isinstance(findings, list), type(findings)
    again = _POLICY("Bash", {"command": command})
    assert [repr(f) for f in findings] == [repr(f) for f in again]
    if _refused(findings):
        assert _refused(_POLICY("Bash", {"command": f"  {command}\t"})), command[:80]


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
