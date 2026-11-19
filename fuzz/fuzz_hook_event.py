"""Fuzz the Claude Code PreToolUse hook with arbitrary bytes in place of the event.

The hook reads its event from stdin, which is written by another process. Whatever
arrives, with a policy that refuses every call:

- the hook never raises (an exception would exit the hook without a decision);
- it always emits a decision, and that decision is a deny: a garbled, truncated or
  oddly shaped event must not skip the gate;
- what it writes to stdout is one JSON object carrying the same decision it returned.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys

try:  # the fuzzing engine; absent when the harness is exercised by the test suite
    import atheris
except ImportError:  # pragma: no cover
    atheris = None

# Coverage guidance comes from instrumenting the code under test as it is imported.
# Without the engine the same imports run plain.
with atheris.instrument_imports() if atheris else contextlib.nullcontext():
    from recusal import Finding
    from recusal.claude_code import run_pretooluse_hook


def _refuse_everything(tool_name, tool_input):
    return [Finding.fail("fuzz_refuse", severity="CRITICAL", message="refused")]


def TestOneInput(data: bytes) -> None:
    text = data.decode("utf-8", errors="surrogateescape")
    out = io.StringIO()
    decision = run_pretooluse_hook(_refuse_everything, stdin=io.StringIO(text), stdout=out)

    assert isinstance(decision, dict), f"the hook deferred on {text[:80]!r}"
    specific = decision["hookSpecificOutput"]
    assert specific["permissionDecision"] == "deny", specific
    assert json.loads(out.getvalue()) == decision


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
