"""Fuzz the verdict fold with arbitrary JSON in place of a policy's findings.

``compute_verdict`` is the one function every gate's decision passes through, and what
it is handed comes from policy code and, in the CLI, from a findings file. For any JSON
value:

- in strict mode it returns a verdict or raises ``ValueError`` or ``TypeError``, and
  nothing else; lenient mode is held to the same;
- strict mode never passes evidence that lenient mode fails or retries: asking for
  stricter reading can only refuse more;
- the decision is always one of PASS, RETRY and FAIL, and ``passed`` agrees with it.
"""

from __future__ import annotations

import contextlib
import json
import sys

try:  # the fuzzing engine; absent when the harness is exercised by the test suite
    import atheris
except ImportError:  # pragma: no cover
    atheris = None

# Coverage guidance comes from instrumenting the code under test as it is imported.
# Without the engine the same imports run plain.
with atheris.instrument_imports() if atheris else contextlib.nullcontext():
    from recusal.evidence import Decision, compute_verdict

_RANK = {Decision.PASS: 0, Decision.RETRY: 1, Decision.FAIL: 2}


def _fold(findings, strict):
    try:
        verdict = compute_verdict(findings, strict=strict)
    except (ValueError, TypeError):
        return None
    assert verdict.decision in _RANK, verdict.decision
    assert verdict.passed == (verdict.decision is Decision.PASS), verdict
    return verdict


def TestOneInput(data: bytes) -> None:
    try:
        value = json.loads(data)
    except (ValueError, RecursionError):
        return
    findings = value if isinstance(value, list) else [value]

    lenient = _fold(findings, strict=False)
    strict = _fold(findings, strict=True)
    if lenient is not None and strict is not None:
        assert _RANK[strict.decision] >= _RANK[lenient.decision], (strict, lenient)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
