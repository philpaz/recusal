"""Fuzz the audit log's readers with arbitrary bytes in place of a log file.

An audit file is read back by a process that did not write it, possibly after a crash, a
partial append or tampering. Whatever bytes are in it:

- ``verify_file`` answers ``(intact, problems)`` and never raises, and it never answers
  "intact" with a problem listed, or "not intact" with none;
- ``load`` (the tolerant reader an application resumes from) never raises;
- a file that verifies intact is a log ``load`` reads as the same entries that verify.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile

try:  # the fuzzing engine; absent when the harness is exercised by the test suite
    import atheris
except ImportError:  # pragma: no cover
    atheris = None

# Coverage guidance comes from instrumenting the code under test as it is imported.
# Without the engine the same imports run plain.
with atheris.instrument_imports() if atheris else contextlib.nullcontext():
    from recusal import audit

_DIR = tempfile.mkdtemp(prefix="recusal-fuzz-audit-")
_PATH = os.path.join(_DIR, "audit.jsonl")


def TestOneInput(data: bytes) -> None:
    with open(_PATH, "wb") as handle:
        handle.write(data)

    intact, problems = audit.verify_file(_PATH)
    assert isinstance(intact, bool) and isinstance(problems, list), (intact, problems)
    assert all(isinstance(problem, str) for problem in problems), problems
    assert intact == (not problems), (intact, problems)

    entries = audit.load(_PATH)
    assert isinstance(entries, list)
    if intact:
        again, why = audit.verify(entries)
        assert again and not why, why


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
