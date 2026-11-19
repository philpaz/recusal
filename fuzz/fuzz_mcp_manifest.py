"""Fuzz the MCP manifest loader with arbitrary bytes in place of a pinned manifest.

A pinned manifest is a file on disk that decides which MCP servers and tools are
approved, so a damaged or hand-edited one must be refused cleanly. Whatever bytes are
in it:

- ``load_manifest`` either returns a JSON object or raises ``ValueError`` (which
  ``json.JSONDecodeError`` and ``UnicodeDecodeError`` both are), and nothing else;
- a manifest it accepts is accepted again when written back out and reloaded, so
  acceptance does not depend on the bytes' formatting.
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
    from recusal import mcp

_DIR = tempfile.mkdtemp(prefix="recusal-fuzz-manifest-")
_PATH = os.path.join(_DIR, "manifest.json")
_AGAIN = os.path.join(_DIR, "again.json")


def TestOneInput(data: bytes) -> None:
    with open(_PATH, "wb") as handle:
        handle.write(data)
    try:
        manifest = mcp.load_manifest(_PATH)
    except ValueError:
        return
    assert isinstance(manifest, dict), type(manifest)

    with open(_AGAIN, "w", encoding="utf-8") as handle:
        handle.write(mcp.manifest_to_text(manifest))
    assert mcp.load_manifest(_AGAIN) == manifest


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
