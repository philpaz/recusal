"""Sync the recusal-gate plugin's vendored runtime from the package source.

The plugin adjudicates with its OWN copy of recusal (``claude-plugin/vendor/recusal``),
so the installed plugin identity IS the implementation that decides - no pip install,
no ambient-package substitution. That claim only holds while the vendored copy is
byte-identical to ``recusal/``; a drift-lock test (tests/test_claude_plugin.py)
enforces it, and this script is the one way to re-sync:

    py tools/vendor_plugin.py

Deterministic: copies every ``*.py`` plus ``py.typed`` from ``recusal/`` into the
vendor tree and deletes anything else found there, so the vendor tree is a pure
function of the package source.
"""

import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO_ROOT, "recusal")
VENDOR = os.path.join(REPO_ROOT, "claude-plugin", "vendor", "recusal")


def vendored_names():
    return sorted(name for name in os.listdir(SOURCE) if name.endswith(".py") or name == "py.typed")


def main() -> int:
    if os.path.isdir(VENDOR):
        shutil.rmtree(VENDOR)
    os.makedirs(VENDOR)
    for name in vendored_names():
        shutil.copyfile(os.path.join(SOURCE, name), os.path.join(VENDOR, name))
    print(f"vendored {len(vendored_names())} file(s) -> {os.path.relpath(VENDOR, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
