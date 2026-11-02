#!/bin/sh
# Recusal Gate launcher: the fail-closed POSIX launcher from recusal/__main__.py
# (posix_launcher), pointed at this plugin's vendored gate. Registered by
# hooks/hooks.json as `sh "${CLAUDE_PLUGIN_ROOT}/scripts/gate.sh"`. Regenerate with
# `py tools/vendor_plugin.py`; tests/test_claude_plugin.py keeps it byte-for-byte.
[ -n "${CLAUDE_PLUGIN_ROOT}" ] || { echo 'recusal gate: CLAUDE_PLUGIN_ROOT is empty, so the gate script cannot be found; failing closed' >&2; exit 2; }; for p in python3 python py; do "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null && { "$p" "${CLAUDE_PLUGIN_ROOT}/scripts/recusal_gate.py"; rc=$?; [ "$rc" = 0 ] || { echo 'recusal gate: hook did not run cleanly; failing closed' >&2; exit 2; }; exit 0; }; done; echo 'recusal gate: no working python>=3.9 interpreter; failing closed' >&2; exit 2
