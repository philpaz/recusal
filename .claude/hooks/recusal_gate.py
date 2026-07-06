#!/usr/bin/env python3
"""
Recusal governs its own repository.

This is a real Claude Code ``PreToolUse`` hook, registered via
``.claude/settings.json`` (copy ``.claude/settings.json.example`` to activate it, a
deliberate step Claude Code asks you to confirm). Once registered, when a Claude Code
session works on *this* repo, every tool call is adjudicated first: destructive shell
commands, writes to secret/protected files, and edits *or deletions* of the gate's own
configuration are refused before they run, even under bypassPermissions.

The deny-list engine itself now lives in the installable package, ``recusal.deny_list``
(:func:`recusal.deny_list.deny_list_policy`), so it is versioned, unit-tested, and fixes
reach adopters through ``pip install -U`` instead of a copy-paste. This file is the thin
shim that wires that policy into the hook and points it at *this* gate's control paths
(the module defaults already match, so ``deny_list_policy()`` needs no arguments here).

This is the **deny-list** path (refuse known-bad, defer the rest), used here deliberately:
a general-purpose dev repo runs an unbounded set of legitimate commands, so a default-deny
allowlist would be all friction. A deny-list is a *baseline*, not a guarantee, a determined
command can be obfuscated past any literal matcher, and for a narrow high-stakes channel the
**allowlist** path (:func:`recusal.claude_code.allowlist_policy`, refuse-by-default) fits
better. Neither is "better" in the abstract; the channel decides. What this file proves is
the seam, an independent gate that refuses before the tool runs and guards its own
kill-switch, not that this exact list is exhaustive.
"""

import os
import sys

# Make `recusal` importable from the repo without an install. Append (not insert-at-0):
# a repo-root file must NEVER shadow a stdlib module the package imports. If _REPO were at
# the front of sys.path, an agent could plant `<repo>/hashlib.py` (or json/shlex/re) -- a
# path that carries no protected segment, so a naive guard defers the write -- and it would
# be imported in place of the real stdlib module the next time the hook runs, hijacking or
# disabling the gate. Appending puts the standard library (and any installed distribution)
# first, so `recusal` is resolved from the repo only when nothing legitimate provides it,
# and the repo directory can shadow nothing. (`.claude/hooks`, the script dir Python puts at
# sys.path[0], is itself a protected control path, so it cannot be used to shadow either.)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_REPO)

from recusal.claude_code import run_pretooluse_hook  # noqa: E402
from recusal.deny_list import deny_list_policy  # noqa: E402

# The reference deny-list, aimed at this gate's own control paths (module defaults match).
# Exposed at module scope so the dogfood test suite can adjudicate against the real hook.
policy = deny_list_policy()


if __name__ == "__main__":
    run_pretooluse_hook(policy)
