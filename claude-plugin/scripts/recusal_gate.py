#!/usr/bin/env python3
"""Recusal PreToolUse gate, the plugin edition.

Installed via the recusal-gate Claude Code plugin, every tool call in a session is
adjudicated here BEFORE it runs; a deny holds even under bypassPermissions. This is the
same thin shim the recusal repository dogfoods and ``python -m recusal init`` scaffolds:
the policy logic lives in the pip-installed ``recusal`` package, so fixes reach you
through ``pip install -U recusal``, not a plugin update.

A plugin cannot vendor its Python dependency, so the package is required separately, and
the failure mode is deliberate: if ``recusal`` is not importable in the interpreter the
launcher found, this script exits nonzero and the registered launcher coerces that into
exit 2, the one BLOCKING hook exit code. Missing dependency = every tool call refused,
never a silently absent gate.

To customize the policy, scaffold a project-local gate instead (``python -m recusal
init``) and edit ``.claude/hooks/recusal_gate.py``; a project-local deny wins alongside
this plugin (hooks compose, deny-wins).
"""

import sys

try:
    from recusal.claude_code import run_pretooluse_hook
    from recusal.deny_list import deny_list_policy
except ImportError:
    sys.stderr.write(
        "recusal gate (plugin): the `recusal` package is not installed for this "
        "interpreter; failing closed. Fix: pip install recusal\n"
    )
    sys.exit(3)  # any nonzero exit -> the registered launcher coerces to blocking exit 2

# Reference deny-list: refuse known-destructive commands (recursive deletes,
# force-pushes, secret-file writes, gate kill-switch edits), defer everything else
# to Claude Code's normal permission flow.
policy = deny_list_policy()


if __name__ == "__main__":
    run_pretooluse_hook(policy)
