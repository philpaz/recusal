#!/usr/bin/env python3
"""Recusal PreToolUse gate, the plugin edition.

Installed via the recusal-gate Claude Code plugin, every tool call in a session is
adjudicated here BEFORE it runs; a deny holds even under bypassPermissions. This is the
same thin shim the recusal repository dogfoods and ``python -m recusal init`` scaffolds.

**Declared-version binding**: a deterministic control must be identifiable, so this
gate refuses an importable ``recusal`` package whose declared ``__version__`` differs
from ``EXPECTED_RECUSAL_VERSION`` below, failing CLOSED with both versions named -
plugin X adjudicating with recusal Y would make the audit trail misname what decided.
Stated precisely: this binds the plugin to a DECLARED package version. It does not
attest package bytes, installation provenance, or a modified package that retains the
expected version string; for stronger assurance install the exact version into a
dedicated, write-protected virtual environment (hash-locked where required). Fix a
mismatch with ``pip install "recusal==<expected>"``, or update the plugin so the pair
advances together.

Claude Code does not automatically install Python packages for a plugin, so the package
is installed separately, and the failure mode is deliberate: if ``recusal`` is not
importable in the interpreter the launcher found, this script exits nonzero and the
registered launcher coerces that into exit 2, the one BLOCKING hook exit code. Missing
or mismatched dependency = every tool call refused, never a silently absent gate.

To customize the policy, scaffold a project-local gate instead (``python -m recusal
init``) and edit ``.claude/hooks/recusal_gate.py``; a project-local deny wins alongside
this plugin (hooks compose, deny-wins).
"""

import sys

#: The exact adjudicator this plugin version identifies. A version-drift-lock test keeps
#: this equal to the plugin manifest version and the package version.
EXPECTED_RECUSAL_VERSION = "0.5.3"

try:
    import recusal
    from recusal.claude_code import run_pretooluse_hook
    from recusal.deny_list import deny_list_policy
except ImportError:
    sys.stderr.write(
        "recusal gate (plugin): the `recusal` package is not installed for this "
        "interpreter; failing closed. Fix: pip install "
        f'"recusal=={EXPECTED_RECUSAL_VERSION}"\n'
    )
    sys.exit(3)  # any nonzero exit -> the registered launcher coerces to blocking exit 2

if getattr(recusal, "__version__", None) != EXPECTED_RECUSAL_VERSION:
    sys.stderr.write(
        f"recusal gate (plugin): plugin {EXPECTED_RECUSAL_VERSION} found recusal "
        f"{getattr(recusal, '__version__', 'unknown')}; refusing to adjudicate under a "
        "mismatched identity (the audit trail must name what decided). Fix: pip install "
        f'"recusal=={EXPECTED_RECUSAL_VERSION}", or update the plugin.\n'
    )
    sys.exit(3)  # coerced to blocking exit 2 by the launcher: fail closed, named

# Reference deny-list: refuse known-destructive commands (recursive deletes,
# force-pushes, secret-file writes, gate kill-switch edits), defer everything else
# to Claude Code's normal permission flow.
policy = deny_list_policy()


if __name__ == "__main__":
    run_pretooluse_hook(
        policy,
        control={
            "policy_id": "recusal-plugin-deny-list",
            # the shipped policy travels with the package, so its version IS the
            # package version; a custom policy should declare its own
            "policy_version": EXPECTED_RECUSAL_VERSION,
        },
    )
