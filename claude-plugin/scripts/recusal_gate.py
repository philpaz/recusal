#!/usr/bin/env python3
"""Recusal PreToolUse gate, the plugin edition.

Installed via the recusal-gate Claude Code plugin, every tool call in a session is
adjudicated here BEFORE it runs; a deny holds even under bypassPermissions. This is the
same thin shim the recusal repository dogfoods and ``python -m recusal init`` scaffolds.

**Vendored runtime**: the plugin ships the exact recusal implementation it executes
(``vendor/recusal``, byte-identical to the released package source, drift-locked in the
repository), so installing the plugin needs no ``pip install`` and the installed plugin
identity IS the implementation that decides. The shim refuses to adjudicate with
anything else: an import that resolves outside the plugin's own vendor tree (an
ambient site-packages recusal, a PYTHONPATH copy) fails CLOSED naming the
substitution, and a missing or unimportable vendor tree fails CLOSED too - every tool
call refused, never a silently absent or silently swapped gate.

**Declared-version binding**: the vendored runtime's ``__version__`` must equal
``EXPECTED_RECUSAL_VERSION`` below; a mismatch means a partially updated or hand-edited
plugin and fails CLOSED with both versions named (the audit trail must name what
decided). Stated precisely: this binds plugin identity to the vendored DECLARED
version and origin. It does not attest file bytes at rest - a modified vendored module
that retains the version string still runs; the byte-identity claim is enforced in the
repository by the drift-lock test, and the installed plugin directory should be
write-protected like any control-plane file.

To customize the policy, scaffold a project-local gate instead (``python -m recusal
init``) and edit ``.claude/hooks/recusal_gate.py``; a project-local deny wins alongside
this plugin (hooks compose, deny-wins).
"""

import os
import sys

#: The exact adjudicator this plugin version identifies. A version-drift-lock test keeps
#: this equal to the plugin manifest version and the package version.
EXPECTED_RECUSAL_VERSION = "0.7.0"

#: The plugin's own runtime. CLAUDE_PLUGIN_ROOT is how Claude Code addresses the
#: installed plugin; the __file__-relative fallback serves direct invocation (tests,
#: a repository checkout). Inserted at position 0 so the vendored copy, when present,
#: always wins the import - and the provenance check below refuses anything else.
_PLUGIN_ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
_VENDOR = os.path.join(_PLUGIN_ROOT, "vendor")
sys.path.insert(0, _VENDOR)

try:
    import recusal
    from recusal.claude_code import run_pretooluse_hook
    from recusal.deny_list import deny_list_policy
except ImportError:
    sys.stderr.write(
        "recusal gate (plugin): the plugin's vendored recusal runtime is missing or "
        "unimportable; failing closed. Fix: reinstall or update the recusal-gate "
        "plugin.\n"
    )
    sys.exit(3)  # any nonzero exit -> the registered launcher coerces to blocking exit 2

_vendor_prefix = os.path.normcase(os.path.realpath(_VENDOR)) + os.sep
_loaded_from = os.path.normcase(os.path.realpath(getattr(recusal, "__file__", "") or ""))
if not _loaded_from.startswith(_vendor_prefix):
    sys.stderr.write(
        f"recusal gate (plugin): import resolved to {getattr(recusal, '__file__', None)!r}, "
        "not the plugin's vendored runtime; refusing to adjudicate with substituted "
        "code (the plugin identity must name the implementation that decides). Fix: "
        "reinstall or update the recusal-gate plugin.\n"
    )
    sys.exit(3)  # coerced to blocking exit 2 by the launcher: fail closed, named

if getattr(recusal, "__version__", None) != EXPECTED_RECUSAL_VERSION:
    sys.stderr.write(
        f"recusal gate (plugin): plugin {EXPECTED_RECUSAL_VERSION} found vendored recusal "
        f"{getattr(recusal, '__version__', 'unknown')}; refusing to adjudicate under a "
        "mismatched identity (a partially updated or hand-edited plugin). Fix: "
        "reinstall or update the recusal-gate plugin.\n"
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
            # the shipped policy travels with the vendored runtime, so its version IS
            # the plugin/package version; a custom policy should declare its own
            "policy_version": EXPECTED_RECUSAL_VERSION,
        },
    )
