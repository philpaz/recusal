"""
MCP tool calls through the same gate, no MCP-specific adapter needed.

In Claude Code, MCP server tools arrive at the ``PreToolUse`` hook as ordinary tools
named ``mcp__<server>__<tool>`` (``mcp__github__create_issue``,
``mcp__filesystem__write_file``); the hooks reference states they "appear as regular
tools in tool events" (code.claude.com/docs/en/hooks; per-server matchers like
``mcp__github__.*`` are also documented). So the shipped ``.*`` matcher routes every MCP
call through the same ``policy(tool_name, tool_input)`` seam that vets ``Bash``. This file
shows the MCP-shaped rules: pin the servers you expect, refuse destructive verbs, scope
the rest, and (part two) allowlist mode, where an MCP tool is refused unless
affirmatively named.

This file is both a demo and a real policy:

    python examples/mcp_governance.py      # prints the call-by-call verdicts

    # or wire it as a Claude Code PreToolUse hook (absolute path in .claude/settings.json).
    # Use the interpreter-probe launcher, not a bare python3, so a missing interpreter fails
    # CLOSED (a hook that can't launch is a non-blocking error in Claude Code -> fail open):
    #   { "type": "command",
    #     "command": "for p in python3 python py; do \"$p\" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null && { \"$p\" .../examples/mcp_governance.py --hook; rc=$?; [ \"$rc\" = 0 ] || exit 2; exit 0; }; done; exit 2" }
    # The --hook flag runs the gate against a real PreToolUse event instead of the demo.

The boundary is honest: this is *call-time* governance, the policy sees the proposed tool
name and arguments. It does not see what the server declared at discovery (identity, tool
descriptions, schemas, ``tools/list`` changes), and a poisoned tool *description* can steer
the model before any call is proposed. Screen what an MCP tool *returns* with cookbook
recipe 6 (injection quarantine); govern the catalog itself with ``recusal mcp pin`` /
``recusal mcp verify`` and ``recusal.mcp.manifest_policy`` (the discovery boundary; see
``examples/mcp_manifest_rugpull.py``).

In a custom Agent SDK or MCP-client loop nothing intercepts for you: call the gate between
the model's proposed MCP call and the client dispatching it (``recusal.claude.gate_tool_use``
is the same seam).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding  # noqa: E402
from recusal.claude_code import allowlist_policy, decide, run_pretooluse_hook  # noqa: E402

# The MCP servers this deployment expects. An unexpected server name at call time is the
# cheapest supply-chain tripwire you can buy: a tool from a server you never installed
# should refuse, not run.
APPROVED_SERVERS = frozenset({"github", "salesforce", "filesystem"})
APPROVED_REPOS = frozenset({"philpaz/recusal"})
WORKSPACE = os.path.abspath("./workspace")

# First verb of the MCP action name; tune to your servers (a server whose "remove_label"
# is benign wants that call vetted by an explicit rule instead).
_DESTRUCTIVE_VERBS = frozenset({"delete", "drop", "truncate", "remove", "destroy"})


def _mcp(tool_name: str):
    """``mcp__github__create_issue`` -> ``("github", "create_issue")``, else None."""
    parts = tool_name.split("__", 2)
    return (parts[1], parts[2]) if len(parts) == 3 and parts[0] == "mcp" else None


def _under(root: str, path: str) -> bool:
    # realpath both sides so a symlink under the root cannot point the write outside it.
    try:
        root_real = os.path.realpath(root)
        return os.path.commonpath([root_real, os.path.realpath(os.path.abspath(path))]) == root_real
    except ValueError:  # different drives on Windows
        return False


def mcp_policy(tool_name: str, tool_input: dict) -> list:
    named = _mcp(tool_name)
    if named is None:
        return []  # not an MCP call -> your native-tool policy's job (compose them)
    server, action = named
    if server not in APPROVED_SERVERS:
        return [
            Finding.fail(
                "mcp_unapproved_server",
                severity="CRITICAL",
                message=f"MCP server '{server}' is not on the approved list",
                tool=tool_name,
            )
        ]
    if action.split("_", 1)[0] in _DESTRUCTIVE_VERBS:
        return [
            Finding.fail(
                "mcp_destructive_action",
                severity="CRITICAL",
                message=f"destructive MCP action `{tool_name}` is not approved",
                tool=tool_name,
            )
        ]
    if tool_name == "mcp__github__merge_pull_request":
        repo = tool_input.get("repo")
        if repo not in APPROVED_REPOS:
            return [
                Finding.fail(
                    "mcp_repository_scope",
                    severity="CRITICAL",
                    message=f"repository {repo!r} is outside the approved scope",
                    tool=tool_name,
                )
            ]
    if tool_name == "mcp__filesystem__write_file":
        path = str(tool_input.get("path", ""))
        if not (path and _under(WORKSPACE, path)):
            return [
                Finding.fail(
                    "mcp_write_confinement",
                    severity="CRITICAL",
                    message=f"MCP write path {path!r} is outside the approved workspace",
                    tool=tool_name,
                )
            ]
    return []


# The calls the demo adjudicates: (label, tool_name, tool_input).
_CALLS = [
    ("create an issue", "mcp__github__create_issue", {"repo": "philpaz/recusal", "title": "x"}),
    ("bulk Salesforce delete", "mcp__salesforce__delete_records", {"object": "Contact"}),
    ("merge PR, approved repo", "mcp__github__merge_pull_request", {"repo": "philpaz/recusal"}),
    ("merge PR, foreign repo", "mcp__github__merge_pull_request", {"repo": "attacker/repo"}),
    ("write inside workspace", "mcp__filesystem__write_file", {"path": "workspace/notes.md"}),
    ("write outside workspace", "mcp__filesystem__write_file", {"path": "/etc/passwd"}),
    ("tool from unknown server", "mcp__pastebin__upload", {"content": "..."}),
    ("native Read (not MCP)", "Read", {"file_path": "README.md"}),
]


def main() -> None:
    print("RECUSAL - MCP tool governance (offline)\n")
    print("  MCP server tools reach the PreToolUse hook as ordinary tools named")
    print("  mcp__<server>__<tool>, so the same gate that vets Bash vets them, no")
    print("  MCP-specific adapter, no extra wiring.\n")
    print(f"  {'proposed call':<26}{'tool_name':<36}verdict")
    print("  " + "-" * 70)
    for label, tool, tool_input in _CALLS:
        verdict, reason = decide(tool, tool_input, mcp_policy)
        note = "" if verdict == "defer" else f"  {reason.split(']: ', 1)[-1]}"
        print(f"  {label:<26}{tool:<36}{verdict.upper()}{note}")
    print("\n  DEFER = the gate had no opinion (Claude Code's own permission flow still")
    print("  runs); DENY = refused before the MCP server ever sees the call.\n")

    print("  Allowlist mode (default-deny) already covers MCP: an MCP tool is refused")
    print("  unless affirmatively named with an `allow=` predicate.\n")
    vet = {"mcp__github__create_issue": lambda i: i.get("repo") in APPROVED_REPOS}
    strict = allowlist_policy(allow=vet)
    for label, tool, tool_input in [
        ("named + predicate passes", "mcp__github__create_issue", {"repo": "philpaz/recusal"}),
        ("named, predicate refuses", "mcp__github__create_issue", {"repo": "attacker/repo"}),
        ("unlisted MCP tool", "mcp__salesforce__query", {"soql": "SELECT Id FROM Contact"}),
    ]:
        verdict, _ = decide(tool, tool_input, strict)
        print(f"  {label:<26}{tool:<36}{verdict.upper()}")
    print(
        "\n  The boundary, stated plainly: this is CALL-TIME governance (proposed name +\n"
        "  arguments). It does not vet what the server declared at discovery -- identity,\n"
        "  tool descriptions, schemas -- and a poisoned description can steer the model\n"
        "  before any call is proposed. Screen tool OUTPUT with cookbook recipe 6; pin\n"
        "  the catalog itself with `recusal mcp pin` (examples/mcp_manifest_rugpull.py)."
    )


if __name__ == "__main__":
    # `python examples/mcp_governance.py`         -> the demo (default)
    # `python examples/mcp_governance.py --hook`  -> act as a real Claude Code PreToolUse hook
    if "--hook" in sys.argv:
        run_pretooluse_hook(mcp_policy)
    else:
        main()
