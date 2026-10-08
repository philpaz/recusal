"""Recipe 10: compose existing offline policy checks without expanding the kernel.

Run from the repository root: python examples/composed_policies.py

This is a teaching example: the deny-list inspects only explicit tool inputs.
It does not sandbox commands, prevent all network access, or grant permission.
An empty finding list defers to the tool host's normal permission handling.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.destructive_shell_policy import make_policy as make_destructive_policy  # noqa: E402
from examples.egress_allowlist import make_egress_allowlist  # noqa: E402
from examples.workspace_policy import make_workspace_policy  # noqa: E402
from recusal.claude_code import decide  # noqa: E402


def compose_policies(*policies):
    """Keep every finding; let the existing Recusal decision fold pick severity.

    Do not swallow a policy error: the real adapter's default fail_closed=True
    turns it into a denial instead of silently dropping a security check.
    """

    def policy(tool_name, tool_input):
        findings = []
        for check in policies:
            findings.extend(check(tool_name, tool_input))
        return findings

    return policy


def make_policy(workspace_root, allowed_domains=("example.com",)):
    """Combine existing shell, path and explicit egress checks into one gate."""
    return compose_policies(
        make_destructive_policy(),
        make_workspace_policy(workspace_root),
        make_egress_allowlist(allowed_domains),
    )


def main():
    # Synthetic, offline inputs; no tool is invoked and no network call is made.
    with tempfile.TemporaryDirectory(prefix="recusal-composed-") as temporary:
        workspace = Path(temporary) / "workspace"
        workspace.mkdir()
        policy = make_policy(workspace, {"example.com"})
        cases = (
            ("clean read", "Read", {"file_path": str(workspace / "notes.txt")}),
            ("destructive shell", "Bash", {"command": "rm -rf /"}),
            (
                "out-of-workspace write",
                "Write",
                {"file_path": str(workspace.parent / "outside.txt")},
            ),
            ("unapproved destination", "http_post", {"url": "https://evil.example/events"}),
        )
        for label, tool, arguments in cases:
            decision, _reason = decide(tool, arguments, policy)
            print("{}: {}".format(label, decision))


if __name__ == "__main__":
    main()
