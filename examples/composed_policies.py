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


class ComposedPolicy:
    """Every finding of every policy, in order, with the control identity of the one that
    carries it.

    Findings: concatenated, so the existing Recusal decision fold picks the worst severity.
    A policy error is not swallowed: the real adapter's default ``fail_closed=True`` turns
    it into a denial instead of silently dropping a security check.

    Control identity: ``recusal.claude_code.decide`` reads a policy's
    ``get_control_identity()`` for the audit record (``recusal.mcp.manifest_policy`` reports
    the manifest digest this adjudication verified through it) and calls
    ``reset_control_identity()`` before each adjudication. A composition that forwarded
    findings but not these two would keep the verdict and lose the evidence of which
    approved manifest governed it, so both are forwarded here. Exactly one inner policy may
    carry a control identity (a nested composition is flattened into its leaves first, so a
    group of plain policies never counts as one): two manifest policies in one gate would
    have two digests and
    one record, so that composition is refused up front rather than recorded ambiguously.
    The invariant the tests pin: adding a no-op policy changes neither the verdict nor the
    control identity.
    """

    def __init__(self, *policies):
        # A nested composition is unwrapped into its leaves, in order: a group of plain
        # policies carries no identity of its own, so counting it as a carrier would
        # refuse harmless groups and miscount the real manifest policy beside one.
        leaves = []
        for check in policies:
            leaves.extend(check.policies if isinstance(check, ComposedPolicy) else (check,))
        self.policies = tuple(leaves)
        carriers = [p for p in self.policies if callable(getattr(p, "get_control_identity", None))]
        if len(carriers) > 1:
            raise ValueError(
                "compose_policies: more than one policy carries a control identity; one gate "
                "records one manifest digest, so compose one manifest_policy per gate"
            )
        self._carrier = carriers[0] if carriers else None

    def __call__(self, tool_name, tool_input):
        findings = []
        for check in self.policies:
            findings.extend(check(tool_name, tool_input))
        return findings

    def get_control_identity(self):
        return dict(self._carrier.get_control_identity()) if self._carrier is not None else {}

    def reset_control_identity(self):
        for check in self.policies:
            reset = getattr(check, "reset_control_identity", None)
            if callable(reset):
                reset()


def compose_policies(*policies):
    """Compose policies into one gate; see :class:`ComposedPolicy`."""
    return ComposedPolicy(*policies)


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
