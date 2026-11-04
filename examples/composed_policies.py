"""Recipe 10: compose existing offline policy checks without expanding the kernel.

Run from the repository root: python examples/composed_policies.py

This is a teaching example: the deny-list inspects only explicit tool inputs.
It does not sandbox commands, prevent all network access, or grant permission.
An empty finding list defers to the tool host's normal permission handling.
"""

import copy
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.destructive_shell_policy import make_policy as make_destructive_policy  # noqa: E402
from examples.egress_allowlist import make_egress_allowlist  # noqa: E402
from examples.workspace_policy import make_workspace_policy  # noqa: E402
from recusal.claude_code import _findings_of, _identity_getter, decide  # noqa: E402


class ComposedPolicy:
    """Every finding of every policy, in order, with the control identity of the one that
    carries it.

    Findings: concatenated, so the existing Recusal decision fold picks the worst severity.
    Each leaf's return is held to the evaluator's contract first (``None`` or an iterable
    of findings; anything else is a policy error), so a leaf returning ``{}`` or ``""``
    fails closed through the composition exactly as it does alone. A policy error is not
    swallowed: the real adapter's default ``fail_closed=True`` turns it into a denial
    instead of silently dropping a security check.

    Control identity: ``recusal.claude_code.decide`` reads a policy's
    ``get_control_identity()`` for the audit record (``recusal.mcp.manifest_policy`` reports
    the manifest digest this adjudication verified through it) and calls
    ``reset_control_identity()`` before each adjudication. A composition that forwarded
    findings but not these two would keep the verdict and lose the evidence of which
    approved manifest governed it, so both are forwarded here. Exactly one inner policy may
    carry a control identity (a nested composition of this exact class is flattened into its
    leaves first, so a group of plain policies never counts as one; a subclass stays one
    leaf, so an overridden ``__call__`` keeps running): two manifest policies in one gate would
    have two digests and
    one record, so that composition is refused up front rather than recorded ambiguously.
    Connection binding: a leaf bound to one MCP server connection by ``claude_named``
    (examples/openai_agents_mcp_gate.py) carries ``recusal_bound_server``; the composition
    carries the same, so a gate refuses it for any other server exactly as it refuses the
    leaf. Two leaves bound to different servers, or a bound leaf beside a policy that still
    needs binding (one declaring Claude Code's naming domain), are refused up front rather
    than installed half-adapted.

    Input isolation: each leaf receives a deep copy of ``tool_input``, so the composition is
    a set of observations of one invocation and a leaf that mutates its argument affects
    nothing but itself; transformation leaves (a normalizer meant to change what later
    leaves see) are not supported. The supported input is the data adapters hand a policy:
    JSON values and the projections the gates build from them. A value whose
    ``__deepcopy__`` returns itself is outside that boundary; a value that cannot be copied
    raises, and the evaluator fails closed.

    The invariant the tests pin: adding a no-op policy changes neither the verdict, nor the
    control identity, nor the server the composition is bound to; and reordering
    observational leaves changes nothing either.
    """

    def __init__(self, *policies):
        # A nested composition of this exact class is unwrapped into its leaves, in order:
        # a group of plain policies carries no identity of its own, so counting it as a
        # carrier would refuse harmless groups and miscount the real manifest policy beside
        # one. A SUBCLASS is kept whole as one leaf: it may override ``__call__`` to add or
        # change checks, and unwrapping it to its ``.policies`` would run those leaves and
        # silently skip the override, so a refusal (or a raise) it adds would be lost at the
        # composition boundary.
        leaves = []
        for check in policies:
            leaves.extend(check.policies if type(check) is ComposedPolicy else (check,))
        self.policies = tuple(leaves)
        # Carrier discovery is the evaluator's own contract (``_identity_getter``: a
        # ``get_control_identity()`` method, else a ``last_manifest_digest`` attribute), so
        # whatever ``decide`` would record for a leaf on its own, the composition records too.
        # A kept subclass is the one shape judged here instead: every ComposedPolicy has the
        # getter, so it counts only when it carries a manifest policy itself or overrides the
        # getter to report an identity of its own.
        carriers = []
        for p in self.policies:
            if isinstance(p, ComposedPolicy):
                if p._carrier is not None or _overrides_identity(type(p)):
                    carriers.append(p)
            elif _identity_getter(p) is not None:
                carriers.append(p)
        if len(carriers) > 1:
            raise ValueError(
                "compose_policies: more than one policy carries a control identity; one gate "
                "records one manifest digest, so compose one manifest_policy per gate"
            )
        self._carrier = carriers[0] if carriers else None
        servers = []
        for p in self.policies:
            server = getattr(p, "recusal_bound_server", None)
            if server is not None and server not in servers:
                servers.append(server)
        if len(servers) > 1:
            raise ValueError(
                f"compose_policies: policies bound to different servers {servers!r}; one gate "
                "judges one connection, so compose one server's policies per gate"
            )
        self._bound_server = servers[0] if servers else None
        if self._bound_server is not None and self.recusal_name_domain is not None:
            raise ValueError(
                f"compose_policies: a policy bound to server {self._bound_server!r} is composed "
                "with one that judges Claude Code's MCP names and is not bound; bind that "
                "policy with claude_named too, or compose the unbound policies and bind the whole"
            )

    def __call__(self, tool_name, tool_input):
        findings = []
        for check in self.policies:
            # Every leaf judges its own copy of the input: the fold assumes one fixed input
            # for every finding, and a leaf that rewrote a shared dict (a normalizer that
            # turned DELETE into SELECT) changed what every later leaf judged while the
            # caller executed the original. A copy failure raises and so fails closed.
            # Each leaf's return is then held to the evaluator's contract before it is
            # read: extending with a bare {} or "" would turn a buggy leaf into a screen.
            findings.extend(_findings_of(check(tool_name, copy.deepcopy(tool_input))))
        return findings

    @property
    def recusal_name_domain(self):
        """The naming domain of the composition: that of any leaf declaring one (a
        manifest policy judges Claude Code's ``mcp__<server>__<tool>`` names), else None, so
        a gate that hands policies a server's own tool name can refuse the pairing."""
        for check in self.policies:
            domain = getattr(check, "recusal_name_domain", None)
            if domain:
                return domain
        return None

    @property
    def recusal_bound_server(self):
        """The MCP server connection a leaf is bound to (``claude_named``), else None: a
        gate that installs this composition on a server checks it is that one."""
        return self._bound_server

    def get_control_identity(self):
        if self._carrier is None:
            return {}
        getter = _identity_getter(self._carrier)
        return dict(getter()) if getter is not None else {}

    def reset_control_identity(self):
        for check in self.policies:
            reset = getattr(check, "reset_control_identity", None)
            if callable(reset):
                reset()


def _overrides_identity(cls) -> bool:
    """True when a ComposedPolicy subclass reports an identity of its own."""
    return cls.get_control_identity is not ComposedPolicy.get_control_identity


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
