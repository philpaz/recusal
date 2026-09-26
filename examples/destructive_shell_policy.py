"""Recipe 1: reuse Recusal's reference deny list, then add local shell markers.

A deny list cannot see commands built at runtime or code inside an interpreter.
Use cookbook recipe 11's allowlist mode for a default-deny boundary. Custom
markers are normalized substring tripwires, not shell parsing. The reference
policy also protects secret files and control paths; those checks remain active.
"""

from recusal import Finding
from recusal.deny_list import deny_list_policy


def make_policy(extra_markers=()):
    markers = []
    for marker in extra_markers:
        if not isinstance(marker, str) or not marker.strip():
            raise ValueError("extra markers must be nonempty strings")
        markers.append(" ".join(marker.casefold().split()))
    reference = deny_list_policy()

    def policy(tool_name, tool_input):
        findings = reference(tool_name, tool_input)
        command = tool_input.get("command")
        if tool_name == "Bash" and isinstance(command, str):
            normalized = " ".join(command.casefold().split())
            if any(marker in normalized for marker in markers):
                findings.append(
                    Finding.fail(
                        "destructive_shell",
                        severity="CRITICAL",
                        message="command matches a local deny-list marker",
                    )
                )
        return findings

    return policy


policy = make_policy()
