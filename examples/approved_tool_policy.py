"""Recipe 9: only explicitly approved tool names defer to other controls.

Names are exact and case-sensitive. Approval of a name does not validate its
arguments: combine this with argument-level policies, especially for Bash.
"""

from recusal import Finding

APPROVED = frozenset({"Read", "Grep", "Glob", "Bash", "update_record"})


def policy(tool_name, tool_input):
    if not isinstance(tool_name, str) or tool_name not in APPROVED:
        return [
            Finding.fail(
                "tool_allowlist",
                severity="CRITICAL",
                message="missing, invalid or unapproved tool name",
            )
        ]
    return []
