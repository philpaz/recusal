"""Recipe 3: protect secret paths and confine explicit writes to a workspace.

Run: python examples/workspace_policy.py

Relative paths use the hook process's current working directory, not an inferred
agent project directory. The root is captured when the policy is constructed.
realpath resolves existing symlinks, including parents of not-yet-created files;
normcase follows the host's path rules (including drive/case handling on Windows).
Secret matching deliberately remains a substring check: secrets-howto.md is denied.
Both supplied and resolved names below the workspace root are checked, so a
harmless alias cannot hide a secret name. Ancestor directory names are ignored. This is a teaching recipe, not a filesystem sandbox: links can
change after the check, and shell commands or tools other than those below are
not inspected. Use OS-level confinement for protection against such races.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding

PROTECTED = (".env", ".pem", ".key", ".p12", "id_rsa", "credentials", "secrets")


def make_workspace_policy(root):
    """Bind a trusted root; paths in tool input must be nonempty strings."""
    supplied_root = os.path.normcase(os.path.abspath(root))
    safe_root = os.path.normcase(os.path.realpath(root))

    def policy(tool_name, tool_input):
        if tool_name not in ("Write", "Edit", "MultiEdit"):
            return []
        path = tool_input.get("file_path")
        if not isinstance(path, str) or not path or "\x00" in path:
            return [
                Finding.fail(
                    "path_confinement", severity="CRITICAL", message="missing or invalid write path"
                )
            ]
        try:
            resolved = os.path.normcase(os.path.realpath(path))
            inside = os.path.commonpath([safe_root, resolved]) == safe_root
            supplied_name = os.path.relpath(os.path.normcase(os.path.abspath(path)), supplied_root)
            resolved_name = os.path.relpath(resolved, safe_root)
        except (OSError, ValueError):
            return [
                Finding.fail(
                    "path_confinement", severity="CRITICAL", message="cannot resolve write path"
                )
            ]
        if any(
            marker in candidate.lower()
            for candidate in (supplied_name, resolved_name)
            for marker in PROTECTED
        ):
            return [
                Finding.fail(
                    "protected_file",
                    severity="CRITICAL",
                    message="refusing write to a secret/credential path",
                )
            ]
        if not inside:
            return [
                Finding.fail(
                    "path_confinement",
                    severity="CRITICAL",
                    message="write outside the workspace root",
                )
            ]
        return []

    return policy


def main():
    policy = make_workspace_policy("./workspace")
    for path in ("workspace/notes.txt", "workspace/.env", "workspace/../outside.txt"):
        print(path, "DENY" if policy("Write", {"file_path": path}) else "DEFER")


if __name__ == "__main__":
    main()
