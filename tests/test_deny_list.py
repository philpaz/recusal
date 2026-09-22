"""Unit tests for the extracted, importable deny-list engine (``recusal.deny_list``).

The dogfood hook is a thin shim over this module (proven end-to-end in test_dogfood.py);
the obfuscation/self-protect coverage is exercised in test_dogfood_redteam.py and
test_subversion_hook.py, which now import this module directly. Here we pin the module's
own public API: ``analyze_command`` as a pure function, and ``deny_list_policy``'s
parameterization (the capability the extraction adds over a copy-paste script)."""

import pytest

from recusal import Decision, Finding, compute_verdict
from recusal.claude_code import decide
from recusal.deny_list import (
    analyze_command,
    deny_list_policy,
)


def _decision(policy, tool, tool_input):
    return decide(tool, tool_input, policy)[0]


# --- analyze_command: pure function, Finding-shaped -------------------------------------


def test_analyze_command_returns_findings_for_destructive():
    findings = analyze_command("rm -rf /")
    assert findings and all(isinstance(f, Finding) for f in findings)
    assert compute_verdict(findings).decision is Decision.FAIL


def test_analyze_command_empty_for_benign():
    assert analyze_command("echo hello && ls -la") == []


def test_analyze_command_length_capped():
    findings = analyze_command("a" * 5000)
    assert findings and findings[0].check == "command_too_long"


def test_analyze_command_honors_custom_protected_paths():
    # A write verb aimed at a caller-supplied control path is refused via the substring
    # check; an ordinary path outside the custom set (and not one of the universal control
    # dirs) is not. (The .claude/.git/recusal directory-op guard is always on, independent
    # of protected_paths, since those dirs are universal to a Claude Code adopter.)
    assert analyze_command("rm .mygate/config.yaml", protected_paths=(".mygate/",))
    assert analyze_command("rm workspace/config.yaml", protected_paths=(".mygate/",)) == []


# --- deny_list_policy: default behavior --------------------------------------------------


def test_default_policy_denies_destructive_defers_benign():
    policy = deny_list_policy()
    assert _decision(policy, "Bash", {"command": "rm -rf build"}) == "deny"
    assert _decision(policy, "Bash", {"command": "pytest -q"}) == "defer"


def test_default_policy_guards_its_kill_switch():
    policy = deny_list_policy()
    assert _decision(policy, "Edit", {"file_path": "/repo/.claude/settings.json"}) == "deny"
    assert _decision(policy, "Bash", {"command": "cd .claude && rm settings.json"}) == "deny"


def test_default_policy_refuses_secret_write():
    policy = deny_list_policy()
    assert _decision(policy, "Write", {"file_path": "/app/.env"}) == "deny"
    assert _decision(policy, "Write", {"file_path": "/app/server.pem"}) == "deny"


# --- deny_list_policy: parameterization (the new capability) ----------------------------


def test_policy_protects_a_custom_control_path():
    policy = deny_list_policy(protected_paths=(".mygate/", ".git/hooks"))
    # The caller's own gate dir is protected...
    assert _decision(policy, "Edit", {"file_path": "/repo/.mygate/rules.py"}) == "deny"
    assert _decision(policy, "Bash", {"command": "echo x > .mygate/rules.py"}) == "deny"
    # ...and a path only in the DEFAULT set is not, under this custom config.
    assert _decision(policy, "Edit", {"file_path": "/repo/recusal/evidence.py"}) == "defer"


def test_policy_honors_custom_secret_basenames():
    policy = deny_list_policy(secret_basenames={"vault.txt"})
    assert _decision(policy, "Write", {"file_path": "/app/vault.txt"}) == "deny"


def test_policy_honors_custom_read_only_tools():
    # A tool marked read-only is exempt from the generic kill-switch guard.
    policy = deny_list_policy(read_only_tools={"myreader"})
    assert _decision(policy, "myreader", {"path": ".claude/settings.json"}) == "defer"


def test_the_subagent_tool_is_read_only_under_its_current_and_former_names():
    # Claude Code's subagent tool is `Agent` (formerly `Task`). Spawning a subagent
    # touches no file itself; each of the subagent's own tool calls reaches the hook
    # separately, so a prompt that MENTIONS a protected path is not a write to it.
    policy = deny_list_policy()
    prompt = {"prompt": "Review .claude/settings.json and report", "description": "review"}
    assert _decision(policy, "Agent", prompt) == "defer"
    assert _decision(policy, "Task", prompt) == "defer"


FORCE_PUSHES = (
    "git push --force",
    "git push -f origin main",
    "git push origin main --force",  # the flag after the remote
    "git push origin -f",
    "git -C . push --force",  # git global options before the verb
    "git push --force-with-lease",
    "git push origin main --force-if-includes",
    "git push origin +main",  # force via +refspec
    "git push origin main -uf",  # combined short flags
    "GIT PUSH origin main --FORCE",
)
ORDINARY_GIT = (
    "git push origin main",
    "git push -u origin feature",
    "git push --follow-tags origin main",
    "git push -o ci.skip origin main",
    "git push --dry-run origin main",
    "git push origin feature-force",  # a branch name, not a flag
    "git fetch -f",  # force on another verb
    "git push origin main && rm -f /tmp/x.log",  # -f belongs to the next command
    "git push origin main | grep -f pats",
)


@pytest.mark.parametrize("command", FORCE_PUSHES)
def test_every_force_push_form_is_refused(command):
    assert _decision(deny_list_policy(), "Bash", {"command": command}) == "deny"


@pytest.mark.parametrize("command", ORDINARY_GIT)
def test_ordinary_git_pushes_are_not_mistaken_for_force(command):
    assert _decision(deny_list_policy(), "Bash", {"command": command}) == "defer"


def test_policy_analyzes_commands_under_custom_keys():
    policy = deny_list_policy(command_keys={"run"})
    assert _decision(policy, "mcp__runner", {"run": "rm -rf /repo"}) == "deny"


def test_default_policy_protects_the_mcp_control_plane():
    # P1-3: `.mcp.json` decides which server processes launch; `mcp-manifest.json` is the
    # approved truth manifest_policy reloads at call time. An agent that can rewrite
    # either changes what "approved" means before its next tool call, so both are
    # kill-switch-rank paths in the DEFAULT set.
    policy = deny_list_policy()
    assert _decision(policy, "Edit", {"file_path": "/repo/.mcp.json"}) == "deny"
    assert _decision(policy, "Write", {"file_path": "/repo/mcp-manifest.json"}) == "deny"
    assert _decision(policy, "Bash", {"command": "echo x > .mcp.json"}) == "deny"
    assert _decision(policy, "Bash", {"command": "rm mcp-manifest.json"}) == "deny"
    # an MCP filesystem tool is a side channel, not an exemption
    assert _decision(policy, "mcp__fs__write_file", {"path": "/repo/mcp-manifest.json"}) == "deny"
    # reading the control plane stays deferred (read-only tools are exempt from the guard)
    assert _decision(policy, "Read", {"file_path": "/repo/.mcp.json"}) == "defer"
