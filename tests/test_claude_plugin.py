"""The recusal-gate Claude Code plugin must stay in lockstep with the package.

Three drift locks and an end-to-end run:
- the plugin's launcher is the canonical fail-closed launcher, path-swapped for
  ``${CLAUDE_PLUGIN_ROOT}``, byte-for-byte;
- the plugin gate shim carries the same policy wiring the scaffolder emits;
- every version surface (pyproject, ``recusal.__version__``, plugin.json,
  marketplace.json) agrees;
- the plugin's gate script, run exactly as the launcher runs it, refuses a
  destructive call and defers a safe one.
"""

import json
import os
import re
import subprocess
import sys

import recusal
from recusal.__main__ import LAUNCHER_COMMAND

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_DIR = os.path.join(REPO_ROOT, "claude-plugin")
GATE_SCRIPT = os.path.join(PLUGIN_DIR, "scripts", "recusal_gate.py")


def _load(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


def _run_gate(event, extra_env=None):
    env = {**os.environ, "PYTHONPATH": REPO_ROOT, **(extra_env or {})}
    stdin = event if isinstance(event, str) else json.dumps(event)
    return subprocess.run(
        [sys.executable, GATE_SCRIPT],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


# --- drift locks ---------------------------------------------------------------------


def test_plugin_launcher_is_the_canonical_launcher_path_swapped():
    hooks = _load(PLUGIN_DIR, "hooks", "hooks.json")
    groups = hooks["hooks"]["PreToolUse"]
    assert len(groups) == 1 and groups[0]["matcher"] == ".*"
    plugin_cmd = groups[0]["hooks"][0]["command"]
    expected = LAUNCHER_COMMAND.replace(
        '"$CLAUDE_PROJECT_DIR/.claude/hooks/recusal_gate.py"',
        '"${CLAUDE_PLUGIN_ROOT}/scripts/recusal_gate.py"',
    )
    assert plugin_cmd == expected


def test_plugin_manifest_relies_on_the_autoloaded_hooks_file():
    manifest = _load(PLUGIN_DIR, ".claude-plugin", "plugin.json")
    assert manifest["name"] == "recusal-gate"
    # hooks/hooks.json is loaded automatically by convention; naming it AGAIN in
    # manifest.hooks is a duplicate reference and the plugin fails to load
    # (verified live against `claude plugin install`, 2026-07-07)
    assert "hooks" not in manifest
    assert os.path.exists(os.path.join(PLUGIN_DIR, "hooks", "hooks.json"))
    assert (
        "fails closed" in manifest["description"].lower()
        or "FAILS CLOSED" in manifest["description"]
    )


def test_marketplace_lists_the_plugin_by_relative_source():
    market = _load(REPO_ROOT, ".claude-plugin", "marketplace.json")
    entries = {p["name"]: p for p in market["plugins"]}
    assert entries["recusal-gate"]["source"] == "./claude-plugin"


def test_all_version_surfaces_agree():
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), encoding="utf-8") as f:
        pyproject_version = re.search(r'^version = "([^"]+)"', f.read(), re.M).group(1)
    plugin_version = _load(PLUGIN_DIR, ".claude-plugin", "plugin.json")["version"]
    market_version = _load(REPO_ROOT, ".claude-plugin", "marketplace.json")["metadata"]["version"]
    assert pyproject_version == recusal.__version__ == plugin_version == market_version


def test_plugin_gate_wires_the_same_policy_as_the_scaffolder():
    with open(GATE_SCRIPT, encoding="utf-8") as f:
        src = f.read()
    compile(src, GATE_SCRIPT, "exec")
    assert "deny_list_policy()" in src
    assert "run_pretooluse_hook(policy)" in src
    # missing dependency must exit nonzero (launcher coerces to blocking exit 2),
    # never fall through to an ungated pass
    assert "except ImportError" in src and "sys.exit(3)" in src


# --- end to end ------------------------------------------------------------------------


def test_plugin_gate_refuses_destructive_call():
    proc = _run_gate({"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/x"}})
    assert proc.returncode == 0, proc.stderr
    decision = json.loads(proc.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"


def test_plugin_gate_defers_safe_call():
    proc = _run_gate({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


def test_plugin_gate_fails_closed_on_malformed_event():
    proc = _run_gate("garbage, not an event")
    assert proc.returncode == 0, proc.stderr
    decision = json.loads(proc.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
