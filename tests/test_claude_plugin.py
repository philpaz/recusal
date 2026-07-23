"""The recusal-gate Claude Code plugin must stay in lockstep with the package.

Drift locks and end-to-end runs:
- the plugin's launcher is the canonical fail-closed launcher, path-swapped for
  ``${CLAUDE_PLUGIN_ROOT}``, byte-for-byte;
- the plugin gate shim carries the same policy wiring the scaffolder emits;
- every version surface (pyproject, ``recusal.__version__``, plugin.json,
  marketplace.json) agrees;
- the VENDORED runtime (``claude-plugin/vendor/recusal``) is byte-identical to the
  package source, file for file - the plugin IS the implementation, and
  ``py tools/vendor_plugin.py`` is the one way to re-sync;
- the plugin's gate script, run exactly as the launcher runs it and WITHOUT any
  ambient recusal on PYTHONPATH, refuses a destructive call and defers a safe one;
- a missing vendor tree fails closed, and an import that resolves to an ambient
  package instead of the vendored copy is refused as substitution.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import recusal
from recusal.__main__ import LAUNCHER_COMMAND

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_DIR = os.path.join(REPO_ROOT, "claude-plugin")
GATE_SCRIPT = os.path.join(PLUGIN_DIR, "scripts", "recusal_gate.py")
VENDOR_DIR = os.path.join(PLUGIN_DIR, "vendor", "recusal")
SOURCE_DIR = os.path.join(REPO_ROOT, "recusal")


def _load(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


def _bare_env(extra_env=None):
    # no PYTHONPATH: the vendored runtime must carry the gate on its own
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(extra_env or {})
    return env


def _run_gate(event, extra_env=None, script=GATE_SCRIPT):
    stdin = event if isinstance(event, str) else json.dumps(event)
    return subprocess.run(
        [sys.executable, script],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
        env=_bare_env(extra_env),
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
    # ONE Python source of truth (recusal/__init__.py, read by hatch); pyproject must
    # stay dynamic so a literal can never drift back in. JSON distribution surfaces
    # keep literals (JSON cannot import Python) and are drift-locked here.
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), encoding="utf-8") as f:
        pyproject = f.read()
    assert re.search(r'^version = "', pyproject, re.M) is None  # no literal allowed
    assert 'dynamic = ["version"]' in pyproject
    assert 'path = "recusal/__init__.py"' in pyproject
    plugin_version = _load(PLUGIN_DIR, ".claude-plugin", "plugin.json")["version"]
    market_version = _load(REPO_ROOT, ".claude-plugin", "marketplace.json")["metadata"]["version"]
    assert recusal.__version__ == plugin_version == market_version


def test_vendored_runtime_is_byte_identical_to_the_package():
    # The plugin's claim is "plugin = implementation"; it holds only while the vendor
    # tree is a byte-for-byte copy of the package source. Re-sync with:
    #     py tools/vendor_plugin.py
    expected = sorted(
        name for name in os.listdir(SOURCE_DIR) if name.endswith(".py") or name == "py.typed"
    )
    vendored = sorted(name for name in os.listdir(VENDOR_DIR) if name != "__pycache__")
    assert vendored == expected, (
        "vendored file set differs from recusal/ source; run `py tools/vendor_plugin.py`"
    )
    for name in expected:
        with open(os.path.join(SOURCE_DIR, name), "rb") as f:
            source_bytes = f.read()
        with open(os.path.join(VENDOR_DIR, name), "rb") as f:
            vendor_bytes = f.read()
        assert source_bytes == vendor_bytes, (
            f"claude-plugin/vendor/recusal/{name} differs from recusal/{name}; "
            "run `py tools/vendor_plugin.py`"
        )


def test_the_vendor_sync_tool_is_deterministic_and_matches_the_lock():
    import tools.vendor_plugin as vendor_plugin

    expected = sorted(
        name for name in os.listdir(SOURCE_DIR) if name.endswith(".py") or name == "py.typed"
    )
    assert vendor_plugin.vendored_names() == expected


def test_onboarding_version_pins_track_the_package_version():
    # Version-bound onboarding surfaces are not narrative: a stale exact pin makes a
    # user following the docs exactly install a wrong combination. Shipped stale once
    # (0.5.12 README pinned 0.5.11); locked here since. The plugin no longer needs a
    # paired `pip install "recusal==X"` line (it vendors its runtime), so an exact
    # install pin is no longer REQUIRED in the README - but any that appears, and
    # every Action usage example, must match the package version. Historical mentions
    # ("Since 0.5.11 ...", CHANGELOG, PROVEN) are exempt: this locks only the exact
    # pin spellings.
    with open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    with open(os.path.join(REPO_ROOT, "action.yml"), encoding="utf-8") as f:
        action = f.read()
    install_pins = re.findall(r'recusal==([\d.]+)"', readme)
    action_pins = re.findall(r"philpaz/recusal@v([\d.]+)", readme + action)
    assert action_pins, "README/action.yml lost the Action usage example"
    for pin in install_pins + action_pins:
        assert pin == recusal.__version__, (
            f"onboarding pin {pin} != package {recusal.__version__}; a user following "
            "the docs exactly would install a wrong combination"
        )


def test_plugin_gate_wires_the_same_policy_as_the_scaffolder():
    with open(GATE_SCRIPT, encoding="utf-8") as f:
        src = f.read()
    compile(src, GATE_SCRIPT, "exec")
    assert "deny_list_policy()" in src
    assert re.search(r"run_pretooluse_hook\(\s*policy", src)  # control args allowed
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


def test_plugin_version_is_bound_to_the_adjudicator_version():
    # P0: a deterministic control must be identifiable. The plugin's EXPECTED version,
    # the plugin manifest version, and the package version must all agree, so the
    # installed plugin identity names the exact implementation that decides.
    import json
    import re

    import recusal

    with open(GATE_SCRIPT, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r'EXPECTED_RECUSAL_VERSION = "([^"]+)"', src)
    assert m, "the plugin gate must declare EXPECTED_RECUSAL_VERSION"
    assert m.group(1) == recusal.__version__
    plugin_dir = os.path.dirname(os.path.dirname(GATE_SCRIPT))
    with open(os.path.join(plugin_dir, ".claude-plugin", "plugin.json"), encoding="utf-8") as f:
        assert json.load(f)["version"] == recusal.__version__


def _copy_plugin(tmp_path):
    clone = os.path.join(str(tmp_path), "plugin")
    shutil.copytree(PLUGIN_DIR, clone, ignore=shutil.ignore_patterns("__pycache__"))
    return clone


def test_plugin_gate_refuses_a_mismatched_adjudicator(tmp_path):
    # Plugin X adjudicating with vendored recusal Y would make the audit trail lie
    # about what decided (a partially updated plugin); the shim must fail closed
    # (nonzero -> launcher coerces to blocking 2).
    clone = _copy_plugin(tmp_path)
    shim = os.path.join(clone, "scripts", "recusal_gate.py")
    src = open(shim, encoding="utf-8").read()
    patched = re.sub(
        r'EXPECTED_RECUSAL_VERSION = "[^"]+"',
        'EXPECTED_RECUSAL_VERSION = "9.9.9-mismatch"',
        src,
        count=1,
    )
    open(shim, "w", encoding="utf-8").write(patched)
    proc = _run_gate({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}, script=shim)
    assert proc.returncode == 3
    assert "mismatched identity" in proc.stderr


def test_plugin_gate_refuses_a_substituted_runtime(tmp_path):
    # The substitution attack the vendoring exists to close: vendor tree gone, an
    # ambient recusal importable on PYTHONPATH. Adjudicating with it would break the
    # plugin-identity-names-the-implementation claim; the shim refuses instead.
    clone = _copy_plugin(tmp_path)
    shutil.rmtree(os.path.join(clone, "vendor"))
    proc = _run_gate(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        extra_env={"PYTHONPATH": REPO_ROOT},
        script=os.path.join(clone, "scripts", "recusal_gate.py"),
    )
    assert proc.returncode == 3
    assert "not the plugin's vendored runtime" in proc.stderr


def test_plugin_gate_fails_closed_without_any_runtime(tmp_path):
    # No vendor tree and no ambient package on PYTHONPATH: every tool call refused,
    # never a silently absent gate. (An ambient site-packages recusal, if one exists
    # in the environment, is refused as substitution: still exit 3, still vendored.)
    clone = _copy_plugin(tmp_path)
    shutil.rmtree(os.path.join(clone, "vendor"))
    proc = _run_gate(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}},
        script=os.path.join(clone, "scripts", "recusal_gate.py"),
    )
    assert proc.returncode == 3
    assert "vendored" in proc.stderr
