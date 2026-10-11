"""An empty project directory must be refused by name, before any interpreter starts.

The launcher builds the gate path from ``CLAUDE_PROJECT_DIR`` (``CLAUDE_PLUGIN_ROOT`` for
the plugin). Empty, the path collapses to ``/.claude/hooks/recusal_gate.py``: the root of
the current drive under PowerShell, Git's install folder under Git Bash. Every test here
pairs a refusal with a control that still runs the gate, so a guard that refused
everything would fail too.
"""

import io
import json
import os
import re
import shutil
import string
import subprocess
import sys

import pytest

from recusal.__main__ import (
    _PROBE_PYTHON,
    _PROJECT_GATE,
    GATE_FILENAME,
    LAUNCHER_COMMAND_POSIX,
    LAUNCHER_COMMAND_POWERSHELL,
    LEGACY_LAUNCHERS,
    UNGUARDED_LAUNCHERS,
    _launcher_platform_findings,
    init,
    posix_launcher,
    repair_launcher,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_DIR = os.path.join(REPO_ROOT, "claude-plugin")
DESTRUCTIVE = '{"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/x"}}'
PROJECT_MESSAGE = "CLAUDE_PROJECT_DIR is empty, so the gate script cannot be found"
PLUGIN_MESSAGE = "CLAUDE_PLUGIN_ROOT is empty, so the gate script cannot be found"
_WINDOWS = os.name == "nt"


def _find_bash():
    """A bash that can run the launcher (Git Bash on Windows, never WSL's System32 bash)."""
    candidates = []
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        candidates.append(found)
    if _WINDOWS:
        candidates.append(r"C:\Program Files\Git\bin\bash.exe")
    for c in candidates:
        try:
            if subprocess.run([c, "-c", "exit 0"], capture_output=True, timeout=30).returncode == 0:
                return c
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


_BASH = _find_bash()
needs_bash = pytest.mark.skipif(_BASH is None, reason="no usable POSIX shell for the launcher")
windows_only = pytest.mark.skipif(not _WINDOWS, reason="the PowerShell launcher is Windows-only")


def _bash_path(path):
    path = path.replace("\\", "/")
    if len(path) > 1 and path[1] == ":":
        return "/" + path[0].lower() + path[2:]
    return path


def _env(project_dir="UNSET", plugin_root="UNSET", **extra):
    env = {**os.environ, "PYTHONPATH": REPO_ROOT, **extra}
    for name, value in (("CLAUDE_PROJECT_DIR", project_dir), ("CLAUDE_PLUGIN_ROOT", plugin_root)):
        env.pop(name, None)
        if value != "UNSET":
            env[name] = value
    return env


def _scaffold(tmp_path, name="project"):
    project = tmp_path / name
    project.mkdir()
    assert init(str(project), stdout=io.StringIO()) == 0
    return project


def _logging_shims(tmp_path):
    """python3/python/py that record every invocation, then run the real interpreter."""
    shims = tmp_path / "shims"
    shims.mkdir(parents=True)
    log = tmp_path / "invocations.log"
    exe = sys.executable.replace("\\", "/")
    for name in ("python3", "python", "py"):
        shim = shims / name
        shim.write_text(
            f'#!/bin/sh\necho "{name} $*" >> "{_bash_path(str(log))}"\nexec "{exe}" "$@"\n',
            encoding="utf-8",
        )
        shim.chmod(0o755)
    return str(shims), log


def _run_bash(command, env, shims=None):
    if shims:
        command = f'PATH="{_bash_path(shims)}:/usr/bin:/bin"; {command}'
    return subprocess.run(
        [_BASH, "-c", command],
        input=DESTRUCTIVE,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def _run_powershell(command, env, cwd=None):
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        input=DESTRUCTIVE,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=cwd,
    )


def _plugin_command():
    with open(os.path.join(PLUGIN_DIR, "hooks", "hooks.json"), encoding="utf-8") as fh:
        return json.load(fh)["hooks"]["PreToolUse"][0]["hooks"][0]["command"]


def _assert_denied(proc):
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


# --- the POSIX launcher ------------------------------------------------------------------


@needs_bash
@pytest.mark.parametrize("value", ["UNSET", ""], ids=["unset", "empty"])
def test_posix_launcher_refuses_an_empty_project_dir_before_any_interpreter(tmp_path, value):
    shims, log = _logging_shims(tmp_path)
    proc = _run_bash(LAUNCHER_COMMAND_POSIX, _env(project_dir=value), shims)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert PROJECT_MESSAGE in proc.stderr
    assert proc.stdout == ""
    assert not log.exists(), log.read_text(encoding="utf-8")  # no interpreter ever started


@needs_bash
def test_posix_launcher_control_runs_the_gate_through_the_same_shims(tmp_path):
    # the control for the test above: same shims, a real project dir, the gate runs
    project = _scaffold(tmp_path)
    shims, log = _logging_shims(tmp_path)
    proc = _run_bash(LAUNCHER_COMMAND_POSIX, _env(project_dir=str(project)), shims)
    _assert_denied(proc)
    assert GATE_FILENAME in log.read_text(encoding="utf-8")


@needs_bash
def test_posix_launcher_accepts_a_project_dir_with_spaces(tmp_path):
    project = _scaffold(tmp_path, "my project dir")
    _assert_denied(_run_bash(LAUNCHER_COMMAND_POSIX, _env(project_dir=str(project))))


@needs_bash
def test_the_0_12_0_posix_launcher_gave_no_cause(tmp_path):
    # the defect this fixes, pinned: the old launcher refused but never named the cause
    old = [c for c in UNGUARDED_LAUNCHERS if c.startswith("for p in")][0]
    proc = _run_bash(old, _env(project_dir=""))
    assert proc.returncode == 2
    assert "CLAUDE_PROJECT_DIR" not in proc.stderr
    assert "hook did not run cleanly" in proc.stderr


# --- the plugin launcher -----------------------------------------------------------------


@needs_bash
@pytest.mark.parametrize("value", ["UNSET", ""], ids=["unset", "empty"])
def test_plugin_launcher_refuses_an_empty_plugin_root(tmp_path, value):
    shims, log = _logging_shims(tmp_path)
    proc = _run_bash(_plugin_command(), _env(plugin_root=value), shims)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert PLUGIN_MESSAGE in proc.stderr
    assert not log.exists()


@needs_bash
def test_plugin_launcher_control_runs_the_vendored_gate():
    env = _env(plugin_root=_bash_path(PLUGIN_DIR) if _WINDOWS else PLUGIN_DIR)
    env.pop("PYTHONPATH")  # the plugin carries its own runtime
    _assert_denied(_run_bash(_plugin_command(), env))


# --- the PowerShell launcher -------------------------------------------------------------


@windows_only
@pytest.mark.parametrize("value", ["UNSET", ""], ids=["unset", "empty"])
def test_powershell_launcher_refuses_an_empty_project_dir(value):
    proc = _run_powershell(LAUNCHER_COMMAND_POWERSHELL, _env(project_dir=value))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert PROJECT_MESSAGE in proc.stderr
    assert proc.stdout == ""


@windows_only
def test_powershell_launcher_control_runs_the_gate(tmp_path):
    project = _scaffold(tmp_path, "my project dir")
    _assert_denied(_run_powershell(LAUNCHER_COMMAND_POWERSHELL, _env(project_dir=str(project))))


def _free_drive_letter():
    used = {d[0].upper() for d in os.listdrives()} if hasattr(os, "listdrives") else set()
    for letter in reversed(string.ascii_uppercase[3:]):
        if letter not in used and not os.path.exists(f"{letter}:\\"):
            return letter
    return None


@pytest.fixture
def planted_drive_root(tmp_path):
    """A drive whose root holds a planted ``.claude/hooks/recusal_gate.py`` that ALLOWS.

    ``subst`` maps a temp folder to a free drive letter (no admin rights needed), which
    stands in for the root of a real drive that any signed-in user can write to.
    """
    if not _WINDOWS or shutil.which("subst") is None:
        pytest.skip("needs Windows subst")
    letter = _free_drive_letter()
    if letter is None:
        pytest.skip("no free drive letter")
    root = tmp_path / "driveroot"
    hooks = root / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    (hooks / GATE_FILENAME).write_text("print('PLANTED GATE RAN')\n", encoding="utf-8")
    if subprocess.run(["subst", f"{letter}:", str(root)], capture_output=True).returncode != 0:
        pytest.skip("subst failed")
    try:
        yield f"{letter}:\\"
    finally:
        subprocess.run(["subst", f"{letter}:", "/d"], capture_output=True)


@windows_only
def test_the_0_12_0_powershell_launcher_ran_a_gate_planted_at_the_drive_root(planted_drive_root):
    # the defect, demonstrated: with the variable empty, the old launcher ran whatever
    # gate sat at the root of the current drive, and that gate allowed the call
    old = [c for c in UNGUARDED_LAUNCHERS if c.startswith("$ErrorActionPreference")][0]
    proc = _run_powershell(old, _env(project_dir=""), cwd=planted_drive_root)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "PLANTED GATE RAN" in proc.stdout


@windows_only
def test_powershell_launcher_never_runs_a_gate_planted_at_the_drive_root(planted_drive_root):
    proc = _run_powershell(
        LAUNCHER_COMMAND_POWERSHELL, _env(project_dir=""), cwd=planted_drive_root
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "PLANTED" not in proc.stdout
    assert PROJECT_MESSAGE in proc.stderr


# --- every launcher copy carries the guard -----------------------------------------------

_LAUNCHER_FILES = (
    "README.md",
    "docs/HOWTO.md",
    "docs/EXAMPLE.md",
    "examples/claude_code_gate.py",
    "recusal/claude_code.py",
    ".claude/settings.json.example",
    "claude-plugin/hooks/hooks.json",
)
_COMMAND = re.compile(r'"command":\s*"((?:[^"\\]|\\.)*)"')


def _launcher_copies():
    copies = []
    for rel in _LAUNCHER_FILES:
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as fh:
            for raw in _COMMAND.findall(fh.read()):
                command = json.loads(f'"{raw}"')
                if "for p in python3 python py" in command:
                    copies.append((rel, command))
    return copies


def test_every_launcher_copy_refuses_an_empty_root_first():
    copies = _launcher_copies()
    assert {rel for rel, _ in copies} == set(_LAUNCHER_FILES)  # none silently skipped
    for rel, command in copies:
        root = re.search(r'"\$p" "(\$\{?\w+\}?)/', command).group(1)
        name = root.strip("${}")
        guard = f'[ -n "{root}" ] || {{ echo \''
        assert command.startswith(guard), (rel, command[:80])
        assert (
            f"{name} is empty, so the gate script cannot be found; failing closed' >&2; exit 2; }}; for p in"
            in command
        ), rel


@needs_bash
def test_documented_launchers_refuse_an_empty_project_dir(tmp_path):
    for rel, command in _launcher_copies():
        if "CLAUDE_PROJECT_DIR" not in command:
            continue
        shims, log = _logging_shims(tmp_path / rel.replace("/", "_"))
        proc = _run_bash(command, _env(project_dir=""), shims)
        assert proc.returncode == 2, (rel, proc.returncode, proc.stderr)
        assert PROJECT_MESSAGE in proc.stderr, rel
        assert not log.exists(), rel


def test_posix_launcher_refuses_a_root_the_gate_path_does_not_use():
    with pytest.raises(ValueError):
        posix_launcher('"${CLAUDE_PLUGIN_ROOT}/scripts/recusal_gate.py"', _PROBE_PYTHON)
    with pytest.raises(ValueError):
        posix_launcher(_PROJECT_GATE, root="${CLAUDE_PLUGIN_ROOT}")


# --- doctor and --repair-launcher know the unguarded launchers ----------------------------


def _hook(command, **extra):
    return {"type": "command", "command": command, **extra}


def test_unguarded_launchers_are_the_current_ones_without_the_guard():
    assert UNGUARDED_LAUNCHERS == {
        LAUNCHER_COMMAND_POSIX.split(" exit 2; }; ", 1)[1],
        LAUNCHER_COMMAND_POWERSHELL.split(" }; ", 1)[1],
    }
    assert not UNGUARDED_LAUNCHERS & LEGACY_LAUNCHERS


@pytest.mark.parametrize("kind", ["posix", "powershell"])
def test_doctor_names_an_unguarded_launcher_as_outdated(kind):
    command = [c for c in UNGUARDED_LAUNCHERS if ("foreach" in c) == (kind == "powershell")][0]
    shell = {"shell": "powershell"} if kind == "powershell" else {}
    findings = _launcher_platform_findings([_hook(command, **shell)])
    assert any(
        not f.passed and "0.10.1 through 0.12.0" in f.message and "--repair-launcher" in f.message
        for f in findings
    )


@pytest.mark.parametrize("kind", ["posix", "powershell"])
def test_doctor_does_not_flag_the_current_launcher(kind):
    command = LAUNCHER_COMMAND_POWERSHELL if kind == "powershell" else LAUNCHER_COMMAND_POSIX
    shell = {"shell": "powershell"} if kind == "powershell" else {}
    findings = _launcher_platform_findings([_hook(command, **shell)])
    assert not any("0.12.0" in f.message for f in findings)


@pytest.mark.parametrize("kind", ["posix", "powershell"])
def test_repair_launcher_upgrades_an_unguarded_launcher(tmp_path, kind):
    old = [c for c in UNGUARDED_LAUNCHERS if ("foreach" in c) == (kind == "powershell")][0]
    hook = _hook(old, **({"shell": "powershell"} if kind == "powershell" else {}))
    custom = _hook("echo my-own-hook")
    settings = {
        "model": "opus",
        "hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [hook, custom]}]},
    }
    (tmp_path / ".claude").mkdir()
    path = tmp_path / ".claude" / "settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    assert repair_launcher(str(tmp_path), launcher=kind, stdout=io.StringIO()) == 0
    after = json.loads(path.read_text(encoding="utf-8"))
    commands = [h["command"] for g in after["hooks"]["PreToolUse"] for h in g["hooks"]]
    expected = LAUNCHER_COMMAND_POWERSHELL if kind == "powershell" else LAUNCHER_COMMAND_POSIX
    assert expected in commands and old not in commands
    assert "echo my-own-hook" in commands
    assert after["model"] == "opus"
    out = io.StringIO()
    assert repair_launcher(str(tmp_path), launcher=kind, stdout=out) == 0
    assert "no change" in out.getvalue()
