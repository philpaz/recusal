"""The doctor judges whether the registration lets the gate decide (0.12.4).

Before this, ``doctor`` passed a gate registered as a background hook, under a matcher
naming no tool, with every hook disabled, or without ``"onFailure": "block"``: it only
checked that a PreToolUse command named the gate file and contained ``exit 2``. These
tests scaffold a real install with ``init``, alter one thing at a time, and read the
verdict through the CLI exit code the way a CI job would.
"""

import copy
import io
import json
import os

import pytest

from recusal.__main__ import (
    HOOK_ON_FAILURE,
    HOOK_ON_FAILURE_MIN_CLAUDE_CODE,
    _hook_entries,
    doctor_command,
    doctor_findings,
    init,
    main,
    repair_launcher,
)


def _settings_path(project):
    return os.path.join(str(project), ".claude", "settings.json")


def _scaffold(project):
    assert init(str(project), stdout=io.StringIO()) == 0
    with open(_settings_path(project), encoding="utf-8") as fh:
        return json.load(fh)


def _write(project, settings):
    with open(_settings_path(project), "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)


def _checks(project):
    return {f.check: f for f in doctor_findings(str(project))}


def _gate_hook(settings):
    for group in settings["hooks"]["PreToolUse"]:
        for hook in group["hooks"]:
            if "recusal_gate.py" in hook["command"]:
                return group, hook
    raise AssertionError("no gate hook registered")


# --- a fresh install is healthy, and says why ---------------------------------------------


def test_fresh_install_passes_every_registration_check(tmp_path):
    _scaffold(tmp_path)
    checks = _checks(tmp_path)
    for name in (
        "hooks_enabled",
        "hook_synchronous",
        "hook_matcher_covers_every_tool",
        "hook_blocks_on_failure",
    ):
        assert checks[name].passed, checks[name].message
    assert "hook_type" not in checks  # only reported when something is wrong
    assert doctor_command(str(tmp_path), stdout=io.StringIO()) == 0


def test_generated_entries_register_on_failure_block_on_every_launcher():
    for launcher in ("posix", "powershell", "both"):
        for group in _hook_entries(launcher):
            assert group["matcher"] == ".*"
            for hook in group["hooks"]:
                assert hook["type"] == "command"
                assert hook["onFailure"] == HOOK_ON_FAILURE == "block"
                assert "async" not in hook


# --- one alteration at a time -----------------------------------------------------------


def test_async_hook_is_critical_and_not_installed(tmp_path):
    settings = _scaffold(tmp_path)
    _, hook = _gate_hook(settings)
    hook["async"] = True
    _write(tmp_path, settings)
    checks = _checks(tmp_path)
    assert not checks["hook_synchronous"].passed
    assert checks["hook_synchronous"].severity.value == "CRITICAL"
    assert "background" in checks["hook_synchronous"].message
    assert doctor_command(str(tmp_path), stdout=io.StringIO()) == 2


def test_async_rewake_hook_is_critical(tmp_path):
    settings = _scaffold(tmp_path)
    _, hook = _gate_hook(settings)
    hook["asyncRewake"] = True
    _write(tmp_path, settings)
    assert not _checks(tmp_path)["hook_synchronous"].passed


def test_disable_all_hooks_is_critical(tmp_path):
    settings = _scaffold(tmp_path)
    settings["disableAllHooks"] = True
    _write(tmp_path, settings)
    checks = _checks(tmp_path)
    assert not checks["hooks_enabled"].passed
    assert "disableAllHooks" in checks["hooks_enabled"].message
    assert doctor_command(str(tmp_path), stdout=io.StringIO()) == 2


def test_disable_all_hooks_false_is_fine(tmp_path):
    settings = _scaffold(tmp_path)
    settings["disableAllHooks"] = False
    _write(tmp_path, settings)
    assert _checks(tmp_path)["hooks_enabled"].passed


@pytest.mark.parametrize("matcher", ["Bash", "NoSuchTool", "Write|Edit", "mcp__.*"])
def test_narrowed_matcher_is_degraded(tmp_path, matcher):
    settings = _scaffold(tmp_path)
    group, _ = _gate_hook(settings)
    group["matcher"] = matcher
    _write(tmp_path, settings)
    checks = _checks(tmp_path)
    assert not checks["hook_matcher_covers_every_tool"].passed
    assert matcher in checks["hook_matcher_covers_every_tool"].message
    assert checks["hook_matcher_covers_every_tool"].severity.value == "ERROR"
    assert doctor_command(str(tmp_path), stdout=io.StringIO()) == 1


@pytest.mark.parametrize("matcher", [".*", "", "*", None])
def test_every_tool_matchers_pass(tmp_path, matcher):
    settings = _scaffold(tmp_path)
    group, _ = _gate_hook(settings)
    if matcher is None:
        del group["matcher"]
    else:
        group["matcher"] = matcher
    _write(tmp_path, settings)
    assert _checks(tmp_path)["hook_matcher_covers_every_tool"].passed


def test_missing_on_failure_block_is_degraded_with_repair_guidance(tmp_path):
    settings = _scaffold(tmp_path)
    _, hook = _gate_hook(settings)
    del hook["onFailure"]
    _write(tmp_path, settings)
    checks = _checks(tmp_path)
    finding = checks["hook_blocks_on_failure"]
    assert not finding.passed
    assert finding.severity.value == "ERROR"
    assert "--repair-launcher" in finding.message
    assert HOOK_ON_FAILURE_MIN_CLAUDE_CODE in finding.message
    assert doctor_command(str(tmp_path), stdout=io.StringIO()) == 1


def test_on_failure_continue_is_degraded(tmp_path):
    settings = _scaffold(tmp_path)
    _, hook = _gate_hook(settings)
    hook["onFailure"] = "continue"
    _write(tmp_path, settings)
    assert not _checks(tmp_path)["hook_blocks_on_failure"].passed


def test_non_command_hook_type_is_critical(tmp_path):
    settings = _scaffold(tmp_path)
    _, hook = _gate_hook(settings)
    hook["type"] = "http"
    _write(tmp_path, settings)
    checks = _checks(tmp_path)
    assert not checks["hook_type"].passed
    assert checks["hook_type"].severity.value == "CRITICAL"


def test_a_short_timeout_is_a_metric_not_a_defect(tmp_path):
    settings = _scaffold(tmp_path)
    _, hook = _gate_hook(settings)
    hook["timeout"] = 1
    _write(tmp_path, settings)
    checks = _checks(tmp_path)
    assert checks["hook_timeout"].passed
    assert checks["hook_timeout"].severity.value == "INFO"
    assert "1" in checks["hook_timeout"].message
    assert doctor_command(str(tmp_path), stdout=io.StringIO()) == 0


def test_the_pre_0_12_4_registration_reads_as_degraded_not_healthy(tmp_path):
    """The exact shape ``init`` wrote before 0.12.4: no onFailure. Doctor must not call it
    healthy, because a timed-out hook lets the call through on that install."""
    settings = _scaffold(tmp_path)
    for group in settings["hooks"]["PreToolUse"]:
        for hook in group["hooks"]:
            hook.pop("onFailure", None)
    _write(tmp_path, settings)
    assert doctor_command(str(tmp_path), stdout=io.StringIO()) == 1


# --- repair: existing installs get the field without losing anything -----------------------


def test_repair_launcher_adds_on_failure_block_to_a_canonical_install(tmp_path):
    settings = _scaffold(tmp_path)
    for group in settings["hooks"]["PreToolUse"]:
        for hook in group["hooks"]:
            hook.pop("onFailure", None)
    settings["permissions"] = {"allow": ["Bash(ls:*)"]}
    _write(tmp_path, settings)
    assert doctor_command(str(tmp_path), stdout=io.StringIO()) == 1
    out = io.StringIO()
    assert repair_launcher(str(tmp_path), stdout=out) == 0
    assert "replaced" in out.getvalue()
    with open(_settings_path(tmp_path), encoding="utf-8") as fh:
        repaired = json.load(fh)
    _, hook = _gate_hook(repaired)
    assert hook["onFailure"] == "block"
    assert repaired["permissions"] == {"allow": ["Bash(ls:*)"]}
    assert doctor_command(str(tmp_path), stdout=io.StringIO()) == 0
    # idempotent
    out = io.StringIO()
    assert repair_launcher(str(tmp_path), stdout=out) == 0
    assert "no change" in out.getvalue()


def test_repair_launcher_leaves_a_customized_hook_alone(tmp_path):
    settings = _scaffold(tmp_path)
    _, hook = _gate_hook(settings)
    hook["command"] = hook["command"] + " # customized"
    hook.pop("onFailure", None)
    _write(tmp_path, settings)
    before = copy.deepcopy(settings)
    out = io.StringIO()
    repair_launcher(str(tmp_path), stdout=out)
    with open(_settings_path(tmp_path), encoding="utf-8") as fh:
        after = json.load(fh)
    customized = [
        h
        for g in after["hooks"]["PreToolUse"]
        for h in g["hooks"]
        if h["command"].endswith("# customized")
    ]
    assert customized == [_gate_hook(before)[1]]
    assert "onFailure" not in customized[0]
    assert "customized recusal hook" in out.getvalue()


def test_init_on_an_old_install_points_at_repair(tmp_path):
    settings = _scaffold(tmp_path)
    for group in settings["hooks"]["PreToolUse"]:
        for hook in group["hooks"]:
            hook.pop("onFailure", None)
    _write(tmp_path, settings)
    out = io.StringIO()
    assert init(str(tmp_path), stdout=out) == 0
    text = out.getvalue()
    assert "already register the gate" in text
    assert "--repair-launcher" in text
    assert '"onFailure": "block"' in text


def test_init_on_a_current_install_is_quiet_about_repair(tmp_path):
    _scaffold(tmp_path)
    out = io.StringIO()
    assert init(str(tmp_path), stdout=out) == 0
    assert "--repair-launcher" not in out.getvalue()


def test_cli_repair_path_end_to_end(tmp_path):
    settings = _scaffold(tmp_path)
    for group in settings["hooks"]["PreToolUse"]:
        for hook in group["hooks"]:
            hook.pop("onFailure", None)
    _write(tmp_path, settings)
    assert main(["doctor", "--dir", str(tmp_path)]) == 1
    assert main(["init", "--dir", str(tmp_path), "--repair-launcher"]) == 0
    assert main(["doctor", "--dir", str(tmp_path)]) == 0


# --- negative control: the launcher-text checks still fire ------------------------------


def test_launcher_without_exit_2_is_still_flagged(tmp_path):
    settings = _scaffold(tmp_path)
    _, hook = _gate_hook(settings)
    hook["command"] = "python $CLAUDE_PROJECT_DIR/.claude/hooks/recusal_gate.py"
    _write(tmp_path, settings)
    assert not _checks(tmp_path)["launcher_fails_closed"].passed
