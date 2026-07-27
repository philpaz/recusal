"""``recusal demo``: the refusal an adopter sees first, pinned.

This command exists so the first refusal costs one command instead of a clone, which
makes its output a public claim like any other. These tests pin what it shows, that it
shows the same thing twice, that it runs as a standalone process from a directory that
is not the checkout, and that it stays a pure stdout narrative: nothing written, nothing
spawned, nothing fetched.
"""

import os
import subprocess
import sys

from recusal.__main__ import main

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_SOURCE = os.path.join(REPO, "recusal", "_demo.py")


def _run(capsys, *argv):
    code = main(["demo", *argv])
    return code, capsys.readouterr().out


def test_demo_runs_all_three_scenarios_by_default(capsys):
    code, out = _run(capsys)
    assert code == 0
    assert "WRONG-SUBJECT WRITE" in out
    assert "DESTRUCTIVE SHELL" in out
    assert "MCP DISCOVERY DRIFT" in out


def test_wrong_subject_refuses_then_allows(capsys):
    code, out = _run(capsys, "--scenario", "wrong-subject")
    assert code == 0
    assert "REFUSED before the tool ran" in out
    assert "C-9988" in out  # the wrong-subject target is named in the refusal
    assert "ALLOWED, the tool executes" in out
    # The other scenarios stayed out of a single-scenario run.
    assert "DESTRUCTIVE SHELL" not in out


def test_destructive_shell_refuses_including_the_gates_own_uninstall(capsys):
    code, out = _run(capsys, "--scenario", "destructive-shell")
    assert code == 0
    for line in out.splitlines():
        if any(
            key in line
            for key in ("destructive removal", "uninstalling the gate", "writing a secret file")
        ):
            assert line.rstrip().endswith("DENY"), line
        if "an ordinary command" in line:
            assert line.rstrip().endswith("DEFER"), line  # never auto-allow
    assert "recusal init --posture allowlist" in out  # the deny-list ceiling names its answer


def test_mcp_drift_passes_the_pin_and_refuses_both_drifts(capsys):
    code, out = _run(capsys, "--scenario", "mcp-drift")
    assert code == 0
    for line in out.splitlines():
        if "the same catalog, re-observed" in line:
            assert line.rstrip().endswith("PASS"), line
        if "rewritten after approval" in line or "unreviewed tool appeared" in line:
            assert line.rstrip().endswith("FAIL"), line
    assert "mcp_tool_changed" in out
    assert "mcp_unpinned_tool" in out
    # The manifest stores hashes: the poisoned description must never be echoed back.
    assert "~/.ssh" not in out


def test_list_names_every_scenario(capsys):
    code, out = _run(capsys, "--list")
    assert code == 0
    for name in ("wrong-subject", "destructive-shell", "mcp-drift"):
        assert name in out


def test_unknown_scenario_is_refused_by_the_parser():
    """A typo must not silently run something else."""
    try:
        main(["demo", "--scenario", "no-such-scenario"])
    except SystemExit as exc:  # argparse exits 2, the repo's operational-error code
        assert exc.code == 2
    else:  # pragma: no cover - a passing parse here would be the defect
        raise AssertionError("an unknown scenario must not be accepted")


def test_demo_output_is_byte_identical_across_runs(capsys):
    """Determinism is the product claim; the demo is not exempt from it."""
    _, first = _run(capsys)
    _, second = _run(capsys)
    assert first == second


def test_demo_runs_standalone_from_a_neutral_directory(tmp_path):
    """The actual user path: installed package, some other working directory."""
    proc = subprocess.run(
        [sys.executable, "-m", "recusal", "demo"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert proc.stderr == ""
    assert "REFUSED before the tool ran" in proc.stdout
    # A demo that writes files is not a demo you can run anywhere.
    assert os.listdir(tmp_path) == []


def test_demo_module_stays_a_pure_stdout_narrative():
    """No network, no subprocess, no disk: locked at the source, not just observed."""
    source = open(DEMO_SOURCE, encoding="utf-8").read()
    for forbidden in ("import subprocess", "import socket", "import urllib", "import tempfile"):
        assert forbidden not in source, forbidden
    assert "open(" not in source
    assert "from examples" not in source  # ships in the wheel; examples/ does not
