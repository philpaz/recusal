"""Recipe 3 checks real destinations without claiming to be a filesystem sandbox."""

import os
from pathlib import Path

import pytest

from examples.workspace_policy import make_workspace_policy
from recusal import compute_verdict


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.mark.parametrize("tool", ["Write", "Edit", "MultiEdit"])
def test_new_nested_file_inside_workspace_defers(workspace, tool):
    assert (
        make_workspace_policy(workspace)(tool, {"file_path": str(workspace / "new" / "notes.txt")})
        == []
    )


@pytest.mark.parametrize(
    "destination", ["../outside.txt", "../workspace_evil/notes.txt", "sub/../../outside.txt"]
)
def test_escapes_are_refused(workspace, destination):
    findings = make_workspace_policy(workspace)(
        "Write", {"file_path": str(workspace / destination)}
    )
    assert compute_verdict(findings).refused


@pytest.mark.parametrize(
    "name", [".env", "server.pem", "id_rsa", "docs/secrets-howto.md", "CREDENTIALS.txt"]
)
def test_secret_substrings_are_intentionally_refused(workspace, name):
    assert compute_verdict(
        make_workspace_policy(workspace)("Write", {"file_path": str(workspace / name)})
    ).refused


@pytest.mark.parametrize("value", [None, "", 42, "bad\x00name"])
def test_invalid_paths_are_refused(workspace, value):
    assert compute_verdict(make_workspace_policy(workspace)("Write", {"file_path": value})).refused


def test_relative_paths_use_hook_cwd(workspace, monkeypatch):
    policy = make_workspace_policy(workspace)
    monkeypatch.chdir(workspace)
    assert policy("Write", {"file_path": "notes.txt"}) == []
    monkeypatch.chdir(workspace.parent)
    assert compute_verdict(policy("Write", {"file_path": "notes.txt"})).refused


def test_symlink_parent_escape_for_new_file_is_refused(workspace):
    outside = workspace.parent / "outside"
    outside.mkdir()
    link = workspace / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    assert compute_verdict(
        make_workspace_policy(workspace)("Write", {"file_path": str(link / "new.txt")})
    ).refused


def test_symlink_alias_to_secret_is_refused(workspace):
    secret = workspace / ".env"
    secret.touch()
    link = workspace / "notes.txt"
    try:
        link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    assert compute_verdict(
        make_workspace_policy(workspace)("Write", {"file_path": str(link)})
    ).refused


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_windows_case_and_different_drives(workspace):
    policy = make_workspace_policy(workspace)
    assert policy("Write", {"file_path": str(workspace / "notes.txt").upper()}) == []
    other_drive = "Z:" if workspace.drive.upper() != "Z:" else "Y:"
    assert compute_verdict(
        policy("Write", {"file_path": str(Path(other_drive + "\\outside.txt"))})
    ).refused


def test_uninspected_tools_defer(workspace):
    assert make_workspace_policy(workspace)("Read", {"file_path": ".env"}) == []
