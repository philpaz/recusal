"""Exercise the operator recipe's real refusals and recovered files, without network I/O."""

import json
import runpy
from pathlib import Path

import pytest

from examples.audit_operations import PARTIAL, anchor_example, framing_example, integrity_example
from recusal.audit import load, verify_file


@pytest.mark.parametrize(
    "demo,name",
    [(framing_example, "framing"), (integrity_example, "integrity"), (anchor_example, "anchor")],
)
def test_examples_refuse_to_overwrite_existing_files(tmp_path, demo, name):
    path = tmp_path / (name + ".jsonl")
    path.write_bytes(b"existing operator record")

    with pytest.raises(FileExistsError):
        demo(tmp_path)

    assert path.read_bytes() == b"existing operator record"


@pytest.mark.parametrize(
    "demo,name",
    [
        (framing_example, "framing"),
        (integrity_example, "integrity"),
        (anchor_example, "anchor"),
    ],
)
def test_examples_do_not_replace_a_preexisting_damaged_copy(tmp_path, demo, name):
    backup = tmp_path / (name + ".damaged")
    backup.write_bytes(b"previous evidence")

    with pytest.raises(FileExistsError):
        demo(tmp_path)

    assert backup.read_bytes() == b"previous evidence"
    path = tmp_path / (name + ".jsonl")
    if name == "framing":
        assert path.read_bytes().endswith(PARTIAL)
    elif name == "integrity":
        assert verify_file(str(path))[0] is False
    else:
        assert len(load(str(path))) == 1


def test_framing_recovery_keeps_original_prefix_and_damaged_tail(tmp_path, capsys):
    path, backup, offset = framing_example(tmp_path)

    damaged = backup.read_bytes()
    assert damaged.endswith(PARTIAL)
    assert damaged[offset:] == PARTIAL
    assert path.read_bytes().startswith(damaged[:offset])
    assert [entry["action"] for entry in load(str(path))] == ["original-0", "resumed"]
    assert verify_file(str(path)) == (True, [])
    assert verify_file(str(backup))[0] is False
    output = capsys.readouterr().out
    assert "FRAMING REFUSED" in output
    assert str(offset) in output and backup.name in output


def test_integrity_recovery_retains_evidence_and_restores_trusted_records(tmp_path, capsys):
    path, backup = integrity_example(tmp_path)

    assert verify_file(str(backup))[0] is False
    assert json.loads(backup.read_bytes().splitlines()[0])["actor"] == "unexpected-actor"
    entries = load(str(path))
    assert [entry["action"] for entry in entries] == ["original-0", "original-1", "resumed"]
    assert all(entry["actor"] is None for entry in entries)
    assert verify_file(str(path)) == (True, [])
    output = capsys.readouterr().out
    assert "INTEGRITY REFUSED" in output
    assert "Investigate" in output and "do not delete records or recompute hashes" in output


def test_anchor_recovery_does_not_accept_a_self_consistent_truncated_prefix(tmp_path, capsys):
    path, backup, original_head, next_head = anchor_example(tmp_path)

    assert len(load(str(backup))) == 1
    assert verify_file(str(backup)) == (True, [])  # unanchored verification cannot see the loss
    assert verify_file(str(backup), expected_head=original_head)[0] is False
    entries = load(str(path))
    assert original_head == (2, entries[1]["hash"])
    assert next_head == (3, entries[2]["hash"])
    assert [entry["action"] for entry in entries] == ["original-0", "original-1", "resumed"]
    assert verify_file(str(path), expected_head=next_head) == (True, [])
    output = capsys.readouterr().out
    assert "ANCHOR REFUSED" in output
    assert "do not lower it to the local prefix" in output


def test_demo_uses_only_its_temporary_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    example = Path(__file__).resolve().parents[1] / "examples" / "audit_operations.py"

    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert all(label + " REFUSED" in output for label in ("FRAMING", "INTEGRITY", "ANCHOR"))
    assert list(tmp_path.iterdir()) == []
