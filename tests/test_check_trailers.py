"""``tools/check_trailers.py``: no commit may list an Anthropic account as a contributor.

The CI job ``No Anthropic co-author trailer`` runs this script over every PR's commits.
These tests pin what it refuses (an Anthropic address as co-author, author or
committer, in any spelling a coding assistant writes) and, as the negative control, what
it must never refuse (a human co-author named Claude, AI help mentioned in prose).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_trailers import main, offending_lines  # noqa: E402

PHILIP = "Philip Paz <philip.paz@gmail.com>"

REFUSED = [
    "Co-Authored-By: Claude <noreply@anthropic.com>",
    "Co-authored-by: Claude Opus 5.5 <noreply@anthropic.com>",
    "co-authored-by: Claude Fable 5 <noreply@anthropic.com>",
    "CO-AUTHORED-BY:Claude <NOREPLY@ANTHROPIC.COM>",
    "  Co-authored-by: Someone <someone@anthropic.com>",
    "Co-authored-by: claude <209825114+claude@users.noreply.github.com>",
    "Co-authored-by: claude <claude@users.noreply.github.com>",
]

ALLOWED = [
    "Co-authored-by: Claude Dupont <claude.dupont@example.fr>",
    "Co-authored-by: Philip Paz <philip.paz@gmail.com>",
    "Signed-off-by: Ana <ana@example.com>",
    "Written with help from Claude Code; reviewed and tested by hand.",
    "Mentions noreply@anthropic.com in prose, not as a trailer.",
    "Co-authored-by: Notclaude <notclaude@users.noreply.github.com>",
]


@pytest.mark.parametrize("line", REFUSED)
def test_an_anthropic_co_author_is_refused(line):
    message = f"fix: something\n\nBody.\n\n{line}\n"
    assert offending_lines(PHILIP, PHILIP, message) == [line.strip()]


@pytest.mark.parametrize("line", ALLOWED)
def test_people_and_prose_are_never_refused(line):
    assert offending_lines(PHILIP, PHILIP, f"fix: something\n\n{line}\n") == []


def test_an_anthropic_author_or_committer_is_refused():
    bot = "Claude <noreply@anthropic.com>"
    assert offending_lines(bot, PHILIP, "x") == [f"author: {bot}"]
    assert offending_lines(PHILIP, bot, "x") == [f"committer: {bot}"]


def test_every_offending_line_is_reported():
    message = (
        "x\n\nCo-authored-by: A <a@anthropic.com>\nCo-authored-by: B <noreply@anthropic.com>\n"
    )
    assert len(offending_lines(PHILIP, PHILIP, message)) == 2


# --- end to end, against a real repository ----------------------------------------------

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _commit(repo, name, message, author="Philip Paz <philip.paz@gmail.com>"):
    (repo / name).write_text(name, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "--author", author, "-m", message)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Philip Paz")
    _git(tmp_path, "config", "user.email", "philip.paz@gmail.com")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _commit(tmp_path, "base", "base")
    _git(tmp_path, "tag", "base")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@needs_git
def test_a_clean_range_passes(repo, capsys):
    _commit(repo, "a", "feat: a\n\nCo-authored-by: Claude Dupont <claude.dupont@example.fr>")
    assert main(["base..HEAD"]) == 0
    assert "1 commit(s) checked" in capsys.readouterr().out


@needs_git
def test_a_range_with_the_trailer_fails_and_says_how_to_fix_it(repo, capsys):
    _commit(repo, "a", "feat: a")
    _commit(repo, "b", "feat: b\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    assert main(["base..HEAD"]) == 1
    out = capsys.readouterr().out
    assert "1 of 2 commit(s)" in out
    assert "Co-Authored-By: Claude <noreply@anthropic.com>" in out
    assert "git commit --amend" in out


@needs_git
def test_an_anthropic_author_fails(repo):
    _commit(repo, "a", "feat: a", author="Claude <noreply@anthropic.com>")
    assert main(["base..HEAD"]) == 1


@needs_git
def test_commits_before_the_range_are_not_judged(repo):
    _commit(repo, "a", "feat: a\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    _git(repo, "tag", "after")
    _commit(repo, "b", "feat: b")
    assert main(["after..HEAD"]) == 0
    assert main(["HEAD"]) == 1  # the whole history, as the job checks main


@needs_git
def test_an_unknown_range_is_an_operational_error(repo):
    assert main(["no-such-ref..HEAD"]) == 2


def test_usage_error():
    assert main([]) == 2
