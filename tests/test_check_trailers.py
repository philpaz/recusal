"""``tools/check_trailers.py``: find commits that would list an AI tool or a bot as a contributor.

The CI job ``Contributor credit`` runs this script as a heads-up over every PR's commits
(``--advisory``, never blocking) and strictly over main. These tests pin what it flags
(an AI vendor's address, an AI tool's GitHub account or any ``[bot]`` account, as
co-author, author or committer), that a PR is never blocked, and, as the negative
control, what it must never flag (people named Claude, Codex or Grok, look-alike
domains, AI help mentioned in prose).
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
    "Co-authored-by: Claude <209825114+claude[bot]@users.noreply.github.com>",
    "Co-authored-by: Codex <noreply@openai.com>",
    "Co-authored-by: Codex <codex@openai.com>",
    "Co-authored-by: Codex <267193182+codex@users.noreply.github.com>",
    "Co-authored-by: chatgpt-codex-connector[bot] <199175422+chatgpt-codex-connector[bot]@users.noreply.github.com>",
    "Co-authored-by: Grok <noreply@x.ai>",
    "Co-authored-by: Grok 4 <grok@x.ai>",
    "Co-authored-by: Cursor Agent <cursoragent@cursor.com>",
    "Co-authored-by: Copilot <198982749+Copilot@users.noreply.github.com>",
    "Co-authored-by: OpenHands <175740463+openhands-agent@users.noreply.github.com>",
    "Co-authored-by: Jules <161369871+google-labs-jules[bot]@users.noreply.github.com>",
    "Co-authored-by: Devin <158243242+devin-ai-integration[bot]@users.noreply.github.com>",
    "Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
    "Co-authored-by: Someone <someone@research.openai.com>",
]

ALLOWED = [
    "Co-authored-by: Claude Dupont <claude.dupont@example.fr>",
    "Co-authored-by: Philip Paz <philip.paz@gmail.com>",
    "Signed-off-by: Ana <ana@example.com>",
    "Written with help from Claude Code; reviewed and tested by hand.",
    "Mentions noreply@anthropic.com in prose, not as a trailer.",
    "Co-authored-by: Notclaude <notclaude@users.noreply.github.com>",
    "Co-authored-by: Codex Fan <codexfan@gmail.com>",
    "Co-authored-by: Grok Smith <grok@example.com>",
    "Co-authored-by: Max <max@max.ai>",
    "Co-authored-by: Ana <ana@notopenai.com>",
    "Co-authored-by: Ana <ana@openai.community>",
    "Co-authored-by: Ana <ana@anthropic.com.example.org>",
    "Co-authored-by: Ana <1+copilotfan@users.noreply.github.com>",
    "Written with Codex, Cursor and Grok; every line reviewed by me.",
]


@pytest.mark.parametrize("line", REFUSED)
def test_an_ai_or_bot_co_author_is_refused(line):
    message = f"fix: something\n\nBody.\n\n{line}\n"
    assert offending_lines(PHILIP, PHILIP, message) == [line.strip()]


@pytest.mark.parametrize("line", ALLOWED)
def test_people_and_prose_are_never_refused(line):
    assert offending_lines(PHILIP, PHILIP, f"fix: something\n\n{line}\n") == []


@pytest.mark.parametrize(
    "bot",
    [
        "Claude <noreply@anthropic.com>",
        "Codex <noreply@openai.com>",
        "Cursor Agent <cursoragent@cursor.com>",
        "copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
        "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>",
    ],
)
def test_an_ai_or_bot_author_or_committer_is_refused(bot):
    assert offending_lines(bot, PHILIP, "x") == [f"author: {bot}"]
    assert offending_lines(PHILIP, bot, "x") == [f"committer: {bot}"]


def test_github_itself_as_committer_is_not_refused():
    # A commit made in GitHub's web editor has GitHub as committer and the person as
    # author; GitHub does not list itself as a contributor.
    assert offending_lines(PHILIP, "GitHub <noreply@github.com>", "x") == []


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
    # nobody has to touch git: the maintainer removes the line when merging
    assert "Nothing for you to\ndo: the maintainer removes these lines when merging" in out


@needs_git
def test_an_ai_author_fails(repo):
    _commit(repo, "a", "feat: a", author="Claude <noreply@anthropic.com>")
    assert main(["base..HEAD"]) == 1


@needs_git
def test_a_codex_trailer_fails(repo, capsys):
    _commit(repo, "a", "feat: a\n\nCo-authored-by: Codex <noreply@openai.com>")
    assert main(["base..HEAD"]) == 1
    assert "--reset-author" in capsys.readouterr().out


@needs_git
def test_advisory_mode_never_blocks_a_pr_but_still_says_what_it_found(repo, capsys, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    _commit(repo, "a", "feat: a\n\nCo-authored-by: Codex <noreply@openai.com>")
    assert main(["--advisory", "base..HEAD"]) == 0
    out = capsys.readouterr().out
    assert "Heads-up: 1 of 1 commit(s)" in out
    assert "::notice title=Contributor credit::" in out
    assert "Co-authored-by: Codex <noreply@openai.com>" in out


@needs_git
def test_advisory_mode_on_a_clean_range(repo, capsys):
    _commit(repo, "a", "feat: a")
    assert main(["--advisory", "base..HEAD"]) == 0
    assert "credits people only" in capsys.readouterr().out


def test_advisory_mode_still_reports_a_bad_range(repo):
    assert main(["--advisory", "no-such-ref..HEAD"]) == 2


def test_advisory_usage_error():
    assert main(["--advisory"]) == 2


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
