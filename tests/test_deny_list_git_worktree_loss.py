"""Git commands that throw away working-tree changes are refused, in every spelling.

The deny list models git working-tree loss. Every refused form here is paired with a
safe neighbour that must still defer (``checkout -f`` vs ``checkout main``, ``stash
drop`` vs ``stash list``), so a pattern that refused everything would fail too. Global
options before the subcommand (``git -C repo``, ``git -c k=v``) must not hide it.
"""

import itertools
import re
import time

import pytest

from recusal.claude_code import decide
from recusal.deny_list import analyze_command, deny_list_policy

policy = deny_list_policy()


def _decision(command):
    return decide("Bash", {"command": command}, policy)[0]


def _refused(command):
    return any(not f.passed for f in analyze_command(command))


REFUSED = [
    # already refused before this change
    "git clean -fd",
    "git clean --force",
    "git checkout -- .",
    "git checkout HEAD -- a.py",
    "git restore src/a.py",
    "git reset --hard",
    # checkout that discards local changes
    "git checkout -f main",
    "git checkout --force main",
    "git checkout -fb topic",
    "git checkout .",
    "git checkout HEAD .",
    "git checkout ./src",
    "git checkout main ./docs/a.md",
    # switch that discards local changes
    "git switch -f main",
    "git switch --force main",
    "git switch --discard-changes main",
    "git switch main --discard-changes",
    # a worktree removed with its uncommitted changes
    "git worktree remove --force ../wt",
    "git worktree remove -f ../wt",
    "git worktree remove ../wt --force",
    "git worktree remove -f -f ../wt",
    # saved working-tree changes discarded
    "git stash drop",
    "git stash drop stash@{1}",
    "git stash clear",
]

GLOBAL_OPTIONS = [
    "-C .",
    "-C ../repo",
    '-C "my repo"',
    "-C 'my repo'",
    "-c core.autocrlf=false",
    "-c user.name=x -c user.email=y",
    "--git-dir=.git --work-tree=.",
    "--git-dir .git --work-tree .",
    "--no-pager",
    "-P",
    "--no-optional-locks -C .",
    "--literal-pathspecs",
]

SPELLINGS = [
    "GIT CHECKOUT -F main",  # case
    "git   checkout    -f   main",  # spacing
    'g""it checkout -f main',  # quote splitting
    "g\\it stash clear",  # escape splitting
    "git.exe stash clear",
    "/usr/bin/git checkout -f main",
    "C:\\Program Files\\Git\\bin\\git.exe checkout -f main",
    "cd repo && git checkout -f main",
    "git fetch && git switch --discard-changes main",
    "git status; git stash clear",
    "GIT_DIR=.git git worktree remove --force ../wt",
]

SAFE = [
    "git status",
    "git checkout main",
    "git checkout -b topic",
    "git checkout -b fix-f",
    "git checkout -b feature/x origin/feature/x",
    "git checkout --track origin/x",
    "git checkout --detach",
    "git checkout .github/workflows/ci.yml",
    "git switch main",
    "git switch -c topic",
    "git switch --create topic",
    "git switch --detach v1",
    "git worktree list",
    "git worktree add ../wt",
    "git worktree add -f ../wt topic",
    "git worktree remove ../wt",
    "git worktree prune",
    "git stash",
    "git stash list",
    "git stash show -p",
    "git stash push -m drop",
    "git stash pop",
    "git stash apply",
    "git add .",
    "git diff .",
    "git log -- .",
    "git status .",
    "git restore-mtime",
    "git -C ../repo status",
    "git -c core.pager=less log",
    "git --no-pager diff",
    'git commit -m "restore the docs"',
    'git commit -m "switch to the new parser"',
    "git log --grep=clean",
    "echo stash clear",
]


@pytest.mark.parametrize("command", REFUSED)
def test_refused(command):
    assert _decision(command) == "deny", command


@pytest.mark.parametrize("command", SAFE)
def test_safe_neighbour_defers(command):
    assert _decision(command) == "defer", command


@pytest.mark.parametrize(
    "options,verb", list(itertools.product(GLOBAL_OPTIONS, [c[4:] for c in REFUSED]))
)
def test_global_options_do_not_hide_the_verb(options, verb):
    assert _refused(f"git {options} {verb}"), f"git {options} {verb}"


@pytest.mark.parametrize("options", GLOBAL_OPTIONS)
def test_global_options_alone_do_not_refuse(options):
    for verb in ("status", "checkout main", "switch -c topic", "stash list", "worktree list"):
        assert not _refused(f"git {options} {verb}"), f"git {options} {verb}"


@pytest.mark.parametrize("command", SPELLINGS)
def test_spellings(command):
    assert _decision(command) == "deny", command


def test_an_mcp_shell_tool_is_gated_the_same_way():
    # a command under a command-like key of any tool is adjudicated, not just Bash
    assert decide("mcp__shell__run", {"command": "git stash clear"}, policy)[0] == "deny"
    assert decide("mcp__shell__run", {"command": "git stash list"}, policy)[0] == "defer"


# --- no command refused before is allowed now -------------------------------------------

# The git working-tree patterns exactly as 0.12.0 shipped them.
_OLD = re.compile(
    r"\bgit\s+clean\b[^|&;]{0,256}-[a-z]{0,8}f"
    r"|\bgit\s+checkout\b[^|&;]{0,256}\s--(\s|$)"
    r"|\bgit\s+restore(?![\w-])"
)


def _corpus():
    verbs = ["clean", "checkout", "restore", "switch", "worktree remove", "stash"]
    args = [
        "",
        "-f",
        "-fd",
        "-xdf",
        "--force",
        "--",
        "-- .",
        ".",
        "main",
        "-b x",
        "drop",
        "clear",
        "list",
        "--staged a",
        "--source=h a",
        "-q -f",
        "./a",
        "../wt",
    ]
    prefixes = ["git", "git -C .", "git -c a=b", "cd x && git", "g''it", "git --no-pager"]
    for p, v, a in itertools.product(prefixes, verbs, args):
        yield f"{p} {v} {a}".strip()


def test_every_command_the_old_patterns_refused_is_still_refused():
    checked = 0
    for command in _corpus():
        if _OLD.search(command.lower()):
            checked += 1
            assert _refused(command), command
    assert checked >= 40  # the corpus really exercises the old patterns (50 today)


# --- the new pattern stays linear ---------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git " + "-c x=y " * 600 + "status",
        "git " + "-C " * 1300,
        "git checkout " + "a " * 2000,
        "git worktree remove " + "-x " * 1300,
        "git " + '-C "' * 1000,
    ],
)
def test_adversarial_input_is_fast(command):
    start = time.perf_counter()
    analyze_command(command[:4096])
    assert time.perf_counter() - start < 1.0
