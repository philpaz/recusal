"""Git commands that delete branches, tags, remote refs or recovery history are refused.

Every refused form here is paired with a safe neighbour that must still defer (``branch
-D`` vs ``branch -d``, ``tag -d`` vs ``tag -l``, ``push origin :x`` vs ``push origin
HEAD:x``, ``git prune`` vs ``git remote prune``), so a pattern that refused everything
would fail too. Branch flags are case-sensitive in git (``-d`` refuses an unmerged branch,
``-D`` deletes it anyway), so case matters there and nowhere else.
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
    # a branch deleted or overwritten whatever it holds
    "git branch -D feature",
    "git branch -D a b c",
    "git branch -rD origin/feature",
    "git branch -Dr origin/feature",
    "git branch -d -f feature",
    "git branch -f -d feature",
    "git branch -df feature",
    "git branch -fd feature",
    "git branch --delete --force feature",
    "git branch --force --delete feature",
    "git branch -d --force feature",
    "git branch --delete -f feature",
    "git branch -M main",
    "git branch -M old main",
    "git branch -C old main",
    # a tag deleted
    "git tag -d v1.0",
    "git tag -d v1.0 v1.1",
    "git tag --delete v1.0",
    # a ref deleted on the remote
    "git push origin --delete feature",
    "git push --delete origin feature",
    "git push -d origin feature",
    "git push origin -d feature",
    "git push origin :feature",
    "git push origin :refs/heads/feature",
    "git push origin :refs/tags/v1.0",
    "git push origin main :old",
    "git push --mirror backup",
    "git push --prune origin",
    "git push origin main --prune",
    "git push origin --prune refs/heads/*:refs/heads/*",
    # the history that recovers a lost commit
    "git reflog expire --expire=now --all",
    "git reflog expire --all",
    "git reflog delete HEAD@{1}",
    "git gc --prune=now",
    "git gc --prune=all",
    "git gc --aggressive --prune=now",
    "git prune",
    "git prune --expire=now",
    "git prune -v",
    "git update-ref -d refs/heads/feature",
    "git update-ref -d HEAD",
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
    "GIT BRANCH -D feature",  # git and branch in any case; the flag exactly
    "Git Branch -D feature",
    "GIT TAG -D v1.0",  # tag, push and the rest are not case-sensitive in this check
    "GIT PUSH ORIGIN --DELETE feature",
    "git   branch    -D   feature",  # spacing
    "git\tbranch\t-D\tfeature",
    'g""it branch -D feature',  # quote splitting
    "g''it tag -d v1.0",
    'git branch "-D" feature',
    "g\\it branch -D feature",  # escape splitting
    "g\\it reflog expire --all",
    "git${IFS}branch${IFS}-D${IFS}feature",  # $IFS word splitting
    "git$IFS'prune'",
    "git.exe branch -D feature",
    "/usr/bin/git branch -D feature",
    "C:\\Program Files\\Git\\cmd\\git.exe branch -D feature",
    "cd repo && git branch -D feature",
    "git fetch; git push origin :feature",
    "GIT_DIR=.git git branch -D feature",
    "git status | cat; git tag -d v1",
]

SAFE = [
    # branch: what git refuses or what keeps every commit
    "git branch -d feature",
    "git branch --delete feature",
    "git branch -d a b c",
    "git branch -rd origin/feature",
    "git branch -m old new",
    "git branch -c old new",
    "git branch -f feature origin/feature",
    "git branch --force feature main",
    "git branch feature",
    "git branch -a",
    "git branch -r",
    "git branch -vv",
    "git branch --merged",
    "git branch --no-merged main",
    "git branch --contains abc123",
    "git branch --list 'release-*'",
    "git branch -u origin/main",
    "git branch --set-upstream-to=origin/Main",
    "git branch --sort=-committerdate",
    "git branch feature-D",
    "git branch fix-M-thing",
    'git branch --format="%(refname:short) -Date"',  # a word, not a flag
    "git branch --format='%(refname) -Main -Copy'",
    "git -C . branch -a",
    # tag: create, list, inspect
    "git tag v1.0",
    "git tag -a v1.0 -m done",
    "git tag -s v1.0 -m signed",
    "git tag -l",
    "git tag -n",
    "git tag --list 'v1.*'",
    "git tag --contains abc123",
    "git tag -f v1.0",
    "git tag v1.0-d",
    # push: add or update, never delete
    "git push",
    "git push origin main",
    "git push origin HEAD:main",
    "git push origin HEAD:refs/heads/main",
    "git push origin :",
    "git push origin feature:feature",
    "git push --dry-run origin main",
    "git push -u origin feature",
    "git push --set-upstream origin feature",
    "git push --tags",
    "git push --follow-tags",
    "git push origin v1.0",
    "git push origin feature-d",
    # history kept
    "git fetch --prune",
    "git fetch -p",
    "git pull --prune",
    "git remote prune origin",
    "git worktree prune",
    "git prune-packed",
    "git reflog",
    "git reflog show main",
    "git reflog --date=iso",
    "git reflog exists refs/heads/main",
    "git gc",
    "git gc --auto",
    "git gc --aggressive",
    "git gc --prune=2.weeks.ago",
    "git update-ref refs/heads/x HEAD",
    "git update-ref refs/heads/x HEAD abc123",
    "git log --decorate",
    "git describe --tags",
    "git show-ref --tags",
    "git for-each-ref refs/heads",
    "git commit -m 'prune the docs'",
    "git commit -m 'tag the release'",
    # not git
    "branch -D feature",
    "npm prune",
    "docker system prune",
    "docker image prune -a",
    "kubectl delete pod x",
    "echo branch -D",
]


@pytest.mark.parametrize("command", REFUSED)
def test_refused(command):
    assert _decision(command) == "deny", command


@pytest.mark.parametrize("command", REFUSED)
def test_the_refusal_says_why(command):
    (finding,) = [f for f in analyze_command(command) if not f.passed]
    assert finding.check == "destructive_command"
    assert "git history loss" in finding.message


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
    for verb in ("branch -d x", "branch -a", "tag -l", "push origin main", "reflog", "gc"):
        assert not _refused(f"git {options} {verb}"), f"git {options} {verb}"


@pytest.mark.parametrize("command", SPELLINGS)
def test_spellings(command):
    assert _decision(command) == "deny", command


def test_case_decides_only_the_branch_flags():
    # the one place case matters: git itself reads -d and -D differently
    assert not _refused("git branch -d feature")
    assert _refused("git branch -D feature")
    assert not _refused("git branch -m a b")
    assert _refused("git branch -M a b")
    assert not _refused("git branch -c a b")
    assert _refused("git branch -C a b")


def test_one_segment_only():
    # a force flag in a later command is not read as a branch flag
    assert not _refused("git branch -d feature && ls -D")
    assert not _refused("git branch -d feature; git status -f")
    assert not _refused("git branch -d feature | grep -f x")
    assert not _refused("git tag -l && echo -d")


def test_quoted_text_is_read_like_every_other_check():
    # the deny list strips quotes (so g""it cannot hide git); a commit message that QUOTES
    # a destructive command is refused here exactly as `git commit -m 'git clean -fd'` is
    assert _refused("git commit -m 'git branch -D feature'")
    assert _refused("git commit -m 'git clean -fd'")


def test_an_mcp_shell_tool_is_gated_the_same_way():
    assert decide("mcp__shell__run", {"command": "git branch -D x"}, policy)[0] == "deny"
    assert decide("mcp__shell__run", {"command": "git branch -d x"}, policy)[0] == "defer"


# --- no command refused before is allowed now -------------------------------------------


def test_every_command_the_worktree_patterns_refused_is_still_refused():
    # the forms the previous change refused still are, alongside the new ones
    for command in (
        "git clean -fd",
        "git checkout -- .",
        "git checkout -f main",
        "git switch --discard-changes main",
        "git restore a.py",
        "git worktree remove --force ../wt",
        "git stash drop",
        "git push --force",
        "git push origin +main",
        "git reset --hard",
    ):
        assert _refused(command), command


def test_the_ifs_reading_is_unchanged_for_lowercase_input():
    # _deobfuscate now matches $IFS in any case; on the lowercased reading it never sees
    # uppercase, so every earlier $IFS refusal holds and the uppercase form is caught too
    assert _refused("rm${IFS}-rf /tmp/x")
    assert _refused("rm${ifs}-rf /tmp/x")
    assert _refused("git${IFS}branch${IFS}-D${IFS}x")


# --- the new patterns stay linear ---------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git branch " + "-d " * 1300,
        "git branch " + "-x " * 1300 + "-D",
        "git " + "-c x=y " * 600 + "branch -d x",
        "GIT " + "-C . " * 800 + "BRANCH -d x",
        "git push origin " + ":" * 4000,
        "git push origin " + "a " * 2000,
        "git tag " + "v " * 2000,
        "git gc " + "--prune=x " * 400,
        "git update-ref " + "-x " * 1300,
        "git branch " + "-" + "a" * 4000,
    ],
)
def test_adversarial_input_is_fast(command):
    start = time.perf_counter()
    analyze_command(command[:4096])
    assert time.perf_counter() - start < 1.0


def test_patterns_are_bounded():
    # no unbounded repetition of a broad class (`.*`, `\S+`, `[^...]*`): only letter runs
    # inside a single flag may use `*`, as the earlier git patterns do
    from recusal import deny_list

    for rx in (deny_list._GIT_HISTORY_LOSS, deny_list._GIT_BRANCH_FORCE):
        assert not re.search(r"(?<!\\)\.[*+]|\\S[*+]|\[\^[^\]]*\][*+]", rx.pattern), rx.pattern
