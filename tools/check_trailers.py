"""Refuse commits that would list an Anthropic account as a repository contributor.

GitHub adds every ``Co-authored-by:`` address it can match to an account to the
repository's Contributors list, and coding assistants add such a trailer by default.
This project credits people, so a commit that names an Anthropic address (as a
co-author, author or committer) is refused here, in CI, before it can merge. AI-assisted
contributions are welcome; only the trailer that turns the tool into a listed
contributor is not.

The match is on the address, not the word "Claude", so a human co-author named Claude
is never refused:

    python tools/check_trailers.py <rev-range>     # e.g. origin/main..HEAD, or HEAD

Exit 0 when every commit in the range is clean, 1 when any is not (each offending
commit and line printed, with the fix), 2 when git cannot list the range.
"""

import re
import subprocess
import sys
from typing import List, Tuple

#: Addresses GitHub resolves to an Anthropic account: anything at anthropic.com
#: (``noreply@anthropic.com`` is the one coding assistants write) and the noreply
#: address of the ``claude`` GitHub account.
_ANTHROPIC_ADDRESS = re.compile(
    r"@anthropic\.com\b|(?:^|[<\s+])claude@users\.noreply\.github\.com\b", re.IGNORECASE
)
_COAUTHOR = re.compile(r"^\s*co-authored-by\s*:(.*)$", re.IGNORECASE | re.MULTILINE)

_FIELD = "\x1f"
_RECORD = "\x1e"

FIX = (
    "Remove the trailer and force-push your branch, for example:\n"
    "  git rebase -i origin/main   (mark each listed commit 'reword', delete the line)\n"
    "or, for the last commit only:\n"
    "  git commit --amend          (delete the line), then git push --force-with-lease\n"
    "Your AI-assisted work is welcome; only the co-author trailer is not."
)


def offending_lines(author: str, committer: str, message: str) -> List[str]:
    """The reasons one commit is refused, empty when it is clean."""
    found = []
    for who, address in (("author", author), ("committer", committer)):
        if _ANTHROPIC_ADDRESS.search(address):
            found.append(f"{who}: {address}")
    for match in _COAUTHOR.finditer(message):
        if _ANTHROPIC_ADDRESS.search(match.group(1)):
            found.append(match.group(0).strip())
    return found


def commits(rev_range: str) -> List[Tuple[str, str, str, str]]:
    """(sha, author, committer, message) for every commit in ``rev_range``."""
    fmt = _FIELD.join(["%H", "%an <%ae>", "%cn <%ce>", "%B"]) + _RECORD
    out = subprocess.run(
        ["git", "log", f"--format={fmt}", rev_range],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    rows = []
    for record in out.split(_RECORD):
        record = record.strip("\n")
        if record:
            sha, author, committer, message = record.split(_FIELD, 3)
            rows.append((sha, author, committer, message))
    return rows


def main(argv: List[str]) -> int:
    if len(argv) != 1:
        print("usage: python tools/check_trailers.py <rev-range>", file=sys.stderr)
        return 2
    try:
        rows = commits(argv[0])
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"cannot list commits in {argv[0]!r}: {exc}", file=sys.stderr)
        return 2
    bad = [(sha, lines) for sha, a, c, m in rows if (lines := offending_lines(a, c, m))]
    if not bad:
        print(f"{len(rows)} commit(s) checked: no Anthropic co-author, author or committer")
        return 0
    print(f"{len(bad)} of {len(rows)} commit(s) would list an Anthropic account as a contributor:")
    for sha, lines in bad:
        for line in lines:
            print(f"  {sha[:12]}  {line}")
    print()
    print(FIX)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
