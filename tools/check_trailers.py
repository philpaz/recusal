"""Find commits that would list an AI tool or a bot as a repository contributor.

GitHub adds every ``Co-authored-by:`` address it can match to an account to the
repository's Contributors list, and coding assistants add such a line by default. This
project credits the people who contribute, and AI-assisted work is welcome, so the line
is simply removed when the change is merged; the contributor stays the author.

On a pull request this is a heads-up and never blocks (``--advisory``). On main it is
strict, so a line that reached main is noticed at once:

    python tools/check_trailers.py <rev-range>              # strict, e.g. HEAD
    python tools/check_trailers.py --advisory <rev-range>   # heads-up, always exit 0

The match is on the address, never the name, so a person named Claude, Codex or Grok is
never flagged. Exit 0 when every commit in the range is clean (or in advisory mode), 1
when any is not (each commit and line printed), 2 when git cannot list the range.
"""

import os
import re
import subprocess
import sys
from typing import List, Tuple

#: Email domains that belong to an AI vendor. Any address at one of them is refused.
#: Each is a default an assistant writes, seen in real commits on GitHub:
#: ``noreply@anthropic.com`` (Claude Code), ``noreply@openai.com`` (Codex),
#: ``cursoragent@cursor.com`` (Cursor's agent), ``noreply@x.ai`` (Grok).
AI_DOMAINS = ("anthropic.com", "openai.com", "x.ai", "cursor.com")

#: GitHub accounts of AI tools that are not ``[bot]`` accounts, matched in their noreply
#: form ``[<id>+]<login>@users.noreply.github.com``. ``Copilot`` is the login in the
#: address GitHub Copilot's coding agent commits with. Every ``[bot]`` account is
#: refused without being listed here.
AI_ACCOUNTS = ("claude", "codex", "Copilot", "cursoragent", "openhands-agent")

_AI_ADDRESS = re.compile(
    r"@(?:[a-z0-9-]+\.)*(?:" + "|".join(re.escape(d) for d in AI_DOMAINS) + r")\b(?![.-])"
    r"|(?:^|[<\s])(?:\d+\+)?(?:"
    + "|".join(re.escape(a) for a in AI_ACCOUNTS)
    + r"|[a-z0-9-]+\[bot\])@users\.noreply\.github\.com\b",
    re.IGNORECASE,
)
_COAUTHOR = re.compile(r"^\s*co-authored-by\s*:(.*)$", re.IGNORECASE | re.MULTILINE)

_FIELD = "\x1f"
_RECORD = "\x1e"

NOTE = (
    "Thanks for contributing, and AI-assisted work is welcome here. Nothing for you to\n"
    "do: the maintainer removes these lines when merging, and you stay the author.\n"
    "\n"
    "If you'd like to tidy it up yourself, for example:\n"
    "  git commit --amend          (delete the line), then git push --force-with-lease\n"
    "  git commit --amend --reset-author   (if a tool made the commit under its name)"
)


def offending_lines(author: str, committer: str, message: str) -> List[str]:
    """The reasons one commit is flagged, empty when it is clean."""
    found = []
    for who, address in (("author", author), ("committer", committer)):
        if _AI_ADDRESS.search(address):
            found.append(f"{who}: {address}")
    for match in _COAUTHOR.finditer(message):
        if _AI_ADDRESS.search(match.group(1)):
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
    advisory = argv[:1] == ["--advisory"]
    args = argv[1:] if advisory else argv
    if len(args) != 1:
        print("usage: python tools/check_trailers.py [--advisory] <rev-range>", file=sys.stderr)
        return 2
    try:
        rows = commits(args[0])
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"cannot list commits in {args[0]!r}: {exc}", file=sys.stderr)
        return 2
    bad = [(sha, lines) for sha, a, c, m in rows if (lines := offending_lines(a, c, m))]
    if not bad:
        print(f"{len(rows)} commit(s) checked: every one credits people only")
        return 0
    print(f"Heads-up: {len(bad)} of {len(rows)} commit(s) would list an AI tool or a bot")
    print("as a GitHub contributor:")
    for sha, lines in bad:
        for line in lines:
            print(f"  {sha[:12]}  {line}")
            if advisory and os.environ.get("GITHUB_ACTIONS") == "true":
                print(f"::notice title=Contributor credit::{sha[:12]} {line}")
    print()
    print(NOTE)
    return 0 if advisory else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
