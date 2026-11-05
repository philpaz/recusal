"""Keep the repository's clone and view traffic beyond GitHub's 14-day window.

GitHub's traffic API returns per-day clones and views for the last 14 days only; a day
that leaves the window is gone from GitHub. This script reads the window and merges it
into two CSV files kept on a data branch (``traffic`` by default), one row per day, so
the record accumulates for as long as the schedule runs. A day already on record is
overwritten by the window's value for it, because GitHub revises the most recent days
(the current day is partial until it ends), so the window is always the better source.

The files are written through the contents API, never by a git push: the workflow that
runs this holds no git credentials, and the commits it makes land on the data branch
only, so the default branch's contributor list never carries a bot.

Nothing here estimates or back-fills. The record begins at the first captured window,
and the branch's README says so.

    python tools/traffic_record.py --repo philpaz/recusal            # capture
    python tools/traffic_record.py --repo philpaz/recusal --dry-run  # print, write nothing

``GITHUB_TOKEN`` must be set; the traffic endpoints need push access to the repository.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Callable, Dict, Optional, Tuple

API = "https://api.github.com"
KINDS = ("clones", "views")
HEADER = "date,count,uniques"
Row = Tuple[int, int]  # (count, uniques) for one day
Table = Dict[str, Row]  # ISO date -> row


# --- the pure part ----------------------------------------------------------------------------


def parse_csv(text: str) -> Table:
    """Read ``date,count,uniques`` rows; a blank text or a header alone is an empty table."""
    table: Table = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line == HEADER:
            continue
        date, count, uniques = line.split(",")
        table[date] = (int(count), int(uniques))
    return table


def window_rows(payload: dict, kind: str) -> Table:
    """The per-day rows of one traffic API response, keyed by ISO date."""
    return {
        item["timestamp"][:10]: (int(item["count"]), int(item["uniques"]))
        for item in payload.get(kind, [])
    }


def merge(existing: Table, window: Table) -> Table:
    """Every day on record, with the window's value winning for the days it covers."""
    merged = dict(existing)
    merged.update(window)
    return merged


def render(table: Table) -> str:
    lines = [HEADER] + [f"{d},{c},{u}" for d, (c, u) in sorted(table.items())]
    return "\n".join(lines) + "\n"


def summary(tables: Dict[str, Table]) -> str:
    """A Markdown summary of what the record holds. Daily uniques are summed as the
    upper bound they are: the same cloner on two days counts twice."""
    out = ["# Totals since capture began", ""]
    for kind in KINDS:
        table = tables.get(kind, {})
        if not table:
            out.append(f"- **{kind}**: no days on record")
            continue
        days = sorted(table)
        count = sum(c for c, _ in table.values())
        uniques = sum(u for _, u in table.values())
        out.append(
            f"- **{kind}**: {count} over {len(days)} days, {days[0]} to {days[-1]}; "
            f"sum of daily uniques {uniques} (an upper bound, not distinct actors)"
        )
    out.append("")
    out.append("Rows: `clones.csv`, `views.csv`. Captured by `.github/workflows/traffic.yml`.")
    return "\n".join(out) + "\n"


# --- the GitHub part ---------------------------------------------------------------------------


Request = Callable[[str, str, Optional[dict]], Tuple[int, dict]]


def github_request(token: str) -> Request:
    def call(method: str, path: str, body: Optional[dict] = None) -> Tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(API + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8") or "{}")

    return call


def read_file(request: Request, repo: str, branch: str, path: str) -> Tuple[str, Optional[str]]:
    """The file's text and blob sha on the branch, or ``("", None)`` when absent."""
    status, payload = request("GET", f"/repos/{repo}/contents/{path}?ref={branch}")
    if status == 404:
        return "", None
    if status != 200:
        raise RuntimeError(f"GET {path}@{branch}: HTTP {status} {payload.get('message', '')}")
    return base64.b64decode(payload["content"]).decode("utf-8"), payload["sha"]


def write_file(
    request: Request, repo: str, branch: str, path: str, text: str, sha: Optional[str], message: str
) -> None:
    body = {
        "message": message,
        "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha is not None:
        body["sha"] = sha
    status, payload = request("PUT", f"/repos/{repo}/contents/{path}", body)
    if status not in (200, 201):
        raise RuntimeError(f"PUT {path}@{branch}: HTTP {status} {payload.get('message', '')}")


def capture(request: Request, repo: str, branch: str, dry_run: bool, out=sys.stdout) -> int:
    """Merge the current window into the branch's files. Returns the number of files written."""
    tables: Dict[str, Table] = {}
    shas: Dict[str, Optional[str]] = {}
    changed = []
    for kind in KINDS:
        status, payload = request("GET", f"/repos/{repo}/traffic/{kind}")
        if status != 200:
            raise RuntimeError(
                f"GET traffic/{kind}: HTTP {status} {payload.get('message', '')} "
                "(the token needs push access to the repository)"
            )
        window = window_rows(payload, kind)
        text, shas[kind] = read_file(request, repo, branch, f"{kind}.csv")
        before = parse_csv(text)
        tables[kind] = merge(before, window)
        if tables[kind] != before:
            changed.append(kind)
        print(
            f"{kind}: window {min(window) if window else '-'}..{max(window) if window else '-'} "
            f"({len(window)} days), on record {len(before)} -> {len(tables[kind])} days",
            file=out,
        )
    if not changed:
        print("nothing new; files unchanged", file=out)
        return 0
    last = max(d for t in tables.values() for d in t)
    if dry_run:
        for kind in changed:
            print(f"--- {kind}.csv (dry run)\n{render(tables[kind])}", file=out)
        print(summary(tables), file=out)
        return 0
    for kind in changed:
        write_file(
            request,
            repo,
            branch,
            f"{kind}.csv",
            render(tables[kind]),
            shas[kind],
            f"traffic: {kind} through {last}",
        )
    text, sha = read_file(request, repo, branch, "SUMMARY.md")
    write_file(
        request, repo, branch, "SUMMARY.md", summary(tables), sha, f"traffic: totals through {last}"
    )
    return len(changed) + 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--branch", default="traffic", help="data branch (default: traffic)")
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = parser.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("GITHUB_TOKEN is not set", file=sys.stderr)
        return 2
    try:
        capture(github_request(token), args.repo, args.branch, args.dry_run)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
