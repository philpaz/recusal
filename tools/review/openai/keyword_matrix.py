"""Call the three entry points of ``recusal-openai`` with every combination of the
recording keywords and check what ran and what was written.

    python tools/review/openai/keyword_matrix.py --count      # how many; needs no SDK
    python tools/review/openai/keyword_matrix.py --summary    # run them all; one line
    python tools/review/openai/keyword_matrix.py              # the same, findings shown
    python tools/review/openai/keyword_matrix.py --every 53   # a slice of the keyword sets

What it does. The recording keywords are ``audit``, ``surface``, ``actor``, ``control``,
``allow_on_pass`` and ``tool_use_id``. Each is given every value in a short list: not
passed at all, ``None``, an ordinary value, and values the kernel does not record (an
empty or padded surface, a reserved one, a value of the wrong type), with a call id at
the gate's bound of 256 characters and one past it. Every combination of those values is one keyword set.
``gate_function_calls`` and ``gate_tool_calls`` are called with each of their sets on a
turn of two ordinary calls, and ``judge_and_run`` with each of its sets on one call. The
policy allows everything and the tool is an ordinary function, so nothing but the
keywords can refuse a call. An audit log, where one is passed, is a real log on disk
with a counter in front of its ``append``. No network and no API key are used.

What a finding is. Each of these is counted, and any one makes the exit status 1:

- ``raised_after_run``: the call raised, and a tool body had already run;
- ``raised_but_recorded``: the call raised, and an entry was appended or is on file;
- ``answered_not_run``: the call returned answers, and a tool body did not run. The
  policy allows everything, so this is a keyword turning into a refusal of the call
  instead of an error the caller sees;
- ``entries``: with a log, an answered turn where the appends, or the entries on file,
  are not exactly one per call;
- ``loops_differ``: the same keyword set is answered by one of the two loops and raises
  in the other, or raises a different exception class.

What it does not show. It says nothing about which combinations should raise: that is
the README's contract and the adapter's tests pin it case by case. Zero findings means
no combination in these lists ran a tool without the record the caller asked for, or
refused one silently. It is not a proof about values that are not in the lists.
``docs/REVIEW.md`` says what the review was and what it does not show.

Exit status: 0 with no finding, 1 with any, 2 if the SDK is missing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _openai_fixtures as fixtures  # noqa: E402

TOOL = "tools/review/openai/keyword_matrix.py"
ABSENT = "<not passed>"  # a keyword left out of the call; no list below holds this text
LOG = "<an audit log>"  # stands for a new log on disk, made for each call
KEYWORDS = ("audit", "surface", "actor", "control", "allow_on_pass", "tool_use_id")
AUDIT = (ABSENT, None, LOG)
SURFACE = (
    ABSENT,
    None,
    "my.surface",
    "",
    " x",
    "x ",
    5,
    "claude_code.pretooluse",
    "openai.keyword_check",
    ["s"],
)
ACTOR = (ABSENT, None, "alice", 5)
CONTROL = (ABSENT, None, {}, {"k": 1}, "v1", ["a"], 0)
ALLOW_ON_PASS = (ABSENT, True, False)
CALL_ID_LOOPS = (ABSENT, None, "call_1")  # the loops take no call id of the caller's
CALL_ID_JUDGE = (ABSENT, None, "call_1", "", 5, "x" * 256, "x" * 257, "\n`")
KINDS = (
    "raised_after_run",
    "raised_but_recorded",
    "answered_not_run",
    "entries",
    "loops_differ",
)


def keyword_sets(entry: str) -> Iterator[dict]:
    """Every keyword set for ``entry``: "loops" (both loops take the same) or "judge"."""
    call_ids = CALL_ID_JUDGE if entry == "judge" else CALL_ID_LOOPS
    for values in itertools.product(AUDIT, SURFACE, ACTOR, CONTROL, ALLOW_ON_PASS, call_ids):
        yield {
            keyword: value
            for keyword, value in zip(KEYWORDS, values)
            if not (isinstance(value, str) and value == ABSENT)
        }


def counts() -> dict:
    """How many combinations there are. Needs no SDK: nothing is run."""
    loops = sum(1 for _ in keyword_sets("loops"))
    judge = sum(1 for _ in keyword_sets("judge"))
    return {
        "tool": TOOL,
        "gate_function_calls": loops,
        "gate_tool_calls": loops,
        "judge_and_run": judge,
        "combinations": 2 * loops + judge,
    }


class _Counted:
    """A real audit log with a count of the appends made to it."""

    def __init__(self, log: Any) -> None:
        self._log = log
        self.appends = 0

    def append(self, *args: Any, **kwargs: Any) -> Any:
        self.appends += 1
        return self._log.append(*args, **kwargs)


def _allow_all(_name: str, _arguments: dict) -> list:
    return []


def _call(sdk: Any, logs: Any, entry: str, keywords: dict) -> tuple:
    """One call of one entry point: ``(status, bodies run, appends, entries on file)``."""
    ran: list = []

    def t(a: int = 7) -> str:
        ran.append(a)
        return "ok"

    passed = dict(keywords)
    counted, path = None, None
    if isinstance(passed.get("audit"), str) and passed["audit"] == LOG:
        log, path = logs.new()
        counted = passed["audit"] = _Counted(log)
    two = [("c1", "t", '{"a": 1}'), ("c2", "t", '{"a": 2}')]
    try:
        if entry == "gate_function_calls":
            items = [fixtures.function_call(*call) for call in two]
            turn = fixtures.response(sdk, items, validated=True)
            sdk.gate_function_calls(turn, {"t": t}, _allow_all, **passed)
        elif entry == "gate_tool_calls":
            turn = fixtures.completion(sdk, two, validated=True)
            sdk.gate_tool_calls(turn, {"t": t}, _allow_all, **passed)
        else:
            sdk.judge_and_run("t", '{"a": 1}', {"t": t}, _allow_all, **passed)
        status = "answered"
    except Exception as exc:  # noqa: BLE001, a raised call is a result, not an error here
        status = f"raised {type(exc).__name__}"
    on_file = len(logs.entries(path)) if path is not None else 0
    return status, len(ran), counted.appends if counted else 0, on_file


def check(entry: str, status: str, ran: int, appends: int, on_file: int, logged: bool) -> list:
    """The kinds of finding for one call. An empty list is none."""
    calls = 1 if entry == "judge_and_run" else 2
    found = []
    if status != "answered":
        if ran:
            found.append("raised_after_run")
        if appends or on_file:
            found.append("raised_but_recorded")
        return found
    if ran != calls:
        found.append("answered_not_run")
    if logged and (appends != calls or on_file != calls):
        found.append("entries")
    return found


def run(every: int = 1, offset: int = 0, show: int = 0) -> dict:
    """Run the combinations and return the counts.

    ``every`` and ``offset`` take a slice of the keyword sets: those whose position,
    counted from 0 (the loops' sets first, then ``judge_and_run``'s), leaves ``offset``
    when divided by ``every``. Both loops are called with each of their sets in the slice.
    """
    if every < 1 or not 0 <= offset < every:
        raise ValueError("every must be at least 1 and offset must be from 0 to every - 1")
    sdk = fixtures.load()
    summary: dict = {"tool": TOOL, "slice": f"{offset}/{every}"}
    summary.update({"gate_function_calls": 0, "gate_tool_calls": 0, "judge_and_run": 0})
    summary.update({"combinations": 0, "answered": 0, "raised": 0})
    findings = {kind: 0 for kind in KINDS}

    def report(kind: str, entry: str, keywords: dict, detail: str) -> None:
        findings[kind] += 1
        if findings[kind] <= show:
            print(f"FINDING {kind}: {entry} {keywords!a}\n    {detail}")

    position = 0
    with fixtures.Logs(sdk) as logs:
        for group in ("loops", "judge"):
            entries = (
                ("gate_function_calls", "gate_tool_calls")
                if group == "loops"
                else ("judge_and_run",)
            )
            for keywords in keyword_sets(group):
                position += 1
                if (position - 1) % every != offset:
                    continue
                statuses = []
                for entry in entries:
                    status, ran, appends, on_file = _call(sdk, logs, entry, keywords)
                    statuses.append(status)
                    summary[entry] += 1
                    summary["combinations"] += 1
                    summary["answered" if status == "answered" else "raised"] += 1
                    logged = keywords.get("audit") == LOG
                    for kind in check(entry, status, ran, appends, on_file, logged):
                        detail = f"{status}; {ran} ran, {appends} appends, {on_file} on file"
                        report(kind, entry, keywords, detail)
                if len(set(statuses)) > 1:
                    report("loops_differ", "both loops", keywords, " / ".join(statuses))
    summary.update(findings)
    summary["findings"] = sum(findings.values())
    summary.update(sdk.versions)
    return summary


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Call gate_function_calls, gate_tool_calls and judge_and_run with every "
            "combination of the recording keywords and check what ran and what was "
            "written. Prints one JSON line of counts; exits 1 if there is any finding."
        ),
        epilog="See the top of this file for what each count is, and docs/REVIEW.md.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="print how many combinations there are and run nothing (needs no SDK)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="print the line of counts and nothing else"
    )
    parser.add_argument(
        "--every", type=int, default=1, metavar="N", help="run every Nth keyword set only"
    )
    parser.add_argument(
        "--offset", type=int, default=0, metavar="K", help="with --every, start at set K"
    )
    parser.add_argument(
        "--show", type=int, default=5, metavar="N", help="findings to print per kind (default 5)"
    )
    options = parser.parse_args(argv)
    if options.count:
        print(json.dumps(counts()))
        return 0
    if options.every < 1 or not 0 <= options.offset < options.every:
        parser.error("--every must be at least 1 and --offset from 0 to N - 1")
    summary = run(options.every, options.offset, 0 if options.summary else options.show)
    print(json.dumps(summary))
    return 1 if summary["findings"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as stop:
        if isinstance(stop.code, str):
            print(stop.code, file=sys.stderr)
            sys.exit(2)
        raise
