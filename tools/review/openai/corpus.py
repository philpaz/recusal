"""Run a generated corpus of model output through the plain and the audited loop of
``recusal-openai`` and compare the two.

    python tools/review/openai/corpus.py --count      # how many cases; needs no SDK
    python tools/review/openai/corpus.py --summary    # run them all; one line of counts
    python tools/review/openai/corpus.py              # the same, with each finding shown
    python tools/review/openai/corpus.py --set forms  # one of the two sets
    python tools/review/openai/corpus.py --every 89   # a slice: every 89th case

What it does. A case is one turn the model could send: a Responses API response or a
Chat Completions completion carrying function calls. Every case is run twice from the
same input, once through the plain loop (``gate_function_calls`` or ``gate_tool_calls``)
and once through the audited loop (``audited_function_calls`` or ``audited_tool_calls``)
with a new audit log on disk. The tools record each time their body is entered. No
network and no API key are used: the turns are built in memory, by the SDK.

There are two sets, and no two cases in a set have the same input.

- ``forms``: turns of one call (a few of two), on Responses API responses the SDK
  validated and Chat Completions completions as received. One list of ways to write the
  arguments (numbers that are not JSON, nesting at and past the
  limit, duplicate keys, text that is not quite JSON), a list of tool names and tool
  kinds, and the status, namespace and ``finish_reason`` values one at a time.
- ``grid``: turns of two calls on responses as received (not validated). The first call
  is an ordinary allowed one. The second is every tool name in a list (known tools of
  each kind, look-alike names, names that are not strings) with every form of the
  arguments in a list (JSON text and values that are not text at all), under every
  combination of item status, namespace and response status for Responses and every
  ``finish_reason`` for Chat Completions. The plainest of those are also run with
  ``answer_tool_errors=True``. The three longest argument forms are run under the
  plainest combination only.

What a finding is. Each of these is counted, and any one makes the exit status 1:

- ``divergence``: the two loops disagree on whether the turn was answered or raised, or
  on which tool bodies ran and with what arguments;
- ``unknown_ran``: more tool bodies ran than the turn has calls whose name is exactly a
  name in the tool table;
- ``entry_count``: an answered turn in the audited loop that did not write exactly one
  entry per call, or a raised turn that wrote any (a turn raised by a tool's own
  exception has one entry for each call judged up to it, which is the documented
  behavior: the entry records the decision, not what the tool then did);
- ``entry_ids``: the entries are not under the calls' own ids, in order;
- ``answer_ids``: an answered turn whose answers are not one per call under the calls'
  own ids, in order;
- ``text``: the answer texts differ between the loops by more than the wording the two
  are documented to differ in (see ``_same_text``);
- ``raised_after_run``: a turn the gate refused by raising, after a tool body had run;
- ``long_answer`` and ``long_entry``: an answer of more than 400 characters, or the
  entries of one turn of more than 4,000, when every tool in the corpus answers ``ok``
  (so a long answer or entry would be the model's own text coming back);
- ``marker``: the text a tool or a policy raised with appears in an answer or an entry.

What it does not show. Zero findings means these cases agree on these checks, with
this SDK version. It is not a proof about inputs that are not in the corpus.
``docs/REVIEW.md`` says what the review was and what it does not show.

Exit status: 0 with no finding, 1 with any, 2 if the SDK is missing.
"""

import argparse
import functools
import hashlib
import itertools
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _openai_fixtures as fixtures  # noqa: E402

TOOL = "tools/review/openai/corpus.py"
SETS = ("forms", "grid")
MARKER = "REVIEW-MARKER-7f3a"  # the text the corpus's raising tool and policy raise with
LONG_ANSWER = 400
LONG_ENTRIES = 4000
KINDS = (
    "divergence",
    "unknown_ran",
    "entry_count",
    "entry_ids",
    "answer_ids",
    "text",
    "raised_after_run",
    "long_answer",
    "long_entry",
    "marker",
)


class Case(NamedTuple):
    """One turn. ``calls`` is a list of ``(id, name, arguments, item fields)``, or None
    for a Chat Completions message with no ``tool_calls`` field."""

    corpus: str
    label: str
    shape: str  # "responses" or "chat"
    calls: Optional[list]
    response_fields: dict  # fields of the response itself (Responses only)
    choice_fields: Optional[dict]  # fields of the choice (Chat only); None is the default
    message_fields: dict  # more fields of the message (Chat only)
    validated: bool
    keywords: dict  # keywords handed to the loop


class ToolFailure(RuntimeError):
    """What the corpus's raising tool raises."""


class Outcome(NamedTuple):
    status: str  # "answered", or "raised <class name>"
    tool_raised: bool
    ran: list
    texts: list
    ids: list
    entries: list


# --- the tools and policies the cases are run against --------------------------------------


def _forms_table(ran: list) -> tuple:
    """Tools of each kind the gate tells apart, and a deny-list policy."""

    def t(a: int = 7, b: str = "d") -> str:
        ran.append(("t", a, b))
        return "ok"

    def many(
        x: float,
        tags: List[str] = [],  # mutable defaults on purpose: the gate reads them as data
        opts: Dict[str, int] = {},
        flag: bool = False,
        n: Optional[int] = None,
    ) -> str:
        ran.append(("many", x, tags, opts, flag, n))
        return "ok"

    def gen(a: int = 1):
        ran.append(("gen",))
        yield 1

    async def co(a: int = 1):
        ran.append(("co",))

    def retgen(a: int = 1):
        ran.append(("retgen",))
        return (i for i in range(2))

    class Service:
        def m(self, a: int = 1) -> str:
            ran.append(("m",))
            return "ok"

        def g(self, a: int = 1):
            ran.append(("g",))
            yield a

    def noann(a=1):
        ran.append(("noann",))
        return "ok"

    def kw(**kwargs):
        ran.append(("kw",))
        return "ok"

    service = Service()
    tools = {
        "t": t,
        "many": many,
        "gen": gen,
        "co": co,
        "retgen": retgen,
        "m": service.m,
        "g": service.g,
        "noann": noann,
        "kw": kw,
    }

    def policy(name, args):
        from recusal import Finding

        a = args.get("a", 0)
        if isinstance(a, (int, float)) and not isinstance(a, bool) and a > 100:
            return [Finding.fail("too_big", severity="CRITICAL", message="a too big")]
        if "DROP" in json.dumps(args):
            return [Finding.fail("drop", severity="CRITICAL", message="no drop")]
        return []

    return tools, policy, []


FORMS_NAMES = ("t", "many", "gen", "co", "retgen", "m", "g", "noann", "kw")
GRID_KNOWN = (
    "t",
    "cached",
    "retiter",
    "retgen",
    "gen",
    "cgen",
    "co",
    "lco",
    "boom",
    "odd`name",
    "L" * 70,
)


def _grid_table(ran: list) -> tuple:
    """Tools of each kind, one that raises, two under names a refusal would not quote,
    and a policy that denies, allows or raises by the arguments."""

    def t(a: int = 7, b: str = "d") -> str:
        ran.append(("t", a, b))
        return "ok"

    @functools.lru_cache(maxsize=None)
    def cached(a: int = 1, b: str = "d") -> str:
        ran.append(("cached", a, b))
        return "ok"

    def retiter(a: int = 1, b: str = "d"):
        ran.append(("retiter", a, b))
        return iter([1])

    def retgen(a: int = 1, b: str = "d"):
        ran.append(("retgen", a, b))
        return (i for i in [1])

    def gen(a: int = 1, b: str = "d"):
        ran.append(("gen",))
        yield 1

    async def co(a: int = 1, b: str = "d"):
        ran.append(("co",))

    def boom(a: int = 1, b: str = "d"):
        ran.append(("boom", a, b))
        raise ToolFailure(MARKER)

    functions = {
        "t": t,
        "cached": cached,
        "retiter": retiter,
        "retgen": retgen,
        "gen": gen,
        "cgen": functools.lru_cache(maxsize=None)(gen),
        "co": co,
        "lco": functools.lru_cache(maxsize=None)(co),
        "boom": boom,
        "odd`name": t,
        "L" * 70: t,
    }
    assert tuple(functions) == GRID_KNOWN

    def policy(name, args):
        from recusal import Finding

        if args.get("a") == 101 or "DROP" in json.dumps(args):
            return [Finding.fail("deny", severity="CRITICAL", message="denied by policy")]
        if args.get("b") == "RAISE":
            raise RuntimeError("policy bug " + MARKER)
        return []

    # a cached tool is cleared between the two runs, or the second would be a cache hit
    return functions, policy, [cached.cache_clear]


# --- the forms set ---------------------------------------------------------------------------


def _deep_list(levels: int) -> str:
    return '{"a":' + "[" * (levels - 1) + "1" + "]" * (levels - 1) + "}"


def _deep_object(levels: int) -> str:
    return '{"a":' + '{"k":' * (levels - 1) + "1" + "}" * (levels - 1) + "}"


FORMS_ARGUMENTS = {
    "empty": "",
    "space": " ",
    "newline": "\n",
    "object": "{}",
    "object in spaces": " {} ",
    "allowed": '{"a": 3}',
    "denied": '{"a": 101}',
    "denied word": '{"b": "DROP"}',
    "NaN": '{"a": NaN}',
    "Infinity": '{"a": Infinity}',
    "-Infinity": '{"a": -Infinity}',
    "1e999": '{"a": 1e999}',
    "-1e999": '{"a": -1e999}',
    "NaN in a list": '{"a": [NaN]}',
    "1E400 nested": '{"b": [{"x": 1E400}]}',
    "NaN as a key": '{"NaN": 1}',
    "NaN as a string": '{"b": "NaN"}',
    "1e308": '{"a": 1e308}',
    "1e-999": '{"a": 1e-999}',
    "integer of 400 digits": '{"a": ' + "9" * 400 + "}",
    "hexadecimal": '{"a": 0x10}',
    "leading zero": '{"a": 01}',
    "plus sign": '{"a": +1}',
    "no leading digit": '{"a": .5}',
    "trailing comma": '{"a": 1,}',
    "single quotes": "{'a': 1}",
    "comment": '{"a": 1} // x',
    "two documents": '{"a": 1}{"a": 200}',
    "trailing text": '{"a": 1} x',
    "byte order mark": "\ufeff{}",
    "array": "[]",
    "string": '"{}"',
    "number": "1",
    "null": "null",
    "true": "true",
    "cut off": '{"a": 1',
    "cut off before the value": '{"a": ',
    "duplicate key, allowed then denied": '{"a": 1, "a": 200}',
    "duplicate key, denied then allowed": '{"a": 200, "a": 1}',
    "duplicate key, one escaped": '{"a": 1, "\\u0061": 200}',
    "extra argument": '{"a": 1, "zz": 2}',
    "only an extra argument": '{"zz": 2}',
    "null value": '{"a": null}',
    "string where a number goes": '{"a": "101"}',
    "list value": '{"a": [200]}',
    "object value": '{"a": {"x": 200}}',
    "boolean value": '{"a": true}',
    "63 lists deep": _deep_list(63),
    "64 lists deep": _deep_list(64),
    "65 lists deep": _deep_list(65),
    "66 lists deep": _deep_list(66),
    "64 objects deep": _deep_object(64),
    "65 objects deep": _deep_object(65),
    "300 lists deep": _deep_list(300),
    "700 lists deep": _deep_list(700),
    "3000 lists deep": _deep_list(3000),
    "100000 lists deep": _deep_list(100000),
    "100000 objects deep": _deep_object(100000),
    "100000 lists opened, none closed": '{"a":' + "[" * 100000,
    "one string of 200000 characters": json.dumps({"b": "x" * 200000}),
    "lone surrogate": '{"b": "\\ud800"}',
    "NUL": '{"b": "\\u0000"}',
    "raw control character": '{"b": "\x01"}',
    "key that is not ASCII": '{"\xe9": 1}',
    "empty key": '{"": 1}',
    "key named self": '{"self": 1}',
}
FORMS_MANY = {
    "one float": '{"x": 1.5}',
    "missing required": "{}",
    "every parameter": '{"x": 1, "tags": ["a"], "opts": {"k": 1}, "flag": true, "n": null}',
    "duplicate key": '{"x": 1, "x": 2}',
    "infinity": '{"x": 1e999}',
    "63 lists deep in a parameter": '{"x": 1, "tags": ' + "[" * 63 + "]" * 63 + "}",
    "64 lists deep in a parameter": '{"x": 1, "tags": ' + "[" * 64 + "]" * 64 + "}",
    "65 lists deep in a parameter": '{"x": 1, "tags": ' + "[" * 65 + "]" * 65 + "}",
}
FORMS_UNKNOWN = ("nope", "T", " t", "t ", "", "__class__", "t\n")
FORMS_RESPONSE_STATUS = (
    "completed",
    "in_progress",
    "incomplete",
    "failed",
    "cancelled",
    "queued",
    None,
)
FORMS_ITEM_STATUS = ("completed", "in_progress", "incomplete", None)
FORMS_NAMESPACE = ("crm", "", " ", None)
FORMS_FINISH = (
    "stop",
    "tool_calls",
    "function_call",
    "length",
    "content_filter",
    None,
    "eos",
    "",
)


def _forms() -> Iterator[Case]:
    def turn(label, shapes, calls, response_fields=None, choice=None, message=None):
        if choice is None:
            choice = {"finish_reason": "tool_calls"}
        for shape in shapes:
            yield Case(
                "forms",
                f"{label} | {shape}",
                shape,
                calls,
                (response_fields or {}) if shape == "responses" else {},
                choice if shape == "chat" else None,
                (message or {}) if shape == "chat" else {},
                shape == "responses",
                {},
            )

    both = ("responses", "chat")
    for label, arguments in FORMS_ARGUMENTS.items():
        yield from turn(f"arguments: {label}", both, [("c1", "t", arguments, {})])
    for label, arguments in FORMS_MANY.items():
        yield from turn(f"arguments of `many`: {label}", both, [("c1", "many", arguments, {})])
    for name in FORMS_NAMES + FORMS_UNKNOWN:
        yield from turn(f"name {name!a}", both, [("c1", name, "{}", {})])
        yield from turn(f"name {name!a}, arguments cut off", both, [("c1", name, "{", {})])
    for status, item in itertools.product(FORMS_RESPONSE_STATUS, FORMS_ITEM_STATUS):
        yield from turn(
            f"response status {status!r}, item status {item!r}",
            ("responses",),
            [("c1", "t", "", {"status": item})],
            response_fields={"status": status},
        )
    for namespace in FORMS_NAMESPACE:
        yield from turn(
            f"namespace {namespace!r}",
            ("responses",),
            [("c1", "t", "{}", {"namespace": namespace})],
        )
    for finish in FORMS_FINISH:
        yield from turn(
            f"finish_reason {finish!r}",
            ("chat",),
            [("c1", "t", "", {})],
            choice={"finish_reason": finish},
        )
    yield from turn("no finish_reason", ("chat",), [("c1", "t", "", {})], choice={})
    legacy = {"function_call": {"name": "t", "arguments": "{}"}}
    yield from turn(
        "legacy function_call only",
        ("chat",),
        None,
        choice={"finish_reason": "function_call"},
        message=legacy,
    )
    yield from turn(
        "legacy function_call beside tool_calls",
        ("chat",),
        [("c1", "t", "{}", {})],
        message={"function_call": {"name": "t", "arguments": '{"a": 200}'}},
    )
    yield from turn(
        "two calls, the second cut off", both, [("c1", "t", "{}", {}), ("c2", "t", "{", {})]
    )
    yield from turn(
        "two calls, the first too deep",
        both,
        [("c1", "t", _deep_list(100000), {}), ("c2", "t", "{}", {})],
    )
    yield from turn(
        "two calls under one id", both, [("c1", "t", "{}", {}), ("c1", "t", '{"a": 200}', {})]
    )
    yield from turn("empty id", both, [("", "t", "{}", {})])


# --- the grid set ----------------------------------------------------------------------------


def _nested(levels: int) -> Any:
    value: Any = 1
    for _ in range(levels):
        value = {"k": value}
    return value


GRID_NAMES = GRID_KNOWN + (
    "nope",
    "T",
    "t ",
    " t",
    "t\u200b",
    "t\n",
    "t`",
    "t\x00",
    "t\u202e",
    "",
    "x" * 64,
    "x" * 65,
    "x" * 5000,
    "\ud800",
    "<a tool name that is not a string>",
    "__class__",
    "cached ",
    "Cached",
    None,
    0,
    True,
    1.5,
    ["t"],
    {"t": 1},
)
GRID_ARGUMENTS = (
    # JSON text
    "",
    "{}",
    '{"a": 3}',
    '{"a": 101}',
    '{"b": "DROP"}',
    '{"b": "RAISE"}',
    '{"a": [1]}',
    '{"b": {"k": 1}}',
    '{"a": null}',
    '{"zz": 1}',
    "{",
    "[]",
    '{"a": NaN}',
    '{"a": 1e999}',
    '{"a":' + "[" * 70 + "]" * 70 + "}",
    '{"a":' + "[" * 100000,
    '{"b": "\ud800"}',
    '{"a": 1, "a": 101}',
    '{"a": 101, "a": 1}',
    '{"a": ' + "9" * 4000 + "}",
    '{"a": ' + "9" * 5000 + "}",
    '{"a": 1e308}',
    '{"a": -0.0}',
    '{"b": "\u0000"}',
    '{"a": true}',
    ' {"a": 3} ',
    '{"a": 3}\n{"a": 101}',
    "\ufeff{}",
    '{"__class__": 1}',
    '{"a":' + "[" * 63 + "]" * 63 + "}",
    '{"a":' + "[" * 64 + "]" * 64 + "}",
    '{"a":' + "[" * 65 + "]" * 65 + "}",
    '{"a": 3, "b": "\u202e"}',
    '{"a": -Infinity}',
    '{"a": 0x10}',
    "{'a': 3}",
    '{"a": 3,}',
    "null",
    '"t"',
    "3",
    "true",
    # values that are not text: only a response the SDK did not validate can carry them
    None,
    0,
    True,
    1.5,
    float("nan"),
    b"{}",
    ["x"],
    (1, 2),
    {"a": 3},
    {"a": float("nan")},
    {"a": float("inf")},
    {"a": b"x"},
    {1: 2},
    _nested(64),
    _nested(65),
    _nested(66),
)
GRID_ITEM = (
    {},
    {"namespace": "crm"},
    {"status": "in_progress"},
    {"status": "completed"},
    {"status": "weird_future"},
)
GRID_RESPONSE = ({}, {"status": "incomplete"}, {"status": "completed"})
GRID_FINISH = ("tool_calls", "stop", "length", None, "future_reason")
GRID_KEYWORDS = ({}, {"answer_tool_errors": True})
GRID_LONG = 1000  # an argument form longer than this runs under the plainest combination only


def _short(value: Any) -> str:
    text = str(value)
    return ascii(text) if len(text) <= 24 else f"{ascii(text[:24])}...({len(text)})"


def _grid() -> Iterator[Case]:
    first = ("c0", "t", '{"a": 1}', {})
    contexts = [
        ("responses", item, top, None) for item, top in itertools.product(GRID_ITEM, GRID_RESPONSE)
    ] + [("chat", {}, {}, finish) for finish in GRID_FINISH]
    plainest = {"responses": contexts[0], "chat": contexts[len(GRID_ITEM) * len(GRID_RESPONSE)]}
    for shape in ("responses", "chat"):
        for name, arguments, context, keywords in itertools.product(
            GRID_NAMES, GRID_ARGUMENTS, contexts, GRID_KEYWORDS
        ):
            _, item, top, finish = context
            if context[0] != shape:
                continue
            if keywords and (item or top or finish not in (None, "tool_calls")):
                continue
            if len(str(arguments)) > GRID_LONG and context is not plainest[shape]:
                continue
            where = f"item={item} response={top}" if shape == "responses" else f"finish={finish!r}"
            yield Case(
                "grid",
                f"{shape} name={_short(name)} arguments={_short(arguments)} {where} {keywords}",
                shape,
                [first, ("c1", name, arguments, item)],
                top,
                {"finish_reason": finish} if shape == "chat" else None,
                {},
                False,
                keywords,
            )


# --- the cases, counted ----------------------------------------------------------------------


def _fingerprint(case: Case) -> bytes:
    """What makes two cases the same input: everything but the label."""
    return hashlib.sha256(repr(case[2:]).encode("utf-8")).digest()


def cases(which: str = "all") -> Iterator[Case]:
    """Every case of the chosen set, in a fixed order, each input once."""
    for name, generate in (("forms", _forms), ("grid", _grid)):
        if which not in ("all", name):
            continue
        seen = set()
        for case in generate():
            key = _fingerprint(case)
            if key not in seen:
                seen.add(key)
                yield case


def counts() -> dict:
    """How many cases each set has. Needs no SDK: nothing is run."""
    found = {name: sum(1 for _ in cases(name)) for name in SETS}
    return {"tool": TOOL, **found, "cases": sum(found.values())}


# --- running one case and reading the result -------------------------------------------------


def _once(sdk, logs, case: Case, table: tuple, ran: list, audited: bool) -> Outcome:
    tools, policy, resets = table
    ran.clear()
    for reset in resets:
        reset()
    if case.shape == "responses":
        items = [
            fixtures.function_call(call_id, name, arguments, **item)
            for call_id, name, arguments, item in case.calls or []
        ]
        turn = fixtures.response(sdk, items, case.validated, **case.response_fields)
        loops = (sdk.gate_function_calls, sdk.audited_function_calls)
        text_key, id_key = "output", "call_id"
    else:
        calls = None if case.calls is None else [call[:3] for call in case.calls]
        turn = fixtures.completion(
            sdk, calls, case.validated, case.choice_fields, **case.message_fields
        )
        loops = (sdk.gate_tool_calls, sdk.audited_tool_calls)
        text_key, id_key = "content", "tool_call_id"
    path = None
    tool_raised = False
    try:
        if audited:
            log, path = logs.new()
            answers = loops[1](turn, tools, policy, log, **case.keywords)
        else:
            answers = loops[0](turn, tools, policy, **case.keywords)
        status = "answered"
        texts = [answer[text_key] for answer in answers]
        ids = [answer[id_key] for answer in answers]
    except Exception as exc:  # noqa: BLE001, a raised turn is a result, not an error here
        status, texts, ids = f"raised {type(exc).__name__}", [], []
        tool_raised = isinstance(exc, ToolFailure)
    entries = logs.entries(path) if path is not None else []
    return Outcome(status, tool_raised, list(ran), texts, ids, entries)


def _same_text(plain: str, audited: str) -> bool:
    """Whether the two loops gave the model the same answer.

    They are documented to word a refusal differently in two places, and those are set
    aside here: the audited loop's reason is the one on the record, which carries the
    finding's verdict tag (`` [FAIL]``), and the plain loop words an unknown tool as
    ``Recusal refused: unknown tool 'name'`` where the record says
    ``Recusal refused `name`: unknown tool``.
    """

    def normal(text: str) -> str:
        text = text.replace(" [FAIL]", "")
        unknown = re.fullmatch(r"Recusal refused: unknown tool '(.*)'", text, re.S)
        return f"Recusal refused `{unknown.group(1)}`: unknown tool" if unknown else text

    return normal(plain) == normal(audited)


def check(case: Case, tools: dict, plain: Outcome, audited: Outcome) -> list:
    """The findings for one case, as ``(kind, detail)``. An empty list is none."""
    found = []
    calls = case.calls or []
    expected = [call[0] for call in calls]
    known = sum(1 for call in calls if isinstance(call[1], str) and call[1] in tools)
    if plain.status != audited.status or repr(plain.ran) != repr(audited.ran):
        found.append(
            (
                "divergence",
                f"plain {plain.status} {plain.ran!a}, audited {audited.status} {audited.ran!a}",
            )
        )
    elif audited.status == "answered":
        if plain.ids != expected or audited.ids != expected:
            found.append(("answer_ids", f"plain {plain.ids!a}, audited {audited.ids!a}"))
        if len(audited.entries) != len(calls):
            found.append(("entry_count", f"{len(audited.entries)} entries, {len(calls)} calls"))
        else:
            recorded = [entry["action"].get("tool_use_id") for entry in audited.entries]
            if recorded != expected:
                found.append(("entry_ids", f"{recorded!a}"))
        if len(plain.texts) != len(audited.texts) or not all(
            _same_text(one, other) for one, other in zip(plain.texts, audited.texts)
        ):
            found.append(("text", f"plain {plain.texts!a}, audited {audited.texts!a}"))
    else:
        # a tool's own exception leaves the turn after each call up to it was judged and
        # recorded; in this corpus every call before the raising one is allowed and runs
        wanted = len(audited.ran) if audited.tool_raised else 0
        if len(audited.entries) != wanted:
            found.append(
                ("entry_count", f"raised, {len(audited.entries)} entries, {wanted} expected")
            )
        if not audited.tool_raised and (plain.ran or audited.ran):
            found.append(("raised_after_run", f"{audited.status} after {audited.ran!a}"))
    for outcome in (plain, audited):
        if len(outcome.ran) > known:
            found.append(("unknown_ran", f"{outcome.ran!a}"))
        for text in outcome.texts:
            if len(text) > LONG_ANSWER:
                found.append(("long_answer", f"{len(text)} characters"))
            if MARKER in text:
                found.append(("marker", f"in an answer: {text[:120]!a}"))
    if audited.entries:
        written = json.dumps(audited.entries)
        if len(written) > LONG_ENTRIES:
            found.append(("long_entry", f"{len(written)} characters"))
        if MARKER in written:
            found.append(("marker", "in an entry"))
    return found


def run(which: str = "all", every: int = 1, offset: int = 0, show: int = 0) -> dict:
    """Run the chosen cases through both loops and return the counts.

    ``every`` and ``offset`` take a slice: the cases whose position, counted from 0
    across the chosen sets, leaves ``offset`` when divided by ``every``. With ``show``,
    up to that many findings of each kind are printed as they are found.
    """
    if every < 1 or not 0 <= offset < every:
        raise ValueError("every must be at least 1 and offset must be from 0 to every - 1")
    sdk = fixtures.load()
    # the gate logs the traceback of a tool that raised; here one raises on purpose
    logger, quiet = logging.getLogger("recusal_openai"), logging.NullHandler()
    logger.addHandler(quiet)
    propagate, logger.propagate = logger.propagate, False
    ran: list = []
    tables = {"forms": _forms_table(ran), "grid": _grid_table(ran)}
    summary = {"tool": TOOL, "set": which, "slice": f"{offset}/{every}"}
    summary.update({name: 0 for name in SETS})
    summary.update({"cases": 0, "loop_runs": 0, "answered": 0, "raised": 0, "raised_by_a_tool": 0})
    findings = {kind: 0 for kind in KINDS}
    try:
        with fixtures.Logs(sdk) as logs:
            for position, case in enumerate(cases(which)):
                if position % every != offset:
                    continue
                table = tables[case.corpus]
                plain = _once(sdk, logs, case, table, ran, audited=False)
                audited = _once(sdk, logs, case, table, ran, audited=True)
                summary[case.corpus] += 1
                summary["cases"] += 1
                summary["loop_runs"] += 2
                summary["answered" if audited.status == "answered" else "raised"] += 1
                summary["raised_by_a_tool"] += audited.tool_raised
                for kind, detail in check(case, table[0], plain, audited):
                    findings[kind] += 1
                    if findings[kind] <= show:
                        print(f"FINDING {kind}: {case.corpus} {case.label}\n    {detail[:400]}")
    finally:
        logger.propagate = propagate
        logger.removeHandler(quiet)
    summary.update(findings)
    summary["findings"] = sum(findings.values())
    summary.update(sdk.versions)
    return summary


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a generated corpus of model output through the plain and the audited "
            "loop of recusal-openai and compare them. Prints one JSON line of counts; "
            "exits 1 if there is any finding."
        ),
        epilog="See the top of this file for what each count is, and docs/REVIEW.md.",
    )
    parser.add_argument(
        "--set", choices=("all",) + SETS, default="all", help="which cases (default: all)"
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="print how many cases there are and run nothing (needs no SDK)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="print the line of counts and nothing else"
    )
    parser.add_argument(
        "--every", type=int, default=1, metavar="N", help="run every Nth case only (a slice)"
    )
    parser.add_argument(
        "--offset", type=int, default=0, metavar="K", help="with --every, start at case K"
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
    summary = run(
        options.set, options.every, options.offset, 0 if options.summary else options.show
    )
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
