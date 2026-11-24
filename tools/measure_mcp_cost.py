"""Measure what the gate's check of a call's arguments costs, on this machine.

The README of this package quotes the output of this script. It times
``Listed.complete`` (the validation, the two refusals and the defaults, everything the
gate does with a listed schema before the policy runs) on arguments of several sizes
and shapes, counts the steps it spent of the steps it was allowed, and, where a
Pydantic type publishes the schema, times Pydantic's own validation of the same
arguments, which is what a server built with the MCP Python SDK runs. The arguments
are decoded from JSON text first, as the SDK hands them to the gate: a list built in
Python as ``[0.1] * 10_000`` is one number object ten thousand times, which the gate
checks once, and the same list decoded from text is ten thousand objects.

The check of one call stops at a ceiling of steps (``STEP_CEILING``), which an
application can raise for a connection (``step_ceiling=``). So each shape is measured
three ways: at a size every shape completes, at the largest size the default ceiling
completes (found by trying sizes), and at larger sizes with the ceiling raised, which is
what raising it costs.

The steps do not bound the size of a call, which has a ceiling of its own
(``SIZE_CEILING``, ``size_ceiling=``) that is checked where the SDK hands the gate the
call, before ``Listed.complete``. So the last part times the whole of that:
``call_tool`` on a gated server object, without and with an audit log, for the largest
calls of the first part and for the arguments that take longest at the ceiling on the
size.

On Python 3.11 and later one more thing moves the time of a call, which no best of
three shows: where in the interpreter's stack the call is made. CPython keeps frames in
blocks of 16 KB, and a Python function called where the block ends maps a new block and
frees it again on return, at every call. So the same call takes longer from some depths
of the application's own stack than from others, and a call whose check recurses (a
recursive schema, the fingerprint of an audit entry) takes longer at some depths of its
arguments' nesting. ``stack`` times four calls of the last part from each of 144
depths of plain Python frames, once each, and prints the least, the median and the
most: the range the page states. It ends with the other thing the place decides: the
validation recurses with a recursive schema, so the deepest call such a schema admits
completes only where the interpreter's recursion limit leaves room for it, and
``stack`` prints from how many frames down that stops.

    python tools/measure_mcp_cost.py
    python tools/measure_mcp_cost.py size     (the last part alone)
    python tools/measure_mcp_cost.py stack    (the range by where the call is made)

It needs the package and its dependencies installed. Times differ from machine to
machine and run to run; each is the best of three runs, made from three places in the
stack, but for ``stack``, which says so.
Run it on a machine that is doing nothing else: every figure is of one thread.
"""

from __future__ import annotations

import asyncio
import gc
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional, Union

from agents.mcp import MCPServer
from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult, TextContent, Tool
from pydantic import BaseModel, Field, TypeAdapter, create_model
from recusal_openai_agents_mcp import (
    SIZE_CEILING,
    STEP_CEILING,
    _gate,
    _schema,
    gate_mcp_server,
    recording,
)
from recusal_openai_agents_mcp._schema import Listed, Refused
from typing_extensions import TypeAliasType

from recusal.audit import AuditLog

#: Every multiple of this has the hash 0: Python reduces an integer by it.
ONE_HASH = 2**61 - 1
#: A ceiling no call here reaches: what a call costs when the application raised it.
RAISED = 10**12


class Item(BaseModel):
    path: str
    content: str
    mode: str = "x"
    flag: bool = False


class Node(BaseModel):
    v: int = 1
    kids: list["Node"] = []
    kid: Optional["Node"] = None


#: Pydantic's documented recursive JSON type, its scalars first.
Json = TypeAliasType(  # type: ignore[misc]
    "Json", "Union[str, int, float, bool, None, list[Json], dict[str, Json]]"
)


def chain(count: int) -> dict[str, Any]:
    """A ``Node`` whose ``kid`` leads 60 nodes down to one that is not valid (``v`` is a
    string) and holds ``count`` kids."""
    node: dict[str, Any] = {"v": "x", "kids": [{} for _ in range(count)]}
    for _ in range(60):
        node = {"kid": node}
    return node


def nested(depth: int, leaf: Any) -> Any:
    for _ in range(depth):
        leaf = [leaf]
    return leaf


class Server(MCPServer):
    """A server object that lists one tool and runs nothing."""

    def __init__(self, schema: dict[str, Any]):
        super().__init__()
        self.schema = schema

    @property
    def name(self) -> str:
        return "measured"

    async def connect(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def list_tools(self, run_context: Any = None, agent: Any = None) -> Any:
        return [Tool(name="t", description="t", inputSchema=self.schema)]

    async def call_tool(self, tool_name: str, arguments: Any, meta: Any = None) -> Any:
        return CallToolResult(content=[TextContent(type="text", text="ran")])

    async def list_prompts(self) -> Any:
        return ListPromptsResult(prompts=[])

    async def get_prompt(self, name: str, arguments: Any = None) -> Any:
        return GetPromptResult(messages=[])


def seam(
    label: str,
    schema: dict[str, Any],
    arguments: dict[str, Any],
    folder: str,
    alone: bool = True,
    places: int = 0,
) -> None:
    """One call through ``call_tool`` on a gated server object, without and with an
    audit log, beside ``Listed.complete`` alone (unless ``alone`` is false: arguments
    the gate refuses before the check): the best of three each, to ``places`` decimal
    places."""
    arguments = decoded(arguments)
    size = len(json.dumps(arguments, separators=(",", ":"), ensure_ascii=False))
    line = f"{label}: {size:,} characters"
    if alone and size <= SIZE_CEILING:
        line += f": Listed.complete {gate(schema, arguments)[0]:,.0f} ms"
    for logged in (False, True):
        server = Server(schema)
        if logged:
            log = AuditLog(path=str(Path(tempfile.mkdtemp(dir=folder)) / "log.jsonl"))
            gate_mcp_server(server, lambda *_: [], recording(log))
        else:
            gate_mcp_server(server, lambda *_: [])
        asyncio.run(server.list_tools())
        outcome = ""

        def run() -> None:
            nonlocal outcome
            outcome = asyncio.run(server.call_tool("t", arguments)).content[0].text

        spent = best(run)
        told = "ran" if outcome == "ran" else "refused: " + outcome.split(": ", 1)[-1][:48]
        how = "with an audit log" if logged else "alone"
        line += f"; call_tool {how} {spent:,.{places}f} ms ({told})"
    print(line, flush=True)


def tagged_union(count: int) -> Any:
    """A list of a discriminated union of ``count`` models, each with a ``Literal`` tag
    and three fields that have defaults."""
    models = [
        create_model(
            f"M{number}",
            kind=(Literal[f"m{number}"], ...),
            a=(int, number),
            b=(str, "x"),
            c=(Optional[float], None),
        )
        for number in range(count)
    ]
    return list[Annotated[Union[tuple(models)], Field(discriminator="kind")]]  # type: ignore[valid-type]


def decoded(arguments: dict[str, Any]) -> dict[str, Any]:
    """The arguments as the SDK hands them to the gate: decoded from JSON text."""
    return dict(json.loads(json.dumps(arguments, ensure_ascii=False)))


#: How many plain frames further down each of the three runs of ``best`` is made: three
#: places in one of the interpreter's 16 KB blocks of frames (the module's own text).
PLACES = (0, 48, 96)


def best(run: Callable[[], Any]) -> float:
    """The shortest of three runs, in milliseconds, each made from another place in the
    stack (``PLACES``): on Python 3.11 and later the place moves the time, and the
    shortest of three runs from one place is the time of that place. The collector is
    run first, so that one measurement does not pay for the garbage of the one before
    it."""

    def timed() -> float:
        start = time.perf_counter()
        run()
        return time.perf_counter() - start

    gc.collect()
    return min(_under(frames, timed) for frames in PLACES) * 1000


def gate(
    schema: dict[str, Any], arguments: dict[str, Any], ceiling: int = STEP_CEILING, runs: int = 3
) -> tuple[float, int, int, str]:
    """``(milliseconds, steps spent, steps allowed, outcome)`` for one call."""
    arguments = decoded(arguments)
    made: list[Any] = []
    original = _schema._Work.__init__

    def remembering(work: Any, allowed: int) -> None:
        original(work, allowed)
        made.append(work)

    outcome = "completed"
    listed = Listed(schema, ceiling)

    def run() -> None:
        nonlocal outcome
        try:
            listed.complete(arguments)
        except Refused as refused:
            outcome = "refused: " + refused.reason

    setattr(_schema._Work, "__init__", remembering)
    try:
        if runs == 1:
            start = time.perf_counter()
            run()
            spent = (time.perf_counter() - start) * 1000
        else:
            spent = best(run)
    finally:
        setattr(_schema._Work, "__init__", original)
    if not made:  # the schema itself was refused, before any step
        return spent, 0, 0, outcome
    return spent, made[-1].steps, made[-1].allowed, outcome


def row(
    label: str,
    schema: dict[str, Any],
    arguments: dict[str, Any],
    kind: Any = None,
    ceiling: int = STEP_CEILING,
) -> None:
    spent, steps, allowed, outcome = gate(schema, arguments, ceiling)
    size = len(json.dumps(arguments, separators=(",", ":"), ensure_ascii=False))
    values = _schema._count(arguments)
    line = (
        f"{label}: {size:,} bytes, {values:,} values: gate {spent:,.0f} ms, {steps:,} of "
        f"{allowed:,} steps ({steps / values:,.1f} for each value, "
        f"{spent * 1000 / max(steps, 1):,.2f} microseconds a step), {outcome[:112]}"
    )
    if kind is not None and outcome == "completed":
        adapter = TypeAdapter(kind)
        value = decoded(arguments)["p"]
        own = best(lambda: adapter.validate_python(value))
        line += f"; Pydantic {own:,.1f} ms; the gate takes {spent / own:,.0f} times as long"
    print(line, flush=True)


def under(kind: Any) -> dict[str, Any]:
    """The schema of a tool with one parameter ``p`` of this type, the definitions the
    type's schema refers to at the root, where a server lists them."""
    listed = TypeAdapter(kind).json_schema()
    definitions = listed.pop("$defs", None)
    schema: dict[str, Any] = {"type": "object", "properties": {"p": listed}}
    if definitions:
        schema["$defs"] = definitions
    return schema


def largest(schema: dict[str, Any], make: Callable[[int], Any]) -> int:
    """The most items of this shape one call can hold under the default ceiling."""

    def fits(count: int) -> bool:
        return gate(schema, {"p": make(count)}, runs=1)[3] == "completed"

    low, high = 0, 64
    while fits(high):
        low, high = high, high * 2
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def main(only: str = "") -> None:
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"the ceiling on the steps of one call: {STEP_CEILING:,}")
    if only == "stack":
        stack()
        return
    if only != "size":
        steps()
    sizes()


def steps() -> None:
    """What the check costs, by its steps: ``Listed.complete`` alone."""

    print("\nuniqueItems (`set[int]`): ordinary integers, and integers that share one hash")
    unique = under(set[int])
    for count, ceiling in ((10_000, STEP_CEILING), (20_000, STEP_CEILING), (100_000, RAISED)):
        note = "" if ceiling == STEP_CEILING else ", the ceiling raised"
        row(f"  {count:,} ordinary{note}", unique, {"p": list(range(1, count + 1))}, None, ceiling)
        colliding = [number * ONE_HASH for number in range(1, count + 1)]
        assert len({hash(number) for number in colliding}) == 1
        row(f"  {count:,} of one hash{note}", unique, {"p": colliding}, None, ceiling)

    print("\nunevaluatedProperties on a hand-written schema: the members cost their square")
    unevaluated = {
        "type": "object",
        "properties": {
            "p": {
                "type": "object",
                "allOf": [{"additionalProperties": {"type": "number"}}],
                "unevaluatedProperties": False,
            }
        },
    }

    def members(count: int) -> dict[str, int]:
        return {f"k{number}": 1 for number in range(count)}

    for count in (100, 250, 400, 500, 30_000):
        row(f"  {count:,} members", unevaluated, {"p": members(count)})
    for count in (1_000, 1_100):
        row(
            f"  {count:,} members, the ceiling raised",
            unevaluated,
            {"p": members(count)},
            None,
            RAISED,
        )
    print(f"  the most members the default ceiling completes: {largest(unevaluated, members):,}")

    print("\na hand-written schema that reaches one subschema 4,096 ways (12 levels of allOf)")
    lattice: dict[str, Any] = {
        "type": "object",
        "properties": {"p": {"type": "array", "items": {"$ref": "#/$defs/a0"}}},
        "$defs": {"a12": {"type": "integer"}},
    }
    for level in range(12):
        onward = {"$ref": f"#/$defs/a{level + 1}"}
        lattice["$defs"][f"a{level}"] = {"allOf": [onward, dict(onward)]}
    for count in (100, 1_000):
        row(
            f"  {count:,} one-item lists, each failing at the bottom",
            lattice,
            {"p": [[n] for n in range(count)]},
        )

    print("\na recursive Node: a chain of `kid` 60 deep whose last node is not valid")
    for count in (5_000, 62_500, 250_000):
        row(f"  the last node holds {count:,} kids", under(Node), {"p": chain(count)})

    print("\none long string: a step that reads text costs its length, whatever the text is")
    ten = {"anyOf": [{"type": "integer", "minimum": number} for number in range(10)]}
    texts: list[tuple[str, dict[str, Any]]] = [
        ("Optional[str]", under(Optional[str])),
        ("Union[int, str]", under(Union[int, str])),
        ("ten alternatives, none a string", {"type": "object", "properties": {"p": ten}}),
    ]
    for length in (100_000, 1_000_000, 10_000_000, 50_000_000):
        for name, character in (("a", "a"), ("NUL", "\x00"), ("a byte order mark", "\ufeff")):
            for kind, schema in texts:
                if length > 1_000_000 and (name == "NUL" or kind == "Union[int, str]"):
                    continue
                if length > 10_000_000 and name != "a":
                    continue
                row(f"  {length:,} x {name}, {kind}", schema, {"p": character * length})

    cases = shapes_of_calls()

    print("\nthe gate beside a server's own validation, 300 items")
    for label, kind, make in cases:
        row(f"  {label}, 300", under(kind), {"p": make(300)}, kind)

    print("\nthe largest call of each shape the default ceiling completes, and the next one")
    for label, kind, make in cases:
        schema = under(kind)
        count = largest(schema, make)
        row(f"  {label}, {count:,}", schema, {"p": make(count)}, kind)
        row(f"  {label}, {count + 1:,}", schema, {"p": make(count + 1)}, kind)

    print("\nwith the ceiling raised (step_ceiling=), 10,000 items, and 50,000 of four shapes")
    for label, kind, make in cases:
        for count in (10_000, 50_000):
            if count == 10_000 or kind in (list[int], list[str], list[Item]) or "float]]" in label:
                row(f"  {label}, {count:,}", under(kind), {"p": make(count)}, kind, RAISED)


def shapes_of_calls() -> list[tuple[str, Any, Callable[[int], Any]]]:
    """The shapes of ordinary arguments measured: a label, the type, and the arguments
    of a given number of items."""
    return [
        ("list[int]", list[int], lambda count: list(range(count))),
        ("list[str]", list[str], lambda count: [f"a{number}" for number in range(count)]),
        (
            "dict[str, Optional[float]]",
            dict[str, Optional[float]],
            lambda count: {f"k{number}": 1.5 for number in range(count)},
        ),
        (
            "dict[str, Optional[str]]",
            dict[str, Optional[str]],
            lambda count: {f"x{number}": f"x{number}" for number in range(count)},
        ),
        (
            "list[Union[int, str]], short strings",
            list[Union[int, str]],
            lambda count: [f"x{number}" for number in range(count)],
        ),
        (
            "list[Union[int, float, bool, str, None]], short strings",
            list[Union[int, float, bool, str, None]],
            lambda count: [f"x{number}" for number in range(count)],
        ),
        (
            "a recursive JSON type, 60 nested lists around numbers",
            Json,
            lambda count: nested(60, [0.1] * count),
        ),
        (
            "list[Item], four fields, two defaults written",
            list[Item],
            lambda count: [{"path": f"p{number}", "content": "c"} for number in range(count)],
        ),
        (
            "a recursive Node, its kids sent as {}",
            Node,
            lambda count: {"kids": [{} for _ in range(count)]},
        ),
        (
            "a list of a discriminated union of 10 models, items {kind}",
            tagged_union(10),
            lambda count: [{"kind": f"m{number % 10}"} for number in range(count)],
        ),
        (
            "a list of a discriminated union of 30 models, items {kind}",
            tagged_union(30),
            lambda count: [{"kind": f"m{number % 30}"} for number in range(count)],
        ),
    ]


def sizes() -> None:
    """What one call costs through ``call_tool``, up to the ceiling on its size."""
    cases = shapes_of_calls()
    ten = {"anyOf": [{"type": "integer", "minimum": number} for number in range(10)]}
    print(f"\nthe ceiling on the size of one call's arguments: {SIZE_CEILING:,} characters")
    print("through call_tool on a gated server object, both ceilings as they are by default")
    anything = {"type": "object", "properties": {"p": {}}}
    text = {"type": "object", "properties": {"p": {"type": "string"}}}
    tens = {"type": "object", "properties": {"p": ten}}

    def most(make: Callable[[int], dict[str, Any]]) -> dict[str, Any]:
        """The largest arguments of this shape that are not over the ceiling on the size."""
        low, high = 0, SIZE_CEILING
        while high - low > 1:
            middle = (low + high) // 2
            if _schema.too_large(make(middle), SIZE_CEILING):
                high = middle
            else:
                low = middle
        return make(low)

    shapes: list[tuple[str, dict[str, Any], Callable[[int], dict[str, Any]]]] = [
        ("one string of `a`", text, lambda count: {"p": "a" * count}),
        ("one string of byte order marks", text, lambda count: {"p": "\ufeff" * count}),
        ("one string of NUL, ten alternatives", tens, lambda count: {"p": "\x00" * count}),
        (
            "`0` in an argument the schema does not name",
            anything,
            lambda count: {"zz": [0] * count},
        ),
        (
            "`{}` in an argument the schema does not name",
            anything,
            lambda count: {"zz": [{} for _ in range(count)]},
        ),
        (
            "`[]` in an argument the schema does not name",
            anything,
            lambda count: {"zz": [[] for _ in range(count)]},
        ),
        (
            "`{}` 60 lists down, in an argument the schema does not name",
            anything,
            lambda count: {"zz": nested(60, [{} for _ in range(count)])},
        ),
        ("`{}` under `Any`", anything, lambda count: {"p": [{} for _ in range(count)]}),
        (
            "`{}` 60 lists down, under `Any`",
            anything,
            lambda count: {"p": nested(60, [{} for _ in range(count)])},
        ),
        (
            "the recursive Node above, 60 deep and its last node not valid, its kids",
            under(Node),
            lambda count: {"p": chain(count)},
        ),
        ("`0` under `list[int]`", under(list[int]), lambda count: {"p": [0] * count}),
        ("empty strings under `list[str]`", under(list[str]), lambda count: {"p": [""] * count}),
    ]
    #: arguments the gate refuses before the check, at the ceiling: not JSON data (a
    #: number too large for a float decodes to one that is not finite), and too deep
    unread: list[tuple[str, Callable[[int], dict[str, Any]]]] = [
        (
            "`{}` 60 lists down, then a number that is not finite (not JSON data)",
            lambda count: {"zz": nested(60, [{} for _ in range(count)]), "zzz": float("inf")},
        ),
        (
            "`[]` 60 lists down, then a number that is not finite (not JSON data)",
            lambda count: {"zz": nested(60, [[] for _ in range(count)]), "zzz": float("inf")},
        ),
        (
            "a string of byte order marks, then a number that is not finite",
            lambda count: {"p": "\ufeff" * count, "zzz": float("inf")},
        ),
        (
            "`{}` 60 lists down, then a value 66 levels down (too deep)",
            lambda count: {"zz": nested(60, [{} for _ in range(count)]), "zzz": nested(66, 0)},
        ),
    ]
    #: a little over the ceiling, where that is known only once most of it is read
    over: list[tuple[str, dict[str, Any]]] = [
        ("a string of 999,000 NUL", {"p": "\x00" * 999_000}),
        ("a string of 999,000 quotes", {"p": '"' * 999_000}),
        ("a string of 500,001 backslashes", {"p": "\\" * 500_001}),
        ("500,001 x `0`", {"zz": [0] * 500_001}),
        ("250,001 x `1.5`", {"zz": [1.5] * 250_001}),
        ("166,700 x `12345`", {"zz": [12345] * 166_700}),
        ("333,400 x `{}`", {"zz": [{} for _ in range(333_400)]}),
        ("333,400 x `{}` 30 lists down", {"zz": nested(30, [{} for _ in range(333_400)])}),
        ("111,200 members `0`", {"zz": {f"k{number:05d}": 0 for number in range(111_200)}}),
        ("233 integers of 4,300 digits", {"zz": [10**4299] * 233}),
    ]
    with tempfile.TemporaryDirectory() as folder:
        for label, kind, make in cases[:6]:
            schema = under(kind)
            count = largest(schema, make)
            seam(f"  {label}, {count:,}", schema, {"p": make(count)}, folder)
        for label, schema, make in shapes:
            seam(f"  at the ceiling: {label}", schema, most(make), folder)
        for label, make in unread:
            seam(f"  at the ceiling: {label}", anything, most(make), folder, alone=False)
        label, schema, both = together()
        seam(f"  the two ceilings together: {label}", schema, both, folder)
        print("over the ceiling on the size (refused; to a tenth of a millisecond)")
        seam("  one character over the ceiling", text, {"p": "a" * SIZE_CEILING}, folder, places=1)
        seam("  a string of 50,000,000 characters", text, {"p": "a" * 50_000_000}, folder, places=1)
        seam(
            "  2,000,000 x `{}`",
            anything,
            {"zz": [{} for _ in range(2_000_000)]},
            folder,
            places=1,
        )
        for label, arguments in over:
            seam(f"  a little over: {label}", anything, arguments, folder, places=1)


def together() -> tuple[str, dict[str, Any], dict[str, Any]]:
    """The call that is at both ceilings: the largest list of short strings the step
    ceiling completes for a union of five types, and, in an argument the schema does
    not name, as many ``{}`` 60 lists down as the ceiling on the size has room left
    for. The steps are spent on the first and the rest of the size on the second."""
    label, kind, make = shapes_of_calls()[5]
    schema = under(kind)
    strings = make(largest(schema, make))
    low, high = 0, SIZE_CEILING
    while high - low > 1:
        middle = (low + high) // 2
        filled = {"p": strings, "zz": nested(60, [{} for _ in range(middle)])}
        if _schema.too_large(filled, SIZE_CEILING):
            high = middle
        else:
            low = middle
    return (
        f"{label}, {len(strings):,}, and {low:,} x `{{}}` 60 lists down in an argument "
        "the schema does not name",
        schema,
        {"p": strings, "zz": nested(60, [{} for _ in range(low)])},
    )


#: How many depths of the stack ``stack`` times a call from: more than the plain frames
#: of ``_under`` that fit one of the interpreter's 16 KB blocks, so every place in a
#: block is met.
DEPTHS = 144


def _under(frames: int, run: Callable[[], float]) -> float:
    """``run()``, from ``frames`` more plain Python frames down the stack."""
    return _under(frames - 1, run) if frames else run()


def stack() -> None:
    """The time of one call by where in the stack it is made (the module's own text says
    why it moves): four calls of ``sizes``, each from ``DEPTHS`` depths, once each."""
    print(f"\nby where the call is made: from each of {DEPTHS} depths of plain Python frames,")
    print("once each (not a best of three); the least, the median and the most")
    cases = shapes_of_calls()

    def most(make: Callable[[int], dict[str, Any]]) -> dict[str, Any]:
        low, high = 0, SIZE_CEILING
        while high - low > 1:
            middle = (low + high) // 2
            if _schema.too_large(make(middle), SIZE_CEILING):
                high = middle
            else:
                low = middle
        return make(low)

    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for label, kind, make in (cases[5], cases[6]):
        schema = under(kind)
        count = largest(schema, make)
        rows.append((f"{label}, {count:,}", schema, {"p": make(count)}))
    label, schema, both = together()
    rows.append((f"the two ceilings together: {label}", schema, both))
    rows.append(
        (
            "at the ceiling: the recursive Node, 60 deep and its last node not valid",
            under(Node),
            most(lambda count: {"p": chain(count)}),
        )
    )
    with tempfile.TemporaryDirectory() as folder:
        for label, schema, arguments in rows:
            arguments = decoded(arguments)
            line = f"  {label}:"
            for logged in (False, True):
                server = Server(schema)
                if logged:
                    log = AuditLog(path=str(Path(tempfile.mkdtemp(dir=folder)) / "log.jsonl"))
                    gate_mcp_server(server, lambda *_: [], recording(log))
                else:
                    gate_mcp_server(server, lambda *_: [])
                asyncio.run(server.list_tools())

                def run() -> float:
                    start = time.perf_counter()
                    asyncio.run(server.call_tool("t", arguments))
                    return (time.perf_counter() - start) * 1000

                times = []
                for frames in range(DEPTHS):
                    gc.collect()
                    times.append((_under(frames, run), frames))
                ordered = sorted(times)
                slowest = ", ".join(f"{ms:,.0f} ms from {at}" for ms, at in ordered[-3:][::-1])
                line += (
                    f" call_tool {'with an audit log' if logged else 'alone'}: least "
                    f"{ordered[0][0]:,.0f}, median {ordered[len(ordered) // 2][0]:,.0f}, most "
                    f"{ordered[-1][0]:,.0f} ms (the three longest: {slowest});"
                )
            print(line, flush=True)
    room()


def room() -> None:
    """From how many plain frames down the stack the deepest call a recursive model
    admits stops completing: a ``Node`` whose ``kid`` leads 63 nodes down, which puts
    the last one 64 levels below the arguments, the deepest the gate reads."""
    deepest: dict[str, Any] = {}
    for _ in range(63):
        deepest = {"kid": deepest}
    arguments = {"p": deepest}
    assert _schema.measure(arguments, SIZE_CEILING) == _schema.FITS
    listed = Listed(under(Node))

    def judge() -> float:
        sent, _ = _gate._judged_call("t", arguments, listed, lambda *_: [])
        if sent is not None:
            return 1.0
        return 0.0

    assert _under(0, judge) == 1.0
    low, high = 0, sys.getrecursionlimit()
    while high - low > 1:
        middle = (low + high) // 2
        try:
            completed = _under(middle, judge) == 1.0
        except RecursionError:  # this script's own frames reached the limit
            completed = False
        low, high = (middle, high) if completed else (low, middle)
    print(
        f"\na recursive model nested to the 64 levels the gate reads (`kid` 63 deep), under "
        f"the interpreter's recursion limit of {sys.getrecursionlimit():,}: completed when "
        f"the gate is called from up to {low:,} plain frames below this script's own, and "
        f"refused from {high:,}"
    )


if __name__ == "__main__":
    main(only=sys.argv[1] if sys.argv[1:] in (["size"], ["stack"]) else "")
