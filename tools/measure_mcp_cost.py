"""Measure what the gate's check of a call's arguments costs, on this machine.

The README of this package quotes the output of this script. It times
``Listed.complete`` (the validation, the two refusals and the defaults, everything the
gate does with a listed schema before the policy runs) on arguments of several sizes
and shapes, counts the steps it spent of the steps it was allowed, and, where a
Pydantic type publishes the schema, times Pydantic's own validation of the same
arguments, which is what a server built with the MCP Python SDK runs.

The check of one call stops at a ceiling of steps (``STEP_CEILING``), which an
application can raise for a connection (``step_ceiling=``). So each shape is measured
three ways: at a size every shape completes, at the largest size the default ceiling
completes (found by trying sizes), and at larger sizes with the ceiling raised, which is
what raising it costs.

    python tools/measure_mcp_cost.py

It needs the package and its dependencies installed. Times differ from machine to
machine and run to run; each is the best of three.
"""

from __future__ import annotations

import json
import platform
import time
from typing import Annotated, Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, create_model
from recusal_openai_agents_mcp import STEP_CEILING, _schema
from recusal_openai_agents_mcp._schema import Listed, Refused

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


def best(run: Callable[[], Any]) -> float:
    """The shortest of three runs, in milliseconds."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return min(times) * 1000


def gate(
    schema: dict[str, Any], arguments: dict[str, Any], ceiling: int = STEP_CEILING, runs: int = 3
) -> tuple[float, int, int, str]:
    """``(milliseconds, steps spent, steps allowed, outcome)`` for one call."""
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
    size = len(json.dumps(arguments, separators=(",", ":")))
    values = _schema._count(arguments)
    line = (
        f"{label}: {size:,} bytes, {values:,} values: gate {spent:,.0f} ms, {steps:,} of "
        f"{allowed:,} steps ({steps / values:,.1f} for each value), {outcome[:112]}"
    )
    if kind is not None and outcome == "completed":
        adapter = TypeAdapter(kind)
        own = best(lambda: adapter.validate_python(arguments["p"]))
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


def main() -> None:
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"the ceiling on the steps of one call: {STEP_CEILING:,}")

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

    print("\none long string, for `Optional[str]`: a step that reads text costs its length")
    for length in (100_000, 1_000_000):
        row(f"  {length:,} characters", under(Optional[str]), {"p": "a" * length})

    cases: list[tuple[str, Any, Callable[[int], Any]]] = [
        ("list[int]", list[int], lambda count: list(range(count))),
        ("list[str]", list[str], lambda count: [f"a{number}" for number in range(count)]),
        (
            "dict[str, Optional[float]]",
            dict[str, Optional[float]],
            lambda count: {f"k{number}": 1.5 for number in range(count)},
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
            if count == 10_000 or kind in (list[int], list[str], list[Item]) or "dict" in label:
                row(f"  {label}, {count:,}", under(kind), {"p": make(count)}, kind, RAISED)


if __name__ == "__main__":
    main()
