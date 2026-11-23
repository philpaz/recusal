"""Measure what the gate's check of a call's arguments costs, on this machine.

The README of this package quotes the output of this script. It times
``Listed.complete`` (the validation, the two refusals and the defaults, everything the
gate does with a listed schema before the policy runs) on arguments of several sizes
and shapes, counts the steps it spent of the steps it was allowed, and, where a
Pydantic type publishes the schema, times Pydantic's own validation of the same
arguments, which is what a server built with the MCP Python SDK runs.

    python tools/measure_mcp_cost.py

It needs the package and its dependencies installed. Times differ from machine to
machine and run to run; each is the best of three.
"""

from __future__ import annotations

import platform
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel, TypeAdapter
from recusal_openai_agents_mcp import _schema
from recusal_openai_agents_mcp._schema import Listed, Refused

#: Every multiple of this has the hash 0: Python reduces an integer by it.
ONE_HASH = 2**61 - 1


class Item(BaseModel):
    path: str
    content: str
    mode: str = "x"
    flag: bool = False


def best(run: Callable[[], Any]) -> float:
    """The shortest of three runs, in milliseconds."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return min(times) * 1000


def gate(schema: dict[str, Any], arguments: dict[str, Any]) -> tuple[float, int, int, str]:
    """``(milliseconds, steps spent, steps allowed, outcome)`` for one call."""
    made: list[Any] = []
    original = _schema._Work.__init__

    def remembering(work: Any, allowed: int) -> None:
        original(work, allowed)
        made.append(work)

    outcome = "completed"

    def run() -> None:
        nonlocal outcome
        try:
            Listed(schema).complete(arguments)
        except Refused as refused:
            outcome = "refused: " + refused.reason

    setattr(_schema._Work, "__init__", remembering)
    try:
        spent = best(run)
    finally:
        setattr(_schema._Work, "__init__", original)
    if not made:  # the schema itself was refused, before any step
        return spent, 0, 0, outcome
    return spent, made[-1].steps, made[-1].allowed, outcome


def row(label: str, schema: dict[str, Any], arguments: dict[str, Any], kind: Any = None) -> None:
    spent, steps, allowed, outcome = gate(schema, arguments)
    line = f"{label}: gate {spent:,.0f} ms, {steps:,} of {allowed:,} steps, {outcome[:60]}"
    if kind is not None:
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


def main() -> None:
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")

    print("\nuniqueItems (`set[int]`): ordinary integers, and integers that share one hash")
    unique = under(set[int])
    for count in (10_000, 20_000, 100_000):
        row(f"  {count:,} ordinary", unique, {"p": list(range(1, count + 1))})
        colliding = [number * ONE_HASH for number in range(1, count + 1)]
        assert len({hash(number) for number in colliding}) == 1
        row(f"  {count:,} of one hash", unique, {"p": colliding})

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
    for count in (250, 500, 1_000, 1_100, 30_000):
        members = {f"k{number}": 1 for number in range(count)}
        row(f"  {count:,} members", unevaluated, {"p": members})

    print("\nthe gate beside a server's own validation, 10,000 and 50,000 items")
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
    ]
    for label, kind, make in cases:
        for count in (10_000, 50_000):
            row(f"  {label}, {count:,}", under(kind), {"p": make(count)}, kind)


if __name__ == "__main__":
    main()
