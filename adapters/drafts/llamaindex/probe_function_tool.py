"""Compatibility probe for the LlamaIndex FunctionTool seam.

Run with the pinned dependency from the repository root:

    .venv/bin/python adapters/drafts/llamaindex/probe_function_tool.py

The probe is deliberately offline.  It records the arguments that a local function
actually receives, including Python defaults and FunctionTool partial parameters, and
proves that LlamaIndex's callback observes a result only after that function has run.
"""

import asyncio
from typing import Any

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow.context import Context

SYNC_EXECUTIONS: list[dict[str, Any]] = []
ASYNC_EXECUTIONS: list[dict[str, Any]] = []
CALLBACK_OBSERVATIONS: list[str] = []


def write_record(
    table: str,
    mode: str = "hard",
    tenant: str = "default",
    ctx: Context = None,  # type: ignore[assignment]
) -> str:
    """Record the exact arguments that reached the local function."""
    SYNC_EXECUTIONS.append({"table": table, "mode": mode, "tenant": tenant, "ctx": ctx})
    return "sync complete"


async def write_record_async(
    table: str,
    mode: str = "hard",
    tenant: str = "default",
    ctx: Context = None,  # type: ignore[assignment]
) -> str:
    """Async form of the same local function."""
    ASYNC_EXECUTIONS.append({"table": table, "mode": mode, "tenant": tenant, "ctx": ctx})
    return "async complete"


def post_execution_callback(result: Any) -> None:
    assert SYNC_EXECUTIONS, "callback ran before the function"
    CALLBACK_OBSERVATIONS.append(str(result))


def sync_probe() -> None:
    marker = object()
    tool = FunctionTool.from_defaults(
        fn=write_record,
        partial_params={"tenant": "partial-tenant"},
        callback=post_execution_callback,
    )

    output = tool.call(table="orders", ctx=marker)

    assert SYNC_EXECUTIONS == [
        {"table": "orders", "mode": "hard", "tenant": "partial-tenant", "ctx": marker}
    ]
    assert CALLBACK_OBSERVATIONS == ["sync complete"]
    # Context is passed to the function but omitted from the model-facing ToolOutput.
    # `FunctionTool`'s raw input is *not* yet the full invocation: ordinary Python
    # defaults are still absent here and Python applies them only when `_fn` runs.
    # An enforcement wrapper must complete those defaults before judgment.
    assert output.raw_input == {
        "args": (),
        "kwargs": {"table": "orders", "tenant": "partial-tenant"},
    }

    # Call kwargs override partial_params in FunctionTool.call().  A gate must judge this
    # final merged value rather than assuming partial_params are immutable.
    tool.call(table="orders", tenant="per-call", ctx=marker)
    assert SYNC_EXECUTIONS[-1]["tenant"] == "per-call"


async def async_probe() -> None:
    marker = object()
    tool = FunctionTool.from_defaults(
        async_fn=write_record_async,
        partial_params={"tenant": "partial-tenant"},
    )

    output = await tool.acall(table="orders", ctx=marker)
    assert ASYNC_EXECUTIONS == [
        {"table": "orders", "mode": "hard", "tenant": "partial-tenant", "ctx": marker}
    ]
    assert output.raw_input == {
        "args": (),
        "kwargs": {"table": "orders", "tenant": "partial-tenant"},
    }


def main() -> None:
    sync_probe()
    asyncio.run(async_probe())
    print(
        "llama-index-core FunctionTool probe passed: Python defaults are applied after "
        "the tool's raw input; partial params, context, sync, and async were also observed"
    )


if __name__ == "__main__":
    main()
