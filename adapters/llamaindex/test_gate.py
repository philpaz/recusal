"""Offline behavior tests for the explicit LlamaIndex FunctionTool gate."""

import asyncio
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow.context import Context
from recusal import Finding, Severity
from recusal_llamaindex import gated_tool, is_gated


def test_defaults_are_judged_before_the_function_runs():
    ran, seen = [], []

    def write(table: str, mode: str = "hard") -> str:
        ran.append((table, mode))
        return "ok"

    def policy(name, invocation):
        seen.append((name, dict(invocation)))
        return [Finding.fail("no_hard", severity=Severity.ERROR, message="hard mode needs approval")]

    tool = gated_tool(FunctionTool.from_defaults(fn=write), policy)
    output = tool.call(table="orders")
    assert "hard mode needs approval" in output.content
    assert ran == []
    assert seen == [("write", {"table": "orders", "mode": "hard"})]


def test_context_is_not_policy_input_and_partial_can_be_overridden():
    ran, seen = [], []

    def write(table: str, tenant: str = "default", ctx: Context = None) -> str:  # type: ignore[assignment]
        ran.append((table, tenant, ctx))
        return "ok"

    def policy(name, invocation):
        seen.append(dict(invocation))
        return []

    marker = object()
    tool = gated_tool(FunctionTool.from_defaults(fn=write, partial_params={"tenant": "partial"}), policy)
    assert is_gated([tool]) and not is_gated([tool, FunctionTool.from_defaults(fn=write)])
    tool.call(table="orders", tenant="per-call", ctx=marker)
    assert seen == [{"table": "orders", "tenant": "per-call"}]
    assert ran == [("orders", "per-call", marker)]


def test_decorated_shape_that_hides_a_second_default_is_refused():
    ran, seen = [], []

    def with_timeout(fn):
        @functools.wraps(fn)
        def wrapper(*args, timeout: int = 30, **kwargs):
            return fn(*args, **kwargs)
        return wrapper

    @with_timeout
    def fetch(url: str, timeout: int = 5) -> str:
        ran.append((url, timeout))
        return "ok"

    def policy(name, invocation):
        seen.append(invocation)
        return []

    output = gated_tool(FunctionTool.from_defaults(fn=fetch), policy).call(url="https://example.test")
    assert "signature cannot be judged" in output.content
    assert ran == [] and seen == []


def test_async_call_is_gated_before_execution():
    ran = []

    async def delete(path: str, mode: str = "hard") -> str:
        ran.append((path, mode))
        return "deleted"

    def policy(name, invocation):
        return [Finding.fail("no_delete", severity=Severity.CRITICAL, message="blocked")]

    output = asyncio.run(gated_tool(FunctionTool.from_defaults(async_fn=delete), policy).acall(path="/tmp/a"))
    assert "blocked" in output.content
    assert ran == []
