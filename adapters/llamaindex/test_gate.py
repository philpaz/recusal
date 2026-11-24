"""Offline behavior tests for the explicit LlamaIndex FunctionTool gate."""

import asyncio
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow.context import Context
from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
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
    assert output.is_error is True
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


def test_workflow_agent_tool_route_uses_the_gated_acall_surface():
    """LlamaIndex's BaseWorkflowAgent invokes FunctionTool.acall, not fn/real_fn."""
    ran, seen = [], []

    def write(table: str, mode: str = "hard") -> str:
        ran.append((table, mode))
        return "ok"

    def policy(name, invocation):
        seen.append(dict(invocation))
        return [Finding.fail("no_hard", severity=Severity.ERROR, message="blocked by policy")]

    tool = gated_tool(FunctionTool.from_defaults(fn=write), policy)
    output = asyncio.run(BaseWorkflowAgent._call_tool(object(), object(), tool, {"table": "orders"}))
    assert output.is_error is True
    assert "blocked by policy" in output.content
    assert seen == [{"table": "orders", "mode": "hard"}]
    assert ran == []


def test_policy_evidence_equals_what_the_function_receives():
    """The central adapter contract: policy evidence and invocation match exactly."""
    ran, seen = [], []

    def scale(value: int, mode: str = "hard") -> str:
        ran.append({"value": value, "mode": mode})
        return "ok"

    def policy(name, invocation):
        seen.append(dict(invocation))
        return []

    tool = gated_tool(FunctionTool.from_defaults(fn=scale), policy)
    tool.call(value="5")
    # FunctionTool itself does not coerce this direct call: both policy and function see
    # the same string. If a future LlamaIndex route validates/coerces first, this test must
    # be extended at that route rather than assuming the direct surface did so.
    assert seen == [{"value": "5", "mode": "hard"}]
    assert ran == [{"value": "5", "mode": "hard"}]
    assert type(seen[0]["value"]) is type(ran[0]["value"]) is str


def test_workflow_route_overwrites_model_supplied_context_before_policy():
    """The workflow supplies host Context and never exposes it to the policy."""
    ran, seen = [], []

    def write(table: str, ctx: Context = None) -> str:  # type: ignore[assignment]
        ran.append({"table": table, "ctx": ctx})
        return "ok"

    def policy(name, invocation):
        seen.append(dict(invocation))
        return []

    host_context = object()
    tool = gated_tool(FunctionTool.from_defaults(fn=write), policy)
    output = asyncio.run(
        BaseWorkflowAgent._call_tool(
            object(), host_context, tool, {"table": "orders", "ctx": "model-value"}
        )
    )
    assert output.is_error is False
    assert seen == [{"table": "orders"}]
    assert ran == [{"table": "orders", "ctx": host_context}]
