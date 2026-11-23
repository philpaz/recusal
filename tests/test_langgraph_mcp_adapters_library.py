"""The names the LangGraph gate reads from ``langchain-mcp-adapters``, against the library.

The gate recognises the function ``langchain-mcp-adapters`` builds for an MCP tool by the
library's own code object, and reads one variable of it. The package's own tests pin that
with a stand-in of the same structure, since the library is not part of the package's
framework and is not installed in its lanes. This file is the check against the library
itself: it runs wherever ``langchain-mcp-adapters`` is installed and is skipped elsewhere,
which is why it is not one of the adapter's own test files (a release refuses a skip
there).
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "langgraph_gate.py"
MCP_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "depth": {"type": "integer", "default": 2}},
    "required": ["path"],
}


@pytest.fixture
def example():
    pytest.importorskip("langgraph")
    spec = importlib.util.spec_from_file_location("recusal_langgraph_gate_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _adapter():
    """The in-tree package: the example, loaded first, puts it on the path."""
    module = importlib.import_module("recusal_langgraph._gate")
    assert "adapters" in Path(module.__file__).parts, module.__file__
    return module


def _call(name, args, ident="c1"):
    return {"name": name, "args": args, "id": ident, "type": "tool_call"}


def _no_delete(seen):
    from recusal import Finding

    def policy(name, args):
        seen.append((name, args))
        if "DELETE" in repr(args):
            return [Finding.fail("no_delete", severity="CRITICAL", message="no delete")]
        return []

    return policy


def _run(example, tools, calls, policy):
    return asyncio.run(example.run_calls_async(calls, policy=policy, tools=tools, gated=True))[0]


def test_the_names_read_from_langchain_mcp_adapters_are_the_librarys_own(example):
    """The builder, the function it defines, the variable that holds its interceptors, and a
    tool it builds running with exactly the judged arguments."""
    pytest.importorskip("mcp")
    tools = pytest.importorskip("langchain_mcp_adapters.tools")
    from mcp.types import CallToolResult, TextContent
    from mcp.types import Tool as MCPTool

    adapter = _adapter()
    assert tools.__name__ == adapter._MCP_MODULE
    builder = getattr(tools, adapter._MCP_BUILDER)
    received, seen = [], []

    class Session:
        async def call_tool(self, name, arguments, **options):
            received.append((name, arguments))
            return CallToolResult(content=[TextContent(type="text", text="served")])

    described = MCPTool(name="read_file", description="d", inputSchema=MCP_SCHEMA)
    tool = builder(Session(), described)
    assert adapter._MCP_INTERCEPTORS in tool.coroutine.__code__.co_freevars
    assert adapter._calls_an_mcp_server(tool.coroutine) is True
    calls = [_call("read_file", {"path": "/x"})]
    messages = _run(example, [tool], calls, _no_delete(seen))
    assert messages[0].status == "success", messages[0].content
    assert seen == [("read_file", {"path": "/x"})]
    assert received == [("read_file", {"path": "/x"})]

    async def rewrite(request, handler):
        return await handler(request.override(args={**request.args, "depth": 99}))

    intercepted = builder(Session(), described, tool_interceptors=[rewrite])
    messages = _run(example, [intercepted], calls, _no_delete(seen))
    assert "with tool interceptors" in messages[0].content
    assert received == [("read_file", {"path": "/x"})], "nothing more was sent"
