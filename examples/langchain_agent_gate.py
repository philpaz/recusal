"""Offline LangChain create_agent example sharing the existing Recusal ToolNode gate.

Requires Python 3.10+, langgraph==1.2.12, langchain-core==1.6.5,
and langchain==1.4.3. No model, credential, or database is contacted.
Install pins as in README and run: python examples/langchain_agent_gate.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

LANGGRAPH_VERSION = "1.2.12"
LANGCHAIN_CORE_VERSION = "1.6.5"
LANGCHAIN_VERSION = "1.4.3"

if sys.version_info < (3, 10):
    print(
        "langchain_agent_gate.py requires Python 3.10+ "
        f"(LangChain {LANGCHAIN_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_tool_call
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"langgraph=={LANGGRAPH_VERSION} langchain-core=={LANGCHAIN_CORE_VERSION} "
        f"langchain=={LANGCHAIN_VERSION}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.langgraph_gate import EXECUTED, echo, recusal_gate, run_sql, sql_scope_policy  # noqa: E402


class ScriptedChatModel(GenericFakeChatModel):
    """Offline response stream with a no-op tool binder; no provider is called."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


def run_calls(
    calls: list[dict[str, Any]], *, policy=sql_scope_policy, gated: bool = True
) -> tuple[list[ToolMessage], list[tuple[str, dict[str, Any]]]]:
    """Run fake model proposals through create_agent, one policy gate per tool call."""
    EXECUTED.clear()
    model = ScriptedChatModel(
        messages=iter([AIMessage(content="", tool_calls=calls), AIMessage(content="done")])
    )
    hooks = [wrap_tool_call(recusal_gate(policy))] if gated else []
    agent = create_agent(model=model, tools=[echo, run_sql], middleware=hooks)
    result = agent.invoke({"messages": [HumanMessage(content="run the scripted tools")]})
    messages = [item for item in result["messages"] if isinstance(item, ToolMessage)]
    return messages, list(EXECUTED)


def main() -> None:
    calls = [
        {"name": "echo", "args": {"text": "hi"}, "id": "e", "type": "tool_call"},
        {"name": "run_sql", "args": {"sql": "DELETE FROM audit_log"}, "id": "d", "type": "tool_call"},
    ]
    messages, executed = run_calls(calls)
    for message in messages:
        print(f"{message.name} [{message.status}]: {message.content}")
    print(f"tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
