"""
Offline CrewAI example: gate every tool call with CrewAI's own tool-call hooks.

Requires Python 3.10+ (CrewAI's minimum) and is tested with exactly crewai==1.15.34.
Recusal itself remains Python 3.9+ and has no CrewAI dependency.

The hook is CrewAI's own. CrewAI runs its ``before_tool_call`` hooks immediately before a
tool executes, with the tool name and the argument dict the tool will receive; a hook that
returns ``False`` blocks the call. CrewAI then answers the model with a fixed line, "Tool
execution blocked by hook. Tool: <name>", and runs the ``after_tool_call`` hooks even on a
blocked call; an after hook that returns a string replaces that answer. So the gate is two
hooks that work as one: ``recusal_before_tool_call(policy)`` judges the call with
``recusal.claude_code.decide`` and blocks a refusal, and ``recusal_after_tool_call(policy)``
replaces the fixed line with the reason the before hook blocked on, so the model reads why
on its next turn. ``install_recusal_hooks(policy)`` registers both and returns the function
that removes them.

The reason is the one the before hook decided, never a second judgment: a policy may be
stateful, and the blocked call must be explained by the decision that blocked it. The two
hooks correlate by the argument dict object itself: CrewAI hands the same ``tool_input``
dict to the before and after hooks of one invocation on one thread (verified for the sync
Crew runner, where every invocation also gets a fresh dict), so the before hook keeps the
reason beside a strong reference to that object in a small per-thread list, and the after
hook returns it only for that very object. Any other case, a different object, a different
thread, an entry already used or dropped, keeps CrewAI's generic line; a reason is never
borrowed from another call, and blocking never depends on the explanation being kept.

CrewAI's hook registry is process-global: one install covers every agent, task and crew in
the process, including the coworker an agent delegates to, and every tool kind CrewAI runs
through its hooks (its own tools, ``@tool`` functions, MCP tools). Two consequences the tests
pin:

- hooks run in registration order and a later hook sees, and can change, the argument dict
  the gate judged. Install the gate AFTER your other hooks, so it judges the call exactly as
  it will run; a hook registered after it that rewrites the arguments bypasses the gate;
- the gate judges the tool name and arguments, not what a tool does once it runs. A tool
  that executes code or shell it is handed (CrewAI's code interpreter, for instance) needs a
  policy that names it; the gate cannot see inside.

No model, API key or network is used: ``ScriptedLLM`` is a ``BaseLLM`` that proposes the
calls from a script, and run_sql only records SQL text.

Run after installing:
    pip install crewai==1.15.34
    python examples/crewai_gate.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from types import SimpleNamespace
from typing import Any, Callable, List, Optional

CREWAI_VERSION = "1.15.34"

if sys.version_info < (3, 10):
    print(
        "crewai_gate.py requires Python 3.10+ "
        f"(crewai {CREWAI_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

# offline, and quiet: no telemetry, no tracing banner. Set before CrewAI is imported.
for _var, _value in (
    ("CREWAI_DISABLE_TELEMETRY", "true"),
    ("CREWAI_TRACING_ENABLED", "false"),
    ("OTEL_SDK_DISABLED", "true"),
):
    os.environ.setdefault(_var, _value)

try:
    from crewai import Agent, Crew, Task
    from crewai.hooks import (
        register_after_tool_call_hook,
        register_before_tool_call_hook,
        unregister_after_tool_call_hook,
        unregister_before_tool_call_hook,
    )
    from crewai.hooks.tool_hooks import ToolCallHookContext
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import tool
except ImportError as exc:
    raise SystemExit(f"Install the example dependency first: crewai=={CREWAI_VERSION}") from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal.claude_code import decide  # noqa: E402

Policy = Callable[[str, dict], List[Any]]
EXECUTED: list[tuple[str, dict[str, Any]]] = []

#: The answer CrewAI gives the model for a call a hook blocked; the after hook replaces it.
BLOCKED_PREFIX = "Tool execution blocked by hook."


@tool("echo")
def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


@tool("run_sql")
def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


def _refusal(tool_name: str, tool_input: Any, policy: Policy) -> Optional[str]:
    """Return the reason a call is refused, or None to run it.

    CrewAI hands the hook the same dict the tool will be called with, so the policy judges
    exactly what would run. A non-dict is refused: the policy cannot read it.
    """
    if not isinstance(tool_input, dict):
        return f"Recusal refused `{tool_name}`: arguments are not a mapping"
    try:
        decision, reason = decide(tool_name, tool_input, policy)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): {exc}"
    return reason if decision == "deny" else None


#: How many blocked invocations one thread may hold reasons for before the oldest is
#: dropped (a dropped entry degrades to CrewAI's generic line, never to a wrong reason).
PENDING_LIMIT = 16


class _Denials:
    """The reasons the before hook blocked on, per thread, keyed by the argument object.

    One entry per blocked invocation: ``(the tool_input object, reason)``. The object is
    held strongly so its id cannot be reused while the entry lives. The before hook first
    drops any entry for the same object, whatever the decision: one object cannot carry two
    live invocations on one thread, so an entry still present is from an invocation whose
    after hook never ran, and must not be attached to the next call that reuses the dict.
    """

    def __init__(self):
        self._local = threading.local()

    def _entries(self) -> list:
        entries = getattr(self._local, "entries", None)
        if entries is None:
            entries = self._local.entries = []
        return entries

    def forget(self, tool_input: Any) -> None:
        entries = self._entries()
        entries[:] = [e for e in entries if e[0] is not tool_input]

    def keep(self, tool_input: Any, reason: str) -> None:
        entries = self._entries()
        entries.append((tool_input, reason))
        del entries[:-PENDING_LIMIT]

    def take(self, tool_input: Any) -> Optional[str]:
        entries = self._entries()
        for index, (held, reason) in enumerate(entries):
            if held is tool_input:
                del entries[index]
                return reason
        return None


def recusal_before_tool_call(policy: Policy, denials: Optional[_Denials] = None):
    """Return a ``before_tool_call`` hook: ``False`` (block) on a refusal, ``None`` to run.

    ``denials`` is shared with the matching after hook (``install_recusal_hooks`` builds the
    pair); without it the reason is not kept and the after hook keeps CrewAI's line.
    """
    store = denials

    def gate(context: ToolCallHookContext) -> Optional[bool]:
        tool_input = context.tool_input
        if store is not None:
            try:
                store.forget(tool_input)
            except Exception:  # noqa: BLE001, bookkeeping never changes the decision
                pass
        reason = _refusal(context.tool_name, tool_input, policy)
        if reason is None:
            return None
        if store is not None:
            try:
                store.keep(tool_input, reason)
            except Exception:  # noqa: BLE001, a failure to keep the reason cannot unblock
                pass
        return False

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


def recusal_after_tool_call(policy: Policy, denials: Optional[_Denials] = None):
    """Return an ``after_tool_call`` hook that puts the before hook's reason in a blocked
    call's answer.

    The reason is the one kept for this invocation's argument object; the call is never
    judged again. A call another hook blocked, or one whose reason was not kept (another
    thread, a dropped entry, no shared ``denials``), keeps CrewAI's own line.
    """
    store = denials

    def explain(context: ToolCallHookContext) -> Optional[str]:
        if not str(context.tool_result).startswith(BLOCKED_PREFIX):
            return None
        if store is None:
            return None
        try:
            return store.take(context.tool_input)
        except Exception:  # noqa: BLE001, the generic line is the safe fallback
            return None

    explain.recusal_gate = True  # type: ignore[attr-defined]
    return explain


def install_recusal_hooks(policy: Policy) -> Callable[[], None]:
    """Register the gate's two hooks for every tool call in this process; return ``uninstall``.

    Register it after your other hooks: hooks run in registration order, and a hook that
    runs after the gate can change the arguments the gate judged.
    """
    denials = _Denials()
    before = recusal_before_tool_call(policy, denials)
    after = recusal_after_tool_call(policy, denials)
    register_before_tool_call_hook(before)
    register_after_tool_call_hook(after)

    def uninstall() -> None:
        unregister_before_tool_call_hook(before)
        unregister_after_tool_call_hook(after)

    return uninstall


# --- offline fixtures: a scripted model, no network --------------------------------------


def _tool_call(call_id: str, name: str, arguments: Any) -> SimpleNamespace:
    """One native tool call in the OpenAI shape CrewAI's executor reads."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
        ),
    )


class ScriptedLLM(BaseLLM):
    """A model that proposes the scripted calls on its first turn, then answers "done"."""

    def __init__(self, calls: list[tuple[str, Any, str]]):
        super().__init__(model="scripted")
        self.calls = list(calls)
        self.turns = 0

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **_: Any):
        self.turns += 1
        if self.turns == 1 and self.calls:
            return [_tool_call(call_id, name, arguments) for name, arguments, call_id in self.calls]
        return "Final Answer: done"

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 100_000


def scripted_agent(calls: list[tuple[str, Any, str]], tools: Optional[list] = None) -> Agent:
    return Agent(
        role="assistant",
        goal="do the task",
        backstory="an offline agent",
        llm=ScriptedLLM(calls),
        tools=[echo, run_sql] if tools is None else tools,
        verbose=False,
    )


def run_crew(agent: Agent) -> dict[str, str]:
    """Run one task through a Crew; return what the model received for each call, by id."""
    task = Task(description="go", expected_output="done", agent=agent)
    Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    executor = agent.agent_executor
    responses: dict[str, str] = {}
    for message in getattr(executor.state, "messages", []) if executor is not None else []:
        if isinstance(message, dict) and message.get("role") == "tool":
            responses[str(message.get("tool_call_id"))] = str(message.get("content"))
    return responses


def run_calls(
    calls: list[tuple[str, Any, str]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
) -> tuple[dict[str, str], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent with echo and run_sql; return answers and what ran."""
    EXECUTED.clear()
    uninstall = install_recusal_hooks(policy) if gated else (lambda: None)
    try:
        responses = run_crew(scripted_agent(calls))
    finally:
        uninstall()
    return responses, list(EXECUTED)


def main() -> None:
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    responses, executed = run_calls(calls)
    for name, _, call_id in calls:
        print(f"{name} ({call_id}): {responses[call_id]}")
    print(f"tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
