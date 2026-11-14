"""
Offline OpenAI Agents SDK example: every gated MCP call on a tamper-evident audit log.

examples/openai_agents_mcp_gate.py refuses unsafe MCP calls in an OpenAI Agents SDK agent,
on local servers and on hosted MCP. This example adds the record: each call a gate sees,
allowed or refused, is one hash-chained entry on a Recusal AuditLog, so a quiet edit, a
reordering, or a dropped entry is detectable with ``recusal audit verify``. The log holds a
SHA-256 fingerprint of each call's arguments, never the arguments themselves.

The audited gate is the ``recusal-openai-agents-mcp`` package
(``adapters/openai-agents-mcp`` in this repository), imported here.
``audited_agent(agent, policy, log)`` is its ``gated_agent`` with a
``recording`` that passes the recording arguments through to ``recusal.decide``, which
writes the entry before returning. If the log cannot be written the decision becomes a
deny, so nothing runs unrecorded. One log covers the agent's three seams, each under its
own surface label:

- ``openai_agents.tool_input_guardrail``: function tools, under the SDK's tool call id;
- ``openai_agents.mcp.call_tool``: a local MCP server's ``call_tool``. The SDK hands that
  method the server's own tool name and the arguments, and no call id, so these entries
  carry none. The server's name is recorded as ``control.mcp_server``, and the entry is
  bound to the arguments the server receives;
- ``openai_agents.hosted_mcp.approval``: a hosted MCP call OpenAI asked approval for, under
  the approval request's id, with the server label as ``control.mcp_server``, bound to the
  arguments in the request.

What is not on the log: a hosted call OpenAI does not ask approval for (the gate never
sees it, which is why the gate sets ``require_approval`` to ``"always"``), and what the
tool's own approval callback answered after Recusal allowed a call: the entry records
Recusal's decision, not that callback's.

A local server is a shared connection and carries one policy and one recording: gating it
again with a different log is refused, as a different policy is.

Requires Python 3.10+ and is tested with exactly openai-agents==0.24.0 and openai==3.34.0.
Recusal itself remains Python 3.9+ and has no OpenAI dependency. No model, API key, process
or network is used.

Run after installing:
    pip install openai-agents==0.24.0 openai==3.34.0
    python examples/openai_agents_mcp_audit.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

if sys.version_info < (3, 10):
    print(
        "openai_agents_mcp_audit.py requires Python 3.10+ (openai-agents 0.24.0); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapters are read from the tree; installed, `pip install
# recusal-openai-agents-mcp` provides the same packages and these paths do not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "openai-agents", "src"))
sys.path.insert(0, os.path.join(_ROOT, "adapters", "openai-agents-mcp", "src"))

from recusal_openai_agents_mcp import SURFACES, audited_agent, recording  # noqa: E402

from examples.openai_agents_mcp_gate import (  # noqa: E402
    CALLED,
    OPENAI_AGENTS_VERSION,
    OPENAI_VERSION,
    Agent,
    HostedMCPTool,
    InProcessMCPServer,
    MCPServer,
    Policy,
    ScriptedModel,
    approval_request,
    approval_responses,
    assistant_message,
    run_agent,
    scripted_model,
    sql_scope_policy,
    tool_outputs,
)
from recusal.audit import AuditLog  # noqa: E402

__all__ = ["OPENAI_AGENTS_VERSION", "OPENAI_VERSION", "SURFACES", "audited_agent", "recording"]


def run_calls(
    calls: list[tuple[str, Any, str]],
    log: AuditLog,
    *,
    policy: Policy = sql_scope_policy,
    server: Optional[MCPServer] = None,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], list[tuple[str, str, dict[str, Any]]]]:
    """Propose the calls to an agent whose tools come from a local MCP server behind the
    audited gate; return each call's output and what the server was asked to run."""
    CALLED.clear()
    model = scripted_model(calls)
    agent = Agent(name="assistant", model=model, mcp_servers=[server or InProcessMCPServer()])
    agent = audited_agent(agent, policy, log, actor=actor, control=control)
    asyncio.run(run_agent(agent))
    return tool_outputs(model), list(CALLED)


def run_hosted(
    requests: list[tuple[str, Any, str]],
    log: AuditLog,
    *,
    policy: Policy = sql_scope_policy,
    tool: Optional[HostedMCPTool] = None,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> dict[str, dict[str, Any]]:
    """Script the approval requests OpenAI would send for hosted calls, behind the audited
    gate; return the SDK's approval responses by request id."""
    model = ScriptedModel(
        [
            [
                approval_request(name, arguments, request_id)
                for name, arguments, request_id in requests
            ],
            [assistant_message("done")],
        ]
    )
    hosted = tool or HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "db",
            "server_url": "https://example.invalid/mcp",
            "require_approval": "always",
        },
        on_approval_request=lambda request: {"approve": True},
    )
    agent = Agent(name="assistant", model=model, tools=[hosted])
    agent = audited_agent(agent, policy, log, actor=actor, control=control)
    asyncio.run(run_agent(agent))
    return approval_responses(model)


def verify_with_cli(path: str) -> tuple[int, str]:
    """Run ``recusal audit verify`` on the log the way an auditor would; (exit code, text)."""
    result = subprocess.run(
        [sys.executable, "-m", "recusal", "audit", "verify", path],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def tamper(path: str, seq: int, decision: str = "allow") -> None:
    """Rewrite one entry's recorded decision in place, the quiet edit the chain must catch.
    By sequence number: a local server's entries carry no call id."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for index, line in enumerate(lines):
        entry = json.loads(line)
        if entry["seq"] == seq:
            entry["action"]["decision"] = decision
            lines[index] = json.dumps(entry, sort_keys=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "openai-agents-mcp-audit.jsonl")
        log = AuditLog(path=path)
        _, called = run_calls(calls, log, actor="demo-agent")
        run_hosted([(name, args, f"r{i}") for i, (name, args, _) in enumerate(calls, 1)], log)

        print("RECUSAL - OpenAI Agents SDK MCP calls, every call on the record (offline)\n")
        print(f"  the local server ran: {', '.join(tool for _, tool, _ in called)}\n")
        for entry in log.entries:
            action = entry["action"]
            print(
                f"  {entry['seq']} {action['surface']:<34} {action['tool']:<8} "
                f"{action['decision']:<5} id={action.get('tool_use_id', '-'):<3} "
                f"server={action['control']['mcp_server']} {entry['hash'][:12]}..."
            )

        code, text = verify_with_cli(path)
        print(f"\n  verify intact:   exit {code}  {text}")

        refused = next(e["seq"] for e in log.entries if e["action"]["decision"] == "deny")
        tamper(path, refused)  # a refused DELETE now reads as allowed
        code, text = verify_with_cli(path)
        print(f"  verify tampered: exit {code}  {text}")


if __name__ == "__main__":
    main()
