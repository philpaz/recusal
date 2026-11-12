# Examples

Runnable demos of Recusal. The offline ones need **no API key** and are exactly what CI
runs (capture any of them for a demo screenshot); run any of them straight from a clone:

```bash
git clone https://github.com/philpaz/recusal && cd recusal
python examples/claude_refusal.py
```

**Just want to see it refuse?** `recusal demo` ships inside the package and needs no clone:
`pip install recusal && recusal demo` runs four scenarios back to back, three of them
short versions of these narratives. These
files are the long form, annotated and adaptable; the shipped demo is the thirty-second one.

## Start here, see the refusal (offline, no key)

| Example | What it shows |
|---|---|
| [`claude_refusal.py`](claude_refusal.py) | A Claude agent stages a write to the **wrong customer**; the gate refuses *before the tool runs* and hands Claude a reason it can act on. The signature demo. |
| [`gallery.py`](gallery.py) | The same gate across the common autonomous-agent failure modes, mapped to the **OWASP Top 10 for Agentic Applications** (refuse / retry / allow). |
| [`quickstart.py`](quickstart.py) | The kernel directly: collect evidence → `compute_verdict` → PASS/RETRY/FAIL, then a staged release gate that refuses to ship. |

## Plug it into an agent

| Example | What it shows |
|---|---|
| [`composed_policies.py`](composed_policies.py) | **Cookbook recipe 10:** combine the existing shell, workspace and egress policies in one offline, test-backed gate. A clean call defers; a policy exception fails closed. No shell or network call is run. |
| [`claude_code_gate.py`](claude_code_gate.py) | A drop-in Claude Code **`PreToolUse` hook**, refuses destructive bash and secret-file writes even under `bypassPermissions`; defers on anything it has no opinion on. |
| [`allowlist_gate.py`](allowlist_gate.py) | **Default-deny** (the stronger posture), wiring the shipped `recusal.claude_code.allowlist_policy`: the same runtime-constructed `rm` that a deny-list *defers*, the allowlist *refuses*, bare interpreters (`python script.py`) included, while a vetted binary still runs. Shows the ceiling a deny-list cannot clear. Runs as a demo or a hook. |
| [`mcp_governance.py`](mcp_governance.py) | **MCP calls through the same gate**: `mcp__<server>__<tool>` names hit the same `PreToolUse` policy seam as native tools, so approved-server pinning, destructive-verb refusal, repo/path scope, and allowlist mode (MCP tools refused unless affirmatively named) all apply with no MCP-specific adapter. Honest about the boundary: call-time, not discovery-time. Runs as a demo or a hook. |
| [`mcp_manifest_rugpull.py`](mcp_manifest_rugpull.py) | **The MCP discovery boundary**: pin a reviewed `tools/list` catalog (`recusal.mcp`), then watch a post-approval description rewrite (the **rug pull**) and an unreviewed tool both refuse deterministically, and the same pin refuse an unpinned `mcp__` call at call time (`manifest_policy`, "no pin, no MCP"). |
| [`mcp_security_demo.py`](mcp_security_demo.py) | **The executable MCP launch-command proof**: pin a real local MCP process, change its configured command to a marker-writing program, and watch `verify` refuse before the substituted command runs; then refuse instructions-only drift and an unpinned call. Offline, no key. |
| [`mcp_full_stack.py`](mcp_full_stack.py) | **All three MCP boundaries composed** (cookbook recipe 15): one `manifest_policy(policy=...)` gate runs discovery (unpinned tools refuse) then invocation (your argument-level rules), and a separate output screen quarantines a poisoned tool *response*. Offline. |
| [`claude_agent_live.py`](claude_agent_live.py) | The **live** version (real Anthropic SDK, manual agent loop): Claude proposes a wrong-subject write, Recusal refuses, Claude self-corrects. Needs `pip install anthropic` + a key. |
| [`agent_loop.py`](agent_loop.py) | **Framework-neutral**, a full gate in a plain `propose → gate → act` loop whose only import is `recusal`. No Claude, no SDK. Proof it works in any agent loop. |

## Cookbook recipe modules (offline, no key)

These are the standalone policy implementations from the [cookbook](../docs/COOKBOOK.md).
Some are imported by other examples rather than run directly.

| Recipe | Example | What it refuses |
|---|---|---|
| 1 | [`destructive_shell_policy.py`](destructive_shell_policy.py) | Destructive shell commands, protected file operations and extra local deny-list markers. |
| 2 | [`sql_scope_policy.py`](sql_scope_policy.py) | `DELETE` or `UPDATE` without a top-level `WHERE`, including nested mutations; it is a lexical guard, not a SQL parser. |
| 3 | [`workspace_policy.py`](workspace_policy.py) | Explicit writes outside the allowed workspace or to protected credential paths. |
| 4 | [`subject_guard.py`](subject_guard.py) | Record writes without a bound subject or to a different subject. |
| 5 | [`egress_allowlist.py`](egress_allowlist.py) | Email or HTTP destinations missing from the configured allowlist (only literal tool arguments are inspected). |
| 7 | [`action_budget.py`](action_budget.py) | Actions beyond a persisted hard attempt budget (returns a RETRY verdict; the soft limit only warns). |
| 8 | [`quality_gate.py`](quality_gate.py) | Missing or invalid quality measurements, low coverage and failing tests (RETRY, without starting a deployment). |
| 9 | [`approved_tool_policy.py`](approved_tool_policy.py) | Unnamed, malformed or unapproved tool names; approved names still need argument checks. |

## Framework adapters (offline, optional pinned SDKs)

Each adapter uses scripted calls, not a live model or API key. Install the versions shown
in its module docstring before running it; the Recusal core itself has no SDK dependencies.

| Example | Tested dependencies | What it shows |
|---|---|---|
| [`langgraph_gate.py`](langgraph_gate.py) | `langgraph==1.2.12`, `langchain-core==1.6.5` | A LangGraph `ToolNode` gate refuses unsafe sync and async tool calls before execution. |
| [`langgraph_audit.py`](langgraph_audit.py) | `langgraph==1.2.12`, `langchain-core==1.6.5` | The same gate writes an audit record for allowed and refused calls, and denies a call if recording fails. |
| [`langchain_agent_gate.py`](langchain_agent_gate.py) | `langchain==1.4.3`, `langgraph==1.2.12`, `langchain-core==1.6.5` | A LangChain `create_agent` middleware gates calls from a scripted model without a provider connection. |
| [`openai_agents_gate.py`](openai_agents_gate.py) | `openai-agents==0.24.0`, `openai==3.34.0` | An Agents SDK input guardrail refuses unsafe function tools and unsupported tool surfaces. |
| [`openai_agents_audit.py`](openai_agents_audit.py) | `openai-agents==0.24.0`, `openai==3.34.0` | The same gate writes an audit record for allowed and refused function tool calls, and denies a call if recording fails. |
| [`openai_agents_mcp_gate.py`](openai_agents_mcp_gate.py) | `openai-agents==0.24.0`, `openai==3.34.0` | A gate on a local MCP server's own `call_tool`, judging the tool name and arguments the server runs, and the approval callback as the gate for hosted MCP. |
| [`openai_responses_gate.py`](openai_responses_gate.py) | `openai==3.34.0` | An application-controlled Responses or Chat Completions loop refuses malformed, unknown or unsafe function calls. |
| [`openai_responses_audit.py`](openai_responses_audit.py) | `openai==3.34.0` | The same loop writes an audit record for run and refused calls, and denies a call if recording fails. |
| [`google_adk_gate.py`](google_adk_gate.py) | `google-adk==2.13.0`, `google-genai==2.34.0` | A Google ADK before-tool callback denies calls judged unsafe, including those in nested agents. |
| [`google_adk_audit.py`](google_adk_audit.py) | `google-adk==2.13.0`, `google-genai==2.34.0` | The same gate writes an audit record for allowed and refused calls, and denies a call if recording fails. |
| [`crewai_gate.py`](crewai_gate.py) | `crewai==1.15.34` | CrewAI before/after hooks stop refused tool calls and return the gate's reason to the scripted model. |
| [`pydantic_ai_gate.py`](pydantic_ai_gate.py) | `pydantic-ai-slim==2.64.0` | A Pydantic AI capability judges each call in `before_tool_execute` and answers a refusal as a failed tool result; a delegated agent needs its own gate. |
| [`pydantic_ai_audit.py`](pydantic_ai_audit.py) | `pydantic-ai-slim==2.64.0` | The same gate writes an audit record for allowed and refused calls, and denies a call if recording fails. |

## The other surfaces (offline, no key)

| Example | What it shows |
|---|---|
| [`audit_demo.py`](audit_demo.py) | The tamper-evident **audit log**: record verdicts, verify the hash chain, edit one record, watch the chain catch it. |
| [`audit_sink.py`](audit_sink.py) | A local audit-log mirror and saved-head check: truncated history is refused on resume. This is not an external tamper-proof anchor. |
| [`classify_demo.py`](classify_demo.py) | The deterministic **failure classifier/router**, a failure string in, a class + remediation route out (retry / refuse / quarantine / ask-human). |
| [`injection_quarantine.py`](injection_quarantine.py) | **Quarantine prompt-injection in tool output** (OWASP LLM01 / ASI01; MITRE ATLAS AML.T0086). Adjudicate what a tool *returned* before the agent acts on it: poisoned output is refused and routed to `quarantine`, never fed back as trusted context. Cookbook recipe 6, made runnable. |
| [`scenarios.py`](scenarios.py) | The reusable policy library behind `gallery.py` *and* the test suite, the same policies are demonstrated **and** proven. Import, don't run. |

---

Want a complete, narrated setup for one real use case? See the
[**worked example**](../docs/EXAMPLE.md) (a database-admin agent left in auto mode). Looking
for policies to drop into your own gate? See the [**policy cookbook**](../docs/COOKBOOK.md).
New to the project? Start with the [README](../README.md) and the [FAQ](../docs/FAQ.md).
