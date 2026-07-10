# Examples

Runnable demos of Recusal. The offline ones need **no API key** and are exactly what CI
runs (capture any of them for a demo screenshot); run any of them straight from a clone:

```bash
git clone https://github.com/philpaz/recusal && cd recusal
python examples/claude_refusal.py
```

## Start here, see the refusal (offline, no key)

| Example | What it shows |
|---|---|
| [`claude_refusal.py`](claude_refusal.py) | A Claude agent stages a write to the **wrong customer**; the gate refuses *before the tool runs* and hands Claude a reason it can act on. The signature demo. |
| [`gallery.py`](gallery.py) | The same gate across the common autonomous-agent failure modes, mapped to the **OWASP Top 10 for Agentic Applications** (refuse / retry / allow). |
| [`quickstart.py`](quickstart.py) | The kernel directly: collect evidence → `compute_verdict` → PASS/RETRY/FAIL, then a staged release gate that refuses to ship. |

## Plug it into an agent

| Example | What it shows |
|---|---|
| [`claude_code_gate.py`](claude_code_gate.py) | A drop-in Claude Code **`PreToolUse` hook**, refuses destructive bash and secret-file writes even under `bypassPermissions`; defers on anything it has no opinion on. |
| [`allowlist_gate.py`](allowlist_gate.py) | **Default-deny** (the stronger posture), wiring the shipped `recusal.claude_code.allowlist_policy`: the same runtime-constructed `rm` that a deny-list *defers*, the allowlist *refuses*, bare interpreters (`python script.py`) included, while a vetted binary still runs. Shows the ceiling a deny-list cannot clear. Runs as a demo or a hook. |
| [`mcp_governance.py`](mcp_governance.py) | **MCP calls through the same gate**: `mcp__<server>__<tool>` names hit the same `PreToolUse` policy seam as native tools, so approved-server pinning, destructive-verb refusal, repo/path scope, and allowlist mode (MCP tools refused unless affirmatively named) all apply with no MCP-specific adapter. Honest about the boundary: call-time, not discovery-time. Runs as a demo or a hook. |
| [`mcp_manifest_rugpull.py`](mcp_manifest_rugpull.py) | **The MCP discovery boundary**: pin a reviewed `tools/list` catalog (`recusal.mcp`), then watch a post-approval description rewrite (the **rug pull**) and an unreviewed tool both refuse deterministically, and the same pin refuse an unpinned `mcp__` call at call time (`manifest_policy`, "no pin, no MCP"). |
| [`mcp_full_stack.py`](mcp_full_stack.py) | **All three MCP boundaries composed** (cookbook recipe 15): one `manifest_policy(policy=...)` gate runs discovery (unpinned tools refuse) then invocation (your argument-level rules), and a separate output screen quarantines a poisoned tool *response*. Offline. |
| [`claude_agent_live.py`](claude_agent_live.py) | The **live** version (real Anthropic SDK, manual agent loop): Claude proposes a wrong-subject write, Recusal refuses, Claude self-corrects. Needs `pip install anthropic` + a key. |
| [`agent_loop.py`](agent_loop.py) | **Framework-neutral**, a full gate in a plain `propose → gate → act` loop whose only import is `recusal`. No Claude, no SDK. Proof it works in any agent loop. |

## The other surfaces (offline, no key)

| Example | What it shows |
|---|---|
| [`audit_demo.py`](audit_demo.py) | The tamper-evident **audit log**: record verdicts, verify the hash chain, edit one record, watch the chain catch it. |
| [`classify_demo.py`](classify_demo.py) | The deterministic **failure classifier/router**, a failure string in, a class + remediation route out (retry / refuse / quarantine / ask-human). |
| [`injection_quarantine.py`](injection_quarantine.py) | **Quarantine prompt-injection in tool output** (OWASP LLM01 / ASI01; MITRE ATLAS AML.T0086). Adjudicate what a tool *returned* before the agent acts on it: poisoned output is refused and routed to `quarantine`, never fed back as trusted context. Cookbook recipe 6, made runnable. |
| [`scenarios.py`](scenarios.py) | The reusable policy library behind `gallery.py` *and* the test suite, the same policies are demonstrated **and** proven. Import, don't run. |

---

Want a complete, narrated setup for one real use case? See the
[**worked example**](../docs/EXAMPLE.md) (a database-admin agent left in auto mode). Looking
for policies to drop into your own gate? See the [**policy cookbook**](../docs/COOKBOOK.md).
New to the project? Start with the [README](../README.md) and the [FAQ](../docs/FAQ.md).
