# Examples

Runnable demos of Recusal. The offline ones need **no API key** and are exactly what CI
runs (and what the screenshots come from); run any of them straight from a clone:

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
| [`claude_agent_live.py`](claude_agent_live.py) | The **live** version (real Anthropic SDK, manual agent loop): Claude proposes a wrong-subject write, Recusal refuses, Claude self-corrects. Needs `pip install anthropic` + a key. |
| [`agent_loop.py`](agent_loop.py) | **Framework-neutral**, a full gate in a plain `propose → gate → act` loop whose only import is `recusal`. No Claude, no SDK. Proof it works in any agent loop. |

## The other surfaces (offline, no key)

| Example | What it shows |
|---|---|
| [`audit_demo.py`](audit_demo.py) | The tamper-evident **audit log**: record verdicts, verify the hash chain, edit one record, watch the chain catch it. |
| [`classify_demo.py`](classify_demo.py) | The deterministic **failure classifier/router**, a failure string in, a class + remediation route out (retry / refuse / quarantine / ask-human). |
| [`scenarios.py`](scenarios.py) | The reusable policy library behind `gallery.py` *and* the test suite, the same policies are demonstrated **and** proven. Import, don't run. |

---

Want a complete, narrated setup for one real use case? See the
[**worked example**](../docs/EXAMPLE.md) (a database-admin agent left in auto mode). Looking
for policies to drop into your own gate? See the [**policy cookbook**](../docs/COOKBOOK.md).
New to the project? Start with the [README](../README.md) and the [FAQ](../docs/FAQ.md).
