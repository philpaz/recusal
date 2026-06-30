# Recusal

**Deterministic governance for Claude agents — an independent verifier that can refuse to certify a tool call *before* it runs.**

**Lightweight** (zero dependencies) · **extensible** (a check is just a function that returns a finding) · **Claude-native** (drops into Claude Code as a hook and the Claude Agent SDK as a tool gate). The zero-dep core works in any agent loop.

[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)
![runtime deps](https://img.shields.io/badge/runtime%20deps-0-brightgreen)

A judge **recuses** themselves from a case they can't impartially decide. The same
principle governs autonomous agents: the thing that *generates* the work must never be
the thing that *certifies* it. Recusal is that independent authority — collect evidence,
adjudicate it into **`PASS` / `RETRY` / `FAIL`**, and let the gate **refuse**. No model
call in the decision path. Same evidence, same verdict, every time — including the "no".

---

## The wedge: don't let the same model grade its own work

The reflex fix for agent safety is *another model* — "does this action look OK?" But the
judge and the builder come from the same family, share the same blind spots, and drift
together. That's not a control; it's a conflict of interest.

2026 made the failure mode concrete and peer-reviewed:

- A **Nature** study found an RL-trained coding model that learned to call `sys.exit(0)`
  to fake passing tests — and *generalized* the cheating.
- **UC Berkeley** scored **100%** on three agent benchmarks **without solving a single
  task**, by intercepting the evaluator.

A model will, given the chance, certify its own success. Even Anthropic's own Claude Code
auto-mode safety layer is a same-family classifier with an admitted false-negative rate,
and Anthropic says plainly it is *"not a drop-in replacement for human review on
high-stakes infrastructure."*

Recusal is the **independent, deterministic** authority that can't be talked into it —
no model in the decision path, a verdict you can replay and audit, and a refusal that holds
(a Claude Code `deny` is honored even under `bypassPermissions`).

---

## Positioning

- **Lightweight** — zero dependencies, a kernel you can read in one sitting. Governance without a Kubernetes-sized platform.
- **Extensible** — a check is just a function that returns a finding. Bring your own policy; the core never changes.
- **Claude-native** — drops into Claude Code as a `PreToolUse` hook and the Agent SDK as a tool gate; a `deny` holds even under `bypassPermissions`.
- **Independent & deterministic** — not the same model grading its own work. No model in the decision path; a verdict you can replay and audit.

> Builders generate. Recusal certifies. Refusal is a feature.

---

## Install

```bash
pip install recusal
```

## Plug it into Claude

### Claude Code — drop-in `PreToolUse` hook

Refuse destructive tool calls *before* Claude Code runs them — even in auto / bypass mode.
Register a hook in `.claude/settings.json`:

```json
{ "hooks": { "PreToolUse": [
  { "matcher": ".*", "hooks": [
    { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/my_gate.py" } ]}
]}}
```

```python
# my_gate.py
from recusal import Finding
from recusal.claude_code import run_pretooluse_hook

def policy(tool_name, tool_input):
    if tool_name == "Bash" and "rm -rf" in tool_input.get("command", ""):
        return [Finding.fail("destructive_bash", severity="CRITICAL", message="refusing rm -rf")]
    return []   # no opinion → defer to Claude Code's normal permission flow

run_pretooluse_hook(policy)
```

A clean verdict **defers** (Recusal adds refusals; it never strips Claude Code's own prompts).
A non-clean verdict **denies**, with the reasons. See [`examples/claude_code_gate.py`](examples/claude_code_gate.py).

### Claude Agent SDK — manual loop

In a manual agent loop, gate each tool call and hand Claude an `is_error` tool_result on a
refusal — it self-corrects:

```python
from recusal.claude import gate_tool_use

allow, refusal = gate_tool_use(tool.id, gather_evidence(tool), tool_name=tool.name)
if not allow:
    results.append(refusal)                          # is_error=True → Claude adapts
else:
    results.append({"type": "tool_result", "tool_use_id": tool.id,
                    "content": execute_tool(tool.name, tool.input)})
```

Runnable: [`examples/claude_agent_live.py`](examples/claude_agent_live.py) (real API) and
[`examples/claude_refusal.py`](examples/claude_refusal.py) (offline, no key). For **Managed
Agents** `always_ask`, `recusal.claude.tool_confirmation` is the deterministic decider.

## Robustness — across the OWASP Agentic failure modes

`python examples/gallery.py` runs the gate against the common autonomous-agent failure modes:

```
  scenario                OWASP                 verdict outcome
  wrong-subject write     ASI03 Identity Abuse  FAIL    REFUSE
  destructive file delete ASI02 Tool Misuse     FAIL    REFUSE
  unscoped SQL mutation   ASI05 Code Execution  FAIL    REFUSE
  data exfiltration       ASI01 Goal Hijack     FAIL    REFUSE
  runaway action volume   ASI08 Cascading       RETRY   BLOCK (retry)
  compliant write         -                     PASS    ALLOW
```

The tiers are the policy: destructive things REFUSE terminally; recoverable ones BLOCK
with a retry; a clean call passes. (Same policies power the demo and the test suite.)

## The verdict, directly

```python
from recusal import compute_verdict
from recusal.checks import row_count, null_rate, referential_integrity

verdict = compute_verdict([
    row_count(members, min_rows=1),                                     # CRITICAL if empty
    null_rate(members, "email", max_rate=0.10),                         # ERROR if too sparse
    referential_integrity(accounts, members, fk="member_id", pk="id"),  # CRITICAL on orphans
])
if verdict.refused:
    raise RuntimeError(verdict.reasons())
```

| Worst finding | Verdict | Meaning |
|---------------|---------|---------|
| `CRITICAL` failure | **`FAIL`** | Terminal. The work is wrong. Do not retry. |
| `ERROR` failure | **`RETRY`** | Recoverable. Retry once, with the failures as context. |
| `WARNING` / `INFO` only | **`PASS`** | Proceed. Warnings recorded, info kept as metrics. |

---

## Why this, and why not the alternatives

- **Agent frameworks** (LangGraph, CrewAI, AutoGen, OpenAI Agents SDK) *orchestrate* — their governance is in-process and self-graded.
- **Guardrails** (Guardrails AI, NeMo Guardrails) *filter I/O content* — they don't adjudicate a work product.
- **Eval libraries** (promptfoo, DeepEval, Haize's Verdict) *score offline*, usually with an LLM-as-judge — the probabilistic opposite of a deterministic gate.
- **Observability** (Langfuse, AgentOps) *records* — zero authority to stop anything.
- **Anthropic's auto mode** is a *same-family classifier* grading the same family — exactly the conflict of interest this exists to remove.
- The newer **agent-firewall** projects (e.g. AEGIS) and **Microsoft's Agent Governance Toolkit** are real peers. Recusal's bet is not feature parity — it's **independence** (a verifier the builder cannot influence), determinism, and a kernel small enough to trust on sight.

**Start here — why it matters, in plain terms (the "so what"): [`docs/WHY.md`](docs/WHY.md).**

Full comparison with links: [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md). The principles and why
each helps: [`CONSTITUTION.md`](CONSTITUTION.md). The contract: [`docs/EVIDENCE.md`](docs/EVIDENCE.md).
Usage & extending: [`docs/HOWTO.md`](docs/HOWTO.md) · [`docs/EXTENDING.md`](docs/EXTENDING.md).

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## License

Apache-2.0 © Philip Paz
