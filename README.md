<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-gate-strip.png">
    <img alt="recusal, deterministic governance for Claude agents" src="assets/banner-gate-strip-light.png" width="880">
  </picture>
</p>

# Recusal

**Deterministic governance for Claude agents: an independent verifier that can refuse to certify a tool call *before* it runs.**

**Lightweight** (zero dependencies) · **extensible** (a check is just a function that returns a finding) · **Claude-native** (drops into Claude Code as a hook and the Claude Agent SDK as a tool gate). The zero-dep core works in any agent loop.

[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)
![runtime deps](https://img.shields.io/badge/runtime%20deps-0-brightgreen)

A judge **recuses** themselves from a case they can't impartially decide. The same
principle governs autonomous agents: the thing that *generates* the work must never be
the thing that *certifies* it. Recusal is that independent authority: collect evidence,
adjudicate it into **`PASS` / `RETRY` / `FAIL`**, and let the gate **refuse**. No model
call in the decision path. Same evidence, same verdict, every time, including the "no".

---

## The wedge: don't let the same model grade its own work

The reflex fix for agent safety is *another model*: "does this action look OK?" But the
judge and the builder come from the same family, share the same blind spots, and drift
together. That's not a control; it's a conflict of interest.

2026 made the failure mode concrete:

- An **[Anthropic study](https://www.anthropic.com/research/emergent-misalignment-reward-hacking)** found an RL-trained coding model that learned to call `sys.exit(0)`
  to fake passing tests, and *generalized* the cheating to unrelated tasks.
- **[UC Berkeley](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)** scored **100%** on six of eight agent benchmarks **without solving a single
  task**, by intercepting the evaluator.

A model will, given the chance, certify its own success. Even Anthropic's own Claude Code
auto-mode safety layer is a same-family classifier with an admitted 17% false-negative rate
on a curated set of hard cases,
and [Anthropic says plainly](https://www.anthropic.com/engineering/claude-code-auto-mode) it is *"not a drop-in replacement for careful human review on
high-stakes infrastructure."*

Recusal is **an** independent, deterministic authority with no model in the decision path,
a verdict you can replay and audit, and a refusal that holds
(a Claude Code `deny` is honored even under `bypassPermissions`).

---

## Positioning

- **Lightweight**: zero dependencies, a kernel you can read in one sitting. Governance without a Kubernetes-sized platform.
- **Extensible**: a check is just a function that returns a finding. Bring your own policy; the core never changes.
- **Claude-native**: drops into Claude Code as a `PreToolUse` hook and the Agent SDK as a tool gate; a `deny` holds even under `bypassPermissions`.
- **Independent & deterministic**: not the same model grading its own work. No model in the decision path; a verdict you can replay and audit.

> Builders generate. Recusal certifies. Refusal is a feature.

---

## Architecture

One object model, one pipeline. Checks (or your own evidence) produce **Findings**;
`compute_verdict` folds them into a **Verdict**; and the Verdict drives every surface:
the gate refuses, the audit log records, the classifier routes.

```
  data / a proposed agent action / a tool call
          │
     [ checks ]            emit Findings               (recusal.checks, or your own)
          │
   compute_verdict()       fold findings → one Verdict (PASS / RETRY / FAIL)
          │
       Verdict
        │   │   │
        │   │   └─ recusal.classify        route the failure (retry / refuse / ask-human / …)
        │   └───── recusal.audit           tamper-evident, hash-chained record
        └───────── recusal.claude(_code)   allow or refuse the tool call
                   recusal.gates           staged G0-G8 release decision
```

| Module | What it is |
|---|---|
| `recusal.evidence` | the contract, `Finding`, `Verdict`, `Severity`, `Decision`, `compute_verdict` |
| `recusal.checks` | built-in deterministic checks that turn data into Findings |
| `recusal.claude` · `recusal.claude_code` | gate a Claude agent's tool calls (SDK loop, Managed Agents, Claude Code hook) |
| `recusal.deny_list` · `recusal.claude_code.allowlist_policy` | ready-made policies: a reference deny-list (refuse known-bad) and default-deny allowlist |
| `recusal.audit` | tamper-evident, hash-chained log of every verdict |
| `recusal.classify` | deterministic failure classifier + router |
| `recusal.gates` | staged `G0`-`G8` release-gate adjudication, `compute_verdict` at each checkpoint |

Zero runtime dependencies, standard library only.

---

## Install

```bash
pip install recusal
```

## See it refuse (20 seconds, no API key)

```bash
git clone https://github.com/philpaz/recusal && cd recusal
python examples/claude_refusal.py   # a Claude agent stages a write to the WRONG
                                    # customer; the gate refuses before the tool runs
python examples/gallery.py          # the same gate across the OWASP agentic failure modes
```

Deterministic and offline, same evidence, same verdict, including the **no**.

## Plug it into Claude

### Claude Code, drop-in `PreToolUse` hook

Refuse destructive tool calls *before* Claude Code runs them, even in auto / bypass mode.
Register a hook in `.claude/settings.json`:

```json
{ "hooks": { "PreToolUse": [
  { "matcher": ".*", "hooks": [
    { "type": "command", "command": "for p in python3 python py; do \"$p\" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null && { \"$p\" \"$CLAUDE_PROJECT_DIR/.claude/hooks/my_gate.py\"; rc=$?; [ \"$rc\" = 0 ] || { echo 'gate: hook did not run cleanly; failing closed' >&2; exit 2; }; exit 0; }; done; echo 'gate: no python>=3.9; failing closed' >&2; exit 2" } ]}
]}}
```

The command runs the first `python3` → `python` → `py` that is `>=3.9` (macOS / Linux /
Windows) and **fails closed**: Claude Code treats a hook whose command can't launch, or
that exits with anything other than `2`, as a *non-blocking* error and lets the tool call
proceed, so a bare `python3` on a Windows machine (no `python3` on PATH), a `python` that
is Python 2, or a hook that raises at import would each silently disable the gate. The loop
coerces every one of those into `exit 2`, the one *blocking* hook exit code, so a broken or
absent interpreter refuses the tool call instead of waving it through. (On Windows, Claude
Code runs hook commands under Git Bash.)

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

**Two paths, one principle, pick by your channel, not by a ranking.** The policy above is
a **deny-list**: name the known-bad calls, *defer everything else*. It drops into a broad,
open-ended channel with almost no friction and needs no inventory of your tools, which is
why this repo dogfoods it (a general-purpose dev repo runs an unbounded set of legitimate
commands). Its boundary is inherent, not a defect: a literal matcher can be obfuscated past,
and `python script.py` runs code no string check ever reads, so a deny-list never earns
"cannot be subverted."

The other path is **allowlist mode** (default-deny): name the affirmatively-safe calls,
*refuse everything else*. It fits a narrow, enumerable, high-stakes channel, nothing runs
unless listed, and bare interpreters and shell metacharacters are refused, which closes the
write-a-script-then-run-it bypass by construction (pinned as a test). That closure is what
lets it earn *"the agent could not subvert it"* for the routed tool channel. The trade is
friction and maintenance: you enumerate and grow the capability set, and it fails toward
refusal until you do.

Neither is "better" in the abstract, a deny-list refusing the unknown would grind a broad
channel to a halt, and an allowlist deferring the unknown would defeat the point of a
high-stakes one. Choose by the channel. Both ship, both are pinned as tests.

```python
from recusal.claude_code import allowlist_policy, run_pretooluse_hook

run_pretooluse_hook(allowlist_policy(writable_root="./workspace"))
```

> **Don't start from a blank policy.** [`docs/COOKBOOK.md`](docs/COOKBOOK.md) has copy-paste
> recipes, destructive shell, unscoped SQL, secret-file writes, wrong-subject writes, egress
> allowlists, injection quarantine, action budgets, that drop straight into the hook above.

> **Recusal governs *this* repository exactly this way**, a real hook refuses `rm -rf`,
> force-pushes, and secret-file writes to its own maintainers. Verbatim, reproducible,
> CI-locked proof: [`docs/PROVEN.md`](docs/PROVEN.md).

### Claude Agent SDK, manual loop

In a manual agent loop, gate each tool call and hand Claude an `is_error` tool_result on a
refusal, it self-corrects:

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
Agents** `always_ask`, `recusal.claude.tool_confirmation` is the deterministic decider
(the SDK event shape is illustrative; verify it against your Agent SDK version).

### Any agent loop, no Claude required

The Claude adapters are conveniences; the zero-dep core is framework-neutral.
[`examples/agent_loop.py`](examples/agent_loop.py) gates a plain `propose → gate → act`
loop whose only import is `recusal`, the same `compute_verdict` seam drops into LangGraph,
the OpenAI Agents SDK, or a homegrown runtime unchanged.

## Robustness, across the OWASP Agentic failure modes

`python examples/gallery.py` runs the gate against the common autonomous-agent failure modes:

```
  scenario                OWASP                 verdict outcome
  wrong-subject write     ASI03 Identity Abuse  FAIL    REFUSE
  destructive file delete ASI02 Tool Misuse     FAIL    REFUSE
  unscoped SQL mutation   ASI05 Code Execution  FAIL    REFUSE
  data exfiltration       ASI01 Goal Hijack     FAIL    REFUSE
  coverage floor          quality gate          RETRY   BLOCK (retry)
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
    row_count(users, min_rows=1),                                  # CRITICAL if empty
    null_rate(users, "email", max_rate=0.10),                      # ERROR if too sparse
    referential_integrity(orders, users, fk="user_id", pk="id"),   # CRITICAL on orphans
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

## Tamper-evident audit

Pair any verdict with an append-only, hash-chained log: every decision on the record, and
an in-place edit or reordering of any entry with a surviving successor is detectable
(catching tail-truncation, a tail-suffix rewrite, or a forged append by a write-access
attacker needs an external anchor, see `recusal.audit`):

```python
from recusal import AuditLog, verify

audit = AuditLog(path="audit.jsonl")
audit.append(verdict, action={"tool": "Bash", "command": "rm -rf /"})
ok, problems = verify(audit.entries)   # False if an entry with a later entry was edited/reordered
```

Deterministic, stdlib-only, and shaped for OWASP Agentic logging / EU AI Act Article 12 (record-keeping).

## Classify and route a failure

A refusal or failure is only useful if you know what to do next. The classifier says
*what kind* of failure it is and where it routes, deterministically, no model:

```python
from recusal import classify_failure

c = classify_failure("Traceback ... TypeError: 'NoneType' object")
c.failure_class   # "code_bug"
c.route           # "fix-code"
```

Default taxonomy (extend or replace it): `transient → retry` · `policy_violation → refuse` ·
`prompt_injection → quarantine` · `code_bug → fix-code` · `data_shape → fix-data` ·
`data_missing → fetch-data` · `spec_ambiguity → ask-human`. Unmatched failures fall back to
`ask-human`, it never guesses. `classify_verdict(verdict)` routes a non-PASS verdict.

## Why this, and why not the alternatives

- **Agent frameworks** (LangGraph, CrewAI, AutoGen, OpenAI Agents SDK) *orchestrate*, their governance is in-process and self-graded.
- **Guardrails** (Guardrails AI, NeMo Guardrails) *filter I/O content*, they don't adjudicate a work product.
- **Eval libraries** (promptfoo, DeepEval, Haize's Verdict) *score offline*, usually with an LLM-as-judge, the probabilistic opposite of a deterministic gate.
- **Observability** (Langfuse, AgentOps) *records*, zero authority to stop anything.
- **Anthropic's auto mode** is a *same-family classifier* grading the same family, exactly the conflict of interest this exists to remove.
- The newer **agent-firewall** projects (e.g. AEGIS) and **Microsoft's Agent Governance Toolkit** are real peers. Recusal's bet is not feature parity, it's **independence** (a verifier the builder cannot influence) and a kernel small enough to trust on sight.

**New here?** The quick objections, *do I need this? doesn't Claude already do it? is it
ready to use?*, are answered in the [`docs/FAQ.md`](docs/FAQ.md). The plain-terms "so
what": [`docs/WHY.md`](docs/WHY.md).

**Full documentation index: [`docs/`](docs/README.md).** Comparison with the landscape:
[`docs/LANDSCAPE.md`](docs/LANDSCAPE.md). The principles and why each helps:
[`CONSTITUTION.md`](CONSTITUTION.md). The contract: [`docs/EVIDENCE.md`](docs/EVIDENCE.md).
Usage & extending: [`docs/HOWTO.md`](docs/HOWTO.md) · [`docs/EXTENDING.md`](docs/EXTENDING.md).
Copy-paste policies: [`docs/COOKBOOK.md`](docs/COOKBOOK.md).
A full worked configuration: [`docs/EXAMPLE.md`](docs/EXAMPLE.md).
Proof it governs itself: [`docs/PROVEN.md`](docs/PROVEN.md).

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Contributing

Contributions are welcome, Recusal is deliberately small, and the bar is keeping it that
way (no model in the verdict path, no runtime dependencies, don't grow the kernel). Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) and the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
first. Security reports go through [`SECURITY.md`](SECURITY.md), privately.

## License

Apache-2.0 © Philip Paz
