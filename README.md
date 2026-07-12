<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-gate-strip.png">
    <img alt="recusal, deterministic governance for Claude agents" src="assets/banner-gate-strip-light.png" width="880">
  </picture>
</p>

# Recusal

**Deterministic governance for Claude agents: an independent verifier that can refuse to certify a tool call *before* it runs.**

**Lightweight** (zero dependencies) · **extensible** (a check is just a function that returns a finding) · **Claude-native** (drops into Claude Code as a hook, [MCP tool calls included](#mcp-tools-the-same-gate), and the Claude Agent SDK as a tool gate). The zero-dep core works in any agent loop.

[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)
![runtime deps](https://img.shields.io/badge/runtime%20deps-0-brightgreen)

<p align="center">
  <img alt="Two verbatim terminal transcripts: the dogfooded hook refuses rm -rf in a live Claude Code session running under --dangerously-skip-permissions, then the offline demo refuses a write to the wrong customer and allows the corrected call" src="assets/demo-refusal.gif" width="880">
</p>
<p align="center"><sub>Verbatim transcripts, rendered: a live Claude Code session where the repo's own hook refuses <code>rm -rf</code> under <code>--dangerously-skip-permissions</code>, then the offline demo (<code>python examples/claude_refusal.py</code>), no API key.</sub></p>

A judge **recuses** themselves from a case they can't impartially decide. The same
principle governs autonomous agents: the thing that *generates* the work must never be
the thing that *certifies* it. Recusal is that independent authority: collect evidence,
adjudicate it into **`PASS` / `RETRY` / `FAIL`**, and let the gate **refuse**. No model
call in the decision path. Same evidence, same verdict, every time, including the "no".

---

## The wedge: don't let the same model grade its own work

The reflex fix for agent safety is another model asking "does this action look OK?", but a
judge from the same family shares the builder's blind spots and drifts with it: that is a
conflict of interest, not a control. Recusal is an independent, deterministic authority
instead: no model in the decision path, a verdict you can replay and audit, and a refusal
that holds (a Claude Code `deny` is honored even under `bypassPermissions`). "Independent"
means the verdict is produced outside the model's decision path; deployment isolation (who
owns the config, the file permissions, the runtime) remains the adopter's responsibility.

The published evidence (models faking passing tests, benchmarks gamed by intercepting the
evaluator, and the stated limits of Anthropic's own same-family safety layer) is laid out
in [`docs/WHY.md`](docs/WHY.md), every source verified in
[`docs/REFERENCES.md`](docs/REFERENCES.md).

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
| `recusal.mcp` · `recusal.mcp_fetch` | MCP discovery governance: pin a server's tool catalog, refuse drift, enforce the pin at call time (pure kernel); collect a live catalog over stdio (fetcher, the one module that spawns a process) |
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

Deterministic and offline: same evidence, same verdict, including the **no**.

## Plug it into Claude

### Claude Code, drop-in `PreToolUse` hook

Refuse destructive tool calls *before* Claude Code runs them, even in auto / bypass mode.

**One command:**

```bash
python -m recusal init          # or: recusal init
```

scaffolds `.claude/hooks/recusal_gate.py` (the deny-list starter, edit it, it's yours) and
registers the fail-closed launcher in `.claude/settings.json`, merging with (never
clobbering) an existing file; re-running is a no-op, and an existing gate file is never
overwritten. `--posture allowlist` scaffolds the default-deny variant instead. Claude Code
asks you to confirm the new hook on the next session: a permission-changing hook is a
deliberate step.

**Or as a plugin** (one gate across every project, no per-project setup):

```bash
claude plugin marketplace add philpaz/recusal
claude plugin install recusal-gate@recusal
pip install recusal        # the plugin fails CLOSED without it
```

The plugin ships the same deny-list shim; if the `recusal` package is missing it refuses
every tool call rather than silently disabling itself. For a policy tailored to one
project, prefer `python -m recusal init` and edit the scaffolded gate.

Prefer to see exactly what it writes? The manual path is the same two pieces. Register a
hook in `.claude/settings.json`:

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

**Two paths, one principle: pick by your channel, not by a ranking.** The policy above is
a **deny-list**: name the known-bad calls, *defer everything else*. It drops into a broad,
open-ended channel with almost no friction and needs no inventory of your tools, which is
why this repo dogfoods it (a general-purpose dev repo runs an unbounded set of legitimate
commands). Its boundary is inherent, not a defect: a literal matcher can be obfuscated past,
and `python script.py` runs code no string check ever reads, so a deny-list never earns
"cannot be subverted."

The other path is **allowlist mode** (default-deny): name the affirmatively-safe calls,
*refuse everything else*. It fits a narrow, enumerable, high-stakes channel: nothing runs
unless listed, and bare interpreters and shell metacharacters are refused, which closes the
documented command-construction and bare-interpreter bypass classes by construction (pinned
as tests). Within a correctly registered routed tool channel, an unapproved capability is
refused by default rather than inferred safe; what sits outside that channel is named in
[`SECURITY.md`](SECURITY.md). One more honest line: the default-safe tools are *nonmutating*,
not authorized for all data - `cat` can read a credential file - so add path- and
subject-level read rules where confidentiality matters. The trade is friction and
maintenance: you enumerate and grow the capability set, and it fails toward refusal until
you do.

Neither is "better" in the abstract: a deny-list refusing the unknown would grind a broad
channel to a halt, and an allowlist deferring the unknown would defeat the point of a
high-stakes one. Choose by the channel. Both ship, and both are pinned as tests.

```python
from recusal.claude_code import allowlist_policy, run_pretooluse_hook

run_pretooluse_hook(allowlist_policy(writable_root="./workspace"))
```

> **Don't start from a blank policy.** [`docs/COOKBOOK.md`](docs/COOKBOOK.md) has copy-paste
> recipes (destructive shell, unscoped SQL, secret-file writes, wrong-subject writes, egress
> allowlists, injection quarantine, action budgets) that drop straight into the hook above.

> **Recusal governs *this* repository exactly this way**: a real hook refuses `rm -rf`,
> force-pushes, and secret-file writes to its own maintainers. Verbatim, reproducible,
> CI-locked proof: [`docs/PROVEN.md`](docs/PROVEN.md).

### MCP tools, the same gate

MCP server tools reach Claude Code as ordinary tools: the hooks reference documents that
they "appear as regular tools in tool events" (`PreToolUse`, ...) under the naming pattern
`mcp__<server>__<tool>` (`mcp__github__create_issue`, `mcp__filesystem__write_file`). So
the `.*` matcher above already routes every MCP call through the same
`policy(tool_name, tool_input)` seam: no MCP-specific adapter, no extra wiring. The
same call-time controls apply to MCP exactly as to `Bash`: destructive-operation refusal,
repository/record scope, write-path confinement, egress and action budgets, the
tamper-evident audit record. In allowlist mode the posture is stronger still: an MCP tool
is **refused unless affirmatively named** (`allow={"mcp__github__create_issue": vet}`),
the least-privilege default the MCP spec's own security guidance pushes toward. Pinned as
tests.

```python
def policy(tool_name, tool_input):
    if tool_name == "mcp__salesforce__delete_records":
        return [Finding.fail("mcp_destructive_action", severity="CRITICAL",
                             message="bulk Salesforce deletion is not approved")]
    if tool_name == "mcp__github__merge_pull_request":
        repo = tool_input.get("repo")
        if repo not in {"philpaz/recusal"}:
            return [Finding.fail("mcp_repository_scope", severity="CRITICAL",
                                 message=f"repository {repo!r} is outside the approved scope")]
    return []   # defer everything else to Claude Code's normal flow
```

Runnable: [`examples/mcp_governance.py`](examples/mcp_governance.py) (approved-server
pinning, destructive-verb refusal, path confinement, allowlist mode). Pinned:
[`tests/test_mcp_governance.py`](tests/test_mcp_governance.py). Recipe:
[`docs/COOKBOOK.md`](docs/COOKBOOK.md) §12. In a custom Agent SDK or MCP-client loop
nothing intercepts for you: invoke the gate between the model's proposed MCP call and the
client dispatching it (the same `gate_tool_use` seam as below).

**The three MCP boundaries, stated plainly.** A call-time policy adjudicates the proposed
tool name and arguments; MCP has two more boundaries, and Recusal covers each with its own
evidence:

| Boundary | Threat (as the field names it) | Recusal |
|---|---|---|
| Discovery (`tools/list`) | tool-description poisoning (benchmarked against real-world MCP servers by MCPTox), post-approval definition changes (the rug pull), name collisions | **pin + refuse drift**: `recusal mcp pin` / `recusal mcp verify` / `recusal.mcp.manifest_policy` (next section) |
| Invocation (the call) | tool misuse (OWASP ASI02), wrong-subject writes (ASI03), exfiltration via tool invocation (MITRE ATLAS AML.T0086) | **this section** |
| Response (the result) | indirect prompt injection in tool output (OWASP LLM01) | quarantine, [cookbook recipe 6](docs/COOKBOOK.md) |

Transport and authorization threats (confused deputy, token passthrough, session
hijacking) are the MCP specification's own
[Security Best Practices](https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices)
layer, complementary to this gate, neither replaces the other. Every source here is
verified in [`docs/REFERENCES.md`](docs/REFERENCES.md).

### MCP discovery, pin the catalog, refuse the rug pull

The model chooses tools by reading their declared descriptions, so a poisoned declaration
steers the agent *before any call exists* for a call-time policy to see, and the call that
follows looks structurally valid. `recusal.mcp` adds deterministic integrity controls at
that boundary the way this library governs every boundary: deterministic evidence, with
the human where the judgment is:

```bash
recusal mcp pin --claude-config .mcp.json    # review once, pin the approved catalog
recusal mcp verify --claude-config .mcp.json # CI / session start: same catalog, or refuse
```

> **`--claude-config` and `--stdio` execute the declared server commands** to ask them for
> `tools/list`. Treat `.mcp.json` as executable code: review its `command`/`args` lines
> like you review the declarations, and pass `--minimal-env` so a server you are still
> deciding about does not inherit the API keys in your shell.

Recusal does not judge whether a description is *malicious*: that is semantic judgment, a
human's call at pin time (a deterministic marker screen surfaces the obvious, and `pin`
refuses to write over a flagged catalog until `--force` records that a human reviewed it).
What it detects, deterministically, is **unpinned capability and post-approval change**:
the rug pull, the new tool, the mutated schema. The pin is the confirmed human decision
promoted to a deterministic artifact: manifest bytes are reproducible, hashes only (a
poisoned description is never embedded anywhere), and the same observed catalog against
the same pin yields the same verdict, every time. `verify` fails **closed**: a missing
manifest, a failed fetch, a wholly empty observation, or a pinned server that can no longer
be reached for integrity-checking (e.g. silently swapped to a URL transport) is a refusal,
never a clean-looking pass. (A pinned server *legitimately removed* from the config is
recorded as a warning, not refused: a shrunk capability set is not an attack.) The pin also
enforces at call time: `recusal.mcp.manifest_policy("mcp-manifest.json")` drops into the
same `PreToolUse` gate and refuses any `mcp__server__tool` call that was never pinned (no
pin, no MCP), composing with the argument-level rules above. A minimal zero-dependency
stdio client collects `tools/list`; **remote/HTTP servers** are pinned from a JSON dump you
produce with any MCP client (`--from`, copy-paste recipe:
[`docs/COOKBOOK.md`](docs/COOKBOOK.md) §14; local/`.mcp.json` servers pin directly, §13).
Recusal owns the deterministic adjudication, not the transport, so it inherits neither the
HTTP client's dependencies nor its SSRF surface. Collection is never decision: the kernel
adjudicates what was observed.

The honest boundary: this is *discovery-time and call-time* integrity, not a live tap on
every message. `verify` proves the catalog at the moment it runs (wire it into CI and
session start); the call-time gate then enforces *approved tools only*. A server that
serves one catalog to `verify` and a different one to the live session (a client- or
time-discriminating server) is a residual this layer names rather than claims to close:
run `verify` against the same endpoint the session uses, close in time. And the manifest
pins the **declared catalog, not the identity of the process that declares it**: a
rewritten `.mcp.json` command runs at observe time, before the catalog it returns can
fail verification. Until server-launch specifications are pinned (a named roadmap item),
protect `.mcp.json` and `mcp-manifest.json` as control-plane files - the default
deny-list does - and treat the config as executable code.

See the refusal: [`examples/mcp_manifest_rugpull.py`](examples/mcp_manifest_rugpull.py)
(offline). Pinned as tests: [`tests/test_mcp_manifest.py`](tests/test_mcp_manifest.py),
[`tests/test_mcp_policy_bridge.py`](tests/test_mcp_policy_bridge.py),
[`tests/test_mcp_fetch.py`](tests/test_mcp_fetch.py),
[`tests/test_mcp_cli.py`](tests/test_mcp_cli.py).

### Claude Agent SDK, manual loop

In a manual agent loop, gate each tool call and hand Claude an `is_error` tool_result on a
refusal; it self-corrects:

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
loop whose only import is `recusal`; the same `compute_verdict` seam drops into LangGraph,
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

In the Claude Code hook it is one argument: `run_pretooluse_hook(policy,
audit=AuditLog("audit.jsonl", resume="tail"))` puts every adjudication - defer, allow,
and deny - on the chain, with the proposed `tool_input` bound by SHA-256 fingerprint,
never embedded, and an unwritable log failing closed to a deny (the record is part of
the control). `resume="tail"` keeps the per-call cost flat as the log grows.

Deterministic, stdlib-only, and shaped for OWASP Agentic logging / EU AI Act Article 12 (record-keeping).

## Gate your CI

CI is, by construction, not the session that did the work, which makes it the natural
place for a recusal verdict. The same kernel runs as a command line with blocking exit
codes (`PASS` 0, `RETRY` 1, `FAIL` 2; any operational error exits 2, failing **closed**):

```bash
recusal verdict findings.json --json   # adjudicate any tool's findings; nonzero blocks the job
recusal audit verify audit.jsonl --expect-head "42:<hash>"   # a missing log is NOT an intact log
recusal doctor                         # "the gate silently isn't installed" fails CI, not prod
```

Or as a GitHub Action ([`action.yml`](action.yml), dogfooded by this repo's own CI,
including the negative case: a tampered audit log must make the gate refuse):

```yaml
- uses: actions/setup-python@v6
  with:
    python-version: "3.12"
- uses: philpaz/recusal@v0.4.1
  with:
    findings: reports/findings.json   # RETRY exits 1, FAIL exits 2 → the merge is blocked
    audit-log: reports/audit.jsonl
    doctor-dir: "."
```

The action ref selects the implementation: it installs the recusal bundled with the
selected ref, so pinning the action pins the code (an explicit `version:` input is the
one deliberate override).

Given nothing to adjudicate, the action exits 2 rather than pass vacuously: an evidence
set that proves nothing certifies nothing.

## Classify and route a failure

A refusal or failure is only useful if you know what to do next. The classifier says
*what kind* of failure it is and where it routes, deterministically, with no model:

```python
from recusal import classify_failure

c = classify_failure("Traceback ... TypeError: 'NoneType' object")
c.failure_class   # "code_bug"
c.route           # "fix-code"
```

Default taxonomy (extend or replace it): `transient → retry` · `policy_violation → refuse` ·
`prompt_injection → quarantine` · `code_bug → fix-code` · `data_shape → fix-data` ·
`data_missing → fetch-data` · `spec_ambiguity → ask-human`. Unmatched failures fall back to
`ask-human`; it never guesses. `classify_verdict(verdict)` routes a non-PASS verdict.

## Documentation

**New here?** The quick objections (*do I need this? doesn't Claude already do it? is it
ready to use?*) are answered in the [`docs/FAQ.md`](docs/FAQ.md). The plain-terms "so
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

Contributions are welcome. Recusal is deliberately small, and the bar is keeping it that
way (no model in the verdict path, no runtime dependencies, don't grow the kernel). Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) and the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
first. Security reports go through [`SECURITY.md`](SECURITY.md), privately.

## Contact

Built by [Philip Paz](https://www.linkedin.com/in/philippaz/). Messages are open,
especially from teams running agents in regulated environments, and doubly so if you
wired the gate in and found where it leaks
([tell me here](https://github.com/philpaz/recusal/discussions/1)).

## License

Apache-2.0 © [Philip Paz](https://www.linkedin.com/in/philippaz/)
