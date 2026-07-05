# How to use Recusal

Recusal is one idea applied three ways. The mental model is always the same:

> **gather evidence → compute a verdict → act on the verdict (allow / retry / refuse).**

```bash
pip install recusal
```

---

## 1. Claude Code, drop-in `PreToolUse` hook (the main path)

Register a hook in `.claude/settings.json` and Recusal refuses unsafe tool calls before
Claude Code runs them, even under `bypassPermissions` (a `PreToolUse` `deny` overrides it):

```json
{ "hooks": { "PreToolUse": [
  { "matcher": ".*", "hooks": [
    { "type": "command", "command": "for p in python3 python py; do \"$p\" -c '' 2>/dev/null && exec \"$p\" \"$CLAUDE_PROJECT_DIR/.claude/hooks/my_gate.py\"; done; echo 'gate: no python; failing closed' >&2; exit 2" } ]}
]}}
```

**Why the interpreter loop, and why `exit 2`:** Claude Code treats a hook whose command
fails to *launch* (e.g. `python3` not found — the default on Windows, where only `python`
or the `py` launcher exists) as a **non-blocking** error: the tool call proceeds and the
gate is silently disabled. The loop probes `python3` → `python` → `py` and runs the first
interpreter that actually works; if none do, `exit 2` is the one exit code Claude Code
treats as **blocking**, so a missing interpreter refuses the tool call instead of waving
it through. On Windows, Claude Code runs hook commands under Git Bash, so this POSIX
one-liner is portable. Verify your install end-to-end before trusting it:

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}' | python .claude/hooks/my_gate.py
# Windows (py launcher): ... | py .claude/hooks/my_gate.py
# expect: {"hookSpecificOutput": {..., "permissionDecision": "deny", ...}}
```

```python
# my_gate.py
from recusal import Finding
from recusal.claude_code import run_pretooluse_hook

def policy(tool_name, tool_input):
    if tool_name == "Bash" and "rm -rf" in tool_input.get("command", ""):
        return [Finding.fail("destructive_bash", severity="CRITICAL", message="refusing rm -rf")]
    return []   # no opinion → defer to Claude Code's normal flow

run_pretooluse_hook(policy)
```

A clean verdict **defers** (the gate adds refusals; it never strips Claude Code's own
prompts). A non-clean verdict **denies**, with the reasons. See `examples/claude_code_gate.py`.

### Two postures, two claims

The policy above is a **deny-list**: refuse known-bad calls, defer the rest. It stops the
accidental and common cases and its `deny` holds even in auto mode — but a literal matcher
can be obfuscated past, and `python script.py` runs code no string check ever reads. Never
read a deny-list as "cannot be subverted."

For high-stakes channels, flip to **allowlist mode** (default-deny), shipped as a factory:

```python
from recusal.claude_code import allowlist_policy, run_pretooluse_hook

run_pretooluse_hook(allowlist_policy(writable_root="./workspace"))
```

Nothing runs unless affirmatively named: unlisted tools, shell metacharacters, and **bare
interpreters** (`python script.py`) are refused — closing the write-a-script-then-run-it
bypass no deny-list can see (pinned in `tests/test_claude_code_allowlist.py`). This is the
posture that earns *"the agent could not subvert it,"* scoped honestly to the tool channel
routed through the hook; it says nothing about channels outside Claude Code's tool loop.
The trade-off is maintenance: you add binaries, roots, and per-tool predicates
(`safe_binaries=`, `writable_root=`, `allow={...}`) as the agent legitimately needs them,
and it fails *toward* refusal in the meantime. A vetted call still defers to Claude Code's
normal permission flow — allowlist mode, too, only ever adds refusals. See
`examples/allowlist_gate.py` for the deny-list-vs-allowlist comparison, and
`docs/COOKBOOK.md` recipe 11 for tuning.

## 2. Claude Agent SDK, manual loop

Use the **manual** agent loop (not the auto tool-runner) so you can adjudicate each
tool call before it executes. Gather whatever evidence proves the call is safe, get a
verdict, and on a non-PASS verdict hand Claude an `is_error` tool_result, it self-corrects.

```python
import os
from recusal import Finding
from recusal.claude import gate_tool_use

SAFE_ROOT = os.path.abspath("/workspace/tmp")

def gather_evidence(tool):
    # Whatever proves THIS call is safe: preconditions, policy, a dry-run, an allowlist.
    # commonpath, not startswith: "/workspace/tmp_evil" would slip past startswith.
    if tool.name == "delete_file":
        target = os.path.abspath(tool.input["path"])
        try:
            inside = os.path.commonpath([SAFE_ROOT, target]) == SAFE_ROOT
        except ValueError:  # different drives on Windows
            inside = False
        if not inside:
            return [Finding.fail("path_allowlist", severity="CRITICAL",
                                 message=f"{tool.input['path']} is outside {SAFE_ROOT}")]
    return [Finding.ok("path_allowlist", severity="CRITICAL")]

# inside the manual loop, for each tool_use block:
allow, refusal = gate_tool_use(tool.id, gather_evidence(tool), tool_name=tool.name)
if not allow:
    results.append(refusal)            # tool_result, is_error=True → Claude adapts
else:
    results.append({"type": "tool_result", "tool_use_id": tool.id,
                    "content": execute_tool(tool.name, tool.input)})
```

Runnable: `examples/claude_agent_live.py` (real API) and `examples/claude_refusal.py`
(offline, no key).

## 3. Gate a Managed Agent (`always_ask`)

For agents running `permission_policy: {"type": "always_ask"}`, the session idles
awaiting a `user.tool_confirmation`. Make Recusal the decider. The SDK surfaces below
(`permission_policy`, the `user.tool_confirmation` event, `sessions.events.send`) are
**illustrative**: verify them against your Agent SDK version. `tool_confirmation` only
builds the dict and carries no SDK dependency:

```python
from recusal.claude import tool_confirmation

event = tool_confirmation(tool_use_id, gather_evidence(tool))
# → {"type": "user.tool_confirmation", "result": "allow"|"deny", "deny_message": "..."}
client.beta.sessions.events.send(session_id=session.id, events=[event])
```

## 4. Adjudicate data / a work product directly

Use the built-in checks to turn data into evidence, then decide:

```python
from recusal import compute_verdict
from recusal.checks import row_count, null_rate, referential_integrity

verdict = compute_verdict([
    row_count(users, min_rows=1),
    null_rate(users, "email", max_rate=0.10),
    referential_integrity(orders, users, fk="user_id", pk="id"),
])

if verdict.refused:
    raise RuntimeError(verdict.reasons())   # don't ship it
```

## 5. Staged release gate (CI)

```python
from recusal import Finding, GateAdjudicator

gate = GateAdjudicator()
# Each gate folds its findings into a typed Verdict via the shared kernel.
g5 = gate.adjudicate(
    "G5", [Finding.fail("coverage_floor", severity="CRITICAL", message="coverage 61% < 75%")]
)
release = gate.release("run-001", [g5])
assert not release.release_ready          # G5 failed, so the release is refused
print([r.gate_id for r in release.blocking])   # ['G5']
```

---

## 6. Audit every decision (tamper-evident)

Pair the gate with an append-only, hash-chained log so every verdict is on the record and
in-place edits or reordering of existing entries are detectable:

```python
from recusal import compute_verdict, AuditLog, verify

audit = AuditLog(path="audit.jsonl")   # omit path for in-memory
verdict = compute_verdict(findings)
audit.append(verdict, action={"tool": tool.name, "input": tool.input}, actor=session_id)

ok, problems = verify(audit.entries)   # (False, [...reasons]) if an entry was edited or reordered
```

Each entry carries the SHA-256 of the entry before it, so editing or reordering an existing
record breaks the chain. Tail-truncation or a full re-hash by a write-access attacker needs
an external anchor (`verify(..., expected_head=(count, last_hash))`), the chain is
tamper-evident, not tamper-proof. See `examples/audit_demo.py`.

## 7. Classify and route a failure

When the gate refuses or an action fails, decide *what kind* of failure it is and where it
goes, deterministically, no model:

```python
from recusal import classify_failure, classify_verdict

c = classify_failure(error_text)          # or classify_verdict(verdict)
if c.route == "retry":
    ...                                    # transient, try again
elif c.route == "refuse":
    ...                                    # policy violation, don't retry as-is
elif c.route == "ask-human":
    ...                                    # ambiguous, escalate
```

Default classes (order is precedence, security-critical first): `policy_violation`,
`prompt_injection`, `transient`, `code_bug`, `data_shape`, `data_missing`, `spec_ambiguity`.
Unmatched failures fall back to
`ask-human`, it never guesses. See `examples/classify_demo.py`.

## Patterns & choices

- **Where evidence comes from is yours.** Recusal doesn't gather evidence, it adjudicates it. Preconditions, dry-runs, policy checks, an allowlist, the output of your existing validators (Great Expectations, pytest, a linter): anything that produces `Finding`s.
- **Pick severity by consequence, not by check.** The *same* check can be `CRITICAL` in one context and `WARNING` in another. Severity is a parameter, set it where you call the check.
- **`RETRY` is a real signal.** An `ERROR`-severity failure returns `RETRY`; in an agent loop that means "let the model try once more with `verdict.reasons()` as context," not "give up."
- **The gate refuses; it does not act.** It returns a verdict (and, for Claude, a tool_result). *You* decide what your loop/pipeline does with it. That separation is deliberate, see CONSTITUTION.md.
