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
    return []   # no opinion → defer to Claude Code's normal flow

run_pretooluse_hook(policy)
```

A clean verdict **defers** (the gate adds refusals; it never strips Claude Code's own
prompts). A non-clean verdict **denies**, with the reasons. See `examples/claude_code_gate.py`.

## 2. Claude Agent SDK, manual loop

Use the **manual** agent loop (not the auto tool-runner) so you can adjudicate each
tool call before it executes. Gather whatever evidence proves the call is safe, get a
verdict, and on a non-PASS verdict hand Claude an `is_error` tool_result, it self-corrects.

```python
from recusal import Finding
from recusal.claude import gate_tool_use

def gather_evidence(tool):
    # Whatever proves THIS call is safe: preconditions, policy, a dry-run, an allowlist.
    if tool.name == "delete_file" and not tool.input["path"].startswith("/workspace/tmp/"):
        return [Finding.fail("path_allowlist", severity="CRITICAL",
                             message=f"{tool.input['path']} is outside /workspace/tmp/")]
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
awaiting a `user.tool_confirmation`. Make Recusal the decider:

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
assert release.release_ready, f"release refused: {[r.gate_id for r in release.blocking]}"
```

---

## 6. Audit every decision (tamper-evident)

Pair the gate with an append-only, hash-chained log so every verdict is on the record and
any later edit is detectable:

```python
from recusal import compute_verdict, AuditLog, verify

audit = AuditLog(path="audit.jsonl")   # omit path for in-memory
verdict = compute_verdict(findings)
audit.append(verdict, action={"tool": tool.name, "input": tool.input}, actor=session_id)

ok, problems = verify(audit.entries)   # (False, [...reasons]) if anything was altered
```

Each entry carries the SHA-256 of the entry before it, so deleting, editing, or reordering
a record breaks the chain. See `examples/audit_demo.py`.

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
