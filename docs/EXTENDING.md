# Extending Recusal

Everything extends through one contract: a check emits `Finding`s, a consumer reads a
`Verdict`. You never touch the core to add behavior. (See `docs/EVIDENCE.md` for the
contract itself.)

---

## 1. Write a custom check

A check is any function that returns one or more `Finding`s. That's the whole interface.

```python
from recusal import Finding


def sql_is_safe(statement: str, *, severity="CRITICAL") -> Finding:
    """Refuse destructive SQL without a WHERE clause."""
    s = statement.lower()
    if ("delete" in s or "update" in s) and "where" not in s:
        return Finding.fail(
            "sql_safety",
            severity=severity,
            message="destructive statement without a WHERE clause",
            statement=statement,
        )
    return Finding.ok("sql_safety", severity=severity)
```

Use it anywhere evidence is gathered:

```python
from recusal import compute_verdict
verdict = compute_verdict([sql_is_safe(stmt), ...other findings...])
```

**Guidelines:** one check = one concern; put structured detail in `context` (kwargs),
not the message; take `severity` as a parameter so callers choose the consequence;
keep it pure (no I/O surprises) so verdicts stay deterministic.

## 2. Bundle checks into a reusable policy

A "policy" is just a function that returns a list of findings, compose freely:

```python
def write_policy(tool_input, active_subject) -> list:
    return [
        subject_match(tool_input, active_subject),  # CRITICAL
        field_allowlist(tool_input),  # ERROR
        rate_limit(tool_input),  # WARNING
    ]


allow, refusal = gate_tool_use(tool.id, write_policy(tool.input, active), tool_name=tool.name)
```

Two ready-made policies ship as exactly this pattern, read them for a worked example or use
them as a baseline you extend: `recusal.deny_list.deny_list_policy()` (refuse known-bad, the
hardened engine behind the dogfood hook) and `recusal.claude_code.allowlist_policy()`
(default-deny). Both are `(tool_name, tool_input) -> findings` functions like the one above.

## 3. Write an adapter for another agent framework

`recusal.claude` is ~120 lines and the template for any framework: it turns a `Verdict`
into that framework's allow/deny shape. To support, say, a generic callback-based loop:

```python
from recusal import compute_verdict


def gate(findings):
    """Return (allow, reason). Plug into any 'before tool call' hook."""
    v = compute_verdict(findings)
    return v.passed, (None if v.passed else v.reasons())
```

For **LangGraph**: call `gate(...)` inside a node and route to a "refused" edge when
`allow` is False. For **CrewAI**: call it in a task/tool guard. For **Claude Code /
Agent SDK hooks**: call it in your `PreToolUse` hook and block on `allow is False`.
The pattern is identical, *the core never changes; only the wire shape does.*

## 4. Custom severities / policy tiers

Severity is the policy dial. The four tiers map to fixed verdict behavior
(`CRITICAL→FAIL`, `ERROR→RETRY`, `WARNING/INFO→PASS`), so you express *your* policy by
choosing which severity a given failure carries, per call site, per environment:

```python
sev = "CRITICAL" if env == "prod" else "WARNING"
findings.append(null_rate(rows, "email", max_rate=0.05, severity=sev))
```

If you need a different fold (e.g. "three WARNINGs should FAIL"), do it in a thin
wrapper around `compute_verdict` rather than changing the core, keep the kernel boring.

## 5. Add a staged gate

`GateAdjudicator` covers a `G0-G8` release pipeline, but the gates are pure labels,
pass your own ordered `(id, description)` staging to the constructor. Each gate is just
`compute_verdict` over that gate's findings, so there is nothing to subclass: produce
`Finding`s (from `recusal.checks` or your own checks), hand them to `adjudicate(gate_id,
findings)`, and `release(...)` rolls the typed verdicts into one decision.

```python
from recusal import GateAdjudicator

gate = GateAdjudicator(gates=(("LINT", "style clean"), ("SEC", "no secrets in diff")))
results = [gate.adjudicate("LINT", lint_findings), gate.adjudicate("SEC", secret_findings)]
release = gate.release("pr-482", results)
```

---

## 6. Custom failure taxonomy

The classifier ships a default taxonomy, but it's just data, supply your own classes:

```python
from recusal import classify_failure, FailureClass

TAXONOMY = (
    FailureClass("billing", "notify-finance", ("payment declined", "insufficient funds")),
    FailureClass("auth", "reauth", ("401", "token expired", "unauthorized")),
)
classify_failure(error_text, taxonomy=TAXONOMY)
```

Markers are case-insensitive substrings; taxonomy order is precedence. Keep
security-critical classes (policy refusals, injection) first so a refusal is never
misread as a generic code/data error, but keep those markers *narrow*, a broad one
(`"forbidden"`, bare `"429"`) mis-escalates ordinary errors. The default taxonomy
intentionally leaves some failures uncovered (out-of-memory, disk-full, auth) so they fall
to `ask-human` rather than be guessed; add markers for the ones your system should auto-route.

## What *not* to do

- Don't put a model call inside a check or the verdict path. Evidence-gathering can use a
  model (upstream); adjudication must stay deterministic, or you lose the whole point.
- Don't make the core do I/O. Checks read data you pass in; they don't fetch it.
- Don't grow the kernel. New capability = new check or new adapter, never a change to
  `compute_verdict`.
