# The Evidence Contract

This is the spine of Recusal. Everything — checks, the verdict kernel, the Claude
adapter, the release gates — reduces to two objects and one function. Get this
contract right and the rest is small. (Pure standard library: `dataclasses` + `enum`,
zero dependencies.)

```
  data / a proposed action
          │
     [ checks ]            emit Findings
          │
     Finding, Finding, …
          │
   compute_verdict()       fold findings → one decision
          │
       Verdict             PASS / RETRY / FAIL
          │
   recusal.claude          allow or refuse a Claude tool call
```

## `Finding` — one observation about the work

```python
from recusal import Finding, Severity

Finding(
    check="row_count",       # what produced it
    severity=Severity.CRITICAL,  # how bad it is *if it failed*
    passed=False,            # did the check hold?
    message="0 rows in `members`",
    context={"actual": 0, "min_rows": 1},  # arbitrary structured detail
)
```

Ergonomic constructors:

```python
Finding.ok("row_count", severity="INFO", actual=5)          # held
Finding.fail("row_count", severity="CRITICAL", message="empty", actual=0)  # failed
```

`severity` accepts a `Severity` or the plain string (`"CRITICAL"`). Extra keyword
args land in `context`.

### Severity — what each tier does to the verdict

| Severity | If the finding **fails** |
|----------|--------------------------|
| `CRITICAL` | **FAIL** — the work is wrong. Terminal, no retry. |
| `ERROR` | **RETRY** — recoverable. Try once more with the failures as context. |
| `WARNING` | Proceed, but record it as a warning. |
| `INFO` | Never blocks. Recorded as a metric (held or not). |

A *passed* finding of any severity is fine — it held. Severity only matters on failure.

### Loose-dict input (coercion)

You don't have to build `Finding` objects. `compute_verdict` (and `recusal.claude`)
coerce loose dicts, which is the convenient form when wiring up an agent:

```python
{"severity": "CRITICAL", "status": "fail", "message": "...", "check": "subject_match", ...context}
```

`status` is one of `pass` / `fail` / `error` / `warn` (`fail`/`error`/`warn` → not passed).
`type` is accepted as an alias for `check`. Everything else becomes `context`.

## `Verdict` — the decision the findings add up to

```python
from recusal import compute_verdict

v = compute_verdict(findings)   # findings: Finding objects or loose dicts

v.decision           # Decision.PASS | RETRY | FAIL
v.passed             # bool — decision is PASS
v.refused            # bool — decision is FAIL
v.retryable          # bool — decision is RETRY
v.highest_severity   # Severity
v.failures           # tuple[Finding, ...] — what forced FAIL/RETRY
v.warnings           # tuple[Finding, ...]
v.metrics            # tuple[Finding, ...]
v.message            # one-line summary
v.reasons()          # the specific failure messages, for a human or an agent to act on
```

## `compute_verdict` — the rule

Deterministic. Same findings in, same verdict out. First match wins:

1. any **failed CRITICAL** → `FAIL`
2. else any **failed ERROR** → `RETRY`
3. else → `PASS`

Failed `WARNING`s are surfaced as `warnings` (they don't block). `INFO` findings —
and any failed `INFO`, which is a contradiction — are kept as `metrics`.

## Why a contract, not just dicts

For an *adjudication* library, the definition of "what is evidence" and "what is a
verdict" **is the product**. Typing it:

- makes the four surfaces (checks, verdict, gates, Claude adapter) one coherent
  pipeline instead of cousins passing loose dicts;
- makes a verdict auditable and replayable — a frozen, comparable object you can log;
- lets you extend cleanly (new checks just emit `Finding`s; new consumers just read
  `Verdict`s) without breaking the core.

The objects are frozen (immutable) and dependency-free on purpose: the thing
empowered to **refuse** should be the most boring, trustworthy part of your system.
