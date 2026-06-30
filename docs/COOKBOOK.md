# Policy cookbook

Copy-paste policies for the agent actions people actually want to gate. Each recipe is a
**starting point you own and adapt** — Recusal adjudicates evidence; *you* decide what
proves an action is safe. The kernel never changes; only your policy does.

The mental model is the same everywhere: **a proposed action → findings → a verdict
(`PASS` / `RETRY` / `FAIL`).** Pick the severity to pick the outcome:

| Severity | Verdict | Use it when… |
|---|---|---|
| `CRITICAL` | **FAIL** — refuse, terminal | the action is destructive or wrong; never let it run |
| `ERROR` | **RETRY** — block, try once more | recoverable; the agent should adjust and re-propose |
| `WARNING` | **PASS** — allow, but record | worth noting, not worth stopping |
| `INFO` | **PASS** — allow, kept as a metric | calibration only |

## Wiring (do this once)

Every recipe below is a `policy(tool_name, tool_input)` that returns findings. Two ways to
run one — pick your surface, then drop any recipe's body in:

**Claude Code — `PreToolUse` hook** (`.claude/hooks/my_gate.py`, registered in
`.claude/settings.json`):

```python
from recusal.claude_code import run_pretooluse_hook

def policy(tool_name, tool_input):
    ...  # a recipe body — return findings, or [] to defer
    return []

run_pretooluse_hook(policy)   # a clean verdict defers; a non-clean one denies
```

**Any agent loop** (Claude Agent SDK, LangGraph, OpenAI Agents, homegrown):

```python
from recusal import compute_verdict

verdict = compute_verdict(policy(tool_name, tool_input))
if verdict.refused or verdict.retryable:
    ...  # don't execute; hand the agent verdict.reasons() so it self-corrects
```

> Tested versions of the core recipes (wrong-subject, destructive path, unscoped SQL,
> egress, coverage, budget) live in [`../examples/scenarios.py`](../examples/scenarios.py)
> and are exercised by the suite — lift from there when you want the proven form.

---

## 1. Block destructive shell commands

Refuse `rm -rf`, disk wipes, recursive `chmod`, and pipe-to-shell installs before they run.

```python
from recusal import Finding

DESTRUCTIVE = ("rm -rf", "rm -fr", "mkfs", "dd if=", ":(){", "chmod -r 777", "> /dev/sd")

def policy(tool_name, tool_input):
    if tool_name != "Bash":
        return []
    cmd = tool_input.get("command", "").lower()
    hits = [m for m in DESTRUCTIVE if m in cmd]
    if "curl" in cmd and ("| sh" in cmd or "| bash" in cmd):
        hits.append("curl | shell")
    if hits:
        return [Finding.fail("destructive_shell", severity="CRITICAL",
                             message=f"refusing destructive command: {', '.join(hits)}")]
    return []
```

## 2. Require a `WHERE` on destructive SQL

The classic data-loss bug: a `DELETE`/`UPDATE` with no scope. (Applies to a custom `run_sql`
tool, or to `Bash` invoking a DB client.)

```python
from recusal import Finding

def policy(tool_name, tool_input):
    if tool_name not in ("run_sql", "query"):
        return []
    sql = str(tool_input.get("sql", "")).lower()
    destructive = any(k in sql for k in ("delete", "update", "drop", "truncate"))
    if destructive and "where" not in sql and "truncate" not in sql:
        return [Finding.fail("sql_scope", severity="CRITICAL",
                             message="destructive SQL without a WHERE clause")]
    if "drop table" in sql or "truncate" in sql:
        return [Finding.fail("sql_scope", severity="CRITICAL",
                             message="schema-destructive SQL (DROP/TRUNCATE)")]
    return []
```

## 3. Protect secret files and confine writes to the workspace

Refuse writes to credentials, keys, and anything outside an allowed root.

```python
import os
from recusal import Finding

SAFE_ROOT = os.path.abspath("./workspace")
PROTECTED = (".env", ".pem", ".key", ".p12", "id_rsa", "credentials", "secrets")

def policy(tool_name, tool_input):
    if tool_name not in ("Write", "Edit", "MultiEdit"):
        return []
    path = tool_input.get("file_path", "")
    low = path.lower()
    if any(p in low for p in PROTECTED):
        return [Finding.fail("protected_file", severity="CRITICAL",
                             message=f"refusing write to a secret/credential file: {path}")]
    if not os.path.abspath(path).startswith(SAFE_ROOT):
        return [Finding.fail("path_confinement", severity="CRITICAL",
                             message=f"write outside the workspace root: {path}")]
    return []
```

## 4. Wrong-subject write guard (the signature recipe)

Mid-conversation about one subject, the agent stages a write against a *different* one. The
data is valid; it's just applied to the wrong record — an invariant the model can't
self-enforce, because it doesn't know which subject your system says is active.

```python
from recusal import Finding

def make_subject_guard(active_id):
    def policy(tool_name, tool_input):
        if tool_name != "update_record":
            return []
        target = tool_input.get("id")
        if target != active_id:
            return [Finding.fail("subject_match", severity="CRITICAL",
                                 message=f"write targets {target}, not the active subject {active_id}")]
        return []
    return policy

# policy = make_subject_guard(active_id="C1001")   # bind the active subject per session/turn
```

## 5. Egress allowlist — stop exfiltration

Outbound email/HTTP must go to an allowlisted destination. This is your guard against a
prompt-injected "send the data to attacker@evil.com".

```python
from recusal import Finding

ALLOWED_DOMAINS = {"acme.com", "internal.example"}

def policy(tool_name, tool_input):
    if tool_name not in ("send_email", "http_post", "webhook"):
        return []
    target = str(tool_input.get("to") or tool_input.get("url") or "")
    domain = target.split("@")[-1].split("/")[0].lower()
    if domain and domain not in ALLOWED_DOMAINS:
        return [Finding.fail("egress_allowlist", severity="CRITICAL",
                             message=f"destination '{domain}' is not on the egress allowlist")]
    return []
```

## 6. Quarantine prompt-injection in tool output

Untrusted content a tool returns (a web page, an MCP server, a file) can carry instructions
that hijack the next action. Adjudicate the *observation* before the agent acts on it.

```python
from recusal import Finding

INJECTION_MARKERS = (
    "ignore previous instructions", "disregard the above", "ignore the system prompt",
    "new instructions:", "send the api key", "exfiltrate",
)

def screen_tool_output(text):
    low = (text or "").lower()
    hits = [m for m in INJECTION_MARKERS if m in low]
    if hits:
        return [Finding.fail("prompt_injection", severity="CRITICAL",
                             message=f"tool output carries injected instructions: {hits[0]!r}")]
    return []

# verdict = compute_verdict(screen_tool_output(observation))
# if verdict.refused: quarantine the observation; do NOT feed it back as trusted context.
```

> Pair this with `classify_failure(...)`, which routes a `prompt_injection` failure to
> `quarantine` deterministically.

## 7. Cap runaway action volume

Tiered budget: warn over a soft cap, **stop** (RETRY) over a hard one. A `PreToolUse` hook
is a fresh process per call, so persist the count (here, a small file).

```python
import json, os
from recusal import Finding

COUNT_FILE = "/tmp/recusal_action_count.json"

def _bump():
    n = 0
    if os.path.exists(COUNT_FILE):
        n = json.load(open(COUNT_FILE)).get("n", 0)
    n += 1
    json.dump({"n": n}, open(COUNT_FILE, "w"))
    return n

def policy(tool_name, tool_input, soft=25, hard=100):
    n = _bump()
    if n > hard:
        return [Finding.fail("action_budget", severity="ERROR",
                             message=f"{n} actions exceeds the hard cap {hard} — stop the loop")]
    if n > soft:
        return [Finding.fail("action_budget", severity="WARNING",
                             message=f"{n} actions over the soft budget {soft}")]
    return []
```

## 8. Quality gate before a merge or deploy

A recoverable gate — below the coverage floor or with failing tests is `ERROR` (RETRY), not
a terminal refusal. Drop this in a CI step or a `merge_pr` / `deploy` tool guard.

```python
from recusal import Finding

def quality_gate(coverage, failed, min_coverage=75):
    findings = []
    if coverage < min_coverage:
        findings.append(Finding.fail("coverage_floor", severity="ERROR",
                                     message=f"coverage {coverage}% < required {min_coverage}%"))
    if failed > 0:
        findings.append(Finding.fail("tests", severity="ERROR",
                                     message=f"{failed} test(s) failing"))
    return findings

# verdict = compute_verdict(quality_gate(coverage=61, failed=2))   # RETRY
```

## 9. Approved-tool allowlist

Let the agent call only tools you've vetted; refuse anything else (guards a hijacked agent
reaching for an unexpected capability).

```python
from recusal import Finding

APPROVED = {"Read", "Grep", "Glob", "Bash", "update_record"}

def policy(tool_name, tool_input):
    if tool_name and tool_name not in APPROVED:
        return [Finding.fail("tool_allowlist", severity="CRITICAL",
                             message=f"tool '{tool_name}' is not on the approved list")]
    return []
```

## 10. Compose several policies into one gate

Policies are just functions returning findings — concatenate them and let `compute_verdict`
fold the lot. The worst severity across *all* of them decides the verdict.

```python
from recusal.claude_code import run_pretooluse_hook

POLICIES = [block_destructive_shell, protect_files, subject_guard, egress_allowlist]

def policy(tool_name, tool_input):
    findings = []
    for p in POLICIES:
        findings.extend(p(tool_name, tool_input))
    return findings

run_pretooluse_hook(policy)   # one hook, every rule; a clean verdict still defers
```

---

**These are starting points, not turnkey security.** Read each one, tune the lists and
thresholds to your system, and add the evidence *your* actions actually need. The discipline
that makes them trustworthy is the same one the whole library is built on: the verdict is
deterministic, replayable, and has no model in the decision path — so the same proposed
action gets the same answer, including the **no**.
