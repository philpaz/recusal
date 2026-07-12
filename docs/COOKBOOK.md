# Policy cookbook

Copy-paste policies for the agent actions people actually want to gate. Each recipe is a
**starting point you own and adapt**, Recusal adjudicates evidence; *you* decide what
proves an action is safe. The kernel never changes; only your policy does.

The mental model is the same everywhere: **a proposed action → findings → a verdict
(`PASS` / `RETRY` / `FAIL`).** Pick the severity to pick the outcome:

| Severity | Verdict | Use it when… |
|---|---|---|
| `CRITICAL` | **FAIL**, refuse, terminal | the action is destructive or wrong; never let it run |
| `ERROR` | **RETRY**, block, try once more | recoverable; the agent should adjust and re-propose |
| `WARNING` | **PASS**, allow, but record | worth noting, not worth stopping |
| `INFO` | **PASS**, allow, kept as a metric | calibration only |

## Wiring (do this once)

Every recipe below is a `policy(tool_name, tool_input)` that returns findings. Two ways to
run one, pick your surface, then drop any recipe's body in:

**Claude Code, `PreToolUse` hook** (`.claude/hooks/my_gate.py`, registered in
`.claude/settings.json`):

```python
from recusal.claude_code import run_pretooluse_hook

def policy(tool_name, tool_input):
    ...  # a recipe body, return findings, or [] to defer
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
> and are exercised by the suite, lift from there when you want the proven form.

> **Don't hand-roll the common protections.** The destructive-shell, secret-write, and
> kill-switch recipes below ship hardened and red-team-tested as one importable policy,
> `recusal.deny_list.deny_list_policy()` (de-obfuscation, pipe-into-any-interpreter,
> reverse shells, `cd`/variable indirection, symlink resolution). Use it as your baseline
> and add the domain-specific recipes on top; the recipes below are here to show the shape
> and to adapt when you need something the ready-made policy does not cover:
>
> ```python
> from recusal.deny_list import deny_list_policy
> from recusal.claude_code import run_pretooluse_hook
>
> run_pretooluse_hook(deny_list_policy())   # your gate's paths: protected_paths=(".mygate/",)
> ```

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
    # commonpath, not startswith: "/workspace_evil".startswith("/workspace") is a bypass.
    ap = os.path.abspath(path)
    try:
        inside = os.path.commonpath([SAFE_ROOT, ap]) == SAFE_ROOT
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        return [Finding.fail("path_confinement", severity="CRITICAL",
                             message=f"write outside the workspace root: {path}")]
    return []
```

## 4. Wrong-subject write guard (the signature recipe)

Mid-conversation about one subject, the agent stages a write against a *different* one. The
data is valid; it's just applied to the wrong record, an invariant the model can't
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

## 5. Egress allowlist, stop exfiltration

Outbound email/HTTP must go to an allowlisted destination. This is your guard against a
prompt-injected "send the data to attacker@evil.com".

```python
from urllib.parse import urlparse
from recusal import Finding

ALLOWED_DOMAINS = {"acme.com", "internal.example"}

def _destination_host(tool_input):
    # An email `to` (user@host) or a URL (http_post/webhook). Parse each properly:
    # a naive split on "/" turns "https://acme.com/x" into "https:", refusing every URL.
    to = str(tool_input.get("to") or "")
    if to:
        return to.rsplit("@", 1)[-1].split(">")[0].strip().lower()
    url = str(tool_input.get("url") or "")
    return (urlparse(url if "://" in url else "//" + url).hostname or "").lower()

def policy(tool_name, tool_input):
    if tool_name not in ("send_email", "http_post", "webhook"):
        return []
    host = _destination_host(tool_input)
    allowed = any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)  # host + subdomains
    if host and not allowed:
        return [Finding.fail("egress_allowlist", severity="CRITICAL",
                             message=f"destination '{host}' is not on the egress allowlist")]
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
                             message=f"{n} actions exceeds the hard cap {hard}, stop the loop")]
    if n > soft:
        return [Finding.fail("action_budget", severity="WARNING",
                             message=f"{n} actions over the soft budget {soft}")]
    return []
```

## 8. Quality gate before a merge or deploy

A recoverable gate, below the coverage floor or with failing tests is `ERROR` (RETRY), not
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

Policies are just functions returning findings, concatenate them and let `compute_verdict`
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

## 11. Allowlist mode (default-deny), the refuse-by-default path

Every recipe above is a *deny-list*: it refuses known-bad calls and defers the rest, the
right path for a broad channel where deferring the unknown keeps friction low. A deny-list
cannot catch a command whose name is built at runtime (`c=$'\x72\x6d'; $c -rf`,
`eval $(... | base64 -d)`), no string match ever sees the `rm`, and it cannot see code run
*inside* an interpreter, `python script.py` executes a program the gate never reads. For a
narrow, high-stakes channel the other path fits: **deny by default, allow only what you
affirmatively vet** (neither path is "better" in the abstract; the channel decides). For
`Bash`, reject shell metacharacters (so a command can't chain, substitute, or expand into
something else) and require a vetted first binary, with interpreters deliberately unvetted.
That defeats the runtime-construction and bare-interpreter bypasses a deny-list cannot.

This posture ships as library API, `recusal.claude_code.allowlist_policy` (tune it with
`safe_binaries=`, `writable_root=`, `read_only_tools=`, and per-tool `allow={...}`
predicates; the bare-interpreter refusal is pinned in `tests/test_claude_code_allowlist.py`):

```python
from recusal.claude_code import allowlist_policy, run_pretooluse_hook

run_pretooluse_hook(allowlist_policy(writable_root="./workspace"))
```

The hand-rolled equivalent, if you want to own every line of the decision:

```python
import os, shlex
from recusal import Finding

WORKSPACE = os.path.abspath("./workspace")
# Binaries safe under EVERY argument: read/inspect tools with no flag that executes code,
# spawns a process, or writes a file. Tools that can mutate (git, sed, find, rm) or run code
# through an argument (pytest imports conftest.py; mypy loads plugins; rg has `--pre`) are
# deliberately NOT here -- allowlisting their name would reopen the write-a-script-then-run
# bypass this posture exists to close. Gate those with explicit per-argument rules instead.
SAFE_BINARIES = {"ls", "cat", "head", "tail", "grep", "wc", "pwd", "stat", "diff"}
SHELL_META = set(";|&`$<>(){}\n\\")   # chaining / substitution / redirection / expansion

def _under(root, path):
    try:
        return os.path.commonpath([root, os.path.abspath(path)]) == root
    except ValueError:                 # different drives on Windows
        return False

def _bash_ok(cmd):
    if set(cmd) & SHELL_META:          # can't reason about an expanded command -> refuse
        return False
    try:
        argv = shlex.split(cmd)
    except ValueError:                 # unbalanced quotes -> refuse
        return False
    return bool(argv) and argv[0] in SAFE_BINARIES

ALLOW = {
    "Read":  lambda i: True,
    "Grep":  lambda i: True,
    "Glob":  lambda i: True,
    "Bash":  lambda i: _bash_ok(i.get("command", "")),
    "Write": lambda i: _under(WORKSPACE, i.get("file_path", "")),
    "Edit":  lambda i: _under(WORKSPACE, i.get("file_path", "")),
}

def policy(tool_name, tool_input):
    check = ALLOW.get(tool_name)
    if check and check(tool_input):
        return []                      # affirmatively allowed -> defer
    return [Finding.fail("not_allowlisted", severity="CRITICAL",
                         message=f"{tool_name} call is not on the allowlist")]
```

The trade-off: an allowlist is stricter and needs maintenance (you add capabilities as the
agent legitimately needs them), but it fails *toward* refusal instead of away from it. For
high-stakes agents prefer this to a deny-list, or compose both (recipe 10).

## 12. Govern MCP tool calls

MCP server tools arrive at the hook as ordinary tools named `mcp__<server>__<tool>`
(`mcp__github__create_issue`), so every recipe above already applies to them. This one adds
the MCP-shaped rules: pin the servers you expect (a tool from a server you never installed
should refuse, not run), refuse destructive verbs, scope the rest.

```python
import os
from recusal import Finding

APPROVED_SERVERS = {"github", "salesforce", "filesystem"}
APPROVED_REPOS = {"philpaz/recusal"}
DESTRUCTIVE_VERBS = {"delete", "drop", "truncate", "remove", "destroy"}  # tune to your servers

def _mcp(tool_name):
    parts = tool_name.split("__", 2)   # "mcp__github__create_issue" -> (github, create_issue)
    return (parts[1], parts[2]) if len(parts) == 3 and parts[0] == "mcp" else None

def policy(tool_name, tool_input):
    named = _mcp(tool_name)
    if named is None:
        return []                      # not an MCP call -> your other recipes' job
    server, action = named
    if server not in APPROVED_SERVERS:
        return [Finding.fail("mcp_unapproved_server", severity="CRITICAL",
                             message=f"MCP server '{server}' is not on the approved list")]
    if action.split("_", 1)[0] in DESTRUCTIVE_VERBS:
        return [Finding.fail("mcp_destructive_action", severity="CRITICAL",
                             message=f"destructive MCP action `{tool_name}` is not approved")]
    if tool_name == "mcp__github__merge_pull_request":
        repo = tool_input.get("repo")
        if repo not in APPROVED_REPOS:
            return [Finding.fail("mcp_repository_scope", severity="CRITICAL",
                                 message=f"repository {repo!r} is outside the approved scope")]
    return []
```

Runnable version with path confinement and allowlist mode:
[`examples/mcp_governance.py`](../examples/mcp_governance.py); pinned:
[`tests/test_mcp_governance.py`](../tests/test_mcp_governance.py). In allowlist mode
(recipe 11) MCP tools are refused unless affirmatively named with an `allow=` predicate.

> **Call-time only.** This sees the proposed name and arguments, not what the server
> declared at discovery (identity, tool descriptions, schemas, `tools/list` changes), and a
> poisoned tool *description* steers the model before any call is proposed. Screen what an
> MCP tool *returns* with recipe 6; govern the tool catalog itself with `recusal mcp pin` /
> `recusal mcp verify` and enforce the pin at call time by wrapping this recipe in
> `recusal.mcp.manifest_policy("mcp-manifest.json", policy=policy)` — unpinned MCP calls
> refuse before your rules even run. Transport/authorization threats (confused deputy,
> token passthrough, session hijacking) belong to the MCP spec's own Security Best
> Practices layer, complementary to this gate.

## 13. Pin your MCP servers and enforce the pin

Recipe 12 refuses *unexpected* servers and verbs at call time. This is the discovery-boundary
companion: **pin the exact tool catalog your approved servers declare, then refuse any call to
a tool that was never pinned** — catching the rug pull (a description quietly rewritten after
approval) and the tool-that-appeared, deterministically.

**1. Pin once** — the reviewed, human step. recusal fetches local stdio servers for you:

```bash
recusal mcp pin --claude-config .mcp.json --approve-server-launch --out mcp-manifest.json
# or a single server:
recusal mcp pin --stdio github "npx -y @modelcontextprotocol/server-github@1.2.3" \
    --approve-server-launch --out mcp-manifest.json
```

> **`--claude-config` and `--stdio` EXECUTE the declared server commands** to ask them
> for `tools/list`; there is no other way to ask a process for its catalog. That is why
> the first pin requires `--approve-server-launch`: review the `command`/`args`/`env`
> lines the same way you review the declarations, then record the approval. The
> manifest pins each **source specification** (unexpanded command template, args, cwd,
> and env value *templates*; for remote servers the `url` template and header names)
> alongside the catalog, and `verify` compares it **before** launching, so a rewritten
> command, a same-key env value swap, or an added server of any transport is refused
> without anything executing. A mixed config pins only with `--from` supplying the
> remote catalogs. Reference secrets as `${VAR}` (a literal env value becomes manifest
> content, and the pin warns). Pin package versions in the args
> (`server-github@1.2.3`, not `server-github`): PATH and the registry resolve what
> they are asked for. Servers run with a minimal environment by default
> (`--inherit-env` opts out); minimal environment is not a sandbox.

The manifest stores tool declarations as **hashes only** (a poisoned description is
never embedded); source templates are stored readable so drift can be explained - keep
secrets out of them (the pin warns). It is
byte-deterministic. `pin` refuses to write when its screen flags injection phrasing in a
declaration, until you pass `--force` to record that a human reviewed it. Commit
`mcp-manifest.json` — it is approved truth.

**2. Enforce at call time** — wire `manifest_policy` into a PreToolUse hook
(`.claude/hooks/mcp_gate.py`), the same way as any gate in the Wiring section above:

```python
from recusal.claude_code import run_pretooluse_hook
from recusal.deny_list import deny_list_policy
from recusal.mcp import manifest_policy

# The pin is only as strong as the files it lives in: .mcp.json decides which server
# processes launch, and mcp-manifest.json is what "approved" MEANS at call time - so the
# inner deny-list (whose defaults protect both) refuses the agent rewriting either.
# manifest_policy composes on top: any mcp__server__tool call not in the pinned manifest
# is refused ("no pin, no MCP"), failing CLOSED if the manifest is missing or unreadable.
# Non-MCP tools fall through to the inner policy.
run_pretooluse_hook(manifest_policy("mcp-manifest.json", policy=deny_list_policy()))
```

**3. Verify in CI / at session start** — catch drift before it reaches an agent:

```bash
recusal mcp verify --claude-config .mcp.json --manifest mcp-manifest.json
# exit 0 = the observed catalog matches the pin; exit 2 = drift, an unpinned tool/server,
# a pinned server gone unverifiable, or a failed observation. Wire it as a blocking CI step.
```

A **remote/HTTP** server is pinned the same way, except you supply its `tools/list` yourself
(recipe 14). To also apply argument-level rules on top of the pin, pass an inner policy —
recipe 15.

## 14. Pin a remote (HTTP) MCP server

Recipe 13's `--stdio` / `--claude-config` fetch a **local stdio** server for you — the
zero-dependency client recusal ships speaks stdio only. A **remote/HTTP** server
(streamable-HTTP or SSE) is governed exactly the same way, but you obtain its `tools/list`
with a real MCP client and hand recusal the dump via `--from`. That is deliberate: recusal
owns the *adjudication* (deterministic, no deps), not the *transport* (an HTTP+OAuth client is
heavy, and its own SSRF/redirect surface is precisely what the MCP spec's Security Best
Practices warn about — best left to the maintained SDKs).

Dump `tools/list` into recusal's `{server_name: [declaration, ...]}` shape with the
official [`mcp`](https://pypi.org/project/mcp/) SDK (or `fastmcp`, or any client):

```python
# pip install mcp    (NOT a recusal dependency — this is your collection step)
import anyio, json
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def dump(url, server_name, out_path, headers=None):
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
    catalog = {server_name: [t.model_dump(exclude_none=True, mode="json") for t in tools]}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2, sort_keys=True)

anyio.run(dump, "https://example.com/mcp", "example", "example.tools.json",
          {"Authorization": "Bearer <token>"})
```

Then pin and, in CI or at session start, verify:

```bash
recusal mcp pin    --from example.tools.json --out mcp-manifest.json   # review once
recusal mcp verify --from example.tools.json --manifest mcp-manifest.json   # or refuse
```

> **Three cautions that make the dump trustworthy** (recusal fingerprints exactly the bytes
> you feed it, so these are on you, not the tool):
> - **Same dumper for pin and verify.** The fingerprint is over the *serialized* declaration;
>   if pin uses one client's `model_dump` and verify uses another's, harmless serialization
>   differences read as drift. Pick one dumper and reuse it.
> - **Same endpoint the agent session uses.** Point the dumper at the exact URL (and auth)
>   your agent connects to. A server that serves one catalog to your dumper and another to
>   the live session defeats the check — see the "honest boundary" note in the README.
> - **Close in time.** `verify` proves the catalog *when it runs*. Run it at session start
>   (and in CI), not once a week, so the window a rug-pull can hide in stays small.

Local stdio servers skip the dump step (recipe 13 fetches them for you). Real end-to-end
runs against live HTTP servers (a FastMCP gateway and Salesforce Hosted MCP) are pinned as
[`tests/test_mcp_live.py`](../tests/test_mcp_live.py).

## 15. The three-boundary MCP governance pattern

The three MCP tool-call boundaries compose into one setup. `manifest_policy` takes an inner `policy=`,
so the **pin** (discovery) and your **call-time rules** (invocation, recipe 12) run in a
single hook: an unpinned tool refuses first, then your argument-level rules run on what
survives. The **response** boundary (recipe 6) is a separate screen on what a tool *returned*,
before you feed it back as context.

```python
from recusal.claude_code import run_pretooluse_hook
from recusal.mcp import manifest_policy
from recusal import Finding

def call_time_rules(tool_name, tool_input):        # recipe 12's argument-level rules
    if tool_name == "mcp__github__merge_pull_request" and tool_input.get("repo") not in {"me/repo"}:
        return [Finding.fail("mcp_repository_scope", severity="CRITICAL",
                             message=f"repo {tool_input.get('repo')!r} is out of scope")]
    return []

# discovery + invocation in one hook: unpinned tools refuse first, then your rules run.
policy = manifest_policy("mcp-manifest.json", policy=call_time_rules)
run_pretooluse_hook(policy)
```

| Boundary | What it checks | Recipe |
|---|---|---|
| Discovery (`tools/list`) | is this tool in the pinned catalog? | 13 (local) / 14 (HTTP) |
| Invocation (the call) | are the name + arguments allowed? | 12 |
| Response (the result) | is the returned content safe to act on? | 6 |

Runnable end-to-end (offline): [`examples/mcp_full_stack.py`](../examples/mcp_full_stack.py),
pinned by [`tests/test_mcp_cookbook.py`](../tests/test_mcp_cookbook.py).

---

**These are starting points, not turnkey security.** Read each one, tune the lists and
thresholds to your system, and add the evidence *your* actions actually need. The discipline
that makes them trustworthy is the same one the whole library is built on: the verdict is
deterministic, replayable, and has no model in the decision path, so the same proposed
action gets the same answer, including the **no**.
