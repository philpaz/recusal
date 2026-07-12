# Security Policy

Recusal is a governance tool, so its own integrity matters.

## Reporting a vulnerability

Please report security issues **privately**, open a
[GitHub Security Advisory](https://github.com/philpaz/recusal/security/advisories) or email
the maintainer, rather than filing a public issue. We'll acknowledge and work a fix before
any public disclosure.

## Things worth knowing

- The verdict kernel (`compute_verdict`) has **zero runtime dependencies** and performs
  **no I/O**: it adjudicates evidence you pass in. (The opt-in `AuditLog` writes a log file
  when you give it a path; the kernel itself does not.) Most of the real attack surface is
  in *your* policy and evidence-gathering code, not in Recusal.
- The Claude Code hook **fails closed**: a policy that raises, *or* a malformed / non-object
  `PreToolUse` event, results in a `deny`, not a silent allow. Set `fail_closed=False` only if
  you understand the trade-off.
- `allow_on_pass=True` suppresses Claude Code's permission prompt for *any* call your policy
  passes, including the long tail it has no findings for. Use it only with an allowlist-style
  policy that affirmatively clears each call; otherwise leave it off (the default).
- A `PreToolUse` `deny` is honored even under `bypassPermissions`. Treat your policy as a
  security control and review changes to it, and to the hook settings, with that rigor.
- **Two postures, two claims.** A substring **deny-list** is a baseline: it stops the
  accidental and common cases, but a determined command can be obfuscated past a literal
  matcher and `python script.py` executes code no string check ever reads, never describe
  a deny-list as "cannot be subverted." **Allowlist mode** (default-deny) ships as
  `recusal.claude_code.allowlist_policy`: nothing runs unless affirmatively named, shell
  metacharacters and **bare interpreters** are refused (closing the
  write-a-script-then-run-it bypass; pinned in `tests/test_claude_code_allowlist.py`), and
  it defeats the runtime-constructed names a deny-list cannot. State that posture's claim
  precisely: **within a correctly registered routed tool channel, an unapproved capability
  is refused by default rather than inferred safe.** It says nothing about side channels
  outside that loop, a mis-registered hook, or a bug in your predicates. And read-only
  means *nonmutating*, not authorized for all data: the default-safe tools can still read
  credentials, keys, and regulated data, so add path- and subject-level read rules where
  confidentiality matters. Runnable comparison: [`examples/allowlist_gate.py`](examples/allowlist_gate.py).
  Protect the hook's own config in either posture so an agent cannot disable it. **`safe_binaries`
  must be arg-safe.** The default set is read/inspect tools only; a binary is safe to add
  *only if it cannot execute code, spawn a process, or write a file under any argument*.
  `pytest` (auto-imports `conftest.py`), `mypy` (loads plugins), `rg` (`--pre` spawns a
  command), and `git` (`-c core.pager=` / `alias.x='!sh'`) are deliberately **not** default-safe;
  gate a tool like that with an `allow=` predicate that vets the arguments.
- **PreToolUse governs tool calls, not every way content enters context.** A direct
  prompt-time `@file` reference can load a file without a `Read` tool event, so it never
  reaches this hook; govern those paths with Claude Code permission rules or separate
  controls.
- **The gate governs tool calls in *this repo*, not the whole machine.** The hook runs under
  the ambient Python interpreter, whose `site-packages` / user-site directories sit outside
  the repo and outrank it on `sys.path`. An actor who can write there, a `.pth` file (executed
  at interpreter startup), `sitecustomize`/`usercustomize`, or a trojaned `recusal`
  distribution, controls the gate, and the hook's control-path guard does **not** cover paths outside the
  repo. This is the documented scope boundary, not a bug: for stronger isolation run the hook
  from a repo-local virtualenv (its `site-packages` then lives inside the protected tree) or
  under an interpreter started with `-I`/`-S` to disable `.pth`/user-site/`PYTHON*` env at
  startup. In-repo import shadowing *is* closed: the hook appends the repo to `sys.path` (so a
  repo-root file cannot shadow a stdlib module) and refuses writes into the `recusal/` package.
- The hook's command/path checks are primarily **string-based**: a protected path is matched
  on the literal path in the command/tool input. For **tool-based writes** (`Write`/`Edit` and
  arbitrary MCP filesystem tools) there is now a **best-effort `realpath` layer**: the target
  is symlink-resolved and refused if it lands on a protected control path, so the classic
  "innocent name -> protected target" symlink (create `notes.txt` -> `.claude/settings.json`,
  then write `notes.txt`) is caught. Resolution covers a **bare filename with no path
  separator** on the MCP filesystem path too, not only separator-bearing paths, so the two
  write paths (`Write`/`Edit` and MCP) refuse the identical link. It is *best-effort* on purpose: a link that does not exist
  at hook time cannot be resolved, `Bash` command fragments are still matched as strings, and
  an allowlist of writable paths remains the real defense. A command whose *name* is built at
  runtime (hex/char-codes/`eval` of decoded data) is likewise uncatchable by any deny-list;
  both limits are pinned as tests.
- **The deny-list over-refuses in the safe direction.** Because it strips quotes/backticks
  before matching (so `r""m -rf` can't hide), it also matches a dangerous string that appears
  as *text* rather than a command, e.g. `git commit -m "remove the rm -rf call"` or
  `echo "do not run rm -rf /"` are refused, and a `cp`/`sed -i`/`python -c` that merely
  *reads* a protected path is refused too. This is deliberate: the gate fails toward refusal,
  never toward silently allowing. Rephrase the command, or use allowlist mode where reads are
  affirmatively vetted rather than pattern-matched.
- `recusal.audit` is **tamper-evident, not tamper-proof**: the hash chain detects in-place
  edits and reordering, but the digest is unkeyed and the head is unanchored, so an attacker
  with write access can truncate the tail or rewrite the whole chain. Commit the head
  `(count, last_hash)` somewhere they cannot reach and pass `verify(..., expected_head=...)`
  to catch that. It is single-writer.

## Attack surfaces, and how this is architected for them

Recusal is a **reference architecture** for governing autonomous agents, not a turnkey
production security product. The table shows the surfaces the design addresses, the
architectural response, and the honest limit you own. Where a row cites a recipe, see
[`docs/COOKBOOK.md`](docs/COOKBOOK.md); the dogfood hook
[`.claude/hooks/recusal_gate.py`](.claude/hooks/recusal_gate.py) is a thin shim over the
deny-list engine, which lives in the installable package as `recusal.deny_list`
(`deny_list_policy(...)`), so it is versioned and unit-tested rather than copy-pasted.

| Attack surface | How this is architected for it | Honest limit (your job) |
|---|---|---|
| **A model grading its own work** (the core risk) | No model in the decision path; `compute_verdict` is deterministic, replayable, and can refuse (`PASS`/`RETRY`/`FAIL`). An independent verifier the builder cannot talk out of it. | The verdict is only as good as the evidence you gather. |
| **Destructive tool call** (`rm -rf`, unscoped `DELETE`, secret-file write) | A `PreToolUse` hook adjudicates every call and denies before it runs; the deny holds even under `bypassPermissions`. | You define the policy; the shipped ones are examples. |
| **Obfuscated / evasive shell** (spacing, quoting, escapes, `$IFS`, `base64 -d \| sh`, pipe into `python`/`perl`/`ruby`/`node`, reverse shells `/dev/tcp` / `nc -e` / `socat EXEC:`, Windows `rd`/`del`/`Remove-Item`) | The hook normalizes and de-obfuscates the command, matches alternate forms uniformly (destructive, secret, *and* self-protect checks all run against the de-obfuscated command and three path readings), and refuses recursive `rm` even without `-f`; red-team tested in `tests/test_dogfood_redteam.py` and the wider `tests/test_subversion_hook.py`. | A deny-list **cannot** catch a command whose name is built at runtime (hex/char-codes/`eval`/variable indirection) or run inside an interpreter (`python -c`, `python script.py`). Use **allowlist mode** (`recusal.claude_code.allowlist_policy`; [`COOKBOOK.md`](docs/COOKBOOK.md) recipe 11), which *does* defeat these, for high-stakes tools. Both the deny-list limit and the allowlist's closure of it are pinned as tests. |
| **Denial of service** (huge or pathological command, ReDoS) | A length cap refuses an over-long command rather than adjudicating it, and the matchers are bounded so a long adversarial input can't trigger catastrophic backtracking; both are timed in the red-team suite. | A length cap is blunt; tune it to your commands. |
| **Disabling the gate itself** | The hook refuses edits *and deletions* of its own kill-switch, the settings and hook scripts (`.claude/settings*`, `.claude/hooks/**`), the in-repo `recusal/` **enforcement package** the hook imports and delegates every decision to (poisoning `recusal/*.py` would neutralize the gate on the next tool call), `.git/hooks/**` / `git config core.hooksPath`, *and* moving or removing the `.claude` / `.git` control directory itself, via `Write`/`Edit`, via Bash (`rm`/`mv`/`del`/redirects/inline-script writes), *and* via any other non-read tool: a generic guard refuses any non-read-only tool (an MCP filesystem tool) whose inputs reference a protected control path, matched against the de-obfuscated command and three path readings so quote-splitting, a backslash-escape, or a Windows separator cannot slip past. Pinned in `tests/test_dogfood.py`. | Same deny-list caveat; the allowlist path refuses by default (choose it for narrow high-stakes channels, see the posture note in the README). This covers the dogfood hook, which runs the gate from in-repo source; an adopter who `pip install`s `recusal` into site-packages keeps the enforcement code outside the governed tree entirely. A bare interpreter (`python file.py`) is allowed to *run* the hook; only its inline-code forms (`python -c`) are gated. Allowlist mode refuses bare interpreters outright. |
| **Ungated side-channel tool** (an MCP shell / filesystem tool used instead of `Bash`/`Write`) | Any tool carrying a command under a command-like key (`command`/`cmd`/`shell`/`script`, matched **case-insensitively and at any nesting depth**, with argv-array values joined) gets the exact same command analysis as `Bash`; the kill-switch guard above covers filesystem-style tools. | The command-key *names* and the read-only-tool allowlist are conventions; map *your* MCP tools to the policy explicitly. A read-only MCP tool that references a protected path is refused (safe-side false positive). |
| **Malformed / drifted hook envelope** (non-object JSON, missing `tool_name`, non-dict `tool_input`) | Fails **closed** to `deny` by default instead of normalizing and continuing. | `fail_closed=False` opts out. |
| **Rewritten MCP control plane** (`.mcp.json` server command swapped, `mcp-manifest.json` rewritten to redefine "approved") | The default deny-list protects `.mcp.json` and `mcp-manifest.json` as kill-switch-rank paths; `manifest_policy` fails closed on a deleted/corrupt manifest; `--minimal-env` keeps shell secrets away from a server being observed. | Observing a stdio catalog **executes the configured command** - the manifest pins declarations, not launch identity (launch-spec pinning is a named roadmap item) - so review the config as executable code before `pin`/`verify` runs it. |
| **Ambiguous / buggy policy evidence** (a finding dict with no status degrading to PASS; a stringified `"passed": "false"` reading as truthy; a policy that raises) | The enforcement adapters adjudicate with `strict=True` and fail closed; `Finding.coerce` reads a string `passed` or `status` against a pass *allowlist*, so `"false"`/`"no"`/`"0"` and any unrecognized token (`"maybe"`) read as a *failure* (not raw truthiness); a raising policy fails closed too. | The lenient no-status default applies only to `compute_verdict` called directly, not at the adapters. |
| **Prompt injection via tool output** | Adjudicate the observation before acting; `classify_failure` routes injected content to `quarantine` deterministically (recipe). | You must actually gate on the screen result. |
| **Data exfiltration** | Egress-allowlist policy refuses outbound calls to non-allowlisted destinations (recipe). | Define the allowlist. |
| **Wrong-subject write** | Subject-guard policy: a write must target the session's active subject (the signature recipe; a real catch in `docs/PROVEN.md`). | Bind the active subject per turn. |
| **Runaway loop / cost** | Tiered action-budget policy (recipe). | The hook is per-call; persist the counter. |
| **Tampering with the audit record** | Hash-chained log detects an edit or reorder of any entry with a surviving successor. | Tail truncation, tail-suffix rewrite (down to the last entry), and forged appends need an external head anchor (`verify(expected_head=...)`); unkeyed, single-writer. |
| **Supply chain (CI)** | Every third-party action in the CI and release workflows is pinned to an immutable commit SHA (a moving tag is a rug-pull surface), the release workflow reruns the full gate at the exact release commit before building, and the GitHub Action installs the recusal bundled with its own selected ref. | The SHAs are updated deliberately, by hand, when an action is upgraded. |

The through-line: the design gives you an **independent, deterministic seam that fails
closed** and an honest account of where a deny-list ends and an allowlist must begin. Harden
the policies and the deployment to your own threat model before relying on it in production.
