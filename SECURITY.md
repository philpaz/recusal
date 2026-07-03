# Security Policy

Recusal is a governance tool, so its own integrity matters.

## Reporting a vulnerability

Please report security issues **privately**, open a
[GitHub Security Advisory](https://github.com/philpaz/recusal/security/advisories) or email
the maintainer, rather than filing a public issue. We'll acknowledge and work a fix before
any public disclosure.

## Things worth knowing

- The core has **zero runtime dependencies** and performs **no I/O**: it adjudicates
  evidence you pass in. Most of the real attack surface is in *your* policy and
  evidence-gathering code, not in Recusal.
- The Claude Code hook **fails closed**: a policy that raises, *or* a malformed / non-object
  `PreToolUse` event, results in a `deny`, not a silent allow. Set `fail_closed=False` only if
  you understand the trade-off.
- `allow_on_pass=True` suppresses Claude Code's permission prompt for *any* call your policy
  passes, including the long tail it has no findings for. Use it only with an allowlist-style
  policy that affirmatively clears each call; otherwise leave it off (the default).
- A `PreToolUse` `deny` is honored even under `bypassPermissions`. Treat your policy as a
  security control and review changes to it, and to the hook settings, with that rigor. A
  substring deny-list is a baseline, a determined command can be obfuscated past a literal
  matcher, so prefer an allowlist posture where it matters, and protect the hook's own config
  so an agent cannot disable it. A runnable allowlist reference policy ships as
  [`examples/allowlist_gate.py`](examples/allowlist_gate.py) (it defeats the runtime-constructed
  names a deny-list cannot).
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
- `recusal.audit` is **tamper-evident, not tamper-proof**: the hash chain detects in-place
  edits and reordering, but the digest is unkeyed and the head is unanchored, so an attacker
  with write access can truncate the tail or rewrite the whole chain. Commit the head
  `(count, last_hash)` somewhere they cannot reach and pass `verify(..., expected_head=...)`
  to catch that. It is single-writer.

## Attack surfaces, and how this is architected for them

Recusal is a **reference architecture** for governing autonomous agents, not a turnkey
production security product. The table shows the surfaces the design addresses, the
architectural response, and the honest limit you own. Where a row cites a recipe, see
[`docs/COOKBOOK.md`](docs/COOKBOOK.md); the dogfood hook is
[`.claude/hooks/recusal_gate.py`](.claude/hooks/recusal_gate.py).

| Attack surface | How this is architected for it | Honest limit (your job) |
|---|---|---|
| **A model grading its own work** (the core risk) | No model in the decision path; `compute_verdict` is deterministic, replayable, and can refuse (`PASS`/`RETRY`/`FAIL`). An independent verifier the builder cannot talk out of it. | The verdict is only as good as the evidence you gather. |
| **Destructive tool call** (`rm -rf`, unscoped `DELETE`, secret-file write) | A `PreToolUse` hook adjudicates every call and denies before it runs; the deny holds even under `bypassPermissions`. | You define the policy; the shipped ones are examples. |
| **Obfuscated / evasive shell** (spacing, quoting, escapes, `$IFS`, `base64 -d \| sh`, pipe into `python`/`perl`/`ruby`/`node`, reverse shells `/dev/tcp` / `nc -e` / `socat EXEC:`, Windows `rd`/`del`/`Remove-Item`) | The hook normalizes and de-obfuscates the command, matches alternate forms uniformly (destructive, secret, *and* self-protect checks all run against the de-obfuscated command and three path readings), and refuses recursive `rm` even without `-f`; red-team tested in `tests/test_dogfood_redteam.py` and the wider `tests/test_subversion_hook.py`. | A deny-list **cannot** catch a command whose name is built at runtime (hex/char-codes/`eval`/variable indirection) or run inside an interpreter (`python -c`). Use an **allowlist** ([`COOKBOOK.md`](docs/COOKBOOK.md) recipe 11, which *does* defeat these) for high-stakes tools. That limit is pinned as a test. |
| **Denial of service** (huge or pathological command, ReDoS) | A length cap refuses an over-long command rather than adjudicating it, and the matchers are bounded so a long adversarial input can't trigger catastrophic backtracking; both are timed in the red-team suite. | A length cap is blunt; tune it to your commands. |
| **Disabling the gate itself** | The hook refuses edits *and deletions* of its own kill-switch (`.claude/settings*`, `.claude/hooks/**`, and `.git/hooks/**` / `git config core.hooksPath`) via `Write`/`Edit`, via Bash (`rm`/`mv`/`del`/redirects/inline-script writes), *and* via any other non-read tool: a generic guard refuses any non-read-only tool (an MCP filesystem tool) whose inputs reference a protected control path, matched against the de-obfuscated command and three path readings so quote-splitting, a backslash-escape, or a Windows separator cannot slip past. | Same deny-list caveat; an allowlist is stronger. A bare interpreter (`python file.py`) is allowed to *run* the hook; only its inline-code forms (`python -c`) are gated. |
| **Ungated side-channel tool** (an MCP shell / filesystem tool used instead of `Bash`/`Write`) | Any tool carrying a command under a command-like key (`command`/`cmd`/`shell`/`script`, matched **case-insensitively and at any nesting depth**, with argv-array values joined) gets the exact same command analysis as `Bash`; the kill-switch guard above covers filesystem-style tools. | The command-key *names* and the read-only-tool allowlist are conventions; map *your* MCP tools to the policy explicitly. A read-only MCP tool that references a protected path is refused (safe-side false positive). |
| **Malformed / drifted hook envelope** (non-object JSON, missing `tool_name`, non-dict `tool_input`) | Fails **closed** to `deny` by default instead of normalizing and continuing. | `fail_closed=False` opts out. |
| **Ambiguous / buggy policy evidence** (a finding dict with no status degrading to PASS; a stringified `"passed": "false"` reading as truthy; a policy that raises) | The enforcement adapters adjudicate with `strict=True` and fail closed; `Finding.coerce` reads a stringified `"false"`/`"no"`/`"0"` as a *failure* (not raw truthiness); a raising policy fails closed too. | The lenient no-status default applies only to `compute_verdict` called directly, not at the adapters. |
| **Prompt injection via tool output** | Adjudicate the observation before acting; `classify_failure` routes injected content to `quarantine` deterministically (recipe). | You must actually gate on the screen result. |
| **Data exfiltration** | Egress-allowlist policy refuses outbound calls to non-allowlisted destinations (recipe). | Define the allowlist. |
| **Wrong-subject write** | Subject-guard policy: a write must target the session's active subject (the signature recipe; a real catch in `docs/PROVEN.md`). | Bind the active subject per turn. |
| **Runaway loop / cost** | Tiered action-budget policy (recipe). | The hook is per-call; persist the counter. |
| **Tampering with the audit record** | Hash-chained log detects in-place edits and reordering. | Truncation / full re-hash need an external head anchor (`verify(expected_head=...)`); unkeyed, single-writer. |
| **Supply chain (CI)** | Actions are tag-pinned (reference-architecture convention; documented in the workflows). | Pin to immutable commit SHAs for production. |

The through-line: the design gives you an **independent, deterministic seam that fails
closed** and an honest account of where a deny-list ends and an allowlist must begin. Harden
the policies and the deployment to your own threat model before relying on it in production.
