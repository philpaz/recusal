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
  so an agent cannot disable it.
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
| **Obfuscated / evasive shell** (spacing, quoting, escapes, `$IFS`, `base64 -d \| sh`, Windows `rd`/`del`/`Remove-Item`) | The hook normalizes and de-obfuscates the command and matches alternate forms; red-team tested in `tests/test_dogfood_redteam.py` (0/44 catchable bypasses). | A deny-list **cannot** catch a command whose name is built at runtime (hex/char-codes/`eval`/variable indirection). Use an **allowlist** ([`COOKBOOK.md`](docs/COOKBOOK.md) recipe 11, which *does* defeat these) for high-stakes tools. The limit is pinned as a test. |
| **Denial of service** (huge or pathological command, ReDoS) | A length cap refuses an over-long command rather than adjudicating it, and the matchers are bounded so a long adversarial input can't trigger catastrophic backtracking; both are timed in the red-team suite. | A length cap is blunt; tune it to your commands. |
| **Disabling the gate itself** | The hook refuses edits to its own kill-switch, `.claude/settings.json`, `settings.local.json`, and `.claude/hooks/**`, by file path and by Bash write commands. | Same deny-list caveat; an allowlist is stronger. |
| **Malformed / drifted hook envelope** (non-object JSON, missing `tool_name`, non-dict `tool_input`) | Fails **closed** to `deny` by default instead of normalizing and continuing. | `fail_closed=False` opts out. |
| **Ambiguous / buggy policy evidence** (a finding dict with no status degrading to PASS; a policy that raises) | The enforcement adapters adjudicate with `strict=True` and fail closed; a raising policy fails closed too. | The lenient default applies only to `compute_verdict` called directly, not at the adapters. |
| **Prompt injection via tool output** | Adjudicate the observation before acting; `classify_failure` routes injected content to `quarantine` deterministically (recipe). | You must actually gate on the screen result. |
| **Data exfiltration** | Egress-allowlist policy refuses outbound calls to non-allowlisted destinations (recipe). | Define the allowlist. |
| **Wrong-subject write** | Subject-guard policy: a write must target the session's active subject (the signature recipe; a real catch in `docs/PROVEN.md`). | Bind the active subject per turn. |
| **Runaway loop / cost** | Tiered action-budget policy (recipe). | The hook is per-call; persist the counter. |
| **Tampering with the audit record** | Hash-chained log detects in-place edits and reordering. | Truncation / full re-hash need an external head anchor (`verify(expected_head=...)`); unkeyed, single-writer. |
| **Supply chain (CI)** | Actions are tag-pinned (reference-architecture convention; documented in the workflows). | Pin to immutable commit SHAs for production. |

The through-line: the design gives you an **independent, deterministic seam that fails
closed** and an honest account of where a deny-list ends and an allowlist must begin. Harden
the policies and the deployment to your own threat model before relying on it in production.
