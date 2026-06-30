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
