# Security Policy

Recusal is a governance tool, so its own integrity matters.

## Reporting a vulnerability

Please report security issues **privately** — open a
[GitHub Security Advisory](https://github.com/philpaz/recusal/security/advisories) or email
the maintainer — rather than filing a public issue. We'll acknowledge and work a fix before
any public disclosure.

## Things worth knowing

- The core has **zero runtime dependencies** and performs **no I/O** — it adjudicates
  evidence you pass in. Most of the real attack surface is in *your* policy and
  evidence-gathering code, not in Recusal.
- The Claude Code hook **fails closed**: a policy that raises an exception results in a
  `deny`, not a silent allow. Set `fail_closed=False` only if you understand the trade-off.
- A `PreToolUse` `deny` is honored even under `bypassPermissions`. Treat your policy as a
  security control and review changes to it with the same rigor.
- Recusal can refuse an action; it does not, by itself, prove an action *was* refused.
  Pair it with your own audit/logging when you need tamper-evident records.
