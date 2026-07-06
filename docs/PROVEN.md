# Proven, Recusal governs its own repository

Recusal isn't only shown on toy inputs. It is built to govern the development of *this*
repository. A real Claude Code `PreToolUse` hook, [`.claude/hooks/recusal_gate.py`](../.claude/hooks/recusal_gate.py),
ships with the repo and, once registered (a deliberate step, see below), adjudicates every
tool call first, so the dangerous ones the policy recognizes, destructive shell commands,
writes to secret/protected files, and edits that would disable the gate itself, are
**refused before they run**, even under `bypassPermissions`. Every refusal below is
reproducible by anyone, registered or not, by piping the exact payload into the hook.

The governance library is the thing governing its own development. That is the dog food.

## What's installed

- [`.claude/hooks/recusal_gate.py`](../.claude/hooks/recusal_gate.py): the hook and its
  repo-protection policy (refuse `rm -rf` and its flag variants, force-push, `reset --hard`,
  `curl … | sh`, writes to `.env` / `*.pem` / `*.key` and shell redirects to them, and edits
  to the gate's own settings/hook so an agent can't disable it). It is a substring/regex
  deny-list, a *baseline*, not a guarantee: an obfuscated command can evade any literal
  matcher, and for a narrow high-stakes channel the refuse-by-default allowlist path fits
  better (neither is "better" in the abstract; the channel decides). What this proves is the
  seam, not an exhaustive list.
- Registration, see [`.claude/settings.json.example`](../.claude/settings.json.example).
  Installing a permission-changing hook is a **deliberate, reviewed step**: Claude Code's
  own auto mode will (correctly) ask you to confirm it. That confirmation *is* the
  separation-of-powers point, a control that changes what an agent may do should not be
  installed silently.

## It refuses, for real

Each block below is the **verbatim** output of piping the exact JSON Claude Code sends on
`PreToolUse` into the installed hook (the path in a reason simply echoes the path your
payload carried). Reproduce any line yourself:

```bash
echo '<payload>' | python .claude/hooks/recusal_gate.py   # or python3 / py, whichever your OS has
```

**`rm -rf`** (the recursive-`rm` guard fires on `-r` in any flag order, force or not)
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing destructive command (rm -r)"}}
```

**`git push --force`**
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing destructive command (git push --force)"}}
```

**`curl … | sh`** (and `… | python`/`perl`/`ruby`/`node`, same attack class)
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing to pipe output straight into a shell/interpreter"}}
```

**Write to `.env`**
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Write` [FAIL]: refusing write to a protected/secret file: .env"}}
```

**Edit the gate's own config (self-protection)**
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Edit` [FAIL]: refusing a `Edit` call that targets a protected control path (gate config/hook or git hooks): .claude/settings.json"}}
```

**Delete the hook itself (self-protection covers removal, not just edits)**, `rm .claude/hooks/recusal_gate.py`
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing a command that edits or removes a protected control path (gate config/hook or git hooks)"}}
```

A clean call, a `Read`, or `Bash` running `pytest -q`, produces **no output**: the hook
*defers* to Claude Code's normal permission flow. The gate only ever adds refusals; it never
strips your existing prompts.

## Locked by CI

[`tests/test_dogfood.py`](../tests/test_dogfood.py) loads this exact policy and asserts the
refusals (and the defers), so the proof cannot silently rot, every CI run re-proves it.

## A reference architecture: catching a real bug

Recusal also ships as a **reference integration** showing how to wire it into an agent: a
FastAPI agent (a relationship agent with confirm-gated CRM writes) where a subject-match
guard closes a named audit finding, **wrong-subject write**. The agent could stage a write
against a *different* member than the conversation was about, and the human confirm card
would wave it through.

The guard sits at the confirm endpoint and adjudicates the write with Recusal's
`compute_verdict`: the write must target the member who was active when it was proposed.
Output from the wired guard:

```text
wrong subject  -> write targets C-9988 but the active member this turn is C1001   (refused)
right subject  -> None                                                            (allowed)
no active mbr  -> None                                                            (fail-open)
```

On a mismatch the confirm endpoint returns `409 Refused by subject guard …` and the CRM
write never executes. This is a reference pattern, not a deployment claim: it lives in a
separate private codebase and is not reproducible from this repo, and the integration is
intentionally fail-open (Recusal is an optional dependency), so it can never block a
previously-working flow. What it demonstrates is the pattern: a deterministic, independent
guard catching a real, named wrong-subject bug before the write runs.

## Honest scope

This proves the **enforcement path** end to end on the real wire format: a real hook, the
real PreToolUse JSON, a real `deny` that Claude Code honors. It does not claim
deployment at fleet scale, that is on the roadmap (tamper-evident
audit logging already ships in `recusal.audit`). What it does claim is true: Recusal
refuses real, dangerous tool calls in the tool people actually use, and it does so to its
own maintainers first.
