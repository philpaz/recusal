# Proven, Recusal governs its own repository

Recusal isn't only shown on toy inputs. It governs the development of *this* repository.
A real Claude Code `PreToolUse` hook, [`.claude/hooks/recusal_gate.py`](../.claude/hooks/recusal_gate.py)
- is installed so that when a Claude Code session works on this repo, every tool call is
adjudicated first. Destructive shell commands and writes to secret/protected files are
**refused before they run**, even under `bypassPermissions`.

The governance library is the thing governing its own development. That is the dog food.

## What's installed

- [`.claude/hooks/recusal_gate.py`](../.claude/hooks/recusal_gate.py): the hook and its
  repo-protection policy (refuse `rm -rf`, force-push, `reset --hard`, `curl … | sh`, writes
  to `.env` / `*.pem` / `*.key`, …).
- Registration, see [`.claude/settings.json.example`](../.claude/settings.json.example).
  Installing a permission-changing hook is a **deliberate, reviewed step**: Claude Code's
  own auto mode will (correctly) ask you to confirm it. That confirmation *is* the
  separation-of-powers point, a control that changes what an agent may do should not be
  installed silently.

## It refuses, for real

Each block below is the **verbatim** output of piping the exact JSON Claude Code sends on
`PreToolUse` into the installed hook. Reproduce any line yourself:

```bash
echo '<payload>' | python3 .claude/hooks/recusal_gate.py
```

**`rm -rf`**
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing destructive command containing 'rm -rf'"}}
```

**`git push --force`**
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing destructive command containing 'git push --force'"}}
```

**`curl … | sh`**
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing to pipe a network download straight into a shell"}}
```

**Write to `.env`**
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Write` [FAIL]: refusing write to a protected/secret file: /repo/.env"}}
```

A clean call, a `Read`, or `Bash` running `pytest -q`, produces **no output**: the hook
*defers* to Claude Code's normal permission flow. The gate only ever adds refusals; it never
strips your existing prompts.

## Locked by CI

[`tests/test_dogfood.py`](../tests/test_dogfood.py) loads this exact policy and asserts the
refusals (and the defers), so the proof cannot silently rot, every CI run re-proves it.

## Proven again, a real bug in a real system

Recusal is also wired into a **production FastAPI agent backend** (a relationship agent
with confirm-gated CRM writes) to close a named audit finding, **wrong-subject write**:
the agent could stage a write against a *different* member than the conversation was about,
and the human confirm card would wave it through.

A subject-match guard at the confirm endpoint adjudicates the write with Recusal's
`compute_verdict`, the write must target the member who was active when it was proposed.
Real output from the wired guard:

```text
wrong subject  -> write targets C-9988 but the active member this turn is C1001   (refused)
right subject  -> None                                                            (allowed)
no active mbr  -> None                                                            (fail-open)
```

On a mismatch the confirm endpoint returns `409 Refused by subject guard …` and the CRM
write never executes. It is proven by tests against the live endpoint, and the service's
existing confirm/CRM tests (27) still pass, a deterministic, independent verifier catching
a real, named bug with **zero regression**. The integration is intentionally fail-open and
treats Recusal as an optional dependency, so it can never block a previously-working flow.

## Honest scope

This proves the **enforcement path** end to end on the real wire format: a real hook, the
real PreToolUse JSON, a real `deny` that Claude Code honors. It does not yet claim
production scale or deployment across a fleet, those are on the roadmap (tamper-evident
audit logging already ships in `recusal.audit`). What it does claim is true: Recusal
refuses real, dangerous tool calls in the tool people actually use, and it does so to its
own maintainers first.
