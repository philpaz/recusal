# recusal-crewai

The [Recusal](https://github.com/philpaz/recusal) gate for
[CrewAI](https://docs.crewai.com/): every tool call a crew makes is judged by your policy
before the tool runs, and a refused call never runs.

This is a separate distribution from `recusal`. The kernel has zero runtime dependencies
and a frozen compatibility perimeter; this package carries the CrewAI dependency and its
own version, so a CrewAI release never forces a release of the kernel.

**Status: not yet published to PyPI.** Until it is, install it from the repository:

```bash
pip install "recusal-crewai @ git+https://github.com/philpaz/recusal#subdirectory=adapters/crewai"
```

Requires Python 3.10+ (CrewAI's minimum), `recusal>=0.14.0` and `crewai>=1.15.34`. It is
tested on every pull request against exactly `crewai==1.15.34`, and weekly against the
newest release.

## Use

```python
from recusal import Finding
from recusal_crewai import install_recusal_hooks


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


uninstall = install_recusal_hooks(policy)  # after your other hooks
try:
    crew.kickoff()
finally:
    uninstall()
```

The gate is CrewAI's own tool-call hooks. The before hook judges the call and blocks a
refusal; the after hook replaces CrewAI's fixed "Tool execution blocked by hook" line with
Recusal's reason, so the model reads why on its next turn. A policy that raises refuses
the call (fail closed).

**The install is process-global.** CrewAI's hook registry covers every agent, task and
crew in the process, including a coworker an agent delegates to, and every tool kind
CrewAI runs through its hooks.

**Install the gate after your other hooks.** Hooks run in registration order, and a hook
registered after the gate can change the arguments the gate judged.

The policy judges the invocation the tool receives, not the dict the model proposed:
omitted defaults are completed from the tool's schema and values are validated once, the
same way CrewAI prepares them at execution.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_crewai import install_audited_hooks

log = AuditLog(path="decisions.jsonl")
uninstall = install_audited_hooks(policy, log)
```

Each call the before hook sees, allowed or refused, is one hash-chained entry labeled
`crewai.before_tool_call` with a SHA-256 fingerprint of the arguments (never the
arguments). If the log cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

**Entries carry no call id.** CrewAI's hook context has no id for the call, so the record
does not invent one. Entries are told apart by their order on the chain, the tool, the
fingerprint, and `control.crewai_agent`, the role of the agent that made the call.
CrewAI runs one turn's tool calls on a thread pool, so the order of a turn's entries on
the chain is not the order the model proposed them.

## What the gate refuses, and what it does not see

- **Refused before the policy runs:** a tool or schema that is not CrewAI's stock one (a
  schema with a validator, an alias, a constraint, a `default_factory` or an intermediate
  base; a tool class that overrides CrewAI's run or validation methods), an unsupported
  annotation, an unknown or missing parameter, a validation failure, and a callable that
  declares a parameter its schema omits.
- **Not judged:** what a tool does once it runs. A tool that executes code or shell it is
  handed needs a policy that names it.
- **Not on the audit log:** a call a hook registered before the gate blocked, and whatever
  a hook registered after the gate changed.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_crewai.__all__`. When a CrewAI release changes what
the hooks are handed, this package gets a release and the kernel does not.

A runnable offline demonstration, with a scripted model and no API key, is
[`examples/crewai_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/crewai_gate.py)
and
[`examples/crewai_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/crewai_audit.py).
