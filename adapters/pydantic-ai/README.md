# recusal-pydantic-ai

The [Recusal](https://github.com/philpaz/recusal) gate for
[Pydantic AI](https://ai.pydantic.dev/): every tool call an agent proposes is judged by
your policy before the tool runs, and a refused call never runs.

This is a separate distribution from `recusal`. The kernel has zero runtime dependencies
and a frozen compatibility perimeter; this package carries the Pydantic AI dependency and
its own version, so a Pydantic AI release never forces a release of the kernel.

**Status: not yet published to PyPI.** Until it is, install it from the repository:

```bash
pip install "recusal-pydantic-ai @ git+https://github.com/philpaz/recusal#subdirectory=adapters/pydantic-ai"
```

Requires Python 3.11+ (Pydantic AI's minimum), `recusal>=0.14.0` and
`pydantic-ai-slim>=2.64.0`. It is tested on every pull request against exactly
`pydantic-ai-slim==2.64.0`, and weekly against the newest release.

## Use

```python
from pydantic_ai import Agent
from recusal import Finding
from recusal_pydantic_ai import RecusalGate


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


agent = Agent(model, tools=[run_sql], capabilities=[RecusalGate(policy)])
```

A refused call is answered to the model as a failed tool result carrying Recusal's
reason. The tool does not run and no retry budget is spent. A policy that raises refuses
the call (fail closed).

`RecusalGate` declares Pydantic AI's `innermost` position, so it judges the arguments as
they reach the tool, after every ordinary capability. The same gate as a plain hook is
`Hooks(before_tool_execute=recusal_before_tool_execute(policy))`; `Hooks` declares no
ordering, so that form must be listed last.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_pydantic_ai import RecusalAuditedGate

log = AuditLog(path="decisions.jsonl")
agent = Agent(model, tools=[run_sql], capabilities=[RecusalAuditedGate(policy, log=log)])
```

Each call the gate sees, allowed or refused, is one hash-chained entry labeled
`pydantic_ai.before_tool_execute` with Pydantic AI's tool call id and a SHA-256
fingerprint of the arguments (never the arguments). If the log cannot be written the call
is refused. Check the log with `recusal audit verify decisions.jsonl`.

The audited gate calls the plain gate's own refusal function, so there is one decision
path.

## What the gate does not see

- **A delegated agent.** An agent called from inside a tool function runs under its own
  capabilities. Give every agent its own gate; `is_gated(agent)` reads only the agent it
  is given.
- **A call that fails validation.** Pydantic AI answers it with a retry prompt before any
  execute hook runs. The tool does not run either, and the call is not on the audit log.
- **Arguments that are not plain data.** JSON scalars, lists and string-keyed dicts are
  judged. Models, dataclasses, Enum members and other objects are refused before the
  policy runs. Declare a `str` or `int` and resolve application types inside the tool.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_pydantic_ai.__all__`. When a Pydantic AI release
changes what the hook is handed, this package gets a release and the kernel does not.

A runnable offline demonstration, with a scripted model and no API key, is
[`examples/pydantic_ai_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/pydantic_ai_gate.py)
and
[`examples/pydantic_ai_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/pydantic_ai_audit.py).
