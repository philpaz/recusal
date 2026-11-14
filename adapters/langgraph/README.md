# recusal-langgraph

The [Recusal](https://github.com/philpaz/recusal) gate for
[LangGraph](https://langchain-ai.github.io/langgraph/): every tool call a `ToolNode` is
about to execute is judged by your policy first, and a refused call never runs.

This is a separate distribution from `recusal`. The kernel has zero runtime dependencies
and a frozen compatibility perimeter; this package carries the LangGraph dependency and
its own version, so a LangGraph release never forces a release of the kernel.

**Status: not yet published to PyPI.** Until it is, install it from the repository:

```bash
pip install "recusal-langgraph @ git+https://github.com/philpaz/recusal#subdirectory=adapters/langgraph"
```

Requires Python 3.10+, `recusal>=0.14.0`, `langgraph>=1.2.12` and `langchain-core>=1.6.5`.
It is tested on every pull request against exactly `langgraph==1.2.12` with
`langchain-core==1.6.5`, and weekly against the newest releases.

## Use

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from recusal import Finding
from recusal_langgraph import recusal_gate


@tool
def run_sql(sql: str) -> str:
    """Run a SQL statement."""
    ...


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


node = ToolNode([run_sql], wrap_tool_call=recusal_gate(policy))
```

Your `@tool` list goes into `ToolNode` unchanged. Every call in a model message crosses
the gate separately, immediately before execution. A refused call is answered with an
error `ToolMessage` carrying Recusal's reason, and its tool does not run. A policy that
raises refuses the call (fail closed). For an async graph, attach
`recusal_agate(policy)` with `awrap_tool_call` and run with `ainvoke`.

The policy judges the invocation the tool executes, not the dict the model wrote: the
input is validated once, defaults the model omitted are completed, and those same
objects are passed to the function. Arguments the host injects (state, store, runtime)
are left out of judgment, and the model can never supply them.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_langgraph import recusal_audited_gate

log = AuditLog(path="decisions.jsonl")
node = ToolNode([run_sql], wrap_tool_call=recusal_audited_gate(policy, log))
```

`recusal_audited_agate` is the async form. Each call the gate sees, allowed or refused,
is one hash-chained entry labeled `langgraph.toolnode` with LangGraph's tool call id and
a SHA-256 fingerprint of the arguments (never the arguments). If the log cannot be
written the call is refused. Check the log with `recusal audit verify decisions.jsonl`.

The audited gates call the plain gate's own judging function, so there is one decision
path.

## What the gate refuses, and what it does not see

- **Refused before the policy runs:** a tool whose class or instance overrides the
  standard execution lifecycle (it could run without reaching the gate), input that does
  not validate against the tool's schema, and a tool whose callable declares a parameter
  the schema omits, since that is a value no proposal can carry and the policy would
  never see.
- **Not on the audit log:** a call another `wrap_tool_call` hook answered without
  reaching the gate.

This package reads two private LangChain and LangGraph helpers to find the arguments the
host injects. That is why its framework requirements have an upper bound, and why the
weekly lane against the newest releases exists.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_langgraph.__all__`. When a LangGraph release
changes what the hook is handed, this package gets a release and the kernel does not.

A runnable offline demonstration, with a scripted graph and no API key, is
[`examples/langgraph_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/langgraph_gate.py)
and
[`examples/langgraph_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/langgraph_audit.py).
