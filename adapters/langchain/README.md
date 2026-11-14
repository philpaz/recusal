# recusal-langchain

The [Recusal](https://github.com/philpaz/recusal) gate for a
[LangChain](https://docs.langchain.com/) `create_agent` agent: every tool call the agent
proposes is judged by your policy before the tool runs, and a refused call never runs.

This is a separate distribution from `recusal`. The kernel has zero runtime dependencies
and a frozen compatibility perimeter; this package carries the LangChain dependency and
its own version, so a LangChain release never forces a release of the kernel.

A `create_agent` agent executes its tools through a LangGraph `ToolNode`, so this package
adds no decision logic of its own. It wraps the gate of
[`recusal-langgraph`](https://github.com/philpaz/recusal/tree/main/adapters/langgraph) as
LangChain middleware and depends on that package.

**Status: not yet published to PyPI.** Until it is, install both from the repository,
`recusal-langgraph` first since this package requires it:

```bash
pip install "recusal-langgraph @ git+https://github.com/philpaz/recusal#subdirectory=adapters/langgraph"
pip install "recusal-langchain @ git+https://github.com/philpaz/recusal#subdirectory=adapters/langchain"
```

Requires Python 3.10+, `recusal>=0.14.0`, `recusal-langgraph>=0.1.0` and
`langchain>=1.4.3`. It is tested on every pull request against exactly
`langchain==1.4.3` with `langgraph==1.2.12` and `langchain-core==1.6.5`, and weekly
against the newest releases.

## Use

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from recusal import Finding
from recusal_langchain import gate_middleware


@tool
def run_sql(sql: str) -> str:
    """Run a SQL statement."""
    ...


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


agent = create_agent(model=model, tools=[run_sql], middleware=[gate_middleware(policy)])
```

A refused call is answered with an error `ToolMessage` carrying Recusal's reason, and its
tool does not run. A policy that raises refuses the call (fail closed). For
`agent.ainvoke`, use `agate_middleware(policy)`; the sync gate alone raises there.

**List the middleware last.** The first middleware in the list is the outermost, so
anything listed after the gate runs between the gate and the tool.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_langchain import audited_gate_middleware

log = AuditLog(path="decisions.jsonl")
agent = create_agent(
    model=model, tools=[run_sql], middleware=[audited_gate_middleware(policy, log)]
)
```

`audited_agate_middleware` is the async form. Each call the gate sees, allowed or
refused, is one hash-chained entry labeled `langchain.create_agent` with LangChain's tool
call id and a SHA-256 fingerprint of the arguments (never the arguments). If the log
cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

## What the gate does not see

The gate judges, and records, inside the tool's own lifecycle. A middleware listed after
it that rewrites the arguments is still judged and recorded as rewritten. A middleware
listed after it that swaps the tool object, or answers in the tool's place, removes the
judging copy, and that call is neither judged nor recorded. That is why the gate goes
last.

Everything `recusal-langgraph` refuses before the policy runs is refused here too: a tool
that overrides the standard execution lifecycle, input that does not validate, and a
callable with a parameter its schema omits.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_langchain.__all__`. It uses only the public names
of `recusal-langgraph`.

A runnable offline demonstration, with a scripted model and no API key, is
[`examples/langchain_agent_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/langchain_agent_gate.py)
and
[`examples/langchain_agent_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/langchain_agent_audit.py).
