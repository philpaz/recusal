# recusal-openai-agents

The [Recusal](https://github.com/philpaz/recusal) gate for the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/): every function tool
call an agent proposes is judged by your policy before the tool runs, and a refused call
never runs.

This is a separate distribution from `recusal`. The kernel has zero runtime dependencies
and a frozen compatibility perimeter; this package carries the SDK dependency and its own
version, so an SDK release never forces a release of the kernel.

**Status: not yet published to PyPI.** Until it is, install it from the repository:

```bash
pip install "recusal-openai-agents @ git+https://github.com/philpaz/recusal#subdirectory=adapters/openai-agents"
```

Requires Python 3.10+ (the SDK's minimum), `recusal>=0.14.0` and `openai-agents>=0.24.0`.
It is tested on every pull request against exactly `openai-agents==0.24.0` with
`openai==3.34.0`, and weekly against the newest releases.

## Use

```python
from agents import Agent, function_tool
from recusal import Finding
from recusal_openai_agents import gated_agent


@function_tool
def run_sql(sql: str) -> str: ...


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


agent = gated_agent(Agent(name="assistant", tools=[run_sql]), policy)
```

`gated_agent` returns a copy of the agent with a Recusal tool input guardrail on every
function tool. A refused call's output is Recusal's reason, which the model reads on its
next turn. The tool does not run. A policy that raises refuses the call (fail closed).
`Runner.run`, `Runner.run_streamed` and a human approval (`needs_approval`) all pass
through the guardrail, which runs when the tool is about to execute, after any approval.

The gate judges the arguments as the SDK prepares them from the tool's schema, defaults
included, and refuses a tool whose schema transforms its input after validation, since
what would execute could not be judged.

Build the agent completely, then gate it. A tool appended to the gated agent's list
afterwards is not gated. `gate_tools(tools, policy)` gates a list of tools directly.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_openai_agents import audited_agent

log = AuditLog(path="decisions.jsonl")
agent = audited_agent(Agent(name="assistant", tools=[run_sql]), policy, log)
```

`audited_tools(tools, policy, log)` is the list form. Each call a guardrail sees, allowed
or refused, is one hash-chained entry labeled `openai_agents.tool_input_guardrail` with
the SDK's tool call id and a SHA-256 fingerprint of the arguments (never the arguments).
If the log cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

The audited forms are the plain ones with a recording passed in, so there is one decision
path.

## What the gate refuses, and what it does not see

`gated_agent` raises at construction for an agent that can run something the guardrail
never sees. The rule is default-deny:

- **Any tool that is not a `FunctionTool`:** hosted tools that run on OpenAI's servers
  (web search, file search, code interpreter, image generation, hosted MCP, tool search)
  and local ones the SDK executes itself (shell, apply patch, computer, custom). A tool
  type added in a later SDK release is refused too.
- **MCP servers attached with `mcp_servers=`, and handoffs to other agents.**
- **A sub-agent behind `agent.as_tool(...)` whose own tools are not gated.** Gate the
  sub-agent first, `gated_agent(sub, policy).as_tool(...)`, and the parent accepts it.
- **`Agent` subclasses** such as `SandboxAgent`, which add tools of their own at run time.

Not on the audit log: a call the SDK rejects before the guardrail runs. A tool the agent
does not have raises `ModelBehaviorError` in the SDK, and nothing runs.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_openai_agents.__all__`. When an SDK release changes
what the guardrail is handed, this package gets a release and the kernel does not.

A runnable offline demonstration, with the SDK's scripted model and no API key, is
[`examples/openai_agents_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_agents_gate.py)
and
[`examples/openai_agents_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_agents_audit.py).
