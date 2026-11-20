<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
    <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
  </picture>
</p>

# recusal-openai-agents

[![PyPI](https://img.shields.io/pypi/v/recusal-openai-agents)](https://pypi.org/project/recusal-openai-agents/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

The [Recusal](https://github.com/philpaz/recusal) gate for the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/): every function tool
call an agent proposes is judged by your policy before the tool runs, and a refused call
never runs.

**New to Recusal?** It checks each tool call an AI agent proposes against your policy and
refuses the unsafe or unapproved ones before the tool runs. No model makes that decision.
A policy is a plain Python function: it is given the tool's name and its arguments and
returns a list of findings. An empty list lets the call run; a failing finding refuses it
and gives the reason.
[Writing a policy](https://github.com/philpaz/recusal#write-the-policy-two-postures-one-principle)
covers the two postures, allowlist and deny-list.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/gate-position.svg">
    <img alt="Before the tool runs. An agent proposes a tool call. Recusal, your policy with no model in it, decides. With no objection the tool runs. On a failing finding, or when the policy itself raises, the call is refused: the tool never runs and the reason goes back to the model. Every decision can be written to a hash-chained audit log, if you pass one." src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/gate-position-light.svg" width="880">
  </picture>
</p>

This is a separate distribution from `recusal`. The kernel has zero runtime dependencies
and a frozen compatibility perimeter; this package carries the SDK dependency and its own
version, so an SDK release never forces a release of the kernel.

Install it from PyPI:

```bash
pip install recusal-openai-agents
```

Requires Python 3.10+ (the SDK's minimum), `recusal>=0.14.0` and `openai-agents>=0.24.0`.
It is tested on every pull request against exactly `openai-agents==0.24.0` with
`openai==3.49.0`, and weekly against the newest releases.

The package ships inline types (`py.typed`) for your editor and type checker.

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

## More

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/adapter-map.svg">
    <img alt="One kernel, eight adapters. Each adapter package depends on recusal: recusal-crewai, recusal-google-adk, recusal-langchain, recusal-langgraph, recusal-openai, recusal-openai-agents, recusal-openai-agents-mcp and recusal-pydantic-ai. recusal-langchain also depends on recusal-langgraph, and recusal-openai-agents-mcp on recusal-openai-agents. The kernel depends on nothing." src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/adapter-map-light.svg" width="880">
  </picture>
</p>

An arrow means "depends on". No arrow leaves the kernel, so a framework's breaking
change is answered by a release of that framework's adapter, not by a release of the
kernel.

- [Framework notes for the OpenAI Agents SDK](https://github.com/philpaz/recusal/blob/main/docs/FRAMEWORKS.md#openai-agents-sdk-function-tools):
  the detail behind this page.
- [All eight adapter packages](https://github.com/philpaz/recusal/tree/main/adapters) and
  [the kernel](https://github.com/philpaz/recusal), which has zero runtime dependencies.
- [Verifying a release](https://github.com/philpaz/recusal/blob/main/docs/VERIFY.md#adapter-packages):
  each release is built in CI, signed with Sigstore and published to PyPI by trusted
  publishing.
- [Changelog](https://github.com/philpaz/recusal/blob/main/CHANGELOG.md),
  [issues](https://github.com/philpaz/recusal/issues),
  [security policy](https://github.com/philpaz/recusal/blob/main/SECURITY.md) and
  [support](https://github.com/philpaz/recusal/blob/main/SUPPORT.md).

Licensed under Apache-2.0.
