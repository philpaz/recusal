<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
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

Requires Python 3.10+ (the SDK's minimum), `recusal>=0.15.0` and `openai-agents>=0.24.0`.
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
Under `RunConfig(tool_execution=ToolExecutionConfig(pre_approval_tool_input_guardrails=True))`
the SDK runs the guardrail before it asks a person as well, so such a call is judged
twice: once before the approval and once when it is about to run.

The gate judges the arguments as the SDK prepares them from the tool's schema: validated,
types converted (`{"amount": "50"}` reaches an `int` parameter as `50`) and defaults
filled, each under the name of the parameter that receives it. Which parameter is the
context the SDK fills itself is read from the SDK's own schema for the tool. An allowed
call's prepared arguments are pinned on the tool context, so the SDK runs what was judged.

Build the agent completely, then gate it. A tool appended to the gated agent's list
afterwards is not gated. The gated copy has its own `tools`, `handoffs` and `mcp_servers`
lists, so a handoff or a server added to the original agent after gating is not the
copy's. `gate_tools(tools, policy)` gates a list of tools directly.

### Parameter types

Declare data-only parameters. These are gated and judged by value:

- `str`, `int`, `float`, `bool`, `None`, `Optional[...]` and unions of these;
- `Literal[...]` of strings, integers and booleans;
- `list[...]` and `dict[str, ...]` of the above, and `Any` (judged as the JSON it was sent as);
- any of them with `Field` limits such as `ge=`, `le=` or `max_length=`.

`gate_tools` and `gated_agent` raise `TypeError`, naming the tool and the parameter, for a
parameter the SDK prepares only by running code of the tool's own, because no call to
such a tool could be judged: an `Enum`, a Pydantic model, a dataclass, a `TypedDict`,
`datetime` and `date`, `tuple`, `set`, `bytes`, `Field(pattern=...)`, a custom validator
and a default factory. Take a `str`, an `int` or a `dict` in its place and build the
application type inside the tool body.

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
path: a call the audited gate refuses, the plain gate refuses, for the same cause.

A call that waits for a person's approval is judged, and recorded, when it is about to
run, so a call a person rejected is not on the log. The exception is the SDK option
`pre_approval_tool_input_guardrails=True` (above): the gate then runs before the approval
as well, so a call has an entry before a person decides, a rejected call keeps that one
entry, and an approved call has two entries under the same call id.

## What the gate refuses, and what it does not see

`gated_agent` raises `ValueError` at construction for an agent that can run something the
guardrail never sees. The rule is default-deny:

- **MCP servers attached with `mcp_servers=`, and handoffs to other agents.** For MCP
  servers use
  [`recusal-openai-agents-mcp`](https://github.com/philpaz/recusal/tree/main/adapters/openai-agents-mcp).
- **An agent with a `prompt`.** A prompt stored on OpenAI's servers can configure tools
  that are not in the agent's `tools` list. Put the instructions and the tools on the
  agent itself.
- **`Agent` subclasses** such as `SandboxAgent` and `RealtimeAgent`: a subclass can add
  tools of its own at run time.
- **Function tools that share a name.** Two tools under one name are resolved by the SDK
  last-wins. Two tools under one name in different `tool_namespace` groups both run, but
  the policy and the audit record are given the tool's own name without the namespace
  and could not tell them apart. Give every function tool a unique name.

`gate_tools` and `gated_agent` raise `TypeError`, naming the tool, for a tool the
guardrail cannot judge:

- **Any tool that is not a `FunctionTool`:** hosted tools that run on OpenAI's servers
  (web search, file search, code interpreter, image generation, hosted MCP, tool search)
  and local ones the SDK executes itself (shell, apply patch, computer, custom). A tool
  type added in a later SDK release is refused too. The experimental Codex tool is
  refused as well: it runs its own commands.
- **A `FunctionTool` built by hand** (`FunctionTool(..., on_invoke_tool=...)`). The gate
  could judge only the model's raw JSON, not what your invoker makes of it. Build the
  tool with `function_tool`.
- **A parameter type outside the data-only set** listed under Parameter types above.
- **A variadic parameter** (`*args`, `**kwargs`). Its values have no names to judge by.
  Declare each parameter by name.
- **A wrapped function whose wrapper takes a parameter.** The SDK reads a
  `functools.wraps` decorator as the function it wraps. A wrapper that adds a parameter
  of its own (`dry_run=False`) would run it at a default that is in no schema, and one
  that takes or reorders the function's parameters would hand the function other values
  than the policy judged. A wrapper that forwards `*args, **kwargs` and names no
  parameter is gated. A function with a declared `__signature__` is refused.
- **A sub-agent behind `agent.as_tool(...)` whose own tools are not gated.** Gate the
  sub-agent first, `gated_agent(sub, policy).as_tool(...)`, and the parent accepts it.

Refused call by call, before the policy runs, with a reason the model reads:

- arguments that are not a JSON object, or that do not validate against the tool's
  parameters;
- a non-finite number (`NaN`, `Infinity`, a literal too large for a float, or a string a
  `float` parameter reads as one). It is not JSON data: a policy cannot compare it and a
  record cannot fingerprint it;
- arguments given to a tool that takes none.

What the gate does not see:

- **What a wrapper or the tool body does with the arguments.** The gate judges the
  values the function is called with. A decorator that changes them in its body, and
  anything the tool reads from elsewhere, is not seen.
- **A call the SDK rejects before the guardrail runs.** A tool the agent does not have
  raises `ModelBehaviorError` in the SDK; arguments that hold a lone surrogate character
  raise `UnicodeEncodeError` inside the SDK. Nothing runs, and neither is on the audit
  log.
- **A realtime session.** `gated_agent` refuses a `RealtimeAgent`. `gate_tools` accepts a
  list of function tools whatever they are attached to, but a realtime session runs the
  guardrail on a tool context of its own, so the arguments the gate pinned are not the
  ones that session executes, and this package has no test of a realtime session.

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
