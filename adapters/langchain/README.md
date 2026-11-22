<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
</p>

# recusal-langchain

[![PyPI](https://img.shields.io/pypi/v/recusal-langchain)](https://pypi.org/project/recusal-langchain/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

The [Recusal](https://github.com/philpaz/recusal) gate for a
[LangChain](https://docs.langchain.com/) `create_agent` agent: every tool call the agent
proposes is judged by your policy before the tool runs, and a refused call never runs.

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
and a frozen compatibility perimeter; this package carries the LangChain dependency and
its own version, so a LangChain release never forces a release of the kernel.

A `create_agent` agent executes its tools through a LangGraph `ToolNode`, so this package
adds no decision logic of its own. It wraps the gate of
[`recusal-langgraph`](https://github.com/philpaz/recusal/tree/main/adapters/langgraph) as
LangChain middleware and depends on that package.

Install it from PyPI. `recusal-langgraph`, which this package requires, comes with it:

```bash
pip install recusal-langchain
```

Requires Python 3.10+, `recusal>=0.15.0`, `recusal-langgraph>=0.3.0` and
`langchain>=1.4.10`. It is tested on every pull request against exactly
`langchain==1.4.10` with `langgraph==1.2.20` and `langchain-core==1.7.7`, and weekly
against the newest releases.

The package ships inline types (`py.typed`) for your editor and type checker.

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
tool does not run. A policy that raises refuses the call (fail closed).

`gate_middleware(policy)` is one middleware object that implements both the sync and the
async tool-call hook, so the same agent is gated under `invoke`, `stream` and `batch`
and under `ainvoke`, `astream`, `astream_events` and `abatch`. One is enough:
`agate_middleware(policy)` returns the same kind of object and is kept only because
earlier releases needed it for `ainvoke`. Do not list both; `create_agent` refuses two
middleware with one name. A tool that has only a coroutine runs gated on the async
routes and is refused on the sync ones, where it has no function to judge.

**List the middleware last.** The first middleware in the list is the outermost, so
anything listed after the gate runs between the gate and the tool.

**Give the agent only tools it runs itself.** `create_agent` also takes a tool as a
dict, which it hands to the model as it is: the model provider runs it on its own side
(a provider's web search or code execution), and no tool-call middleware is called for
it. The gate reads the tools of every model call before the model is called. If one is a
dict, the model is not called and the agent's answer is the refusal, which begins
`Recusal refused this model call`. Remove the dict from `tools=`, or give the agent a
tool of your own in its place (`@tool`), which the gate judges.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_langchain import audited_gate_middleware

log = AuditLog(path="decisions.jsonl")
agent = create_agent(
    model=model, tools=[run_sql], middleware=[audited_gate_middleware(policy, log)]
)
```

It serves `invoke` and `ainvoke` alike; `audited_agate_middleware` is the same
middleware under its earlier name. Each tool call the gate sees, allowed or
refused, is one hash-chained entry labeled `langchain.create_agent` with LangChain's tool
call id and a SHA-256 fingerprint of the arguments (never the arguments). A call whose id
is empty or not a string is refused, and its entry carries no call id. If the log
cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

A model call refused because it offers a provider-run tool is not on the log: no tool
call exists at that point.

## What the gate refuses

Everything `recusal-langgraph` refuses is refused here too, with the same reasons; its
README has the full list and says what to do about each
([What the gate takes, what it refuses, and what it does not see](https://github.com/philpaz/recusal/tree/main/adapters/langgraph#what-the-gate-takes-what-it-refuses-and-what-it-does-not-see)).
In short, before the policy runs:

- a tool that overrides the standard execution lifecycle, and input that does not
  validate;
- a parameter the schema omits, on the tool's function or on a decorator around it
  (`def wrapper(*args, force=True, **kwargs)`), and a parameter a forwarding decorator
  names that the function it wraps names too;
- an argument value the policy cannot be shown faithfully: `pathlib.Path`, `set`,
  `bytes`, `timedelta`, Pydantic's `HttpUrl` and `SecretStr` among them. Declare the
  parameter as `str` and convert it inside the tool;
- a function parameter named `config` that is not the function's `RunnableConfig`;
- a single-input `Tool(name, func, description)` whose function names more than one
  parameter. One whose function takes exactly one named parameter is judged under that
  parameter's name;
- a call whose name is not the name of the tool that would run. An earlier middleware
  that attaches a tool to a call (`request.override(tool=...)`) must set the call's name
  to that tool's name in the same override; otherwise the policy would judge the model's
  string, and the call is refused under the tool's own name;
- a call whose `id` is empty or not a string.

The call id is the model's. A parameter annotated `InjectedToolCallId` is shown to the
policy under its own name; `runtime.tool_call_id` cannot be, so do not use it as data.

## What the gate does not see

The gate judges, and records, inside the tool's own lifecycle, in a copy of the tool it
hands on for execution. That decides what it sees:

- **A middleware listed before the gate that runs a tool itself.** A middleware that
  executes a tool in its own `wrap_tool_call` and answers without calling the handler
  ends the call before the gate is reached. The call is not judged and not recorded,
  although the gate is listed last. Have such a middleware attach the tool to the
  request and call the handler instead, so the gate runs it.
- **A middleware listed after the gate that swaps the tool object or answers in the
  tool's place** (a human-in-the-loop edit that redirects the call to another tool, a
  tool emulator). The gate's copy is gone, so the call is not judged. The gate notices
  that its judging step never ran: it withholds the answer, returns a refusal saying so,
  and the audited form records that refusal. A tool that was swapped in has run by then.
  A middleware listed after the gate that only rewrites the arguments is still judged,
  and recorded, as rewritten. This is why the gate goes last.
- **A provider-run tool the gate was not shown.** The refusal above reads the tools of
  the model call as the gate receives them. A middleware listed after the gate that adds
  a dict tool to the model call adds it after the gate looked, and a tool the provider is
  told about through an option on the model object itself is not in that list at all.
- **A sub-agent's tools.** An agent called as a tool is judged as that one tool call.
  The tools the sub-agent runs are gated only if the sub-agent has a gate of its own.
- **`runtime.tools`.** A tool that calls another tool directly bypasses every
  middleware.
- **What a tool does once it runs.**

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_langchain.__all__`. It uses only the public names
of `recusal-langgraph`.

A runnable offline demonstration, with a scripted model and no API key, is
[`examples/langchain_agent_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/langchain_agent_gate.py)
and
[`examples/langchain_agent_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/langchain_agent_audit.py).

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

- [Framework notes for LangChain `create_agent`](https://github.com/philpaz/recusal/blob/main/docs/FRAMEWORKS.md#langchain-create_agent):
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
