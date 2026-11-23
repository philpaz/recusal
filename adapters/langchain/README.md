<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
</p>

# recusal-langchain

[![Website](https://img.shields.io/badge/website-recusal.io-2453c4)](https://recusal.io)
[![PyPI](https://img.shields.io/pypi/v/recusal-langchain)](https://pypi.org/project/recusal-langchain/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/philpaz/recusal/badge)](https://scorecard.dev/viewer/?uri=github.com/philpaz/recusal)
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
call id and a SHA-256 fingerprint of the arguments (never the arguments). An answer the
gate withholds (see "What the gate does not see" below) is one more entry for that call.
A call whose id is empty or not a string is refused, and its entry carries no call id. If
the log cannot be written the call is refused. Check the log with
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
- a parameter the schema omits, on the tool's function or on a `functools.wraps`
  decorator around it (`def wrapper(*args, force=True, **kwargs)`), and a parameter a
  forwarding decorator names that the function it wraps names too;
- a function that hands its arguments on to code the gate cannot read: the innermost
  function the gate reaches takes `*args` or `**kwargs` and does not name every field of
  the schema, or takes nothing but `*args`, `**kwargs` and parameters that never count
  as its own. Those are: a parameter LangChain or the agent's ToolNode fills; one the
  tool marks injected; one named `run_manager`, `callbacks`, `config` or `runtime`,
  whatever its annotation or default; and one whose annotation is, or is built from,
  `RunnableConfig`, `ToolRuntime`, an injected-argument marker or a callback manager or
  run manager type. The annotation is resolved with `typing.get_type_hints` and opened
  through a fixed list of forms (a class and its bases, `Optional` and other generics,
  `Annotated`, a `TypeVar`, a `NewType`, a `type` alias, a forward reference); the
  `recusal-langgraph` README gives the list, and says what happens to a type it cannot
  relate to one of those: the parameter counts as the function's own, and what the
  function hands on is not seen. A tool whose only own parameter
  is one of these is refused. That is a decorator built without `functools.wraps`
  (`def wrapper(*args, **kwargs)` or `def wrapper(*args, config: RunnableConfig,
  **kwargs)`, under a JSON schema dict or the schema `@tool` infers from it), a
  `BaseTool` whose `_arun(*args, **kwargs)` hands the call to `_run` (under `ainvoke`),
  a `BaseTool` with no `args_schema` whose `_run` is `(*args, run_manager=None,
  **kwargs)` or `(*args, config: Optional[RunnableConfig] = None, **kwargs)`, and a
  function that takes only `**kwargs`,
  `RunnableLambda(...).as_tool(args_schema=...)` for one. Give the agent a function of
  your own that names the parameters and calls the tool. Two exceptions: a tool
  `langchain-mcp-adapters` built, below; and a forwarder under an explicit Pydantic
  `args_schema` that declares no field and does not allow extras, through which the
  model can send nothing, so the call is judged as `{}` and runs (such a schema also
  makes LangChain drop the `state`, `store` or `runtime` the agent injected, so a
  function under it runs its own default for that parameter, unseen; under a schema
  with fields the same happens to a parameter the schema does not declare that is named
  `runtime` without the `ToolRuntime` annotation, or is annotated
  `Optional[ToolRuntime]`);
- a function that declares `__signature__`, which LangSmith's `@traceable` and
  `decorator.decorator` set (apply `@traceable` inside the tool's function, or trace at
  the graph level), and one Python cannot inspect, which `cachetools.cached` returns
  (call the cached function from a plain function and give the agent that);
- an argument value the policy cannot be shown faithfully: `pathlib.Path`, `set`,
  `bytes`, `timedelta`, Pydantic's `HttpUrl` and `SecretStr` among them. Declare the
  parameter as `str` and convert it inside the tool;
- a call that sets a name under which the function would not receive the value, when
  the tool's schema lets the key through (a JSON schema dict, a `BaseTool` with no
  `args_schema`, a schema with `extra="allow"`): `config` or `run_manager` on a `@tool`
  or `StructuredTool`, `callbacks` on one whose function names a parameter `callbacks`,
  `self` on any of them, and `func` or `executor_or_config` under `ainvoke` on a
  `BaseTool` subclass with no `_arun` of its own (a tool whose `_run` names one of
  those two as its own parameter is refused there on every call: rename it or define
  `_arun`). Under a Pydantic schema that does not
  declare the name, the schema `@tool` infers included, LangChain drops the key first:
  the call runs, judged without it, and the function never receives it;
- a function parameter named `config` that is not the function's `RunnableConfig`.
  `RunnableLambda(...).as_tool()` over a function typed to take one string is refused
  for its `config`: LangChain builds it around the runnable's own
  `invoke(input, config)`;
- a single-input `Tool(name, func, description)` whose function names more than one
  parameter, or is a builtin such as `str.upper` or `len`, whose parameter is
  positional-only, or whose first parameter is one LangChain fills (`run_manager`,
  `callbacks`, a `config` annotated exactly `RunnableConfig`), or that takes `**kwargs`
  besides one parameter with a reserved name (`def f(config, **kwargs)`). One whose
  function takes exactly one named parameter is judged under that parameter's name,
  `def lookup(config)` included;
- a call whose name is not the name of the tool that would run. An earlier middleware
  that attaches a tool to a call (`request.override(tool=...)`) must set the call's name
  to that tool's name in the same override; otherwise the policy would judge the model's
  string, and the call is refused under the tool's own name;
- a call whose `id` is empty or not a string.

The call id is the model's. A parameter annotated `InjectedToolCallId` is shown to the
policy under its own name; `runtime.tool_call_id` cannot be, so do not use it as data.

**MCP tools.** The tools `langchain-mcp-adapters` builds are judged and run under
`ainvoke` and `astream`; the policy is shown the arguments the adapter sends to the MCP
server, exactly as it sends them. A tool built with `tool_interceptors=` is refused,
because an interceptor can replace the arguments after they were judged. A call in which
the model sets an argument named `config`, `run_manager` or `self` is refused: LangChain
keeps those names when it calls a tool, so the server would not receive the argument,
and a server tool that declares one cannot be given it through this library
(`callbacks` does reach the server, and is judged). Under `invoke`
such tools are refused, as the library gives them no sync function. The
`recusal-langgraph` README has the detail and the releases this is checked against.

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
  tool emulator, a fallback that answers an error by running another tool). The gate
  does not judge what such a middleware runs, and a tool it swapped in has run by the
  time the gate sees the answer. The gate passes an answer on only when it fits what
  its judging step last did for the call, and otherwise withholds it, returns a refusal
  saying so, and in the audited form records that refusal as one more entry. It
  withholds an answer when its judging step never ran; when that step refused the call
  and the answer is not that refusal (an error `ToolMessage` whose text holds the
  gate's reason, with or without a reviewer's note in front); and when that step
  allowed the call and the answer is a `ToolMessage` under another tool's name (a tool
  of your own that builds its `ToolMessage` under a `name` other than the tool's is
  withheld by this too, after it ran: leave `name` out, or use the tool's name). It
  passes on any other answer: after an allowed call, one under the judged tool's name,
  with no name, or of another type (a `Command`), even if the middleware ran a second
  tool to produce it. A middleware listed after the gate that only rewrites the
  arguments is still judged, and recorded, as rewritten, including a value it writes
  under a parameter the tool's function marks as injected: only the state, store and
  runtime the agent's `ToolNode` fills itself are left out of judgment. This is why the
  gate goes last.
- **A decorator that applies another decorator inside itself**, with a kernel that does
  not detect it. `functools.wraps(fn)` makes the outer wrapper point at `fn`, and the
  gate follows that pointer, not what the wrapper calls, so a keyword the inner
  decorator adds or keeps runs with a value the policy did not see. Apply each
  decorator to the function directly, one above the other.
- **A function that names its parameters and then calls another function.** The gate
  reads signatures, not function bodies. The function that is called runs with its own
  defaults, and the policy is shown only the arguments of the first. A decorator built
  without `functools.wraps` that names a parameter
  (`def wrapper(q, *args, **kwargs): return fn(q, *args, **kwargs)`) is such a
  function; one that names no parameter is refused.
- **A default an MCP server applies.** For a `langchain-mcp-adapters` tool the policy
  is shown what is sent to the server. An argument the model leaves out is not sent, and
  the server fills it from its own schema or code; the gate does not see that value.
- **A provider-run tool the gate was not shown.** The refusal above reads the tools of
  the model call as the gate receives them. A middleware listed after the gate that adds
  a dict tool to the model call adds it after the gate looked, and a tool the provider is
  told about through an option on the model object itself is not in that list at all.
- **A sub-agent's tools.** An agent called as a tool is judged as that one tool call.
  The tools the sub-agent runs are gated only if the sub-agent has a gate of its own.
- **`runtime.tools`.** A tool that calls another tool directly bypasses every
  middleware.
- **What a tool does once it runs.**

**Beyond the gate.** This gate judges the tool calls the framework routes through it;
the paths around it inside the framework are listed above. Like every Recusal gate, it
does not stop an agent that can reach a resource another way: a credential with wider
rights, a shell or network path outside the agent, or a file it can write that
configures the gate. Use it as one layer of several: least-privilege credentials for the
agent, a sandbox or container, limits on its network egress, and an audit anchor the
writer cannot reach ([the limits in
full](https://github.com/philpaz/recusal/blob/main/SECURITY.md#things-worth-knowing)).

## Status

**In-depth review complete.** Its fixes are released in 0.2.0. Recusal is a Beta (`0.x`) open-source project in active
development: it is not a finished or certified security product, and it is provided under
Apache-2.0 without warranty.
[What the reviews are, and where every adapter stands](https://github.com/philpaz/recusal#project-status).

**Help make it better.** Read the code, test this adapter against your own stack and tell
us what you find. Reviews, bug reports and pull requests are all welcome:
[how to contribute](https://github.com/philpaz/recusal/blob/main/CONTRIBUTING.md), [open issues](https://github.com/philpaz/recusal/issues).
Please report a security gap [privately](https://github.com/philpaz/recusal/security/advisories/new), not in a public
issue.

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
