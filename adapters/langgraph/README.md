<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
</p>

# recusal-langgraph

[![Website](https://img.shields.io/badge/website-recusal.io-2453c4)](https://recusal.io)
[![PyPI](https://img.shields.io/pypi/v/recusal-langgraph)](https://pypi.org/project/recusal-langgraph/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/philpaz/recusal/badge)](https://scorecard.dev/viewer/?uri=github.com/philpaz/recusal)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

The [Recusal](https://github.com/philpaz/recusal) gate for
[LangGraph](https://langchain-ai.github.io/langgraph/): every tool call a `ToolNode` is
about to execute is judged by your policy first, and a refused call never runs.

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
and a frozen compatibility perimeter; this package carries the LangGraph dependency and
its own version, so a LangGraph release never forces a release of the kernel.

Install it from PyPI:

```bash
pip install recusal-langgraph
```

Requires Python 3.10+, `recusal>=0.15.0`, `langgraph>=1.2.20` and `langchain-core>=1.7.7`.
It also declares the two packages it imports that come with those: `langgraph-prebuilt>=1.1.1`
(where `ToolNode` lives) and `pydantic>=2.14.3`.
It is tested on every pull request against exactly `langgraph==1.2.20` with
`langchain-core==1.7.7`, `langgraph-prebuilt==1.1.1` and `pydantic==2.14.3`, and weekly
against the newest releases.

The package ships inline types (`py.typed`) for your editor and type checker.

## Use

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from recusal import Finding
from recusal_langgraph import gated_tool_node


@tool
def run_sql(sql: str) -> str:
    """Run a SQL statement."""
    ...


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


node = gated_tool_node([run_sql], policy)
```

Your `@tool` list goes into `ToolNode` unchanged. Every call in a model message crosses
the gate separately, immediately before execution. A refused call is answered with an
error `ToolMessage` carrying Recusal's reason, and its tool does not run. A policy that
raises refuses the call (fail closed).

`gated_tool_node` returns an ordinary `ToolNode` with the gate on **both** of its hook
slots, so the node refuses the same way on `invoke`, `stream`, `ainvoke` and `astream`. It
takes ToolNode's own options (`name`, `handle_tool_errors`, ...). To build the node
yourself, pass both hooks together:

```python
from recusal_langgraph import is_gated, recusal_gates

node = ToolNode([run_sql], **recusal_gates(policy))
assert is_gated(node)
```

**Why both.** A `ToolNode` has a sync hook slot and an async one, and its sync route
(`invoke`, `stream`) looks only at the sync slot. A node given only `awrap_tool_call` runs
every tool there with no hook at all. `wrap_tool_call=recusal_gate(policy)` on its own is
also safe on every route, since a node with no async hook runs the sync one there too;
a tool that has only a coroutine is then refused on the async route.

A hook left alone in the async slot is the unsafe shape, and the package raises
wherever it is able to:

- `recusal_agate(policy)` raises `TypeError` when you build it, unless you state
  `host_refuses_sync_runs=True`. That statement is true of LangChain's `create_agent`
  middleware and false of a ToolNode; making it for a ToolNode builds the unsafe shape.
- `recusal_gate(policy)` passed as `awrap_tool_call` raises `TypeError` when the node
  runs on its async route (a node built with `handle_tool_errors=True` turns that error
  into a refused call instead; the tool does not run either way).

**What no hook can do:** raise on the sync route of a node whose sync slot is empty. A
ToolNode calls no hook there, so a node built in one of the two ways above and only ever
run with `invoke` or `stream` runs its tools with no gate and no error. Build the node
with `gated_tool_node`, or check it:

`is_gated(node)` is True only for exactly a `ToolNode` (not a subclass) with nothing on
the instance replacing how it runs calls, a sync gate from this package in the sync
slot, and in the async slot either nothing or this package's async gate for the same
policy and the same audit log. A hook wrapped in another function, a hook from
elsewhere, and hooks for two different policies are answered False: some of those may be
fine, but it cannot tell. It reads the node as it is when you call it.

Before 0.2.0 this page told you to attach `recusal_agate(policy)` with `awrap_tool_call`
for an async graph. A node built that way and then run with `invoke` or `stream`
executed its tools ungated and unrecorded. If you have one, replace it with the form
above.

The policy judges the invocation the tool executes, not the dict the model wrote: the
input is validated once, defaults the model omitted are completed, and those same
objects are passed to the function. Arguments the host supplies (state, store, runtime)
are left out of judgment. An argument counts as the host's only when ToolNode put it in
the tool's input for this call, which it does for the tool's state, store and runtime
parameters and for no other: a value that reaches the tool under any other injected name
is judged, whoever wrote it (the model, or a hook that ran after the gate), and so is a
value the model sent that reaches the tool unchanged, whatever the tool's annotations
say. An injected argument that
nothing filled is judged too, at the value the function runs with: the default behind an
`InjectedState("role")` parameter when the state has no `role`, and the default of an
`InjectedToolArg` parameter no host supplies. When the tool's schema marks the name
injected, ToolNode strips whatever the model wrote under it, so the model cannot fill it
and the policy sees the default that runs. In two cases ToolNode does not strip it, the
model's value reaches the function, and that value is judged: when the function marks a
parameter injected and the tool's `args_schema` declares it as an ordinary argument
(ToolNode reads the schema), and when an earlier hook attaches another tool object under
the registered tool's name (ToolNode strips and fills by the tool it registered).

**The call id is the model's.** A parameter annotated `InjectedToolCallId` receives the
tool call id, and the model writes that id. The policy is shown it under the parameter's
own name (for example `tool_call_id`), so for such a tool the policy sees one argument
more than the model's schema lists. `runtime.tool_call_id` is the same model-written
string, and the gate cannot show the policy that one: the runtime object is the host's
and is left out of judgment. Do not use `runtime.tool_call_id` as data (a file name, a
key, part of a query); take the id as an `InjectedToolCallId` parameter, which is judged.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_langgraph import audited_tool_node

log = AuditLog(path="decisions.jsonl")
node = audited_tool_node([run_sql], policy, log)
```

`ToolNode(tools, **recusal_audited_gates(policy, log))` is the same pair for a node you
build yourself. Each call the gate sees, allowed or refused,
is one hash-chained entry labeled `langgraph.toolnode` with LangGraph's tool call id and
a SHA-256 fingerprint of the arguments (never the arguments). An answer the gate
withholds (see "Refused after the fact" below) is one more entry for that call. A call
whose id is empty or not a string is refused, and its entry carries no call id. If the
log cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

The audited gates call the plain gate's own judging function, so there is one decision
path.

## What the gate takes, what it refuses, and what it does not see

**Tools it takes.** A `@tool` or `StructuredTool` function; a `BaseTool` subclass that
implements `_run` (and `_arun`), with an `args_schema` or without one (the gate then
holds it to the schema LangChain infers from `_run`, and completes what the model left
out from the defaults of the method the route runs, since LangChain validates nothing
for such a tool: `_run`, or on `ainvoke` and `astream` the class's own `_arun` when it
defines one, so the value the policy is shown is the one that method runs with; a
`BaseTool` subclass without an `args_schema` whose `_run` names no parameter of its own
besides `*args`, `**kwargs` and the ones LangChain fills is refused, see "A function
that hands its arguments on" below); and a
single-input `Tool(name, func, description)` or `@tool(infer_schema=False)` whose
function takes exactly one named parameter. LangChain hands that function one positional
value whatever key the model wrote it under, so the policy sees it under the function's
own parameter name: `Tool(name="lookup", func=lambda q: ...)` is judged as `{"q": ...}`.

**MCP tools.** The tools `langchain-mcp-adapters` builds (`MultiServerMCPClient.get_tools()`,
`load_mcp_tools`) are judged and run on `ainvoke` and `astream`. The policy is shown the
arguments the adapter sends to the MCP server, exactly as it sends them, under the tool's
name. Four things to know:

- The adapter does not validate the arguments or fill a default the server's schema
  gives: when the model leaves an argument out, the server applies its own default and
  the policy is shown the call without it. See "Not seen" below.
- A tool built with tool interceptors (`tool_interceptors=`) is refused before the policy
  runs: an interceptor runs between the adapter and the server and can replace the
  arguments after they were judged. Build the tools without interceptors, or give the
  node a function of your own that names the parameters and calls the tool.
- A call in which the model sets an argument named `config`, `run_manager` or `self` is
  refused before the policy runs. LangChain keeps the first two names for itself when it
  calls a tool and Python takes the third for the tool object, so the server would not
  receive the argument (with `self` the call fails). A server tool that declares an
  argument under one of these names cannot be given it through `langchain-mcp-adapters`.
  An argument named `callbacks` does reach the server as the model sent it, and is judged.
- The library gives its tools no sync function, so `invoke` and `stream` refuse them (an
  ungated `ToolNode` raises there).

This is the one exception to the rule below about a function that takes only `**kwargs`,
and it is made for that library's own function, recognised by the code object the
library holds, not by its name or signature: a function of your own of the same shape,
and a decorator or forwarder in front of the library's function, are held to the rule.
It was checked with a real MCP server against every `langchain-mcp-adapters` release
from 0.1.12 to 0.3.2. A tool from 0.1.11 or earlier is refused: those releases have no
interceptors, and their function does not have the variable the gate reads to know that.

**Argument values it can judge.** `None`, `bool`, `int`, `str` and a finite `float`; an
`Enum` (by its value); `datetime`, `date` and `time` (as `isoformat()` text); a finite
`Decimal` and a `UUID` (as text); a `list` or `tuple` (as a list); a `dict` with string
keys; a Pydantic v2 model or a dataclass (as a dict of its fields, read field by field so
no serializer runs). A call carrying any other value is refused, because the policy
could not be shown what the tool would receive: `pathlib.Path`, `set` and `frozenset`,
`bytes`, `timedelta`, Pydantic's `HttpUrl` and `SecretStr`, a Pydantic v1 model, a
non-finite number, a value that contains itself. Declare such a parameter as `str` (or a
list) and convert it inside the tool body.

**Refused before the policy runs.** Each of these is answered with the reason, the tool
does not run, and an audited gate records it as one entry:

- A tool whose class or instance overrides the standard execution lifecycle (it could
  run without reaching the gate), or a `StructuredTool` subclass that replaces `_run` or
  `_arun` (it would execute something other than the function the gate inspects). Build
  the tool with `@tool` or `StructuredTool`.
- Input that does not validate against the tool's schema.
- A parameter the schema omits, on the tool's function or on a decorator the gate can
  read around it: that is a value no proposal can carry and the policy would never see.
  The gate reads a decorator built with `functools.wraps`, and each one in a stack of
  them. One that adds a keyword of its own (`def wrapper(*args, force=True, **kwargs)`)
  is the common case. Declare the parameter in the schema, or set the value inside the
  function.
- A function that hands its arguments on to code the gate cannot read. The gate reads
  the tool's function and, through `functools.wraps`, each function a decorator wraps.
  The rule is on the innermost one it reaches: if that function takes `*args` or
  `**kwargs`, every field the schema declares must be one of its parameters. Otherwise
  it passes those arguments on, and whatever receives them fills its own defaults after
  the policy has judged. The call is refused, naming one such field. And if that function
  takes nothing but `*args` and `**kwargs`, the call is refused, with the one exception
  given below. For this rule, and for this rule only, four kinds of parameter never
  count as the function's own:
  - one LangChain or ToolNode fills for this tool: the gate asks them, with the helpers
    they use themselves (see "Private names"), and their answer is never set aside;
  - one the tool marks injected, in its `args_schema` or in its function;
  - one named `run_manager`, `callbacks`, `config` or `runtime`, whatever its
    annotation or default. These are the names LangChain or ToolNode reserve: the ones
    they may write their own value under, keep for themselves or leave out of a schema;
  - one whose annotation is, or is built from, a filled type: `RunnableConfig`,
    `ToolRuntime`, an injected-argument marker (`InjectedState`, `InjectedStore`,
    `InjectedToolArg`, `InjectedToolCallId`, or a subclass of one) or a callback
    manager or run manager type.

  How the gate reads an annotation for that last kind. It first resolves the function's
  annotations with `typing.get_type_hints`, as the frameworks do, so an import alias, a
  quoted name and an annotation under `from __future__ import annotations` are the types
  they name. It then opens these forms, and no others:
  - a class: one of the filled types, a subclass of one, or a `TypedDict` that has
    `RunnableConfig` as a base;
  - `Optional[...]`, a union (`X | None`) and any other generic: its origin and every
    argument;
  - `Annotated[...]`: the type, and each metadata item that is not a string (a marker
    class or an instance of one);
  - a `TypeVar`: its bound and its constraints;
  - a `NewType`: the type it is made from;
  - a `type X = ...` alias (Python 3.12): its value;
  - a forward reference (`Optional['ToolRuntime']`), when the annotations could not be
    resolved together: resolved on its own against the globals of the function's module.

  A string that is `Annotated` metadata (a description) and the values of a `Literal`
  are not read. When a name cannot be resolved at all (it is not in the module's
  globals), the gate falls back to its text: the parameter does not count as the
  function's own when the text contains `RunnableConfig`, `ToolRuntime`, `Injected`,
  `CallbackManager` or `RunManager`, and the refusal says that the type could not be
  resolved and was matched by name. An annotation nested more than 16 levels deep is
  taken for a filled type. A parameter whose type cannot be related to a filled type by
  these means counts as the function's own: a type of your own that holds a filled type
  in a field, a `Protocol`, or an unresolvable name that does not contain one of those
  five words. The function then names a parameter, so it is not refused under this rule,
  and what it hands on falls under "A function that names its parameters and then calls
  another function" in "Not seen".

  So `def wrapper(*args, config: RunnableConfig, **kwargs)`,
  `def _run(self, *args, config: Optional[RunnableConfig] = None, **kwargs)`,
  `def _run(self, config=None, **kwargs)` and
  `def _run(self, *args, run_manager=None, **kwargs)` are all forwarders like
  `def wrapper(*args, **kwargs)`, whether or not LangChain fills that parameter there. A
  tool whose only parameter of its own is one of these is refused, and the refusal names
  the parameter it set aside. Give the node a function of your own that names the
  arguments and calls the tool. This refuses:
  - a decorator built without `functools.wraps` (`def wrapper(*args, **kwargs)`), under
    an explicit `args_schema` and under the schema `@tool` infers from the wrapper, whose
    fields are the wrapper's own `args` and `kwargs`. A `functools.wraps` decorator over
    such a wrapper is refused the same way. Build the decorator with `functools.wraps`,
    or give the tool the function itself;
  - a `BaseTool` subclass whose `_arun(*args, **kwargs)` hands the call to `_run`, on
    `ainvoke` and `astream` (`invoke` and `stream` run `_run` and are judged). Name the
    parameters in `_arun`, or leave `_arun` out: LangChain then runs `_run` on a thread;
  - a function that takes only `**kwargs` or only `*args`:
    `RunnableLambda(...).as_tool(args_schema=...)`; a `BaseTool` whose `_run` is
    `(**kwargs)`, `(*args)` or `(*args, run_manager=None, **kwargs)`, with an
    `args_schema` that declares fields or with no `args_schema` at all; and a function
    of your own written as `def call(**arguments)`, `@tool def call(**kwargs)`
    included. Among published tools (read from their source, and checked on tools built
    to the same signatures and schemas, not by installing them) this is the shape of
    the following. The list is not exhaustive: it names the tools a reading of some
    integration packages found, and any other tool written the same way is refused the
    same way.
    langchain-community's `SlackGetChannel` (no `args_schema`;
    `_run(self, *args, run_manager=None)`); its AINetwork tools (`AINBaseTool`), its
    `ConneryAction` and langchain-ibm's `WatsonxTool` whenever their schema declares a
    field, which their `_run` does not name; toolbox-langchain's `ToolboxTool` and
    `AsyncToolboxTool`, which is every tool Google's MCP Toolbox client loads
    (`_run(self, **kwargs)`); langchain-anchorbrowser's tools (`_run(self, **kwargs)`
    under schemas with fields); langchain-writer's `NoCodeAppTool`
    (`_run(self, *args, **kwargs)`, no `args_schema`); the tools langchain-community's
    Databricks `UCFunctionToolkit` builds (a `StructuredTool` over
    `def func(*args, **kwargs)` under a generated `args_schema`); and the tools of
    langchain-snowflake's MCP integration (`Tool(func=wrapper.run)` with
    `run(self, *args, **kwargs)`, refused under the single-input rule below). Two that
    take no input run under
    the exception below, because their schema declares no field: langchain-aws's
    `NovaSystemTool` and langchain-cohere's `DirectlyAnswerTool`. The gate cannot tell
    such a function from a decorator that forwards. Give the node a function that names
    the parameters and calls the tool
    (`async def read_file(path: str, depth: int = 2): return await
    other_tool.ainvoke({"path": path, "depth": depth})` under `@tool`). One exception
    is the function `langchain-mcp-adapters` defines for its tools (see "MCP tools"
    above), when it is the tool's own callable and the tool has no interceptors.

  The other exception: a forwarder runs when the model can put nothing through the tool's
  schema. That is an explicit Pydantic v2 `args_schema` that declares no field and does
  not set `extra="allow"`. LangChain hands the function only the fields such a schema
  declares, so whatever the model sends is dropped: the call carries no value the model
  chose, it is judged as `{}`, and it runs. What the function then does is its own
  body, which the gate does not read. No other schema counts: a JSON schema dict does
  not, whatever its `properties` and `additionalProperties` say, because LangChain
  validates nothing against a dict and passes every key on; the schema inferred from a
  forwarder does not (it declares `args` and `kwargs`); and a `BaseTool` subclass with
  no `args_schema` does not. One consequence of such a schema applies to any function
  under it, forwarder or not: LangChain drops what ToolNode injected as well, so a
  `state`, `store` or `runtime` parameter of the function receives the function's own
  default, not the host's value, and the policy is not shown that default. It is a
  constant of the code, the same with no gate, and nothing the model can change. The
  same happens under a schema that does declare fields, to a parameter the schema omits
  and LangChain does not itself take for injected: one named `runtime` whose annotation
  is not `ToolRuntime` (`def lookup(q: str, runtime: str = "eu")` under a schema that
  declares only `q`), and one annotated `Optional[ToolRuntime]`. LangChain hands the
  function only what the schema declares and what it takes for injected, so the function
  runs its own default: the policy is shown `{"q": ...}` and the function runs
  `runtime="eu"`. Under a schema with fields, a parameter annotated `ToolRuntime`
  itself, or `Annotated` with an injected-argument marker, does receive the host's
  value.

  A function that takes `**kwargs` and also names every schema field runs: the variadic
  then carries only keys the schema does not declare (the ones an `extra="allow"` schema
  accepts, which are judged). The schema LangChain infers for `def f(q, **kwargs)`
  declares `kwargs` itself; on a function that names at least one parameter of its own,
  the variadic's own name counts as a parameter, so that runs too. The gate reads
  signatures, not bodies: if such a function is itself a forwarder
  (`def wrapper(q, *args, **kwargs): return fn(q, *args, **kwargs)`), the defaults of
  the function it calls are not seen (see "Not seen").
- A parameter that a forwarding decorator (`*args`, `**kwargs`) names and the function it
  wraps names too (`def wrapper(*args, timeout=30, **kwargs)` around
  `def fetch(url, timeout=...)`): the decorator keeps the judged argument and the
  function runs with its own default. Drop the name from the decorator's signature and
  read the argument from `kwargs` there, so it still reaches the function.
- A function that declares `__signature__`: the gate cannot tell which parameters it
  runs with. LangSmith's `@traceable` and the `decorator` package's
  `decorator.decorator` both set one on the function they return, so `@tool` over either
  is refused. Apply `@traceable` inside the tool's function (to a function it calls), or
  trace at the graph level; for `decorator.decorator`, call the decorated function from
  the tool's function.
- A function whose parameters cannot be inspected. A class-based decorator that is also a
  descriptor (its class defines `__get__`) is one: Python's `inspect.signature` finds no
  signature for it. `cachetools.cached` returns such an object. A function under more
  than 31 `functools.wraps` decorators, one inside the other, is another: the walk reads
  no deeper, and the refusal says so. Call the decorated function from a plain function
  and give the tool that.
- A positional-only parameter.
- A function parameter named `config` on a `@tool` or `StructuredTool` function, unless
  it is the function's `RunnableConfig`: LangChain writes the run configuration under
  that name, so the function never receives the model's value. `gated_tool_node` and
  `audited_tool_node` raise `TypeError` for such a tool when the node is built. Rename
  the parameter. The first parameter of a single-input `Tool`'s function is not such a
  parameter: it receives the tool's one value by position whatever it is called, so
  `Tool(name="lookup", func=lookup)` over `def lookup(config)` is built and judged as
  `{"config": ...}`. A call is refused for the same reason when it sets a name under which
  the function would not receive the value, and only then: `config` or `run_manager` on
  a `@tool` or `StructuredTool` (LangChain keeps both for itself there); `callbacks`
  on one whose function names a parameter `callbacks`, itself or behind a
  `functools.wraps` decorator (LangChain overwrites it); `run_manager` on a `BaseTool`
  subclass whose method for that route (`_run`, or its own `_arun`) names it. Where
  the function does receive the value, the call is judged with it: `callbacks` that
  reaches a function through `**kwargs`, and `callbacks` on a `BaseTool` subclass's
  `_run`. Two more names are refused because of a collision in the call itself. A
  call that sets `self` could not be made at all: LangChain passes the arguments as
  keywords to a method of the tool object, and Python raises for a second `self`. And
  on `ainvoke` and `astream` of a `BaseTool` subclass that has no `_arun` of its own,
  LangChain runs `_run` through `run_in_executor`, which takes `func` and
  `executor_or_config` for itself: a call that sets one of them could not be made, and
  a tool whose `_run` names one of them as its own parameter is refused there on every
  call, because the gate hands the function each judged value by name and that name
  cannot be passed (with no gate such a tool runs while the model leaves the parameter
  out, and fails when it sends one). Rename the parameter, or define `_arun`.
  All of this applies only when the key reaches the gate's judging step, which depends
  on the schema. A Pydantic schema that does not declare the name drops the key before
  that: under the schema `@tool` infers, a call that sets `self`, `config` or
  `run_manager` runs, judged without the key, and the function never receives it. A
  JSON schema dict, a `BaseTool` subclass with no `args_schema`, and a schema with
  `extra="allow"` let the key through, and there the call is refused. A single-input
  `Tool` takes one value whatever key the model wrote it under, `self` included, and
  judges it under the function's parameter name.
  `RunnableLambda(...).as_tool()` over a function typed to take one string is
  refused as well, when the node is built and on every call: LangChain builds that tool
  around the runnable's own `invoke(input, config)`. Write a function that calls the
  runnable and give the node that
  (`def shout(text: str) -> str: return runnable.invoke(text)` under `@tool`).
- A `config` parameter that is not annotated exactly `RunnableConfig`
  (`Optional[RunnableConfig]`, `RunnableConfig | None`, no annotation). LangChain fills
  `config` only under the exact annotation, so it never passes a run configuration to
  such a parameter. What the gate then does depends on the tool's schema. Under an
  explicit `args_schema` that does not declare `config`, it is a parameter the schema
  omits, and the call is refused under that rule above. Under the schema LangChain
  infers for a `BaseTool` subclass with no `args_schema`, `config` is a declared field
  the model can set, and the call is judged with it, unless `config` is the tool's only
  parameter besides `*args` and `**kwargs`: then the tool is a forwarder and is refused
  under the rule above, because `config` is a reserved name. All eleven browser tools
  of langchain-aws are written this way and are refused (read from their source, and
  checked on tools built to the same signatures and schemas): the four that take no
  input, `ThreadAwareNavigateBackTool`, `ThreadAwareExtractTextTool`,
  `ThreadAwareExtractHyperlinksTool` and `ThreadAwareCurrentWebPageTool`
  (`_run(self, config: Optional[RunnableConfig] = None, **_)`), and the seven that take
  input, `ThreadAwareNavigateTool`, `ThreadAwareClickTool`, `ThreadAwareGetElementsTool`,
  `ThreadAwareScreenshotTool`, `ThreadAwareScrollTool`, `ThreadAwareTypeTool` and
  `ThreadAwareWaitForElementTool` (for example
  `_run(self, url: str, config: Optional[RunnableConfig] = None, **_)`). Give the node
  a function of your own that names the arguments and calls the tool
  (`def navigate(url: str) -> str: return aws_tool.invoke({"url": url})` under `@tool`).
- A parameter named `runtime` with a plain type such as `str`, under a schema that
  declares it (the one `@tool` infers does): ToolNode puts its runtime object under that
  name, so the input does not validate. Rename the parameter.
- A Pydantic v1 `args_schema`: the gate reads no fields from it, so it cannot hold the
  function to it and every call is refused. Use a Pydantic v2 model.
- A single-input `Tool` when any function the call reaches names a second parameter;
  when its function does not take the input as its first named parameter (a decorator
  with `*args` in front of it, or a builtin such as `Tool(func=str.upper)` or
  `Tool(func=len)`, whose one parameter is positional-only; a method that takes only
  `*args` and `**kwargs`; or a function whose first parameter is one LangChain fills:
  `run_manager`, `callbacks`, or a `config` annotated exactly `RunnableConfig`); when
  the innermost function the gate reaches takes `*args` or `**kwargs` and does not name
  the input; or when that function takes `*args` or `**kwargs` and its one named
  parameter is one that never counts as its own under the forwarder rule above
  (`def f(config, **kwargs)`). Use `@tool` with a schema, or wrap the builtin
  (`func=lambda text: text.upper()`).
- A call whose name is not the name of the tool that would run. That happens when an
  earlier hook attaches a tool to a call ToolNode could not resolve
  (`request.override(tool=...)`): the policy would judge the model's string and the
  record would name a tool that did not run. The refusal is made, and recorded, under
  the tool's own registered name. Repair the name in the hook too
  (`request.override(tool_call={**request.tool_call, "name": tool.name}, tool=tool)`).
- A call whose `id` is empty or not a string: its answer and its record could not be
  tied to the call.
- A tool whose host-injected arguments cannot be read from the installed LangChain and
  LangGraph (see the private names below).
- A tool the node does not have.

**Refused after the fact, when a tool may have run.** The gate judges inside the copy
of the tool it hands to execution, and it passes an answer on only when the answer fits
what that copy's judging step last did for the call. In three cases it withholds the
answer and returns a refusal that says so, and an audited gate records that refusal as
one entry:

- The judging step never ran. Nothing was judged. This is what happens if an installed
  LangGraph ran the tool some other way than through the copy, or if a hook between the
  gate and the tool swapped the tool or answered for it.
- The judging step refused the call, and the answer that came back is not that refusal
  (an error `ToolMessage` whose text holds the gate's reason). This is what happens if a
  hook between the gate and the tool answers a refused call by running another tool, a
  fallback to a replica for example. The log then holds two entries for the call: the
  refusal, and the withheld answer.
- The judging step allowed the call, and the answer is a `ToolMessage` under another
  tool's name. The gate cannot tell who made such a message, so this also withholds the
  answer of a judged tool that builds its own `ToolMessage` and gives it a `name` other
  than the tool's, after that tool has run. In such a tool leave `name` out, or set it
  to the tool's name.

The refusal does not undo anything: if a tool ran, what it did has happened.

**Refused, never run unjudged.** A tool body reached without the gate's judging step.
The copy of the tool the gate hands to execution runs its function only once per
allowance, and only with the argument objects that step judged. This covers the callable
the tool calls; a decorator that changes the arguments on their way to the function it
wraps is covered only by the decorator refusals above.

**Not on the audit log.** A call another `wrap_tool_call` hook answered without reaching
the gate. And a call ToolNode itself fails before it invokes the tool, for example
because a required `InjectedState` field is not in the state: the tool does not run, the
gate has decided nothing, and no entry is written.

**Not seen.**

- A `ToolNode` you did not gate. `is_gated` reads one node; a graph can hold others,
  and `create_react_agent(model, [tools])` builds its own ungated one (pass it your
  gated node instead).
- A tool that calls another tool directly, for example through `runtime.tools`: it
  bypasses every hook.
- `runtime.tool_call_id`, as said above.
- A default an MCP server applies. For a `langchain-mcp-adapters` tool the policy is
  shown what is sent to the server. An argument the model leaves out is not sent, the
  server fills it from its own schema or code, and neither the adapter nor the gate sees
  that value. A policy that must bound such an argument has to refuse the call when the
  argument is absent.
- With a kernel that does not detect it, a decorator that applies another decorator
  inside itself is not read. `functools.wraps(fn)` makes the outer wrapper point at
  `fn`, and the gate follows that pointer, not what the wrapper calls:
  `def resilient(fn): guarded = with_timeout(30)(fn)` with a `functools.wraps(fn)`
  wrapper that calls `guarded`. A keyword the inner decorator adds, or keeps from the
  function, then runs with a value the policy did not see. Apply each decorator to the
  function directly, one above the other; the gate reads that stack.
- A function that names its parameters and then calls another function. The gate reads
  signatures, not function bodies: the function it calls runs with its own defaults and
  with whatever values the first one hands it, and the policy is shown only the
  arguments of the first. A decorator built without `functools.wraps` that names a
  parameter (`def wrapper(q, *args, **kwargs): return fn(q, *args, **kwargs)`) is such
  a function: `fn`'s defaults are not seen. One that names no parameter is refused.
- A tool that a hook between the gate and the tool ran besides the judged call, unless
  the answer that comes back fails one of the three checks under "Refused after the
  fact". After a call the judging step allowed, the gate withholds only a `ToolMessage`
  under another tool's name: an answer under the judged tool's name, with no name, or of
  another type (a `Command`) is passed on.
- A parameter named `run_manager` or `callbacks` in a place where LangChain does not
  fill it. LangChain keeps those two names out of every schema, and the gate does not
  hold a parameter of either name to the schema. In four places nothing fills it, so
  the function runs with its own default for that parameter and the policy does not
  see it: `run_manager` on a `@tool` or `StructuredTool` function; `run_manager` on a
  single-input `Tool` function; `callbacks` on a `BaseTool` subclass's `_run`; and a
  `callbacks` keyword on a `functools.wraps` decorator over a `@tool` or
  `StructuredTool` function that does not itself name `callbacks` (LangChain reads the
  wrapped function's signature and finds none). On a `BaseTool` subclass's `_run` a
  parameter named `callbacks` is also a real argument the model can set when the tool's
  schema lets the key through, as it does when the tool has no `args_schema`: a call
  that sets it is judged with the value, and a call that leaves it out runs the default
  unseen. With
  `def _run(self, callbacks: str = "all")` and no `args_schema`, the policy is shown
  `{}` and the function runs `callbacks="all"`. Rename the parameter.
- A host value a schema drops: a `state`, `store` or `runtime` parameter under an
  explicit schema that declares no field, and under a schema with fields that does not
  declare it, a parameter named `runtime` that is not annotated `ToolRuntime` or one
  annotated `Optional[ToolRuntime]`. The function runs
  its own default for it, a constant of the code (see the closed-schema exception
  above).
- What a tool does once it runs.

**Private names.** This package depends on ten names that are private to LangChain and
LangGraph (each begins with an underscore and carries no compatibility promise):

- two helpers it imports, `langchain_core.tools.base._get_runnable_config_param` and
  `langgraph.prebuilt.tool_node._get_all_injected_args` (of what the second returns it
  reads `all_injected_keys`, `state`, `store` and `runtime`);
- four members of `BaseTool`: `_injected_args_keys`, which it reads;
  `_to_args_and_kwargs`, which it replaces on a per-call copy of the tool; and `_run` and
  `_arun`, which it reads and guards on that copy;
- four members of `ToolNode`, read only by `is_gated`: `_wrap_tool_call`,
  `_awrap_tool_call`, `_func` and `_afunc`.

A missing name never falls back to running ungated. Without either helper the package
does not import. A tool whose `_injected_args_keys` cannot be read, or for which the
injected-arguments helper raises, is refused on every call. A tool class that does not
have LangChain's own `_to_args_and_kwargs` is refused as a lifecycle override. `is_gated`
answers False for a node that is not laid out as it expects. This is why `langgraph`,
`langchain-core`, `langgraph-prebuilt` and `pydantic` are each declared with an upper
bound at the next major version, and why the weekly lane against the newest releases
exists: a release inside those bounds can still move a private name, and
`tests/test_compat_langgraph.py` fails when one does.

For `langchain-mcp-adapters`, which this package does not import or require, the gate
reads three names when that library is loaded: the module `langchain_mcp_adapters.tools`,
its function `convert_mcp_tool_to_langchain_tool`, and the variable `tool_interceptors`
that function's inner `call_tool` holds. If a release moves any of them, its tools are
refused (as a function that hands its arguments on, or as one whose interceptors cannot
be read); none is run unjudged. A test pins the three against the library where it is
installed; the pull request lanes do not install it.

**Beyond the gate.** This gate judges the tool calls the framework routes through it;
the paths around it inside the framework are listed above. Like every Recusal gate, it
does not stop an agent that can reach a resource another way: a credential with wider
rights, a shell or network path outside the agent, or a file it can write that
configures the gate. Use it as one layer of several: least-privilege credentials for the
agent, a sandbox or container, limits on its network egress, and an audit anchor the
writer cannot reach ([the limits in
full](https://github.com/philpaz/recusal/blob/main/SECURITY.md#things-worth-knowing)).

## Status

**In-depth review complete.** Its fixes are released in 0.3.0. Recusal is a Beta (`0.x`) open-source project in active
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
public names are the ones in `recusal_langgraph.__all__`. When a LangGraph release
changes what the hook is handed, this package gets a release and the kernel does not.

A runnable offline demonstration, with a scripted graph and no API key, is
[`examples/langgraph_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/langgraph_gate.py)
and
[`examples/langgraph_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/langgraph_audit.py).

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

- [Framework notes for LangGraph](https://github.com/philpaz/recusal/blob/main/docs/FRAMEWORKS.md#langgraph):
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
