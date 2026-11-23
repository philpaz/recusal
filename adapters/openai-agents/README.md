<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
</p>

# recusal-openai-agents

[![Website](https://img.shields.io/badge/website-recusal.io-2453c4)](https://recusal.io)
[![PyPI](https://img.shields.io/pypi/v/recusal-openai-agents)](https://pypi.org/project/recusal-openai-agents/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/philpaz/recusal/badge)](https://scorecard.dev/viewer/?uri=github.com/philpaz/recusal)
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

A sub-agent behind `agent.as_tool(...)` is called like a function tool, and the parent's
policy judges the input the sub-agent is given. With the default `as_tool` that is the
arguments as the model wrote them, `{"input": "..."}`. With `as_tool(parameters=Model)`
it is what the SDK's own adapter for that model makes of them: validated, types
converted (`{"rows": "9"}` reaches the sub-agent as `9`), defaults filled, and a key the
model does not name dropped. Arguments the model does not accept are refused before the
policy runs. The sub-agent's own tool calls are judged by the sub-agent's gate, which
`gate_tools` requires.

Build the agent completely, then gate it. A tool appended to the gated agent's list
afterwards is not gated. The gated copy has its own `tools`, `handoffs` and `mcp_servers`
lists, so a handoff or a server added to the original agent after gating is not the
copy's. `gate_tools(tools, policy)` gates a list of tools directly.

### Parameter types

Declare data-only parameters. These are gated and judged by value:

- `str`, `int`, `float`, `bool`, `None`, `Optional[...]` and unions of these;
- `Literal[...]` of strings, integers and booleans;
- `list[...]` and `dict[str, ...]` of the above, a `dict` keyed by a `Literal` of strings
  (`dict[Literal["a", "b"], int]`), and `Any` (judged as the JSON it was sent as);
- any of them with `Field` limits such as `ge=`, `le=` or `max_length=`;
- any of them with a default that is JSON data, or with `Field(default_factory=list)` or
  `Field(default_factory=dict)`.

A `dict` parameter needs `function_tool(..., strict_mode=False)`: under the SDK's default
strict schema the SDK itself refuses to build such a tool (`UserError`,
"additionalProperties should not be set").

`gate_tools` and `gated_agent` raise `TypeError`, naming the tool and the parameter, for a
parameter no call to which could be judged:

- a type or a validation the SDK prepares only by running code of the tool's own: an
  `Enum`, a Pydantic model, a dataclass, a `TypedDict`, `datetime` and `date`, `tuple`,
  `set`, `Sequence[...]` and `Iterable[...]`, `bytes`, `Field(pattern=...)` and a custom
  validator. Take a `str`, an `int`, a `list` or a `dict` in its place and build the
  application type inside the tool body;
- a default factory other than `list` or `dict` (`Field(default_factory=lambda: [...])`).
  Use a plain default and build the value inside the tool body;
- a default that is not JSON data (`tags: list[str] = ()`, `timeout: float = math.inf`).
  Give it a JSON default: `None`, a string, a finite number, a boolean, a list or a dict;
- a mapping whose key type is neither `str` nor a `Literal` of strings (`dict[int, str]`,
  `dict[Literal[1, 2], str]`). The SDK converts the keys of the JSON object the model
  sends, and the gate judges arguments as JSON data, whose keys are strings. Declare
  `dict[str, ...]` and convert the keys inside the tool body.

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
- **An agent whose `model_settings` can put tools into the request.** `extra_body` or
  `extra_args` with a `tools`, `prompt`, `plugins`, `web_search_options` or `mcp_servers`
  key is refused. The SDK merges `extra_body` into the body of every model request: a
  `tools` key there replaces the agent's own tool list in the request, and each of the
  other four is sent beside it. `extra_args` are passed as arguments of the request call:
  a `prompt` there is sent with a Responses request, and a `tools` key makes the SDK's
  OpenAI models raise `TypeError` before anything is sent. Put the tools on the agent
  itself.
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
- **A parameter outside the data-only set** listed under Parameter types above: its
  type, its default, or the key type of a mapping.
- **A variadic parameter** (`*args`, `**kwargs`). Its values have no names to judge by.
  Declare each parameter by name.
- **A wrapped function whose wrapper takes other parameters than the function.** The SDK
  reads a `functools.wraps` decorator as the function it wraps. A wrapper that adds a
  parameter of its own (`dry_run=False`) would run it at a default that is in no schema,
  and one that takes some of the function's parameters, or takes them in another order,
  would hand the function other values than the policy judged. Two kinds of wrapper are
  gated: one that forwards `*args, **kwargs` and names no parameter, and one that takes
  exactly the function's parameters, by the same names, in the same order and of the
  same kinds. So is a stack of them, each level read in turn. A stack of more than 31
  wrappers, or more than eight `functools.lru_cache` layers one on another, is refused
  as more than the gate reads. A function with a declared `__signature__` is refused.
- **A decorator whose signature cannot be read.** A class-based decorator that is also a
  descriptor (`cachetools.cached`, `cachetools.func.ttl_cache`, `cachetools.func.lru_cache`)
  hides the functions under it from inspection. Call the decorated function from a plain
  function and register that. `functools.lru_cache` and `functools.cache` are read
  through and gated.
- **A `functools.singledispatch` function or a `singledispatchmethod`, with or without
  an implementation registered.** The implementation that runs is chosen at call time by
  the type of an argument, each one has its own parameters and defaults, and one can be
  registered at any moment. The SDK checks every call of a model turn before any tool
  body runs, so an implementation that one call of a turn registers (or that the tool
  registers on its own first use) would run, unjudged, for a later call of that turn. So
  every single-dispatch function is refused when the agent is built, whether or not
  anything is registered. Register one plain function as the tool and branch on the type
  inside its body (`if isinstance(...)`). Do not call a single-dispatch function from
  that body in its place: a single-dispatch function called from the body is not read,
  and the implementation it picks runs with its own defaults. The dispatcher is refused
  wherever it stands among the callables the gate reads: as the tool itself, as a
  method, a class method or a static method (`function_tool(service.handle)`), as a
  callable object's `__call__`, under `functools.wraps` wrappers or
  `functools.lru_cache`, and as the callable a `functools.partialmethod` holds. It is
  recognised in three ways, any one of which is enough: by the code of the function
  `functools.singledispatch` returns (as read when this package was imported, and as
  `functools` stands when the tool is checked, so after `importlib.reload(functools)`
  too); by how that function is built (its closure holds a `dispatch` function whose
  closure holds a `registry` dict, a `dispatch_cache` and a `cache_token`, as
  `functools` builds it on Python 3.10 to 3.15); and by its attributes (a `registry`
  that is a `types.MappingProxyType` beside a callable `dispatch` and a callable
  `register`, on any object, so a service object of your own that has all three is
  refused too). A `functools.wraps` wrapper over a dispatcher carries a copy of those
  attributes and is refused with it. Not taken for one: a callable object that happens
  to have a `dispatch` method and a `registry` attribute that is not a mapping proxy,
  and an event bus of your own whose function closes over a `dispatch` function and a
  `registry` dict. A dispatcher that a wrapper calls from a closure, a global or an
  attribute, and not through `__wrapped__`, is not found: see "What the gate does not
  see" below.
- **A `functools.partialmethod`.** What the tool calls is the callable the partialmethod
  holds, and Python reads the tool's parameters from that callable without anything
  reading what is under it: a single-dispatch function or a wrapper with a parameter of
  its own there would run with a value no policy judged. So a function that is or
  carries a partialmethod is refused, whatever it holds, a plain function too. That is
  what you get when you read a partialmethod from the class (`Db.export`), and from an
  object (`db.export`) when the callable it holds has no `__get__`: a class-based
  decorator's instance, or a `functools.partial` up to Python 3.13. A partialmethod over
  a plain function or method read from an object is a `functools.partial`, which the
  SDK refuses itself (`function_tool` raises). Register a plain function that calls it;
  the gate judges that function's parameters, and what its body calls is not read.
- **A `weakref.proxy`**, as the tool's function or anywhere among the callables the gate
  reads. A proxy shows the attributes of the object it stands for and is not that
  object, so the gate cannot tell which callables a call to it runs. Register the
  function itself, or a plain function that calls it.
- **A sub-agent behind `agent.as_tool(...)` whose own tools are not gated.** Gate the
  sub-agent first, `gated_agent(sub, policy).as_tool(...)`, and the parent accepts it.
- **An agent tool whose `parameters` type is not a data-only Pydantic model.** The gate
  works out what the sub-agent is given with the SDK's adapter for that type, and the
  SDK then validates the arguments again itself, so the two agree only for a model the
  SDK's own rule for data-only validation admits (`can_prepare_without_user_code`) and
  that is written out with no code of its own. For a cause that lies in one field the
  refusal names the field and what puts it outside: its pydantic core type, a string
  pattern, what a list, a mapping or a union holds, a validator, or its default. For a
  cause on the type as a whole (a `model_validator`, a dataclass, a private attribute)
  it names that and no field. Refused, with what to declare in its place:
  - a `date`, `datetime`, `time` or `timedelta` field, a `UUID`, a `Path`, an `AnyUrl`,
    a `Decimal`, `bytes`: a `str` (or a `float` for the `Decimal`), read inside the
    sub-agent's own tools;
  - a `tuple`, a `set`, a `frozenset`, a `Sequence[str]`: a `list`;
  - an `Enum`: a `Literal` of its values;
  - a nested model, a model that refers to itself, a dataclass: flat fields;
  - a field with a pattern, a validator, a serializer, a computed field, or
    `Field(exclude_if=...)` (the SDK runs that callable again when it writes the input
    out): plain fields, checked inside the sub-agent's own tools;
  - a default that is not JSON data (`tags: list[str] = ()`) and a default factory other
    than `list` or `dict`: a JSON default, or `Field(default_factory=list)`;
  - a model with a `PrivateAttr`, and one built with `defer_build=True`: neither.

Refused call by call, before the policy runs, with a reason the model reads:

- arguments that are not a JSON object, or that do not validate against the tool's
  parameters (for an agent tool, against its `parameters` model);
- a non-finite number (`NaN`, `Infinity`, a literal too large for a float, or a string a
  `float` parameter reads as one). It is not JSON data: a policy cannot compare it and a
  record cannot fingerprint it;
- arguments given to a tool that takes none;
- a call to a tool whose function was changed after the tool was gated into one the
  list above refuses. The guardrail asks the same questions again on every call, and
  the SDK runs every guardrail of a model turn before any tool body of that turn. So a
  change made before a run, or between runs, is seen by the next call; a change a tool
  body makes during a run is seen from the next model turn, and not by the other calls
  of the turn it was made in (see "What the gate does not see");
- arguments that hold a value more than 64 levels below the arguments object. The limit
  is the gate's own and fixed: it is the same with and without an audit log, and does not
  move with the interpreter's recursion limit.

What the gate does not see:

- **What a wrapper or the tool body does with the arguments.** The gate judges the
  values the function is called with. A decorator that changes them in its body, and
  anything the tool reads from elsewhere, is not seen.
- **A function a wrapper holds some other way than `__wrapped__`.** The gate reads the
  function, what its `__wrapped__` chain leads to, a bound method's function and a
  callable object's `__call__`. A decorator that applies a second decorator inside itself
  and calls that one (the inner wrapper sits in a closure, and `functools.wraps` points
  past it), or a callable object that copies another function's `__wrapped__`, runs a
  wrapper the gate never read. A parameter that wrapper takes or adds is not judged.
  The same holds for a single-dispatch function: a wrapper that names a plain function
  as `__wrapped__` and calls a dispatcher it holds in a closure or a global is gated,
  and an implementation registered on that dispatcher runs with its own defaults. So is
  a plain function whose body calls a dispatcher.
- **A change a tool body makes to another tool's function, for the calls of the same
  model turn.** The SDK runs the guardrail of every call of a turn before any tool body
  of that turn. A tool body that re-wraps the function of another tool (sets its
  `__wrapped__` to a wrapper with a parameter of its own, say) is not seen by a call to
  that tool in the same turn, which runs the new wrapper unjudged. A call in the next
  model turn is refused.
- **Code of your callables that runs while a tool is checked.** To find the callables a
  call reaches, the gate reads `__wrapped__` the ordinary way, as Python and
  `inspect.signature` do, so a `__wrapped__` property or a `__getattr__` of a callable
  object runs when the tool is gated and again on every call. The tests that recognise a
  single-dispatch function, a partialmethod and a weak proxy read the object's type, its
  closure cells and its attributes with `inspect.getattr_static`, which runs no property
  and no `__getattr__` of the object; the `isinstance` tests among them read an object's
  `__class__`.
- **Model settings given to a run.** Only the agent's own `model_settings` are read when
  it is gated. `RunConfig(model_settings=...)` is merged in when a run starts, and an
  `extra_body` there is not seen.
- **Legacy `functions` definitions in `extra_body`.** A `functions` key there is sent in
  the body of the request and is not refused. It only describes functions to the model:
  the SDK runs a call by looking its name up among the agent's own tools, so a call to a
  name that is not one of them raises `ModelBehaviorError` and nothing runs, and a call
  to a name that is one passes through that tool's guardrail.
- **A call the SDK rejects before the guardrail runs.** For each of these the tool does
  not run and the call is not on the audit log. These raise inside the SDK and stop the
  run: a tool call id the model reuses for another call, within one response or in a
  later turn, and a blank tool call id (`ModelBehaviorError`); arguments that hold a lone
  surrogate character (`UnicodeEncodeError`); and arguments nested about a thousand
  levels deep or more (`RecursionError`, at a depth that depends on the interpreter's
  recursion limit). A tool the agent does not have raises `ModelBehaviorError` too,
  unless the run is given `RunConfig(tool_not_found_behavior="return_error_to_model")`,
  under which the model reads the SDK's "not found" error and the run goes on. A call
  the model repeats under the id and with the arguments of a call that already ran is
  not run a second time by the SDK, and is not judged a second time.
- **An error of the interpreter inside the guardrail.** Under a recursion limit too small
  for the guardrail's own walk over the arguments it raises `RecursionError`. The SDK
  then stops the run with an error, the tool does not run, and there is no entry.
- **A realtime session.** `gated_agent` refuses a `RealtimeAgent`. `gate_tools` accepts a
  list of function tools whatever they are attached to, but a realtime session runs the
  guardrail on a tool context of its own, so the arguments the gate pinned are not the
  ones that session executes, and this package has no test of a realtime session.

**Beyond the gate.** This gate judges the tool calls the framework routes through it;
the paths around it inside the framework are listed above. Like every Recusal gate, it
does not stop an agent that can reach a resource another way: a credential with wider
rights, a shell or network path outside the agent, or a file it can write that
configures the gate. Use it as one layer of several: least-privilege credentials for the
agent, a sandbox or container, limits on its network egress, and an audit anchor the
writer cannot reach ([the limits in
full](https://github.com/philpaz/recusal/blob/main/SECURITY.md#things-worth-knowing)).

## Status

**In-depth review under way.** Its fixes so far are released in 0.2.0, and the review continues. Recusal is a Beta (`0.x`) open-source project in active
development: it is not a finished or certified security product, and it is provided under
Apache-2.0 without warranty.
[What the reviews are, and where every adapter stands](https://github.com/philpaz/recusal#project-status).

**From the latest review round.** Round five found three gaps, all about
`functools.singledispatch`, and the 0.2.0 release has them: a single-dispatch tool with
nothing registered was accepted, so an implementation registered during a model turn ran
unjudged for a later call of that turn; a dispatcher that is a callable object's
`__call__` was not found under a `functools.wraps` wrapper; and a dispatcher was
recognised by one code object read at import. The three are closed after 0.2.0: every
single-dispatch function is refused as a tool (see the bullet on single dispatch above).
Round six, on those fixes, found one more gap, which the 0.2.0 release has too: a
`functools.partialmethod` hid the callable it holds, so a single-dispatch function or a
wrapper with a parameter of its own under it ran with a value no policy judged. It also
found that a `weakref.proxy` of a single-dispatch function was accepted, and that an
ordinary event bus was refused as a single-dispatch function. These are closed after
0.2.0 as well (see the bullets on `functools.partialmethod` and `weakref.proxy` above),
and no review round has been run on that change yet.
What is still not found is listed under "What the gate does not see".

**Help make it better.** Read the code, test this adapter against your own stack and tell
us what you find. Reviews, bug reports and pull requests are all welcome:
[how to contribute](https://github.com/philpaz/recusal/blob/main/CONTRIBUTING.md), [open issues](https://github.com/philpaz/recusal/issues).
Please report a security gap [privately](https://github.com/philpaz/recusal/security/advisories/new), not in a public
issue.

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
