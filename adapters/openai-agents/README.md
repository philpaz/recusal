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

Requires Python 3.10+ (the SDK's minimum), `recusal>=0.17.0` and `openai-agents>=0.24.0`.
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
  as more than the gate reads.
- **A function that carries a declared signature.** A `__signature__` attribute,
  whatever its value, or a `__text_signature__` that is not the interpreter's own, on
  the function or on a callable on the way to it. `inspect.signature`, which the SDK
  reads a tool with, can read such a declaration in place of the parameters Python
  binds, and which ones it reads differs between Python versions. The kernel's walk
  relies on none. It refuses each one, also where this Python would not read it (a
  `__signature__` of `None`) and where it matches the real parameters, so a tool refused
  for this may run correctly. Remove the declaration, so that the parameters Python
  binds are what is read.
- **A decorator whose signature cannot be read.** A class-based decorator that is also a
  descriptor (`cachetools.cached`, `cachetools.func.ttl_cache`, `cachetools.func.lru_cache`)
  hides the functions under it from inspection. Register the function without that
  decorator, or use one the gate reads (a `functools.wraps` function decorator, or a
  class-based decorator with no `__get__` that calls `functools.update_wrapper(self,
  fn)`). A `staticmethod` object that something other than a class
  holds (`functools.partial(staticmethod(f))` under a decorator) is such an object too:
  Python shows no signature for it, so the tool is refused with the same reason,
  although it runs correctly. 0.2.1 beside `recusal` 0.15.1 refused it as well under an
  ordinary `functools.wraps` decorator, and gated it under a decorator written with
  `functools.update_wrapper` over a `functools.partial`. `functools.lru_cache` and `functools.cache` are read
  through and gated. With `functools.wraps(f)` written above the cache, the cache names
  `f` and calls the wrapper it was built over. The kernel's walk reads that wrapper: one
  that only forwards is gated, and one with a parameter of its own is refused. Since
  0.2.3 the gate's own walk follows both, what the cache names and what it calls
  (`recusal.invocation.cache_target`), so its own questions (a single-dispatch function,
  a `functools.partialmethod`, a weak proxy) are asked behind such a cache too. Refused
  although it runs correctly: such a cache over a `functools.partial` of a second
  `functools.lru_cache` wrapper or of a `staticmethod` object, for which Python shows no
  signature. Over a `staticmethod` object the reason is the one above; over a second
  cache it names only the exception class (`ValueError`), and 0.2.1 beside `recusal`
  0.15.1 gated both. A cache whose function cannot be told from what the cache refers
  to is refused; no interpreter this package is tested on gives one.
- **A decorator that holds a second wrapper of the function.** A decorator can apply
  another `functools.wraps` wrapper inside itself and call the result, while its own
  `functools.wraps` names the undecorated function. The wrapper in between is then not
  on the `__wrapped__` chain. The kernel's walk reads what a wrapper on the chain holds
  in its closure, in the module globals its code names, or, for a callable object, in
  its instance attributes. From a wrapper held there it goes on to the function: through
  plain functions and `functools.lru_cache` wrappers (to what each names as
  `__wrapped__`, and a cache named again to the function it calls), bound methods,
  `functools.partial` objects (to their function, and to a `__wrapped__` a partial
  carries), `staticmethod` and `classmethod` objects, class-based decorators and other
  callable objects (through what their instance dictionary holds), a `weakref.proxy` of
  a callable the walk stood on, and a second wrapper that is itself only held. This is
  the walk of `recusal` 0.17.0, which 0.2.3 requires; that of 0.16.0 followed fewer of
  these. It judges the way only if that ends at a function the walk read. The tool is
  then refused when a function on that way:
  - takes a parameter the tool does not declare;
  - passes the call on (`*args` or `**kwargs`) and takes a parameter the tool declares,
    which the function then does not receive;
  - takes, in a place an argument passed by position can fill, another name than the
    tool declares in that place (`def flipped(force, sql)` held over
    `def fetch(sql, force=False)`): the reason names the two parameters;
  - takes its leading parameter in the place of an argument bound on the way (the
    object of a method), where the tool declares that parameter and the wrapper passes
    the call on, or the tool does not declare it and it has a default;

  and when the way cannot be judged by parameters: it passes a single-dispatch function
  (whatever is registered on it), or a cache named again that calls anything but a
  plain function, or it does not end within 32 steps, or the held values lead to more
  than 16,384 callables. A `weakref.proxy` a wrapper holds of a callable the walk did not
  stand on (a helper) is refused too. The reason names the parameter in the first four
  cases (in the third, both) and names none in the others. A tool refused for one of the others may run
  correctly: see "Refused although it runs correctly" under Status. A held wrapper the
  walk does not find is not refused: see "What the gate does not see" below.
- **A wrapper that is handed a bound argument and may not pass it on.** A facade method
  written as `@functools.wraps(search)` over `def search(self, *args, **kwargs)`, which
  calls `search(self.tenant, *args, **kwargs)`, takes its object as `self`, where the
  function it wraps has another first parameter. Whether such a wrapper hands the bound
  argument on, keeps it or passes a value of its own in its place cannot be read from
  its signature, so the kernel's walk cannot tell which parameter of the function each
  judged value fills. A facade that takes its object in `*args` is refused the same way.
  Which signatures the kernel refuses is its rule (`BoundArgumentKept` in
  `recusal.invocation`): a parameter the bound arguments could leave open that nothing
  is known to fill with a judged value, where that parameter has a default, or where
  the tool declares it and a wrapper on the way takes that name itself. Where it has no
  default and the tool does not declare it, the walk of `recusal` 0.17.0 reads the
  facade, because a call by keyword would then fail; this SDK calls by position, so the
  gate refuses that shape itself (the next bullet). A wrapper of a refused signature
  that runs correctly is refused with it, because the two cannot be told apart. The
  reason carries the kernel's text, which names the parameter that could be left open
  and, where a wrapper took the bound argument under a name, that name too. That text
  is the kernel's and is quoted whole. In `recusal` 0.16.0 it could say a parameter
  "runs at a default no policy judged" where the parameter has no default; the text of
  0.17.0, which 0.2.3 requires, says only what holds for the parameter. Register the
  function itself; or write the method without `functools.wraps`, naming the
  parameters the tool takes in its own signature, and pass them on to the function by
  keyword.
- **A method, or a `functools.partial` with positional arguments, over a wrapper of a
  function not written to take what is bound (since 0.2.3).** A decorator written with
  `functools.wraps` that exposes a module function as a method,
  `def method(self, *args, **kwargs): return fn(*args, **kwargs)` over
  `def execute(database, sql="SELECT 1", note="")`, as `Warehouse().run_sql`:
  `inspect.signature` reads the method as `execute` and leaves `database` out as the
  object's place, and the SDK passes every declared parameter by position
  (`FuncSchema.to_call_args`), so the method's call drops the object and `execute` runs
  with each value one place off. 0.2.2 and 0.2.3 as first written gated it: the policy
  judged `sql="SELECT 1"` and `note="DELETE FROM audit_log"`, and the function ran
  `database="SELECT 1"` and `sql="DELETE FROM audit_log"`. The gate now refuses a method
  whose function is a wrapper (it names `__wrapped__`) of a function whose first
  parameter is not named `self` or `cls`, however the wrapper takes the object (as
  `self`, in `*args`, under the function's own first name, under a
  `functools.lru_cache`), and a `functools.partial` that binds a positional argument to
  such a wrapper in the place of a parameter so named. The rule is by name: where the
  dropped place is named `self` or `cls`, it is the ordinary decorated method, and a
  decorator that forgets to pass the object on is not found (stated under Open below).
  Refused although it runs correctly: one that does hand the object on, since the
  signature does not show it; a correct decorated method whose object's place has
  another name (`this`): name it `self`; and two such facades, one over the other,
  behind a cache named again for a function that shows the whole signature, where the
  SDK reads that signature and each drop is what makes the call correct. A decorated
  method of an ordinary class (`def run_sql(self, sql)`) is read as before. Register the
  function itself, not the method or the partial that wraps it.
- **A `functools.singledispatch` function or a `singledispatchmethod`, with or without
  an implementation registered.** The implementation that runs is chosen at call time by
  the type of an argument, each one has its own parameters and defaults, and one can be
  registered at any moment. The SDK checks every call of a model turn before any tool
  body runs, so an implementation that one call of a turn registers (or that the tool
  registers on its own first use) would run, unjudged, for a later call of that turn. So
  a single-dispatch function is refused when the agent is built, whether or not
  anything is registered, wherever the gate's own walk reaches it. Register one plain function as the tool, with the code of each implementation in its own
  body under a branch on the type (`if isinstance(...)`). Do not call a single-dispatch function from
  that body in its place: a single-dispatch function called from the body is not read,
  and the implementation it picks runs with its own defaults. The dispatcher is refused
  wherever it stands among the callables the gate reads: as the tool itself, as a
  method, a class method or a static method (`function_tool(service.handle)`), as a
  callable object's `__call__`, under `functools.wraps` wrappers or
  `functools.lru_cache`, behind a cache named again for another function (since 0.2.3),
  and as the callable a `functools.partialmethod` holds. It is
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
  attribute, and not through `__wrapped__`, is refused by the kernel's walk where it is
  a wrapper of the function the walk reads (`functools.singledispatch(f)` held by a
  decorator of `f`), whatever is registered on it. One built over another function, and
  one reached through a helper that names nothing as `__wrapped__`, are not found: see
  "What the gate does not see" below. A dispatcher a decorator holds is refused with
  the reason for a held wrapper, whose advice is to hold no single-dispatch function
  there or to register the function itself. The kernel's walk also refuses a
  single-dispatch function with nothing registered that it stands on
  (`OpenDispatch` in `recusal.invocation`), and the gate has a reason of its own for
  that: it runs its base now, an implementation registered after the check would run
  for the judged call, and the base function itself is what to register where nothing
  is to be registered. Since 0.2.3 the gate's own walk stands wherever the kernel's
  does and finds such a dispatcher first, so no tool is known that gets this reason.
- **A `functools.partialmethod`.** What the tool calls is the callable the partialmethod
  holds, and Python reads the tool's parameters from that callable without anything
  reading what is under it: a single-dispatch function or a wrapper with a parameter of
  its own there would run with a value no policy judged. So a function that is or
  carries a partialmethod is refused wherever the gate's own walk or the kernel's reaches
  it, whatever it holds, a plain function too. That is
  what you get when you read a partialmethod from the class (`Db.export`), and from an
  object (`db.export`) when the callable it holds does not bind as a method: a
  class-based decorator's instance, or a `functools.partial`. A partialmethod over
  a plain function or method read from an object is a `functools.partial`, which the
  SDK refuses itself (`function_tool` raises). Write the method out in its place: the
  work of the callable it holds in the method's own body, and the values the
  partialmethod binds as defaults of the method's own parameters (`def export(self, sql,
  force=False)` for `export = functools.partialmethod(run, force=False)`), with no call
  to that callable; and register that. A plain function that calls the partialmethod is gated with the cause
  still inside it: the gate reads its parameters and never its body.
  Behind a cache named again for another function the gate's own walk follows what the
  cache calls (since 0.2.3), and a partialmethod there is refused; where a
  `weakref.proxy` stands on the way it is refused as a proxy. 0.2.2 gated a
  partialmethod behind a proxy of a second cache named again where a
  `functools.partial` or a wrapper stood between that cache and it. A partialmethod
  that a plain function's body calls is not read, as nothing a body calls is.
- **A `weakref.proxy`**, as the tool's function or anywhere among the callables the
  gate's own walk reaches, behind a cache named again for another function too (since
  0.2.3). A proxy shows the attributes of the object it stands for and is not that
  object, so the gate cannot tell which callables a call to it runs. Put the
  object the proxy stands for in its place. A proxy of a plain function is
  refused although it runs correctly. A proxy a decorator holds, and does not reach
  through `__wrapped__`, is refused by the kernel's walk unless it stands for a callable
  that walk read (see the bullet on a decorator that holds a second wrapper).
- **A callable that answers for another object.** A class-based decorator that keeps the
  function in an attribute (`self.fn = fn`) and hands every other attribute on to it
  (`def __getattr__(self, name): return getattr(self.fn, name)`), over a function that
  names a `__wrapped__` of its own (any `functools.wraps` decorator, any
  `functools.lru_cache`), answers `__wrapped__` with what THAT function wraps, so a walk
  would step past the function it calls. The kernel's walk of `recusal` 0.17.0 refuses
  such an object (`ProxiedCallable`), and any other that answers `__wrapped__` without
  holding one or shows another class than its own (its `__class__` is not its type).
  The reason quotes the kernel's text, which says which was found. Such a decorator
  ordinarily runs correctly, and is refused all the same: call
  `functools.update_wrapper(self, fn)` in its `__init__`, and it is read like any other
  wrapper, so one that only forwards is gated. For another such object, register the
  function it calls itself.
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

- arguments that did not arrive as JSON text (a `dict`, a list, a number). The SDK
  hands a function tool's arguments as text; a seam another package builds on this one
  (the hosted approval callback of `recusal-openai-agents-mcp`) may be handed something
  else. It is not read as arguments decoded already: it is refused, and the reason
  names the type it arrived as and nothing it holds. The type is asked exactly: an
  instance of a subclass of `str` or `bytes` is refused too;
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
- **A function a wrapper holds some other way than `__wrapped__`, beyond the shapes that
  are refused.** The gate walks the function, what its `__wrapped__` chain leads to, a
  bound method's function, a callable object's `__call__` and the function a
  `functools.lru_cache` wrapper calls. A wrapper that a decorator holds in a closure, a
  global or an instance attribute is not walked by the gate itself. The kernel's walk
  refuses the shapes of it that the bullet "A decorator that holds a second wrapper of
  the function" above lists. Not found, and gated (a plain inner function, a wrapper
  built on first use, and one in a slot, a `types.SimpleNamespace`, a list, a function's
  own attribute, a partial's argument and behind a `weakref.ref` were each run through
  this gate beside `recusal` 0.17.0, with a held wrapper whose own `skip_review=True`
  ran at that default):
  - a second wrapper that names nothing as `__wrapped__` (a plain inner function, a
    lambda, a helper);
  - a wrapper built on first use (`if "held" not in state: state["held"] = guarded(f)`
    in the body of the wrapper that calls it): it does not exist when the tool is
    checked;
  - one held where no instance dictionary shows it: in a slot, in a C structure (a
    wrapper a C extension builds), or in a dictionary behind a `__dict__` property;
  - one held by an object that is not callable (a settings object, a
    `types.SimpleNamespace`), in a container, in a class attribute or in a function's
    own attribute;
  - a wrapper handed in as an argument a `functools.partial` binds
    (`functools.update_wrapper(functools.partial(helper, wrapper), f)`, where `helper`
    calls `wrapper`), and one behind a `weakref.ref` the wrapper calls to get it.

  A parameter such a wrapper takes or adds is not judged. The same holds for a
  single-dispatch function: a wrapper that names a plain function as `__wrapped__` and
  calls a dispatcher of another function that it holds in a closure or a global, or one
  it reaches through a helper that names nothing as `__wrapped__`, is gated, and an
  implementation registered on that dispatcher runs with its own defaults. So is a
  plain function whose body calls a dispatcher. The kernel's changelog entry for 0.17.0
  lists the rest of what its walk does not find, under "Still not found, by design".
- **A held wrapper whose leading parameter the kernel takes for an argument bound on
  the way, where it is not.** Under a method, the kernel counts the places of a wrapper
  a decorator holds by their names and does not read the decorator. A held wrapper with
  a required parameter of its own in the leading place (`def second(mode, sql="x",
  force=False)`), under a method decorator that keeps the object and calls
  `second(*args, **kwargs)`, is read as if `mode` were the object's place. It is gated,
  and the SDK's call hands the judged values on one place off: the policy judged
  `sql="S"` and `force=True`, and the function ran with `sql=True` and `force=False`.
  A decorator that hands the object on to a held wrapper whose leading place is named
  as the tool names a parameter is gated too; every call of it raises `TypeError` and
  the function does not run. The kernel states both, in 0.16.0 too, and 0.2.2 is the
  same.
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
  `__class__`. What a `functools.lru_cache` wrapper calls is read with the kernel's
  `cache_target`, which asks only the wrapper's own type: it lists what the cache refers
  to with `gc.get_referents` the first time it reads that cache, and calls nothing the
  cache holds.
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

**In-depth review under way.** Its fixes so far are released in 0.2.3, and the review continues. Recusal is a Beta (`0.x`) open-source project in active
development: it is not a finished or certified security product, and it is provided under
Apache-2.0 without warranty.
[What the reviews are, and where every adapter stands](https://github.com/philpaz/recusal#project-status).

**From the latest review rounds.** Rounds five and six found gaps that the 0.2.0 release
has and 0.2.1 closes: a `functools.singledispatch` tool with nothing registered was
accepted, so an implementation registered during a model turn ran unjudged for a later
call of that turn; a dispatcher behind a callable object's `__call__` was not found; a
dispatcher was recognised by one code object read at import; a `functools.partialmethod`
hid the callable it holds; a `weakref.proxy` of a single-dispatch function was accepted;
and an ordinary event bus was refused as a single-dispatch function. A single-dispatch
function, a partialmethod and a weak proxy are now refused as a tool wherever the gate's
own walk reaches one (see the bullets above). Up to 0.2.2 it did not reach one behind a
cache named again for another function; since 0.2.3 it does (below).

**From round seven, closed by the kernel.** Round seven found two stacks of decorators
that the kernel's walk in `recusal` 0.15.0 and 0.15.1 reads as a pass-through, so a
parameter of a wrapper under them ran with a value the policy was not shown. Both are in
0.2.0 and 0.2.1 installed beside either of those kernels. The walk of `recusal` 0.16.0
reads both, and 0.2.2 requires `recusal` 0.16.0:

- **A `functools.partial` that carries a `__wrapped__`.** A decorator written as
  `functools.update_wrapper(functools.partial(helper, f), f)` (or
  `functools.wraps(f)(functools.partial(helper, f))`) is followed to `f`, and a wrapper
  under it that adds a parameter of its own is refused. A wrapper the partial binds as
  an argument, other than the function it names, is not read (see "What the gate does
  not see").
- **`functools.wraps(f)` written above `functools.lru_cache`.** The cache names `f` and
  calls the wrapper it was built over. The kernel's walk reads that wrapper: one that
  only forwards is gated, and one with a parameter of its own is refused where the walk
  reaches it. Behind a `weakref.proxy` the walk of 0.16.0 did not reach it; 0.2.3
  refuses the proxy (below).

**What 0.2.2 changes.** Every refusal the kernel's walk makes by a class of its own
reaches you with a reason that says what was found and what to do. A bound argument the
walk cannot follow through a wrapper, a cache whose function cannot be told, and a
declared signature were each reported as "cannot be inspected" with the name of an
exception class. The reason for a decorator that holds a second wrapper said that the
wrapper takes a parameter of its own, which is not the cause of every such refusal.
Every reason that advises registering a plain function now says that the gate judges
that function's parameters and does not read what its body calls.

**What 0.2.3 changes.** It requires `recusal` 0.17.0, and closes the first three of the
four items 0.2.2 stated as open. Each was gated by 0.2.2 beside `recusal` 0.16.0 and ran
a value no policy judged:

- **Behind a cache named again for another function.** The gate's own walk followed
  what the cache names, so its own questions were not asked of what the cache calls,
  and the kernel's walk of 0.16.0 refused less there. A single-dispatch function with
  nothing registered was gated (an implementation registered after the check ran for
  the judged call, and the judged call itself could register it), and so was whatever
  stood behind a `weakref.proxy`: a wrapper with a parameter of its own, a dispatcher
  with an implementation registered before the tool was gated, and a second cache named
  again that calls a `functools.partialmethod` through a `functools.partial` or a
  wrapper. The gate's own walk now also follows what the cache calls, with the kernel's
  `cache_target`, and refuses the dispatcher and the proxy there with the reasons it
  gives the ones it finds elsewhere. The kernel's walk of 0.17.0 refuses them too.
- **A held wrapper that takes the tool's parameters in another order** is refused by the
  kernel's walk of 0.17.0, with a reason that names the two parameters. Beside 0.16.0 it
  was gated, and the function ran with the values swapped.
- **A held wrapper with a parameter of its own on a way 0.16.0 did not follow**, through
  a class-based decorator, a callable object, a `functools.partial` that carries a
  `__wrapped__`, a weak proxy, a cache named again or a second held wrapper, is refused
  by the kernel's walk of 0.17.0, with a reason that names the parameter.

The three refusals `recusal` 0.17.0 adds have reasons of their own: a single-dispatch
function with nothing registered (`OpenDispatch`), a callable that answers for another
object (`ProxiedCallable`), and a weak proxy (`WeakProxy`, which gets the reason the gate
gives a proxy its own walk finds). 0.2.2 beside 0.17.0 named the class alone ("the
function it runs cannot be inspected (WeakProxy)"). A cache whose function cannot be
told, met by the gate's own walk, is refused with the reason for one, not with the reason
for a callable the gate cannot question. The tests in `TestStatedLimits` hold each closed
item as refused, before any policy is asked and before anything runs.

The reviews of 0.2.3 before its release found more, answered here:

- **A method facade that drops its object**, over a function whose first parameter has
  no default, was gated and ran with every value one place off (the bullet on a method
  over a wrapper of a function not written to take what is bound, above). The kernel's
  walk of 0.17.0 reads it; the gate refuses it itself where the dropped place is not
  named `self` or `cls`, and the same shape through a `functools.partial`. Where the
  place is named `self` or `cls` the shape is a decorator that forgets the object, and
  it is open (below).
- **The advice of the reasons.** Followed literally, 11 of 15 reasons gave a gated tool
  that ran the cause: most advised calling the refused function from a plain function and
  registering that, and the gate reads a plain function's parameters and never its body.
  Every reason now advises removing the cause (drop a wrapper's own parameter or pass it
  explicitly, keep a held wrapper in the tool's order or keyword-only, register the
  function itself, call `functools.update_wrapper(self, fn)`, write the method out in
  place of a partialmethod with the held callable's work in its own body, branch on the
  type in one plain function's body), and ends by saying why the other way does not. A
  reason that quotes the kernel's text, whose own advice can be to call the refused
  function from a plain function, says first not to follow that part, and gives the
  kernel's text last. `TestReasonAdvice` follows each piece of advice and runs the
  result.
- **A false refusal**: a callable object whose class has a member named `partialmethod`,
  behind a cache named again, was refused as running a `functools.partialmethod`. That
  name is no name Python reads; it now counts only where the object itself holds it.

**Refused although it runs correctly.** The kernel's walk reads parameters and not
bodies, so it refuses some signatures a correct tool can have. Each is described in the
list of refused tools above.

- Since 0.2.2: a facade that is handed a bound argument and does pass it on; a declared
  signature that this Python would not read or that matches the real parameters; a
  decorator that holds a cache named again for the function over anything but a plain
  function; a cache named again over a `functools.partial` of a second cache or of a
  `staticmethod` object; and a `functools.partial` of a `staticmethod` object under a
  decorator. 0.2.1 beside `recusal` 0.15.1 gated the second and the third, and the
  fourth where the decorator is written with `functools.update_wrapper` over a
  `functools.partial` (under an ordinary `functools.wraps` decorator it refused that one
  too).
- New in 0.2.3, beside `recusal` 0.17.0. Each was gated by 0.2.2 beside `recusal`
  0.16.0, and each was run through this gate: it runs correctly and is refused.
  - A `weakref.proxy` of a plain function behind a cache named again for another
    function. Register the function itself.
  - A single-dispatch function that nothing is ever registered on, behind such a cache.
    Register the base function itself.
  - A class-based decorator that keeps the function in an attribute and hands every
    other attribute on with `__getattr__`, over a function that names a `__wrapped__`
    (any `functools.wraps` decorator, any `functools.lru_cache`). Call
    `functools.update_wrapper(self, fn)` in its `__init__`.
  - A decorator that holds a wrapper which takes the tool's parameters in another order
    and calls it by keyword or puts the values back. Give the held wrapper the tool's
    parameters in the tool's order, or by keyword only.
  - A decorator that holds a `weakref.proxy` of a helper. Hold the helper itself.
  - A decorator that holds a cache named again for another function over anything but
    a plain function (a `functools.partial`). Hold a cache of the function itself,
    or register the function itself.
  - A method, or a `functools.partial` with positional arguments, over a wrapper of a
    function whose first parameter is not named `self` or `cls`, where the wrapper does
    hand the bound argument on. Register the function itself. This one is refused by the
    gate's own rule, not the kernel's, as are the next two.
  - A correct decorated method whose object's place is named otherwise than `self`
    (`this`). Name it `self`.
  - Two method facades that drop their object, one over the other, behind a cache named
    again for a function that shows the whole signature: the SDK reads that signature,
    and each drop is what makes the call correct. Register the function itself.

  The kernel's entry for 0.17.0 in the changelog has the whole list for its walk, each
  with what to write instead; the rest of it (an object whose `__class__` is not its
  type, a subclass of `wrapt.CallableObjectProxy` whose `__call__` names the
  parameters, a chain of held callables 32 links deep, held values that lead to more
  than 16,384 callables) was not run through this gate.

**Open**, stated so you can judge them. Each is described under "What the gate does not
see" or above:

- Anything the kernel's walk raises for a callable it cannot read, other than by a class
  of its own, is refused with the name of the exception class alone, with no cause and
  no advice: "the function it runs cannot be inspected (ValueError)" for a cache named
  again over a `functools.partial` of a second cache, which runs correctly. This is the
  fourth item 0.2.2 stated, narrower now that the three classes of 0.17.0 have reasons.
- A held wrapper the kernel's walk does not find (one that names nothing as
  `__wrapped__`, one built on first use, one in a slot, a container, a settings object
  or a function's own attribute, one a `functools.partial` binds as an argument) is
  gated, and a parameter of its own runs at its default. The same holds for a
  single-dispatch function of another function that a wrapper holds or reaches through
  a helper, and for one a tool body calls: gated, and an implementation registered on it
  runs with its own defaults. The kernel states these as not found.
- A held wrapper with a required parameter of its own in the leading place, under a
  method decorator that keeps the object, is gated, and the function runs with the
  judged values one place off. The kernel states this as what its name rule reads. It
  is in 0.2.2 too, and this page did not state it before 0.2.3.
- **A method decorator that forgets to pass the object on.** Over a method of an ordinary
  class, `@functools.wraps(m) def method(self, *args, **kwargs): return m(*args, **kwargs)`
  over `def run_sql(self, sql, force=False)`, as `Tools().run_sql`. It is gated, and every
  judged value runs one place off: the policy judged `sql="S"` and `force=True`, and the
  method ran with `self="S"`, `sql=True` and `force=False`. The same holds for a class
  method (`cls`). From the signature the gate cannot tell whether a method decorator
  passes the object on: one that does is the ordinary decorated method, and only the
  body differs. The mistake is in the application's decorator, and the same call misruns
  without the gate. Pass `self` on in the decorator (`return m(self, *args, **kwargs)`),
  or register the method without that decorator. Found by the confirmation review of 0.2.3, in 0.2.2
  too; `test_open_a_method_decorator_that_forgets_the_object` holds it as gated and run
  one place off.
- A change a tool body makes to another tool's function is seen from the next model
  turn, not by the other calls of the same turn.
- A bounded `functools.lru_cache` named again over a callable that cannot be weakly
  referenced (an object with `__slots__`) is listed on every check, because the kernel
  cannot keep what it found: twice by the kernel's walk and, since 0.2.3, once by the
  gate's. On the review's machine a check of one with 400,000 entries took 52 ms, and 35 ms
  beside 0.2.2. Keeping the answer between checks would need a strong reference the
  kernel avoids on purpose.

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
