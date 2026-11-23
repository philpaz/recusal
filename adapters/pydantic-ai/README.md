<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
</p>

# recusal-pydantic-ai

[![Website](https://img.shields.io/badge/website-recusal.io-2453c4)](https://recusal.io)
[![PyPI](https://img.shields.io/pypi/v/recusal-pydantic-ai)](https://pypi.org/project/recusal-pydantic-ai/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/philpaz/recusal/badge)](https://scorecard.dev/viewer/?uri=github.com/philpaz/recusal)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

The [Recusal](https://github.com/philpaz/recusal) gate for
[Pydantic AI](https://ai.pydantic.dev/): every tool call an agent proposes is judged by
your policy before the tool runs, and a refused call never runs.

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
and a frozen compatibility perimeter; this package carries the Pydantic AI dependency and
its own version, so a Pydantic AI release never forces a release of the kernel.

Install it from PyPI:

```bash
pip install recusal-pydantic-ai
```

Requires Python 3.11+ (Pydantic AI's minimum), `recusal>=0.15.0` and
`pydantic-ai-slim>=2.67.0`. It is tested on every pull request against exactly
`pydantic-ai-slim==2.67.0`, and weekly against the newest release.

The package ships inline types (`py.typed`) for your editor and type checker. From 0.1.2
every function signature in it is annotated.

## Use

```python
from pydantic_ai import Agent
from recusal import Finding
from recusal_pydantic_ai import RecusalGate


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


agent = Agent(model, tools=[run_sql], capabilities=[RecusalGate(policy)])
```

A refused call is answered to the model as a failed tool result carrying Recusal's
reason. The tool does not run and no retry budget is spent. A policy that raises refuses
the call (fail closed).

`RecusalGate` declares Pydantic AI's `innermost` position, so its hooks run after every
ordinary capability's. List it directly and last. The gate reads the order the run
dispatches in, and a call is refused when a capability with a `before_tool_execute` of
yours comes after the gate, since it could change the arguments the gate judged. That
happens in two ways: another `innermost` capability listed after the gate, and a gate
placed inside a `WrapperCapability`, which does not pass the gate's position on. The
same holds for output functions and `before_output_process`. A hook bundled before the
gate, in the same list or behind the same wrapper, runs before it and is not refused.

**A decorated tool must be transparent.** Python fills a default in when a function is
called, after every hook, and a decorated tool is several functions: Pydantic AI builds
the tool's schema from the innermost one and calls the outermost. Its validation fills
in the innermost function's defaults, so the policy sees those. A wrapper on the way
that has a parameter of its own is read by nothing, and whether its value reaches the
function under it depends on the wrapper's body, which cannot be read. So all four forms
of the gate (the two capabilities and the two hook functions) read the stack from the
callable Pydantic AI calls down to the innermost function, as Python hands the call on
(a wrapper's `__wrapped__`, a bound method's function, a callable object's `__call__`,
the function of a `functools.partial`, whatever `__wrapped__` the partial itself
carries), and each wrapper on it must be
one of these (the wrappers under a `functools.partial` that binds keywords are held to
this twice, as the partial shows them and as they are written, because a partial with a
keyword shows every later parameter as keyword-only and would make a wrapper that takes
a parameter as another kind than the function look like one that restates it):

- it takes only `*args` and `**kwargs` (a logging, timing, tracing or retry decorator
  that passes everything through);
- it restates the function's own named parameters: the same names, kinds and order,
  each with no default or the function's own default. It has a fixed signature, or it
  also restates the function's `**extra` under the same name; the second only over a
  function that takes no `*args` and no positional-only parameter;
- it forwards `**kwargs` and also names parameters (`def wrapper(table, **kwargs)`),
  where a wrapper that kept one would fail the call, and where binding by name and by
  position cannot differ. Each parameter it names is of the same kind as in the function
  and has no default on either side; the positional ones are the function's first
  parameters, in the function's order; and the function takes no `*args` and no
  positional-only parameter, since Pydantic AI passes such a function's values by
  position. Such a wrapper may forward `*args` as well
  (`def wrapper(table, *args, **kwargs)`) only while no wrapper above it on the stack
  names more positional parameters than it does: one above could pass them on by
  position. A wrapper under it that names more does not count.

For a tool that takes the run context, a wrapper's first positional parameter is the
context, whatever the wrapper calls it (`def wrapper(context, *args, **kwargs)`), unless
it has the name of another parameter of the function. The context counts as the
function's first parameter in these rules, so a wrapper that names it and forwards is
refused over a function that takes `*args` or a positional-only parameter after the
context. A logging wrapper of that form above one that restates the function's
parameters is an ordinary pair and is not refused; under it, it is.

Anything else is refused before the policy runs, with a reason that says what the
wrapper does and names the parameter: a wrapper with a keyword of its own
(`def wrapper(*args, retries=3, **kwargs)`), a parameter it takes as another kind than
the function does (`def wrapper(*args, table, **kwargs)` over `def f(table, limit=10)`
takes `table` keyword-only), a default that differs from the function's, a parameter
left out or reordered. Give such a wrapper its setting when it is built
(`@retrying(times=3)`) and not as a parameter of the wrapper itself. The wrapper
`beartype` generates has keywords of its own and is refused by this rule: call the
type-checked function from a plain function and register that one. Also refused:

- **A function that declares a signature of its own** (`__signature__`): the parameters
  Python binds cannot be read. The decorators the `decorator` library builds do this,
  and so does Pydantic AI's own `tavily_search_tool` when it is given a search setting
  to fix (`tavily_search_tool(client=client, include_domains=[...])`). Build that tool
  without the fixed setting and hold the model to the value in your policy, which is
  given each search argument the model can set, or call the Tavily client from a plain
  function of your own that has the fixed value in its body.
- **A decorator that cannot be inspected**: a class-based decorator that is also a
  descriptor, such as `cachetools.cached`, a `functools.lru_cache` or `functools.cache`
  wrapper under a `functools.partial`, or a stack more than 31 decorators deep. Call the
  decorated function from a plain function and register that one.
- **A decorator that holds a second wrapper of the function**, beside a kernel later
  than `recusal` 0.15.0 (0.15.0 is the newest release as this is written). A decorator that applies another `functools.wraps` wrapper of
  the same function inside itself and calls that one, from its closure, by a module global it names or from a
  class-based decorator's instance, where the inner wrapper takes a parameter of its
  own: `functools.wraps` points past the inner wrapper, so it is not read like one
  applied directly. The reason names the parameter. Call the decorated function from a
  plain function and register that one. Beside 0.15.0 this is not detected, and the
  inner wrapper's parameter runs with a value the policy was not handed. An inner
  wrapper held any other way (a plain function or a lambda that names nothing as
  `__wrapped__`, a callable object, a container) is not found by either kernel.
- **A `functools.partialmethod` over a callable object**, beside a kernel later than
  `recusal` 0.15.0: refused, with a reason that says so and what to do. Beside 0.15.0 it is
  open, see "Status" below. Over a plain function the same `partialmethod` is read in
  full beside either kernel.
- **A single-dispatch function** (`functools.singledispatch`) with an implementation
  registered beside its base one, or in its place (`register(object)`): the
  implementation that runs is chosen at call time, by the type of an argument, so its
  parameters and defaults cannot be read. Call it from a plain function and register
  that one. With its base implementation alone it is read like any function, and a
  function that only has attributes called `registry` and `dispatch` is not taken for
  one. (Pydantic AI cannot run a `functools.singledispatchmethod` as a tool built from
  its signature. Given its schema by hand, one with a registered implementation is
  refused by this rule.)
- **A `functools.partial` that binds a value by position over a wrapper that names a
  parameter.** The partial's own signature, which is what Pydantic AI and the gate
  read, drops the leading parameter of every wrapper under it, and that parameter's
  default with it: `functools.partial(scoped(default_table(purge)), "acme")` with
  `def wrapper(tenant, *args, **kwargs)` over `def wrapper(table="archive", *, limit=100)`
  looks like a stack that names nothing, and ran `table="archive"` with the policy
  handed `{"limit": 5}`. The gate does not reason about what such a wrapper's body
  would do. Under a partial that binds by position there may be only wrappers that
  take nothing but `*args` and `**kwargs` (and other partials); a wrapper there that
  names any parameter of its own, of any kind and name, with or without a default, the
  run context included, refuses the call, whether or not a body that kept the value
  would fail. So `functools.partial(scoped(purge), "acme")` is refused for any
  `scoped` whose wrapper names `tenant`. Bind the value by keyword, or call the wrapped
  function from a plain function and register that one. A value bound by position
  directly over the function, or over wrappers that only forward, is read as the
  function's leading parameter and is not refused. Two more refusals of such a partial:
  one that was named with `functools.update_wrapper` or `functools.wraps` (see the next
  point), and one that binds a positional-only parameter of a function that also takes
  `**kwargs` (`functools.partial(fetch, "acme")` over `def fetch(tenant, /, table,
  **extra)`): an argument sent as `tenant` is not a duplicate to Python, it lands in
  `**extra` while the function runs the bound value.
- **A signature Pydantic AI reads that is not the function the call reaches.** Pydantic
  AI validates the arguments against `inspect.signature` of the callable it is given,
  and decides from it which values to pass by position. That reading follows
  `__wrapped__`; the gate reads what Python binds. For every tool built from its
  signature the two must be the same parameters (names, kinds, order and defaults), or
  the call is refused. The case that differs is a partial named the standard-library
  way, `functools.update_wrapper(functools.partial(insert, "acme"), insert)`: the
  signature is then the function's own, with the parameter the partial fills still in
  it, and `insert(tenant, *rows)` ran `tenant="acme"` with the model's `tenant` as its
  first row. A keyword partial named that way is refused too: Pydantic AI would show,
  and fill in, the function's default in place of the value the partial binds. Set
  `__name__` and `__qualname__` on the partial instead, or register a plain function.
- **The object of a bound method past a wrapper that names a positional parameter**
  (`def wrapper(self, *args, **kwargs)`). The object is read as filling the function's
  leading parameter, and the wrapper may keep it. Refused where a wrapper that kept it
  might not fail the call: that parameter of the function has a default; the function
  takes `*args` or a positional-only parameter after it; or the values after it can be
  passed on by position and move up a place, with a default to fill the gap they
  leave. The last is counted: a wrapper on the stack that names positional parameters
  beyond the object can pass that many on by position (the run context of a
  context-taking tool is one such value, and so is a value a partial binds), and the
  call is refused when the last parameter of the function they reach has a default. A
  `functools.lru_cache` or `functools.cache` wrapper keeps nothing and does not count
  as such a wrapper. So a method decorator that restates the method's parameters
  (`def wrapper(self, table, limit=10)`) over a method with a positional parameter that
  has a default is refused. That refusal is conservative: no wrapper body that passes
  the object on can misuse such a stack, and it is the price of not reading bodies. An
  ordinary method decorator (`def wrapper(self, *args, **kwargs)`) over a method whose
  other parameters are named and not positional-only is not refused.
- **A keyword a `functools.partial` binds that does not reach the parameter it names**:
  `functools.partial(fetch, limit=77)` over `def fetch(table, limit=10, /, **extra)`.
  Python does not bind a keyword to a positional-only parameter, so `**extra` takes
  the 77, `limit` runs on its own default, and the partial no longer shows `limit` at
  all. Refused when a keyword names a positional-only or variadic parameter of the
  function or of a wrapper under the partial.
- **A value for the function's `*args` that is not a list.** Pydantic AI hands the
  function its items one by one, so a string would reach it letter by letter.
  Validation makes the value a list; another one reaches the gate only where a hook of
  yours skips validation (`SkipToolValidation`) or answers a validation error.

**Defaults are judged.** A default of the innermost function that the arguments leave
out is added, judged, and passed to the function written out. For a tool built from its
signature Pydantic AI's validation has already done that. A tool given its schema by
hand (`Tool.from_schema`) is not validated, and a hook of yours can hand arguments in
in place of validation (`SkipToolValidation`), so there the gate completes whatever was
left out. A value a `functools.partial` binds by keyword is the default of the
parameter it names. A class used as the tool function is completed from the signature
Pydantic AI validates against, also behind a partial that binds keywords. Two more
refusals follow from it:

- **A default that is not plain data** (a logger, a client, a sentinel object) cannot be
  shown to a policy. For a tool given its schema by hand the refusal names the
  parameter. For a tool built from its signature Pydantic AI has already filled the
  default in, and the call is refused as carrying arguments that are not plain data.
  Give the parameter a JSON default or `None` and resolve the object inside the
  function. For a tool given its schema by hand, a positional-only parameter with a
  default that the model left out is refused too, since the default cannot be written
  out by name.
- **For a tool given its schema by hand, a function or a wrapper that takes `*args` or
  `**kwargs`**: it can fill a value in in its own body, which cannot be read. The
  refusal says whether it is the function or a wrapper around it. The tools
  `pydantic_ai.ext.langchain` builds are of this kind. Give the function named
  parameters (in a wrapper, restate them), or build the tool from its signature.

A toolset runs after every capability hook, so one that runs other tools' calls itself
could also change the arguments after the gate judged them. A run with one in its
toolsets has every call refused:

- a toolset of yours that holds other toolsets and has its own `call_tool` (a
  `WrapperToolset` subclass overriding it, or a container written from scratch, which
  would also hide what it holds). The gate looks for held toolsets in the toolset's
  attributes and four levels into the lists, tuples, sets, dicts and plain objects
  there, nearest first, and gives up after 4,000 values: a toolset with its own
  `call_tool` that it could not finish searching is refused as well. A toolset that
  only keeps a reference to another one, and never forwards to it, is refused by the
  same rule: keep such a reference outside the toolset;
- a subclass of a Pydantic AI toolset that replaces its `call_tool`;
- a toolset with a `process_tool_call` function, such as an MCP server given one.

The gate reads the run's toolset at the time of each call, so this holds for a wrapper
another capability adds and for one a dynamic toolset (`@agent.toolset`) returns. Not
refused: the toolsets Pydantic AI ships (prefixing, renaming, filtering, approval, tool
search, durable execution), a subclass of one that leaves `call_tool` alone, and a
toolset of your own that holds no other toolset: that one is the tool itself, and its
body is yours, like a tool function's.

The same gate as a plain hook is
`Hooks(before_tool_execute=recusal_before_tool_execute(policy))`. It judges tool calls
only, not output functions or calls handed to the application, and `Hooks` declares no
ordering: list it after every other capability, and use it only where no `innermost`
capability is added, since one runs after it wherever it is listed. Prefer `RecusalGate`.

**Output functions** (`output_type=fn`) run the application's function with values the
model chose, so `RecusalGate` judges them too, under the function's name, before they
run: a dict (a multi-argument function's arguments, or a single `dict` argument) is
judged as it is, any other single value as `{"output": value}`. A refusal asks the model
again; the function does not run, and when the output retries run out the run ends with
an error. Plain structured output has no output function and is not judged.

- The name is the function's `__name__`, and it is all the policy is given. A decorator
  that does not copy it (`functools.wraps` does) leaves its wrapper's name, so the
  function is judged as `wrapper`. A tool and an output function of one name are both
  judged under it: give an output function a name no tool has.
- Two output functions with the same name (`output_type=[review.submit, prod.submit]`)
  are refused, each time the model picks one: Pydantic AI gives each its own output
  tool, and the policy could not tell which of them it was judging. A text output
  function (`TextOutput`) is refused when an output tool runs a function of its name.
  Give each output function a name of its own. One function listed twice is one
  function.
- The data-only rule holds here too: an output function whose parameter is a Pydantic
  model, a dataclass or an Enum is refused. Declare plain fields (`str`, `int`, `list`,
  `dict`) and build the model inside the function.
- A union of output functions under `NativeOutput` or `PromptedOutput` is refused:
  Pydantic AI does not tell the hook which of them it will call. Use tool output (the
  default), where each function is its own output tool and is judged by name.
- A `Choices` output is judged as `choice` with `{"choice": key, "set": name}` when the
  pick is final, `name` being the set's (`Choices(..., name=...)`, `"Choices"` by
  default). Pydantic AI calls the picked value when it is callable, and the hook is
  handed the key and not the values, so a pick is judged with or without an action.
  Inside a `NativeOutput` or `PromptedOutput` union Pydantic AI does not call the action,
  and the pick is not judged.
- With `run_stream`, Pydantic AI runs an output function on partial output as it arrives;
  each of those calls is judged, so one answer can have several decisions.

**Calls handed to the application** (an `ExternalToolset`, a tool that raises
`CallDeferred`) are judged before they leave the run, as the model sent them, which is
what the application receives: no defaults are filled in and no types converted, so
write the policy for such a tool against the JSON. A refused call is answered with the
refusal, which the model reads; an allowed one is handed on with exactly the arguments
judged. A call waiting for approval is judged when it runs, with the arguments it runs
with, an approver's overrides included.

To answer deferred calls inside the run, give the handler to the gate:

```python
agent = Agent(model, toolsets=[external], capabilities=[RecusalGate(policy, deferred=handler)])
```

`handler(ctx, requests)` is the function `HandleDeferredToolCalls` takes. The gate judges
the calls first and calls it with the ones it allowed. Pydantic AI offers deferred calls
to each capability in turn and the gate comes last, so a handler in any other capability
(`HandleDeferredToolCalls`, a `Hooks` with a deferred hook, or one of your own) would be
handed calls no policy has judged: a run with one is refused with a `UserError` that
says this, before the run starts, so also when a run is resumed from a history that ends
in a tool call.

**Tools the provider runs** (web search, code execution, web fetch, a provider-side MCP
server and the like) run with arguments the model chose where no hook of your
application sees them. A gated agent that offers one is refused with a `UserError`
before the request is sent, as is one whose model settings switch such a tool on
(`openai_native_tools`, or `tools`, `plugins`, `web_search_options`, `mcp_servers` or
`search_parameters` in `extra_body`), on the run or on the model. The gate asks the
model what the request will carry, and each candidate of a `FallbackModel` in turn, so a
tool the model drops, or replaces with its local version (`WebSearch`, `WebFetch` and
`MCP` on a model without the provider side), is not refused; where the provider side
would be used, turn it off with `native=False`. Provider-side tool search is allowed: it
finds tools, and the calls to them are judged like any other.

**Realtime sessions** (`agent.realtime(...).session()`) are refused with a `UserError`
when they start: a session calls none of the request and output hooks the gate judges
in.

**The name the policy is given** is the one the model calls the tool by: under a
`PrefixedToolset` or `RenamedToolset` that is the prefixed or new name (`db_wipe` for a
tool `wipe` under the prefix `db`), so write the policy for the names you configured.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_pydantic_ai import RecusalAuditedGate

log = AuditLog(path="decisions.jsonl")
agent = Agent(model, tools=[run_sql], capabilities=[RecusalAuditedGate(policy, log=log)])
```

Each decision the gate makes, allowed or refused, is one hash-chained entry labeled with
the hook that made it: `pydantic_ai.before_tool_execute` for a tool call,
`pydantic_ai.before_output_process` for an output function or a `Choices` pick and
`pydantic_ai.handle_deferred_tool_calls` for a call handed to the application. Each
carries Pydantic AI's tool call id and a SHA-256 fingerprint of the arguments (never the
arguments). If the log cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

The audited gate calls the plain gate's own refusal function, so there is one decision
path.

## What the gate does not see

- **Capabilities an override puts in its place.** `agent.override(spec=...)` with a spec
  that names capabilities replaces the agent's own, the gate with them; `is_gated(agent)`
  reads the capabilities in effect, so it is `False` there.
- **A gate that is not loaded.** Pydantic AI runs no hook of a capability loaded on
  demand until it is loaded. `RecusalGate(..., defer_loading=True)` is refused when it is
  built, and a gate inside a group loaded on demand does not count for `is_gated`. List
  the gate directly.
- **`is_gated` reads the capabilities as listed.** For a gate a capability function
  returns (`lambda ctx: RecusalGate(...)`), or one inside a `WrapperCapability`, it says
  `False` although the gate runs.
- **A toolset of yours that reaches another toolset through a function.** One that
  builds or fetches the toolset it forwards to inside `call_tool` (a factory, a closure)
  holds nothing the gate can find, so it is taken for the tool itself: the policy judged
  the arguments it was handed, and what it does with them is its own code.
- **A WebRTC offer answered for the browser.** `answer_webrtc_offer()` hands the
  provider the agent's tool list without starting a run, so no hook runs and nothing is
  refused there. No tool of yours runs from it either: that takes a session, which is
  refused.
- **A provider tool switched on some other way.** The settings named above are the ones
  the gate knows. A model that searches or runs code by itself, or a setting not listed,
  is not seen.
- **Whoever reads the event stream.** Pydantic AI emits a tool call as an event before
  the call is judged. Code that acts on that event, instead of on the tool running or on
  the deferred request it is handed, acts before the gate.
- **A wrapper toolset under a run context with no tool manager.** The gate reads the
  run's toolset from `RunContext.tool_manager` at the time of each call. Where Pydantic AI
  provides none (its Temporal activity context), the gate relies on the check it makes
  when the run's toolsets are assembled, which does not see a wrapper another capability
  adds afterwards or one a dynamic toolset returns. It finds the function a tool runs,
  and a run's output functions, the same way, so there it completes no defaults, reads
  no wrappers and does not see two output functions of one name either.
- **What a function does in its own body.** A value a function sets itself, a wrapper
  that restates the function's parameters and then does not pass one on (the function
  runs on its own default for it, or on the value a partial under the wrapper binds;
  or, where the wrapper passes the others on by position, each of those reaches the
  parameter before its own), a function that calls
  a single-dispatch function, and a decorator that applies another decorator inside
  itself (the gate follows the function a wrapper says it wraps, `__wrapped__`, and
  that is not the one such a wrapper calls): the policy judged the arguments, and what
  the code does with them is its own.
- **What a class used as a tool function does when it is built.** The gate reads the
  signature Pydantic AI validates the class against and completes the arguments from
  it. It does not read `__init__`, `__new__` or a metaclass themselves, so none of
  these is seen: a decorator on one of them with a parameter of its own; a `__new__`
  that shows fewer parameters than `__init__` (`def __new__(cls, table, **kwargs)`
  beside `def __init__(self, table, force=False, **kwargs)` is judged without `force`); a
  declared `__signature__` that leaves a defaulted parameter of `__init__` out; a
  metaclass `__call__` that takes other parameters than `__init__` does.
- **A method whose decorators were written for free functions.** For a bound method the
  object is read as filling the leading parameter of the function, and the rules above
  assume each wrapper under it takes the object first, as `def wrapper(self, *args,
  **kwargs)` does. A method without `self` under decorators that take something else
  there is not read that way: `class Repo` with `@scoped @default_table def
  purge(table, *, limit=100)`, where `scoped` has `def wrapper(tenant, *args, **kwargs)`
  and `default_table` has `def wrapper(table="archive", *, limit=100)`, puts the object
  in `tenant` and runs `table="archive"`, the wrapper's own default, with the policy
  handed `{"limit": 5}`. So does a wrapper that hands the object on by keyword
  (`fn(*args, owner=self, **kwargs)`) above one whose leading parameter the model can
  then fill through the function's `**extra`. The same stacks behind a
  `functools.partial` that binds by position are refused.
- **An output validator.** A function registered with `@agent.output_validator` runs on
  the model's output, plain text and plain structured output included, and is not
  judged: the gate judges output functions. Behind an output function it is handed
  that function's result, and does not run when the function is refused.
- **A default applied somewhere else.** A default an MCP server or a toolset of your own
  applies to a call it receives, and a decorator on an output function: for output
  functions the gate judges the value and does not read the function's wrappers.
- **A delegated agent.** An agent called from inside a tool function runs under its own
  capabilities. Give every agent its own gate; `is_gated(agent)` reads only the agent it
  is given.
- **A call that fails validation.** Pydantic AI answers it with a retry prompt before any
  execute hook runs. The tool does not run either, and the call is not on the audit log.
- **Arguments that are not plain data.** JSON scalars, lists and string-keyed dicts are
  judged. Models, dataclasses, Enum members and other objects are refused before the
  policy runs. Declare a `str` or `int` and resolve application types inside the tool.

## Status

**In-depth review under way.** Its fixes so far are released in 0.2.0, and the review continues. Recusal is a Beta (`0.x`) open-source project in active
development: it is not a finished or certified security product, and it is provided under
Apache-2.0 without warranty.
[What the reviews are, and where every adapter stands](https://github.com/philpaz/recusal#project-status).

**Open from the latest review round**, stated so you can judge them:

- Beside `recusal` 0.15.0, a tool built with `functools.partialmethod` over a callable
  object (a class-based decorator instance, for example) is read one level deep: what
  is under the object is not read, so a value or a default can run that the policy was
  not handed. Over a plain function the same `partialmethod` is read in full. It is
  reachable with unannotated parameters or a tool given its schema by hand. The fix is
  in the kernel: a kernel later than 0.15.0 refuses such a tool before the policy
  runs, and 0.15.0 is the newest release as this is written. Beside 0.15.0, do not register a `partialmethod` over a callable object as a
  tool.
- A keyword that a `functools.partial` binds and that names no parameter of the function
  lands in its `**kwargs` and is not shown to the policy. It is the application's own
  constant, never the model's.

**Help make it better.** Read the code, test this adapter against your own stack and tell
us what you find. Reviews, bug reports and pull requests are all welcome:
[how to contribute](https://github.com/philpaz/recusal/blob/main/CONTRIBUTING.md), [open issues](https://github.com/philpaz/recusal/issues).
Please report a security gap [privately](https://github.com/philpaz/recusal/security/advisories/new), not in a public
issue.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_pydantic_ai.__all__`. When a Pydantic AI release
changes what the hook is handed, this package gets a release and the kernel does not.

A runnable offline demonstration, with a scripted model and no API key, is
[`examples/pydantic_ai_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/pydantic_ai_gate.py)
and
[`examples/pydantic_ai_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/pydantic_ai_audit.py).

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

- [Framework notes for Pydantic AI](https://github.com/philpaz/recusal/blob/main/docs/FRAMEWORKS.md#pydantic-ai):
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
