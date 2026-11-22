<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
</p>

# recusal-crewai

[![Website](https://img.shields.io/badge/website-recusal.io-2453c4)](https://recusal.io)
[![PyPI](https://img.shields.io/pypi/v/recusal-crewai)](https://pypi.org/project/recusal-crewai/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/philpaz/recusal/badge)](https://scorecard.dev/viewer/?uri=github.com/philpaz/recusal)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

The [Recusal](https://github.com/philpaz/recusal) gate for
[CrewAI](https://docs.crewai.com/): every tool call CrewAI sends through its tool hooks is
judged by your policy before the tool runs, and a refused call never runs.

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
and a frozen compatibility perimeter; this package carries the CrewAI dependency and its
own version, so a CrewAI release never forces a release of the kernel.

Install it from PyPI:

```bash
pip install recusal-crewai
```

Requires Python 3.11+, `recusal>=0.14.0` and `crewai>=1.15.40`. It is
tested on every pull request against exactly `crewai==1.15.40`, and weekly against the
newest release.

The package ships inline types (`py.typed`) for your editor and type checker.

## Use

```python
from recusal import Finding
from recusal_crewai import install_recusal_hooks


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


uninstall = install_recusal_hooks(policy)  # after your other hooks
try:
    crew.kickoff()
finally:
    uninstall()
```

The gate is CrewAI's own tool-call hooks. The before hook judges the call and blocks a
refusal; the after hook replaces CrewAI's fixed "Tool execution blocked by hook" line with
Recusal's reason, so the model reads why on its next turn. A policy that raises refuses
the call (fail closed).

**The install is process-global.** CrewAI's hook registry covers every agent, task and
crew in the process, including a coworker an agent delegates to, and every tool kind
CrewAI runs through its hooks.

**Install the gate after your other hooks.** Hooks run in registration order, and a hook
registered after the gate can change the arguments the gate judged. A hook registered
for one execution only (CrewAI's `scoped_hooks`) runs after every process-wide hook,
the gate included, whatever the order: do not change tool arguments in one. A `@CrewBase`
crew registers its `@before_tool_call` and `@on(InterceptionPoint.PRE_TOOL_CALL)`
methods process-wide when the crew class is
instantiated, which is usually after the gate was installed: either install the gate
after instantiating the crew, or do not change tool arguments in those methods.

The policy judges the invocation the tool receives, not the dict the model proposed:
omitted defaults are completed from the tool's schema and values are validated once, the
same way CrewAI prepares them at execution.

**The name your policy is given is the one CrewAI calls the tool by**, which is the
declared name normalised: a tool declared `"Run SQL"` reaches the policy as `run_sql`.
Write rules against the normalised name.

**Give each tool a distinct name.** If two of the tools a call is chosen from normalise
to the same name, CrewAI itself does not keep them apart the same way on every route: it
can hand a hook one tool and run the other. The gate does not take the handed tool on
trust. It reads, in the CrewAI function that ran the hooks and is about to run the tool,
the tools the call is being chosen from and the function about to run, and it judges a
call only when the handed tool is the one that will run. Otherwise it refuses, before
your policy runs:

- on the text (ReAct) route, a call to a name two different tools there share (one tool
  listed twice counts as one);
- on the native routes, a call where the function CrewAI would run is not the handed
  tool's own, and a call to a name CrewAI made up for a renamed tool (`run_sql_2`): a
  rule written for `run_sql` would not match it;
- a call whose hooks CrewAI ran from a function this version of the package does not
  know. A CrewAI release that adds a route through the same hooks is refused there
  until the package supports it. A release that changes what a known function does is
  for the weekly test against the newest CrewAI to catch, not the gate at run time.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_crewai import install_audited_hooks

log = AuditLog(path="decisions.jsonl")
uninstall = install_audited_hooks(policy, log)
```

Each call the before hook sees, allowed or refused, is one hash-chained entry labeled
`crewai.before_tool_call` with a SHA-256 fingerprint of the arguments (never the
arguments). If the log cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

**Entries carry no call id.** CrewAI's hook context has no id for the call, so the record
does not invent one. Entries are told apart by their order on the chain, the tool, the
fingerprint, and `control.crewai_agent`, the role of the agent that made the call.
When a turn proposes several tool calls CrewAI may run them on a thread pool, so the
order of a turn's entries on the chain is not always the order the model proposed them.

## What the gate refuses, and what it does not see

- **Refused before the policy runs:** a tool or schema that is not CrewAI's stock one (a
  schema with a validator, an alias, a constraint, a `default_factory` other than `list`
  or `dict` itself, a default not of the exact JSON types, a metaclass of its own, a
  validator or serializer that is not Pydantic's, or an intermediate base; a schema whose
  validation would change the arguments again when CrewAI validates them before the tool
  runs; a plain `BaseTool` that takes `Tool.run`; a tool class with its own
  `__getattribute__` or `__getattr__`; a tool class that overrides CrewAI's run or validation methods, a `Tool`
  subclass with its own `_run`, a tool object with one of those methods set on it), an
  unsupported
  annotation, an unknown or missing parameter, a validation failure, a callable that
  declares a parameter its schema omits (including one behind a `functools.wraps`
  wrapper), and a call the gate cannot tie to the tool that
  will run (see "Give each tool a distinct name" above).
- **Not judged:** what a tool does once it runs. A tool that executes code or shell it is
  handed needs a policy that names it. That includes a decorator on the tool's function
  that does not use `functools.wraps`: the gate reads the decorator's own parameters, and
  a function it calls with values of its own, or with that function's defaults, is the
  tool's code. Use `functools.wraps`, and the wrapped function's parameters are held to
  the schema too. The gate holds a tool to CrewAI's and Pydantic's own machinery and
  refuses what it can tell is not stock; code written to show the gate one thing and do
  another is, like the tool's body, the tool's own code.
- **Annotations:** `str`, `int`, `float`, `bool`, optionals, and lists and string-keyed
  dicts of those. `Literal`, `Enum`, `Any`, `Annotated` and models are refused; use
  `str` and check the value in your policy. CrewAI builds an MCP tool's `enum`
  parameter as a `Literal`, so an MCP tool with one is refused on every call.
- **A hook that raises:** CrewAI runs the tool when a hook raises anything but its own
  abort. The gate catches every exception inside itself and blocks the call; an error
  Python raises before the gate's first line runs (a stack overflow at the call itself)
  is outside it.
- **Not seen: tools CrewAI runs without its hooks.** The gate is CrewAI's own
  `before_tool_call` hook, so it sees what CrewAI sends through that hook: an agent's
  tool calls in a crew, a coworker's, and CrewAI's own agent tools. It does not see a
  function passed to `LLM.call(available_functions=...)`, a tool action in a Flow, a tool
  run by one of CrewAI's agent adapters (its OpenAI Agents and LangGraph adapters call
  the tool themselves), a provider's built-in tool such as web search or the code
  interpreter, which runs on the provider's side, or a delegation to a remote agent over
  CrewAI's A2A support, which does not go through the tool hooks.
- **After the decision.** An entry records the gate's decision for one hook call, not how
  often the tool then ran or whether it ran. On CrewAI's text (ReAct) route a tool that
  raises is called again by CrewAI, up to six times in all on the tested release, under
  the one decision and the one entry: a tool with an effect before it raises repeats
  that effect; with the gate installed this includes a call with no arguments, which
  CrewAI then runs with the defaults the gate judged. A call CrewAI then
  declines itself (a tool at its usage limit, a repeated call on the text route) is on
  the log as allowed, and so is a call CrewAI answers from its tool cache without
  running the tool.
- **Not on the audit log:** a call a hook registered before the gate blocked, and whatever
  a hook registered after the gate changed.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_crewai.__all__`. When a CrewAI release changes what
the hooks are handed, this package gets a release and the kernel does not.

A runnable offline demonstration, with a scripted model and no API key, is
[`examples/crewai_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/crewai_gate.py)
and
[`examples/crewai_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/crewai_audit.py).

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

- [Framework notes for CrewAI](https://github.com/philpaz/recusal/blob/main/docs/FRAMEWORKS.md#crewai):
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
