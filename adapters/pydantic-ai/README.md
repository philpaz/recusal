<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
</p>

# recusal-pydantic-ai

[![PyPI](https://img.shields.io/pypi/v/recusal-pydantic-ai)](https://pypi.org/project/recusal-pydantic-ai/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
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

Requires Python 3.11+ (Pydantic AI's minimum), `recusal>=0.14.0` and
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
ordinary capability's. Another `innermost` capability listed after it still runs after it,
so list the gate last. A toolset runs after every capability hook: one that wraps other
tools and runs their calls itself (a `WrapperToolset` subclass overriding `call_tool`, or
a toolset with a `process_tool_call` function, such as an MCP server given one) could
change the arguments after the gate judged them, so a run with one in its toolsets has
every call refused. The gate reads the run's toolset at the time of each call, so this
holds for a wrapper another capability adds and for one a dynamic toolset
(`@agent.toolset`) returns. The wrappers Pydantic AI ships (prefixing, renaming,
filtering, approval, tool search, durable execution) do not change arguments and are not
refused.

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
an error. Plain structured output, with no function, runs nothing and is not judged.

- The data-only rule holds here too: an output function whose parameter is a Pydantic
  model, a dataclass or an Enum is refused. Declare plain fields (`str`, `int`, `list`,
  `dict`) and build the model inside the function.
- A union of output functions under `NativeOutput` or `PromptedOutput` is refused:
  Pydantic AI does not tell the hook which of them it will call. Use tool output (the
  default), where each function is its own output tool and is judged by name.
- A `Choices` output is judged as `choice` with `{"choice": key}` when the pick is final.
  Pydantic AI calls the picked value when it is callable, and the hook is handed the key
  and not the values, so every `Choices` pick is judged, with or without an action.
- With `run_stream`, Pydantic AI runs an output function on partial output as it arrives;
  each of those calls is judged, so one answer can have several decisions.

**Calls handed to the application** (an `ExternalToolset`, a tool that raises
`CallDeferred`) are judged before they leave the run: a refused one is answered with the
refusal, which the model reads; an allowed one is handed on with exactly the arguments
judged. A call waiting for approval is judged when it runs.

Pydantic AI offers such calls to each capability in turn, and the gate comes last. So
where another capability answers them (`HandleDeferredToolCalls`, a `Hooks` with a
deferred hook, or one of your own), the gate judges every call as it is validated,
before any capability is offered it, as the model sent it. A call that then runs is
judged again before it runs, so in such a run it has two decisions.

**Tools the provider runs** (web search, code execution, web fetch, a provider-side MCP
server and the like) run with arguments the model chose where no hook of your
application sees them. A gated agent that offers one is refused with a `UserError`
before the request is sent. Provider-side tool search is allowed: it finds tools, and
the calls to them are judged like any other.

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
`pydantic_ai.before_output_process` for an output function or a `Choices` pick,
`pydantic_ai.handle_deferred_tool_calls` for a call handed to the application and
`pydantic_ai.after_tool_validate` for a call judged as it is validated. Each
carries Pydantic AI's tool call id and a SHA-256 fingerprint of the arguments (never the
arguments). If the log cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

The audited gate calls the plain gate's own refusal function, so there is one decision
path.

## What the gate does not see

- **Capabilities an override puts in its place.** `agent.override(spec=...)` with a spec
  that names capabilities replaces the agent's own, the gate with them; `is_gated(agent)`
  reads the capabilities in effect, so it is `False` there.
- **A wrapper toolset under a run context with no tool manager.** The gate reads the
  run's toolset from `RunContext.tool_manager` at the time of each call. Where Pydantic AI
  provides none (its Temporal activity context), the gate relies on the check it makes
  when the run's toolsets are assembled, which does not see a wrapper another capability
  adds afterwards or one a dynamic toolset returns.
- **A delegated agent.** An agent called from inside a tool function runs under its own
  capabilities. Give every agent its own gate; `is_gated(agent)` reads only the agent it
  is given.
- **A call that fails validation.** Pydantic AI answers it with a retry prompt before any
  execute hook runs. The tool does not run either, and the call is not on the audit log.
- **Arguments that are not plain data.** JSON scalars, lists and string-keyed dicts are
  judged. Models, dataclasses, Enum members and other objects are refused before the
  policy runs. Declare a `str` or `int` and resolve application types inside the tool.

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
