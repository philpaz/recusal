<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
</p>

# recusal-google-adk

[![Website](https://img.shields.io/badge/website-recusal.io-2453c4)](https://recusal.io)
[![PyPI](https://img.shields.io/pypi/v/recusal-google-adk)](https://pypi.org/project/recusal-google-adk/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/philpaz/recusal/badge)](https://scorecard.dev/viewer/?uri=github.com/philpaz/recusal)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

The [Recusal](https://github.com/philpaz/recusal) gate for
[Google ADK](https://google.github.io/adk-docs/): every tool call an agent proposes is
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
and a frozen compatibility perimeter; this package carries the ADK dependency and its own
version, so an ADK release never forces a release of the kernel.

Install it from PyPI:

```bash
pip install recusal-google-adk
```

Requires Python 3.11+ (the ADK's minimum), `recusal>=0.15.0` and `google-adk>=2.14.0`. It
is tested on every pull request against exactly `google-adk==2.14.0` with
`google-genai==2.34.0`, and weekly against the newest releases.

The package ships inline types (`py.typed`) for your editor and type checker.

## Use

```python
from google.adk.agents import LlmAgent
from recusal import Finding
from recusal_google_adk import gated_agent


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


agent = gated_agent(LlmAgent(name="assistant", model=model, tools=[run_sql]), policy)
```

`gated_agent` returns a copy of the agent with the gate appended as the last before-tool
callback, and gates its `sub_agents` and the agent behind every `AgentTool` the same way.
A refused call is answered to the model as `{"error": reason}`. The tool does not run. A
policy that raises refuses the call (fail closed).

The gate prepares the invocation (defaults completed, values validated against the
function's types whatever the ADK's own validation setting), judges a detached copy, and
the tool runs with exactly what was judged, apart from the parameters the ADK fills itself
(see below). A parameter whose default is `None` and that the call leaves out or sets to
`null` is judged as `None`, and runs as `None`: it is written into the call when a wrapper
on the way declares another default for it. A decorator whose own code fills or changes an
argument (`kwargs.setdefault(...)`) is the tool's code, which no gate reads. A value that
does not validate against its type (`"abc"` for an `int`) is refused, naming the parameter.
When a `None` has to be written into the call (a wrapper declares another default) and the
ADK's `FUNCTION_TOOL_ARG_VALIDATION` is on, a parameter not declared `Optional` is refused,
since the ADK would refuse the `None`. A wrapper with its own default over a
`pydantic.validate_call` function that rejects `None` fails the call; nothing runs.

For an app whose agents you do not build yourself, `RecusalPlugin(policy)` is the same
gate as a runner plugin, given to the app:
`InMemoryRunner(app=App(name="shop", root_agent=agent, plugins=[RecusalPlugin(policy)]))`
(`from google.adk.apps import App`). Plugin
callbacks run before any agent's own, so the plugin judges a call only when nothing it
cannot see runs after it. If the executing agent has another before-tool callback, or a
later plugin overrides `before_tool_callback`, it refuses the call and names the fix. A
plugin is handed tool calls only, so before a run starts it checks every agent in the
tree, from the root (a continued session can start at a sub-agent and transfer anywhere),
with the same rules `gated_agent` applies, and refuses the whole run when one of them can
execute something no callback sees: a `code_executor`, a provider-side tool, a toolset, a
`task` or `single_turn` sub-agent, or an agent class it does not know. It also refuses a
run whose `RunConfig` sets `support_cfc`, which gives the agent the model provider's code
execution.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_google_adk import audited_agent

log = AuditLog(path="decisions.jsonl")
agent = audited_agent(LlmAgent(name="assistant", model=model, tools=[run_sql]), policy, log)
```

`audited_plugin(policy, log)` is the plugin form. Each call a gate sees, allowed or
refused, is one hash-chained entry labeled `google_adk.before_tool_callback` with the
ADK's function call id and a SHA-256 fingerprint of the arguments (never the arguments).
If the log cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

The audited forms are the plain ones with a recording passed in, so there is one decision
path.

## What the gate refuses, and what it does not see

- **`gated_agent` raises at construction** (and `is_gated` is `False`) for an agent that
  can run something the callback never sees: an agent with a `code_executor` (model-written
  code runs outside the tool path), an `LlmAgent` subclass, an agent class it does not know
  how to gate, and any tool that is not a function, a `FunctionTool` or a plain
  `AgentTool`. That includes the built-in tools the model provider runs on its own side
  (`google_search`, `url_context`, `VertexAiSearchTool` and the like), which never reach a
  callback, tools declared in `generate_content_config`, and toolsets (MCP, OpenAPI),
  whose tools it cannot judge. Two ADK tools that run locally are accepted:
  `preload_memory` (it only adds memory to the model request) and `load_artifacts`
  (whose calls the gate refuses, since it cannot judge them). An agent in `task` mode is
  refused: the gate cannot judge how it finishes its task.
- **`gated_agent` cannot see how you run the agent.** It is handed the agent, not the
  `RunConfig`: do not set `support_cfc`, which gives the agent the model provider's code
  execution. `RecusalPlugin` sees the run and refuses it.
- **Refused before the policy runs:** a `FunctionTool` whose class or object replaces any
  method on its run path (preparing, filtering, confirming or calling), or whose function
  was replaced after it was built; a signature that cannot be judged by name, including a
  `functools.wraps` wrapper that adds a parameter of its own or names one the function it
  wraps also takes; an unknown or missing
  parameter; arguments that do not validate, or that validating again would change.
- **Agent tools:** a call to a plain `AgentTool` is judged as the sub-agent receives it
  (the `request` text, or the arguments as JSON), and only when the sub-agent's own tool
  calls are judged too: it is gated (`gated_agent` gates it), or it inherits
  `RecusalPlugin` (`include_plugins=True`, the default, with the plugin on the app).
  Otherwise the call is refused. An `AgentTool` whose agent declares an `input_schema`, an
  `AgentTool` subclass, and the tools ADK makes for sub-agents in `single_turn` or `task`
  mode are refused.
- **Not judged: an agent's answer.** The gate judges tool calls, not what an agent says.
  A structured answer (`output_schema`) is the agent's output either way it arrives: the
  `set_model_response` call the ADK adds when the agent has tools, which runs nothing and
  passes without the policy, or the final text, which no callback sees. Judge whatever
  acts on the answer (the next agent's tool calls are judged by their own gate).
- **Not judged:** the parameters the ADK fills itself, `tool_context` and `input_stream`.
  A call that sets either is refused; outside a live run an `input_stream` parameter takes
  its own default, which is the tool's code.
- **Not on the audit log:** a call another before-tool callback answered before the gate
  ran (the tool did not run either), and a `set_model_response` call, which is the agent's
  answer and passes without the policy.
- **Provider options on the model:** a model object that passes its provider an option
  turning on the provider's own tool, `LiteLlm(..., web_search_options=...)`, also when
  nested (`extra_body`), or an OpenRouter web plugin in its options, is refused, in a
  `FallbackModel` too. Other provider options and model names are configuration the gate
  does not read, and a model that searches on its own is the model's choice.

## Status

**Security reviewed.** The fixes from the review are released in 0.2.0. Recusal is a Beta (`0.x`) open-source project in active
development: it is not a finished or certified security product, and it is provided under
Apache-2.0 without warranty.
[What "security reviewed" means, and where every adapter stands](https://github.com/philpaz/recusal#project-status).

**Help make it better.** Read the code, test this adapter against your own stack and tell
us what you find. Reviews, bug reports and pull requests are all welcome:
[how to contribute](https://github.com/philpaz/recusal/blob/main/CONTRIBUTING.md), [open issues](https://github.com/philpaz/recusal/issues).
Please report a security gap [privately](https://github.com/philpaz/recusal/security/advisories/new), not in a public
issue.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_google_adk.__all__`. When an ADK release changes
what the callback is handed, this package gets a release and the kernel does not.

A runnable offline demonstration, with a scripted model and no API key, is
[`examples/google_adk_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/google_adk_gate.py)
and
[`examples/google_adk_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/google_adk_audit.py).

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

- [Framework notes for Google ADK](https://github.com/philpaz/recusal/blob/main/docs/FRAMEWORKS.md#google-adk):
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
