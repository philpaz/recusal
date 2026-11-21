<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
    <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
  </picture>
</p>

# recusal-google-adk

[![PyPI](https://img.shields.io/pypi/v/recusal-google-adk)](https://pypi.org/project/recusal-google-adk/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
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

Requires Python 3.11+ (the ADK's minimum), `recusal>=0.14.0` and `google-adk>=2.14.0`. It
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

The gate prepares the invocation the way the ADK will (defaults completed, values
validated), judges a detached copy, and the tool runs with exactly what was judged.

For an app whose agents you do not build yourself, `RecusalPlugin(policy)` is the same
gate as a runner plugin, given to the app:
`InMemoryRunner(app=App(name="shop", root_agent=agent, plugins=[RecusalPlugin(policy)]))`
(`from google.adk.apps import App`). Plugin
callbacks run before any agent's own, so the plugin judges a call only when nothing it
cannot see runs after it. If the executing agent has another before-tool callback, or a
later plugin overrides `before_tool_callback`, it refuses the call and names the fix.

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
  whose tools it cannot judge.
- **Refused before the policy runs:** a `FunctionTool` whose class or object replaces any
  method on its run path (preparing, filtering, confirming or calling), or whose function
  was replaced after it was built; a signature that cannot be judged by name, including a
  `functools.wraps` wrapper that adds a parameter of its own; an unknown or missing
  parameter; arguments that do not validate, or that validating again would change.
- **Agent tools:** a call to a plain `AgentTool` is judged as the sub-agent receives it
  (the `request` text, or the arguments as JSON), and the sub-agent's own tool calls are
  judged by its own gate. An `AgentTool` whose agent declares an `input_schema`, an
  `AgentTool` subclass, and the tools ADK makes for sub-agents in `single_turn` or `task`
  mode are refused.
- **Not on the audit log:** a call another before-tool callback answered before the gate
  ran. The tool did not run either.

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
