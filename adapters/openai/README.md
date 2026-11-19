<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
    <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
  </picture>
</p>

# recusal-openai

[![PyPI](https://img.shields.io/pypi/v/recusal-openai)](https://pypi.org/project/recusal-openai/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

The [Recusal](https://github.com/philpaz/recusal) gate for a loop you own on the
[OpenAI API](https://platform.openai.com/docs/api-reference): every function call the
model proposes is judged by your policy before your code runs it, and a refused call
never runs. It covers the Responses API and Chat Completions with one decision function.

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
and a frozen compatibility perimeter; this package carries the OpenAI SDK dependency and
its own version, so an SDK release never forces a release of the kernel.

Install it from PyPI:

```bash
pip install recusal-openai
```

Requires Python 3.10+ (the SDK's minimum), `recusal>=0.14.0` and `openai>=3.34.0`. It is
tested on every pull request against exactly `openai==3.34.0`, and weekly against the
newest release.

The package ships inline types (`py.typed`) for your editor and type checker.

## Use

```python
from openai import OpenAI
from recusal import Finding
from recusal_openai import gate_function_calls


def run_sql(sql: str) -> str: ...


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


client = OpenAI()
response = client.responses.create(model=..., input=..., tools=...)
answers = gate_function_calls(response, {"run_sql": run_sql}, policy)
# pass ``answers`` as the next request's input
```

`gate_tool_calls(completion, tools, policy)` does the same for Chat Completions and
returns `role: "tool"` messages. A completion requested with `n > 1` carries alternative
messages, so pass `choice=<list position>`; without it the gate refuses the completion
before anything runs.

Each proposed call gets exactly one answer under its own id: the tool's result, or
Recusal's reason. A policy that raises refuses the call (fail closed).

The gate binds the decoded arguments to the tool's signature with its defaults applied,
judges that invocation, and calls the tool with exactly those arguments. A default the
model omitted is therefore judged at the value the function would use.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_openai import audited_function_calls

log = AuditLog(path="decisions.jsonl")
answers = audited_function_calls(response, tools, policy, log)
```

`audited_tool_calls` is the Chat Completions form. Each call the model proposes, run or
refused, is one hash-chained entry labeled `openai.responses.function_call` or
`openai.chat_completions.tool_call` with the API's own call id and a SHA-256 fingerprint
of the arguments (never the arguments). If the log cannot be written the call is refused.
Check the log with `recusal audit verify decisions.jsonl`.

The audited functions call the plain ones, so there is one decision path.

## What the gate refuses, and what it does not see

- **Refused with a reason, never run:** arguments that are not valid JSON or not a JSON
  object, a tool the application does not have, arguments that do not bind to the tool,
  and a tool whose signature cannot be judged by name (variadic or positional-only
  parameters, an unsupported annotation, a default that is not JSON data).
- **Refused by raising, before anything in the turn runs:** an output item or tool call
  of a type the gate does not handle (`custom_tool_call`, a hosted tool call, a type a
  later SDK adds), a call without an id, and a duplicate id. A turn refused this way
  writes no audit entry.
- **Not seen at all:** hosted tools (web search, file search, code interpreter, image
  generation, remote MCP) run on the provider's side and never reach your loop.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_openai.__all__`. When an SDK release changes the
response types the gate reads, this package gets a release and the kernel does not.

A runnable offline demonstration, with SDK-validated responses and no API key, is
[`examples/openai_responses_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_responses_gate.py)
and
[`examples/openai_responses_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_responses_audit.py).

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

- [Framework notes for the OpenAI API](https://github.com/philpaz/recusal/blob/main/docs/FRAMEWORKS.md#openai-api-your-own-loop):
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
