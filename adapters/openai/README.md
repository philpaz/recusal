# recusal-openai

The [Recusal](https://github.com/philpaz/recusal) gate for a loop you own on the
[OpenAI API](https://platform.openai.com/docs/api-reference): every function call the
model proposes is judged by your policy before your code runs it, and a refused call
never runs. It covers the Responses API and Chat Completions with one decision function.

This is a separate distribution from `recusal`. The kernel has zero runtime dependencies
and a frozen compatibility perimeter; this package carries the OpenAI SDK dependency and
its own version, so an SDK release never forces a release of the kernel.

**Status: not yet published to PyPI.** Until it is, install it from the repository:

```bash
pip install "recusal-openai @ git+https://github.com/philpaz/recusal#subdirectory=adapters/openai"
```

Requires Python 3.10+ (the SDK's minimum), `recusal>=0.14.0` and `openai>=3.34.0`. It is
tested on every pull request against exactly `openai==3.34.0`, and weekly against the
newest release.

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
