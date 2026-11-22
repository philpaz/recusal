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

Requires Python 3.10+, `recusal>=0.14.0`, `langgraph>=1.2.20` and `langchain-core>=1.7.7`.
It is tested on every pull request against exactly `langgraph==1.2.20` with
`langchain-core==1.7.7`, and weekly against the newest releases.

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
are left out of judgment. An argument counts as the host's only when ToolNode really
replaced it for this call: a value the model sent that reaches the tool unchanged is
judged like any other, whatever the tool's annotations say.

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
a SHA-256 fingerprint of the arguments (never the arguments). If the log cannot be
written the call is refused. Check the log with `recusal audit verify decisions.jsonl`.

The audited gates call the plain gate's own judging function, so there is one decision
path.

## What the gate refuses, and what it does not see

- **Refused before the policy runs:** a tool whose class or instance overrides the
  standard execution lifecycle (it could run without reaching the gate), a
  `StructuredTool` subclass that replaces `_run` or `_arun` (it would execute something
  other than the function the gate inspects), input that does
  not validate against the tool's schema, and a tool whose callable declares a parameter
  the schema omits, since that is a value no proposal can carry and the policy would
  never see.
- **Refused, never run unjudged:** a tool body reached without the gate's judging step.
  The gate judges inside the tool's parse step; the copy of the tool it hands to
  execution runs its function only once per allowance, and only with the argument
  objects that step judged, so a framework release that changed the lifecycle would be
  refused, not run.
- **Not on the audit log:** a call another `wrap_tool_call` hook answered without
  reaching the gate.
- **Not seen:** a `ToolNode` you did not gate. `is_gated` reads one node; a graph can
  hold others, `create_react_agent(model, [tools])` builds its own ungated one (pass it
  your gated node instead), and a tool that calls another tool directly, for example
  through `runtime.tools`, bypasses every hook.

This package reads two private LangChain and LangGraph helpers to find the arguments the
host injects. That is why its framework requirements have an upper bound, and why the
weekly lane against the newest releases exists.

## Status

**In-depth review under way.** Its fixes are in [#298](https://github.com/philpaz/recusal/pull/298) and are not released yet. Recusal is a Beta (`0.x`) open-source project in active
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
