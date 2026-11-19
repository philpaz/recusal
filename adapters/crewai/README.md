<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
    <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
  </picture>
</p>

# recusal-crewai

[![PyPI](https://img.shields.io/pypi/v/recusal-crewai)](https://pypi.org/project/recusal-crewai/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

The [Recusal](https://github.com/philpaz/recusal) gate for
[CrewAI](https://docs.crewai.com/): every tool call a crew makes is judged by your policy
before the tool runs, and a refused call never runs.

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

Requires Python 3.10+ (CrewAI's minimum), `recusal>=0.14.0` and `crewai>=1.15.34`. It is
tested on every pull request against exactly `crewai==1.15.34`, and weekly against the
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
registered after the gate can change the arguments the gate judged.

The policy judges the invocation the tool receives, not the dict the model proposed:
omitted defaults are completed from the tool's schema and values are validated once, the
same way CrewAI prepares them at execution.

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
CrewAI runs one turn's tool calls on a thread pool, so the order of a turn's entries on
the chain is not the order the model proposed them.

## What the gate refuses, and what it does not see

- **Refused before the policy runs:** a tool or schema that is not CrewAI's stock one (a
  schema with a validator, an alias, a constraint, a `default_factory` or an intermediate
  base; a tool class that overrides CrewAI's run or validation methods), an unsupported
  annotation, an unknown or missing parameter, a validation failure, and a callable that
  declares a parameter its schema omits.
- **Not judged:** what a tool does once it runs. A tool that executes code or shell it is
  handed needs a policy that names it.
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
