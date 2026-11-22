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

`RecusalGate` declares Pydantic AI's `innermost` position, so it judges the arguments as
they reach the tool, after every ordinary capability. The same gate as a plain hook is
`Hooks(before_tool_execute=recusal_before_tool_execute(policy))`; `Hooks` declares no
ordering, so that form must be listed last.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_pydantic_ai import RecusalAuditedGate

log = AuditLog(path="decisions.jsonl")
agent = Agent(model, tools=[run_sql], capabilities=[RecusalAuditedGate(policy, log=log)])
```

Each call the gate sees, allowed or refused, is one hash-chained entry labeled
`pydantic_ai.before_tool_execute` with Pydantic AI's tool call id and a SHA-256
fingerprint of the arguments (never the arguments). If the log cannot be written the call
is refused. Check the log with `recusal audit verify decisions.jsonl`.

The audited gate calls the plain gate's own refusal function, so there is one decision
path.

## What the gate does not see

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
