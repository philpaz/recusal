<p align="center">
  <a href="https://recusal.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip.png">
      <img alt="recusal, deterministic governance for AI agents" src="https://raw.githubusercontent.com/philpaz/recusal/main/assets/banner-gate-strip-light.png" width="880">
    </picture>
  </a>
</p>

# recusal-openai-agents-mcp

[![PyPI](https://img.shields.io/pypi/v/recusal-openai-agents-mcp)](https://pypi.org/project/recusal-openai-agents-mcp/)
[![CI](https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)](https://github.com/philpaz/recusal/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

The [Recusal](https://github.com/philpaz/recusal) gate for MCP tools on an
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) agent: every call to a
local MCP server, and every hosted MCP call OpenAI asks approval for, is judged by your
policy first.

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
and a frozen compatibility perimeter; this package carries the SDK and MCP dependencies
and its own version, so a release of either never forces a release of the kernel.

It builds on
[`recusal-openai-agents`](https://github.com/philpaz/recusal/tree/main/adapters/openai-agents),
the function-tool gate, and depends on that package.

Install it from PyPI. `recusal-openai-agents`, which this package requires, comes with it:

```bash
pip install recusal-openai-agents-mcp
```

Requires Python 3.10+, `recusal>=0.15.0`, `recusal-openai-agents>=0.2.0`,
`openai-agents>=0.24.0`, `mcp>=2.3.0`, `jsonschema>=4.26.0` and `referencing>=0.37.0` (the
MCP SDK already depends on the last two). It is tested on every pull request against
exactly `openai-agents==0.24.0` with `openai==3.49.0`, and weekly against the newest
releases.

The package ships inline types (`py.typed`) for your editor and type checker.

## Use

```python
from agents import Agent
from recusal import Finding
from recusal_openai_agents_mcp import gated_agent


def policy(tool_name, tool_input):
    sql = tool_input.get("sql", "")
    if tool_name == "run_sql" and sql.lstrip().upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


agent = gated_agent(Agent(name="assistant", tools=[...], mcp_servers=[server]), policy)
```

`gated_agent` gates all three kinds of tool at once: function tools (through
`recusal-openai-agents`), every server in `mcp_servers`, and every `HostedMCPTool`.
`gate_mcp_server(server, policy)` and `gate_hosted_mcp(tool, policy)` gate one of each.

**Local MCP servers are gated in place, at `server.call_tool`.** That is the one point
where the SDK hands the server its own tool name and the arguments that are sent to it,
after any approval a person gave. A refusal returns an MCP error result carrying
Recusal's reason, and the server is never reached. The policy sees the name the server
executes under, whatever public name the model called it by, so a policy keyed by tool
name must name the server's own tools.

**The policy judges what is sent, checked against and completed from the server's listed
schema.** A server does two things with what it receives that the gate cannot watch: it
fills what the call leaves out from its defaults, and it converts a value of another type
(the string `"false"` to a boolean, `"500"` to a number). The tool's listed `inputSchema`
is the only statement of both, so before the policy runs the gate:

- **requires the arguments to be valid against the whole listed schema, as they stand.**
  They are validated with [`jsonschema`](https://pypi.org/project/jsonschema/), in the
  draft the schema declares (2020-12 when it declares none; draft-07 and 2019-09 are read
  too), through every keyword that library validates: `anyOf`, `oneOf` and `allOf`, a
  `$ref` into the schema's own `$defs` or `definitions`, `items`, `prefixItems`,
  `additionalProperties`, `required`, `enum`, ranges, lengths and patterns. A `format` is
  not asserted. JSON Schema validation does not convert, so a string is not a boolean or
  a number. The gate is stricter in two places where a server would still convert: `5.0`
  is not an `integer`, and an `enum` or `const` value must be of the same JSON type as
  the listed one (`1.0` and `true` are not `1`). A JSON string sent for a parameter the
  schema types as an object or a list is refused like any other value of another type;
  the MCP Python SDK's server would parse it into one.
- **writes every default the schema lists for a property the arguments leave out, at
  every depth.** The gate walks the arguments with the schema, through `$ref`,
  `properties`, `items`, `prefixItems` (in the older drafts a list of `items`, and
  `additionalItems`), `additionalProperties`, `allOf`, and the one alternative of an
  `anyOf` or `oneOf` the value is valid against, and adds each property that is left out
  and has a `default`. A default that is an object is completed in turn. The completed
  arguments are validated again.

The policy judges the completed arguments, and the completed arguments are what is sent.

**One exception: a listed default that is not valid against its own property schema is
judged and not sent.** Pydantic publishes `{"default": null, "type": "string"}` for a
parameter written `q: str = None`. The server runs with that default when the parameter
is left out, and would reject it if it were sent. So the policy is given `q: None`, the
audit entry is bound to it, and the server is sent the arguments without `q`.

**What the gate cannot check or complete, it refuses** before the policy runs, with a
reason the model reads. The reason names the place in the arguments (`opts.recursive`,
`items[1]`), written only from property names the schema lists and positions in a list (a
key the schema does not list is written `*`), and says what the schema requires there. It
carries no argument text. Refused:

- arguments that are not valid against the schema. A value of another JSON type is told
  the types the schema declares: send the declared type, and an object as an object, not
  as JSON text. A value outside an `enum`, a `required` property left out, a member that
  `additionalProperties: false` forbids and any other constraint each say so;
- a schema that is not a JSON object, is not a valid JSON Schema, or declares a draft
  other than draft-07, 2019-09 and 2020-12;
- a `$ref` that does not resolve inside the listed schema itself. A remote reference is
  never fetched, and a reference to an `$anchor` is not resolved. `$dynamicRef`,
  `$recursiveRef` and a subschema with an `$id` of its own are refused too;
- a `default` listed under a keyword the walk does not follow (`if`, `then`, `else`,
  `not`, `patternProperties`, `dependentSchemas`, `dependencies`, `contains`,
  `propertyNames`, `unevaluatedProperties`, `unevaluatedItems`), because the gate cannot
  tell whether the server would apply it. Those keywords are still validated;
- an object or a list that more than one alternative of an `anyOf` or `oneOf` is valid
  for, when the alternatives would write different defaults;
- a list that leaves out an item a `prefixItems` entry lists a default for;
- a schema whose references lead back to themselves without end, or whose defaults never
  settle;
- arguments that are not JSON data (a non-finite number such as `1e400`), or that hold a
  value more than 64 levels below the arguments object.

An argument the schema neither names nor forbids is judged and sent as it is.

To read the schema the gate also wraps the server's `list_tools`, in place. `list_tools`
returns what the server listed, unchanged, and the gate keeps each tool's schema from the
most recent listing: each listing replaces the one before, so a call to a tool the server
no longer lists is refused. A call to a tool that is not in the most recent listing makes
the gate call `list_tools()` itself, with no run context and no agent, once for that
call. A tool that is still not listed, and a server whose tools cannot be listed that
way, is refused.

**A server carries one policy.** The gate is installed on the server object, so every
agent that lists that object shares it. Gating it again with the same policy is a no-op;
gating it with a different policy raises `ValueError`. Use a separate server object per
policy.

**Hosted MCP is approval-gated, not execution-gated.** Those calls run on OpenAI's
servers. `gate_hosted_mcp` returns a copy whose `require_approval` is `"always"` and
whose approval callback is Recusal's: a refusal answers `{"approve": False, "reason":
...}`, so the hosted call is not made. A hosted tool whose `require_approval` is anything
but `"always"` is reported as ungated. The policy sees the arguments as the model sent
them: the hosted server's schema is not read, so the check above does not apply.

**Build the agent completely, then gate it.** A tool or a server appended to the gated
agent afterwards is not gated, and `is_gated(agent)` then returns `False`. The gated copy
has its own `tools`, `mcp_servers` and `handoffs` lists, so a later change to the
original agent's lists does not reach it. The servers themselves are shared with the
original agent and stay gated for both. A sub-agent behind `as_tool` is accepted when it
is gated itself with this package's `gated_agent`, its own servers and hosted tools
included.

## Put every decision on the record

```python
from recusal.audit import AuditLog
from recusal_openai_agents_mcp import audited_agent

log = AuditLog(path="decisions.jsonl")
agent = audited_agent(Agent(name="assistant", mcp_servers=[server]), policy, log)
```

One log covers the three seams, each under its own surface label:

| Seam | Surface | Call id on the entry |
|---|---|---|
| Function tools | `openai_agents.tool_input_guardrail` | the SDK's tool call id |
| Local MCP server | `openai_agents.mcp.call_tool` | none: the SDK hands `call_tool` no id; the server is `control.mcp_server` |
| Hosted MCP | `openai_agents.hosted_mcp.approval` | the approval request id; the server label is `control.mcp_server` |

Each entry holds a SHA-256 fingerprint of the arguments, never the arguments. For a local
server's judged call that is the fingerprint of the arguments as judged: the arguments
sent, completed with the listed defaults, and with any listed default that could not be
sent (above). A call the gate refuses before the policy runs (arguments the schema does
not allow, a schema the gate cannot check, a tool the server does not list, arguments
that are not JSON data) is one entry too, a deny, bound to the arguments as the SDK
handed them. Arguments that are not JSON data have no fingerprint of their own, so that
entry is bound to the text the gate writes them as, under the key `input`: JSON without
spaces, a non-finite number as `nan`, `inf` or `-inf`, an object that is not data as its
type's name in angle brackets, and `...` for anything more than 64 levels down. If the
log cannot be written the call is refused. Check the log with
`recusal audit verify decisions.jsonl`.

## What the gate does not see

- **A parameter the call leaves out that has no listed default.** It is the server's to
  fill, and the schema does not say with what. Most optional parameters of hand-written
  schemas are like this, and so is every Pydantic `default_factory` default, which is not
  published. A policy that depends on such a parameter should refuse a call that leaves
  it out.
- **A server whose code does more than its schema says.** The gate guarantees what is
  sent, and that it is valid against the listed schema. Two cases in the MCP Python SDK's
  server (`mcp` 2.3.0): it lists `type: string` for a parameter that has no annotation and
  then parses the string, so `"null"` runs as `None` and `"[]"` as a list; and for a
  parameter whose schema allows a string beside an object or a list (`Union[dict, str]`)
  it parses JSON text into the object. Both pass the gate as the strings they are.
- **An integer too large for a float parameter.** A JSON integer above 2^53 sent for a
  `number` parameter is judged as the exact integer; a server that reads the parameter as
  a float runs the nearest float.
- **Whether the server reads an argument its schema does not name.** Such an argument is
  judged and sent as it is.
- **A schema that changes after it was listed.** The gate checks against the schema of
  the most recent listing made through the gated server object. A server that changes a
  tool between that listing and the call, and a listing the SDK serves from its cache
  (`cache_tools_list=True`), leave the gate with the older schema.
- **Request metadata.** What a server's `tool_meta_resolver` returns is sent as `_meta`
  beside the arguments. It is not given to the policy and is not in the fingerprint.
- **Hosted MCP: the server's defaults and type conversion.** The hosted server's schema
  is not read, so the policy sees the arguments as the model sent them. Which of two
  duplicate keys in the argument text the hosted server reads is not seen either: the
  policy is given the last one, as `json.loads` reads it.
- **A hosted call OpenAI does not ask approval for.** That is why the gate sets
  `require_approval` to `"always"`.
- **A hosted approval the SDK answers from its own memory.** The gate keeps no state
  between turns. When the provider repeats the id and the arguments of an approval
  request the gate already allowed, the SDK approves it again without asking the
  callback, so there are two approvals for one consultation of the policy and one entry.
  A policy that counts calls counts consultations, not approvals.
- **What the tool's own approval callback answered** after Recusal allowed a call. The
  entry records Recusal's decision, not that callback's.
- **A call the SDK rejects before `call_tool`.** Arguments for a local MCP tool that are
  malformed JSON, or not a JSON object, are rejected by the SDK itself, and the model
  reads the SDK's tool error. Nothing runs and there is no entry.
- **Handoffs, an agent with a `prompt`, an agent whose `model_settings` put tools into the
  request, an ungated sub-agent behind `as_tool`, and `Agent` subclasses.** `gated_agent`
  raises for these at construction. A prompt stored on OpenAI's servers can configure
  tools, hosted MCP among them, that are not in the agent's `tools` list, and so can
  `model_settings.extra_body` or `extra_args` with a `tools`, `prompt`, `plugins`,
  `web_search_options` or `mcp_servers` key.
- **Model settings given to a run.** Only the agent's own `model_settings` are read when
  it is gated. `RunConfig(model_settings=...)` is merged in when a run starts, and an
  `extra_body` there is not seen.

## Compatibility

This package is outside the kernel's frozen perimeter (`STABILITY.md`, Integrations). Its
public names are the ones in `recusal_openai_agents_mcp.__all__`. It uses only the public
names of `recusal-openai-agents`.

A runnable offline demonstration, with an in-process MCP server and no API key, is
[`examples/openai_agents_mcp_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_agents_mcp_gate.py)
and
[`examples/openai_agents_mcp_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_agents_mcp_audit.py).

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

- [Framework notes for MCP tools on the OpenAI Agents SDK](https://github.com/philpaz/recusal/blob/main/docs/FRAMEWORKS.md#openai-agents-sdk-mcp-tools):
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
