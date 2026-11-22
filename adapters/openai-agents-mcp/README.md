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
MCP SDK requires `jsonschema` itself, and `referencing` comes with `jsonschema`). It is
tested on every pull request against exactly `openai-agents==0.24.0` with
`openai==3.49.0`, and weekly against the newest releases.

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
  too, each under its `http` or `https` address, with or without the trailing `#`),
  through every keyword that library validates: `anyOf`, `oneOf` and `allOf`, a `$ref`
  into the schema's own `$defs` or `definitions`, `items`, `prefixItems`,
  `additionalProperties`, `required`, `enum`, ranges and lengths. A `format` is not
  asserted. A `pattern` is checked only where Python can read it: with Python's `re`,
  whose reading is not always the server's (`$` also matches before a trailing newline),
  and not at all when Python cannot compile it (`\p{L}`, which JavaScript reads).
  JSON Schema validation does not convert, so a string is not a boolean or a number. The
  gate is stricter in two places where a server would still convert: `5.0` is not an
  `integer`, and an `enum` or `const` value must be of the same JSON type as the listed
  one (`1.0` and `true` are not `1`). A JSON string sent for a parameter the schema
  types as an object or a list is refused like any other value of another type; the MCP
  Python SDK's server would parse it into one.
- **refuses JSON text where a server may read it as another value.** The MCP Python
  SDK's server decodes the text of a string argument as JSON when the parameter is not
  annotated exactly `str`, and runs with what it decoded when that is null, a list or an
  object: `{"scope": "null"}` for `scope: Optional[str]` runs as `None`. The listed
  schema of such a parameter allows another JSON type than string, so for a property
  named in the schema's top-level `properties` whose schema does (an `anyOf` or a `type`
  list with `null` in it, a schema with no `type`), a string that is JSON text for null,
  a list or an object is refused: send the value as JSON, not as text. Text for a
  boolean that SDK keeps a string (`"true"` for `Optional[str]` runs as `"true"`), so
  the gate does not refuse it. Text for a boolean, a number or a string, and text that
  is not JSON, is judged and sent as the
  string it is, and a property whose schema allows only strings keeps every string.
- **writes every default the schema lists for a property the arguments leave out, at
  every depth.** The gate walks the arguments with the schema, through `$ref`,
  `properties`, `items`, `prefixItems` (in the older drafts a list of `items`, and
  `additionalItems`), `additionalProperties`, `allOf`, and the one alternative of an
  `anyOf` or `oneOf` the value is valid against, and adds each property that is left out
  and has a default. The default is the property's own `default`; when its schema lists
  none itself, it is the one listed by the schema its `$ref` names, by a member of its
  `allOf` or by an alternative of its `anyOf` or `oneOf` (zod publishes
  `z.boolean().default(true).nullable()` that way), where all that are listed must be
  the same value. A default that is an object is completed in turn. The completed
  arguments are validated again.

The policy judges the completed arguments, and the completed arguments are what is sent.

**One exception: a listed default the server would not run with if it were sent is
judged and not sent.** There are two kinds. One that is not valid against its own
property schema: Pydantic publishes `{"default": null, "type": "string"}` for a parameter
written `q: str = None`. The server runs with that default when the parameter is left
out, and would reject it if it were sent. And a top-level default that is itself JSON
text of the kind refused above, as `mode: Optional[str] = "null"` lists: the server runs
with the string `"null"` when the parameter is left out, and would decode it to `None`
if it were sent. In both cases the policy is given the default (`q: None`,
`mode: "null"`), the audit entry is bound to it, and the server is sent the arguments
without it.

**What the gate cannot check or complete, it refuses** before the policy runs, with a
reason the model reads. The reason names the place in the arguments (`opts.recursive`,
`items[1]`), written only from property names the schema lists and positions in a list (a
key the schema does not list is written `*`), and says what the schema requires there. It
carries no argument text. Refused:

- arguments that are not valid against the schema. A value of another JSON type is told
  the types the schema declares: send the declared type, and an object as an object, not
  as JSON text. A value outside an `enum`, a `required` property left out, a member that
  `additionalProperties: false` forbids and any other constraint each say so;
- a string that holds JSON text for null, a list or an object, for a property whose
  schema allows another JSON type than string (above);
- a schema that is not a JSON object, is not a valid JSON Schema, or declares a draft
  other than draft-07, 2019-09 and 2020-12;
- a `$ref` that does not resolve inside the listed schema itself. A remote reference is
  never fetched, and a reference to an `$anchor` is not resolved. `$dynamicRef`,
  `$recursiveRef` and a subschema with an `$id` of its own are refused too;
- a `default` listed where the gate does not read one, because it cannot tell whether
  the server would apply it: under a keyword the walk does not follow (`if`, `then`,
  `else`, `not`, `patternProperties`, `dependentSchemas`, `dependencies`, `contains`,
  `propertyNames`, `unevaluatedProperties`, `unevaluatedItems`, `contentSchema`), which
  is still validated, and under any keyword the gate does not know (a vendor's
  `x-shape`, a keyword of a later draft). `const`, `enum`, `examples`, `example` and
  `default` hold data and are not looked into;
- a property left out whose alternatives list different defaults
  (`{"anyOf": [{"type": "boolean", "default": true}, {"type": "string", "default":
  "no"}]}`), and a place two parts of the schema list different defaults for;
- an object or a list that more than one alternative of an `anyOf` or `oneOf` is valid
  for, when the alternatives would write different defaults;
- a list that leaves out an item a `prefixItems` entry lists a default for;
- a schema that leads back to itself through `$ref`, `allOf`, `anyOf`, `oneOf`, `not`,
  `if`, `then`, `else` or `dependentSchemas` alone, with no value in between
  (`{"$ref": "#"}`), or whose defaults never settle;
- members named by a `patternProperties` pattern Python cannot compile, because the gate
  cannot tell which members it governs;
- arguments that cost more to check than the gate allows for their size (below);
- arguments that are not JSON data (a non-finite number such as `1e400`), or that hold a
  value more than 64 levels below the arguments object.

An argument the schema neither names nor forbids is judged and sent as it is.

A `default` is never applied where a value is always present: on the schema of the
arguments themselves (the gate always sends an object), of the items of a list and of
`additionalProperties`. JSON cannot send a member or an item without a value.

**The work is bounded.** A recursive schema (`where: Union[And, Or, Leaf]`, a model with
`child: Optional[Node]`) is checked once for each value: nothing is validated twice for
the same schema and the same value while the arguments stand as they are, and the
defaults for one schema and one place in the arguments are worked out once. Every step
is counted against an allowance of 10,000 steps and 1,000 more for every value in the
arguments (an object, a list, and each member, item and scalar in them). The largest
ordinary calls measured spend under a twentieth of it; a call over it is refused before
the policy runs, with "the arguments are too costly to check against the server's
listed schema for this tool", and is one deny entry on an audit log.

**What the rules above mean for ordinary schemas:**

- A parameter typed as a `Union` of two ordinary Pydantic models (extra members ignored,
  every field with a default) lists two alternatives that both accept any object, so a
  call that leaves a defaulted field of either out is refused as matching more than one
  alternative with different defaults. A discriminated union (a `Literal` tag field,
  `oneOf`) is told apart by its tag and passes. Models with `extra="forbid"` are told
  apart by the members a call gives; an empty object still matches both.
- A `NamedTuple` or tuple parameter with a defaulted last member publishes a
  `prefixItems` entry with a default, so a call that leaves that item out is refused;
  send every item.
- A zod object publishes `additionalProperties: false` unless it is written with
  `.passthrough()`, so an argument its schema does not name is refused.
- OpenAPI's `nullable: true` is not JSON Schema: the gate reads the declared `type`, so
  `null` for `{"type": "string", "nullable": true}` is refused. A schema that means null
  lists it (`"type": ["string", "null"]`, or an `anyOf` with `{"type": "null"}`).
- A recursive model nested as deep as the 64-level limit on the arguments is completed.

To read the schema the gate also wraps the server's `list_tools`, in place. `list_tools`
returns what the server listed, unchanged, and the gate keeps each tool's schema from the
most recent listing: each listing replaces the one before, so a call to a tool the server
no longer lists is refused. A call to a tool that is not in the most recent listing makes
the gate call `list_tools()` itself, with no run context and no agent, once for that
call. A tool that is still not listed, and a server whose tools cannot be listed that
way, is refused.

A `tool_filter` hides tools from one agent; the schemas are the server's. The SDK's own
server classes keep the listing as it was fetched, before the filter (`cached_tools`),
and that whole listing is what the gate keeps. So two agents that share one server
object under a dynamic filter each reach the tools they were listed, and the gate's own
`list_tools()`, which such a filter cannot answer without an agent, still leaves it the
fetched listing. The gate does not enforce a filter: the SDK never routes a model's call
to a tool the filter hides, and a call your own code makes to one through
`server.call_tool` is judged by the policy like any other. A server class of your own
that keeps no such listing (`cached_tools` is `None`, the base class's answer) leaves
the gate with the tools of the latest listing as filtered for one agent: give each agent
its own server object there.

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
sent, completed with the listed defaults, and with any listed default that is not sent
(above). A call the gate refuses before the policy runs (arguments the schema does
not allow, JSON text a server may read as another value, arguments too costly to check,
a schema the gate cannot check, a tool the server does not list, arguments that are not
JSON data) is one entry too, a deny, bound to the arguments as the SDK handed them.
Arguments that are not JSON data have no fingerprint of their own, so that entry is
bound to the text the gate writes them as, under the key `input`: JSON without
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
  sent, and that it is valid against the listed schema. What a server's own code then
  makes of a value is not in the schema. Cases seen on servers built with the official
  SDKs (`mcp` 2.3.0, and the TypeScript SDK with zod):
  - **A parameter without an annotation.** The MCP Python SDK lists `type: string` for it
    and then decodes the string as JSON, so `"null"` runs as `None` and `"[]"` as a
    list. A schema that allows only strings keeps every string, so the gate does not
    refuse these. Annotate the parameter `str`, and the server keeps the string.
  - **A numeric string for a date.** A `format` is not asserted: `"0"` for a parameter
    typed `date` runs as 1970-01-01 and `"1700000000"` for a `datetime` as that
    timestamp.
  - **A validator or a constraint that changes the value.** A string constraint that
    strips and lowers (`StringConstraints(strip_whitespace=True, to_lower=True)`,
    `z.string().trim().toLowerCase()`) runs `"  /ETC "` as `"/etc"`. This also changes a
    default the gate wrote: Pydantic does not validate a default when the parameter is
    left out, so `name = "X "` runs as `"X "` without the gate, and as `"x"` with it,
    because the gate sent the default and the server validated what it was sent. The
    policy judged `"X "`.
  - **A fallback for a value that fails.** zod's `.catch(...)` is in no schema:
    `z.boolean().catch(undefined).or(z.null())` runs `null` as undefined.
- **An integer too large for a float parameter.** A JSON integer above 2^53 sent for a
  `number` parameter is judged as the exact integer; a server that reads the parameter as
  a float runs the nearest float.
- **Whether the server reads an argument its schema does not name.** Such an argument is
  judged and sent as it is, a string that holds JSON text included: the text check reads
  the properties the schema names, at the top of the arguments.
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
  reads the SDK's tool error. Arguments that hold a lone surrogate character raise
  `UnicodeEncodeError` inside the SDK and stop the run. Nothing runs and there is no
  entry.
- **Handoffs, an agent with a `prompt`, an agent whose `model_settings` put tools into the
  request, an ungated sub-agent behind `as_tool`, and `Agent` subclasses.** `gated_agent`
  raises for these at construction. A prompt stored on OpenAI's servers can configure
  tools, hosted MCP among them, that are not in the agent's `tools` list, and so can
  `model_settings.extra_body` or `extra_args` with a `tools`, `prompt`, `plugins`,
  `web_search_options` or `mcp_servers` key.
- **Model settings given to a run.** Only the agent's own `model_settings` are read when
  it is gated. `RunConfig(model_settings=...)` is merged in when a run starts, and an
  `extra_body` there is not seen.
- **Legacy `functions` definitions in `extra_body`.** A `functions` key there is sent in
  the body of the request and is not refused. It only describes functions to the model:
  the SDK runs a call by looking its name up among the agent's own tools, so a call to a
  name that is not one of them raises `ModelBehaviorError` and nothing runs, and a call
  to a name that is one passes through that tool's gate.

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
