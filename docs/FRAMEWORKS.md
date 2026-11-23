# Framework gates in detail

The README lists the agent frameworks Recusal gates and links here. This page is the
detail for each one: where the gate attaches, what it hands the policy, what it refuses
before the policy runs, what its hook cannot see, and which versions it is tested with.
Each framework's gate is an installable package under [`adapters/`](../adapters/README.md),
and each has a runnable offline example under [`examples/`](../examples/README.md).

The same three things hold for every gate here, and are stated once in the README:
a policy that raises refuses the call, every gate can put its decisions on a hash-chained
audit log through the same code path, and a gate judges the call the model proposes, not
what a tool does once it runs.

## LangGraph

Package: [`recusal-langgraph`](../adapters/langgraph/README.md).

For a concrete LangGraph integration, [`examples/langgraph_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/langgraph_gate.py)
puts existing `@tool` functions unchanged into LangGraph's `ToolNode` and attaches
`recusal_gate(policy)` with `wrap_tool_call`, so `recusal.claude_code.decide` runs immediately
before each individual tool call. The policy judges the invocation the tool executes:
execution gets a per-call copy of the tool whose parse step validates the input once,
after ToolNode has injected the host's state, store and runtime, holds every callable the
route reaches to the schema (a parameter that the function, or a `functools.wraps` decorator
around it, declares and the schema omits, and that neither the host nor the framework fills,
refuses the call: Python would fill it after judgment and the policy would never see the
value that runs; so does a parameter a forwarding decorator names that the function it wraps
names too; and so does an innermost function that takes `*args` or `**kwargs` without naming
every schema field, or that takes nothing else besides parameters that never count as its
own (one LangChain or ToolNode fills, one the tool marks injected, one named `run_manager`,
`callbacks`, `config` or `runtime` whatever its annotation, and one whose annotation, resolved
with `typing.get_type_hints` and opened through the forms the package README lists, is or is
built from `RunnableConfig`, `ToolRuntime`, an injected-argument marker or a callback manager
or run manager type),
which hands arguments on to code the gate cannot read: a decorator built without
`functools.wraps` that names no parameter of its own (`def wrapper(*args, **kwargs)`, under a
JSON schema dict or the schema inferred from it; one that names a parameter and then calls
the function is not read, like any function that calls another), an `_arun(*args, **kwargs)`
that hands the call to `_run`, and a function that takes only `**kwargs`; such a forwarder
runs only under an explicit Pydantic schema that declares no field and does not allow extras,
through which the model can send nothing; the other exception is the function
`langchain-mcp-adapters` defines for its tools, recognised by its code object, which hands
the judged arguments to the MCP server as they stand, so those tools are judged and run on
the async routes unless they were built with tool interceptors, and a default the server
applies to an argument the model left out is not seen), completes any schema parameter the framework's parse dropped (an `exclude=True`
field, a root serializer) from the default that would run, projects every model-controlled
argument field by field through a projection that runs no serializer of its own, decides,
and passes those same completed objects to the function. Host arguments are not judged, and
an argument counts as the host's only when ToolNode put it in the tool's input for this
call, which it does for state, store and runtime only: a value under any other injected name
is judged, whoever wrote it, and a value the model sent that arrives unchanged is judged, whatever the tool's
annotations say, and so is a value the tool's own parse step fills under an injected name,
which is the tool call id (the model writes it) for an `InjectedToolCallId` parameter and
the schema default for an injected argument no host supplied. A suppressed parameter with no
default, a value the projection does not support, a tool whose class or instance overrides
the standard `invoke`/`run` lifecycle, a call whose name is not the name of the tool that
would run, and a call with an empty id, are refused before the policy runs. An answer that
does not fit what the judging copy's parse step last did for the call is withheld and refused
after the fact: one that comes back without that step having run, one that is not the gate's
refusal after that step refused, and one under another tool's name after it allowed. A tool
may have run by then. With a kernel that does not detect it, a decorator that applies another
decorator inside itself is not read. The package README lists every refusal and what to
do about it. A ToolNode's sync route consults only its sync hook slot,
so the gate goes on both slots: `gated_tool_node(tools, policy)`, or `ToolNode(tools,
**recusal_gates(policy))`, refuses the same way on `invoke`, `stream`, `ainvoke` and `astream`,
and `is_gated(node)` checks a node. The async hook cannot be built on its own unless the caller states `host_refuses_sync_runs=True`. It is offline and deterministic (no model or API key), requires
**Python 3.10+**, and is tested with exactly `langgraph==1.2.20` and
`langchain-core==1.7.7`. LangGraph remains an example-only dependency; the Recusal package
and its Python 3.9+ support are unchanged. To put every one of those calls on the record,
[`examples/langgraph_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/langgraph_audit.py) swaps in
`recusal_audited_gate(policy, log)` (or both hooks with `recusal_audited_gates(policy, log)`): the
same decision, written as a hash-chained `AuditLog` entry labeled `langgraph.toolnode` with
LangGraph's tool call id and a fingerprint of the arguments, before the call runs or is
refused, and an unwritable log refuses the call.

**Pause, resume and replay.** The gate decides at execution time, under the policy in force
at that moment, so a decision made before a pause never becomes permission that outlives
it. A review node that calls `interrupt()` before the `ToolNode` is judged once, on resume,
as the call will run: a reviewer's edit to a `DELETE` is refused, and a policy revoked or a
grant expired during the pause refuses the resumed call. A tool that calls `interrupt()`
itself is judged before the pause and judged again on resume, one execution. A replay or
fork from a checkpoint is a new execution and is judged again. One pattern is outside any
pre-tool gate: a value supplied on resume and consumed inside the tool, which the gate never
sees; route reviewer edits through a review node before the `ToolNode` instead. All of it is
pinned, with zero executions on every refusal and the limitation asserted on purpose, in
[`tests/test_langgraph_interrupt.py`](https://github.com/philpaz/recusal/blob/main/tests/test_langgraph_interrupt.py).

## LangChain `create_agent`

Package: [`recusal-langchain`](../adapters/langchain/README.md).

For LangChain's `create_agent` middleware, [`examples/langchain_agent_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/langchain_agent_gate.py)
reuses the **same** `recusal_gate(policy)` through `wrap_tool_call` without a second policy
implementation. It is offline and needs Python 3.10+ with `langgraph==1.2.20`,
`langchain-core==1.7.7` and `langchain==1.4.10` (example-only pins). List the gate **last**
in `middleware=`, so it judges the call exactly as it will run: the first middleware is the
outermost, and one listed after the gate acts between the gate and the tool. An argument
rewriter listed there, or a human-in-the-loop edit that changes only the arguments, still
reaches the gate's judging copy, so the rewritten call is judged as rewritten. A
human-in-the-loop edit that redirects the call to another tool does not: `HumanInTheLoopMiddleware`
swaps the tool object, the other tool runs unjudged, and so does any middleware that swaps
the tool or answers in its place (`LLMToolEmulator`). The gate sees that its judging step
never ran, withholds the answer and returns a refusal that says so, but a swapped-in tool
has run by then. A middleware listed there that answers a call the gate refused by running
another tool is withheld the same way, because the answer is not the gate's refusal, and so
is an answer under another tool's name after an allowed call; an answer under the judged
tool's name is passed on. Listed last, the gate judges whatever the others produced. One middleware
object serves `agent.invoke` and `agent.ainvoke`; `agate_middleware(policy)` is the same
middleware under the name earlier releases used for `ainvoke`. A tool given to
`create_agent` as a dict is run by the model provider and reaches no tool-call middleware,
so the gate refuses a model call that offers one: the model is not called.
To put every one of those calls on the record,
[`examples/langchain_agent_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/langchain_agent_audit.py)
swaps in `audited_gate_middleware(policy, log)`: the LangGraph audited gate as middleware,
for both routes, writing each decision as a hash-chained `AuditLog` entry labeled
`langchain.create_agent` with the tool call id. A middleware listed after it that answers in
the tool's place leaves that call unjudged; the entry written for it is the refusal saying
so, which is the same reason to list the gate last. A withheld answer to a call the gate had
already refused is a second entry for that call.

## OpenAI Agents SDK, function tools

Package: [`recusal-openai-agents`](../adapters/openai-agents/README.md).

For the OpenAI Agents SDK, [`examples/openai_agents_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_agents_gate.py)
attaches a tool input guardrail to each function tool through `gate_tools`, bound to that
tool (a standalone `recusal_guardrail(policy)` is bound to nothing and refuses every
call), which the SDK runs immediately before the tool executes. Function tools must have
unique names: the SDK resolves a shared name last-wins, and tools that share a name across
`tool_namespace` groups reach the policy under the same name, so `gate_tools`, `gated_agent`
and `is_gated` refuse an agent whose tools share one. The guardrail judges the invocation the SDK
prepares from the model's arguments, validated against the tool's parameter model with
defaults filled and types coerced (`{"amount": "50"}` reaches an `int` parameter as
`50`), each value under the name of the parameter that receives it (which parameter is
the context the SDK fills is read from the SDK's own schema for the tool), and pins that
invocation so the SDK executes exactly what was judged. For a sub-agent behind
`agent.as_tool(...)` the parent's policy judges the input the sub-agent is given: the
arguments as proposed for the default `as_tool`, and for `as_tool(parameters=Model)` the
arguments as the SDK's own adapter for the model validates them and writes them back out
(types converted, defaults filled, other keys dropped), with arguments the model does not
accept refused before the policy runs. A non-finite number (`NaN`,
`Infinity`) and arguments that hold a value more than 64 levels deep are refused before
the policy runs, the same way with or without an audit log. A tool no
call to which could be judged is refused when it is gated, by `gate_tools` and
`gated_agent`, with a `TypeError` naming the tool: a parameter model the SDK will not
prepare (one that runs a custom validator or holds an application type such as an Enum, a
model, a dataclass, a date, a `Sequence` or an `Iterable`, a default that is not JSON
data, a default factory other than `list` or `dict`), a mapping whose key type is neither
`str` nor a `Literal` of strings, a variadic parameter, a
`functools.wraps` wrapper at any level of a stack that
takes other parameters than the function (one of its own, some of the function's, or the
function's in another order), which would run with a value no
policy judged, a `functools.singledispatch` function or a `singledispatchmethod`, with
or without an implementation registered (as the tool itself, as a method, as a callable
object's `__call__`, or under a forwarding wrapper), a `functools.partialmethod` (read
from its class, or over a callable without `__get__`) and a `weakref.proxy`,
a decorator whose signature cannot be read (a class-based decorator that is also a
descriptor, such as `cachetools.cached`), a function under more wrappers than the walk
reads (more than 31), and an agent tool whose `parameters` type is not a data-only
Pydantic model (a dataclass, or a model with a validator, a serializer, a computed
field, `exclude_if` or a field that is not data-only; the refusal names the field and
its type); keep parameters data-only and validate inside the tool body. The
walk reads the function, its `__wrapped__` chain, a bound method's function and a callable
object's `__call__`; a wrapper that calls a function it holds in a closure is not read. A refusal
becomes the call's
output, so the model reads Recusal's reason and the tool never runs. Gate a finished agent with
`gated_agent(agent, policy)`: it returns a gated copy with its own `tools`, `handoffs` and
`mcp_servers` lists, and it refuses, rather than skips,
anything the guardrail cannot see. That means hosted tools and the SDK's other non-function
tools (shell, computer, apply patch and the rest), `mcp_servers`, handoffs, an agent with
a `prompt` (a stored prompt can configure tools of its own), an agent whose
`model_settings` carry a `tools`, `prompt`, `plugins`, `web_search_options` or
`mcp_servers` key in `extra_body` or `extra_args` (in `extra_body` a `tools` key replaces
the agent's own tool list in the request and the others are sent beside it), an
`agent.as_tool(...)` whose sub-agent is not itself gated, and `Agent` subclasses such as
`SandboxAgent`. `Runner.run`, `Runner.run_streamed` and calls a person approved through
`needs_approval` are all judged when the tool is about to run. It is offline (the SDK's own
`ScriptedModel`, no API key) and needs Python 3.10+ with `openai-agents==0.24.0` and
`openai==3.49.0` (example-only pins). To put every one of those calls on the record,
[`examples/openai_agents_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_agents_audit.py)
swaps in `audited_agent(agent, policy, log)` (or `audited_tools(tools, policy, log)`):
the same decision, written as a hash-chained `AuditLog` entry labeled
`openai_agents.tool_input_guardrail` with the SDK's tool call id and a fingerprint of the
prepared invocation, before the call runs or is refused, and an unwritable log refuses
the call. A call waiting for approval is recorded when it is about to run, so a call a
person rejected is not on the log; it did not run. Under the SDK's
`RunConfig(tool_execution=ToolExecutionConfig(pre_approval_tool_input_guardrails=True))`
the guardrail runs before the approval as well: a call then has an entry before a person
decides, a rejected call keeps that entry, and an approved call has two.

## OpenAI Agents SDK, MCP tools

Package: [`recusal-openai-agents-mcp`](../adapters/openai-agents-mcp/README.md).

For the SDK's MCP tools, [`examples/openai_agents_mcp_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_agents_mcp_gate.py)
covers the two paths that example names as out of its sight. A local MCP server
(`mcp_servers=`) has its tools listed and renamed for the model at run time, prefixed,
sanitized, hashed on collision or shortened, so a gate keyed on the public name judges a
presentation of the capability. `gate_mcp_server(server, policy)` instead wraps the
server's own `call_tool`, in place, the one point where the SDK hands over the server's
tool name and the arguments that are sent to it, after any approval. A server fills a
parameter the call leaves out from its default, converts a value of another type, and
where its schema offers several object forms picks one by a rule of its own. Its listed
`inputSchema` is the only statement of what it accepts and states none of those rules,
so the gate also wraps
`list_tools`, in place, to keep each tool's schema from the most recent listing (the
whole listing, before any `tool_filter`, where the server object keeps it as the SDK's
own servers do), and
before the policy runs the arguments must be valid against the whole of that schema as
they stand (validated with `jsonschema` in the draft the schema declares, through
`anyOf`, `$ref`, `additionalProperties`, `prefixItems` and the rest, and stricter where a
server would still convert: a string is not a boolean or a number, `5.0` is not an
integer; a server's `pattern` is never evaluated and is not checked), and every default
the schema lists for a property they leave out, at any depth,
is written into them: the property's own, or, when its schema lists none itself, the one
its `$ref`, a member of its `allOf` or an alternative of its `anyOf` or `oneOf` lists
(what zod publishes for `z.boolean().default(true).nullable()`). Two things a schema
cannot say are refused by rule. An object where the schema offers more than one object
form (two or more alternatives of an `anyOf` or `oneOf` that admit an object, which a
`Union` of two models and a zod union publish) is refused whatever it holds, because
Pydantic and zod each pick the form by a rule that is not JSON Schema's, unless every
form fixes one same-named property to a `const` of its own and the object carries the
value of exactly one; a list where it offers more than one list form is refused. And a
string is refused where the schema also admits the type a server reads its text as: the
text of a number or a word for a boolean where the schema names that type beside string
(`"1700000000"` for `Union[int, date]` runs as the integer), at any depth, and, for an
argument, JSON text for null, a list or an object where the schema admits another type
than string, because the MCP Python SDK's server decodes such text and runs
with the value (`"null"` for an `Optional[str]` runs as `None`).
Arguments the
gate cannot check or complete are refused with a
reason that names the place and what the schema requires there, and carries no argument
text: a `$ref` that does not resolve inside the schema, a draft other than draft-07,
2019-09 and 2020-12, a default under a keyword the walk does not write through or the
gate does not know, a left-out property whose alternatives list different defaults or
whose default has a member its own schema does not allow, a tool the server no longer
lists. The work is bounded by counting it: each keyword applied to a value, each value
walked or read for a comparison is a step, a recursive schema is checked once for each
value, and a call that costs more
steps than the gate allows for the size of its arguments is refused. The policy
judges a copy of the completed arguments and the server is sent the completed arguments,
so a policy's write to what it judged changes its copy, not the call. The one exception
is a listed default that cannot be sent: one that is not valid
against its own property schema (`q: str = None` publishes one), which the server would
reject, and text a call would be refused for sending (`mode: Optional[str] =
"null"`, which the server would decode). It is judged as listed and
not sent. That holds where the listing is what the server runs with, as on the MCP
Python SDK; a zod server derives some of these defaults another way
(`z.number().default(Infinity)` lists `null` and runs `Infinity`), and the README lists
the cases. A parameter left
out that has no listed default is the server's to fill and is not seen, nor is a server
whose code does more than its schema says, and for hosted
MCP the server's schema is not read at all. A refusal is returned
as an MCP error result carrying the reason, so the model reads it and the server is never
called. A policy keyed by tool name names the server's own tools. A policy written for
Claude Code's names, `recusal.mcp.manifest_policy` or a composition carrying one, judges
`mcp__<server>__<tool>`; hand it to either gate through `claude_named(server.name, policy)`,
which presents each call under that name and keeps the manifest digest on the audit
record. Unbound, the gate refuses the pairing at setup, because a membership check that
never sees an MCP name would raise no objection. A composition carrying a bound policy
is bound to the same server, in either order of wrapping; policies bound to different
servers, or a bound policy beside an unbound manifest policy, cannot be composed.
Hosted MCP runs on OpenAI's servers, and the one local point is the approval callback, so
`gate_hosted_mcp(tool, policy)` returns a copy that requires approval for every call and
answers a refusal as a rejection with Recusal's reason; a call OpenAI does not ask approval
for is never seen, which is why a hosted tool with any other `require_approval` is reported
ungated. `gated_agent(agent, policy)` there gates function tools, servers and hosted tools
together. Same pins, same CI jobs, same offline fixtures, plus an in-process MCP server.
To put every one of those calls on the record,
[`examples/openai_agents_mcp_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_agents_mcp_audit.py)
swaps in `audited_agent(agent, policy, log)`: one hash-chained `AuditLog` for the agent's
three seams, each under its own label. A local server's entry is labeled
`openai_agents.mcp.call_tool`, names the server as `control.mcp_server` and carries no
call id, because the SDK hands `call_tool` none; a hosted approval's entry is labeled
`openai_agents.hosted_mcp.approval` under the approval request's id; a function tool's is
the function-tool example's. An unwritable log refuses the call or rejects the approval.
A server is a shared connection and carries one recording, as it carries one policy.

## OpenAI API, your own loop

Package: [`recusal-openai`](../adapters/openai/README.md).

No framework at all, the OpenAI API called directly: [`examples/openai_responses_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_responses_gate.py)
gates the loop your application already owns. `gate_function_calls(response, tools, policy)`
judges every `function_call` item in a Responses API result and returns one
`function_call_output` per call under its own `call_id`; `gate_tool_calls(completion, tools,
policy)` does the same for the `tool_calls` of one Chat Completions choice, returning
`role: "tool"` messages: a completion with several choices (`n > 1`) carries alternative
messages, so it is refused until the application passes `choice=` for the one it selected.
Each call is bound to the tool's signature with its defaults applied before judgment
(argument binding, no type coercion on this path), so a default the model omitted is
judged at the value the function would use, and the bound invocation is what runs; a
tool with variadic, positional-only or unannotated parameters, or a default outside the
JSON domain, is refused, as is a call that does not bind (the refusal names the
parameters the tool declares and counts the arguments it does not take, never quoting a
key or a value the model sent).
Arguments that are not a string, invalid JSON (`NaN` and `Infinity` included), a number
that is not finite (`1e999`), arguments nested more than 64 levels deep, non-object
arguments, an unknown tool and a raising policy are refused, never run (the gate always
fails closed: it does not take `fail_closed`). A refusal names a tool the application has
by its name; any other name is quoted only if it is 1 to 64 printable characters with no
backtick, and is otherwise described by its length, in the answer and on the audit log.
Also refused: a call that
is not finished (a `function_call` item or a response whose `status` is not `completed`,
a Chat Completions choice whose `finish_reason` is not `stop`, `tool_calls` or
`function_call`) and a `function_call` item that carries a `namespace`. An output item or
tool call of a type the adapter does not handle raises, naming it, as does a Chat
Completions message with the legacy `function_call` field and a keyword argument the gate
does not take (a mistyped one, `fail_closed`, `tool_use_id`, `surface`, `actor` or
`control` with a value when `audit` is not passed at all, `audit` with a log and no
`surface`), and a call without an id is refused because the SDK's answer types do not
require one. A call id goes back in the answer and onto the audit entry exactly as sent,
so the gate takes one of at most 256 characters and a longer one raises before anything
runs. `audit=None` passed explicitly is the plain gate, with which `surface`, `actor` and
`control` are accepted and not used, so a loop whose log is optional is one call
(`audit=maybe_log, surface=...`); the two audited functions need a log and raise for
`None`. A `surface` or `control` that `recusal.decide` would refuse (an empty or reserved
surface, a control that is not a dict) raises `ValueError` naming the keyword before any
call is looked at. The gate keeps no state between turns and has no helper for streaming:
it takes a complete response.
Whether a tool's signature can be judged is the kernel's answer, which is why the package
requires `recusal>=0.15.0`, the first kernel that walks the wrappers of a decorated tool:
any reason the kernel gives, and any error it raises, refuses the call before the policy
runs. Two shapes are not seen: a wrapper that restates the function's parameters and
does not pass one on in its body, which no signature shows, and, with a kernel that does
not detect it, a decorator applied inside another decorator's closure.
The gate calls tools synchronously: an `async def` tool and a generator function are
refused before the policy runs, read through a `functools.partial`, a `staticmethod`, a
bound method and `functools.lru_cache` but not through a wrapper the application wrote.
A tool that returns a generator or another iterator is answered as having run with no
result available, never as not run. A `functools.lru_cache` tool handed a list or an
object, which the cache cannot hash, is refused before the policy runs. A function call
marked `async`, or whose `caller` is a program, is judged and run like any other. A tool
that raises reaches the application by default, and with
`answer_tool_errors=True` is answered with its exception's class name under its own id
while the rest of the turn is judged and run. Hosted tools (web search, file search, code
interpreter, image generation, remote MCP) run on the provider's side, so the gate never
judges them; the provider's record of one (`web_search_call`, `mcp_call`, ...) raises by
default and, when finished, passes unjudged and unrecorded with
`allow_hosted_records=True`. A remote MCP `mcp_approval_request` always raises: this gate
does not answer it. Offline (`model_validate` builds real SDK objects, no key),
Python 3.10+, tested with exactly `openai==3.49.0`. To put every one of those calls on the
record, [`examples/openai_responses_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/openai_responses_audit.py)
swaps in `audited_function_calls(response, tools, policy, log)` and
`audited_tool_calls(completion, tools, policy, log)`: the same decision, written as a
hash-chained `AuditLog` entry labeled `openai.responses.function_call` or
`openai.chat_completions.tool_call` with the API's call id and a fingerprint of the bound
invocation, defaults included, before the call runs or is refused, and an unwritable log
refuses the call. A call refused before judgment is one entry too. A turn refused as a
whole (an unhandled item type, a legacy `function_call`, a missing or duplicate id, a
keyword the gate does not take) raises before anything is judged, so it runs nothing and
writes no entry.

## Google ADK

Package: [`recusal-google-adk`](../adapters/google-adk/README.md).

For Google's Agent Development Kit, [`examples/google_adk_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/google_adk_gate.py)
uses the ADK's own `before_tool_callback`: `recusal_before_tool(policy)` judges each tool call
immediately before the `LlmAgent` runs it and answers a refusal as the tool's response, so the
model reads the reason and the tool never runs. The callback prepares the invocation the
tool will execute: omitted parameters are completed from the function's defaults into the
arguments, the tool's own preparation (coercion, when the ADK's argument validation is on)
runs once on them, the converted values are written back and a copy is judged, so the
policy sees exactly what the function receives. Only a `FunctionTool` keeping the stock
preparation methods, with parameters annotated `str`, `int`, `float`, `bool`, optionals
and lists or string-keyed dicts of those, is supported; a model-typed, `Annotated`,
`Enum`, variadic or unannotated parameter is refused, and a validation failure is
refused without the SDK's message. `gated_agent(agent, policy)` returns a gated
copy with the gate as the last callback (a callback listed after it could change the
arguments it judged) and gates `sub_agents` and the agent behind every `AgentTool` the same
way; it refuses an agent with a `code_executor` or an `LlmAgent` subclass, which can run
things the callback never sees. `RecusalPlugin(policy)` is the same gate as a runner plugin,
covering every agent the runner drives, with an explicit ordering contract: plugin
callbacks run before any agent's callbacks, so the plugin judges a call only when nothing
it cannot see runs after it, and refuses the call, naming the fix, when the executing agent
carries a `before_tool_callback` that is not a Recusal gate (use `gated_agent` for that
agent) or a plugin listed after it overrides `before_tool_callback` (list `RecusalPlugin`
last). Offline (a scripted model, no key), Python 3.11+,
tested with exactly `google-adk==2.14.0` and `google-genai==2.34.0`. To put every one of
those calls on the record, [`examples/google_adk_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/google_adk_audit.py)
swaps in `audited_agent(agent, policy, log)` (or `audited_plugin(policy, log)` for the
runner): the same decision, written as a hash-chained `AuditLog` entry labeled
`google_adk.before_tool_callback` with the ADK's function call id and a fingerprint of the
prepared invocation, before the call runs or is refused, and an unwritable log refuses
the call. A call another callback answered before the gate ran is not on the log; the
tool did not run.

## CrewAI

Package: [`recusal-crewai`](../adapters/crewai/README.md).

For CrewAI, [`examples/crewai_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/crewai_gate.py) uses CrewAI's own tool-call
hooks: `install_recusal_hooks(policy)` registers a `before_tool_call` hook that judges each
call and blocks a refusal, and an `after_tool_call` hook that replaces CrewAI's fixed
"blocked by hook" line with the reason the before hook blocked on (kept per invocation and
never judged again; where the reason was not kept for a call, CrewAI's fixed line stays), so the model
reads why. The policy judges the invocation the tool receives, not the proposal: the hook
completes omitted fields from the schema's defaults, runs the tool's own `args_schema`
validation once, writes the validated values back into the dict CrewAI executes and judges
a copy, so `charge(amount: int = 100)` proposed with `{}` is judged as `{"amount": 100}`.
An omitted field whose default is `None` (`limit: int = None`, which would fail validation
written in) is left for the schema to supply, as CrewAI's own validation supplies it, and
judged as `None`.
CrewAI does not always execute the dict a hook is handed, nor always the tool: on the text
route an empty call is invoked with a new empty dict, a hook may replace the dict or point the
context at another tool (CrewAI runs the tool it selected itself), and when two
tools normalise to one name ("Run SQL" and `run_sql`) one tool can be handed to the hook
while another runs. So the gate reads, in the CrewAI function that ran the hooks and is
about to run the call, the tools the call is chosen from, the function about to run and
the dict it will run with, and judges only a call it can tie to them: it makes an empty
call's dict the one CrewAI runs, and refuses before judgment a call to a name two tools
share, a call whose function, dict or selected name is not the handed tool's, a call to a name CrewAI
made up for a renamed tool, and a call whose hooks CrewAI ran from a function it does not
know. The policy is given the normalised name.
It supports a stock tool and schema only: a `CrewStructuredTool` and `BaseTool` that keep
CrewAI's run and validation methods, and a schema that is a direct `BaseModel` with no
members of its own, configuration limited to `extra` and `title`, and bare field
declarations (a supported annotation among `str`, `int`, `float`, `bool`, optionals, lists
and string-keyed dicts of those, an optional default of the exact JSON types or a
`default_factory` that is `list` or `dict` itself, description and title),
which is what CrewAI generates for `@tool` functions and `BaseTool` subclasses. A
customised schema (a validator, an overridden `model_validate`, an intermediate base, a
field with `exclude`, `exclude_if`, any other `default_factory`, an alias or a constraint), an
unsupported annotation, an unknown parameter, a missing required one and a validation
failure (named by field, never with Pydantic's text) are refused before judgment, as is a
tool class that overrides `to_structured_tool`, a `Tool` subclass with its own `_run` or
`_arun`, a run or validation method set on the tool object, and a callable that
declares a `__signature__` or has a parameter its schema omits, counting the parameters
of every callable a call reaches: through a method, a `functools.partial`, a callable
object's `__call__` and every `functools.wraps` wrapper (`lru_cache`, `cache` and
`staticmethod`, on a function or a method, are read as the function they wrap). A plain
`BaseTool` that takes `Tool.run`, a schema with a metaclass of its own, a validator or
serializer that is not Pydantic's, a tool class with its own attribute lookup, a callable
object whose `__call__` is not a plain method and a class as the tool's callable are
refused too. CrewAI validates the dict the gate writes once more before the tool runs, so
the gate judges arguments only when they are plain JSON and validating them again gives
them back exactly. The
registry is
process-global: one install covers every agent, task and crew in the process, and the
function it returns uninstalls both hooks. Install it after your other hooks, because a hook
that runs after the gate can change the argument dict the gate judged; the tests show the
bypass and the fix. Offline (a scripted `BaseLLM`, no key), Python 3.11+, tested with
exactly `crewai==1.15.44`. To put every one of those calls on the record,
[`examples/crewai_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/crewai_audit.py)
swaps in `install_audited_hooks(policy, log)`: the same decision, written as a hash-chained
`AuditLog` entry labeled `crewai.before_tool_call` with a fingerprint of the prepared
invocation, before the call runs or is blocked, and an unwritable log blocks the call.
CrewAI's hook context carries no call id, so these entries carry none; the agent's role
is recorded as `control.crewai_agent`. A call a hook registered before the gate blocked
never reaches it and is not on the log.

## Pydantic AI

Package: [`recusal-pydantic-ai`](../adapters/pydantic-ai/README.md).

For Pydantic AI, [`examples/pydantic_ai_gate.py`](https://github.com/philpaz/recusal/blob/main/examples/pydantic_ai_gate.py) uses the
framework's own `before_tool_execute` hook, which runs with the validated argument dict
immediately before a tool runs. `RecusalGate(policy)` is a capability that judges each call
there (the policy sees detached data; the tool runs with the validated dict itself,
so a policy's write to its containers changes its observation, not the call) and raises
`ToolFailed(reason)` on a refusal, so the model reads the reason as a failed
tool result, the tool never runs, and no retry budget is spent; attach it with
`Agent(..., capabilities=[RecusalGate(policy)])` or per run. It declares itself `innermost`,
so Pydantic AI sorts it after every ordinary capability (a hook that ran after the gate could
change the arguments it judged), and it reads the order the run dispatches in: a call is
refused when a `before_tool_execute` of the application's comes after the gate, so list the
gate directly and last. From `recusal-pydantic-ai` 0.2.0 the capability judges more than the
tool hook: output functions and `Choices` picks before they run, calls handed to the
application (with the application's handler given to the gate, `deferred=handler`), and
defaults a function would fill in itself; and it refuses what it cannot hold to the policy
(a toolset that runs other tools' calls itself, a tool the provider runs, another capability
that answers deferred calls, a realtime session). The package's
[README](../adapters/pydantic-ai/README.md) is the full account, with what the gate does
not see. `Hooks(before_tool_execute=recusal_before_tool_execute(policy))`
is the tool hook alone as a plain function, with no ordering of its own. An agent called from
inside a tool function runs under its own capabilities and needs its own gate; the tests show
the miss and both fixes. Offline (a scripted `FunctionModel`, no key), Python 3.11+, tested
with exactly `pydantic-ai-slim==2.67.0`. To put every one of those calls on the record,
[`examples/pydantic_ai_audit.py`](https://github.com/philpaz/recusal/blob/main/examples/pydantic_ai_audit.py) swaps in
`RecusalAuditedGate(policy, log=log)` (or `recusal_audited_before_tool_execute(policy, log)`
for `Hooks`): the same decision, written as a hash-chained `AuditLog` entry labeled with
the hook that decided (`pydantic_ai.before_tool_execute` for a tool call) with Pydantic AI's
tool call id and a fingerprint of the arguments the tool is called with, before the call
runs or is refused, and an unwritable log refuses the call. A call Pydantic AI rejects at validation never reaches the hook, so
it does not run and is not on the log.

Its supported arguments are data-only: JSON scalars, lists and string-keyed dicts.
Containers are detached by construction, without application copy methods or serializers.
Models, dataclasses, Enum members and other objects refuse before policy evaluation;
declare data-only parameters (a `str` or `int` in place of an Enum) and resolve
application types inside the tool body. A detachment failure names
only the exception class, never its text.
