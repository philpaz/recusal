# Architecture and compatibility boundaries

Recusal uses a functional core with adapters at the edges. Checks and policies emit
`Finding` objects, `compute_verdict` folds them into a `Verdict`, and adapters translate
that verdict into a Claude hook decision, a CLI exit status, an audit record, or a release
decision. The evidence kernel must remain deterministic and free of filesystem, process,
network, and framework dependencies.

## Dependency direction

The intended dependency direction is:

```text
checks / policies / MCP observations
                |
                v
            evidence
                |
                v
    adapters, audit, gates, and CLI
```

`recusal.evidence` is the innermost boundary. It must not import adapters, persistence,
the CLI, or MCP collection. `recusal.mcp` is deterministic MCP adjudication; live process
collection remains in `recusal.mcp_fetch`.

## What the framework gates share

A gate in front of a tool call promises that the arguments the policy judged are the
arguments the tool ran with. The parts of that promise that are about data, not about a
framework, live in `recusal.invocation`: the annotations a gate can judge by value
(`supported_annotation`), the JSON domain a default must sit in (`json_data`), whether
every parameter of a callable can be judged by name (`unsupported_signature`), and the
copy a policy may write to without reaching what executes (`detached`). The module imports
only the standard library and calls no policy.

The gates under `examples/` import these instead of carrying their own copy, and
`tests/test_invocation.py` fails if a file under `examples/` defines one of them again.
What stays in each gate is the framework's own: where its hook sits, how it prepares an
invocation, and which parameters it injects. The LangGraph gate's projection of models and
dataclasses (`_faithful`) is specific to that framework's argument types and stays there.

## How a gate puts its decision on the record

Every framework gate has an audited route, in a companion file beside it
(`examples/<runtime>_audit.py`). None of them decides anything itself. The plain gate's own
refusal function accepts the recording arguments of `decide` (`audit=`, `surface=`,
`tool_use_id=`, `actor=`, `control=`) and passes them through, so the audited gate and the
plain gate are one code path, and `decide` writes the entry before it returns. The
companion file adds only what is the framework's: where the call id comes from, and the
surface label.

| Runtime | Audited example | Surface on the entry | Call id on the entry |
|---|---|---|---|
| LangGraph `ToolNode` | `examples/langgraph_audit.py` | `langgraph.toolnode` | the tool call id |
| LangChain `create_agent` | `examples/langchain_agent_audit.py` | `langchain.create_agent` | the tool call id |
| OpenAI Agents SDK, function tools | `examples/openai_agents_audit.py` | `openai_agents.tool_input_guardrail` | the tool call id |
| OpenAI Agents SDK, local MCP server | `examples/openai_agents_mcp_audit.py` | `openai_agents.mcp.call_tool` | none: the SDK hands `call_tool` no id; the server is `control.mcp_server` |
| OpenAI Agents SDK, hosted MCP | `examples/openai_agents_mcp_audit.py` | `openai_agents.hosted_mcp.approval` | the approval request id |
| OpenAI Responses API, your own loop | `examples/openai_responses_audit.py` | `openai.responses.function_call` | the `call_id` |
| OpenAI Chat Completions, your own loop | `examples/openai_responses_audit.py` | `openai.chat_completions.tool_call` | the tool call id |
| Google ADK | `examples/google_adk_audit.py` | `google_adk.before_tool_callback` | the function call id |
| CrewAI | `examples/crewai_audit.py` | `crewai.before_tool_call` | none: the hook context has no id; the agent's role is `control.crewai_agent` |
| Pydantic AI | `examples/pydantic_ai_audit.py` | `pydantic_ai.before_tool_execute` | the tool call id |

Three rules hold on every row, and each gate's audit tests hold them:

- **A judged call is bound to what executes.** The entry's `input_sha256` is the
  fingerprint of the invocation the tool runs with (defaults completed, values coerced,
  as that framework prepares it), whatever the policy did to the copy it was handed.
- **A call refused before judgment is still one entry.** A call the gate cannot judge
  (arguments that are not a mapping, a signature or schema it cannot read) is recorded as
  a deny with the finding `recusal_unjudgeable_input`, bound to the arguments as the model
  proposed them.
- **The record is part of the control.** An unwritable log or a failing sink turns the
  decision into a deny, and so does a recording callable that raises.

What a gate never sees is not on its log, and each companion file names it: a call the
framework rejects before the hook runs (a validation failure in Pydantic AI, an unknown
tool in the OpenAI Agents SDK), a call another hook answered or blocked first, a hosted
call the provider did not ask approval for, and a whole turn the direct-loop gate refuses
by raising. `tests/test_audited_routes.py` fails if a framework gate is added without its
audited companion, or if a surface label in this table stops matching the code.

## Adapter packages

A framework gate starts as a tested example and can become its own distribution under
`adapters/<runtime>/`. Two exist: `adapters/pydantic-ai`, the package
`recusal-pydantic-ai`, and `adapters/openai`, the package `recusal-openai`. The
dependency points one way:

```text
recusal-pydantic-ai  ->  recusal (public names only)  +  pydantic-ai-slim
recusal-openai       ->  recusal (public names only)  +  openai
recusal              ->  nothing
```

The kernel declares no dependency, imports nothing from an adapter and does not name
one. An adapter imports `recusal`, `recusal.audit` and `recusal.invocation` and nothing
private. A packaged gate has no default policy and no demonstration code: what to
refuse is the adopter's decision. The example files for a packaged gate (for instance
`examples/pydantic_ai_gate.py` and `examples/pydantic_ai_audit.py`) import the package and
keep only the offline demonstration, so the code a user installs is the code the
examples and their tests run; there is no second copy. `tests/test_adapter_packages.py`
and `tests/test_audited_routes.py` hold each of these statements, and a CI job builds
the wheel with the hash-locked release toolchain, installs it with the kernel's wheel
in a clean environment, and runs the adapter's `smoke_test.py` from outside the
checkout.

An adapter has its own version and its own release. None has been published yet.

## Public compatibility perimeter

This section describes the perimeter; [`../STABILITY.md`](../STABILITY.md) is the promise
attached to it, including which changes force a major version and what 1.0 has to earn.

For released versions, compatibility is larger than Python function signatures. A
behavior-preserving internal refactor must retain:

- documented imports from `recusal.mcp`;
- function signatures, return shapes, finding order, check names, severities, and context;
- canonical manifest bytes and all source, instruction, and tool fingerprints;
- manifest validation and fail-closed behavior;
- CLI commands, JSON shapes, and exit codes, including the one command whose exit code is
  deliberately not a verdict: `recusal demo` exits 0 when it ran, and adjudication belongs
  to `recusal verdict`;
- Claude Code hook defer/deny behavior and audit control identity;
- the `recusal.authorization` surface (0.9.0): the `ActionRequest`, `AuthorizationContext`,
  `Constraints`, `Supplied` and `DecisionReceipt` shapes, the nine dimension names and
  their `authorization.<dimension>` check names in order, the rule that a required
  dimension with no finding refuses, the receipt's canonical body and digest, and the
  `runtime` context keys the hook records with provenance `claude_pretooluse_event`
  outside the control identity;
- the version lock among the Python package, Claude plugin, marketplace, examples, and
  release metadata.

Private helpers are not a promised API merely because repository tests exercise them.
They should still be moved deliberately because external users may have coupled to them.
Recusal currently defines no `recusal.mcp.__all__`; adding a restrictive one would change
wildcard-import behavior and requires a separately reviewed compatibility decision.

## Incremental MCP decomposition

`recusal.mcp` has accumulated several cohesive responsibilities. Decomposition is
incremental, with the historical module remaining the public facade. The first extraction
is done (0.6.0): declaration and server-instruction screening lives in the private
`recusal._mcp_screening`, chosen first because it is pure and does not participate in
manifest serialization, runtime authorization caching, process collection, or hook
protocol handling. `recusal.mcp` still exposes it.

The second extraction is done (0.12.7): the call-time manifest policy, the file-backed,
locked, cached half with its invocation-local digest, lives in the private
`recusal._manifest_policy`, and `recusal.mcp` keeps `manifest_policy` as a wrapper with the same
signature, docstring and module. The policy module imports the artifact half's validator only at call time, so
either module can be imported first (a deferred dependency, not a removed one). It also
imports the evaluator's return-contract helper (`recusal._adjudicate._findings_of`) at
module load, so a wrapped policy's return is held to the same contract `decide` applies;
`recusal._adjudicate` imports only the kernel and `recusal.authorization`, never
`recusal.mcp`. What
remains in `recusal.mcp` is the artifact half: building, comparing and validating
manifests and observations, plus `load_manifest`, which reads a manifest file, so that
module is not free of filesystem access; the kernel in `recusal.evidence` is.

The same pattern was applied to `recusal.claude_code` in the same change: the neutral
evaluator that every adapter reuses (`decide`, the one audit-record path and the control
identity it writes) lives in the private `recusal._adjudicate`, and `recusal.claude_code`
keeps the hook, its event parsing and the allowlist, with `decide` as the same kind of wrapper and
every private name re-exported. `decide` is also exported from the package itself
(`from recusal import decide`, the same function), since every runtime's gate calls it and
only one of them is Claude Code's. The evaluator imports no adapter,
persistence or MCP code; `tests/test_extractions.py` pins both facades, both boundaries
and a golden audit record captured before the move.

The safe sequence is:

1. lock the observable contract with source and installed-wheel tests;
2. extract one private implementation module;
3. retain public wrappers in `recusal.mcp` so imports, signatures, and callable identity
   remain stable;
4. make no policy, message, manifest, or feature changes in the extraction commit;
5. run the full cross-platform suite and installed-artifact smoke test before release.

Converting `recusal.mcp` from a module into a package is intentionally deferred. It is a
larger import-system and packaging change and is not needed to obtain the first
reviewability benefit.
