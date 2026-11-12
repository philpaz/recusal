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
