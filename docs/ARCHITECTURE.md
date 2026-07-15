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

## Public compatibility perimeter

For released versions, compatibility is larger than Python function signatures. A
behavior-preserving internal refactor must retain:

- documented imports from `recusal.mcp`;
- function signatures, return shapes, finding order, check names, severities, and context;
- canonical manifest bytes and all source, instruction, and tool fingerprints;
- manifest validation and fail-closed behavior;
- CLI commands, JSON shapes, and exit codes;
- Claude Code hook defer/deny behavior and audit control identity;
- the version lock among the Python package, Claude plugin, marketplace, examples, and
  release metadata.

Private helpers are not a promised API merely because repository tests exercise them.
They should still be moved deliberately because external users may have coupled to them.
Recusal currently defines no `recusal.mcp.__all__`; adding a restrictive one would change
wildcard-import behavior and requires a separately reviewed compatibility decision.

## Incremental MCP decomposition

`recusal.mcp` has accumulated several cohesive responsibilities. Decomposition should be
incremental, with the historical module remaining the public facade. The first candidate
is declaration and server-instruction screening because it is pure and does not participate
in manifest serialization, runtime authorization caching, process collection, or hook
protocol handling.

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
