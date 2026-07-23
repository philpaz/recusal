# MCP governance

The same gate that adjudicates `Bash` adjudicates every MCP tool call, and `recusal.mcp`
adds deterministic integrity controls at the discovery boundary: pin the approved server
instructions and tool declarations once, then refuse represented drift. This page is the
full statement, boundaries and named residuals included; the README carries the summary.

## MCP tools, the same gate

MCP server tools reach Claude Code as ordinary tools: the hooks reference documents that
they "appear as regular tools in tool events" (`PreToolUse`, ...) under the naming pattern
`mcp__<server>__<tool>` (`mcp__github__create_issue`, `mcp__filesystem__write_file`). So
the `.*` matcher in the README's hook registration already routes every MCP call through
the same `policy(tool_name, tool_input)` seam: no MCP-specific adapter, no extra wiring.
The same call-time controls apply to MCP exactly as to `Bash`: destructive-operation
refusal, repository/record scope, write-path confinement, egress and action budgets, the
tamper-evident audit record. In allowlist mode the posture is stronger still: an MCP tool
is **refused unless affirmatively named** (`allow={"mcp__github__create_issue": vet}`),
the least-privilege default the MCP spec's own security guidance pushes toward. Pinned as
tests.

```python
def policy(tool_name, tool_input):
    if tool_name == "mcp__salesforce__delete_records":
        return [Finding.fail("mcp_destructive_action", severity="CRITICAL",
                             message="bulk Salesforce deletion is not approved")]
    if tool_name == "mcp__github__merge_pull_request":
        repo = tool_input.get("repo")
        if repo not in {"philpaz/recusal"}:
            return [Finding.fail("mcp_repository_scope", severity="CRITICAL",
                                 message=f"repository {repo!r} is outside the approved scope")]
    return []   # defer everything else to Claude Code's normal flow
```

Runnable: [`examples/mcp_governance.py`](../examples/mcp_governance.py) (approved-server
pinning, destructive-verb refusal, path confinement, allowlist mode). Pinned:
[`tests/test_mcp_governance.py`](../tests/test_mcp_governance.py). Recipe:
[`COOKBOOK.md`](COOKBOOK.md) §12. In a custom Agent SDK or MCP-client loop
nothing intercepts for you: invoke the gate between the model's proposed MCP call and the
client dispatching it (the same `gate_tool_use` seam as the README's Agent SDK section).

**The three MCP tool-call boundaries, stated plainly.** A call-time policy adjudicates the proposed
tool name and arguments; MCP has two more boundaries, and Recusal covers each with its own
evidence:

| Boundary | Threat (as the field names it) | Recusal |
|---|---|---|
| Discovery (`initialize.instructions` + `tools/list`) | model-facing server instructions, tool-description poisoning (benchmarked against real-world MCP servers by MCPTox), unapproved capability, post-approval declaration changes (the rug pull), name collisions | **pin + refuse drift**: `recusal mcp pin` / `recusal mcp verify` / `recusal.mcp.manifest_policy` (next section); legacy tools-only observations keep an explicitly weaker instruction claim |
| Invocation (the call) | tool misuse (OWASP ASI02), wrong-subject writes (ASI03), exfiltration via tool invocation (MITRE ATLAS AML.T0086) | **this section** |
| Response (the result) | indirect prompt injection in tool output (OWASP LLM01) | quarantine, [cookbook recipe 6](COOKBOOK.md) |

Transport and authorization threats (confused deputy, token passthrough, session
hijacking) are the MCP specification's own
[Security Best Practices](https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices)
layer, complementary to this gate, neither replaces the other. Every source here is
verified in [`REFERENCES.md`](REFERENCES.md).

## MCP discovery integrity: pin server instructions and tool declarations, refuse represented drift

The model chooses tools by reading their declared descriptions, so a poisoned declaration
steers the agent *before any call exists* for a call-time policy to see, and the call that
follows looks structurally valid. `recusal.mcp` adds deterministic integrity controls at
that boundary the way this library governs every boundary: deterministic evidence, with
the human where the judgment is:

```bash
recusal mcp pin --claude-config .mcp.json --approve-server-launch   # review once, pin
recusal mcp verify --claude-config .mcp.json  # CI / session start: same represented source, server instructions, and tool declarations, or refuse
```

> **`--claude-config` and `--stdio` execute the declared server commands** to ask them
> for `tools/list`; there is no other way to ask a process for its catalog. The first
> pin is therefore an explicit trust event: review the `command`/`args` lines like you
> review the declarations, then pass `--approve-server-launch` to record it. After the
> pin, `verify` compares each launch specification against the manifest **before**
> launching anything, and stdio servers run with a minimal environment by default
> (`--inherit-env` is the named opt-out). Minimal environment is not a sandbox: the
> server still runs with your user's filesystem, process, and network permissions. And
> `--from` pins the supplied declaration set; it does not attest which remote endpoint
> produced the dump.

Recusal does not judge whether a description is *malicious*: that is semantic judgment, a
human's call at pin time (a deterministic marker screen surfaces the obvious, and `pin`
refuses to write over a flagged catalog until `--force` records that a human reviewed it).
What it detects, deterministically, is **unpinned capability and post-approval change**:
the rug pull, the new tool, the mutated schema. The pin is the confirmed human decision
promoted to a deterministic artifact: manifest bytes are reproducible, tool
declarations and server instructions are stored as hashes only (poisoned text is
never embedded anywhere) while source templates are stored readable so drift can be
explained - keep secrets out of them, the pin warns - and the same complete
observation against the same pin, under the same recusal version, yields the same
verification result, every time. `verify` fails **closed**: a missing
manifest, a failed fetch, a wholly empty observation, or a pinned server that can no longer
be reached for integrity-checking (e.g. silently swapped to a URL transport) is a refusal,
never a clean-looking pass. (A pinned server *legitimately removed* from the config is
recorded as a warning, not refused: a shrunk capability set is not an attack.) The pin also
enforces at call time: `recusal.mcp.manifest_policy("mcp-manifest.json")` drops into the
same `PreToolUse` gate and refuses any `mcp__server__tool` call that was never pinned (no
pin, no MCP), composing with the argument-level rules above. A minimal zero-dependency
stdio client collects `tools/list`; **remote/HTTP servers** are pinned from a JSON dump you
produce with any MCP client (`--from`, copy-paste recipe:
[`COOKBOOK.md`](COOKBOOK.md) §14; local/`.mcp.json` servers pin directly, §13).
Recusal owns the deterministic adjudication, not the transport, so it inherits neither the
HTTP client's dependencies nor its SSRF surface. Collection is never decision: the kernel
adjudicates what was observed.

The honest boundary: this is *discovery-time and call-time* integrity, not a live tap on
every message. `verify` compares the represented source templates, observed
server-instruction state, and tool declarations at the moment it runs (wire it into CI
and session start); the call-time gate then enforces *approved tools only*. And the
server SET is inventory-checked: a pinned server absent from the entire observation is
a CRITICAL refusal (`mcp_server_unobserved`), never a silent pass - a partial
observation must not verify clean while the manifest keeps authorizing that server's
pinned runtime names. Acknowledge a deliberate removal with `--removed NAME`
(recorded, not blocking), then re-pin to make the shrunk set the approved truth. A server that
serves one catalog to `verify` and a different one to the live session (a client- or
time-discriminating server) is a residual this layer names rather than claims to close:
run `verify` against the same endpoint the session uses, close in time. The manifest
pins the **source specification as well as the declared catalog**, and `verify` compares
it **before** any process starts. For stdio servers that is the unexpanded command
template, args, cwd, and the environment value *templates* as written in the config, so
a rewritten command, a same-key env value swap (`NODE_OPTIONS`, `LD_PRELOAD`), or a
`${VAR}` reference rename is refused without the replacement ever executing (each pinned
by an adversarial test proving the substituted command's marker file is never written).
Every server entry in the SUPPLIED `.mcp.json` is represented or the operation
refuses: a remote entry pins its `url_template`, header value *templates*, and
`headersHelper` command template, plus - for `http`/`sse`, the transports Claude
applies its preconfigured OAuth flags to (SSE is deprecated upstream in favor of
HTTP) - the OAuth policy fields; `ws` is header-only and an entry carrying `oauth`
refuses; a server name Claude reserves for its built-ins (`workspace`,
`claude-in-chrome`, `computer-use`, `Claude Preview`, `Claude Browser`) refuses
before anything launches, since Claude skips such entries and representing one would
misdescribe the effective configuration; an added or transport-swapped server of any
kind is drift, and a config entry the parser cannot faithfully represent fails
closed.

Launch-file identity is opt-in since manifest v7: `pin --resolve-executable` pins the
`{path, sha256}` of the file each stdio command's argv[0] resolves to
(`shutil.which` semantics), and verify then refuses when the resolved path or the
file's bytes change even though the command template did not. Every v7 server entry
states the claim explicitly: `null` is the template-only pin, never an omission.

An observation-scope label is opt-in since manifest v8: `pin --scope LABEL` stores an
operator-supplied claim about WHAT this pin observed (a project config, a machine, an
environment), top-level and verbatim; `null` is the explicit no-scope-declared state,
never an omission. The label is operator metadata: `verify` prints it for review
context, and a re-pin under `--update` whose scope differs from the approved one
emits the named `mcp_observation_scope_changed` WARNING (a review signal, never a
refusal, because the replacement is already deliberate). Recusal verifies the label's
stability, never its truth: a scope that says "production config" proves nothing about
which config was actually observed.

The remaining residuals, named: the operator-shell *values* behind `${VAR}` references
are not pinned (the reference is); under a `null` (template-only) pin,
`npx`/`uvx`-style launchers resolve through PATH and fetch what the registry serves
(pin package versions in the args) and executable bytes are not attested; even a
strict pin attests the FIRST process image only, so what an interpreter or launcher
loads afterward (a script argument, a registry download at run time) stays behind the
template, not the hash; and a `--from`-only pin records `transport: external`,
attesting the declaration set, not the endpoint that produced it. Keep protecting
`.mcp.json` and `mcp-manifest.json` as control-plane files - the default deny-list
does.

Scope, stated exactly: Recusal verifies the configuration artifact YOU supply
(`--claude-config`/`--stdio`/`--from`); it does not reconstruct Claude Code's effective
MCP environment across local, project, user, plugin, claude.ai connector, CLI/SDK,
project-approval, disabled-server, or managed-deployment state, and a successful
verification does not prove Claude accepted, enabled, connected to, or selected the
supplied entry as the effective definition - use Claude managed MCP policy
(`allowedMcpServers`) to constrain the effective server set, then pin what it allows.
Recusal governs MCP *tools* and, since manifest v5, the initialize-result server
*instructions* (pinned as a hash; added, removed, or changed instructions are drift).
With Claude Code's default tool-search behavior those instructions and the tool names
load at session start while full tool definitions are deferred; full definitions load
up front when tool search is disabled or falls back, when a server sets `alwaysLoad`,
or when a tool declares `anthropic/alwaysLoad` (that tool-level flag lives inside the
declaration, so it IS part of the declaration fingerprint; the server-level
`alwaysLoad`/`timeout` fields are shape-validated but deliberately not source
identity - recusal pins declaration content, not Claude's loading strategy). Recusal
fingerprints the complete observed instruction string while Claude truncates what it
loads into context (currently 2KB each for instructions and tool descriptions), so a
change outside the loaded prefix still drifts - the safe side of that asymmetry.
**Instruction coverage for remote servers requires the rich `--from` shape**
(`{server: {"instructions": ..., "tools": [...]}}`, or `--server` with the same
single-server object); legacy `{server: [tools]}` dumps stay supported but record
`observed: false` and establish no instruction claim. For OAuth, Recusal pins the
*configured* policy fields in `.mcp.json` (including the configured `scopes` string,
whose change is drift); it does not observe the final authorization request, scopes
Claude appends (such as `offline_access` when the server advertises it), the issued
token, granted authority, or the server-side authorization result. Claude's reference applies
preconfigured OAuth flags to the `http` and `sse` transports (SSE deprecated upstream
in favor of HTTP), and documents WebSocket authentication as header-only, so a `ws`
entry carrying `oauth` is refused as a shape Claude does not support. Prompts, resources,
resource templates, channels, and elicitation can still introduce context without a
tool invocation and are outside `manifest_policy`. So are MCP *roots*: Recusal does
not pin or govern `roots/list` responses (the session launch directory plus
additional working directories Claude answers with) or
`notifications/roots/list_changed`; Claude working-directory permissions,
additional-directory configuration, server-side root enforcement, and
operating-system controls own that boundary. Claude Code supports dynamic `list_changed`
updates: a NEW runtime name stays blocked at call time (for plugin-bundled servers,
subject to the callable-alias residual below), but a changed description under
an already-pinned name is invisible to the call-time hook until you verify again -
verification is point-in-time, not continuous attestation. And Recusal never
authenticates to a remote endpoint: Claude Code (or the MCP client producing your
`--from` dump) owns OAuth, headers, `headersHelper` execution, TLS, and transport;
Recusal records the approved nonsecret templates and adjudicates the supplied catalog.
Plugin-bundled MCP servers use scoped runtime names: for plugin `my-plugin`, server
`database-tools`, tool `query`, the runtime name is
`mcp__plugin_my-plugin_database-tools__query`, so the manifest SERVER key must be
`plugin_my-plugin_database-tools` and the tool key `query` (never the whole tool name
in the server field). Recusal does not discover plugin metadata; supply the exact
runtime server segment yourself. Since manifest v6, runtime identity
is modeled explicitly: pin the server with `--claude-plugin NAME` and Recusal stores
BOTH identities - the raw declaration name (discovery verification, fingerprints) and
the callable name derived per Claude's documented normalization (any character
outside `A-Z a-z 0-9 _ -` becomes `_`), which is what `PreToolUse` membership checks.
A spec-valid dotted tool name (`admin.tools.list`) therefore pins and authorizes
under Claude's spelling (`admin_tools_list`), the raw dotted spelling is NOT treated
as a callable, two raw names that normalize to one callable REFUSE the pin
(ambiguous callable identity certifies nothing), the loader re-derives every stored
callable name and refuses a mismatch, and the mode is explicit in the manifest -
never inferred from a key's spelling (`--claude-plugin` is operator input; the
server key must be the callable-safe runtime segment). The boundary that remains,
point-in-time as ever: `PreToolUse` carries only the callable name, so a
post-verification raw-declaration swap that preserves an already-approved callable
is indistinguishable at call time until the next `verify`, which refuses on raw
identity - exactly why both identities are pinned. v5 manifests are refused with a
re-pin instruction. Verifying a config that contains remote servers
needs their fresh catalogs alongside it:

```bash
recusal mcp verify --claude-config .mcp.json --manifest mcp-manifest.json          # stdio-only
recusal mcp verify --claude-config .mcp.json --from remote-catalogs.json     --manifest mcp-manifest.json                                                   # mixed/remote
```

See the refusal: [`examples/mcp_manifest_rugpull.py`](../examples/mcp_manifest_rugpull.py)
(offline). Pinned as tests: [`tests/test_mcp_manifest.py`](../tests/test_mcp_manifest.py),
[`tests/test_mcp_policy_bridge.py`](../tests/test_mcp_policy_bridge.py),
[`tests/test_mcp_fetch.py`](../tests/test_mcp_fetch.py),
[`tests/test_mcp_cli.py`](../tests/test_mcp_cli.py),
[`tests/test_mcp_runtime_identity.py`](../tests/test_mcp_runtime_identity.py).
