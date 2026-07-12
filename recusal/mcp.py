"""
MCP tool and server-instruction integrity: pin supported source templates, observed
initialize-result instructions, and complete tool declarations; refuse represented drift.

The call-time gate (``recusal.claude_code``) adjudicates a *proposed call*, the tool name
and arguments. This module governs the boundary before that one: **what the MCP server
declared at discovery** (``tools/list``), the surface an attacker reaches with a poisoned
tool description, a post-approval definition change (the "rug pull"), or a tool that was
never approved at all. The model chooses tools by reading their declared descriptions, so
a poisoned declaration steers the agent before any call exists for a call-time policy to
see.

The design is the constitution applied to a new evidence surface, nothing more:

1. **Pin** (a human decision, promoted to a deterministic artifact). ``build_manifest``
   canonicalizes each declared tool and fingerprints it (SHA-256 over canonical JSON).
   Tool declarations and server instructions are stored as *hashes*, so pinning a
   poisoned declaration never embeds the poison anywhere; approved source configuration
   templates are stored readable, so source drift can be compared and explained (keep
   secrets out of them - the pin warns). Pinning is the reviewed, deliberate step; ``recusal mcp pin`` runs a
   deterministic marker screen first (``screen_tool_declarations``, over the whole
   declaration) so obvious injection phrasing is surfaced for review instead of silently
   blessed.
2. **Verify** (deterministic, replayable, no model). ``diff_observation`` compares a
   complete fresh observation (source templates, server instructions, tool catalog)
   against the pin and emits Findings: an unpinned server or tool is a CRITICAL
   failure, a changed declaration is a CRITICAL failure that names the changed fields
   (a changed *description* is the rug-pull vector), a removed tool is a recorded
   WARNING. The same complete observation, pinned manifest, and recusal implementation
   version produce the same findings, every time. (``diff_manifest`` alone compares
   only the tool catalog; use ``diff_observation`` for a full manifest-v5 verify.)
3. **Enforce at call time.** ``manifest_policy`` bridges the pin into the existing
   PreToolUse gate: an ``mcp__server__tool`` call whose server/tool is not in the pinned
   manifest is refused before the server ever sees it, and a missing or unreadable
   manifest fails CLOSED for MCP calls rather than waving them through.

Honest limits, stated up front:

- The marker screen is a deny-list with a deny-list's ceiling: it catches known injection
  phrasing, not novel or paraphrased poison. Whether a description is *malicious* is a
  semantic judgment this library deliberately does not make; the human makes it at pin
  time, and everything after the pin is deterministic drift detection.
- A PreToolUse event carries the tool *name and input*, not the declaration, so
  ``manifest_policy`` enforces "approved tools only" at call time; description/schema
  *integrity* is checked whenever ``recusal mcp verify`` runs (CI, session start, cron).
- The manifest pins each server's *source specification* (for stdio: unexpanded command
  template, args, cwd, and env value TEMPLATES as written; for remote transports:
  url_template, header value templates, the headersHelper command template, and - for
  http/sse - the represented OAuth policy fields; ws is header-only per Claude's
  documented surface) plus its initialize-result INSTRUCTIONS (as a hash)
  alongside the catalog, and verification compares it all BEFORE launching - a changed
  command, a same-key env value swap, a same-name header-template swap, a changed
  helper command, a widened OAuth scope set, changed instructions, or an added server
  of any transport is refused without executing anything. The identity is
  template-level: the operator-shell *values* behind ``${VAR}`` references are not
  pinned (the references are), PATH resolves what PATH says (pin package versions in
  the args for ``npx``/``uvx``-style launchers), executable bytes are not attested, and
  ``transport: "external"`` (dump-supplied catalogs) attests the declaration set, not
  the endpoint. The FIRST pin still executes the stdio commands it observes -
  ``--approve-server-launch`` records that a human approved exactly that.
- Fingerprints are byte-exact over canonical JSON (sorted keys, no whitespace, UTF-8,
  no unicode normalization): a homoglyph swap in a description IS a change and fails.

This module is pure: standard library only, no subprocess, no network, spawns no
threads (a lock guards the policy cache), a function of its inputs. *Collecting* a catalog from a live server (the one place that
spawns a process) lives in the sibling :mod:`recusal.mcp_fetch`, deliberately apart, so
the decision surface never touches a process. Collection is nondeterministic I/O;
adjudication (everything here) is deterministic. Remote HTTP/SSE/WebSocket servers are
not contacted by the stdio-only collector; pin them from a JSON dump instead
(``recusal mcp pin --from``).
"""

import contextvars
import hashlib
import json
import re
import threading
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .evidence import Finding

#: Version 5 pins MCP server INSTRUCTIONS alongside the tool catalog. With Claude
#: Code's default tool-search behavior, tool names and server instructions load at
#: session start while full tool definitions are deferred (full definitions may load
#: up front when tool search is disabled or falls back, when a server sets
#: ``alwaysLoad``, or when a tool declares ``anthropic/alwaysLoad``), which makes the
#: initialize-result ``instructions`` field a discovery-time influence surface: a
#: server that keeps its tools byte-identical but rewrites its instructions steers
#: when and why the model reaches for them. Instructions are pinned
#: as a hash (never readable text), screened at pin time with the same bounded marker
#: review as declarations, and added/removed/changed instructions are drift. An
#: observation that did not carry instructions (a legacy dump) is recorded as
#: ``observed: false`` and never silently receives the stronger claim. Older manifests
#: are refused with a re-pin instruction rather than accepted at a weaker guarantee.
MANIFEST_VERSION = 5

#: Declaration fields fingerprinted individually so a drift refusal can name what moved.
#: The whole declaration is also fingerprinted, so a change in any *other* field still
#: fails, it is just reported without a field name.
_TRACKED_FIELDS = ("description", "inputSchema", "annotations", "title", "outputSchema")

#: The identity fields pinned per transport, and ONLY these per transport: a source
#: carrying fields outside its transport's set is contradictory and refused, never
#: silently trimmed. Templates are pinned UNEXPANDED (what the config says, before
#: ``${VAR}`` expansion); values that would resolve from the environment never appear,
#: and neither do hashes of values (a low-entropy secret's hash is an oracle).
#: ``transport: "external"`` marks a catalog supplied as a dump (``--from``): recusal
#: never launches or contacts it, so identity is out of scope, recorded not implied.
_REMOTE_SOURCE_FIELDS: Tuple[str, ...] = (
    "transport",
    "url_template",
    "header_templates",
    "headers_helper_template",
    "oauth",
)
#: WebSocket is header-only: Claude Code documents that "HTTP supports OAuth ... while
#: WebSocket supports neither", so a ws source carrying oauth is a configuration Claude
#: does not support and pinning it would misrepresent the config surface. (Claude's
#: reference applies preconfigured OAuth flags "only ... to HTTP and SSE transports",
#: so sse keeps the http shape - noting SSE is deprecated upstream in favor of HTTP -
#: and ws refuses.)
_WS_SOURCE_FIELDS: Tuple[str, ...] = (
    "transport",
    "url_template",
    "header_templates",
    "headers_helper_template",
)
_SOURCE_FIELDS_BY_TRANSPORT: Dict[str, Tuple[str, ...]] = {
    "external": ("transport",),
    "stdio": ("transport", "command", "args", "cwd", "env_templates"),
    "http": _REMOTE_SOURCE_FIELDS,
    "sse": _REMOTE_SOURCE_FIELDS,
    "ws": _WS_SOURCE_FIELDS,
}

#: The OAuth policy fields pinned inside a remote source's ``oauth`` object. The client
#: SECRET is never among them (Claude stores it in the OS keychain; a secret does not
#: belong in a manifest, and neither does its hash).
_OAUTH_FIELDS: Tuple[str, ...] = (
    "client_id",
    "callback_port",
    "auth_server_metadata_url_template",
    "scopes",
)


# --- canonicalization and fingerprints ---------------------------------------------------


def _canonical(value: Any) -> bytes:
    """Canonical JSON bytes: sorted keys, no whitespace, UTF-8, no unicode normalization.

    Byte-exact on purpose: two visually identical descriptions that differ in codepoints
    (a homoglyph swap) MUST fingerprint differently.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def tool_fingerprint(tool: Dict[str, Any]) -> str:
    """Fingerprint one declared tool: SHA-256 over its canonical full declaration."""
    if not isinstance(tool, dict):
        raise ValueError(f"a tool declaration must be an object, got {type(tool).__name__}")
    return _sha256(tool)


def split_mcp_tool_name(tool_name: str) -> Optional[Tuple[str, str]]:
    """``mcp__github__create_issue`` → ``("github", "create_issue")``; None if not MCP."""
    parts = str(tool_name).split("__", 2)
    if len(parts) == 3 and parts[0] == "mcp" and parts[1] and parts[2]:
        return parts[1], parts[2]
    return None


# --- the pin: catalog -> deterministic manifest -------------------------------------------


def normalize_source(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a server source specification to the pinned identity shape.

    Transports: ``"stdio"`` (a server recusal launches; identity = the UNEXPANDED
    configuration template: ``command``, ``args``, ``cwd``, and ``env_templates`` - the
    env variable names mapped to their as-written value templates, so a same-key value
    swap in the config IS drift); ``"http"``/``"sse"``/``"ws"`` (a remote server;
    identity = ``url_template``, ``header_templates`` (as-written value templates,
    so a same-name credential-reference swap is drift), the ``headers_helper_template``
    command Claude executes at connect time, and - for ``http``/``sse`` only - the
    pinned ``oauth`` policy: Claude documents WebSocket authentication as header-only,
    so a ``ws`` source carrying ``oauth`` refuses); ``"external"`` (a dump-supplied
    catalog; identity out of scope, recorded not implied). Values that resolve from the environment never appear in a pin, and
    neither do hashes of values: a low-entropy secret's hash is an oracle. Fields
    outside the transport's own set are contradictory and refused, never trimmed.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"a server source must be an object, got {type(spec).__name__}")
    transport = spec.get("transport")
    allowed = _SOURCE_FIELDS_BY_TRANSPORT.get(transport if isinstance(transport, str) else "")
    if allowed is None:
        raise ValueError(
            f"server source transport must be one of "
            f"{sorted(_SOURCE_FIELDS_BY_TRANSPORT)}, got {transport!r}"
        )
    if transport == "ws" and "oauth" in spec:
        raise ValueError(
            "a ws source cannot carry 'oauth': Claude Code documents WebSocket MCP "
            "authentication as header-only (HTTP supports OAuth, WebSocket does not); "
            "pinning an unsupported shape would misrepresent the configuration surface"
        )
    unknown = set(spec) - set(allowed)
    if unknown:
        raise ValueError(
            f"a {transport} source carries fields outside its transport's identity: "
            f"{sorted(unknown)} - a contradictory source is refused, not trimmed"
        )
    if transport == "external":
        return {"transport": "external"}
    if transport == "stdio":
        command = spec.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("a stdio source needs a nonempty string 'command' template")
        args = spec["args"] if "args" in spec else []
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ValueError("a stdio source's 'args' must be a list of strings")
        cwd = spec.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError("a stdio source's 'cwd' must be a string or null")
        env_templates = spec["env_templates"] if "env_templates" in spec else {}
        if not isinstance(env_templates, dict) or not all(
            isinstance(k, str) and k and isinstance(v, str) for k, v in env_templates.items()
        ):
            raise ValueError(
                "a stdio source's 'env_templates' must map variable names to their "
                "as-written (unexpanded) value templates"
            )
        return {
            "transport": "stdio",
            "command": command,
            "args": list(args),
            "cwd": cwd,
            "env_templates": {k: env_templates[k] for k in sorted(env_templates)},
        }
    url_template = spec.get("url_template")
    if not isinstance(url_template, str) or not url_template:
        raise ValueError(f"a {transport} source needs a nonempty string 'url_template'")
    header_templates = spec["header_templates"] if "header_templates" in spec else {}
    if not isinstance(header_templates, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) for k, v in header_templates.items()
    ):
        raise ValueError(
            f"a {transport} source's 'header_templates' must map header names to their "
            "as-written (unexpanded) value templates"
        )
    helper = spec.get("headers_helper_template")
    if helper is not None and (not isinstance(helper, str) or not helper):
        raise ValueError(
            f"a {transport} source's 'headers_helper_template' must be the command "
            "template string, or null"
        )
    oauth = spec.get("oauth")
    if oauth is not None:
        if not isinstance(oauth, dict):
            raise ValueError(f"a {transport} source's 'oauth' must be an object or null")
        unknown_oauth = set(oauth) - set(_OAUTH_FIELDS)
        if unknown_oauth:
            raise ValueError(
                f"a {transport} source's 'oauth' carries unknown fields: {sorted(unknown_oauth)}"
            )
        client_id = oauth.get("client_id")
        if client_id is not None and (not isinstance(client_id, str) or not client_id):
            raise ValueError("oauth 'client_id' must be a nonempty string or null")
        port = oauth.get("callback_port")
        if port is not None and (
            not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535
        ):
            raise ValueError("oauth 'callback_port' must be a TCP port (1-65535) or null")
        meta = oauth.get("auth_server_metadata_url_template")
        if meta is not None:
            if not isinstance(meta, str) or not meta:
                raise ValueError(
                    "oauth 'auth_server_metadata_url_template' must be a nonempty string or null"
                )
            # Claude requires https for the metadata URL. A LITERAL url is checked
            # here; a ${VAR} template stays unexpanded, and the resolved scheme is
            # Claude's to enforce - this parser cannot validate what it cannot see.
            if "${" not in meta and not meta.startswith("https://"):
                raise ValueError(
                    "oauth 'auth_server_metadata_url_template' must use https:// "
                    "(Claude rejects other schemes)"
                )
        scopes = oauth.get("scopes")
        if scopes is not None:
            if not isinstance(scopes, str) or not scopes.strip():
                raise ValueError(
                    "oauth 'scopes' must be the nonempty space-separated scope string or null"
                )
            scope_list = scopes.split()
            if len(scope_list) != len(set(scope_list)):
                raise ValueError(
                    "oauth 'scopes' carries duplicate scope entries; an ambiguous "
                    "authority set is refused"
                )
        oauth = {field: oauth.get(field) for field in _OAUTH_FIELDS}
    normalized: Dict[str, Any] = {
        "transport": transport,
        "url_template": url_template,
        "header_templates": {k: header_templates[k] for k in sorted(header_templates)},
        "headers_helper_template": helper,
    }
    if transport != "ws":
        # ws is header-only per the documented Claude surface; its canonical pinned
        # shape carries no oauth member at all (a ws spec with one refused above)
        normalized["oauth"] = oauth
    return normalized


def source_fingerprint(spec: Dict[str, Any]) -> str:
    """Fingerprint a launch specification: SHA-256 over its canonical normalized form."""
    return _sha256(normalize_source(spec))


def diff_source(
    server: str, pinned_entry: Dict[str, Any], observed: Dict[str, Any]
) -> List[Finding]:
    """Compare a server's observed launch specification against its pin, BEFORE launch.

    This is the pre-execution half of verification: a changed ``.mcp.json`` command must
    be refused *without starting the replacement process* - a post-execution catalog
    mismatch proves drift only after the substituted command already ran. CRITICAL on
    any mismatch (the changed fields are named), CRITICAL on a pin whose stored
    fingerprint does not match its own source (a hand-edited pin certifies nothing),
    and an affirmative ok finding when the identity holds.
    """
    pinned_source = pinned_entry.get("source")
    if not isinstance(pinned_source, dict):
        return [
            Finding.fail(
                "mcp_launch_spec_unpinned",
                severity="CRITICAL",
                message=f"server {server!r} has no pinned launch specification; refusing "
                "to launch what was never approved (re-pin with --approve-server-launch)",
                server=server,
            )
        ]
    pinned_norm = normalize_source(pinned_source)
    if pinned_entry.get("source_fingerprint") != _sha256(pinned_norm):
        return [
            Finding.fail(
                "mcp_launch_spec_corrupt",
                severity="CRITICAL",
                message=f"server {server!r}: source_fingerprint does not match the pinned "
                "source - a hand-edited or corrupt pin certifies nothing",
                server=server,
            )
        ]
    observed_norm = normalize_source(observed)
    if observed_norm == pinned_norm:
        return [
            Finding.ok(
                "mcp_launch_spec",
                severity="CRITICAL",
                message=f"server {server!r} launch specification matches the pin",
                server=server,
            )
        ]
    changed = [
        field
        for field in sorted(set(pinned_norm) | set(observed_norm))
        if pinned_norm.get(field) != observed_norm.get(field)
    ]
    return [
        Finding.fail(
            "mcp_launch_spec_changed",
            severity="CRITICAL",
            message=f"server {server!r} launch specification changed since the pin "
            f"(fields: {', '.join(changed)}); refusing WITHOUT executing the configured "
            "command - re-pin deliberately if this change is yours",
            server=server,
            changed_fields=changed,
        )
    ]


def instructions_record(observed: bool, text: Optional[str]) -> Dict[str, Any]:
    """The pinned shape of one server's initialize-result ``instructions``.

    Three states, never conflated: ``{"observed": false}`` (the observation did not
    carry instructions - a legacy dump - and the pin claims nothing about them);
    ``{"observed": true, "present": false}`` (the server was asked and declares none);
    ``{"observed": true, "present": true, "fingerprint": sha256}`` (declared, pinned as
    a hash - the text itself never enters the manifest).
    """
    if not observed:
        return {"observed": False}
    if text is None:
        return {"observed": True, "present": False}
    if not isinstance(text, str):
        raise ValueError(
            f"server instructions must be a string or absent, got {type(text).__name__}"
        )
    return {"observed": True, "present": True, "fingerprint": _sha256(text)}


def diff_instructions(
    server: str, pinned_entry: Dict[str, Any], observed: bool, text: Optional[str]
) -> List[Finding]:
    """Compare a server's observed instructions against the pin, as Findings.

    Under Claude's default tool-search behavior server instructions load at session
    start, so they are discovery-time influence: added, removed, or changed
    instructions under an unchanged tool catalog are CRITICAL drift. (Recusal
    fingerprints the COMPLETE observed instruction string; Claude truncates what it
    loads into context, currently at 2KB, so a change outside the loaded prefix still
    drifts here - the safe side of that asymmetry.) A pin whose observation never carried instructions holds the
    weaker claim honestly: verifying it with another instruction-blind observation
    passes with the boundary named, while an observation that NOW carries instructions
    refuses with a re-pin instruction - unreviewed influence content must not ride in
    on a capability upgrade.
    """
    pinned = pinned_entry.get("server_instructions")
    if not isinstance(pinned, dict):
        return [
            Finding.fail(
                "mcp_instructions_unpinned",
                severity="CRITICAL",
                message=f"server {server!r} has no pinned instructions record; re-pin "
                "with this recusal version",
                server=server,
            )
        ]
    current = instructions_record(observed, text)
    if not pinned.get("observed"):
        if not observed:
            return [
                Finding.ok(
                    "mcp_instructions",
                    severity="WARNING",
                    message=f"server {server!r}: instructions were not observed at pin "
                    "or verify (legacy dump); the pin does not cover them",
                    server=server,
                )
            ]
        return [
            Finding.fail(
                "mcp_instructions_unpinned",
                severity="CRITICAL",
                message=f"server {server!r} now presents instructions that were never "
                "pinned or reviewed; re-pin (with the rich observation format) so a "
                "human sees them first",
                server=server,
            )
        ]
    if not observed:
        return [
            Finding.fail(
                "mcp_instructions_unobserved",
                severity="CRITICAL",
                message=f"server {server!r} was pinned WITH instruction coverage but "
                "this observation does not carry instructions (legacy dump format?); "
                "supply the rich {'instructions': ..., 'tools': [...]} observation",
                server=server,
            )
        ]
    if current == pinned:
        return [
            Finding.ok(
                "mcp_instructions",
                severity="CRITICAL",
                message=f"server {server!r} instructions match the pin",
                server=server,
            )
        ]
    if pinned.get("present") and not current.get("present"):
        what = "removed its pinned instructions"
    elif not pinned.get("present") and current.get("present"):
        what = "added instructions that were never pinned"
    else:
        what = "changed its instructions (the discovery-influence rug pull)"
    return [
        Finding.fail(
            "mcp_instructions_changed",
            severity="CRITICAL",
            message=f"server {server!r} {what}; refusing until a human re-reviews and re-pins",
            server=server,
        )
    ]


def build_manifest(
    catalog: Dict[str, List[dict]],
    sources: Optional[Dict[str, Dict[str, Any]]] = None,
    instructions: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Build a manifest from ``{server_name: [tool declarations]}``.

    Deterministic: the same catalog always produces the same manifest (there is no
    timestamp inside; *when* a pin happened belongs to the audit log, not the artifact).
    Declarations are hashed, never stored; source templates are stored readable.
    Raises ``ValueError`` on a catalog
    that cannot be pinned unambiguously: empty, a nameless tool, or two tools with the
    same name on one server (an ambiguous catalog certifies nothing).

    ``sources`` maps a server name to its launch specification (see
    :func:`normalize_source`); a server without one is pinned as ``transport:
    "external"`` - its catalog is governed, its launch is not recusal's to govern.

    ``instructions`` maps a server name to its observed initialize-result
    ``instructions`` text (``None`` = the server was asked and declares none). A server
    absent from the mapping was NOT observed for instructions (a legacy dump), which is
    recorded as ``observed: false`` rather than silently claiming coverage.
    """
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("an empty catalog certifies nothing; nothing to pin")
    servers: Dict[str, Any] = {}
    for server, tools in catalog.items():
        if not isinstance(server, str) or not server:
            raise ValueError(f"server name must be a nonempty string, got {server!r}")
        if not isinstance(tools, list):
            raise ValueError(f"server {server!r}: tools must be a list of declarations")
        pinned_tools: Dict[str, Any] = {}
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                raise ValueError(f"server {server!r}: every tool needs a string 'name'")
            name = tool["name"]
            if not name:
                # the loader refuses empty stored names; building an artifact the
                # loader would reject is a pin that certifies nothing
                raise ValueError(f"server {server!r}: a tool name must be nonempty")
            if name in pinned_tools:
                raise ValueError(
                    f"server {server!r} declares tool {name!r} twice; an ambiguous "
                    "catalog cannot be pinned"
                )
            pinned_tools[name] = {
                "fingerprint": tool_fingerprint(tool),
                "fields": {k: _sha256(tool[k]) for k in _TRACKED_FIELDS if k in tool},
            }
        # missing source -> external by design; a SUPPLIED source is validated as
        # given, so an explicit-but-invalid one (e.g. {}) raises instead of silently
        # downgrading to external
        if sources is None or server not in sources:
            source = {"transport": "external"}
        else:
            source = normalize_source(sources[server])
        source = normalize_source(source)
        if instructions is not None and server in instructions:
            server_instructions = instructions_record(True, instructions[server])
        else:
            server_instructions = instructions_record(False, None)
        servers[server] = {
            "source": source,
            "source_fingerprint": _sha256(source),
            "server_instructions": server_instructions,
            "tools": pinned_tools,
        }
    return {"manifest_version": MANIFEST_VERSION, "servers": servers}


def manifest_to_text(manifest: Dict[str, Any]) -> str:
    """The manifest's on-disk form; deterministic bytes so a re-pin diff is meaningful."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def load_manifest(path: str) -> Dict[str, Any]:
    """Load and shape-check a pinned manifest; raises ``ValueError`` on anything off."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    _validate_manifest(data)
    return data


#: Exact shape of every stored digest: the algorithm prefix and 64 lowercase hex chars.
#: Anything else in a digest position is corruption or hand-editing, both refused.
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _validate_manifest(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("manifest is not a JSON object")
    if data.get("manifest_version") == 1:
        raise ValueError(
            "manifest_version 1 predates launch-identity pinning (it covers the declared "
            "catalog but not the process that declares it); re-pin with `recusal mcp pin` "
            "to record the launch specifications"
        )
    if data.get("manifest_version") == 2:
        raise ValueError(
            "manifest_version 2 predates environment-template and remote-source pinning "
            "(a same-key env value swap or an added remote server could pass it); re-pin "
            "with `recusal mcp pin` to record the complete source identities"
        )
    if data.get("manifest_version") == 3:
        raise ValueError(
            "manifest_version 3 predates remote authentication identity (a same-name "
            "header template swap, a changed headersHelper command, or a changed OAuth "
            "scope set could pass it); re-pin with `recusal mcp pin` to record the "
            "complete remote source identities"
        )
    if data.get("manifest_version") == 4:
        raise ValueError(
            "manifest_version 4 predates server-instruction pinning (a server could "
            "keep its tools byte-identical and rewrite only its initialize-result "
            "instructions, a discovery-time influence surface Claude loads at session "
            "start); re-pin with `recusal mcp pin` to record instruction coverage"
        )
    if data.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            f"manifest_version {data.get('manifest_version')!r} is not {MANIFEST_VERSION}"
        )
    unknown_top = set(data) - {"manifest_version", "servers"}
    if unknown_top:
        raise ValueError(
            f"manifest carries fields this loader does not define: {sorted(unknown_top)} - "
            "a deterministic control artifact must not carry ignored fields whose "
            "meaning is undefined"
        )
    servers = data.get("servers")
    if not isinstance(servers, dict) or not servers:
        raise ValueError("manifest has no servers")
    for server, entry in servers.items():
        if not isinstance(server, str) or not server:
            raise ValueError(f"manifest server name must be a nonempty string, got {server!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"manifest server {server!r} entry is not an object")
        unknown_entry = set(entry) - {
            "source",
            "source_fingerprint",
            "server_instructions",
            "tools",
        }
        if unknown_entry:
            raise ValueError(
                f"manifest server {server!r} carries undefined fields: {sorted(unknown_entry)}"
            )
        try:
            source = normalize_source(entry.get("source") or {})
        except ValueError as exc:
            raise ValueError(f"manifest server {server!r}: {exc}") from exc
        fingerprint = entry.get("source_fingerprint")
        if not isinstance(fingerprint, str) or not _DIGEST_RE.fullmatch(fingerprint):
            raise ValueError(
                f"manifest server {server!r} source_fingerprint is not sha256:<64 lowercase hex>"
            )
        if fingerprint != _sha256(source):
            raise ValueError(
                f"manifest server {server!r}: source_fingerprint does not match the "
                "pinned source - a hand-edited or corrupt pin certifies nothing"
            )
        record = entry.get("server_instructions")
        if not isinstance(record, dict) or not isinstance(record.get("observed"), bool):
            raise ValueError(f"manifest server {server!r} has no server_instructions record")
        # Exactly the three canonical shapes instructions_record() emits, nothing else:
        # a contradictory encoding ("present" under observed:false, a fingerprint under
        # present:false, an extra key) is one the builder never writes, so accepting it
        # would give an undefined record a defined meaning.
        if not record["observed"]:
            if set(record) != {"observed"}:
                raise ValueError(
                    f"manifest server {server!r} server_instructions is not canonical: "
                    'an unobserved record is exactly {"observed": false}'
                )
        elif not isinstance(record.get("present"), bool):
            raise ValueError(
                f"manifest server {server!r} server_instructions needs a boolean 'present'"
            )
        elif record["present"]:
            if set(record) != {"observed", "present", "fingerprint"} or not (
                isinstance(record.get("fingerprint"), str)
                and _DIGEST_RE.fullmatch(record["fingerprint"])
            ):
                raise ValueError(
                    f"manifest server {server!r} server_instructions is not canonical: "
                    'a present record is exactly {"observed": true, "present": true, '
                    '"fingerprint": sha256:<64 lowercase hex>}'
                )
        elif set(record) != {"observed", "present"}:
            raise ValueError(
                f"manifest server {server!r} server_instructions is not canonical: an "
                'observed-but-absent record is exactly {"observed": true, "present": false}'
            )
        tools = entry.get("tools")
        if not isinstance(tools, dict):
            raise ValueError(f"manifest server {server!r} has no tools object")
        for name, pin in tools.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"manifest server {server!r} pins a tool with a non-string/empty name"
                )
            if not isinstance(pin, dict) or not isinstance(pin.get("fingerprint"), str):
                raise ValueError(f"manifest tool {server!r}/{name!r} has no fingerprint")
            # canonical pin shape, exactly what the builder emits: BOTH members present
            # ('fields' may be empty, never absent), nothing else
            if set(pin) != {"fingerprint", "fields"}:
                raise ValueError(
                    f"manifest tool {server!r}/{name!r} is not canonical: a pin is exactly "
                    "{'fingerprint': ..., 'fields': {...}}"
                )
            if not _DIGEST_RE.fullmatch(pin["fingerprint"]):
                raise ValueError(
                    f"manifest tool {server!r}/{name!r} fingerprint is not "
                    "sha256:<64 lowercase hex> - a corrupt pin certifies nothing"
                )
            fields = pin["fields"]
            if not isinstance(fields, dict):
                raise ValueError(
                    f"manifest tool {server!r}/{name!r} has a non-object 'fields' entry"
                )
            unknown_fields = set(fields) - set(_TRACKED_FIELDS)
            if unknown_fields:
                raise ValueError(
                    f"manifest tool {server!r}/{name!r} 'fields' carries names outside the "
                    f"tracked diagnostic set {sorted(_TRACKED_FIELDS)}: "
                    f"{sorted(map(repr, unknown_fields))} - an undefined field hash has no "
                    "defined diagnostic meaning"
                )
            for field, digest in fields.items():
                if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
                    raise ValueError(
                        f"manifest tool {server!r}/{name!r} field {field!r} hash is not "
                        "sha256:<64 lowercase hex>"
                    )


# --- the verify: observed catalog vs pin -> Findings --------------------------------------


def diff_manifest(
    pinned: Dict[str, Any],
    observed: Dict[str, List[dict]],
    *,
    unverifiable: Sequence[str] = (),
) -> List[Finding]:
    """Compare a freshly observed TOOL CATALOG against the pinned manifest, as Findings.

    **Catalog-only by design**: this compares tool declarations. It does NOT compare
    source specifications or server instructions, both of which a manifest v5 also
    pins; a clean result here says "the declared tools match", nothing more. For the
    complete, omission-resistant v5 verify use :func:`diff_observation`, which is what
    ``recusal mcp verify`` runs.

    - an observed server or tool that was never pinned → CRITICAL (unapproved capability);
    - a pinned tool whose declaration changed → CRITICAL, naming the changed fields (a
      changed ``description`` is the rug-pull vector);
    - a tool declared twice by one server → CRITICAL (ambiguous, possibly shadowed);
    - a pinned tool or server that is absent → WARNING (recorded, not blocking: a shrunk
      capability set is not an attack, and the swap-in-a-replacement half already fails);
      a server observed with an *empty* tool list is exactly this case, its pinned tools
      read as removed, consistent with ``build_manifest`` accepting an empty tool list;
    - a pinned server named in ``unverifiable`` → CRITICAL: it is declared but could not be
      reached for integrity-checking (e.g. silently swapped to a URL transport the fetcher
      cannot observe), and a pinned capability that cannot be checked must not verify clean.
      ``unverifiable`` is *collected* by the caller (it comes from config parsing); the
      adjudication of it lives here so every consumer of the kernel gets the rule;
    - a *wholly* empty observation (no servers named at all) → CRITICAL (nothing was
      observed; a failed fetch must not read as "no drift"). A live fetch that fails raises
      ``McpFetchError`` before reaching here, so this guards a user-supplied empty dump.

    A clean comparison returns an affirmative ok Finding, never an empty list, because an
    empty evidence set certifies nothing.
    """
    _validate_manifest(pinned)
    pinned_servers: Dict[str, Any] = pinned["servers"]

    # A pinned server that is declared but unreachable for integrity-checking cannot verify
    # clean, even if the rest of the catalog matches. (Collected by the caller; adjudicated
    # here so a programmatic caller of diff_manifest gets the rule for free.)
    findings: List[Finding] = [
        Finding.fail(
            "mcp_pinned_server_unverifiable",
            severity="CRITICAL",
            message=f"pinned server '{name}' is declared but could not be reached for "
            "integrity-checking (a non-stdio/URL transport this fetcher cannot observe); "
            "a pinned capability that cannot be integrity-checked must not verify clean",
            server=name,
        )
        for name in sorted(set(unverifiable) & set(pinned_servers))
    ]

    if not isinstance(observed, dict) or not observed:
        findings.append(
            Finding.fail(
                "mcp_manifest",
                severity="CRITICAL",
                message="observed catalog is empty; an empty observation certifies nothing "
                "(did the fetch fail?)",
            )
        )
        return findings

    checked = 0
    for server, tools in observed.items():
        if not isinstance(tools, list):
            # a malformed catalog value must refuse, never be truthiness-normalized to
            # an empty tool list (which would read as a shrunk set: a mere WARNING)
            findings.append(
                Finding.fail(
                    "mcp_malformed_catalog",
                    severity="CRITICAL",
                    message=f"server '{server}' catalog entry is {type(tools).__name__}, "
                    "not a list of tool declarations; a malformed observation certifies "
                    "nothing",
                    server=server,
                )
            )
            continue
        seen: Dict[str, dict] = {}
        for tool in tools:
            name = tool.get("name") if isinstance(tool, dict) else None
            if not isinstance(name, str) or not name:
                findings.append(
                    Finding.fail(
                        "mcp_malformed_tool",
                        severity="CRITICAL",
                        message=f"server '{server}' declares a tool with no name",
                        server=server,
                    )
                )
                continue
            if name in seen:
                findings.append(
                    Finding.fail(
                        "mcp_duplicate_tool",
                        severity="CRITICAL",
                        message=f"server '{server}' declares tool '{name}' more than once; "
                        "an ambiguous catalog may shadow the pinned tool",
                        server=server,
                        tool=name,
                    )
                )
                continue
            seen[name] = tool

        pinned_entry = pinned_servers.get(server)
        if pinned_entry is None:
            findings.append(
                Finding.fail(
                    "mcp_unpinned_server",
                    severity="CRITICAL",
                    message=f"server '{server}' ({len(seen)} tool(s)) is not in the pinned "
                    "manifest; an unapproved server's tools must not reach the agent",
                    server=server,
                )
            )
            continue

        pinned_tools: Dict[str, Any] = pinned_entry["tools"]
        for name, tool in seen.items():
            pin = pinned_tools.get(name)
            if pin is None:
                findings.append(
                    Finding.fail(
                        "mcp_unpinned_tool",
                        severity="CRITICAL",
                        message=f"tool '{name}' on server '{server}' is not pinned; "
                        "a capability that was never approved must not reach the agent",
                        server=server,
                        tool=name,
                    )
                )
                continue
            checked += 1
            if tool_fingerprint(tool) != pin["fingerprint"]:
                changed = [
                    field
                    for field in _TRACKED_FIELDS
                    if pin.get("fields", {}).get(field)
                    != (_sha256(tool[field]) if field in tool else None)
                ]
                what = ", ".join(changed) if changed else "an untracked field"
                rug = (
                    " (a changed description is the rug-pull vector)"
                    if "description" in changed
                    else ""
                )
                findings.append(
                    Finding.fail(
                        "mcp_tool_changed",
                        severity="CRITICAL",
                        message=f"tool '{name}' on server '{server}' changed after it was "
                        f"pinned: {what}{rug}; re-review and re-pin deliberately",
                        server=server,
                        tool=name,
                        changed_fields=changed,
                    )
                )
        for name in pinned_tools:
            if name not in seen:
                findings.append(
                    Finding.fail(
                        "mcp_tool_removed",
                        severity="WARNING",
                        message=f"pinned tool '{name}' on server '{server}' is no longer "
                        "declared (recorded; a shrunk capability set is not refused)",
                        server=server,
                        tool=name,
                    )
                )

    for server in pinned_servers:
        if server not in observed:
            findings.append(
                Finding.fail(
                    "mcp_server_absent",
                    severity="WARNING",
                    message=f"pinned server '{server}' was not observed (offline or removed; "
                    "recorded, not refused)",
                    server=server,
                )
            )

    if not any(not f.passed and f.severity.value in ("CRITICAL", "ERROR") for f in findings):
        n_servers = sum(1 for s in observed if observed.get(s))
        findings.append(
            Finding.ok(
                "mcp_manifest",
                severity="CRITICAL",
                message=f"{checked} tool declaration(s) across {n_servers} server(s) match "
                "the pinned manifest",
            )
        )
    return findings


class McpObservation(NamedTuple):
    """One COMPLETE observation of the MCP surface a manifest v5 pins.

    A strict contract, not a convenience bag: :func:`diff_observation` refuses an
    observation that is malformed or incomplete rather than comparing the subset it
    happens to carry, because a partial comparison silently reads as a full one.

    ``catalog`` maps every observed server name to its declared tool list.
    ``sources`` maps EVERY catalog server to its source specification (see
    :func:`normalize_source`; every value is structurally validated before any
    comparison); a dump-supplied catalog carries the explicit weak claim
    ``{"transport": "external"}``, never an omission. ``instructions`` maps EVERY
    catalog server to exactly ``{"observed": bool, "text": str | None}`` (``observed``
    a real bool; ``text`` None when ``observed`` is false). ``unverifiable`` names
    pinned servers that were declared but could not be observed at all. ``removed``
    names pinned servers whose removal a human deliberately acknowledges: a pinned
    server absent from EVERY component and not named here is a CRITICAL refusal
    (``mcp_server_unobserved``) - a partial observation must never read as a full
    one. ``unverifiable`` and ``removed`` must be lists or tuples of unique nonempty
    names; any represented server missing from ``sources`` or ``instructions`` is a
    CRITICAL refusal.
    """

    catalog: Dict[str, List[dict]]
    sources: Optional[Dict[str, Dict[str, Any]]] = None
    instructions: Optional[Dict[str, Dict[str, Any]]] = None
    unverifiable: Sequence[str] = ()
    removed: Sequence[str] = ()


def _validate_observation(observation: McpObservation) -> None:
    """Structural validation of the observation itself; raises ``ValueError``.

    A malformed observation is not evidence: comparing it would launder shape bugs
    (a truthy string ``"false"``, a falsey ``{}`` catalog value) into drift verdicts.
    Coverage gaps (a server missing a component) are adjudicated as Findings by
    :func:`diff_observation`; *shape* violations refuse outright here.
    """
    if not isinstance(observation.catalog, dict):
        raise ValueError("observation catalog must be a dict of {server: [tool, ...]}")
    for server, tools in observation.catalog.items():
        if not isinstance(server, str) or not server:
            raise ValueError(
                f"observation catalog server name must be a nonempty string, got {server!r}"
            )
        if not isinstance(tools, list):
            raise ValueError(
                f"observation catalog for server {server!r} must be a list of tool "
                f"declarations, got {type(tools).__name__} - a malformed catalog value "
                "must refuse, never normalize to an empty tool list"
            )
    for label, mapping in (
        ("sources", observation.sources),
        ("instructions", observation.instructions),
    ):
        if mapping is None:
            continue
        if not isinstance(mapping, dict):
            raise ValueError(f"observation {label} must be a dict or None")
        for server in mapping:
            if not isinstance(server, str) or not server:
                raise ValueError(
                    f"observation {label} server name must be a nonempty string, got {server!r}"
                )
    for server, record in (observation.instructions or {}).items():
        if not isinstance(record, dict) or set(record) != {"observed", "text"}:
            raise ValueError(
                f"instruction observation for server {server!r} must be exactly "
                "{'observed': bool, 'text': str|None}"
            )
        if not isinstance(record["observed"], bool):
            raise ValueError(
                f"instruction observation for server {server!r}: 'observed' must be a real "
                f"boolean, got {record['observed']!r} - truthiness is not observation"
            )
        if record["text"] is not None and not isinstance(record["text"], str):
            raise ValueError(
                f"instruction observation for server {server!r}: 'text' must be a string or None"
            )
        if not record["observed"] and record["text"] is not None:
            raise ValueError(
                f"instruction observation for server {server!r} is contradictory: "
                "unobserved with text"
            )
    for server, source in (observation.sources or {}).items():
        if not isinstance(source, dict):
            raise ValueError(
                f"source observation for server {server!r} must be an object, got "
                f"{type(source).__name__}"
            )
        try:
            # structural validity is independent of whether the manifest pins the
            # server: a malformed source on an unpinned component-only server is
            # still a malformed observation, not evidence
            normalize_source(source)
        except ValueError as exc:
            raise ValueError(f"source observation for server {server!r}: {exc}") from exc
    for label, seq in (
        ("unverifiable", observation.unverifiable),
        ("removed", observation.removed),
    ):
        if not isinstance(seq, (list, tuple)):
            # a string would iterate to characters, a dict to keys, None to a
            # TypeError: generic iteration is not the declared contract
            raise ValueError(
                f"observation {label} must be a list or tuple of unique nonempty "
                f"server names, got {type(seq).__name__}"
            )
        names = list(seq)
        # element types BEFORE duplicate detection: an unhashable member (a list, a
        # dict) would raise TypeError from set() before its type was rejected, leaking
        # a generic container error where the contract promises ValueError
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"observation {label} must be unique nonempty server names")
        if len(set(names)) != len(names):
            raise ValueError(f"observation {label} carries duplicate server names")
    represented = (
        set(observation.catalog)
        | set(observation.sources or {})
        | set(observation.instructions or {})
        | set(observation.unverifiable)
    )
    contradictory = set(observation.removed) & represented
    if contradictory:
        raise ValueError(
            f"observation marks server(s) {sorted(contradictory)} as removed while also "
            "representing them; a removal acknowledgement contradicts an observation"
        )


def diff_observation(pinned: Dict[str, Any], observation: McpObservation) -> List[Finding]:
    """The complete manifest-v5 verify: every pinned surface, one call, no silent subset.

    Omission-resistance, stated exactly: for every server the observation REPRESENTS
    (in ``catalog``, ``sources``, ``instructions``, or ``unverifiable``), all three
    surfaces are adjudicated, and a missing component is a CRITICAL refusal, never a
    weaker comparison. And a pinned server absent from EVERY component is a CRITICAL
    refusal too (``mcp_server_unobserved``) unless a human deliberately names it in
    ``removed``: the manifest is approved truth, so a server-set change is reviewed,
    never inferred from an incomplete observation. (A ``removed`` acknowledgement is
    recorded as a passing WARNING naming the deliberate transition; re-pin to make
    the shrunk server set the new approved truth. The catalog-only
    :func:`diff_manifest` primitive keeps its absent-server WARNING, because it
    claims only catalog comparison.)

    1. the observation's own shape is validated (:func:`_validate_observation`, a
       ``ValueError`` on malformation - a truthy non-bool ``observed``, a non-list
       catalog value, duplicate ``unverifiable`` names);
    2. the manifest is validated (:func:`load_manifest` shape rules);
    3. every represented server missing from the pin is a CRITICAL refusal, whichever
       component it appears in (an unapproved server must not ride in as
       source-only or instruction-only);
    4. every catalog server's SOURCE observation is compared (:func:`diff_source`);
       a catalog server with no source observation is a CRITICAL refusal - supply
       the explicit weak claim ``{"transport": "external"}`` for dump catalogs;
    5. every catalog server's INSTRUCTION state is compared
       (:func:`diff_instructions`); a catalog server with no instruction record is a
       CRITICAL refusal - the legacy weaker claim is the explicit
       ``{"observed": False, "text": None}``, never an omission;
    6. a pinned server represented only by source/instructions (no catalog entry) is
       a CRITICAL refusal unless it is named ``unverifiable``;
    7. a pinned server absent from every component is a CRITICAL refusal unless
       named ``removed`` (whose acknowledgement is a recorded, passing WARNING);
       ``removed`` may name only pinned servers and contradicts any representation -
       EXCEPT that naming every pinned server with nothing represented at all refuses
       with ``mcp_full_decommission_unsupported`` (an empty observation certifies
       nothing; decommission ALL MCP capability by removing the manifest itself);
    8. the tool catalog is compared (:func:`diff_manifest`), including the
       ``unverifiable`` adjudication.

    ``recusal mcp verify`` routes through this function; call the lower-level
    primitives directly only when you deliberately want a partial comparison, and
    name that choice in your own claim.
    """
    _validate_observation(observation)
    _validate_manifest(pinned)
    pinned_servers: Dict[str, Any] = pinned["servers"]
    sources = observation.sources or {}
    instructions = observation.instructions or {}
    unverifiable = set(observation.unverifiable)
    removed = set(observation.removed)

    unknown_removed = removed - set(pinned_servers)
    if unknown_removed:
        raise ValueError(
            f"observation marks server(s) {sorted(unknown_removed)} as removed, but the "
            "manifest does not pin them; a removal acknowledgement must name a pinned server"
        )

    findings: List[Finding] = []

    represented = set(observation.catalog) | set(sources) | set(instructions) | unverifiable
    for name in sorted(represented - set(pinned_servers) - set(observation.catalog)):
        # unpinned AND not in the catalog: diff_manifest never sees it, so the refusal
        # must come from here (catalog-borne unpinned servers get diff_manifest's own)
        findings.append(
            Finding.fail(
                "mcp_unpinned_server",
                severity="CRITICAL",
                message=f"server '{name}' appears in the observation "
                "(source/instructions/unverifiable) but is not in the pinned manifest; "
                "an unapproved server must not ride in outside the catalog",
                server=name,
            )
        )

    for name in sorted(observation.catalog):
        source = sources.get(name)
        if source is None:
            findings.append(
                Finding.fail(
                    "mcp_source_unobserved",
                    severity="CRITICAL",
                    message=f"server '{name}' was observed with no source observation; a "
                    "complete manifest-v5 verify requires one (a dump-supplied catalog's "
                    "explicit weak claim is {'transport': 'external'}, never an omission)",
                    server=name,
                )
            )
        elif isinstance(pinned_servers.get(name), dict):
            findings.extend(diff_source(name, pinned_servers[name], source))
        record = instructions.get(name)
        if record is None:
            findings.append(
                Finding.fail(
                    "mcp_instructions_unobserved",
                    severity="CRITICAL",
                    message=f"server '{name}' was observed with no instruction record; a "
                    "complete manifest-v5 verify requires one (the legacy weaker claim is "
                    "the explicit {'observed': False, 'text': None}, never an omission)",
                    server=name,
                )
            )
        elif isinstance(pinned_servers.get(name), dict):
            findings.extend(
                diff_instructions(name, pinned_servers[name], record["observed"], record["text"])
            )

    for name in sorted((set(sources) | set(instructions)) - set(observation.catalog)):
        if name in pinned_servers and name not in unverifiable:
            findings.append(
                Finding.fail(
                    "mcp_observation_incomplete",
                    severity="CRITICAL",
                    message=f"server '{name}' has a source/instruction observation but no "
                    "catalog; a pinned server with a partial observation must be named "
                    "unverifiable or observed completely, never adjudicated as merely absent",
                    server=name,
                )
            )

    represented_all = set(observation.catalog) | set(sources) | set(instructions) | unverifiable
    for name in sorted(set(pinned_servers) - represented_all - removed):
        findings.append(
            Finding.fail(
                "mcp_server_unobserved",
                severity="CRITICAL",
                message=f"pinned server '{name}' is absent from every component of this "
                "observation and was not explicitly marked removed; a partial observation "
                "must not verify clean while the manifest keeps authorizing the server's "
                "pinned runtime names - observe it, name it unverifiable, or acknowledge "
                "its removal and re-pin",
                server=name,
            )
        )
    for name in sorted(removed):
        findings.append(
            Finding.fail(
                "mcp_server_removed",
                severity="WARNING",
                message=f"pinned server '{name}' is acknowledged as deliberately removed "
                "(recorded, not blocking); re-pin so the manifest stops authorizing its "
                "runtime names",
                server=name,
            )
        )

    if removed and removed == set(pinned_servers) and not represented_all:
        # Full decommission: every pinned server acknowledged removed, nothing observed.
        # ``removed`` supports transitions where at least one pinned server remains
        # observable; an empty observation certifies nothing, so this refuses with the
        # PRECISE reason instead of the generic empty-observation message (and instead
        # of a passing verdict a stale manifest would keep contradicting: the old
        # manifest still authorizes every runtime name until it is replaced or gone).
        findings.append(
            Finding.fail(
                "mcp_full_decommission_unsupported",
                severity="CRITICAL",
                message="every pinned server is acknowledged as removed and nothing was "
                "observed; an empty observation certifies nothing, and the manifest keeps "
                "authorizing all pinned runtime names regardless of this acknowledgement - "
                "to decommission ALL MCP capability, remove or replace the manifest itself "
                "(manifest_policy fails closed: no pin, no MCP)",
            )
        )
        return findings

    findings.extend(
        diff_manifest(pinned, observation.catalog, unverifiable=tuple(observation.unverifiable))
    )
    return findings


# --- pin-time declaration screen (a review aid, not a malice detector) --------------------

#: High-signal injection phrasing seen in poisoned tool declarations. A deny-list with a
#: deny-list's ceiling: it surfaces the obvious for human review; it cannot clear novel or
#: paraphrased poison, and it never substitutes for reading the declarations at pin time.
DECLARATION_MARKERS: Tuple[str, ...] = (
    "ignore previous instructions",
    "disregard the above",
    "ignore the system prompt",
    "new instructions:",
    "do not tell the user",
    "do not inform the user",
    "without telling the user",
    "hide this from the user",
    "before using this tool, you must",
    "instead, use the tool",
    "send the api key",
    "send the contents of",
    "exfiltrate",
    "<important>",
    "~/.ssh",
    "id_rsa",
)

#: A tool whose declared text is too long to plausibly be reviewed by a human at pin time
#: is itself a finding: oversized declarations are where poisoned instructions hide.
MAX_DECLARED_CHARS = 4000

#: The depth analogue of the size cap: a declaration nested deeper than this cannot
#: plausibly be reviewed either, so it is a finding, and the walk stops descending there.
#: Legitimate JSON Schemas nest tens of levels at the most.
MAX_DECLARED_DEPTH = 200


def _declared_text(value: Any, *, max_depth: int = MAX_DECLARED_DEPTH) -> Tuple[List[str], bool]:
    """Every human-language string the model may read in a tool declaration, plus whether
    the declaration nests past ``max_depth``.

    Instructions do not only hide in the top-level ``description``: a poisoned
    ``inputSchema`` property description, an ``enum`` value, a ``title``, an
    ``annotations`` note, or even a property *name* (a dict key the model reads) is seen by
    the model just the same. The screen walks all of it, dict keys and values and list
    items, rather than a single field, so a deny-list ceiling is the only limit, not a
    blind spot.

    The walk is iterative (an explicit stack, depth-first in declaration order): a hostile
    server must not be able to crash the screen out of returning a verdict with a
    thousands-deep schema, and a crash is not a refusal. Past ``max_depth`` the walk
    records the excess and stops descending, which also bounds a self-referencing
    (non-JSON) input instead of looping on it.
    """
    out: List[str] = []
    too_deep = False
    stack: List[Tuple[Any, int]] = [(value, 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            if depth >= max_depth:
                too_deep = True
                continue
            for k, v in reversed(list(node.items())):
                stack.append((v, depth + 1))
                if isinstance(k, str):
                    stack.append((k, depth + 1))
        elif isinstance(node, (list, tuple)):
            if depth >= max_depth:
                too_deep = True
                continue
            for v in reversed(node):
                stack.append((v, depth + 1))
    return out, too_deep


def screen_server_instructions(
    instructions: Dict[str, Optional[str]],
    *,
    markers: Sequence[str] = DECLARATION_MARKERS,
    max_chars: int = MAX_DECLARED_CHARS,
) -> List[Finding]:
    """The pin-time review screen for initialize-result server instructions.

    The same bounded deny-list marker scan and size cap as the declaration screen, for
    the same reason: instructions are discovery-time model-facing text. ERROR routes to
    human review; this is never semantic malice detection.
    """
    findings: List[Finding] = []
    for server, text in sorted(instructions.items()):
        if text is None:
            continue
        low = text.lower()
        hits = [m for m in markers if m in low]
        if hits:
            findings.append(
                Finding.fail(
                    "mcp_instructions_marker",
                    severity="ERROR",
                    message=f"server {server!r} instructions carry injection phrasing: "
                    f"{hits[0]!r}; review before pinning",
                    server=server,
                    markers=hits,
                )
            )
        if len(text) > max_chars:
            findings.append(
                Finding.fail(
                    "mcp_instructions_size",
                    severity="ERROR",
                    message=f"server {server!r} declares {len(text)} chars of "
                    f"instructions (cap {max_chars}); too large to plausibly review is "
                    "itself a review flag",
                    server=server,
                )
            )
    return findings


def screen_tool_declarations(
    catalog: Dict[str, List[dict]],
    *,
    markers: Sequence[str] = DECLARATION_MARKERS,
    max_chars: int = MAX_DECLARED_CHARS,
) -> List[Finding]:
    """Deterministically screen every declared string before a pin; ERROR routes to review.

    Screens the whole declaration, not just ``description`` (which is why it is named for
    declarations, not descriptions): ``title``, ``annotations``, and the strings inside
    ``inputSchema`` / ``outputSchema`` (property names, property descriptions, enum values)
    are all read by the model and so are all scanned (see :func:`_declared_text`).

    ERROR (RETRY), deliberately not CRITICAL: a marker hit means "a human must look",
    not "provably malicious". ``recusal mcp pin`` refuses to write on a non-clean screen
    unless ``--force`` records that a human reviewed and accepted it.
    """
    findings: List[Finding] = []
    screened = 0
    for server, tools in (catalog or {}).items():
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "?"))
            screened += 1
            texts, too_deep = _declared_text(tool)
            if too_deep:
                findings.append(
                    Finding.fail(
                        "mcp_declaration_depth",
                        severity="ERROR",
                        message=f"tool '{name}' on server '{server}' nests its declaration "
                        f"deeper than {MAX_DECLARED_DEPTH} levels; too deep to plausibly "
                        "review is itself a review flag",
                        server=server,
                        tool=name,
                    )
                )
            low = "\n".join(texts).lower()
            hits = [m for m in markers if m in low]
            if hits:
                findings.append(
                    Finding.fail(
                        "mcp_declaration_marker",
                        severity="ERROR",
                        message=f"tool '{name}' on server '{server}' carries injection "
                        f"phrasing in its declaration: {hits[0]!r}; review before pinning",
                        server=server,
                        tool=name,
                        markers=hits,
                    )
                )
            total = sum(len(t) for t in texts)
            if total > max_chars:
                findings.append(
                    Finding.fail(
                        "mcp_declaration_size",
                        severity="ERROR",
                        message=f"tool '{name}' on server '{server}' declares {total} chars "
                        f"of text (cap {max_chars}); too large to plausibly review is itself "
                        "a review flag",
                        server=server,
                        tool=name,
                    )
                )
    if not findings:
        findings.append(
            Finding.ok(
                "mcp_declaration_screen",
                severity="ERROR",
                message=f"{screened} declaration(s) screened, no injection markers",
            )
        )
    return findings


# --- call-time bridge: enforce the pin inside the existing PreToolUse gate ----------------


def manifest_policy(
    manifest_path: str,
    *,
    policy: Optional[Callable[[str, dict], List[Any]]] = None,
) -> Callable[[str, dict], List[Any]]:
    """A ``policy(tool_name, tool_input)`` that refuses MCP calls not in the pinned manifest.

    - a non-MCP tool name (one that does not start with ``mcp__``) is handed to the wrapped
      ``policy`` (or deferred if none);
    - an ``mcp__server__tool`` call not present in the pinned manifest → CRITICAL refusal,
      and the wrapped ``policy`` is NOT invoked for it (nor when the manifest is missing
      or corrupt): membership is established first, so an unapproved capability never
      triggers downstream policy work or its side effects;
    - a pinned call is then handed to the wrapped ``policy``, so argument-level rules
      (repo scope, path confinement, cookbook recipe 12) compose on top of the pin;
    - a missing or malformed manifest fails CLOSED for MCP calls: no pin, no MCP;
    - the manifest bytes are read on every MCP call and reparsed only when their SHA-256
      changes: a re-pin (or a REVOCATION) is picked up on the very next call, even one
      written with a preserved timestamp and identical size, and there is no
      stat-then-open race - authorization is keyed on the exact bytes read.

    Membership is checked by the *full* runtime name (``mcp__{server}__{tool}``,
    reconstructed from the pin), never by re-splitting the incoming name, so this never
    mis-*attributes* one pinned tool's approval to a differently-named call.

    Honest limits:

    - A PreToolUse event carries the name and input, not the declaration, so this enforces
      *approved tools only*; declaration integrity is ``recusal mcp verify``'s job (CI,
      session start, or a cron). Between verifies, a server that discriminates by client
      can serve a clean catalog to ``verify`` and a poisoned one to the live session, and a
      name-only call-time check cannot see that, run ``verify`` against the same endpoint
      the session uses, close in time.
    - MCP's runtime naming is *flat* (``mcp__server__tool``), so ``mcp__a__b__c`` is the
      only string both ``(server a, tool b__c)`` and ``(server a__b, tool c)`` can produce.
      Pinning either therefore authorizes a call that either would emit. This is inherent to
      the flat namespace, not removable at this layer; server names with ``__`` are the only
      case, and both readings collapse to the same capability.
    """

    # Content-digest cache: the manifest BYTES are read on every MCP call (the file is
    # small by design) and reparsed only when their SHA-256 changes. Unlike an
    # (mtime, size) signature, this can never serve stale authorization: a same-size
    # replacement with a preserved timestamp - exactly how a REVOCATION might land via
    # deployment tooling - is a different digest, and there is no stat-then-open race
    # because the decision is keyed on the exact bytes that were read. A failed parse is
    # never cached, so a changed-but-corrupt file refuses on this call and every next
    # one. The (digest, names) pair is ONE immutable value swapped under a lock, so a
    # multi-threaded runtime can never observe one manifest's digest associated with
    # another manifest's name set; the returned names are always derived from the bytes
    # THIS call read, never from the cache of a racing call.
    _cache_lock = threading.Lock()
    _cache: List[Tuple[Optional[str], "frozenset[str]"]] = [(None, frozenset())]
    # Audit provenance is INVOCATION-local, carried in a ContextVar rather than a
    # mutable attribute on the policy object: an attribute would be shared state, and
    # under concurrent reuse of one policy object (threads, asyncio) invocation A's
    # audit record could read invocation B's digest, or a clear could erase a digest
    # another invocation was about to record. A ContextVar isolates per thread and per
    # asyncio task, so each adjudication's audit record sees exactly the digest THAT
    # adjudication verified. The audit layer reads it through ``get_control_identity``.
    _digest_var: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
        f"recusal_manifest_digest_{id(_cache)}", default=None
    )

    def _pinned_names() -> "frozenset[str]":
        with open(manifest_path, "rb") as fh:
            raw = fh.read()
        digest = hashlib.sha256(raw).hexdigest()
        with _cache_lock:
            cached_digest, cached_names = _cache[0]
        if cached_digest == digest:
            # a cache hit means these exact bytes were validated before
            _digest_var.set(f"sha256:{digest}")
            return cached_names
        data = json.loads(raw.decode("utf-8"))  # parse outside the lock
        _validate_manifest(data)
        # audit control identity, recorded ONLY after successful parse + validation: a
        # corrupt manifest must never be recorded as a successfully enforced one
        _digest_var.set(f"sha256:{digest}")
        names = frozenset(
            f"mcp__{server}__{tool}"
            for server, entry in data["servers"].items()
            for tool in entry["tools"]
        )
        with _cache_lock:
            _cache[0] = (digest, names)
        return names

    def _policy(tool_name: str, tool_input: dict) -> List[Any]:
        # invocation-local provenance: cleared up front, so a non-MCP call (or a refused
        # manifest read) never inherits a PREVIOUS call's manifest digest in its audit
        # record - and cleared in THIS invocation's context only, never another's
        _digest_var.set(None)
        if not str(tool_name).startswith("mcp__"):
            # not an MCP call -> the wrapped policy's business
            return list(policy(tool_name, tool_input)) if policy else []
        # Least-authority ORDER: manifest availability and runtime-name membership are
        # established BEFORE the wrapped business policy runs. Adopter policies are
        # not required to be side-effect-free (subject lookups, file reads, network
        # evidence-gathering), and an unapproved capability must not trigger that
        # downstream work - the membership check is the cheapest and most fundamental
        # refusal, so it goes first.
        try:
            pinned_names = _pinned_names()
        except (OSError, ValueError) as exc:
            return [
                Finding.fail(
                    "mcp_manifest_unavailable",
                    severity="CRITICAL",
                    message=f"no usable MCP manifest at {manifest_path!r} ({exc}); failing "
                    "closed: no pin, no MCP (run `recusal mcp pin`)",
                    tool=tool_name,
                )
            ]
        if tool_name not in pinned_names:
            return [
                Finding.fail(
                    "mcp_not_pinned",
                    severity="CRITICAL",
                    message=f"`{tool_name}` is not in the pinned MCP manifest; a capability "
                    "that was never approved must not run",
                    tool=tool_name,
                )
            ]
        return list(policy(tool_name, tool_input)) if policy else []

    def _get_control_identity() -> Dict[str, Any]:
        """Invocation-local audit provenance, read by the audit layer AFTER the policy
        call in the same thread/task context, so it is the digest THIS adjudication
        verified (or None when it verified none)."""
        return {"manifest_sha256": _digest_var.get()}

    def _reset_control_identity() -> None:
        """Called by the hook adapter at the START of an invocation, BEFORE event
        parsing: a malformed event never reaches ``_policy`` (which also clears), so
        without this reset its audit record - written from the same reused context -
        would inherit the digest of the last VALID adjudication. An event that never
        reached the policy must never carry the policy's provenance."""
        _digest_var.set(None)

    _policy.get_control_identity = _get_control_identity  # type: ignore[attr-defined]
    _policy.reset_control_identity = _reset_control_identity  # type: ignore[attr-defined]
    return _policy
