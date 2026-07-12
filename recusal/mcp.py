"""
MCP tool-catalog governance: pin the tool catalog a server declares, refuse drift.

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
   The manifest stores *hashes only*, so pinning a poisoned declaration never embeds the
   poison anywhere. Pinning is the reviewed, deliberate step; ``recusal mcp pin`` runs a
   deterministic marker screen first (``screen_tool_declarations``, over the whole
   declaration) so obvious injection phrasing is surfaced for review instead of silently
   blessed.
2. **Verify** (deterministic, replayable, no model). ``diff_manifest`` compares a freshly
   observed catalog against the pin and emits Findings: an unpinned server or tool is a
   CRITICAL failure, a changed declaration is a CRITICAL failure that names the changed
   fields (a changed *description* is the rug-pull vector), a removed tool is a recorded
   WARNING. Same catalogs, same kernel version, same verdict, every time.
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
  url_template and header names) alongside the catalog, and verification compares it
  BEFORE launching - a changed command, a same-key env value swap, or an added server
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
adjudication (everything here) is deterministic. Servers reachable only over HTTP can be
pinned from a JSON dump instead (``recusal mcp pin --from``).
"""

import hashlib
import json
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .evidence import Finding

#: Version 4 completes remote source identity: header value TEMPLATES (not just names,
#: so a same-name Authorization swap between ${READ_ONLY_TOKEN} and ${ADMIN_TOKEN} is
#: drift), the ``headersHelper`` command template (Claude executes it at connect time -
#: it is executable configuration), and the OAuth policy surface (client_id,
#: callback_port, auth_server_metadata_url_template, scopes - the scope set is the
#: native mechanism for constraining requested authority). Runtime-only tuning fields
#: (``timeout``, ``alwaysLoad``) are deliberately NOT identity. Older manifests are
#: refused with a re-pin instruction rather than silently accepted at a weaker
#: guarantee.
MANIFEST_VERSION = 4

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
_SOURCE_FIELDS_BY_TRANSPORT: Dict[str, Tuple[str, ...]] = {
    "external": ("transport",),
    "stdio": ("transport", "command", "args", "cwd", "env_templates"),
    "http": _REMOTE_SOURCE_FIELDS,
    "sse": _REMOTE_SOURCE_FIELDS,
    "ws": _REMOTE_SOURCE_FIELDS,
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
    command Claude executes at connect time, and the pinned ``oauth`` policy);
    ``"external"`` (a dump-supplied catalog; identity out of scope, recorded not
    implied). Values that resolve from the environment never appear in a pin, and
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
        if port is not None and (not isinstance(port, int) or isinstance(port, bool)):
            raise ValueError("oauth 'callback_port' must be an integer or null")
        meta = oauth.get("auth_server_metadata_url_template")
        if meta is not None and (not isinstance(meta, str) or not meta):
            raise ValueError(
                "oauth 'auth_server_metadata_url_template' must be a nonempty string or null"
            )
        scopes = oauth.get("scopes")
        if scopes is not None and not isinstance(scopes, str):
            raise ValueError("oauth 'scopes' must be the space-separated scope string or null")
        oauth = {field: oauth.get(field) for field in _OAUTH_FIELDS}
    return {
        "transport": transport,
        "url_template": url_template,
        "header_templates": {k: header_templates[k] for k in sorted(header_templates)},
        "headers_helper_template": helper,
        "oauth": oauth,
    }


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


def build_manifest(
    catalog: Dict[str, List[dict]],
    sources: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a manifest from ``{server_name: [tool declarations]}``.

    Deterministic: the same catalog always produces the same manifest (there is no
    timestamp inside; *when* a pin happened belongs to the audit log, not the artifact).
    Hashes only, never the declarations themselves. Raises ``ValueError`` on a catalog
    that cannot be pinned unambiguously: empty, a nameless tool, or two tools with the
    same name on one server (an ambiguous catalog certifies nothing).

    ``sources`` maps a server name to its launch specification (see
    :func:`normalize_source`); a server without one is pinned as ``transport:
    "external"`` - its catalog is governed, its launch is not recusal's to govern.
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
        servers[server] = {
            "source": source,
            "source_fingerprint": _sha256(source),
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
    if data.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            f"manifest_version {data.get('manifest_version')!r} is not {MANIFEST_VERSION}"
        )
    servers = data.get("servers")
    if not isinstance(servers, dict) or not servers:
        raise ValueError("manifest has no servers")
    for server, entry in servers.items():
        if not isinstance(server, str) or not server:
            raise ValueError(f"manifest server name must be a nonempty string, got {server!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"manifest server {server!r} entry is not an object")
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
            if not _DIGEST_RE.fullmatch(pin["fingerprint"]):
                raise ValueError(
                    f"manifest tool {server!r}/{name!r} fingerprint is not "
                    "sha256:<64 lowercase hex> - a corrupt pin certifies nothing"
                )
            fields = pin.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError(
                    f"manifest tool {server!r}/{name!r} has a non-object 'fields' entry"
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
    """Compare a freshly observed catalog against the pinned manifest, as Findings.

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
        tools = tools or []
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
    - an ``mcp__server__tool`` call not present in the pinned manifest → CRITICAL refusal;
    - a pinned call is then ALSO handed to the wrapped ``policy``, so argument-level rules
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

    def _pinned_names() -> "frozenset[str]":
        with open(manifest_path, "rb") as fh:
            raw = fh.read()
        digest = hashlib.sha256(raw).hexdigest()
        # audit control identity: the hook records WHICH pin adjudicated this call
        setattr(_policy, "last_manifest_digest", f"sha256:{digest}")
        with _cache_lock:
            cached_digest, cached_names = _cache[0]
        if cached_digest == digest:
            return cached_names
        data = json.loads(raw.decode("utf-8"))  # parse outside the lock
        _validate_manifest(data)
        names = frozenset(
            f"mcp__{server}__{tool}"
            for server, entry in data["servers"].items()
            for tool in entry["tools"]
        )
        with _cache_lock:
            _cache[0] = (digest, names)
        return names

    def _policy(tool_name: str, tool_input: dict) -> List[Any]:
        inner = policy(tool_name, tool_input) if policy else []
        if not str(tool_name).startswith("mcp__"):
            return inner  # not an MCP call -> the wrapped policy's business
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
            ] + list(inner)
        return list(inner)

    return _policy
