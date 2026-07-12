"""
MCP discovery governance: pin the tool catalog a server declares, refuse drift.

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
   WARNING. Same catalogs, same verdict, every time.
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
- The manifest pins the *declared catalog*, not the identity of the server process that
  declares it: observing a stdio server executes the command the config names, so a
  rewritten ``.mcp.json`` runs at observe time, before its catalog can fail verification.
  Treat the config as executable code and protect it (and the manifest) as control-plane
  files; pinning the launch specification itself is a named roadmap item, not shipped.
- Fingerprints are byte-exact over canonical JSON (sorted keys, no whitespace, UTF-8,
  no unicode normalization): a homoglyph swap in a description IS a change and fails.

This module is pure: standard library only, no subprocess, no thread, no network, a
function of its inputs. *Collecting* a catalog from a live server (the one place that
spawns a process) lives in the sibling :mod:`recusal.mcp_fetch`, deliberately apart, so
the decision surface never touches a process. Collection is nondeterministic I/O;
adjudication (everything here) is deterministic. Servers reachable only over HTTP can be
pinned from a JSON dump instead (``recusal mcp pin --from``).
"""

import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .evidence import Finding

MANIFEST_VERSION = 1

#: Declaration fields fingerprinted individually so a drift refusal can name what moved.
#: The whole declaration is also fingerprinted, so a change in any *other* field still
#: fails, it is just reported without a field name.
_TRACKED_FIELDS = ("description", "inputSchema", "annotations", "title", "outputSchema")


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


def build_manifest(catalog: Dict[str, List[dict]]) -> Dict[str, Any]:
    """Build a manifest from ``{server_name: [tool declarations]}``.

    Deterministic: the same catalog always produces the same manifest (there is no
    timestamp inside; *when* a pin happened belongs to the audit log, not the artifact).
    Hashes only, never the declarations themselves. Raises ``ValueError`` on a catalog
    that cannot be pinned unambiguously: empty, a nameless tool, or two tools with the
    same name on one server (an ambiguous catalog certifies nothing).
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
            if name in pinned_tools:
                raise ValueError(
                    f"server {server!r} declares tool {name!r} twice; an ambiguous "
                    "catalog cannot be pinned"
                )
            pinned_tools[name] = {
                "fingerprint": tool_fingerprint(tool),
                "fields": {k: _sha256(tool[k]) for k in _TRACKED_FIELDS if k in tool},
            }
        servers[server] = {"tools": pinned_tools}
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
        tools = entry.get("tools") if isinstance(entry, dict) else None
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
    # never cached, so a changed-but-corrupt file refuses on this call and every next one.
    _cache: Dict[str, Any] = {"digest": None, "names": frozenset()}

    def _pinned_names() -> "frozenset[str]":
        with open(manifest_path, "rb") as fh:
            raw = fh.read()
        digest = hashlib.sha256(raw).hexdigest()
        if _cache["digest"] != digest:
            data = json.loads(raw.decode("utf-8"))
            _validate_manifest(data)
            _cache["names"] = frozenset(
                f"mcp__{server}__{tool}"
                for server, entry in data["servers"].items()
                for tool in entry["tools"]
            )
            _cache["digest"] = digest
        return _cache["names"]

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
