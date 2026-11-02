"""
The runtime manifest policy: enforce a pinned MCP manifest at call time.

:func:`manifest_policy` is the file-backed, call-time half of MCP governance: it reads the
pinned manifest's bytes on every MCP call, reparses only when their digest changes, keeps
the (digest, names) pair under a lock, and carries the digest this adjudication verified in
a ContextVar for the audit record. The pure artifact half, building, comparing and
validating manifests and observations, stays in :mod:`recusal.mcp`, which re-exports this
function so the public import path, ``recusal.mcp.manifest_policy``, is unchanged.

Private module: import ``manifest_policy`` through :mod:`recusal.mcp`. The code was moved
here verbatim (``tests/test_extractions.py`` holds the behavior to its tests), so the
function's history is in that file's.
"""

import contextvars
import hashlib
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .evidence import Finding


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
        from .mcp import _validate_manifest  # the artifact half owns validation; imported
        # here, not at module load, because recusal.mcp re-exports this function

        _validate_manifest(data)
        # audit control identity, recorded ONLY after successful parse + validation: a
        # corrupt manifest must never be recorded as a successfully enforced one
        _digest_var.set(f"sha256:{digest}")
        names = frozenset(
            f"mcp__{server}__"
            + (pin["callable_name"] if entry["runtime"]["mode"] == "claude_plugin" else tool)
            for server, entry in data["servers"].items()
            for tool, pin in entry["tools"].items()
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
