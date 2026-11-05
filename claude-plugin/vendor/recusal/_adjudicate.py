"""
The neutral evaluator: one policy call, one verdict fold, one audit-record path.

This is the part of the Claude Code adapter that every other adapter reuses. ``decide``
runs a policy, folds its findings to a verdict with the kernel, and returns a decision the
caller enforces; with ``audit=`` it also writes the one hash-chained record every surface
writes, carrying the control identity (the package version and, for a manifest policy, the
digest this adjudication verified). Nothing here parses a Claude Code event, speaks the
hook's stdin/stdout protocol, or knows an allowlist: that stays in
:mod:`recusal.claude_code`, which re-exports every name defined here so the public import
path, ``recusal.claude_code.decide``, is unchanged.

Private module: import these names through :mod:`recusal.claude_code`. The code was moved
here verbatim (``tests/test_extractions.py`` holds the behavior to a golden record), so a
function's history is in that file's.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .authorization import fingerprint as _strict_fingerprint
from .evidence import Finding, Verdict, compute_verdict

if TYPE_CHECKING:  # runtime never needs the import; the hook only calls audit.append
    from .audit import AuditLog

# A policy maps a proposed tool call to evidence findings.
Policy = Callable[[str, dict], List[Any]]

#: Return values that are not a collection of findings and must not be read as "none".
_NOT_FINDINGS = (bool, int, float, complex, str, bytes, bytearray)


def _findings_of(result: Any) -> List[Any]:
    """Hold a policy's return value to its contract: a list (any iterable) of findings, or
    ``None`` for no objections.

    ``False``, ``0``, ``""`` and ``{}`` are not "no findings". A policy that returns one of
    them has a bug (a predicate written as a policy, an error sentinel), and reading the
    value as a clean screen would switch that check off silently. They raise here, inside
    the evaluator's policy-error handling, so the gate fails closed and the reason names the
    value. A mapping is refused for the same reason: one bare finding dict is not a list of
    findings, and iterating it would judge its keys."""
    if result is None:
        return []
    if isinstance(result, _NOT_FINDINGS) or isinstance(result, Mapping):
        raise TypeError(
            f"policy returned {type(result).__name__} {result!r}; a policy returns a list of "
            "findings (an empty list, or None, for no objections)"
        )
    try:
        return list(result)
    except TypeError:
        raise TypeError(
            f"policy returned {type(result).__name__}, which is not a list of findings"
        ) from None


def _adjudicate(
    tool_name: str,
    tool_input: dict,
    policy: Policy,
    *,
    allow_on_pass: bool,
    fail_closed: bool,
) -> Tuple[str, str, Verdict]:
    """``decide`` plus the Verdict that drove it, so a caller can put it on the record.

    A policy error has no verdict of its own; one is synthesized so the audit entry
    states what actually happened (CRITICAL when it denied, WARNING when ignored)."""
    try:
        findings = _findings_of(policy(tool_name, tool_input))
        # strict at the enforcement boundary: ambiguous evidence (a dict with no
        # status/passed) fails closed here rather than silently degrading to PASS.
        verdict = compute_verdict(findings, strict=True)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        if fail_closed:
            verdict = compute_verdict(
                [Finding.fail("recusal_policy_error", severity="CRITICAL", message=str(exc))]
            )
            return "deny", f"Recusal failed closed (policy error): {exc}", verdict
        verdict = compute_verdict(
            [
                Finding.fail(
                    "recusal_policy_error",
                    severity="WARNING",
                    message=f"ignored (fail_closed=False): {exc}",
                )
            ]
        )
        return ("allow" if allow_on_pass else "defer"), f"policy error ignored: {exc}", verdict
    if verdict.passed:
        return ("allow" if allow_on_pass else "defer"), verdict.message, verdict
    return (
        "deny",
        f"Recusal refused `{tool_name}` [{verdict.decision.value}]: {verdict.reasons()}",
        verdict,
    )


def decide(
    tool_name: str,
    tool_input: dict,
    policy: Policy,
    *,
    allow_on_pass: bool = False,
    fail_closed: bool = True,
    audit: Optional["AuditLog"] = None,
    surface: Optional[str] = None,
    tool_use_id: Optional[str] = None,
    actor: Optional[str] = None,
    control: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Run the policy, fold to a verdict, return ``(decision, reason)``.

    ``decision`` is ``"defer"`` (PASS, and not auto-allowing), ``"allow"`` (PASS with
    ``allow_on_pass=True``), or ``"deny"`` (RETRY/FAIL).

    Without ``audit`` this is a pure function. Pass ``audit=`` (a
    :class:`recusal.audit.AuditLog`) to put the adjudication on the record as one
    hash-chained entry, the same entry :func:`run_pretooluse_hook` writes: the tool, the
    decision, the reasons, a SHA-256 fingerprint of ``tool_input`` (contents are never
    embedded), and the control identity (see ``control`` on :func:`run_pretooluse_hook`).
    ``surface`` is then required and names the runtime that adjudicated, for example
    ``"my_runtime.tool_gate"``; there is no default, because a record that guesses where
    it came from is not evidence. ``"claude_code.pretooluse"`` is reserved for
    :func:`run_pretooluse_hook`. ``tool_use_id`` links the entry to the runtime's call
    id, and ``actor`` labels it. If the log cannot be written the decision becomes a
    deny, the record being part of the control, unless ``fail_closed=False``.

    The recording arguments are refused (``ValueError``) without ``audit``, so a caller
    who believes it is recording is never silently not recording.

    Ownership of ``tool_input``: the policy is called with the object the caller passed,
    not a copy, and this function executes nothing. The record (with ``audit=``)
    fingerprints the proposal BEFORE the policy sees it, so a policy's write to the dict
    cannot reach the record; it can reach whatever the caller executes from that same
    object afterwards. A caller that executes the call it adjudicated therefore hands the
    policy a detached copy and executes what it fingerprinted, or keeps its own prepared
    invocation apart from what it passes here. The framework adapters under ``examples/``
    do this at their execution seam; the conformance suite's invariant 11 holds them to it.
    """
    if audit is None:
        if any(v is not None for v in (surface, tool_use_id, actor, control)):
            raise ValueError("surface, tool_use_id, actor and control require audit=")
        decision, reason, _ = _adjudicate(
            tool_name, tool_input, policy, allow_on_pass=allow_on_pass, fail_closed=fail_closed
        )
        return decision, reason
    if not isinstance(surface, str) or not surface or surface != surface.strip():
        raise ValueError("audit= requires surface, a nonempty name of the adjudicating runtime")
    if surface == _HOOK_SURFACE:
        raise ValueError(f"surface {_HOOK_SURFACE!r} is reserved for run_pretooluse_hook")
    if tool_use_id is not None and (not isinstance(tool_use_id, str) or not tool_use_id):
        raise ValueError("tool_use_id must be a nonempty string")
    if control is not None and not isinstance(control, dict):
        raise ValueError("control must be a dict")
    _reset_control_identity(policy)
    # The record binds the input AS PROPOSED: it is fingerprinted before the policy sees
    # it, so a policy that mutates its argument cannot write the mutation onto the record
    # while the caller executes the proposal.
    input_sha256, refused = _bind_input(tool_input, fail_closed=fail_closed)
    if refused is not None:
        decision, reason, verdict = refused
    else:
        decision, reason, verdict = _adjudicate(
            tool_name, tool_input, policy, allow_on_pass=allow_on_pass, fail_closed=fail_closed
        )
    return _record(
        audit,
        verdict,
        surface=surface,
        tool_name=tool_name,
        decision=decision,
        reason=reason,
        input_sha256=input_sha256,
        event_ids={"tool_use_id": tool_use_id} if tool_use_id is not None else {},
        policy=policy,
        control=control,
        actor=actor,
        fail_closed=fail_closed,
    )


#: The surface label :func:`run_pretooluse_hook` writes; :func:`decide` refuses it so an
#: entry carrying it always came from a Claude Code PreToolUse event.
_HOOK_SURFACE = "claude_code.pretooluse"


def _reset_control_identity(policy: Policy) -> None:
    """Clear invocation-local control identity before an adjudication, so a reused policy
    object never lends the previous call's provenance (a manifest digest) to this record."""
    reset = getattr(policy, "reset_control_identity", None)
    if callable(reset):
        try:
            reset()
        except Exception:  # noqa: BLE001, a broken reset must not take down the gate
            pass


def _record(
    audit: "AuditLog",
    verdict: Verdict,
    *,
    surface: str,
    tool_name: Optional[str],
    decision: str,
    reason: str,
    input_sha256: Optional[str],
    event_ids: Dict[str, Any],
    policy: Policy,
    control: Optional[Dict[str, Any]],
    actor: Optional[str],
    fail_closed: bool,
) -> Tuple[str, str]:
    """Append one adjudication to ``audit``: the ONE recording path for every surface.

    Returns the final ``(decision, reason)``: unchanged when the entry is written; a deny
    when it cannot be and ``fail_closed`` (the record is part of the control); unchanged
    when it cannot be and the caller chose fail-open."""
    action: Dict[str, Any] = {
        "surface": surface,
        "tool": tool_name,
        "decision": decision,
        "reason": reason,
    }
    if input_sha256 is not None:
        action["input_sha256"] = input_sha256
    action.update(event_ids)
    action["control"] = _control_identity(policy, control)
    try:
        audit.append(verdict, action=action, actor=actor)
    except Exception as exc:  # noqa: BLE001 - an unwritable log must not go unnoticed
        if fail_closed:
            return (
                "deny",
                f"Recusal failed closed: audit log unavailable ({exc}); "
                "the record is part of the control",
            )
    return decision, reason


#: Control-identity fields the IMPLEMENTATION owns. Caller-supplied values for these are
#: discarded, never merged: an audit record whose "recusal_version" or "manifest_sha256"
#: could be written by the caller would be provenance theater, not provenance.
_AUTHORITATIVE_CONTROL_KEYS = frozenset({"recusal_version", "manifest_sha256"})


def _identity_getter(policy: Any) -> Optional[Callable[[], Dict[str, Any]]]:
    """The one identity-discovery contract: the callable that yields a policy's control
    identity, or None when the policy carries none.

    The evaluator reads a manifest digest through this, and so must anything that composes
    policies (``examples/composed_policies.py``), or a composition would record provenance
    the direct policy records and lose it for a shape the evaluator supports. Two shapes:
    ``get_control_identity()`` when the policy has one (:func:`recusal.mcp.manifest_policy`,
    or a custom policy's override), else a ``last_manifest_digest`` attribute, the legacy
    shape for custom policy objects, read at call time (shared mutable state, so safe for
    sequential use only)."""
    getter = getattr(policy, "get_control_identity", None)
    if callable(getter):
        return getter
    if hasattr(policy, "last_manifest_digest"):
        return lambda: {"manifest_sha256": getattr(policy, "last_manifest_digest", None)}
    return None


def _control_identity(policy: Policy, control: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The control identity recorded on EVERY audit path, built in one place.

    Caller metadata first (reserved keys stripped), then the implementation-owned truth
    written over it: the actual package version always, and the manifest digest a
    :func:`recusal.mcp.manifest_policy` verified during THIS adjudication when present.

    The digest is read through the policy's ``get_control_identity()`` when it has one:
    :func:`recusal.mcp.manifest_policy` keeps it in a ``ContextVar``, so under
    concurrent reuse of one policy object each audit record sees the digest its own
    invocation verified, set only after successful validation (a corrupt manifest or a
    non-MCP call never carries manifest provenance). A plain ``last_manifest_digest``
    attribute is honored as a fallback for custom policy objects; being shared mutable
    state, that seam is safe for sequential use only.
    """
    from . import __version__ as _recusal_version  # local import: no cycle at load

    identity = {k: v for k, v in (control or {}).items() if k not in _AUTHORITATIVE_CONTROL_KEYS}
    identity["recusal_version"] = _recusal_version
    getter = _identity_getter(policy)
    manifest_digest = None
    if getter is not None:
        try:
            manifest_digest = dict(getter()).get("manifest_sha256")
        except Exception:  # noqa: BLE001, a broken getter must not take down the audit path
            manifest_digest = None
    if isinstance(manifest_digest, str):
        identity["manifest_sha256"] = manifest_digest
    return identity


def _input_fingerprint(tool_input: dict) -> str:
    """SHA-256 over the canonical JSON of the proposed tool input. The audit entry binds
    to the exact proposed call without embedding its contents (a Write's file body, an
    env value): hashes only, the same doctrine as the MCP manifest.

    The input is held to the strict JSON domain (the one serializer
    :func:`recusal.authorization.fingerprint` uses): a non-string key, a non-finite
    number or any non-JSON object raises ``ValueError`` instead of being folded into a
    look-alike, so two different inputs can never share a fingerprint. Every valid JSON
    input hashes exactly as before."""
    return _strict_fingerprint(tool_input)


def _bind_input(
    tool_input: dict, *, fail_closed: bool
) -> Tuple[Optional[str], Optional[Tuple[str, str, Verdict]]]:
    """Fingerprint a proposed input for its audit record, before the policy sees it.

    Returns ``(fingerprint, None)``. An input the record cannot bind is refused when
    ``fail_closed`` (the record is part of the control) and the policy is never called:
    ``(None, (decision, reason, verdict))`` carries the denial. Fail-open records no
    fingerprint and lets the policy decide: ``(None, None)``. Taken first, so the snapshot
    on the record is the proposal, whatever a policy later does to the dict it was handed."""
    try:
        return _input_fingerprint(tool_input), None
    except Exception as exc:  # noqa: BLE001 - an input the record cannot bind must not slip through
        if not fail_closed:
            return None, None
        verdict = compute_verdict(
            [Finding.fail("recusal_unfingerprintable_input", severity="CRITICAL", message=str(exc))]
        )
        return None, (
            "deny",
            f"Recusal failed closed: tool_input cannot be fingerprinted ({exc})",
            verdict,
        )
