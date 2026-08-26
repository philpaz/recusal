"""
Authorization adjudication: is this agent authorized for THIS exact action?

Recusal is a deterministic authorization adjudicator for agent actions. It verifies each
proposed action against explicit, provenance-bearing authority evidence and can refuse
before execution. This module is that verification, built the way the rest of the
library is built: pure functions over supplied evidence, one named :class:`Finding` per
authorization dimension, folded through :func:`recusal.certify_evidence`. The verdict
kernel is untouched; this is a sibling module of checks.

A deny-list asks "does this command look dangerous?". This module asks the question the
deny-list cannot: where did the authority for this action come from, precisely what does
it permit, and does this call remain inside it? Actions that look harmless on their own
but are contextually wrong (updating the wrong customer, invoking a legitimate tool after
the delegated task expired, repeating an otherwise valid payment, escalating from a
read-only task into a write) are refused here on evidence, not on pattern matching.

**What this module does not do, stated so the claim stays checkable.**

- It does not establish identity. A runtime label such as Claude Code's ``agent_id`` is
  what the runtime reported, not an authenticated principal. The principal dimension
  passes ONLY through an adopter-supplied trust rule mapping a runtime label to a
  configured principal; with no rule it refuses. A label is exactly the kind of authority
  claim this library exists to scrutinize, never a pass on its own.
- It does not issue or validate delegation chains. Delegation, approval, and scope arrive
  as evidence inside :class:`AuthorizationContext`, each item carrying the provenance the
  adopter assigns; this module verifies the action against that evidence.
- It does not read a clock, keep a counter, or hold a nonce set. ``now``, ``calls_so_far``
  and ``used_nonces`` are supplied. Expiry, budget and replay checks therefore verify the
  evidence they were handed and do **not** enforce atomic quotas or prevent concurrent
  replay: two callers that both observe ``calls_so_far=2`` both pass ``max_calls=3``, both
  execute, and the final count is 4. Trustworthy clocks and atomic state transitions
  remain the caller's responsibility, in the same way trustworthy evidence always has.
- It does not sign. :class:`DecisionReceipt` binds a decision to canonical bytes and a
  SHA-256 digest, the same boundary as :mod:`recusal.audit`: tamper-evident when anchored,
  not an identity assertion, and not independently authenticated by its digest alone.

**The composition invariant.** :func:`certify_evidence` refuses an empty set, and that is
all it can promise: one unrelated passing finding satisfies it. Authorization needs more.
:func:`certify_dimensions` requires a named finding for EVERY required dimension and
synthesizes a failed CRITICAL ``authorization.dimension_missing`` for each one absent, so
a request missing principal or resource evidence cannot be certified by the presence of
some other evidence. :func:`certify_authorization` runs the built-in checks and applies
that invariant; it is the entry point an adopter wires in.

Standard library only. Every check is a pure function of its arguments.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .evidence import Finding, Severity, Verdict, certify_evidence

#: Provenance label the Claude Code hook assigns to fields it copies out of a
#: ``PreToolUse`` event (``agent_id``, ``agent_type``, ``permission_mode``). The label
#: says where a value came from; it says nothing about whether to trust it.
PROVENANCE_CLAUDE_PRETOOLUSE = "claude_pretooluse_event"

#: The authorization dimensions, in the order their findings are produced. Each maps to
#: one built-in check and one check name, ``authorization.<dimension>``.
DIMENSIONS: Tuple[str, ...] = (
    "principal",
    "subject",
    "tool_operation",
    "arguments",
    "expiry",
    "budget",
    "replay",
    "policy_binding",
    "manifest_binding",
)

#: What :func:`certify_authorization` requires unless told otherwise: every dimension.
#: Narrowing ``required=`` is an explicit weak claim the adopter makes at the call site.
DEFAULT_DIMENSIONS: Tuple[str, ...] = DIMENSIONS

_CHECK_PREFIX = "authorization."


def canonical_json(value: Any) -> str:
    """The one serialization every fingerprint and receipt in this module uses."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def fingerprint(value: Any) -> str:
    """SHA-256 hex digest of :func:`canonical_json` of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenMapping:
    """An immutable, key-sorted mapping; the frozen form of a dict. Distinct from a
    tuple so an empty mapping and an empty list never read as each other."""

    items: Tuple[Tuple[str, Any], ...]

    def plain(self) -> Dict[str, Any]:
        return {k: _thaw(v) for k, v in self.items}


def _freeze(value: Any) -> Any:
    """Recursively convert mappings and sequences into immutable, order-stable forms."""
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(tuple(sorted((str(k), _freeze(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_freeze(v) for v in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=lambda v: canonical_json(_thaw(v)))
        return tuple(items)
    return value


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze`: ordinary dicts and lists, for serialization."""
    if isinstance(value, FrozenMapping):
        return value.plain()
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class Supplied:
    """One item of authority evidence and where it came from.

    ``provenance`` is a label the adopter assigns (``"operator_config"``,
    ``"session_store"``, :data:`PROVENANCE_CLAUDE_PRETOOLUSE`, ...). It is recorded in
    every finding that consumes the item and in the receipt, so a reviewer can see what
    each dimension was decided on. It is a label, not an attestation.
    """

    value: Any
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("Supplied.provenance must be a nonempty string")
        object.__setattr__(self, "value", _freeze(self.value))

    def plain(self) -> Any:
        """The value in ordinary Python containers (dicts and lists)."""
        return _thaw(self.value)


@dataclass(frozen=True)
class Constraints:
    """The bounds an authorization was granted under. Every member is optional; an
    absent member means "no constraint of this kind was declared", which the matching
    check reports as a passing finding that SAYS so, never silently.

    - ``fields``: the only argument keys the action may carry.
    - ``max_calls``: the call budget; the check compares against supplied ``calls_so_far``.
    - ``expires_at``: ISO 8601, timezone-aware; the check compares against supplied ``now``.
    - ``max_value`` / ``value_field``: an upper bound on one numeric argument.
    """

    fields: Optional[Tuple[str, ...]] = None
    max_calls: Optional[int] = None
    expires_at: Optional[str] = None
    max_value: Optional[float] = None
    value_field: Optional[str] = None

    def __post_init__(self) -> None:
        if self.fields is not None:
            object.__setattr__(self, "fields", tuple(str(f) for f in self.fields))
        if self.max_calls is not None and (isinstance(self.max_calls, bool) or self.max_calls < 0):
            raise ValueError("Constraints.max_calls must be a nonnegative integer")
        if (self.max_value is None) != (self.value_field is None):
            raise ValueError("Constraints.max_value and value_field must be given together")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": list(self.fields) if self.fields is not None else None,
            "max_calls": self.max_calls,
            "expires_at": self.expires_at,
            "max_value": self.max_value,
            "value_field": self.value_field,
        }


@dataclass(frozen=True)
class ActionRequest:
    """The action an agent proposes, as the adjudicator sees it.

    ``principal`` is the label the caller attributes the request to, often a runtime
    label; it is bound to a configured principal only through the context's trust rule.
    ``arguments`` are the call's arguments (frozen on construction); the fingerprint is
    over their canonical JSON, so a receipt binds to the exact call without embedding it.
    ``nonce`` is the caller's replay token for this call, if it issues one.
    """

    principal: str
    tool: str
    operation: str
    resource: Optional[str] = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    nonce: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("principal", "tool", "operation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"ActionRequest.{name} must be a nonempty string")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("ActionRequest.arguments must be a mapping")
        object.__setattr__(self, "arguments", _freeze(self.arguments))

    def arguments_plain(self) -> Dict[str, Any]:
        thawed = _thaw(self.arguments)
        return thawed if isinstance(thawed, dict) else {}

    @property
    def arguments_fingerprint(self) -> str:
        return fingerprint(self.arguments_plain())

    @property
    def action_fingerprint(self) -> str:
        """Binds principal, tool, operation, resource, arguments and nonce together."""
        return fingerprint(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "principal": self.principal,
            "tool": self.tool,
            "operation": self.operation,
            "resource": self.resource,
            "arguments": self.arguments_plain(),
            "nonce": self.nonce,
        }


@dataclass(frozen=True)
class AuthorizationContext:
    """Everything the checks decide on, each item a :class:`Supplied` with provenance.

    - ``trusted_principals``: mapping runtime label -> configured principal. The ONLY way
      the principal dimension passes.
    - ``active_subject``: the resource this session is about (the "active member").
    - ``allowed_operations``: mapping tool -> sequence of permitted operations.
    - ``constraints``: a :class:`Constraints`.
    - ``now``: ISO 8601, timezone-aware. Supplied, never read here.
    - ``calls_so_far``: how many calls this authorization has already spent. Supplied.
    - ``used_nonces``: nonces already consumed. Supplied.
    - ``approval``: mapping with ``policy_version`` and optionally ``manifest_fingerprint``,
      the identities the authorization was approved under.
    - ``policy_version`` / ``manifest_fingerprint``: the identities in force now.
    """

    trusted_principals: Optional[Supplied] = None
    active_subject: Optional[Supplied] = None
    allowed_operations: Optional[Supplied] = None
    constraints: Optional[Supplied] = None
    now: Optional[Supplied] = None
    calls_so_far: Optional[Supplied] = None
    used_nonces: Optional[Supplied] = None
    approval: Optional[Supplied] = None
    policy_version: Optional[Supplied] = None
    manifest_fingerprint: Optional[Supplied] = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None and not isinstance(value, Supplied):
                raise ValueError(f"AuthorizationContext.{name} must be a Supplied or None")

    def provenance(self) -> Dict[str, str]:
        """Which evidence was supplied and from where; absent items are omitted."""
        out: Dict[str, str] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None:
                out[name] = value.provenance
        return out


# --- the checks: one named finding per dimension, pure, never raising on evidence -----


def _ok(dimension: str, message: str, **context: Any) -> Finding:
    return Finding.ok(
        _CHECK_PREFIX + dimension, severity=Severity.CRITICAL, message=message, **context
    )


def _fail(dimension: str, message: str, **context: Any) -> Finding:
    return Finding.fail(
        _CHECK_PREFIX + dimension, severity=Severity.CRITICAL, message=message, **context
    )


def _missing(dimension: str, what: str) -> Finding:
    return _fail(
        dimension, f"no {what} supplied; absent authority evidence refuses, it does not pass"
    )


def check_principal(request: ActionRequest, context: AuthorizationContext) -> Finding:
    """The request's label maps to a configured principal through the supplied trust rule."""
    rule = context.trusted_principals
    if rule is None:
        return _fail(
            "principal",
            "no trust rule supplied; a runtime label is not an authenticated identity and "
            "cannot authorize itself",
            label=request.principal,
        )
    mapping = rule.plain()
    if not isinstance(mapping, dict):
        return _fail("principal", "trust rule is not a mapping", provenance=rule.provenance)
    principal = mapping.get(request.principal)
    if not isinstance(principal, str) or not principal:
        return _fail(
            "principal",
            f"label {request.principal!r} is not bound to any configured principal",
            label=request.principal,
            provenance=rule.provenance,
        )
    return _ok(
        "principal",
        f"label {request.principal!r} is bound to principal {principal!r}",
        label=request.principal,
        principal=principal,
        provenance=rule.provenance,
    )


def check_subject(request: ActionRequest, context: AuthorizationContext) -> Finding:
    """The action's resource is the subject this session is about."""
    active = context.active_subject
    if active is None:
        return _missing("subject", "active subject")
    if request.resource is None:
        return _fail(
            "subject",
            "the action names no resource, so it cannot be bound to the active subject",
            active=active.plain(),
            provenance=active.provenance,
        )
    if request.resource != active.plain():
        return _fail(
            "subject",
            f"action targets {request.resource} but the active subject is {active.plain()}",
            target=request.resource,
            active=active.plain(),
            provenance=active.provenance,
        )
    return _ok(
        "subject",
        "action targets the active subject",
        target=request.resource,
        provenance=active.provenance,
    )


def check_tool_operation(request: ActionRequest, context: AuthorizationContext) -> Finding:
    """The tool and operation are within what the authorization permits."""
    allowed = context.allowed_operations
    if allowed is None:
        return _missing("tool_operation", "allowed operations")
    table = allowed.plain()
    if not isinstance(table, dict):
        return _fail(
            "tool_operation", "allowed operations is not a mapping", provenance=allowed.provenance
        )
    ops = table.get(request.tool)
    if ops is None:
        return _fail(
            "tool_operation",
            f"tool {request.tool!r} is not authorized",
            tool=request.tool,
            provenance=allowed.provenance,
        )
    if isinstance(ops, str):
        ops = [ops]
    if request.operation not in [str(o) for o in ops]:
        return _fail(
            "tool_operation",
            f"operation {request.operation!r} on {request.tool!r} is not authorized",
            tool=request.tool,
            operation=request.operation,
            provenance=allowed.provenance,
        )
    return _ok(
        "tool_operation",
        f"{request.tool}.{request.operation} is authorized",
        tool=request.tool,
        operation=request.operation,
        provenance=allowed.provenance,
    )


def _constraints(context: AuthorizationContext) -> Tuple[Optional[Constraints], Optional[str]]:
    supplied = context.constraints
    if supplied is None:
        return None, None
    value = supplied.value
    if isinstance(value, Constraints):
        return value, supplied.provenance
    return None, supplied.provenance


def check_arguments(request: ActionRequest, context: AuthorizationContext) -> Finding:
    """The arguments stay inside the declared field set and value bound."""
    constraints, provenance = _constraints(context)
    if provenance is None:
        return _missing("arguments", "constraints")
    if constraints is None:
        return _fail(
            "arguments", "constraints evidence is not a Constraints", provenance=provenance
        )
    args = request.arguments_plain()
    if constraints.fields is not None:
        extra = sorted(set(args) - set(constraints.fields))
        if extra:
            return _fail(
                "arguments",
                f"arguments outside the authorized fields: {', '.join(extra)}",
                extra=extra,
                fields=list(constraints.fields),
                provenance=provenance,
            )
    if constraints.max_value is not None and constraints.value_field is not None:
        raw = args.get(constraints.value_field)
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return _fail(
                "arguments",
                f"argument {constraints.value_field!r} is not a number, so the value bound "
                "cannot be verified",
                value_field=constraints.value_field,
                provenance=provenance,
            )
        if value > constraints.max_value:
            return _fail(
                "arguments",
                f"argument {constraints.value_field!r} = {value} exceeds the bound {constraints.max_value}",
                value_field=constraints.value_field,
                value=value,
                max_value=constraints.max_value,
                provenance=provenance,
            )
    declared = constraints.fields is not None or constraints.max_value is not None
    return _ok(
        "arguments",
        "arguments are inside the declared constraints"
        if declared
        else "no argument constraint was declared (stated, not inferred)",
        arguments_fingerprint=request.arguments_fingerprint,
        provenance=provenance,
    )


def _parse_instant(text: Any) -> Optional[datetime]:
    if not isinstance(text, str):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None  # a naive instant cannot be compared across systems
    return parsed


def check_expiry(request: ActionRequest, context: AuthorizationContext) -> Finding:
    """The authorization has not expired at the supplied ``now``."""
    constraints, provenance = _constraints(context)
    if provenance is None:
        return _missing("expiry", "constraints")
    if constraints is None:
        return _fail("expiry", "constraints evidence is not a Constraints", provenance=provenance)
    if constraints.expires_at is None:
        return _ok("expiry", "no expiry was declared (stated, not inferred)", provenance=provenance)
    expires = _parse_instant(constraints.expires_at)
    if expires is None:
        return _fail(
            "expiry",
            "expires_at is not a timezone-aware ISO 8601 instant",
            expires_at=constraints.expires_at,
            provenance=provenance,
        )
    if context.now is None:
        return _missing("expiry", "clock evidence (now)")
    now = _parse_instant(context.now.plain())
    if now is None:
        return _fail(
            "expiry",
            "now is not a timezone-aware ISO 8601 instant",
            now=context.now.plain(),
            provenance=context.now.provenance,
        )
    if now >= expires:
        return _fail(
            "expiry",
            f"authorization expired at {constraints.expires_at} (now {context.now.plain()})",
            expires_at=constraints.expires_at,
            now=context.now.plain(),
            provenance=provenance,
            clock_provenance=context.now.provenance,
        )
    return _ok(
        "expiry",
        "authorization is within its validity window at the supplied now",
        expires_at=constraints.expires_at,
        now=context.now.plain(),
        provenance=provenance,
        clock_provenance=context.now.provenance,
    )


def check_budget(request: ActionRequest, context: AuthorizationContext) -> Finding:
    """This call stays inside the call budget, given the supplied count.

    Verifies supplied counter evidence only: it does not reserve a slot, so concurrent
    callers observing the same count can each pass (see the module docstring).
    """
    constraints, provenance = _constraints(context)
    if provenance is None:
        return _missing("budget", "constraints")
    if constraints is None:
        return _fail("budget", "constraints evidence is not a Constraints", provenance=provenance)
    if constraints.max_calls is None:
        return _ok(
            "budget", "no call budget was declared (stated, not inferred)", provenance=provenance
        )
    if context.calls_so_far is None:
        return _missing("budget", "counter evidence (calls_so_far)")
    spent = context.calls_so_far.plain()
    if isinstance(spent, bool) or not isinstance(spent, int) or spent < 0:
        return _fail(
            "budget",
            "calls_so_far is not a nonnegative integer",
            calls_so_far=spent,
            provenance=context.calls_so_far.provenance,
        )
    if spent >= constraints.max_calls:
        return _fail(
            "budget",
            f"call budget exhausted: {spent} of {constraints.max_calls} already spent",
            calls_so_far=spent,
            max_calls=constraints.max_calls,
            provenance=provenance,
            counter_provenance=context.calls_so_far.provenance,
        )
    return _ok(
        "budget",
        f"call {spent + 1} of {constraints.max_calls} (supplied count, not reserved)",
        calls_so_far=spent,
        max_calls=constraints.max_calls,
        provenance=provenance,
        counter_provenance=context.calls_so_far.provenance,
    )


def check_replay(request: ActionRequest, context: AuthorizationContext) -> Finding:
    """The request's nonce has not been used, per the supplied use-state."""
    if request.nonce is None:
        return _fail("replay", "the action carries no nonce, so replay cannot be ruled out")
    if context.used_nonces is None:
        return _missing("replay", "use-state evidence (used_nonces)")
    used = context.used_nonces.plain()
    if isinstance(used, dict):
        used = list(used)
    if not isinstance(used, list):
        return _fail(
            "replay", "used_nonces is not a collection", provenance=context.used_nonces.provenance
        )
    if request.nonce in [str(n) for n in used]:
        return _fail(
            "replay",
            f"nonce {request.nonce!r} was already used",
            nonce=request.nonce,
            provenance=context.used_nonces.provenance,
        )
    return _ok(
        "replay",
        "nonce is unused per the supplied use-state (verified, not reserved)",
        nonce=request.nonce,
        provenance=context.used_nonces.provenance,
    )


def _approval(context: AuthorizationContext) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if context.approval is None:
        return None, None
    value = context.approval.plain()
    return (value if isinstance(value, dict) else None), context.approval.provenance


def check_policy_binding(request: ActionRequest, context: AuthorizationContext) -> Finding:
    """The policy in force is the one the authorization was approved under."""
    approval, provenance = _approval(context)
    if provenance is None:
        return _missing("policy_binding", "approval")
    if approval is None:
        return _fail("policy_binding", "approval evidence is not a mapping", provenance=provenance)
    approved = approval.get("policy_version")
    if not isinstance(approved, str) or not approved:
        return _fail("policy_binding", "approval names no policy_version", provenance=provenance)
    if context.policy_version is None:
        return _missing("policy_binding", "current policy_version")
    current = context.policy_version.plain()
    if current != approved:
        return _fail(
            "policy_binding",
            f"policy changed since approval: approved under {approved!r}, now {current!r}",
            approved=approved,
            current=current,
            provenance=provenance,
        )
    return _ok(
        "policy_binding",
        f"policy {approved!r} is the one approved",
        policy_version=approved,
        provenance=provenance,
    )


def check_manifest_binding(request: ActionRequest, context: AuthorizationContext) -> Finding:
    """The MCP manifest in force is the one the authorization was approved under.

    An approval with no manifest fingerprint and no manifest in force passes with that
    stated; one side present without the other refuses.
    """
    approval, provenance = _approval(context)
    if provenance is None:
        return _missing("manifest_binding", "approval")
    if approval is None:
        return _fail(
            "manifest_binding", "approval evidence is not a mapping", provenance=provenance
        )
    approved = approval.get("manifest_fingerprint")
    current = (
        context.manifest_fingerprint.plain() if context.manifest_fingerprint is not None else None
    )
    if approved is None and current is None:
        return _ok(
            "manifest_binding",
            "no manifest is bound to this authorization (stated, not inferred)",
            provenance=provenance,
        )
    if approved is None or current is None:
        return _fail(
            "manifest_binding",
            "manifest binding is one-sided: approved fingerprint and current fingerprint "
            "must both be present or both absent",
            approved=approved,
            current=current,
            provenance=provenance,
        )
    if approved != current:
        return _fail(
            "manifest_binding",
            "MCP manifest changed since approval",
            approved=approved,
            current=current,
            provenance=provenance,
        )
    return _ok(
        "manifest_binding",
        "MCP manifest is the one approved",
        manifest_fingerprint=approved,
        provenance=provenance,
    )


_CHECKS = {
    "principal": check_principal,
    "subject": check_subject,
    "tool_operation": check_tool_operation,
    "arguments": check_arguments,
    "expiry": check_expiry,
    "budget": check_budget,
    "replay": check_replay,
    "policy_binding": check_policy_binding,
    "manifest_binding": check_manifest_binding,
}


def run_checks(request: ActionRequest, context: AuthorizationContext) -> List[Finding]:
    """Every built-in check, in :data:`DIMENSIONS` order. A check that raises (which
    would be a bug, not evidence) becomes a failed CRITICAL finding for its dimension:
    the gate fails closed and names why."""
    findings: List[Finding] = []
    for dimension in DIMENSIONS:
        try:
            findings.append(_CHECKS[dimension](request, context))
        except Exception as exc:  # noqa: BLE001 - a broken check must refuse, not disable the gate
            findings.append(_fail(dimension, f"check raised {type(exc).__name__}: {exc}"))
    return findings


def certify_dimensions(
    findings: Iterable[Finding],
    *,
    required: Sequence[str] = DEFAULT_DIMENSIONS,
) -> Verdict:
    """Fold ``findings`` as certification evidence, requiring one named finding per
    ``required`` dimension.

    For each required dimension with no finding named ``authorization.<dimension>``, a
    failed CRITICAL ``authorization.dimension_missing`` finding is synthesized before the
    fold, so an unrelated passing finding can never certify a request that lacks evidence
    on a required dimension. The fold itself is :func:`recusal.certify_evidence`.
    """
    items = list(findings)
    present = {f.check for f in items}
    for dimension in required:
        if _CHECK_PREFIX + dimension not in present:
            items.append(
                Finding.fail(
                    _CHECK_PREFIX + "dimension_missing",
                    severity=Severity.CRITICAL,
                    message=f"required authorization dimension {dimension!r} produced no finding",
                    dimension=dimension,
                )
            )
    return certify_evidence(items)


@dataclass(frozen=True)
class AuthorizationDecision:
    """The findings and the verdict they folded into, kept together so a receipt can
    carry the passing findings the verdict summary drops."""

    request: ActionRequest
    findings: Tuple[Finding, ...]
    verdict: Verdict
    required: Tuple[str, ...]

    @property
    def authorized(self) -> bool:
        return self.verdict.passed


def certify_authorization(
    request: ActionRequest,
    context: AuthorizationContext,
    *,
    required: Sequence[str] = DEFAULT_DIMENSIONS,
    findings: Iterable[Finding] = (),
) -> AuthorizationDecision:
    """Adjudicate ``request`` against ``context``: run every built-in check, add any
    adopter ``findings`` (custom dimensions, named ``authorization.<name>`` to count
    toward ``required``), require a finding per required dimension, and fold as
    certification evidence. ``authorized`` is True only when every finding passed.
    """
    req = tuple(str(r) for r in required)
    all_findings = tuple(run_checks(request, context)) + tuple(findings)
    verdict = certify_dimensions(all_findings, required=req)
    return AuthorizationDecision(request, all_findings, verdict, req)


# --- the receipt -----------------------------------------------------------------------


def _finding_record(finding: Finding) -> Dict[str, Any]:
    return {
        "check": finding.check,
        "passed": finding.passed,
        "severity": finding.severity.value,
        "message": finding.message,
        "context": dict(finding.context),
    }


@dataclass(frozen=True)
class DecisionReceipt:
    """A deterministic record of one authorization decision, bound by digest.

    Built from an :class:`AuthorizationDecision` by :meth:`build`. ``body`` is the
    canonical content and ``digest`` its SHA-256 over :func:`canonical_json`, so the same
    decision on the same evidence yields the same bytes and the same digest. When an audit
    entry is supplied the receipt also carries that entry's ``seq`` and ``hash`` (the
    chain position and head the audit log's ``expected_head`` anchors accept).

    Stated exactly: tamper-evident when anchored (the digest changes if any byte of the
    body does); not an identity assertion (nothing here proves who produced it); not
    independently authenticated by its digest alone (anyone can compute a digest over
    bytes of their choosing). No timestamp is included unless one is supplied.
    """

    body: Mapping[str, Any]
    digest: str

    @classmethod
    def build(
        cls,
        decision: AuthorizationDecision,
        *,
        context: Optional[AuthorizationContext] = None,
        policy_version: Optional[str] = None,
        manifest_fingerprint: Optional[str] = None,
        audit_entry: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> "DecisionReceipt":
        from . import __version__ as recusal_version  # local import: no cycle at load

        body: Dict[str, Any] = {
            "receipt_version": 1,
            "recusal_version": recusal_version,
            "action_fingerprint": decision.request.action_fingerprint,
            "arguments_fingerprint": decision.request.arguments_fingerprint,
            "principal_label": decision.request.principal,
            "tool": decision.request.tool,
            "operation": decision.request.operation,
            "resource": decision.request.resource,
            "required": list(decision.required),
            "findings": [_finding_record(f) for f in decision.findings],
            "decision": decision.verdict.decision.value,
            "authorized": decision.authorized,
            "policy_version": policy_version,
            "manifest_fingerprint": manifest_fingerprint,
            "evidence_provenance": context.provenance() if context is not None else {},
            "audit_seq": None,
            "audit_head": None,
            "timestamp": timestamp,
        }
        if audit_entry is not None:
            seq = audit_entry.get("seq")
            head = audit_entry.get("hash")
            if not isinstance(seq, int) or not isinstance(head, str) or not head:
                raise ValueError("audit_entry must carry an integer seq and a hash")
            body["audit_seq"] = seq
            body["audit_head"] = head
        return cls(body, fingerprint(body))

    def to_json(self) -> str:
        """The receipt as one canonical JSON document, digest included."""
        return canonical_json({"body": dict(self.body), "digest": self.digest})

    @classmethod
    def from_json(cls, text: str) -> "DecisionReceipt":
        loaded = json.loads(text)
        if not isinstance(loaded, dict) or "body" not in loaded or "digest" not in loaded:
            raise ValueError("not a receipt document")
        return cls(loaded["body"], str(loaded["digest"]))

    def digest_matches(self) -> bool:
        """True when ``digest`` is the SHA-256 of the canonical body. This detects a
        changed byte; it does not establish who produced the receipt."""
        return fingerprint(dict(self.body)) == self.digest


__all__ = [
    "PROVENANCE_CLAUDE_PRETOOLUSE",
    "DIMENSIONS",
    "DEFAULT_DIMENSIONS",
    "Supplied",
    "Constraints",
    "ActionRequest",
    "AuthorizationContext",
    "AuthorizationDecision",
    "DecisionReceipt",
    "canonical_json",
    "fingerprint",
    "run_checks",
    "certify_dimensions",
    "certify_authorization",
    "check_principal",
    "check_subject",
    "check_tool_operation",
    "check_arguments",
    "check_expiry",
    "check_budget",
    "check_replay",
    "check_policy_binding",
    "check_manifest_binding",
]
