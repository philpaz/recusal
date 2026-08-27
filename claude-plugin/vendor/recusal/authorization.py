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

**Malformed evidence is rejected, never normalized.** Every value that enters a request,
a context, a fingerprint or a receipt is held to a strict JSON domain: object keys are
strings, leaves are strings, booleans, finite numbers, or null, containers are objects
and arrays. Sets, arbitrary objects, non-string keys, NaN and infinities raise
``ValueError`` at the public entry point, :func:`canonical_json` included, so nothing is
converted through ``str`` on the way to a digest: canonicalization performs no lossy
normalization, and digest collision resistance is inherited from SHA-256. Numeric bounds
compare JSON numbers only: a numeric string is a different value, and a boolean is not a
number.

**The composition invariant.** :func:`certify_evidence` refuses an empty set, and that is
all it can promise: one unrelated passing finding satisfies it. Authorization needs more.
Every built-in dimension is mandatory and cannot be switched off; :func:`certify_dimensions`
requires a named finding for each one (and for any ``additional_required`` custom
dimension) and synthesizes a failed CRITICAL ``authorization.dimension_missing`` for each
absent, so a request missing principal or resource evidence cannot be certified by the
presence of some other evidence. :func:`certify_authorization` runs the built-in checks
and applies that invariant; it is the entry point an adopter wires in.

**A receipt is derived, not described.** :class:`AuthorizationDecision` carries the final
finding set (synthesized findings included) and a snapshot of the adjudicated evidence
provenance, policy version and manifest fingerprint, all captured inside
:func:`certify_authorization`. :meth:`DecisionReceipt.build` takes only the decision, so
no caller can label a receipt with a policy, manifest or provenance the decision was not
made under, and :meth:`DecisionReceipt.verify` re-folds the receipt's own findings through
the kernel and requires the result to equal the recorded decision.

Standard library only. Every check is a pure function of its arguments.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .evidence import Finding, Severity, Verdict, certify_evidence

#: Provenance label the Claude Code hook assigns to fields it copies out of a
#: ``PreToolUse`` event (``agent_id``, ``agent_type``, ``permission_mode``). The label
#: says where a value came from; it says nothing about whether to trust it.
PROVENANCE_CLAUDE_PRETOOLUSE = "claude_pretooluse_event"

#: The authorization dimensions, in the order their findings are produced. Each maps to
#: one built-in check and one check name, ``authorization.<dimension>``. All of them are
#: mandatory in :func:`certify_authorization`; none can be disabled.
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

_CHECK_PREFIX = "authorization."
_MISSING_CHECK = _CHECK_PREFIX + "dimension_missing"
_BUILTIN_CHECKS = frozenset(_CHECK_PREFIX + d for d in DIMENSIONS)
_RESERVED_CHECKS = _BUILTIN_CHECKS | {_MISSING_CHECK}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DIMENSION_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

#: The receipt format this module writes and accepts.
RECEIPT_VERSION = 1


# --- the strict JSON domain --------------------------------------------------------------


def is_json_number(value: Any) -> bool:
    """A JSON number: an int that is not a bool, or a finite float."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


@dataclass(frozen=True)
class FrozenMapping:
    """An immutable, key-sorted JSON object; the frozen form of a dict. Distinct from a
    tuple so an empty object and an empty array never read as each other. Validated on
    construction: string keys, strictly ascending and unique, every value inside the
    frozen JSON domain, so a hand-built instance cannot carry what ``Supplied`` refuses."""

    items: Tuple[Tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise ValueError("FrozenMapping.items must be a tuple of (key, value) pairs")
        previous: Optional[str] = None
        for pair in self.items:
            if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[0], str):
                raise ValueError("FrozenMapping.items must be (str, value) pairs")
            key, value = pair
            if previous is not None and key <= previous:
                raise ValueError("FrozenMapping keys must be strictly ascending and unique")
            previous = key
            _check_frozen(value, f"FrozenMapping[{key!r}]")

    def plain(self) -> Dict[str, Any]:
        return {k: _thaw(v) for k, v in self.items}


def _check_frozen(value: Any, path: str) -> None:
    """Validate a value already in frozen form (tuples, FrozenMapping, JSON leaves)."""
    if isinstance(value, FrozenMapping):
        return  # validated by its own constructor
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not is_json_number(value):
            raise ValueError(f"{path}: {value!r} is not a finite JSON number")
        return
    if isinstance(value, tuple):
        for i, item in enumerate(value):
            _check_frozen(item, f"{path}[{i}]")
        return
    raise ValueError(f"{path}: {type(value).__name__} is outside the JSON domain")


def _freeze(value: Any, path: str = "value") -> Any:
    """Convert a JSON-domain value into an immutable, order-stable form, or raise
    ``ValueError`` naming the offending path. Never converts through ``str``."""
    if isinstance(value, FrozenMapping):
        return value
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if not is_json_number(value):
            raise ValueError(f"{path}: {value!r} is not a finite JSON number")
        return value
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: object key {key!r} is not a string")
            items.append((key, _freeze(item, f"{path}.{key}")))
        return FrozenMapping(tuple(sorted(items, key=lambda kv: kv[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{path}[{i}]") for i, item in enumerate(value))
    raise ValueError(f"{path}: {type(value).__name__} is outside the JSON domain")


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze`: ordinary dicts and lists, for serialization."""
    if isinstance(value, FrozenMapping):
        return value.plain()
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    """The one serialization every fingerprint and receipt in this module uses. The value
    is validated against the strict JSON domain first (non-string keys, sets, objects,
    NaN and infinities raise ``ValueError``), then encoded with sorted keys, no
    whitespace, ASCII only, ``allow_nan=False`` and no ``default=``."""
    plain = _thaw(_freeze(value, "canonical_json"))
    return json.dumps(
        plain, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def fingerprint(value: Any) -> str:
    """SHA-256 hex digest of :func:`canonical_json` of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _nonempty_str(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a nonempty string")
    return value


def _optional_str(value: Any, what: str) -> Optional[str]:
    if value is None:
        return None
    return _nonempty_str(value, what)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.fullmatch(value))


# --- the evidence types --------------------------------------------------------------------


@dataclass(frozen=True)
class Constraints:
    """The bounds an authorization was granted under. Every member is optional; an
    absent member means "no constraint of this kind was declared", which the matching
    check reports as a passing finding that SAYS so, never silently.

    - ``fields``: the only argument keys the action may carry (unique, nonempty strings).
    - ``max_calls``: the call budget, a nonnegative integer; compared against supplied
      ``calls_so_far``.
    - ``expires_at``: ISO 8601, timezone-aware; compared against supplied ``now``.
    - ``max_value`` / ``value_field``: a finite upper bound on one JSON-number argument.
    """

    fields: Optional[Tuple[str, ...]] = None
    max_calls: Optional[int] = None
    expires_at: Optional[str] = None
    max_value: Optional[float] = None
    value_field: Optional[str] = None

    def __post_init__(self) -> None:
        if self.fields is not None:
            if isinstance(self.fields, str) or not isinstance(self.fields, (list, tuple)):
                raise ValueError("Constraints.fields must be a list or tuple of strings")
            names = tuple(self.fields)
            for name in names:
                _nonempty_str(name, "Constraints.fields member")
            if len(set(names)) != len(names):
                raise ValueError("Constraints.fields must not repeat a name")
            object.__setattr__(self, "fields", names)
        if self.max_calls is not None:
            if isinstance(self.max_calls, bool) or not isinstance(self.max_calls, int):
                raise ValueError("Constraints.max_calls must be an integer")
            if self.max_calls < 0:
                raise ValueError("Constraints.max_calls must be nonnegative")
        _optional_str(self.expires_at, "Constraints.expires_at")
        if (self.max_value is None) != (self.value_field is None):
            raise ValueError("Constraints.max_value and value_field must be given together")
        if self.max_value is not None:
            if not is_json_number(self.max_value):
                raise ValueError("Constraints.max_value must be a finite number")
            _nonempty_str(self.value_field, "Constraints.value_field")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": list(self.fields) if self.fields is not None else None,
            "max_calls": self.max_calls,
            "expires_at": self.expires_at,
            "max_value": self.max_value,
            "value_field": self.value_field,
        }


@dataclass(frozen=True)
class Supplied:
    """One item of authority evidence and where it came from.

    ``provenance`` is a label the adopter assigns (``"operator_config"``,
    ``"session_store"``, :data:`PROVENANCE_CLAUDE_PRETOOLUSE`, ...). It is recorded in
    every finding that consumes the item and in the receipt, so a reviewer can see what
    each dimension was decided on. It is a label, not an attestation. ``value`` must be
    in the JSON domain, or a :class:`Constraints`; anything else raises.
    """

    value: Any
    provenance: str

    def __post_init__(self) -> None:
        _nonempty_str(self.provenance, "Supplied.provenance")
        if not isinstance(self.value, Constraints):
            object.__setattr__(self, "value", _freeze(self.value, "Supplied.value"))

    def plain(self) -> Any:
        """The value in ordinary Python containers (dicts and lists)."""
        return _thaw(self.value)


@dataclass(frozen=True)
class ActionRequest:
    """The action an agent proposes, as the adjudicator sees it.

    ``principal`` is the label the caller attributes the request to, often a runtime
    label; it is bound to a configured principal only through the context's trust rule.
    ``arguments`` are the call's arguments, held to the JSON domain and frozen on
    construction; the fingerprint is over their canonical JSON, so a receipt binds to
    the exact call without embedding it. ``nonce`` is the caller's replay token.
    """

    principal: str
    tool: str
    operation: str
    resource: Optional[str] = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    nonce: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("principal", "tool", "operation"):
            _nonempty_str(getattr(self, name), f"ActionRequest.{name}")
        _optional_str(self.resource, "ActionRequest.resource")
        _optional_str(self.nonce, "ActionRequest.nonce")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("ActionRequest.arguments must be a mapping")
        object.__setattr__(self, "arguments", _freeze(self.arguments, "ActionRequest.arguments"))

    def arguments_plain(self) -> Dict[str, Any]:
        thawed = _thaw(self.arguments)
        return thawed if isinstance(thawed, dict) else {}

    @property
    def arguments_fingerprint(self) -> str:
        return fingerprint(self.arguments_plain())

    @property
    def action_fingerprint(self) -> str:
        """Binds principal label, tool, operation, resource, arguments and nonce."""
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
    - ``allowed_operations``: mapping tool -> list of permitted operation names.
    - ``constraints``: a :class:`Constraints`.
    - ``now``: ISO 8601, timezone-aware. Supplied, never read here.
    - ``calls_so_far``: how many calls this authorization has already spent. Supplied.
    - ``used_nonces``: nonces already consumed, a list of strings. Supplied.
    - ``approval``: mapping with ``policy_version`` and optionally ``manifest_fingerprint``
      (64 lowercase hex), the identities the authorization was approved under.
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
    subject = active.plain()
    if not isinstance(subject, str) or not subject:
        return _fail(
            "subject", "active subject is not a nonempty string", provenance=active.provenance
        )
    if request.resource is None:
        return _fail(
            "subject",
            "the action names no resource, so it cannot be bound to the active subject",
            active=subject,
            provenance=active.provenance,
        )
    if request.resource != subject:
        return _fail(
            "subject",
            f"action targets {request.resource} but the active subject is {subject}",
            target=request.resource,
            active=subject,
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
    if not isinstance(ops, list) or not all(isinstance(o, str) and o for o in ops):
        return _fail(
            "tool_operation",
            f"allowed operations for {request.tool!r} is not a list of operation names",
            tool=request.tool,
            provenance=allowed.provenance,
        )
    if request.operation not in ops:
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
    """The arguments stay inside the declared field set and value bound.

    The value bound compares JSON numbers only: a numeric string, a boolean, or a
    non-finite float is not a number here and refuses, never coerces.
    """
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
    bound, value_field = constraints.max_value, constraints.value_field
    if bound is not None and value_field is not None:
        raw = args.get(value_field)
        # spelled out rather than through is_json_number so the type narrows for the
        # comparison below; the predicate is identical
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            return _fail(
                "arguments",
                f"argument {value_field!r} is not a finite JSON number, so the value bound "
                "cannot be verified",
                value_field=value_field,
                provenance=provenance,
            )
        if raw > bound:
            return _fail(
                "arguments",
                f"argument {value_field!r} = {raw} exceeds the bound {bound}",
                value_field=value_field,
                value=raw,
                max_value=bound,
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
    if not isinstance(used, list) or not all(isinstance(n, str) for n in used):
        return _fail(
            "replay",
            "used_nonces is not a list of strings",
            provenance=context.used_nonces.provenance,
        )
    if request.nonce in used:
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
    if not isinstance(current, str) or not current:
        return _fail(
            "policy_binding",
            "current policy_version is not a nonempty string",
            provenance=context.policy_version.provenance,
        )
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
    stated; one side present without the other refuses; a fingerprint that is not 64
    lowercase hex characters refuses.
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
    for label, value in (("approved", approved), ("current", current)):
        if not _is_hex64(value):
            return _fail(
                "manifest_binding",
                f"{label} manifest fingerprint is not a 64-character lowercase sha256",
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


#: The context fields each dimension decides on. Every built-in finding records, under
#: ``evidence``, the provenance of exactly these fields as supplied, so a receipt's
#: evidence provenance can be reconciled against its findings field by field.
EVIDENCE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "principal": ("trusted_principals",),
    "subject": ("active_subject",),
    "tool_operation": ("allowed_operations",),
    "arguments": ("constraints",),
    "expiry": ("constraints", "now"),
    "budget": ("constraints", "calls_so_far"),
    "replay": ("used_nonces",),
    "policy_binding": ("approval", "policy_version"),
    "manifest_binding": ("approval", "manifest_fingerprint"),
}


def _current_policy(context: AuthorizationContext) -> Optional[str]:
    """The policy version in force as supplied, or None when absent or malformed."""
    if context.policy_version is None:
        return None
    value = context.policy_version.plain()
    return value if isinstance(value, str) and value else None


def _current_manifest(context: AuthorizationContext) -> Optional[str]:
    """The manifest fingerprint in force as supplied, or None when absent or malformed."""
    if context.manifest_fingerprint is None:
        return None
    value = context.manifest_fingerprint.plain()
    return value if _is_hex64(value) else None


def _evidence_of(dimension: str, provenance: Mapping[str, str]) -> Dict[str, str]:
    return {f: provenance[f] for f in EVIDENCE_FIELDS[dimension] if f in provenance}


def run_checks(request: ActionRequest, context: AuthorizationContext) -> List[Finding]:
    """Every built-in check, in :data:`DIMENSIONS` order. A check that raises (which
    would be a bug, not evidence) becomes a failed CRITICAL finding for its dimension:
    the gate fails closed and names why.

    Each finding's context carries ``evidence`` (the provenance of the fields its
    dimension decides on, see :data:`EVIDENCE_FIELDS`); the policy and manifest findings
    also carry ``current_policy_version`` / ``current_manifest_fingerprint`` as supplied,
    passing or failing, so a receipt can be reconciled against its own findings.
    """
    provenance = context.provenance()
    findings: List[Finding] = []
    for dimension in DIMENSIONS:
        try:
            finding = _CHECKS[dimension](request, context)
        except Exception as exc:  # noqa: BLE001 - a broken check must refuse, not disable the gate
            finding = _fail(dimension, f"check raised {type(exc).__name__}: {exc}")
        extra: Dict[str, Any] = {"evidence": _evidence_of(dimension, provenance)}
        if dimension == "policy_binding":
            extra["current_policy_version"] = _current_policy(context)
        elif dimension == "manifest_binding":
            extra["current_manifest_fingerprint"] = _current_manifest(context)
        findings.append(
            Finding(
                finding.check,
                finding.severity,
                finding.passed,
                finding.message,
                {**finding.context, **extra},
            )
        )
    return findings


# --- composition: every dimension evidenced, or refused --------------------------------


def _validate_extra_dimensions(additional_required: Any) -> Tuple[str, ...]:
    """The custom dimension names, validated: nonempty, well-formed, unique, and never a
    built-in (built-ins are always required and cannot be named here)."""
    if isinstance(additional_required, str) or not isinstance(additional_required, (list, tuple)):
        raise ValueError("additional_required must be a list or tuple of names")
    extra: List[str] = []
    for name in additional_required:
        if not isinstance(name, str) or not name or not _DIMENSION_NAME.fullmatch(name):
            raise ValueError(f"additional_required name {name!r} is not a valid dimension name")
        if name in DIMENSIONS or name == "dimension_missing":
            raise ValueError(f"{name!r} is a built-in dimension and is always required")
        if name in extra:
            raise ValueError(f"additional_required repeats {name!r}")
        extra.append(name)
    return tuple(extra)


def _validate_findings(findings: Any) -> Tuple[Finding, ...]:
    if isinstance(findings, (str, bytes, Mapping)):
        raise ValueError("findings must be an iterable of Finding objects")
    try:
        items = tuple(findings)
    except TypeError:
        raise ValueError("findings must be an iterable of Finding objects") from None
    for finding in items:
        if not isinstance(finding, Finding):
            raise ValueError(f"findings must be Finding objects, got {type(finding).__name__}")
    return items


def _augment(findings: Tuple[Finding, ...], required: Tuple[str, ...]) -> Tuple[Finding, ...]:
    """Append a failed CRITICAL ``dimension_missing`` for each required dimension with no
    finding named ``authorization.<dimension>``. This is the full set a verdict is folded
    from, and the set a receipt carries."""
    present = {f.check for f in findings}
    synthesized: List[Finding] = []
    for dimension in required:
        if _CHECK_PREFIX + dimension not in present:
            synthesized.append(
                Finding.fail(
                    _MISSING_CHECK,
                    severity=Severity.CRITICAL,
                    message=f"required authorization dimension {dimension!r} produced no finding",
                    dimension=dimension,
                )
            )
    return findings + tuple(synthesized)


def certify_dimensions(
    findings: Iterable[Finding],
    *,
    additional_required: Sequence[str] = (),
) -> Verdict:
    """Fold ``findings`` as certification evidence, requiring one named finding per
    built-in dimension and per ``additional_required`` custom dimension.

    For each required dimension with no finding named ``authorization.<dimension>``, a
    failed CRITICAL ``authorization.dimension_missing`` finding is synthesized before the
    fold, so an unrelated passing finding can never certify a request that lacks evidence
    on a required dimension. Built-in dimensions cannot be removed from the requirement.
    Malformed ``findings`` or names raise ``ValueError``. The fold itself is
    :func:`recusal.certify_evidence`.
    """
    required = DIMENSIONS + _validate_extra_dimensions(additional_required)
    return certify_evidence(_augment(_validate_findings(findings), required))


@dataclass(frozen=True)
class AuthorizationDecision:
    """The complete outcome of one adjudication, captured inside
    :func:`certify_authorization` so a receipt can be derived from it and nothing else.

    ``findings`` is the FINAL set the verdict was folded from: built-in checks, adopter
    findings, and any synthesized ``dimension_missing`` findings. ``evidence_provenance``,
    ``policy_version`` and ``manifest_fingerprint`` are the snapshot of what was adjudicated
    (the context's provenance labels, and the policy and manifest in force as supplied),
    not values a caller can set afterwards.
    """

    request: ActionRequest
    findings: Tuple[Finding, ...]
    verdict: Verdict
    required: Tuple[str, ...]
    evidence_provenance: FrozenMapping
    policy_version: Optional[str]
    manifest_fingerprint: Optional[str]

    @property
    def authorized(self) -> bool:
        return self.verdict.passed

    @property
    def principal(self) -> Optional[str]:
        """The configured principal the label resolved to, or None when it did not."""
        for finding in self.findings:
            if finding.check == _CHECK_PREFIX + "principal" and finding.passed:
                value = finding.context.get("principal")
                return value if isinstance(value, str) else None
        return None


def certify_authorization(
    request: ActionRequest,
    context: AuthorizationContext,
    *,
    additional_required: Sequence[str] = (),
    findings: Iterable[Finding] = (),
) -> AuthorizationDecision:
    """Adjudicate ``request`` against ``context``: run every built-in check, add any
    adopter ``findings`` (custom dimensions named ``authorization.<name>``), require a
    finding per built-in dimension and per ``additional_required`` name, and fold as
    certification evidence. ``authorized`` is True only when every finding passed.

    Built-in dimensions are always required and cannot be disabled here. An adopter
    finding whose check name is a built-in dimension's (or ``dimension_missing``) raises:
    a custom finding must not impersonate or replace a built-in one.
    """
    if not isinstance(request, ActionRequest):
        raise ValueError("request must be an ActionRequest")
    if not isinstance(context, AuthorizationContext):
        raise ValueError("context must be an AuthorizationContext")
    required = DIMENSIONS + _validate_extra_dimensions(additional_required)
    extra = _validate_findings(findings)
    for finding in extra:
        if finding.check in _RESERVED_CHECKS:
            raise ValueError(f"finding {finding.check!r} impersonates a built-in dimension")
    all_findings = _augment(tuple(run_checks(request, context)) + extra, required)
    verdict = certify_evidence(all_findings)
    return AuthorizationDecision(
        request=request,
        findings=all_findings,
        verdict=verdict,
        required=required,
        evidence_provenance=_freeze(context.provenance(), "evidence_provenance"),
        policy_version=_current_policy(context),
        manifest_fingerprint=_current_manifest(context),
    )


# --- the receipt -----------------------------------------------------------------------


def _finding_record(finding: Finding) -> Dict[str, Any]:
    return {
        "check": finding.check,
        "passed": finding.passed,
        "severity": finding.severity.value,
        "message": finding.message,
        "context": _thaw(_freeze(dict(finding.context), "finding.context")),
    }


#: Every field a receipt body carries. A body with a field missing, or one not listed
#: here, is malformed.
RECEIPT_FIELDS: Tuple[str, ...] = (
    "receipt_version",
    "recusal_version",
    "action_fingerprint",
    "arguments_fingerprint",
    "principal_label",
    "principal",
    "tool",
    "operation",
    "resource",
    "required",
    "findings",
    "decision",
    "authorized",
    "policy_version",
    "manifest_fingerprint",
    "evidence_provenance",
    "audit_seq",
    "audit_head",
    "timestamp",
)

_FINDING_FIELDS = frozenset({"check", "passed", "severity", "message", "context"})
_SEVERITIES = frozenset(s.value for s in Severity)
_DECISIONS = ("PASS", "RETRY", "FAIL")


def _is_str(value: Any, nonempty: bool = True) -> bool:
    return isinstance(value, str) and (bool(value) or not nonempty)


def _is_opt_str(value: Any) -> bool:
    return value is None or _is_str(value)


def _is_seq_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_finding_records(findings: Any, problems: List[str]) -> List[Finding]:
    """Shape-check each finding record, recursively for its context, and rebuild the
    Finding objects so the verdict can be recomputed."""
    rebuilt: List[Finding] = []
    if not isinstance(findings, list):
        problems.append("findings is not a list")
        return rebuilt
    for i, f in enumerate(findings):
        if not isinstance(f, dict) or set(f) != _FINDING_FIELDS:
            problems.append(f"findings[{i}] does not have exactly the finding fields")
            continue
        ok = True
        if not _is_str(f["check"]):
            problems.append(f"findings[{i}].check is not a nonempty string")
            ok = False
        if not isinstance(f["passed"], bool):
            problems.append(f"findings[{i}].passed is not a boolean")
            ok = False
        if not isinstance(f["severity"], str) or f["severity"] not in _SEVERITIES:
            problems.append(f"findings[{i}].severity is not a severity")
            ok = False
        if not isinstance(f["message"], str):
            problems.append(f"findings[{i}].message is not a string")
            ok = False
        if not isinstance(f["context"], dict):
            problems.append(f"findings[{i}].context is not an object")
            ok = False
        else:
            try:
                _freeze(f["context"], f"findings[{i}].context")
            except ValueError as exc:
                problems.append(str(exc))
                ok = False
        if ok:
            rebuilt.append(
                Finding(
                    f["check"], Severity(f["severity"]), f["passed"], f["message"], f["context"]
                )
            )
    return rebuilt


def _reconcile(body: Dict[str, Any], findings: List[Finding], problems: List[str]) -> None:
    """The relationships a well-shaped body must also satisfy: the findings re-fold to
    the recorded decision, every built-in dimension appears exactly once, every custom
    required dimension is evidenced or explicitly missing, and nothing else is."""
    required = body["required"]
    extras = tuple(required[len(DIMENSIONS) :])
    try:
        _validate_extra_dimensions(extras)
    except ValueError as exc:
        problems.append(f"required: {exc}")
        return
    counts: Dict[str, int] = {}
    missing_names: List[str] = []
    for f in findings:
        counts[f.check] = counts.get(f.check, 0) + 1
        if f.check == _MISSING_CHECK:
            name = f.context.get("dimension")
            missing_names.append(name if isinstance(name, str) else "")
    for dimension in DIMENSIONS:
        if counts.get(_CHECK_PREFIX + dimension, 0) != 1:
            problems.append(f"built-in dimension {dimension!r} must appear exactly once")
    for dimension in required:
        evidenced = counts.get(_CHECK_PREFIX + dimension, 0) > 0
        declared_missing = dimension in missing_names
        if evidenced == declared_missing:
            problems.append(
                f"required dimension {dimension!r} must be either evidenced or declared missing"
            )
    for name in missing_names:
        if name not in required:
            problems.append(f"dimension_missing names {name!r}, which is not required")
    # Metadata must agree with the findings it was derived from: the resolved principal,
    # the policy and manifest in force, and the provenance of every consumed field.
    by_check = {f.check: f for f in findings}
    principal_finding = by_check.get(_CHECK_PREFIX + "principal")
    if principal_finding is not None:
        resolved = principal_finding.context.get("principal") if principal_finding.passed else None
        expected_principal = resolved if isinstance(resolved, str) and resolved else None
        if body["principal"] != expected_principal:
            problems.append("principal does not match the principal finding")
    policy_finding = by_check.get(_CHECK_PREFIX + "policy_binding")
    if policy_finding is not None:
        if body["policy_version"] != policy_finding.context.get("current_policy_version"):
            problems.append("policy_version does not match the policy_binding finding")
    manifest_finding = by_check.get(_CHECK_PREFIX + "manifest_binding")
    if manifest_finding is not None:
        current = manifest_finding.context.get("current_manifest_fingerprint")
        if body["manifest_fingerprint"] != current:
            problems.append("manifest_fingerprint does not match the manifest_binding finding")
    provenance = body["evidence_provenance"]
    for dimension in DIMENSIONS:
        finding = by_check.get(_CHECK_PREFIX + dimension)
        if finding is None:
            continue
        if finding.context.get("evidence") != _evidence_of(dimension, provenance):
            problems.append(
                f"evidence_provenance does not match the evidence recorded by the "
                f"{dimension!r} finding"
            )
    recomputed = certify_evidence(findings)
    if recomputed.decision.value != body["decision"]:
        problems.append(
            f"decision {body['decision']!r} does not match the findings, which fold to "
            f"{recomputed.decision.value!r}"
        )
    if body["authorized"] != recomputed.passed:
        problems.append("authorized does not match the findings")


def validate_receipt_body(body: Any) -> List[str]:
    """Every way a receipt body can be malformed or internally inconsistent, named.
    Empty means well-formed and reconciled: the findings re-fold to the recorded
    decision. It does not establish who produced the receipt."""
    problems: List[str] = []
    if not isinstance(body, dict):
        return ["body is not an object"]
    keys = set(body)
    for missing in [f for f in RECEIPT_FIELDS if f not in keys]:
        problems.append(f"missing field {missing!r}")
    for unknown in sorted(keys - set(RECEIPT_FIELDS)):
        problems.append(f"unknown field {unknown!r}")
    if problems:
        return problems
    if body["receipt_version"] != RECEIPT_VERSION or isinstance(body["receipt_version"], bool):
        problems.append(f"receipt_version is not {RECEIPT_VERSION}")
    if not _is_str(body["recusal_version"]):
        problems.append("recusal_version is not a nonempty string")
    for name in ("action_fingerprint", "arguments_fingerprint"):
        if not _is_hex64(body[name]):
            problems.append(f"{name} is not a 64-character lowercase sha256")
    for name in ("principal_label", "tool", "operation"):
        if not _is_str(body[name]):
            problems.append(f"{name} is not a nonempty string")
    for name in ("principal", "resource", "policy_version", "timestamp"):
        if not _is_opt_str(body[name]):
            problems.append(f"{name} is not null or a nonempty string")
    required_ok = True
    required = body["required"]
    if not isinstance(required, list) or not all(_is_str(r) for r in required):
        problems.append("required is not a list of names")
        required_ok = False
    elif tuple(required[: len(DIMENSIONS)]) != DIMENSIONS:
        problems.append("required does not begin with every built-in dimension in order")
        required_ok = False
    findings = _validate_finding_records(body["findings"], problems)
    if not isinstance(body["decision"], str) or body["decision"] not in _DECISIONS:
        problems.append("decision is not PASS, RETRY or FAIL")
    if not isinstance(body["authorized"], bool):
        problems.append("authorized is not a boolean")
    if body["manifest_fingerprint"] is not None and not _is_hex64(body["manifest_fingerprint"]):
        problems.append("manifest_fingerprint is not null or a 64-character lowercase sha256")
    prov = body["evidence_provenance"]
    if not isinstance(prov, dict) or not all(_is_str(k) and _is_str(v) for k, v in prov.items()):
        problems.append("evidence_provenance is not an object of nonempty strings")
    seq, head = body["audit_seq"], body["audit_head"]
    if (seq is None) != (head is None):
        problems.append("audit_seq and audit_head must both be present or both absent")
    if seq is not None and not _is_seq_int(seq):
        problems.append("audit_seq is not a nonnegative integer")
    if head is not None and not _is_hex64(head):
        problems.append("audit_head is not a 64-character lowercase sha256")
    try:
        canonical_json(body)
    except (TypeError, ValueError) as exc:
        problems.append(f"body is outside the JSON domain: {exc}")
    if not problems and required_ok:
        _reconcile(body, findings, problems)
    return problems


@dataclass(frozen=True)
class DecisionReceipt:
    """A deterministic record of one authorization decision, bound by digest.

    Built from an :class:`AuthorizationDecision` by :meth:`build`, or parsed by
    :meth:`from_json`; both refuse a malformed or internally inconsistent body, or a
    malformed digest, with ``ValueError``. Everything adjudicated (findings, decision,
    principal, policy version, manifest fingerprint, evidence provenance) is derived from
    the decision; only the audit anchor and an optional timestamp are supplied by the
    caller, because they are genuinely outside the adjudication. The body is frozen
    recursively; :attr:`body` returns a fresh plain copy. ``digest`` is SHA-256 over
    :func:`canonical_json` of the body, so the same decision on the same evidence yields
    the same bytes and the same digest. When an audit entry is supplied the receipt also
    carries that entry's ``seq`` and ``hash`` (the chain position and head the audit log's
    ``expected_head`` anchors accept).

    Stated exactly: tamper-evident when anchored (the digest changes if any byte of the
    body does); not an identity assertion (nothing here proves who produced it); not
    independently authenticated by its digest alone (anyone can compute a digest over
    bytes of their choosing). No timestamp is included unless one is supplied.
    """

    frozen_body: FrozenMapping
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_body, FrozenMapping):
            raise ValueError("DecisionReceipt body must be a FrozenMapping; use build/from_json")
        problems = validate_receipt_body(self.frozen_body.plain())
        if problems:
            raise ValueError("malformed receipt body: " + "; ".join(problems))
        if not _is_hex64(self.digest):
            raise ValueError("receipt digest is not a 64-character lowercase sha256")

    @property
    def body(self) -> Dict[str, Any]:
        """A fresh plain copy of the body; mutating it does not touch the receipt."""
        return self.frozen_body.plain()

    @classmethod
    def build(
        cls,
        decision: AuthorizationDecision,
        *,
        audit_entry: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> "DecisionReceipt":
        from . import __version__ as recusal_version  # local import: no cycle at load

        if not isinstance(decision, AuthorizationDecision):
            raise ValueError("decision must be an AuthorizationDecision")
        _optional_str(timestamp, "timestamp")
        body: Dict[str, Any] = {
            "receipt_version": RECEIPT_VERSION,
            "recusal_version": recusal_version,
            "action_fingerprint": decision.request.action_fingerprint,
            "arguments_fingerprint": decision.request.arguments_fingerprint,
            "principal_label": decision.request.principal,
            "principal": decision.principal,
            "tool": decision.request.tool,
            "operation": decision.request.operation,
            "resource": decision.request.resource,
            "required": list(decision.required),
            "findings": [_finding_record(f) for f in decision.findings],
            "decision": decision.verdict.decision.value,
            "authorized": decision.authorized,
            "policy_version": decision.policy_version,
            "manifest_fingerprint": decision.manifest_fingerprint,
            "evidence_provenance": decision.evidence_provenance.plain(),
            "audit_seq": None,
            "audit_head": None,
            "timestamp": timestamp,
        }
        if audit_entry is not None:
            if not isinstance(audit_entry, Mapping):
                raise ValueError("audit_entry must be a mapping")
            seq = audit_entry.get("seq")
            head = audit_entry.get("hash")
            if not _is_seq_int(seq):
                raise ValueError("audit_entry seq must be a nonnegative integer")
            if not _is_hex64(head):
                raise ValueError("audit_entry hash must be a 64-character lowercase sha256")
            body["audit_seq"] = seq
            body["audit_head"] = head
        frozen = _freeze(body, "receipt")
        return cls(frozen, fingerprint(frozen.plain()))

    def to_json(self) -> str:
        """The receipt as one canonical JSON document, digest included."""
        return canonical_json({"body": self.body, "digest": self.digest})

    @classmethod
    def from_json(cls, text: str) -> "DecisionReceipt":
        """Parse a receipt document strictly: exactly ``body`` and ``digest``, a
        well-formed and reconciled body, a 64-hex digest. Raises ``ValueError``
        otherwise. The digest is not checked here; call :meth:`verify`."""
        try:
            loaded = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"not a receipt document: {exc}") from None
        if not isinstance(loaded, dict) or set(loaded) != {"body", "digest"}:
            raise ValueError("not a receipt document: expected exactly body and digest")
        if not isinstance(loaded["body"], dict):
            raise ValueError("malformed receipt body: body is not an object")
        try:
            frozen = _freeze(loaded["body"], "receipt")
        except ValueError as exc:
            raise ValueError(f"malformed receipt body: {exc}") from None
        if not _is_hex64(loaded["digest"]):
            raise ValueError("receipt digest is not a 64-character lowercase sha256")
        return cls(frozen, loaded["digest"])

    def digest_matches(self) -> bool:
        """True when ``digest`` is the SHA-256 of the canonical body; False for any
        malformed state rather than raising. This detects a changed byte; it does not
        establish who produced the receipt."""
        try:
            return fingerprint(self.body) == self.digest
        except (TypeError, ValueError):
            return False

    def verify(self) -> Tuple[bool, List[str]]:
        """Schema, internal consistency and digest together: ``(intact, problems)``, in
        the same shape as :func:`recusal.audit.verify`. Intact means well-formed, the
        findings re-fold to the recorded decision, and the digest matches; it still says
        nothing about who produced the receipt."""
        problems = validate_receipt_body(self.body)
        if not self.digest_matches():
            problems.append("digest does not match the canonical body")
        return (not problems), problems


__all__ = [
    "PROVENANCE_CLAUDE_PRETOOLUSE",
    "DIMENSIONS",
    "RECEIPT_VERSION",
    "RECEIPT_FIELDS",
    "EVIDENCE_FIELDS",
    "Supplied",
    "FrozenMapping",
    "Constraints",
    "ActionRequest",
    "AuthorizationContext",
    "AuthorizationDecision",
    "DecisionReceipt",
    "is_json_number",
    "canonical_json",
    "fingerprint",
    "run_checks",
    "certify_dimensions",
    "certify_authorization",
    "validate_receipt_body",
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
