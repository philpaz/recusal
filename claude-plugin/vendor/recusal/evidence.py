"""
The evidence contract, the spine of Recusal.

Everything reduces to two objects:

    Finding   one observation about the work under adjudication.
    Verdict   the decision a set of findings adds up to.

Checks emit ``Finding``s. ``compute_verdict`` folds findings into a ``Verdict``.
The Claude adapter turns a ``Verdict`` into an allow / refuse decision on a tool
call. One object model, one pipeline, that is what makes this a governance layer
and not a pile of helpers.

The contract is the product: it is the typed, documented definition of what
"evidence" and "a verdict" *are*. Pure standard library, ``dataclasses`` + ``enum``,
no pydantic, no dependencies. See docs/EVIDENCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, List, Mapping, Tuple, Union


class Severity(str, Enum):
    """How bad a finding is *if it failed*, and therefore what it does to the verdict.

    Subclasses ``str`` so values compare and serialize as plain strings
    (``Severity.CRITICAL == "CRITICAL"``), keeping evidence JSON-clean.
    """

    CRITICAL = "CRITICAL"  # the work is wrong → FAIL, no retry
    ERROR = "ERROR"  # recoverable → RETRY once, with the failures as context
    WARNING = "WARNING"  # proceed, but record it as evidence
    INFO = "INFO"  # never blocks; recorded as a metric


# Back-compat alias for the original name.
RuleSeverity = Severity


def _as_severity(value: Union[Severity, str]) -> Severity:
    return value if isinstance(value, Severity) else Severity(str(value).upper())


# Strings that affirmatively mean *passed* for the ``status`` field. This is an
# allowlist on purpose: a ``status`` we do not positively recognize as passing is
# treated as a failure (fail closed), so a value like ``"failed"``/``"denied"``/
# ``"0"`` - anything a producer might emit for a bad outcome - cannot slip through
# as a pass the way a hardcoded fail-blocklist would let it.
_PASS_LIKE = frozenset({"pass", "passed", "ok", "okay", "success", "succeeded", "green"})

# Boolean-ish affirmative tokens accepted for a *string* ``passed`` value, alongside the
# ``status`` vocabulary above. Same allowlist posture: a string we do not positively
# recognize as passing (``"maybe"``, ``"unknown"``, ``""``) reads as a failure. A raw
# ``bool("false")`` is ``True`` (any non-empty string is truthy), so string values are
# never read with Python truthiness - a stringified ``"passed": "false"`` reading as a
# silent PASS is the exact failure mode this library exists to prevent.
_TRUE_LIKE = frozenset({"true", "t", "1", "yes", "y", "on"}) | _PASS_LIKE


def _as_bool(value: Any, *, strict: bool = False) -> bool:
    """Interpret a loose ``passed`` value. A genuine ``bool``, or an ``int`` ``0``/``1``,
    is used directly; a *string* passes only when it is an affirmative token
    (``"true"``/``"yes"``/``"1"``/``"pass"``/...). Anything unrecognized (``"maybe"``,
    ``"false"``, empty) fails closed, mirroring how ``status`` is read.

    Any other object (a list, a dict, ``None``, an arbitrary instance) is not an outcome.
    Python truthiness is never applied to it: a non-empty container or a bare object would
    read as a pass, which is the failure this library exists to prevent. In strict mode it
    is refused (``TypeError``); in lenient mode it reads as a failure."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_LIKE
    if strict:
        raise TypeError(
            f"'passed' must be a bool, 0/1 or an outcome token, not {type(value).__name__}"
        )
    return False


def _validated_passed(value: Any) -> bool:
    """The ``passed`` an already-constructed Finding may carry: a ``bool``, or exactly
    ``0``/``1``. A string, a container or any other object is refused, so a Finding that
    says ``passed="no"`` can never fold as a pass."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise TypeError(
        f"Finding.passed must be a bool, not {type(value).__name__}; build loose evidence "
        "through Finding.coerce"
    )


@dataclass(frozen=True)
class Finding:
    """One observation about the work under adjudication.

    ``severity`` is how bad it is *if it failed*; ``passed`` is whether the check
    held. A passed CRITICAL check is fine; a failed CRITICAL check refuses the work.
    ``context`` carries arbitrary structured detail (counts, columns, ids, …).
    """

    check: str
    severity: Severity
    passed: bool
    message: str = ""
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate at construction, so a malformed Finding cannot exist.

        ``severity`` is normalized to the :class:`Severity` enum (a string such as
        ``"CRITICAL"`` is accepted; an unknown level raises ``ValueError``) and
        ``passed`` must be a ``bool`` (``TypeError`` otherwise). Before this check, a
        Finding built with a string severity, or with ``passed="no"``, folded as a PASS
        in :func:`compute_verdict`: the fold compared the enum by identity and read the
        outcome by truthiness, so a failed CRITICAL finding went uncounted.
        """
        object.__setattr__(self, "severity", _as_severity(self.severity))
        object.__setattr__(self, "passed", _validated_passed(self.passed))

    @classmethod
    def ok(
        cls,
        check: str,
        *,
        severity: Union[Severity, str] = Severity.INFO,
        message: str = "",
        **context: Any,
    ) -> "Finding":
        return cls(check, _as_severity(severity), True, message, dict(context))

    @classmethod
    def fail(
        cls,
        check: str,
        *,
        severity: Union[Severity, str] = Severity.ERROR,
        message: str = "",
        **context: Any,
    ) -> "Finding":
        return cls(check, _as_severity(severity), False, message, dict(context))

    @classmethod
    def coerce(cls, obj: Union["Finding", Mapping[str, Any]], *, strict: bool = False) -> "Finding":
        """Accept a Finding as-is, or a loose evidence dict, and return a Finding.

        The dict shape is the ergonomic input form:
        ``{"severity": "CRITICAL", "status": "fail"|"pass"|"warn"|"error" (or
           "passed": bool), "message": ..., "check"|"type": ..., ...context}``.

        **Footgun:** a dict with *no* ``status`` or ``passed`` key defaults to
        ``passed=True``, so ``{"severity": "CRITICAL", "message": "..."}`` reads as a
        *passing* check, not a failure. Pass ``strict=True`` (or
        ``compute_verdict(..., strict=True)``) to reject such ambiguous evidence rather
        than silently pass it, which is the safer choice when wiring an adapter.

        A ``passed`` value is read for intent, not raw truthiness: a *string* counts as
        a pass only when it is an affirmative token (``"true"``/``"yes"``/``"1"``/
        ``"pass"``/...); a stringified ``"false"``/``"no"``/``"0"`` - or an unrecognized
        token like ``"maybe"`` - counts as a failure (a bare ``bool("false")`` is
        ``True``), so a failing check serialized loosely cannot slip through as a pass.
        A ``status`` value is read against the same pass *allowlist* posture: only an
        affirmative token (``"pass"``/``"passed"``/``"ok"``/…) reads as passing, so a
        ``status`` of ``"failed"``/``"denied"``/``"0"`` - or any token not recognized as
        passing - fails closed rather than slipping through.
        """
        if isinstance(obj, Finding):
            return _validated(obj)
        if isinstance(obj, Mapping):
            severity = _as_severity(obj.get("severity", Severity.INFO))
            if "passed" in obj:
                passed = _as_bool(obj["passed"], strict=strict)
            elif "status" in obj:
                # Fail closed on any status we do not positively recognize as passing
                # (see _PASS_LIKE): "failed"/"false"/"denied"/… must not read as a pass.
                passed = str(obj["status"]).strip().lower() in _PASS_LIKE
            elif strict:
                raise ValueError(
                    "ambiguous evidence dict: no 'status' or 'passed' key. In strict mode "
                    "you must state the outcome explicitly so a failure cannot pass silently."
                )
            else:
                passed = True  # lenient default: absence of a status reads as a pass
            check = obj.get("check") or obj.get("type") or "check"
            message = obj.get("message", "")
            known = {"severity", "status", "passed", "message", "check", "type"}
            context = {k: v for k, v in obj.items() if k not in known}
            return cls(str(check), severity, passed, str(message), context)
        raise TypeError(f"cannot coerce {type(obj).__name__} into a Finding")


def _validated(finding: Finding) -> Finding:
    """Re-check an existing Finding at the fold boundary.

    Construction already validates, but a frozen dataclass can still be altered through
    ``object.__setattr__``, and an enum member from a reloaded module is equal to but not
    identical with ours. The fold therefore trusts no instance it did not just check:
    severity is normalized by value and ``passed`` must be a bool, or the fold refuses.
    """
    severity = _as_severity(finding.severity)
    passed = _validated_passed(finding.passed)
    if severity is finding.severity and passed is finding.passed:
        return finding
    return Finding(finding.check, severity, passed, finding.message, finding.context)


class Decision(str, Enum):
    """The verdict's binding outcome."""

    PASS = "PASS"  # certified, proceed
    RETRY = "RETRY"  # recoverable, try once more with the failures as context
    FAIL = "FAIL"  # refused, terminal


@dataclass(frozen=True)
class Verdict:
    """The decision a set of findings adds up to. Deterministic and auditable."""

    decision: Decision
    highest_severity: Severity
    failures: Tuple[Finding, ...]  # the findings that forced FAIL / RETRY
    warnings: Tuple[Finding, ...]  # WARNING-level findings (informational)
    metrics: Tuple[Finding, ...]  # INFO-level findings (calibration only)
    message: str

    @property
    def passed(self) -> bool:
        return self.decision is Decision.PASS

    @property
    def refused(self) -> bool:
        return self.decision is Decision.FAIL

    @property
    def retryable(self) -> bool:
        return self.decision is Decision.RETRY

    def reasons(self) -> str:
        """The specific failure messages behind a non-PASS verdict, what a caller
        (or an agent) needs to actually correct the work. Falls back to the summary."""
        detail = "; ".join(f.message or f.check for f in self.failures if f.message or f.check)
        return detail or self.message


def compute_verdict(
    findings: Iterable[Union[Finding, Mapping[str, Any]]], *, strict: bool = False
) -> Verdict:
    """Fold an iterable of findings (Finding objects or loose dicts) into one verdict.

    Decision rule (first match wins):
      * any failed CRITICAL  → FAIL  (terminal)
      * else any failed ERROR → RETRY (recoverable)
      * else                  → PASS

    Failed WARNING findings are surfaced as warnings (they don't block). INFO
    findings, and any failed INFO, which is a contradiction, are kept as metrics.

    Pass ``strict=True`` to reject loose evidence dicts that omit an explicit
    ``status``/``passed`` instead of defaulting them to a pass (see ``Finding.coerce``).
    """
    items: List[Finding] = [Finding.coerce(f, strict=strict) for f in findings]

    critical = tuple(f for f in items if not f.passed and f.severity is Severity.CRITICAL)
    errors = tuple(f for f in items if not f.passed and f.severity is Severity.ERROR)
    warnings = tuple(f for f in items if not f.passed and f.severity is Severity.WARNING)
    metrics = tuple(f for f in items if f.severity is Severity.INFO)

    if critical:
        return Verdict(
            Decision.FAIL,
            Severity.CRITICAL,
            critical,
            warnings,
            metrics,
            f"{len(critical)} CRITICAL failure(s) - refused, no retry.",
        )
    if errors:
        return Verdict(
            Decision.RETRY,
            Severity.ERROR,
            errors,
            warnings,
            metrics,
            f"{len(errors)} ERROR failure(s) - retry once with failure context.",
        )
    return Verdict(
        Decision.PASS,
        Severity.WARNING if warnings else Severity.INFO,
        (),
        warnings,
        metrics,
        f"Passed with {len(warnings)} warning(s) and {len(metrics)} metric(s).",
    )


# --- what an EMPTY findings set means: the two named semantics ----------------------------
#
# ``compute_verdict([])`` is PASS, and that single behavior has always served two very
# different intents. A policy screen that raises no objection SHOULD pass: absence of a
# finding is its deliberate "no objection" signal. A certification handed no evidence
# SHOULD refuse: work that presented nothing has not been graded, and a vacuous pass is
# exactly the builder-grades-own-work failure this library exists to prevent. The two
# entry points below name that difference so a call site states which contract it means
# instead of inheriting whichever one ``compute_verdict`` happens to imply. The raw fold
# is unchanged; both reduce to it.


def evaluate_policy(
    findings: Iterable[Union[Finding, Mapping[str, Any]]], *, strict: bool = False
) -> Verdict:
    """Adjudicate findings as POLICY OBJECTIONS: an empty set means "no objection" and
    the verdict is PASS.

    This is the semantics every screening surface (deny list, allowlist, MCP
    declaration screen) already relies on: checks emit findings only when they object,
    so nothing-to-report is a deliberate pass, not missing evidence. Identical to
    ``compute_verdict`` today, named so the intent survives at the call site.
    """
    return compute_verdict(findings, strict=strict)


def certify_evidence(
    findings: Iterable[Union[Finding, Mapping[str, Any]]], *, strict: bool = True
) -> Verdict:
    """Adjudicate findings as CERTIFICATION EVIDENCE: an empty set REFUSES.

    A certification that proves nothing is not a pass, so no findings folds to FAIL
    through a synthesized failed-CRITICAL ``no_evidence`` finding (the verdict still
    explains itself through ``failures``/``reasons()`` like every other refusal).
    ``strict`` defaults to ``True``, unlike ``compute_verdict``: certification is the
    adapter-wiring posture where ambiguous evidence (a dict with no stated outcome)
    must be rejected rather than defaulted to a pass (see ``Finding.coerce``).

    The input is materialized once, so a generator is safe to pass.
    """
    items = list(findings)
    if not items:
        return compute_verdict(
            [
                Finding.fail(
                    "no_evidence",
                    severity=Severity.CRITICAL,
                    message="no evidence presented; a certification that proves nothing "
                    "is not a pass",
                )
            ]
        )
    return compute_verdict(items, strict=strict)
