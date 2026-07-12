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


def _as_bool(value: Any) -> bool:
    """Interpret a loose ``passed`` value. A genuine ``bool``/``int`` is used directly; a
    *string* passes only when it is an affirmative token (``"true"``/``"yes"``/``"1"``/
    ``"pass"``/...). Anything unrecognized (``"maybe"``, ``"false"``, empty) fails closed,
    mirroring how ``status`` is read."""
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_LIKE
    return bool(value)


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
            return obj
        if isinstance(obj, Mapping):
            severity = _as_severity(obj.get("severity", Severity.INFO))
            if "passed" in obj:
                passed = _as_bool(obj["passed"])
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
