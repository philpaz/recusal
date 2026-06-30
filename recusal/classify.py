"""
Deterministic failure classification + routing.

When an agent's action or output fails, the next move depends on *what kind* of
failure it is: a transient hiccup (retry), a policy refusal (don't retry as-is),
injected instructions in tool output (quarantine), a code bug (fix the code), the
wrong data shape (re-shape), missing data (fetch), or an ambiguous request
(ask a human). Guessing wastes a retry; asking a model is non-deterministic.

``classify_failure`` decides by explicit marker rules — same input, same class,
same route, every time. It ships a default taxonomy (extracted from a real
autonomous build system and generalized) that you can extend or replace.

Pure logic, standard library only.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from .evidence import Verdict


@dataclass(frozen=True)
class FailureClass:
    """One category of failure and where it routes.

    ``markers`` are case-insensitive substrings that signal this class; ``route`` is
    the remediation channel/action a downstream system acts on (e.g. ``"retry"``,
    ``"fix-code"``, ``"ask-human"``).
    """

    name: str
    route: str
    markers: Sequence[str]
    description: str = ""


@dataclass(frozen=True)
class Classification:
    """The class a failure was assigned and how it routes."""

    failure_class: str
    route: str
    marker: Optional[str]  # the marker that matched, or None if it fell through
    message: str

    @property
    def matched(self) -> bool:
        return self.marker is not None


# Order matters: the most specific / security-critical classes come first so a
# policy refusal or an injection is never misread as a generic code/data error.
DEFAULT_TAXONOMY: Sequence[FailureClass] = (
    FailureClass(
        "policy_violation",
        "refuse",
        (
            "refused by the gate",
            "recusal refused",
            "subject guard",
            "blocked by policy",
            "permission denied",
            "not allowed",
            "forbidden",
        ),
        "An action a deterministic policy refused — do not retry as-is; escalate or change the plan.",
    ),
    FailureClass(
        "prompt_injection",
        "quarantine",
        (
            "ignore previous instructions",
            "disregard the above",
            "ignore the system prompt",
            "exfiltrate",
            "send the api key",
            "new instructions:",
        ),
        "Tool output appears to carry injected instructions — quarantine, don't act on it.",
    ),
    FailureClass(
        "transient",
        "retry",
        (
            "timed out",
            "timeout",
            "rate limit",
            "429",
            "503",
            "connection reset",
            "econnreset",
            "temporarily unavailable",
            "service unavailable",
        ),
        "A recoverable infrastructure hiccup — retry with backoff.",
    ),
    FailureClass(
        "code_bug",
        "fix-code",
        (
            "traceback",
            "syntaxerror",
            "nameerror",
            "typeerror",
            "attributeerror",
            "assertionerror",
            "compile error",
            "exit code 1",
            "test failed",
            "tests failed",
        ),
        "The generated code is wrong — route to the builder/coding agent.",
    ),
    FailureClass(
        "data_shape",
        "fix-data",
        (
            "schema mismatch",
            "invalid type",
            "column not found",
            "unexpected field",
            "validation error",
            "null rate",
            "keyerror",
        ),
        "Data is the wrong shape — re-synthesize or re-shape it.",
    ),
    FailureClass(
        "data_missing",
        "fetch-data",
        ("0 rows", "no rows", "empty result", "not found", "does not exist", "orphan", "missing"),
        "Expected data is absent — fetch or re-migrate it.",
    ),
    FailureClass(
        "spec_ambiguity",
        "ask-human",
        (
            "ambiguous",
            "unclear",
            "underspecified",
            "which record",
            "did you mean",
            "need clarification",
            "acceptance criteria",
        ),
        "The request is ambiguous — escalate to a human for refinement.",
    ),
)


def classify_failure(
    text: str,
    *,
    taxonomy: Sequence[FailureClass] = DEFAULT_TAXONOMY,
    fallback_class: str = "unknown",
    fallback_route: str = "ask-human",
) -> Classification:
    """Classify a failure (an error string, a verdict reason, a log line) and route it.

    First matching class wins (taxonomy order is precedence). If nothing matches,
    returns the fallback class/route — never guesses.
    """
    haystack = (text or "").lower()
    for fc in taxonomy:
        for marker in fc.markers:
            if marker.lower() in haystack:
                return Classification(
                    fc.name,
                    fc.route,
                    marker,
                    f"{fc.name} -> {fc.route} (matched '{marker}')",
                )
    return Classification(
        fallback_class,
        fallback_route,
        None,
        f"unclassified -> {fallback_route}",
    )


def classify_verdict(
    verdict: Verdict,
    *,
    taxonomy: Sequence[FailureClass] = DEFAULT_TAXONOMY,
    fallback_class: str = "unknown",
    fallback_route: str = "ask-human",
) -> Classification:
    """Classify a non-PASS ``Verdict`` from its reasons — what kind of failure, and where it goes."""
    return classify_failure(
        verdict.reasons(),
        taxonomy=taxonomy,
        fallback_class=fallback_class,
        fallback_route=fallback_route,
    )
