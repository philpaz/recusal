"""
Staged release gates, the verdict spine, applied to a delivery pipeline.

A gate is one named checkpoint (``G0``…``Gn``). Each gate collects **Findings** -
from ``recusal.checks`` or your own evidence, and folds them into a typed
``Verdict`` through the very same ``compute_verdict`` the rest of Recusal uses.
There is exactly one decision function in the library; a gate is just that
function applied at a checkpoint, plus an ordering. A release is ready only when
*every* gate PASSes.

The adjudicator never *generates* the work it judges, it only reads evidence
others produced. Builders cannot grade their own work.

Domain-neutral on purpose: the gate ids and what they mean are yours to define.
This module ships a neutral default staging as a starting point, with no
assumption about your stack (no database, no vendor, no migration). Pure standard
library; the verdict, and the release rollup, are pure functions of the
findings, so they replay and compare exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from .evidence import Decision, Finding, Verdict, compute_verdict

# A neutral default staging, purely labels + ordering, no domain assumptions.
# Override freely by passing your own ``gates`` to ``GateAdjudicator``.
DEFAULT_GATES: Tuple[Tuple[str, str], ...] = (
    ("G0", "environment / required systems reachable"),
    ("G1", "schema / contract loaded"),
    ("G2", "data quality within tolerance"),
    ("G3", "required artifacts present"),
    ("G4", "counts reconcile"),
    ("G5", "tests pass, coverage floor met"),
    ("G6", "metadata valid"),
    ("G7", "zero critical policy violations"),
    ("G8", "evidence complete"),
)


@dataclass(frozen=True)
class GateResult:
    """One gate's outcome: its id and the ``Verdict`` its findings folded into."""

    gate_id: str
    verdict: Verdict

    @property
    def passed(self) -> bool:
        return self.verdict.passed

    @property
    def decision(self) -> Decision:
        return self.verdict.decision

    def reasons(self) -> str:
        return self.verdict.reasons()

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-clean summary (e.g. for an audit record)."""
        return {
            "gate_id": self.gate_id,
            "verdict": self.verdict.decision.value,
            "highest_severity": self.verdict.highest_severity.value,
            "failures": [f.message or f.check for f in self.verdict.failures],
        }


@dataclass(frozen=True)
class ReleaseEvidence:
    """Staged gate results rolled up into one release-control record."""

    mission_id: str
    results: Tuple[GateResult, ...]
    required: Tuple[str, ...] = ()

    @property
    def missing(self) -> Tuple[str, ...]:
        """Required gate ids that were never adjudicated (no evidence supplied)."""
        seen = {r.gate_id for r in self.results}
        return tuple(g for g in self.required if g not in seen)

    @property
    def blocking(self) -> Tuple[GateResult, ...]:
        """The gates that are holding the release back."""
        return tuple(r for r in self.results if not r.passed)

    @property
    def release_ready(self) -> bool:
        """Ship only if there *is* gate evidence, every gate PASSed, and no required
        gate is missing. An empty result set is never ready: absence of evidence is not
        a pass, and a missing required gate blocks the release."""
        return bool(self.results) and not self.blocking and not self.missing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "release_ready": self.release_ready,
            "missing_gates": list(self.missing),
            "gate_summary": [r.to_dict() for r in self.results],
        }


class GateAdjudicator:
    """Adjudicate staged release gates over the shared verdict kernel.

    ``gates`` is an ordered ``(id, description)`` list, pure labels; supply your
    own staging or keep :data:`DEFAULT_GATES`. The adjudicator holds no domain
    logic: each gate's decision is ``compute_verdict`` over that gate's findings,
    exactly like every other surface in Recusal.
    """

    def __init__(self, gates: Sequence[Tuple[str, str]] = DEFAULT_GATES) -> None:
        self.gates: Dict[str, str] = {gid: desc for gid, desc in gates}

    def describe(self, gate_id: str) -> str:
        """Human label for a gate id (empty string if it isn't a known gate)."""
        return self.gates.get(gate_id, "")

    def adjudicate(
        self,
        gate_id: str,
        findings: Iterable[Union[Finding, Mapping[str, Any]]],
    ) -> GateResult:
        """Fold a gate's findings into a typed ``Verdict``. ``findings`` may be
        ``Finding`` objects or loose evidence dicts, ``compute_verdict`` coerces
        them. The same kernel, applied at a checkpoint. Strict: ambiguous evidence
        (a dict with no status/passed) becomes a CRITICAL gate failure, not a pass."""
        try:
            verdict = compute_verdict(findings, strict=True)
        except Exception as exc:  # noqa: BLE001
            verdict = compute_verdict(
                [
                    Finding.fail(
                        "evidence_error", severity="CRITICAL", message=f"invalid evidence: {exc}"
                    )
                ]
            )
        return GateResult(gate_id, verdict)

    def release(
        self,
        mission_id: str,
        results: Sequence[GateResult],
        *,
        required: Sequence[str] = (),
    ) -> ReleaseEvidence:
        """Roll staged gate results up into a single release decision. Pass
        ``required`` gate ids to refuse the release when any of them is absent, so an
        incomplete gate set cannot ship as 'ready'."""
        return ReleaseEvidence(mission_id, tuple(results), tuple(required))

    def adjudicate_all(
        self,
        mission_id: str,
        evidence_by_gate: Mapping[str, Iterable[Union[Finding, Mapping[str, Any]]]],
        *,
        require_all: bool = True,
    ) -> ReleaseEvidence:
        """Adjudicate the gates in a ``{gate_id: findings}`` map (configured gates
        first, in order, then any extras) and roll them up. By default every configured
        gate must be present: a missing gate refuses the release instead of passing
        vacuously. Pass ``require_all=False`` to allow a partial set."""
        ordered_ids = [g for g in self.gates if g in evidence_by_gate]
        extra_ids = [g for g in evidence_by_gate if g not in self.gates]
        results: List[GateResult] = [
            self.adjudicate(gid, evidence_by_gate[gid]) for gid in ordered_ids + extra_ids
        ]
        required = tuple(self.gates) if require_all else ()
        return self.release(mission_id, results, required=required)
