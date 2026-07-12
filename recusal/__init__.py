"""
Recusal, separation-of-powers governance for Claude agents.

A judge recuses themselves from a case they can't impartially decide. The same
principle governs autonomous agents: the thing that *generates* the work must
never be the thing that *certifies* it. Recusal is that independent authority -
you collect evidence, it adjudicates deterministically into PASS / RETRY / FAIL,
and it can **refuse to certify**. No model call in the decision path. Zero
dependencies. The same normalized evidence and policy inputs, under the same
recusal version, produce the same verdict.

The spine is the evidence contract (see ``evidence`` and docs/EVIDENCE.md):

    Finding   one observation about the work.
    Verdict   the decision those findings add up to.

Everything hangs off it:

- ``evidence``, ``Finding`` / ``Verdict`` / ``Severity`` / ``Decision`` and
                 ``compute_verdict(findings)`` (the typed core).
- ``checks``, built-in deterministic checks that turn raw data into Findings.
- ``claude``, drop the gate in front of a Claude agent's tool calls so it can
                 refuse *before* a tool runs.
- ``gates``, staged release checkpoints (G0-G8) rolled into release evidence.

The constitution it encodes: **builders cannot grade their own work** ;
deterministic before AI ; the judge owns evidence, not progression ; no shadow
authority. (See CONSTITUTION.md.)

Zero runtime dependencies, standard library only.
"""

from .audit import AuditLog, verify
from .classify import (
    DEFAULT_TAXONOMY,
    Classification,
    FailureClass,
    classify_failure,
    classify_verdict,
)
from .evidence import (
    Decision,
    Finding,
    RuleSeverity,
    Severity,
    Verdict,
    compute_verdict,
)
from .gates import GateAdjudicator, GateResult, ReleaseEvidence

__all__ = [
    "Severity",
    "RuleSeverity",
    "Finding",
    "Verdict",
    "Decision",
    "compute_verdict",
    "GateAdjudicator",
    "GateResult",
    "ReleaseEvidence",
    "AuditLog",
    "verify",
    "classify_failure",
    "classify_verdict",
    "FailureClass",
    "Classification",
    "DEFAULT_TAXONOMY",
]

__version__ = "0.5.4"
