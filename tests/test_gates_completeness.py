"""The release gate must not ship on empty or incomplete gate evidence.

Regression tests for the "vacuous pass" footgun: `all([])` is True, so an empty or
partial gate set could otherwise report `release_ready=True`.
"""

from recusal.checks import row_count
from recusal.gates import GateAdjudicator

_OK = [row_count([{"id": 1}], min_rows=1)]  # a passing gate's findings


def test_empty_release_is_not_ready():
    # all([]) is True, but a release with no gate evidence must not be "ready".
    assert GateAdjudicator().release("m", []).release_ready is False


def test_missing_required_gate_refuses_release():
    adj = GateAdjudicator()
    rel = adj.release("m", [adj.adjudicate("G1", _OK)], required=("G1", "G5"))
    assert rel.release_ready is False
    assert rel.missing == ("G5",)


def test_release_ready_when_all_required_present_and_pass():
    adj = GateAdjudicator()
    rel = adj.release("m", [adj.adjudicate("G1", _OK)], required=("G1",))
    assert rel.release_ready is True
    assert rel.missing == ()


def test_adjudicate_all_requires_every_configured_gate_by_default():
    adj = GateAdjudicator(gates=(("A", "a"), ("B", "b")))

    partial = adj.adjudicate_all("m", {"A": _OK})  # B not supplied
    assert partial.release_ready is False
    assert partial.missing == ("B",)

    full = adj.adjudicate_all("m", {"A": _OK, "B": _OK})
    assert full.release_ready is True


def test_adjudicate_all_allows_partial_when_not_required():
    adj = GateAdjudicator(gates=(("A", "a"), ("B", "b")))
    rel = adj.adjudicate_all("m", {"A": _OK}, require_all=False)
    assert rel.release_ready is True  # only A, passing, nothing required
    assert rel.missing == ()


def test_ambiguous_evidence_becomes_a_critical_gate_failure():
    # a status-less dict must not let a gate pass vacuously; it fails closed.
    result = GateAdjudicator().adjudicate("G2", [{"severity": "CRITICAL", "message": "no status"}])
    assert result.decision.value == "FAIL"
