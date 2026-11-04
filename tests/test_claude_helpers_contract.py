"""The Claude evidence helpers refuse a malformed outer container (2026-11-04 adapters review).

``gate_tool_use`` and ``tool_confirmation`` fold whatever iterable they are given; ``{}``,
``""`` and ``b""`` iterate to nothing, so an invalid empty container passed as a clean
screen. The outer shape is validated now; a real empty list keeps its pass meaning, and
``None`` and ``False`` keep refusing, with a reason that names the shape.
"""

import pytest

from recusal import Finding
from recusal.claude import _evidence_of, gate_tool_use, tool_confirmation

MALFORMED = [{}, {"check": "a", "passed": False}, "", "x", b"", b"x", None, False, True, 0, 1.5]


@pytest.mark.parametrize("evidence", MALFORMED, ids=repr)
def test_gate_tool_use_refuses_a_malformed_container_by_name(evidence):
    allow, refusal = gate_tool_use("t1", evidence, tool_name="run_sql")
    assert allow is False
    assert refusal["is_error"] is True
    assert "invalid evidence" in refusal["content"]
    assert type(evidence).__name__ in refusal["content"]


@pytest.mark.parametrize("evidence", MALFORMED, ids=repr)
def test_tool_confirmation_denies_a_malformed_container_by_name(evidence):
    event = tool_confirmation("t1", evidence)
    assert event["result"] == "deny"
    assert "invalid evidence" in event["deny_message"]
    assert type(evidence).__name__ in event["deny_message"]


@pytest.mark.parametrize("evidence", [[], (), iter(())], ids=["list", "tuple", "generator"])
def test_a_real_empty_collection_is_still_a_clean_screen(evidence):
    assert gate_tool_use("t1", evidence, tool_name="run_sql") == (True, None)


def test_a_real_empty_collection_still_allows_a_confirmation():
    assert tool_confirmation("t1", [])["result"] == "allow"


def test_findings_in_any_iterable_still_fold():
    critical = Finding.fail("unsafe", severity="CRITICAL", message="refuse this")
    assert gate_tool_use("t1", (critical,), tool_name="run_sql")[0] is False
    assert gate_tool_use("t1", (f for f in [critical]), tool_name="run_sql")[0] is False
    assert tool_confirmation("t1", [critical])["result"] == "deny"
    assert gate_tool_use("t1", [Finding.ok("fine", severity="WARNING")], tool_name="x")[0] is True


def test_the_helper_names_the_shape_without_its_value():
    with pytest.raises(TypeError, match="evidence was dict; pass a list of findings"):
        _evidence_of({"secret": "MODEL_SECRET_SENTINEL"})
    with pytest.raises(TypeError, match="evidence was str; pass a list of findings"):
        _evidence_of("MODEL_SECRET_SENTINEL")
    with pytest.raises(TypeError, match="not iterable"):
        _evidence_of(object())
    assert _evidence_of(x for x in [1]) == [1]
