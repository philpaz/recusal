"""recusal.authorization: an action is authorized only inside supplied authority evidence.

What these tests pin, in the order the module makes its claims:

- every dimension refuses when its evidence is absent (absence never passes);
- the principal dimension passes ONLY through a supplied trust rule;
- malformed evidence is rejected at every public entry point, never normalized:
  non-string keys, sets, arbitrary objects, NaN and infinities raise, canonical_json and
  fingerprint included, and a hand-built FrozenMapping is validated; numeric bounds
  compare JSON numbers only;
- the composition invariant: built-in dimensions are always required, an unrelated
  passing finding cannot certify a missing one, a custom finding cannot impersonate a
  built-in, and the synthesized dimension_missing findings are retained;
- expiry, budget and replay verify SUPPLIED clock, counter and use-state evidence, and
  the documented race (two callers observing the same count both pass) is real;
- the receipt is derived from the decision alone (no caller-supplied policy, manifest or
  provenance), strictly constructed and parsed, reconciled against its own findings, and
  detects a changed byte; its own docstring limits hold;
- the kernel is untouched: recusal/evidence.py is byte-identical to its 0.8.0 content;
- the module depends only on the standard library and recusal.evidence.
"""

import ast
import hashlib
import json
import os

import pytest

from recusal import Finding, certify_evidence
from recusal.authorization import (
    DIMENSIONS,
    EVIDENCE_FIELDS,
    PROVENANCE_CLAUDE_PRETOOLUSE,
    RECEIPT_FIELDS,
    ActionRequest,
    AuthorizationContext,
    Constraints,
    DecisionReceipt,
    FrozenMapping,
    Supplied,
    canonical_json,
    certify_authorization,
    certify_dimensions,
    check_budget,
    check_expiry,
    check_principal,
    check_replay,
    fingerprint,
    is_json_number,
    run_checks,
    validate_receipt_body,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOW = "2026-08-26T19:00:00+00:00"
LATER = "2026-08-26T21:00:00+00:00"
HEX = "f" * 64


def _request(**overrides):
    base = dict(
        principal="agent:invoice-assistant",
        tool="crm.update_record",
        operation="update",
        resource="customer:456",
        arguments={"status": "resolved", "notes": "paid"},
        nonce="n-002",
    )
    base.update(overrides)
    return ActionRequest(**base)


def _context(**overrides):
    base = dict(
        trusted_principals=Supplied({"agent:invoice-assistant": "svc:invoice"}, "operator_config"),
        active_subject=Supplied("customer:456", "session_store"),
        allowed_operations=Supplied({"crm.update_record": ["update"]}, "grant"),
        constraints=Supplied(
            Constraints(
                fields=("status", "notes"), max_calls=3, expires_at="2026-08-26T20:00:00+00:00"
            ),
            "grant",
        ),
        now=Supplied(NOW, "adopter_clock"),
        calls_so_far=Supplied(1, "adopter_counter"),
        used_nonces=Supplied(["n-001"], "nonce_store"),
        approval=Supplied({"policy_version": "v3"}, "grant"),
        policy_version=Supplied("v3", "operator_config"),
    )
    base.update(overrides)
    return AuthorizationContext(**base)


def _by_dimension(request, context):
    return {f.check.split(".", 1)[1]: f for f in run_checks(request, context)}


def _bound_ctx(max_value=100.0, field="amount"):
    return _context(constraints=Supplied(Constraints(max_value=max_value, value_field=field), "g"))


def _receipt(**ctx_overrides):
    return DecisionReceipt.build(certify_authorization(_request(), _context(**ctx_overrides)))


def _document(receipt, **body_changes):
    doc = json.loads(receipt.to_json())
    doc["body"].update(body_changes)
    return doc


def _resigned(doc):
    """A document whose digest is recomputed over its (possibly altered) body."""
    return json.dumps({"body": doc["body"], "digest": fingerprint(doc["body"])})


# --- the happy path, then absence -------------------------------------------------------


def test_fully_evidenced_request_is_authorized_with_one_passing_finding_per_dimension():
    decision = certify_authorization(_request(), _context())
    assert decision.authorized
    assert decision.verdict.decision.value == "PASS"
    assert [f.check for f in decision.findings] == ["authorization." + d for d in DIMENSIONS]
    assert all(f.passed for f in decision.findings)
    assert all("provenance" in f.context for f in decision.findings)
    assert decision.principal == "svc:invoice"
    assert decision.required == DIMENSIONS
    assert decision.policy_version == "v3" and decision.manifest_fingerprint is None
    assert decision.evidence_provenance.plain()["trusted_principals"] == "operator_config"


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_every_dimension_refuses_when_its_evidence_is_absent(dimension):
    """Absent authority evidence is a refusal, never a vacuous pass."""
    ctx = AuthorizationContext()
    findings = _by_dimension(_request(), ctx)
    assert not findings[dimension].passed, dimension
    assert findings[dimension].severity.value == "CRITICAL"
    decision = certify_authorization(_request(), ctx)
    assert not decision.authorized and decision.principal is None
    assert decision.policy_version is None and decision.evidence_provenance.plain() == {}


# --- principal: a label is a claim, the trust rule is the authority ----------------------


def test_principal_passes_only_through_a_supplied_trust_rule():
    ok = check_principal(_request(), _context())
    assert ok.passed and ok.context["principal"] == "svc:invoice"

    no_rule = check_principal(_request(), _context(trusted_principals=None))
    assert not no_rule.passed
    assert "runtime label is not an authenticated identity" in no_rule.message

    unbound = check_principal(_request(), _context(trusted_principals=Supplied({}, "cfg")))
    assert not unbound.passed
    assert "not bound to any configured principal" in unbound.message

    not_a_rule = check_principal(_request(), _context(trusted_principals=Supplied(["x"], "cfg")))
    assert not not_a_rule.passed and "not a mapping" in not_a_rule.message


def test_a_runtime_label_alone_never_authorizes_even_with_every_other_dimension_satisfied():
    """The security adjustment in one test: Claude's agent_id is not an identity."""
    decision = certify_authorization(
        _request(principal="a343c667c7dc99439"), _context(trusted_principals=None)
    )
    assert not decision.authorized
    assert [f.check for f in decision.verdict.failures] == ["authorization.principal"]


def test_runtime_provenance_label_is_the_one_the_hook_writes():
    assert PROVENANCE_CLAUDE_PRETOOLUSE == "claude_pretooluse_event"


# --- the strict JSON domain: rejected at every public entry point -------------------------


_OUT_OF_DOMAIN = [
    {1: "a"},  # non-string key
    {1: "a", "1": "b"},  # the collision that str(k) would have silently merged
    {"s": {1, 2}},  # a set is not a JSON value
    {"o": object()},  # an arbitrary object
    {"n": float("nan")},
    {"i": float("inf")},
    {"ni": float("-inf")},
    {"nested": [{"deep": {2: "x"}}]},
    {"b": b"bytes"},
]


@pytest.mark.parametrize("bad", _OUT_OF_DOMAIN)
def test_out_of_domain_evidence_raises_at_every_public_entry_point(bad):
    with pytest.raises(ValueError):
        Supplied(bad, "p")
    with pytest.raises(ValueError):
        ActionRequest(principal="a", tool="t", operation="o", arguments=bad)
    with pytest.raises(ValueError):
        canonical_json(bad)
    with pytest.raises(ValueError):
        fingerprint(bad)


def test_public_canonicalization_keeps_int_and_str_keys_apart_by_refusing_int_keys():
    """json.dumps would stringify {1: "a"} to the same bytes as {"1": "a"}; the public
    helpers must refuse instead of normalizing."""
    with pytest.raises(ValueError, match="object key 1 is not a string"):
        fingerprint({1: "a"})
    assert fingerprint({"1": "a"})  # the string-keyed twin is fine


def test_a_hand_built_frozen_mapping_is_validated():
    for items in (
        (("a", float("nan")),),
        ((1, "x"),),
        (("b", 1), ("a", 2)),  # not ascending
        (("a", 1), ("a", 2)),  # duplicate
        (("a", {1, 2}),),
        (("a", ("x", {1: 2})),),
        [("a", 1)],  # not a tuple
    ):
        with pytest.raises(ValueError):
            FrozenMapping(items)  # type: ignore[arg-type]
    good = FrozenMapping((("a", 1), ("b", ("x", FrozenMapping(())))))
    assert good.plain() == {"a": 1, "b": ["x", {}]}
    assert Supplied(good, "p").value is good


def test_freeze_thaw_round_trips_the_json_domain_and_keeps_empty_object_and_array_apart():
    value = {"a": [1, 2.5, "s", None, True, {"z": [], "y": {}}], "b": {}}
    supplied = Supplied(value, "p")
    assert supplied.plain() == value
    assert isinstance(supplied.value, FrozenMapping)
    assert Supplied({}, "p").plain() == {} and Supplied([], "p").plain() == []
    assert canonical_json({}) != canonical_json([])


def test_is_json_number_excludes_bool_nan_and_infinity():
    assert is_json_number(1) and is_json_number(1.5) and is_json_number(-3)
    for bad in (True, False, float("nan"), float("inf"), "1", None, [1]):
        assert not is_json_number(bad), bad


# --- numeric bounds: JSON numbers only, no coercion ---------------------------------------


@pytest.mark.parametrize("value", ["nan", "inf", "100", "99", True, False, None, [50], {"v": 1}])
def test_value_bound_refuses_anything_that_is_not_a_finite_json_number(value):
    f = _by_dimension(_request(arguments={"amount": value}), _bound_ctx())["arguments"]
    assert not f.passed, value
    assert "finite JSON number" in f.message
    assert not certify_authorization(_request(arguments={"amount": value}), _bound_ctx()).authorized


def test_value_bound_is_verified_on_the_named_argument():
    inside = _by_dimension(_request(arguments={"amount": 99.5}), _bound_ctx())["arguments"]
    exact = _by_dimension(_request(arguments={"amount": 100}), _bound_ctx())["arguments"]
    over = _by_dimension(_request(arguments={"amount": 100.5}), _bound_ctx())["arguments"]
    missing = _by_dimension(_request(arguments={"other": 1}), _bound_ctx())["arguments"]
    assert inside.passed and exact.passed
    assert not over.passed and over.context["value"] == 100.5
    assert not missing.passed


def test_constraints_reject_malformed_bounds_and_fields():
    for kwargs in (
        {"max_calls": -1},
        {"max_calls": True},
        {"max_calls": 2.0},
        {"max_calls": "3"},
        {"max_value": 1.0},  # value_field missing
        {"max_value": float("nan"), "value_field": "v"},
        {"max_value": float("inf"), "value_field": "v"},
        {"max_value": True, "value_field": "v"},
        {"max_value": "100", "value_field": "v"},
        {"max_value": 1.0, "value_field": ""},
        {"fields": "status"},
        {"fields": ("status", "status")},
        {"fields": ("status", "")},
        {"fields": ("status", 3)},
        {"expires_at": ""},
    ):
        with pytest.raises(ValueError):
            Constraints(**kwargs)
    assert Constraints(fields=["a", "b"]).fields == ("a", "b")


# --- the composition invariant -----------------------------------------------------------


def test_an_unrelated_passing_finding_cannot_certify_a_missing_dimension():
    """certify_evidence alone would PASS here; certify_dimensions must not."""
    unrelated = [Finding.ok("something_else", severity="CRITICAL", message="fine")]
    assert certify_evidence(unrelated).passed  # the hazard, demonstrated
    verdict = certify_dimensions(unrelated)
    assert not verdict.passed
    missing = [f for f in verdict.failures if f.check == "authorization.dimension_missing"]
    assert [f.context["dimension"] for f in missing] == list(DIMENSIONS)


def test_built_in_dimensions_are_always_required_and_cannot_be_named_as_additional():
    for name in DIMENSIONS + ("dimension_missing",):
        with pytest.raises(ValueError):
            certify_dimensions([], additional_required=(name,))
    for bad in ("", "has space", 3, None):
        with pytest.raises(ValueError):
            certify_dimensions([], additional_required=(bad,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        certify_dimensions([], additional_required=("residency", "residency"))
    with pytest.raises(ValueError):
        certify_dimensions([], additional_required="residency")  # a string is not a list


def test_certify_dimensions_rejects_malformed_findings_deliberately():
    for bad in ([{"check": "x"}], ["x"], "x", {"check": "x"}, [None]):
        with pytest.raises(ValueError):
            certify_dimensions(bad)  # type: ignore[arg-type]


def test_adopter_findings_can_satisfy_an_additional_required_dimension():
    custom = Finding.ok("authorization.residency", severity="CRITICAL", message="EU")
    decision = certify_authorization(
        _request(), _context(), additional_required=("residency",), findings=[custom]
    )
    assert decision.authorized and decision.required == DIMENSIONS + ("residency",)
    without = certify_authorization(_request(), _context(), additional_required=("residency",))
    assert not without.authorized
    assert without.verdict.failures[0].check == "authorization.dimension_missing"
    failing = Finding.fail("authorization.residency", severity="CRITICAL", message="US")
    assert not certify_authorization(_request(), _context(), findings=[failing]).authorized


def test_synthesized_missing_findings_are_retained_on_the_decision():
    """A receipt must carry what the verdict was folded from, including what was absent."""
    decision = certify_authorization(_request(), _context(), additional_required=("residency",))
    names = [f.check for f in decision.findings]
    assert names[-1] == "authorization.dimension_missing"
    assert decision.findings[-1].context["dimension"] == "residency"
    assert certify_evidence(decision.findings).decision == decision.verdict.decision


def test_a_custom_finding_cannot_impersonate_a_built_in_dimension():
    forged = Finding.ok("authorization.principal", severity="CRITICAL", message="trust me")
    with pytest.raises(ValueError):
        certify_authorization(_request(), _context(trusted_principals=None), findings=[forged])
    synthetic = Finding.ok("authorization.dimension_missing", severity="CRITICAL")
    with pytest.raises(ValueError):
        certify_authorization(_request(), _context(), findings=[synthetic])
    with pytest.raises(ValueError):
        certify_authorization(_request(), _context(), findings=[{"check": "x"}])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        certify_authorization({"tool": "t"}, _context())  # type: ignore[arg-type]


def test_a_failed_built_in_check_refuses_regardless_of_additional_required():
    ctx = _context(used_nonces=None)
    decision = certify_authorization(_request(), ctx, additional_required=("residency",))
    assert not decision.authorized
    assert {f.check for f in decision.verdict.failures} == {
        "authorization.replay",
        "authorization.dimension_missing",
    }


# --- subject, tool, arguments ------------------------------------------------------------


def test_wrong_subject_is_refused_and_named():
    f = _by_dimension(_request(resource="customer:999"), _context())["subject"]
    assert not f.passed and f.context == {
        "target": "customer:999",
        "active": "customer:456",
        "provenance": "session_store",
        "evidence": {"active_subject": "session_store"},
    }
    assert not _by_dimension(_request(), _context(active_subject=Supplied(456, "s")))[
        "subject"
    ].passed


def test_tool_and_operation_must_both_be_granted_and_the_grant_must_be_well_formed():
    by = _by_dimension
    assert not by(_request(tool="crm.delete_record"), _context())["tool_operation"].passed
    assert not by(_request(operation="delete"), _context())["tool_operation"].passed
    assert by(_request(), _context())["tool_operation"].passed
    for malformed in ({"crm.update_record": "update"}, {"crm.update_record": [1]}, ["update"]):
        f = by(_request(), _context(allowed_operations=Supplied(malformed, "g")))["tool_operation"]
        assert not f.passed, malformed


def test_arguments_outside_the_granted_fields_are_refused():
    f = _by_dimension(_request(arguments={"status": "x", "balance": 0}), _context())["arguments"]
    assert not f.passed and f.context["extra"] == ["balance"]


# --- expiry, budget, replay: supplied evidence, stated limits -----------------------------


def test_expiry_compares_supplied_now_against_the_grant():
    assert check_expiry(_request(), _context()).passed
    expired = check_expiry(_request(), _context(now=Supplied(LATER, "adopter_clock")))
    assert not expired.passed and "expired at" in expired.message
    assert expired.context["clock_provenance"] == "adopter_clock"


def test_expiry_refuses_naive_or_unparseable_instants():
    naive = check_expiry(_request(), _context(now=Supplied("2026-08-26T19:00:00", "clock")))
    assert not naive.passed
    ctx = _context(constraints=Supplied(Constraints(expires_at="soon"), "grant"))
    assert not check_expiry(_request(), ctx).passed


def test_expiry_with_no_expiry_declared_passes_and_says_so():
    ctx = _context(constraints=Supplied(Constraints(), "grant"), now=None)
    f = check_expiry(_request(), ctx)
    assert f.passed and "no expiry was declared" in f.message


def test_budget_is_verified_against_the_supplied_count():
    assert check_budget(_request(), _context(calls_so_far=Supplied(2, "c"))).passed
    exhausted = check_budget(_request(), _context(calls_so_far=Supplied(3, "c")))
    assert not exhausted.passed and "exhausted" in exhausted.message
    assert not check_budget(_request(), _context(calls_so_far=Supplied(True, "c"))).passed
    assert not check_budget(_request(), _context(calls_so_far=Supplied(2.0, "c"))).passed


def test_the_documented_budget_race_is_real_and_not_hidden():
    """Two callers observe calls_so_far=2 under max_calls=3: both pass. The module
    verifies supplied counters; it does not reserve. This test pins the limit so the
    docs cannot drift into claiming atomic quota enforcement."""
    ctx = _context(calls_so_far=Supplied(2, "c"))
    first = check_budget(_request(nonce="a"), ctx)
    second = check_budget(_request(nonce="b"), ctx)
    assert first.passed and second.passed
    assert "not reserved" in first.message


def test_replay_needs_a_nonce_and_supplied_use_state():
    assert check_replay(_request(), _context()).passed
    assert not check_replay(_request(nonce=None), _context()).passed
    assert not check_replay(_request(), _context(used_nonces=None)).passed
    replayed = check_replay(_request(nonce="n-001"), _context())
    assert not replayed.passed and "already used" in replayed.message
    assert not check_replay(_request(), _context(used_nonces=Supplied([1], "n"))).passed
    assert not check_replay(_request(), _context(used_nonces=Supplied({"n-001": 1}, "n"))).passed


# --- policy and manifest binding ---------------------------------------------------------


def test_policy_change_since_approval_is_refused():
    f = _by_dimension(_request(), _context(policy_version=Supplied("v4", "cfg")))["policy_binding"]
    assert not f.passed and f.context == {
        "approved": "v3",
        "current": "v4",
        "provenance": "grant",
        "evidence": {"approval": "grant", "policy_version": "cfg"},
        "current_policy_version": "v4",
    }
    assert not _by_dimension(_request(), _context(policy_version=Supplied(3, "cfg")))[
        "policy_binding"
    ].passed


def test_manifest_binding_is_two_sided_and_hex_shaped():
    by = _by_dimension
    assert by(_request(), _context())["manifest_binding"].passed  # neither side: stated
    one_sided = by(_request(), _context(manifest_fingerprint=Supplied(HEX, "mcp")))
    assert not one_sided["manifest_binding"].passed
    assert "one-sided" in one_sided["manifest_binding"].message
    approved = Supplied({"policy_version": "v3", "manifest_fingerprint": HEX}, "grant")
    bound = _context(approval=approved, manifest_fingerprint=Supplied(HEX, "m"))
    assert by(_request(), bound)["manifest_binding"].passed
    assert certify_authorization(_request(), bound).manifest_fingerprint == HEX
    drift = by(
        _request(), _context(approval=approved, manifest_fingerprint=Supplied("e" * 64, "m"))
    )
    assert not drift["manifest_binding"].passed
    short = Supplied({"policy_version": "v3", "manifest_fingerprint": "abc"}, "grant")
    bad = by(_request(), _context(approval=short, manifest_fingerprint=Supplied("abc", "m")))
    assert (
        not bad["manifest_binding"].passed and "lowercase sha256" in bad["manifest_binding"].message
    )


# --- the request and context types --------------------------------------------------------


def test_request_is_immutable_and_fingerprints_are_canonical():
    a = _request(arguments={"b": 1, "a": [1, 2]})
    b = _request(arguments={"a": [1, 2], "b": 1})
    assert a.arguments_fingerprint == b.arguments_fingerprint
    assert a.action_fingerprint == b.action_fingerprint
    assert a.action_fingerprint != _request(nonce="other").action_fingerprint
    with pytest.raises((AttributeError, TypeError)):
        a.tool = "x"  # type: ignore[misc]
    assert json.loads(json.dumps(a.to_dict()))["arguments"] == {"a": [1, 2], "b": 1}


def test_request_validates_every_string_part():
    for name in ("principal", "tool", "operation"):
        with pytest.raises(ValueError):
            _request(**{name: ""})
    for name in ("resource", "nonce"):
        with pytest.raises(ValueError):
            _request(**{name: ""})
        with pytest.raises(ValueError):
            _request(**{name: 5})
    with pytest.raises(ValueError):
        _request(arguments=[("a", 1)])  # type: ignore[arg-type]


def test_context_only_accepts_supplied_items_and_reports_provenance():
    with pytest.raises(ValueError):
        AuthorizationContext(active_subject="customer:456")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Supplied("x", "")
    assert _context(now=None).provenance()["active_subject"] == "session_store"
    assert "now" not in _context(now=None).provenance()


def test_a_check_that_raises_becomes_a_refusal_not_an_outage():
    class Bad(Supplied):
        def plain(self):
            raise RuntimeError("boom")

    f = _by_dimension(_request(), _context(active_subject=Bad("x", "p")))["subject"]
    assert not f.passed and "boom" in f.message


# --- the receipt: derived from the decision, reconciled with itself ------------------------


def test_receipt_is_deterministic_and_derived_entirely_from_the_decision():
    r1 = _receipt()
    r2 = _receipt()
    assert r1.to_json() == r2.to_json() and r1.digest == r2.digest
    body = r1.body
    assert body["authorized"] is True and body["decision"] == "PASS"
    assert body["principal_label"] == "agent:invoice-assistant"
    assert body["principal"] == "svc:invoice"  # the resolved principal, bound in
    assert body["policy_version"] == "v3"  # from the adjudicated context, not a parameter
    assert body["manifest_fingerprint"] is None
    assert body["evidence_provenance"]["trusted_principals"] == "operator_config"
    assert len(body["findings"]) == len(DIMENSIONS)
    assert body["timestamp"] is None  # nothing is stamped unless supplied
    assert set(body) == set(RECEIPT_FIELDS)
    refused = DecisionReceipt.build(certify_authorization(_request(), AuthorizationContext()))
    assert refused.body["principal"] is None and refused.body["authorized"] is False
    assert refused.body["policy_version"] is None and refused.body["evidence_provenance"] == {}


def test_receipt_cannot_be_labeled_with_metadata_the_decision_was_not_made_under():
    """The previous API took context=, policy_version= and manifest_fingerprint=; they are
    gone, so a receipt for a decision under v3 cannot claim v999 or unrelated provenance."""
    decision = certify_authorization(_request(), _context())
    with pytest.raises(TypeError):
        DecisionReceipt.build(decision, policy_version="v999")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DecisionReceipt.build(decision, context=_context())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DecisionReceipt.build(decision, manifest_fingerprint=HEX)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        DecisionReceipt.build("not a decision")  # type: ignore[arg-type]
    other = _receipt(policy_version=Supplied("v4", "cfg"))  # policy drift: refused, and recorded
    assert other.body["policy_version"] == "v4" and other.body["authorized"] is False


def test_receipt_carries_the_synthesized_missing_findings_it_was_folded_from():
    decision = certify_authorization(_request(), _context(), additional_required=("residency",))
    receipt = DecisionReceipt.build(decision)
    checks = [f["check"] for f in receipt.body["findings"]]
    assert checks[-1] == "authorization.dimension_missing"
    assert receipt.body["required"] == list(DIMENSIONS) + ["residency"]
    assert receipt.body["decision"] == "FAIL" and receipt.verify() == (True, [])


def test_receipt_body_is_a_copy_and_the_receipt_is_frozen():
    receipt = _receipt()
    body = receipt.body
    body["authorized"] = False
    assert receipt.body["authorized"] is True
    assert receipt.digest_matches()
    with pytest.raises((AttributeError, TypeError)):
        receipt.digest = "0" * 64  # type: ignore[misc]


def test_receipt_digest_is_sha256_over_the_canonical_body_and_detects_a_changed_byte():
    receipt = _receipt()
    canonical = json.dumps(receipt.body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert receipt.digest == hashlib.sha256(canonical.encode()).hexdigest()
    assert receipt.verify() == (True, [])
    doc = _document(receipt, timestamp="2026-08-26T19:00:00+00:00")  # well-formed, digest stale
    parsed = DecisionReceipt.from_json(json.dumps(doc))
    assert not parsed.digest_matches()
    intact, problems = parsed.verify()
    assert not intact and problems == ["digest does not match the canonical body"]


def test_receipt_verification_reconciles_findings_with_the_verdict():
    """A PASS receipt listing a failed CRITICAL finding is inconsistent even with a
    freshly recomputed digest; the findings must re-fold to the recorded decision."""
    receipt = _receipt()
    doc = _document(receipt)
    doc["body"]["findings"][0]["passed"] = False  # principal now failed, decision still PASS
    with pytest.raises(ValueError, match="does not match the findings"):
        DecisionReceipt.from_json(_resigned(doc))
    problems = validate_receipt_body(doc["body"])
    assert "decision 'PASS' does not match the findings, which fold to 'FAIL'" in problems
    assert "authorized does not match the findings" in problems
    # the converse: a FAIL receipt whose findings all pass
    refused = DecisionReceipt.build(
        certify_authorization(_request(), _context(now=Supplied(LATER, "c")))
    )
    doc = _document(refused)
    for f in doc["body"]["findings"]:
        f["passed"] = True
    with pytest.raises(ValueError, match="does not match the findings"):
        DecisionReceipt.from_json(_resigned(doc))


def test_receipt_verification_requires_every_built_in_exactly_once_and_extras_evidenced():
    receipt = _receipt()
    doc = _document(receipt)
    doc["body"]["findings"] = doc["body"]["findings"][1:]  # drop principal
    assert any("exactly once" in p for p in validate_receipt_body(doc["body"]))
    doc = _document(receipt)
    doc["body"]["findings"].append(doc["body"]["findings"][0])  # duplicate principal
    assert any("exactly once" in p for p in validate_receipt_body(doc["body"]))
    doc = _document(receipt, required=list(DIMENSIONS) + ["residency"])  # required, no finding
    assert any(
        "either evidenced or declared missing" in p for p in validate_receipt_body(doc["body"])
    )
    doc = _document(receipt, required=list(DIMENSIONS) + ["principal"])  # built-in as extra
    assert any(p.startswith("required:") for p in validate_receipt_body(doc["body"]))
    doc = _document(receipt, required=list(DIMENSIONS) + ["x", "x"])
    assert any("repeats" in p for p in validate_receipt_body(doc["body"]))
    doc = _document(receipt)
    doc["body"]["findings"].append(
        {
            "check": "authorization.dimension_missing",
            "passed": False,
            "severity": "CRITICAL",
            "message": "m",
            "context": {"dimension": "ghost"},
        }
    )
    doc["body"]["decision"], doc["body"]["authorized"] = "FAIL", False
    assert any("not required" in p for p in validate_receipt_body(doc["body"]))


def test_receipt_finding_contexts_are_validated_recursively():
    receipt = _receipt()
    doc = _document(receipt)
    doc["body"]["findings"][0]["context"] = {"ok": {"deep": {"k": [1, {"x": None}]}}}
    assert not any("context" in p for p in validate_receipt_body(doc["body"]))
    raw = json.dumps(doc).replace('"ok": {"deep"', '"ok": {"deep": 1e999, "d"')
    with pytest.raises(ValueError):
        DecisionReceipt.from_json(raw)  # json.loads yields inf; the domain refuses it


def test_receipt_digest_alone_does_not_authenticate_anything():
    """Anyone can build a self-consistent receipt; the digest proves bytes, not origin."""
    receipt = _receipt()
    forged = DecisionReceipt.from_json(receipt.to_json())  # a copy is indistinguishable
    assert forged.verify() == (True, [])


def test_receipt_binds_to_an_audit_entry_when_one_is_supplied():
    decision = certify_authorization(_request(), _context())
    receipt = DecisionReceipt.build(decision, audit_entry={"seq": 7, "hash": HEX})
    assert receipt.body["audit_seq"] == 7 and receipt.body["audit_head"] == HEX
    for bad in (
        {"seq": "7", "hash": HEX},
        {"seq": True, "hash": HEX},
        {"seq": -1, "hash": HEX},
        {"seq": 7, "hash": "f"},
        {"seq": 7, "hash": "F" * 64},
        {"seq": 7},
        [7, HEX],
    ):
        with pytest.raises(ValueError):
            DecisionReceipt.build(decision, audit_entry=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DecisionReceipt.build(decision, timestamp="")


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        "{}",
        '{"body": 5, "digest": 7}',
        '{"body": {}, "digest": "' + HEX + '"}',
        '{"body": {}, "digest": "' + HEX + '", "extra": 1}',
        "not json",
    ],
)
def test_receipt_parsing_is_strict(document):
    with pytest.raises(ValueError):
        DecisionReceipt.from_json(document)


def test_receipt_parsing_rejects_each_malformed_field_by_name():
    receipt = _receipt()
    cases = {
        "receipt_version": 2,
        "audit_seq": True,
        "audit_head": "F" * 64,
        "action_fingerprint": "abc",
        "decision": "MAYBE",
        "authorized": "true",
        "required": ["principal"],
        "findings": [{"check": "x"}],
        "evidence_provenance": {"a": 1},
        "manifest_fingerprint": "abc",
        "principal": "",
        "tool": "",
    }
    for field, value in cases.items():
        with pytest.raises(ValueError, match="malformed receipt body"):
            DecisionReceipt.from_json(json.dumps(_document(receipt, **{field: value})))
    with pytest.raises(ValueError):
        DecisionReceipt.from_json(json.dumps(_document(receipt, audit_seq=3)))  # one-sided
    doc = _document(receipt)
    del doc["body"]["timestamp"]
    with pytest.raises(ValueError, match="missing field"):
        DecisionReceipt.from_json(json.dumps(doc))
    doc = _document(receipt)
    doc["body"]["surprise"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        DecisionReceipt.from_json(json.dumps(doc))
    doc = _document(receipt)
    doc["digest"] = "abc"
    with pytest.raises(ValueError, match="digest"):
        DecisionReceipt.from_json(json.dumps(doc))


def test_validate_receipt_body_names_every_problem_and_never_raises():
    assert validate_receipt_body(None) == ["body is not an object"]
    problems = validate_receipt_body({})
    assert len(problems) == len(RECEIPT_FIELDS) and all(p.startswith("missing") for p in problems)
    good = _receipt().body
    assert validate_receipt_body(good) == []
    good["authorized"] = False  # contradicts the findings, which fold to PASS
    assert validate_receipt_body(good) == ["authorized does not match the findings"]


def test_receipt_round_trips_through_json():
    receipt = _receipt()
    again = DecisionReceipt.from_json(receipt.to_json())
    assert again.digest == receipt.digest and again.verify() == (True, [])
    assert again.body == receipt.body


def test_receipt_metadata_is_reconciled_with_the_findings_it_was_derived_from():
    """Relabeling principal, policy_version, manifest_fingerprint or evidence provenance
    and recomputing the digest is detectable from the findings, and refused."""
    receipt = _receipt()
    for field, value, expected in (
        ("principal", "OTHER", "principal does not match the principal finding"),
        ("principal", None, "principal does not match the principal finding"),
        ("policy_version", "other", "policy_version does not match the policy_binding finding"),
        ("policy_version", None, "policy_version does not match the policy_binding finding"),
        (
            "manifest_fingerprint",
            HEX,
            "manifest_fingerprint does not match the manifest_binding finding",
        ),
        ("evidence_provenance", {"x": "fake"}, "evidence_provenance does not match"),
        ("evidence_provenance", {}, "evidence_provenance does not match"),
    ):
        doc = _document(receipt, **{field: value})
        problems = validate_receipt_body(doc["body"])
        assert any(expected in p for p in problems), (field, value, problems)
        with pytest.raises(ValueError, match="malformed receipt body"):
            DecisionReceipt.from_json(_resigned(doc))
    # one provenance label swapped inside an otherwise intact map is caught too
    doc = _document(receipt)
    doc["body"]["evidence_provenance"]["trusted_principals"] = "elsewhere"
    assert any("'principal' finding" in p for p in validate_receipt_body(doc["body"]))


def test_every_built_in_finding_records_the_evidence_it_decided_on():
    decision = certify_authorization(_request(), _context(now=None))
    prov = decision.evidence_provenance.plain()
    for f in decision.findings:
        dimension = f.check.split(".", 1)[1]
        assert f.context["evidence"] == {
            k: prov[k] for k in EVIDENCE_FIELDS[dimension] if k in prov
        }
    assert set(sum((list(v) for v in EVIDENCE_FIELDS.values()), [])) == set(
        AuthorizationContext.__dataclass_fields__
    )  # every context field is consumed by some dimension, so provenance reconciles fully
    policy = [f for f in decision.findings if f.check.endswith("policy_binding")][0]
    assert policy.context["current_policy_version"] == "v3"
    manifest = [f for f in decision.findings if f.check.endswith("manifest_binding")][0]
    assert manifest.context["current_manifest_fingerprint"] is None


def test_malformed_parsed_values_never_escape_as_type_errors():
    receipt = _receipt()
    doc = _document(receipt)
    doc["body"]["findings"][0]["severity"] = []
    assert any("severity" in p for p in validate_receipt_body(doc["body"]))
    doc = _document(receipt, decision=[], authorized=[])
    problems = validate_receipt_body(doc["body"])
    assert any("decision" in p for p in problems) and any("authorized" in p for p in problems)
    doc = _document(receipt)
    doc["body"]["findings"][0]["check"] = ["x"]
    assert any("check" in p for p in validate_receipt_body(doc["body"]))
    for bad in (None, 5, object()):
        with pytest.raises(ValueError):
            certify_dimensions(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            certify_authorization(_request(), _context(), findings=bad)  # type: ignore[arg-type]


# --- the freeze and the dependency direction ---------------------------------------------


def test_the_kernel_is_untouched():
    """recusal/evidence.py at 0.9.0 is byte-identical to 0.8.0 (STABILITY.md, frozen item 1)."""
    with open(os.path.join(REPO, "recusal", "evidence.py"), "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    assert digest == EVIDENCE_PY_SHA256_AT_0_8_0


# Pinned from the v0.8.0 tag (tests/test_authorization.py is the only place that may
# change this value, and only in a MAJOR release per STABILITY.md).
EVIDENCE_PY_SHA256_AT_0_8_0 = "95c8f9635f5be8f018e6b9b9beb089d716d36ebcc15e73f1ae85a6e4c2028a3c"


def test_module_depends_only_on_the_standard_library_and_the_evidence_kernel():
    """Dependency direction: authorization -> evidence, nothing else in recusal, no third
    party. The kernel must never import authorization (checked from the other side)."""
    with open(os.path.join(REPO, "recusal", "authorization.py"), encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    assert imported == {
        "hashlib",
        "json",
        "math",
        "re",
        "dataclasses",
        "datetime",
        "typing",
        ".evidence",
        ".",
    }
    with open(os.path.join(REPO, "recusal", "evidence.py"), encoding="utf-8") as fh:
        assert "authorization" not in fh.read()


def test_module_is_stdlib_only_and_never_reads_a_clock():
    with open(os.path.join(REPO, "recusal", "authorization.py"), encoding="utf-8") as fh:
        source = fh.read()
    for forbidden in (
        "datetime.now",
        "time.time",
        "utcnow",
        "import time",
        "import random",
        "import os",
        "default=str",
    ):
        assert forbidden not in source, forbidden
    assert "import hmac" not in source and "sign(" not in source  # digest-bound, not signed
    assert "allow_nan=False" in source
