"""recusal.authorization: an action is authorized only inside supplied authority evidence.

What these tests pin, in the order the module makes its claims:

- every dimension refuses when its evidence is absent (absence never passes);
- the principal dimension passes ONLY through a supplied trust rule;
- the composition invariant: an unrelated passing finding cannot certify a request that
  lacks a required dimension, because a missing dimension is synthesized as a refusal;
- expiry, budget and replay verify SUPPLIED clock, counter and use-state evidence, and
  the documented race (two callers observing the same count both pass) is real;
- the receipt is deterministic, digest-bound, and detects a changed byte, and its own
  docstring limits hold (no timestamp unless supplied, not authenticated by digest);
- the kernel is untouched: recusal/evidence.py is byte-identical to its 0.8.0 content.
"""

import hashlib
import json
import os

import pytest

from recusal import Finding, certify_evidence
from recusal.authorization import (
    DEFAULT_DIMENSIONS,
    DIMENSIONS,
    PROVENANCE_CLAUDE_PRETOOLUSE,
    ActionRequest,
    AuthorizationContext,
    Constraints,
    DecisionReceipt,
    Supplied,
    certify_authorization,
    certify_dimensions,
    check_budget,
    check_expiry,
    check_principal,
    check_replay,
    run_checks,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOW = "2026-08-26T19:00:00+00:00"
LATER = "2026-08-26T21:00:00+00:00"


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


def _checks_by_dimension(request, context):
    return {f.check.split(".", 1)[1]: f for f in run_checks(request, context)}


# --- the happy path, then absence -------------------------------------------------------


def test_fully_evidenced_request_is_authorized_with_one_passing_finding_per_dimension():
    decision = certify_authorization(_request(), _context())
    assert decision.authorized
    assert decision.verdict.decision.value == "PASS"
    assert [f.check for f in decision.findings] == ["authorization." + d for d in DIMENSIONS]
    assert all(f.passed for f in decision.findings)
    # every finding names where its evidence came from
    assert all("provenance" in f.context for f in decision.findings)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_every_dimension_refuses_when_its_evidence_is_absent(dimension):
    """Absent authority evidence is a refusal, never a vacuous pass."""
    ctx = AuthorizationContext()  # nothing supplied at all
    findings = _checks_by_dimension(_request(), ctx)
    assert not findings[dimension].passed, dimension
    assert findings[dimension].severity.value == "CRITICAL"
    assert not certify_authorization(_request(), ctx).authorized


def test_default_required_dimensions_are_all_of_them():
    assert DEFAULT_DIMENSIONS == DIMENSIONS


# --- principal: a label is a claim, the trust rule is the authority ----------------------


def test_principal_passes_only_through_a_supplied_trust_rule():
    ok = check_principal(_request(), _context())
    assert ok.passed and ok.context["principal"] == "svc:invoice"

    no_rule = check_principal(_request(), _context(trusted_principals=None))
    assert not no_rule.passed
    assert "runtime label is not an authenticated identity" in no_rule.message

    unbound = check_principal(
        _request(), _context(trusted_principals=Supplied({}, "operator_config"))
    )
    assert not unbound.passed
    assert "not bound to any configured principal" in unbound.message


def test_a_runtime_label_alone_never_authorizes_even_with_every_other_dimension_satisfied():
    """The security adjustment in one test: Claude's agent_id is not an identity."""
    ctx = _context(
        trusted_principals=None,
    )
    decision = certify_authorization(_request(principal="a343c667c7dc99439"), ctx)
    assert not decision.authorized
    assert [f.check for f in decision.verdict.failures] == ["authorization.principal"]


def test_runtime_provenance_label_is_the_one_the_hook_writes():
    assert PROVENANCE_CLAUDE_PRETOOLUSE == "claude_pretooluse_event"


# --- the composition invariant -----------------------------------------------------------


def test_an_unrelated_passing_finding_cannot_certify_a_missing_dimension():
    """certify_evidence alone would PASS here; certify_dimensions must not."""
    unrelated = [Finding.ok("something_else", severity="CRITICAL", message="fine")]
    assert certify_evidence(unrelated).passed  # the hazard, demonstrated
    verdict = certify_dimensions(unrelated, required=("principal", "subject"))
    assert not verdict.passed
    missing = [f for f in verdict.failures if f.check == "authorization.dimension_missing"]
    assert sorted(f.context["dimension"] for f in missing) == ["principal", "subject"]


def test_certify_dimensions_with_nothing_required_still_refuses_empty_evidence():
    assert not certify_dimensions([], required=()).passed  # certify_evidence's own rule


def test_adopter_findings_can_satisfy_a_custom_required_dimension():
    custom = Finding.ok("authorization.residency", severity="CRITICAL", message="EU")
    decision = certify_authorization(
        _request(), _context(), required=DIMENSIONS + ("residency",), findings=[custom]
    )
    assert decision.authorized
    without = certify_authorization(_request(), _context(), required=DIMENSIONS + ("residency",))
    assert not without.authorized
    assert without.verdict.failures[0].check == "authorization.dimension_missing"


def test_narrowing_required_is_explicit_and_still_runs_every_check():
    ctx = _context(used_nonces=None)  # replay evidence absent
    assert not certify_authorization(_request(), ctx).authorized
    narrowed = certify_authorization(_request(), ctx, required=("principal", "subject"))
    # the replay finding is still produced and still fails; the fold ignores no failure
    assert not narrowed.authorized
    assert [f.check for f in narrowed.verdict.failures] == ["authorization.replay"]


# --- subject, tool, arguments ------------------------------------------------------------


def test_wrong_subject_is_refused_and_named():
    f = _checks_by_dimension(_request(resource="customer:999"), _context())["subject"]
    assert not f.passed and f.context == {
        "target": "customer:999",
        "active": "customer:456",
        "provenance": "session_store",
    }


def test_tool_and_operation_must_both_be_granted():
    by = _checks_by_dimension
    assert not by(_request(tool="crm.delete_record"), _context())["tool_operation"].passed
    assert not by(_request(operation="delete"), _context())["tool_operation"].passed
    assert by(_request(), _context())["tool_operation"].passed


def test_arguments_outside_the_granted_fields_are_refused():
    f = _checks_by_dimension(_request(arguments={"status": "x", "balance": 0}), _context())[
        "arguments"
    ]
    assert not f.passed and f.context["extra"] == ["balance"]


def test_value_bound_is_verified_on_the_named_argument():
    ctx = _context(
        constraints=Supplied(Constraints(max_value=100.0, value_field="amount"), "grant")
    )
    inside = _checks_by_dimension(_request(arguments={"amount": 99.5}), ctx)["arguments"]
    over = _checks_by_dimension(_request(arguments={"amount": 100.5}), ctx)["arguments"]
    nan = _checks_by_dimension(_request(arguments={"amount": "lots"}), ctx)["arguments"]
    assert inside.passed and not over.passed and not nan.passed


def test_constraints_validate_their_own_shape():
    with pytest.raises(ValueError):
        Constraints(max_calls=-1)
    with pytest.raises(ValueError):
        Constraints(max_value=1.0)  # value_field missing


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


# --- policy and manifest binding ---------------------------------------------------------


def test_policy_change_since_approval_is_refused():
    f = _checks_by_dimension(_request(), _context(policy_version=Supplied("v4", "cfg")))[
        "policy_binding"
    ]
    assert not f.passed and f.context == {"approved": "v3", "current": "v4", "provenance": "grant"}


def test_manifest_binding_is_two_sided():
    by = _checks_by_dimension
    assert by(_request(), _context())["manifest_binding"].passed  # neither side: stated
    one_sided = by(_request(), _context(manifest_fingerprint=Supplied("abc", "mcp")))[
        "manifest_binding"
    ]
    assert not one_sided.passed and "one-sided" in one_sided.message
    both = _context(
        approval=Supplied({"policy_version": "v3", "manifest_fingerprint": "abc"}, "grant"),
        manifest_fingerprint=Supplied("abc", "mcp"),
    )
    assert by(_request(), both)["manifest_binding"].passed
    drift = _context(
        approval=Supplied({"policy_version": "v3", "manifest_fingerprint": "abc"}, "grant"),
        manifest_fingerprint=Supplied("def", "mcp"),
    )
    assert not by(_request(), drift)["manifest_binding"].passed


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


def test_request_rejects_empty_identity_parts():
    for name in ("principal", "tool", "operation"):
        with pytest.raises(ValueError):
            _request(**{name: ""})


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

    f = _checks_by_dimension(_request(), _context(active_subject=Bad("x", "p")))["subject"]
    assert not f.passed and "boom" in f.message


# --- the receipt -----------------------------------------------------------------------------


def test_receipt_is_deterministic_and_carries_the_passing_findings():
    d1 = certify_authorization(_request(), _context())
    d2 = certify_authorization(_request(), _context())
    r1 = DecisionReceipt.build(d1, context=_context(), policy_version="v3")
    r2 = DecisionReceipt.build(d2, context=_context(), policy_version="v3")
    assert r1.to_json() == r2.to_json()
    assert r1.digest == r2.digest
    assert r1.body["authorized"] is True
    assert len(r1.body["findings"]) == len(DIMENSIONS)
    assert r1.body["evidence_provenance"]["trusted_principals"] == "operator_config"
    assert r1.body["timestamp"] is None  # nothing is stamped unless supplied


def test_receipt_digest_is_sha256_over_the_canonical_body_and_detects_a_changed_byte():
    receipt = DecisionReceipt.build(certify_authorization(_request(), _context()))
    canonical = json.dumps(
        dict(receipt.body), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    assert receipt.digest == hashlib.sha256(canonical.encode()).hexdigest()
    assert receipt.digest_matches()
    tampered = json.loads(receipt.to_json())
    tampered["body"]["authorized"] = False
    assert not DecisionReceipt.from_json(json.dumps(tampered)).digest_matches()


def test_receipt_digest_alone_does_not_authenticate_anything():
    """Anyone can build a self-consistent receipt; the digest proves bytes, not origin."""
    forged = DecisionReceipt(
        {"authorized": True}, hashlib.sha256(b'{"authorized":true}').hexdigest()
    )
    assert forged.digest_matches()  # consistent, and meaningless without an anchor


def test_receipt_binds_to_an_audit_entry_when_one_is_supplied():
    decision = certify_authorization(_request(), _context())
    entry = {"seq": 7, "hash": "f" * 64}
    receipt = DecisionReceipt.build(decision, audit_entry=entry)
    assert receipt.body["audit_seq"] == 7 and receipt.body["audit_head"] == "f" * 64
    with pytest.raises(ValueError):
        DecisionReceipt.build(decision, audit_entry={"seq": "7"})


def test_receipt_round_trips_through_json():
    receipt = DecisionReceipt.build(certify_authorization(_request(), _context()))
    again = DecisionReceipt.from_json(receipt.to_json())
    assert again.digest == receipt.digest and again.digest_matches()
    with pytest.raises(ValueError):
        DecisionReceipt.from_json("[]")


# --- the freeze -----------------------------------------------------------------------------


def test_the_kernel_is_untouched():
    """recusal/evidence.py at 0.9.0 is byte-identical to 0.8.0 (STABILITY.md, frozen item 1)."""
    with open(os.path.join(REPO, "recusal", "evidence.py"), "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    assert digest == EVIDENCE_PY_SHA256_AT_0_8_0


# Pinned from the v0.8.0 tag (tests/test_authorization.py is the only place that may
# change this value, and only in a MAJOR release per STABILITY.md).
EVIDENCE_PY_SHA256_AT_0_8_0 = "95c8f9635f5be8f018e6b9b9beb089d716d36ebcc15e73f1ae85a6e4c2028a3c"


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
    ):
        assert forbidden not in source, forbidden
    assert "import hmac" not in source and "sign(" not in source  # digest-bound, not signed
