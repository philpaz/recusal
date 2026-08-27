"""Property tests for recusal.authorization (Hypothesis, dev extra).

What random testing can establish here, stated so it is not read as more:

- freeze/thaw is the identity on the JSON domain, and canonical bytes are stable under
  key order;
- canonicalization performs no lossy normalization: the Python values that compare
  equal or that json.dumps would merge (int and str keys, 1 and 1.0 and True, -0.0 and
  0.0, {} and []) are kept apart or refused, checked directly, not by sampling digests;
- malformed or missing evidence never authorizes: any context with at least one built-in
  dimension's evidence absent refuses, and arbitrary JSON in the trust-rule slot passes
  the principal dimension only when it binds the exact label to a nonempty string.

Digest collision resistance is inherited from SHA-256 and is not something a property
test can establish; nothing below claims it.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from recusal.authorization import (
    ActionRequest,
    AuthorizationContext,
    Constraints,
    Supplied,
    canonical_json,
    certify_authorization,
    fingerprint,
)

_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=12),
)
json_values = st.recursive(
    _scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=6), children, max_size=4),
    ),
    max_leaves=12,
)
json_objects = st.dictionaries(st.text(min_size=1, max_size=6), json_values, max_size=4)


@settings(max_examples=300)
@given(json_values)
def test_freeze_thaw_is_the_identity_on_the_json_domain(value):
    assert canonical_json(Supplied(value, "p").plain()) == canonical_json(value)


@settings(max_examples=200)
@given(json_objects)
def test_canonical_bytes_are_stable_under_key_order(arguments):
    reordered = dict(reversed(list(arguments.items())))
    assert canonical_json(arguments) == canonical_json(reordered)
    a = ActionRequest(principal="a", tool="t", operation="o", arguments=arguments)
    b = ActionRequest(principal="a", tool="t", operation="o", arguments=reordered)
    assert a.arguments_fingerprint == b.arguments_fingerprint
    assert a.action_fingerprint == b.action_fingerprint


@pytest.mark.parametrize(
    "left, right",
    [
        ({"x": 1}, {"x": 1.0}),  # equal in Python, different JSON numbers
        ({"x": 1}, {"x": True}),  # 1 == True in Python
        ({"x": 0}, {"x": False}),
        ({"x": -0.0}, {"x": 0.0}),
        ({"x": {}}, {"x": []}),  # empty object vs empty array
        ({"x": "1"}, {"x": 1}),
        ({"x": None}, {"x": "null"}),
        ({"x": [1, 2]}, {"x": [2, 1]}),  # arrays are ordered
        ({"1": "a"}, {"01": "a"}),
    ],
)
def test_collision_prone_python_values_are_kept_apart(left, right):
    assert canonical_json(left) != canonical_json(right)
    assert fingerprint(left) != fingerprint(right)


@pytest.mark.parametrize(
    "value",
    [{1: "a"}, {1.5: "a"}, {True: "a"}, {None: "a"}, {(1, 2): "a"}, {"x": {1: "a"}}],
)
def test_values_json_dumps_would_normalize_are_refused(value):
    """json.dumps stringifies these keys; the strict domain refuses them instead."""
    with pytest.raises(ValueError):
        canonical_json(value)


_FIELDS = (
    "trusted_principals",
    "active_subject",
    "allowed_operations",
    "constraints",
    "now",
    "calls_so_far",
    "used_nonces",
    "approval",
    "policy_version",
)
_GOOD = {
    "trusted_principals": Supplied({"label": "svc"}, "cfg"),
    "active_subject": Supplied("r", "s"),
    "allowed_operations": Supplied({"t": ["op"]}, "g"),
    "constraints": Supplied(Constraints(max_calls=3, expires_at="2026-08-26T20:00:00+00:00"), "g"),
    "now": Supplied("2026-08-26T19:00:00+00:00", "c"),
    "calls_so_far": Supplied(0, "c"),
    "used_nonces": Supplied([], "n"),
    "approval": Supplied({"policy_version": "v"}, "g"),
    "policy_version": Supplied("v", "cfg"),
}


@settings(max_examples=200)
@given(st.sets(st.sampled_from(_FIELDS), min_size=1))
def test_missing_evidence_never_authorizes_whatever_else_is_supplied(absent):
    """Drop at least one required item, keep the rest well-formed. The decision must
    refuse: absence is never a pass, whatever else is present."""
    kwargs = {name: _GOOD[name] for name in _FIELDS if name not in absent}
    context = AuthorizationContext(**kwargs)
    request = ActionRequest(principal="label", tool="t", operation="op", resource="r", nonce="n")
    decision = certify_authorization(request, context)
    assert not decision.authorized
    assert decision.verdict.decision.value == "FAIL"


@settings(max_examples=200)
@given(json_values)
def test_arbitrary_evidence_in_the_trust_rule_slot_authorizes_only_when_it_binds_the_label(value):
    context = AuthorizationContext(**{**_GOOD, "trusted_principals": Supplied(value, "cfg")})
    request = ActionRequest(principal="label", tool="t", operation="op", resource="r", nonce="n")
    decision = certify_authorization(request, context)
    binds = (
        isinstance(value, dict) and isinstance(value.get("label"), str) and bool(value.get("label"))
    )
    assert decision.authorized == binds
    if not binds:
        assert any(f.check == "authorization.principal" for f in decision.verdict.failures)
