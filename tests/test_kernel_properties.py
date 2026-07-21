"""Property tests for the frozen evidence-kernel invariants (ledger item 5).

Generative locks over ``compute_verdict``, ``Finding.coerce``, and
``tool_fingerprint``: the invariants below must hold for ALL inputs the
strategies can produce, not just hand-picked examples. ``derandomize=True`` keeps
every run (local and CI, any platform) on the same deterministic example stream.

The affirmative-token sets are DUPLICATED here from the documented contract in
recusal/evidence.py on purpose: if the kernel's allowlist drifts, this file
fails, which is the review conversation the freeze exists to force.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from recusal.evidence import Decision, Finding, Severity, compute_verdict
from recusal.mcp import tool_fingerprint

DETERMINISTIC = settings(derandomize=True, max_examples=150)

# The documented contract: only these strings may read as an affirmative
# ``passed`` value / passing ``status`` (see _TRUE_LIKE / _PASS_LIKE).
PASS_TOKENS = {"pass", "passed", "ok", "okay", "success", "succeeded", "green"}
TRUE_TOKENS = {"true", "t", "1", "yes", "y", "on"} | PASS_TOKENS

_RANK = {Decision.PASS: 0, Decision.RETRY: 1, Decision.FAIL: 2}

findings = st.builds(
    Finding,
    check=st.text(min_size=1, max_size=8),
    severity=st.sampled_from(list(Severity)),
    passed=st.booleans(),
    message=st.text(max_size=16),
)
finding_lists = st.lists(findings, max_size=12)


@DETERMINISTIC
@given(finding_lists, findings)
def test_adding_a_finding_never_weakens_the_decision(items, extra):
    before = _RANK[compute_verdict(items).decision]
    after = _RANK[compute_verdict([*items, extra]).decision]
    assert after >= before


@DETERMINISTIC
@given(finding_lists.flatmap(lambda xs: st.tuples(st.just(xs), st.permutations(xs))))
def test_verdict_is_independent_of_finding_order(pair):
    original, shuffled = pair
    a, b = compute_verdict(original), compute_verdict(shuffled)
    assert a.decision is b.decision
    assert a.highest_severity is b.highest_severity
    assert sorted((f.check, f.severity) for f in a.failures) == sorted(
        (f.check, f.severity) for f in b.failures
    )


@DETERMINISTIC
@given(finding_lists)
def test_decision_rule_matches_the_documented_fold(items):
    verdict = compute_verdict(items)
    if any(not f.passed and f.severity is Severity.CRITICAL for f in items):
        assert verdict.decision is Decision.FAIL
        assert verdict.highest_severity is Severity.CRITICAL
    elif any(not f.passed and f.severity is Severity.ERROR for f in items):
        assert verdict.decision is Decision.RETRY
        assert verdict.highest_severity is Severity.ERROR
    else:
        assert verdict.decision is Decision.PASS
        assert not verdict.failures


@DETERMINISTIC
@given(st.text(max_size=16))
def test_coerce_passed_string_never_fails_open(value):
    coerced = Finding.coerce({"severity": "CRITICAL", "passed": value})
    assert coerced.passed == (value.strip().lower() in TRUE_TOKENS)


@DETERMINISTIC
@given(st.text(max_size=16))
def test_coerce_status_string_never_fails_open(value):
    coerced = Finding.coerce({"severity": "CRITICAL", "status": value})
    assert coerced.passed == (value.strip().lower() in PASS_TOKENS)


@DETERMINISTIC
@given(st.dictionaries(st.sampled_from(["message", "detail", "count"]), st.text(max_size=8)))
def test_coerce_strict_refuses_dicts_with_no_stated_outcome(context_only):
    try:
        Finding.coerce({"severity": "ERROR", **context_only}, strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError("strict coerce must refuse evidence with no stated outcome")


json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-(10**9), 10**9) | st.text(max_size=8),
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=6), children, max_size=3)
    ),
    max_leaves=12,
)
tool_declarations = st.dictionaries(st.text(min_size=1, max_size=6), json_values, max_size=4)


def _reinserted(value):
    """The same JSON value with every dict's key insertion order reversed."""
    if isinstance(value, dict):
        return {k: _reinserted(v) for k, v in reversed(list(value.items()))}
    if isinstance(value, list):
        return [_reinserted(v) for v in value]
    return value


@DETERMINISTIC
@given(tool_declarations)
def test_fingerprint_is_stable_across_calls_and_key_order(tool):
    first = tool_fingerprint(tool)
    assert first.startswith("sha256:")
    assert tool_fingerprint(tool) == first
    assert tool_fingerprint(_reinserted(tool)) == first
