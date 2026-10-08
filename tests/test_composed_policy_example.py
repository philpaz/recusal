"""Recipe 10: composition is deterministic, conservative and fail-closed."""

from itertools import permutations

from examples.composed_policies import compose_policies, make_policy
from recusal import Decision, Finding, Severity, compute_verdict
from recusal.claude_code import decide


def _passed(_name, _input):
    return [Finding.ok("checked", severity="WARNING")]


def _recoverable(_name, _input):
    return [Finding.fail("needs_revision", severity="ERROR")]


def _critical(_name, _input):
    return [Finding.fail("unsafe_action", severity="CRITICAL")]


def test_worst_severity_dominates_even_with_passing_checks():
    evidence = compose_policies(_passed, _recoverable, _critical)("Read", {})
    verdict = compute_verdict(evidence)
    assert verdict.decision is Decision.FAIL
    assert verdict.highest_severity is Severity.CRITICAL
    assert [finding.check for finding in verdict.failures] == ["unsafe_action"]


def test_clean_read_defers_to_normal_host_permissions(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = make_policy(workspace)
    decision, _ = decide("Read", {"file_path": str(workspace / "notes.txt")}, policy)
    assert decision == "defer"
    assert compute_verdict(policy("Read", {})).decision is Decision.PASS


def test_existing_policies_all_refuse_their_own_risky_inputs(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = make_policy(workspace, {"example.com"})
    cases = (
        ("Bash", {"command": "rm -rf /"}),
        ("Write", {"file_path": str(tmp_path / "outside.txt")}),
        ("http_post", {"url": "https://evil.example/events"}),
    )
    for tool, inputs in cases:
        decision, _reason = decide(tool, inputs, policy)
        assert decision == "deny", tool


def test_policy_error_fails_closed_in_existing_adapter():
    def broken(_name, _input):
        raise RuntimeError("synthetic policy error")

    policy = compose_policies(_passed, broken)
    decision, reason = decide("Read", {}, policy)
    assert decision == "deny"
    assert "failed closed" in reason.lower()
    assert "policy error" in reason.lower()


def test_rule_order_does_not_change_verdict():
    for rules in permutations((_passed, _recoverable, _critical)):
        verdict = compute_verdict(compose_policies(*rules)("Read", {}))
        assert verdict.decision is Decision.FAIL
        assert verdict.highest_severity is Severity.CRITICAL
        assert [finding.check for finding in verdict.failures] == ["unsafe_action"]


def test_recoverable_only_is_retry_not_a_terminal_refusal():
    verdict = compute_verdict(compose_policies(_passed, _recoverable)("Read", {}))
    assert verdict.decision is Decision.RETRY
