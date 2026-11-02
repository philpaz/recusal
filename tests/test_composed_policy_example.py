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


# --- control identity survives composition --------------------------------------------------


def _noop(_name, _input):
    return []


def _manifest_and_log(tmp_path, label="a"):
    from recusal.mcp import build_manifest, manifest_policy, manifest_to_text

    manifest_path = tmp_path / f"mcp-manifest-{label}.json"
    manifest_path.write_text(
        manifest_to_text(build_manifest({"github": [{"name": "create_issue"}]})),
        encoding="utf-8",
    )
    return manifest_policy(str(manifest_path))


def _record(tmp_path, policy, label):
    import json

    from recusal.audit import AuditLog

    path = tmp_path / f"audit-{label}.jsonl"
    decision, _ = decide(
        "mcp__github__create_issue", {}, policy, audit=AuditLog(path=str(path)), surface="test"
    )
    entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return decision, entry["action"]["control"]


def test_composition_keeps_the_manifest_digest_the_direct_policy_records(tmp_path):
    direct = _manifest_and_log(tmp_path)
    composed = compose_policies(_manifest_and_log(tmp_path))
    decision_direct, control_direct = _record(tmp_path, direct, "direct")
    decision_composed, control_composed = _record(tmp_path, composed, "composed")
    assert decision_direct == decision_composed == "defer"
    assert control_direct["manifest_sha256"].startswith("sha256:")
    assert control_composed == control_direct


def test_adding_a_no_op_policy_changes_neither_the_verdict_nor_the_control_identity(tmp_path):
    alone = compose_policies(_manifest_and_log(tmp_path))
    with_noop = compose_policies(_noop, _manifest_and_log(tmp_path), _noop)
    assert _record(tmp_path, alone, "alone") == _record(tmp_path, with_noop, "with_noop")


def test_a_composition_resets_the_identity_it_forwards(tmp_path):
    composed = compose_policies(_noop, _manifest_and_log(tmp_path))
    composed("mcp__github__create_issue", {})
    assert composed.get_control_identity()["manifest_sha256"].startswith("sha256:")
    composed.reset_control_identity()
    assert composed.get_control_identity()["manifest_sha256"] is None


def test_two_manifest_policies_in_one_gate_are_refused_up_front(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="one manifest_policy per gate"):
        compose_policies(_manifest_and_log(tmp_path, "a"), _manifest_and_log(tmp_path, "b"))


def test_a_composition_of_plain_policies_carries_no_identity_of_its_own():
    composed = compose_policies(_passed, _noop)
    assert composed.get_control_identity() == {}
    composed.reset_control_identity()  # nothing to reset, nothing raised


# --- nested compositions ---------------------------------------------------------------------


def test_nested_groups_of_plain_policies_compose_and_keep_their_order():
    order = []

    def a(_n, _i):
        order.append("a")
        return []

    def b(_n, _i):
        order.append("b")
        return []

    def c(_n, _i):
        order.append("c")
        return [Finding.fail("c", severity="CRITICAL")]

    composed = compose_policies(compose_policies(a), compose_policies(b, compose_policies(c)))
    assert [p.__name__ for p in composed.policies] == ["a", "b", "c"]  # flattened, in order
    decision, _ = decide("Bash", {}, composed)
    assert decision == "deny" and order == ["a", "b", "c"]
    assert composed.get_control_identity() == {}


def test_a_nested_plain_group_beside_one_manifest_policy_keeps_its_identity(tmp_path):
    manifest = _manifest_and_log(tmp_path)
    composed = compose_policies(compose_policies(_noop, _passed), manifest, compose_policies(_noop))
    direct = _record(tmp_path, manifest, "direct")
    assert _record(tmp_path, composed, "nested") == direct


def test_two_manifest_policies_across_nesting_are_still_refused(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="one manifest_policy per gate"):
        compose_policies(
            compose_policies(_manifest_and_log(tmp_path, "a")),
            compose_policies(_noop, _manifest_and_log(tmp_path, "b")),
        )
