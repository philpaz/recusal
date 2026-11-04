"""A no-op wrapper changes nothing (2026-11-04 review, finding 3 and the wrapper contract).

``decide`` holds a policy's return to its contract; the wrappers (``compose_policies`` and
``manifest_policy(..., policy=)``) read the return themselves, so each must hold it to the
same contract before touching it, or ``{}`` becomes an empty screen and ``None`` an error.
Every shape below is compared with the same leaf evaluated directly: decision, reason,
invocation count, finding order, bound server, naming domain and control identity.
"""

import json
import sys
from pathlib import Path

import pytest

from recusal import Finding
from recusal.audit import AuditLog
from recusal.claude_code import decide
from recusal.mcp import build_manifest, manifest_policy, manifest_to_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.composed_policies import compose_policies  # noqa: E402

MCP_NAME = "mcp__db__run_sql"
RAW_NAME = "run_sql"


@pytest.fixture
def pin(tmp_path):
    path = tmp_path / "pin.json"
    path.write_text(
        manifest_to_text(build_manifest({"db": [{"name": "run_sql"}]})), encoding="utf-8"
    )
    return str(path)


class Leaf:
    """A policy returning a fixed value, counting its invocations."""

    def __init__(self, value):
        self.value, self.calls = value, 0

    def __call__(self, name, args):
        self.calls += 1
        return self.value() if callable(self.value) else self.value


def shapes(leaf, pin):
    """Every wrapper shape, with the tool name that reaches the leaf through it."""
    return {
        "direct": (leaf, RAW_NAME),
        "composed": (compose_policies(leaf), RAW_NAME),
        "nested": (compose_policies(compose_policies(leaf), lambda n, a: []), RAW_NAME),
        "manifest_non_mcp": (manifest_policy(pin, policy=leaf), RAW_NAME),
        "manifest_approved_mcp": (manifest_policy(pin, policy=leaf), MCP_NAME),
        "manifest_around_composed": (manifest_policy(pin, policy=compose_policies(leaf)), MCP_NAME),
        "composed_around_manifest": (compose_policies(manifest_policy(pin, policy=leaf)), MCP_NAME),
    }


def _critical():
    return Finding.fail("unsafe_action", severity="CRITICAL", message="refuse this")


def _empty_generator():
    return iter(())


def _finding_generator():
    yield _critical()


def _raises_after_yield():
    yield Finding.ok("fine", severity="WARNING")
    raise RuntimeError("leaf broke after yielding")


VALUES = {
    "None": None,
    "[]": [],
    "()": (),
    "empty generator": _empty_generator,
    "[finding]": [_critical()],
    "(finding,)": (_critical(),),
    "finding generator": _finding_generator,
    "[ok, critical]": [Finding.ok("a", severity="WARNING"), _critical()],
    "False": False,
    "0": 0,
    "1": 1,
    "''": "",
    "b''": b"",
    "{}": {},
    "bare finding dict": {"check": "a", "passed": False},
    "[malformed dict]": [{"no": "status"}],
    "raises after yield": _raises_after_yield,
}


@pytest.mark.parametrize("label", list(VALUES), ids=list(VALUES))
def test_every_wrapper_decides_exactly_as_the_direct_evaluation(label, pin):
    value = VALUES[label]
    direct = Leaf(value)
    expected = decide(RAW_NAME, {"sql": "x"}, direct)
    assert direct.calls == 1
    for shape, (wrapped, name) in shapes(Leaf(value), pin).items():
        if shape == "direct":
            continue
        decision, reason = decide(name, {"sql": "x"}, wrapped)
        # the reason names the tool as the wrapper presented it; everything else is equal
        assert (decision, reason.replace(name, RAW_NAME)) == expected, (shape, reason, expected)


def test_the_expected_decisions_are_the_contract_not_an_accident(pin):
    """What the table above pins: None and empty iterables defer; bools, numbers, strings,
    bytes and bare mappings are policy errors; malformed findings and a leaf that raises fail
    closed; a CRITICAL finding is a refusal."""
    for label, decision in {
        "None": "defer",
        "[]": "defer",
        "empty generator": "defer",
        "[finding]": "deny",
        "finding generator": "deny",
        "False": "deny",
        "0": "deny",
        "''": "deny",
        "b''": "deny",
        "{}": "deny",
        "bare finding dict": "deny",
        "[malformed dict]": "deny",
        "raises after yield": "deny",
    }.items():
        for shape, (wrapped, name) in shapes(Leaf(VALUES[label]), pin).items():
            got, reason = decide(name, {"sql": "x"}, wrapped)
            assert got == decision, (label, shape, reason)
            if label in ("False", "0", "''", "b''", "{}", "bare finding dict"):
                assert "policy error" in reason and "list of findings" in reason, (label, shape)
            if label == "None":
                assert "policy error" not in reason, (label, shape)


def test_every_wrapper_invokes_the_leaf_exactly_once_per_decision(pin):
    for shape, (wrapped, name) in shapes(Leaf([]), pin).items():
        leaf = wrapped if shape == "direct" else None
        counted = Leaf([])
        wrapped, name = shapes(counted, pin)[shape]
        decide(name, {}, wrapped)
        decide(name, {}, wrapped)
        assert counted.calls == 2, shape
        assert leaf is None or leaf.calls == 0


def test_a_false_valued_callable_is_still_a_policy(pin):
    class FalseValued:
        calls = 0

        def __bool__(self):
            return False

        def __call__(self, name, args):
            self.calls += 1
            return [_critical()]

    for shape, (wrapped, name) in shapes(FalseValued(), pin).items():
        decision, reason = decide(name, {}, wrapped)
        assert decision == "deny" and "refuse this" in reason, (shape, reason)


def test_finding_order_survives_composition(pin):
    first = Leaf([Finding.ok("one", severity="WARNING"), Finding.ok("two", severity="WARNING")])
    second = Leaf((Finding.fail("three", severity="ERROR"),))
    composed = compose_policies(first, second)
    assert [f.check for f in composed(RAW_NAME, {})] == ["one", "two", "three"]
    around = manifest_policy(pin, policy=composed)
    assert [f.check for f in around(MCP_NAME, {})] == ["one", "two", "three"]


def test_the_manifest_wrapper_refuses_before_the_leaf_when_it_cannot_approve(pin, tmp_path):
    leaf = Leaf({})  # would be a policy error if reached
    unavailable = manifest_policy(str(tmp_path / "missing.json"), policy=leaf)
    decision, reason = decide(MCP_NAME, {}, unavailable)
    assert decision == "deny" and "no usable MCP manifest" in reason and leaf.calls == 0
    unapproved = manifest_policy(pin, policy=leaf)
    decision, reason = decide("mcp__db__drop_table", {}, unapproved)
    assert decision == "deny" and "not in the pinned MCP manifest" in reason and leaf.calls == 0


def test_a_no_op_wrapper_keeps_binding_domain_and_identity(pin, tmp_path):
    def bound(_name, _input):
        return []

    bound.recusal_bound_server = "db"
    assert compose_policies(bound, lambda n, a: []).recusal_bound_server == "db"
    assert compose_policies(bound).recusal_name_domain is None

    manifest = manifest_policy(pin)
    assert manifest.recusal_name_domain == "claude_code"
    assert compose_policies(manifest, lambda n, a: []).recusal_name_domain == "claude_code"

    log_direct = AuditLog(path=str(tmp_path / "direct.jsonl"))
    log_composed = AuditLog(path=str(tmp_path / "composed.jsonl"))
    decide(MCP_NAME, {}, manifest, audit=log_direct, surface="t")
    decide(
        MCP_NAME, {}, compose_policies(manifest, lambda n, a: []), audit=log_composed, surface="t"
    )
    direct = json.loads((tmp_path / "direct.jsonl").read_text(encoding="utf-8"))["action"][
        "control"
    ]
    composed = json.loads((tmp_path / "composed.jsonl").read_text(encoding="utf-8"))["action"][
        "control"
    ]
    assert direct["manifest_sha256"].startswith("sha256:") and composed == direct
    composed_policy = compose_policies(manifest)
    composed_policy.reset_control_identity()
    assert composed_policy.get_control_identity()["manifest_sha256"] is None


def test_no_wrapper_reads_a_leaf_return_without_the_contract():
    """The drift lock: both wrappers read returns only through ``_findings_of``, and the
    manifest wrapper treats only ``None`` as an absent policy."""
    manifest_source = (ROOT / "recusal" / "_manifest_policy.py").read_text(encoding="utf-8")
    assert "list(policy(tool_name, tool_input))" not in manifest_source
    assert "if policy else" not in manifest_source
    assert manifest_source.count("_findings_of(policy(tool_name, tool_input))") == 2
    composed_source = (ROOT / "examples" / "composed_policies.py").read_text(encoding="utf-8")
    assert "findings.extend(check(tool_name, tool_input))" not in composed_source
    assert (
        "findings.extend(_findings_of(check(tool_name, copy.deepcopy(tool_input))))"
        in composed_source
    )


def test_a_mutating_leaf_beside_a_no_op_is_still_a_no_op_for_the_others(pin):
    """Adapters review, finding 2: a leaf that rewrites its argument must not change what the
    manifest wrapper or a sibling leaf judges, in any wrapper shape."""

    def rewrite(_name, args):
        args["sql"] = "SELECT 1"
        return []

    seen = []

    def recording(_name, args):
        seen.append(dict(args))
        return []

    for shape, (wrapped, name) in shapes(compose_policies(rewrite, recording), pin).items():
        seen.clear()
        decide(name, {"sql": "DELETE FROM t"}, wrapped)
        assert seen == [{"sql": "DELETE FROM t"}], shape
