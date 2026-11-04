"""The evaluator holds a policy's return value to its contract (fifth 2026-11-03 review).

``findings = policy(...) or []`` read every false value, ``False``, ``0``, ``""``, ``{}``, as
"no findings", so a predicate written as a policy, or an error sentinel, switched its check
off silently: the decision was ``defer`` and the tool ran. Now only ``None`` and an iterable
of findings are accepted; everything else is a policy error, which fails closed.
"""

import json
import sys
from pathlib import Path

import pytest

from recusal import Finding
from recusal.audit import AuditLog
from recusal.claude_code import _findings_of, decide

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOT_FINDINGS = [False, True, 0, 1, 0.0, 2.5, "", "x", b"", {}, {"check": "a", "passed": False}]
NO_OBJECTIONS = [None, [], (), iter(())]


@pytest.mark.parametrize("value", NOT_FINDINGS, ids=repr)
def test_a_value_that_is_not_a_list_of_findings_fails_closed(value):
    decision, reason = decide("Bash", {"command": "rm -rf /"}, lambda n, a: value)
    assert decision == "deny"
    assert "failed closed (policy error)" in reason
    assert "list of findings" in reason and type(value).__name__ in reason


@pytest.mark.parametrize("value", NO_OBJECTIONS, ids=lambda v: type(v).__name__)
def test_none_and_an_empty_iterable_are_no_objections(value):
    assert decide("Read", {"file_path": "x"}, lambda n, a: value)[0] == "defer"


def test_any_iterable_of_findings_is_folded_not_only_a_list():
    def as_tuple(n, a):
        return (Finding.fail("unsafe", severity="CRITICAL"),)

    def as_generator(n, a):
        yield Finding.fail("unsafe", severity="CRITICAL")

    assert decide("Bash", {}, as_tuple)[0] == "deny"
    assert decide("Bash", {}, as_generator)[0] == "deny"


def test_the_helper_names_the_offending_value():
    with pytest.raises(TypeError, match=r"policy returned bool False"):
        _findings_of(False)
    with pytest.raises(TypeError, match=r"policy returned dict \{\}"):
        _findings_of({})
    with pytest.raises(TypeError, match="not a list of findings"):
        _findings_of(object())
    assert _findings_of(None) == []
    assert _findings_of(x for x in [1]) == [1]


def test_fail_open_configuration_still_reports_the_contract_violation():
    decision, reason = decide("Read", {}, lambda n, a: False, fail_closed=False)
    assert decision == "defer" and "policy error ignored" in reason and "bool False" in reason


def test_the_audit_record_of_a_contract_violation_states_the_policy_error(tmp_path):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    decision, _ = decide("Bash", {"command": "x"}, lambda n, a: 0, audit=log, surface="test")
    assert decision == "deny"
    entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["decision"] == "FAIL"  # the verdict the record carries
    assert "recusal_policy_error" in json.dumps(entry)


def test_a_real_adapter_runs_nothing_for_a_policy_that_returns_false():
    """The reviewer's probe: the OpenAI Responses loop needs no SDK and executes nothing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recusal_contract_responses_example", ROOT / "examples" / "openai_responses_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXECUTED.clear()
    response = module.response_with(
        [module.function_call("run_sql", {"sql": "DELETE FROM audit_log"}, "c1")]
    )
    outputs = module.gate_function_calls(response, module.TOOLS, lambda n, a: False)
    assert module.EXECUTED == []
    assert "failed closed" in outputs[0]["output"]
