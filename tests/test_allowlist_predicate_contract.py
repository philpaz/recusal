"""The allowlist predicate's result is a bool, nothing else (2026-11-04 adapters review).

The explicit predicate seam is annotated ``Callable[[dict], bool]`` but was read with
ordinary truthiness, so ``"false"``, ``{"allowed": False}`` and a failed ``Finding`` all
read as True and removed the refusal (the decision was ``defer``). Only ``True`` allows and
only ``False`` refuses now; any other result is a named denial.
"""

import pytest

from recusal import Finding
from recusal.claude_code import allowlist_policy, decide

MALFORMED = [
    "false",
    "true",
    {"allowed": False, "note": "MODEL_SECRET_SENTINEL"},
    Finding.fail("x", severity="CRITICAL"),
    [Finding.fail("x", severity="CRITICAL")],
    [],
    None,
    0,
    1,
    1.0,
]


@pytest.mark.parametrize("result", MALFORMED, ids=repr)
def test_a_result_that_is_not_a_bool_is_a_named_denial(result):
    policy = allowlist_policy(allow={"Bash": lambda _input: result})
    decision, reason = decide("Bash", {"command": "ls"}, policy)
    assert decision == "deny"
    assert "predicate returned" in reason and "not a bool" in reason
    assert type(result).__name__ in reason
    assert "allowed" not in reason and "MODEL" not in reason, "the reason names the type only"


def test_true_defers_and_false_refuses_with_the_original_reason():
    allowing = allowlist_policy(allow={"Bash": lambda _input: True})
    assert decide("Bash", {"command": "ls"}, allowing)[0] == "defer"
    refusing = allowlist_policy(allow={"Bash": lambda _input: False})
    decision, reason = decide("Bash", {"command": "ls"}, refusing)
    assert decision == "deny" and "predicate refused it" in reason


def test_the_denial_is_a_finding_the_fold_can_read():
    policy = allowlist_policy(allow={"Read": lambda _input: "yes"})
    findings = policy("Read", {"file_path": "x"})
    assert [f.check for f in findings] == ["allowlist_predicate_contract"]
    assert findings[0].passed is False
