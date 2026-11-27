"""Drift locks for action.yml's shell contract.

GitHub runs `shell: bash` steps with -e (fail-fast): without an explicit `set +e`, the
first failing gate would abort the script before `bump $?` ran, so later gates would
never execute and the intended highest-exit-code aggregation would silently not happen
(safe-side, but not the documented behavior).
"""

import os
import re

ACTION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "action.yml")


def test_the_adjudication_script_disables_fail_fast_before_aggregating():
    text = open(ACTION, encoding="utf-8").read()
    set_e = text.index("set +e")
    assert set_e < text.index("ran=0"), "set +e must precede the aggregation state"
    assert set_e < text.index("bump")


def test_every_gate_command_is_aggregated():
    text = open(ACTION, encoding="utf-8").read()
    assert len(re.findall(r"bump \$\?", text)) >= 4  # doctor, audit x2 (anchor branch), verdict x2


def _declared_inputs(text):
    block = text.split("\ninputs:\n", 1)[1].split("\nruns:", 1)[0]
    return re.findall(r"^  ([a-z][a-z-]*):\s*$", block, re.M)


def _readme_action_section():
    readme = os.path.join(os.path.dirname(ACTION), "README.md")
    text = open(readme, encoding="utf-8").read()
    return text.split("Or as a GitHub Action", 1)[1].split("\n## ", 1)[0]


def test_every_input_of_the_action_has_a_row_in_the_readme():
    """An input added to or renamed in action.yml without its
    row in the README's table of inputs fails here, and a row for an input that no longer
    exists fails too."""
    inputs = _declared_inputs(open(ACTION, encoding="utf-8").read())
    assert len(inputs) == 7, inputs
    rows = re.findall(r"^\| `([a-z-]+)` \|", _readme_action_section(), re.M)
    assert rows == inputs
