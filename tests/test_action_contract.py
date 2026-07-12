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
