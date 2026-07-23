"""Drift locks for workflow-pipeline hardening (0.7.0 scope).

The CI pipeline is itself an attack surface, so its hardening is pinned like any
other behavior: zizmor audits every workflow file as a blocking CI job (version-
pinned, so a new zizmor release cannot silently change what green means), and every
checkout runs with ``persist-credentials: false`` - no job in this repository pushes,
and the default would leave a live token in ``.git`` for any later step to read.
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")


def _read(name):
    with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as fh:
        return fh.read()


def _workflow_files():
    return sorted(n for n in os.listdir(WORKFLOWS) if n.endswith((".yml", ".yaml")))


def test_zizmor_runs_as_a_blocking_versioned_ci_job():
    ci = _read("ci.yml")
    assert "zizmor" in ci, "the zizmor workflow-audit job left ci.yml"
    m = re.search(r"pip install zizmor==([\d.]+)", ci)
    assert m, "zizmor must be version-pinned; an unpinned auditor can change what green means"
    assert re.search(r"run: zizmor .*\.github/workflows", ci), (
        "the zizmor job must audit the workflow files themselves"
    )


def test_every_checkout_disables_credential_persistence():
    for name in _workflow_files():
        text = _read(name)
        # the captured block is ONLY the checkout step's own continuation lines: the
        # list-item rejection runs BEFORE any indentation is consumed (a lookahead
        # after `[ \t]+` can be backtracked around, silently absorbing the next step
        # and satisfying this checkout with a later step's settings - caught by
        # negative-case verification, kept fixed)
        for m in re.finditer(
            r"uses:\s*actions/checkout@[0-9a-f]{40}[^\n]*\n((?:(?![ \t]*-\s)[ \t]+[^\n]*\n)*)",
            text,
        ):
            block = m.group(1)
            assert "persist-credentials: false" in block, (
                f"{name}: a checkout without persist-credentials: false leaves a live "
                "token in .git for every later step (zizmor: artipacked)"
            )
