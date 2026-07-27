"""The stability policy is a promise, so it is drift-locked like every other claim.

STABILITY.md names a frozen manifest schema version, a Python floor, and a maturity
classifier. Each of those is a fact stated elsewhere in the repository, and a policy that
silently disagrees with the code it governs is worse than no policy: it is a promise the
project has already broken without noticing.
"""

import os
import re

import recusal
from recusal.mcp import MANIFEST_VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def test_the_frozen_manifest_version_is_the_shipped_one():
    policy = _read("STABILITY.md")
    stated = re.search(r"\*\*Version (\d+) is frozen\.\*\*", policy)
    assert stated, "STABILITY.md must name the frozen manifest schema version"
    assert int(stated.group(1)) == MANIFEST_VERSION, (
        f"STABILITY.md freezes manifest v{stated.group(1)} but the package ships "
        f"v{MANIFEST_VERSION}; a schema change is a MAJOR under this policy"
    )


def test_the_stated_python_floor_matches_the_package_metadata():
    policy = _read("STABILITY.md")
    floor = re.search(r"Python \*\*(\d+\.\d+) or newer\*\*", policy)
    assert floor, "STABILITY.md must state the supported Python floor"
    assert f'requires-python = ">={floor.group(1)}"' in _read("pyproject.toml")


def test_the_maturity_claim_agrees_with_the_classifier():
    policy = _read("STABILITY.md")
    pyproject = _read("pyproject.toml")
    stated = re.search(r"the classifier is `Development Status :: ([^`]+)`", policy)
    assert stated, "STABILITY.md must name the packaging maturity classifier"
    assert f'"Development Status :: {stated.group(1)}"' in pyproject
    # 1.0 is earned, not announced: while the version is 0.x the policy must not claim
    # Production/Stable, and the preconditions must still be listed.
    if recusal.__version__.startswith("0."):
        assert "Production/Stable" not in stated.group(1)
        assert "what earns it" in policy


def test_the_policy_is_reachable_from_the_documents_it_governs():
    for path, label in (
        ("README.md", "the README"),
        (os.path.join("docs", "README.md"), "the documentation index"),
        ("CONTRIBUTING.md", "the contributor guide"),
        (os.path.join("docs", "ARCHITECTURE.md"), "the architecture page"),
    ):
        assert "STABILITY.md" in _read(path), f"{label} does not link the stability policy"


def test_the_policy_names_the_deprecation_channel():
    """stdout is the PreToolUse decision channel; a warning there is a protocol defect."""
    policy = _read("STABILITY.md")
    assert "stderr" in policy and "never stdout" in policy
