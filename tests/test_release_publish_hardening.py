"""Drift lock for release publish hardening (review 9, item 6b).

pypa/gh-action-pypi-publish hard-fails the upload when the version already exists
on PyPI unless ``skip-existing`` suppresses it. The suppression was removed on
purpose: a release re-run that would republish an existing version must fail
loudly so a divergent rebuild can never silently no-op over the published
artifacts. This lock keeps the suppression from drifting back in.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _release_workflow():
    path = os.path.join(ROOT, ".github", "workflows", "release.yml")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _effective_lines(text):
    """Workflow lines with comments stripped; prose may name a key, YAML may not."""
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def test_publish_step_is_the_expected_action():
    assert "pypa/gh-action-pypi-publish@" in _release_workflow(), (
        "the publish mechanism changed; re-review the fail-on-existing-version "
        "behavior this lock exists to preserve, then update both together"
    )


def test_publish_never_suppresses_an_existing_version():
    offending = [line for line in _effective_lines(_release_workflow()) if "skip-existing" in line]
    assert offending == [], (
        f"{offending!r}: skip-existing suppresses the upload failure when the "
        "version already exists on PyPI; a silent no-op re-run can mask a "
        "divergent rebuild, so republishing must hard-fail for a human to look at"
    )
