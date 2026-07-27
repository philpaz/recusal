"""Release metadata and workflow invariants introduced for the 0.7.1 patch."""

import os
import re

import recusal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def test_citation_version_matches_the_package_and_has_a_release_date():
    citation = _read("CITATION.cff")
    version = re.search(r'^version:\s*"([^"]+)"\s*$', citation, re.MULTILINE)
    release_date = re.search(r'^date-released:\s*"\d{4}-\d{2}-\d{2}"\s*$', citation, re.MULTILINE)
    assert version, "CITATION.cff must declare a quoted release version"
    assert version.group(1) == recusal.__version__
    assert release_date, "CITATION.cff must declare an ISO release date"


def test_ci_and_release_assert_the_python39_compatible_mypy_line():
    for path in (
        (".github", "workflows", "ci.yml"),
        (".github", "workflows", "release.yml"),
    ):
        workflow = _read(*path)
        assert "major < 2" in workflow, (
            f"{'/'.join(path)} does not enforce the mypy <2 ceiling needed to "
            "type-check the declared Python 3.9 floor"
        )


def test_published_release_is_the_only_automatic_release_trigger():
    release = _read(".github", "workflows", "release.yml")
    trigger = release.split("\npermissions:", 1)[0]
    assert "release:" in trigger
    assert "workflow_dispatch:" in trigger
    assert re.search(r"^\s+push:\s*$", trigger, re.MULTILINE) is None, (
        "tag push plus release publication produces two builds and two provenance "
        "attestations; the published release must remain the sole automatic trigger"
    )


def test_signing_claim_names_every_release_artifact_class():
    security = _read("SECURITY.md")
    for claim in ("built sdist and wheel", "tag `.tar.gz` and `.zip` source archives"):
        assert claim in security
    release = _read(".github", "workflows", "release.yml")
    assert "release-signing-artifacts: true" in release
