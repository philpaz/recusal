"""Drift locks for the adopter-facing verification procedure (docs/VERIFY.md).

The verification commands published to adopters name specific workflows: build
provenance must name the reusable BUILDER (``build-dist.yml``), and the Sigstore
certificate identity must name the workflow that SIGNS (``release.yml``). Those are
different workflows answering different questions, and a rename or a pipeline
reshuffle would silently turn the published commands into instructions that either
fail for everyone or, worse, pass while checking something weaker.

These tests pin the documentation against the pipeline it describes.
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")
VERIFY_DOC = os.path.join(ROOT, "docs", "VERIFY.md")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _doc():
    return _read(VERIFY_DOC)


def _workflow(name):
    return _read(os.path.join(WORKFLOWS, name))


def test_documented_signer_workflow_is_the_real_reusable_builder():
    """The --signer-workflow path in the docs must be a workflow that actually builds."""
    doc = _doc()
    signer = re.findall(r"--signer-workflow\s+(\S+)", doc)
    assert signer, "docs/VERIFY.md no longer shows a --signer-workflow verification"
    for path in signer:
        if "not-the-builder" in path:  # the negative control, deliberately nonexistent
            assert not os.path.exists(os.path.join(WORKFLOWS, "not-the-builder.yml"))
            continue
        assert path.startswith("philpaz/recusal/.github/workflows/"), path
        name = path.rsplit("/", 1)[-1]
        assert os.path.exists(os.path.join(WORKFLOWS, name)), (
            f"docs/VERIFY.md names {name} as the signer workflow; no such workflow exists"
        )
        build = _workflow(name)
        assert "actions/attest-build-provenance@" in build, (
            f"{name} is documented as the provenance signer but does not attest anything"
        )


def test_documented_certificate_identity_is_the_workflow_that_signs():
    """The Sigstore cert identity must name the workflow holding the signing step."""
    doc = _doc()
    identities = re.findall(r"--cert-identity \"https://github\.com/\S+/([\w.-]+)@", doc)
    assert identities, "docs/VERIFY.md no longer shows a --cert-identity verification"
    signing_workflow = "release.yml"
    assert "sigstore/gh-action-sigstore-python@" in _workflow(signing_workflow), (
        "the signing step moved out of release.yml; docs/VERIFY.md names it as the signer"
    )
    assert signing_workflow in identities, (
        "the documented certificate identity no longer names the signing workflow"
    )
    # The negative control must name a DIFFERENT workflow, or it proves nothing.
    assert any(name != signing_workflow for name in identities), (
        "docs/VERIFY.md lost its certificate-identity negative control"
    )


def test_the_documented_procedure_is_executable():
    """A documented command nobody runs is a claim; verify-release.yml runs them."""
    workflow = _workflow("verify-release.yml")
    assert "--signer-workflow" in workflow
    assert "sigstore verify github" in workflow
    assert "integrity/recusal" in workflow  # the PEP 740 provenance check
    assert workflow.count("NEGATIVE CONTROL") >= 2, (
        "both negative controls must run, or a passing verification proves nothing"
    )
    assert "actions/checkout@" not in workflow, (
        "verification must work from the published artifacts alone, without the checkout"
    )
    assert VERIFY_DOC.endswith("VERIFY.md") and "verify-release.yml" in _doc(), (
        "docs/VERIFY.md must point at the workflow that executes it"
    )


def test_the_verifier_client_is_pinned_exactly():
    """An unpinned verifier defines 'verified' by whatever released last (the ruff lesson)."""
    assert re.search(r'pip install "sigstore==\d+\.\d+\.\d+"', _workflow("verify-release.yml")), (
        "the Sigstore client must be pinned exactly in the verification workflow"
    )


def test_the_docs_state_what_verification_does_not_establish():
    """Provenance binds an artifact to a build; it does not read the code."""
    doc = _doc()
    assert "Not byte-reproducibility" in doc
    assert "does not read the code" in doc
    assert "workflow run\nartifact" in doc or "workflow run artifact" in doc, (
        "the SBOM's retention boundary must stay stated, not quietly dropped"
    )


def test_the_readme_and_index_route_people_to_it():
    assert "docs/VERIFY.md" in _read(os.path.join(ROOT, "README.md")), (
        "the README must link the verification procedure"
    )
    assert "VERIFY.md" in _read(os.path.join(ROOT, "docs", "README.md")), (
        "the documentation index must link the verification procedure"
    )
