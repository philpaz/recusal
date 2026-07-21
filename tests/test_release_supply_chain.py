"""Drift locks for the release supply-chain steps (ledger item 3, 0.6.0 scope).

The build job must generate an SBOM for the built distributions and attest their
build provenance, with the minimum permissions those attestations need. Every
action in the release workflow stays pinned to an immutable commit SHA; a moving
tag is a rug-pull surface.
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _release_workflow():
    path = os.path.join(ROOT, ".github", "workflows", "release.yml")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_build_generates_an_sbom_and_attests_provenance():
    text = _release_workflow()
    assert "anchore/sbom-action@" in text, "the SBOM step left the release workflow"
    assert "actions/attest-build-provenance@" in text, (
        "the provenance attestation step left the release workflow"
    )
    assert "attestations: write" in text, (
        "attest-build-provenance cannot write to the attestation store without "
        "the attestations: write permission"
    )


def test_every_release_action_is_sha_pinned():
    for line in _release_workflow().splitlines():
        m = re.search(r"uses:\s*(\S+)", line)
        if m:
            assert re.search(r"@[0-9a-f]{40}(\s|$)", m.group(1)), (
                f"{m.group(1)!r} is not pinned to a full commit SHA; a moving "
                "tag is a rug-pull surface"
            )
