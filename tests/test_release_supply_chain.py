"""Drift locks for the release supply-chain steps (ledger item 3, 0.6.0 scope;
reusable-workflow delegation, 0.7.0 scope).

The release build must generate an SBOM for the built distributions and attest their
build provenance, with the minimum permissions those attestations need - and since
0.7.0 those steps live in the reusable ``build-dist.yml`` workflow that ``release.yml``
delegates to, the pattern GitHub documents for SLSA v1 Build Level 3 (the provenance
records the reusable workflow's identity as the signer, so a verifier can require it
with ``--signer-workflow``). Every action in every workflow stays pinned to an
immutable commit SHA; a moving tag is a rug-pull surface.
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


def test_build_generates_an_sbom_and_attests_provenance():
    build = _read("build-dist.yml")
    assert "anchore/sbom-action@" in build, "the SBOM step left the release build workflow"
    assert "actions/attest-build-provenance@" in build, (
        "the provenance attestation step left the release build workflow"
    )
    assert "attestations: write" in build, (
        "attest-build-provenance cannot write to the attestation store without "
        "the attestations: write permission"
    )


def test_release_delegates_the_build_to_the_reusable_workflow():
    # the SLSA L3 pattern is the delegation itself: provenance signed by
    # build-dist.yml's identity, not by an inline job the caller could reshape;
    # an inline `steps:` build drifting back into release.yml would silently
    # change the signer identity every published verification command names
    release = _read("release.yml")
    assert "uses: ./.github/workflows/build-dist.yml" in release, (
        "release.yml no longer delegates the build to the reusable build-dist.yml"
    )
    assert "actions/attest-build-provenance@" not in release, (
        "provenance attestation belongs in build-dist.yml (the signer identity a "
        "verifier pins with --signer-workflow), not inline in release.yml"
    )
    build = _read("build-dist.yml")
    assert "workflow_call" in build, "build-dist.yml must remain a reusable workflow"


def test_every_action_in_every_workflow_is_sha_pinned():
    for name in _workflow_files():
        for line in _read(name).splitlines():
            m = re.search(r"uses:\s*(\S+)", line)
            if m and not m.group(1).startswith("./"):
                assert re.search(r"@[0-9a-f]{40}(\s|$)", m.group(1)), (
                    f"{name}: {m.group(1)!r} is not pinned to a full commit SHA; a "
                    "moving tag is a rug-pull surface (a same-repo ./ reusable "
                    "workflow runs at the caller's own commit and needs no pin)"
                )
