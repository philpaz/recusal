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


def test_distributions_are_sigstore_signed_and_publish_fails_closed_behind_it():
    # the 0.7.0 signing posture (decision 2026-07-23): signatures live at the release
    # boundary in CI, the runtime verifies nothing. The sign job is the ONE job
    # allowed contents: write (to attach bundles to the release), and publish must
    # not run when signing failed - an unsignable distribution is a human-review
    # signal, not a shippable artifact.
    release = _read("release.yml")
    assert "sigstore/gh-action-sigstore-python@" in release, (
        "the Sigstore signing step left the release workflow"
    )
    assert "release-signing-artifacts: true" in release, (
        "signature bundles must be attached to the release so a verifier can fetch them"
    )
    assert re.search(r"needs:\s*\[build,\s*sign\]", release), (
        "publish must fail closed behind signing (needs: [build, sign])"
    )


def test_every_action_in_every_workflow_is_sha_pinned():
    for name in _workflow_files():
        for line in _read(name).splitlines():
            m = re.search(r"uses:\s*(\S+)", line)
            if m and not m.group(1).startswith(("./", "$/")):
                assert re.search(r"@[0-9a-f]{40}(\s|$)", m.group(1)), (
                    f"{name}: {m.group(1)!r} is not pinned to a full commit SHA; a "
                    "moving tag is a rug-pull surface (a same-repo ./ or $/ reference "
                    "runs at the caller's own commit and needs no pin)"
                )


# --- the provenance bundle is attached to the release as a file ---------------------------
#
# The attestation is stored by GitHub. The same signed bundle is also attached to the
# GitHub Release as ``<distribution>-<version>.intoto.jsonl``, so a verifier has the
# provenance beside the artifacts and can check it with ``gh attestation verify --bundle``.


def _settings(workflow):
    """The workflow with its comment lines removed: what it sets, not what it says."""
    return "\n".join(line for line in workflow.splitlines() if not line.lstrip().startswith("#"))


PIPELINES = (
    ("build-dist.yml", "release.yml", "verify-release.yml"),
    ("build-adapter-dist.yml", "release-adapter.yml", "verify-adapter-release.yml"),
)


def test_the_builder_keeps_its_provenance_bundle_as_a_separate_artifact():
    for build_name, _, _ in PIPELINES:
        build = _read(build_name)
        assert "        id: attest\n        uses: actions/attest-build-provenance@" in build
        assert "BUNDLE: ${{ steps.attest.outputs.bundle-path }}" in build, build_name
        # named for the one sdist, and refused if there is not exactly one to name it by
        assert 'cp "$BUNDLE" "provenance/$(basename "$1" .tar.gz).intoto.jsonl"' in build
        assert 'if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then' in build, build_name
        assert 'test -s "$BUNDLE"' in build, build_name
        # its own artifact: dist/ is what gets signed and published to PyPI
        upload = build.split("- name: Upload the provenance bundle", 1)[1].split("- name:", 1)[0]
        assert "name: provenance" in upload and "path: provenance/" in upload, build_name
        assert "if-no-files-found: error" in upload, build_name
        dist_upload = build.split("- name: Upload build artifacts", 1)[1]
        assert "path: dist/" in dist_upload and "provenance" not in dist_upload, build_name
        # the build job still cannot write to the release
        assert "contents: write" not in _settings(build), build_name


def test_the_release_job_attaches_the_bundle_and_nothing_else_gains_the_permission():
    for _, release_name, _ in PIPELINES:
        release = _read(release_name)
        sign = release.split("\n  sign:\n", 1)[1].split("\n  publish:\n", 1)[0]
        assert "name: provenance" in sign and "path: provenance/" in sign, release_name
        assert (
            'run: gh release upload "$TAG" provenance/*.intoto.jsonl --repo "$GITHUB_REPOSITORY"'
            in sign
        ), release_name
        assert "TAG: ${{ github.event.release.tag_name }}" in sign, release_name
        # attached after signing, so a release that could not be signed gets no bundle
        assert sign.index("release-signing-artifacts: true") < sign.index("gh release upload")
        # one job may write release assets, and it is this one
        assert _settings(release).count("contents: write") == 1, release_name
        assert "contents: write" in _settings(sign), release_name
        # a rehearsal has no release to attach to: the job runs for a release only
        assert "if: github.event_name == 'release'" in sign, release_name


def test_the_verifier_checks_the_attached_bundle_against_the_reusable_builder():
    for build_name, _, verify_name in PIPELINES:
        verify = _read(verify_name)
        step = verify.split(
            "- name: The provenance bundle attached to the release verifies the same files", 1
        )[1].split("- name:", 1)[0]
        assert '--bundle "$BUNDLE"' in step, verify_name
        assert f'--signer-workflow "${{REPO}}/.github/workflows/{build_name}"' in step, verify_name
        assert ".intoto.jsonl" in step, verify_name
        # an older release has no such asset; the API check before this step covers it
        assert 'if [ ! -f "$BUNDLE" ]; then' in step and "exit 0" in step, verify_name
        before = verify.split("- name: The provenance bundle attached", 1)[0]
        assert "gh attestation verify" in before, verify_name


def test_the_verification_guide_describes_the_attached_bundle_and_no_longer_predates_the_adapters():
    with open(os.path.join(ROOT, "docs", "VERIFY.md"), encoding="utf-8") as fh:
        guide = fh.read()
    assert "--bundle recusal-<version>.intoto.jsonl" in guide
    assert "recusal_langgraph-X.Y.Z.intoto.jsonl" in guide
    assert "None has been released yet" not in guide
    assert "All eight are on PyPI." in guide
    adapters = [
        name
        for name in os.listdir(os.path.join(ROOT, "adapters"))
        if os.path.isfile(os.path.join(ROOT, "adapters", name, "pyproject.toml"))
    ]
    assert len(adapters) == 8
