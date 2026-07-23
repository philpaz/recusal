"""Drift locks for the hash-locked release toolchain (review 7, P2-1).

The lock is only a control while it stays consistent with the build configuration it
claims to pin: pyproject's [build-system] backend and the lock must name the SAME
hatchling, every requirement line must carry hashes (one unhashed line and
--require-hashes refuses the whole install, so an unhashed line is a broken lock, not
a partial one), and the workflows must actually install through the lock.
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _lock_pins():
    pins = {}
    for line in _read("release-requirements.txt").splitlines():
        m = re.match(r"^([a-z0-9-]+)==([^ \\]+)", line)
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def test_lock_pins_the_same_hatchling_as_pyproject():
    m = re.search(r'requires\s*=\s*\["hatchling==([^"]+)"\]', _read("pyproject.toml"))
    assert m, "pyproject [build-system] must pin hatchling exactly"
    assert _lock_pins().get("hatchling") == m.group(1), (
        "release-requirements.txt and pyproject pin different hatchling versions; "
        "the --no-isolation release build would not use the reviewed backend"
    )


def test_lock_contains_the_toolchain_roots_and_only_hashed_lines():
    pins = _lock_pins()
    for root in ("build", "twine", "hatchling"):
        assert root in pins, f"lock is missing toolchain root {root!r}"
    lines = _read("release-requirements.txt").splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^[a-z0-9-]+==", line):
            assert line.endswith("\\") and "--hash=sha256:" in lines[i + 1], (
                f"{line!r} has no hash continuation; one unhashed line breaks "
                "--require-hashes for the whole install"
            )


def test_workflows_install_through_the_lock():
    # the release build steps live in the reusable build-dist.yml (SLSA L3 pattern);
    # release.yml delegates to it, so the lock requirement follows the steps
    build = _read(".github", "workflows", "build-dist.yml")
    ci = _read(".github", "workflows", "ci.yml")
    for text, name in ((build, "build-dist.yml"), (ci, "ci.yml")):
        assert "--require-hashes -r release-requirements.txt" in text, (
            f"{name} does not install the release toolchain through the hash lock"
        )
        assert "build --no-isolation" in text, (
            f"{name} builds with an uncontrolled PEP 517 isolation environment"
        )
    assert "pip install build==" not in build and "pip install twine==" not in build, (
        "build-dist.yml still installs an unlocked toolchain alongside the lock"
    )
