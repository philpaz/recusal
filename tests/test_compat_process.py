"""Every tested runtime is kept in both lanes, with its surface probe.

CONTRIBUTING.md ("Adding a runtime") and STABILITY.md ("Integrations") describe one
process for every agent framework Recusal supports: exact versions on every pull request
(ci.yml), the newest release every week (compat.yml), and a surface probe that pins what
the gate can see. This test reads the two workflows and fails if a runtime has joined one
lane without the other, if the lanes install different packages, or if its probe is
missing, so the process cannot erode one integration at a time.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CI = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
COMPAT = (WORKFLOWS / "compat.yml").read_text(encoding="utf-8")


def _jobs(workflow: str):
    """Top-level job bodies keyed by job id."""
    body = workflow.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([\w-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def _slug(runtime: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", runtime.lower()).strip("_")


def pinned_lane():
    """{runtime slug: (set of pinned package names, set of test files run)} from ci.yml."""
    lane = {}
    for body in _jobs(CI).values():
        name = re.search(r"^    name: (.+) example \(Python [\d.]+\)$", body, re.M)
        if not name:
            continue
        install = re.search(r'pip install -e "\.\[dev\]" (.+)$', body, re.M)
        assert install, f"{name.group(1)}: no install line"
        pins = install.group(1).split()
        assert all("==" in p for p in pins), f"{name.group(1)}: every package must be pinned"
        tests = set(re.findall(r"tests/test_\w+\.py", body))
        slug = _slug(name.group(1))
        packages = {p.split("==")[0] for p in pins}
        previous = lane.setdefault(slug, (packages, tests))
        assert previous == (packages, tests), f"{slug}: its pinned jobs disagree"
    return lane


def latest_lane():
    """{runtime slug: (set of package names, set of test files run)} from compat.yml."""
    lane = {}
    entries = re.findall(
        r"- runtime: (\S+)\n\s+python-version: \"[\d.]+\"\n"
        r"\s+packages: \"([^\"]+)\"\n\s+tests: \"([^\"]+)\"",
        COMPAT,
    )
    for runtime, packages, tests in entries:
        assert "==" not in packages, f"{runtime}: the latest lane must not pin"
        value = (set(packages.split()), set(tests.split()))
        previous = lane.setdefault(runtime, value)
        assert previous == value, f"{runtime}: its latest-lane entries disagree"
    return lane


def test_the_parsers_find_the_known_runtime():
    # guards the parsers: an empty parse must not pass by accident
    assert "langgraph" in pinned_lane()
    assert "langgraph" in latest_lane()


def test_every_pinned_runtime_is_in_the_latest_lane_and_back():
    assert set(pinned_lane()) == set(latest_lane())


def test_both_lanes_install_the_same_packages_and_run_the_same_tests():
    latest = latest_lane()
    for runtime, (packages, tests) in pinned_lane().items():
        assert latest[runtime][0] == packages, f"{runtime}: the lanes install different packages"
        assert latest[runtime][1] == tests, f"{runtime}: the lanes run different tests"


def test_every_runtime_has_its_example_tests_and_surface_probe_in_both_lanes():
    for runtime, (_, tests) in pinned_lane().items():
        probe = f"tests/test_compat_{runtime}.py"
        assert (ROOT / probe).is_file(), f"{runtime}: missing {probe}"
        assert probe in tests, f"{runtime}: the lanes do not run {probe}"
        example_tests = tests - {probe}
        assert example_tests, f"{runtime}: the lanes run no example tests"
        for path in example_tests:
            assert (ROOT / path).is_file(), f"{runtime}: missing {path}"


def test_the_latest_lane_never_gates_a_pull_request():
    triggers = COMPAT.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    assert "schedule:" in triggers
    assert "workflow_dispatch:" in triggers
    assert "pull_request" not in triggers
    assert "push:" not in triggers


def test_the_process_is_written_down_where_contributors_look():
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    stability = (ROOT / "STABILITY.md").read_text(encoding="utf-8")
    support = (ROOT / "SUPPORT.md").read_text(encoding="utf-8")
    assert "\n## Adding a runtime\n" in contributing
    assert "tests/test_compat_process.py" in contributing
    assert "\n## Integrations\n" in stability
    assert "compat.yml" in stability
    assert "STABILITY.md#integrations" in support
