"""Cookbook recipe 14 runs with the real MCP SDK in both CI lanes, never only skips.

The SDK is not a Recusal dependency, so the main test job always skips recipe 14. Two
jobs install it: ci.yml at a pinned version on every pull request, compat.yml at the
newest release each week. Both set RECUSAL_REQUIRE_MCP_SDK so a missing or broken SDK
fails instead of skipping. This test fails if either job disappears, stops requiring the
SDK, or pins a version older than the one the cookbook promises.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
JOB = "cookbook-mcp-sdk"
TEST = "tests/test_cookbook_blocks.py"
REQUIRE = 'RECUSAL_REQUIRE_MCP_SDK: "1"'


def _job(workflow):
    text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([\w-]+):\n", body, flags=re.M)
    jobs = dict(zip(parts[1::2], parts[2::2]))
    assert JOB in jobs, f"{workflow}: no {JOB} job"
    return jobs[JOB]


def _version(text):
    return tuple(int(part) for part in text.split("."))


def test_the_pinned_lane_runs_recipe_14_on_every_pull_request():
    job = _job("ci.yml")
    install = re.search(r'pip install -e "\.\[dev\]" mcp==([\d.]+)$', job, re.M)
    assert install, "ci.yml: the MCP SDK must be installed at an exact version"
    assert REQUIRE in job, "ci.yml: a skipped recipe 14 must fail this job"
    assert f"pytest -q {TEST}" in job


def test_the_pin_meets_the_version_the_cookbook_promises():
    cookbook = (ROOT / "docs" / "COOKBOOK.md").read_text(encoding="utf-8")
    promised = re.search(r'pip install "mcp>=([\d.]+)"', cookbook)
    assert promised, "recipe 14 no longer names its minimum SDK version"
    pinned = re.search(r"mcp==([\d.]+)", _job("ci.yml"))
    assert pinned
    assert _version(pinned.group(1)) >= _version(promised.group(1))


def test_the_latest_lane_runs_recipe_14_on_the_newest_sdk():
    job = _job("compat.yml")
    assert re.search(r'pip install -e "\.\[dev\]" --upgrade mcp$', job, re.M)
    assert "mcp==" not in job, "compat.yml: the latest lane must not pin"
    assert REQUIRE in job, "compat.yml: a skipped recipe 14 must fail this job"
    assert f"pytest -q {TEST}" in job


def test_the_cookbook_test_honours_the_requirement():
    source = (ROOT / TEST).read_text(encoding="utf-8")
    assert 'os.environ.get("RECUSAL_REQUIRE_MCP_SDK") == "1"' in source
