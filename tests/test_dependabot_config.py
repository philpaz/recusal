"""Dependabot watches every kind of dependency the repository pins.

The workflows pin GitHub Actions by commit SHA and the release build pins its toolchain by
hash; a pin nobody updates goes stale silently. This test fails if either ecosystem drops
out of .github/dependabot.yml.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")


def _ecosystems():
    return set(re.findall(r"^\s+- package-ecosystem: ([\w-]+)$", CONFIG, re.M))


def test_both_pinned_ecosystems_are_updated():
    assert _ecosystems() == {"github-actions", "pip"}


def test_what_dependabot_watches_is_really_pinned_here():
    # the pins it keeps current exist: SHA-pinned actions and a hash-locked release env
    workflows = "".join(
        p.read_text(encoding="utf-8") for p in (ROOT / ".github" / "workflows").glob("*.yml")
    )
    assert re.search(r"uses: [\w./-]+@[0-9a-f]{40}", workflows)
    assert "--hash=sha256:" in (ROOT / "release-requirements.txt").read_text(encoding="utf-8")


def test_updates_run_weekly_and_wait_a_week_for_new_releases():
    assert CONFIG.count("interval: weekly") == len(_ecosystems())
    # a release pulled or compromised in its first days never reaches a pull request
    assert CONFIG.count("default-days: 7") == len(_ecosystems())
