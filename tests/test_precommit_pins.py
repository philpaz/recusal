"""The pre-commit hooks run the same ruff and mypy lines as the dev extras.

Dependabot bumps the ruff pin in pyproject.toml but does not touch .pre-commit-config.yaml;
a hook on a different ruff formats differently from CI, so a local commit passes and the
pull request fails (or the reverse).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
PRECOMMIT = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")


def _hook_rev(repo):
    m = re.search(re.escape(repo) + r"\n\s+rev: v([\w.]+)", PRECOMMIT)
    assert m, f"{repo} has no rev in .pre-commit-config.yaml"
    return m.group(1)


def test_ruff_hook_matches_the_exact_dev_pin():
    pin = re.search(r'"ruff==([\w.]+)"', PYPROJECT)
    assert pin, "pyproject dev extras must pin ruff exactly"
    assert _hook_rev("astral-sh/ruff-pre-commit") == pin.group(1)


def test_mypy_hook_stays_inside_the_dev_range():
    # mypy 2 cannot check the declared Python 3.9 floor; the dev extras cap it below 2
    assert '"mypy>=1.8,<2"' in PYPROJECT
    assert _hook_rev("pre-commit/mirrors-mypy").split(".")[0] == "1"
