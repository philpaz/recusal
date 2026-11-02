"""The examples index must link to every Python module it publishes."""

import re
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_examples_readme_links_every_python_module():
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(([^()/]+\.py)\)", readme))
    present = {path.name for path in EXAMPLES.glob("*.py")}

    assert present == linked, (
        f"Missing example links: {sorted(present - linked)}; "
        f"stale links: {sorted(linked - present)}"
    )
