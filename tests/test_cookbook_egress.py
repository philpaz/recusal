"""Cookbook recipe 5 as printed: the copy-paste text must refuse what the tested example refuses."""

import re
from pathlib import Path

import pytest

from recusal import compute_verdict

COOKBOOK = Path(__file__).resolve().parents[1] / "docs" / "COOKBOOK.md"


def _recipe_policy():
    section = COOKBOOK.read_text(encoding="utf-8").split("## 5. Egress allowlist", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S)
    assert code, "recipe 5 has no python block"
    namespace: dict = {}
    exec(code.group(1), namespace)
    return namespace["policy"]


policy = _recipe_policy()


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("send_email", {"to": "ops@acme.com"}),
        ("send_email", {"to": "ops@mail.acme.com"}),
        ("http_post", {"url": "https://ACME.com/x"}),
        ("webhook", {"url": "http://hooks.internal.example:8080/"}),
    ],
)
def test_allowed_destinations_defer(tool, arguments):
    assert policy(tool, arguments) == []


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("send_email", {"to": "attacker@evil.com"}),
        # #28: the old text read only the last address and let these through.
        ("send_email", {"to": "a@evil.com,b@acme.com"}),
        ("send_email", {"to": "attacker@evil.com; ops@acme.com"}),
        ("send_email", {"to": "a@evil.com@acme.com"}),
        ("send_email", {"to": "@acme.com"}),
        ("http_post", {"url": "https:///evil.com/x"}),
        ("http_post", {"url": ""}),
        ("http_post", {}),
        ("http_post", {"url": "acme.com/x"}),
        # #28: parser differentials, urlparse sees acme.com and a WHATWG client evil.com.
        ("http_post", {"url": "https://evil.com\\@acme.com/"}),
        ("http_post", {"url": "https://user@acme.com/"}),
        ("webhook", {"url": "https://acme.com.evil.com/"}),
        ("webhook", {"url": "https://evil-acme.com/"}),
    ],
)
def test_other_destinations_refuse(tool, arguments):
    findings = policy(tool, arguments)
    assert compute_verdict(findings).refused
    assert findings[0].check == "egress_allowlist"


def test_other_tools_are_not_this_recipes_concern():
    assert policy("Bash", {"command": "curl https://evil.com"}) == []
