"""Tenant-specific literal destinations; all examples run without network access."""

import runpy
from pathlib import Path

import pytest

from examples.egress_allowlist import http_host
from examples.tenant_http_policy import make_tenant_http_policy
from recusal import decide

HOSTS = {"C1001": {"a.example.com"}, "C2002": {"b.example.com"}}


@pytest.mark.parametrize("tool", ["WebFetch", "http_get"])
@pytest.mark.parametrize("url", ["https://a.example.com/tickets", "https://A.EXAMPLE.COM./tickets"])
def test_active_tenant_host_defers(tool, url):
    policy = make_tenant_http_policy("C1001", HOSTS, http_tools=("WebFetch", "http_get"))
    assert decide(tool, {"url": url}, policy)[0] == "defer"


@pytest.mark.parametrize(
    "url",
    [
        "https://b.example.com/tickets",
        "https://child.a.example.com/",
        "https://a.example.com.evil.example/",
        "https://evil-a.example.com/",
        "https://b.example.com@a.example.com/",
        "https://user:password@a.example.com/",
        "https://evil.example\\@a.example.com/",
        "https:\\\\evil.example\\@a.example.com/",
        "https://127.0.0.1/",
        "https://[::1]/",
        "https://[invalid/",
        "https://2130706433/",
        "https://0177.0.0.1/",
        "https://0x7f000001/",
        "file://a.example.com/path",
        "a.example.com/path",
        "\x00https://a.example.com/",
        "https://a.example.com/\x1b",
        "https://a.example.com/with space",
        "",
        None,
        7,
    ],
)
def test_other_or_ambiguous_destinations_refuse(url):
    policy = make_tenant_http_policy("C1001", HOSTS)
    assert decide("WebFetch", {"url": url}, policy)[0] == "deny"


@pytest.mark.parametrize("active_id", [None, "", " ", 7, []])
def test_missing_or_malformed_tenant_refuses(active_id):
    policy = make_tenant_http_policy(active_id, HOSTS)
    findings = policy("WebFetch", {"url": "https://a.example.com/"})
    assert findings[0].check == "subject_unbound"
    assert decide("WebFetch", {"url": "https://a.example.com/"}, policy)[0] == "deny"


def test_unknown_tenant_and_missing_url_refuse():
    assert (
        decide(
            "WebFetch", {"url": "https://a.example.com/"}, make_tenant_http_policy("C3003", HOSTS)
        )[0]
        == "deny"
    )
    assert decide("WebFetch", {}, make_tenant_http_policy("C1001", HOSTS))[0] == "deny"


def test_tool_input_cannot_select_another_tenant():
    policy = make_tenant_http_policy("C1001", HOSTS)
    arguments = {"url": "https://b.example.com/", "active_id": "C2002", "tenant": "C2002"}
    assert decide("WebFetch", arguments, policy)[0] == "deny"


def test_tenant_configuration_is_snapshotted():
    hosts = {"C1001": {"a.example.com"}}
    policy = make_tenant_http_policy("C1001", hosts)
    hosts["C1001"].add("b.example.com")
    assert decide("WebFetch", {"url": "https://b.example.com/"}, policy)[0] == "deny"


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "2130706433", "0x7f000001", "0177.0.0.1"])
def test_configuring_an_ip_literal_does_not_allow_it(host):
    policy = make_tenant_http_policy("C1001", {"C1001": {host}})
    authority = "[::1]" if host == "::1" else host
    assert decide("WebFetch", {"url": "https://" + authority + "/"}, policy)[0] == "deny"


def test_unconfigured_tool_is_not_inspected_negative_control():
    policy = make_tenant_http_policy(None, HOSTS)
    assert policy("Bash", {"command": "python send.py"}) == []
    assert policy("http_post", {"url": "https://b.example.com/"}) == []


def test_shared_parser_preserves_recipe_five_behavior():
    assert http_host("https://API.EXAMPLE.COM.:8443/path") == "api.example.com"
    assert http_host("https://user@example.com/") == ""
    assert http_host("https://evil.example\\@example.com/") == ""


def test_runnable_demo_states_its_boundary(capsys):
    example = Path(__file__).resolve().parents[1] / "examples" / "tenant_http_policy.py"
    runpy.run_path(str(example), run_name="__main__")
    output = capsys.readouterr().out
    assert "active tenant: defer" in output
    assert "other tenant: deny" in output
    assert "userinfo trick: deny" in output
    assert "NOT INSPECTED" in output
