"""Recipe 23: bind literal HTTP destinations to a trusted active tenant.

Run: python examples/tenant_http_policy.py

This teaching policy checks named tools' explicit URL arguments. Redirects, DNS
resolution, URL paths, ports and requests made inside other tools need transport-
or application-level controls. A clean finding list defers to host permissions.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.egress_allowlist import http_host
from recusal import Finding, decide

_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_NUMERIC_END = re.compile(r"(?:[0-9]+|0x[0-9a-f]+)")


def _dns_host(value):
    """Accept ASCII DNS names, excluding IP literals and numeric-address spellings."""
    if not isinstance(value, str):
        return ""
    host = value.lower().rstrip(".")
    labels = host.split(".")
    if len(host) > 253 or not all(_LABEL.fullmatch(label) for label in labels):
        return ""
    if _NUMERIC_END.fullmatch(labels[-1]):
        return ""
    return host


def make_tenant_http_policy(active_id, hosts_by_tenant, http_tools=("WebFetch",)):
    """Bind the host-owned subject and an exact-host snapshot for one session or turn."""
    bound = isinstance(active_id, str) and bool(active_id.strip())
    hosts = hosts_by_tenant.get(active_id, ()) if bound else ()
    allowed = frozenset(filter(None, (_dns_host(host) for host in hosts)))
    tools = frozenset(http_tools)

    def policy(tool_name, tool_input):
        if tool_name not in tools:
            return []
        if not bound:
            return [
                Finding.fail(
                    "subject_unbound",
                    severity="CRITICAL",
                    message="no active tenant is bound; HTTP request refused",
                )
            ]
        value = tool_input.get("url")
        # Refuse C0/DEL before urlparse: its treatment of leading controls changed
        # between Python versions. The tenant policy must decide identically.
        host = ""
        if isinstance(value, str) and not any(ord(c) < 32 or ord(c) == 127 for c in value):
            host = _dns_host(http_host(value))
        if host and host in allowed:
            return []
        return [
            Finding.fail(
                "tenant_http_destination",
                severity="CRITICAL",
                message="HTTP destination is missing, malformed, or outside the active tenant",
            )
        ]

    return policy


def main():
    hosts = {"C1001": {"a.example.com"}, "C2002": {"b.example.com"}}
    policy = make_tenant_http_policy("C1001", hosts)
    for label, url in (
        ("active tenant", "https://a.example.com/tickets"),
        ("other tenant", "https://b.example.com/tickets"),
        ("userinfo trick", "https://b.example.com@a.example.com/tickets"),
    ):
        decision, _ = decide("WebFetch", {"url": url}, policy)
        print(f"{label}: {decision}")
    print("Redirects, DNS changes and requests inside other tools are NOT INSPECTED.")


if __name__ == "__main__":
    main()
