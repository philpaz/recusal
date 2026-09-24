"""Recipe 5: inspect literal email/HTTP destinations (offline).

Run: python examples/egress_allowlist.py

This is a teaching policy, not a network firewall. It sees only the explicit
arguments of send_email, http_post and webhook. It cannot see addresses built
inside a shell/interpreter, redirects, DNS changes or a tool's internal requests.
An empty finding list defers; it does not certify that no data can leave. Use
transport-level controls or a vetted default-deny tool policy for that boundary.
Email inputs here are a single bare mailbox, not lists or display-name syntax.
URLs with a backslash or userinfo (user@host) are refused: parsers disagree on
their host, so the host this policy checks may not be the one a client contacts.
"""

import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding


def make_egress_allowlist(allowed_domains):
    """Bind exact hosts and their dot-delimited subdomains to a local policy."""
    domains = frozenset(domain.lower().rstrip(".") for domain in allowed_domains)

    def policy(tool_name, tool_input):
        if tool_name not in ("send_email", "http_post", "webhook"):
            return []
        key = "to" if tool_name == "send_email" else "url"
        value = tool_input.get(key)
        host = ""
        if isinstance(value, str) and value and not any(c.isspace() for c in value):
            if key == "to":
                if value.count("@") == 1 and not any(c in value for c in ",;<>"):
                    local, host = value.split("@")
                    if not local:
                        host = ""
            elif "\\" not in value:
                # A backslash or userinfo is where URL parsers disagree: urlparse reads
                # "https://evil.com\@example.com/" as example.com, a browser as evil.com.
                try:
                    parsed = urlparse(value)
                    if parsed.scheme in ("http", "https") and "@" not in parsed.netloc:
                        host = parsed.hostname or ""
                except ValueError:
                    pass
        host = host.lower().rstrip(".")
        if host and any(host == domain or host.endswith("." + domain) for domain in domains):
            return []
        return [
            Finding.fail(
                "egress_allowlist",
                severity="CRITICAL",
                message="destination is missing, malformed, or outside the egress allowlist",
            )
        ]

    return policy


def main():
    policy = make_egress_allowlist({"example.com"})
    for label, tool, arguments in (
        ("allowed email", "send_email", {"to": "person@example.com"}),
        ("allowed subdomain", "http_post", {"url": "https://api.example.com/events"}),
        ("lookalike host", "webhook", {"url": "https://evil-example.com/events"}),
    ):
        print(f"{label}: {'DENY' if policy(tool, arguments) else 'DEFER'}")
    # This command is never executed. The recipe sees neither its eventual URL nor traffic.
    hidden = {"command": "python construct_and_send.py"}
    assert policy("Bash", hidden) == []
    print("Runtime-built URL: NOT INSPECTED; DEFER is not an egress guarantee.")


if __name__ == "__main__":
    main()
