"""The stdio fetcher against a real (fake) MCP server subprocess.

The fetcher's contract: it completes ``initialize`` → ``notifications/initialized`` →
paginated ``tools/list`` against a well-behaved server (surviving stderr noise and
interleaved notifications), and EVERY irregularity, timeout, early exit, a JSON-RPC
error, an unparseable line, a missing binary, raises ``McpFetchError`` so the caller
fails closed. A failed observation must never read as an empty (clean) catalog.
"""

import json
import os
import sys

import pytest

from recusal.mcp_fetch import (
    McpFetchError,
    fetch_tools_stdio,
    servers_from_claude_config,
    split_command,
)

# A minimal newline-delimited JSON-RPC MCP server; the MODE argv selects a behavior.
FAKE_SERVER = r"""
import json, sys, time

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"
TOOLS = [
    {"name": "create_issue", "description": "Create an issue.", "inputSchema": {"type": "object"}},
    {"name": "read_file", "description": "Read a file.", "inputSchema": {"type": "object"}},
]

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method, rid = msg.get("method"), msg.get("id")
    print("fake-server log noise", file=sys.stderr)  # stderr must never confuse the client
    if method == "initialize":
        if MODE == "hang":
            time.sleep(60)
        if MODE == "exit-early":
            sys.exit(0)
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info"}})
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "0"}}
        if MODE == "newest-proto":
            result["protocolVersion"] = "2025-11-25"
        elif MODE == "old-proto":
            result["protocolVersion"] = "2024-11-05"
        elif MODE == "bad-proto":
            result["protocolVersion"] = "1999-01-01"
        elif MODE == "no-proto":
            del result["protocolVersion"]
        elif MODE == "no-caps":
            del result["capabilities"]
        elif MODE == "no-tools-cap":
            result["capabilities"] = {"prompts": {}}
        elif MODE == "no-serverinfo":
            del result["serverInfo"]
        if MODE == "bad-jsonrpc":
            sys.stdout.write(json.dumps({"id": rid, "result": result}) + "\n")
            sys.stdout.flush()
        else:
            send({"jsonrpc": "2.0", "id": rid, "result": result})
    elif method == "tools/list":
        if MODE == "error":
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -1, "message": "boom"}})
        elif MODE == "badjson":
            sys.stdout.write("this is not json\n"); sys.stdout.flush()
        elif MODE == "flood":
            for _ in range(1500):  # > MAX_UNRELATED_MESSAGES
                send({"jsonrpc": "2.0", "method": "notifications/message", "params": {}})
            send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        elif MODE == "non-object-tool":
            send({"jsonrpc": "2.0", "id": rid,
                  "result": {"tools": [TOOLS[0], "malformed-entry"]}})
        elif MODE == "nan":
            sys.stdout.write('{"jsonrpc": "2.0", "id": %d, "result": {"tools": [], "n": NaN}}\n'
                             % rid); sys.stdout.flush()
        elif MODE == "deep":
            deep = "[" * 200000 + "]" * 200000
            sys.stdout.write('{"jsonrpc": "2.0", "id": %d, "result": {"tools": %s}}\n'
                             % (rid, deep)); sys.stdout.flush()
        elif MODE == "bad-cursor":
            send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [TOOLS[0]],
                  "nextCursor": 42}})
        elif MODE == "paginate":
            cursor = (msg.get("params") or {}).get("cursor")
            if cursor:
                send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [TOOLS[1]]}})
            else:
                send({"jsonrpc": "2.0", "id": rid,
                      "result": {"tools": [TOOLS[0]], "nextCursor": "page2"}})
        else:
            send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
"""


@pytest.fixture()
def fake_server(tmp_path):
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")

    def _command(mode="normal"):
        return [sys.executable, str(script), mode]

    return _command


def test_happy_path_survives_notifications_and_stderr_noise(fake_server):
    tools = fetch_tools_stdio(fake_server("normal"), timeout=30)
    assert [t["name"] for t in tools] == ["create_issue", "read_file"]


def test_pagination_is_followed_to_the_end(fake_server):
    tools = fetch_tools_stdio(fake_server("paginate"), timeout=30)
    assert [t["name"] for t in tools] == ["create_issue", "read_file"]


def test_a_jsonrpc_error_raises_never_returns_partial(fake_server):
    with pytest.raises(McpFetchError, match="boom"):
        fetch_tools_stdio(fake_server("error"), timeout=30)


def test_an_unparseable_line_raises(fake_server):
    with pytest.raises(McpFetchError, match="unparseable"):
        fetch_tools_stdio(fake_server("badjson"), timeout=30)


def test_a_hung_server_times_out(fake_server):
    with pytest.raises(McpFetchError, match="timed out"):
        fetch_tools_stdio(fake_server("hang"), timeout=1.5)


def test_a_server_that_exits_early_raises(fake_server):
    with pytest.raises(McpFetchError, match="exited"):
        fetch_tools_stdio(fake_server("exit-early"), timeout=30)


def test_invalid_utf8_from_the_server_raises_a_truthful_error(tmp_path):
    # C6: raw non-UTF-8 bytes must surface as "could not read ... invalid UTF-8", NOT the
    # untruthful "server exited" (the server has not exited), and must not dump a thread
    # traceback. A tiny server that writes a lone 0xFF byte to its stdout buffer.
    script = tmp_path / "bad_utf8_server.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'\\xff\\xff\\xff\\n')\n"
        "sys.stdout.buffer.flush()\n"
        "import time; time.sleep(5)\n",  # stay alive so "exited" would be the WRONG diagnosis
        encoding="utf-8",
    )
    with pytest.raises(McpFetchError, match="invalid UTF-8|could not read"):
        fetch_tools_stdio([sys.executable, str(script)], timeout=10)


def test_a_missing_binary_raises():
    with pytest.raises(McpFetchError, match="not found"):
        fetch_tools_stdio(["definitely-not-a-real-binary-abc123"])


def test_an_empty_command_raises():
    with pytest.raises(McpFetchError, match="empty"):
        fetch_tools_stdio([])


# --- the --stdio command splitter ---------------------------------------------------------


def test_split_command_keeps_quoted_arguments_together():
    assert split_command('py server.py "two words"') == ["py", "server.py", "two words"]


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_split_command_does_not_eat_windows_backslashes():
    # POSIX shlex corrupts C:\Users\me\server.py into C:Usersmeserver.py, a server that
    # "exits before answering"; the splitter must hand paths through intact.
    argv = split_command(r'py C:\Users\me\server.py --flag "a b"')
    assert argv == ["py", r"C:\Users\me\server.py", "--flag", "a b"]


def test_split_command_end_to_end_reaches_a_real_server(fake_server, tmp_path):
    # The exact seam `--stdio NAME COMMAND` uses: a command STRING with a native path.
    command = " ".join(part if " " not in part else f'"{part}"' for part in fake_server("normal"))
    tools = fetch_tools_stdio(split_command(command), timeout=30)
    assert [t["name"] for t in tools] == ["create_issue", "read_file"]


# --- the .mcp.json reader -----------------------------------------------------------------


def test_claude_config_yields_stdio_servers_and_surfaces_the_rest(tmp_path):
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "server-github"],
                        "env": {"TOKEN": "x"},
                    },
                    "hosted": {"type": "http", "url": "https://example.com/mcp"},
                }
            }
        ),
        encoding="utf-8",
    )
    servers, remote = servers_from_claude_config(str(config))
    github = servers["github"]
    assert github["command"] == ["npx", "-y", "server-github"]
    assert github["env"]["TOKEN"] == "x"
    # Claude Code parity: the project root rides along to the spawned server
    assert github["env"]["CLAUDE_PROJECT_DIR"] == os.path.dirname(os.path.abspath(config))
    # the UNEXPANDED launch template, ready to pin and to diff BEFORE a launch
    assert github["source"] == {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "server-github"],
        "cwd": None,
        "env_templates": {"TOKEN": "x"},
    }
    # the remote server is a first-class IDENTITY, never a silently dropped name
    assert remote == {
        "hosted": {
            "transport": "http",
            "url_template": "https://example.com/mcp",
            "header_keys": [],
        }
    }


def test_claude_config_with_no_servers_raises(tmp_path):
    config = tmp_path / ".mcp.json"
    for body in ("{}", '{"mcpServers": {}}', "[]"):
        config.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError):
            servers_from_claude_config(str(config))


def test_claude_config_expands_vars_with_claude_semantics(tmp_path):
    # ${VAR} and ${VAR:-default} in command/args/env values, per current Claude Code
    # docs; the SOURCE template stays unexpanded (that is what gets pinned).
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "banking": {
                        "command": "${MCP_BIN}",
                        "args": ["--project", "${CLAUDE_PROJECT_DIR:-.}"],
                        "env": {"API_KEY": "${BANK_API_KEY}"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    servers, _ = servers_from_claude_config(
        str(config), base_env={"MCP_BIN": "/opt/mcp/bin/banking", "BANK_API_KEY": "sk-123"}
    )
    banking = servers["banking"]
    assert banking["command"] == ["/opt/mcp/bin/banking", "--project", "."]
    assert banking["env"]["API_KEY"] == "sk-123"
    assert banking["source"]["command"] == "${MCP_BIN}"  # template, not the expansion
    # the TEMPLATE is pinned: the reference is recorded, the resolved value never is
    assert banking["source"]["env_templates"] == {"API_KEY": "${BANK_API_KEY}"}


def test_claude_config_fails_closed_on_an_unset_variable(tmp_path):
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps({"mcpServers": {"s": {"command": "${NOPE_UNSET_VAR}"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="NOPE_UNSET_VAR"):
        servers_from_claude_config(str(config), base_env={})


def test_claude_config_rejects_non_string_types_instead_of_coercing(tmp_path):
    # A non-string arg or env value silently str()-ed could launch a command Claude Code
    # would have refused or run differently; types are rejected, not coerced.
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps({"mcpServers": {"s": {"command": "x", "args": ["ok", 42]}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="list of strings"):
        servers_from_claude_config(str(config))
    config.write_text(
        json.dumps({"mcpServers": {"s": {"command": "x", "env": {"N": 1}}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strings to strings"):
        servers_from_claude_config(str(config))


# --- environment handed to a not-yet-trusted server ---------------------------------------

# A server that declares, as its one tool's description, the value of an env var: exactly
# what a catalog-collection subprocess could exfiltrate if the parent env rides along.
ENV_ECHO_SERVER = r"""
import json, os, sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method, rid = msg.get("method"), msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2025-06-18",
             "capabilities": {"tools": {}}, "serverInfo": {"name": "env-echo", "version": "0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [
            {"name": "t", "description": os.environ.get("RECUSAL_TEST_SECRET", "ABSENT"),
             "inputSchema": {"type": "object"}}]}})
"""


@pytest.fixture()
def env_echo_server(tmp_path):
    script = tmp_path / "env_echo_server.py"
    script.write_text(ENV_ECHO_SERVER, encoding="utf-8")
    return [sys.executable, str(script)]


def test_default_env_inherits_the_parent_shell(env_echo_server, monkeypatch):
    # Documented default: full inheritance, matching how Claude Code launches the server.
    monkeypatch.setenv("RECUSAL_TEST_SECRET", "hunter2")
    tools = fetch_tools_stdio(env_echo_server, timeout=30)
    assert tools[0]["description"] == "hunter2"


def test_minimal_env_withholds_parent_secrets(env_echo_server, monkeypatch):
    # A server being pinned is not yet trusted: with minimal_env=True an API key in the
    # shell must NOT ride along to it.
    monkeypatch.setenv("RECUSAL_TEST_SECRET", "hunter2")
    tools = fetch_tools_stdio(env_echo_server, timeout=30, minimal_env=True)
    assert tools[0]["description"] == "ABSENT"


def test_minimal_env_still_passes_explicit_env(env_echo_server, monkeypatch):
    # Deliberately handed vars (a server's own config env) are not withheld.
    monkeypatch.setenv("RECUSAL_TEST_SECRET", "hunter2")
    tools = fetch_tools_stdio(
        env_echo_server, timeout=30, minimal_env=True, env={"RECUSAL_TEST_SECRET": "given"}
    )
    assert tools[0]["description"] == "given"


# --- a single endless line must refuse, not buffer until the timeout ----------------------


def test_a_runaway_single_line_refuses(tmp_path, monkeypatch):
    import recusal.mcp_fetch as fetch_mod

    monkeypatch.setattr(fetch_mod, "MAX_LINE_CHARS", 1000)
    script = tmp_path / "endless_line_server.py"
    script.write_text(
        "import sys, time\n"
        "sys.stdout.write('a' * 5000)\n"  # no newline: would buffer until timeout
        "sys.stdout.flush()\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    with pytest.raises(McpFetchError, match="runaway stream"):
        fetch_tools_stdio([sys.executable, str(script)], timeout=30)


# --- the initialize negotiation is a binding compatibility decision -----------------------


def test_supported_negotiated_versions_are_accepted(fake_server):
    # The server may answer with any revision it supports; every revision this client
    # speaks must be accepted (newest, the requested-era one, and the oldest).
    for mode in ("newest-proto", "normal", "old-proto"):
        tools = fetch_tools_stdio(fake_server(mode), timeout=30)
        assert [t["name"] for t in tools] == ["create_issue", "read_file"]


def test_an_unsupported_negotiated_version_refuses(fake_server):
    with pytest.raises(McpFetchError, match="unsupported protocol version.*1999-01-01"):
        fetch_tools_stdio(fake_server("bad-proto"), timeout=30)


def test_a_missing_protocol_version_refuses(fake_server):
    with pytest.raises(McpFetchError, match="unsupported protocol version"):
        fetch_tools_stdio(fake_server("no-proto"), timeout=30)


def test_missing_capabilities_refuse(fake_server):
    with pytest.raises(McpFetchError, match="no capabilities object"):
        fetch_tools_stdio(fake_server("no-caps"), timeout=30)


def test_a_server_without_the_tools_capability_refuses(fake_server):
    with pytest.raises(McpFetchError, match="did not advertise the 'tools' capability"):
        fetch_tools_stdio(fake_server("no-tools-cap"), timeout=30)


# --- aggregate bounds and wire strictness ---------------------------------------------------


def test_a_notification_flood_refuses(fake_server):
    with pytest.raises(McpFetchError, match="message flood"):
        fetch_tools_stdio(fake_server("flood"), timeout=60)


def test_a_non_object_tool_declaration_refuses_the_whole_catalog(fake_server):
    # Fail-closed collection: silently filtering the malformed entry would certify a
    # SUBSET as if it were the declared surface.
    with pytest.raises(McpFetchError, match="non-object tool declaration"):
        fetch_tools_stdio(fake_server("non-object-tool"), timeout=30)


def test_nonstandard_json_constants_are_refused(fake_server):
    with pytest.raises(McpFetchError, match="unparseable"):
        fetch_tools_stdio(fake_server("nan"), timeout=30)


def test_hostile_nesting_depth_on_the_wire_is_a_refusal_not_a_crash(fake_server):
    # The property under test: hostile depth yields a NAMED McpFetchError, never a raw
    # RecursionError escaping the error contract. WHICH named refusal fires depends on
    # the platform's parse depth (3.12+ on Linux/macOS parses far deeper before the
    # C-stack guard trips): past the parser's limit it is the depth wrap; within it, the
    # nested arrays reach the catalog checks and the non-object tool refusal fires.
    with pytest.raises(McpFetchError, match="nested beyond parseable depth|non-object tool"):
        fetch_tools_stdio(fake_server("deep"), timeout=30)


def test_an_invalid_next_cursor_refuses(fake_server):
    with pytest.raises(McpFetchError, match="invalid nextCursor"):
        fetch_tools_stdio(fake_server("bad-cursor"), timeout=30)


def test_the_aggregate_character_budget_refuses(tmp_path, monkeypatch):
    import recusal.mcp_fetch as fetch_mod

    monkeypatch.setattr(fetch_mod, "MAX_TOTAL_CHARS", 5000)
    script = tmp_path / "chatty_server.py"
    script.write_text(
        "import json, sys, time\n"
        "line = json.dumps({'jsonrpc': '2.0', 'method': 'noise', 'params': {'pad': 'x' * 100}})\n"
        "for _ in range(200):\n"
        "    sys.stdout.write(line + chr(10))\n"
        "sys.stdout.flush()\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    with pytest.raises(McpFetchError, match="characters in one observation"):
        fetch_tools_stdio([sys.executable, str(script)], timeout=30)


def test_a_response_without_serverinfo_refuses(fake_server):
    with pytest.raises(McpFetchError, match="serverInfo"):
        fetch_tools_stdio(fake_server("no-serverinfo"), timeout=30)


def test_a_response_without_the_jsonrpc_envelope_refuses(fake_server):
    with pytest.raises(McpFetchError, match="jsonrpc 2.0 envelope"):
        fetch_tools_stdio(fake_server("bad-jsonrpc"), timeout=30)


def test_the_observation_deadline_is_shared_across_the_whole_exchange(fake_server):
    # Per-request timeouts alone let a server answering just inside each deadline
    # stretch initialize plus 100 pages into about 101x the timeout; the observation
    # deadline is one monotonic budget across all of it.
    with pytest.raises(McpFetchError, match="observation exceeded"):
        fetch_tools_stdio(fake_server("hang"), timeout=30, observation_timeout=1.5)


def test_the_reader_queue_stays_tiny():
    # Drift lock for the peak-memory bound: with the budget enforced by the READER
    # before enqueue, peak buffered memory is a few lines; a big queue would quietly
    # turn "bounded" into gigabytes.
    import recusal.mcp_fetch as fetch_mod

    assert fetch_mod._QUEUE_MAXSIZE <= 4
