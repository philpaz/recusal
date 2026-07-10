"""
MCP catalog collection: a minimal, standard-library stdio ``tools/list`` client.

This is the one place in the package that spawns a subprocess and a thread. It is kept in
its own module, apart from the pure kernel (:mod:`recusal.mcp`), on purpose: the decision
path never touches a process. Collection is nondeterministic I/O; adjudication is a pure
function of what was collected. Everything here only *observes* a server's declared
catalog, ``recusal.mcp`` (and the CLI) decide what that observation means.

The contract that keeps the seam clean: every irregularity, a timeout, an early exit, a
JSON-RPC error, an unparseable or non-UTF-8 line, a missing binary, raises
:class:`McpFetchError`. A failed observation must be an *error*, never an empty
(clean-looking) catalog that a verify could misread as "no drift".

Servers reachable only over HTTP are not fetched here; :func:`servers_from_claude_config`
surfaces them as ``skipped`` so the caller pins them from a JSON dump (``recusal mcp pin
--from``) rather than silently dropping them. No model, no network in any verdict.
"""

import json
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__

#: Upper bound on tools a single stdio server may declare before the fetch refuses. A
#: hostile or broken server otherwise amplifies unbounded memory/CPU into ``build_manifest``
#: (which canonicalizes every declaration). A real catalog is dozens of tools, not thousands.
MAX_TOOLS = 5000


class McpFetchError(RuntimeError):
    """The catalog could not be observed; callers must fail closed, never assume clean."""


def split_command(text: str) -> List[str]:
    """Split a ``--stdio`` command string into argv, Windows paths included.

    POSIX ``shlex`` treats a backslash as an escape, which silently corrupts
    ``py C:\\Users\\me\\server.py`` into ``C:Usersmeserver.py``, a server that "exits
    before answering". On Windows, split in non-POSIX mode and strip the quotes shlex
    leaves on quoted tokens; everywhere else, standard shell rules.
    """
    if os.name != "nt":
        return shlex.split(text)
    argv = shlex.split(text, posix=False)
    return [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" else a for a in argv]


def _resolve_command(command: Sequence[str]) -> List[str]:
    """Resolve argv[0] on PATH (handles Windows .cmd/.exe shims like ``npx``)."""
    argv = [str(a) for a in command]
    if not argv or not argv[0]:
        raise McpFetchError("empty server command")
    resolved = shutil.which(argv[0])
    if resolved is None and not os.path.exists(argv[0]):
        raise McpFetchError(f"server command not found on PATH: {argv[0]!r}")
    return [resolved or argv[0]] + argv[1:]


def fetch_tools_stdio(
    command: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: float = 30.0,
) -> List[dict]:
    """Spawn a stdio MCP server, run ``initialize`` → ``tools/list``, return the tools.

    Newline-delimited JSON-RPC 2.0 over the child's stdin/stdout; stderr is discarded
    (servers log there). Messages that are not the awaited response (notifications,
    unrelated ids) are skipped. Any irregularity, timeout, EOF, a JSON-RPC error, an
    unparseable line, raises :class:`McpFetchError`: an observation that cannot be
    completed must be an error, never an empty (clean-looking) catalog.
    """
    argv = _resolve_command(command)
    merged_env = dict(os.environ)
    merged_env.update(env or {})
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=merged_env,
            cwd=cwd,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        raise McpFetchError(f"could not start MCP server {argv[0]!r}: {exc}") from exc

    # The queue carries a line (str), an EOF sentinel (None), or the exception the reader
    # hit, so a decode/read failure surfaces as a TRUTHFUL McpFetchError instead of the
    # main thread misreporting "server exited" while Python dumps the thread's traceback.
    lines: "queue.Queue[Any]" = queue.Queue()

    def _reader() -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                lines.put(line)
        except (UnicodeError, OSError, ValueError) as exc:
            lines.put(exc)  # invalid UTF-8 (JSON must be UTF-8) or a broken pipe
        finally:
            lines.put(None)  # EOF sentinel

    threading.Thread(target=_reader, daemon=True).start()
    next_id = 0

    def _send(payload: Dict[str, Any]) -> None:
        try:
            proc.stdin.write(json.dumps(payload) + "\n")  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]
        except OSError as exc:
            raise McpFetchError(f"MCP server closed its stdin: {exc}") from exc

    def _request(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal next_id
        next_id += 1
        rid = next_id
        _send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpFetchError(f"timed out after {timeout}s waiting for {method!r}")
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                raise McpFetchError(f"timed out after {timeout}s waiting for {method!r}") from None
            if isinstance(line, BaseException):
                raise McpFetchError(
                    f"could not read from the MCP server (invalid UTF-8 or I/O error) "
                    f"while waiting for {method!r}: {line}"
                ) from line
            if line is None:
                raise McpFetchError(f"MCP server exited before answering {method!r}")
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError as exc:
                raise McpFetchError(f"unparseable line from MCP server: {line[:200]!r}") from exc
            if not isinstance(message, dict) or message.get("id") != rid:
                continue  # a notification or an unrelated message; not ours
            if "error" in message:
                raise McpFetchError(
                    f"MCP server returned an error for {method!r}: {message['error']!r}"
                )
            result = message.get("result")
            if not isinstance(result, dict):
                raise McpFetchError(f"MCP server returned no result object for {method!r}")
            return result

    try:
        _request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "recusal", "version": __version__},
            },
        )
        _send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools: List[dict] = []
        cursor: Optional[str] = None
        for _ in range(100):  # pagination backstop; a longer catalog is an error, not a loop
            params: Dict[str, Any] = {"cursor": cursor} if cursor else {}
            result = _request("tools/list", params)
            page = result.get("tools")
            if not isinstance(page, list):
                raise McpFetchError("tools/list result has no 'tools' array")
            tools.extend(t for t in page if isinstance(t, dict))
            if len(tools) > MAX_TOOLS:  # bound memory/CPU a hostile server could amplify
                raise McpFetchError(
                    f"MCP server declared more than {MAX_TOOLS} tools; refusing a runaway catalog"
                )
            cursor = result.get("nextCursor")
            if not cursor:
                return tools
        raise McpFetchError("tools/list paginated past 100 pages; refusing a runaway catalog")
    finally:
        try:
            proc.stdin.close()  # type: ignore[union-attr]
        except OSError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()  # reap: a killed child must not linger as a zombie on POSIX


def servers_from_claude_config(path: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Read stdio servers from a Claude Code ``.mcp.json``.

    Returns ``(servers, skipped)``: ``servers`` maps a server name to
    ``{"command": [...], "env": {...}}`` ready for :func:`fetch_tools_stdio`; ``skipped``
    names servers this fetcher cannot reach (URL-based transports), which must be pinned
    from a JSON dump instead so they are surfaced, never silently dropped.
    Raises ``ValueError`` on a config that does not parse or has no servers.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path} has no mcpServers")
    servers: Dict[str, Dict[str, Any]] = {}
    skipped: List[str] = []
    for name, entry in entries.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("command"), str):
            skipped.append(str(name))
            continue
        args = entry.get("args") or []
        env = entry.get("env") or {}
        if not isinstance(args, list) or not isinstance(env, dict):
            skipped.append(str(name))
            continue
        servers[str(name)] = {
            "command": [entry["command"]] + [str(a) for a in args],
            "env": {str(k): str(v) for k, v in env.items()},
        }
    return servers, skipped
