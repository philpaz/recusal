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

Remote HTTP, SSE, and WebSocket servers are not contacted by this stdio-only
collector; :func:`servers_from_claude_config`
surfaces them as ``skipped`` so the caller pins them from a JSON dump (``recusal mcp pin
--from``) rather than silently dropping them. No model, no network in any verdict.

**The configuration is executable code.** Fetching a stdio catalog EXECUTES the command
the config declares - there is no way to ask a process for ``tools/list`` without running
it. The layer above pins each launch specification and compares it BEFORE calling this
fetcher (``recusal.mcp.diff_source``); the FIRST pin is the explicit trust event
(``--approve-server-launch``). Review ``command``/``args`` before that first observation,
protect ``.mcp.json`` as a control-plane file, and pass ``minimal_env=True`` (the CLI
default) so a server still being decided about does not inherit secrets.
"""

import hashlib
import json
import os
import queue
import re
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

#: MCP protocol revisions this client speaks, newest first; the newest is what we request.
#: The handshake and ``tools/list`` (with cursor pagination) are identical across these.
#: Negotiation per the spec: the server answers with a version of its choosing, and a
#: client that does not support the answered version must disconnect - here, refuse.
SUPPORTED_PROTOCOL_VERSIONS: Tuple[str, ...] = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)

#: Upper bound on a single stdout line from the server. Without it, a server emitting one
#: endless line with no newline buffers unboundedly until the timeout; with it, the fetch
#: refuses. Generous on purpose: a legitimate ``tools/list`` page is one line, and even a
#: full MAX_TOOLS catalog of oversized declarations fits well inside it.
MAX_LINE_CHARS = 10_000_000

#: Aggregate budget across the whole observation: total characters read from the server
#: and total messages that were not the awaited response (notifications, unrelated ids).
#: The per-line cap bounds one line; these bound what a chatty or hostile server can make
#: the observer buffer and churn through in total before it refuses.
MAX_TOTAL_CHARS = 50_000_000
MAX_UNRELATED_MESSAGES = 1_000

#: Reader-to-adjudicator queue depth. Deliberately TINY: the aggregate budget is
#: enforced by the READER before enqueue, and a small queue keeps the peak buffered
#: memory at a few lines (queue x line cap), not a multiple of the whole budget - "the
#: queue is technically bounded" must not stand in for "hostile input is bounded".
_QUEUE_MAXSIZE = 4

#: Environment variables a child process needs merely to *launch and run* - what
#: ``minimal_env=True`` keeps. Paths and locale only; nothing here carries a secret.
_LAUNCH_ENV: Tuple[str, ...] = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
)


class McpFetchError(RuntimeError):
    """The catalog could not be observed; callers must fail closed, never assume clean."""


def _reject_constant(name: str) -> Any:
    # Python's json accepts NaN/Infinity/-Infinity by default; JSON-RPC is strict JSON,
    # so a peer emitting them is out of contract and the line is refused as unparseable.
    raise ValueError(f"nonstandard JSON constant {name!r} is not valid JSON-RPC")


def _loads_strict(line: str) -> Any:
    """``json.loads`` for wire input: strict constants, and hostile nesting depth is an
    :class:`McpFetchError` (a crash escaping the error contract is not a refusal)."""
    try:
        return json.loads(line, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise McpFetchError(
            "MCP server sent JSON nested beyond parseable depth; refusing a hostile message"
        ) from exc


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


def resolve_executable_identity(command: Sequence[str]) -> Dict[str, str]:
    """The ``{path, sha256}`` identity of the file a command's argv[0] resolves to.

    Uses the SAME resolution the fetcher uses before spawning (:func:`_resolve_command`,
    ``shutil.which`` semantics including Windows ``.cmd``/``.exe`` shims), then hashes
    that file's bytes. This is the strict-mode pin/verify collection primitive: the
    identity is the FIRST process image only - a script or interpreter that argv[0]
    delegates to (``py server.py`` pins the launcher, not ``server.py``) stays inside
    the launch-template boundary, not this one. Raises :class:`McpFetchError` when the
    command cannot be resolved or the resolved file cannot be read; the caller decides
    whether that fails the pin outright or becomes the explicit ``None`` observation.
    """
    argv = _resolve_command(command)
    path = os.path.abspath(argv[0])
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise McpFetchError(f"cannot read resolved executable {path!r}: {exc}") from exc
    return {"path": path, "sha256": f"sha256:{digest.hexdigest()}"}


def fetch_server_stdio(
    command: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: float = 30.0,
    observation_timeout: float = 300.0,
    minimal_env: bool = False,
) -> Dict[str, Any]:
    """Spawn a stdio MCP server; return ``{"tools": [...], "instructions": str|None}``.

    The initialize-result ``instructions`` field is discovery-time model-facing content
    (under Claude's default tool-search behavior, server instructions load at session
    start), so it is observed alongside the tool catalog; a non-string value refuses. Everything else is
    :func:`fetch_tools_stdio`'s contract, which remains available for tools-only use.
    """
    return _fetch_stdio(
        command,
        env=env,
        cwd=cwd,
        timeout=timeout,
        observation_timeout=observation_timeout,
        minimal_env=minimal_env,
    )


def fetch_tools_stdio(
    command: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: float = 30.0,
    observation_timeout: float = 300.0,
    minimal_env: bool = False,
) -> List[dict]:
    """Tools-only convenience over :func:`fetch_server_stdio` (same contract)."""
    return _fetch_stdio(
        command,
        env=env,
        cwd=cwd,
        timeout=timeout,
        observation_timeout=observation_timeout,
        minimal_env=minimal_env,
    )["tools"]


def _fetch_stdio(
    command: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: float = 30.0,
    observation_timeout: float = 300.0,
    minimal_env: bool = False,
) -> Dict[str, Any]:
    """Spawn a stdio MCP server, run ``initialize`` → ``tools/list``, return both.

    Newline-delimited JSON-RPC 2.0 over the child's stdin/stdout; stderr is discarded
    (servers log there). Messages that are not the awaited response (notifications,
    unrelated ids) are skipped. Any irregularity, timeout, EOF, a JSON-RPC error, an
    unparseable line, raises :class:`McpFetchError`: an observation that cannot be
    completed must be an error, never an empty (clean-looking) catalog.

    By default the child inherits the full parent environment plus ``env`` (matching how
    Claude Code launches the same server). A server being *pinned* is by definition not
    yet trusted, so ``minimal_env=True`` hands it only what a process needs to launch
    (PATH and friends, see ``_LAUNCH_ENV``) plus the explicitly passed ``env`` - an API
    key in your shell does not ride along to a server you are still deciding about.
    """
    argv = _resolve_command(command)
    if minimal_env:
        merged_env = {k: os.environ[k] for k in _LAUNCH_ENV if k in os.environ}
    else:
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
    # Bounded: a flood of notifications backpressures the pipe instead of growing memory.
    lines: "queue.Queue[Any]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
    budget = {"chars": 0, "unrelated": 0}  # aggregate caps across the whole observation

    def _reader() -> None:
        try:
            while True:
                # bounded readline: one endless newline-less line must refuse, not buffer
                # until the timeout (readline with a limit returns at most that many chars)
                line = proc.stdout.readline(MAX_LINE_CHARS)  # type: ignore[union-attr]
                if not line:
                    break
                if len(line) >= MAX_LINE_CHARS and not line.endswith("\n"):
                    lines.put(
                        McpFetchError(
                            f"MCP server emitted a single line longer than {MAX_LINE_CHARS} "
                            "characters; refusing a runaway stream"
                        )
                    )
                    return
                # the READER is the budget's accountant, reserving BEFORE enqueue: with
                # the budget checked only on dequeue, a hostile server could park
                # queue x line-cap characters in memory before the limit ever fired
                budget["chars"] += len(line)
                if budget["chars"] > MAX_TOTAL_CHARS:
                    lines.put(
                        McpFetchError(
                            f"MCP server sent more than {MAX_TOTAL_CHARS} characters in "
                            "one observation; refusing a runaway catalog"
                        )
                    )
                    return
                lines.put(line)
        except (UnicodeError, OSError, ValueError) as exc:
            lines.put(exc)  # invalid UTF-8 (JSON must be UTF-8) or a broken pipe
        finally:
            lines.put(None)  # EOF sentinel

    threading.Thread(target=_reader, daemon=True).start()
    next_id = 0
    # one monotonic deadline for the WHOLE observation: per-request timeouts alone let a
    # server that answers just inside each deadline stretch initialize + up to 100 pages
    # into ~101x the configured timeout
    observation_deadline = time.monotonic() + observation_timeout

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
            if time.monotonic() >= observation_deadline:
                raise McpFetchError(
                    f"observation exceeded {observation_timeout}s across "
                    "initialize/pagination; refusing a stalling server"
                )
            remaining = min(deadline, observation_deadline) - time.monotonic()
            if remaining <= 0:
                raise McpFetchError(f"timed out after {timeout}s waiting for {method!r}")
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                if time.monotonic() >= observation_deadline:
                    raise McpFetchError(
                        f"observation exceeded {observation_timeout}s across "
                        "initialize/pagination; refusing a stalling server"
                    ) from None
                raise McpFetchError(f"timed out after {timeout}s waiting for {method!r}") from None
            if isinstance(line, McpFetchError):
                raise line  # the reader already named the refusal (e.g. runaway line)
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
                message = _loads_strict(line)
            except ValueError as exc:
                raise McpFetchError(f"unparseable line from MCP server: {line[:200]!r}") from exc
            if (
                isinstance(message, dict)
                and message.get("id") == rid
                and (message.get("jsonrpc") != "2.0")
            ):
                raise McpFetchError(
                    f"MCP server response for {method!r} is missing the jsonrpc 2.0 "
                    "envelope; refusing an out-of-contract peer"
                )
            if not isinstance(message, dict) or message.get("id") != rid:
                # a notification or an unrelated message; not ours - but a flood of them
                # is a resource attack on the observer, so they are budgeted, not free
                budget["unrelated"] += 1
                if budget["unrelated"] > MAX_UNRELATED_MESSAGES:
                    raise McpFetchError(
                        f"MCP server sent more than {MAX_UNRELATED_MESSAGES} messages that "
                        f"were not the awaited response; refusing a message flood"
                    )
                continue
            if "error" in message:
                raise McpFetchError(
                    f"MCP server returned an error for {method!r}: {message['error']!r}"
                )
            result = message.get("result")
            if not isinstance(result, dict):
                raise McpFetchError(f"MCP server returned no result object for {method!r}")
            return result

    try:
        init = _request(
            "initialize",
            {
                "protocolVersion": SUPPORTED_PROTOCOL_VERSIONS[0],
                "capabilities": {},
                "clientInfo": {"name": "recusal", "version": __version__},
            },
        )
        # The negotiated response is a binding compatibility decision, not decoration:
        # an unsupported or missing version, or a server that does not advertise the
        # tools capability, is a refusal - proceeding would adjudicate a catalog
        # obtained under a contract this client never agreed to.
        negotiated = init.get("protocolVersion")
        if negotiated not in SUPPORTED_PROTOCOL_VERSIONS:
            raise McpFetchError(
                f"MCP server negotiated an unsupported protocol version: {negotiated!r} "
                f"(supported: {', '.join(SUPPORTED_PROTOCOL_VERSIONS)})"
            )
        server_info = init.get("serverInfo")
        if (
            not isinstance(server_info, dict)
            or not isinstance(server_info.get("name"), str)
            or not server_info["name"]
        ):
            raise McpFetchError(
                "MCP server returned no serverInfo object with a name on initialize; the "
                "lifecycle requires implementation info, and an anonymous peer is refused"
            )
        capabilities = init.get("capabilities")
        if not isinstance(capabilities, dict):
            raise McpFetchError("MCP server returned no capabilities object on initialize")
        if not isinstance(capabilities.get("tools"), dict):
            raise McpFetchError(
                "MCP server did not advertise the 'tools' capability (as an object); "
                "there is no tool catalog to observe"
            )
        instructions = init.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise McpFetchError(
                "MCP server returned non-string initialize instructions; refusing a "
                "malformed declaration"
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
            for item in page:
                if not isinstance(item, dict):
                    # fail-closed collection: silently filtering a malformed entry would
                    # certify a SUBSET as if it were the declared surface
                    raise McpFetchError(
                        f"tools/list returned a non-object tool declaration "
                        f"({type(item).__name__}); refusing the catalog"
                    )
            tools.extend(page)
            if len(tools) > MAX_TOOLS:  # bound memory/CPU a hostile server could amplify
                raise McpFetchError(
                    f"MCP server declared more than {MAX_TOOLS} tools; refusing a runaway catalog"
                )
            cursor = result.get("nextCursor")
            if not cursor:
                return {"tools": tools, "instructions": instructions}
            if not isinstance(cursor, str) or len(cursor) > 10_000:
                raise McpFetchError("tools/list returned an invalid nextCursor; refusing")
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


#: Claude Code's documented expansion syntax in ``.mcp.json``: ``${VAR}`` and
#: ``${VAR:-default}``, in ``command``, ``args``, and ``env`` values.
_EXPAND_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand(text: str, env: Dict[str, str], *, where: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` with Claude Code's semantics, failing
    CLOSED on a referenced variable that is unset and has no default: launching a
    partially-expanded command would observe a different server than the live session."""

    def _sub(match: "re.Match[str]") -> str:
        name, default = match.group(1), match.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        raise ValueError(
            f"{where}: ${{{name}}} is not set and has no default; refusing to launch a "
            "partially-expanded command"
        )

    return _EXPAND_RE.sub(_sub, text)


#: Remote transport ``type`` values Claude Code configures; "streamable-http" is the
#: same transport "http" names, so both pin as ``http`` (a rename must not read as drift).
_REMOTE_TYPES = {"http": "http", "streamable-http": "http", "sse": "sse", "ws": "ws"}

#: Per-entry field classification, mirroring Claude Code's documented config surface.
#: IDENTITY fields participate in the pinned source. ``timeout`` and ``alwaysLoad`` are
#: known Claude operational and context-loading fields: shape-validated below, but
#: deliberately excluded from source-template identity, so changing them is not source
#: drift. (``alwaysLoad`` changes WHEN tool definitions enter model context, not what
#: the pin certifies: recusal pins declaration content, not Claude's loading strategy;
#: a tool-level ``anthropic/alwaysLoad`` inside a declaration's ``_meta`` IS part of
#: the declaration fingerprint.) Anything else FAILS CLOSED - a field this parser
#: cannot classify could be executable configuration it would otherwise silently drop
#: (exactly how ``headersHelper`` was once invisible to verification).
_STDIO_ENTRY_FIELDS = {"type", "command", "args", "env", "cwd"}

#: Server names Claude Code reserves for its built-in servers (documented list,
#: case-sensitive). Claude SKIPS a user config entry using one of these at load time
#: and warns; `claude mcp add` rejects them. Representing (or LAUNCHING) such an entry
#: here would misdescribe the effective configuration surface, so it refuses instead -
#: before any command executes. Mirrors current Claude behavior; maintained as Claude
#: adds reserved names.
_CLAUDE_RESERVED_MCP_NAMES = frozenset(
    {"workspace", "claude-in-chrome", "computer-use", "Claude Preview", "Claude Browser"}
)
_REMOTE_ENTRY_FIELDS = {"type", "url", "headers", "headersHelper", "oauth"}
_RUNTIME_ONLY_FIELDS = {"timeout", "alwaysLoad"}
_OAUTH_ENTRY_FIELDS = {"clientId", "callbackPort", "authServerMetadataUrl", "scopes"}


def _required(entry: Dict[str, Any], key: str, default: Any) -> Any:
    # presence-checked, never `or`-coerced: a falsey INVALID value ("" / 0 / []) must
    # reach the type check and be refused, not silently normalized to the default
    return entry[key] if key in entry else default


def _validate_runtime_fields(entry: Dict[str, Any], where: str) -> None:
    """Validate the KNOWN runtime-only fields' shapes, even though they are not identity.

    ``timeout`` is Claude's per-server tool-execution timeout in milliseconds; values
    below 1000 are ignored by Claude, so accepting one here would silently misrepresent
    the configuration. ``alwaysLoad`` is a context-loading policy: it loads the
    server's full tool definitions at session start instead of deferring them to tool
    search, so it changes what enters model context, not merely execution tuning.
    Malformed values are refused, never treated as faithfully represented config.
    """
    if "timeout" in entry:
        value = entry["timeout"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1000:
            raise ValueError(
                f"{where} 'timeout' must be an integer >= 1000 milliseconds (Claude "
                f"ignores smaller values), got {value!r}"
            )
    if "alwaysLoad" in entry and not isinstance(entry["alwaysLoad"], bool):
        raise ValueError(f"{where} 'alwaysLoad' must be a boolean, got {entry['alwaysLoad']!r}")


def servers_from_claude_config(
    path: str, *, base_env: Optional[Dict[str, str]] = None
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Read every server from a Claude Code ``.mcp.json``, classified by TRANSPORT TYPE.

    Returns ``(stdio_servers, remote_servers)``. Each stdio server maps to::

        {
          "source":  {...},   # the UNEXPANDED launch template (recusal.mcp.normalize_source
                              #  shape, env_templates included) - what gets pinned and
                              #  compared BEFORE any launch
          "command": [...],   # resolved argv: ${VAR} / ${VAR:-default} expanded
          "env":     {...},   # resolved env for the child, CLAUDE_PROJECT_DIR included
          "cwd":     None|str,
        }

    Each remote server maps to its identity source per ``recusal.mcp.normalize_source``:
    ``{transport, url_template, header_templates, headers_helper_template}`` plus, for
    ``http``/``sse`` only, ``oauth`` (Claude applies preconfigured OAuth flags to HTTP
    and SSE transports; WebSocket is header-only and a ws entry carrying ``oauth``
    refuses). This fetcher never contacts remote servers - supply the catalog via
    ``--from`` (the rich ``{"instructions": ..., "tools": [...]}`` shape carries the
    discovery-content surface too).

    Classification mirrors Claude Code's rules and fails CLOSED on anything else: no
    ``type`` with a string ``command`` is stdio; ``type: "stdio"`` requires a command;
    ``http``/``streamable-http``/``sse``/``ws`` are remote and require a ``url`` (and a
    remote entry also carrying stdio launch fields is a contradiction, refused - the
    observer must never execute a command merely because a malformed remote entry
    contains one); a URL with no ``type`` is invalid; an unsupported ``type`` is
    invalid; a malformed entry raises, never a silent skip (a config the verifier
    cannot fully represent must not verify clean).

    Fidelity notes, matched to current Claude Code behavior: ``${VAR}`` and
    ``${VAR:-default}`` expand in ``command``, ``args``, and ``env`` values (a referenced
    variable that is unset with no default fails CLOSED); ``CLAUDE_PROJECT_DIR`` is set
    in the child environment to the config file's directory, the same value Claude Code
    injects, unless the config sets it explicitly. Types are rejected, not coerced -
    including falsey garbage (``"args": ""`` is an error, not an empty list).
    """
    env_base = dict(os.environ) if base_env is None else dict(base_env)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path} has no mcpServers")
    project_dir = os.path.dirname(os.path.abspath(path))
    stdio_servers: Dict[str, Dict[str, Any]] = {}
    remote_servers: Dict[str, Dict[str, Any]] = {}
    for name, entry in entries.items():
        where = f"{path}: server {name!r}"
        if name in _CLAUDE_RESERVED_MCP_NAMES:
            raise ValueError(
                f"{where}: this name is reserved by Claude Code for a built-in server "
                "(Claude skips such an entry at load time and warns); it is not a "
                "loadable user MCP server name, so representing or launching it would "
                "misdescribe the effective configuration - rename the server"
            )
        if not isinstance(entry, dict):
            raise ValueError(f"{where} is not an object")
        transport_type = entry.get("type")
        if transport_type is not None and not isinstance(transport_type, str):
            raise ValueError(f"{where} 'type' must be a string")

        if transport_type in _REMOTE_TYPES:
            _validate_runtime_fields(entry, where)
            url = entry.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError(f"{where} ({transport_type}) needs a string 'url'")
            conflicting = [k for k in ("command", "args", "env", "cwd") if k in entry]
            if conflicting:
                raise ValueError(
                    f"{where} declares remote type {transport_type!r} but also carries "
                    f"stdio launch fields {conflicting}; a contradictory entry is refused, "
                    "and a command in a remote entry is never executed"
                )
            if _REMOTE_TYPES[transport_type] == "ws" and "oauth" in entry:
                raise ValueError(
                    f"{where}: Claude Code documents WebSocket MCP authentication as "
                    "header-only (HTTP supports OAuth, WebSocket does not); 'oauth' is "
                    "not a supported ws field, and a shape Claude does not support must "
                    "not be represented as faithful configuration"
                )
            unknown = set(entry) - _REMOTE_ENTRY_FIELDS - _RUNTIME_ONLY_FIELDS
            if unknown:
                raise ValueError(
                    f"{where} carries fields this parser cannot classify: {sorted(unknown)} "
                    "- an unclassified field could be executable configuration, so it "
                    "fails closed instead of being silently dropped"
                )
            headers = _required(entry, "headers", {})
            if not isinstance(headers, dict) or not all(
                isinstance(k, str) and k and isinstance(v, str) for k, v in headers.items()
            ):
                raise ValueError(f"{where} 'headers' must map header names to strings")
            helper = entry.get("headersHelper")
            if helper is not None and (not isinstance(helper, str) or not helper):
                raise ValueError(f"{where} 'headersHelper' must be a command string")
            oauth_entry = entry.get("oauth")
            oauth = None
            if oauth_entry is not None:
                if not isinstance(oauth_entry, dict):
                    raise ValueError(f"{where} 'oauth' must be an object")
                unknown_oauth = set(oauth_entry) - _OAUTH_ENTRY_FIELDS
                if unknown_oauth:
                    raise ValueError(
                        f"{where} 'oauth' carries unclassified fields: {sorted(unknown_oauth)}"
                    )
                oauth = {
                    "client_id": oauth_entry.get("clientId"),
                    "callback_port": oauth_entry.get("callbackPort"),
                    "auth_server_metadata_url_template": oauth_entry.get("authServerMetadataUrl"),
                    "scopes": oauth_entry.get("scopes"),
                }
            remote_source: Dict[str, Any] = {
                "transport": _REMOTE_TYPES[transport_type],
                "url_template": url,  # unexpanded: the pinned identity is the template
                # header value TEMPLATES as written: a same-name Authorization swap
                # between ${READ_ONLY_TOKEN} and ${ADMIN_TOKEN} must be drift, and the
                # resolved credential never appears in a pin
                "header_templates": dict(headers),
                # Claude EXECUTES this command at connect time; it is executable
                # configuration, previously invisible to verification
                "headers_helper_template": helper,
            }
            if _REMOTE_TYPES[transport_type] != "ws":
                # ws is header-only per the documented Claude surface: no oauth member
                # in its canonical source shape (an entry carrying one refused above)
                remote_source["oauth"] = oauth
            remote_servers[str(name)] = remote_source
            continue

        if transport_type not in (None, "stdio"):
            raise ValueError(f"{where} has unsupported type {transport_type!r}")
        command = entry.get("command")
        if not isinstance(command, str) or not command:
            if "url" in entry:
                raise ValueError(
                    f"{where} has a 'url' but no transport 'type'; Claude Code treats "
                    "that as a configuration error, and so does this parser"
                )
            raise ValueError(f"{where} needs a string 'command' (stdio) or a remote 'type'")
        unknown = set(entry) - _STDIO_ENTRY_FIELDS - _RUNTIME_ONLY_FIELDS
        if unknown:
            raise ValueError(
                f"{where} carries fields this parser cannot classify: {sorted(unknown)} "
                "- an unclassified field could be executable configuration, so it fails "
                "closed instead of being silently dropped"
            )
        _validate_runtime_fields(entry, where)

        args = _required(entry, "args", [])
        env = _required(entry, "env", {})
        cwd = entry.get("cwd")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ValueError(f"{where} 'args' must be a list of strings")
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and k and isinstance(v, str) for k, v in env.items()
        ):
            raise ValueError(f"{where} 'env' must map strings to strings")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError(f"{where} 'cwd' must be a string")
        resolved_env = {k: _expand(v, env_base, where=where) for k, v in env.items()}
        resolved_env.setdefault("CLAUDE_PROJECT_DIR", project_dir)
        stdio_servers[str(name)] = {
            "source": {
                "transport": "stdio",
                "command": command,
                "args": list(args),
                "cwd": cwd,
                # as-written value templates: a same-key value swap in the config
                # (NODE_OPTIONS, LD_PRELOAD) is DRIFT, not an invisible change
                "env_templates": dict(env),
            },
            "command": [_expand(command, env_base, where=where)]
            + [_expand(a, env_base, where=where) for a in args],
            "env": resolved_env,
            "cwd": cwd,
        }
    return stdio_servers, remote_servers
