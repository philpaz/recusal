"""The gate for MCP tools on an OpenAI Agents SDK agent: local servers and hosted MCP.

Local MCP servers (``Agent(mcp_servers=[...])``). The SDK connects to these from your own
process, lists each server's tools, and gives every tool a public name the model calls:
the tool's own name by default, or under ``mcp_config={"include_server_in_tool_names":
True}`` a prefixed form, sanitized to the characters the model API accepts, hashed when two
names collide or a function tool already holds the name, and shortened when too long. A
gate keyed on that public name judges a presentation of the capability, not the
capability. The seam this gate uses is the one place the SDK hands the server the
capability itself: ``server.call_tool(tool_name, arguments)``, invoked with the server's own
tool name and the argument dict that is sent to the server, after any approval a person
gave. ``gate_mcp_server(server, policy)`` wraps that method on the server, in place, because
a server object is a connection and a copy would be a second one. A refusal returns an MCP
error result carrying Recusal's reason, which the SDK surfaces to the model as the call's
output, and the server's own ``call_tool`` is never reached. The policy therefore sees the
name the server executes under, whatever the model called it, and a policy keyed by tool
name must name the server's own tools.

What is sent is not yet what the server runs: a server fills what the call leaves out
from its defaults and converts a value of another type, after the gate. The server's
listed input schema is the only statement of that, so the gate also wraps ``list_tools``,
in place, to keep each tool's listed schema, and before the policy runs the arguments
must be valid against the whole of that schema as they stand, and every default the
schema lists for a property they leave out, at any depth, is written into them
(``_schema.Listed.complete``). Arguments the gate cannot check or complete are refused,
and so is what a schema cannot say the outcome of: an object or a list where the schema
offers more than one form, and a string whose text a server would read as another type
the schema also admits (the text of a number beside a number; JSON text for an argument,
which the MCP Python SDK's server would decode and run as that value).
The policy judges the completed arguments and the server is sent the completed arguments,
with one exception: a listed default that cannot be sent (one that is not valid against
its own property schema, or text the server would read as another type) is judged as
listed and not sent. A parameter left out that has no listed default, and whatever a
server does beyond what its schema says, is not seen.

Hosted MCP (``HostedMCPTool``). Those calls run on OpenAI's servers, and the one local point
is the SDK's approval callback, which runs only for a call OpenAI asks approval for.
``gate_hosted_mcp(tool, policy)`` returns a copy whose ``require_approval`` is ``"always"``
and whose ``on_approval_request`` is ``recusal_approval(policy)``: a refusal answers
``{"approve": False, "reason": ...}``, which the SDK sends back as the approval response,
so the hosted call is not made and the model reads the reason. An existing callback keeps
running for the calls the gate allows. This is approval-gated, not execution-gated: a call
OpenAI does not ask approval for is never seen, so a hosted tool whose ``require_approval``
is anything but ``"always"`` is reported as ungated. The policy sees the arguments as the
model sent them; the hosted server's schema is not read here, so its defaults and its type
conversion are not seen.

``gated_agent(agent, policy)`` gates all three at once: function tools (through the
``recusal-openai-agents`` package's ``gate_tools``), every server in ``mcp_servers`` in place, and every
``HostedMCPTool`` as a gated copy. It refuses, rather than skips, what no gate can see:
handoffs, an agent with a ``prompt`` (a stored prompt can configure tools of its own), an
agent whose ``model_settings`` put tools into the request, a sub-agent behind ``as_tool``
that is not itself gated, and ``Agent`` subclasses.

A server carries one policy. Because the gate is installed on the server object, every
agent that lists that object shares it: gating it again with the same policy is a no-op,
and gating it with a different policy is refused with a ``ValueError`` rather than kept or
replaced silently. Use a separate server object per policy.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from json.encoder import encode_basestring_ascii as _as_ascii_json
from typing import Any, Callable, Iterator, List, Optional

from agents import Agent, HostedMCPTool, MCPToolApprovalRequest
from agents.mcp import MCPServer
from mcp.types import CallToolResult, TextContent
from recusal_openai_agents import gate_tools
from recusal_openai_agents import refusal as _refusal
from recusal_openai_agents import settings_reason as _settings_reason
from recusal_openai_agents import shared_name_reason as _shared_name_reason
from recusal_openai_agents import tools_gated as _tools_gated
from recusal_openai_agents import unjudgeable as _unjudgeable

from recusal import decide
from recusal.claude_code import identity_getter
from recusal.invocation import detached as _detached

from ._schema import DEEP as _DEEP
from ._schema import LONG as _LONG
from ._schema import LONGEST_INTEGER as _LONGEST_INTEGER
from ._schema import OVER as _OVER
from ._schema import PASSED_ON as _PASSED_ON
from ._schema import SIZE_CEILING, STEP_CEILING
from ._schema import Listed as _Listed
from ._schema import Refused as _Refused
from ._schema import ceiling_of as _ceiling_of
from ._schema import long_integer_reason as _long_integer_reason
from ._schema import measure as _measure
from ._schema import size_ceiling_of as _size_ceiling_of
from ._schema import too_large_reason as _too_large_reason

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict], List[Any]]
#: Given which seam is deciding and that seam's own context, the recording arguments
#: ``decide`` takes for the call (``audit=``, ``surface=``, ``tool_use_id=``, ...). The seam
#: is ``"function_tool"`` (context: the SDK's tool context), ``"mcp_server"`` (context: the
#: server object) or ``"hosted_mcp"`` (context: the approval request).
#: ``recusal_openai_agents_mcp.recording`` builds one.
Recording = Callable[[str, Any], dict]
_FAILED_CLOSED = "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({name})"
#: The refusal of a call the gate could neither judge nor put on the record.
_UNJUDGED = "Recusal failed closed: the gate could not judge this call ({name})"
#: The deepest a value may sit below the arguments object: the function-tool gate's
#: limit (``recusal_openai_agents``), which nothing here reads below.
_MAX_DEPTH = 64


def _rendered(value: Any) -> str:
    """Arguments that are not JSON data, written as text so that a refusal can be bound to
    them: JSON, with a non-finite number as Python writes it (``nan``, ``inf``), an object
    that is not data as its type's name, and ``...`` for whatever sits below the depth
    the gate reads and for a list or an object that holds itself.

    Written without recursion (the text of ``_schema`` says why). An integer is written
    out here, and none has more digits than the gate reads: the arguments are measured
    first (``_schema.measure``)."""
    out: list[str] = []
    write = out.append
    above: set[int] = set()
    #: for each list or object being written: what is left of it, whether its members
    #: are named, the bracket that closes it, and its identity
    levels: list[tuple[Iterator[Any], bool, str, int]] = [(iter((value,)), False, "", 0)]
    first = True
    while levels:
        members, named, closing, identity = levels[-1]
        shown = len(levels) <= _MAX_DEPTH + 1
        for each in members:
            if not first:
                write(",")
            first = False
            if named:
                name, each = each
                if not shown:
                    write("...")
                elif type(name) is str:
                    write(_as_ascii_json(name))
                else:
                    write(_rendered(name))  # a key that is not text is not JSON data
                write(":")
            if not shown:
                write("...")
                continue
            kind = type(each)
            if kind is str:
                write(_as_ascii_json(each))
            elif kind is dict or kind is list:
                if id(each) in above:
                    write("...")
                    continue
                above.add(id(each))
                if kind is dict:
                    write("{")
                    levels.append((iter(each.items()), True, "}", id(each)))
                else:
                    write("[")
                    levels.append((iter(each), False, "]", id(each)))
                first = True
                break
            elif kind is float or kind is int:
                write(repr(each))
            elif each is None:
                write("null")
            elif kind is bool:
                write("true" if each else "false")
            else:
                write(f"<{kind.__name__}>")
        else:
            levels.pop()
            write(closing)
            above.discard(identity)
            first = False
    return "".join(out)


def _oversize(size_ceiling: int) -> str:
    """What the refusal of a call over the ceiling on its size is bound to (which the
    record holds under the key ``input``): the arguments themselves are not read."""
    return f"<arguments of more than {size_ceiling:,} characters as JSON text>"


#: What the refusal of a call that holds an integer of more digits than the gate reads
#: is bound to, the same way: a fingerprint of the arguments would write the integer out.
_LONG_INTEGER = f"<arguments that hold an integer of more than {_LONGEST_INTEGER:,} digits>"
#: What the refusal of a call the gate could not judge is bound to (:func:`_judged_call`).
_NOT_JUDGED = "<arguments the gate could not judge>"


#: The schema of a tool the server has not listed.
_UNLISTED: Any = object()
#: The schema of a tool the server's listing names more than once. The SDK calls the
#: first of them and a later one would be the last the gate read: the gate reads one
#: schema for a tool, and does not choose between two.
_LISTED_TWICE: Any = object()


def _judged_call(
    tool_name: str,
    arguments: Any,
    schema: Any,
    policy: Policy,
    *,
    size_ceiling: int = SIZE_CEILING,
    **record: Any,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Judge a server call: ``(the arguments to send, None)``, or ``(None, reason)`` when it
    is refused.

    ``arguments`` is the dict the SDK hands ``call_tool`` (``None`` for a call without
    any) and ``schema`` the input schema the server listed for the tool. The arguments are
    copied as JSON data, validated against the whole schema and completed with every
    default it lists (``_schema.Listed.complete``); the policy judges a copy of the
    completed arguments and the server is sent the completed arguments, so a policy that
    writes to what it was handed changes its copy and not the call. A listed default
    that cannot be sent (one that is not valid against its own property schema, or text
    the server would read as another type) is judged as listed and not sent. Refused
    before the policy runs: a non-mapping, arguments that are not JSON data or are nested
    too deeply, a tool the server does not list, arguments that are not valid against the
    schema as they stand, an object or a list the schema offers more than one form for, a
    string a server may read as another type the schema admits, arguments that cost more
    to check than the gate allows (a ``schema`` that is a ``Listed`` carries its
    connection's ceiling; any other is checked under ``STEP_CEILING``), and a schema the
    gate cannot check or complete them against.

    Refused before all of that, and before anything else reads them: arguments of more
    than ``size_ceiling`` characters as JSON text, and arguments that hold an integer of
    more than 4,300 digits (``_schema.measure``, one reading that stops at the ceiling
    and writes no integer out). Everything after it (the copy, the check, the
    fingerprint of an audit entry) takes time in proportion to the size of the
    arguments, and writing a long integer as text takes time that grows with the square
    of its digits where the interpreter sets no limit on it, so none of it is done for
    such a call. The same reading says whether a value sits more than 64 levels down.

    ``record`` goes straight to ``decide``, which then writes the decision to an audit
    log; the entry of a judged call carries the fingerprint of the arguments as judged,
    and a call refused before judgment is bound to the arguments as the SDK handed them,
    or, when they are not JSON data and so have no fingerprint, to the text
    :func:`_rendered` writes them as (which the record holds under the key ``input``).
    A call refused for its size or for a long integer is bound to a statement of that
    (:func:`_oversize`, ``_LONG_INTEGER``) and not to its arguments, which are not
    read: its entry holds no fingerprint of them.

    Nothing leaves this function but its answer and what stops the interpreter or the
    event loop (``_schema.PASSED_ON``). Any other exception, one that is not an
    ``Exception`` among them, is a refusal of the call, on the record where the record
    can be written: the gate's own reading does not recurse, but the validation, the
    policy and the record do, and from deep in an application's stack the
    interpreter's recursion limit can be reached anywhere in them.
    """
    try:
        return _judged(tool_name, arguments, schema, policy, size_ceiling, record)
    except _PASSED_ON:
        raise
    except BaseException as exc:  # noqa: BLE001, nothing leaves the gate but a refusal
        name = type(exc).__name__
    if name == "RecursionError":
        detail = (
            "the gate could not judge this call within what is left of the interpreter's "
            "recursion limit where it was called"
        )
    else:
        detail = f"the gate could not judge this call ({name})"
    try:
        return None, _unjudgeable(tool_name, detail, _NOT_JUDGED, record)
    except _PASSED_ON:
        raise
    except BaseException as exc:  # noqa: BLE001, the refusal stands without its record
        return None, _UNJUDGED.format(name=type(exc).__name__)


def _judged(
    tool_name: str, arguments: Any, schema: Any, policy: Policy, size_ceiling: int, record: dict
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """:func:`_judged_call`, which turns whatever leaves this into a refusal."""
    args = {} if arguments is None else arguments
    # one reading, before any other: the size, an integer too long to write, the depth
    found = _measure(args, size_ceiling, _MAX_DEPTH)
    if found == _OVER:
        detail = _too_large_reason(size_ceiling)
        return None, _unjudgeable(tool_name, detail, _oversize(size_ceiling), record)
    if found == _LONG:
        return None, _unjudgeable(tool_name, _long_integer_reason(), _LONG_INTEGER, record)
    proposed: Any = None
    fault: Optional[str] = "arguments are nested too deeply" if found == _DEEP else None
    if fault is None:
        try:
            proposed = _detached(args)
        except TypeError as exc:
            fault = f"arguments are not JSON data ({exc})"

    def refused(detail: str) -> tuple[None, str]:
        # A refusal before judgment is bound to the arguments as the SDK handed them
        # where they are JSON data, which the copy has settled, and they are not read
        # again to find that out; otherwise to the text they are written as
        # (:func:`_rendered`, which the record holds under the key ``input``).
        bound = args if fault is None else _rendered(args)
        return None, _unjudgeable(tool_name, detail, bound, record)

    if not isinstance(args, dict):
        return refused("arguments are not a mapping")
    if fault is not None:
        return refused(fault)
    if schema is _UNLISTED:
        return refused(
            "the server lists no tool of this name, or its tools could not be listed, so the "
            "gate cannot read the schema its arguments would be checked against"
        )
    if schema is _LISTED_TWICE:
        return refused(
            "the server's listing names this tool more than once; the gate checks the "
            "arguments of a tool against the one schema the server lists for it, and "
            "does not choose between two listings of one name"
        )
    try:
        listed = schema if isinstance(schema, _Listed) else _Listed(schema)
        judged, sent = listed.complete(proposed, owned=True)
    except _Refused as cause:
        return refused(cause.reason)
    try:
        decision, reason = decide(tool_name, judged, policy, safe_errors=True, **record)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return None, _FAILED_CLOSED.format(name=type(exc).__name__)
    return (None, reason) if decision == "deny" else (sent, None)


def claude_named(server_name: str, policy: Policy) -> Policy:
    """Bind a policy written for Claude Code's MCP tool names to one server connection.

    :func:`recusal.mcp.manifest_policy` (and any composition carrying one) judges
    ``mcp__<server>__<tool>``, the name Claude Code gives an MCP tool. The gates in this
    module hand a policy the server's own tool name, so that policy would never see an MCP
    name, run no membership check and raise no objection. The bound policy presents each
    call as ``mcp__{server_name}__{tool}``, carries the server it is bound to (a gate refuses
    it for any other server, and so does a composition carrying it), and forwards the
    manifest digest to the audit record through the policy's own control identity. A policy
    already bound to this server is returned as it is, so binding twice never prefixes
    twice; one bound to another server is refused.
    """
    if not isinstance(server_name, str) or not server_name:
        raise ValueError("claude_named needs the name of the server the policy is bound to")
    already = getattr(policy, "recusal_bound_server", None)
    if already is not None:
        if already != server_name:
            raise ValueError(
                f"Recusal: this policy is already bound to server {already!r} by claude_named, "
                f"not to {server_name!r}; bind one policy object per server"
            )
        return policy

    def bound(tool_name: str, arguments: dict) -> List[Any]:
        return policy(f"mcp__{server_name}__{tool_name}", arguments)

    bound.recusal_bound_server = server_name  # type: ignore[attr-defined]
    bound.recusal_bound_policy = policy  # type: ignore[attr-defined]
    getter = identity_getter(policy)
    if getter is not None:
        bound.get_control_identity = getter  # type: ignore[attr-defined]
    reset = getattr(policy, "reset_control_identity", None)
    if callable(reset):
        bound.reset_control_identity = reset  # type: ignore[attr-defined]
    return bound


def _name_domain_refusal(policy: Policy, server_name: Any, where: str) -> Optional[str]:
    """Why this policy cannot judge this server's own tool names, or None if it can."""
    # The naming requirement is checked before the binding: a policy that still judges
    # Claude Code's names is never installable, bound neighbour or not.
    if getattr(policy, "recusal_name_domain", None) == "claude_code":
        return (
            "Recusal: this policy judges Claude Code's MCP names (`mcp__<server>__<tool>`, the "
            f"shape of recusal.mcp.manifest_policy), but {where} hands a policy the server's "
            "own tool name, so its membership check would see no MCP name and raise no "
            f"objection. Bind it to the connection: claude_named({server_name!r}, policy)."
        )
    bound_to = getattr(policy, "recusal_bound_server", None)
    if bound_to is not None and bound_to != server_name:
        return (
            f"Recusal: this policy is bound to server {bound_to!r} by claude_named, not to "
            f"{server_name!r}; bind one policy object per server"
        )
    return None


def _already_gated(
    server: Any,
    policy: Policy,
    recording: Optional[Recording],
    step_ceiling: Any,
    size_ceiling: Any,
) -> bool:
    """Whether ``server`` already carries this gate, exactly: False for a server no gate
    is on. Raises for everything :func:`gate_mcp_server` refuses, and changes nothing,
    so a caller with several servers can ask about all of them before it gates one."""
    if not isinstance(server, MCPServer):
        raise TypeError(f"Recusal cannot gate {type(server).__name__}: not an agents.mcp.MCPServer")
    ceiling = _ceiling_of(step_ceiling)
    size = _size_ceiling_of(size_ceiling)
    mismatch = _name_domain_refusal(policy, server.name, "gate_mcp_server")
    if mismatch is not None:
        raise ValueError(mismatch)
    if not is_gated_mcp_server(server):
        return False
    # A server object is a connection, shared by every agent that lists it, and it
    # carries ONE policy. Installing the same policy again is a no-op; asking for a
    # different one is refused, because silently keeping the first would leave the
    # caller believing a stricter policy took effect, and silently replacing it would
    # change the policy under every other agent sharing the connection.
    gate: Any = server.call_tool
    if gate.recusal_policy is not policy:
        raise ValueError(
            f"Recusal: server {server.name!r} is already gated with a different policy; a "
            "connection carries one policy for every agent that shares it. Gate a separate "
            "server object for a different policy."
        )
    if gate.recusal_recording is not recording:
        raise ValueError(
            f"Recusal: server {server.name!r} is already gated with a different "
            "recording; a connection carries one audit record for every agent that "
            "shares it. Gate a separate server object to record elsewhere."
        )
    for keyword, asked, carried in (
        ("step_ceiling", ceiling, gate.recusal_step_ceiling),
        ("size_ceiling", size, gate.recusal_size_ceiling),
    ):
        if asked != carried:
            raise ValueError(
                f"Recusal: server {server.name!r} is already gated with a different "
                f"{keyword}; a connection carries one for every agent that shares it. "
                "Gate a separate server object for another ceiling."
            )
    return True


def gate_mcp_server(
    server: MCPServer,
    policy: Policy,
    recording: Optional[Recording] = None,
    *,
    step_ceiling: int = STEP_CEILING,
    size_ceiling: int = SIZE_CEILING,
) -> MCPServer:
    """Wrap the server's ``call_tool`` so every call is judged by the server's own tool
    name and the arguments that are sent to it, in place; returns the same server.

    The SDK invokes ``call_tool`` with the raw tool name whatever public name the model
    used, after any approval, so this is the point where the policy judges the capability
    that executes. A refusal is returned as an MCP error result carrying the reason, and
    the server's own ``call_tool`` is not reached. ``recording`` puts every decision on an
    audit log; like the policy, a connection carries one.

    The server's ``list_tools`` is wrapped too, in place, and only to read: it returns what
    the server listed and keeps each tool's input schema, which :func:`_judged_call`
    checks and completes the arguments against. Each listing replaces the one before, so a
    tool the server no longer lists is no longer known. Where the server object keeps its
    listing as it was before any ``tool_filter`` (``cached_tools``, which the SDK's own
    servers do), that whole listing is what the gate keeps, so agents that share the
    server under different filters share one set of schemas; a server object that keeps
    none leaves the gate with the tools of the latest listing as filtered. A call to a
    tool that is not in
    the most recent listing makes the gate call ``list_tools()`` itself, with no run
    context and no agent, once for that call; a tool that is still not listed is refused.

    The check runs on the event loop, and two ceilings bound how long one call can hold
    it; a call over either is refused before the policy runs. ``step_ceiling`` is the
    most steps the check of one call's arguments may cost (``STEP_CEILING``, 200,000,
    unless given), whatever the arguments hold. ``size_ceiling`` is the most characters
    the arguments of one call may be as JSON text (``SIZE_CEILING``, 1,000,000, unless
    given), which is checked before anything else reads them: what a step costs grows
    with the length of the string it reads, and the gate copies and counts every value
    it is handed, a value in an argument the schema does not name among them, which
    costs no step. Raise them for a server whose ordinary calls are larger; like the
    policy, a connection carries one of each.
    """
    if _already_gated(server, policy, recording, step_ceiling, size_ceiling):
        return server
    ceiling, size = step_ceiling, size_ceiling
    original = server.call_tool
    original_list_tools = server.list_tools
    #: tool name -> the input schema in the server's most recent listing
    listed: dict[str, Any] = {}

    def keep(tools: Any) -> None:
        latest: dict[str, Any] = {}
        for tool in tools:
            if tool.name in latest:
                # one name, two listings: every call of it is refused
                latest[tool.name] = _LISTED_TWICE
                continue
            schema = getattr(tool, "input_schema", None)
            if schema is None:
                schema = getattr(tool, "inputSchema", None)
            latest[tool.name] = _Listed(copy.deepcopy(schema), ceiling)
        listed.clear()  # a tool the server withdrew is no longer listed
        listed.update(latest)

    def whole_listing() -> Any:
        """The server's most recent listing before any ``tool_filter``, where the server
        object keeps one (the SDK's own servers do, as ``cached_tools``), or None."""
        try:
            return server.cached_tools
        except Exception:  # noqa: BLE001, a server object without one keeps none
            return None

    async def gated_list_tools(*args: Any, **kwargs: Any) -> Any:
        try:
            tools = await original_list_tools(*args, **kwargs)
        except Exception:
            # the SDK fetches the listing and then filters it: a filter that raises (a
            # dynamic one asked without an agent) leaves the fetched listing to read
            whole = whole_listing()
            if whole is not None:
                keep(whole)
            raise
        whole = whole_listing()
        # a ``tool_filter`` hides tools from one agent; the schemas are the server's, so
        # the gate keeps the whole listing and two agents with two filters share it
        keep(tools if whole is None else whole)
        return tools

    async def listed_schema(tool_name: str) -> Any:
        """The schema the server listed for the tool: from the most recent listing made
        through this server object, or from one the gate asks for when the tool is not
        in it."""
        if tool_name not in listed:
            try:
                await gated_list_tools()
            except Exception:  # noqa: BLE001, a server that cannot list leaves the tool unlisted
                pass
        return listed.get(tool_name, _UNLISTED)

    async def gated_call_tool(
        tool_name: str, arguments: Optional[dict[str, Any]], meta: Any = None
    ) -> CallToolResult:
        sent: Optional[dict[str, Any]] = None
        try:
            record = dict(recording("mcp_server", server)) if recording is not None else {}
        except Exception as exc:  # noqa: BLE001, a gate that cannot record must not run the call
            reason: Optional[str] = _FAILED_CLOSED.format(name=type(exc).__name__)
        else:
            schema = await listed_schema(tool_name)
            sent, reason = _judged_call(
                tool_name, arguments, schema, policy, size_ceiling=size, **record
            )
        if sent is None:
            # ``isError`` is the field's name on the wire, which MCP's model accepts as an
            # alias; a type checker knows only the Python name, ``is_error``
            return CallToolResult(
                content=[TextContent(type="text", text=str(reason))],
                isError=True,  # type: ignore[call-arg]
            )
        if meta is None:
            return await original(tool_name, sent)
        return await original(tool_name, sent, meta=meta)

    gated_call_tool.recusal_gate = True  # type: ignore[attr-defined]
    gated_call_tool.recusal_policy = policy  # type: ignore[attr-defined]
    gated_call_tool.recusal_recording = recording  # type: ignore[attr-defined]
    gated_call_tool.recusal_step_ceiling = ceiling  # type: ignore[attr-defined]
    gated_call_tool.recusal_size_ceiling = size  # type: ignore[attr-defined]
    server.call_tool = gated_call_tool  # type: ignore[method-assign]
    server.list_tools = gated_list_tools  # type: ignore[method-assign]
    return server


def is_gated_mcp_server(server: Any) -> bool:
    """True if the server's ``call_tool`` is wrapped by the Recusal gate."""
    return isinstance(server, MCPServer) and bool(getattr(server.call_tool, "recusal_gate", False))


# --- hosted MCP -------------------------------------------------------------------------------


def recusal_approval(policy: Policy, then: Any = None, recording: Optional[Recording] = None):
    """Return an ``on_approval_request`` callback that judges each hosted call OpenAI asks about.

    A refusal answers ``{"approve": False, "reason": reason}``. An allowed call is passed to
    ``then`` (the callback the tool already had), or approved when there is none.
    ``recording`` puts Recusal's decision on an audit log before ``then`` is asked; what
    ``then`` answers is not Recusal's decision and is not on the log.
    """

    async def gate(request: MCPToolApprovalRequest) -> dict:
        try:
            record = dict(recording("hosted_mcp", request)) if recording is not None else {}
        except Exception as exc:  # noqa: BLE001, a gate that cannot record must not approve
            return {"approve": False, "reason": _FAILED_CLOSED.format(name=type(exc).__name__)}
        reason = _refusal(request.data.name, request.data.arguments, policy, **record)
        if reason is not None:
            return {"approve": False, "reason": reason}
        if then is None:
            return {"approve": True}
        result = then(request)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


def gate_hosted_mcp(
    tool: HostedMCPTool, policy: Policy, recording: Optional[Recording] = None
) -> HostedMCPTool:
    """Return a copy of the hosted MCP tool that requires approval for every call and judges
    each approval request with Recusal; the original is unchanged."""
    if not isinstance(tool, HostedMCPTool):
        raise TypeError(f"Recusal cannot gate {type(tool).__name__}: not a HostedMCPTool")
    mismatch = _name_domain_refusal(policy, tool.tool_config.get("server_label"), "gate_hosted_mcp")
    if mismatch is not None:
        raise ValueError(mismatch)
    # the SDK's own typed dict, with one key replaced
    config: Any = {**tool.tool_config, "require_approval": "always"}
    return dataclasses.replace(
        tool,
        tool_config=config,
        on_approval_request=recusal_approval(
            policy, then=tool.on_approval_request, recording=recording
        ),
    )


def is_gated_hosted_mcp(tool: Any) -> bool:
    """True if every call to this hosted tool is sent for approval and judged by Recusal."""
    return (
        isinstance(tool, HostedMCPTool)
        and tool.tool_config.get("require_approval") == "always"
        and getattr(tool.on_approval_request, "recusal_gate", False)
    )


# --- the whole agent --------------------------------------------------------------------------


def _ungateable_reason(agent: Any) -> Optional[str]:
    if type(agent) is not Agent:
        return (
            f"{type(agent).__name__} is not a plain Agent; a subclass can add tools at run "
            "time that no gate sees"
        )
    if agent.handoffs:
        return "it has handoffs, which hand the conversation to another agent's tools"
    if agent.prompt is not None:
        return (
            "it has a prompt, and a prompt stored on OpenAI's servers can configure tools "
            "(hosted MCP among them) that are not in the agent's tools list, which no gate "
            "sees. Put the instructions and the tools on the agent itself."
        )
    settings = _settings_reason(agent)
    if settings is not None:
        return f"its {settings}"
    shared = _shared_name_reason(agent.tools)
    if shared is not None:
        return f"its {shared}"
    return None


def is_gated(agent: Any, _seen: Optional[set[int]] = None) -> bool:
    """True if every call this agent can make, function tool, local MCP tool or hosted MCP
    call OpenAI asks approval for, passes through a Recusal gate. A sub-agent behind
    ``as_tool`` is held to the same rule, its own MCP servers and hosted tools included."""
    seen = set() if _seen is None else _seen
    if id(agent) in seen:
        return True
    seen.add(id(agent))
    if _ungateable_reason(agent) is not None:
        return False
    hosted = [tool for tool in agent.tools if isinstance(tool, HostedMCPTool)]
    others = [tool for tool in agent.tools if not isinstance(tool, HostedMCPTool)]
    if not all(is_gated_hosted_mcp(tool) for tool in hosted):
        return False
    if not all(is_gated_mcp_server(server) for server in agent.mcp_servers):
        return False
    return _tools_gated(others, lambda sub: is_gated(sub, seen))


def gated_agent(
    agent: Agent,
    policy: Policy,
    recording: Optional[Recording] = None,
    *,
    step_ceiling: int = STEP_CEILING,
    size_ceiling: int = SIZE_CEILING,
) -> Agent:
    """Return a copy of the agent whose every tool call is judged by Recusal.

    Function tools and hosted MCP tools are replaced by gated copies; the servers in
    ``mcp_servers`` are gated in place and shared with the original. Raises ValueError if
    the agent can run something no gate sees, and TypeError for a tool the function-tool
    gate refuses. ``recording`` puts every decision of all three gates on an audit log.
    The copy has its own ``tools``, ``mcp_servers`` and ``handoffs`` lists, so a later
    change to the original agent's lists does not reach it. Build the agent completely,
    then gate it: a tool or a server appended to the gated copy afterwards is not gated,
    and ``is_gated`` then says so. ``step_ceiling`` and ``size_ceiling`` are given to
    ``gate_mcp_server`` for every server in ``mcp_servers``: the most steps the check of
    one call to a local server may cost, and the most characters its arguments may be.
    When it raises, no server has been gated: a server ``gate_mcp_server`` would refuse
    (one already gated with another policy, recording or ceiling) is found before the
    first is touched.
    """
    ceiling = _ceiling_of(step_ceiling)
    size = _size_ceiling_of(size_ceiling)
    reason = _ungateable_reason(agent)
    if reason is not None:
        raise ValueError(f"Recusal cannot gate agent {agent.name!r}: {reason}")
    for_function_tools = None
    if recording is not None:

        def for_function_tools(ctx: Any) -> dict:
            return recording("function_tool", ctx)

    tools: list[Any] = []
    for tool in agent.tools:
        if isinstance(tool, HostedMCPTool):
            tools.append(gate_hosted_mcp(tool, policy, recording))
        else:
            # a sub-agent behind as_tool is gated when THIS package says so: it may hold
            # MCP servers and hosted tools, which the function-tool package alone refuses
            tools.extend(gate_tools([tool], policy, for_function_tools, agent_gated=is_gated))
    # A server is gated in place, so all of them or none: every server is asked whether
    # it can carry this gate, and the copy of the agent is made, before the first one
    # is touched. A server the gate refuses (one already gated another way) then leaves
    # the ones listed before it as they were.
    servers = list(agent.mcp_servers)
    for server in servers:
        _already_gated(server, policy, recording, ceiling, size)
    gated = agent.clone(tools=tools, mcp_servers=servers, handoffs=list(agent.handoffs))
    for server in servers:
        gate_mcp_server(server, policy, recording, step_ceiling=ceiling, size_ceiling=size)
    return gated
