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

What is sent is not yet what the server runs: a server fills a parameter the call leaves
out from its default and converts a value of another type, after the gate. The server's
listed input schema is the only statement of that, so the gate also wraps ``list_tools``,
in place, to keep each tool's listed schema, and before the policy runs it writes every
listed default the call leaves out into the arguments and refuses an argument that is not
already of the JSON type the schema declares (:func:`_schema_refusal`). The policy judges
the completed arguments and the server is sent the completed arguments. A parameter left
out that has no listed default, and whatever else the server does with what it receives,
is not seen.

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
handoffs, an agent with a ``prompt`` (a stored prompt can configure tools of its own), a
sub-agent behind ``as_tool`` that is not itself gated, and ``Agent`` subclasses.

A server carries one policy. Because the gate is installed on the server object, every
agent that lists that object shares it: gating it again with the same policy is a no-op,
and gating it with a different policy is refused with a ``ValueError`` rather than kept or
replaced silently. Use a separate server object per policy.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from typing import Any, Callable, List, Optional

from agents import Agent, HostedMCPTool, MCPToolApprovalRequest
from agents.mcp import MCPServer
from mcp.types import CallToolResult, TextContent
from recusal_openai_agents import gate_tools
from recusal_openai_agents import refusal as _refusal
from recusal_openai_agents import shared_name_reason as _shared_name_reason
from recusal_openai_agents import tools_gated as _tools_gated
from recusal_openai_agents import unjudgeable as _unjudgeable

from recusal import decide
from recusal.claude_code import identity_getter
from recusal.invocation import detached as _detached

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict], List[Any]]
#: Given which seam is deciding and that seam's own context, the recording arguments
#: ``decide`` takes for the call (``audit=``, ``surface=``, ``tool_use_id=``, ...). The seam
#: is ``"function_tool"`` (context: the SDK's tool context), ``"mcp_server"`` (context: the
#: server object) or ``"hosted_mcp"`` (context: the approval request).
#: ``recusal_openai_agents_mcp.recording`` builds one.
Recording = Callable[[str, Any], dict]
_FAILED_CLOSED = "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({name})"


#: The JSON type names a listed schema may declare, and the Python type each one is.
_JSON_TYPES = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _same(a: Any, b: Any) -> bool:
    """Equal, and of the same JSON type all the way down (``1`` is not ``True`` or ``1.0``)."""
    if type(a) is not type(b):
        return False
    if type(a) is list:
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if type(a) is dict:
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return bool(a == b)


def _type_refusal(value: Any, declared: Any) -> Optional[str]:
    """Why ``value`` is not already what the listed schema ``declared`` for it, or None.

    Read from the schema: ``type`` (one name or a list of names), ``enum`` and ``const``,
    and the same three inside ``items`` (for each item of a list) and ``properties`` (for
    each member of an object that names one). Nothing else is read: a ``$ref``, ``anyOf``,
    ``oneOf``, a format or a range leaves the value unchecked. The check is strict, because
    a server converts what does not match after the policy judged it: a string is not a
    boolean or a number, ``true`` is not an integer, and ``5.0`` is not an integer."""
    if type(declared) is not dict:
        return None
    actual = _JSON_TYPES[type(value)]
    named = declared.get("type")
    allowed = [named] if type(named) is str else named if type(named) is list else None
    if allowed is not None:
        names = [name for name in allowed if type(name) is str]
        if actual not in names and not (actual == "integer" and "number" in names):
            return (
                f"is of JSON type {actual} where the server's schema declares {' or '.join(names)}"
            )
    choices = declared.get("enum")
    if type(choices) is list and not any(_same(value, choice) for choice in choices):
        return "is not one of the values the server's schema lists"
    if "const" in declared and not _same(value, declared["const"]):
        return "is not the value the server's schema fixes"
    inner: Optional[str] = None
    if type(value) is list:
        inner = next(filter(None, (_type_refusal(v, declared.get("items")) for v in value)), None)
    elif type(value) is dict and type(declared.get("properties")) is dict:
        members = declared["properties"]
        inner = next(
            filter(None, (_type_refusal(v, members.get(k)) for k, v in value.items())), None
        )
    return f"holds a value that {inner}" if inner is not None else None


def _schema_refusal(args: dict[str, Any], schema: Any) -> Optional[str]:
    """Complete ``args`` from the tool's listed input schema, in place, and say why they are
    not what the server would run with as sent, or None when they are.

    The gate sees what is sent to the server, and the server's listed schema is the only
    statement of what the server then does with it. Two things a server does after the
    policy judged would otherwise go unseen. It fills a parameter the call leaves out from
    its default: every property with a ``default`` that the call leaves out is written
    into ``args`` with that default, so the policy judges it and the server receives it
    written out. It converts a value of another type (the string ``"false"`` to a boolean,
    ``"500"`` to a number): each argument the schema types must already be of that type
    (:func:`_type_refusal`), a written default included.

    Nothing is completed or checked when the schema is not an object schema with
    ``properties``, and an argument the schema does not name is left as sent."""
    properties = schema.get("properties") if type(schema) is dict else None
    if type(properties) is not dict or schema.get("type") != "object":
        return None
    for name, declared in properties.items():
        if name not in args and type(declared) is dict and "default" in declared:
            try:
                args[name] = _detached(declared["default"])
            except (TypeError, RecursionError):
                return f"the default the server's schema lists for `{name}` is not JSON data"
    for name, value in args.items():
        why = _type_refusal(value, properties.get(name))
        if why is not None:
            return (
                f"argument `{name}` {why}; the server would not run with the value the "
                "policy judged. Send the declared JSON type."
            )
    return None


#: The schema of a tool the server has not listed.
_UNLISTED: Any = object()


def _judged_call(
    tool_name: str, arguments: Any, schema: Any, policy: Policy, **record: Any
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Judge a server call: ``(the arguments to send, None)``, or ``(None, reason)`` when it
    is refused.

    ``arguments`` is the dict the SDK hands ``call_tool`` (``None`` for a call without
    any) and ``schema`` the input schema the server listed for the tool. The arguments are
    copied as JSON data, completed with the schema's defaults and checked against its types
    (:func:`_schema_refusal`); the policy judges a copy of the completed arguments and the
    server is sent the completed arguments, so a policy that writes to what it was handed
    changes its copy and not the call. Refused before the policy runs: a non-mapping,
    arguments that are not JSON data, a tool the server does not list, and an argument of
    another type than the schema declares.

    ``record`` goes straight to ``decide``, which then writes the decision to an audit
    log; the entry of a judged call carries the fingerprint of the completed arguments,
    and a call refused before judgment is bound to the arguments as the SDK handed them.
    """
    args = {} if arguments is None else arguments
    if not isinstance(args, dict):
        return None, _unjudgeable(tool_name, "arguments are not a mapping", args, record)
    try:
        sent = _detached(args)
    except RecursionError:
        return None, _unjudgeable(
            tool_name, "arguments are nested too deeply to judge", args, record
        )
    except TypeError as exc:
        return None, _unjudgeable(tool_name, f"arguments are not JSON data ({exc})", args, record)
    if schema is _UNLISTED:
        return None, _unjudgeable(
            tool_name,
            "the server lists no tool of this name, or its tools could not be listed, so the "
            "gate cannot read the schema its arguments would be checked against",
            args,
            record,
        )
    why = _schema_refusal(sent, schema)
    if why is not None:
        return None, _unjudgeable(tool_name, why, args, record)
    try:
        decision, reason = decide(tool_name, _detached(sent), policy, safe_errors=True, **record)
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


def gate_mcp_server(
    server: MCPServer, policy: Policy, recording: Optional[Recording] = None
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
    completes and checks the arguments against. A tool the gate has seen no listing for is
    looked up with one ``list_tools()`` call of the gate's own; a tool the server does not
    list is refused.
    """
    if not isinstance(server, MCPServer):
        raise TypeError(f"Recusal cannot gate {type(server).__name__}: not an agents.mcp.MCPServer")
    mismatch = _name_domain_refusal(policy, server.name, "gate_mcp_server")
    if mismatch is not None:
        raise ValueError(mismatch)
    if is_gated_mcp_server(server):
        # A server object is a connection, shared by every agent that lists it, and it
        # carries ONE policy. Installing the same policy again is a no-op; asking for a
        # different one is refused, because silently keeping the first would leave the
        # caller believing a stricter policy took effect, and silently replacing it would
        # change the policy under every other agent sharing the connection.
        if server.call_tool.recusal_policy is policy:  # type: ignore[attr-defined]
            if server.call_tool.recusal_recording is not recording:  # type: ignore[attr-defined]
                raise ValueError(
                    f"Recusal: server {server.name!r} is already gated with a different "
                    "recording; a connection carries one audit record for every agent that "
                    "shares it. Gate a separate server object to record elsewhere."
                )
            return server
        raise ValueError(
            f"Recusal: server {server.name!r} is already gated with a different policy; a "
            "connection carries one policy for every agent that shares it. Gate a separate "
            "server object for a different policy."
        )
    original = server.call_tool
    original_list_tools = server.list_tools
    #: tool name -> the input schema the server listed for it, most recent listing last
    listed: dict[str, Any] = {}

    async def gated_list_tools(*args: Any, **kwargs: Any) -> Any:
        tools = await original_list_tools(*args, **kwargs)
        for tool in tools:
            schema = getattr(tool, "input_schema", None)
            if schema is None:
                schema = getattr(tool, "inputSchema", None)
            listed[tool.name] = copy.deepcopy(schema)
        return tools

    async def listed_schema(tool_name: str) -> Any:
        """The schema the server listed for the tool: from the listings the SDK has made
        through this server object, or from one the gate asks for when it has seen none."""
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
            sent, reason = _judged_call(tool_name, arguments, schema, policy, **record)
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


def gated_agent(agent: Agent, policy: Policy, recording: Optional[Recording] = None) -> Agent:
    """Return a copy of the agent whose every tool call is judged by Recusal.

    Function tools and hosted MCP tools are replaced by gated copies; the servers in
    ``mcp_servers`` are gated in place and shared with the original. Raises ValueError if
    the agent can run something no gate sees, and TypeError for a tool the function-tool
    gate refuses. ``recording`` puts every decision of all three gates on an audit log.
    The copy has its own ``tools``, ``mcp_servers`` and ``handoffs`` lists, so a later
    change to the original agent's lists does not reach it. Build the agent completely,
    then gate it: a tool or a server appended to the gated copy afterwards is not gated,
    and ``is_gated`` then says so.
    """
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
    servers = [gate_mcp_server(server, policy, recording) for server in agent.mcp_servers]
    return agent.clone(tools=tools, mcp_servers=servers, handoffs=list(agent.handoffs))
