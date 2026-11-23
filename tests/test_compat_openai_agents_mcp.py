"""What the OpenAI Agents SDK's MCP paths let a gate see, pinned so a new SDK release cannot change it quietly.

`tests/test_openai_agents_mcp_gate.py` proves the example's behavior. This file pins the SDK
surface that behavior depends on. For local MCP servers: the SDK invokes the server's
`call_tool` with the server's own tool name and the parsed argument dict, whatever public
name the model called (prefixed, sanitized, hashed or shortened), it honors an instance-level
`call_tool`, it passes `meta` only when a resolver produced one, an error result's text is
what the model reads, and a person's approval happens before `call_tool`. For hosted MCP: the
`HostedMCPTool` fields the gate sets, the approval callback's request (name, arguments, server
label) and the response the SDK sends back for a rejection with a reason, and that without a
callback an approval request becomes an interruption rather than a call. An ungated server and
tool are the negative controls, so a probe that stopped observing anything would fail rather
than pass.

It runs in the pinned CI jobs (the exact versions the README names) and weekly against the
newest SDK in `.github/workflows/compat.yml`. A failure there means an SDK release changed
what the gate can see; STABILITY.md says what happens next.
"""

import asyncio
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "openai_agents_mcp_gate.py"

SELECT = "SELECT * FROM t"
DELETE = "DELETE FROM t"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)


@pytest.fixture(scope="module")
def ex():
    pytest.importorskip("agents.mcp", reason="openai-agents is not installed")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recusal_openai_agents_mcp_compat_example", EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_server(ex, calls, servers, **agent_kwargs):
    from agents import Agent

    ex.CALLED.clear()
    model = ex.scripted_model(calls)
    agent = Agent(name="a", model=model, mcp_servers=list(servers), **agent_kwargs)
    asyncio.run(ex.run_agent(agent))
    return ex.tool_outputs(model), list(ex.CALLED)


def _public_names(ex, servers, **agent_kwargs):
    from agents import Agent, RunContextWrapper

    agent = Agent(name="a", model=ex.scripted_model([]), mcp_servers=list(servers), **agent_kwargs)
    return [t.name for t in asyncio.run(agent.get_mcp_tools(RunContextWrapper(context=None)))]


def _record_calls(server):
    """Replace the server's call_tool with a recorder, the way the gate wraps it."""
    from mcp.types import CallToolResult, TextContent

    seen = []

    async def recorder(tool_name, arguments, meta=None):
        seen.append((tool_name, arguments, meta))
        return CallToolResult(content=[TextContent(type="text", text="recorded")])

    server.call_tool = recorder
    return seen


# --- local MCP servers ---------------------------------------------------------------------


def test_call_tool_takes_the_tool_name_the_arguments_and_an_optional_meta(ex):
    from agents.mcp import MCPServer

    params = inspect.signature(MCPServer.call_tool).parameters
    assert list(params) == ["self", "tool_name", "arguments", "meta"]
    assert params["meta"].default is None


def test_the_sdk_calls_an_instance_level_call_tool_with_the_parsed_arguments(ex):
    server = ex.InProcessMCPServer()
    seen = _record_calls(server)
    arguments = json.dumps({"sql": SELECT})
    outputs, called = _run_server(ex, [("run_sql", arguments, "c1")], [server])
    assert seen == [("run_sql", json.loads(arguments), None)]
    assert called == []  # the class method was not reached: the instance attribute won
    assert ex._text(outputs["c1"]) == "recorded"


@pytest.mark.parametrize(
    "label, server_specs, mcp_config, raw_names",
    [
        ("prefixed", [("db", ("run_sql",))], {"include_server_in_tool_names": True}, ["run_sql"]),
        (
            "collision",
            [("db.a", ("run_sql",)), ("db_a", ("run_sql",))],
            {"include_server_in_tool_names": True},
            ["run_sql", "run_sql"],
        ),
        ("sanitized", [("db", ("run.sql",))], {"include_server_in_tool_names": True}, ["run.sql"]),
        ("raw looks prefixed", [("db", ("mcp_db__run_sql",))], None, ["mcp_db__run_sql"]),
        (
            "shortened",
            [("db", ("run_sql_" + "x" * 70,))],
            {"include_server_in_tool_names": True},
            ["run_sql_" + "x" * 70],
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_call_tool_receives_the_servers_own_name_whatever_the_public_name(
    ex, label, server_specs, mcp_config, raw_names
):
    servers = [ex.InProcessMCPServer(name, tools) for name, tools in server_specs]
    recorders = [_record_calls(server) for server in servers]
    kwargs = {"mcp_config": mcp_config} if mcp_config else {}
    public = _public_names(ex, servers, **kwargs)
    assert len(public) == len(raw_names), label
    if mcp_config:
        assert all(name.startswith("mcp_") for name in public)  # the public names differ
    calls = [(name, {"sql": SELECT}, f"c{i}") for i, name in enumerate(public, 1)]
    _run_server(ex, calls, servers, **kwargs)
    raw_seen = sorted(name for seen in recorders for name, _, _ in seen)
    assert raw_seen == sorted(raw_names), label
    assert all(args == {"sql": SELECT} for seen in recorders for _, args, _ in seen)


def test_meta_is_passed_only_when_a_resolver_produced_one(ex):
    plain = ex.InProcessMCPServer()
    seen_plain = _record_calls(plain)
    _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], [plain])
    assert seen_plain[0][2] is None

    with_meta = ex.InProcessMCPServer(tool_meta_resolver=lambda *a, **k: {"trace": "t1"})
    seen_meta = _record_calls(with_meta)
    _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], [with_meta])
    assert seen_meta[0][2] == {"trace": "t1"}


def test_an_error_results_text_is_what_the_model_reads_and_the_run_goes_on(ex):
    from mcp.types import CallToolResult, TextContent

    server = ex.InProcessMCPServer()

    async def refuse(tool_name, arguments, meta=None):
        return CallToolResult(content=[TextContent(type="text", text="stopped here")], isError=True)

    server.call_tool = refuse
    outputs, called = _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], [server])
    assert called == []
    assert ex._text(outputs["c1"]) == "stopped here"


def test_a_call_tool_that_raises_becomes_an_error_the_model_reads_not_a_tool_run(ex):
    """The SDK turns an exception from call_tool into the call's error output (its default
    failure handler) and the run goes on; nothing ran. The gate never relies on raising:
    it returns an error result, so this is pinned only as the surrounding behavior."""
    server = ex.InProcessMCPServer()

    async def boom(tool_name, arguments, meta=None):
        raise RuntimeError("call_tool exploded")

    server.call_tool = boom
    outputs, called = _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], [server])
    assert called == []
    assert "error" in ex._text(outputs["c1"]).lower()


def test_a_persons_approval_happens_before_call_tool(ex):
    from agents import Agent, RunConfig, Runner

    server = ex.InProcessMCPServer(require_approval="always")
    seen = _record_calls(server)
    model = ex.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
    agent = Agent(name="a", model=model, mcp_servers=[server])
    config = RunConfig(tracing_disabled=True)

    async def flow():
        result = await Runner.run(agent, "go", run_config=config)
        assert len(result.interruptions) == 1 and seen == []
        state = result.to_state()
        state.approve(result.interruptions[0])
        await Runner.run(agent, state, run_config=config)

    asyncio.run(flow())
    assert seen == [("run_sql", {"sql": SELECT}, None)]


def test_negative_control_an_ungated_server_runs_delete(ex):
    outputs, called = _run_server(
        ex, [("run_sql", {"sql": DELETE}, "c1")], [ex.InProcessMCPServer()]
    )
    assert called == [("db", "run_sql", {"sql": DELETE})]
    assert ex._text(outputs["c1"]) == f"recorded SQL: {DELETE}"


# --- hosted MCP -----------------------------------------------------------------------------


def test_hosted_mcp_tool_has_the_fields_the_gate_sets(ex):
    import dataclasses

    from agents import HostedMCPTool

    names = {f.name for f in dataclasses.fields(HostedMCPTool)}
    assert {"tool_config", "on_approval_request"} <= names
    tool = HostedMCPTool(
        tool_config={"type": "mcp", "server_label": "db", "server_url": "https://x.invalid"}
    )
    assert tool.on_approval_request is None and tool.name == "hosted_mcp"
    copied = dataclasses.replace(
        tool, tool_config={**tool.tool_config, "require_approval": "always"}
    )
    assert copied.tool_config["require_approval"] == "always"
    assert "require_approval" not in tool.tool_config


def test_the_approval_callback_receives_name_arguments_and_server_label(ex):
    from agents import HostedMCPTool, MCPToolApprovalRequest

    seen = []

    def probe(request):
        assert isinstance(request, MCPToolApprovalRequest)
        seen.append((request.data.name, request.data.arguments, request.data.server_label))
        return {"approve": True}

    tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "db",
            "server_url": "https://x.invalid",
            "require_approval": "always",
        },
        on_approval_request=probe,
    )
    arguments = json.dumps({"sql": SELECT})
    responses = ex.run_hosted([("run_sql", arguments, "r1")], gated=False, tool=tool)
    assert seen == [("run_sql", arguments, "db")]
    assert responses["r1"] == {
        "approval_request_id": "r1",
        "approve": True,
        "type": "mcp_approval_response",
    }


def test_a_rejection_with_a_reason_is_sent_back_as_the_approval_response(ex):
    from agents import HostedMCPTool

    tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "db",
            "server_url": "https://x.invalid",
            "require_approval": "always",
        },
        on_approval_request=lambda request: {"approve": False, "reason": "because"},
    )
    responses = ex.run_hosted([("run_sql", {"sql": SELECT}, "r1")], gated=False, tool=tool)
    assert responses["r1"] == {
        "approval_request_id": "r1",
        "approve": False,
        "type": "mcp_approval_response",
        "reason": "because",
    }


def test_without_a_callback_an_approval_request_is_an_interruption_not_a_call(ex):
    from agents import Agent, HostedMCPTool, RunConfig, Runner
    from agents.testing import ScriptedModel, assistant_message

    tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "db",
            "server_url": "https://x.invalid",
            "require_approval": "always",
        }
    )
    model = ScriptedModel(
        [[ex.approval_request("run_sql", {"sql": SELECT}, "r1")], [assistant_message("done")]]
    )
    result = asyncio.run(
        Runner.run(
            Agent(name="a", model=model, tools=[tool]),
            "go",
            run_config=RunConfig(tracing_disabled=True),
        )
    )
    assert len(result.interruptions) == 1
    assert model.last_call is not None and not any(
        isinstance(item, dict) and item.get("type") == "mcp_approval_response"
        for item in model.last_call.input
    )


def test_negative_control_a_tool_that_approves_everything_approves_delete(ex):
    responses = ex.run_hosted([("run_sql", {"sql": DELETE}, "r1")], gated=False)
    assert responses["r1"]["approve"] is True


# --- the listed schema (the 2026-11-21 review) -------------------------------------------------


def test_the_sdk_lists_through_an_instance_level_list_tools_before_it_calls(ex):
    """The gate keeps each tool's listed input schema by wrapping ``list_tools`` on the
    server object, as it wraps ``call_tool``. The SDK must list through that attribute,
    with the run context and the agent, before any call."""
    from agents.mcp import MCPServer

    params = inspect.signature(MCPServer.list_tools).parameters
    assert list(params) == ["self", "run_context", "agent"]
    server = ex.InProcessMCPServer()
    original = server.list_tools
    order = []

    async def recorder(*args, **kwargs):
        tools = await original(*args, **kwargs)
        order.append(("list", [tool.name for tool in tools]))
        return tools

    server.list_tools = recorder
    seen = _record_calls(server)
    _run_server(ex, [("run_sql", {"sql": SELECT}, "c1")], [server])
    assert order and order[0] == ("list", ["echo", "run_sql"])
    assert [name for name, _, _ in seen] == ["run_sql"]


def test_a_listed_tool_carries_its_input_schema_where_the_gate_reads_it(ex):
    (echo, run_sql) = asyncio.run(ex.InProcessMCPServer().list_tools())
    schema = getattr(run_sql, "input_schema", None)
    if schema is None:
        schema = run_sql.inputSchema
    assert schema == {
        "type": "object",
        "properties": {"sql": {"type": "string"}},
        "required": ["sql"],
    }


def test_the_sdk_sends_a_local_server_the_arguments_without_reading_its_schema(ex):
    """Why the gate reads the schema itself: the SDK hands ``call_tool`` the model's
    arguments as parsed, a string for a parameter the schema types otherwise included."""
    server = ex.InProcessMCPServer()
    seen = _record_calls(server)
    _run_server(ex, [("run_sql", {"sql": 5, "limit": "10"}, "c1"), ("run_sql", {}, "c2")], [server])
    assert [arguments for _, arguments, _ in seen] == [{"sql": 5, "limit": "10"}, {}]


# --- the whole listed schema (the second review of 2026-11-21) --------------------------------


def test_jsonschema_has_the_surface_the_gate_validates_with():
    """The gate validates a call's arguments against the listed schema with ``jsonschema``:
    a validator per draft, chosen by ``$schema``; validation that does not convert and does
    not assert formats; a registry that fetches nothing; and ``descend``, which validates a
    value against a subschema with references resolved against the whole schema."""
    validators = pytest.importorskip("jsonschema.validators", reason="jsonschema is not installed")
    from referencing import Registry

    for name in ("Draft7Validator", "Draft201909Validator", "Draft202012Validator"):
        draft = getattr(validators, name)
        declared = {"$schema": draft.META_SCHEMA["$schema"]}
        assert validators.validator_for(declared, default=None) is draft
    unknown = {"$schema": "https://example.invalid/dialect"}
    assert validators.validator_for(unknown, default=None) is None

    latest = validators.Draft202012Validator
    assert not latest({"type": "boolean"}).is_valid("false"), "validation does not convert"
    assert not latest({"type": "integer"}).is_valid(True)
    # the two places the gate is stricter than JSON Schema
    assert latest({"type": "integer"}).is_valid(1.0)
    assert latest({"enum": [1]}).is_valid(1.0)
    assert latest({"type": "string", "format": "email"}).is_valid("not an address")

    strict = validators.extend(
        latest, type_checker=latest.TYPE_CHECKER.redefine("integer", lambda _, v: type(v) is int)
    )
    assert not strict({"type": "integer"}).is_valid(1.0) and strict({"type": "number"}).is_valid(1)

    root = {"$defs": {"Flag": {"type": "boolean"}}}
    validator = latest(root, registry=Registry())
    assert list(validator.descend("true", {"$ref": "#/$defs/Flag"}))
    assert not list(validator.descend(True, {"$ref": "#/$defs/Flag"}))
    with pytest.raises(Exception, match="Unresolvable"):
        list(validator.descend(1, {"$ref": "https://example.invalid/schema.json"}))
    (error,) = latest({"properties": {"a": {"items": {"type": "integer"}}}}).iter_errors(
        {"a": [1, "x"]}
    )
    assert list(error.absolute_path) == ["a", 1]
    assert error.validator == "type" and error.validator_value == "integer"
    assert error.instance == "x"


# --- the third review of 2026-11-21 -------------------------------------------------------------


def test_a_server_object_shows_its_whole_listing_as_cached_tools(ex):
    """A ``tool_filter`` is applied to what ``list_tools`` returns, for one agent. The gate
    keeps the listing before the filter where the server object shows it: ``cached_tools``,
    a property every ``MCPServer`` has, ``None`` on the base class and the last fetched
    listing on the SDK's own session servers
    (``tests/test_openai_agents_mcp_gate.py`` runs two filtered agents on a real one)."""
    from agents.mcp import MCPServer, MCPServerStdio

    assert isinstance(inspect.getattr_static(MCPServer, "cached_tools"), property)
    assert ex.InProcessMCPServer().cached_tools is None
    held = inspect.getattr_static(MCPServerStdio, "cached_tools")
    assert isinstance(held, property) and held is not MCPServer.__dict__["cached_tools"]
    never_started = MCPServerStdio(params={"command": "never-started"})
    assert never_started.cached_tools is None


def test_jsonschema_has_the_surface_the_gate_counts_and_reads_patterns_with():
    """What the third review added to the gate's use of ``jsonschema``: every keyword
    function is replaced by one that counts, called as ``(validator, value, instance,
    schema)``; a schema is checked with a format checker that leaves ``regex`` out, since
    ``jsonschema`` refuses a ``pattern`` Python cannot compile as an invalid schema and
    raises on one when it validates."""
    jsonschema = pytest.importorskip("jsonschema", reason="jsonschema is not installed")
    import re

    from jsonschema import validators
    from jsonschema.exceptions import SchemaError

    for draft in (
        validators.Draft7Validator,
        validators.Draft201909Validator,
        validators.Draft202012Validator,
    ):
        assert "regex" in draft.FORMAT_CHECKER.checkers
        for keyword in ("anyOf", "oneOf", "pattern", "enum", "const", "$ref", "properties"):
            assert callable(draft.VALIDATORS[keyword]), keyword
        unreadable = {"properties": {"name": {"type": "string", "pattern": "^\\p{L}+$"}}}
        with pytest.raises(SchemaError):
            draft.check_schema(unreadable)
        formats = jsonschema.FormatChecker(())
        formats.checkers = {
            name: check for name, check in draft.FORMAT_CHECKER.checkers.items() if name != "regex"
        }
        draft.check_schema(unreadable, format_checker=formats)
        with pytest.raises(re.error):
            list(draft(unreadable).iter_errors({"name": "x"}))

    called = []

    def counted(validator, value, instance, schema):
        called.append((value, instance, schema))
        return ()

    latest = validators.Draft202012Validator
    counting = validators.extend(latest, validators={"type": counted})
    schema = {"type": "string"}
    assert counting(schema).is_valid(5) and called == [("string", 5, schema)]
    # ``descend`` hands a keyword the very subschema object, which the gate keys its work by
    inner = {"type": "boolean"}
    list(counting({"properties": {"a": inner}}).iter_errors({"a": 1}))
    assert called[-1][2] is inner
    referred = {"$defs": {"Flag": inner}, "properties": {"a": {"$ref": "#/$defs/Flag"}}}
    list(counting(referred).iter_errors({"a": 1}))
    assert called[-1][2] is inner, "a `$ref` resolves to the object in the schema, not a copy"


def test_the_gates_reading_of_text_is_never_narrower_than_pydantics():
    """The gate refuses a string where the schema names a number or a boolean beside
    string and Pydantic's lax conversion would read the string as one. Its own reading
    must hold every string Pydantic's does: checked against the installed Pydantic over
    the forms a number and a boolean are written in."""
    import importlib
    import itertools

    pydantic = pytest.importorskip("pydantic")
    _adapter_module = importlib.import_module("recusal_openai_agents_mcp._schema")
    work = _adapter_module._Work(10**9)
    adapters = {
        "number": [pydantic.TypeAdapter(int), pydantic.TypeAdapter(float)],
        "boolean": [pydantic.TypeAdapter(bool)],
    }

    def pydantic_reads(kind, text):
        for adapter in adapters[kind]:
            try:
                adapter.validate_python(text)
            except pydantic.ValidationError:
                continue
            return True
        return False

    pieces = [
        "",
        " ",
        "\t",
        "\n",
        "\u00a0",
        "\x0b",
        "\x1c",
        "+",
        "-",
        "0",
        "1",
        "5",
        "07",
        "9" * 30,
    ]
    pieces += [
        ".",
        "_",
        "__",
        "e",
        "E",
        "e3",
        "E-3",
        "inf",
        "Infinity",
        "nan",
        "NaN",
        "x",
        ",",
        "0x",
    ]
    pieces += ["true", "True", "FALSE", "yes", "No", "on", "OFF", "t", "F", "y", "N", "null", "2"]
    texts = {
        "".join(parts) for size in (1, 2, 3) for parts in itertools.product(pieces, repeat=size)
    }
    texts |= {"9" * 400, "9" * 5000, "1" + "0" * 400 + ".0", "1_0e1_0", "1._5", "1_.5", "1e_3"}
    assert len(texts) > 60_000
    read = {"number": 0, "boolean": 0}
    for text in texts:
        mine = _adapter_module._reads(work, text)
        for kind in ("number", "boolean"):
            if pydantic_reads(kind, text):
                read[kind] += 1
                assert kind in mine, (kind, text)
    assert read["number"] > 1000 and read["boolean"] > 10, "the comparison saw both readings"
    # and what it is wider by is small enough to say: white space around a boolean word,
    # underscores Pydantic rejects, digits that are not ASCII
    for text in (" true ", "1__0", "_1", "--5", "\u0663"):
        assert _adapter_module._reads(work, text), text
        assert not pydantic_reads("number", text) and not pydantic_reads("boolean", text)
    for text in ("five", "tru", "0x", "x10", "1,000", "5-5"):
        assert _adapter_module._reads(work, text) == frozenset(), text
    # what Pydantic keeps a string and JavaScript's `Number()` reads as a number has a
    # reading of its own, so the reason can say whose rule it is
    for text in ("", " \t", "0x10", "0B11", "0o7", chr(0xFEFF) + "5"):
        assert _adapter_module._reads(work, text) == frozenset({"javascript"}), text
        assert not pydantic_reads("number", text) and not pydantic_reads("boolean", text)


def test_jsonschema_evaluates_a_schemas_patterns_where_the_gate_replaces_it():
    """Why the gate replaces these keywords: ``jsonschema`` runs ``re.search`` with the
    schema's pattern on the instance for ``pattern``, ``patternProperties``,
    ``additionalProperties`` and ``unevaluatedProperties``. The gate's validators run it
    for none of them, and each keyword it replaces is one every draft it reads has."""
    import importlib
    import re

    validators = pytest.importorskip("jsonschema.validators", reason="jsonschema is not installed")

    module = importlib.import_module("recusal_openai_agents_mcp._schema")
    schemas = {
        "pattern": ({"pattern": "^a"}, "abc"),
        "patternProperties": ({"patternProperties": {"^a": {}}}, {"abc": 1}),
        "additionalProperties": (
            {"patternProperties": {"^a": {}}, "additionalProperties": False},
            {"abc": 1},
        ),
        "unevaluatedProperties": (
            {"patternProperties": {"^a": {}}, "unevaluatedProperties": False},
            {"abc": 1},
        ),
    }
    searched = []
    original = re.search

    def counting(pattern, string, *args, **kwargs):
        searched.append(pattern)
        return original(pattern, string, *args, **kwargs)

    re.search = counting
    try:
        for keyword, (schema, instance) in schemas.items():
            searched.clear()
            assert validators.Draft202012Validator(schema).is_valid(instance)
            assert "^a" in searched, keyword
            if keyword == "unevaluatedProperties":
                continue  # refused as a schema by the gate, beside `patternProperties`
            searched.clear()
            assert module._DRAFTS[validators.Draft202012Validator](schema).is_valid(instance)
            assert searched == [], keyword
    finally:
        re.search = original
    replaced = {"enum", "const", "uniqueItems", "pattern", "patternProperties"}
    replaced |= {"additionalProperties", "anyOf", "oneOf"}
    for base, strict in module._DRAFTS.items():
        assert replaced <= set(base.VALIDATORS), base
        assert set(strict.VALIDATORS) == set(base.VALIDATORS)
    # an error raised under `anyOf` carries what the gate reads it by
    strict = module._DRAFTS[validators.Draft202012Validator]
    (error,) = strict({"anyOf": [{"type": "null"}]}).iter_errors(1)
    assert error.validator == "anyOf" and error.validator_value == [{"type": "null"}]
    assert error.instance == 1


def test_the_gates_keywords_assert_what_jsonschemas_assert():
    """The gate checks every keyword with a function of its own, so that none writes a
    message from the value (``jsonschema`` writes the value into the message of every
    error it makes). Each asserts what the function of ``jsonschema`` asserts: the same
    errors, by keyword and by place, for every schema and value here, in every draft the
    gate reads. The reference is the validator of ``jsonschema`` with only what the gate
    makes stricter or does not evaluate on purpose: the comparisons, the patterns, the
    alternatives, and an integer that is written as a float."""
    import importlib

    validators = pytest.importorskip("jsonschema.validators", reason="jsonschema is not installed")
    module = importlib.import_module("recusal_openai_agents_mcp._schema")
    on_purpose = {"enum", "const", "uniqueItems", "pattern", "patternProperties", "format"}
    on_purpose |= {"additionalProperties", "anyOf", "oneOf"}
    integer, string, number = {"type": "integer"}, {"type": "string"}, {"type": "number"}
    schemas = [
        integer,
        {"type": ["integer", "string"]},
        {"type": "number"},
        {"type": "array"},
        {"type": "object"},
        {"type": "null"},
        {"type": "boolean"},
        {"minProperties": 2},
        {"maxProperties": 1},
        {"required": ["x", "zz"]},
        {"not": {}},
        {"not": integer},
        {"not": False},
        {"properties": {"x": False}},
        {"properties": {"x": integer, "a": string}},
        {"propertyNames": False},
        {"propertyNames": {"maxLength": 1}},
        {"if": {}, "then": False},
        {"if": integer, "then": {"minimum": 3}, "else": string},
        {"if": False, "else": False},
        {"additionalProperties": integer},
        {"properties": {"x": {}}, "additionalProperties": False},
        {"allOf": [integer, {"minimum": 1}]},
        {"allOf": [False]},
        {"anyOf": [False, {"minItems": 2}]},
        {"oneOf": [integer, number]},
        {"minItems": 2},
        {"maxItems": 1},
        {"contains": integer},
        {"contains": False},
        {"contains": integer, "minContains": 2},
        {"contains": integer, "maxContains": 1},
        {"contains": integer, "minContains": 0},
        {"contains": {}, "minContains": 5},
        {"items": False},
        {"items": integer},
        {"items": [integer, string]},
        {"items": [integer], "additionalItems": False},
        {"items": [integer], "additionalItems": string},
        {"items": integer, "additionalItems": False},
        {"additionalItems": False},
        {"prefixItems": [integer, string]},
        {"prefixItems": [integer], "items": False},
        {"prefixItems": [False]},
        {"prefixItems": [integer], "items": string},
        {"unevaluatedItems": False},
        {"unevaluatedItems": integer},
        {"prefixItems": [integer], "unevaluatedItems": False},
        {"items": [integer], "unevaluatedItems": False},
        {"allOf": [{"prefixItems": [integer]}], "unevaluatedItems": False},
        {"contains": integer, "unevaluatedItems": False},
        {"if": {"contains": string}, "then": {"items": True}, "unevaluatedItems": False},
        {"unevaluatedProperties": False},
        {"unevaluatedProperties": integer},
        {"properties": {"x": integer}, "unevaluatedProperties": False},
        {"allOf": [{"properties": {"x": {}}}], "unevaluatedProperties": False},
        {"dependentSchemas": {"x": {"properties": {"a": {}}}}, "unevaluatedProperties": False},
        {
            "if": {"required": ["x"]},
            "then": {"properties": {"x": {}}},
            "unevaluatedProperties": False,
        },
        {"minLength": 2},
        {"maxLength": 1},
        {"maximum": 1},
        {"minimum": 1},
        {"exclusiveMaximum": 1},
        {"exclusiveMinimum": 1},
        {"maximum": 1.5},
        {"minimum": 10**400},
        {"multipleOf": 7},
        {"multipleOf": 0.5},
        {"multipleOf": 1e-300},
        {"multipleOf": 10**400},
        {"dependentRequired": {"x": ["a", "zz"], "a": ["x"]}},
        {"dependentSchemas": {"x": {"required": ["a"]}}},
        {"dependentSchemas": {"x": False}},
        {"dependencies": {"x": ["a", "zz"], "a": {"minProperties": 3}}},
        {"dependencies": {"x": False}},
        {"$ref": "#/$defs/never", "$defs": {"never": False}},
        {"$ref": "#/$defs/i", "type": "string", "$defs": {"i": integer}},
        {"type": "object", "properties": {"x": integer, "kid": {"$ref": "#"}}},
        {"items": {"anyOf": [integer, {"type": "array", "items": {"maxLength": 0}}]}},
    ]
    instances = [
        *(0, 1, -1, 5, 7, 14, 2**53 + 1, 10**400, 10**400 + 1, 0.0, 1.0, 1.5, 3.5, 7.0, 1e308),
        *(True, False, None, "", "a", "ab", "5"),
        *([], [1], ["1"], [1, "a"], [1, 1], [1, 2, 3], [[1]], ["a", "b"], [1, "a", 2], [None]),
        *([{"x": 1}], [5, [5, ["a", ""]]]),
        *({}, {"x": 1}, {"x": "1"}, {"a": 1}, {"x": 1, "a": "s"}, {"zz": 1}, {"x": 5.0}),
        *({"a": 1, "x": 1, "zz": None}, {"x": 1, "kid": {"x": 2, "kid": {"x": "3"}}}),
    ]

    def errors(validator, instance):
        try:
            return sorted(
                (str(error.validator), tuple(map(str, error.absolute_path)))
                for error in validator.iter_errors(instance)
            )
        except (OverflowError, module._NoFloat):
            return "a number too large for a float"
        except Exception as exc:  # noqa: BLE001, what one raises the other must raise
            return type(exc).__name__

    compared = 0
    for base, strict in module._DRAFTS.items():
        own = module._checks(base)
        reference = validators.extend(
            base,
            validators={keyword: own[keyword] for keyword in on_purpose if keyword in own},
            type_checker=base.TYPE_CHECKER.redefine("integer", module._is_integer),
        )
        for schema in schemas:
            ours, theirs = strict(schema), reference(schema)
            for instance in instances:
                found = errors(ours, instance)
                assert found == errors(theirs, instance), (base.__name__, schema, instance)
                compared += bool(found)
    # the comparison is not of nothing: most of these values fail
    assert compared > 2_500
