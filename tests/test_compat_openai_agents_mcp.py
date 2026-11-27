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
    alternatives, and an integer that is written as a float; and, from round nine, the
    keywords that turn a failure into a pass (``not``, ``if``, and ``contains`` with its
    ``maxContains``), which the gate reads by two bounding readings and which have a
    comparison of their own with ``jsonschema`` as it ships
    (``test_what_the_gate_sends_is_valid_for_jsonschema_as_it_ships``).

    One error is placed and named differently on purpose, and asserted for the same
    values: the one of the ``false`` schema, which has no keyword. ``jsonschema`` leaves
    it at the list or the object that holds the value; the gate puts it at the value,
    which is the place its reason names. And where ``propertyNames`` is ``false``,
    ``jsonschema`` makes one such error for every member, at the object, and the gate
    one error of ``propertyNames`` there (a name has no place of its own). From round
    nine the gate makes that one error for any ``propertyNames`` a name does not meet,
    where ``jsonschema`` leaves the error of the keyword the name failed at the object;
    and one error for the names a ``required``, a ``dependentRequired`` or a list of
    ``dependencies`` leaves out, where ``jsonschema`` makes one for each name: the gate
    stops reading such a list at the first name that is missing."""
    import importlib

    validators = pytest.importorskip("jsonschema.validators", reason="jsonschema is not installed")
    module = importlib.import_module("recusal_openai_agents_mcp._schema")
    on_purpose = {"enum", "const", "uniqueItems", "pattern", "patternProperties", "format"}
    on_purpose |= {"additionalProperties", "anyOf", "oneOf", "not", "if", "contains"}
    #: one error where `jsonschema` makes one for every name that is left out
    once = {"required", "dependentRequired", "dependencies"}
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

    def split(found):
        """The places of the errors of the ``false`` schema, in order."""
        return sorted(place for keyword, place in found if keyword == "None")

    compared = moved = 0
    for base, strict in module._DRAFTS.items():
        own = module._checks(base, module._GATE, strict.RECUSAL_SIDES)
        reference = validators.extend(
            base,
            validators={keyword: own[keyword] for keyword in on_purpose if keyword in own},
            type_checker=base.TYPE_CHECKER.redefine("integer", module._is_integer),
        )
        for schema in schemas:
            ours, theirs = strict(schema), reference(schema)
            for instance in instances:
                found, wanted = errors(ours, instance), errors(theirs, instance)
                case, failed = (base.__name__, schema, instance), bool(found)
                if type(found) is list and type(wanted) is list:
                    never, nowhere = split(found), split(wanted)
                    if schema.get("propertyNames") is False:
                        # one error of the keyword for theirs of every member's name
                        assert not never and bool(nowhere) is bool(found), case
                        assert all(place == () for place in nowhere), case
                        assert found == [("propertyNames", ())] * bool(nowhere), case
                        found = wanted = []
                    elif "propertyNames" in schema:
                        # one error of the keyword for theirs of each name that fails,
                        # which they leave at the object under the keyword it failed
                        assert all(place == () for _, place in wanted), case
                        assert found == [("propertyNames", ())] * bool(wanted), case
                        found = wanted = []
                    else:
                        # the same errors, each at the holder or one part further in
                        assert len(never) == len(nowhere), case
                        for mine, theirs_place in zip(never, nowhere):
                            assert mine[: len(theirs_place)] == theirs_place, case
                            assert len(mine) - len(theirs_place) in (0, 1), case
                        moved += sum(len(a) != len(b) for a, b in zip(never, nowhere))
                        found = [each for each in found if each[0] != "None"]
                        wanted = [each for each in wanted if each[0] != "None"]
                if type(found) is list and type(wanted) is list:
                    # one error of ours for theirs of each name left out, at the same place
                    wanted = [
                        each
                        for index, each in enumerate(wanted)
                        if each[0] not in once or each not in wanted[:index]
                    ]
                assert found == wanted, case
                compared += failed
    # the comparison is not of nothing: most of these values fail
    assert compared > 2_500
    # and the `false` schema's error was moved to the value in some of them
    assert moved > 20


# --- review round nine: what the gate sends, against `jsonschema` as it ships ---------------------

_DRAFT_ADDRESSES = (
    ("http://json-schema.org/draft-07/schema#", "Draft7Validator"),
    ("https://json-schema.org/draft/2019-09/schema", "Draft201909Validator"),
    ("https://json-schema.org/draft/2020-12/schema", "Draft202012Validator"),
)

#: Schemas the three stricter readings and the stock one answer differently for some
#: value below, and plain ones beside them.
_NINE_LEAVES = [
    {"type": "integer"},
    {"type": ["integer", "string"]},
    {"type": "number"},
    {"const": 0},
    {"const": 1},
    {"const": True},
    {"const": 0.0},
    {"enum": [1, 13]},
    {"enum": [1.0, "a", None]},
    {"const": [1]},
    {"const": {"k": 1}},
    {"enum": [[0], {"k": True}]},
    {"uniqueItems": True},
    {"minimum": 1},
    {"multipleOf": 2},
    {"type": "string"},
    {"required": ["k"]},
    {"properties": {"k": {"type": "integer"}}},
    {"items": {"type": "integer"}},
    {"contains": {"const": 1}},
    {},
    False,
]
_NINE_BESIDE = [{"type": "number"}, {"minimum": 1}, {"const": 1}, False]
_NINE_VALUES = [
    *(0, 1, -1, 5, 13, 0.0, 1.0, 5.0, 13.0, 1.5, True, False, None, "", "a"),
    *([1], [1.0], [True], [1, 1.0], [1, True], [0, 0.0, False], [1, 2], []),
    *({"k": 1}, {"k": 1.0}, {"k": True}, {}, {"k": 5.0, "a": 1}),
]


def _nine_positions(a, b):
    """``a`` where a keyword only adds a demand, and under every keyword that turns a
    failure into a pass or counts what is met, with ``b`` beside it."""
    yield a
    yield {"not": a}
    yield {"not": {"not": a}}
    yield {"if": a, "then": b}
    yield {"if": a, "else": b}
    yield {"if": a, "then": b, "else": {"not": b}}
    yield {"if": {"not": a}, "then": b}
    yield {"oneOf": [a, b]}
    yield {"oneOf": [a, {"not": b}]}
    yield {"not": {"oneOf": [a, b]}}
    yield {"anyOf": [a, b]}
    yield {"anyOf": [{"not": a}, b]}
    yield {"allOf": [{"not": a}, b]}
    yield {"dependentSchemas": {"k": {"not": a}}}
    yield {"dependencies": {"k": {"not": a}}}
    yield {"contains": a, "maxContains": 1}
    yield {"contains": a, "minContains": 2}
    yield {"contains": {"not": a}, "maxContains": 0}
    yield {"properties": {"k": {"not": a}}}
    yield {"items": {"not": a}}
    yield {"additionalProperties": {"oneOf": [a, b]}}
    yield {"unevaluatedProperties": {"not": a}}
    yield {
        "if": {"required": ["k"], "properties": {"k": a}},
        "then": {"properties": {"a": {}}},
        "unevaluatedProperties": False,
    }


def _nine_random(rng, depth=0):
    """A schema of the keywords the gate evaluates, at random, nested four deep."""
    if depth > 3 or rng.random() < 0.1:
        return rng.choice(_NINE_LEAVES)
    made = {}
    for _ in range(rng.randint(1, 3)):
        keyword = rng.choice(
            (
                *("not", "if", "oneOf", "anyOf", "allOf", "contains", "properties", "items"),
                *("additionalProperties", "dependentSchemas", "propertyNames", "leaf", "leaf"),
                *("unevaluatedProperties", "unevaluatedItems", "ref"),
            )
        )
        if keyword == "leaf":
            leaf = rng.choice(_NINE_LEAVES)
            if type(leaf) is dict:
                made.update(leaf)
        elif keyword == "if":
            made["if"] = _nine_random(rng, depth + 1)
            for branch in ("then", "else"):
                if rng.random() < 0.7:
                    made[branch] = _nine_random(rng, depth + 1)
        elif keyword in ("oneOf", "anyOf", "allOf"):
            made[keyword] = [_nine_random(rng, depth + 1) for _ in range(rng.randint(1, 3))]
        elif keyword == "contains":
            made["contains"] = _nine_random(rng, depth + 1)
            if rng.random() < 0.7:
                made[rng.choice(("minContains", "maxContains"))] = rng.randint(0, 2)
        elif keyword in ("properties", "dependentSchemas"):
            made[keyword] = {rng.choice("ka"): _nine_random(rng, depth + 1)}
        elif keyword == "ref":
            made["$ref"] = "#/$defs/d"
        else:
            made[keyword] = _nine_random(rng, depth + 1)
    return made


class _NoPattern:
    """Counts every call that hands the ``re`` module a pattern while the gate works:
    each of its functions that takes one, and the one function all of them compile
    through."""

    ENTRIES = (
        *("compile", "search", "match", "fullmatch", "split", "findall", "finditer", "sub"),
        *("subn", "_compile"),
    )

    def __init__(self):
        import re

        self.re, self.ran, self.on = re, [], False
        self.originals = {name: getattr(re, name) for name in self.ENTRIES if hasattr(re, name)}
        assert "_compile" in self.originals and "search" in self.originals

    def __enter__(self):
        def spying(name):
            def spy(pattern, *args, **kwargs):
                if self.on:
                    # noted and stopped, never run: a pattern here may not finish
                    self.ran.append((name, pattern))
                    raise RuntimeError("a pattern was handed to `re` while the gate worked")
                return self.originals[name](pattern, *args, **kwargs)

            return spy

        for name in self.originals:
            setattr(self.re, name, spying(name))
        return self

    def __exit__(self, *exc):
        for name, original in self.originals.items():
            setattr(self.re, name, original)


def test_what_the_gate_sends_is_valid_for_jsonschema_as_it_ships(ex):
    """The property the stricter readings must keep: for every schema and every value,
    if the gate sends the arguments, ``jsonschema`` as it ships finds them valid against
    the listed schema. The gate may refuse more, never send more.

    Round nine found it broken under the keywords that turn a failure into a pass
    (``{"not": {"const": 0}}`` sent ``0.0``; an ``if`` on ``{"type": "integer"}`` took
    the other branch for ``5000.0``; a ``oneOf`` of an integer and a number was met
    once by ``5.0``). So: every schema above, alone and under every such keyword, in
    the three drafts, against every value above; and schemas made at random from the
    same keywords, four deep. ``pattern``, ``format`` and ``patternProperties`` are not
    in these schemas: the gate leaves them to the server, and they have a corpus of
    their own below. While the gate works, nothing hands the ``re`` module a pattern."""
    import importlib
    import random

    validators = pytest.importorskip("jsonschema.validators", reason="jsonschema is not installed")
    module = importlib.import_module("recusal_openai_agents_mcp._schema")
    tried = sent = 0
    kept = {"not": 0, "if": 0, "oneOf": 0, "maxContains": 0}

    def check(base, root, value):
        nonlocal tried, sent
        tried += 1
        spy.on = True
        try:
            _, completed = listed.complete({"x": value})
        except module.Refused as refused:
            for position in kept:
                kept[position] += f"stands under a `{position}`" in refused.reason
                kept[position] += f"stands under an `{position}`" in refused.reason
            return
        finally:
            spy.on = False
        sent += 1
        assert base(root).is_valid(completed), (base.__name__, root, value)

    with _NoPattern() as spy:
        for address, name in _DRAFT_ADDRESSES:
            base = getattr(validators, name)
            for leaf in _NINE_LEAVES:
                for beside in _NINE_BESIDE:
                    for schema in _nine_positions(leaf, beside):
                        root = {
                            "$schema": address,
                            "type": "object",
                            "properties": {"x": schema},
                        }
                        base.check_schema(root)
                        spy.on = True
                        listed = module.Listed(root)
                        spy.on = False
                        for value in _NINE_VALUES:
                            check(base, root, value)
        structured = tried
        rng = random.Random(9)
        for _ in range(4_000):
            address, name = rng.choice(_DRAFT_ADDRESSES)
            base = getattr(validators, name)
            root = {
                "$schema": address,
                "type": "object",
                "properties": {"x": _nine_random(rng)},
                "$defs": {"d": _nine_random(rng, 2)},
            }
            base.check_schema(root)
            listed = module.Listed(root)
            for value in rng.sample(_NINE_VALUES, 4):
                check(base, root, value)
        assert spy.ran == []
    positions = len(list(_nine_positions({}, {})))
    assert positions == 23
    assert structured == 3 * len(_NINE_LEAVES) * len(_NINE_BESIDE) * positions * len(_NINE_VALUES)
    assert structured == 170_016 and tried == structured + 16_000
    # not of nothing: a good part is sent, and each of the four keywords was left
    # undecided for some of them
    assert sent == 117_990
    assert all(count > 100 for count in kept.values()), kept


def test_no_pattern_of_a_listed_schema_is_handed_to_the_re_module(ex):
    """ "The gate never evaluates a server's regular expression", by counting: schemas
    that hold a pattern in every keyword that takes one (``pattern``,
    ``patternProperties``, a ``format`` of ``regex``, ``propertyNames`` with a pattern),
    where a keyword only adds a demand and under every keyword of round nine, reached
    directly, through ``$defs`` and through a ``$ref`` to a place no keyword holds; in
    the three drafts, against strings and objects whose names the patterns would be
    run on. Reading the listing, checking the call and writing the reason hand the
    ``re`` module nothing."""
    import importlib

    pytest.importorskip("jsonschema.validators", reason="jsonschema is not installed")
    module = importlib.import_module("recusal_openai_agents_mcp._schema")
    evil = "^(a+)+$"
    holders = [
        {"type": "string", "pattern": evil},
        {"pattern": evil},
        {"type": "object", "patternProperties": {evil: {"type": "integer"}}},
        {"patternProperties": {evil: False}, "additionalProperties": False},
        {"patternProperties": {evil: {}}, "additionalProperties": {"type": "integer"}},
        {"propertyNames": {"pattern": evil}},
        {"type": "string", "format": "regex"},
        {"format": "date-time", "pattern": evil},
        {"patternProperties": {evil: {}}, "unevaluatedProperties": False},
    ]
    long_name = "a" * 60 + "!"
    values = [long_name, "", 5, {long_name: 1}, {long_name: "s", "k": 1}, [long_name], {}]
    cases = refused = 0
    with _NoPattern() as spy:
        for address, _ in _DRAFT_ADDRESSES:
            for holder in holders:
                for schema in _nine_positions(holder, {"type": "number"}):
                    for where in ("in place", "$defs", "x-shared"):
                        root = {"$schema": address, "type": "object"}
                        if where == "in place":
                            root["properties"] = {"x": schema}
                        else:
                            root["properties"] = {"x": {"$ref": f"#/{where}/held"}}
                            root[where] = {"held": schema}
                        spy.on = True
                        try:
                            listed = module.Listed(root)
                            for value in values:
                                cases += 1
                                try:
                                    listed.complete({"x": value})
                                except module.Refused:
                                    refused += 1
                        finally:
                            spy.on = False
        assert spy.ran == [], spy.ran[:3]
    assert cases == 3 * len(holders) * 23 * 3 * len(values) == 13_041
    assert refused == 6_378
