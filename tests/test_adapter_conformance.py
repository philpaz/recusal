"""One contract, every framework adapter: the invariants a gate must hold wherever it sits.

Each adapter under ``examples/`` proves its own behavior in its own test file, in its own
words. This file states the invariants once and runs them against every adapter through a
small normalizer per adapter, so a gate that holds them in one framework and not another
shows up as a failure here rather than as a gap in prose. The invariants:

1. **The arguments the policy judged are the arguments the tool ran with.** A recording
   policy and the adapter's own execution recorder must agree, name and arguments, for an
   allowed call.
2. **A refusal runs nothing, and its reason reaches the model.** The tool is not executed
   and the model's next turn carries Recusal's reason.
3. **Calls in one turn are judged separately.** A refused call does not block the allowed
   calls beside it, and the allowed ones run with their own arguments.
4. **A policy that raises fails closed.** Nothing runs; the model reads that the gate failed
   closed.
5. **The negative control.** Ungated, the same refused call runs: the adapter's recorder is
   live and the refusal above is the gate's doing.
6. **A schema that changes an allowed proposal never runs the changed value.** The tool's
   own validation rewrites ``SELECT`` into ``DELETE`` after the model proposed it. The gate
   either judges the value the tool would execute (and refuses it) or refuses what it cannot
   judge; either way nothing runs. Registered for the adapters whose framework validates
   arguments between the hook and the function: the OpenAI Agents function tools (the
   parameter model the SDK generates), LangGraph's ``ToolNode`` and the LangChain middleware
   built on it (``args_schema``). Not registered where the hook already sits after
   validation or no schema layer exists: Pydantic AI's ``before_tool_execute`` receives the
   validated arguments by the framework's design, the Responses and Chat Completions loops
   dispatch the model's JSON themselves, and the MCP ``call_tool`` seam carries the exact
   arguments the server receives. CrewAI validates against the tool's ``args_schema`` at
   execution, so it is registered: its gate refuses a schema that is not a stock model (a
   validator is a customisation), so a rewriting schema never runs. Google ADK's one
   preparation layer is covered by invariant 9, not by a rewriting schema (the ADK has no
   schema object to rewrite with).
7. **The record binds what executed.** See the paragraph below the list.
8. **A default the proposal omitted is judged at the value the callable uses.** A tool
   ``charge(amount: int = 100)`` proposed with ``{}`` reaches the policy as
   ``{"amount": 100}`` (an ``int``, exactly), and a budget policy capping at 10 refuses it;
   proposed with ``{"amount": 5}`` it runs with 5. Registered for the direct Responses and
   Chat loops (``bind`` plus ``apply_defaults``), Google ADK's callback and plugin (defaults
   completed into the raw arguments before the tool's own preparation), LangGraph and the
   LangChain middleware (``_completed`` in the judging copy), the OpenAI Agents function
   tools (the SDK's prepared invocation) and CrewAI (defaults completed from the schema
   before its single validation). Pydantic AI judges the validated arguments by the
   framework's design (the reviewer's control showed it). MCP has no Python callable.
9. **Re-preparation cannot change a judged value.** Where a framework prepares arguments
   again at execution (ADK's ``_preprocess_args_with_validation``, CrewAI's
   ``args_schema.model_validate``), the gate supports only parameter shapes whose
   preparation is a pure function of value and annotation and refuses the rest; the adapter
   tests (`tests/test_google_adk_gate.py`, `tests/test_openai_responses_gate.py`,
   `tests/test_crewai_gate.py`) prove exact type-and-value agreement between the judged and
   received values on the supported set and refusal before execution for a model-typed,
   ``Annotated``, ``Enum``, variadic or unannotated parameter, and on CrewAI for a
   customised model class or a field setting beyond a default, description and title.
10. **A transform the framework runs after the gate's seam cannot change what runs.** An
   agent-level before-tool callback that rewrites an allowed ``SELECT`` into a ``DELETE``
   either runs before the gate, so the gate judges the rewritten value and refuses it, or
   is refused outright because the gate cannot be placed after it. Registered for Google
   ADK's two routes: ``gated_agent`` appends the gate last among the agent's callbacks
   (judged as rewritten); ``RecusalPlugin`` runs before every agent callback, so it refuses
   a call whose executing agent carries a before-tool callback that is not a Recusal gate,
   or whose runner lists a plugin overriding ``before_tool_callback`` after it (nothing
   runs, the policy never sees the proposal). Not registered where the framework offers
   no way to see a later hook from inside the gate (CrewAI's process-global hook list,
   LangGraph's and the OpenAI SDK's middleware chains): there the README says to install
   the gate last and the adapter tests show the bypass and the fix.
11. **A policy's write to what it judged cannot change what runs.** The policy receives a
   detached observation of the invocation and the tool runs with the invocation itself. A
   policy that gathers passing evidence for ``SELECT`` and then, by a bug, writes ``DELETE``
   into the dict it was handed has changed its copy: the tool still runs ``SELECT``, and
   the policy's own record of what it saw is ``SELECT``. Registered for every route of
   every adapter. Containment of a buggy trusted policy, not a model-input bypass: the
   same proposal under a correct policy is still judged and still refused when it should
   be (invariant 2). Where the framework re-prepares arguments (invariant 9) the detached
   copy is of the prepared invocation, so the observation and the execution agree in type
   and value as well as in text. Pydantic AI supports data-only values, detaching JSON
   containers by construction and retaining only immutable-scalar-valued Enum members;
   its adapter tests refuse rich objects and mutable Enum values on both routes before
   policy evaluation, and retain an observational/ungated control.

Every route an adapter offers (sync and async, callback and plugin, run and streamed) is a
separate row, because each is a separate code path through the framework.

7. **The record binds what executed.** Where a gate writes an audit entry, its input
   fingerprint is that of the invocation the tool ran with, whatever a policy did to the
   dict it was handed. The evaluator fingerprints the proposal before the policy sees it
   (`tests/test_decide_audit.py`, `tests/test_hook_audit.py`); the one adapter with an
   audited route, the LangGraph audited gate, proves it on the executed invocation
   (`tests/test_langgraph_audit.py`). Not a row here, because no other adapter example
   writes a record yet.

Invariants that only some adapters can express stay in those adapters' own tests, and this
docstring says where: structural batch rejection with zero executions
(`tests/test_openai_responses_gate.py`), a public rename that preserves the decision for the
same capability (`tests/test_openai_agents_mcp_gate.py`), delegation that preserves mediation
(`tests/test_google_adk_gate.py`, `tests/test_pydantic_ai_gate.py`,
`tests/test_openai_agents_gate.py`), and composition that preserves control identity
(`tests/test_composed_policy_example.py`).

A registry drift lock keeps the suite complete: every ``examples/*_gate.py`` the README's
Works-with table marks Tested must be registered here, and this file must run in every
runtime lane of ``ci.yml`` and ``compat.yml``. An adapter whose SDK is not installed is
skipped, never silently passed.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"
Call = tuple[str, dict, str]  # (tool name, arguments, call id)
Outcome = tuple[bool, str]  # (refused, the text the model received)
Executed = list[tuple[str, dict]]


# --- the registry ------------------------------------------------------------------------------


@dataclass
class Adapter:
    """How to drive one adapter example through one route, in the suite's terms."""

    example: str  # file name under examples/
    requires: str  # module whose absence skips the adapter
    python_min: tuple[int, int]
    routes: dict[str, Callable[[Any, list[Call], Any, bool], tuple[dict[str, Outcome], Executed]]]
    ungated_routes: set[str] = field(default_factory=set)  # routes with a negative control
    #: route -> run the adapter's rewriting-schema fixture for one allowed SELECT (invariant 6)
    transforming: dict[str, Callable[[Any, Any], tuple[dict[str, Outcome], Executed]]] = field(
        default_factory=dict
    )
    #: route -> run the adapter's ``charge`` fixture for the given calls (invariant 8)
    omitted_default: dict[
        str, Callable[[Any, list[Call], Any], tuple[dict[str, Outcome], Executed]]
    ] = field(default_factory=dict)
    #: route -> run one allowed SELECT with a rewriting transform the framework places
    #: after the gate's seam (invariant 10)
    later_transform: dict[str, Callable[[Any, Any], tuple[dict[str, Outcome], Executed]]] = field(
        default_factory=dict
    )


def _refused(text: str) -> bool:
    return text.startswith("Recusal refused") or text.startswith("Recusal failed closed")


def _tool_call_dicts(calls: list[Call]) -> list[dict]:
    return [{"name": n, "args": a, "id": i, "type": "tool_call"} for n, a, i in calls]


def _from_tool_messages(messages: Any) -> dict[str, Outcome]:
    return {m.tool_call_id: (m.status == "error", str(m.content)) for m in messages}


def _from_text_outputs(
    outputs: dict[str, Any], text: Callable[[Any], str] = str
) -> dict[str, Outcome]:
    return {k: (_refused(text(v)), text(v)) for k, v in outputs.items()}


# langgraph ------------------------------------------------------------------------------------


def _langgraph_sync(m, calls, policy, gated):
    messages, executed = m.run_calls(_tool_call_dicts(calls), policy=policy, gated=gated)
    return _from_tool_messages(messages), executed


def _langgraph_async(m, calls, policy, gated):
    messages, executed = asyncio.run(
        m.run_calls_async(_tool_call_dicts(calls), policy=policy, gated=gated)
    )
    return _from_tool_messages(messages), executed


_TRANSFORM_CALL: list[Call] = [("run_sql", {"sql": SELECT}, "c1")]


def _langgraph_charge_sync(m, calls, policy):
    messages, executed = m.run_calls(
        _tool_call_dicts(calls), policy=policy, tools=[m.charge_tool()]
    )
    return _from_tool_messages(messages), executed


def _langgraph_charge_async(m, calls, policy):
    messages, executed = asyncio.run(
        m.run_calls_async(_tool_call_dicts(calls), policy=policy, tools=[m.charge_tool()])
    )
    return _from_tool_messages(messages), executed


def _langgraph_transforming_sync(m, policy):
    messages, executed = m.run_calls(
        _tool_call_dicts(_TRANSFORM_CALL), policy=policy, tools=[m.transforming_run_sql()]
    )
    return _from_tool_messages(messages), executed


def _langgraph_transforming_async(m, policy):
    messages, executed = asyncio.run(
        m.run_calls_async(
            _tool_call_dicts(_TRANSFORM_CALL), policy=policy, tools=[m.transforming_run_sql()]
        )
    )
    return _from_tool_messages(messages), executed


# langchain ------------------------------------------------------------------------------------


def _langchain_sync(m, calls, policy, gated):
    messages, executed = m.run_calls(_tool_call_dicts(calls), policy=policy, gated=gated)
    return _from_tool_messages(messages), executed


def _langchain_async(m, calls, policy, gated):
    kwargs = {} if gated else {"middleware": []}
    messages, executed = asyncio.run(m.arun_calls(_tool_call_dicts(calls), policy=policy, **kwargs))
    return _from_tool_messages(messages), executed


def _langchain_charge_sync(m, calls, policy):
    messages, executed = m.run_calls(
        _tool_call_dicts(calls), policy=policy, tools=[m.charge_tool()]
    )
    return _from_tool_messages(messages), executed


def _langchain_charge_async(m, calls, policy):
    messages, executed = asyncio.run(
        m.arun_calls(_tool_call_dicts(calls), policy=policy, tools=[m.charge_tool()])
    )
    return _from_tool_messages(messages), executed


def _langchain_transforming_sync(m, policy):
    messages, executed = m.run_calls(
        _tool_call_dicts(_TRANSFORM_CALL), policy=policy, tools=[m.transforming_run_sql()]
    )
    return _from_tool_messages(messages), executed


def _langchain_transforming_async(m, policy):
    messages, executed = asyncio.run(
        m.arun_calls(
            _tool_call_dicts(_TRANSFORM_CALL), policy=policy, tools=[m.transforming_run_sql()]
        )
    )
    return _from_tool_messages(messages), executed


# openai agents sdk, function tools --------------------------------------------------------------


def _openai_agents(streamed):
    def route(m, calls, policy, gated):
        outputs, executed = m.run_calls(calls, policy=policy, gated=gated, streamed=streamed)
        return _from_text_outputs(outputs), executed

    return route


def _openai_agents_charge(streamed):
    def route(m, calls, policy):
        outputs, executed = m.run_calls(
            calls, policy=policy, streamed=streamed, tools=[m.charge_tool()]
        )
        return _from_text_outputs(outputs), executed

    return route


def _openai_agents_transforming(streamed):
    def route(m, policy):
        outputs, executed = m.run_calls(
            _TRANSFORM_CALL, policy=policy, streamed=streamed, tools=[m.transforming_sql_tool()]
        )
        return _from_text_outputs(outputs), executed

    return route


# openai agents sdk, local mcp server --------------------------------------------------------------


def _openai_agents_mcp(m, calls, policy, gated):
    outputs, called = m.run_calls(calls, policy=policy, gated=gated)
    return _from_text_outputs(outputs, m._text), [(tool, args) for _, tool, args in called]


# openai responses api and chat completions, your own loop -----------------------------------------


def _openai_responses(m, calls, policy, gated):
    m.EXECUTED.clear()
    response = m.response_with([m.function_call(n, a, i) for n, a, i in calls])
    outputs = (
        m.gate_function_calls(response, m.TOOLS, policy)
        if gated
        else m.dispatch_ungated(response, m.TOOLS)
    )
    return _from_text_outputs({o["call_id"]: o["output"] for o in outputs}), list(m.EXECUTED)


def _standard_charge(route):
    """The adapter's own default tools already include ``charge``: drive the standard route."""

    def run(m, calls, policy):
        return route(m, calls, policy, True)

    return run


def _openai_chat(m, calls, policy, gated):
    m.EXECUTED.clear()
    completion = m.completion_with([m.tool_call(n, a, i) for n, a, i in calls])
    outputs = m.gate_tool_calls(completion, m.TOOLS, policy)
    return _from_text_outputs({o["tool_call_id"]: o["content"] for o in outputs}), list(m.EXECUTED)


# google adk ---------------------------------------------------------------------------------------


def _google_adk(via_plugin):
    def route(m, calls, policy, gated):
        responses, executed = m.run_calls(calls, policy=policy, gated=gated, via_plugin=via_plugin)
        outcomes = {}
        for call_id, response in responses.items():
            if isinstance(response, dict) and "error" in response:
                outcomes[call_id] = (True, str(response["error"]))
            else:
                outcomes[call_id] = (
                    False,
                    str(response.get("result") if isinstance(response, dict) else response),
                )
        return outcomes, executed

    return route


def _google_adk_later_transform(via_plugin):
    """One allowed SELECT with an agent before-tool callback that rewrites it into a DELETE,
    placed where the framework runs it relative to the gate: after the plugin (plugins run
    before agent callbacks), before the gate ``gated_agent`` appends last."""

    def run(m, policy):
        def rewrite(tool, args, tool_context):
            if tool.name == "run_sql":
                args["sql"] = DELETE
            return None

        m.EXECUTED.clear()
        agent = m.scripted_agent(_TRANSFORM_CALL).clone(update={"before_tool_callback": rewrite})
        if via_plugin:
            responses = asyncio.run(m.run_agent(agent, plugins=[m.RecusalPlugin(policy)]))
        else:
            responses = asyncio.run(m.run_agent(m.gated_agent(agent, policy)))
        outcomes = {}
        for call_id, response in responses.items():
            if isinstance(response, dict) and "error" in response:
                outcomes[call_id] = (True, str(response["error"]))
            else:
                outcomes[call_id] = (False, str(response))
        return outcomes, list(m.EXECUTED)

    return run


# crewai -------------------------------------------------------------------------------------------


def _crewai(m, calls, policy, gated):
    responses, executed = m.run_calls(calls, policy=policy, gated=gated)
    return _from_text_outputs(responses), executed


def _crewai_transforming(m, policy):
    responses, executed = m.run_calls(
        _TRANSFORM_CALL, policy=policy, tools=[m.transforming_run_sql()]
    )
    return _from_text_outputs(responses), executed


# pydantic ai --------------------------------------------------------------------------------------


def _pydantic_ai(via_hooks):
    def route(m, calls, policy, gated):
        responses, executed = m.run_calls(calls, policy=policy, gated=gated, via_hooks=via_hooks)
        return {
            k: (outcome == "failed", str(content)) for k, (outcome, content) in responses.items()
        }, executed

    return route


ADAPTERS: dict[str, Adapter] = {
    "langgraph": Adapter(
        "langgraph_gate.py",
        "langgraph",
        (3, 10),
        {"sync": _langgraph_sync, "async": _langgraph_async},
        {"sync", "async"},
        transforming={"sync": _langgraph_transforming_sync, "async": _langgraph_transforming_async},
        omitted_default={"sync": _langgraph_charge_sync, "async": _langgraph_charge_async},
    ),
    "langchain": Adapter(
        "langchain_agent_gate.py",
        "langchain",
        (3, 10),
        {"sync": _langchain_sync, "async": _langchain_async},
        {"sync", "async"},
        transforming={
            "sync": _langchain_transforming_sync,
            "async": _langchain_transforming_async,
        },
        omitted_default={"sync": _langchain_charge_sync, "async": _langchain_charge_async},
    ),
    "openai_agents": Adapter(
        "openai_agents_gate.py",
        "agents.testing",
        (3, 10),
        {"run": _openai_agents(False), "streamed": _openai_agents(True)},
        {"run", "streamed"},
        transforming={
            "run": _openai_agents_transforming(False),
            "streamed": _openai_agents_transforming(True),
        },
        omitted_default={
            "run": _openai_agents_charge(False),
            "streamed": _openai_agents_charge(True),
        },
    ),
    "openai_agents_mcp": Adapter(
        "openai_agents_mcp_gate.py",
        "agents.mcp",
        (3, 10),
        {"local_server": _openai_agents_mcp},
        {"local_server"},
    ),
    "openai_responses": Adapter(
        "openai_responses_gate.py",
        "openai",
        (3, 10),
        {"responses": _openai_responses, "chat": _openai_chat},
        {"responses"},
        omitted_default={
            "responses": _standard_charge(_openai_responses),
            "chat": _standard_charge(_openai_chat),
        },
    ),
    "google_adk": Adapter(
        "google_adk_gate.py",
        "google.adk",
        (3, 11),
        {"callback": _google_adk(False), "plugin": _google_adk(True)},
        {"callback", "plugin"},
        omitted_default={
            "callback": _standard_charge(_google_adk(False)),
            "plugin": _standard_charge(_google_adk(True)),
        },
        later_transform={
            "callback": _google_adk_later_transform(False),
            "plugin": _google_adk_later_transform(True),
        },
    ),
    "crewai": Adapter(
        "crewai_gate.py",
        "crewai",
        (3, 10),
        {"hooks": _crewai},
        {"hooks"},
        transforming={"hooks": _crewai_transforming},
        omitted_default={"hooks": _standard_charge(_crewai)},
    ),
    "pydantic_ai": Adapter(
        "pydantic_ai_gate.py",
        "pydantic_ai",
        (3, 11),
        {"capability": _pydantic_ai(False), "hooks": _pydantic_ai(True)},
        {"capability", "hooks"},
    ),
}

ROWS = [(name, route) for name, adapter in ADAPTERS.items() for route in adapter.routes]


@pytest.fixture(scope="module")
def modules():
    """Each adapter's example module, loaded once, or the reason it is skipped."""
    loaded: dict[str, Any] = {}
    for name, adapter in ADAPTERS.items():
        if sys.version_info < adapter.python_min:
            loaded[name] = (
                f"{adapter.example} requires Python {adapter.python_min[0]}.{adapter.python_min[1]}+"
            )
            continue
        if importlib.util.find_spec(adapter.requires.split(".")[0]) is None:
            loaded[name] = f"{adapter.requires} is not installed"
            continue
        try:
            importlib.import_module(adapter.requires)
        except Exception as exc:  # noqa: BLE001, an import error is a skip reason, not a failure here
            loaded[name] = f"{adapter.requires} is not importable: {exc}"
            continue
        spec = importlib.util.spec_from_file_location(
            f"recusal_conformance_{name}", EXAMPLES / adapter.example
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def _drive(modules, name, route):
    module = modules[name]
    if isinstance(module, str):
        pytest.skip(module)
    return module, ADAPTERS[name].routes[route]


class Recording:
    """A policy wrapper that records every (name, arguments) it was asked to judge."""

    def __init__(self, policy):
        self.policy = policy
        self.seen: list[tuple[str, Any]] = []

    def __call__(self, name, args):
        self.seen.append((name, dict(args) if isinstance(args, dict) else args))
        return self.policy(name, args)


def _broken(name, args):
    raise RuntimeError("policy exploded")


# --- the invariants ------------------------------------------------------------------------------


@pytest.mark.parametrize("name, route", ROWS, ids=[f"{n}-{r}" for n, r in ROWS])
def test_1_the_arguments_the_policy_judged_are_the_arguments_the_tool_ran_with(
    modules, name, route
):
    module, run = _drive(modules, name, route)
    policy = Recording(module.sql_scope_policy)
    outcomes, executed = run(module, [("run_sql", {"sql": SELECT}, "c1")], policy, True)
    assert policy.seen == [("run_sql", {"sql": SELECT})]
    assert executed == [("run_sql", {"sql": SELECT})]
    assert outcomes["c1"][0] is False, outcomes


@pytest.mark.parametrize("name, route", ROWS, ids=[f"{n}-{r}" for n, r in ROWS])
def test_2_a_refusal_runs_nothing_and_its_reason_reaches_the_model(modules, name, route):
    module, run = _drive(modules, name, route)
    outcomes, executed = run(
        module, [("run_sql", {"sql": DELETE}, "c1")], module.sql_scope_policy, True
    )
    assert executed == []
    refused, text = outcomes["c1"]
    assert refused and "Recusal refused `run_sql`" in text and "top-level WHERE" in text


def _normalizer(_name, args):
    """A composed leaf that rewrites its argument: formerly the next leaf judged the rewrite."""
    args["sql"] = SELECT
    return []


@pytest.mark.parametrize("mutator_first", [True, False], ids=["mutator-first", "check-first"])
@pytest.mark.parametrize("name, route", ROWS, ids=[f"{n}-{r}" for n, r in ROWS])
def test_2b_a_mutating_composed_leaf_cannot_launder_the_input_for_a_later_check(
    modules, name, route, mutator_first
):
    """Adapters review, finding 2: with the normalizer listed first, the SQL check saw
    SELECT and the adapter executed DELETE. Every leaf now judges its own copy."""
    from examples.composed_policies import compose_policies

    module, run = _drive(modules, name, route)
    check = Recording(module.sql_scope_policy)
    leaves = (_normalizer, check) if mutator_first else (check, _normalizer)
    outcomes, executed = run(
        module, [("run_sql", {"sql": DELETE}, "c1")], compose_policies(*leaves), True
    )
    # every judgment the check made saw the original (the CrewAI gate judges a blocked call
    # a second time to explain it, so the count is that route's, not this invariant's)
    assert check.seen and all(s == ("run_sql", {"sql": DELETE}) for s in check.seen), check.seen
    assert executed == []
    refused, text = outcomes["c1"]
    assert refused and "Recusal refused `run_sql`" in text


@pytest.mark.parametrize("name, route", ROWS, ids=[f"{n}-{r}" for n, r in ROWS])
def test_3_calls_in_one_turn_are_judged_separately(modules, name, route):
    module, run = _drive(modules, name, route)
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": DELETE}, "c2"),
        ("run_sql", {"sql": SELECT}, "c3"),
    ]
    outcomes, executed = run(module, calls, module.sql_scope_policy, True)
    assert sorted(executed, key=repr) == sorted(
        [("echo", {"text": "hello"}), ("run_sql", {"sql": SELECT})], key=repr
    )
    assert [outcomes[c][0] for c in ("c1", "c2", "c3")] == [False, True, False], outcomes


@pytest.mark.parametrize("name, route", ROWS, ids=[f"{n}-{r}" for n, r in ROWS])
def test_4_a_policy_that_raises_fails_closed(modules, name, route):
    module, run = _drive(modules, name, route)
    outcomes, executed = run(module, [("echo", {"text": "unsafe"}, "c1")], _broken, True)
    assert executed == []
    refused, text = outcomes["c1"]
    assert refused and "failed closed" in text and "policy exploded" in text


@pytest.mark.parametrize("name, route", ROWS, ids=[f"{n}-{r}" for n, r in ROWS])
def test_5_negative_control_the_same_refused_call_runs_ungated(modules, name, route):
    if route not in ADAPTERS[name].ungated_routes:
        pytest.skip(f"{name} {route} has no ungated form in its example")
    module, run = _drive(modules, name, route)
    outcomes, executed = run(
        module, [("run_sql", {"sql": DELETE}, "c1")], module.sql_scope_policy, False
    )
    assert executed == [("run_sql", {"sql": DELETE})]
    assert outcomes["c1"][0] is False, outcomes


TRANSFORM_ROWS = [(n, r) for n, a in ADAPTERS.items() for r in a.transforming]


@pytest.mark.parametrize("name, route", TRANSFORM_ROWS, ids=[f"{n}-{r}" for n, r in TRANSFORM_ROWS])
def test_6_a_schema_that_changes_an_allowed_proposal_never_runs_the_changed_value(
    modules, name, route
):
    module = modules[name]
    if isinstance(module, str):
        pytest.skip(module)
    policy = Recording(module.sql_scope_policy)
    outcomes, executed = ADAPTERS[name].transforming[route](module, policy)
    assert executed == [], executed
    refused, text = outcomes["c1"]
    assert refused and "Recusal refused `run_sql`" in text, text
    # the gate judged the value the tool would have executed, or judged nothing because it
    # could not know that value; it never judged the proposal and let the rewrite run
    assert policy.seen in ([], [("run_sql", {"sql": DELETE})]), policy.seen


DEFAULT_ROWS = [(n, r) for n, a in ADAPTERS.items() for r in a.omitted_default]


def _budget(name, args):
    from recusal import Finding

    if args.get("amount", 0) > 10:
        return [Finding.fail("budget", severity="CRITICAL", message="over 10")]
    return []


@pytest.mark.parametrize("name, route", DEFAULT_ROWS, ids=[f"{n}-{r}" for n, r in DEFAULT_ROWS])
def test_8_a_default_the_proposal_omitted_is_judged_at_the_value_the_callable_uses(
    modules, name, route
):
    module = modules[name]
    if isinstance(module, str):
        pytest.skip(module)
    run = ADAPTERS[name].omitted_default[route]
    policy = Recording(_budget)
    outcomes, executed = run(module, [("charge", {}, "c1")], policy)
    assert policy.seen == [("charge", {"amount": 100})], policy.seen
    assert type(policy.seen[0][1]["amount"]) is int, "judged as the int the callable uses"
    assert executed == [], executed
    refused, text = outcomes["c1"]
    assert refused and "Recusal refused `charge`" in text and "over 10" in text, text
    # the positive control: an explicit value under the cap is judged and runs as given
    policy = Recording(_budget)
    outcomes, executed = run(module, [("charge", {"amount": 5}, "c2")], policy)
    assert policy.seen == [("charge", {"amount": 5})]
    assert executed == [("charge", {"amount": 5})], executed
    assert outcomes["c2"][0] is False


def test_the_omitted_default_rows_cover_every_adapter_the_docstring_names():
    assert {n for n, _ in DEFAULT_ROWS} == {
        "langgraph",
        "langchain",
        "openai_agents",
        "openai_responses",
        "google_adk",
        "crewai",
    }
    for name in {n for n, _ in DEFAULT_ROWS}:
        assert set(ADAPTERS[name].omitted_default) == set(ADAPTERS[name].routes), name


def test_the_transforming_rows_cover_every_adapter_the_docstring_names():
    assert {n for n, _ in TRANSFORM_ROWS} == {"langgraph", "langchain", "openai_agents", "crewai"}
    for name in ("langgraph", "langchain", "openai_agents", "crewai"):
        assert set(ADAPTERS[name].transforming) == set(ADAPTERS[name].routes), name


LATER_ROWS = [(n, r) for n, a in ADAPTERS.items() for r in a.later_transform]


@pytest.mark.parametrize("name, route", LATER_ROWS, ids=[f"{n}-{r}" for n, r in LATER_ROWS])
def test_10_a_transform_the_framework_runs_after_the_gates_seam_cannot_change_what_runs(
    modules, name, route
):
    module = modules[name]
    if isinstance(module, str):
        pytest.skip(module)
    policy = Recording(module.sql_scope_policy)
    outcomes, executed = ADAPTERS[name].later_transform[route](module, policy)
    assert executed == [], executed
    refused, text = outcomes["c1"]
    assert refused and "Recusal refused `run_sql`" in text, text
    # the gate judged the rewritten value (it was placed last), or judged nothing because it
    # could not be placed last; it never judged the proposal and let the rewrite run
    assert policy.seen in ([], [("run_sql", {"sql": DELETE})]), policy.seen


def test_the_later_transform_rows_cover_every_route_the_docstring_names():
    assert {n for n, _ in LATER_ROWS} == {"google_adk"}
    assert set(ADAPTERS["google_adk"].later_transform) == set(ADAPTERS["google_adk"].routes)


class Mutating:
    """A policy that judges correctly, records what it saw, then writes to what it was handed.

    The write is the bug under test: an accidental assignment after the evidence was
    gathered. ``seen`` holds a deep copy taken BEFORE the write, so it states what the
    policy observed, not what it left behind.
    """

    def __init__(self, policy):
        self.policy = policy
        self.seen: list[tuple[str, Any]] = []

    def __call__(self, name, args):
        self.seen.append((name, copy.deepcopy(args)))
        findings = self.policy(name, args)
        if isinstance(args, dict):
            args["sql"] = DELETE
        return findings


@pytest.mark.parametrize("name, route", ROWS, ids=[f"{n}-{r}" for n, r in ROWS])
def test_11_a_policys_write_to_what_it_judged_cannot_change_what_runs(modules, name, route):
    module, run = _drive(modules, name, route)
    policy = Mutating(module.sql_scope_policy)
    outcomes, executed = run(module, [("run_sql", {"sql": SELECT}, "c1")], policy, True)
    assert policy.seen == [("run_sql", {"sql": SELECT})], policy.seen
    assert executed == [("run_sql", {"sql": SELECT})], executed
    assert outcomes["c1"][0] is False, outcomes


@pytest.mark.parametrize("name, route", ROWS, ids=[f"{n}-{r}" for n, r in ROWS])
def test_11_control_the_same_policy_still_refuses_a_proposal_it_should(modules, name, route):
    # the detached copy changes what a policy's write reaches, never what the policy decides
    module, run = _drive(modules, name, route)
    policy = Mutating(module.sql_scope_policy)
    outcomes, executed = run(module, [("run_sql", {"sql": DELETE}, "c1")], policy, True)
    assert executed == [], executed
    assert outcomes["c1"][0] is True and "Recusal refused `run_sql`" in outcomes["c1"][1]


# --- the registry drift locks --------------------------------------------------------------------


def _works_with_tested_examples() -> set[str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    found = set()
    for line in table.splitlines():
        if "| Tested |" not in line:
            continue
        # README links are absolute (PyPI renders the README without the repository)
        found.update(
            re.findall(
                r"\((?:https://github\.com/philpaz/recusal/blob/main/)?examples/([a-z_]+_gate\.py)\)",
                line,
            )
        )
    return found


def test_every_tested_gate_example_is_registered_here():
    tested = _works_with_tested_examples()
    registered = {adapter.example for adapter in ADAPTERS.values()}
    assert tested, "the Works-with table lists no *_gate.py example"
    assert tested <= registered, (
        f"Tested but not in the conformance registry: {sorted(tested - registered)}"
    )
    assert registered <= tested, (
        f"registered but not Tested in the README: {sorted(registered - tested)}"
    )
    for adapter in ADAPTERS.values():
        assert (EXAMPLES / adapter.example).is_file(), adapter.example


def test_this_file_runs_in_every_runtime_lane():
    this = "tests/test_adapter_conformance.py"
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lanes = [
        line
        for line in ci.splitlines()
        if "run: pytest -q tests/test_" in line and "_gate.py" in line
    ]
    assert lanes, "no runtime lane found in ci.yml"
    for line in lanes:
        assert this in line, line.strip()
    compat = (ROOT / ".github" / "workflows" / "compat.yml").read_text(encoding="utf-8")
    rows = [line for line in compat.splitlines() if line.strip().startswith("tests:")]
    assert rows, "no runtime row found in compat.yml"
    for line in rows:
        assert this in line, line.strip()
