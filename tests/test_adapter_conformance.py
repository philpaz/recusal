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
   arguments the server receives. Google ADK and CrewAI have not been examined for a
   validation layer between their hook and the function; they are listed here as not yet
   covered, not as covered.

Every route an adapter offers (sync and async, callback and plugin, run and streamed) is a
separate row, because each is a separate code path through the framework.

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


# crewai -------------------------------------------------------------------------------------------


def _crewai(m, calls, policy, gated):
    responses, executed = m.run_calls(calls, policy=policy, gated=gated)
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
    ),
    "google_adk": Adapter(
        "google_adk_gate.py",
        "google.adk",
        (3, 11),
        {"callback": _google_adk(False), "plugin": _google_adk(True)},
        {"callback", "plugin"},
    ),
    "crewai": Adapter("crewai_gate.py", "crewai", (3, 10), {"hooks": _crewai}, {"hooks"}),
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


def test_the_transforming_rows_cover_every_adapter_the_docstring_names():
    assert {n for n, _ in TRANSFORM_ROWS} == {"langgraph", "langchain", "openai_agents"}
    for name in ("langgraph", "langchain", "openai_agents"):
        assert set(ADAPTERS[name].transforming) == set(ADAPTERS[name].routes), name


# --- the registry drift locks --------------------------------------------------------------------


def _works_with_tested_examples() -> set[str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    found = set()
    for line in table.splitlines():
        if "| Tested |" not in line:
            continue
        found.update(re.findall(r"\(examples/([a-z_]+_gate\.py)\)", line))
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
