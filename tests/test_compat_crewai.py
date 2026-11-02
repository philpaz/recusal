"""What CrewAI's tool-call hooks can see, pinned so a new CrewAI release cannot change it quietly.

`tests/test_crewai_gate.py` proves the example's behavior. This file pins the CrewAI surface
that behavior depends on: the hook registry functions and their global scope, the context a
hook receives, that ``False`` from a before hook blocks the call and CrewAI answers with its
fixed line, that after hooks run on a blocked call and a string return replaces the answer,
that hooks run in registration order over one shared argument dict, and that a tool-call list
in the OpenAI shape drives the native tool path. An unhooked crew is the negative control, so
a probe that stopped observing anything would fail rather than pass.

It runs in the pinned CI jobs (the exact version the README names) and weekly against the
newest CrewAI in `.github/workflows/compat.yml`. A failure there means a CrewAI release changed
what the gate can see; STABILITY.md says what happens next.
"""

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "crewai_gate.py"

SELECT = "SELECT * FROM t"
DELETE = "DELETE FROM t"

pytestmark = pytest.mark.skipif(sys.version_info < (3, 10), reason="CrewAI requires Python 3.10+")


@pytest.fixture
def crew(request):
    pytest.importorskip("crewai", reason="crewai is not installed")
    import importlib.util

    spec = importlib.util.spec_from_file_location("recusal_crewai_compat_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from crewai.hooks import clear_after_tool_call_hooks, clear_before_tool_call_hooks

    clear_before_tool_call_hooks()
    clear_after_tool_call_hooks()
    yield module
    clear_before_tool_call_hooks()
    clear_after_tool_call_hooks()


def _run(crew, calls, tools=None):
    crew.EXECUTED.clear()
    return crew.run_crew(crew.scripted_agent(calls, tools=tools)), list(crew.EXECUTED)


def test_the_registry_functions_the_example_relies_on_exist():
    pytest.importorskip("crewai", reason="crewai is not installed")
    from crewai import hooks

    for name in (
        "register_before_tool_call_hook",
        "register_after_tool_call_hook",
        "unregister_before_tool_call_hook",
        "unregister_after_tool_call_hook",
        "get_before_tool_call_hooks",
        "get_after_tool_call_hooks",
        "clear_before_tool_call_hooks",
        "clear_after_tool_call_hooks",
    ):
        assert callable(getattr(hooks, name)), name
    assert "hook" in inspect.signature(hooks.register_before_tool_call_hook).parameters


def test_a_before_hook_receives_the_tool_name_and_the_argument_dict_the_tool_runs_with(crew):
    from crewai.hooks import register_before_tool_call_hook
    from crewai.hooks.tool_hooks import ToolCallHookContext

    seen = []

    def probe(context):
        seen.append(
            (
                isinstance(context, ToolCallHookContext),
                context.tool_name,
                dict(context.tool_input),
                type(context.tool).__name__,
                getattr(context.agent, "role", None),
            )
        )
        return None

    register_before_tool_call_hook(probe)
    _, executed = _run(crew, [("run_sql", {"sql": SELECT}, "c1")])
    assert seen == [(True, "run_sql", {"sql": SELECT}, "CrewStructuredTool", "assistant")]
    assert executed == [("run_sql", {"sql": SELECT})]


def test_false_from_a_before_hook_blocks_the_call_with_crewais_fixed_line(crew):
    from crewai.hooks import register_before_tool_call_hook

    register_before_tool_call_hook(lambda context: False)
    responses, executed = _run(crew, [("run_sql", {"sql": SELECT}, "c1")])
    assert executed == []
    assert responses["c1"] == "Tool execution blocked by hook. Tool: run_sql"
    assert responses["c1"].startswith(crew.BLOCKED_PREFIX)


def test_none_from_a_before_hook_runs_the_tool(crew):
    from crewai.hooks import register_before_tool_call_hook

    register_before_tool_call_hook(lambda context: None)
    responses, executed = _run(crew, [("run_sql", {"sql": SELECT}, "c1")])
    assert executed == [("run_sql", {"sql": SELECT})]
    assert responses["c1"] == f"recorded SQL: {SELECT}"


def test_after_hooks_run_on_a_blocked_call_and_a_string_replaces_the_answer(crew):
    from crewai.hooks import register_after_tool_call_hook, register_before_tool_call_hook

    seen = []

    def after(context):
        seen.append((context.tool_name, str(context.tool_result)))
        return "replaced by the after hook"

    register_before_tool_call_hook(lambda context: False)
    register_after_tool_call_hook(after)
    responses, executed = _run(crew, [("run_sql", {"sql": SELECT}, "c1")])
    assert executed == []
    assert seen == [("run_sql", "Tool execution blocked by hook. Tool: run_sql")]
    assert responses["c1"] == "replaced by the after hook"


def test_an_after_hook_returning_none_keeps_the_answer(crew):
    from crewai.hooks import register_after_tool_call_hook

    register_after_tool_call_hook(lambda context: None)
    responses, _ = _run(crew, [("echo", {"text": "hi"}, "c1")])
    assert responses["c1"] == "hi"


def test_before_hooks_run_in_registration_order_over_one_shared_dict(crew):
    from crewai.hooks import register_before_tool_call_hook

    order = []

    def first(context):
        order.append(("first", dict(context.tool_input)))
        context.tool_input["text"] = "changed by first"
        return None

    def second(context):
        order.append(("second", dict(context.tool_input)))
        return None

    register_before_tool_call_hook(first)
    register_before_tool_call_hook(second)
    _, executed = _run(crew, [("echo", {"text": "original"}, "c1")])
    assert order == [("first", {"text": "original"}), ("second", {"text": "changed by first"})]
    assert executed == [("echo", {"text": "changed by first"})]


def test_a_hook_that_raises_does_not_block_and_the_tool_runs(crew):
    """Pinned as observed: CrewAI swallows a hook exception and continues. The gate therefore
    never relies on raising; it returns False, and ``decide`` fails closed on its own."""
    from crewai.hooks import register_before_tool_call_hook

    def boom(context):
        raise RuntimeError("hook exploded")

    register_before_tool_call_hook(boom)
    responses, executed = _run(crew, [("run_sql", {"sql": DELETE}, "c1")])
    assert executed == [("run_sql", {"sql": DELETE})]
    assert responses["c1"] == f"recorded SQL: {DELETE}"


def test_the_hooks_are_process_global_and_cover_a_second_crew(crew):
    from crewai.hooks import register_before_tool_call_hook

    seen = []
    register_before_tool_call_hook(lambda context: seen.append(context.tool_name))
    _run(crew, [("echo", {"text": "a"}, "c1")])
    _run(crew, [("echo", {"text": "b"}, "c2")])
    assert seen == ["echo", "echo"]


def test_the_openai_tool_call_shape_drives_the_native_tool_path(crew):
    from crewai.utilities.agent_utils import extract_tool_call_info, is_tool_call_list

    call = crew._tool_call("c9", "run_sql", {"sql": SELECT})
    assert is_tool_call_list([call])
    call_id, name, arguments = extract_tool_call_info(call)
    assert (call_id, name) == ("c9", "run_sql")
    assert arguments == '{"sql": "SELECT * FROM t"}'


def test_negative_control_an_unhooked_crew_runs_its_tools(crew):
    from crewai.hooks import get_after_tool_call_hooks, get_before_tool_call_hooks

    assert get_before_tool_call_hooks() == [] and get_after_tool_call_hooks() == []
    responses, executed = _run(crew, [("run_sql", {"sql": DELETE}, "c1")])
    assert executed == [("run_sql", {"sql": DELETE})]
    assert responses["c1"] == f"recorded SQL: {DELETE}"
