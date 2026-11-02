"""What the Google ADK before-tool callback can see, pinned so a new ADK release cannot change it quietly.

`tests/test_google_adk_gate.py` proves the example's behavior. This file pins the ADK surface
that behavior depends on: the callback's signature, that a dict return answers the call and
None runs the tool, that callbacks run in order and stop at the first answer, that plugin
callbacks run before agent callbacks, that every tool class the example relies on passes
through the callback, and that the agent fields `gated_agent` reads exist. An ungated agent is
the negative control, so a probe that stopped observing anything would fail rather than pass.

It runs in the pinned CI jobs (the exact versions the README names) and weekly against the
newest ADK in `.github/workflows/compat.yml`. A failure there means an ADK release changed
what the gate can see; STABILITY.md says what happens next.
"""

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "google_adk_gate.py"

SELECT = "SELECT * FROM t"
DELETE = "DELETE FROM t"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="the Google ADK requires Python 3.11+"
)


@pytest.fixture(scope="module")
def adk():
    pytest.importorskip("google.adk", reason="google-adk is not installed")
    import importlib.util

    spec = importlib.util.spec_from_file_location("recusal_google_adk_compat_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(adk, agent, plugins=None):
    adk.EXECUTED.clear()
    return asyncio.run(adk.run_agent(agent, plugins=plugins)), list(adk.EXECUTED)


def test_the_callback_receives_tool_args_and_context_and_can_answer_the_call(adk):
    from google.adk.agents.llm_agent import _SingleBeforeToolCallback
    from google.adk.tools import BaseTool, ToolContext

    params = _SingleBeforeToolCallback.__args__[:-1]
    assert params[0] is BaseTool and params[2] is ToolContext
    seen = []

    def probe(tool, args, tool_context):
        seen.append((isinstance(tool, BaseTool), args, isinstance(tool_context, ToolContext)))
        return {"answered": "by the callback"}

    agent = adk.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]).clone(
        update={"before_tool_callback": probe}
    )
    responses, executed = _run(adk, agent)
    assert seen == [(True, {"sql": SELECT}, True)]
    assert responses["c1"] == {"answered": "by the callback"}
    assert executed == []  # a dict answer means the tool never runs


def test_none_from_the_callback_runs_the_tool_with_the_same_args(adk):
    agent = adk.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]).clone(
        update={"before_tool_callback": lambda tool, args, ctx: None}
    )
    responses, executed = _run(adk, agent)
    assert executed == [("run_sql", {"sql": SELECT})]
    assert responses["c1"] == {"result": f"recorded SQL: {SELECT}"}


def test_callbacks_run_in_order_and_stop_at_the_first_answer(adk):
    order = []

    def first(tool, args, ctx):
        order.append("first")
        return None

    def second(tool, args, ctx):
        order.append("second")
        return {"stopped": "here"}

    def third(tool, args, ctx):
        order.append("third")
        return None

    agent = adk.scripted_agent([("echo", {"text": "x"}, "c1")]).clone(
        update={"before_tool_callback": [first, second, third]}
    )
    responses, executed = _run(adk, agent)
    assert order == ["first", "second"]
    assert responses["c1"] == {"stopped": "here"}
    assert executed == []


def test_a_callback_sees_and_can_change_the_args_the_tool_runs_with(adk):
    def rewrite(tool, args, ctx):
        args["text"] = "rewritten"
        return None

    agent = adk.scripted_agent([("echo", {"text": "original"}, "c1")]).clone(
        update={"before_tool_callback": rewrite}
    )
    responses, executed = _run(adk, agent)
    assert executed == [("echo", {"text": "rewritten"})]


def test_plugin_callbacks_run_before_agent_callbacks_and_can_answer(adk):
    from google.adk.plugins import BasePlugin

    order = []

    class Probe(BasePlugin):
        async def before_tool_callback(self, *, tool, tool_args, tool_context):
            order.append(("plugin", tool.name, dict(tool_args)))
            return {"answered": "by the plugin"}

    def agent_cb(tool, args, ctx):
        order.append(("agent", tool.name))
        return None

    agent = adk.scripted_agent([("run_sql", {"sql": DELETE}, "c1")]).clone(
        update={"before_tool_callback": agent_cb}
    )
    responses, executed = _run(adk, agent, plugins=[Probe(name="probe")])
    assert order == [("plugin", "run_sql", {"sql": DELETE})]
    assert responses["c1"] == {"answered": "by the plugin"}
    assert executed == []


def test_a_callback_that_raises_stops_the_call(adk):
    def boom(tool, args, ctx):
        raise RuntimeError("callback exploded")

    agent = adk.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]).clone(
        update={"before_tool_callback": boom}
    )
    # the ADK surfaces a raising callback as a failed run, not as a tool response: the
    # tool never ran, and the caller sees the failure instead of a silent pass
    with pytest.raises(Exception, match="callback exploded"):
        _run(adk, agent)
    assert adk.EXECUTED == []


@pytest.mark.parametrize("kind", ["function", "long_running"])
def test_every_tool_class_the_example_relies_on_passes_through_the_callback(adk, kind):
    from google.adk.tools import FunctionTool, LongRunningFunctionTool

    seen = []

    def probe(tool, args, ctx):
        seen.append((type(tool).__name__, tool.name, args))
        return None

    def worker(text: str) -> str:
        """Record text."""
        adk.EXECUTED.append(("worker", {"text": text}))
        return text

    tool = FunctionTool(worker) if kind == "function" else LongRunningFunctionTool(worker)
    agent = adk.scripted_agent([("worker", {"text": "x"}, "c1")], tools=[tool]).clone(
        update={"before_tool_callback": probe}
    )
    _, executed = _run(adk, agent)
    assert seen == [(type(tool).__name__, "worker", {"text": "x"})]
    assert executed == [("worker", {"text": "x"})]


def test_a_missing_args_field_reaches_the_callback_as_an_empty_dict(adk):
    seen = []

    def probe(tool, args, ctx):
        seen.append(args)
        return None

    agent = adk.scripted_agent([("echo", None, "c1")]).clone(update={"before_tool_callback": probe})
    _run(adk, agent)
    assert seen == [{}]


def test_the_agent_fields_gated_agent_reads_exist_with_these_names(adk):
    from google.adk.agents import BaseAgent, LlmAgent
    from google.adk.tools import AgentTool

    assert {"code_executor", "before_tool_callback", "tools", "sub_agents"} <= set(
        LlmAgent.model_fields
    )
    assert "sub_agents" in BaseAgent.model_fields
    assert "update" in inspect.signature(BaseAgent.clone).parameters
    params = inspect.signature(AgentTool.__init__).parameters
    assert "agent" in params and "skip_summarization" in params
    child = adk.scripted_agent([])
    assert AgentTool(child).agent is child


def test_a_transfer_runs_the_sub_agents_tools_under_its_own_callbacks(adk):
    """The fact gated_agent's recursion and RecusalPlugin exist for: the parent's callback
    sees only transfer_to_agent; the child's tool call runs under the child's callbacks."""
    from google.adk.agents import LlmAgent
    from google.adk.models import LlmResponse
    from google.adk.tools import FunctionTool
    from google.genai import types

    seen = []

    def parent_cb(tool, args, ctx):
        seen.append(tool.name)
        return None

    class Transfer(adk.ScriptedLlm):
        async def generate_content_async(self, llm_request, stream=False):
            self.turns += 1
            if self.turns == 1:
                parts = [
                    types.Part(
                        function_call=types.FunctionCall(
                            id="t1", name="transfer_to_agent", args={"agent_name": "child"}
                        )
                    )
                ]
            else:
                parts = [types.Part(text="done")]
            yield LlmResponse(content=types.Content(role="model", parts=parts))

    child = LlmAgent(
        name="child",
        model=adk.ScriptedLlm(model="s", calls=[("run_sql", {"sql": DELETE}, "s1")]),
        tools=[FunctionTool(adk.run_sql)],
    )
    parent = LlmAgent(
        name="parent", model=Transfer(model="s"), sub_agents=[child], before_tool_callback=parent_cb
    )
    _, executed = _run(adk, parent)
    assert seen == ["transfer_to_agent"]
    assert executed == [("run_sql", {"sql": DELETE})]


def test_negative_control_an_ungated_agent_runs_its_listed_tools(adk):
    agent = adk.scripted_agent([("run_sql", {"sql": DELETE}, "c1")])
    assert agent.before_tool_callback is None
    responses, executed = _run(adk, agent)
    assert executed == [("run_sql", {"sql": DELETE})]
    assert responses["c1"] == {"result": f"recorded SQL: {DELETE}"}
