"""What a Pydantic AI before_tool_execute hook can see, pinned so a new release cannot change it quietly.

`tests/test_pydantic_ai_gate.py` proves the example's behavior. This file pins the Pydantic
AI surface that behavior depends on: the hook's keyword signature and that `args` is the
validated dict the tool runs with, that `ToolFailed` from the hook answers the call with a
failed tool result the tool never ran for and spends no retry budget, that the returned dict
is what runs, that hooks run in capability order with `innermost` sorted last, that a hook
that raises stops the call, that every tool registration route the example relies on
(`tools=`, `@agent.tool_plain`, `toolsets=`, a per-run capability) passes through the hook,
that an argument-validation failure never reaches the hook and never runs the tool, and that
the agent fields `is_gated` reads exist. An ungated agent is the negative control, so a
probe that stopped observing anything would fail rather than pass.

It runs in the pinned CI jobs (the exact version the README names) and weekly against the
newest Pydantic AI in `.github/workflows/compat.yml`. A failure there means a release
changed what the gate can see; STABILITY.md says what happens next.
"""

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pydantic_ai_gate.py"

SELECT = "SELECT * FROM t"
DELETE = "DELETE FROM t"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Pydantic AI requires Python 3.11+"
)


@pytest.fixture(scope="module")
def pai():
    pytest.importorskip("pydantic_ai", reason="pydantic-ai-slim is not installed")
    import importlib.util

    spec = importlib.util.spec_from_file_location("recusal_pydantic_ai_compat_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(pai, agent, **kwargs):
    pai.EXECUTED.clear()
    result = agent.run_sync("go", **kwargs)
    return pai.responses_of(result), list(pai.EXECUTED)


def _hooks(func):
    from pydantic_ai.capabilities import Hooks

    return Hooks(before_tool_execute=func)


def test_the_hook_receives_the_call_the_tool_def_and_the_validated_args_by_keyword(pai):
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition

    params = inspect.signature(AbstractCapability.before_tool_execute).parameters
    assert list(params) == ["self", "ctx", "call", "tool_def", "args"]
    assert all(
        params[name].kind is inspect.Parameter.KEYWORD_ONLY for name in ("call", "tool_def", "args")
    )
    seen = []

    async def probe(ctx, *, call, tool_def, args):
        seen.append(
            (
                isinstance(call, ToolCallPart),
                call.tool_name,
                isinstance(tool_def, ToolDefinition),
                args,
            )
        )
        return args

    agent = pai.scripted_agent([("run_sql", {"sql": SELECT}, "c1")], capabilities=[_hooks(probe)])
    responses, executed = _run(pai, agent)
    assert seen == [(True, "run_sql", True, {"sql": SELECT})]
    assert executed == [("run_sql", {"sql": SELECT})]
    assert responses["c1"] == ("success", f"recorded SQL: {SELECT}")


def test_tool_failed_from_the_hook_answers_the_call_without_running_the_tool(pai):
    from pydantic_ai.exceptions import ToolFailed

    async def refuse(ctx, *, call, tool_def, args):
        raise ToolFailed("refused by the probe")

    agent = pai.scripted_agent([("run_sql", {"sql": SELECT}, "c1")], capabilities=[_hooks(refuse)])
    responses, executed = _run(pai, agent)
    assert executed == []
    assert responses["c1"] == ("failed", "refused by the probe")


def test_tool_failed_spends_no_retry_budget(pai):
    """With `retries=0`, one `ModelRetry` would end the run; a `ToolFailed` does not."""
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import ModelRetry, ToolFailed, UnexpectedModelBehavior
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    def model_insisting_twice():
        turns = {"n": 0}

        def model(messages, info):
            turns["n"] += 1
            if turns["n"] <= 2:
                call = ToolCallPart(
                    tool_name="run_sql", args={"sql": SELECT}, tool_call_id=f"c{turns['n']}"
                )
                return ModelResponse(parts=[call])
            return ModelResponse(parts=[TextPart(content="done")])

        return FunctionModel(model)

    async def failed(ctx, *, call, tool_def, args):
        raise ToolFailed("no")

    async def retry(ctx, *, call, tool_def, args):
        raise ModelRetry("again")

    pai.EXECUTED.clear()
    agent = Agent(
        model_insisting_twice(), tools=[pai.run_sql], capabilities=[_hooks(failed)], retries=0
    )
    result = agent.run_sync("go")
    assert result.output == "done" and pai.EXECUTED == []
    assert {k: v[0] for k, v in pai.responses_of(result).items()} == {
        "c1": "failed",
        "c2": "failed",
    }
    agent = Agent(
        model_insisting_twice(), tools=[pai.run_sql], capabilities=[_hooks(retry)], retries=0
    )
    with pytest.raises(UnexpectedModelBehavior):
        agent.run_sync("go")
    assert pai.EXECUTED == []


def test_the_dict_the_hook_returns_is_what_runs(pai):
    async def rewrite(ctx, *, call, tool_def, args):
        return {**args, "text": "rewritten"}

    agent = pai.scripted_agent(
        [("echo", {"text": "original"}, "c1")], capabilities=[_hooks(rewrite)]
    )
    responses, executed = _run(pai, agent)
    assert executed == [("echo", {"text": "rewritten"})]
    assert responses["c1"] == ("success", "rewritten")


def test_hooks_run_in_capability_order_and_hand_their_args_to_the_next(pai):
    order = []

    def tag(name):
        async def hook(ctx, *, call, tool_def, args):
            order.append((name, dict(args)))
            return {**args, "text": args["text"] + name}

        return hook

    agent = pai.scripted_agent(
        [("echo", {"text": "x"}, "c1")],
        capabilities=[_hooks(tag("A")), _hooks(tag("B")), _hooks(tag("C"))],
    )
    _, executed = _run(pai, agent)
    assert order == [("A", {"text": "x"}), ("B", {"text": "xA"}), ("C", {"text": "xAB"})]
    assert executed == [("echo", {"text": "xABC"})]


def test_an_innermost_capability_is_sorted_after_ordinary_ones_wherever_it_is_listed(pai):
    from dataclasses import dataclass

    from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering

    order = []

    @dataclass
    class Inner(AbstractCapability):
        def get_ordering(self):
            return CapabilityOrdering(position="innermost")

        async def before_tool_execute(self, ctx, *, call, tool_def, args):
            order.append("inner")
            return args

    async def ordinary(ctx, *, call, tool_def, args):
        order.append("ordinary")
        return args

    agent = pai.scripted_agent(
        [("echo", {"text": "x"}, "c1")], capabilities=[Inner(), _hooks(ordinary)]
    )
    _run(pai, agent)
    assert order == ["ordinary", "inner"]
    # two innermost capabilities keep the listed order between them
    order.clear()

    @dataclass
    class Inner2(Inner):
        async def before_tool_execute(self, ctx, *, call, tool_def, args):
            order.append("inner2")
            return args

    agent = pai.scripted_agent([("echo", {"text": "x"}, "c1")], capabilities=[Inner(), Inner2()])
    _run(pai, agent)
    assert order == ["inner", "inner2"]


def test_a_hook_that_raises_stops_the_call_and_the_run(pai):
    async def boom(ctx, *, call, tool_def, args):
        raise RuntimeError("hook exploded")

    agent = pai.scripted_agent([("run_sql", {"sql": SELECT}, "c1")], capabilities=[_hooks(boom)])
    # a raising hook surfaces as a failed run, not as a tool result: the tool never ran,
    # and the caller sees the failure instead of a silent pass
    with pytest.raises(RuntimeError, match="hook exploded"):
        _run(pai, agent)
    assert pai.EXECUTED == []


@pytest.mark.parametrize("route", ["tools", "tool_plain", "toolsets", "per_run"])
def test_every_tool_route_the_example_relies_on_passes_through_the_hook(pai, route):
    from pydantic_ai import Agent
    from pydantic_ai.toolsets import FunctionToolset

    seen = []

    async def probe(ctx, *, call, tool_def, args):
        seen.append((call.tool_name, args))
        return args

    calls = [("worker", {"text": "x"}, "c1")]

    def worker(text: str) -> str:
        """Record text."""
        pai.EXECUTED.append(("worker", {"text": text}))
        return text

    run_kwargs = {}
    if route == "tools":
        agent = Agent(pai.scripted_model(calls), tools=[worker], capabilities=[_hooks(probe)])
    elif route == "tool_plain":
        agent = Agent(pai.scripted_model(calls), capabilities=[_hooks(probe)])
        agent.tool_plain(worker)
    elif route == "toolsets":
        agent = Agent(
            pai.scripted_model(calls),
            toolsets=[FunctionToolset([worker])],
            capabilities=[_hooks(probe)],
        )
    else:
        agent = Agent(pai.scripted_model(calls), tools=[worker])
        run_kwargs = {"capabilities": [_hooks(probe)]}
    _, executed = _run(pai, agent, **run_kwargs)
    assert seen == [("worker", {"text": "x"})]
    assert executed == [("worker", {"text": "x"})]


def test_an_argument_validation_failure_never_reaches_the_hook_or_the_tool(pai):
    seen = []

    async def probe(ctx, *, call, tool_def, args):
        seen.append(args)
        return args

    agent = pai.scripted_agent([("run_sql", {"nope": 1}, "c1")], capabilities=[_hooks(probe)])
    responses, executed = _run(pai, agent)
    assert seen == [] and executed == []
    assert responses["c1"][0] == "retried"


def test_the_agent_fields_is_gated_reads_exist_with_these_names(pai):
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import CombinedCapability

    agent = pai.scripted_agent([])
    assert isinstance(agent.root_capability, CombinedCapability)
    assert hasattr(agent.root_capability, "capabilities")
    assert "capabilities" in inspect.signature(Agent.run_sync).parameters
    assert "capabilities" in inspect.signature(Agent.__init__).parameters


def test_negative_control_an_ungated_agent_runs_its_listed_tools(pai):
    agent = pai.scripted_agent([("run_sql", {"sql": DELETE}, "c1")])
    assert not pai.is_gated(agent)
    responses, executed = _run(pai, agent)
    assert executed == [("run_sql", {"sql": DELETE})]
    assert responses["c1"] == ("success", f"recorded SQL: {DELETE}")
