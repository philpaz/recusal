"""What the gate can see in LangChain `create_agent`, pinned so a new release cannot change it quietly.

`tests/test_langchain_agent_gate.py` proves the example's behavior, including why the gate
must be listed last in `middleware=`. This file pins the `create_agent` surface that
behavior depends on: every way an agent can run a tool call passes through the
`wrap_tool_call` middleware, the middleware sees the arguments the tool runs with, and a
middleware that raises stops the call. An agent without the gate is the negative control,
so a probe that stopped observing anything would fail rather than pass.

It runs in the pinned CI jobs (the exact versions the README names) and weekly against the
newest LangChain in `.github/workflows/compat.yml`. A failure there means a LangChain
release changed what the gate can see; STABILITY.md says what happens next.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "langchain_agent_gate.py"

SELECT = "SELECT * FROM t"
DELETE = "DELETE FROM t"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="LangChain requires Python 3.10+"
)


@pytest.fixture(scope="module")
def lc():
    langchain = pytest.importorskip("langchain")
    if int(langchain.__version__.split(".", 1)[0]) < 1:
        pytest.skip("create_agent needs LangChain 1.x")
    spec = importlib.util.spec_from_file_location("recusal_langchain_compat_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _call(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


CALLS = [
    _call("echo", {"text": "hi"}, "1"),
    _call("run_sql", {"sql": DELETE}, "2"),
    _call("run_sql", {"sql": SELECT}, "3"),
]
INPUT = {"messages": [{"role": "user", "content": "run the scripted tools"}]}


def _agent(lc, calls, middleware):
    """`create_agent` on a scripted model that also streams its tool calls.

    The example's model streams only text, so under `stream_mode="messages"` (where the
    agent streams the model) its tool calls would vanish and no route would be exercised.
    This one yields each scripted message as a single chunk carrying the tool calls.
    """
    import json

    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    class StreamingScripted(lc.ScriptedChatModel):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            message = next(self.messages)
            chunks = [
                {
                    "name": c["name"],
                    "args": json.dumps(c["args"]),
                    "id": c["id"],
                    "index": i,
                    "type": "tool_call_chunk",
                }
                for i, c in enumerate(message.tool_calls)
            ]
            chunk = AIMessageChunk(content=message.content, tool_call_chunks=chunks)
            yield ChatGenerationChunk(message=chunk)

    model = StreamingScripted(
        messages=iter([AIMessage(content="", tool_calls=calls), AIMessage(content="done")])
    )
    return create_agent(model=model, tools=[lc.echo, lc.run_sql], middleware=middleware)


class Seen:
    """A policy wrapper recording every call the gate judged."""

    def __init__(self, policy):
        self.policy = policy
        self.calls = []

    def __call__(self, name, args):
        self.calls.append((name, dict(args)))
        return self.policy(name, args)


def _drain(stream):
    for _ in stream:
        pass


async def _adrain(stream):
    async for _ in stream:
        pass


def _routes():
    """Every way to run an agent: (label, uses the async gate, runner)."""
    return [
        ("invoke", False, lambda a: a.invoke(INPUT)),
        *[
            (f"stream {mode}", False, lambda a, m=mode: _drain(a.stream(INPUT, stream_mode=m)))
            for mode in ("values", "updates", "messages", "debug")
        ],
        ("ainvoke, async gate", True, lambda a: asyncio.run(a.ainvoke(INPUT))),
        (
            "astream, async gate",
            True,
            lambda a: asyncio.run(_adrain(a.astream(INPUT, stream_mode="updates"))),
        ),
    ]


def test_create_agent_still_takes_wrap_tool_call_middleware(lc):
    import inspect

    from langchain.agents import create_agent

    assert "middleware" in inspect.signature(create_agent).parameters
    assert callable(lc.wrap_tool_call)


@pytest.mark.parametrize("route", _routes(), ids=lambda r: r[0])
def test_every_route_passes_every_call_through_the_gate(lc, route):
    label, use_async, run = route
    seen = Seen(lc.sql_scope_policy)
    gate = lc.agate_middleware(seen) if use_async else lc.gate_middleware(seen)
    lc.EXECUTED.clear()
    run(_agent(lc, CALLS, [gate]))
    assert sorted(seen.calls, key=str) == sorted(
        [(c["name"], c["args"]) for c in CALLS], key=str
    ), f"{label}: the gate did not judge every proposed call"
    assert ("run_sql", {"sql": DELETE}) not in lc.EXECUTED, f"{label}: a refused call ran"
    assert sorted(lc.EXECUTED, key=str) == sorted(
        [("echo", {"text": "hi"}), ("run_sql", {"sql": SELECT})], key=str
    ), f"{label}: an allowed call did not run"


def test_ainvoke_with_only_the_sync_gate_runs_nothing(lc):
    lc.EXECUTED.clear()
    with pytest.raises(NotImplementedError):
        asyncio.run(_agent(lc, CALLS, [lc.gate_middleware()]).ainvoke(INPUT))
    assert lc.EXECUTED == [], "the sync gate under ainvoke let a tool run"


def test_negative_control_an_ungated_agent_runs_the_refused_call(lc):
    lc.EXECUTED.clear()
    _agent(lc, CALLS, []).invoke(INPUT)
    assert ("run_sql", {"sql": DELETE}) in lc.EXECUTED


def test_the_gate_sees_the_arguments_the_tool_runs_with(lc):
    seen = Seen(lambda name, args: [])
    lc.EXECUTED.clear()
    _agent(lc, CALLS, [lc.gate_middleware(seen)]).invoke(INPUT)
    assert sorted(seen.calls, key=str) == sorted(lc.EXECUTED, key=str)


def test_a_middleware_that_raises_stops_the_call(lc):
    from langchain.agents.middleware import wrap_tool_call

    @wrap_tool_call
    def broken(request, handler):
        raise RuntimeError("gate crashed")

    lc.EXECUTED.clear()
    try:
        _agent(lc, CALLS, [broken]).invoke(INPUT)
    except Exception:
        pass
    assert lc.EXECUTED == [], "a crashing middleware let a tool run"


def test_the_first_middleware_is_still_the_outermost(lc):
    """The gate-last rule rests on this ordering; if LangChain reversed it, the rule would."""
    from langchain.agents.middleware import wrap_tool_call

    order = []

    def recorder(label):
        def record(request, handler):
            order.append(label)
            return handler(request)

        return wrap_tool_call(record, name=f"record_{label}")

    _agent(lc, [CALLS[0]], [recorder("first"), recorder("second")]).invoke(INPUT)
    assert order == ["first", "second"]
