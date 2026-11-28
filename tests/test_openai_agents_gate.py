"""Regression tests for the offline OpenAI Agents SDK + Recusal guardrail example."""

import asyncio
import copy
import re
import subprocess
import sys
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "openai_agents_gate.py"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"

#: How every reason that says what to write ends, and the note on the kernel's own advice
#: in a reason that quotes the kernel's text. The reasons are held word for word here, apart
#: from the adapter's constants, so that a change of one is a change of a test too.
BODY_NOT_READ = (
    "The gate reads a plain function's parameters and never its body, so a plain function "
    "that calls the refused one is gated with the cause still inside it."
)
KERNEL_ADVICE = (
    "The kernel's report below ends with advice of its own, which can be to call the "
    "refused function from a plain function and register that: do not follow that part. "
    + BODY_NOT_READ
)
SINGLEDISPATCH_REASON = (
    "it runs a `functools.singledispatch` function: the implementation that runs is chosen "
    "at call time by the type of an argument, one can be registered at any moment, and "
    "each has its own parameters and defaults, so the gate cannot tie a call to the "
    "function that will run. Register one plain function as the tool, with the code of "
    "each implementation in its own body under a branch on the type of the argument "
    "(`if isinstance(...)`), and no single-dispatch function. " + BODY_NOT_READ
)
NOT_READ_REASON = (
    "the gate could not read a callable the call reaches ({}), so it cannot tell whether "
    "that is one it refuses (a `functools.singledispatch` function, a "
    "`functools.partialmethod`, a weak proxy). Register the function the tool is to run "
    "itself, without the callable that could not be read on the way to it. " + BODY_NOT_READ
)
PARTIALMETHOD_REASON = (
    "it runs a `functools.partialmethod`: the call goes to the callable the partialmethod "
    "holds, and the tool's parameters are read from that callable without the gate "
    "reading what is under it, so a wrapper or a single-dispatch function there could run "
    "with a value no policy judged. Write the method out in place of the partialmethod: "
    "the work of the callable it holds in the method's own body, and the values it binds "
    "as defaults of the method's own parameters (`def export(self, sql, force=False)` in "
    "place of `export = functools.partialmethod(run, force=False)`), with no call to that "
    "callable, and register that. " + BODY_NOT_READ
)
WEAK_PROXY_REASON = (
    "a callable the call reaches is a `weakref.proxy`: it shows the attributes of the "
    "object it stands for and is not that object, so the gate cannot tell which callables "
    "a call to it runs. Put the object the proxy stands for in its place, so that nothing "
    "on the way to the function is a proxy. " + BODY_NOT_READ
)
OPEN_DISPATCH_REASON = (
    "it runs a single-dispatch function (`functools.singledispatch`, or the callable a "
    "`singledispatchmethod` hands out) that has nothing registered but its base. It runs "
    "that base now, but an implementation can be registered on it at any moment, also "
    "after the gate checked a call and before it runs (by another call of the same model "
    "turn, or by a wrapper on its first use), and that implementation would then run for "
    "the judged call, with any parameter of its own at a default no policy judged. A tool "
    "refused for this runs correctly as long as nothing is ever registered on it. Where "
    "nothing is to be registered, register the base function itself; otherwise register "
    "one plain function as the tool, with the code of each implementation in its own body "
    "under a branch on the type of the argument. " + BODY_NOT_READ
)
PROXIED_REASON = (
    "a callable on the way to the function it runs answers for another object (it answers "
    "for `__wrapped__` without holding one, as a proxy or a class-based decorator that "
    "hands every attribute on with `__getattr__` does, or it shows another class than its "
    "own), so the gate cannot tell what a call to it runs. A tool refused for this may run "
    "correctly. A class-based decorator that keeps the function in an attribute and hands "
    "the other attributes on with `__getattr__` is read once it holds `__wrapped__` "
    "itself: call `functools.update_wrapper(self, fn)` in its `__init__`. For any other "
    "such object, register the function it calls itself. "
    + KERNEL_ADVICE
    + " The kernel's walk reports: {}."
)
HIDDEN_REASON = (
    "a decorator on the function it runs holds a second wrapper of it that the gate does "
    "not follow (in its closure, a global it names or an attribute of the decorator "
    "object, and not reached through `__wrapped__`). Remove the cause from the wrapper "
    "the decorator holds: give it no parameter of its own (pass such a value explicitly "
    "in its body), give it the tool's parameters in the tool's order or as keyword-only "
    "ones, and hold no single-dispatch function, weak proxy or cache named again on its "
    "way to the function; or register the function itself, without the decorator. "
    + KERNEL_ADVICE
    + " The kernel's walk reports: {}."
)
UNREAD_CACHE_REASON = (
    "a `functools.lru_cache` wrapper on the way to the function it runs cannot be read: "
    "which function the cache calls could not be told from what the cache refers to, and "
    "that need not be the function it names as `__wrapped__`, so a wrapper with a "
    "parameter of its own could run under the cache with a value no policy judged. "
    "Register the function itself, without the cache. " + BODY_NOT_READ
)
BOUND_REASON = (
    "a wrapper on the way to the function it runs is handed a bound argument (the object "
    "of a method, or a positional argument of a `functools.partial`), and the gate cannot "
    "tell from the wrapper's signature whether it passes that argument on, keeps it or "
    "passes a value of its own in its place, so it cannot tell which parameter of the "
    "function each judged value fills. A tool refused for this may run correctly. "
    "Register the function itself; or write the method without `functools.wraps`, naming "
    "the parameters the tool takes in its own signature, and pass them on to the function "
    "by keyword. " + KERNEL_ADVICE + " The kernel's walk reports: {}."
)
KEPT_RECEIVER_REASON = (
    "a wrapper on the way to the function it runs (it names `__wrapped__`) is handed an "
    "argument bound on the way (the object of a method, or a positional argument of a "
    "`functools.partial`) in the place of `{name}`, a parameter of the function it wraps "
    "that is not named as an object's (`self` or `cls`). Whether the wrapper hands that "
    "argument on, or drops it so that the values it is given fill the function's "
    "parameters from `{name}` on, cannot be read from its signature, so the tool is "
    "refused, whichever it does. Where the SDK reads the tool's parameters through that "
    "method or partial, it leaves `{name}` out of them and passes the judged values on by "
    "position, and a wrapper that drops the bound argument runs every value one parameter "
    "off (the first as `{name}`). Register the function itself, not the method or the "
    "partial that wraps it; or, where `{name}` is the object's place of a method of your "
    "own under another name, name it `self`, and the method is read as any decorated "
    "method is. " + BODY_NOT_READ
)
DECLARED_REASON = (
    "the function it runs, or a callable on the way to it, carries a declared signature (a "
    "`__signature__` attribute, whatever its value, a `__text_signature__`, or a code "
    "object a callable object shows). `inspect.signature` can read such a declaration in "
    "place of the parameters Python binds, and which ones it reads differs between Python "
    "versions, so the kernel's walk does not rely on any: it refuses each one, also where "
    "this Python would not read it and where it matches the real parameters. A tool "
    "refused for this may run correctly. Remove the declaration (delete the "
    "`__signature__` or the `__text_signature__`, or the `__code__` the object shows), so "
    "that the parameters Python binds are what is read. " + BODY_NOT_READ
)
TOO_MANY_REASON = (
    "the function it runs is under more wrappers than the gate reads (a stack of more than "
    "31 wrappers, or more than eight `functools.lru_cache` layers one on another), so the "
    "functions under them cannot be read. Take wrappers off the function until the gate "
    "reads them all, or register the function itself. " + BODY_NOT_READ
)
OPAQUE_REASON = (
    "a callable on the way to the function cannot be inspected: it is an object whose "
    "class defines `__get__`, for which Python shows no signature (a class-based decorator "
    "that is also a descriptor, such as `cachetools.cached`, or a `staticmethod` object "
    "that something other than a class holds), so the functions under it cannot be read. "
    "Register the function itself, without that decorator, or use one the gate reads (a "
    "function decorator written with `functools.wraps`, or a class-based decorator with no "
    "`__get__` that calls `functools.update_wrapper(self, fn)`); hold a function where a "
    "`staticmethod` object stands. " + BODY_NOT_READ
)


def _adapter():
    """The package module the gate's code lives in. Every test that calls this has loaded
    the example first, which puts the in-tree adapter on the path, so this is the code
    under review and not an installed copy."""
    import importlib
    from pathlib import Path

    module = importlib.import_module("recusal_openai_agents._gate")
    assert "adapters" in Path(module.__file__).parts, module.__file__
    return module


def test_python_version_refusal_is_clear_on_every_python():
    code = (
        "import runpy, sys; "
        "sys.version_info = (3, 9, 0, 'final', 0); "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "openai_agents_gate.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_the_sdk_stays_out_of_the_recusal_package_and_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "openai" not in pyproject
    for path in (ROOT / "recusal").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "openai" not in source.lower(), path
        assert "from agents" not in source and "import agents" not in source, path


def test_pinned_versions_are_explicit_in_example_docs_and_ci():
    expected = ("openai-agents==0.24.0", "openai==3.49.0")
    for path in (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        text = path.read_text(encoding="utf-8")
        for pin in expected:
            assert pin in text, f"{path.name}: {pin}"
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # both are roots of the hash-locked files the pull request lanes install: the two
    # example jobs and the recusal-openai-agents wheel job install one, the
    # recusal-openai-agents-mcp wheel job another, and the recusal-openai wheel job's
    # carries the SDK alone
    for lock, pins in (
        ("adapter-openai-agents", expected),
        ("adapter-openai-agents-mcp", expected),
        ("adapter-openai", ("openai==3.49.0",)),
    ):
        roots = (ROOT / "requirements" / f"{lock}.txt").read_text(encoding="utf-8").splitlines()[0]
        for pin in pins:
            assert pin in roots.split(), (lock, pin)
    assert ci.count("requirements/adapter-openai-agents.txt") == 4
    assert ci.count("requirements/adapter-openai-agents-mcp.txt") == 2
    assert ci.count("requirements/adapter-openai.txt") == 2


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestOpenAIAgentsGate:
    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_openai_agents_gate_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def three_calls():
        return [
            ("echo", {"text": "hello"}, "c1"),
            ("run_sql", {"sql": DELETE}, "c2"),
            ("run_sql", {"sql": SELECT}, "c3"),
        ]

    def test_allowed_call_runs_its_tool(self, example):
        outputs, executed = example.run_calls([("run_sql", {"sql": SELECT}, "c1")])
        assert executed == [("run_sql", {"sql": SELECT})]
        assert outputs == {"c1": f"recorded SQL: {SELECT}"}

    def test_refused_call_does_not_run_and_reason_reaches_the_model(self, example):
        outputs, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")])
        assert executed == []
        assert outputs["c1"].startswith("Recusal refused `run_sql` [FAIL]")
        assert "top-level WHERE" in outputs["c1"]

    @pytest.mark.parametrize("streamed", [False, True], ids=["run", "run_streamed"])
    def test_several_calls_in_one_turn_are_gated_separately(self, example, streamed):
        outputs, executed = example.run_calls(self.three_calls(), streamed=streamed)
        assert outputs["c1"] == "hello"
        assert "Recusal refused" in outputs["c2"]
        assert outputs["c3"] == f"recorded SQL: {SELECT}"
        assert sorted(executed) == [("echo", {"text": "hello"}), ("run_sql", {"sql": SELECT})]

    def test_policy_exception_fails_closed(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        outputs, executed = example.run_calls(
            [("echo", {"text": "unsafe"}, "c1")], policy=broken_policy
        )
        assert executed == []
        assert (
            outputs["c1"]
            == "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        )

    def test_a_crash_inside_decide_also_fails_closed(self, example, monkeypatch):
        # decide() already denies when a policy raises; the guardrail's own except covers
        # a failure anywhere else in the decision path
        def crashing_decide(*args, **kwargs):
            raise RuntimeError("decide crashed")

        monkeypatch.setattr(_adapter(), "decide", crashing_decide)
        outputs, executed = example.run_calls([("echo", {"text": "unsafe"}, "c1")])
        assert executed == []
        assert (
            outputs["c1"]
            == "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        )

    @pytest.mark.parametrize(
        "arguments, reason",
        [
            ('{"sql": ', "arguments are not valid JSON"),
            ('["DELETE FROM audit_log"]', "arguments are not a JSON object"),
            ('"DELETE FROM audit_log"', "arguments are not a JSON object"),
        ],
    )
    def test_malformed_and_non_object_arguments_are_refused(self, example, arguments, reason):
        outputs, executed = example.run_calls([("run_sql", arguments, "c1")])
        assert executed == []
        assert outputs["c1"] == f"Recusal refused `run_sql`: {reason}"

    @pytest.mark.parametrize(
        "arguments, runs",
        [
            ('{"sql": "SELECT 1", "sql": "DELETE FROM t"}', None),
            ('{"sql": "DELETE FROM t", "sql": "SELECT 1"}', "SELECT 1"),
        ],
    )
    def test_duplicate_keys_are_judged_as_the_tool_reads_them(self, example, arguments, runs):
        # the SDK and the guardrail both decode with json.loads, so the last key wins in both
        outputs, executed = example.run_calls([("run_sql", arguments, "c1")])
        assert executed == ([] if runs is None else [("run_sql", {"sql": runs})])

    def test_negative_control_runs_delete_without_the_guardrail(self, example):
        outputs, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")], gated=False)
        assert executed == [("run_sql", {"sql": DELETE})]
        assert outputs["c1"] == f"recorded SQL: {DELETE}"

    def test_gate_tools_refuses_every_non_function_tool_in_the_sdk_union(self, example):
        import agents.tool

        members = [typing.get_origin(m) or m for m in typing.get_args(agents.tool.Tool)]
        assert example.FunctionTool in members
        others = [m for m in members if m is not example.FunctionTool]
        assert len(others) >= 12  # guards the iteration: the 0.24.0 union has 12 others
        for cls in others:
            with pytest.raises(TypeError, match=f"cannot gate {cls.__name__} "):
                example.gate_tools([cls.__new__(cls)], example.sql_scope_policy)

    def test_gate_tools_leaves_the_originals_unchanged_and_keeps_their_guardrails(self, example):
        @_adapter().tool_input_guardrail
        def existing(data):
            return _adapter().ToolGuardrailFunctionOutput.allow()

        tool = copy.copy(example.run_sql)
        tool.tool_input_guardrails = [existing]
        (gated,) = example.gate_tools([tool], example.sql_scope_policy)
        assert tool.tool_input_guardrails == [existing]
        assert gated.tool_input_guardrails[0] is existing
        assert [g.name for g in gated.tool_input_guardrails] == ["existing", "recusal"]
        assert not example.run_sql.tool_input_guardrails

    def test_gate_tools_refuses_the_codex_tool(self, example):
        from agents.extensions.experimental.codex import codex_tool

        codex = codex_tool()
        with pytest.raises(TypeError, match="Codex tool"):
            example.gate_tools([codex], example.sql_scope_policy)

    def test_gated_agent_refuses_mcp_servers_handoffs_and_agent_subclasses(self, example):
        from agents.mcp import MCPServerStdio
        from agents.sandbox import SandboxAgent

        Agent = example.Agent
        server = MCPServerStdio(params={"command": "never-started"})
        cases = [
            (Agent(name="a", mcp_servers=[server]), "mcp_servers"),
            (Agent(name="a", handoffs=[Agent(name="b")]), "handoffs"),
            (SandboxAgent(name="a"), "SandboxAgent is not a plain Agent"),
        ]
        for agent, reason in cases:
            with pytest.raises(ValueError, match=reason):
                example.gated_agent(agent, example.sql_scope_policy)

    def test_gated_agent_returns_a_copy(self, example):
        agent = example.Agent(name="a", tools=[example.echo, example.run_sql])
        gated = example.gated_agent(agent, example.sql_scope_policy)
        assert gated is not agent
        assert agent.tools == [example.echo, example.run_sql]
        assert example.is_gated(gated) and not example.is_gated(agent)

    def _parent_and_sub(self, example, *, gate_sub):
        """A parent whose only tool is a sub-agent that proposes DELETE."""
        sub = example.Agent(
            name="sub",
            model=example.scripted_model([("run_sql", {"sql": DELETE}, "s1")]),
            tools=[example.run_sql],
        )
        if gate_sub:
            sub = example.gated_agent(sub, example.sql_scope_policy)
        parent_model = example.scripted_model([("ask_sub", {"input": "tidy the log"}, "p1")])
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask the sub-agent.")
        return parent_model, example.Agent(name="parent", model=parent_model, tools=[tool])

    def test_an_agent_tool_whose_sub_agent_is_ungated_is_refused(self, example):
        _, parent = self._parent_and_sub(example, gate_sub=False)
        with pytest.raises(TypeError, match="its agent's own tool calls are not gated"):
            example.gated_agent(parent, example.sql_scope_policy)

    def test_a_gated_sub_agent_refuses_its_own_calls(self, example):
        _, parent = self._parent_and_sub(example, gate_sub=True)
        example.EXECUTED.clear()
        asyncio.run(example.run_agent(example.gated_agent(parent, example.sql_scope_policy)))
        assert example.EXECUTED == []

    def test_negative_control_a_gate_on_the_agent_tool_alone_misses_the_sub_agent(self, example):
        # what gate_tools refuses to build: the parent's guardrail (bound by hand to the
        # agent tool) sees only the sub-agent's input, and the ungated sub-agent runs the DELETE
        _, parent = self._parent_and_sub(example, gate_sub=False)
        (tool,) = parent.tools
        holder = [tool]
        tool.tool_input_guardrails = [_adapter()._guardrail_for(example.sql_scope_policy, holder)]
        example.EXECUTED.clear()
        asyncio.run(example.run_agent(parent))
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]

    def test_a_standalone_recusal_guardrail_is_bound_to_nothing_and_refuses(self, example):
        tool = copy.copy(example.run_sql)
        tool.tool_input_guardrails = [example.recusal_guardrail(example.sql_scope_policy)]
        model = example.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
        example.EXECUTED.clear()
        asyncio.run(example.run_agent(example.Agent(name="a", model=model, tools=[tool])))
        assert example.EXECUTED == []
        assert "bound to no tool" in example.tool_outputs(model)["c1"]

    @pytest.mark.parametrize("sql, ran", [(SELECT, True), (DELETE, False)])
    def test_an_approved_call_is_still_judged_when_it_runs(self, example, sql, ran):
        tool = copy.copy(example.run_sql)
        tool.needs_approval = True
        model = example.scripted_model([("run_sql", {"sql": sql}, "c1")])
        agent = example.gated_agent(
            example.Agent(name="a", model=model, tools=[tool]), example.sql_scope_policy
        )
        config = example.RunConfig(tracing_disabled=True)

        async def approve_and_resume():
            result = await example.Runner.run(agent, "go", run_config=config)
            assert len(result.interruptions) == 1
            assert example.EXECUTED == []
            state = result.to_state()
            state.approve(result.interruptions[0])
            await example.Runner.run(agent, state, run_config=config)

        example.EXECUTED.clear()
        asyncio.run(approve_and_resume())
        assert example.EXECUTED == ([("run_sql", {"sql": sql})] if ran else [])

    def test_example_runs_end_to_end(self, example):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "run_sql (c2): Recusal refused `run_sql` [FAIL]" in result.stdout
        assert "tools that ran: 2" in result.stdout


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestEffectiveArguments:
    """Review 5, P1: the guardrail judged the model's raw JSON; the SDK then validated it
    through the tool's parameter model, and a validator, a default or a coercion could make
    the function receive something else. The gate now judges the invocation the SDK prepared
    and pins it, so what was judged is what executes; a schema the SDK cannot prepare (user
    code in the model) is refused."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_openai_effective_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    class Recording:
        def __init__(self, policy):
            self.policy, self.seen = policy, []

        def __call__(self, name, args):
            self.seen.append((name, dict(args)))
            return self.policy(name, args)

    def test_a_schema_that_rewrites_an_allowed_proposal_is_refused_and_nothing_runs(self, example):
        """Refused twice over: when the agent is built, so the developer learns then, and by
        the guardrail's own answer for a call, should the tool reach one another way."""
        import json

        policy = self.Recording(example.sql_scope_policy)
        tool = example.transforming_sql_tool()
        for build in (
            lambda: example.gate_tools([tool], policy),
            lambda: example.run_calls([("run_sql", {"sql": SELECT}, "c1")], tools=[tool]),
        ):
            with pytest.raises(TypeError, match="cannot gate 'run_sql': parameter `sql` has"):
                build()
        example.EXECUTED.clear()
        reason = _adapter()._function_tool_refusal(
            "run_sql", json.dumps({"sql": SELECT}), policy, tool
        )
        assert reason.startswith("Recusal refused `run_sql`: parameter `sql` has a type or a")
        assert "a custom validator" in reason and "no call could be judged" in reason
        assert policy.seen == []  # nothing to judge: the value that would run is unknowable
        assert example.EXECUTED == []

    def test_negative_control_the_same_rewriting_tool_runs_the_rewritten_sql_ungated(self, example):
        outputs, executed = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")],
            gated=False,
            tools=[example.transforming_sql_tool()],
        )
        assert executed == [("run_sql", {"sql": DELETE})]  # the recorder is live; the gate is why

    def test_defaults_and_coercion_are_judged_as_the_tool_receives_them(self, example):
        from recusal import Finding

        ran = []

        @example.function_tool(strict_mode=False)
        def charge(amount: int = 100) -> str:
            """Record a charge; contacts nothing."""
            ran.append((amount, type(amount).__name__))
            return "recorded"

        def budget(name, args):
            amount = args.get("amount", 0)
            if isinstance(amount, int) and amount > 10:
                return [Finding.fail("amount_limit", severity="CRITICAL", message="over 10")]
            return []

        for proposed, expect_ran, expect_seen in [
            ({}, [], {"amount": 100}),  # the default, 100, is what would run: refused
            ({"amount": "50"}, [], {"amount": 50}),  # the string the raw check missed: refused
            ({"amount": "5"}, [(5, "int")], {"amount": 5}),  # coerced, allowed, executed as judged
            ({"amount": 7}, [(7, "int")], {"amount": 7}),
        ]:
            ran.clear()
            policy = self.Recording(budget)
            outputs, _ = example.run_calls(
                [("charge", proposed, "c1")], policy=policy, tools=[charge]
            )
            assert policy.seen == [("charge", expect_seen)], proposed
            assert ran == expect_ran, (proposed, outputs)
            assert (outputs["c1"] == "recorded") is bool(expect_ran), outputs

    def test_a_context_taking_tool_is_judged_by_its_data_parameters(self, example):
        from agents import RunContextWrapper

        ran = []

        @example.function_tool
        def ctx_sql(ctx: RunContextWrapper, sql: str) -> str:
            """Record SQL with a context parameter the SDK fills."""
            ran.append(sql)
            return "recorded"

        policy = self.Recording(example.sql_scope_policy)
        outputs, _ = example.run_calls(
            [("ctx_sql", {"sql": SELECT}, "c1")], policy=policy, tools=[ctx_sql]
        )
        assert policy.seen == [("ctx_sql", {"sql": SELECT})] and ran == [SELECT]

    def test_a_nested_application_model_is_refused_because_the_sdk_will_not_prepare_it(
        self, example
    ):
        """The SDK prepares only data-only parameter models; an application model inside
        the parameters could run code, so nothing is prepared and the gate refuses the tool
        when the agent is built. A dict parameter carries the same data and is judged."""
        from pydantic import BaseModel

        class Query(BaseModel):
            sql: str
            limit: int = 10

        ran = []

        @example.function_tool
        def query(q: Query) -> str:
            """Run a structured query; records only."""
            ran.append(q.model_dump())
            return "recorded"

        @example.function_tool(strict_mode=False)
        def query_dict(q: dict) -> str:
            """The same query as plain data."""
            ran.append(dict(q))
            return "recorded"

        policy = self.Recording(lambda n, a: [])
        with pytest.raises(TypeError, match="cannot gate 'query': parameter `q` has a type"):
            example.run_calls(
                [("query", {"q": {"sql": SELECT}}, "c1")], policy=policy, tools=[query, query_dict]
            )
        assert ran == [] and policy.seen == []
        outputs, _ = example.run_calls(
            [("query_dict", {"q": {"sql": SELECT}}, "c2")], policy=policy, tools=[query_dict]
        )
        assert policy.seen == [("query_dict", {"q": {"sql": SELECT}})]
        assert ran == [{"sql": SELECT}] and outputs["c2"] == "recorded"

    def test_arguments_that_do_not_validate_are_refused_by_the_gate(self, example):
        policy = self.Recording(example.sql_scope_policy)
        outputs, executed = example.run_calls([("run_sql", {"sql": 5}, "c1")], policy=policy)
        assert executed == [] and policy.seen == []
        assert "cannot be judged" in outputs["c1"]

    def test_a_hand_built_function_tool_is_refused_at_setup(self, example):
        async def invoke(ctx, arguments: str) -> str:
            return "ran"

        hand_built = example.FunctionTool(
            name="raw",
            description="a tool built by hand",
            params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
            on_invoke_tool=invoke,
        )
        with pytest.raises(TypeError, match="function_tool"):
            example.gate_tools([hand_built], example.sql_scope_policy)

    def test_the_ordinary_path_is_unchanged_and_still_judges_the_exact_sql(self, example):
        policy = self.Recording(example.sql_scope_policy)
        outputs, executed = example.run_calls([("run_sql", {"sql": SELECT}, "c1")], policy=policy)
        assert policy.seen == [("run_sql", {"sql": SELECT})]
        assert executed == [("run_sql", {"sql": SELECT})]


class TestCapabilityIdentity:
    """Adapters review, finding 1: the guardrail looked its tool up by name (first match)
    while the SDK executes the last tool of that name. With a gated sub-agent exposed as
    `run_sql` listed before a function tool of the same name whose schema rewrites SELECT
    to DELETE, the guardrail judged raw SELECT and the function tool ran DELETE. Each
    guardrail is now bound to its own tool, and shared names are refused at setup."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_openai_identity_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def sub_agent_as_run_sql(example):
        sub = example.Agent(
            name="sub",
            model=example.scripted_model([("echo", {"text": "hi"}, "s1")]),
            tools=[example.echo],
        )
        sub = example.gated_agent(sub, example.sql_scope_policy)
        return sub.as_tool(tool_name="run_sql", tool_description="A sub-agent named like the tool.")

    @pytest.mark.parametrize(
        "agent_tool_first", [True, False], ids=["agent-tool-first", "function-first"]
    )
    @pytest.mark.parametrize("streamed", [False, True], ids=["run", "streamed"])
    def test_the_reviewers_duplicate_configuration_is_refused_at_setup(
        self, example, agent_tool_first, streamed
    ):
        agent_tool = self.sub_agent_as_run_sql(example)
        function = example.transforming_sql_tool()
        tools = [agent_tool, function] if agent_tool_first else [function, agent_tool]
        with pytest.raises(TypeError, match="share a name \\['run_sql'\\]"):
            example.gate_tools(tools, example.sql_scope_policy)
        model = example.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
        agent = example.Agent(name="dup", model=model, tools=tools)
        with pytest.raises(ValueError, match="share a name"):
            example.gated_agent(agent, example.sql_scope_policy)
        example.EXECUTED.clear()
        assert example.EXECUTED == [], "nothing ran: the agent was never built"

    @staticmethod
    def limited_sql_tool(example):
        """A second, data-only tool named ``run_sql``: its prepared invocation carries a
        default the raw arguments do not."""

        @example.function_tool(name_override="run_sql", strict_mode=False)
        def run_sql(sql: str, limit: int = 10) -> str:
            """Record SQL text and a row limit without connecting to a database."""
            example.EXECUTED.append(("run_sql", {"sql": sql, "limit": limit}))
            return f"recorded SQL: {sql}"

        return run_sql

    def test_is_gated_is_false_for_an_agent_whose_gated_tools_share_a_name(self, example):
        (a,) = example.gate_tools([example.run_sql], example.sql_scope_policy)
        (b,) = example.gate_tools([self.limited_sql_tool(example)], example.sql_scope_policy)
        model = example.scripted_model([])
        agent = example.Agent(name="dup", model=model, tools=[a, b])
        assert not example.is_gated(agent)
        assert example.is_gated(example.Agent(name="one", model=model, tools=[a]))

    def test_a_bound_guardrail_judges_its_own_tool_whatever_the_agent_lists(self, example):
        """Kills the lookup mutant on its own: the agent's metadata lists a same-named
        agent tool FIRST, which a name lookup would find and judge as the raw arguments;
        the bound guardrail judges the function tool it was built for, as the SDK prepares
        it, with the default the raw arguments leave out."""
        import json
        from types import SimpleNamespace

        judged = []

        def recording_policy(name, tool_input):
            judged.append(dict(tool_input))
            return example.sql_scope_policy(name, tool_input)

        (gated,) = example.gate_tools([self.limited_sql_tool(example)], recording_policy)
        decoy = self.sub_agent_as_run_sql(example)
        misleading_agent = SimpleNamespace(name="misleading", tools=[decoy, gated])
        guardrail = gated.tool_input_guardrails[-1]
        ctx = SimpleNamespace(tool_name="run_sql", tool_arguments=json.dumps({"sql": SELECT}))
        guardrail.guardrail_function(SimpleNamespace(context=ctx, agent=misleading_agent))
        assert judged == [{"sql": SELECT, "limit": 10}]
        assert ctx._function_tool_arguments.args == [SELECT, 10]

    def test_an_allowed_call_pins_the_invocation_that_was_judged(self, example):
        """The gate judges the invocation the SDK prepared and pins that same object on the
        tool context, so the SDK executes it without preparing a second time. A refused
        call pins nothing."""
        import json
        from types import SimpleNamespace

        from recusal import Finding

        judged = []

        def recording_policy(name, tool_input):
            judged.append(dict(tool_input))
            return example.sql_scope_policy(name, tool_input)

        (gated,) = example.gate_tools([example.charge_tool()], recording_policy)
        guardrail = gated.tool_input_guardrails[-1]
        prepare = gated.on_invoke_tool.prepare_arguments

        ctx = SimpleNamespace(tool_name="charge", tool_arguments=json.dumps({}))
        guardrail.guardrail_function(SimpleNamespace(context=ctx, agent=None))
        assert judged == [{"amount": 100}], "the omitted default is judged at its value"
        pinned = ctx._function_tool_arguments
        assert pinned is not None
        assert pinned.kwargs == prepare(json.dumps({}), "charge").kwargs

        def refuse(name, tool_input):
            return [Finding.fail("no", severity="CRITICAL", message="no")]

        (refusing,) = example.gate_tools([example.charge_tool()], refuse)
        ctx = SimpleNamespace(tool_name="charge", tool_arguments=json.dumps({}))
        refusing.tool_input_guardrails[-1].guardrail_function(
            SimpleNamespace(context=ctx, agent=None)
        )
        assert not getattr(ctx, "_function_tool_arguments", None)

    def test_a_bound_guardrail_refuses_a_call_for_another_name(self, example):
        import json
        from types import SimpleNamespace

        (gated,) = example.gate_tools([example.run_sql], example.sql_scope_policy)
        guardrail = gated.tool_input_guardrails[-1]
        ctx = SimpleNamespace(tool_name="echo", tool_arguments=json.dumps({"text": "x"}))
        output = guardrail.guardrail_function(SimpleNamespace(context=ctx, agent=None))
        assert "bound to 'run_sql'" in str(output)

    def test_unique_names_run_exactly_as_before(self, example):
        outputs, executed = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1"), ("echo", {"text": "hi"}, "c2")]
        )
        assert executed == [("run_sql", {"sql": SELECT}), ("echo", {"text": "hi"})]


#: A callable-object tool whose context parameter is annotated through an alias, in a
#: module with string annotations: neither the text ``RunContextWrapper`` nor the class is
#: in the annotation as written, and only resolving it the SDK's way finds the context.
_ALIASED_CONTEXT_SOURCE = '''
from __future__ import annotations

from typing import Any

from agents import RunContextWrapper as RCW

Ctx = RCW[Any]


class SqlRunner:
    def __init__(self, ran):
        self.ran = ran

    def __call__(self, ctx: Ctx, sql: str) -> str:
        """Record SQL text without connecting to a database."""
        self.ran.append(sql)
        return "recorded"
'''


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestJudgedIsWhatRuns:
    """The 2026-11-21 review. What the policy judged was not always what ran: a context
    parameter the gate detected differently from the SDK shifted every argument one name
    over; a wrapper's own parameter ran at a default no policy saw; the gated copy shared
    its ``handoffs`` list with the original. And a tool the gate could never judge was
    accepted when the agent was built and refused on every call."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_openai_judged_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    Recording = TestEffectiveArguments.Recording

    # --- the context parameter is the SDK's decision, never a second reading ----------------

    @pytest.mark.parametrize(
        "sql, runs", [(DELETE, False), (SELECT, True)], ids=["delete", "select"]
    )
    def test_an_aliased_context_on_a_callable_object_does_not_shift_the_names(
        self, example, sql, runs
    ):
        namespace: dict = {}
        exec(compile(_ALIASED_CONTEXT_SOURCE, "<aliased context>", "exec"), namespace)
        ran: list = []
        tool = example.function_tool(namespace["SqlRunner"](ran), name_override="run_sql")
        policy = self.Recording(example.sql_scope_policy)
        outputs, _ = example.run_calls(
            [("run_sql", {"sql": sql}, "c1")], policy=policy, tools=[tool]
        )
        # judged under the name the function receives it by, not under the context's
        assert policy.seen == [("run_sql", {"sql": sql})]
        assert ran == ([sql] if runs else [])
        assert (outputs["c1"] == "recorded") is runs, outputs

    def test_negative_control_the_aliased_context_tool_runs_delete_ungated(self, example):
        namespace: dict = {}
        exec(compile(_ALIASED_CONTEXT_SOURCE, "<aliased context>", "exec"), namespace)
        ran: list = []
        tool = example.function_tool(namespace["SqlRunner"](ran), name_override="run_sql")
        example.run_calls([("run_sql", {"sql": DELETE}, "c1")], gated=False, tools=[tool])
        assert ran == [DELETE]

    def test_prepared_values_that_are_not_one_per_parameter_are_refused(self, example):
        from types import SimpleNamespace

        @example.function_tool(strict_mode=False)
        def search(sql: str, *, dry: bool = False) -> str:
            """Record a search; contacts nothing."""
            return "recorded"

        gate = _adapter()
        schema = gate._sdk_schema(search)
        named = gate._effective_arguments(
            schema, SimpleNamespace(args=[SELECT], kwargs={"dry": True})
        )
        assert named == {"sql": SELECT, "dry": True}
        for args, kwargs in (
            ([], {"dry": True}),  # a value short: the names would shift
            ([SELECT, "one too many"], {"dry": True}),
            ([SELECT], {}),  # a keyword parameter left to the function's own default
            ([SELECT], {"dry": True, "force": True}),  # a keyword the tool does not declare
        ):
            with pytest.raises(TypeError, match="not one for each of the tool's parameters"):
                gate._effective_arguments(schema, SimpleNamespace(args=args, kwargs=kwargs))

    # --- a wrapper cannot run a value no policy judged ----------------------------------------

    @staticmethod
    def wrapped_tools(example, ran):
        import functools

        def adds_a_parameter(fn):
            @functools.wraps(fn)
            def wrapper(*args, dry_run=False, **kwargs):
                ran.append(("wrapper", dry_run))
                return fn(*args, **kwargs)

            return wrapper

        def takes_the_functions_parameter(fn):
            @functools.wraps(fn)
            def wrapper(sql, *args, **kwargs):
                return fn(*args, **kwargs)  # the function then runs on its own default

            return wrapper

        def reorders(fn):
            @functools.wraps(fn)
            def wrapper(table, sql):
                return fn(sql, table)

            return wrapper

        def forwards(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def run_sql(sql: str = DELETE) -> str:
            """Record SQL text without connecting to a database."""
            ran.append(("run_sql", sql))
            return "recorded"

        def copy_rows(sql: str, table: str) -> str:
            """Record a copy; contacts nothing."""
            ran.append(("copy_rows", sql, table))
            return "recorded"

        def tool(decorator, func):
            return example.function_tool(
                decorator(func), name_override=func.__name__, strict_mode=False
            )

        return {
            "adds": (tool(adds_a_parameter, run_sql), "dry_run"),
            "takes": (tool(takes_the_functions_parameter, run_sql), "sql"),
            "reorders": (tool(reorders, copy_rows), "table"),
            "forwards": (tool(forwards, run_sql), None),
        }

    @pytest.mark.parametrize("case", ["adds", "takes", "reorders"])
    def test_a_wrapper_with_a_parameter_of_its_own_is_refused_when_the_agent_is_built(
        self, example, case
    ):
        import json

        ran: list = []
        tool, parameter = self.wrapped_tools(example, ran)[case]
        expected = f"cannot gate {tool.name!r}: a function the call reaches takes `{parameter}`"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)
        with pytest.raises(TypeError, match=expected):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)
        # and the guardrail's own answer for a call, should the tool reach one another way
        policy = self.Recording(example.sql_scope_policy)
        reason = _adapter()._function_tool_refusal(
            tool.name, json.dumps({"sql": SELECT, "table": "t"}), policy, tool
        )
        assert reason.startswith(
            f"Recusal refused `{tool.name}`: a function the call reaches takes `{parameter}`"
        )
        assert policy.seen == [] and ran == []

    def test_negative_control_the_wrapper_that_takes_the_parameter_runs_delete_ungated(
        self, example
    ):
        ran: list = []
        tool, _ = self.wrapped_tools(example, ran)["takes"]
        example.run_calls([("run_sql", {"sql": SELECT}, "c1")], gated=False, tools=[tool])
        assert ran == [("run_sql", DELETE)]  # proposed SELECT, ran the function's default

    def test_a_wrapper_that_only_forwards_is_gated_and_judged_as_the_function(self, example):
        ran: list = []
        tool, _ = self.wrapped_tools(example, ran)["forwards"]
        policy = self.Recording(example.sql_scope_policy)
        outputs, _ = example.run_calls(
            [("run_sql", {}, "c1"), ("run_sql", {"sql": SELECT}, "c2")],
            policy=policy,
            tools=[tool],
        )
        # the function's default, DELETE, is written out by the SDK, judged and refused
        assert sorted(policy.seen, key=str) == sorted(
            [("run_sql", {"sql": DELETE}), ("run_sql", {"sql": SELECT})], key=str
        )
        assert ran == [("run_sql", SELECT)]
        assert outputs["c1"].startswith("Recusal refused `run_sql` [FAIL]")

    def test_a_function_that_declares_another_signature_is_refused(self, example):
        import inspect

        def run_sql(sql: str, limit: int = 10) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        run_sql.__signature__ = inspect.Signature(
            [inspect.Parameter("sql", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)]
        )
        tool = example.function_tool(run_sql)
        with pytest.raises(
            TypeError, match="carries a declared signature \\(a `__signature__` attribute"
        ):
            example.gate_tools([tool], example.sql_scope_policy)

    # --- the gated copy shares no list with the original --------------------------------------

    def test_the_gated_copy_has_its_own_lists(self, example):
        other = example.Agent(name="billing", tools=[example.run_sql])
        agent = example.Agent(name="triage", tools=[example.echo])
        gated = example.gated_agent(agent, example.sql_scope_policy)
        assert gated.handoffs is not agent.handoffs
        assert gated.mcp_servers is not agent.mcp_servers
        assert gated.tools is not agent.tools
        agent.handoffs.append(other)  # the original changes after gating
        agent.mcp_servers.append(object())
        assert gated.handoffs == [] and gated.mcp_servers == []
        assert example.is_gated(gated)

    def test_a_handoff_added_to_the_original_afterwards_is_not_the_gated_copys(self, example):
        from agents.exceptions import ModelBehaviorError

        billing = example.Agent(
            name="billing",
            model=example.scripted_model([("run_sql", {"sql": DELETE}, "b1")]),
            tools=[example.run_sql],
        )
        model = example.scripted_model([("transfer_to_billing", {}, "c1")])
        agent = example.Agent(name="triage", model=model, tools=[example.echo])
        gated = example.gated_agent(agent, example.sql_scope_policy)
        agent.handoffs.append(billing)
        example.EXECUTED.clear()
        with pytest.raises(ModelBehaviorError, match="transfer_to_billing"):
            asyncio.run(example.run_agent(gated))
        assert example.EXECUTED == []

    def test_an_agent_with_a_prompt_is_refused(self, example):
        prompt = {"id": "pmpt_stored"}
        agent = example.Agent(name="a", tools=[example.echo], prompt=prompt)
        with pytest.raises(ValueError, match="cannot gate agent 'a': it has a prompt"):
            example.gated_agent(agent, example.sql_scope_policy)
        gated = example.gated_agent(
            example.Agent(name="a", tools=[example.echo]), example.sql_scope_policy
        )
        assert example.is_gated(gated)
        gated.prompt = prompt
        assert not example.is_gated(gated)

    # --- what the gate could never judge is refused when the agent is built ------------------

    @staticmethod
    def typed_tool(example, annotation, name="typed"):
        def typed(value) -> str:
            """Record a value; contacts nothing."""
            example.EXECUTED.append((name, {"value": value}))
            return "recorded"

        typed.__annotations__ = {"value": annotation, "return": str}
        return example.function_tool(typed, name_override=name, strict_mode=False)

    @staticmethod
    def refused_annotations():
        import dataclasses
        import datetime
        import enum

        from pydantic import BaseModel, Field

        class Color(enum.Enum):
            RED = "red"

        from typing_extensions import TypedDict  # pydantic wants it before Python 3.12

        class Location(TypedDict):
            lat: float

        class Model(BaseModel):
            x: int

        @dataclasses.dataclass
        class Data:
            x: int

        return {
            "enum": Color,
            "typed-dict": Location,
            "model": Model,
            "dataclass": Data,
            "datetime": datetime.datetime,
            "date": datetime.date,
            "tuple": typing.Tuple[int, int],
            "set": typing.Set[str],
            "bytes": bytes,
            "list-of-typed-dict": typing.List[Location],
            "pattern": typing.Annotated[str, Field(pattern="^a")],
        }

    @pytest.mark.parametrize(
        "case",
        [
            "enum",
            "typed-dict",
            "model",
            "dataclass",
            "datetime",
            "date",
            "tuple",
            "set",
            "bytes",
            "list-of-typed-dict",
            "pattern",
        ],
    )
    def test_a_parameter_type_the_sdk_will_not_prepare_is_refused_at_setup(self, example, case):
        tool = self.typed_tool(example, self.refused_annotations()[case])
        expected = "cannot gate 'typed': parameter `value` has a type or a validation"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)
        agent = example.Agent(name="a", tools=[example.echo, tool])
        with pytest.raises(TypeError, match=expected):
            example.gated_agent(agent, example.sql_scope_policy)

    def test_the_refused_parameter_is_named_among_the_others(self, example):
        import enum

        class Mode(enum.Enum):
            FAST = "fast"

        @example.function_tool
        def search(sql: str, mode: Mode, limit: int) -> str:
            """Record a search; contacts nothing."""
            return "recorded"

        with pytest.raises(TypeError, match="cannot gate 'search': parameter `mode` has a type"):
            example.gate_tools([search], example.sql_scope_policy)

    @pytest.mark.parametrize(
        "annotation, value",
        [
            (str, "a"),
            (int, 1),
            (float, 1.5),
            (bool, True),
            (typing.Optional[int], None),
            (typing.List[str], ["a"]),
            (typing.Dict[str, int], {"a": 1}),
            (typing.Any, {"k": [1]}),
            (typing.Literal["a", "b"], "a"),
            (typing.Union[int, str], "a"),
        ],
        ids=["str", "int", "float", "bool", "optional", "list", "dict", "any", "literal", "union"],
    )
    def test_the_data_only_parameter_types_are_gated_and_judged(self, example, annotation, value):
        tool = self.typed_tool(example, annotation)
        policy = self.Recording(lambda name, args: [])
        outputs, executed = example.run_calls(
            [("typed", {"value": value}, "c1")], policy=policy, tools=[tool]
        )
        assert policy.seen == [("typed", {"value": value})]
        assert executed == [("typed", {"value": value})] and outputs["c1"] == "recorded"

    @pytest.mark.parametrize("stars", ["*", "**"])
    def test_a_variadic_parameter_is_refused_at_setup(self, example, stars):
        def many(sql: str, *parts: str) -> str:
            """Record SQL parts; contacts nothing."""
            return "recorded"

        def keyed(sql: str, **options: int) -> str:
            """Record SQL options; contacts nothing."""
            return "recorded"

        func, name = (many, "parts") if stars == "*" else (keyed, "options")
        tool = example.function_tool(func, strict_mode=False)
        expected = (
            f"cannot gate '{func.__name__}': its parameter `{re.escape(stars)}{name}` is variadic"
        )
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_a_tool_not_built_by_function_tool_has_no_schema_to_read(self, example):
        tool = copy.copy(example.run_sql)
        tool.on_invoke_tool = lambda ctx, arguments: "ran"
        with pytest.raises(TypeError, match="Build the tool with function_tool"):
            _adapter()._sdk_schema(tool)
        assert "function_tool" in _adapter()._tool_refusal(tool)

    # --- names shared across namespaces: refused, for the reason that is true ---------------

    def test_a_name_shared_across_namespaces_is_refused_for_what_the_policy_is_given(self, example):
        from agents import tool_namespace

        crm = tool_namespace(name="crm", description="crm tools", tools=[example.run_sql])
        erp = tool_namespace(name="erp", description="erp tools", tools=[example.run_sql])
        expected = (
            "function tools in different namespaces share a name \\['run_sql'\\]; a policy "
            "and the audit record are given a tool's own name without its namespace"
        )
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([*crm, *erp], example.sql_scope_policy)
        agent = example.Agent(name="a", tools=[*crm, *erp])
        with pytest.raises(ValueError, match=expected):
            example.gated_agent(agent, example.sql_scope_policy)
        # one namespace alone is gated, and the policy is given the tool's own name
        from agents.testing import ScriptedModel, assistant_message, function_call

        call = function_call("run_sql", {"sql": SELECT}, call_id="c1", namespace="crm")
        model = ScriptedModel([[call], [assistant_message("done")]])
        policy = self.Recording(example.sql_scope_policy)
        gated = example.gated_agent(example.Agent(name="a", model=model, tools=crm), policy)
        example.EXECUTED.clear()
        asyncio.run(example.run_agent(gated))
        assert policy.seen == [("run_sql", {"sql": SELECT})]
        assert example.EXECUTED == [("run_sql", {"sql": SELECT})]

    def test_a_name_shared_in_one_namespace_keeps_the_last_wins_reason(self, example):
        from agents import tool_namespace

        crm = tool_namespace(
            name="crm", description="crm tools", tools=[example.run_sql, example.run_sql]
        )
        for tools in (crm, [example.run_sql, example.run_sql]):
            with pytest.raises(TypeError, match="share a name \\['run_sql'\\]; the SDK resolves"):
                example.gate_tools(tools, example.sql_scope_policy)

    # --- a refusal says what happened ---------------------------------------------------------

    def test_a_tool_without_parameters_refuses_arguments_and_says_so(self, example):
        ran: list = []

        @example.function_tool(strict_mode=False)
        def ping() -> str:
            """Record a ping; contacts nothing."""
            ran.append("ping")
            return "pong"

        policy = self.Recording(lambda name, args: [])
        outputs, _ = example.run_calls(
            [("ping", {"extra": 1}, "c1"), ("ping", {}, "c2")], policy=policy, tools=[ping]
        )
        assert outputs["c1"] == (
            "Recusal refused `ping`: the arguments the tool would execute cannot be judged: "
            "the tool takes no arguments and the call supplies some"
        )
        assert outputs["c2"] == "pong"
        assert policy.seen == [("ping", {})] and ran == ["ping"]

    def test_arguments_that_do_not_validate_say_so(self, example):
        outputs, executed = example.run_calls([("run_sql", {"sql": ["MARKER"]}, "c1")])
        assert executed == []
        assert outputs["c1"] == (
            "Recusal refused `run_sql`: the arguments the tool would execute cannot be "
            "judged: they do not validate against the tool's parameters"
        )

    @pytest.mark.parametrize(
        "arguments, detail",
        [
            ('{"amount": NaN}', "arguments hold a non-finite number, which is not JSON data"),
            ('{"amount": -Infinity}', "arguments hold a non-finite number, which is not JSON data"),
            ('{"amount": 1e999}', "arguments hold a non-finite number, which is not JSON data"),
            ('{"amount": [1, NaN]}', "arguments hold a non-finite number, which is not JSON data"),
            (
                '{"amount": "nan"}',
                "the arguments the tool would execute cannot be judged (non-finite numbers "
                "are not JSON data)",
            ),
        ],
        ids=["nan", "minus-infinity", "overflow", "nested", "string-read-as-nan"],
    )
    def test_a_non_finite_number_is_refused_before_the_policy_runs(
        self, example, arguments, detail
    ):
        from recusal import Finding

        ran: list = []

        @example.function_tool(strict_mode=False)
        def pay(amount: float) -> str:
            """Record a payment; contacts nothing."""
            ran.append(amount)
            return "paid"

        def ceiling(name, args):  # NaN > 1000 is False: this policy alone lets NaN through
            if args.get("amount", 0) > 1000:
                return [Finding.fail("limit", severity="CRITICAL", message="over limit")]
            return []

        policy = self.Recording(ceiling)
        outputs, _ = example.run_calls([("pay", arguments, "c1")], policy=policy, tools=[pay])
        assert outputs["c1"] == f"Recusal refused `pay`: {detail}"
        assert policy.seen == [] and ran == []

    @pytest.mark.parametrize("depth", [65, 1_500, 100_000])
    def test_arguments_nested_too_deeply_are_refused_not_raised(self, example, depth):
        """Three depths: one past the gate's own limit, one the JSON parser reads and the
        gate's limit refuses, and one the JSON parser itself gives up on. In a real run the
        SDK raises on the larger two before the guardrail is asked (the README says so);
        ``TestSecondReview`` runs the limit itself through a real run."""
        deep = "[" * depth + "]" * depth
        policy = self.Recording(lambda name, args: [])
        reason = _adapter()._function_tool_refusal(
            "run_sql", '{"sql": ' + deep + "}", policy, example.run_sql
        )
        assert reason == "Recusal refused `run_sql`: arguments are nested too deeply"
        assert policy.seen == []

    # --- the pin, in a real run ----------------------------------------------------------------

    @pytest.mark.parametrize("gated", [True, False], ids=["gated", "ungated-control"])
    def test_a_real_run_executes_the_invocation_the_gate_pinned(self, example, monkeypatch, gated):
        """The SDK prepares an invocation by parsing the arguments once. Gated, the one
        parse is the gate's, the pin is on the tool context when the tool starts, and the
        SDK runs it without parsing again. Ungated, there is no pin and the one parse is
        the SDK's own."""
        import agents.tool as sdk_tool
        from agents import RunHooks

        parses: list = []
        original = sdk_tool._parse_function_tool_json_input

        def counting(**kwargs):
            parses.append(kwargs["tool_name"])
            return original(**kwargs)

        monkeypatch.setattr(sdk_tool, "_parse_function_tool_json_input", counting)
        pins: list = []

        class Hooks(RunHooks):
            async def on_tool_start(self, context, agent, tool):
                pin = getattr(context, "_function_tool_arguments", None)
                pins.append(None if pin is None else list(pin.args))

        policy = self.Recording(lambda name, args: [])
        agent = example.Agent(
            name="a",
            model=example.scripted_model([("charge", {}, "c1")]),
            tools=[example.charge_tool()],
        )
        if gated:
            agent = example.gated_agent(agent, policy)
        example.EXECUTED.clear()
        asyncio.run(
            example.Runner.run(
                agent, "go", run_config=example.RunConfig(tracing_disabled=True), hooks=Hooks()
            )
        )
        assert example.EXECUTED == [("charge", {"amount": 100})]
        assert parses == ["charge"], "prepared once; the SDK did not prepare a second time"
        assert pins == ([[100]] if gated else [None])
        assert policy.seen == ([("charge", {"amount": 100})] if gated else [])


class _DescriptorDecorator:
    """A class-based decorator that is also a descriptor, the shape of ``cachetools.cached``:
    ``inspect`` reads such an object as a built-in and finds no signature for it."""

    def __init__(self, fn):
        import functools

        functools.update_wrapper(self, fn)
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __get__(self, instance, owner=None):
        return self


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestSecondReview:
    """The second review of 2026-11-21. The walk over the callables a call reaches was
    pinned only for a single wrapper; a ``functools.singledispatch`` function ran an
    implementation whose default no policy saw; the plain and the audited gate refused
    deeply nested arguments at different depths and for different stated causes; and some
    tools were accepted when the agent was built and then refused on every call."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording

    # --- the walk goes past the outermost callable --------------------------------------------

    @staticmethod
    def stack(example, ran, order):
        """``run_sql(sql, force=False)`` under a stack of decorators, outermost first:
        ``f`` forwards everything, ``t`` takes the function's ``force`` and drops it."""
        import functools

        def forwards(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def takes_force(fn):
            @functools.wraps(fn)
            def wrapper(*args, force=False, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def run_sql(sql: str, *, force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            ran.append({"sql": sql, "force": force})
            return "recorded"

        func = run_sql
        for letter in reversed(order):
            func = (forwards if letter == "f" else takes_force)(func)
        return example.function_tool(func, name_override="run_sql", strict_mode=False)

    @pytest.mark.parametrize("order", ["ft", "fft", "ffft", "ftf", "fftf"])
    def test_a_wrapper_under_other_wrappers_is_found_at_any_level(self, example, order):
        ran: list = []
        tool = self.stack(example, ran, order)
        expected = "cannot gate 'run_sql': a function the call reaches takes `force`"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_negative_control_ungated_the_buried_wrapper_drops_the_argument(self, example):
        ran: list = []
        tool = self.stack(example, ran, "ffft")
        example.run_calls(
            [("run_sql", {"sql": SELECT, "force": True}, "c1")], gated=False, tools=[tool]
        )
        assert ran == [{"sql": SELECT, "force": False}]  # proposed force=True

    def test_a_stack_of_wrappers_that_only_forward_is_gated_and_judged(self, example):
        ran: list = []
        tool = self.stack(example, ran, "ffff")
        policy = self.Recording(lambda name, args: [])
        example.run_calls(
            [("run_sql", {"sql": SELECT, "force": True}, "c1")], policy=policy, tools=[tool]
        )
        assert policy.seen == [("run_sql", {"sql": SELECT, "force": True})]
        assert ran == [{"sql": SELECT, "force": True}]

    # --- a function that picks its code by the type of an argument ---------------------------

    @staticmethod
    def dispatching(ran):
        import functools

        @functools.singledispatch
        def lookup(key, force: bool = False) -> str:
            """Record a lookup; contacts nothing."""
            ran.append({"key": key, "force": force})
            return "recorded"

        @lookup.register
        def _(key: str, force: bool = False, include_deleted: bool = True) -> str:
            ran.append({"key": key, "force": force, "include_deleted": include_deleted})
            return "recorded"

        return lookup

    @pytest.mark.parametrize(
        "wrapped", ["no", "wrapper", "partial"], ids=["bare", "under-a-wrapper", "under-a-partial"]
    )
    def test_a_singledispatch_function_is_refused_when_the_agent_is_built(self, example, wrapped):
        import functools

        func = self.dispatching([])
        if wrapped != "no":
            inner = func if wrapped == "wrapper" else functools.partial(func, force=True)

            @functools.wraps(inner)
            def func(*args, **kwargs):
                return inner(*args, **kwargs)

        tool = example.function_tool(func, name_override="lookup", strict_mode=False)
        expected = "cannot gate 'lookup': it runs a `functools.singledispatch` function"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)
        with pytest.raises(TypeError, match=expected):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)

    def test_a_singledispatchmethod_is_refused_too(self, example):
        import functools

        class Service:
            @functools.singledispatchmethod
            def lookup(self, key, force: bool = False) -> str:
                """Record a lookup; contacts nothing."""
                return "recorded"

        tool = example.function_tool(Service().lookup, name_override="lookup", strict_mode=False)
        with pytest.raises(TypeError, match="it runs a `functools.singledispatch` function"):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_negative_control_ungated_the_registered_implementation_runs_its_own_default(
        self, example
    ):
        ran: list = []
        tool = example.function_tool(
            self.dispatching(ran), name_override="lookup", strict_mode=False
        )
        assert "include_deleted" not in tool.params_json_schema["properties"]
        example.run_calls([("lookup", {"key": "k"}, "c1")], gated=False, tools=[tool])
        assert ran == [{"key": "k", "force": False, "include_deleted": True}]

    # --- a decorator whose signature cannot be read --------------------------------------------

    @pytest.mark.parametrize("wrapped", [False, True], ids=["outermost", "under-a-wrapper"])
    def test_a_descriptor_class_decorator_is_refused_with_what_to_do(self, example, wrapped):
        import functools

        def run_sql(sql: str) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        func = _DescriptorDecorator(run_sql)
        if wrapped:
            inner = func

            @functools.wraps(inner)
            def func(*args, **kwargs):
                return inner(*args, **kwargs)

        tool = example.function_tool(func, name_override="run_sql")
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == "Recusal cannot gate 'run_sql': " + OPAQUE_REASON

    def test_a_decorator_that_calls_the_decorated_function_from_a_plain_one_is_gated(self, example):
        ran: list = []

        @_DescriptorDecorator
        def cached(sql: str) -> str:
            ran.append(sql)
            return "recorded"

        @example.function_tool
        def run_sql(sql: str) -> str:
            """Record SQL text without connecting to a database."""
            return cached(sql)

        policy = self.Recording(example.sql_scope_policy)
        example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1"), ("run_sql", {"sql": DELETE}, "c2")],
            policy=policy,
            tools=[run_sql],
        )
        assert ran == [SELECT]

    @staticmethod
    def forwarding_stack(depth, decorate=None):
        """``run_sql(sql)`` under ``depth`` wrappers that only forward, or under ``depth``
        layers of ``decorate``."""
        import functools

        def run_sql(sql: str) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        def forwards(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        func = run_sql
        for _ in range(depth):
            func = (decorate or forwards)(func)
        return func

    def test_too_many_wrappers_are_refused_for_that_reason(self, example):
        """Too many wrappers is a ``ValueError`` of the walk's own, with no decorator object
        on the chain: neither the reason about a class-based decorator nor "cannot be
        inspected (ValueError)" would say what is wrong with it."""
        import functools

        expected = "Recusal cannot gate 'run_sql': " + TOO_MANY_REASON
        cache = functools.lru_cache(maxsize=None)
        refused = [
            self.forwarding_stack(32),
            cache(self.forwarding_stack(40)),
            self.forwarding_stack(9, cache),
        ]
        for candidate in refused:
            tool = example.function_tool(candidate, name_override="run_sql")
            with pytest.raises(TypeError) as refusal:
                example.gate_tools([tool], example.sql_scope_policy)
            assert str(refusal.value) == expected
        for candidate in (self.forwarding_stack(31), self.forwarding_stack(8, cache)):
            tool = example.function_tool(candidate, name_override="run_sql")
            assert example.gate_tools([tool], example.sql_scope_policy)

    def test_another_unreadable_callable_keeps_the_plain_reason(self, example, monkeypatch):
        """A ``ValueError`` that is not the walk's own limit is still named for what it is."""

        def unreadable(func):
            raise ValueError("no signature found")

        monkeypatch.setattr(_adapter(), "_reached_parameters", unreadable)
        with pytest.raises(TypeError, match="cannot be inspected \\(ValueError\\)"):
            example.gate_tools([example.run_sql], example.sql_scope_policy)

    # --- a mapping keyed by anything but str ----------------------------------------------------

    @pytest.mark.parametrize(
        "annotation",
        [
            typing.Dict[int, str],
            typing.Dict[bool, str],
            typing.Optional[typing.Dict[int, str]],
            typing.List[typing.Dict[int, str]],
            typing.Dict[str, typing.Dict[float, str]],
            typing.Union[str, typing.Dict[int, str]],
        ],
        ids=["int", "bool", "optional", "in-a-list", "nested", "in-a-union"],
    )
    def test_a_mapping_keyed_by_another_type_than_str_is_refused_at_setup(
        self, example, annotation
    ):
        tool = TestJudgedIsWhatRuns.typed_tool(example, annotation)
        expected = "cannot gate 'typed': parameter `value` is a mapping whose key type is not `str`"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)
        with pytest.raises(TypeError, match=expected):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)

    def test_negative_control_ungated_the_sdk_converts_the_keys(self, example):
        tool = TestJudgedIsWhatRuns.typed_tool(example, typing.Dict[int, str])
        _, executed = example.run_calls(
            [("typed", {"value": {"1": "a"}}, "c1")], gated=False, tools=[tool]
        )
        assert executed == [("typed", {"value": {1: "a"}})]

    @pytest.mark.parametrize(
        "annotation",
        [typing.Dict[str, int], dict, typing.Dict[typing.Any, int], typing.Mapping[str, int]],
        ids=["str", "bare", "any", "mapping"],
    )
    def test_a_mapping_keyed_by_str_is_still_gated_and_judged(self, example, annotation):
        tool = TestJudgedIsWhatRuns.typed_tool(example, annotation)
        policy = self.Recording(lambda name, args: [])
        _, executed = example.run_calls(
            [("typed", {"value": {"1": 2}}, "c1")], policy=policy, tools=[tool]
        )
        assert policy.seen == [("typed", {"value": {"1": 2}})]
        assert executed == [("typed", {"value": {"1": 2}})]

    # --- defaults: which are prepared, and the true reason for the ones that are not --------

    def test_a_default_that_is_not_json_data_is_refused_for_that_reason(self, example):
        import math

        def tagged(sql: str, tags: typing.List[str] = ()) -> str:  # type: ignore[assignment]
            """Record SQL text without connecting to a database."""
            return "recorded"

        def timed(sql: str, timeout: float = math.inf) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        for func, parameter in ((tagged, "tags"), (timed, "timeout")):
            tool = example.function_tool(func, strict_mode=False)
            with pytest.raises(TypeError) as refusal:
                example.gate_tools([tool], example.sql_scope_policy)
            assert str(refusal.value).startswith(
                f"Recusal cannot gate {func.__name__!r}: parameter `{parameter}` has a default "
                "that is not plain JSON data"
            )
            assert "has a type or a validation" not in str(refusal.value)

    def test_a_default_factory_of_the_tools_own_is_refused_for_that_reason(self, example):
        from pydantic import Field

        def tagged(
            sql: str,
            tags: typing.Annotated[typing.List[str], Field(default_factory=lambda: ["all"])],
        ) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        tool = example.function_tool(tagged, strict_mode=False)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value).startswith(
            "Recusal cannot gate 'tagged': parameter `tags` has a default factory other than "
            "`list` or `dict`"
        )

    @pytest.mark.parametrize("factory, empty", [(list, []), (dict, {})], ids=["list", "dict"])
    def test_a_list_or_dict_default_factory_is_gated_and_judged(self, example, factory, empty):
        from pydantic import Field

        ran: list = []
        annotation = typing.List[str] if factory is list else typing.Dict[str, str]

        def tagged(sql, tags) -> str:
            """Record SQL text without connecting to a database."""
            ran.append({"sql": sql, "tags": tags})
            return "recorded"

        tagged.__annotations__ = {
            "sql": str,
            "tags": typing.Annotated[annotation, Field(default_factory=factory)],
            "return": str,
        }
        tool = example.function_tool(tagged, strict_mode=False)
        policy = self.Recording(lambda name, args: [])
        example.run_calls([("tagged", {"sql": SELECT}, "c1")], policy=policy, tools=[tool])
        assert policy.seen == [("tagged", {"sql": SELECT, "tags": empty})]
        assert ran == [{"sql": SELECT, "tags": empty}]

    def test_a_refused_type_beside_a_refused_default_is_named_for_the_type(self, example):
        import datetime

        def both(when: datetime.date, tags: typing.List[str] = ()) -> str:  # type: ignore[assignment]
            """Record a search; contacts nothing."""
            return "recorded"

        tool = example.function_tool(both, strict_mode=False)
        with pytest.raises(TypeError, match="parameter `when` has a type or a validation"):
            example.gate_tools([tool], example.sql_scope_policy)

    # --- the parent's policy judges a call to an agent tool ----------------------------------

    def test_the_parent_policy_can_refuse_a_call_to_an_agent_tool(self, example):
        from recusal import Finding

        def no_deletes_to_the_sub_agent(name, args):
            if name == "ask_sub" and "DELETE" in str(args.get("input", "")).upper():
                return [Finding.fail("no_deletes", severity="CRITICAL", message="not to the sub")]
            return []

        def run(text):
            sub_model = example.scripted_model([("run_sql", {"sql": SELECT}, "s1")])
            sub = example.gated_agent(
                example.Agent(name="sub", model=sub_model, tools=[example.run_sql]),
                lambda name, args: [],
            )
            parent_model = example.scripted_model([("ask_sub", {"input": text}, "p1")])
            tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask the sub-agent.")
            parent = example.gated_agent(
                example.Agent(name="parent", model=parent_model, tools=[tool]),
                no_deletes_to_the_sub_agent,
            )
            example.EXECUTED.clear()
            asyncio.run(example.run_agent(parent))
            return example.tool_outputs(parent_model)["p1"], list(example.EXECUTED)

        output, executed = run("please DELETE everything")
        assert output == "Recusal refused `ask_sub` [FAIL]: not to the sub"
        assert executed == [], "the sub-agent was never started"
        output, executed = run("tidy the log")
        assert executed == [("run_sql", {"sql": SELECT})], "an allowed input reaches the sub-agent"

    # --- one fixed depth limit, in a real run --------------------------------------------------

    @staticmethod
    def nested(depth, shape):
        """JSON text whose deepest value sits ``depth`` levels below the value it is."""
        if shape == "list":
            return "[" * depth + "]" * depth
        return '{"a":' * (depth - 1) + "1" + "}" * (depth - 1)

    @pytest.mark.parametrize("shape", ["list", "dict"])
    def test_arguments_at_the_depth_limit_run_and_one_level_deeper_are_refused(
        self, example, shape
    ):
        limit = _adapter()._MAX_DEPTH
        assert limit == 64
        tool = TestJudgedIsWhatRuns.typed_tool(example, typing.Any)
        policy = self.Recording(lambda name, args: [])
        calls = [
            ("typed", '{"value": ' + self.nested(limit, shape) + "}", "c1"),
            ("typed", '{"value": ' + self.nested(limit + 1, shape) + "}", "c2"),
        ]
        outputs, executed = example.run_calls(calls, policy=policy, tools=[tool])
        assert outputs["c1"] == "recorded"
        assert outputs["c2"] == "Recusal refused `typed`: arguments are nested too deeply"
        assert len(executed) == 1 and len(policy.seen) == 1

    def test_the_depth_limit_does_not_move_with_the_recursion_limit(self, example):
        tool = TestJudgedIsWhatRuns.typed_tool(example, typing.Any)
        policy = self.Recording(lambda name, args: [])
        previous = sys.getrecursionlimit()
        sys.setrecursionlimit(previous * 4)
        try:
            reason = _adapter()._function_tool_refusal(
                "typed", '{"value": ' + self.nested(65, "list") + "}", policy, tool
            )
        finally:
            sys.setrecursionlimit(previous)
        assert reason == "Recusal refused `typed`: arguments are nested too deeply"
        assert policy.seen == []

    # --- model settings that put tools into the request ---------------------------------------

    @pytest.mark.parametrize("field", ["extra_body", "extra_args"])
    @pytest.mark.parametrize(
        "key", ["tools", "prompt", "plugins", "web_search_options", "mcp_servers"]
    )
    def test_model_settings_that_carry_tools_are_refused(self, example, field, key):
        from agents import ModelSettings

        settings = ModelSettings(**{field: {key: [{"type": "web_search"}]}})
        agent = example.Agent(name="a", tools=[example.run_sql], model_settings=settings)
        expected = f"cannot gate agent 'a': its model_settings.{field} sets `{key}`"
        with pytest.raises(ValueError, match=expected):
            example.gated_agent(agent, example.sql_scope_policy)
        gated = example.gated_agent(
            example.Agent(name="a", tools=[example.run_sql]), example.sql_scope_policy
        )
        assert example.is_gated(gated)
        gated.model_settings = settings
        assert not example.is_gated(gated)

    def test_model_settings_that_carry_no_tools_are_gated(self, example):
        from agents import ModelSettings

        settings = ModelSettings(
            temperature=0.1, extra_body={"service_tier": "flex"}, extra_args={"user": "u1"}
        )
        agent = example.Agent(name="a", tools=[example.run_sql], model_settings=settings)
        assert example.is_gated(example.gated_agent(agent, example.sql_scope_policy))


class Router:
    """A callable service object that happens to have a ``dispatch`` method and a
    ``registry`` attribute, as a ``functools.singledispatch`` function does."""

    def __init__(self, ran):
        self.ran = ran
        self.registry = {"default": "db1"}

    def dispatch(self, event):
        return None

    def __call__(self, sql: str, limit: int = 10) -> str:
        """Record SQL text without connecting to a database."""
        self.ran.append({"sql": sql, "limit": limit})
        return "recorded"


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestThirdReview:
    """The third review of 2026-11-21. With ``agent.as_tool(parameters=Model)`` the parent's
    policy judged the model's raw JSON while the sub-agent was handed the validated,
    converted and defaulted input; three checks refused what they had no reason to (a
    service object taken for a ``singledispatch`` function, a mapping keyed by a
    ``Literal`` of strings, a stack of wrappers named as "cannot be inspected"); and three
    mutations of the gate passed the suite."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording

    # --- an agent tool with a parameters model ------------------------------------------------

    @staticmethod
    def ask_model():
        from pydantic import BaseModel

        class Ask(BaseModel):
            task: str
            confirm: bool = False
            rows: int = 1

        return Ask

    @staticmethod
    def no_confirmed_or_large(name, args):
        """Refuses a confirmed request and one for more than five rows."""
        from recusal import Finding

        if args.get("confirm") is True:
            return [Finding.fail("no_confirm", severity="CRITICAL", message="not confirmed")]
        if type(args.get("rows", 1)) is int and args.get("rows", 1) > 5:
            return [Finding.fail("too_many", severity="CRITICAL", message="at most five rows")]
        return []

    def ask(self, example, arguments, *, parameters, policy=None, gated=True):
        """A parent that calls ``ask_sub`` with ``arguments``: ``(what the parent's policy
        was handed, the call's output, what the sub-agent was given or None)``."""
        from agents.testing import ScriptedModel, assistant_message

        sub_model = ScriptedModel([[assistant_message("sub done")]])
        sub = example.gated_agent(
            example.Agent(name="sub", model=sub_model, tools=[example.run_sql]),
            lambda name, args: [],
        )
        keywords = {} if parameters is None else {"parameters": parameters}
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask the sub-agent.", **keywords)
        parent_model = example.scripted_model([("ask_sub", arguments, "p1")])
        parent = example.Agent(name="parent", model=parent_model, tools=[tool])
        recording = self.Recording(policy or self.no_confirmed_or_large)
        if gated:
            parent = example.gated_agent(parent, recording)
        asyncio.run(example.run_agent(parent))
        given = None
        if sub_model.first_call is not None:
            given = [item["content"] for item in sub_model.first_call.input]
        return recording.seen, example.tool_outputs(parent_model)["p1"], given

    @pytest.mark.parametrize(
        "arguments, judged",
        [
            (
                {"task": "list", "confirm": False, "rows": 2},
                {"task": "list", "confirm": False, "rows": 2},
            ),
            (
                {"task": "list", "confirm": "false", "rows": "2"},
                {"task": "list", "confirm": False, "rows": 2},
            ),
            ({"task": "list"}, {"task": "list", "confirm": False, "rows": 1}),
            (
                {"task": "list", "rows": 3, "force": True},
                {"task": "list", "confirm": False, "rows": 3},
            ),
        ],
        ids=["typed", "strings-the-sdk-converts", "defaults-left-out", "a-key-the-model-drops"],
    )
    def test_the_parent_policy_is_handed_what_the_sub_agent_is_given(
        self, example, arguments, judged
    ):
        import json

        seen, output, given = self.ask(example, arguments, parameters=self.ask_model())
        assert seen == [("ask_sub", judged)]
        assert output == "sub done"
        assert [json.loads(text) for text in given] == [judged]

    @pytest.mark.parametrize(
        "arguments",
        [{"task": "list", "confirm": "true"}, {"task": "list", "rows": "9"}],
        ids=["confirm-as-a-string", "rows-as-a-string"],
    )
    def test_the_parent_policy_refuses_the_converted_value_it_now_sees(self, example, arguments):
        seen, output, given = self.ask(example, arguments, parameters=self.ask_model())
        assert output.startswith("Recusal refused `ask_sub` [FAIL]: ")
        assert given is None, "the sub-agent was never started"

    def test_negative_control_ungated_the_sub_agent_is_given_the_converted_values(self, example):
        import json

        arguments = {"task": "list", "confirm": "true", "rows": "9", "force": True}
        _, _, given = self.ask(example, arguments, parameters=self.ask_model(), gated=False)
        assert [json.loads(text) for text in given] == [
            {"task": "list", "confirm": True, "rows": 9}
        ]

    @pytest.mark.parametrize(
        "arguments",
        [{"task": "list", "rows": "many"}, {"confirm": True}, {}],
        ids=["a-value-of-another-type", "a-required-field-left-out", "nothing"],
    )
    def test_arguments_the_parameters_model_does_not_accept_are_refused_before_the_policy(
        self, example, arguments
    ):
        seen, output, given = self.ask(example, arguments, parameters=self.ask_model())
        assert output == (
            "Recusal refused `ask_sub`: the input the sub-agent would be given cannot be "
            "judged (the arguments do not validate against its parameters)"
        )
        assert seen == [] and given is None

    def test_the_default_as_tool_is_judged_by_the_arguments_as_proposed(self, example):
        arguments = {"input": "tidy the log", "note": "kept"}
        seen, output, given = self.ask(example, arguments, parameters=None)
        assert seen == [("ask_sub", arguments)]
        assert given == ["tidy the log"]

    @staticmethod
    def unreadable_parameters():
        """``parameters`` types whose validated input the gate cannot work out without
        running code of the type's own, by the name of what makes them so."""
        import dataclasses
        import datetime

        from pydantic import BaseModel, Field, computed_field, field_serializer, field_validator

        class Validated(BaseModel):
            task: str

            @field_validator("task")
            @classmethod
            def upper(cls, value):
                return value.upper()

        class Serialized(BaseModel):
            task: str

            @field_serializer("task")
            def upper(self, value):
                return value.upper()

        class Computed(BaseModel):
            task: str

            @computed_field  # type: ignore[prop-decorator]
            @property
            def shout(self) -> str:
                return self.task.upper()

        class Inner(BaseModel):
            task: str

        class Nested(BaseModel):
            inner: Inner

        class Dated(BaseModel):
            when: datetime.date

        class Built(BaseModel):
            tags: typing.List[str] = Field(default_factory=lambda: ["all"])

        @dataclasses.dataclass
        class Plain:
            task: str

        return {
            "a-validator": Validated,
            "a-serializer": Serialized,
            "a-computed-field": Computed,
            "a-nested-model": Nested,
            "a-date": Dated,
            "a-default-factory": Built,
            "a-dataclass": Plain,
        }

    @pytest.mark.parametrize(
        "case, named",
        [
            (
                "a-validator",
                "model has a field outside the data-only set: `task` (a validator, pydantic "
                "core type `function-after`). The gate works out",
            ),
            ("a-serializer", "model writes `task` out through code of its own"),
            ("a-computed-field", "model writes `shout` out through code of its own"),
            (
                "a-nested-model",
                "model has a field outside the data-only set: `inner` (pydantic core type "
                "`model`). ",
            ),
            (
                "a-date",
                "model has a field outside the data-only set: `when` (pydantic core type `date`). ",
            ),
            (
                "a-default-factory",
                "model has a field outside the data-only set: `tags` (a default that is not "
                "JSON data, or a default factory other than `list` or `dict`). ",
            ),
            (
                "a-dataclass",
                "type is not a Pydantic model the gate reads: its pydantic core type is "
                "`dataclass` ",
            ),
        ],
    )
    def test_a_parameters_type_that_runs_its_own_code_is_refused_at_setup(
        self, example, case, named
    ):
        parameters = self.unreadable_parameters()[case]
        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=parameters)
        expected = f"Recusal cannot gate agent tool 'ask_sub': its `parameters` {named}"
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value).startswith(expected)
        with pytest.raises(TypeError) as refusal:
            example.gated_agent(example.Agent(name="p", tools=[tool]), example.sql_scope_policy)
        assert str(refusal.value).startswith(expected)

    def test_a_refused_parameters_type_is_refused_on_a_call_too(self, example):
        """Should the tool reach a call another way than through ``gate_tools``."""
        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(
            tool_name="ask_sub",
            tool_description="Ask.",
            parameters=self.unreadable_parameters()["a-validator"],
        )
        policy = self.Recording(lambda name, args: [])
        reason = _adapter()._function_tool_refusal("ask_sub", '{"task": "x"}', policy, tool)
        assert reason.startswith(
            "Recusal refused `ask_sub`: its `parameters` model has a field outside the "
            "data-only set: `task` (a validator, pydantic core type `function-after`)"
        )
        assert policy.seen == []

    def test_negative_control_a_validator_gives_the_sub_agent_what_no_policy_saw(self, example):
        import json

        parameters = self.unreadable_parameters()["a-validator"]
        _, _, given = self.ask(example, {"task": "list"}, parameters=parameters, gated=False)
        assert [json.loads(text) for text in given] == [{"task": "LIST"}]

    def test_an_agent_tool_without_the_sdks_adapter_is_refused(self, example):
        """An agent tool the gate cannot read the SDK's adapter from is refused, never
        judged by its raw arguments."""
        import dataclasses

        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.")

        async def invoke(ctx, arguments):
            return "ran"

        rebuilt = dataclasses.replace(tool, on_invoke_tool=invoke)
        rebuilt._is_agent_tool = True
        rebuilt._agent_instance = sub
        with pytest.raises(TypeError, match="does not expose the adapter the SDK validates"):
            example.gate_tools([rebuilt], example.sql_scope_policy)

    def test_a_crash_inside_decide_fails_closed_for_an_agent_tool(self, example, monkeypatch):
        def crashing_decide(*args, **kwargs):
            raise RuntimeError("decide crashed")

        monkeypatch.setattr(_adapter(), "decide", crashing_decide)
        for parameters in (None, self.ask_model()):
            arguments = {"input": "go"} if parameters is None else {"task": "list"}
            _, output, given = self.ask(example, arguments, parameters=parameters)
            assert (
                output
                == "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
            )
            assert given is None, "the sub-agent was never started"

    # --- a sub-agent is gated only by Recusal's own guardrail ---------------------------------

    def test_a_sub_agent_whose_tool_carries_only_another_guardrail_is_not_gated(self, example):
        import dataclasses

        @_adapter().tool_input_guardrail
        def logging_only(data):
            return _adapter().ToolGuardrailFunctionOutput.allow()

        foreign = dataclasses.replace(example.run_sql, tool_input_guardrails=[logging_only])
        sub_model = example.scripted_model([("run_sql", {"sql": DELETE}, "s1")])
        sub = example.Agent(name="sub", model=sub_model, tools=[foreign])
        assert not example.is_gated(sub)
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask the sub-agent.")
        with pytest.raises(TypeError, match="its agent's own tool calls are not gated"):
            example.gated_agent(example.Agent(name="p", tools=[tool]), example.sql_scope_policy)
        (gated,) = example.gate_tools([foreign], example.sql_scope_policy)
        assert example.is_gated(example.Agent(name="sub", tools=[gated]))

    # --- empty argument text -------------------------------------------------------------------

    def test_empty_argument_text_is_judged_as_no_arguments(self, example):
        """The SDK reads empty argument text as no arguments, and so does the gate."""
        ran: list = []

        @example.function_tool
        def tidy(limit: int = 10) -> str:
            """Record a tidy; contacts nothing."""
            ran.append(limit)
            return "recorded"

        policy = self.Recording(lambda name, args: [])
        outputs, _ = example.run_calls([("tidy", "", "c1")], policy=policy, tools=[tidy])
        assert outputs == {"c1": "recorded"}
        assert policy.seen == [("tidy", {"limit": 10})]
        assert ran == [10]

    # --- what is and is not a singledispatch function -----------------------------------------

    def test_a_service_object_with_a_dispatch_method_and_a_registry_is_gated(self, example):
        ran: list = []
        tool = example.function_tool(Router(ran), name_override="run_sql")
        policy = self.Recording(example.sql_scope_policy)
        outputs, _ = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1"), ("run_sql", {"sql": DELETE}, "c2")],
            policy=policy,
            tools=[tool],
        )
        assert policy.seen[0] == ("run_sql", {"sql": SELECT, "limit": 10})
        assert ran == [{"sql": SELECT, "limit": 10}]
        assert outputs["c2"].startswith("Recusal refused `run_sql` [FAIL]")

    def test_reading_for_a_singledispatch_function_runs_no_code_of_the_object(self, example):
        """``registry`` and ``dispatch`` are read without calling a property or a
        ``__getattr__`` of the tool's own."""
        import types

        touched: list = []

        class Lazy:
            @property
            def registry(self):
                touched.append("registry")
                return {}

            @property
            def dispatch(self):
                touched.append("dispatch")
                return lambda event: None

            def __call__(self, sql: str) -> str:
                """Record SQL text without connecting to a database."""
                return "recorded"

        tool = example.function_tool(Lazy(), name_override="lazy")
        assert example.gate_tools([tool], example.sql_scope_policy)
        assert touched == []

        class Registered(Lazy):
            # what ``functools.singledispatch`` sets, with a ``dispatch`` that is not one
            registry = types.MappingProxyType({})

        tool = example.function_tool(Registered(), name_override="lazy")
        assert example.gate_tools([tool], example.sql_scope_policy)
        assert touched == []

    def test_a_singledispatch_function_with_nothing_registered_is_refused(self, example):
        """It was gated until the fifth review: an implementation can be registered
        after a call was judged and before it runs (``TestFifthReview``)."""
        import functools

        @functools.singledispatch
        def lookup(key: str, force: bool = False) -> str:
            """Record a lookup; contacts nothing."""
            return "recorded"

        assert len(lookup.registry) == 1
        tool = example.function_tool(lookup, strict_mode=False)
        with pytest.raises(TypeError, match="it runs a `functools.singledispatch` function"):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_a_registry_whose_base_is_another_function_is_refused(self, example):
        """``registry[object]`` that is not the function ``__wrapped__`` names counts as
        an implementation registered."""
        import functools

        @functools.singledispatch
        def lookup(key: str, force: bool = False) -> str:
            """Record a lookup; contacts nothing."""
            return "recorded"

        @lookup.register(object)
        def _(key, force: bool = False, include_deleted: bool = True) -> str:
            return "recorded"

        assert len(lookup.registry) == 1
        tool = example.function_tool(lookup, strict_mode=False)
        with pytest.raises(TypeError, match="it runs a `functools.singledispatch` function"):
            example.gate_tools([tool], example.sql_scope_policy)

    # --- a mapping keyed by a Literal of strings -------------------------------------------------

    def test_a_mapping_keyed_by_a_literal_of_strings_is_gated_and_judged(self, example):
        annotation = typing.Dict[typing.Literal["a", "b"], int]
        tool = TestJudgedIsWhatRuns.typed_tool(example, annotation)
        policy = self.Recording(lambda name, args: [])
        calls = [
            ("typed", {"value": {"a": 1}}, "c1"),
            ("typed", {"value": {"c": 1}}, "c2"),
        ]
        outputs, executed = example.run_calls(calls, policy=policy, tools=[tool])
        assert policy.seen == [("typed", {"value": {"a": 1}})]
        assert executed == [("typed", {"value": {"a": 1}})]
        assert outputs["c2"] == (
            "Recusal refused `typed`: the arguments the tool would execute cannot be judged: "
            "they do not validate against the tool's parameters"
        )

    @pytest.mark.parametrize(
        "annotation",
        [
            typing.Dict[typing.Literal[1, 2], int],
            typing.Dict[typing.Literal["a", 1], int],
            typing.Dict[typing.Literal[True], int],
        ],
        ids=["integers", "a-string-and-an-integer", "a-boolean"],
    )
    def test_a_mapping_keyed_by_any_other_literal_is_still_refused(self, example, annotation):
        tool = TestJudgedIsWhatRuns.typed_tool(example, annotation)
        expected = (
            "cannot gate 'typed': parameter `value` is a mapping whose key type is not `str` "
            "\\(or a `Literal` of strings\\)"
        )
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)

    # --- the refusal names the abstract collections ---------------------------------------------

    @pytest.mark.parametrize(
        "annotation",
        [typing.Sequence[str], typing.Iterable[str]],
        ids=["sequence", "iterable"],
    )
    def test_a_sequence_or_an_iterable_is_refused_and_named(self, example, annotation):
        tool = TestJudgedIsWhatRuns.typed_tool(example, annotation)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert "parameter `value` has a type or a validation" in str(refusal.value)
        assert "a Sequence or an Iterable" in str(refusal.value)

    # --- an error of the interpreter inside the guardrail ---------------------------------------

    def test_a_recursion_error_inside_the_guardrail_stops_the_run_and_nothing_runs(
        self, example, monkeypatch
    ):
        """Under a recursion limit too small for the walk over the arguments the guardrail
        raises. The SDK then stops the run: the tool does not run."""

        def out_of_stack(value, depth=0):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(_adapter(), "_too_deep", out_of_stack)
        policy = self.Recording(lambda name, args: [])
        with pytest.raises(Exception, match="maximum recursion depth exceeded"):
            example.run_calls([("run_sql", {"sql": SELECT}, "c1")], policy=policy)
        assert example.EXECUTED == [] and policy.seen == []


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestFourthReview:
    """The fourth review of 2026-11-22. A ``functools.singledispatch`` function used as a
    method was no longer recognised (the round-three check did not see through a bound
    method), so a registered implementation ran with a default no policy saw; one with
    nothing registered was refused under a forwarding wrapper, which carries a copy of the
    dispatcher's attributes; and the refusal of an agent tool's ``parameters`` type named
    no field and gave a cause that was not true of most of the types it refused."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording

    # --- the dispatcher is found exactly, wherever it stands on the chain ----------------------

    @staticmethod
    def service(ran, register):
        """The reviewer's shape: ``singledispatch`` on a method, an implementation
        registered for the class itself, which adds ``dry_run``."""
        import functools

        class Service:
            @functools.singledispatch
            def handle(self, sql: str = "", force: bool = False) -> str:
                """Record SQL text without connecting to a database."""
                ran.append(("base", sql, force))
                return "recorded"

            on_class = classmethod(
                functools.singledispatch(
                    lambda cls, sql="", force=False: ran.append(("base", sql, force)) or "recorded"
                )
            )

        if register:

            @Service.handle.register(Service)
            def _(self, sql: str = "", force: bool = False, dry_run: bool = False):
                ran.append(("registered", sql, force, dry_run))
                return "recorded"

            @Service.__dict__["on_class"].__func__.register(type)
            def _(cls, sql: str = "", force: bool = False, dry_run: bool = False):
                ran.append(("registered", sql, force, dry_run))
                return "recorded"

        return Service()

    @staticmethod
    def forwarding(inner):
        import functools

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            return inner(*args, **kwargs)

        return wrapper

    @pytest.mark.parametrize(
        "how", ["bound", "classmethod", "wrapped", "wrapped-twice"], ids=lambda how: how
    )
    def test_a_singledispatch_method_with_an_implementation_registered_is_refused(
        self, example, how
    ):
        service = self.service([], register=True)
        func = {
            "bound": lambda: service.handle,
            "classmethod": lambda: service.on_class,
            "wrapped": lambda: self.forwarding(service.handle),
            "wrapped-twice": lambda: self.forwarding(self.forwarding(service.handle)),
        }[how]()
        tool = example.function_tool(func, name_override="handle", strict_mode=False)
        expected = "cannot gate 'handle': it runs a `functools.singledispatch` function"
        with pytest.raises(TypeError, match=expected):
            example.gate_tools([tool], example.sql_scope_policy)
        with pytest.raises(TypeError, match=expected):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)

    def test_negative_control_ungated_the_registered_method_runs_its_own_default(self, example):
        ran: list = []
        tool = example.function_tool(
            self.service(ran, register=True).handle, name_override="handle", strict_mode=False
        )
        assert "dry_run" not in tool.params_json_schema["properties"]
        example.run_calls([("handle", {"sql": "x"}, "c1")], gated=False, tools=[tool])
        assert ran == [("registered", "x", False, False)]

    def test_where_the_dispatchers_code_is_not_known_it_is_still_found(self, example, monkeypatch):
        """With the code object not found, at import or now, the dispatcher is known by
        its closure and by its attributes, and what the third review pinned holds."""
        adapter = _adapter()
        monkeypatch.setattr(adapter, "_SINGLEDISPATCH_WRAPPER", None)
        monkeypatch.setattr(adapter, "_singledispatch_wrapper_code", lambda: None)
        refused = example.function_tool(
            TestSecondReview.dispatching([]), name_override="lookup", strict_mode=False
        )
        with pytest.raises(TypeError, match="it runs a `functools.singledispatch` function"):
            example.gate_tools([refused], example.sql_scope_policy)
        third = TestThirdReview()
        third.test_a_service_object_with_a_dispatch_method_and_a_registry_is_gated(example)
        third.test_reading_for_a_singledispatch_function_runs_no_code_of_the_object(example)
        third.test_a_singledispatch_function_with_nothing_registered_is_refused(example)
        third.test_a_registry_whose_base_is_another_function_is_refused(example)
        # a service object whose ``registry`` is a plain dict is not one
        router = example.function_tool(Router([]), name_override="run_sql")
        assert example.gate_tools([router], example.sql_scope_policy)

    # --- an agent tool's parameters type: the field, its type and the true reason --------------

    @staticmethod
    def refused_fields():
        import datetime
        import decimal
        import pathlib
        import uuid

        from pydantic import AnyUrl

        return {
            "time": (datetime.time, "time"),
            "timedelta": (datetime.timedelta, "timedelta"),
            "decimal": (decimal.Decimal, "decimal"),
            "uuid": (uuid.UUID, "uuid"),
            "path": (pathlib.Path, "lax-or-strict"),
            "url": (AnyUrl, "function-wrap"),  # a validator of pydantic's own
            "frozenset": (typing.FrozenSet[str], "frozenset"),
            "sequence": (typing.Sequence[str], "json-or-python"),
            "tuple": (typing.Tuple[int, int], "tuple"),
            "set": (typing.Set[str], "set"),
            "bytes": (bytes, "bytes"),
        }

    @pytest.mark.parametrize(
        "case",
        [
            "time",
            "timedelta",
            "decimal",
            "uuid",
            "path",
            "url",
            "frozenset",
            "sequence",
            "tuple",
            "set",
            "bytes",
        ],
    )
    def test_a_refused_parameters_field_is_named_with_its_type(self, example, case):
        from pydantic import create_model

        annotation, kind = self.refused_fields()[case]
        parameters = create_model("Ask", task=(str, ...), value=(annotation, ...))
        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=parameters)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        cause = f"pydantic core type `{kind}`"
        if kind.startswith("function-"):
            cause = f"a validator, {cause}"
        assert str(refusal.value) == (
            "Recusal cannot gate agent tool 'ask_sub': its `parameters` model has a field "
            f"outside the data-only set: `value` ({cause}). The gate works "
            "out what the sub-agent is given only for a Pydantic model the SDK's own rule for "
            "data-only validation admits (`can_prepare_without_user_code`) and that is written "
            "out with no code of its own. Declare fields that are str, int, float, bool, None, "
            "Literal and lists of them, with JSON defaults, and build other types inside the "
            "sub-agent's own tools."
        )

    @staticmethod
    def refused_models():
        from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PrivateAttr

        class Private(BaseModel):
            task: str
            _seen: int = PrivateAttr(default=0)

        class Deferred(BaseModel):
            model_config = ConfigDict(defer_build=True)
            task: str

        class Recursive(BaseModel):
            task: str
            next: typing.Optional["Recursive"] = None

        Recursive.model_rebuild(_types_namespace={"Recursive": Recursive})

        class NotJson(BaseModel):
            task: str
            tags: typing.List[str] = ()  # type: ignore[assignment]

        class Excluded(BaseModel):
            task: str
            note: str = Field("", exclude_if=lambda value: value == "")

        class InAUnion(BaseModel):
            task: str
            rows: typing.Union[
                typing.Annotated[int, PlainSerializer(lambda value: value + 100)], str
            ] = 0

        model = "model is not one the SDK's rule for data-only validation admits"
        return {
            "a-private-attribute": (Private, model),
            "defer-build": (Deferred, model),
            "a-model-that-refers-to-itself": (
                Recursive,
                "type is not a Pydantic model the gate reads: its pydantic core type is "
                "`definitions` ",
            ),
            "a-default-that-is-not-json": (
                NotJson,
                "model has a field outside the data-only set: `tags` (a default that is not "
                "JSON data",
            ),
            "exclude-if": (Excluded, "model writes `note` out through code of its own"),
            "a-serializer-in-a-union-member": (
                InAUnion,
                "model writes `rows` out through code of its own",
            ),
        }

    @pytest.mark.parametrize(
        "case",
        [
            "a-private-attribute",
            "defer-build",
            "a-model-that-refers-to-itself",
            "a-default-that-is-not-json",
            "exclude-if",
            "a-serializer-in-a-union-member",
        ],
    )
    def test_a_refused_parameters_model_says_what_puts_it_outside(self, example, case):
        parameters, named = self.refused_models()[case]
        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=parameters)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value).startswith(
            f"Recusal cannot gate agent tool 'ask_sub': its `parameters` {named}"
        )

    def test_negative_control_ungated_exclude_if_runs_the_models_own_code(self, example):
        from agents.testing import ScriptedModel, assistant_message
        from pydantic import BaseModel, Field

        ran: list = []

        class Excluded(BaseModel):
            task: str
            note: str = Field("", exclude_if=lambda value: ran.append(value) or value == "")

        sub_model = ScriptedModel([[assistant_message("sub done")]])
        sub = example.Agent(name="sub", model=sub_model, tools=[example.run_sql])
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=Excluded)
        parent_model = example.scripted_model([("ask_sub", {"task": "list"}, "p1")])
        asyncio.run(example.run_agent(example.Agent(name="p", model=parent_model, tools=[tool])))
        assert ran, "the callable ran when the SDK wrote the input out"

    def test_text_a_float_field_reads_as_a_non_finite_number_is_refused(self, example):
        """``"nan"`` validates as a float and is not JSON data: refused before the policy,
        for that reason."""
        from pydantic import BaseModel

        class Ask(BaseModel):
            task: str
            ratio: float = 0.0

        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        (tool,) = example.gate_tools(
            [sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=Ask)],
            lambda name, args: [],
        )
        policy = self.Recording(lambda name, args: [])
        for text in ("nan", "inf", "-inf", "1e999"):
            arguments = '{"task": "list", "ratio": "%s"}' % text
            reason = _adapter()._function_tool_refusal("ask_sub", arguments, policy, tool)
            assert reason == (
                "Recusal refused `ask_sub`: the input the sub-agent would be given cannot be "
                "judged (the validated arguments are not JSON data: non-finite numbers are "
                "not JSON data)"
            )
        assert policy.seen == []
        assert (
            _adapter()._function_tool_refusal(
                "ask_sub", '{"task": "list", "ratio": "0.5"}', policy, tool
            )
            is None
        )
        assert policy.seen == [("ask_sub", {"task": "list", "ratio": 0.5})]


_REPLACED_AND_RELOADED = '''
import functools
import importlib

pristine = functools.singledispatch


class Traced:
    """A class-based instrumentation wrapper: it has no ``__code__``."""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)


functools.singledispatch = Traced(pristine)


@pristine
def early(value: str, force: bool = False) -> str:
    """Made by the function as it was, before the package is imported."""
    return "base"


import recusal_openai_agents  # noqa: E402
from agents import function_tool  # noqa: E402
from recusal_openai_agents import _gate, gate_tools  # noqa: E402

print("IMPORTED", _gate._SINGLEDISPATCH_WRAPPER)


def attempt(label, func):
    try:
        gate_tools([function_tool(func)], lambda name, args: [])
    except TypeError as exc:
        print(label, "REFUSED" if "it runs a `functools.singledispatch` function" in str(exc) else exc)
    else:
        print(label, "GATED")


attempt("early", early)
importlib.reload(functools)
assert functools.singledispatch is not pristine


@functools.singledispatch
def late(value: str, force: bool = False) -> str:
    """Made after the reload, by a function with another code object."""
    return "base"


attempt("late", late)
late.register(str)(lambda value, force=False, recursive=True: "registered")
attempt("late-registered", late)
'''


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestFifthReview:
    """The fifth review of 2026-11-22, closed on 2026-11-23. A ``functools.singledispatch``
    tool with nothing registered was gated, and the SDK judges every call of a model turn
    before any tool body runs, so an implementation registered during a turn (by another
    call of it, or by the tool's own first use) ran with a parameter no policy saw. A
    dispatcher that is a callable object's ``__call__`` was not found under a forwarding
    wrapper. The dispatcher was known by one code object read at import. And the refusal
    of an agent tool's ``parameters`` type named a type the data-only set admits."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording
    REFUSED = "it runs a `functools.singledispatch` function"

    # --- registered in the same turn: every dispatcher is refused when the agent is built ----

    @staticmethod
    def plugin_tools(example, events):
        """The reviewer's shape: ``export`` has nothing registered, and ``load_plugin``
        registers an implementation with a parameter of its own."""
        import functools

        @functools.singledispatch
        def export(target: str, overwrite: bool = False) -> str:
            """Record an export; contacts nothing."""
            events.append(("base", target, overwrite))
            return "exported"

        def load_plugin(name: str) -> str:
            """Record a plugin load; contacts nothing."""

            @export.register(str)
            def _(target, overwrite=False, recursive=True):
                events.append(("registered", target, overwrite, recursive))
                return "exported by the plugin"

            return "loaded"

        return [
            example.function_tool(export, strict_mode=False),
            example.function_tool(load_plugin),
        ]

    @staticmethod
    def lazy_tool(example, events):
        """One tool whose own first use registers an implementation."""
        import functools

        @functools.singledispatch
        def render(doc: str, strict: bool = True) -> str:
            """Record a rendering; contacts nothing."""
            events.append(("base", doc, strict))
            if len(render.registry) == 1:

                @render.register(str)
                def _(doc, strict=True, unsafe_html=True):
                    events.append(("registered", doc, strict, unsafe_html))
                    return "text"

            return "base"

        return example.function_tool(render, strict_mode=False)

    @pytest.mark.parametrize("order", ["load-then-export", "export-then-load"])
    def test_a_dispatcher_another_call_of_the_turn_would_register_on_is_refused(
        self, example, order
    ):
        events: list = []
        tools = self.plugin_tools(example, events)
        if order == "export-then-load":
            tools.reverse()
        policy = self.Recording(lambda name, args: [])
        with pytest.raises(TypeError, match=f"cannot gate 'export': {self.REFUSED}"):
            example.gated_agent(example.Agent(name="a", tools=tools), policy)
        with pytest.raises(TypeError, match=f"cannot gate 'export': {self.REFUSED}"):
            example.gate_tools(tools, policy)
        assert events == [] and policy.seen == []

    def test_negative_control_ungated_an_implementation_registered_in_the_turn_runs(self, example):
        """Why registered or not makes no difference: the call to ``export`` is checked
        while nothing is registered, and runs the implementation the earlier call of the
        same turn registered, with ``recursive`` at its own default."""
        events: list = []
        tools = self.plugin_tools(example, events)
        assert "recursive" not in tools[0].params_json_schema["properties"]
        outputs, _ = example.run_calls(
            [("load_plugin", {"name": "s3"}, "c1"), ("export", {"target": "bucket"}, "c2")],
            gated=False,
            tools=tools,
        )
        assert outputs == {"c1": "loaded", "c2": "exported by the plugin"}
        assert events == [("registered", "bucket", False, True)]

    def test_a_dispatcher_that_registers_on_its_own_first_use_is_refused(self, example):
        events: list = []
        tool = self.lazy_tool(example, events)
        with pytest.raises(TypeError, match=f"cannot gate 'render': {self.REFUSED}"):
            example.gated_agent(example.Agent(name="a", tools=[tool]), lambda name, args: [])
        assert events == []

    def test_negative_control_ungated_the_second_call_of_the_turn_runs_the_new_one(self, example):
        events: list = []
        tool = self.lazy_tool(example, events)
        example.run_calls(
            [("render", {"doc": "a"}, "c1"), ("render", {"doc": "b"}, "c2")],
            gated=False,
            tools=[tool],
        )
        assert events == [("base", "a", True), ("registered", "b", True, True)]

    @staticmethod
    def shapes(register):
        """A dispatcher in every place a call reaches one, by name."""
        import functools

        forwarding = TestFourthReview.forwarding
        service = TestFourthReview.service([], register=register)

        def plain():
            @functools.singledispatch
            def handle(sql: str = "", force: bool = False) -> str:
                """Record SQL text without connecting to a database."""
                return "recorded"

            if register:
                handle.register(str)(lambda sql="", force=False, dry_run=False: "recorded")
            return handle

        class Callable:
            __call__ = functools.singledispatch(lambda self, sql="", force=False: "recorded")

        class Static:
            handle = staticmethod(plain())

        class Method:
            @functools.singledispatchmethod
            def handle(self, sql: str = "", force: bool = False) -> str:
                """Record SQL text without connecting to a database."""
                return "recorded"

        if register:
            Callable.__call__.register(Callable)(
                lambda self, sql="", force=False, dry_run=False: "recorded"
            )
            Method.handle.register(str)(lambda self, sql="", force=False, dry_run=False: "recorded")
        return {
            "the-tool": plain,
            "wrapped": lambda: forwarding(plain()),
            "wrapped-twice": lambda: forwarding(forwarding(plain())),
            "cached": lambda: functools.lru_cache(maxsize=None)(plain()),
            "wrapped-over-cached": lambda: forwarding(functools.lru_cache(maxsize=None)(plain())),
            "bound": lambda: service.handle,
            "bound-wrapped": lambda: forwarding(service.handle),
            "classmethod": lambda: service.on_class,
            "staticmethod": lambda: Static().handle,
            "the-staticmethod-object": lambda: Static.__dict__["handle"],
            "singledispatchmethod": lambda: Method().handle,
            "singledispatchmethod-wrapped": lambda: forwarding(Method().handle),
            "bound-call": lambda: Callable().__call__,
            "wrapped-over-a-callable-object": lambda: forwarding(Callable()),
            "wrapped-twice-over-a-callable-object": lambda: forwarding(forwarding(Callable())),
        }

    SHAPES = [
        "the-tool",
        "wrapped",
        "wrapped-twice",
        "cached",
        "wrapped-over-cached",
        "bound",
        "bound-wrapped",
        "classmethod",
        "staticmethod",
        "the-staticmethod-object",
        "singledispatchmethod",
        "singledispatchmethod-wrapped",
        "bound-call",
        "wrapped-over-a-callable-object",
        "wrapped-twice-over-a-callable-object",
    ]

    @pytest.mark.parametrize("register", [False, True], ids=["nothing-registered", "registered"])
    @pytest.mark.parametrize("how", SHAPES)
    def test_a_dispatcher_is_refused_wherever_it_stands_registered_or_not(
        self, example, how, register
    ):
        shapes = self.shapes(register)
        assert sorted(shapes) == sorted(self.SHAPES)
        tool = example.function_tool(shapes[how](), name_override="handle", strict_mode=False)
        with pytest.raises(TypeError, match=f"cannot gate 'handle': {self.REFUSED}"):
            example.gate_tools([tool], example.sql_scope_policy)
        with pytest.raises(TypeError, match=f"cannot gate 'handle': {self.REFUSED}"):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)

    def test_negative_control_ungated_a_wrapped_callable_objects_dispatcher_runs_its_own(
        self, example
    ):
        """The reviewer's second shape: the dispatcher is a callable object's ``__call__``
        under a forwarding wrapper, and the registered implementation adds ``dry_run``."""
        import functools

        ran: list = []

        class Runner:
            __call__ = functools.singledispatch(
                lambda self, sql="", force=False: ran.append(("base", sql, force)) or "recorded"
            )

        @Runner.__call__.register(Runner)
        def _(self, sql: str = "", force: bool = False, dry_run: bool = False):
            ran.append(("registered", sql, force, dry_run))
            return "recorded"

        tool = example.function_tool(
            TestFourthReview.forwarding(Runner()), name_override="handle", strict_mode=False
        )
        assert "dry_run" not in tool.params_json_schema["properties"]
        example.run_calls([("handle", {"sql": "x"}, "c1")], gated=False, tools=[tool])
        assert ran == [("registered", "x", False, False)]

    def test_the_refusal_says_the_cause_and_what_to_do(self, example):
        tool = example.function_tool(
            self.shapes(False)["the-tool"](), name_override="handle", strict_mode=False
        )
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == "Recusal cannot gate 'handle': " + SINGLEDISPATCH_REASON

    def test_the_guardrail_asks_again_on_every_call(self, example):
        """A tool that was gated and whose function is then changed so that it reaches a
        dispatcher is refused on its next call, before the policy and the body."""
        import functools

        ran: list = []

        def run_sql(sql: str) -> str:
            """Record SQL text without connecting to a database."""
            ran.append(sql)
            return "recorded"

        policy = self.Recording(lambda name, args: [])
        (gated,) = example.gate_tools([example.function_tool(run_sql)], policy)
        outputs, _ = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c1")], gated=False, tools=[gated]
        )
        assert outputs == {"c1": "recorded"} and ran == [SELECT]
        run_sql.__wrapped__ = functools.singledispatch(lambda sql: "other")  # type: ignore[attr-defined]
        outputs, _ = example.run_calls(
            [("run_sql", {"sql": SELECT}, "c2")], gated=False, tools=[gated]
        )
        assert outputs["c2"].startswith(f"Recusal refused `run_sql`: {self.REFUSED}")
        assert ran == [SELECT] and len(policy.seen) == 1

    # --- every callable the kernel's walk visits is asked ---------------------------------------

    def test_the_gates_walk_takes_every_step_the_kernels_walk_takes(self, example):
        """The dispatcher question is asked of the callables ``_reached_callables`` finds.
        For each shape, whatever the kernel's own step leads to from a callable the gate
        reached is reached by the gate too, as the function under it. Asked of the step of
        the kernel in the tree (``recusal`` 0.16.0 takes a walk as its second argument).
        No shape here is a cache named again for another function, the one step of that
        walk the gate does not take (``_steps``; ``TestKernelRefusals``)."""
        import functools
        import inspect

        from recusal import invocation

        adapter = _adapter()

        def under(link):
            while inspect.ismethod(link) or isinstance(link, functools.partial):
                link = link.__func__ if inspect.ismethod(link) else link.func
            return link

        def plain(sql: str = "", force: bool = False) -> str:
            return "recorded"

        class Callable:
            def __call__(self, sql: str = "", force: bool = False) -> str:
                return "recorded"

        class Decorator:
            def __init__(self, wrapped):
                functools.update_wrapper(self, wrapped)

            def __call__(self, *args, **kwargs):
                return self.__wrapped__(*args, **kwargs)

        forwarding = TestFourthReview.forwarding
        shapes = [
            plain,
            forwarding(forwarding(plain)),
            functools.lru_cache(maxsize=None)(forwarding(plain)),
            functools.partial(forwarding(plain), force=True),
            Callable(),
            Callable().__call__,
            forwarding(Callable()),
            Decorator(plain),
            forwarding(Decorator(forwarding(plain))),
            staticmethod(plain),
            *(make() for make in self.shapes(True).values()),
        ]
        compared = 0
        for func in shapes:
            reached, complete = adapter._reached_callables(func)
            assert complete and reached[0] is func
            assert all(sum(link is other for other in reached) == 1 for link in reached)
            for link in reached:
                try:
                    following = invocation._next_callables(link, invocation._walk(link, None))
                except (TypeError, ValueError):
                    # a ``__call__`` the kernel refuses to read, or a dispatcher with an
                    # implementation registered, which its step refuses itself
                    continue
                for nxt in following:
                    assert any(under(nxt) is other for other in reached), (func, link, nxt)
                    compared += 1
        assert compared > len(shapes)  # the kernel's step was asked, and led somewhere

    def test_a_walk_that_does_not_finish_is_refused(self, example, monkeypatch):
        tool = example.function_tool(TestFourthReview.forwarding(self.plain_function()))
        assert example.gate_tools([tool], example.sql_scope_policy)
        monkeypatch.setattr(_adapter(), "_MAX_REACHED", 1)
        with pytest.raises(TypeError, match="under more wrappers than the gate reads"):
            example.gate_tools([tool], example.sql_scope_policy)

    @staticmethod
    def plain_function():
        def run_sql(sql: str) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        return run_sql

    def test_a_callable_the_question_cannot_be_asked_of_is_refused(self, example, monkeypatch):
        tool = example.function_tool(self.plain_function())

        def broken(link):
            raise RuntimeError("no answer")

        monkeypatch.setattr(_adapter(), "_is_dispatcher", broken)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == "Recusal cannot gate 'run_sql': " + NOT_READ_REASON.format(
            "RuntimeError"
        )

    def test_a_kernel_that_refuses_a_dispatcher_itself_gets_the_same_reason(
        self, example, monkeypatch
    ):
        """The kernel's walk raises ``DispatchedAtCallTime`` for a dispatcher it finds. It is
        the kernel's own class that is raised here (``recusal`` 0.16.0 is the floor)."""
        from recusal.invocation import DispatchedAtCallTime

        adapter = _adapter()
        tool = example.function_tool(self.plain_function())

        def walk(func):
            raise DispatchedAtCallTime("chosen at call time")

        monkeypatch.setattr(adapter, "_reached_parameters", walk)
        with pytest.raises(TypeError, match=f"cannot gate 'run_sql': {self.REFUSED}"):
            example.gate_tools([tool], example.sql_scope_policy)

        def other(func):
            raise ValueError("something else")

        monkeypatch.setattr(adapter, "_reached_parameters", other)
        with pytest.raises(TypeError, match=r"cannot be inspected \(ValueError\)"):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_a_kernel_that_refuses_a_hidden_wrapper_gets_a_reason_that_names_it(
        self, example, monkeypatch
    ):
        """The kernel's walk raises ``HiddenWrapper`` (a ``ValueError``) for a wrapper that
        holds a second wrapper of the function it does not follow. The refusal says so and
        gives the remedy, with the kernel's own text, which names the parameter. It is the
        kernel's own class that is raised here (``recusal`` 0.16.0 is the floor)."""
        from recusal.invocation import HiddenWrapper

        adapter = _adapter()
        tool = example.function_tool(self.plain_function())

        def walk(func):
            raise HiddenWrapper("`force` is a parameter of a wrapper the walk does not follow")

        monkeypatch.setattr(adapter, "_reached_parameters", walk)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        reason = str(refusal.value)
        assert "holds a second wrapper of it that the gate does not follow" in reason
        assert "`force` is a parameter of a wrapper the walk does not follow" in reason
        assert "cannot be inspected" not in reason

    # --- how a dispatcher is known ---------------------------------------------------------------

    @staticmethod
    def dispatcher():
        import functools

        @functools.singledispatch
        def lookup(key: str) -> str:
            return key

        return lookup

    def test_each_of_the_three_ways_knows_a_dispatcher_on_its_own(self, example, monkeypatch):
        import types

        adapter = _adapter()
        lookup = self.dispatcher()
        assert type(lookup) is types.FunctionType
        assert type(adapter._SINGLEDISPATCH_WRAPPER) is types.CodeType
        assert lookup.__code__ is adapter._SINGLEDISPATCH_WRAPPER
        assert lookup.__code__ is adapter._singledispatch_wrapper_code()
        dispatch = adapter._closure_value(lookup, "dispatch")
        assert dispatch is lookup.dispatch
        assert type(adapter._closure_value(dispatch, "registry")) is dict
        assert adapter._closure_value(lookup, "no_such_name") is None
        for plain in (print, len, Router([]), None, self.plain_function(), "text"):
            assert adapter._dispatch_function(plain) is False
            assert adapter._is_dispatcher(plain) is False

        # the attributes alone: no code known, a closure that cannot be read
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_SINGLEDISPATCH_WRAPPER", None)
            patch.setattr(adapter, "_singledispatch_wrapper_code", lambda: None)
            patch.setattr(adapter, "_closure_value", lambda func, name: None)
            assert adapter._dispatch_function(lookup) is True
            # a wrapper that carries a copy of them is taken for one
            assert adapter._dispatch_function(TestFourthReview.forwarding(lookup)) is True

        bare = self.dispatcher()
        for name in ("registry", "dispatch", "register"):
            delattr(bare, name)
        assert adapter._dispatch_function(bare) is True
        # the closure alone
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_SINGLEDISPATCH_WRAPPER", None)
            patch.setattr(adapter, "_singledispatch_wrapper_code", lambda: None)
            assert adapter._dispatch_function(bare) is True
            patch.setattr(adapter, "_closure_value", lambda func, name: None)
            assert adapter._dispatch_function(bare) is False
        # the code as it was at import alone, and as functools has it now alone
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_closure_value", lambda func, name: None)
            assert adapter._dispatch_function(bare) is True
            patch.setattr(adapter, "_singledispatch_wrapper_code", lambda: None)
            assert adapter._dispatch_function(bare) is True
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_closure_value", lambda func, name: None)
            patch.setattr(adapter, "_SINGLEDISPATCH_WRAPPER", None)
            assert adapter._dispatch_function(bare) is True

    @pytest.mark.parametrize("missing", ["registry", "dispatch", "register", "nothing"])
    def test_the_attributes_are_a_mapping_proxy_beside_two_callables(self, example, missing):
        """An object that is no function is known by all three attributes together; a
        ``registry`` of another type, or one of the two callables missing, is not one."""
        import types

        class Service:
            def __call__(self, sql: str) -> str:
                """Record SQL text without connecting to a database."""
                return "recorded"

        attributes = {
            "registry": types.MappingProxyType({object: None}),
            "dispatch": lambda self, cls: None,
            "register": lambda self, cls: None,
        }
        if missing == "registry":
            attributes["registry"] = {object: None}  # a plain dict
        elif missing != "nothing":
            del attributes[missing]
        for name, value in attributes.items():
            setattr(Service, name, value)
        tool = example.function_tool(Service(), name_override="run_sql")
        if missing == "nothing":
            assert _adapter()._dispatch_function(Service()) is True
            with pytest.raises(TypeError, match=self.REFUSED):
                example.gate_tools([tool], example.sql_scope_policy)
        else:
            assert _adapter()._dispatch_function(Service()) is False
            assert example.gate_tools([tool], example.sql_scope_policy)

    def test_asking_runs_no_code_of_the_object(self, example):
        touched: list = []

        class Lazy:
            def __getattr__(self, name):
                touched.append(name)
                raise AttributeError(name)

            @property
            def register(self):
                touched.append("register")
                return lambda cls: None

            @property
            def _unbound(self):
                touched.append("_unbound")
                return None

            @property
            def dispatcher(self):
                touched.append("dispatcher")
                return None

        assert _adapter()._is_dispatcher(Lazy()) is False
        assert touched == []

    def test_a_dispatcher_built_by_another_copy_of_functools_is_refused(self, example):
        """What ``importlib.reload(functools)`` leaves behind: a ``singledispatch`` with
        another code object and a ``singledispatchmethod`` of another class. Built here
        from a second copy of the module, so the interpreter's own is left alone."""
        import functools
        import importlib.util

        adapter = _adapter()
        spec = importlib.util.find_spec("functools")
        assert spec is not None and spec.loader is not None
        other = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(other)
        assert other.singledispatch is not functools.singledispatch
        assert other.singledispatchmethod is not functools.singledispatchmethod

        @other.singledispatch
        def lookup(key: str, force: bool = False) -> str:
            """Record a lookup; contacts nothing."""
            return "recorded"

        assert lookup.__code__ is not adapter._SINGLEDISPATCH_WRAPPER
        assert lookup.__code__ is not adapter._singledispatch_wrapper_code()

        class Service:
            @other.singledispatchmethod
            def lookup(self, key: str, force: bool = False) -> str:
                """Record a lookup; contacts nothing."""
                return "recorded"

        assert not isinstance(Service.__dict__["lookup"], functools.singledispatchmethod)
        for func in (lookup, TestFourthReview.forwarding(lookup), Service().lookup):
            tool = example.function_tool(func, name_override="lookup", strict_mode=False)
            with pytest.raises(TypeError, match=f"cannot gate 'lookup': {self.REFUSED}"):
                example.gate_tools([tool], example.sql_scope_policy)

    def test_the_code_is_read_without_raising_whatever_stands_at_singledispatch(
        self, example, monkeypatch
    ):
        import functools
        import types

        adapter = _adapter()
        pristine = functools.singledispatch
        assert adapter._singledispatch_wrapper_code() is self.dispatcher().__code__

        class NoCode:
            def __call__(self, func):
                return pristine(func)

        class Raises:
            @property
            def __code__(self):
                raise RuntimeError("instrumented")

        class NotCode:
            __code__ = "text"

        def two_of_them(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            first = wrapper

            def wrapper(*args, **kwargs):  # noqa: F811
                return first(*args, **kwargs)

            return wrapper

        def one_of_them(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        def none_of_them(func):
            return func

        class LooksLikeCode:
            # not a code object, though it holds one nested function named ``wrapper``
            __code__ = types.SimpleNamespace(co_consts=one_of_them.__code__.co_consts)

        assert [c.co_name for c in LooksLikeCode.__code__.co_consts if hasattr(c, "co_name")] == [
            "wrapper"
        ]
        replacements = (
            NoCode(),
            Raises(),
            NotCode(),
            LooksLikeCode(),
            two_of_them,
            none_of_them,
            None,
        )
        for replacement in replacements:
            monkeypatch.setattr(functools, "singledispatch", replacement)
            assert adapter._singledispatch_wrapper_code() is None, replacement
        monkeypatch.setattr(functools, "singledispatch", one_of_them)
        found = adapter._singledispatch_wrapper_code()
        assert type(found) is types.CodeType and found is one_of_them(len).__code__
        monkeypatch.setattr(functools, "singledispatch", pristine)

    def test_how_a_singledispatchmethod_is_known(self, example):
        """By its class (a subclass too), or by the class's name together with a
        ``dispatcher`` that is a dispatch function; what it hands out, by the bound
        ``register`` it carries or by the descriptor it holds as ``_unbound`` (the shape
        Python 3.14 gives it, built by hand here so every Python runs it)."""
        import functools

        adapter = _adapter()

        class Renamed(functools.singledispatchmethod):
            pass

        class Service:
            @Renamed
            def lookup(self, key: str, force: bool = False) -> str:
                """Record a lookup; contacts nothing."""
                return "recorded"

            @functools.singledispatchmethod
            def plain(self, key: str = "") -> str:
                return key

        descriptor = Service.__dict__["lookup"]
        assert type(descriptor).__name__ == "Renamed"
        assert adapter._dispatch_method_descriptor(descriptor) is True
        assert adapter._dispatch_method(Service().lookup) is True
        tool = example.function_tool(Service().lookup, strict_mode=False)
        with pytest.raises(TypeError, match=f"cannot gate 'lookup': {self.REFUSED}"):
            example.gate_tools([tool], example.sql_scope_policy)

        # the name of the class says nothing without a dispatcher under it
        class singledispatchmethod:  # noqa: N801, the name is the point
            dispatcher = None

        assert adapter._dispatch_method_descriptor(singledispatchmethod()) is False
        singledispatchmethod.dispatcher = self.dispatcher()
        assert adapter._dispatch_method_descriptor(singledispatchmethod()) is True

        class HandedOut:
            def __init__(self, unbound):
                self._unbound = unbound

        assert adapter._dispatch_method(HandedOut(Service.__dict__["plain"])) is True
        assert adapter._dispatch_method(HandedOut(None)) is False
        assert adapter._dispatch_method(HandedOut(self.plain_function())) is False

    def test_importing_the_package_under_a_replaced_singledispatch_and_a_reload(self, example):
        """In a process of its own: ``functools.singledispatch`` replaced by an object
        with no ``__code__`` before the import, which made the import raise, then
        ``functools`` reloaded, after which a new dispatcher was gated."""
        import os

        src = ROOT / "adapters" / "openai-agents" / "src"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(src)]))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", _REPLACED_AND_RELOADED],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            "IMPORTED None",
            "early REFUSED",
            "late REFUSED",
            "late-registered REFUSED",
        ]

    # --- an agent tool's parameters type: what puts a field outside ----------------------------

    @staticmethod
    def outside():
        import datetime

        from pydantic import BaseModel, Field, field_validator, model_validator

        class Inner(BaseModel):
            name: str

        class Pattern(BaseModel):
            task: str
            value: str = Field(pattern=r"^[a-z]+$")

        class OptionalPattern(BaseModel):
            task: str
            value: typing.Optional[str] = Field(None, pattern=r"^[a-z]+$")

        class NestedModels(BaseModel):
            task: str
            value: typing.List[Inner]

        class ListOfDates(BaseModel):
            task: str
            value: typing.List[typing.List[datetime.date]] = []

        class InAUnion(BaseModel):
            task: str
            value: typing.Union[int, datetime.date] = 0

        class OptionalDate(BaseModel):
            task: str
            value: typing.Optional[datetime.date] = None

        class FloatLiteral(BaseModel):
            task: str
            value: typing.Literal[0.5, 1.5] = 0.5

        class FieldValidator(BaseModel):
            task: str
            value: str = ""

            @field_validator("value")
            @classmethod
            def check(cls, value):
                return value

        class ModelValidator(BaseModel):
            task: str

            @model_validator(mode="after")
            def check(self):
                return self

        field = "model has a field outside the data-only set: `value` "
        return {
            "a-pattern": (Pattern, field + "(a string pattern). "),
            "an-optional-pattern": (OptionalPattern, field + "(a string pattern). "),
            "a-list-of-models": (NestedModels, field + "(a list of pydantic core type `model`). "),
            "a-list-of-lists-of-dates": (
                ListOfDates,
                field + "(a list of a list of pydantic core type `date`). ",
            ),
            "a-date-in-a-union": (InAUnion, field + "(a union with pydantic core type `date`). "),
            "an-optional-date": (OptionalDate, field + "(pydantic core type `date`). "),
            "a-literal-of-floats": (
                FloatLiteral,
                field + "(a `Literal` of values other than None, a bool, an int or a str). ",
            ),
            "a-field-validator": (
                FieldValidator,
                field + "(a validator, pydantic core type `function-after`). ",
            ),
            "a-model-validator": (
                ModelValidator,
                "type has a validator of its own on the type as a whole (a `model_validator`, "
                "for one; its pydantic core type is `function-after`). ",
            ),
        }

    @pytest.mark.parametrize(
        "case",
        [
            "a-pattern",
            "an-optional-pattern",
            "a-list-of-models",
            "a-list-of-lists-of-dates",
            "a-date-in-a-union",
            "an-optional-date",
            "a-literal-of-floats",
            "a-field-validator",
            "a-model-validator",
        ],
    )
    def test_a_refused_parameters_type_names_what_puts_it_outside(self, example, case):
        """``str`` and ``list`` are in the data-only set: a pattern field was reported as
        "pydantic core type `str`", a list of models as "`list`", and a model with a
        validator as a type that "is not a Pydantic model"."""
        parameters, named = self.outside()[case]
        sub = example.gated_agent(
            example.Agent(name="sub", tools=[example.run_sql]), lambda name, args: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=parameters)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == (
            f"Recusal cannot gate agent tool 'ask_sub': its `parameters` {named}"
            "The gate works out what the sub-agent is given only for a Pydantic model the "
            "SDK's own rule for data-only validation admits (`can_prepare_without_user_code`) "
            "and that is written out with no code of its own. Declare fields that are str, "
            "int, float, bool, None, Literal and lists of them, with JSON defaults, and build "
            "other types inside the sub-agent's own tools."
        )

    def test_the_cause_of_a_schema_the_gate_cannot_read_is_still_a_sentence(self, example):
        cause = _adapter()._outside_cause
        assert cause(None) == "pydantic core type unknown"
        assert cause({}) == "pydantic core type unknown"
        assert cause({"type": "dict", "values_schema": {"type": "date"}}) == (
            "a mapping with values of pydantic core type `date`"
        )
        assert cause({"type": "dict", "keys_schema": {"type": "date"}}) == (
            "a mapping with keys of pydantic core type `date`"
        )
        nested = {"type": "default", "schema": {"type": "nullable", "schema": {"type": "set"}}}
        assert cause(nested) == "pydantic core type `set`"


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestSixthReview:
    """The sixth review of 2026-11-23. A ``functools.partialmethod`` hid what it calls: a
    single-dispatch function or a wrapper with a parameter of its own under it ran with a
    value no policy saw. A ``weakref.proxy`` of a dispatcher was gated. An ordinary event
    bus was taken for a dispatcher. Two sentences said more than is true, the refusal's
    advice could be read as "call the dispatcher from a plain function", and five
    mutations of the gate passed the suite."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording
    DISPATCH = TestFifthReview.REFUSED
    PARTIALMETHOD = "it runs a `functools.partialmethod`"
    PROXY = "a callable the call reaches is a `weakref.proxy`"

    # --- functools.partialmethod ------------------------------------------------------------------

    @staticmethod
    def database(events, functools_module=None):
        """The reviewer's shapes and the ones found while closing them: a partialmethod
        over a class-based decorator, over a ``functools.partial`` and over a plain
        function, each over a single-dispatch function or a wrapper that adds
        ``dry_run``."""
        import functools

        made_by = functools if functools_module is None else functools_module

        class Logged:
            """A class-based decorator: forwards everything, keeps the function's metadata."""

            def __init__(self, inner):
                functools.update_wrapper(self, inner)
                self.inner = inner

            def __call__(self, *args, **kwargs):
                return self.inner(*args, **kwargs)

        @functools.singledispatch
        def export(conn, sql: str = "", force: bool = False) -> str:
            """Record an export; contacts nothing."""
            events.append(("base", sql, force))
            return "base"

        def run_on(conn, sql: str = "", force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            events.append(("ran", sql, force))
            return "recorded"

        def with_option(inner):
            @functools.wraps(inner)
            def wrapper(*args, dry_run="unjudged", **kwargs):
                events.append(("wrapper", dry_run))
                return inner(*args, **kwargs)

            return wrapper

        class Db:
            object_over_dispatcher = made_by.partialmethod(Logged(export))
            partial_over_dispatcher = made_by.partialmethod(functools.partial(export))
            object_over_wrapper = made_by.partialmethod(Logged(with_option(run_on)))
            object_over_wrapper_bound = made_by.partialmethod(
                Logged(with_option(run_on)), force=True
            )
            partial_over_wrapper = made_by.partialmethod(functools.partial(with_option(run_on)))
            object_over_function = made_by.partialmethod(Logged(run_on))
            dispatcher = made_by.partialmethod(export)
            wrapper = made_by.partialmethod(with_option(run_on))
            function = made_by.partialmethod(run_on)
            function_bound = made_by.partialmethod(run_on, force=True)

        def registered(conn, sql="", force=False, dry_run="unjudged"):
            events.append(("registered", sql, force, dry_run))
            return "registered"

        export.register(Db)(registered)
        export.register(str)(registered)
        return Db

    #: name -> (where it is read from, the attribute, the reason it is refused with)
    PARTIALMETHODS = {
        "object-over-dispatcher": ("object", "object_over_dispatcher", "DISPATCH"),
        "partial-over-dispatcher": ("object", "partial_over_dispatcher", "DISPATCH"),
        "object-over-wrapper": ("object", "object_over_wrapper", "PARTIALMETHOD"),
        "object-over-wrapper-bound": ("object", "object_over_wrapper_bound", "PARTIALMETHOD"),
        "partial-over-wrapper": ("object", "partial_over_wrapper", "PARTIALMETHOD"),
        "object-over-function": ("object", "object_over_function", "PARTIALMETHOD"),
        "class-read-object-over-dispatcher": ("class", "object_over_dispatcher", "DISPATCH"),
        "class-read-dispatcher": ("class", "dispatcher", "DISPATCH"),
        "class-read-wrapper": ("class", "wrapper", "PARTIALMETHOD"),
        "class-read-function": ("class", "function", "PARTIALMETHOD"),
        "class-read-function-bound": ("class", "function_bound", "PARTIALMETHOD"),
    }

    def refused(self, example, func, reason):
        """``func`` is refused when it is gated, with ``reason``. Where Python hands out a
        ``functools.partial`` for the partialmethod (over a ``partial`` from 3.14), the
        SDK refuses to build a tool from it, which is the same end."""
        import functools

        if isinstance(func, functools.partial):
            with pytest.raises(Exception, match="functools.partial"):
                example.function_tool(func, name_override="handle", strict_mode=False)
            return
        tool = example.function_tool(func, name_override="handle", strict_mode=False)
        policy = self.Recording(lambda name, args: [])
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], policy)
        assert str(refusal.value).startswith(f"Recusal cannot gate 'handle': {reason}")
        with pytest.raises(TypeError) as refusal:
            example.gated_agent(example.Agent(name="a", tools=[tool]), policy)
        assert str(refusal.value).startswith(f"Recusal cannot gate 'handle': {reason}")
        assert policy.seen == []

    @pytest.mark.parametrize("wrapped", [False, True], ids=["the-tool", "under-a-wrapper"])
    @pytest.mark.parametrize("how", sorted(PARTIALMETHODS))
    def test_a_partialmethod_is_refused_wherever_it_is_read_from(self, example, how, wrapped):
        events: list = []
        where, name, reason = self.PARTIALMETHODS[how]
        db = self.database(events)
        func = getattr(db() if where == "object" else db, name)
        if wrapped:
            func = TestFourthReview.forwarding(func)
        self.refused(example, func, getattr(self, reason))
        assert events == []

    @pytest.mark.parametrize(
        "how, arguments",
        [
            ("object-over-dispatcher", {"sql": "x"}),
            ("class-read-dispatcher", {"conn": "c", "sql": "x"}),
        ],
    )
    def test_negative_control_ungated_a_dispatcher_under_a_partialmethod_runs_its_own(
        self, example, how, arguments
    ):
        """Why it is refused: the tool's schema is the base function's, and the registered
        implementation runs with ``dry_run`` at its own default."""
        events: list = []
        where, name, _ = self.PARTIALMETHODS[how]
        db = self.database(events)
        func = getattr(db() if where == "object" else db, name)
        tool = example.function_tool(func, name_override="handle", strict_mode=False)
        assert "dry_run" not in tool.params_json_schema["properties"]
        outputs, _ = example.run_calls([("handle", arguments, "c1")], gated=False, tools=[tool])
        assert outputs == {"c1": "registered"}
        assert events == [("registered", "x", False, "unjudged")]

    def test_negative_control_ungated_a_wrapper_under_a_partialmethod_runs_its_own(self, example):
        events: list = []
        func = self.database(events)().object_over_wrapper
        tool = example.function_tool(func, name_override="handle", strict_mode=False)
        assert "dry_run" not in tool.params_json_schema["properties"]
        example.run_calls([("handle", {"sql": "x"}, "c1")], gated=False, tools=[tool])
        assert events == [("wrapper", "unjudged"), ("ran", "x", False)]

    def test_a_partialmethod_over_a_plain_method_read_from_an_object_is_the_sdks_to_refuse(
        self, example
    ):
        """Read from an object, a partialmethod over a plain function is a
        ``functools.partial``, and the SDK builds no tool from one. The README says so."""
        import functools

        func = self.database([])().function_bound
        assert isinstance(func, functools.partial)
        with pytest.raises(Exception, match="functools.partial"):
            example.function_tool(func, name_override="handle", strict_mode=False)

    def test_the_partialmethod_refusal_says_the_cause_and_what_to_do(self, example):
        tool = example.function_tool(
            self.database([])().object_over_wrapper, name_override="handle", strict_mode=False
        )
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == "Recusal cannot gate 'handle': " + PARTIALMETHOD_REASON

    def test_how_a_partialmethod_is_known(self, example, monkeypatch):
        """By its class (a subclass too) as it was at import and as ``functools`` has it
        now, or by the class's name; on the function ``functools`` hands out, under the
        name each Python gives it. The walk steps to the callable it holds."""
        import functools

        adapter = _adapter()
        events: list = []
        db = self.database(events)
        held = db.__dict__["object_over_wrapper"]
        assert adapter._is_partialmethod(held) is True
        assert adapter._held_partialmethod(held) is held
        bound = db().object_over_wrapper
        assert adapter._held_partialmethod(bound) is None  # the method object carries none
        assert adapter._held_partialmethod(bound.__func__) is held
        assert adapter._held_partialmethod(db.function) is db.__dict__["function"]
        carried = [name for name in adapter._PARTIALMETHOD_NAMES if name in vars(bound.__func__)]
        assert carried == ["_partialmethod" if sys.version_info < (3, 13) else "__partialmethod__"]
        reached, complete = adapter._reached_callables(bound)
        assert complete and any(link is held.func for link in reached)
        assert adapter._steps(held) == [held.func]

        class Renamed(functools.partialmethod):
            pass

        class partialmethod:  # noqa: N801, the name is the point
            func = None

        def plain():
            return None

        for name in ("_partialmethod", "__partialmethod__", "partialmethod"):
            carrier = TestFifthReview.plain_function()
            setattr(carrier, name, Renamed(plain))
            assert adapter._held_partialmethod(carrier) is getattr(carrier, name)
            setattr(carrier, name, plain)  # under the name, something that is not one
            assert adapter._held_partialmethod(carrier) is None
        for not_one in (plain, None, "text", functools.partial(plain), db, db()):
            assert adapter._is_partialmethod(not_one) is False
            assert adapter._held_partialmethod(not_one) is None

        renamed = Renamed(plain)
        assert adapter._is_partialmethod(renamed) is True
        assert adapter._is_partialmethod(partialmethod()) is True  # by the name alone
        with monkeypatch.context() as patch:  # the class as it was at import, alone
            patch.setattr(functools, "partialmethod", None)
            assert adapter._is_partialmethod(renamed) is True
        with monkeypatch.context() as patch:  # the class as functools has it now, alone
            patch.setattr(adapter, "_PARTIALMETHOD", None)
            assert adapter._is_partialmethod(renamed) is True
            patch.setattr(functools, "partialmethod", None)
            assert adapter._is_partialmethod(renamed) is False
        assert events == []

    def test_a_partialmethod_built_by_another_copy_of_functools_is_refused(self, example):
        """What ``importlib.reload(functools)`` leaves behind: a partialmethod of another
        class. ``inspect`` does not read through that one, so the SDK shows the tool the
        ``*args`` of the function ``functools`` hands out and the gate refuses it as
        variadic; asked on its own, the walk's check knows it by the class's name."""
        import functools
        import importlib.util

        adapter = _adapter()
        spec = importlib.util.find_spec("functools")
        assert spec is not None and spec.loader is not None
        other = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(other)
        assert other.partialmethod is not functools.partialmethod
        db = self.database([], functools_module=other)
        assert not isinstance(db.__dict__["object_over_wrapper"], functools.partialmethod)
        for func, reason in (
            (db().object_over_wrapper, self.PARTIALMETHOD),
            (db.function, self.PARTIALMETHOD),
            (db().object_over_dispatcher, self.DISPATCH),
        ):
            tool = example.function_tool(func, name_override="handle", strict_mode=False)
            with pytest.raises(
                TypeError, match="cannot gate 'handle': its parameter .* is variadic"
            ):
                example.gate_tools([tool], example.sql_scope_policy)
            assert adapter._wrapper_refusal(tool, adapter._sdk_schema(tool)).startswith(reason)

    def test_asking_for_a_partialmethod_runs_no_code_of_the_object(self, example):
        touched: list = []

        class Lazy:
            def __getattr__(self, name):
                touched.append(name)
                raise AttributeError(name)

            @property
            def _partialmethod(self):
                touched.append("_partialmethod")
                return None

            @property
            def __partialmethod__(self):
                touched.append("__partialmethod__")
                return None

            @property
            def partialmethod(self):
                touched.append("partialmethod")
                return None

            @property
            def func(self):
                touched.append("func")
                return None

        assert _adapter()._held_partialmethod(Lazy()) is None
        assert _adapter()._is_partialmethod(Lazy()) is False
        assert touched == []

        class Claims:
            """Says it is a partialmethod through ``__class__``; its type is asked."""

            @property
            def __class__(self):
                import functools

                touched.append("__class__")
                return functools.partialmethod

        assert _adapter()._is_partialmethod(Claims()) is False
        assert touched == []

    def test_a_kernel_that_refuses_a_partialmethod_itself_gets_the_same_reason(
        self, example, monkeypatch
    ):
        """The kernel's walk raises ``PartialMethodSignature`` (a ``TypeError``) for a
        callable read through a partialmethod, and the refusal gives the remedy that holds
        here. It is the kernel's own class that is raised here (``recusal`` 0.16.0 is the
        floor)."""
        from recusal.invocation import PartialMethodSignature

        adapter = _adapter()
        tool = example.function_tool(TestFifthReview.plain_function())

        def walk(func):
            raise PartialMethodSignature("read through a functools.partialmethod")

        monkeypatch.setattr(adapter, "_reached_parameters", walk)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        reason = str(refusal.value)
        assert reason.startswith(f"Recusal cannot gate 'run_sql': {self.PARTIALMETHOD}")
        assert reason == "Recusal cannot gate 'run_sql': " + PARTIALMETHOD_REASON
        assert "cannot be inspected" not in reason

        def other(func):
            raise TypeError("something else")

        monkeypatch.setattr(adapter, "_reached_parameters", other)
        with pytest.raises(TypeError, match=r"cannot be inspected \(TypeError\)"):
            example.gate_tools([tool], example.sql_scope_policy)

    # --- weakref.proxy ----------------------------------------------------------------------------

    @staticmethod
    def proxied(events):
        import functools
        import weakref

        @functools.singledispatch
        def export(sql: str = "", force: bool = False) -> str:
            """Record an export; contacts nothing."""
            events.append(("base", sql, force))
            return "base"

        @export.register(str)
        def _(sql="", force=False, dry_run="unjudged"):
            events.append(("registered", sql, force, dry_run))
            return "registered"

        def plain(sql: str = "", force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            events.append(("ran", sql, force))
            return "recorded"

        class Service:
            def handle(self, sql: str = "", force: bool = False) -> str:
                """Record SQL text without connecting to a database."""
                return "recorded"

        service = Service()
        keep = [export, plain, service]  # a weak proxy keeps nothing alive
        return keep, {
            "a-dispatcher": weakref.proxy(export),
            "a-plain-function": weakref.proxy(plain),
            "under-a-wrapper": TestFourthReview.forwarding(weakref.proxy(plain)),
            "as-a-methods-function": types_method(weakref.proxy(plain), service),
        }

    @pytest.mark.parametrize(
        "how", ["a-dispatcher", "a-plain-function", "under-a-wrapper", "as-a-methods-function"]
    )
    def test_a_weak_proxy_is_refused_wherever_it_stands(self, example, how):
        import weakref

        events: list = []
        keep, shapes = self.proxied(events)
        func = shapes[how]
        reached, _ = _adapter()._reached_callables(func)
        assert any(type(link) in weakref.ProxyTypes for link in reached)
        assert not any(_adapter()._is_dispatcher(link) for link in reached)
        tool = example.function_tool(func, name_override="handle", strict_mode=False)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == "Recusal cannot gate 'handle': " + WEAK_PROXY_REASON
        with pytest.raises(TypeError, match=f"cannot gate 'handle': {self.PROXY}"):
            example.gated_agent(example.Agent(name="a", tools=[tool]), example.sql_scope_policy)
        assert events == [] and keep

    def test_a_weak_proxy_that_is_not_callable_is_refused_too(self, example):
        """Both proxy types are refused: a function whose ``__wrapped__`` is changed, after
        the tool was gated, to a proxy of an object that is not callable."""
        import weakref

        class Holder:
            pass

        holder = Holder()
        run_sql = TestFifthReview.plain_function()
        tool = example.function_tool(run_sql)
        assert example.gate_tools([tool], example.sql_scope_policy)
        run_sql.__wrapped__ = weakref.proxy(holder)  # type: ignore[attr-defined]
        assert type(run_sql.__wrapped__) is weakref.ProxyType  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match=f"cannot gate 'run_sql': {self.PROXY}"):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_negative_control_ungated_a_proxied_dispatcher_runs_its_own(self, example):
        events: list = []
        keep, shapes = self.proxied(events)
        tool = example.function_tool(
            shapes["a-dispatcher"], name_override="handle", strict_mode=False
        )
        assert "dry_run" not in tool.params_json_schema["properties"]
        example.run_calls([("handle", {"sql": "x"}, "c1")], gated=False, tools=[tool])
        assert events == [("registered", "x", False, "unjudged")] and keep

    # --- the closure test: functools' own free variables -----------------------------------------

    @staticmethod
    def event_bus(ran):
        """An ordinary event bus: a closure named ``dispatch`` over a dict named
        ``registry``, and nothing of ``functools``."""
        registry: dict = {}

        def dispatch(event):
            return registry.get(event)

        def publish(event: str, payload: str = "") -> str:
            """Record an event; contacts nothing."""
            ran.append((event, payload))
            handler = dispatch(event)
            return "no handler" if handler is None else handler(payload)

        return publish

    def test_an_ordinary_event_bus_is_not_taken_for_a_dispatcher(self, example):
        adapter = _adapter()
        ran: list = []
        publish = self.event_bus(ran)
        dispatch = adapter._closure_value(publish, "dispatch")
        assert type(adapter._closure_value(dispatch, "registry")) is dict
        assert adapter._dispatch_function(publish) is False
        policy = self.Recording(lambda name, args: [])
        (gated,) = example.gate_tools([example.function_tool(publish)], policy)
        outputs, _ = example.run_calls(
            [("publish", {"event": "e"}, "c1")], gated=False, tools=[gated]
        )
        assert outputs == {"c1": "no handler"} and ran == [("e", "")]
        assert policy.seen == [("publish", {"event": "e", "payload": ""})]

    @pytest.mark.parametrize(
        "case, expected",
        [
            ("as-functools-builds-it", True),
            ("no-dispatch-cache", False),
            ("no-cache-token", False),
            ("a-registry-that-is-no-dict", False),
        ],
    )
    def test_the_closure_test_asks_for_a_registry_dict_and_both_of_functools_names(
        self, example, monkeypatch, case, expected
    ):
        """Asked with the code test and the attribute test out of the way: a function is a
        dispatcher by its closure only when ``dispatch`` closes over a ``registry`` that
        is a dict, a ``dispatch_cache`` and a ``cache_token``."""
        import types

        adapter = _adapter()
        dispatch_cache: dict = {}
        cache_token = None
        registry: typing.Any = {}
        if case == "a-registry-that-is-no-dict":
            registry = types.MappingProxyType({})

        if case == "no-dispatch-cache":

            def dispatch(cls):
                return registry, cache_token

        elif case == "no-cache-token":

            def dispatch(cls):
                return registry, dispatch_cache

        else:

            def dispatch(cls):
                return registry, dispatch_cache, cache_token

        def wrapper(*args, **kwargs):
            return dispatch(args[0].__class__)

        monkeypatch.setattr(adapter, "_SINGLEDISPATCH_WRAPPER", None)
        monkeypatch.setattr(adapter, "_singledispatch_wrapper_code", lambda: None)
        assert adapter._dispatch_function(wrapper) is expected
        # every dispatcher functools builds has the three, on the Python this runs on
        real = adapter._closure_value(TestFifthReview.dispatcher(), "dispatch")
        assert {"registry", "dispatch_cache", "cache_token"} <= set(real.__code__.co_freevars)
        bare = TestFifthReview.dispatcher()
        for name in ("registry", "dispatch", "register"):
            delattr(bare, name)
        assert adapter._dispatch_function(bare) is True

    # --- checks the suite did not pin -------------------------------------------------------------

    def test_call_is_read_from_the_type_as_python_reads_it(self, example):
        """Python calls the ``__call__`` of an object's type and never one set on the
        object, so the walk reads the type's: a dispatcher there is found whatever the
        object's own ``__dict__`` holds under that name."""
        import functools

        adapter = _adapter()
        ran: list = []

        class Runner:
            __call__ = functools.singledispatch(
                lambda self, sql="", force=False: ran.append(("base", sql, force)) or "recorded"
            )

        def decoy(sql: str = "", force: bool = False) -> str:
            return "decoy"

        runner = Runner()
        runner.__dict__["__call__"] = decoy
        assert runner() == "recorded" and ran == [("base", "", False)]
        steps = adapter._steps(runner)
        assert any(link is Runner.__dict__["__call__"] for link in steps)
        assert not any(link is decoy for link in steps)
        tool = example.function_tool(
            TestFourthReview.forwarding(runner), name_override="handle", strict_mode=False
        )
        with pytest.raises(TypeError, match=f"cannot gate 'handle': {self.DISPATCH}"):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_a_singledispatchmethod_is_known_by_either_class_alone(self, example, monkeypatch):
        """The class as it was at import and the class as ``functools`` has it now are two
        tests, and each is enough: an object of a subclass under another name, which the
        name test does not know, is known by the one and by the other."""
        import functools
        import importlib.util

        adapter = _adapter()
        spec = importlib.util.find_spec("functools")
        assert spec is not None and spec.loader is not None
        other = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(other)
        pristine = functools.singledispatchmethod
        assert adapter._SINGLEDISPATCHMETHOD is pristine
        assert other.singledispatchmethod is not pristine

        class Early(pristine):  # of the class this package saw when it was imported
            pass

        class Late(other.singledispatchmethod):  # of the class functools has afterwards
            pass

        def lookup(self, key: str = "") -> str:
            return key

        early, late = Early(lookup), Late(lookup)
        assert adapter._dispatch_method_descriptor(early) is True
        assert adapter._dispatch_method_descriptor(late) is False
        monkeypatch.setattr(functools, "singledispatchmethod", other.singledispatchmethod)
        assert adapter._dispatch_method_descriptor(early) is True  # as it was at import
        assert adapter._dispatch_method_descriptor(late) is True  # as functools has it now
        assert adapter._is_dispatcher(early) and adapter._is_dispatcher(late)

    def test_the_singledispatchmethod_object_itself_is_a_dispatcher(self, example):
        """The walk reaches the descriptor itself when it is a callable object's
        ``__call__`` as the class holds it."""
        import functools

        adapter = _adapter()

        class Renamed(functools.singledispatchmethod):
            pass

        class Runner:
            @Renamed
            def __call__(self, sql: str = "", force: bool = False) -> str:
                """Record SQL text without connecting to a database."""
                return "recorded"

        descriptor = Runner.__dict__["__call__"]
        assert adapter._dispatch_method(descriptor) is True
        assert adapter._is_dispatcher(descriptor) is True
        reached, complete = adapter._reached_callables(Runner())
        assert complete and any(link is descriptor for link in reached)
        tool = example.function_tool(
            TestFourthReview.forwarding(Runner()), name_override="handle", strict_mode=False
        )
        with pytest.raises(TypeError, match=f"cannot gate 'handle': {self.DISPATCH}"):
            example.gate_tools([tool], example.sql_scope_policy)

    # --- what the README says about a change made during a run -----------------------------------

    def test_a_change_a_tool_body_makes_is_seen_from_the_next_model_turn(self, example):
        """The guardrail asks its questions again on every call, and the SDK runs every
        guardrail of a model turn before any tool body of that turn. So a tool body that
        re-wraps another tool's function is not seen by a call of the same turn, which
        runs the new wrapper unjudged, and is seen by the call of the next turn, which is
        refused. The README states both halves."""
        import functools

        events: list = []

        def base(sql: str, force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            events.append(("base", sql, force))
            return "recorded"

        @functools.wraps(base)
        def run_sql(*args, **kwargs):
            return run_sql.__wrapped__(*args, **kwargs)  # type: ignore[attr-defined]

        def enable_tracing(level: str) -> str:
            """Record a tracing level; contacts nothing."""
            inner = run_sql.__wrapped__  # type: ignore[attr-defined]

            @functools.wraps(inner)
            def traced(*args, dry_run="unjudged", **kwargs):
                events.append(("traced", dry_run))
                return inner(*args, **kwargs)

            run_sql.__wrapped__ = traced  # type: ignore[attr-defined]
            return "tracing on"

        model = example.ScriptedModel(
            [
                [
                    example.function_call("enable_tracing", {"level": "x"}, call_id="c1"),
                    example.function_call("run_sql", {"sql": "same turn"}, call_id="c2"),
                ],
                [example.function_call("run_sql", {"sql": "next turn"}, call_id="c3")],
                [example.assistant_message("done")],
            ]
        )
        policy = self.Recording(lambda name, args: [])
        tools = [
            example.function_tool(run_sql, name_override="run_sql"),
            example.function_tool(enable_tracing),
        ]
        agent = example.gated_agent(example.Agent(name="a", model=model, tools=tools), policy)
        asyncio.run(example.run_agent(agent))
        outputs = example.tool_outputs(model)
        assert outputs["c1"] == "tracing on" and outputs["c2"] == "recorded"
        assert events == [("traced", "unjudged"), ("base", "same turn", False)]
        assert outputs["c3"].startswith(
            "Recusal refused `run_sql`: a function the call reaches takes `dry_run`"
        )
        assert [name for name, _ in policy.seen] == ["enable_tracing", "run_sql"]


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestKernelRefusals:
    """0.2.2, beside ``recusal`` 0.16.0. The kernel's walk refuses, by a class of its own,
    what it cannot follow to the end. Two of those classes reached the developer as
    "cannot be inspected (BoundArgumentKept)" and "cannot be inspected (UnreadCache)", a
    declared signature as "cannot be inspected (DeclaredSignature)", and a hidden wrapper
    with a sentence that was not true of every one ("takes a parameter of its own").

    0.2.3, beside ``recusal`` 0.17.0, which adds ``OpenDispatch``, ``ProxiedCallable`` and
    ``WeakProxy``: beside 0.2.2 each was "cannot be inspected" with the class's name.
    Each has a reason now, and the gate's own walk follows what a cache named again calls.

    One test for each class. The tool is a real one of that shape and the kernel's own
    walk raises; nothing stands in for the kernel. Where the gate's own walk finds the
    shape first (a dispatcher, a weak proxy), its question is turned off to reach the
    kernel's refusal, and the test says so. Each is refused twice: when it is
    gated, before any call, and by the guardrail, for a tool whose function is changed
    into that shape after it was gated. Both times the policy is not asked and no body
    runs, and the reason holds no default and no argument value."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording

    #: A default of every wrapper below and the argument of every call: never in a reason.
    SECRET = "SECRET-VALUE-93417"
    #: How every reason that says what to write ends.
    NOT_READ = BODY_NOT_READ

    HIDDEN = HIDDEN_REASON
    UNREAD_CACHE = UNREAD_CACHE_REASON
    BOUND = BOUND_REASON
    DECLARED = DECLARED_REASON
    DISPATCH = SINGLEDISPATCH_REASON
    PARTIALMETHOD = PARTIALMETHOD_REASON
    KEPT_RECEIVER = KEPT_RECEIVER_REASON

    # --- the tools ------------------------------------------------------------------------------

    def fetch(self, ran):
        secret = self.SECRET

        def fetch(sql: str = secret, limit: int = 10) -> str:
            """Record SQL text without connecting to a database."""
            ran.append(("fetch", sql, limit))
            return "recorded"

        return fetch

    def with_option(self, ran, inner):
        """A wrapper with a parameter of its own, which runs at a default nobody judged."""
        import functools

        secret = self.SECRET

        @functools.wraps(inner)
        def wrapper(*args, dry_run=secret, **kwargs):
            ran.append(("wrapper", dry_run))
            return inner(*args, **kwargs)

        return wrapper

    def holding(self, ran, inner):
        """A decorator that applies ``with_option`` inside itself and calls the result,
        while ``functools.wraps`` names the undecorated function."""
        import functools

        hidden = self.with_option(ran, inner)

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            return hidden(*args, **kwargs)

        return wrapper

    def facade(self, ran):
        """The bound method of a facade that is ``functools.wraps`` of a plain function
        and passes a value of its own in the place of the first parameter."""
        import functools

        secret = self.SECRET

        def search(tenant: str = secret, sql: str = "", limit: int = 10) -> str:
            """Record a search without contacting anything."""
            ran.append(("search", tenant, sql, limit))
            return "recorded"

        function = search

        class Tools:
            tenant = "acme"

            @functools.wraps(function)
            def search(self, *args, **kwargs):
                return function(self.tenant, *args, **kwargs)

        return Tools().search

    @staticmethod
    def bare(inner):
        """A function that names ``inner`` as ``__wrapped__`` and copies nothing else of
        it, so nothing of ``inner`` can be told from its attributes."""

        def wrapper(*args, **kwargs):
            return inner(*args, **kwargs)

        wrapper.__wrapped__ = inner  # type: ignore[attr-defined]
        return wrapper

    @staticmethod
    def named_again(called, named):
        """``functools.lru_cache`` over ``called``, named again for ``named``: what
        ``functools.wraps(named)`` written above the cache gives."""
        import functools

        return functools.update_wrapper(functools.lru_cache(maxsize=None)(called), named)

    def dispatcher(self, ran, register):
        import functools

        secret = self.SECRET

        @functools.singledispatch
        def export(sql: str = "", limit: int = 10) -> str:
            """Record an export; contacts nothing."""
            ran.append(("base", sql, limit))
            return "recorded"

        def registered(sql="", limit=10, force=secret):
            ran.append(("registered", sql, limit, force))
            return "recorded"

        if register:
            export.register(str)(registered)
        return export, lambda: export.register(str)(registered)

    def partialmethod(self, ran):
        """The method a ``functools.partialmethod`` over a class-based decorator hands
        out, over a wrapper with a parameter of its own."""
        import functools

        def run_on(conn, sql: str = "", limit: int = 10) -> str:
            """Record SQL text without connecting to a database."""
            ran.append(("run_on", sql, limit))
            return "recorded"

        class Logged:
            def __init__(self, inner):
                functools.update_wrapper(self, inner)
                self.inner = inner

            def __call__(self, *args, **kwargs):
                return self.inner(*args, **kwargs)

        class Db:
            handle = functools.partialmethod(Logged(self.with_option(ran, run_on)))

        return Db().handle

    # --- the two refusals -----------------------------------------------------------------------

    def refused_when_gated(self, example, func, reason, ran):
        """``func`` is refused as a tool by ``gate_tools`` and by ``gated_agent``, with
        exactly ``reason``, before the policy is asked and with no body run."""
        tool = example.function_tool(func, name_override="handle", strict_mode=False)
        policy = self.Recording(lambda name, args: [])
        for gate in (
            lambda: example.gate_tools([tool], policy),
            lambda: example.gated_agent(example.Agent(name="a", tools=[tool]), policy),
        ):
            with pytest.raises(TypeError) as refusal:
                gate()
            assert str(refusal.value) == f"Recusal cannot gate 'handle': {reason}"
        assert self.SECRET not in reason
        assert policy.seen == [] and ran == []

    def refused_when_called(self, example, change, reason, ran):
        """A plain tool is gated and runs. ``change(function)`` then turns its function
        into the shape, and the next call is refused by the guardrail with exactly
        ``reason``: the policy is not asked again and no body runs."""

        def handle(sql: str = "", limit: int = 10) -> str:
            """Record SQL text without connecting to a database."""
            ran.append(("handle", sql, limit))
            return "recorded"

        policy = self.Recording(lambda name, args: [])
        (gated,) = example.gate_tools([example.function_tool(handle, strict_mode=False)], policy)
        outputs, _ = example.run_calls(
            [("handle", {"sql": SELECT}, "c1")], gated=False, tools=[gated]
        )
        assert outputs == {"c1": "recorded"} and ran == [("handle", SELECT, 10)]
        assert policy.seen == [("handle", {"sql": SELECT, "limit": 10})]
        change(handle)
        outputs, _ = example.run_calls(
            [("handle", {"sql": self.SECRET}, "c2")], gated=False, tools=[gated]
        )
        assert outputs == {"c2": f"Recusal refused `handle`: {reason}"}
        assert self.SECRET not in outputs["c2"]
        assert len(policy.seen) == 1 and ran == [("handle", SELECT, 10)]

    @staticmethod
    def wrapping(shape):
        """Make a gated function a wrapper of ``shape``: the walk then reaches it."""

        def change(function):
            function.__wrapped__ = shape

        return change

    # --- HiddenWrapper --------------------------------------------------------------------------

    def test_a_hidden_wrapper_says_what_the_kernel_found(self, example):
        """``HiddenWrapper``: a decorator applies a second wrapper inside itself and calls
        it. The reason carries the kernel's text, which names the parameter, and no longer
        says the wrapper "takes a parameter of its own" of a refusal the kernel makes for
        another cause."""
        from recusal import invocation

        ran: list = []
        found = (
            "`dry_run` is a parameter of a function the call reaches that the tool does not "
            "declare, so it would run with a value no policy judged"
        )
        with pytest.raises(invocation.HiddenWrapper) as raised:
            invocation.reached_parameters(self.holding(ran, self.fetch(ran)))
        assert type(raised.value) is invocation.HiddenWrapper and str(raised.value) == found
        self.refused_when_gated(
            example, self.holding(ran, self.fetch(ran)), self.HIDDEN.format(found), ran
        )
        self.refused_when_called(
            example,
            self.wrapping(self.holding(ran, self.fetch(ran))),
            self.HIDDEN.format(found),
            ran,
        )

    def test_a_hidden_way_that_cannot_be_judged_is_not_said_to_take_a_parameter(self, example):
        """``HiddenWrapper`` for another cause: the held wrapper's way to the function
        passes a single-dispatch function with an implementation registered, so the way
        cannot be judged by parameters and no parameter was found. The reason used to say
        "that wrapper takes a parameter of its own" of this too."""
        import functools

        from recusal import invocation

        ran: list = []
        fetch = self.fetch(ran)
        export = functools.singledispatch(fetch)
        export.register(str)(lambda sql="", limit=10, force=self.SECRET: "registered")
        # a second wrapper, of the dispatcher, that the tool holds in its closure
        held = functools.update_wrapper(lambda *args, **kwargs: export(*args, **kwargs), export)

        @functools.wraps(fetch)
        def tool(*args, **kwargs):
            return held(*args, **kwargs)

        with pytest.raises(invocation.HiddenWrapper) as raised:
            invocation.reached_parameters(tool)
        found = str(raised.value)
        assert found == invocation._HIDDEN_WRAPPER and "parameter" not in found
        reason = self.HIDDEN.format(found)
        assert "takes a parameter of its own" not in reason
        self.refused_when_gated(example, tool, reason, ran)

    # --- UnreadCache ----------------------------------------------------------------------------

    def test_a_cache_whose_function_cannot_be_told_has_a_reason_of_its_own(
        self, example, monkeypatch
    ):
        """``UnreadCache``, a ``HiddenWrapper``: what a ``functools.lru_cache`` wrapper
        refers to is not laid out as the kernel knows it, so the function it calls cannot
        be told. No supported interpreter lays it out another way, so the listing is
        shortened here, as the kernel's own test of this refusal does; the cache, the
        walk and the class raised are the real ones."""
        from recusal import invocation

        ran: list = []
        fetch = self.fetch(ran)
        invocation.reached_parameters(fetch)  # the first walk has been made
        listing = invocation._referents
        monkeypatch.setattr(invocation, "_referents", lambda value: listing(value)[:-1])

        def cache():
            return self.named_again(lambda *args, **kwargs: fetch(*args, **kwargs), fetch)

        with pytest.raises(invocation.UnreadCache) as raised:
            invocation.reached_parameters(cache())
        assert isinstance(raised.value, invocation.HiddenWrapper)
        assert "holds a second wrapper" not in self.UNREAD_CACHE
        # since 0.2.3 the gate's own walk asks the kernel's ``cache_target`` what the cache
        # calls, and meets the refusal first; it is the same reason, not the one for a
        # callable the gate cannot question
        with pytest.raises(invocation.UnreadCache):
            _adapter()._reached_callables(cache())
        self.refused_when_gated(example, cache(), self.UNREAD_CACHE, ran)
        self.refused_when_called(example, self.wrapping(cache()), self.UNREAD_CACHE, ran)

    # --- BoundArgumentKept ----------------------------------------------------------------------

    def test_a_bound_argument_a_wrapper_takes_under_another_name_has_a_reason(self, example):
        """``BoundArgumentKept``: a method takes its object as ``self`` over a function
        whose first parameter is ``tenant``, which has a default. The reason carries the
        kernel's text, which names the two parameters and says what to do."""
        from recusal import invocation

        ran: list = []
        with pytest.raises(invocation.BoundArgumentKept) as raised:
            invocation.reached_parameters(self.facade(ran))
        found = str(raised.value)
        assert "is taken by a wrapper as `self`" in found
        assert "`tenant`, the last they would fill, runs at a default no policy judged" in found
        # the kernel's own advice, which 0.17.1 rewrote
        assert found.endswith(
            getattr(
                invocation,
                "_RENAMED_REMEDY",
                "names every parameter the tool takes, and register that one",
            )
        )
        self.refused_when_gated(example, self.facade(ran), self.BOUND.format(found), ran)
        self.refused_when_called(
            example, self.wrapping(self.facade(ran)), self.BOUND.format(found), ran
        )

    # --- DeclaredSignature ----------------------------------------------------------------------

    def test_a_declared_signature_has_a_reason(self, example):
        """``DeclaredSignature``: the function carries a ``__signature__`` that leaves a
        parameter out, which is what ``inspect.signature`` and the SDK read."""
        import inspect

        from recusal import invocation

        ran: list = []
        shown = inspect.Signature(
            [inspect.Parameter("sql", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)]
        )

        def declare(function):
            function.__signature__ = shown

        fetch = self.fetch(ran)
        declare(fetch)
        with pytest.raises(invocation.DeclaredSignature):
            invocation.reached_parameters(fetch)
        self.refused_when_gated(example, fetch, self.DECLARED, ran)
        self.refused_when_called(example, declare, self.DECLARED, ran)

    # --- DispatchedAtCallTime and OpenDispatch -------------------------------------------------

    def test_a_dispatcher_behind_a_cache_named_again_is_found_by_the_gates_own_walk(self, example):
        """0.2.3. A single-dispatch function behind a cache that was named again for
        another function. The walk of 0.2.2 followed what the cache names and reached no
        dispatcher, so one with nothing registered was gated beside ``recusal`` 0.16.0. The
        gate's own walk now also follows what the cache calls (the kernel's
        ``cache_target``), finds the dispatcher, registered or not, and refuses it with the
        reason it gives every dispatcher it finds. The kernel's walk refuses the same tool,
        as a ``DispatchedAtCallTime`` or an ``OpenDispatch``, and the reason for the first
        is the gate's own as well."""
        from recusal import invocation

        adapter = _adapter()
        ran: list = []
        fetch = self.fetch(ran)

        def tool(register):
            export, register_now = self.dispatcher(ran, register)
            return self.named_again(self.bare(export), fetch), register_now

        for register, kernel_class in (
            (True, invocation.DispatchedAtCallTime),
            (False, invocation.OpenDispatch),
        ):
            func, _ = tool(register)
            reached, complete = adapter._reached_callables(func)
            assert complete and any(adapter._is_dispatcher(link) for link in reached)
            with pytest.raises(kernel_class) as raised:
                invocation.reached_parameters(func)
            if register:
                assert adapter._unread_reason(raised.value, reached) == self.DISPATCH
            self.refused_when_gated(example, func, self.DISPATCH, ran)

        # a dispatcher with nothing registered is refused at the next call, before anything
        # is registered on it
        func, _ = tool(register=False)
        self.refused_when_called(example, self.wrapping(func), self.DISPATCH, ran)

    OPEN_DISPATCH = OPEN_DISPATCH_REASON

    def test_an_open_dispatcher_only_the_kernel_finds_has_a_reason_of_its_own(
        self, example, monkeypatch
    ):
        """``OpenDispatch`` (``recusal`` 0.17.0): a single-dispatch function with nothing
        registered but its base. Beside 0.2.2 it was "the function it runs cannot be
        inspected (OpenDispatch)". The gate's own walk refuses every dispatcher it reaches
        before the kernel's walk is asked, and since 0.2.3 it stands wherever the kernel's
        does, so no tool is known for which only the kernel finds one. The gate's own
        question is turned off here to stand for such a tool; the dispatcher, the walk and
        the class raised are the real ones."""
        import functools

        from recusal import invocation

        adapter = _adapter()
        ran: list = []
        export, register_now = self.dispatcher(ran, register=False)
        with pytest.raises(invocation.OpenDispatch) as raised:
            invocation.reached_parameters(export)
        assert not isinstance(raised.value, invocation.DispatchedAtCallTime)
        assert adapter._unread_reason(raised.value, []) == self.OPEN_DISPATCH
        monkeypatch.setattr(adapter, "_is_dispatcher", lambda link: False)

        @functools.wraps(export)
        def tool(*args, **kwargs):
            return export(*args, **kwargs)

        self.refused_when_gated(example, tool, self.OPEN_DISPATCH, ran)
        self.refused_when_called(example, self.wrapping(tool), self.OPEN_DISPATCH, ran)
        ran.clear()
        assert "nothing registered" in self.OPEN_DISPATCH
        assert "would then run for the judged call" in self.OPEN_DISPATCH
        # once an implementation is registered it is a ``DispatchedAtCallTime``
        register_now()
        with pytest.raises(invocation.DispatchedAtCallTime):
            invocation.reached_parameters(export)
        self.refused_when_gated(example, tool, self.DISPATCH, ran)

    # --- WeakProxy and ProxiedCallable ----------------------------------------------------------

    def test_a_weak_proxy_only_the_kernel_finds_gets_the_gates_own_reason(
        self, example, monkeypatch
    ):
        """``WeakProxy`` (``recusal`` 0.17.0, a ``ProxiedCallable``): beside 0.2.2 it was
        "the function it runs cannot be inspected (WeakProxy)". It gets the reason the
        gate gives a weak proxy its own walk finds. As for a dispatcher, the gate's own
        walk finds every proxy it reaches first, so its question is turned off here."""
        import types
        import weakref

        from recusal import invocation

        adapter = _adapter()
        ran: list = []
        fetch = self.fetch(ran)
        with pytest.raises(invocation.WeakProxy) as raised:
            invocation.reached_parameters(weakref.proxy(fetch))
        assert adapter._unread_reason(raised.value, []) == self.WEAK_PROXY
        monkeypatch.setattr(adapter, "weakref", types.SimpleNamespace(ProxyTypes=()))
        tool = self.bare(weakref.proxy(fetch))
        self.refused_when_gated(example, tool, self.WEAK_PROXY, ran)
        self.refused_when_called(example, self.wrapping(tool), self.WEAK_PROXY, ran)

    WEAK_PROXY = WEAK_PROXY_REASON
    PROXIED = PROXIED_REASON

    class Forwards:
        """A class-based decorator that keeps the function in an attribute and hands every
        attribute it does not have on to it."""

        def __init__(self, fn):
            self.fn = fn

        def __getattr__(self, name):
            return getattr(self.fn, name)

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    def test_a_decorator_that_answers_for_wrapped_has_a_reason_of_its_own(self, example):
        """``ProxiedCallable`` (``recusal`` 0.17.0): a class-based decorator that hands every
        attribute on with ``__getattr__``, over a ``functools.wraps`` wrapper, answers
        ``__wrapped__`` with what THAT wrapper wraps. Beside 0.2.2 it was "the function it
        runs cannot be inspected (ProxiedCallable)". It runs correctly; the reason says it
        may, and that ``functools.update_wrapper(self, fn)`` gets it read: with that the
        same tool is gated and runs."""
        import functools

        from recusal import invocation

        ran: list = []

        def tool(cls):
            return self.bare(cls(self.bare_wraps(self.fetch(ran))))

        func = tool(self.Forwards)
        assert func("q") == "recorded" and ran.pop() == ("fetch", "q", 10)  # it runs correctly
        with pytest.raises(invocation.ProxiedCallable) as raised:
            invocation.reached_parameters(func)
        assert type(raised.value) is invocation.ProxiedCallable
        found = str(raised.value)
        assert found == invocation._ANSWERS_FOR_WRAPPED
        assert found.endswith("register a plain function that calls it")
        self.refused_when_gated(example, func, self.PROXIED.format(found), ran)
        self.refused_when_called(
            example, self.wrapping(tool(self.Forwards)), self.PROXIED.format(found), ran
        )
        ran.clear()

        class Updated(self.Forwards):
            def __init__(self, fn):
                functools.update_wrapper(self, fn)
                self.fn = fn

        gated = example.gate_tools(
            [example.function_tool(tool(Updated), name_override="handle", strict_mode=False)],
            self.Recording(lambda name, args: []),
        )
        outputs, _ = example.run_calls(
            [("handle", {"sql": SELECT}, "c1")], gated=False, tools=gated
        )
        assert outputs == {"c1": "recorded"} and ran == [("fetch", SELECT, 10)]

    @staticmethod
    def bare_wraps(inner):
        import functools

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            return inner(*args, **kwargs)

        return wrapper

    def test_an_object_that_shows_another_class_has_the_same_reason(self, example):
        """``ProxiedCallable`` for its other shape: an object whose ``__class__`` is not its
        type and that holds no ``__wrapped__`` of its own. The kernel's text says which
        shape was found, and the gate's words hold for both."""
        import types

        from recusal import invocation

        ran: list = []
        fetch = self.fetch(ran)

        class Shows:
            """Shows the class of a function, and a function's face: that of ``fetch``."""

            def __init__(self, fn):
                self.fn = fn
                for name in ("__code__", "__defaults__", "__kwdefaults__", "__annotations__"):
                    setattr(self, name, getattr(fn, name))
                self.__name__ = fn.__name__

            @property
            def __class__(self):
                return types.FunctionType

            def __call__(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        shows = Shows(fetch)
        assert isinstance(shows, types.FunctionType) and type(shows) is Shows
        func = self.bare(shows)
        with pytest.raises(invocation.ProxiedCallable) as raised:
            invocation.reached_parameters(func)
        found = str(raised.value)
        assert found == invocation._SHOWS_ANOTHER_CLASS
        assert found.endswith("register a plain function that calls it")
        self.refused_when_called(example, self.wrapping(func), self.PROXIED.format(found), ran)

    # --- PartialMethodSignature -----------------------------------------------------------------

    def test_a_partialmethod_behind_a_cache_named_again_is_found_by_the_gates_own_walk(
        self, example
    ):
        """``PartialMethodSignature``: the method of a ``functools.partialmethod`` behind a
        cache that was named again for another function. The walk of 0.2.2 reached no
        partialmethod there and the kernel's walk refused it; 0.2.3 follows what the cache
        calls and finds it itself. The kernel's walk still refuses it, and the reason for
        that is the gate's own too."""
        from recusal import invocation

        adapter = _adapter()
        ran: list = []
        fetch = self.fetch(ran)

        def tool():
            return self.named_again(self.bare(self.partialmethod(ran)), fetch)

        reached, complete = adapter._reached_callables(tool())
        assert complete
        assert any(adapter._held_partialmethod(link) is not None for link in reached)
        with pytest.raises(invocation.PartialMethodSignature) as raised:
            invocation.reached_parameters(tool())
        assert adapter._unread_reason(raised.value, reached) == self.PARTIALMETHOD
        self.refused_when_gated(example, tool(), self.PARTIALMETHOD, ran)
        self.refused_when_called(example, self.wrapping(tool()), self.PARTIALMETHOD, ran)

    # --- known by class, not by name ------------------------------------------------------------

    def test_the_refusals_are_the_kernels_own_classes(self, example, monkeypatch):
        """The gate imports the kernel's classes (``recusal`` 0.17.0 is the floor) and asks
        ``isinstance``. An exception that only has the NAME of one, raised by other code
        on the way, is not taken for it: its text is not the kernel's, so it is not passed
        on, and the refusal names its class alone."""
        from recusal import invocation

        adapter = _adapter()
        for name in (
            "BoundArgumentKept",
            "DeclaredSignature",
            "DispatchedAtCallTime",
            "HiddenWrapper",
            "OpenDispatch",
            "PartialMethodSignature",
            "ProxiedCallable",
            "UnreadCache",
            "WeakProxy",
            "cache_target",
        ):
            assert getattr(adapter, f"_{name}") is getattr(invocation, name), name
        for name in ("_KERNEL_DISPATCH", "_KERNEL_HIDDEN", "_KERNEL_PARTIALMETHOD"):
            assert not hasattr(adapter, name), name

        tool = example.function_tool(TestFifthReview.plain_function())
        for name, base in (
            ("BoundArgumentKept", ValueError),
            ("DeclaredSignature", Exception),
            ("DispatchedAtCallTime", ValueError),
            ("HiddenWrapper", ValueError),
            ("OpenDispatch", ValueError),
            ("PartialMethodSignature", TypeError),
            ("ProxiedCallable", TypeError),
            ("UnreadCache", ValueError),
            ("WeakProxy", TypeError),
        ):
            lookalike = type(name, (base,), {})

            def walk(func, lookalike=lookalike):
                raise lookalike(f"the value was {self.SECRET}")

            monkeypatch.setattr(adapter, "_reached_parameters", walk)
            with pytest.raises(TypeError) as refusal:
                example.gate_tools([tool], example.sql_scope_policy)
            assert str(refusal.value) == (
                f"Recusal cannot gate 'run_sql': the function it runs cannot be inspected ({name})"
            )

        class Narrower(invocation.HiddenWrapper):
            """A refusal a later kernel may add as a ``HiddenWrapper``, as ``UnreadCache`` is."""

        def narrower(func):
            raise Narrower("`force` is a parameter the tool does not declare")

        monkeypatch.setattr(adapter, "_reached_parameters", narrower)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], example.sql_scope_policy)
        assert str(refusal.value) == "Recusal cannot gate 'run_sql': " + self.HIDDEN.format(
            "`force` is a parameter the tool does not declare"
        )

    def test_the_package_requires_the_kernel_that_has_the_classes(self):
        adapter_root = ROOT / "adapters" / "openai-agents"
        pyproject = (adapter_root / "pyproject.toml").read_text(encoding="utf-8")
        readme = (adapter_root / "README.md").read_text(encoding="utf-8")
        assert '"recusal>=0.17.0,<1"' in pyproject and "`recusal>=0.17.0`" in readme
        for older in ("recusal>=0.15", "recusal>=0.16"):
            assert older not in pyproject and older not in readme, older

    # --- what each reason says holds for every tool it is given for -----------------------------

    def test_a_declared_signature_is_refused_where_it_matches_and_where_it_is_none(self, example):
        """The kernel refuses a declared signature without asking what it says: one that
        equals the function's real parameters, and a ``__signature__`` of None, are
        refused like one that leaves a parameter out. So the reason does not say that the
        declaration is "not its own" or that ``inspect.signature`` reads it; it says the
        walk relies on none, and that a tool refused for this may run correctly."""
        import inspect

        ran: list = []
        matching = self.fetch(ran)
        matching.__signature__ = inspect.signature(matching)
        nothing = self.fetch(ran)
        nothing.__signature__ = None
        assert list(inspect.signature(nothing).parameters) == ["sql", "limit"]
        for func in (matching, nothing):
            assert func(SELECT) == "recorded" and ran.pop() == ("fetch", SELECT, 10)
            self.refused_when_gated(example, func, self.DECLARED, ran)
        assert "not its own" not in self.DECLARED
        assert "may run correctly" in self.DECLARED and "matches the real" in self.DECLARED

    def test_a_bound_argument_reason_holds_where_one_parameter_is_named_or_none_defaults(
        self, example
    ):
        """Two more places the kernel raises ``BoundArgumentKept``. A facade that takes its
        object in ``*args``: the kernel's text then names one parameter, not two. And one
        whose open parameter has no default. The gate's own words say neither "two
        parameters" nor "default", so they hold for both; the kernel's text is passed on
        as what the walk reports."""
        import functools

        from recusal import invocation

        ran: list = []

        def find(tenant, limit: int = 10) -> str:
            """Record a search without contacting anything."""
            ran.append(("find", tenant, limit))
            return "recorded"

        @functools.wraps(find)
        def capped(limit=100, *args, **kwargs):
            return find(*args, **kwargs)

        class InArgs:
            @functools.wraps(capped)
            def search(*args, **kwargs):
                return capped(*args[1:], tenant="acme", **kwargs)

        def find_all(tenant, query: str, limit: int = 10) -> str:
            """Record a search without contacting anything."""
            ran.append(("find_all", tenant, query, limit))
            return "recorded"

        @functools.wraps(find_all)
        def scoped(query, *args, **kwargs):
            return find_all("acme", query, *args, **kwargs)

        class Required:
            @functools.wraps(scoped)
            def search(self, *args, **kwargs):
                return scoped(*args, **kwargs)

        with pytest.raises(invocation.BoundArgumentKept) as raised:
            invocation.reached_parameters(InArgs().search)
        found = str(raised.value)
        assert "would fill `limit`, by position" in found and found.count("`limit`") == 2
        assert "is taken by a wrapper as" not in found  # no second parameter is named
        self.refused_when_gated(example, InArgs().search, self.BOUND.format(found), ran)

        assert Required().search("q") == "recorded"
        assert ran.pop() == ("find_all", "acme", "q", 10)  # it runs correctly
        with pytest.raises(invocation.BoundArgumentKept) as raised:
            invocation.reached_parameters(Required().search)
        found = str(raised.value)
        # ``recusal`` 0.16.0 said "`query`, the last they would fill, runs at a default";
        # the floor is 0.17.0, whose text says what holds
        assert "`query` is left to a keyword of that name" in found
        assert "default" not in found  # ``query`` has none, and the text says none
        self.refused_when_gated(example, Required().search, self.BOUND.format(found), ran)

        framing = self.BOUND.format("")
        assert "default" not in framing and "two parameters" not in framing
        assert "may run correctly" in framing

    def test_every_reason_says_how_to_remove_the_cause_and_that_a_body_is_not_read(self, example):
        """0.2.3, after its review. Up to 0.2.2 most reasons advised calling the refused
        function from a plain function and registering that, which the gate gates and
        whose body it never reads, so the cause ran as before. Every reason now advises
        removing the cause, and ends by saying why the other way does not. None of the
        gate's own sentences advises calling the refused function from a plain function;
        a reason that quotes the kernel's text, whose advice can, says which plain
        function is meant. ``TestReasonAdvice`` follows each piece of advice."""
        adapter = _adapter()
        reasons = {
            name: value
            for name, value in vars(adapter).items()
            if name.isupper()
            and name not in ("_BODY_NOT_READ", "_KERNEL_ADVICE")
            and type(value) is str
            and BODY_NOT_READ in value
        }
        assert sorted(reasons) == [
            "_BOUND_ARGUMENT_KEPT",
            "_DECLARED_SIGNATURE",
            "_DISPATCH_NOT_READ",
            "_HIDDEN_WRAPPER",
            "_KEPT_RECEIVER",
            "_OPAQUE_DECORATOR",
            "_OPEN_DISPATCH",
            "_PARTIALMETHOD_REFUSED",
            "_PROXIED_CALLABLE",
            "_SINGLEDISPATCH",
            "_TOO_MANY_WRAPPERS",
            "_UNREAD_CACHE",
            "_WEAK_PROXY",
        ]
        assert adapter._BODY_NOT_READ == BODY_NOT_READ
        assert adapter._KERNEL_ADVICE == KERNEL_ADVICE
        quoting = {"_BOUND_ARGUMENT_KEPT", "_HIDDEN_WRAPPER", "_PROXIED_CALLABLE"}
        for name, value in reasons.items():
            assert (KERNEL_ADVICE in value) == (name in quoting), name
            if name in quoting:
                # the gate's note comes before the kernel's text, whose advice is last
                assert value.endswith(KERNEL_ADVICE + " The kernel's walk reports: {}."), name
            else:
                assert value.endswith(BODY_NOT_READ), name
            for old in (
                "from a plain function and register",
                "a plain function that calls it",
                "Register a plain function that names",
                "what its body calls is not read",
            ):
                # the note names that advice in order to say not to follow it
                assert old not in value.replace(KERNEL_ADVICE, ""), (name, old)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestStatedLimits:
    """What the package page states as closed, as open or as refused although correct,
    each as a tool that is gated and run, or refused. Found by the review of 0.2.2 before
    its release. These tests hold the page to what happens: one that fails because a
    limit was closed or opened means the page has to change with it.

    0.2.2 stated four items as open beside ``recusal`` 0.16.0. The first three are closed
    in 0.2.3, which requires ``recusal`` 0.17.0 (no lane runs these tests beside an
    older kernel): the gate's own walk follows what a cache named again calls, so it
    asks its own questions there and refuses a ``weakref.proxy`` and a single-dispatch
    function with nothing registered behind such a cache, and the kernel's walk of 0.17.0
    refuses a held wrapper in another order and a held wrapper on a way 0.16.0 did not
    follow. Each test named ``test_closed_`` holds one of those, with the tool that ran a
    value no policy judged beside 0.16.0, now refused before any policy is asked and
    before anything runs. Two more found by the review of 0.2.3 are closed the same way.

    What is still open is held as it happens, gated and run (the tests named
    ``test_open_``): the fourth item, narrower; the held wrappers the kernel's walk does
    not find, in eight of the ways the page lists (a wrapper in a C structure, in a
    dictionary behind a ``__dict__`` property or in a class attribute is stated and not
    held here); the held wrapper whose leading place the kernel takes for the object's,
    stated from 0.2.3; and a method decorator that forgets to pass the object on, found by
    the confirmation review of 0.2.3. Of the page's other open items, a change a tool body
    makes in the same turn is held by ``TestSixthReview``; a dispatcher a tool body
    calls, and the cost of listing a cache, are stated and not tested."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording

    #: The reasons the gate gives a weak proxy and a single-dispatch function it finds.
    WEAK_PROXY = TestKernelRefusals.WEAK_PROXY
    DISPATCH = TestKernelRefusals.DISPATCH

    def refused(self, example, func, reason, events):
        """``func`` is refused when it is gated, with ``reason``; no policy is asked and
        nothing runs."""
        tool = example.function_tool(func, name_override="run_sql", strict_mode=False)
        policy = self.Recording(lambda name, args: [])
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], policy)
        assert str(refusal.value) == f"Recusal cannot gate 'run_sql': {reason}"
        assert policy.seen == [] and events == []

    @staticmethod
    def named(events):
        def named(sql: str, force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            events.append(("named", sql, force))
            return "named"

        return named

    @staticmethod
    def fetch(events):
        def fetch(sql: str, force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            events.append(("fetch", sql, force))
            return "recorded"

        return fetch

    @staticmethod
    def reviewed(events, inner):
        import functools

        @functools.wraps(inner)
        def reviewed(*args, skip_review=True, **kwargs):
            events.append(("reviewed", skip_review))
            return inner(*args, **kwargs)

        return reviewed

    @staticmethod
    def again(called, named):
        import functools

        return functools.update_wrapper(functools.lru_cache(maxsize=None)(called), named)

    def run(self, example, func, arguments, turns=1):
        """Gate ``func`` as the one tool of an agent and call it once in each of
        ``turns`` model turns; the outputs by call id and what the policy saw."""
        policy = self.Recording(lambda name, args: [])
        tool = example.function_tool(func, name_override="run_sql", strict_mode=False)
        model = example.ScriptedModel(
            [
                *(
                    [example.function_call("run_sql", arguments, call_id=f"c{turn + 1}")]
                    for turn in range(turns)
                ),
                [example.assistant_message("done")],
            ]
        )
        agent = example.gated_agent(example.Agent(name="a", model=model, tools=[tool]), policy)
        asyncio.run(example.run_agent(agent))
        return example.tool_outputs(model), policy.seen

    def reached_by_the_gates_walk(self, tool, question):
        """Whether the gate's own walk reaches a callable ``question`` says yes to."""
        reached, complete = _adapter()._reached_callables(tool)
        assert complete
        return any(question(link) for link in reached)

    @staticmethod
    def a_proxy(link):
        import weakref

        return type(link) in weakref.ProxyTypes

    # --- closed: open beside 0.2.2 and recusal 0.16.0, refused by 0.2.3 -------------------------

    def test_closed_a_weak_proxy_behind_a_cache_named_again_over_a_wrappers_parameter(
        self, example
    ):
        """No single-dispatch function anywhere. A cache named again for another function
        calls a ``weakref.proxy`` of a second such cache over a wrapper with a parameter
        of its own. Beside 0.16.0 the tool was gated and ``skip_review`` ran at its
        default: the gate's own walk followed what the caches name, and the kernel's walk
        did not refuse a proxy. The gate's own walk now follows what the outer cache
        calls, meets the proxy and refuses it."""
        import weakref

        events: list = []
        inner = self.again(self.reviewed(events, self.fetch(events)), self.named(events))
        tool = self.again(weakref.proxy(inner), self.named(events))
        assert self.reached_by_the_gates_walk(tool, self.a_proxy)
        self.refused(example, tool, self.WEAK_PROXY, events)

    def test_closed_a_weak_proxy_behind_a_cache_named_again_over_a_registered_dispatcher(
        self, example
    ):
        """An implementation registered BEFORE the tool is gated, on a dispatcher behind a
        ``weakref.proxy`` behind a cache named again. Beside 0.16.0 it was gated, and the
        implementation ran on every turn with a parameter of its own that no policy
        judged; the call was never refused. Refused now, as a weak proxy."""
        import functools
        import weakref

        events: list = []

        @functools.singledispatch
        def export(sql, force: bool = False) -> str:
            """Record an export; contacts nothing."""
            events.append(("base", sql, force))
            return "base"

        @export.register(str)
        def _(sql, force=False, skip_review=True):
            events.append(("registered", sql, force, skip_review))
            return "registered"

        tool = self.again(functools.partial(weakref.proxy(export)), self.named(events))
        assert self.reached_by_the_gates_walk(tool, self.a_proxy)
        self.refused(example, tool, self.WEAK_PROXY, events)

    def test_closed_a_dispatcher_behind_a_cache_named_again_registered_by_the_same_call(
        self, example
    ):
        """A single-dispatch function with nothing registered, behind a cache named again
        (here for the dispatcher's own base function), reached by a wrapper that carries
        none of the dispatcher's attributes. The wrapper registers an implementation on
        its first use. Beside 0.16.0 the very call that was judged ran it, with the
        parameter the implementation adds at its default. Refused now when the tool is
        gated, before anything is registered: the gate's own walk follows what the cache
        calls and finds the dispatcher."""
        import functools

        events: list = []

        def base(sql: str, force: bool = False) -> str:
            """Record an export; contacts nothing."""
            events.append(("base", sql, force))
            return "base"

        export = functools.singledispatch(base)

        def load():
            if export.dispatch(str) is base:

                @export.register(str)
                def _(sql, force=False, skip_review=True):
                    events.append(("registered", sql, force, skip_review))
                    return "registered"

        @functools.wraps(export, updated=())
        def lazy(*args, **kwargs):
            load()
            return export(*args, **kwargs)

        tool = self.again(lazy, base)
        assert self.reached_by_the_gates_walk(tool, _adapter()._is_dispatcher)
        self.refused(example, tool, self.DISPATCH, events)
        assert export.dispatch(str) is base  # nothing was registered

    @pytest.mark.parametrize("how", ["partial", "wraps-updated-nothing", "object-wrapped-only"])
    def test_closed_wrappers_that_carry_none_of_a_dispatchers_attributes(self, example, how):
        """Three ordinary wrappers through which a dispatcher with nothing registered,
        behind a cache named again, was gated beside 0.16.0. None of them carries the
        dispatcher's attributes, and the gate's own walk reaches the dispatcher through
        what the cache calls."""
        import functools

        events: list = []
        export = functools.singledispatch(self.fetch(events))

        class WrappedOnly:
            def __init__(self, func):
                self.__wrapped__ = func

            def __call__(self, *args, **kwargs):
                return self.__wrapped__(*args, **kwargs)

        def forwards(*args, **kwargs):
            return export(*args, **kwargs)

        wrapper = {
            "partial": functools.partial(export),
            "wraps-updated-nothing": functools.wraps(export, updated=())(forwards),
            "object-wrapped-only": WrappedOnly(export),
        }[how]
        tool = self.again(wrapper, self.named(events))
        assert self.reached_by_the_gates_walk(tool, _adapter()._is_dispatcher)
        self.refused(example, tool, self.DISPATCH, events)

    def test_closed_a_held_wrapper_that_takes_the_tools_parameters_in_another_order(self, example):
        """A decorator holds a ``functools.wraps`` wrapper that takes the tool's own two
        parameters the other way round. Beside 0.16.0 it named no parameter the tool does
        not declare, so the kernel did not refuse it held, and the function ran with the
        two values swapped. The kernel's walk of 0.17.0 refuses it, naming the two
        parameters. The same wrapper on the ``__wrapped__`` chain is refused by the
        gate's own rule, as before."""
        import functools

        from recusal import invocation

        events: list = []
        fetch = self.fetch(events)

        @functools.wraps(fetch)
        def flipped(force, sql):
            return fetch(sql=sql, force=force)

        @functools.wraps(fetch)
        def outer(*args, **kwargs):
            return flipped(*args, **kwargs)

        cause = invocation._HELD_IN_ANOTHER_PLACE.format(name="force", declared="sql")
        assert "`force`" in cause and "`sql`" in cause
        self.refused(example, outer, TestKernelRefusals.HIDDEN.format(cause), events)
        tool = example.function_tool(flipped, name_override="run_sql", strict_mode=False)
        with pytest.raises(TypeError, match="takes `force` in a way the tool does not declare"):
            example.gate_tools([tool], example.sql_scope_policy)

    def test_closed_a_held_wrapper_whose_way_passes_a_class_based_decorator(self, example):
        """A decorator holds a ``functools.wraps`` wrapper with a parameter of its own, and
        that wrapper wraps a class-based decorator of the function. Beside 0.16.0 the way
        from the held wrapper to the function passed an object the kernel did not read on
        a held way, so the tool was gated and ``skip_review`` ran at its default. The
        kernel's walk of 0.17.0 follows it and names the parameter."""
        import functools

        events: list = []
        fetch = self.fetch(events)

        class Timed:
            def __init__(self, func):
                functools.update_wrapper(self, func)

            def __call__(self, *args, **kwargs):
                return self.__wrapped__(*args, **kwargs)

        reviewed = self.reviewed(events, Timed(fetch))

        @functools.wraps(fetch)
        def outer(*args, **kwargs):
            return reviewed(*args, **kwargs)

        cause = (
            "`skip_review` is a parameter of a function the call reaches that the tool "
            "does not declare, so it would run with a value no policy judged"
        )
        self.refused(example, outer, TestKernelRefusals.HIDDEN.format(cause), events)

    # --- refused although it runs correctly -----------------------------------------------------

    def refused_although_correct(self, example, make, reason):
        events: list = []
        assert make(events)("x", force=True) == "recorded"
        assert events == [("fetch", "x", True)]  # nothing but the function, as called
        tool = example.function_tool(make([]), name_override="run_sql", strict_mode=False)
        policy = self.Recording(lambda name, args: [])
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], policy)
        assert str(refusal.value) == f"Recusal cannot gate 'run_sql': {reason}"
        assert policy.seen == []

    def test_refused_although_correct_a_held_cache_named_again_over_a_partial(self, example):
        """A decorator holds a cache of ``functools.partial(f)``, named again for ``f``.
        Nothing has a parameter of its own and nothing dispatches. The kernel refuses a
        held cache named again that calls anything but a plain function, as a
        ``HiddenWrapper`` with no parameter named."""
        import functools

        from recusal import invocation

        def make(events):
            fetch = self.fetch(events)
            held = self.again(functools.partial(fetch), fetch)

            @functools.wraps(fetch)
            def outer(*args, **kwargs):
                return held(*args, **kwargs)

            return outer

        self.refused_although_correct(
            example, make, TestKernelRefusals.HIDDEN.format(invocation._HIDDEN_WRAPPER)
        )
        assert "parameter" not in invocation._HIDDEN_WRAPPER

    @pytest.mark.parametrize("under", ["a-cache", "a-staticmethod"])
    def test_refused_although_correct_a_partial_under_a_cache_named_again(self, example, under):
        """A cache named again calls a ``functools.partial`` of a second
        ``functools.lru_cache`` wrapper, or of a ``staticmethod`` object. The kernel's walk
        reads what the cache calls, and ``inspect.signature`` shows no signature for that
        partial. Under a second cache the tool is refused with the name of the exception
        class alone, which is the item still open. Under a ``staticmethod`` object the
        gate's own walk now reaches that object through what the cache calls, and the
        reason says what it is (beside 0.2.2 it named the exception class alone)."""
        import functools

        def make(events):
            fetch = self.fetch(events)
            inner = (
                functools.lru_cache(maxsize=None)(fetch)
                if under == "a-cache"
                else staticmethod(fetch)
            )
            return self.again(functools.partial(inner), self.named(events))

        reason = (
            "the function it runs cannot be inspected (ValueError)"
            if under == "a-cache"
            else self.OPAQUE
        )
        self.refused_although_correct(example, make, reason)

    @pytest.mark.parametrize(
        "shape",
        [
            "proxy-of-a-plain-function-behind-a-cache-named-again",
            "open-dispatcher-behind-a-cache-named-again",
            "held-wrapper-in-another-order-called-by-keyword",
            "held-weak-proxy-of-a-helper",
            "held-cache-named-again-for-another-function-over-a-partial",
            "dispatcher-lookalike-behind-a-cache-named-again",
        ],
    )
    def test_refused_although_correct_beside_the_kernel_0_17_0(self, example, shape):
        """Tools that run correctly, that 0.2.2 gated beside ``recusal`` 0.16.0, and that
        0.2.3 refuses: the first two by the gate's own walk, which follows what a cache
        named again calls, the next three by the kernel's walk of 0.17.0. The package
        page lists them, each with what to write instead. The kernel's changelog entry
        for 0.17.0 has the whole list for its walk.

        The last is the one verdict 0.2.3 changes beside 0.17.0 too: an object of your
        own with the attributes the gate takes for a dispatcher's, which the gate refuses
        wherever its own walk reaches it, now behind a cache named again as well."""
        import functools
        import types
        import weakref

        from recusal import invocation

        keep: list = []

        def make(events):
            fetch = self.fetch(events)
            if shape == "dispatcher-lookalike-behind-a-cache-named-again":

                class Service:
                    registry = types.MappingProxyType({})

                    def dispatch(self, cls):
                        return fetch

                    def register(self, cls, func=None):
                        return func

                    def __call__(self, *args, **kwargs):
                        return fetch(*args, **kwargs)

                return self.again(Service(), self.named(events))
            if shape == "proxy-of-a-plain-function-behind-a-cache-named-again":
                keep.append(fetch)
                return self.again(weakref.proxy(fetch), self.named(events))
            if shape == "open-dispatcher-behind-a-cache-named-again":
                export = functools.singledispatch(fetch)
                return self.again(functools.partial(export), self.named(events))
            if shape == "held-wrapper-in-another-order-called-by-keyword":

                @functools.wraps(fetch)
                def second(force, sql):
                    return fetch(sql, force)

                @functools.wraps(fetch)
                def outer(*args, **kwargs):
                    # the values are put back by name: it runs correctly
                    values = dict(zip(("sql", "force"), args), **kwargs)
                    return second(force=values.get("force", False), sql=values["sql"])

                return outer
            if shape == "held-weak-proxy-of-a-helper":

                def audit(*args, **kwargs):
                    return None

                keep.append(audit)
                audited = weakref.proxy(audit)

                @functools.wraps(fetch)
                def outer(*args, **kwargs):
                    audited(*args, **kwargs)
                    return fetch(*args, **kwargs)

                return outer
            held = self.again(functools.partial(fetch), self.named(events))

            @functools.wraps(fetch)
            def outer(*args, **kwargs):
                return held(*args, **kwargs)

            return outer

        hidden = TestKernelRefusals.HIDDEN.format
        reason = {
            "proxy-of-a-plain-function-behind-a-cache-named-again": self.WEAK_PROXY,
            "open-dispatcher-behind-a-cache-named-again": self.DISPATCH,
            "held-wrapper-in-another-order-called-by-keyword": hidden(
                invocation._HELD_IN_ANOTHER_PLACE.format(name="force", declared="sql")
            ),
            "held-weak-proxy-of-a-helper": hidden(invocation._HELD_WEAK_PROXY),
            "held-cache-named-again-for-another-function-over-a-partial": hidden(
                invocation._HIDDEN_WRAPPER
            ),
            "dispatcher-lookalike-behind-a-cache-named-again": self.DISPATCH,
        }[shape]
        self.refused_although_correct(example, make, reason)

    OPAQUE = OPAQUE_REASON

    def test_refused_although_correct_a_staticmethod_object_under_a_partial(self, example):
        """A ``functools.partial`` of a ``staticmethod`` object, under a decorator written
        with ``functools.update_wrapper`` over a partial. Python shows no signature for
        the ``staticmethod`` object, which is an object whose class defines ``__get__``:
        the reason says so, and no longer calls it a class-based decorator."""
        import functools

        def helper(func, *args, **kwargs):
            return func(*args, **kwargs)

        def make(events):
            inner = functools.partial(staticmethod(self.fetch(events)))
            decorated = functools.update_wrapper(functools.partial(helper, inner), inner)

            @functools.wraps(decorated)
            def forwarding(*args, **kwargs):
                return decorated(*args, **kwargs)

            return forwarding

        reached, _ = _adapter()._reached_callables(make([]))
        assert any(type(link) is staticmethod for link in reached)
        self.refused_although_correct(example, make, self.OPAQUE)

    @pytest.mark.parametrize(
        "how",
        [
            "the-method",
            "proxy-of-the-function",
            "partial-of-a-proxy",
            "proxy-of-a-function-naming-it",
            "proxy-of-a-cache-named-again",
        ],
    )
    def test_a_partialmethod_behind_a_cache_named_again_is_refused(self, example, how):
        """A partialmethod behind a cache named again, refused in each shape. Beside 0.2.2
        all five were refused with the reason for a partialmethod: in four the kernel's
        walk found the mark through what the cache calls, and in the fifth the gate's
        own walk found it on the second cache, onto which ``functools.lru_cache`` copied
        it. The gate's own walk now follows what the outer cache calls, so where a
        ``weakref.proxy`` stands there it meets the proxy first and gives the reason for
        one. The next test holds the shapes that 0.2.2 gated."""
        import functools
        import weakref

        events: list = []

        def run_on(conn, sql: str = "", force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            events.append(("run_on", sql, force))
            return "recorded"

        class Logged:
            def __init__(self, inner):
                functools.update_wrapper(self, inner)
                self.inner = inner

            def __call__(self, *args, **kwargs):
                return self.inner(*args, **kwargs)

        class Db:
            over_object = functools.partialmethod(Logged(self.reviewed(events, run_on)))
            over_function = functools.partialmethod(self.reviewed(events, run_on))

        db = Db()
        method, from_class = db.over_object, Db.over_function

        def naming(*args, **kwargs):
            return method(*args, **kwargs)

        naming.__wrapped__ = method  # type: ignore[attr-defined]
        inner = self.again(method, self.named(events))
        called = {
            "the-method": method,
            "proxy-of-the-function": functools.partial(weakref.proxy(from_class), db),
            "partial-of-a-proxy": functools.partial(
                functools.partial(weakref.proxy(from_class)), db
            ),
            "proxy-of-a-function-naming-it": weakref.proxy(naming),
            "proxy-of-a-cache-named-again": weakref.proxy(inner),
        }[how]
        tool = example.function_tool(
            self.again(called, self.named(events)), name_override="run_sql", strict_mode=False
        )
        policy = self.Recording(lambda name, args: [])
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], policy)
        expected = TestKernelRefusals.PARTIALMETHOD if how == "the-method" else self.WEAK_PROXY
        assert str(refusal.value) == f"Recusal cannot gate 'run_sql': {expected}"
        assert policy.seen == [] and events == []

    @pytest.mark.parametrize("between", ["a-partial", "a-wrapper-naming-it"])
    def test_closed_a_partialmethod_behind_a_proxy_of_a_cache_named_again(self, example, between):
        """Part of what stood behind a ``weakref.proxy`` behind a cache named again: a
        cache named again calls a proxy of a second cache named again, and that one calls
        the partialmethod's method through a ``functools.partial`` or through a wrapper
        that names it as ``__wrapped__``. Beside 0.2.2 and ``recusal`` 0.16.0 nothing
        copied the mark onto a cache, the kernel's walk read nothing behind the proxy,
        and the tool was gated: the callable the partialmethod holds ran with a parameter
        of its own at its default. Refused now, as a weak proxy, by the gate's own walk."""
        import functools
        import weakref

        events: list = []

        class Reviewed:
            """A callable object with no ``__get__``: a partialmethod over it hands out a
            method of a function ``functools`` builds, which carries the mark."""

            def __call__(self, owner, sql="", force=False, skip_review=True):
                events.append(("reviewed", sql, force, skip_review))
                return "recorded"

        class Tools:
            run = functools.partialmethod(Reviewed())

        method = Tools().run
        assert _adapter()._held_partialmethod(method.__func__) is not None

        def naming(*args, **kwargs):
            return method(*args, **kwargs)

        naming.__wrapped__ = method  # type: ignore[attr-defined]
        inner = self.again(
            functools.partial(method) if between == "a-partial" else naming, self.named(events)
        )
        tool = self.again(weakref.proxy(inner), self.named(events))
        assert self.reached_by_the_gates_walk(tool, self.a_proxy)
        self.refused(example, tool, self.WEAK_PROXY, events)

    # --- closed by the gate itself after the review of 0.2.3 -----------------------------------

    @staticmethod
    def warehouse(events, how):
        """A module function exposed as a method by a decorator written with
        ``functools.wraps``, as ``Warehouse.run_sql``. ``inspect.signature`` reads the
        method as the function and leaves its first parameter, ``database``, out as the
        object's place. ``how`` is how the decorator takes the object it drops: as
        ``self`` ("drops"), in ``*args`` ("in-args"), under the function's own first name
        ("same-name"), or as "drops" under a ``functools.lru_cache`` ("cached")."""
        import functools

        def execute(database: str, sql: str = "SELECT 1", note: str = "") -> str:
            """Run SQL against a named database (here: only records it)."""
            events.append(("execute", database, sql, note))
            return "recorded"

        def drops(fn):
            @functools.wraps(fn)
            def method(self, *args, **kwargs):
                return fn(*args, **kwargs)

            return method

        def in_args(fn):
            @functools.wraps(fn)
            def method(*args, **kwargs):
                return fn(*args[1:], **kwargs)

            return method

        def same_name(fn):
            @functools.wraps(fn)
            def method(database, *args, **kwargs):
                return fn(*args, **kwargs)

            return method

        class Updated:
            def __init__(self, fn):
                functools.update_wrapper(self, fn)

            def __call__(self, *args, **kwargs):
                return self.__wrapped__(*args, **kwargs)

        decorate = {
            "drops": drops,
            "in-args": in_args,
            "same-name": same_name,
            "cached": lambda fn: functools.lru_cache(maxsize=None)(drops(fn)),
            # what the review's differential found once the first form was refused: the
            # function is a partial of a class-based decorator, and `inspect.signature`
            # reads through both to `execute`
            "over-a-partial": lambda fn: drops(functools.partial(Updated(fn))),
        }[how]

        class Warehouse:
            run_sql = decorate(execute)

        return execute, Warehouse

    @staticmethod
    def kept_or_bound(tool):
        """The reason for a facade that may drop what is bound, over ``execute``. The
        gate's own rule gives it beside ``recusal`` 0.17.0, whose walk reads the facade; a
        kernel whose walk refuses the facade itself as a ``BoundArgumentKept`` is asked
        first, and its reason is that one."""
        from recusal import invocation

        try:
            invocation.reached_parameters(tool)
        except invocation.BoundArgumentKept as raised:
            return TestKernelRefusals.BOUND.format(raised)
        return TestKernelRefusals.KEPT_RECEIVER.format(name="database")

    @pytest.mark.parametrize("how", ["drops", "in-args", "same-name", "cached", "over-a-partial"])
    def test_closed_a_method_facade_that_drops_its_object(self, example, how):
        """Found by the review of 0.2.3, in 0.2.2 too: the SDK passes every declared
        parameter by position, and a method whose function is a wrapper of a function not
        written to take the object ran with every value one place off. The policy judged
        ``sql="SELECT 1"`` and ``note="DELETE FROM audit_log"``; the function ran
        ``database="SELECT 1"``, ``sql="DELETE FROM audit_log"``. The kernel's walk of
        ``recusal`` 0.17.0 reads it, because the parameter the object would fill has no
        default; the gate refuses it itself, naming that parameter."""
        events: list = []
        _, Warehouse = self.warehouse(events, how)
        tool = Warehouse().run_sql
        assert "database" not in __import__("inspect").signature(tool).parameters
        tool("SELECT 1", "DELETE FROM audit_log")  # as the SDK calls it
        assert events.pop() == ("execute", "SELECT 1", "DELETE FROM audit_log", "")
        self.refused(example, tool, self.kept_or_bound(tool), events)

    def test_closed_a_partial_that_binds_an_argument_a_wrapper_drops(self, example):
        """The same shape with a ``functools.partial`` in place of the method: it binds a
        positional argument to a wrapper of ``execute``, which drops it. Gated by 0.2.2 and
        by 0.2.3 as first written, and run one place off."""
        import functools

        events: list = []
        execute, _ = self.warehouse(events, "drops")

        @functools.wraps(execute)
        def drops(first, *args, **kwargs):
            return execute(*args, **kwargs)

        bound = functools.partial(drops, "the-object")

        @functools.wraps(execute)
        def tool(*args, **kwargs):
            return bound(*args, **kwargs)

        tool.__wrapped__ = bound  # type: ignore[attr-defined]
        tool("SELECT 1", "DELETE FROM audit_log")  # as the SDK calls it
        assert events.pop() == ("execute", "SELECT 1", "DELETE FROM audit_log", "")
        self.refused(example, tool, self.kept_or_bound(tool), events)

    def test_refused_although_correct_two_dropping_methods_behind_a_cache_named_again(
        self, example
    ):
        """Found by the confirmation review of 0.2.3: two method facades that each drop
        their object, one over the other, behind a cache named again for a function that
        shows the whole signature. The SDK reads that signature, not the methods', and
        each drop is what makes the call correct, so it runs as judged. The gate's rule
        refuses it, naming ``force``; the reason says "where the SDK reads the tool's
        parameters through that method", which is not so here, and does not say it is."""
        import functools

        def facade(inner):
            @functools.wraps(inner)
            def first(self, *args, **kwargs):
                return inner(*args, **kwargs)

            class Holder:
                m = first

            return Holder().m

        def make(events):
            return self.again(facade(facade(self.fetch(events))), self.named(events))

        self.refused_although_correct(
            example, make, TestKernelRefusals.KEPT_RECEIVER.format(name="force")
        )

    def test_a_class_member_named_partialmethod_is_not_a_mark(self, example):
        """Found by the review of 0.2.3: the gate also asks for a partialmethod's mark under
        the name ``partialmethod``, which Python never reads, and found a class member of
        that name on a callable object behind a cache named again, so a tool that runs
        correctly was refused there, where 0.2.2 gated it. That name now counts only where
        the object itself holds it; the two names ``inspect`` reads count as before."""
        import functools

        events: list = []
        fetch = self.fetch(events)

        class Exporter:
            def export(self, sql, force=False):
                return fetch(sql, force)

            partialmethod = functools.partialmethod(export, force=False)

            def __call__(self, sql, force=False):
                return fetch(sql, force)

        exporter = Exporter()
        assert _adapter()._held_partialmethod(exporter) is None
        exporter.partialmethod = functools.partialmethod(fetch)  # the object's own
        assert _adapter()._held_partialmethod(exporter) is exporter.partialmethod
        del exporter.partialmethod
        outputs, seen = self.run(example, self.again(exporter, fetch), {"sql": "S"})
        assert outputs == {"c1": "recorded"} and events == [("fetch", "S", False)]

    def test_a_decorated_method_written_to_take_its_object_is_gated(self, example):
        """The same decorator over a method of an ordinary class, whose first parameter is
        ``self``: gated, and it runs as judged."""
        import functools

        events: list = []

        def logged(fn):
            @functools.wraps(fn)
            def method(self, *args, **kwargs):
                return fn(self, *args, **kwargs)

            return method

        class Warehouse:
            @logged
            def run_sql(self, sql: str = "SELECT 1", note: str = "") -> str:
                """Run SQL (here: only records it)."""
                events.append(("run_sql", sql, note))
                return "recorded"

        outputs, seen = self.run(example, Warehouse().run_sql, {"sql": "S", "note": "n"})
        assert outputs == {"c1": "recorded"}
        assert seen == [("run_sql", {"sql": "S", "note": "n"})]
        assert events == [("run_sql", "S", "n")]

    # --- open: what 0.2.2 stated and 0.2.3 does not close, and what 0.2.3 states --------------

    @pytest.mark.parametrize("receiver", ["self", "cls-of-a-classmethod"])
    def test_open_a_method_decorator_that_forgets_the_object(self, example, receiver):
        """Found by the confirmation review of 0.2.3, in 0.2.2 too, and stated as open. A
        method decorator written as ``def method(self, *a, **k): return m(*a, **k)`` over
        a method of an ordinary class, ``def run_sql(self, sql, force=False)``, forgets
        to pass the object on. From the signature the gate cannot tell it from one that
        passes it on, which is the ordinary decorated method; only the body differs. It
        is gated, and the SDK's call runs every judged value one place off: judged
        ``sql="S"``, ``force=True``; ran ``self="S"``, ``sql=True``, ``force=False``. The
        same call misruns without the gate: the mistake is in the decorator."""
        import functools

        events: list = []

        def forgets(m):
            @functools.wraps(m)
            def method(self, *args, **kwargs):
                return m(*args, **kwargs)

            return method

        if receiver == "self":

            class Tools:
                @forgets
                def run_sql(self, sql: str, force: bool = False) -> str:
                    """Record SQL text without connecting to a database."""
                    events.append(("run_sql", self, sql, force))
                    return "recorded"

            tool = Tools().run_sql
        else:

            class Tools:  # type: ignore[no-redef]
                @classmethod
                @forgets
                def run_sql(cls, sql: str, force: bool = False) -> str:
                    """Record SQL text without connecting to a database."""
                    events.append(("run_sql", cls, sql, force))
                    return "recorded"

            tool = Tools.run_sql
        outputs, seen = self.run(example, tool, {"sql": "S", "force": True})
        assert outputs == {"c1": "recorded"}
        assert seen == [("run_sql", {"sql": "S", "force": True})]
        assert events == [("run_sql", "S", True, False)]

    @staticmethod
    def not_found(events, where):
        """A decorator of ``fetch`` that calls a wrapper with a parameter of its own,
        ``skip_review=True``, which the kernel's walk of 0.17.0 does not find: ``where``
        says how the decorator reaches it."""
        import functools
        import types
        import weakref

        fetch = TestStatedLimits.fetch(events)

        def guarded(fn):
            @functools.wraps(fn)
            def guarded(*args, skip_review=True, **kwargs):
                events.append(("guarded", skip_review))
                return fn(*args, **kwargs)

            return guarded

        if where == "names-nothing":

            def inner(*args, skip_review=True, **kwargs):
                events.append(("guarded", skip_review))
                return fetch(*args, **kwargs)

            reach = lambda: inner  # noqa: E731
        elif where == "built-on-first-use":
            state: dict = {}

            def reach():
                if "held" not in state:
                    state["held"] = guarded(fetch)
                return state["held"]

        elif where == "container":
            held = [guarded(fetch)]
            reach = lambda: held[0]  # noqa: E731
        elif where == "settings-object":
            settings = types.SimpleNamespace(run=guarded(fetch))
            reach = lambda: settings.run  # noqa: E731
        elif where == "slot":

            class Slotted:
                __slots__ = ("held",)

            slotted = Slotted()
            slotted.held = guarded(fetch)
            reach = lambda: slotted.held  # noqa: E731
        elif where == "function-attribute":

            def holder():
                return None

            holder.held = guarded(fetch)  # type: ignore[attr-defined]
            reach = lambda: holder.held  # noqa: E731
        elif where == "weakref-ref":
            kept = guarded(fetch)
            TestStatedLimits._KEEP.append(kept)  # alive for as long as the tests run
            reference = weakref.ref(kept)
            reach = lambda: reference()  # noqa: E731
        else:
            assert where == "partial-argument", where

            def helper(func, *args, **kwargs):
                return func(*args, **kwargs)

            bound = functools.update_wrapper(functools.partial(helper, guarded(fetch)), fetch)
            reach = lambda: bound  # noqa: E731

        @functools.wraps(fetch)
        def outer(*args, **kwargs):
            return reach()(*args, **kwargs)

        if where == "partial-argument":
            outer.__wrapped__ = bound  # the partial is on the chain, and binds the wrapper
        return outer

    _KEEP: list = []

    @pytest.mark.parametrize(
        "where",
        [
            "names-nothing",
            "built-on-first-use",
            "container",
            "settings-object",
            "slot",
            "function-attribute",
            "weakref-ref",
            "partial-argument",
        ],
    )
    def test_open_a_held_wrapper_the_kernel_does_not_find(self, example, where):
        """Stated on the package page under what the gate does not see, and by the
        kernel's entry for 0.17.0 as not found by design: a held wrapper reached in one of
        these ways is gated, and its own parameter runs at its default."""
        events: list = []
        func = self.not_found(events, where)
        outputs, seen = self.run(example, func, {"sql": "x"})
        assert outputs == {"c1": "recorded"}
        assert seen == [("run_sql", {"sql": "x", "force": False})]
        assert events == [("guarded", True), ("fetch", "x", False)]

    def test_open_a_held_wrapper_whose_leading_place_is_taken_for_the_objects(self, example):
        """Stated from 0.2.3, in 0.2.2 too: the kernel's name rule for a bound place, which
        its entry for 0.17.0 states. A method decorator keeps the object and calls a held
        wrapper that has a required parameter of its own in the leading place. The kernel
        takes that place for the object's and finds nothing to refuse; the SDK hands the
        judged values on by position, so the function runs with them one place off."""
        import functools

        events: list = []

        def run(self, sql: str = "x", force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            events.append(("run", sql, force))
            return "recorded"

        @functools.wraps(run)
        def with_its_own(mode, sql="x", force=False):
            return run(None, sql, force)

        def keeping(method):
            @functools.wraps(method)
            def first(self, *args, **kwargs):
                return with_its_own(*args, **kwargs)

            return first

        class Tools:
            kept = keeping(run)

        outputs, seen = self.run(example, Tools().kept, {"sql": "S", "force": True})
        assert outputs == {"c1": "recorded"}
        assert seen == [("run_sql", {"sql": "S", "force": True})]
        assert events == [("run", True, False)]  # judged sql="S", force=True

    def test_open_what_the_walk_raises_for_a_callable_it_cannot_read_names_its_class(self, example):
        """The fourth item 0.2.2 stated as open, narrower now. The kernel's walk raises a
        plain ``ValueError`` or ``TypeError`` for a callable it cannot read, and the gate
        names the exception class alone, with no cause and no advice: here a cache named
        again over a ``functools.partial`` of a second cache, which runs correctly. The
        three classes ``recusal`` 0.17.0 adds each have a reason now
        (``TestKernelRefusals``)."""
        import functools

        events: list = []
        fetch = self.fetch(events)
        tool = self.again(
            functools.partial(functools.lru_cache(maxsize=None)(fetch)), self.named(events)
        )
        self.refused(example, tool, "the function it runs cannot be inspected (ValueError)", events)
        assert tool("x", force=True) == "recorded" and events == [("fetch", "x", True)]


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestReasonAdvice:
    """Found by the review of 0.2.3: following the advice of a reason literally gave, in 11
    of 15 cases, a gated tool that ran the cause. The advice was to call the refused
    function from a plain function and register that, and the gate reads a plain
    function's parameters and never its body. Each reason now advises removing the cause.

    For each reason: a tool refused with it, and the tool the reason's advice says to
    write, which is gated and runs as judged, through a real agent run. Where the cause
    lies under another (a wrapper with a parameter of its own under a proxy), following
    the advice for the first meets the second, and that is refused too: it is never
    gated with the cause inside it."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording
    ARGS = {"sql": "S", "force": True}

    @staticmethod
    def fetch(events):
        def fetch(sql: str, force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            events.append(("fetch", sql, force))
            return "recorded"

        return fetch

    @staticmethod
    def forwarding(inner):
        import functools

        @functools.wraps(inner)
        def forwarding(*args, **kwargs):
            return inner(*args, **kwargs)

        return forwarding

    @staticmethod
    def reviewed(events, inner):
        import functools

        @functools.wraps(inner)
        def reviewed(*args, skip_review=True, **kwargs):
            events.append(("UNJUDGED skip_review", skip_review))
            return inner(*args, **kwargs)

        return reviewed

    @staticmethod
    def again(called, named):
        import functools

        return functools.update_wrapper(functools.lru_cache(maxsize=None)(called), named)

    def refused(self, example, func, reason):
        tool = example.function_tool(func, name_override="run_sql", strict_mode=False)
        with pytest.raises(TypeError) as refusal:
            example.gate_tools([tool], self.Recording(lambda name, args: []))
        assert str(refusal.value) == f"Recusal cannot gate 'run_sql': {reason}"

    def runs_as_judged(self, example, func, events, ran, arguments=None):
        """``func`` is gated, and a call of it runs exactly ``ran`` and nothing else."""
        arguments = dict(self.ARGS) if arguments is None else arguments
        policy = self.Recording(lambda name, args: [])
        tool = example.function_tool(func, name_override="run_sql", strict_mode=False)
        (gated,) = example.gate_tools([tool], policy)
        del events[:]
        outputs, _ = example.run_calls([("run_sql", arguments, "c1")], gated=False, tools=[gated])
        assert outputs == {"c1": "recorded"}, outputs
        assert policy.seen == [("run_sql", arguments)]
        assert events == ran

    def test_a_held_wrapper_with_a_parameter_of_its_own(self, example):
        """``HiddenWrapper``: give the held wrapper no parameter of its own, passing the
        value explicitly in its body; or register the function itself."""
        import functools

        events: list = []
        fetch = self.fetch(events)
        held = self.reviewed(events, fetch)

        @functools.wraps(fetch)
        def holding(*args, **kwargs):
            return held(*args, **kwargs)

        found = (
            "`skip_review` is a parameter of a function the call reaches that the tool does "
            "not declare, so it would run with a value no policy judged"
        )
        self.refused(example, holding, HIDDEN_REASON.format(found))

        @functools.wraps(fetch)
        def explicit(*args, **kwargs):
            skip_review = True  # the value, written in the body
            events.append(("reviewed", skip_review))
            return fetch(*args, **kwargs)

        @functools.wraps(fetch)
        def holding_explicit(*args, **kwargs):
            return explicit(*args, **kwargs)

        self.runs_as_judged(
            example, holding_explicit, events, [("reviewed", True), ("fetch", "S", True)]
        )
        self.runs_as_judged(example, fetch, events, [("fetch", "S", True)])

    def test_a_held_wrapper_in_another_order(self, example):
        """``HiddenWrapper`` for the order: give the held wrapper the tool's parameters in
        the tool's order. The old advice, a plain function that calls the decorated one by
        position, ran ``fetch(sql=True, force='S')``."""
        import functools

        from recusal import invocation

        events: list = []
        fetch = self.fetch(events)

        @functools.wraps(fetch)
        def flipped(force, sql):
            return fetch(sql=sql, force=force)

        @functools.wraps(fetch)
        def holding(*args, **kwargs):
            return flipped(*args, **kwargs)

        cause = invocation._HELD_IN_ANOTHER_PLACE.format(name="force", declared="sql")
        self.refused(example, holding, HIDDEN_REASON.format(cause))

        @functools.wraps(fetch)
        def in_order(sql, force=False):
            return fetch(sql=sql, force=force)

        @functools.wraps(fetch)
        def holding_in_order(*args, **kwargs):
            return in_order(*args, **kwargs)

        @functools.wraps(fetch)
        def keyword_only(*, sql, force=False):
            return fetch(sql=sql, force=force)

        @functools.wraps(fetch)
        def holding_by_keyword(*args, **kwargs):
            return keyword_only(**dict(zip(("sql", "force"), args), **kwargs))

        for rewrite in (holding_in_order, holding_by_keyword):
            self.runs_as_judged(example, rewrite, events, [("fetch", "S", True)])

    def test_a_held_dispatcher(self, example):
        """``HiddenWrapper`` for a dispatcher a decorator holds: hold no single-dispatch
        function, or register the function itself. The old advice gated a plain function
        that called the decorated one, and an implementation registered afterwards ran."""
        import functools

        from recusal import invocation

        events: list = []
        fetch = self.fetch(events)
        export = functools.singledispatch(fetch)

        @functools.wraps(fetch)
        def holding(*args, **kwargs):
            return export(*args, **kwargs)

        self.refused(example, holding, HIDDEN_REASON.format(invocation._HIDDEN_WRAPPER))
        self.runs_as_judged(example, fetch, events, [("fetch", "S", True)])

    def test_a_weak_proxy(self, example):
        """``WeakProxy``: put the object the proxy stands for in its place. Over a wrapper
        that only forwards the result is gated and runs; over one with a parameter of its
        own it is refused for that, where the old advice gated it."""
        import weakref

        events: list = []
        fetch = self.fetch(events)
        inner = self.again(self.forwarding(fetch), fetch)
        tool = self.again(weakref.proxy(inner), fetch)
        self.refused(example, tool, WEAK_PROXY_REASON)
        self.runs_as_judged(example, self.again(inner, fetch), events, [("fetch", "S", True)])

        under = self.again(self.reviewed(events, fetch), fetch)
        self.refused(example, self.again(weakref.proxy(under), fetch), WEAK_PROXY_REASON)
        with pytest.raises(TypeError, match="`skip_review`"):
            example.gate_tools(
                [example.function_tool(self.again(under, fetch), strict_mode=False)],
                example.sql_scope_policy,
            )

    def test_a_decorator_that_answers_for_wrapped(self, example):
        """``ProxiedCallable``: call ``functools.update_wrapper(self, fn)`` in the
        decorator's ``__init__``."""
        import functools

        from recusal import invocation

        events: list = []
        fetch = self.fetch(events)
        Forwards = TestKernelRefusals.Forwards

        class Updated(Forwards):
            def __init__(self, fn):
                functools.update_wrapper(self, fn)
                self.fn = fn

        bare = TestKernelRefusals.bare
        self.refused(
            example,
            bare(Forwards(self.forwarding(fetch))),
            PROXIED_REASON.format(invocation._ANSWERS_FOR_WRAPPED),
        )
        self.runs_as_judged(
            example, bare(Updated(self.forwarding(fetch))), events, [("fetch", "S", True)]
        )

    def test_a_partialmethod(self, example):
        """``functools.partialmethod``: write the method out, with the bound values as its
        own defaults. The old advice, a plain function that calls the method, ran what
        stood under the callable the partialmethod holds."""
        import functools

        events: list = []

        class Recorder:
            def __call__(self, owner, sql="", force=False):
                events.append(("fetch", sql, force))
                return "recorded"

        class Db:
            export = functools.partialmethod(Recorder(), force=False)

        self.refused(example, Db().export, PARTIALMETHOD_REASON)

        class Written:
            def export(self, sql: str = "", force: bool = False) -> str:
                """Record SQL text without connecting to a database."""
                events.append(("fetch", sql, force))
                return "recorded"

        self.runs_as_judged(example, Written().export, events, [("fetch", "S", True)])

    def test_too_many_wrappers(self, example):
        """Take wrappers off until the gate reads them all, or register the function."""
        events: list = []
        fetch = self.fetch(events)
        tool = fetch
        for _ in range(40):
            tool = self.forwarding(tool)
        self.refused(example, tool, TOO_MANY_REASON)
        fewer = fetch
        for _ in range(5):
            fewer = self.forwarding(fewer)
        self.runs_as_judged(example, fewer, events, [("fetch", "S", True)])

    def test_a_decorator_whose_signature_cannot_be_read(self, example):
        """Register the function without that decorator, or use one the gate reads."""
        events: list = []
        fetch = self.fetch(events)
        self.refused(example, _DescriptorDecorator(fetch), OPAQUE_REASON)
        self.runs_as_judged(example, self.forwarding(fetch), events, [("fetch", "S", True)])

    def test_a_single_dispatch_function(self, example):
        """Register one plain function with each implementation's code under a branch on
        the type."""
        import functools

        events: list = []

        @functools.singledispatch
        def export(sql, force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            return "recorded"

        @export.register(str)
        def _(sql, force=False):
            events.append(("fetch", sql, force))
            return "recorded"

        self.refused(example, export, SINGLEDISPATCH_REASON)

        def branching(sql: str, force: bool = False) -> str:
            """Record SQL text without connecting to a database."""
            if isinstance(sql, str):
                events.append(("fetch", sql, force))
            return "recorded"

        self.runs_as_judged(example, branching, events, [("fetch", "S", True)])
        self.runs_as_judged(example, self.fetch(events), events, [("fetch", "S", True)])

    def test_a_cache_whose_function_cannot_be_told(self, example, monkeypatch):
        """``UnreadCache``: register the function itself, without the cache. The listing
        is shortened as in ``TestKernelRefusals``."""
        from recusal import invocation

        events: list = []
        fetch = self.fetch(events)
        invocation.reached_parameters(fetch)
        listing = invocation._referents
        monkeypatch.setattr(invocation, "_referents", lambda value: listing(value)[:-1])
        self.refused(example, self.again(self.forwarding(fetch), fetch), UNREAD_CACHE_REASON)
        self.runs_as_judged(example, fetch, events, [("fetch", "S", True)])

    def test_a_declared_signature(self, example):
        """Remove the declaration."""
        import inspect

        events: list = []
        fetch = self.fetch(events)
        fetch.__signature__ = inspect.signature(fetch)
        self.refused(example, fetch, DECLARED_REASON)
        del fetch.__signature__
        self.runs_as_judged(example, fetch, events, [("fetch", "S", True)])

    def test_a_bound_argument(self, example):
        """``BoundArgumentKept``: register the function, or write the method without
        ``functools.wraps``, naming the parameters, and pass them on by keyword. The old
        advice ran ``search(tenant="Q", query=5, limit=10)``."""
        import functools

        events: list = []

        def search(tenant: str = "default", query: str = "", limit: int = 10) -> str:
            """Record a search without contacting anything."""
            events.append(("search", tenant, query, limit))
            return "recorded"

        function = search

        class Facade:
            tenant = "acme"

            @functools.wraps(function)
            def search(self, *args, **kwargs):
                return function(*args, **kwargs)

        with pytest.raises(TypeError, match="handed a bound argument"):
            example.gate_tools(
                [example.function_tool(Facade().search, strict_mode=False)],
                example.sql_scope_policy,
            )

        class Written:
            tenant = "acme"

            def search(self, query: str = "", limit: int = 10) -> str:
                """Record a search without contacting anything."""
                return function(tenant=self.tenant, query=query, limit=limit)

        arguments = {"query": "Q", "limit": 5}
        self.runs_as_judged(
            example, Written().search, events, [("search", "acme", "Q", 5)], arguments
        )
        self.runs_as_judged(
            example,
            search,
            events,
            [("search", "t", "Q", 5)],
            {"tenant": "t", "query": "Q", "limit": 5},
        )

    def test_a_decorated_method_whose_object_place_has_another_name(self, example):
        """Found by the confirmation review of 0.2.3: a correct decorated method whose
        object's place is named ``this`` is refused by the gate's name rule, and
        registering the function itself gives another refusal (``this`` is then a
        parameter of the tool). The reason says to name it ``self``; so named, the method
        is gated and runs as judged."""
        import functools

        events: list = []

        def logged(m):
            @functools.wraps(m)
            def method(self, *args, **kwargs):
                return m(self, *args, **kwargs)

            return method

        class Named:
            @logged
            def run_sql(this, sql: str, force: bool = False) -> str:  # noqa: N805
                """Record SQL text without connecting to a database."""
                events.append(("fetch", sql, force))
                return "recorded"

        self.refused(example, Named().run_sql, KEPT_RECEIVER_REASON.format(name="this"))
        assert "name it `self`" in KEPT_RECEIVER_REASON

        class Renamed:
            @logged
            def run_sql(self, sql: str, force: bool = False) -> str:
                """Record SQL text without connecting to a database."""
                events.append(("fetch", sql, force))
                return "recorded"

        self.runs_as_judged(example, Renamed().run_sql, events, [("fetch", "S", True)])

    def test_a_method_that_wraps_a_function_not_written_to_take_the_object(self, example):
        """The gate's own refusal for a method facade over a function whose first
        parameter is not the object's: register the function itself."""
        events: list = []
        execute, Warehouse = TestStatedLimits.warehouse(events, "drops")
        self.refused(example, Warehouse().run_sql, KEPT_RECEIVER_REASON.format(name="database"))
        arguments = {"database": "d", "sql": "SELECT 1", "note": "n"}
        policy = self.Recording(lambda name, args: [])
        (gated,) = example.gate_tools(
            [example.function_tool(execute, name_override="run_sql", strict_mode=False)], policy
        )
        outputs, _ = example.run_calls([("run_sql", arguments, "c1")], gated=False, tools=[gated])
        assert outputs == {"c1": "recorded"} and policy.seen == [("run_sql", arguments)]
        assert events == [("execute", "d", "SELECT 1", "n")]


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestArgumentsThatAreNotText:
    """Found in review round nine of ``recusal-openai-agents-mcp``, on the hosted approval
    seam that package builds on ``refusal``: arguments that arrive as a ``dict``, a list or
    a number, not as JSON text, left the gate as a ``TypeError`` out of ``json.loads``.
    Nothing was approved and nothing ran, but it was an exception and not a refusal, and
    an empty ``dict`` or list, ``0`` and ``False`` were read as no arguments and judged.
    Every one is refused now, before the policy, on each way into the one decode the
    package has, and nothing the arguments hold is in the reason or on the log."""

    example = TestJudgedIsWhatRuns.example
    Recording = TestEffectiveArguments.Recording

    SECRET = "SECRET-VALUE-93417"

    class Falsy(str):
        """Text that says it is false: ``arguments or "{}"`` read it as no arguments."""

        def __bool__(self):
            return False

    class OwnDecoding(bytes):
        """Bytes that decode to something else than they hold."""

        def decode(self, *args, **kwargs):
            return "{}"

    #: name -> arguments that are not text, and the name of the type in the reason
    NOT_TEXT = {
        "str-subclass-that-is-false": (
            lambda secret: TestArgumentsThatAreNotText.Falsy('{"sql": "%s"}' % secret),
            "Falsy",
        ),
        "bytes-subclass-with-its-own-decode": (
            lambda secret: TestArgumentsThatAreNotText.OwnDecoding(
                ('{"sql": "%s"}' % secret).encode()
            ),
            "OwnDecoding",
        ),
        "memoryview": (lambda secret: memoryview(b"{}"), "memoryview"),
        "dict": (lambda secret: {"sql": secret}, "dict"),
        "list": (lambda secret: [secret], "list"),
        "int": (lambda secret: 5, "int"),
        "float": (lambda secret: 1.5, "float"),
        "true": (lambda secret: True, "bool"),
        "empty-dict": (lambda secret: {}, "dict"),
        "empty-list": (lambda secret: [], "list"),
        "zero": (lambda secret: 0, "int"),
        "false": (lambda secret: False, "bool"),
    }

    @staticmethod
    def reason(tool, kind):
        return (
            f"Recusal refused `{tool}`: arguments are not JSON text (they arrived as a "
            f"`{kind}`, which is not exactly `str`, `bytes` or `bytearray`), so they cannot be "
            "judged"
        )

    @pytest.mark.parametrize("how", sorted(NOT_TEXT))
    def test_the_hosted_seam_refuses_them(self, example, how):
        """``refusal`` is what the hosted approval callback of ``recusal-openai-agents-mcp``
        decides with: a reason means "do not approve"."""
        import recusal_openai_agents

        make, kind = self.NOT_TEXT[how]
        policy = self.Recording(lambda name, args: [])
        answer = recusal_openai_agents.refusal("lookup", make(self.SECRET), policy)
        assert answer == self.reason("lookup", kind)
        assert policy.seen == [] and self.SECRET not in answer

    @pytest.mark.parametrize("how", sorted(NOT_TEXT))
    def test_the_guardrail_of_a_function_tool_refuses_them(self, example, how):
        """The guardrail as the SDK calls it, handed a tool context whose arguments are
        not text. The SDK hands text; this is the same decode, held to the same answer."""
        from types import SimpleNamespace

        make, kind = self.NOT_TEXT[how]
        ran: list = []

        def run_sql(sql: str = "") -> str:
            """Record SQL text without connecting to a database."""
            ran.append(sql)
            return "recorded"

        policy = self.Recording(lambda name, args: [])
        (gated,) = example.gate_tools([example.function_tool(run_sql, strict_mode=False)], policy)
        guardrail = gated.tool_input_guardrails[0].guardrail_function
        ctx = SimpleNamespace(tool_name="run_sql", tool_arguments=make(self.SECRET))
        output = guardrail(SimpleNamespace(context=ctx))
        assert output.behavior == {
            "type": "reject_content",
            "message": self.reason("run_sql", kind),
        }
        assert policy.seen == [] and ran == []
        assert not hasattr(ctx, "_function_tool_arguments")  # nothing was pinned to run

    @pytest.mark.parametrize("how", sorted(NOT_TEXT))
    def test_an_agent_tool_refuses_them(self, example, how):
        make, kind = self.NOT_TEXT[how]
        policy = self.Recording(lambda name, args: [])
        sub = example.gated_agent(example.Agent(name="sub", tools=[]), policy)
        (tool,) = example.gate_tools([sub.as_tool(tool_name="ask", tool_description="d")], policy)
        answer = _adapter()._function_tool_refusal("ask", make(self.SECRET), policy, tool)
        assert answer == self.reason("ask", kind)
        assert policy.seen == []

    @pytest.mark.parametrize("how", ["dict", "list", "int"])
    def test_with_a_log_the_refusal_is_one_entry_that_holds_no_value(self, example, how, tmp_path):
        """Recorded, the refusal goes through ``decide`` like every other: one entry,
        bound to the name of the type the arguments arrived as and to nothing they hold."""
        import recusal_openai_agents

        from recusal.audit import AuditLog, verify

        make, kind = self.NOT_TEXT[how]
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path=str(path))
        policy = self.Recording(lambda name, args: [])
        answer = recusal_openai_agents.refusal(
            "lookup", make(self.SECRET), policy, audit=log, surface="test.hosted", tool_use_id="r1"
        )
        assert answer is not None and answer.startswith("Recusal refused `lookup` [FAIL]")
        assert f"they arrived as a `{kind}`" in answer and self.SECRET not in answer
        assert policy.seen == []
        assert len(log.entries) == 1 and log.entries[0]["decision"] == "FAIL"
        assert self.SECRET not in path.read_text(encoding="utf-8")
        assert verify(log.entries) == (True, [])

    def test_a_type_name_that_is_not_a_word_is_not_put_in_the_reason(self, example):
        """The reason and the record name the type the arguments arrived as. A class can
        be given any name; only a plain identifier is passed on."""
        import recusal_openai_agents

        odd = type('x`\n"}], "decision": "PASS', (), {})
        policy = self.Recording(lambda name, args: [])
        answer = recusal_openai_agents.refusal("lookup", odd(), policy)
        assert answer == self.reason("lookup", "object") and policy.seen == []

    def test_empty_text_and_empty_bytes_are_no_arguments(self, example):
        import recusal_openai_agents

        for arguments in ("", b"", bytearray()):
            policy = self.Recording(lambda name, args: [])
            assert recusal_openai_agents.refusal("lookup", arguments, policy) is None
            assert policy.seen == [("lookup", {})]
        policy = self.Recording(lambda name, args: [])
        assert recusal_openai_agents.refusal("lookup", "  ", policy) == (
            "Recusal refused `lookup`: arguments are not valid JSON"
        )
        assert policy.seen == []

    def test_text_none_and_bytes_are_judged_as_before(self, example):
        """What is not changed: a string, the empty string, None and bytes are decoded and
        judged. A ``dict`` is not taken for arguments that were decoded already."""
        import recusal_openai_agents

        for arguments, judged in (
            ('{"sql": "x"}', {"sql": "x"}),
            ("", {}),
            (None, {}),
            (b'{"sql": "x"}', {"sql": "x"}),
            (bytearray(b'{"sql": "x"}'), {"sql": "x"}),
        ):
            policy = self.Recording(lambda name, args: [])
            assert recusal_openai_agents.refusal("lookup", arguments, policy) is None
            assert policy.seen == [("lookup", judged)]


def types_method(func, obj):
    """A bound method of ``func`` on ``obj``, built by hand."""
    import types

    return types.MethodType(func, obj)
