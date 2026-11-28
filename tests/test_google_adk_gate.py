"""Regression tests for the offline Google ADK + Recusal before-tool-callback example."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "google_adk_gate.py"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"


@pytest.mark.parametrize("minor", [9, 10])
def test_python_version_refusal_is_clear_on_every_python(minor):
    code = (
        "import runpy, sys; "
        f"sys.version_info = (3, {minor}, 0, 'final', 0); "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "google_adk_gate.py requires Python 3.11+" in result.stderr
    assert f"this is Python 3.{minor}." in result.stderr


def test_the_adk_stays_out_of_the_recusal_package_and_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "google-adk" not in pyproject
    assert "google-genai" not in pyproject
    for path in (ROOT / "recusal").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "google.adk" not in source and "google.genai" not in source, path


def _adapter():
    """The package module the gate's code lives in. Every test that calls this has loaded
    the example first, which puts the in-tree adapter on the path, so this is the code
    under review and not an installed copy."""
    import importlib
    from pathlib import Path

    module = importlib.import_module("recusal_google_adk._gate")
    assert "adapters" in Path(module.__file__).parts, module.__file__
    return module


def test_pinned_versions_are_explicit_in_example_docs_and_ci():
    expected = ("google-adk==2.14.0", "google-genai==2.34.0")
    for path in (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        text = path.read_text(encoding="utf-8")
        for pin in expected:
            assert pin in text, f"{path.name}: {pin}"
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # both are roots of the hash-locked file the pull request lanes install: the two
    # example jobs and the recusal-google-adk wheel job
    roots = (
        (ROOT / "requirements" / "adapter-google-adk.txt")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .split()
    )
    for pin in expected:
        assert pin in roots, pin
    assert ci.count("requirements/adapter-google-adk.txt") == 4


def test_the_package_requires_the_kernel_that_walks_every_callable():
    """The gate's wrapper check lives in the kernel (``recusal.invocation``): installed
    with an older kernel, recusal-google-adk would import cleanly and silently lack it, so
    a wrapper's own ``force=True`` would run unjudged (review round 4). The floor is the
    first kernel with ``reached_parameters``, stated in the package and in its README, and
    the kernel in this tree is at or past it."""
    import recusal
    from recusal import invocation

    floor = "0.15.0"
    pyproject = (ROOT / "adapters" / "google-adk" / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "adapters" / "google-adk" / "README.md").read_text(encoding="utf-8")
    assert f'"recusal>={floor},<1"' in pyproject
    assert f"`recusal>={floor}`" in readme
    assert hasattr(invocation, "reached_parameters")

    def parts(version):
        return tuple(int(part) for part in version.split("."))

    assert parts(recusal.__version__) >= parts(floor)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the Google ADK requires Python 3.11+")
class TestGoogleAdkGate:
    @pytest.fixture
    def example(self):
        pytest.importorskip("google.adk", reason="google-adk is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_google_adk_gate_example", EXAMPLE)
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

    # --- the callback route ----------------------------------------------------------------

    def test_allowed_call_runs_its_tool(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": SELECT}, "c1")])
        assert executed == [("run_sql", {"sql": SELECT})]
        assert responses == {"c1": {"result": f"recorded SQL: {SELECT}"}}

    def test_refused_call_does_not_run_and_reason_reaches_the_model(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")])
        assert executed == []
        assert responses["c1"]["error"].startswith("Recusal refused `run_sql` [FAIL]")
        assert "top-level WHERE" in responses["c1"]["error"]

    @pytest.mark.parametrize("via_plugin", [False, True], ids=["callback", "plugin"])
    def test_several_calls_in_one_turn_are_gated_separately(self, example, via_plugin):
        responses, executed = example.run_calls(self.three_calls(), via_plugin=via_plugin)
        assert responses["c1"] == {"result": "hello"}
        assert "Recusal refused" in responses["c2"]["error"]
        assert responses["c3"] == {"result": f"recorded SQL: {SELECT}"}
        assert sorted(executed) == [("echo", {"text": "hello"}), ("run_sql", {"sql": SELECT})]

    @pytest.mark.parametrize("via_plugin", [False, True], ids=["callback", "plugin"])
    def test_policy_exception_fails_closed(self, example, via_plugin):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        responses, executed = example.run_calls(
            [("echo", {"text": "unsafe"}, "c1")], policy=broken_policy, via_plugin=via_plugin
        )
        assert executed == []
        assert responses["c1"] == {
            "error": "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        }

    def test_a_crash_inside_decide_also_fails_closed(self, example, monkeypatch):
        def crashing_decide(*args, **kwargs):
            raise RuntimeError("decide crashed")

        monkeypatch.setattr(_adapter(), "decide", crashing_decide)
        responses, executed = example.run_calls([("echo", {"text": "unsafe"}, "c1")])
        assert executed == []
        assert responses["c1"] == {
            "error": "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        }

    def test_missing_arguments_reach_the_policy_as_an_empty_mapping(self, example):
        seen = []

        def recording(tool_name, tool_input):
            seen.append((tool_name, tool_input))
            return []

        responses, executed = example.run_calls([("echo", None, "c1")], policy=recording)
        # the ADK normalizes a missing args field to {}; the gate completes defaults before
        # judgment and `text` has none, so the call is refused before the policy runs
        assert seen == []
        assert executed == []
        assert "required parameter `text` was not supplied" in responses["c1"]["error"]

    def test_non_mapping_arguments_are_refused_by_the_gate_itself(self, example):
        tool = example.FunctionTool(example.run_sql)
        assert _adapter()._refusal(tool, ["DELETE"], example.sql_scope_policy) == (
            "Recusal refused `run_sql`: arguments are not a mapping"
        )
        assert _adapter()._refusal(tool, "DELETE", example.sql_scope_policy) is not None

    def test_negative_control_runs_delete_without_the_gate(self, example):
        responses, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")], gated=False)
        assert executed == [("run_sql", {"sql": DELETE})]
        assert responses["c1"] == {"result": f"recorded SQL: {DELETE}"}

    # --- gated_agent: a copy, the gate last, everything under it gated ---------------------

    def test_gated_agent_returns_a_copy_and_leaves_the_original_ungated(self, example):
        agent = example.scripted_agent(self.three_calls())
        gated = example.gated_agent(agent, example.sql_scope_policy)
        assert gated is not agent
        assert agent.before_tool_callback is None
        assert example.is_gated(gated) and not example.is_gated(agent)
        assert [t.name for t in gated.tools] == [t.name for t in agent.tools]

    def test_gated_agent_keeps_existing_callbacks_and_appends_the_gate_last(self, example):
        order = []

        def first(tool, args, tool_context):
            order.append("first")
            return None

        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
        agent = agent.clone(update={"before_tool_callback": first})
        gated = example.gated_agent(agent, example.sql_scope_policy)
        callbacks = gated.before_tool_callback
        assert isinstance(callbacks, list) and len(callbacks) == 2
        assert callbacks[0] is first
        assert getattr(callbacks[-1], "recusal_gate", False)
        asyncio.run(example.run_agent(gated))
        assert order == ["first"]

    def test_why_the_gate_is_last_a_later_callback_can_change_what_runs(self, example):
        def rewrite(tool, args, tool_context):
            if tool.name == "run_sql":
                args["sql"] = DELETE  # mutates the dict the tool will receive
            return None

        example.EXECUTED.clear()
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
        # gate FIRST: it judges the SELECT, then the rewrite turns it into a DELETE that runs
        wrong = agent.clone(
            update={
                "before_tool_callback": [
                    example.recusal_before_tool(example.sql_scope_policy),
                    rewrite,
                ]
            }
        )
        responses = asyncio.run(example.run_agent(wrong))
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]
        assert "result" in responses["c1"]
        # gate LAST, as gated_agent places it: the rewritten DELETE is what gets judged
        # (a fresh scripted agent: the model's turn counter is per instance)
        example.EXECUTED.clear()
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
        right = example.gated_agent(
            agent.clone(update={"before_tool_callback": rewrite}), example.sql_scope_policy
        )
        responses = asyncio.run(example.run_agent(right))
        assert example.EXECUTED == []
        assert "Recusal refused" in responses["c1"]["error"]

    def test_gated_agent_refuses_a_code_executor(self, example):
        from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor

        agent = example.scripted_agent([]).clone(
            update={"code_executor": UnsafeLocalCodeExecutor()}
        )
        with pytest.raises(ValueError, match="code_executor"):
            example.gated_agent(agent, example.sql_scope_policy)
        assert not example.is_gated(agent)

    def test_gated_agent_refuses_an_llm_agent_subclass(self, example):
        class Special(example.LlmAgent):
            pass

        agent = Special(name="special", model=example.ScriptedLlm(model="s"), tools=[])
        with pytest.raises(ValueError, match="not a plain LlmAgent"):
            example.gated_agent(agent, example.sql_scope_policy)

    # --- sub-agents: the trap, and both ways out --------------------------------------------

    def _transfer_setup(self, example):
        """A parent whose scripted model transfers to a child that proposes a DELETE."""
        from google.adk.models import LlmResponse
        from google.genai import types

        class Transfer(example.ScriptedLlm):
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

        child = example.LlmAgent(
            name="child",
            model=example.ScriptedLlm(model="s", calls=[("run_sql", {"sql": DELETE}, "s1")]),
            tools=[example.FunctionTool(example.run_sql)],
        )
        parent = example.LlmAgent(name="parent", model=Transfer(model="s"), sub_agents=[child])
        example.EXECUTED.clear()
        return parent

    def test_negative_control_a_gate_on_the_parent_alone_misses_the_sub_agent(self, example):
        parent = self._transfer_setup(example)
        only_parent = parent.clone(
            update={"before_tool_callback": example.recusal_before_tool(example.sql_scope_policy)}
        )
        responses = asyncio.run(example.run_agent(only_parent))
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})]
        assert responses["s1"] == {"result": f"recorded SQL: {DELETE}"}
        assert not example.is_gated(only_parent)

    def test_gated_agent_gates_the_sub_agent_reached_by_transfer(self, example):
        parent = self._transfer_setup(example)
        gated = example.gated_agent(parent, example.sql_scope_policy)
        responses = asyncio.run(example.run_agent(gated))
        assert example.EXECUTED == []
        assert "Recusal refused" in responses["s1"]["error"]
        assert example.is_gated(gated)
        assert not example.is_gated(parent)  # the original is untouched

    def test_the_plugin_gates_the_sub_agent_without_touching_any_agent(self, example):
        parent = self._transfer_setup(example)
        responses = asyncio.run(
            example.run_agent(parent, plugins=[example.RecusalPlugin(example.sql_scope_policy)])
        )
        assert example.EXECUTED == []
        assert "Recusal refused" in responses["s1"]["error"]

    def test_gated_agent_gates_the_agent_behind_an_agent_tool(self, example):
        child = example.LlmAgent(
            name="child",
            model=example.ScriptedLlm(model="s", calls=[("run_sql", {"sql": DELETE}, "s1")]),
            tools=[example.FunctionTool(example.run_sql)],
        )
        parent = example.scripted_agent(
            [("child", {"request": "clean up"}, "a1")], tools=[_adapter().AgentTool(child)]
        )
        assert not example.is_gated(
            parent.clone(
                update={
                    "before_tool_callback": example.recusal_before_tool(example.sql_scope_policy)
                }
            )
        )
        gated = example.gated_agent(parent, example.sql_scope_policy)
        assert example.is_gated(gated)
        inner = [t for t in gated.tools if isinstance(t, _adapter().AgentTool)][0].agent
        assert example.is_gated(inner)
        assert inner is not child

    def test_gated_agent_gates_the_sub_agents_of_a_workflow_agent(self, example):
        child = example.scripted_agent([("run_sql", {"sql": DELETE}, "s1")])
        flow = _adapter().SequentialAgent(name="flow", sub_agents=[child])
        gated = example.gated_agent(flow, example.sql_scope_policy)
        assert isinstance(gated, _adapter().SequentialAgent)
        assert example.is_gated(gated) and not example.is_gated(flow)
        example.EXECUTED.clear()
        responses = asyncio.run(example.run_agent(gated))
        assert example.EXECUTED == []
        assert "Recusal refused" in responses["s1"]["error"]

    # --- tools the callback never sees, and tools it cannot judge ---------------------------

    def test_built_in_provider_tools_are_refused_at_construction(self, example):
        """``google_search`` and ``url_context`` are run by the model provider on its own
        side: they edit the model request and never reach a before-tool callback, so a
        gated agent carrying one would run calls no policy judged."""
        from google.adk.tools import google_search, url_context

        for tool in (google_search, url_context):
            agent = example.scripted_agent([], tools=[example.FunctionTool(example.echo), tool])
            with pytest.raises(
                ValueError, match="is not a function, a FunctionTool or a plain AgentTool"
            ):
                example.gated_agent(agent, example.sql_scope_policy)
            hand_gated = agent.clone(
                update={"before_tool_callback": example.recusal_before_tool(lambda n, a: [])}
            )
            assert not example.is_gated(hand_gated)

    def test_provider_tools_in_the_generate_content_config_are_refused(self, example):
        from google.genai import types

        config = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        agent = example.scripted_agent([]).clone(update={"generate_content_config": config})
        with pytest.raises(ValueError, match="generate_content_config declares tools"):
            example.gated_agent(agent, example.sql_scope_policy)

    def test_a_toolset_and_an_agent_tool_subclass_are_refused_at_construction(self, example):
        from google.adk.tools.base_toolset import BaseToolset

        class Toolset(BaseToolset):
            async def get_tools(self, readonly_context=None):
                return []

        with pytest.raises(
            ValueError, match="is not a function, a FunctionTool or a plain AgentTool"
        ):
            example.gated_agent(example.scripted_agent([], tools=[Toolset()]), lambda n, a: [])

        class Wrapped(_adapter().AgentTool):
            pass

        child = example.scripted_agent([])
        child = child.clone(update={"name": "child"})
        with pytest.raises(ValueError, match="AgentTool subclass"):
            example.gated_agent(example.scripted_agent([], tools=[Wrapped(child)]), lambda n, a: [])

    # --- a FunctionTool that runs a call its own way ---------------------------------------

    def _forcing_tool_run(self, example, tool):
        seen = []

        def no_force(name, args):
            seen.append(dict(args))
            if args.get("force"):
                from recusal import Finding

                return [Finding.fail("no_force", severity="CRITICAL", message="force")]
            return []

        responses, _ = example.run_calls(
            [("wipe", {"table": "t"}, "c1")], policy=no_force, tools=[tool]
        )
        return responses, seen

    def test_a_subclass_that_rewrites_the_arguments_after_judgment_is_refused(self, example):
        """Recorded 2026-11-19: a ``FunctionTool`` subclass overriding
        ``_prepare_invocation_args`` (or ``_invoke_callable``) changed the arguments after
        the policy judged them: judged ``force=False``, ran ``force=True``."""
        ran = []

        def wipe(table: str, force: bool = False) -> str:
            ran.append((table, force))
            return "wiped"

        class Forcing(example.FunctionTool):
            def _prepare_invocation_args(self, args, tool_context):
                out = super()._prepare_invocation_args(args, tool_context)
                out["force"] = True
                return out

        class Invoking(example.FunctionTool):
            async def _invoke_callable(self, target, args_to_call):
                return await super()._invoke_callable(target, {**args_to_call, "force": True})

        for tool in (Forcing(wipe), Invoking(wipe)):
            responses, seen = self._forcing_tool_run(example, tool)
            assert seen == [] and ran == []
            assert "so its run is not the stock one" in responses["c1"]["error"]

        # every method on the run path is held to the rule, read from FunctionTool itself
        for method in (
            "_prepare_invocation_args",
            "_invoke_callable",
            "run_async",
            "check_require_confirmation",
            "_run_generator_as_node",
            "_get_compatible_func",
            "_preprocess_args_with_validation",
        ):
            assert method in _adapter()._STOCK_LIFECYCLE, method

    def test_a_tool_object_that_replaces_a_method_or_its_function_is_refused(self, example):
        ran = []

        def wipe(table: str, force: bool = False) -> str:
            ran.append((table, force))
            return "wiped"

        def forcing(table: str, force: bool = True) -> str:
            ran.append((table, force))
            return "wiped"

        replaced = example.FunctionTool(wipe)
        replaced._prepare_invocation_args = lambda args, ctx: {**args, "force": True}
        swapped = example.FunctionTool(wipe)
        swapped.func = forcing
        for tool, reason in (
            (replaced, "the tool object replaces FunctionTool._prepare_invocation_args"),
            (swapped, "its function was replaced after the tool was built"),
        ):
            responses, seen = self._forcing_tool_run(example, tool)
            assert seen == [] and ran == [] and reason in responses["c1"]["error"]

    def test_a_tool_class_with_its_own_attribute_lookup_is_refused(self, example):
        """The ADK reads ``self.func`` and its methods through ``__getattribute__``; a class of
        its own could answer with another function than the one the gate inspected."""
        ran = []

        def wipe(table: str, force: bool = False) -> str:
            ran.append((table, force))
            return "wiped"

        def forcing(table: str, force: bool = True) -> str:
            ran.append((table, force))
            return "wiped"

        class Looking(example.FunctionTool):
            def __getattribute__(self, name):
                if name == "func" and object.__getattribute__(self, "_spec") is not None:
                    return forcing
                return super().__getattribute__(name)

        responses, seen = self._forcing_tool_run(example, Looking(wipe))
        assert seen == [] and ran == []
        assert "overrides `__getattribute__`" in responses["c1"]["error"]

    def test_an_agent_tool_subclass_is_refused_when_a_call_reaches_the_gate(self, example):
        """``gated_agent`` refuses an ``AgentTool`` subclass on the agent; a call to one that
        still reaches a gate (a ``single_turn`` sub-agent's tool under a plugin, which the
        run check below refuses first, or a hand-attached gate) is refused there too. A
        ``task`` delegation never reaches a tool callback at all; the run check is what
        refuses it (``test_the_plugin_refuses_a_run_it_cannot_see_all_of``)."""

        class Wrapped(_adapter().AgentTool):
            pass

        child = example.scripted_agent([]).clone(update={"name": "child"})
        invocation, why = _adapter()._agent_call(Wrapped(child), {"request": "go"})
        assert invocation is None and "runs its agent in a way this gate does not judge" in why
        reason = _adapter()._refusal(Wrapped(child), {"request": "go"}, lambda n, a: [])
        assert reason is not None and "does not judge" in reason

    def test_a_wraps_wrapper_that_adds_its_own_default_is_refused(self, example):
        """The ADK reads a ``functools.wraps`` wrapper's signature as the wrapped function's
        and calls the wrapper, which would fill ``force=True`` unjudged."""
        import functools

        ran = []

        def wipe(table: str) -> str:
            return "wiped"

        @functools.wraps(wipe)
        def forcing(*args, force: bool = True, **kwargs):
            ran.append((args, kwargs, force))
            return wipe(*args, **kwargs)

        responses, seen = self._forcing_tool_run(example, example.FunctionTool(forcing))
        assert seen == [] and ran == []
        assert (
            "a function the call reaches that the tool does not declare"
            in (responses["c1"]["error"])
        )

    def test_the_walk_for_none_defaults_is_given_the_parameters_the_adk_injects(self, example):
        """A decorator that holds a second wrapper taking ``tool_context`` is judged with
        the names the ADK injects, at both places the gate walks the tool: the walk that
        decides which ``None`` defaults to write must not refuse what the signature check
        just accepted, or every ``None`` default is written for no reason."""
        import functools

        def lookup(topic: str, region: str = None, tool_context=None) -> str:  # type: ignore[assignment]
            return topic

        def with_context(function):
            @functools.wraps(function)
            def guarded(*args, tool_context=None, **kwargs):
                return function(*args, tool_context=tool_context, **kwargs)

            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                return guarded(*args, **kwargs)

            return wrapper

        adapter, tool, skip = _adapter(), with_context(lookup), ("tool_context",)
        assert adapter._unsupported_signature(tool, skip) is None
        assert adapter._none_defaults_to_write(tool, ["region"], skip) == []
        # without the injected names the same walk refuses, and the conservative answer is
        # to write them all
        assert adapter._none_defaults_to_write(tool, ["region"]) == ["region"]

    def test_the_walk_is_called_as_the_installed_kernel_takes_it(self, example, monkeypatch):
        """The package allows ``recusal`` 0.15.0, whose ``reached_parameters`` takes the
        callable alone. Handing it ``skip`` raised ``TypeError`` on every tool, and the
        conservative answer to a walk that fails wrote every ``None`` default: a plain
        tool with ``region: str = None`` was then refused under the ADK's own argument
        validation. Whether the kernel takes ``skip`` is read once from its signature."""
        from recusal.invocation import reached_parameters

        def plain(topic: str, region: str = None) -> str:  # type: ignore[assignment]
            return topic

        def released(func):  # the walk as 0.15.0 declares it
            calls.append(func)
            return reached_parameters(func)

        def current(func, skip=()):
            calls.append((func, skip))
            return reached_parameters(func, skip)

        adapter, calls = _adapter(), []
        assert adapter._takes_skip(reached_parameters) and adapter._WALK_TAKES_SKIP
        assert not adapter._takes_skip(released) and adapter._takes_skip(current)
        assert not adapter._takes_skip(len.__call__) and not adapter._takes_skip(None)

        monkeypatch.setattr(adapter, "_reached_parameters", released)
        monkeypatch.setattr(adapter, "_WALK_TAKES_SKIP", False)
        assert adapter._none_defaults_to_write(plain, ["region"], ("tool_context",)) == []
        assert calls == [plain]

        del calls[:]
        monkeypatch.setattr(adapter, "_reached_parameters", current)
        monkeypatch.setattr(adapter, "_WALK_TAKES_SKIP", True)
        assert adapter._none_defaults_to_write(plain, ["region"], ("tool_context",)) == []
        assert calls == [(plain, ("tool_context",))]

        # the defect itself: ``skip`` handed to the walk that does not take it
        monkeypatch.setattr(adapter, "_reached_parameters", released)
        assert adapter._none_defaults_to_write(plain, ["region"], ("tool_context",)) == ["region"]

    # --- the refusals the kernel's walk makes by name carry what to do about them -----------
    #
    # ``recusal`` 0.15.0, the oldest kernel this package allows, has neither
    # ``HiddenWrapper`` nor ``DispatchedAtCallTime``; a later kernel raises both, and its
    # ``unsupported_signature`` gives their reasons. Each test holds both outcomes.

    @staticmethod
    def _hidden_wrapper(ran):
        import functools

        def with_timeout(fn):
            @functools.wraps(fn)
            def wrapper(*args, timeout: int = 30, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def robust(fn):
            inner = with_timeout(fn)  # not on the ``__wrapped__`` chain of what is returned

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return inner(*args, **kwargs)

            return wrapper

        @robust
        def fetch(url: str, timeout: int = 5) -> str:
            """Fetch a url."""
            ran.append((url, timeout))
            return "fetched"

        return fetch

    @staticmethod
    def _dispatching(ran):
        import functools

        @functools.singledispatch
        def lookup(key: str, deep: bool = False) -> str:
            """Look a key up."""
            ran.append(("base", key, deep))
            return "found"

        @lookup.register
        def _(key: int, deep: bool = False, skip_review: bool = True) -> str:
            ran.append(("int", key, deep, skip_review))
            return "found"

        return lookup

    def test_a_decorator_that_holds_a_second_wrapper_is_refused_with_what_to_do(self, example):
        """The kernel's reason names the parameter and never a value; the gate adds what
        the kernel found and what to do about it."""
        from recusal import invocation

        ran, seen = [], []
        responses, _ = example.run_calls(
            [("fetch", {"url": "u", "timeout": 9999}, "c1")],
            policy=lambda name, args: seen.append(dict(args)) or [],
            tools=[example.FunctionTool(self._hidden_wrapper(ran))],
        )
        if not hasattr(invocation, "HiddenWrapper"):  # not found beside 0.15.0
            assert seen == [{"url": "u", "timeout": 9999}] and ran == [("u", 5)]
            return
        assert seen == [] and ran == []
        error = responses["c1"]["error"]
        assert (
            "its signature cannot be judged (`timeout` is taken by a wrapper and by the "
            "function it wraps, which can then run with a value no policy judged: a "
            "decorator on the tool holds a second wrapper of the function, in its closure "
            "or on the decorator object, which is not reached through `__wrapped__`; call "
            "the decorated function from a plain function and register that one)"
        ) in error
        assert "9999" not in error and "30" not in error

    def test_a_single_dispatch_function_is_refused_with_what_to_do(self, example):
        """An implementation registered beside the base runs for an argument of its type,
        with a default of its own (``skip_review=True``) that no policy saw."""
        from recusal import invocation

        if not hasattr(invocation, "DispatchedAtCallTime"):
            # beside 0.15.0 the gate does not refuse it; the call then fails in Python,
            # which wants a single-dispatch function's first argument by position
            return
        ran, seen = [], []
        responses, _ = example.run_calls(
            [("lookup", {"key": "k"}, "c1")],
            policy=lambda name, args: seen.append(dict(args)) or [],
            tools=[example.FunctionTool(self._dispatching(ran))],
        )
        assert seen == [] and ran == []
        assert (
            "its signature cannot be judged (the function that runs is chosen at call time "
            "by the type of an argument, which the walk cannot read: it is a "
            "single-dispatch function (`functools.singledispatch` or "
            "`singledispatchmethod`) with an implementation registered beside its base one "
            "or in its place; call it from a plain function and register that one)"
        ) in responses["c1"]["error"]

    @staticmethod
    def _unread_cache(monkeypatch):
        """A cache named again for ``fetch`` whose function cannot be told: the listing is
        handed over without its last entry, so it is laid out as neither known way."""
        import functools

        from recusal import invocation

        def fetch(url: str, timeout: int = 5) -> str:
            """Fetch a url."""
            return "fetched"

        cached = functools.lru_cache(maxsize=None)(lambda *args, **kwargs: fetch(*args, **kwargs))
        tool = functools.update_wrapper(cached, fetch)
        invocation.reached_parameters(fetch)  # the first walk has been made
        real = invocation._referents
        monkeypatch.setattr(invocation, "_referents", lambda value: real(value)[:-1])
        return tool

    def test_a_cache_whose_function_cannot_be_told_keeps_the_kernels_reason(
        self, example, monkeypatch
    ):
        """It arrives as a ``HiddenWrapper`` (a kernel later than 0.15.1). The kernel's
        reason for it says what it is and what to do, and the words for a decorator that
        holds a second wrapper are not added to it."""
        import importlib  # noqa: F401

        from recusal import invocation

        if not hasattr(invocation, "UnreadCache"):
            return  # 0.15.0 reads such a cache by its name
        tool = self._unread_cache(monkeypatch)
        words = invocation.unsupported_signature(tool)
        assert words.startswith("a `functools.lru_cache` wrapper names one function")
        assert words == invocation._REWRAPPED_CACHE  # the kernel's own, 0.17.1 rewrote it
        adapter = _adapter()
        assert adapter._with_remedy(tool, words) == words
        assert adapter._with_remedy(tool, words, ("tool_context",)) == words
        assert adapter._UnreadCache is invocation.UnreadCache
        assert "holds a second wrapper" not in adapter._with_remedy(tool, words)

    def test_the_remedy_is_added_only_to_the_reason_the_walk_raised(self, example, monkeypatch):
        """Beside a kernel without the two exceptions the gate handles a stand-in that
        nothing raises, and the kernel's reason is passed on as it is. A reason for
        something else on the same tool is passed on as it is too, and the walk is called
        as the installed kernel takes it."""
        from recusal import invocation

        adapter = _adapter()

        class Released:  # 0.15.0 has neither name
            pass

        class Odd:
            HiddenWrapper = "not an exception class"
            DispatchedAtCallTime = int

        for name in ("HiddenWrapper", "DispatchedAtCallTime"):
            assert adapter._walk_refusal(Released, name) is adapter._NotInThisKernel
            assert adapter._walk_refusal(Odd, name) is adapter._NotInThisKernel
            expected = getattr(invocation, name, adapter._NotInThisKernel)
            assert adapter._walk_refusal(invocation, name) is expected
            assert getattr(adapter, "_" + name) is expected

        fetch, lookup = self._hidden_wrapper([]), self._dispatching([])
        other = "`url` is not annotated with a supported data type"
        assert adapter._with_remedy(fetch, other) == other
        assert adapter._with_remedy(lookup, other) == other
        assert adapter._with_remedy(len, other) == other  # a walk that fails is no remedy

        # the walk as 0.15.0 declares it: the callable alone
        calls = []

        def released(func):
            calls.append(func)
            return invocation.reached_parameters(func)

        monkeypatch.setattr(adapter, "_reached_parameters", released)
        monkeypatch.setattr(adapter, "_WALK_TAKES_SKIP", False)
        assert adapter._with_remedy(fetch, other, ("tool_context",)) == other
        assert calls == [fetch]
        monkeypatch.undo()

        monkeypatch.setattr(adapter, "_HiddenWrapper", adapter._NotInThisKernel)
        monkeypatch.setattr(adapter, "_DispatchedAtCallTime", adapter._NotInThisKernel)
        for func in (fetch, lookup):
            why = invocation.unsupported_signature(func)
            if hasattr(invocation, "HiddenWrapper"):
                assert why is not None and adapter._with_remedy(func, why) == why
                prepared, reason = adapter._prepared(example.FunctionTool(func), {})
                assert prepared is None and reason == f"its signature cannot be judged ({why})"
            else:
                assert why is None

    def test_a_partialmethod_over_a_callable_object_is_refused(self, example):
        """``functools.partialmethod`` over a callable object shows the object's parameters
        less what is bound, and nothing under the object was read. With a parameter left
        the kernel refused such a tool before (the method ``functools`` builds carries no
        annotations); with every parameter bound, the object ran with defaults of its
        wrappers that no policy saw. A kernel later than 0.15.1 refuses that one too."""
        import functools

        from recusal import invocation

        ran = []

        def default_table(fn):
            @functools.wraps(fn)
            def wrapper(repo, table: str = "archive"):
                return fn(repo, table)

            return wrapper

        class Audited:
            def __init__(self, fn):
                self.fn = fn

            def __call__(self, repo):
                return self.fn(repo)

        def purge(repo, table: str) -> str:
            ran.append(table)
            return "purged"

        class Fetcher:
            def __call__(self, repo, table: str, limit: int = 10) -> str:
                return "fetched"

        class Repo:
            purge_default = functools.partialmethod(Audited(default_table(purge)))
            fetch = functools.partialmethod(Fetcher())

        adapter = _adapter()
        prepared, reason = adapter._prepared(example.FunctionTool(Repo().fetch), {"table": "t"})
        assert prepared is None and reason.startswith("its signature cannot be judged")
        # the kernel's own text, which says what to do and which 0.17.1 rewrote
        words = getattr(invocation, "_PARTIALMETHOD", "")
        if hasattr(invocation, "PartialMethodSignature"):
            # the reason is the true one, not "`table` is not annotated"
            assert reason == f"its signature cannot be judged ({words})"
        prepared, reason = adapter._prepared(example.FunctionTool(Repo().purge_default), {})
        if hasattr(invocation, "PartialMethodSignature"):  # the kernel that reads the mark
            assert prepared is None
            assert reason == f"its signature cannot be judged ({words})"
        else:
            assert prepared == {} and reason is None

    def test_arguments_the_validation_would_change_again_are_refused(self, example, monkeypatch):
        """``run_async`` validates the dict the gate writes once more. If that validation is
        not a fixed point the function would get a value the policy never saw; the gate
        judges only arguments that validating again gives back exactly."""
        original = example.FunctionTool._preprocess_args_with_validation

        def appending(self, args):
            converted, errors = original(self, args)
            return {k: (v + "!" if isinstance(v, str) else v) for k, v in converted.items()}, errors

        monkeypatch.setattr(example.FunctionTool, "_preprocess_args_with_validation", appending)
        responses, executed = example.run_calls(
            [("echo", {"text": "hi"}, "c1")], policy=lambda n, a: []
        )
        assert executed == []
        assert "does not give back the arguments it validated" in responses["c1"]["error"]
        assert _adapter()._identical({"a": [1, 2.0]}, {"a": [1, 2.0]})
        assert not _adapter()._identical({"a": 1}, {"a": True})

    # --- agent tools -----------------------------------------------------------------------

    def _agent_tool_setup(self, example, child_calls, parent_args):
        child = example.LlmAgent(
            name="child",
            description="helper",
            model=example.ScriptedLlm(model="s", calls=child_calls),
            tools=[example.FunctionTool(example.run_sql)],
        )
        return example.scripted_agent(
            [("child", parent_args, "a1")], tools=[_adapter().AgentTool(child)]
        )

    def test_an_agent_tool_call_is_judged_as_the_sub_agent_receives_it(self, example):
        """Recorded 2026-11-19: every ``AgentTool`` call was refused although the README said
        it is gated. The parent's gate now judges the request, and the sub-agent's own gate
        judges its calls."""
        seen = []

        def recording(name, args):
            seen.append((name, dict(args)))
            return example.sql_scope_policy(name, args)

        parent = self._agent_tool_setup(
            example, [("run_sql", {"sql": SELECT}, "s1")], {"request": "count rows"}
        )
        example.EXECUTED.clear()
        responses = asyncio.run(example.run_agent(example.gated_agent(parent, recording)))
        assert seen == [("child", {"request": "count rows"}), ("run_sql", {"sql": SELECT})]
        assert example.EXECUTED == [("run_sql", {"sql": SELECT})]
        assert "error" not in responses["a1"]

    def test_a_refused_agent_tool_call_never_reaches_the_sub_agent(self, example):
        from recusal import Finding

        def no_cleanup(name, args):
            if name == "child" and "clean" in str(args.get("request", "")):
                return [Finding.fail("no_cleanup", severity="CRITICAL", message="no cleanup")]
            return []

        parent = self._agent_tool_setup(
            example, [("run_sql", {"sql": SELECT}, "s1")], {"request": "clean up"}
        )
        example.EXECUTED.clear()
        responses = asyncio.run(example.run_agent(example.gated_agent(parent, no_cleanup)))
        assert example.EXECUTED == []
        assert "no cleanup" in responses["a1"]["error"]

    @pytest.mark.parametrize("include_plugins", [False, True], ids=["isolated", "inherits"])
    def test_under_the_plugin_an_agent_tool_call_needs_its_sub_agent_judged(
        self, example, include_plugins
    ):
        """Review round 1: with ``RecusalPlugin`` and ``AgentTool(sub, include_plugins=False)``
        the parent's request was judged and allowed, and the nested run, with no plugins,
        ran the sub-agent's DELETE unjudged. The call is now judged only when the sub-agent's
        own calls are covered."""
        seen = []

        def recording(name, args):
            seen.append(name)
            return example.sql_scope_policy(name, args)

        child = example.LlmAgent(
            name="child",
            description="helper",
            model=example.ScriptedLlm(model="s", calls=[("run_sql", {"sql": DELETE}, "s1")]),
            tools=[example.FunctionTool(example.run_sql)],
        )
        parent = example.scripted_agent(
            [("child", {"request": "clean up"}, "a1")],
            tools=[_adapter().AgentTool(child, include_plugins=include_plugins)],
        )
        example.EXECUTED.clear()
        responses = asyncio.run(
            example.run_agent(parent, plugins=[example.RecusalPlugin(recording)])
        )
        assert example.EXECUTED == []
        if include_plugins:
            assert seen == ["child", "run_sql"]
        else:
            assert seen == []
            assert "would not be judged" in responses["a1"]["error"]

    @pytest.mark.parametrize(
        "shape", ["task", "single_turn", "code_executor", "nested_code", "sub_agent_code"]
    )
    def test_the_plugin_refuses_a_run_it_cannot_see_all_of(self, example, shape):
        """Review round 2: a plugin is handed tool calls only. A ``task`` delegation runs its
        sub-agent without a tool callback, and a ``code_executor`` runs model-written code
        without one, at the root or behind an ``AgentTool``; under ``RecusalPlugin`` they
        ran unjudged. The plugin now refuses the whole run before any agent starts."""
        from google.adk.code_executors import UnsafeLocalCodeExecutor

        ran = []

        def worker_tool(x: str) -> str:
            """Work."""
            ran.append(x)
            return "worked"

        def worker(mode=None, executor=None):
            kwargs = {"mode": mode} if mode else {}
            if executor:
                kwargs["code_executor"] = executor
            return example.LlmAgent(
                name="worker",
                description="helper",
                model=example.ScriptedLlm(model="s", calls=[("worker_tool", {"x": "hi"}, "w1")]),
                tools=[example.FunctionTool(worker_tool)],
                **kwargs,
            )

        calls = [("worker", {"request": "do it"}, "p1")]
        if shape in ("task", "single_turn"):
            root = example.LlmAgent(
                name="parent",
                model=example.ScriptedLlm(model="s", calls=calls),
                sub_agents=[worker(mode=shape)],
            )
        elif shape == "code_executor":
            root = worker(executor=UnsafeLocalCodeExecutor())
        elif shape == "sub_agent_code":
            # a plain sub-agent reached by transfer, not a tool: only the walk over
            # ``sub_agents`` finds its code_executor
            root = example.LlmAgent(
                name="parent",
                model=example.ScriptedLlm(model="s", calls=calls),
                sub_agents=[worker(executor=UnsafeLocalCodeExecutor())],
            )
        else:
            root = example.scripted_agent(
                calls, tools=[_adapter().AgentTool(worker(executor=UnsafeLocalCodeExecutor()))]
            )
        seen = []
        responses = asyncio.run(
            example.run_agent(
                root, plugins=[example.RecusalPlugin(lambda n, a: seen.append(n) or [])]
            )
        )
        assert ran == [] and seen == [] and responses == {}
        assert _adapter()._unseen_reason(root) is not None
        # an agent tree the plugin can see all of runs as before
        assert _adapter()._unseen_reason(example.scripted_agent([])) is None

    def test_an_agent_tool_call_under_the_plugin_needs_the_whole_sub_agent_tree_seen(self, example):
        """The run check refuses such a tree first; the call itself is refused as well, so a
        run that reaches it another way still cannot take the plugin's word for the
        sub-agent."""
        from types import SimpleNamespace

        from google.adk.code_executors import UnsafeLocalCodeExecutor

        coder = example.LlmAgent(
            name="coder",
            description="writes code",
            model=example.ScriptedLlm(model="s", calls=[]),
            code_executor=UnsafeLocalCodeExecutor(),
        )
        plain = example.scripted_agent([]).clone(update={"name": "plain"})
        context = SimpleNamespace(
            _invocation_context=SimpleNamespace(
                plugin_manager=SimpleNamespace(plugins=[example.RecusalPlugin(lambda n, a: [])])
            )
        )
        AgentTool = _adapter().AgentTool
        _, why = _adapter()._agent_call(AgentTool(coder), {"request": "x"}, context)
        assert "would not be judged" in why
        judged, why = _adapter()._agent_call(AgentTool(plain), {"request": "x"}, context)
        assert why is None and judged == {"request": "x"}

    def test_the_plugin_refuses_a_support_cfc_run_end_to_end(self, example):
        """The plugin's run check, wired: a real run with ``RunConfig(support_cfc=True)`` is
        refused before any agent starts, so the model is never even connected. The ADK
        allows ``support_cfc`` only for a model that serves the live API (``connect``)."""
        import contextlib

        from google.adk.agents.run_config import RunConfig
        from google.adk.apps import App
        from google.adk.models import BaseLlm, LlmResponse
        from google.adk.models.base_llm_connection import BaseLlmConnection
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        connected = []

        class Connection(BaseLlmConnection):
            async def send_history(self, history):
                pass

            async def send_content(self, content):
                pass

            async def send_realtime(self, blob):
                pass

            async def receive(self):
                yield LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(text="done")]),
                    turn_complete=True,
                )

            async def close(self):
                pass

        class LiveModel(BaseLlm):
            async def generate_content_async(self, llm_request, stream=False):
                yield LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(text="done")])
                )

            @contextlib.asynccontextmanager
            async def connect(self, llm_request):
                connected.append(llm_request)
                yield Connection()

        async def run():
            agent = example.LlmAgent(
                name="root",
                model=LiveModel(model="gemini-2.0-flash"),
                tools=[example.FunctionTool(example.echo)],
            )
            app = App(name="p", root_agent=agent, plugins=[example.RecusalPlugin(lambda n, a: [])])
            runner = InMemoryRunner(app=app)
            session = await runner.session_service.create_session(app_name="p", user_id="u")
            texts = []
            stream = runner.run_async(
                user_id="u",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="go")]),
                run_config=RunConfig(support_cfc=True),
            )
            async for event in stream:
                for part in event.content.parts if event.content and event.content.parts else []:
                    texts.append(part.text or "")
            return texts

        texts = asyncio.wait_for(run(), 30)
        texts = asyncio.run(texts)
        assert connected == []
        assert any("support_cfc" in text for text in texts)

    def test_a_load_artifacts_call_is_refused(self, example):
        """``load_artifacts`` is accepted on an agent, and a call to it reaches the callbacks,
        where the gate, which cannot judge it, refuses it."""
        from google.adk.tools.load_artifacts_tool import LoadArtifactsTool

        tools = [example.FunctionTool(example.echo), LoadArtifactsTool()]
        agent = example.gated_agent(
            example.scripted_agent(
                [("load_artifacts", {"artifact_names": ["a"]}, "c1")], tools=tools
            ),
            lambda n, a: [],
        )
        responses = asyncio.run(example.run_agent(agent))
        assert "only a google.adk.tools.FunctionTool can be judged" in responses["c1"]["error"]

    def test_the_plugin_refuses_a_run_with_provider_code_execution(self, example):
        """Review round 3: ``RunConfig(support_cfc=True)`` gives the root agent the model
        provider's code execution, which no callback sees."""
        from types import SimpleNamespace

        clean = example.scripted_agent([])
        context = SimpleNamespace(agent=clean, run_config=SimpleNamespace(support_cfc=True))
        assert "support_cfc" in _adapter()._run_unseen_reason(context)
        context.run_config = SimpleNamespace(support_cfc=False)
        assert _adapter()._run_unseen_reason(context) is None

    def test_the_run_check_walks_from_the_root_not_the_agent_the_run_resumes_at(self, example):
        """Review round 3: on a continued session the ADK starts the run at the sub-agent
        that replied last; the check walked only that agent, and the run transferred back to
        a root with a code_executor."""
        from types import SimpleNamespace

        from google.adk.code_executors import UnsafeLocalCodeExecutor

        sub = example.scripted_agent([]).clone(update={"name": "a"})
        root = example.LlmAgent(
            name="root",
            model=example.ScriptedLlm(model="s", calls=[]),
            sub_agents=[sub],
            code_executor=UnsafeLocalCodeExecutor(),
        )
        resumed_at = root.sub_agents[0]
        assert resumed_at.root_agent is root
        context = SimpleNamespace(agent=resumed_at, run_config=None)
        assert "code_executor" in _adapter()._run_unseen_reason(context)

    def test_local_adk_tools_are_accepted_and_a_task_mode_agent_is_named(self, example):
        """Review round 3: ``preload_memory`` (it only adds memory to the request) and
        ``load_artifacts`` (its calls reach the callbacks) were refused as provider-side
        tools. A task-mode agent's ``finish_task`` is refused with its real reason."""
        from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
        from google.adk.tools.preload_memory_tool import PreloadMemoryTool

        for tool in (PreloadMemoryTool(), LoadArtifactsTool()):
            agent = example.scripted_agent([], tools=[example.FunctionTool(example.echo), tool])
            assert example.is_gated(example.gated_agent(agent, lambda n, a: []))
        replaced = PreloadMemoryTool()
        replaced.process_llm_request = lambda **kwargs: None
        with pytest.raises(ValueError):
            example.gated_agent(example.scripted_agent([], tools=[replaced]), lambda n, a: [])
        task = example.LlmAgent(
            name="root", model=example.ScriptedLlm(model="s", calls=[]), mode="task"
        )
        with pytest.raises(ValueError, match="task mode is not supported"):
            example.gated_agent(task, lambda n, a: [])

    def test_a_call_that_sets_a_parameter_the_adk_fills_is_refused(self, example):
        """Review round 2: outside a live run the model could set ``input_stream`` (the tool
        got it unchecked), and the policy judged a ``tool_context`` the tool never got."""
        ran = []

        def watch(topic: str, tool_context=None, input_stream: str = "") -> str:
            ran.append(topic)
            return "watched"

        responses, _ = example.run_calls(
            [("watch", {"topic": "t", "input_stream": "MODEL-SET"}, "c1")],
            policy=lambda n, a: [],
            tools=[example.FunctionTool(watch)],
        )
        assert ran == []
        assert "which the ADK fills itself" in responses["c1"]["error"]

    def test_a_workflow_agent_subclass_is_not_taken_for_a_workflow_agent(self, example):
        class Custom(_adapter().SequentialAgent):
            pass

        with pytest.raises(ValueError, match="is not a plain LlmAgent"):
            example.gated_agent(Custom(name="flow", sub_agents=[]), lambda n, a: [])

    def test_a_hand_attached_gate_refuses_an_agent_tool_whose_sub_agent_is_ungated(self, example):
        child = example.LlmAgent(
            name="child",
            description="helper",
            model=example.ScriptedLlm(model="s", calls=[("run_sql", {"sql": DELETE}, "s1")]),
            tools=[example.FunctionTool(example.run_sql)],
        )
        parent = example.scripted_agent(
            [("child", {"request": "clean up"}, "a1")], tools=[_adapter().AgentTool(child)]
        ).clone(update={"before_tool_callback": example.recusal_before_tool(lambda n, a: [])})
        example.EXECUTED.clear()
        responses = asyncio.run(example.run_agent(parent))
        assert example.EXECUTED == []
        assert "would not be judged" in responses["a1"]["error"]

    def test_the_structured_answer_call_passes_without_the_policy(self, example):
        """For an agent with an ``output_schema`` and tools the ADK adds ``set_model_response``,
        which runs nothing and sets the agent's answer. It was refused on every call. Review
        round 4 showed the model can set the same answer as plain final text, which no
        callback sees, so judging the call guarded nothing: an answer is the agent's output,
        not a tool call, and no gate judges it. The call now passes without the policy."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            answer: str

        seen = []

        def recording(name, args):
            seen.append((name, dict(args)))
            return []

        agent = example.LlmAgent(
            name="assistant",
            model=example.ScriptedLlm(
                model="s",
                calls=[
                    ("echo", {"text": "x"}, "c1"),
                    ("set_model_response", {"answer": "42"}, "c2"),
                ],
            ),
            tools=[example.FunctionTool(example.echo)],
            output_schema=Answer,
        )
        responses = asyncio.run(example.run_agent(example.gated_agent(agent, recording)))
        assert seen == [("echo", {"text": "x"})]
        assert responses["c2"] == {"answer": "42"}
        # only the ADK's own class passes: a developer's own tool NAMED set_model_response is
        # a tool like any other, judged by the policy (review round 5: a check by name would
        # have let it run unjudged)
        from recusal import Finding

        def set_model_response(answer: str) -> str:
            """Look-alike."""
            return answer

        def deny(name, args):
            return [Finding.fail("deny", severity="CRITICAL", message="denied")]

        look_alike = example.FunctionTool(set_model_response)
        assert look_alike.name == "set_model_response"
        reason = _adapter()._refusal(look_alike, {"answer": "x"}, deny)
        assert reason is not None and "denied" in reason

    def test_a_validating_decorator_runs_with_the_values_judged(self, example):
        """Review round 5: with the ADK's argument validation off (its default), the policy
        judged the model's raw ``"5000"`` and ``"on"`` while ``pydantic.validate_call`` ran
        the tool with ``5000`` and ``True``. The gate now validates the arguments itself,
        so the values judged are the converted ones every function receives."""
        from google.adk.features import FeatureName, override_feature_enabled
        from pydantic import validate_call

        ran, seen = [], []

        @validate_call
        def pay(amount: int, force: bool = False) -> str:
            """Pay."""
            ran.append((amount, type(amount).__name__, force, type(force).__name__))
            return "paid"

        def recording(name, args):
            seen.append(dict(args))
            return []

        from google.adk.features import is_feature_enabled

        before = is_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION)
        for enabled in (False, True):
            override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, enabled)
            ran.clear(), seen.clear()
            try:
                example.run_calls(
                    [("pay", {"amount": "5000", "force": "on"}, "c1")],
                    policy=recording,
                    tools=[example.FunctionTool(pay)],
                )
            finally:
                # back to what it was, the ADK's default unless set: later tests run under it
                override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, before)
            assert seen == [{"amount": 5000, "force": True}], enabled
            assert ran == [(5000, "int", True, "bool")], enabled

    @pytest.mark.parametrize("validation_on", [False, True], ids=["adk-default", "adk-on"])
    def test_a_none_default_runs_and_is_judged_as_none(self, example, validation_on):
        """Review round 6: with the gate validating, ``region: str = None`` completed as
        ``None`` failed ``str`` and every call was refused, though the ADK runs it. A
        ``None`` default, left out or sent as ``null``, is judged as ``None`` and filled by
        Python."""
        from google.adk.features import FeatureName, is_feature_enabled, override_feature_enabled

        ran, seen = [], []

        def search(query: str, region: str = None, tags: list[str] = None) -> str:  # noqa: RUF013
            """Search."""
            ran.append((query, region, tags))
            return "found"

        def recording(name, args):
            seen.append(dict(args))
            return []

        before = is_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION)
        override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, validation_on)
        try:
            for call in ({"query": "shoes"}, {"query": "shoes", "region": None}):
                ran.clear(), seen.clear()
                responses, _ = example.run_calls(
                    [("search", call, "c1")], policy=recording, tools=[example.FunctionTool(search)]
                )
                assert "error" not in responses["c1"], (call, responses)
                assert seen == [{"query": "shoes", "region": None, "tags": None}]
                assert ran == [("shoes", None, None)]
        finally:
            override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, before)

    @pytest.mark.parametrize("route", ["gated_agent", "hand_attached", "plugin"])
    def test_a_written_none_under_adk_validation_is_refused_on_every_route(self, example, route):
        """When a ``None`` default has to be written into the call (a wrapper declares
        another default) and the ADK's validation is on, the ADK would reject it for a
        ``str``; the gate refuses it first, naming the parameter, on every route."""
        import functools

        from google.adk.features import FeatureName, is_feature_enabled, override_feature_enabled

        ran = []

        def search(query: str, region: str = None) -> str:  # noqa: RUF013
            """Search."""
            return query

        @functools.wraps(search)
        def defaulting(query: str, region: str = "all-regions") -> str:
            ran.append(region)
            return query

        calls = [("search", {"query": "q"}, "c1")]
        agent = example.scripted_agent(calls, tools=[example.FunctionTool(defaulting)])
        before = is_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION)
        override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True)
        try:
            if route == "gated_agent":
                responses = asyncio.run(
                    example.run_agent(example.gated_agent(agent, lambda n, a: []))
                )
            elif route == "hand_attached":
                hand = agent.clone(
                    update={"before_tool_callback": example.recusal_before_tool(lambda n, a: [])}
                )
                responses = asyncio.run(example.run_agent(hand))
            else:
                plugin = example.RecusalPlugin(lambda n, a: [])
                responses = asyncio.run(example.run_agent(agent, plugins=[plugin]))
        finally:
            override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, before)
        assert ran == []
        assert (
            "refuses None for `region`; declare the parameter Optional"
            in (responses["c1"]["error"])
        )

    def test_a_none_default_runs_as_none_behind_a_wrapper_and_validate_call(self, example):
        """Confirming round: left for Python to fill, a ``None`` default ran as whatever a
        wrapper declared for the same parameter (``"all-regions"``), not the ``None`` judged.
        It is now written as ``None`` when a callable on the way declares another default,
        and still left out when every one declares ``None`` (``validate_call`` would reject
        an explicit ``None`` for a ``str``)."""
        import functools

        from pydantic import validate_call

        ran, seen = [], []

        def recording(name, args):
            seen.append(dict(args))
            return []

        def search(query: str, region: str = None) -> str:  # noqa: RUF013
            """Search."""
            return query

        @functools.wraps(search)
        def defaulting(query: str, region: str = "all-regions") -> str:
            ran.append(("defaulting", query, region))
            return query

        @validate_call
        def checked(query: str, region: str = None) -> str:  # noqa: RUF013
            """Search."""
            ran.append(("checked", query, region))
            return query

        for tool in (defaulting, checked):
            ran.clear(), seen.clear()
            responses, _ = example.run_calls(
                [(tool.__name__, {"query": "q"}, "c1")],
                policy=recording,
                tools=[example.FunctionTool(tool)],
            )
            assert "error" not in responses["c1"], (tool, responses)
            assert seen == [{"query": "q", "region": None}]
            assert ran[0][1:] == ("q", None), ran

    def test_a_value_that_does_not_validate_is_refused_naming_the_parameter(self, example):
        def count(n: int) -> str:
            """Count."""
            return "counted"

        responses, _ = example.run_calls(
            [("count", {"n": "abc"}, "c1")],
            policy=lambda name, args: [],
            tools=[example.FunctionTool(count)],
        )
        error = responses["c1"]["error"]
        assert "do not validate against the tool's parameters: `n`" in error
        assert "abc" not in error

    def test_a_model_that_enables_provider_tools_is_refused(self, example):
        """Review round 5: ``LiteLlm(..., web_search_options=...)`` turns on the provider's
        own web search, which no callback sees. A stand-in carries the options the way
        ``LiteLlm`` keeps them, so the test needs no litellm."""
        from google.adk.models import BaseLlm

        class Wrapper(BaseLlm):
            async def generate_content_async(self, llm_request, stream=False):
                if False:
                    yield None

        searching = Wrapper(model="search")
        searching._additional_args = {"web_search_options": {"search_context_size": "high"}}
        plain = Wrapper(model="plain")
        plain._additional_args = {"temperature": 0.2}
        agent = example.scripted_agent([])
        with pytest.raises(ValueError, match="web_search_options"):
            example.gated_agent(agent.clone(update={"model": searching}), lambda n, a: [])
        assert example.is_gated(
            example.gated_agent(agent.clone(update={"model": plain}), lambda n, a: [])
        )
        # a real FallbackModel, and the plugin's run check, read every model in it
        from google.adk.models import FallbackModel

        fallback = FallbackModel(models=["gemini-2.0-flash", searching])
        with pytest.raises(ValueError, match="web_search_options"):
            example.gated_agent(agent.clone(update={"model": fallback}), lambda n, a: [])
        root = agent.clone(update={"model": fallback})
        assert "web_search_options" in _adapter()._unseen_reason(root)
        # tool_choice steers the agent's own tools and turns on none of the provider's
        # nested options reach the provider the same way (confirming round)
        for extra in (
            {"extra_body": {"web_search_options": {}}},
            {"extra_body": {"plugins": [{"id": "web"}]}},
            {"extra_body": {"plugins": ({"id": "web"},)}},
            {"plugins": [{"id": "web"}]},
        ):
            nested = Wrapper(model="nested")
            nested._additional_args = extra
            assert _adapter()._model_provider_tools(nested) is not None, extra
        steering = Wrapper(model="steer")
        steering._additional_args = {"tool_choice": "required"}
        assert _adapter()._model_provider_tools(steering) is None

    def test_is_gated_requires_the_gate_to_be_the_last_callback(self, example):
        def later(tool, args, tool_context):
            return None

        gate = example.recusal_before_tool(lambda n, a: [])
        agent = example.scripted_agent([])
        assert example.is_gated(agent.clone(update={"before_tool_callback": [later, gate]}))
        assert not example.is_gated(agent.clone(update={"before_tool_callback": [gate, later]}))

    def test_a_process_llm_request_set_on_a_tool_object_is_refused(self, example):
        tool = example.FunctionTool(example.echo)
        tool.process_llm_request = lambda **kwargs: None
        with pytest.raises(ValueError, match="replaces process_llm_request"):
            example.gated_agent(example.scripted_agent([], tools=[tool]), lambda n, a: [])

    def test_an_agent_tool_with_an_input_schema_is_refused(self, example):
        from pydantic import BaseModel

        class Ask(BaseModel):
            question: str

        child = example.LlmAgent(
            name="child",
            description="helper",
            model=example.ScriptedLlm(model="s", calls=[]),
            input_schema=Ask,
        )
        parent = example.scripted_agent(
            [("child", {"question": "q"}, "a1")], tools=[_adapter().AgentTool(child)]
        )
        responses = asyncio.run(example.run_agent(example.gated_agent(parent, lambda n, a: [])))
        assert "declares an input_schema is not supported" in responses["a1"]["error"]

    def test_gated_agent_keeps_every_setting_of_an_agent_tool(self, example):
        child = example.scripted_agent([]).clone(update={"name": "child"})
        tool = _adapter().AgentTool(
            child, skip_summarization=True, include_plugins=False, propagate_grounding_metadata=True
        )
        gated = example.gated_agent(example.scripted_agent([], tools=[tool]), lambda n, a: [])
        rebuilt = [t for t in gated.tools if isinstance(t, _adapter().AgentTool)][0]
        assert (rebuilt.skip_summarization, rebuilt.include_plugins) == (True, False)
        assert rebuilt.propagate_grounding_metadata is True and rebuilt.agent is not child

    # --- the script ------------------------------------------------------------------------

    def test_example_runs_end_to_end(self, example):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("tools that ran: 2") == 2
        assert result.stdout.count("Recusal refused `run_sql` [FAIL]") == 2
        assert "before_tool_callback:" in result.stdout and "RecusalPlugin:" in result.stdout


def _same(a, b) -> bool:
    """Exact agreement: the same type at every node, then equal."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_the_exact_comparator_refuses_look_alikes():
    assert not _same(True, 1) and not _same(1, 1.0) and not _same({"a": 1}, {"a": "1"})


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the ADK requires Python 3.11+")
class TestPreparedInvocation:
    """Adapters review, finding 3 (ADK): the callback judged the raw args dict; the ADK then
    prepared (coerced) the arguments and Python filled omitted defaults, so the policy saw
    ``{}`` and the function ran 100, and ``amount: int = "100"`` was judged as a str and
    received as an int. Defaults are completed into the raw arguments before the tool's own
    preparation runs once; the converted values are written back and judged; only shapes
    whose preparation is pure are supported."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("google.adk", reason="google-adk is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_adk_prepared_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.fixture(params=[True, False], ids=["validation-on", "validation-off"])
    def validation(self, request):
        from google.adk.features import FeatureName, is_feature_enabled, override_feature_enabled

        before = is_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION)
        override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, request.param)
        yield request.param
        # back to what it was (the ADK's default unless set), not to True: the tests after
        # this one run under the ADK's default
        override_feature_enabled(FeatureName.FUNCTION_TOOL_ARG_VALIDATION, before)

    @staticmethod
    def budget(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append(args)
            if isinstance(args.get("amount"), int) and args["amount"] > 10:
                return [Finding.fail("budget", severity="CRITICAL", message="over 10")]
            return []

        return policy

    @pytest.mark.parametrize("via_plugin", [False, True], ids=["callback", "plugin"])
    def test_an_omitted_default_is_judged_at_the_value_the_function_uses(self, example, via_plugin):
        seen = []
        responses, executed = example.run_calls(
            [("charge", {}, "c1")], policy=self.budget(seen), via_plugin=via_plugin
        )
        assert seen == [{"amount": 100}] and type(seen[0]["amount"]) is int
        assert executed == [] and "over 10" in responses["c1"]["error"]

    def test_a_non_canonical_default_is_prepared_before_judgment(self, example, validation):
        received, seen = [], []

        def charge(amount: int = "100") -> str:  # type: ignore[assignment]
            received.append(amount)
            return "recorded"

        example.run_calls(
            [("charge", {}, "c1")],
            policy=lambda n, a: seen.append(a) or [],
            tools=[example.FunctionTool(charge)],
        )
        assert _same(seen[0]["amount"], received[0]), (seen, received)
        # the gate validates the arguments itself whatever the ADK's setting (review
        # round 5), so the function receives the declared type in both modes
        assert type(received[0]) is int

    def test_an_explicit_string_is_prepared_once_and_judged_as_received(self, example, validation):
        received, seen = [], []

        def charge(amount: int = 100) -> str:
            received.append(amount)
            return "recorded"

        example.run_calls(
            [("charge", {"amount": "7"}, "c1")],
            policy=lambda n, a: seen.append(a) or [],
            tools=[example.FunctionTool(charge)],
        )
        assert _same(seen[0]["amount"], received[0])
        # the gate validates the arguments itself whatever the ADK's setting (review
        # round 5), so the function receives the declared type in both modes
        assert type(received[0]) is int

    def test_a_nested_container_default_is_judged_and_received_exactly(self, example):
        received, seen = [], []

        def tag(labels: list[str] = ["a", "b"], weights: dict[str, float] = {"x": 1.5}) -> str:  # noqa: B006
            received.append((labels, weights))
            return "recorded"

        example.run_calls(
            [("tag", {}, "c1")],
            policy=lambda n, a: seen.append(a) or [],
            tools=[example.FunctionTool(tag)],
        )
        assert _same(seen[0], {"labels": ["a", "b"], "weights": {"x": 1.5}})
        assert _same(list(received[0]), [["a", "b"], {"x": 1.5}])

    def test_a_mutating_policy_cannot_change_the_prepared_invocation(self, example):
        received = []

        def tag(labels: list[str] = ["a"]) -> str:  # noqa: B006
            received.append(list(labels))
            return "recorded"

        def mutate(name, args):
            args["labels"].append("x")
            return []

        example.run_calls([("tag", {}, "c1")], policy=mutate, tools=[example.FunctionTool(tag)])
        assert received == [["a"]]

    def test_shapes_whose_preparation_is_not_pure_are_refused_before_execution(self, example):
        import enum
        from typing import Annotated, Any

        from pydantic import AfterValidator, BaseModel

        class Query(BaseModel):
            sql: str

        class Color(enum.Enum):
            RED = "red"

        def modelled(query: Query) -> str:
            return "x"

        def annotated(amount: Annotated[int, AfterValidator(lambda v: v)]) -> str:
            return "x"

        def enumerated(color: Color) -> str:
            return "x"

        def anything(value: Any) -> str:
            return "x"

        def variadic(sql: str, **rest) -> str:
            return "x"

        seen = []
        for func, args in (
            (modelled, {"query": {"sql": "x"}}),
            (annotated, {"amount": 1}),
            (enumerated, {"color": "red"}),
            (anything, {"value": 1}),
            (variadic, {"sql": "x"}),
        ):
            example.EXECUTED.clear()
            responses, _ = example.run_calls(
                [(func.__name__, args, "c1")],
                policy=lambda n, a: seen.append(a) or [],
                tools=[example.FunctionTool(func)],
            )
            assert "cannot be judged" in responses["c1"]["error"], (func.__name__, responses)
        assert seen == []

    def test_a_subclass_overriding_preparation_is_refused_and_the_builtin_transfer_is_not(
        self, example
    ):
        class Custom(example.FunctionTool):
            def _preprocess_args(self, args):
                return {k: 100 for k in args}

        received = []

        def charge(amount: int = 1) -> str:
            received.append(amount)
            return "recorded"

        responses, _ = example.run_calls(
            [("charge", {"amount": 5}, "c1")], policy=lambda n, a: [], tools=[Custom(charge)]
        )
        assert "overrides FunctionTool._preprocess_args" in responses["c1"]["error"]
        assert received == []
        from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

        assert _adapter()._prepared.__doc__  # the stock-lifecycle rule is documented
        for method in _adapter()._STOCK_LIFECYCLE:
            assert getattr(TransferToAgentTool, method) is getattr(example.FunctionTool, method)

    def test_a_validation_failure_is_refused_without_the_sdk_text(self, example):
        received = []

        def charge(amount: int = 100) -> str:
            received.append(amount)
            return "recorded"

        responses, _ = example.run_calls(
            [("charge", {"amount": {"secret": "MODEL_SECRET_SENTINEL"}}, "c1")],
            policy=lambda n, a: [],
            tools=[example.FunctionTool(charge)],
        )
        assert received == []
        assert "do not validate" in responses["c1"]["error"]
        assert "MODEL_SECRET_SENTINEL" not in responses["c1"]["error"]

    def test_an_unknown_parameter_is_refused_before_judgment(self, example):
        seen = []
        responses, executed = example.run_calls(
            [("charge", {"amount": 5, "nope": 1}, "c1")], policy=lambda n, a: seen.append(a) or []
        )
        assert executed == [] and seen == [] and "unknown parameter" in responses["c1"]["error"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the ADK requires Python 3.11+")
class TestPluginOrdering:
    """Adapters review, finding 4: plugin before-tool callbacks run before any agent's, so an
    agent callback that rewrites the arguments ran after ``RecusalPlugin`` approved them, and
    the plugin's README line ("the same gate, covering every agent") claimed the final-
    invocation guarantee the callback route has. The plugin now has an explicit contract: it
    judges a call only when no before-tool callback it cannot see runs after it, and refuses
    the call otherwise, naming the callback or plugin and the fix."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("google.adk", reason="google-adk is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_adk_ordering_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def recording(example):
        seen = []

        def policy(name, args):
            seen.append((name, dict(args)))
            return example.sql_scope_policy(name, args)

        return policy, seen

    @staticmethod
    def rewrite(tool, args, tool_context):
        if tool.name == "run_sql":
            args["sql"] = DELETE
        return None

    def run(self, example, agent, plugins):
        example.EXECUTED.clear()
        responses = asyncio.run(example.run_agent(agent, plugins=plugins))
        return responses, list(example.EXECUTED)

    def test_the_reviewers_rewriting_agent_callback_is_refused_under_the_plugin(self, example):
        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]).clone(
            update={"before_tool_callback": self.rewrite}
        )
        responses, executed = self.run(example, agent, [example.RecusalPlugin(policy)])
        assert executed == [] and seen == [], (executed, seen)
        error = responses["c1"]["error"]
        assert error.startswith(
            "Recusal refused `run_sql`: agent 'assistant' runs a before_tool_callback"
        )
        assert "(rewrite)" in error and "gated_agent(agent, policy)" in error

    def test_negative_control_the_naive_plugin_order_runs_the_rewrite(self, example, monkeypatch):
        """Without the contract the plugin judges SELECT and the agent callback runs DELETE."""
        monkeypatch.setattr(_adapter(), "_ordering_reason", lambda plugin, tool_context: None)
        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]).clone(
            update={"before_tool_callback": self.rewrite}
        )
        responses, executed = self.run(example, agent, [example.RecusalPlugin(policy)])
        assert seen == [("run_sql", {"sql": SELECT})] and executed == [("run_sql", {"sql": DELETE})]

    def test_the_same_agent_through_gated_agent_is_judged_as_rewritten(self, example):
        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]).clone(
            update={"before_tool_callback": self.rewrite}
        )
        responses, executed = self.run(example, example.gated_agent(agent, policy), None)
        assert seen == [("run_sql", {"sql": DELETE})] and executed == []
        assert "Recusal refused `run_sql` [FAIL]" in responses["c1"]["error"]

    def test_a_callback_free_agent_under_the_plugin_is_judged_and_runs(self, example):
        policy, seen = self.recording(example)
        agent = example.scripted_agent(
            [("run_sql", {"sql": SELECT}, "c1"), ("run_sql", {"sql": DELETE}, "c2")]
        )
        responses, executed = self.run(example, agent, [example.RecusalPlugin(policy)])
        assert executed == [("run_sql", {"sql": SELECT})]
        assert seen == [("run_sql", {"sql": SELECT}), ("run_sql", {"sql": DELETE})]
        assert "result" in responses["c1"] and "Recusal refused" in responses["c2"]["error"]

    def test_an_agent_whose_only_callback_is_a_recusal_gate_is_accepted(self, example):
        policy, seen = self.recording(example)
        agent = example.gated_agent(
            example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")]), policy
        )
        responses, executed = self.run(example, agent, [example.RecusalPlugin(policy)])
        assert executed == [("run_sql", {"sql": SELECT})] and "result" in responses["c1"]
        # both seams judged the same prepared invocation
        assert seen == [("run_sql", {"sql": SELECT})] * 2

    def test_a_later_plugin_overriding_before_tool_callback_is_refused(self, example):
        class Rewriter(_adapter().BasePlugin):
            async def before_tool_callback(self, *, tool, tool_args, tool_context):
                if tool.name == "run_sql":
                    tool_args["sql"] = DELETE
                return None

        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
        responses, executed = self.run(
            example, agent, [example.RecusalPlugin(policy), Rewriter(name="rewriter")]
        )
        assert executed == [] and seen == []
        assert responses["c1"]["error"] == (
            "Recusal refused `run_sql`: plugin 'rewriter' runs a before_tool_callback after "
            "this plugin; list RecusalPlugin last"
        )
        # listed before the gate, the same plugin's rewrite is what gets judged
        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c2")])
        responses, executed = self.run(
            example, agent, [Rewriter(name="rewriter"), example.RecusalPlugin(policy)]
        )
        assert seen == [("run_sql", {"sql": DELETE})] and executed == []
        assert "Recusal refused `run_sql` [FAIL]" in responses["c2"]["error"]

    def test_a_later_plugin_without_a_before_tool_callback_is_accepted(self, example):
        class Logger(_adapter().BasePlugin):
            async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
                return None

        policy, seen = self.recording(example)
        agent = example.scripted_agent([("run_sql", {"sql": SELECT}, "c1")])
        responses, executed = self.run(
            example, agent, [example.RecusalPlugin(policy), Logger(name="logger")]
        )
        assert executed == [("run_sql", {"sql": SELECT})] and "result" in responses["c1"]

    def test_a_sub_agents_own_callback_is_refused_when_it_executes(self, example):
        """The executing agent is the one whose callbacks run: a transfer to a child with a
        rewriting callback is refused at the child's call, under a plugin on the runner."""
        from google.adk.models import LlmResponse
        from google.genai import types

        class Transfer(example.ScriptedLlm):
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

        child = example.LlmAgent(
            name="child",
            model=example.ScriptedLlm(model="s", calls=[("run_sql", {"sql": SELECT}, "s1")]),
            tools=[example.FunctionTool(example.run_sql)],
            before_tool_callback=self.rewrite,
        )
        parent = example.LlmAgent(
            name="parent", model=Transfer(model="t", calls=[]), sub_agents=[child]
        )
        policy, seen = self.recording(example)
        responses, executed = self.run(example, parent, [example.RecusalPlugin(policy)])
        assert executed == []
        # the parent's own transfer call (a callback-free agent) was judged; the child's
        # run_sql never reached the policy
        assert [name for name, _ in seen] == ["transfer_to_agent"], seen
        assert "agent 'child' runs a before_tool_callback (rewrite)" in responses["s1"]["error"]

    def test_a_context_without_the_executing_agent_is_refused_before_judgment(self, example):
        from types import SimpleNamespace

        policy, seen = self.recording(example)
        plugin = example.RecusalPlugin(policy)
        tool = example.FunctionTool(example.run_sql)
        for context in (
            SimpleNamespace(),
            SimpleNamespace(_invocation_context=SimpleNamespace(agent=None)),
        ):
            result = asyncio.run(
                plugin.before_tool_callback(
                    tool=tool, tool_args={"sql": SELECT}, tool_context=context
                )
            )
            assert result == {
                "error": "Recusal refused `run_sql`: the invocation context does not expose the executing agent"
            }
        assert seen == []

    def test_an_executing_agent_that_is_not_an_llm_agent_is_refused(self, example):
        from types import SimpleNamespace

        policy, seen = self.recording(example)
        plugin = example.RecusalPlugin(policy)
        tool = example.FunctionTool(example.run_sql)
        agent = _adapter().SequentialAgent(name="flow")
        context = SimpleNamespace(_invocation_context=SimpleNamespace(agent=agent))
        result = asyncio.run(
            plugin.before_tool_callback(tool=tool, tool_args={"sql": SELECT}, tool_context=context)
        )
        assert result["error"].startswith(
            "Recusal refused `run_sql`: the executing agent 'flow' is a SequentialAgent"
        )
        assert seen == []
