"""Regression tests for the offline LangGraph + Recusal ToolNode example."""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "langgraph_gate.py"


def _adapter():
    """The package module the gate's code lives in. Every test that calls this has loaded
    the example first, which puts the in-tree adapter on the path, so this is the code
    under review and not an installed copy."""
    import importlib
    from pathlib import Path

    module = importlib.import_module("recusal_langgraph._gate")
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
    assert "langgraph_gate.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_langgraph_stays_out_of_the_recusal_package_and_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "langgraph" not in pyproject
    assert "langchain-core" not in pyproject
    assert "langchain" not in pyproject
    for path in (ROOT / "recusal").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "langgraph" not in source, path
        assert "langchain" not in source, path


def test_pinned_versions_are_explicit_in_example_docs_and_ci():
    expected = ("langgraph==1.2.20", "langchain-core==1.7.7")
    for path in (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        text = path.read_text(encoding="utf-8")
        for pin in expected:
            assert pin in text
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # both are roots of the two hash-locked files the pull request lanes install: the
    # recusal-langgraph wheel job installs its own, and the two example jobs and the
    # recusal-langchain wheel job install the one for the gate that wraps it
    for lock in ("adapter-langgraph", "adapter-langchain"):
        roots = (ROOT / "requirements" / f"{lock}.txt").read_text(encoding="utf-8").splitlines()[0]
        for pin in expected:
            assert pin in roots.split(), (lock, pin)
    assert ci.count("requirements/adapter-langgraph.txt") == 1
    assert ci.count("requirements/adapter-langchain.txt") == 3


def test_the_package_requires_the_kernel_that_walks_every_callable():
    """The gate's wrapper check is the kernel's walk (``recusal.invocation``): installed
    with an older kernel, recusal-langgraph would not import. The floor is the first kernel
    with ``reached_parameters``, stated in the package and in its README, and the kernel in
    this tree is at or past it."""
    import recusal
    from recusal import invocation

    floor = "0.15.0"
    pyproject = (ROOT / "adapters" / "langgraph" / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "adapters" / "langgraph" / "README.md").read_text(encoding="utf-8")
    assert f'"recusal>={floor},<1"' in pyproject
    assert f"`recusal>={floor}`" in readme
    assert hasattr(invocation, "reached_parameters") and hasattr(invocation, "DeclaredSignature")

    def parts(version):
        return tuple(int(part) for part in version.split("."))

    assert parts(recusal.__version__) >= parts(floor)


def test_the_package_declares_and_bounds_every_distribution_its_gate_imports():
    """The gate imports ``langgraph.prebuilt`` (the ``langgraph-prebuilt`` distribution,
    where the private helper it reads lives) and ``pydantic``. Both are declared, with an
    upper bound, at the versions the pinned lane installs."""
    pyproject = (ROOT / "adapters" / "langgraph" / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "adapters" / "langgraph" / "README.md").read_text(encoding="utf-8")
    lock = (ROOT / "requirements" / "adapter-langgraph.txt").read_text(encoding="utf-8")
    for name, pinned, ceiling in (
        ("langgraph", "1.2.20", "2"),
        ("langchain-core", "1.7.7", "2"),
        ("langgraph-prebuilt", "1.1.1", "2"),
        ("pydantic", "2.14.3", "3"),
    ):
        assert f'"{name}>={pinned},<{ceiling}"' in pyproject, name
        assert f"`{name}>={pinned}`" in readme, name
        assert f"\n{name}=={pinned} \\\n" in lock, f"{name} is not in the hash-locked file"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestLangGraphGate:
    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_langgraph_gate_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def call(name, args, ident="1"):
        return {"name": name, "args": args, "id": ident, "type": "tool_call"}

    def test_allowed_call_runs_its_tool(self, example):
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": "SELECT * FROM audit_log"})]
        )
        assert executed == [("run_sql", {"sql": "SELECT * FROM audit_log"})]
        assert messages[0].status == "success"

    def test_refused_call_does_not_run_and_reason_reaches_agent(self, example):
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": "DELETE FROM audit_log"})]
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "Recusal refused" in str(messages[0].content)
        assert "top-level WHERE" in str(messages[0].content)

    def test_several_calls_are_gated_separately(self, example):
        calls = [
            self.call("echo", {"text": "hello"}, "1"),
            self.call("run_sql", {"sql": "DELETE FROM audit_log"}, "2"),
            self.call("run_sql", {"sql": "SELECT * FROM audit_log"}, "3"),
        ]
        messages, executed = example.run_calls(calls)
        assert [m.status for m in messages] == ["success", "error", "success"]
        assert len(executed) == 2

    def test_gate_is_plug_and_play_for_an_unseen_tool_list(self, example):
        @example.tool
        def query(sql: str) -> str:
            """A policy-covered tool unknown to the example's built-in list."""
            raise AssertionError("refused tool must not run")

        node = example.ToolNode(
            [query], wrap_tool_call=example.recusal_gate(example.sql_scope_policy)
        )
        messages = example.run_in_graph(node, [self.call("query", {"sql": "DELETE FROM x"}, "x")])
        assert messages[0].status == "error"

    def test_policy_exception_fails_closed(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        messages, executed = example.run_calls(
            [self.call("echo", {"text": "unsafe"})], policy=broken_policy
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)" in str(
            messages[0].content
        )

    def test_negative_control_runs_delete_without_hook(self, example):
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": "DELETE FROM audit_log"})], gated=False
        )
        assert executed == [("run_sql", {"sql": "DELETE FROM audit_log"})]
        assert messages[0].status == "success"

    def test_async_allowed_call_runs_its_tool(self, example):
        messages, executed = asyncio.run(
            example.run_calls_async([self.call("run_sql", {"sql": "SELECT * FROM audit_log"})])
        )
        assert executed == [("run_sql", {"sql": "SELECT * FROM audit_log"})]
        assert messages[0].status == "success"

    def test_async_refusal_reaches_agent_without_running_tool(self, example):
        messages, executed = asyncio.run(
            example.run_calls_async([self.call("run_sql", {"sql": "DELETE FROM audit_log"})])
        )
        assert executed == []
        assert messages[0].status == "error"
        assert messages[0].tool_call_id == "1"
        assert messages[0].name == "run_sql"
        assert "Recusal refused" in str(messages[0].content)
        assert "top-level WHERE" in str(messages[0].content)

    def test_async_several_calls_are_gated_separately(self, example):
        messages, executed = asyncio.run(
            example.run_calls_async(
                [
                    self.call("echo", {"text": "hello"}, "1"),
                    self.call("run_sql", {"sql": "DELETE FROM audit_log"}, "2"),
                    self.call("run_sql", {"sql": "SELECT * FROM audit_log"}, "3"),
                ]
            )
        )
        assert [(m.tool_call_id, m.status) for m in messages] == [
            ("1", "success"),
            ("2", "error"),
            ("3", "success"),
        ]
        assert sorted(name for name, _ in executed) == ["echo", "run_sql"]
        assert ("run_sql", {"sql": "DELETE FROM audit_log"}) not in executed

    def test_async_gate_awaits_an_unseen_async_tool(self, example):
        executed = []

        @example.tool
        async def query(sql: str) -> str:
            """Record an async SQL call without a database."""
            await asyncio.sleep(0)
            executed.append(sql)
            return sql

        node = example.ToolNode([query], **example.recusal_gates(example.sql_scope_policy))
        messages = asyncio.run(
            example.run_in_graph_async(
                node,
                [
                    self.call("query", {"sql": "SELECT * FROM x"}, "allowed"),
                    self.call("query", {"sql": "DELETE FROM x"}, "refused"),
                ],
            )
        )
        assert executed == ["SELECT * FROM x"]
        assert [m.status for m in messages] == ["success", "error"]

    def test_async_policy_exception_fails_closed(self, example):
        def broken_policy(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        messages, executed = asyncio.run(
            example.run_calls_async([self.call("echo", {"text": "unsafe"})], policy=broken_policy)
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)" in str(
            messages[0].content
        )

    def test_async_negative_control_runs_delete_without_hook(self, example):
        messages, executed = asyncio.run(
            example.run_calls_async(
                [self.call("run_sql", {"sql": "DELETE FROM audit_log"})], gated=False
            )
        )
        assert executed == [("run_sql", {"sql": "DELETE FROM audit_log"})]
        assert messages[0].status == "success"


SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestEffectiveArguments:
    """Review 5, P1: the gate judged ``tool_call["args"]`` as the model wrote them; the tool
    then validated them through its ``args_schema``, and a field validator, a default or a
    coercion could make the function receive something else. The gate now judges the
    validated invocation and executes exactly that, once."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "recusal_langgraph_effective_example", EXAMPLE
        )
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

    @staticmethod
    def call(name, args, ident="1"):
        return {"name": name, "args": args, "id": ident, "type": "tool_call"}

    def test_a_schema_that_rewrites_an_allowed_proposal_is_judged_on_the_rewritten_value(
        self, example
    ):
        policy = self.Recording(example.sql_scope_policy)
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})],
            policy=policy,
            tools=[example.transforming_run_sql()],
        )
        assert policy.seen == [("run_sql", {"sql": DELETE})]  # what the tool would receive
        assert executed == []
        assert messages[0].status == "error" and "Recusal refused `run_sql`" in messages[0].content

    def test_negative_control_the_rewriting_tool_runs_the_rewritten_sql_ungated(self, example):
        _, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})],
            gated=False,
            tools=[example.transforming_run_sql()],
        )
        assert executed == [("run_sql", {"sql": DELETE})]

    def test_the_async_gate_judges_the_rewritten_value_too(self, example):
        policy = self.Recording(example.sql_scope_policy)
        _, executed = asyncio.run(
            example.run_calls_async(
                [self.call("run_sql", {"sql": SELECT})],
                policy=policy,
                tools=[example.transforming_run_sql()],
            )
        )
        assert policy.seen == [("run_sql", {"sql": DELETE})] and executed == []

    def test_defaults_and_coercion_are_judged_as_the_tool_receives_them(self, example):
        from langchain_core.tools import tool
        from pydantic import BaseModel

        from recusal import Finding

        class Args(BaseModel):
            sql: str
            limit: int = 10

        ran = []

        @tool("query", args_schema=Args)
        def query(sql: str, limit: int = 10) -> str:
            """Record a bounded query."""
            ran.append({"sql": sql, "limit": limit, "type": type(limit).__name__})
            return "recorded"

        def cap(name, args):
            if args.get("limit", 0) > 50:
                return [Finding.fail("limit", severity="CRITICAL", message="over 50")]
            return []

        policy = self.Recording(cap)
        example.run_calls([self.call("query", {"sql": SELECT})], policy=policy, tools=[query])
        assert policy.seen[-1] == ("query", {"sql": SELECT, "limit": 10})
        assert ran[-1] == {"sql": SELECT, "limit": 10, "type": "int"}
        example.run_calls(
            [self.call("query", {"sql": SELECT, "limit": "7"})], policy=policy, tools=[query]
        )
        assert policy.seen[-1] == ("query", {"sql": SELECT, "limit": 7})
        assert ran[-1]["limit"] == 7 and ran[-1]["type"] == "int"
        ran.clear()
        messages, _ = example.run_calls(
            [self.call("query", {"sql": SELECT, "limit": "99"})], policy=policy, tools=[query]
        )
        assert policy.seen[-1] == ("query", {"sql": SELECT, "limit": 99}) and ran == []
        assert "Recusal refused `query`" in messages[0].content

    def test_arguments_the_tool_rejects_are_refused_with_the_reason(self, example):
        policy = self.Recording(example.sql_scope_policy)
        messages, executed = example.run_calls([self.call("run_sql", {"sql": 5})], policy=policy)
        assert executed == [] and policy.seen == []
        assert "do not validate" in messages[0].content

    def test_a_tool_the_node_does_not_know_is_refused_not_errored(self, example):
        messages, executed = example.run_calls([self.call("nope", {"x": 1})])
        assert executed == [] and "no tool by that name" in messages[0].content

    def test_the_judging_copy_runs_only_what_it_judged(self, example):
        call = {"name": "run_sql", "args": {"sql": SELECT}, "id": "c1", "type": "tool_call"}
        copy = _adapter()._judging(example.run_sql, call, example.sql_scope_policy, {}, False)
        assert copy._to_args_and_kwargs({"sql": SELECT}, "c1") == ((), {"sql": SELECT})
        with pytest.raises(example.RecusalRefusal, match="Recusal refused `run_sql`"):
            copy._to_args_and_kwargs({"sql": DELETE}, "c1")
        with pytest.raises(RuntimeError, match="different call"):
            copy._to_args_and_kwargs({"sql": SELECT}, "c2")
        # the original tool is untouched
        assert example.run_sql._to_args_and_kwargs({"sql": "x"}, "c9") == ((), {"sql": "x"})
        assert example.run_sql.handle_tool_error is False

    def test_unjudgeable_inputs_are_refused_through_decide_and_reach_the_audit_log(
        self, example, tmp_path
    ):
        import json

        from recusal.audit import AuditLog

        log = AuditLog(path=str(tmp_path / "audit.jsonl"))
        call = {"name": "run_sql", "args": {"sql": 5}, "id": "c1", "type": "tool_call"}
        copy = _adapter()._judging(
            example.run_sql, call, example.sql_scope_policy, {"audit": log, "surface": "t"}, False
        )
        with pytest.raises(example.RecusalRefusal) as refused:
            copy._to_args_and_kwargs({"sql": 5}, "c1")
        assert str(refused.value).startswith("Recusal refused `run_sql`")
        assert "do not validate" in str(
            refused.value
        ) and "sql: Input should be a valid string" in str(refused.value)
        entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert entry["action"]["decision"] == "deny" and entry["action"]["tool"] == "run_sql"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestFinalInvocation:
    """Review 6, P1 and P2: the gate validated once in the hook (before ToolNode injected the
    host's state), judged a ``to_jsonable_python`` projection of the result, then validated
    again inside a pinned copy and compared projections. A nested model's serializer showed
    the policy SELECT while the field held DELETE; a stateful validator returned ``1`` then
    ``True`` and ``1 == True`` passed the comparison; a tool with a required
    ``InjectedState`` failed validation before the state existed. The gate now judges once,
    inside the tool's own parse step after injection, through a serializer-free projection,
    and executes the very objects it parsed."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_langgraph_final_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def call(name, args, ident="c1"):
        return {"name": name, "args": args, "id": ident, "type": "tool_call"}

    @staticmethod
    def serializer_tool(ran):
        """The review's probe: a nested model whose serializer displays SELECT over DELETE."""
        from langchain_core.tools import tool
        from pydantic import BaseModel, field_serializer

        class Query(BaseModel):
            sql: str

            @field_serializer("sql")
            def public_sql(self, value):
                return SELECT

        class Args(BaseModel):
            query: Query

        @tool("query", args_schema=Args)
        def query(query: Query) -> str:
            """Record SQL only."""
            ran.append({"sql": query.sql, "type": type(query).__name__})
            return "recorded"

        return query

    @staticmethod
    def select_only(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append(args)
            if args["query"]["sql"] == SELECT:
                return []
            return [Finding.fail("sql", severity="CRITICAL", message="not the approved SELECT")]

        return policy

    # --- the serializer probe, both routes --------------------------------------------------

    def test_a_display_serializer_cannot_hide_the_value_the_tool_receives(self, example):
        seen, ran = [], []
        messages, _ = example.run_calls(
            [self.call("query", {"query": {"sql": DELETE}})],
            policy=self.select_only(seen),
            tools=[self.serializer_tool(ran)],
        )
        assert seen == [{"query": {"sql": DELETE}}], "the policy sees the field, not the serializer"
        assert ran == []
        assert messages[0].status == "error" and "not the approved SELECT" in messages[0].content

    def test_the_async_gate_sees_through_the_serializer_too(self, example):
        seen, ran = [], []
        messages, _ = asyncio.run(
            example.run_calls_async(
                [self.call("query", {"query": {"sql": DELETE}})],
                policy=self.select_only(seen),
                tools=[self.serializer_tool(ran)],
            )
        )
        assert seen == [{"query": {"sql": DELETE}}] and ran == []
        assert messages[0].status == "error"

    def test_negative_control_ungated_the_serializer_tool_runs_the_delete(self, example):
        ran = []
        example.run_calls(
            [self.call("query", {"query": {"sql": DELETE}})],
            gated=False,
            tools=[self.serializer_tool(ran)],
        )
        assert ran == [{"sql": DELETE, "type": "Query"}]

    def test_an_approved_nested_value_runs_as_the_model_object(self, example):
        seen, ran = [], []
        messages, _ = example.run_calls(
            [self.call("query", {"query": {"sql": SELECT}})],
            policy=self.select_only(seen),
            tools=[self.serializer_tool(ran)],
        )
        assert messages[0].status == "success"
        assert ran == [{"sql": SELECT, "type": "Query"}], "the tool still receives its model"

    # --- one validation: the policy's object is the tool's object --------------------------

    @staticmethod
    def stateful_amount(counter, second):
        from langchain_core.tools import tool
        from pydantic import BaseModel, field_validator

        class Args(BaseModel):
            amount: Any

            @field_validator("amount")
            @classmethod
            def convert(cls, value):
                counter["n"] += 1
                return 1 if counter["n"] == 1 else second

        ran = []

        @tool("charge", args_schema=Args)
        def charge(amount: Any) -> str:
            """Record a typed value."""
            ran.append((amount, type(amount).__name__))
            return "recorded"

        return charge, ran

    @staticmethod
    def int_only(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append((args["amount"], type(args["amount"]).__name__))
            if type(args["amount"]) is int:
                return []
            return [Finding.fail("type", severity="CRITICAL", message="not an int")]

        return policy

    def test_a_stateful_validator_runs_once_so_the_tool_gets_the_judged_value(self, example):
        counter, seen = {"n": 0}, []
        charge, ran = self.stateful_amount(counter, second=True)
        messages, _ = example.run_calls(
            [self.call("charge", {"amount": 1})], policy=self.int_only(seen), tools=[charge]
        )
        assert counter["n"] == 1
        assert seen == [(1, "int")] and ran == [(1, "int")]
        assert messages[0].status == "success"

    def test_a_validator_that_yields_a_bool_on_its_one_run_is_judged_as_a_bool(self, example):
        from langchain_core.tools import tool
        from pydantic import BaseModel, field_validator

        class Args(BaseModel):
            amount: Any

            @field_validator("amount")
            @classmethod
            def convert(cls, value):
                return True

        ran = []

        @tool("charge", args_schema=Args)
        def charge(amount: Any) -> str:
            """Record a typed value."""
            ran.append(amount)
            return "recorded"

        seen = []
        messages, _ = example.run_calls(
            [self.call("charge", {"amount": 1})], policy=self.int_only(seen), tools=[charge]
        )
        assert seen == [(True, "bool")] and ran == []
        assert "not an int" in messages[0].content

    # --- host injection: present at the seam, excluded from judgment, never the model's ----

    @staticmethod
    def state_tool(ran):
        from typing import Annotated

        from langchain_core.tools import tool
        from langgraph.prebuilt import InjectedState

        @tool
        def with_state(sql: str, state: Annotated[dict, InjectedState]) -> str:
            """Record SQL with the host's state."""
            ran.append({"sql": sql, "state_keys": sorted(state)})
            return "recorded"

        return with_state

    @staticmethod
    def recording(seen, inner):
        def policy(name, args):
            seen.append(dict(args))
            return inner(name, args)

        return policy

    def test_a_required_injected_state_runs_gated_and_the_policy_sees_only_the_model(self, example):
        ran, seen = [], []
        messages, _ = example.run_calls(
            [self.call("with_state", {"sql": SELECT})],
            policy=self.recording(seen, example.sql_scope_policy),
            tools=[self.state_tool(ran)],
        )
        assert messages[0].status == "success", messages[0].content
        assert ran == [{"sql": SELECT, "state_keys": ["messages"]}]
        assert seen == [{"sql": SELECT}], "host state is not the model's proposal"

    def test_the_async_gate_passes_injected_state_too(self, example):
        ran, seen = [], []
        messages, _ = asyncio.run(
            example.run_calls_async(
                [self.call("with_state", {"sql": SELECT})],
                policy=self.recording(seen, example.sql_scope_policy),
                tools=[self.state_tool(ran)],
            )
        )
        assert messages[0].status == "success" and seen == [{"sql": SELECT}]
        assert ran == [{"sql": SELECT, "state_keys": ["messages"]}]

    def test_a_forged_state_in_the_proposal_is_replaced_by_the_hosts(self, example):
        ran, seen = [], []
        example.run_calls(
            [self.call("with_state", {"sql": SELECT, "state": {"forged": True}})],
            policy=self.recording(seen, example.sql_scope_policy),
            tools=[self.state_tool(ran)],
        )
        assert ran == [{"sql": SELECT, "state_keys": ["messages"]}]
        assert seen == [{"sql": SELECT}]

    def test_the_tool_runtime_is_injected_and_a_generic_injected_arg_is_never_the_models(
        self, example
    ):
        from typing import Annotated

        from langchain_core.tools import InjectedToolArg, tool
        from langgraph.prebuilt import ToolRuntime

        ran, seen = [], []

        @tool
        def with_runtime(
            sql: str, runtime: ToolRuntime, note: Annotated[str, InjectedToolArg] = "host"
        ) -> str:
            """Record SQL with the runtime and a host-only note."""
            ran.append({"sql": sql, "runtime": type(runtime).__name__, "note": note})
            return "recorded"

        messages, _ = example.run_calls(
            [self.call("with_runtime", {"sql": SELECT, "note": "forged", "runtime": "x"})],
            policy=self.recording(seen, example.sql_scope_policy),
            tools=[with_runtime],
        )
        assert messages[0].status == "success", messages[0].content
        assert ran == [{"sql": SELECT, "runtime": "ToolRuntime", "note": "host"}]
        # the model's "forged" is stripped and nothing replaces it, so the function runs
        # with its own default: a value no host supplied, which the policy is shown
        assert seen == [{"sql": SELECT, "note": "host"}], "the model cannot fill a host argument"

    def test_the_store_is_injected_when_the_graph_has_one(self, example):
        from typing import Annotated

        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.tools import tool
        from langgraph.graph import START, MessagesState, StateGraph
        from langgraph.prebuilt import InjectedStore, ToolNode, tools_condition
        from langgraph.store.base import BaseStore
        from langgraph.store.memory import InMemoryStore

        ran, seen = [], []

        @tool
        def with_store(sql: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
            """Record SQL with the host's store."""
            ran.append({"sql": sql, "store": type(store).__name__})
            return "recorded"

        calls = [self.call("with_store", {"sql": SELECT, "store": "forged"})]
        script = iter([AIMessage(content="", tool_calls=calls)])
        builder = StateGraph(MessagesState)
        builder.add_node(
            "agent", lambda state: {"messages": [next(script, AIMessage(content="done"))]}
        )
        node = ToolNode(
            [with_store],
            wrap_tool_call=example.recusal_gate(self.recording(seen, example.sql_scope_policy)),
        )
        builder.add_node("tools", node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        graph = builder.compile(store=InMemoryStore())
        messages = [
            m for m in graph.invoke({"messages": []})["messages"] if isinstance(m, ToolMessage)
        ]
        assert messages[0].status == "success", messages[0].content
        assert ran == [{"sql": SELECT, "store": "InMemoryStore"}] and seen == [{"sql": SELECT}]

    # --- every model-controlled argument is judged, not only the published properties ------

    @staticmethod
    def extra_tool(ran):
        import json

        from langchain_core.tools import tool
        from pydantic import BaseModel, ConfigDict

        class Args(BaseModel):
            model_config = ConfigDict(extra="allow")
            sql: str

        @tool("query", args_schema=Args)
        def query(sql: str, **kwargs) -> str:
            """Record a query and whatever extra options the schema accepted."""
            ran.append(json.loads(json.dumps({"sql": sql, **kwargs})))
            return "recorded"

        return query

    @staticmethod
    def own_subject_only(name, args):
        from recusal import Finding

        if args.get("subject", "me") != "me":
            return [Finding.fail("subject", severity="CRITICAL", message="another subject")]
        return []

    def test_an_extra_field_the_schema_accepts_is_judged(self, example):
        ran, seen = [], []
        query = self.extra_tool(ran)
        assert list(query.args) == ["sql"], "the published schema does not list the extra"
        messages, _ = example.run_calls(
            [self.call("query", {"sql": SELECT, "subject": "other-customer"})],
            policy=self.recording(seen, self.own_subject_only),
            tools=[query],
        )
        assert seen == [{"sql": SELECT, "subject": "other-customer"}]
        assert ran == [] and "another subject" in messages[0].content
        messages, _ = example.run_calls(
            [self.call("query", {"sql": SELECT})],
            policy=self.recording(seen, self.own_subject_only),
            tools=[query],
        )
        assert messages[0].status == "success" and ran == [{"sql": SELECT}]

    # --- the lifecycle the seam relies on ----------------------------------------------------

    def test_a_tool_whose_class_bypasses_the_parse_step_is_refused_not_executed(self, example):
        from langchain_core.messages import ToolMessage
        from langchain_core.tools import BaseTool

        ran = []

        class DirectTool(BaseTool):
            name: str = "direct_sql"
            description: str = "A custom invocation path, recording only."

            def _run(self, sql: str) -> str:
                ran.append(sql)
                return "recorded"

            def invoke(self, input, config=None, **kwargs):
                return ToolMessage(
                    content=self._run(**input["args"]), name=self.name, tool_call_id=input["id"]
                )

        messages, _ = example.run_calls(
            [self.call("direct_sql", {"sql": DELETE})], tools=[DirectTool()]
        )
        assert ran == []
        assert messages[0].status == "error"
        assert "cannot be judged (DirectTool overrides `invoke`)" in messages[0].content
        # the same tool, ungated, runs: the refusal is the gate's doing
        example.run_calls(
            [self.call("direct_sql", {"sql": DELETE})], gated=False, tools=[DirectTool()]
        )
        assert ran == [DELETE]

    def test_an_async_only_override_is_refused_on_both_routes(self, example):
        from langchain_core.tools import BaseTool

        ran = []

        class AsyncDirect(BaseTool):
            name: str = "adirect"
            description: str = "Overrides only the async entry point."

            def _run(self, sql: str) -> str:
                ran.append(sql)
                return "recorded"

            async def arun(self, tool_input, *args, **kwargs):
                return self._run(**tool_input)

        messages, _ = example.run_calls(
            [self.call("adirect", {"sql": SELECT})], tools=[AsyncDirect()]
        )
        assert ran == [] and "AsyncDirect overrides `arun`" in messages[0].content
        messages, _ = asyncio.run(
            example.run_calls_async([self.call("adirect", {"sql": SELECT})], tools=[AsyncDirect()])
        )
        assert ran == [] and "AsyncDirect overrides `arun`" in messages[0].content

    def test_an_instance_level_override_is_refused(self, example):
        copy = example.run_sql.model_copy()
        object.__setattr__(copy, "run", lambda *a, **k: "bypassed")
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})], tools=[copy]
        )
        assert executed == []
        assert "`run` is overridden on the tool instance" in messages[0].content

    # --- the projection -------------------------------------------------------------------------

    def test_a_mutating_policy_cannot_change_the_invocation(self, example):
        ran = []

        def mutate(name, args):
            args["query"]["sql"] = DELETE
            args["extra"] = 1
            return []

        messages, _ = example.run_calls(
            [self.call("query", {"query": {"sql": SELECT}})],
            policy=mutate,
            tools=[self.serializer_tool(ran)],
        )
        assert messages[0].status == "success"
        assert ran == [{"sql": SELECT, "type": "Query"}]

    def test_a_value_that_cannot_be_projected_is_refused_without_executing(self, example):
        from langchain_core.tools import tool
        from pydantic import BaseModel, field_validator

        class Args(BaseModel):
            names: Any

            @field_validator("names")
            @classmethod
            def to_set(cls, value):
                return set(value)

        ran = []

        @tool("tag", args_schema=Args)
        def tag(names: Any) -> str:
            """Record names."""
            ran.append(names)
            return "recorded"

        seen = []
        messages, _ = example.run_calls(
            [self.call("tag", {"names": ["a"]})],
            policy=self.recording(seen, lambda n, a: []),
            tools=[tag],
        )
        assert ran == [] and seen == []
        assert "a set value cannot be judged faithfully" in messages[0].content

    def test_the_projection_is_explicit_about_what_it_supports(self, example):
        import dataclasses
        import datetime
        import decimal
        import enum
        import uuid

        import pydantic.v1
        from pydantic import BaseModel

        class Color(enum.Enum):
            RED = "red"

        class Level(enum.IntEnum):
            HIGH = 3

        @dataclasses.dataclass
        class Point:
            x: int
            y: float

        class Inner(BaseModel):
            flag: bool

        class Outer(BaseModel):
            inner: Inner
            when: datetime.date

        faithful = _adapter()._faithful
        assert faithful((1, True, "s", None)) == [1, True, "s", None]
        projected = faithful([1, True])
        assert [type(v) for v in projected] == [int, bool], "1 and True stay distinct"
        assert faithful(Color.RED) == "red" and faithful(Level.HIGH) == 3
        assert faithful(datetime.datetime(2026, 11, 4, 12, 0)) == "2026-11-04T12:00:00"
        assert faithful(decimal.Decimal("1.50")) == "1.50"
        ident = uuid.UUID(int=1)
        assert faithful(ident) == str(ident)
        assert faithful(Point(1, 2.5)) == {"x": 1, "y": 2.5}
        assert faithful(Outer(inner=Inner(flag=False), when=datetime.date(2026, 1, 2))) == {
            "inner": {"flag": False},
            "when": "2026-01-02",
        }
        for bad in (
            float("nan"),
            float("inf"),
            decimal.Decimal("NaN"),
            {1: "int key"},
            {"a"},
            object(),
            pydantic.v1.create_model("V1", x=(int, 1))(),
        ):
            with pytest.raises(TypeError):
                faithful(bad)
        cyclic = []
        cyclic.append(cyclic)
        with pytest.raises(TypeError, match="cyclic"):
            faithful(cyclic)
        source = {"q": [1]}
        copy = faithful(source)
        copy["q"].append(2)
        assert source == {"q": [1]}, "the projection is detached"

    # --- refusal handling: narrow, and ordinary errors keep their path ---------------------

    @pytest.mark.parametrize(
        "handler", [True, lambda exc: "replaced by the node"], ids=["bool", "callable"]
    )
    def test_a_refusal_survives_the_nodes_error_handler(self, example, handler):
        from langgraph.prebuilt import ToolNode

        node = ToolNode(
            [example.run_sql],
            wrap_tool_call=example.recusal_gate(example.sql_scope_policy),
            handle_tool_errors=handler,
        )
        example.EXECUTED.clear()
        messages = example.run_in_graph(node, [self.call("run_sql", {"sql": DELETE})])
        assert example.EXECUTED == []
        assert messages[0].status == "error"
        assert messages[0].content.startswith("Recusal refused `run_sql`"), messages[0].content

    @staticmethod
    def erroring_tool(**update):
        from langchain_core.tools import ToolException, tool

        @tool
        def erroring(sql: str) -> str:
            """Raise an ordinary tool error."""
            raise ToolException("ordinary tool failure")

        return erroring.model_copy(update=update) if update else erroring

    def test_an_ordinary_tool_error_still_propagates_when_the_tool_does_not_handle_it(
        self, example
    ):
        from langchain_core.tools import ToolException
        from langgraph.prebuilt import ToolNode

        node = ToolNode(
            [self.erroring_tool()],
            wrap_tool_call=example.recusal_gate(lambda n, a: []),
            handle_tool_errors=False,
        )
        with pytest.raises(ToolException, match="ordinary tool failure"):
            example.run_in_graph(node, [self.call("erroring", {"sql": SELECT})])

    @pytest.mark.parametrize(
        ("setting", "expected"),
        [
            (True, "ordinary tool failure"),
            ("the tool's own text", "the tool's own text"),
            (lambda exc: f"handled: {exc}", "handled: ordinary tool failure"),
        ],
        ids=["bool", "str", "callable"],
    )
    def test_an_ordinary_tool_error_keeps_the_tools_own_handling(self, example, setting, expected):
        messages, _ = example.run_calls(
            [self.call("erroring", {"sql": SELECT})],
            policy=lambda n, a: [],
            tools=[self.erroring_tool(handle_tool_error=setting)],
        )
        assert messages[0].status == "error" and messages[0].content == expected

    # --- redaction after injection ------------------------------------------------------------

    def test_a_validation_error_on_host_state_never_shows_the_state(self, example):
        from typing import Annotated

        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langgraph.prebuilt import InjectedState, ToolNode

        @tool
        def wants_int_state(sql: str, state: Annotated[int, InjectedState]) -> str:
            """Declare a host-state shape the graph does not provide."""
            return "recorded"

        node = ToolNode([wants_int_state], wrap_tool_call=example.recusal_gate(lambda n, a: []))
        graph = example._graph(node, [self.call("wants_int_state", {"sql": SELECT})])
        result = graph.invoke({"messages": [HumanMessage(content="HOST_SECRET_SENTINEL")]})
        message = [m for m in result["messages"] if getattr(m, "tool_call_id", None) == "c1"][0]
        assert message.status == "error"
        assert "HOST_SECRET_SENTINEL" not in message.content
        assert "state: int_type" in message.content, message.content

    @staticmethod
    def host_echoing_tools():
        """Two validators that build their message from injected state: one at the model
        root, one on a model-facing field."""
        from typing import Annotated

        from langchain_core.tools import tool
        from langgraph.prebuilt import InjectedState
        from pydantic import BaseModel, field_validator, model_validator

        class RootArgs(BaseModel):
            sql: str
            state: Annotated[dict, InjectedState]

            @model_validator(mode="after")
            def check_host_context(self):
                raise ValueError("invalid host context: " + self.state["messages"][0].content)

        class FieldArgs(BaseModel):
            state: Annotated[dict, InjectedState]
            sql: str

            @field_validator("sql")
            @classmethod
            def check_sql(cls, value, info):
                raise ValueError("rejected for " + info.data["state"]["messages"][0].content)

        @tool("root_query", args_schema=RootArgs)
        def root_query(sql: str, state: Annotated[dict, InjectedState]) -> str:
            """Record only; the failing validator runs first."""
            return "recorded"

        @tool("field_query", args_schema=FieldArgs)
        def field_query(sql: str, state: Annotated[dict, InjectedState]) -> str:
            """Record only; the failing validator runs first."""
            return "recorded"

        return root_query, field_query

    @pytest.mark.parametrize("which", ["root_query", "field_query"])
    def test_a_validator_message_built_from_host_state_never_reaches_the_model(
        self, example, which
    ):
        from langchain_core.messages import HumanMessage
        from langgraph.prebuilt import ToolNode

        tools = {t.name: t for t in self.host_echoing_tools()}
        node = ToolNode([tools[which]], wrap_tool_call=example.recusal_gate(lambda n, a: []))
        graph = example._graph(node, [self.call(which, {"sql": SELECT})])
        result = graph.invoke({"messages": [HumanMessage(content="HOST_SECRET_SENTINEL")]})
        message = [m for m in result["messages"] if getattr(m, "tool_call_id", None) == "c1"][0]
        assert message.status == "error"
        assert "HOST_SECRET_SENTINEL" not in message.content, message.content
        expected = "model: value_error" if which == "root_query" else "sql: value_error"
        assert expected in message.content, message.content

    def test_the_sanitizer_keeps_only_pydantic_template_messages(self, example):
        from pydantic import BaseModel, ValidationError, field_validator, model_validator

        class Args(BaseModel):
            sql: str
            count: int

            @field_validator("sql")
            @classmethod
            def no_secret(cls, value):
                raise ValueError("echo SECRET " + value)

            @model_validator(mode="after")
            def whole(self):
                raise ValueError("whole SECRET")

        with pytest.raises(ValidationError) as caught:
            Args(sql="x", count="SECRET-not-an-int")
        text = _adapter()._sanitized(caught.value, frozenset())
        assert "SECRET" not in text, text
        assert "count: Input should be a valid integer" in text, text
        assert "sql: value_error" in text, text

        class Whole(BaseModel):
            sql: str

            @model_validator(mode="after")
            def whole(self):
                raise ValueError("whole SECRET")

        with pytest.raises(ValidationError) as caught:
            Whole(sql="x")
        assert _adapter()._sanitized(caught.value, frozenset()) == "model: value_error"

        class Asserting(BaseModel):
            sql: str

            @field_validator("sql")
            @classmethod
            def must(cls, value):
                assert value == "never", "SECRET in an assertion"
                return value

        with pytest.raises(ValidationError) as caught:
            Asserting(sql="x")
        assert _adapter()._sanitized(caught.value, frozenset()) == "sql: assertion_error"

    def test_a_validation_error_on_a_model_argument_names_the_location_not_the_input(self, example):
        messages, _ = example.run_calls(
            [self.call("run_sql", {"sql": {"nested": "MODEL_SECRET_SENTINEL"}})]
        )
        assert "sql: Input should be a valid string" in messages[0].content
        assert "MODEL_SECRET_SENTINEL" not in messages[0].content

    # --- a parameter the parse suppressed is completed from the callable --------------------

    @staticmethod
    def suppressed_tool(shape, ran, default=100):
        from langchain_core.tools import tool
        from pydantic import BaseModel, Field, model_serializer

        if shape == "excluded_field":

            class Args(BaseModel):
                amount: int = Field(default=default, exclude=True)

        elif shape == "root_serializer":

            class Args(BaseModel):
                amount: int = default

                @model_serializer
                def display(self):
                    return {}

        else:

            class Args(BaseModel):
                amount: int = default

        def charge(amount: int = default) -> str:
            """Record a hypothetical charge; contact no service."""
            ran.append(amount)
            return "recorded"

        return tool("charge", args_schema=Args)(charge)

    @staticmethod
    def budget(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append(dict(args))
            if args.get("amount", 0) > 10:
                return [Finding.fail("budget", severity="CRITICAL", message="over 10")]
            return []

        return policy

    @pytest.mark.parametrize("shape", ["excluded_field", "root_serializer"])
    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_suppressed_parameter_is_judged_at_the_value_the_function_would_use(
        self, example, shape, route
    ):
        """The review's probe: the parse drops `amount`, the function would run its default
        of 100, and the policy saw `{}`. It now sees 100 and refuses."""
        ran, seen = [], []
        calls = [self.call("charge", {"amount": 5})]
        tools = [self.suppressed_tool(shape, ran)]
        if route == "sync":
            messages, _ = example.run_calls(calls, policy=self.budget(seen), tools=tools)
        else:
            messages, _ = asyncio.run(
                example.run_calls_async(calls, policy=self.budget(seen), tools=tools)
            )
        assert seen == [{"amount": 100}], seen
        assert ran == [] and "over 10" in messages[0].content

    @pytest.mark.parametrize("shape", ["excluded_field", "root_serializer"])
    def test_negative_control_ungated_the_suppressed_default_runs(self, example, shape):
        ran = []
        example.run_calls(
            [self.call("charge", {"amount": 5})],
            gated=False,
            tools=[self.suppressed_tool(shape, ran)],
        )
        assert ran == [100]

    def test_a_suppressed_default_under_the_cap_runs_as_judged(self, example):
        ran, seen = [], []
        messages, _ = example.run_calls(
            [self.call("charge", {"amount": 5})],
            policy=self.budget(seen),
            tools=[self.suppressed_tool("excluded_field", ran, default=7)],
        )
        assert seen == [{"amount": 7}] and ran == [7]
        assert messages[0].status == "success"

    def test_an_ordinary_schema_keeps_the_proposal_and_its_judged_view(self, example):
        ran, seen = [], []
        messages, _ = example.run_calls(
            [self.call("charge", {"amount": 5})],
            policy=self.budget(seen),
            tools=[self.suppressed_tool("plain", ran)],
        )
        assert seen == [{"amount": 5}] and ran == [5] and messages[0].status == "success"

    def test_a_suppressed_parameter_with_no_default_is_refused_not_executed(self, example):
        from langchain_core.tools import tool
        from pydantic import BaseModel, Field

        class Args(BaseModel):
            amount: int = Field(default=100, exclude=True)

        ran = []

        @tool("charge", args_schema=Args)
        def charge(amount: int) -> str:
            """Record a charge; the callable has no default of its own."""
            ran.append(amount)
            return "recorded"

        seen = []
        messages, _ = example.run_calls(
            [self.call("charge", {"amount": 5})], policy=self.budget(seen), tools=[charge]
        )
        assert ran == [] and seen == []
        assert "the schema suppressed `amount`" in messages[0].content

    def test_framework_filled_parameters_are_not_completed(self, example):
        from langchain_core.callbacks import CallbackManagerForToolRun
        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import tool

        ran, seen = [], []

        @tool
        def with_runtime_params(
            sql: str,
            config: RunnableConfig,
            run_manager: CallbackManagerForToolRun | None = None,
        ) -> str:
            """Record SQL; config and run_manager come from the framework at call time."""
            ran.append((sql, type(config).__name__, run_manager is not None))
            return "recorded"

        messages, _ = example.run_calls(
            [self.call("with_runtime_params", {"sql": SELECT})],
            policy=self.recording(seen, example.sql_scope_policy),
            tools=[with_runtime_params],
        )
        assert messages[0].status == "success", messages[0].content
        assert seen == [{"sql": SELECT}]
        assert ran == [(SELECT, "dict", False)], "config came from the framework, not the gate"

    # --- the callable is held to the schema: a parameter it declares beyond it refuses -------

    @staticmethod
    def narrow_tool(ran, declared=False, default=100, required=False):
        """``charge(label, amount)`` behind a schema that declares ``label`` and, only when
        ``declared``, ``amount``."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        if declared:

            class Args(BaseModel):
                label: str
                amount: int = default

        else:

            class Args(BaseModel):
                label: str

        if required:

            def charge(label: str, amount: int) -> str:
                """Record a hypothetical charge; contact no service."""
                ran.append((label, amount))
                return "recorded"

        else:

            def charge(label: str, amount: int = default) -> str:
                """Record a hypothetical charge; contact no service."""
                ran.append((label, amount))
                return "recorded"

        return StructuredTool.from_function(charge, name="charge", args_schema=Args)

    @staticmethod
    def on_route(example, route, calls, policy, tools, gated=True):
        if route == "sync":
            return example.run_calls(calls, policy=policy, tools=tools, gated=gated)
        return asyncio.run(example.run_calls_async(calls, policy=policy, tools=tools, gated=gated))

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_callable_parameter_the_schema_omits_refuses_before_the_policy(self, example, route):
        """The 2026-11-06 review's P1: ``charge(label, amount=100)`` behind a schema declaring
        only ``label`` was judged as ``{"label": "review"}`` and ran with ``amount=100``."""
        ran, seen = [], []
        messages, executed = self.on_route(
            example,
            route,
            [self.call("charge", {"label": "review"})],
            self.budget(seen),
            [self.narrow_tool(ran)],
        )
        assert ran == [] and executed == [] and seen == []
        assert messages[0].status == "error"
        text = messages[0].content
        assert "the callable's parameter `amount` is absent from the argument schema" in text
        assert "100" not in text, "the default is never disclosed"

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_negative_control_ungated_the_schema_absent_default_runs(self, example, route):
        ran = []
        self.on_route(
            example,
            route,
            [self.call("charge", {"label": "review"})],
            self.budget([]),
            [self.narrow_tool(ran)],
            gated=False,
        )
        assert ran == [("review", 100)]

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_declaring_the_parameter_judges_it_at_the_value_that_runs(self, example, route):
        ran, seen = [], []
        messages, _ = self.on_route(
            example,
            route,
            [self.call("charge", {"label": "review"})],
            self.budget(seen),
            [self.narrow_tool(ran, declared=True)],
        )
        assert seen == [{"label": "review", "amount": 100}] and ran == []
        assert "over 10" in messages[0].content
        ran, seen = [], []
        messages, _ = self.on_route(
            example,
            route,
            [self.call("charge", {"label": "review", "amount": 5})],
            self.budget(seen),
            [self.narrow_tool(ran, declared=True)],
        )
        assert seen == [{"label": "review", "amount": 5}] and ran == [("review", 5)]
        assert messages[0].status == "success"

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_required_callable_parameter_the_schema_omits_refuses_too(self, example, route):
        ran, seen = [], []
        messages, _ = self.on_route(
            example,
            route,
            [self.call("charge", {"label": "review"})],
            self.budget(seen),
            [self.narrow_tool(ran, required=True)],
        )
        assert ran == [] and seen == []
        assert "the callable's parameter `amount` is absent from the argument schema" in (
            messages[0].content
        )

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_suppressed_default_is_completed_from_the_callable_the_route_runs(
        self, example, route
    ):
        """``func`` and ``coroutine`` with different defaults: the async route completes from
        the coroutine's, never the sync function's."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        class Args(BaseModel):
            amount: int = Field(default=100, exclude=True)

        ran, seen = [], []

        def charge(amount: int = 5) -> str:
            """Record a charge; the sync default is under the cap."""
            ran.append(("sync", amount))
            return "recorded"

        async def acharge(amount: int = 100) -> str:
            ran.append(("async", amount))
            return "recorded"

        tool = StructuredTool.from_function(
            charge, coroutine=acharge, name="charge", args_schema=Args
        )
        messages, _ = self.on_route(
            example, route, [self.call("charge", {"amount": 7})], self.budget(seen), [tool]
        )
        if route == "sync":
            assert seen == [{"amount": 5}] and ran == [("sync", 5)]
            assert messages[0].status == "success", messages[0].content
        else:
            assert seen == [{"amount": 100}] and ran == []
            assert "over 10" in messages[0].content

    def test_a_basetool_subclass_is_inspected_on_the_route_it_runs(self, example):
        """``_run`` and ``_arun`` are inspected as bound methods (``self`` is no parameter),
        ``run_manager`` is the SDK's to fill, and the async route selects ``_arun`` only when
        the class overrides it."""
        from langchain_core.callbacks import (
            AsyncCallbackManagerForToolRun,
            CallbackManagerForToolRun,
        )
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel

        class Args(BaseModel):
            sql: str

        ran = []

        class Direct(BaseTool):
            name: str = "run_sql"
            description: str = "records SQL"
            args_schema: type[BaseModel] = Args

            def _run(self, sql: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
                ran.append(("sync", sql, run_manager is not None))
                return "recorded"

            async def _arun(
                self, sql: str, run_manager: AsyncCallbackManagerForToolRun | None = None
            ) -> str:
                ran.append(("async", sql, run_manager is not None))
                return "recorded"

        for route in ("sync", "async"):
            seen = []
            messages, _ = self.on_route(
                example,
                route,
                [self.call("run_sql", {"sql": SELECT})],
                self.recording(seen, example.sql_scope_policy),
                [Direct()],
            )
            assert messages[0].status == "success", messages[0].content
            assert seen == [{"sql": SELECT}]
        assert ran == [("sync", SELECT, True), ("async", SELECT, True)]
        tool = Direct()
        assert _adapter()._execution_callable(tool, False).__func__ is Direct._run
        assert _adapter()._execution_callable(tool, True).__func__ is Direct._arun

    def test_the_reserved_names_are_the_sdks_own(self, example):
        from langchain_core.tools.base import FILTERED_ARGS

        assert _adapter()._RESERVED == frozenset(FILTERED_ARGS) == {"run_manager", "callbacks"}

    def test_a_positional_only_callable_parameter_is_refused(self, example):
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        class Args(BaseModel):
            label: str

        ran, seen = [], []

        def charge(label: str, /) -> str:
            """A parameter the gate cannot pass by name."""
            ran.append(label)
            return "recorded"

        tool = StructuredTool.from_function(charge, name="charge", args_schema=Args)
        messages, _ = example.run_calls(
            [self.call("charge", {"label": "review"})], policy=self.budget(seen), tools=[tool]
        )
        assert ran == [] and seen == []
        assert "parameter `label` is positional-only" in messages[0].content

    def test_every_schema_subset_of_the_callable_refuses_unless_complete(self, example):
        """Exhaustive over ``charge(a=1, b=2, c=3)``: for every subset of parameters the
        schema declares, every proposal over the declared ones, on both routes, the gate
        refuses before the policy when any parameter is undeclared (naming the first), and
        otherwise judges exactly the invocation the function receives."""
        from itertools import combinations

        from langchain_core.tools import StructuredTool
        from pydantic import create_model

        names, defaults = ("a", "b", "c"), {"a": 1, "b": 2, "c": 3}
        cases = 0
        for size in range(len(names) + 1):
            for declared in combinations(names, size):
                schema = create_model("Args", **{n: (int, defaults[n]) for n in declared})
                for proposal_size in range(len(declared) + 1):
                    for proposed in combinations(declared, proposal_size):
                        proposal = {n: 10 + defaults[n] for n in proposed}
                        for route in ("sync", "async"):
                            ran, seen = [], []

                            def charge(a: int = 1, b: int = 2, c: int = 3) -> str:
                                """Record the invocation."""
                                ran.append({"a": a, "b": b, "c": c})
                                return "recorded"

                            tool = StructuredTool.from_function(
                                charge, name="charge", args_schema=schema
                            )
                            messages, _ = self.on_route(
                                example,
                                route,
                                [self.call("charge", proposal)],
                                self.recording(seen, lambda n, a: []),
                                [tool],
                            )
                            cases += 1
                            where = (declared, proposal, route)
                            if len(declared) < len(names):
                                assert seen == [] and ran == [], where
                                assert messages[0].status == "error", where
                                missing = sorted(set(names) - set(declared))[0]
                                assert f"parameter `{missing}` is absent" in messages[0].content
                            else:
                                expected = {**defaults, **proposal}
                                assert seen == [expected] and ran == [expected], where
                                assert messages[0].status == "success", where
        assert cases == 2 * 27  # every (declared subset, proposal subset) pair, both routes

    # --- the old comparison machinery is gone -------------------------------------------------

    def test_no_second_validation_and_no_comparison_remain(self):
        source = EXAMPLE.read_text(encoding="utf-8")
        for gone in ("_pinned", "ChangedAfterJudgment", "refuse_changed", "to_jsonable_python"):
            assert gone not in source, gone


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestWhatRunsIsWhatWasJudged:
    """Review round 1 of the package (2026-11): values the function ran with that the policy
    never saw (a model-written call id, a wrapper's own default, the default behind a
    stripped injected argument), ordinary tools that were always refused, and refusals that
    did not say why. Each test names its finding."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_langgraph_round1_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def call(name, args, ident="c1"):
        return {"name": name, "args": args, "id": ident, "type": "tool_call"}

    @staticmethod
    def no_delete(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append((name, args))
            if "DELETE" in repr(args):
                return [Finding.fail("no_delete", severity="CRITICAL", message="no delete")]
            return []

        return policy

    def run(self, example, route, tools, calls, policy, gated=True):
        if route == "async":
            return asyncio.run(
                example.run_calls_async(calls, policy=policy, tools=tools, gated=gated)
            )[0]
        return example.run_calls(calls, policy=policy, tools=tools, gated=gated)[0]

    ROUTES = pytest.mark.parametrize("route", ["sync", "async"])

    # --- 1a: the call id is the model's ----------------------------------------------------

    @staticmethod
    def ticket_tool(ran):
        from typing import Annotated

        from langchain_core.tools import InjectedToolCallId, tool

        @tool
        def open_ticket(title: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
            """Open a ticket; the call id is stored as its reference."""
            ran.append((title, tool_call_id))
            return "opened"

        return open_ticket

    @ROUTES
    def test_a_call_id_the_tool_takes_as_a_parameter_is_judged_under_that_name(
        self, example, route
    ):
        ran, seen = [], []
        calls = [
            self.call("open_ticket", {"title": "x"}, "ok-1"),
            self.call("open_ticket", {"title": "x"}, "DELETE-ALL'; --"),
        ]
        messages = self.run(example, route, [self.ticket_tool(ran)], calls, self.no_delete(seen))
        assert sorted(seen, key=str) == sorted(
            [
                ("open_ticket", {"title": "x", "tool_call_id": "ok-1"}),
                ("open_ticket", {"title": "x", "tool_call_id": "DELETE-ALL'; --"}),
            ],
            key=str,
        )
        assert ran == [("x", "ok-1")]
        assert {m.tool_call_id: m.status for m in messages} == {
            "ok-1": "success",
            "DELETE-ALL'; --": "error",
        }

    def test_negative_control_ungated_the_model_written_call_id_reaches_the_tool(self, example):
        ran = []
        calls = [self.call("open_ticket", {"title": "x"}, "DELETE-ALL'; --")]
        self.run(example, "sync", [self.ticket_tool(ran)], calls, None, gated=False)
        assert ran == [("x", "DELETE-ALL'; --")]

    def test_a_call_id_the_model_also_wrote_as_an_argument_is_replaced_and_judged_once(
        self, example
    ):
        ran, seen = [], []
        calls = [self.call("open_ticket", {"title": "x", "tool_call_id": "forged"}, "real")]
        self.run(example, "sync", [self.ticket_tool(ran)], calls, self.no_delete(seen))
        assert seen == [("open_ticket", {"title": "x", "tool_call_id": "real"})]
        assert ran == [("x", "real")]

    # --- 2a, 2b: a decorator between the tool and its function -----------------------------

    @staticmethod
    def own_default_tool(ran):
        import functools

        from langchain_core.tools import tool

        def confirmable(fn):
            @functools.wraps(fn)
            def wrapper(*args, force=True, **kwargs):
                ran.append(("force", force))
                return fn(*args, **kwargs)

            return wrapper

        @tool
        @confirmable
        def drop_table(table: str) -> str:
            """Drop a table."""
            ran.append(("drop_table", table))
            return "dropped"

        return drop_table

    @staticmethod
    def shadowing_tool(ran):
        import functools

        from langchain_core.tools import tool

        def with_timeout(fn):
            @functools.wraps(fn)
            def wrapper(*args, timeout=30, **kwargs):
                return fn(*args, **kwargs)  # the wrapper keeps the judged timeout

            return wrapper

        @tool
        @with_timeout
        def fetch(url: str, timeout: int = 999999) -> str:
            """Fetch a url."""
            ran.append((url, timeout))
            return "fetched"

        return fetch

    @staticmethod
    def declared_signature_tool(ran):
        import inspect

        from langchain_core.tools import tool

        def raw(table, mode="DELETE"):
            ran.append((table, mode))
            return "read"

        def shown(table: str) -> str: ...

        raw.__signature__ = inspect.signature(shown)
        raw.__annotations__ = {"table": str, "return": str}
        raw.__doc__ = "Read a table."
        return tool(raw)

    @ROUTES
    def test_a_wrappers_own_keyword_default_is_refused_before_the_policy(self, example, route):
        ran, seen = [], []
        calls = [self.call("drop_table", {"table": "t"})]
        messages = self.run(
            example, route, [self.own_default_tool(ran)], calls, self.no_delete(seen)
        )
        assert ran == [] and seen == []
        assert messages[0].status == "error"
        assert "the callable's parameter `force` is absent from the argument schema" in (
            messages[0].content
        )
        assert "True" not in messages[0].content, "the default is never disclosed"

    @ROUTES
    def test_a_parameter_a_wrapper_keeps_from_the_function_it_wraps_is_refused(
        self, example, route
    ):
        ran, seen = [], []
        calls = [self.call("fetch", {"url": "u", "timeout": 5})]
        messages = self.run(example, route, [self.shadowing_tool(ran)], calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert messages[0].status == "error"
        assert (
            "the callable's parameter `timeout` is taken by a wrapper and by the function it "
            "wraps" in messages[0].content
        )

    def test_a_declared_signature_is_refused_before_the_policy(self, example):
        ran, seen = [], []
        calls = [self.call("raw", {"table": "t"})]
        tools = [self.declared_signature_tool(ran)]
        messages = self.run(example, "sync", tools, calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert "declares a signature that is not its own" in messages[0].content

    def test_negative_control_ungated_each_wrapped_tool_runs_the_unseen_value(self, example):
        ran = []
        self.run(
            example,
            "sync",
            [self.own_default_tool(ran)],
            [self.call("drop_table", {"table": "t"})],
            None,
            gated=False,
        )
        assert ran == [("force", True), ("drop_table", "t")]
        ran.clear()
        self.run(
            example,
            "sync",
            [self.shadowing_tool(ran)],
            [self.call("fetch", {"url": "u", "timeout": 5})],
            None,
            gated=False,
        )
        assert ran == [("u", 999999)], "the function ran its own default, not the model's 5"
        ran.clear()
        self.run(
            example,
            "sync",
            [self.declared_signature_tool(ran)],
            [self.call("raw", {"table": "t"})],
            None,
            gated=False,
        )
        assert ran == [("t", "DELETE")]

    @ROUTES
    def test_a_wrapper_that_only_passes_arguments_through_runs_as_judged(self, example, route):
        import functools

        from langchain_core.tools import tool

        ran, seen = [], []

        def logged(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        @tool
        @logged
        def lookup(key: str, limit: int = 10) -> str:
            """Look a key up."""
            ran.append((key, limit))
            return "found"

        messages = self.run(
            example, route, [lookup], [self.call("lookup", {"key": "k"})], self.no_delete(seen)
        )
        assert messages[0].status == "success", messages[0].content
        assert seen == [("lookup", {"key": "k", "limit": 10})] and ran == [("k", 10)]

    def test_a_callable_that_cannot_be_walked_is_refused_with_the_class_of_the_failure(
        self, example, monkeypatch
    ):
        ran, seen = [], []
        adapter = _adapter()

        def unreadable(func):
            raise ValueError("too many wrappers to inspect")

        monkeypatch.setattr(adapter, "_reached_parameters", unreadable)
        messages = self.run(
            example,
            "sync",
            [self.own_default_tool(ran)],
            [self.call("drop_table", {"table": "t"})],
            self.no_delete(seen),
        )
        assert ran == [] and seen == []
        assert "cannot be inspected for its parameters (ValueError)" in messages[0].content

    # --- 2c: a stripped injected argument nothing replaced ----------------------------------

    def test_the_default_behind_an_injected_state_field_the_state_lacks_is_judged(self, example):
        from typing import Annotated

        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.tools import tool
        from langgraph.graph import START, MessagesState, StateGraph
        from langgraph.prebuilt import InjectedState, tools_condition
        from typing_extensions import NotRequired

        class State(MessagesState):
            role: NotRequired[str]

        ran, seen = [], []

        @tool
        def act(what: str, role: Annotated[str, InjectedState("role")] = "admin") -> str:
            """Act; the role comes from graph state."""
            ran.append((what, role))
            return "acted"

        def run(state_in):
            calls = [self.call("act", {"what": "x", "role": "forged"})]
            script = iter([AIMessage(content="", tool_calls=calls)])
            builder = StateGraph(State)
            builder.add_node(
                "agent", lambda state: {"messages": [next(script, AIMessage(content="done"))]}
            )
            builder.add_node("tools", example.gated_tool_node([act], self.no_delete(seen)))
            builder.add_edge(START, "agent")
            builder.add_conditional_edges("agent", tools_condition)
            builder.add_edge("tools", "agent")
            out = builder.compile().invoke(state_in)["messages"]
            return [m for m in out if isinstance(m, ToolMessage)]

        # the state has no role: ToolNode strips the model's value and injects nothing, so
        # the function runs its own default, and that default is what the policy is shown
        assert run({"messages": []})[0].status == "success"
        assert seen == [("act", {"what": "x", "role": "admin"})] and ran == [("x", "admin")]
        # the state has one: it is the host's, and is left out of judgment
        seen.clear(), ran.clear()
        assert run({"messages": [], "role": "viewer"})[0].status == "success"
        assert seen == [("act", {"what": "x"})] and ran == [("x", "viewer")]

    def test_a_schema_default_under_an_injected_key_the_host_never_fills_is_judged(self, example):
        """The schema marks ``tenant`` injected, the function does not, and no host supplies
        one: the schema default is what the function receives."""
        from typing import Annotated

        from langchain_core.tools import InjectedToolArg, StructuredTool
        from pydantic import BaseModel

        ran, seen = [], []

        class Args(BaseModel):
            what: str
            tenant: Annotated[str, InjectedToolArg] = "DELETE-tenant"

        def impl(what: str, tenant: str = "fn-default") -> str:
            ran.append((what, tenant))
            return "ok"

        tool = StructuredTool.from_function(
            func=impl, name="impl", description="d", args_schema=Args
        )
        messages = self.run(
            example, "sync", [tool], [self.call("impl", {"what": "x"})], self.no_delete(seen)
        )
        assert seen == [("impl", {"what": "x", "tenant": "DELETE-tenant"})]
        assert ran == [] and messages[0].status == "error"

    # --- a BaseTool subclass that declares no schema ----------------------------------------

    @staticmethod
    def schemaless(ran):
        from langchain_core.tools import BaseTool

        class Lookup(BaseTool):
            name: str = "lookup"
            description: str = "Look up."

            def _run(self, query: str, limit: int = 10, run_manager=None) -> str:
                ran.append((query, limit))
                return "found"

        return Lookup()

    @ROUTES
    def test_a_tool_with_no_declared_schema_is_held_to_the_one_langchain_infers(
        self, example, route
    ):
        ran, seen = [], []
        tool = self.schemaless(ran)
        assert tool.args_schema is None
        calls = [
            self.call("lookup", {"query": "ok"}, "a"),
            self.call("lookup", {"query": "DELETE"}, "b"),
        ]
        messages = self.run(example, route, [tool], calls, self.no_delete(seen))
        # the model left `limit` out and nothing validated the input, so it is completed
        # from the function's default and judged
        assert sorted(seen, key=str) == [
            ("lookup", {"query": "DELETE", "limit": 10}),
            ("lookup", {"query": "ok", "limit": 10}),
        ]
        assert ran == [("ok", 10)]
        assert {m.tool_call_id: m.status for m in messages} == {"a": "success", "b": "error"}

    def test_a_required_parameter_a_schemaless_tool_was_not_given_is_refused(self, example):
        ran, seen = [], []
        messages = self.run(
            example,
            "sync",
            [self.schemaless(ran)],
            [self.call("lookup", {"limit": 3})],
            self.no_delete(seen),
        )
        assert ran == [] and seen == []
        assert "the call leaves out `query`" in messages[0].content

    def test_a_schema_that_cannot_be_inferred_is_refused(self, example, monkeypatch):
        ran, seen = [], []
        tool = self.schemaless(ran)

        def broken(self, config=None):
            raise RuntimeError("no schema")

        # built directly, and broken only afterwards: ToolNode's own constructor and the
        # host-argument helper read the schema too
        call = self.call("lookup", {"query": "x"})
        copy = _adapter()._judging(tool, call, self.no_delete(seen), {}, False)
        monkeypatch.setattr(type(tool), "get_input_schema", broken)
        with pytest.raises(example.RecusalRefusal, match="cannot be inferred .RuntimeError."):
            copy._to_args_and_kwargs({"query": "x"}, "c1")
        assert ran == [] and seen == []

    # --- a single-input Tool ------------------------------------------------------------------

    @ROUTES
    def test_a_single_input_tool_is_judged_under_the_name_of_its_parameter(self, example, route):
        from langchain_core.tools import Tool

        ran, seen = [], []

        def lookup(q):
            ran.append(q)
            return "found"

        tool = Tool(name="lookup", func=lookup, description="Look up.")
        calls = [
            self.call("lookup", {"__arg1": "ok"}, "a"),
            self.call("lookup", {"anything": "DELETE"}, "b"),
        ]
        messages = self.run(example, route, [tool], calls, self.no_delete(seen))
        assert sorted(seen, key=str) == [("lookup", {"q": "DELETE"}), ("lookup", {"q": "ok"})]
        assert ran == ["ok"]
        assert {m.tool_call_id: m.status for m in messages} == {"a": "success", "b": "error"}

    def test_a_tool_decorated_without_schema_inference_is_judged_the_same_way(self, example):
        from langchain_core.tools import tool

        ran, seen = [], []

        @tool(infer_schema=False)
        def simple(q: str) -> str:
            """Simple."""
            ran.append(q)
            return "ok"

        messages = self.run(
            example, "sync", [simple], [self.call("simple", {"__arg1": "x"})], self.no_delete(seen)
        )
        assert messages[0].status == "success", messages[0].content
        assert seen == [("simple", {"q": "x"})] and ran == ["x"]

    @pytest.mark.parametrize("shape", ["second_default", "variadic_first", "wrapper_adds"])
    def test_a_single_input_tool_whose_name_is_not_certain_is_refused(self, example, shape):
        """Pins the positional refusal the review's mutation M19 removed unnoticed."""
        import functools

        from langchain_core.tools import Tool

        ran, seen = [], []

        if shape == "second_default":

            def lookup(q, mode="DELETE"):
                ran.append((q, mode))
                return "found"

            expected = "the callable's parameter `mode` is not the tool's single input"
        elif shape == "variadic_first":

            def lookup(*parts):
                ran.append(parts)
                return "found"

            expected = "does not take its input as a first named parameter"
        else:

            def inner(q):
                ran.append(q)
                return "found"

            @functools.wraps(inner)
            def lookup(q, force=True):
                ran.append(("force", force))
                return inner(q)

            expected = "the callable's parameter `force` is not the tool's single input"

        tool = Tool(name="lookup", func=lookup, description="Look up.")
        messages = self.run(
            example, "sync", [tool], [self.call("lookup", {"__arg1": "x"})], self.no_delete(seen)
        )
        assert ran == [] and seen == []
        assert messages[0].status == "error" and expected in messages[0].content
        # negative control: ungated, the same tool runs
        self.run(example, "sync", [tool], [self.call("lookup", {"__arg1": "x"})], None, gated=False)
        assert ran != []

    def test_a_positional_input_to_any_other_tool_is_refused(self, example):
        """A string input reaches ``_run`` as one positional value. ToolNode never sends
        one, so the parse step is driven directly."""
        seen = []
        call = self.call("run_sql", "SELECT 1")
        copy = _adapter()._judging(example.run_sql, call, self.no_delete(seen), {}, False)
        with pytest.raises(
            example.RecusalRefusal, match="a positional tool input cannot be judged by name"
        ):
            copy._to_args_and_kwargs("SELECT 1", "c1")
        assert seen == []

    def test_the_single_input_allowance_is_for_the_judged_value_only(self, example):
        from langchain_core.tools import Tool

        ran = []
        tool = Tool(name="lookup", func=lambda q: ran.append(q), description="Look up.")
        call = self.call("lookup", {"__arg1": "x"})
        copy = _adapter()._judging(tool, call, lambda name, args: [], {}, False)
        args, kwargs = copy._to_args_and_kwargs({"__arg1": "x"}, "c1")
        assert args == ("x",) and kwargs == {}
        with pytest.raises(example.RecusalRefusal, match="arguments other than the ones"):
            copy.func("y")
        copy._to_args_and_kwargs({"__arg1": "x"}, "c1")
        with pytest.raises(example.RecusalRefusal, match="arguments other than the ones"):
            copy.func("x", "more")
        copy._to_args_and_kwargs({"__arg1": "x"}, "c1")
        with pytest.raises(example.RecusalRefusal, match="arguments other than the ones"):
            copy.func()
        assert ran == []
        args, _ = copy._to_args_and_kwargs({"__arg1": "x"}, "c1")
        copy.func(*args)
        assert ran == ["x"]

    # --- names LangChain fills itself ----------------------------------------------------------

    def test_the_names_a_function_holding_tool_keeps_are_the_sdks_own(self, example):
        assert _adapter()._HOLDER_TAKES == {"config", "run_manager"}

    @staticmethod
    def config_tool(ran):
        from langchain_core.tools import tool

        @tool
        def setup(name: str, config: str = "safe") -> str:
            """Set up with a named configuration."""
            ran.append((name, config))
            return "ok"

        return setup

    @ROUTES
    def test_a_parameter_named_config_is_refused_with_the_reason(self, example, route):
        """LangChain writes the run configuration under ``config`` and ``StructuredTool``
        keeps it, so the function never receives the model's value: it was refused as
        "arguments other than the ones the gate judged", after the policy had run."""
        ran, seen = [], []
        calls = [self.call("setup", {"name": "f", "config": "prod"})]
        messages = self.run(example, route, [self.config_tool(ran)], calls, self.no_delete(seen))
        assert ran == [] and seen == [], "refused before the policy runs"
        assert messages[0].status == "error"
        assert "the invocation sets `config`, which LangChain fills itself" in messages[0].content
        assert "prod" not in messages[0].content

    def test_negative_control_ungated_the_function_never_receives_the_models_config(self, example):
        ran = []
        calls = [self.call("setup", {"name": "f", "config": "prod"})]
        self.run(example, "sync", [self.config_tool(ran)], calls, None, gated=False)
        assert ran == [("f", "safe")]

    def test_the_node_helpers_refuse_to_build_with_a_parameter_named_config(
        self, example, tmp_path
    ):
        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import tool
        from recusal_langgraph import audited_tool_node

        from recusal.audit import AuditLog

        ran = []
        allow = lambda name, args: []  # noqa: E731
        with pytest.raises(TypeError, match="tool `setup`: its function's parameter `config`"):
            example.gated_tool_node([self.config_tool(ran)], allow)
        log = AuditLog(path=str(tmp_path / "audit.jsonl"))
        with pytest.raises(TypeError, match="tool `setup`: its function's parameter `config`"):
            audited_tool_node([self.config_tool(ran)], allow, log)

        @tool
        def configured(name: str, config: RunnableConfig) -> str:
            """Takes the run configuration, which is the framework's to fill."""
            return "ok"

        assert example.is_gated(example.gated_tool_node([configured, example.run_sql], allow))

    def test_a_reserved_name_the_model_sets_is_refused_before_the_policy(self, example):
        from langchain_core.tools import StructuredTool

        ran, seen = [], []

        def js(table, **rest):
            ran.append((table, rest))
            return "ok"

        schema = {"type": "object", "properties": {"table": {"type": "string"}}}
        tool = StructuredTool(name="js", description="d", func=js, args_schema=schema)
        calls = [self.call("js", {"table": "t", "run_manager": "x", "config": "y"})]
        messages = self.run(example, "sync", [tool], calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert "the invocation sets `config`, `run_manager`, which LangChain fills itself" in (
            messages[0].content
        )

    # --- the call's id and name ------------------------------------------------------------------

    @pytest.mark.parametrize("ident", ["", None], ids=["empty", "none"])
    @ROUTES
    def test_a_call_without_an_id_is_refused_before_the_policy(self, example, route, ident):
        seen = []
        calls = [self.call("run_sql", {"sql": SELECT}, ident)]
        messages = self.run(example, route, None, calls, self.no_delete(seen))
        assert seen == []
        assert messages[0].status == "error"
        assert "the tool call has no id" in messages[0].content

    @ROUTES
    def test_a_call_whose_name_is_not_the_tool_that_would_run_is_refused(self, example, route):
        """An earlier hook attached another tool than the one the call names (LangChain's
        ``request.override(tool=...)``). The policy would judge the model's string."""
        seen = []
        policy = self.no_delete(seen)
        gates = example.recusal_gates(policy)

        def repair(request, execute):
            return gates["wrap_tool_call"](request.override(tool=example.run_sql), execute)

        async def arepair(request, execute):
            return await gates["awrap_tool_call"](request.override(tool=example.run_sql), execute)

        node = example.ToolNode(
            [example.echo, example.run_sql], wrap_tool_call=repair, awrap_tool_call=arepair
        )
        calls = [self.call("Run_SQL", {"sql": SELECT})]
        example.EXECUTED.clear()
        if route == "async":
            messages = asyncio.run(example.run_in_graph_async(node, calls))
        else:
            messages = example.run_in_graph(node, calls)
        assert example.EXECUTED == [] and seen == []
        assert messages[0].status == "error" and messages[0].name == "run_sql"
        assert messages[0].content.startswith("Recusal refused `run_sql`")
        assert "the call names another tool than this one" in messages[0].content
        assert "Run_SQL" not in messages[0].content

    def test_a_hook_that_repairs_the_name_with_the_tool_is_judged_under_the_tools_name(
        self, example
    ):
        """What the package README tells such a hook to do: override the call's name
        together with the tool, so the two agree."""
        seen = []
        gate = example.recusal_gate(self.no_delete(seen))

        def repair(request, execute):
            fixed = {**request.tool_call, "name": example.run_sql.name}
            return gate(request.override(tool_call=fixed, tool=example.run_sql), execute)

        node = example.ToolNode([example.echo, example.run_sql], wrap_tool_call=repair)
        calls = [
            self.call("Run_SQL", {"sql": SELECT}, "a"),
            self.call("Run_SQL", {"sql": DELETE}, "b"),
        ]
        example.EXECUTED.clear()
        messages = example.run_in_graph(node, calls)
        assert sorted(seen, key=str) == [("run_sql", {"sql": DELETE}), ("run_sql", {"sql": SELECT})]
        assert example.EXECUTED == [("run_sql", {"sql": SELECT})]
        assert {m.tool_call_id: m.status for m in messages} == {"a": "success", "b": "error"}

    # --- the gate's own failure, and a framework that runs the tool another way ------------

    def test_a_decision_that_cannot_be_made_refuses_the_call(self, example, monkeypatch):
        """Pins the review's mutation M22: the branch that answers a ``decide`` that raises
        could be flipped to allow with every test passing."""
        adapter = _adapter()
        real = adapter.decide

        def raising(name, tool_input, policy, **kwargs):
            if getattr(policy, "__name__", "") == "unjudgeable":
                return real(name, tool_input, policy, **kwargs)
            raise ValueError("the decision cannot be made")

        monkeypatch.setattr(adapter, "decide", raising)
        messages, executed = example.run_calls([self.call("run_sql", {"sql": SELECT})])
        assert executed == []
        assert messages[0].status == "error"
        assert messages[0].content == (
            "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (ValueError)"
        )

    @ROUTES
    def test_an_answer_that_never_passed_the_judging_step_is_withheld(
        self, example, route, monkeypatch
    ):
        """A ToolNode release whose executor resolved the tool by name again would run the
        registered tool, not the judging copy: unjudged, and answered as a success."""
        from langgraph.prebuilt import ToolNode

        seen = []
        name = "_execute_tool_async" if route == "async" else "_execute_tool_sync"
        original = getattr(ToolNode, name)

        def swapped(self, request):
            return request.override(tool=self.tools_by_name[request.tool_call["name"]])

        if route == "async":

            async def drifted(self, request, input_type, config):
                return await original(self, swapped(self, request), input_type, config)

        else:

            def drifted(self, request, input_type, config):
                return original(self, swapped(self, request), input_type, config)

        monkeypatch.setattr(ToolNode, name, drifted)
        calls = [self.call("run_sql", {"sql": DELETE})]
        node = example.gated_tool_node([example.echo, example.run_sql], self.no_delete(seen))
        assert example.is_gated(node)
        example.EXECUTED.clear()
        if route == "async":
            messages = asyncio.run(example.run_in_graph_async(node, calls))
        else:
            messages = example.run_in_graph(node, calls)
        assert example.EXECUTED == [("run_sql", {"sql": DELETE})], "the drifted node ran it"
        assert seen == []
        assert messages[0].status == "error", "the unjudged answer is not passed on"
        assert "without passing the gate's judging step" in messages[0].content
        assert "its result is withheld" in messages[0].content

    def test_an_answer_that_passed_the_judging_step_is_returned_as_it_is(self, example):
        messages, executed = example.run_calls([self.call("run_sql", {"sql": SELECT})])
        assert executed == [("run_sql", {"sql": SELECT})]
        assert messages[0].status == "success" and "judging step" not in messages[0].content

    def test_host_arguments_that_cannot_be_read_refuse_the_call(self, example, monkeypatch):
        """A private framework name the gate reads is missing: refused, never a silent
        fallback to "the tool has no host arguments"."""
        from langchain_core.tools import StructuredTool

        seen = []

        def missing(self):
            raise AttributeError("_injected_args_keys")

        monkeypatch.setattr(StructuredTool, "_injected_args_keys", property(missing))
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})], policy=self.no_delete(seen)
        )
        assert executed == [] and seen == []
        assert messages[0].status == "error"
        assert "cannot be read from the installed LangChain and LangGraph (AttributeError)" in (
            messages[0].content
        )

    # --- the projection and is_gated, pinned ----------------------------------------------------

    def test_an_extra_field_of_a_nested_model_is_judged(self, example):
        """Pins the review's mutation M44: without the ``model_extra`` projection the policy
        saw ``{'f': {'table': 't'}}`` while the nested model carried ``raw_sql``."""
        from langchain_core.tools import tool
        from pydantic import BaseModel, ConfigDict

        ran, seen = [], []

        class Filter(BaseModel):
            model_config = ConfigDict(extra="allow")
            table: str

        @tool
        def query(f: Filter) -> str:
            """Query with a filter object."""
            ran.append((f.table, f.model_extra))
            return "ok"

        calls = [self.call("query", {"f": {"table": "t", "raw_sql": "DELETE FROM t"}})]
        messages = self.run(example, "sync", [query], calls, self.no_delete(seen))
        assert seen == [("query", {"f": {"table": "t", "raw_sql": "DELETE FROM t"}})]
        assert ran == [] and messages[0].status == "error"

    def test_is_gated_is_false_for_the_async_hook_in_the_sync_slot(self, example):
        """Pins the review's mutation M46: the sync slot must hold the sync hook, not just
        a hook this package built."""
        allow = lambda name, args: []  # noqa: E731
        hook = example.recusal_agate(allow, host_refuses_sync_runs=True)
        assert example.is_gated(example.ToolNode([example.run_sql], wrap_tool_call=hook)) is False
        both = example.recusal_gates(allow)
        swapped = example.ToolNode(
            [example.run_sql],
            wrap_tool_call=both["awrap_tool_call"],
            awrap_tool_call=both["awrap_tool_call"],
        )
        assert example.is_gated(swapped) is False
        assert example.is_gated(example.ToolNode([example.run_sql], **both)) is True

    # --- edges of the checks above, each pinned -------------------------------------------------

    def test_a_cache_around_the_function_names_its_parameters_twice_and_still_runs(self, example):
        """``functools.lru_cache`` shows the function's parameters on two callables without
        forwarding ``*args``; only a forwarding wrapper that names a parameter is refused."""
        import functools

        from langchain_core.tools import StructuredTool

        ran, seen = [], []

        @functools.lru_cache(maxsize=None)
        def cached(key: str, limit: int = 3) -> str:
            ran.append((key, limit))
            return "ok"

        tool = StructuredTool.from_function(func=cached, name="cached", description="d")
        messages = self.run(
            example, "sync", [tool], [self.call("cached", {"key": "k"})], self.no_delete(seen)
        )
        assert messages[0].status == "success", messages[0].content
        assert seen == [("cached", {"key": "k", "limit": 3})] and ran == [("k", 3)]

    def test_a_single_input_is_one_positional_value_and_nothing_else(self, example):
        from langchain_core.tools import Tool

        adapter = _adapter()
        tool = Tool(name="lookup", func=lambda q: q, description="Look up.")
        assert adapter._single_input(tool, ("x",), {}, False) == "q"
        for args, kwargs in ((("x", "y"), {}), (("x",), {"more": 1}), ((), {})):
            with pytest.raises(TypeError, match="a positional tool input cannot be judged"):
                adapter._single_input(tool, args, kwargs, False)

        def takes_a_reserved_name(callbacks):
            return callbacks

        reserved = Tool(name="lookup", func=takes_a_reserved_name, description="Look up.")
        with pytest.raises(TypeError, match="does not take its input as a first named"):
            adapter._single_input(reserved, ("x",), {}, False)

    def test_the_policy_gets_a_copy_of_a_single_input_not_the_object_that_runs(self, example):
        from langchain_core.tools import Tool

        ran = []

        def store(doc):
            ran.append(doc)
            return "stored"

        def mutating(name, args):
            args["doc"]["rows"].append("DELETE")
            return []

        tool = Tool(name="store", func=store, description="Store a document.")
        calls = [self.call("store", {"__arg1": {"rows": ["a"]}})]
        messages = self.run(example, "sync", [tool], calls, mutating)
        assert messages[0].status == "success", messages[0].content
        assert ran == [{"rows": ["a"]}]

    def test_a_string_input_counts_no_argument_as_the_hosts(self, example):
        """Only ToolNode's input dict can carry a host's value. With any other input an
        injected parameter is one nothing filled, and its default is not skipped."""
        from typing import Annotated

        from langchain_core.tools import InjectedToolArg, StructuredTool
        from pydantic import BaseModel

        ran, seen = [], []

        class Empty(BaseModel):
            pass

        def act(tenant: Annotated[str, InjectedToolArg] = "DELETE-tenant") -> str:
            ran.append(tenant)
            return "ok"

        tool = StructuredTool.from_function(
            func=act, name="act", description="d", args_schema=Empty
        )
        call = self.call("act", "go")
        copy = _adapter()._judging(tool, call, self.no_delete(seen), {}, False)
        with pytest.raises(example.RecusalRefusal, match="parameter `tenant` is absent"):
            copy._to_args_and_kwargs("go", "c1")
        assert ran == [] and seen == []

    def test_an_id_that_is_not_a_string_is_refused_with_an_answer_the_graph_accepts(self, example):
        """A model message cannot carry one, so the hook is called directly."""
        from langgraph.prebuilt.tool_node import ToolCallRequest

        seen = []
        request = ToolCallRequest(
            tool_call={"name": "run_sql", "args": {"sql": SELECT}, "id": 5, "type": "tool_call"},
            tool=example.run_sql,
            state={},
            runtime=None,
        )
        example.EXECUTED.clear()
        answer = example.recusal_gate(self.no_delete(seen))(request, lambda judged: "ran")
        assert seen == [] and example.EXECUTED == []
        assert answer.status == "error" and answer.tool_call_id == ""
        assert "the tool call has no id" in answer.content


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestWhatTheGateCannotReadIsRefused:
    """Review round 2 of the package (2026-11): arguments handed on to code the gate could
    not read (a decorator built without ``functools.wraps``, an ``_arun`` that hands the
    call to ``_run``), an answer that did not come from the execution the gate judged, and
    six checks no test pinned. Each test names its finding."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_langgraph_round2_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    call = staticmethod(TestWhatRunsIsWhatWasJudged.call)
    no_delete = staticmethod(TestWhatRunsIsWhatWasJudged.no_delete)
    run = TestWhatRunsIsWhatWasJudged.run
    ROUTES = TestWhatRunsIsWhatWasJudged.ROUTES

    HANDS_ON = "the tool's innermost callable takes `*args` or `**kwargs` and does not name `{}`"

    # --- F-B: a decorator built without functools.wraps -------------------------------------

    @staticmethod
    def unwrapped_tools(ran):
        """The review's two shapes: a decorator without ``functools.wraps`` that keeps the
        judged ``timeout`` for itself, and one that hides the function's own default."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        def with_timeout(fn):
            def wrapper(*args, timeout=30, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def logged(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def fetch(url: str, timeout: int = 999999) -> str:
            ran.append((url, timeout))
            return "fetched"

        class UrlTimeout(BaseModel):
            url: str
            timeout: int = 30

        class UrlOnly(BaseModel):
            url: str

        return [
            StructuredTool.from_function(
                func=with_timeout(fetch), name="keeps", description="d", args_schema=UrlTimeout
            ),
            StructuredTool.from_function(
                func=logged(fetch), name="hides", description="d", args_schema=UrlOnly
            ),
        ]

    @ROUTES
    def test_a_decorator_without_functools_wraps_is_refused_before_the_policy(self, example, route):
        ran, seen = [], []
        calls = [
            self.call("keeps", {"url": "u", "timeout": 5}, "a"),
            self.call("hides", {"url": "u"}, "b"),
        ]
        messages = self.run(example, route, self.unwrapped_tools(ran), calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert [m.status for m in messages] == ["error", "error"]
        for message in messages:
            assert self.HANDS_ON.format("url") in message.content
            assert "build a decorator with `functools.wraps`" in message.content

    def test_negative_control_ungated_the_unwrapped_decorators_run_the_unseen_default(
        self, example
    ):
        ran = []
        calls = [
            self.call("keeps", {"url": "u", "timeout": 5}, "a"),
            self.call("hides", {"url": "u"}, "b"),
        ]
        self.run(example, "sync", self.unwrapped_tools(ran), calls, None, gated=False)
        assert ran == [("u", 999999), ("u", 999999)], "neither ran the 5 the model sent"

    def test_a_function_with_kwargs_runs_when_it_names_every_schema_field(self, example):
        """``**kwargs`` alone is not refused: the variadic then carries only what the schema
        does not declare. LangChain's inferred schema declares the variadic itself, under
        its own name, and that is a parameter of the function too."""
        from langchain_core.tools import StructuredTool, tool

        ran, seen = [], []

        def js(table, **rest):
            ran.append((table, rest))
            return "ok"

        schema = {"type": "object", "properties": {"table": {"type": "string"}}}
        named = StructuredTool(name="js", description="d", func=js, args_schema=schema)

        @tool
        def inferred(q: str, **kwargs: Any) -> str:
            """Inferred."""
            ran.append((q, kwargs))
            return "ok"

        assert set(inferred.args_schema.model_fields) == {"q", "kwargs"}
        calls = [self.call("js", {"table": "t"}, "a"), self.call("inferred", {"q": "x"}, "b")]
        messages = self.run(example, "sync", [named, inferred], calls, self.no_delete(seen))
        assert [m.status for m in messages] == ["success", "success"], messages
        assert sorted(seen, key=str) == [("inferred", {"q": "x"}), ("js", {"table": "t"})]
        assert sorted(ran, key=str) == [("t", {}), ("x", {})]

    @ROUTES
    def test_a_function_that_takes_only_kwargs_under_a_schema_is_refused(self, example, route):
        """A function of your own with the shape ``langchain-mcp-adapters`` gives its tools
        (``call_tool(runtime=None, **arguments)`` under a JSON schema), and a ``BaseTool``
        whose ``_run`` is ``(**kwargs)``: the gate cannot tell either from a decorator that
        forwards. Only the function that library itself defines is the exception."""
        from langchain_core.tools import BaseTool, StructuredTool
        from pydantic import BaseModel

        ran, seen = [], []

        def call_tool(runtime=None, **arguments):
            ran.append(arguments)
            return "ok"

        async def acall_tool(runtime=None, **arguments):
            ran.append(arguments)
            return "ok"

        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "depth": {"type": "integer", "default": 2}},
        }
        mcp_shaped = StructuredTool(
            name="mcp", description="d", func=call_tool, coroutine=acall_tool, args_schema=schema
        )

        class Args(BaseModel):
            q: str

        class Loose(BaseTool):
            name: str = "loose"
            description: str = "d"
            args_schema: type[BaseModel] = Args

            def _run(self, **kwargs: Any) -> str:
                ran.append(kwargs)
                return "ok"

        calls = [self.call("mcp", {"path": "/x"}, "a"), self.call("loose", {"q": "x"}, "b")]
        messages = self.run(example, route, [mcp_shaped, Loose()], calls, self.no_delete(seen))
        assert ran == [] and seen == []
        by_id = {m.tool_call_id: m.content for m in messages}
        assert self.HANDS_ON.format("depth") in by_id["a"]
        assert self.HANDS_ON.format("q") in by_id["b"]

    def test_a_function_with_no_variadic_is_not_held_to_fields_it_does_not_name(self, example):
        """The rule is about a function that can hand arguments on. One with no ``*args``
        or ``**kwargs`` cannot receive a name it does not declare: the call is judged, and
        Python refuses the extra argument when the tool is called."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        ran, seen = [], []

        class Args(BaseModel):
            q: str
            note: str = "n"

        def lookup(q: str) -> str:
            ran.append(q)
            return "found"

        tool = StructuredTool(name="lookup", description="d", func=lookup, args_schema=Args)
        call = self.call("lookup", {"q": "x"})
        copy = _adapter()._judging(tool, call, self.no_delete(seen), {}, False)
        args, kwargs = copy._to_args_and_kwargs({"q": "x"}, "c1")
        assert seen == [("lookup", {"q": "x", "note": "n"})] and kwargs == {"q": "x", "note": "n"}
        with pytest.raises(TypeError, match="unexpected keyword argument 'note'"):
            copy.func(*args, **kwargs)
        assert ran == []

    def test_the_innermost_callable_is_the_last_one_the_kernels_walk_returns(self, example):
        """The check reads the last level of ``reached_parameters``. A class-based decorator
        built with ``functools.update_wrapper`` has a ``__call__(*args, **kwargs)`` and a
        ``__wrapped__``: the function it wraps must stay the last level, or this ordinary
        tool would be refused."""
        import functools

        from langchain_core.tools import StructuredTool

        ran, seen = [], []

        class Counted:
            def __init__(self, fn):
                functools.update_wrapper(self, fn)
                self.fn = fn

            def __call__(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        def lookup(key: str, limit: int = 10) -> str:
            ran.append((key, limit))
            return "found"

        levels = _adapter()._reached(Counted(lookup))
        assert list(levels[-1]) == ["key", "limit"]
        tool = StructuredTool.from_function(func=Counted(lookup), name="lookup", description="d")
        messages = self.run(
            example, "sync", [tool], [self.call("lookup", {"key": "k"})], self.no_delete(seen)
        )
        assert messages[0].status == "success", messages[0].content
        assert seen == [("lookup", {"key": "k", "limit": 10})] and ran == [("k", 10)]

    # --- F-C: a schemaless BaseTool whose async method hands the call to _run ---------------

    @staticmethod
    def export_tools(ran):
        from langchain_core.tools import BaseTool

        class Export(BaseTool):
            name: str = "export"
            description: str = "Export rows."

            def _run(self, table: str, limit: int = 1000000) -> str:
                ran.append(("export", table, limit))
                return "exported"

            async def _arun(self, *args: Any, **kwargs: Any) -> str:
                return await asyncio.to_thread(self._run, *args, **kwargs)

        class Count(BaseTool):
            name: str = "count"
            description: str = "Count rows."

            def _run(self, table: str, limit: int = 10) -> str:
                ran.append(("sync", table, limit))
                return "counted"

            async def _arun(self, table: str, limit: int = 99) -> str:
                ran.append(("async", table, limit))
                return "counted"

        return [Export(), Count()]

    @staticmethod
    def limit_under(cap, seen):
        from recusal import Finding

        def policy(name, args):
            seen.append((name, dict(args)))
            if args.get("limit", 0) > cap:
                return [Finding.fail("limit", severity="CRITICAL", message="limit too high")]
            return []

        return policy

    def test_an_arun_that_hands_the_call_to_run_is_refused_on_the_async_route(self, example):
        """The schema is inferred from ``_run``; ``_arun(*args, **kwargs)`` names none of it,
        so the default ``_run`` fills (``limit=1000000``) ran unseen on ``ainvoke``."""
        ran, seen = [], []
        calls = [self.call("export", {"table": "users"})]
        policy = self.limit_under(1000, seen)
        messages = self.run(example, "sync", self.export_tools(ran), calls, policy)
        assert seen == [("export", {"table": "users", "limit": 1000000})] and ran == []
        assert "limit too high" in messages[0].content
        seen.clear()
        messages = self.run(example, "async", self.export_tools(ran), calls, policy)
        assert seen == [] and ran == [], "refused before the policy, and not run"
        assert messages[0].status == "error"
        assert self.HANDS_ON.format("limit") in messages[0].content

    def test_negative_control_ungated_the_async_route_runs_the_unseen_limit(self, example):
        ran = []
        calls = [self.call("export", {"table": "users"})]
        self.run(example, "async", self.export_tools(ran), calls, None, gated=False)
        assert ran == [("export", "users", 1000000)]

    @ROUTES
    def test_a_schemaless_tool_is_completed_from_the_method_its_route_runs(self, example, route):
        """With ``_run`` and ``_arun`` both defined, each route is completed from, and
        runs, its own method's default."""
        ran, seen = [], []
        calls = [self.call("count", {"table": "users"})]
        messages = self.run(
            example, route, self.export_tools(ran), calls, self.limit_under(1000, seen)
        )
        assert messages[0].status == "success", messages[0].content
        limit = 99 if route == "async" else 10
        assert seen == [("count", {"table": "users", "limit": limit})]
        assert ran == [(route, "users", limit)]

    # --- F-A: any failure of the kernel's walk is a refusal ---------------------------------

    @pytest.mark.parametrize("kind", [ValueError, RuntimeError])
    def test_a_walk_failure_of_a_kind_the_gate_does_not_know_is_a_refusal(
        self, example, monkeypatch, kind
    ):
        """The kernel decides what its walk refuses. Whatever it raises, the adapter turns
        into a refusal before the policy; it does not need to know the type."""

        class AppliesAnotherDecoratorInside(kind):
            pass

        def unreadable(func):
            raise AppliesAnotherDecoratorInside("a wrapper holds another wrapper of its function")

        ran, seen = [], []
        monkeypatch.setattr(_adapter(), "_reached_parameters", unreadable)
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})], policy=self.no_delete(seen)
        )
        assert executed == [] and seen == [] and ran == []
        assert messages[0].status == "error"
        assert "cannot be inspected for its parameters (AppliesAnotherDecoratorInside)" in (
            messages[0].content
        )
        assert "a wrapper holds another wrapper" not in messages[0].content

    # --- the refusals the kernel's walk makes by name are reported as what they are ---------
    #
    # ``recusal`` 0.15.0, the oldest kernel this package allows, finds neither shape; a
    # later kernel raises ``HiddenWrapper`` and ``DispatchedAtCallTime``. Each test holds
    # both outcomes, so it runs beside either kernel.

    @staticmethod
    def hidden_wrapper_tool(ran):
        import functools

        from langchain_core.tools import tool

        def with_timeout(fn):
            @functools.wraps(fn)
            def wrapper(*args, timeout: int = 30, **kwargs):
                return fn(*args, **kwargs)  # keeps the judged timeout

            return wrapper

        def robust(fn):
            inner = with_timeout(fn)  # not on the ``__wrapped__`` chain of what is returned

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return inner(*args, **kwargs)

            return wrapper

        @tool
        @robust
        def fetch(url: str, timeout: int = 5) -> str:
            """Fetch a url."""
            ran.append((url, timeout))
            return "fetched"

        return fetch

    @ROUTES
    def test_a_decorator_that_holds_a_second_wrapper_is_refused_with_its_cause(
        self, example, route
    ):
        """A decorator that applies another wrapper inside itself: the policy would judge
        ``timeout=9999`` and the function run with its own ``5``. The reason says what the
        kernel found, names the parameter and never a value, and says what to do."""
        from recusal import invocation

        ran, seen = [], []
        calls = [self.call("fetch", {"url": "u", "timeout": 9999})]
        messages = self.run(
            example, route, [self.hidden_wrapper_tool(ran)], calls, self.no_delete(seen)
        )
        content = messages[0].content
        if hasattr(invocation, "HiddenWrapper"):
            assert ran == [] and seen == []
            assert messages[0].status == "error"
            assert "holds a second wrapper of that function" in content
            assert "`timeout` is taken by a wrapper and by the function it wraps" in content
            assert "call the decorated function from a plain function" in content
            assert "cannot be inspected" not in content and "descriptor" not in content
            assert "9999" not in content and "30" not in content
        else:  # the README's limit beside the released kernel
            assert seen == [("fetch", {"url": "u", "timeout": 9999})] and ran == [("u", 5)]

    @staticmethod
    def dispatching_tool(ran):
        import functools

        from langchain_core.tools import tool

        @functools.singledispatch
        def lookup(key: str, deep: bool = False) -> str:
            """Look a key up."""
            ran.append(("base", key, deep))
            return "found"

        @lookup.register
        def _(key: int, deep: bool = False, skip_review: bool = True) -> str:
            ran.append(("int", key, deep, skip_review))
            return "found"

        return tool(lookup)

    @ROUTES
    def test_a_single_dispatch_function_is_refused_as_one(self, example, route):
        """An implementation registered beside the base runs for an argument of its type,
        with a default of its own (``skip_review=True``) that no policy saw."""
        from recusal import invocation

        if not hasattr(invocation, "DispatchedAtCallTime"):
            # beside 0.15.0 the gate does not refuse it; the call then fails in Python,
            # which wants a single-dispatch function's first argument by position
            return
        ran, seen = [], []
        calls = [self.call("lookup", {"key": "k"})]
        messages = self.run(
            example, route, [self.dispatching_tool(ran)], calls, self.no_delete(seen)
        )
        content = messages[0].content
        assert ran == [] and seen == []
        assert messages[0].status == "error"
        assert "is a single-dispatch function" in content
        assert "chosen at call time, by the type of an argument" in content
        assert "call it from a plain function and give the tool that" in content
        assert "cannot be inspected" not in content and "descriptor" not in content

    @ROUTES
    def test_a_partialmethod_over_a_callable_object_is_refused(self, example, route):
        """``functools.partialmethod`` over a callable object shows the object's parameters
        less what is bound, and nothing under the object was read: the wrapper there ran
        its own ``force=True``. A kernel later than 0.15.0 refuses it as a callable that
        cannot be inspected. (Over a plain function the same ``partialmethod`` is a
        ``functools.partial``, which the kernel reads in full; ``ToolNode`` itself cannot
        take one as a tool's function on this Python, so that half is the kernel's test.)"""
        import functools

        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        from recusal import invocation

        ran, seen = [], []

        def forcing(function):
            @functools.wraps(function)
            def wrapper(*args, force=True, **kwargs):
                ran.append(("force", force))
                return function(*args, **kwargs)

            return wrapper

        class Typed:
            def __init__(self, function):
                functools.update_wrapper(self, function)
                self.function = function

            def __call__(self, account, table: str) -> str:
                return self.function(account, table)

        def drop(account, table: str) -> str:
            ran.append(("dropped", table))
            return "dropped"

        class Account:
            over_object = functools.partialmethod(Typed(forcing(drop)))

        class Args(BaseModel):
            table: str

        def as_tool(func):
            return StructuredTool.from_function(
                func=func, name="drop_table", description="d", args_schema=Args
            )

        calls = [self.call("drop_table", {"table": "t"})]
        tools = [as_tool(Account().over_object)]
        messages = self.run(example, route, tools, calls, self.no_delete(seen))
        if hasattr(invocation, "HiddenWrapper"):  # the kernel that also reads the mark
            assert ran == [] and seen == []
            assert messages[0].status == "error"
            assert "cannot be inspected for its parameters (TypeError)" in messages[0].content
        else:
            assert seen == [("drop_table", {"table": "t"})]
            assert ran == [("force", True), ("dropped", "t")]

    def test_beside_a_kernel_without_the_named_refusals_the_walk_still_refuses(
        self, example, monkeypatch
    ):
        """The package allows ``recusal`` 0.15.0, which has neither exception: the adapter
        then handles a stand-in that nothing raises. And an adapter that does not know a
        refusal by name still refuses it, as any other failure of the walk."""
        from recusal import invocation

        adapter = _adapter()

        class Released:  # the names 0.15.0 has
            DeclaredSignature = invocation.DeclaredSignature

        class Odd:
            HiddenWrapper = "not an exception class"
            DispatchedAtCallTime = int

        for name in ("HiddenWrapper", "DispatchedAtCallTime"):
            assert adapter._walk_refusal(Released, name) is adapter._NotInThisKernel
            assert adapter._walk_refusal(Odd, name) is adapter._NotInThisKernel
            expected = getattr(invocation, name, adapter._NotInThisKernel)
            assert adapter._walk_refusal(invocation, name) is expected
            assert getattr(adapter, "_" + name) is expected
        assert adapter._walk_refusal(Released, "DeclaredSignature") is invocation.DeclaredSignature

        monkeypatch.setattr(adapter, "_HiddenWrapper", adapter._NotInThisKernel)
        monkeypatch.setattr(adapter, "_DispatchedAtCallTime", adapter._NotInThisKernel)
        if not hasattr(invocation, "HiddenWrapper"):
            return
        for build, name, args in (
            (self.hidden_wrapper_tool, "fetch", {"url": "u", "timeout": 9999}),
            (self.dispatching_tool, "lookup", {"key": "k"}),
        ):
            ran, seen = [], []
            messages = self.run(
                example, "sync", [build(ran)], [self.call(name, args)], self.no_delete(seen)
            )
            assert ran == [] and seen == [] and messages[0].status == "error"
            assert "cannot be inspected for its parameters" in messages[0].content

    def test_a_class_based_decorator_that_is_a_descriptor_is_refused_with_what_to_do(self, example):
        """``cachetools.cached`` returns such an object. ``inspect.signature`` takes an
        object whose class defines ``__get__`` for a builtin and finds no signature."""
        import functools
        import types

        from langchain_core.tools import StructuredTool, tool

        ran, seen = [], []

        class Cached:
            def __init__(self, fn):
                functools.update_wrapper(self, fn)
                self.fn = fn

            def __call__(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

            def __get__(self, obj, objtype=None):
                return self if obj is None else types.MethodType(self, obj)

        @Cached
        def lookup(key: str) -> str:
            """Look a key up."""
            ran.append(key)
            return "found"

        calls = [self.call("lookup", {"key": "k"})]
        messages = self.run(example, "sync", [tool(lookup)], calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert "cannot be inspected for its parameters (ValueError)" in messages[0].content
        assert "call the decorated function from a plain function" in messages[0].content
        # negative control: ungated it runs, and the README's advice runs gated
        self.run(example, "sync", [tool(lookup)], calls, None, gated=False)
        assert ran == ["k"]

        def plain(key: str) -> str:
            return lookup(key)

        advised = StructuredTool.from_function(func=plain, name="lookup", description="d")
        messages = self.run(example, "sync", [advised], calls, self.no_delete(seen))
        assert messages[0].status == "success" and seen == [("lookup", {"key": "k"})]

    def test_a_schema_the_gate_cannot_read_is_refused(self, example):
        """A Pydantic v1 ``args_schema`` declares no fields the gate reads. A function
        that takes only ``**kwargs`` would otherwise pass every check against it."""
        import pydantic.v1
        from langchain_core.tools import StructuredTool

        ran, seen = [], []

        class V1Args(pydantic.v1.BaseModel):
            q: str

        def loose(**kwargs):
            ran.append(kwargs)
            return "ok"

        tool = StructuredTool(name="v1", description="d", func=loose, args_schema=V1Args)
        messages = self.run(
            example, "sync", [tool], [self.call("v1", {"q": "x"})], self.no_delete(seen)
        )
        assert ran == [] and seen == []
        assert "neither a Pydantic v2 model nor a JSON schema dict" in messages[0].content

    # --- the review's surviving mutations, each pinned ----------------------------------------

    def test_a_keyword_default_on_a_decorator_in_the_middle_of_a_stack_is_refused(self, example):
        """Pins M26: the absent-from-schema refusal applied only to the first callable."""
        import functools

        from langchain_core.tools import tool

        ran, seen = [], []

        def logged(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def confirmable(fn):
            @functools.wraps(fn)
            def wrapper(*args, force=True, **kwargs):
                ran.append(("force", force))
                return fn(*args, **kwargs)

            return wrapper

        @tool
        @logged
        @confirmable
        @logged
        def drop_table(table: str) -> str:
            """Drop a table."""
            ran.append(("drop_table", table))
            return "dropped"

        calls = [self.call("drop_table", {"table": "t"})]
        messages = self.run(example, "sync", [drop_table], calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert "the callable's parameter `force` is absent from the argument schema" in (
            messages[0].content
        )
        self.run(example, "sync", [drop_table], calls, None, gated=False)
        assert ran == [("force", True), ("drop_table", "t")]

    @pytest.mark.parametrize("shape", ["inner_adds", "wrapper_in_front", "inner_forwards"])
    def test_a_single_input_tool_is_held_to_every_callable_it_reaches(self, example, shape):
        """Pins M24 (another parameter looked for only on the first callable) and N48 (the
        input's name taken from the innermost callable, so a ``*args`` decorator in front
        went unnoticed), and holds a single input to the innermost callable too."""
        import functools

        from langchain_core.tools import Tool

        ran, seen = [], []

        if shape == "inner_adds":

            def inner(q, mode="DELETE"):
                ran.append((q, mode))
                return "found"

            @functools.wraps(inner)
            def lookup(q):
                return inner(q)

            expected = "the callable's parameter `mode` is not the tool's single input"
        elif shape == "wrapper_in_front":

            def inner(q):
                ran.append(q)
                return "found"

            @functools.wraps(inner)
            def lookup(*args, **kwargs):
                return inner(*args, **kwargs)

            expected = "does not take its input as a first named parameter"
        else:

            def inner(*parts, **options):
                ran.append(parts)
                return "found"

            @functools.wraps(inner)
            def lookup(q):
                return inner(q)

            expected = self.HANDS_ON.format("q")

        tool = Tool(name="lookup", func=lookup, description="Look up.")
        calls = [self.call("lookup", {"__arg1": "x"})]
        messages = self.run(example, "sync", [tool], calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert messages[0].status == "error" and expected in messages[0].content
        self.run(example, "sync", [tool], calls, None, gated=False)
        assert ran != []

    def test_a_single_input_function_with_kwargs_that_names_its_input_runs(self, example):
        from langchain_core.tools import Tool

        ran, seen = [], []

        def lookup(q, **options):
            ran.append((q, options))
            return "found"

        tool = Tool(name="lookup", func=lookup, description="Look up.")
        calls = [self.call("lookup", {"__arg1": "x"})]
        messages = self.run(example, "sync", [tool], calls, self.no_delete(seen))
        assert messages[0].status == "success", messages[0].content
        assert seen == [("lookup", {"q": "x"})] and ran == [("x", {})]

    @staticmethod
    def request(tool, name="run_sql", args=None, ident="c1"):
        from langgraph.prebuilt.tool_node import ToolCallRequest

        call = {"name": name, "args": args or {"sql": SELECT}, "id": ident, "type": "tool_call"}
        return ToolCallRequest(tool_call=call, tool=tool, state={}, runtime=None)

    def test_a_refusal_before_the_judging_copy_is_answered_under_the_calls_id(self, example):
        """Pins M30: ``refuse()`` could answer every such call with an empty id."""
        seen = []
        gate = example.recusal_gate(self.no_delete(seen))

        def never(request):
            raise AssertionError("a refused call is not executed")

        unknown = gate(self.request(None, name="nope"), never)
        misnamed = gate(self.request(example.run_sql, name="Run_SQL"), never)
        assert seen == []
        for answer in (unknown, misnamed):
            assert answer.status == "error" and answer.tool_call_id == "c1"
        assert "no tool by that name" in unknown.content
        assert "names another tool" in misnamed.content

    def test_the_node_helpers_read_a_coroutine_for_a_parameter_named_config(self, example):
        """Pins N38: ``_checked`` read ``func`` only, so a coroutine-only tool was built and
        then refused on every call."""
        from langchain_core.tools import StructuredTool

        async def setup(name: str, config: str = "safe") -> str:
            return "ok"

        tool = StructuredTool.from_function(coroutine=setup, name="setup", description="d")
        assert tool.func is None
        with pytest.raises(TypeError, match="tool `setup`: its function's parameter `config`"):
            example.gated_tool_node([tool], lambda name, args: [])

    def test_a_tool_that_overrides_ainvoke_is_refused_on_both_routes(self, example):
        """Pins N51: ``ainvoke`` could be dropped from the lifecycle the gate checks, and a
        tool that overrides it reaches its body on the async route without the parse step."""
        from langchain_core.messages import ToolMessage
        from langchain_core.tools import BaseTool

        ran = []

        class Direct(BaseTool):
            name: str = "direct"
            description: str = "Overrides only ainvoke."

            def _run(self, sql: str) -> str:
                ran.append(sql)
                return "recorded"

            async def ainvoke(self, input, config=None, **kwargs):
                content = self._run(**input["args"])
                return ToolMessage(content=content, tool_call_id=input["id"], name=self.name)

        calls = [self.call("direct", {"sql": DELETE})]
        for route in ("sync", "async"):
            messages = self.run(example, route, [Direct()], calls, self.no_delete([]))
            assert ran == [], route
            assert "Direct overrides `ainvoke`" in messages[0].content
        self.run(example, "async", [Direct()], calls, None, gated=False)
        assert ran == [DELETE]

    # --- F-D: the answer is the judged execution's ----------------------------------------------

    @staticmethod
    def replica():
        from langchain_core.tools import tool

        ran = []

        @tool
        def run_sql_replica(sql: str) -> str:
            """Run SQL on the replica."""
            ran.append(sql)
            return "replica ran " + sql

        return run_sql_replica, ran

    def falling_back(self, example, route, policy, other, when):
        """A node whose gate is followed by a step that runs ``other`` in the tool's place:
        ``when="error"`` after an error answer, ``when="always"`` after any answer."""
        gates = example.recusal_gates(policy)

        def again(first):
            return when == "always" or getattr(first, "status", None) == "error"

        def after(request, execute):
            def fallback(judged):
                first = execute(judged)
                return execute(judged.override(tool=other)) if again(first) else first

            return gates["wrap_tool_call"](request, fallback)

        async def aafter(request, execute):
            async def fallback(judged):
                first = await execute(judged)
                return await execute(judged.override(tool=other)) if again(first) else first

            return await gates["awrap_tool_call"](request, fallback)

        return example.ToolNode(
            [example.echo, example.run_sql, other], wrap_tool_call=after, awrap_tool_call=aafter
        )

    def in_graph(self, example, route, node, calls):
        example.EXECUTED.clear()
        if route == "async":
            return asyncio.run(example.run_in_graph_async(node, calls))
        return example.run_in_graph(node, calls)

    @ROUTES
    def test_a_refused_call_cannot_come_back_as_another_tools_answer(self, example, route):
        """The policy refused ``run_sql``; a step after the gate then ran the same DELETE
        on another tool and returned its success. The judging step had been entered once,
        so that answer was passed on."""
        seen = []
        other, ran = self.replica()
        node = self.falling_back(example, route, self.no_delete(seen), other, "error")
        messages = self.in_graph(example, route, node, [self.call("run_sql", {"sql": DELETE})])
        assert seen == [("run_sql", {"sql": DELETE})]
        assert ran == [DELETE] and example.EXECUTED == [], "the other tool ran; that is done"
        assert messages[0].status == "error" and messages[0].name == "run_sql"
        assert messages[0].tool_call_id == "c1"
        assert "replica ran" not in messages[0].content
        assert "did not allow the call" in messages[0].content
        assert "its result is withheld" in messages[0].content

    @ROUTES
    def test_an_answer_under_another_tools_name_is_withheld(self, example, route):
        """The judged call was allowed and ran; a step after the gate then ran another tool
        and returned that tool's answer in its place."""
        seen = []
        other, ran = self.replica()
        node = self.falling_back(example, route, self.no_delete(seen), other, "always")
        messages = self.in_graph(example, route, node, [self.call("run_sql", {"sql": SELECT})])
        assert seen == [("run_sql", {"sql": SELECT})]
        assert example.EXECUTED == [("run_sql", {"sql": SELECT})] and ran == [SELECT]
        assert messages[0].status == "error" and messages[0].name == "run_sql"
        assert "replica ran" not in messages[0].content
        assert "under the name of another tool than the one the gate judged" in (
            messages[0].content
        )
        assert "run_sql_replica" not in messages[0].content

    def test_a_tool_that_names_its_own_message_otherwise_is_withheld_as_the_readme_says(
        self, example
    ):
        """The gate cannot tell who made a ``ToolMessage`` under another name. A tool that
        builds its own must leave the name out or use the tool's; both of those pass."""
        from typing import Annotated

        from langchain_core.messages import ToolMessage
        from langchain_core.tools import InjectedToolCallId, tool

        def answering(tool_name, **named):
            @tool(tool_name)
            def answer(sql: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> ToolMessage:
                """Answer with a message of its own."""
                return ToolMessage(content="own " + sql, tool_call_id=tool_call_id, **named)

            return answer

        tools = [
            answering("unnamed"),
            answering("named", name="named"),
            answering("misnamed", name="something_else"),
        ]
        calls = [self.call(t.name, {"sql": SELECT}, t.name) for t in tools]
        messages = self.run(example, "sync", tools, calls, self.no_delete([]))
        by_id = {m.tool_call_id: m for m in messages}
        assert by_id["unnamed"].content == by_id["named"].content == "own " + SELECT
        assert by_id["misnamed"].status == "error"
        assert "under the name of another tool" in by_id["misnamed"].content

    def test_what_fits_the_judging_step_is_passed_on_and_nothing_else(self, example):
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command

        settled = _adapter()._settled

        def answer(content, status="success", name="run_sql"):
            return ToolMessage(content=content, tool_call_id="c1", name=name, status=status)

        reason = "Recusal refused `run_sql` [FAIL]: no delete"
        # never entered
        assert "never judged" in settled(answer("ran"), [], "run_sql")
        # allowed: the tool's answer, an error the node made of the tool's exception, a
        # message the tool built itself without a name, a Command
        for result in (
            answer("ran"),
            answer("boom", status="error"),
            answer("ran", name=None),
            Command(update={}),
        ):
            assert settled(result, [True], "run_sql") is None
        assert "another tool" in settled(answer("ran", name="other"), [True], "run_sql")
        # refused: only the refusal itself, bare or with a note in front of it
        assert settled(answer(reason, status="error"), [reason], "run_sql") is None
        assert settled(answer("Note. " + reason, status="error"), [reason], "run_sql") is None
        for result in (
            answer(reason),
            answer("another error", status="error"),
            answer([{"type": "text", "text": reason}], status="error"),
            answer([reason], status="error"),
            Command(update={}),
        ):
            assert "did not allow the call" in settled(result, [reason], "run_sql")
        # what the step did last is what counts
        assert settled(answer(reason, status="error"), [True, reason], "run_sql") is None
        assert settled(answer("ran"), [reason, True], "run_sql") is None
        unsettled = answer("boom", status="error")
        assert "did not allow the call" in settled(unsettled, [True, None], "run_sql")

    def test_a_retry_of_the_judging_copy_is_judged_again_and_its_answer_passed_on(self, example):
        """A step after the gate that calls the same request again reaches the judging
        copy again: nothing is withheld."""
        seen = []
        gate = example.recusal_gate(self.no_delete(seen))

        def retrying(request, execute):
            def twice(judged):
                execute(judged)
                return execute(judged)

            return gate(request, twice)

        node = example.ToolNode([example.echo, example.run_sql], wrap_tool_call=retrying)
        for sql, status in ((SELECT, "success"), (DELETE, "error")):
            seen.clear()
            messages = self.in_graph(example, "sync", node, [self.call("run_sql", {"sql": sql})])
            assert seen == [("run_sql", {"sql": sql})] * 2
            assert messages[0].status == status and "withheld" not in messages[0].content

    # --- an injected name ToolNode does not fill is never the host's --------------------------

    @staticmethod
    def disagreeing_tool(ran):
        """The function marks ``secret`` injected; the schema declares it as an ordinary
        argument, and ToolNode reads the schema: it neither strips nor fills ``secret``."""
        from typing import Annotated

        from langchain_core.tools import InjectedToolArg, StructuredTool
        from pydantic import BaseModel

        class Args(BaseModel):
            x: int
            secret: str = "safe"

        def act(x: int, secret: Annotated[str, InjectedToolArg] = "safe") -> str:
            ran.append((x, secret))
            return "acted"

        return StructuredTool.from_function(func=act, name="act", description="d", args_schema=Args)

    @ROUTES
    @pytest.mark.parametrize("sent", [{"x": 1, "secret": "a"}, {"x": 1}], ids=["sent", "unsent"])
    def test_a_value_written_after_the_gate_under_an_injected_name_is_judged(
        self, example, route, sent
    ):
        """A hook between the gate and the tool rewrote ``secret``. It arrived changed, so
        it was read as the host's and left out of judgment (judged ``{'x': 1}``, ran
        ``secret='DELETE'``). ToolNode fills state, store and runtime only; a value under
        any other injected name is judged."""
        ran, seen = [], []
        gates = example.recusal_gates(self.no_delete(seen))

        def rewritten(judged):
            call = dict(judged.tool_call)
            call["args"] = {**call["args"], "secret": "DELETE"}
            return judged.override(tool_call=call)

        def after(request, execute):
            return gates["wrap_tool_call"](request, lambda judged: execute(rewritten(judged)))

        async def aafter(request, execute):
            async def rewriting(judged):
                return await execute(rewritten(judged))

            return await gates["awrap_tool_call"](request, rewriting)

        node = example.ToolNode(
            [self.disagreeing_tool(ran)], wrap_tool_call=after, awrap_tool_call=aafter
        )
        messages = self.in_graph(example, route, node, [self.call("act", sent)])
        assert seen == [("act", {"x": 1, "secret": "DELETE"})]
        assert ran == [] and messages[0].status == "error"

    def test_a_value_toolnode_fills_is_still_the_hosts_whatever_a_hook_wrote_there(self, example):
        """ToolNode removes what a hook wrote under a state parameter and puts the state
        there: the host's value runs, and it is left out of judgment."""
        from typing import Annotated

        from langchain_core.tools import tool
        from langgraph.prebuilt import InjectedState

        ran, seen = [], []

        @tool
        def act(x: int, state: Annotated[dict, InjectedState]) -> str:
            """Act on the graph state."""
            ran.append((x, sorted(state)))
            return "acted"

        gate = example.recusal_gate(self.no_delete(seen))

        def after(request, execute):
            def rewriting(judged):
                call = dict(judged.tool_call)
                call["args"] = {**call["args"], "state": "DELETE"}
                return execute(judged.override(tool_call=call))

            return gate(request, rewriting)

        node = example.ToolNode([act], wrap_tool_call=after)
        messages = self.in_graph(example, "sync", node, [self.call("act", {"x": 1})])
        assert messages[0].status == "success", messages[0].content
        assert seen == [("act", {"x": 1})] and ran == [(1, ["messages"])]
        adapter = _adapter()
        assert adapter._node_fills(act) == {"state"}
        assert adapter._node_fills(self.disagreeing_tool([])) == frozenset()
        assert adapter._host_keys(self.disagreeing_tool([])) == {"secret"}

    def test_a_value_the_parse_step_writes_itself_under_a_state_name_is_judged(self, example):
        """The host's keys are read before the tool's own parse step runs. A tool whose
        parse step writes a value under a state parameter the host did not fill (the state
        has no ``role``) wrote that value itself: it is judged, not taken for the host's."""
        from typing import Annotated

        from langchain_core.tools import StructuredTool
        from langgraph.prebuilt import InjectedState

        ran, seen = [], []

        class WritesItself(StructuredTool):
            def _parse_input(self, tool_input, tool_call_id):
                tool_input["role"] = "DELETE"
                return super()._parse_input(tool_input, tool_call_id)

        def act(what: str, role: Annotated[str, InjectedState("role")] = "viewer") -> str:
            ran.append((what, role))
            return "acted"

        tool = WritesItself.from_function(func=act, name="act", description="d")
        assert _adapter()._node_fills(tool) == {"role"}
        call = self.call("act", {"what": "x"})
        copy = _adapter()._judging(tool, call, self.no_delete(seen), {}, False)
        with pytest.raises(example.RecusalRefusal, match="no delete"):
            copy._to_args_and_kwargs({"what": "x"}, "c1")
        assert seen == [("act", {"what": "x", "role": "DELETE"})] and ran == []

    def test_the_names_toolnode_fills_that_cannot_be_read_refuse_the_call(
        self, example, monkeypatch
    ):
        adapter = _adapter()
        real = adapter._get_all_injected_args
        seen = []

        class WithoutState:
            def __init__(self, injected):
                self.all_injected_keys = injected.all_injected_keys
                self.store, self.runtime = injected.store, injected.runtime

        monkeypatch.setattr(
            adapter, "_get_all_injected_args", lambda tool: WithoutState(real(tool))
        )
        gate = example.recusal_gate(self.no_delete(seen))
        answer = gate(self.request(example.run_sql), lambda judged: "ran")
        assert seen == [] and answer.status == "error"
        assert "cannot be read from the installed LangChain and LangGraph (AttributeError)" in (
            answer.content
        )

    # --- the one exception: the function langchain-mcp-adapters defines to call a server ----

    MCP_SCHEMA = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "depth": {"type": "integer", "default": 2}},
        "required": ["path"],
    }

    @staticmethod
    def mcp_library(monkeypatch, holds_interceptors=True):
        """A stand-in for ``langchain_mcp_adapters.tools`` with the library's own structure
        (0.2.2 to 0.3.2): a builder ``convert_mcp_tool_to_langchain_tool`` whose nested
        ``call_tool(runtime=None, **arguments)`` hands ``arguments`` to the server through
        its ``tool_interceptors``. The stand-in is put where the library would be imported,
        so the gate's recognition runs as it does against the library. Returns the builder;
        ``received`` collects what the server is handed."""
        import types
        from typing import Annotated

        from langchain_core.tools import InjectedToolArg, StructuredTool

        if holds_interceptors:

            def convert_mcp_tool_to_langchain_tool(received, schema, tool_interceptors=None):
                async def call_tool(
                    runtime: Annotated[object, InjectedToolArg()] = None, **arguments: Any
                ) -> tuple:
                    for interceptor in tool_interceptors or ():
                        arguments = interceptor(arguments)
                    received.append(arguments)
                    return "served", None

                return StructuredTool(
                    name="read_file",
                    description="Read a file.",
                    args_schema=schema,
                    coroutine=call_tool,
                    response_format="content_and_artifact",
                )

        else:

            def convert_mcp_tool_to_langchain_tool(received, schema, tool_interceptors=None):
                async def call_tool(
                    runtime: Annotated[object, InjectedToolArg()] = None, **arguments: Any
                ) -> tuple:
                    received.append(arguments)
                    return "served", None

                return StructuredTool(
                    name="read_file",
                    description="Read a file.",
                    args_schema=schema,
                    coroutine=call_tool,
                    response_format="content_and_artifact",
                )

        module = types.ModuleType("langchain_mcp_adapters.tools")
        module.convert_mcp_tool_to_langchain_tool = convert_mcp_tool_to_langchain_tool
        monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.tools", module)
        return convert_mcp_tool_to_langchain_tool

    def test_a_tool_langchain_mcp_adapters_built_runs_with_exactly_the_judged_arguments(
        self, example, monkeypatch
    ):
        """Its function takes only ``**arguments`` under the server's schema, which the
        hand-on rule refuses. What it hands on is the judged arguments as they stand, to
        the server, so it is the one exception. A default the server applies is not seen:
        the adapter does not fill ``depth``, and the policy is shown the call without it."""
        received, seen = [], []
        build = self.mcp_library(monkeypatch)
        tool = build(received, self.MCP_SCHEMA)
        calls = [
            self.call("read_file", {"path": "/x", "depth": 3}, "both"),
            self.call("read_file", {"path": "/y"}, "omitted"),
            self.call("read_file", {"path": "/z", "mode": "DELETE"}, "refused"),
        ]
        messages = self.run(example, "async", [tool], calls, self.no_delete(seen))
        assert {m.tool_call_id: m.status for m in messages} == {
            "both": "success",
            "omitted": "success",
            "refused": "error",
        }
        judged = sorted((args for _, args in seen), key=str)
        assert judged == [
            {"path": "/x", "depth": 3},
            {"path": "/y"},
            {"path": "/z", "mode": "DELETE"},
        ]
        assert sorted(received, key=str) == [{"path": "/x", "depth": 3}, {"path": "/y"}]
        # the library gives its tools no sync function: refused there, as ungated it raises
        seen.clear(), received.clear()
        messages = self.run(example, "sync", [tool], calls[:1], self.no_delete(seen))
        assert "the tool has no callable for this route" in messages[0].content
        assert seen == [] and received == []

    def test_anything_in_front_of_the_mcp_function_is_held_to_the_ordinary_rule(
        self, example, monkeypatch
    ):
        """The exception is for the library's function as the tool's own callable. A
        forwarder in front of it, with or without ``functools.wraps``, is refused."""
        import functools

        from langchain_core.tools import StructuredTool

        received, seen = [], []
        inner = self.mcp_library(monkeypatch)(received, self.MCP_SCHEMA).coroutine

        async def forwarder(*args, **kwargs):
            return await inner(*args, **kwargs)

        @functools.wraps(inner)
        async def wrapped(*args, **kwargs):
            return await inner(*args, **kwargs)

        for front in (forwarder, wrapped):
            tool = StructuredTool(
                name="read_file",
                description="d",
                args_schema=self.MCP_SCHEMA,
                coroutine=front,
                response_format="content_and_artifact",
            )
            calls = [self.call("read_file", {"path": "/x"})]
            messages = self.run(example, "async", [tool], calls, self.no_delete(seen))
            assert seen == [] and received == []
            assert self.HANDS_ON.format("depth") in messages[0].content

    def test_a_look_alike_of_the_mcp_function_is_refused(self, example, monkeypatch):
        """Same name, same signature, the library's ``__module__`` and ``__qualname__``
        written onto it: it is not the code object the library's builder holds."""
        from langchain_core.tools import StructuredTool

        received, seen = [], []
        real = self.mcp_library(monkeypatch)(received, self.MCP_SCHEMA).coroutine

        async def call_tool(runtime=None, **arguments):
            received.append(arguments)
            return "served", None

        call_tool.__module__ = real.__module__ = "langchain_mcp_adapters.tools"
        call_tool.__qualname__ = real.__qualname__
        tool = StructuredTool(
            name="read_file",
            description="d",
            args_schema=self.MCP_SCHEMA,
            coroutine=call_tool,
            response_format="content_and_artifact",
        )
        calls = [self.call("read_file", {"path": "/x"})]
        messages = self.run(example, "async", [tool], calls, self.no_delete(seen))
        assert seen == [] and received == []
        assert self.HANDS_ON.format("depth") in messages[0].content
        assert _adapter()._calls_an_mcp_server(real) is True
        assert _adapter()._calls_an_mcp_server(call_tool) is False

    def test_the_mcp_function_is_recognised_only_while_the_library_holds_it(
        self, example, monkeypatch
    ):
        received = []
        real = self.mcp_library(monkeypatch)(received, self.MCP_SCHEMA).coroutine
        adapter = _adapter()
        assert adapter._calls_an_mcp_server(real) is True
        # a callable that is not a function, and a builder that is not one
        assert adapter._calls_an_mcp_server(lambda **arguments: None) is False
        assert adapter._calls_an_mcp_server(print) is False
        module = sys.modules["langchain_mcp_adapters.tools"]
        monkeypatch.setattr(module, "convert_mcp_tool_to_langchain_tool", None)
        assert adapter._calls_an_mcp_server(real) is False
        monkeypatch.delitem(sys.modules, "langchain_mcp_adapters.tools")
        assert adapter._calls_an_mcp_server(real) is False

    def test_an_mcp_tool_built_with_interceptors_is_refused_before_the_policy(
        self, example, monkeypatch
    ):
        """A tool interceptor runs between the library's function and the server and can
        replace the arguments (judged ``depth=3``, the server received ``depth=99``)."""
        received, seen = [], []
        build = self.mcp_library(monkeypatch)
        tool = build(received, self.MCP_SCHEMA, [lambda arguments: {**arguments, "depth": 99}])
        calls = [self.call("read_file", {"path": "/x", "depth": 3})]
        messages = self.run(example, "async", [tool], calls, self.no_delete(seen))
        assert seen == [] and received == []
        assert "built by langchain-mcp-adapters with tool interceptors" in messages[0].content
        # negative control: ungated, the server receives what no policy saw
        self.run(example, "async", [tool], calls, None, gated=False)
        assert received == [{"path": "/x", "depth": 99}]
        # an empty list of interceptors is none
        received.clear()
        tool = build(received, self.MCP_SCHEMA, [])
        messages = self.run(example, "async", [tool], calls, self.no_delete(seen))
        assert messages[0].status == "success" and received == [{"path": "/x", "depth": 3}]

    def test_an_mcp_function_that_does_not_show_its_interceptors_is_refused(
        self, example, monkeypatch
    ):
        """A release whose function does not hold ``tool_interceptors`` where the gate
        reads it: the gate cannot tell whether the arguments are sent as judged."""
        received, seen = [], []
        build = self.mcp_library(monkeypatch, holds_interceptors=False)
        tool = build(received, self.MCP_SCHEMA)
        calls = [self.call("read_file", {"path": "/x"})]
        messages = self.run(example, "async", [tool], calls, self.no_delete(seen))
        assert seen == [] and received == []
        assert "does not show the gate whether the tool has interceptors" in messages[0].content

    # --- the refusals the README names, and what it says to use instead ----------------------

    def test_the_readme_advice_for_a_kwargs_only_tool_runs_gated(self, example):
        """A function of your own that names the parameters and calls the tool."""
        from langchain_core.tools import StructuredTool, tool

        ran, seen = [], []

        async def call_tool(runtime=None, **arguments):
            ran.append(arguments)
            return "ok"

        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "depth": {"type": "integer", "default": 2}},
        }
        mcp_tool = StructuredTool(
            name="mcp", description="d", coroutine=call_tool, args_schema=schema
        )

        @tool
        async def read_file(path: str, depth: int = 2) -> str:
            """Read a file."""
            return await mcp_tool.ainvoke({"path": path, "depth": depth})

        calls = [self.call("read_file", {"path": "/x"})]
        messages = self.run(example, "async", [read_file], calls, self.no_delete(seen))
        assert messages[0].status == "success", messages[0].content
        assert seen == [("read_file", {"path": "/x", "depth": 2})]
        assert ran == [{"path": "/x", "depth": 2}]

    def test_a_tool_over_langsmith_traceable_is_refused_and_tracing_inside_it_runs(self, example):
        langsmith = pytest.importorskip("langsmith")
        from langchain_core.tools import tool

        ran, seen = [], []

        @tool
        @langsmith.traceable
        def traced(q: str) -> str:
            """Traced from outside."""
            ran.append(q)
            return "ok"

        @langsmith.traceable
        def work(q: str) -> str:
            ran.append(q)
            return "ok"

        @tool
        def inside(q: str) -> str:
            """Traced from inside."""
            return work(q)

        calls = [self.call("traced", {"q": "x"}, "a"), self.call("inside", {"q": "x"}, "b")]
        messages = self.run(example, "sync", [traced, inside], calls, self.no_delete(seen))
        by_id = {m.tool_call_id: m for m in messages}
        assert "declares a signature that is not its own" in by_id["a"].content
        assert by_id["b"].status == "success", by_id["b"].content
        assert seen == [("inside", {"q": "x"})] and ran == ["x"]

    def test_a_runnable_made_into_a_tool_is_refused_and_a_function_that_calls_it_runs(
        self, example
    ):
        import warnings

        from langchain_core.runnables import RunnableLambda
        from langchain_core.tools import tool
        from pydantic import BaseModel

        seen = []

        def exclaim(text: str) -> str:
            return text + "!"

        shout = RunnableLambda(exclaim)

        class Pair(BaseModel):
            a: int
            b: int

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            of_a_string = shout.as_tool(name="shout", description="d")
            of_a_schema = RunnableLambda(lambda d: d["a"] + d["b"]).as_tool(
                Pair, name="add", description="d"
            )
        with pytest.raises(TypeError, match="tool `shout`: its function's parameter `config`"):
            example.gated_tool_node([of_a_string], self.no_delete(seen))
        calls = [self.call("shout", {"__arg1": "x"}, "a"), self.call("add", {"a": 1, "b": 2}, "b")]
        messages = self.run(
            example, "sync", [of_a_string, of_a_schema], calls, self.no_delete(seen)
        )
        by_id = {m.tool_call_id: m.content for m in messages}
        assert seen == []
        assert "the callable's parameter `config` is not the tool's single input" in by_id["a"]
        assert self.HANDS_ON.format("a") in by_id["b"]

        @tool("shout")
        def advised(text: str) -> str:
            """Shout."""
            return shout.invoke(text)

        messages = self.run(
            example, "sync", [advised], [self.call("shout", {"text": "x"})], self.no_delete(seen)
        )
        assert messages[0].content == "x!" and seen == [("shout", {"text": "x"})]

    def test_a_single_input_tool_over_a_builtin_is_refused_and_a_wrapped_one_runs(self, example):
        from langchain_core.tools import Tool

        seen = []
        builtin = Tool(name="upper", func=str.upper, description="d")
        wrapped = Tool(name="upper", func=lambda text: text.upper(), description="d")
        calls = [self.call("upper", {"__arg1": "x"})]
        messages = self.run(example, "sync", [builtin], calls, self.no_delete(seen))
        assert seen == []
        assert "does not take its input as a first named parameter" in messages[0].content
        messages = self.run(example, "sync", [wrapped], calls, self.no_delete(seen))
        assert messages[0].content == "X" and seen == [("upper", {"text": "x"})]


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestForwardersAndTheNamesLangChainKeeps:
    """Review round 3 of the package (2026-11): a forwarder that takes only ``*args`` and
    ``**kwargs`` under the schema LangChain infers from it, refusal reasons that were not
    true of the call they refused, and five checks no test pinned. Each test names its
    finding."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_langgraph_round3_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    call = staticmethod(TestWhatRunsIsWhatWasJudged.call)
    no_delete = staticmethod(TestWhatRunsIsWhatWasJudged.no_delete)
    run = TestWhatRunsIsWhatWasJudged.run
    ROUTES = TestWhatRunsIsWhatWasJudged.ROUTES
    limit_under = staticmethod(TestWhatTheGateCannotReadIsRefused.limit_under)
    in_graph = TestWhatTheGateCannotReadIsRefused.in_graph
    mcp_library = staticmethod(TestWhatTheGateCannotReadIsRefused.mcp_library)
    MCP_SCHEMA = TestWhatTheGateCannotReadIsRefused.MCP_SCHEMA

    ONLY_VARIADICS = "the tool's innermost callable takes only `*args` or `**kwargs`"

    # --- C1: a forwarder under the schema LangChain infers from it ----------------------------

    @staticmethod
    def forwarded(ran, shape, is_async):
        """``export_rows(table="users", limit=1000000)`` behind a decorator built without
        ``functools.wraps`` (it copies the name and the docstring by hand), or behind a
        ``functools.wraps`` decorator over that one. ``@tool`` infers the schema from the
        forwarder: its fields are ``args`` and ``kwargs``."""
        import functools

        from langchain_core.tools import tool

        if is_async:

            async def export_rows(table: str = "users", limit: int = 1000000) -> str:
                """Export the rows of a table."""
                ran.append((table, limit))
                return "exported"

            def timed(fn):
                async def wrapper(*args, **kwargs):
                    return await fn(*args, **kwargs)

                wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
                return wrapper

        else:

            def export_rows(table: str = "users", limit: int = 1000000) -> str:
                """Export the rows of a table."""
                ran.append((table, limit))
                return "exported"

            def timed(fn):
                def wrapper(*args, **kwargs):
                    return fn(*args, **kwargs)

                wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
                return wrapper

        forwarder = timed(export_rows)
        inner = forwarder
        if shape == "wraps_over_it":
            if is_async:

                @functools.wraps(forwarder)
                async def outer(*args, **kwargs):
                    return await forwarder(*args, **kwargs)

            else:

                @functools.wraps(forwarder)
                def outer(*args, **kwargs):
                    return forwarder(*args, **kwargs)

            inner = outer
        built = tool(inner)
        assert set(built.args) == {"args", "kwargs"}
        return built

    @pytest.mark.parametrize("shape", ["forwarder", "wraps_over_it"])
    @ROUTES
    def test_a_forwarder_under_the_schema_inferred_from_it_is_refused(self, example, route, shape):
        """The schema's fields were the forwarder's own ``args`` and ``kwargs``, so every
        field counted as a parameter: the policy judged ``{}`` and the function behind the
        forwarder ran ``limit=1000000``."""
        ran, seen = [], []
        tool = self.forwarded(ran, shape, route == "async")
        calls = [self.call("export_rows", {})]
        messages = self.run(example, route, [tool], calls, self.limit_under(1000, seen))
        assert ran == [] and seen == [], "refused before the policy, and not run"
        assert messages[0].status == "error"
        assert self.ONLY_VARIADICS in messages[0].content
        assert "build a decorator with `functools.wraps`" in messages[0].content
        # negative control: ungated, the default nobody judged runs
        self.run(example, route, [tool], calls, None, gated=False)
        assert ran == [("users", 1000000)]

    def test_a_function_of_only_kwargs_is_refused_whatever_its_schema_declares(self, example):
        """``@tool def f(**kwargs)``: the inferred schema declares ``kwargs`` itself. The
        gate cannot tell it from a forwarder, under that schema or under one that declares
        nothing."""
        from langchain_core.tools import StructuredTool, tool

        ran, seen = [], []

        @tool
        def loose(**kwargs: Any) -> str:
            """Loose."""
            ran.append(kwargs)
            return "ok"

        def bare(**kwargs):
            ran.append(kwargs)
            return "ok"

        empty = StructuredTool(
            name="bare", description="d", func=bare, args_schema={"type": "object"}
        )
        calls = [self.call("loose", {}, "a"), self.call("bare", {}, "b")]
        messages = self.run(example, "sync", [loose, empty], calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert all(self.ONLY_VARIADICS in m.content for m in messages)

    def test_limitation_a_function_that_names_its_parameter_and_calls_another_is_not_read(
        self, example
    ):
        """Documented in the package README under what the gate does not see: the gate
        reads signatures, not bodies. A forwarder that names its first parameter is, to
        the gate, ``def f(table, **kwargs)``; the function it calls runs its own default."""
        from langchain_core.tools import tool

        ran, seen = [], []

        def export(table: str, limit: int = 1000000) -> str:
            ran.append((table, limit))
            return "exported"

        @tool
        def export_rows(table: str, *args: Any, **kwargs: Any) -> str:
            """Export the rows of a table."""
            return export(table, *args, **kwargs)

        calls = [self.call("export_rows", {"table": "users"})]
        messages = self.run(example, "sync", [export_rows], calls, self.limit_under(1000, seen))
        assert messages[0].status == "success", messages[0].content
        assert seen == [("export_rows", {"table": "users"})]
        assert ran == [("users", 1000000)], "the limit that ran was not shown to the policy"

    # --- the names LangChain keeps, exactly ---------------------------------------------------

    @staticmethod
    def keyword_tools(ran):
        from langchain_core.tools import BaseTool, StructuredTool

        schema = {"type": "object", "properties": {"q": {"type": "string"}}}

        def takes_extras(q: str, **extra: Any) -> str:
            ran.append(("extras", q, extra))
            return "ok"

        def names_callbacks(q: str, callbacks: Any = "default") -> str:
            ran.append(("names", q, callbacks if isinstance(callbacks, str) else "LangChain's"))
            return "ok"

        class NamesBoth(BaseTool):
            name: str = "both"
            description: str = "A BaseTool whose _run names callbacks and run_manager."

            def _run(self, q: str, callbacks: Any = "default", run_manager=None) -> str:
                ran.append(("both", q, callbacks))
                return "ok"

        class TakesExtras(BaseTool):
            name: str = "loose"
            description: str = "A BaseTool whose _run names neither."

            def _run(self, q: str, **extra: Any) -> str:
                ran.append(("loose", q, extra))
                return "ok"

        return [
            StructuredTool(name="extras", description="d", func=takes_extras, args_schema=schema),
            StructuredTool(name="names", description="d", func=names_callbacks, args_schema=schema),
            NamesBoth(),
            TakesExtras(),
        ]

    def test_a_reserved_name_the_function_really_receives_is_judged_not_refused(self, example):
        """``callbacks`` through ``**kwargs``, ``callbacks`` on a ``BaseTool`` subclass's
        ``_run`` and ``run_manager`` through that ``_run``'s ``**kwargs``: nothing writes
        there, the function receives the model's value, and the call was refused with "the
        tool would not receive the judged value". It is judged, and runs as judged."""
        ran, seen = [], []
        calls = [
            self.call("extras", {"q": "x", "callbacks": "m"}, "a"),
            self.call("both", {"q": "x", "callbacks": "m"}, "b"),
            self.call("loose", {"q": "x", "run_manager": "m", "callbacks": "m"}, "c"),
            self.call("extras", {"q": "x", "callbacks": "DELETE"}, "d"),
        ]
        messages = self.run(example, "sync", self.keyword_tools(ran), calls, self.no_delete(seen))
        assert {m.tool_call_id: m.status for m in messages} == {
            "a": "success",
            "b": "success",
            "c": "success",
            "d": "error",
        }
        assert sorted(seen, key=str) == [
            ("both", {"q": "x", "callbacks": "m"}),
            ("extras", {"q": "x", "callbacks": "DELETE"}),
            ("extras", {"q": "x", "callbacks": "m"}),
            ("loose", {"q": "x", "run_manager": "m", "callbacks": "m"}),
        ]
        assert sorted(ran, key=str) == [
            ("both", "x", "m"),
            ("extras", "x", {"callbacks": "m"}),
            ("loose", "x", {"run_manager": "m", "callbacks": "m"}),
        ]

    @pytest.mark.parametrize(
        ("tool_name", "key"),
        [
            ("extras", "config"),
            ("extras", "run_manager"),
            ("names", "callbacks"),
            ("names", "config"),
            ("both", "run_manager"),
        ],
    )
    def test_a_name_langchain_writes_or_keeps_is_refused_and_ungated_never_arrives(
        self, example, tool_name, key
    ):
        """Each of these is a name the function does not receive the model's value under:
        ungated, the value is dropped or replaced by LangChain's own."""
        ran, seen = [], []
        calls = [self.call(tool_name, {"q": "x", key: "model"})]
        messages = self.run(example, "sync", self.keyword_tools(ran), calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert f"the invocation sets `{key}`, which LangChain fills itself" in messages[0].content
        assert "rename the parameter" in messages[0].content
        assert "MCP server" not in messages[0].content
        self.run(example, "sync", self.keyword_tools(ran), calls, None, gated=False)
        assert len(ran) == 1 and "model" not in repr(ran)

    def test_the_names_langchain_keeps_are_read_for_the_tool_and_the_function(self, example):
        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import BaseTool, StructuredTool

        adapter = _adapter()
        extras, names, both, loose = self.keyword_tools([])
        assert adapter._fills_itself(extras, extras.func) == {"config", "run_manager"}
        assert adapter._fills_itself(names, names.func) == {"config", "run_manager", "callbacks"}
        assert adapter._fills_itself(both, both._run) == {"run_manager"}
        assert adapter._fills_itself(loose, loose._run) == frozenset()

        def configured(q: str, cfg: RunnableConfig) -> str:
            return q

        held = StructuredTool.from_function(func=configured, name="c", description="d")
        assert adapter._fills_itself(held, configured) == {"config", "run_manager", "cfg"}

        class Configured(BaseTool):
            name: str = "c"
            description: str = "d"

            def _run(self, q: str, cfg: RunnableConfig) -> str:
                return q

        assert adapter._fills_itself(Configured(), Configured()._run) == {"cfg"}

    def test_a_signature_that_cannot_be_read_counts_every_reserved_name_as_kept(
        self, example, monkeypatch
    ):
        adapter = _adapter()
        extras = self.keyword_tools([])[0]

        def unreadable(func):
            raise ValueError("no signature found")

        monkeypatch.setattr(adapter.inspect, "signature", unreadable)
        assert adapter._fills_itself(extras, extras.func) == {"config", "run_manager", "callbacks"}

    def test_a_call_that_sets_self_is_refused_before_the_policy(self, example):
        """LangChain calls the tool's bound ``_run`` with the arguments as keywords. A key
        named ``self`` raised ``TypeError: got multiple values for argument 'self'`` after
        the policy had allowed the call, and the run failed."""
        ran, seen = [], []
        tools = self.keyword_tools(ran)
        calls = [
            self.call("extras", {"q": "x", "self": "m"}, "a"),
            self.call("loose", {"q": "x", "self": "m"}, "b"),
        ]
        messages = self.run(example, "sync", tools, calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert all("the invocation sets `self`" in m.content for m in messages)
        with pytest.raises(TypeError, match="multiple values for argument 'self'"):
            self.run(example, "sync", tools, calls[:1], None, gated=False)

    def test_an_mcp_argument_with_a_name_langchain_keeps(self, example, monkeypatch):
        """For a tool ``langchain-mcp-adapters`` built: ``callbacks`` reaches the server as
        the model sent it, so it is judged; ``config``, ``run_manager`` and ``self`` cannot
        reach the server, and the refusal says so in the server's terms, not "rename the
        parameter"."""
        received, seen = [], []
        tool = self.mcp_library(monkeypatch)(received, self.MCP_SCHEMA)
        calls = [
            self.call("read_file", {"path": "/x", "callbacks": "m"}, "callbacks"),
            self.call("read_file", {"path": "/x", "config": "m"}, "config"),
            self.call("read_file", {"path": "/x", "run_manager": "m"}, "run_manager"),
            self.call("read_file", {"path": "/x", "self": "m"}, "self"),
        ]
        messages = self.run(example, "async", [tool], calls, self.no_delete(seen))
        by_id = {m.tool_call_id: m for m in messages}
        assert by_id["callbacks"].status == "success", by_id["callbacks"].content
        assert seen == [("read_file", {"path": "/x", "callbacks": "m"})]
        assert received == [{"path": "/x", "callbacks": "m"}]
        for key in ("config", "run_manager"):
            assert f"the invocation sets `{key}`" in by_id[key].content
            assert "the MCP server would not receive it" in by_id[key].content
            assert "rename the parameter" not in by_id[key].content
        assert "the invocation sets `self`" in by_id["self"].content

    # --- a stack of decorators too deep to read ------------------------------------------------

    def test_a_stack_deeper_than_the_walk_reads_is_refused_with_that_reason(self, example):
        import functools

        from langchain_core.tools import StructuredTool

        ran, seen = [], []
        depth = _adapter()._READ_DEPTH

        def logged(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def lookup(key: str, limit: int = 10) -> str:
            ran.append((key, limit))
            return "found"

        def stacked(count):
            fn = lookup
            for _ in range(count):
                fn = logged(fn)
            assert _adapter()._stacked(fn) == min(count, depth + 1)
            return StructuredTool.from_function(func=fn, name="lookup", description="d")

        calls = [self.call("lookup", {"key": "k"})]
        messages = self.run(example, "sync", [stacked(depth)], calls, self.no_delete(seen))
        assert messages[0].status == "success", "the kernel reads a stack this deep"
        assert seen == [("lookup", {"key": "k", "limit": 10})] and ran == [("k", 10)]
        for count in (depth + 1, depth + 9):
            seen.clear(), ran.clear()
            messages = self.run(example, "sync", [stacked(count)], calls, self.no_delete(seen))
            assert seen == [] and ran == []
            assert f"under more decorators than the gate reads (more than {depth}" in (
                messages[0].content
            )
            assert "descriptor" not in messages[0].content

    def test_a_shallow_callable_that_cannot_be_read_is_not_called_a_deep_stack(
        self, example, monkeypatch
    ):
        seen = []

        def unreadable(func):
            raise ValueError("no signature found")

        monkeypatch.setattr(_adapter(), "_reached_parameters", unreadable)
        messages, _ = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})], policy=self.no_delete(seen)
        )
        assert "cannot be inspected for its parameters (ValueError)" in messages[0].content
        assert "more decorators" not in messages[0].content

    # --- the review's surviving mutations, each pinned ----------------------------------------

    @ROUTES
    def test_a_value_the_model_sent_under_a_name_toolnode_fills_is_judged_when_unchanged(
        self, example, route
    ):
        """Pins Z03. An earlier hook attaches a variant of the registered tool whose
        function takes the graph state. ToolNode injects by the registered tool, which has
        no state parameter, so the model's ``state`` arrives as the model sent it: it is
        the model's, and it is judged."""
        from typing import Annotated

        from langchain_core.tools import tool
        from langgraph.prebuilt import InjectedState

        ran, seen = [], []

        @tool("lookup")
        def registered(q: str) -> str:
            """Look up."""
            ran.append(("registered", q))
            return "ok"

        @tool("lookup")
        def variant(q: str, state: Annotated[dict, InjectedState] = {"role": "user"}) -> str:  # noqa: B006
            """Look up, by role."""
            ran.append(("variant", q, state))
            return "ok"

        assert _adapter()._node_fills(variant) == {"state"}
        gates = example.recusal_gates(self.no_delete(seen))

        def attach(request, execute):
            return gates["wrap_tool_call"](request.override(tool=variant), execute)

        async def aattach(request, execute):
            return await gates["awrap_tool_call"](request.override(tool=variant), execute)

        node = example.ToolNode([registered], wrap_tool_call=attach, awrap_tool_call=aattach)
        calls = [self.call("lookup", {"q": "x", "state": {"role": "DELETE"}})]
        messages = self.in_graph(example, route, node, calls)
        assert seen == [("lookup", {"q": "x", "state": {"role": "DELETE"}})]
        assert ran == [] and messages[0].status == "error"

    @ROUTES
    def test_an_error_answer_that_is_not_the_refusal_is_withheld_after_a_refused_call(
        self, example, route
    ):
        """Pins Z41. The policy refused; a step after the gate ran another tool, which
        failed, and answered with that failure under the judged tool's name. An error
        answer passes only when it holds the gate's own reason."""
        from langchain_core.messages import ToolMessage

        seen, ran = [], []
        gates = example.recusal_gates(self.no_delete(seen))

        def failed(judged, first):
            if first.status != "error":
                return first
            ran.append(judged.tool_call["args"]["sql"])
            return ToolMessage(
                content="fallback failed after it ran the statement",
                tool_call_id=judged.tool_call["id"],
                name="run_sql",
                status="error",
            )

        def after(request, execute):
            return gates["wrap_tool_call"](request, lambda judged: failed(judged, execute(judged)))

        async def aafter(request, execute):
            async def fallback(judged):
                return failed(judged, await execute(judged))

            return await gates["awrap_tool_call"](request, fallback)

        node = example.ToolNode(
            [example.echo, example.run_sql], wrap_tool_call=after, awrap_tool_call=aafter
        )
        messages = self.in_graph(example, route, node, [self.call("run_sql", {"sql": DELETE})])
        assert seen == [("run_sql", {"sql": DELETE})] and ran == [DELETE]
        assert messages[0].status == "error"
        assert "fallback failed" not in messages[0].content
        assert "did not allow the call" in messages[0].content

    def test_a_validation_message_at_the_model_root_is_never_forwarded(self, example):
        """Pins Z23. A root validator can build its message from anything it was handed,
        the host's state included, under any error type: only the type is reported."""
        from pydantic import BaseModel, ValidationError, model_validator
        from pydantic_core import PydanticCustomError

        class Args(BaseModel):
            sql: str

            @model_validator(mode="before")
            @classmethod
            def check(cls, data):
                raise PydanticCustomError(
                    "tenant_mismatch", "the state says {who}", {"who": "admin"}
                )

        with pytest.raises(ValidationError) as caught:
            Args(sql="x")
        text = _adapter()._sanitized(caught.value, frozenset())
        assert text == "model: tenant_mismatch"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestAForwarderThatNamesOnlyWhatOthersFill:
    """Review round 4 of the package (2026-11): a forwarder whose only named parameter is
    one the framework or the host fills, the schema a model can put nothing through, names
    the thread runner takes, and three checks no test pinned. Each test names its
    finding."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_langgraph_round4_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    call = staticmethod(TestWhatRunsIsWhatWasJudged.call)
    no_delete = staticmethod(TestWhatRunsIsWhatWasJudged.no_delete)
    run = TestWhatRunsIsWhatWasJudged.run
    ROUTES = TestWhatRunsIsWhatWasJudged.ROUTES
    limit_under = staticmethod(TestWhatTheGateCannotReadIsRefused.limit_under)

    ONLY_VARIADICS = "the tool's innermost callable takes only `*args` or `**kwargs`"

    # --- finding A: the forwarder's only named parameter is filled by someone else -------------

    @staticmethod
    def forwarders(ran):
        """Each hands what it gets to ``export_rows(table="users", limit=1000000)`` and
        names, besides its variadics, only a parameter LangChain or ToolNode fills."""
        from typing import Annotated

        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import BaseTool, InjectedToolArg
        from langgraph.prebuilt import InjectedState, ToolRuntime

        def export_rows(table: str = "users", limit: int = 1000000) -> str:
            ran.append((table, limit))
            return "exported"

        def with_tenant(tenant: Annotated[str, InjectedToolArg] = "t", **kwargs):
            return export_rows(**kwargs)

        def with_untyped_runtime(runtime: Any = None, **kwargs):
            return export_rows(**kwargs)

        def with_config(*args, config: RunnableConfig, **kwargs):
            return export_rows(*args, **kwargs)

        def with_runtime(runtime: ToolRuntime, **kwargs):
            return export_rows(**kwargs)

        def with_state(state: Annotated[dict, InjectedState], **kwargs):
            return export_rows(**kwargs)

        def with_callbacks(*args, callbacks: Any = None, **kwargs):
            return export_rows(*args, **kwargs)

        class Forwards(BaseTool):
            name: str = "export_rows"
            description: str = "Export rows of a table."

            def _run(self, *args: Any, run_manager: Any = None, **kwargs: Any) -> str:
                return export_rows(*args, **kwargs)

        return {
            "config": with_config,
            "runtime": with_runtime,
            "state": with_state,
            "callbacks": with_callbacks,
            "run_manager": Forwards,
            "tenant": with_tenant,
            "untyped_runtime": with_untyped_runtime,
        }

    @pytest.mark.parametrize(
        "kept",
        ["config", "runtime", "state", "callbacks", "run_manager", "tenant", "untyped_runtime"],
    )
    @ROUTES
    def test_a_forwarder_that_names_only_what_the_framework_or_the_host_fills_is_refused(
        self, example, route, kept
    ):
        """``def wrapper(*args, config: RunnableConfig, **kwargs)`` counted as naming a
        parameter of its own: the policy judged ``{}`` and the function behind it ran
        ``limit=1000000``. Under the schemaless ``BaseTool`` the model could send the limit
        as well."""
        from langchain_core.tools import StructuredTool

        ran, seen = [], []
        shape = self.forwarders(ran)[kept]
        if kept == "run_manager":
            tool = shape()
            assert tool.args_schema is None
        else:
            tool = StructuredTool.from_function(func=shape, name="export_rows", description="d")
        calls = [self.call("export_rows", {})]
        messages = self.run(example, route, [tool], calls, self.limit_under(1000, seen))
        assert ran == [] and seen == [], "refused before the policy, and not run"
        assert self.ONLY_VARIADICS in messages[0].content
        named = "runtime" if kept == "untyped_runtime" else kept
        assert f"(and `{named}`, which LangChain or the host reserves)" in messages[0].content
        self.run(example, route, [tool], calls, None, gated=False)
        assert ran == [("users", 1000000)], "ungated, the default nobody judged runs"

    def test_a_forwarder_under_a_json_schema_is_refused_whatever_the_schema_closes(self, example):
        """LangChain validates nothing against a dict schema: with ``additionalProperties``
        false the model's ``limit`` still reaches the function."""
        from langchain_core.tools import StructuredTool

        ran, seen = [], []
        forward = self.forwarders(ran)["config"]
        closed = {"type": "object", "properties": {}, "additionalProperties": False}
        for schema in (closed, {"type": "object"}):
            tool = StructuredTool(
                name="export_rows", description="d", func=forward, args_schema=schema
            )
            calls = [self.call("export_rows", {}, "a"), self.call("export_rows", {"limit": 5}, "b")]
            messages = self.run(example, "sync", [tool], calls, self.limit_under(1000, seen))
            assert ran == [] and seen == []
            assert all(self.ONLY_VARIADICS in m.content for m in messages)
            self.run(example, "sync", [tool], calls[1:], None, gated=False)
            assert ran == [("users", 5)], "ungated, the dict schema let the key through"
            ran.clear()

    # --- the exception: a schema the model can put nothing through -----------------------------

    @ROUTES
    def test_a_forwarder_under_a_schema_with_no_fields_runs_with_nothing_of_the_models(
        self, example, route
    ):
        """An explicit Pydantic ``args_schema`` with no fields: LangChain hands the function
        only declared fields, so whatever the model sends is dropped. The call carries no
        value the model chose; it is judged as ``{}`` and runs."""
        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import BaseTool, StructuredTool
        from pydantic import BaseModel

        ran, seen = [], []

        class NoInput(BaseModel):
            pass

        class CurrentPage(BaseTool):
            name: str = "current_page"
            description: str = "No input."
            args_schema: type[BaseModel] = NoInput

            def _run(self, config: RunnableConfig = None, **_: Any) -> str:  # type: ignore[assignment]
                ran.append(("current_page", _))
                return "page"

            async def _arun(self, config: RunnableConfig = None, **_: Any) -> str:  # type: ignore[assignment]
                ran.append(("current_page", _))
                return "page"

        function = StructuredTool(
            name="export_rows",
            description="d",
            func=self.forwarders(ran)["config"],
            args_schema=NoInput,
        )
        calls = [
            self.call("current_page", {}, "a"),
            self.call("current_page", {"limit": 5, "kwargs": {"limit": 5}}, "b"),
            self.call("export_rows", {"limit": 5, "table": "secrets"}, "c"),
        ]
        messages = self.run(example, route, [CurrentPage(), function], calls, self.no_delete(seen))
        assert [m.status for m in messages] == ["success"] * 3, [m.content for m in messages]
        assert sorted(seen, key=str) == [
            ("current_page", {}),
            ("current_page", {}),
            ("export_rows", {}),
        ]
        assert sorted(ran, key=str) == [
            ("current_page", {}),
            ("current_page", {}),
            ("users", 1000000),
        ], "nothing the model sent reached a function"

    def test_which_schemas_the_model_can_put_nothing_through(self, example):
        from langchain_core.tools import BaseTool, StructuredTool
        from pydantic import BaseModel, ConfigDict

        closed = _adapter()._closed_to_the_model

        class NoInput(BaseModel):
            pass

        class Forbids(BaseModel):
            model_config = ConfigDict(extra="forbid")

        class Allows(BaseModel):
            model_config = ConfigDict(extra="allow")

        class OneField(BaseModel):
            q: str = ""

        def forward(**kwargs):
            return "ok"

        def over(schema):
            return StructuredTool(name="t", description="d", func=forward, args_schema=schema)

        class Schemaless(BaseTool):
            name: str = "t"
            description: str = "d"

            def _run(self, **kwargs: Any) -> str:
                return "ok"

        assert closed(over(NoInput)) is True and closed(over(Forbids)) is True
        assert closed(over(Allows)) is False and closed(over(OneField)) is False
        assert (
            closed(over({"type": "object", "properties": {}, "additionalProperties": False}))
            is False
        )
        assert (
            closed(StructuredTool.from_function(func=forward, name="t", description="d")) is False
        )
        assert closed(Schemaless()) is False

    def test_a_forwarder_under_an_empty_schema_that_allows_extras_is_refused(self, example):
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, ConfigDict

        ran, seen = [], []

        class Allows(BaseModel):
            model_config = ConfigDict(extra="allow")

        tool = StructuredTool(
            name="export_rows",
            description="d",
            func=self.forwarders(ran)["config"],
            args_schema=Allows,
        )
        messages = self.run(
            example, "sync", [tool], [self.call("export_rows", {})], self.no_delete(seen)
        )
        assert ran == [] and seen == [] and self.ONLY_VARIADICS in messages[0].content

    # --- the real tools the review's census found ----------------------------------------------

    def test_the_shapes_of_the_census_tools_are_refused_and_the_named_function_runs(self, example):
        """The signatures and schemas of langchain-aws 1.10.0's ``ThreadAwareNavigateBackTool``
        (an empty Pydantic schema; ``config: Optional[RunnableConfig] = None``, which
        LangChain does not fill because the annotation is not exactly ``RunnableConfig``)
        and langchain-community's ``SlackGetChannel`` (no schema; ``*args`` and
        ``run_manager``). Both are refused; a function of your own that calls the tool runs,
        as the README says."""
        from typing import Optional

        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import BaseTool, tool
        from pydantic import BaseModel

        ran, seen = [], []

        class NavigateBackToolInput(BaseModel):
            pass

        class NavigateBack(BaseTool):
            name: str = "navigate_back"
            description: str = "Navigate back to the previous page"
            args_schema: type[BaseModel] = NavigateBackToolInput

            def _run(self, config: Optional[RunnableConfig] = None, **_: Any) -> str:
                ran.append("back")
                return "went back"

        class GetChannel(BaseTool):
            name: str = "get_channelid_name_dict"
            description: str = "There is no input to this tool"

            def _run(self, *args: Any, run_manager: Any = None) -> str:
                ran.append("channels")
                return "{}"

        back, channels = NavigateBack(), GetChannel()
        calls = [
            self.call("navigate_back", {}, "a"),
            self.call("get_channelid_name_dict", {}, "b"),
        ]
        messages = self.run(example, "sync", [back, channels], calls, self.no_delete(seen))
        by_id = {m.tool_call_id: m.content for m in messages}
        assert ran == [] and seen == []
        assert "the callable's parameter `config` is absent from the argument schema" in by_id["a"]
        assert self.ONLY_VARIADICS in by_id["b"]
        assert "(and `run_manager`, which LangChain or the host reserves)" in by_id["b"]

        @tool
        def navigate_back() -> str:
            """Navigate back to the previous page."""
            return back.invoke({})

        @tool
        def get_channels() -> str:
            """Get the channels."""
            return channels.invoke({})

        calls = [self.call("navigate_back", {}, "a"), self.call("get_channels", {}, "b")]
        messages = self.run(
            example, "sync", [navigate_back, get_channels], calls, self.no_delete(seen)
        )
        assert [m.status for m in messages] == ["success", "success"]
        assert sorted(seen) == [("get_channels", {}), ("navigate_back", {})]
        assert sorted(ran) == ["back", "channels"]

    # --- the review's surviving mutations, each pinned ----------------------------------------

    def test_a_callable_of_only_star_args_is_a_forwarder_too(self, example):
        """Pins X10: only ``**kwargs`` counted as forwarding."""
        from langchain_core.tools import BaseTool, tool

        ran, seen = [], []

        def wrapper(*args):
            ran.append(args)
            return "ok"

        wrapper.__doc__ = "Positional only."

        class Positional(BaseTool):
            name: str = "positional"
            description: str = "d"

            def _run(self, *args: Any) -> str:
                ran.append(args)
                return "ok"

        calls = [self.call("wrapper", {}, "a"), self.call("positional", {}, "b")]
        messages = self.run(
            example, "sync", [tool(wrapper), Positional()], calls, self.no_delete(seen)
        )
        assert ran == [] and seen == []
        assert all(self.ONLY_VARIADICS in m.content for m in messages)

    def test_the_names_langchain_keeps_are_read_from_the_method_the_route_runs(self, example):
        """Pins X12: ``_arun`` names ``run_manager`` and ``_run`` does not. On ``ainvoke``
        LangChain overwrites the model's value; on ``invoke`` the function receives it."""
        from langchain_core.tools import BaseTool

        ran, seen = [], []

        class Split(BaseTool):
            name: str = "split"
            description: str = "d"

            def _run(self, q: str, **extra: Any) -> str:
                ran.append(("sync", q, extra))
                return "ok"

            async def _arun(self, q: str, run_manager: Any = None, **extra: Any) -> str:
                ran.append(("async", q, extra))
                return "ok"

        calls = [self.call("split", {"q": "x", "run_manager": "m"})]
        messages = self.run(example, "sync", [Split()], calls, self.no_delete(seen))
        assert messages[0].status == "success", messages[0].content
        assert seen == [("split", {"q": "x", "run_manager": "m"})]
        assert ran == [("sync", "x", {"run_manager": "m"})]
        seen.clear(), ran.clear()
        messages = self.run(example, "async", [Split()], calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert "the invocation sets `run_manager`, which LangChain fills itself" in (
            messages[0].content
        )

    def test_callbacks_is_read_where_langchain_reads_it_behind_a_wraps_decorator(self, example):
        """Pins X02: LangChain reads the function's signature through ``__wrapped__``, finds
        ``callbacks`` and writes its own value there. The outer wrapper does not name it."""
        import functools

        from langchain_core.tools import StructuredTool

        ran, seen = [], []

        def lookup(q: str, callbacks: Any = "default") -> str:
            ran.append((q, callbacks if isinstance(callbacks, str) else "LangChain's"))
            return "ok"

        @functools.wraps(lookup)
        def logged(*args, **kwargs):
            return lookup(*args, **kwargs)

        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        tool = StructuredTool(name="lookup", description="d", func=logged, args_schema=schema)
        assert "callbacks" in _adapter()._fills_itself(tool, logged)
        calls = [self.call("lookup", {"q": "x", "callbacks": "model"})]
        messages = self.run(example, "sync", [tool], calls, self.no_delete(seen))
        assert ran == [] and seen == []
        assert "the invocation sets `callbacks`, which LangChain fills itself" in (
            messages[0].content
        )
        self.run(example, "sync", [tool], calls, None, gated=False)
        assert ran == [("x", "LangChain's")], "ungated, the model's value never arrives"

    # --- names the thread runner takes ------------------------------------------------------------

    @staticmethod
    def threaded_tools(ran):
        from langchain_core.tools import BaseTool, StructuredTool

        schema = {"type": "object", "properties": {"q": {"type": "string"}}}

        class Loose(BaseTool):
            name: str = "loose"
            description: str = "No _arun of its own."

            def _run(self, q: str, **extra: Any) -> str:
                ran.append(("loose", q, extra))
                return "ok"

        def held(q: str, **extra: Any) -> str:
            ran.append(("held", q, extra))
            return "ok"

        async def aheld(q: str, **extra: Any) -> str:
            ran.append(("aheld", q, extra))
            return "ok"

        return [
            Loose(),
            StructuredTool(name="held", description="d", func=held, args_schema=schema),
            StructuredTool(
                name="both", description="d", func=held, coroutine=aheld, args_schema=schema
            ),
        ]

    @pytest.mark.parametrize("key", ["func", "executor_or_config"])
    def test_a_name_the_thread_runner_takes_is_refused_on_the_async_route(self, example, key):
        """A ``BaseTool`` subclass with no ``_arun`` of its own runs ``_run`` through
        ``run_in_executor`` on ``ainvoke``, with the arguments as keywords: a key named
        like one of that function's parameters raised ``TypeError`` after the policy had
        allowed the call. A ``StructuredTool`` does not run that way, with or without a
        coroutine, and neither does the sync route: there the key is judged."""
        ran, seen = [], []
        calls = [
            self.call("loose", {"q": "x", key: "m"}, "loose"),
            self.call("held", {"q": "x", key: "m"}, "held"),
            self.call("both", {"q": "x", key: "m"}, "both"),
        ]
        messages = self.run(example, "async", self.threaded_tools(ran), calls, self.no_delete(seen))
        by_id = {m.tool_call_id: m for m in messages}
        assert f"the invocation sets `{key}`, a parameter of the function LangChain runs" in (
            by_id["loose"].content
        )
        assert by_id["held"].status == by_id["both"].status == "success"
        assert sorted(seen) == [("both", {"q": "x", key: "m"}), ("held", {"q": "x", key: "m"})]
        assert sorted(ran) == [("aheld", "x", {key: "m"}), ("held", "x", {key: "m"})]
        seen.clear(), ran.clear()
        messages = self.run(example, "sync", self.threaded_tools(ran), calls, self.no_delete(seen))
        assert [m.status for m in messages] == ["success"] * 3
        assert len(seen) == 3 and all(extra == {key: "m"} for _, _, extra in ran)
        # negative control: ungated, the async route of the BaseTool subclass raises
        with pytest.raises(TypeError, match=f"multiple values for argument '{key}'"):
            self.run(example, "async", self.threaded_tools(ran), calls[:1], None, gated=False)

    def test_the_names_the_thread_runner_takes_are_read_from_its_signature(self, example):
        assert _adapter()._THREAD_TAKES == {"executor_or_config", "func"}

    # --- a failure while the invocation is prepared is a refusal -----------------------------

    @pytest.mark.parametrize("kind", [NameError, ValueError, TypeError])
    def test_a_signature_read_that_raises_counts_every_reserved_name_as_kept(
        self, example, monkeypatch, kind
    ):
        """Python 3.14 raises ``NameError`` from ``inspect.signature`` for an annotation it
        cannot resolve; only ``TypeError`` and ``ValueError`` were caught."""
        from langchain_core.tools import StructuredTool

        adapter = _adapter()

        def takes_extras(q: str, **extra: Any) -> str:
            return "ok"

        tool = StructuredTool.from_function(func=takes_extras, name="t", description="d")

        def unreadable(func):
            raise kind("name 'Later' is not defined")

        monkeypatch.setattr(adapter.inspect, "signature", unreadable)
        assert adapter._fills_itself(tool, takes_extras) == {"config", "run_manager", "callbacks"}

    def test_anything_raised_while_the_invocation_is_prepared_is_a_refusal(
        self, example, monkeypatch
    ):
        adapter = _adapter()
        seen = []

        def failing(kwargs, tool, host_keys, is_async):
            raise NameError("name 'Later' is not defined")

        monkeypatch.setattr(adapter, "_completed", failing)
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": SELECT})], policy=self.no_delete(seen)
        )
        assert executed == [] and seen == []
        assert messages[0].status == "error"
        assert "the invocation could not be prepared for judgment (NameError)" in (
            messages[0].content
        )
        assert "Later" not in messages[0].content

    # --- what the README says about `self`, `config` and `run_manager`, to the letter ------------

    def test_a_reserved_key_the_schema_drops_never_reaches_the_function(self, example):
        """Under a Pydantic schema that does not declare them, LangChain drops ``self``,
        ``config`` and ``run_manager`` before the gate's judging step: the call runs, judged
        without them. They are refused only where the schema lets the key through."""
        from langchain_core.tools import tool

        ran, seen = [], []

        @tool
        def lookup(q: str) -> str:
            """Look up."""
            ran.append(q)
            return "found"

        calls = [self.call("lookup", {"q": "x", "self": "m", "config": "m", "run_manager": "m"})]
        messages = self.run(example, "sync", [lookup], calls, self.no_delete(seen))
        assert messages[0].status == "success", messages[0].content
        assert seen == [("lookup", {"q": "x"})] and ran == ["x"]

    def test_a_single_input_tool_takes_a_key_named_self_as_its_one_value(self, example):
        from langchain_core.tools import Tool

        ran, seen = [], []

        def lookup(q):
            ran.append(q)
            return "found"

        tool = Tool(name="lookup", func=lookup, description="Look up.")
        messages = self.run(
            example, "sync", [tool], [self.call("lookup", {"self": "x"})], self.no_delete(seen)
        )
        assert messages[0].status == "success", messages[0].content
        assert seen == [("lookup", {"q": "x"})] and ran == ["x"]


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestAReservedNameIsNeverAParameterOfItsOwn:
    """Review round 5 of the package (2026-11): the third forwarder with one more inert
    named parameter (``config: Optional[RunnableConfig] = None`` on a schemaless
    ``BaseTool``). The family is closed by rule: in the forwarder rule a parameter with a
    reserved name, an injected one, or one annotated with a type the framework or the host
    fills is never the callable's own. Plus the low items and six checks no test pinned."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_langgraph_round5_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    call = staticmethod(TestWhatRunsIsWhatWasJudged.call)
    no_delete = staticmethod(TestWhatRunsIsWhatWasJudged.no_delete)
    run = TestWhatRunsIsWhatWasJudged.run
    ROUTES = TestWhatRunsIsWhatWasJudged.ROUTES
    limit_under = staticmethod(TestWhatTheGateCannotReadIsRefused.limit_under)

    ONLY_VARIADICS = "the tool's innermost callable takes only `*args` or `**kwargs`"

    # --- finding 1: a `config` LangChain does not fill ------------------------------------------

    @staticmethod
    def schemaless_forwarder(ran, shape):
        """A ``BaseTool`` with no ``args_schema`` whose ``_run`` hands the call to
        ``export_rows(table="users", limit=1000000)`` and names only ``config``."""
        from typing import Optional

        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import BaseTool

        def export_rows(table: str = "users", limit: int = 1000000) -> str:
            ran.append((table, limit))
            return "exported"

        if shape == "optional":

            def _run(self, *args: Any, config: Optional[RunnableConfig] = None, **kwargs: Any):
                return export_rows(*args, **kwargs)

        elif shape == "pipe":

            def _run(self, *args: Any, config: RunnableConfig | None = None, **kwargs: Any):
                return export_rows(*args, **kwargs)

        elif shape == "unannotated":

            def _run(self, *args: Any, config=None, **kwargs: Any):
                return export_rows(*args, **kwargs)

        elif shape == "wrong":

            def _run(self, *args: Any, config: str = "x", **kwargs: Any):
                return export_rows(*args, **kwargs)

        else:  # the shape of langchain-aws's no-input tools, without their schema

            def _run(self, config: Optional[RunnableConfig] = None, **_: Any):
                return export_rows(**_)

        body = _run

        class Export(BaseTool):
            name: str = "export"
            description: str = "Export rows of a table."
            _run = body

        tool = Export()
        assert tool.args_schema is None
        return tool

    @pytest.mark.parametrize("shape", ["optional", "pipe", "unannotated", "wrong", "aws"])
    @ROUTES
    def test_a_forwarder_whose_only_parameter_has_a_reserved_name_is_refused(
        self, example, route, shape
    ):
        """LangChain fills ``config`` only when it is annotated exactly ``RunnableConfig``.
        Any other ``config`` was read as a parameter of the tool's own: the policy judged
        ``{'config': None}`` and the function behind ran ``limit=1000000``, or whatever
        limit the model sent."""
        ran, seen = [], []
        tool = self.schemaless_forwarder(ran, shape)
        calls = [self.call("export", {}, "a"), self.call("export", {"limit": 5}, "b")]
        messages = self.run(example, route, [tool], calls, self.limit_under(1000, seen))
        assert ran == [] and seen == [], "refused before the policy, and not run"
        for message in messages:
            assert self.ONLY_VARIADICS in message.content
            assert "(and `config`, which LangChain or the host reserves)" in message.content
        self.run(example, route, [tool], calls, None, gated=False)
        assert sorted(ran) == [("users", 5), ("users", 1000000)], "ungated, the model sets it"

    # --- the rule, stated once -----------------------------------------------------------------

    def test_the_reserved_names_are_the_union_of_the_gates_authorities(self, example):
        adapter = _adapter()
        assert adapter._RESERVED_NAMES == (
            adapter._RESERVED | adapter._HOLDER_TAKES | adapter._NODE_NAMES
        )
        assert adapter._RESERVED_NAMES == {"run_manager", "callbacks", "config", "runtime"}

    def test_which_parameters_are_never_a_forwarders_own(self, example):
        import inspect
        from typing import Annotated, Optional

        from langchain_core.callbacks import (
            AsyncCallbackManagerForToolRun,
            CallbackManagerForToolRun,
            Callbacks,
        )
        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import InjectedToolArg, InjectedToolCallId
        from langgraph.prebuilt import InjectedState, InjectedStore, ToolRuntime

        namespace = {**globals(), **locals()}

        def its_own(name, parameter, filled):
            return _adapter()._not_its_own(name, parameter, filled, {}, namespace) is None

        def parameter(name, annotation=inspect.Parameter.empty, default=inspect.Parameter.empty):
            kind = inspect.Parameter.KEYWORD_ONLY
            return inspect.Parameter(name, kind, annotation=annotation, default=default)

        # a reserved name, whatever its annotation or default
        for name in ("run_manager", "callbacks", "config", "runtime"):
            for annotation in (inspect.Parameter.empty, str, Optional[RunnableConfig], dict):
                for default in (inspect.Parameter.empty, None):
                    assert not its_own(name, parameter(name, annotation, default), frozenset())
        # a type the framework or the host fills, anywhere inside the annotation
        filled = [
            RunnableConfig,
            Optional[RunnableConfig],
            RunnableConfig | None,
            Annotated[RunnableConfig, "note"],
            Annotated[Optional[RunnableConfig], "note"],
            ToolRuntime,
            Optional[ToolRuntime],
            CallbackManagerForToolRun,
            Optional[AsyncCallbackManagerForToolRun],
            Callbacks,
            Annotated[dict, InjectedState],
            Annotated[str, InjectedState("role")],
            Annotated[Any, InjectedStore()],
            Annotated[str, InjectedToolCallId],
            Optional[Annotated[str, InjectedToolArg]],
            ToolRuntime[dict, dict],
            Optional[ToolRuntime[dict, dict]],
            "Optional[RunnableConfig]",
            "ToolRuntime",
            "Annotated[dict, InjectedState]",
            "Optional[CallbackManagerForToolRun]",
        ]
        for annotation in filled:
            assert not its_own("x", parameter("x", annotation), frozenset()), annotation
        # a name the tool marks injected, by its schema or its function
        assert not its_own("tenant", parameter("tenant", str), frozenset({"tenant"}))
        # an annotation nested deeper than the gate reads is not taken for the callable's own
        deep: Any = str
        for _ in range(20):
            deep = list[deep]
        assert not its_own("x", parameter("x", deep), frozenset())
        assert its_own("x", parameter("x", list[list[list[str]]]), frozenset())
        # everything else is the callable's own
        for annotation in (
            inspect.Parameter.empty,
            str,
            Optional[str],
            dict,
            Annotated[str, "n"],
            "str",
        ):
            assert its_own("x", parameter("x", annotation), frozenset()), annotation

    @pytest.mark.parametrize(
        "kept",
        ["schema_marked", "untyped_runtime", "state", "optional_run_manager", "optional_runtime"],
    )
    def test_a_forwarder_is_refused_for_every_way_a_parameter_is_not_its_own(self, example, kept):
        """One per branch of the rule: injected by the schema only, reserved by name only,
        injected by annotation, and a filled type inside ``Optional``. The second and
        third on a ``BaseTool`` subclass, which has no function for ToolNode to read."""
        from typing import Annotated, Optional

        from langchain_core.callbacks import CallbackManagerForToolRun
        from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
        from langgraph.prebuilt import InjectedState, ToolRuntime
        from pydantic import BaseModel

        ran, seen = [], []

        def export_rows(table: str = "users", limit: int = 1000000) -> str:
            ran.append((table, limit))
            return "exported"

        if kept == "schema_marked":

            class Args(BaseModel):
                tenant: Annotated[str, InjectedToolArg] = "t"

            def forward(tenant: str = "t", **kwargs: Any) -> str:
                return export_rows(**kwargs)

            tool = StructuredTool(name="export", description="d", func=forward, args_schema=Args)
            named = "tenant"
        elif kept == "optional_run_manager":

            def forward(rm: Optional[CallbackManagerForToolRun] = None, **kwargs: Any) -> str:
                return export_rows(**kwargs)

            schema = {"type": "object"}
            tool = StructuredTool(name="export", description="d", func=forward, args_schema=schema)
            named = "rm"
        else:
            if kept == "untyped_runtime":

                def _run(self, runtime=None, **kwargs: Any) -> str:
                    return export_rows(**kwargs)

                named = "runtime"
            elif kept == "state":

                def _run(self, state: Annotated[dict, InjectedState] = None, **kwargs: Any) -> str:  # type: ignore[assignment]
                    return export_rows(**kwargs)

                named = "state"
            else:

                def _run(self, rt: Optional[ToolRuntime] = None, **kwargs: Any) -> str:
                    return export_rows(**kwargs)

                named = "rt"
            body = _run

            class Export(BaseTool):
                name: str = "export"
                description: str = "d"
                _run = body

            tool = Export()
        messages = self.run(
            example, "sync", [tool], [self.call("export", {})], self.limit_under(1000, seen)
        )
        assert ran == [] and seen == []
        assert self.ONLY_VARIADICS in messages[0].content
        assert f"(and `{named}`, which LangChain or the host reserves)" in messages[0].content

    def test_a_schema_field_the_host_fills_must_still_be_named_by_a_forwarder(self, example):
        """Pins Y17: a field the schema declares and the innermost callable does not name
        is handed on, whoever fills it."""
        from typing import Annotated

        from langchain_core.tools import StructuredTool
        from langgraph.prebuilt import InjectedState
        from pydantic import BaseModel

        ran, seen = [], []

        class Args(BaseModel):
            q: str
            state: Annotated[dict, InjectedState]

        def forward(q: str, **kwargs: Any) -> str:
            ran.append((q, sorted(kwargs)))
            return "ok"

        tool = StructuredTool(name="lookup", description="d", func=forward, args_schema=Args)
        messages = self.run(
            example, "sync", [tool], [self.call("lookup", {"q": "x"})], self.no_delete(seen)
        )
        assert ran == [] and seen == []
        assert "takes `*args` or `**kwargs` and does not name `state`" in messages[0].content

    # --- the closed-schema exception, pinned further ---------------------------------------------

    def test_a_pure_forwarder_under_a_schema_with_no_fields_runs(self, example):
        """Pins Y04 and Y05: the exception covers a callable of nothing but variadics, and
        a schema that sets ``extra="ignore"`` explicitly is as closed as one that leaves the
        default."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, ConfigDict

        ran, seen = [], []

        class Ignores(BaseModel):
            model_config = ConfigDict(extra="ignore")

        def export_rows(table: str = "users", limit: int = 1000000) -> str:
            ran.append((table, limit))
            return "exported"

        def forward(*args: Any, **kwargs: Any) -> str:
            return export_rows(*args, **kwargs)

        tool = StructuredTool(name="export", description="d", func=forward, args_schema=Ignores)
        assert _adapter()._closed_to_the_model(tool) is True
        calls = [self.call("export", {"limit": 5, "table": "secrets"})]
        messages = self.run(example, "sync", [tool], calls, self.no_delete(seen))
        assert messages[0].status == "success", messages[0].content
        assert seen == [("export", {})] and ran == [("users", 1000000)]

    def test_limitation_a_schema_with_no_fields_drops_what_the_host_injects_too(self, example):
        """Documented in the package README under the closed-schema exception: LangChain
        hands the function only declared fields, so the state ToolNode injected is dropped
        with everything else, and the function runs its own default for that parameter.
        The value is a constant of the code, the same ungated; the policy is not shown it."""
        from typing import Annotated

        from langchain_core.tools import StructuredTool
        from langgraph.prebuilt import InjectedState
        from pydantic import BaseModel

        ran, seen = [], []

        class NoInput(BaseModel):
            pass

        def current(state: Annotated[dict, InjectedState] = None, **kwargs: Any) -> str:  # type: ignore[assignment]
            ran.append((state, kwargs))
            return "ok"

        tool = StructuredTool(name="current", description="d", func=current, args_schema=NoInput)
        calls = [self.call("current", {"limit": 5})]
        messages = self.run(example, "sync", [tool], calls, self.no_delete(seen))
        assert messages[0].status == "success", messages[0].content
        assert seen == [("current", {})] and ran == [(None, {})]
        ran.clear()
        self.run(example, "sync", [tool], calls, None, gated=False)
        assert ran == [(None, {})], "ungated, the injected state is dropped the same way"

    # --- the thread runner's names, pinned further -----------------------------------------------

    @staticmethod
    def threaded(ran, kind):
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, ConfigDict

        class Allows(BaseModel):
            model_config = ConfigDict(extra="allow")
            q: str

        if kind == "own":

            class Tool(BaseTool):
                name: str = "t"
                description: str = "Its own parameter is called func."

                def _run(self, q: str, func: str = "sum") -> str:
                    ran.append((q, func))
                    return "ok"

        else:
            schema: Any = Allows if kind == "allow" else {"type": "object", "properties": {"q": {}}}

            class Tool(BaseTool):  # type: ignore[no-redef]
                name: str = "t"
                description: str = "A schema that lets extra keys through."
                args_schema: Any = schema

                def _run(self, q: str, **extra: Any) -> str:
                    ran.append((q, extra))
                    return "ok"

        return Tool()

    def test_a_tools_own_parameter_named_like_the_thread_runners_is_refused_as_that(self, example):
        """Pins Y10, and makes the reason true. The model sent only ``q``; the gate
        completed ``func`` from the tool's own default, and the refusal said "the
        invocation sets `func`". The collision is with the tool's own parameter: ungated,
        the call runs while the model leaves ``func`` out and raises when it sends one."""
        ran, seen = [], []
        calls = [self.call("t", {"q": "x"}, "a"), self.call("t", {"q": "x", "func": "max"}, "b")]
        messages = self.run(
            example, "async", [self.threaded(ran, "own")], calls, self.no_delete(seen)
        )
        assert ran == [] and seen == []
        for message in messages:
            assert "the tool's own parameter `func` has the name of a parameter" in message.content
            assert "rename the parameter or define `_arun`" in message.content
            assert "the invocation sets" not in message.content
        self.run(example, "async", [self.threaded(ran, "own")], calls[:1], None, gated=False)
        assert ran == [("x", "sum")], "ungated, the default is filled inside the thread"
        ran.clear()
        with pytest.raises(TypeError, match="multiple values for argument 'func'"):
            self.run(example, "async", [self.threaded(ran, "own")], calls[1:], None, gated=False)
        messages = self.run(
            example, "sync", [self.threaded(ran, "own")], calls, self.no_delete(seen)
        )
        assert [m.status for m in messages] == ["success", "success"]
        assert sorted(ran) == [("x", "max"), ("x", "sum")]
        assert sorted(seen, key=str) == [
            ("t", {"q": "x", "func": "max"}),
            ("t", {"q": "x", "func": "sum"}),
        ]

    @pytest.mark.parametrize("kind", ["dict", "allow"])
    def test_a_thread_runner_name_is_refused_under_any_schema_that_lets_it_through(
        self, example, kind
    ):
        """Pins Y09: the refusal was tested only on a tool with no ``args_schema``."""
        ran, seen = [], []
        calls = [self.call("t", {"q": "x", "func": "m"})]
        messages = self.run(
            example, "async", [self.threaded(ran, kind)], calls, self.no_delete(seen)
        )
        assert ran == [] and seen == []
        assert "the invocation sets `func`, a parameter of the function LangChain runs" in (
            messages[0].content
        )
        messages = self.run(
            example, "sync", [self.threaded(ran, kind)], calls, self.no_delete(seen)
        )
        assert messages[0].status == "success" and ran == [("x", {"func": "m"})]

    # --- the published tools the census found, by their signatures and schemas ----------------

    def test_the_shapes_of_more_census_tools(self, example):
        """Signatures and schemas replicated from the packages' source (not imported):
        langchain-aws ``ThreadAwareNavigateTool`` (one of the seven with input),
        toolbox-langchain ``ToolboxTool``, langchain-writer ``NoCodeAppTool`` and a
        langchain-anchorbrowser task tool are refused; langchain-aws ``NovaSystemTool`` and
        langchain-cohere ``DirectlyAnswerTool`` run under the closed-schema exception."""
        from typing import Optional

        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel

        ran, seen = [], []

        class NavigateToolInput(BaseModel):
            url: str

        class TaskListInputSchema(BaseModel):
            page: str = "1"
            limit: str = "10"

        class EmptyInput(BaseModel):
            pass

        class Navigate(BaseTool):
            name: str = "navigate_browser"
            description: str = "d"
            args_schema: type[BaseModel] = NavigateToolInput

            def _run(self, url: str, config: Optional[RunnableConfig] = None, **_: Any) -> str:
                ran.append("navigate")
                return "ok"

        class Toolbox(BaseTool):
            name: str = "toolbox"
            description: str = "d"
            args_schema: type[BaseModel] = NavigateToolInput

            def _run(self, **kwargs: Any) -> str:
                ran.append("toolbox")
                return "ok"

        class NoCodeApp(BaseTool):
            name: str = "no_code_app"
            description: str = "d"

            def _run(self, *args: Any, **kwargs: Any) -> str:
                ran.append("app")
                return "ok"

        class AnchorTaskList(BaseTool):
            name: str = "anchor_task_list"
            description: str = "d"
            args_schema: type[BaseModel] = TaskListInputSchema

            def _run(self, **kwargs: Any) -> str:
                ran.append("anchor")
                return "ok"

        class NovaSystem(BaseTool):
            name: str = "nova_system"
            description: str = "d"
            args_schema: type[BaseModel] = EmptyInput

            def _run(self, **kwargs: Any) -> str:
                ran.append("nova")
                return "ok"

        class DirectlyAnswer(BaseTool):
            name: str = "directly_answer"
            description: str = "d"
            args_schema: type[BaseModel] = EmptyInput

            def _run(self, *args: Any, **kwargs: Any) -> str:
                ran.append("answer")
                return "ok"

        tools = [
            Navigate(),
            Toolbox(),
            NoCodeApp(),
            AnchorTaskList(),
            NovaSystem(),
            DirectlyAnswer(),
        ]
        calls = [
            self.call("navigate_browser", {"url": "u"}, "navigate"),
            self.call("toolbox", {"url": "u"}, "toolbox"),
            self.call("no_code_app", {}, "app"),
            self.call("anchor_task_list", {}, "anchor"),
            self.call("nova_system", {"x": 1}, "nova"),
            self.call("directly_answer", {"x": 1}, "answer"),
        ]
        messages = self.run(example, "sync", tools, calls, self.no_delete(seen))
        by_id = {m.tool_call_id: m for m in messages}
        assert "parameter `config` is absent from the argument schema" in by_id["navigate"].content
        assert "does not name `url`" in by_id["toolbox"].content
        assert self.ONLY_VARIADICS in by_id["app"].content
        assert "does not name `limit`" in by_id["anchor"].content
        assert by_id["nova"].status == by_id["answer"].status == "success"
        assert sorted(ran) == ["answer", "nova"]
        assert sorted(seen) == [("directly_answer", {}), ("nova_system", {})]


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestTheFrameworksAnswerIsNeverSetAside:
    """Review round 6 of the package (2026-11). Round five's rule read a parameter's raw
    annotation and stopped asking the frameworks what they fill, so a parameter LangChain
    or ToolNode really fills was read as the tool's own when its annotation was an alias
    or a forward reference. The frameworks' answers are back, the rule reads annotations
    as the frameworks do and opens a stated list of forms, and text is matched only where
    an annotation could not be resolved."""

    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_langgraph_round6_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    call = staticmethod(TestWhatRunsIsWhatWasJudged.call)
    no_delete = staticmethod(TestWhatRunsIsWhatWasJudged.no_delete)
    run = TestWhatRunsIsWhatWasJudged.run
    ROUTES = TestWhatRunsIsWhatWasJudged.ROUTES
    limit_under = staticmethod(TestWhatTheGateCannotReadIsRefused.limit_under)

    ONLY_VARIADICS = "the tool's innermost callable takes only `*args` or `**kwargs`"

    #: forwarders whose one named parameter the framework fills, written so that only
    #: ``typing.get_type_hints`` finds the type: (parameter, source, what fills it)
    HINTED = {
        "alias_pep563": (
            "cfg",
            "from __future__ import annotations\n"
            "def fn({self}*args, cfg: RC = None, **kwargs):\n    return export_rows(*args, **kwargs)\n",
        ),
        "quoted_alias": (
            "cfg",
            "def fn({self}*args, cfg: 'RC' = None, **kwargs):\n    return export_rows(*args, **kwargs)\n",
        ),
        "partly_quoted_runtime": (
            "rt",
            "def fn({self}*args, rt: Optional['ToolRuntime'] = None, **kwargs):\n"
            "    return export_rows(*args, **kwargs)\n",
        ),
        "alias_of_runtime_pep563": (
            "rt",
            "from __future__ import annotations\n"
            "def fn({self}*args, rt: TR = None, **kwargs):\n    return export_rows(*args, **kwargs)\n",
        ),
        "alias_of_marker_pep563": (
            "tenant",
            "from __future__ import annotations\n"
            "def fn({self}tenant: Annotated[str, Marker] = 't', **kwargs):\n"
            "    return export_rows(**kwargs)\n",
        ),
    }

    @classmethod
    def hinted(cls, ran, shape, kind):
        from typing import Annotated, Optional

        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
        from langgraph.prebuilt import ToolRuntime

        def export_rows(table: str = "users", limit: int = 1000000) -> str:
            ran.append((table, limit))
            return "exported"

        namespace = {
            "RC": RunnableConfig,
            "TR": ToolRuntime,
            "Marker": InjectedToolArg,
            "ToolRuntime": ToolRuntime,
            "Optional": Optional,
            "Annotated": Annotated,
            "export_rows": export_rows,
        }
        named, source = cls.HINTED[shape]
        exec(source.format(self="self, " if kind == "base" else ""), namespace)  # noqa: S102
        function = namespace["fn"]
        function.__doc__ = "Export rows of a table."
        if kind == "base":

            class Export(BaseTool):
                name: str = "export"
                description: str = "Export rows of a table."
                _run = function

            return named, Export()
        return named, StructuredTool.from_function(func=function, name="export", description="d")

    @pytest.mark.parametrize("kind", ["base", "function"])
    @pytest.mark.parametrize("shape", sorted(HINTED))
    @ROUTES
    def test_a_forwarder_whose_filled_parameter_only_type_hints_reveal_is_refused(
        self, example, route, shape, kind
    ):
        """The regression: ``cfg: RC`` under ``from __future__ import annotations``, a quoted
        alias and ``Optional['ToolRuntime']`` are a ``RunnableConfig`` and a ``ToolRuntime``
        to LangChain and ToolNode, which fill them. The rule read the raw annotation, took
        the parameter for the tool's own, and the policy judged ``{}`` while the function
        behind ran ``limit=1000000``."""
        ran, seen = [], []
        named, tool = self.hinted(ran, shape, kind)
        calls = [self.call("export", {})]
        messages = self.run(example, route, [tool], calls, self.limit_under(1000, seen))
        assert ran == [] and seen == [], "refused before the policy, and not run"
        assert self.ONLY_VARIADICS in messages[0].content
        assert f"(and `{named}`, which LangChain or the host reserves)" in messages[0].content
        self.run(example, route, [tool], calls, None, gated=False)
        assert ran == [("users", 1000000)], "ungated, the default nobody judged runs"

    @pytest.mark.parametrize("filled_by", ["langchain", "toolnode", "toolnode_base", "the_tool"])
    def test_what_the_frameworks_say_they_fill_refuses_a_forwarder_without_the_rule(
        self, example, monkeypatch, filled_by
    ):
        """The frameworks' own answers are an authority the rule adds to and never
        replaces. With the rule's own reading switched off (no reserved names, no
        annotation read), a forwarder is still refused for a parameter LangChain fills
        (``_framework_keys``), one ToolNode fills (``_node_fills``, also on a ``BaseTool``
        subclass) and one the tool marks injected (``_host_keys``)."""
        from typing import Annotated

        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool

        adapter = _adapter()
        monkeypatch.setattr(adapter, "_RESERVED_NAMES", frozenset())
        monkeypatch.setattr(adapter, "_filled_type", lambda *args, **kwargs: None)
        ran, seen = [], []

        def export_rows(table: str = "users", limit: int = 1000000) -> str:
            ran.append((table, limit))
            return "exported"

        if filled_by == "langchain":

            def forward(cfg: RunnableConfig, **kwargs: Any) -> str:
                return export_rows(**kwargs)

            named = "cfg"
        elif filled_by == "the_tool":

            def forward(tenant: Annotated[str, InjectedToolArg] = "t", **kwargs: Any) -> str:
                return export_rows(**kwargs)

            named = "tenant"
        else:

            def forward(runtime=None, **kwargs: Any) -> str:
                return export_rows(**kwargs)

            named = "runtime"
        if filled_by == "toolnode_base":

            def _run(self, runtime=None, **kwargs: Any) -> str:
                return export_rows(**kwargs)

            body = _run

            class Export(BaseTool):
                name: str = "export"
                description: str = "d"
                _run = body

            tool: Any = Export()
        else:
            tool = StructuredTool.from_function(func=forward, name="export", description="d")
        messages = self.run(
            example, "sync", [tool], [self.call("export", {})], self.limit_under(1000, seen)
        )
        assert ran == [] and seen == []
        assert f"(and `{named}`, which LangChain or the host reserves)" in messages[0].content

    # --- what an annotation is opened for, form by form ----------------------------------------

    def test_the_forms_an_annotation_is_opened_for(self, example):
        import typing
        from typing import Annotated, Dict, List, Literal, NewType, Optional, TypeVar, Union

        from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForToolRun
        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import InjectedToolArg
        from langgraph.prebuilt import ToolRuntime
        from pydantic import BaseModel

        filled_type = _adapter()._filled_type
        bound = TypeVar("bound", bound=RunnableConfig)
        constrained = TypeVar("constrained", str, ToolRuntime)
        free = TypeVar("free")
        new = NewType("new", RunnableConfig)
        plain_new = NewType("plain_new", str)
        Cfg = RunnableConfig

        class MyCfg(RunnableConfig, total=False):
            tenant: str

        class MyRuntime(ToolRuntime):
            pass

        class MyManager(CallbackManagerForToolRun):
            pass

        class MyMarker(InjectedToolArg):
            pass

        class RunManagerOptions(BaseModel):
            retries: int = 1

        class InjectedPrompt(str):
            pass

        namespace = {**globals(), **locals()}
        resolved = [
            Optional["RunnableConfig"],
            Optional["CallbackManagerForToolRun"],
            Optional["Cfg"],
            "Optional[RunnableConfig]",
            "Cfg",
            bound,
            constrained,
            new,
            MyCfg,
            MyRuntime,
            MyManager,
            Annotated[str, MyMarker],
            Annotated[str, MyMarker()],
            List[RunnableConfig],
            Dict[str, RunnableConfig],
            Union[RunnableConfig, str],
            Annotated[Annotated[RunnableConfig, "a"], "b"],
            ToolRuntime[Any, Any],
            typing.ForwardRef("ToolRuntime"),
            list["RunnableConfig"],
            dict[str, "Optional[Cfg]"],
            "'RunnableConfig'",
        ]
        for annotation in resolved:
            assert filled_type(annotation, namespace) == "resolved", annotation
        if sys.version_info >= (3, 12):
            scope = dict(namespace)
            exec("type TA = Optional[RunnableConfig]\ntype TB = Optional[str]", scope)  # noqa: S102
            assert filled_type(scope["TA"], namespace) == "resolved"
            assert filled_type(scope["TB"], namespace) is None
        # text is matched only where the name could not be resolved, and says so
        assert filled_type("AsyncParentRunManager", {}) == "by name"
        assert filled_type(Optional["ToolRuntime"], {}) == "by name"
        assert filled_type("RC", {}) is None, "an unresolved alias that does not hold the name"
        beside = Union[typing.ForwardRef("UnknownRunManager"), RunnableConfig]
        assert filled_type(beside, {}) == "resolved", "a resolved type beside an unresolved name"
        # not a filled type: a user type whose name holds one, a string that is metadata
        # or a Literal value, a handler (not a manager), an unbound variable
        own = [
            RunManagerOptions,
            InjectedPrompt,
            "RunManagerOptions",
            Annotated[str, "the RunnableConfig key to read"],
            Annotated[str, "Injected into the prompt"],
            Literal["RunManager", "ToolRuntime"],
            List[BaseCallbackHandler],
            free,
            plain_new,
            Optional[str],
            dict,
            "str",
        ]
        for annotation in own:
            assert filled_type(annotation, namespace) is None, annotation

    def test_a_type_that_could_not_be_resolved_is_matched_by_name_and_the_reason_says_so(
        self, example
    ):
        """Pins Q08 end to end: the annotation names a run manager the module never
        imported. It is text nobody can resolve; it is matched by the name it holds, and
        the refusal says that is how. The tool declares its schema, since LangChain cannot
        infer one from annotations it cannot resolve."""
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, ConfigDict

        ran, seen = [], []
        namespace: dict = {"ran": ran}
        exec(  # noqa: S102
            "def _run(self, *args, mgr: 'AsyncParentRunManager' = None, **kwargs):\n"
            "    ran.append(kwargs)\n    return 'ok'\n",
            namespace,
        )
        body = namespace["_run"]

        class Open(BaseModel):
            model_config = ConfigDict(extra="allow")

        class Export(BaseTool):
            name: str = "export"
            description: str = "d"
            args_schema: type = Open
            _run = body

        messages = self.run(
            example, "sync", [Export()], [self.call("export", {})], self.no_delete(seen)
        )
        assert ran == [] and seen == []
        assert "(and `mgr`, whose type could not be resolved and was matched by name" in (
            messages[0].content
        )
        assert "(and `mgr`, which LangChain or the host reserves)" not in messages[0].content

    def test_text_that_only_mentions_a_filled_type_does_not_make_a_parameter_reserved(
        self, example
    ):
        """A description inside ``Annotated``, a ``Literal`` value and a user type whose
        name holds "Injected" were matched as text and the tool refused as a forwarder,
        with a reason that was false. The parameter is the function's own: the call is
        judged with it and runs."""
        from typing import Annotated, Literal

        from langchain_core.tools import tool

        ran, seen = [], []

        class InjectedPrompt(str):
            pass

        @tool
        def described(
            q: Annotated[str, "the RunnableConfig key to read"] = "dq", **kwargs: Any
        ) -> str:
            """Described."""
            ran.append(("described", q))
            return "ok"

        @tool
        def chosen(q: Literal["RunManager", "x"] = "x", **kwargs: Any) -> str:
            """Chosen."""
            ran.append(("chosen", q))
            return "ok"

        @tool
        def typed(q: InjectedPrompt = InjectedPrompt("dq"), **kwargs: Any) -> str:
            """Typed."""
            ran.append(("typed", str(q)))
            return "ok"

        calls = [
            self.call("described", {}, "a"),
            self.call("chosen", {}, "b"),
            self.call("typed", {}, "c"),
        ]
        messages = self.run(
            example, "sync", [described, chosen, typed], calls, self.no_delete(seen)
        )
        assert [m.status for m in messages] == ["success"] * 3, [m.content for m in messages]
        assert sorted(ran) == [("chosen", "x"), ("described", "dq"), ("typed", "dq")]
        assert sorted(name for name, _ in seen) == ["chosen", "described", "typed"]

    def test_an_annotation_is_resolved_the_way_typing_resolves_it(self, example):
        """The annotations are read with ``typing.get_type_hints`` first. A forward
        reference that carries its own module is resolved there, though the function's
        globals do not hold the name: the type is known, and the reason says the name is
        reserved, not that it was matched by its text."""
        import typing
        from typing import Optional

        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, ConfigDict

        ran, seen = [], []
        ref = typing.ForwardRef("CallbackManagerForToolRun", module="langchain_core.callbacks")
        namespace: dict = {"ran": ran, "Optional": Optional, "ref": ref}
        exec(  # noqa: S102
            "def _run(self, *args, mgr: Optional[ref] = None, **kwargs):\n"
            "    ran.append(kwargs)\n    return 'ok'\n",
            namespace,
        )
        body = namespace["_run"]

        class Open(BaseModel):
            model_config = ConfigDict(extra="allow")

        class Export(BaseTool):
            name: str = "export"
            description: str = "d"
            args_schema: type = Open
            _run = body

        messages = self.run(
            example, "sync", [Export()], [self.call("export", {})], self.no_delete(seen)
        )
        assert ran == [] and seen == []
        assert "(and `mgr`, which LangChain or the host reserves)" in messages[0].content
        assert "could not be resolved" not in messages[0].content

    def test_one_name_that_cannot_be_resolved_does_not_stop_the_others_being_read(self, example):
        """When ``typing.get_type_hints`` fails for the function, each annotation is read
        on its own against the globals of the function's module: the alias is resolved to
        the manager type it names, and the unknown name is matched by its text."""
        from langchain_core.callbacks import CallbackManagerForToolRun
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, ConfigDict

        ran, seen = [], []
        namespace: dict = {"ran": ran, "M": CallbackManagerForToolRun}
        exec(  # noqa: S102
            "def _run(self, *args, mgr: 'M' = None, other: 'UnknownRunManager' = None, **kwargs):\n"
            "    ran.append(kwargs)\n    return 'ok'\n",
            namespace,
        )
        body = namespace["_run"]

        class Open(BaseModel):
            model_config = ConfigDict(extra="allow")

        class Export(BaseTool):
            name: str = "export"
            description: str = "d"
            args_schema: type = Open
            _run = body

        messages = self.run(
            example, "sync", [Export()], [self.call("export", {})], self.no_delete(seen)
        )
        assert ran == [] and seen == []
        assert "(and `mgr`, which LangChain or the host reserves)" in messages[0].content
        assert "(and `other`, whose type could not be resolved and was matched by name" in (
            messages[0].content
        )

    def test_annotations_are_read_on_the_innermost_function(self, example):
        """The forwarder rule reads the innermost function's parameters, so that is the
        function whose annotations and globals are read: through ``__wrapped__`` and
        through a ``functools.partial``."""
        import functools
        from typing import Annotated, Optional

        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import InjectedToolArg

        adapter = _adapter()
        namespace: dict = {"RC": RunnableConfig}
        exec("def inner(*args, cfg: 'RC' = None, **kwargs):\n    return 'ok'\n", namespace)  # noqa: S102
        inner = namespace["inner"]

        def wrapper(*args, **kwargs):
            return inner(*args, **kwargs)

        wrapper.__wrapped__ = inner  # type: ignore[attr-defined]
        for func in (inner, wrapper, functools.partial(wrapper), functools.partial(inner)):
            # Python 3.10 reads a ``= None`` default as Optional; later versions do not
            assert adapter._hints(func) in (
                {"cfg": RunnableConfig},
                {"cfg": Optional[RunnableConfig]},
            )
            assert adapter._namespace(func) is inner.__globals__
        assert adapter._hints(len) == {} and adapter._namespace(len) == {}

        def marked(tenant: Annotated[str, InjectedToolArg] = "t", **kwargs):
            return tenant

        assert adapter._hints(marked) == {"tenant": Annotated[str, InjectedToolArg]}, (
            "the metadata is kept: a marker is read from it"
        )

    # --- the single-input Tool -----------------------------------------------------------------

    def test_a_single_input_tool_whose_one_parameter_is_called_config_is_judged(self, example):
        """``gated_tool_node`` raised when the node was built, saying the function would
        never receive the model's value, while ``recusal_gates`` judged ``{'config': 'x'}``
        and the function received ``'x'``: a single-input ``Tool`` hands its one value to
        the first parameter by position. The node helper now builds, and both judge."""
        from langchain_core.tools import StructuredTool, Tool

        ran, seen = [], []

        def lookup(config):
            ran.append(config)
            return "found"

        tool = Tool(name="lookup", func=lookup, description="Look up.")
        node = example.gated_tool_node([tool], self.no_delete(seen))
        assert example.is_gated(node)
        messages = example.run_in_graph(node, [self.call("lookup", {"__arg1": "x"})])
        assert messages[0].status == "success", messages[0].content
        assert seen == [("lookup", {"config": "x"})] and ran == ["x"]

        def second(q, config="safe"):
            return q

        with pytest.raises(TypeError, match="its function's parameter `config`"):
            example.gated_tool_node(
                [Tool(name="lookup", func=second, description="d")], self.no_delete(seen)
            )
        structured = StructuredTool.from_function(func=lookup, name="lookup", description="d")
        with pytest.raises(TypeError, match="its function's parameter `config`"):
            example.gated_tool_node([structured], self.no_delete(seen))

    def test_a_single_input_forwarder_is_refused_whatever_its_one_parameter_is_called(
        self, example
    ):
        """Pins Q11: the single-input check must never treat a forwarder as one the model
        can put nothing through. ``Tool(func=f)`` with ``def f(config, **kwargs)`` hands
        the model's value to a reserved name and everything else on."""
        from langchain_core.tools import Tool

        ran, seen = [], []

        def forward(config, **kwargs):
            ran.append((config, kwargs))
            return "ok"

        tool = Tool(name="lookup", func=forward, description="Look up.")
        gates = example.recusal_gates(self.no_delete(seen))
        node = example.ToolNode([tool], **gates)
        messages = example.run_in_graph(node, [self.call("lookup", {"__arg1": "x"})])
        assert ran == [] and seen == []
        assert self.ONLY_VARIADICS in messages[0].content
        assert "(and `config`, which LangChain or the host reserves)" in messages[0].content

    # --- limits the README states, asserted on purpose ------------------------------------------

    def test_limitation_a_host_value_the_schema_omits_is_dropped_under_a_schema_with_fields(
        self, example
    ):
        """Documented in the package README: LangChain hands the function only the fields
        its ``args_schema`` declares. The function has a parameter ``runtime`` the schema
        does not declare and nothing marks injected, so whatever is put under that name is
        dropped and the function runs its own default for it. A constant, the same with no
        gate."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        ran, seen = [], []

        class Args(BaseModel):
            q: str

        def lookup(q: str, runtime: str = "drt") -> str:
            ran.append((q, runtime))
            return "found"

        tool = StructuredTool(name="lookup", description="d", func=lookup, args_schema=Args)
        calls = [self.call("lookup", {"q": "x"})]
        messages = self.run(example, "sync", [tool], calls, self.no_delete(seen))
        assert messages[0].status == "success", messages[0].content
        assert seen == [("lookup", {"q": "x"})] and ran == [("x", "drt")]
        ran.clear()
        self.run(example, "sync", [tool], calls, None, gated=False)
        assert ran == [("x", "drt")]

    def test_limitation_a_callbacks_parameter_of_a_basetool_runs_its_default_unseen(self, example):
        """Documented in the package README under what the gate does not see: on a
        ``BaseTool`` subclass nothing fills ``callbacks`` and the gate does not hold a
        parameter of that name to the schema, so its default runs unseen. It is still a
        real argument there: when the model sets it, the call is judged with it."""
        from langchain_core.tools import BaseTool

        ran, seen = [], []

        class Lookup(BaseTool):
            name: str = "lookup"
            description: str = "d"

            def _run(self, callbacks: str = "dcb") -> str:
                ran.append(callbacks)
                return "found"

        calls = [self.call("lookup", {}, "a"), self.call("lookup", {"callbacks": "m"}, "b")]
        messages = self.run(example, "sync", [Lookup()], calls, self.no_delete(seen))
        assert [m.status for m in messages] == ["success", "success"]
        assert sorted(seen, key=str) == [("lookup", {"callbacks": "m"}), ("lookup", {})]
        assert sorted(ran) == ["dcb", "m"]

    def test_the_shapes_of_two_more_census_tools(self, example):
        """Signatures replicated from the packages' source (not imported):
        langchain-community's Databricks ``UCFunctionToolkit`` builds a ``StructuredTool``
        over ``def func(*args, **kwargs)`` under a generated schema, and langchain-snowflake's
        MCP integration builds ``Tool(func=wrapper.run)`` with ``run(self, *args,
        **kwargs)``. Both are refused."""
        from langchain_core.tools import StructuredTool, Tool
        from pydantic import BaseModel

        ran, seen = [], []

        class Generated(BaseModel):
            x: int

        def func(*args: Any, **kwargs: Any) -> str:
            ran.append(kwargs)
            return "ok"

        class Wrapper:
            def run(self, *args: Any, **kwargs: Any) -> str:
                ran.append(args)
                return "ok"

        tools = [
            StructuredTool(name="uc_function", description="d", func=func, args_schema=Generated),
            Tool(name="snowflake_mcp", func=Wrapper().run, description="d"),
        ]
        calls = [
            self.call("uc_function", {"x": 1}, "a"),
            self.call("snowflake_mcp", {"__arg1": "q"}, "b"),
        ]
        node = example.ToolNode(tools, **example.recusal_gates(self.no_delete(seen)))
        messages = {m.tool_call_id: m.content for m in example.run_in_graph(node, calls)}
        assert ran == [] and seen == []
        assert "takes `*args` or `**kwargs` and does not name `x`" in messages["a"]
        assert "does not take its input as a first named parameter" in messages["b"]
