"""Regression tests for the offline LangGraph audit-trail example (#77).

Every call a LangGraph ``ToolNode`` proposes through ``recusal_audited_gate`` lands on a
hash-chained log exactly once, labeled ``langgraph.toolnode`` and keyed by LangGraph's own
tool call id, and the audited gate decides exactly as the plain ``recusal_gate`` does.
``ToolNode`` runs one message's calls in parallel, so entries are matched by id, never by
position.
"""

import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "langgraph_audit.py"
SURFACE = "langgraph.toolnode"


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
    assert "langgraph_audit.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_pinned_versions_are_explicit_in_the_example():
    text = EXAMPLE.read_text(encoding="utf-8")
    for pin in ("langgraph==1.2.20", "langchain-core==1.7.7"):
        assert pin in text


def test_both_ci_lanes_run_this_file_with_the_langgraph_example():
    # the issue asked for no new jobs: the two LangGraph jobs and the weekly lane run it
    for workflow in ("ci.yml", "compat.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert text.count("tests/test_langgraph_audit.py") == 2, workflow


def test_the_works_with_table_marks_the_audit_trail_tested():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "| LangGraph audit trail | Tested | [`examples/langgraph_audit.py`]" in readme
    assert "issues/77" not in readme.split("<!-- works-with:end -->", 1)[0]


def _tamper_copies(path):
    """One copy of the log per entry, each with that single entry's decision edited."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for index in range(len(lines)):
        edited = list(lines)
        entry = json.loads(edited[index])
        entry["action"]["decision"] = "allow" if entry["action"]["decision"] != "allow" else "deny"
        edited[index] = json.dumps(entry, sort_keys=True)
        copy = path.with_name(f"tampered-{index}.jsonl")
        copy.write_text("\n".join(edited) + "\n", encoding="utf-8")
        yield index, copy


def _call(name, args, ident="c1"):
    return {"name": name, "args": args, "id": ident, "type": "tool_call"}


THREE = [
    _call("echo", {"text": "hello"}, "c1"),
    _call("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
    _call("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
]


def _corpus():
    def raising(tool_name, tool_input):
        raise RuntimeError("policy exploded")

    def silent(tool_name, tool_input):
        return None  # no objection: PASS

    def ambiguous(tool_name, tool_input):
        return [{"check": "vague"}]  # neither status nor passed: strict fails closed

    def warning_only(tool_name, tool_input):
        from recusal import Finding

        return [Finding.fail("soft", severity="WARNING", message="noted")]

    def error_level(tool_name, tool_input):
        from recusal import Finding

        return [Finding.fail("hard", severity="ERROR", message="retry this")]

    long_sql = "SELECT " + "x" * 50_000 + " FROM t"
    nested = {"text": {"deep": [1, {"deeper": [None, 2.5, "s"]}], "flag": True}}
    return {
        "sql-scope-three": (None, THREE),
        "sql-scope-none-value": (None, [_call("run_sql", {"sql": None})]),
        "sql-scope-long": (None, [_call("run_sql", {"sql": long_sql})]),
        "sql-scope-nested": (None, [_call("echo", nested)]),
        "sql-scope-missing-arg": (None, [_call("echo", {})]),
        "sql-scope-unknown-tool": (None, [_call("nope", {"a": 1})]),
        "raising": (raising, THREE),
        "silent": (silent, THREE),
        "ambiguous": (ambiguous, THREE),
        "warning-only": (warning_only, THREE),
        "error-level": (error_level, THREE),
    }


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangGraph requires Python 3.10+")
class TestLangGraphAudit:
    @pytest.fixture
    def example(self):
        pytest.importorskip("langgraph")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location("recusal_langgraph_audit_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def gate(self, example):
        """The plain gate example, the SAME module object the audited one imports from."""
        return importlib.import_module("examples.langgraph_gate")

    @pytest.fixture
    def log_path(self, tmp_path):
        return tmp_path / "langgraph-audit.jsonl"

    @pytest.fixture
    def log(self, example, log_path):
        return example.AuditLog(path=str(log_path))

    call = staticmethod(_call)

    THREE = THREE

    @staticmethod
    def by_id(log):
        entries = {}
        for entry in log.entries:
            ident = entry["action"]["tool_use_id"]
            assert ident not in entries, f"{ident} recorded twice"
            entries[ident] = entry
        return entries

    @staticmethod
    def observed(messages, executed):
        """What the agent and the tools saw, order-independent for the parallel ToolNode."""
        seen = sorted((m.tool_call_id, m.name, m.status, str(m.content)) for m in messages)
        return seen, sorted((name, json.dumps(args, sort_keys=True)) for name, args in executed)

    # --- every call recorded once, allowed and refused alike -------------------------------

    def test_every_call_is_recorded_once_with_surface_and_id(self, example, log):
        messages, executed = example.run_calls(self.THREE, log, actor="demo-agent")
        assert {m.tool_call_id: m.status for m in messages} == {
            "c1": "success",
            "c2": "error",
            "c3": "success",
        }
        assert sorted(name for name, _ in executed) == ["echo", "run_sql"]
        entries = self.by_id(log)
        assert set(entries) == {"c1", "c2", "c3"}
        for entry in entries.values():
            assert entry["action"]["surface"] == SURFACE
            assert entry["actor"] == "demo-agent"
        assert {k: v["action"]["decision"] for k, v in entries.items()} == {
            "c1": "defer",
            "c2": "deny",
            "c3": "defer",
        }
        assert {k: v["decision"] for k, v in entries.items()} == {
            "c1": "PASS",
            "c2": "FAIL",
            "c3": "PASS",
        }
        assert entries["c2"]["action"]["tool"] == "run_sql"
        assert "top-level WHERE" in entries["c2"]["reasons"]

    def test_async_gate_records_every_call_once_with_surface_and_id(self, example, log):
        messages, executed = asyncio.run(
            example.run_calls_async(self.THREE, log, actor="demo-agent")
        )
        assert {m.tool_call_id: m.status for m in messages} == {
            "c1": "success",
            "c2": "error",
            "c3": "success",
        }
        assert sorted(name for name, _ in executed) == ["echo", "run_sql"]
        entries = self.by_id(log)
        assert set(entries) == {"c1", "c2", "c3"}
        assert all(e["action"]["surface"] == SURFACE for e in entries.values())
        assert entries["c2"]["action"]["decision"] == "deny"

    def test_entries_form_one_chain_whatever_the_completion_order(self, example, log):
        many = [self.call("echo", {"text": f"t{i}"}, f"id{i}") for i in range(12)]
        many.insert(5, self.call("run_sql", {"sql": "DELETE FROM t"}, "refused"))
        asyncio.run(example.run_calls_async(many, log))
        assert len(self.by_id(log)) == 13
        assert [e["seq"] for e in log.entries] == list(range(13))
        for previous, entry in zip(log.entries, log.entries[1:]):
            assert entry["prev_hash"] == previous["hash"]
        from recusal.audit import verify

        ok, problems = verify(log.entries)
        assert ok, problems

    # --- decision equivalence with the plain gate -----------------------------------------

    @pytest.mark.parametrize("case", sorted(_corpus()))
    def test_sync_decisions_match_the_plain_gate(self, example, gate, log, case):
        policy, calls = _corpus()[case]
        kwargs = {} if policy is None else {"policy": policy}
        plain = self.observed(*gate.run_calls(calls, **kwargs))
        audited = self.observed(*example.run_calls(calls, log, **kwargs))
        assert audited == plain
        assert len(self.by_id(log)) == len(calls)

    @pytest.mark.parametrize("case", sorted(_corpus()))
    def test_async_decisions_match_the_plain_gate(self, example, gate, log, case):
        policy, calls = _corpus()[case]
        kwargs = {} if policy is None else {"policy": policy}
        plain = self.observed(*asyncio.run(gate.run_calls_async(calls, **kwargs)))
        audited = self.observed(*asyncio.run(example.run_calls_async(calls, log, **kwargs)))
        assert audited == plain
        assert len(self.by_id(log)) == len(calls)

    def test_a_refused_call_does_not_run(self, example, log):
        messages, executed = example.run_calls(
            [self.call("run_sql", {"sql": "DELETE FROM audit_log"})], log
        )
        assert executed == []
        assert messages[0].status == "error"
        assert "Recusal refused" in str(messages[0].content)
        assert self.by_id(log)["c1"]["action"]["decision"] == "deny"

    # --- the record is part of the control ---------------------------------------------

    def test_policy_that_raises_is_denied_nothing_runs_and_the_denial_is_recorded(
        self, example, gate, log, log_path
    ):
        def broken(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        messages, executed = example.run_calls(
            [self.call("echo", {"text": "x"})], log, policy=broken
        )
        assert executed == []
        assert messages[0].status == "error"
        plain, _ = gate.run_calls([self.call("echo", {"text": "x"})], policy=broken)
        assert str(messages[0].content) == str(plain[0].content)
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["decision"] == "FAIL"
        assert [f["message"] for f in entry["failures"]] == ["RECUSAL_POLICY_ERROR (RuntimeError)"]
        assert "policy exploded" not in str(messages[0].content)
        assert "policy exploded" not in log_path.read_text(encoding="utf-8")

    def test_unwritable_log_refuses_every_call_and_no_tool_runs(self, example, log, log_path):
        example.run_calls([self.call("echo", {"text": "warm-up"}, "w")], log)
        assert log_path.is_file()
        log_path.chmod(stat.S_IREAD)
        try:
            with open(log_path, "a", encoding="utf-8"):
                pass
        except PermissionError:
            pass
        else:
            pytest.skip("this filesystem does not enforce a read-only file")
        try:
            messages, executed = example.run_calls(self.THREE, log)
        finally:
            log_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        assert executed == []
        assert [m.status for m in messages] == ["error", "error", "error"]
        for message in messages:
            assert "audit log unavailable" in str(message.content)
            assert "the record is part of the control" in str(message.content)
        # the file holds only the warm-up entry, and it still verifies
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    def test_failing_sink_refuses_the_call_and_no_tool_runs(self, example, log_path):
        class Offline:
            def write(self, entry):
                raise OSError("anchor store unreachable")

        log = example.AuditLog(path=str(log_path), sinks=[Offline()])
        messages, executed = asyncio.run(example.run_calls_async(self.THREE, log))
        assert executed == []
        assert all(m.status == "error" for m in messages)
        assert all("RECUSAL_AUDIT_UNAVAILABLE" in str(m.content) for m in messages)
        assert not any("anchor store unreachable" in str(m.content) for m in messages)

    # --- the file: fingerprints, no argument text, tamper detection -----------------------

    # --- review 6: the serializer probe on the audited routes, and redaction ----------------

    @staticmethod
    def serializer_tool(ran):
        from langchain_core.tools import tool
        from pydantic import BaseModel, field_serializer

        class Query(BaseModel):
            sql: str

            @field_serializer("sql")
            def public_sql(self, value):
                return "SELECT * FROM audit_log"

        class Args(BaseModel):
            query: Query

        @tool("query", args_schema=Args)
        def query(query: Query) -> str:
            """Record SQL only."""
            ran.append(query.sql)
            return "recorded"

        return query

    @staticmethod
    def select_only(name, args):
        from recusal import Finding

        if args["query"]["sql"] == "SELECT * FROM audit_log":
            return []
        return [Finding.fail("sql", severity="CRITICAL", message="not the approved SELECT")]

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_the_audited_gate_sees_through_a_display_serializer_and_records_the_deny(
        self, example, gate, log, route
    ):
        ran = []
        calls = [self.call("query", {"query": {"sql": "DELETE FROM audit_log"}}, "c1")]
        if route == "sync":
            node = example.ToolNode(
                [self.serializer_tool(ran)],
                wrap_tool_call=example.recusal_audited_gate(self.select_only, log),
            )
            messages = gate.run_in_graph(node, calls)
        else:
            node = example.ToolNode(
                [self.serializer_tool(ran)],
                **example.recusal_audited_gates(self.select_only, log),
            )
            messages = asyncio.run(gate.run_in_graph_async(node, calls))
        assert ran == []
        assert messages[0].status == "error" and "not the approved SELECT" in messages[0].content
        assert [e["action"]["decision"] for e in log.entries] == ["deny"]
        assert log.entries[0]["action"]["tool_use_id"] == "c1"

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_suppressed_parameter_is_judged_and_recorded_at_the_executed_value(
        self, example, gate, log, route
    ):
        from langchain_core.tools import tool
        from pydantic import BaseModel, Field

        from recusal import Finding

        class Args(BaseModel):
            amount: int = Field(default=100, exclude=True)

        ran, seen = [], []

        @tool("charge", args_schema=Args)
        def charge(amount: int = 100) -> str:
            """Record a hypothetical charge; contact no service."""
            ran.append(amount)
            return "recorded"

        def budget(name, args):
            seen.append(dict(args))
            if args.get("amount", 0) > 10:
                return [Finding.fail("budget", severity="CRITICAL", message="over 10")]
            return []

        calls = [self.call("charge", {"amount": 5}, "c1")]
        if route == "sync":
            node = example.ToolNode(
                [charge], wrap_tool_call=example.recusal_audited_gate(budget, log)
            )
            messages = gate.run_in_graph(node, calls)
        else:
            node = example.ToolNode([charge], **example.recusal_audited_gates(budget, log))
            messages = asyncio.run(gate.run_in_graph_async(node, calls))
        assert seen == [{"amount": 100}] and ran == []
        assert "over 10" in messages[0].content
        assert [e["action"]["decision"] for e in log.entries] == ["deny"]

    @staticmethod
    def budget(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append(dict(args))
            if args.get("amount", 0) > 10:
                return [Finding.fail("budget", severity="CRITICAL", message="over 10")]
            return []

        return policy

    def audited(self, example, gate, log, route, tool, calls, policy):
        gate.EXECUTED.clear()
        if route == "sync":
            node = example.ToolNode(
                [tool], wrap_tool_call=example.recusal_audited_gate(policy, log)
            )
            return gate.run_in_graph(node, calls)
        node = example.ToolNode([tool], **example.recusal_audited_gates(policy, log))
        return asyncio.run(gate.run_in_graph_async(node, calls))

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_the_audited_route_completes_from_the_callable_the_route_runs(
        self, example, gate, log, route
    ):
        """``func`` and ``coroutine`` with different defaults behind an excluding schema: the
        audited async gate must name its route, or the sync default would be judged and
        recorded while the coroutine's ran."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        class Args(BaseModel):
            amount: int = Field(default=100, exclude=True)

        ran = []

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
        seen = []
        messages = self.audited(
            example,
            gate,
            log,
            route,
            tool,
            [self.call("charge", {"amount": 7}, "c1")],
            self.budget(seen),
        )
        decisions = [e["action"]["decision"] for e in log.entries]
        if route == "sync":
            assert seen == [{"amount": 5}] and ran == [("sync", 5)]
            assert messages[0].status == "success" and decisions != ["deny"]
        else:
            assert seen == [{"amount": 100}] and ran == []
            assert "over 10" in messages[0].content and decisions == ["deny"]

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_callable_parameter_the_schema_omits_is_refused_and_recorded_as_one_deny(
        self, example, gate, log, route
    ):
        """The 2026-11-06 review's audited reproduction: the record bound the fingerprint of
        ``{"label": "review"}`` while ``amount=100`` executed. The call is now refused before
        the policy, nothing runs, and the log holds one deny for it."""
        seen = []
        messages = self.audited(
            example,
            gate,
            log,
            route,
            gate.narrow_charge_tool(),
            [self.call("charge", {"label": "review"}, "c1")],
            self.budget(seen),
        )
        assert seen == [] and gate.EXECUTED == []
        text = messages[0].content
        assert "the callable's parameter `amount` is absent from the argument schema" in text
        assert "100" not in text
        actions = [(e["action"]["decision"], e["action"]["tool_use_id"]) for e in log.entries]
        assert actions == [("deny", "c1")]

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_permitted_complete_invocation_is_recorded_at_the_fingerprint_of_what_ran(
        self, example, gate, log, route
    ):
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        from recusal.authorization import fingerprint

        class Args(BaseModel):
            label: str
            amount: int = 100

        ran = []

        def charge(label: str, amount: int = 100) -> str:
            """Record a hypothetical charge; contact no service."""
            ran.append({"label": label, "amount": amount})
            return "recorded"

        tool = StructuredTool.from_function(charge, name="charge", args_schema=Args)
        seen = []
        messages = self.audited(
            example,
            gate,
            log,
            route,
            tool,
            [self.call("charge", {"label": "review", "amount": 5}, "c1")],
            self.budget(seen),
        )
        assert messages[0].status == "success", messages[0].content
        assert seen == [{"label": "review", "amount": 5}] and ran == seen
        [entry] = log.entries
        assert entry["action"]["decision"] != "deny"
        assert entry["action"]["input_sha256"] == fingerprint(ran[0])

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_root_validator_built_from_host_state_is_recorded_without_the_state(
        self, example, gate, log, log_path, route
    ):
        """The follow-up review's probe: a model_validator composing its ValueError from
        injected state reached both the ToolMessage and the audit reason."""
        from typing import Annotated

        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langgraph.prebuilt import InjectedState
        from pydantic import BaseModel, model_validator

        class Args(BaseModel):
            sql: str
            state: Annotated[dict, InjectedState]

            @model_validator(mode="after")
            def check_host_state(self):
                raise ValueError("invalid host context: " + self.state["messages"][0].content)

        ran = []

        @tool("query", args_schema=Args)
        def query(sql: str, state: Annotated[dict, InjectedState]) -> str:
            """Record only; the failing validator runs first."""
            ran.append(sql)
            return "recorded"

        calls = [self.call("query", {"sql": "SELECT 1"}, "c1")]
        if route == "sync":
            node = example.ToolNode(
                [query], wrap_tool_call=example.recusal_audited_gate(lambda n, a: [], log)
            )
            result = gate._graph(node, calls).invoke(
                {"messages": [HumanMessage(content="HOST_SECRET_SENTINEL")]}
            )
        else:
            node = example.ToolNode([query], **example.recusal_audited_gates(lambda n, a: [], log))
            result = asyncio.run(
                gate._graph(node, calls).ainvoke(
                    {"messages": [HumanMessage(content="HOST_SECRET_SENTINEL")]}
                )
            )
        message = [m for m in result["messages"] if getattr(m, "tool_call_id", None) == "c1"][0]
        assert ran == [] and message.status == "error"
        assert "HOST_SECRET_SENTINEL" not in message.content
        assert "model: value_error" in message.content
        text = log_path.read_text(encoding="utf-8")
        assert "HOST_SECRET_SENTINEL" not in text
        assert [e["action"]["decision"] for e in log.entries] == ["deny"]

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_the_record_binds_what_executed_whatever_the_policy_did_to_its_copy(
        self, example, gate, log, route
    ):
        """The follow-up review's probe: the policy mutated its detached view to DELETE,
        the tool ran SELECT, and the record fingerprinted DELETE."""
        from recusal.claude_code import _input_fingerprint

        def mutate(name, args):
            args["sql"] = "DELETE FROM audit_log"
            return []

        calls = [self.call("run_sql", {"sql": "SELECT * FROM audit_log"}, "c1")]
        if route == "sync":
            messages, executed = example.run_calls(calls, log, policy=mutate)
        else:
            messages, executed = asyncio.run(example.run_calls_async(calls, log, policy=mutate))
        assert executed == [("run_sql", {"sql": "SELECT * FROM audit_log"})]
        assert messages[0].status == "success"
        recorded = log.entries[0]["action"]["input_sha256"]
        assert recorded == _input_fingerprint({"sql": "SELECT * FROM audit_log"})
        assert recorded != _input_fingerprint({"sql": "DELETE FROM audit_log"})

    def test_the_record_binds_a_nested_invocation_as_executed(self, example, gate, log):
        from langchain_core.tools import tool
        from pydantic import BaseModel

        from recusal.claude_code import _input_fingerprint

        class Query(BaseModel):
            sql: str

        class Args(BaseModel):
            query: Query

        ran = []

        @tool("query", args_schema=Args)
        def query(query: Query) -> str:
            """Record SQL only."""
            ran.append(query.sql)
            return "recorded"

        def mutate(name, args):
            args["query"]["sql"] = "DELETE FROM audit_log"
            return []

        node = example.ToolNode([query], wrap_tool_call=example.recusal_audited_gate(mutate, log))
        gate.run_in_graph(node, [self.call("query", {"query": {"sql": "SELECT 1"}}, "c1")])
        assert ran == ["SELECT 1"]
        assert log.entries[0]["action"]["input_sha256"] == _input_fingerprint(
            {"query": {"sql": "SELECT 1"}}
        )

    def test_a_host_state_validation_failure_is_recorded_without_the_state(
        self, example, gate, log, log_path
    ):
        from typing import Annotated

        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langgraph.prebuilt import InjectedState

        @tool
        def wants_int_state(sql: str, state: Annotated[int, InjectedState]) -> str:
            """Declare a host-state shape the graph does not provide."""
            return "recorded"

        node = example.ToolNode(
            [wants_int_state], wrap_tool_call=example.recusal_audited_gate(lambda n, a: [], log)
        )
        calls = [self.call("wants_int_state", {"sql": "SELECT 1"}, "c1")]
        graph = gate._graph(node, calls)
        result = graph.invoke({"messages": [HumanMessage(content="HOST_SECRET_SENTINEL")]})
        message = [m for m in result["messages"] if getattr(m, "tool_call_id", None) == "c1"][0]
        assert message.status == "error" and "HOST_SECRET_SENTINEL" not in message.content
        assert [e["action"]["decision"] for e in log.entries] == ["deny"]
        text = log_path.read_text(encoding="utf-8")
        assert "HOST_SECRET_SENTINEL" not in text and "SELECT 1" not in text
        assert "recusal_unjudgeable_input" in text

    def test_no_tool_argument_text_appears_in_the_log(self, example, log, log_path):
        secret = "DELETE FROM customers WHERE ssn = 'MARKER-7f3e9'"
        calls = [
            self.call("run_sql", {"sql": secret}, "s1"),
            self.call("echo", {"text": "MARKER-echo-text"}, "s2"),
        ]
        example.run_calls(calls, log)
        text = log_path.read_text(encoding="utf-8")
        assert "MARKER" not in text
        assert "customers" not in text
        entries = self.by_id(log)
        for call in calls:
            canonical = json.dumps(
                call["args"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            assert entries[call["id"]]["action"]["input_sha256"] == expected
            assert entries[call["id"]]["action"]["control"]["recusal_version"]

    def test_caller_control_is_recorded_but_cannot_forge_the_version(self, example, log):
        control = {"policy_id": "sql-scope", "policy_version": "3", "recusal_version": "9.9.9"}
        example.run_calls([self.call("echo", {"text": "x"})], log, control=control)
        recorded = self.by_id(log)["c1"]["action"]["control"]
        assert recorded["policy_id"] == "sql-scope"
        assert recorded["policy_version"] == "3"
        assert recorded["recusal_version"] != "9.9.9"

    def test_log_verifies_intact_and_editing_any_entry_breaks_it(self, example, log, log_path):
        example.run_calls(self.THREE, log, actor="demo-agent")
        code, text = example.verify_with_cli(str(log_path))
        assert code == 0, text
        assert "intact - 3 entries" in text
        for index, copy in _tamper_copies(log_path):
            code, text = example.verify_with_cli(str(copy))
            assert code == 1, (index, text)
            assert "BROKEN" in text

    def test_the_example_tamper_helper_targets_the_refused_call(self, example, log, log_path):
        example.run_calls(self.THREE, log)
        example.tamper(str(log_path), "c2")
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        edited = [e for e in entries if e["action"]["tool_use_id"] == "c2"]
        assert edited and edited[0]["action"]["decision"] == "allow"
        code, _ = example.verify_with_cli(str(log_path))
        assert code == 1

    def test_truncating_the_tail_is_caught_by_an_external_head(self, example, log, log_path):
        example.run_calls(self.THREE, log)
        head = (log.entries[-1]["seq"] + 1, log.entries[-1]["hash"])
        lines = log_path.read_text(encoding="utf-8").splitlines()
        log_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        from recusal.audit import AuditIntegrityError

        with pytest.raises(AuditIntegrityError):
            example.AuditLog(path=str(log_path), expected_head=head)

    # --- refusals made before, and after, the judging step: one entry each -----------------

    @pytest.mark.parametrize("ident", ["", None], ids=["empty", "none"])
    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_call_without_an_id_is_refused_and_recorded_once_without_an_id(
        self, example, log, route, ident
    ):
        """Review round 1: an empty id made ``decide`` raise, which the gate reported as a
        policy error; the policy was never called and nothing was written. The plain gate
        ran the same call."""
        seen = []

        def recording(name, args):
            seen.append(name)
            return []

        calls = [_call("run_sql", {"sql": "SELECT * FROM audit_log"}, ident)]
        if route == "sync":
            messages, executed = example.run_calls(calls, log, policy=recording)
        else:
            messages, executed = asyncio.run(example.run_calls_async(calls, log, policy=recording))
        assert executed == [] and seen == []
        assert messages[0].status == "error"
        assert "the tool call has no id" in messages[0].content
        assert "RECUSAL_POLICY_ERROR" not in messages[0].content
        assert len(log.entries) == 1
        action = log.entries[0]["action"]
        assert action["decision"] == "deny" and action["tool"] == "run_sql"
        assert "tool_use_id" not in action
        assert action["reason"] == messages[0].content

    def test_a_call_whose_name_is_not_the_tool_that_would_run_is_recorded_under_the_tool(
        self, example, gate, log
    ):
        seen = []

        def recording(name, args):
            seen.append(name)
            return []

        hook = example.recusal_audited_gate(recording, log)

        def repair(request, execute):
            return hook(request.override(tool=gate.run_sql), execute)

        node = gate.ToolNode([gate.echo, gate.run_sql], wrap_tool_call=repair)
        gate.EXECUTED.clear()
        messages = gate.run_in_graph(node, [_call("Run_SQL", {"sql": "DELETE FROM audit_log"})])
        assert gate.EXECUTED == [] and seen == []
        assert "the call names another tool than this one" in messages[0].content
        entry = self.by_id(log)["c1"]
        assert len(log.entries) == 1
        assert entry["action"]["tool"] == "run_sql" and entry["action"]["decision"] == "deny"
        assert "Run_SQL" not in json.dumps(entry)

    def test_an_answer_that_never_passed_the_judging_step_is_one_recorded_refusal(
        self, example, gate, log, monkeypatch
    ):
        """A ToolNode whose executor ran the registered tool instead of the judging copy:
        the tool has run, the gate withholds the answer, and the one entry says the call was
        never judged."""
        original = gate.ToolNode._execute_tool_sync

        def drifted(self, request, input_type, config):
            registered = self.tools_by_name[request.tool_call["name"]]
            return original(self, request.override(tool=registered), input_type, config)

        monkeypatch.setattr(gate.ToolNode, "_execute_tool_sync", drifted)
        messages, executed = example.run_calls(
            [_call("run_sql", {"sql": "DELETE FROM audit_log"})], log
        )
        assert executed == [("run_sql", {"sql": "DELETE FROM audit_log"})]
        assert messages[0].status == "error"
        entry = self.by_id(log)["c1"]
        assert len(log.entries) == 1 and entry["action"]["decision"] == "deny"
        assert "without passing the gate's judging step" in entry["action"]["reason"]
        assert entry["action"]["reason"] == messages[0].content

    def test_limitation_a_call_toolnode_fails_before_the_tool_is_reached_leaves_no_entry(
        self, example, gate, log
    ):
        """Documented in the package README: ToolNode injects the host's arguments before
        it invokes the tool. When that fails (a required ``InjectedState`` field the state
        does not have) the tool does not run, the gate has decided nothing, and nothing is
        written."""
        from typing import Annotated

        from langchain_core.tools import tool
        from langgraph.prebuilt import InjectedState

        ran, seen = [], []

        @tool
        def act(what: str, role: Annotated[str, InjectedState("role")]) -> str:
            """Act; the role is required from graph state."""
            ran.append((what, role))
            return "ok"

        def recording(name, args):
            seen.append(name)
            return []

        node = example.audited_tool_node([act], recording, log, handle_tool_errors=True)
        messages = gate.run_in_graph(node, [_call("act", {"what": "x"})])
        assert messages[0].status == "error" and "KeyError" in messages[0].content
        assert ran == [] and seen == [] and log.entries == []

    def test_a_record_that_cannot_be_started_refuses_the_call(self, example, gate, log):
        """``decide`` raises on recording arguments it does not accept (an empty surface).
        The gate's own answer to that is a refusal; nothing runs."""
        node = gate.ToolNode(
            [gate.echo, gate.run_sql],
            wrap_tool_call=example.recusal_audited_gate(lambda name, args: [], log, surface=""),
        )
        gate.EXECUTED.clear()
        messages = gate.run_in_graph(node, [_call("run_sql", {"sql": "SELECT * FROM audit_log"})])
        assert gate.EXECUTED == []
        assert messages[0].status == "error"
        assert "RECUSAL_POLICY_ERROR (ValueError)" in messages[0].content

    @pytest.mark.parametrize("ident", [5, b"c1", ["c1"]], ids=["int", "bytes", "list"])
    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_an_id_that_is_not_a_string_is_one_recorded_refusal_without_an_id(
        self, example, gate, log, route, ident
    ):
        """A model message cannot carry most of these, so the hooks are called directly."""
        from langgraph.prebuilt.tool_node import ToolCallRequest

        seen = []

        def recording(name, args):
            seen.append(name)
            return []

        call = {"name": "run_sql", "args": {"sql": "SELECT 1"}, "id": ident, "type": "tool_call"}
        request = ToolCallRequest(tool_call=call, tool=gate.run_sql, state={}, runtime=None)
        gates = example.recusal_audited_gates(recording, log)
        gate.EXECUTED.clear()
        if route == "async":

            async def never(judged):
                raise AssertionError("a refused call is not executed")

            answer = asyncio.run(gates["awrap_tool_call"](request, never))
        else:

            def never(judged):
                raise AssertionError("a refused call is not executed")

            answer = gates["wrap_tool_call"](request, never)
        assert seen == [] and gate.EXECUTED == []
        assert answer.status == "error" and answer.tool_call_id == ""
        assert "the tool call has no id" in answer.content
        assert [e["action"]["decision"] for e in log.entries] == ["deny"]
        assert "tool_use_id" not in log.entries[0]["action"]

    # --- review round 2: what the gate cannot read, and an answer that is not its own -------

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_walk_failure_of_a_kind_the_gate_does_not_know_is_one_recorded_refusal(
        self, example, gate, log, route, monkeypatch
    ):
        """Whatever the kernel's walk raises is a refusal before the policy, written once
        with the call's id: the adapter does not need to know the type."""
        adapter = importlib.import_module("recusal_langgraph._gate")

        class AppliesAnotherDecoratorInside(ValueError):
            pass

        def unreadable(func):
            raise AppliesAnotherDecoratorInside("a wrapper holds another wrapper of its function")

        monkeypatch.setattr(adapter, "_reached_parameters", unreadable)
        seen = []
        messages = self.audited(
            example,
            gate,
            log,
            route,
            gate.run_sql,
            [self.call("run_sql", {"sql": "SELECT * FROM audit_log"}, "c1")],
            self.budget(seen),
        )
        assert seen == [] and gate.EXECUTED == []
        assert "cannot be inspected for its parameters (AppliesAnotherDecoratorInside)" in (
            messages[0].content
        )
        actions = [(e["action"]["decision"], e["action"]["tool_use_id"]) for e in log.entries]
        assert actions == [("deny", "c1")]
        assert log.entries[0]["action"]["reason"] == messages[0].content

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_decorator_without_functools_wraps_is_refused_and_recorded_as_one_deny(
        self, example, gate, log, route
    ):
        """The review's audited reproduction: the record bound ``timeout=5`` while the
        function ran its own ``timeout=999999``."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        ran, seen = [], []

        def with_timeout(fn):
            def wrapper(*args, timeout=30, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        @with_timeout
        def fetch(url: str, timeout: int = 999999) -> str:
            ran.append((url, timeout))
            return "fetched"

        class Args(BaseModel):
            url: str
            timeout: int = 30

        tool = StructuredTool.from_function(
            func=fetch, name="fetch", description="d", args_schema=Args
        )
        calls = [self.call("fetch", {"url": "u", "timeout": 5}, "c1")]
        messages = self.audited(example, gate, log, route, tool, calls, self.budget(seen))
        assert seen == [] and ran == []
        assert "takes `*args` or `**kwargs` and does not name `url`" in messages[0].content
        actions = [(e["action"]["decision"], e["action"]["tool_use_id"]) for e in log.entries]
        assert actions == [("deny", "c1")]

    @pytest.mark.parametrize("route", ["sync", "async"])
    def test_a_refused_call_answered_by_another_tool_is_withheld_and_recorded_once_more(
        self, example, gate, log, route
    ):
        """The policy refused the call (one entry). A step after the gate then ran the same
        statement on another tool and returned its success: that answer is withheld, and
        the withheld answer is one more entry, a deny with the call's id."""
        from langchain_core.tools import tool

        ran, seen = [], []

        @tool
        def run_sql_replica(sql: str) -> str:
            """Run SQL on the replica."""
            ran.append(sql)
            return "replica ran " + sql

        def no_delete(name, args):
            seen.append(name)
            return gate.sql_scope_policy(name, args)

        gates = example.recusal_audited_gates(no_delete, log)

        def after(request, execute):
            def fallback(judged):
                first = execute(judged)
                if first.status != "error":
                    return first
                return execute(judged.override(tool=run_sql_replica))

            return gates["wrap_tool_call"](request, fallback)

        async def aafter(request, execute):
            async def fallback(judged):
                first = await execute(judged)
                if first.status != "error":
                    return first
                return await execute(judged.override(tool=run_sql_replica))

            return await gates["awrap_tool_call"](request, fallback)

        node = gate.ToolNode(
            [gate.run_sql, run_sql_replica], wrap_tool_call=after, awrap_tool_call=aafter
        )
        calls = [self.call("run_sql", {"sql": "DELETE FROM audit_log"}, "c1")]
        gate.EXECUTED.clear()
        if route == "async":
            messages = asyncio.run(gate.run_in_graph_async(node, calls))
        else:
            messages = gate.run_in_graph(node, calls)
        assert seen == ["run_sql"] and gate.EXECUTED == []
        assert ran == ["DELETE FROM audit_log"], "the other tool ran; the record cannot undo it"
        assert messages[0].status == "error" and "replica ran" not in messages[0].content
        actions = [
            (e["action"]["tool"], e["action"]["decision"], e["action"]["tool_use_id"])
            for e in log.entries
        ]
        assert actions == [("run_sql", "deny", "c1"), ("run_sql", "deny", "c1")]
        judged, withheld = (e["action"]["reason"] for e in log.entries)
        assert "did not allow the call" not in judged
        assert "did not allow the call" in withheld and withheld == messages[0].content
        assert "run_sql_replica" not in json.dumps(log.entries)

    # --- negative controls --------------------------------------------------------------

    def test_negative_control_plain_gate_writes_no_entries(self, gate, log, log_path):
        messages, executed = gate.run_calls(self.THREE)
        assert [m.status for m in messages] == ["success", "error", "success"]
        assert log.entries == []
        assert not log_path.exists()

    def test_negative_control_ungated_node_records_nothing_and_runs_the_delete(self, gate, log):
        messages, executed = gate.run_calls(self.THREE, gated=False)
        assert ("run_sql", {"sql": "DELETE FROM audit_log"}) in executed
        assert log.entries == []

    # --- the script itself --------------------------------------------------------------

    def test_main_runs_end_to_end_and_shows_the_verifier_refusing(self, example):
        env = dict(os.environ)
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=False, env=env
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "c2 run_sql  error" in out
        assert f"c2 {SURFACE} run_sql  deny" in out
        assert "verify intact:   exit 0" in out
        assert "verify tampered: exit 1" in out
        assert "DELETE" not in out.split("verify intact", 1)[1]

    def test_main_leaves_no_log_behind(self, example, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setenv("TEMP", str(tmp_path))
        monkeypatch.setenv("TMP", str(tmp_path))
        before = set(tmp_path.iterdir())
        example.main()
        assert set(tmp_path.iterdir()) == before
