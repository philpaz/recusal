"""Regression tests for the offline LangChain create_agent audit-trail example.

Every call a ``create_agent`` agent proposes through the audited middleware, sync or async,
lands on a hash-chained log exactly once, labeled ``langchain.create_agent`` and keyed by
LangChain's own tool call id, and the audited middleware decides exactly as the plain one
does. One message's calls run in parallel, so entries are matched by id, never by position.
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
EXAMPLE = ROOT / "examples" / "langchain_agent_audit.py"
SURFACE = "langchain.create_agent"
PINS = ("langgraph==1.2.12", "langchain-core==1.6.5", "langchain==1.4.3")

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"

ROUTES = pytest.mark.parametrize("is_async", [False, True], ids=["invoke", "ainvoke"])


def test_python_version_refusal_is_clear_on_every_python():
    code = (
        "import runpy, sys; "
        "sys.version_info = (3, 9, 0, 'final', 0); "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "langchain_agent_audit.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_pinned_versions_are_explicit_in_the_example():
    text = EXAMPLE.read_text(encoding="utf-8")
    for pin in PINS:
        assert pin in text


def test_both_ci_lanes_run_this_file_with_the_langchain_example():
    # no new jobs: the two LangGraph and LangChain jobs and the weekly lane run it
    for workflow in ("ci.yml", "compat.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert text.count("tests/test_langchain_agent_audit.py") == 2, workflow


def test_the_works_with_table_marks_the_audit_trail_tested():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    assert (
        "| LangChain `create_agent` audit trail | Tested | [`examples/langchain_agent_audit.py`]"
        in table
    )


def _sha(args):
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _call(name, args, ident="c1"):
    return {"name": name, "args": args, "id": ident, "type": "tool_call"}


THREE = [
    _call("echo", {"text": "hello"}, "c1"),
    _call("run_sql", {"sql": DELETE}, "c2"),
    _call("run_sql", {"sql": SELECT}, "c3"),
]


def _corpus():
    def raising(tool_name, tool_input):
        raise RuntimeError("policy exploded")

    def silent(tool_name, tool_input):
        return None  # no objection: PASS

    def ambiguous(tool_name, tool_input):
        return [{"check": "vague"}]  # neither status nor passed: strict fails closed

    def error_level(tool_name, tool_input):
        from recusal import Finding

        return [Finding.fail("hard", severity="ERROR", message="retry this")]

    return {
        "sql-scope-three": (None, THREE),
        "sql-scope-unicode": (None, [_call("echo", {"text": "héllo ☃"})]),
        "raising": (raising, THREE),
        "silent": (silent, THREE),
        "ambiguous": (ambiguous, THREE),
        "error-level": (error_level, THREE),
    }


@pytest.mark.skipif(sys.version_info < (3, 10), reason="LangChain requires Python 3.10+")
class TestLangChainAgentAudit:
    @pytest.fixture
    def example(self):
        langchain = pytest.importorskip("langchain")
        if int(langchain.__version__.split(".", 1)[0]) < 1:
            pytest.skip("create_agent needs LangChain 1.x")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location(
            "recusal_langchain_agent_audit_example", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def gate(self, example):
        """The plain gate example, the SAME module object the audited one imports from."""
        return importlib.import_module("examples.langchain_agent_gate")

    @pytest.fixture
    def log_path(self, tmp_path):
        return tmp_path / "langchain-agent-audit.jsonl"

    @pytest.fixture
    def log(self, example, log_path):
        return example.AuditLog(path=str(log_path))

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
        """What the agent and the tools saw, order-independent for parallel tool calls."""
        seen = sorted((m.tool_call_id, m.name, m.status, str(m.content)) for m in messages)
        return seen, sorted((name, json.dumps(args, sort_keys=True)) for name, args in executed)

    @staticmethod
    def audited(example, calls, log, is_async, **kwargs):
        if is_async:
            return asyncio.run(example.arun_calls(calls, log, **kwargs))
        return example.run_calls(calls, log, **kwargs)

    @staticmethod
    def plain(gate, calls, is_async, **kwargs):
        if is_async:
            return asyncio.run(gate.arun_calls(calls, **kwargs))
        return gate.run_calls(calls, **kwargs)

    # --- every call recorded once, allowed and refused alike -------------------------------

    @ROUTES
    def test_every_call_is_recorded_once_with_surface_and_id(self, example, log, is_async):
        messages, executed = self.audited(example, THREE, log, is_async, actor="demo-agent")
        assert {m.tool_call_id: m.status for m in messages} == {
            "c1": "success",
            "c2": "error",
            "c3": "success",
        }
        assert sorted(name for name, _ in executed) == ["echo", "run_sql"]
        entries = self.by_id(log)
        assert set(entries) == {"c1", "c2", "c3"}
        for entry in entries.values():
            # the create_agent label, not the ToolNode one the shared hook defaults to
            assert entry["action"]["surface"] == SURFACE
            assert entry["actor"] == "demo-agent"
        assert {k: v["action"]["decision"] for k, v in entries.items()} == {
            "c1": "defer",
            "c2": "deny",
            "c3": "defer",
        }
        assert entries["c2"]["decision"] == "FAIL"
        assert "top-level WHERE" in entries["c2"]["reasons"]
        refused = next(m for m in messages if m.tool_call_id == "c2")
        assert entries["c2"]["action"]["reason"] == str(refused.content)

    def test_the_toolnode_hook_keeps_its_own_label_by_default(self, example):
        audit = importlib.import_module("examples.langgraph_audit")
        assert audit.SURFACE == "langgraph.toolnode" != SURFACE

    @ROUTES
    @pytest.mark.parametrize("case", sorted(_corpus()))
    def test_decisions_match_the_plain_middleware(self, example, gate, log, case, is_async):
        policy, calls = _corpus()[case]
        kwargs = {} if policy is None else {"policy": policy}
        plain = self.observed(*self.plain(gate, calls, is_async, **kwargs))
        audited = self.observed(*self.audited(example, calls, log, is_async, **kwargs))
        assert audited == plain
        assert len(self.by_id(log)) == len(calls)

    @ROUTES
    def test_a_refused_call_does_not_run(self, example, log, is_async):
        messages, executed = self.audited(
            example, [_call("run_sql", {"sql": DELETE})], log, is_async
        )
        assert executed == []
        assert messages[0].status == "error" and "Recusal refused" in str(messages[0].content)
        assert self.by_id(log)["c1"]["action"]["decision"] == "deny"

    # --- the record is part of the control ---------------------------------------------

    @ROUTES
    def test_policy_that_raises_is_denied_and_the_denial_is_recorded_without_its_text(
        self, example, log, log_path, is_async
    ):
        def broken(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        messages, executed = self.audited(
            example, [_call("echo", {"text": "x"})], log, is_async, policy=broken
        )
        assert executed == []
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert [f["message"] for f in entry["failures"]] == ["RECUSAL_POLICY_ERROR (RuntimeError)"]
        assert "policy exploded" not in str(messages[0].content)
        assert "policy exploded" not in log_path.read_text(encoding="utf-8")

    @ROUTES
    def test_unwritable_log_refuses_every_call_and_no_tool_runs(
        self, example, log, log_path, is_async
    ):
        self.audited(example, [_call("echo", {"text": "warm-up"}, "w")], log, is_async)
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
            messages, executed = self.audited(example, THREE, log, is_async)
        finally:
            log_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        assert executed == []
        assert [m.status for m in messages] == ["error", "error", "error"]
        for message in messages:
            assert "audit log unavailable" in str(message.content)
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1

    @ROUTES
    def test_failing_sink_refuses_the_call_and_no_tool_runs(self, example, log_path, is_async):
        class Offline:
            def write(self, entry):
                raise OSError("anchor store unreachable")

        log = example.AuditLog(path=str(log_path), sinks=[Offline()])
        messages, executed = self.audited(example, THREE, log, is_async)
        assert executed == []
        assert all("RECUSAL_AUDIT_UNAVAILABLE" in str(m.content) for m in messages)
        assert not any("anchor store unreachable" in str(m.content) for m in messages)

    # --- the record binds what executed ---------------------------------------------------

    @ROUTES
    def test_no_tool_argument_text_appears_in_the_log(self, example, log, log_path, is_async):
        secret = "DELETE FROM customers WHERE ssn = 'MARKER-7f3e9'"
        calls = [
            _call("run_sql", {"sql": secret}, "s1"),
            _call("echo", {"text": "MARKER-echo-text"}, "s2"),
        ]
        self.audited(example, calls, log, is_async)
        text = log_path.read_text(encoding="utf-8")
        assert "MARKER" not in text and "customers" not in text
        entries = self.by_id(log)
        for call in calls:
            assert entries[call["id"]]["action"]["input_sha256"] == _sha(call["args"])

    @ROUTES
    def test_the_record_binds_what_executed_whatever_the_policy_did_to_its_copy(
        self, example, log, is_async
    ):
        def mutating(tool_name, tool_input):
            tool_input["text"] = "REWRITTEN BY THE POLICY"
            return []

        _, executed = self.audited(
            example, [_call("echo", {"text": "as proposed"})], log, is_async, policy=mutating
        )
        assert executed == [("echo", {"text": "as proposed"})]
        assert self.by_id(log)["c1"]["action"]["input_sha256"] == _sha(executed[0][1])

    def test_a_rewrite_listed_after_the_gate_is_recorded_as_rewritten(self, example, gate, log):
        """The gate judges, and records, inside the tool's own parse step, so a middleware
        listed after it that rewrites the arguments is recorded at the value it produced."""
        from langchain.agents.middleware import wrap_tool_call
        from langchain_core.messages import HumanMessage

        @wrap_tool_call
        def rewrite(request, handler):
            call = dict(request.tool_call)
            call["args"] = {"text": "rewritten after the gate"}
            return handler(request.override(tool_call=call))

        middleware = [example.audited_gate_middleware(lambda *_: [], log), rewrite]
        agent = gate.scripted_agent([_call("echo", {"text": "as proposed"})], middleware)
        gate.EXECUTED.clear()
        agent.invoke({"messages": [HumanMessage(content="go")]})
        assert gate.EXECUTED == [("echo", {"text": "rewritten after the gate"})]
        entry = self.by_id(log)["c1"]
        assert entry["action"]["input_sha256"] == _sha({"text": "rewritten after the gate"})

    def test_a_middleware_that_answers_in_the_tools_place_leaves_no_entry(self, example, gate, log):
        """As the example says: listed after the gate, a middleware that answers without
        executing removes the judging copy, so the call is neither judged nor recorded.
        Nothing ran either."""
        from langchain.agents.middleware import wrap_tool_call
        from langchain_core.messages import HumanMessage, ToolMessage

        @wrap_tool_call
        def answer(request, handler):
            return ToolMessage(
                content="answered in the tool's place",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )

        middleware = [example.audited_gate_middleware(gate.sql_scope_policy, log), answer]
        agent = gate.scripted_agent([_call("run_sql", {"sql": DELETE})], middleware)
        gate.EXECUTED.clear()
        result = agent.invoke({"messages": [HumanMessage(content="go")]})
        assert gate.EXECUTED == []
        assert str(gate.tool_messages(result)[0].content) == "answered in the tool's place"
        assert log.entries == []

    # --- control identity, verification, tamper detection ---------------------------------

    def test_caller_control_is_recorded_but_cannot_forge_the_version(self, example, log):
        control = {"policy_id": "sql-scope", "policy_version": "3", "recusal_version": "9.9.9"}
        example.run_calls([_call("echo", {"text": "x"})], log, control=control)
        recorded = self.by_id(log)["c1"]["action"]["control"]
        assert recorded["policy_id"] == "sql-scope"
        assert recorded["recusal_version"] != "9.9.9"

    def test_log_verifies_intact_and_tampering_breaks_it(self, example, log, log_path):
        example.run_calls(THREE, log, actor="demo-agent")
        code, text = example.verify_with_cli(str(log_path))
        assert code == 0, text
        assert "intact - 3 entries" in text
        example.tamper(str(log_path), "c2")
        code, text = example.verify_with_cli(str(log_path))
        assert code == 1 and "BROKEN" in text

    # --- negative controls --------------------------------------------------------------

    @ROUTES
    def test_negative_control_plain_middleware_writes_no_entries(
        self, gate, log, log_path, is_async
    ):
        messages, _ = self.plain(gate, THREE, is_async)
        assert sorted(m.status for m in messages) == ["error", "success", "success"]
        assert log.entries == []
        assert not log_path.exists()

    def test_negative_control_ungated_agent_records_nothing_and_runs_the_delete(self, gate, log):
        _, executed = gate.run_calls(THREE, gated=False)
        assert ("run_sql", {"sql": DELETE}) in executed
        assert log.entries == []

    # --- the script itself --------------------------------------------------------------

    def test_main_runs_end_to_end_and_shows_the_verifier_refusing(self, example):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)],
            capture_output=True,
            text=True,
            check=False,
            env=dict(os.environ),
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "c2 run_sql  error" in out
        assert f"c2 {SURFACE} run_sql  deny" in out
        assert "verify intact:   exit 0" in out
        assert "verify tampered: exit 1" in out
        assert "DELETE" not in out

    def test_main_leaves_no_log_behind(self, example, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setenv("TEMP", str(tmp_path))
        monkeypatch.setenv("TMP", str(tmp_path))
        before = set(tmp_path.iterdir())
        example.main()
        assert set(tmp_path.iterdir()) == before
