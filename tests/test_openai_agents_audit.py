"""Regression tests for the offline OpenAI Agents SDK audit-trail example.

Every function tool call an audited guardrail sees, through ``Runner.run`` or
``Runner.run_streamed``, lands on a hash-chained log exactly once, labeled
``openai_agents.tool_input_guardrail`` and keyed by the SDK's own tool call id, and the
audited gate decides exactly as the plain one does. One model turn's calls can complete in
any order, so entries are matched by id, never by position.
"""

import asyncio
import copy
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
EXAMPLE = ROOT / "examples" / "openai_agents_audit.py"
SURFACE = "openai_agents.tool_input_guardrail"
PINS = ("openai-agents==0.24.0", "openai==3.49.0")

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"

ROUTES = pytest.mark.parametrize("streamed", [False, True], ids=["run", "streamed"])


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
    assert result.returncode == 2
    assert "openai_agents_audit.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_pinned_versions_are_explicit_in_the_example():
    text = EXAMPLE.read_text(encoding="utf-8")
    for pin in PINS:
        assert pin in text


def test_both_openai_lanes_run_this_file():
    # no new jobs: the two OpenAI jobs and the weekly lane run it
    for workflow in ("ci.yml", "compat.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert text.count("tests/test_openai_agents_audit.py") == 2, workflow


def test_the_works_with_table_marks_the_audit_trail_tested():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    assert (
        "| OpenAI Agents SDK audit trail, function tools | Tested | "
        "[`examples/openai_agents_audit.py`]" in table
    )


def _sha(args):
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


THREE = [
    ("echo", {"text": "hello"}, "c1"),
    ("run_sql", {"sql": DELETE}, "c2"),
    ("run_sql", {"sql": SELECT}, "c3"),
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
    return {
        "sql-scope-three": (None, THREE),
        "sql-scope-long": (None, [("run_sql", {"sql": long_sql}, "c1")]),
        "sql-scope-unicode": (None, [("echo", {"text": "héllo ☃"}, "c1")]),
        "raising": (raising, THREE),
        "silent": (silent, THREE),
        "ambiguous": (ambiguous, THREE),
        "warning-only": (warning_only, THREE),
        "error-level": (error_level, THREE),
    }


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestOpenAIAgentsAudit:
    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location(
            "recusal_openai_agents_audit_example", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def gate(self, example):
        """The plain gate example, the SAME module object the audited one imports from."""
        return importlib.import_module("examples.openai_agents_gate")

    @pytest.fixture
    def log_path(self, tmp_path):
        return tmp_path / "openai-agents-audit.jsonl"

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
    def observed(outputs, executed):
        """What the model and the tools saw, order-independent: the order one turn's tools
        ran in is not part of the decision."""
        return outputs, sorted((name, json.dumps(args, sort_keys=True)) for name, args in executed)

    @staticmethod
    def charge_tool(gate, ran):
        @gate.function_tool(strict_mode=False)
        def charge(amount: int = 100) -> str:
            """Record a charge; contacts nothing."""
            ran.append((amount, type(amount).__name__))
            return "recorded"

        return charge

    # --- every call recorded once, allowed and refused alike -------------------------------

    @ROUTES
    def test_every_call_is_recorded_once_with_surface_and_id(self, example, log, streamed):
        outputs, executed = example.run_calls(THREE, log, actor="demo-agent", streamed=streamed)
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
        # the reason the model read is the reason on the record
        assert entries["c2"]["action"]["reason"] == outputs["c2"]

    @ROUTES
    def test_entries_form_one_chain_whatever_the_completion_order(self, example, log, streamed):
        many = [("echo", {"text": f"t{i}"}, f"id{i}") for i in range(12)]
        many.insert(5, ("run_sql", {"sql": "DELETE FROM t"}, "refused"))
        example.run_calls(many, log, streamed=streamed)
        assert len(self.by_id(log)) == 13
        assert [e["seq"] for e in log.entries] == list(range(13))
        for previous, entry in zip(log.entries, log.entries[1:]):
            assert entry["prev_hash"] == previous["hash"]
        from recusal.audit import verify

        ok, problems = verify(log.entries)
        assert ok, problems

    # --- decision equivalence with the plain gate -----------------------------------------

    @ROUTES
    @pytest.mark.parametrize("case", sorted(_corpus()))
    def test_decisions_match_the_plain_gate(self, example, gate, log, case, streamed):
        policy, calls = _corpus()[case]
        kwargs = {} if policy is None else {"policy": policy}
        plain = self.observed(*gate.run_calls(calls, streamed=streamed, **kwargs))
        audited = self.observed(*example.run_calls(calls, log, streamed=streamed, **kwargs))
        assert audited == plain
        assert len(self.by_id(log)) == len(calls)

    def test_the_audited_agent_is_gated_and_the_original_is_not(self, example, gate, log):
        agent = gate.Agent(name="a", model=gate.scripted_model([]), tools=[gate.echo])
        audited = example.audited_agent(agent, gate.sql_scope_policy, log)
        assert gate.is_gated(audited) and not gate.is_gated(agent)

    @ROUTES
    def test_a_refused_call_does_not_run(self, example, log, streamed):
        outputs, executed = example.run_calls(
            [("run_sql", {"sql": DELETE}, "c1")], log, streamed=streamed
        )
        assert executed == []
        assert outputs["c1"].startswith("Recusal refused `run_sql` [FAIL]")
        assert self.by_id(log)["c1"]["action"]["decision"] == "deny"

    # --- the record is part of the control ---------------------------------------------

    @ROUTES
    def test_policy_that_raises_is_denied_nothing_runs_and_the_denial_is_recorded(
        self, example, gate, log, log_path, streamed
    ):
        def broken(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        calls = [("echo", {"text": "x"}, "c1")]
        outputs, executed = example.run_calls(calls, log, policy=broken, streamed=streamed)
        assert executed == []
        plain, _ = gate.run_calls(calls, policy=broken, streamed=streamed)
        assert outputs == plain
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["decision"] == "FAIL"
        assert [f["message"] for f in entry["failures"]] == ["RECUSAL_POLICY_ERROR (RuntimeError)"]
        assert "policy exploded" not in outputs["c1"]
        assert "policy exploded" not in log_path.read_text(encoding="utf-8")

    @ROUTES
    def test_unwritable_log_refuses_every_call_and_no_tool_runs(
        self, example, log, log_path, streamed
    ):
        example.run_calls([("echo", {"text": "warm-up"}, "w")], log, streamed=streamed)
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
            outputs, executed = example.run_calls(THREE, log, streamed=streamed)
        finally:
            log_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        assert executed == []
        assert set(outputs) == {"c1", "c2", "c3"}
        for content in outputs.values():
            assert "audit log unavailable" in content
            assert "the record is part of the control" in content
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1

    @ROUTES
    def test_failing_sink_refuses_the_call_and_no_tool_runs(self, example, log_path, streamed):
        class Offline:
            def write(self, entry):
                raise OSError("anchor store unreachable")

        log = example.AuditLog(path=str(log_path), sinks=[Offline()])
        outputs, executed = example.run_calls(THREE, log, streamed=streamed)
        assert executed == []
        assert all("RECUSAL_AUDIT_UNAVAILABLE" in content for content in outputs.values())
        assert not any("anchor store unreachable" in content for content in outputs.values())

    def test_a_recording_that_raises_refuses_the_call(self, example, gate):
        def broken(ctx):
            raise RuntimeError("RECORDING_SECRET_SENTINEL")

        model = gate.scripted_model([("echo", {"text": "x"}, "c1")])
        agent = gate.gated_agent(
            gate.Agent(name="a", model=model, tools=[gate.echo]), gate.sql_scope_policy, broken
        )
        gate.EXECUTED.clear()
        asyncio.run(gate.run_agent(agent))
        assert gate.EXECUTED == []
        assert gate.tool_outputs(model)["c1"] == (
            "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        )

    # --- the record binds what executed ---------------------------------------------------

    @ROUTES
    def test_no_tool_argument_text_appears_in_the_log(self, example, log, log_path, streamed):
        secret = "DELETE FROM customers WHERE ssn = 'MARKER-7f3e9'"
        calls = [
            ("run_sql", {"sql": secret}, "s1"),
            ("echo", {"text": "MARKER-echo-text"}, "s2"),
        ]
        example.run_calls(calls, log, streamed=streamed)
        text = log_path.read_text(encoding="utf-8")
        assert "MARKER" not in text
        assert "customers" not in text
        entries = self.by_id(log)
        for _, args, call_id in calls:
            assert entries[call_id]["action"]["input_sha256"] == _sha(args)
            assert entries[call_id]["action"]["control"]["recusal_version"]

    @pytest.mark.parametrize(
        "proposed, received",
        [({}, 100), ({"amount": "5"}, 5), ({"amount": 7}, 7)],
        ids=["default", "coerced", "as-given"],
    )
    def test_defaults_and_coercion_are_recorded_at_the_value_the_function_receives(
        self, example, gate, log, proposed, received
    ):
        ran = []
        outputs, _ = example.run_calls(
            [("charge", proposed, "c1")],
            log,
            policy=lambda *_: [],
            tools=[self.charge_tool(gate, ran)],
        )
        assert ran == [(received, "int")]
        assert self.by_id(log)["c1"]["action"]["input_sha256"] == _sha({"amount": received})

    @ROUTES
    def test_the_record_binds_what_executed_whatever_the_policy_did_to_its_copy(
        self, example, log, streamed
    ):
        def mutating(tool_name, tool_input):
            tool_input["text"] = "REWRITTEN BY THE POLICY"
            tool_input["extra"] = [1, 2, 3]
            return []

        outputs, executed = example.run_calls(
            [("echo", {"text": "as proposed"}, "c1")], log, policy=mutating, streamed=streamed
        )
        assert executed == [("echo", {"text": "as proposed"})]
        assert self.by_id(log)["c1"]["action"]["input_sha256"] == _sha(executed[0][1])

    # --- refused before judgment: still one entry, bound to the proposal --------------------

    @pytest.mark.parametrize(
        "arguments, detail, proposal",
        [
            ('{"sql": "MARKER', "arguments are not valid JSON", {"input": '{"sql": "MARKER'}),
            ('["MARKER"]', "arguments are not a JSON object", {"input": ["MARKER"]}),
            (
                '{"sql": {"nested": "MARKER"}}',
                "the arguments the tool would execute cannot be judged",
                {"sql": {"nested": "MARKER"}},
            ),
        ],
        ids=["bad-json", "not-object", "does-not-validate"],
    )
    def test_every_refusal_before_judgment_is_one_recorded_deny(
        self, example, log, log_path, arguments, detail, proposal
    ):
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        outputs, executed = example.run_calls([("run_sql", arguments, "c1")], log, policy=policy)
        assert executed == [] and seen == []
        reason = outputs["c1"]
        assert reason.startswith("Recusal refused `run_sql` [FAIL]") and detail in reason
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny" and entry["decision"] == "FAIL"
        assert entry["action"]["reason"] == reason
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["input_sha256"] == _sha(proposal)
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    def test_a_guardrail_bound_to_another_tool_is_one_recorded_deny(self, example, gate, log):
        holder = [gate.echo]
        guardrail = _adapter()._guardrail_for(gate.sql_scope_policy, holder, example.recording(log))
        tool = copy.copy(gate.run_sql)
        tool.tool_input_guardrails = [guardrail]
        model = gate.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
        gate.EXECUTED.clear()
        asyncio.run(gate.run_agent(gate.Agent(name="a", model=model, tools=[tool])))
        assert gate.EXECUTED == []
        reason = gate.tool_outputs(model)["c1"]
        assert "this guardrail is bound to 'echo'" in reason
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["reason"] == reason
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]

    # --- an agent tool with a parameters model: the entry is bound to what the sub-agent gets --

    def ask(self, example, gate, log, arguments):
        """An audited parent calls ``ask_sub(task, confirm=False, rows=1)`` with
        ``arguments``: ``(the call's output, whether the sub-agent was started)``."""
        from agents.testing import ScriptedModel, assistant_message
        from pydantic import BaseModel

        class Ask(BaseModel):
            task: str
            confirm: bool = False
            rows: int = 1

        sub_model = ScriptedModel([[assistant_message("sub done")]])
        sub = gate.gated_agent(
            gate.Agent(name="sub", model=sub_model, tools=[gate.run_sql]), lambda *_: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=Ask)
        parent_model = gate.scripted_model([("ask_sub", arguments, "p1")])
        parent = example.audited_agent(
            gate.Agent(name="parent", model=parent_model, tools=[tool]), lambda *_: [], log
        )
        asyncio.run(gate.run_agent(parent))
        return gate.tool_outputs(parent_model)["p1"], sub_model.first_call is not None

    def test_an_agent_tool_call_is_recorded_at_the_input_the_sub_agent_is_given(
        self, example, gate, log
    ):
        output, started = self.ask(example, gate, log, {"task": "list", "rows": "3", "x": 1})
        assert output == "sub done" and started
        entry = self.by_id(log)["p1"]
        assert entry["action"]["decision"] != "deny"
        assert entry["action"]["input_sha256"] == _sha(
            {"task": "list", "confirm": False, "rows": 3}
        )

    def test_agent_tool_arguments_that_do_not_validate_are_one_recorded_deny(
        self, example, gate, log, log_path
    ):
        proposed = {"task": "MARKER", "rows": "many"}
        output, started = self.ask(example, gate, log, proposed)
        assert not started
        assert output == (
            "Recusal refused `ask_sub` [FAIL]: the input the sub-agent would be given cannot "
            "be judged (the arguments do not validate against its parameters)"
        )
        assert len(log.entries) == 1
        entry = self.by_id(log)["p1"]
        assert entry["action"]["decision"] == "deny" and entry["action"]["reason"] == output
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["input_sha256"] == _sha(proposed)
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    def test_an_agent_tool_refused_for_its_parameters_type_on_a_call_is_one_recorded_deny(
        self, example, gate, log, log_path
    ):
        """An agent tool whose ``parameters`` type the gate does not read is refused when
        it is gated; should its guardrail be attached another way, the refusal on the
        call is on the log like every other."""
        from agents.testing import ScriptedModel, assistant_message
        from pydantic import BaseModel, field_validator

        class Ask(BaseModel):
            task: str

            @field_validator("task")
            @classmethod
            def upper(cls, value):
                return value.upper()

        sub_model = ScriptedModel([[assistant_message("sub done")]])
        sub = gate.gated_agent(
            gate.Agent(name="sub", model=sub_model, tools=[gate.run_sql]), lambda *_: []
        )
        tool = sub.as_tool(tool_name="ask_sub", tool_description="Ask.", parameters=Ask)
        holder = [tool]
        seen: list = []
        guardrail = _adapter()._guardrail_for(
            lambda name, args: seen.append(name) or [], holder, example.recording(log)
        )
        tool.tool_input_guardrails = [guardrail]
        parent_model = gate.scripted_model([("ask_sub", {"task": "MARKER"}, "p1")])
        asyncio.run(gate.run_agent(gate.Agent(name="p", model=parent_model, tools=[tool])))
        reason = gate.tool_outputs(parent_model)["p1"]
        assert reason.startswith(
            "Recusal refused `ask_sub` [FAIL]: its `parameters` model has a field outside the "
            "data-only set: `task` (pydantic core type `function-after`)"
        )
        assert sub_model.first_call is None and seen == []
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny" and entry["action"]["reason"] == reason
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["input_sha256"] == _sha({"task": "MARKER"})
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    # --- approvals, and what the guardrail never sees, as the example says ------------------

    @pytest.mark.parametrize("approve", [True, False], ids=["approved", "rejected"])
    def test_a_call_needing_approval_is_recorded_when_it_is_about_to_run(
        self, example, gate, log, approve
    ):
        tool = copy.copy(gate.run_sql)
        tool.needs_approval = True
        model = gate.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
        agent = example.audited_agent(
            gate.Agent(name="a", model=model, tools=[tool]), gate.sql_scope_policy, log
        )
        config = gate.RunConfig(tracing_disabled=True)

        async def decide_and_resume():
            result = await gate.Runner.run(agent, "go", run_config=config)
            assert len(result.interruptions) == 1
            # nothing ran and nothing is recorded while the call waits for a human
            assert gate.EXECUTED == [] and log.entries == []
            state = result.to_state()
            if approve:
                state.approve(result.interruptions[0])
            else:
                state.reject(result.interruptions[0])
            await gate.Runner.run(agent, state, run_config=config)

        gate.EXECUTED.clear()
        asyncio.run(decide_and_resume())
        if approve:
            assert gate.EXECUTED == [("run_sql", {"sql": SELECT})]
            assert self.by_id(log)["c1"]["action"]["decision"] == "defer"
        else:
            assert gate.EXECUTED == []
            assert log.entries == []

    def test_an_unknown_tool_stops_the_run_and_writes_no_entry(self, example, log):
        from agents.exceptions import ModelBehaviorError

        with pytest.raises(ModelBehaviorError):
            example.run_calls([("nope", {"a": 1}, "c1")], log)
        assert log.entries == []

    # --- control identity, verification, tamper detection ---------------------------------

    def test_caller_control_is_recorded_but_cannot_forge_the_version(self, example, log):
        control = {"policy_id": "sql-scope", "policy_version": "3", "recusal_version": "9.9.9"}
        example.run_calls([("echo", {"text": "x"}, "c1")], log, control=control)
        recorded = self.by_id(log)["c1"]["action"]["control"]
        assert recorded["policy_id"] == "sql-scope"
        assert recorded["policy_version"] == "3"
        assert recorded["recusal_version"] != "9.9.9"

    def test_log_verifies_intact_and_editing_any_entry_breaks_it(self, example, log, log_path):
        example.run_calls(THREE, log, actor="demo-agent")
        code, text = example.verify_with_cli(str(log_path))
        assert code == 0, text
        assert "intact - 3 entries" in text
        lines = log_path.read_text(encoding="utf-8").splitlines()
        for index in range(len(lines)):
            edited = list(lines)
            entry = json.loads(edited[index])
            flipped = "allow" if entry["action"]["decision"] != "allow" else "deny"
            entry["action"]["decision"] = flipped
            edited[index] = json.dumps(entry, sort_keys=True)
            tampered = log_path.with_name(f"tampered-{index}.jsonl")
            tampered.write_text("\n".join(edited) + "\n", encoding="utf-8")
            code, text = example.verify_with_cli(str(tampered))
            assert code == 1, (index, text)
            assert "BROKEN" in text

    def test_the_example_tamper_helper_targets_the_refused_call(self, example, log, log_path):
        example.run_calls(THREE, log)
        example.tamper(str(log_path), "c2")
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        edited = [e for e in entries if e["action"]["tool_use_id"] == "c2"]
        assert edited and edited[0]["action"]["decision"] == "allow"
        code, _ = example.verify_with_cli(str(log_path))
        assert code == 1

    # --- negative controls --------------------------------------------------------------

    @ROUTES
    def test_negative_control_plain_gate_writes_no_entries(self, gate, log, log_path, streamed):
        outputs, _ = gate.run_calls(THREE, streamed=streamed)
        assert outputs["c2"].startswith("Recusal refused")
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
        assert "c2 run_sql  refused" in out
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


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestOneDecisionPath:
    """The 2026-11-21 review. "The audited forms are the plain ones with a recording passed
    in" was false for a non-finite number: the plain gate let ``NaN`` reach a policy that
    could not compare it, and the audited gate refused it because the record could not
    fingerprint it. Both now refuse it before the policy runs, for the same reason. And a
    call waiting for approval is on the log before a person decides when the SDK is told
    to run tool input guardrails before approval."""

    example = TestOpenAIAgentsAudit.example
    gate = TestOpenAIAgentsAudit.gate
    log_path = TestOpenAIAgentsAudit.log_path
    log = TestOpenAIAgentsAudit.log

    @staticmethod
    def pay_tool(gate, ran):
        @gate.function_tool(strict_mode=False)
        def pay(amount: float) -> str:
            """Record a payment; contacts nothing."""
            ran.append(amount)
            return "paid"

        return pay

    @staticmethod
    def ceiling(name, args):
        """``NaN > 1000`` is False: on its own this policy lets a NaN amount through."""
        from recusal import Finding

        if args.get("amount", 0) > 1000:
            return [Finding.fail("limit", severity="CRITICAL", message="over limit")]
        return []

    @ROUTES
    @pytest.mark.parametrize(
        "arguments, detail, bound",
        [
            (
                '{"amount": NaN}',
                "arguments hold a non-finite number, which is not JSON data",
                {"input": '{"amount": NaN}'},
            ),
            (
                '{"amount": -Infinity}',
                "arguments hold a non-finite number, which is not JSON data",
                {"input": '{"amount": -Infinity}'},
            ),
            (
                '{"amount": 1e999}',
                "arguments hold a non-finite number, which is not JSON data",
                {"input": '{"amount": 1e999}'},
            ),
            (
                '{"amount": "nan"}',
                "the arguments the tool would execute cannot be judged (non-finite numbers "
                "are not JSON data)",
                {"amount": "nan"},
            ),
        ],
        ids=["nan", "minus-infinity", "overflow", "string-read-as-nan"],
    )
    def test_a_non_finite_number_is_refused_the_same_way_with_and_without_a_log(
        self, example, gate, log, streamed, arguments, detail, bound
    ):
        ran: list = []
        seen: list = []

        def policy(name, args):
            seen.append(name)
            return self.ceiling(name, args)

        calls = [("pay", arguments, "c1")]
        plain, _ = gate.run_calls(
            calls, policy=policy, streamed=streamed, tools=[self.pay_tool(gate, ran)]
        )
        audited, _ = example.run_calls(
            calls, log, policy=policy, streamed=streamed, tools=[self.pay_tool(gate, ran)]
        )
        assert plain == {"c1": f"Recusal refused `pay`: {detail}"}
        assert audited == {"c1": f"Recusal refused `pay` [FAIL]: {detail}"}
        assert ran == [] and seen == [], "refused before the policy ran, and nothing ran"
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny" and entry["action"]["tool_use_id"] == "c1"
        assert entry["action"]["reason"] == audited["c1"]
        assert entry["action"]["input_sha256"] == _sha(bound)

    @staticmethod
    def note_tool(gate, ran):
        import typing

        @gate.function_tool(strict_mode=False)
        def note(payload: typing.Any) -> str:
            """Record a note; contacts nothing."""
            ran.append("ran")
            return "noted"

        return note

    @ROUTES
    @pytest.mark.parametrize("depth", [65, 300, 488, 600])
    def test_deeply_nested_arguments_are_refused_the_same_way_with_and_without_a_log(
        self, example, gate, log, streamed, depth
    ):
        """The second review, F3: at about 488 levels the plain gate ran the call and the
        audited one refused it (the record could not fingerprint it), and deeper the two
        refused for different stated causes. One fixed limit, applied by the gate before
        anything else reads the arguments, now gives one answer in both."""
        ran: list = []
        seen: list = []

        def policy(name, args):
            seen.append(name)
            return []

        arguments = '{"payload": ' + "[" * depth + "]" * depth + "}"
        calls = [("note", arguments, "c1")]
        if sys.version_info < (3, 11) and depth >= 488:
            # Python 3.10 runs out of stack in the SDK's own normalising of the arguments,
            # before any tool or hook runs: the run raises, and nothing ran or was judged.
            with pytest.raises(RecursionError):
                gate.run_calls(
                    calls, policy=policy, streamed=streamed, tools=[self.note_tool(gate, ran)]
                )
            assert ran == [] and seen == []
            return
        plain, _ = gate.run_calls(
            calls, policy=policy, streamed=streamed, tools=[self.note_tool(gate, ran)]
        )
        audited, _ = example.run_calls(
            calls, log, policy=policy, streamed=streamed, tools=[self.note_tool(gate, ran)]
        )
        assert plain == {"c1": "Recusal refused `note`: arguments are nested too deeply"}
        assert audited == {"c1": "Recusal refused `note` [FAIL]: arguments are nested too deeply"}
        assert ran == [] and seen == []
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny" and entry["action"]["tool_use_id"] == "c1"
        assert entry["action"]["reason"] == audited["c1"]
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["input_sha256"] == _sha({"input": arguments})

    def test_arguments_at_the_depth_limit_take_the_same_path_in_both(self, example, gate, log):
        ran: list = []
        arguments = '{"payload": ' + "[" * 64 + "]" * 64 + "}"
        calls = [("note", arguments, "c1")]
        plain, _ = gate.run_calls(calls, policy=lambda *_: [], tools=[self.note_tool(gate, ran)])
        audited, _ = example.run_calls(
            calls, log, policy=lambda *_: [], tools=[self.note_tool(gate, ran)]
        )
        assert plain == audited == {"c1": "noted"} and ran == ["ran", "ran"]
        (entry,) = log.entries
        assert entry["action"]["decision"] == "defer"
        assert entry["action"]["input_sha256"] == _sha(json.loads(arguments))

    def test_negative_control_a_finite_amount_takes_the_same_path_in_both(self, example, gate, log):
        ran: list = []
        calls = [("pay", '{"amount": 5}', "c1"), ("pay", '{"amount": 5000}', "c2")]
        plain, _ = gate.run_calls(calls, policy=self.ceiling, tools=[self.pay_tool(gate, ran)])
        audited, _ = example.run_calls(
            calls, log, policy=self.ceiling, tools=[self.pay_tool(gate, ran)]
        )
        assert plain == audited
        assert plain["c1"] == "paid" and "over limit" in plain["c2"]
        assert ran == [5.0, 5.0]

    @pytest.mark.parametrize("approve", [True, False], ids=["approved", "rejected"])
    def test_with_guardrails_before_approval_a_call_is_recorded_before_a_person_decides(
        self, example, gate, log, approve
    ):
        """``RunConfig(tool_execution=ToolExecutionConfig(pre_approval_tool_input_guardrails=
        True))`` makes the SDK run the guardrail before it asks a person as well as when
        the tool is about to run: a rejected call has one passing entry, an approved call
        has two under the same call id. Nothing runs without the second judgment."""
        from agents.run_config import ToolExecutionConfig

        tool = copy.copy(gate.run_sql)
        tool.needs_approval = True
        model = gate.scripted_model([("run_sql", {"sql": SELECT}, "c1")])
        agent = example.audited_agent(
            gate.Agent(name="a", model=model, tools=[tool]), gate.sql_scope_policy, log
        )
        config = gate.RunConfig(
            tracing_disabled=True,
            tool_execution=ToolExecutionConfig(pre_approval_tool_input_guardrails=True),
        )

        async def decide_and_resume():
            result = await gate.Runner.run(agent, "go", run_config=config)
            assert len(result.interruptions) == 1
            # judged, recorded and waiting: on the log before a person decided
            assert gate.EXECUTED == [] and len(log.entries) == 1
            state = result.to_state()
            if approve:
                state.approve(result.interruptions[0])
            else:
                state.reject(result.interruptions[0])
            await gate.Runner.run(agent, state, run_config=config)

        gate.EXECUTED.clear()
        asyncio.run(decide_and_resume())
        decisions = [(e["action"]["tool_use_id"], e["action"]["decision"]) for e in log.entries]
        if approve:
            assert gate.EXECUTED == [("run_sql", {"sql": SELECT})]
            assert decisions == [("c1", "defer"), ("c1", "defer")]
        else:
            assert gate.EXECUTED == []
            assert decisions == [("c1", "defer")]
