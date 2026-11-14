"""Regression tests for the offline Google ADK audit-trail example.

Every call an audited gate sees, through ``audited_agent`` (the before-tool callback) or
``audited_plugin`` (the runner plugin), lands on a hash-chained log exactly once, labeled
``google_adk.before_tool_callback`` and keyed by the ADK's own function call id, and the
audited gate decides exactly as the plain one does. One model turn's calls can complete in
any order, so entries are matched by id, never by position.
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
EXAMPLE = ROOT / "examples" / "google_adk_audit.py"
SURFACE = "google_adk.before_tool_callback"
PINS = ("google-adk==2.13.0", "google-genai==2.34.0")

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"

ROUTES = pytest.mark.parametrize("via_plugin", [False, True], ids=["callback", "plugin"])


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
    assert "google_adk_audit.py requires Python 3.11+" in result.stderr
    assert f"this is Python 3.{minor}." in result.stderr


def _adapter():
    """The package module the gate's code lives in. Every test that calls this has loaded
    the example first, which puts the in-tree adapter on the path, so this is the code
    under review and not an installed copy."""
    import importlib
    from pathlib import Path

    module = importlib.import_module("recusal_google_adk._gate")
    assert "adapters" in Path(module.__file__).parts, module.__file__
    return module


def test_pinned_versions_are_explicit_in_the_example():
    text = EXAMPLE.read_text(encoding="utf-8")
    for pin in PINS:
        assert pin in text


def test_both_ci_lanes_run_this_file_with_the_adk_example():
    # no new jobs: the two ADK jobs and the weekly lane run it
    for workflow in ("ci.yml", "compat.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert text.count("tests/test_google_adk_audit.py") == 2, workflow


def test_the_works_with_table_marks_the_audit_trail_tested():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    assert "| Google ADK audit trail | Tested | [`examples/google_adk_audit.py`]" in table


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
        "default-omitted": (None, [("charge", {}, "c1")]),
        "raising": (raising, THREE),
        "silent": (silent, THREE),
        "ambiguous": (ambiguous, THREE),
        "warning-only": (warning_only, THREE),
        "error-level": (error_level, THREE),
    }


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the Google ADK requires Python 3.11+")
class TestGoogleAdkAudit:
    @pytest.fixture
    def example(self):
        pytest.importorskip("google.adk", reason="google-adk is not installed")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location("recusal_google_adk_audit_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def gate(self, example):
        """The plain gate example, the SAME module object the audited one imports from."""
        return importlib.import_module("examples.google_adk_gate")

    @pytest.fixture
    def log_path(self, tmp_path):
        return tmp_path / "google-adk-audit.jsonl"

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
    def observed(responses, executed):
        """What the model and the tools saw, order-independent: the order one turn's tools
        ran in is not part of the decision."""
        return responses, sorted(
            (name, json.dumps(args, sort_keys=True)) for name, args in executed
        )

    @staticmethod
    def refused(response):
        return isinstance(response, dict) and "error" in response

    # --- every call recorded once, allowed and refused alike -------------------------------

    @ROUTES
    def test_every_call_is_recorded_once_with_surface_and_id(self, example, log, via_plugin):
        responses, executed = example.run_calls(
            THREE, log, actor="demo-agent", via_plugin=via_plugin
        )
        assert {k: self.refused(v) for k, v in responses.items()} == {
            "c1": False,
            "c2": True,
            "c3": False,
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
        # the reason the model read is the reason on the record
        assert entries["c2"]["action"]["reason"] == responses["c2"]["error"]

    @ROUTES
    def test_entries_form_one_chain_whatever_the_completion_order(self, example, log, via_plugin):
        many = [("echo", {"text": f"t{i}"}, f"id{i}") for i in range(12)]
        many.insert(5, ("run_sql", {"sql": "DELETE FROM t"}, "refused"))
        example.run_calls(many, log, via_plugin=via_plugin)
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
    def test_decisions_match_the_plain_gate(self, example, gate, log, case, via_plugin):
        policy, calls = _corpus()[case]
        kwargs = {} if policy is None else {"policy": policy}
        plain = self.observed(*gate.run_calls(calls, via_plugin=via_plugin, **kwargs))
        audited = self.observed(*example.run_calls(calls, log, via_plugin=via_plugin, **kwargs))
        assert audited == plain
        assert len(self.by_id(log)) == len(calls)

    def test_the_audited_agent_is_gated_and_the_original_is_not(self, example, gate, log):
        agent = gate.scripted_agent(THREE)
        audited = example.audited_agent(agent, gate.sql_scope_policy, log)
        assert gate.is_gated(audited) and not gate.is_gated(agent)

    @ROUTES
    def test_a_refused_call_does_not_run(self, example, log, via_plugin):
        responses, executed = example.run_calls(
            [("run_sql", {"sql": DELETE}, "c1")], log, via_plugin=via_plugin
        )
        assert executed == []
        assert responses["c1"]["error"].startswith("Recusal refused `run_sql` [FAIL]")
        assert self.by_id(log)["c1"]["action"]["decision"] == "deny"

    # --- the record is part of the control ---------------------------------------------

    @ROUTES
    def test_policy_that_raises_is_denied_nothing_runs_and_the_denial_is_recorded(
        self, example, gate, log, log_path, via_plugin
    ):
        def broken(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        calls = [("echo", {"text": "x"}, "c1")]
        responses, executed = example.run_calls(calls, log, policy=broken, via_plugin=via_plugin)
        assert executed == []
        plain, _ = gate.run_calls(calls, policy=broken, via_plugin=via_plugin)
        assert responses == plain
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["decision"] == "FAIL"
        assert [f["message"] for f in entry["failures"]] == ["RECUSAL_POLICY_ERROR (RuntimeError)"]
        assert "policy exploded" not in responses["c1"]["error"]
        assert "policy exploded" not in log_path.read_text(encoding="utf-8")

    @ROUTES
    def test_unwritable_log_refuses_every_call_and_no_tool_runs(
        self, example, log, log_path, via_plugin
    ):
        example.run_calls([("echo", {"text": "warm-up"}, "w")], log, via_plugin=via_plugin)
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
            responses, executed = example.run_calls(THREE, log, via_plugin=via_plugin)
        finally:
            log_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        assert executed == []
        assert set(responses) == {"c1", "c2", "c3"}
        for response in responses.values():
            assert "audit log unavailable" in response["error"]
            assert "the record is part of the control" in response["error"]
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1

    @ROUTES
    def test_failing_sink_refuses_the_call_and_no_tool_runs(self, example, log_path, via_plugin):
        class Offline:
            def write(self, entry):
                raise OSError("anchor store unreachable")

        log = example.AuditLog(path=str(log_path), sinks=[Offline()])
        responses, executed = example.run_calls(THREE, log, via_plugin=via_plugin)
        assert executed == []
        assert all("RECUSAL_AUDIT_UNAVAILABLE" in r["error"] for r in responses.values())
        assert not any("anchor store unreachable" in r["error"] for r in responses.values())

    @ROUTES
    def test_a_recording_that_raises_refuses_the_call(self, example, gate, via_plugin):
        def broken(tool_context):
            raise RuntimeError("RECORDING_SECRET_SENTINEL")

        agent = gate.scripted_agent([("echo", {"text": "x"}, "c1")])
        plugins = None
        if via_plugin:
            plugins = [gate.RecusalPlugin(gate.sql_scope_policy, recording=broken)]
        else:
            agent = gate.gated_agent(agent, gate.sql_scope_policy, broken)
        gate.EXECUTED.clear()
        responses = asyncio.run(gate.run_agent(agent, plugins=plugins))
        assert gate.EXECUTED == []
        assert responses["c1"] == {
            "error": "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        }

    # --- the record binds what executed ---------------------------------------------------

    @ROUTES
    def test_no_tool_argument_text_appears_in_the_log(self, example, log, log_path, via_plugin):
        secret = "DELETE FROM customers WHERE ssn = 'MARKER-7f3e9'"
        calls = [
            ("run_sql", {"sql": secret}, "s1"),
            ("echo", {"text": "MARKER-echo-text"}, "s2"),
        ]
        example.run_calls(calls, log, via_plugin=via_plugin)
        text = log_path.read_text(encoding="utf-8")
        assert "MARKER" not in text
        assert "customers" not in text
        entries = self.by_id(log)
        for _, args, call_id in calls:
            assert entries[call_id]["action"]["input_sha256"] == _sha(args)
            assert entries[call_id]["action"]["control"]["recusal_version"]

    @ROUTES
    def test_a_default_the_model_omitted_is_recorded_at_the_value_that_runs(
        self, example, log, via_plugin
    ):
        responses, executed = example.run_calls(
            [("charge", {}, "c1")], log, policy=lambda *_: [], via_plugin=via_plugin
        )
        assert executed == [("charge", {"amount": 100})]
        entry = self.by_id(log)["c1"]
        assert entry["action"]["input_sha256"] == _sha({"amount": 100})
        assert entry["action"]["input_sha256"] != _sha({})

    @ROUTES
    def test_the_record_binds_what_executed_whatever_the_policy_did_to_its_copy(
        self, example, log, via_plugin
    ):
        def mutating(tool_name, tool_input):
            tool_input["text"] = "REWRITTEN BY THE POLICY"
            tool_input["extra"] = [1, 2, 3]
            return []

        responses, executed = example.run_calls(
            [("echo", {"text": "as proposed"}, "c1")], log, policy=mutating, via_plugin=via_plugin
        )
        assert executed == [("echo", {"text": "as proposed"})]
        assert self.by_id(log)["c1"]["action"]["input_sha256"] == _sha(executed[0][1])

    def test_the_record_binds_the_arguments_an_earlier_callback_rewrote(self, example, gate, log):
        """The gate is the last before-tool callback, so it judges, and records, the
        arguments as they reach the tool: an earlier callback's rewrite is what ran."""

        def rewrite(tool, args, tool_context):
            args["text"] = "rewritten before the gate"
            return None

        agent = gate.scripted_agent([("echo", {"text": "as proposed"}, "c1")]).clone(
            update={"before_tool_callback": [rewrite]}
        )
        audited = example.audited_agent(agent, gate.sql_scope_policy, log)
        gate.EXECUTED.clear()
        asyncio.run(gate.run_agent(audited))
        assert gate.EXECUTED == [("echo", {"text": "rewritten before the gate"})]
        entry = self.by_id(log)["c1"]
        assert entry["action"]["input_sha256"] == _sha({"text": "rewritten before the gate"})

    # --- refused before judgment: still one entry, bound to the proposal --------------------

    @ROUTES
    def test_every_refusal_before_judgment_is_one_recorded_deny(
        self, example, log, log_path, via_plugin
    ):
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        calls = [
            ("run_sql", {"sql": SELECT, "surprise": "MARKER-extra"}, "unknown_param"),
            ("run_sql", {}, "missing"),
        ]
        responses, executed = example.run_calls(calls, log, policy=policy, via_plugin=via_plugin)
        assert executed == [] and seen == []
        entries = self.by_id(log)
        assert set(entries) == {c[2] for c in calls}
        details = {
            "unknown_param": "unknown parameter(s) ['surprise']",
            "missing": "required parameter `sql` was not supplied",
        }
        for name, args, call_id in calls:
            entry = entries[call_id]
            reason = responses[call_id]["error"]
            assert reason.startswith(f"Recusal refused `{name}` [FAIL]")
            assert details[call_id] in reason
            assert entry["action"]["decision"] == "deny" and entry["decision"] == "FAIL"
            assert entry["action"]["reason"] == reason
            assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
            assert entry["action"]["input_sha256"] == _sha(args)
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    def test_a_plugin_that_is_not_the_last_seam_is_one_recorded_deny(self, example, gate, log):
        def rewrite(tool, args, tool_context):
            args["text"] = "rewritten after the plugin"
            return None

        agent = gate.scripted_agent([("echo", {"text": "as proposed"}, "c1")]).clone(
            update={"before_tool_callback": [rewrite]}
        )
        gate.EXECUTED.clear()
        responses = asyncio.run(
            gate.run_agent(agent, plugins=[example.audited_plugin(gate.sql_scope_policy, log)])
        )
        assert gate.EXECUTED == []
        reason = responses["c1"]["error"]
        assert "runs a before_tool_callback (rewrite) after this plugin" in reason
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["reason"] == reason
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["input_sha256"] == _sha({"text": "as proposed"})

    def test_non_mapping_arguments_are_one_recorded_deny(self, example, gate, log):
        tool = gate.FunctionTool(gate.run_sql)
        record = {"audit": log, "surface": SURFACE, "tool_use_id": "c1"}
        reason = _adapter()._refusal(tool, ["DELETE"], gate.sql_scope_policy, **record)
        assert reason is not None and "arguments are not a mapping" in reason
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["input_sha256"] == _sha({"input": ["DELETE"]})
        # without the recording arguments the plain text is unchanged and nothing is written
        assert _adapter()._refusal(tool, ["DELETE"], gate.sql_scope_policy) == (
            "Recusal refused `run_sql`: arguments are not a mapping"
        )
        assert len(log.entries) == 1

    def test_a_blank_call_id_is_recorded_as_no_id(self, example, gate, log):
        class Context:
            function_call_id = ""

        record = example.recording(log)(Context())
        assert record["tool_use_id"] is None
        tool = gate.FunctionTool(gate.echo)
        assert _adapter()._refusal(tool, {"text": "x"}, lambda *_: [], **record) is None
        assert len(log.entries) == 1
        assert "tool_use_id" not in log.entries[0]["action"]

    # --- what the gate never sees is not on the log, as the example says -------------------

    def test_a_call_an_earlier_callback_answered_does_not_run_and_is_not_recorded(
        self, example, gate, log
    ):
        def answer(tool, args, tool_context):
            return {"result": "answered by another callback"}

        agent = gate.scripted_agent([("run_sql", {"sql": DELETE}, "c1")]).clone(
            update={"before_tool_callback": [answer]}
        )
        audited = example.audited_agent(agent, gate.sql_scope_policy, log)
        gate.EXECUTED.clear()
        responses = asyncio.run(gate.run_agent(audited))
        assert gate.EXECUTED == []
        assert responses["c1"] == {"result": "answered by another callback"}
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
            copy = log_path.with_name(f"tampered-{index}.jsonl")
            copy.write_text("\n".join(edited) + "\n", encoding="utf-8")
            code, text = example.verify_with_cli(str(copy))
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
    def test_negative_control_plain_gate_writes_no_entries(self, gate, log, log_path, via_plugin):
        responses, _ = gate.run_calls(THREE, via_plugin=via_plugin)
        assert self.refused(responses["c2"])
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
