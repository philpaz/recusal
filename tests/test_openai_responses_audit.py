"""Regression tests for the offline OpenAI direct-loop audit-trail example.

Every function call the audited loop sees, through the Responses API shape or the Chat
Completions shape, lands on a hash-chained log exactly once, labeled with that shape's
surface and keyed by the API's own call id, and the audited loop decides exactly as the
plain ``gate_function_calls`` and ``gate_tool_calls`` do.
"""

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
EXAMPLE = ROOT / "examples" / "openai_responses_audit.py"
SURFACES = {False: "openai.responses.function_call", True: "openai.chat_completions.tool_call"}

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"

SHAPES = pytest.mark.parametrize("chat", [False, True], ids=["responses", "chat"])


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
    assert "openai_responses_audit.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_pinned_version_is_explicit_in_the_example():
    assert "openai==3.34.0" in EXAMPLE.read_text(encoding="utf-8")


def test_both_openai_lanes_run_this_file():
    # no new jobs: the two OpenAI jobs and the weekly lane run it
    for workflow in ("ci.yml", "compat.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert text.count("tests/test_openai_responses_audit.py") == 2, workflow


def test_the_works_with_table_marks_the_audit_trail_tested():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    assert (
        "| OpenAI Responses and Chat Completions audit trail | Tested | "
        "[`examples/openai_responses_audit.py`]" in table
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
        "sql-scope-none-value": (None, [("run_sql", {"sql": None}, "c1")]),
        "default-omitted": (None, [("charge", {}, "c1")]),
        "raising": (raising, THREE),
        "silent": (silent, THREE),
        "ambiguous": (ambiguous, THREE),
        "warning-only": (warning_only, THREE),
        "error-level": (error_level, THREE),
    }


@pytest.mark.skipif(sys.version_info < (3, 10), reason="the OpenAI SDK requires Python 3.10+")
class TestOpenAIResponsesAudit:
    @pytest.fixture
    def example(self):
        pytest.importorskip("openai", reason="openai is not installed")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location(
            "recusal_openai_responses_audit_example", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def gate(self, example):
        """The plain gate example, the SAME module object the audited one imports from."""
        return importlib.import_module("examples.openai_responses_gate")

    @pytest.fixture
    def log_path(self, tmp_path):
        return tmp_path / "openai-audit.jsonl"

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
    def plain(gate, calls, *, chat, policy=None, tools=None):
        """The same calls through the plain gate: (answers by id, what ran)."""
        policy = gate.sql_scope_policy if policy is None else policy
        registry = gate.TOOLS if tools is None else tools
        gate.EXECUTED.clear()
        if chat:
            outputs = gate.gate_tool_calls(
                gate.completion_with([gate.tool_call(*c) for c in calls]), registry, policy
            )
            answers = {o["tool_call_id"]: o["content"] for o in outputs}
        else:
            outputs = gate.gate_function_calls(
                gate.response_with([gate.function_call(*c) for c in calls]), registry, policy
            )
            answers = {o["call_id"]: o["output"] for o in outputs}
        return answers, list(gate.EXECUTED)

    # --- every call recorded once, run and refused alike ----------------------------------

    @SHAPES
    def test_every_call_is_recorded_once_with_surface_and_id(self, example, log, chat):
        answers, executed = example.run_calls(THREE, log, actor="demo-agent", chat=chat)
        assert executed == [("echo", {"text": "hello"}), ("run_sql", {"sql": SELECT})]
        entries = self.by_id(log)
        assert set(entries) == {"c1", "c2", "c3"}
        for entry in entries.values():
            assert entry["action"]["surface"] == SURFACES[chat]
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
        assert entries["c2"]["action"]["reason"] == answers["c2"]

    @SHAPES
    def test_entries_form_one_chain_in_call_order(self, example, log, chat):
        many = [("echo", {"text": f"t{i}"}, f"id{i}") for i in range(12)]
        many.insert(5, ("run_sql", {"sql": "DELETE FROM t"}, "refused"))
        example.run_calls(many, log, chat=chat)
        assert [e["action"]["tool_use_id"] for e in log.entries] == [c[2] for c in many]
        assert [e["seq"] for e in log.entries] == list(range(13))
        for previous, entry in zip(log.entries, log.entries[1:]):
            assert entry["prev_hash"] == previous["hash"]
        from recusal.audit import verify

        ok, problems = verify(log.entries)
        assert ok, problems

    # --- decision equivalence with the plain gate -----------------------------------------

    @SHAPES
    @pytest.mark.parametrize("case", sorted(_corpus()))
    def test_decisions_match_the_plain_gate(self, example, gate, log, case, chat):
        policy, calls = _corpus()[case]
        kwargs = {} if policy is None else {"policy": policy}
        plain = self.plain(gate, calls, chat=chat, **kwargs)
        audited = example.run_calls(calls, log, chat=chat, **kwargs)
        assert audited == plain
        assert len(self.by_id(log)) == len(calls)

    @SHAPES
    def test_a_refused_call_does_not_run(self, example, log, chat):
        answers, executed = example.run_calls([("run_sql", {"sql": DELETE}, "c1")], log, chat=chat)
        assert executed == []
        assert answers["c1"].startswith("Recusal refused `run_sql` [FAIL]")
        assert self.by_id(log)["c1"]["action"]["decision"] == "deny"

    # --- the record is part of the control ---------------------------------------------

    @SHAPES
    def test_policy_that_raises_is_denied_nothing_runs_and_the_denial_is_recorded(
        self, example, gate, log, log_path, chat
    ):
        def broken(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        calls = [("echo", {"text": "x"}, "c1")]
        answers, executed = example.run_calls(calls, log, policy=broken, chat=chat)
        assert executed == []
        assert answers == self.plain(gate, calls, chat=chat, policy=broken)[0]
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["decision"] == "FAIL"
        assert [f["message"] for f in entry["failures"]] == ["RECUSAL_POLICY_ERROR (RuntimeError)"]
        assert "policy exploded" not in answers["c1"]
        assert "policy exploded" not in log_path.read_text(encoding="utf-8")

    @SHAPES
    def test_unwritable_log_refuses_every_call_and_no_tool_runs(self, example, log, log_path, chat):
        example.run_calls([("echo", {"text": "warm-up"}, "w")], log, chat=chat)
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
            answers, executed = example.run_calls(THREE, log, chat=chat)
        finally:
            log_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        assert executed == []
        assert set(answers) == {"c1", "c2", "c3"}
        for content in answers.values():
            assert "audit log unavailable" in content
            assert "the record is part of the control" in content
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1

    @SHAPES
    def test_failing_sink_refuses_the_call_and_no_tool_runs(self, example, log_path, chat):
        class Offline:
            def write(self, entry):
                raise OSError("anchor store unreachable")

        log = example.AuditLog(path=str(log_path), sinks=[Offline()])
        answers, executed = example.run_calls(THREE, log, chat=chat)
        assert executed == []
        assert all("RECUSAL_AUDIT_UNAVAILABLE" in content for content in answers.values())
        assert not any("anchor store unreachable" in content for content in answers.values())

    # --- the record binds what executed ---------------------------------------------------

    @SHAPES
    def test_no_tool_argument_text_appears_in_the_log(self, example, log, log_path, chat):
        secret = "DELETE FROM customers WHERE ssn = 'MARKER-7f3e9'"
        calls = [
            ("run_sql", {"sql": secret}, "s1"),
            ("echo", {"text": "MARKER-echo-text"}, "s2"),
        ]
        example.run_calls(calls, log, chat=chat)
        text = log_path.read_text(encoding="utf-8")
        assert "MARKER" not in text
        assert "customers" not in text
        entries = self.by_id(log)
        for _, args, call_id in calls:
            assert entries[call_id]["action"]["input_sha256"] == _sha(args)
            assert entries[call_id]["action"]["control"]["recusal_version"]

    @SHAPES
    def test_a_default_the_model_omitted_is_recorded_at_the_value_that_runs(
        self, example, log, chat
    ):
        answers, executed = example.run_calls(
            [("charge", {}, "c1")], log, policy=lambda *_: [], chat=chat
        )
        assert executed == [("charge", {"amount": 100})]
        entry = self.by_id(log)["c1"]
        assert entry["action"]["input_sha256"] == _sha({"amount": 100})
        assert entry["action"]["input_sha256"] != _sha({})

    @SHAPES
    def test_the_record_binds_what_executed_whatever_the_policy_did_to_its_copy(
        self, example, log, chat
    ):
        def mutating(tool_name, tool_input):
            tool_input["text"] = "REWRITTEN BY THE POLICY"
            tool_input["extra"] = [1, 2, 3]
            return []

        answers, executed = example.run_calls(
            [("echo", {"text": "as proposed"}, "c1")], log, policy=mutating, chat=chat
        )
        assert executed == [("echo", {"text": "as proposed"})]
        entry = self.by_id(log)["c1"]
        assert entry["action"]["input_sha256"] == _sha(executed[0][1])

    @SHAPES
    def test_a_nested_invocation_is_recorded_as_executed(self, example, log, chat):
        ran = []

        def store(payload: dict[str, list[int]], tags: list[str]) -> str:
            ran.append({"payload": payload, "tags": tags})
            return "stored"

        args = {"payload": {"a": [1, 2], "b": [3]}, "tags": ["x", "y"]}
        answers, _ = example.run_calls(
            [("store", args, "c1")], log, policy=lambda *_: [], tools={"store": store}, chat=chat
        )
        assert answers["c1"] == "stored"
        assert ran == [args]
        assert self.by_id(log)["c1"]["action"]["input_sha256"] == _sha(ran[0])

    # --- refused before judgment: still one entry, bound to the proposal --------------------

    @SHAPES
    def test_every_refusal_before_judgment_is_one_recorded_deny(
        self, example, gate, log, log_path, chat
    ):
        def variadic(*args: str) -> str:
            return "ran"

        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        tools = dict(gate.TOOLS, variadic=variadic)
        calls = [
            ("run_sql", "{not json MARKER-raw", "bad_json"),
            ("run_sql", ["MARKER-list"], "not_object"),
            ("drop_db", {"name": "MARKER-prod"}, "unknown"),
            ("variadic", {"x": "MARKER-variadic"}, "signature"),
            ("run_sql", {"sql": SELECT, "surprise": "MARKER-extra"}, "unbound"),
        ]
        answers, executed = example.run_calls(calls, log, policy=policy, tools=tools, chat=chat)
        assert executed == [] and seen == []
        entries = self.by_id(log)
        assert set(entries) == {c[2] for c in calls}
        expected = {
            "bad_json": ("arguments are not valid JSON", {"input": "{not json MARKER-raw"}),
            "not_object": ("arguments are not a JSON object", {"input": ["MARKER-list"]}),
            "unknown": ("unknown tool", {"name": "MARKER-prod"}),
            "signature": ("its signature cannot be judged", {"x": "MARKER-variadic"}),
            "unbound": (
                "arguments do not bind to the tool",
                {"sql": SELECT, "surprise": "MARKER-extra"},
            ),
        }
        for call_id, (detail, proposal) in expected.items():
            entry = entries[call_id]
            assert entry["action"]["decision"] == "deny" and entry["decision"] == "FAIL"
            assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
            assert detail in answers[call_id]
            assert entry["action"]["reason"] == answers[call_id]
            assert entry["action"]["input_sha256"] == _sha(proposal)
        assert "MARKER" not in log_path.read_text(encoding="utf-8")
        # the plain gate's text for the same refusals is unchanged
        plain, _ = self.plain(gate, calls, chat=chat, policy=policy, tools=tools)
        assert plain["bad_json"] == "Recusal refused `run_sql`: arguments are not valid JSON"
        assert plain["unknown"] == "Recusal refused: unknown tool 'drop_db'"

    # --- a turn refused as a whole runs nothing and writes nothing, as the example says ----

    def test_a_turn_with_an_unhandled_item_type_raises_and_writes_no_entry(
        self, example, gate, log
    ):
        response = gate.response_with(
            [
                gate.function_call("echo", {"text": "x"}, "c1"),
                {
                    "type": "custom_tool_call",
                    "id": "ct_1",
                    "call_id": "c9",
                    "name": "x",
                    "input": "raw",
                },
            ]
        )
        gate.EXECUTED.clear()
        with pytest.raises(TypeError):
            example.audited_function_calls(response, gate.TOOLS, gate.sql_scope_policy, log)
        assert gate.EXECUTED == [] and log.entries == []

    @SHAPES
    def test_a_turn_with_a_duplicate_id_raises_and_writes_no_entry(self, example, gate, log, chat):
        calls = [("echo", {"text": "a"}, "dup"), ("echo", {"text": "b"}, "dup")]
        gate.EXECUTED.clear()
        with pytest.raises(ValueError):
            example.run_calls(calls, log, chat=chat)
        assert gate.EXECUTED == [] and log.entries == []

    def test_only_the_chosen_alternative_is_judged_and_recorded(self, example, gate, log):
        completion = gate.completion_with(
            [gate.tool_call("echo", {"text": "first"}, "a1")],
            [gate.tool_call("echo", {"text": "second"}, "b1")],
        )
        gate.EXECUTED.clear()
        with pytest.raises(ValueError):
            example.audited_tool_calls(completion, gate.TOOLS, gate.sql_scope_policy, log)
        assert log.entries == []
        example.audited_tool_calls(completion, gate.TOOLS, gate.sql_scope_policy, log, choice=1)
        assert gate.EXECUTED == [("echo", {"text": "second"})]
        assert set(self.by_id(log)) == {"b1"}

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

    @SHAPES
    def test_negative_control_plain_gate_writes_no_entries(self, gate, log, log_path, chat):
        answers, _ = self.plain(gate, THREE, chat=chat)
        assert answers["c2"].startswith("Recusal refused")
        assert log.entries == []
        assert not log_path.exists()

    def test_negative_control_ungated_loop_records_nothing_and_runs_the_delete(self, gate, log):
        gate.EXECUTED.clear()
        gate.dispatch_ungated(
            gate.response_with([gate.function_call(*c) for c in THREE]), gate.TOOLS
        )
        assert ("run_sql", {"sql": DELETE}) in gate.EXECUTED
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
        assert "call_2 refused" in out
        assert f"call_2 {SURFACES[False]} run_sql  deny" in out
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
