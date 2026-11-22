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
    assert "openai==3.49.0" in EXAMPLE.read_text(encoding="utf-8")


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
            # a key is the model's text too: an argument the tool does not take is counted
            # in the refusal, never quoted
            ("echo", {"text": "x", "MARKER-key password=hunter2": 1}, "s3"),
            ("echo", {"MARKER-only-key": "MARKER-value"}, "s4"),
        ]
        answers, _ = example.run_calls(calls, log, chat=chat)
        text = log_path.read_text(encoding="utf-8")
        assert "MARKER" not in text
        assert "customers" not in text and "hunter2" not in text
        assert "MARKER" not in answers["s3"] + answers["s4"]
        entries = self.by_id(log)
        assert entries["s3"]["reasons"] == (
            "arguments do not bind to the tool (1 argument the tool does not take)"
        )
        assert entries["s4"]["reasons"] == (
            "arguments do not bind to the tool "
            "(missing required `text`; 1 argument the tool does not take)"
        )
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

        def rows(sql: str):
            yield sql

        nan = '{"sql": NaN, "m": "MARKER-nan"}'
        infinite = '{"sql": 1e999, "m": "MARKER-inf"}'
        deep = '{"sql": "MARKER-deep", "m": ' + "[" * 600 + "]" * 600 + "}"
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        tools = dict(gate.TOOLS, variadic=variadic, rows=rows)
        calls = [
            ("run_sql", "{not json MARKER-raw", "bad_json"),
            ("run_sql", ["MARKER-list"], "not_object"),
            ("drop_db", {"name": "MARKER-prod"}, "unknown"),
            ("variadic", {"x": "MARKER-variadic"}, "signature"),
            ("run_sql", {"sql": SELECT, "MARKER-key": "MARKER-extra"}, "unbound"),
            ("run_sql", nan, "nan"),
            ("run_sql", infinite, "infinite"),
            ("run_sql", deep, "deep"),
            ("rows", {"sql": "MARKER-rows"}, "generator"),
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
                "arguments do not bind to the tool (1 argument the tool does not take)",
                {"sql": SELECT, "MARKER-key": "MARKER-extra"},
            ),
            # refused for what the text is: bound to the text as sent
            "nan": ("arguments are not valid JSON", {"input": nan}),
            "infinite": ("a number in the arguments is not finite", {"input": infinite}),
            "deep": ("arguments are nested too deeply", {"input": deep}),
            "generator": ("it is a generator function", {"sql": "MARKER-rows"}),
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

    # --- synchronous dispatch and hosted records on the record (0.2.0) ---------------------

    @SHAPES
    def test_an_async_tool_is_one_recorded_deny_and_never_runs(self, example, gate, log, chat):
        ran = []

        async def fetch(text: str) -> str:
            ran.append(text)
            return text

        answers, _ = example.run_calls(
            [("fetch", {"text": "x"}, "c1")],
            log,
            policy=lambda n, a: [],
            tools={"fetch": fetch},
            chat=chat,
        )
        answer = answers["c1"]
        entry = self.by_id(log)["c1"]
        assert entry["decision"] == "FAIL" and entry["action"]["decision"] == "deny"
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert "declared async" in answer and entry["action"]["reason"] == answer
        assert ran == [] and len(log.entries) == 1

    def test_by_default_a_tool_that_raises_propagates_after_its_own_entry_and_no_later_one(
        self, example, gate, log
    ):
        def boom(text: str) -> str:
            raise RuntimeError("SECRET-MARKER")

        response = gate.response_with(
            [
                gate.function_call("boom", {"text": "x"}, "c1"),
                gate.function_call("echo", {"text": "after"}, "c2"),
            ]
        )
        gate.EXECUTED.clear()
        with pytest.raises(RuntimeError):
            example.audited_function_calls(
                response, {"boom": boom, "echo": gate.echo}, lambda n, a: [], log
            )
        assert [e["action"]["tool_use_id"] for e in log.entries] == ["c1"]
        assert gate.EXECUTED == []

    @SHAPES
    def test_an_answered_tool_error_is_recorded_as_allowed_and_the_next_call_still_runs(
        self, example, gate, log, log_path, chat
    ):
        def boom(text: str) -> str:
            raise RuntimeError("SECRET-MARKER")

        calls = [("boom", {"text": "x"}, "c1"), ("echo", {"text": "after"}, "c2")]
        tools = {"boom": boom, "echo": gate.echo}
        gate.EXECUTED.clear()
        if chat:
            outs = example.audited_tool_calls(
                gate.completion_with([gate.tool_call(*c) for c in calls]),
                tools,
                lambda n, a: [],
                log,
                answer_tool_errors=True,
            )
            answers = [o["content"] for o in outs]
        else:
            outs = example.audited_function_calls(
                gate.response_with([gate.function_call(*c) for c in calls]),
                tools,
                lambda n, a: [],
                log,
                answer_tool_errors=True,
            )
            answers = [o["output"] for o in outs]
        assert answers == ["Tool `boom` raised RuntimeError", "after"]
        entries = self.by_id(log)
        # the gate's decision was to allow; what the tool then did is the tool's
        assert entries["c1"]["decision"] == "PASS" and entries["c2"]["decision"] == "PASS"
        assert gate.EXECUTED == [("echo", {"text": "after"})]
        assert "SECRET-MARKER" not in log_path.read_text(encoding="utf-8")

    def test_a_returned_coroutine_is_on_the_log_as_allowed_and_answered_as_not_waited_for(
        self, example, gate, log
    ):
        async def inner(text: str) -> str:
            return text

        def outer(text: str) -> str:
            return inner(text)  # type: ignore[return-value]

        response = gate.response_with([gate.function_call("outer", {"text": "x"}, "c1")])
        outs = example.audited_function_calls(response, {"outer": outer}, lambda n, a: [], log)
        assert "returned a coroutine, not a result" in outs[0]["output"]
        assert self.by_id(log)["c1"]["decision"] == "PASS"

    def test_a_hosted_record_is_never_an_entry(self, example, gate, log):
        response = gate.response_with(
            [
                {
                    "type": "web_search_call",
                    "id": "ws",
                    "status": "completed",
                    "action": {"type": "search"},
                },
                gate.function_call("echo", {"text": "x"}, "c1"),
            ]
        )
        with pytest.raises(TypeError, match="allow_hosted_records=True"):
            example.audited_function_calls(response, gate.TOOLS, lambda n, a: [], log)
        assert log.entries == []
        outs = example.audited_function_calls(
            response, gate.TOOLS, lambda n, a: [], log, allow_hosted_records=True
        )
        assert [o["call_id"] for o in outs] == ["c1"]
        assert [e["action"]["tool_use_id"] for e in log.entries] == ["c1"]

    # --- what cannot be tied to what runs: one recorded deny each (0.2.2) -------------------

    @SHAPES
    @pytest.mark.parametrize("levels", [65, 600, 5000])
    def test_a_call_nested_too_deeply_is_one_recorded_deny_and_its_neighbours_run(
        self, example, log, chat, levels
    ):
        """At about 600 levels the copy for the policy raised inside the gate: the refusal
        was not on the log and read "policy error". At about 5000 the parser raised out of
        the turn after the first call had run."""
        ran = []

        def nest(path: str, extra: list[str] = None) -> str:  # type: ignore[assignment]
            ran.append(path)
            return "nested"

        seen = []
        deep = '{"path": "p", "extra": ' + "[" * levels + "]" * levels + "}"
        answers, _ = example.run_calls(
            [("nest", {"path": "a"}, "c1"), ("nest", deep, "c2"), ("nest", {"path": "z"}, "c3")],
            log,
            policy=lambda n, a: seen.append(a["path"]) or [],
            tools={"nest": nest},
            chat=chat,
        )
        assert ran == ["a", "z"] and seen == ["a", "z"]
        assert [(e["action"]["tool_use_id"], e["action"]["decision"]) for e in log.entries] == [
            ("c1", "defer"),
            ("c2", "deny"),
            ("c3", "defer"),
        ]
        entry = self.by_id(log)["c2"]
        assert answers["c2"] == "Recusal refused `nest` [FAIL]: arguments are nested too deeply"
        assert entry["action"]["reason"] == answers["c2"] and "policy error" not in answers["c2"]
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["input_sha256"] == _sha({"input": deep})

    @SHAPES
    @pytest.mark.parametrize(
        "arguments, detail",
        [
            ('{"amount": NaN}', "arguments are not valid JSON"),
            ('{"amount": -Infinity}', "arguments are not valid JSON"),
            ('{"amount": 1e999}', "a number in the arguments is not finite"),
            ('{"amount": -1e999}', "a number in the arguments is not finite"),
        ],
    )
    def test_both_loops_refuse_a_number_that_is_not_json_the_same_way(
        self, example, gate, log, chat, arguments, detail
    ):
        """The plain loop ran these and the audited loop refused them."""
        from recusal import Finding

        ran = []

        def pay(amount: float) -> str:
            ran.append(amount)
            return "paid"

        def limit(name, args):
            if args["amount"] > 100:
                return [Finding.fail("limit", severity="CRITICAL", message="over limit")]
            return []

        calls = [("pay", arguments, "c1")]
        plain, _ = self.plain(gate, calls, chat=chat, policy=limit, tools={"pay": pay})
        answers, _ = example.run_calls(calls, log, policy=limit, tools={"pay": pay}, chat=chat)
        assert plain["c1"] == f"Recusal refused `pay`: {detail}"
        assert answers["c1"] == f"Recusal refused `pay` [FAIL]: {detail}"
        assert ran == [] and len(log.entries) == 1
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["input_sha256"] == _sha({"input": arguments})

    def test_a_namespaced_call_is_one_recorded_deny_bound_to_the_arguments_as_sent(
        self, example, gate, log
    ):
        ran = []

        def delete(path: str) -> str:
            ran.append(path)
            return "deleted"

        seen = []
        call = dict(gate.function_call("delete", {"path": "x"}, "c1"), namespace="prod_db")
        response = gate.response_with([call, gate.function_call("delete", {"path": "y"}, "c2")])
        outs = example.audited_function_calls(
            response, {"delete": delete}, lambda n, a: seen.append(a) or [], log
        )
        assert outs[0]["output"] == (
            "Recusal refused `delete` [FAIL]: it carries a namespace, and namespaced function "
            "calls are not supported"
        )
        assert outs[1]["output"] == "deleted" and ran == ["y"] and seen == [{"path": "y"}]
        entries = self.by_id(log)
        assert set(entries) == {"c1", "c2"} and len(log.entries) == 2
        entry = entries["c1"]
        assert entry["decision"] == "FAIL" and entry["action"]["decision"] == "deny"
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["reason"] == outs[0]["output"]
        assert entry["action"]["input_sha256"] == _sha({"input": call["arguments"]})
        assert "prod_db" not in json.dumps(entry)

    @pytest.mark.parametrize("where", ["item", "response"])
    def test_an_unfinished_function_call_is_one_recorded_deny(self, example, gate, log, where):
        ran = []

        def pay(amount: float = 5.0) -> str:
            ran.append(amount)
            return "paid"

        call = gate.function_call("pay", "", "c1")
        if where == "item":
            call["status"] = "incomplete"
        response = gate.response_with([call])
        if where == "response":
            response.status = "incomplete"
        outs = example.audited_function_calls(response, {"pay": pay}, lambda n, a: [], log)
        what = "call" if where == "item" else "response"
        assert outs[0]["output"] == (
            f"Recusal refused `pay` [FAIL]: the {what} is not finished "
            "(its status is not `completed`)"
        )
        assert ran == [] and len(log.entries) == 1
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["reason"] == outs[0]["output"]
        assert entry["action"]["input_sha256"] == _sha({"input": ""})

    def test_an_unfinished_chat_message_is_one_recorded_deny_per_call(self, example, gate, log):
        ran = []

        def pay(amount: float = 5.0) -> str:
            ran.append(amount)
            return "paid"

        completion = gate.completion_with(
            [gate.tool_call("pay", {"amount": 1}, "k1"), gate.tool_call("pay", "", "k2")]
        )
        completion.choices[0].finish_reason = "length"
        outs = example.audited_tool_calls(completion, {"pay": pay}, lambda n, a: [], log)
        assert all("the message is not finished" in o["content"] for o in outs)
        assert ran == []
        assert [(e["action"]["tool_use_id"], e["action"]["decision"]) for e in log.entries] == [
            ("k1", "deny"),
            ("k2", "deny"),
        ]
        entries = self.by_id(log)
        assert entries["k1"]["action"]["input_sha256"] == _sha({"input": '{"amount": 1}'})
        assert entries["k2"]["action"]["input_sha256"] == _sha({"input": ""})

    def test_arguments_that_are_not_a_string_are_one_recorded_deny(self, example, log):
        from types import SimpleNamespace

        ran = []

        def pay(amount: float = 5.0) -> str:
            ran.append(amount)
            return "paid"

        response = SimpleNamespace(
            output=[SimpleNamespace(type="function_call", call_id="c1", name="pay", arguments=None)]
        )
        outs = example.audited_function_calls(response, {"pay": pay}, lambda n, a: [], log)
        assert outs[0]["output"] == "Recusal refused `pay` [FAIL]: arguments are not a string"
        assert ran == [] and len(log.entries) == 1
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["input_sha256"] == _sha({"input": None})

    def test_an_invocation_that_cannot_be_detached_is_one_recorded_deny(
        self, example, log, monkeypatch
    ):
        module = importlib.import_module("recusal_openai._gate")

        def refuse(value):
            raise RecursionError("too deep")

        monkeypatch.setattr(module, "_detached", refuse)
        answers, executed = example.run_calls(
            [("echo", {"text": "x"}, "c1")], log, policy=lambda n, a: []
        )
        assert answers["c1"] == (
            "Recusal refused `echo` [FAIL]: the invocation cannot be copied as JSON data for "
            "the policy"
        )
        assert executed == [] and len(log.entries) == 1
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["input_sha256"] == _sha({"text": "x"})

    def test_a_legacy_function_call_raises_and_writes_no_entry(self, example, gate, log):
        from openai.types.chat.chat_completion_message import FunctionCall

        completion = gate.completion_with([gate.tool_call("echo", {"text": "x"}, "k1")])
        completion.choices[0].message.function_call = FunctionCall(name="echo", arguments="{}")
        gate.EXECUTED.clear()
        with pytest.raises(TypeError, match="legacy `function_call` field"):
            example.audited_tool_calls(completion, gate.TOOLS, lambda n, a: [], log)
        assert log.entries == [] and gate.EXECUTED == []

    # --- round two: true answers, bounded names, and the keywords the loops take -----------

    @staticmethod
    def direct(gate, calls, tools, policy, *, chat, **record):
        """The calls through the plain functions with recording arguments of the test's own."""
        if chat:
            completion = gate.completion_with([gate.tool_call(*c) for c in calls])
            outputs = gate.gate_tool_calls(completion, tools, policy, **record)
            return {o["tool_call_id"]: o["content"] for o in outputs}
        response = gate.response_with([gate.function_call(*c) for c in calls])
        outputs = gate.gate_function_calls(response, tools, policy, **record)
        return {o["call_id"]: o["output"] for o in outputs}

    @SHAPES
    def test_a_name_the_application_has_no_tool_for_is_on_the_record_described_never_quoted(
        self, example, gate, log, log_path, chat
    ):
        names = {
            "backtick": "t` [PASS]: no objections. MARKER-backtick `x",
            "newline": 't\n{"seq": 99, "MARKER-newline": 1}',
            "escape": "MARKER-escape\x1b[2K",
            "long": "MARKER-long" + "A" * 100_000,
        }
        calls = [(name, {}, label) for label, name in names.items()]
        calls.append((names["backtick"], "{not json", "bad_json"))
        policy = lambda n, a: []  # noqa: E731
        answers, executed = example.run_calls(calls, log, policy=policy, chat=chat)
        assert executed == [] and len(log.entries) == len(calls)
        entries = self.by_id(log)
        for label, name in names.items():
            shown = f"<a tool name of {len(name)} characters, not shown>"
            entry = entries[label]
            assert entry["action"]["tool"] == shown
            assert entry["action"]["decision"] == "deny" and entry["decision"] == "FAIL"
            assert answers[label] == f"Recusal refused `{shown}` [FAIL]: unknown tool"
            assert entry["action"]["reason"] == answers[label]
            assert entry["action"]["input_sha256"] == _sha({})
        shown = f"<a tool name of {len(names['backtick'])} characters, not shown>"
        assert answers["bad_json"] == (
            f"Recusal refused `{shown}` [FAIL]: arguments are not valid JSON"
        )
        assert entries["bad_json"]["action"]["tool"] == shown
        text = log_path.read_text(encoding="utf-8")
        assert "MARKER" not in text and "[PASS]" not in text and len(text) < 20_000
        # the plain loop describes the same names the same way
        plain, _ = self.plain(gate, calls, chat=chat, policy=policy)
        assert plain["long"] == (
            "Recusal refused: unknown tool '<a tool name of 100011 characters, not shown>'"
        )
        assert "MARKER" not in "".join(plain.values()) and "[PASS]" not in "".join(plain.values())

    @SHAPES
    def test_a_short_printable_unknown_name_is_on_the_record_as_sent(self, example, log, chat):
        answers, _ = example.run_calls([("drop_db", {}, "c1")], log, chat=chat)
        assert answers["c1"] == "Recusal refused `drop_db` [FAIL]: unknown tool"
        assert self.by_id(log)["c1"]["action"]["tool"] == "drop_db"

    def test_a_tool_name_that_is_not_a_string_is_one_recorded_deny(self, example, log):
        from types import SimpleNamespace

        ran = []

        def pay(amount: float = 5.0) -> str:
            ran.append(amount)
            return "paid"

        response = SimpleNamespace(
            output=[
                SimpleNamespace(type="function_call", call_id="c0", name="pay", arguments=""),
                SimpleNamespace(type="function_call", call_id="c1", name=["pay"], arguments="{}"),
            ]
        )
        outs = example.audited_function_calls(response, {"pay": pay}, lambda n, a: [], log)
        assert [o["output"] for o in outs] == [
            "paid",
            "Recusal refused `<a tool name that is not a string>` [FAIL]: unknown tool",
        ]
        assert ran == [5.0] and len(log.entries) == 2
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["tool"] == "<a tool name that is not a string>"
        assert entry["action"]["reason"] == outs[1]["output"]
        assert entry["action"]["input_sha256"] == _sha({})

    @SHAPES
    def test_a_cached_tool_handed_a_list_is_one_recorded_deny(
        self, example, gate, log, log_path, chat
    ):
        import functools

        ran = []

        @functools.lru_cache(maxsize=None)
        def lookup(key: str) -> str:
            ran.append(key)
            return f"found {key}"

        seen = []
        policy = lambda n, a: seen.append(a) or []  # noqa: E731
        calls = [("lookup", {"key": ["MARKER-a"]}, "c1"), ("lookup", {"key": "b"}, "c2")]
        answers, _ = example.run_calls(
            calls, log, policy=policy, tools={"lookup": lookup}, chat=chat
        )
        assert answers == {
            "c1": "Recusal refused `lookup` [FAIL]: it is cached with `functools.lru_cache`, "
            "which cannot take a list or an object as an argument",
            "c2": "found b",
        }
        assert ran == ["b"] and seen == [{"key": "b"}] and len(log.entries) == 2
        entries = self.by_id(log)
        assert entries["c1"]["action"]["decision"] == "deny"
        assert [f["check"] for f in entries["c1"]["failures"]] == ["recusal_unjudgeable_input"]
        assert entries["c1"]["action"]["reason"] == answers["c1"]
        assert entries["c1"]["action"]["input_sha256"] == _sha({"key": ["MARKER-a"]})
        assert entries["c2"]["action"]["decision"] == "defer"
        assert "MARKER" not in log_path.read_text(encoding="utf-8")
        ran.clear()
        plain, _ = self.plain(gate, calls, chat=chat, policy=policy, tools={"lookup": lookup})
        assert plain["c1"] == answers["c1"].replace(" [FAIL]", "") and plain["c2"] == "found b"

    @SHAPES
    def test_a_generator_function_behind_lru_cache_is_one_recorded_deny(self, example, log, chat):
        import functools

        ran = []

        @functools.lru_cache(maxsize=None)
        def rows(sql: str):
            ran.append(sql)
            yield sql

        seen = []
        answers, _ = example.run_calls(
            [("rows", {"sql": "x"}, "c1")],
            log,
            policy=lambda n, a: seen.append(a) or [],
            tools={"rows": rows},
            chat=chat,
        )
        assert answers["c1"] == (
            "Recusal refused `rows` [FAIL]: it is a generator function, and calling it would "
            "run none of its body"
        )
        assert ran == [] and seen == [] and rows.cache_info().misses == 0
        entry = self.by_id(log)["c1"]
        assert entry["action"]["decision"] == "deny" and len(log.entries) == 1
        assert entry["action"]["reason"] == answers["c1"]

    def test_a_returned_generator_is_on_the_log_as_allowed_and_answered_as_having_run(
        self, example, gate, log
    ):
        sent = []

        def send_invoices(month: str):
            sent.append(month)  # the side effect: it happens when the function is called
            return (row for row in sent)

        response = gate.response_with([gate.function_call("send_invoices", {"month": "11"}, "c1")])
        outs = example.audited_function_calls(
            response, {"send_invoices": send_invoices}, lambda n, a: [], log
        )
        assert outs[0]["output"] == (
            "Tool `send_invoices` ran and returned a generator, which this gate does not "
            "consume: its result is not available"
        )
        assert sent == ["11"] and "did not run" not in outs[0]["output"]
        assert self.by_id(log)["c1"]["decision"] == "PASS" and len(log.entries) == 1

    @SHAPES
    def test_a_log_that_is_none_is_the_plain_gate_and_writes_nothing(
        self, gate, log, log_path, chat
    ):
        """``audit=maybe_log`` with no log configured: decided as the plain loop decides."""
        plain = self.plain(gate, THREE, chat=chat)
        gate.EXECUTED.clear()
        answers = self.direct(gate, THREE, gate.TOOLS, gate.sql_scope_policy, chat=chat, audit=None)
        assert (answers, list(gate.EXECUTED)) == plain
        assert not any("policy error" in answer for answer in answers.values())
        assert log.entries == [] and not log_path.exists()

    @SHAPES
    def test_an_audit_log_without_a_surface_raises_and_writes_no_entry(self, gate, log, chat):
        gate.EXECUTED.clear()
        with pytest.raises(TypeError, match="audit= requires surface="):
            self.direct(gate, THREE, gate.TOOLS, gate.sql_scope_policy, chat=chat, audit=log)
        assert gate.EXECUTED == [] and log.entries == []

    @SHAPES
    def test_a_raising_policy_cannot_be_made_to_fail_open(self, example, gate, log, chat):
        """``fail_closed=False`` let the tool run when the policy raised; it is not taken."""

        def broken(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        calls = [("echo", {"text": "x"}, "c1")]
        gate.EXECUTED.clear()
        with pytest.raises(TypeError, match="unexpected keyword argument 'fail_closed'"):
            self.direct(
                gate,
                calls,
                gate.TOOLS,
                broken,
                chat=chat,
                audit=log,
                surface=SURFACES[chat],
                fail_closed=False,
            )
        audited = example.audited_tool_calls if chat else example.audited_function_calls
        shaped = (
            gate.completion_with([gate.tool_call(*c) for c in calls])
            if chat
            else gate.response_with([gate.function_call(*c) for c in calls])
        )
        with pytest.raises(TypeError, match="fail_closed"):
            audited(shaped, gate.TOOLS, broken, log, fail_closed=False)
        assert gate.EXECUTED == [] and log.entries == []
        answers, executed = example.run_calls(calls, log, policy=broken, chat=chat)
        assert executed == [] and "RECUSAL_POLICY_ERROR" in answers["c1"]
        assert self.by_id(log)["c1"]["action"]["decision"] == "deny"

    @SHAPES
    def test_a_callers_own_call_id_is_not_taken_and_each_entry_keeps_the_calls_id(
        self, example, gate, log, chat
    ):
        with pytest.raises(TypeError, match="unexpected keyword argument 'tool_use_id'"):
            self.direct(
                gate,
                THREE,
                gate.TOOLS,
                gate.sql_scope_policy,
                chat=chat,
                audit=log,
                surface=SURFACES[chat],
                tool_use_id="MINE",
            )
        assert log.entries == []
        example.run_calls(THREE, log, chat=chat)
        assert [e["action"]["tool_use_id"] for e in log.entries] == ["c1", "c2", "c3"]
