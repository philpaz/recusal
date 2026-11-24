"""Regression tests for the offline OpenAI Agents SDK MCP audit-trail example.

Every call a gate of the audited agent sees lands on a hash-chained log exactly once,
under its seam's surface: a local MCP server's ``call_tool`` (no call id exists at that
seam, so entries are matched by tool and fingerprint), a hosted MCP approval request (the
request id) and a function tool (the tool call id). The audited agent decides exactly as
the plain ``gated_agent`` does.
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
EXAMPLE = ROOT / "examples" / "openai_agents_mcp_audit.py"
LOCAL = "openai_agents.mcp.call_tool"
HOSTED = "openai_agents.hosted_mcp.approval"
FUNCTION = "openai_agents.tool_input_guardrail"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"


def _adapter():
    """The package module the gate's code lives in. Every test that calls this has loaded
    the example first, which puts the in-tree adapter on the path, so this is the code
    under review and not an installed copy."""
    import importlib
    from pathlib import Path

    module = importlib.import_module("recusal_openai_agents_mcp._gate")
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
    assert "openai_agents_mcp_audit.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


def test_pinned_versions_are_explicit_in_the_example():
    text = EXAMPLE.read_text(encoding="utf-8")
    for pin in ("openai-agents==0.24.0", "openai==3.49.0"):
        assert pin in text


def test_both_openai_lanes_run_this_file():
    # no new jobs: the two OpenAI jobs and the weekly lane run it
    for workflow in ("ci.yml", "compat.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert text.count("tests/test_openai_agents_mcp_audit.py") == 2, workflow


def test_the_works_with_table_marks_the_audit_trail_tested():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    assert (
        "| OpenAI Agents SDK audit trail, MCP tools | Tested | "
        "[`examples/openai_agents_mcp_audit.py`]" in table
    )


def _sha(args):
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _text(output):
    if isinstance(output, list):
        return "".join(item.get("text", "") for item in output if isinstance(item, dict))
    return str(output)


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

    def error_level(tool_name, tool_input):
        from recusal import Finding

        return [Finding.fail("hard", severity="ERROR", message="retry this")]

    return {
        "sql-scope-three": (None, THREE),
        "sql-scope-unicode": (None, [("echo", {"text": "héllo ☃"}, "c1")]),
        "raising": (raising, THREE),
        "silent": (silent, THREE),
        "ambiguous": (ambiguous, THREE),
        "error-level": (error_level, THREE),
    }


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the OpenAI Agents SDK requires Python 3.10+"
)
class TestOpenAIAgentsMcpAudit:
    @pytest.fixture
    def example(self):
        pytest.importorskip("agents.testing", reason="openai-agents is not installed")
        pytest.importorskip("mcp", reason="the MCP SDK is not installed")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location(
            "recusal_openai_agents_mcp_audit_example", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def gate(self, example):
        """The plain gate example, the SAME module object the audited one imports from."""
        return importlib.import_module("examples.openai_agents_mcp_gate")

    @pytest.fixture
    def log_path(self, tmp_path):
        return tmp_path / "openai-agents-mcp-audit.jsonl"

    @pytest.fixture
    def log(self, example, log_path):
        return example.AuditLog(path=str(log_path))

    @staticmethod
    def hosted(calls):
        return [(name, args, f"r{i}") for i, (name, args, _) in enumerate(calls, 1)]

    # --- local MCP servers ----------------------------------------------------------------

    def test_every_local_call_is_recorded_once_with_surface_and_server(self, example, log):
        outputs, called = example.run_calls(THREE, log, actor="demo-agent")
        assert sorted(tool for _, tool, _ in called) == ["echo", "run_sql"]
        assert len(log.entries) == 3
        by_input = {e["action"]["input_sha256"]: e for e in log.entries}
        assert set(by_input) == {_sha(args) for _, args, _ in THREE}
        for entry in log.entries:
            assert entry["action"]["surface"] == LOCAL
            assert entry["actor"] == "demo-agent"
            assert entry["action"]["control"]["mcp_server"] == "db"
            # the SDK hands call_tool no call id, and the record does not invent one
            assert "tool_use_id" not in entry["action"]
        refused = by_input[_sha({"sql": DELETE})]
        assert refused["action"]["decision"] == "deny" and refused["decision"] == "FAIL"
        assert refused["action"]["tool"] == "run_sql"
        assert refused["action"]["reason"] == _text(outputs["c2"])
        assert by_input[_sha({"sql": SELECT})]["action"]["decision"] == "defer"

    @pytest.mark.parametrize("case", sorted(_corpus()))
    def test_local_decisions_match_the_plain_gate(self, example, gate, log, case):
        policy, calls = _corpus()[case]
        kwargs = {} if policy is None else {"policy": policy}
        plain_out, plain_called = gate.run_calls(calls, **kwargs)
        out, called = example.run_calls(calls, log, **kwargs)
        assert {k: _text(v) for k, v in out.items()} == {k: _text(v) for k, v in plain_out.items()}
        assert sorted(map(repr, called)) == sorted(map(repr, plain_called))
        assert len(log.entries) == len(calls)

    def test_the_record_binds_what_was_sent_to_the_server_whatever_the_policy_did(
        self, example, log
    ):
        def mutating(tool_name, tool_input):
            tool_input["text"] = "REWRITTEN BY THE POLICY"
            return []

        _, called = example.run_calls(
            [("echo", {"text": "as proposed"}, "c1")], log, policy=mutating
        )
        assert called == [("db", "echo", {"text": "as proposed"})]
        assert log.entries[0]["action"]["input_sha256"] == _sha(called[0][2])

    def test_the_record_binds_the_arguments_completed_with_the_listed_defaults(
        self, example, gate, log
    ):
        """What is sent is the call completed with the defaults the server lists, and that
        is what the entry fingerprints: not the ``{}`` the model wrote."""
        server = _purge_server(gate)
        _, called = example.run_calls(
            [("purge", {}, "c1")], log, policy=lambda *_: [], server=server
        )
        completed = {"path": "/", "recursive": True, "dry_run": False}
        assert called == [("files", "purge", completed)]
        (entry,) = log.entries
        assert entry["action"]["decision"] == "defer"
        assert entry["action"]["input_sha256"] == _sha(completed) != _sha({})

    def test_a_call_without_arguments_is_recorded_as_the_empty_mapping(self, example, gate, log):
        record = example.recording(log)("mcp_server", gate.InProcessMCPServer())
        sent, reason = _adapter()._judged_call("echo", None, {}, lambda *_: [], **record)
        assert sent == {} and reason is None
        assert log.entries[0]["action"]["input_sha256"] == _sha({})

    def test_non_mapping_arguments_are_one_recorded_deny(self, example, gate, log):
        record = example.recording(log)("mcp_server", gate.InProcessMCPServer())
        sent, reason = _adapter()._judged_call(
            "run_sql", ["DELETE"], {}, gate.sql_scope_policy, **record
        )
        assert sent is None and "arguments are not a mapping" in reason
        entry = log.entries[0]
        assert entry["action"]["decision"] == "deny"
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["input_sha256"] == _sha({"input": ["DELETE"]})
        assert _adapter()._judged_call("run_sql", ["DELETE"], {}, gate.sql_scope_policy) == (
            None,
            "Recusal refused `run_sql`: arguments are not a mapping",
        )
        assert len(log.entries) == 1

    # --- refused before the policy runs: always exactly one entry --------------------------

    @pytest.mark.parametrize(
        "arguments, plain",
        [
            ({"dry_run": "MARKER"}, "argument `dry_run` is of JSON type string where"),
            ({"path": "/data", "recursive": 1}, "argument `recursive` is of JSON type integer"),
        ],
        ids=["string-for-boolean", "integer-for-boolean"],
    )
    def test_an_argument_of_another_type_is_one_recorded_deny(
        self, example, gate, log, log_path, arguments, plain
    ):
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        outputs, called = example.run_calls(
            [("purge", arguments, "c1")], log, policy=policy, server=_purge_server(gate)
        )
        assert called == [] and seen == []
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny"
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert plain in entry["action"]["reason"] == _text(outputs["c1"])
        # bound to the arguments as the SDK handed them, and never their text
        assert entry["action"]["input_sha256"] == _sha(arguments)
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    def test_a_tool_the_server_does_not_list_is_one_recorded_deny(self, example, gate, log):
        server = example.audited_agent(
            gate.Agent(name="a", mcp_servers=[gate.InProcessMCPServer()]),
            lambda *_: [],
            log,
        ).mcp_servers[0]
        gate.CALLED.clear()
        result = asyncio.run(server.call_tool("drop_all", {"sql": SELECT}))
        assert result.model_dump(by_alias=True)["isError"] is True
        assert "the server lists no tool of this name" in result.content[0].text
        assert gate.CALLED == []
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny" and entry["action"]["tool"] == "drop_all"

    @pytest.mark.parametrize("kind", ["too-deep", "non-finite", "not-data"])
    def test_arguments_that_are_not_json_data_are_one_recorded_deny(self, example, gate, log, kind):
        """The review's F3: arguments nested about 900 deep raised inside the gate before
        ``decide`` ran, so the refusal the model read was on no log."""
        arguments: dict = {"sql": SELECT}
        if kind == "too-deep":
            nested: list = []
            for _ in range(5000):
                nested = [nested]
            arguments["meta"] = nested
        elif kind == "non-finite":
            arguments["limit"] = float("nan")
        else:
            arguments["when"] = object()
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        record = example.recording(log)("mcp_server", gate.InProcessMCPServer())
        sent, reason = _adapter()._judged_call("run_sql", arguments, {}, policy, **record)
        assert sent is None and seen == []
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["reason"] == reason
        plain = _adapter()._judged_call("run_sql", arguments, {}, policy)
        assert plain[0] is None and plain[1].startswith("Recusal refused `run_sql`: arguments ")
        assert seen == [] and len(log.entries) == 1
        # the second review: such an entry carried no fingerprint, and the reason on it was
        # the kernel's "cannot be fingerprinted", not the gate's own. It is the gate's own
        # refusal, bound to the arguments written as text.
        assert reason == plain[1].replace("`run_sql`:", "`run_sql` [FAIL]:")
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        written = _adapter()._rendered(arguments)
        assert entry["action"]["input_sha256"] == _sha({"input": written})
        if kind == "non-finite":
            assert written == '{"sql":"SELECT * FROM audit_log","limit":nan}'
        elif kind == "not-data":
            assert written == '{"sql":"SELECT * FROM audit_log","when":<object>}'
        else:
            assert written.endswith("[" * 64 + "..." + "]" * 64 + "}")

    def test_a_number_too_large_for_a_float_is_one_recorded_deny_in_a_real_run(
        self, example, gate, log
    ):
        """The second review's probe: ``1e400`` is ``inf`` once the SDK has parsed it."""
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        outputs, called = example.run_calls(
            [("echo", '{"text": "x", "n": 1e400}', "c1")], log, policy=policy
        )
        assert called == [] and seen == []
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny"
        assert (
            entry["action"]["reason"]
            == _text(outputs["c1"])
            == (
                "Recusal refused `echo` [FAIL]: arguments are not JSON data (non-finite numbers "
                "are not JSON data)"
            )
        )
        assert entry["action"]["input_sha256"] == _sha({"input": '{"text":"x","n":inf}'})

    def test_a_cyclic_argument_is_written_without_end_and_refused(self, example, gate, log):
        arguments: dict = {"sql": SELECT}
        arguments["self"] = arguments
        record = example.recording(log)("mcp_server", gate.InProcessMCPServer())
        sent, reason = _adapter()._judged_call("run_sql", arguments, {}, lambda *_: [], **record)
        assert sent is None and "arguments are nested too deeply" in reason
        (entry,) = log.entries
        assert entry["action"]["input_sha256"] == _sha(
            {"input": '{"sql":"SELECT * FROM audit_log","self":...}'}
        )

    def test_the_record_binds_what_was_judged_when_a_default_cannot_be_sent(
        self, example, gate, log
    ):
        """The one case where what is judged is not what is sent: a listed default its own
        property schema does not allow. The server runs with it, so it is judged and it is
        what the entry fingerprints; it is not sent."""
        schema = {
            "type": "object",
            "properties": {
                "q": {"type": "string", "default": None},
                "limit": {"type": "integer", "default": 10},
            },
        }
        _, called = example.run_calls(
            [("purge", {}, "c1")], log, policy=lambda *_: [], server=_purge_server(gate, schema)
        )
        assert called == [("files", "purge", {"limit": 10})]
        (entry,) = log.entries
        assert entry["action"]["decision"] == "defer"
        assert entry["action"]["input_sha256"] == _sha({"q": None, "limit": 10})

    @pytest.mark.parametrize(
        "schema, arguments, plain",
        [
            (
                {"properties": {"dry_run": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}}},
                {"dry_run": "MARKER"},
                "argument `dry_run` is of JSON type string where the server's schema declares "
                "boolean or null",
            ),
            (
                {"properties": {"flags": {"additionalProperties": {"type": "boolean"}}}},
                {"flags": {"MARKER": "MARKER"}},
                "argument `flags.*` is of JSON type string",
            ),
            (
                {"properties": {"a": {"$ref": "https://example.invalid/MARKER.json"}}},
                {"a": "MARKER"},
                "the server's listed schema for this tool has a `$ref` that does not resolve",
            ),
            (
                {"properties": {"scope": {"type": ["string", "null"]}}},
                {"scope": '["MARKER"]'},
                "argument `scope` is a string that holds JSON text",
            ),
            (
                {
                    "properties": {
                        "note": {"type": "string"},
                        "recursive": {
                            "anyOf": [
                                {"type": "boolean", "default": True},
                                {"type": "string", "default": "no"},
                            ]
                        },
                    }
                },
                {"note": "MARKER"},
                "argument `recursive` is left out, and the alternatives the server's schema "
                "lists for it list different defaults",
            ),
            (
                {"properties": {"a": {"type": "string", "x-shape": {"default": "MARKER"}}}},
                {"a": "MARKER"},
                "the server's listed schema for this tool lists a default under a keyword the "
                "gate does not read",
            ),
        ],
        ids=[
            "an-alternative",
            "a-member-the-schema-does-not-name",
            "a-schema-it-cannot-check",
            "json-text-a-server-may-read",
            "alternatives-with-different-defaults",
            "a-default-under-an-unknown-keyword",
        ],
    )
    def test_arguments_the_whole_schema_does_not_allow_are_one_recorded_deny(
        self, example, gate, log, log_path, schema, arguments, plain
    ):
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        outputs, called = example.run_calls(
            [("purge", arguments, "c1")], log, policy=policy, server=_purge_server(gate, schema)
        )
        assert called == [] and seen == []
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny"
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert plain in entry["action"]["reason"] == _text(outputs["c1"])
        assert entry["action"]["input_sha256"] == _sha(arguments)
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    def test_arguments_too_costly_to_check_are_one_recorded_deny(
        self, example, gate, log, log_path, monkeypatch
    ):
        schema_module = importlib.import_module("recusal_openai_agents_mcp._schema")
        monkeypatch.setattr(schema_module, "_STEPS", 3)
        monkeypatch.setattr(schema_module, "_STEPS_PER_VALUE", 0)
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        arguments = {"path": "MARKER", "recursive": False, "dry_run": True}
        outputs, called = example.run_calls(
            [("purge", arguments, "c1")], log, policy=policy, server=_purge_server(gate)
        )
        assert called == [] and seen == []
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny"
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert (
            entry["action"]["reason"]
            == _text(outputs["c1"])
            == (
                "Recusal refused `purge` [FAIL]: the arguments are too costly to check against the "
                "server's listed schema for this tool: the check costs more than the 3 steps and "
                "0 more for every value in the arguments that the gate allows a call. That is "
                "what the schema costs for each value here, which raising the gate's ceiling on "
                "a call (`step_ceiling=`) does not change."
            )
        )
        assert entry["action"]["input_sha256"] == _sha(arguments)
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    def test_arguments_over_the_size_ceiling_are_one_recorded_deny_that_reads_none_of_them(
        self, example, gate, log, log_path
    ):
        """The entry is bound to a statement of the size and holds no fingerprint of the
        arguments: writing one would read all of them, which the ceiling is there to
        prevent."""
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        arguments = {"path": "MARKER" * 200_000, "recursive": False, "dry_run": True}
        outputs, called = example.run_calls(
            [("purge", arguments, "c1")], log, policy=policy, server=_purge_server(gate)
        )
        assert called == [] and seen == []
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny"
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert (
            entry["action"]["reason"]
            == _text(outputs["c1"])
            == (
                "Recusal refused `purge` [FAIL]: the arguments are more than 1,000,000 "
                "characters as JSON text, the ceiling the gate sets on the size of one "
                "call's arguments so that the time a call takes does not grow with what is "
                "sent; the gate has not read them. Send less in one call. Where calls of this "
                "size are expected, the application can raise the ceiling: `size_ceiling=` on "
                "`gate_mcp_server`, `gated_agent` or `audited_agent`."
            )
        )
        assert entry["action"]["input_sha256"] == _sha(
            {"input": "<arguments of more than 1,000,000 characters as JSON text>"}
        )
        assert entry["action"]["input_sha256"] != _sha(arguments)
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    def test_a_long_integer_is_one_recorded_deny_that_writes_none_of_it(
        self, example, gate, log, log_path
    ):
        """The eighth review. The fingerprint of the arguments would write the integer
        out, which is quadratic in its digits where the interpreter sets no limit on
        them, and under the default limit cannot be done at all: the entry is bound to
        a statement, as for the size. With the limit switched off where this Python
        has one, which is the interpreter the review found it on."""
        seen = []

        def policy(name, args):
            seen.append(name)
            return []

        gate.CALLED.clear()
        has_limit = hasattr(sys, "set_int_max_str_digits")
        before = sys.get_int_max_str_digits() if has_limit else None
        if has_limit:
            sys.set_int_max_str_digits(0)
        try:
            server = _purge_server(gate)
            gate.gate_mcp_server(server, policy, example.recording(log))
            arguments = {"path": "/tmp", "recursive": False, "count": 1 << 3_000_000}
            result = asyncio.run(server.call_tool("purge", arguments))
        finally:
            if has_limit:
                sys.set_int_max_str_digits(before)
        assert result.model_dump(by_alias=True).get("isError") is True
        assert seen == [] and gate.CALLED == []
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny"
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["reason"] == result.content[0].text
        assert entry["action"]["reason"].startswith(
            "Recusal refused `purge` [FAIL]: the arguments hold an integer of more than 4,300 "
            "digits, the most an integer in one call's arguments may have"
        )
        assert entry["action"]["input_sha256"] == _sha(
            {"input": "<arguments that hold an integer of more than 4,300 digits>"}
        )

    def test_a_call_the_gate_could_not_judge_is_one_recorded_deny(
        self, example, gate, log, monkeypatch
    ):
        """Whatever leaves the gate's own reading is a refusal, and it is on the record,
        bound to a statement: the arguments are not read again by a gate that has just
        failed to read them."""
        adapter = _adapter()

        def out_of_stack(*args):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(adapter, "_measure", out_of_stack)
        record = example.recording(log)("mcp_server", gate.InProcessMCPServer())
        sent, reason = adapter._judged_call("run_sql", {"sql": SELECT}, {}, lambda *_: [], **record)
        assert sent is None and reason == (
            "Recusal refused `run_sql` [FAIL]: the gate could not judge this call within what "
            "is left of the interpreter's recursion limit where it was called"
        )
        (entry,) = log.entries
        assert entry["action"]["decision"] == "deny" and entry["action"]["reason"] == reason
        assert entry["action"]["input_sha256"] == _sha(
            {"input": "<arguments the gate could not judge>"}
        )

    def test_two_servers_are_told_apart_on_one_log(self, example, gate, log):
        first, second = gate.InProcessMCPServer(name="db"), gate.InProcessMCPServer(name="crm")
        record = example.recording(log)
        policy = gate.sql_scope_policy
        gate.gate_mcp_server(first, policy, record)
        gate.gate_mcp_server(second, policy, record)

        async def call_both():
            await first.call_tool("echo", {"text": "x"})
            await second.call_tool("echo", {"text": "x"})

        asyncio.run(call_both())
        assert [e["action"]["control"]["mcp_server"] for e in log.entries] == ["db", "crm"]

    def test_a_server_carries_one_recording(self, example, gate, log, tmp_path):
        server = gate.InProcessMCPServer()
        policy = gate.sql_scope_policy
        record = example.recording(log)
        gate.gate_mcp_server(server, policy, record)
        assert gate.gate_mcp_server(server, policy, record) is server  # same recording: a no-op
        other = example.recording(example.AuditLog(path=str(tmp_path / "other.jsonl")))
        with pytest.raises(ValueError, match="already gated with a different recording"):
            gate.gate_mcp_server(server, policy, other)
        with pytest.raises(ValueError, match="already gated with a different recording"):
            gate.gate_mcp_server(server, policy)  # asking for no record is a difference too

    def test_unwritable_log_refuses_every_local_call(self, example, log, log_path):
        example.run_calls([("echo", {"text": "warm-up"}, "w")], log)
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
            outputs, called = example.run_calls(THREE, log)
        finally:
            log_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        assert called == []
        for output in outputs.values():
            assert "audit log unavailable" in _text(output)
        assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1

    def test_failing_sink_refuses_local_and_hosted_calls(self, example, log_path):
        class Offline:
            def write(self, entry):
                raise OSError("anchor store unreachable")

        log = example.AuditLog(path=str(log_path), sinks=[Offline()])
        outputs, called = example.run_calls(THREE, log)
        assert called == []
        assert all("RECUSAL_AUDIT_UNAVAILABLE" in _text(o) for o in outputs.values())
        assert not any("anchor store unreachable" in _text(o) for o in outputs.values())
        responses = example.run_hosted(self.hosted(THREE), log)
        assert all(r["approve"] is False for r in responses.values())
        assert all("RECUSAL_AUDIT_UNAVAILABLE" in r["reason"] for r in responses.values())

    def test_a_recording_that_raises_refuses_local_and_hosted_calls(self, example, gate):
        def broken(seam, context):
            raise RuntimeError("RECORDING_SECRET_SENTINEL")

        failed = "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        server = gate.gate_mcp_server(gate.InProcessMCPServer(), lambda *_: [], broken)
        gate.CALLED.clear()
        result = asyncio.run(server.call_tool("echo", {"text": "x"}))
        assert gate.CALLED == [] and result.content[0].text == failed
        approval = gate.recusal_approval(lambda *_: [], recording=broken)

        class Request:
            data = gate.approval_request("echo", {"text": "x"}, "r1")

        assert asyncio.run(approval(Request())) == {"approve": False, "reason": failed}

    # --- hosted MCP approvals --------------------------------------------------------------

    def test_every_hosted_request_is_recorded_once_under_its_request_id(self, example, log):
        responses = example.run_hosted(self.hosted(THREE), log, actor="demo-agent")
        assert {k: v["approve"] for k, v in responses.items()} == {
            "r1": True,
            "r2": False,
            "r3": True,
        }
        entries = {e["action"]["tool_use_id"]: e for e in log.entries}
        assert set(entries) == {"r1", "r2", "r3"} and len(log.entries) == 3
        for (_, args, _), request_id in zip(THREE, ("r1", "r2", "r3")):
            entry = entries[request_id]
            assert entry["action"]["surface"] == HOSTED
            assert entry["actor"] == "demo-agent"
            assert entry["action"]["control"]["mcp_server"] == "db"
            assert entry["action"]["input_sha256"] == _sha(args)
        assert entries["r2"]["action"]["decision"] == "deny"
        assert entries["r2"]["action"]["reason"] == responses["r2"]["reason"]

    @pytest.mark.parametrize("case", sorted(_corpus()))
    def test_hosted_decisions_match_the_plain_gate(self, example, gate, log, case):
        policy, calls = _corpus()[case]
        kwargs = {} if policy is None else {"policy": policy}
        requests = self.hosted(calls)
        assert example.run_hosted(requests, log, **kwargs) == gate.run_hosted(requests, **kwargs)
        assert len(log.entries) == len(calls)

    @pytest.mark.parametrize(
        "arguments, detail, proposal",
        [
            ('{"sql": "MARKER', "arguments are not valid JSON", {"input": '{"sql": "MARKER'}),
            ('["MARKER"]', "arguments are not a JSON object", {"input": ["MARKER"]}),
            (
                '{"sql": "MARKER", "limit": NaN}',
                "arguments hold a non-finite number, which is not JSON data",
                {"input": '{"sql": "MARKER", "limit": NaN}'},
            ),
        ],
        ids=["bad-json", "not-object", "non-finite"],
    )
    def test_a_hosted_request_refused_before_judgment_is_one_recorded_deny(
        self, example, log, log_path, arguments, detail, proposal
    ):
        responses = example.run_hosted([("run_sql", arguments, "r1")], log)
        assert responses["r1"]["approve"] is False and detail in responses["r1"]["reason"]
        entry = log.entries[0]
        assert entry["action"]["decision"] == "deny"
        assert entry["action"]["reason"] == responses["r1"]["reason"]
        assert [f["check"] for f in entry["failures"]] == ["recusal_unjudgeable_input"]
        assert entry["action"]["input_sha256"] == _sha(proposal)
        assert "MARKER" not in log_path.read_text(encoding="utf-8")

    def test_the_entry_is_recusals_decision_not_the_tools_own_callback(self, example, gate, log):
        """Recusal allows the call and the tool's own approval callback rejects it: the entry
        records what Recusal decided, as the example says, and the response is the callback's."""
        tool = gate.HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "db",
                "server_url": "https://example.invalid/mcp",
                "require_approval": "always",
            },
            on_approval_request=lambda request: {"approve": False, "reason": "a person said no"},
        )
        responses = example.run_hosted([("echo", {"text": "x"}, "r1")], log, tool=tool)
        assert responses["r1"]["approve"] is False
        assert responses["r1"]["reason"] == "a person said no"
        assert log.entries[0]["action"]["decision"] == "defer"

    # --- one log, three seams ---------------------------------------------------------------

    def test_function_tools_of_the_same_agent_are_recorded_under_their_own_surface(
        self, example, gate, log
    ):
        base = importlib.import_module("examples.openai_agents_gate")
        model = gate.scripted_model([("echo", {"text": "hello"}, "c1")])
        agent = example.audited_agent(
            gate.Agent(name="a", model=model, tools=[base.echo]), gate.sql_scope_policy, log
        )
        asyncio.run(gate.run_agent(agent))
        assert [e["action"]["surface"] for e in log.entries] == [FUNCTION]
        assert log.entries[0]["action"]["tool_use_id"] == "c1"
        assert gate.is_gated(agent)

    def test_caller_control_is_recorded_but_cannot_forge_the_server_or_the_version(
        self, example, log
    ):
        control = {"policy_id": "sql-scope", "mcp_server": "forged", "recusal_version": "9.9.9"}
        example.run_calls([("echo", {"text": "x"}, "c1")], log, control=control)
        recorded = log.entries[0]["action"]["control"]
        assert recorded["policy_id"] == "sql-scope"
        assert recorded["mcp_server"] == "db"
        assert recorded["recusal_version"] != "9.9.9"

    def test_no_tool_argument_text_appears_in_the_log(self, example, log, log_path):
        secret = "DELETE FROM customers WHERE ssn = 'MARKER-7f3e9'"
        calls = [("run_sql", {"sql": secret}, "s1"), ("echo", {"text": "MARKER-echo"}, "s2")]
        example.run_calls(calls, log)
        example.run_hosted(self.hosted(calls), log)
        text = log_path.read_text(encoding="utf-8")
        assert "MARKER" not in text and "customers" not in text
        assert len(log.entries) == 4

    # --- verification, tamper detection, negative controls ---------------------------------

    def test_log_verifies_intact_and_editing_any_entry_breaks_it(self, example, log, log_path):
        example.run_calls(THREE, log)
        example.run_hosted(self.hosted(THREE), log)
        code, text = example.verify_with_cli(str(log_path))
        assert code == 0, text
        assert "intact - 6 entries" in text
        for seq in range(6):
            tampered = log_path.with_name(f"tampered-{seq}.jsonl")
            tampered.write_bytes(log_path.read_bytes())
            flipped = "allow" if log.entries[seq]["action"]["decision"] != "allow" else "deny"
            example.tamper(str(tampered), seq, flipped)
            code, text = example.verify_with_cli(str(tampered))
            assert code == 1, (seq, text)
            assert "BROKEN" in text

    def test_negative_control_plain_gate_writes_no_entries(self, gate, log, log_path):
        outputs, _ = gate.run_calls(THREE)
        assert _text(outputs["c2"]).startswith("Recusal refused")
        gate.run_hosted(self.hosted(THREE))
        assert log.entries == []
        assert not log_path.exists()

    def test_negative_control_ungated_agent_records_nothing_and_runs_the_delete(self, gate, log):
        _, called = gate.run_calls(THREE, gated=False)
        assert ("db", "run_sql", {"sql": DELETE}) in called
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
        assert out.count(LOCAL) == 3 and out.count(HOSTED) == 3
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


#: The input schema a server lists for ``purge``: defaults the server fills when the call
#: leaves a parameter out, and types it converts a value to.
PURGE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "default": "/"},
        "recursive": {"type": "boolean", "default": True},
        "dry_run": {"type": "boolean", "default": False},
    },
}


def _purge_server(gate, schema=None, name="files"):
    """An in-process server that lists one tool, ``purge``, under ``schema``, counts its
    listings and records what it is sent."""
    import copy

    from mcp.types import Tool

    listed = PURGE_SCHEMA if schema is None else schema

    class PurgeServer(gate.InProcessMCPServer):
        listings = 0

        async def list_tools(self, run_context=None, agent=None):
            self.listings += 1
            return [
                Tool(
                    name="purge",
                    description="Delete files under a path.",
                    inputSchema=copy.deepcopy(listed),
                )
            ]

    return PurgeServer(name=name, tools=("purge",))
