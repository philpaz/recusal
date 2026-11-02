"""Regression tests for the offline OpenAI Responses and Chat Completions gate example (#81).

The application's own loop reads function calls out of a response, runs them and answers
each one. The example judges every call before it runs, answers each exactly once by its own
id, and refuses, rather than skips, anything it does not understand.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "openai_responses_gate.py"

SELECT = "SELECT * FROM audit_log"
DELETE = "DELETE FROM audit_log"


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
    assert "openai_responses_gate.py requires Python 3.10+" in result.stderr
    assert "this is Python 3.9." in result.stderr


@pytest.mark.skipif(sys.version_info < (3, 10), reason="below 3.10 the version refusal fires first")
def test_missing_sdk_is_an_install_message_not_a_traceback():
    code = (
        "import sys, runpy; "
        "sys.modules['openai'] = None; sys.modules['openai.types'] = None; "
        "sys.modules['openai.types.chat'] = None; sys.modules['openai.types.responses'] = None; "
        f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "Install the example dependency first: openai==3.34.0" in result.stderr
    assert "Traceback" not in result.stderr.split("Install the example dependency")[-1]


def test_the_sdk_stays_out_of_the_recusal_package_and_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "openai" not in pyproject
    for path in (ROOT / "recusal").rglob("*.py"):
        assert "openai" not in path.read_text(encoding="utf-8").lower(), path


def test_pinned_version_is_explicit_in_the_example_and_docs():
    for path in (EXAMPLE, ROOT / "README.md", ROOT / "docs" / "COOKBOOK.md"):
        assert "openai==3.34.0" in path.read_text(encoding="utf-8"), path.name


def test_both_openai_lanes_run_this_file():
    for workflow in ("ci.yml", "compat.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert text.count("tests/test_openai_responses_gate.py") == 2, workflow


def test_the_works_with_table_marks_the_direct_api_tested():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    assert "| OpenAI Responses and Chat Completions, your own loop | Tested |" in table
    assert "examples/openai_responses_gate.py" in table
    assert "issues/81" not in table


@pytest.mark.skipif(sys.version_info < (3, 10), reason="the OpenAI SDK requires Python 3.10+")
class TestOpenAIResponsesGate:
    @pytest.fixture
    def ex(self):
        pytest.importorskip("openai", reason="openai is not installed")
        spec = importlib.util.spec_from_file_location("recusal_openai_responses_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.EXECUTED.clear()
        return module

    @staticmethod
    def executed_sql(ex):
        return [args.get("sql") for name, args in ex.EXECUTED if name == "run_sql"]

    @staticmethod
    def responses_outputs(ex, calls, **kwargs):
        policy = kwargs.pop("policy", ex.sql_scope_policy)
        tools = kwargs.pop("tools", ex.TOOLS)
        response = ex.response_with([ex.function_call(*c) for c in calls])
        outs = ex.gate_function_calls(response, tools, policy)
        return {o["call_id"]: o["output"] for o in outs}, outs

    @staticmethod
    def chat_outputs(ex, calls, **kwargs):
        policy = kwargs.pop("policy", ex.sql_scope_policy)
        tools = kwargs.pop("tools", ex.TOOLS)
        completion = ex.completion_with([ex.tool_call(*c) for c in calls])
        outs = ex.gate_tool_calls(completion, tools, policy)
        return {o["tool_call_id"]: o["content"] for o in outs}, outs

    # --- allowed runs, refused does not, both API shapes ---------------------------------

    @pytest.mark.parametrize("shape", ["responses", "chat"])
    def test_allowed_call_runs_and_refused_call_does_not(self, ex, shape):
        run = self.responses_outputs if shape == "responses" else self.chat_outputs
        by_id, _ = run(ex, [("run_sql", {"sql": SELECT}, "a"), ("run_sql", {"sql": DELETE}, "b")])
        assert by_id["a"] == f"recorded SQL: {SELECT}"
        assert "Recusal refused `run_sql` [FAIL]" in by_id["b"]
        assert "top-level WHERE" in by_id["b"]
        assert self.executed_sql(ex) == [SELECT]

    @pytest.mark.parametrize("shape", ["responses", "chat"])
    def test_five_calls_in_one_turn_each_answered_once_by_id(self, ex, shape):
        run = self.responses_outputs if shape == "responses" else self.chat_outputs
        by_id, outs = run(ex, ex.CALLS)
        assert [o.get("call_id") or o.get("tool_call_id") for o in outs] == [
            "call_1",
            "call_2",
            "call_3",
            "call_4",
            "call_5",
        ]
        assert by_id["call_1"] == f"recorded SQL: {SELECT}"
        assert "top-level WHERE" in by_id["call_2"]
        assert by_id["call_3"] == "Recusal refused `run_sql`: arguments are not valid JSON"
        assert by_id["call_4"] == "Recusal refused: unknown tool 'drop_db'"
        assert by_id["call_5"] == "Recusal refused `run_sql`: arguments are not a JSON object"
        assert self.executed_sql(ex) == [SELECT]

    @pytest.mark.parametrize("shape", ["responses", "chat"])
    def test_both_shapes_give_the_same_answers(self, ex, shape):
        a, _ = self.responses_outputs(ex, ex.CALLS)
        ex.EXECUTED.clear()
        b, _ = self.chat_outputs(ex, ex.CALLS)
        assert a == b

    @pytest.mark.parametrize("shape", ["responses", "chat"])
    def test_raising_policy_fails_closed_and_nothing_runs(self, ex, shape):
        def broken(tool_name, tool_input):
            raise RuntimeError("policy exploded")

        run = self.responses_outputs if shape == "responses" else self.chat_outputs
        by_id, _ = run(ex, [("run_sql", {"sql": SELECT}, "a")], policy=broken)
        assert by_id["a"] == "Recusal failed closed (policy error): policy exploded"
        assert ex.EXECUTED == []

    @pytest.mark.parametrize("shape", ["responses", "chat"])
    def test_arguments_are_judged_as_decoded_not_as_text(self, ex, shape):
        seen = []

        def recording(tool_name, tool_input):
            seen.append((tool_name, tool_input))
            return []

        run = self.responses_outputs if shape == "responses" else self.chat_outputs
        run(ex, [("echo", '{"text": "a\\u0062c"}', "a")], policy=recording)
        assert seen == [("echo", {"text": "abc"})]
        assert ex.EXECUTED == [("echo", {"text": "abc"})]

    @pytest.mark.parametrize("shape", ["responses", "chat"])
    def test_empty_arguments_decode_as_an_empty_object(self, ex, shape):
        seen = []

        def recording(tool_name, tool_input):
            seen.append(tool_input)
            return []

        run = self.responses_outputs if shape == "responses" else self.chat_outputs
        by_id, _ = run(ex, [("echo", "", "a")], policy=recording, tools={"echo": lambda: "ok"})
        assert seen == [{}]
        assert by_id["a"] == "ok"

    # --- default-deny on what the adapter does not handle ---------------------------------

    @pytest.mark.parametrize(
        "item",
        [
            {"type": "custom_tool_call", "id": "ct", "call_id": "c9", "name": "x", "input": "raw"},
            {
                "type": "code_interpreter_call",
                "id": "ci",
                "container_id": "c",
                "code": "1",
                "outputs": None,
                "status": "completed",
            },
            {
                "type": "web_search_call",
                "id": "ws",
                "status": "completed",
                "action": {"type": "search"},
            },
            {"type": "mcp_call", "id": "mc", "name": "t", "server_label": "s", "arguments": "{}"},
        ],
        ids=lambda item: item["type"],
    )
    def test_unhandled_output_item_raises_naming_the_type(self, ex, item):
        response = ex.response_with([ex.function_call("run_sql", {"sql": SELECT}, "a"), item])
        with pytest.raises(TypeError, match=item["type"]):
            ex.gate_function_calls(response, ex.TOOLS, ex.sql_scope_policy)
        # the SELECT before it ran, so the raise surfaces mid-turn; nothing after it runs
        assert ex.EXECUTED == [("run_sql", {"sql": SELECT})]

    def test_a_plain_message_and_reasoning_pass_a_function_call_is_judged(self, ex):
        response = ex.response_with(
            [
                {
                    "type": "message",
                    "id": "m",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                },
                {"type": "reasoning", "id": "r", "summary": []},
                ex.function_call("run_sql", {"sql": DELETE}, "a"),
            ]
        )
        outs = ex.gate_function_calls(response, ex.TOOLS, ex.sql_scope_policy)
        assert [o["call_id"] for o in outs] == ["a"]
        assert "top-level WHERE" in outs[0]["output"]
        assert ex.EXECUTED == []

    def test_chat_custom_tool_call_raises_naming_the_type(self, ex):
        completion = ex.completion_with(
            [
                ex.tool_call("run_sql", {"sql": SELECT}, "a"),
                {"id": "b", "type": "custom", "custom": {"name": "x", "input": "raw"}},
            ]
        )
        with pytest.raises(TypeError, match="custom"):
            ex.gate_tool_calls(completion, ex.TOOLS, ex.sql_scope_policy)

    def test_response_with_no_calls_answers_nothing(self, ex):
        response = ex.response_with([])
        assert ex.gate_function_calls(response, ex.TOOLS, ex.sql_scope_policy) == []
        completion = ex.completion_with([])
        assert ex.gate_tool_calls(completion, ex.TOOLS, ex.sql_scope_policy) == []

    # --- call ids: the SDK does not require them, the adapter does ------------------------

    def test_sdk_answer_types_do_not_require_call_id_so_the_adapter_must(self, ex):
        from openai.types.chat import ChatCompletionToolMessageParam
        from openai.types.responses.response_input_param import FunctionCallOutput

        assert "call_id" not in FunctionCallOutput.__required_keys__
        assert "tool_call_id" not in ChatCompletionToolMessageParam.__required_keys__

    def test_function_call_without_call_id_is_refused_before_running(self, ex):
        item = ex.function_call("run_sql", {"sql": SELECT}, "a")
        item["call_id"] = ""
        response = ex.response_with([item])
        with pytest.raises(ValueError, match="no call id"):
            ex.gate_function_calls(response, ex.TOOLS, ex.sql_scope_policy)
        assert ex.EXECUTED == []

    def test_chat_tool_call_without_id_is_refused_before_running(self, ex):
        call = ex.tool_call("run_sql", {"sql": SELECT}, "")
        completion = ex.completion_with([call])
        with pytest.raises(ValueError, match="no call id"):
            ex.gate_tool_calls(completion, ex.TOOLS, ex.sql_scope_policy)
        assert ex.EXECUTED == []

    def test_duplicate_call_ids_are_refused(self, ex):
        response = ex.response_with(
            [
                ex.function_call("echo", {"text": "1"}, "same"),
                ex.function_call("echo", {"text": "2"}, "same"),
            ]
        )
        with pytest.raises(ValueError, match="duplicate call ids"):
            ex.gate_function_calls(response, ex.TOOLS, ex.sql_scope_policy)

    def test_an_answer_for_a_call_that_was_not_made_is_refused(self, ex):
        with pytest.raises(ValueError, match="calls \\['a'\\] but answers \\['a', 'ghost'\\]"):
            ex._check_answered_once(["a"], ["a", "ghost"])
        with pytest.raises(ValueError, match="answers \\['a'\\]"):
            ex._check_answered_once(["a", "b"], ["a"])

    # --- the answers are what the SDK will accept -----------------------------------------

    def test_outputs_validate_against_the_sdk_input_types(self, ex):
        from openai.types.chat import ChatCompletionToolMessageParam
        from openai.types.responses.response_input_param import FunctionCallOutput
        from pydantic import TypeAdapter

        _, outs = self.responses_outputs(ex, ex.CALLS)
        for out in outs:
            TypeAdapter(FunctionCallOutput).validate_python(out, strict=True)
            assert set(out) == {"type", "call_id", "output"}
        ex.EXECUTED.clear()
        _, outs = self.chat_outputs(ex, ex.CALLS)
        for out in outs:
            TypeAdapter(ChatCompletionToolMessageParam).validate_python(out, strict=True)
            assert set(out) == {"role", "tool_call_id", "content"}

    def test_fixtures_are_real_sdk_objects(self, ex):
        from openai.types.chat import ChatCompletion
        from openai.types.responses import Response, ResponseFunctionToolCall

        response = ex.response_with([ex.function_call("echo", {"text": "x"}, "a")])
        assert isinstance(response, Response)
        assert isinstance(response.output[0], ResponseFunctionToolCall)
        completion = ex.completion_with([ex.tool_call("echo", {"text": "x"}, "a")])
        assert isinstance(completion, ChatCompletion)
        assert completion.choices[0].message.tool_calls[0].type == "function"

    # --- negative control --------------------------------------------------------------

    def test_negative_control_naive_dispatch_runs_the_delete(self, ex):
        response = ex.response_with(
            [
                ex.function_call("run_sql", {"sql": SELECT}, "a"),
                ex.function_call("run_sql", {"sql": DELETE}, "b"),
            ]
        )
        outs = ex.dispatch_ungated(response, ex.TOOLS)
        assert [o["call_id"] for o in outs] == ["a", "b"]
        assert self.executed_sql(ex) == [SELECT, DELETE]

    def test_a_policy_that_names_no_tool_lets_the_call_through(self, ex):
        # a policy covers only the tools it names; this is the documented boundary
        by_id, _ = self.responses_outputs(ex, [("echo", {"text": "DELETE FROM audit_log"}, "a")])
        assert by_id["a"] == "DELETE FROM audit_log"

    # --- the script ----------------------------------------------------------------------

    def test_main_runs_end_to_end(self, ex):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert out.count("call_2 -> Recusal refused `run_sql` [FAIL]") == 2
        assert out.count("executed: ['SELECT * FROM audit_log']") == 2
        assert "UNGATED executed: ['SELECT * FROM audit_log', 'DELETE FROM audit_log']" in out
        assert "custom_tool_call: Recusal refused output item of type 'custom_tool_call'" in out
        assert json.dumps({"sql": DELETE}) not in out
