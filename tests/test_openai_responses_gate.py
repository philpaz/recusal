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
        assert (
            by_id["a"]
            == "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (RuntimeError)"
        )
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
        # the whole turn is checked before anything is dispatched, so the SELECT listed
        # before the unhandled item did not run either: a refused turn has run nothing
        assert ex.EXECUTED == []

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

    # --- alternative choices: one chosen message per dispatch (adapters review) ------------

    @staticmethod
    def charge_tools(ran):
        def charge(amount: int) -> str:
            ran.append(amount)
            return "recorded"

        return {"charge": charge}

    @staticmethod
    def two_choices(ex):
        return ex.completion_with(
            [ex.tool_call("charge", {"amount": 5}, "c1")],
            [ex.tool_call("charge", {"amount": 7}, "c2")],
        )

    def test_several_choices_without_a_selection_are_refused_before_anything_runs(self, ex):
        ran, seen = [], []

        def recording(name, args):
            seen.append(dict(args))
            return []

        with pytest.raises(ValueError, match="2 choices; pass choice="):
            ex.gate_tool_calls(self.two_choices(ex), self.charge_tools(ran), recording)
        assert ran == [] and seen == []

    @pytest.mark.parametrize("choice, amount, call_id", [(0, 5, "c1"), (1, 7, "c2")])
    def test_only_the_chosen_message_is_judged_and_run(self, ex, choice, amount, call_id):
        ran, seen = [], []

        def recording(name, args):
            seen.append(dict(args))
            return []

        outs = ex.gate_tool_calls(
            self.two_choices(ex), self.charge_tools(ran), recording, choice=choice
        )
        assert seen == [{"amount": amount}], "the unselected alternative invokes no policy"
        assert ran == [amount], "the unselected alternative runs no tool"
        assert [o["tool_call_id"] for o in outs] == [call_id]

    @pytest.mark.parametrize(
        "choice, error",
        [(-1, ValueError), (2, ValueError), (True, TypeError), ("0", TypeError), (1.0, TypeError)],
        ids=repr,
    )
    def test_an_invalid_choice_is_refused_before_anything_runs(self, ex, choice, error):
        ran = []
        with pytest.raises(error, match="Recusal refused the completion"):
            ex.gate_tool_calls(
                self.two_choices(ex), self.charge_tools(ran), lambda n, a: [], choice=choice
            )
        assert ran == []

    def test_a_completion_with_no_choices_is_refused(self, ex):
        completion = ex.completion_with([])
        completion.choices = []
        with pytest.raises(ValueError, match="no choices"):
            ex.gate_tool_calls(completion, ex.TOOLS, ex.sql_scope_policy)

    def test_one_choice_dispatches_with_or_without_a_selection(self, ex):
        ran = []
        single = ex.completion_with([ex.tool_call("charge", {"amount": 5}, "c1")])
        ex.gate_tool_calls(single, self.charge_tools(ran), lambda n, a: [])
        ex.gate_tool_calls(single, self.charge_tools(ran), lambda n, a: [], choice=0)
        assert ran == [5, 5]

    def test_negative_control_the_naive_flattening_would_run_both(self, ex):
        """What the dispatcher did before: every choice's calls, both charges."""
        ran = []
        tools = self.charge_tools(ran)
        for choice in self.two_choices(ex).choices:
            for call in choice.message.tool_calls:
                ex.judge_and_run(
                    call.function.name, call.function.arguments, tools, lambda n, a: []
                )
        assert ran == [5, 7]

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

    def test_duplicate_call_ids_are_refused_before_anything_runs(self, ex):
        """Ids are validated before dispatch: a turn refused for a duplicate id has run
        nothing, so a caller that retries the turn cannot repeat a side effect."""
        ex.EXECUTED.clear()
        response = ex.response_with(
            [
                ex.function_call("echo", {"text": "1"}, "same"),
                ex.function_call("echo", {"text": "2"}, "same"),
            ]
        )
        with pytest.raises(ValueError, match="duplicate call ids"):
            ex.gate_function_calls(response, ex.TOOLS, ex.sql_scope_policy)
        assert ex.EXECUTED == []
        completion = ex.completion_with(
            [
                ex.tool_call("echo", {"text": "1"}, "same"),
                ex.tool_call("echo", {"text": "2"}, "same"),
            ]
        )
        with pytest.raises(ValueError, match="duplicate call ids"):
            ex.gate_tool_calls(completion, ex.TOOLS, ex.sql_scope_policy)
        assert ex.EXECUTED == []

    def test_a_missing_id_anywhere_in_the_batch_refuses_the_turn_before_anything_runs(self, ex):
        ex.EXECUTED.clear()
        response = ex.response_with(
            [
                ex.function_call("echo", {"text": "1"}, "ok"),
                ex.function_call("echo", {"text": "2"}, ""),
            ]
        )
        with pytest.raises(ValueError, match="no call id"):
            ex.gate_function_calls(response, ex.TOOLS, ex.sql_scope_policy)
        assert ex.EXECUTED == []

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
    assert _same({"a": [1, 2.0, "x", True]}, {"a": [1, 2.0, "x", True]})
    assert not _same(True, 1) and not _same(1, 1.0) and not _same(0, False)
    assert not _same({"a": 1}, {"a": "1"}) and not _same([1], (1,))


class TestBoundInvocation:
    """Adapters review, finding 3 (direct loops): the loop judged the decoded JSON and let
    Python fill omitted defaults after judgment; ``charge(amount=100)`` proposed with ``{}``
    reached a budget policy as ``{}`` and ran 100. The call is bound to the signature with
    defaults applied before judgment, and the bound invocation is what runs."""

    @pytest.fixture
    def ex(self):
        pytest.importorskip("openai", reason="the openai package is not installed")
        import importlib.util

        spec = importlib.util.spec_from_file_location("recusal_responses_bound_example", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def budget(seen):
        from recusal import Finding

        def policy(name, args):
            seen.append(args)
            if args.get("amount", 0) > 10:
                return [Finding.fail("budget", severity="CRITICAL", message="over 10")]
            return []

        return policy

    def test_an_omitted_default_is_judged_at_the_value_the_function_uses(self, ex):
        seen = []
        ex.EXECUTED.clear()
        out = ex.judge_and_run("charge", "{}", ex.TOOLS, self.budget(seen))
        assert seen == [{"amount": 100}] and type(seen[0]["amount"]) is int
        assert ex.EXECUTED == [] and "over 10" in out

    def test_an_explicit_value_is_judged_and_runs_exactly_as_bound(self, ex):
        seen = []
        ex.EXECUTED.clear()
        ex.judge_and_run("charge", json.dumps({"amount": 5}), ex.TOOLS, self.budget(seen))
        assert _same(seen[0], {"amount": 5}) and _same(ex.EXECUTED[0][1], {"amount": 5})

    def test_a_non_canonical_default_is_judged_as_the_function_receives_it(self, ex):
        """No coercion exists on this path: the policy sees the default exactly as Python
        hands it to the function, a str here, and so does the function."""
        received, seen = [], []

        def charge(amount: int = "100") -> str:  # type: ignore[assignment]
            received.append(amount)
            return "recorded"

        ex.judge_and_run("charge", "{}", {"charge": charge}, lambda n, a: seen.append(a) or [])
        assert _same(seen[0]["amount"], "100") and _same(received[0], "100")

    def test_a_mutating_policy_cannot_change_the_bound_invocation(self, ex):
        received = []

        def charge(amount: int = 100, tags: list[str] = None) -> str:  # type: ignore[assignment]
            received.append((amount, tags))
            return "recorded"

        def mutate(name, args):
            args["amount"] = 999
            if args["tags"] is not None:
                args["tags"].append("x")
            return []

        ex.judge_and_run(
            "charge", json.dumps({"amount": 5, "tags": ["a"]}), {"charge": charge}, mutate
        )
        assert received == [(5, ["a"])]

    @pytest.mark.parametrize(
        "arguments, fragment",
        [
            ('{"sql": "x", "nope": 1}', "unexpected keyword argument 'nope'"),
            ("{}", "missing a required argument: 'sql'"),
        ],
    )
    def test_a_call_that_does_not_bind_is_refused_naming_the_parameter_only(
        self, ex, arguments, fragment
    ):
        ex.EXECUTED.clear()
        out = ex.judge_and_run("run_sql", arguments, ex.TOOLS, lambda n, a: [])
        assert (
            out.startswith("Recusal refused `run_sql`: arguments do not bind") and fragment in out
        )
        assert ex.EXECUTED == []

    def test_a_signature_that_cannot_be_judged_by_name_is_refused(self, ex):
        import enum
        from typing import Any

        class Color(enum.Enum):
            RED = "red"

        def variadic(sql: str, **options) -> str:
            return "x"

        def unannotated(sql) -> str:
            return "x"

        def anything(value: Any) -> str:
            return "x"

        def enumerated(color: Color) -> str:
            return "x"

        def positional(sql: str, /) -> str:
            return "x"

        def odd_default(when: str = object()) -> str:  # type: ignore[assignment]
            return "x"

        seen = []
        for func, fragment in (
            (variadic, "variadic"),
            (unannotated, "not annotated"),
            (anything, "not annotated"),
            (enumerated, "not annotated"),
            (positional, "positional-only"),
            (odd_default, "outside the JSON domain"),
        ):
            out = ex.judge_and_run(
                "t", json.dumps({"sql": "x"}), {"t": func}, lambda n, a: seen.append(a) or []
            )
            assert (
                out.startswith("Recusal refused `t`: its signature cannot be judged")
                and fragment in out
            ), out
        assert seen == []

    def test_supported_shapes_bind_and_run(self, ex):
        from typing import Optional

        received = []

        def shapes(
            a: int, b: Optional[str] = None, c: list[int] = None, d: dict[str, float] = None
        ) -> str:  # type: ignore[assignment]
            received.append((a, b, c, d))
            return "ok"

        out = ex.judge_and_run(
            "s", json.dumps({"a": 1, "c": [1, 2], "d": {"x": 1.5}}), {"s": shapes}, lambda n, a: []
        )
        assert out == "ok" and received == [(1, None, [1, 2], {"x": 1.5})]

    def test_negative_control_the_naive_loop_runs_the_omitted_default(self, ex):
        ex.EXECUTED.clear()
        ex.dispatch_ungated(ex.response_with([ex.function_call("charge", {}, "c1")]), ex.TOOLS)
        assert ex.EXECUTED == [("charge", {"amount": 100})]
