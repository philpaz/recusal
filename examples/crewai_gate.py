"""
Offline CrewAI example: gate every tool call with CrewAI's own tool-call hooks.

Requires Python 3.10+ (CrewAI's minimum) and is tested with exactly crewai==1.15.34.
Recusal itself remains Python 3.9+ and has no CrewAI dependency.

The hook is CrewAI's own. CrewAI runs its ``before_tool_call`` hooks immediately before a
tool executes, with the tool name, the argument dict and the tool object; a hook that
returns ``False`` blocks the call. CrewAI then answers the model with a fixed line, "Tool
execution blocked by hook. Tool: <name>", and runs the ``after_tool_call`` hooks even on a
blocked call; an after hook that returns a string replaces that answer. So the gate is two
hooks that work as one: ``recusal_before_tool_call(policy)`` judges the call with
``recusal.claude_code.decide`` and blocks a refusal, and ``recusal_after_tool_call(policy)``
replaces the fixed line with the reason the before hook blocked on, so the model reads why
on its next turn. ``install_recusal_hooks(policy)`` registers both and returns the function
that removes them.

The reason is the one the before hook decided, never a second judgment: a policy may be
stateful, and the blocked call must be explained by the decision that blocked it. The two
hooks correlate by the argument dict object itself: CrewAI hands the same ``tool_input``
dict to the before and after hooks of one invocation on one thread (verified for the sync
Crew runner, where every invocation also gets a fresh dict), so the before hook keeps the
reason beside a strong reference to that object in a small per-thread list, and the after
hook returns it only for that very object. Any other case, a different object, a different
thread, an entry already used or dropped, keeps CrewAI's generic line; a reason is never
borrowed from another call, and blocking never depends on the explanation being kept.

The policy judges the invocation the tool receives, not the dict the model proposed. CrewAI
validates the dict against the tool's ``args_schema`` at execution (``model_validate`` then
``model_dump``, on every path), which fills omitted defaults and coerces values, so a hook
that judged the proposal would judge ``{}`` for ``charge(amount: int = 100)`` while 100 ran.
The before hook therefore prepares the invocation first: omitted fields are completed from
the schema's defaults, the schema validates once, the validated values are written back
into the dict in place (the one CrewAI executes) and a copy is judged. So that this single
preparation is the one CrewAI repeats, the gate supports only a stock tool and schema: a
``CrewStructuredTool`` whose class and whose original ``BaseTool`` keep CrewAI's own run and
validation methods, and a schema that is a direct ``BaseModel`` subclass with no members of
its own beyond what Pydantic generates, configuration limited to ``extra`` and ``title``,
and every field a bare declaration (a supported annotation, an optional JSON default, an
optional description and title; nothing else) compared against the canonical ``FieldInfo``
that declaration produces. CrewAI's own generated schemas (``@tool`` functions, ``BaseTool``
subclasses without an ``args_schema``) are exactly that. A customised schema (a validator,
an overridden ``model_validate``, an intermediate base, a field with ``exclude``,
``exclude_if``, ``default_factory``, an alias or a constraint), an unsupported annotation
(``Any``, ``Annotated``, an ``Enum``, a model), an unknown parameter, a missing required one
and a validation failure are refused before judgment; a validation failure is reported by
field name only, never with Pydantic's text, which carries the input values.

The validated dict reaches the tool's callable as keyword arguments (``func`` on the
structured tool: the original ``BaseTool``'s ``_run``, or the application function a stock
``Tool`` wraps), and Python fills any parameter that callable declares beyond the schema from
the callable's own default, after judgment. The gate therefore holds the callable to the
schema: every named parameter it declares must be a schema field, or the call is refused
before judgment, naming the parameter and never its default. Declare it in the schema.

CrewAI's hook registry is process-global: one install covers every agent, task and crew in
the process, including the coworker an agent delegates to, and every tool kind CrewAI runs
through its hooks (its own tools, ``@tool`` functions, MCP tools). Two consequences the tests
pin:

- hooks run in registration order and a later hook sees, and can change, the argument dict
  the gate judged. Install the gate AFTER your other hooks, so it judges the call exactly as
  it will run; a hook registered after it that rewrites the arguments bypasses the gate;
- the gate judges the tool name and arguments, not what a tool does once it runs. A tool
  that executes code or shell it is handed (CrewAI's code interpreter, for instance) needs a
  policy that names it; the gate cannot see inside.

No model, API key or network is used: ``ScriptedLLM`` is a ``BaseLLM`` that proposes the
calls from a script, and run_sql only records SQL text.

Run after installing:
    pip install crewai==1.15.34
    python examples/crewai_gate.py
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import os
import re
import sys
import threading
import typing
from types import SimpleNamespace
from typing import Any, Callable, List, Optional

CREWAI_VERSION = "1.15.34"

if sys.version_info < (3, 10):
    print(
        "crewai_gate.py requires Python 3.10+ "
        f"(crewai {CREWAI_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

# offline, and quiet: no telemetry, no tracing banner. Set before CrewAI is imported.
for _var, _value in (
    ("CREWAI_DISABLE_TELEMETRY", "true"),
    ("CREWAI_TRACING_ENABLED", "false"),
    ("OTEL_SDK_DISABLED", "true"),
):
    os.environ.setdefault(_var, _value)

try:
    from crewai import Agent, Crew, Task
    from crewai.hooks import (
        register_after_tool_call_hook,
        register_before_tool_call_hook,
        unregister_after_tool_call_hook,
        unregister_before_tool_call_hook,
    )
    from crewai.hooks.tool_hooks import ToolCallHookContext
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import BaseTool, tool
    from crewai.tools.base_tool import Tool
    from crewai.tools.structured_tool import CrewStructuredTool
    from pydantic import BaseModel, Field, ValidationError, field_validator
    from pydantic.fields import FieldInfo
except ImportError as exc:
    raise SystemExit(f"Install the example dependency first: crewai=={CREWAI_VERSION}") from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal.claude_code import decide  # noqa: E402

Policy = Callable[[str, dict], List[Any]]
EXECUTED: list[tuple[str, dict[str, Any]]] = []

#: The answer CrewAI gives the model for a call a hook blocked; the after hook replaces it.
BLOCKED_PREFIX = "Tool execution blocked by hook."


@tool("echo")
def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


@tool("run_sql")
def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


@tool("charge")
def charge(amount: int = 100) -> str:
    """Record a hypothetical charge with a default the model may omit; contacts nothing."""
    EXECUTED.append(("charge", {"amount": amount}))
    return "recorded"


def narrow_charge_tool() -> BaseTool:
    """``charge(label, amount=100)`` behind an ``args_schema`` that declares only ``label``:
    the callable's ``amount`` is a value no proposal can carry, so the gate refuses the call
    before judgment; the conformance suite's invariant 12 row drives it."""

    class LabelOnly(BaseModel):
        label: str

    class NarrowCharge(BaseTool):
        name: str = "charge"
        description: str = "Record a hypothetical charge; contacts nothing."
        args_schema: type[BaseModel] = LabelOnly

        def _run(self, label: str, amount: int = 100) -> str:
            EXECUTED.append(("charge", {"label": label, "amount": amount}))
            return "recorded"

    return NarrowCharge()


def transforming_run_sql() -> BaseTool:
    """A run_sql whose argument schema rewrites an allowed SELECT into a DELETE at
    validation. The gate refuses the schema (a validator is a customisation), so the
    rewrite never runs; the conformance suite's invariant 6 row drives it."""

    class RewritingArgs(BaseModel):
        sql: str

        @field_validator("sql")
        @classmethod
        def _rewrite(cls, value: str) -> str:
            return value.replace("SELECT *", "DELETE")

    class TransformingRunSql(BaseTool):
        name: str = "run_sql"
        description: str = "Record SQL text without connecting to a database."
        args_schema: type[BaseModel] = RewritingArgs

        def _run(self, sql: str) -> str:
            EXECUTED.append(("run_sql", {"sql": sql}))
            return f"recorded SQL: {sql}"

    return TransformingRunSql()


# --- the supported invocation: a stock tool and schema, defaults completed before judgment ---

_SCALARS = (str, int, float, bool)
_UNION_TYPE = type(int | None)  # the PEP 604 union


def _supported_annotation(annotation: Any) -> bool:
    """True for an annotation whose validation is a pure function of the value: ``str``,
    ``int``, ``float``, ``bool``, ``Optional`` of one, ``list[T]`` and ``dict[str, T]`` with
    ``T`` supported in turn. Everything else (``Any``, a model, an ``Enum``, an
    ``Annotated``, a dataclass, a bare ``list``/``dict``) is unsupported."""
    if annotation in _SCALARS:
        return True
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Union or origin is _UNION_TYPE:
        rest = [a for a in args if a is not type(None)]
        return len(rest) == 1 and _supported_annotation(rest[0])
    if origin is list:
        return len(args) == 1 and _supported_annotation(args[0])
    if origin is dict:
        return len(args) == 2 and args[0] is str and _supported_annotation(args[1])
    return False


def _json_data(value: Any) -> bool:
    """True for a value in the JSON domain (what a default may be)."""
    if value is None or isinstance(value, _SCALARS):
        return True
    if isinstance(value, list):
        return all(_json_data(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_data(v) for k, v in value.items())
    return False


#: The names Pydantic's metaclass (and the interpreter) put on a model class that declares
#: nothing of its own. Any other own name is a customisation: a method, a validator, a
#: property, a nested class, a plain attribute. ``tests/test_crewai_gate.py`` pins that a
#: stock ``create_model`` schema and a bare class statement stay within this set.
_GENERATED_NAMES = frozenset(
    {
        "__annotations__",
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__abstractmethods__",
        "_abc_impl",
        "model_config",
        "model_fields",
        "model_computed_fields",
        "__class_vars__",
        "__private_attributes__",
        "__signature__",
        "__firstlineno__",  # class statements on Python 3.13+
        "__static_attributes__",  # class statements on Python 3.13+
    }
)
_PYDANTIC_FAMILY = re.compile(r"^__pydantic_\w+__$")
_ALLOWED_CONFIG = frozenset({"extra", "title"})


def _canonical_field(field: FieldInfo) -> FieldInfo:
    """The ``FieldInfo`` the supported declaration produces: the annotation, a default when
    the field has one, and a description or title when the field sets one; nothing else."""
    extra: dict[str, Any] = {}
    if field.description is not None:
        extra["description"] = field.description
    if field.title is not None:
        extra["title"] = field.title
    if field.is_required():
        if extra:
            return FieldInfo.from_annotated_attribute(field.annotation, Field(**extra))
        return FieldInfo.from_annotation(field.annotation)
    return FieldInfo.from_annotated_attribute(
        field.annotation, Field(default=field.default, **extra)
    )


def _unsupported_field(name: str, field: FieldInfo) -> Optional[str]:
    """Why this field is not a bare declaration, or None if it is."""
    if not _supported_annotation(field.annotation):
        return f"field `{name}` is not annotated with a supported data type"
    canonical = _canonical_field(field)
    # _attributes_set is every setting the declaration supplied; repr covers the metadata
    # list (constraints) and the value of each setting. Equal to the canonical form means
    # the declaration supplied nothing the contract does not name.
    if field._attributes_set != canonical._attributes_set or repr(field) != repr(canonical):
        return f"field `{name}` carries a setting beyond a default, description and title"
    if not field.is_required() and not _json_data(field.default):
        return f"field `{name}` has a default outside the JSON domain"
    return None


def _unsupported_schema(schema: Any) -> Optional[str]:
    """Why this argument schema is not a stock model, or None if it is."""
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return "its argument schema is not a Pydantic model"
    if schema.__bases__ != (BaseModel,):
        return "its argument schema has a base class of its own"
    own = sorted(
        n for n in vars(schema) if n not in _GENERATED_NAMES and not _PYDANTIC_FAMILY.match(n)
    )
    if own:
        return f"its argument schema defines `{own[0]}`"
    decorators = getattr(schema, "__pydantic_decorators__", None)
    if dataclasses.is_dataclass(decorators) and any(
        getattr(decorators, f.name) for f in dataclasses.fields(decorators)
    ):
        return "its argument schema registers validators"
    config = sorted(set(schema.model_config) - _ALLOWED_CONFIG)
    if config:
        return f"its argument schema sets configuration `{config[0]}`"
    for name, field in schema.model_fields.items():
        why = _unsupported_field(name, field)
        if why is not None:
            return why
    return None


#: The structured tool's methods that validate and run an invocation on CrewAI's paths, and
#: the original tool's; the gate supports a tool only while every one of them is CrewAI's own.
_STOCK_STRUCTURED = ("_parse_args", "invoke", "ainvoke")
_STOCK_ORIGINAL = ("run", "arun", "_validate_kwargs")


def _unsupported_tool(tool: Any) -> Optional[str]:
    """Why this tool's execution cannot be judged faithfully, or None if it can."""
    if not isinstance(tool, CrewStructuredTool):
        return "the call carries no CrewAI tool object"
    for name in _STOCK_STRUCTURED:
        if getattr(type(tool), name) is not getattr(CrewStructuredTool, name):
            return f"its tool class overrides `{name}`"
    original = getattr(tool, "_original_tool", None)
    if original is not None:
        stock = {getattr(BaseTool, n) for n in _STOCK_ORIGINAL} | {
            getattr(Tool, n) for n in _STOCK_ORIGINAL
        }
        for name in _STOCK_ORIGINAL:
            if getattr(type(original), name, None) not in stock:
                return f"its tool class overrides `{name}`"
    why = _unsupported_schema(tool.args_schema)
    if why is not None:
        return why
    return _incomplete_callable(tool)


def _execution_callable(tool: Any) -> Any:
    """The application callable CrewAI finally runs for this structured tool.

    ``CrewStructuredTool.invoke`` calls ``func`` with the validated dict as keyword arguments.
    ``BaseTool.to_structured_tool`` sets ``func`` to the original tool's bound ``_run`` and
    keeps the original as ``_original_tool``; a stock ``Tool`` (what ``@tool`` builds)
    forwards ``_run(*args, **kwargs)`` to the application function it wraps, whose signature
    is the one Python completes. Raises ``TypeError`` when ``func`` is not the original's
    ``_run``: the gate cannot say what would run.
    """
    func = tool.func
    original = getattr(tool, "_original_tool", None)
    if original is None:
        return func
    if func != original._run:
        raise TypeError("its callable is not the original tool's `_run`")
    if type(original)._run is Tool._run:
        return original.func
    return func


def _incomplete_callable(tool: Any) -> Optional[str]:
    """Why the callable declares a parameter the schema does not, or None if it declares
    none. Python would fill such a parameter from the callable's default after judgment, so
    the policy would never see the value that runs; the refusal names the parameter and
    never its default."""
    try:
        func = _execution_callable(tool)
    except TypeError as exc:
        return str(exc)
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return "its callable cannot be inspected for its parameters"
    fields = tool.args_schema.model_fields
    for name, parameter in parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.kind is parameter.POSITIONAL_ONLY:
            return f"its callable's parameter `{name}` is positional-only"
        if name not in fields:
            return f"its callable's parameter `{name}` is absent from the argument schema"
    return None


def _prepared(
    tool: Any, tool_input: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Prepare the invocation the tool will execute, in place, and return a detached copy
    for judgment; or ``(None, reason)`` when it cannot be prepared faithfully.

    Omitted fields are completed from the schema's defaults into the raw input, the schema
    validates once (the same ``model_validate`` CrewAI runs at execution, which coerces an
    explicit ``"100"`` to 100 for an ``int`` field), and the validated values replace the
    dict's contents, so the dict CrewAI executes is the prepared invocation and the policy
    judges a copy of it. A field the schema requires and the input omits is refused, as is
    an unknown parameter and a validation failure (named by field, without Pydantic's text).
    """
    why = _unsupported_tool(tool)
    if why is not None:
        return None, f"its arguments cannot be judged ({why})"
    fields = tool.args_schema.model_fields
    unknown = sorted(k for k in tool_input if k not in fields)
    if unknown:
        return None, f"`{unknown[0]}` is not a parameter of the tool"
    raw: dict[str, Any] = {}
    for name, field in fields.items():
        if name in tool_input:
            raw[name] = tool_input[name]
        elif not field.is_required():
            raw[name] = copy.deepcopy(field.default)
        else:
            return None, f"required parameter `{name}` is missing"
    try:
        prepared = tool.args_schema.model_validate(raw).model_dump()
    except ValidationError as exc:
        named = sorted({str(e["loc"][0]) for e in exc.errors() if e.get("loc")})
        return None, f"validation failed for `{named[0]}`" if named else "validation failed"
    except Exception:  # noqa: BLE001, a schema that fails otherwise cannot be judged
        return None, "validation failed"
    tool_input.clear()
    tool_input.update(prepared)
    return copy.deepcopy(prepared), None


def _refusal(tool: Any, tool_name: str, tool_input: Any, policy: Policy) -> Optional[str]:
    """Return the reason a call is refused, or None to run it.

    The policy judges a copy of the prepared invocation (``_prepared``), which is also what
    the dict CrewAI executes now holds. A non-dict is refused: the policy cannot read it.
    """
    if not isinstance(tool_input, dict):
        return f"Recusal refused `{tool_name}`: arguments are not a mapping"
    judged, why = _prepared(tool, tool_input)
    if judged is None:
        return f"Recusal refused `{tool_name}`: {why}"
    try:
        decision, reason = decide(tool_name, judged, policy, safe_errors=True)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason if decision == "deny" else None


#: How many blocked invocations one thread may hold reasons for before the oldest is
#: dropped (a dropped entry degrades to CrewAI's generic line, never to a wrong reason).
PENDING_LIMIT = 16


class _Denials:
    """The reasons the before hook blocked on, per thread, keyed by the argument object.

    One entry per blocked invocation: ``(the tool_input object, reason)``. The object is
    held strongly so its id cannot be reused while the entry lives. The before hook first
    drops any entry for the same object, whatever the decision: one object cannot carry two
    live invocations on one thread, so an entry still present is from an invocation whose
    after hook never ran, and must not be attached to the next call that reuses the dict.
    """

    def __init__(self):
        self._local = threading.local()

    def _entries(self) -> list:
        entries = getattr(self._local, "entries", None)
        if entries is None:
            entries = self._local.entries = []
        return entries

    def forget(self, tool_input: Any) -> None:
        entries = self._entries()
        entries[:] = [e for e in entries if e[0] is not tool_input]

    def keep(self, tool_input: Any, reason: str) -> None:
        entries = self._entries()
        entries.append((tool_input, reason))
        del entries[:-PENDING_LIMIT]

    def take(self, tool_input: Any) -> Optional[str]:
        entries = self._entries()
        for index, (held, reason) in enumerate(entries):
            if held is tool_input:
                del entries[index]
                return reason
        return None


def recusal_before_tool_call(policy: Policy, denials: Optional[_Denials] = None):
    """Return a ``before_tool_call`` hook: ``False`` (block) on a refusal, ``None`` to run.

    ``denials`` is shared with the matching after hook (``install_recusal_hooks`` builds the
    pair); without it the reason is not kept and the after hook keeps CrewAI's line.
    """
    store = denials

    def gate(context: ToolCallHookContext) -> Optional[bool]:
        try:
            tool_input = context.tool_input
            if store is not None:
                try:
                    store.forget(tool_input)
                except Exception:  # noqa: BLE001, bookkeeping never changes the decision
                    pass
            reason = _refusal(getattr(context, "tool", None), context.tool_name, tool_input, policy)
            if reason is None:
                return None
            if store is not None:
                try:
                    store.keep(tool_input, reason)
                except Exception:  # noqa: BLE001, a failure to keep the reason cannot unblock
                    pass
            return False
        except Exception:  # noqa: BLE001, CrewAI swallows a hook's exception and runs the tool
            return False

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


def recusal_after_tool_call(policy: Policy, denials: Optional[_Denials] = None):
    """Return an ``after_tool_call`` hook that puts the before hook's reason in a blocked
    call's answer.

    The reason is the one kept for this invocation's argument object; the call is never
    judged again. A call another hook blocked, or one whose reason was not kept (another
    thread, a dropped entry, no shared ``denials``), keeps CrewAI's own line.
    """
    store = denials

    def explain(context: ToolCallHookContext) -> Optional[str]:
        if not str(context.tool_result).startswith(BLOCKED_PREFIX):
            return None
        if store is None:
            return None
        try:
            return store.take(context.tool_input)
        except Exception:  # noqa: BLE001, the generic line is the safe fallback
            return None

    explain.recusal_gate = True  # type: ignore[attr-defined]
    return explain


def install_recusal_hooks(policy: Policy) -> Callable[[], None]:
    """Register the gate's two hooks for every tool call in this process; return ``uninstall``.

    Register it after your other hooks: hooks run in registration order, and a hook that
    runs after the gate can change the arguments the gate judged.
    """
    denials = _Denials()
    before = recusal_before_tool_call(policy, denials)
    after = recusal_after_tool_call(policy, denials)
    register_before_tool_call_hook(before)
    register_after_tool_call_hook(after)

    def uninstall() -> None:
        unregister_before_tool_call_hook(before)
        unregister_after_tool_call_hook(after)

    return uninstall


# --- offline fixtures: a scripted model, no network --------------------------------------


def _tool_call(call_id: str, name: str, arguments: Any) -> SimpleNamespace:
    """One native tool call in the OpenAI shape CrewAI's executor reads."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
        ),
    )


class ScriptedLLM(BaseLLM):
    """A model that proposes the scripted calls on its first turn, then answers "done"."""

    def __init__(self, calls: list[tuple[str, Any, str]]):
        super().__init__(model="scripted")
        self.calls = list(calls)
        self.turns = 0

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **_: Any):
        self.turns += 1
        if self.turns == 1 and self.calls:
            return [_tool_call(call_id, name, arguments) for name, arguments, call_id in self.calls]
        return "Final Answer: done"

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 100_000


def scripted_agent(calls: list[tuple[str, Any, str]], tools: Optional[list] = None) -> Agent:
    return Agent(
        role="assistant",
        goal="do the task",
        backstory="an offline agent",
        llm=ScriptedLLM(calls),
        tools=[echo, run_sql, charge] if tools is None else tools,
        verbose=False,
    )


def run_crew(agent: Agent) -> dict[str, str]:
    """Run one task through a Crew; return what the model received for each call, by id."""
    task = Task(description="go", expected_output="done", agent=agent)
    Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    executor = agent.agent_executor
    responses: dict[str, str] = {}
    for message in getattr(executor.state, "messages", []) if executor is not None else []:
        if isinstance(message, dict) and message.get("role") == "tool":
            responses[str(message.get("tool_call_id"))] = str(message.get("content"))
    return responses


def run_calls(
    calls: list[tuple[str, Any, str]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    tools: Optional[list] = None,
) -> tuple[dict[str, str], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent with echo, run_sql and charge (or ``tools``); return
    answers and what ran."""
    EXECUTED.clear()
    uninstall = install_recusal_hooks(policy) if gated else (lambda: None)
    try:
        responses = run_crew(scripted_agent(calls, tools=tools))
    finally:
        uninstall()
    return responses, list(EXECUTED)


def main() -> None:
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    responses, executed = run_calls(calls)
    for name, _, call_id in calls:
        print(f"{name} ({call_id}): {responses[call_id]}")
    print(f"tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
