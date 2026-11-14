"""Every framework gate has an audited route, and the documents that list them are true.

No SDK is imported: these read the example sources and the Markdown. A gate added under
``examples/`` without its audited companion fails here, and so does a surface label in
``docs/ARCHITECTURE.md`` or ``docs/HOWTO.md`` that the code no longer writes.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

#: ``*_gate.py`` files that are not a framework gate with a companion: two are Claude Code
#: hook scripts, whose audited route is the package's own ``run_pretooluse_hook(policy,
#: audit=...)``, and ``quality_gate.py`` adjudicates a work product, not a tool call.
NOT_A_FRAMEWORK_GATE = {"allowlist_gate.py", "claude_code_gate.py", "quality_gate.py"}

#: Each framework gate, its audited companion, and the surface labels the companion writes.
ROUTES = {
    "langgraph_gate.py": ("langgraph_audit.py", ["langgraph.toolnode"]),
    "langchain_agent_gate.py": ("langchain_agent_audit.py", ["langchain.create_agent"]),
    "openai_agents_gate.py": (
        "openai_agents_audit.py",
        ["openai_agents.tool_input_guardrail"],
    ),
    "openai_agents_mcp_gate.py": (
        "openai_agents_mcp_audit.py",
        [
            "openai_agents.tool_input_guardrail",
            "openai_agents.mcp.call_tool",
            "openai_agents.hosted_mcp.approval",
        ],
    ),
    "openai_responses_gate.py": (
        "openai_responses_audit.py",
        ["openai.responses.function_call", "openai.chat_completions.tool_call"],
    ),
    "google_adk_gate.py": ("google_adk_audit.py", ["google_adk.before_tool_callback"]),
    "crewai_gate.py": ("crewai_audit.py", ["crewai.before_tool_call"]),
    "pydantic_ai_gate.py": ("pydantic_ai_audit.py", ["pydantic_ai.before_tool_execute"]),
}

#: Gates whose code is an installable package under ``adapters/``. The example is then a
#: demonstration that imports the package, and the checks below read the package: the
#: module that decides, the module that records, and the name the example must import.
PACKAGED = {
    "openai_agents_mcp_gate.py": (
        "adapters/openai-agents-mcp/src/recusal_openai_agents_mcp/_gate.py",
        "adapters/openai-agents-mcp/src/recusal_openai_agents_mcp/_audit.py",
        "recusal_openai_agents_mcp",
    ),
    "crewai_gate.py": (
        "adapters/crewai/src/recusal_crewai/_gate.py",
        "adapters/crewai/src/recusal_crewai/_audit.py",
        "recusal_crewai",
    ),
    "langchain_agent_gate.py": (
        "adapters/langchain/src/recusal_langchain/_gate.py",
        "adapters/langchain/src/recusal_langchain/_audit.py",
        "recusal_langchain",
    ),
    "langgraph_gate.py": (
        "adapters/langgraph/src/recusal_langgraph/_gate.py",
        "adapters/langgraph/src/recusal_langgraph/_audit.py",
        "recusal_langgraph",
    ),
    "openai_agents_gate.py": (
        "adapters/openai-agents/src/recusal_openai_agents/_gate.py",
        "adapters/openai-agents/src/recusal_openai_agents/_audit.py",
        "recusal_openai_agents",
    ),
    "google_adk_gate.py": (
        "adapters/google-adk/src/recusal_google_adk/_gate.py",
        "adapters/google-adk/src/recusal_google_adk/_audit.py",
        "recusal_google_adk",
    ),
    "openai_responses_gate.py": (
        "adapters/openai/src/recusal_openai/_gate.py",
        "adapters/openai/src/recusal_openai/_audit.py",
        "recusal_openai",
    ),
    "pydantic_ai_gate.py": (
        "adapters/pydantic-ai/src/recusal_pydantic_ai/_gate.py",
        "adapters/pydantic-ai/src/recusal_pydantic_ai/_audit.py",
        "recusal_pydantic_ai",
    ),
}

DOCUMENTS = ("docs/ARCHITECTURE.md", "docs/HOWTO.md")


def _deciding_source(gate: str) -> Path:
    """The file a gate's ``decide`` calls are in: the package module, or the example."""
    return ROOT / PACKAGED[gate][0] if gate in PACKAGED else EXAMPLES / gate


def _recording_source(gate: str) -> Path:
    """The file a gate's surface labels are defined in."""
    return ROOT / PACKAGED[gate][1] if gate in PACKAGED else EXAMPLES / ROUTES[gate][0]


def _decide_calls(path: Path) -> list:
    return [
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "decide"
    ]


def _string_constants(path: Path) -> set:
    """Every string literal assigned at module level, including inside a dict literal."""
    found = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            for child in ast.walk(node.value):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    found.add(child.value)
    return found


def _table_rows(document: str) -> list:
    """The rows of the audited-routes table: (example file, surface label)."""
    text = (ROOT / document).read_text(encoding="utf-8")
    rows = re.findall(r"^\|[^|]*\| `examples/(\w+_audit\.py)` \| `([\w.]+)` \|", text, re.M)
    assert rows, f"{document} has no audited-routes table"
    return rows


def test_every_framework_gate_is_registered_here():
    gates = {p.name for p in EXAMPLES.glob("*_gate.py")} - NOT_A_FRAMEWORK_GATE
    assert gates == set(ROUTES), (
        "a *_gate.py example is missing its audited route, or this registry is stale: "
        f"{sorted(gates ^ set(ROUTES))}"
    )


@pytest.mark.parametrize("gate", sorted(ROUTES))
def test_the_gate_has_an_audited_companion_and_its_tests(gate):
    companion, _ = ROUTES[gate]
    assert (EXAMPLES / companion).is_file(), companion
    assert (ROOT / "tests" / f"test_{companion}").is_file(), f"tests/test_{companion}"


@pytest.mark.parametrize("gate", sorted(ROUTES))
def test_the_companion_defines_the_registered_surfaces(gate):
    _, surfaces = ROUTES[gate]
    source = _recording_source(gate)
    missing = sorted(set(surfaces) - _string_constants(source))
    assert not missing, f"{source.name} does not define the surface(s) {missing}"


@pytest.mark.parametrize("gate", sorted(ROUTES))
def test_the_gate_passes_recording_arguments_to_decide(gate):
    """One decision path: the plain gate's own call to ``decide`` carries ``**record``, so
    the audited route cannot be a second copy. The LangChain gate has no ``decide`` call of
    its own (it wraps the LangGraph hooks)."""
    calls = _decide_calls(_deciding_source(gate))
    if gate == "langchain_agent_gate.py":
        assert calls == []
        return
    assert calls, gate
    for call in calls:
        assert any(k.arg is None for k in call.keywords), (
            f"{gate}:{call.lineno}: a decide(...) call takes no **record, so a decision made "
            "there cannot be put on an audit log"
        )


def test_every_framework_gate_is_an_installable_package():
    """Each framework gate's code lives in an adapter package under ``adapters/``, and
    its example demonstrates that package."""
    assert set(PACKAGED) == set(ROUTES), sorted(set(PACKAGED) ^ set(ROUTES))


@pytest.mark.parametrize("gate", sorted(PACKAGED))
def test_a_packaged_gate_has_no_second_copy_in_its_examples(gate):
    """The example and its audited companion import the package and decide nothing
    themselves, so the code a user installs is the code the examples and their tests
    run."""
    decides, records, package = PACKAGED[gate]
    assert (ROOT / decides).is_file() and (ROOT / records).is_file()
    for name in (gate, ROUTES[gate][0]):
        path = EXAMPLES / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert package in imported, f"{name} does not import {package}"
        assert _decide_calls(path) == [], f"{name} calls decide itself"
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assert not defined & {"_refusal", "_unjudgeable", "_recording"}, name


@pytest.mark.parametrize("document", DOCUMENTS)
def test_the_documented_table_matches_the_code(document):
    documented = set(_table_rows(document))
    expected = {
        (companion, surface)
        for companion, surfaces in ROUTES.values()
        for surface in surfaces
        # the function-tool surface of the MCP agent is the function-tool example's row
        if not (companion == "openai_agents_mcp_audit.py" and surface.endswith("guardrail"))
    }
    assert documented == expected, sorted(documented ^ expected)


def test_the_works_with_table_lists_every_audited_companion():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    for companion, _ in ROUTES.values():
        assert f"examples/{companion}" in table, companion


def test_negative_control_the_table_reader_sees_a_wrong_label(tmp_path):
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "`crewai.before_tool_call`" in text
    broken = text.replace("`crewai.before_tool_call`", "`crewai.some_other_hook`")
    rows = set(re.findall(r"^\|[^|]*\| `examples/(\w+_audit\.py)` \| `([\w.]+)` \|", broken, re.M))
    assert ("crewai_audit.py", "crewai.before_tool_call") not in rows
    assert ("crewai_audit.py", "crewai.some_other_hook") in rows
