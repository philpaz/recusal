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

DOCUMENTS = ("docs/ARCHITECTURE.md", "docs/HOWTO.md")


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
    companion, surfaces = ROUTES[gate]
    constants = _string_constants(EXAMPLES / companion)
    missing = sorted(set(surfaces) - constants)
    assert not missing, f"{companion} does not define the surface(s) {missing}"


@pytest.mark.parametrize("gate", sorted(ROUTES))
def test_the_gate_passes_recording_arguments_to_decide(gate):
    """One decision path: the plain gate's own call to ``decide`` carries ``**record``, so
    the audited route cannot be a second copy. The LangChain gate has no ``decide`` call of
    its own (it wraps the LangGraph hooks)."""
    source = (EXAMPLES / gate).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "decide"
    ]
    if gate == "langchain_agent_gate.py":
        assert calls == []
        return
    assert calls, gate
    for call in calls:
        assert any(k.arg is None for k in call.keywords), (
            f"{gate}:{call.lineno}: a decide(...) call takes no **record, so a decision made "
            "there cannot be put on an audit log"
        )


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
