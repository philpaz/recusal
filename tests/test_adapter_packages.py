"""The adapter packages under ``adapters/`` are separate distributions, and the kernel does
not know they exist.

No SDK is imported: these read the package metadata and sources. They hold the boundary the
separation exists for: ``recusal`` keeps zero runtime dependencies and imports nothing from
an adapter, each adapter depends on the kernel through its public surface only, and what an
adapter's wheel declares matches what is in the tree.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"

#: Every adapter package: directory, distribution name, import name, the framework's
#: distribution it must depend on.
PACKAGES = {
    "openai-agents-mcp": (
        "recusal-openai-agents-mcp",
        "recusal_openai_agents_mcp",
        "openai-agents",
    ),
    "crewai": ("recusal-crewai", "recusal_crewai", "crewai"),
    "langchain": ("recusal-langchain", "recusal_langchain", "langchain"),
    "langgraph": ("recusal-langgraph", "recusal_langgraph", "langgraph"),
    "openai-agents": ("recusal-openai-agents", "recusal_openai_agents", "openai-agents"),
    "google-adk": ("recusal-google-adk", "recusal_google_adk", "google-adk"),
    "openai": ("recusal-openai", "recusal_openai", "openai"),
    "pydantic-ai": ("recusal-pydantic-ai", "recusal_pydantic_ai", "pydantic-ai-slim"),
}

#: The kernel modules an adapter may import. ``recusal`` is the public package surface;
#: ``recusal.invocation`` is public from 0.14.0 (CHANGELOG.md) and ``recusal.audit`` is the
#: audit module the README documents.
KERNEL_IMPORTS = {"recusal", "recusal.audit", "recusal.invocation"}

#: Names an adapter may take from a kernel module that is otherwise off limits.
#: ``identity_getter`` is public from 0.14.0 (CHANGELOG.md) and lives beside the Claude
#: Code hook, whose other names an adapter has no business importing.
KERNEL_NAMES = {"recusal.claude_code": {"identity_getter", "findings_of"}}

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="tomllib is in the standard library from Python 3.11"
)


def _project(directory: str) -> dict:
    import tomllib

    return tomllib.loads((ADAPTERS / directory / "pyproject.toml").read_text(encoding="utf-8"))


def _sources(directory: str):
    return sorted((ADAPTERS / directory / "src").rglob("*.py"))


def _imports(path: Path) -> set:
    """Every absolute module a file imports, at any depth."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def test_every_adapter_directory_is_registered_here():
    on_disk = {path.parent.name for path in ADAPTERS.glob("*/pyproject.toml")}
    assert on_disk == set(PACKAGES), sorted(on_disk ^ set(PACKAGES))


def test_the_kernel_declares_no_dependency_and_names_no_adapter():
    import tomllib

    kernel = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert kernel["project"]["dependencies"] == []
    assert kernel["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["recusal"]
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for _, (distribution, module, _framework) in PACKAGES.items():
        assert distribution not in text and module not in text


def test_the_kernel_imports_nothing_from_an_adapter():
    modules = {module for _, module, _ in PACKAGES.values()}
    for path in sorted((ROOT / "recusal").rglob("*.py")):
        tops = {name.split(".")[0] for name in _imports(path)}
        assert not tops & modules, f"{path.name} imports an adapter package"
        assert "adapters" not in tops, path.name


@pytest.mark.parametrize("directory", sorted(PACKAGES))
def test_the_adapter_is_its_own_distribution_depending_on_the_kernel(directory):
    distribution, module, framework = PACKAGES[directory]
    project = _project(directory)["project"]
    assert project["name"] == distribution
    assert project["license"] == "Apache-2.0"
    requirements = project["dependencies"]
    kernel = [r for r in requirements if re.match(r"recusal(?![-\w])", r)]
    assert len(kernel) == 1, requirements
    # A floor at the release that made `decide` public, and a ceiling below the next major.
    assert re.fullmatch(r"recusal>=0\.\d+\.\d+,<1", kernel[0]), kernel[0]
    assert any(r.startswith(framework) for r in requirements), requirements
    for requirement in requirements:
        assert ">=" in requirement and "<" in requirement, f"{requirement} is not bounded"
    # an adapter that imports another adapter declares it as a requirement
    imported = {name.split(".")[0] for path in _sources(directory) for name in _imports(path)}
    for other, (other_dist, other_module, _f) in PACKAGES.items():
        if other != directory and other_module in imported:
            assert any(r.startswith(other_dist + ">=") for r in requirements), other_dist


@pytest.mark.parametrize("directory", sorted(PACKAGES))
def test_the_adapter_builds_with_the_kernels_pinned_backend(directory):
    import tomllib

    kernel = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert _project(directory)["build-system"] == kernel["build-system"]


@pytest.mark.parametrize("directory", sorted(PACKAGES))
def test_the_adapter_ships_the_repository_license(directory):
    license_file = ADAPTERS / directory / "LICENSE"
    assert license_file.read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert _project(directory)["project"]["license-files"] == ["LICENSE"]


@pytest.mark.parametrize("directory", sorted(PACKAGES))
def test_the_wheel_packages_the_module_the_metadata_names(directory):
    _, module, _ = PACKAGES[directory]
    hatch = _project(directory)["tool"]["hatch"]
    assert hatch["build"]["targets"]["wheel"]["packages"] == [f"src/{module}"]
    assert hatch["version"]["path"] == f"src/{module}/__init__.py"
    init = (ADAPTERS / directory / "src" / module / "__init__.py").read_text(encoding="utf-8")
    assert re.search(r'^__version__ = "\d+\.\d+\.\d+"$', init, re.M)


@pytest.mark.parametrize("directory", sorted(PACKAGES))
def test_every_exported_name_is_defined(directory):
    _, module, _ = PACKAGES[directory]
    tree = ast.parse(
        (ADAPTERS / directory / "src" / module / "__init__.py").read_text(encoding="utf-8")
    )
    exported = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "__all__"
    )
    bound = {"__version__"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            bound.update(alias.asname or alias.name for alias in node.names)
    assert set(exported) <= bound, sorted(set(exported) - bound)
    assert not [name for name in exported if name.startswith("_") and name != "__version__"]


@pytest.mark.parametrize("directory", sorted(PACKAGES))
def test_the_adapter_reaches_the_kernel_through_its_public_surface_only(directory):
    """No ``recusal._private`` module, no ``recusal.claude_code`` internals, and no import
    of the repository's examples or tests: an installed adapter has neither."""
    siblings = {module for key, (_d, module, _f) in PACKAGES.items() if key != directory}
    for path in _sources(directory):
        for name in _imports(path):
            top = name.split(".")[0]
            if top == "recusal" and name not in KERNEL_NAMES:
                assert name in KERNEL_IMPORTS, f"{path.name} imports {name}"
            assert top not in {"examples", "tests", "adapters"}, f"{path.name} imports {name}"
            # another adapter is reached through its public, top-level names only
            if top in siblings:
                assert name == top, f"{path.name} imports {name}, private to {top}"


@pytest.mark.parametrize("directory", sorted(PACKAGES))
def test_the_adapter_takes_only_the_named_public_names_from_a_guarded_kernel_module(
    directory,
):
    for path in _sources(directory):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in KERNEL_NAMES, f"{path.name}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module in KERNEL_NAMES:
                taken = {alias.name for alias in node.names}
                extra = sorted(taken - KERNEL_NAMES[node.module])
                assert not extra, f"{path.name} takes {extra} from {node.module}"


@pytest.mark.parametrize("directory", sorted(PACKAGES))
def test_the_adapter_has_no_default_policy_and_no_demo_code(directory):
    """What to refuse is the adopter's decision. A library gate that fell back to a
    demonstration policy would allow everything that policy does not name."""
    for path in _sources(directory):
        text = path.read_text(encoding="utf-8")
        assert "sql_scope_policy" not in text, path.name
        assert "EXECUTED" not in text and "def main(" not in text, path.name
        assert "sys.path" not in text and "SystemExit" not in text, path.name


@pytest.mark.parametrize("directory", sorted(PACKAGES))
def test_ci_builds_installs_and_smoke_tests_the_wheel(directory):
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert f"adapters/{directory}/smoke_test.py" in ci
    assert f"python -m build --no-isolation --outdir dist-adapters adapters/{directory}" in ci
    assert (ADAPTERS / directory / "smoke_test.py").is_file()
    assert (ADAPTERS / directory / "README.md").is_file()


def _table(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    return text.split("<!-- adapter-packages:start -->", 1)[1].split(
        "<!-- adapter-packages:end -->", 1
    )[0]


def test_the_architecture_table_lists_exactly_the_packages_and_their_requirements():
    rows = re.findall(
        r"^\| `([\w-]+)` \| `adapters/([\w-]+)` \| `(\w+)` \| `([^`]+)` \| (3\.\d+)\+ \|$",
        _table("docs/ARCHITECTURE.md"),
        re.M,
    )
    documented = {
        directory: (dist, module, req, python) for dist, directory, module, req, python in rows
    }
    assert set(documented) == set(PACKAGES), sorted(set(documented) ^ set(PACKAGES))
    for directory, (distribution, module, framework) in PACKAGES.items():
        project = _project(directory)["project"]
        dist, mod, requirement, python = documented[directory]
        assert (dist, mod) == (distribution, module)
        assert requirement in project["dependencies"], requirement
        assert requirement.startswith(framework)
        assert project["requires-python"] == f">={python}"


def test_the_readme_table_lists_exactly_the_packages():
    rows = re.findall(r"^\| \[`([\w-]+)`\]\(([^)]+)\) \|", _table("README.md"), re.M)
    expected = {
        (dist, f"https://github.com/philpaz/recusal/tree/main/adapters/{directory}")
        for directory, (dist, _module, _framework) in PACKAGES.items()
    }
    assert set(rows) == expected, sorted(set(rows) ^ expected)
    table = _table("README.md")
    for _dist, module, _framework in PACKAGES.values():
        assert f"from {module} import" in table, module


def test_negative_control_the_import_reader_sees_a_private_kernel_import(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("from recusal.claude_code import decide\nimport examples.x\n", "utf-8")
    assert _imports(probe) == {"recusal.claude_code", "examples.x"}
    assert not {"recusal.claude_code"} <= KERNEL_IMPORTS
