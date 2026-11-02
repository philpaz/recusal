"""Keep the public API imports in documentation examples valid."""

import ast
import importlib
import re
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (ROOT / "docs" / "EXTENDING.md", ROOT / "docs" / "HOWTO.md")


class PythonBlock(NamedTuple):
    line: int
    code: str


def _python_blocks(document: Path):
    """Yield Python fenced blocks, including fences inside Markdown blockquotes."""
    fence_character = None
    fence_length = 0
    language = None
    start = 0
    code = []

    for number, raw_line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
        line = re.sub(r"^(?:> ?)+", "", raw_line)
        stripped = line.strip()
        if fence_character is None:
            match = re.match(r"(?P<fence>`{3,}|~{3,})(?P<info>.*)$", stripped)
            if match:
                marker = match.group("fence")
                fence_character = marker[0]
                fence_length = len(marker)
                info = match.group("info").strip()
                language = info.split(maxsplit=1)[0].lower() if info else ""
                start = number
                code = []
            continue

        if re.fullmatch(re.escape(fence_character) + "{" + str(fence_length) + r",}\s*", stripped):
            if language == "python":
                yield PythonBlock(start, "\n".join(code))
            fence_character = None
            fence_length = 0
            language = None
            continue

        code.append(line.rstrip())

    assert fence_character is None, f"Unclosed fence in {document.relative_to(ROOT)}"


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.name)
def test_python_documentation_blocks_compile_and_import_recusal_names(document):
    blocks = list(_python_blocks(document))
    assert blocks, f"No Python examples found in {document.relative_to(ROOT)}"

    for block in blocks:
        filename = f"{document.relative_to(ROOT)}:{block.line}"
        compile(block.code, filename, "exec")
        tree = ast.parse(block.code, filename=filename)

        for statement in tree.body:
            if not isinstance(statement, ast.ImportFrom) or not statement.module:
                continue
            if not statement.module.startswith("recusal"):
                continue

            module = importlib.import_module(statement.module)
            for imported in statement.names:
                if imported.name == "*":
                    continue
                assert hasattr(module, imported.name), (
                    f"{filename}: {statement.module} does not export {imported.name}"
                )


def test_reusable_policy_example_imports_gate_tool_use():
    document = ROOT / "docs" / "EXTENDING.md"
    block = next(block for block in _python_blocks(document) if "def write_policy" in block.code)
    tree = ast.parse(block.code, filename=f"{document.relative_to(ROOT)}:{block.line}")

    assert any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "recusal.claude"
        and any(imported.name == "gate_tool_use" for imported in statement.names)
        for statement in tree.body
    ), "the reusable policy example must import gate_tool_use from recusal.claude"
