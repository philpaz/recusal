"""Keep editor defaults aligned with repository-enforced formatting rules."""

import configparser
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITORCONFIG = ROOT / ".editorconfig"
PYPROJECT = ROOT / "pyproject.toml"


def _editorconfig() -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    # EditorConfig permits `root = true` before its first section; prefix a
    # synthetic default section so the standard-library INI parser can read it.
    config.read_string("[DEFAULT]\n" + EDITORCONFIG.read_text(encoding="utf-8"))
    return config


def _ruff_line_length() -> int:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^line-length\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    assert match, "pyproject.toml must define [tool.ruff] line-length"
    return int(match.group(1))


def test_editorconfig_matches_repository_formatting_rules():
    config = _editorconfig()

    assert config.defaults()["root"] == "true"
    assert config["*"]["charset"] == "utf-8"
    assert config["*"]["end_of_line"] == "lf"
    assert config["*"]["insert_final_newline"] == "true"
    assert config["*"]["indent_style"] == "space"
    assert config["*.py"]["indent_size"] == "4"
    assert int(config["*.py"]["max_line_length"]) == _ruff_line_length()
    assert config["*.{yml,yaml,json}"]["indent_size"] == "2"
