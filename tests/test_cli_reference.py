"""docs/CLI.md names every flag of every command the parser defines, in that command's
section, so a flag cannot ship undocumented."""

import argparse
import re
from pathlib import Path

import pytest

from recusal.__main__ import main

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "CLI.md"


class _Built(Exception):
    pass


def _parser(monkeypatch):
    """The parser ``main`` builds, taken at the moment it would parse."""
    built = {}

    def capture(self, args=None, namespace=None):
        built["parser"] = self
        raise _Built

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(_Built):
        main([])
    return built["parser"]


def _flags(parser, path=()):
    """Yield (command path, flag) for every option of the parser and its subcommands."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                yield from _flags(child, (*path, name))
        elif isinstance(action, argparse._HelpAction):
            continue
        else:
            for flag in action.option_strings:
                if flag.startswith("--"):
                    yield path, flag


def _sections():
    """Each ``## `` heading's text, mapped to the rows of the tables under it."""
    text = REFERENCE.read_text(encoding="utf-8")
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    sections = {"": parts[0]}
    sections.update(zip(parts[1::2], parts[2::2]))
    return {
        heading: re.findall(r"^\| `(--[a-z-]+)", body, re.M) for heading, body in sections.items()
    }


def _section_for(path, sections):
    if not path:
        return ""
    command = " ".join(path)
    matches = [heading for heading in sections if f"`{command}`" in heading]
    assert len(matches) == 1, (command, sorted(sections))
    return matches[0]


def test_every_flag_has_a_row_in_its_commands_section(monkeypatch):
    sections = _sections()
    found = list(_flags(_parser(monkeypatch)))
    assert len(found) >= 30, found
    missing = [
        f"{' '.join(path) or '(top level)'} {flag}"
        for path, flag in found
        if flag not in sections[_section_for(path, sections)]
    ]
    assert not missing, missing


def test_no_row_names_a_flag_the_parser_does_not_have(monkeypatch):
    real = {flag for _, flag in _flags(_parser(monkeypatch))}
    documented = {flag for rows in _sections().values() for flag in rows}
    assert documented <= real, sorted(documented - real)


def test_every_command_has_a_section(monkeypatch):
    sections = _sections()
    leaves = {path for path, _ in _flags(_parser(monkeypatch)) if path}
    for path in sorted(leaves):
        _section_for(path, sections)
