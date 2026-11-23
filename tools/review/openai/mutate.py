"""Mutation testing of ``recusal-openai``: switch one check off at a time, in a copy of
this checkout, and see whether the adapter's tests notice.

    python tools/review/openai/mutate.py --count      # how many, and that each applies
    python tools/review/openai/mutate.py --list       # every mutation, numbered
    python tools/review/openai/mutate.py --summary    # run them all; one line of counts
    python tools/review/openai/mutate.py              # the same, one line per mutation
    python tools/review/openai/mutate.py --only 0:10  # a slice, by number

What a mutation is. One small edit to the adapter that removes or weakens one check:
a condition replaced by ``False``, a bound moved by one, a value left out of a set, an
argument not passed on. Each is written out below as the exact text it replaces and the
text it puts there, in ``_gate.py``, ``_audit.py`` or the package's ``pyproject.toml``.
They were written by hand during the review, one or more for each check a review round
added and for each mutation an earlier round's tests had missed. The label of each says
where it comes from: no prefix is round one, ``R2`` round two, ``R3`` round three, and
``F1`` and the like are the finding numbers in the section comments of
``tests/test_openai_responses_gate.py``.

What "caught" means. For each mutation the two adapter test files are run against the
edited copy. If at least one test fails, the mutation is caught: the tests would notice
that check going missing. If every test still passes, it survived, which means no test
depends on that check, and that is a finding.

What it does not show. That every mutation here is caught says the tests depend on
these checks. It does not say the checks are right, or that a check nobody thought to
write a mutation for is tested. The mutations are a hand-written list, not every
possible edit. ``docs/REVIEW.md`` says what the review was and what it does not show.

The working tree is never written. The checkout is copied to a new temporary directory
(without ``.git`` and caches), every edit is made in the copy, the copy is removed at
the end, and the three files in the working tree are compared, byte for byte, with what
they were at the start. The run refuses to report if the tests do not pass on the
unedited copy, or if a mutation's text is no longer in the source.

A full run starts the test suite once per mutation and takes several minutes. The tests
need the OpenAI SDK: ``python -m pip install --require-hashes -r
requirements/adapter-openai.txt``.

Exit status: 0 if every selected mutation was caught, 1 if any survived, did not apply
or timed out, 2 if the unedited copy does not pass its tests.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Optional

TOOL = "tools/review/openai/mutate.py"
ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path("adapters") / "openai" / "src" / "recusal_openai"
GATE = PACKAGE / "_gate.py"
AUDIT = PACKAGE / "_audit.py"
PYPROJECT = Path("adapters") / "openai" / "pyproject.toml"
TESTS = ("tests/test_openai_responses_gate.py", "tests/test_openai_responses_audit.py")
NOT_COPIED = (
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "*.egg-info",
)
TIME_LIMIT = 900  # seconds for one run of the two test files


class Mutation(NamedTuple):
    """Replace occurrence number ``occurrence`` (from 0) of ``old`` with ``new`` in
    ``file``. ``count`` is how many times ``old`` must be in the file for the edit to mean
    what its label says: if the source has changed under it, the mutation does not apply."""

    file: Path
    label: str
    old: str
    new: str
    occurrence: int = 0
    count: int = 1


MUTATIONS = [
    # round one
    Mutation(GATE, "F1 parse_constant hook off", "parse_constant=_not_json, ", ""),
    Mutation(GATE, "F1 parse_float hook off", ", parse_float=_finite)", ")"),
    Mutation(
        GATE, "F1 _finite accepts infinity", "    if not math.isfinite(number):", "    if False:"
    ),
    Mutation(
        GATE,
        "F1 _not_json returns instead of raising",
        '    raise ValueError(f"{token} is not JSON")',
        "    return float(token)",
    ),
    Mutation(
        GATE,
        "F1 not-finite handler falls to ValueError text",
        "    except _NotFinite:",
        "    except ZeroDivisionError:",
    ),
    Mutation(
        GATE,
        "F3 RecursionError from the parser not caught",
        "    except RecursionError:\n        detail = _TOO_DEEP",
        "    except ZeroDivisionError:\n        detail = _TOO_DEEP",
    ),
    Mutation(GATE, "F2 depth check off", "    if _too_deep(args):", "    if False:"),
    Mutation(GATE, "F2 depth limit 65", "_MAX_DEPTH = 64", "_MAX_DEPTH = 65"),
    Mutation(GATE, "F2 depth limit 63", "_MAX_DEPTH = 64", "_MAX_DEPTH = 63"),
    Mutation(
        GATE, "F2 depth ignores lists", "        elif type(item) is list:", "        elif False:"
    ),
    Mutation(GATE, "F2 depth ignores dicts", "        if type(item) is dict:", "        if False:"),
    Mutation(
        GATE,
        "F2 too-deep refusal bound to decoded args",
        '        detail = _TOO_DEEP\n        return _unjudgeable(\n            shown, f"Recusal refused `{shown}`: {detail}", detail, arguments, record\n        )\n    if not isinstance(args, dict):',
        '        detail = _TOO_DEEP\n        return _unjudgeable(\n            shown, f"Recusal refused `{shown}`: {detail}", detail, args, record\n        )\n    if not isinstance(args, dict):',
    ),
    Mutation(
        GATE, "F8b non-string check off", "    if not isinstance(arguments, str):", "    if False:"
    ),
    Mutation(
        GATE,
        "F8d check off in judge_and_run",
        "answer_tool_errors, _recording(record, _CALL_KEYWORDS)\n",
        "answer_tool_errors, record\n",
    ),
    Mutation(
        GATE,
        "F8d check off in gate_function_calls",
        "    record = _recording(record)\n    unfinished =",
        "    unfinished =",
    ),
    Mutation(
        GATE,
        "F8d check off in gate_tool_calls",
        "    record = _recording(record)\n    chosen = _chosen(completion, choice)",
        "    chosen = _chosen(completion, choice)",
    ),
    Mutation(GATE, "F8d _check_record never raises", "    if unknown:", "    if False:"),
    Mutation(
        GATE,
        "F8d safe_errors accepted as a keyword",
        '_RECORD_KEYWORDS = frozenset({"audit", "surface", "actor", "control", "allow_on_pass"})',
        '_RECORD_KEYWORDS = frozenset({"audit", "surface", "actor", "control", "allow_on_pass", "safe_errors"})',
    ),
    Mutation(GATE, "F8c generator check off", "    if _is_generator(func):", "    if False:"),
    Mutation(
        GATE,
        "F8c generator __call__ not read",
        "    return inspect.isgeneratorfunction(target) or inspect.isgeneratorfunction(call)",
        "    return inspect.isgeneratorfunction(target)",
    ),
    Mutation(
        GATE,
        "F8c generator staticmethod not unwrapped",
        "        elif isinstance(target, staticmethod) or inspect.ismethod(target):",
        "        elif inspect.ismethod(target):",
    ),
    Mutation(
        GATE,
        "F8c returned generator answered as repr",
        "    if isinstance(result, (Iterator, AsyncIterator)):",
        "    if False:",
    ),
    Mutation(
        GATE,
        "F5 bind refusal quotes the TypeError",
        "    except TypeError:  # its text can quote a key the model chose, so it is not used\n        detail = _unbound(signature.parameters, args)",
        '    except TypeError as exc:\n        detail = f"arguments do not bind to the tool ({exc})"',
    ),
    Mutation(GATE, "F5 _unbound counts no extras", "    if extra:", "    if False:"),
    Mutation(GATE, "F5 _unbound names no missing", "    if missing:", "    if False:"),
    Mutation(
        GATE,
        "F5 _unbound names defaulted parameters as missing",
        "        if name not in args and parameter.default is parameter.empty",
        "        if name not in args",
    ),
    Mutation(GATE, "F5 _unbound always plural", "{'s' if extra != 1 else ''}", "s"),
    Mutation(
        GATE,
        "F1 policy judges the invocation itself, not a copy",
        "        judged = _detached(invocation)",
        "        judged = invocation",
    ),
    Mutation(
        GATE,
        "F2 detach TypeError not a recorded refusal",
        "    except (TypeError, RecursionError):",
        "    except (RecursionError,):",
    ),
    Mutation(
        GATE,
        "F2 detach RecursionError not a recorded refusal",
        "    except (TypeError, RecursionError):",
        "    except (TypeError,):",
    ),
    Mutation(
        GATE,
        "F4 namespace check off",
        '        if getattr(call, "namespace", None):',
        "        if False:",
    ),
    Mutation(
        GATE,
        "F8a response status ignored",
        '    unfinished = getattr(response, "status", None) not in _FUNCTION_CALL_FINISHED',
        "    unfinished = False",
    ),
    Mutation(
        GATE,
        "F8a unfinished response branch off",
        "        elif unfinished:",
        "        elif False:",
    ),
    Mutation(
        GATE,
        "F8a item status check off",
        '        elif getattr(call, "status", None) not in _FUNCTION_CALL_FINISHED:',
        "        elif False:",
    ),
    Mutation(
        GATE,
        "F8a incomplete counts as finished",
        '_FUNCTION_CALL_FINISHED = frozenset({None, "completed"})',
        '_FUNCTION_CALL_FINISHED = frozenset({None, "completed", "incomplete"})',
    ),
    Mutation(
        GATE,
        "F8a in_progress counts as finished",
        '_FUNCTION_CALL_FINISHED = frozenset({None, "completed"})',
        '_FUNCTION_CALL_FINISHED = frozenset({None, "completed", "in_progress"})',
    ),
    Mutation(
        GATE,
        "F8a failed counts as finished",
        '_FUNCTION_CALL_FINISHED = frozenset({None, "completed"})',
        '_FUNCTION_CALL_FINISHED = frozenset({None, "completed", "failed"})',
    ),
    Mutation(
        GATE,
        "F8a no status refused",
        '_FUNCTION_CALL_FINISHED = frozenset({None, "completed"})',
        '_FUNCTION_CALL_FINISHED = frozenset({"completed"})',
    ),
    Mutation(
        GATE,
        "F8a completed refused",
        '_FUNCTION_CALL_FINISHED = frozenset({None, "completed"})',
        "_FUNCTION_CALL_FINISHED = frozenset({None})",
    ),
    Mutation(
        GATE,
        "F8a chat finish_reason check off",
        '    if getattr(chosen, "finish_reason", None) not in _CHAT_FINISHED:',
        "    if False:",
    ),
    Mutation(
        GATE,
        "F8a chat length counts as finished",
        '_CHAT_FINISHED = frozenset({"stop", "tool_calls", "function_call"})',
        '_CHAT_FINISHED = frozenset({"stop", "tool_calls", "function_call", "length"})',
    ),
    Mutation(
        GATE,
        "F8a chat content_filter counts as finished",
        '_CHAT_FINISHED = frozenset({"stop", "tool_calls", "function_call"})',
        '_CHAT_FINISHED = frozenset({"stop", "tool_calls", "function_call", "content_filter"})',
    ),
    Mutation(
        GATE,
        "F8a chat missing reason counts as finished",
        '_CHAT_FINISHED = frozenset({"stop", "tool_calls", "function_call"})',
        '_CHAT_FINISHED = frozenset({"stop", "tool_calls", "function_call", None})',
    ),
    Mutation(
        GATE,
        "F8a chat stop refused",
        '_CHAT_FINISHED = frozenset({"stop", "tool_calls", "function_call"})',
        '_CHAT_FINISHED = frozenset({"tool_calls", "function_call"})',
    ),
    Mutation(
        GATE,
        "F8a chat tool_calls refused",
        '_CHAT_FINISHED = frozenset({"stop", "tool_calls", "function_call"})',
        '_CHAT_FINISHED = frozenset({"stop", "function_call"})',
    ),
    Mutation(
        GATE,
        "F8a chat function_call reason refused",
        '_CHAT_FINISHED = frozenset({"stop", "tool_calls", "function_call"})',
        '_CHAT_FINISHED = frozenset({"stop", "tool_calls"})',
    ),
    Mutation(
        GATE,
        "F6 legacy function_call check off",
        '    if getattr(chosen.message, "function_call", None):',
        "    if False:",
    ),
    Mutation(GATE, "loop refusal not answered", "    if refused is not None:", "    if False:"),
    # round two
    Mutation(
        GATE,
        "R2 F1 audit=None is not the plain gate",
        "    return record if audit is not None else {}\n",
        "    return record\n",
    ),
    Mutation(
        GATE,
        "R2 F1 a recording keyword without audit= is accepted",
        "    if needs:",
        "    if False:",
    ),
    Mutation(
        GATE,
        "R2 F1 the plain gate keeps the recording arguments",
        "        return {}\n    if audit is not None and record.get",
        "        return record\n    if audit is not None and record.get",
    ),
    Mutation(
        GATE,
        "R2 F1 audit= without surface= is not checked",
        '    if audit is not None and record.get("surface") is None:',
        "    if False:",
    ),
    Mutation(
        GATE,
        "R2 F1 the loops take tool_use_id",
        '_RECORD_KEYWORDS = frozenset({"audit", "surface", "actor", "control", "allow_on_pass"})',
        '_RECORD_KEYWORDS = frozenset({"audit", "surface", "actor", "control", "allow_on_pass", "tool_use_id"})',
    ),
    Mutation(
        GATE,
        "R2 fail_closed is taken again",
        '_RECORD_KEYWORDS = frozenset({"audit", "surface", "actor", "control", "allow_on_pass"})',
        '_RECORD_KEYWORDS = frozenset({"audit", "surface", "actor", "control", "allow_on_pass", "fail_closed"})',
    ),
    Mutation(
        GATE,
        "R2 F1 judge_and_run does not take tool_use_id",
        '_CALL_KEYWORDS = _RECORD_KEYWORDS | {"tool_use_id"}',
        "_CALL_KEYWORDS = _RECORD_KEYWORDS",
    ),
    Mutation(
        GATE,
        "R2 F1 surface does not need audit",
        '_NEED_AUDIT = ("surface", "tool_use_id", "actor", "control")',
        '_NEED_AUDIT = ("tool_use_id", "actor", "control")',
    ),
    Mutation(
        GATE,
        "R2 F1 tool_use_id does not need audit",
        '_NEED_AUDIT = ("surface", "tool_use_id", "actor", "control")',
        '_NEED_AUDIT = ("surface", "actor", "control")',
    ),
    Mutation(
        GATE,
        "R2 F1 actor does not need audit",
        '_NEED_AUDIT = ("surface", "tool_use_id", "actor", "control")',
        '_NEED_AUDIT = ("surface", "tool_use_id", "control")',
    ),
    Mutation(
        GATE,
        "R2 F1 control does not need audit",
        '_NEED_AUDIT = ("surface", "tool_use_id", "actor", "control")',
        '_NEED_AUDIT = ("surface", "tool_use_id", "actor")',
    ),
    Mutation(
        GATE,
        "R2 F1 a keyword that is not taken is not explained",
        "            f\"{f' ({why})' if why else ''}; the \"",
        '            "; the "',
    ),
    Mutation(
        GATE,
        "R2 F2 only a generator is answered as an iterator",
        "    if isinstance(result, (Iterator, AsyncIterator)):",
        "    if inspect.isgenerator(result):",
    ),
    Mutation(
        GATE,
        "R2 F2 an async generator is answered as its repr",
        "    if isinstance(result, (Iterator, AsyncIterator)):",
        "    if isinstance(result, Iterator):",
    ),
    Mutation(
        GATE,
        "R2 F2 every iterable is answered as an iterator",
        "from collections.abc import AsyncIterator, Iterator\n",
        "from collections.abc import AsyncIterator\nfrom collections.abc import Iterable as Iterator\n",
    ),
    Mutation(
        GATE,
        "R2 F2 an async generator is answered as a coroutine not run",
        "    if inspect.iscoroutine(result):\n        if inspect.getcoroutinestate(result)",
        "    if inspect.iscoroutine(result) or inspect.isasyncgen(result):\n        if inspect.iscoroutine(result) and inspect.getcoroutinestate(result)",
    ),
    Mutation(
        GATE,
        "R2 F2 a generator is called an iterator",
        '            "a generator"\n            if inspect.isgenerator(result)',
        '            "an iterator"\n            if inspect.isgenerator(result)',
    ),
    Mutation(
        GATE,
        "R2 F2 an async generator is called an iterator",
        '            else "an async generator"',
        '            else "an iterator"',
    ),
    Mutation(
        GATE,
        "R2 F3 lru_cache is not looked through",
        "        elif isinstance(target, _LRU_CACHE):",
        "        elif False:",
    ),
    Mutation(
        GATE,
        "R2 F3 a bound method is not looked through",
        "        elif isinstance(target, staticmethod) or inspect.ismethod(target):",
        "        elif isinstance(target, staticmethod):",
    ),
    Mutation(
        GATE,
        "R2 F3 a partial is not looked through",
        "        if isinstance(target, functools.partial):",
        "        if False:",
    ),
    Mutation(
        GATE,
        "R2 F3 the async check does not read through wrappers",
        "    target = _declared(func)\n    if inspect.iscoroutinefunction(target)",
        "    target = func\n    if inspect.iscoroutinefunction(target)",
    ),
    Mutation(
        GATE,
        "R2 F3 the generator check does not read through wrappers",
        "    target = _declared(func)\n    call = _declared(",
        "    target = func\n    call = _declared(",
    ),
    Mutation(
        GATE,
        "R2 F3 a functools.wraps wrapper is looked through",
        "        else:\n            break\n    return target",
        '        elif getattr(target, "__wrapped__", None) is not None:\n            target = target.__wrapped__\n        else:\n            break\n    return target',
    ),
    Mutation(
        GATE, "R2 F4 cached check off", "    if _cached(func) and any(", "    if False and any("
    ),
    Mutation(
        GATE,
        "R2 F4 an object is not counted",
        "any(type(value) in (list, dict) for value",
        "any(type(value) in (list,) for value",
    ),
    Mutation(
        GATE,
        "R2 F4 a list is not counted",
        "any(type(value) in (list, dict) for value",
        "any(type(value) in (dict,) for value",
    ),
    Mutation(
        GATE,
        "R2 F4 a cached method is not seen",
        "        if isinstance(target, staticmethod) or inspect.ismethod(target):\n            target = target.__func__\n        else:\n            break\n    return isinstance(target, _LRU_CACHE)",
        "        if isinstance(target, staticmethod):\n            target = target.__func__\n        else:\n            break\n    return isinstance(target, _LRU_CACHE)",
    ),
    Mutation(
        GATE,
        "R2 F4 every tool is treated as cached",
        "    return isinstance(target, _LRU_CACHE)",
        "    return True",
    ),
    Mutation(
        GATE,
        "R2 F5 an unknown response status counts as finished",
        '    unfinished = getattr(response, "status", None) not in _FUNCTION_CALL_FINISHED',
        '    unfinished = getattr(response, "status", None) in {"in_progress", "incomplete", "failed", "cancelled", "queued"}',
    ),
    Mutation(
        GATE,
        "R2 F5 an unknown item status counts as finished",
        '        elif getattr(call, "status", None) not in _FUNCTION_CALL_FINISHED:',
        '        elif getattr(call, "status", None) in {"in_progress", "incomplete"}:',
    ),
    Mutation(
        GATE,
        "R2 F5 an unknown finish_reason counts as finished",
        '    if getattr(chosen, "finish_reason", None) not in _CHAT_FINISHED:',
        '    if getattr(chosen, "finish_reason", None) in {"length", "content_filter", None}:',
    ),
    Mutation(
        GATE,
        "R2 F5 a queued response counts as finished",
        '    unfinished = getattr(response, "status", None) not in _FUNCTION_CALL_FINISHED',
        '    unfinished = getattr(response, "status", None) not in {None, "completed", "queued"}',
    ),
    Mutation(
        GATE,
        "R2 F5 a cancelled response counts as finished",
        '    unfinished = getattr(response, "status", None) not in _FUNCTION_CALL_FINISHED',
        '    unfinished = getattr(response, "status", None) not in {None, "completed", "cancelled"}',
    ),
    Mutation(
        GATE,
        "R2 F6 a name that is not a string is looked up",
        "    if not isinstance(name, str) or name not in tools:",
        "    if name not in tools:",
    ),
    Mutation(
        GATE,
        "R2 F6 the name check is off where the name is shown",
        '    if not isinstance(name, str):\n        return "<a tool name that is not a string>"\n',
        "",
    ),
    Mutation(
        GATE,
        "R2 name: a backtick is shown",
        '    if 0 < len(name) <= _NAME_SHOWN and name.isprintable() and "`" not in name:',
        "    if 0 < len(name) <= _NAME_SHOWN and name.isprintable():",
    ),
    Mutation(
        GATE,
        "R2 name: a character that does not print is shown",
        '    if 0 < len(name) <= _NAME_SHOWN and name.isprintable() and "`" not in name:',
        '    if 0 < len(name) <= _NAME_SHOWN and "`" not in name:',
    ),
    Mutation(
        GATE,
        "R2 name: no bound",
        '    if 0 < len(name) <= _NAME_SHOWN and name.isprintable() and "`" not in name:',
        '    if 0 < len(name) and name.isprintable() and "`" not in name:',
    ),
    Mutation(
        GATE,
        "R2 name: an empty name is shown",
        '    if 0 < len(name) <= _NAME_SHOWN and name.isprintable() and "`" not in name:',
        '    if len(name) <= _NAME_SHOWN and name.isprintable() and "`" not in name:',
    ),
    Mutation(GATE, "R2 name: bound 65", "_NAME_SHOWN = 64", "_NAME_SHOWN = 65"),
    Mutation(GATE, "R2 name: bound 63", "_NAME_SHOWN = 64", "_NAME_SHOWN = 63"),
    Mutation(
        GATE,
        "R2 name: a known tool's name is described too",
        "    if name in tools:\n        return name\n",
        "",
    ),
    Mutation(
        GATE,
        "R2 name: the loop's refusal quotes the name raw",
        "        shown = _shown(name, tools)\n        return _unjudgeable(",
        "        shown = name\n        return _unjudgeable(",
    ),
    Mutation(
        GATE,
        "R2 name: judge_and_run quotes the name raw",
        "    shown = _shown(name, tools)\n    if not isinstance(arguments, str):",
        "    shown = str(name)\n    if not isinstance(arguments, str):",
    ),
    Mutation(
        GATE,
        "R2 name: the unknown-tool refusal quotes the name raw",
        '            shown, f"Recusal refused: unknown tool {shown!r}", "unknown tool", args, record',
        '            str(name), f"Recusal refused: unknown tool {name!r}", "unknown tool", args, record',
    ),
    Mutation(
        GATE,
        "R2 name: a missing id quotes the name raw (Responses)",
        'f"function_call {_shown(call.name, tools)!r}"',
        'f"function_call {call.name!r}"',
    ),
    Mutation(
        GATE,
        "R2 name: a missing id quotes the name raw (Chat)",
        'f"tool call {_shown(call.function.name, tools)!r}"',
        'f"tool call {call.function.name!r}"',
    ),
    Mutation(
        GATE,
        "R2 the depth check recurses",
        "    pending = [(value, 0)]\n    while pending:\n        item, depth = pending.pop()\n        if depth > _MAX_DEPTH:\n            return True\n        if type(item) is dict:\n            pending.extend((inner, depth + 1) for inner in item.values())\n        elif type(item) is list:\n            pending.extend((inner, depth + 1) for inner in item)\n    return False\n",
        "    def walk(item, depth):\n        if depth > _MAX_DEPTH:\n            return True\n        if type(item) in (dict, list):\n            inner = item.values() if type(item) is dict else item\n            return any(walk(each, depth + 1) for each in inner)\n        return False\n\n    return walk(value, 0)\n",
    ),
    Mutation(
        GATE,
        "R2 depth: > becomes >=",
        "        if depth > _MAX_DEPTH:",
        "        if depth >= _MAX_DEPTH:",
    ),
    # round three
    Mutation(
        PYPROJECT, "R3 F1 the floor is 0.14.0 again", '"recusal>=0.15.0,<1"', '"recusal>=0.14.0,<1"'
    ),
    Mutation(
        GATE,
        "R3 F1 the package imports beside an older kernel",
        'if not hasattr(_invocation, "reached_parameters"):',
        "if False:",
    ),
    Mutation(
        GATE,
        "R3 F2 an error from the signature check leaves the gate",
        "    except Exception:  # noqa: BLE001, a signature the kernel cannot read",
        "    except ZeroDivisionError:  # noqa: BLE001, a signature the kernel cannot read",
    ),
    Mutation(
        GATE,
        "R3 F2 only a ValueError from the signature check is a refusal",
        "    except Exception:  # noqa: BLE001, a signature the kernel cannot read",
        "    except ValueError:  # noqa: BLE001, a signature the kernel cannot read",
    ),
    Mutation(
        GATE,
        "R3 F2 a new reason from the signature check is ignored",
        '    if why is not None:\n        detail = f"its signature cannot be judged ({why})"',
        '    if why is not None and why.startswith("`"):\n        detail = f"its signature cannot be judged ({why})"',
    ),
    Mutation(
        GATE,
        "R3 F4 an explicit audit=None is read as absent",
        "    if audit is _ABSENT:",
        "    if audit is _ABSENT or audit is None:",
    ),
    Mutation(
        GATE,
        "R3 F4 an absent audit is read as None",
        '    audit = record.get("audit", _ABSENT)',
        '    audit = record.get("audit", None)',
    ),
    Mutation(
        AUDIT,
        "R3 F4 audited_function_calls takes log=None",
        '    _require_log(log, "audited_function_calls", "gate_function_calls", "SURFACE_RESPONSES")\n',
        "",
    ),
    Mutation(
        AUDIT,
        "R3 F4 audited_tool_calls takes log=None",
        '    _require_log(log, "audited_tool_calls", "gate_tool_calls", "SURFACE_CHAT")\n',
        "",
    ),
    Mutation(
        AUDIT, "R3 F4 the audited functions take log=None", "    if log is None:", "    if False:"
    ),
    Mutation(
        GATE,
        "R3 F5 recording values are not checked",
        "    _check_values(record)\n    return record",
        "    return record",
    ),
    Mutation(
        GATE,
        "R3 F5 recording values are checked only with a log",
        "    _check_values(record)\n    return record",
        "    if audit is not None:\n        _check_values(record)\n    return record",
    ),
    Mutation(
        GATE,
        "R3 F5 recording values are checked only without a log",
        "    _check_values(record)\n    return record",
        "    if audit is None:\n        _check_values(record)\n    return record",
    ),
    Mutation(
        GATE,
        "R3 F5 surface is not checked",
        '_CHECKED_BY_DECIDE = ("surface", "control", "tool_use_id")',
        '_CHECKED_BY_DECIDE = ("control", "tool_use_id")',
    ),
    Mutation(
        GATE,
        "R3 F5 control is not checked",
        '_CHECKED_BY_DECIDE = ("surface", "control", "tool_use_id")',
        '_CHECKED_BY_DECIDE = ("surface", "tool_use_id")',
    ),
    Mutation(
        GATE,
        "R3 F5 tool_use_id is not checked",
        '_CHECKED_BY_DECIDE = ("surface", "control", "tool_use_id")',
        '_CHECKED_BY_DECIDE = ("surface", "control")',
    ),
    Mutation(
        GATE,
        "R3 F5 only a ValueError from decide refuses the value",
        "        except Exception as exc:  # noqa: BLE001, whatever decide raises",
        "        except ValueError as exc:  # noqa: BLE001, whatever decide raises",
    ),
    Mutation(
        GATE,
        "R3 F5 the refusal does not name the keyword",
        'f"Recusal refused the turn: {keyword}= is not a value recusal.decide "',
        'f"Recusal refused the turn: a keyword is not a value recusal.decide "',
    ),
    Mutation(
        GATE,
        "R3 F5 the question is put to the caller's log",
        '        asked: dict[str, Any] = {"audit": _NoLog(), "surface": _CHECK_SURFACE, keyword: value}',
        '        asked: dict[str, Any] = {"audit": record.get("audit") or _NoLog(), "surface": _CHECK_SURFACE, keyword: value}',
    ),
    Mutation(
        GATE,
        "R3 F5 the surface handed over with another keyword is not one decide accepts",
        '_CHECK_SURFACE = "openai.keyword_check"',
        '_CHECK_SURFACE = ""',
    ),
    Mutation(
        GATE,
        "R3 F5 the caller's surface is asked about with every keyword",
        '        asked: dict[str, Any] = {"audit": _NoLog(), "surface": _CHECK_SURFACE, keyword: value}',
        '        asked: dict[str, Any] = {"audit": _NoLog(), "surface": record.get("surface"), keyword: value}',
    ),
    Mutation(
        GATE,
        "R3 F5 a tool_use_id of any length is taken",
        "    if isinstance(call_id, str) and len(call_id) > _ID_MAX:",
        "    if False:",
    ),
    Mutation(
        GATE,
        "R3 a: arguments that are not JSON data go to decide as they are",
        "    if not _json_data(proposal):\n        proposal = _NOT_JSON_DATA\n",
        "",
    ),
    Mutation(
        GATE,
        "R3 a: every refused proposal is bound to the stand-in",
        "    if not _json_data(proposal):\n",
        "    if True:\n",
    ),
    Mutation(
        GATE,
        "R3 b: a call id of any length is taken",
        "    if len(call_id) > _ID_MAX:",
        "    if False:",
    ),
    Mutation(GATE, "R3 b: bound 257", "_ID_MAX = 256", "_ID_MAX = 257"),
    Mutation(GATE, "R3 b: bound 255", "_ID_MAX = 256", "_ID_MAX = 255"),
    Mutation(
        GATE,
        "R3 b: an id that does not print is quoted",
        '    if 0 < len(call_id) <= _ID_MAX and call_id.isprintable() and "`" not in call_id:',
        '    if 0 < len(call_id) <= _ID_MAX and "`" not in call_id:',
    ),
    Mutation(
        GATE,
        "R3 b: an id with a backtick is quoted",
        '    if 0 < len(call_id) <= _ID_MAX and call_id.isprintable() and "`" not in call_id:',
        "    if 0 < len(call_id) <= _ID_MAX and call_id.isprintable():",
    ),
    Mutation(
        GATE,
        "R3 b: an id of any length is quoted",
        '    if 0 < len(call_id) <= _ID_MAX and call_id.isprintable() and "`" not in call_id:',
        '    if 0 < len(call_id) and call_id.isprintable() and "`" not in call_id:',
    ),
    Mutation(
        GATE,
        "R3 b: an empty id is quoted",
        '    if 0 < len(call_id) <= _ID_MAX and call_id.isprintable() and "`" not in call_id:',
        '    if len(call_id) <= _ID_MAX and call_id.isprintable() and "`" not in call_id:',
    ),
    Mutation(
        GATE,
        "R3 b: an id that is not a string is quoted",
        '    if not isinstance(call_id, str):\n        return "<a call id that is not a string>"\n',
        "    if not isinstance(call_id, str):\n        return repr(call_id)\n",
    ),
    Mutation(
        GATE,
        "R3 b: a message names every id",
        "for call_id in call_ids[:4])",
        "for call_id in call_ids)",
    ),
    Mutation(
        GATE,
        "R3 b: a message does not say how many more",
        "    more = len(call_ids) - 4\n",
        "    more = 0\n",
    ),
    Mutation(
        GATE,
        "R3 b: the duplicate-id refusal lists the ids raw",
        "duplicate call ids {_repeated(call_ids)}",
        "duplicate call ids {call_ids}",
        0,
        2,
    ),
    Mutation(
        GATE,
        "R3 b: the answered-once refusal lists the calls raw",
        "calls {_ids_shown(sorted(call_ids))} but",
        "calls {sorted(call_ids)} but",
    ),
    Mutation(
        GATE,
        "R3 b: the answered-once refusal lists the answers raw",
        '{_ids_shown(sorted(answered))}"',
        '{sorted(answered)}"',
    ),
    Mutation(
        GATE,
        "R3 b: an id is named once per repeat",
        "        if call_id in seen and call_id not in repeated:",
        "        if call_id in seen:",
    ),
    Mutation(
        GATE,
        "R3 c: a staticmethod around a cache is not seen",
        "        if isinstance(target, staticmethod) or inspect.ismethod(target):\n            target = target.__func__\n        else:\n            break\n    return isinstance(target, _LRU_CACHE)",
        "        if inspect.ismethod(target):\n            target = target.__func__\n        else:\n            break\n    return isinstance(target, _LRU_CACHE)",
    ),
    Mutation(
        GATE,
        "R3 c: one level only",
        "    for _ in range(16):  # a bound, not a rule: these wrappers do not nest this deep\n        if isinstance(target, staticmethod) or",
        "    for _ in range(1):  # a bound, not a rule: these wrappers do not nest this deep\n        if isinstance(target, staticmethod) or",
    ),
    Mutation(
        GATE,
        "R3 [59] the tool is run with the model's arguments",
        "    if not answer_tool_errors:\n        return _answer(name, func(**invocation))",
        "    if not answer_tool_errors:\n        return _answer(name, func(**args))",
    ),
    Mutation(
        GATE,
        "R3 [60] the tool is run with the model's arguments (answer_tool_errors)",
        "    try:\n        return _answer(name, func(**invocation))",
        "    try:\n        return _answer(name, func(**args))",
    ),
    Mutation(
        GATE,
        "R3 [51] only the last value of an object is walked",
        "            pending.extend((inner, depth + 1) for inner in item.values())",
        "            pending.extend((inner, depth + 1) for inner in list(item.values())[-1:])",
    ),
    Mutation(
        GATE,
        "R3 [51b] only the first value of an object is walked",
        "            pending.extend((inner, depth + 1) for inner in item.values())",
        "            pending.extend((inner, depth + 1) for inner in list(item.values())[:1])",
    ),
    Mutation(
        GATE,
        "R3 [51c] only the last item of a list is walked",
        "            pending.extend((inner, depth + 1) for inner in item)",
        "            pending.extend((inner, depth + 1) for inner in item[-1:])",
    ),
    Mutation(
        AUDIT,
        "R3 [63] audited_tool_calls does not pass control",
        "        control=control,\n",
        "",
        1,
        2,
    ),
    Mutation(
        AUDIT,
        "R3 audited_function_calls does not pass control",
        "        control=control,\n",
        "",
        0,
        2,
    ),
    Mutation(
        AUDIT, "R3 audited_tool_calls does not pass actor", "        actor=actor,\n", "", 1, 2
    ),
    Mutation(
        GATE,
        "R3 [26] a log that is falsy is no log (surface)",
        '    if audit is not None and record.get("surface") is None:',
        '    if audit and record.get("surface") is None:',
    ),
    Mutation(
        GATE,
        "R3 [26] a log that is falsy is no log (record)",
        "    return record if audit is not None else {}\n",
        "    return record if audit else {}\n",
    ),
]


def not_applied(root: Path = ROOT) -> List[str]:
    """The labels of the mutations whose text is not in the source the stated number of
    times. Reads the files and runs nothing."""
    texts = {
        path: (root / path).read_text(encoding="utf-8") for path in {m.file for m in MUTATIONS}
    }
    return [m.label for m in MUTATIONS if texts[m.file].count(m.old) != m.count]


def counts() -> dict:
    """How many mutations there are, by file, and how many apply. Runs nothing."""
    missing = not_applied()
    return {
        "tool": TOOL,
        "mutations": len(MUTATIONS),
        "in_gate": sum(1 for m in MUTATIONS if m.file == GATE),
        "in_audit": sum(1 for m in MUTATIONS if m.file == AUDIT),
        "in_pyproject": sum(1 for m in MUTATIONS if m.file == PYPROJECT),
        "not_applied": len(missing),
    }


def _mutated(text: str, mutation: Mutation) -> str:
    start = -1
    for _ in range(mutation.occurrence + 1):
        start = text.index(mutation.old, start + 1)
    return text[:start] + mutation.new + text[start + len(mutation.old) :]


def _run_tests(copy: Path) -> tuple:
    """Run the two adapter test files in ``copy``: ``(result, first failing test)``,
    where result is "passed", "failed" or "timed out"."""
    shutil.rmtree(copy / PACKAGE / "__pycache__", ignore_errors=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    env.pop("PYTHONPATH", None)  # the tests find the copy by themselves; nothing else may
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS]
    try:
        done = subprocess.run(
            command,
            cwd=str(copy),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIME_LIMIT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timed out", ""
    lines = done.stdout.strip().splitlines()
    failed = next((line for line in lines if line.startswith(("FAILED ", "ERROR "))), "")
    if done.returncode == 0:
        return "passed", ""
    return "failed", failed.split(" - ", 1)[0] or (
        lines[-1] if lines else done.stderr.strip()[-200:]
    )


def run(selected: Optional[List[int]] = None, verbose: bool = False) -> dict:
    """Run the selected mutations (numbers into ``MUTATIONS``; all of them by default) in
    a temporary copy of the checkout and return the counts."""
    numbers = list(range(len(MUTATIONS))) if selected is None else list(selected)
    files = sorted({m.file for m in MUTATIONS})
    before = {path: (ROOT / path).read_bytes() for path in files}
    summary: dict = {"tool": TOOL, "mutations": len(MUTATIONS), "selected": len(numbers)}
    summary.update({"caught": 0, "survived": 0, "not_applied": 0, "timed_out": 0})
    survivors: List[str] = []
    holder = Path(tempfile.mkdtemp(prefix="recusal-mutate-"))
    copy = holder / "copy"
    try:
        shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(*NOT_COPIED))
        result, failed = _run_tests(copy)
        summary["baseline"] = result
        if verbose:
            print(f"baseline, nothing edited: {result} {failed}", flush=True)
        if result == "passed":
            for number in numbers:
                mutation = MUTATIONS[number]
                text = before[mutation.file].decode("utf-8")
                if text.count(mutation.old) != mutation.count:
                    outcome, failed = "not_applied", ""
                else:
                    edited = _mutated(text, mutation)
                    assert edited != text, mutation.label
                    (copy / mutation.file).write_bytes(edited.encode("utf-8"))
                    try:
                        result, failed = _run_tests(copy)
                    finally:
                        (copy / mutation.file).write_bytes(before[mutation.file])
                    outcome = {"failed": "caught", "passed": "survived"}.get(result, "timed_out")
                summary[outcome] += 1
                if outcome == "survived":
                    survivors.append(mutation.label)
                if verbose:
                    print(f"{outcome:<11} [{number}] {mutation.label}  {failed}", flush=True)
    finally:
        shutil.rmtree(holder, ignore_errors=True)
    summary["survivors"] = survivors
    summary["working_tree_unchanged"] = all(
        (ROOT / path).read_bytes() == data for path, data in before.items()
    )
    return summary


def _numbers(only: Optional[str]) -> Optional[List[int]]:
    """``A:B`` as the numbers from A up to, not including, B; either end may be left out."""
    if only is None:
        return None
    start, _, stop = only.partition(":")
    chosen = range(len(MUTATIONS))[int(start) if start else 0 : int(stop) if stop else None]
    return list(chosen)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mutation testing of recusal-openai: switch one check off at a time in a "
            "temporary copy of this checkout and run the adapter's tests against it. "
            "Prints one JSON line of counts; exits 1 if any mutation survived."
        ),
        epilog="See the top of this file for what a mutation is, and docs/REVIEW.md.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="print how many mutations there are and how many apply; run nothing",
    )
    parser.add_argument("--list", action="store_true", help="print every mutation, numbered")
    parser.add_argument(
        "--summary", action="store_true", help="print the line of counts and nothing else"
    )
    parser.add_argument(
        "--only", metavar="A:B", help="run mutations number A up to, not including, B"
    )
    options = parser.parse_args(argv)
    if options.list:
        for number, mutation in enumerate(MUTATIONS):
            print(f"[{number}] {mutation.file.name}: {mutation.label}")
        return 0
    if options.count:
        found = counts()
        print(json.dumps(found))
        return 1 if found["not_applied"] else 0
    try:
        numbers = _numbers(options.only)
    except ValueError:
        parser.error("--only takes two numbers with a colon between them, such as 0:10")
    summary = run(numbers, verbose=not options.summary)
    print(json.dumps(summary))
    if summary["baseline"] != "passed":
        return 2
    clean = summary["caught"] == summary["selected"] and summary["working_tree_unchanged"]
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
