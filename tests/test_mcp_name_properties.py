"""Property tests for ``plugin_callable_name`` and ``split_mcp_tool_name``.

Generative locks over the two helpers: the invariants below must hold for every
input the strategies produce, not just hand-picked examples. ``derandomize=True``
keeps every run (local and CI, any platform) on the same deterministic example
stream. The settings match ``tests/test_kernel_properties.py``.

``plugin_callable_name`` replaces any character outside ``A-Z a-z 0-9 _ -`` with
``_``. That class is duplicated here on purpose: if the helper's rule drifts,
this file fails. The replacement is idempotent, and every output character is
inside the class.

``split_mcp_tool_name`` returns None for anything that does not start with
``mcp__`` ("None if not MCP"). A name built as ``mcp__{server}__{tool}``
round-trips when both parts are nonempty, neither contains ``__``, and
``server`` does not end in ``_``. A trailing ``_`` merges with the separator, so
``mcp__a___b`` is the flat name of both ``(server a_, tool b)`` and
``(server a, tool _b)``. ``manifest_policy`` documents the same flat-namespace
limit for ``mcp__a__b__c``: the only string both ``(server a, tool b__c)`` and
``(server a__b, tool c)`` can produce. This file does not pin a round trip for
that string.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from recusal.mcp import plugin_callable_name, split_mcp_tool_name

DETERMINISTIC = settings(derandomize=True, max_examples=150)

# Documented callable-safe set from plugin_callable_name: A-Z a-z 0-9 _ -
_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")

names = st.text(max_size=32)


def _no_double_underscore(text: str) -> bool:
    return "__" not in text


def _not_mcp(text: str) -> bool:
    return not text.startswith("mcp__")


# A trailing "_" on the server is the separator, so the flat name is not unique.
servers = st.text(min_size=1, max_size=24).filter(
    lambda text: _no_double_underscore(text) and not text.endswith("_")
)
tools = st.text(min_size=1, max_size=24).filter(_no_double_underscore)
# Plain text almost never contains two separators, which is the only input that
# reaches the "mcp" prefix check. Mix in names built from arbitrary pieces so
# that check is actually exercised.
_piece = st.text(min_size=1, max_size=16)
separated_not_mcp = (
    st.tuples(_piece, _piece, _piece).map(lambda parts: "__".join(parts)).filter(_not_mcp)
)
not_mcp_names = st.one_of(names.filter(_not_mcp), separated_not_mcp)


def _documented_callable_name(raw: str) -> str:
    """The docstring rule: anything outside the safe class becomes ``_``."""
    return "".join(ch if ch in _SAFE else "_" for ch in raw)


@DETERMINISTIC
@given(names)
def test_plugin_callable_name_is_idempotent(raw):
    once = plugin_callable_name(raw)
    assert plugin_callable_name(once) == once


@DETERMINISTIC
@given(names)
def test_plugin_callable_name_matches_the_safe_character_class(raw):
    assert all(ch in _SAFE for ch in plugin_callable_name(raw))


@DETERMINISTIC
@given(names)
def test_plugin_callable_name_replaces_characters_outside_the_safe_class(raw):
    assert plugin_callable_name(raw) == _documented_callable_name(raw)


@DETERMINISTIC
@given(not_mcp_names)
def test_split_mcp_tool_name_is_none_without_the_mcp_prefix(tool_name):
    assert split_mcp_tool_name(tool_name) is None


@DETERMINISTIC
@given(servers, tools)
def test_split_mcp_tool_name_round_trips_server_and_tool(server, tool):
    assert split_mcp_tool_name(f"mcp__{server}__{tool}") == (server, tool)
