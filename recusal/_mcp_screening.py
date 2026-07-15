"""Private implementation of MCP declaration and instruction screening.

The public compatibility facade remains :mod:`recusal.mcp`. This module is deliberately
private so its layout can evolve without creating a second public import path.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .evidence import Finding


def _declared_text(value: Any, *, max_depth: int) -> Tuple[List[str], bool]:
    """Collect declaration strings iteratively; report nesting beyond ``max_depth``."""
    out: List[str] = []
    too_deep = False
    stack: List[Tuple[Any, int]] = [(value, 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            if depth >= max_depth:
                too_deep = True
                continue
            for key, child in reversed(list(node.items())):
                stack.append((child, depth + 1))
                if isinstance(key, str):
                    stack.append((key, depth + 1))
        elif isinstance(node, (list, tuple)):
            if depth >= max_depth:
                too_deep = True
                continue
            for child in reversed(node):
                stack.append((child, depth + 1))
    return out, too_deep


def _screen_server_instructions(
    instructions: Dict[str, Optional[str]],
    *,
    markers: Sequence[str],
    max_chars: int,
) -> List[Finding]:
    findings: List[Finding] = []
    for server, text in sorted(instructions.items()):
        if text is None:
            continue
        low = text.lower()
        hits = [marker for marker in markers if marker in low]
        if hits:
            findings.append(
                Finding.fail(
                    "mcp_instructions_marker",
                    severity="ERROR",
                    message=f"server {server!r} instructions carry injection phrasing: "
                    f"{hits[0]!r}; review before pinning",
                    server=server,
                    markers=hits,
                )
            )
        if len(text) > max_chars:
            findings.append(
                Finding.fail(
                    "mcp_instructions_size",
                    severity="ERROR",
                    message=f"server {server!r} declares {len(text)} chars of "
                    f"instructions (cap {max_chars}); too large to plausibly review is "
                    "itself a review flag",
                    server=server,
                )
            )
    return findings


def _screen_tool_declarations(
    catalog: Dict[str, List[dict]],
    *,
    markers: Sequence[str],
    max_chars: int,
    max_depth: int,
) -> List[Finding]:
    findings: List[Finding] = []
    screened = 0
    for server, tools in (catalog or {}).items():
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "?"))
            screened += 1
            texts, too_deep = _declared_text(tool, max_depth=max_depth)
            if too_deep:
                findings.append(
                    Finding.fail(
                        "mcp_declaration_depth",
                        severity="ERROR",
                        message=f"tool '{name}' on server '{server}' nests its declaration "
                        f"deeper than {max_depth} levels; too deep to plausibly review is "
                        "itself a review flag",
                        server=server,
                        tool=name,
                    )
                )
            low = "\n".join(texts).lower()
            hits = [marker for marker in markers if marker in low]
            if hits:
                findings.append(
                    Finding.fail(
                        "mcp_declaration_marker",
                        severity="ERROR",
                        message=f"tool '{name}' on server '{server}' carries injection "
                        f"phrasing in its declaration: {hits[0]!r}; review before pinning",
                        server=server,
                        tool=name,
                        markers=hits,
                    )
                )
            total = sum(len(text) for text in texts)
            if total > max_chars:
                findings.append(
                    Finding.fail(
                        "mcp_declaration_size",
                        severity="ERROR",
                        message=f"tool '{name}' on server '{server}' declares {total} chars "
                        f"of text (cap {max_chars}); too large to plausibly review is itself "
                        "a review flag",
                        server=server,
                        tool=name,
                    )
                )
    if not findings:
        findings.append(
            Finding.ok(
                "mcp_declaration_screen",
                severity="ERROR",
                message=f"{screened} declaration(s) screened, no injection markers",
            )
        )
    return findings
