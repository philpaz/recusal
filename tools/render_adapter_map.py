"""Regenerate ``assets/adapter-map.svg`` and ``assets/adapter-map-light.svg``, the README's
picture of the kernel and its adapter packages.

    python tools/render_adapter_map.py            # rewrite both files
    python tools/render_adapter_map.py --check    # exit 1 if either file is out of date

The picture is drawn from the adapters' own ``pyproject.toml`` files, so it cannot show a
package that does not exist or miss one that does: every directory under ``adapters/`` is a
box, an adapter that requires another sits beside it with an arrow to it, and every box
has its own line to the kernel, since every adapter requires the kernel itself. Nothing
is drawn out of the kernel, because it depends on nothing.
``tests/test_adapter_map.py`` fails if the committed files differ from what this prints.

The README needs an image and not a Mermaid block because the README is also the PyPI
project page, which shows Mermaid as source. The Mermaid diagram in ``adapters/README.md``
carries the exact edges.

Standard library only, and no fonts are embedded: the text is drawn by whatever renders the
SVG, in a serif and a monospace family, so the output is the same bytes on every machine.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"
ASSETS = ROOT / "assets"

WIDTH = 1200
BOX_W, BOX_H = 300, 54
LEFT, TOP, ROW_GAP, COL_GAP = 70, 150, 26, 44
KERNEL_W, KERNEL_H = 300, 190
SERIF = "Georgia, 'Palatino Linotype', 'Book Antiqua', Palatino, serif"
MONO = "Consolas, 'SF Mono', Menlo, 'DejaVu Sans Mono', monospace"
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

#: The two palettes, taken from the project's banners: the seal (dark) and the ledger (light).
THEMES = {
    "adapter-map.svg": {
        "background": "#131829",
        "frame": "#4a3640",
        "ink": "#efe7d8",
        "muted": "#9aa0b4",
        "accent": "#c1553a",
        "box": "#1b2238",
        "line": "#6f7791",
    },
    "adapter-map-light.svg": {
        "background": "#e9ebf0",
        "frame": "#d3c3c6",
        "ink": "#1b1e29",
        "muted": "#6b6470",
        "accent": "#b4443a",
        "box": "#f4f5f8",
        "line": "#8b8894",
    },
}


def packages() -> dict[str, list[str]]:
    """Each adapter distribution and the adapter distributions it requires."""
    import tomllib

    found: dict[str, list[str]] = {}
    for path in sorted(ADAPTERS.glob("*/pyproject.toml")):
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
        requires = []
        for requirement in project["dependencies"]:
            name = re.match(r"[A-Za-z0-9_.-]+", requirement).group(0)  # type: ignore[union-attr]
            if name.startswith("recusal-"):
                requires.append(name)
        found[project["name"]] = requires
    return found


def rows(found: dict[str, list[str]]) -> list[tuple[str, str, bool]]:
    """The picture's rows, each two boxes: ``(left, right, left requires right)``.

    An adapter that requires another shares a row with it, the dependent on the left. The
    rest fill rows two at a time, in name order."""
    paired: list[tuple[str, str, bool]] = []
    used: set[str] = set()
    for name in sorted(found):
        for required in found[name]:
            paired.append((name, required, True))
            used.update((name, required))
    rest = [name for name in sorted(found) if name not in used]
    if len(rest) % 2:
        raise SystemExit("the adapter map draws two boxes a row; lay out the odd one first")
    return paired + [(rest[i], rest[i + 1], False) for i in range(0, len(rest), 2)]


def render(theme: dict[str, str], found: dict[str, list[str]]) -> str:
    layout = rows(found)
    grid_h = len(layout) * BOX_H + (len(layout) - 1) * ROW_GAP
    height = TOP + grid_h + 166
    right_x = LEFT + BOX_W + COL_GAP
    bus_x = right_x + BOX_W + 46
    kernel_x = WIDTH - LEFT - KERNEL_W
    kernel_y = TOP + (grid_h - KERNEL_H) // 2
    mid_y = kernel_y + KERNEL_H // 2
    t = theme
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
        f'width="{WIDTH}" height="{height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">recusal: one kernel, {NUMBERS[len(found)]} adapter packages</title>',
        '<desc id="desc">Each adapter package depends on the recusal kernel. '
        + " ".join(
            f"{left} also depends on {right}." for left, right, requires in layout if requires
        )
        + " The kernel depends on nothing.</desc>",
        "<defs>",
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" '
        f'fill="{t["line"]}"/></marker>',
        "</defs>",
        f'<rect width="{WIDTH}" height="{height}" fill="{t["background"]}"/>',
        f'<rect x="22" y="22" width="{WIDTH - 44}" height="{height - 44}" fill="none" '
        f'stroke="{t["frame"]}" stroke-width="1.5"/>',
        f'<text x="{LEFT}" y="92" font-family="{MONO}" font-size="19" letter-spacing="6" '
        f'fill="{t["accent"]}">ONE KERNEL · {NUMBERS[len(found)].upper()} ADAPTERS</text>',
        f'<text x="{WIDTH - LEFT}" y="92" text-anchor="end" font-family="{MONO}" '
        f'font-size="15" letter-spacing="3" fill="{t["muted"]}">AN ARROW MEANS DEPENDS ON</text>',
    ]
    for index, (left, right, requires) in enumerate(layout):
        y = TOP + index * (BOX_H + ROW_GAP)
        cy = y + BOX_H // 2
        for x, name in ((LEFT, left), (right_x, right)):
            out.append(
                f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="6" '
                f'fill="{t["box"]}" stroke="{t["line"]}" stroke-width="1.2"/>'
            )
            out.append(
                f'<text x="{x + BOX_W // 2}" y="{cy + 6}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="17" fill="{t["ink"]}">{escape(name)}</text>'
            )
        if requires:
            out.append(
                f'<path d="M{LEFT + BOX_W} {cy}H{right_x - 3}" fill="none" stroke="{t["line"]}" '
                f'stroke-width="1.6" marker-end="url(#arrow)"/>'
            )
        # Every adapter requires the kernel itself, so every box has its own line to the
        # bus that runs to it: the right-hand box straight across, the left-hand box down
        # into the gap under its row and along it.
        under = y + BOX_H + ROW_GAP // 2
        out.append(
            f'<path d="M{right_x + BOX_W} {cy}H{bus_x}" fill="none" stroke="{t["line"]}" '
            f'stroke-width="1.6"/>'
        )
        out.append(
            f'<path d="M{LEFT + BOX_W // 2} {y + BOX_H}V{under}H{bus_x}" fill="none" '
            f'stroke="{t["line"]}" stroke-width="1.6"/>'
        )
    first = TOP + BOX_H // 2
    last = TOP + (len(layout) - 1) * (BOX_H + ROW_GAP) + BOX_H + ROW_GAP // 2
    out += [
        f'<path d="M{bus_x} {first}V{last}" fill="none" stroke="{t["line"]}" stroke-width="1.6"/>',
        f'<path d="M{bus_x} {mid_y}H{kernel_x - 4}" fill="none" stroke="{t["line"]}" '
        f'stroke-width="1.6" marker-end="url(#arrow)"/>',
        f'<rect x="{kernel_x}" y="{kernel_y}" width="{KERNEL_W}" height="{KERNEL_H}" rx="8" '
        f'fill="{t["box"]}" stroke="{t["accent"]}" stroke-width="2"/>',
        f'<text x="{kernel_x + KERNEL_W // 2}" y="{kernel_y + 84}" text-anchor="middle" '
        f'font-family="{SERIF}" font-size="60" font-weight="bold" fill="{t["ink"]}">recusal'
        f'<tspan fill="{t["accent"]}">.</tspan></text>',
        f'<text x="{kernel_x + KERNEL_W // 2}" y="{kernel_y + 124}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="15" letter-spacing="2" fill="{t["muted"]}">'
        "THE KERNEL</text>",
        f'<text x="{kernel_x + KERNEL_W // 2}" y="{kernel_y + 156}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="14" fill="{t["ink"]}">decides · records</text>',
        f'<text x="{kernel_x + KERNEL_W // 2}" y="{kernel_y + 178}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="14" fill="{t["ink"]}">depends on nothing</text>',
        f'<path d="M{WIDTH // 2 - 60} {height - 104}H{WIDTH // 2 + 60}" stroke="{t["accent"]}" '
        f'stroke-width="1.5"/>',
        f'<text x="{WIDTH // 2}" y="{height - 62}" text-anchor="middle" font-family="{SERIF}" '
        f'font-size="26" font-style="italic" fill="{t["muted"]}">'
        "The thing that generates the work must never certify it.</text>",
        "</svg>",
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if a file is out of date")
    args = parser.parse_args(argv)
    found = packages()
    stale = []
    for name, theme in THEMES.items():
        text = render(theme, found)
        path = ASSETS / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.write_bytes(text.encode("utf-8"))
            print(f"wrote assets/{name}")
    if stale:
        print(f"out of date: {', '.join(stale)}; run tools/render_adapter_map.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
