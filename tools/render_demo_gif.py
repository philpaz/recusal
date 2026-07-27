"""Regenerate ``assets/demo-refusal.gif``, the README's hero image.

The README calls that GIF "verbatim transcripts, rendered". This script is what makes
that claim checkable rather than something you take on faith:

- **Panel 1** (the live Claude Code session refusing ``rm -rf`` under
  ``--dangerously-skip-permissions``) is carried over from the existing GIF frame for
  frame. It records a real session that cannot be re-run from a script, so it is
  preserved, never redrawn.
- **Panel 2** is rendered from the **captured stdout of the command it shows**. The
  demo output is executed here, at render time, from a neutral working directory, and
  every character on screen came out of that process. Colour is the only thing this
  script adds; no text is written by hand, shortened, or reordered.

Requires Pillow, which is deliberately NOT in the dev extra: this regenerates an asset,
it is not part of the gate, and the runtime stays at zero dependencies.

    pip install pillow
    py tools/render_demo_gif.py                 # rewrites assets/demo-refusal.gif
    py tools/render_demo_gif.py --python .venv/Scripts/python.exe

The interpreter passed with ``--python`` (default: the one running this script) must
have recusal importable; the version it reports is what the panel will show.
"""

import argparse
import os
import subprocess
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = os.path.join(REPO, "assets", "demo-refusal.gif")

W, H = 920, 760  # taller than a stock terminal: the demo's real output needs the room
X0, Y0, LH = 18, 46, 21  # text origin and line height, matched to the preserved panel
WRAP = 111  # characters that fit the window at this font size
SCENARIO = "wrong-subject"

#: GitHub dark palette, sampled from the preserved panel so the two halves match.
BODY = (13, 17, 23)
TEXT = (201, 209, 217)
DIM = (110, 118, 129)
BLUE = (121, 192, 255)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
CURSOR = (139, 148, 158)

#: Frames 0..43 of the existing asset are the live session, preserved verbatim.
PANEL1_END = 44

MONO = "C:/Windows/Fonts/consola.ttf"
MONO_BOLD = "C:/Windows/Fonts/consolab.ttf"


def _fonts():
    try:
        return ImageFont.truetype(MONO, 15), ImageFont.truetype(MONO_BOLD, 15)
    except OSError:  # pragma: no cover - non-Windows render host
        sys.exit(
            "This renderer expects Consolas at 15px, the font the preserved panel uses. "
            "Point MONO/MONO_BOLD at a metrically identical mono (8px advance) first."
        )


FONT, BOLD = _fonts()


def captured_output(python: str) -> "list[str]":
    """Run the command the panel shows and return its stdout, unmodified."""
    proc = subprocess.run(
        [python, "-m", "recusal", "demo", "--scenario", SCENARIO],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.expanduser("~"),  # a neutral directory: no checkout in sight
    )
    if proc.stderr:
        sys.exit(f"the demo wrote to stderr, which it must never do: {proc.stderr!r}")
    return proc.stdout.rstrip("\n").split("\n")


def wrap(line: str) -> "list[str]":
    if len(line) <= WRAP:
        return [line]
    indent = " " * (len(line) - len(line.lstrip()) + 2)
    return textwrap.wrap(
        line, WRAP, subsequent_indent=indent, break_long_words=False, break_on_hyphens=False
    )


def spans(line: str):
    """Colour one verbatim line. Rendering only: no character is added or removed."""
    stripped = line.strip()
    if stripped.startswith("RECUSAL,"):
        return [(line, BLUE, BOLD)]
    if stripped.startswith(("Offline:", "Policies to copy:", "https://")) or set(stripped) == {"-"}:
        return [(line, DIM, FONT)]
    if stripped[:2].isdigit() is False and stripped[1:3] == ". " and stripped[0].isdigit():
        return [(line, BLUE, BOLD)]  # a scenario heading, "1. WRONG-SUBJECT WRITE ..."
    for token, colour in (("verdict FAIL", RED), ("verdict PASS", GREEN)):
        if token in line:
            head, tail = line.split(token, 1)
            return [(head, TEXT, FONT), (token, colour, BOLD), (tail, colour, BOLD)]
    if stripped.startswith("proposed:"):
        head, tail = line.split("proposed:", 1)
        return [(head + "proposed:", DIM, FONT), (tail, TEXT, FONT)]
    if stripped.startswith("recusal ") and "  " in stripped:
        cmd, rest = stripped.split("  ", 1)
        pad = line[: len(line) - len(stripped)]
        return [(pad + cmd, BLUE, FONT), ("  " + rest, DIM, FONT)]
    return [(line, TEXT, FONT)]


def draw(base: Image.Image, lines, cursor: bool = False) -> Image.Image:
    img = base.copy()
    pen = ImageDraw.Draw(img)
    for row, line in enumerate(lines):
        y = Y0 + row * LH
        x = X0
        for text, colour, font in line:
            pen.text((x, y), text, font=font, fill=colour)
            x += int(font.getlength(text))
        if cursor and row == len(lines) - 1:
            pen.rectangle([x + 1, y + 2, x + 8, y + 15], fill=CURSOR)
    return img


def prompt(command: str):
    return [("PS recusal>", GREEN, BOLD), (" " + command, BLUE, FONT)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the README's hero GIF.")
    parser.add_argument("--python", default=sys.executable, help="interpreter with recusal")
    parser.add_argument("--out", default=ASSET, help="output GIF path")
    args = parser.parse_args()

    version = subprocess.run(
        [args.python, "-c", "import recusal; print(recusal.__version__)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    source = Image.open(ASSET)
    frames, durations = [], []

    def extend(frame: Image.Image) -> Image.Image:
        # The window has no side or bottom border (title bar rows 0..32, uniform body
        # below), so a taller canvas filled with the body colour is seamless.
        canvas = Image.new("RGB", (W, H), BODY)
        canvas.paste(frame, (0, 0))
        return canvas

    for index in range(PANEL1_END):
        source.seek(index)
        frames.append(extend(source.convert("RGB")))
        durations.append(source.info.get("duration", 40))

    source.seek(PANEL1_END)
    template = extend(source.convert("RGB"))
    ImageDraw.Draw(template).rectangle([0, 33, W, H], fill=BODY)  # keep chrome, clear body

    body = [spans(part) for line in captured_output(args.python) for part in wrap(line)]
    if Y0 + (3 + len(body)) * LH > H:
        sys.exit("the captured output no longer fits the canvas; raise H rather than truncate")

    screen: list = []

    def push(cursor=True, ms=40):
        frames.append(draw(template, screen, cursor=cursor))
        durations.append(ms)

    install = "pip install recusal"
    for n in range(0, len(install) + 1, 4):
        screen = [prompt(install[:n])]
        push()
    screen = [prompt(install)]
    push(ms=260)
    screen.append([(f"Successfully installed recusal-{version}", TEXT, FONT)])
    push(cursor=False, ms=700)

    demo = f"recusal demo --scenario {SCENARIO}"
    header = list(screen)
    for n in range(0, len(demo) + 1, 4):
        screen = header + [prompt(demo[:n])]
        push()
    screen = header + [prompt(demo)]
    push(ms=300)

    header = list(screen)
    shown = 0
    for count, hold in (
        (2, 460), (2, 380), (2, 460), (2, 620), (1, 900), (3, 1300),
        (2, 460), (2, 900), (3, 700), (4, 800), (5, 700),
    ):  # fmt: skip
        shown = min(shown + count, len(body))
        frames.append(draw(template, header + body[:shown]))
        durations.append(hold)
    frames.append(draw(template, header + body))
    durations.append(4500)

    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"{args.out}: {len(frames)} frames, {sum(durations) / 1000:.1f}s, recusal {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
