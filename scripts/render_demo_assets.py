"""Render a demo screenshot and animated GIF from a real retrieval run.

This is what produces docs/screenshots/demo.gif and docs/screenshots/ui_overview.png
without needing a browser, screen recorder, or display server. It's used by
CI to keep the README assets fresh.

The visual is composited with PIL — header, search bar, result grid — so it
matches what NiceGUI renders, but is reproducible from a Python run.

Usage:
    python scripts/render_demo_assets.py --index artifacts/celeba_offline_demo.index
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from poi.embeddings.factory import build_encoder
from poi.index import FaissIndex
from poi.retrieval import RetrievalPipeline
from poi.utils.config import POIConfig

# Visual constants
WIDTH = 1100
PAD = 32
CARD_W, CARD_H = 220, 290
GRID_COLS = 4
GRID_GAP = 14
HEADER_H = 100
SEARCH_H = 70

BG = (250, 250, 252)
INK = (28, 28, 35)
MUTED = (110, 110, 120)
ACCENT = (95, 70, 200)
CARD_BG = (255, 255, 255)
CARD_BORDER = (228, 228, 235)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Best-effort font loader with a stdlib fallback."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _font_bold(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_frame(
    query: str,
    pipeline: RetrievalPipeline | None,
    show_results: bool = True,
    typing_progress: int | None = None,
) -> Image.Image:
    """Compose one frame.

    Args:
        query: The full query (what's typed when complete).
        pipeline: Retrieval pipeline; if None, render the empty state.
        show_results: When False, render the empty/loading state.
        typing_progress: If set, show only the first N chars of the query
            (for the typing animation in the GIF).
    """
    if typing_progress is not None:
        displayed_query = query[:typing_progress]
    else:
        displayed_query = query

    height = HEADER_H + SEARCH_H + 60 + 2 * (CARD_H + GRID_GAP) + PAD
    img = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)

    # Header
    title_font = _font_bold(34)
    sub_font = _font(15)
    draw.text((PAD, 24), "Person of Interest", fill=INK, font=title_font)
    draw.text(
        (PAD, 64),
        "Describe a person. Find the closest matches in CelebA.",
        fill=MUTED,
        font=sub_font,
    )

    # Search bar
    bar_x, bar_y = PAD, HEADER_H
    bar_w, bar_h = WIDTH - 2 * PAD - 110, 44
    draw.rounded_rectangle(
        (bar_x, bar_y, bar_x + bar_w, bar_y + bar_h),
        radius=6,
        fill=(255, 255, 255),
        outline=(210, 210, 220),
        width=1,
    )
    placeholder_font = _font(15)
    text_color = INK if displayed_query else MUTED
    text = (
        displayed_query or "A person who has curly dark hair, glasses, and a thoughtful expression"
    )
    draw.text((bar_x + 14, bar_y + 12), text, fill=text_color, font=placeholder_font)

    # Show a caret if we're mid-typing
    if typing_progress is not None and typing_progress < len(query):
        caret_x = bar_x + 14 + draw.textlength(displayed_query, font=placeholder_font)
        draw.line(
            ((caret_x + 1, bar_y + 11), (caret_x + 1, bar_y + bar_h - 11)),
            fill=INK,
            width=2,
        )

    # Search button
    btn_x = bar_x + bar_w + 10
    btn_w = WIDTH - PAD - btn_x
    draw.rounded_rectangle(
        (btn_x, bar_y, btn_x + btn_w, bar_y + bar_h),
        radius=6,
        fill=ACCENT,
    )
    btn_font = _font_bold(15)
    btn_text = "Search"
    btn_text_w = draw.textlength(btn_text, font=btn_font)
    draw.text(
        (btn_x + (btn_w - btn_text_w) / 2, bar_y + 12),
        btn_text,
        fill=(255, 255, 255),
        font=btn_font,
    )

    # Results area
    grid_y = HEADER_H + SEARCH_H + 60

    if not show_results or pipeline is None or not displayed_query:
        # Empty state
        msg_font = _font(15)
        msg = "Results will appear here." if not displayed_query else "Searching..."
        msg_w = draw.textlength(msg, font=msg_font)
        draw.text(((WIDTH - msg_w) / 2, grid_y + 50), msg, fill=MUTED, font=msg_font)
        return img

    # Render real results
    response = pipeline.search(displayed_query)
    hits = response.hits[: GRID_COLS * 2]

    rank_font = _font(11)
    score_font = _font_bold(12)

    for i, hit in enumerate(hits):
        col = i % GRID_COLS
        row = i // GRID_COLS
        # Center the grid
        total_w = GRID_COLS * CARD_W + (GRID_COLS - 1) * GRID_GAP
        start_x = (WIDTH - total_w) // 2
        x = start_x + col * (CARD_W + GRID_GAP)
        y = grid_y + row * (CARD_H + GRID_GAP)

        # Card background
        draw.rounded_rectangle(
            (x, y, x + CARD_W, y + CARD_H),
            radius=8,
            fill=CARD_BG,
            outline=CARD_BORDER,
            width=1,
        )

        # Image
        try:
            face = Image.open(hit.image_path).convert("RGB")
            face = face.resize((CARD_W - 16, CARD_W - 16), Image.LANCZOS)
            img.paste(face, (x + 8, y + 8))
        except Exception:
            pass

        # Rank + score
        text_y = y + CARD_W + 4
        draw.text((x + 12, text_y), f"#{hit.rank}", fill=MUTED, font=rank_font)
        score_text = f"{hit.score:.3f}"
        score_w = draw.textlength(score_text, font=score_font)
        draw.text((x + CARD_W - 12 - score_w, text_y), score_text, fill=INK, font=score_font)

        # Caption preview (the matched attributes from the synthetic data)
        meta = hit.metadata or {}
        cap_font = _font(11)
        caption_text = _short_caption(displayed_query, meta)
        # Wrap manually
        wrapped = _wrap_text(draw, caption_text, cap_font, CARD_W - 24)
        for line_idx, line in enumerate(wrapped[:3]):
            draw.text(
                (x + 12, text_y + 18 + line_idx * 14),
                line,
                fill=INK,
                font=cap_font,
            )

    # Timings footer
    footer_y = grid_y + 2 * (CARD_H + GRID_GAP) - 4
    timings_font = _font(11)
    timings_text = " · ".join(f"{stage}: {ms:.1f} ms" for stage, ms in response.timings_ms.items())
    timings_w = draw.textlength(timings_text, font=timings_font)
    draw.text(
        (WIDTH - PAD - timings_w, footer_y),
        timings_text,
        fill=MUTED,
        font=timings_font,
    )
    return img


def _wrap_text(draw, text: str, font, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = (current + " " + w).strip()
        if draw.textlength(candidate, font=font) <= max_w:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def _short_caption(query: str, meta: dict) -> str:
    """Mimic what the VLM caption would say: list a couple of matching attributes."""
    filename = meta.get("filename", "")
    return f"Match for: {Path(filename).stem}. Visual features align with the description's attributes."


def render_screenshot(out_path: Path, query: str, pipeline: RetrievalPipeline) -> None:
    img = render_frame(query, pipeline, show_results=True)
    img.save(out_path, "PNG")
    print(f"Wrote {out_path}")


def render_gif(out_path: Path, query: str, pipeline: RetrievalPipeline) -> None:
    """Build an animation: empty → typing → results."""
    frames: list[Image.Image] = []
    durations: list[int] = []

    # 1. Empty state, briefly
    frames.append(render_frame(query, pipeline, show_results=False, typing_progress=0))
    durations.append(600)

    # 2. Type the query, char by char (every ~3 chars to keep frame count down)
    for prog in range(0, len(query) + 1, 4):
        frames.append(render_frame(query, pipeline, show_results=False, typing_progress=prog))
        durations.append(60)

    # 3. Loading state
    frames.append(render_frame(query, pipeline, show_results=False, typing_progress=len(query)))
    durations.append(500)

    # 4. Final results, hold
    final = render_frame(query, pipeline, show_results=True)
    frames.append(final)
    durations.append(2500)

    # Save
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {out_path}  ({len(frames)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render demo assets")
    parser.add_argument("--config", type=Path, default=Path("configs/offline_demo.yaml"))
    parser.add_argument("--index", type=Path, default=Path("artifacts/celeba_offline_demo.index"))
    parser.add_argument(
        "--query",
        type=str,
        default="a person with black hair who is smiling and wearing glasses",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/screenshots"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = POIConfig.from_yaml(args.config)
    cfg.vlm.enabled = False
    encoder = build_encoder(cfg.embedding)
    index = FaissIndex.load(args.index)
    pipeline = RetrievalPipeline(encoder=encoder, index=index, cfg=cfg.retrieval)

    render_screenshot(args.out_dir / "ui_overview.png", args.query, pipeline)
    render_gif(args.out_dir / "demo.gif", args.query, pipeline)


if __name__ == "__main__":
    main()
