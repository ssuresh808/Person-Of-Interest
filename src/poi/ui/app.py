"""NiceGUI frontend.

A minimal but polished search UI:
    - One text input for the natural-language query
    - A grid of result cards: image + similarity score + VLM caption
    - Latency breakdown for each stage so the user can see where time goes

Why NiceGUI: it's a Python-native FastAPI-backed UI framework that gets out
of the way. Streamlit reruns the whole script on every interaction, which
forces awkward caching gymnastics for an LLM-loading app. Gradio is great
for ML demos but its component model is more rigid than NiceGUI's. NiceGUI
lets us hold the heavy objects (encoder, index, captioner) in module scope
and call them directly from event handlers.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from nicegui import app as nicegui_app
from nicegui import ui

from poi.embeddings.factory import build_encoder
from poi.index import FaissIndex
from poi.retrieval import RetrievalPipeline
from poi.utils.config import POIConfig
from poi.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


# ----- Module-level state -----
# Held here so they survive across UI events without rebuilding.
# This is the main reason we use NiceGUI instead of Streamlit.

_pipeline: RetrievalPipeline | None = None
_config: POIConfig | None = None


def _build_pipeline(cfg: POIConfig, index_path: Path) -> RetrievalPipeline:
    """Construct the full retrieval pipeline from a config."""
    log.info("Building encoder...")
    encoder = build_encoder(cfg.embedding)

    log.info(f"Loading index from {index_path}")
    index = FaissIndex.load(index_path)
    if cfg.index.use_gpu:
        index.to_gpu()

    captioner = None
    if cfg.vlm.enabled:
        log.info("Loading VLM captioner...")
        # Lazy import — only pay the import cost if the user enables captioning.
        from poi.vlm import QwenVLCaptioner

        captioner = QwenVLCaptioner(
            model_name=cfg.vlm.model_name,
            max_new_tokens=cfg.vlm.max_new_tokens,
            temperature=cfg.vlm.temperature,
            prompt_template=cfg.vlm.prompt_template,
        )

    return RetrievalPipeline(encoder=encoder, index=index, cfg=cfg.retrieval, captioner=captioner)


# ----- UI page -----


@ui.page("/")
def index_page() -> None:
    """The single-page UI."""
    ui.add_head_html("""
        <style>
            .poi-header { font-family: 'Georgia', serif; }
            .poi-card { transition: transform 0.15s ease; }
            .poi-card:hover { transform: translateY(-2px); }
            .poi-score { font-variant-numeric: tabular-nums; }
        </style>
    """)

    with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-4"):
        # Header
        with ui.column().classes("gap-1"):
            ui.label("Person of Interest").classes("poi-header text-3xl font-semibold")
            ui.label(
                "Describe a person. Find the closest matches in CelebA. "
                "A vision-language model explains each match."
            ).classes("text-sm text-gray-500")

        # Search bar
        query_input = (
            ui.input(
                placeholder="A person who has curly dark hair, glasses, and a thoughtful expression",
            )
            .props("clearable outlined dense")
            .classes("w-full")
        )

        # Results container — populated on search
        results_container = ui.column().classes("w-full gap-3")
        timings_label = ui.label().classes("text-xs text-gray-500 self-end")

        async def do_search() -> None:
            query = (query_input.value or "").strip()
            results_container.clear()
            timings_label.text = ""

            if not query:
                with results_container:
                    ui.label("Type a description above to start.").classes(
                        "text-gray-400 italic p-4"
                    )
                return

            if _pipeline is None:
                with results_container:
                    ui.label("Pipeline not initialized.").classes("text-red-500 p-4")
                return

            # Show a spinner while the pipeline runs. NiceGUI captures the
            # element via the `with` context — we don't need to bind it.
            with results_container:
                ui.spinner(size="lg").classes("self-center")

            try:
                response = await asyncio.to_thread(_pipeline.search, query)
            except Exception as e:
                log.exception("search failed")
                results_container.clear()
                with results_container:
                    ui.label(f"Search failed: {e}").classes("text-red-500 p-4")
                return

            results_container.clear()

            if not response.hits:
                with results_container:
                    ui.label("No matches above the score threshold.").classes(
                        "text-gray-500 italic p-4"
                    )
            else:
                _render_hits(results_container, response)

            # Show timings
            timings_str = " · ".join(
                f"{stage}: {ms:.0f} ms" for stage, ms in response.timings_ms.items()
            )
            timings_label.text = timings_str

        query_input.on("keydown.enter", do_search)
        ui.button("Search", on_click=do_search).props("color=primary unelevated")

        # Initial empty state
        with results_container:
            ui.label("Results will appear here.").classes("text-gray-400 italic p-4")


def _render_hits(container, response) -> None:
    """Render a grid of result cards into the given container."""
    with container, ui.grid(columns=4).classes("w-full gap-3"):
        for hit in response.hits:
            _render_hit_card(hit)


def _render_hit_card(hit) -> None:
    """One card per result: image, score, optional caption."""
    img_url = _expose_image(hit.image_path)
    with ui.card().classes("poi-card w-full"):
        ui.image(img_url).classes("w-full aspect-square object-cover rounded")
        with ui.column().classes("p-2 gap-1"):
            with ui.row().classes("justify-between items-baseline w-full"):
                ui.label(f"#{hit.rank}").classes("text-xs text-gray-400")
                ui.label(f"{hit.score:.3f}").classes("poi-score text-xs font-mono")
            if hit.caption:
                ui.label(hit.caption).classes("text-xs text-gray-700 leading-snug")


# Map filesystem image paths to served URLs.
# NiceGUI exposes static directories via app.add_static_files. We register
# each unique parent directory once.
_exposed_dirs: set[Path] = set()


def _expose_image(path: Path) -> str:
    parent = path.parent.resolve()
    if parent not in _exposed_dirs:
        url_prefix = f"/img-{len(_exposed_dirs)}"
        nicegui_app.add_static_files(url_prefix, str(parent))
        _exposed_dirs.add(parent)
        # Stash mapping for later filename → url construction
        _expose_image._mapping[parent] = url_prefix  # type: ignore[attr-defined]

    mapping = _expose_image._mapping  # type: ignore[attr-defined]
    return f"{mapping[parent]}/{path.name}"


_expose_image._mapping = {}  # type: ignore[attr-defined]


# ----- Entry point -----


def main() -> None:
    parser = argparse.ArgumentParser(description="Person of Interest UI")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--index", type=Path, required=True, help="Path to a built FAISS index")
    parser.add_argument("--host", default=os.environ.get("POI_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POI_PORT", "8080")))
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM captioning")
    args = parser.parse_args()

    setup_logging()

    global _pipeline, _config
    if args.config.exists():
        _config = POIConfig.from_yaml(args.config)
    else:
        log.warning(f"Config not found at {args.config}; using defaults")
        _config = POIConfig()

    if args.no_vlm:
        _config.vlm.enabled = False

    _pipeline = _build_pipeline(_config, args.index)

    ui.run(
        host=args.host,
        port=args.port,
        title="Person of Interest",
        favicon="🔍",
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
