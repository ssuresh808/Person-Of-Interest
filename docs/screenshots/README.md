# Screenshots and demo

The files in this directory are auto-generated from the offline demo pipeline. Regenerate them any time the UI or pipeline changes:

```bash
# Build the offline demo index first (one-time)
python scripts/generate_synthetic_data.py --out data/celeba_synthetic --n 500
python scripts/build_index.py --config configs/offline_demo.yaml \
    --images data/celeba_synthetic/img_align_celeba \
    --out artifacts/celeba_offline_demo.index

# Render demo.gif and ui_overview.png
python scripts/render_demo_assets.py
```

[`scripts/render_demo_assets.py`](../../scripts/render_demo_assets.py) composites the UI in PIL — header, search bar, result grid — using real retrieval output. No browser, screen recorder, or display server required. CI can regenerate these on every push.

## Files

- `demo.gif` — Animated typing-then-results sequence using the offline demo pipeline
- `ui_overview.png` — Single-frame screenshot of the search interface with eight real results
- `architecture_diagram.png` — Visual version of the ASCII architecture in the README (optional, hand-drawn)

## Recording from the live NiceGUI app instead

For a screen recording of the actual NiceGUI server (rather than a PIL composite), launch the UI on the cluster, port-forward to your laptop, and use:

- **Linux**: `peek` or `byzanz-record`
- **macOS**: built-in screen recorder, then `ffmpeg` to convert to GIF
- **Cross-platform**: [LICEcap](https://www.cockos.com/licecap/) or [Kap](https://getkap.co/)

```bash
# Convert .mov to optimized GIF
ffmpeg -i demo.mov -vf "fps=12,scale=900:-1:flags=lanczos,palettegen" palette.png
ffmpeg -i demo.mov -i palette.png -filter_complex "fps=12,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse" demo_real.gif
gifsicle -O3 --lossy=80 demo_real.gif -o demo_real.gif
```

Target file size: < 4 MB so the README loads quickly. Live-recorded GIFs typically replace the auto-generated `demo.gif` for the public repo.
