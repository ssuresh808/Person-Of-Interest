"""Generate the figures that ship with the README.

Reads from evals/results_*.json files (produced by scripts/evaluate.py)
and writes:
    evals/figures/recall_curves.png
    evals/figures/latency_vs_recall.png
    evals/figures/random_vectors_concentration.png
    evals/figures/encoder_anisotropy.png

Run after at least one successful evaluation. The notebooks in notebooks/
produce the same figures interactively; this script is for CI / batch use.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110

EVAL_DIR = Path("evals")
FIG_DIR = EVAL_DIR / "figures"


def _load_results() -> list[dict]:
    """Load every results_*.json found in evals/."""
    out: list[dict] = []
    for path in sorted(EVAL_DIR.glob("results_*.json")):
        with path.open() as f:
            out.append(json.load(f))
    return out


def plot_recall_curves(results: list[dict]) -> None:
    if not results:
        print("No results to plot. Run scripts/evaluate.py first.")
        return

    k_values = [1, 5, 10, 50]
    _, ax = plt.subplots(figsize=(9, 5.5))

    for r in results:
        recalls = [r["metrics"]["recall"].get(f"recall@{k}", 0.0) for k in k_values]
        ax.plot(k_values, recalls, marker="o", linewidth=2, label=r["encoder"])

    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels(k_values)
    ax.set_xlabel("K (top-K retrieved)")
    ax.set_ylabel("Recall@K")
    n = results[0]["metrics"]["n_queries"]
    ax.set_title(f"Retrieval recall on {n} synthesized queries")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out_path = FIG_DIR / "recall_curves.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def plot_latency_vs_recall(results: list[dict]) -> None:
    if not results:
        return
    _, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        x = r["metrics"]["latency_ms"]["p95"]
        y = r["metrics"]["recall"].get("recall@1", 0.0)
        ax.scatter(x, y, s=120)
        ax.annotate(
            r["encoder"],
            (x, y),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=10,
        )
    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("Recall@1")
    ax.set_title("Quality vs latency")
    plt.tight_layout()
    out_path = FIG_DIR / "latency_vs_recall.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def plot_random_concentration() -> None:
    """The Week 2 thought experiment: variance of cosine similarity vs dimension.

    No data needed — purely synthetic vectors. Always reproducible.
    """
    _, ax = plt.subplots(figsize=(9, 5))
    for d in [2, 8, 64, 768]:
        rng = np.random.default_rng(d)
        v = rng.standard_normal(size=(2000, d)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        idx = rng.integers(0, len(v), 50_000)
        jdx = rng.integers(0, len(v), 50_000)
        mask = idx != jdx
        sims = (v[idx[mask]] * v[jdx[mask]]).sum(axis=1)
        sns.kdeplot(sims, ax=ax, label=f"d = {d}", linewidth=2)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("Random unit vectors: cosine collapses to 0 as d grows")
    ax.legend(title="dimension")
    plt.tight_layout()
    out_path = FIG_DIR / "random_vectors_concentration.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_results()
    plot_recall_curves(results)
    plot_latency_vs_recall(results)
    plot_random_concentration()


if __name__ == "__main__":
    main()
