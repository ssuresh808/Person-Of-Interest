"""Configuration schemas.

Every tunable knob in the project is declared here. Configs are loaded from
YAML and validated by Pydantic. The CLI scripts accept --config <path> and
--override key=value for sweeps.

Why Pydantic over dataclasses or attrs: free validation, free serialization,
and the error messages when a YAML key is missing or misspelled are
genuinely helpful instead of cryptic AttributeError.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    """Which encoder to use, and how to call it."""

    backend: Literal["siglip2", "clip", "deepface", "hash"] = "siglip2"
    model_name: str = "google/siglip2-base-patch16-256"
    image_size: int = 256
    batch_size: int = 64
    normalize: bool = True  # L2-normalize embeddings before indexing
    cache_dir: Path | None = None


class IndexConfig(BaseModel):
    """FAISS index parameters."""

    index_type: Literal["flat", "ivf_flat", "ivf_pq"] = "ivf_flat"
    nlist: int = 256  # Voronoi cells. Rule of thumb: 4*sqrt(n) to 16*sqrt(n)
    nprobe: int = 8  # Cells to search at query time. Recall/latency knob.
    metric: Literal["ip", "l2"] = "ip"  # Inner product = cosine when normalized
    use_gpu: bool = True

    @field_validator("nprobe")
    @classmethod
    def _check_nprobe(cls, v: int, info) -> int:
        nlist = info.data.get("nlist", 256)
        if v > nlist:
            raise ValueError(f"nprobe ({v}) cannot exceed nlist ({nlist})")
        return v


class VLMConfig(BaseModel):
    """Vision-language model for caption / explanation step."""

    enabled: bool = True
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_new_tokens: int = 96
    temperature: float = 0.2  # Low — we want descriptive, not creative
    # Prompt template path, relative to the package
    prompt_template: str = "vlm/prompts/explain_match.txt"


class RetrievalConfig(BaseModel):
    """End-to-end retrieval pipeline."""

    top_k: int = 12
    rerank_top_k: int | None = None  # If set, take top-N from FAISS, rerank to top-K
    min_score: float = 0.0  # Drop matches below this cosine similarity


class DataConfig(BaseModel):
    """Where the corpus lives."""

    dataset: Literal["celeba", "lfw"] = "celeba"
    images_dir: Path = Path("data/celeba/img_align_celeba")
    attributes_csv: Path | None = Path("data/celeba/list_attr_celeba.csv")
    eval_split_csv: Path | None = Path("data/celeba/list_eval_partition.csv")


class POIConfig(BaseModel):
    """Top-level config object."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    artifacts_dir: Path = Path("artifacts")
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path | str) -> POIConfig:
        """Load and validate a config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Write the current config to YAML for reproducibility logging."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
