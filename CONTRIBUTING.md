# Contributing

This is primarily a personal portfolio project, but the codebase is structured so contributions are easy if you want to extend it.

## Local setup

```bash
git clone https://github.com/<you>/person-of-interest.git
cd person-of-interest
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,eval]"
pre-commit install
```

## Running tests

```bash
# Fast tests only (no GPU, no model loading)
pytest -m "not gpu and not integration"

# Full test suite, including ones that load real models
pytest

# A specific test
pytest tests/test_index.py::TestPersistence::test_roundtrip -v
```

## Code style

- **Ruff** for linting and formatting. CI fails if `ruff check` or `ruff format --check` finds issues.
- **mypy** for type checking, advisory only (CI does not block on mypy errors yet).
- Type-hint public APIs. Internal functions can be loose.
- Docstrings on public functions and classes. Reasoning over reciting — explain *why*, not just *what*.

## Adding a new encoder

The `Encoder` protocol in [`src/poi/embeddings/base.py`](src/poi/embeddings/base.py) has four required attributes:

```python
name: str          # for logging and artifact names
dim: int           # output dimensionality
encode_images(...) # PIL/path → ndarray
encode_texts(...)  # str → ndarray
```

Implement those, register the new backend in [`factory.py`](src/poi/embeddings/factory.py), and the index, retrieval, and UI layers will pick it up without modification.

If your encoder is image-only (like DeepFace), have `encode_texts` raise `NotImplementedError`. The factory will instantiate it for image-only ablations.

## Adding a new index type

[`src/poi/index/faiss_index.py`](src/poi/index/faiss_index.py) has a `_build_empty_index` switch. Add a branch there. Keep the metadata sidecar API unchanged — that's what the rest of the project depends on.

## Submitting changes

1. Create a branch.
2. Make your changes. Add tests if behavior changed.
3. Run `pre-commit run --all-files` and `pytest -m "not gpu"`.
4. Open a PR with a description that explains *why*, not just *what*.

## What's out of scope

- Identity matching (closed-world face recognition) — see [ARCHITECTURE.md](ARCHITECTURE.md) for why this isn't the goal.
- Fine-tuning the encoder on CelebA — listed as a next step but not currently in the codebase.
- Demographic-attribute queries — explicitly excluded for fairness reasons. PRs that add them will not be accepted.
