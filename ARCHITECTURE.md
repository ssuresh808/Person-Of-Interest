# Architecture

This document explains the design decisions behind Person of Interest. It assumes familiarity with embeddings, contrastive learning, and vision-language models. If you want a quick tour, the [README](README.md) is enough.

## The core problem

> Given a natural-language description of a person, find them in a corpus of face images.

This is multimodal retrieval. The query and the targets live in different modalities, so the only way to compare them is to project both into a shared representation space where similarity has meaning.

That sentence — "a shared representation space where similarity has meaning" — hides the entire design.

## Why a shared embedding space at all

In a low-dimensional space, distances and angles are intuitive. In a high-dimensional space, almost every pair of random vectors is nearly orthogonal — the famous concentration of measure phenomenon. Random points are predictably *far*; meaningful proximity is the exception, not the rule.

That's why semantic search works. If two embeddings are close on the unit hypersphere, it's not coincidence — it's because some training process aligned them. The signal-to-noise ratio of "this match means something" is very high in high dimensions, because chance similarity is exponentially suppressed.

For our use case, this means: if a face embedding ends up near a text embedding for "woman with glasses and curly hair," that proximity is doing real work. It survives the geometry.

## Why SigLIP-2 over CLIP

CLIP normalizes its contrastive loss with a softmax over the batch. That coupling means every example interacts with every other, and the loss requires very large batches to be stable.

SigLIP replaces softmax with independent sigmoid losses on each pair. Each (image, text) pair is judged on its own merits. This has two consequences that matter here:

1. **Smaller batches train fine.** This isn't relevant for us since we're using pretrained weights, but it's why SigLIP exists.
2. **The resulting embedding space is more uniformly distributed on the hypersphere.** This shows up in the histogram of pairwise cosine similarities — CLIP's distribution is anisotropic and skewed positive, SigLIP-2's is closer to the geometric ideal.

That second property is what we care about. A more uniform embedding space means more discriminative power per dimension, which translates into higher Recall@1 in retrieval. The ablation in [evals/results.md](evals/results.md) shows this directly: same FAISS, same dataset, same queries, ~7-point Recall@1 improvement going from CLIP-ViT-L/14 to SigLIP-2-large.

We use **SigLIP-2-base** as the default for latency and **SigLIP-2-large** as the high-quality option. The repo supports swapping in either via config.

## Why Qwen2.5-VL for captioning

The retrieval step alone gives you a list of faces. That's useful but not interesting — you can't tell *why* the system picked them.

A VLM captioning step on top of retrieval gives you grounded explanations. Qwen2.5-VL takes the matched image and the original query and produces a sentence like:

> "This face matches the description's mention of curly dark hair and glasses, though the expression appears more neutral than 'thoughtful.'"

That sentence is doing two things at once. It's confirming the matches that worked, and it's flagging the parts of the query the system handled poorly. That second part is the interesting signal — it tells you where to invest in the next iteration.

We chose Qwen2.5-VL-7B over LLaVA-1.5 for two reasons:

1. **Dynamic resolution.** Qwen2.5-VL handles images at native resolution rather than forcing them into a 224×224 grid. CelebA images are 178×218, which is fine, but if we extend to higher-resolution corpora, this matters.
2. **Grounding.** Qwen2.5-VL was trained with bounding-box supervision and tends to make spatially specific claims ("the person on the left," "the figure in the foreground"). Even when we don't use the boxes, the captions are noticeably more concrete than LLaVA's.

The cost is real — Qwen2.5-VL-7B in bf16 is ~14 GB of VRAM. That fits a single shared shard on the cluster's 4090 nodes, but it's a meaningful chunk of the budget. For latency-sensitive deployments, the 3B variant works with a noticeable but acceptable quality drop.

## Why a separate captioning stage instead of end-to-end VLM retrieval

A natural alternative is to skip the embedding stage entirely and use a VLM to score every (query, image) pair directly. That's a cross-encoder approach, and it's strictly more expressive than a bi-encoder.

It's also strictly impossible at scale.

Cross-encoders are O(n) per query — you have to feed every candidate through the model. With 200k images, that's 200k VLM forward passes for every search. Bi-encoders pre-compute the image side once, store the vectors, and reduce query time to a single text encoding plus a fast nearest-neighbor lookup.

The standard production pattern is **retrieve then rerank**: bi-encoder narrows the field from 200k to 100, cross-encoder (or VLM) reorders or annotates the survivors. We do the lighter version: bi-encoder retrieval, then VLM captioning of the top-k for explanation. A full reranker is on the [next-steps list](README.md#what-id-do-next).

## Why FAISS IVF-Flat

For 200k vectors, brute-force search is fine — a single GPU sweeps the whole index in milliseconds. We use IVF-Flat anyway, for two reasons:

1. **The repo should scale.** If someone clones this and runs it on a million-image corpus, IVF-Flat still works; brute-force does not.
2. **It's a teaching choice.** IVF-Flat partitions the embedding space into Voronoi cells via k-means, then searches only the nearest few cells. The fact that this works at all is a consequence of the same geometric phenomenon that makes embeddings useful: similar things cluster, dissimilar things don't.

The number of probes (`nprobe`) is the recall-vs-latency knob. Default is 8, which gives 99%+ of brute-force recall at 5× the speed. Configurable in [`src/poi/index/faiss_index.py`](src/poi/index/faiss_index.py).

## Anisotropy and the embedding histogram

One of the things the [first notebook](notebooks/01_embedding_geometry.ipynb) does is compute the pairwise cosine similarity histogram for each encoder on a random 5000-image subset of CelebA.

The histograms tell a story:

- **CLIP**: distribution centered around 0.3, long right tail. The space is anisotropic — vectors cluster in a narrow cone.
- **SigLIP-2-base**: distribution centered around 0.05, much tighter. Closer to the orthogonality you'd expect from random unit vectors in 768 dimensions.
- **SigLIP-2-large**: distribution centered around 0.02, very tight.

A tighter distribution around zero means that *anything* with above-zero similarity to a query is a real signal, not background noise. That's the geometric reason SigLIP-2 retrieves better.

## Endophora and query ambiguity

Real-world queries are full of underspecified references. "The woman from yesterday's photo." "The guy who looked like my cousin." A description-based system can't resolve these — there's no shared context.

What we *can* do is fail gracefully. The UI surfaces the top-k matches with their captions, lets the user pick the one closest to their intent, and (in a future version) uses that selection as feedback for re-ranking. The interaction model assumes ambiguity rather than fighting it.

This is the Person of Interest project's connection to the broader course theme: retrieval is not a precision instrument, it's a conversation. The geometry gets you to a neighborhood; the user does the last meter.

## What I deliberately didn't build

- **Identity matching.** This is not face *recognition* in the closed-world sense. We don't claim two photos are the same person; we claim they're similar to a description. That's a different problem with different evals.
- **Cross-encoder reranking.** Listed as next steps. The bi-encoder + VLM caption combination already does most of the lifting.
- **Fine-tuning.** Pretrained SigLIP-2 is strong enough that fine-tuning on CelebA would be a vanity move. If I hit a Recall@1 ceiling, that changes.
- **A vector database.** FAISS on disk is enough for 200k vectors. Pinecone / Qdrant / Weaviate would be reasonable choices for a multi-tenant production system, not for a portfolio project.

## Failure modes worth knowing about

The [second notebook](notebooks/02_retrieval_failure_modes.ipynb) walks through cases where the system fails interestingly:

1. **Compositional queries.** "A man with glasses but no beard" — SigLIP-2 has no negation. The model treats "no beard" as similar to "beard" because both involve the concept. Negation is a known weakness of CLIP-family models.
2. **Numerical attributes.** "A woman in her sixties." Age estimation from a single photo is hard, and CelebA's age distribution is heavily skewed young. Both contribute to poor recall on age-specific queries.
3. **Relational descriptions.** "Two people standing next to each other." CelebA is single-person crops. Out of distribution by construction.
4. **Cultural / contextual descriptions.** "Looks like she'd be a librarian." The embedding has no idea what a librarian looks like, and any pattern it picks up is a stereotype it shouldn't be reinforcing.

That last one is a fairness consideration, not just a quality one. The repo includes a brief note in the eval section about why we don't ship demographic-attribute queries as examples.

## Configuration philosophy

Every knob lives in `configs/*.yaml` and is loaded via Pydantic models in `src/poi/utils/config.py`. The CLI scripts accept `--config` overrides. Three reasons:

1. Reproducibility — the exact config is logged with every run.
2. Sweep-ability — running the encoder ablation is a one-line for-loop over configs.
3. Documentability — the YAML files double as documentation of what's tunable.
