"""Multimodal encoders. SigLIP-2 is the default; CLIP and DeepFace exist as baselines.

Backends are not imported at package level — each one pulls in torch and
transformers, which is heavy. Import what you need:

    from poi.embeddings.base import Encoder              # protocol only, no torch
    from poi.embeddings.factory import build_encoder     # the usual entry point
    from poi.embeddings.siglip import SigLIPEncoder      # if you want to bypass the factory
"""

from poi.embeddings.base import Encoder

__all__ = ["Encoder"]
