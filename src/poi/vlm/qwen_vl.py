"""Qwen2.5-VL captioner.

Given a (query, image) pair, produces a one-sentence analysis explaining
which features of the description match the image and which don't. The
prompt is engineered (see prompts/explain_match.txt) to enforce conciseness
and discourage speculation about identity or demographic attributes.

Why this design rather than a generic "describe the image":
    A free-form description doesn't help the user judge retrieval quality.
    They already typed the query — they want to know how this particular
    face does or doesn't match it. The prompt frames the model as a critic
    of the match, not a narrator of the image.

This is roughly the COSTAR pattern from Week 6, applied to the VLM layer:
    Context: face image + user query
    Objective: explain the match in one sentence
    Style: concise, factual
    Tone: neutral analyst
    Audience: a user evaluating retrieval quality
    Response format: single sentence, ≤30 words

Memory: Qwen2.5-VL-7B in bf16 needs ~14 GB. Fits a 4090 shard. For 3B
variant, drop the model_name to Qwen/Qwen2.5-VL-3B-Instruct.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from PIL import Image

from poi.utils.device import get_device_info
from poi.utils.logging import get_logger

log = get_logger(__name__)


@runtime_checkable
class Captioner(Protocol):
    """Anything that can caption (image, query) → str."""

    def caption(self, image: Image.Image, query: str) -> str: ...


class QwenVLCaptioner:
    """Qwen2.5-VL caption model.

    Args:
        model_name: HuggingFace identifier.
        max_new_tokens: Cap on generated tokens. Higher than ~96 is wasteful
            for the one-sentence output we ask for.
        temperature: Sampling temperature. Low because we want descriptive
            consistency, not creativity.
        prompt_template: Path to the prompt text, relative to the package
            (e.g. "vlm/prompts/explain_match.txt") OR an absolute path.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens: int = 96,
        temperature: float = 0.2,
        prompt_template: str | Path = "vlm/prompts/explain_match.txt",
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._prompt = self._load_prompt(prompt_template)

        device_info = get_device_info()
        self._device = device_info.device
        self._dtype = device_info.dtype

        log.info(f"Loading Qwen-VL: {model_name} on {self._device} ({self._dtype})")
        self.model, self.processor = self._load_model(model_name)

    @staticmethod
    def _load_prompt(template: str | Path) -> str:
        path = Path(template)
        if path.is_absolute() and path.exists():
            return path.read_text()
        # Resolve as a package resource. importlib.resources is the modern
        # way to access files inside an installed package.
        rel = str(template)
        try:
            resource = files("poi").joinpath(rel)
            return resource.read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError) as e:
            raise FileNotFoundError(f"Prompt template not found: {template}") from e

    def _load_model(self, model_name: str):
        # Lazy import — transformers Qwen2.5-VL classes weren't always available.
        # Falls back to AutoModel for forward compatibility.
        try:
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )

            model_cls = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoProcessor

            model_cls = AutoModelForCausalLM

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        )
        model = model.to(self._device).eval()
        return model, processor

    @torch.inference_mode()
    def caption(self, image: Image.Image, query: str) -> str:
        """Generate a one-sentence match analysis."""
        prompt_text = self._prompt.format(query=query)

        # Qwen2.5-VL uses a chat template with image tokens inline.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        do_sample = self.temperature > 0.0
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else 1.0,
        )

        # Strip the prompt prefix from the generated tokens.
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]

        decoded = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        # Defensive: take only the first sentence if the model overshoots.
        for terminator in [". ", ".\n", "\n\n"]:
            if terminator in decoded:
                decoded = decoded.split(terminator)[0] + "."
                break

        return decoded
