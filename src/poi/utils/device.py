"""Device and dtype helpers.

The cluster has a mix of 3090, 4090, 5090, and RTX PRO 6000 cards. We do
not assume any particular device is available — the code falls back to CPU
for tests and CI, and picks bf16 over fp16 when the hardware supports it
(Ampere and later).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from poi.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    dtype: torch.dtype
    name: str
    vram_gb: float | None  # None on CPU

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


@lru_cache(maxsize=1)
def get_device_info(prefer_bf16: bool = True) -> DeviceInfo:
    """Probe CUDA and pick a sensible (device, dtype) combination.

    Cached because device probing is non-trivial and the answer never changes
    during a process lifetime.

    Args:
        prefer_bf16: If True and the GPU supports it, pick bf16 over fp16.
            bf16 has the same dynamic range as fp32, which avoids the silent
            overflow issues fp16 hits on attention scores. Ampere (RTX 30xx)
            and later support it natively.
    """
    if not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        return DeviceInfo(
            device=torch.device("cpu"),
            dtype=torch.float32,
            name="cpu",
            vram_gb=None,
        )

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)

    if prefer_bf16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    log.info(f"Using {name} ({vram_gb:.1f} GB) with dtype={dtype} on device={device}")
    return DeviceInfo(device=device, dtype=dtype, name=name, vram_gb=vram_gb)


def free_vram_gb() -> float:
    """Return free VRAM in GB on the active CUDA device, or 0 on CPU."""
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / (1024**3)


def cuda_clear() -> None:
    """Clear the CUDA cache. Useful between batched index builds."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
