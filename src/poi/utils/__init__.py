"""Shared utilities: logging, device management, configuration.

Imports are intentionally narrow at the top level. The device helpers pull
in torch, which is fine in production but makes lightweight tests
(config-only, no model loading) drag in CUDA libraries unnecessarily.
Import directly from submodules when you need the heavy stuff:

    from poi.utils.config import POIConfig          # no torch needed
    from poi.utils.device import get_device_info    # imports torch
    from poi.utils.logging import get_logger        # no torch needed
"""

from poi.utils.config import POIConfig
from poi.utils.logging import get_logger, setup_logging

__all__ = ["POIConfig", "get_logger", "setup_logging"]
