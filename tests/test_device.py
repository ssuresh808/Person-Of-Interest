"""Tests for the device utilities."""

from __future__ import annotations

from poi.utils.device import DeviceInfo, get_device_info


class TestDeviceInfo:
    def test_returns_valid_device(self) -> None:
        info = get_device_info()
        assert isinstance(info, DeviceInfo)
        assert info.device.type in {"cuda", "cpu"}
        assert info.dtype is not None

    def test_cpu_has_no_vram(self) -> None:
        info = get_device_info()
        if info.device.type == "cpu":
            assert info.vram_gb is None
            assert info.is_cuda is False
        else:
            assert info.vram_gb is not None
            assert info.vram_gb > 0
            assert info.is_cuda is True

    def test_caching(self) -> None:
        # lru_cache means we get the same instance back
        a = get_device_info()
        b = get_device_info()
        assert a is b
