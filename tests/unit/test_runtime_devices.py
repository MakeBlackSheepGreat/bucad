"""Unit tests for runtime devices."""

from __future__ import annotations

from src.engine.devices import resolve_torch_device
from src.utils import runtime


class _Cuda:
    """Represent Cuda for this module."""
    def __init__(self, available: bool) -> None:
        """Initialize this lightweight test helper."""
        self._available = available

    def is_available(self) -> bool:
        """Run is available."""
        return self._available


class _Torch:
    """Represent Torch for this module."""
    def __init__(self, cuda_available: bool) -> None:
        """Initialize this lightweight test helper."""
        self.cuda = _Cuda(cuda_available)


def test_auto_device_prefers_cuda_when_available() -> None:
    """Verify auto device prefers cuda when available."""
    assert resolve_torch_device("auto", _Torch(cuda_available=True)) == "cuda"


def test_cuda_request_falls_back_to_cpu_when_unavailable() -> None:
    """Verify cuda request falls back to cpu when unavailable."""
    assert resolve_torch_device("cuda", _Torch(cuda_available=False)) == "cpu"


def test_explicit_cpu_is_stable() -> None:
    """Verify explicit cpu is stable."""
    assert resolve_torch_device("cpu", _Torch(cuda_available=True)) == "cpu"


def test_runtime_select_device_uses_shared_resolver(monkeypatch) -> None:
    """Verify runtime select device uses shared resolver."""
    monkeypatch.setattr(runtime, "optional_import", lambda _: _Torch(cuda_available=False))

    assert runtime.select_device("cuda") == "cpu"
