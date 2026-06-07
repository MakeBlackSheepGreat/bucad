from __future__ import annotations

from src.engine.devices import resolve_torch_device


class _Cuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _Torch:
    def __init__(self, cuda_available: bool) -> None:
        self.cuda = _Cuda(cuda_available)


def test_auto_device_prefers_cuda_when_available() -> None:
    assert resolve_torch_device("auto", _Torch(cuda_available=True)) == "cuda"


def test_cuda_request_falls_back_to_cpu_when_unavailable() -> None:
    assert resolve_torch_device("cuda", _Torch(cuda_available=False)) == "cpu"


def test_explicit_cpu_is_stable() -> None:
    assert resolve_torch_device("cpu", _Torch(cuda_available=True)) == "cpu"
