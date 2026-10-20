"""Synthetic CUDA discovery and errors without importing a native driver."""
import sys
from types import ModuleType, SimpleNamespace


def numba_error(code, message="controlled accelerator fixture"):
    error_type = type(
        "CudaAPIError", (RuntimeError,),
        {"__module__": "numba.cuda.cudadrv.driver"},
    )
    error = error_type(message)
    error.code = code
    return error


def install_fake_numba_cuda(monkeypatch, *, is_available=lambda: False):
    import numba

    cuda = ModuleType("numba.cuda")
    cuda.__path__ = []
    cuda.is_available = is_available
    cudadrv = ModuleType("numba.cuda.cudadrv")
    cudadrv.__path__ = []
    driver = ModuleType("numba.cuda.cudadrv.driver")
    driver.driver = SimpleNamespace(is_available=True)
    cuda.cudadrv = cudadrv
    cudadrv.driver = driver
    for module in (cuda, cudadrv, driver):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(numba, "cuda", cuda, raising=False)
    return cuda
