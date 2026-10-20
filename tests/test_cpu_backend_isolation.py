"""CPU imports must not load optional CUDA modules or touch native drivers."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_cpu_imports_and_disabled_discovery_never_import_cuda():
    # A fresh process catches accidental eager imports that sys.modules in the
    # test runner can hide. A finder rejects imports before native code loads.
    source = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        def is_gpu(name):
            return (name == "numba.cuda" or name.startswith("numba.cuda.")
                    or name == "cupy" or name.startswith("cupy.")
                    or name == "cupy_backends" or name.startswith("cupy_backends."))

        class NoGPUImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if is_gpu(fullname):
                    raise AssertionError("CPU operation imported " + fullname)
                return None

        sys.meta_path.insert(0, NoGPUImports())
        from temsim.physics import compute_backend as backend
        from temsim.physics import ray_integrator as rays
        from temsim.physics import electrostatic_column_transport as electric
        from temsim.physics.ray_device_cache import DEVICE_CACHE
        from temsim.gui import main_window

        # With no visibility override, merely importing CPU transport must
        # still leave GPU discovery and kernel construction untouched.
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
        assert rays.NUMBA_AVAILABLE and callable(rays._serial_rk4)
        assert callable(electric._serial)
        assert rays.cuda is None and electric.cuda is None
        assert not any(is_gpu(name) for name in sys.modules)

        for visible in ("", "-1", " -1 "):
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
            for probe in (backend.cuda_capability, backend.cupy_capability):
                status = probe()
                assert not status.available
                assert "CUDA_VISIBLE_DEVICES" in status.detail
            actual, reason = backend.choose_ray_backend(
                "Auto", acceleration_enabled=True, ray_count=1)
            assert actual == "CPU" and "CUDA_VISIBLE_DEVICES" in reason
            actual, reason = backend.choose_wave_backend(
                "Auto", acceleration_enabled=True, work_items=1)
            assert actual == "NumPy CPU" and "CUDA_VISIBLE_DEVICES" in reason
            for attempt in (
                rays.cuda_rk4,
                electric._ensure_cuda_kernel,
                backend.cupy_module,
                lambda: backend.choose_ray_backend(
                    "Require GPU", acceleration_enabled=True, ray_count=1),
                lambda: backend.choose_wave_backend(
                    "Require GPU", acceleration_enabled=True, work_items=1),
            ):
                try:
                    attempt()
                except backend.GPUExecutionError as error:
                    assert error.category == "unavailable"
                    assert "CUDA_VISIBLE_DEVICES" in str(error)
                else:
                    raise AssertionError("Disabled CUDA was accepted")
        # Disabling devices must not relabel an existing corrupt context as an
        # ordinary unavailable device and silently allow a GPU retry.
        DEVICE_CACHE.quarantine("controlled corrupt context")
        for probe in (backend.cuda_capability, backend.cupy_capability,
                      rays.cuda_rk4, electric._ensure_cuda_kernel):
            try:
                probe()
            except backend.GPUExecutionError as error:
                assert error.category == "context_corrupted"
            else:
                raise AssertionError("Corrupt CUDA context was ignored")
        assert not any(is_gpu(name) for name in sys.modules)
        print("CPU isolation passed")
        """
    )
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["PYTHONPATH"] = str(source)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU isolation passed" in result.stdout
