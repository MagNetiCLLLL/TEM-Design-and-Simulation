"""Tiny CUDA-simulator dispatch checks, not physical GPU qualification."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_lazy_ray_and_electric_kernels_execute_in_cpu_cuda_simulator():
    source = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from types import SimpleNamespace
        import numpy as np

        class NoNativeGPU(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                # CUDASIM installs its driver stub into sys.modules before this
                # name is requested; a real driver's import must never occur.
                if (fullname == "numba.cuda.cudadrv.driver"
                        or fullname == "cupy" or fullname.startswith("cupy.")):
                    raise AssertionError("Native GPU import: " + fullname)
                return None

        sys.meta_path.insert(0, NoNativeGPU())
        from temsim.physics import ray_integrator as rays
        from temsim.physics import electrostatic_column_transport as electric
        assert rays.cuda is None and electric.cuda is None
        assert "numba.cuda" not in sys.modules

        # Two steps and two rays exercise the same scalar RK4 formula, output
        # ownership and optional particle clock used by real device kernels.
        n = 2
        stages = 2*n+1
        x = np.array([1.e-6, -2.e-6])
        inputs = (
            np.full(stages, 200.), np.full(stages, -200.),
            np.zeros(stages), np.zeros(stages), np.zeros(stages), np.ones(2),
            np.zeros(n+1), np.zeros(n+1), np.zeros(n+1), np.full(n, 1.e-4),
            x, x*100., -x*.5, x*30., np.zeros(n+1), np.zeros(n+1),
            np.array([0, n], dtype=np.int64), np.array([n], dtype=np.int64),
            np.full(stages, 80.), np.zeros(3*n), np.zeros(3*n), np.ones(1),
        )
        expected = rays.vectorised_rk4(*inputs)
        actual = rays.cuda_rk4(*inputs)
        for got, wanted in zip(actual, expected):
            np.testing.assert_allclose(got, wanted, rtol=3e-7, atol=1e-20)
        original_kernels = (rays._cuda_rk4_kernel, rays._cuda_time_rk4_kernel)
        timing = dict(initial_time_s=np.array([0., 1.e-9]),
                      inverse_speed=np.full(2, 4.e-9))
        expected = rays.vectorised_rk4(*inputs, **timing)
        actual = rays.cuda_rk4(*inputs, **timing)
        for got, wanted in zip(actual, expected):
            np.testing.assert_allclose(got, wanted, rtol=3e-7, atol=1e-20)
        assert original_kernels == (rays._cuda_rk4_kernel, rays._cuda_time_rk4_kernel)
        assert electric.cuda is None

        from temsim.physics.closed_gun_field import ClosedGunField
        from temsim.physics.instrument_electric import InstrumentElectricField
        radius = np.array([0., .01, .02])
        z_grid = np.array([0., .0001, .001])
        voltage = np.broadcast_to(2.e6*z_grid, (3, 3)).copy()
        base = ClosedGunField({}, radius, z_grid, voltage)
        field = InstrumentElectricField(
            base, base, SimpleNamespace(exit_plane_z_mm=0.),
            np.array([[-.02, -.02, 0.], [.02, .02, .001]]),
            "fixture", "fixture", "fixture", (),
        )
        momentum = electric._momentum_speed(300000.)[0]
        electric_inputs = list(inputs)
        electric_inputs[5] = np.full(2, 1./momentum)
        electric_inputs[21] = np.array([momentum])
        arguments = dict(
            z_mm=np.array([0., .1, .2]), electric_field=field,
            initial_kinetic_energy_ev=np.full(2, 300000.),
            initial_time_s=np.zeros(2),
        )
        expected, actual_backend, reason = electric.electrostatic_column_rk4(
            tuple(electric_inputs), backend="CPU", **arguments)
        assert electric.cuda is None
        actual, actual_backend, reason = electric.electrostatic_column_rk4(
            tuple(electric_inputs), backend="CUDA GPU", policy="require_gpu",
            **arguments)
        assert actual_backend == "CUDA GPU" and reason is None
        for got, wanted in zip(actual, expected):
            np.testing.assert_allclose(got, wanted, rtol=3e-12, atol=1e-20)
        kernel = electric._cuda_kernel
        electric._ensure_cuda_kernel()
        assert electric._cuda_kernel is kernel
        assert "simulator" in sys.modules["numba.cuda.cudadrv.driver"].__file__
        print("CUDA simulator dispatch passed; no hardware GPU validation")
        """
    )
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("NUMBA_DISABLE_CUDA", None)
    environment["PYTHONPATH"] = str(source)
    environment.update(
        NUMBA_ENABLE_CUDASIM="1", NUMBA_NUM_THREADS="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA simulator dispatch passed" in result.stdout
