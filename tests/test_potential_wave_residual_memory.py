"""Actual residual allocation measurements and unchanged physical operators."""
import math

import numpy as np
import pytest

from temsim.physics.potential_wave_carrier import _residual_grid_resource_checks
from temsim.physics.potential_wave_device import (
    execute_residual_exponential, RESIDUAL_DEVICE_BYTES_PER_CELL,
)
from temsim.physics.wave_grid import WaveGridNumerics, WaveGridBudgetError


def _input(shape=(127, 131)):
    ny, nx = shape
    y, x = np.meshgrid((np.arange(ny)-ny//2)/ny,
        (np.arange(nx)-nx//2)/nx, indexing="ij")
    amplitude = np.exp(-(x*x+y*y)*100)*(1+.1j*x)*np.exp(.37j)
    amplitude /= np.linalg.norm(amplitude)
    velocity = np.array((x, y))*1e-5
    scalar = (x*x-y*y)*1e-7
    mechanical = np.array((x+.1, y-.2))*.001
    fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    frequency = np.array((fx, fy))*.002
    defect = (1.+.1*x)*1e-6
    return amplitude, velocity, scalar, mechanical, frequency, defect, 1.2


def test_device_preflight_uses_its_own_estimate_and_keeps_host_budget_separate():
    shape = (128, 160)
    bytes_required = RESIDUAL_DEVICE_BYTES_PER_CELL*math.prod(shape)
    numerics = WaveGridNumerics(maximum_device_working_bytes=bytes_required)
    require, prefer = _residual_grid_resource_checks(numerics, "Require GPU")
    assert prefer is None
    require(shape)
    with pytest.raises(WaveGridBudgetError, match="Required residual device budget"):
        require((129, 160))
    assert numerics.maximum_working_bytes == 72*1024**3
    _, preference = _residual_grid_resource_checks(numerics, "Auto")
    assert preference(shape) and not preference((129, 160))


@pytest.mark.parametrize("order,scalar,quartic,higher", [
    (2, False, False, False), (2, True, False, False),
    (4, False, False, False), (4, True, False, False), (4, True, True, False),
    (6, False, False, False), (6, True, False, True),
    (8, False, False, False), (8, True, False, True),
])
def test_actual_cuda_lifetime_estimate_covers_all_supported_taylor_variants(order, scalar, quartic, higher):
    from temsim.physics.compute_backend import cupy_capability
    if not cupy_capability().available:
        pytest.skip("Optional real CUDA device unavailable")
    import cupy as cp
    class Measure(cp.cuda.MemoryHook):
        name = "TEMForgeResidualLivenessRegression"
        def __init__(self):
            self.live, self.peak = {}, 0
        def malloc_postprocess(self, **row):
            if row["mem_ptr"]:
                self.live[row["pmem_id"]] = row["mem_size"]
                self.peak = max(self.peak, sum(self.live.values()))
        def free_postprocess(self, **row):
            self.live.pop(row["pmem_id"], None)
    args = _input()
    cells = args[0].size
    options = dict(phase=args[2]*25, scalar_splitting=scalar, kinetic_order=order,
        quartic_scalar_splitting=quartic, higher_order_scalar_splitting=higher,
        count=2, degree=12, k_distance=50., boundary_limit=1e-9,
        envelope_limit=1e-10, cancelled=lambda: False,
        maximum_device_working_bytes=RESIDUAL_DEVICE_BYTES_PER_CELL*cells)
    expected, _ = execute_residual_exponential(*args, **options, requested="CPU")
    measure = Measure()
    with measure:
        actual, evidence = execute_residual_exponential(*args, **options, requested="Require GPU")
    assert evidence["compute_backend"] == "cupy"
    assert evidence["fallback_reason"] is None
    assert evidence["device_planning_bytes_per_cell"] == RESIDUAL_DEVICE_BYTES_PER_CELL
    assert measure.peak <= options["maximum_device_working_bytes"]
    assert not measure.live
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-14
