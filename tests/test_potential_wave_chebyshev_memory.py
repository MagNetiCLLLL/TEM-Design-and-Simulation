"""Real CUDA workspace bounds for complete grouped residual exponentials."""
import numpy as np
import pytest

from temsim.physics.compute_backend import cupy_capability
from temsim.physics.potential_wave_chebyshev import chebyshev_working_bytes
from temsim.physics.potential_wave_device import (
    RESIDUAL_DEVICE_BYTES_PER_CELL, execute_residual_exponential,
)
from temsim.physics.wave_magnetic_residual import _series_plan
from test_potential_wave_residual_memory import _input


@pytest.mark.parametrize("shape", [(127, 131), (128, 160), (512, 512)])
@pytest.mark.parametrize("order,scalar,quartic,higher,group", [
    (2, False, False, False, 8), (2, True, False, False, 8),
    (4, False, False, False, 8), (4, True, False, False, 8),
    (4, True, True, False, 8), (6, False, False, False, 8),
    (6, True, False, True, 8), (8, False, False, False, 8),
    (8, True, False, True, 8), (8, False, False, False, 1),
    (8, True, False, True, 1),
])
def test_actual_cuda_grouped_workspace_covers_supported_variants(shape, order, scalar, quartic, higher, group):
    if not cupy_capability().available:
        pytest.skip("Optional real CUDA device unavailable")
    import cupy as cp

    class Measure(cp.cuda.MemoryHook):
        name = "TEMForgeGroupedResidualWorkspaceRegression"
        def __init__(self):
            self.live, self.peak = {}, 0
        def malloc_postprocess(self, **row):
            if row["mem_ptr"]:
                self.live[row["pmem_id"]] = row["mem_size"]
                self.peak = max(self.peak, sum(self.live.values()))
        def free_postprocess(self, **row):
            self.live.pop(row["pmem_id"], None)

    args = _input(shape)
    cells = args[0].size
    count, degree, bound = _series_plan(16.25, 1e-12)
    planned = chebyshev_working_bytes(cells, group, base_bytes_per_cell=384)
    limit = max(planned, RESIDUAL_DEVICE_BYTES_PER_CELL*cells)
    options = dict(phase=args[2]*25, scalar_splitting=scalar, kinetic_order=order,
        quartic_scalar_splitting=quartic, higher_order_scalar_splitting=higher,
        count=count, degree=degree, k_distance=16.25/1e-5,
        boundary_limit=.99, envelope_limit=.99, cancelled=lambda: False,
        exponential_method="chebyshev", generator_norm_bound=1e-5,
        series_error_budget=bound, maximum_chebyshev_group=group,
        maximum_device_working_bytes=limit)
    original = args[0].copy()
    measured = Measure()
    with measured:
        actual, record = execute_residual_exponential(*args, **options, requested="Require GPU")
    assert record["compute_backend"] == "cupy" and record["fallback_reason"] is None
    assert record["exponential_method"] == "chebyshev"
    certificate = record["chebyshev_certificate"]
    assert certificate["planning_base_bytes_per_cell"] == 384
    assert max(certificate["group_sizes"]) == group
    assert certificate["estimated_working_bytes"] == planned
    assert measured.peak <= limit and not measured.live
    np.testing.assert_array_equal(args[0], original)
    if shape == (127, 131):
        expected, _ = execute_residual_exponential(*args, **options, requested="CPU")
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-13


def test_actual_gpu_reduces_group_under_device_limit_without_using_cpu_base():
    if not cupy_capability().available:
        pytest.skip("Optional real CUDA device unavailable")
    args = _input((127, 131))
    count, degree, bound = _series_plan(16.25, 1e-12)
    limit = chebyshev_working_bytes(args[0].size, 2, base_bytes_per_cell=384)
    options = dict(phase=args[2]*25, scalar_splitting=True, kinetic_order=8,
        quartic_scalar_splitting=False, higher_order_scalar_splitting=True,
        count=count, degree=degree, k_distance=16.25/1e-5,
        boundary_limit=.99, envelope_limit=.99, cancelled=lambda: False,
        exponential_method="chebyshev", generator_norm_bound=1e-5,
        series_error_budget=bound, maximum_chebyshev_group=8,
        maximum_device_working_bytes=limit)
    actual, record = execute_residual_exponential(*args, **options, requested="Require GPU")
    certificate = record["chebyshev_certificate"]
    assert max(certificate["group_sizes"]) == 2
    assert certificate["planning_base_bytes_per_cell"] == 384
    expected, cpu = execute_residual_exponential(*args, **options, requested="CPU")
    assert cpu["chebyshev_certificate"]["planning_base_bytes_per_cell"] == 640
    assert max(cpu["chebyshev_certificate"]["group_sizes"]) == 8
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-13
