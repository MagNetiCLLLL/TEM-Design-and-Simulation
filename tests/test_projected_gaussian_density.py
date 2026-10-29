"""Exact Gaussian-density operator checks using synthetic arrays only.

No finite-source transport, field solve, specimen acquisition or coherent
calculation is executed. Positions are nanometres, covariance is nm squared,
and the returned occupancy density is nm^-2.
"""

import math

import numpy as np
import pytest

from temsim.specimen import projected_scattering as model


@pytest.fixture
def native_executor(monkeypatch):
    if model._gaussian_density_native is None:
        pytest.skip("Optional native Gaussian compiler is unavailable")
    import numba

    previous_threads = numba.get_num_threads()
    numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
    monkeypatch.setattr(model, "NATIVE_MIN_PAIRS", 0)
    try:
        yield model._gaussian_density_native
    finally:
        numba.set_num_threads(previous_threads)


@pytest.fixture(params=("numpy", "native"))
def density_executor(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setattr(model, "_gaussian_density_native", None)
    else:
        request.getfixturevalue("native_executor")
    return request.param


def _reference_density(positions, occupancies, centres, covariance):
    """Independent Cholesky reference, retaining every original site row."""
    positions = np.asarray(positions, dtype=float)
    centres = np.asarray(centres, dtype=float)
    if not len(positions) or not len(centres):
        return np.zeros(len(centres))
    lower = np.linalg.cholesky(covariance)
    delta = centres[:, None, :] - positions[None, :, :]
    whitened = np.linalg.solve(lower, delta.reshape(-1, 2).T).T
    squared_radius = np.sum(whitened * whitened, axis=1).reshape(
        len(centres), len(positions)
    )
    density = np.exp(-0.5 * squared_radius)
    density /= 2.0 * math.pi * np.prod(np.diag(lower))
    return np.sum(density * np.asarray(occupancies)[None, :], axis=1)


def _rotated_covariance():
    angle = math.radians(29.0)
    rotation = np.array(((math.cos(angle), -math.sin(angle)),
                         (math.sin(angle), math.cos(angle))))
    return rotation @ np.diag((0.0001, 0.02)) @ rotation.T


@pytest.mark.parametrize(
    "covariance,tolerance",
    [
        (np.eye(2) * 0.04, 8e-14),
        (np.diag((0.001, 0.7)), 8e-14),
        (np.array(((0.015, 0.011), (0.011, 0.015))), 2e-13),
        (_rotated_covariance(), 2e-12),
        (np.array(((0.08, 0.07992), (0.07992, 0.08))), 3e-12),
        (np.array(((25.0, 7.0), (7.0, 16.0))), 8e-14),
    ],
    ids=("isotropic", "anisotropic", "correlated", "rotated",
         "strongly-correlated", "broad"),
)
def test_density_matches_independent_full_covariance_reference(covariance, tolerance, density_executor):
    rng = np.random.default_rng(81573)
    positions = rng.uniform(-0.5, 0.5, (37, 2))
    occupancies = rng.uniform(0.01, 0.9, len(positions))
    # Include exact sites and off-site points within the covariance scale.
    # Strongly correlated probes otherwise put random Cartesian points deep
    # into exponential underflow, where relative conditioning dominates.
    offsets = rng.normal(size=(19, 2)) @ np.linalg.cholesky(covariance).T
    centres = np.vstack((positions[::3], positions[:19] + 0.7 * offsets))
    expected = _reference_density(positions, occupancies, centres, covariance)
    actual = model._gaussian_density(positions, occupancies, centres, covariance)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=1e-300)
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0.0)


def test_duplicate_sites_keep_occupancy_sum_without_normalizing_weights(density_executor):
    positions = np.array(((0.0, 0.0), (0.3, -0.1), (0.0, 0.0),
                          (-0.2, 0.4), (0.3, -0.1)))
    occupancies = np.array((0.15, 0.3, 0.7, 0.25, 0.6))
    centres = np.array(((0.0, 0.0), (0.1, -0.2), (-0.3, 0.4)))
    covariance = np.array(((0.07, -0.013), (-0.013, 0.04)))
    actual = model._gaussian_density(positions, occupancies, centres, covariance)
    expected = _reference_density(positions, occupancies, centres, covariance)
    np.testing.assert_allclose(actual, expected, rtol=4e-14)
    scaled = model._gaussian_density(positions, occupancies * 3.0, centres, covariance)
    np.testing.assert_allclose(scaled, actual * 3.0, rtol=4e-14)


def test_adjacent_float_sites_are_not_rounded_into_one_atomic_column(density_executor):
    positions = np.array(((1.0, 0.0), (np.nextafter(1.0, 2.0), 0.0)))
    occupancies = np.array((0.25, 0.75))
    covariance = np.diag((1e-33, 4e-33))
    expected = _reference_density(positions, occupancies, positions, covariance)
    actual = model._gaussian_density(positions, occupancies, positions, covariance)
    np.testing.assert_allclose(actual, expected, rtol=5e-14)
    assert actual[1] > 2.9 * actual[0]


def test_permuted_repeated_and_nonrectangular_scan_centres_keep_point_values(density_executor):
    positions = np.array(((-0.4, 0.2), (0.1, -0.3), (0.5, 0.7)))
    occupancies = np.array((0.2, 0.6, 0.3))
    centres = np.array(((0.0, 0.0), (0.17, 0.52), (-0.3, 0.11), (1.2, -0.9)))
    covariance = np.array(((0.13, 0.05), (0.05, 0.09)))
    baseline = model._gaussian_density(positions, occupancies, centres, covariance)
    order = np.array((2, 0, 3, 2, 1, 0))
    actual = model._gaussian_density(positions, occupancies, centres[order], covariance)
    np.testing.assert_allclose(actual, baseline[order], rtol=3e-14)
    for index, centre in enumerate(centres):
        single = model._gaussian_density(positions, occupancies, centre[None], covariance)
        np.testing.assert_allclose(single, baseline[index:index + 1], rtol=3e-14)


def test_large_common_offset_preserves_density_without_recentering_the_probe(density_executor):
    # Binary fractions retain their exact displacements under this offset.
    positions = np.array(((-0.25, 0.125), (0.5, -0.75), (0.125, 0.25)))
    centres = np.array(((0.125, -0.25), (-0.5, 0.375), (0.75, 0.125)))
    occupancies = np.array((0.25, 0.5, 0.75))
    covariance = np.array(((0.05, 0.01), (0.01, 0.09)))
    offset = np.array((2.0**20, -2.0**19))
    expected = model._gaussian_density(positions, occupancies, centres, covariance)
    actual = model._gaussian_density(
        positions + offset, occupancies, centres + offset, covariance
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-14)


def test_global_roi_sites_are_not_truncated_at_a_per_pixel_sigma_radius(density_executor):
    # A site inside the caller's common ROI still contributes at distant scan
    # points. The operator must not silently introduce a per-point cutoff.
    positions = np.zeros((1, 2))
    occupancies = np.array((0.7,))
    centres = np.array(((0.0, 0.0), (8.5, 0.0), (12.0, 0.0)))
    covariance = np.eye(2)
    actual = model._gaussian_density(positions, occupancies, centres, covariance)
    expected = 0.7 * np.exp(-0.5 * centres[:, 0]**2) / (2.0 * math.pi)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0.0)
    assert np.all(actual > 0.0)


@pytest.mark.parametrize("site_count,centre_count", ((0, 0), (0, 4), (3, 0)))
def test_empty_positions_or_centres_return_correct_empty_or_zero_density(site_count, centre_count, density_executor):
    positions = np.zeros((site_count, 2))
    centres = np.zeros((centre_count, 2))
    actual = model._gaussian_density(positions, np.ones(site_count), centres, np.eye(2))
    np.testing.assert_array_equal(actual, np.zeros(centre_count))
    assert actual.shape == (centre_count,)


def test_readonly_noncontiguous_inputs_keep_values_and_writeability(density_executor):
    rng = np.random.default_rng(179)
    position_storage = rng.normal(size=(18, 4))
    centre_storage = rng.normal(size=(14, 4))
    occupancy_storage = rng.uniform(0.1, 0.9, 36)
    covariance_storage = np.array(((0.2, 99.0, -0.03, 99.0),
                                   (99.0, 99.0, 99.0, 99.0),
                                   (-0.03, 99.0, 0.3, 99.0),
                                   (99.0, 99.0, 99.0, 99.0)))
    arrays = (position_storage[:, ::2], occupancy_storage[::2],
              centre_storage[:, ::2], covariance_storage[::2, ::2])
    saved = tuple(array.copy() for array in arrays)
    for array in arrays:
        assert not array.flags.c_contiguous
        array.setflags(write=False)
    actual = model._gaussian_density(*arrays)
    np.testing.assert_allclose(actual, _reference_density(*saved), rtol=8e-14, atol=0.0)
    for array, original in zip(arrays, saved, strict=True):
        np.testing.assert_array_equal(array, original)
        assert not array.flags.writeable


def _small_inputs():
    positions = np.array(((0.0, 0.0), (0.1, 0.0), (0.0, 0.3), (-0.2, 0.1)))
    occupancies = np.array((0.1, 0.2, 0.3, 0.4))
    centres = np.array(((0.0, 0.0), (0.15, -0.1), (0.03, 0.4)))
    covariance = np.array(((0.07, 0.012), (0.012, 0.04)))
    return positions, occupancies, centres, covariance


def test_native_budget_admits_exact_boundary_and_reports_actual_work(monkeypatch, native_executor):
    inputs = _small_inputs()
    monkeypatch.setattr(model, "MAX_PAIR_EVALUATIONS", 11)
    monkeypatch.setattr(model, "MAX_NATIVE_PAIR_EVALUATIONS", 12)
    details = {}
    actual = model._gaussian_density(*inputs, diagnostics=details)
    np.testing.assert_allclose(actual, _reference_density(*inputs), rtol=5e-14)
    assert details == {
        "executor": "numba_direct_compensated",
        "distinct_projected_sites": 4,
        "scan_points": 3,
        "pair_evaluations": 12,
        "work_budget": 12,
        "per_pixel_spatial_cutoff": False,
    }

    def forbidden(*_args):
        pytest.fail("An over-budget request executed its Gaussian kernel")

    monkeypatch.setattr(model, "_gaussian_density_native", forbidden)
    monkeypatch.setattr(model, "MAX_NATIVE_PAIR_EVALUATIONS", 11)
    untouched = {"previous": "retained"}
    with pytest.raises(ValueError, match="12 atom-pixel pairs.*compiled CPU.*11"):
        model._gaussian_density(*inputs, diagnostics=untouched)
    assert untouched == {"previous": "retained"}


def test_budget_counts_exact_distinct_sites_without_discarding_occupancy(monkeypatch, native_executor):
    positions, weights, centres, covariance = _small_inputs()
    positions = np.repeat(positions, 3, axis=0)
    weights = np.repeat(weights / 3.0, 3)
    monkeypatch.setattr(model, "MAX_PAIR_EVALUATIONS", 11)
    monkeypatch.setattr(model, "MAX_NATIVE_PAIR_EVALUATIONS", 12)
    details = {}
    actual = model._gaussian_density(positions, weights, centres, covariance, diagnostics=details)
    np.testing.assert_allclose(
        actual, _reference_density(positions, weights, centres, covariance), rtol=5e-14
    )
    assert details["distinct_projected_sites"] == 4
    assert details["pair_evaluations"] == 12


def test_optional_native_absence_uses_numpy_budget_and_exact_reference(monkeypatch):
    inputs = _small_inputs()
    monkeypatch.setattr(model, "_gaussian_density_native", None)
    monkeypatch.setattr(model, "NATIVE_MIN_PAIRS", 0)
    monkeypatch.setattr(model, "MAX_PAIR_EVALUATIONS", 12)
    details = {}
    actual = model._gaussian_density(*inputs, diagnostics=details)
    np.testing.assert_allclose(actual, _reference_density(*inputs), rtol=5e-14)
    assert details["executor"] == "numpy_direct"
    assert details["pair_evaluations"] == details["work_budget"] == 12
    monkeypatch.setattr(model, "MAX_PAIR_EVALUATIONS", 11)
    with pytest.raises(ValueError, match="12 atom-pixel pairs.*NumPy CPU.*11"):
        model._gaussian_density(*inputs)


def test_small_requests_do_not_pay_native_startup_or_receive_native_work_allowance(monkeypatch):
    def forbidden(*_args):
        pytest.fail("A request below the native crossover invoked compilation")

    monkeypatch.setattr(model, "_gaussian_density_native", forbidden)
    monkeypatch.setattr(model, "NATIVE_MIN_PAIRS", 13)
    monkeypatch.setattr(model, "MAX_PAIR_EVALUATIONS", 12)
    details = {}
    actual = model._gaussian_density(*_small_inputs(), diagnostics=details)
    np.testing.assert_allclose(actual, _reference_density(*_small_inputs()), rtol=5e-14)
    assert details["executor"] == "numpy_direct"
    monkeypatch.setattr(model, "MAX_PAIR_EVALUATIONS", 11)
    with pytest.raises(ValueError, match="NumPy CPU"):
        model._gaussian_density(*_small_inputs())


def test_native_blocks_bound_pair_work_and_keep_every_scan_centre(monkeypatch, native_executor):
    positions, weights, _centres, covariance = _small_inputs()
    centres = np.random.default_rng(827).normal(scale=0.2, size=(19, 2))
    blocks = []

    def record(positions, weights, centres, inverse, normalization):
        blocks.append(centres.copy())
        assert len(positions) * len(centres) <= 8
        return native_executor(positions, weights, centres, inverse, normalization)

    monkeypatch.setattr(model, "_gaussian_density_native", record)
    monkeypatch.setattr(model, "NATIVE_BLOCK_PAIRS", 8)
    actual = model._gaussian_density(positions, weights, centres, covariance)
    assert len(blocks) == 10
    np.testing.assert_array_equal(np.concatenate(blocks), centres)
    np.testing.assert_allclose(
        actual, _reference_density(positions, weights, centres, covariance), rtol=5e-14
    )


def test_native_cancellation_between_blocks_never_publishes_partial_density(monkeypatch, native_executor):
    calls = []

    def record(*args):
        calls.append(True)
        return native_executor(*args)

    def cancelled():
        if calls:
            raise RuntimeError("Cancelled after one exact block")

    monkeypatch.setattr(model, "_gaussian_density_native", record)
    monkeypatch.setattr(model, "NATIVE_BLOCK_PAIRS", 4)
    details = {"previous": "retained"}
    token = model._CANCEL_CHECK.set(cancelled)
    try:
        with pytest.raises(RuntimeError, match="Cancelled after one exact block"):
            model._gaussian_density(*_small_inputs(), diagnostics=details)
    finally:
        model._CANCEL_CHECK.reset(token)
    assert calls == [True]
    assert details == {"previous": "retained"}
    assert model._gaussian_density_native is record


def test_cancellation_before_dispatch_keeps_native_backend_available(monkeypatch):
    def forbidden(*_args):
        pytest.fail("A pre-cancelled density request reached a kernel")

    def cancelled():
        raise RuntimeError("Cancelled before Gaussian dispatch")

    monkeypatch.setattr(model, "_gaussian_density_native", forbidden)
    token = model._CANCEL_CHECK.set(cancelled)
    try:
        with pytest.raises(RuntimeError, match="Cancelled before Gaussian dispatch"):
            model._gaussian_density(*_small_inputs())
    finally:
        model._CANCEL_CHECK.reset(token)
    assert model._gaussian_density_native is forbidden


@pytest.mark.parametrize("numpy_budget", (11, 12))
def test_native_failure_discards_partial_values_and_respects_fallback_budget(monkeypatch, numpy_budget):
    calls = []

    def failing(positions, weights, centres, inverse, normalization):
        calls.append(len(centres))
        if len(calls) > 1:
            raise RuntimeError("Synthetic optional compiler failure")
        return np.full(len(centres), 1e9)

    monkeypatch.setattr(model, "_gaussian_density_native", failing)
    monkeypatch.setattr(model, "NATIVE_MIN_PAIRS", 0)
    monkeypatch.setattr(model, "NATIVE_BLOCK_PAIRS", 4)
    monkeypatch.setattr(model, "MAX_NATIVE_PAIR_EVALUATIONS", 12)
    monkeypatch.setattr(model, "MAX_PAIR_EVALUATIONS", numpy_budget)
    details = {}
    if numpy_budget < 12:
        monkeypatch.setattr(model, "_gaussian_density_numpy",
                            lambda *_a: pytest.fail("An oversized fallback reached NumPy"))
        with pytest.warns(RuntimeWarning, match="Compiled projected-probe convolution unavailable"):
            with pytest.raises(ValueError, match="12 atom-pixel pairs.*NumPy CPU.*11"):
                model._gaussian_density(*_small_inputs(), diagnostics=details)
        assert details == {}
    else:
        with pytest.warns(RuntimeWarning, match="Compiled projected-probe convolution unavailable"):
            actual = model._gaussian_density(*_small_inputs(), diagnostics=details)
        np.testing.assert_allclose(actual, _reference_density(*_small_inputs()), rtol=5e-14)
        assert details["executor"] == "numpy_direct"
        assert details["work_budget"] == 12
    assert calls == [1, 1]
    assert model._gaussian_density_native is None


def test_native_reduction_is_independent_of_cpu_thread_count(monkeypatch, native_executor):
    import numba

    positions, weights, centres, covariance = _small_inputs()
    positions = np.vstack((positions, np.random.default_rng(725).normal(size=(257, 2))))
    weights = np.r_[weights, np.linspace(0.01, 0.99, 257)]
    numba.set_num_threads(1)
    single = model._gaussian_density(positions, weights, centres, covariance)
    numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
    parallel = model._gaussian_density(positions, weights, centres, covariance)
    np.testing.assert_array_equal(single, parallel)
    np.testing.assert_allclose(
        parallel, _reference_density(positions, weights, centres, covariance), rtol=8e-14
    )
