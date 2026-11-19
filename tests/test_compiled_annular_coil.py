"""Compiled/native annular field comparisons; no source or transport trials."""
from dataclasses import fields
import subprocess
import sys

import numpy as np
import pytest

from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics import compiled_annular_coil as compiled


class NumPyCoil(AnnularCoilPotential):
    """Explicit original evaluator; no production methods are replaced."""
    def axial_derivatives(self, local_z_m, maximum_order=0):
        return self.axial_derivatives_numpy(local_z_m, maximum_order)


def coil(**changes):
    return AnnularCoilPotential(**{**dict(source_id="compiled-reference", inner_radius_m=.006,
        outer_radius_m=.010, length_m=.010, ampere_turns=3000.), **changes})


def numpy_coil(source):
    return NumPyCoil(**{field.name: getattr(source, field.name) for field in fields(source) if field.init})


@pytest.mark.parametrize("order", [0, 1, 4, 11, 12])
@pytest.mark.parametrize("quadrature", [4, 12, 24])
def test_all_retained_axis_orders_match_original_direct_powers(order, quadrature):
    source = coil(quadrature_order=quadrature)
    z = np.array([[-6., -.12, -.005, -.003, 0., .003, .005, .12, 6.]])
    expected = source.axial_derivatives_numpy(z, order)
    actual = source.axial_derivatives(z, order)
    assert actual.shape == expected.shape and np.isfinite(actual).all()
    for reference, result in zip(expected, actual):
        # The existing field/potential comparisons use 3e-12 relative error;
        # per-order scaling avoids dividing by symmetric derivative zeros.
        scale = max(float(np.abs(reference).max()), np.finfo(float).tiny)
        assert float(np.abs(reference-result).max()) <= 3e-12*scale


@pytest.mark.parametrize("current", [3000., -6000., 0.])
def test_near_axis_branch_faces_and_full_tails_keep_existing_field_tolerance(current):
    source = coil(ampere_turns=current)
    reference = numpy_coil(source)
    points = []
    for z in (-6., -.12, -.005, np.nextafter(-.005, 0.), 0., .003, .005, .12, 6.):
        limit = .01*np.hypot(source.inner_radius_m, max(abs(z)-source.length_m/2, 0.))
        for radius in (0., limit*.1, np.nextafter(limit, 0.), limit, np.nextafter(limit, np.inf)):
            points.append((radius, 0., z))
    points = np.array(points)
    expected, actual = reference.sample(points), source.sample(points)
    np.testing.assert_allclose(actual["B_t"], expected["B_t"], rtol=2e-12, atol=2e-15)
    np.testing.assert_allclose(actual["A_t_m"], expected["A_t_m"], rtol=2e-12, atol=2e-18)
    expected_jet, actual_jet = reference.sample_jet(points), source.sample_jet(points)
    for key in ("jac_A_t", "hess_A_t_per_m"):
        np.testing.assert_allclose(actual_jet[key], expected_jet[key], rtol=3e-11, atol=3e-14)


def test_scalar_b_kernel_reuses_same_six_terms_and_pairwise_sum():
    source = coil()
    reference = numpy_coil(source)
    data = compiled.pack_annular_axis(source._loops)
    for point in ((0., 0., 0.), (1e-6, -2e-6, .003), (.0005, .00025, 6.)):
        assert source._near_axis(np.asarray([point]))[0]
        supported, actual = compiled.annular_near_axis_b_one(*point, *data)
        assert supported
        np.testing.assert_allclose(actual, reference.field_at_global_positions_t(point), rtol=2e-12, atol=2e-15)
    supported, _ = compiled.annular_near_axis_b_one(0., 0., 1e300, *data)
    assert not supported


def test_far_axis_elliptic_branch_is_not_redirected_to_axis_acceleration(monkeypatch):
    source = coil()
    def forbidden(*args):
        pytest.fail("The original far-axis branch must not call the axis kernel")
    monkeypatch.setattr(compiled, "compiled_axis_derivatives", forbidden)
    source.sample([[.002, .001, .003]])


@pytest.mark.parametrize("failure", ["unavailable", "backend", "nonfinite"])
def test_unavailable_or_unsafe_compilation_executes_exact_numpy_fallback(monkeypatch, failure):
    source = coil()
    z = np.array([-.012, 0., .004, .02])
    expected = source.axial_derivatives_numpy(z, 12)
    if failure == "unavailable":
        monkeypatch.setattr(compiled, "njit", None)
    elif failure == "backend":
        def failed(*args):
            raise RuntimeError("Controlled compilation failure")
        monkeypatch.setattr(compiled, "_cached_axis_values", failed)
    else:
        monkeypatch.setattr(compiled, "_cached_axis_values", lambda *args: None)
    np.testing.assert_array_equal(source.axial_derivatives(z, 12), expected)


def test_exact_repeated_z_cache_is_immutable_and_return_values_are_owned():
    source = coil()
    z = np.array([.003, 0., .003, -.003])
    compiled._cached_axis_values.cache_clear()
    expected = source.axial_derivatives(z, 12)
    previous = compiled._cached_axis_values.cache_info()
    mutated = source.axial_derivatives(z, 12)
    mutated[:] = 999.
    np.testing.assert_array_equal(source.axial_derivatives(z, 12), expected)
    assert compiled._cached_axis_values.cache_info().hits >= previous.hits+2
    assert not np.shares_memory(mutated, expected)
    for array in compiled.pack_annular_axis(source._loops):
        assert not array.flags.writeable


def test_large_axis_queries_do_not_add_another_full_population_cache(monkeypatch):
    source = coil()
    def forbidden(*args):
        pytest.fail("An axis output larger than 256 KiB must bypass the value cache")
    monkeypatch.setattr(compiled, "_cached_axis_values", forbidden)
    z = np.linspace(-.02, .02, 2600)
    result = source.axial_derivatives(z, 12)
    assert result.shape == (13, len(z)) and np.isfinite(result).all()


def test_compiled_pairwise_kernel_cache_reloads_in_another_python_process():
    script = (
        "import numpy as np; from temsim.physics.annular_coil_potential import AnnularCoilPotential; "
        "c=AnnularCoilPotential('process-check',.006,.01,.01,3000.); "
        "from temsim.physics.compiled_annular_coil import pack_annular_axis,annular_near_axis_b_one; "
        "ok,b=annular_near_axis_b_one(1e-6,2e-6,.003,*pack_annular_axis(c._loops)); "
        "assert ok and np.isfinite(b).all()"
    )
    for _ in range(2):
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout+completed.stderr


@pytest.mark.parametrize("count", [16, 25, 64, 128, 129, 144, 169, 256, 529, 576, 961, 4096])
def test_nonrecursive_sum_preserves_numpy_tree_through_maximum_declared_quadrature(count):
    values = np.random.default_rng(20261114).normal(size=count)
    assert compiled._pairwise_sum(values, 0, count) == np.sum(values)


def test_field_operator_does_not_modify_winding_or_source_metadata():
    source = coil()
    before = source.to_dict()
    source.sample([[1e-6, 2e-6, .003]])
    assert source.to_dict() == before


@pytest.mark.parametrize("changes,z", [
    ({"inner_radius_m": 1e30, "outer_radius_m": 2e30, "length_m": 1e30, "ampere_turns": 1e300}, [0., 1e29]),
    ({"inner_radius_m": 1e-30, "outer_radius_m": 2e-30, "length_m": 1e-30, "ampere_turns": 1e-300}, [0., 1e-31]),
    ({"ampere_turns": 1e-200}, [0., .003]),
    ({"ampere_turns": 1e200}, [0., .003]),
    ({}, [1e6, -1e6]),
])
def test_accelerator_range_misses_preserve_original_numpy_values_or_failure(changes, z):
    source = coil(**changes)
    positions = np.array(z)
    assert compiled.compiled_axis_derivatives(positions, source._loops, 12) is None
    with np.errstate(all="ignore"):
        try:
            reference = source.axial_derivatives_numpy(positions, 12)
        except ValueError:
            with pytest.raises(ValueError, match="finite numerical range"):
                source.axial_derivatives(positions, 12)
        else:
            np.testing.assert_array_equal(source.axial_derivatives(positions, 12), reference)


def test_microscopy_range_guard_does_not_truncate_finite_far_field():
    source = coil()
    position = np.array([[0., 0., 1e6]])
    expected = numpy_coil(source).field_at_global_positions_t(position)
    actual = source.field_at_global_positions_t(position)
    assert expected[0, 2] > 0.
    np.testing.assert_array_equal(actual, expected)
