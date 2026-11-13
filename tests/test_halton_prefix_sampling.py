"""Pure source-boundary/CDF tests; no physical gun, column, or wave jobs."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.electron_gun.emitter import (
    ColdFieldEmitter, EmissionSampling, HALTON_PREFIX_SAMPLING_SCHEMA,
    _halton_dimensions,
)
from temsim.optics.electron_gun.stable_halton_sampling import (
    fixed_energy_interpolation, halton_prefix_diagnostics,
    MAXIMUM_ENERGY_INTERPOLATION_ERROR_EV,
)
from temsim.optics.electron_gun.tip_joint_sampling import (
    calibrate_energy, energy_parameters, sampling_diagnostics,
)


def emitter(**kwargs):
    source = ColdFieldEmitter(0., 1., 1., ray_count=1000, **kwargs)
    source.sampling = EmissionSampling(schema=HALTON_PREFIX_SAMPLING_SCHEMA)
    return source


@pytest.mark.parametrize("curvature", [0., .02])
@pytest.mark.parametrize("width", [.3, 0.])
def test_source_initial_phase_is_a_bitwise_prefix_for_non_power_of_two_counts(curvature, width):
    source = emitter(energy_spread_fwhm_ev=width)
    source.curvature_nm_inv = curvature
    before = deepcopy(source.__dict__)
    bundles = [source.emit(n) for n in (1000, 2048, 16384)]
    for previous, extended in zip(bundles[:-1], bundles[1:]):
        size = len(previous.ray_id)
        for key in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "ray_id"):
            np.testing.assert_array_equal(getattr(previous, key), getattr(extended, key)[:size])
        if curvature:
            for key in ("surface_position_m", "surface_direction", "surface_normal"):
                np.testing.assert_array_equal(getattr(previous, key), getattr(extended, key)[:size])
    for n, bundle in zip((1000, 2048, 16384), bundles):
        np.testing.assert_array_equal(bundle.weight, np.full(n, 1/n))
        assert len(np.unique(np.column_stack((bundle.x_m, bundle.y_m, bundle.tx_rad,
                                             bundle.ty_rad, bundle.energy_offset_ev)), axis=0)) == n
        assert np.min(bundle.energy_offset_ev+source.emission_energy_ev) >= source.minimum_kinetic_energy_ev
        assert np.max(abs(bundle.energy_offset_ev)) <= source.energy_half_range_ev
    assert source.__dict__ == before


def test_new_rule_retains_original_halton_spatial_and_angular_nodes():
    source = emitter()
    stable = source.emit(1000)
    source.sampling = None
    legacy = source.emit(1000)
    for key in ("x_m", "y_m", "tx_rad", "ty_rad", "ray_id"):
        np.testing.assert_array_equal(getattr(stable, key), getattr(legacy, key))
    assert not np.array_equal(stable.energy_offset_ev, legacy.energy_offset_ev)
    # Legacy is reproducible as legacy, and deliberately keeps its old
    # batch-dependent map rather than being silently relabelled as the new rule.
    longer = source.emit(2048)
    assert not np.array_equal(legacy.energy_offset_ev, longer.energy_offset_ev[:1000])


def test_interpolation_table_and_bounds_do_not_depend_on_count_or_observed_samples():
    source = emitter()
    table = fixed_energy_interpolation(source)
    original = table.to_record()
    for count in (1000, 2048, 16384):
        source.ray_count = count
        bundle = source.emit()
        assert fixed_energy_interpolation(source) is table
        assert fixed_energy_interpolation(source).to_record() == original
        assert np.isfinite(bundle.energy_offset_ev).all()
    assert original["analytic_piecewise_linear_error_bound_ev"] <= MAXIMUM_ENERGY_INTERPOLATION_ERROR_EV
    assert original["independent_midpoint_maximum_error_ev"] <= MAXIMUM_ENERGY_INTERPOLATION_ERROR_EV
    with pytest.raises(ValueError):
        table.shape_ev[0] = 0.
    with pytest.raises(ValueError):
        table.kinetic_ev[0] = 0.


@pytest.mark.parametrize("kwargs", [{}, {"young_decay_width_ev": 0.}, {"boersch_sigma_ev": 0.},
    {"energy_spread_fwhm_ev": .15, "energy_half_range_ev": .15}])
def test_table_matches_independent_analytic_cdf_map_including_clipped_tails(kwargs):
    source = emitter(**kwargs)
    table = fixed_energy_interpolation(source)
    calibration = calibrate_energy(*energy_parameters(source))
    u, v = _halton_dimensions(16384, (11, 13))
    u = np.r_[0., 1e-15, 1e-12, u, 1.]
    v = np.r_[0., 1., 1e-12, v, 1.]
    before = (u.copy(), v.copy())
    actual = table.offsets(u, v)
    reference = calibration.offsets(u, v)
    allowance = 64*np.finfo(float).eps*max(1., calibration.upper_ev)
    assert np.max(abs(actual-reference)) <= table.analytic_interpolation_error_bound_ev+allowance
    np.testing.assert_array_equal(u, before[0])
    np.testing.assert_array_equal(v, before[1])
    np.testing.assert_array_equal(table.offsets(u[:1000], v[:1000]), actual[:1000])


def test_interpolation_is_monotone_and_does_not_force_finite_batch_moments():
    source = emitter()
    table = fixed_energy_interpolation(source)
    cdf = np.linspace(0., 1., 2048)
    assert np.all(np.diff(table.offsets(cdf, .5)) >= 0)
    assert np.all(np.diff(table.offsets(.5, cdf)) >= 0)
    calibration = table.calibration
    first, larger = source.emit(1000).energy_offset_ev, source.emit(16384).energy_offset_ev
    assert abs(first.mean()) > 1e-8
    assert abs(first.std()-calibration.target_std_ev) > 1e-8
    assert abs(larger.mean()) < .002
    assert abs(larger.std()-calibration.target_std_ev) < .002
    np.testing.assert_array_equal(first, larger[:1000])


@pytest.mark.parametrize("kwargs", [dict(energy_spread_fwhm_ev=0.),
    dict(young_decay_width_ev=0., boersch_sigma_ev=0.)])
def test_degenerate_width_is_exact_constant_launch_mean(kwargs):
    source = emitter(**kwargs)
    table = fixed_energy_interpolation(source)
    for count in (1000, 2048, 16384):
        energy = source.emit(count).energy_offset_ev
        np.testing.assert_array_equal(energy, np.zeros(count))
        np.testing.assert_array_equal(energy+source.emission_energy_ev,
                                      np.full(count, source.emission_energy_ev))
    assert table.midpoint_maximum_error_ev == 0


@pytest.mark.parametrize("kwargs", [dict(energy_half_range_ev=0.),
    dict(energy_half_range_ev=.05), dict(minimum_kinetic_energy_ev=.29)])
def test_incompatible_new_rule_energy_support_fails_without_legacy_fallback(kwargs):
    source = emitter(**kwargs)
    before = deepcopy(source.__dict__)
    with pytest.raises(ValueError, match="finite positive|non-degenerate bounded"):
        source.emit(1000)
    assert source.__dict__ == before


@pytest.mark.parametrize("count", [True, 1000.5, 8, 1048577])
def test_invalid_halton_count_is_not_silently_rounded(count):
    with pytest.raises(ValueError, match="integer ray count"):
        emitter().emit(count)


@pytest.mark.parametrize("seed", [True, -1, 1, 2**32])
def test_fixed_halton_rule_rejects_unused_or_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed zero"):
        EmissionSampling(schema=HALTON_PREFIX_SAMPLING_SCHEMA, seed=seed).validate()


def test_diagnostics_record_actual_interpolation_and_keep_legacy_distinct():
    source = emitter()
    record = sampling_diagnostics(source)
    assert record == halton_prefix_diagnostics(source)
    assert record["schema"] == HALTON_PREFIX_SAMPLING_SCHEMA
    assert record["halton_first_index"] == 18
    assert record["antithetic_pair_size"] == 1
    assert "np.interp" in record["energy_interpolation"]["method"]
    source.sampling = None
    assert sampling_diagnostics(source)["schema"] == "legacy-joint-halton-batch-moments"
    source.sampling = EmissionSampling(seed=3)
    assert sampling_diagnostics(source)["schema"] == "tip-sobol-antithetic-v1"


def test_sampling_schema_survives_existing_component_payload_without_legacy_relabelling():
    from temsim.optics.electron_gun.field_emission import _component_payload, _restore_component_settings
    original = emitter()
    target = emitter()
    payload = _component_payload(original)
    _restore_component_settings(target, payload)
    assert target.sampling == original.sampling
    np.testing.assert_array_equal(target.emit(1000).energy_offset_ev, original.emit(1000).energy_offset_ev)
    legacy = dict(payload)
    legacy.pop("sampling")
    _restore_component_settings(target, legacy)
    assert target.sampling is None
    assert "sampling" not in _component_payload(target)
