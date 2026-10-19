"""Source quadrature checks without replacing or recentering propagated rays."""
from copy import deepcopy
from dataclasses import asdict, replace
import math

import numpy as np
import pytest
from scipy.special import roots_legendre

from temsim.optics.electron_gun.emitter import (
    ColdFieldEmitter, EmissionQuadrature, EmissionSampling,
)
from temsim.optics.electron_gun.tip_joint_sampling import (
    calibrate_energy, energy_parameters, sampling_diagnostics,
)


def source(**kwargs):
    emitter = ColdFieldEmitter(0., 1., 1., ray_count=4096, **kwargs)
    emitter.sampling = EmissionSampling(seed=3)
    return emitter


def test_nested_positions_directions_energies_and_ids_are_exact_prefixes():
    emitter = source()
    old = emitter.emit()
    new = emitter.emit(16384)
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "ray_id"):
        assert np.array_equal(getattr(old, name), getattr(new, name)[:4096]), name
    assert np.array_equal(old.weight, np.full(4096, 1/4096))
    assert np.array_equal(new.weight, np.full(16384, 1/16384))
    assert old.weight.sum() == new.weight.sum() == 1.
    assert emitter.emitted_current_a == emitter.emission_current_na*1e-9


def test_symmetry_is_at_source_and_does_not_renormalize_asymmetric_acceptance():
    emitter = source()
    bundle = emitter.emit()
    for name in ("x_m", "y_m", "tx_rad", "ty_rad"):
        values = getattr(bundle, name)
        assert np.array_equal(values[::2], -values[1::2])
    assert np.array_equal(bundle.energy_offset_ev[::2], bundle.energy_offset_ev[1::2])
    # A fixed laboratory half-plane stop breaks symmetry physically. Retain
    # original weights, nonzero transmitted centroid, and exactly half current.
    passed = bundle.x_m > 0
    assert bundle.weight[passed].sum() == .5
    assert np.sum(bundle.weight[passed]*bundle.x_m[passed]) > 0
    assert np.array_equal(bundle.weight[passed], np.full(passed.sum(), 1/4096))


def test_truncated_disk_second_moments_and_independent_marginals():
    emitter = source()
    bundle = emitter.emit(65536)
    for x, y, sigma, cutoff in (
        (bundle.x_m, bundle.y_m, emitter.virtual_source_fwhm_nm/2.354820045*1e-9,
         3*emitter.virtual_source_fwhm_nm/2.354820045*1e-9),
        (bundle.tx_rad, bundle.ty_rad, emitter.angular_rms_mrad*1e-3,
         emitter.angular_cutoff_mrad*1e-3),
    ):
        a = cutoff**2/(2*sigma**2)
        variance = sigma**2*(1-(a+1)*np.exp(-a))/(1-np.exp(-a))
        assert np.mean(x*x) == pytest.approx(variance, rel=8e-4)
        assert np.mean(y*y) == pytest.approx(variance, rel=8e-4)
        assert np.max(np.hypot(x, y)) <= cutoff
    xyte = np.column_stack((bundle.x_m, bundle.y_m, bundle.tx_rad,
                           bundle.ty_rad, bundle.energy_offset_ev))
    correlations = np.corrcoef(xyte, rowvar=False)-np.eye(5)
    assert np.max(np.abs(correlations)) < .003


def test_seed_is_reproducible_but_independent_replicates_change_the_population():
    emitter = source()
    first = emitter.emit()
    assert np.array_equal(first.x_m, emitter.emit().x_m)
    emitter.sampling = replace(emitter.sampling, seed=4)
    other = emitter.emit()
    assert not np.array_equal(first.x_m, other.x_m)
    assert not np.array_equal(first.energy_offset_ev, other.energy_offset_ev)
    assert np.array_equal(first.weight, other.weight)


def test_curved_tip_geometry_and_full_direction_are_preserved_in_pairs():
    emitter = source()
    emitter.curvature_nm_inv = .02
    bundle = emitter.emit()
    assert np.any(bundle.surface_position_m[:, 2] < 0)
    assert np.array_equal(bundle.surface_position_m[::2, :2], -bundle.surface_position_m[1::2, :2])
    assert np.array_equal(bundle.surface_position_m[::2, 2], bundle.surface_position_m[1::2, 2])
    assert np.allclose(np.linalg.norm(bundle.surface_direction, axis=1), 1., atol=3e-16)
    assert np.array_equal(bundle.surface_direction[::2, :2], -bundle.surface_direction[1::2, :2])
    assert np.array_equal(bundle.surface_direction[::2, 2], bundle.surface_direction[1::2, 2])


def test_continuous_energy_moments_converge_on_independent_doubled_cdf_rule():
    emitter = source()
    calibration = calibrate_energy(*energy_parameters(emitter))
    errors = []
    for order in (256, 1024):
        nodes, weights = roots_legendre(order)
        u = (nodes+1)/2
        w = np.outer(weights, weights)/4
        e = calibration.offsets(u[:, None], u[None, :])
        mean = float(np.sum(w*e))
        std = float(np.sqrt(np.sum(w*(e-mean)**2)))
        errors.append(np.array((abs(mean), abs(std-calibration.target_std_ev))))
        assert np.min(e+emitter.emission_energy_ev) >= emitter.minimum_kinetic_energy_ev
        assert np.max(np.abs(e)) <= emitter.energy_half_range_ev
    assert np.all(errors[1] < errors[0])
    assert np.max(errors[1]) < 5e-7
    # No batch fitting: finite-N deviations are observable, not forced to zero.
    a = emitter.emit(16).energy_offset_ev
    b = emitter.emit(65536).energy_offset_ev
    assert abs(a.mean()) > 1e-8
    assert abs(b.mean()) < 2e-5
    assert abs(b.std()-calibration.target_std_ev) < 2e-5


def test_energy_map_retains_logistic_shape_and_is_pointwise_not_batch_normalized():
    emitter = source()
    cal = calibrate_energy(*energy_parameters(emitter))
    u = np.array([.02, .2, .5, .8, .99])
    v = np.array([.1, .25, .4, .6, .9])
    raw = (emitter.young_decay_width_ev*(np.log(u)+1)
           + emitter.boersch_sigma_ev*np.log(v/(1-v))/(math.pi/math.sqrt(3)))
    raw *= cal.target_std_ev/math.hypot(emitter.young_decay_width_ev, emitter.boersch_sigma_ev)
    shape = np.clip(raw, -emitter.energy_half_range_ev, emitter.energy_half_range_ev)
    argument = cal.location+cal.scale*(shape-cal.shape_mean_ev)/cal.shape_std_ev
    expected = cal.lower_ev+(cal.upper_ev-cal.lower_ev)/(1+np.exp(-argument))-cal.mean_ev
    assert np.allclose(cal.offsets(u, v), expected, rtol=0, atol=3e-16)
    assert np.array_equal(cal.offsets(u, v), cal.offsets(np.r_[u, .3], np.r_[v, .7])[:5])


@pytest.mark.parametrize("decay,sigma", [(0., .1), (.2, 0.), (.2, .1)])
def test_unbounded_energy_is_explicitly_outside_sobol_v1_but_legacy_remains_valid(decay, sigma):
    emitter = source(energy_half_range_ev=0., young_decay_width_ev=decay, boersch_sigma_ev=sigma)
    with pytest.raises(ValueError, match="finite positive energy half range"):
        emitter.emit()
    with pytest.raises(ValueError, match="finite positive energy half range"):
        calibrate_energy(*energy_parameters(emitter))
    emitter.sampling = None
    legacy = emitter.emit(16)
    assert np.all(legacy.energy_offset_ev+emitter.emission_energy_ev > 0)


@pytest.mark.parametrize("mean,minimum,fwhm,half_range", [
    pytest.param(.3, .01, .3, .05, id="half-range-too-narrow"),
    pytest.param(.3, .29, .3, 1., id="minimum-too-close-to-mean"),
    pytest.param(1., .5, .5*2.354820045, .5, id="exact-maximum-variance"),
])
def test_finite_energy_support_rejects_incompatible_target_variance(mean, minimum, fwhm, half_range):
    # For E in [L,U], E[(E-L)(U-E)] >= 0 bounds its variance by
    # (mean-L)(U-mean). Equality requires masses only at both endpoints,
    # outside the non-degenerate continuous positive-energy map.
    emitter = source(emission_energy_ev=mean, minimum_kinetic_energy_ev=minimum,
        energy_spread_fwhm_ev=fwhm, energy_half_range_ev=half_range)
    before = deepcopy(emitter.__dict__)
    assert math.isfinite(half_range) and half_range > 0
    with pytest.raises(ValueError, match="non-degenerate bounded distribution"):
        calibrate_energy(*energy_parameters(emitter))
    with pytest.raises(ValueError, match="non-degenerate bounded distribution"):
        emitter.emit(16)
    assert emitter.__dict__ == before  # No silent spread reduction or range change.


@pytest.mark.parametrize("kwargs", [dict(energy_spread_fwhm_ev=0.),
    dict(young_decay_width_ev=0., boersch_sigma_ev=0.)])
def test_degenerate_energy_preserves_exact_monochromatic_source(kwargs):
    assert np.array_equal(source(**kwargs).emit().energy_offset_ev, np.zeros(4096))


@pytest.mark.parametrize("count", [9, 15, 17, 3000, 1048577, 4096.5, True])
def test_partial_or_invalid_net_is_rejected(count):
    with pytest.raises(ValueError, match="power-of-two"):
        source().emit(count)


@pytest.mark.parametrize("plan", [EmissionSampling(seed=-1), EmissionSampling(seed=True),
    EmissionSampling(seed=2**32), EmissionSampling(schema="unknown")])
def test_invalid_sampling_records_fail_explicitly(plan):
    with pytest.raises(ValueError, match="versioned Sobol"):
        source().sampling = plan


@pytest.mark.parametrize("incompatible", ["quadrature", "coherence", "surface_model", "probes"])
def test_new_rule_never_silently_replaces_another_physical_source(incompatible):
    emitter = source()
    if incompatible == "quadrature":
        emitter.quadrature = EmissionQuadrature(3, 3, 3)
    elif incompatible == "coherence":
        from temsim.optics.electron_gun.tip_coherence import TipCoherence
        emitter.coherence = TipCoherence()
    elif incompatible == "surface_model":
        from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference
        emitter.surface_model = load_tip_surface_reference()
    else:
        emitter._tuning_boundary_probes = 33
    with pytest.raises(ValueError, match="cannot be"):
        emitter.emit()


def test_legacy_absent_record_and_exact_output_remain_unchanged():
    emitter = ColdFieldEmitter(0., 1., 1.)
    before = emitter.emit(16)
    emitter.sampling = EmissionSampling()
    emitter.sampling = None
    assert "_emission_sampling" not in emitter.__dict__
    after = emitter.emit(16)
    for name in before.__dataclass_fields__:
        assert np.array_equal(getattr(before, name), getattr(after, name))


def test_source_payload_snapshot_signatures_and_seed_identity_round_trip():
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.calculation_cache import calculation_signatures
    from temsim.optics.electron_gun.field_emission import _component_payload, _restore_component_settings
    state = default_state()
    emitter = state.electron_gun.emitter
    legacy = _component_payload(emitter)
    old = calculation_signatures(state)
    emitter.ray_count = 4096
    emitter.sampling = EmissionSampling(seed=7)
    new = calculation_signatures(state)
    assert old["incident"] != new["incident"]
    restored = capture_instrument_snapshot(state).restore()
    assert restored.electron_gun.emitter.sampling == emitter.sampling
    assert np.array_equal(restored.electron_gun.emit().energy_offset_ev, emitter.emit().energy_offset_ev)
    other = deepcopy(emitter)
    _restore_component_settings(other, _component_payload(emitter))
    assert other.sampling == emitter.sampling
    emitter.sampling = EmissionSampling(seed=8)
    assert calculation_signatures(state)["incident"] != new["incident"]
    _restore_component_settings(other, legacy)
    assert other.sampling is None
    assert "sampling" not in _component_payload(other)


def test_profile_round_trip_and_explicit_legacy_clear_are_transactional(tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.optics.electron_gun.profile_controls import capture_gun_profile_controls
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    state.electron_gun.emitter.ray_count = 4096
    state.electron_gun.emitter.sampling = EmissionSampling(seed=11)
    path = tmp_path/"nested.toml"
    save_profile(path, state, selection)
    selected, values = read_profile(path)
    restored = default_state()
    catalog.apply(restored, selected)
    apply_profile_values(restored, values)
    assert restored.electron_gun.emitter.sampling == EmissionSampling(seed=11)
    assert restored.electron_gun.emitter.ray_count == 4096
    before = deepcopy(restored.to_dict())
    controls = capture_gun_profile_controls(restored.electron_gun)
    controls["tip_sampling"]["seed"] = -1
    with pytest.raises(ValueError):
        apply_profile_values(restored, {"__gun_controls__": controls, "objective_lens": {"percent": 12.}})
    assert restored.to_dict() == before
    apply_profile_values(restored, {"objective_lens": {"percent": 12.}})
    assert restored.electron_gun.emitter.sampling == EmissionSampling(seed=11)
    legacy_controls = capture_gun_profile_controls(default_state().electron_gun)
    apply_profile_values(restored, {"__gun_controls__": legacy_controls})
    assert restored.electron_gun.emitter.sampling is None


def test_diagnostics_are_parameter_only_and_do_not_claim_sample_convergence():
    emitter = source()
    before = sampling_diagnostics(emitter)
    emitter.ray_count = 65536
    assert sampling_diagnostics(emitter) == before
    assert before["seed"] == 3
    assert "logistic" in before["energy_calibration"]["schema"]
    assert before["energy_calibration"]["cdf_order"] == 512
