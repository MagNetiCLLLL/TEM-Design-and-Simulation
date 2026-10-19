"""Captured CDF integration nodes use real emission and original probabilities."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from temsim.optics.electron_gun.emitter import (
    ColdFieldEmitter, EmissionSampling, EmissionQuadrature, EmissionCDFQuadrature,
)
from temsim.optics.electron_gun.tip_joint_sampling import sampling_diagnostics


def table(role="quadrature"):
    u = np.column_stack((np.linspace(.1, .9, 16), np.full(16, .1),
        np.linspace(.2, .8, 16), np.full(16, .3),
        np.linspace(.03, .97, 16), np.linspace(.98, .02, 16)))
    weights = np.arange(1., 17.)/136 if role == "quadrature" else np.zeros(16)
    return EmissionCDFQuadrature.from_arrays(u, weights,
        signs=np.ones(16, dtype=int), ray_ids=np.arange(101, 117), role=role,
        construction="test: declared CDF nodes with original unequal probabilities")


def source(record=None):
    record = table() if record is None else record
    emitter = ColdFieldEmitter(0., 1., 1., ray_count=record.total)
    emitter.cdf_quadrature = record
    return emitter


def test_explicit_positions_angles_energies_weights_and_nonzero_centroid_are_preserved():
    emitter = source()
    e = emitter.emit()
    u = np.array(emitter.cdf_quadrature.coordinates)
    sigma = emitter.virtual_source_fwhm_nm/2.354820045*1e-9
    r = sigma*np.sqrt(-2*np.log(1-u[:, 0]*(1-np.exp(-4.5))))
    angular_sigma = emitter.angular_rms_mrad*1e-3
    t = angular_sigma*np.sqrt(-2*np.log(1-u[:, 2]*(1-np.exp(-4.5))))
    assert np.allclose(e.x_m, r*np.cos(2*np.pi*u[:, 1]), rtol=0, atol=1e-24)
    assert np.allclose(e.y_m, r*np.sin(2*np.pi*u[:, 1]), rtol=0, atol=1e-24)
    assert np.allclose(e.tx_rad, t*np.cos(2*np.pi*u[:, 3]), rtol=0, atol=1e-18)
    assert np.allclose(e.ty_rad, t*np.sin(2*np.pi*u[:, 3]), rtol=0, atol=1e-18)
    assert np.sum(e.weight*e.x_m) > 1e-9
    assert np.sum(e.weight*e.tx_rad) < 0
    assert np.array_equal(e.weight, emitter.cdf_quadrature.weights)
    assert np.array_equal(e.ray_id, emitter.cdf_quadrature.ray_ids)
    assert np.ptp(e.energy_offset_ev) > .1
    assert np.all(e.energy_offset_ev+emitter.emission_energy_ev > 0)


def test_signs_change_local_geometry_only_and_curvature_keeps_physical_sag():
    record = table()
    plus = source(record).emit()
    minus_source = source(replace(record, signs=tuple([-1]*16)))
    minus = minus_source.emit()
    for name in ("x_m", "y_m", "tx_rad", "ty_rad"):
        assert np.array_equal(getattr(minus, name), -getattr(plus, name))
    assert np.array_equal(minus.energy_offset_ev, plus.energy_offset_ev)
    assert np.array_equal(minus.weight, plus.weight)
    minus_source.curvature_nm_inv = .02
    curved = minus_source.emit()
    assert np.any(curved.surface_position_m[:, 2] < 0)
    assert np.array_equal(curved.weight, plus.weight)
    assert np.allclose(np.linalg.norm(curved.surface_direction, axis=1), 1.)


def test_probe_role_has_exactly_zero_exposure_without_changing_physical_coordinates():
    probe = source(table("boundary-probes"))
    e = probe.emit()
    physical = source().emit()
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "ray_id"):
        assert np.array_equal(getattr(e, name), getattr(physical, name))
    assert np.array_equal(e.weight, np.zeros(16))
    assert probe.emitted_current_a*e.weight.sum() == 0
    assert sampling_diagnostics(probe)["role"] == "boundary-probes"
    with pytest.raises(ValueError, match="zero physical current"):
        replace(table(), role="boundary-probes").validate()
    with pytest.raises(ValueError, match="sum to one"):
        replace(table("boundary-probes"), role="quadrature").validate()


def test_record_copies_input_arrays_and_never_normalizes_original_weights():
    original = table()
    u, w = np.array(original.coordinates), np.array(original.weights)
    copied = EmissionCDFQuadrature.from_arrays(u, w)
    u[:] = .5
    w[:] = 1/16
    assert copied.coordinates == original.coordinates
    assert copied.weights == original.weights
    with pytest.raises(ValueError, match="never renormalized"):
        EmissionCDFQuadrature.from_arrays(original.coordinates, np.array(original.weights)*.9)


@pytest.mark.parametrize("change", [
    {"schema": "unknown"}, {"role": "unknown"}, {"coordinates": tuple([(.5,)*5]*16)},
    {"coordinates": tuple([(.5,)*5+(float("nan"),)]*16)},
    {"coordinates": tuple([(.5,)*5+(1.1,)]*16)},
    {"weights": tuple([float("inf")]*16)}, {"weights": tuple([-1.]*16)},
    {"weights": tuple([1e308]*16)}, {"weights": tuple([.5]*15)},
    {"signs": tuple([0]*16)}, {"signs": tuple([True]*16)},
    {"ray_ids": tuple([1]*16)}, {"ray_ids": tuple([-1]*16)},
    {"ray_ids": tuple(range(2**63, 2**63+16))},
])
def test_invalid_tables_fail_before_any_emission(change):
    with pytest.raises(ValueError):
        source(replace(table(), **change))


@pytest.mark.parametrize("count", [9, 15, 17, 16., True])
def test_emission_requires_the_complete_captured_node_table(count):
    with pytest.raises(ValueError, match="complete explicit CDF table"):
        source().emit(count)


@pytest.mark.parametrize("model", ["sampling", "quadrature", "coherence", "surface_model", "probes"])
def test_table_cannot_silently_replace_another_source_or_tuning_scheme(model):
    emitter = source()
    if model == "sampling":
        emitter.sampling = EmissionSampling()
    elif model == "quadrature":
        emitter.quadrature = EmissionQuadrature(3, 3, 3)
    elif model == "coherence":
        from temsim.optics.electron_gun.tip_coherence import TipCoherence
        emitter.coherence = TipCoherence()
    elif model == "surface_model":
        from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference
        emitter.surface_model = load_tip_surface_reference()
    else:
        emitter._tuning_boundary_probes = 33
    with pytest.raises(ValueError, match="cannot"):
        emitter.emit()


def test_payload_snapshot_profile_and_cache_identity_include_the_entire_table(tmp_path):
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.calculation_cache import calculation_signatures
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.optics.electron_gun.field_emission import _component_payload, _restore_component_settings
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    emitter = state.electron_gun.emitter
    emitter.cdf_quadrature = table()
    emitter.ray_count = 16
    before = calculation_signatures(state)
    restored = capture_instrument_snapshot(state).restore()
    assert restored.electron_gun.emitter.cdf_quadrature == table()
    assert np.array_equal(restored.electron_gun.emit().weight, emitter.emit().weight)
    payload = _component_payload(emitter)
    other = source(table("boundary-probes"))
    _restore_component_settings(other, payload)
    assert other.cdf_quadrature == table()
    path = tmp_path/"cdf.toml"
    save_profile(path, state, selection)
    selected, values = read_profile(path)
    target = default_state()
    catalog.apply(target, selected)
    apply_profile_values(target, values)
    assert target.electron_gun.emitter.cdf_quadrature == table()
    altered_u = np.array(table().coordinates)
    altered_u[0, 2] = np.nextafter(altered_u[0, 2], 1.)
    changed = EmissionCDFQuadrature.from_arrays(altered_u, table().weights,
        signs=table().signs, ray_ids=table().ray_ids, construction=table().construction)
    assert changed.digest != table().digest
    emitter.cdf_quadrature = changed
    assert calculation_signatures(state)["incident"] != before["incident"]
    # Explicit old gun controls clear the new choice; partial profiles retain it.
    apply_profile_values(target, {"objective_lens": {"percent": 13.}})
    assert target.electron_gun.emitter.cdf_quadrature == table()
    from temsim.optics.electron_gun.profile_controls import capture_gun_profile_controls
    legacy = capture_gun_profile_controls(default_state().electron_gun)
    apply_profile_values(target, {"__gun_controls__": legacy})
    assert target.electron_gun.emitter.cdf_quadrature is None


def test_damaged_incoming_table_rolls_back_whole_profile_transaction():
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.profile_controls import capture_gun_profile_controls
    from temsim.profile_io import apply_profile_values
    state = default_state()
    state.electron_gun.emitter.cdf_quadrature = table()
    state.electron_gun.emitter.ray_count = 16
    before = deepcopy(state.to_dict())
    controls = capture_gun_profile_controls(state.electron_gun)
    controls["tip_cdf_quadrature"]["weights"] = tuple([0.]*16)
    with pytest.raises(ValueError, match="sum to one"):
        apply_profile_values(state, {"__gun_controls__": controls, "objective_lens": {"percent": 13.}})
    assert state.to_dict() == before
