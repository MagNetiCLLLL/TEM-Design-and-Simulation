"""Actual installed-gun tests, plus independent analytic Hamiltonian limits.

These qualify the stated quadratic gun operator, not the missing full chain.
No retired exit emission, source-policy override or synthetic exit pupil is used.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, m_e

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipWaveNumerics, wavelength_m
from temsim.physics.tip_gun_wave import (
    GunWaveNumerics, _energy_transport, _field_coefficients, _momentum_velocity, build_tip_gun_checkpoint,
)

SOURCE = TipWaveNumerics(energy_samples=1)
NUMERICS = GunWaveNumerics(field_step_mm=.05, bore_step_mm=2.)


@pytest.mark.parametrize("value", [True, np.bool_(True), "12.5", None, 1+0j,
    0., -1., np.nan, np.inf, -np.inf, 10**1000])
def test_near_tip_step_rejects_nonphysical_numeric_inputs_before_source_execution(value, monkeypatch):
    from temsim.physics import tip_gun_wave
    monkeypatch.setattr(tip_gun_wave, "generate_tip_emission", lambda *_a, **_k:
                        pytest.fail("Invalid numerical settings must not execute the source"))
    with pytest.raises(ValueError, match="near_tip_maximum_step_nm"):
        build_tip_gun_checkpoint(FieldEmissionGun(),
            numerics=replace(NUMERICS, near_tip_maximum_step_nm=value))


@pytest.mark.parametrize("step", [None, 12.5])
def test_driven_gun_passes_explicit_near_tip_step_without_changing_physical_source(step, monkeypatch):
    from temsim.optics.column import default_state
    from temsim.physics import accelerating_tip_boundary
    from temsim.instrument_snapshot import encode_instrument
    state = default_state()
    state.electron_gun.emitter.coherence = TipCoherence(boundary_model="driven_gaussian_schell")
    state.electron_gun.emitter.energy_spread_fwhm_ev = 0.
    before = encode_instrument(state)
    numerical = GunWaveNumerics() if step is None else GunWaveNumerics(near_tip_maximum_step_nm=step)
    expected = accelerating_tip_boundary.AcceleratingTipNumerics()
    received = []

    class StopBeforeTransport(Exception):
        pass

    def capture(**values):
        received.append(values)
        raise StopBeforeTransport

    monkeypatch.setattr(accelerating_tip_boundary, "AcceleratingTipNumerics", capture)
    with pytest.raises(StopBeforeTransport):
        build_tip_gun_checkpoint(state.electron_gun,
            source_numerics=TipWaveNumerics(grid_pixels=32, energy_samples=1),
            numerics=numerical, _column_state=state)
    assert received == [{"maximum_working_bytes": numerical.maximum_checkpoint_bytes,
        "source_padding_factor": numerical.near_tip_source_padding_factor,
        "maximum_step_nm": expected.maximum_step_nm if step is None else step}]
    assert encode_instrument(state) == before


def test_unified_gun_support_is_bounded_non_executable_and_preserves_source_ownership(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.simulation_modes import switch_mode
    from temsim.physics import core, unified_field_particle
    from temsim.instrument_snapshot import encode_instrument
    state = default_state()
    switch_mode(state, "unified")
    before = encode_instrument(state)
    monkeypatch.setattr(unified_field_particle, "build_unified_plan", lambda *_a, **_k:
                        pytest.fail("Gun support must not ask the post-gun executor for a source plan"))
    support = core.capture_gun_source_support(state, 0., state.electron_gun.exit_plane_z_mm,
                                              maximum_step_mm=1.)
    assert support.record["owner"] == "accelerating_gun_source"
    assert support.record["selected_column_mode"] == "unified"
    assert support.record["executable"] is False
    assert support.record["optical_actions_executed"] is False
    assert support.record["electric_field_identity"]
    assert support.z_mm[0] == 0. and support.z_mm[-1] == state.electron_gun.exit_plane_z_mm
    assert not hasattr(support, "solver_signature") and not hasattr(support, "magnetic_t")
    with pytest.raises(TypeError, match="inspection metadata"):
        core.execute_propagation_plan(state, support, *[np.zeros(1)]*4)
    with pytest.raises(TypeError):
        support.record["executable"] = True
    assert encode_instrument(state) == before
    state.lenses[0].percent += .1
    changed = core.capture_gun_source_support(state, 0., state.electron_gun.exit_plane_z_mm,
                                              maximum_step_mm=1.)
    assert changed.signature != support.signature


@pytest.mark.parametrize("start, stop, step", [(-1., 1., 1.), (0., 451., 1.),
    (1., 1., 1.), (0., 1., 0.), (0., float("nan"), 1.), (False, 1., 1.)])
def test_gun_support_cannot_describe_an_unowned_column_span(start, stop, step):
    from temsim.optics.column import default_state
    from temsim.physics.core import capture_gun_source_support
    with pytest.raises(ValueError, match="inside.*gun exit"):
        capture_gun_source_support(default_state(), start, stop, maximum_step_mm=step)


def test_gun_support_rejects_actual_nonquadratic_and_dynamic_overlaps():
    from temsim.optics.column import default_state
    from temsim.simulation_modes import switch_mode
    from temsim.physics.core import capture_gun_source_support
    state = default_state()
    switch_mode(state, "unified")
    # Moving an excited physical source into the gun changes its ownership;
    # it cannot be discarded just because the column starts at the gun exit.
    state.ac_deflector.z_mm = 100.
    with pytest.raises(ValueError, match="Time-dependent.*joint source"):
        capture_gun_source_support(state, 0., 450., maximum_step_mm=1.)
    state = default_state()
    switch_mode(state, "unified")
    state.lenses[0].z_mm = 100.
    state.lenses[0].cs_mm = 2.
    with pytest.raises(ValueError, match="Non-quadratic.*joint operator"):
        capture_gun_source_support(state, 0., 450., maximum_step_mm=1.)


def test_near_tip_hardware_inspection_never_dispatches_the_downstream_executor(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.simulation_modes import switch_mode
    from temsim.physics.planar_tip_boundary import _guard_hardware
    from temsim.physics import column_wave, unified_field_particle
    state = default_state()
    switch_mode(state, "unified")
    for module, name in ((column_wave, "_prepare_column"), (unified_field_particle, "build_unified_plan")):
        monkeypatch.setattr(module, name, lambda *_a, **_k:
                            pytest.fail("A local hardware inspection cannot execute downstream optics"))
    assert len(_guard_hardware(state, .0004, .00001)) == 64
    state.electron_gun.dpa_aperture.mechanical_center_from_tip_mm = .0002
    with pytest.raises(ValueError, match="mask cannot be omitted"):
        _guard_hardware(state, .0004, .00001)


@pytest.fixture
def gun():
    gun = FieldEmissionGun()
    gun.emitter.coherence = TipCoherence()
    gun.emitter.energy_spread_fwhm_ev = 0.
    return gun


def test_uniform_acceleration_has_correct_variable_momentum_drift_and_time():
    length = .01
    energy, gain = 100., 2900.
    field = SimpleNamespace(potential_v_at_global_positions=lambda p: gain*p[:, 2]/length)
    gun = SimpleNamespace(electric_field=field)
    p0, _ = _momentum_velocity(energy)
    p1, _ = _momentum_velocity(energy+gain)
    mass_ev = m_e*c*c/e
    exact_b = float(p0*c*length/(e*gain)*(np.arccosh(1+(energy+gain)/mass_ev)-np.arccosh(1+energy/mass_ev)))
    exact_t = float((p1-p0)*length/(e*gain))
    errors = []
    for count in (128, 256):
        z = np.linspace(0., length*1e3, count+1)
        phi = gain*(z[1:]+z[:-1])*.5/(length*1e3)
        vector, tensor = np.zeros((count, 2)), np.zeros((count, 2, 2))
        _, row = _energy_transport(gun, z, {float(z[-1])}, (phi, vector, tensor, vector, tensor), energy, lambda: False)
        matrix = np.array(row["canonical_map"])
        assert matrix[0, 0] == 1 and matrix[2, 2] == 1
        assert row["exit_axial_energy_ev"] == energy+gain
        errors.append(abs(matrix[0, 2]-exact_b))
        assert row["reference_flight_time_s"] == pytest.approx(exact_t, rel=1e-4)
    assert 3.8 < errors[0]/errors[1] < 4.2
    assert errors[1]/exact_b < 1e-4


def test_shared_axial_field_gun_operator_converges_to_exact_solenoid():
    energy, length, magnetic = 10000., .001, .001
    momentum, velocity = _momentum_velocity(energy)
    g = -e*magnetic/(2*momentum)
    angle = g*length
    rotation = np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
    exact = np.block([[np.cos(angle)*rotation, np.sin(angle)/g*rotation],
                      [-g*np.sin(angle)*rotation, np.cos(angle)*rotation]])
    gun = SimpleNamespace(electric_field=SimpleNamespace(potential_v_at_global_positions=lambda p: np.zeros(len(p))))
    errors = []
    for count in (16, 32):
        z = np.linspace(0., length*1e3, count+1)
        zeros = np.zeros(count); v = np.zeros((count, 2)); t = np.zeros((count, 2, 2))
        _, row = _energy_transport(gun, z, {float(z[-1])}, (zeros, v, t, v, t), energy,
                                  lambda: False, np.full(count, magnetic))
        errors.append(np.linalg.norm(np.array(row["canonical_map"])-exact))
        assert row["reference_flight_time_s"] == pytest.approx(length/velocity)
        assert row["reference_longitudinal_action_j_s"] == pytest.approx(float(momentum*length), rel=1e-14, abs=0)
    assert errors[0]/errors[1] == pytest.approx(4., rel=.01)


def test_constant_transverse_force_retains_weyl_phase_with_second_order_convergence():
    length, energy, force = .001, 100., .3
    p0, _ = _momentum_velocity(energy)
    gun = SimpleNamespace(electric_field=SimpleNamespace(potential_v_at_global_positions=lambda p: np.zeros(len(p))))
    actions = []
    for count in (16, 32):
        z = np.linspace(0., length*1e3, count+1)
        zeros = np.zeros(count)
        vector, tensor = np.zeros((count, 2)), np.zeros((count, 2, 2))
        magnetic = vector.copy()
        magnetic[:, 0] = force*p0/e
        _, row = _energy_transport(gun, z, {float(z[-1])}, (zeros, vector, tensor, magnetic, tensor), energy, lambda: False)
        offset = row["canonical_offset"]
        assert offset[0] == pytest.approx(force*length**2/2, rel=1e-12)
        assert offset[2] == pytest.approx(force*length, rel=1e-12)
        actions.append(row["weyl_action_m"])
    exact = force**2*length**3/12
    assert (actions[0]-exact)/(actions[1]-exact) == pytest.approx(4., rel=1e-8)






def test_gun_wave_converges_and_agrees_with_boris_in_the_paraxial_limit(gun):
    # A larger explicitly selected TIP gives small diffraction angles. This
    # is an independent near-axis validation case, not a changed default.
    gun.emitter.virtual_source_fwhm_nm = 50.
    gun.emitter.ray_count = 1024
    gun.trace_step_mm = .01
    source = replace(SOURCE, grid_pixels=256)
    coarse = build_tip_gun_checkpoint(gun, source_numerics=source, numerics=NUMERICS, use_cache=False)
    fine = build_tip_gun_checkpoint(gun, source_numerics=source,
        numerics=replace(NUMERICS, field_step_mm=.025), use_cache=False)
    finer = build_tip_gun_checkpoint(gun, source_numerics=source,
        numerics=replace(NUMERICS, field_step_mm=.0125), use_cache=False)
    matrices = [np.array(r.record["energy_transport"][0]["canonical_map"]) for r in (coarse, fine, finer)]
    assert np.linalg.norm(matrices[0]-matrices[1]) > 3*np.linalg.norm(matrices[1]-matrices[2])
    rays = gun.trace_to_exit()
    assert np.all(rays.exit_bundle.alive)
    mode = finer.beam.modes[0]
    covariance = mode.plane.canonical_covariance(float(wavelength_m(mode.energy_kev*1e3)))
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)[:2]),
                               (np.std(rays.exit_bundle.x_m), np.std(rays.exit_bundle.y_m)), rtol=.02)
    assert finer.record["energy_transport"][0]["reference_flight_time_s"] == pytest.approx(
        float(np.mean(next(p for p in rays.plane_arrivals if p.z_mm == gun.exit_plane_z_mm).time_s)), rel=.003)










def test_no_cached_or_caller_exit_can_bypass_tip_configuration_or_cancellation(gun):
    with pytest.raises(InterruptedError):
        build_tip_gun_checkpoint(gun, cancelled=lambda: True)
    gun.source_representation = "effective_gaussian_schell"
    with pytest.raises(ValueError, match="Custom exit"):
        build_tip_gun_checkpoint(gun)
    gun.source_representation = "classical_particles"
    gun.emitter.coherence = None
    with pytest.raises(ValueError, match="explicitly"):
        build_tip_gun_checkpoint(gun)


def test_imported_wien_provider_cannot_be_silently_omitted(gun):
    # Admit a local tip solely to reach the unsupported-field boundary.
    gun.emitter.emission_energy_ev = 4.
    gun.install_monochromator()
    gun.monochromator._field_provider_override = object()
    with pytest.raises(ValueError, match="Imported Wien"):
        build_tip_gun_checkpoint(gun, source_numerics=SOURCE, numerics=NUMERICS)


def test_actual_rotated_gun_stigmator_and_deflector_field_coefficients(gun):
    gun.stigmator.gradient_t_per_m = .3
    gun.stigmator.rotation_deg = 30.
    z = np.array((gun.stigmator.optical_reference_from_tip_mm,))
    *_, magnetic_hessian = _field_coefficients(gun, z)
    angle = np.deg2rad(60.)
    expected = .3*np.array(((np.cos(angle), np.sin(angle)), (np.sin(angle), -np.cos(angle))))
    np.testing.assert_allclose(magnetic_hessian[0], expected, rtol=1e-12, atol=1e-14)
    gun.deflector.upper_field_x_mt = .02
    gun.deflector.upper_field_y_mt = -.03
    z = np.array((gun.deflector.upper_center_from_tip_mm+gun.deflector.field_center_offset_mm,))
    _, _, _, force, _ = _field_coefficients(gun, z)
    np.testing.assert_allclose(force[0], (-.03e-3, -.02e-3), rtol=1e-12, atol=1e-14)


def test_wave_memory_budget_is_checked_before_allocation(gun):
    # Admit a local tip; the memory guard must reject before any wave allocation.
    gun.emitter.emission_energy_ev = 4.
    with pytest.raises(ValueError, match="maximum_checkpoint_bytes"):
        build_tip_gun_checkpoint(gun, source_numerics=replace(SOURCE, grid_pixels=4096),
                                 numerics=replace(NUMERICS, maximum_checkpoint_bytes=1024))


@pytest.mark.parametrize("change", ["none", "voltage", "focus", "wien_slit"])
def test_default_tip_domain_cannot_be_repaired_by_downstream_configuration(gun, monkeypatch, change):
    from temsim.physics import tip_gun_wave
    from temsim.optics.electron_gun.tip_source_domain import TipSourceDomainError
    if change == "voltage":
        gun.accelerator.high_tension_kv = 200.
    elif change == "focus":
        gun.electrostatic_lens.voltage_kv *= 1.01
    elif change == "wien_slit":
        gun.install_monochromator()
        gun.c1_aperture.select_slit_mode(True)
        gun.monochromator.slit.gap_um = 100.
    before = gun.to_dict()
    cached = tuple(tip_gun_wave._CACHE)
    monkeypatch.setattr(tip_gun_wave, "_axial_grid", lambda *a, **kw: pytest.fail("Rejected tip must not start gun transport"))
    with pytest.raises(TipSourceDomainError, match="paraxial domain"):
        build_tip_gun_checkpoint(gun, source_numerics=SOURCE, numerics=NUMERICS)
    assert gun.to_dict() == before
    assert tuple(tip_gun_wave._CACHE) == cached
