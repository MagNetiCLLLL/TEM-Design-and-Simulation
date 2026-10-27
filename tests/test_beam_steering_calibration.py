"""Finite-coil steering checks; local fixtures do not certify a microscope."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import encode_instrument, decode_instrument
from temsim.optics.column import default_state
from temsim.physics import beam_steering_calibration as steering


@pytest.fixture(scope="module")
def native():
    return encode_instrument(default_state())


@pytest.fixture
def drift(native, monkeypatch):
    s = decode_instrument(native)
    for c in (*s.lenses, *s.stigmators, *s.corrector_elements, *s.deflectors):
        c.enabled = False
    s.beam_deflector.enabled = True
    s.acceleration_enabled = False
    s.acceleration_backend = "CPU"
    s.sample.inserted = False
    monkeypatch.setattr(type(s), "sync_objective", lambda _state: None)
    from temsim.physics import core
    monkeypatch.setattr(core, "fields", lambda z, *_args, **_kwargs:
        tuple(np.zeros_like(np.asarray(z), dtype=float) for _ in range(3)))
    # A stated constant electrostatic potential is the analytic drift fixture;
    # finite beam-coil fields and complete node-wise clipping remain production.
    from temsim.physics import instrument_electric
    monkeypatch.setattr(instrument_electric, "capture_instrument_electric_field", lambda _state:
        SimpleNamespace(gun_snapshot=SimpleNamespace(exit_plane_z_mm=0.),
            base_field=SimpleNamespace(z=()), numerical_identity="constant-potential steering fixture",
            potential_rise_v_at_global_positions=lambda points: np.zeros(len(points)),
            is_constant_on_interval=lambda *_args: True))
    return s


def boundary(state, **kwargs):
    return steering.reference_from_values(state,
        z_mm=state.beam_deflector.upper_surface_z_mm,
        phase_space=(2e-6, -1e-6, 3e-4, -2e-4),
        kinetic_energy_ev=300000., flight_time_s=1e-8,
        provenance="Explicit constant-potential local reference fixture", **kwargs)


@pytest.mark.parametrize("mode", ["ideal", "ideal_thin"])
def test_selected_pair_calibration_matches_exact_forced_drift_and_does_not_edit_state(drift, mode):
    s = drift
    s.simulation_mode = mode
    before = json_digest(encode_instrument(s))
    ref = boundary(s)
    target = s.beam_deflector.lower_surface_z_mm+10.
    solver = steering.BeamSteeringCalibrator(s, ref, target_z_mm=target, maximum_step_mm=.5)
    result = solver.calibrate()
    upper = (target-s.beam_deflector.upper_z_mm)*1e-3
    lower = (target-s.beam_deflector.lower_z_mm)*1e-3
    expected = np.array([[upper, 0., lower, 0.], [0., upper, 0., lower],
                         [1., 0., 1., 0.], [0., 1., 0., 1.]])
    np.testing.assert_allclose(result.jacobian, expected, rtol=2e-11, atol=1e-12)
    assert json_digest(encode_instrument(s)) == before
    assert result.reference.phase_space == ref.phase_space
    assert result.condition_number < 20
    assert result.scope == (steering.THIN_SCOPE if mode == "ideal_thin" else steering.SCOPE)


@pytest.mark.parametrize("kind,command", [("tilt", (1e-3, -2e-3)), ("shift", (2e-6, -3e-6))])
def test_finite_command_endpoint_and_transaction_change_only_four_drives(drift, kind, command):
    s = drift
    ref = boundary(s)
    solver = steering.BeamSteeringCalibrator(s, ref,
        target_z_mm=s.beam_deflector.lower_surface_z_mm+10., maximum_step_mm=.5)
    result = solver.solve(kind=kind, x=command[0], y=command[1])
    original = encode_instrument(s)
    steering.apply_beam_steering_solution(s, result)
    restored = decode_instrument(original)
    for field, value in zip(steering.DRIVE_FIELDS, result.drives_mrad):
        setattr(restored.beam_deflector, field, value)
    assert json_digest(encode_instrument(s)) == json_digest(encode_instrument(restored))
    baseline, achieved = np.array(result.baseline), np.array(result.achieved)
    if kind == "tilt":
        np.testing.assert_allclose(achieved[:2], baseline[:2], atol=1e-11)
        np.testing.assert_allclose(np.arctan(achieved[2:])-np.arctan(baseline[2:]), command, atol=1e-9)
    else:
        np.testing.assert_allclose(achieved[:2]-baseline[:2], command, atol=1e-11)
        np.testing.assert_allclose(achieved[2:], baseline[2:], atol=1e-9)
    with pytest.raises(ValueError, match="stale"):
        steering.apply_beam_steering_solution(s, result)


def test_stale_reference_cancel_range_and_invalid_receipt_do_not_change_state(drift):
    s = drift
    ref = boundary(s)
    s.beam_deflector.upper_x_mrad = .1
    with pytest.raises(ValueError, match="stale"):
        steering.BeamSteeringCalibrator(s, ref)
    s.beam_deflector.upper_x_mrad = 0.
    solver = steering.BeamSteeringCalibrator(s, ref,
        target_z_mm=s.beam_deflector.lower_surface_z_mm+10., maximum_step_mm=.5)
    before = json_digest(encode_instrument(s))
    with pytest.raises(InterruptedError):
        solver.calibrate(cancelled=lambda: True)
    with pytest.raises(ValueError, match="No admissible"):
        solver.solve(kind="shift", x=10., y=0.)
    result = solver.solve(kind="shift", x=1e-6, y=0.)
    with pytest.raises(ValueError, match="four finite"):
        replace(result, drives_mrad=(0.,))
    with pytest.raises(ValueError, match="tolerances"):
        replace(result, position_tolerance_m=float("nan"))
    with pytest.raises(ValueError, match="residual"):
        replace(result, achieved=tuple(np.asarray(result.achieved)+1.))
    assert json_digest(encode_instrument(s)) == before


def test_blocked_reference_fails_closed_and_old_cancel_does_not_poison_new_solve(drift):
    s = drift
    target = s.beam_deflector.lower_surface_z_mm+10.
    stopped = [False]
    solver = steering.BeamSteeringCalibrator(s, boundary(s), target_z_mm=target,
        maximum_step_mm=.5, cancelled=lambda: stopped[0])
    solver.calibrate()
    stopped[0] = True
    solver.solve(kind="shift", x=1e-6, y=0., cancelled=lambda: False)
    aperture = s.condenser_aperture_3
    aperture.enabled = True
    aperture.installed = True
    object.__setattr__(aperture, "z_mm", target)
    aperture.diameter_mm = .0001
    with pytest.raises(ValueError, match="blocked"):
        steering.BeamSteeringCalibrator(s, boundary(s), target_z_mm=target,
            maximum_step_mm=.5).calibrate()


def test_singular_pair_is_rejected_with_no_endpoint_claim(drift, monkeypatch):
    solver = steering.BeamSteeringCalibrator(drift, boundary(drift),
        target_z_mm=drift.beam_deflector.lower_surface_z_mm+10.)
    monkeypatch.setattr(solver, "_evaluate", lambda drives, _cancel:
        (np.ones(4)*sum(drives), "test-only rank-one map"))
    with pytest.raises(ValueError, match="singular|ill-conditioned"):
        solver.calibrate()


def test_retained_reference_uses_actual_survivor_energy_clock_and_not_nominal_delta(drift):
    s = drift
    entrance = s.beam_deflector.upper_surface_z_mm
    target = s.sample.z_mm
    bundle = SimpleNamespace(alive=np.ones(3, bool), weight=np.ones(3), ray_id=np.array([30, 31, 32]),
        x_m=np.zeros(3), y_m=np.zeros(3), tx_rad=np.zeros(3), ty_rad=np.zeros(3),
        energy_offset_ev=np.array([10., 20., 30.]), flight_time_s=np.full(3, 1e-8))
    checkpoint = SimpleNamespace(z_mm=np.array([entrance]), x_m=np.array([[0., 2e-6, 1e-6]]),
        y_m=np.zeros((1, 3)), tx_rad=np.array([[0., .001, .002]]), ty_rad=np.zeros((1, 3)),
        kinetic_energy_ev=np.array([[299111., 299222., 299333.]]), flight_time_s=np.array([[3e-8, 4e-8, 5e-8]]))
    result = SimpleNamespace(state_snapshot=decode_instrument(encode_instrument(s)),
        calculation_manifest=SimpleNamespace(solver=SimpleNamespace(source_digest=steering.solver_source_identity())),
        simulation=SimpleNamespace(gun_trace=SimpleNamespace(exit_bundle=bundle), incident_checkpoints=checkpoint,
            incident=SimpleNamespace(blocked_z=np.array([target-1., np.nan, np.nan]))))
    before = json_digest(encode_instrument(s))
    ref = steering.capture_beam_steering_reference(s, result=result)
    assert ref.ray_id == 32  # The on-axis first ray is stopped before the target.
    assert ref.kinetic_energy_ev == 299333.
    assert ref.flight_time_s == 5e-8
    assert ref.phase_space == (1e-6, 0., .002, 0.)
    assert not ref.include_initial_plane_kicks
    with pytest.raises(ValueError, match="No identified"):
        steering.capture_beam_steering_reference(s, result=result, ray_id=30)
    assert json_digest(encode_instrument(s)) == before


def test_local_jacobian_and_iterated_solution_retain_nonlinear_hp_and_cs(drift, monkeypatch):
    from temsim.physics import core
    target = drift.beam_deflector.lower_surface_z_mm+10.
    # Explicit, bounded nonlinear field fixture, evaluated by the same native
    # integrator. The nonzero reference is essential: its tangent is not the
    # nominal on-axis first-order matrix.
    strength = 3e7
    monkeypatch.setattr(core, "hexapole_field_components", lambda z, *_args, **_kwargs:
        (np.full_like(np.asarray(z), strength, dtype=float), np.zeros_like(np.asarray(z), dtype=float)))
    monkeypatch.setattr(core, "spherical_aberration_kick_m3", lambda z, _state:
        np.where(np.asarray(z) == target, 2e12, 0.))
    solver = steering.BeamSteeringCalibrator(drift, boundary(drift), target_z_mm=target,
        maximum_step_mm=.25, perturbation_rad=1e-7)
    calibrated = solver.calibrate()
    expected_x = (target-drift.beam_deflector.upper_z_mm)*1e-3
    assert not np.isclose(calibrated.jacobian[0][0], expected_x, rtol=1e-3)
    solved = solver.solve(kind="tilt", x=5e-4, y=-2e-4)
    assert solved.iterations >= 2
    np.testing.assert_allclose(np.asarray(solved.achieved)[:2], np.asarray(solved.baseline)[:2], atol=1e-11)
    np.testing.assert_allclose(np.arctan(solved.achieved[2:])-np.arctan(solved.baseline[2:]), (5e-4, -2e-4), atol=1e-9)


def test_residual_gas_transport_is_rejected_without_changing_settings(drift):
    drift.vacuum_map.enabled = True
    ref = boundary(drift)
    before = json_digest(encode_instrument(drift))
    with pytest.raises(ValueError, match="residual-gas scattering"):
        steering.capture_beam_steering_reference(drift)
    with pytest.raises(ValueError, match="residual-gas scattering"):
        steering.BeamSteeringCalibrator(drift, ref)
    assert json_digest(encode_instrument(drift)) == before
