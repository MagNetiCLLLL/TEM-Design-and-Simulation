"""Physical channel calibration is local model evidence, never OEM acceptance."""
from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics import corrector_channels as channels
from temsim.optics.hexapole import HexapoleComponent
from temsim.optics.model import Stigmator
from temsim.optics.single_plane_deflector import SinglePlaneDeflectorComponent
from temsim.optics.system_residual_aberrations import (
    capture_system_residual_reference, measure_system_residual,
)


def _state():
    hp = HexapoleComponent("Physical auxiliary", "image_hpol_hexapole", 2., 0., 1e9,
        .2, True, "black", 2., 1., 5., 3., 2., corrector="image")
    return SimpleNamespace(lenses=[], corrector_elements=[hp], deflectors=[],
        stigmators=[Stigmator("Physical stigmator", "objective_stigmator", 2.,
                              length_mm=.2, max_strength_m2=300.)],
        apertures=[], recording_planes=[], sample=SimpleNamespace(z_mm=0.),
        objective_image_plane_z_mm=4., beam_voltage_kv=300., step_mm=.05,
        history_step_mm=.05, acceleration_enabled=False, acceleration_backend="CPU",
        equivalent_image_lenses_enabled=False, lens_intrinsic_aberrations={},
        probe_aberrations={}, image_aberrations={})


def _fixture_digest(state):
    # This deliberately simple non-State fixture is not accepted by the
    # production graph encoder. Include every public physical fixture input.
    def encode(obj):
        if hasattr(obj, "__dict__"):
            return {key: value for key, value in vars(obj).items() if not key.startswith("_")}
        raise TypeError(type(obj).__name__)
    return sha256(json.dumps(state, default=encode, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def physical(monkeypatch):
    monkeypatch.setattr(channels, "_state_digest", _fixture_digest)
    monkeypatch.setattr(channels, "_clone_working_point", deepcopy)
    return _state()


def _calibrate(state, channel="a2_fine"):
    return channels.calibrate_channel(state, "image", channel,
                                      semiangle_mrad=5., maximum_step_mm=.025)


def _candidate(state, calibration, scale=.1):
    delta = calibration.jacobian @ (calibration.steps*np.linspace(scale, -scale/2, len(calibration.steps)))
    return channels.prepare_channel_change(state, calibration, delta)


def _coarse_state():
    state = _state()
    for index, (position, strength) in enumerate(((1.5, 1e8), (2.5, 7e7)), 1):
        state.corrector_elements.append(HexapoleComponent(
            f"Principal {index}", f"image_hp{index}_hexapole", position, strength,
            1e9, .2, True, "black", position, 1., 5., 3., position,
            corrector="image", orientation_rad=.1*index))
    for index, position in enumerate((.7, 2.2), 1):
        state.corrector_elements.append(SinglePlaneDeflectorComponent(
            f"DPH {index}", f"image_dph{index}_deflector", position, 0., 0.,
            .1, True, "black", position, .5, 5., 3., position, 10., corrector="image"))
    return state


@pytest.mark.parametrize("channel", ["a1_coarse", "a2_coarse"])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_coarse_two_axis_response_from_real_dph_and_principal_hexapoles(physical, channel, sign):
    state = _coarse_state()
    original = _fixture_digest(state)
    calibration = _calibrate(state, channel)
    assert np.linalg.matrix_rank(calibration.jacobian) == 2
    assert calibration.diagnostics["condition_number"] < 100
    assert _fixture_digest(state) == original
    candidate = _candidate(state, calibration, sign*.15)
    np.testing.assert_allclose(candidate.achieved_delta_xy_mm, candidate.requested_delta_xy_mm,
                               rtol=.01, atol=1e-11)
    if channel == "a1_coarse":
        assert all("dph" in c["key"] for c in candidate.control_changes)
        assert np.linalg.norm(candidate.diagnostics["direction_after_rad"]) > 0.
        assert np.linalg.norm(candidate.diagnostics["centroid_after_m"]) > 0.
    else:
        assert {c["key"] for c in candidate.control_changes} == {
            "image_hp1_hexapole", "image_hp2_hexapole"}
    channels.apply_channel_change(state, candidate)
    assert _fixture_digest(state) == candidate.resulting_state_digest


def test_frozen_reference_preserves_real_stigmator_a1_response():
    state = _state()
    reference = capture_system_residual_reference(state, "image", semiangle_mrad=5., maximum_step_mm=.025)
    state.stigmators[0].strength_x_percent = .2
    state.stigmators[0].strength_y_percent = -.1
    fixed = measure_system_residual(state, "image", semiangle_mrad=5.,
                                   maximum_step_mm=.025, reference=reference)
    recaptured = measure_system_residual(state, "image", semiangle_mrad=5., maximum_step_mm=.025)
    assert fixed.coefficients.a1_mm > 1e-7
    assert recaptured.coefficients.a1_mm < fixed.coefficients.a1_mm*1e-5
    assert fixed.diagnostics["c1_retained"]
    assert fixed.diagnostics["reference_centroid_slopes_rad"] == [0., 0.]
    for array in (reference.inputs, reference.linear, reference.matrix, reference.training):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(ValueError, match="reference does not match"):
        measure_system_residual(state, "image", semiangle_mrad=6., reference=reference)


@pytest.mark.parametrize("channel", ["a1_fine", "a2_fine"])
def test_real_field_two_axis_response_candidate_and_atomic_apply(physical, channel):
    state = physical
    original = _fixture_digest(state)
    calibration = _calibrate(state, channel)
    assert _fixture_digest(state) == original
    assert calibration.diagnostics["physical_measurement_count"] == 9
    assert calibration.diagnostics["condition_number"] < 2
    assert calibration.diagnostics["signed_half_step_relative_error"] < 1e-4
    assert not calibration.diagnostics["oem_calibration"]
    assert np.linalg.matrix_rank(calibration.jacobian) == 2
    candidate = _candidate(state, calibration)
    assert _fixture_digest(state) == original
    assert candidate.diagnostics["independent_candidate_measured"]
    np.testing.assert_allclose(candidate.achieved_delta_xy_mm, candidate.requested_delta_xy_mm,
                               rtol=2e-4, atol=1e-12)
    assert candidate.control_changes
    assert all(c["field"] in {"strength_m3", "orientation_rad", "strength_x_percent",
                              "strength_y_percent"} for c in candidate.control_changes)
    channels.apply_channel_change(state, candidate)
    assert _fixture_digest(state) == candidate.resulting_state_digest != original
    assert state.image_aberrations == {} and state.probe_aberrations == {}
    with pytest.raises(ValueError, match="stale"):
        channels.apply_channel_change(state, candidate)
    with pytest.raises(ValueError, match="stale"):
        channels.prepare_channel_change(state, calibration, [1e-6, 0.])


def test_zero_auxiliary_strength_has_two_finite_cartesian_derivatives(physical):
    calibration = _calibrate(physical)
    assert physical.corrector_elements[0].strength_m3 == 0.
    assert np.linalg.svd(calibration.jacobian, compute_uv=False)[-1] > 1e-12
    with pytest.raises(ValueError):
        calibration.jacobian[0, 0] = 0
    with pytest.raises(ValueError):
        calibration.steps.setflags(write=True)


@pytest.mark.parametrize("strength", [-1e6, 1e6])
@pytest.mark.parametrize("angle", [-2.00126792, .2, 5.2])
def test_small_cartesian_change_keeps_signed_branch_and_nearest_orientation(physical, strength, angle):
    item = physical.corrector_elements[0]
    item.strength_m3, item.orientation_rad = strength, angle
    specs = channels.channel_control_specs(physical, "image", "a2_fine")
    desired = channels._control_values(physical, specs)+[13., -7.]
    channels._set_controls(physical, specs, desired)
    assert np.sign(item.strength_m3) == np.sign(strength)
    assert abs(item.orientation_rad-angle) < 1e-4
    np.testing.assert_allclose(channels._control_values(physical, specs), desired, rtol=1e-13)


def test_unchanged_cartesian_controls_preserve_exact_signed_noncanonical_representation(physical):
    item = physical.corrector_elements[0]
    item.strength_m3, item.orientation_rad = -345678.123, 5.2
    specs = channels.channel_control_specs(physical, "image", "a2_fine")
    before = _fixture_digest(physical)
    channels._set_controls(physical, specs, channels._control_values(physical, specs))
    assert _fixture_digest(physical) == before


def test_zero_strength_selects_nearest_signed_orientation_without_losing_skew(physical):
    item = physical.corrector_elements[0]
    specs = channels.channel_control_specs(physical, "image", "a2_fine")
    channels._set_controls(physical, specs, [-100., 1.])
    assert item.strength_m3 < 0.
    assert abs(item.orientation_rad) < .01
    np.testing.assert_allclose(channels._control_values(physical, specs), [-100., 1.], atol=1e-12)


@pytest.mark.parametrize("field, value", [("effective_length_mm", .25), ("z_mm", 2.1),
                                         ("orientation_rad", .1), ("strength_m3", 5.)])
def test_changed_physical_working_point_invalidates_calibration(physical, field, value):
    calibration = _calibrate(physical)
    setattr(physical.corrector_elements[0], field, value)
    with pytest.raises(ValueError, match="stale"):
        channels.prepare_channel_change(physical, calibration, [1e-7, 1e-7])


def test_cancellation_before_or_during_signed_probes_never_mutates_live_state(physical):
    original = _fixture_digest(physical)
    with pytest.raises(RuntimeError, match="cancelled"):
        channels.calibrate_channel(physical, "image", "a2_fine", cancelled=lambda: True)
    checks = 0
    def cancelled():
        nonlocal checks
        checks += 1
        return checks > 20
    with pytest.raises(RuntimeError, match="cancelled"):
        channels.calibrate_channel(physical, "image", "a2_fine", cancelled=cancelled)
    assert _fixture_digest(physical) == original


def test_application_cancellation_rolls_back_already_written_drives(physical):
    calibration = _calibrate(physical)
    candidate = _candidate(physical, calibration)
    original = _fixture_digest(physical)
    def cancelled():
        return _fixture_digest(physical) != original
    with pytest.raises(RuntimeError, match="cancelled"):
        channels.apply_channel_change(physical, candidate, cancelled=cancelled)
    assert _fixture_digest(physical) == original


def test_candidate_rejects_wrong_identity_and_out_of_channel_changes(physical):
    calibration = _calibrate(physical)
    candidate = _candidate(physical, calibration)
    original = _fixture_digest(physical)
    forged = replace(candidate, resulting_state_digest="not-the-measured-hardware")
    with pytest.raises(ValueError, match="identity verification"):
        channels.apply_channel_change(physical, forged)
    forged = replace(candidate, control_changes=({"key": "image_hpol_hexapole",
        "field": "effective_length_mm", "before": .2, "after": .3},))
    with pytest.raises(ValueError, match="outside"):
        channels.apply_channel_change(physical, forged)
    assert _fixture_digest(physical) == original


@pytest.mark.parametrize("delta", [[0., 0.], [float("nan"), 0.], [1.], [1e9, 1e9]])
def test_invalid_or_excessive_requested_increment_is_rejected_without_mutation(physical, delta):
    calibration = _calibrate(physical)
    original = _fixture_digest(physical)
    with pytest.raises(ValueError):
        channels.prepare_channel_change(physical, calibration, delta)
    assert _fixture_digest(physical) == original


def test_singular_physical_axis_and_hardware_at_limit_fail_closed(physical):
    # Collinear real stigmator channels cannot independently control A1 x/y.
    physical.stigmators[0].channel_y_angle_deg = 0.
    with pytest.raises(ValueError, match="independent|singular|ill-conditioned"):
        _calibrate(physical, "a1_fine")
    physical.corrector_elements[0].strength_m3 = physical.corrector_elements[0].maximum_strength_m3
    with pytest.raises(ValueError, match="range"):
        _calibrate(physical)


def test_candidate_validation_uses_new_measured_response_not_only_jacobian(physical, monkeypatch):
    calibration = _calibrate(physical)
    original = _fixture_digest(physical)
    monkeypatch.setattr(channels, "_measure", lambda *args: calibration.baseline)
    with pytest.raises(ValueError, match="Independent physical candidate"):
        _candidate(physical, calibration, .4)
    assert _fixture_digest(physical) == original


@pytest.mark.parametrize("design", ["cetcor", "bcor"])
def test_real_assemblies_have_distinct_coarse_fine_physical_channels_and_full_identity(design):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.instrument_configuration import InstrumentUnits
    from temsim.optics.column import default_state
    state, catalog = default_state(), AssemblyCatalog()
    catalog.apply(state, InstrumentUnits(probe_corrector=True, image_corrector=True,
        image_corrector_design=design, energy_filter=False).selection(catalog))
    for system in ("probe", "image"):
        groups = {name: channels.channel_control_specs(state, system, name) for name in channels.CHANNELS}
        for order in ("a1", "a2"):
            coarse = {spec.key for spec in groups[order+"_coarse"]}
            fine = {spec.key for spec in groups[order+"_fine"]}
            assert coarse and fine and coarse.isdisjoint(fine)
        if system == "image" and design == "bcor":
            assert "image_hp3_hexapole" in {s.key for s in groups["a2_coarse"]}
    digest = channels._state_digest(state)
    hpol = next(x for x in state.corrector_elements if x.key == "image_hpol_hexapole")
    hpol.effective_length_mm *= 1.01
    assert channels._state_digest(state) != digest
