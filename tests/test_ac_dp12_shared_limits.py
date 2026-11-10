"""Shared AC/DP12 limits and rollback with state fixtures and drift responses.

These tests emit no finite source and execute no particle or wave propagation.
"""
from copy import deepcopy

import numpy as np
import pytest

from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.optics.column import default_state
from temsim.optics.probe_service import set_probe_output_percent
from temsim.physics import scan_geometry, scan_preparation
from temsim.physics.instrument_magnetic import column_dipole_fields
from temsim.physics.scan_calibration import capture_record, restore_held
from test_unified_scan_calibration import _drift_response


@pytest.fixture(scope="module")
def initial():
    return encode_instrument(default_state())


@pytest.fixture
def state(initial):
    state = decode_instrument(initial)
    ac = state.ac_deflector
    ac.wobble_enabled = False
    ac.scan_enabled = True
    ac.set_pure_shift_coupling(-np.eye(2))
    ac.set_scan_command_matrix_mrad(np.eye(2)*.1)
    state.descan_deflector.scan_enabled = False
    set_probe_output_percent(state, "DP12x", percent=87.)
    # The default upper station has a 99.18 mrad DP12 bias. Its initial
    # 0.05 mrad scan swing fits, but a 1 mrad swing would exceed 100 mrad.
    assert len([c for c in column_dipole_fields(state)
                if c.key.startswith("ac_deflector:")]) == 2
    return state


def _hold(state):
    capture_record(state, descan_calibrated=False, ac_angle_target="canonical")
    state.ac_deflector.calibration_mode = "held"


@pytest.mark.parametrize("restore_descan", [False, True])
@pytest.mark.parametrize("axis, sign", [("x", 1.), ("y", -1.)])
def test_held_fov_rejects_combined_peak_and_restores_drives(state, restore_descan, axis, sign):
    set_probe_output_percent(state, "DP12x", percent=0.)
    set_probe_output_percent(state, "DP12"+axis, percent=sign*87.)
    _hold(state)
    state.ac_deflector.scan_pixel_size_nm *= 20.
    before = encode_instrument(state)
    host = state.descan_deflector._physical_host
    with pytest.raises(ValueError, match="coil.*limit"):
        restore_held(state, restore_descan=restore_descan)
    assert encode_instrument(state) == before
    assert state.descan_deflector._physical_host is host
    column_dipole_fields(state)  # Previous physical drives remain executable.


@pytest.mark.parametrize("mode", ["ideal", "unified"])
def test_public_held_calibration_rejects_before_response_measurement(state, monkeypatch, mode):
    state.simulation_mode = mode
    _hold(state)
    state.ac_deflector.scan_pixel_size_nm *= 20.
    before = encode_instrument(state)

    def forbidden(*args, **kwargs):
        pytest.fail("Over-limit held drives must be rejected before measuring responses")

    monkeypatch.setattr(scan_geometry, "paired_kick_response", forbidden)
    monkeypatch.setattr("temsim.physics.unified_scan_calibration.measure_scan_responses", forbidden)
    with pytest.raises(ValueError, match="coil.*limit"):
        scan_geometry.calibrate_scan_system(state)
    assert encode_instrument(state) == before


def test_held_fov_accepts_valid_shared_drive_without_duplicate_fields(state):
    _hold(state)
    state.ac_deflector.scan_pixel_size_nm *= 16.
    command = restore_held(state)
    np.testing.assert_allclose(command, np.eye(2)*1.6)
    coils = [c for c in column_dipole_fields(state) if c.key.startswith("ac_deflector:")]
    assert len(coils) == 2
    assert all(c.drive_keys == ("ac_deflector", "probe_dp12_scan_deflector") for c in coils)
    assert state.dp12_scan_deflector.alignment_x_mrad == 87.


@pytest.mark.parametrize("inactive", ["channel", "corrector", "host"])
def test_inactive_dp12_or_host_does_not_consume_scan_headroom(state, inactive):
    if inactive == "channel":
        state.dp12_scan_deflector.enabled = False
    elif inactive == "corrector":
        state.probe_corrector_installed = False
    else:
        state.ac_deflector.enabled = False
    _hold(state)
    state.ac_deflector.scan_pixel_size_nm *= 20.
    np.testing.assert_allclose(restore_held(state), np.eye(2)*2.)


@pytest.mark.parametrize("mode", ["ideal", "unified"])
@pytest.mark.parametrize("force", [False, True])
def test_automatic_calibration_rolls_back_combined_overload(state, monkeypatch, mode, force):
    state.simulation_mode = mode
    ac = state.ac_deflector
    # In this drift oracle, a 50 um FOV requires a 2 mrad logical command,
    # or a 1 mrad swing at either physical station.
    ac.scan_pixel_size_nm = 50000. / ac.scan_pixels_x
    monkeypatch.setattr("temsim.physics.unified_scan_calibration.measure_scan_responses", _drift_response)

    def drift(_state, component, z, **kwargs):
        upper = np.vstack((np.eye(2)*(z-component.upper_z_mm)*1e-3, np.eye(2)))
        lower = np.vstack((np.eye(2)*(z-component.lower_z_mm)*1e-3, np.eye(2)))
        return upper, lower, np.zeros(2)

    monkeypatch.setattr(scan_geometry, "_coil_phase_responses", drift)
    monkeypatch.setattr(scan_geometry, "canonical_source_basis", lambda *args, **kwargs: np.eye(4))
    before = encode_instrument(state)
    with pytest.raises(ValueError, match="coil.*limit"):
        scan_geometry.calibrate_scan_system(state, force=force, hold=force)
    assert encode_instrument(state) == before

    # Reducing the alignment leaves room for exactly the same scan solve.
    set_probe_output_percent(state, "DP12x", percent=85.)
    command, _, _ = scan_geometry.calibrate_scan_system(state, force=force, hold=force)
    np.testing.assert_allclose(command, np.eye(2)*2., atol=1e-12)
    column_dipole_fields(state)


def test_standalone_pivot_calibration_checks_lower_station_and_rolls_back(state, monkeypatch):
    monkeypatch.setattr(scan_geometry, "canonical_source_basis", lambda *args, **kwargs: np.eye(4))
    responses = (np.vstack((np.zeros((2, 2)), np.eye(2)*1800.)),
                 np.vstack((np.zeros((2, 2)), np.eye(2))), np.zeros(2))
    before = encode_instrument(state)
    with pytest.raises(ValueError, match="coil.*limit"):
        scan_geometry.calibrate_ac_pure_shift(state, _coil_responses=responses)
    assert encode_instrument(state) == before


def test_complete_calibration_checks_final_ratio_and_amplitude_together(state, monkeypatch):
    # The new ratio with the old scan amplitude temporarily needs 102.18 mrad
    # at the lower coil. The final solved amplitude makes both stations legal.
    responses = (np.vstack((np.eye(2)*2., np.eye(2)*1800.)),
                 np.vstack((np.eye(2)*.001, np.eye(2))), np.zeros(2))
    monkeypatch.setattr(scan_geometry, "_coil_phase_responses", lambda *args, **kwargs: responses)
    monkeypatch.setattr(scan_geometry, "canonical_source_basis", lambda *args, **kwargs: np.eye(4))
    command, _ = scan_geometry.calibrate_ac_scan_scale(state)
    ac = state.ac_deflector
    np.testing.assert_allclose(ac.pure_shift_lower_ratio_matrix, -np.eye(2)*1800.)
    expected = np.diag((ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm))*5e-6
    np.testing.assert_allclose(command, expected, rtol=1e-12, atol=0.)
    assert state.dp12_scan_deflector.alignment_x_mrad == 87.
    column_dipole_fields(state)


@pytest.mark.parametrize("route", ["copy", "global_cache", "session_cache"])
def test_reused_calibration_cannot_install_combined_overload(state, monkeypatch, route):
    candidate = decode_instrument(encode_instrument(state))
    candidate.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*2.)
    candidate.ac_deflector._scan_execution_receipt = {"scope": "Rejected test fixture"}
    candidate.descan_deflector.scan_pixels_x += 1
    before = encode_instrument(state)
    host = state.descan_deflector._physical_host
    monkeypatch.setattr(scan_preparation, "_PREPARED", deepcopy(scan_preparation._PREPARED))
    with pytest.raises(ValueError, match="coil.*limit"):
        if route == "copy":
            scan_preparation.copy_resolved_scan_drives(candidate, state)
        else:
            # A stale cached output must face the same guard as a new solve.
            key = (scan_preparation._preparation_identity("shared-limit-fixture"),
                   frozenset({state.ac_deflector.key}))
            values = scan_preparation._resolved_values(candidate)
            cache = {key: values} if route == "session_cache" else None
            if cache is None:
                scan_preparation._PREPARED[key] = values
            scan_preparation.prepare_scan_drives(state, "shared-limit-fixture", session_cache=cache)
    assert encode_instrument(state) == before
    assert state.descan_deflector._physical_host is host
