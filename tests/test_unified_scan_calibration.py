"""Shared-field calibration controls and independent drift response limits."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.unified_scan_calibration import calibrate_unified_scan


@pytest.fixture(autouse=True)
def isolated_response_storage(monkeypatch, tmp_path):
    from temsim.physics.unified_scan_cache import clear_scan_response_cache
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_CACHE_DIR", str(tmp_path/"scan-cache"))
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "1")
    clear_scan_response_cache()
    yield
    clear_scan_response_cache()


def _state():
    from temsim.operating_modes import apply_condenser_mode
    state = default_state()
    apply_condenser_mode(state, "nano_probe")
    state.simulation_mode = "unified"
    return state


def _drift_response(state, observations, **kwargs):
    """Exact first derivative of uniform finite dipoles in a zero-field drift."""
    from temsim.physics.instrument_magnetic import column_dipole_fields
    targets = tuple(sorted(set(observations)))
    phase, canonical = {}, {}
    for coil in column_dipole_fields(state):
        if not coil.key.startswith(("ac_deflector:", state.descan_deflector._physical_host_key+":")):
            continue
        response = np.zeros((len(targets), 4, 2))
        for i, z in enumerate(targets):
            lo, hi = coil.field_support_mm
            traversed = np.clip(z-lo, 0., hi-lo)/(hi-lo)
            # Integrate d(theta)/dz = command/length exactly through the
            # finite interval, followed by the remaining free drift.
            response[i, :2] = np.eye(2)*(max(z-hi, 0.)*traversed
                +.5*(min(max(z, lo), hi)-lo)*traversed)*1e-3
            response[i, 2:] = np.eye(2)*traversed
        phase[coil.key], canonical[coil.key] = response, response[:, 2:]
    return dict(z_mm=targets, phase=phase, canonical=canonical,
                reference=np.zeros((len(targets), 4)), receipt={"scope": "analytical uniform-dipole drift oracle"})


def test_automatic_drive_uses_finite_coils_and_preserves_physical_inputs(monkeypatch):
    import temsim.physics.unified_scan_calibration as calibration
    state = _state()
    physical = [(lens.key, lens.z_mm, lens.percent) for lens in state.lenses]
    emitter = deepcopy(vars(state.electron_gun.emitter))
    host = state.descan_deflector._physical_host
    monkeypatch.setattr(calibration, "measure_scan_responses", _drift_response)
    command, residual, result = calibrate_unified_scan(state)
    np.testing.assert_allclose(state.ac_deflector.pure_shift_lower_ratio_matrix, -np.eye(2), atol=1e-14)
    separation = (state.ac_deflector.lower_z_mm-state.ac_deflector.upper_z_mm)*1e-3
    desired = np.diag((state.ac_deflector.scan_field_of_view_x_nm,
                       state.ac_deflector.scan_field_of_view_y_nm))*.5e-9
    np.testing.assert_allclose(separation*state.ac_deflector.upper_coil_gain*command*1e-3, desired, rtol=1e-13, atol=0.)
    assert residual < 1e-13 and result.response_match_residual < 1e-13
    assert result.plane_kind == "unqualified"
    assert state.descan_deflector._physical_host is host
    assert physical == [(lens.key, lens.z_mm, lens.percent) for lens in state.lenses]
    assert emitter == vars(state.electron_gun.emitter)
    assert state.ac_deflector._scan_execution_receipt["status"] == "executed_shared_potential_response"


def test_execution_evidence_remains_copyable_by_the_physical_gun_capture(monkeypatch):
    import temsim.physics.unified_scan_calibration as calibration
    from temsim.immutable_json import freeze_json
    state = _state()
    def immutable_receipts(state, observations, **kwargs):
        result = _drift_response(state, observations, **kwargs)
        result["receipt"] = freeze_json({"traces": [{"runtime": "retained", "angles": (1., 2.)}]})
        return result
    monkeypatch.setattr(calibration, "measure_scan_responses", immutable_receipts)
    calibrate_unified_scan(state)
    copied = deepcopy(state.ac_deflector)
    assert copied._scan_execution_receipt == state.ac_deflector._scan_execution_receipt
    copied._scan_execution_receipt["traces"][0]["runtime"] = "changed copy"
    assert state.ac_deflector._scan_execution_receipt["traces"][0]["runtime"] == "retained"


def test_failed_unified_calibration_restores_all_drives_and_shared_host(monkeypatch):
    import temsim.physics.unified_scan_calibration as calibration
    state = _state()
    host = state.descan_deflector._physical_host
    before = [deepcopy({key: value for key, value in vars(part).items() if key != "_physical_host"})
              for part in (state.ac_deflector, state.descan_deflector)]
    def singular(state, targets, **kwargs):
        row = _drift_response(state, targets, **kwargs)
        row["phase"][host.key+":1"][:] = 0.
        return row
    monkeypatch.setattr(calibration, "measure_scan_responses", singular)
    with pytest.raises(ValueError, match="singular"):
        calibrate_unified_scan(state)
    for part, saved in zip((state.ac_deflector, state.descan_deflector), before):
        assert {key: value for key, value in vars(part).items() if key != "_physical_host"} == saved
    assert state.descan_deflector._physical_host is host


def test_upstream_observation_does_not_calibrate_unused_scanners(monkeypatch):
    import temsim.physics.unified_scan_calibration as calibration
    state = _state()
    def forbidden(*args, **kwargs):
        pytest.fail("Upstream-only request must not solve downstream scan drives")
    monkeypatch.setattr(calibration, "measure_scan_responses", forbidden)
    assert calibrate_unified_scan(state, observation_stop_z_mm=500.) is None


def test_unified_scan_preparation_selects_new_executor_and_reuses_receipt(monkeypatch):
    import temsim.physics.unified_scan_calibration as calibration
    import temsim.physics.scan_preparation as preparation
    state = _state()
    monkeypatch.setattr(calibration, "measure_scan_responses", _drift_response)
    def forbidden(*args, **kwargs):
        pytest.fail("Unified drive preparation must not execute legacy calibration")
    monkeypatch.setattr("temsim.physics.scan_geometry.calibrate_scan_system", forbidden)
    preparation.prepare_scan_drives(state, "unified-drift-oracle-preparation")
    receipt = deepcopy(state.ac_deflector._scan_execution_receipt)
    monkeypatch.setattr(calibration, "calibrate_unified_scan", forbidden)
    preparation.prepare_scan_drives(state, "unified-drift-oracle-preparation")
    assert state.ac_deflector._scan_execution_receipt == receipt


def test_position_and_angle_descan_cancels_both_drift_response_rows(monkeypatch):
    import temsim.physics.unified_scan_calibration as calibration
    from temsim.physics.scan_geometry import _descan_target
    state = _state()
    state.descan_deflector.descan_constraint = "position_and_angle"
    monkeypatch.setattr(calibration, "measure_scan_responses", _drift_response)
    _, _, result = calibrate_unified_scan(state)
    target = _descan_target(state)[2]
    measured = _drift_response(state, (target,))
    total = np.zeros((4, 2))
    for part in (state.ac_deflector, state.descan_deflector):
        host = str(getattr(part, "_physical_host_key", part.key))
        command = np.asarray(part.scan_command_matrix_mrad)*1e-3
        for i, mapping in enumerate(part.coil_kick_matrices()):
            total += measured["phase"][f"{host}:{i}"][0]@np.asarray(mapping)@command
    np.testing.assert_allclose(total, 0., atol=2e-23)
    assert result.constraint == "position_and_angle"


def test_pivot_offset_is_retained_in_the_resolved_physical_coil_map(monkeypatch):
    import temsim.physics.unified_scan_calibration as calibration
    state = _state()
    state.ac_deflector.pivot_offset_x = .02
    state.ac_deflector.pivot_offset_y = -.01
    monkeypatch.setattr(calibration, "measure_scan_responses", _drift_response)
    calibrate_unified_scan(state, observation_stop_z_mm=state.sample.z_mm)
    assert (state.ac_deflector.pivot_offset_x, state.ac_deflector.pivot_offset_y) == (.02, -.01)
    gain = state.ac_deflector.upper_coil_gain
    upper, lower = map(np.asarray, state.ac_deflector.coil_kick_matrices())
    np.testing.assert_allclose(upper+lower, gain*np.diag((.02, -.01)), atol=2e-16)


def test_actual_normal_preset_uses_shared_field_response_without_legacy_alignment(monkeypatch):
    """A real finite-field calibration; no gun or finite-beam qualification."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = _state()
    original_lenses = [(lens.key, lens.percent, lens.z_mm) for lens in state.lenses]
    def forbidden(*args, **kwargs):
        pytest.fail("Automatic unified scan must not use legacy canonical calibration")
    monkeypatch.setattr("temsim.optics.direct_alignment.canonical_source_basis", forbidden)
    monkeypatch.setattr("temsim.physics.scan_geometry.calibrate_scan_system", forbidden)
    command, residual, result = calibrate_unified_scan(state,
        observation_stop_z_mm=state.sample.z_mm, maximum_step_mm=.5)
    assert result is None  # The downstream pair is not consumed at the sample.
    assert np.isfinite(command).all() and np.linalg.norm(command) > 0
    assert residual < 1e-12
    assert original_lenses == [(lens.key, lens.percent, lens.z_mm) for lens in state.lenses]
    receipt = state.ac_deflector._scan_execution_receipt
    assert len(receipt["trace_receipts"]) == 9
    assert all(not row["empirical_aberrations_applied"] for row in receipt["trace_receipts"])
    assert len({row["geometry_digest"] for row in receipt["trace_receipts"]}) == 1
    # Evidence is reusable by a real gun capture; it does not recursively
    # become a physical parameter in the next immutable request.
    deepcopy(state.ac_deflector)
    snapshot = capture_instrument_snapshot(state)
    receipt["display_note"] = "Evidence only"
    assert capture_instrument_snapshot(state).digest == snapshot.digest


def test_private_response_retains_cancellation_without_capturing_it_as_input(monkeypatch):
    from temsim.physics.unified_scan_calibration import measure_scan_responses
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = _state()
    original = capture_instrument_snapshot(state).digest
    def cancelled():
        raise InterruptedError("Cancelled before tracing a response")
    state._tuning_cancelled = cancelled
    def trace(private, *args, **kwargs):
        assert private is not state
        assert private._tuning_cancelled is cancelled
        private._tuning_cancelled()
    monkeypatch.setattr("temsim.physics.unified_field_particle.trace_unified_response", trace)
    with pytest.raises(InterruptedError, match="Cancelled before"):
        measure_scan_responses(state, (state.sample.z_mm,))
    assert capture_instrument_snapshot(state).digest == original
