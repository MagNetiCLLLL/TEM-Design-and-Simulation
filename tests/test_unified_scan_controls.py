"""Public scan controls route to shared-field derivatives without legacy optics.

The exact finite-dipole drift oracle isolates routing and drive algebra. It is
not a qualification of a real column or a substitute for particle histories.
"""
from copy import deepcopy

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.physics import unified_scan_calibration as calibration
from temsim.physics.scan_calibration import restore_held, validate_record
from temsim.physics.scan_geometry import calibrate_scan_system, recalibrate_descan_only, _descan_target
from test_unified_scan_calibration import _state, _drift_response


@pytest.fixture
def shared_responses(monkeypatch):
    calls = []
    # No physical propagation runs in this fixture. Keep captured GUI jobs
    # independent of concurrent source edits elsewhere in the checkout.
    monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: "a"*64)

    def measured(state, targets, **kwargs):
        calls.append((tuple(targets), dict(kwargs)))
        return _drift_response(state, targets, **kwargs)

    monkeypatch.setattr(calibration, "measure_scan_responses", measured)

    def legacy(*args, **kwargs):
        pytest.fail("Unified controls must not invoke legacy response or canonical transfer")

    monkeypatch.setattr("temsim.physics.scan_geometry.finite_scan_responses", legacy)
    monkeypatch.setattr("temsim.physics.scan_geometry.canonical_source_basis", legacy)
    return calls


def _values(state):
    return tuple(deepcopy({key: value for key, value in vars(part).items()
                          if key != "_physical_host"})
                 for part in (state.ac_deflector, state.descan_deflector))


def _physical(state):
    return (deepcopy(state.electron_gun.to_dict()),
            [(lens.key, lens.z_mm, lens.percent, lens.cs_mm, lens.cc_mm) for lens in state.lenses],
            deepcopy(vars(state.descan_deflector._physical_host)))


def _view(state, qtbot):
    """Use the real asynchronous page with only response measurements replaced."""
    from temsim.gui.scan_panel import ScanControlView
    errors, changes = [], []
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    view.error.connect(errors.append)
    view.parameters_changed.connect(changes.append)
    return view, errors, changes


@pytest.mark.parametrize("field,value", [
    ("scan_pixel_size_nm", .022), ("scan_pixels_x", 96), ("scan_lines", 32),
    ("scan_frame_period_s", .2),
])
def test_gui_scan_edits_update_shared_raster_and_field_owned_drives(shared_responses, field, value, qtbot):
    from temsim.gui.scan_panel import ScanControlView
    state = _state()
    physical = _physical(state)
    view, errors, changes = _view(state, qtbot)
    ScanControlView._control_changed(view, "ac", field, value)
    assert errors == [] and changes == [f"ac.{field}"]
    assert getattr(state.ac_deflector, field) == getattr(state.descan_deflector, field) == value
    qtbot.waitUntil(lambda: not view.scan_calibration.busy, timeout=10000)
    assert not errors and changes[-1] == "scan_calibration.updated"
    assert len(shared_responses) == 1
    assert _physical(state) == physical
    assert state.ac_deflector._scan_execution_receipt["status"] == "executed_shared_potential_response"


def test_gui_capture_and_descan_buttons_use_shared_fields(shared_responses, qtbot):
    from temsim.gui.scan_panel import ScanControlView
    state = _state()
    physical, host = _physical(state), state.descan_deflector._physical_host
    view, errors, changes = _view(state, qtbot)
    ScanControlView._calibrate_and_hold(view)
    qtbot.waitUntil(lambda: not view.scan_calibration.busy, timeout=10000)
    assert errors == [] and changes == ["scan_calibration.captured"]
    assert state.ac_deflector.calibration_mode == "held"
    first = validate_record(state.ac_deflector.calibration_record_json)
    assert first["descan_calibrated"] and first["descan_physical_host_key"] == host.key
    ScanControlView._calibrate_descan_only(view)
    qtbot.waitUntil(lambda: not view.scan_calibration.busy, timeout=10000)
    assert errors == [] and changes[-1] == "scan_calibration.descan_captured"
    assert len(shared_responses) == 2
    assert state.descan_deflector._physical_host is host and _physical(state) == physical
    second = validate_record(state.ac_deflector.calibration_record_json)
    assert first["ac_ratio"] == second["ac_ratio"]
    assert first["command_mrad"] == second["command_mrad"]
    assert state.ac_deflector._scan_execution_receipt["ac_recalibrated"] is False
    assert state.ac_deflector._scan_execution_receipt["sample_position_relative_residual"] is None


def test_held_fov_edits_rescale_without_measuring_or_refocusing(shared_responses, qtbot):
    from temsim.gui.scan_panel import ScanControlView
    state = _state()
    calibrate_scan_system(state, force=True, hold=True)
    ac_command = np.asarray(state.ac_deflector.scan_command_matrix_mrad).copy()
    ds_command = np.asarray(state.descan_deflector.scan_command_matrix_mrad).copy()
    record = state.ac_deflector.calibration_record_json
    view, errors, changes = _view(state, qtbot)
    ScanControlView._control_changed(view, "ac", "scan_pixel_size_nm", .03)
    assert errors == [] and changes == ["ac.scan_pixel_size_nm"]
    qtbot.waitUntil(lambda: not view.scan_calibration.busy, timeout=10000)
    assert not errors
    assert len(shared_responses) == 1
    np.testing.assert_allclose(state.ac_deflector.scan_command_matrix_mrad, ac_command*1.5, rtol=1e-15)
    np.testing.assert_allclose(state.descan_deflector.scan_command_matrix_mrad, ds_command*1.5, rtol=1e-15)
    assert state.ac_deflector.calibration_record_json == record
    assert state.ac_deflector._scan_execution_receipt["status"] == "explicit_held_drives"


def test_public_held_request_synchronizes_raster_and_preserves_captured_drives(shared_responses):
    state = _state()
    calibrate_scan_system(state, force=True, hold=True)
    ac, descan = state.ac_deflector, state.descan_deflector
    host = descan._physical_host
    record = ac.calibration_record_json
    commands = [np.asarray(part.scan_command_matrix_mrad).copy() for part in (ac, descan)]
    mappings = [deepcopy(part.coil_kick_matrices()) for part in (ac, descan)]
    ac.scan_frame_period_s = .25
    calibrate_scan_system(state)
    assert ac.scan_frame_period_s == descan.scan_frame_period_s == .25
    for part, command in zip((ac, descan), commands):
        np.testing.assert_array_equal(part.scan_command_matrix_mrad, command)
    ac.scan_pixels_x = 96
    calibrate_scan_system(state)
    assert ac.scan_pixels_x == descan.scan_pixels_x == 96
    for part, command, mapping in zip((ac, descan), commands, mappings):
        np.testing.assert_allclose(part.scan_command_matrix_mrad, command @ np.diag((1.5, 1.)), rtol=1e-15)
        np.testing.assert_array_equal(part.coil_kick_matrices(), mapping)
    assert len(shared_responses) == 1 and ac.calibration_record_json == record
    assert ac.calibration_mode == "held" and descan._physical_host is host


@pytest.mark.parametrize("column", [option.name for option in AssemblyCatalog().columns])
@pytest.mark.parametrize("constraint", ["position", "position_and_angle"])
def test_seven_columns_descan_only_preserves_ac_and_shared_host(shared_responses, monkeypatch, column, constraint):
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", column, "No Energy Filter"))
    state.simulation_mode = "unified"
    state.ac_deflector.scan_enabled = state.descan_deflector.scan_enabled = True
    state.descan_deflector.descan_constraint = constraint
    host = state.descan_deflector._physical_host
    host.upper_x_mrad = .001
    host.lower_y_mrad = -.002
    physical = _physical(state)
    calibrate_scan_system(state, force=True, hold=True)
    ac_record = validate_record(state.ac_deflector.calibration_record_json)
    original_ac = (deepcopy(state.ac_deflector.pure_shift_lower_ratio_matrix),
                   deepcopy(state.ac_deflector.scan_command_matrix_mrad))

    def changed_downstream(current, targets, **kwargs):
        result = _drift_response(current, targets, **kwargs)
        # An independent changed downstream response; no AC solve can work on
        # these deliberately singular canonical rows, nor may one be attempted.
        for key in result["phase"]:
            if key.startswith(host.key+":"):
                result["phase"][key] *= .8
            else:
                result["canonical"][key][:] = 0.
        return result

    monkeypatch.setattr(calibration, "measure_scan_responses", changed_downstream)
    result = recalibrate_descan_only(state)
    assert result.constraint == constraint
    np.testing.assert_array_equal(state.ac_deflector.pure_shift_lower_ratio_matrix, original_ac[0])
    np.testing.assert_array_equal(state.ac_deflector.scan_command_matrix_mrad, original_ac[1])
    assert state.ac_deflector.calibration_mode == "held"
    record = validate_record(state.ac_deflector.calibration_record_json)
    assert record["ac_angle_target"] == ac_record["ac_angle_target"]
    assert record["descan_calibrated"] and record["descan_physical_host_key"] == host.key
    assert state.descan_deflector._physical_host is host and _physical(state) == physical
    target = _descan_target(state)[2]
    measured = changed_downstream(state, (target,))
    total = np.zeros((4, 2))
    for part in (state.ac_deflector, state.descan_deflector):
        key = str(getattr(part, "_physical_host_key", part.key))
        for index, mapping in enumerate(part.coil_kick_matrices()):
            total += measured["phase"][f"{key}:{index}"][0] @ np.asarray(mapping) @ (
                np.asarray(part.scan_command_matrix_mrad)*1e-3)
    np.testing.assert_allclose(total[:2], 0., atol=1e-21)
    if constraint == "position_and_angle":
        np.testing.assert_allclose(total[2:], 0., atol=1e-21)


@pytest.mark.parametrize("operation", ["force", "descan"])
@pytest.mark.parametrize("failure", ["singular", "capture", "commit", "cancel"])
def test_failed_explicit_solve_restores_drives_record_and_host(shared_responses, monkeypatch, operation, failure):
    from temsim.physics import unified_scan_cache
    state = _state()
    calibrate_scan_system(state, force=True, hold=True)
    before, physical = _values(state), _physical(state)
    host = state.descan_deflector._physical_host
    discarded = []

    def measured(current, targets, **kwargs):
        result = _drift_response(current, targets, **kwargs)
        result["receipt"]["response_cache"] = {"key": "test-pending-response"}
        if failure == "singular":
            result["phase"][host.key+":1"][:] = 0.
        return result

    def reject(*args, **kwargs):
        raise ValueError("Controlled capture or commit failure")

    monkeypatch.setattr(calibration, "measure_scan_responses", measured)
    monkeypatch.setattr(unified_scan_cache, "discard_scan_response", discarded.append)
    if failure == "capture":
        monkeypatch.setattr(calibration, "capture_record", reject)
    if failure == "commit":
        monkeypatch.setattr(unified_scan_cache, "commit_scan_response", reject)
    if failure == "cancel":
        # Cancel after a response was measured, before publication.
        checks = []
        def cancelled():
            checks.append(True)
            return len(checks) > 1
        state._tuning_cancelled = cancelled
    with pytest.raises(RuntimeError if failure == "cancel" else ValueError):
        if operation == "force":
            calibrate_scan_system(state, force=True, hold=True)
        else:
            recalibrate_descan_only(state)
    assert discarded == ["test-pending-response"]
    assert _values(state) == before and _physical(state) == physical
    assert state.descan_deflector._physical_host is host


def test_descan_only_rejects_missing_hold_bias_and_disabled_shared_host(shared_responses):
    state = _state()
    with pytest.raises(ValueError, match="held AC"):
        recalibrate_descan_only(state)
    calibrate_scan_system(state, force=True, hold=True)
    state.descan_deflector.kick_x_mrad = .001
    before = _values(state)
    with pytest.raises(ValueError, match="static bias"):
        recalibrate_descan_only(state)
    assert _values(state) == before
    state.descan_deflector.kick_x_mrad = 0.
    state.descan_deflector._physical_host.enabled = False
    with pytest.raises(ValueError, match="physical hosts"):
        recalibrate_descan_only(state)
    assert len(shared_responses) == 1


def test_automatic_api_and_common_entry_resolve_identical_drives(shared_responses):
    direct, routed = _state(), _state()
    calibration.calibrate_unified_scan(direct)
    calibrate_scan_system(routed)
    assert _values(direct) == _values(routed)


def test_force_replaces_held_reference_without_changing_requested_mode(shared_responses):
    state = _state()
    calibrate_scan_system(state, force=True, hold=True)
    first = state.ac_deflector.calibration_record_json
    state.ac_deflector.scan_pixel_size_nm *= 1.1
    calibrate_scan_system(state, force=True)
    assert state.ac_deflector.calibration_mode == "held"
    assert state.ac_deflector.calibration_record_json != first
    assert len(shared_responses) == 2
    saved = deepcopy(state.ac_deflector.scan_command_matrix_mrad)
    restore_held(state)
    np.testing.assert_array_equal(state.ac_deflector.scan_command_matrix_mrad, saved)


@pytest.mark.parametrize("step,expected", [(.05, .05), (.5, .2)])
def test_public_controls_match_workflow_response_step_contract(shared_responses, step, expected):
    state = _state()
    state.step_mm = step
    calibrate_scan_system(state, force=True, hold=True)
    recalibrate_descan_only(state)
    assert [call[1]["maximum_step_mm"] for call in shared_responses] == [expected, expected]
    assert state.step_mm == step


@pytest.mark.parametrize("force", [False, True])
def test_upstream_public_request_does_not_consume_scan_calibration(shared_responses, force):
    state = _state()
    before = _values(state)
    assert calibrate_scan_system(state, force=force, hold=True, observation_stop_z_mm=500.) is None
    assert shared_responses == [] and _values(state) == before


@pytest.mark.parametrize("explicit_stop", [False, True])
def test_explicit_capture_can_prepare_disabled_raster_without_enabling_it(shared_responses, explicit_stop):
    state = _state()
    state.ac_deflector.scan_enabled = state.descan_deflector.scan_enabled = False
    stop = state.sample.lower_surface_z_mm if explicit_stop else None
    _, _, descan = calibrate_scan_system(state, force=True, hold=True, observation_stop_z_mm=stop)
    assert descan is None and len(shared_responses) == 1
    assert not state.ac_deflector.scan_enabled and not state.descan_deflector.scan_enabled
    assert state.ac_deflector.calibration_mode == "held"
    assert validate_record(state.ac_deflector.calibration_record_json)["descan_calibrated"] is False


def test_ordinary_downstream_request_skips_disabled_rasters(shared_responses):
    state = _state()
    state.ac_deflector.scan_enabled = state.descan_deflector.scan_enabled = False
    before = _values(state)
    assert calibrate_scan_system(state, observation_stop_z_mm=state.sample.lower_surface_z_mm) is None
    assert shared_responses == [] and _values(state) == before


@pytest.mark.parametrize("explicit_stop", [False, True])
def test_explicit_capture_preserves_disabled_coil_cutoff_policy(shared_responses, explicit_stop):
    state = _state()
    state.ac_deflector.enabled = False
    state.ac_deflector.scan_enabled = state.descan_deflector.scan_enabled = False
    before = _values(state)
    if explicit_stop:
        assert calibrate_scan_system(state, force=True, hold=True,
            observation_stop_z_mm=state.sample.lower_surface_z_mm) is None
    else:
        with pytest.raises(ValueError, match="physical AC coils"):
            calibrate_scan_system(state, force=True, hold=True)
    assert shared_responses == [] and _values(state) == before


@pytest.mark.parametrize("mode", ["ideal", "custom", "ideal_thin", "analytical"])
def test_compatibility_modes_retain_existing_scan_dispatch(monkeypatch, mode):
    from temsim.physics import scan_geometry
    state = _state()
    state.simulation_mode = mode
    calls = []
    command = np.eye(2)

    def forbidden(*args, **kwargs):
        pytest.fail("Compatibility scan must retain its selected optical model")

    monkeypatch.setattr(calibration, "calibrate_unified_scan", forbidden)
    monkeypatch.setattr(scan_geometry, "calibrate_ac_scan_scale", lambda current: (command, 0.))
    monkeypatch.setattr(scan_geometry, "calibrate_descan_image_plane", lambda current: calls.append(current))
    result, residual, _ = calibrate_scan_system(state)
    np.testing.assert_array_equal(result, command)
    assert residual == 0. and calls == [state]


def test_prepare_after_control_edit_reuses_effective_step_response(monkeypatch, tmp_path):
    """Real capture, cache and preparation with analytical derivative traces."""
    from collections import OrderedDict
    from temsim.physics import scan_preparation, unified_scan_cache
    state = _state()
    state.step_mm = .05
    calls = []
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_CACHE_DIR", str(tmp_path / "scan-cache"))
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "1")
    monkeypatch.setattr(scan_preparation, "_PREPARED", OrderedDict())
    unified_scan_cache.clear_scan_response_cache()

    def trace(private, initial, source, targets, **kwargs):
        calls.append(kwargs["maximum_step_m"])
        phases = _drift_response(private, [z*1e3 for z in targets])["phase"]
        values = np.zeros((len(targets), 1, 4))
        for key, kick in (kwargs.get("coil_perturbations_rad") or {}).items():
            values[:, 0] += phases[key] @ np.asarray(kick)
        return {"xy_slopes": values,
                "kinetic_energy_ev": np.full((len(targets), 1), kwargs["energy_ev"]),
                "vector_potential_t_m": np.zeros((len(targets), 1, 3)), "alive": np.array([True]),
                "receipt": {"scope": "Analytical drift trace; not full-column qualification",
                    "swept_bounds_m": [[-1e-4, -1e-4, source], [1e-4, 1e-4, targets[-1]]]}}

    monkeypatch.setattr("temsim.physics.unified_field_particle.trace_unified_response", trace)
    try:
        calibrate_scan_system(state, observation_stop_z_mm=state.sample.z_mm)
        first = deepcopy(state.ac_deflector._scan_execution_receipt)
        command = np.asarray(state.ac_deflector.scan_command_matrix_mrad)
        state.ac_deflector.scan_pixel_size_nm *= 1.1
        scan_preparation.prepare_scan_drives(state, "effective-cap-control-edit",
                                             observation_stop_z_mm=state.sample.z_mm)
        second = state.ac_deflector._scan_execution_receipt
        assert len(calls) == 9 and second["response_cache"]["hit"]
        assert first["response_cache"]["key"] == second["response_cache"]["key"]
        assert first["maximum_step_mm"] == second["maximum_step_mm"] == .05
        np.testing.assert_allclose(state.ac_deflector.scan_command_matrix_mrad, command*1.1, rtol=1e-14)
        finer = calibration.measure_scan_responses(state, (state.sample.z_mm,), maximum_step_mm=.025)
        assert len(calls) == 18 and not finer["receipt"]["response_cache"]["hit"]
        assert finer["receipt"]["response_cache"]["key"] != first["response_cache"]["key"]
        assert state.step_mm == .05
    finally:
        unified_scan_cache.clear_scan_response_cache()


@pytest.mark.parametrize("state_step,cap", [(True, .2), (float("nan"), .2), (.05, True), (.05, float("nan"))])
def test_invalid_response_step_cannot_be_masked_by_other_bound(monkeypatch, state_step, cap):
    state = _state()
    state.step_mm = state_step

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid numerical bounds must fail before runtime construction")

    monkeypatch.setattr("temsim.physics.unified_field_runtime.get_unified_field_runtime", forbidden)
    with pytest.raises(ValueError, match="positive and finite"):
        calibration.measure_scan_responses(state, (state.sample.z_mm,), maximum_step_mm=cap)
