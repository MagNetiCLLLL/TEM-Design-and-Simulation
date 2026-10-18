"""Phase-space Descan integrates with shared hardware and held execution."""
from copy import deepcopy
import json

import numpy as np
import pytest

from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.physics import scan_geometry as scan
from temsim.physics.finite_scan_response import FiniteScanResponse
from temsim.physics.scan_calibration import capture_record, restore_held, validate_record


@pytest.fixture(scope="module")
def normal_snapshot():
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.optics.column import default_state

    state = default_state()
    apply_operating_mode_pair(state, "nano_probe", "diffraction", column_name="C3 + Probe Corrector")
    state.acceleration_enabled = True
    state.acceleration_backend = "Numba CPU"
    state.step_mm = .1
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.scan_enabled = state.descan_deflector.scan_enabled = True
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 4
    state.ac_deflector.scan_pixel_size_nm = .02
    return encode_instrument(state)


@pytest.fixture
def state(normal_snapshot):
    return decode_instrument(normal_snapshot)


@pytest.fixture
def analytical(state, monkeypatch):
    """Cross-coupled incident response and two independently controllable drifts."""
    ac, ds = state.ac_deflector, state.descan_deflector
    ac.set_pure_shift_coupling(-np.eye(2))
    ac.set_scan_command_matrix_mrad(((.001, .0002), (-.0001, .0008)))
    ds.descan_constraint = "position_and_angle"
    incident = np.array(((.12, .02), (-.01, .15), (.7, .2), (-.1, .8)))
    upper = np.vstack((.05 * np.eye(2), np.eye(2)))
    lower = np.vstack((.025 * np.eye(2), np.eye(2)))

    def phases(_state, component, _z, *, with_reference=False):
        result = (incident, np.zeros((4, 2))) if component.key == ac.key else (upper, lower)
        return (*result, np.zeros(2)) if with_reference else result

    def finite(_state, _component, zs, **_kwargs):
        zs = np.asarray(zs, dtype=float)
        return FiniteScanResponse(zs, np.repeat(upper[None], len(zs), axis=0),
            np.repeat(lower[None], len(zs), axis=0), np.zeros((len(zs), 4)),
            float(ac.upper_z_mm),
            ((ds.upper_z_mm-7.5, ds.upper_z_mm+7.5), (ds.lower_z_mm-7.5, ds.lower_z_mm+7.5)),
            "image_diffraction_deflector", 1e-6)

    monkeypatch.setattr(scan, "_coil_phase_responses", phases)
    monkeypatch.setattr(scan, "finite_scan_responses", finite)
    monkeypatch.setattr(scan, "_sample_plane_classification", lambda *_: ("mixed", .023, .4))
    return state, incident, upper, lower


def _hold_phase(state):
    result = scan.calibrate_descan_image_plane(state)
    capture_record(state, descan_calibrated=True, ac_angle_target="canonical")
    state.ac_deflector.calibration_mode = "held"
    return result


def _physical_drives(state):
    ds = state.descan_deflector
    command = np.asarray(ds.scan_command_matrix_mrad) * 1e-3
    return tuple(np.asarray(mapping) @ command for mapping in ds.coil_kick_matrices())


def test_phase_constraint_reaches_independent_physical_coils_and_preserves_ac_host(analytical):
    state, incident, upper, lower = analytical
    ac, ds, host = state.ac_deflector, state.descan_deflector, state.image_diffraction_deflector
    host.upper_x_mrad, host.lower_y_mrad = .15, -.08
    before_ac, before_host = deepcopy(ac.__dict__), deepcopy(host.__dict__)
    result = scan.calibrate_descan_image_plane(state)
    incoming = incident @ (.5 * np.asarray(ac.scan_command_matrix_mrad) * 1e-3)
    # Independent analytical elimination for two free-space kicks 25 mm apart.
    expected_upper = (.025 * incoming[2:] - incoming[:2]) / .025
    expected_lower = -incoming[2:] - expected_upper
    actual_upper, actual_lower = _physical_drives(state)
    np.testing.assert_allclose(actual_upper, expected_upper, rtol=2e-14, atol=1e-20)
    np.testing.assert_allclose(actual_lower, expected_lower, rtol=2e-14, atol=1e-20)
    np.testing.assert_allclose(incoming + upper @ actual_upper + lower @ actual_lower,
                               0., atol=2e-20)
    assert ac.__dict__ == before_ac and host.__dict__ == before_host
    assert result.constraint == "position_and_angle" and result.plane_kind == "mixed"
    assert result.conjugacy_residual_m_per_rad == .023
    assert result.angle_response_residual < 1e-13 and result.response_match_residual < 1e-13
    assert not np.allclose(ds.scan_command_matrix_mrad, -np.asarray(ac.scan_command_matrix_mrad), atol=1e-12)
    assert ds._physical_host is host


def test_legacy_position_option_preserves_opposite_command_and_has_unconstrained_angle(analytical):
    state, _, _, _ = analytical
    state.descan_deflector.descan_constraint = "position"
    result = scan.calibrate_descan_image_plane(state)
    np.testing.assert_array_equal(state.descan_deflector.scan_command_matrix_mrad,
                                  -np.asarray(state.ac_deflector.scan_command_matrix_mrad))
    measured = scan._measure_descan(state)
    assert result.constraint == measured.constraint == "position"
    assert measured.response_match_residual < 1e-13
    assert measured.angle_response_residual > .1


@pytest.mark.parametrize("failure", (
    "singular", "host_limit", "channel_limit", "shared_alignment", "inside_support", "static_bias",
))
def test_rejected_phase_solve_rolls_back_raster_and_all_derived_descan_state(analytical, monkeypatch, failure):
    state, _, upper, _ = analytical
    ac, ds = state.ac_deflector, state.descan_deflector
    original_finite = scan.finite_scan_responses
    if failure in ("singular", "inside_support"):
        from dataclasses import replace

        def broken(*args, **kwargs):
            response = original_finite(*args, **kwargs)
            if failure == "singular":
                return replace(response, lower=response.upper.copy())
            return replace(response, coil_support_mm=((1700., 1750.), (1800., float(response.z_mm[-1]))))
        monkeypatch.setattr(scan, "finite_scan_responses", broken)
    elif failure == "host_limit":
        state.image_diffraction_deflector.maximum_kick_mrad = 1e-6
    elif failure == "channel_limit":
        ds.maximum_kick_mrad = 1e-6
    elif failure == "shared_alignment":
        # The static alignment is legal on its own, but one raster corner adds
        # to it. Validation must cover the combined physical host drive.
        state.image_diffraction_deflector.maximum_kick_mrad = .003
        state.image_diffraction_deflector.upper_x_mrad = .0029
    else:
        ds.kick_x_mrad = .01
    ds.scan_pixels_x = 9  # A failed solve must roll back raster synchronization too.
    before = deepcopy(ds.__dict__)
    before_ac = deepcopy(ac.__dict__)
    before_host = deepcopy(state.image_diffraction_deflector.__dict__)
    with pytest.raises(ValueError):
        scan.calibrate_descan_image_plane(state)
    assert ds.__dict__ == before and ac.__dict__ == before_ac
    assert state.image_diffraction_deflector.__dict__ == before_host


def test_held_record_restores_actual_descan_command_and_scales_each_raster_axis(analytical):
    state, *_ = analytical
    _hold_phase(state)
    ac, ds = state.ac_deflector, state.descan_deflector
    record = json.loads(ac.calibration_record_json)
    assert record["descan_constraint"] == "position_and_angle"
    assert not np.allclose(record["descan_command_mrad"], -np.asarray(record["command_mrad"]), atol=1e-12)
    ac.scan_pixels_x *= 2
    ac.scan_lines *= 3
    ds.set_scan_command_matrix_mrad(np.eye(2) * .01)
    ds.set_image_plane_coupling(-np.eye(2))
    restored = restore_held(state)
    np.testing.assert_allclose(restored, np.asarray(record["command_mrad"]) @ np.diag((2., 3.)))
    np.testing.assert_allclose(ds.scan_command_matrix_mrad,
                               np.asarray(record["descan_command_mrad"]) @ np.diag((2., 3.)))
    assert ds.image_plane_lower_ratio_matrix == tuple(map(tuple, record["descan_ratio"]))
    assert ds._calibrated_constraint == "position_and_angle"
    assert json.loads(ac.calibration_record_json) == record


def test_held_fov_expansion_rejects_shared_host_peak_without_changing_accepted_drives(analytical):
    state, *_ = analytical
    host = state.image_diffraction_deflector
    host.upper_x_mrad, host.lower_y_mrad = .0004, -.0003
    _hold_phase(state)
    ac, ds = state.ac_deflector, state.descan_deflector
    corners = np.array(((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)))
    static = np.array(((host.upper_x_mrad, host.upper_y_mrad),
                       (host.lower_x_mrad, host.lower_y_mrad)))
    peak = max(np.max(np.abs(offset + corners @ (kick * 1e3).T))
               for offset, kick in zip(static, _physical_drives(state)))
    host.maximum_kick_mrad = float(peak * 1.5)
    restore_held(state)  # The original accepted raster and alignment still fit.
    ac.scan_pixels_x *= 3
    ac.scan_lines *= 3
    before = [deepcopy(component.__dict__) for component in (ac, ds, host)]
    with pytest.raises(ValueError, match="shared physical coil limit"):
        restore_held(state)
    for component, original in zip((ac, ds, host), before):
        assert component.__dict__ == original


def test_old_held_record_keeps_opposite_ac_fallback_without_relabelling_requested_constraint(analytical):
    state, *_ = analytical
    state.descan_deflector.descan_constraint = "position"
    _hold_phase(state)
    ac, ds = state.ac_deflector, state.descan_deflector
    record = json.loads(ac.calibration_record_json)
    record.pop("descan_command_mrad")
    record.pop("descan_constraint")
    ac.calibration_record_json = json.dumps(record)
    ds.descan_constraint = "position_and_angle"  # A request is not accepted calibration.
    restore_held(state)
    np.testing.assert_array_equal(ds.scan_command_matrix_mrad, -np.asarray(record["command_mrad"]))
    assert ds._calibrated_constraint == "position"
    assert scan._measure_descan(state).constraint == "position"
    assert ds.descan_constraint == "position_and_angle"


def test_descan_only_recalibration_adopts_requested_constraint_without_resolving_ac(analytical, monkeypatch):
    state, *_ = analytical
    _hold_phase(state)
    ac, ds = state.ac_deflector, state.descan_deflector
    before = ac.pure_shift_lower_ratio_matrix, ac.scan_command_matrix_mrad
    ds.descan_constraint = "position"
    restore_held(state)
    assert scan._measure_descan(state).constraint == "position_and_angle"
    monkeypatch.setattr(scan, "calibrate_ac_pure_shift", lambda *_a, **_k: pytest.fail("Held AC cannot be re-solved"))
    result = scan.recalibrate_descan_only(state)
    assert result.constraint == "position"
    assert (ac.pure_shift_lower_ratio_matrix, ac.scan_command_matrix_mrad) == before
    assert json.loads(ac.calibration_record_json)["descan_constraint"] == "position"
    np.testing.assert_array_equal(ds.scan_command_matrix_mrad, -np.asarray(ac.scan_command_matrix_mrad))


@pytest.mark.parametrize("change", (
    {"descan_command_mrad": None}, {"descan_command_mrad": [[0.], [0.]]},
    {"descan_command_mrad": [[0., float("nan")], [0., 0.]]},
    {"descan_constraint": "unverified"},
))
def test_invalid_new_held_phase_record_is_rejected_before_any_drive_changes(analytical, change):
    state, *_ = analytical
    _hold_phase(state)
    record = json.loads(state.ac_deflector.calibration_record_json)
    record.update(change)
    record = {key: value for key, value in record.items() if value is not None}
    state.ac_deflector.calibration_record_json = json.dumps(record)
    before = [deepcopy(component.__dict__) for component in (state.ac_deflector, state.descan_deflector)]
    with pytest.raises(ValueError):
        restore_held(state)
    for component, original in zip((state.ac_deflector, state.descan_deflector), before):
        assert component.__dict__ == original


@pytest.mark.parametrize("method", ("copy", "cache"))
def test_prepared_handoff_preserves_accepted_constraint_and_independent_commands(analytical, normal_snapshot, monkeypatch, method):
    from temsim.physics import scan_preparation as preparation

    state, *_ = analytical
    scan.calibrate_descan_image_plane(state)
    target = decode_instrument(normal_snapshot)
    target.descan_deflector.descan_constraint = "position_and_angle"
    preparation._PREPARED.clear()
    try:
        if method == "copy":
            preparation.copy_resolved_scan_drives(state, target)
        else:
            preparation.remember_resolved_scan_drives(state, "phase-space-fixture")
            monkeypatch.setattr(scan, "calibrate_scan_system", lambda *_a, **_k: pytest.fail("Accepted cache must be reused"))
            preparation.prepare_scan_drives(target, "phase-space-fixture")
        np.testing.assert_array_equal(target.descan_deflector.scan_command_matrix_mrad,
                                      state.descan_deflector.scan_command_matrix_mrad)
        assert getattr(target.descan_deflector, "_calibrated_constraint", "position") == "position_and_angle"
    finally:
        preparation._PREPARED.clear()


@pytest.mark.parametrize("requested", ("position", "position_and_angle"))
def test_profile_roundtrip_keeps_requested_and_accepted_constraints_separate(analytical, normal_snapshot, tmp_path, requested):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.profile_io import apply_profile_values, read_profile, save_profile

    state, *_ = analytical
    _hold_phase(state)
    state.descan_deflector.descan_constraint = requested
    record = state.ac_deflector.calibration_record_json
    accepted_command = state.descan_deflector.scan_command_matrix_mrad
    catalog = AssemblyCatalog()
    path = tmp_path / "phase-descan.toml"
    save_profile(path, state, catalog.selection_for_resolved(state._resolved_assembly))
    selection, values = read_profile(path)
    assert values["descan_deflector"]["descan_constraint"] == requested
    loaded = decode_instrument(normal_snapshot)
    catalog.apply(loaded, selection)
    apply_profile_values(loaded, values)
    apply_physical_layout_to_state(loaded, preserve_operating_parameters=True)
    restore_held(loaded)
    assert loaded.ac_deflector.calibration_record_json == record
    assert loaded.descan_deflector.descan_constraint == requested
    assert loaded.descan_deflector._calibrated_constraint == "position_and_angle"
    np.testing.assert_array_equal(loaded.descan_deflector.scan_command_matrix_mrad, accepted_command)
    assert loaded.descan_deflector._physical_host is loaded.image_diffraction_deflector


def test_requested_constraint_invalidates_preparation_and_physics_request_identity(state):
    from temsim.calculation_cache import state_model_signature
    from temsim.physics.scan_preparation import scan_drive_identity

    before = state_model_signature(state), scan_drive_identity(state)
    state.descan_deflector.descan_constraint = "position_and_angle"
    after = state_model_signature(state), scan_drive_identity(state)
    assert all(left != right for left, right in zip(before, after))


def test_geometry_captures_downstream_mechanical_spans_from_accepted_finite_coil_paths(analytical, monkeypatch):
    from dataclasses import replace
    from temsim.optics import direct_alignment

    state, incident, upper, lower = analytical
    ac, ds = state.ac_deflector, state.descan_deflector
    ds.descan_constraint = "position"
    calibration = scan.calibrate_descan_image_plane(state)
    original_finite = scan.finite_scan_responses

    def response(_state, component, zs, **kwargs):
        finite = original_finite(_state, component, zs, **kwargs)
        if component is ac:
            return replace(finite, upper=np.repeat(incident[None], len(zs), axis=0),
                           lower=np.zeros_like(finite.lower))
        return finite

    monkeypatch.setattr(scan, "finite_scan_responses", response)
    monkeypatch.setattr(scan, "canonical_source_basis", lambda *_a, **_k: np.eye(4))
    monkeypatch.setattr(direct_alignment, "diffraction_transfers", lambda *_a, **_k: {})
    result = scan.calculate_scan_geometry(state,
        calibration=(np.asarray(ac.scan_command_matrix_mrad), 0., calibration))
    upper_map, lower_map = map(np.asarray, ds.coil_kick_matrices())
    phases = []
    for time in result.times_s.ravel():
        ac_command = np.asarray(ac.scan_kick_mrad(float(time))) * 1e-3
        ds_command = np.asarray(ds.scan_kick_mrad(float(time))) * 1e-3
        phases.append(incident @ (.5 * ac_command)
            + upper @ upper_map @ ds_command + lower @ lower_map @ ds_command)
    expected = np.ptp(np.asarray(phases)[:, 2:], axis=0) * 1e3
    assert np.min(expected) > 0.  # Position compensation leaves a measurable slope.
    assert result.plane_mechanical_angle_spans_mrad.keys() == result.plane_positions_um.keys()
    for key in (state.selected_area_aperture.key, state.fluorescent_screen.key):
        np.testing.assert_allclose(result.plane_mechanical_angle_spans_mrad[key], expected,
                                   rtol=1e-13, atol=1e-18)


def test_offscreen_constraint_selector_changes_request_but_summary_retains_captured_constraint(analytical, qtbot):
    from temsim.gui.scan_panel import ScanControlView

    state, *_ = analytical
    _hold_phase(state)
    record = state.ac_deflector.calibration_record_json
    command = state.descan_deflector.scan_command_matrix_mrad
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    assert view.descan_constraint.currentData() == "position_and_angle"
    changes = []
    view.parameters_changed.connect(changes.append)
    view.descan_constraint.setCurrentIndex(view.descan_constraint.findData("position"))
    assert changes == ["scan_calibration.descan_constraint"]
    assert state.descan_deflector.descan_constraint == "position"
    assert state.descan_deflector._calibrated_constraint == "position_and_angle"
    assert state.descan_deflector.scan_command_matrix_mrad == command
    assert state.ac_deflector.calibration_record_json == record
    assert "descan_constraint" not in state.ac_deflector.__dict__
    assert decode_instrument(encode_instrument(state)).descan_deflector.descan_constraint == "position"
    saa_key, screen_key = state.selected_area_aperture.key, state.fluorescent_screen.key
    captured = scan.ScanGeometryResult(
        times_s=np.zeros((2, 2)), sample_x_um=np.zeros((2, 2)),
        sample_y_um=np.zeros((2, 2)),
        plane_positions_um={saa_key: (np.array([[-3., 4.], [-3., 4.]]),
                                     np.array([[-1., -1.], [2., 2.]])),
                            screen_key: (np.array([[0., 2.], [0., 2.]]),
                                         np.array([[0., 0.], [5., 5.]]))},
        plane_names={saa_key: "Selected area aperture", screen_key: "Screen"},
        plane_mechanical_angle_spans_mrad={saa_key: (.12, .34), screen_key: (0., 0.)},
        requested_pixels_x=2, requested_pixels_y=2, ac_enabled=True, descan_enabled=True,
        ac_drift_pivot_z_mm=None, descan_drift_pivot_z_mm=None,
        ac_lower_from_upper=-np.eye(2), ac_angular_residual=0.,
        descan_target_key=saa_key, descan_target_z_mm=state.selected_area_aperture.z_mm,
        descan_target_name="Selected area aperture", descan_lower_from_upper=-np.eye(2),
        descan_target_plane_kind="mixed", descan_constraint="position_and_angle",
        descan_compensation_residual=3e-14, descan_angle_response_residual=2e-13)
    view.display_result(captured)
    assert "constraint position_and_angle" in view.summary.text()
    assert "mechanical-angle residual 2e-13" in view.summary.text()
    assert view.descan_constraint.currentData() == "position"
    assert "X 7, Y 3 µm" in view.downstream_motion.text()
    assert "Mechanical slope: X 0.12, Y 0.34 mrad" in view.downstream_motion.text()
    view.plane_selector.setCurrentIndex(view.plane_selector.findData(screen_key))
    assert "X 2, Y 5 µm" in view.downstream_motion.text()
    assert "Mechanical slope: X 0, Y 0 mrad" in view.downstream_motion.text()
    captured.plane_mechanical_angle_spans_mrad.clear()  # Old captured results cannot imply zero.
    view.display_result(captured)
    assert "Mechanical slope: unavailable" in view.downstream_motion.text()


@pytest.mark.parametrize("mode,range_key,c3_percent", [
    ("nano_probe", "normal", 9.207984804305022),
    ("nano_probe_large_angle", "large_angle", 46.059815218359184),
])
def test_current_nanoprobe_finite_fields_cancel_position_and_slope_at_saa_and_screen(
        state, mode, range_key, c3_percent):
    from temsim.physics.finite_scan_response import finite_scan_responses
    from temsim.operating_modes import apply_operating_mode_pair

    apply_operating_mode_pair(state, mode, "diffraction", column_name="C3 + Probe Corrector")
    state.ac_deflector.scan_pixel_size_nm = 1000. / state.ac_deflector.scan_pixels_x
    lens_controls = [(lens.key, lens.z_mm, lens.percent) for lens in state.lenses]
    state.descan_deflector.descan_constraint = "position_and_angle"
    _, _, result = scan.calibrate_scan_system(state, force=True, hold=True)
    targets = (state.selected_area_aperture.z_mm, state.fluorescent_screen.z_mm)
    assert result.constraint == "position_and_angle"
    assert state.objective_lens.percent == pytest.approx(68.9801)
    assert state.beam_voltage_kv == 300. and state.nanoprobe_range == range_key
    assert next(lens.percent for lens in state.lenses if lens.key == "condenser_lens_3") == pytest.approx(c3_percent)
    assert [(lens.key, lens.z_mm, lens.percent) for lens in state.lenses] == lens_controls
    assert state.descan_deflector._physical_host is state.image_diffraction_deflector
    assert (state.descan_deflector.upper_z_mm, state.descan_deflector.lower_z_mm) == (1674.4, 1699.4)
    assert targets == (1824.4, 2714.15)
    total = np.zeros((2, 4, 2))
    for component in (state.ac_deflector, state.descan_deflector):
        finite = finite_scan_responses(state, component, targets)
        upper_map, lower_map = map(np.asarray, component.coil_kick_matrices())
        command = np.asarray(component.scan_command_matrix_mrad) * 1e-3
        for i, z in enumerate(targets):
            upper, lower, _ = finite.at(z)
            total[i] += (upper @ upper_map + lower @ lower_map) @ command
    assert np.max(np.abs(total[:, :2])) < 1e-14  # metres, local derivative at a 1 um requested FOV
    assert np.max(np.abs(total[:, 2:])) < 1e-12  # mechanical radians
    record = validate_record(state.ac_deflector.calibration_record_json)
    assert record["descan_physical_host_key"] == "image_diffraction_deflector"
    assert record["descan_constraint"] == "position_and_angle"
