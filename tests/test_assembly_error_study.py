"""Fixed-reference studies, honest learning roles and physical electrical bounds."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim import input_io
from temsim.assembly_catalog import AssemblyCatalog
from temsim.assembly_error_study import (
    AssemblyError, AssemblyErrorCase, AssemblyStudyOptions, capture_study, load_study,
    prepare_error_case, run_study, save_study,
)
from temsim.assembly_error_response import headroom
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.lens_pose import lens_pose_registration, mechanical_pose_registration
from temsim.optics.aberration_basis import coefficient_fields
from temsim.optics.aberrations import EffectiveAberrationSet
from temsim.optics.column import default_state
from temsim.optics.field_aberrations import pupil_rays
from temsim.optics.probe_service import apply_probe_output_deltas, output_reading, set_probe_output_percent
from temsim.optics.system_residual_aberrations import SystemResidualReport


@pytest.fixture(scope="module")
def snapshot():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    input_io.bind_archive(state, input_io.capture_input_archive(state))
    return capture_instrument_snapshot(state)


@pytest.fixture
def state(snapshot):
    return snapshot.restore()


def case(delta=.02, parameter="field_azimuth_error_deg"):
    return AssemblyErrorCase("HP1 placement", (AssemblyError("probe_hp1_hexapole", parameter, delta),))


def options(**kwargs):
    return AssemblyStudyOptions(routes=("AT_A2fine.upper",), target_coefficients=("A2_x",),
        compare_learning=True, semiangle_mrad=5., calibration_step_percent=.1,
        pupil_rms_tolerance_nm=1e-5, **kwargs)


@pytest.fixture
def linear_measurement(monkeypatch):
    """Array-only surrogate with independently specified cross-response law."""
    import temsim.assembly_error_study as study
    import temsim.assembly_error_response as response
    references = []
    def capture(state, system, **kwargs):
        alpha = kwargs["semiangle_mrad"] * 1e-3
        return SimpleNamespace(holdout=pupil_rays(alpha, holdout=True), alpha_rad=alpha,
            energy_kev=state.beam_voltage_kv, z0_mm=0., z1_mm=state.sample.z_mm,
            pupil_reference="synthetic fixed pupil", matrix=np.eye(4), inputs=np.zeros((1, 4)))
    def measure(state, system, *, reference, **kwargs):
        references.append(reference)
        error = state._resolved_assembly.part("probe_hp1_hexapole").data.get("field_azimuth_error_deg", 0.)
        h = output_reading(state, "HPolx").percent
        q = output_reading(state, "QPoly").percent
        values = np.zeros(15)
        values[2] = (h + q) * 1e-3  # Natural A1_y coupling of HPolx; QPoly can cancel it.
        values[5] = .1 * h + .05 * error
        return SystemResidualReport(EffectiveAberrationSet(reference_plane="synthetic probe",
            correction_state="array test", **coefficient_fields(values)), dict(
            reference_centroid_xy_m=[0., 0.], reference_centroid_slopes_rad=[0., 0.],
            holdout_rms_m=0., cartesian_coefficients_mm=values.tolist()))
    monkeypatch.setattr(study, "capture_system_residual_reference", capture)
    monkeypatch.setattr(response, "measure_system_residual", measure)
    return references


def test_geometry_uses_shared_pose_editor_preserves_controls_and_live_inputs(state):
    before = capture_instrument_snapshot(state).physical_digest
    raw = input_io.capture_input_archive(state)
    errors = AssemblyErrorCase("Combined pose", (
        AssemblyError("probe_hp1_hexapole", "offset_x_mm", .002),
        AssemblyError("probe_hp1_hexapole", "offset_z_mm", .003),
        AssemblyError("probe_hp1_hexapole", "field_azimuth_error_deg", .02),
        AssemblyError("probe_hp1_hexapole", "rotation_y_mrad", .01),
    ))
    request = capture_study(state, (errors,))
    perturbed, evidence = prepare_error_case(request, errors)
    assert capture_instrument_snapshot(state).physical_digest == before
    assert input_io.capture_input_archive(state) == raw
    assert state.hp1_hexapole.strength_m3 == perturbed.hp1_hexapole.strength_m3
    assert state.hp1_hexapole.orientation_rad == perturbed.hp1_hexapole.orientation_rad
    optical = lens_pose_registration(perturbed, "probe_hp1_hexapole")
    mechanical = mechanical_pose_registration(perturbed, "probe_hp1_hexapole")
    np.testing.assert_allclose(mechanical.rotation_local_to_global, np.eye(3), atol=1e-14)
    assert not np.allclose(optical.rotation_local_to_global, mechanical.rotation_local_to_global)
    assert mechanical.origin_global_m[0] == pytest.approx(2e-6)
    assert mechanical.origin_global_m[2] == pytest.approx(3e-6)
    assert len(evidence["errors"]) == 4
    assert perturbed.step_mm == state.step_mm
    assert perturbed.history_step_mm == state.history_step_mm


def test_intrinsic_crosstalk_and_configured_learning_remain_separate(state, linear_measurement):
    state.probe_service_percent = dict(learning_enabled=False,
        learning={"AT_A2fine.upper": {"QPoly": -1.}})
    before = capture_instrument_snapshot(state).physical_digest
    request = capture_study(state, (case(),), options())
    result = run_study(request)
    row = result["cases"][0]
    assert row["status"] == "COMPLETE", row.get("failure")
    intrinsic, learned = row["intrinsic_only"], row["with_learning"]
    assert intrinsic["status"] == learned["status"] == "LOCAL_TARGET_MET"
    assert intrinsic["route_change_percent"]["AT_A2fine.upper"] == pytest.approx(-.01, abs=1e-9)
    assert abs(intrinsic["errors"]["coefficient_delta_mm"]["A1_y"]) > 1e-6
    assert abs(learned["errors"]["coefficient_delta_mm"]["A1_y"]) < 1e-12
    assert intrinsic["response"]["matrix"][2][0] == pytest.approx(1e-3, rel=1e-8)
    assert row["learning_only_response"]["matrix"][2][0] == pytest.approx(-1e-3, rel=1e-8)
    assert result["routes"][0]["intrinsic"] == {"HPolx": 1.}
    assert result["routes"][0]["learning"] == {"QPoly": -1.}
    assert all(ref is linear_measurement[0] for ref in linear_measurement)
    assert result["finite_source"]["current_a"] is None
    assert not learned["learning_coefficients_fitted"]
    assert capture_instrument_snapshot(state).physical_digest == before


def test_zero_learning_reuses_identical_evidence_and_zero_error_does_not_retune(state, linear_measurement):
    request = capture_study(state, (case(0.), replace(case(.02, "offset_y_mm"), name="Translation")), options())
    result = run_study(request)
    for row in result["cases"]:
        assert row["status"] == "COMPLETE", row.get("failure")
        assert row["intrinsic_only"]["physical_changes"] == ()
        assert row["with_learning"]["reused_evidence"].startswith("Identical")


def test_search_bound_is_not_reported_as_exhausted_hardware(state, linear_measurement):
    request = capture_study(state, (case(2.),), options(maximum_iterations=1))
    result = run_study(request)
    stage = result["cases"][0]["intrinsic_only"]
    assert stage["status"] == "LOCAL_SEARCH_LIMIT"
    assert stage["headroom_after"]["outputs"][0]["positive_margin_percent"] > 90
    assert "not proof" in stage["qualification"]


def test_vector_margins_and_simultaneous_moves_use_existing_hardware_limits(state):
    set_probe_output_percent(state, "HPoly", percent=0.)
    set_probe_output_percent(state, "HPolx", percent=60.)
    set_probe_output_percent(state, "HPoly", percent=80.)
    margin = headroom(state, [{"HPolx": 1.}], ("AT_A2fine.upper",))["outputs"][0]
    assert margin["positive_margin_percent"] < 1e-5
    assert margin["negative_margin_percent"] == pytest.approx(120., abs=1e-5)
    before = capture_instrument_snapshot(state).physical_digest
    with pytest.raises(ValueError):
        apply_probe_output_deltas(state, {"HPolx": 1., "QPoly": 1.})
    assert capture_instrument_snapshot(state).physical_digest == before
    # This simultaneous vector move stays inside the disk; no intermediate
    # invalid Cartesian component may be committed or tested as the final move.
    apply_probe_output_deltas(state, {"HPolx": 20., "HPoly": -20.})
    assert output_reading(state, "HPolx").percent == pytest.approx(80.)
    assert output_reading(state, "HPoly").percent == pytest.approx(60.)


def test_unknown_intrinsic_weights_and_alias_placement_are_rejected(state):
    with pytest.raises(ValueError, match="uncalibrated"):
        capture_study(state, (case(),), AssemblyStudyOptions(routes=("AT_A1fine.lower",)))
    alias = AssemblyErrorCase("DP12", (AssemblyError("probe_dp12_scan_deflector", "offset_x_mm", .001),))
    with pytest.raises(ValueError, match="placement owner"):
        capture_study(state, (alias,))
    host = AssemblyErrorCase("AC host", (AssemblyError("ac_deflector", "offset_x_mm", .001),))
    assert capture_study(state, (host,)).cases == (host,)


def test_dp12_headroom_is_a_limit_on_the_existing_ac_host(state):
    row = headroom(state, [{"DP12x": 1.}], ("AT_B2.x",))["outputs"][0]
    assert row["physical_host"] == "ac_deflector"
    assert 0 <= row["positive_margin_percent"] <= 100.
    with pytest.raises(ValueError):
        apply_probe_output_deltas(state, {"DP12x": row["positive_margin_percent"] + 1e-4})


def test_bad_geometry_case_is_retained_without_displacing_other_cases(state, linear_measurement, monkeypatch):
    import temsim.assembly_error_study as study
    original = study.prepare_error_case
    bad = replace(case(), name="Invalid clearance")
    good = replace(case(), name="Valid case")
    request = capture_study(state, (bad, good), options())
    def prepare(request, case, **kwargs):
        if case.name == "Invalid clearance":
            raise ValueError("Declared clearance failure")
        return original(request, case, **kwargs)
    monkeypatch.setattr(study, "prepare_error_case", prepare)
    result = run_study(request)
    assert [c["status"] for c in result["cases"]] == ["FAILED", "COMPLETE"]
    assert "clearance" in result["cases"][0]["failure"]


def test_cancellation_during_response_stops_the_study(state, linear_measurement):
    request = capture_study(state, (case(),), options())
    with pytest.raises(RuntimeError, match="cancelled"):
        run_study(request, cancelled=lambda: len(linear_measurement) >= 4)


def test_save_roundtrip_checksum_and_cancellation(state, linear_measurement, tmp_path):
    request = capture_study(state, (case(),), options())
    result = run_study(request)
    path = tmp_path / "study.json"
    save_study(path, request, result)
    restored, report = load_study(path)
    assert restored.digest == request.digest
    assert report == result
    assert restored.snapshot.restore().hp1_hexapole.strength_m3 == state.hp1_hexapole.strength_m3
    with pytest.raises(RuntimeError, match="cancelled"):
        run_study(restored, cancelled=lambda: True)
    data = json.loads(path.read_text())
    data["result"]["qualification"] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        load_study(path)


@pytest.mark.parametrize("parameters", [dict(compare_learning=1), dict(maximum_iterations=True),
    dict(calibration_step_percent=float("nan")), dict(routes=("AT_A2.upper", "AT_A2.upper")),
    dict(target_coefficients=("C5",))])
def test_options_reject_ambiguous_or_unmeasured_requests(parameters):
    with pytest.raises(ValueError):
        AssemblyStudyOptions(**parameters)
