"""Research protocols retain mode identity and never refit held comparisons."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def protocols():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        yield (importlib.import_module("study_descan_phase_space"),
               importlib.import_module("validate_descan_transport"))


@pytest.mark.parametrize("selected", (None, "normal", "large_angle"))
def test_linear_study_applies_actual_range_to_explicit_probe_column(protocols, selected):
    study, _ = protocols
    state, mode = study._configured_state() if selected is None else study._configured_state(selected)
    expected_range = selected or "normal"
    assert state.nanoprobe_range == expected_range
    assert mode == ("nano_probe_large_angle" if selected == "large_angle" else "nano_probe")
    assert dict(state._resolved_assembly.selected_module_paths)["column"] == "column/C3_ProbeCorrector.toml"
    assert state.illumination_mode == "STEM" and state.projector_mode == "diffraction"
    assert state.beam_voltage_kv == 300.
    lenses = {lens.key: lens for lens in state.lenses}
    assert lenses["condenser_lens_3"].percent == pytest.approx(
        46.059815218359184 if selected == "large_angle" else 9.207984804305022)
    assert lenses["adapter_lens"].percent == pytest.approx(
        79.80694335935594 if selected == "large_angle" else 10.)
    assert lenses["objective_lens"].percent == pytest.approx(68.9801)
    assert state.descan_deflector._physical_host is state.image_diffraction_deflector
    assert (state.descan_deflector.upper_z_mm, state.descan_deflector.lower_z_mm) == pytest.approx((1674.4, 1699.4))
    assert (state.selected_area_aperture.z_mm, state.fluorescent_screen.z_mm) == pytest.approx((1824.4, 2714.15))


def test_linear_cli_defaults_to_normal_and_rejects_unknown_range(protocols, monkeypatch):
    study, _ = protocols
    calls = []
    monkeypatch.setattr(study, "run", calls.append)
    monkeypatch.setattr("sys.argv", ["study_descan_phase_space.py"])
    study.main()
    assert calls[0].nanoprobe_range == "normal"
    monkeypatch.setattr("sys.argv", ["study_descan_phase_space.py", "--nanoprobe-range", "large"])
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and len(calls) == 1


def test_cross_range_output_is_rejected_before_any_calibration_or_overwrite(protocols, tmp_path, monkeypatch):
    study, _ = protocols
    receipt = tmp_path / "study.json"
    receipt.write_text(json.dumps({"nanoprobe_range": "normal", "existing": "keep"}), encoding="utf-8")
    before = receipt.read_bytes()
    monkeypatch.setattr(study, "_configured_state", lambda *_: pytest.fail("Mode conflict must fail before setup"))
    with pytest.raises(ValueError, match="different Nanoprobe range"):
        study.run(SimpleNamespace(output=tmp_path, nanoprobe_range="large_angle"))
    assert receipt.read_bytes() == before


def test_normal_held_comparison_requires_large_angle_even_for_direct_run(protocols, tmp_path, monkeypatch):
    study, _ = protocols
    monkeypatch.setattr(study, "_configured_state", lambda *_: pytest.fail("Invalid protocol must fail before setup"))
    with pytest.raises(ValueError, match="requires the Large Angle"):
        study.run(SimpleNamespace(output=tmp_path, nanoprobe_range="normal", compare_normal_held=True))
    assert not (tmp_path / "study.json").exists()


@pytest.mark.parametrize("selected", (None, "normal", "large_angle"))
def test_particle_protocol_applies_requested_range_before_calibration(protocols, monkeypatch, selected):
    _, transport = protocols
    from temsim.physics import scan_geometry

    observed = []
    def calibrate(state, **kwargs):
        c3 = next(lens for lens in state.lenses if lens.key == "condenser_lens_3")
        observed.append((state.nanoprobe_range, c3.percent,
                         state.descan_deflector.descan_constraint))
        return np.eye(2) * .001, 0., None
    monkeypatch.setattr(scan_geometry, "calibrate_scan_system", calibrate)
    args = SimpleNamespace(backend="cpu", rays=32, fov_nm=1000., descan_constraint="position_and_angle")
    if selected is not None:
        args.nanoprobe_range = selected
    state, calibration = transport.prepare_state(args)
    assert observed == [(selected or "normal", pytest.approx(
        46.059815218359184 if selected == "large_angle" else 9.207984804305022), "position_and_angle")]
    assert state.ac_deflector.scan_field_of_view_x_nm == 1000.
    assert state.ac_deflector.scan_field_of_view_y_nm == 1000.
    np.testing.assert_array_equal(calibration["ac_command_matrix_mrad"], np.eye(2) * .001)


@pytest.mark.parametrize("selected", ("normal", "large_angle"))
def test_prepared_particle_state_roundtrips_actual_profile_without_synthetic_controls(protocols, monkeypatch, tmp_path, selected):
    study, transport = protocols
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.physics import scan_geometry
    from temsim.profile_io import apply_profile_values, read_profile, save_profile

    monkeypatch.setattr(scan_geometry, "calibrate_scan_system", lambda *_a, **_k: (np.eye(2)*.001, 0., None))
    state, _ = transport.prepare_state(SimpleNamespace(backend="cpu", rays=32,
        fov_nm=1000., descan_constraint="position_and_angle", nanoprobe_range=selected))
    assert "wobble_enabled" not in vars(state.descan_deflector)
    catalog = AssemblyCatalog()
    path = tmp_path / f"{selected}.toml"
    save_profile(path, state, catalog.selection_for_resolved(state._resolved_assembly))
    selection, values = read_profile(path)
    assert "wobble_enabled" not in values["descan_deflector"]
    loaded, _ = study._configured_state()
    catalog.apply(loaded, selection)
    apply_profile_values(loaded, values)
    apply_physical_layout_to_state(loaded, preserve_operating_parameters=True)
    assert loaded.nanoprobe_range == selected
    assert loaded.descan_deflector.descan_constraint == "position_and_angle"
    assert loaded.ac_deflector.scan_field_of_view_x_nm == 1000.
    np.testing.assert_allclose([l.percent for l in loaded.lenses], [l.percent for l in state.lenses], rtol=0, atol=0)
    assert loaded.descan_deflector._physical_host is loaded.image_diffraction_deflector


def _candidate(nanoprobe_range):
    value = dict(upper_z_mm=1674.4, lower_z_mm=1699.4,
        upper_kick_matrix_mrad=[[.1, .01], [-.01, .1]],
        lower_kick_matrix_mrad=[[-.07, 0.], [0., -.07]])
    if nanoprobe_range is not None:
        value["nanoprobe_range"] = nanoprobe_range
    return value


@pytest.mark.parametrize("selected, saved", (("large_angle", "normal"), ("normal", "large_angle"), ("large_angle", None)))
def test_cross_range_candidate_is_rejected_including_legacy_normal(protocols, tmp_path, selected, saved):
    _, transport = protocols
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(_candidate(saved)), encoding="utf-8")
    before = path.read_bytes()
    args = SimpleNamespace(study=None, candidate=path, compare_constraints=False)
    with pytest.raises(ValueError, match="different Nanoprobe range"):
        transport.selected_cases(args, SimpleNamespace(nanoprobe_range=selected), {})
    assert path.read_bytes() == before


@pytest.mark.parametrize("selected", ("normal", "large_angle"))
def test_matching_candidate_keeps_explicit_mode_and_physical_commands(protocols, tmp_path, selected):
    _, transport = protocols
    path = tmp_path / "candidate.json"
    data = _candidate(selected)
    path.write_text(json.dumps(data), encoding="utf-8")
    result = transport.selected_cases(SimpleNamespace(study=None, candidate=path),
                                     SimpleNamespace(nanoprobe_range=selected), {})
    assert result == [("candidate", data)]


@pytest.mark.parametrize("selected, saved", (("large_angle", "normal"), ("normal", "large_angle")))
def test_cross_range_study_is_rejected_before_consuming_its_ac_matrices(protocols, tmp_path, selected, saved):
    _, transport = protocols
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"nanoprobe_range": saved}), encoding="utf-8")
    calibration = {"ac_command_matrix_mrad": [[.1, 0.], [0., .1]]}
    before = deepcopy(calibration)
    with pytest.raises(ValueError, match="different Nanoprobe range"):
        transport.selected_cases(SimpleNamespace(study=path, candidate=None),
                                 SimpleNamespace(nanoprobe_range=selected), calibration)
    assert calibration == before


@pytest.mark.parametrize("selected, saved", (("large_angle", "normal"), ("normal", "large_angle")))
def test_cross_range_frozen_prefix_is_rejected_before_optics_or_particle_loading(protocols, tmp_path, monkeypatch, selected, saved):
    _, transport = protocols
    source = tmp_path / "source"
    source.mkdir()
    (source / "report.json").write_text(json.dumps({"execution_complete": True, "nanoprobe_range": saved}), encoding="utf-8")
    monkeypatch.setattr(transport, "prepare_state", lambda *_: pytest.fail("Cross-range prefix cannot set up transport"))
    with pytest.raises(ValueError, match="different Nanoprobe range"):
        transport.production_replay_from_saved_prefix(source, tmp_path / "replay", backend="cpu", nanoprobe_range=selected)
    assert not (tmp_path / "replay").exists()


def test_held_comparison_restores_actual_normal_record_without_recalibrating_changed_state(protocols, monkeypatch):
    study, _ = protocols
    from temsim.physics import scan_geometry
    from temsim.physics.scan_calibration import capture_record

    calibrated, measured = [], []
    def calibrate(state, *, force, hold):
        assert force and hold and state.ac_deflector.calibration_mode == "automatic"
        calibrated.append(state.nanoprobe_range)
        factor = 1. if state.nanoprobe_range == "normal" else 1.01
        ac, ds = state.ac_deflector, state.descan_deflector
        ac.set_pure_shift_coupling([[-1.1, .02], [-.03, -.9]])
        ac.set_scan_command_matrix_mrad(np.diag((.001, .0012)) * factor)
        ds.set_image_plane_coupling([[-.7, .1], [-.1, -.7]],
            target_key=state.selected_area_aperture.key, target_z_mm=state.selected_area_aperture.z_mm)
        ds.set_scan_command_matrix_mrad([[.002, .0003], [-.0002, .003]])
        ds._calibrated_constraint = "position_and_angle"
        capture_record(state, descan_calibrated=True, ac_angle_target="canonical")
        ac.calibration_mode = "held"

    def measure(state, targets, fov):
        assert fov == state.ac_deflector.scan_field_of_view_x_nm == 1000.
        measured.append((state.nanoprobe_range, state.ac_deflector.calibration_record_json,
                         study._accepted_drives(state)))
        return {"test_measurement_range": state.nanoprobe_range}

    monkeypatch.setattr(scan_geometry, "calibrate_scan_system", calibrate)
    monkeypatch.setattr(study, "_measure_accepted_drives", measure)
    result = study._normal_held_comparison(1000.)
    assert calibrated == ["normal", "large_angle"]
    assert [item[0] for item in measured] == ["normal", "large_angle", "large_angle"]
    assert measured[0][1:] == measured[1][1:]
    assert measured[1][2] != measured[2][2]
    assert result["normal_record_and_drives_unchanged_after_mode_switch_and_measurement"]
    assert result["normal_accepted_record"]["ac_angle_target"] == "canonical"
    assert result["normal_accepted_record"]["descan_constraint"] == "position_and_angle"
    assert result["fresh_versus_normal_logical_command_relative_change"]["ac_deflector"] == pytest.approx(.01)


@pytest.mark.parametrize("singular", (False, True))
def test_virtual_backward_pivot_is_a_dependent_optical_continuation_with_condition_guard(protocols, monkeypatch, singular):
    study, _ = protocols
    from temsim.physics import finite_scan_response

    identity, zero = np.eye(2), np.zeros((2, 2))
    target = np.block([[.05*identity, .025*identity], [identity, identity]])
    root = np.block([[.02*identity, .02*identity], [identity, zero]])
    sample = np.block([[.5*identity, zero], [zero, identity]])
    ac = SimpleNamespace(upper_z_mm=-100., lower_z_mm=-75.)
    ds = SimpleNamespace()
    state = SimpleNamespace(ac_deflector=ac, descan_deflector=ds, sample=SimpleNamespace(z_mm=0.))

    def finite(_state, component, zs, **_kwargs):
        rows = []
        for z in zs:
            value = {0.: sample, 10.: root, 30.: target}[float(z)].copy()
            if component is ds and z < 20.:
                value[:] = 0.
            if singular and component is ac and z == 30.:
                value[:, 2:] = value[:, :2]
            rows.append(value)
        rows = np.asarray(rows)
        return finite_scan_response.FiniteScanResponse(np.asarray(zs), rows[:, :, :2], rows[:, :, 2:],
            np.zeros((len(zs), 4)), -101., ((18., 22.), (23., 27.)), "fixture", 1e-6)
    monkeypatch.setattr(finite_scan_response, "finite_scan_responses", finite)
    drives = (identity*1e-6, -identity*1e-6)
    candidate = {"centres_mm": [20., 25.], "targets": {"selected_area_aperture": {
        "solutions": {"full_phase_space": {
            "physical_upper_drive_basis_mrad": -identity*.001,
            "physical_lower_drive_basis_mrad": identity*.001}}}}}
    pivots = {"search": {"is_common_pivot": True, "relative_tolerance": 1e-6},
              "selected_and_other_candidates": [{"z_mm": 10.}]}
    if singular:
        with pytest.raises(ValueError, match="singular or ill-conditioned"):
            study._descan_backward_equivalent_pivots(state, drives,
                {"selected_area_aperture": 30.}, candidate, pivots)
        return
    result = study._descan_backward_equivalent_pivots(state, drives,
        {"selected_area_aperture": 30.}, candidate, pivots)
    assert result["verified"] and "not independent validation" in result["dependence"]
    row = result["observations"][0]
    assert row["physical_descan_upstream_response_is_zero"]
    assert row["virtual_plus_ac_position_residual_nm"] < 1e-12
    assert row["virtual_plus_ac_slope_residual_urad"] < 1e-12
    np.testing.assert_allclose(row["virtual_mechanical_slope_basis_mrad"], -identity*.001, atol=1e-16)


@pytest.fixture
def sealed_prefix(protocols, tmp_path):
    from temsim.instrument_snapshot import capture_instrument_snapshot

    study, _ = protocols
    provenance = importlib.import_module("descan_prefix_provenance")
    state, _ = study._configured_state("large_angle")
    state.ac_deflector.scan_enabled = state.descan_deflector.scan_enabled = False
    arrays = dict(ray_id=np.arange(3, dtype=np.int64),
        x_m=np.array([0., 1e-6, -2e-6]), y_m=np.array([1e-6, 0., -1e-6]),
        tx_rad=np.array([0., 1e-3, -1e-3]), ty_rad=np.array([1e-3, 0., -1e-3]),
        weights=np.array([1., .5, .25]), energy_offset_ev=np.array([0., .1, -.1]),
        alive=np.array([True, False, True]), prefix_z_mm=np.array(1379.999),
        exit_bundle_sha256=np.array("a" * 64))
    path = tmp_path / "prefix.npz"
    np.savez_compressed(path, **arrays)
    receipt = provenance.create_prefix_provenance(state, path,
        capture_kind="contemporaneous_prefix_inputs",
        source_snapshot_digest=capture_instrument_snapshot(state).digest)
    provenance.verify_prefix_provenance(receipt, state, path)
    return provenance, state, path, arrays, receipt


@pytest.mark.parametrize("field", ("x_m", "y_m", "tx_rad", "ty_rad", "weights",
    "energy_offset_ev", "alive", "ray_id", "prefix_z_mm", "exit_bundle_sha256"))
def test_sealed_prefix_rejects_numeric_and_source_identity_mutation(sealed_prefix, field):
    provenance, state, path, arrays, receipt = sealed_prefix
    modified = {key: value.copy() for key, value in arrays.items()}
    if field == "alive":
        modified[field][0] = False
    elif field == "ray_id":
        modified[field][0] = 100
    elif field == "prefix_z_mm":
        modified[field] = modified[field] + .001
    elif field == "exit_bundle_sha256":
        modified[field] = np.array("b" * 64)
    else:
        modified[field][0] += .001
    np.savez_compressed(path, **modified)
    with pytest.raises(ValueError):
        provenance.verify_prefix_provenance(receipt, state, path)


@pytest.mark.parametrize("change", ("c2_aperture", "lens_profile"))
def test_sealed_prefix_rejects_upstream_physics_changes_even_when_lens_percent_and_z_are_unchanged(sealed_prefix, change):
    provenance, state, path, _, receipt = sealed_prefix
    before_lenses = [(lens.percent, lens.z_mm) for lens in state.lenses]
    if change == "c2_aperture":
        state.condenser_aperture_2.radius_mm *= .9
    else:
        next(lens for lens in state.lenses if lens.key == "condenser_lens_2").a_mm *= 1.01
    assert [(lens.percent, lens.z_mm) for lens in state.lenses] == before_lenses
    with pytest.raises(ValueError, match="[Uu]pstream|signature|physics"):
        provenance.verify_prefix_provenance(receipt, state, path)


def test_sealed_prefix_allows_changes_wholly_downstream_of_its_plane(sealed_prefix):
    provenance, state, path, arrays, receipt = sealed_prefix
    digest = provenance.prefix_numeric_digest(path)
    assert digest == provenance.prefix_numeric_digest(arrays)
    state.image_diffraction_deflector.mechanical_center_below_sample_mm += 1.
    state.image_diffraction_deflector.upper_x_mrad = .02
    state.descan_deflector.descan_constraint = "position_and_angle"
    state.descan_deflector.set_scan_command_matrix_mrad([[.001, .0001], [-.0001, .002]])
    provenance.verify_prefix_provenance(receipt, state, path)
    assert provenance.prefix_numeric_digest(path) == digest


def test_sealed_prefix_rejects_tampered_receipt_metadata(sealed_prefix):
    provenance, state, path, _, receipt = sealed_prefix
    altered = deepcopy(receipt)
    altered["source_snapshot_digest"] = "b" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        provenance.verify_prefix_provenance(altered, state, path)


def test_reconstructed_prefix_cannot_claim_verification_without_independent_replay(sealed_prefix):
    provenance, state, path, _, receipt = sealed_prefix
    with pytest.raises(ValueError, match="independent saved-exit replay evidence"):
        provenance.create_prefix_provenance(state, path,
            capture_kind="reconstructed_prefix_inputs_verified_by_replay",
            source_snapshot_digest=receipt["source_snapshot_digest"], evidence={})


def test_prefix_physics_identity_allows_backend_and_tail_step_changes(sealed_prefix):
    provenance, state, path, _, receipt = sealed_prefix
    state.acceleration_enabled = True
    state.acceleration_backend = "Require GPU"
    state.step_mm = .1
    state.history_step_mm = .2
    provenance.verify_prefix_provenance(receipt, state, path)
