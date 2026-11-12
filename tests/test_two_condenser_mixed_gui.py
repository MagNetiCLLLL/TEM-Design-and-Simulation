"""Accepted C2 inputs, GUI ownership and geometry only; no physical transport."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.paths import BUNDLED_SI_CIF
from temsim.profile_io import apply_profile_values, read_profile
from temsim.specimen.cif_io import read_cif_atoms
from temsim.specimen.geometry import quaternion_from_zone_axes
from test_working_point_restore_gui import window


ACCEPTED_C2 = (Path(__file__).resolve().parents[1]/"docs/development/"
    "two_condenser_10mrad_20261112/projector/weak-c2-final-exact/final-exact-input.toml")


@pytest.fixture
def c2_state():
    selection, values = read_profile(ACCEPTED_C2)
    state = default_state()
    AssemblyCatalog().apply(state, selection)
    apply_profile_values(state, values)
    state.electron_gun.emitter.ray_count = 16384
    assert state.sample.thickness_nm == 5.
    assert state.electron_gun.emitter.energy_spread_fwhm_ev == .3
    assert state.step_mm == .05
    return state


def optical_controls(state):
    return {item.key: (item.percent, item.enabled, item.z_mm, getattr(item, "polarity", None))
            for item in state.lenses}


def assert_c2_hardware(state):
    assert state.layout_c3_hardware == "two_condenser"
    parts = {part.key for part in state._resolved_assembly.parts}
    assert {"condenser_lens_1", "condenser_lens_2", "mini_condenser", "objective_lens"} <= parts
    assert not {"condenser_lens_3", "adapter_lens", "condenser_aperture_3"} & parts
    assert not any(key.startswith(("probe_hp", "probe_tl")) for key in parts)
    assert all(not lens.enabled for lens in state.lenses
               if lens.key in {"condenser_lens_3", "adapter_lens"})


def test_c2_mixed_request_preserves_accepted_controls_source_and_numerical_budget(window,c2_state,monkeypatch):
    before = capture_instrument_snapshot(c2_state).digest
    controls = optical_controls(c2_state)
    window._install_working_point(c2_state, None)
    assert window.selection.column == "C2"
    assert window.assembly_panel.nanoprobe_range_combo.isHidden()
    assert "C3 " not in window.assembly_panel.operating_mode_status.text()
    tuning = window.workspace.hardware_tuning
    tuning.select_task("condenser_current")
    assert ("condenser_lens_3", "percent") not in tuning.editors
    tuning.select_task("probe_corrector")
    assert not any(key == "adapter_lens" or key.startswith(("probe_hp", "probe_tl"))
                   for key, _name in tuning.editors)
    page = window.workspace.coherent_beam
    page._source_controls["source_image_rms_pm"].setValue(12.)
    shared = page._make_request(c2_state).source
    request = window._mixed_stem_request()
    assert asdict(request.source) == asdict(shared)
    assert request.source.maximum_working_bytes == 24*1024**3
    assert request.maximum_working_bytes == 24*1024**3
    calls = []
    monkeypatch.setattr(window.mixed_stem, "start", lambda *a, **k: calls.append((a,k)))
    window.workspace.scan_control.stem_scan_method.setCurrentIndex(1)
    window.run_page_calculation("stem")
    assert len(calls) == 1
    executed = calls[0][0][0]
    assert capture_instrument_snapshot(executed).digest == before
    assert optical_controls(executed) == controls
    assert executed.electron_gun.emitter.ray_count == 16384
    assert executed.electron_gun.emitter.energy_spread_fwhm_ev == .3
    assert executed.step_mm == .05 and not executed.sample.stem_wave_enabled
    assert_c2_hardware(executed)


def test_c2_coherent_capture_uses_the_accepted_snapshot_without_missing_hardware_requirements(window,c2_state):
    window._install_working_point(c2_state, None)
    page = window.workspace.coherent_beam
    before = capture_instrument_snapshot(c2_state)
    upper = c2_state.sample.upper_surface_z_mm
    captured, request = page._capture_particle_inputs(
        SimpleNamespace(state_snapshot=c2_state), upper, for_execution=False)
    assert capture_instrument_snapshot(captured).digest == before.digest
    assert request.wave_grid.compute_backend == "Require GPU"
    assert request.observation_z_mm == upper
    assert_c2_hardware(captured)
    with pytest.raises(ValueError, match="specimen entrance"):
        page._capture_particle_inputs(SimpleNamespace(state_snapshot=c2_state),
                                     upper-.001, for_execution=False)


def test_c2_sample_page_five_to_ten_nm_keeps_upper_fields_controls_and_snapshot(qtbot,c2_state):
    from temsim.gui.sample_panel import SamplePage
    from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
    from temsim.physics.mixed_stem_scan import preflight_mixed_stem_material
    state = c2_state
    state.sample.cif_path = str(BUNDLED_SI_CIF)
    state.sample.specimen_mode = "atomic"
    state.sample.inserted = True
    state.sample.specimen_orientation_quaternion_wxyz = quaternion_from_zone_axes(
        read_cif_atoms(BUNDLED_SI_CIF).cell.array, (1,1,0), (1,-1,0))
    state.sample.zone_axis_uvw = (1,1,0)
    state.sample.in_plane_axis_uvw = (1,-1,0)
    before = optical_controls(state)
    upper = state.sample.upper_surface_z_mm
    points = np.array([[0.,0.,upper*.001],[1e-9,0.,upper*.001]])
    field = capture_instrument_magnetic_field(state).field_at_global_positions_t(points)
    page = SamplePage()
    qtbot.addWidget(page)
    page.set_state(state)
    errors = []
    page.error.connect(errors.append)
    page.scalar_controls["thickness_nm"].setValue(10.)
    assert not errors
    assert state.sample.thickness_nm == 10.
    assert state.sample.upper_surface_z_mm == upper
    assert state.sample.lower_surface_z_mm == upper+10.e-6
    assert state.sample.lattice_reference_thickness_nm == 5.
    assert optical_controls(state) == before
    np.testing.assert_array_equal(
        capture_instrument_magnetic_field(state).field_at_global_positions_t(points), field)
    preflight_mixed_stem_material(state)
    captured = capture_instrument_snapshot(state)
    restored = captured.restore()
    assert capture_instrument_snapshot(restored).digest == captured.digest
    assert restored.sample.upper_surface_z_mm == upper
    assert restored.sample.thickness_nm == 10.
    assert optical_controls(restored) == before
    assert_c2_hardware(restored)


def test_desktop_run_uses_ordinary_startup_without_an_implicit_working_point(monkeypatch):
    from temsim import app
    calls = []
    monkeypatch.setattr(app, "create_application", lambda _: SimpleNamespace(exec=lambda: 0))
    monkeypatch.setattr(app, "MainWindow", lambda *a, **k:
                        (calls.append((a,k)) or SimpleNamespace(show=lambda: None)))
    assert app.run([]) == 0
    assert calls == [((),{})]


def test_c2_manual_objective_focus_does_not_reapply_other_lenses_or_move_the_surface(window,c2_state,monkeypatch):
    from temsim.specimen.axial_geometry import set_sample_thickness
    set_sample_thickness(c2_state, 10.)
    window._install_working_point(c2_state, None)
    monkeypatch.setattr(window, "schedule_preview", lambda *_a, **_k: None)
    before = optical_controls(c2_state)
    upper = c2_state.sample.upper_surface_z_mm
    tuning = window.workspace.hardware_tuning
    tuning.select_task("stem_focus")
    new_percent = c2_state.objective_lens.percent+.01
    editor = tuning.editors[("objective_lens", "percent")]
    editor.setText(str(new_percent))
    editor.editingFinished.emit()
    assert c2_state.objective_lens.percent == new_percent
    after = optical_controls(c2_state)
    assert {key: value for key, value in after.items() if key != "objective_lens"} == {
        key: value for key, value in before.items() if key != "objective_lens"}
    assert after["objective_lens"][1:] == before["objective_lens"][1:]
    assert c2_state.sample.upper_surface_z_mm == upper
    assert c2_state.sample.thickness_nm == 10.
    assert c2_state.electron_gun.emitter.ray_count == 16384
    assert c2_state.electron_gun.emitter.energy_spread_fwhm_ev == .3
    assert_c2_hardware(c2_state)
