"""Unit assembly and draft transactions; no ray/wave calculation required."""
from dataclasses import replace
from itertools import product
import math

import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.component_keys import BCOR_EXTRA_KEYS, IMAGE_CORRECTOR_KEYS, PROBE_CORRECTOR_KEYS
from temsim.instrument_configuration import InstrumentUnits, check_instrument_configuration, unit_for_component
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state


@pytest.fixture
def state():
    return default_state()


def test_all_catalog_combinations_install_selected_hardware_and_carry_specimen(state):
    catalog = AssemblyCatalog()
    sample = state.sample
    thickness = sample.thickness_nm
    offset = sample.upper_surface_z_mm - state._resolved_assembly.part("sample_holder").center_z_mm
    for gun, column, blanker, recording in product(
        catalog.guns, catalog.columns, catalog.beam_blankers, catalog.recording_systems
    ):
        selection = AssemblySelection(gun.name, column.name, recording.name, blanker.name)
        units = InstrumentUnits.from_selection(catalog, selection)
        assert units.selection(catalog) == selection
        assembly = catalog.apply(state, selection, preserve_operating_parameters=True)
        assert catalog.selection_for_resolved(assembly) == selection
        assert dict(assembly.selected_module_paths) == catalog.selected_paths(selection)
        assert state.electron_gun.type_key == units.source
        assert state.monochromator_installed == units.monochromator
        assert state.nanopulser.installed == units.beam_blanker
        assert state.probe_corrector_installed == units.probe_corrector
        assert state.image_corrector_installed == units.image_corrector
        assert state.energy_filter_installed == units.energy_filter
        keys = {p.key for p in assembly.parts}
        assert ("condenser_lens_3" in keys) == units.c3_lens
        assert ("energy_filter" in keys) == units.energy_filter
        assert {"sample", "objective_lens", "camera", "haadf", "projector_lens_2"} <= keys
        for elements, installed in (
            (PROBE_CORRECTOR_KEYS, units.probe_corrector),
            (IMAGE_CORRECTOR_KEYS, units.image_corrector),
            (BCOR_EXTRA_KEYS, units.image_corrector and units.image_corrector_design == "bcor"),
        ):
            assert keys.intersection(elements) == (set(elements) if installed else set()), selection

        holder = assembly.part("sample_holder")
        stage = assembly.part("sample_stage")
        half_tip = holder.data["holder_tip_thickness_mm"] / 2
        first = max(holder.center_z_mm - half_tip, stage.start_z_mm, assembly.part("objective_upper_pole").end_z_mm)
        last = min(holder.center_z_mm + half_tip, stage.end_z_mm, assembly.part("objective_lower_pole").start_z_mm)
        sample = state.sample
        assert first <= sample.upper_surface_z_mm < sample.lower_surface_z_mm <= last, selection
        assert sample.thickness_nm == thickness
        assert sample.upper_surface_z_mm - holder.center_z_mm == pytest.approx(
            offset, rel=0, abs=4 * math.ulp(holder.center_z_mm)), selection


@pytest.mark.parametrize("units,reason", [
    (InstrumentUnits(source="thermionic", monochromator=True), "requires Cold FEG"),
    (InstrumentUnits(c3_lens=False), "require C3"),
    (InstrumentUnits(c3_lens=False, probe_corrector=False, image_corrector=True), "require C3"),
    (InstrumentUnits(energy_filter="false"), "explicit on/off"),
])
def test_invalid_units_are_rejected_before_mutation(state, units, reason):
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match=reason):
        check_instrument_configuration(state, AssemblyCatalog(), units)
    assert capture_instrument_snapshot(state).digest == before


def test_check_is_read_only_preserves_shared_controls_and_does_not_calculate(state, monkeypatch):
    catalog = AssemblyCatalog()
    state.electron_gun.emitter.curvature_nm_inv = 1e-8
    state.objective_lens.percent = 64.123
    before = capture_instrument_snapshot(state).digest
    monkeypatch.setattr("temsim.physics.simulation.run", lambda *a, **kw: pytest.fail("Check propagated electrons"))
    monkeypatch.setattr("temsim.operating_modes.apply_operating_mode_pair", lambda *a, **kw: pytest.fail("Check applied a preset"))
    checked = check_instrument_configuration(state, catalog, InstrumentUnits(energy_filter=False))
    assert capture_instrument_snapshot(state).digest == before
    candidate = checked.restore_for(state)
    assert candidate.energy_filter_installed is False
    assert candidate.electron_gun.emitter.curvature_nm_inv == 1e-8
    assert candidate.objective_lens.percent == 64.123
    assert candidate is not state
    original_parts = {p.key: p for p in state._resolved_assembly.parts}
    for part in candidate._resolved_assembly.parts:
        if part.key in original_parts:
            assert part.center_z_mm == original_parts[part.key].center_z_mm


def test_changed_column_carries_a_placed_specimen_and_fixed_entrance_in_both_directions(state):
    from temsim.specimen.axial_geometry import set_sample_thickness

    catalog = AssemblyCatalog()
    original = InstrumentUnits.from_selection(catalog, catalog.selection_for_resolved(state._resolved_assembly))
    state.sample.z_mm += 0.025
    set_sample_thickness(state, 17.125)
    upper = state.sample.upper_surface_z_mm
    holder_z = state._resolved_assembly.part("sample_holder").center_z_mm
    mount_offset = upper - holder_z
    lattice_thickness = state.sample.lattice_reference_thickness_nm
    state.objective_lens.percent = 64.125

    for units in (replace(original, c3_lens=False, probe_corrector=False), original):
        checked = check_instrument_configuration(state, catalog, units)
        state = checked.restore_for(state)
        assembly = state._resolved_assembly
        sample = state.sample
        tolerance = 4 * math.ulp(sample.z_mm)
        assert sample.upper_surface_z_mm - assembly.part("sample_holder").center_z_mm == pytest.approx(
            mount_offset, rel=0, abs=tolerance)
        assert sample.thickness_nm == 17.125 and sample.lattice_reference_thickness_nm == lattice_thickness
        assert sample.lower_surface_z_mm == sample.upper_surface_z_mm + 17.125e-6
        assert assembly.part("sample").center_z_mm == sample.z_mm
        assert state.objective_lens.percent == 64.125
    assert state.sample.upper_surface_z_mm == upper


@pytest.mark.parametrize(("start_gun", "start_blanker", "end_gun", "end_blanker"), [
    ("FEG", "None", "Thermionic", "Electrostatic beam blanker"),
    ("Thermionic", "Electrostatic beam blanker", "FEG + Mono", "None"),
])
def test_changing_gun_and_blanker_together_installs_the_requested_modules(
    state, start_gun, start_blanker, end_gun, end_blanker
):
    catalog = AssemblyCatalog()
    start = replace(catalog.default_selection(), gun=start_gun, beam_blanker=start_blanker)
    catalog.apply(state, replace(start, beam_blanker="None"))
    catalog.apply(state, start)
    assert catalog.selection_for_resolved(state._resolved_assembly) == start
    target = replace(start, gun=end_gun, beam_blanker=end_blanker)
    units = InstrumentUnits.from_selection(catalog, target)
    state.nanopulser.voltage_v = 417.0
    checked = check_instrument_configuration(state, catalog, units)
    installed = checked.restore_for(state)
    assert catalog.selection_for_resolved(installed._resolved_assembly) == target
    assert installed.nanopulser.installed == units.beam_blanker
    assert installed.nanopulser.voltage_v == 417.0
    assert catalog.selection_for_resolved(state._resolved_assembly) == start


def test_draft_rejects_a_different_resolved_module_before_installing(state, monkeypatch):
    catalog = AssemblyCatalog()
    original = capture_instrument_snapshot(state).physical_digest
    apply = catalog.apply

    def omit_blanker(candidate, selection, **kwargs):
        return apply(candidate, replace(selection, beam_blanker="None"), **kwargs)

    monkeypatch.setattr(catalog, "apply", omit_blanker)
    with pytest.raises(ValueError, match="differ from the requested modules"):
        check_instrument_configuration(state, catalog, InstrumentUnits(beam_blanker=True))
    assert capture_instrument_snapshot(state).physical_digest == original


def test_changed_column_rejects_a_foil_whose_lower_face_exceeds_the_new_holder(state, tmp_path):
    from temsim.part_model_document import PartModelDocument
    from temsim.shared_tip import copy_catalog_tree
    from temsim.specimen.axial_geometry import set_sample_thickness

    # Both faces fit the installed 1.2 mm tip. Only the downstream face
    # exceeds the 0.4 mm half-thickness of the replacement tip.
    state.sample.z_mm += 0.38
    set_sample_thickness(state, 40000.0)
    before = capture_instrument_snapshot(state).physical_digest
    root = tmp_path / "instruments"
    copy_catalog_tree(state._resolved_assembly.root, root)
    draft = PartModelDocument(root / "column/C2.toml")
    draft.set_dimension(("parts", "sample_holder", "holder_tip_thickness_mm"), 0.8)
    draft.save()
    units = InstrumentUnits(c3_lens=False, probe_corrector=False, energy_filter=False)
    with pytest.raises(ValueError, match="complete specimen thickness.*allowed Z"):
        check_instrument_configuration(state, AssemblyCatalog(root), units)
    assert capture_instrument_snapshot(state).physical_digest == before


def test_unchanged_check_never_resets_the_instrument_and_stale_check_rejected(state):
    state.objective_lens.percent = 61.25
    catalog = AssemblyCatalog()
    units = InstrumentUnits.from_selection(catalog, catalog.selection_for_resolved(state._resolved_assembly))
    checked = check_instrument_configuration(state, catalog, units)
    assert checked.original.digest == checked.candidate.digest
    state.objective_lens.percent += .01
    with pytest.raises(ValueError, match="changed"):
        checked.restore_for(state)


def test_units_include_mechanical_children_and_filter_entrance(state):
    from dataclasses import replace
    catalog = AssemblyCatalog()
    a = catalog.apply(state, replace(catalog.default_selection(), recording="Energy Filter"))
    assert unit_for_component(a, "condenser_lens_3_excitation_coil") == "c3_lens"
    assert unit_for_component(a, "feg_tip") == "source"
    assert unit_for_component(a, "energy_filter_entrance_aperture") == "energy_filter"
    assert unit_for_component(a, "sample") is None


def _dialog(qtbot, state, tmp_path, on_assemble=None):
    from PySide6.QtCore import QSettings, Qt
    from temsim.gui.instrument_configuration_dialog import InstrumentConfigurationDialog
    dialog = InstrumentConfigurationDialog(AssemblyCatalog(), lambda: state,
        on_assemble or (lambda checked: None), settings=QSettings(str(tmp_path/"ui.ini"), QSettings.Format.IniFormat))
    # Successful Assemble closes the dialog; pytest-qt still owns its wrapper
    # and closes it during teardown.
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(dialog)
    return dialog


def test_dialog_dependencies_draft_and_linked_review(qtbot, state, tmp_path):
    d = _dialog(qtbot, state, tmp_path)
    before = capture_instrument_snapshot(state).digest
    assembly = d._assembly
    assert d.assemble_button.isEnabled()
    assert not hasattr(d, "check_button") and not hasattr(d, "check")
    assert d.review.model_editor is None
    d.options["c3_lens"].setChecked(False)
    for key in ("probe_corrector", "image_corrector"):
        assert not d.options[key].isEnabled() and not d.options[key].isChecked()
    d.options["monochromator"].setChecked(True)
    d.source.setCurrentIndex(d.source.findData("thermionic"))
    assert not d.options["monochromator"].isEnabled()
    assert not d.options["monochromator"].isChecked()
    d.options["energy_filter"].setChecked(False)
    assert d.assemble_button.isEnabled()
    assert d._assembly is assembly  # Read-only review retains the installed unit.
    assert capture_instrument_snapshot(state).digest == before
    d.select_component("feg_tip")
    assert d.table.currentRow() == 0
    d.options["c3_lens"].setChecked(True)
    assert d.options["probe_corrector"].isEnabled()
    assert d.assemble_button.isEnabled()
    assert "draft changed" in d.preview_status.text().lower()


def test_dialog_single_click_validates_current_state_and_applies_once(qtbot, state, tmp_path):
    from PySide6.QtWidgets import QDialog
    applied = []

    def apply(checked):
        assert not d.assemble_button.isEnabled()
        d.assemble()  # A nested event cannot apply the same draft again.
        applied.append(checked.restore_for(state))

    d = _dialog(qtbot, state, tmp_path, apply)
    d.options["energy_filter"].setChecked(True)
    # Live edits after opening are captured at this click, not rejected against
    # an obsolete separately checked draft.
    state.objective_lens.percent += .125
    before = capture_instrument_snapshot(state).digest
    d.assemble_button.click()
    assert len(applied) == 1
    assert applied[0].energy_filter_installed
    assert applied[0].objective_lens.percent == state.objective_lens.percent
    assert capture_instrument_snapshot(state).digest == before
    assert d.result() == QDialog.DialogCode.Accepted


def test_dialog_validation_failure_does_not_apply_and_can_retry(qtbot, state, tmp_path, monkeypatch):
    applied = []
    d = _dialog(qtbot, state, tmp_path, applied.append)
    before = capture_instrument_snapshot(state).digest
    valid_units = d.units
    monkeypatch.setattr(d, "units", lambda: InstrumentUnits(source="thermionic", monochromator=True))
    d.show()
    d.assemble_button.click()
    assert not applied and d.isVisible()
    assert "requires Cold FEG" in d.status.text()
    assert d.assemble_button.isEnabled()
    assert capture_instrument_snapshot(state).digest == before
    monkeypatch.setattr(d, "units", valid_units)
    d.assemble_button.click()
    assert len(applied) == 1


def test_dialog_failed_apply_stays_open_and_can_retry(qtbot, state, tmp_path):
    attempts = []

    def apply(checked):
        candidate = checked.restore_for(state)
        attempts.append(candidate)
        if len(attempts) == 1:
            raise RuntimeError("Installation rejected")

    d = _dialog(qtbot, state, tmp_path, apply)
    before = capture_instrument_snapshot(state).digest
    d.show()
    d.assemble_button.click()
    assert "Installation rejected" in d.status.text()
    assert d.isVisible() and d.assemble_button.isEnabled()
    assert capture_instrument_snapshot(state).digest == before
    d.assemble_button.click()
    assert len(attempts) == 2


def test_dialog_layout_is_saved_separately_and_draft_does_not_apply(qtbot, state, tmp_path):
    from PySide6.QtCore import QSettings, Qt
    from temsim.gui.instrument_configuration_dialog import InstrumentConfigurationDialog
    settings = QSettings(str(tmp_path/"layout.ini"), QSettings.Format.IniFormat)
    applied = []
    d = InstrumentConfigurationDialog(AssemblyCatalog(), lambda: state, applied.append,
        settings=settings, layout_id="review")
    # Keep the wrapper alive until pytest-qt closes its registered widgets.
    d.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(d)
    # The offscreen screen is 800 px wide; restoreGeometry correctly clamps
    # windows saved larger than the current screen. Test a fitting width.
    d.resize(780, 700)
    d.show()
    qtbot.wait(20)
    d.splitter.setSizes([260, 500])
    expected = d.size()
    expected_split = d.splitter.sizes()
    d.options["energy_filter"].setChecked(not d.options["energy_filter"].isChecked())
    assert not applied
    d.reject()
    restored = InstrumentConfigurationDialog(AssemblyCatalog(), lambda: state, applied.append,
        settings=settings, layout_id="review")
    qtbot.addWidget(restored)
    restored.show()
    qtbot.wait(20)
    assert restored.size() == expected
    assert restored.splitter.sizes() == expected_split
    assert settings.value("instrument_configuration/v1/default/geometry") is None


def _window(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window as shell, interactive_calculation, instrument_configuration_dialog
    settings = QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(interactive_calculation, "QSettings", lambda: settings)
    monkeypatch.setattr(instrument_configuration_dialog, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    monkeypatch.setattr(shell.MainWindow, "_apply_state_operating_modes", lambda *_: object())
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    monkeypatch.setattr(window.calculations.pool, "start", lambda *_: pytest.fail("Configuration must not calculate"))
    return window


def test_main_window_configuration_applies_once_without_preset_or_ray_solve(qtbot, monkeypatch, tmp_path):
    window = _window(qtbot, monkeypatch, tmp_path)
    previews = []
    monkeypatch.setattr(window, "schedule_preview", lambda *a: previews.append(True))
    monkeypatch.setattr(window, "_start_operating_preset", lambda *a, **kw: pytest.fail("Automatic preset"))
    d = window.open_instrument_configuration()
    assert window.open_instrument_configuration() is d
    d.source.setCurrentIndex(d.source.findData("thermionic"))
    d.options["beam_blanker"].setChecked(True)
    d.options["energy_filter"].setChecked(True)
    d.assemble_button.click()
    assert window.state.electron_gun.type_key == "thermionic"
    assert window.state.nanopulser.installed
    assert window.assembly.selected_path("beam_blanker") == "beam_blanker/ElectrostaticBeamBlanker.toml"
    assert window.state.energy_filter_installed
    assert window.selection.recording == "Energy Filter"
    assert window.assembly_panel.current_selection() == window.selection
    assert previews == [True]
    assert window.workspace.physical_layout._result.assembly is window.assembly
    assert window._configuration_dialog is None


def test_main_window_assembles_without_c3_lens_and_probe_corrector(qtbot, monkeypatch, tmp_path):
    window = _window(qtbot, monkeypatch, tmp_path)
    monkeypatch.setattr(window, "schedule_preview", lambda: None)
    old_holder = window.assembly.part("sample_holder").center_z_mm
    old_upper = window.state.sample.upper_surface_z_mm
    dialog = window.open_instrument_configuration()
    dialog.options["probe_corrector"].setChecked(False)
    dialog.options["c3_lens"].setChecked(False)
    dialog.assemble_button.click()

    assert window._configuration_dialog is None
    assert not window.state.probe_corrector_installed
    assert window.state.layout_c3_hardware == "two_condenser"
    assert {"condenser_lens_3", "adapter_lens"}.isdisjoint(part.key for part in window.assembly.parts)
    holder = window.assembly.part("sample_holder")
    assert holder.center_z_mm < old_holder
    assert window.state.sample.upper_surface_z_mm == pytest.approx(
        old_upper + (holder.center_z_mm - old_holder), rel=0, abs=4 * math.ulp(holder.center_z_mm))
    assert window.workspace.physical_layout._result.assembly is window.assembly


def test_closing_main_window_saves_open_configuration_without_applying(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    window = _window(qtbot, monkeypatch, tmp_path)
    before = capture_instrument_snapshot(window.state).digest
    d = window.open_instrument_configuration()
    settings = QSettings(str(tmp_path / "close.ini"), QSettings.Format.IniFormat)
    d.settings = settings
    key = d.settings_key
    d.options["energy_filter"].setChecked(not d.options["energy_filter"].isChecked())
    window.close()
    assert settings.value(key + "/geometry") is not None
    assert settings.value(key + "/splitter") is not None
    assert window._configuration_dialog is None
    assert capture_instrument_snapshot(window.state).digest == before
