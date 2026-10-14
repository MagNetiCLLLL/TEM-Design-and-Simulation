"""B-COR is an explicit additional assembly, preserved across UI and IO."""
from dataclasses import replace
from itertools import product

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_configuration import InstrumentUnits, check_instrument_configuration, unit_for_component
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.model import State
from temsim.profile_io import apply_profile_values, read_profile, save_profile


def _bcor(catalog, *, probe=False, **changes):
    return InstrumentUnits(probe_corrector=probe, image_corrector=True,
        image_corrector_design="bcor", energy_filter=False, **changes).selection(catalog)


def test_catalog_adds_two_bcor_options_without_changing_default_or_cetcor_names():
    catalog = AssemblyCatalog()
    assert catalog.default_selection().column == "C3 + Probe Corrector"
    assert InstrumentUnits(image_corrector=True, probe_corrector=False).selection(catalog).column == "C3 + Image Corrector"
    assert InstrumentUnits(image_corrector=True).selection(catalog).column == "C3 + Probe Corrector + Image Corrector"
    for probe, expected in ((False, "C3 + B-COR"), (True, "C3 + Probe Corrector + B-COR")):
        for source, mono in (("cold_feg", False), ("cold_feg", True), ("thermionic", False)):
            for blanker, energy_filter in product((False, True), repeat=2):
                units = InstrumentUnits(source, mono, blanker, True, probe, True,
                    energy_filter, image_corrector_design="bcor")
                selection = units.selection(catalog)
                assert selection.column == expected
                assert InstrumentUnits.from_selection(catalog, selection) == units


@pytest.mark.parametrize("bad", ["unknown", "BCOR", None, True])
def test_unknown_image_corrector_design_is_rejected(bad):
    with pytest.raises(ValueError, match="Select CETCOR or B-COR"):
        InstrumentUnits(image_corrector=True, image_corrector_design=bad).selection(AssemblyCatalog())


@pytest.mark.parametrize("probe", [False, True])
def test_bcor_survives_gun_recording_and_optional_module_changes(probe):
    catalog = AssemblyCatalog()
    state = default_state()
    selection = _bcor(catalog, probe=probe)
    catalog.apply(state, selection)
    state.objective_lens.percent = 62.345
    for gun, recording, blanker in (("FEG + Mono", "Energy Filter", "Electrostatic beam blanker"),
                                   ("Thermionic", "No Energy Filter", "None"),
                                   ("FEG", "No Energy Filter", "None")):
        selection = replace(selection, gun=gun, recording=recording, beam_blanker=blanker)
        assembly = catalog.apply(state, selection, preserve_operating_parameters=True)
        assert catalog.selection_for_resolved(assembly) == selection
        assert state.image_corrector_design == "bcor"
        assert state.objective_lens.percent == 62.345
        for key in ("image_hp3_hexapole", "image_tl31_lens", "image_tl32_lens"):
            assert unit_for_component(assembly, key) == "image_corrector"


def test_bcor_roundtrips_snapshot_state_and_operating_profile(tmp_path):
    catalog = AssemblyCatalog()
    state = default_state()
    selection = _bcor(catalog)
    catalog.apply(state, selection)
    hp3 = next(item for item in state.corrector_elements if item.key == "image_hp3_hexapole")
    hp3.strength_m3 = 1234.5
    for restored in (capture_instrument_snapshot(state).restore(), State.from_dict(state.to_dict())):
        assert restored.image_corrector_design == "bcor"
        assert catalog.selection_for_resolved(restored._resolved_assembly) == selection
        assert next(item for item in restored.corrector_elements if item.key == hp3.key).strength_m3 == hp3.strength_m3
    path = tmp_path / "bcor.toml"
    save_profile(path, state, selection)
    saved, values = read_profile(path)
    restored = default_state()
    catalog.apply(restored, saved)
    apply_profile_values(restored, values)
    assert saved == selection
    assert restored.image_corrector_design == "bcor"
    assert next(item for item in restored.corrector_elements if item.key == hp3.key).strength_m3 == hp3.strength_m3
    state.image_corrector_design = "cetcor"
    with pytest.raises(ValueError, match="image_corrector_design"):
        save_profile(tmp_path / "inconsistent.toml", state, selection)


def test_selected_tl31_manifest_field_edit_updates_runtime_without_changing_drive():
    from temsim.column.module_assembly import apply_column_manifest_geometry

    state, catalog = default_state(), AssemblyCatalog()
    catalog.apply(state, _bcor(catalog))
    original = state._resolved_assembly
    key = "image_tl31_lens"
    part = original.part(key)
    edited = replace(original, parts=tuple(
        replace(p, data=dict(p.data, maximum_peak_field_t=.731, field_half_width_mm=6.125,
            maximum_excitation_percent=87.5, field_profile_terms=((1., .2, .7),),
            normalise_field_profile_peak=False)) if p.key == key else p
        for p in original.parts))
    lens = state.image_corrector_system.tl31_lens
    lens.percent = 51.25
    apply_column_manifest_geometry(state, None, assembly=edited, preserve_operating_parameters=True)
    assert lens.percent == 51.25
    assert (lens.b0_t, lens.a_mm, lens.max_percent) == (.731, 6.125, 87.5)
    assert not lens.normalise_profile_peak
    assert [(g.amplitude, g.offset, g.sigma) for g in lens.gaussian] == [(1., .2, .7)]
    assert original.part(key).data == part.data


def test_temporarily_uninstalled_bcor_keeps_saved_drives_after_state_restore(tmp_path):
    catalog, state = AssemblyCatalog(), default_state()
    bcor = _bcor(catalog)
    catalog.apply(state, bcor, preserve_operating_parameters=True)
    state.image_corrector_system.tl31_lens.percent = 12.345
    state.image_corrector_system.hp3_hexapole.strength_m3 = 1234.5
    catalog.apply(state, replace(bcor, column="C3 + Image Corrector"), preserve_operating_parameters=True)
    restored = State.from_dict(state.to_dict())
    path = tmp_path / "dormant-bcor.toml"
    save_profile(path, state, catalog.selection_for_resolved(state._resolved_assembly))
    selected, values = read_profile(path)
    from_profile = default_state()
    catalog.apply(from_profile, selected)
    apply_profile_values(from_profile, values)
    for candidate in (state, restored, from_profile):
        catalog.apply(candidate, bcor, preserve_operating_parameters=True)
        assert candidate.image_corrector_system.tl31_lens.percent == 12.345
        assert candidate.image_corrector_system.hp3_hexapole.strength_m3 == 1234.5


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("enabled", [(True, False, True), (False, True, False)])
def test_dormant_bcor_state_and_profile_preserve_each_enable_preference(tmp_path, probe, enabled):
    catalog, state = AssemblyCatalog(), default_state()
    bcor = _bcor(catalog, probe=probe)
    catalog.apply(state, bcor)
    system = state.image_corrector_system
    extras = (system.tl31_lens, system.tl32_lens, system.hp3_hexapole)
    for component, preference in zip(extras, enabled):
        component.enabled = preference
    system.tl31_lens.percent = 12.345
    system.hp3_hexapole.strength_m3 = 1234.5
    cetcor = replace(bcor, column="C3 + Probe Corrector + Image Corrector" if probe else "C3 + Image Corrector")
    catalog.apply(state, cetcor, preserve_operating_parameters=True)
    restored = State.from_dict(state.to_dict())
    path = tmp_path / "dormant-preferences.toml"
    save_profile(path, state, cetcor)
    selection, values = read_profile(path)
    from_profile = default_state()
    catalog.apply(from_profile, selection)
    apply_profile_values(from_profile, values)
    for candidate in (state, restored, from_profile):
        objects = {c.key: c for c in (*candidate.lenses, *candidate.corrector_elements)}
        assert not any(objects[c.key].enabled for c in extras)
        catalog.apply(candidate, bcor, preserve_operating_parameters=True)
        actual = candidate.image_corrector_system
        assert (actual.tl31_lens.enabled, actual.tl32_lens.enabled, actual.hp3_hexapole.enabled) == enabled
        assert actual.tl31_lens.percent == 12.345
        assert actual.hp3_hexapole.strength_m3 == 1234.5


@pytest.mark.parametrize("probe", [False, True])
def test_bcor_state_restoration_still_produces_a_loadable_complete_profile(tmp_path, probe):
    import tomllib
    import tomli_w

    catalog, state = AssemblyCatalog(), default_state()
    bcor = _bcor(catalog, probe=probe)
    catalog.apply(state, bcor)
    state.image_corrector_system.tl31_lens.percent = 12.345
    state.image_corrector_system.hp3_hexapole.strength_m3 = 1234.5
    state.dph2_deflector.kick_x_mrad = .125
    restored = State.from_dict(state.to_dict())
    path = tmp_path / "restored-bcor.toml"
    save_profile(path, restored, bcor)
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    assert "effective_thickness_mm" not in document["devices"]["probe_dph2_deflector"]
    # Earlier writers accidentally included this geometry readout when the
    # dormant channel had lost its private hardware ownership metadata.
    document["devices"]["probe_dph2_deflector"]["effective_thickness_mm"] = 123.5
    document["devices"]["probe_dph2_deflector"]["optical_plane_separation_mm"] = 456.5
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    selected, values = read_profile(path)
    fresh = default_state()
    catalog.apply(fresh, selected)
    thickness = fresh.dph2_deflector.effective_thickness_mm
    apply_profile_values(fresh, values)
    assert fresh.dph2_deflector.effective_thickness_mm == thickness
    assert fresh.dph2_deflector.kick_x_mrad == .125
    assert fresh.image_corrector_system.tl31_lens.percent == 12.345
    assert fresh.image_corrector_system.hp3_hexapole.strength_m3 == 1234.5


@pytest.mark.parametrize("preferences", [[], {"unknown": True}, {"image_hp3_hexapole": "false"}, {"image_hp3_hexapole": 0}])
def test_invalid_bcor_profile_enable_preferences_are_atomic(preferences):
    from copy import deepcopy

    catalog, state = AssemblyCatalog(), default_state()
    catalog.apply(state, _bcor(catalog))
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="enabled preferences"):
        apply_profile_values(state, {"objective_lens": {"percent": 12.5},
            "__image_corrector_enabled_preferences__": preferences})
    assert state.to_dict() == before


def test_first_gui_bcor_installation_seeds_all_lenses_without_changing_tip_or_probe():
    state, catalog = default_state(), AssemblyCatalog()
    source_before = state.to_dict()["electron_gun"]
    before_drives = {lens.key: (lens.percent, lens.polarity) for lens in state.lenses
                     if not lens.key.startswith("image_")}
    candidate = check_instrument_configuration(state, catalog, InstrumentUnits(
        image_corrector=True, image_corrector_design="bcor", energy_filter=False)).candidate.restore()
    assert candidate.to_dict()["electron_gun"] == source_before
    assert {lens.key: (lens.percent, lens.polarity) for lens in candidate.lenses
            if not lens.key.startswith("image_")} == before_drives
    assert len(candidate.image_corrector_system.round_lens_components) == 7
    for lens in candidate.image_corrector_system.round_lens_components:
        assert lens.percent == candidate._resolved_assembly.part(lens.key).data["default_excitation_percent"]


def test_installed_cetcor_to_bcor_retains_existing_drives_and_seeds_extra_stage():
    state, catalog = default_state(), AssemblyCatalog()
    bcor = _bcor(catalog)
    catalog.apply(state, replace(bcor, column="C3 + Image Corrector"))
    state.image_corrector_system.adapter_lens.percent = 41.23
    catalog.apply(state, bcor, preserve_operating_parameters=True)
    assert state.image_corrector_system.adapter_lens.percent == 41.23
    for lens in (state.image_corrector_system.tl31_lens, state.image_corrector_system.tl32_lens):
        assert lens.percent == state._resolved_assembly.part(lens.key).data["default_excitation_percent"]


def test_bcor_dialog_selection_applies_once_and_reopens_with_same_design(qtbot, tmp_path):
    from PySide6.QtCore import QSettings, Qt
    from temsim.gui.assembly_panel import AssemblyPanel
    from temsim.gui.instrument_configuration_dialog import InstrumentConfigurationDialog

    catalog, state, applied = AssemblyCatalog(), default_state(), []
    settings = QSettings(str(tmp_path / "bcor-ui.ini"), QSettings.Format.IniFormat)
    dialog = InstrumentConfigurationDialog(catalog, lambda: state,
        lambda checked: applied.append(checked.restore_for(state)), settings=settings)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(dialog)
    assert not dialog.image_corrector_design.isEnabled()
    dialog.options["image_corrector"].setChecked(True)
    dialog.image_corrector_design.setCurrentIndex(dialog.image_corrector_design.findData("bcor"))
    assert dialog.image_corrector_design.isEnabled()
    assert dialog.units().selection(catalog).column == "C3 + Probe Corrector + B-COR"
    dialog.assemble_button.click()
    assert len(applied) == 1, dialog.status.text()
    assert applied[0].image_corrector_design == "bcor"
    second = InstrumentConfigurationDialog(catalog, lambda: applied[0], lambda _: None, settings=settings)
    qtbot.addWidget(second)
    assert second.image_corrector_design.currentData() == "bcor"
    second.options["c3_lens"].setChecked(False)
    assert not second.options["image_corrector"].isChecked()
    assert not second.image_corrector_design.isEnabled()
    panel = AssemblyPanel(catalog, catalog.selection_for_resolved(applied[0]._resolved_assembly))
    qtbot.addWidget(panel)
    assert "B-COR" in panel.configuration_summary.text()
