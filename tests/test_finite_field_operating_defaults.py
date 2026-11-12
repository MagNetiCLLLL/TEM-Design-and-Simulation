"""Preset selection and restoration checks without particle or wave propagation."""

from copy import deepcopy
from dataclasses import replace

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import (
    apply_condenser_mode,
    apply_operating_mode_pair,
    load_operating_mode_catalog,
    mode_by_key,
)
from temsim.optics.column import default_state
from temsim.runtime_parameters import runtime_targets
from temsim.simulation_modes import switch_mode


FOCUSED_CONTROLS = {
    "condenser_lens_2": {"percent": 13.899040164772698},
    "condenser_lens_3": {"percent": 15.68823681182136},
    "objective_lens": {"percent": 68.97810600833917},
    "probe_hp2_hexapole": {"strength_m3": 4371028.449290636},
    "probe_hp1_hexapole": {
        "strength_m3": 2616320.416243901,
        "orientation_rad": -0.042012393991127016,
    },
}


@pytest.fixture
def state():
    state = default_state()
    assembly = AssemblyCatalog()
    assembly.apply(state, assembly.default_selection())
    return state


def assert_controls(state, expected):
    targets = runtime_targets(state)
    for key, values in expected.items():
        for field, value in values.items():
            runtime_field = "polarity" if field == "field_polarity" else field
            assert getattr(targets[key].obj, runtime_field) == value, (key, field)


@pytest.fixture
def two_condenser_catalog(tmp_path, monkeypatch):
    """Independent synthetic controls; no measured physical qualification."""
    import tomllib
    import tomli_w
    import temsim.operating_modes as modes
    document = tomllib.loads((modes.OPERATING_MODE_CONFIG_ROOT / "catalog.toml").read_text(encoding="utf-8"))
    document["column_mode_overrides"] = {"C2": {"nano_probe": {
        "devices": {"condenser_lens_1": {"percent": 33.},
                    "condenser_lens_2": {"percent": 44.},
                    "mini_condenser": {"field_polarity": -1},
                    "objective_lens": {"percent": 55.}},
        "apertures": {"condenser_aperture_2": {"enabled": True, "diameter_mm": .10}},
        "targets": {"calibration_model": "unified", "qualification": "SYNTHETIC_TEST_ONLY"},
        "calibration_status": "synthetic_two_condenser_fixture",
        "calibration_reference": "Synthetic independent two-condenser control fixture; not measured.",
    }}}
    path = tmp_path / "catalog.toml"
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    monkeypatch.setattr(modes, "OPERATING_MODE_CONFIG_ROOT", tmp_path)
    modes.load_operating_mode_catalog.cache_clear()
    yield modes.load_operating_mode_catalog("unified"), document, path
    modes.load_operating_mode_catalog.cache_clear()


def two_condenser_state():
    from temsim.assembly_catalog import AssemblySelection
    state = default_state()
    switch_mode(state, "unified")
    AssemblyCatalog().apply(state, AssemblySelection("FEG", "C2", "No Energy Filter"))
    return state


def test_two_condenser_reference_replaces_controls_and_all_measured_claims(two_condenser_catalog):
    from temsim.operating_modes import compatible_modes
    catalog, document, _ = two_condenser_catalog
    actual = mode_by_key("nano_probe", catalog, column_name="C2")
    fixture = document["column_mode_overrides"]["C2"]["nano_probe"]
    assert actual.devices == fixture["devices"]
    assert actual.apertures == fixture["apertures"]
    assert actual.targets == fixture["targets"]
    assert "achieved_convergence_sem_angle_mrad" not in actual.targets
    assert "condenser_lens_3" not in actual.devices
    assert not any(key.startswith("probe_") for key in actual.devices)
    assert compatible_modes("condenser", "C2", "No Energy Filter", catalog) == (actual,)
    assert "condenser_lens_3" in mode_by_key("nano_probe", catalog).devices
    for key in ("micro_probe", "nano_probe_large_angle"):
        with pytest.raises(ValueError, match="no independent C2"):
            mode_by_key(key, catalog, column_name="C2")


def test_two_condenser_microprobe_needs_its_own_separate_reference(two_condenser_catalog):
    import tomli_w
    from temsim.operating_modes import compatible_modes
    _, document, path = two_condenser_catalog
    rows = document["column_mode_overrides"]["C2"]
    micro = deepcopy(rows["nano_probe"])
    micro["devices"]["mini_condenser"]["field_polarity"] = 1
    micro["targets"] = {"qualification": "SYNTHETIC_MICROPROBE_TEST_ONLY"}
    micro["calibration_reference"] = "Independent synthetic C2 Microprobe record, not a measurement."
    rows["micro_probe"] = micro
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    load_operating_mode_catalog.cache_clear()
    catalog = load_operating_mode_catalog("unified")
    definition = mode_by_key("micro_probe", catalog, column_name="C2")
    assert definition.devices == micro["devices"]
    assert definition.targets == micro["targets"]
    assert [mode.key for mode in compatible_modes("condenser", "C2", "No Energy Filter", catalog)] == [
        "micro_probe", "nano_probe"]


def test_two_condenser_application_preserves_gun_and_projector_only_preserves_defocus(two_condenser_catalog):
    from temsim.operating_modes import apply_projector_mode, condenser_mode_key
    catalog, _, _ = two_condenser_catalog
    state = two_condenser_state()
    assert state.simulation_mode == "unified"
    state.nanoprobe_range = "large_angle"  # A legacy field is not a C2 range.
    state.mini_condenser.signed_excitation_percent = -27.5
    source = deepcopy(state.electron_gun.to_dict())
    absent = {lens.key: (lens.percent, lens.enabled) for lens in state.lenses
              if lens.key == "condenser_lens_3" or lens.key == "adapter_lens" or lens.key.startswith("probe_")}
    result = apply_operating_mode_pair(state, "nano_probe", "diffraction", catalog=catalog)
    assert result.condenser == mode_by_key("nano_probe", catalog, column_name="C2")
    assert condenser_mode_key(state) == "nano_probe"
    assert state.column_mode == "two_condenser"
    assert state.mini_condenser.signed_excitation_percent == -27.5
    assert state.electron_gun.to_dict() == source
    assert absent == {lens.key: (lens.percent, lens.enabled) for lens in state.lenses if lens.key in absent}
    state.objective_lens.percent += .25
    state.sync_objective()
    upstream = {lens.key: (lens.percent, lens.enabled) for lens in state.lenses if lens.z_mm <= state.sample.z_mm}
    apply_projector_mode(state, "diffraction", catalog=catalog)
    assert upstream == {lens.key: (lens.percent, lens.enabled) for lens in state.lenses if lens.key in upstream}
    assert state.electron_gun.to_dict() == source


@pytest.mark.parametrize("hardware,column_name,key", [
    ("C2", None, "nano_probe_large_angle"),
    ("C2", "C3 + Probe Corrector", "nano_probe"),
    ("C3 + Probe Corrector", "C2", "nano_probe"),
])
def test_two_condenser_and_three_condenser_presets_cannot_cross_hardware_boundaries(
        two_condenser_catalog, hardware, column_name, key):
    catalog, _, _ = two_condenser_catalog
    state = two_condenser_state() if hardware == "C2" else default_state()
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError):
        apply_operating_mode_pair(state, key, "diffraction", column_name=column_name, catalog=catalog)
    assert state.to_dict() == before


def test_legacy_c2_c3_off_label_migrates_without_retuning_or_recreating_c3():
    from temsim.operating_modes import condenser_mode_key
    from temsim.optics.model import State
    state = two_condenser_state()
    state.condenser_lens_2.lens.percent = 37.125
    state.objective_lens.percent = 23.75
    payload = deepcopy(state.to_dict())
    payload["column_mode"] = "two_lens_c3_off"
    payload["nanoprobe_range"] = "large_angle"
    restored = State.from_dict(payload)
    assert restored.column_mode == restored.to_dict()["column_mode"] == "two_condenser"
    assert condenser_mode_key(restored) == "nano_probe"
    assert restored.condenser_lens_2.lens.percent == 37.125
    assert restored.objective_lens.percent == 23.75
    assert restored.electron_gun.to_dict() == state.electron_gun.to_dict()
    assert "condenser_lens_3" not in {item.key for item in restored._resolved_optics_layout}
    for mode in ("three_lens", "two_lens_c3_off"):
        before = deepcopy(restored.to_dict())
        with pytest.raises(ValueError, match="has no C3"):
            restored.condenser_system.apply_mode(mode)
        assert restored.to_dict() == before


def test_switching_off_installed_c3_retains_three_condenser_hardware():
    from temsim.column.state_layout import apply_physical_layout_to_state
    state = default_state()
    state.condenser_system.apply_mode("two_lens_c3_off")
    assert state.layout_c3_hardware == "three_condenser"
    assert not state.layout_c3_excited
    apply_physical_layout_to_state(state)
    assert state.column_mode == "two_lens_c3_off"
    assert "condenser_lens_3" in {item.key for item in state._resolved_optics_layout}
    assert not state.condenser_lens_3.lens.enabled
    state.condenser_system.apply_mode("three_lens")
    assert state.layout_c3_excited and state.condenser_lens_3.lens.enabled


@pytest.mark.parametrize("mutation", ["missing_targets", "missing_apertures", "c3_control", "large_angle"])
def test_two_condenser_override_rejects_inherited_or_absent_hardware_records(two_condenser_catalog, mutation):
    import tomli_w
    _, document, path = two_condenser_catalog
    rows = document["column_mode_overrides"]["C2"]
    entry = rows["nano_probe"]
    if mutation.startswith("missing_"):
        del entry[mutation.removeprefix("missing_")]
    elif mutation == "c3_control":
        entry["devices"]["condenser_lens_3"] = {"percent": 0.}
    else:
        rows["nano_probe_large_angle"] = deepcopy(entry)
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    load_operating_mode_catalog.cache_clear()
    with pytest.raises(ValueError):
        load_operating_mode_catalog("unified")


def test_two_condenser_gui_exposes_one_nanoprobe_without_range_or_phantom_c3(
        two_condenser_catalog, qtbot):
    from temsim.assembly_catalog import AssemblySelection
    from temsim.gui.assembly_panel import AssemblyPanel
    catalog, _, _ = two_condenser_catalog
    selection = AssemblySelection("FEG", "C2", "No Energy Filter")
    state = two_condenser_state()
    state.nanoprobe_range = "large_angle"
    widget = AssemblyPanel(AssemblyCatalog(), selection)
    qtbot.addWidget(widget)
    widget.operating_mode_catalog = catalog
    widget.load_operating_modes(selection, "nano_probe", "diffraction")
    widget.set_direct_alignment_state(state)
    assert widget.probe_mode.count() == 1
    assert widget.selected_condenser_mode_key() == "nano_probe"
    assert widget.nanoprobe_range_combo.count() == 0
    assert widget.nanoprobe_range_combo.isHidden() and widget.nanoprobe_range_label.isHidden()
    status = widget.operating_mode_status.text()
    assert "Normal" not in status and "Large angle" not in status and "C3" not in status
    assert "unmeasured" in status
    assert "Applied: Nanoprobe + Diffraction" in widget.direct_alignment_panel.mode_status.text()
    requests = []
    widget.operating_mode_requested.connect(lambda *args: requests.append(args))
    widget.apply_operating_mode_button.click()
    assert requests == [("nano_probe", "diffraction")]


def test_current_c2_nanoprobe_diffraction_pair_uses_its_own_projector_and_preserves_incident_edits(monkeypatch):
    """Exercise real catalog defaults without generating any source particles."""
    from temsim.operating_modes import apply_projector_mode
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    monkeypatch.setattr(ColdFieldEmitter, "emit", lambda *_args, **_kwargs:
                        pytest.fail("Applying a stored mode must not generate a finite source"))
    state = two_condenser_state()
    state.electron_gun.electrostatic_lens.voltage_kv = 6.125
    before_gun = deepcopy(state.electron_gun.to_dict())
    catalog = load_operating_mode_catalog("unified")
    c2_diffraction = mode_by_key("diffraction", catalog, column_name="C2")
    c3_diffraction = mode_by_key("diffraction", catalog, column_name="C3 + Probe Corrector")
    assert c3_diffraction.devices["diffraction_lens"]["percent"] == 14.577617197168571
    assert c3_diffraction.devices["intermediate_lens"]["percent"] == 0.035082657259397326
    assert c2_diffraction.devices["diffraction_lens"] != c3_diffraction.devices["diffraction_lens"]
    assert c2_diffraction.devices["intermediate_lens"] != c3_diffraction.devices["intermediate_lens"]
    result = apply_operating_mode_pair(state, "nano_probe", "diffraction",
        column_name="C2", recording_name="No Energy Filter", catalog=catalog)
    assert result.projector == c2_diffraction
    assert state.condenser_aperture_2.diameter_mm == pytest.approx(.07)
    targets = result.condenser.targets
    assert targets["c2_aperture_diameter_um"] == 70.
    assert targets["minimum_convergence_sem_angle_mrad"] == 10.
    assert targets["achieved_convergence_sem_angle_mrad"] >= 10.
    assert targets["calibration_ray_count"] == 4096
    assert targets["sampling_convergence"] == "NOT_CONVERGED"
    assert targets["source_comparison_direct_surface_focus_passed"] is False
    assert targets["qualification_scope"].startswith("fixed_4096_")
    assert_controls(state, c2_diffraction.devices)
    assert state.electron_gun.to_dict() == before_gun
    assert not state.fluorescent_screen.inserted and not state.camera.inserted
    assert all(detector.inserted for detector in state.stem_detectors)
    absent = {"condenser_lens_3", "adapter_lens", "probe_hp1_hexapole", "probe_hp2_hexapole",
              "probe_tl22_lens", "probe_tl21_lens", "probe_tl12_lens"}
    assert absent.isdisjoint({item.key for item in state._resolved_optics_layout})
    assert all(not lens.enabled for lens in state.lenses if lens.key in absent)
    assert all(not item.enabled for item in state.corrector_elements if item.key in absent)
    # A projector-only request must retain a deliberately perturbed focus and
    # the user's latest gun voltage, while restoring the C2 D/I/P1/P2 values.
    state.objective_lens.percent += .02
    state.electron_gun.electrostatic_lens.voltage_kv = 5.9
    state.sync_objective()
    upstream = {lens.key: (lens.percent, lens.polarity, lens.enabled)
                for lens in state.lenses if lens.z_mm <= state.sample.z_mm}
    before_gun = deepcopy(state.electron_gun.to_dict())
    targets = runtime_targets(state)
    targets["diffraction_lens"].obj.percent += .1
    targets["intermediate_lens"].obj.percent += .1
    apply_projector_mode(state, "diffraction", column_name="C2",
                         recording_name="No Energy Filter", catalog=catalog)
    assert_controls(state, c2_diffraction.devices)
    assert upstream == {lens.key: (lens.percent, lens.polarity, lens.enabled)
                        for lens in state.lenses if lens.key in upstream}
    assert state.electron_gun.to_dict() == before_gun
    assert not state.fluorescent_screen.inserted and not state.camera.inserted
    assert c3_diffraction == mode_by_key("diffraction", catalog, column_name="C3 + Probe Corrector")


def test_current_c2_gui_discloses_70um_fixed_4096_scope_and_unconverged_source_sampling(qtbot, monkeypatch):
    from temsim.assembly_catalog import AssemblySelection
    from temsim.gui.assembly_panel import AssemblyPanel
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    monkeypatch.setattr(ColdFieldEmitter, "emit", lambda *_args, **_kwargs:
                        pytest.fail("Reading a stored C2 reference must not emit particles"))
    selection = AssemblySelection("FEG", "C2", "No Energy Filter")
    widget = AssemblyPanel(AssemblyCatalog(), selection)
    qtbot.addWidget(widget)
    widget.load_operating_modes(selection, "nano_probe", "diffraction", simulation_mode="unified")
    assert widget.selected_condenser_mode_key() == "nano_probe"
    assert widget.nanoprobe_range_combo.isHidden()
    assert "70" in widget.operating_mode_status.text()
    tooltip = widget.operating_mode_status.toolTip()
    assert "4096" in tooltip and "70 um" in tooltip
    assert "sampling is NOT converged" in tooltip
    assert "2048" in tooltip and "16 nm upstream" in tooltip
    assert "RMS minimum is 72 nm downstream" in tooltip


def test_unified_normal_uses_focused_controls_with_an_independent_qualification_record(state):
    switch_mode(state, "unified")
    source = deepcopy(state.electron_gun.to_dict())
    step = state.step_mm
    result = apply_operating_mode_pair(state, "nano_probe", "diffraction")
    assert_controls(state, FOCUSED_CONTROLS)
    assert state.mini_condenser.percent == 43.88646617222637
    assert state.mini_condenser.polarity == -1
    assert state.electron_gun.to_dict() == source
    assert state.step_mm == step
    targets = result.condenser.targets
    assert targets["calibration_model"] == "unified"
    assert targets["target_convergence_sem_angle_mrad"] == 10.0
    assert targets["calibration_ray_count"] == 49
    assert targets["calibration_step_mm"] == 0.05
    assert targets["validation_step_mm"] == 0.025
    assert targets["sampling_convergence"] == "NOT_TESTED"
    assert targets["topology_qualification"] == "NOT_TESTED"
    assert "condenser_topology" not in targets
    assert "corrected_nonlinear_error_nm" not in targets
    assert "0.05 and 0.025 mm" in result.condenser.calibration_reference


@pytest.mark.parametrize("model", ["custom", "ideal", "analytical", "ideal_thin"])
def test_legacy_models_keep_their_stored_normal_preset(state, model):
    switch_mode(state, model)
    result = apply_condenser_mode(state, "nano_probe")
    legacy = mode_by_key("nano_probe", load_operating_mode_catalog())
    assert result.condenser == legacy
    assert_controls(state, legacy.devices)
    assert legacy.devices["condenser_lens_2"]["percent"] == 23.225625142668733
    assert legacy.targets["target_convergence_sem_angle_mrad"] == 30.0


def test_normal_large_angle_normal_and_repeated_application_restore_the_focused_preset(state):
    switch_mode(state, "unified")
    apply_condenser_mode(state, "nano_probe")
    normal = deepcopy(state.to_dict())
    apply_condenser_mode(state, "nano_probe_large_angle")
    assert_controls(state, {"condenser_lens_3": {"percent": 46.059815218359184}})
    apply_condenser_mode(state, "nano_probe")
    assert state.to_dict() == normal
    apply_condenser_mode(state, "nano_probe")
    assert state.to_dict() == normal
    assert_controls(state, FOCUSED_CONTROLS)


def test_finite_field_preset_keeps_a_manually_selected_cm_magnitude(state):
    switch_mode(state, "unified")
    state.mini_condenser.percent = 37.25
    state.electron_gun.electrostatic_lens.voltage_kv = 6.1
    result = apply_condenser_mode(state, "nano_probe")
    assert state.mini_condenser.percent == 37.25
    assert state.mini_condenser.polarity == -1
    assert state.electron_gun.electrostatic_lens.voltage_kv == 6.1
    assert_controls(state, FOCUSED_CONTROLS)
    assert "stored reference semi-angle 9.969 mrad" in result.summary
    assert "sample semi-angle" not in result.summary
    assert "not recalculated for current controls, geometry or source model" in result.summary
    assert "require revalidation" in result.summary


@pytest.mark.parametrize("model", ["unified", "custom", "ideal", "analytical", "ideal_thin"])
def test_catalog_application_never_claims_a_new_illumination_measurement(state, model):
    switch_mode(state, model)
    result = apply_condenser_mode(state, "nano_probe")
    angle = float(result.condenser.targets["achieved_convergence_sem_angle_mrad"])
    assert f"stored reference semi-angle {angle:.3f} mrad" in result.summary
    assert "not recalculated" in result.summary
    assert "require revalidation" in result.summary


def test_explicit_catalog_wins_over_the_field_model(state):
    switch_mode(state, "unified")
    legacy = load_operating_mode_catalog()
    original = mode_by_key("nano_probe", legacy)
    devices = deepcopy(original.devices)
    devices["condenser_lens_2"]["percent"] = 17.5
    custom = replace(original, devices=devices)
    explicit = replace(legacy, modes=tuple(
        custom if item.key == "nano_probe" else item for item in legacy.modes
    ))
    result = apply_condenser_mode(state, "nano_probe", catalog=explicit)
    assert result.condenser is custom
    assert_controls(state, devices)
    assert mode_by_key("nano_probe", legacy).devices["condenser_lens_2"]["percent"] == 23.225625142668733


def test_model_switch_does_not_apply_a_preset_or_restore_shelved_currents(state):
    apply_condenser_mode(state, "nano_probe")
    runtime_targets(state)["condenser_lens_2"].obj.percent = 17.5
    before = [(lens.key, lens.percent, lens.polarity) for lens in state.lenses]
    correctors = deepcopy(state.to_dict()["corrector_elements"])
    switch_mode(state, "unified")
    assert [(lens.key, lens.percent, lens.polarity) for lens in state.lenses] == before
    assert state.to_dict()["corrector_elements"] == correctors
    apply_condenser_mode(state, "nano_probe")
    assert_controls(state, FOCUSED_CONTROLS)


def test_failed_projector_application_rolls_back_the_resolved_finite_field_preset(state, monkeypatch):
    from temsim import operating_modes

    switch_mode(state, "unified")
    before = deepcopy(state.to_dict())
    apply_values = operating_modes._apply_values

    def reject_projector(candidate, definition):
        if definition.family == "projector":
            assert_controls(candidate, FOCUSED_CONTROLS)
            raise ValueError("Synthetic projector failure")
        return apply_values(candidate, definition)

    monkeypatch.setattr(operating_modes, "_apply_values", reject_projector)
    with pytest.raises(ValueError, match="Synthetic projector failure"):
        apply_operating_mode_pair(state, "nano_probe", "diffraction")
    assert state.to_dict() == before
