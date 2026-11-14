"""Round-lens law selection and persistence; no source emission or transport."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from temsim.round_lens_models import (
    gaussian_geometry_metadata, gaussian_geometry_scale,
    set_gaussian_geometry_coupling, set_round_lens_field_model,
    validate_round_lens_selection,
)


@pytest.fixture(autouse=True)
def forbid_source_transport(monkeypatch):
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    from temsim.optics.electron_gun import source
    def forbidden(*args, **kwargs):
        pytest.fail("Round-lens configuration tests must not emit or transport a source")
    monkeypatch.setattr(ColdFieldEmitter, "emit", forbidden)
    monkeypatch.setattr(source, "trace_source_to_exit", forbidden)


@pytest.fixture(scope="module")
def default_payload():
    from temsim.optics.column import default_state
    return default_state().to_dict()


@pytest.fixture
def state(default_payload):
    from temsim.optics.model import State
    return State.from_dict(deepcopy(default_payload))


def _geometry_state(*, reference=100., length=100., coupling="axial_length", objective=False):
    parent_data = {"mechanical_profile": "magnetic_lens_assembly",
                   "gaussian_reference_coil_length_mm": reference}
    if objective:
        parent_data.update(local_start_z_mm=0., upper_yoke_start_local_z_mm=0.,
            upper_yoke_end_local_z_mm=30., lower_yoke_start_local_z_mm=70.,
            lower_yoke_end_local_z_mm=100., mechanical_coil_axial_inset_mm=5.)
    owner = NS(key="example_lens", parent_key=None, start_z_mm=10., end_z_mm=110., data=parent_data)
    coil = NS(key="example_coil", parent_key="example_lens", start_z_mm=10., end_z_mm=10.+length,
              data={"mechanical_profile": "magnetic_excitation_coil", "local_start_z_mm": 0.})
    return NS(simulation_mode="unified", round_lens_field_model="gaussian",
              gaussian_geometry_coupling=coupling, lenses=[NS(key=owner.key, enabled=True)],
              _resolved_assembly=NS(parts=(owner, coil)))


def test_historical_fixed_default_does_not_require_or_read_new_metadata():
    bare = NS()
    assert gaussian_geometry_scale(bare, "unknown_historical_lens") == 1.
    assert gaussian_geometry_metadata(bare, "old")["mode"] == "fixed_calibration"


def test_axial_rule_scales_by_current_bearing_length_and_never_mutates_geometry():
    state = _geometry_state(reference=80., length=100.)
    before = deepcopy(state._resolved_assembly)
    report = gaussian_geometry_metadata(state, "example_lens")
    assert report["width_scale"] == 1.25
    assert report["actual_coil_length_mm"] == 100.
    assert report["reference_coil_length_mm"] == 80.
    assert state._resolved_assembly == before
    state._resolved_assembly.parts[1].end_z_mm += 20.
    assert gaussian_geometry_scale(state, "example_lens") == 1.5


def test_objective_uses_split_material_lengths_instead_of_coil_envelope():
    state = _geometry_state(reference=40., length=100., objective=True)
    report = gaussian_geometry_metadata(state, "example_lens")
    assert report["actual_coil_length_mm"] == 40.
    assert report["coils"][0]["current_bearing_intervals_mm"] == [[15., 35.], [85., 105.]]
    assert report["width_scale"] == 1.


@pytest.mark.parametrize("reference", [0., -1., float("nan"), float("inf"), True, None])
def test_invalid_reference_is_not_a_permission_to_infer_a_new_calibration(reference):
    with pytest.raises(ValueError, match="gaussian_reference_coil_length_mm"):
        gaussian_geometry_scale(_geometry_state(reference=reference), "example_lens")


def test_missing_coil_or_overlapping_current_intervals_reject():
    state = _geometry_state()
    coil = state._resolved_assembly.parts[1]
    coil.data["material_intervals_mm"] = [[0., 60.], [50., 100.]]
    with pytest.raises(ValueError, match="ordered, disjoint"):
        gaussian_geometry_scale(state, "example_lens")
    state._resolved_assembly.parts = state._resolved_assembly.parts[:1]
    with pytest.raises(ValueError, match="explicitly owned coil"):
        gaussian_geometry_scale(state, "example_lens")


@pytest.mark.parametrize("mode", ["custom", "ideal", "analytical", "ideal_thin", "linear_geometry", "nonlinear_material"])
@pytest.mark.parametrize("model,coupling", [("copper_coil", "fixed_calibration"), ("gaussian", "axial_length")])
def test_new_laws_cannot_be_silently_ignored_by_other_execution_modes(mode, model, coupling):
    with pytest.raises(ValueError, match="shared Finite-Field Optics"):
        validate_round_lens_selection(model, coupling, mode)


@pytest.mark.parametrize("field,value", [("round_lens_field_model", "invalid"), ("gaussian_geometry_coupling", "invalid"),
                                         ("round_lens_field_model", True), ("gaussian_geometry_coupling", None)])
def test_state_rejects_invalid_source_choices(default_payload, field, value):
    from temsim.optics.model import State
    payload = deepcopy(default_payload)
    payload[field] = value
    with pytest.raises(ValueError):
        State.from_dict(payload)


def test_legacy_payload_omits_new_defaults_and_roundtrips_without_extra_keys(state):
    from temsim.optics.model import State
    payload = state.to_dict()
    assert "round_lens_field_model" not in payload
    assert "gaussian_geometry_coupling" not in payload
    restored = State.from_dict(payload)
    assert restored.round_lens_field_model == "gaussian"
    assert restored.gaussian_geometry_coupling == "fixed_calibration"
    assert restored.to_dict() == payload


def test_selected_law_is_explicit_in_worker_payload_and_input_identity(state):
    from temsim.optics.model import State
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.simulation_modes import switch_mode
    switch_mode(state, "unified")
    original = capture_instrument_snapshot(state)
    controls = [(x.key, x.percent, x.polarity) for x in state.lenses]
    state._runtime_lens_field_provider_cache = {"old": object()}
    assert set_round_lens_field_model(state, "copper_coil")
    assert state._runtime_lens_field_provider_cache == {}
    payload = state.to_dict()
    assert payload["round_lens_field_model"] == "copper_coil"
    restored = State.from_dict(payload)
    assert restored.round_lens_field_model == "copper_coil"
    assert [(x.key, x.percent, x.polarity) for x in restored.lenses] == controls
    assert restored.to_dict() == payload
    assert capture_instrument_snapshot(state).digest != original.digest
    # JSON workers preserve the complete declared state. Exact working-point
    # object graphs (including aliasing and mode shelves) use snapshot.restore.
    snapshot = capture_instrument_snapshot(state)
    exact = snapshot.restore()
    assert exact.round_lens_field_model == "copper_coil"
    assert capture_instrument_snapshot(exact).digest == snapshot.digest


def test_axial_selection_is_atomic_and_validates_active_native_lengths():
    state = _geometry_state(reference=None, coupling="fixed_calibration")
    cache = {"old": object()}
    state._runtime_lens_field_provider_cache = cache
    with pytest.raises(ValueError):
        set_gaussian_geometry_coupling(state, "axial_length")
    assert state.gaussian_geometry_coupling == "fixed_calibration"
    assert state._runtime_lens_field_provider_cache is cache


@pytest.mark.parametrize("source", ["fem", "imported", "bound"])
def test_gaussian_length_selection_does_not_require_metadata_for_mapped_sources(source):
    state = _geometry_state(reference=None, coupling="fixed_calibration")
    state.lens_field_map_descriptors = {}
    if source == "fem":
        state.lens_field_map_descriptors = {"example_lens": {"solver": "axisymmetric_linear_fem"}}
    elif source == "imported":
        state.lens_field_map_descriptors = {"example_lens": {"source_path": "retained-external-map.npz"}}
    else:
        state._lens_field_map_bindings = {"example_lens": object()}
    assert set_gaussian_geometry_coupling(state, "axial_length")
    assert state.gaussian_geometry_coupling == "axial_length"
    # A later actual native fallback still fails closed; this helper does not
    # redefine a missing reference from the current coil dimensions.
    with pytest.raises(ValueError, match="gaussian_reference_coil_length_mm"):
        gaussian_geometry_scale(state, "example_lens")


def test_returning_to_linked_gaussian_rejects_missing_native_metadata_atomically():
    state = _geometry_state(reference=None)
    state.round_lens_field_model = "copper_coil"
    with pytest.raises(ValueError, match="gaussian_reference_coil_length_mm"):
        set_round_lens_field_model(state, "gaussian")
    assert state.round_lens_field_model == "copper_coil"


def test_switching_away_does_not_silently_restore_gaussian(state):
    from temsim.simulation_modes import switch_mode
    switch_mode(state, "unified")
    set_round_lens_field_model(state, "copper_coil")
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="shared Finite-Field Optics"):
        switch_mode(state, "ideal_thin")
    assert state.to_dict() == before


def test_copper_coherent_entry_is_rejected_without_running_a_wave(state):
    from temsim.simulation_modes import switch_mode, coherent_mode_issue, require_coherent_mode
    switch_mode(state, "unified")
    assert coherent_mode_issue(state) is None
    set_round_lens_field_model(state, "copper_coil")
    assert "classical particles only" in coherent_mode_issue(state)
    with pytest.raises(ValueError, match="classical particles only"):
        require_coherent_mode(state)


def test_operating_profile_restores_explicit_model_and_historical_profile_defaults(state, tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.simulation_modes import switch_mode
    switch_mode(state, "unified")
    set_round_lens_field_model(state, "copper_coil")
    path = tmp_path/"copper.toml"
    save_profile(path, state, AssemblyCatalog().selection_for_resolved(state._resolved_assembly))
    _, values = read_profile(path)
    assert values["__simulation_model__"]["settings"]["round_lens_field_model"] == "copper_coil"
    set_round_lens_field_model(state, "gaussian")
    apply_profile_values(state, values)
    assert state.round_lens_field_model == "copper_coil"
    historical = deepcopy(values)
    historical["__simulation_model__"]["settings"].pop("round_lens_field_model")
    historical["__simulation_model__"]["settings"].pop("gaussian_geometry_coupling")
    apply_profile_values(state, historical)
    assert state.round_lens_field_model == "gaussian"
    assert state.gaussian_geometry_coupling == "fixed_calibration"


def test_linked_gaussian_roundtrips_without_rewriting_native_calibrations(state, tmp_path):
    from temsim.optics.model import State
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.simulation_modes import switch_mode
    switch_mode(state, "unified")
    before = [(lens.key, lens.percent, lens.polarity, getattr(lens, "a_mm", None)) for lens in state.lenses]
    assert set_gaussian_geometry_coupling(state, "axial_length")
    for lens in state.lenses:
        if lens.enabled:
            assert gaussian_geometry_scale(state, lens.key) == pytest.approx(1., rel=1e-12)
    payload = state.to_dict()
    assert payload["gaussian_geometry_coupling"] == "axial_length"
    restored = State.from_dict(payload)
    assert restored.gaussian_geometry_coupling == "axial_length"
    assert [(lens.key, lens.percent, lens.polarity, getattr(lens, "a_mm", None)) for lens in state.lenses] == before
    path = tmp_path/"linked-gaussian.toml"
    save_profile(path, state, AssemblyCatalog().selection_for_resolved(state._resolved_assembly))
    _, values = read_profile(path)
    set_gaussian_geometry_coupling(restored, "fixed_calibration")
    apply_profile_values(restored, values)
    assert restored.gaussian_geometry_coupling == "axial_length"


def test_menu_model_and_length_choices_are_explicit_and_reenable_old_modes(state, qtbot):
    from temsim.gui.simulation_menu import SimulationMenu
    from temsim.simulation_modes import switch_mode
    menu = SimulationMenu(); qtbot.addWidget(menu)
    menu.set_state(state)
    assert menu.round_lens_model_actions["gaussian"].isChecked()
    assert not menu.round_lens_model_actions["copper_coil"].isEnabled()
    switch_mode(state, "unified"); menu.set_state(state)
    before = deepcopy(state.to_dict())
    with qtbot.waitSignal(menu.round_lens_model_requested) as signal:
        menu.round_lens_model_actions["copper_coil"].trigger()
    assert signal.args == ["copper_coil"]
    assert state.to_dict() == before  # A menu emits a request; it does not retune optics.
    set_round_lens_field_model(state, "copper_coil"); menu.set_state(state)
    assert not menu.mode_actions["ideal_thin"].isEnabled()
    assert not menu.gaussian_geometry_actions["axial_length"].isEnabled()
    set_round_lens_field_model(state, "gaussian"); menu.set_state(state)
    assert menu.mode_actions["ideal_thin"].isEnabled()
    assert menu.gaussian_geometry_actions["axial_length"].isEnabled()


def test_model_inspector_changes_round_source_without_emission_or_excitation_edits(state, qtbot):
    from temsim.gui.model_inspector import ModelInspectorPage
    from temsim.simulation_modes import switch_mode
    switch_mode(state, "unified")
    page = ModelInspectorPage(); qtbot.addWidget(page); page.set_state(state)
    controls = [(x.key, x.percent, x.polarity) for x in state.lenses]
    with qtbot.waitSignal(page.changed):
        page.round_lens_model.setCurrentIndex(page.round_lens_model.findData("copper_coil"))
    assert state.round_lens_field_model == "copper_coil"
    assert "Pole pieces, iron and saturation are ignored" in page.round_lens_hint.text()
    assert "Air-core NI" in page.current_summary.text()
    assert not page.geometry_field_button.isEnabled()
    assert not page.analytic_field_button.isEnabled()
    assert [(x.key, x.percent, x.polarity) for x in state.lenses] == controls
    assert "Copper-coil field requested" in {page.table.item(i, 2).text() for i in range(page.table.rowCount())}
