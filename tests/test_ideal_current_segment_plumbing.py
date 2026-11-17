"""Input and dormant-recipe checks; no finite source or wave calculation."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from temsim.ideal_coil_segments import FIELD, SCHEMA, normalize_recipe


@pytest.fixture(autouse=True)
def forbid_source_transport(monkeypatch):
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    from temsim.optics.electron_gun import source

    def forbidden(*args, **kwargs):
        pytest.fail("Electrical recipe tests must not emit or transport particles")

    monkeypatch.setattr(ColdFieldEmitter, "emit", forbidden)
    monkeypatch.setattr(source, "trace_source_to_exit", forbidden)


@pytest.fixture
def state():
    from temsim.optics.column import default_state
    result = default_state()
    result.simulation_mode = "unified"
    result.round_lens_field_model = "gaussian"
    return result


def recipe():
    return dict(schema=SCHEMA, windings={"test_winding": [dict(
        interval_index=0, turn_count=10, radial_fraction=[0., 1.],
        axial_fraction=[0., 1.], current_ratio=-.5)]})


def test_state_serialization_detaches_nested_electrical_cells(state):
    from temsim.optics.model import State
    key = state.lenses[0].key
    state.lens_field_map_descriptors[key] = {FIELD: recipe()}
    payload = state.to_dict()
    payload["lens_field_map_descriptors"][key][FIELD]["windings"]["test_winding"][0]["current_ratio"] = -.75
    assert state.lens_field_map_descriptors[key][FIELD] == recipe()
    restored = State.from_dict(payload)
    payload["lens_field_map_descriptors"][key][FIELD]["windings"]["test_winding"][0]["current_ratio"] = -1.
    assert restored.lens_field_map_descriptors[key][FIELD]["windings"]["test_winding"][0]["current_ratio"] == -.75


def test_profile_boundary_validates_segment_syntax_without_resolving_geometry():
    from temsim.simulation_modes import normalise_profiles
    incoming = {"unified": {"lens_field_map_descriptors": {"test_lens": {FIELD: recipe()}}}}
    normalized = normalise_profiles(incoming)
    assert normalized == incoming
    incoming["unified"]["lens_field_map_descriptors"]["test_lens"][FIELD]["windings"]["test_winding"][0]["current_ratio"] = 3.
    assert normalized["unified"]["lens_field_map_descriptors"]["test_lens"][FIELD] == recipe()
    invalid = recipe()
    invalid["windings"]["test_winding"][0]["current_ratio"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        normalise_profiles({"unified": {"lens_field_map_descriptors": {"test_lens": {FIELD: invalid}}}})


def test_gaussian_resolver_ignores_only_dormant_segments_without_map_lookup(state, monkeypatch):
    from temsim.physics import lens_field_provider as fields
    lens = next(item for item in state.lenses if item.key == "objective_lens")
    points = np.array([[0., 0., lens.z_mm*.001], [1e-6, 0., (lens.z_mm+.1)*.001]])
    before = fields.resolve_runtime_lens_field_provider(state, lens.key, lens)
    expected = before.field_at_global_positions_t(points)
    state.lens_field_map_descriptors[lens.key] = {FIELD: recipe()}

    def forbidden(*args, **kwargs):
        pytest.fail("A dormant current recipe must not request a field map")

    monkeypatch.setattr(fields, "_load_descriptor_map", forbidden)
    actual = fields.resolve_runtime_lens_field_provider(state, lens.key, lens)
    np.testing.assert_array_equal(actual.field_at_global_positions_t(points), expected)
    assert actual.fallback_reason == before.fallback_reason
    assert state._field_provider_diagnostics[lens.key]["descriptor_error"] is None
    assert state.lens_field_map_descriptors[lens.key][FIELD] == recipe()


def test_dormant_segments_do_not_skip_native_gaussian_length_validation(monkeypatch):
    from temsim import round_lens_models as models
    state = NS(lenses=[NS(key="test_lens", enabled=True)],
        lens_field_map_descriptors={"test_lens": {FIELD: recipe()}},
        _resolved_assembly=None)
    seen = []
    monkeypatch.setattr(models, "gaussian_geometry_metadata", lambda candidate, key: seen.append(key))
    models._preflight_native_gaussian_lengths(state)
    assert seen == ["test_lens"]


def test_map_binding_and_clear_preserve_dormant_segments(monkeypatch):
    from temsim.physics import lens_field_provider as fields
    original = recipe()
    state = NS(lens_field_map_descriptors={"test_lens": {FIELD: original}},
        _runtime_lens_field_provider_cache={"test_lens": object()})
    binding = NS(geometry_fingerprint="unchanged-geometry")
    field_map = NS(geometry_fingerprint=binding.geometry_fingerprint)
    monkeypatch.setattr(fields, "lens_geometry_binding", lambda *args: binding)
    monkeypatch.setattr(fields, "field_map_descriptor", lambda *args: {"path": "test-map.npz"})
    fields.bind_imported_lens_field_map(state, "test_lens", field_map)
    assert state.lens_field_map_descriptors["test_lens"] == {"path": "test-map.npz", FIELD: recipe()}
    assert state.lens_field_map_descriptors["test_lens"][FIELD] is not original
    fields.clear_imported_lens_field_map(state, "test_lens")
    assert state.lens_field_map_descriptors["test_lens"] == {FIELD: recipe()}
    assert "test_lens" not in state._lens_field_map_bindings
    assert "test_lens" not in state._runtime_lens_field_provider_cache


def test_fem_editor_and_reset_keep_dormant_current_recipe(state, qtbot):
    from temsim.gui.model_inspector import ModelInspectorPage
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    key = page.lens.currentData()
    state.lens_field_map_descriptors[key] = {FIELD: recipe()}
    page._load_field()
    page.permeability.setText("2000")
    page.ampere_turns.setText("250")
    page._apply_field()
    assert state.lens_field_map_descriptors[key]["solver"] == "axisymmetric_linear_fem"
    assert state.lens_field_map_descriptors[key][FIELD] == recipe()
    page._reset_field()
    assert state.lens_field_map_descriptors[key] == {FIELD: recipe()}


def test_dormant_segments_are_not_presented_as_registered_maps(state, qtbot):
    from temsim.gui.model_inspector import ModelInspectorPage
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    key = state.lenses[0].key
    state.lens_field_map_descriptors[key] = {FIELD: recipe()}
    page.set_state(state)
    matching = [row for row in range(page.table.rowCount())
        if page.table.item(row, 0).text() == state.lenses[0].name]
    assert matching
    assert all(page.table.item(row, 2).text() != "Registered field map requested" for row in matching)


def test_operating_profile_preserves_dormant_current_cells(state, tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import save_profile, read_profile
    key = state.lenses[0].key
    state.lens_field_map_descriptors[key] = {FIELD: normalize_recipe(recipe())}
    selection = AssemblyCatalog().selection_for_resolved(state._resolved_assembly)
    path = tmp_path/"segmented-current-profile.toml"
    save_profile(path, state, selection)
    restored_selection, values = read_profile(path)
    assert restored_selection == selection
    assert values["__simulation_model__"]["settings"]["lens_field_map_descriptors"][key][FIELD] == recipe()
