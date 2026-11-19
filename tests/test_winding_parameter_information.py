"""Pure parameter explanations for explicit winding inputs; no field queries."""
from copy import deepcopy

import pytest

from temsim.parameter_impact import describe_parameter_impact
from temsim.parameter_semantics import describe_parameter


@pytest.fixture
def parts():
    return {
        "lens": dict(key="lens", mechanical_profile="magnetic_lens_assembly",
            air_core_reference_ampere_turns=3000., ideal_coil_current_a_at_100_percent=2.5),
        "winding": dict(key="winding", parent_key="lens", mechanical_profile="magnetic_excitation_coil",
            winding_turn_count=100, winding_wire_diameter_mm=.2),
    }


@pytest.mark.parametrize("key,field,unit,category", [
    ("winding", "winding_turn_count", "turn", "physical"),
    ("winding", "winding_wire_diameter_mm", "mm", "physical"),
    ("winding", "winding_wire_cross_section_mm2", "mm²", "physical"),
    ("lens", "ideal_coil_current_a_at_100_percent", "A", "operating"),
])
def test_new_inputs_have_explicit_units_and_user_provenance(parts, key, field, unit, category):
    if field == "winding_wire_cross_section_mm2":
        parts[key].pop("winding_wire_diameter_mm",None)
        parts[key][field] = .1
    before = deepcopy(parts)
    information = describe_parameter(parts[key], ("parts", key, field), by_key=parts)
    assert (information.unit, information.category) == (unit, category)
    assert information.source_kind == "user_defined"
    assert "OEM" in information.source_note
    assert parts == before


def impact(parts, key, field, *, selected=True):
    return describe_parameter_impact(parts[key], ("parts", key, field), by_key=parts,
        simulation_mode="unified", round_lens_settings={"round_lens_field_model": "copper_coil" if selected else "gaussian"})


def test_wire_geometry_does_not_shrink_homogeneous_field_source(parts):
    result = impact(parts, "winding", "winding_wire_diameter_mm")
    assert result.active is False
    assert "magnetic_field" not in result.effects
    assert "insulation" in result.detail and "does not compress" in result.detail
    assert {"3D preview", "Physical layout", "Input validation"} <= set(result.affected_results)


def test_metal_area_excludes_insulation_and_changes_only_cad_packing(parts):
    field = "winding_wire_cross_section_mm2"
    parts["winding"].pop("winding_wire_diameter_mm")
    parts["winding"][field] = .1
    information = describe_parameter(parts["winding"], ("parts", "winding", field), by_key=parts)
    assert "one circular turn" in information.description
    assert "distinct from the winding-pack area" in information.description
    assert "insulation" in information.description
    for selected in (False, True):
        result = impact(parts, "winding", field, selected=selected)
        assert result.active is False
        assert "geometry" in result.effects and "magnetic_field" not in result.effects
        assert "automatically filled" in result.detail
        assert "does not compress" in result.detail


def test_metal_area_has_editable_dimensions_and_diameter_is_only_a_readback():
    from temsim.part_model_3d import part_dimension_specs
    part = dict(key="coil", mechanical_profile="magnetic_excitation_coil",
                winding_turn_count=12,winding_wire_cross_section_mm2=.1)
    specs = part_dimension_specs(dict(parts=[part]),"coil")
    turns = next(item for item in specs if item.path[-1]=="winding_turn_count")
    area = next(item for item in specs if item.path[-1]=="winding_wire_cross_section_mm2")
    diameter = next(item for item in specs if item.path[-1]=="winding_wire_diameter_mm")
    assert area.editable and area.unit == "mm²"
    assert turns.editable and turns.value == 12 and turns.unit == "turn"
    assert not diameter.editable and diameter.path[0] == "derived"
    assert "Computed equivalent diameter" in diameter.meaning.source_note
    legacy = dict(part);legacy.pop("winding_wire_cross_section_mm2");legacy["winding_wire_diameter_mm"] = .2
    legacy_specs = part_dimension_specs(dict(parts=[legacy]),"coil")
    legacy_diameter = next(item for item in legacy_specs if item.path[-1]=="winding_wire_diameter_mm")
    assert not legacy_diameter.editable and "metal cross-section" in legacy_diameter.reason


def test_actual_dimensions_table_exposes_turn_and_area_edits_but_not_diameter(qtbot, tmp_path, monkeypatch):
    from pathlib import Path
    from PySide6.QtCore import Qt
    from temsim.gui.part_model_editor import PartModelEditorPage
    from temsim.part_model_document import PartModelDocument
    source = Path(__file__).resolve().parents[1]/'configs/instruments/project_and_recording_system/NoEnergyFilter.toml'
    coil, owner = 'intermediate_lens_excitation_coil', 'intermediate_lens'
    document = PartModelDocument(source)
    current = document.part(owner)['ideal_coil_current_a_at_100_percent']
    document.configure_winding(coil,winding_turn_count=12,winding_interval_turn_counts=[12],
        winding_wire_cross_section_mm2=.1,ideal_coil_current_a_at_100_percent=current)
    path = tmp_path/'area-dimensions.toml'; document.save_copy(path)
    monkeypatch.setattr(PartModelEditorPage,'_render',lambda *args,**kwargs: True)
    monkeypatch.setattr(PartModelEditorPage,'_queue_overlap_check',lambda *args,**kwargs: None)
    page = PartModelEditorPage(); qtbot.addWidget(page)
    assert page.open_path(path,selected_key=coil)
    def value_item(field):
        return next(page.dimensions.item(row,1) for row in range(page.dimensions.rowCount())
                    if page.dimensions.item(row,1).data(Qt.ItemDataRole.UserRole)[-1] == field)
    before = deepcopy(page.session.part(coil))
    turns = value_item('winding_turn_count')
    assert turns.flags() & Qt.ItemFlag.ItemIsEditable
    assert value_item('winding_wire_cross_section_mm2').flags() & Qt.ItemFlag.ItemIsEditable
    assert not value_item('winding_wire_diameter_mm').flags() & Qt.ItemFlag.ItemIsEditable
    turns.setText('13')
    assert page.session.part(coil)['winding_turn_count'] == 13
    assert page.session.part(coil)['winding_interval_turn_counts'] == [13]
    value_item('winding_wire_cross_section_mm2').setText('0.2')
    assert page.session.part(coil)['winding_wire_cross_section_mm2'] == .2
    assert page.session.part(owner)['ideal_coil_current_a_at_100_percent'] == current
    assert all(page.session.part(coil)[field] == before[field] for field in
               ('mechanical_inner_diameter_mm','mechanical_outer_diameter_mm','local_start_z_mm','local_end_z_mm'))


@pytest.mark.parametrize("key,field", [("winding", "winding_turn_count"), ("lens", "ideal_coil_current_a_at_100_percent")])
def test_turns_and_current_are_active_only_for_selected_ideal_current_model(parts, key, field):
    selected = impact(parts, key, field)
    assert selected.active and "magnetic_field" in selected.effects
    assert "homogeneous current source" in selected.detail
    inactive = impact(parts, key, field, selected=False)
    assert inactive.active is False and "magnetic_field" not in inactive.effects


def test_declared_turns_without_current_keep_legacy_ampere_turns(parts):
    del parts["lens"]["ideal_coil_current_a_at_100_percent"]
    result = impact(parts, "winding", "winding_turn_count")
    assert result.active is False and "legacy total-ampere-turn" in result.detail


def test_explicit_current_disables_only_legacy_ampere_turn_fallback(parts):
    result = impact(parts, "lens", "air_core_reference_ampere_turns")
    assert result.active is False and "not added again" in result.detail
    assert parts["lens"]["air_core_reference_ampere_turns"] == 3000.


def test_explicit_current_disables_legacy_winding_share(parts):
    parts["winding"]["coil_ampere_turn_fraction"] = 1.
    result = impact(parts, "winding", "coil_ampere_turn_fraction")
    assert result.active is False and "not added again" in result.detail


def test_explicit_ideal_current_does_not_disable_fem_current_share_consumer(parts):
    parts["winding"]["coil_ampere_turn_fraction"] = 1.
    result = describe_parameter_impact(parts["winding"], ("parts", "winding", "coil_ampere_turn_fraction"),
        by_key=parts, simulation_mode="linear_geometry", descriptors={"lens": dict(
            solver="axisymmetric_linear_fem", relative_permeability=1000., ampere_turns=100.,
            radial_nodes=16, axial_nodes=24)})
    assert result.active is True and "magnetic_field" in result.effects


def test_valid_declared_provenance_is_not_replaced_by_user_defined_label(parts):
    parts["winding"]["parameter_metadata"] = {
        "winding_turn_count": dict(source_kind="documented", source_note="Declared winding specification supplied by the user.")}
    result = describe_parameter(parts["winding"], ("parts", "winding", "winding_turn_count"), by_key=parts)
    assert result.source_kind == "documented"
