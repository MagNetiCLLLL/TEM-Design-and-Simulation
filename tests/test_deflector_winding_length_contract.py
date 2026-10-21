"""Installation allocations and actual copper lengths remain distinct."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.cad_catalog import posed_solids
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.parameter_impact import describe_parameter_impact


@pytest.mark.parametrize("module", ("column/C3_BCOR.toml", "column/C3_ImageCorrector.toml"))
def test_all_explicit_image_windings_edit_copper_not_field_allocation(tmp_path, module):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    original = deepcopy(draft.document)
    expected = deepcopy(original)
    rows = {part["key"]: part for part in draft.document["parts"]}
    selected = [part["key"] for part in original["parts"] if "winding_axial_length_mm" in part]
    assert len(selected) >= 6
    for key in selected:
        dimensions = {spec.path[2]: spec for spec in part_dimension_specs(draft.document, key)}
        assert not dimensions["length_mm"].editable and dimensions["length_mm"].meaning.category == "envelope"
        assert dimensions["winding_axial_length_mm"].editable
        assert dimensions["winding_axial_length_mm"].meaning.category == "physical"
        snapshot, history = deepcopy(draft.document), deepcopy(draft._history)
        with pytest.raises(ValueError, match="allocation"):
            draft.set_dimension(("parts", key, "length_mm"), draft.part(key)["length_mm"]*1.01)
        assert draft.document == snapshot and draft._history == history
        before_cad = [spec.shape for spec in posed_solids(rows[key], rows)]
        before_native = part_model_from_document(draft.document, key, include_children=False).meshes
        value = draft.part(key)["winding_axial_length_mm"]*.99
        draft.set_dimension(("parts", key, "winding_axial_length_mm"), value)
        rows = {part["key"]: part for part in draft.document["parts"]}
        after_native = part_model_from_document(draft.document, key, include_children=False).meshes
        assert [spec.shape for spec in posed_solids(rows[key], rows)] != before_cad
        assert any(not np.array_equal(a.vertices, b.vertices) for a, b in zip(before_native, after_native))
        next(part for part in expected["parts"] if part["key"] == key)["winding_axial_length_mm"] = value
    # Includes fixed optical centres, effective thicknesses and all controls.
    assert draft.document == expected
    draft.validate()
    draft.save_copy(tmp_path/"winding-lengths.toml")
    # Save-copy intentionally flattens included subassemblies into this file.
    assert PartModelDocument(draft.path).document == {k: v for k, v in expected.items() if k != "subassemblies"}


def test_legacy_probe_without_explicit_copper_length_uses_its_saved_length():
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/"column/C3_ProbeCorrector.toml")
    key = "probe_dph2_coils"
    original = deepcopy(draft.part(key))
    assert "winding_axial_length_mm" not in original
    spec = next(s for s in part_dimension_specs(draft.document, key) if s.path[2] == "length_mm")
    assert spec.editable and spec.meaning.category == "physical"
    rows = {p["key"]: p for p in draft.document["parts"]}
    before = [s.shape for s in posed_solids(rows[key], rows)]
    draft.set_dimension(("parts", key, "length_mm"), original["length_mm"]*.99)
    rows = {p["key"]: p for p in draft.document["parts"]}
    assert [s.shape for s in posed_solids(rows[key], rows)] != before
    for field in ("local_center_z_mm", "effective_thickness_mm", "optical_reference_local_z_mm"):
        assert draft.part(key).get(field) == original.get(field)


@pytest.mark.parametrize("module", (
    "column/C3_ProbeCorrector.toml", "column/C3_BCOR.toml", "column/C3_ImageCorrector.toml"))
def test_explicit_winding_cap_edits_target_material_length_not_installation(module):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    selected = [row for row in draft.document["parts"] if "winding_axial_length_mm" in row]
    assert selected
    for row in selected:
        model = part_model_from_document(draft.document, row["key"], include_children=False)
        copper = [mesh for mesh in model.meshes if mesh.material_class == "copper_winding"]
        assert len(copper) == 4
        dimensions = {spec.path: spec for spec in model.fields}
        target = ("parts", row["key"], "winding_axial_length_mm")
        assert dimensions[target].editable
        for mesh in copper:
            caps = [surface for surface in mesh.surfaces.values() if surface["kind"] == "cap"]
            assert caps
            for surface in caps:
                assert target in surface["parameter_paths"]
                assert ("parts", row["key"], "length_mm") not in surface["parameter_paths"]


def test_legacy_winding_caps_keep_the_existing_material_length_path():
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/"column/C3_ProbeCorrector.toml")
    key = "probe_dph2_coils"
    model = part_model_from_document(draft.document, key, include_children=False)
    for mesh in model.meshes:
        if mesh.material_class != "copper_winding":
            continue
        caps = [surface for surface in mesh.surfaces.values() if surface["kind"] == "cap"]
        assert caps
        for surface in caps:
            assert ("parts", key, "length_mm") in surface["parameter_paths"]
            assert ("parts", key, "winding_axial_length_mm") not in surface["parameter_paths"]


@pytest.mark.parametrize("module", (
    "column/C3_ProbeCorrector.toml", "column/C3_BCOR.toml", "column/C3_ImageCorrector.toml"))
@pytest.mark.parametrize("mode", ("ideal", "analytical", "linear_geometry", "nonlinear_material", "custom"))
def test_explicit_winding_length_impact_reports_only_the_existing_geometry_route(module, mode):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    rows = {row["key"]: row for row in draft.document["parts"]}
    for row in rows.values():
        if "winding_axial_length_mm" not in row:
            continue
        impact = describe_parameter_impact(row, ("parts", row["key"], "winding_axial_length_mm"),
                                          by_key=rows, simulation_mode=mode)
        assert impact.status == "inactive" and impact.active is False
        assert impact.effects == ("display", "geometry")
        assert impact.affected_results == ("3D preview", "Physical layout")
        assert "fixed centre" in impact.detail and "effective field thickness" in impact.detail
        assert "does not establish unchanged real fields" in impact.detail


@pytest.mark.parametrize("module", ("column/C3_ProbeCorrector.toml", "column/C3_ImageCorrector.toml"))
def test_replacement_base_keeps_explicit_native_copper_length_inactive(module):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    row = next(row for row in draft.document["parts"] if "winding_axial_length_mm" in row)
    row["model_3d"] = {"schema_version": 1, "base": {
        "kind": "box", "width_mm": 2., "height_mm": 3., "length_mm": 4.}}
    impact = describe_parameter_impact(row, ("parts", row["key"], "winding_axial_length_mm"))
    assert impact.status == "inactive" and "geometry" not in impact.effects
    assert "replacement base" in impact.detail
    model = part_model_from_document(draft.document, row["key"], include_children=False)
    assert len(model.meshes) == 1
    for surface in model.meshes[0].surfaces.values():
        assert ("parts", row["key"], "winding_axial_length_mm") not in surface["parameter_paths"]
