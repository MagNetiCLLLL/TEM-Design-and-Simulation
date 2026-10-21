"""Installation allocations and actual copper lengths remain distinct."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.cad_catalog import posed_solids
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


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
