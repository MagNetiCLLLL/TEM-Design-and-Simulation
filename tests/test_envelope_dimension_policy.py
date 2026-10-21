"""Only proven allocation references are locked; material controls stay live."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.cad_catalog import posed_solids
from temsim.dimension_edit_policy import dimension_reference_reason
from temsim.parameter_impact import describe_parameter_impact
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


COLUMN = "column/C3_ProbeCorrector.toml"
RECORDING = "project_and_recording_system/NoEnergyFilter.toml"
CASES = [
    ("gun/FEG.toml", "feg_accelerator", "length_mm", "stage_centers_z_mm"),
    ("gun/FEG.toml", "feg_deflector", "length_mm", "active_length_mm"),
    ("gun/FEG.toml", "feg_dpa_aperture", "length_mm", "active_length_mm"),
    ("gun/FEG.toml", "feg_c1_aperture", "length_mm", "active_length_mm"),
    ("gun/Thermionic.toml", "thermionic_anode_aperture", "length_mm", "active_length_mm"),
    (COLUMN, "condenser_aperture_2", "length_mm", "plate_thickness_mm"),
    (COLUMN, "condenser_aperture_3", "length_mm", "plate_thickness_mm"),
    (COLUMN, "condenser_aperture_2", "mechanical_outer_diameter_mm", "aperture_support_plate_half_width_mm"),
    (COLUMN, "condenser_aperture_3", "mechanical_outer_diameter_mm", "aperture_support_plate_half_width_mm"),
    *[(COLUMN, key, "length_mm", "mechanical_coil_length_mm") for key in
      ("condenser_deflector", "beam_deflector", "ac_deflector", "image_diffraction_deflector")],
    (COLUMN, "objective_aperture", "length_mm", "plate_thickness_mm"),
    (RECORDING, "selected_area_aperture", "length_mm", "plate_thickness_mm"),
    (COLUMN, "sample_stage", "length_mm", "stage_port"),
    (COLUMN, "sample_holder", "length_mm", "holder_tip_width_mm"),
    (COLUMN, "eds_detector_system", "length_mm", "detector"),
    ("project_and_recording_system/EnergyFilter.toml", "energy_filter", "length_mm", "child"),
    ("column/C3_BCOR.toml", "image_diffraction_deflector", "length_mm", "mechanical_coil_length_mm"),
]


@pytest.mark.parametrize("module,key,field,alternative", CASES)
def test_known_allocation_edit_is_readonly_and_rejected_before_draft_history_changes(module, key, field, alternative):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    before, history, source = deepcopy(draft.document), deepcopy(draft._history), draft.path.read_bytes()
    spec = next(s for s in part_dimension_specs(draft.document, key) if s.path == ("parts", key, field))
    assert not spec.editable and spec.meaning.category == "envelope"
    assert alternative in spec.reason
    if draft.part(key).get("aperture_plate_form") == "perforated_strip":
        assert spec.label == ("Mechanism envelope length" if field == "length_mm" else "Mechanism envelope diameter")
    else:
        assert spec.label == "Envelope length"
    impact = describe_parameter_impact(draft.part(key), spec.path)
    assert impact.status == "inactive" and "geometry" not in impact.effects
    with pytest.raises(ValueError):
        draft.set_dimension(spec.path, spec.value*1.01 if spec.value else .01)
    assert draft.document == before and draft._history == history and not draft.can_undo
    assert draft.path.read_bytes() == source


@pytest.mark.parametrize("module,key", [
    ("gun/FEG.toml", "feg_accelerator"),
    ("gun/FEG.toml", "feg_deflector"),
    (COLUMN, "condenser_aperture_2"),
    (COLUMN, "sample_holder"),
])
def test_native_copy_keeps_reference_meaning_but_explicit_replacement_length_changes_material(module, key):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    original = deepcopy(draft.part(key))
    copied = draft.copy_component_from(draft.document, key, "allocation_copy", 2500., include_children=False)
    assert dimension_reference_reason(draft.part(copied), "length_mm")
    snapshot, history = deepcopy(draft.document), deepcopy(draft._history)
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", copied, "length_mm"), draft.part(copied)["length_mm"]*1.01)
    assert draft.document == snapshot and draft._history == history
    draft.update_model_3d(copied, {"base": {"kind": "box", "width_mm": 8., "height_mm": 6., "length_mm": 1.}})
    assert not dimension_reference_reason(draft.part(copied), "length_mm")
    draft.set_dimension(("parts", copied, "length_mm"), 2.)
    native, = part_model_from_document(draft.document, copied, include_children=False).meshes
    np.testing.assert_allclose(np.ptp(native.vertices, axis=0), [8., 6., 2.])
    rows = {part["key"]: part for part in draft.document["parts"]}
    assert len(posed_solids(rows[copied], rows)) == 1
    assert draft.part(key) == original


@pytest.mark.parametrize("module,key", [(COLUMN, "objective_aperture"), (RECORDING, "selected_area_aperture")])
def test_schematic_strip_diameter_with_saturated_response_remains_editable(module, key):
    # The +1% scan is flat at OD80 because its schematic reach is clamped.
    # A smaller valid OD changes that contour, so this is not a dead reference.
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    field = "mechanical_outer_diameter_mm"
    spec = next(s for s in part_dimension_specs(draft.document, key) if s.path[2] == field)
    assert spec.editable and not dimension_reference_reason(draft.part(key), field)
    rows = {part["key"]: part for part in draft.document["parts"]}
    before = [s.shape for s in posed_solids(rows[key], rows)]
    native_before = part_model_from_document(draft.document, key, include_children=False).meshes
    draft.set_dimension(("parts", key, field), 20.)
    rows = {part["key"]: part for part in draft.document["parts"]}
    assert [s.shape for s in posed_solids(rows[key], rows)] != before
    native_after = part_model_from_document(draft.document, key, include_children=False).meshes
    assert any(not np.array_equal(a.vertices, b.vertices) for a, b in zip(native_before, native_after))
    draft.validate()


def test_legacy_deflector_without_declared_coil_centres_keeps_envelope_geometry_control():
    part = dict(key="unknown_deflector", mechanical_profile="magnetic_deflector",
                length_mm=24., local_start_z_mm=0., local_end_z_mm=24.)
    assert not dimension_reference_reason(part, "length_mm")
