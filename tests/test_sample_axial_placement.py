"""Movable specimen foil inside fixed objective and stage geometry; no transport."""
from copy import deepcopy

import pytest

from temsim import module_manifest
from temsim.component_operations import translated_component
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


PATH = INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml"


def _rows(document):
    return {row["key"]: row for row in document["parts"]}


def _moved_document(delta):
    draft = PartModelDocument(PATH)
    draft.place_component("sample", draft.part("sample")["local_center_z_mm"]+delta,
                          include_children=False)
    return draft.document


def test_default_objective_remains_centred_without_any_definition_change():
    original = PATH.read_bytes()
    document = module_manifest.read_document(PATH)
    rows = _rows(document)
    centre = (rows["objective_upper_pole"]["local_end_z_mm"]
              +rows["objective_lower_pole"]["local_start_z_mm"])/2
    assert rows["sample"]["local_center_z_mm"] == centre
    assert rows["sample_stage"]["local_center_z_mm"] == centre
    module_manifest.validate_document(document)
    assert PATH.read_bytes() == original


@pytest.mark.parametrize("delta", [4.8e-6, -4.8e-6])
def test_sample_place_moves_only_foil_and_keeps_stage_poles_fields_and_eds_fixed(delta):
    original = PATH.read_bytes()
    before = _rows(PartModelDocument(PATH).document)
    document = _moved_document(delta)
    after = _rows(document)
    module_manifest.validate_document(document)
    assert before.keys() == after.keys()
    for key in before.keys()-{"sample"}:
        assert before[key] == after[key], key
    changes = {name for name, value in before["sample"].items() if after["sample"][name] != value}
    assert changes == {"local_start_z_mm", "local_center_z_mm", "local_end_z_mm",
                       "optical_reference_local_z_mm"}
    for name in changes:
        assert after["sample"][name]-before["sample"][name] == pytest.approx(delta, abs=3e-13)
    assert document["subassemblies"][0]["placement"] == {
        "mode": "fixed", "origin_z_mm": before["sample_stage"]["local_center_z_mm"]}
    assert PATH.read_bytes() == original


def test_placed_sample_roundtrips_through_normal_independent_module_copy(tmp_path):
    draft = PartModelDocument(PATH)
    draft.place_component("sample", draft.part("sample")["local_center_z_mm"]+4.8e-6,
                          include_children=False)
    expected = deepcopy(draft.document)
    destination = tmp_path / "offset-column.toml"
    draft.save_copy(destination)
    loaded = PartModelDocument(destination).document
    module_manifest.validate_document(loaded)
    assert _rows(loaded) == _rows(expected)


@pytest.mark.parametrize("distance", [.7, 2.8])
def test_sample_cannot_leave_fixed_holder_or_pole_gap(distance):
    document = module_manifest.read_document(PATH)
    document["parts"] = [translated_component(row, distance) if row["key"] == "sample" else row
                         for row in document["parts"]]
    with pytest.raises(ValueError, match="holder tip|stage and Objective pole gap|downstream|sample"):
        module_manifest.validate_document(document)


@pytest.mark.parametrize("change", ["stage", "yokes", "field"])
def test_moving_foil_does_not_move_the_objective_symmetry_datum(change):
    document = _moved_document(4.8e-6)
    rows = _rows(document)
    if change == "stage":
        rows["sample_stage"]["local_center_z_mm"] += 4.8e-6
    elif change == "yokes":
        for name in ("upper_yoke_start_local_z_mm", "upper_yoke_end_local_z_mm",
                     "lower_yoke_start_local_z_mm", "lower_yoke_end_local_z_mm"):
            rows["objective_lens"][name] += 4.8e-6
    else:
        rows["objective_lens"]["upper_field_reference_local_z_mm"] += 4.8e-6
        rows["objective_lens"]["lower_field_reference_local_z_mm"] += 4.8e-6
    with pytest.raises(ValueError, match="fixed sample stage|fixed centre|pole-gap centre"):
        module_manifest.validate_document(document)
