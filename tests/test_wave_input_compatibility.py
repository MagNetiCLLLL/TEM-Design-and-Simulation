"""Metadata-only graph compatibility; synthetic inputs and no propagation."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.immutable_json import json_digest, thaw_json
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.wave_input_compatibility import compare_wave_input_graphs
from test_particle_wave_source_io import boundary


ANCHOR = "upper_surface_reference_z_mm"
LATTICE = "lattice_reference_thickness_nm"


def graph():
    return {"schema": "complete-working-point-v1", "root": {"ref": 0}, "nodes": [
        {"type": "temsim.optics.model:State", "fields": ["sample", "lens"],
         "attributes": {"sample": {"ref": 1}, "lens": {"ref": 2}}},
        {"type": "temsim.optics.model:Sample", "fields": ["z_mm", "thickness_nm", "cif_path"],
         "attributes": {"z_mm": 100.000005, "thickness_nm": 10., "cif_path": "original.cif"}},
        {"type": "temsim.optics.condenser_lens:CondenserLensComponent", "fields": ["strength"],
         "attributes": {"strength": 1.25}}], "coordinate_conventions": {"scope": "Synthetic comparator fixture"}}


def with_anchor(original, anchor=None):
    current = deepcopy(original)
    sample = current["nodes"][1]
    sample["fields"].append(ANCHOR)
    old = original["nodes"][1]["attributes"]
    sample["attributes"][ANCHOR] = (old["z_mm"]-old["thickness_nm"]*.5e-6 if anchor is None else anchor)
    return current


def test_exact_graphs_preserve_the_original_strict_path():
    original = graph()
    before = json_digest(original)
    receipt = compare_wave_input_graphs(original, deepcopy(original))
    assert receipt["metadata_migrations"] == []
    assert receipt["physical_inputs_unchanged"] is True
    assert json_digest(original) == before


@pytest.mark.parametrize("old_explicit_none", [False, True])
def test_only_missing_or_none_anchor_migrates_without_mutating_either_graph(old_explicit_none):
    original = graph()
    if old_explicit_none:
        original["nodes"][1]["fields"].append(ANCHOR)
        original["nodes"][1]["attributes"][ANCHOR] = None
    current = with_anchor(graph())
    before = (json_digest(original), json_digest(current))
    receipt = compare_wave_input_graphs(original, current)
    row, = receipt["metadata_migrations"]
    assert row["field"] == ANCHOR and row["node_index"] == 1
    assert row["new_value_mm"] == 100.
    assert row["actual_center_z_mm"] == 100.000005 and row["actual_thickness_nm"] == 10.
    assert row["previous_attribute_present"] is old_explicit_none
    assert (json_digest(original), json_digest(current)) == before


@pytest.mark.parametrize("steps,accepted", [(4, True), (5, False), (-4, True), (-5, False)])
def test_anchor_migration_has_only_four_ulp_round_trip_tolerance(steps, accepted):
    original = graph()
    upper = 100.
    current = with_anchor(original, upper+steps*np.spacing(upper))
    if accepted:
        assert compare_wave_input_graphs(original, current)["physical_inputs_unchanged"]
    else:
        with pytest.raises(ValueError, match="old actual upper surface"):
            compare_wave_input_graphs(original, current)


@pytest.mark.parametrize("change", ["center", "thickness", "lens", "cif", "topology", "schema", "numeric_type"])
def test_initializing_anchor_cannot_hide_any_other_physical_or_schema_change(change):
    original = graph()
    current = with_anchor(original)
    if change == "center":
        current["nodes"][1]["attributes"]["z_mm"] += np.spacing(100.)
    elif change == "thickness":
        current["nodes"][1]["attributes"]["thickness_nm"] = np.nextafter(10., 11.)
    elif change == "lens":
        current["nodes"][2]["attributes"]["strength"] += 1e-12
    elif change == "cif":
        current["nodes"][1]["attributes"]["cif_path"] = "edited.cif"
    elif change == "topology":
        current["nodes"][0]["attributes"]["sample"] = {"ref": 2}
    elif change == "schema":
        current["nodes"][1]["fields"].reverse()
    else:
        current["nodes"][1]["attributes"]["thickness_nm"] = 10
    with pytest.raises(ValueError, match="beyond entrance-reference metadata"):
        compare_wave_input_graphs(original, current)


def test_existing_explicit_anchor_cannot_be_changed_even_by_one_ulp():
    original = with_anchor(graph())
    current = deepcopy(original)
    current["nodes"][1]["attributes"][ANCHOR] += np.spacing(100.)
    with pytest.raises(ValueError):
        compare_wave_input_graphs(original, current)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), True, "100", {"float": "nan"}])
def test_new_anchor_requires_a_finite_real_scalar(bad):
    with pytest.raises(ValueError):
        compare_wave_input_graphs(graph(), with_anchor(graph(), bad))


def test_metadata_is_not_whitelisted_on_a_different_component_type():
    original = graph()
    original["nodes"][1]["type"] = "temsim.optics.objective_lens:ObjectiveLensComponent"
    with pytest.raises(ValueError):
        compare_wave_input_graphs(original, with_anchor(original))


def test_validated_handoff_records_migration_without_relabelling_source_identity(boundary):
    from temsim.physics.mixed_stem_scan import _source_input_handoff
    old_inputs = InstrumentSnapshot(graph(), boundary.inputs.external_inputs, boundary.inputs.implementation)
    source = replace(boundary.source, input_digest=old_inputs.digest)
    record = thaw_json(boundary.record)
    record.update(input_digest=old_inputs.digest, source_digest=source.digest)
    imported = replace(boundary, source=source, inputs=old_inputs, record=record)
    current = InstrumentSnapshot(with_anchor(graph()), imported.inputs.external_inputs, "c"*64)
    old_digest = imported.source.digest
    receipt = _source_input_handoff(current, imported.source, imported)
    assert imported.source.digest == old_digest
    assert imported.source.input_digest == old_inputs.digest != current.digest
    assert receipt["particle_input_digest"] == old_inputs.digest
    assert receipt["particle_producer_implementation"] == old_inputs.implementation
    assert len(receipt["input_graph_compatibility"]["metadata_migrations"]) == 1
    assert receipt["original_particle_identity_preserved"] is True
    changed = InstrumentSnapshot(with_anchor(graph()), ({"role": "changed material", "path": "edited.cif",
        "sha256": "d"*64, "content_hex": None},), "c"*64)
    with pytest.raises(ValueError, match="captured physical inputs"):
        _source_input_handoff(changed, imported.source, imported)


def test_both_new_references_adopt_the_existing_geometry_and_lattice_only():
    original = graph()
    current = with_anchor(original)
    current["nodes"][1]["fields"].append(LATTICE)
    current["nodes"][1]["attributes"][LATTICE] = 10.
    before = (json_digest(original), json_digest(current))
    receipt = compare_wave_input_graphs(original, current)
    assert {row["field"] for row in receipt["metadata_migrations"]} == {ANCHOR, LATTICE}
    row = next(row for row in receipt["metadata_migrations"] if row["field"] == LATTICE)
    assert row["old_lattice_origin_depth_nm"] == row["new_lattice_origin_depth_nm"] == 5.
    assert (json_digest(original), json_digest(current)) == before
    current["nodes"][1]["attributes"][LATTICE] = 9.
    with pytest.raises(ValueError, match="lattice thickness"):
        compare_wave_input_graphs(original, current)


def test_lattice_reference_alone_can_migrate_but_an_existing_reference_cannot_change():
    original = graph()
    current = deepcopy(original)
    current["nodes"][1]["fields"].append(LATTICE)
    current["nodes"][1]["attributes"][LATTICE] = 10.
    assert len(compare_wave_input_graphs(original, current)["metadata_migrations"]) == 1
    changed = deepcopy(current)
    changed["nodes"][1]["attributes"][LATTICE] = np.nextafter(10., 11.)
    with pytest.raises(ValueError):
        compare_wave_input_graphs(current, changed)
