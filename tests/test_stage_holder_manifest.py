"""Native shared holder storage, unchanged beam context and strict dimensions."""
from copy import deepcopy
from pathlib import Path
import tomllib

import pytest

from temsim import module_manifest
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.shared_tip import copy_catalog_tree, dependencies
from temsim.subassemblies import resolve_document, storage_documents


COLUMNS = tuple(sorted((INSTRUMENT_CONFIG_ROOT / "column").glob("*.toml")))


@pytest.mark.parametrize("path", COLUMNS, ids=lambda path: path.stem)
def test_shared_stage_resolves_at_existing_sample_without_duplicate_storage(path):
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    assert not {"sample_stage", "sample_holder"} & {p["key"] for p in raw["parts"]}
    document = module_manifest.validate_document(module_manifest.read_document(path))
    rows = {p["key"]: p for p in document["parts"]}
    stage, holder, sample = (rows[k] for k in ("sample_stage", "sample_holder", "sample"))
    assert stage["local_center_z_mm"] == holder["local_center_z_mm"] == sample["local_center_z_mm"]
    assert stage["local_start_z_mm"] == pytest.approx(sample["local_center_z_mm"]-2.7)
    assert stage["local_end_z_mm"] == pytest.approx(sample["local_center_z_mm"]+2.7)
    assert stage["vacuum_inner_diameter_mm"] == 5.0
    assert not stage.get("axial_vacuum_context_only", False)
    assert holder["axial_vacuum_context_only"] is True
    assert stage["mechanical_only"] is holder["mechanical_only"] is True
    assert sample["parent_key"] == holder["parent_key"] == "sample_stage"
    assert {source.name for source in dependencies(path)} == {"sample_stage.toml"}
    # Storage round trips the composition; independent field/EDS resolvers
    # may materialize additional runtime metadata during full validation.
    restored = storage_documents(raw, resolve_document(raw, path), path)
    for source, data in restored.items():
        assert data == tomllib.loads(source.read_text(encoding="utf-8"))


@pytest.mark.parametrize("key,field,value", [
    ("sample_holder", "holder_shaft_diameter_mm", 9.0),
    ("sample_holder", "holder_tip_opening_diameter_mm", 3.0),
    ("sample_holder", "holder_tip_seat_depth_mm", 1.2),
    ("sample_holder", "holder_retraction_mm", -1.0),
    ("sample_holder", "holder_retraction_mm", 401.0),
    ("sample_holder", "holder_shaft_start_x_mm", float("nan")),
    ("sample_holder", "holder_tip_start_x_mm", -3.0),
    ("sample_holder", "holder_insertion_axis", "y"),
    ("sample_holder", "holder_insertion_direction", 1),
    ("sample_stage", "housing_side_port_diameter_mm", 12.0),
    ("sample_stage", "transverse_envelope_y_mm", 500.0),
])
def test_invalid_holder_dimensions_are_rejected(key, field, value):
    document = module_manifest.read_document(COLUMNS[0])
    row = next(p for p in document["parts"] if p["key"] == key)
    row[field] = value
    with pytest.raises(ValueError):
        module_manifest.validate_document(document)


def test_legacy_embedded_stage_remains_readable():
    document = deepcopy(module_manifest.read_document(COLUMNS[0]))
    document.pop("subassemblies")
    document["parts"] = [p for p in document["parts"] if p["key"] != "sample_holder"]
    rows = {p["key"]: p for p in document["parts"]}
    rows["sample_stage"].pop("mechanical_only")
    rows["sample_stage"]["order"] = rows["objective_lens"]["order"]+1
    module_manifest.validate_document(document)


def test_shared_holder_edit_saves_once_and_reloads_for_other_column(tmp_path):
    root = tmp_path / "instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    path = root / "column/C2.toml"
    draft = PartModelDocument(path)
    original = path.read_bytes()
    sources = dependencies(path)
    draft.set_dimension(("parts", "sample_holder", "holder_retraction_mm"), 25.0)
    draft.save()
    assert path.read_bytes() == original
    assert [p.name for p, data in sources.items() if p.read_bytes() != data] == ["sample_stage.toml"]
    other = PartModelDocument(root / "column/C3_ProbeCorrector.toml")
    assert other.part("sample_holder")["holder_retraction_mm"] == 25.0
    assert other.part("sample_stage")["local_center_z_mm"] == other.part("sample")["local_center_z_mm"]
