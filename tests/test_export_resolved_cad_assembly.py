"""Small independent placement/identity checks; no CAD or simulator imports."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1]/"scripts/export_resolved_cad_assembly.py"
SPEC = importlib.util.spec_from_file_location("resolved_cad_export", PATH)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def _box(x=0., z=0.):
    return dict(kind="box", center_mm=[x, 0., z], size_mm=[2., 2., 2.])


def _solid(key, shape, **extra):
    return dict(key=key, region="body", shape=shape, material="metal",
                color=[.6, .7, .8, 1.], **extra)


def _fixture():
    mounted = dict(kind="transform", shape=_box(2., 3.),
        rotation=[[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], offset_mm=[0., 0., 17.])
    return dict(schema="temsim-material-collision-input-v1", units="mm", modules=[
        dict(key="recording:filter", name="Already-mounted branch",
             solids=[_solid("same_part", mounted)],
             units=[dict(key="filter_unit", solid_ids=["same_part/body"])],
             omitted=[dict(key="reference", reason="No material")]),
        dict(key="column:test", name="Column", solids=[_solid("same_part", _box(z=4.))]),
        dict(key="@physical_liners:fixture", name="Actual liner pieces", solids=[
            _solid("tube", _box(z=101.), instance_key="tube/body/segment-0000"),
            _solid("tube", _box(z=102.), instance_key="tube/body/segment-0001")])],
        assemblies=[dict(key=EXPORT.DEFAULT_ASSEMBLY, selection=dict(gun="FEG"),
            material_liner_scope="physical_adapter", liner_material_decisions=[dict(key="tube", kept=True)],
            placements=[dict(module_key="recording:filter", offset_mm=[0., 0., 100.]),
                        dict(module_key="column:test", offset_mm=[0., 0., 50.]),
                        dict(module_key="@physical_liners:fixture", offset_mm=[0., 0., 0.])])])


def _world_center(shape):
    """Independent column-vector composition, not the adapter's code path."""
    if shape["kind"] == "box":
        return shape["center_mm"]
    point = _world_center(shape["shape"])
    return [sum(a*b for a, b in zip(row, point))+d
            for row, d in zip(shape["rotation"], shape["offset_mm"])]


def test_global_translation_once_preserves_existing_branch_pose_and_original_input():
    document = _fixture()
    before = deepcopy(document)
    catalog = EXPORT.resolved_catalog(document, source_file_sha256="original-input-hash")
    solids = catalog["modules"][0]["solids"]
    assert _world_center(solids[0]["shape"]) == [3., 0., 115.]
    assert _world_center(solids[1]["shape"]) == [0., 0., 54.]
    assert [_world_center(s["shape"])[2] for s in solids[2:]] == [101., 102.]
    assert solids[0]["shape"]["shape"] == document["modules"][0]["solids"][0]["shape"]
    assert document == before
    assert catalog["provenance"]["source_file_sha256"] == "original-input-hash"
    assert catalog["modules"][0]["omitted"][0]["record"]["key"] == "reference"


def test_repeated_part_names_and_liner_segments_remain_distinct_and_owned_once():
    catalog = EXPORT.resolved_catalog(_fixture())
    module = catalog["modules"][0]
    ids = {s["key"]+"/"+s["region"] for s in module["solids"]}
    assigned = [i for u in module["units"] for i in u["solid_ids"]]
    assert len(ids) == 4 and len(assigned) == 4 and set(assigned) == ids
    assert len({u["key"] for u in module["units"]}) == 4
    assert [s["resolved_instance"]["generated_liner_instance_key"]
            for s in module["solids"][2:]] == ["tube/body/segment-0000", "tube/body/segment-0001"]


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["assemblies"][0].update(material_liner_scope="excluded_pending_physical_adapter"), "physical liner"),
    (lambda d: d["assemblies"][0]["placements"][-1].update(offset_mm=[0., 0., 1.]), "already global"),
    (lambda d: d["modules"][-1]["solids"][1].update(instance_key="tube/body/segment-0000"), "Duplicate material"),
    (lambda d: d["assemblies"][0]["placements"][0].update(offset_mm=[0., 0., float("nan")]), "finite"),
])
def test_reject_incomplete_or_ambiguous_prepared_assembly(mutation, match):
    document = _fixture()
    mutation(document)
    with pytest.raises(ValueError, match=match):
        EXPORT.resolved_catalog(document)
