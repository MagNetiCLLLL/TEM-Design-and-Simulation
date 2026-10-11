"""Real OCC import -> native editing -> portable TOML -> exact STEP round trip."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from temsim.cad_catalog import module_catalog
from temsim.part_model_document import PartModelDocument
from temsim.part_model_3d import part_model_from_document
from temsim.step_assets import canonical_json, read_asset

ROOT = Path(__file__).resolve().parents[1]
CAD = ROOT / "environment/cad/Scripts/python.exe"
pytestmark = pytest.mark.skipif(not CAD.exists(), reason="Optional isolated OpenCascade runtime is not installed")


def worker(folder, request):
    folder.mkdir(parents=True, exist_ok=True)
    source, output = folder / "request.json", folder / "response.json"
    source.write_text(canonical_json(request), encoding="utf-8")
    process = subprocess.run([str(CAD), str(ROOT / "scripts/cad_geometry_worker.py"), "--request", str(source),
                              "--response", str(output)], capture_output=True, text=True, timeout=90)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert process.returncode == 0 and receipt["status"] == "complete", receipt
    return receipt["result"]


@pytest.fixture(scope="module")
def asset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("real_step_source")
    source = folder / "source.step"
    worker(folder / "export", dict(op="export_step", path=str(source), solids=[
        dict(key="source", region="body", name="Copper body", color=[.8, .4, .2, 1.], material="copper",
             shape=dict(kind="box", center_mm=[3., -2., 20.], size_mm=[10., 8., 6.]))]))
    return worker(folder / "import", dict(op="import_step", path=str(source)))


def volume(mesh):
    triangles = mesh.vertices[mesh.faces] - mesh.vertices.mean(axis=0)
    return np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6


def test_import_edit_save_reopen_export_exact_volume_and_coordinates(asset, tmp_path):
    doc = PartModelDocument.from_step_asset(asset, tmp_path / "edited.toml", key="imported", name="Imported body")
    assert not doc.path.exists() and doc.dirty
    original = part_model_from_document(doc.document, "imported").meshes[0]
    np.testing.assert_allclose(original.vertices.min(axis=0), [-2, -6, 17], atol=1e-7)
    np.testing.assert_allclose(original.vertices.max(axis=0), [8, 2, 23], atol=1e-7)
    np.testing.assert_allclose(original.color, [.8, .4, .2, 1.], atol=2e-7)
    doc.set_dimension(("parts", "imported", "model_3d", "base", "width_mm"), 12.)
    doc.upsert_model_feature("imported", dict(id="hole", kind="hole", axis="z", center_mm=[-3., 0., 0.], diameter_mm=1., depth_mm=10.))
    doc.upsert_model_feature("imported", dict(id="slot", kind="slot", axis="z", center_mm=[2., 0., 0.], width_mm=1., length_mm=3., depth_mm=10.))
    edited = part_model_from_document(doc.document, "imported", angular_segments=192).meshes[0]
    expected = 12 * 8 * 6 - 6 * (2 * np.pi * .5**2 + 2.)
    assert volume(edited) == pytest.approx(expected, rel=5e-5)
    current = deepcopy(doc.document)
    doc.undo()
    assert len(doc.part("imported")["model_3d"]["features"]) == 1
    doc.redo()
    assert doc.document == current
    doc.save()
    restored = PartModelDocument(doc.path)
    assert restored.document == current and not doc.dirty
    catalog = module_catalog(restored.document)
    result = worker(tmp_path / "evaluate", dict(op="evaluate", solids=catalog["solids"]))
    assert result["metrics"]["volume_mm3"] == pytest.approx(expected, rel=1e-8)
    np.testing.assert_allclose(result["metrics"]["bounds_mm"], np.r_[edited.vertices.min(axis=0), edited.vertices.max(axis=0)], atol=1e-6)
    destination = tmp_path / "edited.step"
    worker(tmp_path / "export", dict(op="export_step", solids=catalog["solids"], path=str(destination)))
    roundtrip = worker(tmp_path / "reimport", dict(op="import_step", path=str(destination)))
    assert roundtrip["metrics"]["volume_mm3"] == pytest.approx(expected, rel=1e-8)
    restored.reset_model_geometry("imported")
    assert restored.part("imported")["model_3d"]["features"] == []
    assert restored.part("imported")["model_3d"]["base"]["width_mm"] == pytest.approx(10.)


def test_asset_checksum_rejects_before_mutating_a_draft(asset, tmp_path):
    doc = PartModelDocument.from_step_asset(asset, tmp_path / "draft.toml")
    before = deepcopy(doc.document)
    bad = deepcopy(asset)
    bad["leaves"][0]["shape"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        doc.import_step_asset(bad)
    assert doc.document == before
    with pytest.raises(ValueError, match="schema|units"):
        read_asset(canonical_json({**asset, "units": "m"}))


def test_new_document_never_overwrites_a_file_created_after_import(asset, tmp_path):
    doc = PartModelDocument.from_step_asset(asset, tmp_path / "draft.toml")
    doc.path.write_text("external data", encoding="utf-8")
    with pytest.raises(ValueError, match="destination now exists"):
        doc.save()
    assert doc.path.read_text() == "external data"


def test_import_replacement_keeps_reference_plane_and_mechanical_role(asset, tmp_path):
    from temsim.component_operations import make_component
    doc = PartModelDocument.from_step_asset(asset, tmp_path / "draft.toml")
    doc.add_component(make_component(key="host", shape="box", center_z_mm=120.))
    doc.import_step_asset(asset, mode="replace", target_key="host", registration="center")
    mesh = part_model_from_document(doc.document, "host").meshes[0]
    np.testing.assert_allclose((mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2, [0, 0, 120.])
    assert doc.part("host")["mechanical_only"] is True
    doc.add_component(make_component(key="child", parent_key="host"))
    before = deepcopy(doc.document)
    with pytest.raises(ValueError, match="material leaf"):
        doc.import_step_asset(asset, mode="replace", target_key="host")
    assert doc.document == before


def test_nonuniform_scale_rotated_cut_matches_exact_cad(asset, tmp_path):
    doc = PartModelDocument.from_step_asset(asset, tmp_path / "draft.toml")
    key = doc.document["parts"][0]["key"]
    doc.upsert_model_feature(key, dict(id="through_x", kind="slot", axis="x", center_mm=[0, 0, 0],
                                      width_mm=1., length_mm=3., depth_mm=30., rotation_deg=23.))
    doc.update_model_3d(key, {"transform": {"scale_xy": [1.2, .8], "rotation_deg": [10., 20., 30.], "offset_mm": [1., 2., 3.]}})
    mesh = part_model_from_document(doc.document, key, angular_segments=192).meshes[0]
    result = worker(tmp_path / "evaluate", dict(op="evaluate", solids=module_catalog(doc.document)["solids"]))
    assert volume(mesh) == pytest.approx(result["metrics"]["volume_mm3"], rel=5e-5)
    np.testing.assert_allclose(result["metrics"]["bounds_mm"], np.r_[mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)], atol=2e-6)


def test_standalone_save_rejects_noop_cut_without_creating_file(asset, tmp_path):
    doc = PartModelDocument.from_step_asset(asset, tmp_path / "invalid.toml")
    key = doc.document["parts"][0]["key"]
    doc.upsert_model_feature(key, dict(id="miss", kind="hole", axis="z", center_mm=[500., 0, 0],
                                      diameter_mm=1., depth_mm=10.))
    with pytest.raises(ValueError, match="does not remove"):
        doc.save()
    assert not doc.path.exists()


def test_replace_physical_dph_coil_keeps_its_nonmaterial_channel(asset):
    from temsim.paths import INSTRUMENT_CONFIG_ROOT
    from temsim.component_representation import non_material_role
    doc = PartModelDocument(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    channel = next(p for p in doc.document["parts"] if "dph" in p["key"] and p.get("parent_key") and non_material_role(p))
    before = deepcopy(channel)
    host = channel["parent_key"]
    doc.import_step_asset(asset, mode="replace", target_key=host, registration="center")
    assert doc.part(channel["key"]) == before
    assert doc.part(host)["model_3d"]["base"]["kind"] == "step"
