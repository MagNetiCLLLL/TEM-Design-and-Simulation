"""B-COR is an additional three-hexapole topology, with one CAD definition."""
from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.cad_catalog import module_catalog, posed_solids
from temsim.cad_mesh import meshes_from_solids
from temsim.module_manifest import read_document, validate_document


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (("C3_ImageCorrector.toml", "C3_BCOR.toml"),
            ("C3_ProbeCorrector_ImageCorrector.toml", "C3_ProbeCorrector_BCOR.toml"))
SEQUENCE = ("image_tl11_lens", "image_tl12_lens", "image_hp1_hexapole",
            "image_tl21_lens", "image_tl22_lens", "image_hp2_hexapole",
            "image_tl31_lens", "image_tl32_lens", "image_hp3_hexapole",
            "image_adapter_lens", "image_sad_plane")


def raw(filename):
    path = ROOT / "configs/instruments/column" / filename
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("baseline,filename", VARIANTS)
def test_bcor_is_valid_separate_module_and_keeps_upstream_parameters(baseline, filename):
    original, document = raw(baseline), raw(filename)
    validate_document(read_document(ROOT / "configs/instruments/column" / filename))
    assert document["module"]["image_corrector_design"] == "bcor"
    assert "not_aberration_qualified" in document["module"]["corrector_qualification"]
    old, new = ({row["key"]: row for row in doc["parts"]} for doc in (original, document))
    for key, part in old.items():
        if key.startswith("image_"):
            continue
        before, after = deepcopy(part), deepcopy(new[key])
        before.pop("order"); after.pop("order")
        assert before == after, key
    assert document["ports"]["exit"]["local_z_mm"] == pytest.approx(
        original["ports"]["exit"]["local_z_mm"] + 144.)
    z = [new[key]["optical_reference_local_z_mm"] for key in SEQUENCE]
    assert np.all(np.diff(z) > 0.)
    assert new["image_hp2_hexapole"]["local_center_z_mm"] - new["image_hp1_hexapole"]["local_center_z_mm"] == pytest.approx(144.)
    assert new["image_hp3_hexapole"]["local_center_z_mm"] - new["image_hp2_hexapole"]["local_center_z_mm"] == pytest.approx(144.)
    assert "image_dph3_deflector" not in new and "image_dph3_coils" not in new


@pytest.mark.parametrize("baseline,filename", VARIANTS)
def test_new_relay_keeps_complete_lens_material_and_explicit_hp_field(filename, baseline):
    document = raw(filename)
    rows = {row["key"]: row for row in document["parts"]}
    for key in ("image_tl31_lens", "image_tl32_lens"):
        part = rows[key]
        children = {row["key"] for row in rows.values() if row.get("parent_key") == key}
        assert children == {key + suffix for suffix in
            ("_upper_pole", "_lower_pole", "_housing", "_yoke", "_excitation_coil")}
        for child in children:
            assert posed_solids(rows[child], rows)
        assert part["mechanical_outer_diameter_mm"] == 180.
        assert part["field_half_width_mm"] > 0.
        assert "not CEOS manufacturing" in part["bcor_geometry_source"]
    hp = rows["image_hp3_hexapole"]
    assert 0 < abs(hp["default_strength_m3"]) <= hp["maximum_strength_m3"]
    assert np.isfinite(hp["orientation_rad"])
    specs = posed_solids(hp, rows)
    assert len(specs) == 13
    assert len([s for s in specs if s.region.startswith("pole_pin_")]) == 6
    assert len([s for s in specs if s.region.startswith("coil_")]) == 6
    assert all("B-COR" in s.note and "not verified" in s.note for s in specs)
    meshes = meshes_from_solids(specs)
    assert meshes and all(np.isfinite(mesh.vertices).all() for mesh in meshes)


def test_bcor_step_and_native_catalog_share_new_hardware():
    document = raw("C3_BCOR.toml")
    catalog = module_catalog(document)
    unit_keys = {unit["key"] for unit in catalog["units"]}
    assert {"image_tl31_lens", "image_tl32_lens", "image_hp3_hexapole"} <= unit_keys
    rows = {row["key"]: row for row in document["parts"]}
    specs = posed_solids(rows["image_hp3_hexapole"], rows)
    saved = {row["region"]: row for row in catalog["solids"] if row["key"] == "image_hp3_hexapole"}
    assert {spec.region for spec in specs} == set(saved)
    for spec in specs:
        assert spec.shape == saved[spec.region]["shape"]


@pytest.mark.parametrize("key", ("image_hp1_hexapole", "image_hp2_hexapole", "image_hp3_hexapole"))
@pytest.mark.parametrize("field,value", (
    ("default_strength_m3", float("nan")),
    ("default_strength_m3", float("inf")),
    ("default_strength_m3", True),
    ("default_strength_m3", 1.e10),
    ("default_strength_m3", -1.e10),
    ("maximum_strength_m3", float("nan")),
    ("maximum_strength_m3", float("inf")),
    ("maximum_strength_m3", 0.),
    ("maximum_strength_m3", -1.),
    ("maximum_strength_m3", True),
    ("maximum_strength_m3", 10 ** 400),
    ("orientation_rad", float("nan")),
    ("orientation_rad", float("inf")),
    ("orientation_rad", True),
    ("orientation_rad", "0"),
))
def test_bcor_rejects_invalid_selected_main_hexapole_fields(key, field, value):
    document = read_document(ROOT / "configs/instruments/column/C3_BCOR.toml")
    row = next(part for part in document["parts"] if part["key"] == key)
    row[field] = value
    with pytest.raises(ValueError, match=key + r"\." + field):
        validate_document(document)


@pytest.mark.parametrize("field", ("default_strength_m3", "maximum_strength_m3", "orientation_rad"))
def test_bcor_requires_explicit_main_hexapole_fields(field):
    document = read_document(ROOT / "configs/instruments/column/C3_BCOR.toml")
    row = next(part for part in document["parts"] if part["key"] == "image_hp1_hexapole")
    row.pop(field)
    with pytest.raises(ValueError, match="image_hp1_hexapole" + r"\." + field):
        validate_document(document)


@pytest.mark.parametrize("factor", (-1., 0., 1.))
def test_bcor_accepts_finite_signed_strength_at_and_between_declared_bounds(factor):
    document = read_document(ROOT / "configs/instruments/column/C3_BCOR.toml")
    for row in document["parts"]:
        if row["key"] in ("image_hp1_hexapole", "image_hp2_hexapole", "image_hp3_hexapole"):
            row["default_strength_m3"] = factor * row["maximum_strength_m3"]
    validate_document(document)


@pytest.mark.parametrize("filename", ("C3_ImageCorrector.toml", "C3_ProbeCorrector_ImageCorrector.toml"))
def test_legacy_cetcor_keeps_implicit_hexapole_defaults(filename):
    document = read_document(ROOT / "configs/instruments/column" / filename)
    for row in document["parts"]:
        if row["key"] in ("image_hp1_hexapole", "image_hp2_hexapole"):
            assert "default_strength_m3" not in row
    validate_document(document)
