"""Declared material clearances without changing the axial solver boundary."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.cad_auxiliary import aperture_solids
from temsim.cad_geometry import transformed, box
from temsim.cad_mesh import _manifold
from temsim.cad_round_lenses import round_lens_solids
from temsim.cad_vacuum_tube import (
    CONTINUOUS_TUBE_KEY, PROFILE, SLOT_KEYS, CLEARANCE, SLOTS,
    decorate_physical_tube_row, tube_solids, validate_material_tube_geometry,
)
from temsim.lens_pose import effective_part_transform_mm
from temsim.module_manifest import read_document, validate_document


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = sorted((ROOT / "configs/instruments/column").glob("*.toml"))


def fixture(document, origin=450.):
    parts = [SimpleNamespace(key=p["key"], module_key="column", data=p,
                center_z_mm=origin+float(p["local_center_z_mm"])) for p in document["parts"]]
    module = SimpleNamespace(key="column", geometry=document["geometry"], parts=parts)
    g = document["geometry"]
    start, end, inner, outer = [float(g["c1_c2_objective_vacuum_tube_"+name])
        for name in ("start_z_mm", "end_z_mm", "inner_diameter_mm", "outer_diameter_mm")]
    row = dict(key=CONTINUOUS_TUBE_KEY, mechanical_profile="vacuum_liner",
               local_start_z_mm=start+origin, local_end_z_mm=end+origin,
               mechanical_inner_diameter_mm=inner, mechanical_outer_diameter_mm=outer)
    return row, SimpleNamespace(modules=[module], parts=parts)


def mat(shape):
    return _manifold(shape, 64)


@pytest.mark.parametrize("path", COLUMNS, ids=lambda p: p.stem)
def test_all_profiles_cover_original_tube_and_preserve_solver_and_parts(path):
    document = read_document(path)
    validate_document(document)
    before = deepcopy(document)
    row, assembly = fixture(document)
    original_row = deepcopy(row)
    decorated = decorate_physical_tube_row(row, assembly)
    spec, = tube_solids(decorated)
    assert document == before
    assert row == original_row
    assert decorated[PROFILE][0][0] == row["local_start_z_mm"]
    assert decorated[PROFILE][-1][0] == row["local_end_z_mm"]
    assert spec.parameters["inner_diameter_mm"] == 5.76
    assert row["mechanical_outer_diameter_mm"] == 19.2
    assert len(decorated[SLOTS]) == (1 if path.stem == "C2" else 2)
    assert spec.status == "engineering_reconstruction_not_oem"
    if path.stem != "C2":
        assert decorated[PROFILE][2:4] == [[830., 17.2], [1000., 17.2]]
        assert (18.-17.2)/2 == pytest.approx(.4)


def test_material_has_no_c3_pole_or_declared_aperture_overlap():
    document = read_document(COLUMNS[1])
    rows = {p["key"]: p for p in document["parts"]}
    row, assembly = fixture(document)
    tube = mat(tube_solids(decorate_physical_tube_row(row, assembly))[0].shape)
    for key in ("condenser_lens_3_upper_pole", "condenser_lens_3_lower_pole"):
        pole, = round_lens_solids(rows[key], rows)
        assert (tube ^ mat(transformed(pole.shape, offset=(0.,0.,450.)))).volume() < 1e-9
    for key in document["geometry"][SLOT_KEYS]:
        aperture, = aperture_solids(rows[key])
        assert (tube ^ mat(transformed(aperture.shape, offset=(0.,0.,450.)))).volume() < 1e-9
    z = 450. + rows["condenser_aperture_2"]["local_center_z_mm"]
    # Explicit +Y insertion, not indiscriminate removal of the entire wall.
    assert (tube ^ mat(box((0.,9.4,z), (.05,.05,.05)))).volume() < 1e-12
    assert (tube ^ mat(box((0.,-9.4,z), (.05,.05,.05)))).volume() > 0.
    assert (tube ^ mat(box((9.4,0.,z), (.05,.05,.05)))).volume() > 0.


def test_aperture_slot_follows_same_pose_and_module_origin():
    document = read_document(COLUMNS[1])
    rows = {p["key"]: p for p in document["parts"]}
    part = rows["condenser_aperture_2"]
    part.update(offset_x_mm=.3, offset_y_mm=-.15, rotation_x_mrad=20., rotation_z_mrad=150.)
    row, assembly = fixture(document, origin=725.)
    decorated = decorate_physical_tube_row(row, assembly)
    slot = next(s for s in decorated[SLOTS] if s["aperture_key"] == part["key"])
    rotation, translation = effective_part_transform_mm(part, rows)
    np.testing.assert_array_equal(slot["rotation_local_to_global"], rotation)
    np.testing.assert_array_equal(slot["translation_local_mm"], translation)
    assert slot["module_origin_z_mm"] == 725.
    assert not np.array_equal(rotation, np.eye(3))
    aperture, = aperture_solids(part)
    placed = transformed(transformed(aperture.shape, rotation=rotation.tolist(),
                         offset=translation.tolist()), offset=(0.,0.,725.))
    assert (mat(tube_solids(decorated)[0].shape) ^ mat(placed)).volume() < 1e-9


@pytest.mark.parametrize("change", [
    lambda g: g[PROFILE].__setitem__(1, [0.,19.2]),
    lambda g: g[PROFILE].__setitem__(0, [1.,19.2]),
    lambda g: g[PROFILE].__setitem__(2, [380.,5.76]),
    lambda g: g[PROFILE].__setitem__(2, [380.,19.3]),
    lambda g: g[PROFILE].__setitem__(2, [380.,18.]),
    lambda g: g.__setitem__(CLEARANCE, 0.),
    lambda g: g.__setitem__(CLEARANCE, float("nan")),
    lambda g: g.__setitem__(SLOT_KEYS, ["objective_lens"]),
    lambda g: g.__setitem__(SLOT_KEYS, ["objective_aperture"]),
    lambda g: g.__setitem__(SLOT_KEYS, ["condenser_aperture_2"]*2),
])
def test_invalid_profile_and_undeclared_or_outside_strip_entries_are_rejected(change):
    document = read_document(COLUMNS[1])
    change(document["geometry"])
    with pytest.raises(ValueError):
        validate_document(document)


def test_legacy_and_other_tubes_do_not_invent_material_features():
    document = read_document(COLUMNS[1])
    document["geometry"] = {k:v for k,v in document["geometry"].items()
                             if not k.startswith("mechanical_material_tube_")}
    validate_material_tube_geometry(document["parts"], document["geometry"])
    row, assembly = fixture(document)
    assert decorate_physical_tube_row(row, assembly) is row
    assert tube_solids(row) is None
    assert decorate_physical_tube_row(row, SimpleNamespace(parts=[], modules=[])) is row
    row["key"] = "@vacuum_liner:@vacuum_drift:test"
    assert decorate_physical_tube_row(row, assembly) is row
