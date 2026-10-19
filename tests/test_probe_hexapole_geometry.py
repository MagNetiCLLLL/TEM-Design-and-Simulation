"""Material-topology checks for the source-backed probe hexapole reconstruction."""
from copy import deepcopy
from itertools import combinations
import math
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.cad_mesh import _manifold
from temsim.cad_multipoles import multipole_solids
from temsim.probe_hexapole_geometry import PREFIX, probe_hexapole_dimensions


KEYS = ("probe_hp1_hexapole", "probe_hp2_hexapole")
CONFIG = Path(__file__).resolve().parents[1] / "configs/instruments/column/C3_ProbeCorrector.toml"


def _row(key):
    document = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    return next(row for row in document["parts"] if row["key"] == key)


def _bodies(row):
    # A multiple of six gives all six positions the same tessellation accuracy.
    return {solid.region: _manifold(solid.shape, 96) for solid in multipole_solids(row, {})}


def _cylindrical_interface_area(body, radius):
    """Area of mesh faces whose vertices lie on the declared yoke bore.

    This avoids requiring a boolean union to weld rotated coplanar surfaces:
    an actual area of radial interface is stronger evidence than zero gap at
    one point and remains independent of how the pin was constructed.
    """
    mesh = body.to_mesh64()
    vertices = np.asarray(mesh.vert_properties)[:, :3]
    faces = np.asarray(mesh.tri_verts)
    radial = np.linalg.norm(vertices[:, :2], axis=1)
    interface = np.all(np.abs(radial[faces] - radius) < 1e-7, axis=1)
    triangles = vertices[faces[interface]]
    return np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2


@pytest.fixture(scope="module", params=KEYS)
def host(request):
    row = _row(request.param)
    return row, probe_hexapole_dimensions(row), _bodies(row)


def test_saved_hosts_preserve_material_identity_and_open_envelope(host):
    row, d, bodies = host
    before = deepcopy(row)
    solids = multipole_solids(row, {})
    assert row == before
    assert set(bodies) == ({"return_yoke"} | {f"pole_pin_{i}" for i in range(1, 7)}
                           | {f"coil_{i}" for i in range(1, 7)})
    assert {s.material for s in solids} == {"soft_magnetic", "copper_winding"}
    for body in bodies.values():
        assert body.volume() > 0
        vertices = np.asarray(body.to_mesh64().vert_properties)[:, :3]
        radial = np.linalg.norm(vertices[:, :2], axis=1)
        assert radial.min() >= d["inner_radius_mm"] - 1e-7
        assert radial.max() <= d["outer_radius_mm"] + 1e-7
        assert vertices[:, 2].min() >= d["start_mm"] - 1e-7
        assert vertices[:, 2].max() <= d["end_mm"] + 1e-7
    yoke_box = bodies["return_yoke"].bounding_box()
    assert yoke_box[2] == pytest.approx(row["local_start_z_mm"])
    assert yoke_box[5] == pytest.approx(row["local_end_z_mm"])
    assert yoke_box[3] - yoke_box[0] == pytest.approx(row["mechanical_outer_diameter_mm"])


def test_rotational_copies_preserve_sixfold_material_symmetry(host):
    _, _, bodies = host
    for family in ("pole_pin", "coil"):
        original = bodies[f"{family}_1"]
        for i in range(1, 6):
            rotated = original.rotate((0.0, 0.0, 60.0 * i))
            actual = bodies[f"{family}_{i + 1}"]
            assert (rotated - actual).volume() == pytest.approx(0.0, abs=1e-6)
            assert (actual - rotated).volume() == pytest.approx(0.0, abs=1e-6)


def test_every_pin_and_winding_is_connected_and_material_parts_do_not_overlap(host):
    _, _, bodies = host
    for region, body in bodies.items():
        assert len(body.decompose()) == 1, region
    for (name_a, body_a), (name_b, body_b) in combinations(bodies.items(), 2):
        assert (body_a ^ body_b).volume() == pytest.approx(0.0, abs=1e-7), (name_a, name_b)


def test_pins_join_shared_yoke_while_windings_clear_the_cores(host):
    _, d, bodies = host
    yoke = bodies["return_yoke"]
    clearance = min((d["coil_window_width_mm"] - d["root_width_mm"]) / 2,
                    (d["coil_window_height_mm"] - d["stem_height_mm"]) / 2)
    assert clearance > 0
    for i in range(1, 7):
        pin, coil = bodies[f"pole_pin_{i}"], bodies[f"coil_{i}"]
        assert pin.min_gap(yoke, 1.0) == pytest.approx(0.0, abs=1e-7)
        # A finite shared face, rather than just a touching point, joins the
        # magnetic circuit. The polygonal approximation changes its area a little.
        interface_area = _cylindrical_interface_area(pin, d["outer_radius_mm"] - d["yoke_thickness_mm"])
        assert interface_area > 0.8 * d["root_width_mm"] * d["stem_height_mm"]
        assert pin.min_gap(coil, clearance * 2) == pytest.approx(clearance, abs=1e-6)
        assert coil.min_gap(yoke, 10.0) > 0


@pytest.mark.parametrize("key", KEYS)
def test_physical_pole_length_changes_geometry_independently_of_field_fwhm(key):
    row = _row(key)
    baseline = multipole_solids(row, {})
    changed_field = {**row, "effective_length_mm": row["effective_length_mm"] * 0.6}
    assert [s.shape for s in multipole_solids(changed_field, {})] == [s.shape for s in baseline]
    d = probe_hexapole_dimensions(row)
    changed_pole = {**row, PREFIX + "pole_length_mm": (d["pole_length_mm"] + d["stem_height_mm"]) / 2}
    altered = {s.region: s for s in multipole_solids(changed_pole, {})}
    for old in baseline:
        if old.region.startswith("pole_pin"):
            assert _manifold(altered[old.region].shape, 96).volume() < _manifold(old.shape, 96).volume()
        else:
            assert altered[old.region].shape == old.shape
    assert changed_pole["effective_length_mm"] == row["effective_length_mm"]


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("field,value", [
    ("pole_length_mm", True), ("pole_length_mm", float("nan")),
    ("pole_length_mm", float("inf")), ("pole_length_mm", 0.0),
    ("pole_length_mm", 1000.0), ("pole_length_mm", 0.1),
    ("stem_height_mm", 1000.0), ("coil_window_height_mm", 0.1),
    ("coil_height_mm", 1000.0), ("tip_width_mm", 1000.0),
    ("root_width_mm", 1000.0), ("coil_window_width_mm", 0.1),
    ("yoke_thickness_mm", 1000.0), ("taper_length_mm", 1000.0),
    ("coil_radial_inset_mm", 0.1), ("coil_radial_length_mm", 1000.0),
    ("coil_width_mm", 1000.0),
])
def test_invalid_physical_dimensions_are_rejected_before_building(key, field, value):
    row = {**_row(key), PREFIX + field: value}
    with pytest.raises(ValueError):
        probe_hexapole_dimensions(row)
    with pytest.raises(ValueError):
        multipole_solids(row, {})


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("radial_gap", (0.001, 0.00001))
def test_accepted_narrow_yoke_clearances_keep_whole_poles_connected(key, radial_gap):
    row = _row(key)
    d = probe_hexapole_dimensions(row)
    inner, outer = d["inner_radius_mm"], d["outer_radius_mm"]
    # Put a short winding immediately outside the taper and almost tangent to
    # the yoke. This stresses clipping the radial stem against a curved yoke.
    coil_end = inner + 5.602
    overrides = dict(taper_length_mm=5.6, coil_radial_inset_mm=5.601,
                     coil_radial_length_mm=0.001, tip_width_mm=2.0,
                     root_width_mm=5.0, coil_window_width_mm=5.5,
                     coil_width_mm=6.0,
                     yoke_thickness_mm=outer - math.hypot(coil_end, 3.0) - radial_gap)
    row.update({PREFIX + name: value for name, value in overrides.items()})
    d = probe_hexapole_dimensions(row)
    bodies = _bodies(row)
    yoke = bodies["return_yoke"]
    for i in range(1, 7):
        pin = bodies[f"pole_pin_{i}"]
        assert len(pin.decompose()) == 1
        assert pin.min_gap(yoke, 1.0) == pytest.approx(0.0, abs=1e-7)
        assert _cylindrical_interface_area(pin, outer - d["yoke_thickness_mm"]) > (
            0.8 * d["root_width_mm"] * d["stem_height_mm"])


@pytest.mark.parametrize("key", ("probe_hpc_hexapole", "probe_hpol_hexapole",
                                  "probe_qpc_quadrupole", "probe_qph1_quadrupole"))
def test_unverified_control_channels_do_not_gain_invented_material(key):
    assert multipole_solids(_row(key), {}) == ()
