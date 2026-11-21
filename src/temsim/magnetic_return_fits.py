"""Material contact at the intended seats of corrector and projector returns."""

from functools import lru_cache
from itertools import combinations, pairwise

import manifold3d as manifold
import numpy as np

from temsim.cad_round_lenses import round_lens_solids
from temsim.magnetic_geometry import CORRECTOR_SPLIT_RETURNS, OBJECTIVE_SPLIT_RETURNS


_ANGULAR_SEGMENTS = 128
_LINEAR_TOL_MM = 1e-7


def _perimeter(section):
    return sum(np.linalg.norm(contour - np.roll(contour, 1, axis=0), axis=1).sum()
               for contour in section.to_polygons())


def _point_edge_distance(points, polygon):
    """Minimum distance from vertices to the closed line segments, in mm."""
    starts = np.asarray(polygon)
    edges = np.roll(starts, -1, axis=0) - starts
    relative = np.asarray(points)[:, None] - starts
    lengths2 = np.sum(edges * edges, axis=-1)
    fractions = np.sum(relative * edges, axis=-1) / np.where(lengths2 > 0, lengths2, 1.)
    closest = relative - fractions.clip(0., 1.)[:, :, None] * edges
    return np.linalg.norm(closest, axis=-1).min()


@lru_cache(maxsize=1024)
def _revolved_joint(first, second, require_contact):
    # In the meridian, common area sweeps out overlapping volume. A shared
    # edge at positive radius sweeps out a face; a single point does not.
    a, b = (manifold.CrossSection([outline], manifold.FillRule.EvenOdd)
            for outline in (first, second))
    area = (a ^ b).area()
    if area > 1e-9:
        return f"material overlap ({area:.6g} mm² in axial section)"
    if require_contact and _perimeter(a) + _perimeter(b) - _perimeter(a + b) <= 2 * _LINEAR_TOL_MM:
        gap = min(_point_edge_distance(first, second), _point_edge_distance(second, first))
        return f"air gap {gap:.6g} mm" if gap > _LINEAR_TOL_MM else "edge-only contact"
    return None


def _as_solid(region):
    from temsim.cad_mesh import _manifold
    if isinstance(region, tuple):
        return _manifold({"kind": "revolve", "section_zr_mm": region}, _ANGULAR_SEGMENTS)
    return region


def _joint_error(first, second, *, require_contact=True):
    if isinstance(first, tuple) and isinstance(second, tuple):
        return _revolved_joint(first, second, require_contact)
    a, b = _as_solid(first), _as_solid(second)
    if a.is_empty() or b.is_empty():
        return "missing material at this seat"
    volume = (a ^ b).volume()
    if volume > 1e-6:
        return f"material overlap ({volume:.6g} mm³ in the 3D mesh)"
    if require_contact and a.surface_area() + b.surface_area() - (a + b).surface_area() <= 2e-6:
        bounds = np.asarray([a.bounding_box(), b.bounding_box()])
        span = bounds[:, 3:].max(axis=0) - bounds[:, :3].min(axis=0)
        gap = a.min_gap(b, float(np.linalg.norm(span)))
        return f"air gap {gap:.6g} mm in the 3D mesh" if gap > _LINEAR_TOL_MM else "edge-only contact in the 3D mesh"
    return None


def _edited_round_meshes(part, by_key, azimuth_deg):
    from scipy.spatial.transform import Rotation
    from temsim.energy_filter_model_3d import rigid_mesh
    from temsim.part_model_3d import unplaced_part_meshes
    from temsim.part_model_features import apply_model_features

    # Sample the round seed on the joint's azimuth grid after placement. The
    # editor still rotates each cut and offset in its original coordinate frame.
    inverse = Rotation.from_euler("z", -azimuth_deg, degrees=True).as_matrix()
    seed = {k: v for k, v in part.items() if k != "model_3d"}
    native, _ = unplaced_part_meshes(seed, by_key, _ANGULAR_SEGMENTS)
    meshes = [rigid_mesh(mesh, inverse, (0., 0., 0.)) for mesh in native]
    return apply_model_features(part, meshes, angular_segments=_ANGULAR_SEGMENTS)


def _material_by_region(part, by_key, split):
    from temsim.part_model_3d import lens_shape_part, unplaced_part_meshes
    from temsim.part_model_features import _input_solid

    shape_part = lens_shape_part(part)
    model = shape_part.get("model_3d", {})
    transform = model.get("transform", {})
    x, y, dz = transform.get("offset_mm", (0., 0., 0.))
    rx, ry, azimuth = transform.get("rotation_deg", (0., 0., 0.))
    sx, sy = transform.get("scale_xy", (1., 1.))
    # Retain analytic circles even for azimuth changes, uniform radial scaling
    # or axial translations. Rotated polygonal cylinders can report a false fit.
    native = model.get("base", {}).get("kind", "existing") == "existing"
    meshes = None
    if native and rx == ry == 0. and sx == sy:
        solids = round_lens_solids(part, by_key)
        labels = {s.region for s in solids or ()}
        expected = {"upper", "lower"} if split else {"body"}
        if labels == expected and all(s.shape["kind"] == "revolve" for s in solids):
            if x == y == 0. and not any(f.get("enabled", True) for f in model.get("features", ())):
                return {s.region: tuple((z + dz, radius * sx) for z, radius in s.shape["section_zr_mm"])
                        for s in solids}
            if azimuth:
                meshes = _edited_round_meshes(shape_part, by_key, azimuth)

    if meshes is None:
        meshes, _ = unplaced_part_meshes(part, by_key, _ANGULAR_SEGMENTS)
    if not meshes:
        raise ValueError(f"{part['key']}: missing material in the magnetic return assembly")
    regions = {mesh.region: _input_solid(mesh, {}) for mesh in meshes}
    if split and {"upper", "lower"} <= regions.keys():
        return regions
    # Replacement CAD (including STEP) has its own leaf names. Test the
    # occupied material at each end without relying on an imported label.
    solid = manifold.Manifold.batch_boolean(list(regions.values()), manifold.OpType.Add)
    if not split:
        return {"body": solid}
    center = by_key[part["parent_key"]]["local_center_z_mm"]
    lower, upper = solid.split_by_plane((0., 0., 1.), center)
    return {"upper": upper, "lower": lower}


def _validate_corrector_return_fits(by_key):
    # The pole, sleeve, plate and yoke share the same lens placement. Check
    # their shapes in that common frame, including any individual CAD edits.
    owners = {row.get("parent_key") for row in by_key.values()
              if row.get("mechanical_part_role") in CORRECTOR_SPLIT_RETURNS}
    if any(row.get("mechanical_part_role") in OBJECTIVE_SPLIT_RETURNS for row in by_key.values()):
        owners.add("objective_lens")
    if owners - by_key.keys():
        raise ValueError("Each installed corrector return requires its parent lens")
    for owner in sorted(owners):
        objective = owner == "objective_lens"
        names = ("upper_pole", "lower_pole", "pole_sleeves", "return_plates", "yoke")
        shapes = {}
        for name in names:
            key = f"objective_{name}" if objective and name != "yoke" else f"{owner}_{name}"
            parent_key = ("objective_lens_yoke" if objective and name in {"pole_sleeves", "return_plates"}
                          else owner)
            part = by_key.get(key)
            if part is None or part.get("parent_key") != parent_key:
                raise ValueError(f"{owner}: missing {key} in the magnetic return assembly")
            split = name in {"pole_sleeves", "return_plates"} or (objective and name == "yoke")
            regions = _material_by_region(part, by_key, split)
            shapes.update({(name, region): material for region, material in regions.items()})
        seats = set()
        for side in ("upper", "lower"):
            path = ((f"{side}_pole", "body"), ("pole_sleeves", side),
                    ("return_plates", side), ("yoke", side if objective else "body"))
            for first, second in pairwise(path):
                joint = f"{owner}: {side} end, {first[0]} to {second[0]}"
                if first not in shapes or second not in shapes:
                    raise ValueError(f"{joint}: missing material")
                error = _joint_error(shapes[first], shapes[second])
                if error:
                    raise ValueError(f"{joint}: {error}")
                seats.add(frozenset((first, second)))
        # A shoulder can reach another component beyond a correctly seated
        # sleeve. Check that material too, without requiring contact across
        # the deliberate pole gap or between the opposite ends of the lens.
        for first, second in combinations(shapes, 2):
            if first[0] == second[0] or frozenset((first, second)) in seats:
                continue
            error = _joint_error(shapes[first], shapes[second], require_contact=False)
            if error:
                raise ValueError(f"{owner}: {'/'.join(first)} to {'/'.join(second)}: {error}")


def validate_magnetic_return_fits(by_key):
    _validate_corrector_return_fits(by_key)
    # Each shared projector plate seats on its upstream lower pole and yoke.
    # At an inter-lens station it also seats on the next upper pole and yoke.
    for plate in by_key.values():
        if plate.get("mechanical_part_role") != "return_plate":
            continue
        parent = by_key.get(plate.get("parent_key"), {})
        if (parent.get("magnetic_circuit_topology") != "shared_pole_multi_gap"
                or not parent.get("magnetic_circuit_id")):
            continue
        key = parent["key"]
        seats = [f"{key}_lower_pole", f"{key}_yoke"]
        downstream = [row for row in by_key.values()
                      if row.get("magnetic_circuit_topology") == "shared_pole_multi_gap"
                      and row.get("magnetic_circuit_id") == parent["magnetic_circuit_id"]
                      and row["local_center_z_mm"] > parent["local_center_z_mm"]]
        if downstream:
            next_key = min(downstream, key=lambda row: row["local_center_z_mm"])["key"]
            seats.extend((f"{next_key}_upper_pole", f"{next_key}_yoke"))
        plate_shape = _material_by_region(plate, by_key, False)["body"]
        for seat in seats:
            if seat not in by_key:
                raise ValueError(f"{plate['key']}: missing return contact with {seat}")
            error = _joint_error(plate_shape, _material_by_region(by_key[seat], by_key, False)["body"])
            if error:
                raise ValueError(f"{plate['key']} to {seat}: {error}")
