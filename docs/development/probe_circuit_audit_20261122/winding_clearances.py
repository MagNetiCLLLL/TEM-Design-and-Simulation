"""Measure the installed winding-pack envelopes against every magnetic solid in the probe corrector."""

import json
from math import radians

import gmsh
import numpy as np

from circuit_audit import HERE, ROUND, audit_inputs
from temsim.cad_catalog import part_solids
from temsim.cad_transverse_winding import transverse_bulk_shape
from temsim.module_manifest import read_document
from temsim.multipole_winding import multipole_winding_definition
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.probe_corrector_fem import _occ_solid
from temsim.transverse_winding import transverse_winding_definition


def sector_volume(cad, shape):
    start, end = shape["start_mm"], shape["end_mm"]
    span = radians(shape["span_deg"])
    axis = (0.0, 0.0, start, 0.0, 0.0, end - start)
    outer = cad.addCylinder(*axis, shape["outer_mm"], angle=span)
    if shape["inner_mm"] > 0:
        inner = cad.addCylinder(*axis, shape["inner_mm"], angle=span)
        pieces, _ = cad.cut([(3, outer)], [(3, inner)])
        (outer,) = [tag for dim, tag in pieces if dim == 3]
    cad.rotate([(3, outer)], 0, 0, 0, 0, 0, 1, radians(shape["angle_deg"]) - span / 2)
    return outer


def winding_volume(cad, shape):
    if shape["kind"] == "difference" and shape["base"]["kind"] == "annular_sector":
        outer = sector_volume(cad, shape["base"])
        tools = [(3, sector_volume(cad, tool)) for tool in shape["tools"]]
        pieces, _ = cad.cut([(3, outer)], tools)
        (volume,) = [tag for dim, tag in pieces if dim == 3]
        return volume
    return _occ_solid(cad, shape)


def measure():
    recipe, _, _ = audit_inputs()
    doc = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    parts = {part["key"]: part for part in doc["parts"]}
    inventory = json.loads((HERE / "material-contacts.json").read_text())["installed_cad_parts"]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("magnetic-circuit-winding-clearances")
        cad = gmsh.model.occ
        bodies = {s["name"]: _occ_solid(cad, s["shape"]) for s in recipe["solids"]}
        bounds = np.array([cad.getBoundingBox(3, item) for item in bodies.values()])
        records = []
        for key, info in inventory.items():
            if not info["material_counts"].get("copper_winding"):
                continue
            if key in {k + "_excitation_coil" for k in ROUND}:
                shapes = [s.shape for s in part_solids(parts[key], parts) if s.material == "copper_winding"]
            else:
                definition = multipole_winding_definition(
                    parts[key], parts
                ) or transverse_winding_definition(parts[key], parts)
                if not definition:
                    raise ValueError(f"Missing physical winding envelope on {key}")
                shapes = [transverse_bulk_shape(c) for c in definition["coils"]]
            for i, shape in enumerate(shapes, 1):
                pack = winding_volume(cad, shape)
                box = np.asarray(cad.getBoundingBox(3, pack))
                axis_gap = np.maximum(np.maximum(box[:3] - bounds[:, 3:], bounds[:, :3] - box[3:]), 0.0)
                lower_bound = np.linalg.norm(axis_gap, axis=1)
                distance, nearest, overlaps = np.inf, None, {}
                named = list(bodies.items())
                for j in np.argsort(lower_bound):
                    if lower_bound[j] > distance + 1e-6:
                        break
                    name, iron = named[j]
                    separation = cad.getDistance(3, pack, 3, iron)[0]
                    if separation < distance:
                        distance, nearest = separation, name
                    if lower_bound[j] <= 1e-6:
                        shared, _ = cad.intersect(
                            [(3, pack)], [(3, iron)], removeObject=False, removeTool=False
                        )
                        volume = sum(cad.getMass(3, tag) for dim, tag in shared if dim == 3)
                        if volume > 1e-6:
                            overlaps[name] = volume
                records.append(
                    dict(
                        part=key,
                        pack=i,
                        clearance_mm=distance,
                        nearest=nearest,
                        pack_volume_mm3=cad.getMass(3, pack),
                        intersections_mm3=overlaps,
                    )
                )
        return records
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    values = measure()
    (HERE / "winding-clearances.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    print(
        len(values),
        "winding packs;",
        sum(bool(r["intersections_mm3"]) for r in values),
        "have a positive-volume intersection with magnetic material",
    )
