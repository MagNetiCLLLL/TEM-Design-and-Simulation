"""Measure common material faces and mechanical clearances on the CAD solids in the solve."""

from collections import Counter, defaultdict
from itertools import combinations
import json

import gmsh

from circuit_audit import HERE, ROUND, audit_inputs
from temsim.cad_catalog import part_solids
from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.probe_corrector_fem import HP, _occ_solid


def contact_graph():
    recipe, _, _ = audit_inputs()
    document = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    parts = {row["key"]: row for row in document["parts"]}
    inventory, registered = {}, set()
    for key, row in parts.items():
        if not key.startswith(("adapter_lens", "probe_")):
            continue
        solids = part_solids(row, parts)
        inventory[key] = dict(
            material_counts=dict(Counter(s.material for s in solids)),
            host_status=row.get("physical_host_status"),
        )
        registered.update(f"{key}/{s.region}" for s in solids if s.material == "soft_magnetic")
    if registered != {s["name"] for s in recipe["solids"]}:
        raise ValueError("The installed corrector magnetic-solid inventory differs from the solve")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("return-circuit-contact-audit")
        cad = gmsh.model.occ
        names = [s["name"] for s in recipe["solids"]]
        bodies = [(3, _occ_solid(cad, s["shape"])) for s in recipe["solids"]]
        _, mapping = cad.fragment(bodies[:1], bodies[1:])
        cad.synchronize()
        tags, faces = {}, defaultdict(list)
        for name, pieces in zip(names, mapping, strict=True):
            (tag,) = [i for dim, i in pieces if dim == 3]
            if tag in tags.values():
                raise ValueError("Duplicate material volume at " + name)
            tags[name] = tag
            for dim, face in gmsh.model.getBoundary([(3, tag)], oriented=False):
                faces[face].append(name)
        contacts = defaultdict(float)
        for face, adjacent in faces.items():
            for a, b in combinations(sorted(set(adjacent)), 2):
                contacts[a, b] += cad.getMass(2, face)
        joints = []

        def measure(a, b):
            area = contacts.get(tuple(sorted((a, b))), 0.0)
            gap = 0.0 if area else cad.getDistance(3, tags[a], 3, tags[b])[0]
            joints.append(dict(first=a, second=b, contact_mm2=area, clearance_mm=gap))

        for key in ROUND:
            for end in ("upper", "lower"):
                chain = [
                    key + "_" + end + "_pole/body",
                    key + "_pole_sleeves/" + end,
                    key + "_return_plates/" + end,
                    key + "_yoke/body",
                ]
                for a, b in zip(chain, chain[1:]):
                    measure(a, b)
            measure(key + "_upper_pole/body", key + "_lower_pole/body")
        for key in HP:
            sleeve_names = [name for name in names if name.startswith(key + "/") and "sleeve" in name]
            assert len(sleeve_names) == 6
            for i in range(1, 7):
                pin = f"{key}/pole_pin_{i}"
                (sleeve,) = [name for name in sleeve_names if name.endswith("_" + str(i))]
                measure(pin, sleeve)
                measure(sleeve, key + "/return_yoke")
                measure(pin, key + "/return_yoke")
        return dict(
            installed_cad_parts=inventory,
            materials={s["name"]: s["material"] for s in recipe["solids"]},
            contacts=[dict(first=a, second=b, area_mm2=v) for (a, b), v in contacts.items()],
            checked_joints=joints,
            isolated_solids=[name for name in names if not any(name in k for k in contacts)],
        )
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    result = contact_graph()
    (HERE / "material-contacts.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for j in result["checked_joints"]:
        if "pin_1" in j["first"] or "sleeve_1" in j["first"] or "probe_tl12" in j["first"]:
            print(j)
    print("Isolated:", result["isolated_solids"])
