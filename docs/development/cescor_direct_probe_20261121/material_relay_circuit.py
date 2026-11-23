"""Assemble the three-dimensional iron and AC saddle circuits in common column coordinates."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from condenser_source_fields import HERE, write
from material_relay_fields import WORK, SAVE, RoundAssembly, digest
from refined_objective_normal import RefinedObjectiveFields
from relay_transfer import PARTS
from temsim.cad_catalog import part_solids
from temsim.corrector_field_cli import cached_mesh
from temsim.part_materials import material_for_region
from temsim.transverse_winding import transverse_winding_definition

sys.path.insert(0,str(HERE.parent/"probe_circuit_audit_20261122"))
from circuit_audit import audit_inputs


def capture():
    f=RefinedObjectiveFields(.5)
    assembly=RoundAssembly(f)
    recipe,bias,_=audit_inputs()
    geom=f.state._resolved_assembly
    rows={p.key:p.data for p in geom.parts}
    curves={s["material"]:s["curve"] for s in recipe["solids"] if s["curve"]}
    for key,entry in assembly.parts.items():
        if key=="objective_terminal_vacuum_tube":
            if any(x["magnetic_response"]=="bh" for x in assembly.regions[key].values()):
                raise ValueError("Include the terminal magnetic tube in the solid mesh")
            continue
        part=geom.part(key)
        offset=part.start_z_mm-part.data["local_start_z_mm"]-450.
        for solid in part_solids(part.data,rows):
            mat=material_for_region(part.data,solid.region)
            if mat and mat["magnetic_response"]!="bh":
                continue
            grade=mat["material_key"] if mat else "armco_pure_iron"
            if solid.shape["kind"]!="revolve":
                raise ValueError(f"Unregistered round material section: {key}/{solid.region}")
            section=np.array(solid.shape["section_zr_mm"])+[offset,0.]
            recipe["solids"].append(dict(name=key+"/"+solid.region,material=grade,curve=curves[grade],
                                               shape=dict(kind="revolve",section_zr_mm=section.tolist())))
    for i,w in enumerate(assembly.coils):
        name=f"{w['owner']}/physical-pack-{i}"
        recipe["sources"][name]=dict(kind="round",turns=w["turns"],inner=1000*w["ri"],outer=1000*w["ro"],
                                     intervals=[[1000*w["z0"]-450,1000*w["z1"]-450]])
        bias[name]=w["current_A"]
    winding=transverse_winding_definition(PARTS["ac_deflector"])
    for station in range(2):
        for axis in "xy":
            name=f"ac/{station}/{axis}"
            coils=[c for c in winding["coils"] if c["station_index"]==station and c["group"]==axis]
            recipe["sources"][name]=dict(kind="saddle",turns=winding["turn_count_per_coil"],coils=coils)
            bias[name]=0.
    bias.update({k:v for k,v in __import__("material_relay_fields").parent()["corrector_A"].items()})
    return recipe,bias


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--pitch",type=float,default=9.)
    args=ap.parse_args()
    recipe,bias=capture()
    geometry=WORK/"meshes"
    geometry.mkdir(exist_ok=True)
    write(WORK/"assembly.json",dict(recipe=recipe,drives=bias))
    print("Meshing material and vacuum",len(recipe["solids"]),"bodies",len(bias),"windings",flush=True)
    mesh_id,mesh=cached_mesh(recipe,args.pitch,170.,geometry)
    print("Mesh",mesh_id,len(mesh[0]),"nodes",len(mesh[1]),"tets",flush=True)
    write(WORK/f"mesh-{args.pitch:g}.json",dict(mesh_id=mesh_id,recipe=digest(recipe),
                                         nodes=len(mesh[0]),elements=len(mesh[1]),pitch=args.pitch))
