"""Verify an archived field against today's installed assembly and zero-command instrument."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from condenser_source_fields import write
from material_relay_circuit import capture
from material_relay_fields import SAVE,WORK,digest,sha
from refined_objective_normal import RefinedObjectiveFields
from temsim.cad_catalog import part_solids
from temsim.part_materials import material_for_region, material_region_key


def verify_stored_winding_loads():
    """Compare the actual cached FE loads against signed line integrals from today's CAD."""
    from numpy.polynomial.legendre import leggauss
    from scipy.constants import mu_0
    from material_relay_global import Global, impressed
    model = Global(9)
    geometry = json.loads((WORK/"assembly.json").read_text())["recipe"]
    # Include the coordinates and geometry of each winding in the independent record.
    fingerprint = digest(geometry)
    rng = np.random.default_rng(22112026)
    selected = rng.choice(len(model.cells), 2048, replace=False)
    vertices = model.mesh[0][model.cells[selected]]
    edges = vertices[:,1:]-vertices[:,:1]
    norms = np.linalg.norm(edges,axis=2)
    result = {}
    for key, row in geometry["sources"].items():
        path = model.out/(key.replace("/","__")+".npy")
        stored = np.load(path,mmap_mode="r")[selected]*mu_0
        stored_projection = np.einsum("ij,ikj->ik",stored,edges)/norms
        direct = np.zeros_like(stored_projection)
        for t,w in zip(*leggauss(4),strict=True):
            points = vertices[:,:1] + .5*(1+t)*edges
            b = impressed(row,points.reshape(-1,3)).reshape(points.shape)
            direct += .5*w*np.einsum("ikj,ikj->ik",b,edges)/norms
        error = np.abs(stored_projection-direct)
        result[key] = dict(source_geometry_sha256=digest(row),stored_load_sha256=sha(path),
                           maximum_relative_to_peak=float(error.max()/np.abs(direct).max()),
                           rms_relative_to_peak=float(np.sqrt(np.mean(error**2))/np.abs(direct).max()))
    return dict(geometry_sha256=fingerprint,mesh_id=model.mesh_id,cells_per_winding=len(selected),
                signed_projections_per_cell=3,gauss_points_per_edge=4,channels=result)


def verify_threefold(recipe,drives):
    angle=2*np.pi/3
    turn=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    def rounded(x):
        a=np.round(x,8);a[abs(a)<1e-8]=0.;return tuple(a.ravel())
    def key(row,initial):
        shape=row["shape"];mat=initial.copy();vec=np.zeros(3)
        while shape["kind"]=="transform":
            vec+=mat@shape["offset_mm"];mat=mat@shape["rotation"];shape=shape["shape"]
        orient=() if shape["kind"]=="revolve" and np.allclose(mat[:,2],[0.,0.,1.]) else rounded(mat)
        return row["material"],digest(shape),rounded(vec),orient
    if Counter(key(s,np.eye(3)) for s in recipe["solids"])!=Counter(key(s,turn) for s in recipe["solids"]):
        raise ValueError("Static symmetry projection is incompatible with the installed iron")
    for name,source in recipe["sources"].items():
        if not drives[name] or source["kind"]=="round":continue
        def winding(c,phi):
            return digest({**{k:v for k,v in c.items() if k not in ("coil_index","azimuth_deg")},
                           "azimuth_deg":round((c["azimuth_deg"]+phi)%360,7)%360})
        coils=source["coils"]
        if Counter(winding(c,0) for c in coils)!=Counter(winding(c,120) for c in coils):
            raise ValueError(f"Static symmetry projection is incompatible with {name}")


def check(solution,pair=None):
    recipe,bias=capture()
    frozen=json.loads((WORK/"assembly.json").read_text())
    meshed=json.loads((WORK/"mesh-9.json").read_text())
    if digest(recipe)!=meshed["recipe"] or digest(recipe)!=digest(frozen["recipe"]):
        raise ValueError("Installed CAD, material curves or winding turns changed")
    with np.load(WORK/"meshes"/(meshed["mesh_id"]+".npz")) as mesh:
        names=mesh["names"]
    names={str(n):"air" if n=="air" else f"s{i}" for i,n in enumerate(names)}
    comparable={**recipe,"solids":[{**s,"name":names[s["name"]]} for s in recipe["solids"]]}
    with np.load(solution) as data:
        solved=json.loads(str(data["report"]))
        if (solved["mesh_id"]!=meshed["mesh_id"] or solved["residual"]>2e-8 or
                digest(json.loads(str(data["recipe"])))!=digest(comparable) or solved["currents_A"]!=solved["excitation"]):
            raise ValueError("The solution did not use the installed circuit")
    cmd=solved["excitation"]
    varying=("probe_tl12_lens","condenser_lens_3/","mini_condenser/","objective_lens/")
    if set(cmd)!=set(bias) or any(cmd[k]!=v for k,v in bias.items() if not k.startswith(varying)):
        raise ValueError("Unexpected change to a fixed excitation")
    if len({v for k,v in cmd.items() if k.startswith("objective_lens/")})!=1:
        raise ValueError("Objective halves must share a single series current")
    verify_threefold(recipe,cmd)
    loads=WORK/"loads"/solved["mesh_id"]
    load_checks={}
    for key in cmd:
        test=json.loads((loads/(key.replace("/","__")+".json")).read_text())
        if test["relative_peak_error"]>=.005 or test["check_points"]<2048:
            raise ValueError(f"Installed winding verification failed for {key}")
        load_checks[key]=test["relative_peak_error"]
    if pair is not None:
        with np.load(pair) as data:
            r=json.loads(str(data["report"]))
            if (r["mesh_id"]!=solved["mesh_id"] or r["residual"]>2e-8 or
                r["excitation"]!={k:(v if k.startswith(("mini_condenser/","objective_lens/")) else 0.) for k,v in cmd.items()}):
                raise ValueError("The isolated front plane uses different physical windings or excitation")
    solids={s["name"]:s["material"] for s in recipe["solids"]}
    parts={p.key:p.data for p in RefinedObjectiveFields(.5).state._resolved_assembly.parts}
    declared={}
    for key,row in parts.items():
        for solid in part_solids(row,parts):
            assignment=material_for_region(row,solid.region) if material_region_key(row,solid.region) else None
            is_iron=assignment["magnetic_response"]=="bh" if assignment else solid.material=="soft_magnetic"
            if is_iron:
                declared[key+"/"+solid.region]=assignment["material_key"] if assignment else "armco_pure_iron"
    if declared!=solids:
        raise ValueError(f"Material census differs: {set(declared.items())^set(solids.items())}")
    keys={k:dict(A=v,turns=solved["turns"][k],geometry=recipe["sources"][k]["kind"]) for k,v in cmd.items()}
    return dict(assembly_parts=len(parts),material_solids=len(solids),material_counts=Counter(solids.values()),
                assembly_sha256=digest(recipe),mesh=meshed,elements_solved=solved["tetrahedra"],
                residual=solved["residual"],material_peaks=solved["material_peaks"],currents=keys,
                material_regions=solids,solution=solution.name,solution_sha256=sha(solution),
                unit_winding_field_checks=load_checks,
                magnetic_force_input="One summed Cartesian field from this nonlinear circuit; the separate axial kick table is zero",
                other_trims="HPC, quadrupole trims, stigmators, alignment deflectors and scan/wobble have zero static drive in the physical-current working point",
                objective_pair="Mini and the incident-side field of the physical series objective. Both 580-turn packs carry its signed current.",
                static_rotational_symmetry="Two 120-degree rotations checked for every material solid and powered winding; AC unprojected",
                electric_field="Installed electrostatic gun electrodes and column potential",
                static_scan_A=[cmd[k] for k in ("ac/0/x","ac/0/y","ac/1/x","ac/1/y")])


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("solution",type=Path);p.add_argument("--pair",type=Path)
    p.add_argument("--label",default="common-bias");p.add_argument("--verify-loads",action="store_true")
    a=p.parse_args()
    if a.verify_loads:
        result=verify_stored_winding_loads()
        write(WORK/"winding-load-projections.json",result)
    else:
        write(SAVE/a.label/"field-audit.json",check(a.solution,a.pair))
