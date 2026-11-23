"""Resolve the signed round winding packs and installed steel in one R-Z material solve."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from condenser_source_fields import EV, FINAL, HERE, LENSES, OWNER, PROJECTION, check_file, sha, write
from refined_objective_normal import EVIDENCE as PARENT, RefinedObjectiveFields
from relay_magnetics import material_recipe
from scan_region_diagnostic import bore_coefficients
from temsim.ideal_coil_segments import FIELD as IDEAL_SEGMENTS
from temsim.magnetic_circuits import radial_profile_mm
from temsim.part_materials import is_magnetostatic_body, material_catalog, validated_material_regions
from temsim.physics.axisymmetric_magnetostatics import _material_region_masks
from temsim.physics.condenser_pair_field import _axis
from temsim.physics.lens_field_provider import lens_geometry_binding
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.physics.nonlinear_magnetostatics import solve_nonlinear
from temsim.pole_geometry import RECESSED_OBJECTIVE_STYLE, recessed_objective_section


WORK = HERE.parents[2] / "outputs/material-relay-20261122"
SAVE = EV / "material-relay"
WORK.mkdir(exist_ok=True, parents=True)
SAVE.mkdir(exist_ok=True, parents=True)
PARENT_HASH = "75a77729a3b66436cf0419a4a8921011d6bc3a31ee078f6e1831c747895401e4"
ROUND = (*LENSES, "mini_condenser", "objective_lens", *PROJECTION)


def parent():
    check_file(PARENT / "working-point.json", PARENT_HASH)
    return json.loads((PARENT / "working-point.json").read_text())


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


class RoundAssembly:
    def __init__(self, f, *, excitation="ampere-turns"):
        self.f = f
        saved = parent()
        self.q = f.current(-30., 16.)
        np.testing.assert_array_equal(self.q, saved["signed_current_A"])
        self.materials = [next(x["bh_material"] for x in material_catalog() if x["material_key"] == "armco_pure_iron")]
        self.parts, self.regions, self.binding = {}, {}, {}
        self.coils = []
        self.excitation = excitation
        overrides = {key:float(os.environ[var]) for key,var in
                     (("condenser_lens_3","TEMSIM_RELAY_C3_A"),("mini_condenser","TEMSIM_RELAY_MINI_A")) if var in os.environ}
        if overrides:
            self.excitation += "-"+"-".join(f"{key}{v:.8g}" for key,v in overrides.items())
        self.electrical = {}
        for key in ROUND:
            data = json.loads(lens_geometry_binding(f.state, key).canonical_geometry_json)
            self.binding[key] = digest(data)
            for part in data["lens_assembly"]["parts"]:
                if is_magnetostatic_body(part["data"]):
                    if part["key"] in self.parts and self.parts[part["key"]] != part:
                        raise ValueError(f"Conflicting instances of {part['key']}")
                    self.parts[part["key"]] = part
            indices = np.flatnonzero(OWNER == key)
            archived_turns = np.array([f.units[i].ampere_turns for i in indices])
            total_turns = int(sum(archived_turns))
            ampere_turns = float(archived_turns @ self.q[indices])
            descriptor = f.state.lens_field_map_descriptors.get(key, {})
            segments = descriptor.pop(IDEAL_SEGMENTS, None)
            try:
                packs = capture_round_lens_potential(f.state, key).potential.coils
            finally:
                if segments is not None:
                    descriptor[IDEAL_SEGMENTS] = segments
            windings = [p["data"] for p in data["lens_assembly"]["parts"] if p["data"].get("field_source_key")==key
                        and p["data"].get("mechanical_profile")=="magnetic_excitation_coil"]
            counts = [n for w in windings for n in w["winding_interval_turn_counts"]]
            if len(counts)!=len(packs) or sum(counts)!=total_turns:
                raise ValueError(f"Physical pack intervals do not match turns in {key}")
            native_ni = float(sum(w.ampere_turns for w in packs))
            current = (native_ni if excitation=="panel" and key in ("condenser_lens_3", "mini_condenser")
                       else ampere_turns)/total_turns
            if key=="objective_lens":
                current = saved["objective_A"]
            current = overrides.get(key,current)
            self.electrical[key] = dict(physical_turns=total_turns, historical_signed_At=ampere_turns,
                                        series_A=current, native_panel_A=native_ni/total_turns)
            for n,w in zip(counts,packs,strict=True):
                self.coils.append(dict(owner=key, turns=n, current_A=current,
                    ri=w.inner_radius_m, ro=w.outer_radius_m, z0=w.center_z_m-w.length_m/2,
                    z1=w.center_z_m+w.length_m/2))
        for key, part in self.parts.items():
            assignments = validated_material_regions(part["data"])
            self.regions[key] = assignments
            for assignment in assignments.values():
                if assignment["magnetic_response"] == "bh" and assignment["bh_material"] not in self.materials:
                    self.materials.append(assignment["bh_material"])
        turns = sum(w["turns"] for w in self.coils if w["owner"] == "objective_lens")
        np.testing.assert_allclose(turns * saved["objective_A"], saved["objective_At"], atol=1e-8)

    def axes(self, pitch):
        r = [0., .24, .48, .72, .96, 1.20, 1.44, 1.68, 1.92, 2.16, 2.40, 2.64, 2.88,
             3.6, 6., 8., 20., 90., 120., 170., 250., 400., 680.]
        z = [-600., 0., 300., 430., 775., 1035., 1250., 1280., 1440., 1510., 1680., 1810., 2620., 3450.]
        for key, part in self.parts.items():
            d = part["data"]
            a, b = part["start_z_mm"], part["end_z_mm"]
            z.extend([a, b])
            z.extend(v for bounds in d.get("material_intervals_mm", []) for v in bounds)
            r.extend(v/2 for k, v in d.items() if k.startswith("mechanical_") and k.endswith("_diameter_mm"))
            prof = radial_profile_mm(d, b-a, key=key)
            if prof is not None:
                z.extend(prof[:, 0] + a)
                r.extend(prof[:, 1:].ravel())
            elif d.get("pole_piece_geometry_style") == RECESSED_OBJECTIVE_STYLE:
                for zz, rr in recessed_objective_section(d, a, b, key=key):
                    z.append(zz)
                    r.append(rr)
        for w in self.coils:
            r.extend([1000*w["ri"], 1000*w["ro"]])
            z.extend([1000*w["z0"], 1000*w["z1"]])
        def dz(v):
            if 1280 <= v <= 1440 or 1510 <= v <= 1605:
                return pitch
            if 430 <= v <= 1035 or 1440 <= v <= 1810:
                return 3*pitch
            return 7*pitch if 1035 <= v <= 2620 else max(10., .12*max(0.,430-v,v-2620.))
        return (_axis(r, lambda v: 2.4*pitch if v < 90 else max(4.,v*.1)), _axis(z, dz))

    def labels(self, r, z):
        ids = np.full(np.broadcast_shapes(r.shape, z.shape), -2, dtype=int)
        for key, part in self.parts.items():
            for mask, assignment in _material_region_masks(part, self.regions[key], r, z):
                j = 0 if assignment is None else (
                    self.materials.index(assignment["bh_material"]) if assignment["magnetic_response"]=="bh" else -1)
                if np.any(mask & (ids != -2) & (ids != j)):
                    raise ValueError(f"Conflicting materials on {key}")
                ids[mask] = j
        return np.maximum(ids, -1)

    def run(self, obj, pitch=.5, isolated=False):
        key = "pair" if isolated else "all"
        out = WORK / f"circuit-{self.excitation}-{key}-{obj:.10g}A-{pitch:g}mm.npz"
        extended = material_recipe({"solids": [{"curve": c} for c in self.materials]}, "log-J")
        curves = [solid["curve"] for solid in extended["solids"]]
        recipe = dict(binding=self.binding, coils=self.coils, materials=curves, obj_A=obj, electrical=self.electrical,
                      mesh_mm=pitch, isolated=isolated, implementation=sha(Path(__file__)))
        identity = digest(recipe)
        if out.exists():
            with np.load(out) as result:
                if str(result["identity"]) == identity:
                    return out
        def current(r,z):
            value = np.zeros_like(r)
            labels = self.labels(r,z)
            for w in self.coils:
                inside = (r>=w["ri"]) & (r<=w["ro"]) & (z>=w["z0"]) & (z<=w["z1"])
                if np.any(inside & (labels>=0)):
                    raise ValueError(f"Windings intersect iron in {w['owner']}")
                amp = obj if w["owner"]=="objective_lens" else w["current_A"]
                if not isolated or w["owner"] in ("mini_condenser","objective_lens"):
                    value[inside] += amp*w["turns"]/((w["ro"]-w["ri"])*(w["z1"]-w["z0"]))
            return value
        r,z = self.axes(pitch)
        print(f"Solving {key} {obj:g} A, {len(r)} x {len(z)} nodes, {len(self.parts)} bodies", flush=True)
        sol=solve_nonlinear(r,z,self.labels,current,curves,relative_tolerance=1e-7)
        coeff=bore_coefficients(r,sol.a_phi_tm)
        np.savez_compressed(out,r=r,z=z,coeff=coeff,aphi=sol.a_phi_tm,residual=sol.relative_residual,
                            peak_T=sol.peak_material_t,identity=identity,recipe=json.dumps(recipe,sort_keys=True))
        print(out.name,sol.relative_residual,sol.peak_material_t,flush=True)
        return out


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--objective",type=float,nargs="+",default=[FINAL["objective_A"]])
    ap.add_argument("--pitch",type=float,default=.5)
    ap.add_argument("--isolated",action="store_true")
    args=ap.parse_args()
    model=RoundAssembly(RefinedObjectiveFields(.5))
    for objective in args.objective:
        model.run(objective,args.pitch,args.isolated)
