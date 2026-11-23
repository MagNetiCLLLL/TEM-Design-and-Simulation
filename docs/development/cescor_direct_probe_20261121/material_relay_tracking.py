"""Measure the zero-scan beam and independently driven AC pivot with fixed source identities."""

import argparse
import json

import cupy as cp
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import brentq

from column_current import ApertureTracker, SURFACE
from condenser_pair_column import column_mesh, round_field, specimen
from condenser_source_fields import EV, FEM, FINAL, OWNER, check_file, write
from joint_pivot_tuning import scan_response, stations
from material_relay_fields import PARENT, SAVE, WORK, RoundAssembly, parent
from particles import Tracker
from refined_objective_normal import PARENT as CONDENSERS, RefinedObjectiveFields
from relay_transfer import advance, electric_reference
from scan_coil_waist_origin import nearest
from scan_waist_audit import angular_map, differential, rotation_to_sample


def installed_mix():
    path = EV / "joint-final-scan.json"
    check_file(path, parent()["scan_calibration_sha256"])
    return np.array(json.loads(path.read_text())["scan"]["mix"])


def column_coefficients(path, nodes):
    with np.load(path) as raw:
        spl = make_interp_spline(raw["z"], raw["coeff"], k=3)
    factors = np.array([2., -16., 384., -18432.]) / .00288**np.arange(0,8,2)
    field = np.zeros((len(nodes),20))
    field[:,:8:2] = spl(nodes) * factors
    field[:,1:8:2] = spl(nodes, nu=1) * factors
    return cp.asarray(field)


class Trial:
    def __init__(self, f, n, step, gun, table=FEM, extra=()):
        self.f = f
        self.m,self.stops = column_mesh(f,step,extra_observation_mm=extra)
        with np.load(gun) as g:
            self.source = tuple(g[k][:n] for k in ("phase","weight","ids","energy_offset_ev")) + (float(g["emitted_A"]),)
        if n < 1000 or len(np.unique(self.source[2]))!=n or np.min(self.source[1])<=0:
            raise ValueError("Invalid original positive-current cohort")
        with np.load(table) as data:
            self.t = ApertureTracker(self.m,dict(data),"cubic",source=self.source)
        self.en = self.t.de + 300000.
        self.phi0 = electric_reference(self.t)
        self.loss = np.full(n,-1,dtype=np.int32)

    def indices(self, z):
        return nearest(self.m.z,np.asarray(z))

    def field(self, unused=None):
        return self.full

    def inherited(self):
        q = self.f.current(-30.,16.)
        path = CONDENSERS/"tuned-material-field.npz"
        check_file(path,parent()["condenser_material_sha256"])
        with np.load(path) as data:
            self.full = round_field(self.f,self.m.nodes,q,dict(data),7)
        mask = OWNER=="mini_condenser"
        zeros = dict(z=np.linspace(.449,1.601,16),induced_T=np.zeros(16))
        self.pair = round_field(self.f,self.m.nodes,np.where(mask,q,0),zeros,7)

    def material(self, assembly, objective, pitch):
        self.full = column_coefficients(assembly.run(objective,pitch),self.m.nodes)
        self.pair = column_coefficients(assembly.run(objective,pitch,True),self.m.nodes)

    def fronts(self):
        grid = stations(self,np.r_[np.arange(1.420,SURFACE,.001),SURFACE])
        tracker = Tracker(self.m,source=self.source)
        index = self.indices(grid)
        axis = self.pair[:,0].get()
        forward = differential(tracker,self.pair,index[0],index)
        to_sample = forward[-1] @ np.linalg.inv(forward)
        rotation,ks = rotation_to_sample(self,tracker,axis,grid)
        values = angular_map(to_sample,ks,rotation)[0]
        fit = CubicSpline(grid,values)
        roots = [brentq(fit,grid[i],grid[i+1]) for i in np.flatnonzero(values[:-1]*values[1:]<0)]
        checks = []
        for root in roots:
            idx = self.indices([root])[0]
            actual = self.m.z[idx]
            rot,_ = rotation_to_sample(self,tracker,axis,[actual])
            cols = differential(tracker,self.pair,idx,index[-1:],angular_only=True)[0]
            matrix = np.zeros((4,4))
            matrix[:,2:] = cols
            _,b,_,d = angular_map(matrix,ks,rot[0])
            checks.append(dict(z_mm=1000*root,checked_z_mm=1000*actual,
                         canonical_D_singular=np.linalg.svd(d,compute_uv=False).tolist(),
                         B_singular_mm=(1000*np.linalg.svd(b,compute_uv=False)).tolist()))
        return checks


def inherited(n, step):
    g = PARENT/"fresh-emission-and-exit.npz"
    check_file(g,parent()["evidence_sha256"][g.name])
    m = Trial(RefinedObjectiveFields(.5),n,step,g)
    m.inherited()
    fronts = m.fronts()
    print("Inherited refined fronts",fronts,flush=True)
    target = min(fronts,key=lambda r:abs(r["z_mm"]-1481.88))["z_mm"]
    scan=scan_response(m,None,installed_mix(),target)
    record=dict(original_source_total=32768,originals_tracked=n,step_mm=step,
                field_scope="Refined-objective parent, original four CAD saddle winding response in vacuum",
                command_urad=1.,fronts=fronts,pivot=scan)
    write(SAVE/f"parent-pivot-{n}.json",record)
    print("Inherited measured pivot candidates",[(v["z_mm"],v.get("position_singular_nm_per_urad")) for v in scan["candidates"]],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--count",type=int,default=1024)
    ap.add_argument("--step",type=float,default=.1)
    args=ap.parse_args()
    inherited(args.count,args.step)
