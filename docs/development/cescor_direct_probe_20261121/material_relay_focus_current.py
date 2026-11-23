"""Bracket specimen D95 by changing the single series-objective supply."""
import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from condenser_pair_column import specimen
from material_relay_column import Column,Field,source_and_electrodes
from material_relay_fields import SAVE,WORK
from refined_objective_normal import RefinedObjectiveFields
from relay_transfer import advance


def evaluate(label,source_label,tangent,span,*,axial_d95=False):
    folder=SAVE/label
    r=json.loads((folder/"tracking.json").read_text())
    source=json.loads((SAVE/source_label/"tracking.json").read_text())
    base_field=Field(WORK/r["field"]["source"],step=r["field"]["step_mm"])
    model=Column(source_and_electrodes() if axial_d95 else RefinedObjectiveFields(.5),base_field,
                 Path(source["gun_source"]),source["count"],r["step_mm"])
    base=model.t.tables[0]
    unit=cp.asarray(Field(tangent,"objective",step=r["field"]["step_mm"]).table())
    objective=next(v for k,v in r["field"]["bias"].items() if k.startswith("objective_lens/"))
    def propagate(offset):
        model.t.tables[0]=base+offset*unit
        return advance(model.t,model.full,model.t.input,model.en,model.phi0,model.loss,0,[len(model.m.z)-1])[0]
    triples=[propagate(off) for off in (-span,0.,span)]
    masks=[loss==-1 for _,loss in triples]
    if not all(np.array_equal(masks[0],m) for m in masks):
        raise ValueError("Objective search crossed an interception boundary")
    weights=model.t.weights[masks[0]]
    left,center,right=(p[m] for (p,_),m in zip(triples,masks))
    slope=(right-left)/(2*span);bend=(right+left-2*center)/(2*span**2)
    if axial_d95:
        # The derivative at the observation plane is kinematic: dx/dz = tx,
        # dy/dz = ty in the saved Cartesian phase. Current and centroid weighting
        # follow the published D95. The final axial scan uses full native transport.
        def axial_derivative(phase):
            xy=phase[:,:2]-np.average(phase[:,:2],axis=0,weights=weights)
            v=phase[:,2:4]-np.average(phase[:,2:4],axis=0,weights=weights)
            radius=np.linalg.norm(xy,axis=1)
            order=np.argsort(radius)
            owner=order[np.searchsorted(np.cumsum(weights[order]),.95*weights.sum())]
            return 2*np.dot(xy[owner],v[owner])/radius[owner]
        def signed(dq):
            return axial_derivative(center+dq*slope+dq**2*bend)
        grid=np.linspace(-span,span,501)
        values=[signed(dq) for dq in grid]
        roots=[]
        for i in np.flatnonzero(np.array(values[:-1])*values[1:]<=0):
            off=brentq(signed,grid[i],grid[i+1],xtol=1e-12)
            roots.append((off,signed(off)))
        if not roots:
            best=grid[np.argmin(np.abs(values))]
        else:
            # At a change of quantile owner a sign jump need not have a zero.
            smooth=[entry for entry in roots if abs(entry[1])<1e-10]
            best=min(smooth or roots,key=lambda a:(abs(a[0]) if smooth else abs(a[1])))[0]
        phase,loss=propagate(best)
        if not np.array_equal(loss==-1,masks[0]):
            raise ValueError("The objective change altered the surviving source IDs")
        record=dict(reference_label=label, fixed_gun_exit_proposal=True, particles=source["count"],
                    objective_before_A=objective, proposed_A=objective+best, delta_A=best,
                    span_A=span, start_D95_gradient_nm_per_mm=values[len(grid)//2]*1e6,
                    proposed_D95_gradient_nm_per_mm=axial_derivative(phase[masks[0]])*1e6,
                    roots_A_and_gradient=[(o,float(v)*1e6) for o,v in roots], actual_objective_only_verification_required=True)
        (folder/"objective-axial-d95-proposal.json").write_text(json.dumps(record,indent=2))
        print(record,flush=True)
        return record
    rows={}
    def landing(offset):
        offset=float(offset)
        if offset not in rows:
            pred=center+offset*slope+offset**2*bend
            rows[offset]=specimen(pred,weights,model.t.current)
        return rows[offset]["D95_um"]
    grid=np.linspace(-span,span,55)
    idx=int(np.argmin([landing(x) for x in grid]))
    if 0<idx<len(grid)-1:
        minimize_scalar(landing,method="bounded",bounds=(grid[idx-1],grid[idx+1]),options={"xatol":2e-6})
    best=min(rows,key=landing)
    p,loss=propagate(best)
    if not np.array_equal(loss==-1,masks[0]):raise ValueError("Objective current changes transmitted identities")
    exact=specimen(p[masks[0]],weights,model.t.current)
    centered=p[masks[0],:4]-np.average(p[masks[0],:4],axis=0,weights=weights)
    growth=float(np.average(np.sum(centered[:,:2]*centered[:,2:4],axis=1),weights=weights))
    data=dict(pre_objective_A=objective,proposed_A=objective+best,delta_A=best,bracketed=bool(grid[0]<best<grid[-1]),
              source_label=source_label,source_count=source["count"],reference_field=r["field"]["source"],
              tangent=tangent.name,criterion="Specimen-plane finite-source D95; local field-tangent search before an independent DC solve",
              independent_native_trace=exact,radial_covariance_at_D95_focus_m_rad=growth,
              interpolation_error_nm=1000*(exact["D95_um"]-rows[best]["D95_um"]),
              samples=[dict(delta_A=q,**rows[q]) for q in sorted(rows)])
    (folder/"objective-d95-search.json").write_text(json.dumps(data,indent=2))
    print("Objective-only D95 proposal",data["proposed_A"],best,rows[best],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("label");p.add_argument("source_label")
    p.add_argument("tangent",type=Path);p.add_argument("--span",type=float,default=.012)
    p.add_argument("--axial-d95",action="store_true")
    a=p.parse_args();evaluate(a.label,a.source_label,a.tangent,a.span,axial_d95=a.axial_d95)
