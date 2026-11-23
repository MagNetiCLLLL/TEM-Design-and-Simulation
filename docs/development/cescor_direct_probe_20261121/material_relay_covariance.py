"""Evaluate full transverse growth at the actual focal and specimen planes for feedback."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from condenser_pair_column import specimen
from material_relay_column import Column,Field
from material_relay_fields import SAVE,WORK
from refined_objective_normal import RefinedObjectiveFields
from relay_transfer import advance
from trace_normal_joint_field import beam_moments


def evaluate(folder,source_folder=None):
    r=json.loads((folder/"tracking.json").read_text())
    source=r if source_folder is None else json.loads((source_folder/"tracking.json").read_text())
    full=WORK/r["field"]["source"];front=r["isolated_pair_fronts"][0]["z_mm"]*.001
    model=Column(RefinedObjectiveFields(.5),Field(full,step=r["field"]["step_mm"]),Path(source["gun_source"]),source["count"],r["step_mm"])
    j=model.indices([front])[0];idx=np.r_[j-2:j+3,len(model.m.z)-1]
    moments=[];growth=[]
    for p,loss in advance(model.t,model.full,model.t.input,model.en,model.phi0,model.loss,0,idx):
        w=model.t.weights[loss==-1];pos=p[loss==-1,:4]-np.average(p[loss==-1,:4],weights=w,axis=0)
        tensor=(pos[:,:2].T*w)@pos[:,2:4]/w.sum()
        moments.append([np.trace(tensor),tensor[0,0]-tensor[1,1],tensor[0,1]+tensor[1,0]])
        growth.append(beam_moments(p,loss==-1,model.t.weights))
    samples=model.m.z[idx[:-1]]
    roots=CubicSpline(samples,moments[:-1])(front)
    slopes=CubicSpline(samples,growth[:-1])(front,1)
    result=dict(field=full.name,source_id=Path(source["gun_source"]).name,source_count=source["count"],
                full_source_in_this_field=source["field"]["solution_sha256"]==r["field"]["solution_sha256"],
                front_mm=1000*front,front_growth_m_rad=roots.tolist(),surface_growth_m_rad=moments[-1],
                local_growth_derivative_um_mrad_per_mm=(slopes*.001).tolist(),
                specimen=specimen(p[loss==-1],w,model.t.current))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("label");p.add_argument("--source-label")
    a=p.parse_args();folder=SAVE/a.label
    result=evaluate(folder,SAVE/a.source_label if a.source_label else None)
    out=folder/f"covariance-{a.source_label or 'same-field'}.json"
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
