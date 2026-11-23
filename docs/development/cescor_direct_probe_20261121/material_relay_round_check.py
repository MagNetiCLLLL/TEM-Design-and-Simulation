"""Compare the isolated three-dimensional front with independent refined round-material solves."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from condenser_source_fields import write
from material_relay_branches import fast
from material_relay_fields import RoundAssembly,SAVE,WORK,sha
from refined_objective_normal import RefinedObjectiveFields


def check(label):
    folder=SAVE/label
    r=json.loads((folder/"tracking.json").read_text())
    currents=r["field"]["bias"]
    def amp(prefix):
        return next(v for k,v in currents.items() if k.startswith(prefix+"/"))
    os.environ["TEMSIM_RELAY_C3_A"]=str(amp("condenser_lens_3"))
    os.environ["TEMSIM_RELAY_MINI_A"]=str(amp("mini_condenser"))
    model=RoundAssembly(RefinedObjectiveFields(.5))
    z=np.linspace(.45,r["surface_z_mm"]*.001,46901)
    def front(b_field):
        a,b,c,d=fast(z,b_field).T
        derivative=-c[-1]*b+d[-1]*a
        ids=np.flatnonzero((z[:-1]>1.42)&(z[:-1]<1.5125)&(derivative[:-1]*derivative[1:]<0))
        return (1000*(z[ids]-np.diff(z)[ids]*derivative[ids]/(derivative[ids+1]-derivative[ids]))).tolist()
    rows=[]
    for step in (1.,.5,.25):
        path=model.run(amp("objective_lens"),step,True)
        with np.load(path) as data:
            values=CubicSpline(data["z"],2*data["coeff"][:,0])(z)
        rows.append(dict(nominal_refinement_mm=step,isolated_axisymmetric_front_mm=front(values),
                         source=path.name,sha256=sha(path)))
    pair=r["pair"]
    with np.load(WORK/(Path(pair["source"]).stem+f"-dc-bore-{pair['step_mm']:g}mm.npz")) as data:
        values=CubicSpline(data["z"],data["coef"][:,0,2])(z)
    write(folder/"axisymmetric-refinement.json",dict(pair_sha256=pair["solution_sha256"],
          pole_geometry="Installed Mini Condenser and series objective with rotationally symmetric iron in the R-Z check",
          round_geometry_results=rows,all_installed_material_3d_same_paraxial_observer_mm=front(values),
          all_material_3d_reported_mm=r["isolated_pair_fronts"][0]["z_mm"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("label")
    check(parser.parse_args().label)
