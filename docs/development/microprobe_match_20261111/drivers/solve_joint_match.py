"""Simultaneous first-order conjugacy and Microprobe illumination proposals."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from propose_match import Proposal, matrices

HERE=Path(__file__).resolve().parent
p=Proposal()
free=("condenser_lens_2","condenser_lens_3","adapter_lens","probe_tl22_lens","probe_tl21_lens","probe_tl12_lens")
seeds=[(10.561,33.6,a,tl22,tl21,t) for tl22,tl21 in ((49.98,61.42),(77.57,46.04))
       for a in (5.,15.,40.) for t in (14.,40.,80.)]
attempts=[]
for initial in seeds:
    def residual(values):
        d=p.drives(free,values)
        m=dict(zip(p.names,matrices(p.fields,d,p.lengths,p.marks,p.rigidity)))
        core=m["hp1"]@np.linalg.inv(m["hp2"])
        hp=m["hp2"]@p.pencil
        ap=m["c2ap"]@p.pencil
        specimen=m["sample"]@p.pencil
        # Continuous filled-pupil approximation avoids discrete interception
        # jumps during proposal generation; native current quantiles follow.
        diameter=2e6*p.state.condenser_aperture_2.radius_mm*.001*np.sqrt(.95)*abs(specimen[0]/ap[0])
        return (np.log(max(diameter,1e-8)),specimen[1]/max(abs(specimen[0]),1e-6)/1000.,
                hp[1]/max(abs(hp[0]),1e-3)/100.,core[0,1]/.05,core[1,0]/20.,.00001*(values[2]-9.2))
    fit=least_squares(residual,initial,bounds=(.05,100.),diff_step=1e-5,max_nfev=500,
                      xtol=1e-10,ftol=1e-9,gtol=1e-8)
    d=p.drives(free,fit.x)
    row,m=p.measure(d)
    entry=dict(drives=d.tolist(),controls={key:dict(percent=float(value)) for key,value in zip(p.keys,d)},
        residual=fit.fun.tolist(),measurement=row,core=(m["hp1"]@np.linalg.inv(m["hp2"])).tolist(),
        pencil={key:(value@p.pencil).tolist() for key,value in m.items()})
    attempts.append(entry)
    print(json.dumps(dict(seed=initial,values=fit.x.tolist(),norm=float(np.linalg.norm(fit.fun)),**row)),flush=True)
    (HERE/"joint-proposals.json").write_text(json.dumps(dict(scope="Paraxial proposals only",attempts=attempts),indent=2,allow_nan=False)+"\n",encoding="utf8")
print("Done",flush=True)
