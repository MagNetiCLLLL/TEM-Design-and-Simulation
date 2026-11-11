"""Predict the condenser zoom domain for one explicit corrector proposal."""
from pathlib import Path
import json
import argparse
import numpy as np
from scipy.optimize import least_squares
from propose_match import Proposal, matrices

HERE=Path(__file__).resolve().parent
p=Proposal()
controls=json.loads((HERE/"column/initial-controls.json").read_text(encoding="utf-8-sig"))["controls"]
base=np.array([controls[key]["percent"] for key in p.keys])
free=("condenser_lens_2","condenser_lens_3")
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--targets",type=float,nargs="+",default=(.75,.8,1.,1.5,2.2))
parser.add_argument("--output",type=Path,default=HERE/"zoom-proposals.json")
args=parser.parse_args()
out=[]
for target in args.targets:
    def residual(values):
        d=p.drives(free,values,base)
        m=dict(zip(p.names,matrices(p.fields,d,p.lengths,p.marks,p.rigidity)))
        ap,hp,s=(m[key]@p.pencil for key in ("c2ap","hp2","sample"))
        diameter=2e6*p.state.condenser_aperture_2.radius_mm*.001*np.sqrt(.95)*abs(s[0]/ap[0])
        return (np.log(max(diameter,1e-9)/target),hp[1]/max(abs(hp[0]),1e-6)/100.)
    fit=least_squares(residual,(base[p.keys.index(free[0])],base[p.keys.index(free[1])]),bounds=(.1,100.),
                      diff_step=1e-5,max_nfev=250,xtol=1e-11,ftol=1e-10,gtol=1e-9)
    d=p.drives(free,fit.x,base)
    row,_=p.measure(d)
    entry=dict(target_um=target,residual=fit.fun.tolist(),measurement=row,
        controls={key:dict(percent=float(value)) for key,value in zip(p.keys,d)})
    out.append(entry)
    print(json.dumps(entry),flush=True)
args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf8")
