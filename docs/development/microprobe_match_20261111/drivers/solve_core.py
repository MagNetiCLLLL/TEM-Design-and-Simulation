"""Propose an afocal conjugate HP relay before finite-source validation."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from propose_match import Proposal, matrices

HERE = Path(__file__).resolve().parent
proposal = Proposal()
keys = ("probe_tl22_lens", "probe_tl21_lens")


def maps(drives):
    return dict(zip(proposal.names, matrices(proposal.fields, drives, proposal.lengths, proposal.marks, proposal.rigidity)))


def relay(drives):
    m = maps(drives)
    return m["hp1"]@np.linalg.inv(m["hp2"])


print("Original HP relay:", relay(proposal.base).tolist(), flush=True)
cores=[]
for a in (20.,40.,55.,70.,90.):
    for b in (20.,40.,60.,80.,95.):
        def residual(values):
            r = relay(proposal.drives(keys,values))
            return (r[0,1]/.1, r[1,0]/10.)
        fit=least_squares(residual,(a,b),bounds=(.1,100.),diff_step=1e-5,max_nfev=120,xtol=1e-10,gtol=1e-9,ftol=1e-10)
        if np.linalg.norm(fit.fun)>.001 or any(np.linalg.norm(fit.x-np.array(old["values"]))<1e-3 for old in cores):
            continue
        drives=proposal.drives(keys,fit.x)
        row=dict(values=fit.x.tolist(), drives=drives.tolist(),relay=relay(drives).tolist())
        cores.append(row)
        print("Afocal core:",json.dumps(row),flush=True)
nominals=[]
free=("condenser_lens_2","condenser_lens_3","adapter_lens","probe_tl12_lens")
for core in cores:
    for initial in ((10.56,33.6,9.2,14.),(10.2,60.,9.2,30.),(10.5,40.,20.,55.),(8.,20.,30.,80.),(10.,85.,20.,14.)):
        def residual(values):
            drives=proposal.drives(free,values,core["drives"])
            row,m=proposal.measure(drives)
            hp=m["hp2"]@proposal.pencil
            return (np.log(max(row["d95_um"],1e-6)),row.get("curvature_per_m",1e6)/1000.,
                    hp[1]/max(abs(hp[0]),1e-3)/100.,.0001*(values[2]-9.2))
        fit=least_squares(residual,initial,bounds=(.1,100.),diff_step=1e-5,max_nfev=350,xtol=1e-10,gtol=1e-8,ftol=1e-9)
        drives=proposal.drives(free,fit.x,core["drives"])
        row,m=proposal.measure(drives)
        if row["count"]<100 or np.linalg.norm(fit.fun[:3])>.03:
            continue
        if any(np.linalg.norm(drives-np.array(old["drives"]))<1e-3 for old in nominals):
            continue
        entry=dict(drives=drives.tolist(),controls={key:dict(percent=float(value)) for key,value in zip(proposal.keys,drives)},
            residual=fit.fun.tolist(),measurement=row,core_relay=relay(drives).tolist(),
            pencil={key:(value@proposal.pencil).tolist() for key,value in m.items()})
        nominals.append(entry)
        print("Nominal:",json.dumps(entry),flush=True)
report=dict(scope="Axial-field paraxial proposal only",keys=proposal.keys,cores=cores,nominals=nominals)
(HERE/"afocal-proposals.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
print("Done",len(cores),len(nominals),flush=True)
