"""Screen an ADL and TL12 rematch after increasing illumination with the condensers."""
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import root

from condenser_zoom import FACTOR, HERE, WORK, evaluate, prepare, read_axis


@njit(cache=True)
def images(q, z, f, launch, ref, c3, indices):
    x, v = launch[0], launch[2]
    rate = np.zeros((2,3))
    res = np.zeros(3); jac = np.zeros((3,3)); station = 0
    amp = np.array([q[0], c3, q[1], q[2]])-ref
    for j in range(indices[-1]):
        h = z[j+1]-z[j]
        k = f[j,0]+f[j,1]*amp[0]+f[j,2]*amp[1]+f[j,3]*amp[2]+f[j,4]*amp[3]
        t=k*h; co=np.cos(t); sn=np.sin(t); shear=-k*sn
        if abs(t)<1e-4:
            drift=h*(1.-t*t/6.+t**4/120.)
            ds=h*h*(-t/3.+t**3/30.)
        else:
            drift=sn/k; ds=(t*co-sn)/(k*k)
        dx=-h*sn*x+ds*v; dv=(-sn-t*co)*x-h*sn*v
        x,v=co*x+drift*v,shear*x+co*v
        for i,fi in enumerate((1,3,4)):
            px=rate[0,i];pv=rate[1,i]
            rate[0,i]=co*px+drift*pv+dx*f[j,fi]
            rate[1,i]=shear*px+co*pv+dv*f[j,fi]
        if j+1==indices[station]:
            res[station]=x
            jac[station]=rate[0]
            station+=1
    scale=np.hypot(launch[0],launch[1])
    return res/scale,jac/scale


def search():
    folder=HERE/'c1-zoom-near-c2'
    rec=json.loads((folder/'tracking.json').read_text());tag=Path(rec['solution']).stem
    centres=((751.05,765.4),(1181.5,1194.,1208.08,1222.,1234.5),(1423.14,))
    z,fc,s,reference=prepare(folder,tag,.35,planes=np.concatenate(centres)*.001)
    mid=.5*(z[:-1]+z[1:])
    cols=[FACTOR*read_axis(WORK/f'{tag}-{key}-bore-0.75mm.npz')(mid) for key in ('adl','tl')]
    field=np.ascontiguousarray(np.column_stack((fc[:,0],fc[:,2:4],*cols)))
    bias=rec['field']['bias']
    ref=np.array([reference[1],reference[2],bias['adapter_lens'],bias['probe_tl12_lens']])
    results=[]; attempts=0
    c3_list=(.5,1.,2.,3.,4.,5.,5.75,6.472,7.,8.)
    for z2 in centres[0]:
        for z3 in centres[1]:
            ix=np.searchsorted(z,np.array([z2,z3,centres[2][0]])*.001)
            carried=[]
            for c3 in c3_list:
                seeds=[(ref[0],a,t) for a in (-100.,-45.,5.,45.,100.) for t in (-160.,-95.,-40.,85.)]
                solutions=[]
                for seed in [*carried,*seeds]:
                    fit=root(images,seed,args=(z,field,s,ref,c3,ix),jac=True,options={'maxfev':45})
                    attempts+=1
                    if np.max(np.abs(fit.fun))>1e-5 or not fit.success:
                        continue
                    c2,adl,tl=fit.x
                    if not(0.<c2<40. and abs(adl)<=170. and abs(tl)<=260.):
                        continue
                    if any(np.linalg.norm(fit.x-p)<.02 for p in solutions):
                        continue
                    solutions.append(fit.x)
                    q=np.array([c2,c3,adl,tl]);r,n,alpha,size,defocus,foil,mini,pole=evaluate(q,z,field,s,ref)
                    inter=r[:min(n,len(r))];inter=inter[inter<1512.5]
                    if len(inter)!=4 or not(450<inter[0]<550 and 750<inter[1]<830 and
                                            1179.1<inter[2]<1237.2 and 1420<inter[3]<1512.5):
                        continue
                    results.append(dict(currents_A=dict(zip(('c2','c3','adl','tl'),q.tolist())),
                        targets_mm=[z2,z3,centres[2][0]],roots_mm=r[:min(n,len(r))].tolist(),sample_rms_angle_mrad=alpha,
                        specimen_rms_nm=size,foil_rms_um=foil,mini_rms_um=mini,upper_pole_rms_um=pole))
                carried=solutions
    results.sort(key=lambda v:-v['sample_rms_angle_mrad'])
    out=dict(method='First-order DC and current-tangent shooting with no specimen diameter or focus score. ADL and TL12 may change in this separate study. Select candidates by occupied angle. A new nonlinear trace is needed to verify each candidate.',
             reference=rec['solution'],source_count=rec['count'],source_current_uA=rec['surface']['specimen_uA'],
             target_planes_mm=centres,weak_positive_C3_A=c3_list,attempts=attempts,candidates=results)
    (HERE/'c1-zoom'/'angle-relay-screen.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('Fits',attempts,'eligible',len(results),'top',results[:4],flush=True)


if __name__=='__main__':
    search()
