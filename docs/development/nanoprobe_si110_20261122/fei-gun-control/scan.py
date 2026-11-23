"""Make a matched voltage sweep with first-hit current and nested source growth."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import get_num_threads, set_num_threads
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from gun_field import HERE, WORK, key
from source import distribution, emit
from transport import field_at, transport, momentum_energy


def digest(file):
    with Path(file).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_field(wiring,fine,v):
    name=key(wiring,fine,v)
    rec=json.loads((HERE/(name+'.json')).read_text())
    with np.load(WORK/(name+'.npz')) as d:
        assert str(d['sha'])==rec['sha']
        arrays=tuple(d[k] for k in ('r','z','voltage','cut_index','lookup','origin','inverse','grad','phi0'))
    r,z,phi=arrays[:3]
    bore=min(q['inner_m'] for q in rec['request']['rings'])
    support=.01*np.minimum(np.maximum(0.,z),bore)
    ix=np.maximum(0,np.searchsorted(r,support,side='right')-1); ix[ix<2]=0
    ss=r[ix]**2
    slopes=np.divide(phi[ix,np.arange(len(z))]-phi[0],ss,out=np.zeros(len(z)),where=ss>0)
    return (*arrays,ss,slopes),rec


def run(wiring,v,fine,scale,n,source_field,bmag,emission_fit):
    ef,rec=load_field(wiring,fine,v)
    phase,weight,angle,law=emit(source_field,n)
    rows=rec['stops']; labels=['exit','tip_return']+[row['key'] for row in rows]
    parts=np.array([[q['start_m'],q['stop_m'],q['inner_m'],q['outer_m']] for q in rows])
    zplot=np.unique(np.r_[np.geomspace(1e-9,1e-4,72),np.arange(0.,.002,.00002),np.arange(.002,.04,.00012),np.arange(.04,.45,.001),.45])
    nplot=1024
    name=key(wiring,fine,v)+f'-step{scale:g}'+('-emission-fit' if emission_fit else '')
    out=WORK/(name+'-rays.npz')
    signature='|'.join([digest(HERE/p) for p in ('scan.py','source.py','transport.py','gun-volume-field.npz')]+[
        rec['sha'],json.dumps(rows,sort_keys=True),str(source_field),str(scale)])
    old=0
    if out.exists():
        with np.load(out) as prev:
            old_sig=str(prev['signature'])
            if old_sig==signature:
                old=len(prev['label'])
                if old>=n:
                    print('Complete',name,old,flush=True); return
                np.testing.assert_array_equal(prev['phase'],phase[:old])
                np.testing.assert_array_equal(prev['weight_uA']*old,weight[:old]*n)
                prior=(prev['label'],prev['end'],prev['track_xy'])
        archive=WORK/'prefixes'
        archive.mkdir(exist_ok=True)
        prefix=f'{name}-{old or "superseded"}-{hashlib.sha256(old_sig.encode()).hexdigest()[:10]}'
        if not (archive/(prefix+'.npz')).exists():
            import shutil
            shutil.copy2(out,archive/(prefix+'.npz'))
            if (HERE/(name+'-currents.json')).exists():
                shutil.copy2(HERE/(name+'-currents.json'),archive/(prefix+'.json'))
    event,end,track=transport(phase[old:],ef,bmag,parts,zplot,min(n-old,max(nplot-old,0)),scale)
    if np.any(event < -2):
        raise ValueError(f'Unresolved trajectory: {np.unique(event[event < -2],return_counts=True)}')
    label=np.where(event<0,-event-1,event+2)
    if old:
        label=np.concatenate((prior[0],label));end=np.concatenate((prior[1],end));track=np.concatenate((prior[2],track))
    group=np.array(labels)[label]
    ledger={}
    for part in dict.fromkeys(labels):
        w=weight[group==part]
        ledger[part]=dict(count=len(w),uA=w.sum(),effective_count=float(w.sum()**2/(w@w)) if len(w) else 0.)
    assert np.isclose(sum(w['uA'] for w in ledger.values()),15.,rtol=0.,atol=1e-11)
    ex=group=='exit'
    initial_potential=np.array([field_at(x,y,z,ef)[0] for x,y,z in phase[:,:3]])
    k0=np.array([momentum_energy(*u) for u in phase[:,3:]])
    k1=np.array([momentum_energy(*u) for u in end[ex,3:6]])
    terminal_phi=np.array([field_at(x,y,z,ef)[0] for x,y,z in end[ex,:3]])
    axis=[]
    for z in [0.,.75,3.,8.,14.,18.,22.,24.,30.,34.,42.,445.,450.]:
        val,_,_,ez=field_at(0.,0.,z*.001,ef)
        er=field_at(.0001,0.,z*.001,ef)[1]
        axis.append(dict(z_mm=z,phi_V=val,Ez_V_per_m=ez,Er_at_r_100um_V_per_m=er))
    result=dict(name=name,n_original=n,weight_effective=float(weight.sum()**2/(weight@weight)),source=law,
        field_sha256=rec['sha'],voltage_V=v,connection=wiring,step_scale=scale,mesh=rec['request']['numerics']['cells_per_bore'],
        source_current_uA=float(weight.sum()),ledger=ledger,axial_samples=axis,
        actual_apex_V_per_nm=-field_at(0.,0.,1e-10,ef)[3]*1e-9,
        max_launch_potential_V=float(np.max(np.abs(initial_potential))),
        maximum_step_energy_error_eV=float(end[:,7].max()),maximum_exit_energy_error_eV=float(np.max(abs(k1-k0[ex]-terminal_phi+initial_potential[ex]),initial=0.)),
        maximum_sampled_radius_mm=float(np.max(end[:,9])*1000),current_prefixes=[])
    for m in (1024,4096,8192,16384,32768,65536):
        if m<=n:
            result['current_prefixes'].append(dict(count=m,uA={p:float(weight[:m][group[:m]==p].sum()*n/m) for p in ledger}))
    temp=out.with_suffix('.tmp.npz')
    np.savez_compressed(temp,label=label,labels=labels,end=end,phase=phase,weight_uA=weight,polar_angle_rad=angle,
                        track_xy=track,z_plot=zplot,signature=signature,ids=np.arange(n))
    temp.replace(out)
    (HERE/(name+'-currents.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(name,n,[(p,round(d['uA'],6)) for p,d in ledger.items() if d['count']],
          'error eV',result['maximum_step_energy_error_eV'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=8192);p.add_argument('--fine',action='store_true')
    p.add_argument('--scale',type=float,default=1.);p.add_argument('--wiring',default='anode')
    p.add_argument('--source-current-fit',action='store_true',
                   help='Use a 15 uA area-integrated Murphy-Good angular law; leave the trajectory fields unchanged.')
    p.add_argument('--voltages',type=float,nargs='+',default=[50,500,1000,1200,1500,1800,2000]);args=p.parse_args()
    set_num_threads(min(get_num_threads(),14))
    baseline,_=load_field('anode',False,1200.)
    apex=-field_at(0.,0.,1e-10,baseline)[3]*1e-9
    if args.source_current_fit:
        apex=brentq(lambda field: distribution(field)[2]-15.,2.,9.)
    with np.load(HERE/'gun-volume-field.npz') as f:
        assert np.allclose(np.diff(f['z_m']),.002) and np.isclose(f['scale_m'],.008)
        b=np.ascontiguousarray(CubicSpline(f['z_m'],f['coeff'],axis=0).c.transpose(1,2,3,0))
    print('Apex field',apex,'V/nm',flush=True)
    del baseline
    for voltage in args.voltages:
        run(args.wiring,voltage,args.fine,args.scale,args.n,apex,b,args.source_current_fit)
