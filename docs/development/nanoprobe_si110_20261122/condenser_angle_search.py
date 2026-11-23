"""Screen condenser winding space for occupied angle with no specimen-focus penalty."""
import argparse
import json
from pathlib import Path

import numpy as np
from numba import set_num_threads, get_num_threads

from condenser_zoom import CHANNELS, FACTOR, HERE, WORK, prepare, read_axis, search_table


def survey(folder,tangent):
    z,field,source,ref = prepare(folder,tangent,pitch_mm=.46)
    set_num_threads(min(14,get_num_threads()))
    # Retain weak, positive C3 for the Normal branch. Reverse polarity is
    # a separate control. Search both C2 polarities and include zero excitation.
    possible_c2 = np.arange(-120.,120.01,.25)
    possible_c3 = np.arange(-8.,8.01,.125)
    c2,c3 = np.meshgrid(possible_c2,possible_c3,indexing='ij')
    drives = np.column_stack((np.full(c2.size,ref[0]),c2.ravel(),c3.ravel(),np.full(c2.size,ref[3])))
    drives = np.vstack((drives,ref))
    data = search_table(drives,z,field,source,ref)
    record = json.loads((folder/'tracking.json').read_text())
    dc = read_axis(WORK/f'{Path(record["solution"]).stem}-dc-bore-{record["bore_step_mm"]:g}mm.npz')
    end = z[-1]
    columns = np.array([read_axis(WORK/f'{tangent}-{key}-bore-0.75mm.npz')(end) for key in CHANNELS])
    rotation = FACTOR*(dc(end)+(drives-ref)@columns)
    with np.load(record['gun_source']) as gun:
        w = gun['weight']
        p = gun['phase'][:,:4]-np.average(gun['phase'][:,:4],axis=0,weights=w)
        xy,vel = p[:,0]+1j*p[:,1],p[:,2]+1j*p[:,3]
        angular_momentum = np.average((xy.conj()*(vel-1j*FACTOR*dc(.45)*xy)).imag,weights=w)
    # Real unit-determinant meridional transfer preserves mean canonical angular momentum.
    # The Larmor boundary term matters for the deliberately defocused comparisons.
    data[:,7] = 1e3*np.sqrt(np.maximum(0.,(data[:,7]*1e-3)**2 + (rotation*data[:,8]*1e-9)**2
                                      + 2*rotation*angular_momentum))
    count = data[:,6].astype(int)
    c1 = (count>=1)&(data[:,0]>450.)&(data[:,0]<550.)
    condenser = c1&(count>=2)&(data[:,1]>750.)&(data[:,1]<830.)&((count==2)|(data[:,2]>1069.5))
    hp = condenser&(count>=3)&(data[:,2]>1179.1)&(data[:,2]<1237.2)
    relay = hp&(count>=4)&(data[:,3]>1420.)&(data[:,3]<1512.5)&((count==4)|(data[:,4]>1512.5))
    established = relay&(abs(data[:,3]-1423.135)<.5)
    chains = dict(all=np.ones(len(drives),bool),C1_image=c1,condenser_crossings=condenser,
                  first_corrector_crossing=hp,all_four_before_Mini=relay,postscan_within_half_mm=established)
    report = {}
    for sign,mask in [('normal_positive', (drives[:,2]>=0)&(drives[:,1]>0)),('all_signed',np.ones(len(drives),bool))]:
        entries = {}
        for name,test in chains.items():
            ids = np.flatnonzero(mask&test)
            if not len(ids):
                entries[name] = dict(count=0)
                continue
            best = ids[np.argmax(data[ids,7])]
            def entry(j):
                return dict(currents_A=drives[j].tolist(),rms_angle_mrad=data[j,7],
                            radial_waists_mm=data[j,:min(6,count[j])].tolist(),total_waist_count=int(count[j]),
                            sample_rms_nm=data[j,8],foil_rms_um=data[j,10],mini_rms_um=data[j,11],upper_pole_rms_um=data[j,12])
            entries[name] = dict(count=len(ids),highest=entry(best),
                     above_rms_0p1_mrad=int(np.count_nonzero(data[ids,7]>=.1)),
                     above_rms_1_mrad=int(np.count_nonzero(data[ids,7]>=1.)),
                     above_rms_10_mrad=int(np.count_nonzero(data[ids,7]>=10.)))
        report[sign]=entries
    output=dict(reference=folder.name,field_tangent=tangent,source_moments_from_original=len(w),
                c1_A=float(ref[0]),objective_A=float(ref[-1]),C2_range_A=[-120.,120.],C3_range_A=[-8.,8.],
                increments_A=[.25,.125],tested=len(drives),specimen_focus_penalty=False,results=report,
                method='First-order round transport with nonlinear DC and differential winding fields. Ranking uses the two-coordinate RMS mechanical slope including the Larmor boundary term. Extremes require a new nonlinear field solve and fresh finite-source trace.')
    (HERE/'c1-zoom'/'angle-first-screen.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    for sign,entries in report.items():
        for name,item in entries.items():
            if item['count']:
                v=item['highest']; print(sign,name,item['count'],np.round(v['currents_A'],3),round(v['rms_angle_mrad'],6),np.round(v['radial_waists_mm'],1),round(v['mini_rms_um'],2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',default='c1-zoom-near-c2')
    parser.add_argument('--tangent',default='global-b932b73416404799')
    args=parser.parse_args()
    survey(HERE/args.label,args.tangent)
