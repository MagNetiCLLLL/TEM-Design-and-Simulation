"""Recover unsaved paths of actual hit and transmitted IDs for the ray diagram."""
import numpy as np
from numba import get_num_threads, set_num_threads
from scipy.interpolate import CubicSpline

from gun_field import HERE, WORK, key, study_field_reader
from transport import transport


def recover():
    set_num_threads(min(14,get_num_threads()))
    load_field = study_field_reader()
    results = {}
    with np.load(HERE/'gun-volume-field.npz') as mag:
        b = np.ascontiguousarray(CubicSpline(mag['z_m'],mag['coeff'],axis=0).c.transpose(1,2,3,0))
    for v in (50,500,1200):
        ef, receipt = load_field('anode',True,v)
        parts = np.array([[row[k] for k in ('start_m','stop_m','inner_m','outer_m')] for row in receipt['stops']])
        name = key('anode',True,v)+'-step0.5'
        with np.load(WORK/(name+'-rays.npz')) as data:
            all_labels = data['labels'][data['label']]
            angles = data['polar_angle_rad']
            ids = []
            for label in np.unique(all_labels):
                group = np.flatnonzero(all_labels==label)
                ordered = group[np.argsort(angles[group])]
                ids.extend(ordered[np.linspace(0,len(ordered)-1,min(40,len(ordered)),dtype=int)])
            ids = np.unique(ids)
            complement = np.setdiff1d(data['ids'],ids)
            ids = np.sort(np.r_[ids,complement[np.linspace(0,len(complement)-1,1024-len(ids),dtype=int)]])
            assert len(np.unique(ids))==1024 and np.all(data['weight_uA'][ids]>0)
            zplot=data['z_plot']
            # Exact replay of the selected original rays, with the same field,
            # step limits and weights. No current estimator uses this subset.
            events,end,trace = transport(data['phase'][ids],ef,b,parts,zplot,len(ids),.5)
            calculated = np.where(events<0,-events-1,events+2)
            np.testing.assert_array_equal(calculated,data['label'][ids])
            np.testing.assert_array_equal(end,data['end'][ids])
            prefix=f'v{v}_'
            results.update({prefix+k:val for k,val in {
                'id':ids,'labels':all_labels[ids],'theta':angles[ids],'weight_uA':data['weight_uA'][ids],
                'xy_m':trace,'end_xyz_m':end[:,:3]}.items()})
            results['z_m']=zplot
    np.savez_compressed(HERE/'representative-trajectories.npz',**results)


if __name__=='__main__':
    recover()
