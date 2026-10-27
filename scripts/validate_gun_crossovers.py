"""Observe frozen-source gun waists at actual integrated first crossings."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.model import State
from temsim.optics.electron_gun.tracing import trace_feg_with_observations
from temsim.physics.gun_field_environment import instrument_gun_field_context
from temsim.physics.particle_sections import _digest_arrays
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.corrector_crossovers import measured_corrector_crossovers
from temsim.optics.condenser_topology import _strict_spatial_minima
from temsim.optics.beam_path_audit import crossover_candidates
from temsim.vacuum import bind_gun_environment


def clean(value):
    if isinstance(value, dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [clean(v) for v in value]
    if isinstance(value,np.ndarray):
        return clean(value.tolist())
    if isinstance(value,np.generic):
        return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rays',type=int,default=193)
    parser.add_argument('--step-scale',type=float,default=1.)
    parser.add_argument('--planes-json',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve completed gun observations')
    if not 0<args.step_scale<=1 or args.rays<9:
        raise ValueError('Positive bounded gun step scale and at least nine physical particles required')
    inputs=json.loads(args.inputs.read_text(encoding='utf-8'))
    state=State.from_dict(inputs)
    gun=state.electron_gun
    gun.emitter.ray_count=args.rays
    gun.trace_step_mm*=args.step_scale
    gun.drift_step_mm*=args.step_scale
    bind_gun_environment(state)
    frozen=state.to_dict()
    snapshot=capture_instrument_snapshot(state)
    planes=(np.asarray(json.loads(args.planes_json.read_text()),float) if args.planes_json else
        np.unique(np.r_[0.,np.geomspace(1e-7,1e-2,21),24.,
            359.8,360.,360.2,360.3,360.35,360.375,360.4,360.45,360.475,360.5,360.6,361.,445.,450.]))
    started=perf_counter()
    with instrument_gun_field_context(state):
        trace,cp=trace_feg_with_observations(gun,planes,args.rays)
    if state.to_dict()!=frozen or capture_instrument_snapshot(state).digest!=snapshot.digest:
        raise ValueError('Exact gun observation changed physical inputs')
    weights=np.asarray(trace.exit_bundle.weight)
    masks=np.asarray(cp.alive)&(weights>0)[None,:]
    ids=np.asarray(trace.exit_bundle.ray_id)
    source_weight=float(weights.sum())
    regions=dict(near_emission=(-1e-8,.01000001),electrostatic_source_image=(359.799,361.001))
    rows={}
    for name,(left,right) in regions.items():
        selected=np.flatnonzero((cp.z_mm>left)&(cp.z_mm<right))
        local=SimpleNamespace(z_mm=cp.z_mm[selected],**{k:getattr(cp,k)[selected]
            for k in ('x_m','y_m','tx_rad','ty_rad')})
        alive=masks[selected]
        candidates=crossover_candidates(local,alive,weights)
        cohort=np.all(alive,axis=0)&(weights>0)
        shared=SimpleNamespace(z_mm=local.z_mm,**{k:getattr(local,k)[:,cohort]
            for k in ('x_m','y_m','tx_rad','ty_rad')})
        shared_masks=np.ones(shared.x_m.shape,bool)
        waists=(_strict_spatial_minima(shared,weights[cohort],
            measured_corrector_crossovers(shared,shared_masks,weights[cohort],left,right))
            if np.count_nonzero(cohort)>=5 else [])
        def point(index):
            s=transverse_beam_statistics(*(getattr(cp,k)[index] for k in ('x_m','y_m','tx_rad','ty_rad')),
                alive=masks[index],weights=weights)
            live=masks[index]
            energy=cp.kinetic_energy_ev[index,live]
            w=weights[live]/weights[live].sum()
            q=np.c_[cp.x_m[index,live],cp.y_m[index,live]]
            t=np.c_[cp.tx_rad[index,live],cp.ty_rad[index,live]]
            q=q-w@q; t=t-w@t
            eigen,basis=np.linalg.eigh(q.T@(w[:,None]*q))
            return dict(z_mm=float(cp.z_mm[index]),statistics=asdict(s),alpha95_mrad=s.convergence_95_mrad,
                principal_rms_nm=np.sqrt(np.maximum(eigen,0))*1e9,
                principal_position_slope_covariance_m_rad=w@((q@basis)*(t@basis)),
                current_a=gun.emitted_current_a*float(weights[live].sum()/source_weight),
                kinetic_energy_min_ev=float(energy.min()),kinetic_energy_max_ev=float(energy.max()),
                kinetic_energy_weighted_mean_ev=float(np.average(energy,weights=weights[live])),
                plane_residual_max_m=float(abs(cp.plane_residual_m[index,live]).max()))
        for waist in waists:
            waist['kind']='current-weighted two-dimensional gun envelope waist'
            waist['source_current_fraction']=float(weights[cohort].sum()/source_weight)
            index=int(np.searchsorted(cp.z_mm,waist['z_mm']))
            waist['actual']=point(index)
            waist['actual_three_planes']=[point(i) for i in (index-1,index,index+1)]
        rows[name]=dict(bounds_mm=[left,right],executed_waists=waists,
            candidates_diagnostic_only=candidates,point_observations=[point(i) for i in selected],
            maximum_gap_mm=float(np.diff(cp.z_mm[selected]).max()),
            status='EXECUTED_2D_WAIST' if waists else 'NO_EXECUTED_2D_WAIST_AT_REQUESTED_PLANES')
    arrays={key:getattr(cp,key) for key in ('z_mm','x_m','y_m','tx_rad','ty_rad','flight_time_s',
        'kinetic_energy_ev','reached','alive','plane_residual_m')}
    arrays.update(weight=weights,source_id=ids,source_energy_offset_ev=trace.exit_bundle.energy_offset_ev)
    e=trace.exit_bundle
    identity=_digest_arrays(*(np.asarray(getattr(e,k)) for k in
        ('x_m','tx_rad','y_m','ty_rad','flight_time_s','energy_offset_ev','weight','alive','ray_id')))
    emission_identity=_digest_arrays(*(np.asarray(trace.emission_reference[k])
        for k in ('ray_id','position_m','direction','normal')),weights)
    report=dict(schema='exact-gun-crossover-validation-v1',utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=perf_counter()-started,snapshot_digest=snapshot.digest,rays=args.rays,
        source_population_identity=identity,emission_geometry_weight_identity=emission_identity,
        actual_emission_energy_ev=gun.emitter.emission_energy_ev,energy_spread_fwhm_ev=gun.emitter.energy_spread_fwhm_ev,
        gun_trace_step_mm=gun.trace_step_mm,gun_drift_step_mm=gun.drift_step_mm,
        exact_observation_schema=cp.schema,plane_residual_max_m=float(np.nanmax(abs(cp.plane_residual_m))),
        regions=rows,source_current_a=gun.emitted_current_a,
        exit_transmitted_count=int(e.alive.sum()),exit_current_a=gun.emitted_current_a*float(weights[e.alive].sum()/source_weight),
        unchanged_optical_parameters=True,
        scope='Actual first crossings on the production geometry-field DG path; selected gun intervals, not whole-gun continuous coverage',
        sample_or_screen_conjugacy='NOT_INFERRED_FROM_GUN_WAIST_ANGLES')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    args.output.with_suffix('.inputs.json').write_text(json.dumps(frozen,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    print(json.dumps(clean(dict(elapsed_s=report['elapsed_s'],source_identity=identity,
        plane_residual_max_m=report['plane_residual_max_m'],regions={k:dict(status=v['status'],
        waists=v['executed_waists'],candidates=v['candidates_diagnostic_only']) for k,v in rows.items()})),indent=2),flush=True)


if __name__=='__main__':
    main()
