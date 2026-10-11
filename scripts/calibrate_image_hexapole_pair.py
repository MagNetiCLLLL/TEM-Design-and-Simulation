"""Finite-field second-order HP-pair diagnostic; no defaults are written."""
import argparse
from pathlib import Path
import json
import numpy as np
from validate_image_corrector import ROOT, save, trace_rings, optical_rings, ring_metrics, inventory


def second_order(field,eps=2e-4):
    # Two eight-direction rings, including each opposite partner. Richardson
    # removes the leading fourth-order contamination of the even response.
    values=[]
    for i,angle in enumerate((eps,eps/2)):
        ring=field[:,1+8*i:9+8*i]
        values.append((.5*(ring[:,:4]+ring[:,4:])-field[:,:1])/angle**2)
    return (4*values[1]-values[0])/3


def execute(args):
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.instrument_magnetic import active_column_events,column_dipole_fields
    before=inventory(); args.output.mkdir(parents=True,exist_ok=True)
    state=default_state();AssemblyCatalog().apply(state,AssemblySelection('FEG','C3 + Image Corrector','No Energy Filter'))
    settings=json.loads(args.candidate.read_text())['candidates'][0]['percent']
    for lens in state.lenses:
        if lens.key in settings:lens.percent=settings[lens.key]
    old_descan=bool(state.descan_deflector.scan_enabled)
    state.descan_deflector.scan_enabled=False
    state.acceleration_enabled=True;state.acceleration_backend='Require GPU'
    source=float(state.sample.z_mm); stop=float(state.image_corrector_system.sad_plane.z_mm)
    nonzero_events=[r for r in active_column_events(state) if source<=r[0]<=stop and (r[1] or r[2])]
    nonzero_coils=[p.key for p in column_dipole_fields(state) if p.field_support_mm[1]>=source
                   and p.field_support_mm[0]<=stop and (p.bx_t or p.by_t)]
    if nonzero_events or nonzero_coils:raise ValueError(f'On-axis bench contains active transverse drives: {nonzero_events}, {nonzero_coils}')
    save(args.output/'inputs.json',capture_instrument_snapshot(state).to_dict())
    hp1,hp2=state.image_corrector_system.hp1_hexapole,state.image_corrector_system.hp2_hexapole
    strength=float(hp1.strength_m3)
    eps=2e-4;theta=np.arange(8)*np.pi/4
    initial=np.zeros((4,17));initial[2,1:]=np.concatenate((eps*np.cos(theta),eps/2*np.cos(theta)))
    initial[3,1:]=np.concatenate((eps*np.sin(theta),eps/2*np.sin(theta)))
    coefficients={};full={};receipts=[]
    with backend_receipts(receipts):
        for name,s1,s2,angle in [('hp1_only',strength,0.,0.),('hp2_normal_only',0.,strength,0.),('hp2_skew_only',0.,strength,np.pi/6)]:
            hp1.strength_m3=s1;hp2.strength_m3=s2;hp2.orientation_rad=angle
            full[name]=trace_rings(state,source,[stop],initial,spherical=False,hexapoles=True,step=.025)[str(stop)]
            coefficients[name]=second_order(full[name])
        scale=np.array((1.,1.,.05,.05))[:,None]
        design=np.column_stack([(coefficients[k]*scale).ravel() for k in ('hp2_normal_only','hp2_skew_only')])
        target=-(coefficients['hp1_only']*scale).ravel()
        vector=np.linalg.lstsq(design,target,rcond=None)[0]
        ratio=float(np.linalg.norm(vector));orientation=float(np.arctan2(vector[1],vector[0])/3)
        hp1.strength_m3=strength;hp2.strength_m3=strength*ratio;hp2.orientation_rad=orientation
        full['both']=trace_rings(state,source,[stop],initial,spherical=False,hexapoles=True,step=.025)[str(stop)]
        coefficients['both']=second_order(full['both'])
        full['both_fine']=trace_rings(state,source,[stop],initial,spherical=False,hexapoles=True,step=.0125)[str(stop)]
        coefficients['both_fine']=second_order(full['both_fine'])
        rings=optical_rings()
        finite={name:trace_rings(state,source,[stop],rings,spherical=cs,hexapoles=hp,step=.025)[str(stop)]
                for name,cs,hp in [('linear',False,False),('hp',False,True),('cs_and_hp',True,True),('cs',True,False)]}
    save(args.output/'report.json',dict(status='FINITE_PAIR_DIAGNOSTIC_NOT_OBJECTIVE_CS_QUALIFICATION',
        scope='On-axis static calibration copy; actual distributed analytic round/HP fields. Sample-point launch does not exercise the native objective Cs kick at the coincident centre plane.',
        source_and_defaults_unchanged=before==inventory(),diagnostic_overrides=dict(descan_scan_enabled=False,original_descan_scan_enabled=old_descan,lens_percent=settings),
        pair=dict(hp1_strength_m3=strength,hp2_strength_m3=strength*ratio,hp2_ratio=ratio,hp2_orientation_rad=orientation,
                  fit='Second-order output phase-space coefficient, sampled at eight pupil angles and two radii, fourth-order Richardson extrapolation; slope weighted by 0.05m'),
        coefficients=coefficients,weighted_relative_second_order_residual=float(np.linalg.norm(coefficients['both']*scale)/np.linalg.norm(coefficients['hp1_only']*scale)),
        weighted_relative_halfstep_difference=float(np.linalg.norm((coefficients['both_fine']-coefficients['both'])*scale)/np.linalg.norm(coefficients['hp1_only']*scale)),
        finite_rings={name:ring_metrics(value,finite['linear']) for name,value in finite.items()},backend_receipts=receipts))
    np.savez_compressed(args.output/'fields.npz',initial=initial,**full,**{'ring_'+k:v for k,v in finite.items()})
    print(ratio,orientation,coefficients['both'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,default=ROOT/'outputs/image_corrector_20261011/candidates/candidate-report.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/image_corrector_20261011/finite-pair')
    execute(parser.parse_args())
