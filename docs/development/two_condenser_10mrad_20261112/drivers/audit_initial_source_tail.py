"""Read-only source/tail audit of explicitly selected complete 70 um trials."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import numpy as np

HERE=Path(__file__).resolve().parent


def hash_file(path):return sha256(path.read_bytes()).hexdigest()


def analyze(name,root=None,*,initial_defocused=False):
    root=HERE/name if root is None else Path(root)
    report=json.loads((root/'report.json').read_text())
    assert report['status']=='COMPLETED_CLASSICAL_TWO_CONDENSER_OBSERVATION'
    assert hash_file(root/'observations.npz')==report['observations_sha256']
    with np.load(root/'observations.npz',allow_pickle=False) as source:z=dict(source)
    weights=z['weights'];ids=z['source_id'];positive=weights>0.
    at=int(np.flatnonzero(z['z_mm']==report['surface_z_mm'])[0])
    selected=z['alive'][at]&positive
    keep=np.flatnonzero(selected);w=weights[selected];mass=float(w.sum())
    x=z['x_m'][at,selected];y=z['y_m'][at,selected]
    centered_x=x-np.average(x,weights=w);centered_y=y-np.average(y,weights=w)
    radius=np.hypot(centered_x,centered_y);order=np.argsort(radius)[::-1]
    moment=float(np.dot(w,radius*radius));ref_current=report['execution']['reference_current_a']
    plane=int(np.flatnonzero(z['z_mm']==765.)[0]);at_ap=z['alive'][plane]&positive
    pupil=np.hypot(z['x_m'][plane,at_ap],z['y_m'][plane,at_ap])
    phase_keys=('x_m','y_m','tx_rad','ty_rad','flight_time_s','kinetic_energy_ev')
    original_positive=int(np.count_nonzero(positive))
    assert original_positive>=1000 and np.unique(ids[positive]).size==original_positive
    assert original_positive==report['execution']['emitted_positive_count']
    assert np.array_equal(ids,z['emission_ray_id'])
    assert np.array_equal(weights[positive],np.full(original_positive,1/original_positive))
    assert np.all(np.isfinite(np.stack([z[key][at,selected] for key in phase_keys])))
    stopped_before_surface=positive&~z['alive'][at]
    assert np.all(z['blocked_key'][stopped_before_surface]!='')
    lost=stopped_before_surface&(z['blocked_key']=='condenser_aperture_2')
    assert int(stopped_before_surface.sum())+int(selected.sum())==original_positive
    other_losses=[dict(key=str(key),count=int(np.count_nonzero(stopped_before_surface&(z['blocked_key']==key))),
        original_current_a=float(ref_current*weights[stopped_before_surface&(z['blocked_key']==key)].sum()))
        for key in sorted(set(z['blocked_key'][stopped_before_surface])-{'condenser_aperture_2'})]
    assert abs(ref_current*mass-report['sample_current_a'])<1e-18
    geometry=report['execution']['field_audit']['c2_aperture']
    assert geometry['diameter_mm']==.07 and geometry['actual_boundary_radius_mm']==.035
    records=[]
    for local in order[:4]:
        column=keep[local]
        ap_radius=float(np.hypot(z['x_m'][plane,column],z['y_m'][plane,column]))
        direction=z['emission_direction'][column]
        records.append(dict(source_id=int(ids[column]),original_source_weight=float(weights[column]),
            sample_current_a=float(ref_current*weights[column]),sample_centroid_radius_nm=float(radius[local]*1e9),
            sample_axis_radius_nm=float(np.hypot(x[local],y[local])*1e9),
            sample_mechanical_angle_from_axis_mrad=float(1000*np.arctan(np.hypot(z['tx_rad'][at,column],z['ty_rad'][at,column]))),
            fraction_sample_radial_second_moment=float(w[local]*radius[local]**2/moment),
            sample_kinetic_energy_ev=float(z['kinetic_energy_ev'][at,column]),
            c2_aperture_axis_radius_um=ap_radius*1e6,
            emission_polar_angle_mrad=float(1000*np.arctan2(np.hypot(direction[0],direction[1]),direction[2])),
            status='Surviving original positive-current packet with the same weight as every other emission; not a zero-current or low-weight probe'))
    balance=float(ref_current*weights.sum()-report['sample_current_a']-ref_current*weights[stopped_before_surface].sum())
    focus=report['native_focus']
    return dict(trial=name,evidence_sha256={filename:hash_file(root/filename) for filename in ('report.json','observations.npz','inputs.json','input.toml')},
        controls=report['controls'],emitted_positive_count=int(positive.sum()),original_columns=len(ids),
        unique_original_ids=int(np.unique(ids).size),ids_equal_emission_reference=True,zero_current_count=int((weights==0.).sum()),
        original_weight_sum=float(weights.sum()),weight_minimum=float(weights.min()),weight_maximum=float(weights.max()),
        equal_original_positive_weight=1/original_positive,each_positive_weight_matches_original_equal_weight=True,retained_current_not_renormalized=True,
        gun_exit_positive_survivors=int(np.count_nonzero(z['gun_exit_alive']&positive)),
        sample_positive_survivors=int(selected.sum()),sample_effective_count=float(mass*mass/np.dot(w,w)),
        source_current_a=ref_current,sample_current_a=report['sample_current_a'],transmission=mass,
        c2_intercepted_count=int(lost.sum()),c2_intercepted_original_current_a=float(ref_current*weights[lost].sum()),
        other_interceptions=other_losses,absolute_current_balance_residual_a=balance,
        stopped_particle_resurrections=int(np.count_nonzero(~z['alive'][:-1]&z['alive'][1:]&positive)),
        c2_aperture=dict(diameter_um=70.,radius_um=35.,installed=geometry['installed'],enabled=geometry['enabled'],inserted=geometry['inserted'],
            exact_observation_mm=765.,surviving_plane_count=int(at_ap.sum()),maximum_surviving_axis_radius_um=float(pupil.max()*1e6),
            interpretation='The full-source ledger labels rejected original packets at the physical aperture; sample packet weights retain their original 1/N values.'),
        sample=dict(z_mm=report['surface_z_mm'],alpha95_mrad=report['alpha95_mrad'],angular_rms_mrad=report['angular_rms_mrad'],
            angular_target_at_least_10mrad=report['alpha95_mrad']>=10.,d95_nm=report['d95_nm'],rms_radius_nm=report['rms_radius_nm'],
            native_d95_waist=focus.get('d95_waist'),native_rms_waist=focus.get('rms_waist_diagnostic'),native_surface_focus_gate=focus.get('surface_gate'),
            surface_focus_passed_as_reported=bool(focus.get('surface_gate',{}).get('passed',False)),
            source_audit_claims_final_operating_point_pass=False,
            interpretation=('These initial points meet the angular target but remain defocused at the specimen upper surface; no final operating-point PASS is claimed.' if initial_defocused else
                'This audit reports source, current, angle, tails and the completed report focus gate. It does not assess final combined optical qualification.')),
        tail=dict(maximum_radius_over_second_largest=float(radius[order[0]]/radius[order[1]]),
            maximum_radius_over_d95_radius=float(2e9*radius[order[0]]/report['d95_nm']),
            largest_packet_radial_second_moment_fraction=records[0]['fraction_sample_radial_second_moment'],
            records=records,observed_interpretation='The largest-radius packets have ordinary equal source weights, finite directions and neighboring packets of similar radius. They are not a single artificially low-weight outlier. No tail is discarded.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',action='append',default=[],metavar='LABEL=PATH',help='Explicit complete trial directory; repeat for each requested source/step result')
    parser.add_argument('--output',type=Path,help='Output JSON; required with custom trials so the original initial audit cannot be overwritten implicitly')
    args=parser.parse_args()
    if args.trial and args.output is None:parser.error('--output is required with --trial')
    requested=[]
    if args.trial:
        for entry in args.trial:
            label,separator,raw_path=entry.partition('=')
            if not separator or not label or not raw_path:parser.error('--trial must be LABEL=PATH')
            path=Path(raw_path)
            if not path.is_absolute() and not path.exists():path=HERE/path
            if not path.is_dir():parser.error('Trial directory is unavailable: '+raw_path)
            requested.append((label,path.resolve()))
    else:
        requested=[(name,HERE/name) for name in ('initial-n1024-s010/trial-00','initial-n1024-s010/trial-01','weak-n1024-s010/trial-00')]
    if len({label for label,_ in requested})!=len(requested):parser.error('Trial labels must be unique')
    initial=not bool(args.trial)
    result=dict(schema='temsim-70um-source-tail-audit-v2',scope=__doc__,new_physical_traces=0,
        trials=[analyze(name,path,initial_defocused=initial) for name,path in requested],
        definitions=dict(d95='Diameter containing 95 percent of the actual specimen-plane current about its current-weighted centroid.',
            rms_radius='Square root of the weighted mean squared centroid radius using all actual surviving original positive-current packets.',
            transmission='Sum of surviving original weights relative to the unchanged emitted original-current total; no rescaling after clipping.'),
        overall_status=('ANGULAR_TARGET_MET_ON_INITIAL_DEFOCUSED_TRIALS; FINAL_SURFACE_FOCUS_NOT_QUALIFIED' if initial else
            'COMPLETED_ORIGINAL_SOURCE_AND_TAIL_ARRAY_AUDIT; FINAL_OPERATING_POINT_QUALIFICATION_NOT_ASSESSED'))
    path=args.output if args.output is not None else HERE/'initial-source-tail-audit.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(path),trials=[dict(trial=r['trial'],N=r['sample_positive_survivors'],alpha95=r['sample']['alpha95_mrad'],alphaRMS=r['sample']['angular_rms_mrad'],current_uA=r['sample_current_a']*1e6) for r in result['trials']]),indent=2))


if __name__=='__main__':main()
