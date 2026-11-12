"""Read-only controlled final source/step comparisons; no new physics execution."""
from pathlib import Path
from hashlib import sha256
import json
import tomllib
import numpy as np
from temsim.physics.beam_statistics import transverse_beam_statistics

HERE=Path(__file__).resolve().parent
NAMES=('focused4096-s005','final4096-s010','reference2048-s005')
SOURCE_ARRAYS=('weights','source_id','source_azimuth_rad','emission_ray_id','emission_position_m','emission_direction','emission_normal',
    'gun_exit_x_m','gun_exit_y_m','gun_exit_tx_rad','gun_exit_ty_rad','gun_exit_flight_time_s','gun_exit_alive')


def file_hash(path):return sha256(path.read_bytes()).hexdigest()


def array_hash(array):
    a=np.ascontiguousarray(array)
    return sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()


def relative_change(primary,reference):
    return dict(primary=primary,reference=reference,delta=primary-reference,
        relative_delta_percent=100*(primary-reference)/reference,denominator='reference')


def main():
    cases={};arrays={};profiles={};requests={};providers={};evidence={}
    for name in NAMES:
        root=HERE/name
        report=json.loads((root/'report.json').read_text())
        assert report['status']=='COMPLETED_CLASSICAL_TWO_CONDENSER_OBSERVATION'
        assert file_hash(root/'observations.npz')==report['observations_sha256']
        with np.load(root/'observations.npz',allow_pickle=False) as source:z=dict(source)
        arrays[name]=z;profiles[name]=tomllib.loads((root/'input.toml').read_text());requests[name]=json.loads((root/'request.json').read_text())
        w=z['weights'];positive=w>0.;count=int(positive.sum());ids=z['source_id']
        assert count>=1000 and np.unique(ids[positive]).size==count and np.array_equal(ids,z['emission_ray_id'])
        assert np.array_equal(w[positive],np.full(count,1/count))
        at=int(np.flatnonzero(z['z_mm']==report['surface_z_mm'])[0]);live=z['alive'][at]&positive
        mass=float(w[live].sum());current=report['execution']['reference_current_a']
        stats=transverse_beam_statistics(z['x_m'][at],z['y_m'][at],z['tx_rad'][at],z['ty_rad'][at],alive=live,weights=w)
        recomputed=dict(d95_nm=2e9*stats.radius_95_m,rms_radius_nm=1e9*stats.radius_rms_m,
            alpha95_mrad=stats.convergence_95_mrad,angular_rms_mrad=1000*stats.convergence_rms_rad)
        checks={key:bool(np.isclose(value,report[key],rtol=1e-13,atol=1e-15)) for key,value in recomputed.items()}
        assert all(checks.values())
        lost=positive&~z['alive'][at]
        losses={str(key):dict(count=int(np.count_nonzero(lost&(z['blocked_key']==key))),
            current_a=float(current*w[lost&(z['blocked_key']==key)].sum())) for key in sorted(set(z['blocked_key'][lost]))}
        assert '' not in losses
        balance=float(current*w.sum()-current*mass-sum(value['current_a'] for value in losses.values()))
        x=z['x_m'][at,live];y=z['y_m'][at,live];lw=w[live]
        radii=np.hypot(x-np.average(x,weights=lw),y-np.average(y,weights=lw));order=np.argsort(radii)[::-1];columns=np.flatnonzero(live)
        second_moment=float(np.dot(lw,radii*radii));tails=[]
        for local in order[:3]:
            col=int(columns[local])
            tails.append(dict(source_id=int(ids[col]),original_weight=float(w[col]),sample_centroid_radius_nm=float(radii[local]*1e9),
                current_a=float(current*w[col]),fraction_radial_second_moment=float(lw[local]*radii[local]**2/second_moment),
                angle_from_axis_mrad=float(1000*np.arctan(np.hypot(z['tx_rad'][at,col],z['ty_rad'][at,col])))))
        topology={label:{cohort:dict(status=row['status'],region_centers_mm=[region.get('representative_z_mm') for region in row['regions']],unresolved=row['unresolved'])
            for cohort,row in populations.items()} for label,populations in report['topology'].items()}
        cases[name]=dict(emitted_original_positive_count=count,unique_original_ids=int(np.unique(ids[positive]).size),zero_current_count=int((w==0.).sum()),
            original_weight_sum=float(w.sum()),each_original_positive_weight=1/count,original_equal_weights_verified=True,
            surviving_positive_count=int(live.sum()),effective_count=float(mass*mass/np.dot(lw,lw)),transmission=mass,
            source_current_a=current,sample_current_a=current*mass,physical_loss_ledger=losses,absolute_current_balance_residual_a=balance,
            no_stopped_particle_resurrection=not bool(np.any(~z['alive'][:-1]&z['alive'][1:]&positive)),
            controls=report['controls'],surface_z_mm=report['surface_z_mm'],statistics=recomputed,report_statistics_reproduced_from_arrays=checks,
            native_d95_waist=report['native_focus']['d95_waist'],native_rms_waist=report['native_focus']['rms_waist_diagnostic'],
            native_surface_focus_gate=report['native_focus']['surface_gate'],direct_surface_focus_as_reported=report['direct_surface_focus'],
            topology=topology,largest_radius_original_particles=tails,
            mesh=dict(integration_nodes=len(z['integration_z_mm']),checkpoints=len(z['z_mm']),
                integration_z_sha256=array_hash(z['integration_z_mm']),checkpoint_z_sha256=array_hash(z['z_mm'])),
            identities=report['execution']['field_audit']['identities'])
        providers[name]=report['execution']['field_audit']['field_ownership']['sources']
        evidence[name]={filename:file_hash(root/filename) for filename in ('report.json','observations.npz','inputs.json','input.toml','request.json')}
    fine,coarse,reference=NAMES
    equality={key:bool(np.array_equal(arrays[fine][key],arrays[coarse][key],equal_nan=True)) for key in SOURCE_ARRAYS}
    equality['gun_exit_checkpoint_kinetic_energy_ev']=bool(np.array_equal(arrays[fine]['kinetic_energy_ev'][0],arrays[coarse]['kinetic_energy_ev'][0]))
    equality['sample_alive_membership']=bool(np.array_equal(arrays[fine]['alive'][np.flatnonzero(arrays[fine]['z_mm']==cases[fine]['surface_z_mm'])[0]],arrays[coarse]['alive'][np.flatnonzero(arrays[coarse]['z_mm']==cases[coarse]['surface_z_mm'])[0]]))
    assert all(equality.values())
    mesh_equal={key:bool(np.array_equal(arrays[fine][key],arrays[reference][key])) for key in ('integration_z_mm','z_mm')}
    assert all(mesh_equal.values())
    request_equal={key:requests[fine][key]==requests[reference][key] for key in ('controls','checkpoint_population_budget','focus_planes_mm','requested_planes_mm','topology_intervals','step')}
    assert all(request_equal.values())
    source_laws={}
    for name,p in profiles.items():
        tip=dict(p['devices']['feg_tip']);tip.pop('ray_count')
        source_laws[name]=dict(tip_parameters_excluding_count=tip,source_model=p['gun_source_model'],gun_controls=p['gun_controls'],
            gun_voltage_inputs={key:p['devices'][key] for key in ('feg_electrostatic_lens','feg_extractor','feg_accelerator')})
    assert source_laws[fine]==source_laws[coarse]==source_laws[reference]
    assert providers[fine]==providers[coarse]==providers[reference]
    chain_path=HERE/'final-scene-count-diagnostic.json';chain=json.loads(chain_path.read_text())
    assert all(row['reconstruction_matches_executed'] for row in chain['scenes'].values())
    assert chain['scene_payload_differences']==[['.gun_inputs.attributes.emitter.attributes.ray_count',4096,2048]]
    step_changes={key:relative_change(cases[fine]['statistics'][key],cases[coarse]['statistics'][key]) for key in ('d95_nm','rms_radius_nm','alpha95_mrad','angular_rms_mrad')}
    source_changes={key:relative_change(cases[fine]['statistics'][key],cases[reference]['statistics'][key]) for key in ('d95_nm','rms_radius_nm','alpha95_mrad','angular_rms_mrad')}
    source_changes['sample_current_a']=relative_change(cases[fine]['sample_current_a'],cases[reference]['sample_current_a'])
    result=dict(schema='temsim-final-two-condenser-controlled-comparison-v1',scope=__doc__,new_trajectories=0,new_fields_solved=0,
        evidence_sha256=evidence,cases=cases,common_source_law=source_laws[fine],source_law_equal_except_count=True,
        fine4096_vs_coarse4096=dict(exact_original_source_array_equalities=equality,comparison_denominator=coarse,changes=step_changes,
            observed_interpretation='The exact same 4096 original source and identical controls retain the same pupil membership and native D95 observation minimum at the surface under both integration steps.'),
        fine4096_vs_fine2048=dict(exact_mesh_equalities=mesh_equal,exact_request_equalities=request_equal,comparison_denominator=reference,
            changes=source_changes,sampling_convergence='NOT_ESTABLISHED',reference2048_surface_focus_passed=cases[reference]['native_surface_focus_gate']['passed'],
            observed_interpretation='At identical controls and exact integration/checkpoint meshes, the 2048 source has native D95 focus at -16 nm and fails the surface gate. Neither source-size nor continuous-source focus convergence is claimed.'),
        field_identity_analysis=dict(recorded_provider_lists_exactly_equal=True,provider_count=len(providers[fine]),
            recorded_numerical_potential_digest_same_for_4096_steps=cases[fine]['identities']['potential_digest']==cases[coarse]['identities']['potential_digest'],
            recorded_numerical_potential_digest_same_for_4096_and_2048=cases[fine]['identities']['potential_digest']==cases[reference]['identities']['potential_digest'],
            scene_count_diagnostic_sha256=file_hash(chain_path),scene_count_diagnostic=chain,
            conclusion='The only restored scene payload difference is emitter.ray_count. Using each scene identity, the identical recorded 20-provider identities and the default static gauge exactly reconstructs both executed aggregate numerical digests. This explains the count-dependent aggregate identity without a difference in recorded provider representations.',
            limitation='No independent byte-by-byte comparison of reconstructed field coefficient buffers was performed; no fields were re-solved or new field queries executed.'),
        final_scope='Observed 4096-particle native D95 focus at the specimen surface, alpha95 above 10 mrad, and a bounded direct final focus gate under two integration steps. RMS-waist position is a distinct +72 nm diagnostic. The 2048 reference fails focus, so there is no finite-source sampling-convergence claim. Upstream actual-population topology retains a local aperture-membership unresolved segment where reported.')
    (HERE/'final-controlled-comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(HERE/'final-controlled-comparison.json'),mesh_equal=mesh_equal,source_step_all_equal=all(equality.values()),source_law_equal=True,provider_identities_equal=True,source_convergence='NOT_ESTABLISHED'),indent=2))


if __name__=='__main__':main()
