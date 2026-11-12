"""Analyze completed original-source observations without new particle tracing."""
from pathlib import Path
from hashlib import sha256
import json
import math
import tomllib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def digest(path):return sha256(path.read_bytes()).hexdigest()


def phase_summary(x,y,tx,ty,w):
    normalized=w/w.sum()
    x=x-np.average(x,weights=w);y=y-np.average(y,weights=w)
    tx=tx-np.average(tx,weights=w);ty=ty-np.average(ty,weights=w)
    return dict(rms_radius_m=float(np.sqrt(np.dot(normalized,x*x+y*y))),
        maximum_axis_centered_radius_m=float(np.max(np.hypot(x,y))),
        rms_mechanical_slope_rad=float(np.sqrt(np.dot(normalized,tx*tx+ty*ty))),
        radial_curvature_per_m=float(np.dot(normalized,x*tx+y*ty)/np.dot(normalized,x*x+y*y)))


def descriptive_fit(z,row,w,ranking):
    """Fit existing positions only; coefficients are not a calibrated lens Cs."""
    u=(z['tx_rad'][row]-np.average(z['tx_rad'][row],weights=w))+1j*(z['ty_rad'][row]-np.average(z['ty_rad'][row],weights=w))
    v=(z['x_m'][row]-np.average(z['x_m'][row],weights=w))+1j*(z['y_m'][row]-np.average(z['y_m'][row],weights=w))
    energy=z['kinetic_energy_ev'][row]-np.average(z['kinetic_energy_ev'][row],weights=w)
    models={
        'linear_angle':([u],['angle']),
        'linear_and_cubic_angle':([u,u*np.abs(u)**2],['angle','angle_times_squared_magnitude']),
        'linear_and_energy_angle':([u,u*energy],['angle','angle_times_energy_deviation_ev']),
        'linear_cubic_and_energy_angle':([u,u*np.abs(u)**2,u*energy],['angle','angle_times_squared_magnitude','angle_times_energy_deviation_ev']),
    }
    result={}
    for name,(features,labels) in models.items():
        matrix=np.column_stack(features)
        scale=np.sqrt(np.average(np.abs(matrix)**2,axis=0,weights=w));scaled=matrix/scale
        coefficient=np.linalg.lstsq(scaled*np.sqrt(w)[:,None],v*np.sqrt(w),rcond=None)[0]
        error=v-scaled@coefficient
        result[name]=dict(fraction_position_variance_explained=float(1-np.dot(w,np.abs(error)**2)/np.dot(w,np.abs(v)**2)),
            coefficients={label:dict(real=float(value.real),imaginary=float(value.imag)) for label,value in zip(labels,coefficient/scale)},
            maximum_radius_particle_fit_residual_nm=float(abs(error[ranking[0]])*1e9))
    return result


def analyze(directory):
    report_path=directory/'report.json'
    if not report_path.is_file():return None,None
    report=json.loads(report_path.read_text())
    if report.get('status')!='COMPLETED_CLASSICAL_TWO_CONDENSER_OBSERVATION':return None,None
    source=directory/'observations.npz'
    checksum=digest(source)
    if checksum!=report['observations_sha256']:raise ValueError('Completed observation checksum mismatch: '+str(directory))
    with np.load(source,allow_pickle=False) as stored:arrays=dict(stored)
    z=arrays;w=z['weights'];ids=z['source_id'];positive=w>0
    row=int(np.flatnonzero(z['z_mm']==report['surface_z_mm'])[0])
    keep=positive&z['alive'][row]
    if not np.all(keep):raise ValueError('This audit requires explicit handling for losses before comparing all-source tails')
    centered_x=z['x_m'][row]-np.average(z['x_m'][row],weights=w)
    centered_y=z['y_m'][row]-np.average(z['y_m'][row],weights=w)
    radii=np.hypot(centered_x,centered_y)
    ranking=np.argsort(radii)[::-1]
    total_moment=float(np.dot(w,radii*radii))
    mean_energy=float(np.average(z['kinetic_energy_ev'][row],weights=w))
    aperture_row=int(np.flatnonzero(z['z_mm']==765.)[0])
    pupil=np.hypot(z['x_m'][aperture_row],z['y_m'][aperture_row])
    phase=phase_summary(z['gun_exit_x_m'],z['gun_exit_y_m'],z['gun_exit_tx_rad'],z['gun_exit_ty_rad'],w)
    records=[]
    for index in ranking[:8]:
        emitted=z['emission_direction'][index]
        history=[]
        for target in (500.,650.,765.,948.5,report['surface_z_mm']):
            match=np.flatnonzero(z['z_mm']==target)
            if len(match):
                j=int(match[0])
                history.append(dict(z_mm=float(target),axis_radius_m=float(np.hypot(z['x_m'][j,index],z['y_m'][j,index])),
                    mechanical_angle_from_axis_mrad=float(1000*np.arctan(np.hypot(z['tx_rad'][j,index],z['ty_rad'][j,index]))),alive=bool(z['alive'][j,index])))
        records.append(dict(source_id=int(ids[index]),original_weight=float(w[index]),
            current_a=float(report['execution']['reference_current_a']*w[index]),
            surface_centroid_radius_nm=float(radii[index]*1e9),
            surface_optical_axis_radius_nm=float(np.hypot(z['x_m'][row,index],z['y_m'][row,index])*1e9),
            fraction_surface_radial_second_moment=float(w[index]*radii[index]**2/total_moment),
            surface_kinetic_energy_ev=float(z['kinetic_energy_ev'][row,index]),
            surface_energy_deviation_from_current_mean_ev=float(z['kinetic_energy_ev'][row,index]-mean_energy),
            emission_position_nm=(z['emission_position_m'][index]*1e9).tolist(),
            emission_mechanical_angle_mrad=float(1000*np.arctan2(np.hypot(emitted[0],emitted[1]),emitted[2])),
            gun_exit_axis_radius_um=float(1e6*np.hypot(z['gun_exit_x_m'][index],z['gun_exit_y_m'][index])),
            c2_aperture_axis_radius_um=float(pupil[index]*1e6),
            observed_checkpoints=history))
    result=dict(directory=str(directory.relative_to(ROOT)).replace('\\','/'),
        status='COMPLETED_ARRAY_AUDIT_NO_NEW_TRACING',
        evidence_sha256={p.name:digest(p) for p in (report_path,source,directory/'input.toml',directory/'inputs.json')},
        original_source=dict(count=len(w),positive_count=int(positive.sum()),unique_id_count=int(np.unique(ids[positive]).size),
            zero_current_count=int((w==0).sum()),weight_sum=float(w.sum()),weight_minimum=float(w.min()),weight_maximum=float(w.max()),
            exact_equal_weight_population=bool(np.array_equal(w,np.full(len(w),1/len(w)))),
            emission_ids_match_original_source_ids=bool(np.array_equal(ids,z['emission_ray_id'])),
            gun_exit_surviving_count=int(np.count_nonzero(z['gun_exit_alive']&positive)),
            specimen_surviving_count=int(keep.sum()),effective_count=float(w.sum()**2/np.dot(w,w)),
            source_current_a=report['execution']['reference_current_a'],sample_current_a=report['sample_current_a'],
            reported_transmission=report['emitted_current_transmission'],
            per_stage_physical_losses=report['physical_losses'],
            no_unlabeled_stop=bool(np.all(z['terminal_alive']) and np.all(z['blocked_key']=='')),
            stopped_particle_resurrections=int(np.count_nonzero(~z['alive'][:-1]&z['alive'][1:]&positive))),
        gun_exit_phase=phase,
        source_emission=dict(spatial_rms_axes_nm=(np.sqrt(np.average(z['emission_position_m'][:,:2]**2,axis=0,weights=w))*1e9).tolist(),
            direction_transverse_rms_axes_mrad=(np.sqrt(np.average(z['emission_direction'][:,:2]**2,axis=0,weights=w))*1e3).tolist(),
            maximum_polar_angle_mrad=float(1000*np.max(np.arctan2(np.linalg.norm(z['emission_direction'][:,:2],axis=1),z['emission_direction'][:,2])))),
        c2_aperture=dict(actual_plane_mm=765.,physical_radius_um=50.,maximum_actual_axis_radius_um=float(pupil.max()*1e6),
            maximum_radial_fill_fraction=float(pupil.max()/.00005),radial_clearance_um=float((.00005-pupil.max())*1e6),
            native_geometry_audit=report['execution']['field_audit']['c2_aperture'],
            interpretation='Complete source rays remain well within the inserted 50 um-radius aperture at its exact observation; production finite-boundary interception reports no stopped source particles.'),
        surface=dict(controls=report['controls'],z_mm=report['surface_z_mm'],d95_nm=report['d95_nm'],rms_radius_nm=report['rms_radius_nm'],
            alpha95_mrad=report['alpha95_mrad'],angular_rms_mrad=report['angular_rms_mrad'],
            native_d95_waist=report['native_focus'].get('d95_waist'),native_rms_waist=report['native_focus'].get('rms_waist_diagnostic'),
            native_focus_gate=report['native_focus'].get('surface_gate'),
            maximum_centroid_radius_nm=float(radii[ranking[0]]*1e9),
            maximum_radius_over_d95_radius=float(2e9*radii[ranking[0]]/report['d95_nm']),
            maximum_radius_over_second_largest=float(radii[ranking[0]]/radii[ranking[1]]),
            outermost_five_percent_packet_fraction_radial_second_moment=float(sum(w[i]*radii[i]**2 for i in ranking[:math.ceil(len(w)*.05)])/total_moment),
            d95_definition='Diameter containing 95% of actual specimen-plane current about its current-weighted centroid; not an angle or an emitted-current cutoff.',
            rms_definition='Square root of the current-weighted mean squared centroid radius using all surviving positive-current particles.'),
        largest_radius_particles=records,
        descriptive_regression=dict(scope='Read-only regression of existing observed complex transverse positions on mechanical angle; no ray removal, retuning, current renormalization or new transport.',
            inference='The strong cubic-angle association and additional energy dependence support a reproducible nonlinear/dispersion tail. This is not an isolated-lens aberration calibration or a causal perturbation experiment.',
            fits=descriptive_fit(z,row,w,ranking)))
    return result,arrays


def main():
    studies=[];sources={}
    for name in ('initial-n1024-s010','fine-n2048-s010'):
        for directory in sorted((HERE/name).glob('trial-*')):
            result,arrays=analyze(directory)
            if result is None:continue
            studies.append(result)
            sources[directory.relative_to(HERE).as_posix()]=arrays
    comparisons=[]
    same_fields=('weights','source_id','emission_ray_id','emission_position_m','emission_direction','emission_normal',
                 'gun_exit_x_m','gun_exit_y_m','gun_exit_tx_rad','gun_exit_ty_rad','gun_exit_flight_time_s','gun_exit_alive')
    for name in ('initial-n1024-s010','fine-n2048-s010'):
        rows=[(key,value) for key,value in sources.items() if key.startswith(name+'/')]
        for key,value in rows[1:]:
            ref_key,ref=rows[0]
            comparisons.append(dict(reference=ref_key,comparison=key,
                exact_array_equality={field:bool(np.array_equal(ref[field],value[field],equal_nan=True)) for field in same_fields}))
    current=tomllib.loads((HERE/'initial-n1024-s010/trial-01/input.toml').read_text())
    old_path=ROOT/'outputs/microprobe-20261111/extractor/new-n2048/input.toml'
    old=tomllib.loads(old_path.read_text())
    historical=ROOT/'outputs/microprobe-20261111/extractor/new-n2048/source.npz'
    launch_path=ROOT/'outputs/microprobe-20261111/extractor/new-n2048/launch.npz'
    with np.load(historical,allow_pickle=False) as z:
        old_phase=phase_summary(z['gun_exit_x_m'],z['gun_exit_y_m'],z['gun_exit_tx_rad'],z['gun_exit_ty_rad'],z['gun_exit_weight'])
    with np.load(launch_path,allow_pickle=False) as z:
        old_angles=dict(launch_slope_rms_axes_mrad=[float(np.sqrt(np.average(z[key]**2,weights=z['weight']))*1000) for key in ('tx_rad','ty_rad')],
            maximum_launch_polar_angle_mrad=float(1000*np.max(np.arctan(np.hypot(z['tx_rad'],z['ty_rad'])))))
    voltages={key:dict(current=current['devices'][key],historical=old['devices'][key],exact_match=current['devices'][key]==old['devices'][key])
              for key in ('feg_electrostatic_lens','feg_extractor','feg_accelerator')}
    report=dict(schema='temsim-two-condenser-original-source-audit-v1',scope=__doc__,new_particle_traces_executed=0,
        observations=studies,same_population_optical_trial_comparisons=comparisons,
        source_model_comparison=dict(current_profile='initial-n1024-s010/trial-01/input.toml',
            historical_profile=str(old_path.relative_to(ROOT)).replace('\\','/'),
            historical_file_sha256={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (old_path,historical,launch_path)},
            current_source_model=current['gun_source_model'],historical_source_model=old['gun_source_model'],
            gun_voltage_records=voltages,historical_gun_exit_phase=old_phase,historical_actual_launch=old_angles,
            observed_difference='The historical profile explicitly uses driven_gaussian_schell with a 20 nm transverse coherence length; the current actual default is classical_particles without that Tip coherence boundary. Actual historical launch slopes are about 21 mrad per axis, versus about 0.971 mrad transverse direction per axis now. Gun voltages are identical.',
            interpretation='The old 6.939 um RMS gun-exit source and its predicted aperture filling are not the currently executed 0.379 um RMS source. Historical curvature was used only as a zero-current focusing proposal and must not qualify current source width, current or aberrations.',
            source_code_references=['src/temsim/optics/electron_gun/emitter.py:414','src/temsim/optics/electron_gun/tip_coherence.py:86','src/temsim/optics/electron_gun/tip_coherence.py:277']),
        conclusions=dict(original_source_integrity='All completed observations retain unique original IDs, positive equal weights, original emission IDs and full physical transmission. Same-count optical trials preserve exact emission and gun-exit arrays.',
            current_transmission='100% transmission is supported by actual original-current accounting and a C2 pupil maximum near 13.3 um inside the real 50 um-radius inserted aperture; it was not created by rejecting or renormalizing tails.',
            tails='The focused 1024 population contains a consequential high-angle tail: its largest particle contributes about 10% of the radial second moment. At 2048 particles a second comparably large tail is resolved, arguing against a single corrupt or isolated packet.',
            statistical_distinction='D95 can exclude an outer current tail that still strongly influences RMS radius and RMS-waist position. Both native waist diagnostics must remain separate.',
            limits='These are classical geometrical particle statistics; no coherent diffraction-limited probe size or wave-resolution qualification is claimed.'))
    path=HERE/'source-audit.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)
    print(json.dumps(dict(completed_observations=len(studies),paths=[row['directory'] for row in studies],output=str(path)),indent=2))


if __name__=='__main__':main()
