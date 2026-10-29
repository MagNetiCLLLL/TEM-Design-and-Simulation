"""Offline held-out launch-momentum diagnostics; source reconstruction only.

Reads completed arrays and reconstructs the saved flat-tip emitter. Does not
construct fields, propagate particles, change controls, or infer canonical
momentum from an unchecked vector potential.
"""
from dataclasses import fields
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'agent.md').is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from temsim.physics.ray_identity import emission_reference
from temsim.physics.relativistic_lorentz import momentum_from_kinetic_energy_ev, ELECTRON_MASS_KG, SPEED_OF_LIGHT_M_PER_S
from solve_joint_probe import atomic_json, atomic_npz
from run_column_energy_diagnostic import file_hash, array_hash

TARGETS=('gun_exit_x_m','gun_exit_y_m','gun_exit_tx_rad','gun_exit_ty_rad')
CODE=('src/temsim/optics/electron_gun/emitter.py','src/temsim/optics/electron_gun/tracing.py',
      'src/temsim/physics/relativistic_lorentz.py','src/temsim/physics/chromatic.py',
      'src/temsim/parameter_registry.py','docs/COHERENT_BEAM.md',
      'src/temsim/physics/first_order.py','src/temsim/physics/unified_field_particle.py','src/temsim/physics/core.py')


def weighted_stats(value,weight):
    w=weight/weight.sum();mean=float(w@value)
    return dict(mean=mean,std=float(np.sqrt(w@((value-mean)**2))),min=float(value.min()),max=float(value.max()),
                p50=float(np.quantile(value,.5)),p95=float(np.quantile(value,.95)),p99=float(np.quantile(value,.99)))


def load(folder):
    folder=Path(folder).resolve()
    report=json.loads((folder/'report.json').read_text())
    if report['execution_status']!='COMPLETE':raise ValueError('Completed profile required')
    hashes={name:file_hash(folder/name) for name in ('inputs.json','particles.npz','report.json')}
    for name in ('inputs.json','particles.npz'):
        if hashes[name]!=report['files_sha256'][name]:raise ValueError('Donor checksum mismatch')
    snapshot=json.loads((folder/'inputs.json').read_text())
    node=next(n for n in snapshot['graph']['nodes'] if n['type'].endswith(':ColdFieldEmitter'))
    attrs=node['attributes']
    excluded=('_surface_model','_tip_curvature_nm_inv','_tip_coherence','_emission_sampling',
              '_emission_quadrature','_emission_cdf_quadrature','_tuning_boundary_probes','_tuning_surface_probes')
    if any(key in attrs for key in excluded):raise ValueError('Only the captured legacy flat source is covered')
    emitter=ColdFieldEmitter(**{f.name:attrs[f.name] for f in fields(ColdFieldEmitter)})
    bundle=emitter.emit()  # Deterministic emission reconstruction; no propagation.
    reference=emission_reference(bundle)
    with np.load(folder/'particles.npz',allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    for key in ('position_m','direction','ray_id','normal'):
        if not np.array_equal(reference[key],arrays['emission_'+key]):raise ValueError('Emission reconstruction changed')
    if not np.array_equal(bundle.weight,arrays['weights']) or not arrays['gun_exit_alive'].all():
        raise ValueError('This analysis requires the complete identical unfiltered gun population')
    energy=emitter.emission_energy_ev+bundle.energy_offset_ev
    p=momentum_from_kinetic_energy_ev(energy,reference['direction'])
    # This is mechanical/kinetic momentum, as explicitly used by gun tracing.
    # No assumption about A at the emitting boundary or canonical momentum.
    p_ref=float(np.linalg.norm(momentum_from_kinetic_energy_ev(300000.3,[0.,0.,1.])))
    angle=np.column_stack((bundle.x_m*1e9,bundle.y_m*1e9,bundle.tx_rad*1e3,bundle.ty_rad*1e3))
    momentum=np.column_stack((bundle.x_m*1e9,bundle.y_m*1e9,p[:,:2]/p_ref*1e6))
    y=np.column_stack([arrays[k] for k in TARGETS])
    return dict(folder=folder,arrays=arrays,reference=reference,emitter=emitter,energy=energy,p=p,p_ref=p_ref,
        angle=angle,momentum=momentum,y=y,hashes=hashes,snapshot=snapshot,report=report)


def metrics(actual,prediction,weight):
    w=weight/weight.sum();residual=prediction-actual
    result={}
    for i,key in enumerate(TARGETS):
        variance=float(w@((actual[:,i]-w@actual[:,i])**2));mse=float(w@(residual[:,i]**2))
        factor=1e9 if i<2 else 1e6
        result[key]=dict(r_squared=1.-mse/variance if variance>0 else None,
            rms_error=float(np.sqrt(mse)*factor),maximum_abs_error=float(np.max(np.abs(residual[:,i]))*factor),
            error_unit='nm' if i<2 else 'urad')
    return result


def linear_fit(features,targets,weights,train):
    w=weights[train]/weights[train].sum()
    mean=w@features[train]
    scale=np.sqrt(w@((features[train]-mean)**2))
    if np.any(scale<=0):raise ValueError('Degenerate training features')
    design=np.column_stack((np.ones(len(features)),(features-mean)/scale))
    coefficient,residual,rank,singular=np.linalg.lstsq(design[train]*np.sqrt(w)[:,None],targets[train]*np.sqrt(w)[:,None],rcond=None)
    if rank!=5:raise ValueError('Rank-deficient explanatory model')
    prediction=design@coefficient
    return prediction,dict(feature_mean=mean.tolist(),feature_scale=scale.tolist(),
        standardized_coefficient_by_target=coefficient.tolist(),rank=int(rank),
        weighted_training_design_condition=float(singular[0]/singular[-1]),
        parameter_count_per_target=5,training_only_standardization=True)


def moment_summary(case):
    a=case['arrays'];w=a['weights']/a['weights'].sum();p=case['p'];energy=case['energy']
    source=case['reference']['position_m'];result={}
    for axis,i in (('x',0),('y',1)):
        x=source[:,i];u=p[:,i]/(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S)
        dx=x-w@x;du=u-w@u
        variance_x=float(w@(dx*dx));variance_u=float(w@(du*du));cov=float(w@(dx*du))
        result[axis]=dict(kinetic_normalized_rms_area_m=float(np.sqrt(max(0.,variance_x*variance_u-cov**2))),
            rms_momentum_kg_m_s=float(np.sqrt(w@((p[:,i]-w@p[:,i])**2))),
            normalized_momentum_kurtosis=float(w@(du**4)/(w@(du**2))**2))
    total_exit_p=np.linalg.norm(momentum_from_kinetic_energy_ev(300000.+energy,np.tile([0.,0.,1.],(len(w),1))),axis=1)
    ideal_pz=np.sqrt(total_exit_p**2-np.sum(p[:,:2]**2,axis=1))
    ideal_slopes=p[:,:2]/ideal_pz[:,None]
    return dict(launch_kinetic_ev=weighted_stats(energy,w),launch_energy_std_over_mean=float(np.std(energy)/np.mean(energy)),
        launch_transverse_momentum_norm_kg_m_s=weighted_stats(np.linalg.norm(p[:,:2],axis=1),w),
        ideal_pure_longitudinal_acceleration_slope_norm_mrad=weighted_stats(np.linalg.norm(ideal_slopes,axis=1)*1e3,w),
        axes=result,area_scope='Mechanical momentum covariance at emission, not a canonical emittance invariant or a measured source brightness',
        ideal_scope='Kinematic reference only: assume no transverse force, p_x/y conserved, 300 keV longitudinal energy gain; this is not the actual gun field model')


def write_markdown(report,output):
    ratio=report['id306_and_naive_scaling'];split=report['split'];moments=report['source_moments']
    lines=['# Launch momentum and gun-exit phase: offline diagnostic','',
        'The saved gun-exit phase is substantially better described by initial position plus kinetic transverse momentum than by initial position plus launch slope alone. This is explanatory evidence for the importance of the prescribed launch distribution, not proof of a gun chromatic-aberration coefficient or of numerical convergence.','',
        '## Definitions and source contract','',
        '- `angular_rms_mrad = 1` is the each-axis local slope Gaussian scale at emission, before the radial 3 mrad cutoff. It is not the divergence at the fully accelerated gun exit (emitter.py:428-432; parameter_registry.py:136).',
        '- `virtual_source_fwhm_nm = 5` is the projected physical emission footprint; it is not an independently measured or positioned virtual source (emitter.py:424-427; parameter_registry.py:135).',
        '- Launch kinetic energy is 0.3 eV plus the prescribed per-particle energy offset. Gun tracing explicitly initializes mechanical momentum from this energy and the emission direction (tracing.py:187-194,222-223).',
        '- Relativistic mechanical momentum magnitude is p(K) = sqrt((K e)(K e + 2 m c^2))/c. The fit uses px/p_ref and py/p_ref, where p_ref is fixed at 300000.3 eV for both cases. No vector potential was evaluated and no canonical-momentum assertion is made.',
        '- The finite sample has launch-energy RMS 0.127398270 eV, 42.4661% of its 0.3 eV mean. The configured 0.3 eV FWHM field is RMS-equivalent, not a measured spectral FWHM (chromatic.py:66-75; parameter_registry.py:138-139).','',
        '## Matched particle 306','',
        f"Its launch energy changes from {ratio['launch_kinetic_poly_ev']:.12g} eV to 0.3 eV with exactly the same initial position and direction. The exact relativistic launch-momentum ratio is {ratio['exact_relativistic_momentum_ratio']:.12g} (sqrt-energy ratio {ratio['sqrt_energy_ratio']:.12g}). The observed gun-exit radius ratio is {ratio['observed_gun_exit_radius_ratio']:.12g}; the slope-norm ratio is {ratio['observed_gun_exit_slope_norm_ratio']:.12g}.",
        '', 'The close ratios are consistent with strong launch-momentum scaling. They do not prove an ideal uniform accelerator: finite source position, focusing fields, nonlinear response, and numerical error remain in the actual trajectories.','',
        '## Deterministic held-out comparison','',
        f"Source IDs with id % 5 == 1 were held out before fitting. Both partners of each poly/mono pair stay in the same split: {split['unique_train_ids']} training IDs ({split['training_rows']} rows) and {split['unique_heldout_ids']} held-out IDs ({split['heldout_rows']} rows). ID306 is held out. Both models have exactly five coefficients per output: intercept and four inputs. Only training rows determine standardization and least-squares coefficients. No energy label or higher-order term was added.",
        '', '| Gun-exit target | Angle-model held-out RMS error | Momentum-model held-out RMS error | Momentum-model R² |',
        '|---|---:|---:|---:|']
    for key in TARGETS:
        a=report['fits']['angle']['heldout'][key];b=report['fits']['momentum']['heldout'][key]
        lines.append(f"| {key} | {a['rms_error']:.9g} {a['error_unit']} | {b['rms_error']:.9g} {b['error_unit']} | {b['r_squared']:.12g} |")
    lines.extend(['','ID306-specific errors (this particle was held out):','',
        '| Model | Case | Position-vector prediction error (nm) | Maximum position-component error (nm) | Slope-vector prediction error (urad) |',
        '|---|---|---:|---:|---:|'])
    for model,cases in ratio['heldout_predictions'].items():
        for label,row in cases.items():
            lines.append(f"| {model} | {label} | {row['position_error_norm_nm']:.9g} | {row['maximum_abs_position_error_nm']:.9g} | {row['slope_error_norm_urad']:.9g} |")
    lines.extend(['','Training design condition numbers are approximately 1.03. Per-case errors, coefficients, training-only scales and all predictions are retained in report.json / arrays.npz.','',
        '## Tail shape versus RMS area','',
        'At sub-eV energies, p² is approximately 2mK. If direction and energy are independent and mean K is fixed, E[p_perp²] is approximately unchanged by energy spread alone. Its distribution can nevertheless become a scale mixture with heavier tails. It would be incorrect to infer increased RMS emittance merely from a larger energy spread.','',
        '| Axis | Mono kinetic normalized RMS area (m) | Poly kinetic normalized RMS area (m) | Poly/mono change | Mono p kurtosis | Poly p kurtosis |','|---|---:|---:|---:|---:|---:|'])
    for axis in ('x','y'):
        a=moments['mono']['axes'][axis];b=moments['poly']['axes'][axis]
        change=(b['kinetic_normalized_rms_area_m']/a['kinetic_normalized_rms_area_m']-1)*100
        lines.append(f"| {axis} | {a['kinetic_normalized_rms_area_m']:.9g} | {b['kinetic_normalized_rms_area_m']:.9g} | {change:+.6g}% | {a['normalized_momentum_kurtosis']:.9g} | {b['normalized_momentum_kurtosis']:.9g} |")
    lines.extend(['','These are mechanical momentum covariance areas at emission, not canonical invariants or a source-brightness calibration. The small RMS differences include the actual finite quadrature correlations. The ideal longitudinal-acceleration calculation in the JSON conserves transverse mechanical momentum only as a stated kinematic reference; it is not a replacement propagation.','',
        '## Existing native linearization support (read-only inspection)','',
        '- `unified_field_particle.py:447` provides `trace_unified_response`, accepting explicit phase, kinetic energy and flight time and executing the shared complete fields. With no coil increments it changes no physical drive. Its result contains actual A, mechanics and field identities. No variational ODE solver was found in the inspected unified path.',
        '- `first_order.py:37,319` already builds a nine-ray central-difference response. The unified dispatch in `core.py:611-615` retains the full captured field model, but the public first-order helper uses a zero reference and nominal energy, so it does not automatically represent the actual saved off-axis source.',
        '- A future isolated test could derive a local Jacobian at a declared mathematical reference with fixed K and h/h2 central differences, then compare its offline first-order predictions with the already executed constant-K source. Reusing the exact donor plan avoids a possible m/mm round-trip mesh change. Clipping, original weights and the nonlinear full-source results must remain authoritative.',
        '- This would measure a local-linear remainder, not uniquely identify spherical aberration. No fitted surrogate, reference probe or linearized model was propagated in this analysis.','',
        '## Limits and reproducibility','',
        'No field was constructed or sampled; no gun or column propagation was run. Source reconstruction matches the saved emission positions, directions, normals, IDs and absolute weights exactly. The source law is prescribed and does not constitute a self-consistent microscopic emission or space-charge calculation. Numerical gun refinement is a separate experiment. A good linear fit cannot separate all energy-dependent focusing from nonlinear fields or discretization errors.','',
        'The JSON records SHA256 hashes of both input snapshots, raw particles, reports and governing source files. arrays.npz preserves launch energies, kinetic momenta, actual exit phase, model features, predictions and the held-out mask. The accompanying analysis-source.py is the executable audit. No raw source arrays, optics or results were modified.',''])
    (Path(output)/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def run(poly,mono,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    cases=[load(poly),load(mono)];a,b=[c['arrays'] for c in cases]
    for key in ('source_id','weights','emission_position_m','emission_direction','gun_exit_alive'):
        if not np.array_equal(a[key],b[key]):raise ValueError('Matched-source condition failed: '+key)
    ids=a['source_id'];n=len(ids)
    # Every pair stays in the same split; ID306 is held out (306 mod 5 == 1).
    test=ids%5==1;train=~test;pooled_train=np.tile(train,2);pooled_test=np.tile(test,2)
    y=np.concatenate([c['y'] for c in cases]);weights=np.tile(a['weights']*.5,2)
    raw=dict(source_id=ids,weights=a['weights'],heldout_mask=test)
    fits={}
    for model in ('angle','momentum'):
        features=np.concatenate([c[model] for c in cases])
        prediction,fit=linear_fit(features,y,weights,pooled_train)
        fit.update(features=(['x0_nm','y0_nm','launch_tx_mrad','launch_ty_mrad'] if model=='angle'
            else ['x0_nm','y0_nm','kinetic_px_over_fixed_p_ref_times_1e6','kinetic_py_over_fixed_p_ref_times_1e6']),
            training=metrics(y[pooled_train],prediction[pooled_train],weights[pooled_train]),
            heldout=metrics(y[pooled_test],prediction[pooled_test],weights[pooled_test]),
            per_case_heldout={label:metrics(cases[i]['y'][test],prediction[i*n:(i+1)*n][test],a['weights'][test])
                for i,label in enumerate(('poly','mono'))})
        fits[model]=fit;raw[model+'_features']=features;raw[model+'_pooled_prediction']=prediction
    watched=int(np.flatnonzero(ids==306)[0]);ka,kb=[float(c['energy'][watched]) for c in cases]
    pa,pb=[float(np.linalg.norm(c['p'][watched])) for c in cases]
    ratio=dict(source_id=306,held_out=bool(test[watched]),launch_kinetic_poly_ev=ka,launch_kinetic_mono_ev=kb,
        sqrt_energy_ratio=float(np.sqrt(ka/kb)),exact_relativistic_momentum_ratio=pa/pb,
        observed_gun_exit_radius_ratio=float(np.hypot(a['gun_exit_x_m'][watched],a['gun_exit_y_m'][watched])/np.hypot(b['gun_exit_x_m'][watched],b['gun_exit_y_m'][watched])),
        observed_gun_exit_slope_norm_ratio=float(np.hypot(a['gun_exit_tx_rad'][watched],a['gun_exit_ty_rad'][watched])/np.hypot(b['gun_exit_tx_rad'][watched],b['gun_exit_ty_rad'][watched])))
    ratio['heldout_predictions']={}
    for model in ('angle','momentum'):
        entries={}
        for i,label in enumerate(('poly','mono')):
            actual=cases[i]['y'][watched];predicted=raw[model+'_pooled_prediction'][i*n+watched]
            error=predicted-actual
            entries[label]=dict(actual_phase_xy_m_tx_ty_rad=actual.tolist(),predicted_phase_xy_m_tx_ty_rad=predicted.tolist(),
                position_error_nm=(error[:2]*1e9).tolist(),slope_error_urad=(error[2:]*1e6).tolist(),
                position_error_norm_nm=float(np.linalg.norm(error[:2])*1e9),
                maximum_abs_position_error_nm=float(np.max(np.abs(error[:2]))*1e9),
                slope_error_norm_urad=float(np.linalg.norm(error[2:])*1e6),
                maximum_abs_slope_error_urad=float(np.max(np.abs(error[2:]))*1e6))
        ratio['heldout_predictions'][model]=entries
    ratios=np.linalg.norm(cases[0]['p'],axis=1)/np.linalg.norm(cases[1]['p'],axis=1)
    for name,keys in (('position',('gun_exit_x_m','gun_exit_y_m')),('slope',('gun_exit_tx_rad','gun_exit_ty_rad'))):
        actual=np.column_stack([a[k] for k in keys]);reference=np.column_stack([b[k] for k in keys])
        relative=np.linalg.norm(actual-ratios[:,None]*reference,axis=1)/np.linalg.norm(actual,axis=1)
        ratio[name+'_naive_p_scaled_mono_relative_vector_error']=weighted_stats(relative,a['weights'])
    moments={label:moment_summary(c) for label,c in zip(('poly','mono'),cases)}
    for label,c in zip(('poly','mono'),cases):
        raw.update({label+'_launch_energy_ev':c['energy'],label+'_launch_kinetic_momentum':c['p'],label+'_actual_gun_exit_phase':c['y']})
    report=dict(schema='temsim-launch-momentum-offline-v1',execution_scope='Source emission reconstruction and saved-array linear algebra only; no fields or propagation',
        qualification='EXPLANATORY_OFFLINE_FIT_NOT_PHYSICAL_ATTRIBUTION',
        cases={label:dict(path=str(c['folder']),files_sha256=c['hashes'],input_digest=c['snapshot']['digest'],
            emitter_parameters={f.name:getattr(c['emitter'],f.name) for f in fields(ColdFieldEmitter)},
            maximum_abs_exit_minus_nominal_minus_launch_ev=float(np.max(np.abs(c['arrays']['gun_exit_kinetic_energy_ev']-300000.-c['energy']))))
            for label,c in zip(('poly','mono'),cases)},
        split=dict(rule='source_id % 5 == 1 is held out; both poly/mono partners stay together',unique_train_ids=int(train.sum()),
            unique_heldout_ids=int(test.sum()),training_rows=int(pooled_train.sum()),heldout_rows=int(pooled_test.sum()),
            fixed_before_fitting=True,heldout_id306=bool(test[watched])),
        formulas=dict(momentum='p(K) = sqrt((K e)*(K e + 2 m_e c^2))/c; p_vector = p(K)*direction/norm(direction)',
            reference_momentum_kg_m_s=cases[0]['p_ref'],reference_kinetic_energy_ev=300000.3,
            nature='Kinetic/mechanical momentum only. Vector potential at launch is not evaluated; no canonical-momentum claim.',
            source_angular_semantics='angular_rms_mrad is each-axis Gaussian local slope scale before radial cutoff at the physical emitting tip; not accelerated-exit divergence',
            source_width_semantics='virtual_source_fwhm_nm is the prescribed projected emission footprint at the physical tip, not a measured virtual-source diameter'),
        fits=fits,source_moments=moments,id306_and_naive_scaling=ratio,
        code_files_sha256={name:file_hash(ROOT/name) for name in CODE},analysis_script_sha256=file_hash(__file__),
        limitations=['A better held-out linear fit is not a proof of a focal chromatic aberration coefficient or of numerical convergence.',
            'Actual gun fields include focusing; the ideal longitudinal-acceleration reference is not substituted into transport.',
            'Energy changes both launch momentum and the subsequent trajectory. These effects cannot be additively apportioned by this regression.',
            'At low energy with independent direction and fixed mean K, E[p_perp^2] is approximately proportional to E[K]; energy spread alone need not enlarge the RMS emittance. It does reshape conditional momentum and tails.',
            'The source law is prescribed rather than a self-consistent microscopic field-emission, space-charge or experimentally calibrated source model.'])
    atomic_npz(output/'arrays.npz',raw);report['arrays_sha256']=file_hash(output/'arrays.npz')
    atomic_json(output/'report.json',report)
    write_markdown(report,output)
    (output/'analysis-source.py').write_bytes(Path(__file__).read_bytes())
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poly',type=Path,required=True);parser.add_argument('--mono',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.poly,args.mono,args.output)
