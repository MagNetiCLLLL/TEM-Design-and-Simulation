"""Bounded refinement and evidence export; no source or wave calculation."""
from __future__ import annotations
from hashlib import sha256
import importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transfer_v1',HERE/'design_transfer.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def objective(model,x,anchor,anchor_weight=1e-6):
    r=model.residual(x,anchor,anchor_weight)
    maps=model.evaluate(x)
    endpoint=[maps[0]['initial_bz_t']-model.targets[0]['initial_bz_t']]
    endpoint.extend(a['final_bz_t']-b['final_bz_t'] for a,b in zip(maps,model.targets))
    return np.r_[r,.5*np.array(endpoint)]

def main():
    started=time.perf_counter()
    prior=json.loads((HERE/'candidate-v2.json').read_text())
    reference_path=old.BASE/'round-lens-field-match/captured-field-coefficients.json'
    raw=json.loads(reference_path.read_text()); cells=prior['cells'];scales=np.array(prior['scales'])
    terms=np.array([term for row in raw['gaussian'].values() for term in row['terms_t_m']])
    best=np.array(prior['best_values']); fine=old.TransferModel(cells,scales,terms,.05)
    fit=least_squares(lambda x:objective(fine,x,best),best,bounds=(-100.,100.),x_scale='jac',
                      max_nfev=500,ftol=1e-11,xtol=1e-11,gtol=1e-11)
    print(json.dumps(dict(stage='Bounded 0.05 mm refinement',nfev=fit.nfev,score=float(np.linalg.norm(objective(fine,fit.x,best,0.))),elapsed_s=time.perf_counter()-started)),flush=True)
    refined=old.TransferModel(cells,scales,terms,.025)
    actual=fine.evaluate(fit.x); high=refined.evaluate(fit.x)
    recipes={};identities={}
    for owner,(start,stop) in prior['owners'].items():
        windings={}
        for i,cell in enumerate(cells[start:stop],start):
            part=cell['parent_source_id'].rsplit(':',2)[1]
            windings.setdefault(part,[]).append(dict(interval_index=cell['parent_interval_index'],turn_count=cell['turn_count'],
                radial_fraction=cell['radial_fraction'],axial_fraction=cell['axial_fraction'],
                current_ratio=float(fit.x[i]*scales[i]/cell['original_actual_master_a'])))
        recipes[owner]=dict(schema='ideal-coil-current-segments-v1',windings=windings)
        md=raw['ideal_current_coils'][owner]['metadata']
        identities[owner]=dict(original_current_definition=md['provenance']['winding_current_definition'],
            original_winding_records=[row['metadata'] for row in md['coils']])
    matrix_metrics=[]
    for i,(a,b,target) in enumerate(zip(actual,high,refined.targets)):
        length=old.PLANES[i+1]-old.PLANES[i]
        norm=np.array(((1.,1/length),(length,1.)))
        radial_relative=(b['radial']-target['radial'])*norm/np.maximum(1.,np.abs(target['radial']*norm))
        matrix_metrics.append(dict(interval_m=[old.PLANES[i],old.PLANES[i+1]],
            candidate_005=old.serializable(a),candidate_0025=old.serializable(b),target_0025=old.serializable(target),
            dimensionless_radial_residual=radial_relative.tolist(),
            radial_residual_norm=float(np.linalg.norm(radial_relative)),
            candidate_radial_mesh_max_difference=float(np.max(np.abs((a['radial']-b['radial'])*norm))),
            candidate_mechanical_mesh_max_difference=float(np.max(np.abs(a['mechanical']-b['mechanical'])))))
    # The final four residual rows are explicit endpoint-field comparisons;
    # the preceding 30 anchor rows are removed before conditioning analysis.
    j=np.r_[fit.jac[:-len(cells)-4],fit.jac[-4:]]
    singular=np.linalg.svd(j,compute_uv=False)
    z=np.linspace(-.5,3.5,16001);currents=fit.x*scales
    b12=old.tools.axis_basis(z,cells,12)@currents
    b24=old.tools.axis_basis(z,cells,24)@currents
    target=old.tools.gaussian(z,terms)
    profile_l2=float(np.linalg.norm(b24-target)/np.linalg.norm(target))
    quad_l2=float(np.linalg.norm(b24-b12)/np.linalg.norm(b24))
    profile={'z_m':z,'candidate_bz_t':b24,'target_gaussian_bz_t':target}
    np.savez_compressed(HERE/'final-profile.npz',**profile)
    gouy=[]
    for part in refined.parts:
        k=old.FACTOR*(part['basis']@fit.x)
        capture=np.arange(1,len(part['z']),dtype=np.int64)
        matrices=old._rk4_axisymmetric_larmor_matrices(np.ascontiguousarray(k),part['z'],capture)
        phase=np.unwrap(np.r_[0.,np.angle(matrices[:,0,0]+1j*matrices[:,0,1]/part['length'])])
        tk=old.FACTOR*part['target']
        tm=old._rk4_axisymmetric_larmor_matrices(np.ascontiguousarray(tk),part['z'],capture)
        tp=np.unwrap(np.r_[0.,np.angle(tm[:,0,0]+1j*tm[:,0,1]/part['length'])])
        gouy.append(dict(candidate_unwrapped_fundamental_phase_rad=float(phase[-1]),target_unwrapped_fundamental_phase_rad=float(tp[-1]),
            candidate_maximum_unwrap_step_rad=float(np.max(np.abs(np.diff(phase)))),target_maximum_unwrap_step_rad=float(np.max(np.abs(np.diff(tp))))))
    out=dict(schema='fixed-envelope-source-free-current-fit-conclusion-v1',status='FAILED_TO_MATCH_REFERENCE_DO_NOT_INSTALL',
        conclusion='The bounded current-only search did not recover the archived Gaussian linear transfer. This is a failed candidate, not proof that every possible fixed-envelope current distribution is impossible.',
        scope='Source-free constant-300keV axisymmetric linear operators only. No particle source, physical acceleration, finite-ray transport, nonlinear corrector, D95, focus, aperture or coherent calculation.',
        reference_sha256=sha256(reference_path.read_bytes()).hexdigest(),geometry_identity_sha256=digest(identities),original_geometry_identities=identities,
        search=dict(coarse_seed_count=len(prior['attempts']),coarse_max_nfev_each=500,fine_max_nfev=500,
            original_interval_subdivision='4 axial integer-turn cells for C1 lens; 2 axial cells per original interval for every other round lens',
            control_count=len(cells),normalized_current_bounds=[-100.,100.],current_scale_a=scales.tolist(),
            resistance_or_power_model=False,endpoint_field_objective_weight_per_tesla=.5),
        candidate_current_a=currents.tolist(),maximum_absolute_current_a=float(np.max(np.abs(currents))),cells=cells,recipes=recipes,
        final_optimizer=dict(success=bool(fit.success),message=fit.message,nfev=fit.nfev,optimality=float(fit.optimality)),
        fine_objective_norm=float(np.linalg.norm(objective(fine,fit.x,best,0.))),refined_objective_norm=float(np.linalg.norm(objective(refined,fit.x,best,0.))),
        residual_jacobian_singular_values=singular.tolist(),residual_jacobian_condition=float(singular[0]/singular[-1]),
        matrix_metrics=matrix_metrics,unwrapped_fundamental_phase=gouy,
        profile_diagnostics=dict(finite_axis_domain_m=[-.5,3.5],point_count=len(z),relative_gaussian_l2=profile_l2,q12_to_q24_relative_l2=quad_l2,
            q12_to_q24_max_absolute_t=float(np.max(np.abs(b24-b12))),scope='Finite-domain profile diagnostics only; full infinite-tail field-fit evidence is separate in round-study.'),
        decision='No default state/profile/source cache is modified. No expensive finite-source or scanning run is justified by this failed candidate.',
        limits=['Current polarity reversal changes signed B but not the positive B-squared focusing term.',
                'A converged matrix mesh does not imply agreement with the reference matrix or specimen focus.',
                'All original winding ID/OD, centers, axial spans, poses, integer total turns and conductor areas are preserved.',
                'The optimizer objective weights and current bounds are study choices, not modifications to project numerical tolerances.'],elapsed_s=time.perf_counter()-started)
    old.write_json(HERE/'final-conclusion.json',out)
    print(json.dumps(dict(stage='DONE',status=out['status'],refined_objective_norm=out['refined_objective_norm'],maximum_current_a=out['maximum_absolute_current_a'],elapsed_s=out['elapsed_s'])),flush=True)

if __name__=='__main__':main()
