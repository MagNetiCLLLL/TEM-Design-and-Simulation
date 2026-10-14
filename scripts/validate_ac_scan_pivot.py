"""Bounded CUDA scan-pivot check with one physical Tip population.

The current installed defaults are captured. Five frozen physical coil commands
sample the centre and FOV corners; the identical executed pre-coil population is
reused. No source, lens, or installed geometry is fitted. This verifies particle
centroids and finite probes, not a coherent image or an OEM instrument.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar

from validate_probe_relay import save_report, source_identity, assert_cuda, statistics, surface_measurement


def linear_pivot_checks(state, planes):
    from temsim.optics.direct_alignment import canonical_source_basis, canonical_transfers
    sample = float(state.sample.z_mm)
    g = canonical_source_basis(state, sample)[2:, :2]
    checks = []
    for plane in planes:
        transfer = canonical_transfers(state, plane, [sample])[sample]
        d = transfer.k_diff - g @ transfer.j_diff_m_per_rad
        checks.append(dict(z_mm=plane, sample_z_mm=sample,
            d_canonical_norm=float(np.linalg.norm(d, ord=2)),
            b_min_singular_m_per_rad=float(np.linalg.svd(transfer.j_diff_m_per_rad, compute_uv=False)[-1])))
    return checks


def assess(report):
    """Reassess saved measurements without repeating physical propagation."""
    records = report['scans']
    limit = report['limits']
    report['step_radius_difference_nm'] = abs(records[-1]['sample']['rms_radius_nm'] -
                                             report['step_refinement']['sample']['rms_radius_nm'])
    report['checks'] = dict(
        sample_position=all(r['position_error_nm'] <= limit['position_error_nm'] for r in records),
        physical_pivot=all(r['pivot_displacement_nm'] <= limit['pivot_motion_nm'] for r in records),
        canonical_direction=all(r['canonical_angle_rad'] <= limit['canonical_angle_rad'] for r in records),
        diffraction_stationarity=all(r['diffraction_displacement_nm'] <= limit['diffraction_motion_nm'] for r in records),
        centre_focus=records[0]['sample']['rms_radius_nm'] <= limit['centre_rms_nm'],
        scan_focus=all(r['sample']['rms_radius_nm'] <= limit['scanned_rms_nm'] for r in records),
        step_convergence=report['step_centroid_difference_nm'] <= limit['step_centroid_error_nm'],
        radius_convergence=report['step_radius_difference_nm'] <= limit['step_radius_error_nm'],
        hp_conjugacy=report['hp_relay']['b_norm'] < 2e-5,
        same_incident_population=all(r['planes'][1]['rays'] == report['prefix_surviving_rays'] for r in records),
        internal_pivot_transfer=all(r['d_canonical_norm'] < 1e-6 and r['b_min_singular_m_per_rad'] > 1e-6
                                    for r in report['linear_pivot_checks']),
    )
    report['all_passed'] = all(report['checks'].values())


def validate(args, report):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.optics.column import default_state
    from temsim.optics.direct_alignment import canonical_source_basis, canonical_transfers, diffraction_transfer
    from temsim.optics.beam_path_audit import incident_checkpoints
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.scan_geometry import calibrate_scan_system
    from temsim.physics.scan_calibration import ac_angle_target
    from temsim.optics.scan_coil_design import evaluate_scan_coil_design
    from temsim.physics.core import propagate
    from temsim.physics.instrument_magnetic import active_column_events
    from temsim.physics.aperture_clipping import clip_segment
    from temsim.physics.column_wall import clip_column_wall

    s = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(s, catalog.default_selection())
    apply_operating_mode_pair(s, 'nano_probe', 'diffraction')
    s.simulation_mode = 'custom'
    s.acceleration_enabled = True
    s.acceleration_backend = 'Require GPU'
    s.step_mm = .05
    s.electron_gun.emitter.ray_count = 3000
    ac = s.ac_deflector
    ac.wobble_enabled = False
    s.descan_deflector.scan_enabled = False
    s.descan_deflector.wobble_enabled = False
    ac.scan_enabled = True
    ac.calibration_mode = 'automatic'
    ac.scan_pixel_size_nm = args.fov_nm / ac.scan_pixels_x
    calibrate_scan_system(s, observation_stop_z_mm=s.sample.z_mm)
    assert ac_angle_target(s) == 'canonical'
    command = np.asarray(ac.scan_command_matrix_mrad)
    snapshot = capture_instrument_snapshot(s)
    (args.output/'inputs.json').write_text(json.dumps(snapshot.to_dict(), indent=2), encoding='utf-8')
    print('Checking finite-coil canonical pivot', flush=True)
    design = evaluate_scan_coil_design(s, target='canonical', response_model='finite_coil', maximum_step_mm=.1)
    assert design.controllable and design.kick_limit_pass and design.pivot.is_common_pivot
    pivot = float(design.pivot.z_mm)
    report.update(fov_nm=args.fov_nm, input_identity=snapshot.digest, model='custom',
                  limits=dict(position_error_nm=.2, pivot_motion_nm=1., canonical_angle_rad=1e-6,
                              centre_rms_nm=.2, scanned_rms_nm=.2, step_centroid_error_nm=.02,
                              diffraction_motion_nm=1., step_radius_error_nm=.005),
                  design=asdict(design), calibration_target=ac_angle_target(s),
                  scope='3000 physical Tip rays; frozen scan states; finite fields, Cs and HP; no material or wave image',
                  backend_receipts=[])
    sample = float(s.sample.upper_surface_z_mm)
    report['sample_reference'] = dict(calibration_z_mm=float(s.sample.z_mm),
        measured_entrance_z_mm=sample, material='vacuum; no specimen interaction in this optical check')
    report['step_scope'] = '0.05/0.025 mm comparison after AC entrance with one identical executed 0.05 mm upstream prefix'
    report['pivot_definition'] = 'Internal effective scan pivots in the complete installed field; D_canonical(pivot to sample)=0, not a field-free cardinal focal plane of an isolated CM/OL assembly'
    report['linear_pivot_checks'] = linear_pivot_checks(s, [pivot, *design.pivot.other_verified_roots_z_mm])
    prefix = float(s._resolved_assembly.part('ac_deflector').start_z_mm) - .001
    sample_planes = np.array([sample-.0002, sample-.0001, sample])
    historical_waist = 1400.0007  # 10 October strong-C3 observation, not today's measured waist.
    report['historical_source_waist_reference'] = dict(z_mm=historical_waist,
        working_point='2026-10-10 strong-C3', verified_for_current_optics=False)
    grid = np.unique(np.r_[prefix, np.arange(np.ceil(prefix), sample, 1.), historical_waist, pivot, sample_planes])
    ac.scan_enabled = False
    ac.kick_x_mrad = ac.kick_y_mrad = 0.
    print('Executing one Tip population and complete incident column on CUDA', flush=True)
    rows = []
    with backend_receipts(rows):
        gun, base, base_masks = incident_checkpoints(s, grid, step_mm=.05)
    assert_cuda(rows)
    report['backend_receipts'].extend(rows)
    weights = gun.exit_bundle.weight
    report['source_identity'] = source_identity(gun.exit_bundle)
    prefix_mask = base_masks[0].copy()
    report['prefix_surviving_rays'] = int(np.count_nonzero(prefix_mask))
    prefix_xy = [getattr(base, n)[0].copy() for n in ('x_m', 'tx_rad', 'y_m', 'ty_rad')]
    energy = gun.exit_bundle.energy_offset_ev

    # Find the first objective diffraction plane in a bounded post-specimen
    # interval using the same canonical position map used by the UI.
    def diffraction_norm(offset):
        return float(np.linalg.norm(diffraction_transfer(s, sample+offset).j_img, ord=2))
    optimum = minimize_scalar(diffraction_norm, bounds=(.1, 12.), method='bounded',
                              options=dict(xatol=1e-8, maxiter=80))
    diffraction_z = sample+float(optimum.x)
    report['diffraction'] = dict(z_mm=diffraction_z, a_norm=float(optimum.fun),
                                  search_success=bool(optimum.success))
    assert optimum.success and optimum.fun < 1e-5
    hp = canonical_transfers(s, s.hp2_hexapole.z_mm, [s.hp1_hexapole.z_mm])[s.hp1_hexapole.z_mm]
    report['hp_relay'] = dict(b_norm=float(np.linalg.norm(hp.j_diff_m_per_rad, ord=2)),
                              magnification=np.linalg.svd(hp.j_img, compute_uv=False).tolist())
    records, traces = [], []

    def run(factors, step):
        kicks = command @ np.asarray(factors, float)
        ac.kick_x_mrad, ac.kick_y_mrad = map(float, kicks)
        ac.validate()
        events = active_column_events(s)
        # Already stopped electrons remain stopped; identical IDs/weights and
        # energy offsets are used for every scan position.
        rows = []
        save_planes = [float(a.z_mm) for a in s.apertures if a.enabled and prefix <= a.z_mm <= diffraction_z]
        targets = np.unique(np.r_[grid[grid >= prefix], diffraction_z])
        with backend_receipts(rows):
            z, x, tx, y, ty, cp = propagate(s, prefix, diffraction_z, *prefix_xy,
                events=events, energy_offset_ev=energy, save_z_mm=save_planes,
                checkpoint_z_mm=targets, return_checkpoints=True, maximum_step_mm=step,
                include_spherical_aberration=True, include_hexapole=True)
        assert_cuda(rows)
        report['backend_receipts'].extend(rows)
        alive = prefix_mask.copy()
        stops = np.where(alive, np.nan, prefix)
        keys = [''] * len(alive)
        alive, stops, keys = clip_segment(s, z, x, y, alive, stops, keys)
        alive, stops, keys = clip_column_wall(s, z, x, y, alive, stops, keys)
        masks = prefix_mask[None, :] & (np.isnan(stops)[None, :] | (stops[None, :] > cp.z_mm[:, None]))
        surface = surface_measurement(cp, masks, weights, sample_planes)
        ids = [int(np.argmin(abs(cp.z_mm-z))) for z in (pivot, sample, diffraction_z)]
        stats = [statistics(cp, i, weights, masks[i]) for i in ids]
        samples = [dict(z_mm=float(cp.z_mm[i]), x_m=v.mean_x_m, y_m=v.mean_y_m,
                        tx_rad=v.mean_tx_rad, ty_rad=v.mean_ty_rad, rays=v.surviving_rays)
                   for i, v in zip(ids, stats)]
        means = np.array([[statistics(cp, i, weights, masks[i]).mean_x_m,
                           statistics(cp, i, weights, masks[i]).mean_y_m] for i in range(len(cp.z_mm))])
        return dict(factors=list(factors), step_mm=step, planes=samples, sample=surface), (cp.z_mm, means)

    factors = [(0., 0.), (-1., -1.), (1., -1.), (-1., 1.), (1., 1.)]
    for factor in factors:
        print(f'CUDA frozen scan {factor}', flush=True)
        row, path = run(factor, .05)
        records.append(row)
        traces.append(path)
        report['scans'] = records
        save_report(args.output/'report.json', report)
    fine, _ = run(factors[-1], .025)
    report['step_refinement'] = fine
    centre = np.array([[r['x_m'], r['y_m']] for r in records[0]['planes']])
    centre_angle = np.array([records[0]['planes'][1][k] for k in ('tx_rad','ty_rad')])
    g = canonical_source_basis(s, sample)[2:, :2]
    for row in records:
        actual = np.array([[r['x_m'], r['y_m']] for r in row['planes']])-centre
        target = np.asarray(row['factors'])*args.fov_nm*.5e-9
        angle = np.array([row['planes'][1][k] for k in ('tx_rad','ty_rad')])-centre_angle
        row.update(position_error_nm=float(np.linalg.norm(actual[1]-target)*1e9),
                   pivot_displacement_nm=float(np.linalg.norm(actual[0])*1e9),
                   diffraction_displacement_nm=float(np.linalg.norm(actual[2])*1e9),
                   canonical_angle_rad=float(np.linalg.norm(angle-g@actual[1])))
    coarse_point = np.array([[r['x_m'],r['y_m']] for r in records[-1]['planes']])
    fine_point = np.array([[r['x_m'],r['y_m']] for r in fine['planes']])
    difference = float(np.max(np.linalg.norm(coarse_point-fine_point, axis=1))*1e9)
    report['step_centroid_difference_nm'] = difference
    # A finite aberrated probe need not have exactly the first-order stationary
    # centroid; report the executed motion separately from the A=0 criterion.
    assess(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for row, (z, mean) in zip(records, traces):
        label = str(tuple(row['factors']))
        axes[0].plot(z, mean[:, 0]*1e6, label=label)
        select = abs(z-pivot) < 12
        axes[1].plot(z[select], mean[select,0]*1e6)
        axes[2].scatter(row['planes'][1]['x_m']*1e9,row['planes'][1]['y_m']*1e9)
    for ax in axes[:2]:
        ax.axvline(pivot,color='k',ls='--',label='First scan pivot')
        ax.set(xlabel='Z (mm)',ylabel='Centroid X (um)')
        ax.grid(alpha=.2)
    axes[0].axvspan(ac.upper_z_mm-7.5,ac.upper_z_mm+7.5,color='teal',alpha=.15)
    axes[0].axvspan(ac.lower_z_mm-7.5,ac.lower_z_mm+7.5,color='teal',alpha=.15)
    axes[0].axvline(historical_waist,color='purple',ls=':',label='Historical 10 Oct waist reference')
    axes[0].legend(fontsize=7)
    axes[2].set(xlabel='Sample X (nm)',ylabel='Sample Y (nm)',aspect='equal',title='Executed probe centres')
    fig.suptitle('Finite AC fields and unchanged lenses | 3000 Tip rays | CUDA | Descan off')
    fig.savefig(args.output/'scan_pivot.png',dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/ac_scan_pivot_validation'))
    parser.add_argument('--fov-nm',type=float,default=1000.)
    args = parser.parse_args()
    if not np.isfinite(args.fov_nm) or args.fov_nm <= 0:
        parser.error('FOV must be finite and positive')
    args.output.mkdir(parents=True,exist_ok=True)
    report = {}
    started = perf_counter()
    try:
        validate(args,report)
    finally:
        report['elapsed_s'] = perf_counter()-started
        save_report(args.output/'report.json',report)
    print(json.dumps(dict(checks=report['checks'], elapsed_s=report['elapsed_s'])),flush=True)
    if not report['all_passed']:
        raise SystemExit('Validation failed; see report, no calibration was changed')


if __name__ == '__main__':
    main()
