"""Conditional downstream refinement of an already executed 765 mm wave.

This diagnostic consumes one committed complete receiving-aperture checkpoint.
It does not execute or re-qualify the source, gun or upstream column. The
producer identity and original arrays remain unchanged in their original store.
Only the downstream column step is varied; all other numerical controls are
read from the baseline request. Running this script explicitly launches GPU
propagation; --prepare-only validates metadata without propagating a field.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from temsim.immutable_json import json_digest, thaw_json
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.column_wave import _component_events, _propagate_column_segmented, _previous_support_error
from temsim.physics.scan_preparation import prepare_scan_drives, scan_drive_identity
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.physics.wave_flux import FLUX_ATOL, FLUX_RTOL
from temsim.physics.tip_wave_pipeline import _observation_stops
from ac_wave_validation_metrics import checkpoint_metrics
from compare_ac_wave_scan import focus_metrics, focus_admitted, validated_rows
from validate_ac_wave_scan import backend_rows


SCHEMA = 'conditional-ac-wave-downstream-v1'
BOUNDARY_MM = 765.
SCOPE = (
    'Conditional on the unchanged complete executed 765 mm aperture field. '
    'Downstream step comparison only for the named scan states; not source/gun, '
    'upstream, energy/mode, full-raster or total-model qualification. Existing '
    'baseline pass/fail decisions and tolerances are unchanged.'
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mode_metadata(checkpoint):
    """No amplitudes loaded; normal store readers verify them during execution."""
    rows = []
    for mode in checkpoint.beam.modes.rows:
        clock = thaw_json(mode['axial_reference'])
        if clock is None or not math.isfinite(clock['flight_time_s']):
            raise ValueError('Every executed mode must retain its tip-to-boundary clock')
        rows.append(dict(mode_id=mode['mode_id'], energy_kev=mode['energy_kev'],
            weight=mode['weight'], axial_reference=clock,
            scattering_history=thaw_json(mode.get('scattering_history', ())),
            arrays=thaw_json(mode['arrays'])))
    return rows


def downstream_receipts(record, boundary_digest):
    """Collect only new downstream evidence, excluding all upstream backends."""
    segments, cursor = [], record
    while isinstance(cursor, Mapping) and cursor.get('schema') == 'executed-column-segment-v1':
        modes = thaw_json(cursor['modes'])
        receipts = backend_rows(modes)
        if not receipts or any(row.get('compute_backend') != 'cupy' for row in receipts):
            raise ValueError('Conditional downstream segment lacks exclusive actual CUDA wave evidence')
        segments.append(dict(parent_plan_signature=cursor['parent_plan_signature'],
            node_span=thaw_json(cursor['node_span']), mode_count=len(modes), backends=receipts,
            physical_clips=[dict(mode_id=mode['mode_id'], **loss)
                for mode in modes for loss in mode.get('losses',())]))
        if cursor['upstream_digest'] == boundary_digest:
            return list(reversed(segments))
        cursor = cursor.get('upstream')
    raise ValueError('Downstream record does not terminate at the supplied executed boundary digest')


def passive_mode_changes(original, current):
    """Absolute mode probabilities may decrease at hardware or numerical crops."""
    if [r['mode_id'] for r in original] != [r['mode_id'] for r in current]:
        raise ValueError('Conditional propagation changed the mode identities or ordering')
    changes = []
    for old, new in zip(original,current):
        before, after = old['weight'], new['weight']
        if (not math.isfinite(after) or after < 0
                or after > before+(FLUX_RTOL+FLUX_ATOL)*before):
            raise ValueError('Passive downstream propagation increased an absolute mode weight beyond roundoff')
        changes.append(dict(mode_id=old['mode_id'],input_weight=before,output_weight=after,
                            weight_decrease=before-after))
    return changes


def validate_baseline_ancestry(folder, report, required, boundary):
    """Bind selected executed observations to this exact upstream field."""
    evidence = []
    for row in report['results']:
        if (row['position'],row['plane']) not in required:
            continue
        path = folder/(row['position']+'_'+row['plane']+'_receipt.json')
        receipt = read(path)
        storage = receipt.get('storage')
        if storage is None or row['checkpoint_digest'] != json_digest(dict(record=receipt,
                stored_payload=storage['manifest_digest'],plane_z_mm=row['z_mm'],
                current_a=row['metrics']['reference_current_a'])):
            raise ValueError('Baseline receipt is not bound to its executed plane checkpoint')
        cursor = receipt
        while isinstance(cursor,dict):
            if cursor.get('upstream_digest') == boundary.digest:
                if json_digest(cursor.get('upstream')) != json_digest(boundary.record):
                    raise ValueError('Baseline ancestry claims the boundary digest but contains different input data')
                break
            cursor = cursor.get('upstream')
        else:
            raise ValueError('Selected baseline plane does not descend from the supplied complete 765 mm field')
        evidence.append(dict(position=row['position'],plane=row['plane'],
            checkpoint_digest=row['checkpoint_digest'],receipt_path=str(path.resolve()),
            receipt_sha256=sha(path),boundary_checkpoint_digest=boundary.digest))
    if len(evidence) != len(required):
        raise ValueError('Missing or duplicate selected baseline ancestry evidence')
    return evidence


def compare_rows(results, baseline, protocol, labels):
    rows = []
    current = {(r['position'], r['plane']): r for r in results if r['category'] == 'protocol'}
    limits = protocol['limits']
    required = {(name, p['label']) for name in labels for p in protocol['planes']}
    if not required <= current.keys():
        return dict(status='INCOMPLETE_CONDITIONAL_COMPARISON', checks={}, rows=[])
    for name, plane in sorted(required):
        old, new = baseline[name, plane], current[name, plane]['metrics']
        displacement = float(np.linalg.norm(np.asarray(old['centroid_xy_m'])-new['centroid_xy_m'])*1e9)
        widths = {key: abs(new[key]/old[key]-1.) for key in ('radial_rms_m','r50_nm','r80_nm','r95_nm')}
        limit = limits['refinement_diffraction_centroid_nm' if plane == 'diffraction' else 'refinement_sample_centroid_nm']
        rows.append(dict(position=name, plane=plane, centroid_difference_nm=displacement,
            centroid_limit_nm=limit, width_change_fractions=widths,
            baseline_probability=old['total_probability'], conditional_probability=new['total_probability'],
            probability_difference=new['total_probability']-old['total_probability'],
            baseline_edge_fraction=old['edge_1pct_fraction'], conditional_edge_fraction=new['edge_1pct_fraction']))
    focus = []
    for name in labels:
        original = focus_metrics({p['label']: baseline[name,p['label']] for p in protocol['planes']},protocol['planes'])
        refined = focus_metrics({p['label']: current[name,p['label']]['metrics'] for p in protocol['planes']},protocol['planes'])
        focus.append(dict(position=name, baseline=original, conditional=refined,
            baseline_admitted=focus_admitted(original,limits['focus_bracket_halfwidth_nm']),
            conditional_admitted=focus_admitted(refined,limits['focus_bracket_halfwidth_nm'])))
    checks = dict(
        centres=all(row['centroid_difference_nm'] <= row['centroid_limit_nm'] for row in rows),
        widths=all(max(row['width_change_fractions'].values()) <= limits['refinement_width_fraction'] for row in rows),
        local_three_plane_focus=all(row['baseline_admitted'] and row['conditional_admitted'] for row in focus))
    return dict(status='PASSED_CONDITIONAL_CHECKS' if all(checks.values()) else 'FAILED_CONDITIONAL_CHECKS',
        checks=checks, rows=rows, focus=focus, scope=SCOPE)


def load_inputs(args):
    baseline_folder = args.baseline.resolve()
    protocol = read(baseline_folder/'protocol.json')
    request = read(baseline_folder/'request.json')
    snapshot = InstrumentSnapshot.from_dict(read(baseline_folder/'inputs.json'))
    if snapshot.digest != protocol['input_identity'] or json_digest(request) != protocol['request_identity']:
        raise ValueError('Baseline physical or numerical identity differs from its protocol')
    if request['column_step_mm'] != .05 or protocol['column_step_mm'] != .05:
        raise ValueError('This bounded diagnostic requires the declared 0.05 mm baseline')
    if not 0 < args.column_step <= request['column_step_mm'] or not math.isfinite(args.column_step):
        raise ValueError('Conditional step must be finite, positive and no coarser than baseline')
    names = [position['label'] for position in protocol['positions']]
    if (len(set(args.labels)) != len(args.labels) or 'centre' not in args.labels
            or len(args.labels) != 2 or not set(args.labels) <= set(names)):
        raise ValueError('--labels must contain centre and exactly one supplied corner label')
    state = snapshot.restore()  # Strict current implementation and external-input verification.
    grid = WaveGridNumerics(**request['wave_grid'])
    grid.validate()
    if (grid.compute_backend != state.acceleration_backend or grid.acceleration_enabled != state.acceleration_enabled
            or grid.compute_backend != 'Require GPU' or not grid.acceleration_enabled):
        raise ValueError('Baseline source and wave request must both retain Require GPU')
    if not request['execution']['segmented']:
        raise ValueError('Baseline must use production segmented column execution')
    source_root = (args.source_cache or Path(request['execution']['cache_directory'])).resolve()
    if not source_root.is_dir() or not (source_root/(args.cache_key+'.json')).is_file():
        raise ValueError('Explicit source cache key is not yet atomically committed')
    dependency = json_digest(dict(instrument=snapshot.digest, source=request['source'], gun=request['gun'],
                                  surface=request['surface'], radial_gun=request['radial_gun']))
    source_store = ExecutedWaveStore(source_root,dependency,request['execution']['maximum_disk_cache_bytes'])
    boundary = source_store.get(args.cache_key)
    if boundary is None or not math.isclose(boundary.plane_z_mm,BOUNDARY_MM,rel_tol=0.,abs_tol=1e-9):
        raise ValueError('Require an actual complete checkpoint after the physical 765 mm receiving aperture')
    record = boundary.record
    if (record.get('schema') != 'aperture-conditioned-cartesian-window-v1'
            or record.get('stop_z_mm') != BOUNDARY_MM
            or not record.get('wall_estimate',{}).get('within_declared_accuracy')):
        raise ValueError('Source key is not the complete admitted receiving-aperture stage')
    modes = mode_metadata(boundary)
    if not modes or {r['mode_id'] for r in record['modes']} != {r['mode_id'] for r in modes}:
        raise ValueError('Receiving-aperture record and complete stored mode population differ')
    if thaw_json(record['numerics']) != request['acceptance_window']:
        raise ValueError('Boundary producer used a different aperture approximation request')
    expected_key = source_store.key(dependency,'acceptance_conditioned_aperture',json_digest((
        record['upstream_digest'],record['schema'],record['span_identity'],request['acceptance_window'],
        grid.column_identity(),request['column_step_mm'])))
    if args.cache_key != expected_key:
        raise ValueError('Boundary cache key does not match the baseline column step and numerical controls')
    _, upstream_owners = _component_events(state,state.electron_gun.exit_plane_z_mm,BOUNDARY_MM)
    if any(owner.get('dynamic',False) for owner in upstream_owners):
        raise ValueError('A single shared boundary cannot replace scan-dependent upstream fields')
    from temsim.specimen.scene import SpecimenScene
    if not SpecimenScene.from_state(state).is_vacuum:
        raise ValueError('This diagnostic only supports the declared vacuum specimen')
    if bool(getattr(state,'equivalent_image_lenses_enabled',False)):
        raise ValueError('Distributed physical optics must remain active')
    if getattr(getattr(state,'nanopulser',None),'installed',False):
        raise ValueError('Installed pulsed blanker needs the complete time-energy model')
    end = max(p['z_mm'] for p in protocol['planes'])
    entrance = float(state.sample.upper_surface_z_mm)
    exit_z = float(state.sample.z_mm)+float(state.sample.thickness_nm)*.5e-6
    stops = _observation_stops(state,state.electron_gun.exit_plane_z_mm,end,entrance,exit_z,True)
    if any(BOUNDARY_MM <= float(d.z_mm) < end for d in stops):
        raise ValueError('Inserted downstream recording stops need full pipeline handling; diagnostic refuses to skip them')
    if getattr(state,'energy_filter_installed',False) and end >= state.energy_filter.entrance_z_mm:
        raise ValueError('This diagnostic cannot skip an installed energy-filter branch')
    report = read(baseline_folder/'wave.json')
    if report.get('status') != 'COMPLETED_REQUESTED_POSITIONS':
        raise ValueError('Baseline wave report must be completed before conditional comparison')
    if report.get('input_identity') != snapshot.digest or report.get('request_identity') != protocol['request_identity']:
        raise ValueError('Baseline wave report does not match the producer inputs')
    baseline_rows = validated_rows(report,protocol,wave=True)
    required = {(name,p['label']) for name in args.labels for p in protocol['planes']}
    if not required <= baseline_rows.keys():
        raise ValueError('Baseline lacks complete centre/corner comparison planes')
    ancestry = validate_baseline_ancestry(baseline_folder,report,required,boundary)
    producer = dict(cache_directory=str(source_root),cache_key=args.cache_key,dependency=dependency,
        checkpoint_digest=boundary.digest, storage=thaw_json(boundary.record['storage']),
        instrument_digest=snapshot.digest,request_identity=protocol['request_identity'],
        plane_z_mm=boundary.plane_z_mm,reference_current_a=boundary.reference_current_a,
        reference_plane=boundary.beam.reference_plane, modes=modes,
        baseline_receipt_ancestry=ancestry,baseline_numerical_cache_key_verified=True,
        phase_policy='Exact original complex cell amplitudes and carrier arrays; no phase fit or replacement source. Production unit-shape/absolute-weight factorization preserves attenuation.',
        array_verification='Committed manifest verified at admission; every mode array checksum is verified by the store reader before propagation')
    return snapshot,state,protocol,request,grid,boundary,producer,baseline_rows,report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--source-cache',type=Path,help='Default: unchanged baseline request cache path')
    parser.add_argument('--cache-key',required=True,help='Explicit committed complete 765 mm aperture checkpoint key')
    parser.add_argument('--output',type=Path,required=True,help='Separate directory outside the baseline directory')
    parser.add_argument('--labels',nargs=2,required=True,metavar=('CENTRE','CORNER'))
    parser.add_argument('--selection-reason',required=True,help='Record how the worst corner was chosen from baseline results')
    parser.add_argument('--column-step',type=float,default=.025)
    parser.add_argument('--supplement-focus',action='store_true',help='Also observe centre at sample +/-40 nm, separately from the original protocol')
    parser.add_argument('--prepare-only',action='store_true',help='Validate committed metadata and save a plan; no propagation or GPU execution')
    args = parser.parse_args(argv)
    output, baseline_folder = args.output.resolve(), args.baseline.resolve()
    if output == baseline_folder or baseline_folder in output.parents or output in baseline_folder.parents:
        parser.error('--output must be separate from the baseline and its ancestors')
    if not args.prepare_only:
        from temsim.physics.compute_backend import require_coherent_cuda
        require_coherent_cuda()
    snapshot,state,protocol,request,grid,boundary,producer,base,baseline_report = load_inputs(args)
    if output == Path(producer['cache_directory']) or Path(producer['cache_directory']) in output.parents:
        parser.error('--output must not be inside the source cache')
    supplemental = []
    if args.supplement_focus:
        sample = next(p['z_mm'] for p in protocol['planes'] if p['label'] == 'sample')
        supplemental = [dict(label='supplement_minus40nm',z_mm=sample-40e-6),
                        dict(label='supplement_plus40nm',z_mm=sample+40e-6)]
    descriptor = dict(schema=SCHEMA,scope=SCOPE,producer=producer,
        baseline_directory=str(baseline_folder),baseline_protocol=protocol,
        baseline_files_sha256={name:sha(baseline_folder/name) for name in ('inputs.json','protocol.json','request.json','wave.json')},
        labels=args.labels,selection_reason=args.selection_reason,
        downstream_column_step_mm=args.column_step,baseline_column_step_mm=request['column_step_mm'],
        all_other_numerics=request, supplemental_centre_planes=supplemental,
        supplement_scope='Additional observation planes only; no baseline protocol mutation or replacement focus decision',
        upstream_validation='Conditioned on producer result only; not recomputed or relabelled',
        new_cache_directory=str(output/'cache'))
    identity = json_digest(descriptor)
    output.mkdir(parents=True,exist_ok=True)
    plan_path = output/'conditional-request.json'
    if plan_path.exists() and read(plan_path) != descriptor:
        raise ValueError('Existing conditional output directory belongs to a different plan')
    save(plan_path,descriptor)
    if args.prepare_only:
        print(json.dumps(dict(status='PREPARED_ONLY',conditional_identity=identity,scope=SCOPE)))
        return
    from temsim.cpu_resources import numerical_job
    from temsim.physics.wave_device import release_device_memory
    store = ExecutedWaveStore(output/'cache',identity,request['execution']['maximum_disk_cache_bytes'])
    report_path = output/'conditional-wave.json'
    report = read(report_path) if report_path.exists() else dict(schema=SCHEMA,conditional_identity=identity,
        producer_checkpoint_digest=boundary.digest,producer_instrument_digest=snapshot.digest,
        producer_request_identity=protocol['request_identity'],scope=SCOPE,results=[])
    if report['conditional_identity'] != identity:
        raise ValueError('Conditional results belong to a different request')
    report.update(status='RUNNING_CONDITIONAL_DOWNSTREAM',started_at_utc=datetime.now(timezone.utc).isoformat())
    report.pop('error',None); report.pop('traceback',None)
    save(report_path,report)
    started = perf_counter()
    last_print = [0.]
    active = {}
    def cancelled():
        return (output/'cancel').exists()
    def progress(done,total,message):
        now=perf_counter()
        if now-last_print[0]>=5. or done==total:
            row=dict(**active,elapsed_s=now-started,done=done,total=total,message=message)
            save(output/'progress.json',row); print(json.dumps(row),flush=True)
            last_print[0]=now
    try:
        for position in protocol['positions']:
            if position['label'] not in args.labels:
                continue
            groups = [('protocol',protocol['planes'])]
            if position['label']=='centre' and supplemental:
                # Supplementary breaks must not change the protocol-plane path.
                groups.append(('supplemental_only',supplemental))
            for category,planes in groups:
                working = snapshot.restore()
                prepare_scan_drives(working,snapshot.digest,observation_stop_z_mm=max(p['z_mm'] for p in planes))
                scan_identity=scan_drive_identity(working,emission_time_s=position['tip_time_s'])
                baseline_scan_ids={r['scan_identity'] for r in baseline_report['results']
                    if r['position']==position['label']}
                if baseline_scan_ids != {scan_identity}:
                    raise ValueError('Resolved physical scan commands differ from the baseline')
                current=boundary
                for plane in sorted(planes,key=lambda p:p['z_mm']):
                    active.update(position=position['label'],plane=plane['label'],category=category)
                    if plane['z_mm'] <= current.plane_z_mm:
                        raise ValueError('Conditional observations must advance from a real upstream checkpoint')
                    before=perf_counter()
                    with numerical_job(8,cancelled=cancelled):
                        current,hit=_propagate_column_segmented(working,current,plane['z_mm'],store=store,
                            segment_steps=request['execution']['segment_steps'],maximum_step_mm=args.column_step,
                            grid_numerics=grid,tip_time_s=position['tip_time_s'],cancelled=cancelled,
                            progress_callback=progress,verify=lambda:snapshot.verify_current_inputs(working))
                    receipts=downstream_receipts(current.record,boundary.digest)
                    output_modes=mode_metadata(current)
                    weight_changes=passive_mode_changes(producer['modes'],output_modes)
                    numerical_loss=[]
                    for mode in output_modes:
                        old_ledger=_previous_support_error(boundary.record,mode['mode_id']) or {}
                        new_ledger=_previous_support_error(current.record,mode['mode_id']) or {}
                        numerical_loss.append(dict(mode_id=mode['mode_id'],
                            recorded_support_probability_loss=(new_ledger.get('probability_loss_per_tip_electron',0.)
                                -old_ledger.get('probability_loss_per_tip_electron',0.))))
                    row=dict(**active,z_mm=plane['z_mm'],tip_time_s=position['tip_time_s'],scan_identity=scan_identity,
                        metrics=checkpoint_metrics(current),checkpoint_digest=current.digest,
                        checkpoint_storage=thaw_json(current.record['storage']),output_modes=output_modes,
                        mode_weight_changes=weight_changes,downstream_numerical_support_loss=numerical_loss,
                        downstream_backend_segments=receipts,cache_hit=hit,elapsed_s=perf_counter()-before)
                    stem=position['label']+'_'+plane['label']
                    save(output/(stem+'_receipt.json'),thaw_json(current.record))
                    report['results']=[r for r in report['results'] if (r['position'],r['plane'])!=(row['position'],row['plane'])]+[row]
                    report['comparison']=compare_rows(report['results'],base,protocol,args.labels)
                    save(report_path,report)
                    print(json.dumps(dict(status='CONDITIONAL_PLANE_COMPLETE',position=row['position'],plane=row['plane'],
                        checkpoint_digest=row['checkpoint_digest'],metrics=row['metrics'])),flush=True)
                del current,working
        report['status']='COMPLETED_CONDITIONAL_DOWNSTREAM'
    except BaseException as error:
        report.update(status='CANCELLED_CONDITIONAL_DOWNSTREAM' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED_CONDITIONAL_DOWNSTREAM',
                      error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['last_execution_elapsed_s']=perf_counter()-started
        save(report_path,report)
        release_device_memory()


if __name__=='__main__':
    main()
