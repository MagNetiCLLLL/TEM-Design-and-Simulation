"""Execute the saved AC optics with a declared Tip boundary and a real raster.

Historical ray inputs lack phase. Keep their nominal emission and optics, add
the documented driven boundary, and capture new inputs for BOTH representations.
No historical execution is re-labelled as a current matched calculation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from temsim.immutable_json import json_digest, thaw_json
from temsim.instrument_snapshot import InstrumentSnapshot, decode_instrument, capture_instrument_snapshot
from temsim.optics.electron_gun.tip_coherence import TipCoherence, DRIVEN_GAUSSIAN_SCHELL
from temsim.physics.coherent_inputs import prepare_coherent_state, wave_input_summary
from temsim.physics.tip_wave_pipeline import TipWaveRequest, TipWaveObservationSession
from temsim.physics.scan_geometry import calibrate_scan_system


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def assert_same_physical_graph(left, right):
    """Compare all captured values and aliases, ignoring graph node numbering."""
    left, right = thaw_json(left), thaw_json(right)
    forward, reverse = {}, {}

    def visit(a, b, location):
        if isinstance(a, dict) and set(a) == {'ref'} and isinstance(b, dict) and set(b) == {'ref'}:
            ai, bi = a['ref'], b['ref']
            if ai in forward or bi in reverse:
                if forward.get(ai) != bi or reverse.get(bi) != ai:
                    raise ValueError(f'Physical graph alias changed at {location}')
                return
            forward[ai], reverse[bi] = bi, ai
            visit(left['nodes'][ai], right['nodes'][bi], location)
        elif type(a) is not type(b):
            raise ValueError(f'Physical input type changed at {location}')
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError(f'Physical input fields changed at {location}')
            for key in a:
                visit(a[key], b[key], f'{location}.{key}')
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError(f'Physical input length changed at {location}')
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f'{location}[{index}]')
        elif a != b:
            raise ValueError(f'Physical input changed at {location}: {a!r} -> {b!r}')

    if left['schema'] != right['schema']:
        raise ValueError('Physical graph schema changed')
    visit(left['root'], right['root'], 'state')
    if len(forward) != len(left['nodes']) or len(reverse) != len(right['nodes']):
        raise ValueError('Physical graph contains unmatched nodes')
    return len(forward)


def recapture_inputs(folder, source):
    """Explicitly capture old physical inputs against the current implementation.

    Old results/cache are evidence only. No executed wave is copied or trusted.
    The ordinary restore path remains strict for every subsequent calculation.
    """
    source_file = source/'inputs.json' if source.is_dir() else source
    source_folder = source_file.parent.resolve()
    if folder.resolve() == source_folder or any(folder.iterdir()):
        raise ValueError('Input recapture requires a new, empty output directory')
    original = InstrumentSnapshot.from_dict(json.loads(source_file.read_text(encoding='utf-8')))
    protocol = json.loads((source_folder/'protocol.json').read_text(encoding='utf-8'))
    if protocol['input_identity'] != original.digest:
        raise ValueError('Source protocol and captured input identities differ')
    state = decode_instrument(original.graph)
    original._verify_external_inputs(state)
    current = capture_instrument_snapshot(state)
    matched_nodes = assert_same_physical_graph(original.graph, current.graph)
    # Ordinary strict restore checks current code and external inputs too.
    state = current.restore()
    assert_same_physical_graph(original.graph, capture_instrument_snapshot(state).graph)
    old_wave = source_folder/'wave.json'
    old_result = json.loads(old_wave.read_text(encoding='utf-8')) if old_wave.exists() else {}
    receipt = dict(schema='ac-wave-input-recapture-v1',
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        source_input_path=str(source_file.resolve()), source_input_identity=original.digest,
        source_input_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        new_input_identity=current.digest, physical_graph_equal=True, matched_nodes=matched_nodes,
        source_implementation=thaw_json(original.implementation),
        new_implementation=thaw_json(current.implementation),
        prior_result_path=str(old_wave), prior_result_status=old_result.get('status'),
        prior_result_error=old_result.get('error'),
        prior_result_sha256=hashlib.sha256(old_wave.read_bytes()).hexdigest() if old_wave.exists() else None,
        policy='Physical input graph preserved exactly. Current code must execute again; no prior result or cache copied.')
    protocol.update(input_identity=current.digest, recaptured_from_identity=original.digest,
        scan_scope='64x64 1000 nm FOV raster, five pixel-centre epochs. Wave coils use per-mode arrival time; particles freeze fields at the tip epoch and must pass the recorded first-order timing-error gate.')
    save(folder/'inputs.json', current.to_dict())
    save(folder/'protocol.json', protocol)
    save(folder/'input-recapture.json', receipt)
    return state, protocol


def prepare(folder, previous):
    old = InstrumentSnapshot.from_dict(json.loads((previous/'inputs.json').read_text()))
    state = decode_instrument(old.graph)
    old._verify_external_inputs(state)
    emitter = state.electron_gun.emitter
    if emitter.coherence is not None or emitter.surface_model is not None:
        raise ValueError('This receipt expects the explicitly documented historical classical Tip')
    emitter.coherence = TipCoherence(boundary_model=DRIVEN_GAUSSIAN_SCHELL,
        incoherent_angle_rms_mrad=emitter.angular_rms_mrad)
    emitter.validate()
    state.acceleration_enabled = True
    state.acceleration_backend = 'Require GPU'
    calibrate_scan_system(state, observation_stop_z_mm=float(state.sample.z_mm))
    state = prepare_coherent_state(state)
    snapshot = capture_instrument_snapshot(state)
    save(folder/'inputs.json', snapshot.to_dict())
    ac = state.ac_deflector
    assert ac.scan_enabled and not ac.wobble_enabled and not state.descan_deflector.scan_enabled
    assert ac.scan_pixels_x == 64 and ac.scan_lines == 64
    assert ac.scan_field_of_view_x_nm == 1000. and ac.scan_field_of_view_y_nm == 1000.
    samples = []
    for label, row, column in [('centre',32,32),('lower_left',0,0),('lower_right',0,63),
                               ('upper_left',63,0),('upper_right',63,63)]:
        time = (row+(column+.5)/ac.scan_pixels_x)/ac.scan_lines*ac.scan_frame_period_s
        samples.append(dict(label=label,row=row,column=column,tip_time_s=time,
                            factors=list(ac.scan_factors(time))))
    entrance = float(state.sample.upper_surface_z_mm)
    old_report = json.loads((previous/'report.reviewed.json').read_text())
    planes = [dict(label='before_sample',z_mm=entrance-.00002),
              dict(label='sample',z_mm=entrance),
              dict(label='after_sample',z_mm=entrance+.00002),
              dict(label='diffraction',z_mm=old_report['diffraction']['z_mm'])]
    protocol = dict(input_identity=snapshot.digest, historical_input_identity=old.digest,
        source_change='Added driven Gaussian-Schell boundary with zero phase curvature/tilt and the original 1 mrad per-axis incoherent RMS. Nominal width/energy/current and optics retained. Historical 3-sigma sampling cutoffs no longer truncate this declared boundary.',
        positions=samples,planes=planes,column_step_mm=.05,
        scan_scope='64x64 1000 nm FOV real raster, five pixel-centre epochs; outer pixels at +/-492.1875 nm, reference pixel at +7.8125 nm. Finite coils use arrival time, not a specimen shift.',
        limits=dict(sample_position_error_nm=.2, sample_ray_wave_displacement_difference_nm=.2,
            diffraction_motion_nm=1., diffraction_ray_wave_displacement_difference_nm=1.,
            scan_containment_width_fraction=.1, refinement_sample_centroid_nm=.05,
            refinement_diffraction_centroid_nm=.2, refinement_width_fraction=.05,
            focus_bracket_halfwidth_nm=20.),
        scope='Vacuum; no sample material, ideal residual gas. All installed lenses, Cs, HP, apertures and coil fields retained. Gun wave is CPU, column wave must be CUDA.')
    save(folder/'protocol.json',protocol)
    return state, protocol


def backend_rows(record):
    rows=[]
    def visit(value):
        if isinstance(value,dict):
            if 'compute_backend' in value:
                rows.append({key:value[key] for key in ('compute_backend','numeric_precision','fallback_reason') if key in value})
            for child in value.values(): visit(child)
        elif isinstance(value,(list,tuple)):
            for child in value: visit(child)
    visit(record)
    unique={json.dumps(row,sort_keys=True):row for row in rows}
    return list(unique.values())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/ac_wave_scan_20261010'))
    parser.add_argument('--previous',type=Path,default=Path('outputs/ac_scan_pivot_20261010'))
    parser.add_argument('--recapture-inputs',type=Path,
        help='Explicitly recapture an existing input file/directory with current code into a new empty --output; preserve every physical parameter')
    parser.add_argument('--column-step',type=float,help='Maximum column step in mm; changed numerics require a new output directory')
    parser.add_argument('--segment-steps',type=int,help='Column steps per persisted segment; changed numerics require a new output directory')
    parser.add_argument('--cache-directory',type=Path,
        help='Identity-checked checkpoint store shared with other validation directories; default is --output/cache')
    parser.add_argument('--tail-budget',type=float,
        help='Maximum support probability loss per crop; numerical bound only, not a change to the physical Tip')
    parser.add_argument('--spectral-budget',type=float,
        help='Maximum discarded Fourier probability per projection; no source or optical changes')
    parser.add_argument('--mode-budget',type=float,
        help='Total omitted independent-mode probability per Tip electron after the gun')
    parser.add_argument('--tail-l2-budget',type=float,
        help='Maximum accumulated support L2 error per mode relative to Tip reference; zero prohibits cropping')
    parser.add_argument('--acceptance-window',action=argparse.BooleanOptionalAction,default=None,
        help='Use the guarded downstream-aperture numerical approximation; physical Tip and optics are unchanged')
    parser.add_argument('--positions',nargs='*',default=None)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    if args.column_step is not None and (not np.isfinite(args.column_step) or args.column_step <= 0):
        parser.error('--column-step must be finite and positive')
    if args.segment_steps is not None and args.segment_steps <= 0:
        parser.error('--segment-steps must be positive')
    for option in ('tail_budget','spectral_budget','mode_budget','tail_l2_budget'):
        value=getattr(args,option)
        if value is not None and (not np.isfinite(value) or value < 0):
            parser.error('--'+option.replace('_','-')+' must be finite and nonnegative')
    if not args.prepare_only:
        from temsim.physics.compute_backend import require_coherent_cuda
        require_coherent_cuda()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.recapture_inputs is not None:
        state,protocol=recapture_inputs(args.output,args.recapture_inputs)
    elif (args.output/'inputs.json').exists():
        snapshot=InstrumentSnapshot.from_dict(json.loads((args.output/'inputs.json').read_text()))
        state=snapshot.restore()
        protocol=json.loads((args.output/'protocol.json').read_text())
    else:
        state,protocol=prepare(args.output,args.previous)
    request_path=args.output/'request.json'
    stored_request=json.loads(request_path.read_text(encoding='utf-8')) if request_path.exists() else None
    column_step=protocol['column_step_mm'] if args.column_step is None else args.column_step
    segment_steps=(stored_request or {}).get('execution',{}).get('segment_steps',128)
    if args.segment_steps is not None:
        segment_steps=args.segment_steps
    cache_directory=(str(args.cache_directory.resolve()) if args.cache_directory is not None else
        (stored_request or {}).get('execution',{}).get('cache_directory',str(args.output/'cache')))
    request=TipWaveRequest(stop='plane',observation_z_mm=protocol['planes'][0]['z_mm'],
                           column_step_mm=column_step)
    request=replace(request,
        wave_grid=replace(request.wave_grid,compute_backend=state.acceleration_backend,
            acceleration_enabled=state.acceleration_enabled,maximum_working_bytes=24*1024**3,
            maximum_device_working_bytes=24*1024**3),
        execution=replace(request.execution,cache_directory=cache_directory,
                          maximum_disk_cache_bytes=256*1024**3,segment_steps=segment_steps))
    support_options={}
    stored_grid=(stored_request or {}).get('wave_grid',{})
    for option,field in (('tail_budget','maximum_support_loss_fraction'),
                         ('spectral_budget','maximum_spectral_loss_fraction'),
                         ('mode_budget','maximum_incoherent_mode_loss_probability'),
                         ('tail_l2_budget','maximum_accumulated_support_l2')):
        value=getattr(args,option)
        if value is None:
            value=stored_grid.get(field)
        if value is not None:
            if not hasattr(request.wave_grid,field):
                raise ValueError(f'This production implementation does not yet support {field}; no calculation started')
            support_options[field]=value
    if support_options:
        request=replace(request,wave_grid=replace(request.wave_grid,**support_options))
    stored_acceptance=(stored_request or {}).get('acceptance_window')
    if args.acceptance_window is not None or stored_acceptance is not None:
        if not hasattr(request,'acceptance_window'):
            raise ValueError('This implementation does not support a guarded acceptance window')
        acceptance_options=dict(stored_acceptance or {})
        if 'guard_scales' in acceptance_options:
            acceptance_options['guard_scales']=tuple(acceptance_options['guard_scales'])
        if args.acceptance_window is not None:
            acceptance_options['enabled']=args.acceptance_window
        request=replace(request,acceptance_window=replace(request.acceptance_window,**acceptance_options))
    request.validate()
    request_document=json.loads(json.dumps(asdict(request)))
    if stored_request is not None and stored_request != request_document:
        raise ValueError('Numerical request differs from this output directory; recapture identical physical inputs into a new directory')
    request_identity=json_digest(request_document)
    if stored_request is None:
        protocol.update(column_step_mm=column_step, request_identity=request_identity,
                        segment_steps=segment_steps,
                        checkpoint_cache_directory=cache_directory,
                        acceptance_window=request_document.get('acceptance_window'),
                        support_numerics={key:request_document['wave_grid'][key] for key in
                            ('maximum_support_loss_fraction','maximum_spectral_loss_fraction',
                             'maximum_incoherent_mode_loss_probability','maximum_accumulated_support_l2')
                            if key in request_document['wave_grid']})
        save(args.output/'protocol.json',protocol)
    elif protocol.get('request_identity',request_identity) != request_identity:
        raise ValueError('Protocol and numerical request identities differ')
    names={position['label'] for position in protocol['positions']}
    if args.positions is not None and (not args.positions or not set(args.positions) <= names):
        raise ValueError('Requested positions must be nonempty labels from the saved protocol')
    summary=thaw_json(wave_input_summary(state,request))
    save(request_path,request_document)
    save(args.output/'source-summary.json',summary)
    print(json.dumps(dict(status='SOURCE',summary=summary)),flush=True)
    assert summary['status']=='SOURCE_READY'
    if args.prepare_only:
        return
    from ac_wave_validation_metrics import checkpoint_metrics
    from temsim.gui.coherent_beam import _intensity_preview
    from temsim.cpu_resources import numerical_job
    report=dict(status='RUNNING',input_identity=protocol['input_identity'],request_identity=request_identity,results=[])
    report_path=args.output/'wave.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report['input_identity'] != protocol['input_identity'] or report.get('request_identity',request_identity) != request_identity:
            raise ValueError('Retained results belong to different inputs or numerics')
        report['request_identity']=request_identity
        report['status']='RUNNING'
        report.pop('error',None)
        report.pop('traceback',None)
    started=perf_counter()
    last_print=[0.]
    active={}
    def cancelled(): return (args.output/'cancel').exists()
    def progress(done,total,message):
        now=perf_counter()
        if now-last_print[0]>=5. or done==total:
            row=dict(**active,elapsed_s=now-started,done=done,total=total,message=message)
            save(args.output/'progress.json',row)
            print(json.dumps(row),flush=True)
            last_print[0]=now
    try:
        for position in protocol['positions']:
            if args.positions is not None and position['label'] not in args.positions:
                continue
            session=TipWaveObservationSession(state,replace(request,tip_time_s=position['tip_time_s']))
            for plane in protocol['planes']:
                active.update(position=position['label'],plane=plane['label'])
                stage_start=perf_counter()
                with numerical_job(8,cancelled=cancelled):
                    result=session.observe(plane['z_mm'],cancelled=cancelled,progress_callback=progress)
                metric=checkpoint_metrics(result.checkpoint)
                records=thaw_json(result.checkpoint.record)
                backends=backend_rows(records)
                if not any(row['compute_backend']=='cupy' for row in backends):
                    raise AssertionError('Column checkpoint lacks actual CuPy execution evidence')
                row=dict(**active,tip_time_s=position['tip_time_s'],z_mm=plane['z_mm'],
                    metrics=metric,backends=backends,checkpoint_digest=result.checkpoint.digest,
                    request_identity=request_identity,
                    instrument_digest=result.instrument_digest,scan_identity=result.resolved_scan_identity,
                    cache_hit=result.propagation_cache_hit,elapsed_s=perf_counter()-stage_start)
                stem=position['label']+'_'+plane['label']
                save(args.output/(stem+'_receipt.json'),records)
                preview=_intensity_preview(result.checkpoint,bins=256)
                np.savez_compressed(args.output/(stem+'.npz'),density=preview.density,bounds_um=preview.bounds_um)
                report['results']=[old for old in report['results'] if (old['position'],old['plane'])!=(row['position'],row['plane'])]+[row]
                save(report_path,report)
                print(json.dumps(dict(status='PLANE_COMPLETE',**row)),flush=True)
                del result,records,preview
        report['status']='COMPLETED_REQUESTED_POSITIONS'
    except Exception as error:
        report.update(status='FAILED',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['last_execution_elapsed_s']=perf_counter()-started
        save(report_path,report)


if __name__=='__main__':
    main()
