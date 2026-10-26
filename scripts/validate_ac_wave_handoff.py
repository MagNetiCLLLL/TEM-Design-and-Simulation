"""Continue an actual committed aperture field under explicitly pinned code changes.

This is a conditional downstream diagnostic, not a replay of the old producer
under a new identity. Its complex boundary stays in the original immutable store.
No production source is edited here. Pin the two reviewed patches before applying
them, then prepare/execute in a new output directory. Rays may be independently
run there with validate_ac_wave_rays.py; ordinary comparison formats are retained.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
import traceback

import numpy as np

from temsim.calculation_manifest import solver_source_identity
from temsim.immutable_json import json_digest, thaw_json
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.column_wave import _component_events, _prepare_column, _propagate_column_segmented
from temsim.physics.scan_preparation import prepare_scan_drives, scan_drive_identity
from temsim.physics.tip_wave_pipeline import _observation_stops
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_grid import WaveGridNumerics
from ac_wave_validation_metrics import checkpoint_metrics
from validate_ac_wave_downstream import mode_metadata, downstream_receipts, passive_mode_changes
from validate_ac_wave_scan import backend_rows


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'executed-aperture-handoff-v1'
BOUNDARY_MM = 765.
ALLOWED_SOURCE_FILES = {'src/temsim/physics/column_wave.py', 'src/temsim/physics/tip_wave_pipeline.py'}
SCOPE = ('Conditional downstream propagation from an unchanged complete executed 765 mm aperture field. '
    'Producer and consumer implementations are different and separately identified. Source, gun and upstream '
    'fields are not recomputed or relabelled. Downstream column-step comparisons do not refine the upstream '
    'producer, source quadrature, or full model. Physical inputs and original observation tolerances are unchanged.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'src/temsim').rglob('*.py'))}


def apply_patch_bytes(original, patch_text, expected_path):
    """Exact single-file unified-diff application in memory; no fuzzy matching."""
    lines = patch_text.splitlines(keepends=True)
    if [line.rstrip('\r\n') for line in lines[:2]] != ['--- a/'+expected_path, '+++ b/'+expected_path]:
        raise ValueError('Approved patch must name exactly its one source file')
    old = original.decode('utf-8').splitlines(keepends=True)
    output, cursor, i = [], 0, 2
    while i < len(lines):
        header = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*', lines[i].rstrip('\r\n'))
        if header is None:
            raise ValueError('Unexpected approved patch structure')
        start = int(header[1])-1
        if start < cursor:
            raise ValueError('Overlapping patch hunks')
        output.extend(old[cursor:start]); cursor = start
        consumed = produced = 0
        i += 1
        while i < len(lines) and not lines[i].startswith('@@ '):
            line = lines[i]
            if not line or line[0] not in ' +-':
                raise ValueError('Unsupported patch line')
            if line[0] in ' -':
                if cursor >= len(old) or old[cursor].rstrip('\r\n') != line[1:].rstrip('\r\n'):
                    raise ValueError('Patch does not match the exact producer source')
                if line[0] == ' ':
                    output.append(old[cursor])
                cursor += 1; consumed += 1
            if line[0] == '+':
                output.append(line[1:]); produced += 1
            elif line[0] == ' ':
                produced += 1
            i += 1
        if consumed != int(header[2] or 1) or produced != int(header[4] or 1):
            raise ValueError('Patch hunk counts differ')
    output.extend(old[cursor:])
    return ''.join(output).encode('utf-8')


def pin_changes(producer, patches):
    inventory_path = producer/'solver-source-inventory-before-handoff.json'
    inventory = read(inventory_path)
    if inventory['files'] != source_inventory():
        raise ValueError('Pin patches before production changes; producer inventory must still match')
    changes = []
    for patch in patches:
        text = patch.read_bytes().decode('utf-8')
        relative = text.splitlines()[0].removeprefix('--- a/')
        if relative not in ALLOWED_SOURCE_FILES:
            raise ValueError('Only the two specifically reviewed source files are allowed')
        before = (ROOT/relative).read_bytes()
        after = apply_patch_bytes(before,text,relative)
        changes.append(dict(path=relative, patch_path=str(patch.resolve()), patch_sha256=sha(patch),
            before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest()))
    if {row['path'] for row in changes} != ALLOWED_SOURCE_FILES or len(changes) != 2:
        raise ValueError('Exactly the reviewed HP scheduling and nonsegmented cache patches are required')
    snapshot = InstrumentSnapshot.from_dict(read(producer/'inputs.json'))
    if snapshot.implementation != solver_source_identity():
        raise ValueError('Producer implementation is not the source being pinned')
    document = dict(schema='approved-wave-handoff-source-changes-v1',producer_implementation=snapshot.implementation,
        inventory_path=str(inventory_path.resolve()),inventory_sha256=sha(inventory_path),changes=changes)
    path = producer/'handoff-approved-source-changes.json'
    if path.exists() and read(path) != document:
        raise ValueError('Preserve the existing pinned source-change approval')
    save(path,document)
    return document


def verify_changes(producer, old):
    path = producer/'handoff-approved-source-changes.json'
    approved = read(path)
    if approved['producer_implementation'] != old.implementation:
        raise ValueError('Approved change record belongs to another producer')
    inventory_path = Path(approved['inventory_path'])
    if sha(inventory_path) != approved['inventory_sha256']:
        raise ValueError('Producer source inventory changed')
    before = read(inventory_path)['files']
    expected = dict(before)
    for row in approved['changes']:
        if before.get(row['path']) != row['before_sha256'] or sha(row['patch_path']) != row['patch_sha256']:
            raise ValueError('Pinned source patch evidence changed')
        expected[row['path']] = row['after_sha256']
    if source_inventory() != expected:
        raise ValueError('Consumer source differs from the exact two pinned patches, or patches are not applied')
    return dict(approval_path=str(path.resolve()),approval_sha256=sha(path),**approved)


def prepare(args):
    producer = args.producer.resolve()
    output = args.output.resolve()
    if output == producer or producer in output.parents or output in producer.parents:
        raise ValueError('Consumer output must be separate from producer and its ancestors')
    old = InstrumentSnapshot.from_dict(read(producer/'inputs.json'))
    protocol = read(producer/'protocol.json')
    original_request = read(producer/'request.json')
    if old.digest != protocol['input_identity'] or json_digest(original_request) != protocol['request_identity']:
        raise ValueError('Producer protocol physical/numerical identities do not match')
    changes = verify_changes(producer,old)
    consumer = InstrumentSnapshot(old.graph,old.external_inputs,solver_source_identity())
    if consumer.implementation == old.implementation:
        raise ValueError('This handoff requires the separately identified patched consumer')
    working = consumer.restore()  # Verify unchanged external bytes under new implementation.
    if consumer.graph != old.graph or consumer.external_inputs != old.external_inputs:
        raise ValueError('Handoff must retain the exact physical graph and external inputs')
    grid = WaveGridNumerics(**original_request['wave_grid']).validate()
    if (grid.compute_backend != 'Require GPU' or not grid.acceleration_enabled
            or working.acceleration_backend != 'Require GPU' or not working.acceleration_enabled
            or not original_request['execution']['segmented']):
        raise ValueError('Handoff requires the original mandatory GPU and segmented numerical policy')
    source_cache = (args.source_cache or Path(original_request['execution']['cache_directory'])).resolve()
    if output == source_cache or source_cache in output.parents:
        raise ValueError('Consumer output cannot be inside producer cache')
    if not (source_cache/(args.cache_key+'.json')).is_file():
        raise ValueError('Receiving-aperture checkpoint has not been atomically committed')
    dependency = json_digest(dict(instrument=old.digest,source=original_request['source'],gun=original_request['gun'],
        surface=original_request['surface'],radial_gun=original_request['radial_gun']))
    store = ExecutedWaveStore(source_cache,dependency,original_request['execution']['maximum_disk_cache_bytes'])
    boundary = store.get(args.cache_key)
    record = boundary.record
    if (boundary.plane_z_mm != BOUNDARY_MM or record.get('schema') != 'aperture-conditioned-cartesian-window-v1'
            or record.get('stop_z_mm') != BOUNDARY_MM or not record.get('wall_estimate',{}).get('within_declared_accuracy')
            or thaw_json(record['numerics']) != original_request['acceptance_window']):
        raise ValueError('Require the complete admitted physical aperture checkpoint with original policy')
    expected_key = store.key(dependency,'acceptance_conditioned_aperture',json_digest((record['upstream_digest'],
        record['schema'],record['span_identity'],original_request['acceptance_window'],grid.column_identity(),
        original_request['column_step_mm'])))
    if args.cache_key != expected_key:
        raise ValueError('Producer cache key does not match its exact column numerical request')
    modes = mode_metadata(boundary)
    if not modes or [r['mode_id'] for r in record['modes']] != [r['mode_id'] for r in modes]:
        raise ValueError('Committed mode population differs from complete producer receipt')
    # The patched operator branch must be absent from the entire prior column.
    pre = _prepare_column(working,working.electron_gun.exit_plane_z_mm,BOUNDARY_MM,original_request['column_step_mm'])
    if (np.any(pre[0].midpoint_hex_normal_m3) or np.any(pre[0].midpoint_hex_skew_m3)
            or any(owner.get('dynamic',False) for owner in pre[3])):
        raise ValueError('Upstream HP/dynamic action prevents this specifically reviewed handoff')
    from temsim.specimen.scene import SpecimenScene
    if not SpecimenScene.from_state(working).is_vacuum or working.equivalent_image_lenses_enabled:
        raise ValueError('This boundary consumer requires the declared vacuum specimen and distributed optics')
    if getattr(getattr(working,'nanopulser',None),'installed',False):
        raise ValueError('Pulsed blanker cannot be skipped')
    end = max(p['z_mm'] for p in protocol['planes'])
    entrance = working.sample.upper_surface_z_mm
    exit_z = working.sample.z_mm+working.sample.thickness_nm*.5e-6
    if _observation_stops(working,BOUNDARY_MM,end,entrance,exit_z,True):
        raise ValueError('Inserted recording stops require the full recording pipeline')
    if working.energy_filter_installed and end >= working.energy_filter.entrance_z_mm:
        raise ValueError('Installed energy filter cannot be skipped')
    if not np.isfinite(args.column_step) or args.column_step <= 0:
        raise ValueError('Downstream step must be finite and positive')
    known = {p['label'] for p in protocol['positions']}
    if args.positions is not None and (not args.positions or len(set(args.positions)) != len(args.positions) or not set(args.positions) <= known):
        raise ValueError('Requested positions must be unique original scan labels')
    boundary_identity = dict(schema=SCHEMA,plane_z_mm=BOUNDARY_MM,checkpoint_digest=boundary.digest,
        producer_input_identity=old.digest,producer_request_identity=protocol['request_identity'],
        producer_implementation=old.implementation,consumer_implementation=consumer.implementation,
        consumer_input_identity=consumer.digest,source_change_approval=changes['approval_sha256'],scope=SCOPE)
    request = deepcopy(original_request)
    request['column_step_mm'] = args.column_step
    request['execution']['cache_directory'] = str(output/'cache')
    request['handoff_boundary'] = boundary_identity
    request_identity = json_digest(request)
    protocol.update(input_identity=consumer.digest,request_identity=request_identity,column_step_mm=args.column_step,
        checkpoint_cache_directory=request['execution']['cache_directory'],handoff_boundary=boundary_identity,
        scope=protocol['scope']+' '+SCOPE)
    descriptor = dict(**boundary_identity,source_cache=str(source_cache),source_cache_key=args.cache_key,
        source_dependency=dependency,producer_directory=str(producer),producer_files_sha256={
            name:sha(producer/name) for name in ('inputs.json','protocol.json','request.json')},
        source_changes=changes,modes=modes,reference_current_a=boundary.reference_current_a,
        reference_plane=boundary.beam.reference_plane,storage=thaw_json(record['storage']),
        upstream_eligibility=dict(first_z_mm=float(pre[0].z_mm[0]),last_z_mm=float(pre[0].z_mm[-1]),
            hexapole_maximum=0.,dynamic_owners=[]),
        phase_policy='Read exact original complex arrays, bases, origins, carriers, weights, energies and axial clocks; no phase fitting, renormalization or replacement source',
        checksum_policy='Original immutable manifest checked at admission and every mode-array checksum verified on loading')
    output.mkdir(parents=True,exist_ok=True)
    for name,document in (('inputs.json',consumer.to_dict()),('producer-inputs.json',old.to_dict()),
            ('protocol.json',protocol),('request.json',request),('handoff.json',descriptor)):
        path=output/name
        if path.exists() and read(path) != document:
            raise ValueError(f'Existing {name} belongs to a different handoff; use a new output directory')
        save(path,document)
    save(output/'producer-boundary-receipt.json',thaw_json(record))
    return consumer,working,protocol,request,grid,boundary,descriptor


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer',type=Path,required=True)
    parser.add_argument('--pin-patches',type=Path,nargs=2,metavar=('HP_PATCH','CACHE_PATCH'))
    parser.add_argument('--cache-key')
    parser.add_argument('--source-cache',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--column-step',type=float,default=.05)
    parser.add_argument('--positions',nargs='+')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    if args.pin_patches:
        print(json.dumps(pin_changes(args.producer.resolve(),args.pin_patches),indent=2));return
    if args.cache_key is None or args.output is None:
        parser.error('--cache-key and --output are required for handoff')
    if not args.prepare_only:
        from temsim.physics.compute_backend import require_coherent_cuda
        require_coherent_cuda()
    snapshot,state,protocol,request,grid,boundary,descriptor=prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(status='PREPARED_HANDOFF_ONLY',input_identity=snapshot.digest,
            request_identity=protocol['request_identity'],handoff=descriptor)));return
    from temsim.cpu_resources import numerical_job
    from temsim.physics.wave_device import release_device_memory
    from temsim.gui.coherent_beam import _intensity_preview
    output=args.output.resolve()
    store=ExecutedWaveStore(output/'cache',json_digest((SCHEMA,snapshot.digest,protocol['request_identity'],boundary.digest)),
        request['execution']['maximum_disk_cache_bytes'])
    report_path=output/'wave.json'
    report=read(report_path) if report_path.exists() else dict(input_identity=snapshot.digest,
        request_identity=protocol['request_identity'],handoff_boundary=request['handoff_boundary'],results=[])
    if report['input_identity'] != snapshot.digest or report['request_identity'] != protocol['request_identity']:
        raise ValueError('Retained wave report belongs to another consumer')
    report.update(status='RUNNING',scope=SCOPE,started_at_utc=datetime.now(timezone.utc).isoformat())
    report.pop('error',None);report.pop('traceback',None)
    save(report_path,report)
    started=perf_counter();last_print=[0.];active={}
    def cancelled():return (output/'cancel').exists()
    def progress(done,total,message):
        now=perf_counter()
        if now-last_print[0]>=5. or done==total:
            row=dict(**active,elapsed_s=now-started,done=done,total=total,message=message)
            save(output/'progress.json',row);print(json.dumps(row),flush=True);last_print[0]=now
    try:
        for position in protocol['positions']:
            if args.positions is not None and position['label'] not in args.positions:continue
            working=snapshot.restore()
            prepare_scan_drives(working,snapshot.digest,observation_stop_z_mm=max(p['z_mm'] for p in protocol['planes']))
            identity=scan_drive_identity(working,emission_time_s=position['tip_time_s'])
            current=boundary
            for plane in protocol['planes']:
                if plane['z_mm'] <= current.plane_z_mm:raise ValueError('Observation must advance beyond boundary')
                active.update(position=position['label'],plane=plane['label'])
                before=perf_counter()
                with numerical_job(8,cancelled=cancelled):
                    current,hit=_propagate_column_segmented(working,current,plane['z_mm'],store=store,
                        segment_steps=request['execution']['segment_steps'],maximum_step_mm=args.column_step,
                        grid_numerics=grid,tip_time_s=position['tip_time_s'],cancelled=cancelled,
                        progress_callback=progress,verify=lambda:snapshot.verify_current_inputs(working))
                receipts=downstream_receipts(current.record,boundary.digest)
                modes=mode_metadata(current)
                changes=passive_mode_changes(descriptor['modes'],modes)
                records=thaw_json(current.record)
                row=dict(**active,z_mm=plane['z_mm'],tip_time_s=position['tip_time_s'],
                    instrument_digest=snapshot.digest,request_identity=protocol['request_identity'],scan_identity=identity,
                    metrics=checkpoint_metrics(current),checkpoint_digest=current.digest,cache_hit=hit,
                    backends=backend_rows(receipts),downstream_backend_segments=receipts,
                    handoff_boundary=request['handoff_boundary'],mode_weight_changes=changes,
                    output_modes=modes,elapsed_s=perf_counter()-before)
                stem=position['label']+'_'+plane['label']
                save(output/(stem+'_receipt.json'),records)
                preview=_intensity_preview(current,bins=256)
                np.savez_compressed(output/(stem+'.npz'),density=preview.density,bounds_um=preview.bounds_um)
                report['results']=[r for r in report['results'] if (r['position'],r['plane'])!=(row['position'],row['plane'])]+[row]
                save(report_path,report)
                print(json.dumps(dict(status='PLANE_COMPLETE',**row)),flush=True)
                del preview,records
            del current,working
        report['status']='COMPLETED_REQUESTED_POSITIONS'
    except BaseException as error:
        report.update(status='CANCELLED' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED',
            error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['last_execution_elapsed_s']=perf_counter()-started
        save(report_path,report);release_device_memory()


if __name__=='__main__':main()
