"""Compare native wave moments with executed rays at the recorded scan epochs."""
from pathlib import Path
import argparse
import json
import numpy as np


FRAME = 'laboratory XY; +Z downstream'


def validated_rows(report, protocol, *, wave):
    """Validate provenance and native observables before any success summary."""
    result = {}
    positions = {p['label']:p for p in protocol['positions']}
    planes = {p['label']:p for p in protocol['planes']}
    for row in report['results']:
        key = row['position'], row['plane']
        if key in result or key[0] not in positions or key[1] not in planes:
            raise ValueError(f'Duplicate or undeclared result at {key}')
        if not np.isclose(row['z_mm'],planes[key[1]]['z_mm'],rtol=0.,atol=1e-9):
            raise ValueError(f'Observation Z differs from protocol at {key}')
        if not np.isclose(row['tip_time_s'],positions[key[0]]['tip_time_s'],rtol=0.,atol=1e-14):
            raise ValueError(f'Emission epoch differs from protocol at {key}')
        metric = row['metrics']
        centre = np.asarray(metric['centroid_xy_m'],dtype=float)
        if centre.shape != (2,) or not np.isfinite(centre).all():
            raise ValueError(f'Invalid native centroid at {key}')
        for name in ('radial_rms_m','total_probability','reference_current_a'):
            value = metric[name]
            if isinstance(value,bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid {name} at {key}')
        if metric['total_probability'] > 1+1e-8 or metric['coordinate_frame'] != FRAME:
            raise ValueError(f'Invalid source normalization or coordinate frame at {key}')
        if wave:
            if row['instrument_digest'] != protocol['input_identity']:
                raise ValueError(f'Wave plane belongs to different physical inputs at {key}')
            request_identity = protocol.get('request_identity')
            if request_identity is not None and row.get('request_identity') != request_identity:
                raise ValueError(f'Wave plane belongs to different numerical request at {key}')
            for radius in ('r50_nm','r80_nm','r95_nm'):
                value = metric[radius]
                if value is None or not np.isfinite(value) or value <= 0:
                    raise ValueError(f'Invalid native containment radius at {key}')
            edge = metric['edge_1pct_fraction']
            if edge is None or not np.isfinite(edge) or not 0 <= edge <= 1+1e-12:
                raise ValueError(f'Invalid native-grid edge diagnostic at {key}')
        result[key] = metric
    return result


def scope_details(protocol):
    return dict(coordinate_frame=FRAME,
        source_comparison='Same captured Tip boundary and energy law; geometric rays omit quantum diffraction, near fields and reflection. Not identical quantum phase-space distributions.',
        measured_signal='Vacuum direct-beam intensity; no CIF or Bragg-spot stability claim.',
        scan_sampling='Five actual raster pixel centres only, not a continuous-raster proof.',
        dynamics='Wave uses per-mode coil-arrival fields. Production particles freeze at the tip epoch; the separately reported correlated first-order timing estimate must pass.',
        focus_scope='Conservative local three-plane check: sample RMS must be strictly smaller than both neighbours at +/-20 nm, and the positive-curvature quadratic minimum must lie within the declared bracket. For an exact quadratic this neighbour check restricts the minimum to +/-10 nm; not a full focus search, aberration or image-resolution qualification.',
        mode_omission_scope='The omitted independent-mode probability bounds integrated intensity L1 only. It does not by itself bound centroids, RMS radii or local relative intensity, which can be sensitive to distant weak tails.',
        numerics='Source-mode, energy, spatial-grid and column-step convergence remain separate from primary observable checks.',
        requested=protocol['scope'])


def focus_metrics(metrics, planes):
    z = np.asarray([p['z_mm'] for p in planes[:3]])
    relative = (z-z[1])*1e6
    squared = np.asarray([metrics[p['label']]['radial_rms_m']**2*1e18 for p in planes[:3]])
    a, b, _ = np.polyfit(relative, squared, 2)
    return dict(sample_bracket_z_mm=z.tolist(), rms_squared_nm2=squared.tolist(),
        variance_curvature=float(2*a), quadratic_minimum_offset_nm=None if a <= 0 else float(-b/(2*a)),
        sample_narrower_than_neighbours=bool(squared[1] < min(squared[0], squared[2])))


def focus_admitted(focus, halfwidth_nm):
    offset = focus['quadratic_minimum_offset_nm']
    return bool(focus['sample_narrower_than_neighbours'] and offset is not None
                and np.isfinite(offset) and abs(offset) <= halfwidth_nm)


def wave_timing_report(folder, protocol, wave, rays):
    """Audit saved per-mode coil clocks using the executed ray response only.

    No field solve or propagation runs here. Captured AC controls must first
    reproduce the ray timing receipt and then each recorded wave coil kick;
    an automatic calibration mismatch is not silently accepted or recomputed.
    """
    from temsim.instrument_snapshot import InstrumentSnapshot
    from temsim.immutable_json import json_digest
    report = dict(status='UNRESOLVED', passed=False, rows=[],
        model='Saved wave per-mode axial arrival times and actual frozen coil kicks; same saved finite-coil first-order response as rays',
        scope='Maximum across executed modes, including repeated segment freeze variation. A local first-order timing estimate, not full nonlinear time-dependent wavepacket propagation or an omitted-mode moment bound.')
    try:
        snapshot = InstrumentSnapshot.from_dict(json.loads((folder/'inputs.json').read_text(encoding='utf-8')))
        if snapshot.digest != protocol['input_identity']:
            raise ValueError('Wave timing controls do not match the captured physical inputs')
        ac = snapshot.restore().ac_deflector
        if not ac.scan_enabled or ac.wobble_enabled:
            raise ValueError('This timing receipt audit requires a raster without wobble')
        response = rays['timing_response']
        z = np.asarray(response['z_mm'],float)
        upper, lower = (np.asarray(response[k],float) for k in ('upper','lower'))
        if (upper.shape != (len(z),4,2) or lower.shape != upper.shape
                or not all(np.isfinite(x).all() for x in (z,upper,lower))):
            raise ValueError('Invalid saved finite-coil timing response')
        positions = {p['position']:p for p in rays['positions']}
        frame, nlines = float(ac.scan_frame_period_s), int(ac.scan_lines)
        coil_z = (float(ac.upper_z_mm),float(ac.lower_z_mm))

        def command(time):
            return np.asarray(ac.coil_kicks_mrad(*ac.instantaneous_kick_mrad(float(time))),float)*1e-3

        def line(time):
            return int((float(time)/frame % 1.)*nlines)

        nominal = {}
        for name, position in positions.items():
            epoch = float(position['tip_time_s'])
            nominal[name] = command(epoch)
            for ci, coil in enumerate(position['timing']['coils']):
                if not np.isclose(coil['z_mm'],coil_z[ci],rtol=0.,atol=1e-9):
                    raise ValueError('Saved ray timing coil positions differ from captured AC')
                times = (coil['minimum_arrival_time_s'],coil['maximum_arrival_time_s'])
                if any(line(t) != line(epoch) for t in times):
                    raise ValueError('Ray timing interval crosses a raster flyback')
                expected = np.max([abs(command(t)[ci]-nominal[name][ci]) for t in times],axis=0)
                if not np.allclose(expected,coil['maximum_command_difference_rad'],rtol=1e-6,atol=1e-18):
                    raise ValueError('Captured AC commands do not reproduce the executed ray calibration')
        for row in wave['results']:
            name, plane = row['position'], row['plane']
            epoch = float(row['tip_time_s'])
            ray_timing = positions[name]['timing']
            limit = float(ray_timing['limit_nm'])
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError('Invalid saved timing-error limit')
            receipt = json.loads((folder/(name+'_'+plane+'_receipt.json')).read_text(encoding='utf-8'))
            storage = receipt.get('storage')
            if storage is None or row['checkpoint_digest'] != json_digest(dict(record=receipt,
                    stored_payload=storage['manifest_digest'],plane_z_mm=row['z_mm'],
                    current_a=row['metrics']['reference_current_a'])):
                raise ValueError('Wave timing receipt is not bound to its executed checkpoint')
            mode_ids = [m['mode_id'] for m in row['metrics']['modes']]
            actions = {mode_id:[[],[]] for mode_id in mode_ids}
            current = receipt
            while isinstance(current,dict):
                for mode in current.get('modes',[]):
                    if mode.get('mode_id') not in actions:
                        continue
                    for action in mode.get('deflector_actions',[]):
                        if not action.get('dynamic') or ac.key not in action.get('drive_keys',[]):
                            continue
                        match = [ci for ci,value in enumerate(coil_z) if np.isclose(
                            action.get('arrival_z_mm',float('nan')),value,rtol=0.,atol=1e-9)]
                        if len(match) != 1:
                            raise ValueError('Recorded wave AC action has no matching physical coil')
                        actions[mode['mode_id']][match[0]].append(action)
                current = current.get('upstream')
            zi = np.flatnonzero(np.isclose(z,row['z_mm'],rtol=0.,atol=1e-9))
            if len(zi) != 1:
                raise ValueError('Missing exact observation plane in saved timing response')
            matrices = upper[zi[0],:2],lower[zi[0],:2]
            mode_rows = []
            for mode_id, coils in actions.items():
                if not all(coils):
                    raise ValueError('Every observed wave mode must retain both AC coil actions')
                deltas, variations, coil_rows = [],[],[]
                same_line = True
                for ci, entries in enumerate(coils):
                    times = np.asarray([entry['arrival_time_s'] for entry in entries],float)
                    actual = np.asarray([entry['kick_rad'] for entry in entries],float)
                    if not np.isfinite(times).all() or actual.shape != (len(entries),2) or not np.isfinite(actual).all():
                        raise ValueError('Invalid executed wave coil clock or kick')
                    predicted = np.asarray([command(t)[ci] for t in times])
                    if not np.allclose(actual,predicted,rtol=1e-11,atol=1e-15):
                        raise ValueError('Recorded wave kick differs from the verified captured AC time law')
                    difference = actual-nominal[name][ci]
                    deltas.append(difference[0])
                    variations.append(float(np.max(np.linalg.norm((difference-difference[0])@matrices[ci].T,axis=1))))
                    same_line &= all(line(t) == line(epoch) for t in times)
                    ray_coil = ray_timing['coils'][ci]
                    coil_rows.append(dict(z_mm=coil_z[ci],minimum_arrival_time_s=float(times.min()),
                        maximum_arrival_time_s=float(times.max()),recorded_segment_actions=len(entries),
                        ray_arrival_interval_contains_wave=bool(times.min() >= ray_coil['minimum_arrival_time_s']
                            and times.max() <= ray_coil['maximum_arrival_time_s'])))
                correlated = matrices[0]@deltas[0]+matrices[1]@deltas[1]
                estimate = float((np.linalg.norm(correlated)+sum(variations))*1e9)
                mode_rows.append(dict(mode_id=mode_id,coils=coil_rows,raster_line_unchanged=bool(same_line),
                    correlated_local_estimate_nm=estimate,segment_freeze_variation_bound_nm=float(sum(variations)*1e9),
                    passed=bool(same_line and estimate <= limit)))
            if not mode_rows:
                raise ValueError('Wave timing audit requires executed mode identities')
            report['rows'].append(dict(position=name,plane=plane,tip_time_s=epoch,limit_nm=limit,
                modes=mode_rows,maximum_correlated_local_estimate_nm=max(m['correlated_local_estimate_nm'] for m in mode_rows),
                passed=all(m['passed'] for m in mode_rows)))
        report['passed'] = bool(report['rows']) and all(r['passed'] for r in report['rows'])
        report['status'] = 'PASSED_FIRST_ORDER_WAVE_TIMING' if report['passed'] else 'FAILED_FIRST_ORDER_WAVE_TIMING'
    except (KeyError, TypeError, ValueError, OSError) as error:
        report['reason'] = str(error)
    return report


def refinement_report(folder, strict_folder, protocol, baseline, positions=None):
    """Bounded centre/corner refinement, never an implicit all-raster claim."""
    from temsim.instrument_snapshot import InstrumentSnapshot
    from temsim.immutable_json import json_digest, thaw_json
    from validate_ac_wave_scan import assert_same_physical_graph
    read=lambda base,name:json.loads((base/name).read_text(encoding='utf-8'))
    if folder.resolve() == strict_folder.resolve():
        raise ValueError('Refinement requires an independent result directory')
    original=InstrumentSnapshot.from_dict(read(folder,'inputs.json'))
    strict=InstrumentSnapshot.from_dict(read(strict_folder,'inputs.json'))
    assert_same_physical_graph(original.graph,strict.graph)
    if (thaw_json(original.implementation) != thaw_json(strict.implementation)
            or thaw_json(original.external_inputs) != thaw_json(strict.external_inputs)):
        raise ValueError('Numerical refinement must preserve implementation and external physical inputs')
    sp=read(strict_folder,'protocol.json')
    sw=read(strict_folder,'wave.json')
    if (sp['input_identity'] != strict.digest or sw['input_identity'] != strict.digest
            or protocol['input_identity'] != original.digest):
        raise ValueError('Refinement input provenance mismatch')
    if sp['positions'] != protocol['positions'] or sp['planes'] != protocol['planes']:
        raise ValueError('Refinement must retain scan epochs and observation planes')
    base_request=read(folder,'request.json')
    strict_request=read(strict_folder,'request.json')
    if (sp.get('request_identity') != json_digest(strict_request)
            or sw.get('request_identity') != sp['request_identity']
            or protocol.get('request_identity') != json_digest(base_request)):
        raise ValueError('Refinement numerical request provenance mismatch')
    # This entry point checks column step/support only. It deliberately does
    # not silently combine changes to source sampling, gun or readout models.
    changes=[]
    for group,key in ((None,'column_step_mm'),
            ('wave_grid','maximum_support_loss_fraction'),
            ('wave_grid','maximum_spectral_loss_fraction'),
            ('wave_grid','maximum_incoherent_mode_loss_probability'),
            ('wave_grid','maximum_accumulated_support_l2')):
        left=base_request if group is None else base_request[group]
        right=strict_request if group is None else strict_request[group]
        before,after=left.get(key,0.),right.get(key,0.)
        if not np.isfinite((before,after)).all() or after > before:
            raise ValueError(f'Refinement loosens {key}')
        if after < before:
            changes.append(dict(parameter=key,baseline=before,refined=after))
        left.pop(key,None); right.pop(key,None)
    for request in (base_request,strict_request):
        request.pop('execution',None)
        for key in ('maximum_working_bytes','maximum_device_working_bytes','maximum_pixels'):
            request['wave_grid'].pop(key,None)
    if base_request != strict_request or not changes:
        raise ValueError('Refinement must tighten column step/support while retaining all other calculation methods')
    sm=validated_rows(sw,sp,wave=True)
    names=positions if positions is not None else sorted({name for name,_ in sm})
    known={p['label'] for p in protocol['positions']}
    if len(set(names)) != len(names) or not set(names) <= known:
        raise ValueError('Refinement positions must be unique declared scan labels')
    report=dict(directory=str(strict_folder.resolve()),input_identity=strict.digest,
        request_identity=sp['request_identity'],changes=changes,positions=names,
        scope='Only the listed centre/corner observations at fixed source/energy quadrature; no claim of full-raster or source/gun convergence.',
        checks={},rows=[],focus=[],status='INCOMPLETE',passed=False)
    required={(name,p['label']) for name in names for p in protocol['planes']}
    if ('centre' not in names or not (set(names)-{'centre'}) or not required <= sm.keys()
            or sw['status'] != 'COMPLETED_REQUESTED_POSITIONS'):
        report['reason']='A completed centre and at least one completed corner are required, each at all four declared planes'
        report['missing_planes']=[dict(position=n,plane=p) for n,p in sorted(required-sm.keys())]
        return report
    limits=protocol['limits']
    for name,plane in sorted(required):
        base,refined=baseline[name,plane],sm[name,plane]
        if not np.isclose(base['reference_current_a'],refined['reference_current_a'],rtol=1e-10,atol=1e-24):
            raise ValueError('Refinement changes the current reference')
        displacement=float(np.linalg.norm(np.asarray(base['centroid_xy_m'])-refined['centroid_xy_m'])*1e9)
        widths={key:abs(refined[key]/base[key]-1.) for key in ('radial_rms_m','r50_nm','r80_nm','r95_nm')}
        centroid_limit=limits['refinement_diffraction_centroid_nm' if plane=='diffraction' else 'refinement_sample_centroid_nm']
        report['rows'].append(dict(position=name,plane=plane,centroid_difference_nm=displacement,
            centroid_limit_nm=centroid_limit,width_change_fractions=widths,
            baseline_probability=base['total_probability'],refined_probability=refined['total_probability'],
            probability_difference=refined['total_probability']-base['total_probability'],
            baseline_edge_fraction=base['edge_1pct_fraction'],refined_edge_fraction=refined['edge_1pct_fraction']))
    for name in names:
        base_focus = focus_metrics({p['label']:baseline[name,p['label']] for p in protocol['planes']},protocol['planes'])
        strict_focus = focus_metrics({p['label']:sm[name,p['label']] for p in protocol['planes']},protocol['planes'])
        old, new = base_focus['quadratic_minimum_offset_nm'], strict_focus['quadratic_minimum_offset_nm']
        report['focus'].append(dict(position=name,baseline=base_focus,refined=strict_focus,
            quadratic_minimum_difference_nm=None if old is None or new is None else float(new-old),
            baseline_admitted=focus_admitted(base_focus,limits['focus_bracket_halfwidth_nm']),
            refined_admitted=focus_admitted(strict_focus,limits['focus_bracket_halfwidth_nm'])))
    checks=dict(
        centres=all(r['centroid_difference_nm']<=r['centroid_limit_nm'] for r in report['rows']),
        widths=all(max(r['width_change_fractions'].values())<=limits['refinement_width_fraction'] for r in report['rows']),
        local_three_plane_focus=all(r['baseline_admitted'] and r['refined_admitted'] for r in report['focus']),
        gpu_execution=all(any(b.get('compute_backend')=='cupy' for b in row.get('backends',[]))
            for row in sw['results'] if (row['position'],row['plane']) in required))
    report.update(checks=checks,passed=all(checks.values()),status='PASSED_BOUNDED_REFINEMENT' if all(checks.values()) else 'FAILED_BOUNDED_REFINEMENT')
    return report


def incomplete_report(folder, protocol, wave, rays, rm):
    """A failed upstream wave is never replaced by a particle-derived image."""
    rows = []
    for item in protocol['positions']:
        name = item['label']
        row = dict(position=name)
        for plane in ('sample', 'diffraction'):
            m = rm[name, plane]
            displacement = (np.asarray(m['centroid_xy_m'])-rm['centre', plane]['centroid_xy_m'])*1e9
            row[plane] = dict(ray_displacement_nm=displacement.tolist(),
                ray_centroid_xy_nm=(np.asarray(m['centroid_xy_m'])*1e9).tolist(),
                ray_motion_nm=float(np.linalg.norm(displacement)), ray_rms_nm=m['radial_rms_m']*1e9,
                ray_probability=m['total_probability'])
        expected = (np.asarray(item['factors'])-protocol['positions'][0]['factors'])*500.
        row['sample']['ray_target_error_nm'] = float(np.linalg.norm(expected-row['sample']['ray_displacement_nm']))
        row['ray_focus'] = focus_metrics({p['label']:rm[name,p['label']] for p in protocol['planes']}, protocol['planes'])
        rows.append(row)
    report = dict(schema='ac-wave-ray-comparison-v1', status='WAVE_NOT_VALIDATED',
        input_identity=protocol['input_identity'], rows=rows, checks={}, all_primary_checks_passed=False,
        validation_complete=False, numerical_qualification_status='NOT_ESTABLISHED',
        wave_status=wave['status'], wave_error=wave.get('error'),
        completed_wave_planes=[dict(position=r['position'],plane=r['plane']) for r in wave['results']],
        timing_qualification=rays['timing_qualification'], timing_positions=rays['positions'],
        refinement_status='NOT_RUN_UPSTREAM_WAVE_FAILED', scope=protocol['scope'], scope_details=scope_details(protocol),
        source_change=protocol['source_change'],
        note='Particle-only results below cannot validate coherent focus, diffraction stationarity or wave/ray agreement.')
    (folder/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14,4.5), constrained_layout=True)
    fig.suptitle('Executed GPU rays only; coherent comparison NOT VALIDATED (upstream wave failure)', fontsize=12)
    for row in rows:
        sx,sy = row['sample']['ray_displacement_nm']
        dx,dy = row['diffraction']['ray_displacement_nm']
        axes[0].scatter(sx,sy)
        axes[2].scatter(dx,dy,label=row['position'])
        f = row['ray_focus']
        z = np.asarray(f['sample_bracket_z_mm'])
        axes[1].plot((z-z[1])*1e6,np.sqrt(f['rms_squared_nm2']),marker='o')
    axes[0].set(xlabel='Sample delta X (nm)',ylabel='Sample delta Y (nm)',title='1 um FOV; 64 x 64 pixel centres',aspect='equal')
    axes[1].set(xlabel='Distance from sample entrance (nm)',ylabel='RMS radius (nm)',title='Sample focus retained')
    diffraction_z = next(p['z_mm'] for p in protocol['planes'] if p['label']=='diffraction')
    axes[2].set(xlabel='Diffraction delta X (nm)',ylabel='Diffraction delta Y (nm)',title=f'Z = {diffraction_z:.6f} mm; Descan off',aspect='equal')
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(folder/'comparison.png',dpi=170)
    plt.close(fig)
    print(json.dumps(dict(status=report['status'],rows=rows),indent=2))


def compare(folder, *, refinement_directory=None, refinement_positions=None):
    protocol=json.loads((folder/'protocol.json').read_text())
    wave=json.loads((folder/'wave.json').read_text())
    rays=json.loads((folder/'rays.json').read_text())
    names=[item['label'] for item in protocol['positions']]
    if wave['input_identity'] != protocol['input_identity'] or rays['input_identity'] != protocol['input_identity']:
        raise ValueError('Wave, particle and protocol input identities must match')
    if protocol.get('request_identity') is not None and wave.get('request_identity') != protocol['request_identity']:
        raise ValueError('Wave and protocol numerical request identities must match')
    wm=validated_rows(wave,protocol,wave=True)
    rm=validated_rows(rays,protocol,wave=False)
    required={(n,p['label']) for n in names for p in protocol['planes']}
    if not required <= rm.keys():
        raise ValueError('Particle results do not cover the declared comparison')
    if not required <= wm.keys():
        incomplete_report(folder,protocol,wave,rays,rm)
        return
    for key in required:
        if not np.isclose(wm[key]['reference_current_a'],rm[key]['reference_current_a'],rtol=1e-10,atol=1e-24):
            raise ValueError(f'Wave and particle source-current references differ at {key}')
    limit=protocol['limits']
    rows=[]
    for item in protocol['positions']:
        name=item['label']
        row=dict(position=name)
        for plane in ('sample','diffraction'):
            w,r=wm[name,plane],rm[name,plane]
            wd=(np.asarray(w['centroid_xy_m'])-wm['centre',plane]['centroid_xy_m'])*1e9
            rd=(np.asarray(r['centroid_xy_m'])-rm['centre',plane]['centroid_xy_m'])*1e9
            row[plane]=dict(wave_displacement_nm=wd.tolist(),ray_displacement_nm=rd.tolist(),
                wave_centroid_xy_nm=(np.asarray(w['centroid_xy_m'])*1e9).tolist(),
                ray_centroid_xy_nm=(np.asarray(r['centroid_xy_m'])*1e9).tolist(),
                absolute_centroid_difference_nm=float(np.linalg.norm(np.asarray(w['centroid_xy_m'])-r['centroid_xy_m'])*1e9),
                displacement_difference_nm=float(np.linalg.norm(wd-rd)),
                wave_motion_nm=float(np.linalg.norm(wd)),ray_motion_nm=float(np.linalg.norm(rd)),
                wave_rms_nm=w['radial_rms_m']*1e9,ray_rms_nm=r['radial_rms_m']*1e9,
                wave_r50_nm=w['r50_nm'],wave_r80_nm=w['r80_nm'],wave_r95_nm=w['r95_nm'],
                wave_probability=w['total_probability'],ray_probability=r['total_probability'],
                wave_numerical_support=w.get('support_approximation'),
                wave_incoherent_mode_approximation=w.get('incoherent_mode_approximation'),
                wave_aperture_conditioned_approximation=w.get('aperture_conditioned_approximation', []),
                wave_edge_1pct_fraction=w['edge_1pct_fraction'])
        expected=(np.asarray(item['factors'])-protocol['positions'][0]['factors'])*500.
        row['sample']['expected_displacement_nm']=expected.tolist()
        row['sample']['wave_target_error_nm']=float(np.linalg.norm(expected-row['sample']['wave_displacement_nm']))
        row['sample']['ray_target_error_nm']=float(np.linalg.norm(expected-row['sample']['ray_displacement_nm']))
        for radius in ('r50','r80'):
            row['sample'][radius+'_change_fraction']=abs(wm[name,'sample'][radius+'_nm']/wm['centre','sample'][radius+'_nm']-1.)
        row['focus']=focus_metrics({p['label']:wm[name,p['label']] for p in protocol['planes']},protocol['planes'])
        row['ray_focus']=focus_metrics({p['label']:rm[name,p['label']] for p in protocol['planes']},protocol['planes'])
        rows.append(row)
    wave_timing = wave_timing_report(folder,protocol,wave,rays)
    checks=dict(
        executions_completed=wave['status']=='COMPLETED_REQUESTED_POSITIONS' and rays['status']=='complete',
        timing_approximation_admitted=(rays.get('timing_admitted') is True
            and len(rays.get('positions',[])) == len(names)
            and all(p['timing']['passed'] is True for p in rays['positions'])
            and all(r.get('timing_admitted') is True for r in rays['results'])),
        wave_timing_approximation_admitted=wave_timing['passed'],
        column_wave_gpu_executed=all(any(b.get('compute_backend')=='cupy' for b in r.get('backends',[])) for r in wave['results']),
        empirical_aperture_accuracy_admitted=all(
            stage['wall_estimate']['within_declared_accuracy'] is True
            and stage['maximum_guard_complex_change'] <= stage['numerics']['maximum_relative_complex_change']
            and stage['maximum_guard_current_change'] <= stage['numerics']['maximum_relative_current_change']
            for metric in wm.values() for stage in metric.get('aperture_conditioned_approximation', [])),
        sample_wave_target=all(r['sample']['wave_target_error_nm']<=limit['sample_position_error_nm'] for r in rows),
        sample_ray_wave_scan=all(r['sample']['displacement_difference_nm']<=limit['sample_ray_wave_displacement_difference_nm'] for r in rows),
        diffraction_wave_stationarity=all(r['diffraction']['wave_motion_nm']<=limit['diffraction_motion_nm'] for r in rows),
        diffraction_ray_wave_scan=all(r['diffraction']['displacement_difference_nm']<=limit['diffraction_ray_wave_displacement_difference_nm'] for r in rows),
        scan_width_r50=all(r['sample']['r50_change_fraction']<=limit['scan_containment_width_fraction'] for r in rows),
        scan_width_r80=all(r['sample']['r80_change_fraction']<=limit['scan_containment_width_fraction'] for r in rows),
        sample_local_focus=all(focus_admitted(r['focus'],limit['focus_bracket_halfwidth_nm']) for r in rows),
        ray_sample_local_focus=all(focus_admitted(r['ray_focus'],limit['focus_bracket_halfwidth_nm']) for r in rows))
    report=dict(schema='ac-wave-ray-comparison-v1',input_identity=protocol['input_identity'],rows=rows,
        request_identity=protocol.get('request_identity'),
        status='PRIMARY_CHECKS_PASSED_REFINEMENT_PENDING' if all(checks.values()) else 'PRIMARY_CHECKS_FAILED',
        checks=checks,all_primary_checks_passed=all(checks.values()),
        validation_complete=False, numerical_qualification_status='NOT_ESTABLISHED',
        maximum_wave_edge_1pct_fraction=max(m['edge_1pct_fraction'] for m in wm.values()),
        edge_diagnostic_note='Recorded diagnostic only; no independent domain-size convergence established.',
        refinement_status='NOT_RUN',scope=protocol['scope'],scope_details=scope_details(protocol),
        timing_qualification=rays.get('timing_qualification'),timing_positions=rays.get('positions'),
        wave_timing_qualification=wave_timing,
        source_change=protocol['source_change'],
        note='Same-source geometric rays need not have the same absolute width or transmission as waves with diffraction/reflection. Compare scan-induced displacement and focus separately.')
    if refinement_directory is not None:
        refinement=refinement_report(folder,refinement_directory,protocol,wm,refinement_positions)
        report.update(refinement=refinement,refinement_status=refinement['status'],
            bounded_validation_complete=all(checks.values()) and refinement['passed'],
            numerical_qualification_status=refinement['status'])
        if all(checks.values()):
            report['status']='PRIMARY_AND_BOUNDED_REFINEMENT_PASSED' if refinement['passed'] else 'PRIMARY_CHECKS_PASSED_REFINEMENT_NOT_PASSED'
    else:
        report['bounded_validation_complete']=False
    (folder/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for key,marker in [('wave','o'),('ray','x')]:
        xy=np.asarray([r['sample'][key+'_displacement_nm'] for r in rows])
        axes[0].scatter(xy[:,0],xy[:,1],marker=marker,label=key,s=55)
        xy=np.asarray([r['diffraction'][key+'_displacement_nm'] for r in rows])
        axes[2].plot(xy[:,0],xy[:,1],marker,label=key)
    axes[0].set(xlabel='Sample delta X (nm)',ylabel='Sample delta Y (nm)',title='Scan relative to reference pixel',aspect='equal')
    axes[2].set(xlabel='Diffraction delta X (nm)',ylabel='Diffraction delta Y (nm)',title='Direct-beam centroid; Descan off')
    for radius in ('rms','r50','r80'):
        axes[1].plot(names,[r['sample']['wave_'+radius+'_nm'] for r in rows],marker='o',label='wave '+radius)
    axes[1].plot(names,[r['sample']['ray_rms_nm'] for r in rows],marker='x',label='ray RMS')
    axes[1].set(ylabel='Radius (nm)',title='Sample widths; native physical grids')
    axes[1].tick_params(axis='x',rotation=30)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.savefig(folder/'comparison.png',dpi=170)
    plt.close(fig)
    print(json.dumps(dict(checks=checks,rows=rows),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/ac_wave_scan_20261010'))
    parser.add_argument('--refinement-directory',type=Path,
        help='Independent stricter column-step/support run; centre plus one corner is sufficient')
    parser.add_argument('--refinement-positions',nargs='+',
        help='Specific completed states to compare; default uses all states present in the refinement result')
    args=parser.parse_args()
    if args.refinement_positions is not None and args.refinement_directory is None:
        parser.error('--refinement-positions requires --refinement-directory')
    compare(args.output,refinement_directory=args.refinement_directory,refinement_positions=args.refinement_positions)
