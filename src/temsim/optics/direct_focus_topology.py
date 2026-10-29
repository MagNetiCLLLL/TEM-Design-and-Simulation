"""Qualification-only evidence for direct Mini Condenser/objective focusing.

No propagation, lens adjustment, automatic refocus or source modification is
performed here. A passing absence check does not establish specimen focus.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
import numpy as np

REQUIREMENT = 'direct_focus_without_extra_pre_specimen_waist'
SURFACE_FOCUS_TOLERANCE_NM = 1.0
MAXIMUM_SPACING_MM = .5
KEYS = ('x_m', 'y_m', 'tx_rad', 'ty_rad')


def required_for_state(state):
    from temsim.operating_modes import condenser_mode_key, is_nanoprobe_mode
    return (getattr(state, 'simulation_mode', None) == 'unified'
            and is_nanoprobe_mode(condenser_mode_key(state)))


def bounds_for_state(state):
    mini = state.mini_condenser
    if not mini.enabled or not getattr(mini, 'installed', True):
        raise ValueError('Direct-focus qualification requires an installed Mini Condenser')
    start = float(mini.mechanical_center_from_tip_mm-.5*mini.mechanical_length_mm)
    surface = float(state.sample.upper_surface_z_mm)
    if not np.isfinite([start, surface]).all() or not float(state.electron_gun.exit_plane_z_mm) < start < surface:
        raise ValueError('Invalid direct-focus physical interval')
    return start, surface


def requested_capture_planes(state, *, maximum_spacing_mm=MAXIMUM_SPACING_MM):
    if not required_for_state(state):
        return ()
    start, surface = bounds_for_state(state)
    spacing = min(float(maximum_spacing_mm), MAXIMUM_SPACING_MM)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('Direct-focus spacing must be finite and positive')
    guard = min(spacing/2, (start-state.electron_gun.exit_plane_z_mm)/2)
    count = math.ceil((surface-start)/spacing)
    if count > 20000:
        raise ValueError('Direct-focus observation budget exceeded')
    offsets_nm = np.array([200., 100., 50., 20., 10., 5., 2., 1., .5, .2, .1, 0.])
    return tuple(map(float, np.unique(np.r_[start-guard,
        np.linspace(start, surface, count+1), surface-offsets_nm*1e-6])))


def evaluate(checkpoints, masks, weights, *, lower_z_mm, surface_z_mm,
             surface_focus_tolerance_nm, maximum_spacing_mm=MAXIMUM_SPACING_MM,
             waist_detector='legacy_point',source_ids=None):
    """Fail closed on missing coverage or unresolved potential upstream waists.

    Terminal classification requires the entire executed covariance bracket to
    fit inside the explicitly declared focus window. An interpolated point
    near the endpoint cannot hide a wide unresolved bracket. Intermediate
    directional splitting is not evaluated as a separate hard condition.
    """
    report = dict(schema='direct-focus-topology-v1', requirement=REQUIREMENT,
        status='UNRESOLVED', passed=False, failures=[], refinement_planes_mm=[],
        executed_waists=[], radial_candidates=[], terminal_candidates=[],
        surface_focus_qualified=False,
        scope='Absence of extra waists at bounded sampling; independent specimen focus, size and numerical convergence remain required')

    def reject(reason, message, *, definite=False):
        report.update(status='FAIL' if definite else 'UNRESOLVED', passed=False, message=message)
        report['failures'].append(reason)
        return report

    try:
        from temsim.optics.waist_evidence import FINITE, validate_detector, finite_interval
        validate_detector(waist_detector)
        report['waist_detector']=waist_detector
        report['source_identity_basis']='original_source_ids' if source_ids is not None else 'checkpoint_column_indices'
        if waist_detector==FINITE:
            report['schema']='direct-focus-finite-region-topology-v2'
        start, end, tolerance, spacing = map(float, (lower_z_mm, surface_z_mm,
            surface_focus_tolerance_nm, maximum_spacing_mm))
        if (not np.isfinite([start,end,tolerance,spacing]).all() or start >= end
                or tolerance <= 0 or not 0 < spacing <= MAXIMUM_SPACING_MM):
            return reject('invalid_direct_focus_limits', 'Finite physical bounds and explicit positive focus tolerance are required')
        terminal_start = end-tolerance*1e-6
        if terminal_start <= start:
            return reject('invalid_terminal_focus_window', 'Terminal focus tolerance cannot cover the full optical interval')
        z = np.asarray(checkpoints.z_mm, float)
        arrays = [np.asarray(getattr(checkpoints,key),float) for key in KEYS]
        alive, weight = np.asarray(masks), np.asarray(weights,float)
        identities=np.arange(weight.size) if source_ids is None else np.asarray(source_ids)
        if (z.ndim != 1 or z.size < 3 or np.any(~np.isfinite(z)) or np.any(np.diff(z)<=0)
                or weight.ndim != 1 or np.any(~np.isfinite(weight)) or np.any(weight<0)
                or not np.isfinite(weight.sum()) or weight.sum() <= 0
                or alive.dtype.kind != 'b' or alive.shape != (z.size,weight.size)
                or any(a.shape != alive.shape for a in arrays)
                or identities.shape!=weight.shape or identities.dtype.kind not in 'iu'
                or len(np.unique(identities))!=len(identities)):
            return reject('invalid_direct_focus_arrays', 'Invalid executed checkpoints or original weights')
        report.update(bounds_mm=[start,end], surface_focus_tolerance_nm=tolerance,
                      terminal_window_mm=[terminal_start,end], requested_maximum_spacing_mm=spacing)
        if (z[0] >= start or z[-1] < end or
                any(not np.any(np.isclose(z,p,atol=1e-9,rtol=0)) for p in (start,terminal_start,end))):
            return reject('missing_direct_focus_coverage', 'Missing Mini entrance guard, physical boundary or terminal-window checkpoints')
        selected = np.flatnonzero((z>=z[max(0,np.searchsorted(z,start)-1)]) & (z<=end+1e-9))
        local_z = z[selected]
        if np.max(np.diff(local_z)) > spacing+1e-9:
            return reject('sparse_direct_focus_coverage', 'Direct-focus checkpoint spacing exceeds its declared bound')
        positive = weight>0
        live = alive[selected] & positive[None,:]
        if np.any(~live[:-1]&live[1:]):
            return reject('direct_focus_particle_reappears', 'A stopped positive-current particle reappears')
        if any(np.any(~np.isfinite(a[selected][live])) for a in arrays):
            return reject('nonfinite_direct_focus_phase', 'Nonfinite current-carrying phase coordinates')
        if np.any(np.sum(live,axis=1)<5):
            return reject('insufficient_direct_focus_support', 'Fewer than five positive-current particles in the direct-focus interval')
        local = SimpleNamespace(z_mm=local_z, **dict(zip(KEYS,[a[selected] for a in arrays])))
        from temsim.optics.beam_path_audit import crossover_candidates
        from temsim.optics.crossover_acceptance import _strict_waists
        changes = np.flatnonzero(np.any(live[1:]!=live[:-1],axis=1))+1
        cuts = np.r_[0,changes,len(local_z)]
        measured = []
        for low,high in zip(cuts[:-1],cuts[1:]):
            cohort=live[low]
            for row in _strict_waists(local,cohort,weight,np.arange(low,high)):
                if start <= row['z_mm'] < end:
                    measured.append(dict(row, original_column_indices=np.flatnonzero(cohort).tolist(),
                        source_current_fraction=float(weight[cohort].sum()/weight.sum())))
        roots = [row for row in crossover_candidates(local,live,weight) if start <= row['z_mm'] < end]
        if waist_detector==FINITE:
            report['legacy_point_executed_waists']=measured
            finite=finite_interval(checkpoints,alive,weight,start,end,maximum_spacing_mm=spacing,source_ids=source_ids)
            report['finite_region_evidence']=finite
            measured=finite['regions']
            # Endpoint focus is a separate one-sided observation. A terminal
            # stationary plane need not have a two-sided region past the sample.
            # Raw radial brackets and interception still fail closed below.
        report.update(executed_waists=measured, radial_candidates=roots,
            maximum_observation_gap_mm=float(np.max(np.diff(local_z))),
            population_change_intervals_mm=[[float(local_z[i-1]),float(local_z[i])] for i in changes])
        forbidden = [row for row in measured if row['bracket_mm'][1] < terminal_start]
        if forbidden:
            report['forbidden_waists']=forbidden
            return reject('executed_pre_specimen_waist_outside_focus_window',
                'An executed two-dimensional waist precedes the allowed final focus window', definite=True)
        # A narrow radial root does not bound the full two-dimensional waist.
        # An overlapping finite region needs refinement unless both executed
        # sides fit the terminal window; it is not a definite extra waist.
        unresolved = ([row for row in measured
            if not terminal_start <= row['bracket_mm'][0] <= row['bracket_mm'][1] <= end]
            if waist_detector == FINITE else [])
        for row in roots:
            left,right=map(float,row['bracket_mm'])
            if left >= terminal_start and right <= end:
                report['terminal_candidates'].append(row)
            else:
                unresolved.append(row)
        for row in unresolved:
            left,right=map(float,row['bracket_mm'])
            root=float(row['z_mm'])
            delta=min(.01,(right-left)/8,(root-start)/4,(end-root)/4)
            if delta>0:
                report['refinement_planes_mm'].extend([root-delta,root,root+delta])
        report['unresolved_candidates']=unresolved
        report['refinement_planes_mm']=sorted(set(report['refinement_planes_mm']))
        if len(changes) or unresolved:
            return reject('unresolved_direct_focus_population_or_waist',
                'A population transition or nonterminal covariance bracket requires independent refinement')
        report.update(status='PASS',passed=True,
            message='No extra pre-specimen source waist detected outside the explicit final-focus window at bounded sampling')
        return report
    except (AttributeError,TypeError,ValueError,FloatingPointError) as error:
        return reject('invalid_direct_focus_evidence',str(error))


def evaluate_for_state(state, checkpoints, masks, weights, *, maximum_spacing_mm=MAXIMUM_SPACING_MM,
                       waist_detector=None,source_ids=None):
    from temsim.optics.waist_evidence import detector_for_state
    start,end=bounds_for_state(state)
    return evaluate(checkpoints,masks,weights,lower_z_mm=start,surface_z_mm=end,
                    surface_focus_tolerance_nm=SURFACE_FOCUS_TOLERANCE_NM,
                    maximum_spacing_mm=min(float(maximum_spacing_mm),MAXIMUM_SPACING_MM),
                    waist_detector=detector_for_state(state) if waist_detector is None else waist_detector,
                    source_ids=source_ids)
