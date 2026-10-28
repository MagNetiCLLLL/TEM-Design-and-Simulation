"""Shared finite-region observation adapters; no propagation or adjustment."""
from types import SimpleNamespace
import numpy as np

LEGACY = 'legacy_point'
FINITE = 'finite_region'
KEYS = ('x_m', 'y_m', 'tx_rad', 'ty_rad')


def detector_for_state(state):
    from temsim.optics.direct_focus_topology import required_for_state
    return FINITE if required_for_state(state) else LEGACY


def validate_detector(value):
    if value not in (LEGACY, FINITE):
        raise ValueError('Unknown waist evidence detector')
    return value


def corrector_regions_for_state(state):
    from temsim.optics.crossover_acceptance import regions_for_state
    chain = regions_for_state(state)
    return {key: dict(bounds_mm=chain[key]['bounds_mm'],maximum_spacing_mm=spacing)
            for key,spacing in (('corrector_internal',1.),('ac_pair',.1))}


def requested_corrector_planes(state):
    pieces=[]
    for row in corrector_regions_for_state(state).values():
        left,right=row['bounds_mm']; count=int(np.ceil((right-left)/row['maximum_spacing_mm']))
        if count>20000:
            raise ValueError('Corrector evidence observation budget exceeded')
        pieces.append(np.linspace(left,right,count+1))
    return tuple(map(float,np.unique(np.r_[*pieces])))


def corrector_evidence_for_state(state, checkpoints, masks, weights, source_ids=None):
    regions={key:finite_interval(checkpoints,masks,weights,*row['bounds_mm'],
        maximum_spacing_mm=row['maximum_spacing_mm'],source_ids=source_ids)
        for key,row in corrector_regions_for_state(state).items()}
    passed=all(row['status']=='RESOLVED_AT_BOUNDED_SAMPLING' and len(row['regions'])==1
               for row in regions.values())
    return dict(schema='required-corrector-finite-regions-v1',passed=passed,
        status='PASS' if passed else 'UNRESOLVED',intervals=regions,
        scope='Exactly one actual-source finite region in the central TL pair and physical AC pair; independent specimen angle, focus and size remain required')


def explicit_bracket_indices(checkpoints, row):
    """Resolve the reported sides, never substitute adjacent checkpoints."""
    z = np.asarray(checkpoints.z_mm, float)
    planes = np.array([row['bracket_mm'][0], row['z_mm'], row['bracket_mm'][1]])
    indices = np.searchsorted(z, planes)
    if (np.any(indices >= len(z)) or np.any(np.diff(indices) <= 0)
            or not np.array_equal(z[indices], planes)):
        raise ValueError('Waist sides must be exact ordered executed planes')
    return indices, np.arange(indices[0], indices[-1] + 1)


def observed_aperture_losses(state, checkpoints, masks, weights, source_ids=None):
    """Attribute only losses supported by captured masks and opening geometry.

    A plane coincidence alone is insufficient. Every lost ray must be outside
    the installed opening at a finite captured endpoint. Tilted/posed openings
    without local-frame contact evidence remain unattributed.
    """
    from temsim.physics.aperture_clipping import _transmits
    z = np.asarray(checkpoints.z_mm, float)
    live = np.asarray(masks) & (np.asarray(weights) > 0)[None, :]
    ids = np.arange(live.shape[1]) if source_ids is None else np.asarray(source_ids)
    rows = []
    for i in np.flatnonzero(np.any(live[:-1] != live[1:], axis=1)) + 1:
        lost = live[i-1] & ~live[i]
        matches = []
        for aperture in getattr(state, 'apertures', ()):
            if not (aperture.enabled and getattr(aperture, 'installed', True)
                    and getattr(aperture, 'inserted', True)):
                continue
            center = float(aperture.z_mm)
            half = max(0., float(getattr(aperture, 'plate_thickness_mm', 0.))) / 2
            if center + half < z[i-1] or center - half > z[i]:
                continue
            try:
                from temsim.physics.aperture_clipping import posed_aperture_registration
                if posed_aperture_registration(state, aperture) is not None:
                    continue
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            blocked = np.zeros(int(lost.sum()), bool)
            for j in (i-1, i):
                x, y = checkpoints.x_m[j, lost]*1e3, checkpoints.y_m[j, lost]*1e3
                finite = np.isfinite(x) & np.isfinite(y)
                blocked |= finite & ~_transmits(aperture, x, y)
            if lost.any() and blocked.all():
                matches.append(str(aperture.key))
        rows.append(dict(bracket_mm=[float(z[i-1]), float(z[i])],
            lost_source_ids=ids[lost].tolist(), physical_aperture_keys=matches,
            status='CAPTURED_LOSS_SUPPORTED_BY_OPENING_GEOMETRY' if matches else 'UNATTRIBUTED_CAPTURED_LOSS'))
    return rows


def finite_interval(checkpoints, masks, weights, lower, upper, *, maximum_spacing_mm,
                    source_ids=None, known_interceptions=()):
    """Observe each actual stable cohort without merging across interception."""
    from temsim.physics.finite_waist_regions import measure_finite_waist_regions
    base = measure_finite_waist_regions(checkpoints, masks, weights, lower, upper,
        maximum_spacing_mm=maximum_spacing_mm, source_ids=source_ids)
    base['population_change_evidence'] = []
    base['stable_segments'] = []
    base['source_identity_basis']='original_source_ids' if source_ids is not None else 'checkpoint_column_indices'
    for row in base['regions']:
        row['observation_bounds_mm'] = base['bounds_mm']
    if not any(row.get('message', '').startswith('Population changes')
               for row in base['unresolved']):
        return base
    z = np.asarray(checkpoints.z_mm, float)
    selected = np.flatnonzero((z >= lower) & (z <= upper))
    live = np.asarray(masks)[selected] & (np.asarray(weights) > 0)[None, :]
    changes = np.flatnonzero(np.any(live[1:] != live[:-1], axis=1)) + 1
    cuts = np.r_[0, changes, len(selected)]
    result = dict(base, regions=[], radial_candidates=[], unresolved=[],
                  stable_segments=[], population_change_evidence=[])
    for change in changes:
        bracket = z[selected[[change-1, change]]].tolist()
        evidence = next((row for row in known_interceptions
            if row['bracket_mm'] == bracket and row.get('status') == 'CAPTURED_LOSS_SUPPORTED_BY_OPENING_GEOMETRY'), None)
        result['population_change_evidence'].append(evidence or dict(bracket_mm=bracket, status='UNATTRIBUTED_CAPTURED_LOSS'))
        if evidence is None:
            result['unresolved'].append(dict(reason='unattributed_population_change', bracket_mm=bracket))
    for a, b in zip(cuts[:-1], cuts[1:]):
        indices = selected[a:b]
        if len(indices) < 3:
            result['unresolved'].append(dict(reason='insufficient_stable_segment',
                bounds_mm=z[indices[[0, -1]]].tolist()))
            continue
        segment = measure_finite_waist_regions(checkpoints, masks, weights,
            float(z[indices[0]]), float(z[indices[-1]]),
            maximum_spacing_mm=maximum_spacing_mm, source_ids=source_ids)
        result['stable_segments'].append(segment)
        segment['source_identity_basis']=base['source_identity_basis']
        for row in segment['regions']:
            row['observation_bounds_mm'] = segment['bounds_mm']
        result['regions'].extend(segment['regions'])
        result['radial_candidates'].extend(segment['radial_candidates'])
        result['unresolved'].extend(segment['unresolved'])
    result['status'] = 'UNRESOLVED' if result['unresolved'] else 'RESOLVED_AT_BOUNDED_SAMPLING'
    return result


def support_on_bracket(checkpoints, masks, weights, row, *, maximum_spacing_mm,
                       source_ids=None):
    """Recheck a second cohort on all executed planes of the explicit bracket."""
    sides, inside = explicit_bracket_indices(checkpoints, row)
    # The observer requires interior sides. Retain the component evidence bounds
    # used by the first observation, rather than inventing endpoint coordinates.
    lower, upper = row['observation_bounds_mm']
    report = finite_interval(checkpoints, masks, weights, lower, upper,
        maximum_spacing_mm=maximum_spacing_mm, source_ids=source_ids)
    supported = []
    for other in report['regions']:
        # Both source images must occupy the same measured radial bracket.
        # Exact representative-plane equality would reintroduce grid-phase bias.
        if (other['radial_bracket_mm'][0] <= row['radial_bracket_mm'][1]
                and row['radial_bracket_mm'][0] <= other['radial_bracket_mm'][1]):
            supported.append(other)
    stable = bool(np.all(np.asarray(masks)[inside] == np.asarray(masks)[inside[0]]))
    same_bracket = False
    if stable and len(supported) == 1:
        from temsim.physics.finite_waist_regions import _transitions
        cohort = np.asarray(masks)[inside[0]] & (np.asarray(weights) > 0)
        if cohort.sum() >= 5:
            w = np.asarray(weights)[cohort]; w = w / w.sum()
            phase = np.stack([np.asarray(getattr(checkpoints,key))[np.ix_(inside,cohort)] for key in KEYS],axis=-1)
            phase = phase - np.einsum('n,kni->ki',w,phase)[:,None,:]
            r,t = phase[:,:,:2],phase[:,:,2:]
            R = np.einsum('n,kni,knj->kij',w,r,r)
            C = np.einsum('n,kni,knj->kij',w,r,t); C = (C+C.transpose(0,2,1))/2
            T = np.einsum('n,kni,knj->kij',w,t,t)
            ce = np.linalg.eigvalsh(C)
            noise = 64*np.finfo(float).eps*np.sqrt(np.trace(R,axis1=1,axis2=2)*np.trace(T,axis1=1,axis2=2))
            center = int(sides[1]-sides[0])
            growth = np.linalg.eigvalsh(R[[0,-1]]-R[center])
            growth_noise = 64*np.finfo(float).eps*np.max(np.linalg.eigvalsh(R))
            transitions = [_transitions(ce[:,axis],noise) for axis in range(2)]
            same_bracket = bool(np.all(ce[0] < -noise[0]) and np.all(ce[-1] > noise[-1])
                and np.all(growth > growth_noise)
                and all(len(changes)==1 and changes[0]['before']==-1 and changes[0]['after']==1 for changes in transitions))
    return dict(status='PASS' if report['status'] != 'UNRESOLVED' and same_bracket else 'UNRESOLVED',
        executed_waists=supported, evidence=report,
        explicit_bracket_checkpoint_count=len(inside),
        same_explicit_covariance_growth_bracket=same_bracket,
        positive_current_membership_stable=stable)
