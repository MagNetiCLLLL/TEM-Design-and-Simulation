"""Post-process retained Image particles; never trace, refit, or change optics.

Inputs are an old validate_c2_screen_aberrations directory or a new
image-intensity-convergence-v1 manifest directory. All runs in one collection
share one physical ROI. Separate collections are never silently treated as
the same source. Raw histogram, optional conservative bilinear display weights,
and the fixed 0.6 mm Gaussian detector response are distinct observables.

Pixel refinement compares piecewise-constant densities: each coarse pixel's
probability is distributed equally over its finer children. Exact aggregation
of fine hard bins is only a conservation check, NOT a convergence criterion.
The original relative tolerance is 5%; zero/zero is unresolved, never a pass.

The Gaussian reference uses integrated Hermite translation terms through total
order six, only for compact clouds with max |r|/sigma <= 0.001 and pixel edges
within +/-5 sigma. A directional seventh-derivative Taylor bound is reported;
32-pixel direct Gaussian integrals independently check absolute intensities and
paired differences. No source sampling confidence interval is inferred.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import numpy as np
from scipy.special import eval_hermitenorm, ndtr

CASES = ('baseline', 'Cs_plus20mm', 'A1_plus0.03mm', 'B2_plus1mm')
TOLERANCE = .05
SIGMA_MM = .6
PROJECTOR_KEYS = ('diffraction_lens', 'intermediate_lens', 'projector_lens_1', 'projector_lens_2')


@dataclass
class Cloud:
    xy_mm: np.ndarray
    weight: np.ndarray
    survivor: np.ndarray
    source_id: np.ndarray
    blocked_key: np.ndarray | None = None
    blocked_z_mm: np.ndarray | None = None
    final_alive: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.weight)
        if (self.xy_mm.shape != (n, 2) or self.survivor.shape != (n,)
                or self.source_id.shape != (n,) or len(np.unique(self.source_id)) != n
                or not np.isfinite(self.xy_mm[(self.weight > 0) & self.survivor]).all()
                or not np.isfinite(self.weight).all()
                or np.any(self.weight < 0)):
            raise ValueError('Invalid retained particle coordinates, weights, or source IDs')
        if (self.blocked_key is None) != (self.blocked_z_mm is None):
            raise ValueError('Retained stop keys and planes must be provided together')
        if self.blocked_key is not None and (
                self.blocked_key.shape != (n,) or self.blocked_key.dtype.kind not in 'US'
                or self.blocked_z_mm.shape != (n,)):
            raise ValueError('Invalid retained particle stop records')
        if self.final_alive is not None and self.final_alive.shape != (n,):
            raise ValueError('Invalid retained final-alive records')

    @property
    def received(self):
        return self.weight*self.survivor


@dataclass
class Run:
    label: str
    rays: int
    step_mm: float
    clouds: dict[str, Cloud]
    provenance: dict


def validate_point_spread(captured):
    if (not isinstance(captured, dict) or captured.get('model') != 'gaussian'
            or captured.get('sigma_x_mm') != SIGMA_MM or captured.get('sigma_y_mm') != SIGMA_MM
            or not math.isfinite(float(captured.get('rotation_deg', math.nan)))):
        raise ValueError('This reference requires a captured isotropic Gaussian PSF with sigma=0.6 mm')
    return captured


def projector_values(captured):
    if isinstance(captured, dict) and 'imaging' in captured:
        captured = captured['imaging']
    if isinstance(captured, dict) and 'lens_percent' in captured:
        captured = captured['lens_percent']
    if not isinstance(captured, dict) or any(key not in captured for key in PROJECTOR_KEYS):
        raise ValueError('Captured fixed D/I/P1/P2 controls are required for intensity comparisons')
    values = tuple(float(captured[key].get('percent') if isinstance(captured[key], dict) else captured[key])
                   for key in PROJECTOR_KEYS)
    if not all(map(math.isfinite, values)):
        raise ValueError('Captured projector controls must be finite')
    return values


def canonical_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def validate_protocol_receipt(document, row, label, directory):
    """Bind collector rows to one declared optics protocol, without restoring it.

    Older retained data has no such per-run receipt and is labelled accordingly.
    The checksum-bound snapshot and request digest prevent silently mixing rows
    collected with another protocol; this is provenance, not optics validation.
    """
    protocol = document.get('protocol')
    if protocol is None:
        return
    identity = canonical_digest(protocol)
    if identity != document.get('protocol_identity'):
        raise ValueError('Collector protocol checksum mismatch')
    if projector_values(protocol.get('projector_controls')) != projector_values(document.get('projector_controls')):
        raise ValueError('Projector controls differ from the common collection protocol')
    if float(row['receiver_z_mm']) != float(protocol['receiver_z_mm']):
        raise ValueError('Observation plane differs from the common collection protocol')
    if any(row.get(field) != protocol['implementation'] for field in ('implementation_before', 'implementation_after')):
        raise ValueError('Production implementation differs from the common collection protocol')
    snapshot = directory/row['snapshot_path']
    data = snapshot.read_bytes()
    if sha256(data).hexdigest() != row['snapshot_sha256']:
        raise ValueError('Retained snapshot checksum differs from collector manifest')
    snapshot_digest = json.loads(data)['digest']
    expected = canonical_digest(dict(protocol=identity, state=snapshot_digest,
                                     target_z_mm=float(row['receiver_z_mm']), run_id=label))
    if row.get('request_identity') != expected:
        raise ValueError('Run request belongs to a different protocol, snapshot, or observation plane')


def load_runs(path):
    path = Path(path)
    directory = path.parent if path.is_file() else path
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        document = json.loads(manifest_path.read_text(encoding='utf-8'))
        if document.get('schema') != 'image-intensity-convergence-v1':
            raise ValueError('Unsupported particle matrix manifest')
        validate_point_spread(document.get('point_spread'))
        fixed_projector = projector_values(document.get('projector_controls'))
        grouped, pending, source_by_population, current_by_population = {}, [], {}, {}
        receiver_z = set()
        for label, row in document['runs'].items():
            if row.get('status') != 'COMPLETE':
                pending.append(dict(label=label, status=row.get('status')))
                continue
            if projector_values(row.get('projector_controls')) != fixed_projector:
                raise ValueError('Projector controls changed across the supposedly fixed intensity matrix')
            validate_protocol_receipt(document, row, label, directory)
            receiver_z.add(float(row['receiver_z_mm']))
            key = (int(row['rays']), float(row['step_mm']))
            source_digest = row.get('source_digest')
            if not isinstance(source_digest, str) or not source_digest:
                raise ValueError('A full physical source digest is required for every completed run')
            if key[0] in source_by_population and source_by_population[key[0]] != source_digest:
                raise ValueError('Full source digest changed across cases or integration steps at fixed population')
            source_by_population[key[0]] = source_digest
            if 'emitted_current_a' in row:
                current = float(row['emitted_current_a'])
                if (not math.isfinite(current) or current < 0
                        or (key[0] in current_by_population and current_by_population[key[0]] != current)):
                    raise ValueError('Emitted source current must be finite and unchanged at fixed population')
                current_by_population[key[0]] = current
            source = directory/row['particles_path']
            digest = sha256(source.read_bytes()).hexdigest()
            for field in ('particles_sha256', 'particles_digest', 'file_sha256'):
                if row.get(field) is not None and row[field] != digest:
                    raise ValueError('Retained particle file checksum differs from collector manifest')
            with np.load(source, allow_pickle=False) as data:
                cloud = Cloud(data['xy_m']*1e3, data['weight'], data['survivor'].astype(bool), data['source_id'],
                    data['blocked_key'] if 'blocked_key' in data else None,
                    data['blocked_z_mm'] if 'blocked_z_mm' in data else None,
                    data['final_alive'].astype(bool) if 'final_alive' in data else None)
            if len(cloud.weight) != key[0]:
                raise ValueError('Manifest population differs from retained arrays')
            bucket = grouped.setdefault(key, {})
            if row['case'] in bucket:
                raise ValueError('Duplicate case in a population/step group')
            bucket[row['case']] = (cloud, dict(row, particles_sha256=digest))
        if len(receiver_z) > 1 or any(not math.isfinite(z) for z in receiver_z):
            raise ValueError('Observation plane changed across the intensity matrix')
        for expected in document.get('matrix', []):
            key = int(expected['rays']), float(expected['step_mm'])
            if key not in grouped:
                pending.append(dict(rays=key[0], step_mm=key[1], status='PLANNED_GROUP_NOT_COMPLETE'))
        runs = []
        for (n, h), bucket in sorted(grouped.items()):
            if not set(CASES).issubset(bucket):
                pending.append(dict(rays=n, step_mm=h, status='INCOMPLETE_CASE_GROUP'))
                continue
            digests = {bucket[name][1]['source_digest'] for name in CASES}
            if len(digests) != 1:
                raise ValueError('Cases within one population/step must use the same source digest')
            runs.append(Run(f'N{n}_h{repr(float(h))}', n, h, {name: bucket[name][0] for name in CASES},
                            {name: bucket[name][1] for name in CASES}))
        validate_source_arrays(runs)
        return runs, dict(input=str(manifest_path.resolve()), pending=pending, schema=document['schema'],
                          point_spread=document['point_spread'], projector_controls=document['projector_controls'],
                          protocol=document.get('protocol'), protocol_identity=document.get('protocol_identity'),
                          source_sampling_note=document.get('source_sampling_note'),
                          fixed_population_sources=source_by_population,
                          emitted_current_a_by_population=current_by_population,
                          source_receipts=document.get('sources'),
                          retained_source_summaries=retained_source_summaries(document, directory, runs),
                          source_sampling_prefix_checks=document.get('source_sampling_prefix_checks'),
                          projector_validation='per-run captured D/I/P1/P2 controls and receiver plane agree')
    document = json.loads((directory/'comparison.json').read_text(encoding='utf-8'))
    validate_point_spread(document.get('point_spread'))
    projector_values(document.get('projector_controls'))
    source = directory/'particles.npz'
    runs = []
    with np.load(source, allow_pickle=False) as data:
        for index, h in enumerate(document['integration_steps_mm']):
            clouds = {}
            for name in CASES:
                prefix = f'step{index}_{name}_imaging_'
                clouds[name] = Cloud(data[prefix+'xy_m']*1e3, data[prefix+'weight'],
                    data[prefix+'survivor'].astype(bool), data['source_id'])
            runs.append(Run(f'N{len(data["source_id"])}_h{repr(float(h))}', len(data['source_id']), float(h), clouds,
                            dict(particles_sha256=sha256(source.read_bytes()).hexdigest(),
                                 execution_source_identity=document.get('solver_source_identity'),
                                 emitted_current_a=document.get('gun', {}).get('emitted_current_a'))))
    return runs, dict(input=str(source.resolve()), pending=[], schema=document['schema'],
                      point_spread=document['point_spread'], projector_controls=document['projector_controls'],
                      emitted_current_a_by_population={runs[0].rays: document.get('gun', {}).get('emitted_current_a')},
                      projector_validation='legacy collector declared one fixed Image solution for all cases and steps; no per-run receipt')


def histogram(cloud, x_edges, y_edges):
    active = cloud.received > 0
    return np.histogram2d(cloud.xy_mm[active, 1], cloud.xy_mm[active, 0],
                          bins=(y_edges, x_edges), weights=cloud.received[active])[0]


def bilinear(cloud, x_edges, y_edges):
    """Conservative display interpolation, not a physical detector kernel."""
    nx, ny = len(x_edges)-1, len(y_edges)-1
    active = cloud.received > 0
    xy, weights = cloud.xy_mm[active], cloud.received[active]
    gx = (xy[:, 0]-x_edges[0])/(x_edges[1]-x_edges[0])-.5
    gy = (xy[:, 1]-y_edges[0])/(y_edges[1]-y_edges[0])-.5
    ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx-ix, gy-iy
    out = np.zeros((ny, nx))
    for ox, oy, fraction in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                              (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        col, row = ix+ox, iy+oy
        valid = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
        out += np.bincount(row[valid]*nx+col[valid],
            weights=(weights*fraction)[valid], minlength=nx*ny).reshape(ny, nx)
    return out


def aggregate_probability(fine, factor):
    if type(factor) is not int or factor < 1 or any(n % factor for n in fine.shape):
        raise ValueError('Nested probability grids require an integer refinement factor')
    return fine.reshape(fine.shape[0]//factor, factor, fine.shape[1]//factor, factor).sum(axis=(1, 3))


def compare_response(coarse, fine, *, numerical_error_bound=0.):
    signal, error = float(np.abs(fine).sum()), float(np.abs(fine-coarse).sum())
    if signal == 0.:
        status, relative = 'UNRESOLVED_ZERO_FINE_SIGNAL', None
    elif numerical_error_bound >= TOLERANCE*signal:
        status, relative = 'UNRESOLVED_NUMERICAL_BOUND', error/signal
    else:
        relative = error/signal
        status = 'PASS_DECLARED_5_PERCENT' if relative <= TOLERANCE else 'FAIL_DECLARED_5_PERCENT'
    return dict(signal_l1=signal, error_l1=error, relative_error=relative,
                status=status, numerical_error_bound_l1=numerical_error_bound)


def density_refinement(coarse, fine, **kwargs):
    if fine.shape[0] % coarse.shape[0] or fine.shape[1] % coarse.shape[1]:
        raise ValueError('Pixel comparison requires nested grids on the same physical ROI')
    fy, fx = fine.shape[0]//coarse.shape[0], fine.shape[1]//coarse.shape[1]
    prolonged = np.repeat(np.repeat(coarse, fy, axis=0), fx, axis=1)/(fx*fy)
    return compare_response(prolonged, fine, **kwargs)


def translation_moments(cloud, baseline=None, *, sigma=SIGMA_MM):
    q = np.asarray(np.where((cloud.received > 0)[:, None], cloud.xy_mm, 0.), dtype=np.longdouble)/sigma
    w = np.asarray(cloud.received, dtype=np.longdouble)
    b = bw = None
    if baseline is not None:
        if not np.array_equal(cloud.source_id, baseline.source_id):
            raise ValueError('Paired same-run cases must preserve source ID order')
        b = np.asarray(np.where((baseline.received > 0)[:, None], baseline.xy_mm, 0.), dtype=np.longdouble)/sigma
        bw = np.asarray(baseline.received, dtype=np.longdouble)
    result = {}
    for p in range(7):
        for r in range(7-p):
            term = w*q[:, 0]**p*q[:, 1]**r
            if baseline is not None:
                term -= bw*b[:, 0]**p*b[:, 1]**r
            result[p, r] = float(np.sum(term, dtype=np.longdouble)/math.factorial(p)/math.factorial(r))
    return result


def gaussian_remainder_bound(cloud, extent, *, sigma=SIGMA_MM):
    """Conservative absolute integral bound for the omitted total orders >=7.

    Along the translation t -> t-lambda*u, the seventh derivative is
    |u|**7 He_7((t-lambda*u).u_hat) times the standard 2D Gaussian.
    Bound Gaussian by 1/(2*pi) and |He_7(v)| by
    |v|**7+21|v|**5+105|v|**3+105|v|, then integrate the Taylor remainder.
    This bound is independent of pixel pitch. It excludes floating roundoff,
    which is separately checked against direct integrals at 32 pixels.
    """
    active = cloud.received > 0
    radius = float(np.linalg.norm(cloud.xy_mm[active]/sigma, axis=1).max(initial=0.))
    if radius > .001 or max(map(abs, extent)) > 5*sigma:
        raise ValueError('Compact Gaussian reference domain exceeded; direct integration is required')
    vmax = math.hypot(max(map(abs, extent[:2])), max(map(abs, extent[2:])))/sigma+radius
    polynomial = vmax**7+21*vmax**5+105*vmax**3+105*vmax
    area = (extent[1]-extent[0])*(extent[3]-extent[2])/sigma**2
    return float(cloud.received.sum())*area*radius**7*polynomial/(2*math.pi*math.factorial(7))


def gaussian_probability(moments, x_edges, y_edges, *, sigma=SIGMA_MM, order=6):
    def basis(edges):
        t = edges/sigma
        phi = np.exp(-.5*t*t)/math.sqrt(2*math.pi)
        result = [ndtr(t[1:])-ndtr(t[:-1])]
        for p in range(1, order+1):
            edge = phi*eval_hermitenorm(p-1, t)
            result.append(edge[:-1]-edge[1:])
        return result
    bx, by = basis(x_edges), basis(y_edges)
    result = np.zeros((len(y_edges)-1, len(x_edges)-1))
    for (p, r), coefficient in moments.items():
        if p+r <= order:
            result += coefficient*np.outer(by[r], bx[p])
    return result


def direct_gaussian(cloud, x_edges, y_edges, *, sigma=SIGMA_MM):
    """Independent positive pixel integrals, used only on a bounded 32² check."""
    active = cloud.received > 0
    xy, weights = cloud.xy_mm[active], cloud.received[active]
    result = np.zeros((len(y_edges)-1, len(x_edges)-1))
    for start in range(0, len(weights), 128):
        q, w = xy[start:start+128], weights[start:start+128]
        def mass(edges, centres):
            t = (edges[:, None]-centres[None, :])/sigma
            lo, hi = t[:-1], t[1:]
            return np.where(lo > 0, ndtr(-lo)-ndtr(-hi), ndtr(hi)-ndtr(lo))
        result += (mass(y_edges, q[:, 1])*w[None, :]) @ mass(x_edges, q[:, 0]).T
    return result


def population_summary(cloud):
    w = cloud.received
    positive = w[w > 0]
    return dict(source_count=len(w), received_count=int(np.count_nonzero(w)),
                source_weight_total=float(cloud.weight.sum()), received_weight_total=float(w.sum()),
                effective_received_count=float(w.sum()**2/np.dot(w, w)) if np.any(w) else 0.,
                single_particle_two_bin_l1_min=float(2*positive.min()) if len(positive) else None,
                single_particle_two_bin_l1_max=float(2*positive.max()) if len(positive) else None)


def stop_summary(weight, keys, planes, mask=None):
    """Describe recorded first stops; absent records never imply no clipping."""
    if keys is None:
        return dict(status='NOT_RECORDED', by_key={})
    if keys.shape != weight.shape or planes.shape != weight.shape or keys.dtype.kind not in 'US':
        raise ValueError('Stop records must match source probabilities')
    selected = np.ones(len(weight), dtype=bool) if mask is None else mask
    result = {}
    for key in np.unique(keys[selected]):
        active = selected & (keys == key)
        z = planes[active]
        finite = z[np.isfinite(z)]
        result[str(key)] = dict(count=int(active.sum()), source_probability=float(weight[active].sum()),
            z_mm_min=float(finite.min()) if len(finite) else None,
            z_mm_max=float(finite.max()) if len(finite) else None,
            nonfinite_z_count=int((~np.isfinite(z)).sum()))
    return dict(status='RECORDED_FIRST_STOP', by_key=result)


def retained_source_summaries(document, directory, runs):
    """Read checksum-bound gun summaries separately from screen survival."""
    result = {}
    for run in runs:
        key = str(run.rays)
        if key in result:
            continue
        receipt = (document.get('sources') or {}).get(key)
        if not receipt or 'particles_path' not in receipt:
            result[key] = dict(status='SOURCE_ARRAYS_NOT_RECORDED')
            continue
        if receipt.get('source_digest') != run.provenance['baseline']['source_digest']:
            raise ValueError('Retained gun source digest differs from screen source receipt')
        path = directory/receipt['particles_path']
        digest = sha256(path.read_bytes()).hexdigest()
        if receipt.get('particles_sha256') is not None and digest != receipt['particles_sha256']:
            raise ValueError('Retained source file checksum differs from collector manifest')
        base = run.clouds['baseline']
        with np.load(path, allow_pickle=False) as data:
            weight, ids, alive = data['gun_exit_weight'], data['gun_exit_ray_id'], data['gun_exit_alive'].astype(bool)
            if (not np.array_equal(weight, base.weight) or not np.array_equal(ids, base.source_id)
                    or alive.shape != weight.shape):
                raise ValueError('Retained gun source IDs and weights differ from screen particles')
            keys = data['gun_blocked_key'] if 'gun_blocked_key' in data else None
            planes = data['gun_blocked_z_mm'] if 'gun_blocked_z_mm' in data else None
            if (keys is None) != (planes is None):
                raise ValueError('Retained gun stop keys and planes must be provided together')
            result[key] = dict(status='RETAINED_GUN_ARRAYS', particles_sha256=digest,
                source_digest=receipt.get('source_digest'), source_count=len(weight),
                source_probability=float(weight.sum()), gun_exit_count=int(alive.sum()),
                gun_exit_probability=float(weight[alive].sum()),
                stops=stop_summary(weight, keys, planes), sampler_diagnostics=receipt.get('sampler_diagnostics'))
    return result


def survival_components(cloud, baseline):
    """Split a paired response without changing source weights or centroids.

    Delta = common-survivor displacement - lost baseline arrivals + gained
    case arrivals. Signed full-plane flux differs from finite-viewport flux.
    The centroid contributions use (delta M - mu_base delta P)/P_case;
    they add to the actual normalised centroid change, including losses.
    """
    if not np.array_equal(cloud.source_id, baseline.source_id) or not np.array_equal(cloud.weight, baseline.weight):
        raise ValueError('Survival decomposition requires identical source IDs and weights')
    positive = cloud.weight > 0
    common = positive & baseline.survivor & cloud.survivor
    lost = positive & baseline.survivor & ~cloud.survivor
    gained = positive & ~baseline.survivor & cloud.survivor
    def subset(c, mask):
        return Cloud(c.xy_mm, c.weight, mask, c.source_id)
    moments = dict(common_displacement=translation_moments(subset(cloud, common), subset(baseline, common)),
        lost_negative={key: -value for key, value in translation_moments(subset(baseline, lost)).items()},
        gained_positive=translation_moments(subset(cloud, gained)))
    def mass_moment(c):
        active = c.received > 0
        w = c.weight[active].astype(np.longdouble)
        return np.sum(w), np.sum(c.xy_mm[active].astype(np.longdouble)*w[:, None], axis=0)
    pb, mb = mass_moment(baseline)
    pc, mc = mass_moment(cloud)
    mu = mb/pb if pb > 0 else None
    summary = dict(definition='Positive-weight common, lost and gained receiver survivors; original source probabilities retained.',
        baseline_probability=float(pb), case_probability=float(pc), signed_full_plane_probability_change=float(pc-pb),
        baseline_centroid_mm=np.asarray(mu, float).tolist() if mu is not None else None,
        case_centroid_mm=np.asarray(mc/pc, float).tolist() if pc > 0 else None,
        actual_centroid_change_mm=np.asarray(mc/pc-mu, float).tolist() if mu is not None and pc > 0 else None,
        zero_weight_membership_changes=int(np.count_nonzero((cloud.survivor != baseline.survivor) & ~positive)), components={})
    for name, mask, sign in [('common_displacement', common, 1), ('lost_negative', lost, -1), ('gained_positive', gained, 1)]:
        w = cloud.weight[mask].astype(np.longdouble)
        if name == 'common_displacement':
            dp = np.longdouble(0)
            coordinates = cloud.xy_mm[mask].astype(np.longdouble)-baseline.xy_mm[mask].astype(np.longdouble)
        else:
            dp = sign*np.sum(w)
            coordinates = (baseline if sign < 0 else cloud).xy_mm[mask].astype(np.longdouble)
        dm = sign*np.sum(coordinates*w[:, None], axis=0)
        summary['components'][name] = dict(count=int(mask.sum()), signed_full_plane_probability=float(dp),
            unnormalised_first_moment_mm=np.asarray(dm, float).tolist(),
            centroid_change_contribution_mm=np.asarray((dm-mu*dp)/pc, float).tolist() if mu is not None and pc > 0 else None,
            baseline_stops=stop_summary(baseline.weight, baseline.blocked_key, baseline.blocked_z_mm, mask),
            case_stops=stop_summary(cloud.weight, cloud.blocked_key, cloud.blocked_z_mm, mask))
        if name != 'common_displacement':
            summary['components'][name]['source_ids'] = cloud.source_id[mask].tolist()
    return moments, summary


def survival_response_summary(components, total_moments, x_edges, y_edges):
    """PSF decomposition on one fixed physical viewport; L1 norms do not add."""
    images = {key: gaussian_probability(value, x_edges, y_edges) for key, value in components.items()}
    total = gaussian_probability(total_moments, x_edges, y_edges)
    sum_image = sum(images.values())
    norms = {key: float(np.abs(value).sum()) for key, value in images.items()}
    sum_l1, total_l1 = sum(norms.values()), float(np.abs(total).sum())
    return dict(components={key: dict(absolute_l1=norms[key], signed_viewport_probability=float(value.sum()))
                            for key, value in images.items()},
        total_absolute_l1=total_l1, total_signed_viewport_probability=float(total.sum()),
        sum_component_absolute_l1=sum_l1, cancelled_component_l1=sum_l1-total_l1,
        cancelled_fraction=(sum_l1-total_l1)/sum_l1 if sum_l1 > 0 else None,
        reconstruction_l1_error=float(np.abs(sum_image-total).sum()),
        reconstruction_max_absolute_error=float(np.abs(sum_image-total).max()),
        role='Signed physical PSF contributions; cancellation is retained and is not a convergence pass.')


def validate_source_arrays(runs):
    """IDs and source weights must match across cases and h, separately for each N.

    The manifest loader additionally requires the full physical source digest,
    which includes energies and directions. Array IDs alone cannot prove this.
    Cross-N sources are intentionally not paired: the collector's deterministic
    quadrature normalises the finite energy distribution independently at each N.
    """
    bases = {}
    for run in runs:
        base = bases.setdefault(run.rays, run.clouds['baseline'])
        for cloud in run.clouds.values():
            if not np.array_equal(cloud.source_id, base.source_id):
                raise ValueError('Fixed-population cases and steps must retain identical source ID order')
            if not np.array_equal(cloud.weight, base.weight):
                raise ValueError('Fixed-population cases and steps must retain identical source weights')


def analyse(runs, pixels, *, phase_offsets=False, include_bilinear=True):
    if not runs:
        raise ValueError('No complete baseline/Cs/A1/B2 run group is available')
    if len({(run.rays, run.step_mm) for run in runs}) != len(runs):
        raise ValueError('Duplicate population/step groups must not be silently merged')
    if len({run.label for run in runs}) != len(runs):
        raise ValueError('Run labels must uniquely identify population/step groups')
    validate_source_arrays(runs)
    if any(b % a for a, b in zip(pixels, pixels[1:])):
        raise ValueError('Requested pixel grids must be nested integer refinements')
    half = max(np.abs(cloud.xy_mm[cloud.received > 0]).max(initial=0.)
               for run in runs for cloud in run.clouds.values())*1.08
    if half <= 0:
        raise ValueError('A finite nonzero common physical ROI is required')
    raw_extent, response_extent = (-half, half, -half, half), (-2.4, 2.4, -2.4, 2.4)
    methods = ['hard_histogram']+(['bilinear_display'] if include_bilinear else [])+['fixed_physical_gaussian']
    output = dict(tolerance=TOLERANCE, raw_extent_mm=raw_extent, response_extent_mm=response_extent,
        sigma_mm=SIGMA_MM, pixels=pixels, runs=[], crosschecks=[], signals=[], axes=[], phase_offsets=[],
        production_psf_checks=[], survival_decomposition=[],
        limitations=['No new source sampling or propagation; input provenance remains authoritative.',
            'All intensity differences use probability per emitted electron; emitted source current is reported separately and is not silently renormalised across N.',
            'Cross-population comparisons refine deterministic source quadrature; they are not IID confidence intervals or paired trajectories.',
            'Hard histograms at fixed finite population do not define a smooth continuum density as pixels shrink.',
            'Bilinear deposition is a pixel-dependent numerical display kernel, not detector physics.',
            'The fixed Gaussian sigma is a provisional simulator parameter, not an instrument calibration.',
            'Finite tilt, centroid shifts, defocus, survival changes and original source weights are retained.'])
    moments, bounds, component_moments = {}, {}, {}
    check_edges = np.linspace(-2.4, 2.4, 33)
    for run in runs:
        base = run.clouds['baseline']
        run_row = dict(label=run.label, rays=run.rays, step_mm=run.step_mm, provenance=run.provenance, cases={})
        for name, cloud in run.clouds.items():
            if not np.array_equal(cloud.source_id, base.source_id):
                raise ValueError('Cases must retain identical source ID order within one run')
            if not np.array_equal(cloud.weight, base.weight):
                raise ValueError('Cases must retain identical source weights within one run')
            run_row['cases'][name] = population_summary(cloud)
            run_row['cases'][name]['recorded_stops'] = stop_summary(cloud.weight, cloud.blocked_key, cloud.blocked_z_mm)
            run_row['cases'][name]['final_alive_count'] = int(cloud.final_alive.sum()) if cloud.final_alive is not None else None
            current = run.provenance.get(name, {}).get('emitted_current_a', run.provenance.get('emitted_current_a'))
            run_row['cases'][name]['emitted_current_a'] = current
            run_row['cases'][name]['received_current_pa'] = (float(current)*1e12*float(cloud.received.sum())
                                                            if current is not None else None)
            run_row['cases'][name]['survival_membership_difference_from_baseline'] = int(np.count_nonzero(cloud.survivor != base.survivor))
            moments[run.label, name] = translation_moments(cloud, None if name == 'baseline' else base)
            bounds[run.label, name] = gaussian_remainder_bound(cloud, response_extent)
            if name != 'baseline':
                component_moments[run.label, name], decomposition = survival_components(cloud, base)
                run_row['cases'][name]['survival_decomposition'] = decomposition
        direct_base = direct_gaussian(base, check_edges, check_edges)
        for name in CASES[1:]:
            actual = direct_gaussian(run.clouds[name], check_edges, check_edges)-direct_base
            reference = gaussian_probability(moments[run.label, name], check_edges, check_edges)
            bound = bounds[run.label, name]+bounds[run.label, 'baseline']
            check = compare_response(actual, reference, numerical_error_bound=bound)
            if check['signal_l1'] > 0 and check['status'] != 'PASS_DECLARED_5_PERCENT':
                raise ValueError(f'Gaussian reference crosscheck unresolved for {run.label}/{name}: {check}')
            output['crosschecks'].append(dict(run=run.label, case=name, pixels=32, **check))
        output['runs'].append(run_row)
    previous = {}
    for n in pixels:
        current = {}
        raw_edges, response_edges = np.linspace(-half, half, n+1), np.linspace(-2.4, 2.4, n+1)
        for run in runs:
            for method in methods:
                if method == 'fixed_physical_gaussian':
                    images = {name: gaussian_probability(moments[run.label, name], response_edges, response_edges) for name in CASES}
                else:
                    operation = histogram if method == 'hard_histogram' else bilinear
                    images = {name: operation(cloud, raw_edges, raw_edges) for name, cloud in run.clouds.items()}
                for name in CASES[1:]:
                    delta = images[name] if method == 'fixed_physical_gaussian' else images[name]-images['baseline']
                    key = run.label, method, name
                    current[key] = delta
                    bound = (bounds[run.label, name]+bounds[run.label, 'baseline']) if method == 'fixed_physical_gaussian' else 0.
                    output['signals'].append(dict(run=run.label, method=method, case=name, pixels=n,
                        signal_l1=float(np.abs(delta).sum()), signed_probability_change=float(delta.sum()),
                        numerical_error_bound_l1=bound,
                        baseline_occupied_pixels=int(np.count_nonzero(images['baseline'])),
                        baseline_pixel_count=n*n))
                    if key in previous:
                        old_n, old_delta = previous[key]
                        metrics = density_refinement(old_delta, delta, numerical_error_bound=2*bound)
                        output['axes'].append(dict(axis='pixels', run=run.label, method=method, case=name,
                            coarse_pixels=old_n, fine_pixels=n,
                            nested_aggregation_difference_l1=float(np.abs(aggregate_probability(delta, n//old_n)-old_delta).sum()),
                            nested_aggregation_role='conservation sanity only; not a pixel convergence criterion', **metrics))
            from temsim.detector.receiver_image import _response_from_hits
            from temsim.detector.point_spread import DetectorPointSpread
            psf = DetectorPointSpread('gaussian', SIGMA_MM, SIGMA_MM, 0.,
                                      'provisional_model_parameter', 'captured isotropic reference')
            responses = {}
            for name, cloud in run.clouds.items():
                active = cloud.received > 0
                responses[name], method, pitch = _response_from_hits(
                    cloud.xy_mm[active], cloud.received[active], response_edges, response_edges, psf)
            for name in CASES[1:]:
                reference = current[run.label, 'fixed_physical_gaussian', name]
                output['survival_decomposition'].append(dict(run=run.label, case=name, pixels=n,
                    **survival_response_summary(component_moments[run.label, name], moments[run.label, name],
                                                response_edges, response_edges)))
                check = compare_response(responses[name]-responses['baseline'], reference,
                    numerical_error_bound=bounds[run.label, name]+bounds[run.label, 'baseline'])
                output['production_psf_checks'].append(dict(run=run.label, case=name, pixels=n,
                    production_method=method, work_pitch_mm=pitch, **check))
        # The source population and integration step each change separately.
        for axis in ('integration_step', 'particle_count'):
            fixed_values = sorted({r.rays if axis == 'integration_step' else r.step_mm for r in runs})
            for fixed in fixed_values:
                selected = [r for r in runs if (r.rays if axis == 'integration_step' else r.step_mm) == fixed]
                selected.sort(key=lambda r: -r.step_mm if axis == 'integration_step' else r.rays)
                for coarse, fine in zip(selected, selected[1:]):
                    for method in methods:
                        for name in CASES[1:]:
                            bound = (sum(bounds[r.label, name]+bounds[r.label, 'baseline'] for r in (coarse, fine))
                                     if method == 'fixed_physical_gaussian' else 0.)
                            output['axes'].append(dict(axis=axis, coarse_run=coarse.label, fine_run=fine.label,
                                method=method, case=name, pixels=n, **compare_response(
                                    current[coarse.label, method, name], current[fine.label, method, name],
                                    numerical_error_bound=bound)))
        previous = {key: (n, value) for key, value in current.items()}
        if phase_offsets:
            for run in runs:
                for ox in (0., .25, .5, .75):
                    for oy in (0., .25, .5, .75):
                        xe, ye = raw_edges+ox*2*half/n, raw_edges+oy*2*half/n
                        for method, operation in [('hard_histogram', histogram), ('bilinear_display', bilinear)]:
                            images = {name: operation(cloud, xe, ye) for name, cloud in run.clouds.items()}
                            for name in CASES[1:]:
                                output['phase_offsets'].append(dict(run=run.label, method=method, case=name, pixels=n,
                                    offset_pixel=[ox, oy], signal_l1=float(np.abs(images[name]-images['baseline']).sum()),
                                    extent_mm=[xe[0], xe[-1], ye[0], ye[-1]],
                                    retained_probability=float(images[name].sum()),
                                    cropped_probability=float(run.clouds[name].received.sum()-images[name].sum()),
                                    role='bin-boundary sensitivity only; finite outer edges shift by less than one pixel'))
    output['axis_availability'] = {axis: 'EVALUATED' if any(r['axis'] == axis for r in output['axes']) else 'NOT_RUN_NO_INDEPENDENT_AXIS_PAIR'
                                   for axis in ('integration_step', 'particle_count', 'pixels')}
    return output


def make_plot(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    run = max(report['runs'], key=lambda r: (r['rays'], -r['step_mm']))['label']
    labels = {'hard_histogram': 'Raw hard bins', 'bilinear_display': 'Bilinear display only',
              'fixed_physical_gaussian': 'Physical Gaussian sigma=0.6 mm'}
    for col, case in enumerate(CASES[1:]):
        for method, label in labels.items():
            signals = [r for r in report['signals'] if r['run'] == run and r['case'] == case and r['method'] == method]
            if not signals:
                continue
            axes[0, col].plot([r['pixels'] for r in signals], [r['signal_l1'] or np.nan for r in signals], 'o-', label=label)
            rows = [r for r in report['axes'] if r['axis'] == 'pixels' and r['run'] == run and r['case'] == case and r['method'] == method]
            axes[1, col].plot([r['fine_pixels'] for r in rows], [r['relative_error'] for r in rows], 'o-')
        axes[0, col].set_title(case)
        axes[0, col].set_yscale('log')
        axes[1, col].axhline(TOLERANCE, color='black', linestyle='--')
        axes[1, col].set_yscale('log')
        for row in range(2):
            axes[row, col].set_xscale('log', base=2)
            axes[row, col].set_xlabel('Pixels per axis')
            axes[row, col].grid(alpha=.25)
    axes[0, 0].set_ylabel('Absolute delta L1 / emitted electron')
    axes[1, 0].set_ylabel('Pixel density refinement relative L1 error')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{run}: fixed retained particles, distinct raw/display/physical observables\nZero raw differences are unresolved; coarse/fine aggregation alone is not convergence')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def postprocessing_provenance():
    """Current rendering implementation, kept separate from input propagation."""
    from temsim.calculation_manifest import solver_source_identity
    receiver = ROOT/'src/temsim/detector/receiver_image.py'
    helper = ROOT/'src/temsim/detector/compact_gaussian_response.py'
    return dict(
        production_response_sha256=sha256(receiver.read_bytes()).hexdigest(),
        compact_gaussian_helper_sha256=sha256(helper.read_bytes()).hexdigest() if helper.exists() else None,
        postprocessing_source_identity=solver_source_identity(),
        postprocessing_identity_role='Current rendering code only; original propagation identity remains in each input protocol and run receipt.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pixels', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    parser.add_argument('--phase-offsets', action='store_true')
    args = parser.parse_args()
    if sorted(set(args.pixels)) != args.pixels or any(n < 2 or n > 512 for n in args.pixels):
        parser.error('Use strictly increasing nested grids from 2 through 512')
    args.output.mkdir(parents=True, exist_ok=True)
    implementation = postprocessing_provenance()
    reports = []
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        for index, path in enumerate(args.input):
            runs, provenance = load_runs(path)
            report = analyse(runs, args.pixels, phase_offsets=args.phase_offsets)
            report['input'] = provenance
            report['matrix_status'] = ('PARTIAL_INPUT_MATRIX' if provenance['pending'] else
                                       'AVAILABLE_INPUT_MATRIX_ANALYSED_NOT_OVERALL_QUALIFICATION')
            reports.append(report)
            make_plot(report, args.output/f'pixel-convergence-{index}.png')
            write_table(args.output/f'axis-comparisons-{index}.csv', report['axes'])
            write_table(args.output/f'signal-amplitudes-{index}.csv', report['signals'])
            write_table(args.output/f'population-summary-{index}.csv', [dict(run=run['label'], case=name, **values)
                        for run in report['runs'] for name, values in run['cases'].items()])
            write_table(args.output/f'survival-decomposition-{index}.csv', [dict(
                run=row['run'], case=row['case'], pixels=row['pixels'], component=name, **values,
                total_absolute_l1=row['total_absolute_l1'], cancelled_component_l1=row['cancelled_component_l1'],
                cancelled_fraction=row['cancelled_fraction'], reconstruction_l1_error=row['reconstruction_l1_error'])
                for row in report['survival_decomposition'] for name, values in row['components'].items()])
    if implementation != postprocessing_provenance():
        raise RuntimeError('Postprocessing implementation changed during analysis; no mixed report is accepted')
    document = dict(schema='image-intensity-analysis-v1', role='independent numerical sensitivity axes; not microscope or source-sampling certification',
                    analysis_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    **implementation,
                    collections=reports)
    (args.output/'analysis.json').write_text(json.dumps(document, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([dict(input=r['input']['input'], axis_availability=r['axis_availability'],
                          comparisons=len(r['axes'])) for r in reports], indent=2))


if __name__ == '__main__':
    main()
