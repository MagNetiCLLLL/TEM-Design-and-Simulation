"""Place the finite-current D95 waist on the specimen using the series objective supply."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np
from scipy.optimize import brentq

from material_trial import Column, Field, HERE, source_and_electrodes
from relay_transfer import advance
from column_current import ApertureTracker
from condenser_pair_column import specimen
from stationary_pupil import half_nodes
from trace_normal_joint_field import beam_moments


def local_caustic(phase, weights):
    """Return the axial offsets of the D95 and RMS minima in local tangent propagation."""
    centered = phase[:, :4] - np.average(phase[:, :4], weights=weights, axis=0)
    x, y, vx, vy = centered.T
    rr, rv, vv = x*x+y*y, x*vx+y*vy, vx*vx+vy*vy
    rms_z = -np.average(rv, weights=weights) / np.average(vv, weights=weights)
    eq = np.all(weights == weights[0])
    rank = int(np.ceil(.95 * len(weights))) - 1

    def quantile_square(dz):
        square = rr + 2*dz*rv + dz*dz*vv
        if eq:
            q = np.partition(square, rank)[rank]
        else:
            order = np.argsort(square)
            q = square[order[np.searchsorted(weights[order].cumsum(), .95*weights.sum())]]
        return max(0., q)

    # At a fixed candidate radius each ray lies inside the disk on a single
    # axial interval. A sweep of their endpoints finds every feasible waist,
    # including narrow minima caused by a change of the 95th-percentile ray.
    moving = vv > 0.
    velocity2 = vv[moving]
    closest = -rv[moving]/velocity2
    min_square = (x[moving]*vy[moving]-y[moving]*vx[moving])**2/velocity2
    w = weights[moving]
    target = .95*weights.sum()

    def feasible(square):
        constant = weights[~moving][rr[~moving] <= square].sum()
        if constant >= target:
            return rms_z
        inside = min_square <= square
        if constant+w[inside].sum() < target:
            return None
        half = np.sqrt((square-min_square[inside])/velocity2[inside])
        starts, ends = closest[inside]-half, closest[inside]+half
        lower, upper = np.argsort(starts), np.argsort(ends)
        # Closed intervals include rays whose right endpoint equals a left endpoint.
        finish = np.searchsorted(ends[upper],starts[lower],side='left')
        covered = w[inside][lower].cumsum()-np.r_[0.,w[inside][upper].cumsum()][finish]+constant
        hits = np.flatnonzero(covered >= target)
        if len(hits):
            return starts[lower[hits[np.argmin(abs(starts[lower[hits]]-rms_z))]]]
        return None

    center = rms_z
    lo, hi = 0., quantile_square(center)
    for _ in range(36):
        square = .5*(lo+hi)
        position = feasible(square)
        if position is None:
            lo = square
        else:
            hi, center = square, position
    return dict(D95_offset_um=1e6*center, RMS_offset_um=1e6*rms_z,
                D95_at_waist_nm=2e9*np.sqrt(quantile_square(center)),
                D95_at_surface_nm=2e9*np.sqrt(quantile_square(0.)))


def propose(folder, span):
    record = json.loads((folder / 'tracking.json').read_text())
    solution = Path(record['solution'])
    dc = Field(solution, step=record['bore_step_mm'])
    unit = Field(solution, 'objective', step=record['bore_step_mm'])
    model = Column(source_and_electrodes(), dc, Path(record['gun_source']), record['count'], record['step_mm'])
    baseline = model.t.tables[0]
    derivative = cp.asarray(unit.table())
    reference = next(v for k, v in dc.receipt['bias'].items() if k.startswith('objective_lens/'))
    with np.load(folder / 'column-and-landings.npz') as raw:
        center, loss = raw['surface_phase'], raw['loss']
        np.testing.assert_array_equal(raw['source_ids'], model.t.ids)
        np.testing.assert_array_equal(raw['weights'], model.t.weights)
    mask = loss == -1
    weights = model.t.weights[mask]

    def propagate(drive):
        model.t.tables[0] = baseline + drive*derivative
        [(phase, lost)] = advance(model.t, model.full, model.t.input, model.en, model.phi0,
                                  model.loss, 0, [len(model.m.z)-1])
        np.testing.assert_array_equal(lost, loss)
        return phase[mask]

    center = center[mask]
    lower, upper = propagate(-span), propagate(span)
    slope = (upper-lower)/(2*span)
    curvature = (upper+lower-2*center)/(2*span**2)
    sampled = {}

    def evaluate(offset):
        if offset not in sampled:
            phase = center+offset*slope+offset**2*curvature
            sampled[offset] = local_caustic(phase, weights)
        return sampled[offset]['D95_offset_um']

    if evaluate(-span)*evaluate(span) > 0:
        raise ValueError(f"Objective bracket in A does not enclose the D95 waist: {sampled}")
    mid = brentq(evaluate, -span, span, xtol=3e-8)
    native = local_caustic(propagate(mid), weights)
    # The regular quadratic interpolation is only a proposal. Correct any residual
    # directly measured with the full tracker at this current before a new DC solve.
    near = max(span*.015, 1e-4)
    sensitivity = (evaluate(mid+near)-evaluate(mid-near))/(2*near)
    corrected = mid - native['D95_offset_um']/sensitivity
    if abs(corrected)>span or abs(corrected-mid)>.1*span:
        raise ValueError('Native objective proposal could not be refined in its measured bracket')
    output = dict(objective_before_A=reference, proposed_A=reference+corrected,
                  bracket_A=[reference-span, reference+span], current_changes_only='series objective',
                  source_count=len(weights), source_solution=solution.name,
                  native_tangent_check=dict(delta_A=mid, **native), sensitivity_um_per_A=sensitivity,
                  sampled=[dict(delta_A=k, **v) for k,v in sorted(sampled.items())],
                  method='Gun held for proposal only. Axial offset of the weighted D95 minimum from the upper surface, using local slopes. Independent nonlinear re-solve, source launch and two-sided native focus check required.')
    (folder/'objective-waist-search.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Objective proposal', output['proposed_A'], 'A, tangent replay',native, flush=True)
    return output


def check_native_profile(folder, half_window_um=250):
    """Sample both sides on a uniform 1 µm grid, retaining the actual source and fields."""
    record = json.loads((folder/'tracking.json').read_text())
    field = Field(Path(record['solution']),step=record['bore_step_mm'])
    top = record['surface_z_mm']*.001
    offsets = np.arange(-half_window_um,half_window_um+1)
    planes = top+offsets*1e-6
    full = Column(source_and_electrodes(),field,Path(record['gun_source']),record['count'],
                  record['step_mm'],observation_mm=[planes[0]*1000])
    j = full.indices(planes[:1])[0]
    np.testing.assert_allclose(full.m.z[j],planes[0],atol=1e-13,rtol=0)
    [(phase,lost)] = advance(full.t,full.full,full.t.input,full.en,full.phi0,full.loss,0,[j])
    with np.load(folder/'column-and-landings.npz') as saved:
        surface, losses = saved['surface_phase'],saved['loss']
    field_nodes = half_nodes(planes)
    electric = full.f.electric.base_field
    ix = np.clip(np.searchsorted(electric.z,field_nodes,side='right')-1,0,len(electric.z)-2)
    dz = np.diff(electric.z)[ix]
    mesh = SimpleNamespace(z=planes,nodes=field_nodes,radius=np.full(len(planes)-1,np.inf),
                          owner=np.full(len(planes)-1,-1),keys=[],
                          electric=(electric.r,electric.voltage,ix,(field_nodes-electric.z[ix])/dz,dz))
    table = dict(z_local_m=field.localz,xy_m=field.xy,B_zyx_T=field.table(),valid_radius_m=field.receipt['local_radius_m'])
    tracker = ApertureTracker(mesh,table,'cubic',source=full.source)
    drive = cp.zeros((len(field_nodes),20))
    rows = []
    max_error = None

    def record_plane(i,p,loss):
        np.testing.assert_array_equal(loss,losses)
        live = loss==-1
        w = tracker.weights[live]
        s = specimen(p[live],w,tracker.current)
        moments = beam_moments(p,live,tracker.weights)
        return dict(offset_um=int(offsets[i]),D95_nm=s['D95_um']*1000,RMS_radius_nm=moments[0]*1000,
                    axis_50pm_nA=s['axis_50pm_pA']*.001,radial_growth=moments[1]+moments[2],
                    principal_growth=[moments[3],moments[4]])

    rows.append(record_plane(0,phase,lost))
    previous = 0
    for batch in np.array_split(np.arange(1,len(planes)),(len(planes)+9)//10):
        states = advance(tracker,drive,phase,full.en,full.phi0,lost,previous,batch)
        for index,(phase,lost) in zip(batch,states,strict=True):
            rows.append(record_plane(index,phase,lost))
            if offsets[index]==0:
                live = lost==-1
                max_error = float(np.linalg.norm(phase[live,:2]-surface[live,:2],axis=1).max())
        previous = batch[-1]
    if max_error is None or max_error > 1e-13:
        raise ValueError('Native focus replay differs from the specimen landings')
    best = min(rows,key=lambda r:r['D95_nm'])
    rms = min(rows,key=lambda r:r['RMS_radius_nm'])
    if abs(best['offset_um'])==half_window_um or abs(rms['offset_um'])==half_window_um:
        raise ValueError('Axial window does not bracket the measured specimen caustic')
    core = max(r['axis_50pm_nA'] for r in rows)
    result=dict(surface_mm=record['surface_z_mm'],field_sha256=field.receipt['solution_sha256'],
                original_count=record['count'],spacing_um=1,half_window_um=half_window_um,
                convention='Positive z is downstream. Unscattered continuation after the physical upper surface.',
                entrance_replay_error_m=max_error,D95_min=best,RMS_min=rms,
                peak_fixed_axis_50pm_nA=core,peak_core_offsets_um=[r['offset_um'] for r in rows if r['axis_50pm_nA']==core],
                profile=rows)
    (folder/'axial-focus-profile.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Native D95 and RMS minima',best,rms,'surface closure',max_error,flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--span', type=float, default=.30)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    if args.check:
        check_native_profile(HERE/args.label)
    else:
        propose(HERE / args.label, args.span)
