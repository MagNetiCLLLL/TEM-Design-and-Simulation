"""Illuminate the installed C2 aperture and retune the normal nanoprobe column."""

import argparse
import json

import numpy as np
from scipy.stats import qmc
from scipy.optimize import least_squares
from numba import njit

from condenser_normal import (
    ACTIVE, FACTOR, FIELD, FOIL, OWNER, RADIUS, START, SURFACE, WORK, fields, prescreen,
    reference, scan_inputs, spans_mm,
)


def winding_groups(split_c2=False):
    owners = OWNER[ACTIVE]
    c1, c2, c3, mini = [
        np.flatnonzero(owners == key)
        for key in ("condenser_lens_1", "condenser_lens_2", "condenser_lens_3", "mini_condenser")
    ]
    c2_groups = np.split(c2, len(c2)) if split_c2 else [c2]
    return [*np.split(c1, len(c1)), *c2_groups, c3, mini]


def controls(gains, split_c2=False):
    q = np.array(reference()["currents_A"])
    for group, gain in zip(winding_groups(split_c2), gains, strict=True):
        q[group] *= gain
    return q


def axis_inputs(step, split_c2=False):
    seed = reference()
    q = np.array(seed["currents_A"])
    _, h, z, events, bands, slope, source_radius = scan_inputs(False, step, illuminate=True)
    mesh, table, bg, coils = fields(step, np.r_[spans_mm().ravel()*.001, FOIL, SURFACE], seed["objective_At"], FIELD)
    center = len(table["xy_m"]) // 2
    bz = np.interp(mesh.nodes - START, table["z_local_m"], table["B_zyx_T"][:, center, center, 2])
    variable = [coils[group, :, 0].T @ q[group] for group in winding_groups(split_c2)]
    fixed = bg[:, 0] + bz + coils[:, :, 0].T @ q - np.sum(variable, axis=0)
    nodes = np.r_[0, 1:len(fixed):2, len(fixed)-1]
    profile = np.ascontiguousarray(np.column_stack([v[nodes] for v in [fixed, *variable]]))
    return profile, h, z, events, bands, slope, source_radius


def screen(args):
    inputs = axis_inputs(args.step, args.split)
    dim = len(winding_groups(args.split))
    unit = qmc.Sobol(dim, scramble=True, seed=211130).random_base2(args.power)
    gains = unit.copy()
    gains[:, :4] = unit[:, :4] * 3.0
    gains[:, 4:-2] = .35 + unit[:, 4:-2] * 2.75
    gains[:, -2] = .08 + .39*unit[:, -2]
    gains[:, -1] = 1.
    if args.split:
        reference_c2 = np.abs(np.asarray(reference()["currents_A"])[OWNER[ACTIVE] == "condenser_lens_2"])
        gains[:, 4:8] = 2200 * unit[:, 4:8] / reference_c2
    if args.common:
        gains[:, :4] = .25 + unit[:, :1] * 2.75
        if not args.split:
            gains[:, 4] = 2.5 + 5 * unit[:, 4]
    if args.sparse:
        q0 = np.abs(np.array(reference()["currents_A"])[:8])
        amps = np.array([3600,3800,2500,2200,1700,1250,950,750])*unit[:,:8]**2
        mode = np.arange(len(amps)) % 16
        amps[:, 2:7] *= .05
        amps[mode < 8, 2:7] = 0.
        for j in range(4):
            use = mode % 4 == j
            amps[use, 4+j] = unit[use, 4+j]*np.array([1700,1250,950,750])[j]
        amps[mode % 2 == 0, 1] = 0.
        gains[:,:8] = amps/q0
    if args.first:
        gains[:, 4:-2], gains[:, -2] = 1.542, .260
    result = prescreen(gains, *inputs)
    np.savez_compressed(WORK / (args.name + ".npz"), gains=gains, result=result)
    ok = (result[:, 0] == 1) & (result[:, 1] == 0) & (result[:, 7] == 0)
    print("Gap pattern / illuminated / largest radius fraction", ok.sum(), (ok & (result[:, 5] > 1.1)).sum(),
          result[ok, 5].max(initial=0), flush=True)
    ix = np.flatnonzero(ok)
    if len(ix):
        score = (np.maximum(0, 1.6-result[ix, 5])*50 + abs(result[ix, 3]-30)
                 + .05*result[ix, 4] + 3*result[ix, 8] + .12*np.maximum(0, abs(result[ix, 2]-799)-20))
        for j in ix[np.argsort(score)[:8]]:
            print(gains[j].round(4), result[j, :9].round(3), flush=True)


@njit(cache=True)
def marginal(gains, f, h, events, initial, source_radius, gap_index):
    k0 = FACTOR * (f[0, 0] + np.dot(f[0, 1:], gains))
    w, v = 1+0j, initial - 1j*k0
    wf, wg, vg = 1+0j, 1+0j, 0j
    for i in range(len(h)):
        k = FACTOR * (f[i+1, 0] + np.dot(f[i+1, 1:], gains))
        a = k*h[i]
        c = np.cos(a)
        s = np.sin(a)/k if abs(k) > 1e-14 else h[i]
        w, v = c*w + s*v, -k*k*s*w + c*v
        if i+1 == events[0] or (i+1 == events[1] and abs(w) > abs(wf)):
            wf = w
        if i == gap_index:
            wg, vg = w, v
    ke = FACTOR * (f[-1, 0] + np.dot(f[-1, 1:], gains))
    slope = v + 1j*ke*w
    return np.array([abs(wf)*source_radius/RADIUS, 1e3*np.arctan(abs(slope/wf)*RADIUS),
                     (wg*vg.conjugate()).real/max(abs(vg)**2, 1e-16),
                     (w*slope.conjugate()).real/max(abs(slope)**2, 1e-16),
                     RADIUS*abs(w/wf)*1e6])


def fit_axis(args):
    inputs = axis_inputs(args.step, True)
    f, h, z, faces, _, initial, radius = inputs
    q0 = np.array(reference()["currents_A"])
    scale = np.r_[1000/abs(q0[:8]), 1., 1.]
    hi = np.r_[[5.3]*4, [3.2]*4, .48, 1.12]
    lo = np.r_[-hi[:8] if args.free else np.zeros(8), .055, .88]
    with np.load(WORK / (args.fit + ".npz")) as raw:
        old, survey = raw["gains"], raw["result"]
    good = (survey[:, 0] == 1) & (survey[:, 1] == 0) & (survey[:, 6] == 1) & (survey[:, 7] == 0)
    indices = np.flatnonzero(good & (survey[:, 2] > 785))
    score = -survey[indices, 5] + .004*abs(survey[indices, 3]-30)
    choices = indices[np.argsort(score)[:40:3]]
    results = []
    for row, idx in enumerate(choices):
        start = np.clip(old[idx]/scale, lo+1e-7, hi-1e-7)
        zg = np.searchsorted(z, np.clip(survey[idx, 2], 797, 820)*.001)
        def residual(x):
            a = marginal(x*scale, f, h, faces, initial, radius, zg)
            b = []
            if args.fixed_c1:
                first=x*scale
                first[4:]=[1.542]*4+[.26,1.]
                b = [marginal(first, f, h, faces, initial, radius, zg)[0]-1.90]
            return np.r_[(a[0]-1.83), (a[1]-30)/12, a[2]/.012, a[3]/.00035, b, .002*(x-start)]
        sol = least_squares(residual, start, bounds=(lo,hi), max_nfev=160)
        g = sol.x*scale
        r = prescreen(g[None,:], *inputs)[0]
        results.append(dict(gains=g.tolist(), screen=r.tolist(), residual=float(np.linalg.norm(sol.fun[:4]))))
        (WORK/(args.name+".json")).write_text(json.dumps(results,indent=2))
        print(row, r[:9].round(3), g[-2:].round(5), flush=True)


def upstream(args):
    step = args.step
    inputs = axis_inputs(step, True)
    f, h, z, events, bands, initial, radius = inputs
    q0 = np.array(reference()["currents_A"])
    scale = np.r_[1000/abs(q0[:8]), 1., 1.]
    rand = np.random.default_rng(211121)
    with np.load(WORK / (args.fit + ".npz")) as raw:
        original, survey = raw["gains"], raw["result"]
    good = (survey[:,0]==1)&(survey[:,1]==0)&(survey[:,6]==1)&(survey[:,7]==0)&(survey[:,2]>785)
    ix = np.flatnonzero(good)
    best = ix[np.argsort(survey[ix,5])[-80:]]
    seeds = original[best].copy()
    iterations = 105 if args.free else 38
    for it in range(iterations):
        n = 8192
        g = seeds[rand.integers(len(seeds), size=n)].copy()
        delta = rand.normal(size=(n,9))*rand.choice([.008,.04,.16,.5,1.2],size=(n,1))*np.array([1,1,1,1,.8,.5,.20,.20,.14])
        g[:,:9] += delta*scale[:9]
        limits = np.r_[[5.2]*4,[3.2,2.8,1.4,1.2]]*scale[:8]
        g[:,:8] = np.clip(g[:,:8], -limits if args.free else 0, limits)
        g[:,-2] = np.clip(g[:,-2], .07,.45)
        g = np.concatenate([seeds,g])
        s = prescreen(g,*inputs)
        ok = (s[:,0]==1)&(s[:,1]==0)&(s[:,6]==1)&(s[:,7]==0)&(s[:,2]>785)&(s[:,2]<824)
        ix = np.flatnonzero(ok)
        # Reward radii in the populated part of the source; retain diverse accessible angles.
        score = (np.minimum(s[:,5],1.95) - .0015*np.abs(s[:,3]-30)
                 - .006*np.maximum(0,8-s[:,3])-.005*s[:,8])
        first_fill = None
        if args.fixed_c1:
            first = g.copy()
            first[:,4:] = [1.542]*4+[.26,1.]
            stop = events[-1]
            first_fill = prescreen(first, f[:stop+2], h[:stop], z[:stop], events, bands, initial, radius)[:,5]
            score = (np.minimum(s[:,5],1.88) + np.minimum(first_fill,1.88)
                     - .002*np.abs(s[:,3]-30) -.009*s[:,8] -.0002*s[:,4])
        top = ix[np.argsort(-score[ix])]
        _, kept = np.unique(np.round(g[top,:8]/scale[:8], 1), axis=0, return_index=True)
        chosen = top[np.sort(kept)[:80]]
        seeds = g[chosen]
        if it % 5 == 0 or it == iterations-1:
            np.savez_compressed(WORK/(args.name+".npz"),gains=seeds,result=s[chosen],
                                initial_fill=(first_fill[chosen] if first_fill is not None else []))
            print(it, s[chosen[0],:9].round(3), (seeds[0]/scale).round(3), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="illumination-winding-screen")
    parser.add_argument("--power", type=int, default=18)
    parser.add_argument("--step", type=float, default=.7)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--first", action="store_true")
    parser.add_argument("--common", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--fit")
    parser.add_argument("--upstream", action="store_true")
    parser.add_argument("--free", action="store_true")
    parser.add_argument("--fixed-c1", action="store_true")
    args = parser.parse_args()
    (upstream if args.upstream else fit_axis if args.fit else screen)(args)
