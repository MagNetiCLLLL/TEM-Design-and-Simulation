"""Search for a finite two-dimensional relay waist and independently focused specimen.

The local winding responses propose currents. Every accepted case must subsequently
be solved nonlinearly and traced from the emitter in the Cartesian field.
"""

import argparse
import json

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares, root

from material_relay_branches import FACTOR, fast, gun_sigma
from material_relay_fields import SAVE, WORK
from material_relay_response import groups


MODES = ("tl", "objective", "mini", "c3")


def axis(stem, channel, pitch, z):
    with np.load(WORK / f"{stem}-{channel}-bore-{pitch:g}mm.npz") as data:
        return CubicSpline(data["z"], data["coef"][:, 0, 2])(z)


def screen(label, tangent, pair_tangent, derivative_pitch, requested_betas, null_surface=False):
    tracking = json.loads((SAVE / label / "tracking.json").read_text())
    z = np.linspace(.45, tracking["surface_z_mm"] * .001, 2701)
    field = tracking["field"]
    supplies = groups(field["bias"])
    bias = np.array([field["bias"][supplies[name][0]] for name in MODES])
    stem = field["source"].removesuffix(".npz")
    pair_stem = tracking["pair"]["source"].removesuffix(".npz")
    dc = axis(stem, "dc", field["step_mm"], z)
    pair = axis(pair_stem, "dc", field["step_mm"], z)
    slopes = np.array([axis(tangent, k, derivative_pitch, z) for k in MODES])
    pair_slopes = np.array([axis(pair_tangent, k, derivative_pitch, z) for k in ("objective", "mini")])
    x, t, weights = gun_sigma(tracking["gun_source"])
    if len(x) < 1000 or np.min(weights) <= 0:
        raise ValueError("A physical source is required")
    v = t - 1j * FACTOR * dc[0] * x
    xx = np.sum(weights * abs(x)**2)
    xv = np.sum(weights * (x.conj() * v).real)
    epsilon = np.sqrt(xx * np.sum(weights * abs(v)**2) - xv**2)
    incident = (xv + 1j * epsilon) / xx
    source_beta = xx / epsilon
    center = tracking["isolated_pair_fronts"][0]["z_mm"] * .001
    # Jointly vary the front coordinate. This keeps the residual continuous when
    # an early iteration has no front inside the scan-to-Mini interval.
    scales = np.array([40., 3., 45., .3, .016])
    lo = np.r_[[-120., -14., -145., -1.65], 1.421 - center] / scales
    hi = np.r_[[120., 14., 145., 1.65], 1.511 - center] / scales

    def state(q):
        delta = q * scales
        f = dc + np.einsum("i,ij->j", delta[:4], slopes)
        g = pair + np.einsum("i,ij->j", delta[1:3], pair_slopes)
        front = center + delta[-1]
        full = fast(z, f)
        abcd = fast(z, g)
        i = np.searchsorted(z, front) - 1
        frac = (front - z[i]) / (z[i+1] - z[i])
        def sample(table):
            return table[i] + frac * (table[i+1] - table[i])
        a, b, c, d = sample(full)
        ap, bp, _, _ = sample(abcd)
        af, bf, cf, df = full[-1]
        tr = np.array([[af*d-bf*c, -af*b+bf*a], [cf*d-df*c, -cf*b+df*a]])
        pair_d = -abcd[-1, 2] * bp + abcd[-1, 3] * ap
        return full, f, front, a+b*incident, c+d*incident, tr, pair_d

    trials = []
    for beta in requested_betas:
        def residual(q):
            _, f, _, p, t, ((a, b), (c, d)), pair_d = state(q)
            expected = p + 1j * beta * t
            # At a waist t = i*p/beta. The downstream angular correlation must
            # also vanish after transport to the physical upper surface.
            residuals = [expected.real * np.sqrt(source_beta / beta),
                         expected.imag * np.sqrt(source_beta / beta),
                         a*c*beta + b*d/beta, pair_d * 20.]
            if null_surface:
                # A test of a different family, allowing a broad but converging
                # waist at the sample while retaining a pronounced relay crossover.
                residuals.append(f[-1] / .01)
            return np.asarray(residuals)
        seeds = [np.zeros(5), np.array([.65, .4, -.2, -.8, .2]),
                 np.array([-.7, -.45, .7, .8, -.3])]
        for seed in seeds:
            result = least_squares(residual, seed, bounds=(lo, hi), method="trf",
                                   max_nfev=160, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            dq = result.x * scales
            full, f, front, p, t, tr, pair_d = state(result.x)
            radius = abs(full[:, 0] + incident * full[:, 1]) * np.sqrt(xx)
            angular = full[:, 2] + incident * full[:, 3]
            u = full[:, 0] + incident * full[:, 1]
            cov = xx * (u.conj()*angular).real
            iz = np.flatnonzero((cov[:-1] < 0) & (cov[1:] >= 0))
            roots = z[iz] - (z[iz+1]-z[iz])*cov[iz]/(cov[iz+1]-cov[iz])
            curv = xx * (abs(angular)**2 - (FACTOR*f*abs(u))**2)
            row = dict(beta_m=beta, currents_A=dict(zip(MODES, (bias+dq[:4]).tolist())),
                       target_mm=front*1000, residual=residual(result.x).tolist(),
                       front_curvature=float(np.interp(front, z, curv)),
                       surface_curvature=float(curv[-1]), field_at_front_T=float(np.interp(front,z,f)),
                       front_radius_nm=float(abs(p)*np.sqrt(xx)*1e9), sample_radius_nm=float(radius[-1]*1e9),
                       meridional_minima_mm=(1000*roots).tolist(), tangent_step=(dq[:4]/scales[:4]).tolist())
            trials.append(row)
    trials.sort(key=lambda case: np.linalg.norm(case["residual"]))
    return dict(reference=label, tangents=[tangent, pair_tangent], source_particles=len(weights),
                source_emittance_m_rad=float(epsilon),
                description="Local field sensitivity proposals; re-solve and re-emit for verification.", trials=trials)


def native_match(label, full_tangent, pair_tangent, pitch, targets, *, d95=False, bracket=False):
    """Correct the transport sensitivity estimate at both native particle measurements."""
    from pathlib import Path
    from material_relay_column import Column, Field, source_and_electrodes
    from relay_transfer import advance

    folder = SAVE / label
    r = json.loads((folder / "tracking.json").read_text())
    z = np.linspace(.45, r["surface_z_mm"]*.001, 10001)
    stem, pair_stem = (r[k]["source"].removesuffix(".npz") for k in ("field", "pair"))
    actual_pitch = r["field"]["step_mm"]
    base, pair = (axis(s, "dc", actual_pitch, z) for s in (stem, pair_stem))
    derivatives = np.array([axis(full_tangent, k, pitch, z) for k in MODES])
    pair_derivatives = np.array([axis(pair_tangent, k, pitch, z) for k in ("objective", "mini")])
    bias = r["field"]["bias"]
    q0 = np.array([bias[groups(bias)[k][0]] for k in MODES])
    # Use the narrow waist even when the first provisional point lies inside the
    # winding. Construction of the comparison never moves a physical station.
    w = min((v for v in r["waists_mm"] if 1370 < v < 1512.5), key=lambda v: abs(v-targets[0]))
    field = Field(WORK / (stem+".npz"), step=actual_pitch)
    col = Column(source_and_electrodes(), field, Path(r["gun_source"]), r["count"], r["step_mm"], observation_mm=[w])
    stop = col.indices([w*.001])
    near = advance(col.t, col.full, col.t.input, col.en, col.phi0, col.loss, 0, stop)[0]
    with np.load(folder / "column-and-landings.npz") as data:
        surface = data["surface_phase"], data["loss"]
    def sigma(state, location):
        phase, loss = state
        live = loss == -1
        wt = col.t.weights[live].copy()
        wt /= wt.sum()
        if len(wt) < 1000:
            raise ValueError("Too few original rays to measure the beam ellipse")
        x, y, tx, ty = (phase[live, :4]-np.average(phase[live, :4], axis=0, weights=wt)).T
        p = x+1j*y
        u = tx+1j*ty-1j*FACTOR*np.interp(location, z, base)*p
        return np.real((np.array([p, u])*wt)@np.array([p, u]).conj().T)
    def matrix(table, location):
        j = min(np.searchsorted(z, location, side="right")-1, len(z)-2)
        part = (location-z[j])/(z[j+1]-z[j])
        return (table[j]+part*(table[j+1]-table[j])).reshape(2, 2)
    reference = fast(z, base)
    def source(state, location):
        mat = np.linalg.inv(matrix(reference, location))
        measured = sigma(state, location)
        return mat@measured@mat.T

    src_w, src_s = source(near, w*.001), source(surface, z[-1])
    if d95:
        phase, loss = surface
        live = loss == -1
        weights = col.t.weights[live]
        x, y, tx, ty = (phase[live,:4]-np.average(phase[live,:4],axis=0,weights=weights)).T
        pos = x+1j*y
        slopes = tx+1j*ty-1j*FACTOR*base[-1]*pos
        sample_rays = np.linalg.solve(reference[-1].reshape(2,2), np.stack([pos,slopes]))
        def quantile_response(sm):
            positions = sm[0,0]*sample_rays[0]+sm[0,1]*sample_rays[1]
            rates = sm[1,0]*sample_rays[0]+sm[1,1]*sample_rays[1]
            radii = abs(positions)
            order = np.argsort(radii)
            owner = order[np.searchsorted(np.cumsum(weights[order]), .95*weights.sum())]
            return 2*(positions[owner].conj()*rates[owner]).real/radii[owner]
    amps = np.array([28., 2., 28., .20])
    sign = sigma(near, w*.001)[1, 1]
    results = []
    for target in targets:
        position = target*.001
        def quantities(delta):
            dq = delta*amps
            f = base + np.einsum("i,ij->j", dq, derivatives)
            g = pair + np.einsum("i,ij->j", dq[1:3], pair_derivatives)
            full, isolated = fast(z, f), fast(z, g)
            mat = matrix(full, position)
            sm = full[-1].reshape(2, 2)
            ww, ss = mat@src_w@mat.T, sm@src_s@sm.T
            pair_at = matrix(isolated, position)
            d = (isolated[-1].reshape(2, 2)@np.linalg.inv(pair_at))[1, 1]
            kw, ks = FACTOR*np.array([np.interp(position,z,f), f[-1]])
            grad95 = quantile_response(sm) if d95 else 0.
            return ww, ss, d, ww[1,1]-kw**2*ww[0,0], ss[1,1]-ks**2*ss[0,0], f[-1], grad95
        if bracket:
            # The D95 order statistic can change owning ray. Follow the optical
            # constraints first, then bracket its axial derivative along that curve.
            solved = {}
            def registered(offset):
                if offset not in solved:
                    delta = np.array([0., offset/amps[1], 0., 0.])
                    def optical(v):
                        delta[[0,2]] = v
                        ww, _, d, *_ = quantities(delta)
                        return [ww[0,1]/(sign*.001), d/.012]
                    guess = solved[min(solved,key=lambda x:abs(x-offset))][0][[0,2]] if solved else np.zeros(2)
                    fit = root(optical, guess, tol=1e-10)
                    errors = optical(fit.x)
                    if max(abs(np.array(errors)))>1e-5:
                        raise ValueError("Lost the narrow relay branch while bracketing objective focus")
                    solved[offset] = delta.copy(), quantities(delta)
                return solved[offset]
            def gradient(o):
                return registered(float(o))[1][-1] * 1e6
            offsets = np.linspace(-.5,.5,151)
            signed = np.array([gradient(o) for o in offsets])
            for j in np.flatnonzero(signed[:-1]*signed[1:] <= 0):
                off = brentq(gradient, offsets[j], offsets[j+1], xtol=1e-11)
                delta, (ww, ss, dd, cw, cs, fs, g) = registered(off)
                results.append(dict(target_mm=target, family="fixed_C3_D95_bracket",
                                    residual=[ww[0,1]/(sign*.001), dd/.012, g/1e-8],
                                    currents_A=dict(zip(MODES,(q0+amps*delta).tolist())),
                                    front_curvature=float(cw), surface_curvature=float(cs),
                                    radii_nm=(1e9*np.sqrt(np.maximum([ww[0,0],ss[0,0]],0))).tolist(),
                                    Bz_surface_T=float(fs), sample_D95_gradient_nm_per_mm=float(g*1e6),
                                    objective_offset_A=float(off)))
            continue
        families = ("surface_D95_and_RMS", "surface_D95_field_zero", "surface_D95_hold_mini") if d95 else ("surface_field_zero", "hold_mini")
        for family in families:
            def residual(delta):
                ww, ss, d, _, _, fs, grad95 = quantities(delta)
                # Scale each optical residual independently. In particular the
                # specimen can have a much larger radius than the relay waist.
                if d95:
                    fourth = {"surface_D95_and_RMS":ss[0,1]/1e-15,
                              "surface_D95_field_zero":fs/.001, "surface_D95_hold_mini":delta[2]}[family]
                    return [ww[0,1]/(sign*.001), d/.012, grad95/1e-8, fourth]
                return [ww[0,1]/(sign*.001), d/.012, ss[0,1]/1e-15, fs/.001 if family=="surface_field_zero" else delta[2]]
            for seed in (np.zeros(4), np.array([.2,.3,-.4,-.1])):
                res = least_squares(residual, seed, bounds=(-3.,3.), max_nfev=180, gtol=1e-10, ftol=1e-11, xtol=1e-11)
                ww, ss, d, cw, cs, fs, grad95 = quantities(res.x)
                results.append(dict(target_mm=target, family=family, residual=res.fun.tolist(),
                                    currents_A=dict(zip(MODES,(q0+amps*res.x).tolist())),
                                    front_curvature=float(cw), surface_curvature=float(cs),
                                    radii_nm=(np.sqrt(np.maximum([ww[0,0],ss[0,0]],0))*1e9).tolist(),
                                    Bz_surface_T=float(fs), sample_D95_gradient_nm_per_mm=float(grad95*1e6) if d95 else None))
    results.sort(key=lambda a: np.linalg.norm(a["residual"]))
    return dict(reference=label, tangents=[full_tangent, pair_tangent], source_particles=r["count"],
                native_relay_mm=w, trials=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("tangent")
    parser.add_argument("pair_tangent")
    parser.add_argument("--pitch", type=float, default=.5)
    parser.add_argument("--betas", type=float, nargs="+", default=[.0003, .001, .004, .015, .04, .1])
    parser.add_argument("--null-surface", action="store_true")
    parser.add_argument("--native-targets", type=float, nargs="+")
    parser.add_argument("--d95", action="store_true")
    parser.add_argument("--bracket", action="store_true")
    args = parser.parse_args()
    if args.native_targets:
        report = native_match(args.label, args.tangent, args.pair_tangent, args.pitch, args.native_targets,
                              d95=args.d95 or args.bracket, bracket=args.bracket)
        suffix = "-registered-d95" if args.bracket else ("-native-d95" if args.d95 else "-native")
    else:
        report = screen(args.label, args.tangent, args.pair_tangent, args.pitch, args.betas, args.null_surface)
        suffix = "-surface-null" if args.null_surface else ""
    (WORK / f"{args.label}-crossover-search{suffix}.json").write_text(json.dumps(report, indent=2))
    for row in report["trials"][:4]:
        print(row, flush=True)
