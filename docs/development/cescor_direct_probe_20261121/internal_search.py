"""Seek CESCOR source images and aperture conjugates in the installed axial fields.

This paraxial calculation proposes currents. All acceptance measurements use the
finite-source 3-D field and physical mechanical boundaries in relay_transfer.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from column_current import FACTOR, POINT, fields, pencil_fields
from column import read_gun
from relay_transfer import FOIL, HP1, HP2, RADIUS, SURFACE, TL21, TL22, save


@njit(cache=True)
def pencil_scan(bz, h, idx, entry_k, exit_k, emission, hp2_row=1, hp1_row=3):
    w, v = 1 + 0j, emission - 1j * entry_k
    a, b, c, d = 1.0, 0.0, 0.0, 1.0
    phase = 0.0
    out = np.zeros((len(idx), 9), dtype=np.complex128)
    j = 0
    for i in range(len(h) + 1):
        if j < len(idx) and i == idx[j]:
            out[j] = np.array([w, v, a, b, c, d, phase, entry_k, exit_k])
            if j == hp2_row or j == hp1_row:
                a, b, c, d = 1.0, 0.0, 0.0, 1.0
                phase = 0.0
            j += 1
        if i == len(h):
            break
        k = FACTOR * bz[i]
        f = np.cos(k * h[i])
        g = np.sin(k * h[i]) / k if abs(k) > 1e-12 else h[i]
        w, v = f * w + g * v, -k * k * g * w + f * v
        a, c = f * a + g * c, -k * k * g * a + f * c
        b, d = f * b + g * d, -k * k * g * b + f * d
        phase += k * h[i]
    return out


def search(args):
    seed = json.loads(args.settings.read_text())
    q0 = np.array(seed["currents_A"])
    mid, scan, pupil = args.middle * .001, args.scan * .001, args.pupil * .001
    planes = np.sort(np.array([FOIL[0], HP2, mid, HP1, pupil, scan, SURFACE]))
    guards = np.r_[.652, np.linspace(.915, 1.0882, 13)] if args.topology else np.array([])
    observations = np.unique(np.r_[planes, guards])
    m, t, bg, coils = fields(.35, observations, seed["objective_At"], args.table)
    f = pencil_fields(m, t, bg, coils)
    h, idx = np.diff(m.z), m.z.searchsorted(observations)
    main = observations.searchsorted(planes)
    protected = observations.searchsorted(guards)
    ctr = len(t["xy_m"]) // 2
    ends = np.array([0, -1])
    boundary = np.interp(m.nodes[ends] - m.z[0], t["z_local_m"], t["B_zyx_T"][:, ctr, ctr, 2])
    ek = FACTOR * np.column_stack([bg[ends, 0] + boundary, coils[:, ends, 0].T])
    phase, weight, *_ = read_gun(POINT)
    pos, slope = phase[:, 0] + 1j * phase[:, 1], phase[:, 2] + 1j * phase[:, 3]
    emission = np.vdot(weight * pos, slope) / np.vdot(weight * pos, pos)
    scales = np.array([600] * 4 + [350] * 4 + [190] * 4 + [700] * 3 + [950] * 3 + [700] * 3)
    supply_names = ["probe_tl21_lens", "probe_tl22_lens"]
    reference_amps = np.array([json.loads(str(t["report"]))["currents_A"][n] for n in supply_names])
    currents = q0.copy()
    if args.maps:
        differences, b_fields, end_fields = [], [], []
        for path in args.maps:
            with np.load(path) as table:
                other = json.loads(str(table["report"]))["currents_A"]
                differences.append(np.array([other[n] for n in supply_names]) - reference_amps)
                yy = table["B_zyx_T"]
                center = yy.shape[1] // 2
                field = np.interp(m.nodes - m.z[0], table["z_local_m"], yy[:, center, center, 2])
                old = np.interp(m.nodes - m.z[0], t["z_local_m"], t["B_zyx_T"][:, ctr, ctr, 2])
                b_fields.append((field - old)[1::2])
                end_fields.append((field - old)[ends])
        f = np.column_stack((f, np.linalg.solve(differences, b_fields).T))
        ek = np.column_stack((ek, FACTOR * np.linalg.solve(differences, end_fields).T))
        currents = np.r_[q0, 0., 0.]
        scales = np.r_[scales, 60., 50.]
    spin, scan_index, pupil_index = (list(planes).index(z) for z in (mid, scan, pupil))
    # Relative foil illumination is matched to the physical reference. Keep the source
    # and its two independent Larmor-frame canonical coordinates throughout the fit.
    def optical(q):
        current = np.r_[1.0, q]
        entry, end = ek @ current
        all_obs = pencil_scan(f @ current, h, idx, entry, end, emission, main[1], main[3])
        s = all_obs[main]
        angle = abs((s[-1, 1] + 1j * end * s[-1, 0]) / s[0, 0]) * RADIUS
        return s, angle, all_obs[protected, 0]

    initial, _, _ = optical(currents)
    w_ref = initial[0, 0]
    w0 = abs(w_ref)
    target_slope = args.edge_sign * (initial[-1, 1] + 1j * initial[-1, 8] * initial[-1, 0]) / w_ref
    # The real source image has finite emittance. Use the real projection on
    # the source mode rather than demanding a zero spot from both phase axes.
    def error(x, weight, unit):
        s, _, at_guard = optical(currents + scales * x)
        p = s[:, 0] / w_ref
        slope = (s[-1, 1] + 1j * s[-1, 8] * s[-1, 0]) / w_ref
        hpair, hpupil = s[3], s[pupil_index]
        elm = np.array([
            hpair[3].real / .022, hpupil[3].real / .03,
            (slope.real - target_slope.real) * RADIUS / .012,
            p[-1].real / .30, (p[0].real - 1) / .3,
            weight * p[spin].real / 2.0, weight * p[scan_index].real / 2.0,
            unit * (hpair[2].real + 1), unit * hpair[4].real / 35,
        ])
        parity = np.ones(len(at_guard))
        parity[1:] = -1
        branch = np.maximum(0, .15 - parity * (at_guard / w_ref).real)
        return np.r_[elm, branch, (x - starting) * .00025]

    def describe(x):
        q = currents + scales * x
        s, angle, at_guard = optical(q)
        tl = reference_amps + (q[21:] if args.maps else 0)
        return dict(currents_A=q[:21].tolist(), proposed_tl21_tl22_A=tl.tolist(),
                    objective_At=seed["objective_At"], planes_mm=(planes * 1000).tolist(),
                    A_Bmm_Cperm_D=[s[3, 2].real, s[3, 3].real * 1000, s[3, 4].real, s[3, 5].real],
                    B_to_pupil_mm=s[pupil_index, 3].real * 1000, HP_rotation_deg=np.rad2deg(s[3, 6].real),
                    normalized_source_ray=[(v.real, v.imag) for v in s[:, 0] / s[0, 0]],
                    signed_guard_amplitudes=((at_guard / w_ref).real).tolist(),
                    paraxial_edge_mrad=np.arctan(angle) * 1000, foil_size_ratio=abs(s[0, 0]) / w0)

    print("Seed", describe(np.zeros(len(scales))), flush=True)
    for j, amp in enumerate([0.0, .28, .50, .75, 1.0, 1.0]):
        starting = np.random.default_rng(208 + j).normal(0, amp, len(scales))
        x = starting.copy()
        for k, (waist, unit) in enumerate([(.12, 0), (1, 0), (1, .4)]):
            fit = least_squares(error, x, args=(waist, unit), bounds=(-4.5, 4.5),
                                max_nfev=args.evals, ftol=5e-12, xtol=5e-12, gtol=5e-12, x_scale="jac")
            x = fit.x
            row = describe(x)
            row["residual"] = error(x, waist, unit)[:9].tolist()
            save(f"{args.name}-{j}-{k}", row)
            print(j, k, np.round(row["residual"], 4), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--middle", type=float, default=1000 * (TL21 + TL22) / 2)
    parser.add_argument("--scan", type=float, default=1400)
    parser.add_argument("--pupil", type=float, default=1408.05425036)
    parser.add_argument("--evals", type=int, default=120)
    parser.add_argument("--maps", nargs=2, type=Path, help="Two independent TL perturbations for proposals only")
    parser.add_argument("--topology", action="store_true", help="Protect the upstream branch with signed paraxial-ray heights")
    parser.add_argument("--edge-sign", type=int, choices=(-1, 1), default=1,
                        help="Paraxial marginal-ray sign relative to the search seed")
    search(parser.parse_args())
