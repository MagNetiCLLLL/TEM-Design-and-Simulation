"""Screen direct-waist branches and replay them with the original finite source."""

import argparse
import json

import numpy as np
from numba import njit
from scipy.optimize import brentq

from column_current import ACTIVE, FACTOR, OWNER, START, SURFACE
from illuminated_focus import column_planes
from joint_pivot_tuning import OUT, focal_planes, stations
from normal_working_point import CACHE, EVIDENCE
from refocus_a2 import objective_receipt
from relay_transfer import Native
from trace_normal_joint_field import beam_moments
from tune import waist_proposal


REFERENCE = json.loads((EVIDENCE / "normal-objective-probe.json").read_text())
NI = REFERENCE["after"]["objective_At"]


def model_at(amps, ni, count, step, half=8, extra=()):
    path = (
        CACHE / "normal-joint-field.npz" if amps == 10 else OUT / f"tl12-{amps:.10g}A-field-{half}.npz"
    )
    if amps != 10:
        with np.load(path) as data:
            report = json.loads(str(data["report"]))
        expected = {**REFERENCE["after"]["corrector_A"], "adapter_lens": 10.0, "probe_tl12_lens": amps}
        if (
            expected != report["currents_A"]
            or report["mesh_id"]
            != json.loads((EVIDENCE / "normal-current-working-point.json").read_text())["mesh_id"]
        ):
            raise ValueError("The requested nonlinear working point has a different current or mesh")
    objective_receipt(-abs(ni))
    m = Native(path, step, count, ni, np.unique(np.r_[column_planes(), extra]))
    names, drives = zip(*REFERENCE["fixed_windings"])
    np.testing.assert_array_equal(names, OWNER[ACTIVE])
    q = np.array(drives)
    expected_covered = np.isin(names, ["adapter_lens", "probe_tl12_lens"])
    np.testing.assert_array_equal(expected_covered, m.m.fem_round_windings)
    q[expected_covered] = 0
    if len(np.unique(m.t.ids[m.t.weights > 0])) != count or count < 1000:
        raise ValueError("Invalid original source population")
    return m, q


def switch_objective(m, ni):
    objective_receipt(-abs(ni))
    m.ni = ni
    m.bg = m.m.base("armco", ni, "gap-fine", 8)


def mono_roots(z, values):
    ids = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))
    return z[ids] - values[ids] * (z[ids + 1] - z[ids]) / (values[ids + 1] - values[ids])


def surface_metrics(m, phase, loss):
    live = loss == -1
    p, w = phase[live], m.t.weights[live]
    center = np.average(p[:, :2], weights=w, axis=0)
    r = np.linalg.norm(p[:, :2] - center, axis=1)
    if not np.all(w == w[0]):
        raise ValueError("Source quantile assumes equal quadrature weights")
    i = int(np.ceil(0.95 * len(r))) - 1
    drift, drms = waist_proposal(p, w)
    axis = np.linalg.norm(p[:, :2], axis=1)
    return dict(
        D95_um=float(np.partition(r, i)[i] * 2e6),
        rms_um=float(np.sqrt(np.average(r * r, weights=w)) * 1e6),
        centroid_um=(center * 1e6).tolist(),
        drift_D95_mm=1000 * drift,
        drift_RMS_mm=1000 * drms,
        alpha95_mrad=float(1000 * np.arctan(np.partition(np.linalg.norm(p[:, 2:4], axis=1), i)[i])),
        axis_50nm_nA=float(m.t.current * 1e9 * w[axis <= 50e-9].sum()),
        axis_50pm_pA=float(m.t.current * 1e12 * w[axis <= 50e-12].sum()),
        specimen_uA=float(m.t.current * 1e6 * w.sum()),
        surviving=int(live.sum()),
    )


def static(m, q, *, full=False):
    front = focal_planes(m, q)
    roots = front["mini_and_objective"]
    if not roots:
        return dict(front_mm=front)
    target = min(roots, key=lambda v: abs(v - 1490.0))
    sample = stations(
        m, np.r_[np.arange(1345, 1599.19, 0.5 if full else 1.0), target, SURFACE * 1000] * 0.001
    )
    field = m.field(q)
    # Chunked capture avoids retaining large populations at hundreds of planes.
    from relay_transfer import advance

    j, phase, loss = 0, m.t.input, m.loss
    stats = []
    previous = None
    for idx in np.array_split(m.indices(sample), (len(sample) + 15) // 16):
        data = advance(m.t, field, phase, m.en, m.phi0, loss, j, idx)
        for phase, loss in data:
            live = loss == -1
            if previous is not None and not np.array_equal(live, previous):
                raise ValueError("A changing contact cohort invalidates the local waist scan")
            previous = live
            stats.append(beam_moments(phase, live, m.t.weights))
        j = idx[-1]
    z = sample * 1000
    rows = np.asarray(stats)
    radial = rows[:, 1] + rows[:, 2]
    r = {
        key: mono_roots(z, s).tolist()
        for key, s in (
            ("radial", radial),
            ("x", rows[:, 1]),
            ("y", rows[:, 2]),
            ("principal_low", rows[:, 3]),
            ("principal_high", rows[:, 4]),
        )
    }
    i = int(np.argmin(abs(z - target)))
    return dict(
        front_mm=front,
        waist_mm=r,
        radial_cov_at_front_um_mrad=float(np.interp(target, z, radial)),
        rms_at_front_um=float(rows[i, 0]),
        surface=surface_metrics(m, phase, loss),
    )


@njit(cache=True)
def transfer(z, bz):
    n = len(z)
    m = np.zeros((n, 4))
    m[0] = [1.0, 0.0, 0.0, 1.0]
    for i in range(1, n):
        h = z[i] - z[i - 1]
        k = FACTOR * 0.5 * (bz[i] + bz[i - 1])
        c = np.cos(k * h)
        s = np.sinc(k * h / np.pi) * h
        a, b, g, d = m[i - 1]
        m[i] = [c * a + s * g, c * b + s * d, -k * k * s * a + c * g, -k * k * s * b + c * d]
    return m


def predictor(args):
    # This linearized search only proposes physical current brackets for the full 3-D solver.
    start = 1.310
    z = np.linspace(start, SURFACE, 1451)
    m, q = model_at(10, NI, 4096, 0.12, extra=[start])
    with np.load(m.path) as f:
        mid = len(f["xy_m"]) // 2

        def axis_at(f, mid):
            return np.interp(z - START, f["z_local_m"], f["B_zyx_T"][:, mid, mid, 2])

        baseline = axis_at(f, mid)
    if len(args.anchors) > 1:
        a0, a1 = args.anchors
        fields = []
        for v in args.anchors:
            if v == 10:
                fields.append(baseline)
            else:
                with np.load(OUT / f"tl12-{v:.10g}A-field-8.npz") as f:
                    fields.append(axis_at(f, len(f["xy_m"]) // 2))
        slope = (fields[1] - fields[0]) / (a1 - a0)
        bias = fields[0] - a0 * slope
    else:
        with np.load(CACHE / "profiles-5-170-edge/probe_tl12_lens.npz") as f:
            slope = np.interp(z * 1000, f["z_mm"], f["bz_t"], left=0, right=0)
        bias = baseline - 10 * slope
    sections = {}
    if args.section:
        for a in [1.0, 10.0, 30.0, 80.0, 145.0, 220.0, 260.0, 300.0, 350.0]:
            spacing = "0.58" if a == 10 else "0.82"
            with np.load(CACHE / "sections" / f"probe_tl12_lens-{a:g}-{spacing}-170.npz") as f:
                sections[a] = np.interp(z * 1000, f["z_mm"], f["bz_t"], left=0, right=0)
        sections[0.0] = 0.0 * z
        keys = np.array(sorted(sections))
    records = []
    mini = OWNER[ACTIVE] == "mini_condenser"
    objective_values = args.objective or [
        -6000.0,
        -9000.0,
        -11000.0,
        -12000.0,
        -13000.0,
        -14000.0,
        NI,
        -15000.0,
        -15500.0,
        -16000.0,
        -17000.0,
        -18000.0,
        -19000.0,
        -20000.0,
        -21500.0,
        -24000.0,
    ]
    for ni in objective_values:
        switch_objective(m, ni)
        f, ((p, loss),) = m.replay(q, [start])
        live = loss == -1
        points = p[live, :4] - np.average(p[live, :4], axis=0, weights=m.t.weights[live])
        x = points[:, 0] + 1j * points[:, 1]
        t = points[:, 2] + 1j * points[:, 3]
        bg = np.interp(z, m.m.nodes, f[:, 0].get())
        # Same magnetic source, with only the downstream pair retained for its front focus.
        iso = m.bg - m.m.group[0] + np.einsum("i,ijk->jk", q[mini], m.coils[mini])
        iso = np.interp(z, m.m.nodes, iso[:, 0])
        mat = transfer(z, iso)
        d = -mat[-1, 2] * mat[:, 1] + mat[-1, 3] * mat[:, 0]
        roots = sorted(np.r_[mono_roots(z, d), mono_roots(z, -d)])
        roots = [r for r in roots if r > 1.42]
        if not roots:
            continue
        front = min(roots, key=lambda v: abs(v - 1.49))

        def profile(amps, *, sample=False):
            total = bg + bias + amps * slope
            if sections:
                j = int(np.clip(np.searchsorted(keys, abs(amps), side="right") - 1, 0, len(keys) - 2))
                a, b = keys[j : j + 2]
                delta = sections[a] + (abs(amps) - a) / (b - a) * (sections[b] - sections[a])
                total = bg + baseline - sections[10.0] + np.sign(amps) * delta
            v = t - 1j * FACTOR * total[0] * x
            var = np.mean(abs(x) ** 2), np.real(np.vdot(x, v)) / len(x), np.mean(abs(v) ** 2)
            a, b, c, d = transfer(z, total).T
            cov = a * c * var[0] + (a * d + b * c) * var[1] + b * d * var[2]
            if sample:
                landing = a[-1] * x + b[-1] * v
                slope_at_sample = c[-1] * x + d[-1] * v + 1j * FACTOR * total[-1] * landing
                out = np.column_stack(
                    (landing.real, landing.imag, slope_at_sample.real, slope_at_sample.imag)
                )
                return 1000 * waist_proposal(out, m.t.weights[live])[0]
            return cov * 1e9, np.sqrt(a * a * var[0] + 2 * a * b * var[1] + b * b * var[2]) * 1e6

        currents = np.linspace(-550.0, 550.0, 221)

        def error(a):
            return float(np.interp(front, z, profile(a)[0]))

        vals = np.array([error(a) for a in currents])
        for j in np.flatnonzero(vals[:-1] * vals[1:] < 0):
            amps = brentq(error, currents[j], currents[j + 1])
            cov, r = profile(amps)
            waists = mono_roots(z, cov)
            direct = bool(
                len(waists)
                and min(abs(waists - front)) < 0.0006
                and not np.any((waists > 1.350) & (waists < front - 0.001))
            )
            if not direct and not args.all_roots:
                continue
            item = dict(
                ni=ni,
                tl12_A=round(amps, 3),
                front_mm=round(front * 1000, 3),
                radial_waists_mm=(waists * 1000).round(3).tolist(),
                surface_cov=round(cov[-1], 4),
                surface_rms_um=round(r[-1], 4),
                direct=direct,
                drift_D95_mm=round(profile(amps, sample=True), 4),
            )
            records.append(item)
            print(item, flush=True)
    (
        OUT
        / f"paraxial-proposals-{objective_values[0]:g}-{objective_values[-1]:g}{'-section' if args.section else ''}.json"
    ).write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("predict", "static"))
    p.add_argument("--amps", type=float, default=10.0)
    p.add_argument("--objective", nargs="+", type=float)
    p.add_argument("--anchors", nargs="+", type=float, default=[10.0])
    p.add_argument("--count", type=int, default=4096)
    p.add_argument("--step", type=float, default=0.12)
    p.add_argument("--half", type=int, default=8)
    p.add_argument("--all-roots", action="store_true")
    p.add_argument("--section", action="store_true")
    args = p.parse_args()
    if args.stage == "predict":
        predictor(args)
    else:
        ns = args.objective or [NI]
        m, q = model_at(args.amps, ns[0], args.count, args.step, args.half)
        for ni in ns:
            switch_objective(m, ni)
            r = static(m, q)
            print(args.amps, ni, r, flush=True)
            path = (
                OUT
                / f"static-{args.amps:.10g}A-{abs(ni):.10g}-{args.count}-{args.step:g}-{args.half}.json"
            )
            path.write_text(json.dumps(r, indent=2, allow_nan=False))
