"""Screen the nonlinear relay with the physical aperture and both upstream beam cohorts."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
from scipy.interpolate import CubicSpline

from column_trace import BoreTracker, covariance, crossings, optics_mesh, values, wq
from gun_images import HERE, GUN
from search_pair import axes, anchor, transfer, FACTOR, STATIONS, PACKS
from volume import BASE, CACHE

HANDOFF = 1.04


def interval_roots(z, M, x, v, w, lo, hi):
    if len(w) < 4:
        return []
    w = w / w.sum()
    x = x - np.sum(w * x)
    v = v - np.sum(w * v)
    xx = w @ abs(x) ** 2
    vv = w @ abs(v) ** 2
    xv = w @ (x.conj() * v).real
    used = (z >= lo) & (z <= hi)
    a, b, c, d = M[used].T
    return crossings(z[used] * 1000, a * c * xx + (a * d + b * c) * xv + b * d * vv)


def packet(c1, c2, ref, z, fields, x, v, w, e, st, limit, cohort):
    k = fields[0] + (c1 - ref[0]) * fields[1] + (c2 - ref[1]) * fields[2]
    M = transfer(z, k)

    def pos(i):
        return M[i, 0] * x + M[i, 1] * v

    alive = np.ones(len(w), bool)
    for plane, rad in ((0, 0.0005), (1, 0.0005)):
        alive &= abs(pos(st[plane])) <= rad
    # Interception by the 2.88 mm column wall at sampled nodes is screened
    # here and measured at all installed surfaces in each final exact run.
    for j in np.arange(st[2], st[3], max(1, (st[3] - st[2]) // 15)):
        alive &= abs(pos(j)) < 0.00288
    pre = interval_roots(z, M, x[alive], v[alive], w[alive], 0.45, 0.7649)
    for plane, rad in ((3, 0.00005), (4, 0.00005), (5, 0.002), (6, 0.002)):
        alive &= abs(pos(st[plane])) <= rad
    if alive.sum() < 8:
        return
    post = interval_roots(z, M, x[alive], v[alive], w[alive], 0.7651, HANDOFF)
    roots = pre + post
    same = interval_roots(z, M, x[alive], v[alive], w[alive], 0.45, HANDOFF)

    def distance(a):
        if len(a) != 2:
            return 1000.0
        return max(450 - a[0], a[0] - 650, 750 - a[1], a[1] - 830, 0.0)

    deviations = {
        "both": max(distance(roots), distance(same)),
        "transmitted": distance(same),
        "live": distance(roots),
    }
    if deviations[cohort] > limit:
        return
    a, b, c, d = M[-1]
    k1 = k[-1]
    rot = np.exp(1j * np.diff(z) @ k)
    xy = (a * x[alive] + b * v[alive]) * rot
    vv = (c * x[alive] + d * v[alive]) * rot + 1j * k1 * xy
    phase = np.column_stack((xy.real, xy.imag, vv.real, vv.imag, np.zeros(len(xy))))
    rim = np.maximum(abs(pos(st[3])[alive]), abs(pos(st[4])[alive]))
    return (c1, c2, roots, same), phase, w[alive], e[alive], rim


def suffix(tracker, m, stations, entries, blank, all_rows):
    sizes = np.array([len(row[2]) for row in entries])
    edges = np.r_[0, sizes.cumsum()]
    p = cp.asarray(np.vstack([row[1] for row in entries]))
    en = cp.asarray(np.concatenate([row[3] for row in entries]))
    n = len(en)
    lost = cp.full(n, -1, cp.int32)
    peak = cp.zeros(n)
    margin = cp.full(n, cp.inf)
    phi = cp.zeros(n)
    first = stations[0]
    cov = [[] for row in entries]
    for k in stations:
        for j in range(first, k, 160):
            tracker.kernel(
                ((n + 127) // 128,),
                (128,),
                (
                    p,
                    en,
                    phi,
                    blank,
                    tracker.ep,
                    tracker.er,
                    np.int32(len(tracker.r2)),
                    tracker.h,
                    tracker.rad,
                    lost,
                    peak,
                    margin,
                    np.int32(j),
                    np.int32(min(j + 160, k)),
                    np.int32(n),
                    tracker.rest,
                    FACTOR,
                    tracker.pc,
                ),
            )
        first = k
        P, L = p.get(), lost.get()
        for i, (_, _, w, _, _) in enumerate(entries):
            sl = slice(edges[i], edges[i + 1])
            a = L[sl] == -1
            cov[i].append(covariance(P[sl][a], w[a])[2:] if a.sum() > 3 else [np.nan, np.nan])
    z = m.z[stations] * 1000
    for i, ((c1, c2, pre, match), _, w, _, rim) in enumerate(entries):
        sl = slice(edges[i], edges[i + 1])
        a = L[sl] == -1
        if a.sum() < 4 or min(L[sl]) < -1:
            continue
        v = values(P[sl][a], w[a])
        s = np.array(cov[i])
        band = rim[a] >= 47.5e-6
        vel = P[sl][a, 2:4]
        chief = np.r_[np.average(vel, axis=0, weights=w[a]), 1.0]
        chief /= np.linalg.norm(chief)
        directions = np.column_stack((vel, np.ones(len(vel))))
        angles = 1e3 * np.arctan2(np.linalg.norm(np.cross(directions, chief), axis=1), directions @ chief)
        all_rows.append(
            dict(
                c1=c1,
                c2=c2,
                pre=pre,
                same_pre=match,
                post=crossings(z, s.sum(axis=1)),
                d95=v["D95_um"],
                alpha=v["alpha95_mrad"],
                nA=v["sample_nA"],
                ess=v["ess"],
                edge=wq(angles[band], w[a][band], 0.5) if band.any() else None,
                nrim=int(band.sum()),
            )
        )


def distance(row):
    a = row["pre"]
    s = row["same_pre"]
    p = row["post"]
    # Permit terminal focus at the upper surface, counted separately.
    p = [r for r in p if r < 1598.7]
    if len(p) != 2 or len(a) != 2 or len(s) != 2:
        return 1000.0
    return max(
        750 - a[1],
        a[1] - 830,
        750 - s[1],
        s[1] - 830,
        1179.1 - p[0],
        p[0] - 1237.2,
        1420 - p[1],
        p[1] - 1512.5,
        0.0,
    )


def run(args):
    table = next(CACHE.glob(args.seed + "*-column-2880um.npz"), None) or next(
        CACHE.glob(args.seed + "*-column.npz")
    )
    name = table.name.split("-column")[0]
    gpath = next(CACHE.glob(name + "-gun-exit-262144*.npz"), None) or next(
        CACHE.glob(name + "-gun-exit-131072*.npz")
    )
    with (
        np.load(table) as f,
        np.load(GUN / "gun-volume-field.npz" if name == BASE.stem else CACHE / (name + "-gun.npz")) as g,
    ):
        use = g["z_m"] < 0.449
        axis = CubicSpline(
            np.r_[g["z_m"][use], f["z_m"]], np.r_[g["coeff"][use, 0, 2], f["coeff"][:, 0, 2]]
        )
        bias = json.loads(str(f["receipt"]))["bias_A"]
        ref = np.array([bias[key] for key in PACKS[:2]])
    z, fields, _, _ = axes(spacing=0.00045)
    part = z < HANDOFF
    z = np.r_[z[part], HANDOFF]
    fields = fields[:, : len(z) - 1].copy()
    fields[0] = FACTOR * axis((z[:-1] + z[1:]) / 2)
    x, v, w = anchor(z, fields[0], axis, gpath)
    with np.load(gpath) as d:
        ids = d["ids"][d["label"] == 0]
        u = d["end"][d["label"] == 0, 3:6]
        t = (u * u).sum(axis=1)
        e = 510998.95069 * t / (1 + np.sqrt(1 + t))
        original = len(d["ids"])
    if not 1024 <= args.count <= original:
        raise ValueError("Unsupported physical source count")
    sel = ids < args.count
    x = x[sel]
    v = v[sel]
    w = w[sel] * original / args.count
    e = e[sel]
    m, _, _ = optics_mesh(0.25)
    stations = (
        np.unique(np.r_[np.arange(1040, 1599, 6), 1179.1, 1237.2, 1420, 1512.5, 1598.7, 1599.1999975])
        * 0.001
    )
    stations = np.unique([np.argmin(abs(m.z - z0)) for z0 in stations])
    nested = table.with_name(table.stem + "-nested.npz")
    t = BoreTracker(
        m,
        nested if nested.exists() else table,
        (np.zeros((1, 5)), np.ones(1), np.arange(1), np.zeros(1), 0.0),
    )
    blank = cp.zeros((len(m.nodes), 20))
    rows = []
    entries = []
    n = 0
    st = np.searchsorted(z, STATIONS)
    for c1 in np.arange(*args.c1):
        for c2 in np.arange(*args.c2):
            res = packet(c1, c2, ref, z, fields, x, v, w, e, st, args.margin, args.cohort)
            if res is None:
                continue
            entries.append(res)
            n += 1
            if len(entries) >= 30:
                suffix(t, m, stations, entries, blank, rows)
                entries = []
    if entries:
        suffix(t, m, stations, entries, blank, rows)
    rows.sort(key=lambda q: (distance(q), q["d95"]))
    (HERE / (args.tag + ".json")).write_text(
        json.dumps(
            dict(
                seed=name,
                count=args.count,
                c1_range_A=args.c1,
                c2_range_A=args.c2,
                upstream_margin_mm=args.margin,
                cohort=args.cohort,
                rows=rows,
            ),
            indent=1,
        )
        + "\n"
    )
    print(
        "Propagated",
        n,
        "packets;",
        len(rows),
        "resolved;",
        sum(distance(q) == 0 for q in rows),
        "eligible",
        flush=True,
    )
    for row in rows[:8]:
        print(row, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--c1", nargs=3, type=float, default=[-49.0, -24.0, 0.5])
    p.add_argument("--c2", nargs=3, type=float, default=[6.0, 24.0, 0.25])
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--seed", default="global-1b21")
    p.add_argument("--tag", default="nonlinear-survey")
    p.add_argument("--count", type=int, default=32768)
    p.add_argument("--cohort", choices=("both", "live", "transmitted"), default="both")
    run(p.parse_args())
