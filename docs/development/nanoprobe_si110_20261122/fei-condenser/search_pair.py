"""Screen C1/C2 pairs on the common-field tangent with the broad gun-exit packet."""

import argparse
import json
import sys

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from gun_images import HERE

sys.path.insert(0, str(HERE.parent))
from condenser_zoom import PACKS, SURFACE, read_axis
from match import transfer
from screen import FACTOR, WORK

BASE = WORK / "global-1b213a3155920907.npz"
TANGENT = "global-b932b73416404799"
GUN_PACKET = HERE.parents[3] / "outputs/fei-gun-acceptance-20261123/anode-mesh15-50V-step0.5-rays.npz"
STATIONS = np.array([0.4449, 0.4451, 0.45, 0.7649, 0.7651, 1.0099, 1.0101, SURFACE])


def axes(seed=BASE, spacing=0.00065):
    with np.load(seed) as d:
        bias = json.loads(str(d["report"]))["excitation"]
    z = np.unique(np.r_[np.linspace(0.375, SURFACE, 1 + round((SURFACE - 0.375) / spacing)), STATIONS])
    dc = read_axis(WORK / (seed.stem + "-dc-bore-0.25mm.npz"))
    deriv = [read_axis(WORK / (TANGENT + "-" + k + "-bore-0.75mm.npz")) for k in ("c1", "c2")]
    middle = (z[:-1] + z[1:]) / 2
    return (
        z,
        np.array([f(middle) * FACTOR for f in [dc, *deriv]]),
        np.array([bias[k] for k in PACKS[:2]]),
        dc,
    )


def anchor(z, k, dc, packet=None):
    with np.load(GUN_PACKET if packet is None else packet) as d:
        exit = d["label"] == 0
        u = d["end"][exit]
        w = d["weight_uA"][exit]
    r = u[:, 0] + 1j * u[:, 1]
    v = (u[:, 3] + 1j * u[:, 4]) / u[:, 5] - 1j * FACTOR * dc(0.45) * r
    nominal = transfer(z, k)
    i = np.searchsorted(z, 0.45)
    A, B, C, D = nominal[i]
    rotation = np.exp(-1j * np.dot(np.diff(z)[:i], k[:i]))
    return rotation * (D * r - B * v), rotation * (-C * r + A * v), w


@njit
def quantile(a, w, p):
    order = np.argsort(a)
    acc = 0.0
    target = p * w.sum()
    for i in order:
        acc += w[i]
        if acc >= target:
            return a[i]
    return a[order[-1]]


@njit
def moments(x, v, w):
    w = w / w.sum()
    a = x - np.sum(w * x)
    b = v - np.sum(w * v)
    return np.sum(w * np.abs(a) ** 2), np.sum(w * (a.conjugate() * b).real), np.sum(w * np.abs(b) ** 2)


@njit
def scored(c1, c2, ref, z, fields, st, x, v, w, ks):
    M = transfer(z, fields[0] + (c1 - ref[0]) * fields[1] + (c2 - ref[1]) * fields[2])
    alive = np.ones(len(x), np.bool_)
    for n, rad in ((0, 0.0005), (1, 0.0005), (3, 50e-6), (4, 50e-6), (5, 0.002), (6, 0.002)):
        A, B, C, D = M[st[n]]
        alive &= np.abs(A * x + B * v) <= rad
    result = np.full(15, np.nan)
    if alive.sum() < 4:
        return result
    xx, vv, ww = x[alive], v[alive], w[alive]
    m = moments(xx, vv, ww)
    cov = np.empty(len(z))
    for j in range(len(z)):
        A, B, C, D = M[j]
        cov[j] = A * C * m[0] + (A * D + B * C) * m[1] + B * D * m[2]
    nroot = 0
    for j in range(st[2], len(z) - 1):
        if cov[j] < 0.0 <= cov[j + 1]:
            root = (z[j] - (z[j + 1] - z[j]) * cov[j] / (cov[j + 1] - cov[j])) * 1000
            if root > 1598.7:
                continue
            if nroot < 6:
                result[nroot] = root
            nroot += 1
    A, B, C, D = M[-1]
    p = A * xx + B * vv
    vel = C * xx + D * vv + 1j * ks * p
    p -= np.sum(ww * p) / ww.sum()
    vel -= np.sum(ww * vel) / ww.sum()
    av = np.abs(vel)
    result[6:] = np.array(
        [
            nroot,
            2e6 * quantile(np.abs(p), ww, 0.95),
            1e3 * np.arctan(quantile(av, ww, 0.95)),
            ww.sum() * 1e3,
            ww.sum() ** 2 / np.sum(ww * ww),
            alive.sum(),
            -1e3 * np.sum(ww * (p.conjugate() * vel).real) / np.sum(ww * av**2),
            max(np.abs(p)) * 1e6,
            np.sum(ww * av**2) / ww.sum(),
        ]
    )
    return result


@njit(parallel=True)
def sweep(grid, ref, z, fields, st, x, v, w, ks):
    out = np.empty((len(grid), 15))
    for i in prange(len(grid)):
        out[i] = scored(grid[i, 0], grid[i, 1], ref, z, fields, st, x, v, w, ks)
    return out


def run(step, name, range1, range2):
    z, field, ref, dc = axes()
    x, v, w = anchor(z, field[0], dc)
    grid = np.array(
        [(a, b) for a in np.arange(*range1, step[0]) for b in np.arange(*range2, step[1])] + [tuple(ref)]
    )
    set_num_threads(min(get_num_threads(), 14))
    data = sweep(grid, ref, z, field, np.searchsorted(z, STATIONS), x, v, w, FACTOR * float(dc(SURFACE)))
    np.savez_compressed(HERE / (name + ".npz"), grid=grid, result=data)
    ok = (
        (data[:, 6] == 4)
        & (data[:, 0] > 450)
        & (data[:, 0] < 650)
        & (data[:, 1] > 750)
        & (data[:, 1] < 830)
        & (data[:, 2] > 1179.1)
        & (data[:, 2] < 1237.2)
        & (data[:, 3] > 1420)
        & (data[:, 3] < 1512.5)
    )
    print("Reference", ref, np.round(data[-1], 5), "eligible", ok.sum(), "of", len(grid), flush=True)
    for col in (7, 8):
        valid = np.flatnonzero(ok)
        order = valid[np.argsort(data[valid, col] * (1 if col == 7 else -1))]
        for i in order[:6]:
            print("Candidate", grid[i], np.round(data[i, [0, 1, 2, 3, 7, 8, 9, 10, 12]], 5), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spacing", nargs=2, type=float, default=[1.0, 0.25])
    p.add_argument("--c1", nargs=2, type=float, default=[-66.0, -13.9])
    p.add_argument("--c2", nargs=2, type=float, default=[3.0, 34.1])
    p.add_argument("--name", default="coarse-screen")
    a = p.parse_args()
    run(a.spacing, a.name, a.c1, a.c2)
