"""Measure the field-driven scan pivot with a fixed two-axis electronics calibration."""

import argparse
import json
import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from numba import get_num_threads, set_num_threads

from relay import (
    CACHE,
    CHANNELS,
    HERE,
    MIX,
    WORK,
    BoreTracker,
    Vacuum,
    advance,
    electric_reference,
    layered,
    optics_mesh,
    propagate,
    relaunch,
    sample,
    sha,
    source_packet,
    write,
)
from volume import basis, _sources
from gun_images import read_gun
from transport import transport
from material_trial import tangent


def scan_basis(path, j, gun):
    cache = CACHE / (path.stem + f"-fei-scan-{'gun' if gun else 'column'}-axis{j}.npz")
    if cache.exists():
        with np.load(cache) as f:
            return f["z"], f["c"], float(f["radius"])
    root = path.parent / (path.stem + "-responses")
    if not (root / "receipt.json").is_file() or not all(
        (root / (k + ".npy")).is_file() for k in CHANNELS
    ):
        tangent(path, CHANNELS, 0.5)
    ticket = json.loads((root / "receipt.json").read_text())
    if ticket["solution_sha256"] != sha(path):
        raise ValueError("Scan response from an unrelated working point")
    for k in CHANNELS:
        if ticket["responses"][k]["field_sha256"] != sha(root / (k + ".npy")):
            raise ValueError("Unverified differential field")
    with np.load(path) as d:
        mom = sum(a * np.load(root / (k + ".npy")) for a, k in zip(MIX[:, j], CHANNELS))
        vac = Vacuum.__new__(Vacuum)
        vac.p, vac.m = _sources(d["corners_m"], mom, 4)
        vac.recipe = json.loads(str(d["recipe"]))["sources"]
        vac.bias = {k.replace("_", "/"): a for k, a in zip(CHANNELS, MIX[:, j])}
    if gun:
        z = np.arange(-2.0, 452.1, 2.0) * 0.001
        radius, degree = 0.008, 4
    else:
        z = (
            np.unique(np.r_[np.arange(449.0, 1601.1, 2.0), np.arange(1330.0, 1480.1, 0.5), 1601.0])
            * 0.001
        )
        radius, degree = 0.00288000000001, 5
    phi = np.arange(12) * 2 * np.pi / 12
    rings = [0.35, 0.69, 1.0] if not gun else [0.48, 1.0, 1.38]
    xy = np.vstack((np.zeros((1, 2)), *[r * np.column_stack((np.cos(phi), np.sin(phi))) for r in rings]))
    pts = np.column_stack((np.tile(xy * radius, (len(z), 1)), np.repeat(z, len(xy))))
    val = vac.direct(pts).reshape(len(z), len(xy), 3)
    co = np.empty((len(z), (degree + 1) * (degree + 2) // 2, 3))
    co[:, 0] = val[:, 0]
    co[:, 1:] = np.einsum("ij,zjk->zik", np.linalg.pinv(basis(xy, degree)[:, 1:]), val - val[:, :1])
    rng = np.random.default_rng(420 + j)
    zz = rng.uniform(z[0], z[-1], 200)
    if not gun:
        zz[:100] = rng.uniform(1.36, 1.45, 100)
    r = radius * np.sqrt(rng.uniform(size=len(zz)))
    ph = rng.uniform(0, 2 * np.pi, len(zz))
    p = np.column_stack((r * np.cos(ph), r * np.sin(ph), zz))
    err = np.linalg.norm(
        vac.direct(p)
        - np.einsum("pi,pik->pk", basis(p[:, :2] / radius, degree), CubicSpline(z, co, axis=0)(zz)),
        axis=1,
    )
    np.savez_compressed(cache, z=z, c=co, radius=radius, error_T=err)
    print("Direct-field scan check", cache.name, "peak error T", max(err), flush=True)
    return z, co, radius


def signed_packet(path, n, j, sign):
    nominal = CACHE / (path.stem + f"-gun-exit-{n}-0.5.npz")
    if not nominal.exists():
        larger = sorted(
            (int(f.stem.split("-")[-2]), f)
            for f in CACHE.glob(path.stem + "-gun-exit-*-0.5.npz")
            if int(f.stem.split("-")[-2]) > n
        )
        if larger:
            from source import emit

            size, archive = larger[0]
            phases, w, _, _ = emit(8.544681193879073, n)
            with np.load(archive) as d:
                np.testing.assert_array_equal(phases, d["phase"][:n])
                np.testing.assert_allclose(w, d["weight_uA"][:n] * size / n, rtol=1e-14)
                payload = {k: d[k][:n] if d[k].ndim else d[k] for k in d.files}
                payload["weight_uA"] = w
            np.savez_compressed(nominal, **payload)
        else:
            nominal = relaunch(path, n)
    if sign == 0:
        return source_packet(nominal)
    tag = CACHE / (path.stem + f"-scan-command-{j}-{sign:+d}-{n}-gun.npz")
    if not tag.exists():
        z, c, _ = scan_basis(path, j, True)
        with np.load(sample(path, True)) as d, np.load(nominal) as g:
            np.testing.assert_array_equal(z, d["z_m"])
            spline = CubicSpline(z, d["coeff"] + sign * c, axis=0)
            mag = np.ascontiguousarray(spline.c.transpose(1, 2, 3, 0))
            ef, metal = read_gun(50)
            set_num_threads(min(14, get_num_threads()))
            ev, end, _ = transport(g["phase"], ef, mag, metal, np.array([0.45]), 0, 0.5)
            if np.any(ev < -2):
                raise ValueError("Gun tracking failed")
            np.savez_compressed(
                tag,
                end=end,
                label=np.where(ev < 0, -ev - 1, ev + 2),
                ids=g["ids"],
                weight_uA=g["weight_uA"],
            )
    return source_packet(tag)


def run(path, folder, n=131072, step=0.09):
    import cupy as cp

    folder.mkdir(exist_ok=True)
    table = layered(path)
    m, _, _ = optics_mesh(step)
    locs = np.unique(np.r_[1.418, np.arange(1.420, 1.5126, 0.0004), 1.5125])
    stops = np.unique(np.searchsorted(m.z, locs))
    z = m.z[stops]
    with np.load(table) as d:
        scales, degree = d["scales"], int(d["degree"])
    paths = []
    for j, sign in [(0, 0), (0, 1), (0, -1), (1, 1), (1, -1)]:
        src = signed_packet(path, n, j, sign)
        t = BoreTracker(m, table, src)
        if sign:
            zz, co, rad = scan_basis(path, j, False)
            powers = np.repeat(np.arange(degree + 1), np.arange(1, degree + 2))
            pol = CubicSpline(zz, co, axis=0)(m.nodes)
            for i, layer in enumerate(scales):
                t.tables[0][:, i] += cp.asarray(pol * (sign * (layer / rad) ** powers)[None, :, None])
        ((_, p, loss),) = propagate(t, m, [stops[0]])
        keep = loss == -1
        ph = p[keep]
        ids = t.ids[keep]
        w = t.weights[keep]
        en = 300000 + t.de[keep]
        ref = electric_reference(t)[keep]
        prev = stops[0]
        out = [ph[:, :4].copy()]
        blank = cp.zeros((len(m.nodes), 20))
        alive = np.full(len(w), -1, np.int32)
        for group in np.array_split(stops[1:], 20):
            for ph, alive in advance(t, blank, ph, en, ref, alive, prev, group):
                if np.any(alive != -1):
                    raise ValueError("Loss inside pivot interval")
                out.append(ph[:, :4].copy())
            prev = group[-1]
        paths.append((ids, w, np.array(out)))
        del t
    common = paths[0][0]
    for ids, w, ph in paths[1:]:
        common = np.intersect1d(common, ids)
    cent = []
    for ids, w, ph in paths:
        rows = np.searchsorted(ids, common)
        cent.append(np.einsum("n,znc->zc", w[rows] / w[rows].sum(), ph[:, rows]))
    response = np.stack([(cent[1] - cent[2]) / 2, (cent[3] - cent[4]) / 2], axis=-1)
    even = np.stack([(cent[1] + cent[2]) / 2 - cent[0], (cent[3] + cent[4]) / 2 - cent[0]], axis=-1)
    spl = CubicHermiteSpline(z, response[:, :2], response[:, 2:], axis=0)

    def root(axis=None):
        def f(q):
            mat = spl(q)
            return np.sum((mat if axis is None else mat[:, axis]) ** 2) * 1e18

        vals = np.array([f(q) for q in z])
        minima = []
        for i in find_peaks(-vals)[0]:
            q = minimize_scalar(
                f, bounds=(z[i - 1], z[i + 1]), method="bounded", options={"xatol": 2e-13}
            ).x
            if q > 1.420:
                minima.append((q, f(q)))
        if not minima:
            return None
        return min(minima, key=lambda pair: pair[1])[0]

    q = root()
    res = dict(
        field=path.name,
        command_urad=1.0,
        source_count=n,
        common_ids=len(common),
        counts=[len(s[0]) for s in paths],
        electronics_A_per_urad=MIX.tolist(),
        z_mm=None if q is None else q * 1000,
        axis_minima_mm=[(a * 1000 if a is not None else None) for a in (root(0), root(1))],
        position_singular_nm=(
            None if q is None else (1e9 * np.linalg.svd(spl(q), compute_uv=False)).tolist()
        ),
        tilt_singular_urad=(
            None if q is None else (1e6 * np.linalg.svd(spl(q, 1), compute_uv=False)).tolist()
        ),
        max_even_nm=float(abs(even[:, :2]).max() * 1e9),
        step_mm=step,
    )
    write(folder / "pivot.json", res)
    np.savez_compressed(
        folder / "pivot-response.npz", z_mm=1000 * z, response=response, even=even, ids=common
    )
    print("Fixed electronics pivot", res, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("field")
    p.add_argument("name")
    p.add_argument("--count", type=int, default=131072)
    a = p.parse_args()
    run(WORK / (a.field + ".npz"), HERE / a.name, a.count)
