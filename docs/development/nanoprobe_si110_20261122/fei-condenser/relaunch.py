"""Relaunch identical original tip particles in the rebias-specific condenser fringe."""

import argparse
import json

import numpy as np
from numba import get_num_threads, set_num_threads

from gun_images import GUN, axial_images, magnetic_spline, read_gun
from transport import transport
from volume import BASE, CACHE, sample, solution
from search_pair import GUN_PACKET
from source import emit


def relaunch(path, n=131072, scale=0.5):
    target = CACHE / (path.stem + f"-gun-exit-{n}-{scale:g}.npz")
    if target.exists():
        return target
    ef, metal = read_gun(50)
    mag = magnetic_spline(GUN / "gun-volume-field.npz" if path == BASE else sample(path, True))
    roots, _ = axial_images(ef, metal, mag, scale)
    refined, _ = axial_images(ef, metal, mag, scale / 2)
    phase, w, theta, law = emit(8.544681193879073, n)
    prefix = GUN_PACKET
    cached = []
    for f in CACHE.glob(path.stem + "-gun-exit-*-" + f"{scale:g}.npz"):
        size = int(f.stem.split("-")[-2])
        if size < n:
            cached.append((size, f))
    if cached and (path != BASE or scale != 0.5 or max(cached)[0] > 131072):
        prefix = max(cached)[1]
    with np.load(prefix) as original:
        old = len(original["label"])
        common = min(n, old)
        np.testing.assert_array_equal(phase[:common], original["phase"][:common])
        np.testing.assert_allclose(w[:common] * n, original["weight_uA"][:common] * old, rtol=1e-14)
        if prefix != GUN_PACKET or (path == BASE and scale == 0.5):
            start = common
            event0 = original["label"][:start]
            event0 = np.where(event0 < 2, -event0 - 1, event0 - 2)
            initial_end = original["end"][:start]
        else:
            start = 0
    set_num_threads(min(14, get_num_threads()))
    if start < n:
        print("Relaunching", n - start, "original tip particles", path.stem, flush=True)
        events, ends, _ = transport(phase[start:], ef, mag, metal, np.array([0.45]), 0, scale)
        if start:
            events = np.r_[event0, events]
            ends = np.vstack((initial_end, ends))
    else:
        events, ends = event0, initial_end
    if np.any(events < -2):
        raise ValueError("Unresolved gun events")
    labels = np.where(events < 0, -events - 1, events + 2)
    np.savez_compressed(
        target,
        label=labels,
        end=ends,
        ids=np.arange(n),
        phase=phase,
        weight_uA=w,
        polar_angle_rad=theta,
        source=json.dumps(law),
        gun_image=json.dumps(roots),
        step_check=json.dumps(refined),
    )
    print(
        "Gun exit",
        path.stem,
        (labels == 0).sum(),
        w[labels == 0].sum(),
        "uA; first image",
        roots,
        flush=True,
    )
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("c1", type=float)
    p.add_argument("c2", type=float)
    p.add_argument("--count", type=int, default=131072)
    p.add_argument("--scale", type=float, default=0.5)
    a = p.parse_args()
    relaunch(solution(a.c1, a.c2), a.count, a.scale)
