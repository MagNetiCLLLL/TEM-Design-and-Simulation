"""Rebias the common magnetic circuit and sample its vacuum field over the broad beam."""

import argparse
import json
import sys
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline

from gun_images import HERE

sys.path.insert(0, str(HERE.parent))
from material_trial import Circuit, Global, apply_supply_changes, digest, groups, load_bias, newton
from material_relay_fields import WORK, sha
from material_relay_global import impressed
from temsim.physics.corrector_vacuum_field import _induced_cuda, _sources

BASE = WORK / "global-1b213a3155920907.npz"
PREDICTOR = WORK / "global-b932b73416404799.npz"
CACHE = HERE.parents[3] / "outputs/fei-condenser-20261123"
CACHE.mkdir(exist_ok=True)


def solution(c1, c2, objective=None):
    changes = {"c1": c1, "c2": c2}
    if objective is not None:
        changes["objective"] = objective
    bias = apply_supply_changes(load_bias(BASE), changes)
    path = WORK / ("global-" + digest(bias)[:16] + ".npz")
    if not path.exists():
        with patch.object(newton, "groups", groups):
            Circuit(Global(9)).solve(bias, BASE, path, predictor=PREDICTOR)
    if load_bias(path) != bias:
        raise ValueError("Drive or nonlinear residual mismatch")
    return path


class Vacuum:
    def __init__(self, path):
        with np.load(path) as d:
            self.p, self.m = _sources(d["corners_m"], d["moments_Am2"], 4)
            self.bias = json.loads(str(d["report"]))["excitation"]
            self.recipe = json.loads(str(d["recipe"]))["sources"]

    def direct(self, points):
        q = np.array(points)
        q[:, 2] -= 0.45
        b = _induced_cuda(q, self.p, self.m)
        for k, amp in self.bias.items():
            if amp:
                b += amp * impressed(self.recipe[k], q)
        return b

    def symmetric(self, points):
        result = np.zeros_like(points)
        for a in np.arange(3) * 2 * np.pi / 3:
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            q = points.copy()
            q[:, :2] = q[:, :2] @ rot.T
            b = self.direct(q)
            b[:, :2] = b[:, :2] @ rot
            result += b / 3
        return result


def basis(xy, degree):
    x, y = xy.T
    return np.column_stack([x**i * y ** (d - i) for d in range(degree + 1) for i in range(d + 1)])


def sample(path, gun=False, refined=False, radial_mm=None):
    tag = ("gun" if gun else "column-2880um") + ("-fine" if refined else "")
    if radial_mm is not None:
        tag += f"-inner{radial_mm:g}mm"
    target = CACHE / (path.stem + "-" + tag + ".npz")
    if target.exists():
        return target
    source = Vacuum(path)
    if gun:
        # Identical support, grid and Cartesian term order to the FEI gun tracker.
        radius = 0.008
        degree = 4
        z = np.arange(-2.0, 452.1, 2.0) * 0.001
        az = np.arange(12) * 2 * np.pi / 12
        rings = [0.48, 1.0, 1.38]
    else:
        # Stay inside the 2.88 mm pole bore. Each fit carries its own
        # independent direct-field receipt.
        radius = 0.00288000000001 if radial_mm is None else radial_mm * 0.001
        degree = 7 if refined else 5
        pitch = 0.25 if refined else 0.5
        z = np.arange(449.0, 1601.0 + pitch / 3, pitch) * 0.001
        az = np.arange(18 if refined else 12) * 2 * np.pi / (18 if refined else 12)
        rings = [0.26, 0.53, 0.77, 1.0] if refined else [0.35, 0.69, 1.0]
    xy = np.vstack([np.zeros((1, 2)), *[r * np.column_stack((np.cos(az), np.sin(az))) for r in rings]])
    design = basis(xy, degree)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Insufficient radial or azimuthal order")
    sites = np.column_stack((np.tile(xy * radius, (len(z), 1)), np.repeat(z, len(xy))))
    print("Sampling", tag, len(sites), path.stem, flush=True)
    vals = source.direct(sites).reshape(len(z), len(xy), 3)
    # Each azimuthal stencil is closed under the installed threefold symmetry.
    sym = vals.copy()
    for h in (1, 2):
        a = h * 2 * np.pi / 3
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        b = vals.copy()
        b[:, 1:] = np.roll(
            vals[:, 1:].reshape(len(z), len(rings), len(az), 3), -h * len(az) // 3, axis=2
        ).reshape(len(z), -1, 3)
        b[:, :, :2] = b[:, :, :2] @ rot
        sym += b
    sym /= 3
    # Exact on-axis values, including the zero transverse DC enforced by symmetry.
    coeff = np.empty((len(z), design.shape[1], 3))
    coeff[:, 0] = sym[:, 0]
    coeff[:, 1:] = np.einsum("ij,zjc->zic", np.linalg.pinv(design[:, 1:]), sym - sym[:, :1])
    rng = np.random.default_rng(63202)
    zz = np.r_[
        rng.uniform(z[0], z[-1], 420),
        rng.uniform(0.450, 0.765, 120) if not gun else rng.uniform(0.37, 0.45, 120),
    ]
    r = radius * np.sqrt(rng.random(len(zz)))
    phi = 2 * np.pi * rng.random(len(zz))
    q = np.column_stack((r * np.cos(phi), r * np.sin(phi), zz))
    expected = source.symmetric(q)
    estimate = np.einsum(
        "pi,pik->pk", basis(q[:, :2] / radius, degree), CubicSpline(z, coeff, axis=0)(zz)
    )
    diff = np.linalg.norm(expected - estimate, axis=1)
    record = dict(
        radius_m=radius,
        degree=degree,
        axial_pitch_mm=(z[1] - z[0]) * 1000,
        solution_sha256=sha(path),
        bias_A=source.bias,
        n_validation=len(q),
        check_error_T=np.quantile(diff, [0.5, 0.95, 1]).tolist(),
        peak_sampled_B_T=float(np.linalg.norm(expected, axis=1).max()),
    )
    if not gun:
        tiny = r < 0.001
        record["sub_mm_error_T"] = np.quantile(diff[tiny], [0.5, 0.95, 1]).tolist()
    print("Volume validation", tag, record["check_error_T"], flush=True)
    np.savez_compressed(
        target, z_m=z, coeff=coeff, scale_m=radius, degree=degree, receipt=json.dumps(record)
    )
    return target


def layered(path, refined=False):
    outer = sample(path, False, refined)
    dest = outer.with_name(outer.stem + "-nested.npz")
    if dest.exists():
        return dest
    tables = [sample(path, False, refined, r) for r in (0.8, 1.9)] + [outer]
    layers = []
    scales = []
    checks = []
    for part in tables:
        with np.load(part) as d:
            layers.append(d["coeff"])
            scales.append(float(d["scale_m"]))
            checks.append(json.loads(str(d["receipt"])))
            z = d["z_m"]
            degree = int(d["degree"])
    np.savez_compressed(
        dest,
        z_m=z,
        degree=degree,
        scale_m=scales[-1],
        coeff=layers[-1],
        layers=layers,
        scales=scales,
        receipt=json.dumps(checks[-1]),
        layer_receipts=json.dumps(checks),
    )
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("c1", type=float)
    p.add_argument("c2", type=float)
    p.add_argument("--objective", type=float)
    p.add_argument("--gun", action="store_true")
    p.add_argument("--column", action="store_true")
    p.add_argument("--fine", action="store_true")
    a = p.parse_args()
    path = solution(a.c1, a.c2, a.objective)
    print("Material solution", path, flush=True)
    if a.gun:
        sample(path, True)
    if a.column:
        sample(path, False, a.fine)
