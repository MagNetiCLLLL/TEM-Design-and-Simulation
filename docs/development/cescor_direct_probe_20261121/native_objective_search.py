"""Place a through-focus D95 minimum using physical objective-current replays."""

import argparse
import json

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from joint_pivot_tuning import OUT
from joint_relay_search import NI, model_at, switch_objective
from relay_transfer import SURFACE, advance


HALF_M = 0.0003


class Profile:
    """Local Hermite interpolation of three installed-field particle checkpoints."""

    def __init__(self, states):
        xy, slopes = states[:, :, :2], states[:, :, 2:4]
        self.center = xy[1]
        self.rank = int(np.ceil(len(self.center) * 0.95)) - 1
        self.radii = np.linalg.norm(self.center, axis=1)
        self.growth = np.sum(self.center * slopes[1], axis=1)
        coefficients = []
        bounds = []
        for i, sign in ((0, -1), (2, 1)):
            v0 = slopes[1] * (sign * HALF_M)
            v1 = slopes[i] * (sign * HALF_M)
            delta = xy[i] - self.center
            b = 3 * delta - 2 * v0 - v1
            c = v0 + v1 - 2 * delta
            coefficients.append((v0, b, c))
            bounds.append(np.linalg.norm(np.abs(v0) + np.abs(b) + np.abs(c), axis=1))
        bound = np.maximum(*bounds)
        lo, hi = self.radii - bound, self.radii + bound
        min_q = np.partition(lo, self.rank)[self.rank]
        max_q = np.partition(hi, self.rank)[self.rank]
        below = hi < min_q
        keep = (~below) & (lo <= max_q)
        self.local_rank = self.rank - int(below.sum())
        self.xy = self.center[keep]
        self.coeff = [[v[keep] for v in segment] for segment in coefficients]

    def diameter_um(self, u):
        a, b, c = self.coeff[int(u >= 0)]
        u = abs(u)
        xy = self.xy + u * (a + u * (b + u * c))
        rr = xy[:, 0] ** 2 + xy[:, 1] ** 2
        return float(2e6 * np.sqrt(np.partition(rr, self.local_rank)[self.local_rank]))

    def minimum(self):
        grid = np.linspace(-1.0, 1.0, 121)
        values = np.array([self.diameter_um(u) for u in grid])
        candidates = [(values[0], -1.0), (values[-1], 1.0), (self.diameter_um(0.0), 0.0)]
        for i in np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])) + 1:
            result = minimize_scalar(
                self.diameter_um,
                method="bounded",
                bounds=grid[[i - 1, i + 1]],
                options={"xatol": 5e-11},
            )
            candidates.append((result.fun, result.x))
        diameter, u = min(candidates)
        return dict(
            surface_um=self.diameter_um(0.0),
            minimum_um=float(diameter),
            offset_nm=float(u * HALF_M * 1e9),
        )


def objective_focus(amps, anchors):
    model, drives = model_at(amps, NI, 32768, 0.0375, 24)
    z = SURFACE + np.array([-300000.0, 0.0, 300000.0]) * 1e-9
    ids = model.indices(z)
    archive = OUT / "native-objective-search.json"
    log = {}

    def replay(ni):
        ni = round(float(ni), 4)
        checkpoint = OUT / f"native-TL12-{amps:.10g}-objective-{ni:.10g}.npz"
        if not checkpoint.exists():
            switch_objective(model, ni)
            states = advance(
                model.t, model.field(drives), model.t.input, model.en, model.phi0, model.loss, 0, ids
            )
            phase = np.stack([p[:, :4] for p, _ in states])
            mask = states[-1][1] == -1
            if any(not np.array_equal(mask, los == -1) for _, los in states):
                raise ValueError("The through-focus cohort changes")
            np.savez_compressed(checkpoint, phase=phase, mask=mask)
        with np.load(checkpoint) as f:
            phase, mask = f["phase"], f["mask"]
        if not np.all(model.t.weights[mask] == model.t.weights[mask][0]):
            raise ValueError("Unequal particle currents require a weighted order statistic")
        phase -= np.mean(phase[:, mask], axis=1)[:, None, :]
        value = Profile(phase[:, mask]).minimum()
        value["surviving"] = int(mask.sum())
        log[ni] = value
        archive.write_text(json.dumps(log, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("Native", ni, value, flush=True)
        return phase, mask

    def predict(left, right):
        a, ma = replay(left)
        b, mb = replay(right)
        # Anchor interpolation proposes currents only. Each reported result
        # uses its own original live population and an independently solved field.
        mask = ma & mb
        a = a[:, mask]
        da = (b[:, mask] - a) / (right - left)
        a -= a.mean(axis=1)[:, None, :]
        da -= da.mean(axis=1)[:, None, :]
        rank = int(np.ceil(a.shape[1] * 0.95)) - 1

        def bundle(ni):
            return a + (ni - left) * da

        def order(ni):
            xy = a[1, :, :2] + (ni - left) * da[1, :, :2]
            rr = np.sum(xy * xy, axis=1)
            p = np.argpartition(rr, [rank - 3, rank + 3])
            return p[rank - 3 : rank + 4]

        def equal(ni, i, j):
            xy = a[1, [i, j], :2] + (ni - left) * da[1, [i, j], :2]
            return np.dot(xy[0], xy[0]) - np.dot(xy[1], xy[1])

        def growth(ni, i):
            ray = a[1, i] + (ni - left) * da[1, i]
            return np.dot(ray[:2], ray[2:4])

        # A finite-cohort percentile minimum can be smooth, or a cusp at a change
        # of the particle defining the 95th percentile. Test both on the surface.
        grid = np.linspace(left, right, max(11, min(401, int(abs(right - left) * 3) + 2)))
        possible = set()
        previous = order(grid[0])
        for nl, nr in zip(grid[:-1], grid[1:]):
            following = order(nr)
            near = np.union1d(previous, following)
            for number, i in enumerate(near):
                if growth(nl, i) * growth(nr, i) < 0:
                    possible.add(round(brentq(growth, nl, nr, args=(i,)), 6))
                for j in near[number + 1 :]:
                    if equal(nl, i, j) * equal(nr, i, j) < 0:
                        root = brentq(equal, nl, nr, args=(i, j))
                        row = bundle(root)[1]
                        same = np.sum(row[:, :2] ** 2, axis=1)
                        i_rank = np.count_nonzero(same < same[i])
                        if abs(i_rank - rank) <= 1:
                            possible.add(round(root, 6))
            previous = following
        best = None
        for ni in sorted(possible, key=lambda n: abs(n - (left + right) / 2)):
            profile = Profile(bundle(ni))
            # A local minimum has non-negative diameter differences on both sides.
            surface = profile.diameter_um(0.0)
            if min(profile.diameter_um(-0.00002), profile.diameter_um(0.00002)) < surface - 2e-11:
                continue
            detail = profile.minimum()
            score = (abs(detail["offset_nm"]), detail["surface_um"])
            if best is None or score < best[0]:
                best = (score, ni)
                if score[0] < 0.3:
                    break
        if best is None:
            trial = min(grid, key=lambda n: abs(Profile(bundle(n)).minimum()["offset_nm"]))
            return round(float(trial), 4)
        return round(float(best[1]), 4)

    left, right = sorted(anchors)
    for _ in range(6):
        candidate = predict(left, right)
        _, _ = replay(candidate)
        if abs(log[candidate]["offset_nm"]) <= 1.0:
            return candidate
        # Recenter the search on the measured field. Shrinking brackets reduce
        # errors from interpolating trajectories in a nonlinear objective field.
        radius = max(
            0.0125, min((right - left) * 0.17, abs(log[candidate]["offset_nm"]) * 0.0015 + 0.1)
        )
        left, right = round(candidate - radius, 4), round(candidate + radius, 4)
    return min(log, key=lambda n: abs(log[n]["offset_nm"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amps", type=float, default=255.71838)
    p.add_argument("--anchors", type=float, nargs=2, default=[-32800.0, -32720.0])
    args = p.parse_args()
    print("Chosen current (At)", objective_focus(args.amps, args.anchors), flush=True)
