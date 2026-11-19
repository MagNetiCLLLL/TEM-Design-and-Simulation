"""Fit specimen angular gain while allowing the intermediate source images to move."""

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import least_squares

import design
from design import FACTOR, basis
from temsim.physics.beam_statistics import transverse_beam_statistics

# Native lens centres and the two-coil region in the selected Normal column.
PLANES = np.array([0.450, 0.650, 0.915, 1.0875, 1.166, 1.250, 1.393, 1.411, 1.5225, 1.5991999975])
MIN_NODES = np.array([1, 2, 2, 3, 4, 5, 6, 6])
MAX_NODES = MIN_NODES + 1


@njit(cache=True)
def pencil(field, steps, stops, initial_slope):
    u, slope = 1.0, initial_slope
    crossings = 0
    result = np.empty((len(stops), 3))
    segment = 0
    for j in range(len(steps)):
        h, k = steps[j], FACTOR * field[j]
        c, si = np.cos(k * h), np.sin(k * h)
        next_u = c * u + (si / k if abs(k) > 1e-15 else h) * slope
        slope = -k * si * u + c * slope
        if u * next_u < 0:
            crossings += 1
        u = next_u
        if j + 1 == stops[segment]:
            phase = crossings + np.arctan2(abs(u), 0.05 * slope * (-1.0) ** crossings) / np.pi
            result[segment] = u, slope, phase
            segment += 1
    return result


@njit(cache=True)
def relay_map(field, steps, start, stop):
    a, b, c, d, theta = 1.0, 0.0, 0.0, 1.0, 0.0
    for j in range(start, stop):
        k, h = FACTOR * field[j], steps[j]
        co, si = np.cos(k * h), np.sin(k * h)
        drift = si / k if abs(k) > 1e-15 else h
        a, b, c, d = co * a + drift * c, co * b + drift * d, -k * si * a + co * c, -k * si * b + co * d
        theta += k * h
    return np.array([a, b, c, d, theta])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gun", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--iterations", type=int, default=260)
    parser.add_argument("--relay-sign", type=int, choices=[-1, 1])
    parser.add_argument("--hp-radius-um", type=float, default=0.0)
    parser.add_argument(
        "--relay-rotation",
        type=float,
        help="Target Larmor rotation between the primary hexapoles, in radians.",
    )
    parser.add_argument("--relay-magnification", type=float, default=1.0)
    parser.add_argument("--free-relay-curvature", action="store_true")
    parser.add_argument("--angle-weight", type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old = json.loads(args.seed.read_text())["recipes"]
    with np.load(args.gun) as g:
        stats = transverse_beam_statistics(
            *(g[k] for k in ("x_m", "y_m", "tx_rad", "ty_rad")), weights=g["weight"], alive=g["alive"]
        )
    goal = 0.030 / stats.radius_95_m
    design.PLANES_M = np.unique(np.r_[PLANES, 1.1445, 1.30085]) if args.relay_sign else PLANES

    def model(mesh):
        cells, scale, nom, ix, matrix, offset, ng, stops, steps = basis(mesh, args.expand)
        scaled = np.ascontiguousarray(matrix[:, ix] * scale[ix])
        selected = stops[np.searchsorted(design.PLANES_M, PLANES[1:]) - 1]
        endpoints = stops[np.searchsorted(design.PLANES_M, [1.1445, 1.30085]) - 1]

        def evaluate(x):
            field = scaled @ x + offset
            relay = (
                relay_map(field[ng:], steps, endpoints[0], endpoints[1])
                if args.relay_sign
                else np.empty(0)
            )
            if args.hp_radius_um:
                upstream = relay_map(field[ng:], steps, 0, endpoints[0])
                pupil_um = (
                    abs(upstream[0] + upstream[1] * stats.radial_wavefront_curvature_per_m)
                    * stats.radius_95_m
                    * 1e6
                )
                relay = np.r_[relay, pupil_um]
            return (
                pencil(field[ng:], steps, selected, stats.radial_wavefront_curvature_per_m),
                field[:ng],
                relay,
            )

        return cells, scale, nom, ix, evaluate

    cells, scale, nom, ix, evaluate = model(0.25)
    seed = []
    for i in ix:
        owner, winding, interval = cells[i]["parent_source_id"].rsplit(":", 2)
        lo, hi = cells[i]["axial_fraction"]
        ratio = sum(
            row["current_ratio"]
            * max(0.0, min(hi, row["axial_fraction"][1]) - max(lo, row["axial_fraction"][0]))
            for row in old[owner]["windings"][winding]
            if row["interval_index"] == int(interval)
        ) / (hi - lo)
        seed.append(ratio * nom[i] / scale[i])
    x = np.array(seed)
    history = []

    def residual(values):
        rays, gun, relay = evaluate(values)
        phase = rays[:-1, 2]
        # No intermediate waist position is prescribed within the allowed regions.
        # Penalize only an incorrect crossing count at one of their boundaries.
        windows = np.minimum(phase - (MIN_NODES + 0.001), 0) + np.maximum(
            phase - (MAX_NODES - 0.001), 0
        )
        u, s, phase_end = rays[-1]
        conjugacy = np.empty(0)
        if args.relay_sign:
            a, b, c, d, theta = relay[:5]
            goal_a = args.relay_sign * args.relay_magnification
            conjugacy = relay_weight * np.array(
                [a - goal_a, b / 0.03, 0.0 if args.free_relay_curvature else 0.015 * c, d - 1 / goal_a]
            )
            if args.relay_rotation is not None:
                conjugacy = np.r_[conjugacy, relay_weight * (theta - args.relay_rotation)]
            if args.hp_radius_um:
                conjugacy = np.r_[
                    conjugacy, pupil_weight * np.log(max(relay[-1], 0.01) / args.hp_radius_um)
                ]
        return np.r_[
            6 * windows,
            12 * (phase_end - 7),
            args.angle_weight * np.log(np.hypot(s, u / 0.05) / target),
            gun / 0.0008,
            0.0003 * (values - seed),
            conjugacy,
        ]

    relay_weight = 0.0
    pupil_weight = 0.0
    initial = evaluate(x)[0][-1]
    for target in np.geomspace(max(abs(initial[1]), 100.0), goal, 11):
        fit = least_squares(
            residual,
            x,
            bounds=(-100.0, 100.0),
            max_nfev=args.iterations,
            x_scale="jac",
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        x = fit.x
        rays, gun, relay = evaluate(x)
        history.append(
            dict(
                target_gain=float(target),
                rays=rays.tolist(),
                peak_sampled_gun_b_t=float(max(abs(gun))),
                norm=float(np.linalg.norm(fit.fun)),
                nfev=fit.nfev,
            )
        )
        print(target, rays[-1], rays[:-1, 2], max(abs(gun)), flush=True)
    if args.relay_sign:
        for relay_weight, pupil_weight in (
            (0.1, 0.15),
            (0.4, 0.5),
            (1.0, 1.0),
            (2.0, 2.0),
            (4.0, 3.0),
            (7.0, 4.0),
        ):
            fit = least_squares(
                residual,
                x,
                bounds=(-100.0, 100.0),
                max_nfev=args.iterations,
                x_scale="jac",
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
            )
            x = fit.x
            rays, gun, relay = evaluate(x)
            print("Relay", relay_weight, relay, "end", rays[-1], flush=True)
    for mesh in (0.05, 0.0125):
        cells, scale, nom, ix, evaluate = model(mesh)
        start = x.copy()
        # Recenter the numerical anchor and preserve the measured weak gun fringe.
        seed = start.copy()
        fit = least_squares(
            residual,
            x,
            bounds=(-100.0, 100.0),
            max_nfev=80,
            x_scale="jac",
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
        )
        x = fit.x
    # Polish final defocus independently. A gain error must never be traded
    # for an off-surface ray intersection in the seed sent to the real column.
    from scipy.optimize import root_scalar

    control = next(i for i in reversed(list(ix)) if cells[i]["owner"] == "objective_lens")

    def place_focus():
        def intercept(delta):
            test = x.copy()
            test[control] += delta
            return evaluate(test)[0][-1, 0]

        solution = root_scalar(intercept, x0=0.0, x1=1e-5, method="secant", xtol=1e-13, maxiter=25)
        if not solution.converged:
            raise ValueError("Could not place the first-order specimen intersection")
        x[control] += solution.root

    place_focus()
    cells, scale, nom, ix, evaluate = model(0.00625)
    place_focus()
    cells, scale, nom, ix, evaluate = model(0.003125)
    rays, gun, relay = evaluate(x)
    recipes = {}
    currents = nom.copy()
    currents[ix] = x * scale[ix]
    for i in ix:
        c = cells[i]
        owner, winding, interval = c["parent_source_id"].rsplit(":", 2)
        data = recipes.setdefault(owner, dict(schema="ideal-coil-current-segments-v1", windings={}))
        data["windings"].setdefault(winding, []).append(
            dict(
                interval_index=int(interval),
                turn_count=c["turn_count"],
                axial_fraction=c["axial_fraction"],
                radial_fraction=c["radial_fraction"],
                current_ratio=currents[i] / nom[i],
            )
        )
    from hashlib import sha256

    result = dict(
        status="Source-free operator, pending full nonlinear particle verification",
        planes_m=PLANES.tolist(),
        hp_relay=relay.tolist(),
        reference_gun_sha256=sha256(args.gun.read_bytes()).hexdigest(),
        gun_r95_m=stats.radius_95_m,
        target_gain_per_m=goal,
        conditional_rays=rays.tolist(),
        step_verification_mm=0.003125,
        conditional_entrance_r95_nm=abs(rays[-1, 0]) * stats.radius_95_m * 1e9,
        conditional_semi_angle_mrad=abs(rays[-1, 1]) * stats.radius_95_m * 1e3,
        peak_sampled_gun_b_t=float(max(abs(gun))),
        recipes=recipes,
        current_a=currents.tolist(),
        owner_per_group=[c["owner"] for c in cells],
        history=history,
    )
    (args.output / "trial-currents.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        "Verification",
        result["conditional_semi_angle_mrad"],
        result["conditional_entrance_r95_nm"],
        rays[-1],
        flush=True,
    )


if __name__ == "__main__":
    main()
