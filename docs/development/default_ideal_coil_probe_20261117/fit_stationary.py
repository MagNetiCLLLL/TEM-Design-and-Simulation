"""Fit a fixed-surface 30 mrad ideal-coil design and rank conditional D95.

All optimizer pupils are zero-current probes. Full gun-exit populations used
for ranking are conditional on the named exit. Changed all-Z fields require
their own emitted-source and mechanical-interception verification.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares, lsq_linear

from joint import linear
from stationary_pupil import (
    CHANNELS,
    UNIT,
    Column,
    current_vector,
    half_nodes,
    hex_vector,
    kinematics,
    read_gun,
    transport,
)
from temsim.physics.beam_statistics import transverse_beam_statistics, _weighted_quantile


def metrics(phase, weights, source_current):
    s = transverse_beam_statistics(*phase[:, :4].T, weights=weights)
    r = np.hypot(phase[:, 0] - s.mean_x_m, phase[:, 1] - s.mean_y_m)
    core = np.hypot(phase[:, 0], phase[:, 1]) <= 50e-12
    return dict(
        D95_nm=2e9 * s.radius_95_m,
        D90_nm=2e9 * _weighted_quantile(r, weights, 0.9),
        D50_nm=2e9 * _weighted_quantile(r, weights, 0.5),
        rms_radius_nm=1e9 * s.radius_rms_m,
        alpha95_mrad=s.convergence_95_mrad,
        angular_rms_mrad=1e3 * s.convergence_rms_rad,
        core_50pm_current_a=float(source_current * weights[core].sum()),
    )


def scaled_recipe(base, currents, hexes):
    out = copy.deepcopy(base)
    original = current_vector(base)
    for key, spec in out["recipes"].items():
        indices = np.flatnonzero(np.array(base["owner_per_group"]) == key)
        cells = [cell for winding in spec["windings"].values() for cell in winding]
        if len(cells) != len(indices):
            raise ValueError("Winding inventory changed")
        for i, cell in zip(indices, cells):
            cell["current_ratio"] *= currents[i] / original[i]
    for (key, attr), val in zip(CHANNELS, hexes):
        out["multipole_commands"].setdefault(key, {})[attr] = float(val * UNIT)
    out["current_a"] = currents.tolist()
    # Ancestor pencil figures describe the seed, not these new currents.
    for key in (
        "conditional_rays",
        "conditional_entrance_r95_nm",
        "conditional_semi_angle_mrad",
        "history",
        "hp_relay",
        "design_comparison",
        "zero_current_aberration_fit",
        "pencil",
        "planes_m",
        "reference_gun_sha256",
        "gun_r95_m",
        "target_gain_per_m",
        "step_verification_mm",
        "peak_sampled_gun_b_t",
    ):
        out.pop(key, None)
    out["status"] = "Stationary optical design awaiting its matching physical gun and column"
    return out


def write(path, data):
    tmp = path.with_suffix(".part.json")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("basis", "recipe", "gun", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--step-mm", type=float, default=0.16)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument(
        "--direct-iterations",
        type=int,
        default=0,
        help="Refine on the full named gun-exit population with local control sensitivities",
    )
    parser.add_argument(
        "--place-angle",
        action="store_true",
        help="Use coupled C2/C3 currents and Objective refocus to place the full-population alpha95",
    )
    parser.add_argument("--target-mrad", type=float, default=30.0)
    parser.add_argument("--rim-mrad", type=float, help="Override the angular target for the reference rim")
    parser.add_argument("--bound-percent", type=float, default=4.0)
    parser.add_argument("--refocus", action="store_true")
    parser.add_argument(
        "--ac-waist",
        action="store_true",
        help="Place the paraxial source image at the physical scan-pair midpoint",
    )
    parser.add_argument(
        "--ac-window-mm",
        type=float,
        default=0.0,
        help="Allowed paraxial deviation around the AC midpoint during design",
    )
    parser.add_argument(
        "--group-gradients",
        action="store_true",
        help="Also vary axial signed-current gradients of installed windings",
    )
    parser.add_argument(
        "--all-groups",
        action="store_true",
        help="Complete the current basis for each independently powered winding group",
    )
    parser.add_argument(
        "--gun-reference-recipe",
        type=Path,
        help="Hold the measured gun's upstream axial field when fitting new column currents",
    )
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    base = json.loads(args.recipe.read_text())
    with np.load(args.basis) as b:
        start, stop = b["z"][0], float(b["surface"])
    z = np.linspace(start, stop, int(np.ceil((stop - start) * 1e3 / args.step_mm)) + 1)
    steps = np.diff(z)
    save = np.array([len(z) - 1])
    col = Column(args.basis, base, half_nodes(z))
    c0, m0 = current_vector(base), hex_vector(base)
    ph, weight, ids, energy, curr = read_gun(args.gun)
    kin = kinematics(300000.0 + energy)
    measured = transverse_beam_statistics(*ph[:, :4].T, weights=weight)
    mean = np.average(energy, weights=weight)
    sd = np.sqrt(np.average((energy - mean) ** 2, weights=weight))
    initial = col.field(c0, m0)

    def run(src, field, momentum):
        return transport(src, field, steps, *momentum, save)[0]

    def conditional(field):
        return metrics(run(ph, field, kin), weight, curr)

    owners = np.array(base["owner_per_group"])
    obj = c0 * (owners == "objective_lens")
    obj_field = col.field(obj, np.zeros_like(m0))
    reference = conditional(initial)
    print("Fixed gun exit, conditional starting measurement", reference, flush=True)
    output = dict(
        reference_recipe=str(args.recipe),
        gun_exit=str(args.gun),
        originals=len(ids),
        initial_conditional=reference,
        step_mm=float(steps.max() * 1e3),
        conditional_scope="Fixed reference gun exit, nonlinear vacuum field expansion, no mechanical stops",
    )
    write(args.output / "search.json", output)
    if args.iterations == 0 and not args.refocus and not args.direct_iterations and not args.place_angle:
        return
    # The Objective scale is solved on each round-field change. This removes
    # paraxial defocus from the other ill-conditioned control derivatives.
    curve, size = measured.radial_wavefront_curvature_per_m, measured.radius_95_m
    factor = kinematics(np.atleast_1d(300000.0 + mean))[0][0]
    h = (stop - start) / (len(z) - 1)
    baseline_u, _ = linear(initial[:, 0] * factor, h, curve)
    labels = [s for s in dict.fromkeys(owners) if s in base["recipes"] and s != "objective_lens"]
    relative = [c0 * (owners == key) * 0.01 for key in labels]
    if args.group_gradients:
        for key in list(labels) + ["objective_lens"]:
            subset = np.flatnonzero(owners == key)
            grad = np.zeros_like(c0)
            grad[subset] = np.linspace(-1.0, 1.0, len(subset)) * max(abs(c0[subset])) * 0.01
            relative.append(grad)
            labels.append(key + " axial gradient")
    if args.all_groups:
        for key in base["recipes"]:
            subset = np.flatnonzero(owners == key)
            existing = [v[subset] for v in relative if np.any(v[subset])]
            if key == "objective_lens":
                existing.append(obj[subset])
            for j, mode in enumerate(null_space(np.stack(existing)).T):
                v = np.zeros_like(c0)
                v[subset] = mode * max(abs(c0[subset])) * 0.01
                relative.append(v)
                labels.append(f"{key} independent group {j+1}")
    relative = np.stack(relative)
    modes = np.stack([col.field(delta, np.zeros_like(m0)) for delta in relative])
    gains = np.maximum(abs(m0) * 0.01, 0.01)
    hmodes = np.stack([col.field(np.zeros_like(c0), dm) for dm in np.diag(gains)])
    actions = np.ascontiguousarray(np.concatenate([modes, hmodes]).reshape(len(labels) + len(gains), -1))
    n = len(labels)
    if args.gun_reference_recipe:
        from design import basis

        cells, _, _, _, raw, _, ng, _, _ = basis(2.0, True)
        if [c["owner"] for c in cells] != list(owners):
            raise ValueError("Gun field constraint requires the same winding groups")
        gun_response = np.vstack((raw[:ng], col.round[:, 0, 0]))
        gun_currents = current_vector(json.loads(args.gun_reference_recipe.read_text()))

        def gun_error(v, delta):
            return gun_response @ (c0 + v[:n] @ relative + delta * obj - gun_currents) / 1e-5

    else:

        def gun_error(v, delta):
            return np.zeros(0)

    def field_at(x):
        field = initial + (x[:-1] @ actions).reshape(initial.shape)
        target_u = baseline_u + x[-1] * 1e-9 / size
        delta = 0.0
        for _ in range(10):
            b = (field[:, 0] + delta * obj_field[:, 0]) * factor
            u, _ = linear(b, h, curve)
            if abs(u - target_u) < 2e-10:
                break
            test, _ = linear(b + 1e-7 * obj_field[:, 0] * factor, h, curve)
            delta -= np.clip((u - target_u) * 1e-7 / (test - u), -0.02, 0.02)
        else:
            raise ValueError("Paraxial focus servo did not converge")
        return field + delta * obj_field, delta

    # Resolve the critical D95 pupil rim and the intervening radial extrema.
    az = np.arange(8) * np.pi / 4.0
    rings = np.array([0.18, 0.38, 0.56, 0.71, 0.83, 0.92, 0.98, 1.015, 1.045])
    xy = (size * rings[:, None, None] * np.column_stack((np.cos(az), np.sin(az)))).reshape(-1, 2)
    rotation = np.sum(weight * (ph[:, 0] * ph[:, 3] - ph[:, 1] * ph[:, 2])) / np.sum(
        weight * (ph[:, 0] ** 2 + ph[:, 1] ** 2)
    )

    def pupil_source(points):
        slopes = curve * points + rotation * np.column_stack((-points[:, 1], points[:, 0]))
        return np.column_stack((points, slopes, np.zeros(len(points))))

    src = pupil_source(xy)
    ray = run(src, initial, kinematics(np.full(len(src), 300000.0 + mean)))
    rim = (
        np.interp(
            1.0, rings, np.arctan(np.linalg.norm(ray[:, 2:4], axis=1)).reshape(len(rings), -1).mean(axis=1)
        )
        * 1e3
    )
    rim_target = (
        args.rim_mrad if args.rim_mrad is not None else rim + args.target_mrad - reference["alpha95_mrad"]
    )
    energy_rings = xy.reshape(len(rings), -1, 2)[[2, 5, 7], ::2].reshape(-1, 2)
    energy_src = pupil_source(energy_rings)
    samples = np.vstack((src, energy_src, energy_src))
    energies = np.r_[
        np.full(len(src), mean), np.full(len(energy_src), mean - sd), np.full(len(energy_src), mean + sd)
    ]
    momenta = kinematics(300000.0 + energies)
    x = np.zeros(len(actions) + 1)
    bounds = np.r_[np.full(n, args.bound_percent), np.full(len(m0), 22.0), 120.0]
    weights = np.repeat(np.array([0.3, 0.5, 0.8, 1.0, 1.15, 1.25, 1.35, 1.1, 0.65]), len(az))
    if args.ac_waist:
        from temsim.optics.crossover_acceptance import regions_for_state
        from temsim.instrument_snapshot import InstrumentSnapshot

        state = InstrumentSnapshot.from_dict(json.loads((args.gun / "inputs.json").read_text())).restore()
        ac_middle = 0.001 * np.mean(regions_for_state(state)["ac_pair"]["bounds_mm"])
        ac_node = int(np.argmin(abs(z - ac_middle)))

        def ac_error(field):
            u, v = linear(field[: 2 * ac_node + 1, 0] * factor, h, curve)
            return 1e3 * (u / v + ac_middle - z[ac_node])

        print("Initial AC paraxial image residual (mm)", ac_error(initial), flush=True)

    def residual(v):
        field, delta = field_at(v)
        at = run(samples, field, momenta)
        pupil = at[: len(src)]
        angles = np.arctan(np.linalg.norm(pupil[:, 2:4], axis=1)).reshape(len(rings), -1) * 1e3
        rim_angles = angles[6] + ((1.0 - rings[6]) / (rings[7] - rings[6])) * (angles[7] - angles[6])
        ac = ac_error(field) if args.ac_waist else 0.0
        return np.r_[
            (pupil[:, :2] * 1e9 * weights[:, None]).ravel(),
            (at[len(src) :, :2] * 1e9 * 0.50).ravel(),
            120 * (rim_angles.mean() - rim_target),
            0.4 * (rim_angles - rim_angles.mean()),
            0.02 * v[:-1],
            4.0 * np.sign(ac) * max(0.0, abs(ac) - args.ac_window_mm),
            gun_error(v, delta),
        ]

    def jacobian(v):
        reference_residual = residual(v)
        increments = np.maximum(abs(v) * 2e-5, 2e-4)
        return np.column_stack(
            [
                (residual(v + d) - reference_residual) / delta
                for d, delta in zip(np.diag(increments), increments)
            ]
        )

    def publish(v, label):
        field, delta = field_at(v)
        current = c0 + v[:n] @ relative + delta * obj
        commands = m0 + v[n:-1] * gains
        proposal = scaled_recipe(base, current, commands)
        measurement = conditional(field)
        print(label, measurement, "Objective delta", delta, flush=True)
        output.setdefault("trials", []).append(
            dict(
                stage=label,
                controls=v.tolist(),
                objective_delta=delta,
                conditional=measurement,
                labels=labels,
                rim_target_mrad=rim_target,
            )
        )
        write(args.output / "search.json", output)
        write(args.output / (label + "-currents.json"), proposal)
        return measurement

    for index, max_eval in enumerate([min(12, args.iterations), max(0, args.iterations - 12)]):
        if max_eval:
            fit = least_squares(
                residual,
                x,
                jac=jacobian,
                bounds=(-bounds, bounds),
                max_nfev=max_eval,
                x_scale="jac",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            x = fit.x
            output.setdefault("solver", []).append(
                dict(
                    status=fit.status,
                    message=fit.message,
                    evaluations=fit.nfev,
                    cost=fit.cost,
                    ac_residual_mm=ac_error(field_at(x)[0]) if args.ac_waist else None,
                )
            )
            publish(x, "pupil-" + str(index + 1))

    # Fit actual mechanical phase, including gun position-angle correlations
    # and individual energies. The objective reports the untrimmed population.
    def beam_observables(at):
        xy = (at[:, :2] - np.average(at[:, :2], axis=0, weights=weight)) * 1e9
        direction = np.column_stack((at[:, 2:4], np.ones(len(at))))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        chief = np.average(direction, axis=0, weights=weight)
        chief /= np.linalg.norm(chief)
        polar = np.arctan2(np.linalg.norm(np.cross(direction, chief), axis=1), direction @ chief) * 1e3
        return xy, polar

    def score(summary, field):
        error = abs(summary["alpha95_mrad"] - args.target_mrad)
        ac = abs(ac_error(field)) if args.ac_waist else 0.0
        return (
            summary["D95_nm"]
            + 6 * max(0.0, error - 0.035)
            + 0.3 * error
            + 2 * max(0.0, ac - args.ac_window_mm)
        )

    if args.place_angle:
        controls = [
            i for i, key in enumerate(labels) if key.split()[0] in ("condenser_lens_2", "condenser_lens_3")
        ]
        controls.append(len(x) - 1)
        for iteration in range(4):
            field, delta = field_at(x)
            at = run(ph, field, kin)
            xy, polar = beam_observables(at)
            angle = _weighted_quantile(polar, weight, 0.95)
            residual_angle = args.target_mrad - angle
            if abs(residual_angle) < 0.004:
                break
            angular_weight = np.exp(-0.5 * ((polar - angle) / 0.10) ** 2) * weight
            angular_weight /= angular_weight.sum()
            rr = np.linalg.norm(xy, axis=1)
            # Protect the bulk and D95 rim from first-order image displacement.
            importance = np.sqrt(
                weight
                * np.minimum(1.0, (_weighted_quantile(rr, weight, 0.97) / np.maximum(rr, 1e-8)) ** 2)
                / weight.sum()
            )
            rows, derivative = [], []
            ac0 = ac_error(field) if args.ac_waist else 0.0
            for j in controls:
                dx = np.zeros_like(x)
                dx[j] = 0.002 if j != len(x) - 1 else 0.03
                test, test_delta = field_at(x + dx)
                yy, aa = beam_observables(run(ph, test, kin))
                rows.append(
                    np.r_[
                        ((yy - xy) * importance[:, None]).ravel(),
                        0.4 * ((ac_error(test) if args.ac_waist else 0.0) - ac0),
                        gun_error(x + dx, test_delta) - gun_error(x, delta),
                    ]
                    / dx[j]
                )
                derivative.append(angular_weight @ (aa - polar) / dx[j])
            jac = np.stack(rows, axis=1)
            deriv = np.array(derivative)
            direction = np.linalg.solve(jac.T @ jac + 1e-4 * np.eye(len(controls)), deriv)
            move = np.zeros_like(x)
            move[controls] = direction * residual_angle / (deriv @ direction)
            candidates = []
            scale_limit = np.min(
                (bounds[controls] - abs(x[controls])) / np.maximum(abs(move[controls]), 1e-18)
            )
            # A finite-population quantile can switch particle rank across
            # a perturbation. Check both directions against the exact value.
            magnitudes = np.array([0.002, 0.008, 0.03125, 0.125, 0.25, 0.5, 0.75, 1.0])
            factors = np.minimum(1.0, scale_limit) * np.r_[magnitudes, -magnitudes]
            for factor0 in factors:
                y = x + factor0 * move
                if np.any(abs(y) > bounds):
                    continue
                metric = conditional(field_at(y)[0])
                candidates.append((abs(metric["alpha95_mrad"] - args.target_mrad), y))
            print(
                "Actual angular residual, step limit and best tested error (mrad)",
                residual_angle,
                scale_limit,
                min((v[0] for v in candidates), default=None),
                flush=True,
            )
            if not candidates:
                break
            err, trial = min(candidates, key=lambda pair: pair[0])
            if err >= abs(residual_angle):
                break
            x = trial
            publish(x, "angle-" + str(iteration + 1))
    for index in range(args.direct_iterations):
        field, delta = field_at(x)
        at = run(ph, field, kin)
        xy, polar = beam_observables(at)
        radius = np.linalg.norm(xy, axis=1)
        summary = metrics(at, weight, curr)
        threshold = _weighted_quantile(polar, weight, 0.95)
        angular_weight = np.exp(-0.5 * ((polar - threshold) / 0.22) ** 2) * weight
        angular_weight /= angular_weight.sum()
        rank = np.empty(len(weight))
        ordering = np.argsort(radius)
        rank[ordering] = np.cumsum(weight[ordering]) / weight.sum()
        design_weight = np.interp(
            rank,
            [0.0, 0.55, 0.82, 0.92, 0.95, 0.973, 0.99, 1.0],
            [0.10, 0.2, 0.7, 2.0, 4.0, 0.9, 0.02, 0.00001],
        )
        design_weight = np.sqrt(weight * design_weight / weight.sum())
        ac_now = ac_error(field) if args.ac_waist else 0.0
        rhs = np.r_[
            -(xy * design_weight[:, None]).ravel(),
            12 * (args.target_mrad - threshold),
            np.clip(ac_now, -args.ac_window_mm, args.ac_window_mm) - ac_now,
            -gun_error(x, delta),
        ]
        columns = []
        for j in range(len(x)):
            dv = np.zeros_like(x)
            dv[j] = 0.0005 if j < len(x) - 1 else 0.03
            varied, changed_delta = field_at(x + dv)
            xy1, polar1 = beam_observables(run(ph, varied, kin))
            ac1 = ac_error(varied) if args.ac_waist else 0.0
            columns.append(
                np.r_[
                    ((xy1 - xy) * design_weight[:, None]).ravel(),
                    12 * angular_weight @ (polar1 - polar),
                    ac1 - ac_now,
                    gun_error(x + dv, changed_delta) - gun_error(x, delta),
                ]
                / dv[j]
            )
        jac = np.stack(columns, axis=1)
        jac = np.vstack((jac, np.diag(np.r_[np.full(len(x) - 1, 0.018), 0.002])))
        rhs = np.r_[rhs, np.zeros(len(x))]
        trust = np.r_[np.ones(n) * 0.75, np.ones(len(m0)) * 2.0, 12.0]
        lower = np.maximum(-trust, -bounds - x)
        upper = np.minimum(trust, bounds - x)
        step = lsq_linear(jac, rhs, bounds=(lower, upper), method="bvls").x
        best_score = score(summary, field)
        accepted = None
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial_x = x + scale * step
            field1, _ = field_at(trial_x)
            value = score(conditional(field1), field1)
            if value < best_score:
                accepted, best_score = trial_x, value
        if accepted is None:
            break
        x = accepted
        publish(x, "population-" + str(index + 1))
    if args.refocus:
        # The D95 order statistic is nonsmooth. Locally linearize the actual
        # full exit population under a small, physical Objective perturbation,
        # then repropagate to check the selected setting.
        for iteration in range(3):
            at = run(ph, field_at(x)[0], kin)
            dv = np.zeros_like(x)
            dv[-1] = 0.5
            change = (run(ph, field_at(x + dv)[0], kin) - at) / dv[-1]
            p = at[:, :2] - np.average(at[:, :2], axis=0, weights=weight)
            dp = change[:, :2] - np.average(change[:, :2], axis=0, weights=weight)

            def width(offset):
                v = p + offset * dp
                return _weighted_quantile(np.hypot(v[:, 0], v[:, 1]), weight, 0.95)

            offset, span = 0.0, (30.0 if iteration == 0 else 2.0)
            for _ in range(4):
                grid = np.linspace(offset - span, offset + span, 51)
                offset = float(grid[np.argmin([width(v) for v in grid])])
                span /= 12.5
            x[-1] += offset
            publish(x, "d95-" + str(iteration + 1))
            if abs(offset) < 1e-4:
                break


if __name__ == "__main__":
    main()
