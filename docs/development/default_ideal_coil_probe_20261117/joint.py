"""Joint optical and hexapole pupil fit with independent physical Objective refocus."""

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.interpolate import make_interp_spline
from scipy.linalg import null_space
from scipy.optimize import least_squares

from aberrations import trace
from design import basis, FACTOR

OWNERS = (
    "condenser_lens_2",
    "condenser_lens_3",
    "adapter_lens",
    "probe_tl22_lens",
    "probe_tl21_lens",
    "probe_tl12_lens",
    "mini_condenser",
)


@njit(cache=True)
def linear(bz, h, g0):
    u, v = 1.0, g0
    for i in range(0, len(bz) - 1, 2):
        q0 = -((FACTOR * bz[i]) ** 2)
        qm = -((FACTOR * bz[i + 1]) ** 2)
        qf = -((FACTOR * bz[i + 2]) ** 2)
        a = v
        b = q0 * u
        c = v + h * b / 2
        d = qm * (u + h * a / 2)
        e = v + h * d / 2
        f = qm * (u + h * c / 2)
        g = v + h * f
        t = qf * (u + h * e)
        u += h * (a + 2 * c + 2 * e + g) / 6
        v += h * (b + 2 * d + 2 * f + t) / 6
    return u, v


def round_modes(recipe, middle, flex, individual):
    cells, scale, nominal, ix, matrix, offset, ng, stops, steps = basis(0.20, True)
    currents = np.array(recipe["current_a"])
    if [c["owner"] for c in cells] != recipe["owner_per_group"] or len(cells) != len(currents):
        raise ValueError("Round winding groups differ from the source-free recipe")
    coordinate = 0.45 + np.cumsum(steps) - steps / 2
    labels = list(OWNERS)
    changes = []
    for name in labels:
        changes.append(
            np.array([current if c["owner"] == name else 0.0 for c, current in zip(cells, currents)])
        )
    if flex:
        for name in ("condenser_lens_1", *OWNERS, "objective_lens"):
            selected = np.array([c["owner"] == name for c in cells])
            centres = np.array([c["center_z_m"] for c in cells])
            if not np.ptp(centres[selected]) > 0.0:
                continue
            shape = np.zeros(len(cells))
            location = centres[selected]
            shape[selected] = (location - location.mean()) / np.ptp(location) * 2
            shape *= np.max(abs(currents[selected]))
            changes.append(shape)
            labels.append(name + " axial gradient")
    if individual:
        # Complete the current basis within each real winding. Exclude the
        # common Objective scale because evaluate() solves it at each step.
        for name in ("condenser_lens_1", *OWNERS, "objective_lens"):
            selected = np.array([c["owner"] == name for c in cells])
            existing = [mode[selected] for mode in changes if np.any(mode[selected])]
            if name == "objective_lens":
                existing.append(currents[selected])
            directions = null_space(np.asarray(existing).reshape(-1, selected.sum()))
            for j, direction in enumerate(directions.T):
                mode = np.zeros(len(cells))
                mode[selected] = direction * np.max(abs(currents[selected]))
                changes.append(mode)
                labels.append(f"{name} independent current {j+1}")
    changes = np.stack(changes)
    fields = matrix @ changes.T
    spline = make_interp_spline(coordinate, fields[ng:], k=7)
    modes = np.stack([spline(middle, nu=n).T for n in range(4)], axis=2)
    return modes, fields[:ng].T, labels, changes, cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("controlled", type=Path)
    parser.add_argument("operator", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--coarsen", type=int, default=2)
    parser.add_argument("--evaluations", type=int, default=160)
    parser.add_argument("--flex", action="store_true")
    parser.add_argument("--core", action="store_true")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--chromatic-gun", type=Path)
    parser.add_argument(
        "--pupil-target",
        action="store_true",
        help="Fit the mechanical angle of the measured r95 gun rim and weight both core and halo.",
    )
    parser.add_argument(
        "--tail-preserve",
        action="store_true",
        help="Include gun-pupil radii out to the observed extreme in the RMS objective.",
    )
    parser.add_argument(
        "--round-bound",
        type=float,
        default=22.0,
        help="Maximum percent increment per round-current coordinate.",
    )
    parser.add_argument(
        "--angle-weight",
        type=float,
        default=4e4,
        help="Penalty per radian of mechanical rim-angle error.",
    )
    parser.add_argument(
        "--core-weight", type=float, default=1.0, help="Multiplier on the four innermost rim-fit radii."
    )
    parser.add_argument(
        "--individual-groups",
        action="store_true",
        help="Use independent complete-turn currents within each selected winding.",
    )
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    base = json.loads(args.recipe.read_text())
    candidate = json.loads(args.controlled.read_text())
    with np.load(args.operator) as cache:
        full = cache["fields"]
        h0 = float(cache["h"])
        channel_names = cache["names"]
        limits = cache["bounds"]
        command_values = cache["values"]
    fullgrid = np.linspace(0.45, 1.5991999975, len(full))
    g = np.linspace(fullgrid[0], fullgrid[-1], 2 * int(np.ceil((len(full) - 1) / 2 / args.coarsen)) + 1)
    h = 2 * (g[1] - g[0])
    reference = make_interp_spline(fullgrid, full, k=3)(g)
    modes, gun, labels, current_changes, cells = round_modes(base, g, args.flex, args.individual_groups)
    nround = len(labels)
    reference_radial = np.ascontiguousarray(reference[:, :4])
    objective = np.ascontiguousarray(reference[:, 4:8])
    mode_matrix = np.ascontiguousarray(modes.reshape(len(modes), -1).T)
    hexes = np.ascontiguousarray(reference[:, 8:].reshape(len(g), 6, 4).transpose(0, 2, 1))
    size = base["gun_r95_m"]
    curve = 0.9020248818935516
    energy_shifts = np.zeros(3)
    if args.chromatic_gun:
        from temsim.physics.beam_statistics import transverse_beam_statistics

        with np.load(args.chromatic_gun) as measured_gun:
            live = measured_gun["alive"] & (measured_gun["weight"] > 0)
            energy = measured_gun["energy_offset_ev"][live]
            w = measured_gun["weight"][live]
            mean = np.average(energy, weights=w)
            sd = np.sqrt(np.average((energy - mean) ** 2, weights=w))
            energy_shifts = mean + sd * np.array([0.0, -1.0, 1.0])
            curve = transverse_beam_statistics(
                *(measured_gun[k] for k in ("x_m", "y_m", "tx_rad", "ty_rad")),
                alive=live,
                weights=measured_gun["weight"],
            ).radial_wavefront_curvature_per_m
    from scipy.constants import m_e, c, e

    rest = m_e * c * c / e
    energies = 300000.0 + energy_shifts
    rigidity = np.sqrt(300000.0 * (300000.0 + 2 * rest) / (energies * (energies + 2 * rest)))
    az = np.arange(8) * np.pi / 4
    rings = np.array([0.15, 0.32, 0.60, 0.84, 1.05])

    def pencil(radii, angles):
        xy = (size * radii[:, None, None] * np.column_stack((np.cos(angles), np.sin(angles)))).reshape(
            -1, 2
        )
        return np.column_stack((xy, curve * xy))

    if args.core:
        rings = np.array([0.06, 0.14, 0.23, 0.32, 0.45, 0.65, 0.85, 1.03])
    if args.pupil_target:
        rings = np.array([0.07, 0.16, 0.28, 0.42, 0.60, 0.78, 0.92, 1.0, 1.06])
    if args.tail_preserve:
        if not args.pupil_target:
            parser.error("--tail-preserve requires --pupil-target")
        rings = np.r_[rings, 1.20, 1.36, 1.55]
    rim_index = np.flatnonzero(rings == 1.0).item() if args.pupil_target else None
    source = pencil(rings, az)

    def evaluate(x, inputs=source, *, chromatic=False):
        # Each optical variable scales its complete existing signed winding recipe.
        rel = x[:nround] * 0.01
        fields = reference_radial + (mode_matrix @ rel).reshape(len(g), 4)
        goal = x[-1] * 1e-9 / size
        d = 0.0
        for _ in range(8):
            b = (fields[:, 0] + d * objective[:, 0]) * rigidity[0]
            u, v = linear(b, h, curve)
            if abs(u - goal) < 1e-8:
                break
            up, _ = linear(b + 1e-6 * objective[:, 0] * rigidity[0], h, curve)
            derivative = (up - u) / 1e-6
            if abs(derivative) < 1e-9:
                raise ValueError("Objective has no focus response on this branch")
            d -= np.clip((u - goal) / derivative, -0.015, 0.015)
        actual = np.column_stack((fields + d * objective, hexes @ x[nround:-1]))
        central = trace(inputs, actual * rigidity[0], h)
        if chromatic:
            # Same physical Objective drive, observation plane and initial pencil.
            core_inputs = pencil(
                (
                    np.array([0.15, 0.32, 0.60, 0.95])
                    if args.tail_preserve
                    else np.array([0.15, 0.32, 0.60])
                ),
                az[::2],
            )
            low = trace(core_inputs, actual * rigidity[1], h)
            high = trace(core_inputs, actual * rigidity[2], h)
            return central, float(d), v * size, 0.5 * (high[:, :2] - low[:, :2]) * 1e9
        return central, float(d), v * size

    ring_weights = (
        [6.0, 10.0, 10.0, 8.0, 4.0, 0.8, 0.25, 0.12] if args.core else [2.0, 2.0, 1.25, 1.0, 0.75]
    )
    if args.pupil_target:
        ring_weights = np.r_[args.core_weight * np.array([5.0, 7.0, 7.0, 4.0]), 1.7, 1.1, 1.0, 1.0, 0.6]
    if args.tail_preserve:
        ring_weights = np.r_[ring_weights, 0.24, 0.11, 0.035]
    weights = np.broadcast_to(np.array(ring_weights)[:, None, None], (len(rings), len(az), 2)).reshape(
        -1, 2
    )

    def residual(x):
        if args.chromatic_gun:
            y, focus, slope, spectral = evaluate(x, chromatic=True)
        else:
            y, focus, slope = evaluate(x)
            spectral = np.empty(0)
        rel = x[:nround] * 0.01
        if args.pupil_target:
            rim = y.reshape(len(rings), len(az), 4)[rim_index, :, 2:]
            polar = np.arctan(np.linalg.norm(rim, axis=1))
            angle_error = np.r_[
                (polar.mean() - 0.030) * args.angle_weight, (polar - polar.mean()) * 2e3
            ]
        else:
            angle_error = (abs(slope) - 0.030) * 1e4
        return np.r_[
            ((y[:, :2] * 1e9) * weights).ravel(),
            angle_error,
            (rel @ gun) / 0.0001,
            0.003 * x[:nround],
            spectral.ravel(),
        ]

    initial_hex = candidate["zero_current_aberration_fit"]["controls"][:6]
    x0 = np.r_[np.zeros(nround), initial_hex, 0.0]
    if args.warm_start:
        prior = json.loads(args.warm_start.read_text())["design_comparison"]
        prior_x = np.array(prior["controls"])
        old_names = prior.get("round_modes", list(OWNERS))
        for i, key in enumerate(old_names):
            if key in labels:
                x0[labels.index(key)] = prior_x[i]
        x0[nround:] = prior_x[len(old_names) :]
    # The independent defocus coordinate is the target radius at the reference
    # pupil edge in nm. The selected Objective current is then solved explicitly.
    y, d, sl = evaluate(x0)
    bounds = np.r_[np.full(nround, args.round_bound), limits, 400.0]
    best = [np.inf]

    def publish(x, iterations):
        error = residual(x)
        norm = float(np.linalg.norm(error))
        if norm >= best[0]:
            return
        best[0] = norm
        yy, focus, slope = evaluate(x)
        holdout_rings = np.r_[np.linspace(0.08, 1.15, 17), 1.25, 1.36, 1.48, 1.59]
        measured = evaluate(x, pencil(holdout_rings, np.arange(24) * np.pi / 12))[0]
        out = json.loads(args.recipe.read_text())
        current = np.array(base["current_a"]) + x[:nround] * 0.01 @ current_changes
        recipes = {}
        for i, cell in enumerate(cells):
            owner = cell["owner"]
            if owner == "objective_lens":
                current[i] += focus * base["current_a"][i]
            if owner not in base["recipes"]:
                continue
            _, winding, interval = cell["parent_source_id"].rsplit(":", 2)
            group = recipes.setdefault(
                owner, dict(schema="ideal-coil-current-segments-v1", windings={})
            )
            group["windings"].setdefault(winding, []).append(
                dict(
                    interval_index=int(interval),
                    turn_count=cell["turn_count"],
                    radial_fraction=cell["radial_fraction"],
                    axial_fraction=cell["axial_fraction"],
                    current_ratio=current[i] / cell["master_a"],
                )
            )
        out["recipes"] = recipes
        out["current_a"] = current.tolist()
        out["multipole_commands"] = {}
        for (key, attr), val in zip(channel_names, x[nround:-1] * command_values):
            out["multipole_commands"].setdefault(str(key), {})[str(attr)] = float(val)
        holdout = (
            np.sqrt(np.mean(np.hypot(*measured[:, :2].T).reshape(len(holdout_rings), 24) ** 2, axis=1))
            * 1e9
        )
        out.update(
            status="Source-free nonlinear joint fit; original-particle verification required",
            design_comparison=dict(
                norm=norm,
                iterations=iterations,
                mesh_mm=h * 1e3,
                energy_offsets_ev=energy_shifts.tolist(),
                round_modes=labels,
                objective_scale=1 + focus,
                controls=x.tolist(),
                conditional_angle_mrad=abs(slope) * 1e3,
                pencil_rim_angle_mrad=(
                    float(
                        np.arctan(
                            np.linalg.norm(yy.reshape(len(rings), len(az), 4)[rim_index, :, 2:], axis=1)
                        ).mean()
                        * 1e3
                    )
                    if args.pupil_target
                    else None
                ),
                optimization_rms_per_ring_nm=np.sqrt(
                    np.mean(np.hypot(*yy[:, :2].T).reshape(len(rings), -1) ** 2, axis=1)
                )
                .__mul__(1e9)
                .tolist(),
                holdout_radius_fractions=holdout_rings.tolist(),
                holdout_rms_nm=holdout.tolist(),
            ),
        )
        (args.output / "trial-currents.json").write_text(
            json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("Fit", iterations, norm, abs(slope) * 1e3, holdout[3], holdout[-3], flush=True)

    for i in range(5):
        fit = least_squares(
            residual,
            x0,
            bounds=(-bounds, bounds),
            max_nfev=args.evaluations // 5,
            x_scale="jac",
            ftol=3e-10,
            xtol=3e-10,
            gtol=3e-10,
            diff_step=2e-5,
        )
        x0 = fit.x
        publish(x0, (i + 1) * args.evaluations // 5)


if __name__ == "__main__":
    main()
