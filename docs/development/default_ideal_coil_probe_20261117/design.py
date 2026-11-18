"""Design a round-coil current seed from a source-free paraxial conjugacy search.

The fitted operator uses fixed integer-turn winding groups and all axial tails.
The source-free pencil has no current. Only new nonlinear full-source transport
can establish physical alpha95, D95, RMS, aperture transmission or source images.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from numba import njit
from scipy.constants import e
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = HERE.parent / "ideal_current_fit_20261116"
sys.path.insert(0, str(ARCHIVE / "fixed-envelope-segmented-current/round-study"))
from fit import axis_basis, cells_for
from temsim.optics.direct_alignment import _electron_momentum_kg_m_s

PLANES_M = np.array([0.450, 0.8275, 1.211633, 1.40875, 1.5991999975])
FACTOR = e / (2 * _electron_momentum_kg_m_s(300.0))


@njit(cache=True)
def propagate(field, steps, stops):
    # Track zero crossings explicitly. Each local pencil restarts at its conjugate.
    result = np.empty((len(stops), 3))
    i = 0
    for segment, last in enumerate(stops):
        u, slope = (1.0, 0.903) if segment == 0 else (0.0, 1.0)
        crossings = 0
        for j in range(i, last):
            k = FACTOR * field[j]
            h = steps[j]
            c = np.cos(k * h)
            s = np.sin(k * h)
            next_u = c * u + (s / k if abs(k) > 1e-15 else h) * slope
            slope = -k * s * u + c * slope
            if next_u * u < 0:
                crossings += 1
            u = next_u
        # Continuous phase in units of pi, zero at the start of a unit-slope segment.
        phase = crossings + np.arctan2(abs(u), 0.05 * slope * (-1.0) ** crossings) / np.pi
        result[segment] = u, slope, phase
        i = last
    return result


def basis(step_mm, expanded=False):
    raw = json.loads((ARCHIVE / "round-lens-field-match/captured-field-coefficients.json").read_text())
    with (ARCHIVE / "round-lens-field-match/candidate-currents.csv").open() as stream:
        scales = {
            row["key"]: abs(float(row["isolated_b2_current_a"])) for row in csv.DictReader(stream)
        }
    grid = [
        np.linspace(a, b, int(np.ceil((b - a) / (step_mm * 0.001))) + 1)
        for a, b in zip(PLANES_M[:-1], PLANES_M[1:])
    ]
    z = np.concatenate([grid[0]] + [part[1:] for part in grid[1:]])
    stops = np.cumsum([len(part) - 1 for part in grid])
    gun = np.linspace(0.015, 0.447, 26)
    positions = np.r_[gun, (z[1:] + z[:-1]) / 2]
    cells, scale, nominal = [], [], []
    for owner, row in raw["ideal_current_coils"].items():
        meta = row["metadata"]
        master = meta["provenance"]["winding_current_definition"]["current_a"]
        parents = []
        for winding in meta["coils"]:
            parent = dict(winding["metadata"])
            parent["turns"] = int(round(parent["ampere_turns"] / master))
            parents.append(parent)
        # Keep downstream projector controls fixed, including their upstream tails.
        groups = (
            3
            if owner == "condenser_lens_1"
            else 2 if owner in ("objective_lens", "condenser_lens_2", "mini_condenser") else 1
        )
        if expanded:
            if owner in ("condenser_lens_1", "condenser_lens_2", "condenser_lens_3", "objective_lens"):
                groups = 4
            elif owner in (
                "mini_condenser",
                "adapter_lens",
                "probe_tl22_lens",
                "probe_tl21_lens",
                "probe_tl12_lens",
            ):
                groups = 3
        for cell in cells_for(owner, parents, 1, groups):
            cell.update(owner=owner, master_a=master)
            cells.append(cell)
            nominal.append(master)
            scale.append(scales[owner])
    matrix = axis_basis(positions, cells, 12)
    varied = np.array(
        [
            i
            for i, c in enumerate(cells)
            if c["owner"]
            not in ("diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2")
        ]
    )
    offset = matrix[:, len(varied) :] @ np.array(nominal)[len(varied) :]
    return (
        cells,
        np.array(scale),
        np.array(nominal),
        varied,
        matrix,
        offset,
        len(gun),
        stops,
        np.diff(z),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument(
        "--gun",
        type=Path,
        default=HERE.parent / "mixed_stem_si110_10nm_20261111/acquisition/original-gun.npz",
    )
    parser.add_argument("--initial-recipe", type=Path)
    parser.add_argument("--expand", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    # The archived Gaussian source suggests this reference pencil at gun exit.
    # The original current weights are used only for the measured reference scale.
    from temsim.physics.beam_statistics import transverse_beam_statistics

    with np.load(args.gun) as data:
        stats = transverse_beam_statistics(
            *(data[k] for k in ("x_m", "y_m", "tx_rad", "ty_rad")),
            alive=data["alive"],
            weights=data["weight"]
        )
    target_gain = 0.030 / stats.radius_95_m
    cells, scale, nominal, ix, matrix, offset, ng, stops, steps = basis(0.4, args.expand)
    scaled = np.ascontiguousarray(matrix[:, ix] * scale[ix])
    seed = np.sign(nominal[ix])
    if args.initial_recipe:
        old = json.loads(args.initial_recipe.read_text())["recipes"]
        guide = []
        for i in ix:
            cell = cells[i]
            owner, winding, interval = cell["parent_source_id"].rsplit(":", 2)
            lo, hi = cell["axial_fraction"]
            children = old[owner]["windings"][winding]
            ratio = sum(
                row["current_ratio"]
                * max(0.0, min(hi, row["axial_fraction"][1]) - max(lo, row["axial_fraction"][0]))
                for row in children
                if row["interval_index"] == int(interval)
            ) / (hi - lo)
            guide.append(ratio * nominal[i] / scale[i])
        guide = np.array(guide)
    else:
        guide = seed

    def quantities(x):
        field = offset + scaled @ x
        maps = propagate(field[ng:], steps, stops)
        slopes = np.hypot(maps[:, 1], maps[:, 0] / 0.05)
        return maps, np.prod(slopes), field[:ng]

    def residual(x):
        maps, gain, gun_b = quantities(x)
        phase = maps[:, 2]
        # One incoming image at each prescribed end. One extra image across the
        # two transfer sections is allowed only outside the C3-to-ADL interval.
        return np.r_[
            7 * (phase - [2.0, 2.0, 2.0, 1.0]),
            2 * np.log(gain / target_gain),
            gun_b / 0.015,
            0.006 * (x - guide),
        ]

    results = []
    rng = np.random.default_rng(20261117)
    for index in range(args.seeds):
        if index == 0:
            start = guide
        elif args.initial_recipe:
            start = np.clip(
                (guide if index % 2 else seed) * rng.uniform(0.5, 1.7, len(ix)), -11.0, 11.0
            )
        else:
            start = seed * rng.uniform(0.15, 2.1, len(ix))
        limit = 50.0 if args.expand else 12.0
        opt = least_squares(
            residual,
            start,
            bounds=(-limit, limit),
            max_nfev=300 if args.expand else 220,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        maps, gain, gun_b = quantities(opt.x)
        currents = nominal.copy()
        currents[ix] = scale[ix] * opt.x
        result = dict(
            score=float(np.linalg.norm(opt.fun)),
            seed=index,
            source_free_interval_maps=maps.tolist(),
            source_free_gain_per_m=gain,
            target_gain_per_m=target_gain,
            peak_sampled_gun_field_t=float(max(abs(gun_b))),
            currents_a=currents.tolist(),
            nfev=opt.nfev,
        )
        results.append(result)
        best = min(results, key=lambda row: row["score"])
        out = dict(
            status="Source-free search seed; nonlinear physical validation required",
            planes_m=PLANES_M.tolist(),
            gun_exit_reference_r95_m=stats.radius_95_m,
            cells=cells,
            best=best,
            search_results=results,
        )
        (args.output / "design.json").write_text(
            json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(index, "score", result["score"], "gain", gain, "phase", maps[:, 2], flush=True)

    # Refine the five optical constraints directly, retaining the selected
    # electrical branch. Verify on a finer mesh before generating a recipe.
    selected = np.array(best["currents_a"])[ix] / scale[ix]
    reference = selected.copy()
    for spacing in (0.1, 0.025):
        cells, scale, nominal, ix, matrix, offset, ng, stops, steps = basis(spacing, args.expand)
        scaled = np.ascontiguousarray(matrix[:, ix] * scale[ix])

        def polish(values):
            maps, gain, gun_b = quantities(values)
            return np.r_[
                maps[:, 0] / (0.05 * abs(maps[:, 1])),
                np.log(gain / target_gain),
                0.00001 * (values - reference),
                0.00001 * gun_b / 0.015,
            ]

        polished = least_squares(
            polish, selected, bounds=(-limit, limit), max_nfev=70, xtol=1e-12, ftol=1e-12, gtol=1e-12
        )
        selected = polished.x
    currents = nominal.copy()
    currents[ix] = selected * scale[ix]
    cells, scale, nominal, ix, matrix, offset, ng, stops, steps = basis(0.0125, args.expand)
    scaled = np.ascontiguousarray(matrix[:, ix] * scale[ix])
    maps, gain, gun_b = quantities(selected)
    recipes = {}
    for i in ix:
        cell = cells[i]
        owner, winding, interval = cell["parent_source_id"].rsplit(":", 2)
        recipe = recipes.setdefault(owner, dict(schema="ideal-coil-current-segments-v1", windings={}))
        recipe["windings"].setdefault(winding, []).append(
            dict(
                interval_index=int(interval),
                turn_count=cell["turn_count"],
                axial_fraction=cell["axial_fraction"],
                radial_fraction=cell["radial_fraction"],
                current_ratio=currents[i] / nominal[i],
            )
        )
    from hashlib import sha256

    result = dict(
        status="Source-free electrical seed for a stationary nonlinear trial",
        recipes=recipes,
        gun_reference_sha256=sha256(args.gun.read_bytes()).hexdigest(),
        reference_r95_m=stats.radius_95_m,
        current_a=currents.tolist(),
        owner_per_group=[c["owner"] for c in cells],
        target_gain_per_m=target_gain,
        source_free_gain_per_m=gain,
        peak_sampled_gun_field_t=float(max(abs(gun_b))),
        planes_m=PLANES_M.tolist(),
        final_verification_step_mm=0.0125,
        interval_maps=maps.tolist(),
    )
    (args.output / "trial-currents.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
