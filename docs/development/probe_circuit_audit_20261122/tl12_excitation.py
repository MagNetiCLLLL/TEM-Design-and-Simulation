"""Re-solve the complete magnetic circuit for each physical TL12 excitation."""

import argparse
import json

import numpy as np

from assembled_edge_fields import assembled_solver
from circuit_audit import ADL, TL12, WORK
from field_profiles import difference
from temsim.corrector_field_cli import _save
from temsim.physics.corrector_vacuum_field import harmonic_table


FOLDER = WORK / "three-plane-tuning"


def paths(current, half):
    stem = f"tl12-{current:.10g}A"
    return FOLDER / (stem + "-solution.npz"), FOLDER / (stem + f"-field-{half}.npz")


def main(currents, half):
    FOLDER.mkdir(exist_ok=True, parents=True)
    context = None
    for current in currents:
        raw, table = paths(current, half)
        if not raw.exists():
            if context is None:
                context = assembled_solver()
            recipe, bias, history, mesh_id, mesh, quality, solver = context
            drives = {**bias, ADL: 10.0, TL12: float(current)}
            choices = [
                WORK / "5-170-both-edge.npz",
                WORK / "5-170-adl-edge.npz",
                *FOLDER.glob("tl12-*-solution.npz"),
            ]
            distances = []
            for p in choices:
                with np.load(p, allow_pickle=False) as src:
                    info = json.loads(str(src["report"]))
                    if info["mesh_id"] == mesh_id and all(
                        info["currents_A"][k] == v for k, v in drives.items() if k != TL12
                    ):
                        distances.append(
                            (abs(info["currents_A"][TL12] - current), info["currents_A"][TL12], p)
                        )
            ordered = sorted(distances)
            _, first, path = ordered[0]
            with np.load(path, allow_pickle=False) as checkpoint:
                seed = checkpoint["free_potential_a"]
            _, second, other = next(item for item in ordered if item[1] != first)
            with np.load(other, allow_pickle=False) as checkpoint:
                seed += (current - first) / (second - first) * (checkpoint["free_potential_a"] - seed)
            air, checkpoint = {}, {}
            corners, moments, report = solver(
                recipe,
                mesh,
                drives,
                backend="cuda",
                tolerance=2e-9,
                potential_seed_a=seed,
                potential_checkpoint=checkpoint,
                air_solution=air,
                max_iterations=300,
                progress=lambda i, v: (
                    print(f"{current:g} A nonlinear residual {i}: {v:.3g}", flush=True)
                    if i % 5 == 0 or v < 2e-9
                    else None
                ),
            )
            report.update(
                mesh_id=mesh_id,
                mesh_mm=5.0,
                padding_mm=170.0,
                historical_sha256=history,
                edge_quadrature=quality,
            )
            _save(
                raw,
                corners_m=corners,
                moments_Am2=moments,
                report=json.dumps(report),
                recipe=json.dumps(recipe, sort_keys=True),
                **air,
                **checkpoint,
            )
        with np.load(raw, allow_pickle=False) as f:
            report = json.loads(str(f["report"]))
            if report["currents_A"][TL12] != current or report["residual"] > 2e-9:
                raise ValueError("Mismatched or unconverged TL12 field")
            if table.is_file():
                with np.load(table) as t:
                    prior = json.loads(str(t["report"]))
                    if (
                        prior["currents_A"] != report["currents_A"]
                        or prior["mesh_id"] != report["mesh_id"]
                    ):
                        raise ValueError("Stale field table")
                continue
            solution = dict(f)
        print(f"Sampling {current:g} A on {2*half+1} transverse stations", flush=True)
        if 24 % half == 0 and (WORK / "normal-joint-field.npz").is_file():
            # Harmonic continuation and Biot-Savart evaluation are linear *after*
            # both nonlinear material problems have converged independently.
            with np.load(WORK / "5-170-both-edge.npz") as f:
                delta = difference(solution, dict(f), 1.0)
            z, xy, grid = harmonic_table(delta, 0.0005, grid_half=half, angles=36)
            stride = 24 // half
            with np.load(WORK / "normal-joint-field.npz") as ref:
                np.testing.assert_array_equal(z, ref["z_local_m"])
                np.testing.assert_allclose(xy, ref["xy_m"][::stride], atol=2e-18, rtol=0)
                grid += ref["B_zyx_T"][:, ::stride, ::stride]
        else:
            z, xy, grid = harmonic_table(solution, 0.0005, grid_half=half, angles=36)
        report.update(
            vacuum_only=False,
            field_registration="module local SI, global origin z=0.45 m",
            transverse_pitch_mm=float(1000 * (xy[1] - xy[0])),
        )
        _save(
            table,
            z_local_m=z,
            xy_m=xy,
            B_zyx_T=grid,
            valid_radius_m=0.00288,
            report=json.dumps(report, sort_keys=True),
        )
        print(table, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("current", type=float, nargs="+")
    p.add_argument("--half", type=int, default=8)
    args = p.parse_args()
    main(args.current, args.half)
