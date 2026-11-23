"""Repeat the assembled DC solves and coil responses using edge-integrated material loads."""

import argparse
import json
from unittest.mock import patch

import numpy as np

from circuit_audit import ADL, TL12, WORK, audit_inputs
from field_profiles import run as sample_assembly
from temsim.corrector_field_cli import _save, cached_mesh
from temsim.physics import probe_corrector_fem as fem
from winding_discretization import winding_on_edges


def assembled_solver():
    recipe, bias, archive = audit_inputs()
    spacing, padding = 5.0, 170.0
    mesh_id, mesh = cached_mesh(recipe, spacing, padding, WORK / "meshes")
    xyz, tet, labels, far = mesh
    good, _ = fem._conforming_nodes(xyz, tet, far)
    centers = xyz[tet[good & (labels != "air")]].mean(axis=1)
    fields, quality = {}, {}
    for key in recipe["sources"]:
        path = WORK / "loads" / mesh_id / (key + ".npz")
        if path.is_file():
            with np.load(path) as prior:
                fields[key] = prior["h_per_amp"]
                quality[key] = json.loads(str(prior["checks"]))
        else:
            p, h, check = winding_on_edges(recipe, mesh, 1.0, key)
            np.testing.assert_array_equal(p, centers)
            fields[key], quality[key] = h, check
            _save(path, h_per_amp=h, checks=json.dumps(check))
    encoded = json.dumps(recipe, sort_keys=True)

    def run_with_edge_load(actual, geometry, currents, **kwargs):
        if json.dumps(actual, sort_keys=True) != encoded:
            raise ValueError("Current basis requires the same material and coil geometry")
        if geometry is not mesh:
            for expected, supplied in zip(mesh, geometry, strict=True):
                np.testing.assert_array_equal(expected, supplied)

        def source(_, points, drives):
            np.testing.assert_array_equal(points, centers)
            return sum(amps * fields[k] for k, amps in drives.items() if amps)

        with patch.object(fem, "impressed_h", source):
            return fem.solve(actual, geometry, currents, **kwargs)

    return recipe, bias, archive, mesh_id, mesh, quality, run_with_edge_load


def run(channels=None):
    spacing, padding = 5.0, 170.0
    recipe, bias, archive, mesh_id, mesh, quality, run_with_edge_load = assembled_solver()
    if channels:
        sample_assembly(
            spacing, padding, True, suffix="-edge", nonlinear_solve=run_with_edge_load, channels=channels
        )
        return
    for name, adl, tl12 in [("both", 10.0, 10.0), ("adl", 10.0, 0.0), ("bias", 0.0, 0.0)]:
        path = WORK / f"5-170-{name}-edge.npz"
        currents = {**bias, ADL: adl, TL12: tl12}
        if path.exists():
            with np.load(path) as prior:
                r = json.loads(str(prior["report"]))
                if (
                    r["mesh_id"] != mesh_id
                    or r["currents_A"] != currents
                    or r["edge_quadrature"] != quality
                ):
                    raise ValueError("Conflicting archived edge-load inputs")
            continue
        with np.load(WORK / f"5-170-{name}.npz") as same_current:
            original = json.loads(str(same_current["report"]))
            assert original["mesh_id"] == mesh_id and original["currents_A"] == currents
            seed = same_current["free_potential_a"]
        air, checkpoint = {}, {}
        corners, moments, r = run_with_edge_load(
            recipe,
            mesh,
            currents,
            backend="cuda",
            tolerance=2e-9,
            potential_seed_a=seed,
            potential_checkpoint=checkpoint,
            air_solution=air,
            max_iterations=200,
            progress=lambda i, v: print(f"{name} edge Newton {i}: {v:.4g}", flush=True),
        )
        r.update(
            mesh_id=mesh_id,
            mesh_mm=spacing,
            padding_mm=padding,
            historical_sha256=archive,
            edge_quadrature=quality,
        )
        _save(
            path,
            corners_m=corners,
            moments_Am2=moments,
            report=json.dumps(r),
            recipe=json.dumps(recipe, sort_keys=True),
            **air,
            **checkpoint,
        )
    sample_assembly(spacing, padding, True, suffix="-edge", nonlinear_solve=run_with_edge_load)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels", nargs="+", help="Measure selected increments at an existing assembled bias"
    )
    run(parser.parse_args().channels)
