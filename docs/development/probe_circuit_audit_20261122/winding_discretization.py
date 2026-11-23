"""Controlled check of source-field cancellation in the reduced-potential iron elements.

The scalar FEM gradient is constant per tetrahedron. Compare its usual
pointwise impressed H with the gradient that reproduces the same winding's
three edge integrals. The geometry, physical currents, material law, open
boundary approximation and air-field evaluator are shared by both cases.
"""

import argparse
import json
from unittest.mock import patch

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import mu_0

from circuit_audit import TL12, WORK, audit_inputs
from field_profiles import induction_by_region, sample_delta
from temsim.corrector_field_cli import _save, cached_mesh
from temsim.physics import probe_corrector_fem as fem
from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics.corrector_vacuum_field import _coil_b


def winding_on_edges(recipe, mesh, amps, key=TL12):
    xyz, tet, labels, far = mesh
    good, _ = fem._conforming_nodes(xyz, tet, far)
    cells = tet[good & (labels != "air")]
    corners = xyz[cells]
    directed = np.stack(np.broadcast_arrays(cells[:, :1], cells[:, 1:]), axis=-1)
    pairs, idx = np.unique(np.sort(directed, axis=-1).reshape(-1, 2), axis=0, return_inverse=True)
    start = xyz[pairs[:, 0]]
    dl = xyz[pairs[:, 1]] - start
    row = recipe["sources"][key]
    if row["kind"] == "round":
        ((lo, hi),) = row["intervals"]
        # Use the same 6x6 winding discretization as the original 3-D solver.
        source = AnnularCoilPotential(
            key,
            row["inner"] * 0.001,
            row["outer"] * 0.001,
            (hi - lo) * 0.001,
            row["turns"] * amps,
            (hi + lo) * 0.0005,
            quadrature_order=6,
        )
        field = source.field_at_global_positions_t
    else:
        one = dict(
            recipe=json.dumps({"sources": {key: row}}), report=json.dumps({"currents_A": {key: amps}})
        )

        def field(points):
            return _coil_b(points, one)

    def integrate(order):
        total = np.zeros(len(pairs))
        nodes, weights = leggauss(order)
        for t, w in zip((nodes + 1) / 2, weights / 2):
            for j in range(0, len(pairs), 32768):
                sl = slice(j, j + 32768)
                h = field(start[sl] + t * dl[sl]) / mu_0
                total[sl] += w * np.einsum("ij,ij->i", h, dl[sl])
        return total

    edge4 = integrate(4)
    edge8 = integrate(8)
    orientation = np.where(directed[:, :, 1] > directed[:, :, 0], 1, -1)
    integral = edge8[idx.reshape(orientation.shape)] * orientation
    edgevec = corners[:, 1:] - corners[:, :1]
    element_h = np.linalg.solve(edgevec, integral[:, :, None])[:, :, 0]
    return (
        corners.mean(axis=1),
        element_h,
        dict(
            edges=len(pairs),
            max_edge_4_to_8_A=float(max(abs(edge4 - edge8))),
            max_abs_edge_integral_A=float(max(abs(edge8))),
        ),
    )


def run(spacing=5.0, padding=170.0):
    recipe, _, _ = audit_inputs()
    mesh_id, mesh = cached_mesh(recipe, spacing, padding, WORK / "meshes")
    currents = dict.fromkeys(recipe["sources"], 0.0)
    currents[TL12] = 10.0
    centre, element_h, checks = winding_on_edges(recipe, mesh, 10.0)

    def physical_source(requested, points, excitations):
        assert requested is recipe and excitations == currents
        np.testing.assert_array_equal(points, centre)
        return element_h.copy()

    air = {}
    # Substitute only this measured discretization for the material-element load.
    # The runtime field provider and all public defaults stay as installed.
    with patch.object(fem, "impressed_h", physical_source):
        corners, moments, report = fem.solve(
            recipe,
            mesh,
            currents,
            backend="cuda",
            tolerance=2e-8,
            air_solution=air,
            progress=lambda i, r: print(f"Edge load Newton {i}: {r:.4g}", flush=True),
        )
    report.update(mesh_id=mesh_id, edge_checks=checks, mesh_mm=spacing, padding_mm=padding)
    s = dict(
        corners_m=corners,
        moments_Am2=moments,
        recipe=json.dumps(recipe, sort_keys=True),
        report=json.dumps(report),
        **air,
    )
    _save(WORK / f"{spacing:g}-{padding:g}-tl12-edge-load.npz", **s)
    folder = WORK / f"profiles-{spacing:g}-{padding:g}"
    sample_delta(s, folder / "tl12-edge-load.npz", report)
    (folder / "edge-induction.json").write_text(
        json.dumps(induction_by_region(s, mesh, recipe), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-mm", type=float, default=5.0)
    parser.add_argument("--padding-mm", type=float, default=170.0)
    args = parser.parse_args()
    run(args.mesh_mm, args.padding_mm)
