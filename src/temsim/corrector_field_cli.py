"""Solve the two installed probe hexapoles and transfer lenses at one DC operating point."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from scipy.interpolate import RegularGridInterpolator
from scipy.constants import mu_0
from temsim.physics.corrector_vacuum_field import AirPotential, _coil_b, harmonic_table, vacuum_field
from temsim.module_manifest import read_document, validate_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.probe_corrector_fem import HP, TL, circuit, mesh_circuit, solve


def _save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
    try:
        with temporary:
            np.savez_compressed(temporary, **arrays)
        Path(temporary.name).replace(path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def cached_mesh(recipe, spacing_mm, padding_mm, cache_dir):
    """Load or generate the iron and air tetrahedra shared by independent current solves."""
    encoded = json.dumps(recipe, sort_keys=True)
    mesh_id = sha256(f"tet-v2/{spacing_mm}/{padding_mm}/{encoded}".encode()).hexdigest()
    mesh_path = cache_dir / (mesh_id + ".npz") if cache_dir else None
    if mesh_path and mesh_path.is_file():
        with np.load(mesh_path) as arrays:
            mesh = arrays["xyz"], arrays["tet"], arrays["names"][arrays["index"]], arrays["far"]
    else:
        mesh = mesh_circuit(recipe, spacing_mm=spacing_mm, padding_mm=padding_mm)
        if mesh_path:
            names, indices = np.unique(mesh[2], return_inverse=True)
            _save(mesh_path, xyz=mesh[0], tet=mesh[1], names=names, index=indices.astype(np.int16), far=mesh[3])
    return mesh_id, mesh


def solve_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path, help="Output operating-point .npz file")
    p.add_argument("--column", type=Path, default=INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    for name in ("hp1", "hp2", "tl21", "tl22"):
        p.add_argument(
            "--" + name, type=float, required=True, help="Signed physical current in A per winding"
        )
    p.add_argument("--mesh-mm", type=float, default=5)
    p.add_argument("--padding-mm", type=float, default=170)
    p.add_argument("--mesh-cache", type=Path)
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args(argv)
    document = read_document(args.column)
    validate_document(document)
    recipe = circuit(document)
    encoded = json.dumps(recipe, sort_keys=True)
    mesh_id, mesh = cached_mesh(recipe, args.mesh_mm, args.padding_mm, args.mesh_cache)
    print(f"Mesh contains {len(mesh[0])} nodes and {len(mesh[1])} tetrahedra", flush=True)
    currents = dict(zip((*HP, *TL), (args.hp2, args.hp1, args.tl22, args.tl21)))
    air = {}
    corners, moments, report = solve(
        recipe,
        mesh,
        currents,
        backend=args.backend,
        air_solution=air,
        progress=lambda i, r: print(f"Newton {i}: {r:.3g}", flush=True),
    )
    report.update(
        mesh_id=mesh_id,
        material_size_mm=args.mesh_mm,
        exterior_padding_mm=args.padding_mm,
        coordinate_frame="column module local, xyz in metres",
        source_file=args.column.name,
    )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    _save(
        args.output,
        corners_m=corners,
        moments_Am2=moments,
        recipe=encoded,
        report=json.dumps(report, sort_keys=True),
        **air,
    )
    print(
        json.dumps({k: report[k] for k in ("currents_A", "material_peaks", "residual", "iterations")}),
        flush=True,
    )


def sample(path, output, step_mm, vacuum_only, transfer_scale=1.0):
    with np.load(path, allow_pickle=False) as data:
        solution = dict(data)
    report = json.loads(str(solution["report"]))
    if vacuum_only:
        solution["corners_m"] = np.empty((0, 4, 3))
        solution["moments_Am2"] = np.empty((0, 3))
        for key in ("probe_tl21_lens", "probe_tl22_lens"):
            report["currents_A"][key] *= transfer_scale
        report.update(material_peaks={}, iterations=0, residual=0.0, transfer_current_scale=transfer_scale)
        solution["report"] = np.array(json.dumps(report))
    elif transfer_scale != 1.0:
        raise ValueError("Changing a material winding current requires a fresh nonlinear solve")
    z, grid, field = harmonic_table(solution, step_mm * 0.001, vacuum_only=vacuum_only)
    # Interior probes staggered from the table. Compare tetrahedral quadrature
    # separately from registration, symmetry projection and field interpolation.
    rng = np.random.default_rng(20261121)
    angle = rng.uniform(-np.pi, np.pi, 256)
    r = 0.00288 * np.sqrt(rng.random(len(angle)))
    points = np.column_stack((r * np.cos(angle), r * np.sin(angle), rng.uniform(0.675, 0.875, len(r))))
    # Native discontinuous P1 gradient provides an independent local check
    # on the harmonic reconstruction of the continuous vacuum potential.
    truth = _coil_b(points, solution)
    if not vacuum_only:
        truth -= mu_0 * AirPotential(solution).evaluate(points, gradient=True)
    tetra = vacuum_field(points, solution)
    approx = RegularGridInterpolator((z, grid, grid), field)(points[:, [2, 1, 0]])
    raw = np.linalg.norm(truth - approx, axis=1)
    peak = np.max(np.linalg.norm(truth, axis=1))
    report.update(
        vacuum_only=vacuum_only,
        field_registration="module local SI, global origin z=0.45 m",
        axis_peak_T=float(np.max(abs(field[:, len(grid) // 2, len(grid) // 2, 2]))),
        integrated_axis_Bz2_T2m=float(np.trapezoid(field[:, len(grid) // 2, len(grid) // 2, 2] ** 2, z)),
        tetra_volume_field_max_delta_T=float(np.linalg.norm(truth - tetra, axis=1).max()),
        raw_mesh_to_symmetric_table_max_delta_T=float(raw.max()),
        raw_mesh_to_symmetric_table_rms_delta_T=float(np.sqrt(np.mean(raw**2))),
        validation_points_peak_T=float(peak),
    )
    _save(
        output,
        xy_m=grid,
        z_local_m=z,
        B_zyx_T=field,
        valid_radius_m=0.00288,
        report=json.dumps(report, sort_keys=True),
    )
    print(json.dumps({k: v for k, v in report.items() if k not in ("volumes_mm3", "turns")}), flush=True)


def sample_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("solution", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--step-mm", type=float, default=1.0)
    p.add_argument("--vacuum-only", action="store_true")
    p.add_argument(
        "--transfer-scale",
        type=float,
        default=1.0,
        help="Scale only the two transfer currents in the vacuum control",
    )
    args = p.parse_args(argv)
    if not 0 < args.step_mm <= 2:
        p.error("The central axial spacing must be in (0, 2] mm")
    sample(args.solution, args.output, args.step_mm, args.vacuum_only, args.transfer_scale)


def main(argv=None):
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Probe corrector magnetostatics")
    parser.add_argument("operation", choices=("solve", "sample"))
    selected = parser.parse_args(args[:1]).operation
    (solve_main if selected == "solve" else sample_main)(args[1:])


if __name__ == "__main__":
    main()
