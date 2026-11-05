"""Read-only aperture electrostatics audit; no source or trajectory calculation."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from temsim.optics.column import default_state
from temsim.physics.closed_gun_field import (
    closed_field_request,
    electrode_boundary_arrays,
    field_cache_directory,
    load_cached_field,
    mesh_axes,
    request_digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    repo = REPO
    gun = default_state().electron_gun
    aperture = gun.dpa_aperture
    request = closed_field_request(gun)
    checks = []
    for name, value in (
        ("radius_mm", 1.0),
        ("plate_thickness_mm", 0.4),
        ("mechanical_center_from_tip_mm", 24.5),
        ("offset_x_mm", 0.1),
        ("enabled", False),
    ):
        previous = getattr(aperture, name)
        setattr(aperture, name, value)
        try:
            changed = closed_field_request(gun)
            checks.append(dict(parameter=name, original=previous, changed=value,
                               field_request_unchanged=changed == request))
        finally:
            setattr(aperture, name, previous)

    cached = load_cached_field(request, field_cache_directory())
    if cached is None:
        raise RuntimeError("Matching baseline cache is unavailable; no automatic solve is allowed by this audit.")
    points_mm = np.array([
        [2.5, 0.0, 23.9], [2.5, 0.0, 24.0], [2.5, 0.0, 24.1],
        [2.0, 0.0, 24.0], [3.0, 0.0, 24.0], [0.0, 0.0, 24.0],
    ])
    potential, electric = cached.interpolate(points_mm * 1e-3)

    # A boundary-array fixture, not a proposed or measured production part.
    fixture = dict(key="audit_annular_plate_fixture", start_m=0.0239,
                   stop_m=0.0241, inner_m=0.002, outer_m=0.006,
                   potential_rise_v=5200.0)
    proposed = deepcopy(request)
    proposed["rings"].append(fixture)
    r, z = mesh_axes(proposed)
    fixed, values = electrode_boundary_arrays(proposed, r, z)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    in_plate_z = (zz >= fixture["start_m"]) & (zz <= fixture["stop_m"])
    metal = in_plate_z & (rr >= fixture["inner_m"]) & (rr <= fixture["outer_m"])
    hole = in_plate_z & (rr < fixture["inner_m"])
    boundary_checks = dict(
        front_face_present=bool(np.any(z == fixture["start_m"])),
        back_face_present=bool(np.any(z == fixture["stop_m"])),
        inner_wall_present=bool(np.any(r == fixture["inner_m"])),
        metal_nodes_fixed=bool(metal.any() and fixed[metal].all()),
        metal_voltage_matches=bool(np.all(values[metal] == fixture["potential_rise_v"])),
        hole_nodes_free=bool(hole.any() and not fixed[hole].any()),
        field_identity_changes=request_digest(proposed) != request_digest(request),
    )
    assert all(boundary_checks.values()), boundary_checks
    output = dict(
        scope="Read-only field-input and cached-field audit; boundary-array fixture only. No source particles, trajectories, coherent calculations or new field solutions.",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        baseline_request_sha256=request_digest(request),
        baseline_request=request,
        aperture_runtime=vars(aperture),
        dependency_checks=checks,
        baseline_field_report=cached.report,
        cached_field_samples=[dict(xyz_mm=p.tolist(), potential_rise_v=float(v),
                                   electric_v_per_m=e.tolist())
                              for p, v, e in zip(points_mm, potential, electric)],
        front_back_difference_v=float(potential[2] - potential[0]),
        radial_2mm_to_3mm_difference_v=float(potential[4] - potential[3]),
        baseline_sample_limit="Numerical values diagnose the omitted conductor in this default model. No mesh convergence or real-device calibration is claimed.",
        boundary_fixture=fixture,
        boundary_fixture_limit="The 6 mm outer radius is a synthetic array-test dimension, not a measured or recommended aperture outline. No potential solution was calculated for this fixture.",
        boundary_array_checks=boundary_checks,
    )
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(path), dependency_checks=checks,
                         front_back_difference_v=output["front_back_difference_v"],
                         boundary_array_checks=boundary_checks), indent=2))


if __name__ == "__main__":
    main()
