"""Compare axial flux through the upper objective's tube joint at 1 A-turn."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY
from temsim.optics.column import default_state
from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_details
from temsim.physics.lens_field_provider import lens_geometry_binding


def axial_flux_wb(solution, z_mm, radii_m):
    """Integrate axial flux over the annulus using its boundary vector potential."""
    index = int(np.argmin(abs(solution.z_m - z_mm * 1e-3)))
    if not np.isclose(solution.z_m[index], z_mm * 1e-3, rtol=0., atol=1e-12):
        raise ValueError("The flux plane must lie on the solved axial grid")
    potential = np.interp(radii_m, solution.r_m, solution.a_phi_tm[:, index])
    return float(np.diff(2 * np.pi * radii_m * potential)[0])


def compare_tube_materials():
    state = default_state()
    assembly = state._resolved_assembly
    binding = lens_geometry_binding(state, "objective_lens")
    geometry = json.loads(binding.canonical_geometry_json)
    lens = geometry["lens_assembly"]
    tube, = (s for s in assembly.vacuum_liner_segments if s.key == CONTINUOUS_TUBE_KEY)
    end = tube.end_z_mm
    upper = assembly.part("objective_upper_pole")
    if not np.isclose(upper.start_z_mm, end, rtol=0., atol=1e-9):
        raise ValueError("The ideal-contact comparison requires coincident tube and pole ends")
    sample = assembly.part("sample").center_z_mm
    wall_radii = 0.0005 * np.array([tube.inner_diameter_mm, tube.outer_diameter_mm])
    contact_radii = 0.0005 * np.array([
        max(tube.inner_diameter_mm, upper.data["pole_mounting_shank_inner_diameter_mm"]),
        min(tube.outer_diameter_mm, upper.data["pole_vacuum_connector_outer_diameter_mm"]),
    ])
    if contact_radii[1] <= contact_radii[0]:
        raise ValueError("The tube and pole require positive annular contact area")
    yoke = next(p for p in lens["parts"] if p["key"] == "objective_lens_yoke")
    outer_radius = yoke["data"]["mechanical_outer_diameter_mm"] / 2
    center = (yoke["start_z_mm"] + yoke["end_z_mm"]) / 2
    half_length = (yoke["end_z_mm"] - yoke["start_z_mm"]) / 2
    # Isolate the objective and the final straight 40 mm of the passive tube.
    # Only the existing objective winding carries imposed current.
    lens["magnetostatic_neighbours"] = [{
        "key": "upper_objective_terminal_tube", "start_z_mm": end - 40., "end_z_mm": end,
        "data": {"mechanical_inner_diameter_mm": tube.inner_diameter_mm,
                 "mechanical_outer_diameter_mm": tube.outer_diameter_mm,
                 "material_class": "trial_tube_wall"},
    }]
    binding = replace(binding, canonical_geometry_json=json.dumps(geometry, sort_keys=True))
    results = []
    for dr, dz, padding in ((1., 1.5, 2.5), (.5, .75, 2.5), (.5, .75, 3.5), (.3, .5, 3.5)):
        radial_limit = outer_radius * padding
        axial_half_span = half_length * padding
        r_grid = np.unique(np.r_[np.linspace(0., radial_limit, round(radial_limit / 3) + 1),
                                 np.arange(0., 25., dr), wall_radii * 1000])
        z_grid = np.unique(np.r_[np.linspace(center - axial_half_span, center + axial_half_span,
                                             round(2 * axial_half_span / 5) + 1),
                                 np.arange(end - 42., sample + 10., dz), end - 2., sample])
        for tube_mu in (1., 1000.):
            settings = dict(relative_permeability=1000., ampere_turns=1., padding_factor=padding,
                            material_permeabilities={"trial_tube_wall": tube_mu},
                            validation_grid_axes_m=[(axis * 1e-3).tolist() for axis in (r_grid, z_grid)])
            _, solution = solve_geometry_field_details(binding, settings)
            at_sample = int(np.argmin(abs(solution.z_m - sample * 1e-3)))
            results.append(dict(
                dr_mm=dr, dz_mm=dz, padding=padding, tube_mu=tube_mu,
                nr=len(solution.r_m), nz=len(solution.z_m),
                wall_flux_Wb=axial_flux_wb(solution, end - 2., wall_radii),
                joint_flux_Wb=axial_flux_wb(solution, end, contact_radii),
                specimen_Bz_T=float(solution.bz_t[0, at_sample]), residual=solution.relative_residual,
            ))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save the matched mesh results as JSON.")
    args = parser.parse_args()
    report = json.dumps(compare_tube_materials(), indent=2, allow_nan=False)
    if args.output is None:
        print(report)
    else:
        args.output.write_text(report + "\n", encoding="utf-8")
