"""Matched R-Z objective solves for three nose contours and a separate tube contrast."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import make_interp_spline

from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, TERMINAL_KEY
from temsim.magnetic_materials import lens_material_defaults
from temsim.optics.column import default_state
from temsim.part_materials import material_catalog
from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_details
from temsim.physics.lens_field_provider import lens_geometry_binding
from temsim.pole_geometry import recessed_objective_section

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / "outputs" / HERE.name
# (face OD, knee OD, terminal wall material); diameters in mm.
CASES = {
    "photo": (8.0, 28.0, "nonmagnetic_stainless_steel"),
    "narrow": (7.2, 22.0, "nonmagnetic_stainless_steel"),
    "wide": (10.0, 36.0, "nonmagnetic_stainless_steel"),
    "photo_iron_tube": (8.0, 28.0, "femm_pure_iron"),
}


def axes(center, half, outer, level, padding):
    radial_limit = outer * padding
    low, high = center - half * padding, center + half * padding
    dr, dz = {
        "base": (0.30, 0.75),
        "fine": (0.20, 0.50),
        "finer": (0.12, 0.30),
        "interface": (0.20, 0.50),
        "reference": (0.12, 0.30),
    }[level]
    local_limits = (-130.0, 111.0) if level in ("interface", "reference") else (-74.0, 66.0)
    dense_r = 12.0 if level == "reference" else 20.0
    # Keep the local grid fixed on expanding the artificial exterior.
    r = np.unique(
        np.r_[
            0.0,
            radial_limit,
            np.arange(0.0, radial_limit, 12.0),
            np.arange(0.0, 91.0, 2.0),
            np.arange(0.0, dense_r, dr),
            np.arange(dense_r, 20.0, 0.5),
            2.88,
            3.6,
            4.0,
            5.0,
            9.6,
            11.0,
            14.0,
            18.0,
        ]
    )
    z = np.unique(
        np.r_[
            low,
            high,
            np.arange(-2000.0, 5000.0, 22.0),
            center + np.arange(-210.0, 211.0, 4.0),
            center + np.arange(*local_limits, dz),
        ]
    )
    return [r * 0.001, z[(z >= low) & (z <= high)] * 0.001]


def solve(case, ni, level, padding):
    began = perf_counter()
    state = default_state()
    assembly = state._resolved_assembly
    binding = lens_geometry_binding(state, "objective_lens")
    geometry = json.loads(binding.canonical_geometry_json)
    circuit = geometry["lens_assembly"]
    # Reproduce the 19 November geometry and its uniform FEMM iron law.
    # The terminal wall is defined explicitly by this historical study below.
    later_parts = {TERMINAL_KEY, "objective_pole_sleeves", "objective_return_plates"}
    circuit["parts"] = [row for row in circuit["parts"] if row["key"] not in later_parts]
    for row in circuit["parts"]:
        row["data"].pop("material_regions", None)
    parts = {p["key"]: p for p in circuit["parts"]}
    # This comparison preceded the lower-pole mirror in the current catalog.
    lower = parts["objective_lower_pole"]["data"]
    lower.update(pole_piece_geometry_style="objective_vertical_back_inserted_shank_tapered_nose",
                 pole_nose_axial_length_mm=27.5584, pole_cone_angle_to_axis_deg=57.94,
                 pole_stem_outer_diameter_mm=60., pole_mounting_shank_inner_diameter_mm=5.76,
                 pole_vacuum_connector_axial_length_mm=6.)
    p = parts["objective_upper_pole"]
    face, knee, material = CASES[case]
    p["data"].update(mechanical_tip_diameter_mm=face, pole_nose_knee_diameter_mm=knee)
    section = recessed_objective_section(p["data"], p["start_z_mm"], p["end_z_mm"], key=p["key"])
    (tube,) = (s for s in assembly.vacuum_liner_segments if s.key == CONTINUOUS_TUBE_KEY)
    assert np.isclose(tube.end_z_mm, p["start_z_mm"], rtol=0, atol=1e-9)
    wall = dict(
        mechanical_inner_diameter_mm=tube.inner_diameter_mm,
        mechanical_outer_diameter_mm=tube.outer_diameter_mm,
        material_regions={"body": next(m for m in material_catalog() if m["material_key"] == material)},
    )
    circuit["magnetostatic_neighbours"] = [
        dict(key="terminal_tube", start_z_mm=tube.end_z_mm - 40.0, end_z_mm=tube.end_z_mm, data=wall)
    ]
    binding = replace(binding, canonical_geometry_json=json.dumps(geometry, sort_keys=True))
    yoke = parts["objective_lens_yoke"]
    center = (yoke["end_z_mm"] + yoke["start_z_mm"]) / 2
    half = (yoke["end_z_mm"] - yoke["start_z_mm"]) / 2
    r, z = axes(center, half, yoke["data"]["mechanical_outer_diameter_mm"] / 2, level, padding)
    material_bh = lens_material_defaults()["bh_material"]
    settings = dict(
        solver="axisymmetric_nonlinear_fem",
        bh_material=material_bh,
        channel_ampere_turns={"objective_lens": ni},
        padding_factor=padding,
        validation_grid_axes_m=[r.tolist(), z.tolist()],
    )
    field, sol = solve_geometry_field_details(binding, settings)
    # Regular bore reconstruction: obtain both field components from a single Aphi/r fit.
    r, z = sol.r_m, sol.z_m
    fit_r = 0.00288
    mask = (r > 0) & (r <= fit_r * (1 + 1e-14))
    design = np.polynomial.polynomial.polyvander((r[mask] / fit_r) ** 2, 3)
    coeff, _, rank, _ = np.linalg.lstsq(design, sol.a_phi_tm[mask] / r[mask, None], rcond=None)
    assert rank == 4
    resid = design @ coeff - sol.a_phi_tm[mask] / r[mask, None]
    zsample = state.sample.upper_surface_z_mm * 0.001
    joint_r = (
        np.array(
            [
                p["data"]["pole_mounting_shank_inner_diameter_mm"],
                min(tube.outer_diameter_mm, p["data"]["pole_vacuum_connector_outer_diameter_mm"]),
            ]
        )
        * 0.0005
    )
    j = np.argmin(abs(z - tube.end_z_mm * 0.001))
    assert abs(z[j] - tube.end_z_mm * 0.001) < 1e-12
    flux = np.diff(2 * np.pi * joint_r * np.interp(joint_r, r, sol.a_phi_tm[:, j])).item()
    spline = make_interp_spline(z, 2 * coeff[0])
    span = np.linspace(zsample - 0.21, zsample + 0.21, 4201)
    bz = spline(span)
    row = dict(
        case=case,
        ni=ni,
        level=level,
        padding=padding,
        nodes=[len(r), len(z)],
        sample_mm=zsample * 1000,
        B_sample_T=float(spline(zsample)),
        peak_axis_T=float(np.max(abs(bz))),
        peak_axis_offset_mm=float((span[np.argmax(abs(bz))] - zsample) * 1000),
        B2_integral_T2m=float(np.trapezoid(bz**2, span)),
        peak_material_T=sol.peak_material_t,
        joint_flux_Wb=flux,
        residual=sol.relative_residual,
        iterations=sol.iterations,
        fit_error_T=float(abs(resid).max()),
        elapsed_s=perf_counter() - began,
        material_hash=material_bh["source_sha256"],
        field_hash=field.provenance.source_sha256,
        section_mm=section,
        current_fraction=parts["objective_lens_excitation_coil"]["data"]["coil_ampere_turn_fraction"],
    )
    tag = f"{case}-{ni:g}-{level}-p{padding:g}"
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / (tag + ".npz"), r=r, z=z, a_phi=sol.a_phi_tm, coeff=coeff, fit_r=fit_r)
    (CACHE / (tag + ".json")).write_text(json.dumps(row, indent=2), encoding="utf8")
    print(
        tag,
        {k: row[k] for k in ("B_sample_T", "peak_axis_T", "peak_material_T", "joint_flux_Wb", "elapsed_s")},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ni", type=float, default=3000.0)
    parser.add_argument(
        "--level", choices=["base", "fine", "finer", "interface", "reference"], default="base"
    )
    parser.add_argument("--padding", type=float, default=8.0)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    for name in args.cases:
        solve(name, args.ni, args.level, args.padding)
