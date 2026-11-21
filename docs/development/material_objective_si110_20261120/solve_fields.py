"""Solve the installed objective circuit on a bore- and pole-refined R-Z mesh."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from temsim.optics.column import default_state
from temsim.part_materials import is_magnetostatic_body, material_catalog
from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_details
from temsim.physics.lens_field_provider import lens_geometry_binding

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / "outputs" / HERE.name
PREVIOUS = HERE.parent / "objective_pole_study_20261119"
sys.path.insert(0, str(PREVIOUS))
from fields import axes


def solve(ampere_turns, level, padding):
    started = perf_counter()
    state = default_state()
    # The selected circuit is isolated. The other excitations use their own
    # prescribed winding fields, including their fringe at the objective.
    binding = lens_geometry_binding(state, "objective_lens")
    geometry = json.loads(binding.canonical_geometry_json)
    circuit = geometry["lens_assembly"]
    parts = {part["key"]: part for part in circuit["parts"]}
    assert not circuit.get("magnetostatic_neighbours")
    required = {"objective_upper_pole", "objective_lower_pole", "objective_pole_sleeves",
                "objective_return_plates", "objective_terminal_vacuum_tube", "objective_lens_yoke"}
    assert required <= parts.keys()
    used = [p for p in parts.values() if is_magnetostatic_body(p["data"])
            or p["data"].get("mechanical_profile") == "magnetic_excitation_coil"]
    low = min(p["start_z_mm"] for p in used)
    high = max(p["end_z_mm"] for p in used)
    outer = max(p["data"]["mechanical_outer_diameter_mm"] / 2 for p in used)
    subdivisions = {"gap": 18, "gap-fine": 32}
    background = "interface" if level in subdivisions else level
    r, z = axes((low + high) / 2, (high - low) / 2, outer, background, padding)
    if level in subdivisions:
        from temsim.physics.axisymmetric_magnetostatics import _refine_pole_axes
        r, z = _refine_pole_axes(r, z, used, subdivisions[level])
    material = next(m["bh_material"] for m in material_catalog() if m["material_key"] == "armco_pure_iron")
    settings = dict(solver="axisymmetric_nonlinear_fem", bh_material=material,
                    channel_ampere_turns={"objective_lens": ampere_turns}, padding_factor=padding,
                    validation_grid_axes_m=[r.tolist(), z.tolist()])
    field, solution = solve_geometry_field_details(binding, settings)
    r, z = solution.r_m, solution.z_m
    bore = parts["objective_upper_pole"]["data"]["mechanical_bore_diameter_mm"] * 0.0005
    assert np.isclose(bore, 0.00288)
    inside = (r > 0) & (r <= bore * (1 + 1e-14))
    design = np.polynomial.polynomial.polyvander((r[inside] / bore) ** 2, 3)
    quotient = solution.a_phi_tm[inside] / r[inside, None]
    coefficients, _, rank, _ = np.linalg.lstsq(design, quotient, rcond=None)
    if rank != 4:
        raise ValueError("Insufficient radial resolution in the physical bore")
    bz = 2 * coefficients[0]
    surface = state.sample.upper_surface_z_mm * 1e-3
    tag = f"armco-{ampere_turns:g}-{level}-p{padding:g}"
    metadata = dict(ampere_turns=ampere_turns, level=level, padding=padding, shape=[len(r), len(z)],
                    iterations=solution.iterations, residual=solution.relative_residual,
                    peak_material_T=solution.peak_material_t, peak_axis_T=float(abs(bz).max()),
                    sample_Bz_T=float(np.interp(surface, z, bz)),
                    field_squared_integral_T2m=float(np.trapezoid(bz**2, z)),
                    max_potential_fit_error_T=float(abs(design @ coefficients - quotient).max()),
                    curve_sha256=material["source_sha256"], field_digest=field.provenance.source_sha256,
                    installed_geometry_digest=sha256(binding.canonical_geometry_json.encode()).hexdigest(),
                    parts=sorted(parts), elapsed_s=perf_counter() - started)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / (tag + ".npz"), r=r, z=z, a_phi=solution.a_phi_tm)
    (CACHE / (tag + ".json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(tag, metadata["shape"], metadata["peak_material_T"], metadata["sample_Bz_T"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current", type=float, nargs="+", default=[3000])
    p.add_argument("--level", choices=["base", "fine", "finer", "reference", "gap", "gap-fine"], default="base")
    p.add_argument("--padding", type=float, default=8)
    args = p.parse_args()
    for value in args.current:
        solve(value, args.level, args.padding)
