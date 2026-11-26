"""Bind the installed HPC trim windings to the saved Nanoprobe Normal material circuit."""

import argparse
import json
from pathlib import Path

import numpy as np

from temsim.cad_catalog import part_solids
from temsim.lens_pose import effective_part_transform_mm, mechanical_part_transform_mm
from temsim.module_manifest import read_document
from temsim.multipole_winding import multipole_winding_definition
from temsim.part_materials import material_catalog, material_for_region
from temsim.paths import INSTRUMENT_CONFIG_ROOT


KEY = "probe_hpc_hexapole"
AXES = [KEY + "_" + component for component in "xy"]
CHANNELS = ["hp1", "hp2", "hpcx", "hpcy", "hpx", "hpy", "objective"]


def add_hpc_windings(recipe, bias):
    """Keep the same iron mesh; bind the two signed six-coil current distributions."""
    document = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    parts = {p["key"]: p for p in document["parts"]}
    row = parts[KEY]
    sleeve = parts[KEY + "_return_sleeve"]
    for part in (row, sleeve):
        if part.get("model_3d"):
            raise ValueError("The selected HPC circuit requires the native winding and sleeve geometry")
        for transform in (effective_part_transform_mm, mechanical_part_transform_mm):
            rotation, offset = transform(part, parts)
            np.testing.assert_allclose(rotation, np.eye(3), rtol=0, atol=1e-14)
            np.testing.assert_allclose(offset, 0, rtol=0, atol=1e-12)
    (solid,) = [region for region in part_solids(sleeve, parts) if region.material == "soft_magnetic"]
    sleeve_name = sleeve["key"] + "/" + solid.region
    (meshed,) = [s for s in recipe["solids"] if s["name"] == sleeve_name]
    material = material_for_region(sleeve, solid.region)
    if material is None:
        material = next(m for m in material_catalog() if m["material_key"] == "armco_pure_iron")
    curve = material["bh_material"]
    n = len(curve["b_t"])
    if (meshed["material"] != material["material_key"] or solid.shape != meshed["shape"]
        or meshed["curve"]["source_sha256"] != curve["source_sha256"]
        or any(meshed["curve"][k][:n] != curve[k] for k in ("h_a_per_m", "b_t"))):
        raise ValueError("HPC return-path geometry or material changed; regenerate the material mesh")
    winding = multipole_winding_definition(row, parts)
    turns = winding["turn_count_per_coil"]
    channels = {}
    for name, axis in zip(AXES, "xy", strict=True):
        coils = [c for c in winding["coils"] if c["group"] == axis]
        if len(coils) != 6 or turns <= 0 or name in recipe["sources"] or name in bias:
            raise ValueError(f"Invalid new HPC source {name}")
        angles = np.array([c["azimuth_deg"] for c in coils])
        signs = np.array([c["current_sign"] for c in coils])
        np.testing.assert_allclose(np.diff(angles), 60., atol=1e-12)
        np.testing.assert_array_equal(signs, [1, -1, 1, -1, 1, -1])
        np.testing.assert_allclose([c["center_z_mm"] for c in coils], row["local_center_z_mm"])
        recipe["sources"][name] = dict(kind="hexapole", turns=turns, coils=coils)
        bias[name] = 0.
        channels[axis] = dict(azimuth_deg=angles.tolist(), current_sign=signs.tolist())
    np.testing.assert_allclose(np.subtract(channels["y"]["azimuth_deg"], channels["x"]["azimuth_deg"]), 30.)
    # Both the historical tetrahedral solve and material_relay_bore use a local
    # coordinate origin 450 mm below the emitting tip.
    return dict(channels=channels, turns_per_coil=turns, sleeve=sleeve_name,
                sleeve_material=meshed["material"], material_source_sha256=curve["source_sha256"],
                winding_center_in_mesh_mm=row["local_center_z_mm"],
                winding_center_from_tip_mm=450. + row["local_center_z_mm"],
                sleeve_section_local_zr_mm=meshed["shape"]["section_zr_mm"],
                host_status=row["physical_host_status"], geometry_status=row["multipole_winding_geometry_status"])


def verify_zero_bore(reference, updated):
    """Compare the same 0.75 mm grid and 50 µm bore stencil in tesla."""
    from material_relay_bore import design, stencil
    from material_relay_fields import sha

    fields = []
    for path in (reference, updated):
        table = path.with_name(path.stem + "-dc-bore-0.75mm.npz")
        with np.load(table) as raw:
            if json.loads(str(raw["receipt"]))["solution_sha256"] != sha(path):
                raise ValueError("DC bore field belongs to a different native solution")
            fields.append((raw["z"], raw["coef"]))
    (z0, old), (z1, new) = fields
    np.testing.assert_array_equal(z0, z1)
    delta = np.einsum("pi,zic->zpc", design(stencil()), new-old)
    return dict(axial_points=len(z1), stencil_radius_um=50.,
                max_field_delta_T=float(np.linalg.norm(delta, axis=-1).max()))


def hpc_response_at_zero(solution):
    """Measure both signed normal/skew harmonics of the material-loaded differential field."""
    from material_relay_bore import design, stencil
    from material_relay_fields import sha

    points = stencil()[1:]
    angles = np.arctan2(points[:, 1], points[:, 0])
    harmonics, channels = [], {}
    for label in ("hpcx", "hpcy"):
        table = solution.with_name(solution.stem + "-" + label + "-bore-0.75mm.npz")
        with np.load(table) as raw:
            rec = json.loads(str(raw["receipt"]))
            if rec["solution_sha256"] != sha(solution) or rec["channel"] != label:
                raise ValueError(f"Stale differential bore response for {label}")
            z = raw["z"]
            b = np.einsum("pi,zic->zpc", design(points), raw["coef"])
        harmonic = np.mean((b[..., 0] + 1j*b[..., 1]) * np.exp(2j*angles), axis=1)
        i = np.argmax(np.abs(harmonic))
        channels[label] = dict(peak_from_tip_mm=float(z[i]*1000),
                               peak_harmonic_T_per_A=float(abs(harmonic[i])))
        harmonics.append(harmonic)
    phase = np.angle(np.vdot(*harmonics), deg=True)
    return dict(radius_um=50., channels=channels, relative_harmonic_phase_deg=float(phase))


def main():
    from material_relay_fields import WORK, digest, sha
    from material_relay_global import Global
    from material_relay_newton import Circuit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--response", action="store_true", help="Build all seven local tangent fields and bore tables")
    args = parser.parse_args()
    with np.load(args.reference) as saved:
        previous = json.loads(str(saved["report"]))
        old_recipe = json.loads(str(saved["recipe"]))
    winding = dict(previous["excitation"])
    winding.update({key: 0. for key in AXES})
    model = Global(9, include_hpc=True)
    old_sources = {k:v for k,v in model.recipe["sources"].items() if k not in AXES}
    if model.mesh_id != previous["mesh_id"] or dict(model.recipe, sources=old_sources) != old_recipe:
        raise ValueError("The original assembly, mesh or source circuits have changed")
    model.sources(selected=AXES)
    loads = {}
    for key in AXES:
        file = model.out / (key + ".json")
        loads[key] = json.loads(file.read_text())
        if (loads[key]["relative_peak_error"] >= .005
                or loads[key]["source_sha256"] != digest(model.recipe["sources"][key])):
            raise ValueError(f"The material-edge source failed its direct field comparison: {key}")
    path = WORK / ("global-" + digest(winding)[:16] + ".npz")
    if not path.exists():
        Circuit(model).solve(winding, args.reference, path)
    with np.load(args.reference) as old, np.load(path) as new:
        rep = json.loads(str(new["report"]))
        if (rep["excitation"] != winding or rep["circuit_sha256"] != digest(model.recipe)
                or json.loads(str(new["recipe"])) != model.recipe or rep["residual"] >= 2e-8):
            raise ValueError("Incomplete or incompatible HPC-zero solve")
        np.testing.assert_array_equal(new["corners_m"], old["corners_m"])
        check = {}
        for name in ("moments_Am2", "free_potential_a"):
            original, updated = old[name], new[name]
            np.testing.assert_allclose(original, updated, rtol=2e-8, atol=2e-11)
            check[name + "_max_abs_delta"] = float(np.max(np.abs(updated - original)))
    receipt = dict(original_sha256=sha(args.reference), field_sha256=sha(path), mesh_id=model.mesh_id,
                   source_recipe_sha256=digest(model.recipe), geometry=model.hpc_geometry,
                   source_loads=loads, field_comparison=check, relative_residual=rep["residual"])
    ticket = path.with_name(path.stem + "-hpc-zero.json")
    ticket.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Verified new HPC-zero circuit", path, check, flush=True)
    del model
    if args.response:
        import gc
        import cupy as cp
        from material_relay_response import solve
        from material_relay_bore import process
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        solve(Global(9, include_hpc=True), path, CHANNELS)
        process(path, ["dc", *CHANNELS], step=.75)
        receipt["bore_comparison"] = verify_zero_bore(args.reference, path)
        receipt["incremental_hpc_bore"] = hpc_response_at_zero(path)
        ticket.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
