"""Static magnetic-circuit audit of the installed CESCOR-family probe assembly."""

import argparse
import json
from pathlib import Path

import numpy as np

from temsim.cad_catalog import part_solids
from temsim.cad_round_winding import round_winding_envelope
from temsim.corrector_field_cli import _save, cached_mesh
from temsim.lens_pose import effective_part_transform_mm, mechanical_part_transform_mm
from temsim.module_manifest import read_document
from temsim.multipole_winding import multipole_winding_definition
from temsim.part_materials import material_catalog, material_for_region
from temsim.paths import INSTRUMENT_CONFIG_ROOT, project_root
from temsim.physics.probe_corrector_fem import circuit, solve


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "cescor_direct_probe_20261121" / "evidence"
WORK = project_root() / "outputs" / "probe-circuit-audit-20261122"
ADL, TL12 = "adapter_lens", "probe_tl12_lens"
ROUND = (ADL, "probe_tl22_lens", "probe_tl21_lens", TL12)
GLOBAL_OFFSET_MM = 450.0
UPPER_MM, LOWER_MM = 1387.5, 1412.5


def archive_inputs():
    name = "normal-final-corrector-solution.npz"
    with (ARCHIVE / name).open("rb") as stream:
        import hashlib

        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = json.loads((ARCHIVE / "normal-objective-probe.json").read_text())
    if digest != provenance["field_archive_sha256"][name]:
        raise ValueError("Historical operating point digest mismatch")
    with np.load(ARCHIVE / name) as data:
        return json.loads(str(data["recipe"])), json.loads(str(data["report"])), digest


def centered(row, parts):
    for pose in (mechanical_part_transform_mm, effective_part_transform_mm):
        rotation, translation = pose(row, parts)
        np.testing.assert_allclose(rotation, np.eye(3), rtol=0, atol=1e-14)
        np.testing.assert_allclose(translation, 0, rtol=0, atol=1e-12)
    if row.get("model_3d"):
        raise ValueError(f"Check custom material geometry on {row['key']} before using this study")


def audit_inputs():
    document = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    parts = {row["key"]: row for row in document["parts"]}
    recipe = circuit(document)
    historical, receipt, digest = archive_inputs()
    curves = {s["material"]: s["curve"] for s in historical["solids"]}
    reference = {r["material_key"]: r for r in material_catalog()}["armco_pure_iron"]
    for key in (ADL, TL12):
        coil = parts[key + "_excitation_coil"]
        centered(coil, parts)
        winding = round_winding_envelope(coil, parts)
        recipe["sources"][key] = dict(
            kind="round",
            turns=coil["winding_turn_count"],
            intervals=winding["intervals_mm"],
            inner=winding["inner_diameter_mm"] / 2,
            outer=winding["outer_diameter_mm"] / 2,
        )
    extra = [
        f"{key}_{role}"
        for key in (ADL, TL12)
        for role in ("upper_pole", "lower_pole", "pole_sleeves", "return_plates", "yoke", "housing")
    ]
    extra += [
        k
        for k, row in parts.items()
        if k.startswith("probe_") and row.get("mechanical_part_role") == "auxiliary_return_sleeve"
    ]
    for key in extra:
        row = parts[key]
        centered(row, parts)
        for solid in part_solids(row, parts):
            if solid.material == "soft_magnetic":
                assignment = material_for_region(row, solid.region) or reference
                recipe["solids"].append(
                    dict(
                        name=f"{key}/{solid.region}",
                        shape=solid.shape,
                        material=assignment["material_key"],
                        curve=assignment.get("bh_material"),
                    )
                )
    key = "probe_hpol_hexapole"
    centered(parts[key], parts)
    winding = multipole_winding_definition(parts[key], parts)
    for group in ("x", "y"):
        recipe["sources"][key + "_" + group] = dict(
            kind="hexapole",
            turns=winding["turn_count_per_coil"],
            coils=[c for c in winding["coils"] if c["group"] == group],
        )
    for key, source in historical["sources"].items():
        if json.dumps(recipe["sources"][key], sort_keys=True) != json.dumps(source, sort_keys=True):
            raise ValueError(f"Winding differs from the archived bias: {key}")
    for solid in recipe["solids"]:
        current = solid["curve"]
        if current is None:
            continue
        wide = curves[solid["material"]]
        n = len(current["b_t"])
        if (
            current["source_sha256"] != wide["source_sha256"]
            or current["h_a_per_m"] != wide["h_a_per_m"][:n]
            or current["b_t"] != wide["b_t"][:n]
        ):
            raise ValueError(f"Material assignment or B-H provenance changed: {solid['name']}")
        solid["curve"] = wide
    return recipe, dict(receipt["currents_A"]), digest


def run(spacing, padding, isolated=False):
    recipe, historical, archive_hash = audit_inputs()
    encoded = json.dumps(recipe, sort_keys=True)
    mesh_id, mesh = cached_mesh(recipe, spacing, padding, WORK / "meshes")
    print(f'{len(recipe["solids"])} solids, {len(mesh[0])} nodes, {len(mesh[1])} tetrahedra', flush=True)
    seed = None
    cases = [("bias", 0.0, 0.0), ("adl", 10.0, 0.0), ("both", 10.0, 10.0)]
    if isolated:
        historical = dict.fromkeys(historical, 0.0)
        cases = [("tl12-only", 0.0, 10.0)]
    for name, adl, tl12 in cases:
        path = WORK / f"{spacing:g}-{padding:g}-{name}.npz"
        currents = {**historical, ADL: adl, TL12: tl12}
        if path.is_file():
            with np.load(path) as old:
                receipt = json.loads(str(old["report"]))
                if receipt["mesh_id"] != mesh_id or receipt["currents_A"] != currents:
                    raise ValueError(f"Incompatible cache: {path}")
                seed = old["free_potential_a"]
            continue
        air, checkpoint = {}, {}
        corners, moments, receipt = solve(
            recipe,
            mesh,
            currents,
            backend="cuda",
            tolerance=2e-8,
            max_iterations=240,
            air_solution=air,
            potential_checkpoint=checkpoint,
            potential_seed_a=seed,
            progress=lambda i, r: print(f"{name} Newton {i}: {r:.4g}", flush=True),
        )
        receipt.update(mesh_id=mesh_id, mesh_mm=spacing, padding_mm=padding, historical_sha256=archive_hash)
        _save(
            path,
            recipe=encoded,
            report=json.dumps(receipt),
            corners_m=corners,
            moments_Am2=moments,
            **air,
            **checkpoint,
        )
        seed = checkpoint["free_potential_a"]
        print(f"Saved {path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-mm", type=float, default=5.0)
    parser.add_argument("--padding-mm", type=float, default=170.0)
    parser.add_argument(
        "--tl12-only", action="store_true", help="Energize TL12 only with all 60 material solids present"
    )
    args = parser.parse_args()
    run(args.mesh_mm, args.padding_mm, args.tl12_only)
