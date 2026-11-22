"""Excite the installed HPol current channels and replay the D95-focus particle source."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from fit_threefold import EVIDENCE, HERE, design, fit_record, read_sample, ring_measure
from temsim.corrector_field_cli import _save, cached_mesh, sample as sample_field
from temsim.lens_pose import effective_part_transform_mm, mechanical_part_transform_mm
from temsim.module_manifest import read_document
from temsim.multipole_winding import multipole_winding_definition
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.probe_corrector_fem import circuit, solve


WORK = HERE.parents[2] / "outputs/cescor-20261121"
OUT = HERE.parents[2] / "outputs/cescor-a2-20261121"
FINE_KEYS = ("probe_hpol_hexapole_x", "probe_hpol_hexapole_y")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(recipe):
    return sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def propose():
    """Hold HP2 fixed and minimize the fine-current norm in the measured response."""
    base = np.array(read_json(EVIDENCE / "threefold-fit.json")["full_fit"]["coefficients"]["A2"]["cos_sin_mm"])

    def measured(name):
        receipt = read_json(OUT / (name + ".json"))
        return np.array(receipt["fit"]["coefficients"]["A2"]["cos_sin_mm"])

    # From (+1,-1) to (+1,0), only HP2 changes. The half-differential
    # measurement gives the independent combination HP1 minus HP2.
    hp2 = measured("hp1-reference") - base
    hp1 = 2*(base - measured("hp-half")) + hp2
    fine = np.column_stack([(measured("hpol-" + axis) - base)/.1 for axis in ("x", "y")])
    fine_at_zero_step = np.linalg.solve(fine, -base)
    fine_per_hp1 = np.linalg.solve(fine, -hp1)
    delta_hp1 = -np.dot(fine_at_zero_step, fine_per_hp1) / np.dot(fine_per_hp1, fine_per_hp1)
    adjustment = fine_at_zero_step + delta_hp1*fine_per_hp1
    print(json.dumps(dict(hp1_A=float(1+delta_hp1), hp2_A=-1., hpol_xy_A=adjustment.tolist(),
                          hp1_response_mm_per_A=hp1.tolist(), hp2_response_mm_per_A=hp2.tolist(),
                          hpol_response_mm_per_A=fine.tolist(), hpol_only_A=fine_at_zero_step.tolist()),
                     indent=2))


def solve_hpol(name, xy_a, hp1=None, hp2=None, mesh_check=False, tl21=None, tl22=None, mesh_mm=None):
    summary = read_json(EVIDENCE / "comparison.json")
    document = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    recipe = circuit(document)
    if digest(recipe) != summary["corrector_circuit_sha256"]:
        raise ValueError("The installed main circuit differs from the selected D95 operating point")
    base_table = WORK / "fine-field.npz"
    if sha256(base_table.read_bytes()).hexdigest() != summary["corrector_table_sha256"]:
        raise ValueError("The reference corrector field differs from the archived D95 field")
    with np.load(base_table) as baseline:
        reference = json.loads(str(baseline["report"]))
    if mesh_check:
        with np.load(WORK / "refined-table.npz") as other:
            independent = json.loads(str(other["report"]))
        if (independent["circuit_sha256"] != reference["circuit_sha256"] or
                independent["exterior_padding_mm"] != reference["exterior_padding_mm"] or
                independent["material_size_mm"] != 5.):
            raise ValueError("Expected the matched 5 mm material mesh at the original padding")
        reference = independent
    if mesh_mm is None:
        with np.load(WORK / "meshes" / (reference["mesh_id"] + ".npz")) as f:
            mesh = f["xyz"], f["tet"], f["names"][f["index"]], f["far"]
    else:
        mesh_id, mesh = cached_mesh(recipe, mesh_mm, reference["exterior_padding_mm"], WORK / "meshes")
        reference.update(mesh_id=mesh_id, material_size_mm=mesh_mm)
    parts = {p["key"]: p for p in document["parts"]}
    row = parts["probe_hpol_hexapole"]
    for pose in (mechanical_part_transform_mm, effective_part_transform_mm):
        rotation, translation = pose(row, parts)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-14, rtol=0)
        np.testing.assert_allclose(translation, 0., atol=1e-12, rtol=0)
    definition = multipole_winding_definition(row, parts)
    currents = dict(summary["corrector_currents_A"])
    for key, value in (("probe_hp1_hexapole", hp1), ("probe_hp2_hexapole", hp2),
                       ("probe_tl21_lens", tl21), ("probe_tl22_lens", tl22)):
        if value is not None:
            currents[key] = value
    for key, group, amps in zip(FINE_KEYS, ("x", "y"), xy_a, strict=True):
        coils = [c for c in definition["coils"] if c["group"] == group]
        if len(coils) != 6 or {c["current_sign"] for c in coils} != {-1, 1}:
            raise ValueError(f"Missing signed closed windings for {key}")
        recipe["sources"][key] = dict(kind="hexapole", turns=definition["turn_count_per_coil"], coils=coils)
        currents[key] = amps
    # The cached tetrahedra describe the iron and air domains. Impressed HPol
    # currents add source terms in this same geometry, without remeshing.
    air = {}
    corners, moments, report = solve(recipe, mesh, currents, backend="cuda", air_solution=air,
                                     progress=lambda i, r: print(f"Newton {i}   {r:.3g}", flush=True))
    report.update(mesh_id=reference["mesh_id"], material_size_mm=reference["material_size_mm"],
                  exterior_padding_mm=reference["exterior_padding_mm"],
                  base_circuit_sha256=summary["corrector_circuit_sha256"],
                  coordinate_frame=reference["coordinate_frame"], source_file=reference["source_file"],
                  auxiliary_host_status=row["physical_host_status"])
    solution = OUT / f"{name}-solution.npz"
    _save(solution, corners_m=corners, moments_Am2=moments, recipe=json.dumps(recipe, sort_keys=True),
          report=json.dumps(report, sort_keys=True), **air)
    print("Sampling nonlinear material field", flush=True)
    sample_field(solution, OUT / f"{name}-field.npz", 0.5, False)


def refine_grid(solution_path, output):
    """Halve the transverse table pitch at fixed material solution and axial sampling."""
    from temsim.physics.corrector_vacuum_field import harmonic_table

    with np.load(solution_path, allow_pickle=False) as data:
        solution = dict(data)
    report = json.loads(str(solution["report"]))
    z, xy, field = harmonic_table(solution, 0.0005, grid_half=24)
    report.update(vacuum_only=False, field_registration="module local SI, global origin z=0.45 m",
                  transverse_pitch_mm=float(1000*(xy[1]-xy[0])))
    _save(output, xy_m=xy, z_local_m=z, B_zyx_T=field, valid_radius_m=.00288,
          report=json.dumps(report, sort_keys=True))


def track(name, field_path, n, step, through_focus, interpolation="linear"):
    from column_current import SURFACE, current_ledger, field_array, metrics, source_digest, source_rate
    from finite_probe import axial, bundle

    summary = read_json(EVIDENCE / "comparison.json")
    baseline = summary["runs"]["d95"]
    winding = read_json(EVIDENCE / baseline["winding_currents_file"])
    q = np.asarray(winding["currents_A"])
    ni = winding["objective_At"]
    rate = source_rate()
    with np.load(EVIDENCE / baseline["trajectory_file"]) as original:
        obs = original["z_m"]
    mesh, tracker, bg, coils = bundle(step, obs, ni, n, rate, field_path, transverse_interpolation=interpolation)
    OUT.mkdir(parents=True, exist_ok=True)
    if through_focus:
        receipt = axial(q, ni, n, step, rate, name, mesh.intervals, field_path, OUT,
                        transverse_interpolation=interpolation)
    else:
        # Keep every original integration node. Copy out only the specimen
        # plane for each full-current trial during response calibration.
        mesh.saved = np.array([np.flatnonzero(mesh.z == SURFACE).item()])
        phases, loss, _, _ = tracker.run(field_array(mesh, bg, coils, q), keep_history=False)
        phase = phases[-1]
        live = loss == -1
        surface = metrics(phase[live], tracker.weights[live], rate)
        loss_owner = np.full(n, -1, dtype=np.int32)
        loss_owner[~live] = mesh.owner[loss[~live]]
        _save(OUT / f"{name}.npz", phase=phase, loss=loss, ids=tracker.ids, weights=tracker.weights,
              energy_offset_eV=tracker.de, loss_owner=loss_owner, z_m=np.array([SURFACE]),
              profiles=np.array([[surface["D95_nm"], surface["rms_radius_nm"], surface["core_50pm_nA"]]]))
        receipt = dict(emitted=n, survived=int(live.sum()), effective=int(live.sum()), surface=surface,
                       current_ledger=current_ledger(mesh, tracker, loss), max_step_mm=step,
                       source_digest=source_digest(tracker), objective_At=ni, currents_A=q.tolist())
    if n == baseline["emitted"] and receipt["source_digest"] != baseline["source_digest"]:
        raise ValueError("Mismatch of original gun-exit particles or current weights")
    phase_path = OUT / f"{name}.npz"
    sample = read_sample(phase_path, receipt, summary)
    basis, names = design(sample)
    record = fit_record(sample, basis, names, np.ones(len(sample.ids), dtype=bool))
    with np.load(field_path) as f:
        field = json.loads(str(f["report"]))
    receipt.update(fit=record, ring=ring_measure(sample.xy_um, sample.current_na),
                   trajectory_file=phase_path.name, trajectory_sha256=sha256(phase_path.read_bytes()).hexdigest(),
                   field_file=field_path.name, field_sha256=sha256(field_path.read_bytes()).hexdigest(),
                   field=field, transverse_interpolation=interpolation)
    (OUT / f"{name}.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(name, record["coefficients"], receipt["surface"], receipt["ring"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("propose")
    refine = sub.add_parser("refine-grid")
    refine.add_argument("solution", type=Path)
    refine.add_argument("output", type=Path)
    field_parser = sub.add_parser("field")
    field_parser.add_argument("name")
    field_parser.add_argument("x_A", type=float)
    field_parser.add_argument("y_A", type=float)
    field_parser.add_argument("--hp1", type=float, help="Main HP1 signed current (A)")
    field_parser.add_argument("--hp2", type=float, help="Main HP2 signed current (A)")
    field_parser.add_argument("--tl21", type=float, help="TL21 signed current (A)")
    field_parser.add_argument("--tl22", type=float, help="TL22 signed current (A)")
    mesh_options = field_parser.add_mutually_exclusive_group()
    mesh_options.add_argument("--mesh-check", action="store_true", help="Replay on the existing 5 mm material mesh")
    mesh_options.add_argument("--mesh-mm", type=float, help="Generate an independent material mesh at fixed padding")
    trace_parser = sub.add_parser("trace")
    trace_parser.add_argument("name")
    trace_parser.add_argument("field", type=Path)
    trace_parser.add_argument("--count", type=int, default=32768)
    trace_parser.add_argument("--step", type=float, default=.075, help="Maximum axial integration step in mm")
    trace_parser.add_argument("--through-focus", action="store_true")
    trace_parser.add_argument("--interpolation", choices=("linear", "cubic"), default="linear",
                              help="Transverse field interpolation (linear replays the original study)")
    args = parser.parse_args()
    if args.operation == "field":
        solve_hpol(args.name, (args.x_A, args.y_A), args.hp1, args.hp2, args.mesh_check, args.tl21, args.tl22,
                   args.mesh_mm)
    elif args.operation == "trace":
        track(args.name, args.field, args.count, args.step, args.through_focus, args.interpolation)
    elif args.operation == "refine-grid":
        refine_grid(args.solution, args.output)
    else:
        propose()
