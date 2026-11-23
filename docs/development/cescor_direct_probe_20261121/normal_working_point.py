"""Load the current Nanoprobe Normal illumination model or its archived vacuum control."""

from hashlib import sha256
import json

import numpy as np

from adjust_a2 import refine_grid
from column_current import ACTIVE, OWNER, START, source_digest
from illuminated_focus import column_planes
from refocus_a2 import WORK as OBJECTIVE_WORK, objective_receipt
from relay_transfer import HERE, ROOT, WORK, Native
from scan_waist_audit import archived_scan


EVIDENCE = HERE / "evidence"
AUDIT = HERE.parent / "probe_circuit_audit_20261122"
CACHE = ROOT / "outputs/probe-circuit-audit-20261122"


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_table(path, expected_sha, solution, solution_sha):
    if file_hash(solution) != solution_sha:
        raise ValueError(f"Unexpected magnetic solution: {solution}")
    if not path.is_file() or file_hash(path) != expected_sha:
        refine_grid(solution, path)
        if file_hash(path) != expected_sha:
            raise ValueError(f"The regenerated tracking table differs from the recorded field: {path}")
    return path


def operating_point(*, archived=False, step_mm=None):
    """Replace the covered vacuum groups while retaining the physical source and other drives."""
    config = json.loads((EVIDENCE / "normal-current-working-point.json").read_text())
    base_text = (EVIDENCE / config["base"]).read_text(encoding="utf-8")
    if sha256(base_text.encode()).hexdigest() != config["base_sha256_lf"]:
        raise ValueError("The inherited Nanoprobe Normal working point has changed")
    base = json.loads(base_text)
    final = base[config["base_stage"]]
    objective = final["objective_At"]
    receipt = objective_receipt(objective)
    objective_file = OBJECTIVE_WORK / "column" / f"armco-{abs(objective):.12g}-gap-fine-p8.npz"
    if (receipt["curve_sha256"] != final["objective_material_sha256"] or
            file_hash(objective_file) != base["field_archive_sha256"]["normal-final-objective-field.npz"]):
        raise ValueError("The installed objective is different from the inherited material field")
    scan = archived_scan()
    if [scan["kick_x_mrad"], scan["kick_y_mrad"]] != config["scan_command_mrad"]:
        raise ValueError("Unexpected scan command")
    expected_currents = dict(final["corrector_A"])
    if archived:
        filename = "normal-final-corrector-solution.npz"
        path = verified_table(WORK / "normal-core-field.npz", final["corrector_table_sha256"],
                              EVIDENCE / filename, base["field_archive_sha256"][filename])
    else:
        path = verified_table(CACHE / "normal-joint-field.npz", config["field_table_sha256"],
                              EVIDENCE / config["solution"], config["solution_sha256"])
        expected_currents.update(config["replacement_current_A"])
    with np.load(path, allow_pickle=False) as data:
        magnetic = json.loads(str(data["report"]))
        if magnetic["currents_A"] != expected_currents:
            raise ValueError("FEM currents do not describe the requested operating point")
        if not archived:
            if any(magnetic[k] != config[k] for k in ("circuit_sha256", "mesh_id")):
                raise ValueError("Incorrect magnetic geometry or material mesh")
            if any(magnetic["turns"][k] != turns for k, turns in config["replacement_turns"].items()):
                raise ValueError("Unexpected physical winding turns")
            grid = data["xy_m"]
            mid = len(grid) // 2
            if grid[mid] != 0.0:
                raise ValueError("Field grid is off center")
            axis = data["B_zyx_T"][:, mid, mid, 2]
            with np.load(AUDIT / "axial-profiles.npz", allow_pickle=False) as audit:
                sampled = np.interp(audit["z_mm"], (START + data["z_local_m"])*1000, axis)
                np.testing.assert_allclose(sampled, audit["total"], atol=5e-12, rtol=0)
    model = Native(path, base["max_step_mm"] if step_mm is None else step_mm, base["source_packets"],
                   objective, column_planes())
    t = model.t
    if (source_digest(t) != base["source_digest"] or t.current != base["source_current_A"] or
            np.unique(t.ids[t.weights > 0]).size != base["source_packets"]):
        raise ValueError("The gun phase space, original source weights or current changed")
    labels, values = zip(*base["fixed_windings"], strict=True)
    np.testing.assert_array_equal(labels, OWNER[ACTIVE])
    expected_mask = np.isin(labels, [] if archived else list(config["replacement_current_A"]))
    np.testing.assert_array_equal(model.m.fem_round_windings, expected_mask)
    vacuum_A = np.asarray(values).copy()
    vacuum_A[expected_mask] = 0.0
    return model, vacuum_A, base
