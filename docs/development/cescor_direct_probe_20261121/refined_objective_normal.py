"""Relaunch the Nanoprobe Normal effective source in the globally refined objective field."""

import argparse
from hashlib import sha256
import json
import os
import shutil

import numpy as np
from scipy.interpolate import make_interp_spline

from condenser_source_fields import EV, FEM, FINAL, HERE, OBJ, OWNER, Fields, check_file, sha, write


PARENT = EV / "condenser-pair"
EVIDENCE = EV / "refined-objective-normal"
WORK = HERE.parents[2] / "outputs/refined-objective-normal-20261122"
PARENT_SHA = "e3a073b36f5a341dcf9cda4ea23e974d23baaec87e3a2ed95a3dc3dc235b9524"
OBJECTIVE_SHA = {
    0.5: "4c39a67b59432b54300270cfaf67f394ace5d985a63117f9e0b02f7c61af6368",
    1.0: "70de5137aeea63ab7cf3fb5e1d2e849bea3a89519e565feb936808cf9b2ed17d",
}
COUNT = 32768


def objective_path(pitch):
    return EVIDENCE / f"objective-{pitch:g}mm.npz"


class RefinedObjectiveFields(Fields):
    """Replace the single objective spline for gun and column, keeping its physical sign."""

    def __init__(self, pitch):
        super().__init__()
        path = objective_path(pitch)
        check_file(path, OBJECTIVE_SHA[pitch])
        self.objective_digest = sha(path)
        receipt = json.loads((EV / "joint-objective-material-field.json").read_text())
        with np.load(path) as field:
            if (str(field["source_sha"]) != sha(OBJ)
                    or str(field["geometry"]) != receipt["installed_geometry_digest"]
                    or str(field["curve"]) != receipt["curve_sha256"]
                    or float(field["objective_At"]) != FINAL["objective_At"]):
                raise ValueError("Objective excitation, installed circuit or parent solution differs")
            z, coeff = field["z"], field["coeff"]
            if (coeff.shape != (len(z), 4) or not np.isfinite(coeff).all()
                    or z[0] > -.002 or z[-1] < 1.601 or not np.all(np.diff(z) > 0)):
                raise ValueError("Objective bore solution does not cover the source and column")
            self.obj = make_interp_spline(z, coeff, k=3)

    def gun_key(self, currents, count):
        key = super().gun_key(currents, count)
        return sha256(f"{key}|global-objective-aphi-v1|{self.objective_digest}".encode()).hexdigest()


def run(pitch, step, reuse_emission):
    # The existing material-pair tracker uses a per-process study directory.
    # Set it before importing that tracker so older gun and column files remain addressable.
    directory = WORK / f"mesh-{pitch:g}mm"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TEMSIM_CONDENSER_PAIR_OUT"] = str(directory)
    from condenser_material_binding import check_materials
    from condenser_pair_study import OUT, Pair
    from condenser_pair_gun import emit, selected_field
    from condenser_pair_column import run as track_column
    from scan_waist_audit import archived_scan

    if OUT.resolve() != directory.resolve():
        raise ValueError("Start a fresh process for each objective mesh")
    check_file(PARENT / "working-point.json", PARENT_SHA)
    parent = json.loads((PARENT / "working-point.json").read_text())
    point = parent["operating_point"]
    check_file(PARENT / "tuned-material-field.npz", point["material_sha256"])
    shutil.copyfile(PARENT / "tuned-material-field.npz", OUT / point["material_file"])
    write(OUT / "point.json", point)
    f = RefinedObjectiveFields(pitch)
    pair = Pair(f)
    q, _ = selected_field(pair)
    np.testing.assert_array_equal(q, parent["signed_current_A"])
    np.testing.assert_array_equal(q[:8], [-30.]*4 + [16.]*4)
    np.testing.assert_array_equal(q[OWNER == "objective_lens"], 0.)
    if parent["C2_aperture_diameter_um"] != 100. or parent["objective_A"] != FINAL["objective_A"]:
        raise ValueError("Reference currents or physical aperture do not match")
    check_materials(f.state)
    check_file(FEM, parent["downstream_tracking_field_sha256"])
    with np.load(FEM) as table:
        assert json.loads(str(table["report"]))["currents_A"] == parent["corrector_A"]
    scan = archived_scan()
    if not reuse_emission:
        emit(pair, COUNT)
    with np.load(OUT / f"tip-{COUNT}.npz") as new, np.load(PARENT / "fresh-emission-and-exit.npz") as old:
        for k in ("launch", "ids", "weight", "launch_energy_eV", "emitted_A"):
            np.testing.assert_array_equal(new[k], old[k])
        # The stored energy offsets are measured at the gun exit after propagation.
        exit_energy_change = float(np.max(np.abs(new["energy_offset_ev"] - old["energy_offset_ev"])))
        if str(new["signature"]) == str(old["signature"]):
            raise ValueError("A coarse-objective gun exit was supplied")
    sample = np.r_[np.arange(1420., 1599.2, .125), 1474.7, 1474.7028632]
    write(OUT / "objective-binding.json", dict(
        objective_sha256=f.objective_digest, objective_A=FINAL["objective_A"],
        objective_At=FINAL["objective_At"], corrector_sha256=sha(FEM),
        parent_sha256=PARENT_SHA, scan=scan, mesh_mm=pitch, source_z_mm=0.,
        specimen_z_mm=f.state.sample.upper_surface_z_mm,
        max_gun_exit_energy_change_from_parent_eV=exit_energy_change,
    ))
    track_column(pair, COUNT, step, 7, extra_observation_mm=sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-mm", type=float, default=.5, choices=(.5, 1.))
    parser.add_argument("--step-mm", type=float, default=.05, choices=(.05, .025))
    parser.add_argument("--reuse-emission", action="store_true", help="Verify and reuse this mesh's fresh gun exit")
    args = parser.parse_args()
    run(args.mesh_mm, args.step_mm, args.reuse_emission)
