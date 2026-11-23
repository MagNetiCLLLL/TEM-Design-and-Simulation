"""Archive the two measured relay trials and check the controls carried between them."""

import argparse
import json
import shutil
import sys

import numpy as np

from joint_pivot_tuning import OUT
from normal_working_point import AUDIT, CACHE, EVIDENCE, file_hash, verified_table
from refocus_a2 import WORK as OBJECTIVE_WORK


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def restore():
    manifest = read(EVIDENCE / "joint-three-plane-result.json")
    final = read(EVIDENCE / "joint-objective-focus.json")
    baseline = read(EVIDENCE / "normal-current-working-point.json")
    amps, ni = final["TL12_A"], final["objective_At"]
    objective = OBJECTIVE_WORK / "column" / f"armco-{abs(ni):.12g}-gap-fine-p8"
    copies = {
        EVIDENCE / "joint-tl12-material-solution.npz": OUT / f"tl12-{amps:.10g}A-solution.npz",
        EVIDENCE / "joint-objective-material-field.npz": objective.parent / (objective.name + ".npz"),
        EVIDENCE / "joint-objective-material-field.json": objective.parent / (objective.name + ".json"),
        EVIDENCE / "joint-relay-scan.json": OUT / "aligned-scan.json",
        EVIDENCE / "joint-final-scan.json": OUT / "focused-scan.json",
    }
    hashes = manifest["evidence_sha256"].copy()
    base_solution = (EVIDENCE / baseline["solution"]).resolve()
    copies[base_solution] = CACHE / "5-170-both-edge.npz"
    hashes[base_solution.name] = baseline["solution_sha256"]
    for source, destination in copies.items():
        expected = hashes[source.name]
        if file_hash(source) != expected:
            raise ValueError(f"Archive digest mismatch for {source.name}")
        if destination.exists():
            if file_hash(destination) != expected:
                raise ValueError(f"Conflicting study input at {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    verified_table(
        CACHE / "normal-joint-field.npz",
        baseline["field_table_sha256"],
        base_solution,
        baseline["solution_sha256"],
    )
    sys.path.insert(0, str(AUDIT))
    from tl12_excitation import main as sample

    sample([amps], 24)
    if file_hash(OUT / f"tl12-{amps:.10g}A-field-24.npz") != manifest["tracking_field_sha256"]:
        raise ValueError("Restored tracking field differs from the measured trial")


def archive():
    first = read(EVIDENCE / "joint-relay-alignment.json")
    last = read(EVIDENCE / "joint-objective-focus.json")
    upstream = read(EVIDENCE / "joint-c1-topology-control.json")
    before, after = (read(OUT / name) for name in ("aligned-scan.json", "focused-scan.json"))
    for record, scan in ((first, before), (last, after)):
        if (
            record["source_digest"] != scan["source_digest"]
            or record["objective_At"] != scan["objective_At"]
            or record["TL12_A"] != scan["tl12_A"]
            or record["emitted"] != 32768
            or record["pupil_um"] != 100.0
        ):
            raise ValueError("Scan and static traces must refer to the same physical working point")
    for key in (
        "source_digest",
        "field_sha256",
        "TL12_A",
        "pupil_um",
        "emitted_A",
        "fixed_vacuum_currents_A",
    ):
        if first[key] != last[key]:
            raise ValueError(f"During objective-only focusing, {key} changed")
    np.testing.assert_array_equal(before["scan"]["mix"], after["scan"]["mix"])
    if upstream["source_digest"] != last["source_digest"]:
        raise ValueError("Inconsistent upstream control source")
    amps, ni = last["TL12_A"], last["objective_At"]
    named = {
        "joint-baseline-scan.json": OUT / "baseline.json",
        "joint-relay-scan.json": OUT / "aligned-scan.json",
        "joint-final-scan.json": OUT / "focused-scan.json",
        "joint-tl12-alignment-history.json": OUT / "aligned-current.json",
        "joint-objective-search.json": OUT / "native-objective-search.json",
        "joint-tl12-material-solution.npz": OUT / f"tl12-{amps:.10g}A-solution.npz",
        "joint-objective-material-field.npz": OBJECTIVE_WORK
        / "column"
        / f"armco-{abs(ni):.12g}-gap-fine-p8.npz",
        "joint-objective-material-field.json": OBJECTIVE_WORK
        / "column"
        / f"armco-{abs(ni):.12g}-gap-fine-p8.json",
    }
    for name, source in named.items():
        target = EVIDENCE / name
        if not target.exists() or file_hash(target) != file_hash(source):
            shutil.copyfile(source, target)
    row = after["scan"]
    (front,) = last["focal_mm"]["mini_and_objective"]
    first_waist, final_waist = last["waists_mm"]["downstream_of_lower_scan"]["radial"]
    closest = min(row["candidates"], key=lambda r: abs(r["z_mm"] - front))
    status = dict(
        mode="300 kV three-condenser Nanoprobe Normal, diagnostic trial",
        accepted_as_operating_point=False,
        observations=dict(
            C1_radial_waists_450_550_mm=len(upstream["installed"]["radial_mm"]),
            C1_and_C2_off_waists=len(upstream["C1_and_C2_off"]["radial_mm"]),
            static_AC_interval_waists=last["waists_mm"]["TL12_to_scan_exit"]["radial"],
            tl12_waist_minus_front_mm=first_waist - front,
            minimum_scan_displacement_plane_minus_front_mm=closest["z_mm"] - front,
            minimum_scan_displacement_singular_nm_per_urad=closest["position_singular_nm_per_urad"],
            final_rms_waist_minus_specimen_mm=final_waist - last["specimen_top_mm"],
        ),
        source_sha256=last["source_digest"],
        tracking_field_sha256=last["field_sha256"],
        evidence_sha256={
            p.name: file_hash(p)
            for p in sorted(EVIDENCE.glob("joint-*"))
            if p.name != "joint-three-plane-result.json"
        },
    )
    (EVIDENCE / "joint-three-plane-result.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore", action="store_true", help="Restore validated material inputs for a replay"
    )
    restore() if parser.parse_args().restore else archive()
