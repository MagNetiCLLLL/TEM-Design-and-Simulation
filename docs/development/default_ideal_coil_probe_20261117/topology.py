"""Observe source-image topology using the saved native planes and their absolute currents."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.optics import direct_focus_topology
from temsim.optics.crossover_acceptance import regions_for_state
from temsim.optics.waist_evidence import corrector_evidence_for_state, finite_interval


PHASE = ("x_m", "y_m", "tx_rad", "ty_rad")


def load_checkpoints(directory, digest):
    paths = sorted((directory / "phase").glob("[0-9][0-9][0-9].npz"))
    records = sorted((directory / "blocks").glob("[0-9][0-9][0-9].json"))
    if not paths or [p.stem for p in paths] != [p.stem for p in records]:
        raise ValueError("Missing recorded phase block")
    with np.load(directory / "gun.npz") as gun:
        ids = gun["ray_id"]
        weights = gun["weight"]
    pieces = {key: [] for key in ("z_mm", "active", *PHASE)}
    previous_sha = None
    for index, (path, record) in enumerate(zip(paths, records)):
        meta = json.loads(record.read_text())
        terminal = record.with_suffix(".npz")
        if sha256(terminal.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Changed terminal checkpoint in {record.name}")
        if index and meta["parent"] != previous_sha:
            raise ValueError(f"Broken terminal checkpoint chain in {record.name}")
        previous_sha = meta["sha256"]
        with np.load(path) as phase, np.load(terminal) as end:
            if str(phase["input_digest"]) != digest or not np.array_equal(ids, phase["source_id"]):
                raise ValueError(f"Invalid phase source identity in {path.name}")
            if not np.array_equal(weights, phase["weights"]):
                raise ValueError("Original current weights changed")
            z = phase["z_mm"]
            if not np.array_equal(z, [r["z_mm"] for r in meta["profile"]]):
                raise ValueError("Executed observation planes differ from recorded statistics")
            if index and z[0] != pieces["z_mm"][-1][-1]:
                raise ValueError("Noncontiguous axial observations")
            for key in pieces:
                values = phase[key]
                if key != "z_mm":
                    if not np.array_equal(
                        values[-1], end["alive" if key == "active" else key], equal_nan=True
                    ):
                        raise ValueError(f"Phase / terminal mismatch for {key}")
                    if index and not np.array_equal(values[0], pieces[key][-1][-1], equal_nan=True):
                        raise ValueError(f"Discontinuous phase in {path.name} for {key}")
                pieces[key].append(values[int(index > 0) :])
    cp = SimpleNamespace(**{key: np.concatenate(values) for key, values in pieces.items()})
    if np.any(np.diff(cp.z_mm) <= 0):
        raise ValueError("Axial observations must be strictly ordered")
    return cp, weights, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    snapshot = InstrumentSnapshot.from_dict(json.loads((args.trial / "inputs.json").read_text()))
    metrics = json.loads((args.trial / "metrics.json").read_text())
    if snapshot.digest != metrics["input_digest"]:
        raise ValueError("Finite-source measurements belong to a different input snapshot")
    state = snapshot.restore()
    cp, weights, ids = load_checkpoints(args.trial, snapshot.digest)
    layout = regions_for_state(state)
    results = {}
    for key, required in (("c2_to_c3", 1), ("c3_to_adl", 0)):
        region = finite_interval(
            cp, cp.active, weights, *layout[key]["bounds_mm"], maximum_spacing_mm=0.5, source_ids=ids
        )
        results[key] = dict(
            required=required,
            observed=len(region["regions"]),
            evidence=region,
            passed=region["status"] == "RESOLVED_AT_BOUNDED_SAMPLING"
            and len(region["regions"]) == required,
        )
    corrector = corrector_evidence_for_state(state, cp, cp.active, weights, ids)
    start, surface = direct_focus_topology.bounds_for_state(state)
    direct = direct_focus_topology.evaluate(
        cp,
        cp.active,
        weights,
        lower_z_mm=start,
        surface_z_mm=surface,
        maximum_spacing_mm=direct_focus_topology.MAXIMUM_SPACING_MM,
        surface_focus_tolerance_nm=direct_focus_topology.SURFACE_FOCUS_TOLERANCE_NM,
        waist_detector="finite_region",
        source_ids=ids,
    )
    report = dict(
        input_digest=snapshot.digest,
        original_positive=int((weights > 0).sum()),
        observation_count=len(cp.z_mm),
        condenser_intervals=results,
        corrector=corrector,
        direct_focus=direct,
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Condenser", [(key, row["observed"], row["passed"]) for key, row in results.items()])
    print("Corrector", corrector["status"], "Direct focus", direct["status"])


if __name__ == "__main__":
    main()
