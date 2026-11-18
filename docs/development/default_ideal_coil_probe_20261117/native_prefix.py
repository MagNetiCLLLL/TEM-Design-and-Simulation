"""Run the original column field evaluator on a weight-preserving matched cohort."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from blocks import capture_blocks
from temsim.instrument_snapshot import InstrumentSnapshot
from trial import observed_minimum, plane_metrics, waist_separation, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--originals", type=int, default=1024)
    args = parser.parse_args()
    if args.originals < 1000:
        parser.error("At least 1000 original positive-current particles are required")
    args.output.mkdir(parents=True, exist_ok=True)
    snap = InstrumentSnapshot.from_dict(json.loads((args.trial / "inputs.json").read_text()))
    record = json.loads((args.trial / "gun.json").read_text())
    path = args.trial / "gun.npz"
    if record["input_digest"] != snap.digest or sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Gun and physical inputs differ")
    state = snap.restore()
    with np.load(path) as g:
        n = args.originals
        if not np.array_equal(g["emission_ray_id"], np.arange(len(g["weight"]))) or not np.array_equal(
            g["emission_ray_id"], g["ray_id"]
        ):
            raise ValueError("Original source order differs")
        if np.count_nonzero(g["weight"][:n] > 0) != n:
            raise ValueError("Insufficient original positive-current particles")
        subset = SimpleNamespace(
            **{
                k: g[k][:n]
                for k in (
                    "ray_id",
                    "weight",
                    "x_m",
                    "y_m",
                    "tx_rad",
                    "ty_rad",
                    "alive",
                    "energy_offset_ev",
                    "flight_time_s",
                )
            }
        )
        identity = (subset.ray_id, g["source_azimuth_rad"][:n])
        gun = SimpleNamespace(
            emitted_current_a=float(g["emitted_current_a"]),
            blocked_z_mm=g["blocked_z_mm"][:n],
            blocked_key=g["blocked_key"][:n],
        )
    complete, _ = capture_blocks(
        state,
        gun,
        subset,
        identity,
        args.output,
        snap,
        state.sample.upper_surface_z_mm,
        plane_metrics,
        observed_minimum,
        waist_separation,
    )
    complete.update(
        complete_gun_originals=record["population"]["positive_current_count"],
        traced_originals=n,
        input_digest=snap.digest,
        gun_sha256=record["sha256"],
        scope="Native full-column Lorentz integration of original IDs with their full-population absolute weights",
    )
    write(args.output / "metrics.json", complete)


if __name__ == "__main__":
    main()
