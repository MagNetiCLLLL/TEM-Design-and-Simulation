"""Read native crossover statistics from a completed physical observation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(path for path in Path(__file__).resolve().parents if (path/"agent.md").is_file())
sys.path.insert(0, str(ROOT/"src"))
from temsim.physics.beam_statistics import transverse_beam_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.trial/"report.json").read_text())
    records = []
    with np.load(args.trial/"observations.npz", allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in ("z_mm", "weights", "alive", "x_m", "y_m", "tx_rad", "ty_rad", "source_id")}
    surface_row, = np.flatnonzero(data["z_mm"] == report["surface_z_mm"])
    final_cohort = data["alive"][surface_row] & (data["weights"] > 0)
    def observation(label, population, plane):
        row, = np.flatnonzero(data["z_mm"] == plane)
        mask = data["alive"][row] & (data["weights"] > 0)
        if population == "sample_survivors":
            mask &= final_cohort
        stats = transverse_beam_statistics(*(data[key][row] for key in ("x_m", "y_m", "tx_rad", "ty_rad")),
                                            alive=mask, weights=data["weights"])
        weight = float(data["weights"][mask].sum())
        records.append(dict(interval=label, population=population, executed_z_mm=float(plane),
            original_source_ids=data["source_id"][mask].tolist(), original_source_weight=weight,
            surviving_positive_count=int(mask.sum()), effective_count=float(weight**2/np.dot(data["weights"][mask], data["weights"][mask])),
            current_a=weight*report["execution"]["reference_current_a"],
            d95_nm=stats.radius_95_m*2e9, rms_radius_nm=stats.radius_rms_m*1e9,
            alpha95_mrad=stats.convergence_95_rad*1e3, angular_rms_mrad=stats.convergence_rms_rad*1e3))
    for interval, populations in report["topology"].items():
        for population, topology in populations.items():
            for region in topology.get("regions", ()):
                observation(interval, population, region["representative_z_mm"])
    observation("specimen upper surface", "actual", report["surface_z_mm"])
    surface = records[-1]
    for key in ("d95_nm", "rms_radius_nm", "alpha95_mrad", "angular_rms_mrad"):
        if not np.isclose(surface[key], report[key], rtol=2e-15, atol=0.):
            raise ValueError(f"Native surface {key} does not reproduce the physical report")
    result = dict(schema="two-condenser-native-crossover-statistics-v1",
        scope="Postprocessing only; no field construction, source emission or transport",
        original_report_sha256=sha256((args.trial/"report.json").read_bytes()).hexdigest(),
        original_observations_sha256=report["observations_sha256"],
        source_count=report["execution"]["emitted_positive_count"],
        native_surface_statistics_reproduced=True, observations=records,
        interpretation="Representative executed 2D source-image regions use RMS covariance topology; a representative RMS waist need not coincide with the native D95 minimum. Current fractions retain the original emission denominator. Read unresolved population-transition windows separately.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps([{k:v for k,v in row.items() if k != "original_source_ids"} for row in records], indent=2))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
