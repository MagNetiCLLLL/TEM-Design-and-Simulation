"""Compare saved complete GPU waves without phase fitting or normalisation."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = (json.loads((p / "measurement.json").read_text(encoding="utf-8"))
                     for p in (args.before, args.after))
    for key in ("checkpoint_digest", "instrument_sha256", "numerics", "input_z_mm", "output_z_mm", "physical_steps"):
        if before[key] != after[key]:
            raise ValueError(f"Benchmark inputs changed: {key}")
    if len(before["modes"]) != len(after["modes"]):
        raise ValueError("Executed mode count changed")
    comparisons = []
    for index, (old, new) in enumerate(zip(before["modes"], after["modes"])):
        for key in ("mode_id", "energy_kev", "shape", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            if old[key] != new[key]:
                raise ValueError(f"Physical/grid metadata changed in mode {index}: {key}")
        a, b = (np.load(p / f"mode-{index}-amplitude.npy", mmap_mode="r", allow_pickle=False)
                for p in (args.before, args.after))
        error = norm = weighted_error = weighted_norm = intensity_error = intensity_norm = 0.
        maximum = 0.
        for start in range(0, len(a), 64):
            x, y = a[start:start+64], b[start:start+64]
            diff = x-y
            maximum = max(maximum, float(np.max(abs(diff))))
            error += float(np.sum(abs(diff)**2))
            norm += float(np.sum(abs(x)**2))
            weighted_error += float(np.sum(abs(x*np.sqrt(old["weight"])-y*np.sqrt(new["weight"]))**2))
            weighted_norm += old["weight"]*float(np.sum(abs(x)**2))
            old_i, new_i = old["weight"]*abs(x)**2, new["weight"]*abs(y)**2
            intensity_error += float(np.sum(abs(old_i-new_i)))
            intensity_norm += float(np.sum(old_i))
        comparison = dict(mode=index, complex_relative_l2=float(np.sqrt(error/norm)),
            weighted_complex_relative_l2=float(np.sqrt(weighted_error/weighted_norm)),
            maximum_amplitude_difference=maximum,
            weighted_intensity_relative_l1=intensity_error/intensity_norm,
            probability_difference=new["probability"]-old["probability"],
            mode_weight_difference=new["weight"]-old["weight"])
        if max(comparison["complex_relative_l2"], comparison["weighted_complex_relative_l2"],
               comparison["weighted_intensity_relative_l1"], abs(comparison["mode_weight_difference"])) > 1e-12:
            raise ValueError(f"GPU before/after parity exceeded 1e-12: {comparison}")
        comparisons.append(comparison)
    result = dict(status="PASS", comparisons=comparisons,
        before_seconds=before["warmed_median_seconds"], after_seconds=after["warmed_median_seconds"],
        speedup=before["warmed_median_seconds"]/after["warmed_median_seconds"],
        input_identity=before["instrument_sha256"], checkpoint_digest=before["checkpoint_digest"],
        scope=after["timing_scope"], phase_alignment="None", normalization="None")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
