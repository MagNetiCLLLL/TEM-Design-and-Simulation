"""Verify archived selected phase bytes and re-observe retained native minima.

This only reads completed observations. It emits no source, constructs no
field runtime and performs no particle or coherent propagation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(path for path in Path(__file__).resolve().parents if (path/"agent.md").is_file())
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from measure_probe_waist import PHASE, analyze_scan


def verify_trial(original, compact):
    report = json.loads((original/"report.json").read_text(encoding="utf-8"))
    request = json.loads((original/"request.json").read_text(encoding="utf-8"))
    checked = []
    with np.load(original/"observations.npz", allow_pickle=False) as source, np.load(compact/"native-phase.npz", allow_pickle=False) as selected:
        indexes = selected["original_checkpoint_index"]
        if indexes.dtype.kind not in "iu" or np.any(np.diff(indexes) <= 0):
            raise ValueError("Native rows must retain their ordered exact original indexes")
        for key in source.files:
            actual = np.asarray(selected[key])
            expected = np.asarray(source[key])
            if key in (*PHASE, "z_mm", "alive"):
                expected = expected[indexes]
            if (actual.dtype != expected.dtype or actual.shape != expected.shape
                    or actual.tobytes() != expected.tobytes()):
                raise ValueError(f"Archived original array bytes differ: {key}")
            checked.append(key)
        arrays = {key: np.asarray(selected[key]) for key in selected.files}
        retained_focus = np.asarray([z for z in request["focus_planes_mm"] if z in arrays["z_mm"]])
        observed = analyze_scan(arrays, retained_focus, surface_z_mm=report["surface_z_mm"],
                                source_current_a=report["execution"]["reference_current_a"])
    expected = report["native_focus"]
    for key in ("d95_waist", "rms_waist_diagnostic", "surface", "surface_gate", "waist_separation"):
        # The production report omits repeated surface source-ID lists; the
        # complete IDs are independently byte-checked in the phase above.
        for field, value in expected[key].items():
            if observed[key].get(field) != value:
                raise ValueError(f"Selected native observations do not reproduce the reported {key}.{field}")
    return {"original_directory": str(original), "compact_directory": str(compact),
            "original_array_count": len(checked), "byte_verified_arrays": checked,
            "selected_native_rows": len(indexes), "selected_focus_rows": len(retained_focus),
            "native_d95_rms_surface_gate_reproduced": True,
            "scope": "Only already executed data were read; this verifies archive selection, not a new physical qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", action="append", required=True, help="LABEL=original complete trial directory")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for specification in args.trial:
        label, raw = specification.split("=", 1)
        if label in records:
            raise ValueError("Use distinct trial labels")
        records[label] = verify_trial(Path(raw).resolve(), args.archive.resolve()/"trials"/label)
    args.output.write_text(json.dumps({"schema": "two-condenser-selected-phase-verification-v1",
        "trials": records, "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()},
        indent=2, ensure_ascii=True, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"verified_trials": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
