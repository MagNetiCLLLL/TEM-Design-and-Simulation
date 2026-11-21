"""Rebuild the selected circuit, transport the original electrons and replay both silicon rasters."""

import argparse
from hashlib import sha256
import json
import subprocess
import sys

import numpy as np

from solve_fields import CACHE, HERE, electrostatic_grid_digest, solve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-entrance", action="store_true", help="Replay only the specimen from the archived rays")
    args = p.parse_args()
    evidence = HERE / "evidence"
    for name, digest in json.loads((evidence / "checksums.json").read_text()).items():
        if sha256((evidence / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Acquisition checksum mismatch: {name}")
    if not args.from_entrance:
        from column import prepare
        from tune_objective import field_trial, report_surface
        prepare(CACHE, CACHE / "replay-basis.npz")
        study = json.loads((evidence / "study.json").read_text())
        if electrostatic_grid_digest(CACHE / "electric.npz") != study["electrostatic_grid_sha256"]:
            raise ValueError("Electrostatic field values or grid changed")
    for name in ("before", "after"):
        source = evidence / name
        if not args.from_entrance:
            m = json.loads((source / "metrics.json").read_text())
            ni, step, controls = m["objective_ampere_turns"], m["max_step_mm"], m["controls"]
            for key in ("columns", "mechanical"):
                if sha256((CACHE / (key + ".npz")).read_bytes()).hexdigest() != m["transport_sha256"][key]:
                    raise ValueError(f"{name}: changed {key} transport input")
            solve(abs(ni), "gap-fine", 8)
            field_path = CACHE / f"armco-{abs(ni):g}-gap-fine-p8.npz"
            with np.load(field_path) as actual, np.load(source / "objective-field.npz") as old:
                for key in ("r", "z"):
                    np.testing.assert_array_equal(actual[key], old[key])
                np.testing.assert_allclose(actual["a_phi"], old["a_phi"], rtol=1e-9, atol=1e-14)
            mesh, tracker, base = field_trial(ni, "gap-fine", step, m["hexapole_scale"])
            report_surface(mesh, tracker, base, controls, ni, "gap-fine", step, "replay-" + name, m["hexapole_scale"])
            new = CACHE / ("replay-" + name)
            with np.load(source / "surface.npz") as old, np.load(new / "surface.npz") as actual:
                for key in ("ray_id", "weight", "alive", "loss_step", "emitted_current_a", "z_mm"):
                    np.testing.assert_array_equal(actual[key], old[key])
                for key in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
                    np.testing.assert_allclose(actual[key], old[key], rtol=1e-9, atol=1e-13)
            source = new
        out = CACHE / ("replay-" + name + "-scan")
        subprocess.run([sys.executable, str(HERE.parent / "default_ideal_coil_probe_20261117" / "scan.py"),
                        str(source), str(out)], check=True)
        archived = json.loads((evidence / name / "report.json").read_text())
        rebuilt = json.loads((out / "report.json").read_text())
        for key in ("cif_sha256", "numerical_model_sha256", "original_positive", "surviving", "incident_current_a",
                    "entrance_mm", "pixel_pitch_nm", "thickness_nm", "annulus_mrad", "bf_mrad"):
            if rebuilt[key] != archived[key]:
                raise ValueError(f"{name}: changed scan input {key}")
        with np.load(out / "scan.npz") as actual, np.load(evidence / name / "scan.npz") as old:
            np.testing.assert_array_equal(actual["coords_nm"], old["coords_nm"])
            np.testing.assert_allclose(actual["channel_current_a"], old["channel_current_a"], rtol=1e-8, atol=1e-16)
        print(f"{name}: all four absolute-current images reproduced", flush=True)


if __name__ == "__main__":
    main()
