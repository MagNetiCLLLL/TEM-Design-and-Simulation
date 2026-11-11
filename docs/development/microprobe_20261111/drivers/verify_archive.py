"""Verify retained phase identity and statistics without new propagation."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from temsim.physics.beam_statistics import transverse_beam_statistics

folder = ROOT / "docs/development/microprobe_20261111"
data = json.loads((folder / "results.json").read_text(encoding="utf-8"))
checks = []
with np.load(folder / "extractor/new-n2048/gun_exit_and_emission.npz", allow_pickle=False) as gun:
    original_ids, original_weights = gun["source_id"], gun["weights"]
    if len(np.unique(original_ids)) != 2048 or np.count_nonzero(original_weights > 0.) != 2048:
        raise ValueError("The archival source is not the full original 2048-particle population")
    for case, index in zip(data["family"], (2, 0, 1)):
        path = folder / "column/n2048-step010" / f"trial-{index:02d}/sample-neighborhood.npz"
        with np.load(path, allow_pickle=False) as phase:
            if (not np.array_equal(phase["source_id"], original_ids)
                    or not np.array_equal(phase["weights"], original_weights)):
                raise ValueError("An archived observation lost or reweighted the original source")
            row = np.searchsorted(phase["z_mm"], case["surface"]["z_mm"])
            if phase["z_mm"][row] != case["surface"]["z_mm"]:
                raise ValueError("The surface is not an actual saved observation")
            alive = phase["alive"][row] & (original_weights > 0.)
            for name in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
                if not np.isfinite(phase[name][row, alive]).all():
                    raise ValueError("Nonfinite retained positive-current phase")
            stats = transverse_beam_statistics(*(phase[key][row] for key in
                ("x_m", "y_m", "tx_rad", "ty_rad")), alive=alive, weights=original_weights)
            recomputed = dict(d95_um=2e6*stats.radius_95_m,
                radial_rms_um=1e6*stats.radius_rms_m, alpha95_mrad=stats.convergence_95_mrad,
                alpha_rms_mrad=1e3*stats.convergence_rms_rad,
                current_a=data["source_current_a"]*original_weights[alive].sum())
            for key, value in recomputed.items():
                if not np.isclose(value, case["surface"][key], rtol=1e-12, atol=1e-15):
                    raise ValueError(f"Archived statistic does not reproduce: {key}")
            if int(alive.sum()) != case["surface"]["count"]:
                raise ValueError("Sample arrival count mismatch")
            checks.append(dict(profile=case["profile"], status="PASS", source_positive_count=2048,
                source_ids_and_original_weights="BITWISE_IDENTICAL", arrivals=int(alive.sum()),
                recomputed=recomputed))
report = dict(scope="Pure-array archival identity and statistic checks; no physical retracing", checks=checks)
(folder / "archive-verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
for path in sorted(folder.rglob("*")):
    if path.is_file() and path.name != "artifact_manifest.json" and "__pycache__" not in path.parts:
        key = str(path.relative_to(folder)).replace("\\", "/")
        fingerprint = dict(sha256=sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
        if key in manifest["files"]:
            previous = manifest["files"][key]
            if any(previous.get(item) != fingerprint[item] for item in fingerprint):
                raise ValueError(f"Previously archived evidence changed: {key}")
        else:
            manifest["files"][key] = dict(generated_from="Completed Microprobe study and offline checks", **fingerprint)
        if key.startswith("drivers/") and "original_path" not in manifest["files"][key]:
            manifest["files"][key]["original_path"] = "outputs/microprobe-20261111/" + path.name
(folder / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
print(json.dumps(dict(status="PASS", cases=len(checks), files=len(manifest["files"])), indent=2))
