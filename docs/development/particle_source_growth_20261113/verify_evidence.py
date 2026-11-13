"""Verify archived bytes and selected native observations; never propagate rays."""
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from temsim.physics.beam_statistics import transverse_beam_statistics

BASE = Path(__file__).resolve().parent
manifest = json.loads((BASE / "artifact-manifest.json").read_text(encoding="utf-8"))
for name, row in manifest["copies"].items():
    raw = (BASE / name).read_bytes()
    assert sha256(raw).hexdigest() == row["sha256"], name
    if name.endswith(".gz") and "original_sha256" in row:
        assert sha256(gzip.decompress(raw)).hexdigest() == row["original_sha256"], name
observations = 0
for directory in ("legacy-8192", "legacy-16384"):
    report = json.loads(gzip.decompress((BASE / directory / "report.json.gz").read_bytes()))
    reference = report["native_focus"]["planes"]
    with np.load(BASE / directory / "selected-native-phase.npz", allow_pickle=False) as data:
        for index, z in enumerate(data["z_mm"]):
            expected = next(row for row in reference if row["z_mm"] == z)
            stats = transverse_beam_statistics(*(data[key][index] for key in
                ("x_m", "y_m", "tx_rad", "ty_rad")),
                alive=data["alive"][index], weights=data["weights"])
            np.testing.assert_allclose(2e9 * stats.radius_95_m, expected["d95_nm"], rtol=1e-13)
            np.testing.assert_allclose(1e9 * stats.radius_rms_m, expected["rms_radius_nm"], rtol=1e-13)
            observations += 1
print(json.dumps({"verified_artifacts": len(manifest["copies"]), "native_observations": observations,
                  "scope": "Read-only byte and original-array verification; no physical or coherent execution"}))
