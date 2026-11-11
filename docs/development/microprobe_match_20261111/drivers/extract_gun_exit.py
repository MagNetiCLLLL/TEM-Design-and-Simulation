"""Retain the executed gun-exit row omitted by the initial selected-plane list."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEST=ROOT/"docs/development/microprobe_match_20261111"
results=json.loads((DEST/"results.json").read_text())
for entry in results["family"]:
    rel=Path("column")/entry["run"]/f"trial-{entry['index']:02d}"
    original=HERE/rel/"observations.npz"
    receipt=json.loads((HERE/rel/"report.json").read_text())
    assert sha256(original.read_bytes()).hexdigest()==receipt["observations_sha256"]
    with np.load(original,allow_pickle=False) as data:
        index=int(np.flatnonzero(data["z_mm"]==450.)[0])
        arrays={key:data[key][index] for key in ("x_m","y_m","tx_rad","ty_rad","kinetic_energy_ev","flight_time_s","alive")}
        arrays.update(z_mm=np.array(450.),source_id=data["source_id"],weights=data["weights"])
    target=DEST/rel/"native-gun-exit.npz"
    if target.exists():raise FileExistsError("Do not overwrite an archived native row")
    np.savez_compressed(target,**arrays)
    print("Retained executed gun exit:",entry["profile"],flush=True)
