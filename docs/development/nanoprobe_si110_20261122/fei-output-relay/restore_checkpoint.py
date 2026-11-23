"""Restore the saved operating point's solved material volume and vacuum tracking tables."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE = HERE / "field-replay"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["archives"].items():
        if digest(ARCHIVE / name) != checksum:
            raise ValueError(f"Archive checksum mismatch for {name}")
    volume = ROOT / manifest["material_field"]["destination"]
    volume.parent.mkdir(parents=True, exist_ok=True)
    expected = manifest["material_field"]["sha256"]
    if not volume.exists() or digest(volume) != expected:
        temp = volume.with_name(volume.stem + "-restore.npz")
        with np.load(ARCHIVE / "material-mesh.npz") as mesh, np.load(ARCHIVE / "material-state.npz") as state:
            np.savez_compressed(temp, corners_m=mesh["corners_m"], **{k: state[k] for k in state.files})
        if digest(temp) != expected:
            raise ValueError("Restored material volume differs from the saved solution")
        temp.replace(volume)
    for name, location in manifest["tracking_tables"].items():
        target = ROOT / location
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != manifest["archives"][name]:
            temporary = target.with_suffix(".restore-pending")
            shutil.copyfile(ARCHIVE / name, temporary)
            temporary.replace(target)
    return volume


if __name__ == "__main__":
    print(restore())
