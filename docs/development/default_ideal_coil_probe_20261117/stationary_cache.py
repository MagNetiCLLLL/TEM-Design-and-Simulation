"""Preserve completed column trajectories during authenticated nested source growth."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def array_digest(*arrays):
    digest = sha256()
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(str((contiguous.dtype.str, contiguous.shape)).encode())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def column_identity(z, saved, field, electric_digest, geometry):
    solver = Path(__file__).with_name("stationary_pupil.py").read_text(encoding="utf-8")
    return dict(
        mesh_field=array_digest(z, saved, field),
        electric=electric_digest,
        solver=sha256(solver.encode()).hexdigest(),
        boundaries=array_digest(
            geometry.origins_mm,
            geometry.radii_mm,
            geometry.axial_lower_m,
            geometry.axial_upper_m,
            geometry.native,
        ),
    )


def load_prefix(directory, gun, identity, ids, phase, energies, weights):
    record = json.loads((directory / "column.json").read_text())
    n = record["surviving"]
    if (
        gun.get("parent_gun_sha256") != record["gun_sha256"]
        or record["identity"] != identity
        or gun["reused_originals"] != record["originals"]
        or n >= len(ids)
    ):
        raise ValueError("Column prefix and nested gun or numerical transport differ")
    if (
        array_digest(ids[:n], phase[:n], energies[:n]) != record["source_phase"]
        or array_digest(weights[:n] / gun["prefix_weight_scale"]) != record["weights"]
    ):
        raise ValueError("Column prefix source states or weights differ")
    # Verify the immutable stored phase, clock and complete swept envelope.
    data = np.load(directory / "column-phase.npy", mmap_mode="r")
    radius = np.load(directory / "column-radius.npy")
    if array_digest(data, radius) != record["transport"]:
        raise ValueError("Column trajectory archive checksum mismatch")
    return data, radius


def save_column(directory, gun, identity, ids, phase, energies, weights, data, radius):
    record = dict(
        identity=identity,
        gun_sha256=gun["sha256"],
        originals=gun["population"]["positive_current_count"],
        surviving=len(ids),
        source_phase=array_digest(ids, phase, energies),
        weights=array_digest(weights),
        transport=array_digest(data, radius),
    )
    for name, array in (("phase", data), ("radius", radius)):
        tmp = directory / f"column-{name}.part.npy"
        np.save(tmp, array)
        tmp.replace(directory / f"column-{name}.npy")
    tmp = directory / "column.part.json"
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(directory / "column.json")
