"""Bounded, read-only mixed-image artifacts; never reconstructed ray histories."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace, MappingProxyType
from zipfile import ZipFile

import numpy as np

from temsim.immutable_json import freeze_json
from temsim.instrument_snapshot import InstrumentSnapshot, decode_instrument, encode_instrument
from temsim.physics.wave_input_compatibility import compare_wave_input_graphs

_FILES = ("scan-and-diffraction.npz", "entrance-source-arrays.npz", "original-gun.npz",
          "inputs.json", "report.json", "source-checkpoint.json")


@dataclass(frozen=True)
class IntensityReference:
    """Saved scalar current denominator; it contains no complex wave modes."""
    total_weight: float


@dataclass(frozen=True)
class SavedSourceSummary:
    reference_current_a: float
    record: object
    digest: str
    beam: IntensityReference
    auxiliary_arrays: object


def _hash(path):
    result = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def _json(path, limit=16*1024**2):
    if path.stat().st_size > limit:
        raise ValueError("Mixed scan metadata exceeds its read budget")
    return json.loads(path.read_text(encoding="utf-8"))


def _arrays(path, maximum_bytes):
    total = 0
    with ZipFile(path) as archive:
        seen = set()
        for entry in archive.infolist():
            if not re.fullmatch(r"[a-z][a-z0-9_]*\.npy", entry.filename) or entry.filename in seen:
                raise ValueError("Mixed scan arrays require unique plain numeric members")
            seen.add(entry.filename)
            total += entry.file_size
            if total > maximum_bytes:
                raise ValueError("Mixed scan arrays exceed the decompressed read budget")
            with archive.open(entry) as handle:
                version = np.lib.format.read_magic(handle)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError("Unsupported mixed scan array encoding")
                shape, _, dtype = reader(handle)
                required = math.prod(shape)*dtype.itemsize
                if dtype.hasobject or dtype.fields or required > entry.file_size or required > maximum_bytes:
                    raise ValueError("Mixed scan array shape or dtype violates the read budget")
    with np.load(path, allow_pickle=False) as archive:
        result = {}
        for name in archive.files:
            values = np.asarray(archive[name])
            result[name] = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
    return result


def load_saved_acquisition(directory, *, maximum_bytes=512*1024**2):
    """Read verified intensities without CUDA, propagation or source conversion."""
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("Mixed scan read budget must be a positive integer")
    folder = Path(directory).resolve(strict=True)
    hashes = _json(folder/"checksums.json", 1024**2)
    if set(hashes) != set(_FILES):
        raise ValueError("Incomplete or unsupported mixed scan checksum inventory")
    if sum((folder/name).stat().st_size for name in _FILES) > maximum_bytes:
        raise ValueError("Mixed scan artifact exceeds its read budget")
    for name in _FILES:
        path = (folder/name).resolve(strict=True)
        if path.parent != folder or not re.fullmatch(r"[a-f0-9]{64}", str(hashes[name])) or _hash(path) != hashes[name]:
            raise ValueError("Mixed scan artifact checksum mismatch: "+name)
    report = _json(folder/"report.json")
    if report.get("schema") != "mixed-particle-stem-scan-v1":
        raise ValueError("Unsupported saved mixed scan")
    snapshot = InstrumentSnapshot.from_dict(_json(folder/"inputs.json"))
    if snapshot.digest != report.get("input_digest"):
        raise ValueError("Saved mixed scan and captured wave inputs differ")
    source_record = _json(folder/"source-checkpoint.json")
    reference = float(report["reference_current_a"])
    retained = float(source_record["retained_source_fraction"])
    if not math.isfinite(reference) or reference <= 0 or not 0 < retained <= 1:
        raise ValueError("Saved mixed scan has an invalid original current reference")
    unpacked = 0
    for name in ("scan-and-diffraction.npz", "original-gun.npz", "entrance-source-arrays.npz"):
        with ZipFile(folder/name) as archive:
            unpacked += sum(entry.file_size for entry in archive.infolist())
    if unpacked > maximum_bytes:
        raise ValueError("Mixed scan arrays exceed the combined decompressed read budget")
    values = _arrays(folder/"scan-and-diffraction.npz", maximum_bytes)
    original = _arrays(folder/"original-gun.npz", maximum_bytes)
    source_arrays = _arrays(folder/"entrance-source-arrays.npz", maximum_bytes)
    ids, weights = original["ray_id"], original["weight"]
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids)
            or weights.shape != ids.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0)
            or np.count_nonzero(weights > 0) != report["original_positive_count"]
            or report["original_positive_count"] < 1000):
        raise ValueError("Saved mixed scan has an invalid original emission ledger")
    order = np.argsort(ids)
    rows = np.searchsorted(ids[order], source_arrays["ray_id"])
    if np.any(rows >= len(ids)):
        raise ValueError("Saved mixed source identities are absent from the emission ledger")
    if (not np.array_equal(ids[order][rows], source_arrays["ray_id"])
            or not np.array_equal(weights[order][rows], source_arrays["weight"])):
        raise ValueError("Saved mixed source ancestry or absolute weights differ")
    fractions = {key[9:]: value for key, value in values.items() if key.startswith("fraction_")}
    currents = {key[10:]: value for key, value in values.items() if key.startswith("current_a_")}
    x, y = values["scan_x_nm"], values["scan_y_nm"]
    if x.ndim != 2 or x.shape != y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Saved mixed scan requires a complete two-dimensional raster")
    for key, data in fractions.items():
        if (data.shape != x.shape or not np.isfinite(data).all() or np.any(data < 0)
                or key not in currents or not np.allclose(currents[key], data*reference, rtol=1e-10, atol=0)):
            raise ValueError("Saved mixed scan current and original-emission fraction disagree")
    if not fractions or np.max(abs(sum(fractions.values())-1)) > 1e-9:
        raise ValueError("Saved mixed scan emission and loss ledger does not close")
    checkpoint = SavedSourceSummary(reference, freeze_json(source_record), report["checkpoint_digest"],
                                    IntensityReference(retained), MappingProxyType(source_arrays))
    scan = SimpleNamespace(scan_x_nm=x, scan_y_nm=y, fractions=fractions, current_a=currents,
        diffraction_probability=values.get("diffraction_probability"), qx_inv_angstrom=values["qx_inv_angstrom"],
        qy_inv_angstrom=values["qy_inv_angstrom"], source_checkpoint=checkpoint, record=freeze_json(report))
    # Saved intensities are historical data, not a request to restart the old
    # solver. Decode only the allow-listed parameter graph and keep the normal
    # archived dependency checks without requiring the producing code version.
    # The immutable original snapshot, including its implementation and digest,
    # remains attached separately from this private display state.
    state = decode_instrument(snapshot.graph)
    snapshot._verify_external_inputs(state)
    sample = getattr(state, "sample", None)
    if sample is not None and hasattr(sample, "ensure_upper_surface_reference"):
        sample.ensure_upper_surface_reference()
    compatibility = compare_wave_input_graphs(snapshot.graph, encode_instrument(state))
    return SimpleNamespace(state_snapshot=state, input_snapshot=snapshot,
        input_compatibility=freeze_json(compatibility), particle_result=None, particle_boundary=None,
        scan=scan, saved_directory=str(folder), read_only=True)
