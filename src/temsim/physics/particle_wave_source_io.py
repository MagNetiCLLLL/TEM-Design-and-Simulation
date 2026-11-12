"""Checked immutable classical phase-space exchange, never a Lorentz restart.

An old producer identity remains attached to the particles and captured input
graph. A later wave solver may explicitly consume this boundary; it must not
relabel the particles as freshly transported by its current implementation.
Reading classical data does not require CUDA. New wave execution still does.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from types import MappingProxyType
from zipfile import ZipFile

import numpy as np

from temsim.immutable_json import freeze_json, thaw_json
from temsim.instrument_snapshot import InstrumentSnapshot, SNAPSHOT_SCHEMA, decode_instrument
from temsim.physics.particle_wave_source import ParticleWaveSource


SCHEMA = "validated-particle-boundary-v1"
_PHASE = ("x_m", "y_m", "tx_rad", "ty_rad", "ray_id", "weight",
          "kinetic_energy_ev", "flight_time_s")
_CANONICAL = "canonical_frequency_shift_inv_angstrom"
_GUN = ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight",
        "ray_id", "alive", "flight_time_s", "blocked_z_mm", "blocked_key")
_EMISSION = ("ray_id", "position_m", "direction", "normal")
_FILES = ("inputs.json", "source.json", "source-phase.npz", "original-gun.npz", "emission-reference.npz")
_MAX_BYTES = 512*1024**2
_MAX_PARTICLES = 4_000_000


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and not set(value)-set("0123456789abcdef")


def _freeze_arrays(values):
    return MappingProxyType({key: np.frombuffer(np.asarray(value).tobytes(),
        dtype=np.asarray(value).dtype).reshape(np.shape(value)) for key, value in values.items()})


def _validate(source, inputs, original, emission, record):
    if not isinstance(source, ParticleWaveSource) or not isinstance(inputs, InstrumentSnapshot):
        raise TypeError("A particle boundary requires its immutable source and complete input snapshot")
    if inputs.graph.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("A standalone particle boundary requires a complete inline input graph")
    if source.input_digest != inputs.digest or record.get("input_digest") != inputs.digest:
        raise ValueError("Particle boundary source and complete input snapshot digests differ")
    if (record.get("schema") != SCHEMA or not _hex(inputs.implementation)
            or record.get("source_generation_implementation") != inputs.implementation
            or not _hex(record.get("source_archive_sha256"))):
        raise ValueError("Particle boundary must retain the original producer and source archive identities")
    if record.get("source_digest") != source.digest:
        raise ValueError("Particle boundary source checksum mismatch")
    for key in ("z_mm", "reference_current_a", "original_positive_count", "source_fraction", "mean_energy_ev"):
        if record.get(key) != getattr(source, key):
            raise ValueError("Particle boundary scalar metadata differs from its raw source: "+key)
    if record.get("handoff_particle_count") != len(source.ray_id):
        raise ValueError("Particle boundary handoff count differs from its raw source")
    if set(original) != set(_GUN) or set(emission) != set(_EMISSION):
        raise ValueError("Particle boundary requires the complete original gun and emission ledgers")
    ids = np.asarray(original["ray_id"])
    count = len(ids) if ids.ndim == 1 else 0
    if (not 1000 <= count <= _MAX_PARTICLES or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids > np.iinfo(np.int64).max)
            or len(np.unique(ids)) != count):
        raise ValueError("Original gun ledger needs unique nonnegative source IDs and a bounded emitted count")
    for key in _GUN:
        value = np.asarray(original[key])
        if value.shape != (count,) or value.dtype.hasobject or value.dtype.fields is not None:
            raise ValueError("Original gun ledger has an invalid vector: "+key)
        if key not in ("ray_id", "alive", "blocked_key") and value.dtype != np.dtype("float64"):
            raise ValueError("Original gun phase space must retain its original float64 arrays")
        if key not in ("alive", "ray_id", "blocked_key", "flight_time_s", "blocked_z_mm") and not np.isfinite(value).all():
            raise ValueError("Original gun ledger contains nonfinite phase space")
    if original["alive"].dtype != np.dtype(bool) or original["blocked_key"].dtype.kind != "U":
        raise ValueError("Original gun survival and stop labels have invalid dtypes")
    if original["blocked_key"].dtype.itemsize > 4096 or np.isinf(original["blocked_z_mm"]).any():
        raise ValueError("Original gun stop metadata is invalid")
    weights = np.asarray(original["weight"])
    positive = int(np.count_nonzero(weights > 0))
    if (np.any(weights < 0) or not 0 < math.fsum(weights) <= 1+1e-10
            or positive != source.original_positive_count):
        raise ValueError("Original positive-current count or absolute emission weights differ from the source")
    emission_digest = sha256()
    for value in (ids, weights):
        array = np.ascontiguousarray(value)
        emission_digest.update(str((array.dtype.str, array.shape)).encode())
        emission_digest.update(array.tobytes())
    if source.provenance.get("original_emission_digest") != emission_digest.hexdigest():
        raise ValueError("Original emission IDs and weights differ from the source ancestry checksum")
    nominal = record.get("original_gun_energy_offset_reference_ev")
    if isinstance(nominal, bool) or not isinstance(nominal, (int, float)) or not math.isfinite(nominal) or nominal <= 0:
        raise ValueError("Original gun energies require their positive nominal exit-energy reference")
    alive = np.asarray(original["alive"])
    clock = np.asarray(original["flight_time_s"])
    if (np.isinf(clock).any() or not np.isfinite(clock[alive]).all()
            or np.any(clock[alive] < 0) or np.any((nominal+original["energy_offset_ev"])[alive] <= 0)):
        raise ValueError("Original transmitted gun particles require actual clocks and positive kinetic energies")
    order = np.argsort(ids)
    found = np.searchsorted(ids[order], source.ray_id)
    if np.any(found >= count):
        raise ValueError("Handoff IDs are absent from the original gun")
    rows = order[found]
    if (not np.array_equal(ids[rows], source.ray_id)
            or not np.array_equal(weights[rows], source.weight) or np.any(source.weight <= 0)
            or not np.all(alive[rows]) or np.isfinite(original["blocked_z_mm"][rows]).any()
            or np.any(source.flight_time_s < clock[rows])):
        raise ValueError("Handoff membership, weights, survival or actual clocks differ from the original gun")
    if not np.array_equal(emission["ray_id"], ids):
        raise ValueError("Original emission identities differ from the original gun")
    for key in ("position_m", "direction", "normal"):
        value = np.asarray(emission[key])
        if value.shape != (count, 3) or value.dtype != np.dtype("float64") or not np.isfinite(value).all():
            raise ValueError("Original emission requires finite float64 positions and source-frame vectors")
        if key != "position_m" and not np.allclose(np.linalg.norm(value, axis=1), 1., rtol=0, atol=2e-10):
            raise ValueError("Original emission directions and normals must retain their unit-vector convention")
    if len(source.ray_id) > _MAX_PARTICLES:
        raise ValueError("Particle boundary exceeds its bounded source population")
    if source.canonical_frequency_shift_inv_angstrom is not None:
        maximum = source.provenance.get("maximum_canonical_frequency_shift_inv_angstrom")
        actual = float(np.max(np.linalg.norm(source.canonical_frequency_shift_inv_angstrom, axis=1)))
        if maximum is not None and not math.isclose(maximum, actual, rel_tol=2e-14, abs_tol=0):
            raise ValueError("Canonical-frequency metadata differs from the raw source shift")


@dataclass(frozen=True)
class ValidatedParticleBoundary:
    """A complete immutable classical boundary, with its original producer."""
    source: ParticleWaveSource
    inputs: InstrumentSnapshot
    original_gun: object
    emission_reference: object
    record: object

    def __post_init__(self):
        original, emission = _freeze_arrays(self.original_gun), _freeze_arrays(self.emission_reference)
        record = freeze_json(self.record)
        _validate(self.source, self.inputs, original, emission, record)
        object.__setattr__(self, "original_gun", original)
        object.__setattr__(self, "emission_reference", emission)
        object.__setattr__(self, "record", record)

    def restore_inputs_for_wave(self):
        """Restore captured parameters for an explicit later wave calculation.

        Only the implementation-version check is intentionally omitted here.
        No restart arrays are decoded, no particle equation is executed, and
        external content/inventory checks remain the normal snapshot checks.
        The original snapshot and source digests are never replaced.
        """
        state = decode_instrument(self.inputs.graph)
        self.inputs._verify_external_inputs(state)
        if state.electron_gun.nominal_exit_energy_ev != self.record["original_gun_energy_offset_reference_ev"]:
            raise ValueError("Captured gun energy reference differs from the original gun ledger")
        return state


def _json(data):
    def duplicate_safe(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate particle-boundary JSON key")
            result[key] = value
        return result
    return json.loads(data.decode("utf-8"), object_pairs_hook=duplicate_safe,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite particle-boundary JSON")))


def _arrays(data, allowed, required, maximum_bytes):
    with ZipFile(BytesIO(data)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if (len(set(names)) != len(names) or any(name not in {key+".npy" for key in allowed} for name in names)
                or not {key+".npy" for key in required} <= set(names)
                or sum(entry.file_size for entry in entries) > maximum_bytes):
            raise ValueError("Invalid or oversized particle-boundary array archive")
        for entry in entries:
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=16384)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=16384)
                else:
                    raise ValueError("Unsupported particle-boundary array header")
                if (dtype.hasobject or dtype.fields is not None or len(shape) > 2
                        or math.prod(shape)*dtype.itemsize > maximum_bytes):
                    raise ValueError("Unsafe or oversized particle-boundary array payload")
    with np.load(BytesIO(data), allow_pickle=False, max_header_size=16384) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def load_particle_wave_source(directory, *, maximum_bytes=_MAX_BYTES):
    """Read checked classical arrays without requiring CUDA or particle restart."""
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("Particle boundary byte budget must be a positive integer")
    folder = Path(directory).resolve()
    paths = [folder/name for name in (*_FILES, "checksums.json")]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("Particle boundary requires its six regular data files")
    if sum(path.stat().st_size for path in paths) > maximum_bytes:
        raise MemoryError("Particle boundary exceeds its read budget")
    data = {path.name: path.read_bytes() for path in paths}
    if sum(map(len, data.values())) > maximum_bytes:
        raise MemoryError("Particle boundary changed beyond its read budget")
    checks = _json(data["checksums.json"])
    if set(checks) != set(_FILES) or any(not _hex(checks[name]) or sha256(data[name]).hexdigest() != checks[name] for name in _FILES):
        raise ValueError("Particle boundary file checksum mismatch")
    inputs = InstrumentSnapshot.from_dict(_json(data["inputs.json"]))
    record = _json(data["source.json"])
    declared_bytes = len(data["inputs.json"])+len(data["source.json"])
    for name in ("source-phase.npz", "original-gun.npz", "emission-reference.npz"):
        with ZipFile(BytesIO(data[name])) as archive:
            declared_bytes += sum(entry.file_size for entry in archive.infolist())
    if declared_bytes > maximum_bytes:
        raise MemoryError("Expanded particle boundary exceeds its read budget")
    phase = _arrays(data["source-phase.npz"], (*_PHASE, _CANONICAL), _PHASE, maximum_bytes)
    for key in _PHASE:
        if key != "ray_id" and phase[key].dtype != np.dtype("float64"):
            raise ValueError("Particle phase space must retain original float64 values")
    if phase["ray_id"].dtype != np.dtype("int64") or (_CANONICAL in phase and phase[_CANONICAL].dtype != np.dtype("float64")):
        raise ValueError("Particle identities and canonical shifts must retain their original dtypes")
    source = ParticleWaveSource(**phase, **{key: record[key] for key in
        ("z_mm", "reference_current_a", "original_positive_count", "input_digest", "provenance")})
    original = _arrays(data["original-gun.npz"], _GUN, _GUN, maximum_bytes)
    emission = _arrays(data["emission-reference.npz"], _EMISSION, _EMISSION, maximum_bytes)
    return ValidatedParticleBoundary(source, inputs, original, emission, record)


def save_particle_wave_source(boundary, directory):
    """Write a validated classical boundary once; never replace an old result."""
    if not isinstance(boundary, ValidatedParticleBoundary):
        raise TypeError("Save requires a validated immutable particle boundary")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (folder/name).write_text(json.dumps(thaw_json(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    phase = {key: getattr(boundary.source, key) for key in _PHASE}
    if boundary.source.canonical_frequency_shift_inv_angstrom is not None:
        phase[_CANONICAL] = boundary.source.canonical_frequency_shift_inv_angstrom
    np.savez_compressed(folder/"source-phase.npz", **phase)
    np.savez_compressed(folder/"original-gun.npz", **boundary.original_gun)
    np.savez_compressed(folder/"emission-reference.npz", **boundary.emission_reference)
    write("inputs.json", boundary.inputs.to_dict())
    write("source.json", boundary.record)
    write("checksums.json", {name: sha256((folder/name).read_bytes()).hexdigest() for name in _FILES})
    return folder


__all__ = ("ValidatedParticleBoundary", "load_particle_wave_source", "save_particle_wave_source")
