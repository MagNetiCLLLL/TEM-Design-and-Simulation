"""Private executed-checkpoint store with streaming arrays and bound dependencies.

Only the pipeline constructs lookup keys from captured physical inputs. This
is not an import API for a user-supplied beam. No pickle or executable codec.
Incomplete writes are never indexed; immutable mode buffers load one at a time.
"""
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from temsim.immutable_json import freeze_json, thaw_json, json_digest
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_flux import WaveMode
from temsim.physics.wave_reference import AxialWaveReference


_READ_SESSION = ContextVar("executed_wave_read_session", default=None)


class _WaveReadSession:
    """One request's bounded, verified immutable snapshots, never a file cache.

    A mode becomes a snapshot only after the normal checksum and numerical
    validation have succeeded. A subsequent request checks the files again,
    including same-size/same-timestamp changes. Context-local ownership keeps
    worker requests isolated and releases the retained arrays on cancellation.
    """

    def __init__(self, maximum_bytes, cancelled):
        self.maximum_bytes = maximum_bytes
        self.cancelled = cancelled
        self._modes = OrderedDict()
        self.resident_bytes = 0
        self.peak_bytes = 0

    def check_cancelled(self):
        if self.cancelled():
            raise InterruptedError("Executed wave read cancelled")

    @staticmethod
    def _key(reader, row):
        # Different manifests can name the same directory; bind all mode
        # metadata and array checksums, not just its path or array timestamp.
        return reader.root, reader.reference, json_digest(row)

    def find(self, reader, row):
        self.check_cancelled()
        key = self._key(reader, row)
        item = self._modes.get(key)
        if item is None:
            return None
        self._modes.move_to_end(key)
        return item[0]

    def retain(self, reader, row, mode):
        self.check_cancelled()
        size = sum(value.nbytes for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad")
                   if (value := getattr(mode.plane, name)) is not None)
        if size > self.maximum_bytes:
            return
        key = self._key(reader, row)
        while self._modes and self.resident_bytes+size > self.maximum_bytes:
            _, (_, removed) = self._modes.popitem(last=False)
            self.resident_bytes -= removed
        self._modes[key] = mode, size
        self.resident_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)

    def clear(self):
        self._modes.clear()
        self.resident_bytes = 0


@contextmanager
def wave_read_session(maximum_bytes, *, cancelled=lambda: False):
    """Share checked mode snapshots only within this synchronous request.

    The caller includes this bounded retention in its memory reservation. A
    zero budget preserves the one-mode streaming path. File verification is
    unchanged outside the session, and no snapshot survives its exit.
    """
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 0:
        raise ValueError("Wave read-session memory budget must be a non-negative integer")
    session = _WaveReadSession(maximum_bytes, cancelled)
    token = _READ_SESSION.set(session)
    try:
        yield session
    finally:
        session.clear()
        _READ_SESSION.reset(token)


def _hash_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _StoredModes(Sequence):
    root: Path
    rows: tuple
    reference: str

    def __post_init__(self):
        object.__setattr__(self, "rows", freeze_json(self.rows))

    def __len__(self):
        return len(self.rows)

    def _array(self, item):
        path = self.root/item["file"]
        if path.parent != self.root or path.is_symlink() or _hash_file(path) != item["sha256"]:
            raise ValueError("Executed wave cache array checksum/path changed")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def geometries(self):
        """Plan a covering domain without reading every large amplitude."""
        session = _READ_SESSION.get()
        for row in self.rows:
            snapshot = None if session is None else session.find(self, row)
            if snapshot is not None:
                plane = snapshot.plane
                yield plane.amplitude.shape, plane.basis_m, plane.origin_m
                continue
            arrays = row["arrays"]
            yield tuple(arrays["amplitude"]["shape"]), self._array(arrays["basis_m"]), self._array(arrays["origin_m"])

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError("Read stored modes one at a time, without materialising a slice")
        row = self.rows[index]
        session = _READ_SESSION.get()
        snapshot = None if session is None else session.find(self, row)
        if snapshot is not None:
            return snapshot
        values = {}
        for field, item in row["arrays"].items():
            # mmap avoids a second heap-sized input; PlaneWave owns the final
            # immutable bytes so later external writes cannot mutate a mode.
            values[field] = self._array(item)
        reference = None if row["axial_reference"] is None else AxialWaveReference(**row["axial_reference"])
        mode = WaveMode(PlaneWave(**values), row["weight"], self.reference, row["mode_id"], row["energy_kev"],
                        reference, tuple(row.get("scattering_history", ())))
        if session is not None:
            session.retain(self, row, mode)
        return mode


@dataclass(frozen=True)
class _StoredBeam:
    modes: _StoredModes
    reference_plane: str
    content_identity: str

    @property
    def total_weight(self):
        return sum(r["weight"] for r in self.modes.rows)

    @property
    def resident_bytes(self):
        return 0


@dataclass(frozen=True)
class _StoredAuxiliary(Mapping):
    """Mandatory near-field traces, verified lazily without retaining buffers."""
    reader: _StoredModes
    rows: object

    def __post_init__(self):
        object.__setattr__(self, "rows", freeze_json(self.rows))

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, name):
        array = self.reader._array(self.rows[name])
        # Own an immutable snapshot after checksum validation. The Mapping
        # retains only metadata; a caller can release this one array at once.
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class _ModeWriter:
    def __init__(self, store, key, reference):
        store._validate_key(key)
        self.store, self.key, self.reference = store, key, reference
        # Only the atomic index publishes a checkpoint. Allocate its unique
        # directory at the final name so Windows readers/scanners may hold
        # child files without blocking a redundant directory rename.
        self.directory = store.root/(key+"-"+uuid4().hex)
        self.directory.mkdir()
        self._owned_directory = self.directory
        self._published = self._aborted = False
        self._index_temporary = None
        self.rows, self.bytes = [], 0
        self.mode_ids = set()
        self.auxiliary = {}

    def _require_open(self):
        if self._published or self._aborted:
            raise ValueError("Executed wave writer is already published or aborted")

    def append_auxiliary(self, name, value):
        """Retain mandatory non-plane state in the same atomic checkpoint."""
        self._require_open()
        if (not isinstance(name, str) or not name or len(name) > 80
                or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)
                or name in self.auxiliary):
            raise ValueError("Auxiliary wave state needs a unique plain array name")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("Executed auxiliary state cannot contain Python objects")
        self.store.check_space(array.nbytes+256)
        path = self.directory/("aux_"+name+".npy")
        np.save(path, array, allow_pickle=False)
        self.auxiliary[name] = {"file": path.name, "sha256": _hash_file(path),
                                "shape": array.shape, "nbytes": array.nbytes}
        self.bytes += path.stat().st_size
        self.store._used_bytes += path.stat().st_size

    def append(self, mode):
        self._require_open()
        if mode.reference_plane != self.reference:
            raise ValueError("Stored modes must retain their original electron reference")
        if mode.mode_id in self.mode_ids:
            raise ValueError("Duplicate stored wave mode")
        required = sum(getattr(mode.plane, n).nbytes+256 for n in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad")
                       if getattr(mode.plane, n) is not None)
        self.store.check_space(required)
        row = {"mode_id": mode.mode_id, "weight": mode.weight_per_reference_electron, "energy_kev": mode.energy_kev,
               "axial_reference": None if mode.axial_reference is None else asdict(mode.axial_reference),
               "scattering_history": thaw_json(mode.scattering_history), "arrays": {}}
        for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            value = getattr(mode.plane, name)
            if value is None:
                continue
            path = self.directory/f"m{len(self.rows)}_{name}.npy"
            np.save(path, value, allow_pickle=False)
            row["arrays"][name] = {"file": path.name, "sha256": _hash_file(path), "shape": value.shape,
                                   "nbytes": value.nbytes}
            self.bytes += path.stat().st_size
            self.store._used_bytes += path.stat().st_size
        self.rows.append(row)
        self.mode_ids.add(mode.mode_id)

    def finish(self, z_mm, current_a, record):
        self._require_open()
        if not self.rows:
            raise ValueError("An executed checkpoint needs at least one mode")
        data = {"schema": "executed-tip-wave-disk-v1", "dependency": thaw_json(self.store.dependency), "key": self.key,
                "reference_plane": self.reference, "plane_z_mm": float(z_mm), "current_a": current_a,
                "modes": self.rows, "record": thaw_json(freeze_json(record))}
        if self.auxiliary:
            data["auxiliary"] = self.auxiliary
        data["manifest_digest"] = json_digest(data)
        path = self.directory/"manifest.json"
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.store.check_space(len(payload)+512)  # manifest and atomic index
        path.write_bytes(payload)
        self.store._used_bytes += len(payload)
        # Index commit is atomic; readers never see a partially written mode.
        index = self.store.root/(self.key+".json")
        temporary = index.with_name("index-"+uuid4().hex+".json")
        self._index_temporary = temporary
        temporary.write_text(json.dumps({"directory": self.directory.name, "digest": data["manifest_digest"]}), encoding="utf-8")
        self.store._used_bytes += temporary.stat().st_size-(index.stat().st_size if index.exists() else 0)
        os.replace(temporary, index)
        # Publication precedes loading: a later read failure must never let
        # finally: abort() remove a checkpoint that is already indexed.
        self._published = True
        self._index_temporary = None
        return self.store.get(self.key)

    def abort(self):
        if self._published or self._aborted:
            return
        # Delete only this writer's original, unindexed directory. Never follow
        # a substituted directory/symlink or touch another writer's checkpoint.
        if (self.directory != self._owned_directory or self.directory.is_symlink()
                or self.directory.parent.resolve() != self.store.root
                or self.directory.resolve() != self._owned_directory):
            raise ValueError("Invalid private cache cleanup path")
        if self.directory.exists():
            shutil.rmtree(self.directory)
        if self._index_temporary is not None:
            self._index_temporary.unlink(missing_ok=True)
        self._aborted = True
        # A failed np.save/index write may precede its accounting update.
        self.store._used_bytes = sum(p.stat().st_size for p in self.store.root.rglob("*") if p.is_file())


class ExecutedWaveStore:
    def __init__(self, directory, dependency, maximum_bytes):
        # Callers, immutable checkpoint records and JSON decoders represent
        # sequences differently. Capture one immutable JSON identity before
        # any caller-owned nested container can change this store's keys.
        self._dependency = freeze_json(dependency)
        self._dependency_digest = json_digest(self._dependency)
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.maximum_bytes = maximum_bytes
        self._used_bytes = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    @property
    def dependency(self):
        return self._dependency

    def key(self, *parts):
        return json_digest(("executed-tip-wave-disk-v1", self.dependency, parts))

    def check_space(self, required):
        if self._used_bytes+required > self.maximum_bytes or shutil.disk_usage(self.root).free < required+64*1024**2:
            raise ValueError("Executed wave disk cache budget/free space exhausted; committed upstream checkpoints remain reusable")

    def writer(self, key, reference):
        return _ModeWriter(self, key, reference)

    @staticmethod
    def _validate_key(key):
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Executed cache keys must be calculated dependency digests")

    def _manifest(self, key):
        self._validate_key(key)
        index = self.root/(key+".json")
        if not index.exists():
            return None
        pointer = json.loads(index.read_text(encoding="utf-8"))
        directory = self.root/pointer["directory"]
        if directory.parent != self.root or directory.is_symlink():
            raise ValueError("Invalid executed cache directory")
        data = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        identity = data.pop("manifest_digest")
        if (identity != pointer["digest"] or identity != json_digest(data)
                or json_digest(data["dependency"]) != self._dependency_digest or data["key"] != key):
            raise ValueError("Executed wave cache identity/dependency changed")
        if data["schema"] != "executed-tip-wave-disk-v1":
            raise ValueError("Unsupported executed wave cache schema")
        return directory, data, identity

    def auxiliary_arrays(self, key):
        """Read only arrays indexed by a committed, dependency-bound result."""
        result = self._manifest(key)
        if result is None:
            return None
        directory, data, _ = result
        reader = _StoredModes(directory, (), data["reference_plane"])
        return {name: reader._array(item) for name, item in data.get("auxiliary", {}).items()}

    def get(self, key):
        from temsim.physics.tip_gun_wave import TipGunCheckpoint
        result = self._manifest(key)
        if result is None:
            return None
        directory, data, identity = result
        beam = _StoredBeam(_StoredModes(directory, data["modes"], data["reference_plane"]), data["reference_plane"], identity)
        return TipGunCheckpoint(beam, data["plane_z_mm"], data["current_a"],
            {**data["record"], "storage": {"manifest_digest": identity, "key": key,
                                          "dependency": self.dependency, "directory": str(directory)}},
            auxiliary_arrays=_StoredAuxiliary(beam.modes, data.get("auxiliary", {})))

    def put(self, key, checkpoint):
        writer = self.writer(key, checkpoint.beam.reference_plane)
        try:
            for mode in checkpoint.beam.modes:
                writer.append(mode)
            for name, array in checkpoint.auxiliary_arrays.items():
                writer.append_auxiliary(name, array)
            return writer.finish(checkpoint.plane_z_mm, checkpoint.reference_current_a, checkpoint.record)
        finally:
            writer.abort()

    def _completed_directory(self, identity):
        self._validate_key(identity)
        return self.root/"completed_planes"/self.key("completed-planes-v1", identity)

    @staticmethod
    def _complete_modes(checkpoint, expected_mode_ids):
        """Reject single-mode work products, even if their payload is valid."""
        record = checkpoint.record
        rows = checkpoint.beam.modes.rows
        mode_ids = tuple(row["mode_id"] for row in rows)
        contexts = record.get("unified_mode_contexts", {})
        if (record.get("schema") != "unified-source-column-v1"
                or record.get("unified_column_executed") is not True
                or not mode_ids or len(set(mode_ids)) != len(mode_ids)
                or mode_ids != tuple(expected_mode_ids) or set(contexts) != set(mode_ids)):
            raise ValueError("Completed-plane reference requires the entire committed mode mixture")
        if any(context.get("unified_wave_identity") != record.get("unified_wave_identity")
               for context in contexts.values()):
            raise ValueError("Completed-plane modes disagree about their executed field identity")
        from temsim.physics.potential_phase_budget import KineticPhaseLedger
        plane = record.get("unified_executed_plane_z_m")
        if (type(plane) not in (int, float) or not math.isfinite(plane)
                or plane*1000. != checkpoint.plane_z_mm):
            raise ValueError("Completed-plane checkpoint lost its exact executed coordinate")
        for context in contexts.values():
            kinetic = KineticPhaseLedger.from_dict(context.get("kinetic_phase_ledger"))
            geometry = context.get("potential_geometry", {})
            if (kinetic.binding is None or kinetic.last_z_m != plane
                    or kinetic.binding["snapshot_digest"] != record.get("snapshot_digest")
                    or geometry.get("last_z_m") != plane
                    or geometry.get("geometry_digest") != record.get("unified_geometry_digest")):
                raise ValueError("Completed-plane phase/geometry histories disagree with the complete mixture")
        return mode_ids

    def register_completed_plane(self, identity, checkpoint, *, expected_mode_ids,
                                 validate, cancelled=lambda: False):
        """Publish a durable reference only; never duplicate complex arrays.

        The caller owns the exact request/source/adapter/numerical identity and
        must validate every phase, electric and support history. The reference
        binds one complete committed mixture, never a collection of partial
        modes. Individual immutable files retain older planes across restarts.
        """
        if cancelled():
            raise InterruptedError("Completed-plane registration cancelled")
        directory = self._completed_directory(identity)
        if directory.resolve() != directory:
            raise ValueError("Invalid completed-plane reference directory")
        storage = checkpoint.record.get("storage", {})
        if json_digest(storage.get("dependency")) != self._dependency_digest or not storage.get("key"):
            raise ValueError("Completed-plane registration requires this store's committed checkpoint")
        committed = self.get(storage["key"])
        if (committed is None or committed.digest != checkpoint.digest
                or committed.record["storage"]["manifest_digest"] != storage.get("manifest_digest")):
            raise ValueError("Completed-plane checkpoint differs from its committed manifest")
        mode_ids = self._complete_modes(committed, expected_mode_ids)
        validate(committed)
        payload = {"schema": "completed-unified-plane-reference-v1", "dependency": thaw_json(self.dependency),
            "identity": identity, "key": storage["key"], "manifest_digest": storage["manifest_digest"],
            "checkpoint_digest": committed.digest, "plane_z_mm": committed.plane_z_mm,
            "mode_ids": mode_ids}
        digest = json_digest(payload)
        path = directory/(digest+".json")
        content = json.dumps({**payload, "digest": digest}, ensure_ascii=False).encode("utf-8")
        if path.exists():
            if (path.is_symlink()
                    or json_digest(json.loads(path.read_text(encoding="utf-8")))
                    != json_digest({**payload, "digest": digest})):
                raise ValueError("Completed-plane reference changed")
            return
        self.check_space(len(content))
        if cancelled():
            raise InterruptedError("Completed-plane registration cancelled before publication")
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory/("pending-"+uuid4().hex+".tmp")
        try:
            temporary.write_bytes(content)
            if cancelled():
                raise InterruptedError("Completed-plane registration cancelled before publication")
            os.replace(temporary, path)
            self._used_bytes += len(content)
        finally:
            temporary.unlink(missing_ok=True)

    def find_completed_plane(self, identity, minimum_z_mm, target_z_mm, *, validate,
                             cancelled=lambda: False):
        """Load the closest complete forward prefix for an exact request.

        All reference and manifest identities fail closed. Array verification
        streams modes, including exact-plane observations. A later checkpoint
        is never propagated backwards or combined with partially saved modes.
        """
        if (not all(isinstance(z, (int, float)) and not isinstance(z, bool) and math.isfinite(z)
                    for z in (minimum_z_mm, target_z_mm)) or minimum_z_mm > target_z_mm):
            raise ValueError("Completed-plane lookup requires finite forward bounds")
        if cancelled():
            raise InterruptedError("Completed-plane lookup cancelled")
        directory = self._completed_directory(identity)
        if directory.resolve() != directory:
            raise ValueError("Invalid completed-plane reference directory")
        candidates = []
        for path in directory.glob("*.json"):
            if cancelled():
                raise InterruptedError("Completed-plane lookup cancelled")
            if path.is_symlink():
                raise ValueError("Invalid completed-plane reference path")
            row = json.loads(path.read_text(encoding="utf-8"))
            digest = row.pop("digest")
            if (digest != path.stem or json_digest(row) != digest
                    or row.get("schema") != "completed-unified-plane-reference-v1"
                    or row.get("identity") != identity
                    or json_digest(row.get("dependency")) != self._dependency_digest):
                raise ValueError("Completed-plane reference identity changed")
            z = row["plane_z_mm"]
            if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z):
                raise ValueError("Invalid completed-plane reference coordinate")
            if minimum_z_mm <= z <= target_z_mm:
                candidates.append(row)
        candidates.sort(key=lambda row: (row["plane_z_mm"], row["key"]), reverse=True)
        if not candidates:
            return None
        row = candidates[0]
        checkpoint = self.get(row["key"])
        if (checkpoint is None or checkpoint.plane_z_mm != row["plane_z_mm"]
                or checkpoint.digest != row["checkpoint_digest"]
                or checkpoint.record["storage"]["manifest_digest"] != row["manifest_digest"]):
            raise ValueError("Completed-plane reference lost or changed its committed checkpoint")
        self._complete_modes(checkpoint, row["mode_ids"])
        for mode in checkpoint.beam.modes:
            if cancelled():
                raise InterruptedError("Completed-plane payload verification cancelled")
            del mode
        validate(checkpoint)
        if cancelled():
            raise InterruptedError("Completed-plane lookup cancelled before reuse")
        return checkpoint


def resident_wave_bytes(beam):
    return beam.resident_bytes if isinstance(beam, _StoredBeam) else sum(m.plane.amplitude.nbytes for m in beam.modes)
