"""Safe classical boundary interchange with synthetic arrays, no physical trial."""
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import InstrumentSnapshot, SNAPSHOT_SCHEMA
from temsim.physics import particle_wave_source_io as codec
from temsim.physics.particle_wave_source import ParticleWaveSource


@pytest.fixture
def boundary():
    inputs = InstrumentSnapshot({"schema": SNAPSHOT_SCHEMA, "scope": "Synthetic format fixture only"}, (), "a"*64)
    ids = np.arange(1000, dtype=np.int64)
    weight = np.full(1000, .001)
    digest = sha256()
    for array in (ids, weight):
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    indices = np.array([7, 12, 19])
    shift = np.array([[1., 2.], [2., 3.], [3., 4.]])*1e-7
    source = ParticleWaveSource(x_m=np.array([-1., 0., 1.])*1e-10, y_m=np.zeros(3),
        tx_rad=np.zeros(3), ty_rad=np.zeros(3), ray_id=ids[indices], weight=weight[indices],
        kinetic_energy_ev=np.array([300000.1, 300000.2, 300000.3]), flight_time_s=np.full(3, 1e-8),
        z_mm=100., reference_current_a=1e-8, original_positive_count=1000, input_digest=inputs.digest,
        canonical_frequency_shift_inv_angstrom=shift,
        provenance={"scope": "Synthetic array interchange fixture; not an executed physical source",
            "original_emission_digest": digest.hexdigest(),
            "maximum_canonical_frequency_shift_inv_angstrom": float(np.max(np.linalg.norm(shift, axis=1)))})
    gun = {key: np.zeros(1000) for key in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev")}
    gun.update(weight=weight, ray_id=ids, alive=np.ones(1000, bool), flight_time_s=np.full(1000, 5e-9),
               blocked_z_mm=np.full(1000, np.nan), blocked_key=np.full(1000, "", dtype="U1"))
    reference = dict(ray_id=ids, position_m=np.zeros((1000, 3)),
        direction=np.tile([0., 0., 1.], (1000, 1)), normal=np.tile([0., 0., 1.], (1000, 1)))
    record = {key: getattr(source, key) for key in ("z_mm", "reference_current_a", "original_positive_count",
        "input_digest", "source_fraction", "mean_energy_ev")}
    record.update(schema=codec.SCHEMA, source_digest=source.digest, provenance=thaw_json(source.provenance),
        source_generation_implementation=inputs.implementation, source_archive_sha256="b"*64,
        original_gun_energy_offset_reference_ev=300000., handoff_particle_count=3)
    return codec.ValidatedParticleBoundary(source, inputs, gun, reference, record)


def _rehash(folder):
    files = json.loads((folder/"checksums.json").read_text())
    for name in files:
        files[name] = sha256((folder/name).read_bytes()).hexdigest()
    (folder/"checksums.json").write_text(json.dumps(files), encoding="utf-8")


def _change_npz(path, key, update):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    update(arrays[key])
    np.savez_compressed(path, **arrays)


def test_classical_boundary_roundtrip_preserves_arrays_identity_and_old_producer_without_cuda(boundary, tmp_path, monkeypatch):
    from temsim.physics import compute_backend
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", lambda *_a, **_k: pytest.fail("Classical read required CUDA"))
    folder = codec.save_particle_wave_source(boundary, tmp_path/"new-boundary")
    loaded = codec.load_particle_wave_source(folder)
    assert loaded.source.digest == boundary.source.digest
    assert loaded.inputs.digest == boundary.inputs.digest
    assert loaded.inputs.implementation == "a"*64
    assert loaded.record["source_generation_implementation"] == "a"*64
    for name in codec._PHASE+ (codec._CANONICAL,):
        np.testing.assert_array_equal(getattr(loaded.source, name), getattr(boundary.source, name))
        assert not getattr(loaded.source, name).flags.writeable
    for key in loaded.original_gun:
        np.testing.assert_array_equal(loaded.original_gun[key], boundary.original_gun[key])
        assert not loaded.original_gun[key].flags.writeable
    with pytest.raises(FileExistsError):
        codec.save_particle_wave_source(boundary, folder)


def test_later_wave_input_restore_checks_inventory_and_retains_historical_implementation(boundary, monkeypatch):
    state = SimpleNamespace(electron_gun=SimpleNamespace(nominal_exit_energy_ev=300000.))
    calls = []
    monkeypatch.setattr(codec, "decode_instrument", lambda graph: state)
    monkeypatch.setattr(InstrumentSnapshot, "_verify_implementation", lambda *_: pytest.fail("Old producer was treated as a current Lorentz restart"))
    monkeypatch.setattr(InstrumentSnapshot, "_verify_external_inputs", lambda self, actual: calls.append((self.digest, actual)))
    assert boundary.restore_inputs_for_wave() is state
    assert calls == [(boundary.inputs.digest, state)]
    assert boundary.inputs.implementation == "a"*64


def test_boundary_byte_budget_and_checksum_fail_before_array_decode(boundary, tmp_path, monkeypatch):
    folder = codec.save_particle_wave_source(boundary, tmp_path/"boundary")
    monkeypatch.setattr(codec, "_arrays", lambda *_: pytest.fail("Bad file reached array decode"))
    with pytest.raises(MemoryError, match="read budget"):
        codec.load_particle_wave_source(folder, maximum_bytes=100)
    (folder/"source-phase.npz").write_bytes(b"bad-file")
    with pytest.raises(ValueError, match="checksum"):
        codec.load_particle_wave_source(folder)


@pytest.mark.parametrize("key,update,message", [
    ("ray_id", lambda a: a.__setitem__(0, a[1]), "unique"),
    ("weight", lambda a: a.__setitem__(20, a[20]*2), "emission|weights"),
    ("alive", lambda a: a.__setitem__(7, False), "membership|survival"),
    ("flight_time_s", lambda a: a.__setitem__(7, 2e-8), "clocks"),
    ("energy_offset_ev", lambda a: a.__setitem__(7, -300001.), "positive kinetic"),
])
def test_rehashed_gun_tampering_cannot_change_the_original_ancestry(boundary, tmp_path, key, update, message):
    folder = codec.save_particle_wave_source(boundary, tmp_path/"boundary")
    _change_npz(folder/"original-gun.npz", key, update)
    _rehash(folder)
    with pytest.raises(ValueError, match=message):
        codec.load_particle_wave_source(folder)


@pytest.mark.parametrize("key", ["weight", "kinetic_energy_ev", "flight_time_s", "canonical_frequency_shift_inv_angstrom"])
def test_rehashed_phase_tampering_cannot_relabel_the_source_digest(boundary, tmp_path, key):
    folder = codec.save_particle_wave_source(boundary, tmp_path/"boundary")
    _change_npz(folder/"source-phase.npz", key, lambda a: a.__setitem__(0, a[0]*1.01+1e-12))
    _rehash(folder)
    with pytest.raises(ValueError, match="source checksum"):
        codec.load_particle_wave_source(folder)


def test_source_snapshot_digest_and_original_producer_are_independently_checked(boundary):
    record = thaw_json(boundary.record)
    record["source_generation_implementation"] = "c"*64
    with pytest.raises(ValueError, match="producer"):
        replace(boundary, record=record)
    other = InstrumentSnapshot(boundary.inputs.graph, (), "c"*64)
    with pytest.raises(ValueError, match="snapshot digests"):
        replace(boundary, inputs=other)


def test_emission_frame_membership_and_unit_convention_are_checked(boundary):
    reference = {key: value.copy() for key, value in boundary.emission_reference.items()}
    reference["direction"][0] *= 2
    with pytest.raises(ValueError, match="unit-vector"):
        replace(boundary, emission_reference=reference)


def test_object_payload_is_rejected_before_numpy_can_unpickle(boundary, tmp_path):
    folder = codec.save_particle_wave_source(boundary, tmp_path/"boundary")
    np.savez_compressed(folder/"source-phase.npz", **{key: np.array([object()], object) for key in codec._PHASE})
    _rehash(folder)
    with pytest.raises(ValueError, match="Unsafe"):
        codec.load_particle_wave_source(folder)


def test_compressed_arrays_share_one_expanded_read_budget(boundary, tmp_path):
    folder = codec.save_particle_wave_source(boundary, tmp_path/"boundary")
    compressed = sum(path.stat().st_size for path in folder.iterdir())
    assert compressed < 50000
    with pytest.raises(MemoryError, match="Expanded particle boundary"):
        codec.load_particle_wave_source(folder, maximum_bytes=50000)
