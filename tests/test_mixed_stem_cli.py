"""Safe raw-array STEM export checks; no physical source or wave propagation."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from temsim import mixed_stem_cli as cli


@pytest.fixture
def acquisition(monkeypatch):
    from temsim import instrument_snapshot
    monkeypatch.setattr(instrument_snapshot, "capture_instrument_snapshot",
        lambda _: SimpleNamespace(to_dict=lambda: {"scope": "pure-array mock"}))
    original = SimpleNamespace(**{key: np.arange(3, dtype=float) for key in
        ("weight", "alive", "x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "flight_time_s")},
        ray_id=np.arange(3, dtype=np.int64))
    gun = SimpleNamespace(exit_bundle=original, blocked_z_mm=np.full(3, np.nan), blocked_key=("", "", ""))
    source = SimpleNamespace(auxiliary_arrays={"particle_id": np.arange(3, dtype=np.int64)},
                             record={"scope": "mock export only"})
    scan = SimpleNamespace(scan_x_nm=np.zeros((2, 2)), scan_y_nm=np.ones((2, 2)),
        fractions={"bf": np.full((2, 2), .1), "adf": np.full((2, 2), .02)},
        current_a={"bf": np.full((2, 2), 1e-9), "adf": np.full((2, 2), 2e-10)},
        diffraction_probability=np.arange(16, dtype=float).reshape(4, 4)*1e-5,
        qx_inv_angstrom=np.arange(4, dtype=float), qy_inv_angstrom=np.arange(4, dtype=float),
        source_checkpoint=source, record={"scope": "mock export only", "diffraction_position_nm": [0., 0.]})
    return SimpleNamespace(scan=scan, state_snapshot="mock inputs",
        particle_result=SimpleNamespace(simulation=SimpleNamespace(gun_trace=gun)))


def test_export_preserves_linear_raw_arrays_and_ledger(tmp_path, acquisition):
    cli.save_acquisition(acquisition, tmp_path)
    with np.load(tmp_path/"scan-and-diffraction.npz", allow_pickle=False) as values:
        np.testing.assert_array_equal(values["diffraction_probability"], acquisition.scan.diffraction_probability)
        np.testing.assert_array_equal(values["current_a_bf"], acquisition.scan.current_a["bf"])
        assert all(not values[name].dtype.hasobject for name in values.files)
    with np.load(tmp_path/"original-gun.npz", allow_pickle=False) as original:
        np.testing.assert_array_equal(original["ray_id"], np.arange(3))
        assert np.isnan(original["blocked_z_mm"]).all()
    hashes = json.loads((tmp_path/"checksums.json").read_text())
    assert len(hashes) == 6 and all(len(value) == 64 for value in hashes.values())


def test_export_never_overwrites_an_existing_result(tmp_path, acquisition):
    destination = tmp_path/"report.json"
    destination.write_bytes(b"original raw receipt")
    with pytest.raises(FileExistsError):
        cli.save_acquisition(acquisition, tmp_path)
    assert destination.read_bytes() == b"original raw receipt"
    assert list(tmp_path.iterdir()) == [destination]


def test_optional_diffraction_is_omitted_without_object_arrays(tmp_path, acquisition):
    acquisition.scan.diffraction_probability = None
    cli.save_acquisition(acquisition, tmp_path)
    with np.load(tmp_path/"scan-and-diffraction.npz", allow_pickle=False) as values:
        assert "diffraction_probability" not in values.files


def test_cuda_failure_occurs_before_any_input_or_output_access(tmp_path, monkeypatch):
    from temsim.physics import compute_backend
    def missing(*args):
        raise RuntimeError("CUDA unavailable")
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", missing)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        cli.main(["--profile", str(tmp_path/"absent.temwp"), "--output", str(tmp_path/"result")])
    assert not (tmp_path/"result").exists()
