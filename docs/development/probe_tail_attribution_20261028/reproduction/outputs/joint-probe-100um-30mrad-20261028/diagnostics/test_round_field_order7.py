"""Immutable reuse gates only; no new physical propagation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

FILE = Path(__file__).with_name("run_round_field_order7.py")
spec = importlib.util.spec_from_file_location("round_order7_wrapper", FILE)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
from test_round_field_sensitivity import arrays


@pytest.fixture
def prior(tmp_path):
    root = tmp_path/"previous"
    root.mkdir()
    raw = arrays()
    donor = SimpleNamespace(arrays=raw, snapshot=SimpleNamespace(digest="saved-input"), files_sha256={"donor": "hash"})
    driver = Path(wrapper.radial.__file__).read_bytes()
    (root/"driver-source.py").write_bytes(driver)
    report = dict(status="COMPLETE", source_input_unchanged=True, source_input_digest=donor.snapshot.digest,
        donor_files_sha256=donor.files_sha256, driver_sha256=wrapper._hash(root/"driver-source.py"),
        input_arrays_sha256={key: wrapper._array_hash(value) for key, value in raw.items()}, stages=[])
    for order in (3, 5):
        folder = root/f"order-{order}"
        folder.mkdir()
        wrapper._json(folder/"runtime.json", {"synthetic_order": order})
        wrapper._json(folder/"field-audit.json", {"synthetic_order": order})
        wrapper._npz(folder/"particles.npz", raw)
        stage = dict(order=order, status="COMPLETE", analysis={}, baseline_reproduction={"passed": True},
            files_sha256={name: wrapper._hash(folder/name) for name in ("runtime.json", "field-audit.json", "particles.npz")})
        wrapper._json(folder/"report.json", stage)
        report["stages"].append(stage)
    wrapper._json(root/"report.json", report)
    return donor, root, report


def test_prior_proofs_are_reused_without_changing_any_file(prior):
    donor, root, report = prior
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    proof, raw, analysis = wrapper.verify_prior(donor, root)
    assert proof["prior_propagations_reused"] == [3, 5]
    assert proof["new_propagations_requested"] == [7]
    assert proof["exact_replay_rechecked"]
    assert set(raw) == {3, 5}
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("kind", ["driver", "raw_hash", "status", "input", "stage_report"])
def test_prior_mismatch_prevents_reuse(prior, kind):
    donor, root, report = prior
    if kind == "driver":
        (root/"driver-source.py").write_text("changed")
    elif kind == "raw_hash":
        (root/"order-5/particles.npz").write_bytes(b"changed")
    elif kind == "status":
        report["status"] = "RUNNING"
        wrapper._json(root/"report.json", report)
    elif kind == "input":
        donor.snapshot.digest = "different"
    else:
        wrapper._json(root/"order-3/report.json", {"status": "COMPLETE"})
    with pytest.raises(ValueError):
        wrapper.verify_prior(donor, root)


def test_rehashed_phase_tamper_still_fails_independent_exact_replay_gate(prior):
    donor, root, report = prior
    changed = deepcopy(donor.arrays)
    changed["x_m"][1, 0] = np.nextafter(changed["x_m"][1, 0], np.inf)
    folder = root/"order-3"
    wrapper._npz(folder/"particles.npz", changed)
    report["stages"][0]["files_sha256"]["particles.npz"] = wrapper._hash(folder/"particles.npz")
    wrapper._json(folder/"report.json", report["stages"][0])
    wrapper._json(root/"report.json", report)
    with pytest.raises(ValueError, match="recheck"):
        wrapper.verify_prior(donor, root)


def test_raw_coordinate_comparison_does_not_subtract_centroids():
    old = arrays()
    new = deepcopy(old)
    new["x_m"] += 1e-9
    result = wrapper.compare_raw(old, new, 0.)
    assert result["maximum_raw_xy_shift_nm"] == pytest.approx(1.)
    assert result["maximum_surface_raw_xy_shift_nm"] == pytest.approx(1.)
    assert result["same_actual_population"]
