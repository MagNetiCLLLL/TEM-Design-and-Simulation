"""Exercise the study writer and recovery with previously recorded arrays."""
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts import study_copper_coil_aberrations as study


@pytest.fixture(scope="module")
def interrupted_study(tmp_path_factory):
    archive = Path(__file__).resolve().parents[1] / "docs/development/copper_coil_aberration_20261113"
    case = archive / "baseline-half-step"
    request = json.loads((archive / "refinement-requests.json").read_text())[0]
    reference = json.loads((case / "reference.json").read_text())
    output = tmp_path_factory.mktemp("recovery") / "case"
    partitions = iter(range(2))

    class RecordedSolution:
        def __init__(self, part):
            self.part = part
            with np.load(case / f"characteristics-{part}.npz") as record:
                self.z, self.phase, self.ts = record["z_m"], record["phase"], record["integration_z_m"]

        def __call__(self, planes):
            np.testing.assert_array_equal(planes, self.z)
            return self.phase.reshape(len(planes), -1).T

    def recorded_wall(solution, radius, ids, **kwargs):
        with np.load(case / f"wall-{solution.part}.npz") as record:
            np.testing.assert_array_equal(record["source_ids"], ids)
            return dict(record, classification_complete=True)

    def interrupt(*args):
        raise RuntimeError("interrupted observer")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "propagate", lambda *args, **kwargs: (RecordedSolution(next(partitions)), {}))
        patch.setattr(study, "classify_cylindrical_wall", recorded_wall)
        patch.setattr(study, "phase_statistics", interrupt)
        with pytest.raises(RuntimeError, match="interrupted observer"):
            study.run_case(request, reference, output)
    return output, request, reference


@pytest.mark.parametrize("damage", ["none", "phase", "wall", "wall_ids", "receipt", "missing_wall_hash", "reference"])
def test_recovery_verifies_both_partitions_before_publishing(interrupted_study, tmp_path, monkeypatch, damage):
    original, request, reference = interrupted_study
    out = tmp_path / "case"
    shutil.copytree(original, out)
    receipt_path = out / "partition-1-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if damage in {"phase", "wall", "wall_ids"}:
        path = out / ("characteristics-1.npz" if damage == "phase" else "wall-1.npz")
        with np.load(path) as raw:
            values = dict(raw)
        if damage == "phase":
            values["phase"][1, 0, 0] += 1e-9
        elif damage == "wall":
            values["hit_z_m"][0] = 0.
        else:
            values["source_ids"] = values["source_ids"][::-1]
        np.savez_compressed(path, **values)
        if damage == "wall_ids":
            receipt["wall_sha256"] = sha256(path.read_bytes()).hexdigest()
    elif damage == "receipt":
        receipt["retained_phase_sha256"] = "0" * 64
    elif damage == "missing_wall_hash":
        del receipt["wall_sha256"]
    elif damage == "reference":
        reference = json.loads((out / "reference.json").read_bytes())
        reference["step_refined"]["coefficients"]["cc_mm"] *= 2
        study.write_json(out / "reference.json", reference)
    if damage in {"wall_ids", "receipt", "missing_wall_hash"}:
        study.write_json(receipt_path, receipt)

    def inventory():
        return {p.name: sha256(p.read_bytes()).digest() for p in out.iterdir() if p.name != "report.json"}

    before = inventory()
    monkeypatch.setattr(study, "propagate", lambda *a, **kw: pytest.fail("unexpected propagation"))
    if damage == "none":
        report = study.run_case(request, reference, out, resume_transport=True)
        assert report["populations"][-1]["at_paraxial_image"]["d95_nm"] == pytest.approx(716.5193581557007)
        assert report["completed_transport_reused_after_observer_failure"]
    else:
        with pytest.raises(ValueError, match="checksum|IDs|receipt|focus reference"):
            study.run_case(request, reference, out, resume_transport=True)
        assert not (out / "report.json").exists()
    assert inventory() == before
