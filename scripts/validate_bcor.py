"""Validate saved B-COR first-order relays on actual CUDA, without modifying inputs.

Finite-difference lab-frame propagation is independent of the candidate solver's
axisymmetric Larmor ODE. This validates position conjugacy, not full B-COR
nonlinear/off-axis aberration correction or physical Tip propagation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from validate_image_corrector import ROOT, inventory, paired_matrices, save


def execute(args):
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.compute_backend import choose_ray_backend

    args.output.mkdir(parents=True, exist_ok=True)
    before = inventory()
    start = perf_counter()
    backend, reason = choose_ray_backend("Require GPU", acceleration_enabled=True, ray_count=16)
    if backend != "CUDA GPU" or reason:
        raise RuntimeError(f"Actual CUDA required: {backend}: {reason}")
    candidate = (json.loads(args.candidate_json.read_text(encoding="utf-8"))["candidates"][0]["percent"]
                 if args.candidate_json else None)
    report = dict(status="RUNNING", scope="300 keV lab-frame first-order optical diagnostic; no source fitting",
                  qualification="NONLINEAR_AND_OFF_AXIS_CORRECTION_NOT_ESTABLISHED",
                  coordinates="(x,y,theta_x,theta_y), metres/radians; +Z downstream",
                  matrix_step_mm=.025, maximum_B_m_per_rad=1e-7, maximum_unit_magnification_error=1e-5,
                  candidate_override=candidate, variants={})
    for name in ("C3 + B-COR", "C3 + Probe Corrector + B-COR"):
        print(name, flush=True)
        state = default_state()
        AssemblyCatalog().apply(state, AssemblySelection("FEG", name, "No Energy Filter"))
        if candidate:
            for p in state.lenses:
                if p.key in candidate:
                    p.percent = float(candidate[p.key])
        state.acceleration_backend = "Require GPU"
        state.acceleration_enabled = True
        state.step_mm = .025
        parts = {p.key: p for p in state.image_corrector_system.components}
        planes = {f"hp{i}": parts[f"image_hp{i}_hexapole"].z_mm for i in (1, 2, 3)}
        planes.update(sample=state.sample.z_mm, sad=parts["image_sad_plane"].z_mm)
        for i in (1, 2, 3):
            planes[f"middle{i}"] = (parts[f"image_tl{i}1_lens"].z_mm+parts[f"image_tl{i}2_lens"].z_mm)/2
        receipts = []
        matrices = {}
        with backend_receipts(receipts):
            for source, targets in (("sample", ("middle1", "middle2", "middle3", "sad")),
                                    ("hp1", ("hp2",)), ("hp2", ("hp3",))):
                values = paired_matrices(state, planes[source], [planes[t] for t in targets],
                                         stable_axisymmetric=False)
                for target in targets:
                    matrices[f"{source}_to_{target}"] = values[str(planes[target])]
        returned = [r for r in receipts if r.get("stage") == "column" and r.get("outcome") == "returned"]
        checks = {"actual_cuda": bool(returned) and all(r.get("actual") == "CUDA GPU" for r in returned)}
        for key, row in matrices.items():
            m = row["canonical_to_canonical"]
            # Only prescribed references are acceptance targets. The third
            # mechanical midpoint is an observer, not a solved image plane.
            if key != "sample_to_middle3":
                checks[key+"_position_conjugacy"] = m["B_operator_norm_m_per_rad"] <= 1e-7
            if key.startswith("hp"):
                checks[key+"_unit_position_magnification"] = bool(np.max(np.abs(np.asarray(m["A_singular_values"])-1)) <= 1e-5)
        report["variants"][name] = dict(planes_mm=planes, checks=checks, matrices=matrices, backend_receipts=receipts,
            third_middle_is_sample_image=matrices["sample_to_middle3"]["canonical_to_canonical"]["B_operator_norm_m_per_rad"] <= 1e-7,
            third_middle_scope="Unconstrained mechanical midpoint; not declared a sample-conjugate plane",
            lens_percent={p.key: p.percent for p in state.image_corrector_system.round_lens_components})
        slug = "probe-bcor" if "Probe" in name else "bcor"
        save(args.output/(slug+"-inputs.json"), capture_instrument_snapshot(state).to_dict())
        save(args.output/"report.json", report)
    report["implementation_and_saved_inputs_unchanged"] = inventory() == before
    report["elapsed_s"] = perf_counter()-start
    passed = report["implementation_and_saved_inputs_unchanged"] and all(
        all(v["checks"].values()) for v in report["variants"].values())
    report["status"] = "PASS_POSITION_RELAYS_ONLY" if passed else "FAIL"
    save(args.output/"report.json", report)
    print(report["status"], report["elapsed_s"], flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/bcor/validation")
    parser.add_argument("--candidate-json", type=Path)
    raise SystemExit(execute(parser.parse_args()))
