"""Measure local zero-current diagnostic responses of installed scalar drives.

These optical response pencils do not sample the finite source, carry current,
qualify a probe, or replace the required fresh original-source calculation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.unified_field_particle import build_unified_plan, execute_unified_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus", action="store_true",
        help="Propose Objective drive from the previously executed native D95 minimum")
    args = parser.parse_args()
    state = InstrumentSnapshot.from_dict(json.loads((args.seed/"instrument.json").read_text())).restore()
    state.step_mm = .05
    parts = {item.key: item for item in (*state.lenses, *state.corrector_elements, *state.stigmators)}
    with np.load(args.seed/"original_source.npz") as f:
        live = f["exit_alive"]
        xy = np.column_stack([f["exit_"+key][live] for key in ("x_m", "y_m")])
        slope = np.column_stack([f["exit_"+key][live] for key in ("tx_rad", "ty_rad")])
        weights = f["exit_weight"][live]
    curvature = float(np.sum(weights[:, None]*xy*slope)/np.sum(weights[:, None]*xy**2))
    phi = np.arange(12)*2*np.pi/12
    radius = .5e-6
    position = np.vstack([np.zeros((1, 2)), radius*np.c_[np.cos(phi), np.sin(phi)]])
    coords = position/radius
    design = np.column_stack([np.ones(13), coords,
        coords[:, 0]**2-coords[:, 1]**2, 2*coords[:, 0]*coords[:, 1]])
    entrance = state.sample.z_mm-state.sample.thickness_nm*.5e-6
    def observe():
        plan = build_unified_plan(state, state.electron_gun.exit_plane_z_mm, entrance, save_z_mm=(entrance,))
        z, x, tx, y, ty = execute_unified_plan(state, plan,
            position[:, 0], position[:, 0]*curvature, position[:, 1], position[:, 1]*curvature,
            np.zeros(13), history_policy="sampled")[:5]
        coefficients = np.linalg.lstsq(design, np.c_[x[-1], y[-1]], rcond=None)[0]
        # The full two-component second-order response is retained, including
        # any residual that the selected normal/skew pair cannot remove.
        quadratic = coefficients[3:].ravel()*1e9
        linear = coefficients[1:].ravel()*1e9
        return quadratic, coefficients, np.c_[x[-1], y[-1], tx[-1], ty[-1]]
    baseline, coefficient, phase = observe()
    if args.focus:
        report = json.loads((args.seed/"report.json").read_text())
        profile = report["focus_profile"]
        best = min(profile, key=lambda row: row["d95_m"])
        if best is profile[0] or best is profile[-1]:
            raise ValueError("A boundary minimum cannot define a focus proposal")
        offset = (best["z_mm"]-entrance)*1e6
        def focus_proxy(value):
            delta = value-value.mean(axis=0)
            return -float(np.sum(delta[:, :2]*delta[:, 2:])/np.sum(delta[:, 2:]**2))*1e9
        lens = parts["objective_lens"]
        original = lens.percent
        lens.percent = original+.0001
        shifted = observe()[2]
        derivative = (focus_proxy(shifted)-focus_proxy(phase))/.0001
        lens.percent = original-offset/derivative
        lens.validate()
        result = dict(qualification="UNQUALIFIED_NATIVE_D95_FOCUS_PROPOSAL",
            seed=str(args.seed.resolve()), observed_native_minimum_offset_nm=offset,
            diagnostic_focus_response_nm_per_percent=derivative,
            diagnostic_count=13, diagnostic_current_a=0., mechanical_pose_changed=False)
        result["controls"] = {part.key: {name: float(getattr(part, name)) for name in
            ("percent", "strength_m3", "skew_strength_m3", "strength_m2", "strength_x_percent", "strength_y_percent")
            if hasattr(part, name)} for part in parts.values() if part.enabled}
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({key: value for key, value in result.items() if key != "controls"}), flush=True)
        print("Objective", lens.percent, flush=True)
        return
    jacobian = []
    hpc = parts["probe_hpc_hexapole"]
    step = 100000.
    for attribute in ("strength_m3", "skew_strength_m3"):
        original = getattr(hpc, attribute)
        values = []
        for sign in (1, -1):
            setattr(hpc, attribute, original+sign*step)
            values.append(observe()[0])
        setattr(hpc, attribute, original)
        jacobian.append((values[0]-values[1])/(2*step))
    jacobian = np.array(jacobian).T
    delta = np.linalg.lstsq(jacobian, -baseline, rcond=None)[0]
    for attribute, value in zip(("strength_m3", "skew_strength_m3"), delta):
        setattr(hpc, attribute, getattr(hpc, attribute)+float(value))
    hpc.validate()
    corrected, _, _ = observe()
    controls = {part.key: {name: float(getattr(part, name)) for name in
        ("percent", "strength_m3", "skew_strength_m3", "strength_m2", "strength_x_percent", "strength_y_percent")
        if hasattr(part, name)} for part in parts.values() if part.enabled}
    result = dict(controls=controls, seed=str(args.seed.resolve()),
        qualification="UNQUALIFIED_ZERO_CURRENT_LOCAL_RESPONSE_PROPOSAL",
        original_source_particles_used_for_qualification=0, diagnostic_count=13, diagnostic_current_a=0.,
        entrance_curvature_per_m=curvature, diagnostic_radius_m=radius,
        baseline_second_order_nm=baseline.tolist(), response_nm_per_m3=jacobian.tolist(),
        adjusted_second_order_nm=corrected.tolist(), mechanical_pose_changed=False,
        control_description="Installed HPC normal/skew intrinsic field drives; no learning table or added aberration kick")
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "controls"}, indent=2), flush=True)
    print("HPC", controls[hpc.key], flush=True)


if __name__ == "__main__":
    main()
