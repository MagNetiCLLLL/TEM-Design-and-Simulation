"""Archive a physically retraced A2 working point and compare the actual specimen maps."""

import argparse
from hashlib import sha256
import json
from shutil import copyfile

import numpy as np

from adjust_a2 import OUT, read_json
from fit_threefold import EVIDENCE, angle_groups, design, grouped_current, held_out, read_sample, ring_measure
from plot_aperture_angle_spots import plot_current_adjustment, plot_groups


def summarize(run):
    return {k: run[k] for k in ("emitted", "survived", "effective", "max_step_mm", "source_digest",
                                 "surface", "fit", "ring", "trajectory_sha256", "field_sha256")}


def publish(args):
    summary = read_json(EVIDENCE / "comparison.json")
    before = summary["runs"]["d95"]
    reference = read_json(EVIDENCE / "threefold-fit.json")
    before_sample = read_sample(EVIDENCE / before["trajectory_file"], before, summary)
    winding = read_json(EVIDENCE / before["winding_currents_file"])
    after = read_json(OUT / (args.final + ".json"))
    half = read_json(OUT / "hp-half.json")
    trials = {name: read_json(OUT / (name + ".json")) for name in args.trials}
    controls = {name: read_json(OUT / (name + ".json")) for name in args.controls}
    field = after["field"]
    for run in (half, after, *controls.values(), *trials.values()):
        np.testing.assert_array_equal(run["currents_A"], winding["currents_A"])
        assert run["objective_At"] == winding["objective_At"]
        assert run["current_ledger"]["C2_entrance_A"] > 0
        if run["emitted"] == before["emitted"]:
            assert run["source_digest"] == before["source_digest"]
            np.testing.assert_allclose(run["current_ledger"]["C2_exit_A"], before["current_ledger"]["C2_exit_A"],
                                       rtol=1e-14, atol=0)
    for run in controls.values():
        assert run["field"]["currents_A"] == field["currents_A"]
    assert after["field"]["base_circuit_sha256"] == summary["corrector_circuit_sha256"]
    assert field["currents_A"]["probe_tl21_lens"] == summary["corrector_currents_A"]["probe_tl21_lens"]
    assert field["currents_A"]["probe_tl22_lens"] == summary["corrector_currents_A"]["probe_tl22_lens"]

    origin = OUT / after["trajectory_file"]
    target = EVIDENCE / "a2-adjusted.npz"
    if sha256(origin.read_bytes()).hexdigest() != after["trajectory_sha256"]:
        raise ValueError("The confirmed specimen checkpoint has changed")
    copyfile(origin, target)
    sample = read_sample(target, after, summary)
    half_sample = read_sample(OUT / half["trajectory_file"], half, summary)
    native = dict(surface=before["surface"], ring=ring_measure(before_sample.xy_um, before_sample.current_na))
    plot_current_adjustment([before_sample, half_sample, sample], [native, half, after])
    groups = grouped_current(sample)
    plot_groups(sample, angle_groups(sample), dict(specimen_z_m=summary["specimen_upper_surface_z_m"],
                angle_groups=groups, max_angle_mrad=float(sample.alpha_mrad.max())),
                filename="a2-adjusted-angle-spots.png", setting="A₂-adjusted")
    basis, names = design(sample)
    validation, _, folds = held_out(sample, basis, names)
    with np.load(target, allow_pickle=False) as f:
        offset = f["offsets_nm"]
        indices = np.searchsorted(f["z_m"], summary["specimen_upper_surface_z_m"] + offset*1e-9)
        near = f["profiles"][indices]
        best = np.argmin(near[:, 0])
        rms = np.argmin(near[:, 1])
        axial = {
            "sampled_offset_range_um": [float(offset[0]*.001), float(offset[-1]*.001)],
            "lowest_D95_um": float(near[best, 0]*.001), "D95_offset_um": float(offset[best]*.001),
            "RMS_radius_at_minimum_um": float(near[rms, 1]*.001), "RMS_offset_um": float(offset[rms]*.001),
            "D95_bracket_nm": after["D95_minimum_bracket_nm"], "RMS_bracket_nm": after["RMS_minimum_bracket_nm"],
            "waists_mm": after["waists_mm"], "common_waists_mm": after["common_waists_mm"],
        }

    result = dict(
        operating_point="Physical A2 adjustment at the original D95-focus round-lens currents",
        trajectory_file=target.name, trajectory_sha256=after["trajectory_sha256"],
        baseline_trajectory_sha256=before["trajectory_sha256"],
        baseline_circuit_sha256=summary["corrector_circuit_sha256"],
        energy_keV=summary["energy_keV"], gun=summary["gun"], C2_aperture_diameter_um=summary["C2_diameter_um"],
        specimen_upper_surface_z_m=summary["specimen_upper_surface_z_m"],
        round_windings=winding, corrector_winding_currents_A=field["currents_A"],
        estimated_hpol_turns_per_winding=field["turns"]["probe_hpol_hexapole_x"],
        field=field, current_ledger=after["current_ledger"], C2_foil=after["C2_foil"],
        before=dict(**native, fit=reference["full_fit"]), after=summarize(after),
        angle_groups=groups, axial=axial, held_out_models=validation, training_fold_estimates=folds,
        calibration_trials={name: dict(**summarize(run), currents_A=run["field"]["currents_A"],
                                        material_size_mm=run["field"]["material_size_mm"],
                                        transverse_pitch_mm=run["field"].get("transverse_pitch_mm", .3))
                            for name, run in trials.items()},
        controls={name: dict(**summarize(run), material_mesh_id=run["field"]["mesh_id"],
                             material_size_mm=run["field"]["material_size_mm"],
                             transverse_pitch_mm=run["field"].get("transverse_pitch_mm", .3))
                  for name, run in controls.items()},
    )
    (EVIDENCE / "a2-adjusted-working-point.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(surface=after["surface"], axial=axial, current=field["currents_A"],
                          fit=after["fit"], ring=after["ring"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final", help="Verified full source, through-focus trial")
    parser.add_argument("--trials", nargs="+", required=True)
    parser.add_argument("--controls", nargs="+", required=True)
    publish(parser.parse_args())
