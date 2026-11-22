"""Verify paired focus trials and plot native specimen landings and through-focus profiles."""

from hashlib import sha256
import json
from shutil import copyfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adjust_a2 import read_json
from fit_threefold import EVIDENCE, HERE, angle_groups, grouped_current, read_sample
from plot_aperture_angle_spots import colour_bar, colour_scale, plot_groups, show_map
from plot_specimen_probe import landing_current_density
from refocus_a2 import FOCUS_OUT


def verify_pair(before, after, winding, corrector, summary):
    before_path = FOCUS_OUT / before["phase_file"]
    after_path = FOCUS_OUT / after["phase_file"]
    records = []
    for path, row in ((before_path, before), (after_path, after)):
        assert sha256(path.read_bytes()).hexdigest() == row["trajectory_sha256"]
        assert row["corrector_currents_A"] == corrector
        sample = read_sample(path, row, summary)
        with np.load(path, allow_pickle=False) as data:
            offsets = data["offsets_nm"]
            i = np.searchsorted(data["z_m"], summary["specimen_upper_surface_z_m"] + offsets*1e-9)
            profile = data["profiles"][i]*np.array([.001, .001, 1.])
        records.append((sample, offsets*.001, profile))

    assert before["source_digest"] == after["source_digest"]
    assert before["corrector_sha256"] == after["corrector_sha256"]
    assert before["current_ledger"]["C2_exit_A"] == after["current_ledger"]["C2_exit_A"]
    assert before["objective_At"] == winding["objective_At"]
    np.testing.assert_array_equal(before["currents_A"], winding["currents_A"])
    mini = np.array(winding["owner_per_group"]) == "mini_condenser"
    expected = np.array(winding["currents_A"])
    expected[mini] *= after["mini_gain"]
    np.testing.assert_array_equal(after["currents_A"], expected)
    assert after["objective_solution"]["curve_sha256"] == before["objective_solution"]["curve_sha256"]
    assert after["objective_solution"]["installed_geometry_digest"] == before["objective_solution"]["installed_geometry_digest"]
    with np.load(before_path) as left, np.load(after_path) as right:
        for key in ("ids", "weights", "energy_offset_eV", "offsets_nm", "component_names"):
            np.testing.assert_array_equal(left[key], right[key])
        same_contacts = np.array_equal(left["loss_owner"], right["loss_owner"])
    condensed = {}
    for (sample, _, _), row, label in zip(records, (before, after), ("before", "after")):
        radius_um = np.linalg.norm(sample.xy_um, axis=1)
        condensed[label] = {k: row[k] for k in (
            "trajectory_sha256", "source_digest", "emitted", "survived", "effective", "max_step_mm",
            "surface", "through_focus", "D95_minimum_bracket_nm", "RMS_minimum_bracket_nm",
            "ring", "fit", "current_ledger", "waists_mm", "common_waists_mm")}
        condensed[label]["angular_RMS_mrad"] = float(np.sqrt(np.average(sample.alpha_mrad**2, weights=sample.current_na)))
        condensed[label]["fixed_axis_core_1um_radius_nA"] = float(sample.current_na[radius_um <= 1.].sum())
    condensed.update(corrector_field_sha256=before["corrector_sha256"],
                     identical_intercepted_particle_ids_and_components=bool(same_contacts))
    return condensed, records


def plot(records, results):
    labels = ("A₂ adjusted", "Objective + Mini Condenser refocused")
    colors = ("#82654f", "#166c79")
    maps = [landing_current_density(s.xy_um, s.current_na, 8., .16)[0] for s, _, _ in records]
    limits = colour_scale(maps)
    fig = plt.figure(figsize=(10.6, 8.3), facecolor="white")
    fig.text(.082, .963, "Probe focus at the specimen upper surface", fontsize=19,
             fontweight="bold", va="top", color="#17212c")
    fig.text(.082, .911, "Fixed A₂ corrector  ·  300 keV  ·  100 µm C2 aperture  ·  32,768 original electrons",
             fontsize=10.6, va="top", color="#465362")
    top = fig.add_gridspec(1, 3, width_ratios=(1., 1., .045), left=.082, right=.884,
                           bottom=.451, top=.800, wspace=.22)
    for j, (grid, label, key) in enumerate(zip(maps, labels, ("before", "after"))):
        ax = fig.add_subplot(top[0, j])
        img = show_map(ax, grid, 8., [-8, -4, 0, 4, 8], limits)
        ax.set_title(label + f"\nSurface D95 {results[key]['surface']['D95_nm']*.001:.3f} µm",
                     fontsize=11, pad=8, linespacing=1.5)
        ax.set_xlabel("x (µm)")
        if j == 0:
            ax.set_ylabel("y (µm)")
    colour_bar(fig, img, fig.add_subplot(top[0, 2]))
    lower = fig.add_gridspec(1, 2, left=.087, right=.953, bottom=.116, top=.327, wspace=.27)
    for col, (title, key, window) in enumerate(
            (("95% current diameter", "D95", (-16, 23)), ("RMS radius", "RMS_radius", (-20, 132)))):
        ax = fig.add_subplot(lower[0, col])
        ax.axvline(0, color="#384956", lw=.9, ls=":")
        for (_, dz, profiles), color, label, state in zip(records, colors, labels, ("before", "after")):
            visible = (dz >= window[0]) & (dz <= window[1])
            ax.plot(dz[visible], profiles[visible, col], color=color, lw=1.5,
                    linestyle="--" if state == "before" else "-", label=label)
            point = results[state]["through_focus"][key]
            ax.plot(point["offset_um"], point["min_um"], "o", color=color, ms=4)
        ax.set(xlim=window, xlabel="Distance from surface (µm)", ylabel="Diameter (µm)" if col == 0 else "Radius (µm)")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        if col == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.text(.082, .028, "Dots mark measured minima. Positive distance follows the beam into the downstream column.",
             fontsize=9.5, color="#465362")
    fig.savefig(HERE / "objective-mini-focus.png", dpi=175)
    plt.close(fig)


def main():
    base = read_json(EVIDENCE / "a2-adjusted-working-point.json")
    summary = read_json(EVIDENCE / "comparison.json")
    winding = base["round_windings"]
    corrector = base["corrector_winding_currents_A"]
    paired = {}
    for before_name, name in (("baseline", "refocused"), ("before-half-step", "half-step"),
                              ("before-prefix", "prefix"), ("before-fine-xy", "fine-xy"), ("before-mesh5", "mesh5")):
        before = read_json(FOCUS_OUT / (before_name + ".json"))
        after = read_json(FOCUS_OUT / (name + ".json"))
        result, records = verify_pair(before, after, winding, corrector, summary)
        paired[name] = result
        if name == "refocused":
            confirmed = after
            np.testing.assert_allclose(before["surface"]["D95_nm"], base["after"]["surface"]["D95_nm"], rtol=1e-8)
            plot(records, result)
            sample = records[1][0]
            groups = grouped_current(sample)
            plot_groups(sample, angle_groups(sample), {
                "specimen_z_m": summary["specimen_upper_surface_z_m"], "angle_groups": groups,
                "max_angle_mrad": float(sample.alpha_mrad.max()),
            }, filename="refocused-angle-spots.png", setting="Objective + Mini Condenser refocused")
    selected = read_json(FOCUS_OUT / "matched-selected.json")
    for key in ("objective_At", "mini_gain"):
        assert selected[key] == confirmed[key]
    path = EVIDENCE / "refocused.npz"
    copyfile(FOCUS_OUT / confirmed["phase_file"], path)
    assert sha256(path.read_bytes()).hexdigest() == confirmed["trajectory_sha256"]
    data = dict(operating_point="D95 surface focus after A2 correction, using Objective and Mini Condenser only",
                energy_keV=base["energy_keV"], gun=base["gun"], C2_aperture_diameter_um=base["C2_aperture_diameter_um"],
                specimen_upper_surface_z_m=base["specimen_upper_surface_z_m"],
                fixed_corrector_windings_A=corrector, owner_per_group=winding["owner_per_group"],
                currents_A=confirmed["currents_A"], objective_At=confirmed["objective_At"],
                objective_turns=winding["objective_turns"], objective_A=confirmed["objective_A"], mini_gain=confirmed["mini_gain"],
                objective_solution=confirmed["objective_solution"], C2_foil=confirmed["C2_foil"],
                trajectory_file=path.name, trajectory_sha256=confirmed["trajectory_sha256"],
                paired_controls=paired, refocused_angle_groups=groups, current_search=read_json(FOCUS_OUT / "matched.json"))
    (EVIDENCE / "refocused-working-point.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
