"""Check native trajectories and publish the matched current and focus comparison."""

from hashlib import sha256
import json
from shutil import copyfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from column_current import ACTIVE, HERE, OUT, OWNER, POINT, STUDY, SURFACE, WORK, metrics
from temsim.module_manifest import read_document
from temsim.optics.column import default_state
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.lens_field_provider import lens_geometry_binding
from temsim.physics.probe_corrector_fem import circuit

EVIDENCE = HERE / "evidence"
PRIMARY = ("before", "rms", "d95")


def load_run(case, gun):
    source = json.loads((OUT / ("anchored-" + case + ".json")).read_text())[0]
    path = OUT / source["phase_file"]
    with np.load(path) as checkpoint:
        data = dict(checkpoint)
    weights, loss, ids = (data[key] for key in ("weights", "loss", "ids"))
    live = loss == -1
    n = len(ids)
    assert n >= 1000 and np.all(loss >= -1) and np.all(weights == 1 / n)
    np.testing.assert_array_equal(ids, gun["ray_id"][:n])
    np.testing.assert_array_equal(data["energy_offset_eV"], gun["energy_offset_ev"][:n])
    assert source["emitted"] == n and source["survived"] == live.sum()
    assert source["effective"] == live.sum()
    receipt = source["current_ledger"]
    gun_current = receipt["gun_exit_A"]
    np.testing.assert_allclose(gun_current, 15e-6, rtol=1e-14, atol=0)
    verified = metrics(data["phase"][live], weights[live], gun_current)
    for key, value in verified.items():
        np.testing.assert_allclose(source["surface"][key], value, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(data["loss_owner"] == -1, live)
    component = data["component_names"][data["loss_owner"][~live]]
    losses = {
        key: float(weights[~live][component == key].sum() * gun_current) for key in np.unique(component)
    }
    assert set(losses) == set(receipt["intercepted_A"])
    for key in losses:
        np.testing.assert_allclose(losses[key], receipt["intercepted_A"][key], rtol=1e-14, atol=0)
    np.testing.assert_allclose(
        receipt["specimen_A"] + sum(losses.values()), gun_current, rtol=1e-14, atol=0
    )
    for face, stage in (("entrance", "C2_entrance_A"), ("exit", "C2_exit_A")):
        np.testing.assert_allclose(source["C2_foil"][face]["current_A"], receipt[stage], rtol=1e-14, atol=0)
    output = {
        key: source[key]
        for key in (
            "source_digest",
            "emitted",
            "survived",
            "effective",
            "max_step_mm",
            "C2_foil",
            "surface",
            "current_ledger",
            "waists_mm",
            "common_waists_mm",
            "D95_minimum_bracket_nm",
            "RMS_minimum_bracket_nm",
        )
    }
    # Check the profile is the archived particle distribution in the specimen plane.
    i = np.flatnonzero(data["z_m"] == SURFACE).item()
    samples = data["profiles"]
    np.testing.assert_allclose(
        samples[i],
        [verified["D95_nm"], verified["rms_radius_nm"], verified["core_50pm_nA"]],
        rtol=1e-9,
        atol=1e-8,
    )
    near = np.searchsorted(data["z_m"], SURFACE + data["offsets_nm"] * 1e-9)
    output["near_min_D95_nm"] = float(samples[near, 0].min())
    output["near_min_RMS_radius_nm"] = float(samples[near, 1].min())
    output["near_max_core_50pm_nA"] = float(samples[near, 2].max())
    output["smallest_sample_axis_distance_m"] = float(np.hypot(*data["phase"][live, :2].T).min())
    if case in PRIMARY:
        output["trajectory_file"] = case + ".npz"
        output["trajectory_sha256"] = sha256(path.read_bytes()).hexdigest()
        copyfile(path, EVIDENCE / output["trajectory_file"])
        q = np.array(source["currents_A"])
        groups = OWNER[ACTIVE]
        excitation = dict(
            objective_At=source["objective_At"],
            objective_turns=1160,
            objective_A=source["objective_At"] / 1160,
            owner_per_group=groups.tolist(),
            currents_A=q.tolist(),
            signed_winding_groups_A={name: q[groups == name].tolist() for name in dict.fromkeys(groups)},
        )
        filename = case + "-currents.json"
        (EVIDENCE / filename).write_text(json.dumps(excitation, indent=2) + "\n")
        output["winding_currents_file"] = filename
    return output, data


def plot(data):
    fig = plt.figure(figsize=(10.4, 6.7), layout="constrained")
    layout = fig.add_gridspec(2, 2)
    envelope = fig.add_subplot(layout[0, :])
    focus = fig.add_subplot(layout[1, 0])
    enclosed = fig.add_subplot(layout[1, 1])
    for key, label, colour in [("before", "Before", "#7e6870"), ("rms", "Surface RMS setting", "#067774")]:
        d = data[key]
        z = d["z_m"] * 1000
        width = np.hypot(d["moments"][:, 1], d["moments"][:, 2]) * 1e6
        envelope.semilogy(z, width, label=label, color=colour, lw=1.5)
        live = d["loss"] == -1
        r = np.hypot(*d["phase"][live, :2].T)
        order = np.argsort(r)
        enclosed.step(
            r[order] * 1e6,
            15 * np.cumsum(d["weights"][live][order]),
            where="post",
            label=label,
            color=colour,
            lw=1.5,
        )
    envelope.set(xlim=(640, 1603), xlabel="Column z (mm)", ylabel="RMS beam radius (µm)")
    for where, name in [
        (650, "C2 lens"),
        (765, "C2 aperture"),
        (915, "C3 lens"),
        (1088.2, "ADL"),
        (1522.5, "Mini"),
        (SURFACE * 1000, "Specimen"),
    ]:
        envelope.axvline(where, lw=0.65, color="#777777", ls=":")
        envelope.text(where, 1.01, name, transform=envelope.get_xaxis_transform(), ha="center", fontsize=8)
    envelope.legend(loc="lower right", fontsize=8)
    d = data["rms"]
    offsets = d["offsets_nm"]
    near = np.searchsorted(d["z_m"], SURFACE + offsets * 1e-9)
    for col, label, color in [(0, "D95 diameter", "#4e61a4"), (1, "RMS radius", "#067774")]:
        v = d["profiles"][near, col]
        focus.plot(offsets / 1000, (v - v.min()) / 1000, color=color, label=label)
    focus.axvline(0, lw=1, ls="--", color="#444444")
    focus.set(
        xlim=(-220, 220),
        ylim=(0, None),
        xlabel="Distance from specimen surface (µm)",
        ylabel="Increase from own minimum (µm)",
        title="Through focus at the RMS setting",
    )
    focus.legend(fontsize=8)
    enclosed.set(
        xlim=(0.01, 300),
        ylim=(0, 15.5),
        xscale="log",
        xlabel="Radius about optical axis (µm)",
        ylabel="Enclosed specimen current (µA)",
        title="Absolute current inside a disk",
    )
    for ax in (envelope, focus, enclosed):
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "current-and-focus.png", dpi=175)
    plt.close(fig)


def archive():
    EVIDENCE.mkdir(exist_ok=True)
    gun_info = json.loads((OUT / "gun-boundary.json").read_text())
    gun_path = POINT / "gun.npz"
    assert sha256(gun_path.read_bytes()).hexdigest() == gun_info["original_gun_sha256"]
    with np.load(gun_path) as gun:
        gun_info["exit_fixed_axis_r95_um"] = float(
            1e6 * np.quantile(np.hypot(gun["x_m"], gun["y_m"]), 0.95)
        )
        slopes = np.hypot(gun["tx_rad"], gun["ty_rad"])
        gun_info["exit_fixed_axis_alpha95_urad"] = float(1e6 * np.quantile(np.arctan(slopes), 0.95))
        results, raw = {}, {}
        for case in (*PRIMARY, "rms-half-step", "rms-prefix", "d95-half-step", "d95-prefix"):
            results[case], raw[case] = load_run(case, gun)
    assert len({results[key]["source_digest"] for key in PRIMARY}) == 1
    for key in ("rms", "d95"):
        for name in ("ids", "weights", "loss_owner"):
            np.testing.assert_array_equal(raw[key][name], raw[key + "-half-step"][name])
        count = len(raw[key + "-prefix"]["ids"])
        np.testing.assert_array_equal(raw[key]["ids"][:count], raw[key + "-prefix"]["ids"])
        for case in (key, key + "-prefix", key + "-half-step"):
            for cohort in ("waists_mm", "common_waists_mm"):
                intervals = results[case][cohort]
                assert [len(v) for v in intervals["C2_to_C3"]] == [1, 1]
                assert intervals["C3_to_ADL"] == intervals["Mini_to_pre_sample"] == [[], []]
                for direction in intervals["C2_centre_window"]:
                    assert len(direction) == 1 and direction[0][1] < 650.0
    magnetic = json.loads((STUDY / "evidence/comparison.json").read_text())["fields"]["material"]
    field_file = WORK / "fine-field.npz"
    assert sha256(field_file.read_bytes()).hexdigest() == magnetic["sha256"]
    objectives = {
        str(ni): json.loads((WORK / f"column/armco-{ni}-gap-fine-p8.json").read_text())
        for ni in (24000, 36000)
    }
    for key in ("installed_geometry_digest", "curve_sha256"):
        assert len({obj[key] for obj in objectives.values()}) == 1
    document = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    assert (
        sha256(json.dumps(circuit(document), sort_keys=True).encode()).hexdigest()
        == magnetic["circuit_sha256"]
    )
    objective = lens_geometry_binding(default_state(), "objective_lens")
    assert (
        sha256(objective.canonical_geometry_json.encode()).hexdigest()
        == objectives["24000"]["installed_geometry_digest"]
    )
    stop_case = json.loads((OUT / "stop-trial.json").read_text())[0]
    summary = dict(
        status="Specimen-current condition unmet; no Si[110] scan or instrument preset.",
        gun=gun_info,
        target_sample_current_order_A=1e-10,
        C2_diameter_um=100,
        specimen_upper_surface_z_m=SURFACE,
        energy_keV=300,
        corrector_table_sha256=magnetic["sha256"],
        corrector_currents_A=magnetic["currents_A"],
        corrector_circuit_sha256=magnetic["circuit_sha256"],
        objective_fields=objectives,
        runs=results,
        high_interception_trial=stop_case,
        pencil_screen=json.loads((OUT / "overfill-summary-24000.json").read_text()),
    )
    (EVIDENCE / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot(raw)


if __name__ == "__main__":
    archive()
