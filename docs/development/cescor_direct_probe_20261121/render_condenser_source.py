"""Publish the matched source images, winding ledger, axial fields and trajectories."""

import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


HERE = Path(__file__).resolve().parent
EV = HERE / "evidence"
OUT = HERE.parents[2] / "outputs/condenser-source-audit-20261122"
BLUE, TEAL, RED, DARK = "#1a68a0", "#007f78", "#c34e32", "#242d3b"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#a1aab0",
        "axes.titleweight": "semibold",
        "legend.frameon": False,
        "axes.labelcolor": DARK,
        "text.color": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
    }
)


def read(path):
    return json.loads(path.read_text())


def regions(ax, labels=False):
    for label, lo, hi in [("C1 lens", 450, 550), ("C2 lens", 550, 750), ("C3 lens", 830, 1000)]:
        ax.axvspan(
            lo,
            hi,
            color=("#e3ebef" if label != "C2 lens" else "#f1eee8"),
            alpha=0.72,
            zorder=-5,
            lw=0,
        )
        if labels:
            ax.text(
                (lo + hi) / 2,
                0.97,
                label,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=10,
            )
    ax.axvline(765.0, color="#77522d", lw=1, ls=":")
    ax.axvline(1070.0, color="#71818a", lw=1)
    ax.axhline(0.0, lw=0.65, color="#98a3a9", zorder=-3)
    ax.grid(axis="y", alpha=0.12)
    ax.set_xlim(385, 1075)


def field_plot(data):
    z = data["z_m"] * 1000
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 6.5), sharex=True)
    fig.subplots_adjust(left=0.095, right=0.96, top=0.84, bottom=0.115, hspace=0.33)
    fig.suptitle(
        "Axial magnetic field of the condenser section",
        x=0.095,
        y=0.965,
        ha="left",
        weight="bold",
        fontsize=19,
    )
    for ax, key, caption in zip(
        axes, ["before", "after"], ["Inherited split-winding currents", "Retuned series currents"]
    ):
        regions(ax, True)
        scale = 1 if key == "before" else 1000
        # Plot owners last so close overlap with the total remains visible in the curves.
        for i, color, name, ls in [
            (3, DARK, "All active sources", "-"),
            (0, BLUE, "C1 lens", "--"),
            (1, RED, "C2 lens", "-."),
            (2, TEAL, "C3 lens", ":"),
        ]:
            ax.plot(
                z, scale * data[key][i], label=name, color=color, ls=ls, lw=1.9 if i != 3 else 2.3
            )
        ax.set_ylabel("Axial $B_z$ (T)" if key == "before" else "Axial $B_z$ (mT)")
        ax.set_title(caption, loc="left", fontsize=12, pad=4)
        ax.set_ylim((-9.6, 3.35) if key == "before" else (-121, 94))
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        ncol=4,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.915),
        fontsize=10,
        columnspacing=2.0,
    )
    axes[-1].set_xlabel("Global axial position z (mm)")
    axes[-1].set_xticks([400, 450, 500, 550, 650, 750, 830, 915, 1000, 1070])
    fig.savefig(HERE / "condenser-source-fields.png", dpi=170, facecolor="white")
    plt.close(fig)


def ray_plot(traces, gun, runs):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.1), gridspec_kw={"width_ratios": [2.25, 1]})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.835, bottom=0.125, hspace=0.46, wspace=0.27)
    fig.suptitle(
        "Effective-source rays from C1 lens to the ADL entrance",
        x=0.075,
        y=0.966,
        ha="left",
        weight="bold",
        fontsize=19,
    )
    fig.text(
        0.075,
        0.905,
        "Same 32,768 emitted particles in both runs. Colored paths show the same 32 source IDs. Black curves show ± centred radial RMS.",
        fontsize=10.5,
    )
    pick = np.arange(0, 64, 2)
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(pick)))
    span_um = 0.0
    for name in traces:
        visible = (gun[name]["gun_z_mm"] >= 385) & (gun[name]["gun_z_mm"] < 450)
        segment_um = (
            np.concatenate((gun[name]["gun_paths_m"][visible], traces[name]["paths"]))[:, pick, 0]
            * 1e6
        )
        span_um = max(
            span_um,
            float(np.nanmax(np.abs(segment_um))),
            float(np.max(traces[name]["stats"][:, 0])),
        )
    span_um = float(np.ceil(1.22 * span_um))
    for row, (case, label) in enumerate(
        (("before", "Inherited currents"), ("after", "Retuned currents"))
    ):
        arr = traces[case]
        z = arr["z"] * 1000
        rms = arr["stats"][:, 0]
        source = gun[case]
        gz = source["gun_z_mm"]
        gmask = (gz >= 380) & (gz < 450 - 1e-8)
        ext = np.r_[gz[gmask], z]
        path = np.concatenate((source["gun_paths_m"][gmask], arr["paths"]))[:, :, 0] * 1e6
        for col, ax in enumerate(axes[row]):
            factor = 1000 if col else 1.0
            if not col:
                regions(ax, True)
            else:
                ax.grid(axis="y", alpha=0.15)
                ax.axhline(0, color="#bbc5cb", lw=0.7)
            for idx, color in zip(pick, colors):
                ax.plot(ext, path[:, idx] * factor, lw=0.65, alpha=0.72, color=color)
            ax.plot(z, rms * factor, color=DARK, lw=1.15)
            ax.plot(z, -rms * factor, color=DARK, lw=1.15)
            if col:
                ax.set_xlim(450, 550)
                ax.set_xticks([450, 475, 500, 525, 550])
                span = 1.20 * np.max(abs(path[(ext >= 450) & (ext <= 550)][:, pick])) * 1000
                ax.set_ylim(-span, span)
                ax.set_ylabel("x (nm)")
                ax.set_title("C1 lens detail", loc="left", fontsize=12)
            else:
                ax.set_xlim(385, 1071)
                ax.set_ylim(-span_um, span_um)
                ax.set_xticks([400, 500, 650, 765, 915, 1070])
                ax.set_ylabel("x (µm)")
                ax.set_title(label, loc="left", fontsize=12)
                ax.text(
                    765,
                    0.02,
                    "C2 aperture",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#77522d",
                )
            if row == 1:
                ax.set_xlabel("Global axial position z (mm)")
            ax.yaxis.set_major_locator(MaxNLocator(5))
        if case == "after":
            zi = runs[case]["images"][-1][0]["z_mm"]
            zw = runs[case]["waists_mm"][2][0]
            for ax in axes[row]:
                ax.axvline(zi, color=RED, lw=1.1, ls="--")
            axes[row, 1].annotate(
                f"Source image\n{zi:.2f} mm",
                xy=(zi, 0),
                xycoords="data",
                xytext=(0.72, 0.80),
                textcoords="axes fraction",
                ha="center",
                fontsize=9,
                color=RED,
                bbox=dict(facecolor="white", edgecolor="none", pad=1),
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8),
            )
            axes[row, 0].axvline(zw, color=RED, lw=1.1, ls="--")
            axes[row, 0].text(
                zw,
                0.75,
                f"{zw:.2f} mm",
                transform=axes[row, 0].get_xaxis_transform(),
                ha="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1),
                color=RED,
                fontsize=9,
            )
    fig.text(
        0.075,
        0.025,
        "Shading marks mechanical spans. The right panels have independent nanometre scales. ADL begins at 1070 mm.",
        fontsize=9,
        color="#52616b",
    )
    fig.savefig(HERE / "condenser-source-rays.png", dpi=165, facecolor="white")
    plt.close(fig)


def off_control_plot(runs):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.7), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.81, bottom=0.14, hspace=0.45, wspace=0.25)
    fig.suptitle(
        "Why four waists remain with C1 lens off",
        x=0.10,
        y=0.97,
        ha="left",
        weight="bold",
        fontsize=18,
    )
    fig.text(
        0.10,
        0.895,
        "Old C2 lens currents. Dots mark measured minima of the centred radial RMS radius.",
        fontsize=10.5,
    )
    for row, (prefix, n, step, label) in enumerate(
        [
            ("archived_", 4096, 0.025, "Fixed archived gun exit"),
            ("", 4096, 0.025, "Relaunched from the tip"),
        ]
    ):
        for col, (case, caption) in enumerate(
            [("c1_off", "C1 lens off"), ("both_off", "Both lenses off")]
        ):
            name = prefix + case
            ax = axes[row, col]
            with np.load(OUT / f"section-{name}-{n}-{step:g}.npz") as f:
                z = f["z"] * 1000
                r = f["stats"][:, 0] * 1000
            sel = (z >= 450 - 1e-6) & (z <= 550 + 1e-6)
            roots = runs[name + ("_check" if prefix else "")]["waists_mm"][0]
            ax.plot(z[sel], r[sel], color=BLUE if col == 0 else TEAL, lw=1.65)
            ax.scatter(roots, np.interp(roots, z, r), color=RED, s=24, zorder=5, clip_on=False)
            ax.set_title(f"{label} · {caption}", loc="left", fontsize=10.5)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.18)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if col == 0:
                ax.set_ylabel("RMS radius (nm)")
            if row == 1:
                ax.set_xlabel("Global axial position z (mm)")
            ax.set_xlim(450, 550)
            ax.set_xticks([450, 475, 500, 525, 550])
    fig.savefig(HERE / "condenser-source-off-controls.png", dpi=160, facecolor="white")
    plt.close(fig)


def publish():
    from hashlib import file_digest

    def digest(p):
        with p.open("rb") as fp:
            return file_digest(fp, "sha256").hexdigest()

    fields = read(OUT / "field-provenance.json")
    q = fields["summaries"]
    faces = [450, 550, 750, 830, 1000, 1070]
    runs = {}
    traces = {}
    guns = {}
    data = {}
    controls = {}
    for case, n, step in [
        ("before", 32768, 0.025),
        ("after", 32768, 0.025),
        ("after_check", 32768, 0.05),
        ("archived_before", 32768, 0.0375),
        ("archived_c1_off", 32768, 0.0375),
        ("archived_both_off", 32768, 0.0375),
        ("archived_c1_off_check", 4096, 0.025),
        ("archived_both_off_check", 4096, 0.025),
        ("c1_off", 4096, 0.025),
        ("both_off", 4096, 0.025),
    ]:
        actual = case.replace("_check", "")
        stem = f"section-{actual}-{n}-{step:g}"
        run = read(OUT / (stem + ".json"))
        runs[case] = run
        controls[case] = dict(
            n=n,
            step_mm=step,
            counts=[len(a) for a in run["waists_mm"]],
            C1_mm=run["waists_mm"][0],
            C2_to_C3_mm=run["waists_mm"][2],
            C3_to_ADL_mm=run["waists_mm"][4],
            principal_counts=run["principal_counts"],
            at_ADL=run["at_ADL"],
            foil_survivors=run["foil_survivors"],
        )
        if case in ("before", "after"):
            with np.load(OUT / (stem + ".npz")) as f:
                traces[case] = dict(f)
            with np.load(OUT / f"gun-{case}-{n}.npz") as f:
                guns[case] = dict(f)
            for k in (
                "stats",
                "paths",
                "ids",
                "phase",
                "source_ids",
                "weight",
                "loss",
                "count",
                "jacobian",
            ):
                data[case + "_" + k] = traces[case][k]
            for k in ("gun_paths_m", "gun_z_mm", "phase", "gun_key"):
                data[case + "_from_tip_" + k] = guns[case][k]
    before, after = runs["before"], runs["after"]
    fine = after["images"][-1][0]
    other = runs["after_check"]["images"][-1][0]
    for key in ("ray_id", "launch", "weight", "launch_energy", "ids"):
        if key in guns["before"]:
            np.testing.assert_array_equal(guns["before"][key], guns["after"][key])
    np.testing.assert_array_equal(before["current_A"][8:], after["current_A"][8:])
    np.testing.assert_array_equal(traces["before"]["z"], traces["after"]["z"])
    gap = traces["after"]["z"] >= 1.0
    assert np.all(traces["after"]["stats"][gap, 3:5] < 0)
    for case in ("after", "after_check"):
        assert controls[case]["counts"] == [1, 0, 1, 0, 0] and all(
            c == [1, 0, 1, 0, 0] for c in controls[case]["principal_counts"]
        )
        assert (
            len(runs[case]["images"][-1]) == 1
            and max(runs[case]["images"][-1][0]["magnifications"]) < 1
        )
        assert max(runs[case]["images"][-1][0]["angle_singular_nm_per_urad"]) < 0.0001
    for case, expected in [
        ("archived_before", 35),
        ("archived_c1_off", 4),
        ("archived_c1_off_check", 4),
        ("archived_both_off_check", 0),
        ("c1_off", 4),
        ("both_off", 0),
        ("archived_both_off", 0),
    ]:
        assert controls[case]["counts"][0] == expected
    with np.load(OUT / "fields.npz") as f:
        fielddata = dict(f)
        field_plot(fielddata)
    ray_plot(traces, guns, runs)
    off_control_plot(runs)
    data["observation_z_m"] = traces["after"]["z"]
    data.update({"field_" + k: v for k, v in fielddata.items()})
    np.savez_compressed(EV / "condenser-source-comparison.npz", **data)
    shutil.copyfile(OUT / "gun-after-32768.npz", EV / "condenser-source-after-gun.npz")
    shutil.copyfile(OUT / "field-provenance.json", EV / "condenser-source-field-audit.json")
    base = read(EV / "joint-objective-focus.json")
    from condenser_source_fields import FEM, OWNER

    with np.load(FEM) as f:
        material = json.loads(str(f["report"]))
    original = read(OUT / "archived-gun-field.json")
    if original["evaluator_sha256"] != fields["archived_gun_table_sha256"]:
        raise ValueError("The archived gun field could not be reproduced")
    summary = dict(
        mode="Upstream source imaging in three-condenser Nanoprobe Normal",
        parent="joint-objective-focus.json",
        scope="Tip to ADL entrance. Full-column convergence, probe and scan qualification are separate.",
        current_model="C1, C2, C3 and Mini use prescribed vacuum windings. Corrector and objective use the inherited material FEM.",
        intervals_mm=list(zip(faces[:-1], faces[1:])),
        C2_aperture_diameter_um=100.0,
        C1_aperture_diameter_um=1000.0,
        emission_A=15e-6,
        original_count=32768,
        reference_launch_energy_eV=0.3,
        before_group_A=before["current_A"],
        after_group_A=after["current_A"],
        owner_per_group=OWNER.tolist(),
        group_turns=fields["group_turns"],
        after_gun_key=str(guns["after"]["gun_key"]),
        after_net_At=q["after"]["net_At"],
        corrector_A=material["currents_A"],
        objective_At=base["objective_At"],
        objective_A=base["objective_A"],
        scan_calibration_file="joint-final-scan.json",
        scan_calibration_sha256=digest(EV / "joint-final-scan.json"),
        archived_gun_reconstruction=original,
        measurements=controls,
        image=fine,
        paired_probe_images=after["images"],
        image_step_check_mm=abs(fine["z_mm"] - other["z_mm"]),
        gap_waist_step_check_mm=abs(
            after["waists_mm"][2][0] - runs["after_check"]["waists_mm"][2][0]
        ),
        field_audit="condenser-source-field-audit.json",
        comparison="condenser-source-comparison.npz",
        fresh_gun="condenser-source-after-gun.npz",
        evidence_sha256={
            k: digest(EV / k)
            for k in (
                "condenser-source-comparison.npz",
                "condenser-source-field-audit.json",
                "condenser-source-after-gun.npz",
            )
        },
    )
    (EV / "condenser-source-working-point.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Published", summary["image"], summary["after_net_At"], flush=True)


if __name__ == "__main__":
    publish()
