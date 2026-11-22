"""Publish the matched probe and corrector source-image measurements."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from relay_transfer import HERE, HP1, HP2, TL21, TL22, WORK


def read(tag):
    return json.loads((WORK / (tag + ".json")).read_text(encoding="utf-8"))


def record(tag):
    d, a = read(tag), read(tag + "-axial")
    lens = {k: d[k] for k in ("currents_A", "objective_At", "field_sha256", "emitted", "survived",
                              "source_digest", "current_ledger", "foil_entrance_D95_um", "specimen",
                              "physical_edge_cohorts", "relay")}
    lens.update(rim=d["rim_azimuth_check"]["angle_mrad_min_median_max"],
                diagnostic_rays=d["rim_azimuth_check"]["reached_sample"],
                **{k: a[k] for k in ("D95_minimum_bracket_nm", "RMS_minimum_bracket_nm", "common_waists_mm")},
                internal=read(tag + "-internal"), run=tag)
    return lens


def publish(tag, extras):
    baseline = "registered-final"
    case_names = [baseline, tag]
    results = [record(n) for n in case_names]
    assert results[0]["source_digest"] == results[1]["source_digest"]
    archive = {}
    maps = []
    traces = []
    for name, result in zip(case_names, results, strict=True):
        with np.load(WORK / (name + ".npz")) as data:
            for k in ("ids", "weights", "energy_offset_eV"):
                if k in archive:
                    np.testing.assert_array_equal(archive[k], data[k])
                else:
                    archive[k] = data[k]
            live = data["loss"] == -1
            p, weights = data["phase"][live, :2], data["weights"][live]
            assert round(15000 * weights.sum(), 7) == round(result["specimen"]["transmitted_uA"] * 1000, 7)
            result["lost_source_ids"] = data["ids"][~live].tolist()
            maps.append((p * 1e6, weights))
            archive[name + "_loss"] = data["loss"]
            archive[name + "_phase"] = data["phase"]
        with np.load(WORK / (name + "-axial.npz")) as data:
            z = data["z_m"] * 1000
            width = data["profiles"][:, 0] * .001
            idx = (z >= 1120) & (z <= 1420)
            traces.append((z[idx], width[idx]))
            result["interpolated_D95_um_at_HP2_TLcenter_HP1"] = np.interp(
                1000 * np.array([HP2, (TL21 + TL22) / 2, HP1]), z, width
            ).tolist()
    doc = {"baseline": results[0], "candidate": results[1], "additional_runs": {}}
    for n in extras:
        raw = read(n)
        doc["additional_runs"][n] = {k: raw[k] for k in (
            "currents_A", "objective_At", "specimen", "field_sha256", "foil_entrance_D95_um", "source_digest",
            "emitted", "survived", "drift_D95_um", "physical_edge_cohorts"
        )}
        doc["additional_runs"][n]["rim_mrad"] = raw["rim_azimuth_check"]["angle_mrad_min_median_max"]
        for suffix in ("-axial", "-internal"):
            if (WORK / (n + suffix + ".json")).is_file():
                v = read(n + suffix)
                doc["additional_runs"][n][suffix[1:]] = (
                    {k: v[k] for k in ("D95_minimum_bracket_nm", "common_waists_mm")} if suffix == "-axial" else v
                )
    evidence = HERE / "evidence"
    (evidence / "internal-optics.json").write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(evidence / "internal-optics-landings.npz", **archive)

    fig = plt.figure(figsize=(10.7, 7.1), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.12, .83], width_ratios=[1, 1, .055, 1.12])
    axes = [fig.add_subplot(gs[0, j]) for j in range(2)]
    cax, enc, core = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, :])
    width = max(r["specimen"]["D95_nm"] for r in results) * .00065
    edges = np.linspace(-width, width, 98)
    pitch = edges[1] - edges[0]
    densities = [np.histogram2d(*p.T, bins=(edges, edges), weights=w * 15000 / pitch**2)[0] for p, w in maps]
    norm = LogNorm(vmin=min(np.min(h[h > 0]) for h in densities), vmax=max(h.max() for h in densities))
    colors = ["#4b596b", "#007974"]
    labels = ["Previous relay", "Retuned illumination"]
    for j, ((p, weights), h, r, ax, (z, wd)) in enumerate(zip(maps, densities, results, axes, traces, strict=True)):
        color, label = colors[j], labels[j]
        im = ax.pcolormesh(edges, edges, np.ma.masked_where(h.T <= 0, h.T), cmap="magma", norm=norm, rasterized=True)
        ax.set(aspect="equal", xlabel="x (µm)", title=label)
        ax.set_ylabel("y (µm)" if j == 0 else "")
        center = np.average(p, axis=0, weights=weights)
        d95 = r["specimen"]["D95_nm"] * .001
        ax.add_patch(plt.Circle(center, d95 / 2, fill=False, ls="--", ec="#259dc1", lw=1))
        radius = np.linalg.norm(p - center, axis=1)
        order = np.argsort(radius)
        enc.plot(radius[order], np.cumsum(weights[order]) / weights.sum(), color=color, label=f"{label}  {d95:.2f} µm")
        core.plot(z, wd, color=color, label=label, lw=1.7)
    fig.colorbar(im, cax=cax, label="Current density (nA/µm²)")
    enc.axhline(.95, c="#777777", ls="--", lw=.9)
    enc.set(xlabel="Radius about centroid (µm)", ylabel="Enclosed current fraction", title="Specimen upper surface",
            xlim=(0, width * 1.2), ylim=(0, 1.035))
    enc.legend(title="D95 diameter", loc="lower right", fontsize=8)
    core.axvspan(TL22 * 1000, TL21 * 1000, color="#7164be", alpha=.12, label="TL22 to TL21")
    core.axvspan(1387.5, 1412.5, color="#e9a23b", alpha=.20, label="Scan coil pair")
    for xx, name in [(HP2 * 1000, "HP2"), (HP1 * 1000, "HP1")]:
        core.axvline(xx, ls=":", lw=1.2, c="#536170")
        core.text(xx + 1.5, .96, name, va="top", transform=core.get_xaxis_transform(), fontsize=9)
    core.set(yscale="log", xlabel="Column z (mm)", ylabel="Beam D95 (µm)", title="Source images in the corrector and scan coils",
             xlim=(1120, 1420))
    core.legend(ncol=4, loc="lower center", fontsize=8)
    fig.suptitle("Probe formation with the 100 µm C2 aperture", fontsize=16)
    fig.savefig(HERE / "internal-optics.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()
    publish(args.candidate, args.extra)
