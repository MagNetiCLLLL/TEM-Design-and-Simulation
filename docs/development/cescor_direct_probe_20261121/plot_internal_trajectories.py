"""Replay the reported illumination on its original mesh and plot identified electron paths."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from column_current import FACTOR, source_digest
from observe import axis_crossings
from relay_transfer import FOIL, HERE, HP1, HP2, PARTS, RADIUS, START, SURFACE, TL21, TL22, WORK, Native, mechanical_span_mm


COLORS = ["#3265a8", "#17836b", "#ab710b", "#ca4a32"]
EDGES = np.array([0, .35, .65, .90, 1.0])


def select_tracks(incident, archive):
    xy = incident[:, 0] + 1j * incident[:, 1]
    fill = np.abs(xy) / RADIUS
    angle = np.angle(xy) % (2 * np.pi)
    live = archive["loss"] == -1
    indices, bands = [], []
    for j, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        pool = np.flatnonzero(live & (fill >= lo) & (fill <= hi if j == 3 else fill < hi))
        if len(pool) <= 12:
            chosen = pool.tolist()
        else:
            chosen = []
            for k in range(12):
                center = 2 * np.pi * (k + .5) / 12
                da = abs(np.angle(np.exp(1j * (angle[pool] - center))))
                sector = da < np.pi / 12
                if not sector.any():
                    continue
                distance = (da[sector] / (np.pi / 12))**2 + (
                    (fill[pool[sector]] - (hi + lo) / 2) / ((hi - lo) / 2)
                )**2
                chosen.append(pool[sector][np.argmin(distance)])
        indices.extend(chosen)
        bands.extend([j] * len(chosen))
    lost = np.flatnonzero(~live)
    indices.extend(lost)
    bands.extend([-1] * len(lost))
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate original particle in trajectory display")
    return np.asarray(indices), np.asarray(bands), fill


def capture(case, table, dest):
    record = json.loads((WORK / (case + ".json")).read_text())
    axial = json.loads((WORK / (case + "-axial.json")).read_text())
    pupil = record["relay"]["1-2"]["z_mm"][-1] * .001
    if sha256(table.read_bytes()).hexdigest() != record["field_sha256"]:
        raise ValueError("Field differs from the reported working point")
    # Use precisely the original observation planes to reproduce the RK4 mesh.
    model = Native(table, record["max_step_mm"], record["emitted"], record["objective_At"], [HP2, HP1, pupil])
    t, z = model.t, model.m.z
    with np.load(WORK / (case + ".npz")) as saved:
        archive = dict(saved)
    np.testing.assert_array_equal(archive["ids"], t.ids)
    np.testing.assert_array_equal(archive["weights"], t.weights)
    np.testing.assert_array_equal(archive["energy_offset_eV"], t.de)
    if source_digest(t) != record["source_digest"] or np.count_nonzero(t.weights > 0) < 1000:
        raise ValueError("The original finite source is required")
    field, (front,) = model.replay(record["currents_A"], FOIL[:1])
    selection, bands, fill = select_tracks(front[0], archive)
    display_z = np.r_[np.arange(START, SURFACE, .001), np.arange(1.12, 1.425, .00035),
                      np.arange(SURFACE - .001, SURFACE, .000005), FOIL, HP2, TL22, TL21, HP1, pupil, SURFACE]
    lost_indices = np.unique(archive["loss"][archive["loss"] >= 0])
    stops = np.unique(np.r_[z.searchsorted(display_z).clip(0, len(z) - 1), lost_indices, lost_indices + 1])
    selected_dev = cp.asarray(selection)
    n = len(t.ids)
    phase, energy, phi0 = cp.asarray(t.input).copy(), cp.asarray(model.en), cp.asarray(model.phi0)
    lost, peak, margin = cp.full(n, -1, np.int32), cp.zeros(n), cp.full(n, cp.inf)
    history = cp.empty((len(stops), len(selection), 4))
    first = 0
    for j, end in enumerate(stops):
        for start in range(first, end, 160):
            t.kernel(((n + 127) // 128,), (128,), (
                phase, energy, phi0, field, t.ep, t.er, np.int32(len(t.r2)), t.h, t.rad,
                lost, peak, margin, np.int32(start), np.int32(min(start + 160, end)), np.int32(n),
                t.rest, FACTOR, t.pc,
            ))
        history[j] = phase[selected_dev, :4]
        first = end
    final, loss = phase.get(), lost.get()
    np.testing.assert_array_equal(loss, archive["loss"])
    errors = np.max(np.abs(final - archive["phase"]), axis=0)
    np.testing.assert_allclose(final[:, :2], archive["phase"][:, :2], rtol=0, atol=1e-13)
    np.testing.assert_allclose(final[:, 2:4], archive["phase"][:, 2:4], rtol=0, atol=1e-12)
    paths = history.get()
    for k, i in enumerate(selection):
        if loss[i] >= 0:
            paths[stops > loss[i], k] = np.nan
    with np.load(WORK / (case + "-axial.npz")) as raw:
        zz = raw["z_m"]
        m = raw["common_moments"] if "common_moments" in raw else raw["moments"]
        c2, c3, adl = np.array([mechanical_span_mm(key) for key in (
            'condenser_lens_2', 'condenser_lens_3', 'adapter_lens')]) * .001
        waists = {name: axis_crossings(zz, m, low, high) for name, (low, high) in {
            "C2 exit section": (model.m.intervals[0], c2[1]), "C2-C3": (c2[1], c3[0]),
            "C3-ADL": (c3[1], adl[0]),
            "TL22-TL21": (TL22, TL21), "HP1-to-scan": (HP1, 1.3875),
            "scan coils": (1.3875, 1.4125), "Mini-to-presample": (model.m.intervals[-1], SURFACE - .00025),
        }.items()}
    summary = dict(case=case, emitted=n, survivors=int(np.sum(loss == -1)), plotted=len(selection),
                   plotted_survivors=int(np.sum(bands >= 0)), diagnostic_rim_rays_plotted=0,
                   selection="Actual C2 entrance radius bands, up to 12 azimuthal sectors per band, plus all intercepted IDs",
                   source_sha256=record["source_digest"], field_sha256=record["field_sha256"],
                   max_abs_replay_error_x_y_m_tx_ty_time_s=errors.tolist(),
                   specimen_z_mm=SURFACE * 1000, source_exit_z_mm=START * 1000,
                   specimen=record["specimen"], rim_mrad=record["rim_azimuth_check"]["angle_mrad_min_median_max"],
                   C2_front_rear_mm=(1000 * FOIL).tolist(), loss_source_ids=t.ids[loss >= 0].tolist(),
                   stopped_interval_mm=[[float(z[i] * 1000), float(z[i + 1] * 1000)] for i in lost_indices],
                   waist_brackets_mm=waists, specimen_D95_focus_bracket_nm=axial["D95_minimum_bracket_nm"],
                   objective_At=record["objective_At"], round_lens_currents_A=record["currents_A"],
                   corrector_currents_A=json.loads(str(np.load(table)["report"]))["currents_A"])
    np.savez_compressed(dest, z_m=z[stops], phase_x_y_m_tx_ty=paths, source_id=t.ids[selection],
                        source_weight=t.weights[selection], energy_offset_eV=t.de[selection],
                        C2_entrance_radius_fraction=fill[selection], band=bands, loss_index=loss[selection],
                        summary=json.dumps(summary, allow_nan=False))
    return summary


def draw_paths(ax, zz, values, bands):
    for j, color in enumerate(COLORS):
        mask = bands == j
        ax.plot(zz, values[:, mask], color=color, alpha=.54 if j < 3 else .78, lw=.65, rasterized=True)
    for path in values[:, bands < 0].T:
        valid = np.isfinite(path)
        ax.plot(zz[valid], path[valid], "--", color="#2e2443", lw=1.1)
        if valid.any() and not valid.all():
            ax.plot(zz[valid][-1], path[valid][-1], marker="x", ms=6, color="#2e2443", mew=1.2)
    ax.grid(axis="y", lw=.45, alpha=.24)
    ax.margins(x=0, y=.10)
    ax.spines[["right", "top"]].set_visible(False)


def guides(axes, lines, *, height=.97):
    for z, label in lines:
        for ax in axes:
            ax.axvline(z, color="#5e6572", lw=.55, ls=":", zorder=0)
        if label:
            axes[0].text(z, height, label, ha="center", va="top", rotation=90, fontsize=8,
                         transform=axes[0].get_xaxis_transform(), color="#263447",
                         bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=1.2))


def plot(archive, out):
    with np.load(archive) as f:
        z, p, bands = f["z_m"] * 1e3, f["phase_x_y_m_tx_ty"][:, :, :2] * 1e6, f["band"]
        s = json.loads(str(f["summary"]))
    objective_pole = START * 1000 + PARTS["objective_upper_pole"]["local_center_z_mm"]
    markers = [(START * 1000, "Gun exit"), (500, "C1"), (650, "C2"), (765, "C2 aperture"), (915, "C3"), (1088.2, "ADL"),
               (HP2 * 1000, "HP2"), (TL22 * 1000, "TL22"), (TL21 * 1000, "TL21"), (HP1 * 1000, "HP1"),
               (1366.3, "TL12"), (1400, "Scan pair"), (1522.5, "Mini"), (objective_pole, "Upper objective pole"),
               (SURFACE * 1000, "Specimen")]
    handles = [Line2D([], [], color=c, lw=2, label=f"{l:.2f}–{h:.2f} R")
               for c, l, h in zip(COLORS, EDGES[:-1], EDGES[1:])]
    handles.append(Line2D([], [], color="#2e2443", ls="--", marker="x", label="C2 interception"))
    fig, axes = plt.subplots(3, 1, figsize=(14, 8.7), sharex=True, layout="constrained",
                             gridspec_kw=dict(height_ratios=[1, 1, .88]))
    for k, ax in enumerate(axes[:2]):
        draw_paths(ax, z, p[:, :, k] / 1000, bands)
        ax.set_ylabel(("x" if k == 0 else "y") + " (mm)")
        ax.axhline(0, lw=.5, color="#717171", zorder=0)
    rad = np.hypot(p[:, :, 0], p[:, :, 1])
    draw_paths(axes[2], z, rad, bands)
    axes[2].set(yscale="log", ylabel="Radius from axis (µm)", xlabel="Column z (mm)")
    guides(axes, markers)
    for interval, label in [("C2 exit section", "Inside C2 lens"), ("C2-C3", "C2–C3 waist"), ("C3-ADL", "Extra waist")]:
        waists = s["waist_brackets_mm"].get(interval, [[], []])[0]
        for bracket in waists:
            mid = np.mean(bracket)
            axes[2].axvline(mid, lw=.8, ls="--", color="#8d2862", zorder=0)
            axes[2].text(mid + 7, .06, label, color="#7b2457", fontsize=8,
                         transform=axes[2].get_xaxis_transform())
    axes[2].set_xlim(z.min(), z.max())
    fig.suptitle(f"Electron trajectories from gun exit to specimen  |  D95 = {s['specimen']['D95_nm']/1000:.3f} µm",
                 fontsize=17)
    axes[0].legend(handles=handles, loc="lower left", ncol=5, title="Entry radius at the C2 aperture (R = 50 µm)",
                   fontsize=8, title_fontsize=9, framealpha=.94)
    fig.savefig(out / "last-point-trajectories.png", dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14.1, 6.3), layout="constrained", sharex="col")
    windows = [(913, 1090), (1137, 1421), (SURFACE * 1000 - .30, SURFACE * 1000)]
    titles = ["C3 to adapter lens", "Corrector and scan coils", "Final 300 µm to the specimen"]
    for j, ((low, high), title) in enumerate(zip(windows, titles)):
        keep = (z >= low) & (z <= high)
        xx = (z[keep] - SURFACE * 1000) * 1000 if j == 2 else z[keep]
        for k in range(2):
            ax = axes[k, j]
            draw_paths(ax, xx, p[keep, :, k], bands)
            ax.axhline(0, lw=.5, color="#717171", zorder=0)
            ax.set_ylabel(("x" if k == 0 else "y") + " (µm)")
            if j == 2:
                ax.axvline(0, color="#151b23", lw=1, clip_on=False)
        axes[0, j].set_title(title, fontsize=12, pad=8)
        axes[1, j].set_xlabel("Distance to upper surface (µm)" if j == 2 else "Column z (mm)")
        if j == 0:
            bracket = s["waist_brackets_mm"]["C3-ADL"][0][0]
            for ax in axes[:, j]:
                ax.axvline(np.mean(bracket), ls="--", lw=1.1, color="#8d2862")
            guides(axes[:, j], [(np.mean(bracket), "Extra waist")])
        elif j == 1:
            guides(axes[:, j], [v for v in markers if low < v[0] < high])
    fig.suptitle("The same original electrons in three sections of the column", fontsize=16)
    fig.savefig(out / "last-point-trajectories-detail.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="internal-waist-1171-0-2")
    parser.add_argument("--table", type=Path, default=WORK / "m330-field.npz")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    file = HERE / "evidence/last-point-trajectories.npz"
    if not args.render_only:
        print(json.dumps(capture(args.case, args.table, file), indent=2), flush=True)
    plot(file, HERE)
