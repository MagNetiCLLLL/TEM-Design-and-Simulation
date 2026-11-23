"""Matched original-source comparisons at fixed current and finer nonlinear mesh."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from condenser_pair_study import EV, OUT
from condenser_source_fields import OUT as VACUUM_OUT, sha, write


def read(path):
    return json.loads(path.read_text())


def run():
    selected = "column-4096-0.05-7"
    cases = [
        (
            VACUUM_OUT / "gun-after-4096.npz",
            VACUUM_OUT / "section-after-4096-0.05.npz",
            "Vacuum windings  ·  −41 A / +19.5 A",
        ),
        (
            OUT / "matched-current/tip-4096.npz",
            OUT / "matched-current" / (selected + ".npz"),
            "Poles and returns  ·  −41 A / +19.5 A",
        ),
        (OUT / "tip-4096.npz", OUT / (selected + ".npz"), "Poles and returns  ·  −30 A / +16 A"),
    ]
    runs = []
    for i, (tip, archive, label) in enumerate(cases):
        with np.load(tip) as source, np.load(archive) as arr:
            if i == 0:
                ids, weights, launch = source["ids"], source["weight"], source["launch"]
            np.testing.assert_array_equal(source["ids"], ids)
            np.testing.assert_array_equal(source["weight"], weights)
            np.testing.assert_array_equal(source["launch"], launch)
            q = source["current"]
            if i < 2:
                np.testing.assert_array_equal(q[:8], [-41.0] * 4 + [19.5] * 4)
            if i == 0:
                fixed = q[8:].copy()
            np.testing.assert_array_equal(fixed, q[8:])
            z = arr["z"] * 1000 if i == 0 else arr["z_mm"]
            paths = arr["paths" if i == 0 else "paths_m"]
            rec = read(archive.with_suffix(".json"))
            record = rec["waists_mm" if i == 0 else "radial_waists_mm"]
            runs.append((label, z, paths, record, sha(tip), sha(archive)))
    fig, axs = plt.subplots(3, 1, figsize=(10.8, 7.5), sharex=True, sharey=True, layout="constrained")
    fig.suptitle(
        "Matched source trajectories  |  field change and current retuning", weight="bold", fontsize=15
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.84, 32))
    lim = max(np.max(abs(p[(z >= 450) & (z <= 1070), ::2, 0])) for _, z, p, *_ in runs) * 1.13e6
    for ax, (label, z, paths, record, _, _) in zip(axs, runs, strict=True):
        ok = (z >= 450) & (z <= 1070)
        for j, c in enumerate(colors):
            ax.plot(z[ok], paths[ok, 2 * j, 0] * 1e6, c=c, lw=0.7, alpha=0.8)
        for lo, hi, title in ((450, 550, "C1 lens"), (550, 750, "C2 lens"), (830, 1000, "C3 lens")):
            ax.axvspan(lo, hi, color="#829ba8", alpha=0.13, lw=0)
            ax.text(
                (lo + hi) / 2,
                0.98,
                title,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
            )
        for at in sum(record, []):
            ax.axvline(at, ls="--", lw=0.8, c="#a2402c")
            ax.text(
                at,
                0.025,
                f"{at:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                color="#a2402c",
                fontsize=8,
            )
        ax.set(xlim=(450, 1070), ylim=(-lim, lim), ylabel="x (µm)")
        ax.spines[["right", "top"]].set_visible(False)
        ax.axhline(0, c="#586871", lw=0.5, zorder=-3)
        ax.set_title(label, loc="left", fontsize=10)
    axs[-1].set_xlabel("Distance from the effective source  z (mm)")
    fig.savefig(EV / "condenser-ray-comparison.png", dpi=160)
    plt.close(fig)
    coarse = read(OUT / (selected + ".json"))
    refined = read(OUT / "mesh-check" / (selected + ".json"))
    np.testing.assert_allclose(
        sum(coarse["radial_waists_mm"], []), sum(refined["radial_waists_mm"], []), rtol=0, atol=0.12
    )
    data = dict(
        count=4096,
        matched_source=True,
        comparisons=[
            dict(case=label, waists_mm=roots, gun_sha256=gun_id, trajectory_sha256=trace_id)
            for label, _, _, roots, gun_id, trace_id in runs
        ],
        selected_1p5mm_550mm=dict(surface=coarse["surface"], waists_mm=coarse["radial_waists_mm"]),
        refined_1mm_800mm=dict(
            surface=refined["surface"],
            waists_mm=refined["radial_waists_mm"],
            field_key=refined["field_key"],
            gun_signature=refined["gun_signature"],
        ),
        refined_minus_tracking_D95_um=refined["surface"]["D95_um"] - coarse["surface"]["D95_um"],
    )
    write(EV / "physical-controls.json", data)
    print(
        "Fixed currents, material waists",
        runs[1][3],
        "refined D95",
        refined["surface"]["D95_um"],
        "difference",
        data["refined_minus_tracking_D95_um"],
        flush=True,
    )


if __name__ == "__main__":
    run()
