"""Archive the measured entrance and scan currents and plot the matched comparison."""

from hashlib import sha256
import json
from shutil import copyfile
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solve_fields import CACHE, HERE, ROOT
from column import GROUPS, POINT, metrics
sys.path.insert(0, str(ROOT / "scripts"))
import scan_joint_probe_si110 as screen


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_scans(result):
    fig = plt.figure(figsize=(9.1, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, .055], height_ratios=[1.65, .82])
    profile = fig.add_subplot(grid[1, :2])
    palette = ("#506a8c", "#ac4b1e")
    maps = []
    for key in ("before", "after"):
        with np.load(result / key / "scan.npz") as data:
            coords = data["coords_nm"]
            maps.append(data["channel_current_a"][1] * 1e9)
    vmin, vmax = min(np.min(a) for a in maps), max(np.max(a) for a in maps)
    pitch = coords[1]-coords[0]
    bounds = [coords[0] - pitch/2, coords[-1] + pitch/2] * 2
    ix = np.searchsorted(coords, 0)
    for i, (name, data) in enumerate(zip(("Before", "After"), maps)):
        ax = fig.add_subplot(grid[0, i])
        im = ax.imshow(data, extent=bounds, origin="lower", cmap="cividis", interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.plot([coords[ix]] * 2, [-.25, .13], color="white", lw=1, ls=":")
        spec = read(result / name.lower() / "metrics.json")["surface"]
        ax.set(title=f"{name}   ·   D95 {spec['D95_nm'] / 1000:.2f} µm", xlabel="Stage x (nm)",
               ylabel="Stage y (nm)" if i == 0 else "")
        keep = (coords >= -.25) & (coords <= .13)
        profile.plot(coords[keep], data[keep, ix], "o-", ms=3, color=palette[i], lw=1.6, label=name)
    assert np.allclose(np.diff(coords), pitch)
    fig.colorbar(im, cax=fig.add_subplot(grid[0, 2]), label="Annular dark-field current (nA)")
    repeat, basis, _ = screen.lattice()
    same_x = basis[np.isclose(basis[:, 0], 0), 1]
    pair = np.sort((same_x + .5 * repeat[1]) % repeat[1] - .5 * repeat[1])
    for i, site in enumerate(pair):
        profile.axvline(site, color="#666666", ls="--", lw=1, label=None if i else "Si columns")
    profile.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    profile.set(xlabel=f"Stage y at x = {coords[ix]:.2f} nm", ylabel="Dark-field current (nA)")
    profile.grid(axis="y", alpha=.22)
    fig.suptitle("Si [110]  ·  5 nm silicon  ·  50–150 mrad", fontsize=15)
    fig.savefig(HERE / "si110-comparison.png", dpi=190)
    plt.close(fig)


def collect():
    result = HERE / "evidence"
    result.mkdir(exist_ok=True)
    lens_order = list(GROUPS[:3]) + ["adapter_lens", "probe_tl22_lens", "probe_tl21_lens",
                                       "probe_tl12_lens", "condenser_lens_1"]
    parked = read(POINT / "currents.json")
    sections = {}
    for winding in parked["recipes"]["objective_lens"]["windings"].values():
        for group in winding:
            section = group["interval_index"]
            sections[section] = sections.get(section, 0) + group["turn_count"]
    turns = [sections[key] for key in sorted(sections)]
    study = dict(fields={p.stem: read(p) for p in sorted(CACHE.glob("armco-*.json"))},
                 trials={}, currents={}, source_prefixes={})
    for path in sorted(CACHE.glob("*/metrics.json")):
        m = read(path)
        study["trials"][path.parent.name] = {k: m[k] for k in ("objective_ampere_turns", "hexapole_scale",
            "controls", "max_step_mm", "field_digest", "surface", "D95_drift_proposal_nm", "surviving_count")}
    for key in ("before", "after"):
        acquisition = key + "-local"
        target = result / key
        target.mkdir(exist_ok=True)
        for name in ("surface.npz", "metrics.json"):
            copyfile(CACHE / acquisition / name, target / name)
        for name in ("scan.npz", "report.json"):
            copyfile(CACHE / (acquisition + "-scan") / name, target / name)
        m, scan = read(target / "metrics.json"), read(target / "report.json")
        assert m["input_digest"] == scan["input_digest"]
        assert m["surviving_count"] == scan["surviving"]
        with np.load(target / "surface.npz") as src:
            phase = np.column_stack([src[f] for f in ("x_m", "y_m", "tx_rad", "ty_rad")])
            rows = []
            for count in (1024, 2048, 8192, 16384, 32768):
                original = src["ray_id"] < count
                live = original & src["alive"] & (src["weight"] > 0)
                values = metrics(phase[live], src["weight"][live], float(src["emitted_current_a"]))
                rows.append(dict(original=count, surviving=int(live.sum()), **values,
                                 transmission_fraction=float(src["weight"][live].sum()/src["weight"][original].sum())))
            study["source_prefixes"][key] = rows
        multiplier = dict(zip(lens_order, m["controls"], strict=True))
        groups = {}
        for lens, current in zip(parked["owner_per_group"], parked["current_a"], strict=True):
            if lens != "objective_lens":
                groups.setdefault(lens, []).append(current * multiplier.get(lens, 1.))
        study["currents"][key] = dict(round_lens_groups_A=groups,
            objective_uniform_series_A=m["objective_ampere_turns"]/sum(turns), turns_per_objective_section=turns)
        field_name, = [k for k, v in study["fields"].items() if v["field_digest"] == m["field_digest"]]
        copyfile(CACHE / (field_name + ".npz"), target / "objective-field.npz")
    for name in ("focus.json", "focus.npz"):
        copyfile(CACHE / "after-local" / name, result / "after" / name)
    alternate = result / "alternate-mesh"
    alternate.mkdir(exist_ok=True)
    for key in ("before", "after"):
        for name in ("scan.npz", "report.json"):
            copyfile(CACHE / (key + "-scan") / name, alternate / (key + "-" + name))
    (result / "study.json").write_text(json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_scans(result)
    files = sorted(p for p in result.rglob("*") if p.is_file() and p.name != "checksums.json")
    hashes = {p.relative_to(result).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}
    (result / "checksums.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    collect()
