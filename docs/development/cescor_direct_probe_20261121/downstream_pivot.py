"""Match the TL12 finite-source waist and the driven AC tilt at the two-lens front focus."""

import argparse
import gzip
import json
from hashlib import sha256

import cupy as cp
import numpy as np
from scipy.optimize import brentq

from column_current import ACTIVE, OWNER, POINT, source_digest
from condenser_normal import FIELD
from relay_transfer import HERE, PARTS, START, SURFACE, WORK, Native, advance
from scan_waist_audit import EVIDENCE, archived_scan


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inputs():
    rec = json.loads((EVIDENCE / "illuminated-normal.json").read_text())["final"]
    focal = json.loads((EVIDENCE / "scan-waist-audit.json").read_text())
    if sha256(FIELD.read_bytes()).hexdigest() != rec["corrector_sha256"] or rec["objective_At"] != focal["objective_At"]:
        raise ValueError("Reference fields do not match the focal-plane audit")
    (focus_mm,) = focal["signed_root_candidates_mm"]["mini_and_objective"]
    return rec, focus_mm * .001


def scaled(drives, gain):
    q = np.array(drives, dtype=float)
    q[OWNER[ACTIVE] == "probe_tl12_lens"] *= gain
    return q


def moment(phase, weights):
    x = phase[:, :4] - np.average(phase[:, :4], weights=weights, axis=0)
    cov = (x.T * weights) @ x / weights.sum()
    cross = cov[:2, 2:]
    return np.r_[np.sqrt(np.trace(cov[:2, :2])), np.trace(cross),
                 np.linalg.eigvalsh((cross + cross.T) / 2)]


def static_study():
    reference, focus = inputs()
    # The 1024-source sweep selected the weaker TL12 branch with no earlier output waist.
    # Solve first on 4096, then on the complete source at the finer integration step.
    trace = np.unique(np.r_[np.arange(1352., 1512.6, .5) * .001,
                            focus + np.arange(-4, 5) * .000025])
    results = []
    for count, step in ((4096, .15), (32768, .075)):
        m = Native(FIELD, step, count, reference["objective_At"],
                   np.r_[trace, SURFACE])
        def covariance(gain):
            _, ((p, loss),) = m.replay(scaled(reference["currents_A"], gain), [focus])
            return moment(p[loss == -1], m.t.weights[loss == -1])[1] * 1e9

        gain = brentq(covariance, .29, .32, xtol=5e-12)
        q = scaled(reference["currents_A"], gain)
        entry = dict(emitted=count, max_step_mm=step, TL12_gain=gain,
                     TL12_winding_A=q[OWNER[ACTIVE] == "probe_tl12_lens"].tolist())
        if count == 32768:
            if source_digest(m.t) != reference["source_digest"]:
                raise ValueError("Gun-exit checkpoint changed")
            for name, controls in (("reference", reference["currents_A"]), ("matched", q)):
                field = m.field(controls)
                first = 0
                phase, losses = m.t.input, m.loss
                moments = []
                before = None
                for idx in m.indices(trace):
                    (phase, losses), = advance(m.t, field, phase, m.en, m.phi0, losses, first, [idx])
                    live = losses == -1
                    if before is not None and not np.array_equal(before, live):
                        raise ValueError("Finite-source membership changed across the waist measurement")
                    before = live.copy()
                    moments.append(moment(phase[live], m.t.weights[live]))
                    first = idx
                moments = np.stack(moments)
                brackets = []
                for axis in range(1, 4):
                    i = np.where((moments[:-1, axis] < 0) & (moments[1:, axis] >= 0))[0]
                    brackets.append((1000 * np.column_stack((trace[i], trace[i+1]))).tolist())
                metric, _, _ = m.measure(controls)
                entry[name] = dict(
                    radial_and_principal_waist_brackets_mm=brackets,
                    rms_radius_at_focal_plane_um=1e6 * moments[np.searchsorted(trace, focus), 0],
                    emitted=count, surviving=int(live.sum()), effective=float(m.t.weights[live].sum()**2
                        / np.sum(m.t.weights[live]**2)), source_digest=source_digest(m.t),
                    specimen=metric["specimen"], specimen_edge_mrad=metric["rim"]["angle_mrad_min_median_max"],
                    rim_survived=metric["rim"]["reached_sample"], specimen_survived=metric["survived"],
                    current_ledger=metric["current_ledger"])
                np.savez(WORK / f"downstream-{name}-rms.npz", z_mm=trace*1e3, moments=moments)
        results.append(entry)
        print("TL12", count, gain, flush=True)
        del m
        cp.get_default_memory_pool().free_all_blocks()
    write(EVIDENCE / "downstream-waist.json", dict(reference="illuminated-normal.json / final",
          focal_plane_mm=focus*1e3, zero_command=archived_scan(), frozen_gun_exit_mm=START*1e3,
          corrector_sha256=reference["corrector_sha256"], objective_At=reference["objective_At"],
          specimen_surface_mm=SURFACE*1e3, runs=results))


def archived_mixing():
    nodes = json.loads(gzip.decompress((POINT / "inputs.json.gz").read_bytes()))["graph"]["nodes"]
    (ac,) = [row["attributes"] for row in nodes if row.get("type", "").endswith(":AcDeflectorComponent")]
    ratio = np.array([v["tuple"] for v in ac["_pure_shift_lower_ratio_matrix"]["tuple"]])
    ratio += np.diag([ac["pivot_offset_x"], ac["pivot_offset_y"]])
    return ac["upper_coil_gain"] * np.vstack((np.eye(2), ratio))


class DrivenTrace:
    def __init__(self, model, axial_subdivision=True):
        from scan_winding_basis import grid, axial_grid, expand
        self.model = model
        z, xy = grid()
        self.receipt = {}
        centers = .001 * np.array(PARTS["ac_deflector"]["interaction_centers_local_z_mm"])
        target_z = axial_grid(z, centers) if axial_subdivision else z
        with np.load(WORK / "downstream-scan-windings.npz") as data:
            digest = sha256(json.dumps(PARTS["ac_deflector"], sort_keys=True).encode()).hexdigest()
            if data["winding_sha256"].item() != digest:
                raise ValueError("AC geometry or current matrix changed")
            self.current_matrix = data["command_matrix_A_per_mrad"]
            self.receipt = dict(winding_sha256=digest, command_matrix_A_per_mrad=self.current_matrix.tolist(),
                field_check_count_per_circuit=int(data["checks_per_winding"]),
                maximum_relative_field_check_per_circuit=data["max_check_relative"].tolist())
            self.basis = [cp.asarray(expand(values, data["coarse_z"], data["coarse_xy"], target_z, xy))
                          for values in data["basis_T_per_A"]]
        tracker = model.t
        original = tracker.tables[0]
        ix = np.clip(np.searchsorted(z, target_z, side="right")-1, 0, len(z)-2)
        fraction = cp.asarray((target_z-z[ix]) / (z[ix+1]-z[ix]))[:, None, None, None]
        self.static_grid = original[ix] + fraction*(original[ix+1]-original[ix])
        nodes = model.m.nodes - START
        iz = np.clip(np.searchsorted(target_z, nodes, side="right")-1, 0, len(target_z)-2)
        self.grid = self.static_grid.copy()
        tracker.tables = [self.grid, cp.asarray(iz, dtype=cp.int32),
                          cp.asarray((nodes-target_z[iz]) / (target_z[iz+1]-target_z[iz]))]
        for symbol, array in zip(("cescor_b", "cescor_iz", "cescor_fz"), tracker.tables, strict=True):
            pointer = tracker.module.get_global(symbol)
            cp.ndarray((1,), dtype=cp.uint64, memptr=pointer).set(np.array([array.data.ptr], np.uint64))

    def means(self, field, station_urad, planes, expected=None):
        amps = self.current_matrix @ (station_urad * .001)
        self.grid[...] = self.static_grid
        for amp, basis in zip(amps, self.basis, strict=True):
            if amp:
                self.grid += amp*basis
        m = self.model
        states = advance(m.t, field, m.t.input, m.en, m.phi0, m.loss, 0, m.indices(planes))
        means, masks = [], []
        for i, (p, loss) in enumerate(states):
            live = loss == -1
            if expected is not None and not np.array_equal(live, expected[i]):
                raise ValueError("A scan perturbation changed the observed particle identities")
            means.append(np.average(p[live, :4], weights=m.t.weights[live], axis=0))
            masks.append(live)
        return np.array(means), masks

    def response(self, field, mixing, planes, amplitude):
        center, masks = self.means(field, np.zeros(4), planes)
        odd, even = [], []
        for direction in mixing.T:
            plus = self.means(field, amplitude*direction, planes, masks)[0]
            minus = self.means(field, -amplitude*direction, planes, masks)[0]
            odd.append((plus-minus)/(2*amplitude))
            even.append((plus+minus)/2-center)
        return np.stack(odd, axis=-1), np.stack(even, axis=-1), masks


def driven_study():
    reference, focus = inputs()
    archived_scan()
    gain = json.loads((EVIDENCE / "downstream-waist.json").read_text())["runs"][-1]["TL12_gain"]
    initial = archived_mixing()
    # Off-plane measurements bracket a pivot found with the driven beam.
    stations = np.unique(np.r_[np.arange(1360., 1513., 2.)*.001, focus,
                                focus + np.array([-.025, .025])*.001, SURFACE])
    target = np.searchsorted(stations, focus)
    report = dict(reference="illuminated-normal.json / final", TL12_gain=gain, focal_plane_mm=focus*1e3,
        lower_coil_exit_mm=START*1e3 + PARTS["ac_deflector"]["local_end_z_mm"],
        angle_convention="Mechanical paraxial slopes in the laboratory frame. P and T differentiate the original-current centroid with respect to two common scan commands.",
        input_command_units="Nominal station urad converted to amperes with the four-channel saved winding matrix.",
        runs=[])
    fitted = None
    for count, step, fine in ((4096, .15, False), (32768, .075, True)):
        m = Native(FIELD, step, count, reference["objective_At"], stations)
        trace = DrivenTrace(m, fine)
        q = scaled(reference["currents_A"], gain)
        field = m.field(q)
        basis, _, _ = trace.response(field, np.eye(4), [focus], 1.)
        pos = basis[0, :2]
        if np.linalg.cond(pos[:, 2:]) > 1e5:
            raise ValueError("Lower scan channels cannot control both target coordinates")
        ratio = -np.linalg.solve(pos[:, 2:], pos[:, :2])
        candidate = np.vstack((initial[:2], ratio @ initial[:2]))
        trial = dict(emitted=count, step_mm=step, refined_winding_z=fine, source_digest=source_digest(m.t),
                     lower_to_upper=ratio.tolist(), winding_field_checks=trace.receipt, observations={})
        comparisons = [("matched", field, candidate, 1.)]
        if fitted is not None:
            comparisons += [("reference", m.field(reference["currents_A"]), initial, 1.),
                            ("original_drive_after_TL12", field, initial, 1.),
                            ("coarse_calibration_replayed", field, fitted, 1.),
                            ("matched_half", field, candidate, .5)]
        for name, f, commands, amplitude in comparisons:
            response, even, masks = trace.response(f, commands, stations, amplitude)
            p, t = response[:, :2]*1e9, response[:, 2:]*1e6
            psv, tsv = (np.linalg.svd(a, compute_uv=False) for a in (p,t))
            trial["observations"][name] = dict(command_amplitude_urad=amplitude,
                original_counts_at_target_and_sample=[int(masks[i].sum()) for i in (target, -1)],
                target_ID_sha256=sha256(m.t.ids[masks[target]].tobytes()).hexdigest(),
                station_command_matrix=commands.tolist(),
                physical_coil_A_per_common_command_urad=(trace.current_matrix @ (commands*.001)).tolist(),
                target_position_nm_per_urad=p[target].tolist(),
                target_tilt_urad_per_urad=t[target].tolist(), target_position_singular_nm_per_urad=psv[target].tolist(),
                target_tilt_singular_urad_per_urad=tsv[target].tolist(),
                target_even_position_norm_nm=float(np.linalg.norm(even[target,:2])*1e9),
                neighboring_max_position_response_nm_per_urad=psv[[target-1,target+1],0].tolist(),
                specimen_shift_singular_nm_per_urad=psv[-1].tolist(),
                specimen_mechanical_tilt_singular_urad_per_urad=tsv[-1].tolist())
            if fine:
                np.savez(WORK / f"downstream-{name}-response.npz", z_mm=stations*1e3,
                         P_nm_per_urad=p,T_urad_per_urad=t)
            print(count, name, "position / tilt singular values", psv[target], tsv[target], flush=True)
        fitted = candidate
        report["runs"].append(trial)
        del m, trace, field
        cp.get_default_memory_pool().free_all_blocks()
    write(EVIDENCE / "downstream-pivot.json", report)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    data = json.loads((EVIDENCE / "downstream-pivot.json").read_text())
    focus = data["focal_plane_mm"]
    static = json.loads((EVIDENCE / "downstream-waist.json").read_text())["runs"][-1]
    waist_radius = static["matched"]["rms_radius_at_focal_plane_um"]
    fig, (waist, pivot) = plt.subplots(1, 2, figsize=(11.8, 4.4), layout="constrained")
    before, after, unmixed = "#4b5766", "#00766e", "#bf5c13"
    for ax, title in zip((waist, pivot), ("Static beam · scan off", "Driven centroid · X and Y scan")):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=13, pad=13)
        ax.axvspan(1380., 1395., color="#ccd7e1", alpha=.48, lw=0)
        ax.axvspan(1405., 1420., color="#ccd7e1", alpha=.48, lw=0)
        ax.axvline(focus, color="#00766e", ls=":", lw=1.35)
        ax.set(xlim=(1352, 1512), xlabel="Column z (mm)", xticks=[1360, 1400, 1440, 1480, 1510])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        for z, name in ((1387.5, "AC 1"), (1412.5, "AC 2")):
            ax.text(z, .98, name, ha="center", va="top", fontsize=8, color="#475766",
                    transform=ax.get_xaxis_transform())
    for case, label, color in (("reference","Original TL12",before), ("matched","Retuned TL12",after)):
        with np.load(WORK / f"downstream-{case}-rms.npz") as raw:
            waist.plot(raw["z_mm"], raw["moments"][:,0]*1e6, label=label, lw=1.9, color=color)
    waist.set(yscale="log", ylabel="Current-weighted RMS radius (µm)", yticks=[1, 10, 100, 1000], ylim=(.8, 1700))
    waist.yaxis.set_major_formatter(ScalarFormatter())
    waist.scatter([focus], [waist_radius], s=19, color=after, zorder=3)
    waist.annotate(f"{focus:.2f} mm", xy=(focus,waist_radius), xytext=(1447,2.5),
                   color=after, fontsize=9, arrowprops=dict(arrowstyle="-", color=after))
    waist.legend(loc="upper left", frameon=False, bbox_to_anchor=(0, .89))
    for case, label, color, style in (("reference","Original setting",before,"-"),
            ("original_drive_after_TL12","TL12 only",unmixed,"--"), ("matched","TL12 + AC mixing",after,"-")):
        with np.load(WORK / f"downstream-{case}-response.npz") as raw:
            y = np.linalg.svd(raw["P_nm_per_urad"], compute_uv=False)[:,0]
            pivot.plot(raw["z_mm"], y, label=label, color=color, ls=style, lw=1.9)
    pivot.set(ylabel="Maximum centroid motion (nm per µrad command)", ylim=(-1.4, 39))
    pivot.scatter([focus],[0],s=19,color=after,zorder=3)
    pivot.annotate("Common tilt pivot", xy=(focus,0), xytext=(1462,8.5), color=after, fontsize=9,
                   arrowprops=dict(arrowstyle="-", color=after))
    pivot.legend(loc="upper left", frameon=False, bbox_to_anchor=(0, .89))
    fig.savefig(HERE / "downstream-pivot.png", dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["static", "driven", "figure"])
    args = parser.parse_args()
    if args.operation == "static":
        static_study()
    elif args.operation == "driven":
        driven_study()
    else:
        figure()
