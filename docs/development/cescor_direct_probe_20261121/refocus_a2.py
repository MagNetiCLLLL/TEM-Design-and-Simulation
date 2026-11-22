"""Refocus the A2-adjusted column using only Objective and Mini Condenser excitation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from adjust_a2 import OUT, read_json
from column_current import SURFACE, WORK, current_ledger, field_array, metrics, shared_inputs, source_digest, source_rate
from finite_probe import axial, bundle
from fit_threefold import EVIDENCE, design, fit_record, read_sample, ring_measure
from tune import waist_proposal


FOCUS_OUT = OUT / "focus"


def inputs(table_path):
    reference = read_json(EVIDENCE / "a2-adjusted-working-point.json")
    winding = reference["round_windings"]
    q = np.array(winding["currents_A"])
    mini = np.asarray(winding["owner_per_group"]) == "mini_condenser"
    if mini.sum() != 3 or sum(~mini) != 18:
        raise ValueError("The reference winding layout has changed")
    with np.load(table_path, allow_pickle=False) as f:
        table_report = json.loads(str(f["report"]))
    native = reference["after"]["field_sha256"]
    known = {v["field_sha256"] for v in reference["controls"].values()} | {native}
    if (sha256(table_path.read_bytes()).hexdigest() not in known or
            table_report["currents_A"] != reference["corrector_winding_currents_A"]):
        raise ValueError("The field table does not describe the fixed A2-adjusted corrector")
    return reference, q, mini


def objective_receipt(ni):
    if not np.isfinite(ni) or ni >= 0 or round(abs(ni), 4) != abs(ni):
        raise ValueError("Expected negative objective excitation on the archived 0.0001 ampere-turn grid")
    stem = f"armco-{abs(ni):.12g}-gap-fine-p8"
    path = WORK / "column" / (stem + ".json")
    if not path.is_file() or not path.with_suffix(".npz").is_file():
        shared_inputs(ni)
    data = read_json(path)
    base = read_json(WORK / "column/armco-24000-gap-fine-p8.json")
    if data["ampere_turns"] != abs(ni) or any(
        data[key] != base[key] for key in ("curve_sha256", "installed_geometry_digest", "level", "padding", "shape")
    ):
        raise ValueError("The objective solution does not match the current, material, geometry or grid")
    return data


def search(args):
    ref, q0, mini = inputs(args.table)
    with np.load(EVIDENCE / ref["trajectory_file"]) as archive:
        obs = archive["z_m"]
    ni0 = ref["round_windings"]["objective_At"]
    rate = source_rate()
    mesh, tracker, background, coils = bundle(args.step, obs, ni0, args.count, rate, args.table)
    if args.count == ref["after"]["emitted"] and source_digest(tracker) != ref["after"]["source_digest"]:
        raise ValueError("Gun-exit particles differ from the fixed A2 reference")
    mesh.saved = np.array([np.flatnonzero(mesh.z == SURFACE).item()])
    # The Mini Condenser retains its installed three-group winding pattern.
    # Its single physical gain multiplies every signed baseline group equally.
    fixed = field_array(mesh, background, coils, q0)
    mini_field = np.einsum("i,ijk->jk", q0[mini], coils[mini])
    objectives = {ni0: background}
    proposals = {}
    journal = FOCUS_OUT / (args.name + ".json")

    def evaluate(ni, gain):
        ni, gain = round(float(ni), 1), float(gain)
        if not (0.97 <= gain <= 1.03 and abs(ni - ni0) <= 1200):
            raise ValueError("Proposed setting lies outside the local two-lens search")
        if (ni, gain) in proposals:
            return proposals[ni, gain]
        if ni not in objectives:
            objective_receipt(ni)
            objectives[ni] = mesh.base("armco", ni, "gap-fine", 8)
        field = fixed.copy()
        field[:, :background.shape[1]] += (objectives[ni] - background) + (gain - 1) * mini_field
        phase, loss, _, _ = tracker.run(field, keep_history=False)
        live = loss == -1
        crossings, weights = phase[-1, live], tracker.weights[live]
        d, r = waist_proposal(crossings, weights)
        row = dict(objective_At=ni, mini_gain=gain, surface=metrics(crossings, weights, rate),
                   drift_D95_um=d * 1e6, drift_RMS_um=r * 1e6,
                   current_ledger=current_ledger(mesh, tracker, loss), emitted=len(loss), survived=int(live.sum()))
        proposals[ni, gain] = row
        journal.write_text(json.dumps(dict(source_digest=source_digest(tracker), step_mm=args.step,
                                      trials=list(proposals.values())), indent=2) + "\n", encoding="utf-8")
        print(f"{ni:9.1f} At  Mini x {gain:.10f}  D95 {row['surface']['D95_nm']*.001:.7f} um  "
              f"waist {d*1e6:.5f} um  alpha {row['surface']['alpha95_mrad']:.7f} mrad", flush=True)
        return row

    def residual(row):
        return np.array([row["drift_D95_um"], row["surface"]["alpha95_mrad"] - 30])

    start = np.array([ni0, 1.])
    row = evaluate(*start)
    if (args.count == ref["after"]["emitted"] and args.step == ref["after"]["max_step_mm"]
            and sha256(args.table.read_bytes()).hexdigest() == ref["after"]["field_sha256"]):
        for key in ("D95_nm", "alpha95_mrad", "transmitted_uA"):
            np.testing.assert_allclose(row["surface"][key], ref["after"]["surface"][key], rtol=2e-7)
    step = np.array([-40., .0001])
    slopes = np.column_stack([(residual(evaluate(*(start + v))) - residual(row))/v[i]
                              for i, v in enumerate(np.diag(step))])
    x = start
    for _ in range(8):
        error = residual(row)
        if abs(error[0]) < .6 and abs(error[1]) < .001:
            break
        correction = np.linalg.solve(slopes, -error)
        correction *= min(1., 400./max(abs(correction[0]), 1.e-12), .004/max(abs(correction[1]), 1.e-12))
        trial = x + correction
        trial[0] = round(trial[0], 1)
        candidate = evaluate(*trial)
        change = trial - x
        if not np.any(change):
            break
        # Update in units of the independently measured excitation steps.
        y = change/step
        slopes += np.outer(residual(candidate) - error - slopes @ change, y/step) / (y @ y)
        x, row = trial, candidate
    # The FEM cache resolves 0.1 At. At fixed objective current the continuous
    # Mini drive can trim the percentile waist without rounding either field.
    ni, centre = x
    delta = max(2e-6, 2*abs(row["drift_D95_um"]/slopes[0, 1]))

    def focus_error(gain):
        return evaluate(ni, gain)["drift_D95_um"]

    for factor in (1, 3, 10):
        lo, hi = centre - factor*delta, centre + factor*delta
        if focus_error(lo)*focus_error(hi) <= 0:
            gain = brentq(focus_error, lo, hi, xtol=1e-10)
            selected = evaluate(ni, gain)
            journal.with_name(args.name + "-selected.json").write_text(
                json.dumps(selected, indent=2) + "\n", encoding="utf-8")
            return
    raise ValueError("The measured Mini Condenser response did not bracket surface focus")


def verify(args):
    ref, q, mini = inputs(args.table)
    q[mini] *= args.gain
    objective = objective_receipt(args.ni)
    with np.load(WORK / "column/mechanical.npz") as f:
        centres = f["intervals"]
    dense_offsets = np.r_[np.arange(-12000., 12001., 100.), np.arange(-1000., 1001., 25.)]
    r = axial(q, args.ni, args.count, args.step, source_rate(), args.name, centres, args.table, FOCUS_OUT,
              extra_offsets_nm=dense_offsets)
    if args.count == ref["after"]["emitted"] and r["source_digest"] != ref["after"]["source_digest"]:
        raise ValueError("The original source phase space changed")
    phase = FOCUS_OUT / r["phase_file"]
    sample = read_sample(phase, r, read_json(EVIDENCE / "comparison.json"))
    basis, labels = design(sample)
    r.update(mini_gain=args.gain, mini_winding_currents_A=q[mini].tolist(),
             objective_A=args.ni / ref["round_windings"]["objective_turns"], objective_solution=objective,
             corrector_sha256=sha256(args.table.read_bytes()).hexdigest(),
             trajectory_sha256=sha256(phase.read_bytes()).hexdigest(),
             corrector_currents_A=ref["corrector_winding_currents_A"],
             fit=fit_record(sample, basis, labels, np.ones(len(sample.ids), dtype=bool)),
             ring=ring_measure(sample.xy_um, sample.current_na))
    with np.load(phase, allow_pickle=False) as f:
        offsets = f["offsets_nm"]
        indexes = np.searchsorted(f["z_m"], SURFACE + offsets*1e-9)
        profiles = f["profiles"][indexes]
        r["through_focus"] = {key: dict(offset_um=float(offsets[profiles[:, i].argmin()]*.001),
                                        min_um=float(profiles[:, i].min()*.001))
                              for i, key in enumerate(("D95", "RMS_radius"))}
        r["through_focus"]["max_50pm_core_nA"] = float(profiles[:, 2].max())
    target = FOCUS_OUT / (args.name + ".json")
    target.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.name, r["surface"], r["through_focus"], r["ring"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    for name in ("search", "verify"):
        subp = sub.add_parser(name)
        subp.add_argument("name")
        subp.add_argument("--table", type=Path, default=OUT / "a2-set-field.npz")
        subp.add_argument("--count", type=int, default=32768)
        subp.add_argument("--step", type=float, default=.075)
        if name == "verify":
            subp.add_argument("--ni", required=True, type=float, help="Signed Objective ampere-turns")
            subp.add_argument("--gain", required=True, type=float, help="Common signed Mini winding multiplier")
    args = p.parse_args()
    FOCUS_OUT.mkdir(exist_ok=True, parents=True)
    (search if args.operation == "search" else verify)(args)
