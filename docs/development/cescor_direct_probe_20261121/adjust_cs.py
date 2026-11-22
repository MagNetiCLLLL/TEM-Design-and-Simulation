"""Tune the effective spherical ray coefficient with the physical corrector and relay."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from adjust_a2 import OUT, read_json
from column_current import SURFACE, WORK, current_ledger, field_array, metrics, source_digest, source_rate
from finite_probe import axial, bundle
from fit_threefold import EVIDENCE, REFERENCE_RAD, angular_sample, design, fit, physical_coefficients, read_sample, ring_measure
from refocus_a2 import objective_receipt
from temsim.physics.beam_statistics import transverse_beam_statistics
from tune import waist_proposal


CS_OUT = OUT / "spherical"
FIELD = OUT / "a2-set-field.npz"


def coefficients(sample, mask=None):
    basis, names = design(sample)
    if mask is None:
        mask = np.ones(len(sample.ids), dtype=bool)
    weights = sample.current_na[mask]
    c, cond = fit(basis[mask], sample.xy_um[mask], weights)
    residual = sample.xy_um[mask] - basis[mask] @ c
    result = physical_coefficients(c, names)
    for name, power in (("C1", 1), ("C3", 3), ("C5", 5)):
        # The local columns act on alpha / 30 mrad; the fitted displacement is in micrometres.
        value = c[names == name].item()
        result[name] = dict(signed_mm=value*.001/REFERENCE_RAD**power, radial_intercept_at_30mrad_um=-value)
    result["condition"] = cond
    result["rms_error_um"] = float(np.sqrt(np.average(np.sum(residual**2, axis=1), weights=weights)))
    return result


def currents(ref, tl_gain, mini_gain):
    q = np.asarray(ref["currents_A"]).copy()
    owner = np.asarray(ref["owner_per_group"])
    for key, gain in (("probe_tl12_lens", tl_gain), ("mini_condenser", mini_gain)):
        mask = owner == key
        assert mask.sum() == 3 and gain > 0
        q[mask] *= gain
    return q


def measured(phase, loss, tracker):
    live = loss == -1
    if live.sum() < 1000:
        raise ValueError("Insufficient physical transmissions for a spherical-coefficient fit")
    p = phase[live]
    w = tracker.weights[live]
    stats = transverse_beam_statistics(*p[:, :4].T, weights=w)
    sample = angular_sample(p, tracker.ids[live], w * tracker.current * 1e9, stats)
    d95, rms = waist_proposal(p, w)
    return dict(surface=metrics(p, w, tracker.current), fit=coefficients(sample), ring=ring_measure(sample.xy_um, sample.current_na),
                drift_D95_um=d95*1e6, drift_RMS_um=rms*1e6,
                emitted=len(live), survived=int(live.sum()), effective=int(live.sum()))


class Trial:
    def __init__(self, a, ref):
        self.ref, self.table = ref, a.table
        with np.load(EVIDENCE / ref["trajectory_file"]) as f:
            obs = f["z_m"]
        with np.load(a.table) as f:
            self.field_report = json.loads(str(f["report"]))
        if (self.field_report.get("vacuum_only") is not False or
                self.field_report["base_circuit_sha256"] != read_json(EVIDENCE / "a2-adjusted-working-point.json")["baseline_circuit_sha256"]):
            raise ValueError("A registered nonlinear field from the installed corrector is required")
        self.mesh, self.t, bg, coils = bundle(a.step, obs, ref["objective_At"], a.count, source_rate(), a.table,
                                               transverse_interpolation=a.interpolation)
        self.mesh.saved = np.array([np.flatnonzero(self.mesh.z == SURFACE).item()])
        self.field = field_array(self.mesh, bg, coils, ref["currents_A"])
        self.baseline_bg = bg
        self.bases = {abs(ref["objective_At"]): bg}
        owner = np.array(ref["owner_per_group"])
        self.drives = {}
        for key in ("mini_condenser", "probe_tl12_lens"):
            g = owner == key
            self.drives[key] = np.einsum("i,ijk->jk", np.asarray(ref["currents_A"])[g], coils[g])
        self.raw = {}
        self.log = CS_OUT / (a.name + ".json")
        self.meta = dict(corrector_file=a.table.name, corrector_sha256=sha256(a.table.read_bytes()).hexdigest(),
                         corrector_currents_A=self.field_report["currents_A"], source_digest=source_digest(self.t),
                         max_step_mm=a.step, transverse_interpolation=a.interpolation)

    def objective(self, ampere_turns):
        if ampere_turns not in self.bases:
            objective_receipt(-ampere_turns)
            self.bases[ampere_turns] = self.mesh.base("armco", -ampere_turns, "gap-fine", 8)
        return self.bases[ampere_turns]

    def run(self, ni, tl, mini, approximate):
        key = (float(ni), float(tl), float(mini), bool(approximate))
        if key in self.raw:
            return self.raw[key]
        absolute = abs(ni)
        if approximate:
            # Interpolate separately solved nonlinear fields only to propose currents.
            # The final accepted point is solved and retraced at its exact excitation.
            grid = np.array([12000., 21500., 22000., 22500., 23000., 23500., 23960.,
                             24000.8, 24040., 25000., 36000.])
            if not grid[0] <= absolute <= grid[-1] or ni >= 0:
                raise ValueError("Objective proposal is outside the nonlinear field sweep")
            i = np.clip(np.searchsorted(grid, absolute), 1, len(grid)-1)
            lo, hi = grid[i-1:i+1]
            if absolute == hi or absolute == lo:
                bg = self.objective(absolute)
            else:
                f = (absolute-lo)/(hi-lo)
                bg = (1-f)*self.objective(lo) + f*self.objective(hi)
        else:
            bg = self.objective(absolute)
        field = self.field.copy()
        field[:, :bg.shape[1]] += (bg-self.baseline_bg) + (tl-1)*self.drives["probe_tl12_lens"] + (mini-1)*self.drives["mini_condenser"]
        phase, loss, _, _ = self.t.run(field, keep_history=False)
        row = measured(phase[-1], loss, self.t)
        row.update(objective_At=float(ni), tl12_gain=float(tl), mini_gain=float(mini), approximate_objective=approximate,
                   ledger=current_ledger(self.mesh, self.t, loss))
        self.raw[key] = row
        self.log.write_text(json.dumps(dict(**self.meta, trials=list(self.raw.values())), indent=2, allow_nan=False)+"\n")
        print(f"{ni:9.1f} At TL {tl:.8f} Mini {mini:.8f}  C3 {row['fit']['C3']['signed_mm']:.3f} mm "
              f"A2 {row['fit']['A2']['magnitude_mm']:.3f} mm D95 {row['surface']['D95_nm']*.001:.4f} um "
              f"z {row['drift_D95_um']:.2f} um alpha {row['surface']['alpha95_mrad']:.4f}", flush=True)
        return row


def match(a, ref):
    trial = Trial(a, ref)
    # Normalized steps are used only in the optimizer. The tracker sees the
    # actual current of each fixed winding and retains every mechanical stop.
    scales = np.array([1000., .1, .1])
    start = np.array([a.ni, a.tl, a.mini])
    x = np.zeros(3)
    low = (np.array([-30000., .65, .70])-start)/scales
    high = (np.array([-18500., 1.35, 1.30])-start)/scales
    chosen = []
    for target in a.targets:
        best_norm = np.inf
        best_x = x.copy()

        def residual(v):
            nonlocal best_norm, best_x
            ni, tl, mini = start+v*scales
            if a.exact:
                ni = round(ni, 1)
            try:
                r = trial.run(ni, tl, mini, not a.exact)
            except ValueError as exc:
                if "Unresolved boundary" not in str(exc) and "Insufficient physical transmissions" not in str(exc):
                    raise
                print("Rejected trajectory:",ni,tl,mini,str(exc),flush=True)
                return np.full(3, 1e4 + np.linalg.norm(v-best_x))
            e = np.array([(r["fit"]["C3"]["signed_mm"]-target)/40.,
                          (r["surface"]["alpha95_mrad"]-a.angle)/.2, r["drift_D95_um"]/80.])
            if np.linalg.norm(e) < best_norm:
                best_norm, best_x = np.linalg.norm(e), v.copy()
            return e

        def response(v):
            delta = np.array([.001, .0007, .0005])
            e = residual(v)
            columns = []
            for dv, h in zip(np.diag(delta), delta):
                for factor in (1., -1., .25, -.25, 2., -2.):
                    probe = residual(v + dv*factor)
                    if np.max(abs(probe)) < 1e4:
                        columns.append((probe-e)/(h*factor))
                        break
                else:
                    raise ValueError("Cannot measure a local optical response inside the field grid")
            return np.column_stack(columns)

        least_squares(residual,x,jac=response,bounds=(low,high),max_nfev=a.iterations,
                      x_scale="jac",gtol=1e-7,ftol=2e-7,xtol=2e-7)
        x = best_x
        ni, tl, mini = start+scales*x
        row = trial.run(round(ni,1) if a.exact else ni,tl,mini,not a.exact)
        chosen.append(dict(target_Cs_mm=target, **row))
        (CS_OUT/(a.name+"-selected.json")).write_text(json.dumps(chosen,indent=2,allow_nan=False)+"\n")
        print("Matched target",target, "scaled residual",residual(x),flush=True)
        if abs(row["drift_D95_um"])>50 or abs(row["surface"]["alpha95_mrad"]-a.angle)>.35:
            break


def lock(a, ref):
    """Match the actual 30 mrad cone and surface D95 waist at a separately solved objective current."""
    t = Trial(a, ref)
    centre = np.array([a.tl, a.mini])
    history = []
    for _ in range(a.iterations):
        def errors(x):
            r = t.run(a.ni, *x, False)
            return np.array([r["surface"]["alpha95_mrad"]-a.angle, r["drift_D95_um"]/100.])

        e = errors(centre)
        history.append(t.run(a.ni, *centre, False))
        (CS_OUT/(a.name+"-selected.json")).write_text(json.dumps(history,indent=2)+"\n")
        if abs(e[0]) < .013 and abs(e[1]) < .0007:
            break
        h = max(2e-7,min(4e-5,np.linalg.norm(e)*8e-5))
        jac = []
        for dv in np.eye(2)*h:
            for factor in (1., -1., .25, -.25, 2., -2.):
                try:
                    jac.append((errors(centre+factor*dv)-e)/(h*factor))
                    break
                except ValueError as exc:
                    if "Unresolved boundary" not in str(exc):
                        raise
            else:
                raise ValueError("No admitted finite-difference step for focus matching")
        step = np.linalg.lstsq(np.column_stack(jac),-e,rcond=1e-8)[0]
        step *= min(1., .012/max(1.e-12,abs(step).max()))
        for scale in (1., .5, .2, .05, .01):
            try:
                candidate = centre + scale*step
                objective = errors(candidate)
            except ValueError as exc:
                if "Unresolved boundary" not in str(exc):
                    raise
                continue
            if np.linalg.norm(objective) < np.linalg.norm(e):
                centre = candidate
                break
        else:
            break
    # At finite sampling alpha95 may change discontinuously when a marginal
    # packet is clipped. Prioritize a sampled native waist within the cone tolerance.
    best = min((r for r in t.raw.values() if abs(r["surface"]["alpha95_mrad"]-a.angle)<.10),
               key=lambda r: abs(r["drift_D95_um"]), default=history[-1])
    from scipy.optimize import brentq
    tl, mini = best["tl12_gain"], best["mini_gain"]
    def focus(x):
        return t.run(a.ni, tl, x, False)["drift_D95_um"]
    for h in (2e-6, 1e-5, 5e-5, .0002):
        try:
            if focus(mini-h)*focus(mini+h)>0:
                continue
            g = brentq(focus,mini-h,mini+h,xtol=2e-11)
            candidate = t.run(a.ni,tl,g,False)
            if abs(candidate["surface"]["alpha95_mrad"]-a.angle)<.10 and abs(candidate["drift_D95_um"])<abs(best["drift_D95_um"]):
                best = candidate
            break
        except ValueError as exc:
            if "Unresolved boundary" not in str(exc):
                raise
    (CS_OUT/(a.name+"-selected.json")).write_text(json.dumps([*history,best], indent=2)+"\n")


def verify(a, ref):
    q = currents(ref, a.tl, a.mini)
    receipt = objective_receipt(a.ni)
    with np.load(WORK / "column/mechanical.npz") as f:
        centres = f["intervals"]
    offsets = np.r_[np.arange(-12000., 12001., 100.), np.arange(-1000., 1001., 25.)]
    r = axial(q, a.ni, a.count, a.step, source_rate(), a.name, centres, a.table, CS_OUT, extra_offsets_nm=offsets,
              transverse_interpolation=a.interpolation)
    path = CS_OUT / r["phase_file"]
    sample = read_sample(path, r, read_json(EVIDENCE / "comparison.json"))
    with np.load(path) as f:
        z = f["offsets_nm"]
        profiles = f["profiles"][np.searchsorted(f["z_m"], SURFACE + z*1e-9)]
    with np.load(a.table) as field:
        field_report = json.loads(str(field["report"]))
        xy_pitch = float(np.diff(field["xy_m"][:2])[0] * 1000)
        z_pitch = float(np.diff(field["z_local_m"][:2])[0] * 1000)
    r.update(transverse_interpolation=a.interpolation, transverse_pitch_mm=xy_pitch, axial_pitch_mm=z_pitch,
             field_file=a.table.name, fit=coefficients(sample), fit_inside_30mrad=coefficients(sample, sample.alpha_mrad<=30),
             ring=ring_measure(sample.xy_um,sample.current_na), objective_solution=receipt,
             tl12_gain=a.tl, mini_gain=a.mini, corrector_field=field_report,
             corrector_sha256=sha256(a.table.read_bytes()).hexdigest(),
             trajectory_sha256=sha256(path.read_bytes()).hexdigest(),
             through_focus={label: dict(min_um=float(profiles[:,i].min()*.001), offset_um=float(z[profiles[:,i].argmin()]*.001))
                            for i,label in enumerate(("D95","RMS_radius"))})
    (CS_OUT/(a.name+".json")).write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(r["surface"], r["through_focus"], r["fit"]["C3"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("point", "responses", "match", "lock", "verify"))
    parser.add_argument("name")
    parser.add_argument("--table", type=Path, default=FIELD)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--step", type=float, default=.12)
    parser.add_argument("--interpolation", choices=("linear","cubic"), default="linear",
                        help="Transverse field interpolation in both the current search and verification")
    parser.add_argument("--ni", type=float, default=-24000.8)
    parser.add_argument("--tl", type=float, default=1.)
    parser.add_argument("--mini", type=float, default=1.)
    parser.add_argument("--targets", nargs="+", type=float, default=[-480.,-300.,-150.,-50.,0.])
    parser.add_argument("--angle", type=float, default=30.)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--exact", action="store_true", help="Solve the nonlinear objective at each proposed current")
    args = parser.parse_args()
    CS_OUT.mkdir(parents=True, exist_ok=True)
    ref = read_json(EVIDENCE / "refocused-working-point.json")
    if args.operation == "verify":
        verify(args, ref)
    elif args.operation == "match":
        match(args, ref)
    elif args.operation == "lock":
        lock(args, ref)
    else:
        trial = Trial(args, ref)
        trial.run(args.ni, args.tl, args.mini, False)
        if args.operation == "responses":
            trial.run(args.ni-40, args.tl, args.mini, True)
            trial.run(args.ni, args.tl+.0002, args.mini, False)
            trial.run(args.ni, args.tl, args.mini+.0002, False)
