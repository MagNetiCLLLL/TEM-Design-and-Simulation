"""Screen installed multipole drives with fixed Nanoprobe Normal relay and condenser settings."""

import argparse
import json
from types import SimpleNamespace

import cupy as cp
import numpy as np
from scipy.optimize import least_squares, lsq_linear

from downstream_pivot import inputs, write
from fit_threefold import angular_sample, design, fit
from relay_transfer import HERE, Native, SURFACE, WORK, advance
from temsim.physics.beam_statistics import transverse_beam_statistics
from tune import waist_proposal


KEYS = ("probe_hp1_hexapole", "probe_hp2_hexapole", "probe_hpol_hexapole_x", "probe_hpol_hexapole_y")
ANCHORS = ("m330", "normal-hp1-15", "internal-hp-eight", "normal-hpol-x", "internal-hp-off")


def fit_surface(phase, loss, tracker):
    live = loss == -1
    p, w = phase[live], tracker.weights[live]
    if len(np.unique(tracker.ids)) != len(loss) or len(w) < 1000 or not np.all(w > 0):
        raise ValueError("Fit requires unique original packets with positive current")
    stats = transverse_beam_statistics(*p[:, :4].T, weights=w)
    sample = angular_sample(p, tracker.ids[live], w * tracker.current * 1e9, stats)
    a, names = design(sample)
    c, condition = fit(a, sample.xy_um, sample.current_na)
    residual = sample.xy_um - a @ c
    z, zr = waist_proposal(p, w)
    radii = np.linalg.norm(p[:, :2], axis=1)
    projected = p[:, :2] + z*p[:, 2:4]
    projected -= np.average(projected, weights=w, axis=0)
    index = int(np.ceil(.95*len(w))) - 1
    if not np.all(w == w[0]):
        raise ValueError("The drift estimate requires the equal-weight reference source")
    return dict(coefficient_um_at_30mrad=dict(zip(names.tolist(), c.tolist())),
                fit_condition=condition, residual_rms_nm=float(1000*np.sqrt(np.average(
                    np.sum(residual**2, axis=1), weights=w))),
                D95_um=float(stats.radius_95_m*2e6),
                rms_radius_um=float(stats.radius_rms_m*1e6),
                centroid_nm=(np.average(p[:, :2], weights=w, axis=0)*1e9).tolist(),
                axis_50nm_nA=float(tracker.current*1e9*w[radii <= 50e-9].sum()),
                survivor_count=len(w), transmitted_uA=float(tracker.current*1e6*w.sum()),
                alpha95_mrad=float(stats.convergence_95_mrad), drift_D95_um=z*1e6,
                drift_RMS_um=zr*1e6, proposed_waist_D95_um=float(2e6*np.partition(
                    np.linalg.norm(projected, axis=1), index)[index]))


def field_path(tag):
    return WORK / (tag + "-field.npz")


class Screen:
    def __init__(self, count, step, name):
        ref, _ = inputs()
        self.name, self.ni = name, ref["objective_At"]
        self.model = m = Native(field_path(ANCHORS[0]), step, count, self.ni)
        self.controls, self.log = [], []
        self.base = m.t.tables[0].copy()
        self.changes = []
        with np.load(field_path(ANCHORS[0])) as data:
            registered = [data[k] for k in ("z_local_m", "xy_m", "valid_radius_m")]
        for idx, tag in enumerate(ANCHORS):
            with np.load(field_path(tag)) as data:
                for key, arr in zip(("z_local_m", "xy_m", "valid_radius_m"), registered):
                    np.testing.assert_array_equal(arr, data[key])
                desc = json.loads(str(data["report"]))
                drives = desc["currents_A"]
                if drives["probe_tl21_lens"] != 330 or drives["probe_tl22_lens"] != -145 or desc["vacuum_only"]:
                    raise ValueError("A response field changed a protected circuit")
                self.controls.append([drives[key] for key in KEYS])
                if idx:
                    self.changes.append(cp.asarray(data["B_zyx_T"]) - self.base)
        self.i0 = np.array(self.controls[0])
        self.response = (np.array(self.controls[1:]) - self.i0).T
        self.axial = m.field(ref["currents_A"])
        hi, lo = -14538.5, -14567.8
        self.db = cp.asarray((m.m.base("armco", hi, "gap-fine", 8) -
                              m.m.base("armco", lo, "gap-fine", 8)) / (hi-lo))
        self.cache = {}

    def run(self, settings):
        key = tuple(np.round(settings, 8))
        if key in self.cache:
            return self.cache[key]
        m = self.model
        field = self.axial.copy()
        delta_ni = settings[0] - self.ni
        field[:, :self.db.shape[1]] += self.db * delta_ni
        mix = np.linalg.solve(self.response, np.asarray(settings[1:]) - self.i0)
        m.t.tables[0][...] = self.base
        for delta, array in zip(mix, self.changes):
            if delta:
                m.t.tables[0] += delta*array
        ((phase, loss),) = advance(m.t, field, m.t.input, m.en, m.phi0, m.loss, 0, m.indices([SURFACE]))
        r = fit_surface(phase, loss, m.t)
        self.cache[key] = r
        self.log.append(dict(settings=list(settings), **r))
        write(WORK/(self.name+".json"), dict(anchors=ANCHORS, winding_keys=KEYS,
              scope="Nonlinear FEM field secants for proposals only; full trajectories from the gun.",
              emitted=len(m.loss), max_step_mm=m.step, trials=self.log))
        c = r["coefficient_um_at_30mrad"]
        print(np.round(settings, 5), "D95, focus, C3, C5, A2, D4",
              np.round([r["D95_um"],r["drift_D95_um"],c["C3"],c["C5"],
                       np.hypot(c["A2_x"],c["A2_y"]),np.hypot(c["D4_x"],c["D4_y"])], 6), flush=True)
        return r


def explore(args):
    survey = Screen(args.count, args.step, args.name)
    start = np.r_[survey.ni, survey.i0]
    scale = np.array([1., 15., 30., 1., 1.])
    base = survey.run(start)
    for delta in np.diag(scale):
        for sign in (1, -1):
            survey.run(start + sign*delta)
    for gain in (8., 40., -40., 100.):
        x = start.copy()
        x[1:3] = gain * start[1:3]
        survey.run(x)

    terms = ("A1", "B2", "A2", "S3", "A3", "D4", "C41", "C45", "C52", "C54", "A5")
    active_terms = [term+"_"+axis for term in terms for axis in ("x","y")]
    def residual(u, target):
        row = survey.run(start + scale*u)
        c = row["coefficient_um_at_30mrad"]
        # Keep each physical anisotropic term visible. Relax spherical residuals
        # only when the installed windings cannot reduce them independently.
        return np.r_[[c[k] for k in active_terms],
                     (c["C3"] - target[0]), (c["C5"] - target[1]),
                     (row["drift_D95_um"] - base["drift_D95_um"])*.015,
                     (row["alpha95_mrad"]-base["alpha95_mrad"])*2, .0005*u]
    b = base["coefficient_um_at_30mrad"]
    limits = np.array([12., 8., 5., 4., 4.])
    u = np.zeros(5)
    for fraction in (0., .01, .10, 1.):
        target = (1-fraction)*np.array([b["C3"],b["C5"]])
        def fun(v):
            return residual(v, target)
        def jac(v):
            step = np.array([.07,.04,.04,.04,.04])
            return np.column_stack([(fun(v+d)-fun(v))/s for d,s in zip(np.diag(step),step)])
        result = least_squares(fun, u, jac=jac, bounds=(-limits,limits), max_nfev=args.iterations,
                               xtol=.005, ftol=1e-4, gtol=1e-5, x_scale="jac")
        u = result.x
        selected = start+scale*u
        write(WORK/(args.name+f"-target-{fraction:g}.json"), dict(settings=selected.tolist(),
              target_fraction=fraction, reference=base, proposal=survey.run(selected)))


def balanced():
    """Use the full original source for the initial term values and matched secants for the response."""
    with np.load(HERE/"evidence/illuminated-normal-particles.npz") as d:
        t = SimpleNamespace(ids=d["source_id"], weights=d["weight"], current=15e-6)
        b = fit_surface(d["specimen_phase"], d["specimen_loss"], t)
    r = json.loads((WORK/"normal-corrector-screen.json").read_text())["trials"]
    def observables(row):
        c = row["coefficient_um_at_30mrad"]
        return np.r_[[c[k] for k in ("A2_x","A2_y","D4_x","D4_y")], row["drift_D95_um"]*.02,
                     c["C3"], c["C5"]]
    response = np.column_stack([(observables(r[i])-observables(r[i+1]))/2 for i in (1,3,5,7,9)])
    target = np.r_[-observables(b)[:5], 0., 0.]
    scale = np.array([1.,15.,30.,1.,1.])
    weights = np.array([1.,1.,1.,1.,1.,.4,.4])
    J = np.vstack((response*weights[:,None], np.diag([.001,.0005,.0005,.0007,.0007])))
    limit = np.array([3.,8.,5.,4.,4.])
    x = lsq_linear(J, np.r_[weights*target, np.zeros(5)], bounds=(-limit,limit)).x
    write(WORK/"normal-balanced-proposal.json", dict(controls=KEYS,
          settings=(np.array(r[0]["settings"])+scale*x).tolist(), full_source=b,
          response_scaled=response.tolist(), scales=scale.tolist()))


def shared():
    """Request a common incremental correction on the two independent material meshes."""
    proposal = json.loads((WORK/"normal-balanced-proposal.json").read_text())
    response = np.array(proposal["response_scaled"])
    scale = np.array(proposal["scales"])
    rows = json.loads((WORK/"normal-balanced-checks.json").read_text())
    coarse = [r for r in rows if "scan" in r]
    fine = json.loads((WORK/"normal-balanced-mesh-checks.json").read_text())
    def obs(row):
        c = row["coefficient_um_at_30mrad"]
        return np.r_[[c[k] for k in ("A2_x","A2_y","D4_x","D4_y")],
                     row["drift_D95_um"]*.02,c["C3"],c["C5"]]
    def settings(row):
        return np.r_[row["objective_At"], [row["corrector_A"][key] for key in KEYS]]
    c0,c1,f0,f1 = [obs(r) for r in (*coarse,*fine)]
    initial = settings(coarse[0])
    u = (settings(coarse[1])-initial)/scale
    response += np.outer(.5*(c1-c0+f1-f0)-response@u, u)/(u@u)
    delta = np.zeros(7)
    for j in (0,2):
        a,b = c0[j:j+2],f0[j:j+2]
        # Choose the common translation that minimizes the larger of the
        # two final-to-initial vector magnitudes on this pair of meshes.
        fraction = np.linalg.norm(a)/(np.linalg.norm(a)+np.linalg.norm(b))
        delta[j:j+2] = -((1-fraction)*a + fraction*b)
    wt = np.array([1.,1.,1.,1.,1.,.1,.1])
    limits = np.array([8.,4.,3.,6.,6.])
    matrix = np.vstack((response*wt[:,None],.00025*np.eye(5)))
    step = lsq_linear(matrix,np.r_[wt*delta,np.zeros(5)],bounds=(-limits,limits)).x
    write(WORK/"normal-shared-proposal.json",dict(controls=KEYS,
          settings=(initial+scale*step).tolist(),requested_common_change=delta.tolist(),
          predicted_coarse=(c0+response@step).tolist(),predicted_fine=(f0+response@step).tolist()))


def compact(args):
    survey = Screen(args.count, args.step, args.name)
    start = np.r_[survey.ni, survey.i0]
    other = [[2.8,39.26,2.4833,-4.21776], [-118.639,136.484,3.991,-1.256]]
    for main in (-220.,-120.,-45.,45.,120.,220.):
        for second in (-120.,120.):
            other.append([main, second, 3.8*np.sign(-main), -3.8])
    for s in [start[1:],*other]:
        x = np.r_[survey.ni,s]
        initial = survey.run(x)
        x[0] += np.clip(initial["drift_D95_um"]/3.1,-20,20)
        survey.run(x)


def trim(args):
    survey = Screen(args.count, args.step, args.name)
    reference = survey.run(np.r_[survey.ni, survey.i0])
    def a2(row):
        c = row["coefficient_um_at_30mrad"]
        return np.array([c["A2_x"], c["A2_y"]])
    r = json.loads((WORK/"normal-corrector-screen.json").read_text())["trials"]
    jac = np.column_stack([(a2(r[i])-a2(r[i+1]))/(2*scale) for i,scale in
                           ((3,15.),(5,30.),(7,1.),(9,1.))])
    for hp1 in (-7., 1., 7., 16.):
        for hp2 in (-65., -20., 25., 70.):
            main = np.array([hp1,hp2])
            hpol = np.clip(survey.i0[2:] + np.linalg.solve(jac[:,2:], -a2(reference)-
                              jac[:,:2]@(main-survey.i0[:2])), -5., 5.)
            settings = np.r_[survey.ni,main,hpol]
            initial = survey.run(settings)
            settings[0] += initial["drift_D95_um"]/3.1
            survey.run(settings)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="normal-corrector-screen")
    p.add_argument("--count", type=int, default=8192)
    p.add_argument("--step", type=float, default=.12)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--stage", choices=("response","balanced","compact","trim","shared"), default="response")
    args = p.parse_args()
    if args.stage == "balanced":
        balanced()
    elif args.stage == "shared":
        shared()
    elif args.stage == "compact":
        compact(args)
    elif args.stage == "trim":
        trim(args)
    else:
        explore(args)
