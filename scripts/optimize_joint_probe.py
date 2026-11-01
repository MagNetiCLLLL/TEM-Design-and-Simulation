"""Bounded multistart C3/C5/A5 exploration with original-source field replay.

Only executed unified-field checkpoints form current images. A local particle
response surrogate proposes inputs; it is neither a transport substitute nor
evidence of a global optimum. Each physical input is immutable and archived.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import qmc

import calibrate_physical_a5 as physical
from design_native_a5 import focus_proxy_nm

ROOT = physical.ROOT
PRIOR = ROOT/"docs/development/probe_a5_physical_20261101"
ARCHIVE = ROOT/"docs/development/probe_joint_20261101"
WORK = ROOT/"outputs/probe-joint-20261101"
LABELS = [row[0] for row in physical.DRIVES]
RANGES = np.array([100., 100., 150., 150., 150., 100., 500., 500.])


def configure():
    physical.ARCHIVE, physical.WORK = ARCHIVE, WORK
    ARCHIVE.mkdir(parents=True, exist_ok=True)


def phase_at(folder):
    with np.load(folder/"surface.npz", allow_pickle=False) as f:
        if len(f["weights"]) != 2000 or not f["alive"].all():
            raise ValueError("A complete original 2000-particle cohort is required")
        return np.column_stack([f[k] for k in ("x_m", "y_m", "tx_rad", "ty_rad")]), f["weights"].copy(), f["source_id"].copy()


def initial_response():
    response = physical.read_json(PRIOR/"protected-response.json")
    if response["drives"] != LABELS:
        raise ValueError("Driver order differs from the measured response")
    phase, weights, ids = phase_at(PRIOR/"baseline")
    columns = []
    for label in LABELS:
        plus, wp, ip = phase_at(PRIOR/f"response-{label}-half-plus")
        minus, wm, im = phase_at(PRIOR/f"response-{label}-half-minus")
        for w, i in ((wp, ip), (wm, im)):
            np.testing.assert_array_equal(w, weights)
            np.testing.assert_array_equal(i, ids)
        columns.append(plus-minus)
    return phase, weights, np.stack(columns, axis=-1), np.array(response["half"])


def weighted_quantile(values, weights, fraction):
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    return float(values[order[min(np.searchsorted(cdf, fraction*cdf[-1]), len(order)-1)]])


def proxy_metrics(phase, weights):
    """Optimization observables only, including shell-resolved angular tails."""
    xy = phase[:, :2]*1e9
    z = xy[:, 0]+1j*xy[:, 1]
    radius = abs(z)
    unit = z/np.maximum(radius, 1e-30)
    norm = weights/weights.sum()
    centered = xy-np.average(xy, axis=0, weights=weights)
    radii = np.linalg.norm(centered, axis=1)
    rms = np.sqrt(np.sum(norm*radii**2))
    d95 = 2*weighted_quantile(radii, weights, .95)
    core = np.sum(norm*expit((.05-radius)/.004))
    inner_edges = [.1, .25, .5, .75, 1.]
    outer_edges = [1., 1.5, 2., 3., np.inf]
    def shells(edges, harmonic):
        values = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            gate = expit((radius-lo)/.015)- (expit((radius-hi)/.015) if np.isfinite(hi) else 0.)
            values.append(np.sum(norm*gate*unit**harmonic))
        return values
    m3, m6 = shells(inner_edges, 3), shells(outer_edges, 6)
    return np.array([rms, d95, core, abs(sum(m3)), sum(map(abs, m3)),
                     abs(sum(m6)), sum(map(abs, m6)), np.sum(norm*expit((radius-1)/.015))])


def propose_batch(batch, scale=1., count=3, seed=715, source="baseline", response_file=None):
    configure()
    original_phase, weights, spatial, jac = initial_response()
    original = physical.read_json(PRIOR/"baseline/result.json")
    if source == "baseline":
        phase, current, controls = original_phase, original, np.zeros(8)
    else:
        phase, w, ids = phase_at(ARCHIVE/source)
        np.testing.assert_array_equal(w, weights)
        current = physical.read_json(ARCHIVE/source/"result.json")
        controls = np.array([current["controls"].get(label, 0.) for label in LABELS])
    if response_file:
        with np.load(response_file, allow_pickle=False) as saved:
            spatial, jac = saved["phase_jacobian"], saved["coefficient_jacobian"]
    ranges = RANGES*scale
    matrix = jac*ranges
    space = spatial*ranges
    focus = np.array([(focus_proxy_nm(phase+space[:, :, i]*1e-4, weights)-
                       focus_proxy_nm(phase-space[:, :, i]*1e-4, weights))/.0002 for i in range(8)])
    # A2 is allowed to vary. These coordinates merely precondition its two
    # large responses and enforce a predicted native-envelope focus equality.
    guard = np.vstack((matrix[[3, 4]]/50., focus[None, :]))
    rhs = np.array([0., 0., -focus_proxy_nm(phase, weights)])
    particular = np.linalg.lstsq(guard, rhs, rcond=1e-10)[0]
    basis = np.column_stack((null_space(guard, rcond=1e-10), np.linalg.pinv(guard, rcond=1e-10)[:, :2]))
    baseline_profile = proxy_metrics(original_phase, weights)
    base_coeff = physical.coefficient_vector(original["fits"][0])
    current_coeff = physical.coefficient_vector(current["fits"][0])
    # Different emphasis preserves several possible probe-quality tradeoffs.
    costs = [np.array([2., 1., -2., 1., 1., 1., 1., 1.]),
             np.array([1., 2., -3., 2., 1., 2., 1., 1.]),
             np.array([3., .5, -1., 1., 2., 1., 2., 1.])]
    attempts = []
    starts = np.vstack((np.zeros(basis.shape[1]), (qmc.Sobol(basis.shape[1], scramble=True, seed=seed).random_base2(3)-.5)*.4))
    for number, start in enumerate(starts):
        cost_weights = costs[number % len(costs)]
        def calc(v):
            x = particular+basis@v
            predicted = phase+space@x
            ratios = proxy_metrics(predicted, weights)/baseline_profile
            delta = matrix@x
            return x, predicted, ratios, delta
        def objective(v):
            x, _, ratio, delta = calc(v)
            # Core has a negative sign; all radial/halo terms are minimized.
            return float(np.dot(cost_weights, ratio**2)+1e-6*np.dot(x, x))
        def constraints(v):
            x, predicted, ratio, delta = calc(v)
            coeff = current_coeff+delta[:25]
            alpha = current["at_specimen"]["alpha95_mrad"]+delta[-1]
            return np.r_[1-x, 1+x, 1-np.linalg.norm(coeff[3:5])/80.,
                1-np.linalg.norm(coeff[1:3]-base_coeff[1:3])/30.,
                (alpha-29.72)/.1, (30.28-alpha)/.1,
                1.02-ratio[[0,1,3,4,5,6,7]], ratio[2]-.995]
        result = minimize(objective, start, method="SLSQP", constraints=[dict(type="ineq", fun=constraints)],
            options=dict(ftol=1e-9, maxiter=400))
        x, predicted, ratios, delta = calc(result.x)
        rec = dict(index=number, success=bool(result.success), message=str(result.message),
            minimum_guard=float(constraints(result.x).min()), cost=objective(result.x),
            controls=dict(zip(LABELS, (controls+ranges*x).tolist())),
            predicted_ratios=ratios.tolist(), predicted_focus_proxy_nm=focus_proxy_nm(predicted, weights),
            predicted_coefficient_vector_pm=(current_coeff+delta[:25]).tolist(),
            predicted_alpha95_mrad=current["at_specimen"]["alpha95_mrad"]+delta[-1])
        attempts.append(rec)
        print(batch, number, rec["success"], round(rec["minimum_guard"],6), np.round(ratios[[0,1,2,3,5]],3), flush=True)
    valid = sorted([r for r in attempts if r["success"] and r["minimum_guard"] >= -1e-6], key=lambda r:r["cost"])
    selected = []
    for rec in valid:
        x = np.array([rec["controls"][k] for k in LABELS])
        if all(np.linalg.norm((x-np.array([prev["controls"][k] for k in LABELS]))/ranges) > .03 for prev in selected):
            selected.append(rec)
        if len(selected) == count:
            break
    for i, rec in enumerate(selected):
        rec["name"] = f"{batch}-{i+1:02d}"
        physical._json(ARCHIVE/(rec["name"]+"-controls.json"), rec["controls"])
    output = dict(status="SURROGATE_PROPOSALS_ONLY", source=source, scale=scale, driver_ranges_steps=ranges.tolist(),
        seed=seed, coefficient_scope="C3, C5, D4, A5 jointly free; A1/A2 bounded, native focus proposed",
        profile_order=["RMS", "D95", "smooth core", "inner M3", "inner shell M3", "outer M6", "outer shell M6", "outer current"],
        attempts=attempts, selected=selected, response_file=str(response_file) if response_file else None)
    physical._json(ARCHIVE/(batch+"-design.json"), output)
    return output


def run_focused(name, controls, step_mm=.05, max_rounds=3):
    configure()
    _, weights, jac, _ = initial_response()
    runs = []
    for index in range(max_rounds):
        trial_name = name if index == 0 else f"{name}-focus{index}"
        rec = physical.run_trial(trial_name, controls, step_mm=step_mm)
        runs.append(trial_name)
        if rec["crossover_status"] != "MEASURED" or rec["at_specimen"]["surviving_rays"] != 2000:
            break
        if abs(rec["at_crossover"]["offset_from_surface_nm"]) <= .25:
            break
        phase, w, _ = phase_at(ARCHIVE/trial_name)
        np.testing.assert_array_equal(w, weights)
        proxy = focus_proxy_nm(phase, weights)
        slope = (focus_proxy_nm(phase+jac[:,:,6]*.01, weights)-focus_proxy_nm(phase-jac[:,:,6]*.01, weights))/.02
        increment = float(np.clip(-proxy/slope, -500., 500.))
        physical._json(ARCHIVE/(trial_name+"-focus-placement.json"), dict(from_trial=trial_name,
            actual_waist_offset_nm=rec["at_crossover"]["offset_from_surface_nm"], surface_covariance_proxy_nm=proxy,
            proposed_objective_increment_steps=increment, scope="Explicit shared Objective change followed by full replay"))
        controls = {**controls, "Objective": controls.get("Objective", 0.)+increment}
    outcome = dict(seed=name, executions=runs, final_trial=runs[-1],
                   placed=bool(rec["crossover_status"] == "MEASURED" and abs(rec["at_crossover"]["offset_from_surface_nm"]) <= .25))
    physical._json(ARCHIVE/(name+"-sequence.json"), outcome)
    print(json.dumps(outcome), flush=True)
    return outcome


def execute_batch(batch, workers=3):
    configure()
    proposal = physical.read_json(ARCHIVE/(batch+"-design.json"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run_focused, rec["name"], rec["controls"]) for rec in proposal["selected"]]
        for job in as_completed(jobs):
            print("FINISHED", job.result(), flush=True)


def assess(name):
    configure()
    rec = physical.read_json(ARCHIVE/name/"result.json")
    assessment_path=ARCHIVE/name/"joint-assessment.json"
    previous=physical.read_json(assessment_path) if assessment_path.exists() else {}
    qualification=(previous.get("topology_and_step_qualification","PENDING")
                   if previous.get("input_digest")==rec["input_digest"] else "PENDING")
    base = physical.read_json(PRIOR/"baseline/result.json")
    b, s = base["at_specimen"], rec["at_specimen"]
    exact = dict(original_particles=rec["original_positive_particles"] == 2000 and rec["source_hashes"] == base["source_hashes"],
        current=s["surviving_rays"] == 2000 and abs(s["current_a"]-b["current_a"]) < 1e-17,
        angle=29.7 <= s["alpha95_mrad"] <= 30.3,
        focus=rec["crossover_status"] == "MEASURED" and abs(rec["at_crossover"]["offset_from_surface_nm"]) <= .25)
    ratios = dict(d95=s["d95_nm"]/b["d95_nm"], rms=s["rms_radius_nm"]/b["rms_radius_nm"],
        core=s["core_50pm_a"]/b["core_50pm_a"],
        inner_m3=s["threefold_tail"]["harmonic_current_a"]/b["threefold_tail"]["harmonic_current_a"],
        inner_shell_m3=s["threefold_tail"]["shell_harmonic_sum_a"]/b["threefold_tail"]["shell_harmonic_sum_a"],
        outer_m6=rec["tails"]["outside_1nm"]["absolute_sixth_a"]/base["tails"]["outside_1nm"]["absolute_sixth_a"],
        outer_shell_m6=rec["tails"]["outer_shell_absolute_sum_a"]/base["tails"]["outer_shell_absolute_sum_a"],
        outer_current=rec["tails"]["outside_1nm"]["current_a"]/base["tails"]["outside_1nm"]["current_a"])
    improvement = {key: value > 1 if key == "core" else value < 1 for key, value in ratios.items()}
    delta = physical.coefficient_vector(rec["fits"][0])-physical.coefficient_vector(base["fits"][0])
    out = dict(name=name, input_digest=rec["input_digest"],hard_checks=exact, ratios=ratios, improvement=improvement,
        all_measured_goals=all(exact.values()) and all(improvement.values()),
        at_crossover=rec["at_crossover"], at_specimen=s, separation=rec["separation"],
        coefficients_pm=physical.coefficient_vector(rec["fits"][0]).tolist(), delta_coefficients_pm=delta.tolist(),
        topology_and_step_qualification=qualification, original_accepted_point="UNCHANGED")
    physical._json(ARCHIVE/name/"joint-assessment.json", out)
    return out


def summarize():
    configure()
    results=[]
    for f in sorted(ARCHIVE.glob("*/result.json")):
        if f.parent.name != "baseline":
            results.append(assess(f.parent.name))
    physical._json(ARCHIVE/"measured-comparison.json", dict(trials=results))
    physical.gallery()
    for rec in results:
        print(rec["name"], rec["hard_checks"], {k:round(v,4) for k,v in rec["ratios"].items()}, flush=True)


def refresh_response(center):
    """Update retained secant directions using new executed observations."""
    configure()
    _, weights, prior, coefficients = initial_response()
    ref = physical.read_json(ARCHIVE/center/"result.json")
    phase, w, ids = phase_at(ARCHIVE/center)
    np.testing.assert_array_equal(w, weights)
    u0 = np.array([ref["controls"].get(k,0.) for k in LABELS])
    c0 = np.r_[physical.coefficient_vector(ref["fits"][0]), ref["at_specimen"]["alpha95_mrad"]]
    observations=[]
    for path in ARCHIVE.glob("*/result.json"):
        if path.parent.name == center: continue
        rec=physical.read_json(path)
        u=np.array([rec["controls"].get(k,0.) for k in LABELS])
        distance=np.linalg.norm((u-u0)/RANGES)
        if distance < 1e-6 or distance > 5. or rec["at_specimen"]["surviving_rays"] != 2000: continue
        p, wp, ip=phase_at(path.parent)
        np.testing.assert_array_equal(wp,weights)
        np.testing.assert_array_equal(ip,ids)
        c=np.r_[physical.coefficient_vector(rec["fits"][0]),rec["at_specimen"]["alpha95_mrad"]]
        observations.append((path.parent.name,u-u0,p-phase,c-c0,distance))
    if len(observations) < 3: raise ValueError("Insufficient new secant observations")
    u=np.array([r[1]/RANGES for r in observations])
    p=np.array([r[2] for r in observations])
    c=np.array([r[3] for r in observations])
    weight=1/np.sqrt(1+np.array([r[4] for r in observations])**2)
    residual_p=p-np.einsum("ijk,nk->nij",prior,u*RANGES)
    residual_c=c-(coefficients@(u*RANGES).T).T
    inverse=np.linalg.pinv(u*weight[:,None],rcond=.01)
    dp=(inverse@(residual_p.reshape(len(u),-1)*weight[:,None])).reshape(8,2000,4).transpose(1,2,0)/RANGES
    dc=(inverse@(residual_c*weight[:,None])).T/RANGES
    jacobian=prior+dp
    singular=np.linalg.svd(u*weight[:,None],compute_uv=False)
    checks=[]
    for index,row in enumerate(observations):
        keep=np.arange(len(u)) != index
        inv=np.linalg.pinv(u[keep]*weight[keep,None],rcond=.01)
        change=(inv@(residual_p[keep].reshape(keep.sum(),-1)*weight[keep,None])).reshape(8,2000,4).transpose(1,2,0)/RANGES
        prediction=np.einsum("ijk,k->ij",prior+change,row[1])
        checks.append(dict(name=row[0],normalized_distance=row[4],
            position_holdout_rms_nm=float(np.sqrt(np.mean((prediction[:,:2]-row[2][:,:2])**2))*1e9)))
    file=ARCHIVE/(center+"-refreshed-response.npz")
    physical._npz(file,dict(phase_jacobian=jacobian,coefficient_jacobian=coefficients+dc))
    physical._json(ARCHIVE/(center+"-refreshed-response.json"),dict(center=center,
        observations=[r[0] for r in observations], singular_values=singular.tolist(),
        retained_secant_rank=int(np.count_nonzero(singular>.01*singular[0])),holdouts=checks,
        scope="Truncated local secant update; discarded directions retain the original measured Jacobian; prediction only, not a physical-rank qualification"))
    print(file,checks,flush=True)


def contact_sheet(names, title):
    configure()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    output=ARCHIVE/(title+".png")
    if output.exists(): raise FileExistsError(output)
    fig,axes=plt.subplots(len(names),2,figsize=(9,4*len(names)),layout="constrained",squeeze=False)
    for row,name in enumerate(names):
        rec=physical.read_json(ARCHIVE/name/"result.json")
        with np.load(ARCHIVE/name/"current.npz",allow_pickle=False) as kernel:
            pixels=kernel["current_a_yx"]*1e9
        for col,limit in enumerate((2.56,.22)):
            axis=axes[row,col]
            axis.imshow(pixels,origin="lower",extent=[-2.56,2.56,-2.56,2.56],cmap="magma",
                        norm=SymLogNorm(linthresh=5.,vmin=0.,vmax=310.),interpolation="nearest")
            axis.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel="X (nm)",ylabel="Y (nm)")
            axis.set_title(name+(" | full field" if col==0 else " | central field"),fontsize=9)
    fig.suptitle("Native specimen current | fixed absolute color scale | same 2000 particles")
    fig.savefig(output,dpi=130)
    plt.close(fig)
    print(output,flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("design", "batch", "run", "summary", "topology", "verify", "refresh", "sheet"))
    p.add_argument("--name", default="baseline")
    p.add_argument("--controls", type=Path)
    p.add_argument("--source", default="baseline")
    p.add_argument("--response", type=Path)
    p.add_argument("--scale", type=float, default=1.)
    p.add_argument("--seed", type=int, default=715)
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--step-mm", type=float, default=.05)
    p.add_argument("--trials", nargs="+")
    args=p.parse_args()
    configure()
    if args.action == "design": propose_batch(args.name,args.scale,args.count,args.seed,args.source,args.response)
    elif args.action == "batch": execute_batch(args.name)
    elif args.action == "summary": summarize()
    elif args.action == "topology": physical.topology(args.name)
    elif args.action == "verify": physical.verify_archive()
    elif args.action == "refresh": refresh_response(args.name)
    elif args.action == "sheet": contact_sheet(args.trials,args.name)
    else: run_focused(args.name,physical.read_json(args.controls) if args.controls else {}, args.step_mm)


if __name__ == "__main__":
    main()
