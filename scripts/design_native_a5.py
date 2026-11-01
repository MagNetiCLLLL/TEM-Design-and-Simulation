"""Propose physical scalar changes using native finite-source focus responses.

Predicted ray positions are only an optimization surrogate. They are never
rendered, treated as propagated data, or used to accept a working point.
"""
from argparse import ArgumentParser
import json
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize

from calibrate_physical_a5 import ARCHIVE, DRIVES, TERMS, read_json, _json


def surface_phase(name):
    with np.load(ARCHIVE/name/"surface.npz", allow_pickle=False) as f:
        if not f["alive"].all() or len(f["weights"]) != 2000:
            raise ValueError("A shared complete original cohort is required")
        return np.column_stack([f[key] for key in ("x_m", "y_m", "tx_rad", "ty_rad")]), f["weights"].copy(), f["source_id"].copy()


def focus_proxy_nm(phase, weights):
    """Local radial-envelope zero estimate, not a certified 2D crossover."""
    centre = np.average(phase, axis=0, weights=weights)
    delta = phase-centre
    return float(np.sum(weights[:, None]*delta[:, :2]*delta[:, 2:]) /
                 np.sum(weights[:, None]*delta[:, 2:]**2)*1e9)


def smooth_profile(phase, weights):
    xy = phase[:, :2]*1e9
    z = xy[:, 0]+1j*xy[:, 1]
    radius = abs(z)
    unit = z/np.maximum(radius, 1e-30)
    outer = 1/(1+np.exp(np.clip((1-radius)/.06, -60, 60)))
    inner = 1/(1+np.exp(np.clip((.1-radius)/.01, -60, 60)))-outer
    centred = xy-np.average(xy, axis=0, weights=weights)
    rms = np.sqrt(np.average(np.sum(centred**2, axis=1), weights=weights))
    core = np.average(1/(1+np.exp(np.clip((radius-.05)/.004, -60, 60))), weights=weights)
    return np.array([rms, core, np.average(outer, weights=weights),
        abs(np.average(outer*unit**6, weights=weights)), abs(np.average(inner*unit**3, weights=weights))])


def design(name, span=100., protection="all", tail_weight=2., minimum_improvement=.01):
    response = read_json(ARCHIVE/"protected-response.json")
    base, jacobian = np.array(response["baseline_vector"]), np.array(response["half"])
    if response["drives"] != [row[0] for row in DRIVES]:
        raise ValueError("The response matrix has another drive inventory")
    ranges = np.array([min(span, 40.) if label == "ADL" else span for label, *_ in DRIVES])
    matrix = jacobian*ranges
    phase, weights, ids = surface_phase("baseline")
    derivatives, focus_full, focus_half = [], [], []
    for label, *_ in DRIVES:
        for size, h, collection in (("full", 1., focus_full), ("half", .5, focus_half)):
            plus, wp, ip = surface_phase(f"response-{label}-{size}-plus")
            minus, wm, im = surface_phase(f"response-{label}-{size}-minus")
            for w, i in ((wp, ip), (wm, im)):
                np.testing.assert_array_equal(w, weights)
                np.testing.assert_array_equal(i, ids)
            collection.append((focus_proxy_nm(plus, weights)-focus_proxy_nm(minus, weights))/(2*h))
            if size == "half":
                derivatives.append(plus-minus)
    spatial = np.stack(derivatives, axis=-1)*ranges
    focus_jac = np.array(focus_half)*ranges
    guard = np.vstack((matrix[[3, 4, 7]]/5, focus_jac[None, :]/.25))
    rhs = np.array([0., 0., 0., -focus_proxy_nm(phase, weights)/.25])
    particular = np.linalg.lstsq(guard, rhs, rcond=1e-10)[0]
    null = null_space(guard, rcond=1e-10)
    # Explicit normalized primary budgets avoid ill-conditioned inequalities
    # in the raw actuator coordinates. Native focus remains an equality.
    permitted = np.linalg.pinv(guard, rcond=1e-10)[:, :3]
    basis = np.column_stack((null, permitted))
    initial_profile = smooth_profile(phase, weights)
    initial_a5 = np.linalg.norm(base[19:21])
    monitored = []
    index = 0
    for label, _, symmetry in TERMS:
        width = 2 if symmetry else 1
        if label not in ("C1", "A2", "C3", "A5") and not (protection == "requested" and label in ("D4", "C5")):
            monitored.append(list(range(index, index+width)))
        index += width

    def calculation(v):
        x = particular+basis@v
        delta = matrix@x
        predicted = phase+spatial@x
        profile = smooth_profile(predicted, weights)/initial_profile
        a5 = np.linalg.norm(base[19:21]+delta[19:21])/initial_a5
        return x, delta, profile, a5, predicted

    def objective(v):
        x, _, profile, a5, _ = calculation(v)
        return float(a5*a5+tail_weight*profile[3]**2+1e-6*np.dot(x, x))

    def constraints(v):
        x, delta, profile, a5, _ = calculation(v)
        alpha = base[-1]+delta[-1]
        slack = v[-3:]
        return np.r_[1-x, 1+x, (alpha-29.71)/.03, (30.29-alpha)/.03,
            .8-np.linalg.norm(slack[:2]), .8-abs(slack[2]),
            1-profile[0], profile[1]-1, 1-profile[2], 1.05-profile[4], 1-minimum_improvement-a5,
            [1-np.linalg.norm(delta[indices])/4. for indices in monitored]]

    attempts = []
    starts = [np.zeros(basis.shape[1]), np.full(basis.shape[1], .1), np.full(basis.shape[1], -.1)]
    for start in starts:
        result = minimize(objective, start, method="SLSQP", constraints=[dict(type="ineq", fun=constraints)],
            options=dict(ftol=1e-12, maxiter=1500))
        x, delta, profile, a5, predicted = calculation(result.x)
        attempts.append(dict(success=bool(result.success), message=str(result.message),
            minimum_guard=float(constraints(result.x).min()), cost=objective(result.x),
            controls=dict(zip([row[0] for row in DRIVES], (x*ranges).tolist())),
            predicted_a5_fraction=float(a5), predicted_profile_fraction=profile.tolist(),
            predicted_delta=delta.tolist(), predicted_focus_nm=focus_proxy_nm(predicted, weights)))
    valid = [rec for rec in attempts if rec["success"] and rec["minimum_guard"] >= -1e-7]
    output = dict(status="LINEAR_PROPOSAL_ONLY" if valid else "NO_FEASIBLE_LOCAL_PROPOSAL",
        scope="A2 and C3 protected; physical finite-source focus compensates C1; all new inputs require replay",
        protection=protection, span_steps=span, tail_weight=tail_weight,
        minimum_improvement=minimum_improvement,
        below_predeclared_material_target=minimum_improvement < .01,
        profile_order=["RMS", "smoothed core", "smoothed outer current", "smoothed outer M6", "smoothed inner M3"],
        primary_guard_rank=int(np.linalg.matrix_rank(guard)), null_dimension=null.shape[1],
        focus_response_full_nm=focus_full, focus_response_half_nm=focus_half,
        baseline_focus_proxy_nm=focus_proxy_nm(phase, weights), attempts=attempts,
        selected=min(valid, key=lambda rec: rec["cost"]) if valid else None,
        smoothing_scope="Optimization only; every native image and acceptance metric uses unsmoothed propagated rays")
    _json(ARCHIVE/(name+"-native-design.json"), output)
    if output["selected"]:
        _json(ARCHIVE/(name+"-controls.json"), output["selected"]["controls"])
    print(json.dumps(output, indent=2), flush=True)
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--span", type=float, default=100.)
    parser.add_argument("--protection", choices=("all", "requested"), default="all")
    parser.add_argument("--tail-weight", type=float, default=2.)
    parser.add_argument("--minimum-improvement", type=float, default=.01)
    args = parser.parse_args()
    design(args.name, args.span, args.protection, args.tail_weight, args.minimum_improvement)
