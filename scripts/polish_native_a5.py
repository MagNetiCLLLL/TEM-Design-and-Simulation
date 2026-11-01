"""Propose a bounded compensation of tails around an executed A5 candidate."""
from argparse import ArgumentParser
import json
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize

from calibrate_physical_a5 import ARCHIVE, DRIVES, read_json, _json, coefficient_vector
from design_native_a5 import surface_phase, focus_proxy_nm, smooth_profile


def propose(source_name, name, span=2.):
    response = read_json(ARCHIVE/"protected-response.json")
    source = read_json(ARCHIVE/source_name/"result.json")
    base = read_json(ARCHIVE/"baseline/result.json")
    j = np.asarray(response["half"])
    phase, weights, ids = surface_phase(source_name)
    original_phase, ow, oi = surface_phase("baseline")
    np.testing.assert_array_equal(weights, ow)
    np.testing.assert_array_equal(ids, oi)
    spatial = []
    for label, *_ in DRIVES:
        plus, wp, ip = surface_phase(f"response-{label}-half-plus")
        minus, wm, im = surface_phase(f"response-{label}-half-minus")
        for w, i in ((wp, ip), (wm, im)):
            np.testing.assert_array_equal(w, weights)
            np.testing.assert_array_equal(i, ids)
        spatial.append(plus-minus)
    spatial = np.stack(spatial, axis=-1)*span
    matrix = j*span
    focus_columns = np.array([(focus_proxy_nm(phase+spatial[:, :, k]*1e-3, weights)-
                               focus_proxy_nm(phase-spatial[:, :, k]*1e-3, weights))/.002
                              for k in range(len(DRIVES))])
    # Keep the achieved A5 vector and C3 fixed to first order. C1 may change
    # only through the same physical excitation used to restore native focus.
    g = np.vstack((matrix[[7, 19, 20]], focus_columns[None, :]/.25))
    rhs = np.array([0., 0., 0., -focus_proxy_nm(phase, weights)/.25])
    particular = np.linalg.lstsq(g, rhs, rcond=1e-10)[0]
    null = null_space(g, rcond=1e-10)
    a0 = coefficient_vector(source["fits"][0])
    b0 = coefficient_vector(base["fits"][0])
    initial = smooth_profile(phase, weights)
    original = smooth_profile(original_phase, weights)

    def calc(v):
        x = particular+null@v
        delta = matrix@x
        profiles = smooth_profile(phase+spatial@x, weights)
        return x, delta, profiles

    def objective(v):
        x, _, profile = calc(v)
        return float((profile[4]/original[4])**2+.5*(profile[3]/original[3])**2+1e-5*np.dot(x, x))

    def guards(v):
        x, delta, profile = calc(v)
        aberrations = a0+delta[:25]-b0
        alpha = source["at_specimen"]["alpha95_mrad"]+delta[-1]
        return np.r_[1-x, 1+x, 1-np.linalg.norm(aberrations[3:5])/4.5,
            1-abs(aberrations[7])/4.8, 1-np.linalg.norm(delta[14:16])/2., 1-abs(delta[18])/2.,
            (alpha-29.7)/.03, (30.3-alpha)/.03,
            1-profile[0]/original[0], profile[1]/original[1]-1,
            1-profile[2]/original[2], 1-profile[3]/original[3]]

    attempts = []
    for start in (np.zeros(null.shape[1]), np.full(null.shape[1], .1), np.full(null.shape[1], -.1)):
        result = minimize(objective, start, method="SLSQP", constraints=[dict(type="ineq", fun=guards)],
                          options=dict(ftol=1e-12, maxiter=1000))
        x, delta, profile = calc(result.x)
        increment = dict(zip([row[0] for row in DRIVES], (span*x).tolist()))
        controls = {label: float(source["controls"].get(label, 0.)+increment[label]) for label, *_ in DRIVES}
        attempts.append(dict(success=bool(result.success), minimum_guard=float(guards(result.x).min()),
            message=str(result.message), cost=objective(result.x), controls=controls, increment_steps=increment,
            predicted_profile_fraction=(profile/original).tolist(), predicted_coefficient_delta_pm=delta[:25].tolist()))
    valid = [rec for rec in attempts if rec["success"] and rec["minimum_guard"] >= -1e-7]
    output = dict(status="LINEAR_PROPOSAL_ONLY" if valid else "NO_FEASIBLE_LOCAL_PROPOSAL",
        source_trial=source_name, increment_span=span, attempts=attempts,
        selected=min(valid, key=lambda rec: rec["cost"]) if valid else None,
        scope="Local response surrogate; no image refocus and no acceptance without physical propagation")
    _json(ARCHIVE/(name+"-polish-design.json"), output)
    if output["selected"]:
        _json(ARCHIVE/(name+"-controls.json"), output["selected"]["controls"])
    print(json.dumps(output, indent=2), flush=True)
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("name")
    parser.add_argument("--span", type=float, default=2.)
    args = parser.parse_args()
    propose(args.source, args.name, args.span)
