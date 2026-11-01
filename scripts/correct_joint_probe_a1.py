"""Measure physical normal/skew A1 trim at the 30 mrad joint working point."""
import argparse
import numpy as np

import calibrate_physical_a5 as physical
from report_probe_zoom20 import harmonics
from scan_joint_probe_si110 import verified_focus, sha

PARENT = physical.ROOT/"docs/development/probe_joint_20261101/extend-joint-01-focus1"
ARCHIVE = physical.ROOT/"docs/development/probe_a1_trim"
WORK = physical.ROOT/"outputs/probe-a1-trim"
BASE_APPLY = physical.apply
STEP = .003  # Percentage points of the installed normal or skew winding scale.
AXES = ("strength_x_percent", "strength_y_percent")
SAMPLE_ANGLE = .030
RESPONSE_CASES = (("x-plus",STEP,0.),("x-minus",-STEP,0.),
                  ("y-plus",0.,STEP),("y-minus",0.,-STEP))
LIMITS = dict(max_condition=10., maximum_A1_nm=.05, maximum_angle_change_mrad=.05,
              maximum_protected_edge_change_pm=5., maximum_focus_offset_nm=.25)


def apply_trim(state, changes):
    previous = dict(changes)
    offsets = [previous.pop(key,0.) for key in AXES]
    BASE_APPLY(state, previous)
    trim = state.condenser_stigmator
    if not trim.enabled or trim.field_model != "normal_skew":
        raise ValueError("The two installed stigmator windings must be active")
    for key,offset in zip(AXES,offsets):
        value = getattr(trim,key)+offset
        if not np.isfinite(value) or abs(value)>100.:
            raise ValueError("Invalid signed stigmator setting")
        setattr(trim,key,value)


def configure():
    physical.ARCHIVE,physical.WORK,physical.apply = ARCHIVE,WORK,apply_trim
    ARCHIVE.mkdir(parents=True,exist_ok=True)


def audit_fields():
    from temsim.instrument_snapshot import decode_instrument
    from audit_probe_drives import normalized_rank
    configure()
    state,_,_,_ = physical.load_baseline(physical.INPUTS)
    parent = physical.read_json(PARENT/"result.json")
    BASE_APPLY(state,parent["controls"])
    snapshot = physical.capture_instrument_snapshot(state)
    if snapshot.digest != parent["input_digest"]:
        raise ValueError("The 30 mrad input cannot be restored exactly")
    verified_focus(PARENT,parent)
    trim = state.condenser_stigmator
    centre = physical.component_plane(state,trim)
    angles = np.arange(12)*2.*np.pi/12.
    points = np.array([(r*np.cos(t),r*np.sin(t),(centre+dz)*.001)
                       for dz in (-3.,0.,3.) for r in (10e-6,40e-6) for t in angles])
    columns = []
    for key in AXES:
        samples = []
        for sign in (-1.,1.):
            candidate = decode_instrument(snapshot.graph)
            part = candidate.condenser_stigmator
            setattr(part,key,getattr(part,key)+sign*STEP)
            field = physical.get_unified_field_runtime(candidate).snapshot
            samples.append(field.sample(points,source_ids=[trim.key])["B_t"])
        columns.append(((samples[1]-samples[0])/(2*STEP)).ravel())
    rank = normalized_rank(columns)
    if rank["rank"] != 2:
        raise ValueError("The installed field responses are not independent")
    physical._json(ARCHIVE/"physical-channel-audit.json",dict(parent_input=snapshot.digest,
        host=trim.key,centre_mm=centre,field_model=trim.field_model,
        reference_percent=[getattr(trim,key) for key in AXES],coordinate_names=AXES,
        winding_axis_deg=[trim.channel_x_angle_deg,trim.channel_y_angle_deg],
        field_scale_m2=trim.max_strength_m2,field_rank=rank,
        field_response_cosine=float(np.dot(*columns)/(np.linalg.norm(columns[0])*np.linalg.norm(columns[1]))),
        scope="Installed simulator field coordinates. Independent hardware supply currents and DAC calibration are unverified."))
    print("Field rank",rank["rank"],flush=True)


def run_case(name,x,y,step=.05,objective_steps=0.):
    configure()
    changes = physical.read_json(PARENT/"result.json")["controls"].copy()
    changes.update(zip(AXES,(float(x),float(y))))
    changes["Objective"] += objective_steps
    rec = physical.run_trial(name,changes,step_mm=step,focus=True,
                             image_focus_tolerance_nm=LIMITS["maximum_focus_offset_nm"])
    path = ARCHIVE/name/"input-recipe.json"
    recipe = physical.read_json(path)
    recipe["restoration"] = ("Restore the portable original source with load_baseline, then call "
                             "correct_joint_probe_a1.apply_trim(state, changes) and set state.step_mm.")
    physical._json(path,recipe)
    return rec


def at(folder):
    rec = physical.read_json(folder/"result.json")
    fit = rec["fits"][0]
    def source_hashes(record):
        return {q["source_id"]:q["potential_digest"] for q in record["field_audit"]["field_ownership"]["sources"]}
    reference = physical.read_json(PARENT/"result.json")
    recent,base = source_hashes(rec),source_hashes(reference)
    if recent.keys() != base.keys() or rec["source_hashes"] != reference["source_hashes"]:
        raise ValueError("Optical inventory or emitted source changed")
    changed = sorted(k for k in base if recent[k] != base[k])
    if set(changed)-{"condenser_stigmator","objective_lens"}:
        raise ValueError("An undeclared optical field changed")
    with np.load(folder/"surface.npz") as f,np.load(PARENT/"surface.npz") as r:
        for key in ("source_id","weights","alive"):
            np.testing.assert_array_equal(f[key],r[key])
    with np.load(folder/"current.npz") as f:
        peak_na = float(np.max(f["current_a_yx"])*1e9)
    m = harmonics(folder)
    raw = rec["at_specimen"]
    obs = physical.read_json(folder/"through-focus.json")
    qualified = False
    if rec["crossover_status"] == "MEASURED":
        try:
            verified_focus(folder,rec)
            qualified = True
        except ValueError:
            pass
    return dict(input_digest=rec["input_digest"], changed_fields=changed, controls=rec["controls"],
        signed_A1_nm=(np.asarray(fit["terms_m"]["A1"])*1e9).tolist(),
        A1_nm=float(np.linalg.norm(fit["terms_m"]["A1"])*1e9),
        quarter_A1_nm=(np.asarray(rec["fits"][1]["terms_m"]["A1"])*1e9).tolist(),
        C3_mm=1e3*fit["terms_m"]["C3"][0], A5_mm=float(np.linalg.norm(fit["terms_m"]["A5"])*1e3),
        signed_A5_mm=(np.asarray(fit["terms_m"]["A5"])*1e3).tolist(),
        fit_holdout_pm=fit["holdout_rms_pm"], current=raw, peak_pixel_na=peak_na, core_M4=m["axis"][0]["moments"]["4"],
        core_count=m["axis"][0]["count"], harmonics=m, focus_pass=qualified,
        waist_offset_nm=(rec["at_crossover"] or {}).get("offset_from_surface_nm"),
        waist_region_count=len(obs["coarse_region"]["regions"]), at_crossover=rec["at_crossover"])


def measure_response():
    reference = at(PARENT)
    rows = {name:at(ARCHIVE/name) for name,_,_ in RESPONSE_CASES}
    base = np.array(reference["signed_A1_nm"])
    columns, curvatures = [],[]
    collateral = {}
    for axis in ("x","y"):
        high,low = (rows[axis+"-"+sign] for sign in ("plus","minus"))
        plus,minus = np.array(high["signed_A1_nm"]),np.array(low["signed_A1_nm"])
        columns.append((plus-minus)/(2*STEP))
        curvatures.append(np.linalg.norm((plus+minus)/2-base)/np.linalg.norm((plus-minus)/2))
        collateral[axis] = dict(C3_mm_per_pp=(high["C3_mm"]-low["C3_mm"])/(2*STEP),
            A5_vector_mm_per_pp=((np.array(high["signed_A5_mm"])-low["signed_A5_mm"])/(2*STEP)).tolist(),
            quarter_A1_nm_per_pp=((np.array(high["quarter_A1_nm"])-low["quarter_A1_nm"])/(2*STEP)).tolist(),
            alpha95_mrad_per_pp=(high["current"]["alpha95_mrad"]-low["current"]["alpha95_mrad"])/(2*STEP))
    jac = np.column_stack(columns)
    singular = np.linalg.svd(jac,compute_uv=False)
    condition = singular[0]/singular[-1]
    if physical.read_json(ARCHIVE/"physical-channel-audit.json")["field_rank"]["rank"] != 2:
        raise ValueError("A rank-two physical field measurement is required")
    if condition > LIMITS["max_condition"] or max(curvatures)>.01:
        raise ValueError("The two signed local responses do not resolve independent A1 controls")
    if any(row["changed_fields"] != ["condenser_stigmator"] for row in rows.values()):
        raise ValueError("The response is not isolated to the installed condenser stigmator")
    if any(row["current"]["surviving_rays"]!=2000 or not row["focus_pass"] for row in rows.values()):
        raise ValueError("Channel measurement did not retain the focused original cohort")
    correction = np.linalg.solve(jac,-base)
    if max(abs(correction)) > 6*STEP:
        raise ValueError("Requested full cancellation exceeds the local trim experiment")
    result = dict(parent=reference["input_digest"], step_percentage_points=STEP,
        limits=LIMITS, reference=reference, responses=rows, jacobian_A1_nm_per_percentage_point=jac.tolist(),
        singular_values_nm_per_percentage_point=singular.tolist(), condition=condition,
        central_difference_fractional_asymmetry=curvatures, collateral_response=collateral,
        proposed_signed_offsets_pp=correction.tolist())
    physical._json(ARCHIVE/"response.json",result)
    print("A1 response",jac,"condition",condition,"increments",correction,flush=True)
    return correction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("audit","probe-channel","solve","correct","topology"))
    parser.add_argument("--index",type=int,choices=range(4))
    parser.add_argument("--name",default="a1-corrected")
    parser.add_argument("--step",type=float,default=.05)
    parser.add_argument("--objective-steps",type=float,default=0.)
    parser.add_argument("--fraction",type=float,default=1.)
    args = parser.parse_args()
    if args.operation == "audit":
        audit_fields()
    elif args.operation == "probe-channel":
        run_case(*RESPONSE_CASES[args.index])
    elif args.operation == "solve":
        measure_response()
    elif args.operation == "topology":
        configure()
        physical.topology(args.name)
    else:
        offsets = np.asarray(physical.read_json(ARCHIVE/"response.json")["proposed_signed_offsets_pp"])
        run_case(args.name,*(args.fraction*offsets),step=args.step,objective_steps=args.objective_steps)
