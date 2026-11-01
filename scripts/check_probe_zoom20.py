"""Independent C2/C3 zoom and shared-Objective focus experiment, fixed corrector."""
import argparse

import numpy as np
from scipy.optimize import root
from numba import njit

import calibrate_physical_a5 as p
from design_native_a5 import focus_proxy_nm
from temsim.physics.round_lens_aberration_benchmark import _charge_over_momentum

PARENT = p.ROOT / "docs/development/probe_joint_20261101/extend-joint-01-focus1"
ARCHIVE = p.ROOT / "docs/development/probe_zoom20"
WORK = p.ROOT / "outputs/probe-zoom20"
CONDENSERS = ("condenser_lens_2", "condenser_lens_3")
BASE_APPLY = p.apply


@njit
def transport_height(height, slope, k2, step):
    # Midpoint constant-field transfer in the Larmor chart. Proposal only.
    for strength in k2:
        k = np.sqrt(strength)
        phase = k*step
        sinc = np.sin(phase)/k if k else step
        c = np.cos(phase)
        height, slope = c*height+sinc*slope, -strength*sinc*height+c*slope
    return np.array([height, slope])


def propose():
    state, _, _, source = p.load_baseline(p.INPUTS)
    parent = p.read_json(PARENT/"result.json")
    p.apply(state,parent["controls"])
    assert p.capture_instrument_snapshot(state).digest == parent["input_digest"]
    # Infer the dominant divergent gun-exit pencil from the entire emitted population.
    xy = np.stack([source["gun_exit_"+k] for k in ("x_m","y_m")], axis=1)
    slopes = np.stack([source["gun_exit_"+k] for k in ("tx_rad","ty_rad")], axis=1)
    w = source["weights"]
    xy -= np.average(xy, axis=0, weights=w)
    slopes -= np.average(slopes, axis=0, weights=w)
    curvature = float(np.sum(w[:,None]*xy*slopes)/np.sum(w[:,None]*xy**2))
    z0 = state.electron_gun.exit_plane_z_mm
    z1 = 1030.
    grid = np.linspace(z0,z1,round((z1-z0)/.025)+1)*.001
    mid = (grid[1:]+grid[:-1])/2
    points = np.column_stack([np.zeros_like(mid),np.zeros_like(mid),mid])
    runtime = p.get_unified_field_runtime(state)
    bfields = np.array([runtime.snapshot.sample(points,source_ids=[key])["B_t"][:,2]
                          for key in ("condenser_lens_1",*CONDENSERS)])
    rigidity_factor = .5*_charge_over_momentum(300000.)
    def ray(factors):
        bz = bfields[0]+factors[0]*bfields[1]+factors[1]*bfields[2]
        return transport_height(1., curvature,(rigidity_factor*bz)**2, grid[1]-grid[0])
    reference = ray([1.,1.])
    target = 20./parent["at_specimen"]["alpha95_mrad"]
    solution = root(lambda x: ray(x)/reference-target,[1.,1.],tol=1e-10)
    if not solution.success:
        raise ValueError(solution.message)
    base = [p.objects(state)[key].percent for key in CONDENSERS]
    proposal = dict(parent=parent["input_digest"], reference_percent=base,
                    zoom_percent=(solution.x*base).tolist(), target_amplitude_ratio=target,
                    comparison_plane_mm=z1, input_curvature_per_m=curvature,
                    model="On-axis first-order Larmor proposal from the captured condenser fields; physical acceptance uses the full unified field and original rays.")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    p._json(ARCHIVE/"proposal.json",proposal)
    print(proposal,flush=True)
    return proposal


def configure():
    p.ARCHIVE, p.WORK = ARCHIVE, WORK
    if "C2" not in {v[0] for v in p.DRIVES}:
        p.DRIVES += tuple((label,key,"percent",.001,"percentage point") for label,key in zip(("C2","C3lens"),CONDENSERS))
    p.apply = apply_drives


def apply_drives(state, controls):
    controls = dict(controls)
    mute = controls.pop("hexapoles_zero", False)
    BASE_APPLY(state,controls)
    if mute:
        for key in ("probe_hp1_hexapole","probe_hp2_hexapole"):
            part = p.objects(state)[key]
            part.strength_m3 = 0.


def zoom_controls(factor=1., objective=0.):
    design = p.read_json(ARCHIVE/"proposal.json")
    controls = p.read_json(PARENT/"input-recipe.json")["changes"]
    controls["Objective"] += objective
    for name,seed,value in zip(("C2","C3lens"),design["reference_percent"],design["zoom_percent"]):
        controls[name] = (value-seed)*factor/.001
    return controls


def trial(name, controls, step, focus):
    configure()
    rec = p.run_trial(name, controls, step_mm=step, focus=focus)
    write_restoration_recipe(ARCHIVE/name)
    with np.load(ARCHIVE/name/"surface.npz") as f:
        keep = f["alive"]
        phase = np.stack([f[k][keep] for k in p.PHASE[:4]],axis=1)
        offset = -focus_proxy_nm(phase,f["weights"][keep])
    print(dict(focus_estimate_nm=offset,alpha=rec["at_specimen"]["alpha95_mrad"],
               alive=rec["at_specimen"]["surviving_rays"]),flush=True)
    return rec, offset


def write_restoration_recipe(folder):
    path = folder/"input-recipe.json"
    recipe = p.read_json(path)
    recipe["restoration"] = ("Restore the portable baseline via load_baseline, call "
                             "check_probe_zoom20.configure and apply_drives(state, changes), "
                             "then set state.step_mm to maximum_step_mm.")
    p._json(path,recipe)


def crossover_chain(name, controls):
    from measure_probe_crossover import _bounded_grid, _continue
    from temsim.optics.waist_evidence import finite_interval
    configure()
    state, _, ref, source = p.load_baseline(p.INPUTS)
    apply_drives(state,controls)
    state.step_mm = .05
    snapshot = p.capture_instrument_snapshot(state)
    if name == "baseline":
        assert snapshot.digest == p.read_json(PARENT/"result.json")["input_digest"]
    planes = {lens.key:p.component_plane(state,lens) for lens in state.lenses}
    ac = state.ac_deflector
    for key,z in (("ac_upper",ac.upper_z_mm),("ac_lower",ac.lower_z_mm)):
        planes[key] = p.component_plane(state,ac,z-ac.z_mm)
    labels = [*CONDENSERS,"adapter_lens","probe_tl22_lens","probe_tl21_lens",
              "probe_tl12_lens","ac_upper","ac_lower"]
    start = float(state.electron_gun.exit_plane_z_mm)
    nodes = np.unique(np.r_[start,np.concatenate([_bounded_grid(planes[a],planes[b],.5)
                                         for a,b in zip(labels,labels[1:])])])
    phase = np.column_stack([source["gun_exit_"+key] for key in p.PHASE[:4]])
    intervals = {}
    print(name,"measuring the complete condenser-to-scan-coil interval",flush=True)
    with p.numerical_job(3):
        cp, live, stops, keys = _continue(state,start,phase,source["gun_exit_kinetic_energy_ev"],
                source["gun_exit_flight_time_s"],source["gun_exit_alive"],source["gun_blocked_z_mm"],
                source["gun_blocked_key"],nodes)
        for a,b in zip(labels,labels[1:]):
            def observe(check,membership,lo,hi,spacing):
                return finite_interval(check,membership,source["weights"],lo,hi,
                                       maximum_spacing_mm=spacing+1e-9,source_ids=source["source_id"])
            coarse = observe(cp,live,planes[a],planes[b],.5)
            result = dict(status=coarse["status"],unresolved=coarse["unresolved"],regions=[])
            for old in coarse["regions"]:
                left = np.searchsorted(cp.z_mm,old["bracket_mm"][0])-2
                right = np.searchsorted(cp.z_mm,old["bracket_mm"][1])+2
                subnodes = _bounded_grid(cp.z_mm[left],cp.z_mm[right],.005)
                sub,mask,_,_ = _continue(state,subnodes[0],np.column_stack([getattr(cp,k)[left] for k in p.PHASE[:4]]),
                    cp.kinetic_energy_ev[left],cp.flight_time_s[left],live[left],stops,keys,subnodes)
                fine = observe(sub,mask,subnodes[0],subnodes[-1],.005)
                if len(fine["regions"]) != 1 or fine["unresolved"]:
                    raise ValueError(f"Local 2-D crossover refinement failed: {a} {b}")
                waist = fine["regions"][0]
                z = waist["z_mm"]
                _, rows = p._native_profile(sub,mask,source["weights"],source["source_id"],[z],z,z,ref["source_current_a"])
                result["regions"].append(dict(z_mm=z,radial_bracket_mm=waist["radial_bracket_mm"],
                      two_dimensional_bracket_mm=waist["bracket_mm"],observation=rows[0]))
            intervals[a+" to "+b] = result
    p._json(ARCHIVE/(name+"-chain.json"),dict(input_digest=snapshot.digest,source_digest=ref["emitted_identity_digest"],
          original_count=2000,coarse_spacing_mm=.5,refined_spacing_mm=.005,controls=controls,intervals=intervals))
    print(name,{k:[r["z_mm"] for r in v["regions"]] for k,v in intervals.items()},flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("propose","trial","topology"))
    parser.add_argument("--name",default="zoom-first")
    parser.add_argument("--step",type=float,default=.05)
    parser.add_argument("--factor",type=float,default=1.)
    parser.add_argument("--objective",type=float,default=0.)
    parser.add_argument("--focus",action="store_true")
    parser.add_argument("--mute-hexapoles",action="store_true")
    args = parser.parse_args()
    if args.action == "propose":
        propose()
    else:
        controls = zoom_controls(args.factor,args.objective)
        if args.mute_hexapoles:
            controls["hexapoles_zero"] = True
        if args.action == "trial":
            trial(args.name,controls,args.step,args.focus)
        else:
            crossover_chain(args.name,controls)
