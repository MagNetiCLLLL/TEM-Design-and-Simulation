"""Measure final probe focus across a condenser zoom, separating core and envelope."""
import argparse

import numpy as np

import calibrate_physical_a5 as physical
import check_probe_zoom20 as zoom
import correct_joint_probe_a1 as trim


OUT = physical.ROOT / "docs/development/probe_zoom_focus_audit"
WORK = physical.ROOT / "outputs/probe-zoom-focus-audit"
HISTORIC_30 = zoom.PARENT
CURRENT_30 = trim.ARCHIVE / "a1-corrected"
ZOOM_ONLY = zoom.ARCHIVE / "zoom-first"
IMAGED_20 = zoom.ARCHIVE / "zoom-focused"


def expanded_cases():
    return {"old-30": HISTORIC_30, "zoom-only": ZOOM_ONLY, "imaged-20": IMAGED_20,
            "imaged-20-fine": zoom.ARCHIVE/"zoom-focused-fine", "current-30": CURRENT_30,
            "current-20": OUT/"current-a1-zoom-only", "core-control": OUT/"historic-core-control"}


def replay_imaged_probe():
    """Uniform 0.05 nm axial sampling through the 20 mrad probe used for the Si frame."""
    from measure_probe_crossover import _continue, _native_profile
    from measure_probe_waist import _hash
    from temsim.optics.waist_evidence import finite_interval
    scan = physical.read_json(IMAGED_20/"si110-scan/metrics.json")
    rec = physical.read_json(IMAGED_20/"result.json")
    prior = physical.read_json(IMAGED_20/"through-focus.json")
    zoom.configure()
    state,_,ref,source = physical.load_baseline(physical.INPUTS)
    zoom.apply_drives(state,rec["controls"])
    state.step_mm = rec["maximum_step_mm"]
    identity = physical.capture_instrument_snapshot(state)
    if (scan["input_digest"] != identity.digest or rec["input_digest"] != identity.digest
            or scan["surface_sha256"] != _hash(IMAGED_20/"surface.npz")
            or scan["current_sha256"] != _hash(IMAGED_20/"current.npz")
            or ref["emitted_identity_digest"] != scan["source_digest"]):
        raise ValueError("The executed optics or original particle cohort do not match the Si scan")
    # A verified Mini checkpoint accelerates the exact replay. A clean checkout
    # can execute the same original gun source all the way to the axial window.
    cache = zoom.WORK / "zoom-focused/arrays.npz"
    surface = float(state.sample.upper_surface_z_mm)
    weights, ids = source["weights"],source["source_id"]
    phase = np.column_stack([source["gun_exit_"+k] for k in physical.PHASE[:4]])
    energy,time,alive = [source["gun_exit_"+key] for key in ("kinetic_energy_ev","flight_time_s","alive")]
    blocked,keys = source["gun_blocked_z_mm"],source["gun_blocked_key"]
    start = state.electron_gun.exit_plane_z_mm
    prefix = []
    if cache.is_file():
        if _hash(cache) != prior["particle_file_sha256"]:
            raise ValueError("Cached source trajectory hash changed")
        with np.load(cache,allow_pickle=False) as f:
            np.testing.assert_array_equal(ids,f["source_id"])
            np.testing.assert_array_equal(weights,f["weights"])
            start = physical.component_plane(state,physical.objects(state)["mini_condenser"])
            i = np.flatnonzero(f["z_mm"]==start).item()
            phase = np.column_stack([f[k][i] for k in physical.PHASE[:4]])
            energy,time,alive = [f[k][i] for k in ("kinetic_energy_ev","flight_time_s","alive")]
            blocked,keys = f["blocked_z_mm"],f["blocked_key"]
    else:
        prefix = [state.condenser_aperture_2.z_mm,
                  physical.component_plane(state,physical.objects(state)["mini_condenser"])]
    offsets = np.arange(-800,801)*.05
    planes = surface+offsets*1e-6
    with physical.numerical_job(3):
        cp,active,_,_ = _continue(state,start,phase,energy,time,alive,blocked,keys,np.r_[start,prefix,planes])
    assert active.all() and len(ids)==2000
    current = ref["source_current_a"]
    _,rows = _native_profile(cp,active,weights,ids,planes,surface,surface,current)
    z = np.asarray(cp.z_mm)
    native = finite_interval(cp,active,weights,planes[0],planes[-1],
                             maximum_spacing_mm=.0505e-6,source_ids=ids)
    if native["unresolved"] or len(native["regions"]) != 1:
        raise ValueError("The dense observation does not resolve a two-direction source image")
    columns = ("core_50pm_a","d50_nm","d95_nm","rms_radius_nm","alpha95_mrad")
    data = {key:np.array([r[key] for r in rows]) for key in columns}
    data["offset_nm"] = offsets
    asurf = int(np.flatnonzero(offsets==0).item())
    i = np.flatnonzero(z==surface).item()
    with np.load(IMAGED_20/"surface.npz") as f:
        position_error_nm = np.hypot(cp.x_m[i]-f["x_m"], cp.y_m[i]-f["y_m"]).max()*1e9
        np.testing.assert_array_equal(ids,f["source_id"])
        np.testing.assert_array_equal(weights,f["weights"])
    if position_error_nm > .001 or abs(data["core_50pm_a"][asurf]-prior["at_specimen"]["core_50pm_a"])>1e-16:
        raise ValueError("Dense replay does not reproduce the imaged entrance")
    peak = data["core_50pm_a"].max()
    best = np.flatnonzero(np.isclose(data["core_50pm_a"],peak,rtol=0,atol=1e-16))
    d95 = int(np.argmin(data["d95_nm"]))
    rms = int(np.argmin(data["rms_radius_nm"]))
    if best.min()==0 or best.max()==len(offsets)-1 or d95 in (0,len(offsets)-1) or rms in (0,len(offsets)-1):
        raise ValueError("Unbounded axial extremum")
    archive = OUT/"imaged-20-dense"
    archive.mkdir(parents=True,exist_ok=True)
    physical._npz(archive/"axial-profile.npz",data)
    snapshots = {}
    for label,index in (("core",int(best[0])),("specimen",asurf),("d95",d95)):
        j = np.flatnonzero(z==planes[index]).item()
        snapshots[label+"_xy_nm"] = np.column_stack((cp.x_m[j],cp.y_m[j]))*1e9
    physical._npz(archive/"planes.npz",dict(snapshots,source_id=ids,
            particle_current_a=current*weights/weights.sum()))
    report = dict(input_digest=identity.digest,source_digest=scan["source_digest"],
        archived_positions_sha256=scan["surface_sha256"],maximum_surface_repeat_error_nm=position_error_nm,
        source_to_entrance_count=len(ids),source_current_a=current,used_verified_mini_cache=cache.is_file(),
        reference_upper_surface_mm=surface,sampling_nm=.05,window_nm=[-40,40],two_dimensional_waist=native["regions"][0],
        maximal_core_offsets_nm=offsets[best].tolist(),core_max_a=peak,
        surface=rows[asurf],example_core_plane=rows[best[0]],minimum_d95=rows[d95],minimum_rms=rows[rms])
    physical._json(archive/"measurement.json",report)
    print("Dense imaged probe",offsets[best],peak,offsets[d95],offsets[rms],flush=True)


def optical_control(name):
    """Replay the original gun, changing just the identified provider(s)."""
    zoom.configure()
    physical.ARCHIVE, physical.WORK, physical.apply = OUT, WORK, trim.apply_trim
    if name == "current-a1-zoom-only":
        # The original zoom and new A1 point share their corrector lens settings.
        baseline = physical.read_json(CURRENT_30 / "result.json")
        changes = baseline["controls"].copy()
        zoom_settings = physical.read_json(ZOOM_ONLY / "result.json")["controls"]
        changes.update({k:zoom_settings[k] for k in ("C2","C3lens")})
    elif name == "historic-core-control":
        baseline = physical.read_json(ZOOM_ONLY / "result.json")
        after = physical.read_json(IMAGED_20 / "result.json")
        changes = baseline["controls"].copy()
        # Use the same earliest sampled disk maximum convention in both records.
        shift = after["core_best"]["offset_from_surface_nm"] - baseline["core_best"]["offset_from_surface_nm"]
        delta_drive = after["controls"]["Objective"] - changes["Objective"]
        changes["Objective"] -= baseline["core_best"]["offset_from_surface_nm"] * delta_drive / shift
    else:
        raise ValueError(name)
    rec = physical.run_trial(name,changes,step_mm=.05,focus=True)
    path = OUT / name / "input-recipe.json"
    recipe = physical.read_json(path)
    recipe.update(restoration="Restore portable baseline, call check_probe_zoom20.configure(), "
                  "correct_joint_probe_a1.apply_trim(state, changes), then set state.step_mm.",
                  matched_parent=baseline["input_digest"])
    physical._json(path,recipe)
    return rec


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name",choices=("historic-core-control","current-a1-zoom-only","dense-replay"))
    name = parser.parse_args().name
    if name == "dense-replay":
        replay_imaged_probe()
    else:
        optical_control(name)
