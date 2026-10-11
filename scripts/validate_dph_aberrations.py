"""Bounded DPH/HP shape validation on one historical entrance population.

Prepare-only by default; --execute requires a frozen implementation. New Ideal
optics must retain the analytic HP and Cs operators. Custom is an independent
mode-routing comparison with identical analytic inputs, not a different source.
No source/optics defaults are modified and no wave calculation is performed.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,fields
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np

from validate_corrector_deflectors import prepare as prepare_state,trace,phase_points,resumed_initial
from validate_probe_relay import save_report,surface_measurement

ROOT=Path(__file__).resolve().parents[1]
LIMITS=dict(minimum_common_current_fraction=.999,upstream_centroid_shift_m=1e-12,
    minimum_inside_hp_centroid_shift_m=1e-10,minimum_hp_shape_slope_signal_rad=1e-10,
    minimum_sample_shape_signal_m=1e-12,maximum_step_relative_response_change=.01,
    maximum_step_absolute_response_change_m=1e-12,mode_relative_output_difference=1e-9,
    mode_absolute_output_difference_m=1e-14,minimum_refined_signal_to_step_difference=5.,affine_rcond=1e-12)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory():
    return {str(p.relative_to(ROOT/"src/temsim")).replace("\\","/"):digest(p)
            for p in sorted((ROOT/"src/temsim").rglob("*.py"))}


def physical_snapshot_audit(before,after):
    """Compare the complete graph, allowing only the observed backend readout.

    active_backend is written by the production tracer after actual execution.
    It is not the user-selected acceleration_backend. All other source/optics,
    mode controls, graph aliases, external inputs and implementation stay exact.
    """
    a,b=before.to_dict(),after.to_dict()
    readouts=[]
    for value in (a,b):
        value.pop("digest")
        state_node=value["graph"]["nodes"][value["graph"]["root"]["ref"]]
        readouts.append(state_node["attributes"].pop("active_backend"))
    return dict(complete_graph_equal_except_backend_readout=a==b,
                backend_readout_before=readouts[0],backend_readout_after=readouts[1],
                excluded_field="Only State.active_backend, the actual execution readout; requested acceleration_backend is compared unchanged")


def centered(a,w):
    a=np.asarray(a,float);w=np.asarray(w,float);w=w/w.sum()
    mean=a@w
    return a-mean[:,None],mean


def weighted_rms(a,w):
    return float(np.sqrt(np.sum(np.asarray(a)**2*(w/w.sum())[None,:])))


def common_cohort(results,incoming,weights):
    live=np.asarray(incoming["alive"],bool) & (weights>0)
    for result in results:
        live &= np.all(result[1],axis=0) & np.asarray(result[2],bool)
    denominator=weights[np.asarray(incoming["alive"],bool) & (weights>0)].sum()
    if not denominator>0 or live.sum()<5:
        raise ValueError("Insufficient same-ray cohort for HP shape comparison")
    return live,float(weights[live].sum()/denominator)


def affine_fit(reference,target,w):
    """Weighted affine observation; predictors are scaled to avoid mixed units."""
    x,_=centered(reference,w);y,_=centered(target,w)
    scale=np.sqrt(np.sum(x*x*(w/w.sum())[None,:],axis=1))
    scale=np.where(scale>0,scale,1.)
    design=x/scale[:,None]
    root=np.sqrt(w/w.sum())
    matrix,_,rank,singular=np.linalg.lstsq((design*root).T,(y*root).T,rcond=LIMITS["affine_rcond"])
    fitted=matrix.T@design
    return dict(matrix=(matrix.T/scale[None,:]).tolist(),predictor_rank=int(rank),
        singular_values=singular.tolist(),residual_rms_m=weighted_rms(y-fitted,w),
        scope="Best weighted affine observation in the declared predictors; a residual alone does not identify a particular aberration"),y-fitted


def compare_shape(reference,target,w,entrance):
    base,centre0=centered(reference[:2],w);changed,centre1=centered(target[:2],w)
    covariance=(changed*w/w.sum())@changed.T
    spatial,_=affine_fit(reference[:2],target[:2],w)
    phase,_=affine_fit(entrance,target[:2]-reference[:2],w)
    matrix=np.asarray(spatial["matrix"])
    symmetric=(matrix+matrix.T)/2
    traceless=symmetric-np.eye(2)*np.trace(symmetric)/2
    return dict(centroid_m=centre1.tolist(),centroid_change_m=(centre1-centre0).tolist(),
        radial_rms_m=weighted_rms(changed,w),covariance_m2=covariance.tolist(),
        principal_rms_m=np.sqrt(np.maximum(0,np.linalg.eigvalsh(covariance))).tolist(),
        centered_same_ray_change_rms_m=weighted_rms(changed-base,w),
        spatial_affine=spatial,spatial_affine_symmetric_traceless_norm=float(np.linalg.norm(traceless)),
        entrance_phase_space_affine_response=phase)


def hexapole_force(xy,normal,skew):
    x,y=xy
    return np.array((-normal*(x*x-y*y)-2*skew*x*y,
                     2*normal*x*y-skew*(x*x-y*y)))


def feeddown_matrix(displacement,normal,skew):
    a,b=displacement
    return np.array(((-2*(normal*a+skew*b),2*(normal*b-skew*a)),
                     (2*(normal*b-skew*a),2*(normal*a+skew*b))))


def _physical_equal(old,new):
    from temsim.calculation_cache import live_lens_parameters
    a,b=old.to_dict(),new.to_dict()
    ignored=("simulation_mode","simulation_mode_profiles")
    for key in ignored:a.pop(key,None);b.pop(key,None)
    checks={"state_except_explicit_mode_definition":a==b,
            "resolved_assembly":old._resolved_assembly==new._resolved_assembly}
    for name in ("lenses","corrector_elements","stigmators","apertures"):
        checks[name]={x.key:live_lens_parameters(x) for x in getattr(old,name)}=={
            x.key:live_lens_parameters(x) for x in getattr(new,name)}
    return checks


def prepare(args):
    from temsim.instrument_snapshot import InstrumentSnapshot,decode_instrument,capture_instrument_snapshot
    from temsim.physics.core import FIELD_SIGMA_CUTOFF
    old_dir=args.boundary
    files={name:digest(old_dir/name) for name in ("inputs.json","protocol.json","report.json","coordinates.npz")}
    old_snapshot=InstrumentSnapshot.from_dict(json.loads((old_dir/"inputs.json").read_text()))
    old=decode_instrument(old_snapshot.graph)  # Historical data reader, never stale snapshot.restore().
    original_report=json.loads((old_dir/"report.json").read_text())
    state,_,_=prepare_state(SimpleNamespace(inputs=None,rays=args.rays,step=args.step,
        delta_mrad=args.drive_mrad,reference_rays=32))
    equality=_physical_equal(old,state)
    if not all(equality.values()):raise ValueError("Historical boundary physical inputs differ: "+str(equality))
    if getattr(state,"lens_field_map_descriptors",{}):
        raise ValueError("This mode comparison requires shared analytic fields, not imported field maps")
    state.simulation_mode="ideal"
    with np.load(old_dir/"coordinates.npz",allow_pickle=False) as stored:
        archive={key:stored[key] for key in stored.files if key.startswith("baseline_") or key in ("weights","ray_id")}
    z=archive["baseline_z_mm"];weights=archive["weights"]
    if len(weights)!=args.rays or len(np.unique(archive["ray_id"]))!=len(weights):
        raise ValueError("Boundary must preserve the declared unique physical ray population")
    checkpoints=SimpleNamespace(**{name:archive["baseline_"+name] for name in (
        "z_mm","x_m","y_m","tx_rad","ty_rad","kinetic_energy_ev","flight_time_s")})
    rows=[];boundaries={}
    by_key={component.key:component for component in state.corrector_elements}
    for dph_key,hp in (("probe_dph2_deflector",state.hp2_hexapole),("probe_dph1_deflector",state.hp1_hexapole)):
        dph=by_key[dph_key]
        if not hp.enabled or not dph.enabled or max(abs(dph.kick_x_mrad),abs(dph.kick_y_mrad))+args.drive_mrad>dph.maximum_kick_mrad:
            raise ValueError("Active HP/DPH and an in-range diagnostic drive are required")
        half=FIELD_SIGMA_CUTOFF*hp.effective_length_mm/2.355
        field_lo,field_hi=hp.z_mm-half,hp.z_mm+half
        before=np.flatnonzero(z<min(field_lo,dph.z_mm-dph.effective_thickness_mm/2))
        if not before.size:raise ValueError("No executed boundary before the entire target HP field")
        start=float(z[before[-1]])
        incoming=resumed_initial(checkpoints,archive["baseline_alive"],start,
            archive["baseline_stops_mm"],archive["baseline_stop_keys"].tolist())
        boundaries[dph_key]=incoming
        sample=float(state.sample.upper_surface_z_mm)
        named=dict(hp_field_entrance=field_lo,hp_body_entrance=hp.z_mm-hp.mechanical_length_mm/2,
            hp_centre=hp.z_mm,hp_body_exit=hp.z_mm+hp.mechanical_length_mm/2,hp_field_exit=field_hi,
            sample=sample)
        sample_planes=[sample-.0002,sample-.0001,sample]
        planes=np.unique(np.r_[list(named.values()),sample_planes,np.arange(field_lo,field_hi,.5)])
        rows.append(dict(dph_key=dph_key,hp_key=hp.key,start_z_mm=start,sample_z_mm=sample,
            sample_planes_mm=sample_planes,named_planes_mm=named,planes_mm=planes.tolist(),
            baseline_command_mrad=[dph.kick_x_mrad,dph.kick_y_mrad],hp_strength_m3=hp.strength_m3,
            hp_orientation_rad=hp.orientation_rad,coil_effective_length_mm=dph.effective_thickness_mm,
            boundary_ray_count=int(incoming["alive"].sum())))
    snapshot=capture_instrument_snapshot(state)
    protocol=dict(schema="dph-hp-aberrations-v1",mode="Ideal with retained analytic nonlinear fields",
        numerical_step_mm=args.step,drive_mrad=args.drive_mrad,rays=args.rays,components=rows,limits=LIMITS,
        comparison_mode="Custom using the same analytic fields, source and optical settings",
        hp_off_control="Only target HP.strength_m3 is temporarily zero; its hardware stays installed/enabled, other HP and all Cs stay active; nothing is saved to defaults",
        predictions="For a local displaced hexapole F(u+d)-F(u)=F(d)+J(d)u. J is a symmetric traceless quadrupole. An isolated translated quadratic screen need not leave any nonaffine residual. Finite propagation and other nonlinear fields may do so.",
        response_scope="Target-HP-specific denotes a target-HP-off counterfactual difference. It includes the changed illumination of downstream HP/Cs and is not a separately measured local aberration coefficient.",
        historical_boundary=dict(directory=str(old_dir.resolve()),file_sha256=files,
            producer_identity=old_snapshot.implementation,input_identity=old_snapshot.digest,
            source_exit_identity=original_report["source_exit_identity"],producer_final_status=original_report["status"],
            physical_input_equalities=equality,
            scope="Explicit historical common entrance distribution; conditional downstream validation, not a fresh full-source run or stale-cache relabelling"),
        refinement="First DPH2 +X target-HP on/off and both baselines repeat at half step from the identical stored boundary. Only this centered sample position response receives a convergence gate; other axes/DPH1/local slopes remain observed at the declared step.",
        mode_output_comparison="DPH2 baseline and +X repeat in Custom from the identical stored boundary; exact numeric plan arrays and actual GPU outputs compared",
        input_identity=snapshot.digest)
    return state,snapshot,protocol,boundaries,weights


def plan_comparison(state,row):
    from temsim.physics.core import build_propagation_plan
    from temsim.physics.instrument_magnetic import active_column_events
    saved=state.simulation_mode;plans=[]
    try:
        for mode in ("ideal","custom"):
            state.simulation_mode=mode
            plans.append(build_propagation_plan(state,row["start_z_mm"],row["sample_z_mm"],
                active_column_events(state),maximum_step_mm=state.step_mm,
                checkpoint_z_mm=row["planes_mm"],save_z_mm=row["planes_mm"]))
    finally:state.simulation_mode=saved
    comparisons={field.name:np.array_equal(getattr(plans[0],field.name),getattr(plans[1],field.name),equal_nan=True)
        for field in fields(plans[0]) if isinstance(getattr(plans[0],field.name),np.ndarray)}
    hp=np.any(plans[0].hex_normal_m3) or np.any(plans[0].hex_skew_m3)
    cs=np.any(plans[0].cs_kick_m3)
    return dict(all_numeric_arrays_identical=all(comparisons.values()),arrays=comparisons,
        ideal_has_nonzero_hexapole=bool(hp),ideal_has_nonzero_cs=bool(cs))


def pack(path,result):
    cp,masks,alive,stops,keys=result
    np.savez_compressed(path,**{name:getattr(cp,name) for name in (
        "z_mm","x_m","y_m","tx_rad","ty_rad","kinetic_energy_ev","flight_time_s")},
        masks=masks,alive=alive,stops_mm=stops,stop_keys=np.asarray(keys,dtype=str))


def phase_at(result,z):
    _,phase=phase_points(result[0],z)
    return phase


def analyse_component(row,hp,cases,weights,incoming):
    from temsim.physics.beam_statistics import transverse_beam_statistics
    # The refinement and mode checks use exactly this same surviving population,
    # including rays lost in either control, rather than silently comparing a
    # stopped ray with a surviving ray.
    live,fraction=common_cohort(cases.values(),incoming,weights)
    w=weights[live];entrance=incoming["coordinates"][:,live]
    out=dict(common_rays=int(live.sum()),common_current_fraction=fraction,planes={},responses=[])
    for plane,z in row["named_planes_mm"].items():
        measured={}
        for control in ("on","off"):
            base=phase_at(cases[f"ideal_{control}_base"],z)[:,live]
            for command in ("base","x+","x-","y+","y-"):
                phase=phase_at(cases[f"ideal_{control}_{command}"],z)[:,live]
                measured[f"{control}_{command}"]=compare_shape(base,phase,w,entrance)
                measured[f"{control}_{command}"]["statistics"]=asdict(transverse_beam_statistics(
                    phase[0],phase[1],phase[2],phase[3],weights=w))
        out["planes"][plane]=dict(z_mm=z,cases=measured)
    for axis in ("x","y"):
        response_planes={}
        for plane,z in row["named_planes_mm"].items():
            plus=phase_at(cases[f"ideal_on_{axis}+"],z)[:,live]
            minus=phase_at(cases[f"ideal_on_{axis}-"],z)[:,live]
            off_plus=phase_at(cases[f"ideal_off_{axis}+"],z)[:,live]
            off_minus=phase_at(cases[f"ideal_off_{axis}-"],z)[:,live]
            odd=(plus-minus)/2;off_odd=(off_plus-off_minus)/2
            odd_c,odd_mean=centered(odd,w);off_c,_=centered(off_odd,w)
            difference=odd_c-off_c
            affine,_=affine_fit(entrance,difference[:2],w)
            response_planes[plane]=dict(odd_centroid_position_m=odd_mean[:2].tolist(),
                on_odd_centered_position_rms_m=weighted_rms(odd_c[:2],w),
                off_odd_centered_position_rms_m=weighted_rms(off_c[:2],w),
                target_hp_specific_centered_position_rms_m=weighted_rms(difference[:2],w),
                target_hp_specific_centered_slope_rms_rad=weighted_rms(difference[2:],w),
                target_hp_specific_phase_space_affine=affine)
        centre=response_planes["hp_centre"];exit=response_planes["hp_field_exit"];sample=response_planes["sample"]
        upstream=response_planes["hp_field_entrance"]
        checks=dict(upstream_unchanged=np.linalg.norm(upstream["odd_centroid_position_m"])<=LIMITS["upstream_centroid_shift_m"],
            beam_moves_inside_hp=np.linalg.norm(centre["odd_centroid_position_m"])>=LIMITS["minimum_inside_hp_centroid_shift_m"],
            target_hp_changes_centred_slope=exit["target_hp_specific_centered_slope_rms_rad"]>=LIMITS["minimum_hp_shape_slope_signal_rad"],
            target_hp_changes_sample_shape=sample["target_hp_specific_centered_position_rms_m"]>=LIMITS["minimum_sample_shape_signal_m"])
        normal,skew=hp.hexapole_strength_components_m3(np.array([hp.z_mm]))
        quadrupole=feeddown_matrix(centre["odd_centroid_position_m"],float(normal[0]),float(skew[0]))
        out["responses"].append(dict(axis=axis,planes=response_planes,checks=checks,
            local_predicted_feeddown_matrix_m2=quadrupole.tolist(),
            prediction_scope="Analytic local quadrupole at HP centre using the executed centroid offset; not a fit of final screen displacement or an integrated whole-system aberration coefficient"))
    out["focus"]={key:surface_measurement(result[0],result[1]&live[None,:],weights,row["sample_planes_mm"])
        for key,result in cases.items() if key.startswith("ideal_") and "fine" not in key}
    out["passed"]=fraction>=LIMITS["minimum_common_current_fraction"] and all(all(r["checks"].values()) for r in out["responses"])
    return out,live


def execute(args,state,snapshot,protocol,boundaries,weights,report):
    from temsim.calculation_manifest import solver_source_identity
    inventory=source_inventory();save_report(args.output/"solver_files.before.json",inventory)
    if solver_source_identity()!=snapshot.implementation:raise ValueError("Source changed after protocol capture")
    report["status"]="RUNNING";save_report(args.output/"report.json",report)
    by_key={component.key:component for component in state.corrector_elements}
    all_results={}
    for ri,row in enumerate(protocol["components"]):
        hp,dph=by_key[row["hp_key"]],by_key[row["dph_key"]]
        mode_check=plan_comparison(state,row)
        if not all(mode_check[k] for k in ("all_numeric_arrays_identical","ideal_has_nonzero_hexapole","ideal_has_nonzero_cs")):
            raise ValueError("New Ideal/Custom analytic nonlinear plan check failed: "+str(mode_check))
        cases={};original=(dph.kick_x_mrad,dph.kick_y_mrad,hp.strength_m3,state.simulation_mode)
        def run(control,command,mode="ideal",fine=False):
            hp.strength_m3=original[2] if control=="on" else 0.;state.simulation_mode=mode
            dx,dy={"base":(0,0),"x+":(1,0),"x-":(-1,0),"y+":(0,1),"y-":(0,-1)}[command]
            dph.kick_x_mrad=original[0]+dx*args.drive_mrad;dph.kick_y_mrad=original[1]+dy*args.drive_mrad
            label=f"{mode}_{control}_{command}"+("_fine" if fine else "")
            print(row["dph_key"]+" "+label,flush=True)
            result=trace(state,row["start_z_mm"],row["sample_z_mm"],**boundaries[row["dph_key"]],
                planes=row["planes_mm"],step=args.step/(2 if fine else 1),receipts=report["backend_receipts"],label=row["dph_key"]+":"+label)
            pack(args.output/(row["dph_key"]+"_"+label+".npz"),result)
            cases[label]=result
            save_report(args.output/"report.json",report)
        try:
            for control in ("on","off"):
                for command in ("base","x+","x-","y+","y-"):run(control,command)
            if ri==0:
                for control in ("on","off"):
                    for command in ("base","x+"):run(control,command,fine=True)
                for command in ("base","x+"):run("on",command,mode="custom")
        finally:dph.kick_x_mrad,dph.kick_y_mrad,hp.strength_m3,state.simulation_mode=original
        metrics,live=analyse_component(row,hp,cases,weights,boundaries[row["dph_key"]])
        metrics["plan_mode_comparison"]=mode_check
        if ri==0:
            w=weights[live];z=row["sample_z_mm"]
            def response(control,suffix=""):
                difference=(phase_at(cases[f"ideal_{control}_x+{suffix}"],z)-phase_at(cases[f"ideal_{control}_base{suffix}"],z))[:2,live]
                return centered(difference,w)[0]
            coarse=response("on")-response("off");fine=response("on","_fine")-response("off","_fine")
            error=weighted_rms(coarse-fine,w);signal=weighted_rms(fine,w)
            metrics["step_refinement"]=dict(absolute_response_difference_m=error,relative_response_difference=error/signal if signal else None,
                refined_signal_rms_m=signal,signal_exceeds_five_step_differences=signal>LIMITS["minimum_refined_signal_to_step_difference"]*error,
                passed=error<=LIMITS["maximum_step_absolute_response_change_m"]+LIMITS["maximum_step_relative_response_change"]*signal and signal>LIMITS["minimum_refined_signal_to_step_difference"]*error)
            comparisons=[]
            for command in ("base","x+"):
                a=phase_at(cases[f"ideal_on_{command}"],z)[:2,live];b=phase_at(cases[f"custom_on_{command}"],z)[:2,live]
                error=weighted_rms(a-b,w);size=weighted_rms(b,w)
                fields_equal={name:np.array_equal(getattr(cases[f"ideal_on_{command}"][0],name),getattr(cases[f"custom_on_{command}"][0],name),equal_nan=True)
                    for name in ("z_mm","x_m","y_m","tx_rad","ty_rad","kinetic_energy_ev","flight_time_s")}
                comparisons.append(dict(command=command,same_survival_masks=np.array_equal(cases[f"ideal_on_{command}"][1],cases[f"custom_on_{command}"][1]),
                    checkpoint_fields_exactly_equal=fields_equal,
                    absolute_position_difference_m=error,passed=all(fields_equal.values()) and error<=LIMITS["mode_absolute_output_difference_m"]+LIMITS["mode_relative_output_difference"]*size))
            metrics["mode_output_comparison"]=comparisons
            metrics["passed"] &= metrics["step_refinement"]["passed"] and all(c["passed"] and c["same_survival_masks"] for c in comparisons)
        report.setdefault("components",[]).append(dict(dph_key=row["dph_key"],metrics=metrics))
        all_results[row["dph_key"]]=(row,cases,live)
        save_report(args.output/"report.json",report)
    after=source_inventory();save_report(args.output/"solver_files.after.json",after)
    if after!=inventory:raise ValueError("Source changed during numerical execution; preserve historical evidence")
    snapshot.verify_current_inputs(state)
    from temsim.instrument_snapshot import capture_instrument_snapshot
    final_snapshot=capture_instrument_snapshot(state)
    save_report(args.output/"inputs.after.json",final_snapshot.to_dict())
    report["final_input_audit"]=physical_snapshot_audit(snapshot,final_snapshot)
    if not report["final_input_audit"]["complete_graph_equal_except_backend_readout"]:
        raise ValueError("Temporary diagnostic controls were not restored to the captured physical inputs")
    old_dir=Path(protocol["historical_boundary"]["directory"])
    if {name:digest(old_dir/name) for name in protocol["historical_boundary"]["file_sha256"]}!=protocol["historical_boundary"]["file_sha256"]:
        raise ValueError("Historical input data changed during the diagnostic")
    report["source_unchanged"]=True
    report["historical_boundary_unchanged"]=True
    report["case_file_sha256"]={p.name:digest(p) for p in sorted(args.output.glob("*.npz"))}
    report["status"]="PASS" if all(r["metrics"]["passed"] for r in report["components"]) else "FAIL"
    render(args.output/"dph_sample_shape.png",all_results,weights)


def render(path,results,weights):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout="constrained")
    for column,(key,(row,cases,live)) in enumerate(results.items()):
        w=weights[live]
        for axis,(control,title) in zip(axes[:,column],(("on","Target HP on"),("off","Target HP off (diagnostic)"))):
            for command,color in (("base","black"),("x+","tab:red"),("x-","tab:blue")):
                phase=phase_at(cases[f"ideal_{control}_{command}"],row["sample_z_mm"])[:2,live]
                positions,_=centered(phase,w)
                axis.scatter(positions[0]*1e9,positions[1]*1e9,s=2,alpha=.25,label=command,c=color)
            axis.set(title=key+" / "+title,xlabel="Centred sample X (nm)",ylabel="Centred sample Y (nm)",aspect="equal")
            axis.legend(markerscale=3);axis.grid(alpha=.2)
    fig.suptitle("Actual CUDA rays — each distribution centred independently; no rescaling or affine removal")
    fig.savefig(path,dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    for column,(key,(row,cases,live)) in enumerate(results.items()):
        w=weights[live];centre=row["named_planes_mm"]["hp_centre"]
        lo=row["named_planes_mm"]["hp_field_entrance"];hi=row["named_planes_mm"]["hp_field_exit"]
        base=cases["ideal_on_base"][0]
        local=np.flatnonzero((base.z_mm>=lo)&(base.z_mm<=hi))
        _,initial_centre=centered(phase_at(cases["ideal_on_base"],centre)[:2,live],w)
        for command,color in (("base","black"),("x+","tab:red"),("x-","tab:blue"),("y+","tab:orange")):
            phase=phase_at(cases[f"ideal_on_{command}"],centre)[:2,live]
            _,beam_centre=centered(phase,w)
            axes[0,column].plot(*(beam_centre-initial_centre)*1e9,"o",color=color,label=command)
            cp=cases[f"ideal_on_{command}"][0]
            dx=(cp.x_m[local][:,live]-base.x_m[local][:,live])@(w/w.sum())
            dy=(cp.y_m[local][:,live]-base.y_m[local][:,live])@(w/w.sum())
            axes[1,column].plot(base.z_mm[local]-centre,dx*1e9,color=color,label=command+" X")
            axes[1,column].plot(base.z_mm[local]-centre,dy*1e9,color=color,ls=":",label=command+" Y")
        axes[0,column].set(title=key+" — HP centre",xlabel="Centroid change X (nm)",ylabel="Centroid change Y (nm)",aspect="equal")
        axes[1,column].axvspan(-row["coil_effective_length_mm"]/2,row["coil_effective_length_mm"]/2,color="grey",alpha=.15,label="DPH active length")
        axes[1,column].set(title="Actual centroid response through HP field",xlabel="Z relative to HP centre (mm)",ylabel="Centroid change (nm)")
        for axis in axes[:,column]:axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.suptitle("Actual CUDA rays — displacement inside the hexapole, relative to the same baseline rays")
    fig.savefig(path.with_name("dph_inside_hp.png"),dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--boundary",type=Path,default=ROOT/"outputs/corrector_dph_20261010/particle_run2")
    parser.add_argument("--rays",type=int,default=3000)
    parser.add_argument("--step",type=float,default=.05)
    parser.add_argument("--drive-mrad",type=float,default=.05)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error("Use a new empty output directory")
    if args.rays<5 or not np.isfinite([args.step,args.drive_mrad]).all() or min(args.step,args.drive_mrad)<=0:
        parser.error("Positive finite step/drive and at least five rays required")
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(status="PREPARING",backend_receipts=[],started_utc=datetime.now(timezone.utc).isoformat())
    started=perf_counter()
    try:
        state,snapshot,protocol,boundaries,weights=prepare(args)
        save_report(args.output/"inputs.json",snapshot.to_dict());save_report(args.output/"protocol.json",protocol)
        for key,boundary in boundaries.items():
            np.savez_compressed(args.output/(key+"_incoming.npz"),weights=weights,
                **{name:np.asarray(value,dtype=str) if name=="keys" else value for name,value in boundary.items()})
        report.update(status="PREPARED_NOT_EXECUTED",protocol=protocol)
        if args.execute:execute(args,state,snapshot,protocol,boundaries,weights,report)
    except Exception as exc:
        report.update(status="FAILED",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_s"]=perf_counter()-started;save_report(args.output/"report.json",report)
        print(json.dumps(dict(status=report["status"],output=str(args.output.resolve()))),flush=True)
    return 0 if report["status"] in ("PREPARED_NOT_EXECUTED","PASS") else 2


if __name__=="__main__":raise SystemExit(main())
