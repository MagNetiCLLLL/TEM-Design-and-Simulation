"""Frozen-optics count/step observations with regenerated complete sources."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.model import State
from temsim.optics.beam_path_audit import incident_checkpoints, crossover_candidates
from temsim.optics.condenser_topology import _strict_spatial_minima
from temsim.optics.surface_probe_focus import local_waist_from_radii
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.corrector_crossovers import measured_corrector_crossovers
from temsim.physics.beam_current import effective_source_current_a
from temsim.physics.particle_sections import _digest_arrays


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [clean(v) for v in value]
    if isinstance(value,np.ndarray):
        return clean(value.tolist())
    if isinstance(value,np.generic):
        return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def physical_input_signature(inputs):
    """Exclude only explicit numerical source counts and column step budget."""
    def strip(value):
        if isinstance(value,dict):
            return {k:strip(v) for k,v in value.items() if k != "ray_count"}
        if isinstance(value,list):
            return [strip(v) for v in value]
        return value
    data=strip(deepcopy(inputs))
    data.pop("step_mm",None)
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()


def execution_identity(snapshot):
    """Keep implementation and content-pinned external inputs separate from N."""
    external = snapshot.to_dict()["external_inputs"]
    return dict(solver=snapshot.implementation, external_inputs=hashlib.sha256(
        json.dumps(external, sort_keys=True, allow_nan=False).encode()).hexdigest())


def require_finite_population(checkpoints, masks, weights):
    """Do not let statistics silently discard a live nonfinite trajectory."""
    weights = np.asarray(weights, float)
    masks = np.asarray(masks, bool)
    if (weights.ndim != 1 or masks.ndim != 2 or masks.shape[1] != len(weights)
            or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError("Invalid original current measure")
    live = masks & (weights > 0)[None, :]
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        values = np.asarray(getattr(checkpoints, name))
        if values.shape != masks.shape or not np.isfinite(values[live]).all():
            raise ValueError(f"Nonfinite or inconsistent live observation: {name}")


def compare_counts(first, second, *, angle_relative=.0025, size_relative=.02, focus_absolute_nm=.25):
    """Report failures; do not retune or relax tolerances to force convergence."""
    if first["physical_input_signature"] != second["physical_input_signature"]:
        raise ValueError("Count comparisons require identical physical inputs")
    if first["step_mm"] != second["step_mm"]:
        raise ValueError("Count comparison must not confound integration step")
    if not first["rays"] < second["rays"]:
        raise ValueError("Count refinement must increase the complete source population")
    limits={"alpha95_mrad":angle_relative,"rms_radius_nm":size_relative,
            "diameter95_nm":size_relative,"current_a":size_relative}
    metrics={}
    for key,limit in limits.items():
        a,b=first["specimen"][key],second["specimen"][key]
        if a is None or b is None or not np.isfinite([a,b]).all() or a <= 0 or b <= 0:
            metrics[key]=dict(passed=False,reason="nonpositive_or_nonfinite_reference")
        else:
            delta=abs(b-a)/abs(b)
            metrics[key]=dict(first=a,second=b,relative_change=delta,limit=limit,passed=delta<=limit)
    a,b=first["specimen"]["local_focus_offset_nm"],second["specimen"]["local_focus_offset_nm"]
    finite=a is not None and b is not None and np.isfinite([a,b]).all()
    metrics["local_focus_offset_nm"]=dict(first=a,second=b,
        absolute_change_nm=abs(b-a) if finite else None,limit_nm=focus_absolute_nm,
        passed=bool(finite and abs(b-a)<=focus_absolute_nm))
    identity = first.get("execution_identity")
    same_execution = (isinstance(identity, dict) and bool(identity.get("solver"))
        and bool(identity.get("external_inputs")) and identity == second.get("execution_identity"))
    metric_pass = all(row["passed"] for row in metrics.values())
    return dict(first_count=first["rays"],second_count=second["rays"],metrics=metrics,
        numerical_metrics_passed=metric_pass,
        execution_identity_status="MATCH" if same_execution else "MISSING_OR_DIFFERENT",
        passed=bool(metric_pass and same_execution),
        definition="Alpha95 uses its nonzero finer-count value as scale; focus uses an absolute budget. Pupil-edge30mrad is a separate measurement.")


def observe(inputs, rays, step, output, *, roots=(503.2,721.,1202.,1400.,1585.9)):
    from temsim.cpu_resources import numerical_thread_budget
    started=perf_counter()
    state=State.from_dict(inputs)
    if state.to_dict()!=inputs:
        raise ValueError("Restoring inputs changed physical state")
    state.electron_gun.emitter.ray_count=rays
    state.step_mm=step
    modified=state.to_dict()
    signature=physical_input_signature(inputs)
    if physical_input_signature(modified)!=signature:
        raise ValueError("Numerical refinement changed a physical parameter")
    snapshot=capture_instrument_snapshot(state)
    start,end=state.electron_gun.exit_plane_z_mm,state.sample.upper_surface_z_mm
    focus=(end-.0002,end-.0001,end)
    windows=[np.linspace(root-1.,root+1.,41) for root in roots]
    planes=np.unique(np.r_[start,np.arange(start,end,2.),*windows,*roots,*focus])
    cpu_budget=numerical_thread_budget()
    print(json.dumps(dict(stage="start",rays=rays,step_mm=step,observations=len(planes),
                         cpu_thread_budget=cpu_budget)),flush=True)
    gun,cp,masks=incident_checkpoints(state,planes,step_mm=step)
    emitted=gun.exit_bundle
    weights=np.asarray(emitted.weight)
    require_finite_population(cp,masks,weights)
    current=effective_source_current_a(state)
    identity=_digest_arrays(*(np.asarray(getattr(emitted,key)) for key in
        ("x_m","tx_rad","y_m","ty_rad","flight_time_s","energy_offset_ev","weight","alive","ray_id")))
    def point(index,cohort=None):
        mask=masks[index] if cohort is None else cohort
        stats=transverse_beam_statistics(*(getattr(cp,k)[index] for k in ("x_m","y_m","tx_rad","ty_rad")),alive=mask,weights=weights)
        return dict(z_mm=float(cp.z_mm[index]),alpha95_mrad=stats.convergence_95_mrad,
            alpha99_mrad=stats.convergence_99_mrad,rms_radius_nm=stats.radius_rms_m*1e9,
            diameter95_nm=2*stats.radius_95_m*1e9,current_a=current*float(weights[mask].sum()/weights.sum()),
            covariance_focus_offset_nm=stats.waist_offset_m*1e9,statistics=asdict(stats))
    specimen=point(len(cp.z_mm)-1)
    indexes=np.searchsorted(cp.z_mm,focus)
    radii=[point(int(i),masks[-1])["rms_radius_nm"]*1e-9 for i in indexes]
    offset,second=local_waist_from_radii(radii,1e-7)
    specimen.update(local_focus_offset_nm=offset,variance_second_derivative=second,
                    focus_role="Three executed upstream planes; local variance derivative, not a downstream field-free projection")
    common=np.all(masks,axis=0)&(weights>0)
    waists=[]
    for root in roots:
        indices=np.flatnonzero((cp.z_mm>=root-1.-1e-9)&(cp.z_mm<=root+1.+1e-9))
        local=SimpleNamespace(z_mm=cp.z_mm[indices],**{k:getattr(cp,k)[indices] for k in ("x_m","y_m","tx_rad","ty_rad")})
        found=_strict_spatial_minima(local,weights,measured_corrector_crossovers(local,masks[indices],weights,root-1.001,root+1.001))
        waists.append(dict(reference_z_mm=root,executed_waists=[dict(row,
            actual=point(int(np.searchsorted(cp.z_mm,row["z_mm"]))),
            common=point(int(np.searchsorted(cp.z_mm,row["z_mm"])),common)) for row in found],
            fixed_plane=point(int(np.searchsorted(cp.z_mm,root))),
            candidates=crossover_candidates(local,masks[indices],weights)))
    if capture_instrument_snapshot(state).digest!=snapshot.digest:
        raise ValueError("Observation changed its frozen inputs")
    result=dict(schema="particle-count-observation-v1",utc=datetime.now(timezone.utc).isoformat(),
        rays=rays,step_mm=step,elapsed_s=perf_counter()-started,physical_input_signature=signature,
        cpu_thread_budget=cpu_budget,
        execution_identity=execution_identity(snapshot),
        snapshot_digest=snapshot.digest,source_population_identity=identity,
        source=dict(emission_mean_ev=state.electron_gun.emitter.emission_energy_ev,
            energy_spread_fwhm_ev=state.electron_gun.emitter.energy_spread_fwhm_ev,
            total_original_weight=float(weights.sum()),emitted_current_a=current,
            gun_exit_current_a=current*float(weights[np.asarray(emitted.alive)].sum()/weights.sum())),
        specimen=specimen,waist_windows=waists,
        sampling=dict(maximum_spacing_mm=float(np.diff(cp.z_mm).max()),local_waist_spacing_mm=.05),
        interpretation="Complete freshly regenerated source for each count; no lens/source-distribution fitting, pruning or current renormalization")
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError("Preserve completed convergence observations")
    snapshot_path=output.with_suffix(".snapshot.json")
    if snapshot_path.exists() and json.loads(snapshot_path.read_text(encoding="utf-8")) != snapshot.to_dict():
        raise ValueError("Existing snapshot differs from this observation")
    snapshot_path.write_text(json.dumps(snapshot.to_dict(),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    output.write_text(json.dumps(clean(result),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    output.with_suffix(".inputs.json").write_text(json.dumps(modified,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"),z_mm=cp.z_mm,x_m=cp.x_m,y_m=cp.y_m,
        tx_rad=cp.tx_rad,ty_rad=cp.ty_rad,flight_time_s=cp.flight_time_s,
        kinetic_energy_ev=cp.kinetic_energy_ev,alive=masks,weight=weights,
        source_id=emitted.ray_id,source_energy_offset_ev=emitted.energy_offset_ev)
    print(json.dumps(clean(dict(rays=rays,step_mm=step,elapsed_s=result["elapsed_s"],specimen=specimen))),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--rays",type=int,required=True)
    parser.add_argument("--step",type=float,default=.05)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cpu-threads",type=int,default=None,
        help="Bound numerical workers; existing stricter environment limits are respected")
    args=parser.parse_args()
    if args.rays<16 or args.rays>50000 or not np.isfinite(args.step) or args.step<=0:
        parser.error("Use 16-50000 particles and a finite positive column step")
    if args.output.exists():
        parser.error("Output already exists; preserve completed evidence")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("CPU thread budget must be positive")
    from temsim.cpu_resources import numerical_job
    from temsim.physics.transport_progress import transport_progress
    last_print,last_stage=0.,""
    def progress(label):
        nonlocal last_print,last_stage
        now=perf_counter()
        stage="|".join(str(label).split("|")[:2])
        if stage!=last_stage or now-last_print>=5 or str(label).endswith("Completed"):
            local=getattr(label,"local_progress",None)
            print(json.dumps(dict(stage="transport",detail=str(label),
                utc=datetime.now(timezone.utc).isoformat(),
                completed=None if local is None else local.completed,
                total=None if local is None else local.total)),flush=True)
            last_print,last_stage=now,stage
    with numerical_job(args.cpu_threads), transport_progress(progress):
        observe(json.loads(args.inputs.read_text(encoding="utf-8")),args.rays,args.step,args.output)


if __name__=="__main__":
    main()
