"""Persist complete static-column observations at native solver planes."""
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

import numpy as np
from temsim.physics.particle_sections import _events, _trace_segment
from temsim.physics.transport_progress import transport_progress

PHASE=("x_m","tx_rad","y_m","ty_rad","flight_time_s","kinetic_energy_ev")


def capture_blocks(state, gun, bundle, identity, work, snapshot, surface, plane_metrics, observed_minimum, waist_separation):
    weights=np.asarray(bundle.weight)
    start=float(state.electron_gun.exit_plane_z_mm)
    initial=tuple(getattr(bundle,key) for key in PHASE[:5])
    kinetic=state.electron_gun.nominal_exit_energy_ev+bundle.energy_offset_ev
    survival=(bundle.alive,gun.blocked_z_mm,gun.blocked_key)
    events=_events(state)
    output=Path(work)
    block_dir=output/"blocks"
    block_dir.mkdir(exist_ok=True)
    upper=surface+2.
    # Observe the whole column for extra crossovers; refine RMS and D95 separately.
    near=np.unique(np.r_[-np.geomspace(.25,2e6,120),0.,np.geomspace(.25,2e6,120)])
    grid=np.unique(np.r_[np.arange(start,upper,.5),surface+near*1e-6])
    detailed=len(weights)>=2048
    if detailed:
        from temsim.optics.waist_evidence import requested_corrector_planes
        from temsim.optics.crossover_acceptance import regions_for_state
        from temsim.optics.direct_focus_topology import requested_capture_planes
        boundaries=[z for region in regions_for_state(state).values() for z in region["bounds_mm"]]
        grid=np.unique(np.r_[grid,boundaries,requested_corrector_planes(state),requested_capture_planes(state)])
        (output/"phase").mkdir(exist_ok=True)
    edges=np.r_[np.arange(start,surface-2.,25.),surface-2.,upper]
    all_rows=[]
    prefix=sha256(b"".join(np.ascontiguousarray(arr).tobytes() for arr in
        (*initial,kinetic,weights,identity[0],survival[0]))).hexdigest()
    column_signature=sha256((Path(__file__).read_text()+snapshot.digest).encode()).hexdigest()
    last=[0.]
    def progress(message):
        if monotonic()-last[0]>60.:
            print(message,flush=True)
            last[0]=monotonic()

    for block,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
        record=block_dir/f"{block:03d}.json"
        path=record.with_suffix(".npz")
        final=block==len(edges)-2
        if record.is_file():
            meta=json.loads(record.read_text())
            if (meta["signature"]!=column_signature or meta["parent"]!=prefix
                    or meta["bounds"]!=[low,high] or sha256(path.read_bytes()).hexdigest()!=meta["sha256"]):
                raise ValueError("Saved column block differs from these inputs, implementation or predecessor")
            with np.load(path,allow_pickle=False) as data:
                initial=tuple(data[key] for key in PHASE[:5])
                kinetic=data[PHASE[5]]
                survival=(data["alive"],data["blocked_z"],data["blocked_key"])
            all_rows.extend(meta["profile"][int(block>0):])
            prefix=meta["sha256"]
            if final:
                return meta["focus"],all_rows
            continue
        selected=grid[(grid>=low)&(grid<=high)]
        runner=dict(state=state,name="incident",start=low,stop=high,initial=initial,
                    energy=bundle.energy_offset_ev,weights=weights,identity=identity,
                    survival=survival,events=events,boundary=np.inf,initial_kicks=block==0,
                    initial_kinetic_energy_ev=kinetic)
        previous=None
        passes=3 if final else 1
        for refinement in range(passes):
            with transport_progress(progress):
                segment,_,_=_trace_segment(**runner,previous=previous,save_z_mm=selected)
            cp=segment.checkpoints
            if not np.array_equal(segment.branch.source_ray_id,identity[0]) or not np.array_equal(segment.branch.ray_weight,weights):
                raise ValueError("Source identity or absolute current changed in a block")
            rows=[plane_metrics(cp,i,weights,gun.emitted_current_a) for i in range(len(cp.z_mm))]
            if not final:
                break
            i=np.flatnonzero(cp.z_mm==surface).item()
            if not all(row["surviving"] for row in rows):
                focus=dict(surface=rows[i],status="No transmitted beam throughout focus window")
                break
            z=cp.z_mm
            stable=all(np.array_equal(cp.unified_state.active[0],a) for a in cp.unified_state.active)
            d95,rms=[observed_minimum(z,[row[key] for row in rows],surface) for key in ("D95_nm","rms_radius_nm")]
            focus=dict(surface=rows[i],local_D95=d95,local_RMS=rms,search_window_mm=[low,high],
                stable_membership=stable,separation=waist_separation(rms,d95,stable=stable,support=True),refinement=refinement)
            extra=[]
            for minimum in (d95,rms):
                a,b=minimum["neighbour_bracket_mm"]
                if minimum["status"]=="BOUNDED_OBSERVED_MINIMUM" and b-a>.5e-6:
                    extra.extend(np.linspace(a,b,51))
            if not extra or refinement==passes-1:
                break
            selected=np.unique(np.r_[selected,extra])
            previous=segment
        if detailed:
            phase_file=output/"phase"/f"{block:03d}.npz"
            tmp_phase=phase_file.with_suffix(".part.npz")
            np.savez_compressed(tmp_phase,z_mm=cp.z_mm,source_id=identity[0],weights=weights,
                active=cp.unified_state.active,input_digest=snapshot.digest,
                **{k:getattr(cp,k) for k in ("x_m","y_m","tx_rad","ty_rad")})
            tmp_phase.replace(phase_file)
        initial=tuple(getattr(cp,key)[-1] for key in PHASE[:5])
        kinetic=cp.kinetic_energy_ev[-1]
        branch=segment.branch
        survival=(branch.alive,branch.blocked_z,np.asarray(branch.blocked_key,dtype=str))
        if not np.array_equal(survival[0],cp.unified_state.active[-1]):
            raise ValueError("Native end-plane and mechanical stopping ledger disagree")
        tmp=path.with_suffix(".part.npz")
        np.savez_compressed(tmp,**dict(zip(PHASE,(*initial,kinetic))),alive=survival[0],
                            blocked_z=survival[1],blocked_key=survival[2])
        tmp.replace(path)
        meta=dict(signature=column_signature,parent=prefix,bounds=[low,high],
                  sha256=sha256(path.read_bytes()).hexdigest(),field_digest=segment.plan.unified_runtime.snapshot.numerical_digest,
                  profile=rows)
        if final:
            i=np.flatnonzero(cp.z_mm==surface).item()
            np.savez_compressed(output/"surface.npz",**{key:getattr(cp,key)[i] for key in PHASE},
                ray_id=identity[0],weight=weights,alive=cp.unified_state.active[i],z_mm=surface,
                emitted_current_a=gun.emitted_current_a,input_digest=snapshot.digest)
            meta["focus"]=focus
        tmp=record.with_suffix(".part.json")
        tmp.write_text(json.dumps(meta,allow_nan=False),encoding="utf-8")
        tmp.replace(record)
        all_rows.extend(rows[int(block>0):])
        prefix=meta["sha256"]
        print("Completed native block",block,high,"mm, surviving:",rows[-1]["surviving"],flush=True)
    return focus,all_rows
