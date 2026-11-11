"""Audit saved original-current particles; never trace, filter or retune."""
from pathlib import Path
from hashlib import sha256
import argparse
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import numpy as np
from temsim.physics.beam_statistics import transverse_beam_statistics,_weighted_quantile
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.optics.beam_path_audit import crossover_intervals,optical_component_planes
from validate_joint_probe import _restore_exact


def statistics(arrays,row,keep,current):
    w=arrays["weights"][keep]
    s=transverse_beam_statistics(*(arrays[key][row] for key in ("x_m","y_m","tx_rad","ty_rad")),
                                alive=keep,weights=arrays["weights"])
    return dict(z_mm=float(arrays["z_mm"][row]),rms_radius_um=float(s.radius_rms_m*1e6),
        d95_um=float(s.radius_95_m*2e6),alpha95_mrad=float(s.convergence_95_mrad),
        alpha_rms_mrad=float(s.convergence_rms_rad*1e3),count=int(keep.sum()),
        effective_count=float(w.sum()**2/np.dot(w,w)),current_a=float(current*w.sum()),
        centroid_xy_um=[float(s.mean_x_m*1e6),float(s.mean_y_m*1e6)],
        mean_slopes_mrad=[float(s.mean_tx_rad*1e3),float(s.mean_ty_rad*1e3)])


def audit(case):
    receipt=json.loads((case/"report.json").read_text())
    frozen=json.loads((case/"inputs.json").read_text())
    snapshot=InstrumentSnapshot.from_dict(frozen)
    state=_restore_exact(snapshot)
    boundaries={part.key:float(part.z_mm) for part in (*state.lenses,*state.corrector_elements)}
    boundaries.update(gun_exit=float(state.electron_gun.exit_plane_z_mm),
        ac_upper=float(state.ac_deflector.upper_z_mm),ac_lower=float(state.ac_deflector.lower_z_mm),
        specimen_entrance=float(state.sample.upper_surface_z_mm))
    if snapshot.digest!=receipt["execution"]["input_digest"]:
        raise ValueError("Executed input snapshot mismatch")
    if sha256((case/"observations.npz").read_bytes()).hexdigest()!=receipt["observations_sha256"]:
        raise ValueError("Completed native observation hash mismatch")
    with np.load(case/"observations.npz",allow_pickle=False) as data:
        arrays={key:data[key] for key in ("x_m","y_m","tx_rad","ty_rad","alive","weights","source_id","z_mm")}
    z,weights=arrays["z_mm"],arrays["weights"]
    if np.count_nonzero(weights>0.)<1000 or len(np.unique(arrays["source_id"]))!=len(weights):
        raise ValueError("Full original positive-current population is required")
    surface=int(np.searchsorted(z,receipt["surface"]["z_mm"]))
    current=receipt["execution"]["source_current_a"]
    final=arrays["alive"][surface]&(weights>0.)
    waists=[]
    curves=[]
    for row in range(len(z)):
        keep=arrays["alive"][row]&(weights>0.)
        w=weights[keep]
        xy=np.c_[arrays["x_m"][row,keep],arrays["y_m"][row,keep]]
        centered=xy-np.average(xy,axis=0,weights=w)
        radius=np.linalg.norm(centered,axis=1)
        curves.append((np.sqrt(np.dot(w,radius**2)/w.sum())*1e6,
                       _weighted_quantile(radius,w,.95)*2e6,current*w.sum(),keep.sum()))
    curves=np.asarray(curves)
    np.savez_compressed(case/"beam-envelope.npz",z_mm=z,rms_radius_um=curves[:,0],d95_um=curves[:,1],
                        current_a=curves[:,2],surviving_count=curves[:,3].astype(int))
    for label,populations in receipt["topology"].items():
        for kind,evidence in populations.items():
            for region in evidence["regions"]:
                center=region["representative_z_mm"]
                if len(evidence["regions"])==1:
                    lower,upper=(boundaries[key] for key in label.split(" to "))
                    rows=np.flatnonzero((z>=lower)&(z<=upper))
                else:
                    rows=np.flatnonzero(np.abs(z-center)<=.75)
                metrics=[]
                for row in rows:
                    keep=arrays["alive"][row]&(weights>0.)
                    if kind=="sample_surviving_cohort":
                        keep &= final
                    metrics.append(statistics(arrays,int(row),keep,current))
                ir=int(np.argmin([item["rms_radius_um"] for item in metrics]))
                id95=int(np.argmin([item["d95_um"] for item in metrics]))
                rr,dd=int(rows[ir]),int(rows[id95])
                waists.append(dict(interval=label,population=kind,observed_region_bracket_mm=region["bracket_mm"],
                    rms_minimum=metrics[ir],d95_minimum=metrics[id95],
                    rms_native_bracket_mm=z[max(rr-1,0):rr+2].tolist(),
                    d95_native_bracket_mm=z[max(dd-1,0):dd+2].tolist(),
                    d95_search_resolved=id95 not in (0,len(metrics)-1),
                    d95_minus_rms_z_um=float((z[dd]-z[rr])*1e3)))
    native_surface=statistics(arrays,surface,final,current)
    for key in ("d95_um","alpha95_mrad","alpha_rms_mrad","current_a"):
        if not np.isclose(native_surface[key],receipt["surface"][key],rtol=1e-12,atol=1e-15):
            raise ValueError("Recomputed surface differs from original receipt")
    xy=np.c_[arrays["x_m"][surface,final],arrays["y_m"][surface,final]]
    tt=np.c_[arrays["tx_rad"][surface,final],arrays["ty_rad"][surface,final]]
    w=weights[final]
    center=np.average(xy,axis=0,weights=w)
    radius=np.linalg.norm(xy-center,axis=1)
    direction=np.c_[tt,np.ones(len(tt))]
    direction/=np.linalg.norm(direction,axis=1)[:,None]
    chief=np.average(direction,axis=0,weights=w);chief/=np.linalg.norm(chief)
    angle=np.arctan2(np.linalg.norm(np.cross(direction,chief),axis=1),direction@chief)
    ids=arrays["source_id"][final]
    tails=[]
    for i in np.argsort(radius)[-5:][::-1]:
        tails.append(dict(source_id=int(ids[i]),radius_um=float(radius[i]*1e6),
            absolute_axis_radius_um=float(np.linalg.norm(xy[i])*1e6),
            angle_mrad=float(angle[i]*1e3),original_current_a=float(current*w[i])))
    actual=sorted([item["rms_minimum"]["z_mm"] for item in waists if item["population"]=="actual_population"])
    report=dict(scope="Postanalysis of complete unchanged native observations",input_digest=snapshot.digest,
        original_positive_count=int(np.count_nonzero(weights>0.)),surface=native_surface,
        actual_waist_count=len(actual),waists=waists,
        ordered_nearest_component_intervals=crossover_intervals(actual,optical_component_planes(state)),
        current_losses=receipt["per_stage_losses"],current_balance_residual_a=receipt["absolute_current_balance_residual_a"],
        tail_original_ids=tails,max_radius_to_r95=float(radius.max()/(native_surface["d95_um"]*.5e-6)),
        max_angle_to_alpha95=float(angle.max()/(native_surface["alpha95_mrad"]*1e-3)),
        minimum_definition="RMS radius and D95 minima reported separately on actual native planes",
        d95_search="Entire optical interval when it has one resolved RMS source-image region; otherwise a local 0.75 mm neighborhood",
        actual_observations_sha256=receipt["observations_sha256"])
    out=case/"waist-audit.json"
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(dict(case=str(case),waists=len(actual),surface=native_surface,
        max_radius_to_r95=report["max_radius_to_r95"],max_angle_to_alpha95=report["max_angle_to_alpha95"])),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",type=Path,nargs="+")
    args=parser.parse_args()
    for case in args.case:
        audit(case.resolve())
