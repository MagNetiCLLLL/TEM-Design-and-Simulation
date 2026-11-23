"""Locate changes in the current-bearing cone using matched, unperturbed particle identities."""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from material_trial import HERE,PARENT,WORK,Column,Field,source_and_electrodes
from relay_transfer import advance
from match import transfer
from screen import FACTOR


def measure(p,w):
    axis_r=np.linalg.norm(p[:,:2],axis=1)
    alpha=1000*np.arctan(np.linalg.norm(p[:,2:4],axis=1))
    centred=p[:,:4]-np.average(p[:,:4],axis=0,weights=w)
    def quant(v,f):
        ix=np.argsort(v)
        return float(v[ix[np.searchsorted(w[ix].cumsum(),f*w.sum())]])
    return dict(axis_r95_um=1e6*quant(axis_r,.95),axis_rmax_um=1e6*axis_r.max(),
                rms_r_um=1e6*np.sqrt(w@np.sum(centred[:,:2]**2,axis=1)/w.sum()),
                alpha50_mrad=quant(alpha,.5),alpha95_mrad=quant(alpha,.95),
                angle_max_mrad=alpha.max(),rms_angle_mrad=1e3*np.sqrt(w@np.sum(centred[:,2:4]**2,axis=1)/w.sum()))


def run(folder,fdef):
    report=json.loads((folder/"tracking.json").read_text())
    field=Field(WORK/report["field"]["source"],step=report["field"]["step_mm"])
    gun=Path(report["gun_source"])
    native=dict(report["surface"])
    with np.load(gun) as raw:
        prep={name:measure(raw[k],raw["weight"]) for name,k in
              [("emission","launch"),("accelerator_exit","phase")]}
        n=len(raw["ids"])
        ids=raw["ids"].copy()
    roots=report["waists_mm"][:4]
    locations=dict(C1_waist=roots[0],C2_waist=roots[1],C2_foil_in=764.9,C2_foil_out=765.1,
                   C3_entrance=830.,C3_center=915.,C3_foil=1009.9,ADL_center=1088.5,
                   HP2=1144.8,TL22=1166.35,TL_gap_waist=roots[2],TL21=1249.95,HP1=1301.8,
                   TL12=1366.3,scan_waist=roots[3],Mini=1522.5,upper_pole_entrance=1534.5,
                   gap_entrance=1596.5,specimen=report["surface_z_mm"])
    model=Column(fdef,field,gun,n,report["step_mm"],observation_mm=list(locations.values()))
    pairs=sorted((int(model.indices([z*.001])[0]),name) for name,z in locations.items())
    stations=[i for i,_ in pairs]
    states=advance(model.t,model.full,model.t.input,model.en,model.phi0,model.loss,0,stations)
    w=model.t.weights
    ph={}
    for (i,name),(p,lost) in zip(pairs,states,strict=True):
        live=lost==-1
        if np.any(lost < -1):
            raise ValueError("A numerical field exit occurred")
        prep[name]=dict(z_mm=model.m.z[i]*1000,count=int(live.sum()),current_uA=model.t.current*w[live].sum()*1e6,
                        **measure(p[live],w[live]))
        ph[name]=p[live,:4].copy()
    np.testing.assert_allclose(prep["specimen"]["alpha95_mrad"],native["alpha95_mrad"],rtol=1e-5)
    np.testing.assert_allclose(prep["specimen"]["current_uA"],native["specimen_uA"],rtol=1e-12)
    for key in ("emission","accelerator_exit"):
        prep[key].update(count=n,current_uA=model.t.current*1e6,z_mm=0. if key=="emission" else 450.)
    round_b=CubicSpline(field.z,field.c[:,0,2])
    z=model.m.z
    mat=transfer(z,FACTOR*round_b((z[1:]+z[:-1])*.5))
    def transport(name):
        i=stations[[n for _,n in pairs].index(name)]
        downstream=mat[-1].reshape(2,2) @ np.linalg.inv(mat[i].reshape(2,2))
        a,b,c,d=downstream.ravel()
        p=ph[name]
        x=p[:,0]+1j*p[:,1]
        v=p[:,2]+1j*p[:,3]
        canon=v-1j*FACTOR*round_b(z[i])*x
        # A common Larmor rotation cancels out of the mechanical angle magnitude.
        from_x=(c+1j*FACTOR*round_b(z[-1])*a)*x
        from_u=(d+1j*FACTOR*round_b(z[-1])*b)*canon
        angle=from_x+from_u
        rms=lambda u:float(np.sqrt(w@abs(u)**2/w.sum())*1e3)
        observed=measure(ph["specimen"],w)
        return dict(z_mm=1000*z[i],A=a,B_mm=1000*b,C_per_m=c,D=d,
                    from_position_rms_mrad=rms(from_x),from_canonical_slope_rms_mrad=rms(from_u),
                    summed_angle_rms_mrad=rms(angle),native_angle_rms_mrad=observed["rms_angle_mrad"],
                    lever_arm_mm=1000*abs(b),backprojected_30mrad_radius_um=1e6*abs(b)*np.tan(.03))
    return dict(source_count=n,surface=native,stations=prep,source_ids=ids,
                transfer={name:transport(name) for name in ("C2_foil_in","scan_waist","Mini","upper_pole_entrance")})


def diagram(records):
    fig,(r,ang)=plt.subplots(2,1,figsize=(10,6.4),layout="constrained")
    palette={"parent":"#727a8b","objective-only":"#bf7029","relay-edge":"#006d79"}
    names={"parent":"Joint FEM baseline","objective-only":"Objective refocus","relay-edge":"Crossover-adjusted"}
    stages=["emission","accelerator_exit","C2_foil_in","C3_foil","TL_gap_waist","scan_waist","Mini","specimen"]
    labels=["Effective\nsource","Gun exit\n450 mm","C2 foil\n764.9 mm","C3 foil\n1009.9 mm",
            "TL22–TL21\nwaist","Post-scan\nwaist","Mini centre\n1522.5 mm","Specimen\n1599.2 mm"]
    for k,rec in records.items():
        if k not in palette:
            continue
        for ax,item in ((r,"axis_r95_um"),(ang,"alpha95_mrad")):
            ax.plot(range(len(stages)),[rec["stations"][i][item] for i in stages],
                    color=palette[k],marker="o",ms=5,lw=1.8,label=names[k])
    ang.axhline(30.,color="#a23f48",ls="--",lw=1.)
    ang.annotate("30 mrad target",xy=(7,30),xytext=(-4,3),textcoords="offset points",ha="right",color="#a23f48")
    r.scatter([2],[50],marker="_",s=240,linewidths=2,color="#a23f48")
    r.annotate("50 µm C2 aperture radius",xy=(2,50),xytext=(8,0),textcoords="offset points",
               color="#a23f48",va="center")
    for ax in (r,ang):
        ax.set_yscale("log");ax.grid(axis="y",alpha=.22)
        ax.set_xticks(range(len(labels)),labels)
        ax.spines[["top","right"]].set_visible(False)
    r.legend(ncol=3,frameon=False,loc="lower center",bbox_to_anchor=(.5,1.03))
    r.set_ylabel("95% current radius (µm)")
    ang.set_ylabel("95% current semi-angle (mrad)")
    fig.suptitle("Beam radius and angular spread along the column",fontsize=15,weight="bold")
    fig.savefig(HERE/"current-angle-by-station.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    f=source_and_electrodes()
    emitter=f.state.electron_gun.emitter
    results={}
    first_ids=None
    for label,folder in [("parent",PARENT),("objective-only",HERE/"objective-only"),("relay-edge",HERE/"relay-edge")]:
        results[label]=run(folder,f)
        ids=results[label].pop("source_ids")
        if first_ids is None:
            first_ids=ids
        np.testing.assert_array_equal(first_ids[:len(ids)],ids)
        print(label,results[label]["transfer"]["Mini"],flush=True)
    diagram(results)
    result=dict(launch_energy_mean_ev=emitter.emission_energy_ev,accelerator_voltage_kv=300.,
                emitter=dict(virtual_source_fwhm_nm=emitter.virtual_source_fwhm_nm,
                             launch_angular_sigma_mrad=emitter.angular_rms_mrad,
                             launch_angular_cutoff_mrad=emitter.angular_cutoff_mrad,
                             curvature_nm_inv=emitter.curvature_nm_inv),
                angular_measurement="Weighted quantiles of atan(hypot(dx/dz, dy/dz)) around the fixed column axis",
                runs=results)
    (HERE/"angle-origin.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
