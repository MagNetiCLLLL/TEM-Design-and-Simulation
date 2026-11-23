"""Draw the recorded downstream trajectories and the absolute current density at the specimen."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from material_relay_fields import SAVE, parent
from relay_transfer import PARTS,START


def render(folder, *, scan_label=None, reference_label=None):
    r=json.loads((folder/"tracking.json").read_text())
    with np.load(folder/"column-and-landings.npz") as file:
        z,line,stats=(file[k] for k in ("z_mm","paths_m","stats"))
        ok=file["loss"]==-1
        xy=file["surface_phase"][ok,:2]*1e9
        current=file["source_A"]*file["weights"][ok]*1e9
        selected=file["selected_ids"]
    source_file=folder/"effective-source-and-exit.npz"
    with np.load(source_file if source_file.is_file() else r["gun_source"]) as source:
        selected_indices=np.linspace(0,len(source["ids"])-1,len(selected),dtype=int)
        np.testing.assert_array_equal(source["ids"][selected_indices],selected)
        launch=source["launch"][selected_indices]
    colors=plt.cm.twilight_shifted((np.arctan2(launch[:,3],launch[:,2])+np.pi)/(2*np.pi))
    surface=r["surface"];fronts=r.get("isolated_pair_fronts",r["operational_fronts"])
    front=min(fronts,key=lambda f:abs(f["z_mm"]-1450))["z_mm"] if fronts else None
    waists=[a for a in r["waists_mm"] if 1420<a<1512.5]
    waist=min(waists,key=lambda a:abs(a-(front or 1450))) if waists else None
    previous_waists=(json.loads((SAVE/reference_label/"tracking.json").read_text())["waists_mm"]
                     if reference_label else parent()["all_radial_waists_mm"])
    previous=next((a for a in reversed(previous_waists) if 1420<a<1512.5),None)
    scan_record=json.loads((SAVE/scan_label/"tracking.json").read_text()) if scan_label else r
    if scan_label and scan_record["field"]["solution_sha256"]!=r["field"]["solution_sha256"]:
        raise ValueError("The displayed pivot must use the displayed static field")
    pivots=scan_record.get("pivot",{}).get("candidates",[])
    response=min(pivots,key=lambda a:abs(a["z_mm"]-(front or 1450))) if pivots else None
    pivot=response["z_mm"] if response else None
    pivot_label="AC pivot" if response and max(response["position_singular_nm_per_urad"])<.1 else "Least scan motion"
    front_label="Mini + objective front" if "isolated_pair_fronts" in r else "Full-field front"
    fig,ax=plt.subplots(2,1,figsize=(10.8,6.6),sharex=True,layout="constrained")
    fig.suptitle("Zero-scan trajectories through TL12 and the scan coils",weight="bold",fontsize=15)
    take=(z>=1320)&(z<=1525)
    centers=np.asarray(PARTS["ac_deflector"]["interaction_centers_local_z_mm"])+1000*START
    tl=PARTS["probe_tl12_lens"]
    spans=[(tl["local_start_z_mm"]+450,tl["local_end_z_mm"]+450,"TL12")]
    coil=PARTS["ac_deflector"]
    length=coil.get("physical_coil_length_mm",15.)
    spans.extend((c-length/2,c+length/2,label) for c,label in zip(centers,("Upper scan","Lower scan")))
    for d,p in enumerate(ax):
        for i in range(line.shape[1]):
            p.plot(z[take],1e6*line[take,i,d],color=colors[i],lw=.58,alpha=.6)
        for lo,hi,label in spans:
            p.axvspan(lo,hi,alpha=.12,color="#39677d",lw=0)
            if d==0:p.text((lo+hi)/2,.95,label,transform=p.get_xaxis_transform(),ha="center",va="top",fontsize=8)
        for k,col,style,label in [(waist,"#245eb2","-.","Radial RMS minimum"),(front,"#b7472e","--",front_label),(pivot,"#008466",":",pivot_label),(previous,"#666666","--","Previous waist")]:
            if k is not None:p.axvline(k,color=col,ls=style,lw=1.65,label=f"{label}  {k:.3f} mm")
        p.set_ylabel(("x" if d==0 else "y")+" (µm)")
        p.spines[["right","top"]].set_visible(False);p.grid(axis="y",alpha=.16)
    ax[1].set_xlim(1320,1525);ax[1].set_xlabel("Distance from the effective source (mm)")
    ax[1].legend(loc="lower left",fontsize=8,framealpha=.9,ncols=2)
    fig.savefig(folder/"tl12-scan-rays.png",dpi=170);plt.close(fig)
    if front is not None:
        keep=(z>=front-15)&(z<=front+15)
        base=np.interp(front,z,stats[:,0])
        fig,(top,bottom)=plt.subplots(2,1,figsize=(8,5.2),sharex=True,layout="constrained")
        fig.suptitle("Zero-scan finite-source envelope at the focal plane",weight="bold",fontsize=14)
        top.plot(z[keep],1e6*(stats[keep,0]-base),color="#245c75")
        top.set_ylabel("Change in RMS radius (pm)")
        top.set_title(f"RMS radius at the front plane  {base*1000:.4f} nm",loc="left",fontsize=10)
        for j,lab,col in [(3,"Minimum directional growth","#216e78"),(4,"Maximum directional growth","#ab4a37")]:
            bottom.plot(z[keep],stats[keep,j]*1e6,label=lab,color=col)
        bottom.set(xlabel="Distance from the effective source (mm)",ylabel="Half-variance slope (nm²/mm)")
        bottom.axhline(0,lw=.8,color="#333333")
        bottom.legend(frameon=False,fontsize=8)
        for p in (top,bottom):
            p.axvline(front,color="#b7472e",ls="--",lw=1.3)
            p.grid(axis="y",alpha=.14);p.spines[["top","right"]].set_visible(False)
        fig.savefig(folder/"front-waist-check.png",dpi=170);plt.close(fig)
    center=np.average(xy,axis=0,weights=current)
    radii=np.linalg.norm(xy-center,axis=1);order=np.argsort(radii)
    d95=surface["D95_um"]*1000
    scale=1 if d95<1600 else 1000
    unit="nm" if scale==1 else "µm"
    # Every original landing contributes to the current integral, including the tails.
    edge=1.035*np.abs(xy).max()/scale
    edges=np.linspace(-edge,edge,69)
    hist,*_=np.histogram2d(xy[:,0]/scale,xy[:,1]/scale,bins=(edges,edges),weights=current)
    np.testing.assert_allclose(hist.sum(),surface["specimen_uA"]*1e3,rtol=2e-12)
    pitch=edges[1]-edges[0]
    fig,(p,c)=plt.subplots(1,2,figsize=(9.8,4.5),width_ratios=(1.22,1),layout="constrained")
    fig.suptitle(f"Specimen upper surface  ·  z = {r['surface_z_mm']:.4f} mm",weight="bold",fontsize=15)
    im=p.pcolormesh(edges,edges,hist.T/pitch**2,cmap="viridis",rasterized=True,shading="flat")
    p.add_patch(Circle(center/scale,d95/(2*scale),fill=False,ls="--",ec="white",lw=1.3))
    p.plot(0,0,"+",color="#ff995e",ms=7)
    p.set(xlabel=f"x ({unit})",ylabel=f"y ({unit})",aspect="equal",title=f"Zero-scan landings · {len(xy):,} particles")
    fig.colorbar(im,ax=p,shrink=.78,label=f"Current density (nA/{unit}²)")
    c.step(radii[order]/scale,np.cumsum(current[order])/1000,where="post",color="#147789",lw=2)
    c.axvline(d95/(2*scale),ls="--",color="#a74832");c.axhline(surface["specimen_uA"]*.95,ls=":",color="#a74832")
    c.set(xlim=(0,None),ylim=(0,surface["specimen_uA"]*1.06),xlabel=f"Radius about the current centroid ({unit})",
          ylabel="Enclosed current (µA)",title=f"D95 = {d95/scale:.3g} {unit}")
    c.grid(alpha=.16);c.spines[["top","right"]].set_visible(False)
    fig.savefig(folder/"specimen-probe.png",dpi=180);plt.close(fig)


if __name__=="__main__":
    arg=argparse.ArgumentParser();arg.add_argument("label")
    arg.add_argument("--scan-label");arg.add_argument("--reference-label")
    args=arg.parse_args()
    render(SAVE/args.label,scan_label=args.scan_label,reference_label=args.reference_label)
