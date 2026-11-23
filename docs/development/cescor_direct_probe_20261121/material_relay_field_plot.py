"""Plot the signed on-axis field actually used by the particle column."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from condenser_source_fields import write
from material_relay_fields import SAVE,WORK
from relay_transfer import PARTS, START


def read(path):
    with np.load(path) as data:return data["z"]*1000,data["coef"][:,0,2]


def plot(folder):
    tracking=json.loads((folder/"tracking.json").read_text());rec=tracking["field"]
    path=WORK/(Path(rec["source"]).stem+f"-dc-bore-{rec['step_mm']:g}mm.npz")
    z,b=read(path)
    baseline=WORK/"global-7a804e13d71e743b-dc-bore-0.5mm.npz";z0,b0=read(baseline)
    top=tracking["surface_z_mm"]
    front=tracking.get("isolated_pair_fronts",tracking["operational_fronts"])[0]["z_mm"]
    origin_mm=1000*START
    scan=PARTS["ac_deflector"]
    upper,lower=origin_mm+np.asarray(scan["interaction_centers_local_z_mm"])
    marks={"TL12 exit":origin_mm+PARTS["probe_tl12_lens"]["local_end_z_mm"],
           "Upper scan centre":float(upper),"Lower scan centre":float(lower),
           "Lower scan exit":float(lower+.5*scan.get("physical_coil_length_mm",15.)),
           "Illumination-side front":front,"Specimen upper surface":top}
    traced=CubicSpline(z,b)
    evidence={k:dict(z_mm=v,Bz_mT=float(1000*traced(v))) for k,v in marks.items()}
    pieces={"C1 lens":(450.,550.),"C2 lens":(550.,750.),"C3 lens":(830.,1000.),"TL12 to scan exit":(1353.6,1420.),
            "Scan exit to Mini":(1420.,1512.5),"Mini to specimen":(1512.5,top)}
    regions={}
    for k,(a,c) in pieces.items():
        nodes=np.r_[a,z[(z>a)&(z<c)],c];vals=np.interp(nodes,z,b);i=np.argmax(abs(vals))
        regions[k]=dict(interval_mm=[a,c],signed_peak_mT=1000*float(vals[i]),signed_peak_z_mm=float(nodes[i]),
                        integrated_Bz2_T2_mm=float(np.trapezoid(vals**2,nodes)))
    write(folder/"on-axis-field.json",dict(samples=evidence,regions=regions,bias=rec["bias"],mesh=rec["mesh_id"]))
    np.savez_compressed(folder/"on-axis-field.npz",z_mm=z,signed_Bz_T=b)
    fig,axes=plt.subplots(2,1,figsize=(10.,6.4),layout="constrained")
    fig.suptitle("Signed axial magnetic field of the coupled column",weight="bold",fontsize=15)
    for p,(lo,hi) in zip(axes,[(420.,top),(1335.,top)]):
        for zz,bb,color,style,label in [(z0,b0,"#6f7983","--","Initial common-circuit trial"),
                                      (z,b,"#146d83","-","Traced operating point")]:
            g=(zz>=lo-.5)&(zz<=hi+.5)
            p.plot(zz[g],1000*bb[g],color=color,ls=style,lw=1.7,label=label)
        p.axhline(0,color="#787e85",lw=.55);p.set(xlim=(lo,hi),ylabel="On-axis Bz (mT)")
        p.grid(axis="y",alpha=.2);p.spines[["top","right"]].set_visible(False)
    for key,label in [("probe_tl12_lens","TL12"),("ac_deflector","Scan pair"),("mini_condenser","Mini"),("objective_upper_pole","Upper pole")]:
        row=PARTS[key];lo=row["local_start_z_mm"]+450;hi=row["local_end_z_mm"]+450
        axes[1].axvspan(lo,hi,color="#527b89",alpha=.10,lw=0)
        axes[1].text((lo+hi)/2,.95,label,transform=axes[1].get_xaxis_transform(),ha="center",va="top",fontsize=8)
    axes[1].axvline(front,color="#af503b",ls=":",lw=1.3,label=f"Front plane {front:.2f} mm")
    axes[0].legend(frameon=False,ncols=2,fontsize=9)
    axes[1].set_xlabel("Distance from the effective source (mm)")
    fig.savefig(folder/"axial-field.png",dpi=175);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("label");a=p.parse_args();plot(SAVE/a.label)
