"""Show the actual illuminated phase space and the two Si column positions on native pixels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

HERE=Path(__file__).resolve().parent


def render(folder):
    r=json.loads((folder/"tracking.json").read_text())
    scan=json.loads((folder/"si110/scan.json").read_text())
    with np.load(folder/"column-and-landings.npz") as col:
        live=col["loss"]==-1
        p=col["surface_phase"][live]
        I=col["weights"][live]*col["source_A"]*1e6
    with np.load(folder/"si110/detectors.npz") as silicon:
        coord=silicon["coordinates_nm"]
        im=silicon["all_window_A"][1]*1e9
        halves=silicon["half_window_A"][:,:,1]/silicon["half_incident_A"][:,None]
    s=r["surface"]
    rad=s["D95_um"]*500
    xy=p[:,:2]*1e9
    center=np.average(xy,axis=0,weights=I)
    limit=max(abs(xy).max()*1.02,rad*1.4)
    edges=np.linspace(-limit,limit,71)
    hist=np.histogram2d(*xy.T,bins=(edges,edges),weights=I)[0].T/np.diff(edges)[0]**2
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.8),layout="constrained")
    ax,df,ln=axes
    pic=ax.pcolormesh(edges,edges,hist,shading="flat",cmap="magma")
    ax.add_patch(Circle(center,rad,fill=False,color="#52d5ca",ls="--",lw=1.6))
    ax.set(aspect="equal",xlabel="x (nm)",ylabel="y (nm)",title=f"Specimen entrance  ·  D95 = {rad*2:.2f} nm")
    fig.colorbar(pic,ax=ax,location="bottom",shrink=.82,aspect=28,label="Particle current density (µA/nm²)")
    extent=[coord[0]-.01,coord[-1]+.01]*2
    sample=df.imshow(im,origin="lower",extent=extent,interpolation="nearest",cmap="gray",aspect="equal")
    ix=int(np.searchsorted(coord,0.))
    df.axvline(coord[ix],color="#efab59",ls="--",lw=1.)
    df.set(title="Si [110]  ·  50–150 mrad ADF",xlabel="x (nm)",ylabel="y (nm)")
    fig.colorbar(sample,ax=df,location="bottom",shrink=.82,aspect=28,label="Collected current (nA)")
    profile=im[:,ix]
    pair=scan["pair_spacing_nm"]
    # The two projected sites at x=0 in this oriented, centred CIF.
    for offset in (-pair,0.):
        ln.axvline(offset,color="#a45c18",lw=1.2,ls=":")
    ln.plot(coord,profile,color="#006a70",lw=1.8,marker=".",label="Full source")
    shapes=halves.reshape(2,len(coord),len(coord))[:,:,ix]
    for sub,c,label in zip(shapes,["#8195b6","#bd83a7"],["First half","Second half"],strict=True):
        ln.plot(coord,sub*I.sum()*1e3,color=c,lw=.9,alpha=.85,label=label)
    ln.set(xlim=(-.31,.18),xlabel="y (nm)",ylabel="Collected current (nA)",
           title=f"Line at x = {coord[ix]:.2f} nm")
    ln.legend(loc="best",fontsize=8,frameon=False)
    ln.text(.02,.01,"Dotted lines mark the CIF pair",transform=ln.transAxes,size=8,va="bottom")
    fig.suptitle("Nanoprobe Normal  |  Specimen-plane particles and classical Si [110] scan",
                 fontsize=15,weight="semibold")
    fig.savefig(folder/"probe-and-si110.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    render(parser.parse_args().folder.resolve())
