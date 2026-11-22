"""Plot native illuminated-column trajectories and absolute specimen-plane current."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from condenser_normal import FOIL, HERE, SURFACE, WORK, spans_mm
from relay_transfer import HP1, HP2, START, TL21, TL22


COLORS = ["#3167a1","#0a8a82","#bd8123","#bb4b41"]


def plot(name,folder=HERE):
    row = json.loads((WORK/(name+".json")).read_text())
    with np.load(WORK/(name+".npz")) as source:
        a = dict(source)
    z=a["track_z_m"]*1000
    s=SURFACE*1000
    band=a["track_band"]
    tracks=a["tracks"]
    fig=plt.figure(figsize=(13.8,7.9),layout="constrained")
    layout=fig.add_gridspec(2,2,height_ratios=[1.12,1])
    full=fig.add_subplot(layout[0,:])
    gap=fig.add_subplot(layout[1,0])
    end=fig.add_subplot(layout[1,1])
    cross=row["waists_mm"]["C2_C3_gap"][0][0]
    for ax,lo,hi in ((full,START*1000,s),(end,s-.16,s)):
        keep=(z>=lo)&(z<=hi)
        xaxis=(z[keep]-s)*1000 if ax is end else z[keep]
        for i,b in enumerate(band):
            ax.plot(xaxis,tracks[keep,i,0]*1e6,color=COLORS[b] if b>=0 else "#9aa0a2",
                    alpha=.73 if b>=0 else .62,lw=.72,ls="-" if b>=0 else ":")
        ax.set_xlim(xaxis[0],xaxis[-1])
        ax.axhline(0,c="#28323c",lw=.55,zorder=0)
        ax.set_ylabel("Transverse x (µm)")
        ax.set_xlabel("Distance from specimen upper surface (µm)" if ax is end else "Column z (mm)")
        ax.spines[["top","right"]].set_visible(False)
        if ax is end:
            ax.axvline(0,color="#253442",lw=.9,clip_on=False)
    full.set_title("Gun exit to specimen upper surface  ·  fixed gun-exit distribution",loc="left",weight="bold",fontsize=13)
    for (l,r),label,y in zip([(450,550),*spans_mm(),(HP2*1000-15,HP1*1000+15),(1510,1535),(1575,s)],
                           ["C1 lens","C2 lens","C3 lens","ADL","Probe corrector","Mini Condenser","Objective"],
                           [.94,.94,.94,.91,.94,.94,.81]):
        full.axvspan(l,r,color="#e3e8eb",alpha=.8,zorder=-2,lw=0)
        full.text((l+r)/2,y,label,fontsize=8,ha="center",transform=full.get_xaxis_transform())
    for position, label in ((HP2,"HP2"),(TL22,"TL22"),(TL21,"TL21"),(HP1,"HP1")):
        full.axvline(position*1000,color="#87959d",lw=.55,zorder=-1)
        full.text(position*1000,.08,label,ha="center",fontsize=7,transform=full.get_xaxis_transform())
    full.axvline(FOIL.mean()*1000,color="#58616d",ls=":",lw=1.2)
    full.text(FOIL.mean()*1000+6,.8,"C2 aperture",fontsize=8,transform=full.get_xaxis_transform())
    handles=[Line2D([0],[0],c=color,lw=2,label=label) for color,label in zip(COLORS,
             ["0–0.35 R","0.35–0.65 R","0.65–0.90 R","0.90–1.00 R"])]
    handles.append(Line2D([0],[0],c="#8e9498",lw=1.4,ls=":",label="Intercepted at foil"))
    full.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,-.16),ncol=5,
                title="Incident radius at the C2 aperture",title_fontsize=8,fontsize=8,frameon=False)
    m=a["moments"]
    region=(z>=648)&(z<=1089)
    for (l,r),label in zip(spans_mm(),["C2 lens","C3 lens","ADL"]):
        gap.axvspan(l,r,color="#e3e8eb",alpha=.85,lw=0)
    gap.semilogy(z[region],1e6*np.hypot(m[region,1],m[region,2]),c="#314e6a",lw=1.65)
    gap.axvline(FOIL.mean()*1000,color="#58616d",ls=":",lw=1.)
    mid=np.mean(cross)
    gap.axvline(mid,c="#a83e37",ls="--",lw=.9)
    ymin=1e6*np.hypot(*m[np.argmin(abs(z-mid)),1:3])
    gap.annotate(f"Crossover  {mid:.2f} mm",xy=(mid,ymin),xytext=(mid+28,ymin*1.9),
                 arrowprops=dict(arrowstyle="-",color="#9d4941"),fontsize=8,color="#843d35")
    gap.set(xlim=(648,1089),xlabel="Column z (mm)",ylabel="RMS radius (µm)")
    gap.set_title("Condenser gaps  ·  sample-reaching population",loc="left",fontsize=11,weight="bold")
    end.set_title("Final approach to the specimen",loc="left",fontsize=11,weight="bold")
    for ax in (gap,end): ax.grid(alpha=.15,lw=.5)
    fig.savefig(folder/"illuminated-normal-column.png",dpi=162,bbox_inches="tight")
    plt.close(fig)

    fig=plt.figure(figsize=(12.0,7.5),layout="constrained")
    grid=fig.add_gridspec(2,2,width_ratios=[1.45,1.])
    spot,pupil,focus=fig.add_subplot(grid[:,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,1])
    live=a["specimen_loss"]==-1
    xy=a["specimen_phase"][live,:2]*1e6
    wi=a["weight"][live]*row["source_current_A"]*1e9
    mu=np.average(xy,axis=0,weights=wi)
    d95=row["specimen"]["D95_nm"]*.001
    limit=max(np.max(abs(xy))*1.02,d95*.6)
    edges=np.linspace(-limit,limit,136)
    current,_,_=np.histogram2d(*xy.T,bins=(edges,edges),weights=wi)
    density=current/(edges[1]-edges[0])**2
    occupied=density[density>0]
    colors=plt.get_cmap("magma")(np.linspace(.23,1,160))
    palette=LinearSegmentedColormap.from_list("sample_current",colors)
    im=spot.pcolormesh(edges,edges,np.ma.masked_less_equal(density.T,0),cmap=palette,
                       norm=LogNorm(vmin=occupied.min()*.9,vmax=occupied.max()),rasterized=True)
    spot.set_facecolor("#100d1a")
    spot.add_patch(Circle(mu,d95/2,fill=False,ls="--",lw=1.2,color="#8de6ed"))
    spot.plot(0,0,"+",color="white",ms=8,mew=1.2)
    spot.set(aspect="equal",xlabel="x (µm)",ylabel="y (µm)",xlim=(-limit,limit),ylim=(-limit,limit))
    spot.set_title(f"Specimen upper surface  ·  D95 = {d95:.3f} µm",loc="left",fontsize=13,weight="bold")
    cb=fig.colorbar(im,ax=spot,location="bottom",fraction=.055,pad=.06,aspect=37)
    cb.set_label("Incident current density (nA/µm²)")
    spot.text(.025,.03,"+  Optical axis     - -  95% current about centroid",color="white",
              transform=spot.transAxes,fontsize=8,bbox=dict(color="#171326",alpha=.80,pad=4,lw=0))
    front=a["front_phase"][a["front_loss"]==-1,:2]*1e6
    val=row["maximum_foil_incident_radius_um"]*1.025
    pupil.hist2d(*front.T,bins=100,range=[[-val,val],[-val,val]],cmin=1,cmap="Blues",norm=LogNorm())
    pupil.add_patch(Circle((0,0),50,fill=False,color="#bd493c",lw=1.4))
    pupil.set(aspect="equal",xlabel="x (µm)",ylabel="y (µm)")
    pupil.set_title("C2 aperture entrance  ·  100 µm bore",loc="left",fontsize=10.5,weight="bold")
    dz=a["offsets_nm"]*.001
    profiles=a["profiles"]
    focus.plot(dz,profiles[:,0]*.001,color="#2c6686",lw=1.8,label="D95")
    focus.axvline(0,color="#b44d3d",ls="--",lw=1.1,label="Specimen surface")
    focus.plot(0,d95,"o",ms=4,color="#2c6686")
    focus.set(xlabel="Distance from specimen upper surface (µm)",ylabel="D95 (µm)",xlim=(-300,300))
    focus.set_title("Measured through-focus diameter",loc="left",fontsize=10.5,weight="bold")
    focus.grid(alpha=.18)
    focus.legend(fontsize=8,frameon=False)
    fig.savefig(folder/"illuminated-normal-probe.png",dpi=165,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("name")
    plot(parser.parse_args().name)
