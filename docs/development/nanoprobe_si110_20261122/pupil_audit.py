"""Measure the illuminated C2 foil and the specimen-conjugate first-order acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from material_trial import Column, Field, source_and_electrodes
from relay_transfer import advance
from screen import FACTOR


def quantiles(x,w):
    i=np.argsort(x)
    return np.interp([.5,.95,1],np.cumsum(w[i])/w.sum(),x[i]).tolist()


def audit(folder):
    r=json.loads((folder/"tracking.json").read_text())
    dc=Field(Path(r["solution"]),step=r["bore_step_mm"])
    c=Column(source_and_electrodes(),dc,Path(r["gun_source"]),r["count"],r["step_mm"],
             observation_mm=(764.9,765.1))
    faces=np.searchsorted(c.m.z,[.7649,.7651])
    states=advance(c.t,c.full,c.t.input,c.en,c.phi0,c.loss,0,faces)
    foil=[]
    for station,(p,loss) in zip(faces,states,strict=True):
        live=loss==-1
        rad=np.linalg.norm(p[live,:2],axis=1)
        foil.append(dict(z_mm=1000*c.m.z[station],live=int(live.sum()),
                         radius_p50_p95_max_um=quantiles(rad*1e6,c.t.weights[live]),
                         count_outer_ten_percent=int(np.count_nonzero(rad>=.9*50e-6))))
    # Back-propagate a unit angular displacement from a point on the optic axis.
    # The result is the position/angle element of the round first-order matrix.
    z=c.m.z
    k=FACTOR*CubicSpline(dc.z,dc.c[:,0,2])(.5*(z[1:]+z[:-1]))
    step=-np.diff(z)
    co=np.cos(k*step)
    si=step*np.sinc(k*step/np.pi)
    b=np.zeros(len(z))
    slopes=np.zeros(len(z))
    slopes[-1]=1
    for i in range(len(k)-1,-1,-1):
        b[i]=co[i]*b[i+1]+si[i]*slopes[i+1]
        slopes[i]=co[i]*slopes[i+1]-k[i]**2*si[i]*b[i+1]
    mag=np.maximum(abs(b[1:]),abs(b[:-1]))
    limiting=c.m.radius/np.maximum(mag,1e-30)
    j=np.argmin(limiting)
    stops=[]
    for owner in np.unique(c.m.owner[c.m.owner>=0]):
        indices=np.flatnonzero(c.m.owner==owner)
        i=indices[np.argmin(limiting[indices])]
        stops.append(dict(element=str(c.m.keys[owner]),z_mm=1000*z[i],radius_mm=1000*c.m.radius[i],
                          paraxial_semiangle_mrad=1000*np.arctan(limiting[i])))
    stops.sort(key=lambda row:row["paraxial_semiangle_mrad"])
    final=np.flatnonzero(z>1.5125)
    h=final[np.argmax(abs(b[final]))]
    target=.030
    gap=np.flatnonzero(abs(b[final])*np.tan(target)>dc.receipt["local_radius_m"])
    last=z[final[gap[-1]]] if len(gap) else None
    radii=abs(b[faces])
    data=dict(foil_diameter_um=100.,foil_faces=foil,
              transfer_model="Round paraxial Larmor response of the re-solved field at 300 keV; extrapolation to the rim",
              c2_position_per_sample_angle_mm=(1000*radii).tolist(),
              formal_c2_edge_mrad=(1000*np.arctan(50e-6/radii)).tolist(),
              limiting_geometric_semiangle_mrad=float(1000*np.arctan(limiting[j])),
              limiting_element=str(c.m.keys[c.m.owner[j]]),limiting_z_mm=1000*z[j],
              model_valid_radius_um=dc.receipt["local_radius_m"]*1e6,
              angle_for_50um_to_mini_mrad=float(1000*np.arctan(dc.receipt["local_radius_m"]/abs(b[h]))),
              required_peak_radius_30mrad_mini_to_surface_mm=float(1000*abs(b[h])*np.tan(target)),
              required_peak_at_mm=1000*z[h],last_outside_bore_for_30mrad_mm=None if last is None else 1000*last,
              aperture_radius_needed_for_30mrad_um=(1e6*radii*np.tan(target)).tolist(),
              geometric_stops_by_acceptance=stops)
    (folder/"pupil.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    print(json.dumps(data),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    audit(parser.parse_args().folder.resolve())
