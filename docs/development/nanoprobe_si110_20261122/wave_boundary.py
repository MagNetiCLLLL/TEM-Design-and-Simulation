"""Check covariance admission for the actual finite-source entrance to a mixed Si scan."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import c,e,h,m_e
from scipy.interpolate import CubicSpline

from material_trial import Field
from material_relay_bore import RADIUS, design
from temsim.physics.compute_backend import require_coherent_cuda
from temsim.physics.particle_wave_source import lifted_centers, angular_phase_model


def check(folder):
    require_coherent_cuda("Require GPU")
    report=json.loads((folder/"tracking.json").read_text())
    field=Field(Path(report["solution"]),step=report["bore_step_mm"])
    with np.load(report["gun_source"]) as g, np.load(folder/"column-and-landings.npz") as col:
        ok=col["loss"]==-1
        records=[("accelerator_exit",.45,g["phase"],g["weight"],g["ids"],g["energy_offset_ev"]),
                 ("specimen",report["surface_z_mm"]*.001,col["surface_phase"][ok],
                  col["weights"][ok],col["source_ids"][ok],g["energy_offset_ev"][ok])]
    result={}
    for key,z,phase,weights,ids,energies in records:
        assert len(np.unique(ids))==len(phase)>=1000 and min(weights)>0
        xy=phase[:,:2]
        assert np.max(np.linalg.norm(xy,axis=1))<RADIUS
        coeff=CubicSpline(field.z,field.c,axis=0)(z)[:,2]
        # The radial gauge obeys curl A = B at the reference section.
        # The t weight gives 1/(2+n) for each degree-n monomial in Bz.
        moment=design(xy)@(coeff/np.array([2.,3.,3.,4.,4.,4.]))
        potential=np.column_stack([-xy[:,1],xy[:,0]])*moment[:,None]
        shift=-e/h*1e-10*potential
        labels=["x_m","y_m","tx_rad","ty_rad"]
        rows={name:phase[:,i] for i,name in enumerate(labels)}
        rows.update(ray_id=ids,canonical_frequency_shift_inv_angstrom=shift)
        energy=np.average(300000.+energies,weights=weights)
        lam=h*c/e*1e10/np.sqrt(energy*(energy+2*m_e*c*c/e))
        try:
            lifted_centers(rows,lam,weights)
        except ValueError as exc:
            if "symplectic eigenemittances" not in str(exc):
                raise
            admission=False
            message=str(exc)
        else:
            admission=True
            message="Admitted by the local source covariance test"
        # Also report direct numbers in the project's position/frequency coordinates.
        prob=weights/weights.sum()
        direction=phase[:,2:4]
        freq=direction/np.sqrt(1+(direction**2).sum(axis=1))[:,None]/lam+shift
        disp,_,_=angular_phase_model(xy*1e10,freq,prob,ids,lam)
        values=np.column_stack((xy*1e10-disp,freq))
        values-=np.average(values,axis=0,weights=prob)
        cov=(values.T*prob)@values
        J=np.block([[np.zeros((2,2)),np.eye(2)],[-np.eye(2),np.zeros((2,2))]])
        em=np.sort(np.abs(np.linalg.eigvals(J@cov)))[::2]
        result[key]=dict(z_mm=z*1000,original_live=len(ids),wavelength_nm=lam*.1,
            Bz_on_axis_T=float(coeff[0]),canonical_shift_max_inv_angstrom=float(np.linalg.norm(shift,axis=1).max()),
            eigenemittances_dimensionless=em.tolist(),threshold=1/(4*np.pi),
            ratios_to_minimum=(4*np.pi*em).tolist(),covariance_matched_wave_admitted=admission,message=message)
    (folder/"wave-boundary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:q["ratios_to_minimum"] for k,q in result.items()}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    check(p.parse_args().folder.resolve())
