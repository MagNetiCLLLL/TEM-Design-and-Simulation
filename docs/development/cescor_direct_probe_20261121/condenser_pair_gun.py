"""Launch the original effective-source distribution in the joint material field."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline

from condenser_pair_study import OUT, Pair
from condenser_source_fields import GunMagnet, check_file, sha
from temsim.optics.electron_gun.base import EmissionBundle
from temsim.optics.electron_gun.tracing import _trace_feg_to_exit
from temsim.physics.instrument_electric import configure_instrument_electric_domain, instrument_electric_end_mm
from temsim.vacuum import bind_gun_environment


def selected_field(pair):
    point = json.loads((OUT / "point.json").read_text())
    path, data = pair.field(point["series_A"], point["mesh_mm"], point["domain_mm"])
    check_file(path, point["material_sha256"])
    return pair.f.current(*point["series_A"]), data


def identity(pair, data, q, count):
    fields = [pair.f.gun_key(q,count), str(data["key"]), sha(Path(__file__))]
    return hashlib.sha256("|".join(fields).encode()).hexdigest()


class MaterialGunMagnet(GunMagnet):
    def __init__(self, fields, q, data):
        z = np.linspace(-.002,.455,1829)
        induced = make_interp_spline(data["z"],data["induced_T"],k=5)
        axial = make_interp_spline(z,q @ fields.round_axis(z) + fields.objective(z)[:,0] + induced(z),k=5)
        sample = np.column_stack([axial(z,nu=i) for i in range(5)] +
                                 [fields.far_corrector(z).reshape(len(z),9)])
        self.dz = z[1]-z[0]
        self.coeff = np.ascontiguousarray(CubicSpline(z,sample).c.transpose(1,2,0))


def emit(pair, count):
    f = pair.f
    if not 1000 <= count <= len(f.emission.weight):
        raise ValueError("Original positive-current source population out of range")
    q, data = selected_field(pair)
    gun = f.state.electron_gun
    bind_gun_environment(f.state)
    configure_instrument_electric_domain(gun, instrument_electric_end_mm(f.state))
    gun._instrument_magnetic_field = MaterialGunMagnet(f,q,data)
    gun._instrument_magnetic_identity = None
    signature = identity(pair,data,q,count)
    gun.emitter.ray_count = count
    physical = gun.emit()
    names = ("x_m","y_m","tx_rad","ty_rad")
    for attr in (*names,"energy_offset_ev","ray_id"):
        np.testing.assert_array_equal(getattr(physical,attr),getattr(f.emission,attr)[:count])
    if not np.all(physical.weight>0) or not np.isclose(sum(physical.weight),1):
        raise ValueError("Invalid emitter weights")
    if gun.emitter.surface_model is not None:
        raise ValueError("This launch adapter requires the saved effective planar source")
    delta=np.diag([1e-9,1e-9,1e-6,1e-6])
    diag=np.vstack([np.zeros((1,4)),delta,-delta,delta/2,-delta/2])
    args={key:np.r_[getattr(physical,key),diag[:,j]] for j,key in enumerate(names)}
    launch=EmissionBundle(**args,energy_offset_ev=np.r_[physical.energy_offset_ev,np.zeros(len(diag))],
        weight=np.r_[physical.weight,np.zeros(len(diag))],
        ray_id=np.r_[physical.ray_id,1000000+count+np.arange(len(diag))])
    print("Fresh effective-source launch",count, "material",str(data["key"])[:12],flush=True)
    result=_trace_feg_to_exit(gun,emitted_bundle=launch)
    exit_bundle=result.exit_bundle
    np.testing.assert_array_equal(exit_bundle.ray_id, launch.ray_id)
    np.testing.assert_array_equal(exit_bundle.weight, launch.weight)
    if not exit_bundle.alive.all():
        raise ValueError("Gun interception: preserve the lost source identities before handoff")
    selected=np.linspace(0,count-1,64,dtype=int)
    np.savez_compressed(OUT/f"tip-{count}.npz", signature=signature,
        phase=np.column_stack([getattr(exit_bundle,k) for k in (*names,"flight_time_s")]),
        launch=np.column_stack([getattr(launch,k) for k in names]),current=q,
        emitted_A=result.emitted_current_a, ids=launch.ray_id, weight=launch.weight,
        energy_offset_ev=exit_bundle.energy_offset_ev,launch_energy_eV=gun.emitter.emission_energy_ev+launch.energy_offset_ev,
        gun_z_mm=result.z_mm,gun_paths_m=np.stack([result.x_m[:,selected],result.y_m[:,selected]],axis=2),
        selected_ids=launch.ray_id[selected])
    print("Fresh source exit ready",count,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count",type=int,default=32768)
    emit(Pair(),ap.parse_args().count)
