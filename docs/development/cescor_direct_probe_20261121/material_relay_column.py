"""Track finite-source particles and four independent scan responses in one coupled solid model."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cupy as cp
import numpy as np
from numba import njit
from scipy.interpolate import CubicHermiteSpline,CubicSpline,make_interp_spline

from column_current import ApertureTracker, SURFACE
from condenser_pair_column import column_mesh
from condenser_source_fields import prepared_source, write
from downstream_pivot import DrivenTrace
from joint_pivot_tuning import scan_response
from material_relay_bore import RADIUS,design
from material_relay_fields import SAVE,WORK,digest,parent,sha
from material_relay_run import launch_in_field,static
from material_relay_tracking import Trial, installed_mix
from relay_transfer import PARTS,WORK as SCAN_CACHE,electric_reference
from temsim.physics.instrument_electric import capture_instrument_electric_field


def source_and_electrodes():
    state, emission = prepared_source()
    return SimpleNamespace(state=state, emission=emission, electric=capture_instrument_electric_field(state))


@njit(cache=True)
def gun_field(p,coefs,start,pitch):
    out=np.zeros_like(p)
    for j in range(len(p)):
        x,y,z=p[j]
        u,v=x/RADIUS,y/RADIUS
        t=z-start;i=int(np.floor(t/pitch));t-=i*pitch
        if i<0 or i>=len(coefs) or u*u+v*v>1.:
            raise ValueError("Gun sample outside the verified material field")
        basis=(1.,u,v,u*u,u*v,v*v)
        for k in range(6):
            for l in range(3):
                a,b,c,d=coefs[i,k,l]
                out[j,l]+=(((a*t+b)*t+c)*t+d)*basis[k]
    return out


class Magnet:
    def __init__(self,z,coeff):
        self.start=z[0];self.pitch=z[1]-z[0]
        spline=CubicSpline(z,coeff)
        self.coeff=np.ascontiguousarray(spline.c.transpose(1,2,3,0))
        q=np.linspace(.000073,.45019,150)
        xy=.000012*np.column_stack([np.sin(213*q),np.cos(175*q)])
        expected=np.einsum("ij,ijk->ik",design(xy),spline(q))
        np.testing.assert_allclose(self.field_at_global_positions_t(np.column_stack([xy,q])),expected,rtol=3e-10,atol=1e-14)
    def field_at_global_positions_t(self,points):
        return gun_field(np.asarray(points).reshape(-1,3),self.coeff,self.start,self.pitch).reshape(np.shape(points))


class Field:
    def __init__(self,path,key="dc",step=.5):
        self.path=WORK/f"{path.stem}-{key}-bore-{step:g}mm.npz"
        with np.load(self.path) as raw:
            self.z,self.c=raw["z"],raw["coef"]
            self.receipt=json.loads(str(raw["receipt"]))
        if self.receipt.get("solution_sha256")!=sha(path) or self.receipt.get("channel")!=key:
            raise ValueError(f"Rebuild the {key} bore table for this converged material solution")
        self.xy=np.linspace(-.00006,.00006,17)
        length=self.z[-1]-.45
        self.localz=np.linspace(0.,length,round(length/.000125)+1)
        y,x=np.meshgrid(self.xy,self.xy,indexing="ij")
        self.basis=design(np.column_stack([x.ravel(),y.ravel()]))

    def table(self):
        coeff=CubicSpline(self.z,self.c,axis=0)(self.localz+.45)
        return np.einsum("ij,zjk->zik",self.basis,coeff).reshape(-1,len(self.xy),len(self.xy),3)


class Column(Trial):
    def __init__(self,f,dc,gun,n,step,*,observation_mm=()):
        self.f=f;self.dc=dc
        self.m,self.stops=column_mesh(f,step,extra_observation_mm=observation_mm)
        with np.load(gun) as g:
            self.source=tuple(g[k] for k in ("phase","weight","ids","energy_offset_ev"))+(float(g["emitted_A"]),)
        if n!=len(self.source[1]) or n<1000 or np.min(self.source[1])<=0:raise ValueError("Invalid source")
        table=dict(z_local_m=dc.localz,xy_m=dc.xy,B_zyx_T=dc.table(),valid_radius_m=RADIUS)
        self.t=ApertureTracker(self.m,table,"cubic",source=self.source)
        self.en=self.t.de+300000.;self.loss=np.full(n,-1,np.int32)
        self.phi0=electric_reference(self.t)
        self.full=cp.zeros((len(self.m.nodes),20))
        self.pair=self.axis_field(dc)

    def axis_field(self,field):
        spline=make_interp_spline(field.z,field.c[:,0,2],k=5)
        result=np.zeros((len(self.m.nodes),20))
        for i in range(6):
            result[:,i]=spline(self.m.nodes,nu=i)
        return cp.asarray(result)


class MaterialScan(DrivenTrace):
    def __init__(self,model,responses,*,from_source=False):
        self.model=model
        import hashlib
        with np.load(SCAN_CACHE/"downstream-scan-windings.npz") as rec:
            if str(rec["winding_sha256"])!=hashlib.sha256(json.dumps(PARTS["ac_deflector"],sort_keys=True).encode()).hexdigest():
                raise ValueError("The AC circuit no longer matches its electrical calibration")
            self.current_matrix=rec["command_matrix_A_per_mrad"]
        self.basis=[]
        for b in responses:
            np.testing.assert_array_equal(b.z,model.dc.z)
            self.basis.append(cp.asarray(b.table()))
        self.static_grid=model.t.tables[0]
        self.grid=self.static_grid.copy()
        model.t.tables[0]=self.grid
        self.from_source=from_source
        self.source_coeff=[r.c for r in responses]
        self.source_hashes=[sha(r.path) for r in [model.dc,*responses]]
        self.original_input=model.t.input
        self.original_offsets=model.t.de

    def set_drive(self,station_urad):
        amps=super().set_drive(station_urad)
        if not self.from_source:return amps
        m=self.model
        if np.any(amps):
            n=len(m.t.ids)
            signature=digest(dict(field=self.source_hashes,amps=amps.tolist(),source=parent()["source_signature"],
                                  n=n,gun_sampler="cartesian-cubic-v2-driven"))
            path=WORK/f"common-gun-{signature}.npz"
            if not path.is_file():
                coeff=m.dc.c+sum(a*c for a,c in zip(amps,self.source_coeff))
                launch_in_field(m.f,n,signature,path,Magnet(m.dc.z,coeff))
            with np.load(path) as gun:
                np.testing.assert_array_equal(m.t.ids,gun["ids"])
                np.testing.assert_array_equal(m.t.weights,gun["weight"])
                m.t.input=gun["phase"];m.t.de=gun["energy_offset_ev"]
        else:
            m.t.input=self.original_input;m.t.de=self.original_offsets
        m.en=m.t.de+300000.
        m.phi0=electric_reference(m.t)
        return amps


def launch(f,dc,n):
    signature=digest(dict(field=sha(dc.path),physical_source=parent()["source_signature"],n=n,gun_sampler="cartesian-cubic-v2"))
    gun=WORK/("common-gun-"+signature+".npz")
    if not gun.exists():launch_in_field(f,n,signature,gun,Magnet(dc.z,dc.c))
    return gun


def run(args):
    f=source_and_electrodes();dc=Field(args.solution,step=args.bore_step)
    gun=launch(f,dc,args.count)
    model=Column(f,dc,gun,args.count,args.step)
    folder=SAVE/args.label;folder.mkdir(parents=True,exist_ok=True)
    record=static(model,gun,folder,args.full,"Coupled 3-D material FEM")
    record.update(operational_fronts=model.fronts(),field=dc.receipt,
                  sha=sha(dc.path),AC_command_static=[0.,0.,0.,0.],step_mm=args.step,
                  bore_spline_sampling_mm=1000*(dc.localz[1]-dc.localz[0]))
    if args.pair:
        pair=Field(args.pair,step=args.bore_step)
        wanted={k:(v if k.startswith(("mini_condenser/","objective_lens/")) else 0.)
                for k,v in dc.receipt["bias"].items()}
        if pair.receipt["mesh_id"]!=dc.receipt["mesh_id"] or pair.receipt["bias"]!=wanted:
            raise ValueError("The isolated lens pair does not match the operating point")
        model.pair=model.axis_field(pair)
        record["isolated_pair_fronts"]=model.fronts()
        record["pair"]=pair.receipt
    write(folder/"tracking.json",record)
    if args.scan:
        fronts=record.get("isolated_pair_fronts",record["operational_fronts"])
        target=min(fronts,key=lambda x:abs(x["z_mm"]-args.near))["z_mm"]
        modes=[Field(args.solution,f"ac_{i}_{axis}",step=args.bore_step) for i in (0,1) for axis in "xy"]
        mix=installed_mix() if not args.mix else np.array(json.loads(Path(args.mix).read_text())["mix"])
        trace=MaterialScan(model,modes,from_source=args.source_scan)
        if args.calibrate:
            goal=target*.001
            ix=np.searchsorted(model.m.z,goal)
            planes=model.m.z[max(0,ix-1):ix+1]
            basis,_,_=trace.response(model.full,np.eye(4),planes,1.)
            signed_position=CubicHermiteSpline(planes,basis[:,:2],basis[:,2:4],axis=0)
            pos=signed_position(goal)
            if np.linalg.cond(pos[:,2:])>1e5:raise ValueError("Ill-conditioned lower scan circuit")
            mix=np.vstack([mix[:2],np.linalg.solve(pos[:,2:],-pos[:,:2]@mix[:2])])
            if args.unit_tilt:
                tilt_matrix=signed_position(goal,nu=1)@mix*1e6
                if np.linalg.cond(tilt_matrix)>1e5:
                    raise ValueError("The pivot cannot support independent horizontal and vertical tilts")
                mix=np.linalg.solve(tilt_matrix.T,mix.T).T
        with patch("joint_pivot_tuning.DrivenTrace",lambda m:trace):
            record["pivot"]=scan_response(model,None,mix,target)
        record["pivot"].update(target_mm=target,new_calibration=args.calibrate,signed_commands_from_source=args.source_scan,
                               application="Signed 1 microradian X and Y commands",mix_parent=args.mix)
        write(folder/"tracking.json",record)
        write(folder/"scan-calibration.json",record["pivot"])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("solution",type=Path);p.add_argument("--pair",type=Path)
    p.add_argument("--count",type=int,default=4096);p.add_argument("--step",type=float,default=.1)
    p.add_argument("--bore-step",type=float,default=.5)
    p.add_argument("--near",type=float,default=1460.)
    p.add_argument("--label",default="coupled");p.add_argument("--mix")
    for name in ("full","scan","calibrate","source-scan","unit-tilt"):p.add_argument("--"+name,action="store_true")
    run(p.parse_args())
