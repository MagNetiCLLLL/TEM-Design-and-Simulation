"""Solve and linearize the assembled winding and iron circuit on a common tetrahedral mesh."""
import argparse
import json
from contextlib import contextmanager
from unittest.mock import patch

import cupy as cp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import mu_0

from material_relay_fields import SAVE, WORK, digest
from condenser_source_fields import write
from temsim.physics import probe_corrector_fem as fem
from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics.corrector_vacuum_field import _induced_cuda
from temsim.physics.transverse_coil_field import _unit_nodes


def ring_b(points,source):
    """Biot-Savart of an annular pack using elliptic ring integrals and SI quadrature."""
    loops=np.asarray(source._loops)
    kernel=cp.RawKernel(r'''
    extern "C" __global__ void ring(const double* pts,const double* loops,int n,int count,double* out) {
      int i=blockDim.x*blockIdx.x+threadIdx.x;
      if(i>=n) return;
      double x=pts[3*i],y=pts[3*i+1],z=pts[3*i+2],r=hypot(x,y),br=0.,bz=0.;
      for(int l=0;l<count;l++) {
        double a=loops[3*l],d=z-loops[3*l+1],I=loops[3*l+2];
        double den=(a-r)*(a-r)+d*d, s2=(a+r)*(a+r)+d*d;
        if(r<1.e-10*a) { bz+=6.283185307179586e-7*I*a*a/pow(a*a+d*d,1.5); continue; }
        double A=1.,B=sqrt(den/s2),m=4*a*r/s2,S=.5*m,p=1.;
        for(int t=0;t<12;t++){double c=.5*(A-B),h=.5*(A+B);S+=p*c*c;p*=2.;
          B=sqrt(A*B);A=h;if(fabs(c)<2.e-16)break;}
        double K=1.5707963267948966/A,E=K*(1.-S),f=2.e-7*I/sqrt(s2);
        bz+=f*(K+(a*a-r*r-d*d)*E/den);
        br+=f*d/r*(-K+(a*a+r*r+d*d)*E/den);
      }
      out[3*i]=r?br*x/r:0;out[3*i+1]=r?br*y/r:0;out[3*i+2]=bz;
    }''',"ring")
    dev=cp.asarray(np.ascontiguousarray(points));res=cp.empty_like(dev)
    kernel(((len(dev)+127)//128,),(128,),(dev,cp.asarray(loops),np.int32(len(dev)),np.int32(len(loops)),res))
    return res.get()


def impressed(row,pts,order=6):
    if row["kind"]=="round":
        (lo,hi),=row["intervals"]
        coil=AnnularCoilPotential("round",row["inner"]*.001,row["outer"]*.001,(hi-lo)*.001,row["turns"],
                                  (hi+lo)*.0005,quadrature_order=order)
        return ring_b(pts,coil)
    nodes,weights,_=_unit_nodes(json.dumps(row["coils"],sort_keys=True),row["turns"],2,8)
    return _induced_cuda(pts,nodes,weights,winding=True)


class Global:
    def __init__(self,pitch, *, include_hpc=False):
        src=json.loads((WORK/"assembly.json").read_text())
        self.recipe,self.bias=src["recipe"],src["drives"]
        info=json.loads((WORK/f"mesh-{pitch:g}.json").read_text())
        self.mesh_id=info["mesh_id"]
        if digest(self.recipe)!=info["recipe"]: raise ValueError("Material or shape changed after meshing")
        if include_hpc:
            from material_relay_hpc import add_hpc_windings
            self.hpc_geometry = add_hpc_windings(self.recipe, self.bias)
        with np.load(WORK/"meshes"/(self.mesh_id+".npz")) as f:
            xyz,tet,names,labels,far=(f[k] for k in ("xyz","tet","names","index","far"))
        self.original_names=names
        short={name:("air" if name=="air" else f"s{i}") for i,name in enumerate(names)}
        self.recipe={**self.recipe,"solids":[{**s,"name":short[s["name"]]} for s in self.recipe["solids"]]}
        self.mesh=xyz,tet,np.asarray([short[name] for name in names])[labels],far
        good,_=fem._conforming_nodes(xyz,tet,far)
        self.cells=tet[good&(names[labels]!="air")]
        self.out=WORK/"loads"/self.mesh_id
        self.out.mkdir(parents=True,exist_ok=True)

    def verify_hpc_loads(self):
        if not hasattr(self, "hpc_geometry"):
            return
        for key in ("probe_hpc_hexapole_x", "probe_hpc_hexapole_y"):
            row = self.recipe["sources"][key]
            file = self.out / (key + ".npy")
            rec = json.loads(file.with_suffix(".json").read_text())
            if (not file.is_file() or rec.get("source_sha256") != digest(row)
                    or rec["relative_peak_error"] >= .005):
                raise ValueError(f"Rebuild the material-edge load for {key}")

    def sources(self, selected=None):
        xyz,_,_,_=self.mesh
        cells=self.cells
        directed=np.stack(np.broadcast_arrays(cells[:,:1],cells[:,1:]),axis=-1)
        pairs,index=np.unique(np.sort(directed,axis=-1).reshape(-1,2),axis=0,return_inverse=True)
        sign=np.where(directed[:,:,1]>directed[:,:,0],1,-1)
        inverse=np.linalg.inv(xyz[cells[:,1:]]-xyz[cells[:,:1]])
        p0=xyz[pairs[:,0]];dl=xyz[pairs[:,1]]-p0
        for key,row in self.recipe["sources"].items():
            if selected is not None and key not in selected:
                continue
            dest=self.out/(key.replace("/","__")+".npy")
            receipt=dest.with_suffix(".json")
            if dest.is_file() and receipt.is_file():
                stored = json.loads(receipt.read_text())
                if stored["relative_peak_error"] < .005 and (
                    not key.startswith("probe_hpc_hexapole_") or stored.get("source_sha256") == digest(row)
                ):
                    continue
            total=np.zeros(len(pairs))
            # R-Z interpolation for a round winding preserves azimuth and signed NI.
            # Check edge integration against direct Biot-Savart on distributed material edges.
            fn=lambda p:impressed(row,p)
            if row["kind"]=="round":
                from scipy.interpolate import RectBivariateSpline
                ri,ro=row["inner"]*.001,row["outer"]*.001
                (lo,hi),=np.array(row["intervals"])*.001
                def axis(knots,spacing):
                    anchors=np.unique(np.round(knots,12))
                    nodes=[anchors[0]]
                    for end in anchors[1:]:
                        while nodes[-1]<end:
                            u=nodes[-1]
                            nodes.append(min(end,u+spacing(u)))
                    return np.array(nodes)
                rm=1.01*np.linalg.norm(xyz[np.unique(cells),:2],axis=1).max()
                rr=axis([0.,ri,ro,rm],lambda r:.00016+max(0,ri-r,r-ro)*.026)
                zz=axis([min(xyz[cells[:,0],2])-.10,lo,hi,max(xyz[cells[:,0],2])+.10],
                        lambda z:.00018+.026*max(lo-z,z-hi,0))
                R,Z=np.meshgrid(rr,zz,indexing="ij")
                values=impressed(row,np.column_stack([R.ravel(),np.zeros(R.size),Z.ravel()])).reshape(*R.shape,3)
                sr=RectBivariateSpline(rr,zz,values[:,:,0]);sz=RectBivariateSpline(rr,zz,values[:,:,2])
                def fn(p):
                    r=np.hypot(p[:,0],p[:,1]);rs=sr.ev(r,p[:,2])/np.maximum(r,1e-20)
                    result=np.column_stack([rs*p[:,0],rs*p[:,1],sz.ev(r,p[:,2])])
                    near=(r>ri-.0015)&(r<ro+.0015)&(p[:,2]>lo-.0015)&(p[:,2]<hi+.0015)
                    if np.any(near):
                        result[near]=impressed(row,p[near])
                    return result
            # Full saddle paths have mirror/rotation symmetry, but the material mesh does not.
            # Sample all retained AC and multipole windings without projecting away dipole response.
            if row["kind"]!="round":
                from scipy.interpolate import RegularGridInterpolator
                nodes,weights,groups=_unit_nodes(json.dumps(row["coils"],sort_keys=True),row["turns"],2,8)
                lower,upper=nodes.min(axis=0),nodes.max(axis=0)
                def knots(a,b,low,high,h):
                    values=[a]
                    while values[-1]<b:
                        u=values[-1]; values.append(min(b,u+max(h,.045*max(low-u,u-high,0))))
                    return np.array(values)
                box=xyz[np.unique(cells)]
                axes=[knots(box[:,d].min()-.001,box[:,d].max()+.001,lower[d]-.002,upper[d]+.002,.00075)
                      for d in range(3)]
                grid=np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1)
                vals=_induced_cuda(grid.reshape(-1,3),nodes,weights,winding=True).reshape(grid.shape)
                interpolant=RegularGridInterpolator(axes,vals)
                def fn(p):
                    result=interpolant(p)
                    if key.startswith("ac/"):
                        close=np.all((p>lower-.006)&(p<upper+.006),axis=1)
                        if close.any():
                            result[close]=impressed(row,p[close])
                    return result
            rng=np.random.default_rng(20261122)
            sample=rng.choice(len(pairs),2048,replace=False)
            qs=p0[sample]+.47*dl[sample]
            exact=impressed(row,qs);approx=fn(qs)
            error=np.linalg.norm(exact-approx,axis=1)
            reference=np.linalg.norm(exact,axis=1)
            abs_relative=float(np.max(error)/max(np.max(reference),1e-30))
            for t,w in zip(*leggauss(4)):
                for i in range(0,len(pairs),80000):
                    sl=slice(i,i+80000)
                    total[sl]+=.5*w*np.einsum("ij,ij->i",fn(p0[sl]+.5*(t+1)*dl[sl]),dl[sl])/mu_0
            field=np.einsum("ijk,ik->ij",inverse,total[index.reshape(sign.shape)]*sign)
            np.save(dest,field)
            write(dest.with_suffix(".json"),dict(relative_peak_error=abs_relative,source_sha256=digest(row),
                peak_T_per_A=float(np.max(reference)),check_points=len(qs),edge_order=4))
            print("Prepared",key,abs_relative,flush=True)

    def solve(self, drives, output, seed=None):
        def source(recipe,coords,currents):
            total=np.zeros_like(coords)
            for k,v in drives.items():
                if v:total+=v*np.load(self.out/(k.replace("/","__")+".npy"),mmap_mode="r")
            return total
        seed_data=None
        if seed:
            with np.load(seed) as prior:
                seed_data=prior["free_potential_a"]
                previous=json.loads(str(prior["report"]))
                if previous["mesh_id"]!=self.mesh_id:
                    raise ValueError("Continuation seed is on a different material mesh")
                from pathlib import Path
                from material_relay_response import groups
                response_dir=Path(seed).with_name(Path(seed).stem+"-responses")
                changes={k:drives[k]-v for k,v in previous["excitation"].items()}
                controls=groups(previous["excitation"])
                supported={k for keys in controls.values() for k in keys}
                if all(not val for k,val in changes.items() if k not in supported):
                    for name,keys in controls.items():
                        path=response_dir/(name+"-potential.npy")
                        delta=changes[keys[0]]
                        if delta and path.is_file() and all(changes[k]==delta for k in keys):
                            seed_data+=delta*np.load(path)
                if previous["excitation"].get("probe_tl21_lens") and not drives.get("probe_tl21_lens") and not drives.get("condenser_lens_1/physical-pack-0"):
                    xyz,tet,labels,far=self.mesh
                    _,prolong=fem._conforming_nodes(xyz,tet,far)
                    free=np.ones(len(xyz),bool);free[far]=False
                    if prolong is None:
                        zz=xyz[free,2]+.45
                    else:
                        order=np.flatnonzero(np.asarray(prolong.sum(axis=1)).ravel()==1)
                        basis=prolong[order]
                        unit=np.diff(basis.indptr)==1
                        zz=np.empty(prolong.shape[1]);zz[basis[unit].indices]=xyz[order[unit],2]+.45
                    select=np.clip((zz-1.29)/.09,0,1)*np.clip((1.97-zz)/.12,0,1)
                    seed_data*=select**2*(3-2*select)
        checkpoint={}
        for key,amp in drives.items():
            if amp and json.loads((self.out/(key.replace("/","__")+".json")).read_text())["relative_peak_error"]>=.005:
                raise ValueError(f"Source interpolation failed for {key}")
        with patch.object(fem,"impressed_h",source):
            corners,moments,report=fem.solve(self.recipe,self.mesh,drives,backend="cuda",tolerance=2e-8,
                max_iterations=140,potential_checkpoint=checkpoint,potential_seed_a=seed_data,
                progress=lambda i,v:print("Joint nonlinear",i,v,flush=True))
        report.update(mesh_id=self.mesh_id,excitation=drives)
        np.savez_compressed(output,corners_m=corners,moments_Am2=moments,report=json.dumps(report),
                            recipe=json.dumps(self.recipe),**checkpoint)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=("loads","solve"))
    ap.add_argument("--pitch",type=float,default=9.)
    ap.add_argument("--tl",type=float,default=255.71838)
    ap.add_argument("--obj",type=float,default=-28.220275948275862)
    ap.add_argument("--c3",type=float)
    ap.add_argument("--mini",type=float)
    ap.add_argument("--seed")
    ap.add_argument("--pair",action="store_true")
    args=ap.parse_args()
    model=Global(args.pitch)
    if args.stage=="loads": model.sources()
    else:
        q=model.bias.copy();q["probe_tl12_lens"]=args.tl
        for key in q:
            if key.startswith("objective_lens/"):q[key]=args.obj
            if args.c3 is not None and key.startswith("condenser_lens_3/"):q[key]=args.c3
            if args.mini is not None and key.startswith("mini_condenser/"):q[key]=args.mini
        if args.pair:
            q={k:v if k.startswith(("objective_lens/","mini_condenser/")) else 0. for k,v in q.items()}
        model.solve(q,WORK/("global-"+digest(q)[:16]+".npz"),args.seed)
