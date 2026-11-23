"""Evaluate the common magnetic circuit in the current-free bore around the actual rays."""
import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np

from material_relay_fields import WORK, sha
from material_relay_global import impressed
from material_relay_response import groups
from temsim.physics.corrector_vacuum_field import _induced_cuda, _sources

RADIUS=50e-6
# Independent, dimensional Cartesian derivatives of the point-dipole Green function.
KERNEL=r'''
extern "C" __global__ void deriv(const double *p,const double *m,const double *zz,
 int n,int tiles,double scale,double *res) {
 int lane=threadIdx.x,g=blockIdx.x,j=blockIdx.y;
 double val[18]={0.},z=zz[j];
 for(int i=g*512+lane;i<min(n,(g+1)*512);i+=blockDim.x) {
   double r[3]={-p[3*i],-p[3*i+1],z-p[3*i+2]};
   double mm[3]={m[3*i],m[3*i+1],m[3*i+2]};
   double inv=1./(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]),k=1.e-7*inv*sqrt(inv),d=0;
   for(int l=0;l<3;l++)d+=mm[l]*r[l];
   for(int l=0;l<3;l++){
     val[l]+=k*(3*r[l]*d*inv-mm[l]);
     for(int a=0;a<2;a++){
       double al=(l==a);
       val[3+3*a+l]+=3*k*inv*scale*(al*d+r[l]*mm[a]+mm[l]*r[a]-5*r[l]*d*r[a]*inv);
       for(int b=a;b<2;b++){
         double bl=(l==b),ab=(a==b);
         double t=al*mm[b]+bl*mm[a]+mm[l]*ab
           -5*inv*(al*d*r[b]+bl*d*r[a]+r[l]*(mm[a]*r[b]+mm[b]*r[a]+d*ab)+mm[l]*r[a]*r[b])
           +35*inv*inv*r[l]*d*r[a]*r[b];
         val[9+3*(a+b)+l]+=3*k*inv*scale*scale*t*(ab?.5:1.);
       }
     }
   }
 }
 __shared__ double s[128];
 for(int e=0;e<18;e++){
   s[lane]=val[e];__syncthreads();
   for(int stride=64;stride>0;stride/=2){if(lane<stride)s[lane]+=s[lane+stride];__syncthreads();}
   if(!lane)res[(j*tiles+g)*18+e]=s[0];__syncthreads();
 }
}'''


def derivative(points,moments,z):
    p,m=cp.asarray(np.ascontiguousarray(points)),cp.asarray(np.ascontiguousarray(moments))
    n=len(m); tiles=(n+511)//512
    out=np.empty((len(z),6,3))
    kernel=cp.RawKernel(KERNEL,"deriv")
    for i in range(0,len(z),32):
        zz=cp.asarray(z[i:i+32]); buf=cp.empty((len(zz),tiles,18))
        kernel((tiles,len(zz)),(128,),(p,m,zz,np.int32(n),np.int32(tiles),RADIUS,buf))
        out[i:i+len(zz)]=buf.sum(axis=1).get().reshape(-1,6,3)
    return out


def design(p):
    u,v=(p/RADIUS).T
    return np.stack([np.ones_like(u),u,v,u*u,u*v,v*v],axis=1)


def stencil():
    phi=np.arange(12)*np.pi/6
    return np.vstack([[0.,0.],RADIUS*np.stack([np.cos(phi),np.sin(phi)],axis=1)])


def impressed_coeff(recipe,current,z):
    xy=stencil()
    points=np.zeros((len(z),len(xy),3))
    points[:,:,:2]=xy
    points[:,:,2]=z[:,None]
    b=np.zeros_like(points)
    for k,v in current.items():
        if v:b+=v*impressed(recipe["sources"][k],points.reshape(-1,3)).reshape(points.shape)
    return np.einsum("ij,zjl->zil",np.linalg.pinv(design(xy)),b)


def symmetrize(coeff):
    points=stencil()
    total=np.zeros((len(coeff),len(points),3))
    for a in (0,2*np.pi/3,4*np.pi/3):
        rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        fields=np.einsum("pi,zic->zpc",design(points@rot.T),coeff)
        fields[:,:,:2]=fields[:,:,:2]@rot
        total+=fields/3
    return np.einsum("ij,zjl->zil",np.linalg.pinv(design(points)),total)


def process(path,keys,step=.5):
    from material_relay_fields import sha
    root=WORK/(path.stem+"-responses")
    solution_sha=sha(path)
    with np.load(path) as raw:
        corners=raw["corners_m"];info=json.loads(str(raw["report"]));recipe=json.loads(str(raw["recipe"]))
        static=raw["moments_Am2"]
    bias=info["excitation"]
    z=np.linspace(-.455,1.17,round(1625/step)+1)
    check=np.unique(np.r_[np.arange(0,len(z),51),np.where((z>=.90)&(z<=1.16))[0][::17]])
    theta=np.linspace(0,4*np.pi,len(check))
    probes=np.column_stack([.7*RADIUS*np.cos(theta),.7*RADIUS*np.sin(theta),z[check]])
    for key in keys:
        output=WORK/f"{path.stem}-{key}-bore-{step:g}mm.npz"
        response_sha=None if key=="dc" else sha(root/(key+".npy"))
        required=dict(solution_sha256=solution_sha,response_sha256=response_sha,channel=key,
                      evaluator="quadratic-bore-c3-dc-v1",step_mm=step,mesh_id=info["mesh_id"])
        if output.exists():
            with np.load(output) as existing:
                stored=json.loads(str(existing["receipt"]))
                if all(k in stored and stored[k]==v for k,v in required.items()):continue
        amps=bias if key=="dc" else {k:1. for k in groups(bias)[key]}
        mom=static if key=="dc" else np.load(root/(key+".npy"))
        points,weights=_sources(corners,mom,4)
        print("Sampling common circuit",key,len(z),"planes",flush=True)
        c=derivative(points,weights,z)
        bi=impressed_coeff(recipe,amps,z)
        pred=np.einsum("ij,ijk->ik",design(probes[:,:2]),c[check])
        ref=_induced_cuda(probes,points,weights)
        err=np.max(np.linalg.norm(pred-ref,axis=1))
        scale=max(1e-18,np.max(np.linalg.norm(ref,axis=1)))
        if err/scale>1e-4:raise ValueError(f"Bore Taylor error for {key}: {err/scale}")
        low=_induced_cuda(probes,corners.mean(axis=1),mom)
        delta=np.linalg.norm(ref-low,axis=1)
        receipt=dict(bias=info["excitation"],mesh_id=info["mesh_id"],peak_relative_Taylor_error=float(err/scale),
            field_check_n=len(check),order1_vs_4_peak_absolute_T=float(delta.max()),check_peak_induced_T=float(scale),
            local_radius_m=RADIUS,step_mm=step,quadrature=4,source=path.name)
        receipt.update(required)
        b=c+bi
        if not key.startswith("ac_"):
            b=symmetrize(b)
        np.savez_compressed(output,z=z+.45,coef=b,receipt=json.dumps(receipt))
        print(output.name,receipt["peak_relative_Taylor_error"],receipt["order1_vs_4_peak_absolute_T"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("solution",type=Path)
    p.add_argument("keys",nargs="+")
    p.add_argument("--step",type=float,default=.5)
    args=p.parse_args()
    process(args.solution,args.keys,args.step)
