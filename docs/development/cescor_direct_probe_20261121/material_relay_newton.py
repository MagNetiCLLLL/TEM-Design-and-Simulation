"""Reassemble the common-circuit Newton system on CUDA for small commissioning offsets."""
import argparse
import json
from pathlib import Path

import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import numpy as np
from numba import njit
from scipy.constants import mu_0
from scipy.sparse import coo_matrix,eye as cpu_eye
from nvmath.sparse.advanced import DirectSolver,DirectSolverMatrixType,DirectSolverMatrixViewType,ExecutionCUDA

from material_relay_fields import WORK,digest
from material_relay_global import Global
from material_relay_response import groups
from temsim.physics import probe_corrector_fem as fem


@njit(cache=True)
def matrix_positions(cell,indptr,indices):
    out=np.empty((len(cell),4,4),np.int32)
    for e in range(len(cell)):
        for i,r in enumerate(cell[e]):
            for j,c in enumerate(cell[e]):
                start=indptr[r]
                n=np.searchsorted(indices[start:indptr[r+1]],c)
                if start+n>=indptr[r+1] or indices[start+n]!=c:raise ValueError("Missing global stiffness entry")
                out[e,i,j]=start+n
    return out


class Circuit:
    def __init__(self,model):
        xyz,tet,labels,bound=model.mesh
        good,P=fem._conforming_nodes(xyz,tet,bound)
        tet,labels=tet[good],labels[good]
        active=labels!="air"
        edges=xyz[tet[:,1:]]-xyz[tet[:,:1]]
        self.v0=np.abs(np.linalg.det(edges))/6.
        g=np.linalg.inv(edges).transpose(0,2,1)
        grad=np.concatenate([-g.sum(axis=1,keepdims=True),g],axis=1)
        del edges,g
        dot=np.einsum("eik,ejk->eij",grad,grad)
        ii,jj=np.broadcast_arrays(tet[:,:,None],tet[:,None,:])
        base=coo_matrix(((self.v0[:,None,None]*dot).ravel(),(ii.ravel(),jj.ravel())),shape=(len(xyz),len(xyz))).tocsr()
        del ii,jj,dot
        base.sort_indices()
        self.t=cp.asarray(tet[active],dtype=cp.int32)
        self.g=cp.asarray(grad[active])
        self.v=cp.asarray(self.v0[active])
        del grad
        self.location=cp.asarray(matrix_positions(tet[active],base.indptr,base.indices))
        self.K=csr_matrix(base); self.M=self.K.copy()
        if P is None:
            free=np.ones(len(xyz),bool);free[bound]=False
            P=cpu_eye(len(xyz),format="csr")[:,free]
        self.P=csr_matrix(P)
        self.curves=[]
        grade={s["name"]:s["material"] for s in model.recipe["solids"]}
        tables={s["material"]:s["curve"] for s in model.recipe["solids"]}
        for key,table in tables.items():
            hv=np.asarray(table["h_a_per_m"]);bv=np.asarray(table["b_t"])/mu_0
            use=np.isin(labels[active],[n for n,v in grade.items() if v==key])
            primitive=np.r_[0,np.cumsum(.5*(bv[:-1]+bv[1:])*np.diff(hv))]
            self.curves.append((key,cp.asarray(use),cp.asarray(hv),cp.asarray(bv),cp.asarray(np.diff(bv)/np.diff(hv)),cp.asarray(primitive)))
        self.corner=xyz[tet[active]]
        self.geo=model
        self.volume={str(k):float(self.v0[labels==k].sum()*1e9) for k in np.unique(labels)}
        self.cell_count=len(tet); self.degenerate=int(np.count_nonzero(~good))

    def assemble(self,u,h0,mode):
        full=self.P@u
        vac=self.K@full
        h=h0-cp.einsum("ei,eij->ej",full[self.t],self.g)
        norm=cp.linalg.norm(h,axis=1)
        energy=.5*full.dot(vac)
        chi=cp.empty_like(norm);parallel=cp.empty_like(norm)
        for name,mask,hv,bv,slope,integral in self.curves:
            n=norm[mask];i=cp.clip(cp.searchsorted(hv,n,side="right")-1,0,len(hv)-2)
            diff=n-hv[i];ind=bv[i]+slope[i]*diff
            chi[mask]=cp.where(n>0,ind/cp.maximum(n,1e-35),slope[i])-1
            parallel[mask]=slope[i]-1
            energy+=self.v[mask].dot(integral[i]+.5*(ind+bv[i])*diff-.5*n*n)
        if mode=="energy":return float(energy)
        m=h*chi[:,None]
        if mode=="moments":
            peaks={key:dict(h_a_per_m=float(norm[mask].max()),b_t=float((mu_0*(1+chi[mask])*norm[mask]).max()))
                   for key,mask,hv,*_ in self.curves}
            if any(peaks[key]["h_a_per_m"]>float(hv[-1])*(1+1e-9) for key,_,hv,*_ in self.curves):
                raise ValueError("Outside the sourced magnetization curve")
            return (self.v[:,None]*m).get(),peaks
        projected=self.v[:,None]*cp.einsum("eij,ej->ei",self.g,m)
        rhs=self.P.T@(vac-cp.bincount(self.t.ravel(),weights=projected.ravel(),minlength=len(full)))
        if mode=="residual":return rhs,float(energy)
        direction=h/cp.maximum(norm[:,None],1e-36)
        pr=cp.einsum("eij,ej->ei",self.g,direction)
        delta=(chi[:,None,None]*cp.einsum("eik,ejk->eij",self.g,self.g)+
               (parallel-chi)[:,None,None]*pr[:,:,None]*pr[:,None,:])*self.v[:,None,None]
        self.M.data[:]=self.K.data+cp.bincount(self.location.ravel(),weights=delta.ravel(),minlength=self.M.nnz)
        mat=(self.P.T@self.M@self.P).tocsr()
        mat.sort_indices()
        return rhs,float(energy),mat

    def solve(self,q,reference,path,*,cold=False,predictor=None):
        h=cp.zeros((len(self.v),3))
        for key,current in q.items():
            if current:
                h+=current*cp.asarray(np.load(self.geo.out/(key.replace("/","__")+".npy"),mmap_mode="r"))
        with np.load(reference) as data:
            orig=json.loads(str(data["report"]))
            if orig["mesh_id"]!=self.geo.mesh_id:raise ValueError("Different mesh")
            u=cp.asarray(data["free_potential_a"])
        previous=orig["excitation"]
        if cold:
            u.fill(0)
        dif={k:q[k]-previous[k] for k in q}
        linearization=predictor if predictor is not None else reference
        resp=linearization.parent/(linearization.stem+"-responses")
        if predictor is not None:
            from material_relay_fields import sha
            receipt=json.loads((resp/"receipt.json").read_text())
            if receipt["solution_sha256"]!=sha(predictor):
                raise ValueError("Initial-guess linearization belongs to a different field")
        for name,keys in groups(q).items():
            delta=dif[keys[0]];direction=resp/(name+"-potential.npy")
            if not cold and delta and direction.exists() and all(dif[k]==delta for k in keys):
                u+=delta*cp.asarray(np.load(direction))
        scale=max(1e-24,float(cp.linalg.norm(self.assemble(cp.zeros_like(u),h,"residual")[0])))
        history=[];factor=None
        try:
            if cold:
                _,_,linear=self.assemble(cp.zeros_like(u),cp.zeros_like(h),"newton")
                susceptibility=cp.empty(len(self.v))
                for key,mask,hv,bv,slope,primitive in self.curves:susceptibility[mask]=slope[0]-1.
                proj=(self.v*susceptibility)[:,None]*cp.einsum("eij,ej->ei",self.g,h)
                b=self.P.T@cp.bincount(self.t.ravel(),weights=proj.ravel(),minlength=self.P.shape[0])
                cp.get_default_memory_pool().free_all_blocks()
                with DirectSolver(linear,b,options={"blocking":True,"sparse_system_type":DirectSolverMatrixType.SPD,
                        "sparse_system_view":DirectSolverMatrixViewType.LOWER},execution=ExecutionCUDA()) as first:
                    first.plan()
                    if first.factorize().info:raise ValueError("Singular low-field material solution")
                    u=first.solve().copy()
            for i in range(140):
                r,energy,mat=self.assemble(u,h,"newton")
                residual=float(cp.linalg.norm(r))/scale
                history.append(residual);print("Material Newton",i,residual,flush=True)
                if residual<2e-8:break
                if factor is None:
                    matrix=mat;b=-r.copy()
                    cp.get_default_memory_pool().free_all_blocks()
                    factor=DirectSolver(matrix,b,options={"blocking":True,"sparse_system_type":DirectSolverMatrixType.SPD,
                                        "sparse_system_view":DirectSolverMatrixViewType.LOWER},
                                        execution=ExecutionCUDA())
                    factor.plan();factor.factorization_config.pivot_eps=0
                else:
                    if matrix.nnz!=mat.nnz:raise ValueError("Sparsity has changed")
                    matrix.data[:]=mat.data;b[:]=-r
                if factor.factorize().info:raise ValueError("Singular common-circuit tangent")
                du=factor.solve()
                descent=float(r.dot(du))
                if descent>=0 or float(cp.linalg.norm(mat@du+r))/float(cp.linalg.norm(r))>1e-6:
                    raise ValueError("Invalid sparse Newton direction")
                for lam in .5**np.arange(18):
                    trial=u+lam*du
                    new=self.assemble(trial,h,"energy")
                    # Near a B-H interpolation knot, two Newton iterates can have
                    # indistinguishable total energy but different field residuals.
                    # Use residual descent to resolve the line search at that scale.
                    if residual < 1e-4 and abs(new-energy)<=2e-13*max(abs(energy),1e-30):
                        rr,_=self.assemble(trial,h,"residual")
                        if float(cp.linalg.norm(rr)) < (1-1e-4*lam)*residual*scale:
                            u=trial;break
                        continue
                    if new<=energy+1e-4*lam*descent:
                        u=trial;break
                    if abs(new-energy)<=2e-13*max(abs(energy),1e-30):
                        rr,_=self.assemble(trial,h,"residual")
                        if cp.linalg.norm(rr)<cp.linalg.norm(r):
                            u=trial;break
                else:raise ValueError("No descent step on the material energy")
            else:raise ValueError("Common-circuit solve did not converge")
        finally:
            if factor is not None:factor.free()
        moments,peaks=self.assemble(u,h,"moments")
        report=dict(orig,excitation=q,currents_A=q,material_peaks=peaks,residual=residual,iterations=i,
                    solver="CUDA P1 energy Newton",iteration_residuals=history)
        if q==previous and not cold:
            with np.load(reference) as data:
                ref=data["moments_Am2"]
                np.testing.assert_allclose(moments,ref,rtol=2e-8,atol=2e-11)
            print("Verified native moments and signed residual on the same mesh",flush=True)
        else:
            pending=path.with_name(path.stem+"-writing.npz")
            np.savez_compressed(pending,corners_m=self.corner,moments_Am2=moments,free_potential_a=u.get(),
                                report=json.dumps(report),recipe=json.dumps(self.geo.recipe))
            pending.replace(path)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("seed",type=Path)
    p.add_argument("--pair",action="store_true");p.add_argument("--cold",action="store_true")
    p.add_argument("--pair-seed",type=Path);p.add_argument("--predictor",type=Path)
    p.add_argument("--pair-predictor",type=Path)
    for k in ("tl","objective","mini","c3"):p.add_argument("--"+k,type=float)
    a=p.parse_args();g=Global(9)
    with np.load(a.seed) as old:q=json.loads(str(old["report"]))["excitation"]
    for label,keys in groups(q).items():
        val=getattr(a,label,None)
        if val is not None:
            for key in keys:q[key]=val
    if a.pair:q={k:(v if k.startswith(("objective_lens/","mini_condenser/")) else 0.) for k,v in q.items()}
    dest=WORK/("global-"+digest(q)[:16]+".npz")
    solver=Circuit(g)
    solver.solve(q,a.seed,dest,cold=a.cold,predictor=a.predictor)
    if a.pair_seed is not None:
        pair={k:v if k.startswith(("objective_lens/","mini_condenser/")) else 0. for k,v in q.items()}
        output=WORK/("global-"+digest(pair)[:16]+".npz")
        if not output.is_file():
            solver.solve(pair,a.pair_seed,output,predictor=a.pair_predictor)
