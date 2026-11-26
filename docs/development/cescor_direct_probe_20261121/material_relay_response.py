"""Incremental response of every installed magnetic volume about the solved DC operating point."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import mu_0
from scipy.sparse import coo_matrix

from material_relay_fields import WORK,sha
from material_relay_global import Global
from temsim.physics import probe_corrector_fem as fem


def groups(bias):
    keys=list(bias)
    return {"tl":[k for k in keys if k=="probe_tl12_lens"],
            "objective":[k for k in keys if k.startswith("objective_lens/")],
            "mini":[k for k in keys if k.startswith("mini_condenser/")],
            "c3":[k for k in keys if k.startswith("condenser_lens_3/")],
            **{label:[key] for label,key in (
                ("hp1","probe_hp1_hexapole"), ("hp2","probe_hp2_hexapole"),
                ("hpx","probe_hpol_hexapole_x"), ("hpy","probe_hpol_hexapole_y"),
                ("hpcx","probe_hpc_hexapole_x"), ("hpcy","probe_hpc_hexapole_y"),
            ) if "probe_hpc_hexapole_x" in bias and key in bias},
            **{k.replace("/","_"):[k] for k in keys if k.startswith("ac/")}}


def solve(model, path, selected=None):
    model.verify_hpc_loads()
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    from nvmath.sparse.advanced import DirectSolver, DirectSolverMatrixType, DirectSolverMatrixViewType, ExecutionCUDA
    output=WORK/(path.stem+"-responses")
    ticket=output/"receipt.json"
    source_sha=sha(path)
    if ticket.exists():
        old=json.loads(ticket.read_text())
        wanted=selected or list(groups(model.bias))
        if old.get("solution_sha256")==source_sha and all(
            name in old["responses"] and
            (output/(name+".npy")).is_file() and
            old["responses"][name].get("field_sha256")==sha(output/(name+".npy"))
            for name in wanted):
            return
    xyz,tet,labels,far=model.mesh
    good,prolong=fem._conforming_nodes(xyz,tet,far)
    tet,labels=tet[good],labels[good]
    active=labels!="air"
    edges=xyz[tet[:,1:]]-xyz[tet[:,:1]]
    volume=np.abs(np.linalg.det(edges))/6.
    g3=np.linalg.inv(edges).transpose(0,2,1)
    grad=np.concatenate([-g3.sum(axis=1,keepdims=True),g3],axis=1)
    del g3,edges
    mgrad=grad[active]
    free=np.ones(len(xyz),bool)
    free[far]=False
    def extend(x):
        if prolong is not None:return prolong@x
        y=np.zeros(len(xyz));y[free]=x
        return y
    def load(k):
        return np.load(model.out/(k.replace("/","__")+".npy"),mmap_mode="r")
    with np.load(path) as raw:
        rec=json.loads(str(raw["report"]))
        bias=rec["excitation"]
        pot=extend(raw["free_potential_a"])
    if rec["mesh_id"]!=model.mesh_id or rec["residual"]>2e-8:
        raise ValueError("The bias solution has not converged on this mesh")
    field=-np.einsum("ei,eij->ej",pot[tet[active]],mgrad)
    for k,v in bias.items():
        if v:field+=v*load(k)
    norm=np.linalg.norm(field,axis=1)
    h=field/np.maximum(norm[:,None],1e-40)
    sec=np.ones(len(h));par=sec.copy()
    materials={}
    for s in model.recipe["solids"]:
        materials.setdefault(s["material"],[s["curve"],[]])[1].append(s["name"])
    for curve,names in materials.values():
        mask=np.isin(labels[active],names)
        hv,bv=np.asarray(curve["h_a_per_m"]),np.asarray(curve["b_t"])/mu_0
        where=np.clip(np.searchsorted(hv,norm[mask],side="right")-1,0,len(hv)-2)
        if np.max(norm[mask])>hv[-1]:raise ValueError("Outside material data")
        par[mask]=np.diff(bv)[where]/np.diff(hv)[where]
        sec[mask]=np.divide(np.interp(norm[mask],hv,bv),norm[mask],out=par[mask].copy(),where=norm[mask]>0)
    def susceptibility(x):
        return (sec[:,None]-1)*x+((par-sec)*np.einsum("ij,ij->i",h,x))[:,None]*h
    v=volume[active]
    local=np.einsum("eik,ejk->eij",grad,grad)
    directional=np.einsum("eij,ej->ei",mgrad,h)
    local[active]=sec[:,None,None]*local[active]+(par-sec)[:,None,None]*directional[:,:,None]*directional[:,None,:]
    local*=volume[:,None,None]
    ii,jj=np.broadcast_arrays(tet[:,:,None],tet[:,None,:])
    matrix=coo_matrix((local.ravel(),(ii.ravel(),jj.ravel())),shape=(len(xyz),len(xyz))).tocsr()
    del local,ii,jj,grad
    mat=matrix[free][:,free] if prolong is None else (prolong.T@matrix@prolong).tocsr()
    del matrix
    rhs=[]
    columns=groups(bias)
    if selected:
        columns={name:columns[name] for name in selected}
    for keys in columns.values():
        raw=sum(load(k) for k in keys)
        b=np.bincount(tet[active].ravel(),
            (v[:,None]*np.einsum("eij,ej->ei",mgrad,susceptibility(raw))).ravel(),minlength=len(xyz))
        rhs.append(b[free] if prolong is None else prolong.T@b)
    R=np.array(rhs).T.copy(order="F")
    output.mkdir(exist_ok=True)
    print("Factoring the DC tangent for all winding columns",len(columns),flush=True)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2), DirectSolver(csr_matrix(mat.sorted_indices()),cp.asarray(R,order="F"),
            options={"blocking":True,"sparse_system_type":DirectSolverMatrixType.SPD,
                     "sparse_system_view":DirectSolverMatrixViewType.LOWER},execution=ExecutionCUDA()) as fac:
        fac.plan();fac.factorization_config.pivot_eps=0
        if fac.factorize().info:raise ValueError("Singular DC tangent")
        U=fac.solve().get()
    records={}
    for j,(label,keys) in enumerate(columns.items()):
        residual=np.linalg.norm(mat@U[:,j]-R[:,j])/np.linalg.norm(R[:,j])
        if residual>1e-7:raise ValueError(f"Unconverged differential solve for {label}: {residual}")
        u=extend(U[:,j])
        np.save(output/(label+"-potential.npy"),U[:,j])
        dh=sum(load(k) for k in keys)-np.einsum("ei,eij->ej",u[tet[active]],mgrad)
        dm=susceptibility(dh)*v[:,None]
        np.save(output/(label+".npy"),dm)
        records[label]=dict(relative_residual=float(residual),keys=keys,max_dH_per_A=float(np.linalg.norm(dh,axis=1).max()),
                            field_sha256=sha(output/(label+".npy")),potential_sha256=sha(output/(label+"-potential.npy")))
    if ticket.is_file():
        prior=json.loads(ticket.read_text())
        if prior["mesh_id"]!=model.mesh_id or prior["drives"]!=bias:
            raise ValueError("Inconsistent response archive")
        if prior.get("solution_sha256")==source_sha:records={**prior["responses"],**records}
    ticket.write_text(json.dumps(dict(mesh_id=model.mesh_id,drives=bias,responses=records,solution_sha256=source_sha),indent=2))
    print(output,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("solution",type=Path)
    p.add_argument("--keys",nargs="+")
    args=p.parse_args()
    with np.load(args.solution) as saved:
        coils=json.loads(str(saved["report"]))["excitation"]
    solve(Global(9, include_hpc="probe_hpc_hexapole_x" in coils), args.solution, args.keys)
