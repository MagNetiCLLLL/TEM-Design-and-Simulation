"""Zero-current paraxial electrical design in unchanged original coil geometry.

This integrates a linear operator, not a finite source. No particle or wave is
emitted, and no focus, D95 or crossover qualification follows from a map fit.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import csv
import importlib.util
import json
import math
import sys
import time

import numpy as np
from scipy.constants import e
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
PROJECT=next(parent for parent in HERE.parents if (parent/"src/temsim").is_dir())
sys.path.insert(0,str(PROJECT/'src'))
from temsim.optics.direct_alignment import _rk4_axisymmetric_larmor_matrices, _electron_momentum_kg_m_s

spec=importlib.util.spec_from_file_location('fixed_cell_tools',HERE.parent/'round-study/fit.py')
tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)

PLANES=(.45,1.07,1.379,1.5991999975)
MOMENTUM=_electron_momentum_kg_m_s(300.)
FACTOR=e/(2*MOMENTUM)
J=np.array(((0.,-1.),(1.,0.)))
CURRENT_LIMIT=10.


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def canonical(radial,theta):
    co,si=math.cos(theta),math.sin(theta)
    rotation=np.array(((co,-si),(si,co)))
    return np.block([[radial[0,0]*rotation,radial[0,1]*rotation],
                     [radial[1,0]*rotation,radial[1,1]*rotation]])


def mechanical(radial,theta,k0,k1):
    source=np.eye(4);source[2:,:2]=-k0*J
    target=np.eye(4);target[2:,:2]=k1*J
    return target@canonical(radial,theta)@source


class TransferModel:
    def __init__(self,cells,scales,gaussian_terms,step_mm):
        self.step_mm=step_mm
        self.parts=[]
        for lower,upper in zip(PLANES[:-1],PLANES[1:]):
            count=int(math.ceil((upper-lower)/(step_mm*.001)))
            z=np.linspace(lower,upper,count+1)
            stage=np.empty(2*count+1);stage[::2]=z;stage[1::2]=(z[:-1]+z[1:])/2
            basis=tools.axis_basis(stage,cells,12)*scales
            target=tools.gaussian(stage,gaussian_terms)
            self.parts.append(dict(z=z,stage=stage,basis=basis,target=target,
                capture=np.array([count],dtype=np.int64),length=upper-lower))
        self.targets=self.evaluate(None)

    def evaluate(self,values):
        result=[]
        for part in self.parts:
            field=part['target'] if values is None else part['basis']@values
            k=FACTOR*field
            radial=_rk4_axisymmetric_larmor_matrices(np.ascontiguousarray(k),part['z'],part['capture'])[0]
            theta=float(np.sum((k[:-2:2]+4*k[1::2]+k[2::2])*np.diff(part['z'])/6))
            result.append(dict(radial=radial,theta=theta,initial_bz_t=float(field[0]),final_bz_t=float(field[-1]),
                canonical=canonical(radial,theta),mechanical=mechanical(radial,theta,k[0],k[-1]),
                symplectic_determinant=float(np.linalg.det(radial)),maximum_bz_t=float(np.max(np.abs(field)))))
        return result

    def residual(self,values,anchor,anchor_weight=.0001,kind='canonical'):
        maps=self.evaluate(values)
        residual=[]
        for part,actual,target in zip(self.parts,maps,self.targets):
            length=part['length']
            scale=np.diag([1.,1.,length,length])
            inverse=np.diag([1.,1.,1/length,1/length])
            if kind=='canonical':
                a=actual['radial']*np.array(((1.,1/length),(length,1.)))
                b=target['radial']*np.array(((1.,1/length),(length,1.)))
                residual.extend(((a-b)/np.maximum(1.,np.abs(b))).ravel())
            else:
                a=scale@actual['mechanical']@inverse
                b=scale@target['mechanical']@inverse
                residual.extend(((a-b)/np.maximum(1.,np.abs(b))).ravel())
            residual.append((actual['theta']-target['theta']))
        # A numerical branch-selection anchor, never a thermal/power model.
        return np.r_[residual,anchor_weight*(values-anchor)]


def serializable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {key:serializable(row) for key,row in value.items()}
    if isinstance(value,list):return [serializable(row) for row in value]
    return value


def main():
    started=time.perf_counter()
    raw=json.loads((BASE/'round-lens-field-match/captured-field-coefficients.json').read_text())
    power_rows={row['key']:row for row in csv.DictReader((BASE/'round-lens-field-match/candidate-currents.csv').open())}
    cells=[];scales=[];initial=[];owners={};target_terms=[]
    for owner,record in raw['ideal_current_coils'].items():
        metadata=record['metadata'];definition=metadata['provenance']['winding_current_definition']
        master=definition['current_a'];parents=[]
        for row in metadata['coils']:
            parent=dict(row['metadata']);parent['turns']=int(round(parent['ampere_turns']/master));parents.append(parent)
        first=len(cells)
        current_cells=tools.cells_for(owner,parents,1,2)
        for cell in current_cells:cell['owner']=owner;cell['original_actual_master_a']=master
        cells+=current_cells
        scale=abs(float(power_rows[owner]['isolated_b2_current_a']))
        scales.extend([scale]*len(current_cells));initial.extend([float(np.sign(master))]*len(current_cells))
        owners[owner]=(first,len(cells));target_terms.extend(raw['gaussian'][owner]['terms_t_m'])
    scales=np.array(scales);initial=np.array(initial);target_terms=np.array(target_terms)
    coarse=TransferModel(cells,scales,target_terms,.2)
    print(json.dumps(dict(stage='Prepared zero-current map model',controls=len(cells),targets=serializable(coarse.targets))),flush=True)
    rng=np.random.default_rng(5701)
    attempts=[];candidates=[]
    seeds=[initial,initial*.7,initial*1.2]
    for _ in range(5):seeds.append(initial+rng.normal(0.,.45,len(initial)))
    for index,seed in enumerate(seeds):
        fit=least_squares(lambda values:coarse.residual(values,initial),seed,bounds=(-CURRENT_LIMIT,CURRENT_LIMIT),
            max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac')
        physics_residual=coarse.residual(fit.x,initial,0.)[:-len(initial)]
        score=float(np.linalg.norm(physics_residual))
        attempts.append(dict(index=index,score=score,nfev=fit.nfev,success=bool(fit.success),optimality=float(fit.optimality),message=fit.message,values=fit.x.tolist()))
        candidates.append((score,fit.x.copy()))
        write_json(HERE/'attempts.json',dict(attempts=attempts,elapsed_s=time.perf_counter()-started))
        print(json.dumps(dict(stage='Coarse canonical search',attempt=index,score=score,nfev=fit.nfev,elapsed_s=time.perf_counter()-started)),flush=True)
    score,best=min(candidates,key=lambda row:row[0])
    fine=TransferModel(cells,scales,target_terms,.05)
    fit=least_squares(lambda values:fine.residual(values,best,.000001),best,bounds=(-CURRENT_LIMIT,CURRENT_LIMIT),
        max_nfev=700,ftol=1e-11,xtol=1e-11,gtol=1e-11,x_scale='jac')
    actual=fine.evaluate(fit.x)
    print(json.dumps(dict(stage='Fine canonical candidate',residual=float(np.linalg.norm(fine.residual(fit.x,best,0.)[:-len(initial)])),nfev=fit.nfev,elapsed_s=time.perf_counter()-started)),flush=True)
    refined=TransferModel(cells,scales,target_terms,.025)
    refined_maps=refined.evaluate(fit.x)
    recipes={}
    for owner,(start,stop) in owners.items():
        windings={}
        for i,cell in enumerate(cells[start:stop],start):
            part=cell['parent_source_id'].rsplit(':',2)[1]
            windings.setdefault(part,[]).append(dict(interval_index=cell['parent_interval_index'],turn_count=cell['turn_count'],
                radial_fraction=cell['radial_fraction'],axial_fraction=cell['axial_fraction'],
                current_ratio=float(fit.x[i]*scales[i]/cell['original_actual_master_a'])))
        recipes[owner]=dict(schema='ideal-coil-current-segments-v1',windings=windings)
    current=fit.x*scales
    matrices=[]
    for index,(a,b,ta,tb) in enumerate(zip(actual,refined_maps,fine.targets,refined.targets)):
        matrices.append(dict(interval_m=[PLANES[index],PLANES[index+1]],target_005=serializable(ta),candidate_005=serializable(a),
            target_0025=serializable(tb),candidate_0025=serializable(b),
            candidate_radial_mesh_max_difference=float(np.max(np.abs(a['radial']-b['radial']))),
            candidate_mechanical_mesh_max_difference=float(np.max(np.abs(a['mechanical']-b['mechanical'])))))
    jac=fine.residual(fit.x,best,0.)[:-len(initial)]
    singular=np.linalg.svd(fit.jac[:-len(initial)],compute_uv=False)
    result=dict(schema='fixed-envelope-zero-current-paraxial-current-design-v1',
        status='ZERO_CURRENT_LINEAR_CANDIDATE_NOT_PHYSICAL_SOURCE_OR_FOCUS_QUALIFIED',
        scope='Pure linear axisymmetric constant-300keV transfer design. No emitter, finite-source particle, wave, aperture interception, hexapole nonlinearity or D95 calculation.',
        integration='Existing low-level project RK4 Larmor operator; canonical rotation and both mechanical endpoint B conversions retained. Production unified Direct Alignment admission is not invoked or bypassed.',
        geometry_policy='Original physical coil intervals, ID/OD, N, pose and metal-area data unchanged; two internal integer-turn axial cells per original interval.',
        current_search_scope=dict(scale='isolated integral-Bz-squared reference current',normalized_bounds=[-CURRENT_LIMIT,CURRENT_LIMIT],resistance_or_power_model=False),
        original_source_reference=raw['boundary'],electrical_control_count=len(cells),current_a=current.tolist(),
        maximum_absolute_current_a=float(np.max(np.abs(current))),current_scale_a=scales.tolist(),
        cells=cells,recipes=recipes,attempts=attempts,
        final_optimizer=dict(success=bool(fit.success),message=fit.message,nfev=fit.nfev,optimality=float(fit.optimality)),
        canonical_dimensionless_residual=jac.tolist(),canonical_dimensionless_residual_norm=float(np.linalg.norm(jac)),
        physical_jacobian_singular_values=singular.tolist(),matrices=matrices,
        limits=['Matching a linear transfer does not prove specimen-surface focus, waist topology, transmission, convergence angle or aberration correction',
            'Canonical match is separate from mechanical map and endpoint-field match, all are reported explicitly',
            'The physical Gun acceleration and all-Z current-dependent Gun transport must be recalculated before any finite-source qualification',
            'The preserved 0.3 eV source energy distribution is not collapsed to this 300keV linear reference in any physical trial'],
        elapsed_s=time.perf_counter()-started)
    write_json(HERE/'candidate.json',result)
    print(json.dumps(dict(stage='DONE',residual=result['canonical_dimensionless_residual_norm'],maximum_current_a=result['maximum_absolute_current_a'],elapsed_s=result['elapsed_s'])),flush=True)


if __name__=='__main__':main()
