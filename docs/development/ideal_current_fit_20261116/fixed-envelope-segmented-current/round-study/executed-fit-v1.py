"""Inverse current segmentation inside unchanged archived winding rectangles.

Field-only study: no physical source, ray, wave, resistance or power model.
Each integer-turn cell tiles its parent cross-section. Cell areas are strictly
proportional to physical turns, so equal current recovers the original uniform
continuous current density without moving any conductor envelope.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import math
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import mu_0
from scipy.linalg import svd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PROJECT = HERE.parents[4]
sys.path.insert(0, str(PROJECT / "src"))
from temsim.physics.annular_coil_potential import AnnularCoilPotential

REFERENCE = BASE / "round-lens-field-match/captured-field-coefficients.json"
COUNTS = (1, 2, 4, 8, 16, 32)
RCONDS = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n",encoding="utf-8")


def whole_axis(order: int) -> tuple[np.ndarray, np.ndarray]:
    edges=np.linspace(-.5,3.5,201)
    nodes,weights=leggauss(order)
    half=np.diff(edges)/2
    z=((edges[:-1]+half)[:,None]+half[:,None]*nodes).ravel()
    w=(half[:,None]*weights).ravel()
    u=(nodes+1)/2; wu=weights/2
    distance=.5*u/(1-u); jacobian=.5/(1-u)**2
    return np.r_[-.5-distance,z,3.5+distance], np.r_[wu*jacobian,w,wu*jacobian]


def split_counts(total: int, requested: int) -> np.ndarray:
    count=min(total,requested)
    values=np.full(count,total//count,dtype=int)
    values[:total%count]+=1
    assert np.all(values>0) and int(values.sum())==total
    return values


def cells_for(owner: str, parents: list[dict], radial: int, axial: int) -> list[dict]:
    output=[]
    for index,parent in enumerate(parents):
        nr=split_counts(parent['turns'],radial)
        rb=np.r_[0,np.cumsum(nr)]/parent['turns']
        lower=parent['center_z_m']-parent['length_m']/2
        for row,row_turns in enumerate(nr):
            nz=split_counts(int(row_turns),axial)
            zb=np.r_[0,np.cumsum(nz)]/int(row_turns)
            for column,n in enumerate(nz):
                ri=parent['inner_radius_m']+(parent['outer_radius_m']-parent['inner_radius_m'])*rb[row]
                ro=parent['inner_radius_m']+(parent['outer_radius_m']-parent['inner_radius_m'])*rb[row+1]
                za=lower+parent['length_m']*zb[column]
                zz=lower+parent['length_m']*zb[column+1]
                output.append(dict(source_id=f"{owner}:interval{index}:r{row}:z{column}",
                    parent_source_id=parent['source_id'],parent_interval_index=index,
                    radial_fraction=[float(rb[row]),float(rb[row+1])],
                    axial_fraction=[float(zb[column]),float(zb[column+1])],
                    turn_count=int(n),inner_radius_m=float(ri),outer_radius_m=float(ro),
                    length_m=float(zz-za),center_z_m=float((za+zz)/2),
                    registration=parent['registration']))
        selected=output[-sum(min(int(n),axial) for n in nr):]
        area=(parent['outer_radius_m']-parent['inner_radius_m'])*parent['length_m']
        assert sum(row['turn_count'] for row in selected)==parent['turns']
        assert np.isclose(sum((c['outer_radius_m']-c['inner_radius_m'])*c['length_m'] for c in selected),area,rtol=2e-14,atol=0)
        for c in selected:
            cell_area=(c['outer_radius_m']-c['inner_radius_m'])*c['length_m']
            assert np.isclose(cell_area/area,c['turn_count']/parent['turns'],rtol=2e-12,atol=0)
    return output


def axis_basis(z: np.ndarray, cells: list[dict], order: int) -> np.ndarray:
    nodes,weights=leggauss(order);weights=weights/2
    matrix=np.zeros((len(z),len(cells)))
    for column,cell in enumerate(cells):
        ri=cell['inner_radius_m'];ro=cell['outer_radius_m']
        r0=(ri+ro)/2;dr=(ro-ri)/2
        z0=cell['center_z_m']; dz=cell['length_m']/2
        for nr,wr in zip(nodes,weights):
            radius=r0+dr*nr
            for nz,wz in zip(nodes,weights):
                matrix[:,column]+=.5*mu_0*cell['turn_count']*wr*wz*radius**2/(radius**2+(z-z0-dz*nz)**2)**1.5
    return matrix


def gaussian(z: np.ndarray, terms: np.ndarray) -> np.ndarray:
    return np.exp(-.5*((z[:,None]-terms[:,1])/terms[:,2])**2)@terms[:,0]


def solve_current(matrix: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
    norm=math.sqrt(float(np.dot(weights,target*target)))
    a=matrix*np.sqrt(weights)[:,None]
    scales=np.linalg.norm(a,axis=0)
    normalized=a/scales
    u,s,vh=svd(normalized,full_matrices=False,check_finite=False)
    rhs=u.T@(target*np.sqrt(weights))
    summaries=[];currents=[]
    for rcond in RCONDS:
        valid=s>s[0]*rcond
        current=(vh[valid].T@(rhs[valid]/s[valid]))/scales
        fitted=matrix@current
        error=math.sqrt(float(np.dot(weights,(fitted-target)**2)))/norm
        individual_norm=np.sqrt(np.einsum('ij,i,ij->j',matrix,weights,matrix))
        bound=float(np.dot(np.abs(current),individual_norm)/norm)
        summaries.append(dict(rcond=rcond,retained_rank=int(valid.sum()),
            normalized_matrix_condition=float(s[0]/s[-1]),retained_condition=float(s[0]/s[valid][-1]),
            relative_l2_error=error,maximum_absolute_current_a=float(np.max(np.abs(current))),
            independent_fractional_current_error_gain_upper_bound=bound,
            estimated_binary64_reduction_error_scale=np.finfo(float).eps*bound,
            forward_current_segments=int(np.count_nonzero(current>0)),reverse_current_segments=int(np.count_nonzero(current<0))))
        currents.append(current)
    return summaries,currents


def main() -> None:
    started=time.perf_counter()
    raw=json.loads(REFERENCE.read_text())
    z,weights=whole_axis(24); hold,hold_weights=whole_axis(48)
    output=[]; libraries={};saved={}
    families=[('axial',1,n) for n in COUNTS]+[('radial_axial',r,a) for r,a in ((2,4),(2,8),(4,4),(4,8),(8,8))]
    for owner,record in raw['ideal_current_coils'].items():
        metadata=record['metadata'];definition=metadata['provenance']['winding_current_definition']
        original_current=float(definition['current_a'])
        parents=[]
        for coil in metadata['coils']:
            parent=dict(coil['metadata'])
            parent['turns']=int(round(parent['ampere_turns']/original_current))
            parents.append(parent)
        assert sum(p['turns'] for p in parents)==definition['total_turn_count']
        terms=np.array(raw['gaussian'][owner]['terms_t_m'])
        target=gaussian(z,terms);hold_target=gaussian(hold,terms)
        variants=[];variant_data=[];seen=set()
        for family,radial,axial in families:
            cells=cells_for(owner,parents,radial,axial)
            identity=json.dumps([(c['parent_interval_index'],c['radial_fraction'],c['axial_fraction']) for c in cells])
            if identity in seen:continue
            seen.add(identity)
            basis=axis_basis(z,cells,12)
            metrics,current_options=solve_current(basis,target,weights)
            legacy=axis_basis(z,[dict(p,turn_count=p['turns']) for p in parents],12).sum(axis=1)*original_current
            uniform=basis.sum(axis=1)*original_current
            uniform_difference=np.linalg.norm((uniform-legacy)*np.sqrt(weights))/np.linalg.norm(legacy*np.sqrt(weights))
            variant=dict(family=family,radial_request=radial,axial_request=axial,
                segment_count=len(cells),physical_turn_count=sum(c['turn_count'] for c in cells),
                equal_current_q12_vs_old_q12_relative_l2=float(uniform_difference),
                equal_current_continuum_density_exact=True,regularizations=metrics)
            variants.append(variant);variant_data.append((cells,current_options,basis))
        # A scientific comparison, not a new project acceptance tolerance.
        for rcond in (1e-10,1e-12,1e-14):
            ri=RCONDS.index(rcond)
            winner=min(range(len(variants)),key=lambda i:variants[i]['regularizations'][ri]['relative_l2_error'])
            cells,currents,_=variant_data[winner]
            current=currents[ri]
            h12=axis_basis(hold,cells,12)@current
            h24=axis_basis(hold,cells,24)@current
            denominator=np.linalg.norm(hold_target*np.sqrt(hold_weights))
            metrics=dict(variants[winner]['regularizations'][ri],
                holdout_q12_relative_l2=float(np.linalg.norm((h12-hold_target)*np.sqrt(hold_weights))/denominator),
                holdout_q24_relative_l2=float(np.linalg.norm((h24-hold_target)*np.sqrt(hold_weights))/denominator),
                q12_to_q24_field_relative_l2=float(np.linalg.norm((h24-h12)*np.sqrt(hold_weights))/denominator),
                segment_count=len(cells),winner_variant=winner)
            entries=[dict(cell,current_a=float(amp),current_ratio=float(amp/original_current)) for cell,amp in zip(cells,current)]
            for entry in entries:
                # This constructor admission is field-only, with all original poses.
                p=AnnularCoilPotential(entry['source_id'],entry['inner_radius_m'],entry['outer_radius_m'],entry['length_m'],entry['turn_count']*entry['current_a'],entry['center_z_m'])
                test=np.array((entry['center_z_m']-.02,entry['center_z_m'],entry['center_z_m']+.03))
                check=axis_basis(test,[entry],12)[:,0]*entry['current_a']
                np.testing.assert_allclose(p.axis_bz_t(test),check,rtol=5e-13,atol=1e-10)
            libraries.setdefault(owner,{})[str(rcond)]=dict(metrics=metrics,cells=entries,
                parent_intervals=parents,original_shared_current_a=original_current,
                original_metal_area_records=definition['windings'])
        conservative=libraries[owner]['1e-10']['metrics']
        output.append(dict(owner=owner,original_turn_count=definition['total_turn_count'],
            variants=variants,selected_metrics={key:row['metrics'] for key,row in libraries[owner].items()}))
        # Save an ordinary finite window for viewing; quantitative norms above use the entire axis.
        center=float(np.average(terms[:,1],weights=np.abs(terms[:,0])))
        plot_z=np.linspace(center-.4,center+.4,4001)
        chosen=libraries[owner]['1e-10']['cells']
        saved[owner+'_z_m']=plot_z;saved[owner+'_target_t']=gaussian(plot_z,terms)
        saved[owner+'_candidate_t']=axis_basis(plot_z,chosen,12)@np.array([c['current_a'] for c in chosen])
        result=dict(schema='fixed-envelope-segmented-round-field-study-v1',
            status='FIELD_ONLY_NOT_FOCUS_OR_RUNTIME_QUALIFIED',reference_sha256=sha256(REFERENCE.read_bytes()).hexdigest(),
            scope='No source emission, particle or wave; original winding ID/OD/z envelope, physical counts, poses and original metal area retained. No resistance or power model.',
            representation='Integer-turn cells exactly tile each original radial-axial current rectangle; signed current per cell. Cell cross-sectional area is proportional to its physical turn count.',
            uniform_current_policy='A production implementation may merge equal-current cells to the unchanged legacy source. Different segmented q12 rules otherwise have a separately recorded numerical difference.',
            regularization_scope='SVD cutoffs are explored numerical conditioning choices, not relaxed project error tolerances or field/focus acceptance thresholds.',
            owners=output,elapsed_s=time.perf_counter()-started)
        write_json(HERE/'results.json',result);write_json(HERE/'candidate-library.json',libraries)
        np.savez_compressed(HERE/'profiles.npz',**saved)
        print(json.dumps(dict(owner=owner,segments=conservative['segment_count'],
            full_axis_error=conservative['holdout_q24_relative_l2'],
            current_max_a=conservative['maximum_absolute_current_a'],
            q12_to24=conservative['q12_to_q24_field_relative_l2'],elapsed_s=time.perf_counter()-started)),flush=True)


if __name__=='__main__':
    main()
