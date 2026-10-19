"""Offline intensity convergence for retained conditional source quadratures.

Compare five-dimensional base count M and radial Gauss order independently.
Final source probabilities are never normalised after clipping. The existing
stable physical Gaussian reference, actual centroids, and common/lost/gained
decomposition are reused; no source generation or propagation occurs here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent/'src/temsim').is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

import numpy as np
from numpy.polynomial.legendre import leggauss
from scripts import source_energy_strata as _energy_helper
from scripts.source_energy_strata import SCHEMA as ENERGY_SCHEMA, energy_band_cdf_intervals
from temsim.optics.electron_gun.tip_joint_sampling import calibrate_energy, energy_parameters
from scripts.analyse_image_intensity_convergence import (
    CASES, Cloud, Run, analyse, canonical_digest, compare_response,
    gaussian_probability, gaussian_remainder_bound, postprocessing_provenance,
    projector_values, translation_moments, validate_point_spread, write_table,
    stop_summary,
)
from scripts.validate_image_intensity_convergence import array_digest

ORBIT_SCHEMA='offline-common-source-azimuth-orbit-v1'
ANALYSIS_FILES=(Path(__file__),ROOT/'scripts/analyse_image_intensity_convergence.py',
    ROOT/'scripts/validate_image_intensity_convergence.py', Path(_energy_helper.__file__))


def analysis_source_identity(files=ANALYSIS_FILES):
    return {str(path.resolve()):sha256(path.read_bytes()).hexdigest() for path in files}


def verify_analysis_source_identity(identity,files=ANALYSIS_FILES):
    if identity!=analysis_source_identity(files):
        raise RuntimeError('Offline analysis or imported numerical reference changed during evaluation')


ENERGY_FIELDS=('emission_energy_ev','minimum_kinetic_energy_ev','energy_spread_fwhm_ev','energy_half_range_ev','young_decay_width_ev','boersch_sigma_ev')

def validate_probabilities(probability, count):
    p=np.asarray(probability,dtype=float)
    if (p.shape!=(count,) or not np.isfinite(p).all() or np.any(p<=0)
            or not math.isclose(math.fsum(p),1.,rel_tol=0.,abs_tol=32*np.finfo(float).eps)):
        raise ValueError('Positive expanded-base probabilities must retain total source measure one')
    return p

def expected_energy_orbit(original_base,cuts,orbit_size):
    """Reconstruct the affine Boersch-CDF partition and both-azimuth orbit.

    Original 5D order: position radius/azimuth, direction azimuth, Young,
    Boersch. Each original base has measure1/M; a positive conditional
    interval has its actual width/M. Zero-width strata carry zero measure
    and may be omitted, without renormalising the retained intervals.
    """
    original=np.asarray(original_base,dtype=float);edges=np.asarray(cuts,dtype=float)
    if (original.ndim!=2 or original.shape[1]!=5 or not len(original)
            or not np.isfinite(original).all() or np.any((original<0)|(original>1))):
        raise ValueError('Original finite5D CDF bases are required')
    m=len(original)
    if (edges.shape!=(m,4) or not np.isfinite(edges).all() or np.any(np.diff(edges,axis=1)<0)
            or np.any(edges[:,0]!=0) or np.any(edges[:,-1]!=1)):
        raise ValueError('Every energy partition must cover the entire original Boersch CDF')
    if type(orbit_size) is not int or orbit_size not in (2,4,8,16):raise ValueError('Invalid orbit size')
    half=orbit_size//2;points=[];probability=[];parents=[];strata=[];orbits=[];lower=[];upper=[]
    for parent,point in enumerate(original):
        for stratum,(lo,hi) in enumerate(zip(edges[parent,:-1],edges[parent,1:])):
            if hi==lo:continue
            mapped=point.copy();mapped[4]=lo+(hi-lo)*point[4]
            for j in range(half):
                rotated=mapped.copy()
                if orbit_size!=2:rotated[[1,2]]=(rotated[[1,2]]+j/orbit_size)%1.
                points.append(rotated);probability.append(((hi-lo)/m)/half)
                parents.append(parent);strata.append(stratum);orbits.append(j);lower.append(lo);upper.append(hi)
    result=dict(points=np.asarray(points),base_probability=np.asarray(probability),
        independent_base_index=np.asarray(parents,dtype=np.int64),energy_stratum_index=np.asarray(strata,dtype=np.int64),
        half_orbit_index=np.asarray(orbits,dtype=np.int64),interval_lower=np.asarray(lower),interval_upper=np.asarray(upper))
    validate_probabilities(result['base_probability'],len(points))
    return result

def weighted_root_probability(roots,base_probability,cases):
    """Per-case located bracket uncertainty with actual source probabilities.

    Existing root-status/bracket/residual validation remains mandatory in the
    loader. This expression only replaces the uniform1/(2*base_count) factor.
    """
    p=validate_probabilities(base_probability,len(base_probability))
    channels={f'{case}:{sign:+d}':0. for case in cases for sign in (1,-1)}
    for root in roots:
        i=root['base_index'];key=root['channel_key'];lo,hi=root['bracket']
        if (type(i) is not int or not 0<=i<len(p) or key not in channels
                or not 0<=lo<=hi<=1):raise ValueError('Invalid root/base measure binding')
        channels[key]+=(p[i]*(hi-lo))/2
    per_case={case:channels[f'{case}:+1']+channels[f'{case}:-1'] for case in cases}
    # Match the archived collector's channel order exactly. Grouping each
    # +/- pair first can change the final budget by one float64 ULP.
    differences={case:sum(value for key,value in channels.items()
        if key.rsplit(':',1)[0] in (cases[0],case)) for case in cases[1:]}
    return dict(per_channel=channels,per_case=per_case,
        case_minus_baseline=differences)

def expected_source_table(partitions,expanded_points,base_probability,order):
    """Reconstruct every physical node, weight, sign, ID and zero anchor.

    Matches the declared collector arithmetic: uniform helper nodeweight
    times(R*base_probability[base]), then each of the two signs receives1/2.
    No survival filter, source renormalisation, or coordinate fitting occurs.
    """
    points=np.asarray(expanded_points,dtype=float);r=len(points)
    p=validate_probabilities(base_probability,r)
    if points.shape!=(r,5) or not np.isfinite(points).all():raise ValueError('Invalid expanded source coordinates')
    if type(order) is not int or order not in (2,4):raise ValueError('Gauss order must be2or4')
    if [part['base_index'] for part in partitions]!=list(range(r)):
        raise ValueError('Every actual expanded base requires exactly one ordered radial partition')
    nodes,gw=leggauss(order);base=[];cell=[];radial=[];weight=[]
    for part in partitions:
        i=part['base_index'];edges=np.asarray(part['edges'],dtype=float);widths=np.diff(edges)
        if edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or edges[0]!=0 or edges[-1]!=1 or np.any(widths<=0):
            raise ValueError('Radial cells must exactly cover[0,1]')
        for j,(lo,width) in enumerate(zip(edges[:-1],widths)):
            base.extend([i]*order);cell.extend([j]*order);radial.extend(lo+width*(nodes+1)/2)
            uniform=width*gw/(2*r)
            weight.extend(uniform*(r*p[i]))
    base=np.asarray(base,dtype=np.int64);cell=np.asarray(cell,dtype=np.int64);radial=np.asarray(radial);weight=np.asarray(weight)
    positive=np.column_stack((points[base,:2],radial,points[base,2:]))
    anchor_base=np.repeat(np.arange(r,dtype=np.int64),2);anchor_radial=np.tile([0.,1.],r)
    anchors=np.column_stack((points[anchor_base,:2],anchor_radial,points[anchor_base,2:]))
    coordinates=np.concatenate((np.repeat(positive,2,axis=0),np.repeat(anchors,2,axis=0)))
    weights=np.r_[np.repeat(weight/2,2),np.zeros(4*r)]
    return dict(coordinates=coordinates,weights=weights,signs=np.tile([1,-1],len(coordinates)//2),
        ray_ids=np.arange(len(coordinates),dtype=np.int64),base_index=np.r_[np.repeat(base,2),np.repeat(anchor_base,2)],
        angular_cdf=np.r_[np.repeat(radial,2),np.repeat(anchor_radial,2)],
        cell_index=np.r_[np.repeat(cell,2),np.full(4*r,-1,dtype=np.int64)])

def verify_source_table(actual,expected):
    for key,value in expected.items():
        if key not in actual or not np.array_equal(actual[key],value):
            raise ValueError(f'Final source table does not reconstruct its captured source measure: {key}')
    return dict(status='EXACT_ARCHIVED_SOURCE_MEASURE_RECONSTRUCTION',
        probability=math.fsum(actual['weights']),positive_nodes=int(np.count_nonzero(actual['weights']>0)),
        zero_probe_nodes=int(np.count_nonzero(actual['weights']==0)))

def source_method(document):
    energy=document.get('energy_stratification')
    if energy is None:
        return dict(schema='uniform-five-dimensional-conditional-v1',identity='uniform-five-dimensional-conditional-v1')
    physical=document['physical_protocol'].get('energy_stratification')
    cuts=document['energy_cuts_metadata']
    if (energy.get('schema')!=ENERGY_SCHEMA or cuts.get('schema')!=ENERGY_SCHEMA
            or physical!={"schema":ENERGY_SCHEMA,"band_ev":energy['band_ev'],"cuts_metadata":cuts}
            or cuts['band_ev']!=energy['band_ev']):
        raise ValueError('Energy method, band and captured calibration must match the physical protocol')
    return dict(schema=ENERGY_SCHEMA,band_ev=energy['band_ev'],identity=canonical_digest(physical))

def verify_energy_snapshot(snapshot,cuts_metadata):
    emitters=[node['attributes'] for node in snapshot['graph']['nodes']
        if node.get('type')=='temsim.optics.electron_gun.emitter:ColdFieldEmitter']
    if len(emitters)!=1:raise ValueError('Energy strata require exactly one captured classical tip emitter')
    emitter=SimpleNamespace(**{name:emitters[0][name] for name in ENERGY_FIELDS})
    calibration=calibrate_energy(*energy_parameters(emitter))
    if asdict(calibration)!=cuts_metadata['energy_calibration']:
        raise ValueError('Archived energy cuts do not use the actual captured emitter calibration')
    return emitter

def load_energy_measure(directory,document,bases,expanded,source_snapshot):
    method=source_method(document)
    if method['schema']!=ENERGY_SCHEMA:raise ValueError('This guard is only for explicit energy-stratified archives')
    cuts=checked_arrays(directory,document['energy_cuts_path'],document['energy_cuts_sha256'],'Conditional energy CDF cuts')
    measure=checked_arrays(directory,document['source_base_measure_path'],document['source_base_measure_sha256'],'Original source-base measure')
    if (array_digest(cuts)!=document['energy_cuts_digest']
            or array_digest(measure)!=document['source_base_measure_digest']):
        raise ValueError('Energy cuts or original source measure digest mismatch')
    emitter=verify_energy_snapshot(source_snapshot,document['energy_cuts_metadata'])
    calculated=energy_band_cdf_intervals(emitter,bases[:,3],method['band_ev'])
    for key in ('edges','interval_probability','conditional_minimum_ev','conditional_maximum_ev'):
        if not np.array_equal(cuts[key],getattr(calculated,key)):
            raise ValueError('Energy CDF cuts differ from the captured physical source law: '+key)
    expected=expected_energy_orbit(bases,cuts['edges'],document.get('azimuth_orbit_size',2))
    if not np.array_equal(expanded,expected.pop('points')):
        raise ValueError('Expanded energy/orbit coordinates do not reconstruct the original source measure')
    if set(measure)!=set(expected):raise ValueError('Incomplete original source-base mapping')
    for key,value in expected.items():
        if not np.array_equal(measure[key],value):
            raise ValueError('Incorrect original source-base mapping: '+key)
    if document['expanded_base_count']!=len(expanded):raise ValueError('Incorrect expanded energy/orbit base count')
    return dict(method=method,cuts=cuts,measure=measure,
        receipt=dict(status='EXACT_CAPTURED_ENERGY_MEASURE_RECONSTRUCTION',method=method,
            energy_cuts_sha256=document['energy_cuts_sha256'],source_base_measure_sha256=document['source_base_measure_sha256'],
            independent_base_count=len(bases),expanded_base_count=len(expanded),probability=float(measure['base_probability'].sum())))

def validate_weighted_roots(partition,diagnostics,measure):
    probability=measure['base_probability']
    # Reuse all existing convergence, interval and actual-residual guards.
    # Its returned uniform budget is deliberately discarded, never relabelled.
    located_root_bounds(partition,diagnostics,len(probability))
    result=weighted_root_probability(partition['roots'],probability,CASES)
    archived=partition['weighted_root_budget']
    if (archived['per_channel']!=result['per_channel']
            or archived['difference_bound_by_case']!=result['case_minus_baseline']):
        raise ValueError('Located-root budget does not use the original nonuniform source probabilities')
    return result

def validate_energy_source_table(table,partition,expanded,measure,order):
    expected=expected_source_table(partition['partitions'],expanded,measure['base_probability'],order)
    receipt=verify_source_table(table,expected)
    for key,values in measure.items():
        if key not in table or not np.array_equal(table[key],values[table['base_index']]):
            raise ValueError('Final emitted node lost its original energy/base mapping: '+key)
    return receipt

def require_same_refinement_method(coarse,fine):
    if coarse['identity']!=fine['identity']:
        raise ValueError('M/K/G refinement cannot mix uniform and energy-stratified methods or different fixed energy bands/calibrations')

@dataclass
class ConditionalRun:
    label: str
    base_count: int
    gauss_order: int
    base_seed: int
    base_points: np.ndarray
    partition_digest: str
    source_table_digest: str
    protocol_digest: str
    step_mm: float
    clouds: dict[str, Cloud]
    provenance: dict
    root_probability_bounds: dict[str,float]
    uncertainty: dict
    azimuth_orbit_size: int = 2
    expanded_base_points: np.ndarray | None = None
    anchor_coordinates: np.ndarray | None = None
    anchor_signs: np.ndarray | None = None
    anchor_margins_mm: np.ndarray | None = None
    source_measure_method: dict | None = None
    energy_cdf_edges: np.ndarray | None = None
    base_probability: np.ndarray | None = None

    @property
    def expanded_base_count(self):
        return len(self.expanded_base_points) if self.expanded_base_points is not None else self.base_count*self.azimuth_orbit_size//2


def checked_bytes(directory,path,digest,label):
    data=(directory/path).read_bytes()
    if not isinstance(digest,str) or sha256(data).hexdigest()!=digest:
        raise ValueError(f'{label} file checksum mismatch')
    return data


def checked_json(directory,path,digest,label):
    return json.loads(checked_bytes(directory,path,digest,label))


def checked_arrays(directory,path,digest,label):
    import io
    with np.load(io.BytesIO(checked_bytes(directory,path,digest,label)),allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def located_root_bounds(partition,diagnostics,base_count):
    """Probability uncertainty of located roots, per physical case and sign.

    Each +/- channel has source measure1/(2M). Delta-case probability error
    includes both its own and baseline's brackets. This is only a bound on
    boundaries actually found, not on missed intervals or propagation error.
    """
    channels=tuple(f'{case}:{sign:+d}' for case in CASES for sign in (1,-1))
    widths=dict.fromkeys(channels,0.)
    roots=partition['roots']
    if (not diagnostics.get('all_located_roots_converged')
            or diagnostics.get('root_count')!=len(roots)):
        raise ValueError('Located conditional roots did not all converge')
    tolerance=float(diagnostics['angular_cdf_tolerance'])
    if not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Invalid recorded root tolerance')
    for root in roots:
        lo,hi=map(float,root['bracket']);u=float(root['u'])
        width=hi-lo;channel=root['channel_key'];base=root['base_index']
        if (channel not in widths or type(base) is not int or not 0<=base<base_count
                or not 0<=lo<=u<=hi<=1 or width>tolerance*(1+1e-12)
                or not np.isfinite(root['actual_margin_residual'])
                or not np.isclose(width,root['bracket_width'],atol=1e-15,rtol=1e-12)):
            raise ValueError('Invalid actual root bracket or residual')
        widths[channel]+=width/(2*base_count)
    case_bounds={case:widths[f'{case}:+1']+widths[f'{case}:-1'] for case in CASES}
    delta_bounds={case:case_bounds['baseline']+case_bounds[case] for case in CASES[1:]}
    return delta_bounds,dict(per_channel_probability=widths,per_case_probability=case_bounds,
        all_channel_probability=sum(widths.values()),
        role='Actual located-root brackets only; probability before clipping; no fitted aperture or source renormalisation.')


def load_final_anchors(directory,receipt,table,expanded):
    """Bind actual C2 endpoint margins to SHA-verified source-table probes."""
    if 'anchor_path' not in receipt:
        return None,None,None,dict(status='NO_RETAINED_FINAL_ANCHOR_MARGINS')
    arrays=checked_arrays(directory,receipt['anchor_path'],receipt['anchor_sha256'],'Final C2 anchor margins')
    margins=arrays['anchor_margin_mm'];n=len(expanded)
    if margins.shape!=(2*n,2*len(CASES)) or not np.isfinite(margins).all():
        raise ValueError('Final C2 anchor margins have invalid shape or values')
    mask=(table['weights']==0)&(table['cell_index']==-1)
    coordinates=table['coordinates'][mask];signs=table['signs'][mask]
    cdf=np.column_stack((np.repeat(expanded[:,:2],2,axis=0),np.tile([0.,1.],n),
        np.repeat(expanded[:,2:],2,axis=0)))
    if (not np.array_equal(coordinates,np.repeat(cdf,2,axis=0))
            or not np.array_equal(signs,np.tile([1,-1],2*n))):
        raise ValueError('Final margin rows do not match all archived zero-weight endpoint probes')
    by_probe=margins.reshape(2*n,len(CASES),2).transpose(0,2,1).reshape(4*n,len(CASES))
    return coordinates,signs,by_probe,dict(status='SHA_VERIFIED_FINAL_C2_ENDPOINTS',
        path=receipt['anchor_path'],sha256=receipt['anchor_sha256'],probe_count=4*n,
        role='Observed zero-current physical probes; no weight or current contribution.')


def compare_final_anchors(coarse,fine,axis):
    """Observe cross-batch physical drift, including case-response cancellation.

    No source IDs are paired across different tables. Exact CDF coordinates
    and explicit inversion signs identify common launch probes; final gun
    energies/trajectories are allowed to differ and their effect is measured.
    This finite endpoint comparison cannot certify the unknown interior.
    """
    result=dict(axis=axis,coarse_run=coarse.label,fine_run=fine.label,
        role='Observed final-batch endpoint diagnostic only; neither a certified interior error bound nor a convergence gate.')
    if coarse.anchor_coordinates is None or fine.anchor_coordinates is None:
        return dict(result,status='NOT_AVAILABLE_RETAINED_FINAL_ANCHORS_MISSING')
    def index(run):
        keys=[tuple(row)+(int(sign),) for row,sign in zip(run.anchor_coordinates,run.anchor_signs)]
        if len(set(keys))!=len(keys):raise ValueError('Final anchor CDF coordinates and signs must be unique')
        return {key:i for i,key in enumerate(keys)}
    a,b=index(coarse),index(fine);shared=sorted(set(a)&set(b))
    result.update(coarse_probe_count=len(a),fine_probe_count=len(b),matched_physical_probe_count=len(shared),
        identity='Exact six-dimensional source CDF coordinate plus inversion sign; no source-ID pairing.')
    if not shared:return dict(result,status='NO_EXACT_SHARED_PHYSICAL_PROBES')
    ma=coarse.anchor_margins_mm[[a[key] for key in shared]]
    mb=fine.anchor_margins_mm[[b[key] for key in shared]]
    difference=mb-ma
    response=(mb-mb[:,[0]])-(ma-ma[:,[0]])
    def metrics(values):
        return dict(maximum_absolute_mm=float(np.abs(values).max()),
            rms_mm=float(np.sqrt(np.mean(values**2))),signed_mean_mm=float(values.mean()))
    return dict(result,status='OBSERVED_NOT_CERTIFIED',cases={case:dict(
        absolute_margin_change=metrics(difference[:,i]),
        case_minus_baseline_margin_change=metrics(response[:,i])) for i,case in enumerate(CASES)})


def load_conditional_runs(paths):
    """Bind screen arrays to shared full-support tables, guns and snapshots."""
    runs=[];inputs=[];pending=[]
    for path in paths:
        path=Path(path);directory=path.parent if path.is_file() else path
        manifest=directory/'manifest.json';data=manifest.read_bytes();document=json.loads(data)
        if document.get('schema')!='conditional-source-convergence-v1':
            raise ValueError('Unsupported conditional source archive schema')
        entry=dict(input=str(manifest.resolve()),manifest_sha256=sha256(data).hexdigest(),status=document.get('status'),
                   error=document.get('error'))
        inputs.append(entry)
        if document.get('status')!='COMPLETE':
            pending.append(dict(input=entry['input'],status=document.get('status'),error=document.get('error')))
        complete={label:row for label,row in document.get('runs',{}).items() if row.get('status')=='COMPLETE'}
        pending.extend(dict(input=entry['input'],run=label,status=row.get('status')) for label,row in document.get('runs',{}).items() if row.get('status')!='COMPLETE')
        if not complete:
            pending.append(dict(input=entry['input'],status='NO_COMPLETED_SCREEN_GROUP'))
            continue
        protocol=document['protocol'];physical=document['physical_protocol']
        if (canonical_digest(protocol)!=document['protocol_digest']
                or canonical_digest(physical)!=document['physical_protocol_digest']):
            raise ValueError('Conditional protocol checksum mismatch')
        orbit_schema=physical.get('azimuth_orbit_schema')
        if orbit_schema not in (None,ORBIT_SCHEMA):
            raise ValueError('Unknown source-azimuth physical protocol schema')
        comparison_physical={key:value for key,value in physical.items() if key!='azimuth_orbit_schema'}
        psf=validate_point_spread(document['point_spread'])
        controls=projector_values(document['projector_controls'])
        if (protocol.get('physical_protocol')!=physical
                or physical['step_mm']!=document['step_mm']
                or physical['source_base_seed']!=document['base_seed']
                or projector_values(physical['projector_controls'])!=controls
                or protocol['implementation']!=document['implementation']):
            raise ValueError('Conditional metadata differs from its captured physical protocol')
        base_arrays=checked_arrays(directory,document['base_points_path'],document['base_points_sha256'],'5D base')
        if array_digest(base_arrays)!=document['base_digest']:
            raise ValueError('Five-dimensional base digest mismatch')
        bases=base_arrays['points'];m=document['base_count'];k=document.get('azimuth_orbit_size',2)
        if (type(m) is not int or m<1 or type(k) is not int or k not in (2,4,8,16)
                or bases.shape!=(m,5)):
            raise ValueError('Invalid independent base count or common-azimuth orbit')
        stratified='energy_stratification' in document
        expanded_count=document['expanded_base_count'] if stratified else m*k//2
        if type(expanded_count) is not int or expanded_count<1:
            raise ValueError('Invalid expanded radial base count')
        if document.get('expanded_base_count',m)!=expanded_count:
            raise ValueError('Expanded radial base count differs from independent M and orbit K')
        if 'expanded_base_points_path' in document:
            expanded_arrays=checked_arrays(directory,document['expanded_base_points_path'],document['expanded_base_points_sha256'],'Expanded 5D base')
            if array_digest(expanded_arrays)!=document['expanded_base_digest']:
                raise ValueError('Expanded five-dimensional base digest mismatch')
            expanded=expanded_arrays['points']
        elif k==2:expanded=bases
        else:raise ValueError('Orbit sampling requires the actual expanded five-dimensional base archive')
        energy_measure=None
        if stratified:
            row=next(iter(complete.values()))
            source_snapshot=checked_json(directory,row['snapshot_path'],row['snapshot_sha256'],'Captured energy-law snapshot')
            energy_measure=load_energy_measure(directory,document,bases,expanded,source_snapshot)
        else:
            _validate_expanded_bases(bases,expanded,k)
        partition=checked_json(directory,document['partition_path'],document['partition_sha256'],'Radial partition')
        partition_digest=canonical_digest(partition)
        if partition_digest!=document['partition_digest']:
            raise ValueError('Radial partition digest mismatch')
        if energy_measure is None:
            root_bounds,root_summary=located_root_bounds(partition,document['root_diagnostics'],expanded_count)
        else:
            weighted=validate_weighted_roots(partition,document['root_diagnostics'],energy_measure['measure'])
            root_bounds=weighted['case_minus_baseline']
            root_summary=dict(per_channel_probability=weighted['per_channel'],per_case_probability=weighted['per_case'],
                all_channel_probability=sum(weighted['per_channel'].values()),
                role='Located actual root brackets weighted by complete original conditional energy-CDF mass; no survival normalization. Uniform helper budget is not used.')
        groups={}
        for label,row in complete.items():
            if (row['base_count']!=m or row.get('azimuth_orbit_size',2)!=k
                    or row.get('expanded_base_count',m)!=expanded_count or row['partition_digest']!=partition_digest
                    or projector_values(row['projector_controls'])!=controls
                    or row['receiver_z_mm']!=document['receiver_z_mm']
                    or any(row[k]!=document['implementation'] for k in ('implementation_before','implementation_after'))):
                raise ValueError('Conditional run changed its physical state, partition, plane or implementation')
            snapshot=checked_json(directory,row['snapshot_path'],row['snapshot_sha256'],'Run snapshot')
            if stratified:verify_energy_snapshot(snapshot,document['energy_cuts_metadata'])
            expected=canonical_digest(dict(protocol=document['protocol_digest'],state=snapshot['digest'],
                target_z_mm=row['receiver_z_mm'],run_id=label,source_table=row['source_table_digest'],partition=partition_digest))
            if row['request_identity']!=expected:
                raise ValueError('Conditional run request identity mismatch')
            arrays=checked_arrays(directory,row['particles_path'],row['particles_sha256'],'Screen particles')
            cloud=Cloud(arrays['xy_m']*1e3,arrays['weight'],arrays['survivor'].astype(bool),arrays['source_id'],
                arrays['blocked_key'],arrays['blocked_z_mm'],arrays['final_alive'].astype(bool))
            if not np.array_equal(cloud.survivor,(cloud.blocked_key=='flu_screen')&(cloud.weight>0)):
                raise ValueError('Screen survivor mask differs from recorded physical first hits')
            group=groups.setdefault(row['gauss_order'],{})
            if row['case'] in group:
                raise ValueError('Duplicate conditional case in one Gauss group')
            group[row['case']]=(cloud,row)
        for order,group in groups.items():
            label=(f'M{m}_E3_K{k}_G{order}' if stratified else f'M{m}_G{order}' if k==2 else f'M{m}_K{k}_G{order}')
            if set(group)!=set(CASES):
                pending.append(dict(input=entry['input'],run=label,status='INCOMPLETE_SCREEN_CASE_GROUP'))
                continue
            receipt=document['sources'][label]
            if (receipt['base_count']!=m or receipt['gauss_order']!=order
                    or receipt.get('azimuth_orbit_size',2)!=k or receipt.get('expanded_base_count',m)!=expanded_count
                    or receipt['partition_digest']!=partition_digest):
                raise ValueError('Final source receipt belongs to another conditional partition')
            table_record=checked_json(directory,receipt['table_record_path'],receipt['table_record_sha256'],'Source table record')
            table_digest=canonical_digest(table_record)
            table=checked_arrays(directory,receipt['table_path'],receipt['table_sha256'],'Source table')
            if table_record.get('role')!='quadrature' or table_digest!=receipt['source_table_digest']:
                raise ValueError('Final source must be a full physical quadrature with matching identity')
            for key in ('coordinates','weights','signs','ray_ids'):
                if not np.array_equal(np.asarray(table_record[key]),table[key]):
                    raise ValueError('Source table record differs from retained arrays')
            if (table['coordinates'].shape!=(len(table['weights']),6)
                    or not np.isfinite(table['coordinates']).all() or np.any((table['coordinates']<0)|(table['coordinates']>1))
                    or not np.isin(table['signs'],(-1,1)).all()):
                raise ValueError('Invalid explicit physical source CDF coordinates')
            mapping_verified=False
            if 'base_index' in table and 'angular_cdf' in table:
                indices=table['base_index'];u=table['angular_cdf']
                if (indices.shape!=table['weights'].shape or indices.dtype.kind not in 'iu'
                        or np.any((indices<0)|(indices>=expanded_count)) or u.shape!=indices.shape):
                    raise ValueError('Invalid archived conditional base/radial mapping')
                expected_cdf=np.column_stack((expanded[indices,:2],u,expanded[indices,2:]))
                if not np.array_equal(table['coordinates'],expected_cdf):
                    raise ValueError('Archived source table differs from its actual base/radial mapping')
                mapping_verified=True
            table_measure_receipt=None
            if stratified:
                actual_snapshot=checked_json(directory,receipt['snapshot_path'],receipt['snapshot_sha256'],'Final source energy-law snapshot')
                verify_energy_snapshot(actual_snapshot,document['energy_cuts_metadata'])
                table_measure_receipt=validate_energy_source_table(table,partition,expanded,energy_measure['measure'],order)
            source=checked_arrays(directory,receipt['source_path'],receipt['source_sha256'],'Actual gun source')
            source_digest=array_digest(source)
            if source_digest!=receipt['source_digest']:
                raise ValueError('Actual gun source digest mismatch')
            source_weight=source['gun_exit_weight'];source_ids=source['gun_exit_ray_id']
            if (not np.array_equal(source_weight,table['weights']) or not np.array_equal(source_ids,table['ray_ids'])):
                raise ValueError('Actual gun changed the final table source IDs or probabilities')
            for cloud,row in group.values():
                if (row['source_table_digest']!=table_digest or row['source_digest']!=source_digest
                        or not np.array_equal(cloud.weight,source_weight) or not np.array_equal(cloud.source_id,source_ids)):
                    raise ValueError('Screen cases did not share the exact final source table and physical gun')
            current=receipt.get('emitted_current_a')
            if current is not None and (not np.isfinite(current) or current<0):
                raise ValueError('Invalid conditional source current')
            if current!=physical['nominal_current_a']:
                raise ValueError('Actual source current differs from the physical protocol')
            anchor_coordinates,anchor_signs,anchor_margins,anchor_receipt=load_final_anchors(directory,receipt,table,expanded)
            provenance={case:dict(row,emitted_current_a=current) for case,(cloud,row) in group.items()}
            uncertainty=dict(topology_completeness=document['root_diagnostics'].get('completeness'),
                transport_implementation_sha256=document['implementation'],
                physical_protocol_comparison_digest=canonical_digest(comparison_physical),
                azimuth_protocol_equivalence=dict(archived_physical_protocol_digest=document['physical_protocol_digest'],
                    archived_azimuth_orbit_schema=orbit_schema,base_and_radial_table_mapping_verified=mapping_verified,
                    rule='Only the known azimuth_orbit_schema metadata is excluded for cross-K physical comparison. All other physical protocol fields remain exact; legacy K2 requires original-to-final CDF mapping verification. Original protocol digests are retained unchanged.'),
                batch_drift_cdf_bound=document.get('batch_drift_cdf_bound'),root_summary=root_summary,
                root_diagnostics=document['root_diagnostics'],
                observed_batch_drift=document.get('batch_drift'),
                batch_anchor_margin_max_change_mm=max((b.get('anchor_margin_max_change_mm',0.) for b in document.get('batches',{}).values()),default=0.),
                source_stops=stop_summary(source_weight,source.get('gun_blocked_key'),source.get('gun_blocked_z_mm')),
                source_probability=float(source_weight.sum()),source_current_a=current,
                final_anchor_receipt=anchor_receipt,
                point_spread=psf,input=entry['input'],
                source_measure_method=source_method(document),
                energy_measure_receipt=energy_measure['receipt'] if energy_measure is not None else None,
                source_table_measure_receipt=table_measure_receipt)
            runs.append(ConditionalRun(label,m,order,document['base_seed'],bases,partition_digest,table_digest,
                document['physical_protocol_digest'],document['step_mm'],{case:v[0] for case,v in group.items()},provenance,root_bounds,uncertainty,k,expanded,
                anchor_coordinates,anchor_signs,anchor_margins,source_method(document),
                energy_measure['cuts']['edges'] if energy_measure is not None else None,
                energy_measure['measure']['base_probability'] if energy_measure is not None else None))
    return runs,dict(inputs=inputs,pending=pending)


def _validate_expanded_bases(bases,expanded,k):
    expected=np.repeat(bases,k//2,axis=0).copy()
    shifts=np.tile(np.arange(k//2),len(bases))/k
    if k!=2:expected[:,[1,2]]=(expected[:,[1,2]]+shifts[:,None])%1.
    if not np.array_equal(expanded,expected):
        raise ValueError('Expanded bases must rotate both source azimuths with the recorded orbit and base-major order')


def _verify_collection(runs):
    if not runs:
        raise ValueError('No complete four-case conditional source group is available')
    methods=[r.source_measure_method or source_method({}) for r in runs]
    for method in methods[1:]:require_same_refinement_method(methods[0],method)
    if len({run.label for run in runs})!=len(runs) or len({(r.base_count,r.azimuth_orbit_size,r.gauss_order) for r in runs})!=len(runs):
        raise ValueError('Conditional run labels and M/K/order groups must be unique')
    if len({(r.uncertainty.get('physical_protocol_comparison_digest',r.protocol_digest),r.base_seed,r.step_mm) for r in runs})!=1:
        raise ValueError('Conditional comparisons require one physical protocol, base seed and integration step')
    if len({r.uncertainty.get('transport_implementation_sha256') for r in runs})!=1:
        raise ValueError('Conditional comparisons require the same archived propagation implementation; postprocessing identity is separate')
    if len({r.protocol_digest for r in runs})!=1:
        for r in runs:
            receipt=r.uncertainty.get('azimuth_protocol_equivalence',{})
            schema=receipt.get('archived_azimuth_orbit_schema')
            if (receipt.get('archived_physical_protocol_digest')!=r.protocol_digest
                    or not receipt.get('base_and_radial_table_mapping_verified')
                    or schema not in (None,ORBIT_SCHEMA) or (schema is None and r.azimuth_orbit_size!=2)):
                raise ValueError('Cross-schema source comparison requires verified legacy K2 identity mapping')
    for r in runs:
        if (type(r.base_count) is not int or r.base_count<1 or type(r.gauss_order) is not int or r.gauss_order not in (2,4)
                or type(r.azimuth_orbit_size) is not int or r.azimuth_orbit_size not in (2,4,8,16)
                or r.base_points.shape!=(r.base_count,5) or not np.isfinite(r.base_points).all()
                or np.any((r.base_points<0)|(r.base_points>1))
                or set(r.root_probability_bounds)!=set(CASES[1:])
                or any(not np.isfinite(b) or b<0 for b in r.root_probability_bounds.values())):
            raise ValueError('Invalid conditional base coordinates, order or root uncertainty')
        if r.energy_cdf_edges is not None:
            if r.source_measure_method is None or r.source_measure_method['schema']!=ENERGY_SCHEMA:
                raise ValueError('Energy CDF partition requires its distinct source method identity')
            expected=expected_energy_orbit(r.base_points,r.energy_cdf_edges,r.azimuth_orbit_size)
            if (not np.array_equal(r.expanded_base_points,expected['points'])
                    or not np.array_equal(r.base_probability,expected['base_probability'])):
                raise ValueError('Energy/orbit reconstruction changed the original source measure')
        elif r.expanded_base_points is not None:
            if r.source_measure_method and r.source_measure_method['schema']==ENERGY_SCHEMA:
                raise ValueError('Energy-stratified refinement requires captured conditional CDF edges')
            _validate_expanded_bases(r.base_points,r.expanded_base_points,r.azimuth_orbit_size)
        elif r.azimuth_orbit_size!=2:
            raise ValueError('Orbit sampling requires expanded source bases')
        if set(r.clouds)!=set(CASES):
            raise ValueError('Every conditional run must retain exactly baseline, Cs, A1 and B2')
        base=r.clouds['baseline']
        if not np.isclose(base.weight.sum(),1.,atol=1e-12,rtol=0):
            raise ValueError('Conditional final source must retain full probability one before losses')
        for c in r.clouds.values():
            if not np.array_equal(c.source_id,base.source_id) or not np.array_equal(c.weight,base.weight):
                raise ValueError('Cases must share exact original source IDs and probabilities')
    by_m={}; by_mk={}
    for r in runs:
        original=by_m.setdefault(r.base_count,r)
        if not np.array_equal(r.base_points,original.base_points):
            raise ValueError('Azimuth refinement requires the same original five-dimensional bases')
        previous=by_mk.setdefault((r.base_count,r.azimuth_orbit_size),r)
        if (not np.array_equal(r.base_points,previous.base_points)
                or r.partition_digest!=previous.partition_digest):
            raise ValueError('Gauss-order refinement requires the same five-dimensional bases and solved partition')
    ordered=sorted(by_m.values(),key=lambda r:r.base_count)
    for a,b in zip(ordered,ordered[1:]):
        if not np.array_equal(a.base_points,b.base_points[:a.base_count]):
            raise ValueError('M refinement requires an exact prefix of the same five-dimensional base sequence')


def analyse_conditional(runs,pixels=(32,64,128,256,512)):
    _verify_collection(runs)
    groups=[]; moments={}; bounds={}; axes=[];anchor_comparisons=[]
    extent=(-2.4,2.4,-2.4,2.4)
    for r in runs:
        # One group at a time prevents variable final table sizes being mistaken
        # for an independent particle-count or integration-step axis.
        legacy=Run(r.label,len(r.clouds['baseline'].weight),r.step_mm,r.clouds,r.provenance)
        report=analyse([legacy],list(pixels),include_bilinear=False)
        base=r.clouds['baseline']; positive=base.weight>0
        counts={}
        for case,cloud in r.clouds.items():
            # The archived survivor mask excludes zero-weight diagnostics.
            # Their physical interception is available separately in stop keys.
            contact=(cloud.blocked_key=='flu_screen' if cloud.blocked_key is not None else cloud.survivor)
            counts[case]=dict(positive_weight_receiver_nodes=int(np.count_nonzero(positive&cloud.survivor)),
                positive_weight_common_nodes=int(np.count_nonzero(positive&base.survivor&cloud.survivor)),
                positive_weight_lost_nodes=int(np.count_nonzero(positive&base.survivor&~cloud.survivor)),
                positive_weight_gained_nodes=int(np.count_nonzero(positive&~base.survivor&cloud.survivor)),
                zero_weight_receiver_probe_nodes=int(np.count_nonzero(~positive&contact)),
                zero_weight_other_probe_nodes=int(np.count_nonzero(~positive&~contact)))
        groups.append(dict(label=r.label,base_count=r.base_count,gauss_order=r.gauss_order,
            source_measure_method=r.source_measure_method or source_method({}),
            azimuth_orbit_size=r.azimuth_orbit_size,expanded_base_count=r.expanded_base_count,
            final_ray_count=legacy.rays,positive_source_nodes=int(np.count_nonzero(r.clouds['baseline'].weight>0)),
            zero_weight_diagnostic_nodes=int(np.count_nonzero(r.clouds['baseline'].weight==0)),
            positive_weight_nodes=int(np.count_nonzero(positive)),zero_weight_probe_nodes=int(np.count_nonzero(~positive)),
            node_counts_by_case=counts,
            count_scope='Final tuple count includes zero-weight endpoint probes; M counts independent five-dimensional bases. Positive-weight counts and signed probabilities describe source transport. Only the five-dimensional bases, not the solved final six-dimensional table, share the M prefix.',
            source_table_digest=r.source_table_digest,
            partition_digest=r.partition_digest,root_probability_bounds=r.root_probability_bounds,
            uncertainty=r.uncertainty,physical_response=report))
        for case in CASES[1:]:
            moments[r.label,case]=translation_moments(r.clouds[case],base)
            bounds[r.label,case]=gaussian_remainder_bound(base,extent)+gaussian_remainder_bound(r.clouds[case],extent)
    axis_fields=dict(base_count=('base_count',('azimuth_orbit_size','gauss_order')),
        radial_gauss_order=('gauss_order',('base_count','azimuth_orbit_size')),
        azimuth_orbit_size=('azimuth_orbit_size',('base_count','gauss_order')))
    for axis,(varying,fixed_fields) in axis_fields.items():
        fixed=sorted({tuple(getattr(r,field) for field in fixed_fields) for r in runs})
        for value in fixed:
            selected=[r for r in runs if tuple(getattr(r,field) for field in fixed_fields)==value]
            selected.sort(key=lambda r:getattr(r,varying))
            for coarse,fine in zip(selected,selected[1:]):
                anchor_comparisons.append(compare_final_anchors(coarse,fine,axis))
                for n in pixels:
                    edges=np.linspace(-2.4,2.4,n+1)
                    for case in CASES[1:]:
                        a=gaussian_probability(moments[coarse.label,case],edges,edges)
                        b=gaussian_probability(moments[fine.label,case],edges,edges)
                        gaussian_bound=bounds[coarse.label,case]+bounds[fine.label,case]
                        root_bound=coarse.root_probability_bounds[case]+fine.root_probability_bounds[case]
                        scope=[]; callback_bound=0.; callback_fine_bound=0.
                        for run in (coarse,fine):
                            if run.uncertainty.get('topology_completeness')!='CERTIFIED':
                                scope.append(f'{run.label}: finite boundary scan does not exclude missed intervals')
                            drift=run.uncertainty.get('batch_drift_cdf_bound')
                            if drift is None:
                                scope.append(f'{run.label}: callback batch drift has no probability error bound')
                            elif not np.isfinite(drift) or drift<0:
                                raise ValueError('Callback probability error bound must be finite and nonnegative')
                            else:
                                callback_bound+=drift
                                if run is fine:callback_fine_bound=drift
                        metrics=compare_response(a,b,numerical_error_bound=gaussian_bound+root_bound+callback_bound)
                        fine_bound=bounds[fine.label,case]+fine.root_probability_bounds[case]+callback_fine_bound
                        error_upper=metrics['error_l1']+metrics['numerical_error_bound_l1']
                        signal_lower=max(0.,metrics['signal_l1']-fine_bound)
                        robust_status=(metrics['status'] if metrics['status']!='PASS_DECLARED_5_PERCENT' else
                            'PASS_DECLARED_5_PERCENT' if error_upper<=.05*signal_lower else
                            'UNRESOLVED_COMBINED_ERROR_BUDGET')
                        axes.append(dict(axis=axis,coarse_run=coarse.label,fine_run=fine.label,
                            coarse_base_count=coarse.base_count,fine_base_count=fine.base_count,
                            coarse_gauss_order=coarse.gauss_order,fine_gauss_order=fine.gauss_order,
                            coarse_azimuth_orbit_size=coarse.azimuth_orbit_size,fine_azimuth_orbit_size=fine.azimuth_orbit_size,
                            coarse_expanded_base_count=coarse.expanded_base_count,fine_expanded_base_count=fine.expanded_base_count,
                            pixels=n,case=case,method='fixed_physical_gaussian',
                            gaussian_truncation_bound_l1=gaussian_bound,
                            located_boundary_probability_bound_l1=root_bound,
                            bounded_callback_probability_error_l1=callback_bound if not any('callback' in s for s in scope) else None,
                            conservative_error_upper_l1=error_upper,
                            conservative_fine_signal_lower_l1=signal_lower,
                            conservative_relative_error_upper=error_upper/signal_lower if signal_lower>0 else None,
                            robust_status=robust_status,
                            unresolved_scope=scope,
                            qualification_status='UNRESOLVED_BOUNDARY_SCOPE' if scope else robust_status,
                            **metrics))
    return dict(schema='conditional-source-analysis-v1',tolerance=.05,groups=groups,axes=axes,
        anchor_comparisons=anchor_comparisons,
        axis_availability={axis:'EVALUATED' if any(row['axis']==axis for row in axes) else 'NOT_RUN_NO_INDEPENDENT_AXIS_PAIR'
                           for axis in axis_fields},
        qualification='CONDITIONAL_NUMERICAL_COMPARISONS_ONLY_NOT_TOPOLOGY_OR_INSTRUMENT_CERTIFICATION',
        limitations=['All differences retain original full-source probabilities, clipping, tilt, defocus and centroid changes.',
            'M refines deterministic other-coordinate quadrature; it is not an IID confidence interval.',
            'Energy stratification preserves original conditional CDF interval probabilities, including clipping atoms. M/K/G comparisons keep its method, fixed band, calibration and gun controls identical; uniform and stratified reports remain separate.',
            'K refines common source azimuth at fixed independent M and radial order. Both source azimuths rotate, and every ray is physically propagated; no output image is rotated or recentered. Expanded M*K/2 root bases are not independent Sobol bases.',
            'Discrete harmonic cancellation is exact for matched radial nodes. Different orientation-dependent partitions and finite radial quadrature can leave further imbalance; no real aberration response is forced to zero.',
            'The radial source table is shared across cases; Gauss2 to4 uses the same solved boundaries.',
            'Located-boundary uncertainty is included in the five-percent numerical gate; finite-scan missed intervals and transport callback drift are not bounded by root tolerance.',
            'The physical Gaussian response uses the captured provisional sigma0.6mm; raw histogram pixels are a separate observable.',
            'A single available M or Gauss order cannot demonstrate that axis has converged.'])


def compact_summary(report):
    """Small report of actual observables, preserving all error/scope gates."""
    cases=[]
    for group in report['groups']:
        response=group['physical_response'];pixels=max(response['pixels'])
        details=response['runs'][0]['cases']
        for case in CASES[1:]:
            detail=details[case]['survival_decomposition']
            image=next(row for row in response['survival_decomposition'] if row['case']==case and row['pixels']==pixels)
            raw=[row for row in response['axes'] if row['case']==case and row['method']=='hard_histogram' and row['axis']=='pixels']
            psf=[row for row in response['axes'] if row['case']==case and row['method']=='fixed_physical_gaussian' and row['axis']=='pixels']
            bound=group['root_probability_bounds'][case];signal=image['total_absolute_l1']
            flux=detail['signed_full_plane_probability_change']
            def pixel_record(rows):
                if not rows:return dict(status='NOT_RUN_NO_PIXEL_AXIS_PAIR')
                row=max(rows,key=lambda row:row['fine_pixels'])
                return dict(status=row['status'],coarse_pixels=row['coarse_pixels'],fine_pixels=row['fine_pixels'],
                    relative_error_percent=100*row['relative_error'] if row['relative_error'] is not None else None,
                    absolute_l1_error=row['error_l1'])
            cases.append(dict(group=group['label'],independent_base_count=group['base_count'],
                source_measure_method=group['source_measure_method'],
                azimuth_orbit_size=group['azimuth_orbit_size'],expanded_base_count=group['expanded_base_count'],
                gauss_order=group['gauss_order'],positive_weight_nodes=group['positive_weight_nodes'],
                zero_weight_probe_nodes=group['zero_weight_probe_nodes'],case=case,pixels=pixels,
                physical_psf_delta_l1=signal,signed_full_plane_probability_change=flux,
                baseline_probability=detail['baseline_probability'],case_probability=detail['case_probability'],
                baseline_centroid_mm=detail['baseline_centroid_mm'],case_centroid_mm=detail['case_centroid_mm'],
                actual_centroid_change_mm=detail['actual_centroid_change_mm'],
                common_displacement_l1=image['components']['common_displacement']['absolute_l1'],
                lost_signed_probability=detail['components']['lost_negative']['signed_full_plane_probability'],
                gained_signed_probability=detail['components']['gained_positive']['signed_full_plane_probability'],
                lost_l1=image['components']['lost_negative']['absolute_l1'],gained_l1=image['components']['gained_positive']['absolute_l1'],
                positive_node_counts=group['node_counts_by_case'][case],
                located_root_probability_bound=bound,
                located_bound_percent_of_delta_l1=100*bound/signal if signal>0 else None,
                located_bound_percent_of_abs_signed_flux=100*bound/abs(flux) if flux!=0 else None,
                raw_histogram_pixel_comparison=pixel_record(raw),physical_psf_pixel_comparison=pixel_record(psf),
                boundary_scope=group['uncertainty'].get('topology_completeness'),
                observed_anchor_cdf_estimate=(group['uncertainty'].get('observed_batch_drift') or {}).get('maximum_anchor_cdf_uncertainty_estimate')))
    finest=max((row['pixels'] for row in report['axes']),default=0)
    axes=[]
    for row in report['axes']:
        if row['pixels']!=finest:continue
        axes.append(dict(axis=row['axis'],coarse_run=row['coarse_run'],fine_run=row['fine_run'],case=row['case'],
            pixels=row['pixels'],fine_delta_l1=row['signal_l1'],absolute_l1_error=row['error_l1'],
            relative_error_percent=100*row['relative_error'] if row['relative_error'] is not None else None,
            located_root_probability_bound=row['located_boundary_probability_bound_l1'],
            conservative_relative_error_upper=row['conservative_relative_error_upper'],
            metric_status=row['status'],robust_status=row['robust_status'],qualification_status=row['qualification_status'],
            unresolved_scope=row['unresolved_scope']))
    return dict(schema='conditional-source-summary-v1',input_matrix_status=report.get('input_matrix_status'),
        qualification=report['qualification'],tolerance=report['tolerance'],axis_availability=report['axis_availability'],
        cases=cases,axis_comparisons=axes,observed_anchor_comparisons=report['anchor_comparisons'],
        provenance={key:value for key,value in report.items() if key.endswith('sha256') or key in ('inputs','analysis_source_identity')},
        scope=['Original source probabilities, actual centroids, tilt, defocus, and clipping are retained.',
            'M, common-azimuth K, radial Gauss order, and image pixels are distinct numerical axes.',
            'Located-root brackets bound only located boundaries; observed endpoint drift is not a certified interior error bound.',
            'A numerical relative-error pass alone is not overall physical qualification; inspect robust and qualification status.',
            'These explicit source tables are offline qualification evidence, not a changed default emission model.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pixels',type=int,nargs='+',default=[32,64,128,256,512])
    args=parser.parse_args()
    if (sorted(set(args.pixels))!=args.pixels or any(n<2 or n>512 for n in args.pixels)
            or any(b%a for a,b in zip(args.pixels,args.pixels[1:]))):
        parser.error('Use strictly increasing nested pixel grids from2 through512')
    implementation=postprocessing_provenance();analysis_identity=analysis_source_identity()
    runs,inputs=load_conditional_runs(args.input)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        report=analyse_conditional(runs,args.pixels)
    if implementation!=postprocessing_provenance():
        raise RuntimeError('Rendering implementation changed during offline conditional analysis')
    verify_analysis_source_identity(analysis_identity)
    report.update(inputs=inputs,analysis_script_sha256=analysis_identity[str(Path(__file__).resolve())],
        analysis_source_identity=analysis_identity,
        input_matrix_status='PARTIAL_AVAILABLE_GROUPS' if inputs['pending'] else 'AVAILABLE_GROUPS_ANALYSED_NOT_OVERALL_CONVERGENCE',
        **implementation)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    summary=compact_summary(report)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_table(args.output/'axis-comparisons.csv',report['axes'])
    write_table(args.output/'survival-decomposition.csv',[dict(group=group['label'],case=row['case'],pixels=row['pixels'],
        total_absolute_l1=row['total_absolute_l1'],cancelled_component_l1=row['cancelled_component_l1'],
        component=name,**component) for group in report['groups']
        for row in group['physical_response']['survival_decomposition'] for name,component in row['components'].items()])
    print(json.dumps(dict(groups=len(runs),comparisons=len(report['axes']),axis_availability=report['axis_availability'],
                         qualification=report['qualification']),indent=2))


if __name__=='__main__':main()
