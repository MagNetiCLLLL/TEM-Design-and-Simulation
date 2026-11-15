"""Closed prescribed-current transverse winding packs in vacuum.

Each physical winding pack is homogenized over radial and nested-loop
coordinates, preserving its full saved bounds and its specified ampere-turns.
Gauss-Legendre quadrature integrates the closed straight/arc circuit and the
pack. The retained Coulomb-gauge A and analytic Cartesian derivatives give B
by curl. Magnetic materials, lead routing, resistance and heating are absent.
No native Gaussian or top-hat envelope is imposed on the full axial tails.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from scipy.constants import e, mu_0

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.lens_field_provider import CoordinateRegistration, _posed_axial_support
from temsim.transverse_current import (
    CALIBRATION_KIND, MATRIX_FIELD, BIAS_FIELD, REFERENCE_ENERGY_FIELD,
    KIND_FIELD, STATUS_FIELD, NOTE_FIELD, validate_transverse_current_inputs,
)


SCHEMA='ideal-vacuum-transverse-coil-potential-v1'
DEFAULT_PACK_ORDER=12
DEFAULT_PATH_ORDER=24
_MAXIMUM_PAIR_VALUES=262144
MAXIMUM_QUADRATURE_NODES=2000000


@lru_cache(maxsize=1)
def _implementation_identity():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _readonly(value):
    a=np.ascontiguousarray(value,dtype=float)
    return np.frombuffer(a.tobytes(),dtype=float).reshape(a.shape)


def _quadrature(order, lower, upper):
    x,w=np.polynomial.legendre.leggauss(order)
    return (lower+upper)/2+(upper-lower)*x/2, w*(upper-lower)/2


def _rotation(angle):
    a=math.radians(angle)
    return np.array(((math.cos(a),-math.sin(a),0.),(math.sin(a),math.cos(a),0.),(0.,0.,1.)))


def _straight(start,end,order):
    t,w=_quadrature(order,0.,1.)
    delta=np.asarray(end)-start
    return np.asarray(start)+t[:,None]*delta, w[:,None]*delta


def _saddle(coil, radius, fraction, path_order):
    half=math.radians(coil['span_deg']/2)
    phi=coil['window_half_angle_rad']+fraction*(half-coil['window_half_angle_rad'])
    centre=coil['center_z_mm']*.001
    length=(coil['end_z_mm']-coil['start_z_mm'])*.001
    h=length/2-coil['band_mm']*.001*(1-fraction)
    a=math.radians(coil['azimuth_deg'])
    nodes=[]; weights=[]
    for angle,lo,hi in ((a+phi,-h,h),(a-phi,h,-h)):
        p,d=_straight(np.array([radius*math.cos(angle),radius*math.sin(angle),centre+lo]),
                      np.array([radius*math.cos(angle),radius*math.sin(angle),centre+hi]),path_order)
        nodes.append(p); weights.append(d)
    for z,lo,hi in ((centre+h,a+phi,a-phi),(centre-h,a-phi,a+phi)):
        angle,w=_quadrature(path_order,lo,hi)
        nodes.append(np.column_stack((radius*np.cos(angle),radius*np.sin(angle),np.full(len(angle),z))))
        weights.append(np.column_stack((-radius*np.sin(angle),radius*np.cos(angle),np.zeros(len(angle))))*w[:,None])
    return np.concatenate(nodes),np.concatenate(weights)


def _rectangle(coil, radial, fraction, path_order):
    # Independently sized straight-side bands share a normalized nesting
    # coordinate. Each complete loop traverses the full window-to-envelope
    # range, including unequal tangential and axial pack thicknesses.
    width=(coil['window_width_mm']+fraction*(coil['outer_width_mm']-coil['window_width_mm']))*.001
    height=(coil['window_height_mm']+fraction*(coil['outer_height_mm']-coil['window_height_mm']))*.001
    z=coil['center_z_mm']*.001
    vertices=np.array([[radial,-width/2,z-height/2],[radial,width/2,z-height/2],
        [radial,width/2,z+height/2],[radial,-width/2,z+height/2]])
    rotation=_rotation(coil['azimuth_deg'])
    vertices=vertices@rotation.T
    nodes=[];weights=[]
    for start,end in zip(vertices,np.roll(vertices,-1,axis=0)):
        p,d=_straight(start,end,path_order)
        nodes.append(p);weights.append(d)
    return np.concatenate(nodes),np.concatenate(weights)


@lru_cache(maxsize=16)
def _unit_nodes(coils_json,turns,pack_order,path_order):
    """One reusable unit-current basis per channel; N does not multiply work."""
    coils=json.loads(coils_json)
    requested=len(coils)*pack_order**2*4*path_order
    if requested>MAXIMUM_QUADRATURE_NODES:
        raise ValueError(f'Transverse closed-winding quadrature requests {requested} nodes; '
                         f'the {MAXIMUM_QUADRATURE_NODES}-node allocation budget was exceeded. '
                         'No source, turn or integration order was omitted')
    all_nodes=[];all_dl=[];groups=[]
    u,uw=_quadrature(pack_order,0.,1.)
    for coil in coils:
        if coil['family']=='saddle':
            low,high=coil['inner_radius_mm']*.001,coil['outer_radius_mm']*.001
            path=_saddle
        elif coil['family']=='radial_rectangle':
            low,high=coil['radial_start_mm']*.001,coil['radial_end_mm']*.001
            path=_rectangle
        else:
            raise ValueError('Unknown transverse prescribed-current circuit family')
        if not np.isfinite((low,high)).all() or not 0<low<high:
            raise ValueError('Transverse winding radial dimensions exceed the finite SI numerical range')
        radial,rw=_quadrature(pack_order,low,high)
        for r,wr in zip(radial,rw/(high-low)):
            for fraction,wf in zip(u,uw):
                nodes,dl=path(coil,r,fraction,path_order)
                all_nodes.append(nodes)
                all_dl.append(dl*(turns*coil['current_sign']*wr*wf*mu_0/(4*math.pi)))
                groups.append(np.full(len(nodes),0 if coil['group']=='x' else 1,dtype=int))
    group_array=np.concatenate(groups)
    frozen_groups=np.frombuffer(group_array.tobytes(),dtype=group_array.dtype)
    return _readonly(np.concatenate(all_nodes)),_readonly(np.concatenate(all_dl)),frozen_groups


def _positions(points):
    a=np.asarray(points)
    if a.dtype.kind not in 'iuf' or a.shape[-1:]!=(3,) or not np.isfinite(a).all():
        raise ValueError('Transverse coil positions must be finite real XYZ metres')
    return np.asarray(a,dtype=float)


def _inside_pack(points,coils):
    inside=np.zeros(points.shape[:-1],dtype=bool)
    for coil in coils:
        z=points[...,2]*1000-coil['center_z_mm']
        if coil['family']=='radial_rectangle':
            local=points@_rotation(coil['azimuth_deg'])*1000
            outer=(local[...,0]>=coil['radial_start_mm'])&(local[...,0]<=coil['radial_end_mm'])
            outer&=(abs(local[...,1])<=coil['outer_width_mm']/2)&(abs(z)<=coil['outer_height_mm']/2)
            hole=(abs(local[...,1])<coil['window_width_mm']/2)&(abs(z)<coil['window_height_mm']/2)
        else:
            r=np.hypot(points[...,0],points[...,1])*1000
            angle=np.angle((points[...,0]+1j*points[...,1])*np.exp(-1j*math.radians(coil['azimuth_deg'])))
            outer=(r>=coil['inner_radius_mm'])&(r<=coil['outer_radius_mm'])
            outer&=(abs(angle)<=math.radians(coil['span_deg']/2))&(points[...,2]*1000>=coil['start_z_mm'])&(points[...,2]*1000<=coil['end_z_mm'])
            hole=(abs(angle)<coil['window_half_angle_rad'])&(points[...,2]*1000>coil['start_z_mm']+coil['band_mm'])&(points[...,2]*1000<coil['end_z_mm']-coil['band_mm'])
        inside|=outer&~hole
    return inside


@dataclass(frozen=True,slots=True)
class TransverseCoilPotential:
    source_id: str
    coils: tuple
    turn_count_per_coil: int
    currents_a: tuple
    registration: CoordinateRegistration=field(default_factory=CoordinateRegistration)
    pack_order: int=DEFAULT_PACK_ORDER
    path_order: int=DEFAULT_PATH_ORDER
    provenance: Mapping=field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.source_id,str) or not self.source_id.strip() or self.source_id!=self.source_id.strip():
            raise ValueError('Transverse coil source requires a nonempty trimmed source identity')
        if isinstance(self.turn_count_per_coil,bool) or not isinstance(self.turn_count_per_coil,(int,np.integer)) or self.turn_count_per_coil<=0:
            raise ValueError('Transverse coil source requires a positive fixed integer turn count')
        if type(self.registration) is not CoordinateRegistration:
            raise ValueError('Transverse coil source requires its explicit rigid registration')
        for order in (self.pack_order,self.path_order):
            if isinstance(order,bool) or not isinstance(order,(int,np.integer)) or not 2<=order<=64:
                raise ValueError('Transverse winding quadrature orders must be integers from 2 to 64')
        if not isinstance(self.coils,(list,tuple)) or not self.coils or len(self.coils)>12:
            raise ValueError('A current station requires one to twelve explicit closed coil descriptors')
        coils=[];indices=set();stations=set();families=set()
        from numbers import Real,Integral
        for raw in self.coils:
            if not isinstance(raw,Mapping):
                raise ValueError('Each transverse closed coil must be a geometry mapping')
            required={'family','coil_index','station_index','group','current_sign','azimuth_deg','center_z_mm'}
            if not required.issubset(raw):
                raise ValueError('Transverse closed-coil geometry is incomplete')
            coil=thaw_json(raw)
            if coil['family'] not in ('saddle','radial_rectangle') or coil['group'] not in ('x','y') or isinstance(coil['current_sign'],bool) or coil['current_sign'] not in (-1,1):
                raise ValueError('Transverse source requires closed signed X/Y circuits')
            for name,minimum in (('coil_index',1),('station_index',0)):
                if isinstance(coil[name],bool) or not isinstance(coil[name],Integral) or coil[name]<minimum:
                    raise ValueError('Transverse coil/station indices require positive/nonnegative integers')
            if coil['coil_index'] in indices:
                raise ValueError('A physical coil descriptor must not be counted twice')
            indices.add(coil['coil_index']);stations.add(coil['station_index']);families.add(coil['family'])
            dimensions=(('start_z_mm','end_z_mm','inner_radius_mm','outer_radius_mm','span_deg','band_mm','window_half_angle_rad')
                        if coil['family']=='saddle' else ('radial_start_mm','radial_end_mm','outer_width_mm',
                        'outer_height_mm','window_width_mm','window_height_mm'))
            if not set(dimensions).issubset(coil):
                raise ValueError('Transverse winding dimensions are incomplete')
            for name in ('azimuth_deg','center_z_mm',*dimensions):
                if isinstance(coil[name],(bool,np.bool_)) or not isinstance(coil[name],Real) or not math.isfinite(coil[name]):
                    raise ValueError('Transverse winding dimensions must be finite real scalars')
                coil[name]=float(coil[name])
            if coil['family']=='saddle':
                valid=(0<coil['inner_radius_mm']<coil['outer_radius_mm'] and coil['end_z_mm']>coil['start_z_mm']
                    and 0<coil['span_deg']<180 and 0<coil['band_mm']<(coil['end_z_mm']-coil['start_z_mm'])/2
                    and 0<coil['window_half_angle_rad']<math.radians(coil['span_deg']/2))
                centre=(coil['start_z_mm']+coil['end_z_mm'])/2
                valid=valid and abs(coil['center_z_mm']-centre)<=64*math.ulp(max(abs(centre),1.))
            else:
                valid=(0<coil['radial_start_mm']<coil['radial_end_mm']
                    and 0<coil['window_width_mm']<coil['outer_width_mm']
                    and 0<coil['window_height_mm']<coil['outer_height_mm'])
            if not valid:
                raise ValueError('Transverse winding requires positive pack thickness and an open central window')
            coils.append(freeze_json(coil))
        coils=tuple(coils)
        if len(stations)!=1 or len(families)!=1:
            raise ValueError('One transverse potential owns exactly one physical station')
        if len(coils)>8 and families!={'radial_rectangle'}:
            raise ValueError('More than eight coils require an explicit radial multipole winding family')
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in self.currents_a):
            raise ValueError('Transverse station currents must be finite real scalars in amperes')
        currents=tuple(float(v) for v in self.currents_a)
        if len(currents)!=2 or not np.isfinite(currents).all():
            raise ValueError('Transverse station requires two finite signed currents in amperes')
        try:
            with np.errstate(over='ignore',invalid='ignore'):
                finite_ni=np.isfinite(np.asarray(currents)*float(self.turn_count_per_coil)).all()
        except (OverflowError,ValueError):finite_ni=False
        if not finite_ni:
            raise ValueError('Transverse winding ampere-turns exceed the finite numerical range')
        object.__setattr__(self,'coils',coils)
        object.__setattr__(self,'currents_a',currents)
        object.__setattr__(self,'provenance',freeze_json(thaw_json(self.provenance)))
        json_digest(self._payload())

    @property
    def digest(self):
        return json_digest(self._payload())

    def _payload(self):
        return dict(schema=SCHEMA,source_id=self.source_id,coils=thaw_json(self.coils),
            turn_count_per_coil=int(self.turn_count_per_coil),currents_a=list(self.currents_a),
            registration=asdict(self.registration),pack_order=int(self.pack_order),path_order=int(self.path_order),
            provenance=thaw_json(self.provenance),mu_0_h_per_m=mu_0,implementation_sha256=_implementation_identity())

    def to_metadata(self):
        return {**self._payload(), 'units':dict(position='m',current='A',A='T m',B='T'),
            'gauge':'Coulomb line-current potential; closed-loop pack quadrature',
            'axial_domain':'all_z','outside_domain':'reject winding-pack material; no axial cutoff',
            'source_model':'uniform radial and normalized nested-loop pack coordinates; preserves full allocated bounds and NI',
            'magnetic_materials':False,'wire_turn_resolved_field':False,'wave_admission':False}

    def to_dict(self):
        return dict(payload=self._payload(),digest=self.digest)

    @classmethod
    def from_dict(cls,record):
        if not isinstance(record,Mapping) or set(record)!= {'payload','digest'}:
            raise ValueError('Invalid transverse coil potential record')
        data=dict(record['payload'])
        if data.pop('schema',None)!=SCHEMA or data.pop('mu_0_h_per_m',None)!=mu_0 or data.pop('implementation_sha256',None)!=_implementation_identity():
            raise ValueError('Transverse coil source implementation or constant changed')
        data['registration']=CoordinateRegistration(**data['registration'])
        result=cls(**data)
        if result.digest!=record['digest']:
            raise ValueError('Transverse coil potential identity mismatch')
        return result

    def __deepcopy__(self,memo):
        memo[id(self)]=self
        return self

    @property
    def native_support_mm(self):
        return (min(c['start_z_mm'] if c['family']=='saddle' else c['center_z_mm']-c['outer_height_mm']/2 for c in self.coils),
                max(c['end_z_mm'] if c['family']=='saddle' else c['center_z_mm']+c['outer_height_mm']/2 for c in self.coils))

    @property
    def radial_support_m(self):
        return min(c['inner_radius_mm'] if c['family']=='saddle' else c['radial_start_mm'] for c in self.coils)*.001

    def _sample(self,positions_m,*,derivatives):
        points=_positions(positions_m)
        local=self.registration.positions_global_to_local_m(points)
        if np.any(_inside_pack(local,self.coils)):
            raise ValueError('Transverse ideal-current source excludes its occupied winding pack from vacuum queries')
        flat=local.reshape(-1,3)
        arrays=[np.zeros((len(flat),3))]
        if derivatives:
            arrays.extend((np.zeros((len(flat),3,3)),np.zeros((len(flat),3,3,3))))
        else:
            arrays.append(np.zeros((len(flat),3)))
        if any(self.currents_a) and len(flat):
            coils_json=json.dumps(thaw_json(self.coils),sort_keys=True,separators=(',',':'),allow_nan=False)
            nodes,dl,groups=_unit_nodes(coils_json,int(self.turn_count_per_coil),int(self.pack_order),int(self.path_order))
            weighted=dl*np.asarray(self.currents_a)[groups,None]
            active=np.any(weighted!=0.,axis=1)
            nodes,weighted=nodes[active],weighted[active]
            block=max(1,_MAXIMUM_PAIR_VALUES//max(1,len(nodes)))
            for start in range(0,len(flat),block):
                stop=min(len(flat),start+block)
                delta=flat[start:stop,None,:]-nodes[None,:,:]
                norm2=np.einsum('pni,pni->pn',delta,delta)
                if not np.isfinite(norm2).all() or np.any(norm2<=0.):
                    raise ValueError('Transverse winding quadrature singularity is outside vacuum')
                inv=1/np.sqrt(norm2)
                inv3=inv/norm2
                arrays[0][start:stop]=np.einsum('ni,pn->pi',weighted,inv)
                if derivatives:
                    arrays[1][start:stop]=-np.einsum('ni,pnj,pn->pij',weighted,delta,inv3)
                    arrays[2][start:stop]=3*np.einsum('ni,pnj,pnk,pn->pijk',weighted,delta,delta,inv3/norm2)
                    arrays[2][start:stop]-=np.einsum('ni,jk,pn->pijk',weighted,np.eye(3),inv3)
                else:
                    arrays[1][start:stop]=np.sum(np.cross(weighted[None,:,:],delta)*inv3[:,:,None],axis=1)
        rotation=self.registration.rotation_array
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError('Transverse winding field exceeds the finite numerical range')
        arrays[0]=arrays[0]@rotation.T
        if derivatives:
            arrays[1]=np.einsum('ai,pij,bj->pab',rotation,arrays[1],rotation)
            arrays[2]=np.einsum('ai,pijk,bj,ck->pabc',rotation,arrays[2],rotation,rotation)
            return dict(A_t_m=arrays[0].reshape(points.shape),jac_A_t=arrays[1].reshape((*points.shape[:-1],3,3)),
                        hess_A_t_per_m=arrays[2].reshape((*points.shape[:-1],3,3,3)))
        arrays[1]=arrays[1]@rotation.T
        return dict(A_t_m=arrays[0].reshape(points.shape),B_t=arrays[1].reshape(points.shape))

    def sample(self,positions_m):
        return self._sample(positions_m,derivatives=False)

    def sample_jet(self,positions_m):
        return self._sample(positions_m,derivatives=True)

    def field_at_global_positions_t(self,positions_m):
        return self.sample(positions_m)['B_t']


@dataclass(frozen=True,slots=True)
class FrozenTransverseCoilField:
    potential: TransverseCoilPotential
    component_key: str
    reference_momentum: float
    event_z_mm: float | None=None
    event_dx_rad: float=0.
    event_dy_rad: float=0.
    captured_time_s: float | None=None
    drive_keys: tuple=()
    dynamic: bool=False

    def __post_init__(self):
        if type(self.potential) is not TransverseCoilPotential or not isinstance(self.component_key,str) or not self.component_key.strip():
            raise ValueError('Frozen transverse fields require a captured winding potential and component identity')
        values=(self.reference_momentum,self.event_dx_rad,self.event_dy_rad)
        if not np.isfinite(values).all() or self.reference_momentum<=0:
            raise ValueError('Frozen transverse event metadata must be finite with positive reference momentum')
        if self.event_z_mm is not None and not math.isfinite(self.event_z_mm):
            raise ValueError('Frozen transverse event plane must be finite')
        if self.captured_time_s is not None and not math.isfinite(self.captured_time_s):
            raise ValueError('Frozen transverse capture time must be finite')
        if type(self.dynamic) is not bool or any(not isinstance(k,str) or not k for k in self.drive_keys):
            raise ValueError('Frozen transverse drives require explicit identities and Boolean dynamic state')
        object.__setattr__(self,'drive_keys',tuple(self.drive_keys))

    @property
    def lens_key(self): return self.potential.source_id
    @property
    def key(self): return self.lens_key
    @property
    def registration(self): return self.potential.registration
    @property
    def axial_domain_all_z(self): return True
    @property
    def fingerprint(self): return self.potential.digest
    @property
    def radial_support_m(self): return self.potential.radial_support_m
    @property
    def native_support_mm(self): return self.potential.native_support_mm
    @property
    def scale(self): return float(any(self.potential.currents_a))
    @property
    def field_support_mm(self):
        return _posed_axial_support(self.native_support_mm,self.registration,self.radial_support_m)
    @property
    def lower_m(self): return self.native_support_mm[0]*.001
    @property
    def upper_m(self): return self.native_support_mm[1]*.001
    @property
    def arrival_z_mm(self):
        z=self.event_z_mm if self.event_z_mm is not None else sum(self.native_support_mm)/2
        return (self.registration.rotation_array@np.array([0.,0.,z*.001])+self.registration.origin_array_m)[2]*1000
    @property
    def bounds_m(self):
        from itertools import product
        lo,hi=np.asarray(self.native_support_mm)*.001
        r=self.radial_support_m
        corners=np.array(tuple(product((-r,r),(-r,r),(lo,hi))))
        placed=corners@self.registration.rotation_array.T+self.registration.origin_array_m
        return np.array((placed.min(axis=0),placed.max(axis=0)))
    @property
    def display_bounds_m(self): return self.bounds_m
    def display_contains(self,positions):
        p=self.registration.positions_global_to_local_m(positions)
        return np.hypot(p[...,0],p[...,1])<=self.radial_support_m
    def field_at_global_positions_t(self,positions): return self.potential.field_at_global_positions_t(positions)
    @property
    def bx_t(self):
        z=self.event_z_mm if self.event_z_mm is not None else sum(self.native_support_mm)/2
        point=np.array([[0.,0.,z*.001]])@self.registration.rotation_array.T+self.registration.origin_array_m
        return float(self.potential.field_at_global_positions_t(point)[0,0])
    @property
    def by_t(self):
        z=self.event_z_mm if self.event_z_mm is not None else sum(self.native_support_mm)/2
        point=np.array([[0.,0.,z*.001]])@self.registration.rotation_array.T+self.registration.origin_array_m
        return float(self.potential.field_at_global_positions_t(point)[0,1])
    def maximum_step_mm(self,requested_step_mm,momentum_kg_m_s):
        if not np.isfinite((requested_step_mm,momentum_kg_m_s)).all() or min(requested_step_mm,momentum_kg_m_s)<=0:
            raise ValueError('Transverse current source needs positive step and momentum')
        return min(float(requested_step_mm),self.radial_support_m*100.)

    def implementation_is_supported(self):
        return all(getattr(TransverseCoilPotential,name,None) is original for name,original in _POTENTIAL_METHODS.items())


def _momentum(energy_ev):
    # Match the project's existing constants and arithmetic order exactly;
    # a newer SciPy electron mass must not change a saved field-strength fit.
    from types import SimpleNamespace
    from temsim.physics.core import electron
    return electron(SimpleNamespace(beam_voltage_kv=float(energy_ev)/1000.))[1]


def _components(state):
    result={c.key:c for name in ('stigmators','corrector_elements','deflectors') for c in getattr(state,name,())}
    gun=getattr(state,'electron_gun',None)
    for name in ('deflector','stigmator'):
        component=getattr(gun,name,None)
        if component is not None:
            result[component.key]=component
    return result


def _bindings(state, *, require_current,component_keys=None):
    from temsim.transverse_winding import transverse_winding_definition, transverse_winding_owner_key
    components=_components(state)
    rows=[];seen=set()
    for record in getattr(getattr(state,'_resolved_assembly',None),'parts',()):
        part=dict(record.data)
        owner=transverse_winding_owner_key(part)
        if component_keys is not None and owner not in component_keys:
            continue
        definition=transverse_winding_definition(part)
        if definition is None:
            validate_transverse_current_inputs(part)
            continue
        component=components.get(owner)
        if component is None:
            raise ValueError(f'{record.key}: transverse winding has no installed native optical owner {owner}')
        if owner in seen:
            raise ValueError(f'{owner}: more than one material winding source claims this optical owner')
        electrical=validate_transverse_current_inputs(part)
        if require_current and electrical is None:
            continue # Historical N-only metadata retains its native field path.
        seen.add(owner)
        offset=float(record.center_z_mm)-float(part['local_center_z_mm'])
        gun=any(component is getattr(getattr(state,'electron_gun',None),name,None) for name in ('deflector','stigmator'))
        if gun:
            offset+=float(component.field_center_offset_mm)
        coils=[]
        for source in definition['coils']:
            coil=dict(source)
            for name in ('center_z_mm','start_z_mm','end_z_mm'):
                if name in coil:coil[name]+=offset
            coils.append(coil)
        rows.append((part,component,definition,tuple(coils),electrical,gun))
    return tuple(rows)


def _command(component, drives):
    if hasattr(component,'upper_field_x_mt'):
        if component.beam_blanked:
            return np.array([0.,component.blanking_field_y_mt,0.,0.]),(),(),False,None
        return np.array([component.upper_field_x_mt,component.upper_field_y_mt,
                         component.lower_field_x_mt,component.lower_field_y_mt]),(),(),False,None
    if hasattr(component,'gradient_t_per_m'):
        angle=math.radians(2*component.rotation_deg)
        return np.array([component.gradient_t_per_m*math.cos(angle),component.gradient_t_per_m*math.sin(angle)]),(),(),False,None
    if hasattr(component,'quadrupole_tensor_m2'):
        return np.array([component.strength_x_percent,component.strength_y_percent]),(),(),False,None
    row=drives.get(component.key)
    if row is None:
        return np.zeros(0),(),(),False,None
    _,events,keys,dynamic,time=row
    return np.array([(dx*1000,dy*1000) for _,dx,dy in events]).reshape(-1),events,keys,dynamic,time


def capture_transverse_coil_fields(state, *, time_s=None, component_keys=None, include_gun=False,
                                  include_hexapole=True):
    """Freeze physical currents from saved electrical gains, never from energy.

    The existing optical commands remain readable. Shared logical deflector
    commands are combined on their physical host before one matrix is applied.
    Selecting a different beam voltage does not change these coil currents.
    """
    if getattr(state,'round_lens_field_model','gaussian')!='copper_coil':
        return ()
    from temsim.physics.instrument_magnetic import column_deflector_drives
    from temsim.lens_pose import lens_pose_registration
    drives={row[0].key:row for row in column_deflector_drives(state,time_s=time_s)}
    result=[]
    for part,component,definition,coils,electrical,gun in _bindings(state,require_current=True,component_keys=component_keys):
        if gun and not include_gun:
            continue
        if component_keys is not None and component.key not in component_keys:
            continue
        if not bool(getattr(component,'enabled',False)) and not bool(getattr(component,'beam_blanked',False)):
            continue
        command,events,drive_keys,dynamic,captured_time=_command(component,drives)
        matrix=np.asarray(electrical['matrix_a_per_unit'])
        if command.shape!=(len(matrix),) or not np.isfinite(command).all():
            raise ValueError(f'{component.key}: electrical calibration does not match its native command channels')
        currents=matrix@command+electrical['current_bias_a']
        registration=lens_pose_registration(state,component.key)
        stations=sorted({c['station_index'] for c in coils})
        is_dipole=coils[0]['family']=='saddle'
        reference_momentum=_momentum(electrical['reference_energy_ev'])
        for station in stations:
            station_coils=tuple(c for c in coils if c['station_index']==station)
            key=f'{component.key}:{station}' if is_dipole else component.key
            potential=TransverseCoilPotential(key,station_coils,definition['turn_count_per_coil'],
                tuple(currents[station*2:station*2+2]),registration,
                provenance=dict(material_host=part['key'],component_key=component.key,
                    electrical=electrical,command_unit=('mT' if gun and is_dipole else
                    'normal_skew_T_per_m' if gun else 'mrad' if is_dipole else 'percent'),
                    source_note=definition['source_note'],homogenized_pack=True,
                    station_index=station,drive_keys=list(drive_keys),captured_time_s=captured_time,
                    dynamic_drive_frozen=bool(dynamic),
                    clear_bore_radius_m=float(getattr(component,'mechanical_clear_bore_diameter_mm',
                        getattr(component,'bore_diameter_mm',definition['coils'][0].get('inner_radius_mm',1.)*2)))*.0005))
            event=(events[station] if events else (None,0.,0.))
            result.append(FrozenTransverseCoilField(potential,component.key,reference_momentum,
                event[0],event[1],event[2],captured_time,tuple(drive_keys),bool(dynamic)))
    from temsim.physics.multipole_coil_field import capture_multipole_coil_fields
    result.extend(capture_multipole_coil_fields(state,component_keys=component_keys,
                                               include_hexapole=include_hexapole))
    return tuple(result)


def quadrupole_gradient_t_per_m(potential,points_m):
    """Project the actual field Jacobian onto normal/skew near-axis gradients."""
    h=potential.sample_jet(points_m)['hess_A_t_per_m']
    jac_b=np.stack((h[...,2,1,:]-h[...,1,2,:],h[...,0,2,:]-h[...,2,0,:],
                    h[...,1,0,:]-h[...,0,1,:]),axis=-2)
    return np.stack(((jac_b[...,1,0]+jac_b[...,0,1])/2,
                     (jac_b[...,1,1]-jac_b[...,0,0])/2),axis=-1)


def calibrate_transverse_current_matrix(state,host_key, *, pack_order=DEFAULT_PACK_ORDER,path_order=DEFAULT_PATH_ORDER):
    """Detached full-channel peak-field fit, preserving geometry/N/commands.

    Multiple deflector stations are calibrated jointly, including the physical
    cross-station tails. This is a field-strength match at the old reference
    planes, not an integral, field-shape, focus or current-rating equivalence.
    """
    from temsim.lens_pose import lens_pose_registration
    matches=[row for row in _bindings(state,require_current=False) if row[0]['key']==host_key]
    if len(matches)!=1:
        raise ValueError(f'{host_key}: select one installed material transverse winding host')
    part,component,definition,coils,_,gun=matches[0]
    registration=lens_pose_registration(state,component.key)
    if any(registration.origin_global_m) or not np.array_equal(registration.rotation_array,np.eye(3)):
        raise ValueError('Initial transverse current calibration requires aligned nominal winding axes')
    stations=sorted({c['station_index'] for c in coils})
    is_dipole=coils[0]['family']=='saddle'
    if gun and is_dipole:
        planes=np.array([component.upper_center_from_tip_mm,component.lower_center_from_tip_mm])+component.field_center_offset_mm
        unit='mT'
    elif gun:
        planes=np.array([component.optical_reference_from_tip_mm]);unit='normal_skew_T_per_m'
    elif is_dipole:
        # Calibrate installed but disabled hardware as an explicit unit-current
        # reference; do not enable it or overwrite its stored command values.
        planes=(np.array([component.upper_z_mm,component.lower_z_mm])
                if hasattr(component,'upper_z_mm') else np.array([component.z_mm]))
        unit='mrad'
    else:
        planes=np.array([component.z_mm]);unit='percent'
    if len(stations)!=len(planes):
        raise ValueError('Native reference-plane count differs from the physical winding stations')
    points=np.column_stack((np.zeros(len(planes)),np.zeros(len(planes)),planes*.001))
    size=2*len(stations)
    response=np.zeros((size,size))
    for station in stations:
        station_coils=tuple(c for c in coils if c['station_index']==station)
        for group in range(2):
            currents=(1.,0.) if group==0 else (0.,1.)
            potential=TransverseCoilPotential('unit_current',station_coils,definition['turn_count_per_coil'],currents,
                registration,pack_order,path_order)
            values=(potential.field_at_global_positions_t(points)[...,:2] if is_dipole else
                    quadrupole_gradient_t_per_m(potential,points))
            response[:,2*station+group]=values.reshape(-1)
    energy=float(state.beam_voltage_kv)*1000
    from temsim.physics.core import electron
    charge,momentum,_=electron(state)
    ratio=momentum/charge
    target=np.zeros_like(response)
    if gun and is_dipole:
        from temsim.optics.electron_gun.electrostatic import _soft_window_with_derivatives
        for station,centre in enumerate(planes):
            envelope=_soft_window_with_derivatives(planes,centre-component.coil_length_mm/2,
                centre+component.coil_length_mm/2,component.soft_edge_mm)[0]
            for plane,factor in enumerate(envelope):
                target[2*plane:2*plane+2,2*station:2*station+2]=np.eye(2)*.001*factor
    elif gun:
        target=np.eye(2)
    elif is_dipole:
        length=float(getattr(component,'effective_thickness_mm',getattr(component,'thickness_mm',0.)))*.001
        if length<=0:raise ValueError('Native deflector calibration requires positive effective thickness')
        for station in stations:
            target[2*station:2*station+2,2*station:2*station+2]=np.array([[0.,ratio*.001/length],[-ratio*.001/length,0.]])
    else:
        angles=np.deg2rad(2*np.array([component.channel_x_angle_deg,component.channel_y_angle_deg]))
        target=float(component.max_strength_m2)/100*ratio*np.stack((np.cos(angles),np.sin(angles)))
    condition=float(np.linalg.cond(response))
    if not math.isfinite(condition) or condition>1e8:
        raise ValueError('Transverse unit-current calibration is singular or poorly conditioned')
    matrix=np.linalg.solve(response,target)
    np.testing.assert_allclose(response@matrix,target,rtol=1e-12,atol=1e-14)
    note=(f'Ideal vacuum closed-coil current calibration at original native field-reference planes; '
          f'commands in {unit}, matrix in A/{unit}, reference energy {energy:.12g} eV. '
          'Full cross-station field response is included. Fixed N and pack envelopes retained. '
          'No magnetic-core/pole response, field-integral/shape equivalence, focus adjustment, '
          'thermal rating or experimental calibration is inferred. Current matrix remains fixed '
          'after beam-energy or geometry edits; re-calibration is explicit.')
    fields={MATRIX_FIELD:matrix.tolist(),BIAS_FIELD:[0.]*size,REFERENCE_ENERGY_FIELD:energy,
        KIND_FIELD:CALIBRATION_KIND,STATUS_FIELD:'calibrated_native_peak',NOTE_FIELD:note}
    return dict(host_key=part['key'],component_key=component.key,command_unit=unit,
        reference_planes_z_mm=planes.tolist(),unit_current_response=response.tolist(),
        native_target_per_unit=target.tolist(),matrix_a_per_unit=matrix.tolist(),
        response_condition_number=condition,maximum_abs_match_error=float(np.max(abs(response@matrix-target))),
        turn_count_per_coil=definition['turn_count_per_coil'],pack_order=pack_order,path_order=path_order,
        fields=fields,source_model='homogenized_closed_pack_in_vacuum',native_controls_unchanged=True,
        focus_or_source_calculation_executed=False)


_POTENTIAL_METHODS={name:getattr(TransverseCoilPotential,name) for name in
    ('sample','sample_jet','field_at_global_positions_t','_sample','to_dict','to_metadata')}
