"""Time-dependent ideal winding currents with their retained induced E.

The existing native raster/sine commands remain in mrad. A separately frozen
electrical gain converts those commands to coil amperes, including every
station's contribution to a physical current. No magnetic field is inferred
from electron energy, and no voltage, inductance or radiation solve is added.
"""
from collections.abc import Mapping
from dataclasses import dataclass,field,replace
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np

from temsim.immutable_json import json_digest
from temsim.physics.transverse_coil_field import TransverseCoilPotential,_unit_nodes,_positions


SCHEMA='ideal-current-dynamic-transverse-coil-v1'


@lru_cache(maxsize=1)
def _implementation_identity():
    return sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True,slots=True)
class DynamicTransverseCoilPotential:
    base: TransverseCoilPotential
    owner_ids: tuple
    drives: tuple
    current_gain_matrices_a_per_mrad: tuple
    reference_time_s: float
    digest: str=field(init=False)

    def __post_init__(self):
        from temsim.physics.dynamic_potential import NativeTransverseDrive,_number,_identifier
        if type(self.base) is not TransverseCoilPotential:
            raise ValueError('Dynamic prescribed currents require a retained closed winding potential')
        owners=tuple(_identifier(value,'Physical current source owner') for value in self.owner_ids)
        drives=tuple(self.drives)
        raw_gains=np.asarray(self.current_gain_matrices_a_per_mrad)
        if raw_gains.dtype.kind not in 'iuf':
            raise ValueError('Dynamic current gains must be finite real numbers')
        gains=np.asarray(raw_gains,dtype=float)
        if not owners or len(set(owners))!=len(owners) or not drives or any(type(d) is not NativeTransverseDrive for d in drives):
            raise ValueError('Dynamic prescribed currents require unique owners and exact native command laws')
        if len({d.key for d in drives})!=len(drives) or gains.shape!=(len(drives),2,2) or not np.isfinite(gains).all():
            raise ValueError('Each native waveform requires one finite signed current gain matrix')
        object.__setattr__(self,'base',TransverseCoilPotential.from_dict(self.base.to_dict()))
        object.__setattr__(self,'owner_ids',owners)
        object.__setattr__(self,'drives',drives)
        object.__setattr__(self,'current_gain_matrices_a_per_mrad',tuple(tuple(tuple(float(v) for v in row) for row in gain) for gain in gains))
        object.__setattr__(self,'reference_time_s',_number(self.reference_time_s,'Current reference time'))
        object.__setattr__(self,'digest',json_digest(self._payload()))

    @property
    def source_id(self): return self.base.source_id
    @property
    def potential(self): return self
    @property
    def kind(self): return 'magnetic'

    def coefficients(self,time_s):
        from temsim.physics.dynamic_potential import _times
        times=_times(time_s)
        current=np.broadcast_to(self.base.currents_a,(*times.shape,2)).copy()
        first=np.zeros_like(current);second=np.zeros_like(current)
        for drive,gain in zip(self.drives,self.current_gain_matrices_a_per_mrad):
            gain=np.asarray(gain)
            value,dot,ddot=drive.evaluate(times)
            current+=(value-drive.evaluate(self.reference_time_s)[0])@gain.T
            first+=dot@gain.T;second+=ddot@gain.T
        if any(not np.isfinite(v).all() for v in (current,first,second)):
            raise ValueError('Dynamic winding currents exceed the finite numerical range')
        return current,first,second

    def sample(self,positions_m,time_s,*,derivatives=False,_axial_side=None):
        from temsim.physics.dynamic_potential import _times
        if _axial_side is not None:
            raise ValueError('Closed winding fields have smooth full axial tails, not compact axial sides')
        positions=_positions(positions_m)
        times=np.broadcast_to(_times(time_s),positions.shape[:-1])
        for drive in self.drives:
            drive.require_differentiable_time(times)
        currents,first,second=self.coefficients(times)
        result={key:np.zeros(positions.shape) for key in
            ('A_t_m','B_t','E_v_per_m','dA_dt_t_m_per_s','d2A_dt2_t_m_per_s2')}
        if derivatives:
            result.update(jac_A_t=np.zeros((*positions.shape,3)),
                          hess_A_t_per_m=np.zeros((*positions.shape,3,3)),
                          jac_dA_dt_t_per_s=np.zeros((*positions.shape,3)))
        for channel in range(2):
            basis=replace(self.base,currents_a=(1.,0.) if channel==0 else (0.,1.))
            sample=basis.sample(positions)
            result['A_t_m']+=sample['A_t_m']*currents[...,channel,None]
            result['B_t']+=sample['B_t']*currents[...,channel,None]
            result['dA_dt_t_m_per_s']+=sample['A_t_m']*first[...,channel,None]
            result['d2A_dt2_t_m_per_s2']+=sample['A_t_m']*second[...,channel,None]
            if derivatives:
                jet=basis.sample_jet(positions)
                result['jac_A_t']+=jet['jac_A_t']*currents[...,channel,None,None]
                result['hess_A_t_per_m']+=jet['hess_A_t_per_m']*currents[...,channel,None,None,None]
                result['jac_dA_dt_t_per_s']+=jet['jac_A_t']*first[...,channel,None,None]
        result['E_v_per_m']=-result['dA_dt_t_m_per_s']
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError('Dynamic prescribed-current fields exceed the finite numerical range')
        return result

    def sample_jet(self,positions_m,time_s,*,axial_side=None):
        return self.sample(positions_m,time_s,derivatives=True,_axial_side=axial_side)

    def require_time_interval(self,start_s,stop_s):
        for drive in self.drives:drive.require_time_interval(start_s,stop_s)

    def next_time_event(self,time_s):
        return min(d.next_time_event(time_s) for d in self.drives)

    def temporal_bounds(self):
        import json
        from temsim.immutable_json import thaw_json
        radius=float(self.base.provenance.get('clear_bore_radius_m',.9*self.base.radial_support_m))
        margin=self.base.radial_support_m-radius
        if margin<=0:
            raise ValueError('Temporal current bounds require positive clearance from the winding pack')
        nodes,dl,groups=_unit_nodes(json.dumps(thaw_json(self.base.coils),sort_keys=True,separators=(',',':')),
            self.base.turn_count_per_coil,self.base.pack_order,self.base.path_order)
        lengths=np.array([np.sum(np.linalg.norm(dl[groups==i],axis=1)) for i in range(2)])
        constant=np.array(self.base.currents_a)
        for drive,gain in zip(self.drives,self.current_gain_matrices_a_per_mrad):
            constant-=np.asarray(gain)@drive.evaluate(self.reference_time_s)[0]
        rows=[]
        for order in range(4):
            bound=abs(constant) if order==0 else np.zeros(2)
            for drive,gain in zip(self.drives,self.current_gain_matrices_a_per_mrad):
                bound+=abs(np.asarray(gain))@drive.derivative_bound(order)
            weighted=float(lengths@bound)
            rows.append(dict(order=order,coefficient_abs_bound=bound.tolist(),
                coefficient_unit=f'A/s^{order}',A_norm_bound=weighted/margin,
                jac_A_norm_bound=weighted/margin**2))
        return dict(source_id=self.source_id,local_radius_m=radius,
            scope='closed-circuit Coulomb A bounds inside the saved clear bore; ordinary derivatives between raster flybacks',derivatives=rows)

    def to_metadata(self):
        return dict(schema=SCHEMA,source_id=self.source_id,owner_ids=self.owner_ids,
            base_domain=self.base.to_metadata(),drives=[d.to_dict() for d in self.drives],
            current_gain_matrices_a_per_mrad=self.current_gain_matrices_a_per_mrad,
            reference_time_s=self.reference_time_s,static=False,dynamic_drive_frozen=False,
            induced_electric_field_modelled=True,temporal_bounds=self.temporal_bounds(),
            time_reference='absolute laboratory seconds',wave_admission=False,
            limitations=['Prescribed quasistatic closed winding currents; no inductance, circuit or radiation solve',
                         'Raster flyback impulses require the same explicit temporal jump admission as native scans'])

    def _payload(self):
        return dict(schema=SCHEMA,base=self.base.to_dict(),owner_ids=self.owner_ids,
            drives=[d.to_dict() for d in self.drives],
            current_gain_matrices_a_per_mrad=self.current_gain_matrices_a_per_mrad,
            reference_time_s=self.reference_time_s,implementation_sha256=_implementation_identity())

    def to_dict(self):return {**self._payload(),'digest':self.digest}

    @classmethod
    def from_dict(cls,record):
        from temsim.physics.dynamic_potential import NativeTransverseDrive
        expected={'schema','base','owner_ids','drives','current_gain_matrices_a_per_mrad','reference_time_s','digest','implementation_sha256'}
        if (not isinstance(record,Mapping) or set(record)!=expected or record['schema']!=SCHEMA
                or record['implementation_sha256']!=_implementation_identity()):
            raise ValueError('Invalid dynamic prescribed-current coil record')
        result=cls(TransverseCoilPotential.from_dict(record['base']),tuple(record['owner_ids']),
            tuple(NativeTransverseDrive.from_dict(d) for d in record['drives']),
            tuple(record['current_gain_matrices_a_per_mrad']),record['reference_time_s'])
        if result.digest!=record['digest']:
            raise ValueError('Dynamic prescribed-current source identity mismatch')
        return result


def capture_dynamic_transverse_coil(row,components):
    """Retain each logical native waveform once, with all station couplings."""
    from temsim.optics.ac_deflector import AcDeflectorComponent
    from temsim.optics.descan_deflector import DescanDeflectorComponent
    from temsim.physics.dynamic_potential import NativeTransverseDrive
    potential=row.potential
    if type(potential) is not TransverseCoilPotential:
        raise ValueError('Dynamic current capture requires its exact closed winding potential')
    data=potential.provenance
    if not data.get('dynamic_drive_frozen',False):return None
    station=int(data['station_index'])
    electrical=np.asarray(data['electrical']['matrix_a_per_unit'])
    current_rows=electrical[2*station:2*station+2]
    laws=[];gains=[]
    for key in data['drive_keys']:
        component=components[key]
        if not (getattr(component,'scan_enabled',False) or getattr(component,'wobble_enabled',False)):
            continue
        if type(component) not in (AcDeflectorComponent,DescanDeflectorComponent):
            raise ValueError(f'{key}: no retained native current waveform for this component type')
        methods=('kick_events','coil_kick_matrices','coil_kicks_mrad','dynamic_kick_mrad',
                 'instantaneous_kick_mrad','scan_kick_mrad','scan_factors')
        if type(component) is AcDeflectorComponent:methods=(*methods,'wobble_factor')
        if any(getattr(getattr(component,name),'__func__',None) is not getattr(type(component),name) for name in methods):
            raise ValueError(f'{key}: overridden native waveforms require an explicit dynamic current model')
        coupling=component.coil_kick_matrices()
        if len(coupling)*2!=electrical.shape[1]:
            raise ValueError('Dynamic native command stages do not match the electrical calibration')
        gain=sum(current_rows[:,2*i:2*i+2]@np.asarray(matrix) for i,matrix in enumerate(coupling))
        if component.scan_enabled:
            law=NativeTransverseDrive(key,'raster',tuple(map(tuple,np.asarray(component.scan_command_matrix_mrad))),
                component.scan_frame_period_s,component.scan_lines)
        else:
            amplitude=(component.wobble_amplitude_x_mrad,component.wobble_amplitude_y_mrad)
            law=NativeTransverseDrive(key,'sine',tuple(map(tuple,np.c_[amplitude,np.zeros(2)])),
                component.wobble_period_s,phase_rad=np.deg2rad(component.wobble_phase_deg))
        if law.varying and np.any(gain!=0.):
            laws.append(law);gains.append(tuple(map(tuple,gain)))
    if not laws:return None
    return DynamicTransverseCoilPotential(potential,row.owner_ids,tuple(laws),tuple(gains),data['captured_time_s'])
