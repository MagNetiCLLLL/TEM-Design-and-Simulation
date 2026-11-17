"""Static closed ideal-current probe and image multipole winding sources.

The line/pack integration and retained A/B are shared with the transverse
winding provider. Calibration uses analytic on-axis magnetic derivatives,
not a fitted finite-radius or imposed Gaussian field. Every axial tail remains.
"""
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from temsim.immutable_json import thaw_json
from temsim.multipole_current import (
    MATRIX_FIELD, BIAS_FIELD, REFERENCE_ENERGY_FIELD, REFERENCE_ORIENTATION_FIELD,
    KIND_FIELD, STATUS_FIELD, NOTE_FIELD, CALIBRATION_KIND,
    validate_multipole_current_inputs,
)
from temsim.physics.transverse_coil_field import (
    TransverseCoilPotential, FrozenTransverseCoilField, _unit_nodes, _positions,
    _components, _momentum, DEFAULT_PACK_ORDER, DEFAULT_PATH_ORDER,
    quadrupole_gradient_t_per_m, potential_weighted_nodes,
)


SCHEMA = 'ideal-vacuum-multipole-current-coil-v1'


@lru_cache(maxsize=1)
def _implementation_identity():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def recommended_basis_angle_deg(component):
    """Declare the nominal radial-current angular datum; do not mutate a pose.

    Radial sextupole loops at zero angle have a skew magnetic coefficient.
    Thirty degrees aligns that winding basis with the native normal phase.
    The signed current fit retains the electron charge sign separately.
    """
    return math.degrees(float(getattr(component,'orientation_rad',0.)))+30.


def magnetic_field_hessian_t_per_m2(potential, positions_m):
    """Analytic global second Cartesian derivatives of the Biot-Savart B.

    Differentiating each closed-circuit line kernel avoids finite-radius
    contamination from higher multipoles and finite-difference cancellation.
    Shape is (..., B_component, coordinate_j, coordinate_k), in T/m2.
    """
    if type(potential) is not TransverseCoilPotential:
        raise TypeError('Magnetic derivatives require an exact retained closed-winding source')
    points = _positions(positions_m)
    local = potential.registration.positions_global_to_local_m(points)
    from temsim.physics.transverse_coil_field import _inside_pack
    if np.any(_inside_pack(local,potential.coils)):
        raise ValueError('Multipole derivatives exclude the occupied winding pack from vacuum')
    flat = local.reshape(-1,3)
    result = np.zeros((len(flat),3,3,3))
    if any(potential.currents_a) and len(flat):
        nodes,weighted=potential_weighted_nodes(potential)
        active=np.any(weighted!=0.,axis=1)
        nodes,weighted=nodes[active],weighted[active]
        basis=np.eye(3)
        crossed=[np.cross(weighted,axis) for axis in basis]
        block=max(1,262144//max(1,len(nodes)))
        for start in range(0,len(flat),block):
            stop=min(start+block,len(flat))
            delta=flat[start:stop,None,:]-nodes[None,:,:]
            squared=np.einsum('pni,pni->pn',delta,delta)
            if np.any(squared<=0.) or not np.isfinite(squared).all():
                raise ValueError('Multipole derivative query is outside the finite vacuum kernel')
            inv5=1/(squared*squared*np.sqrt(squared))
            inv7=inv5/squared
            cross=np.cross(weighted[None,:,:],delta)
            for j in range(3):
                for k in range(j,3):
                    values=15*np.einsum('pni,pn,pn,pn->pi',cross,delta[...,j],delta[...,k],inv7)
                    if j==k:
                        values-=3*np.einsum('pni,pn->pi',cross,inv5)
                    values-=3*np.einsum('ni,pn,pn->pi',crossed[j],delta[...,k],inv5)
                    values-=3*np.einsum('ni,pn,pn->pi',crossed[k],delta[...,j],inv5)
                    result[start:stop,:,j,k]=values
                    result[start:stop,:,k,j]=values
    rotation=potential.registration.rotation_array
    result=np.einsum('ai,pijk,bj,ck->pabc',rotation,result,rotation,rotation)
    if not np.isfinite(result).all():
        raise ValueError('Multipole magnetic derivatives exceed the finite numerical range')
    return result.reshape((*points.shape[:-1],3,3,3))


def multipole_field_coefficients(potential, positions_m, order):
    """Actual normal/skew on-axis Taylor coefficients in T/m**(order-1).

    Quadrupole: Bx=Gn*y-Gs*x; By=Gn*x+Gs*y.
    Sextupole: Bx=2*Hn*x*y-Hs*(x*x-y*y);
               By=Hn*(x*x-y*y)+2*Hs*x*y.
    A harmonic projection removes non-multipole tensor pieces without adding
    a second field. The evaluation point and its axial plane are explicit.
    """
    if isinstance(order,bool) or not isinstance(order,(int,np.integer)):
        raise ValueError('A multipole coefficient order must be integer two or three')
    if order==2:
        return quadrupole_gradient_t_per_m(potential,positions_m)
    if order!=3 or isinstance(order,bool):
        raise ValueError('Only explicitly configured quadrupoles and sextupoles are supported')
    h=magnetic_field_hessian_t_per_m2(potential,positions_m)
    normal=(h[...,1,0,0]-h[...,1,1,1]+2*h[...,0,0,1])/8
    skew=(2*h[...,1,0,1]-h[...,0,0,0]+h[...,0,1,1])/8
    return np.stack((normal,skew),axis=-1)


def _bindings(state, *, require_current, component_keys=None):
    from temsim.multipole_winding import (
        multipole_winding_definition,multipole_winding_owner_key,runtime_basis_delta_rad,
    )
    components=_components(state)
    parts=tuple(getattr(getattr(state,'_resolved_assembly',None),'parts',()))
    by_key={record.key:dict(record.data) for record in parts}
    result=[];seen=set()
    for record in parts:
        part=dict(record.data)
        owner=multipole_winding_owner_key(part)
        if component_keys is not None and owner not in component_keys:
            continue
        if require_current and MATRIX_FIELD not in part:
            continue
        definition=multipole_winding_definition(part,by_key)
        if definition is None:
            validate_multipole_current_inputs(part,by_key)
            continue
        component=components.get(owner)
        if component is None:
            raise ValueError(f'{record.key}: multipole winding has no installed optical owner {owner}')
        if owner in seen:
            raise ValueError(f'{owner}: more than one winding source claims the same multipole')
        electrical=validate_multipole_current_inputs(part,by_key)
        if require_current and electrical is None:
            continue
        seen.add(owner)
        offset=float(record.center_z_mm)-float(part['local_center_z_mm'])
        angular_delta=runtime_basis_delta_rad(part,component)
        definition={**definition,'runtime_basis_delta_rad':angular_delta}
        coils=[]
        for entry in definition['coils']:
            coil=dict(entry)
            coil['azimuth_deg']+=math.degrees(angular_delta)
            for name in ('center_z_mm','start_z_mm','end_z_mm'):
                if name in coil:coil[name]+=offset
            coils.append(coil)
        result.append((part,component,definition,tuple(coils),electrical))
    return tuple(result)


def _native_commands(component, definition, electrical):
    order=definition['order'];count=definition['electrical_channel_count']
    normal=float(getattr(component,f'strength_m{order}'))
    skew=float(getattr(component,f'skew_strength_m{order}',0.))
    angle=float(getattr(component,'orientation_rad',0.)) if order==3 else 0.
    reference=electrical.get('reference_orientation_rad') or 0.
    if not np.isfinite((normal,skew,angle,reference)).all():
        raise ValueError(f'{component.key}: native multipole controls must be finite')
    if count==1:
        if skew!=0.:
            raise ValueError(f'{component.key}: six primary coils have one current basis; a skew drive is not configured')
        return np.array([normal])
    # A dual electrical winding can rotate through current mixing. M remains
    # fixed; only the declared native operating command changes coordinates.
    phase=order*(angle-reference)
    cosine,sine=math.cos(phase),math.sin(phase)
    return np.array([normal*cosine-skew*sine,normal*sine+skew*cosine])


def capture_multipole_coil_fields(state, *, component_keys=None, include_hexapole=True):
    """Capture fixed real current inputs and replace each native owner once."""
    if getattr(state,'round_lens_field_model','gaussian')!='copper_coil':
        return ()
    parts=getattr(getattr(state,'_resolved_assembly',None),'parts',())
    if not any(MATRIX_FIELD in part.data for part in parts):
        return ()
    from temsim.lens_pose import lens_pose_registration
    result=[]
    for part,component,definition,coils,electrical in _bindings(state,require_current=True,component_keys=component_keys):
        if not component.enabled or (definition['order']==3 and not include_hexapole):
            continue
        command=_native_commands(component,definition,electrical)
        currents=np.asarray(electrical['matrix_a_per_unit'])@command+electrical['current_bias_a']
        physical=tuple(currents) if len(currents)==2 else (float(currents[0]),0.)
        potential=TransverseCoilPotential(component.key,coils,definition['turn_count_per_coil'],physical,
            lens_pose_registration(state,component.key),provenance=dict(
                source_model='ideal_multipole_current_coil',material_host=part['key'],component_key=component.key,
                multipole_order=definition['order'],electrical=electrical,
                command_unit=f'm^-{definition["order"]}',homogenized_pack=True,
                magnetic_materials=False,capture_implementation_sha256=_implementation_identity(),
                capture_schema=SCHEMA,
                runtime_basis_delta_rad=definition['runtime_basis_delta_rad'],
                native_installation_orientation_rad=float(getattr(component,'orientation_rad',0.)),
                clear_bore_radius_m=float(getattr(component,'mechanical_clear_bore_diameter_mm'))*.0005))
        from temsim.ideal_transverse_segments import apply_to_potential
        potential=apply_to_potential(state,potential)
        result.append(FrozenTransverseCoilField(potential,component.key,
            _momentum(electrical['reference_energy_ev'])))
    return tuple(result)


def calibrate_multipole_current_matrix(state,host_key, *, pack_order=DEFAULT_PACK_ORDER,path_order=DEFAULT_PATH_ORDER):
    """Detached fixed-N fit at the original signed native coefficient plane."""
    from temsim.lens_pose import lens_pose_registration
    from temsim.physics.core import electron
    rows=[row for row in _bindings(state,require_current=False) if row[0]['key']==host_key]
    if len(rows)!=1:
        raise ValueError(f'{host_key}: choose one installed explicit multipole winding recipe')
    part,component,definition,coils,_=rows[0]
    registration=lens_pose_registration(state,component.key)
    if any(registration.origin_global_m) or not np.array_equal(registration.rotation_array,np.eye(3)):
        raise ValueError('Initial multipole current calibration requires the nominal aligned winding pose')
    order=definition['order'];count=definition['electrical_channel_count']
    points=np.array([[0.,0.,float(component.z_mm)*.001]])
    response=np.zeros((2,count))
    for index in range(count):
        current=[0.,0.];current[index]=1.
        source=TransverseCoilPotential('unit_multipole_current',coils,definition['turn_count_per_coil'],tuple(current),
            registration,pack_order,path_order)
        response[:,index]=multipole_field_coefficients(source,points,order)[0]
    angle=float(getattr(component,'orientation_rad',0.)) if order==3 else 0.
    phase=order*angle
    rotation=np.array(((math.cos(phase),-math.sin(phase)),(math.sin(phase),math.cos(phase))))
    charge,momentum,_=electron(state)
    target=(momentum/charge)*rotation[:,:count]
    native_reference_basis=[dict(column=0,kind='existing_native',input=f'strength_m{order}')]
    if count==2:
        if order==2 and not hasattr(component,'skew_quadrupole_strength_m2'):
            native_reference_basis.append(dict(column=1,kind='engineering_quadrature',input=None,
                construction='45_degree_rotated_copy_of_existing_native_normal_coefficient'))
        else:
            native_reference_basis.append(dict(column=1,kind='existing_native',input=f'skew_strength_m{order}'))
    condition=float(np.linalg.cond(response))
    if not math.isfinite(condition) or condition>1e8:
        raise ValueError(f'{host_key}: multipole unit-current response is singular or ill conditioned')
    matrix=np.linalg.lstsq(response,target,rcond=None)[0]
    residual=response@matrix-target
    relative=float(np.max(abs(residual))/np.max(abs(target)))
    if relative>1e-10:
        raise ValueError(f'{host_key}: fixed winding phase cannot reproduce the native signed multipole (relative mismatch {relative:.6g})')
    np.testing.assert_allclose(response@matrix,target,rtol=1e-10,atol=np.max(abs(target))*1e-12)
    energy=float(state.beam_voltage_kv)*1000
    note=(f'Ideal vacuum closed-coil order-{order} current calibration at the original native reference plane; '
        f'native strength in m^-{order}, matrix in A/(m^-{order}), reference energy {energy:.12g} eV. '
        'Fixed physical N, pack bounds and saved winding angular datum retained. '
        'Signed normal/skew on-axis Taylor coefficients use analytic Biot-Savart derivatives. '
        'No magnetic core, pole response, field-integral/profile equivalence, focus adjustment, thermal rating or experimental calibration is inferred. '
        'The numerical current reference stays fixed after energy or geometry edits; re-calibration is explicit.')
    if any(row['kind']=='engineering_quadrature' for row in native_reference_basis):
        note+=(' This legacy image quadrupole has one existing native normal input; the second electrical '
               'target is an explicitly new engineering 45-degree-rotated copy of that normal coefficient, '
               'not a previously executed native skew control.')
    winding_reference=(float(part[REFERENCE_ORIENTATION_FIELD])
        if count==1 and REFERENCE_ORIENTATION_FIELD in part else angle)
    if count==1:
        note+=(' The existing native installation orientation is a mechanical cartridge rotation relative '
               'to the saved angular reference. Runtime CAD and source rotate together; no current refit '
               'or magnetic-basis projection occurs. An existing angular datum is retained by re-calibration.')
    fields={MATRIX_FIELD:matrix.tolist(),BIAS_FIELD:[0.]*count,REFERENCE_ENERGY_FIELD:energy,
        REFERENCE_ORIENTATION_FIELD:winding_reference,KIND_FIELD:CALIBRATION_KIND,
        STATUS_FIELD:'calibrated_native_peak',NOTE_FIELD:note}
    return dict(host_key=part['key'],component_key=component.key,multipole_order=order,
        command_unit=f'm^-{order}',field_coefficient_unit=f'T/m^{order-1}',reference_planes_z_mm=[float(component.z_mm)],
        reference_orientation_rad=angle,calibration_native_orientation_rad=angle,
        winding_orientation_reference_rad=winding_reference,native_reference_basis=native_reference_basis,
        unit_current_response=response.tolist(),
        native_target_per_unit=target.tolist(),matrix_a_per_unit=matrix.tolist(),
        response_condition_number=condition,maximum_abs_match_error=float(np.max(abs(residual))),
        maximum_relative_match_error=relative,turn_count_per_coil=definition['turn_count_per_coil'],
        pack_order=pack_order,path_order=path_order,fields=fields,
        source_model='homogenized_closed_multipole_pack_in_vacuum',native_controls_unchanged=True,
        focus_or_source_calculation_executed=False)


__all__=('multipole_field_coefficients','magnetic_field_hessian_t_per_m2',
    'recommended_basis_angle_deg','capture_multipole_coil_fields','calibrate_multipole_current_matrix')
