"""Explicit engineering winding packs for magnetic steering and stigmation.

Coordinates are module-local millimetres. Closed circuits, polarities, and the
fixed pack envelope are shared by CAD and the vacuum prescribed-current source.
No pole-piece, permeability, resistance, heating or OEM dimensions are inferred.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

from temsim.component_representation import non_material_role
from temsim.corrector_model_3d import winding_axial_extent, winding_dimensions
from temsim.mechanical_profiles import DEFLECTOR_COIL_PROFILES

PREFIX = 'transverse_winding_'
TURN_FIELD = PREFIX+'turn_count_per_coil'
AREA_FIELD = PREFIX+'wire_cross_section_mm2'
INSULATION_FIELD = PREFIX+'insulation_thickness_mm'
STATUS_FIELD = PREFIX+'geometry_status'
NOTE_FIELD = PREFIX+'source_note'
FIELDS = frozenset((TURN_FIELD, AREA_FIELD, INSULATION_FIELD, STATUS_FIELD, NOTE_FIELD))
MAXIMUM_LAYOUT_TURNS = 65536
SOURCE_NOTE = ('Engineering fine-wire winding reconstruction based on US5994704A, '
    'US20090079531A1 and CN102509689B; not measured or OEM data. Closed turns have '
    'explicit channel assignment and alternating current senses; unresolved lead '
    'routing is excluded. The winding pack envelope is fixed. Magnetic cores, '
    'pole pieces, resistance, heating and current ratings are excluded from the '
    'ideal vacuum field model.')


def _finite(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number')
    try:
        value=float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite number') from error
    if not math.isfinite(value) or (positive and value<=0) or (nonnegative and value<0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def _envelope(part):
    start,end=(_finite(part[k],k) for k in ('local_start_z_mm','local_end_z_mm'))
    ik=next((k for k in ('mechanical_inner_diameter_mm','mechanical_bore_diameter_mm',
        'mechanical_clear_bore_diameter_mm','bore_diameter_mm','clear_bore_diameter_mm') if k in part),None)
    if ik is None:
        raise ValueError('Transverse winding requires an explicit material bore')
    inner=_finite(part[ik],ik,positive=True)/2
    outer=_finite(part.get('mechanical_outer_diameter_mm',part.get('outer_diameter_mm')),
                  'Outer diameter',positive=True)/2
    if not end>start or not outer>inner:
        raise ValueError('Transverse winding requires positive length and an open annular envelope')
    return start,end,inner,outer


def supported_transverse_winding(part):
    """Only identified material coil hosts; aliases never acquire a source."""
    if non_material_role(part) or part.get('branch_path_only'):
        return False
    if part.get('model_3d',{}).get('base',{}).get('kind','existing')!='existing':
        return False
    if part.get('mechanical_profile') in DEFLECTOR_COIL_PROFILES:
        return True
    if part.get('coil_topology')=='two_interleaved_four_pole_windings':
        return True
    if part.get('gun_stigmator_geometry_profile')=='eight_coil_slotted_carrier':
        return True
    key=str(part.get('geometry_template_key',part.get('key','')))
    return ((part.get('mechanical_profile')=='magnetic_deflector' or key.endswith('deflector'))
            and 'interaction_centers_local_z_mm' in part
            and not str(part.get('mechanical_profile','')).startswith('electrostatic_'))


def transverse_winding_owner_key(part):
    """Existing optical owner; material insert names do not create extra drives."""
    key=str(part.get('key',''))
    if part.get('mechanical_profile') in DEFLECTOR_COIL_PROFILES:
        owner=part.get('field_source_key')
        return str(owner) if owner else key.removesuffix('_coils')+'_deflector'
    return key


def transverse_winding_envelopes(part):
    """Return detached per-coil geometry/circuits independently of wire size.

    ``station_index`` is zero-based; ``coil_index`` is one-based. Current senses
    use outward-normal radial rectangular loops and +Z at the positive-angle
    long side of the first saddle, then opposite senses in its opposing mate.
    CAD group names identify wiring, not a promised optical X/Y field phase.
    """
    if not supported_transverse_winding(part):
        return ()
    start,end,inner,outer=_envelope(part)
    width,length=outer-inner,end-start
    coils=[]
    if part.get('gun_stigmator_geometry_profile')=='eight_coil_slotted_carrier':
        from temsim.cad_gun_stigmator import gun_stigmator_dimensions
        d=gun_stigmator_dimensions(part)
        for i in range(8):
            coils.append(dict(family='radial_rectangle',coil_index=i+1,station_index=0,
                group='xy'[i%2],current_sign=(-1)**(i//2),
                azimuth_deg=d['basis_angle_deg']+45*i,center_z_mm=d['centre_mm'],
                radial_start_mm=d['coil_inner_radius_mm'],radial_end_mm=d['coil_outer_radius_mm'],
                outer_width_mm=d['coil_width_mm'],outer_height_mm=d['coil_length_mm'],
                window_width_mm=d['window_width_mm'],window_height_mm=d['window_length_mm']))
    elif part.get('coil_topology')=='two_interleaved_four_pole_windings':
        if (part.get('coil_count')!=8 or isinstance(part.get('coil_count'),bool)
                or not isinstance(part.get('coil_count'),Integral)):
            raise ValueError('Two interleaved four-pole groups require exactly eight material coils')
        basis=tuple(_finite(part.get(f'channel_{axis}_angle_deg',default),'Channel angle')
                    for axis,default in (('x',0.),('y',45.)))
        if abs(math.sin(math.radians(2*(basis[1]-basis[0]))))<1e-6:
            raise ValueError('Stigmator winding groups must be independent quadrupoles')
        angles=[basis[i%2]+90*(i//2) for i in range(8)]
        ordered=sorted(a%360 for a in angles)
        separation=min(b-a for a,b in zip(ordered,ordered[1:]+[ordered[0]+360]))
        low,high=inner+.28*width,inner+.69*width
        available=min(2*low*math.tan(math.radians(separation/2)),.6*width)
        for i,angle in enumerate(angles):
            coils.append(dict(family='radial_rectangle',coil_index=i+1,station_index=0,
                group='xy'[i%2],current_sign=(-1)**(i//2),azimuth_deg=angle,
                center_z_mm=(start+end)/2,radial_start_mm=low,radial_end_mm=high,
                outer_width_mm=.80*available,outer_height_mm=.70*length,
                window_width_mm=.45*available,window_height_mm=.48*length))
    else:
        insert=part.get('mechanical_profile') in DEFLECTOR_COIL_PROFILES
        if insert:
            layer,span,band=winding_dimensions(part)
            a,b=winding_axial_extent(part)
            stations=((a+b)/2,)
            copper_length=b-a
            radii=((inner,inner+layer),(outer-layer,outer))
        else:
            stations=part['interaction_centers_local_z_mm']
            if not isinstance(stations,(list,tuple)) or not 1<=len(stations)<=2:
                raise ValueError('One or two physical deflector stations are required')
            stations=tuple(_finite(v,'Deflector station') for v in stations)
            copper_length=_finite(part.get('mechanical_coil_length_mm',
                part.get('effective_thickness_mm',part.get('active_length_mm',length/len(stations)))),
                'Copper axial length',positive=True)
            layer,span=.10*width,120.
            band=min(.08*copper_length,.2*inner)
            radii=tuple((inner+f*width,inner+(f+.10)*width) for f in (.08,.26))
        if any(c-copper_length/2<start-1e-9 or c+copper_length/2>end+1e-9 for c in stations):
            raise ValueError('Deflector winding leaves its allocated axial envelope')
        if len(stations)==2 and abs(stations[1]-stations[0])<copper_length:
            raise ValueError('Deflector winding stations overlap')
        for station,centre in enumerate(stations):
            for axis,(low,high) in enumerate(radii):
                actual_band=(min(.08*copper_length,.2*low)
                    if insert and part.get('mechanical_profile')!='deflector_coil_insert'
                    and 'winding_band_width_mm' not in part else band)
                half=math.radians(span/2)
                window=half-actual_band/((low+high)/2)
                if not window>0 or not copper_length>2*actual_band:
                    raise ValueError('Saddle winding requires an open central window')
                for side in range(2):
                    coils.append(dict(family='saddle',coil_index=len(coils)+1,
                        station_index=station,group='xy'[axis],current_sign=(-1)**side,
                        azimuth_deg=90*axis+180*side,center_z_mm=centre,
                        start_z_mm=centre-copper_length/2,end_z_mm=centre+copper_length/2,
                        inner_radius_mm=low,outer_radius_mm=high,span_deg=span,
                        band_mm=actual_band,window_half_angle_rad=window))
    return tuple(coils)


def _section(coil):
    if coil['family']=='radial_rectangle':
        length=coil['radial_end_mm']-coil['radial_start_mm']
        band=min((coil['outer_width_mm']-coil['window_width_mm'])/2,
                 (coil['outer_height_mm']-coil['window_height_mm'])/2)
    else:
        length=coil['outer_radius_mm']-coil['inner_radius_mm']
        band=min(coil['band_mm'],coil['inner_radius_mm']*(
            math.radians(coil['span_deg']/2)-coil['window_half_angle_rad']))
    if not length>0 or not band>0:
        raise ValueError('A winding pack must retain a positive straight-side section')
    return length,band


def _pack(length,band,diameter,count):
    # Enamel is included in diameter. Every complete closed turn crosses the
    # narrowest band. The nested loops preserve the original rectangular hole.
    candidates=[]
    guard=8*math.ulp(max(length,band,diameter))
    for layers in range(1,min(count,int(length/diameter))+1):
        rings=(count+layers-1)//layers
        pitch=min(length/layers,band/rings)
        first=(band-(rings-1)*pitch)/2
        if pitch>=diameter-guard and first>diameter/2+guard:
            candidates.append((pitch,-(layers*rings-count),-layers,layers,rings))
    if not candidates:
        raise ValueError('Requested turns and insulated wire do not fit the fixed winding pack; no turns were omitted')
    pitch,_,_,layers,rings=max(candidates)
    base,rem=divmod(count,layers)
    positions=[]
    for layer in range(layers):
        n=base+(layer<rem)
        x=length/2+(layer-(layers-1)/2)*pitch
        for ring in range(n):
            delta=band/2+(ring-(n-1)/2)*pitch
            positions.append((x,delta))
    return dict(positions=tuple(positions),layers=layers,rings=rings,pitch_mm=pitch,
                insulated_wire_diameter_mm=diameter,minimum_gap_mm=max(0.,pitch-diameter))


def transverse_winding_definition(part,by_key=None):
    """Validate saved N/metal area and return a pure deterministic CAD layout."""
    supplied={str(k) for k in part if str(k).startswith(PREFIX)}
    if not supplied:
        return None
    if part.get('model_3d',{}).get('base',{}).get('kind','existing')!='existing':
        return None
    if supplied-FIELDS:
        raise ValueError(f'Unknown transverse winding inputs: {sorted(supplied-FIELDS)}')
    coils=transverse_winding_envelopes(part)
    if not coils:
        raise ValueError('Transverse winding inputs require an identified material coil host')
    n=part.get(TURN_FIELD)
    if isinstance(n,bool) or not isinstance(n,Integral) or n<=0:
        raise ValueError('Transverse winding turns per coil must be a positive integer')
    n=int(n)
    if part.get(STATUS_FIELD)!='engineering_reconstruction_not_oem' or not isinstance(part.get(NOTE_FIELD),str) or not part[NOTE_FIELD].strip():
        raise ValueError('Transverse winding requires engineering status and its source/assumption note')
    t=_finite(part.get(INSULATION_FIELD,0.),'Wire insulation radial thickness',nonnegative=True)
    result=dict(schema='transverse-closed-winding-v1',turn_count_per_coil=n,
        total_turn_count=n*len(coils),coil_count=len(coils),coils=coils,
        insulation_thickness_mm=t,wire_cross_section_mm2=None,wire_diameter_mm=None,
        detailed_cad=False,turn_positions=None,packing=None,source_note=part[NOTE_FIELD])
    if AREA_FIELD not in part:
        return result
    area=_finite(part[AREA_FIELD],'Conductor metal area',positive=True)
    diameter=2*math.sqrt(area/math.pi)
    if result['total_turn_count']>MAXIMUM_LAYOUT_TURNS:
        raise ValueError(f'Transverse winding exceeds the {MAXIMUM_LAYOUT_TURNS}-turn layout budget; no turns were omitted')
    packing=tuple(_pack(*_section(coil),diameter+2*t,n) for coil in coils)
    result.update(wire_cross_section_mm2=area,wire_diameter_mm=diameter,detailed_cad=True,
        turn_positions=tuple(p['positions'] for p in packing),packing=packing)
    return result


def validate_transverse_winding(part,by_key=None):
    transverse_winding_definition(part,by_key)


def estimate_transverse_winding(part,wire_cross_section_mm2=.01,metal_fill_fraction=.60,
                               insulation_thickness_mm=.005):
    """Explicit initial engineering N, bounded by actual insulated-wire packing."""
    coils=transverse_winding_envelopes(part)
    if not coils:
        raise ValueError('Turn estimation requires an identified material transverse coil host')
    area=_finite(wire_cross_section_mm2,'Conductor metal area',positive=True)
    fill=_finite(metal_fill_fraction,'Nominal metal fill fraction',positive=True)
    if fill>=math.pi/4:
        raise ValueError('Nominal metal fill fraction must remain below square-grid round-wire packing')
    t=_finite(insulation_thickness_mm,'Wire insulation radial thickness',nonnegative=True)
    diameter=2*math.sqrt(area/math.pi)
    pitch=max(math.sqrt(area/fill),diameter+2*t)
    sections=tuple(_section(c) for c in coils)
    # Two nested rings need positive bend clearance. A common grid-cell pitch
    # greater than insulated diameter makes the innermost rounded bend regular.
    capacity=min(math.floor(a/pitch)*math.floor(b/pitch) for a,b in sections)
    if capacity<1:
        raise ValueError('Engineering fine-wire allocation cannot fit one complete insulated turn')
    n=capacity
    candidate={**part,TURN_FIELD:n,AREA_FIELD:area,INSULATION_FIELD:t,
               STATUS_FIELD:'engineering_reconstruction_not_oem',NOTE_FIELD:SOURCE_NOTE}
    definition=transverse_winding_definition(candidate)
    return dict(turn_count_per_coil=n,total_turn_count=n*len(coils),
        wire_cross_section_mm2=area,wire_diameter_mm=diameter,insulation_thickness_mm=t,
        nominal_metal_fill_fraction=fill,initial_grid_pitch_mm=pitch,
        limiting_cross_section_mm2=min(a*b for a,b in sections),
        actual_metal_fill_fraction=max(n*area/(a*b) for a,b in sections),
        packing=definition['packing'],fields={field:candidate[field] for field in FIELDS},
        source_kind='estimated',automatic_geometry_binding=False,
        method='fine_wire_common_square_grid_limiting_pack_v1')
