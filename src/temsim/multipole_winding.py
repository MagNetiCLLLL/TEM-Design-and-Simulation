"""Explicit vacuum winding recipes for the installed corrector multipoles.

The auxiliary channels retain their existing transport/control roles. Their
opt-in engineering winding CAD is not an additional opaque cartridge or a new
beam boundary. Primary HP windings keep their existing pack and bore geometry.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
import re

from temsim.transverse_winding import _finite, _pack


PREFIX = 'multipole_winding_'
TURN_FIELD = PREFIX+'turn_count_per_coil'
AREA_FIELD = PREFIX+'wire_cross_section_mm2'
INSULATION_FIELD = PREFIX+'insulation_thickness_mm'
BASIS_FIELD = PREFIX+'basis_angle_deg'
STATUS_FIELD = PREFIX+'geometry_status'
NOTE_FIELD = PREFIX+'source_note'
FIELDS = frozenset((TURN_FIELD,AREA_FIELD,INSULATION_FIELD,BASIS_FIELD,STATUS_FIELD,NOTE_FIELD))
STATUS_VALUE = 'engineering_reconstruction_not_oem'
SOURCE_NOTE = ('Explicit engineering ideal-vacuum corrector winding reconstruction, not '
    'OEM or measured winding data. Closed radial circuits preserve their fixed '
    'allocated envelope, turns and current support. Existing optical controls '
    'and transport boundaries remain authoritative. Auxiliary winding CAD adds '
    'no opaque cartridge. Magnetic cores, pole pieces, permeability, leads, '
    'resistance, heating and current ratings are excluded from the field model.')
MAXIMUM_LAYOUT_TURNS = 65536
_PATTERN = re.compile(r'^(probe|image)_(hp[12]|qpol|qpc|qph[12]|hpol|hpc)_(hexapole|quadrupole)$')
_OLD_TURN = 'hexapole_winding_turn_count_per_coil'
_OLD_AREA = 'hexapole_winding_wire_cross_section_mm2'


def _key(part):
    return str(part.get('geometry_template_key',part.get('key','')))


def supported_multipole_winding(part):
    """Identify supported controls without activating a missing recipe."""
    match = _PATTERN.fullmatch(_key(part))
    if match is None or part.get('branch_path_only'):
        return False
    return (match[2].startswith('q') and match[3]=='quadrupole' or
            not match[2].startswith('q') and match[3]=='hexapole')


def is_primary_multipole(part):
    match = _PATTERN.fullmatch(_key(part))
    return bool(match and match[2] in {'hp1','hp2'} and match[3]=='hexapole')


def uses_legacy_probe_winding(part):
    return is_primary_multipole(part) and _key(part).startswith('probe_')


def turn_field(part):
    """Probe HP keeps its existing canonical, fixed physical N record."""
    return _OLD_TURN if uses_legacy_probe_winding(part) else TURN_FIELD


def area_field(part):
    return _OLD_AREA if uses_legacy_probe_winding(part) else AREA_FIELD


def multipole_winding_owner_key(part):
    """One winding recipe feeds the pre-existing optical/control owner."""
    return str(part.get('field_source_key',part.get('key','')))


def electrical_channel_count(part):
    if not supported_multipole_winding(part):
        raise ValueError('Multipole current channels require a supported corrector owner')
    return 1 if is_primary_multipole(part) else 2


def runtime_basis_delta_rad(part,runtime):
    """Transient primary HP mechanical azimuth relative to its saved reference.

    The native HP orientation is the existing operating installation angle.
    Its change rotates the entire cartridge/source; it never refits current or
    rewrites the saved winding angular datum. Missing CAD operating context
    deliberately displays the saved calibration-reference geometry.
    """
    from temsim.multipole_current import REFERENCE_ORIENTATION_FIELD
    if not is_primary_multipole(part) or REFERENCE_ORIENTATION_FIELD not in part or runtime is None:
        return 0.
    if isinstance(runtime,Mapping):
        if 'orientation_rad' not in runtime:return 0.
        orientation=runtime['orientation_rad']
    elif isinstance(runtime,Real):
        orientation=runtime
    elif hasattr(runtime,'orientation_rad'):
        orientation=runtime.orientation_rad
    else:
        return 0.
    orientation=_finite(orientation,'Runtime native HP orientation')
    reference=_finite(part[REFERENCE_ORIENTATION_FIELD],'Saved native HP orientation reference')
    return orientation-reference


def _envelope(part):
    start,end = (_finite(part[k],k) for k in ('local_start_z_mm','local_end_z_mm'))
    inner_name = next((k for k in ('mechanical_inner_diameter_mm','mechanical_bore_diameter_mm',
        'mechanical_clear_bore_diameter_mm','bore_diameter_mm','clear_bore_diameter_mm') if k in part),None)
    if inner_name is None:
        raise ValueError('Multipole winding requires an explicit allocated bore')
    inner = _finite(part[inner_name],inner_name,positive=True)/2
    outer = _finite(part.get('mechanical_outer_diameter_mm',part.get('outer_diameter_mm')),
        'Outer diameter',positive=True)/2
    if not end>start or not outer>inner:
        raise ValueError('Multipole winding requires positive axial length and an open annular envelope')
    clear = part.get('mechanical_clear_bore_diameter_mm')
    if clear is not None and _finite(clear,'Mechanical clear bore',positive=True)/2>inner:
        raise ValueError('Multipole winding cannot intrude into its declared clear bore')
    return start,end,inner,outer


def multipole_winding_envelopes(part,by_key=None):
    """Pure closed-circuit geometry, available before current calibration.

    Coordinates are module-local millimetres. Positive rectangular current has
    an outward radial magnetic moment. The saved angular datum applies equally
    to source and CAD; it does not alter a native optical orientation setting.
    """
    if not supported_multipole_winding(part):
        return ()
    if part.get('model_3d',{}).get('base',{}).get('kind','existing')!='existing':
        return ()
    start,end,inner,outer = _envelope(part)
    centre,length,width = (start+end)/2,end-start,outer-inner
    order = 2 if _key(part).endswith('quadrupole') else 3
    basis = _finite(part.get(BASIS_FIELD,0.),'Winding angular datum')
    if uses_legacy_probe_winding(part):
        from temsim.probe_hexapole_geometry import probe_hexapole_dimensions
        d = probe_hexapole_dimensions(part)
        low = inner+d['coil_radial_inset_mm']
        high = low+d['coil_radial_length_mm']
        outer_width,outer_height = d['coil_width_mm'],d['coil_height_mm']
        window_width,window_height = d['coil_window_width_mm'],d['coil_window_height_mm']
    elif is_primary_multipole(part):
        # Exact existing image-family native CAD allocation.
        low,high = inner+.28*width,inner+.69*width
        available = min(2*low*math.tan(math.pi/6),.6*width)
        outer_width,outer_height = .80*available,.70*length
        window_width,window_height = .45*available,.48*length
    else:
        if _key(part).startswith('image_'):
            # Image QPol and HPol share an optical/reference envelope. They
            # are distinct windings, so give them disjoint radial sub-packs.
            # The gap includes each rectangular winding's outer corner, not
            # only its radial centre-line support. Full saved ID/OD/Z remain
            # unchanged and no additional transport body is introduced.
            low_fraction,high_fraction = ((.34,.47) if order==2 else (.53,.66))
        else:
            low_fraction,high_fraction = .34,.66
        low,high = inner+low_fraction*width,inner+high_fraction*width
        available = min(2*low*math.tan(math.pi/(4*order)),.65*width)
        outer_width,outer_height = .80*available,.70*length
        window_width,window_height = .45*available,.48*length
    if not (0<window_width<outer_width and 0<window_height<outer_height<=length and inner<low<high<outer):
        raise ValueError('Multipole winding windows and radial support must fit the fixed envelope')
    if math.hypot(high,outer_width/2)>=outer:
        raise ValueError('Multipole winding corner leaves its fixed outer envelope')
    coils=[]
    groups=electrical_channel_count(part)
    for group in range(groups):
        for index in range(2*order):
            coils.append(dict(family='radial_rectangle',coil_index=len(coils)+1,
                station_index=0,group='xy'[group],current_sign=(-1)**index,
                azimuth_deg=basis+group*90/order+index*180/order,center_z_mm=centre,
                radial_start_mm=low,radial_end_mm=high,outer_width_mm=outer_width,
                outer_height_mm=outer_height,window_width_mm=window_width,window_height_mm=window_height))
    return tuple(coils)


def _section(coil):
    return (coil['radial_end_mm']-coil['radial_start_mm'],
        min((coil['outer_width_mm']-coil['window_width_mm'])/2,
            (coil['outer_height_mm']-coil['window_height_mm'])/2))


def multipole_winding_definition(part,by_key=None):
    """Validate an explicit recipe and return exact N positions or bulk CAD."""
    supplied={str(k) for k in part if str(k).startswith(PREFIX)}
    if not supplied:
        return None
    if supplied-FIELDS:
        raise ValueError(f'Unknown multipole winding fields: {sorted(supplied-FIELDS)}')
    if part.get('model_3d',{}).get('base',{}).get('kind','existing')!='existing':
        return None
    if not supported_multipole_winding(part):
        raise ValueError('Multipole winding inputs require a supported corrector owner')
    if uses_legacy_probe_winding(part) and ({TURN_FIELD,AREA_FIELD}&supplied):
        raise ValueError('Probe HP uses its existing canonical hexapole turn count and metal area')
    count=part.get(turn_field(part))
    if isinstance(count,bool) or not isinstance(count,Integral) or count<=0:
        raise ValueError('Multipole winding turns per coil must be a positive integer')
    count=int(count)
    if part.get(STATUS_FIELD)!=STATUS_VALUE or not isinstance(part.get(NOTE_FIELD),str) or not part[NOTE_FIELD].strip():
        raise ValueError('Multipole winding requires explicit engineering status and source note')
    coils=multipole_winding_envelopes(part,by_key)
    if count*len(coils)>MAXIMUM_LAYOUT_TURNS or uses_legacy_probe_winding(part) and count*6>4096:
        raise ValueError('Detailed multipole winding layout exceeds the explicit turn budget; no turns were omitted')
    area=part.get(area_field(part))
    if area is not None:
        area=_finite(area,'Conductor metal cross-sectional area',positive=True)
    enamel=_finite(part.get(INSULATION_FIELD,0.),'Radial wire insulation thickness',nonnegative=True)
    diameter=None if area is None else 2*math.sqrt(area/math.pi)
    detailed=area is not None
    packing=tuple(_pack(*_section(coil),diameter+2*enamel,count) for coil in coils) if detailed else ()
    return dict(owner_key=multipole_winding_owner_key(part),order=2 if _key(part).endswith('quadrupole') else 3,
        electrical_channel_count=electrical_channel_count(part),turn_count_per_coil=count,
        coil_count=len(coils),total_turn_count=count*len(coils),coils=coils,
        wire_cross_section_mm2=area,wire_diameter_mm=diameter,insulation_thickness_mm=enamel,
        basis_angle_deg=float(part.get(BASIS_FIELD,0.)),detailed_cad=detailed,
        turn_positions=tuple(p['positions'] for p in packing),packing=packing,
        minimum_gap_mm=min((p['minimum_gap_mm'] for p in packing),default=None),
        geometry_status=part[STATUS_FIELD],source_note=part[NOTE_FIELD],
        boundary_policy='Existing transport boundaries unchanged; auxiliary winding-only reconstruction')


def validate_multipole_winding(part,by_key=None):
    multipole_winding_definition(part,by_key)


def estimate_multipole_winding(part,wire_cross_section_mm2=.02,metal_fill_fraction=.60,insulation_thickness_mm=.005):
    """Detached engineering initial N from the limiting insulated pack section.

    Existing Probe HP N is retained exactly. No field target enters this
    estimator. Other counts are authored once and remain explicit inputs.
    """
    coils=multipole_winding_envelopes(part)
    if not coils:
        raise ValueError('Initial turn estimation needs supported native winding geometry')
    area=_finite(wire_cross_section_mm2,'Conductor metal area',positive=True)
    fill=_finite(metal_fill_fraction,'Nominal metal fill',positive=True)
    if fill>1:
        raise ValueError('Nominal metal fill fraction must not exceed one')
    enamel=_finite(insulation_thickness_mm,'Wire insulation thickness',nonnegative=True)
    diameter=2*math.sqrt(area/math.pi)
    pitch=max(math.sqrt(area/fill),diameter+2*enamel)
    limiting=min((_section(c) for c in coils),key=lambda pair:pair[0]*pair[1])
    if uses_legacy_probe_winding(part) and part.get(_OLD_TURN) is not None:
        count=part[_OLD_TURN]
        if isinstance(count,bool) or not isinstance(count,Integral) or count<=0:
            raise ValueError('The saved physical Probe HP turn count must remain valid')
        count=int(count)
    else:
        count=min(math.floor(a/pitch)*math.floor(b/pitch) for a,b in map(_section,coils))
        if count<1:
            raise ValueError('The chosen insulated conductor cannot fit the allocated winding section')
    # Require the actual deterministic packing as well as the area estimate.
    for coil in coils:
        _pack(*_section(coil),diameter+2*enamel,count)
    return dict(turn_count_per_coil=count,total_turn_count=count*len(coils),
        wire_cross_section_mm2=area,wire_diameter_mm=diameter,insulation_thickness_mm=enamel,
        nominal_metal_fill_fraction=fill,initial_grid_pitch_mm=pitch,
        limiting_cross_section_mm2=limiting[0]*limiting[1],
        actual_metal_fill_fraction=count*area/(limiting[0]*limiting[1]),
        source_kind='estimated',method='fixed_physical_probe_HP_or_fine_wire_limiting_pack_v1',
        physical_probe_HP_count_preserved=uses_legacy_probe_winding(part),automatic_geometry_binding=False)
