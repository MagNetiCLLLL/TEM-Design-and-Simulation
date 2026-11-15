"""Opt-in radial-axis round-wire CAD for the six-pin probe hexapoles.

The saved copper frame is the fixed pack envelope. Explicit user N and metal
area replace that frame by rounded rectangular wire turns plus its remaining
insulation. This is an engineering CAD recipe, not a current calibration or a
magnetic-material solve. The native multipole field remains independent.
"""
from __future__ import annotations

from numbers import Integral, Real
import math

from temsim.cad_geometry import SolidSpec, annulus, box, difference, torus, transformed, union
from temsim.coil_winding import resolve_wire_diameter_mm
from temsim.probe_hexapole_geometry import is_probe_hexapole, probe_hexapole_dimensions

PREFIX = 'hexapole_winding_'
TURN_FIELD = PREFIX+'turn_count_per_coil'
AREA_FIELD = PREFIX+'wire_cross_section_mm2'
STATUS_FIELD = PREFIX+'geometry_status'
NOTE_FIELD = PREFIX+'source_note'
FIELDS = frozenset((TURN_FIELD, AREA_FIELD, STATUS_FIELD, NOTE_FIELD))
MAXIMUM_TURNS_PER_COIL = 2048
MAXIMUM_TOTAL_TURNS = 4096
_EPSILON = 2.220446049250313e-16


def estimate_probe_winding_turns(part, nominal_allocated_cross_section_mm2_per_turn):
    """Author fixed radial-coil N from the limiting winding-pack section.

    The radial coil axis is perpendicular to the rectangular Y-Z loop. A cut
    through any straight side exposes radial pack length times that side's
    band thickness. Every complete turn must fit through the narrowest side,
    so use the smaller Y/Z band, not the full frame area or six-coil sum.
    The nominal area includes insulation and unused space. It is not metal
    area, a conductor diameter, a current, or an OEM winding prescription.
    This pure authoring helper never binds saved N to future geometry edits.
    """
    if not is_probe_hexapole(part):
        raise ValueError('Radial winding estimates require a material HP1/HP2 hexapole')
    nominal = nominal_allocated_cross_section_mm2_per_turn
    if isinstance(nominal, bool) or not isinstance(nominal, Real):
        raise ValueError('Nominal allocated area per turn must be finite and positive')
    try:
        nominal = float(nominal)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError('Nominal allocated area per turn must be finite and positive') from error
    if not math.isfinite(nominal) or nominal <= 0:
        raise ValueError('Nominal allocated area per turn must be finite and positive')
    d = probe_hexapole_dimensions(part)
    radial_length = d['coil_radial_length_mm']
    tangential_band = (d['coil_width_mm']-d['coil_window_width_mm'])/2
    axial_band = (d['coil_height_mm']-d['coil_window_height_mm'])/2
    band = min(tangential_band, axial_band)
    area = radial_length*band
    unrounded = area/nominal
    if not math.isfinite(area) or not math.isfinite(unrounded) or unrounded <= 0:
        raise ValueError('Estimated radial winding turns exceed the positive finite range')
    whole = math.floor(unrounded)
    count = max(1, whole+int(unrounded-whole >= .5))
    return dict(turn_count_per_coil=count, total_turn_count=6*count,
        source_kind='estimated', method='limiting_radial_winding_pack_area_v1',
        radial_pack_length_mm=radial_length,
        tangential_band_mm=tangential_band, axial_band_mm=axial_band,
        minimum_band_mm=band, cross_section_mm2=area,
        nominal_allocated_cross_section_mm2_per_turn=nominal,
        actual_allocated_cross_section_mm2_per_turn=area/count,
        unrounded_turn_count=unrounded,
        rounding='nearest_half_up; minimum one turn per coil',
        wire_cross_section_mm2=None, wire_diameter_mm=None,
        current_a=None, field_or_current_calibration=False,
        automatic_geometry_binding=False)


def _pack(length, band, diameter, count, radial_start):
    """Maximize common centre pitch while retaining every requested turn."""
    radius = diameter/2
    guard = 8*_EPSILON*max(length, band, diameter)
    candidates = []
    for layers in range(1, count+1):
        rings = (count+layers-1)//layers
        pitch = min(length/layers, band/rings)
        first = (band-(rings-1)*pitch)/2
        # A rounded corner must have positive inner bend radius. A horn torus
        # would be self-touching and is not a valid conductor solid.
        if pitch >= diameter-guard and first > radius+guard:
            candidates.append((pitch, -(layers*rings-count), -abs(length/layers-band/rings), -layers, layers, rings))
    if not candidates:
        raise ValueError('Requested hexapole turns and metal area do not fit the fixed radial winding pack with positive bend radius')
    pitch, _, _, _, layers, rings = max(candidates)
    base, remainder = divmod(count, layers)
    positions = []
    for i in range(layers):
        local_count = base+(i < remainder)
        x = radial_start+length/2+(i-(layers-1)/2)*pitch
        for j in range(local_count):
            delta = band/2+(j-(local_count-1)/2)*pitch
            positions.append(dict(radial_x_mm=x, corner_offset_mm=delta))
    assert len(positions) == count
    return positions, layers, rings, pitch


def probe_winding_dimensions(part):
    """Return detailed packing, or None when inactive or metal size is unknown.

    A known physical turn count remains valid without a guessed metal area.
    The saved bulk frame is retained until the user supplies conductor size.
    """
    if part.get('model_3d', {}).get('base', {}).get('kind', 'existing') != 'existing':
        return None
    supplied = {str(key) for key in part if str(key).startswith(PREFIX)}
    if supplied-FIELDS:
        raise ValueError(f"{part.get('key')}: unknown hexapole winding fields: {sorted(supplied-FIELDS)}")
    if not {TURN_FIELD, AREA_FIELD}.intersection(supplied):
        return None
    if not is_probe_hexapole(part):
        raise ValueError('Detailed radial windings require an installed material HP1/HP2 hexapole')
    if TURN_FIELD not in part:
        raise ValueError('Hexapole conductor metal area requires turns per coil')
    count = part[TURN_FIELD]
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError('Hexapole turns per coil must be a positive integer')
    count = int(count)
    if part.get(STATUS_FIELD) != 'engineering_reconstruction_not_oem':
        raise ValueError('Detailed hexapole winding CAD requires explicit engineering_reconstruction_not_oem status')
    if not isinstance(part.get(NOTE_FIELD), str) or not part[NOTE_FIELD].strip():
        raise ValueError('Detailed hexapole winding CAD requires its source/assumption note')
    if AREA_FIELD not in part:
        return None
    if count > MAXIMUM_TURNS_PER_COIL or 6*count > MAXIMUM_TOTAL_TURNS:
        raise ValueError(f'Hexapole CAD requests {6*count} turns; the {MAXIMUM_TOTAL_TURNS}-turn allocation budget was exceeded. No turns were omitted')
    diameter = resolve_wire_diameter_mm({'winding_wire_cross_section_mm2':part[AREA_FIELD]})
    area = float(part[AREA_FIELD])
    d = probe_hexapole_dimensions(part)
    start = d['inner_radius_mm']+d['coil_radial_inset_mm']
    length = d['coil_radial_length_mm']
    band = min((d['coil_width_mm']-d['coil_window_width_mm'])/2,
               (d['coil_height_mm']-d['coil_window_height_mm'])/2)
    positions, layers, rings, pitch = _pack(length, band, diameter, count, start)
    path_lengths = [2*(d['coil_window_width_mm']+d['coil_window_height_mm'])
                    +2*math.pi*p['corner_offset_mm'] for p in positions]
    volume = area*math.fsum(path_lengths)
    bulk = length*(d['coil_width_mm']*d['coil_height_mm']
                   -d['coil_window_width_mm']*d['coil_window_height_mm'])
    if not math.isfinite(volume) or not 0 < volume < bulk:
        raise ValueError('Hexapole wire metal volume must fit its fixed original copper frame')
    return dict(schema='radial-rounded-rectangle-winding-v1',
        turn_count_per_coil=count, total_turn_count=6*count,
        wire_cross_section_mm2=area, wire_diameter_mm=diameter,
        radial_layers=layers, offset_rings=rings, centre_pitch_mm=pitch,
        interwire_gap_mm=max(0.,pitch-diameter),
        radial_start_mm=start, radial_end_mm=start+length,
        window_width_mm=d['coil_window_width_mm'], window_height_mm=d['coil_window_height_mm'],
        coil_width_mm=d['coil_width_mm'], coil_height_mm=d['coil_height_mm'],
        minimum_band_mm=band, axial_centre_mm=(d['start_mm']+d['end_mm'])/2,
        turn_positions=positions, path_lengths_mm=path_lengths,
        bulk_coil_volume_mm3=bulk, conductor_volume_per_coil_mm3=volume,
        remaining_insulation_volume_per_coil_mm3=bulk-volume,
        coil_axis='radial', pole_pitch_deg=60.,
        geometry_status='engineering_reconstruction_not_oem',
        field_or_current_calibration=False, optical_field_unchanged=True,
        source_note=part[NOTE_FIELD])


def validate_probe_winding(part):
    probe_winding_dimensions(part)


def rounded_rectangle_turn(radial_x_mm, width_mm, height_mm, offset_mm, wire_radius_mm):
    """Closed circular-section tube around an offset rectangle in the YZ plane.

    Width and height describe the original rectangular coil window, not the
    outside conductor boundary. Corners are circular arcs of the stated offset.
    The analytic cylinders and quarter tori also serve the exact STEP builder.
    """
    x, width, height, delta, radius = map(float, (radial_x_mm, width_mm, height_mm, offset_mm, wire_radius_mm))
    if not all(math.isfinite(v) for v in (x,width,height,delta,radius)) or not (width>0 and height>0 and delta>radius>0):
        raise ValueError('Rounded rectangular wire needs positive dimensions and bend radius greater than wire radius')
    axis_x = [[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    axis_y = [[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]
    pieces = []
    for sign in (-1,1):
        pieces.append(transformed(annulus(-width/2,width/2,0.,radius), rotation=axis_y,
                                  offset=(x,0.,sign*(height/2+delta))))
        pieces.append(transformed(annulus(-height/2,height/2,0.,radius),
                                  offset=(x,sign*(width/2+delta),0.)))
    reach = 4*(delta+radius)
    for sy in (-1,1):
        for sz in (-1,1):
            y,z=sy*width/2,sz*height/2
            arc=transformed(torus(delta,radius,0.),rotation=axis_x,offset=(x,y,z))
            outside_y=box((x,y-sy*reach/2,z),(reach,reach,2*reach))
            outside_z=box((x,y,z-sz*reach/2),(reach,2*reach,reach))
            pieces.append(difference(arc,outside_y,outside_z))
    return union(*pieces)


def probe_winding_solids(part):
    """Replace six bulk frames by exact requested wires and remaining insulation."""
    d=probe_winding_dimensions(part)
    if d is None:return None
    radius=d['wire_diameter_mm']/2
    centre=d['axial_centre_mm']
    wires=tuple(transformed(rounded_rectangle_turn(p['radial_x_mm'],d['window_width_mm'],
        d['window_height_mm'],p['corner_offset_mm'],radius),offset=(0.,0.,centre)) for p in d['turn_positions'])
    start,end=d['radial_start_mm'],d['radial_end_mm']
    bulk=difference(box(((start+end)/2,0.,centre),(end-start,d['coil_width_mm'],d['coil_height_mm'])),
        box(((start+end)/2,0.,centre),(2*end,d['window_width_mm'],d['window_height_mm'])))
    insulation=difference(bulk,*wires)
    note=('Explicit user-sized radial-axis rounded rectangular round-wire CAD. '
          'Remaining original winding-frame volume is insulation. Original pole/core, '
          'winding envelope and multipole strength are unchanged; no electrical current '
          'or magnetic calibration is inferred. Not OEM winding data.')
    result=[]
    for coil in range(6):
        angle=math.radians(60*coil);c,s=math.cos(angle),math.sin(angle)
        rotation=[[c,-s,0.],[s,c,0.],[0.,0.,1.]]
        for index,wire in enumerate(wires):
            result.append(SolidSpec(str(part['key']),f'winding_turn_coil_{coil+1}_{index+1:06d}',
                'copper_winding',transformed(wire,rotation=rotation),(),
                'configured_geometry',note,dict(winding_turn_index=index+1,coil_index=coil+1,
                    wire_cross_section_mm2=d['wire_cross_section_mm2'],wire_diameter_mm=d['wire_diameter_mm'],
                    coil_axis='radial',coil_azimuth_deg=60*coil,passive_geometry_only=True,optical_field_unchanged=True)))
        result.append(SolidSpec(str(part['key']),f'winding_insulation_coil_{coil+1}',
            'insulator',transformed(insulation,rotation=rotation),(),
            'configured_geometry',note,dict(winding_geometry=d,winding_turn_count=d['turn_count_per_coil'],
                display_complexity_factor=4,coil_index=coil+1,coil_axis='radial',
                coil_azimuth_deg=60*coil,passive_geometry_only=True,optical_field_unchanged=True)))
    return tuple(result)
