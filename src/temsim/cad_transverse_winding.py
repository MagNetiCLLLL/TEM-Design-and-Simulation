"""Closed fine-wire CAD inside the saved transverse winding allocations."""
from __future__ import annotations

import math

from temsim.cad_geometry import SolidSpec, annulus, box, difference, sector, torus, transformed, union
from temsim.cad_probe_winding import rounded_rectangle_turn
from temsim.transverse_winding import transverse_winding_definition

EVIDENCE=('https://patents.google.com/patent/US5994704A/en',
          'https://patents.google.com/patent/US20090079531A1/en',
          'https://patents.google.com/patent/CN102509689B/en')
NOTE=('Explicit engineering closed round-wire reconstruction, not OEM CAD. '
      'The exact requested conductor count and metal area are retained. Enamel '
      'clearance and remaining insulation fit the original fixed winding pack. '
      'Closed turns omit unresolved lead routing. Magnetic material is not '
      'introduced by the winding CAD or the ideal vacuum current source.')


def _rotation(angle):
    c,s=math.cos(math.radians(angle)),math.sin(math.radians(angle))
    return [[c,-s,0.],[s,c,0.],[0.,0.,1.]]


def _radial_display_loop(x,width,height,delta,radius,centre,angle):
    return dict(kind='closed_wire_loop',path_family='rounded_rectangle',radial_x_mm=x,
        window_width_mm=width,window_height_mm=height,corner_offset_mm=delta,
        wire_radius_mm=radius,center_z_mm=centre,azimuth_deg=angle,
        analytic_shape=transformed(transformed(rounded_rectangle_turn(x,width,height,delta,radius),
            offset=(0.,0.,centre)),rotation=_rotation(angle)))


def _saddle_turn(coil, radial_offset, delta, wire_radius):
    r=coil['inner_radius_mm']+radial_offset
    phi=coil['window_half_angle_rad']+delta/r
    lower=coil['start_z_mm']+coil['band_mm']-delta
    upper=coil['end_z_mm']-coil['band_mm']+delta
    outer=r+2*wire_radius
    clearance=min(radial_offset-wire_radius,
                  coil['outer_radius_mm']-coil['inner_radius_mm']-radial_offset-wire_radius)
    if clearance<=0:
        raise ValueError('Saddle wire requires positive clearance inside its fixed radial layer')
    pieces=[]
    for sign in (-1,1):
        a=sign*phi
        pieces.append(transformed(annulus(lower,upper,0.,wire_radius),
            offset=(r*math.cos(a),r*math.sin(a),0.)))
    for z in (lower,upper):
        # Explicit endpoints avoid major-circle polygon sagitta separating an
        # otherwise thin wire arc from its straight side in the native mesh.
        pieces.append(dict(kind='torus_arc',major_radius_mm=r,minor_radius_mm=wire_radius,
                           center_z_mm=z,start_angle_deg=-math.degrees(phi),
                           end_angle_deg=math.degrees(phi),
                           display_chord_tolerance_mm=clearance/4))
    return dict(kind='closed_wire_loop',path_family='saddle',radius_mm=r,
        half_angle_rad=phi,lower_z_mm=lower,upper_z_mm=upper,
        wire_radius_mm=wire_radius,azimuth_deg=coil['azimuth_deg'],
        display_chord_tolerance_mm=clearance/4,
        analytic_shape=transformed(union(*pieces),rotation=_rotation(coil['azimuth_deg'])))


def transverse_bulk_shape(coil):
    """Exact original bulk winding volume, including its original window."""
    if coil['family']=='radial_rectangle':
        a,b=coil['radial_start_mm'],coil['radial_end_mm'];z=coil['center_z_mm']
        shape=difference(box(((a+b)/2,0.,z),(b-a,coil['outer_width_mm'],coil['outer_height_mm'])),
            box(((a+b)/2,0.,z),(2*(b-a)+1.,coil['window_width_mm'],coil['window_height_mm'])))
        return transformed(shape,rotation=_rotation(coil['azimuth_deg']))
    a,b=coil['start_z_mm'],coil['end_z_mm'];ri,ro=coil['inner_radius_mm'],coil['outer_radius_mm']
    span=coil['span_deg'];window=2*math.degrees(coil['window_half_angle_rad'])
    tolerance=(ro-ri)/1000
    base={**sector(a,b,ri,ro,coil['azimuth_deg'],span),'display_chord_tolerance_mm':tolerance}
    tool={**sector(a+coil['band_mm'],b-coil['band_mm'],max(0.,ri-(ro-ri)),ro+(ro-ri),
                  coil['azimuth_deg'],window),'display_chord_tolerance_mm':tolerance}
    return difference(base,tool)


def transverse_winding_solids(part,by_key=None):
    """All physical turns and complete residual insulation, or legacy bulk CAD."""
    definition=transverse_winding_definition(part,by_key)
    if definition is None or not definition['detailed_cad']:
        return None
    radius=definition['wire_diameter_mm']/2
    result=[]
    for coil,positions,pack in zip(definition['coils'],definition['turn_positions'],definition['packing']):
        wires=[]
        for radial,delta in positions:
            if coil['family']=='radial_rectangle':
                wire=_radial_display_loop(coil['radial_start_mm']+radial,
                    coil['window_width_mm'],coil['window_height_mm'],delta,radius,
                    coil['center_z_mm'],coil['azimuth_deg'])
            else:
                wire=_saddle_turn(coil,radial,delta,radius)
            wires.append(wire)
        common=dict(coil_index=coil['coil_index'],station_index=coil['station_index'],
            current_group=coil['group'],current_sign=coil['current_sign'],
            coil_azimuth_deg=coil['azimuth_deg'],wire_cross_section_mm2=definition['wire_cross_section_mm2'],
            wire_diameter_mm=definition['wire_diameter_mm'],
            wire_insulation_radial_thickness_mm=definition['insulation_thickness_mm'],
            fixed_pack_envelope=True,circuit_topology='closed_individual_turns_no_leads',
            magnetic_material_response=False)
        for index,wire in enumerate(wires):
            result.append(SolidSpec(str(part['key']),
                f"winding_turn_transverse_coil_{coil['coil_index']}_{index+1:06d}",
                'copper_winding',wire,EVIDENCE,'configured_geometry',NOTE,
                {**common,'winding_turn_index':index+1}))
        insulation={**difference(transverse_bulk_shape(coil),*wires),'batch_tools':True}
        result.append(SolidSpec(str(part['key']),
            f"winding_insulation_transverse_coil_{coil['coil_index']}",
            'insulator',insulation,EVIDENCE,'configured_geometry',NOTE,
            {**common,'winding_turn_count':definition['turn_count_per_coil'],
             'display_complexity_factor':1,'winding_geometry':pack}))
    return tuple(result)
