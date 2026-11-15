"""Exact-count winding-only CAD for explicitly reconstructed multipoles."""
from dataclasses import replace
import math

from temsim.cad_geometry import SolidSpec, difference, transformed
from temsim.cad_transverse_winding import _radial_display_loop, transverse_bulk_shape
from temsim.multipole_winding import multipole_winding_definition, runtime_basis_delta_rad


NOTE = ('Engineering closed radial winding reconstruction, not measured or OEM '
    'construction. Exact saved turns and conductor metal area are represented; '
    'enamel clearances and residual insulation remain inside the original pack. '
    'Auxiliary winding-only CAD adds no independent opaque cartridge or beam '
    'boundary. Core/pole material response and lead routing are excluded from '
    'the ideal vacuum source. The angular datum is shared by CAD and source.')
EVIDENCE = ('https://www.jeol.com/words/semterms/20201020.111014.php',
    'https://patents.google.com/patent/CN102509689B/en',
    'https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf')


def multipole_winding_solids(part,by_key=None):
    """Return all closed turns/insulation, or retained bulk packs for unknown A."""
    definition=multipole_winding_definition(part,by_key)
    if definition is None:
        return None
    result=[]
    for index,coil in enumerate(definition['coils']):
        common=dict(coil_index=coil['coil_index'],current_group=coil['group'],
            current_sign=coil['current_sign'],coil_azimuth_deg=coil['azimuth_deg'],
            multipole_order=definition['order'],coil_axis='radial',
            fixed_pack_envelope=True,transport_boundaries_unchanged=True,
            circuit_topology='closed_individual_turns_no_leads',
            wire_cross_section_mm2=definition['wire_cross_section_mm2'],
            wire_diameter_mm=definition['wire_diameter_mm'],
            wire_insulation_radial_thickness_mm=definition['insulation_thickness_mm'])
        if not definition['detailed_cad']:
            result.append(SolidSpec(str(part['key']),f'multipole_winding_bulk_coil_{coil["coil_index"]}',
                'copper_winding',transverse_bulk_shape(coil),EVIDENCE,'engineering_reconstruction',
                NOTE+' Conductor size is unknown; the known N still defines the electrical source.',
                {**common,'winding_turn_count':definition['turn_count_per_coil']}))
            continue
        wires=[]
        radius=definition['wire_diameter_mm']/2
        for radial,delta in definition['turn_positions'][index]:
            wires.append(_radial_display_loop(coil['radial_start_mm']+radial,
                coil['window_width_mm'],coil['window_height_mm'],delta,radius,
                coil['center_z_mm'],coil['azimuth_deg']))
        for turn,wire in enumerate(wires,1):
            result.append(SolidSpec(str(part['key']),
                f'winding_turn_multipole_coil_{coil["coil_index"]}_{turn:06d}',
                'copper_winding',wire,EVIDENCE,'configured_geometry',NOTE,
                {**common,'winding_turn_index':turn}))
        insulation={**difference(transverse_bulk_shape(coil),*wires),'batch_tools':True}
        result.append(SolidSpec(str(part['key']),
            f'winding_insulation_multipole_coil_{coil["coil_index"]}',
            'insulator',insulation,EVIDENCE,'configured_geometry',NOTE,
            {**common,'winding_turn_count':definition['turn_count_per_coil'],
                'display_complexity_factor':1,'winding_geometry':definition['packing'][index]}))
    return tuple(result)


def replace_primary_windings(part,original,by_key=None,*,runtime=None):
    """Keep the existing HP passive geometry and apply its explicit datum.

    The original pin, former and yoke CAD rotate with the winding pack. Their
    radial/axial envelopes and existing transport boundary ownership remain
    unchanged. No extra primary HP coil group is created.
    """
    definition=multipole_winding_definition(part,by_key)
    if definition is None:
        return tuple(original)
    angle=definition['basis_angle_deg']
    c,s=math.cos(math.radians(angle)),math.sin(math.radians(angle))
    rotation=((c,-s,0.),(s,c,0.),(0.,0.,1.))
    passive=[]
    for solid in original:
        if solid.material=='copper_winding' or solid.region.startswith('winding_insulation_coil_'):
            continue
        passive.append(replace(solid,
            shape=transformed(solid.shape,rotation=rotation) if angle else solid.shape,
            parameters={**solid.parameters,'multipole_winding_basis_angle_deg':angle,
                'transport_boundaries_unchanged':True},
            note=solid.note+' Explicit engineering angular datum shared by the complete HP CAD and ideal source.'))
    values=tuple(passive)+multipole_winding_solids(part,by_key)
    delta=runtime_basis_delta_rad(part,runtime)
    if delta:
        c,s=math.cos(delta),math.sin(delta)
        rotation=((c,-s,0.),(s,c,0.),(0.,0.,1.))
        values=tuple(replace(solid,shape=transformed(solid.shape,rotation=rotation),
            parameters={**solid.parameters,'runtime_basis_delta_rad':delta},
            note=solid.note+' The complete primary HP CAD follows its transient native installation orientation.')
            for solid in values)
    return values
