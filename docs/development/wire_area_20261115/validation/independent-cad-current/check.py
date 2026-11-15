"""Current-default Objective illustrative CAD inputs and independent geometry checks only.

No default wire specification is assigned. No magnetic field, particle source,
transport, focus calculation or coherent propagation is called.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from temsim.cad_round_winding import resolve_round_winding_geometry, round_winding_solids
from temsim.module_manifest import read_document

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents
            if (parent/'pyproject.toml').is_file() and (parent/'src/temsim').is_dir())
HERE = ROOT/'outputs/wire-area-20261115/independent-cad-current-replay'
DEFAULT = ROOT/'configs/instruments/column/C3_ProbeCorrector.toml'
KEY = 'objective_lens_excitation_coil'
AREA = 'winding_wire_cross_section_mm2'


def independent_geometry_check(geometry, declared_area, regional_counts):
    """Integrate actual circles/tori, rather than trusting the layout witness."""
    layout = geometry.layout
    radius = layout.wire_diameter_mm/2
    guard = 8*np.finfo(float).eps*max(abs(turn.z_mm) for turn in layout.turns)
    actual_wire_area = math.pi*radius*radius
    np.testing.assert_allclose(actual_wire_area, declared_area, rtol=2e-15, atol=0)
    area, volume = 0., 0.
    minimum_pair_gap = math.inf
    minimum_radial_boundary = math.inf
    minimum_axial_boundary = math.inf
    per_region = []
    for index, allocation in enumerate(geometry.conductor_allocations):
        positions = np.array([(turn.radius_mm,turn.z_mm) for turn in layout.turns
                              if turn.interval_index==index])
        assert len(positions)==regional_counts[index]
        width = allocation['outer_mm']-allocation['inner_mm']
        length = allocation['end_mm']-allocation['start_mm']
        area += width*length
        volume += math.pi*(allocation['outer_mm']**2-allocation['inner_mm']**2)*length
        radial_clearances = np.concatenate((positions[:,0]-radius-allocation['inner_mm'],
                                            allocation['outer_mm']-positions[:,0]-radius))
        axial_clearances = np.concatenate((positions[:,1]-radius-allocation['start_mm'],
                                           allocation['end_mm']-positions[:,1]-radius))
        assert np.min(radial_clearances)>=-guard
        assert np.min(axial_clearances)>=-guard
        minimum_radial_boundary=min(minimum_radial_boundary,float(np.min(radial_clearances)))
        minimum_axial_boundary=min(minimum_axial_boundary,float(np.min(axial_clearances)))
        distance=np.linalg.norm(positions[:,None,:]-positions[None,:,:],axis=-1)
        np.fill_diagonal(distance,math.inf)
        pair_gap=float(np.min(distance))-2*radius
        assert pair_gap>0, 'This illustrative winding must have positive interwire insulation gaps'
        minimum_pair_gap=min(minimum_pair_gap,pair_gap)
        metal_area=len(positions)*actual_wire_area
        metal_volume=math.fsum(2*math.pi*r*actual_wire_area for r in positions[:,0])
        remaining_area=width*length-metal_area
        remaining_volume=math.pi*(allocation['outer_mm']**2-allocation['inner_mm']**2)*length-metal_volume
        assert remaining_area>0 and remaining_volume>0
        per_region.append(dict(index=index,turn_count=len(positions),conductor_area_mm2=metal_area,
                               remaining_insulation_area_mm2=remaining_area,
                               conductor_volume_mm3=metal_volume,remaining_insulation_volume_mm3=remaining_volume,
                               minimum_interwire_gap_mm=pair_gap))
    metal_area=actual_wire_area*len(layout.turns)
    metal_volume=math.fsum(2*math.pi*turn.radius_mm*actual_wire_area for turn in layout.turns)
    np.testing.assert_allclose([area,volume,metal_area,metal_volume],
        [layout.area.cross_section_area_mm2,layout.area.allocation_volume_mm3,
         layout.area.conductor_cross_section_area_mm2,layout.area.conductor_volume_mm3],rtol=2e-14,atol=0)
    np.testing.assert_allclose([metal_area+layout.area.insulation_cross_section_area_mm2,
                               metal_volume+layout.area.insulation_volume_mm3],[area,volume],rtol=2e-14,atol=0)
    field=geometry.field_envelope
    field_inner,field_outer=field['inner_diameter_mm']/2,field['outer_diameter_mm']/2
    field_length=math.fsum(b-a for a,b in field['intervals_mm'])
    field_area=(field_outer-field_inner)*field_length
    field_volume=math.pi*(field_outer**2-field_inner**2)*field_length
    inside_passive_area,inside_passive_volume=0.,0.
    for spec in geometry.passive_solids:
        section=spec.shape.get('section_zr_mm',())
        if not section:
            continue
        z0,z1=min(p[0] for p in section),max(p[0] for p in section)
        r0,r1=min(p[1] for p in section),max(p[1] for p in section)
        assert set(tuple(p) for p in section)=={(z0,r0),(z0,r1),(z1,r0),(z1,r1)}, 'Audit requires rectangular passive sections'
        for start,end in field['intervals_mm']:
            lower,upper=max(start,z0),min(end,z1)
            inner,outer=max(field_inner,r0),min(field_outer,r1)
            if lower<upper and inner<outer:
                inside_passive_area+=(outer-inner)*(upper-lower)
                inside_passive_volume+=math.pi*(outer**2-inner**2)*(upper-lower)
        for allocation in geometry.conductor_allocations:
            assert (z1<=allocation['start_mm'] or z0>=allocation['end_mm']
                    or r1<=allocation['inner_mm'] or r0>=allocation['outer_mm'])
    np.testing.assert_allclose([area+inside_passive_area,volume+inside_passive_volume],
                               [field_area,field_volume],rtol=2e-14,atol=0)
    return dict(declared_metal_area_per_turn_mm2=declared_area,derived_wire_diameter_mm=2*radius,
        total_turns=len(layout.turns),fixed_interval_counts=list(regional_counts),
        field_envelope=geometry.field_envelope,conductor_allocations=list(geometry.conductor_allocations),
        passive_regions=[spec.region for spec in geometry.passive_solids],
        total_allocated_area_mm2=area,total_allocated_volume_mm3=volume,
        total_conductor_area_mm2=metal_area,total_conductor_volume_mm3=metal_volume,
        remaining_insulation_area_mm2=area-metal_area,remaining_insulation_volume_mm3=volume-metal_volume,
        original_field_envelope_area_mm2=field_area,original_field_envelope_volume_mm3=field_volume,
        retained_passive_area_inside_field_pack_mm2=inside_passive_area,
        retained_passive_volume_inside_field_pack_mm3=inside_passive_volume,
        complete_field_pack_partition_closure=True,
        minimum_interwire_gap_mm=minimum_pair_gap,
        minimum_radial_boundary_clearance_mm=minimum_radial_boundary,
        minimum_axial_boundary_clearance_mm=minimum_axial_boundary,
        boundary_clearance_guard_mm=guard,
        boundary_policy='Nonnegative within binary64 coordinate guard; tangency to retained end insulation is allowed',
        independent_circle_no_overlap=True,independent_area_volume_closure=True,regions=per_region)


def plot_upper(geometries,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Patch,Polygon,Rectangle
    orange='#e99b31';blue='#c2ddf3';grey='#a8adb5'
    fig,axes=plt.subplots(1,2,figsize=(8.3,14),constrained_layout=True)
    for ax,geometry,record in zip(axes,geometries,records):
        field=geometry.field_envelope
        start,end=field['intervals_mm'][0]
        inner,outer=field['inner_diameter_mm']/2,field['outer_diameter_mm']/2
        allocation=geometry.conductor_allocations[0]
        ax.add_patch(Rectangle((allocation['inner_mm'],allocation['start_mm']-start),
            allocation['outer_mm']-allocation['inner_mm'],allocation['end_mm']-allocation['start_mm'],
            facecolor=blue,edgecolor='none'))
        for spec in geometry.passive_solids:
            section=spec.shape.get('section_zr_mm',())
            if section:
                ax.add_patch(Polygon([(r,z-start) for z,r in section],facecolor=grey,
                                     edgecolor='#5f636a',linewidth=.4,closed=True))
        radius=geometry.layout.wire_diameter_mm/2
        for turn in geometry.layout.turns:
            if turn.interval_index==0:
                ax.add_patch(Circle((turn.radius_mm,turn.z_mm-start),radius,
                                   facecolor=orange,edgecolor='#b16a15',linewidth=.25))
        ax.add_patch(Rectangle((inner,0),outer-inner,end-start,fill=False,edgecolor='black',linewidth=.9))
        ax.set(xlim=(inner-1,outer+1),ylim=(-2,end-start+2),xlabel='Radius R (mm)',
               ylabel='Z from upper winding start (mm)',
               title=f"{record['fixed_interval_counts'][0]} turns | metal area {record['declared_metal_area_per_turn_mm2']:g} mm²/turn\n"
                     f"Derived diameter {geometry.layout.wire_diameter_mm:.4f} mm\n"
                     f"Minimum interwire gap: {record['minimum_interwire_gap_mm']:.4f} mm")
        ax.invert_yaxis();ax.set_aspect('equal',adjustable='box')
        ax.tick_params(labelsize=8)
    fig.suptitle('Illustrative CAD inputs; not default wire specification\n'
                 'Objective upper winding: fixed ID 60 mm, OD 132 mm, occupied length 145 mm',fontsize=12)
    fig.legend(handles=[Patch(facecolor=orange,label='Conductor metal'),Patch(facecolor=blue,label='Automatic remaining insulation'),
                        Patch(facecolor=grey,label='Retained passive layers'),Patch(facecolor='none',edgecolor='black',label='Fixed field envelope')],
               loc='outside lower center',ncols=2,fontsize=8)
    fig.savefig(HERE/'objective-upper-metal-area-comparison.png',dpi=180)
    plt.close(fig)


def run():
    HERE.mkdir(parents=True,exist_ok=True)
    source_bytes=DEFAULT.read_bytes()
    parts={p['key']:p for p in read_document(DEFAULT)['parts']}
    original=deepcopy(parts[KEY])
    assert original['winding_turn_count']==1160
    assert tuple(original['winding_interval_turn_counts'])==(580,580)
    records=[];geometries=[]
    for area,counts in ((1.,(580,580)),(2.,(580,580)),(2.,(500,500))):
        row=deepcopy(original);row.pop('winding_wire_diameter_mm',None)
        row[AREA]=area;row['winding_turn_count']=sum(counts);row['winding_interval_turn_counts']=list(counts)
        geometry=resolve_round_winding_geometry(row,parts)
        record=independent_geometry_check(geometry,area,counts)
        specs=round_winding_solids(row,parts)
        assert len([s for s in specs if s.shape.get('kind')=='torus'])==sum(counts)
        assert len([s for s in specs if s.shape.get('kind')=='round_winding_insulation'])==2
        records.append(record);geometries.append(geometry)
    for other in geometries[1:]:
        assert other.field_envelope==geometries[0].field_envelope
        assert other.conductor_allocations==geometries[0].conductor_allocations
        assert other.passive_solids==geometries[0].passive_solids
    row=deepcopy(original);row.pop('winding_wire_diameter_mm',None);row[AREA]=9.
    try:
        resolve_round_winding_geometry(row,parts)
    except ValueError as error:
        rejected=str(error)
    else:
        raise AssertionError('9 mm² metal area must not fit the fixed 580+580-turn envelope')
    plot_upper(geometries[:2],records[:2])
    current={p['key']:p for p in read_document(DEFAULT)['parts']}[KEY]
    assert current==original,'The actual default Objective winding definition changed during the CAD audit'
    result=dict(scope='Independent illustrative mechanical CAD only; no default wire area, field or particle calculation',
        default_definition=str(DEFAULT.relative_to(ROOT)),default_file_sha256_at_read=sha256(source_bytes).hexdigest(),
        actual_default_objective_unchanged=True,default_wire_area_assigned=False,
        original_interval_turn_counts=[580,580],all_envelopes_allocations_passives_unchanged=True,
        impossible_9_mm2_input_rejected=True,impossible_input_message=rejected,
        cases=records,figure='objective-upper-metal-area-comparison.png')
    (HERE/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(records),fixed_field_envelope=geometries[0].field_envelope,
                          minimum_interwire_gaps_mm=[r['minimum_interwire_gap_mm'] for r in records],
                          impossible_9_mm2_input_rejected=True,actual_default_objective_unchanged=True),indent=2))


def forbidden(*args,**kwargs):
    raise AssertionError('This mechanical CAD example must not calculate fields, emit or transport particles')


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden), \
         patch('temsim.physics.lens_field_provider.resolve_runtime_lens_field_provider',forbidden), \
         patch('temsim.physics.potential_capture.capture_round_lens_potential',forbidden):
        run()
