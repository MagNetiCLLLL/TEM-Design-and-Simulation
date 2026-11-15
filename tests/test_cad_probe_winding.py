"""Explicit radial probe winding CAD without field/source calculations."""
from collections import Counter
from copy import deepcopy
import math

import numpy as np
import pytest

from temsim.cad_geometry import SolidSpec, box, difference
from temsim.cad_mesh import _check_winding_mesh_budget, _manifold, meshes_from_solids
from temsim.cad_multipoles import multipole_solids
from temsim.cad_probe_winding import (
    AREA_FIELD, FIELDS, MAXIMUM_TOTAL_TURNS, NOTE_FIELD, STATUS_FIELD, TURN_FIELD,
    probe_winding_dimensions, probe_winding_solids, rounded_rectangle_turn,
)
from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from legacy_transverse_helpers import legacy_transverse_document


@pytest.fixture(scope='module')
def parts():
    doc=legacy_transverse_document(read_document(INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml'))
    return {p['key']:p for p in doc['parts']}


def detailed(part, count=3, area=.05):
    return {**part, TURN_FIELD:count, AREA_FIELD:area,
            STATUS_FIELD:'engineering_reconstruction_not_oem',
            NOTE_FIELD:'Explicit test-only CAD input; no measured winding or current calibration.'}


def test_legacy_or_metadata_only_has_no_guessed_turns_or_wire(parts):
    p=parts['probe_hp1_hexapole'];before=deepcopy(p)
    assert probe_winding_dimensions(p) is None
    assert probe_winding_solids(p) is None
    assert probe_winding_dimensions({**p,STATUS_FIELD:'engineering_reconstruction_not_oem'}) is None
    assert p==before


@pytest.mark.parametrize('key',['probe_hp1_hexapole','probe_hp2_hexapole'])
def test_packing_is_radial_fixed_envelope_and_has_analytic_volume_witness(parts,key):
    p=detailed(parts[key]);before=deepcopy(p);d=probe_winding_dimensions(p)
    assert d['turn_count_per_coil']==3 and d['total_turn_count']==18
    assert d['coil_axis']=='radial' and d['pole_pitch_deg']==60
    assert d['wire_cross_section_mm2']==.05
    assert d['wire_diameter_mm']==pytest.approx(2*math.sqrt(.05/math.pi))
    r=d['wire_diameter_mm']/2
    for position in d['turn_positions']:
        assert d['radial_start_mm'] <= position['radial_x_mm']-r
        assert position['radial_x_mm']+r <= d['radial_end_mm']
        assert position['corner_offset_mm']>r
        assert position['corner_offset_mm']+r <= d['minimum_band_mm']
    expected=.05*sum(2*(d['window_width_mm']+d['window_height_mm'])+2*math.pi*t['corner_offset_mm']
                     for t in d['turn_positions'])
    assert d['conductor_volume_per_coil_mm3']==pytest.approx(expected,rel=2e-15)
    assert d['conductor_volume_per_coil_mm3']+d['remaining_insulation_volume_per_coil_mm3']==d['bulk_coil_volume_mm3']
    assert d['optical_field_unchanged'] and not d['field_or_current_calibration']
    assert p==before


def test_area_or_turn_edit_repacks_only_the_original_bulk_frame(parts):
    p=parts['probe_hp1_hexapole']
    first,more,bigger=(probe_winding_dimensions(detailed(p,count,area)) for count,area in ((3,.05),(6,.05),(3,.1)))
    for name in ('radial_start_mm','radial_end_mm','window_width_mm','window_height_mm','coil_width_mm','coil_height_mm','bulk_coil_volume_mm3'):
        assert first[name]==more[name]==bigger[name]
    assert more['conductor_volume_per_coil_mm3']>first['conductor_volume_per_coil_mm3']
    assert bigger['conductor_volume_per_coil_mm3']>first['conductor_volume_per_coil_mm3']
    assert bigger['remaining_insulation_volume_per_coil_mm3']<first['remaining_insulation_volume_per_coil_mm3']


@pytest.mark.parametrize('updates',[
    {TURN_FIELD:True},{TURN_FIELD:0},{TURN_FIELD:-1},{TURN_FIELD:1.5},{TURN_FIELD:MAXIMUM_TOTAL_TURNS},
    {AREA_FIELD:True},{AREA_FIELD:0.},{AREA_FIELD:-1.},{AREA_FIELD:float('nan')},{AREA_FIELD:float('inf')},
    {AREA_FIELD:100.},{STATUS_FIELD:'measured'},{NOTE_FIELD:''},
    {'hexapole_winding_unknown_field':1},
])
def test_invalid_or_unfit_inputs_reject_without_mutating_sources(parts,updates):
    p=detailed(parts['probe_hp1_hexapole']);p.update(updates);before=deepcopy(p)
    with pytest.raises(ValueError):probe_winding_dimensions(p)
    assert p==before


@pytest.mark.parametrize('field',[TURN_FIELD,STATUS_FIELD,NOTE_FIELD])
def test_partial_recipe_cannot_guess_the_missing_definition(parts,field):
    p=detailed(parts['probe_hp1_hexapole']);p.pop(field)
    with pytest.raises(ValueError):probe_winding_dimensions(p)


def test_known_turn_count_retains_bulk_cad_until_metal_area_is_supplied(parts):
    p=detailed(parts['probe_hp1_hexapole']);p.pop(AREA_FIELD)
    before=deepcopy(p)
    assert probe_winding_dimensions(p) is None
    assert probe_winding_solids(p) is None
    assert p[TURN_FIELD]==3 and p==before


def test_replacement_cad_disables_but_restoring_native_revalidates_recipe(parts):
    p=detailed(parts['probe_hp1_hexapole'],count=0)
    p['model_3d']={'base':{'kind':'box'}}
    assert probe_winding_dimensions(p) is None
    p['model_3d']={'base':{'kind':'existing'}}
    with pytest.raises(ValueError):probe_winding_dimensions(p)


def test_nonhexapole_cannot_receive_a_radial_winding_recipe(parts):
    p=detailed(parts['probe_tl22_lens'])
    with pytest.raises(ValueError,match='HP1/HP2'):probe_winding_dimensions(p)


def test_one_rounded_rectangular_turn_is_connected_closed_and_radial():
    shape=rounded_rectangle_turn(0.,6.885,8.688,.7,.15)
    spec=SolidSpec('wire','winding_turn_test','copper_winding',shape)
    body=_manifold(shape,32)
    assert len(body.decompose())==1 and body.volume()>0
    bounds=body.bounding_box()
    assert bounds[0]==pytest.approx(-.15) and bounds[3]==pytest.approx(.15)
    assert bounds[4]-bounds[1]>8 and bounds[5]-bounds[2]>10
    mesh,=meshes_from_solids((spec,),angular_segments=32)
    edges=Counter(tuple(sorted((int(a),int(b)))) for face in mesh.faces for a,b in zip(face,np.roll(face,-1)))
    assert all(count==2 for count in edges.values())


@pytest.mark.parametrize('key',['probe_hp1_hexapole','probe_hp2_hexapole'])
def test_all_turns_and_remaining_insulation_partition_the_six_original_coils(parts,key):
    source=parts[key];p=detailed(source,count=2)
    old=multipole_solids(source,parts);current=multipole_solids(p,{**parts,key:p})
    old_coils=[s for s in old if s.region.startswith('coil_') and s.material=='copper_winding']
    assert len(old_coils)==6
    new=probe_winding_solids(p)
    assert len(new)==18
    wires=[s for s in new if s.material=='copper_winding'];insulators=[s for s in new if s.material=='insulator']
    assert len(wires)==12 and len(insulators)==6
    by_region={s.region:s for s in current}
    for old_spec in old:
        if old_spec not in old_coils:
            assert by_region[old_spec.region]==old_spec
    assert not any(s.region in {o.region for o in old_coils} for s in current)
    bodies={s.region:_manifold(s.shape,24) for s in new}
    for index,old_spec in enumerate(old_coils,1):
        original=_manifold(old_spec.shape,24)
        members=[bodies[s.region] for s in new if s.parameters['coil_index']==index]
        assert sum(b.volume() for b in members)==pytest.approx(original.volume(),rel=3e-9,abs=1e-8)
        for body in members:
            assert (body-original).volume()<1e-8
        for i,body in enumerate(members):
            # Internal wire cavities are negative-oriented closed shells in
            # Manifold.decompose(), not disconnected insulation material.
            if i<2:assert len(body.decompose())==1
            assert body.volume()>0 and str(body.status())=='Error.NoError'
            for other in members[i+1:]:assert (body^other).volume()<1e-8
    volumes=[bodies[s.region].volume() for s in wires]
    np.testing.assert_allclose(volumes[::2],volumes[0],rtol=1e-9,atol=1e-8)
    assert all(s.parameters['coil_axis']=='radial' and s.parameters['optical_field_unchanged'] for s in new)


def test_display_budget_counts_radial_path_complexity_without_omitting_turns():
    ordinary=SolidSpec('coil','winding_insulation_body','insulator',{},parameters={'winding_turn_count':2048})
    _check_winding_mesh_budget((ordinary,),32)
    radial=SolidSpec('coil','winding_insulation_body','insulator',{},parameters={'winding_turn_count':2048,'display_complexity_factor':4})
    with pytest.raises(ValueError,match='No turns were omitted'):_check_winding_mesh_budget((radial,),32)


@pytest.mark.parametrize('factor',[0,-1,1.5,True])
def test_display_complexity_requires_a_positive_integer(factor):
    radial=SolidSpec('coil','winding_insulation_body','insulator',{},parameters={'winding_turn_count':1,'display_complexity_factor':factor})
    with pytest.raises(ValueError,match='integers'):_check_winding_mesh_budget((radial,),32)


def test_step_guard_counts_physical_radial_turns_before_constructing_brep(monkeypatch):
    from temsim import cad_step
    row=dict(key='coil',region='winding_insulation_coil_1',shape=box((0,0,0),(1,1,1)),
             parameters={'winding_turn_count':8193,'display_complexity_factor':4})
    monkeypatch.setattr(cad_step,'build_shape',lambda *args,**kwargs:pytest.fail('Budget must precede BRep'))
    with pytest.raises(ValueError,match='8193|8,193'):cad_step.build_solid_rows((row,))
