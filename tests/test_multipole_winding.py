"""Pure winding geometry, exact-turn CAD and unchanged boundary-role checks."""
from copy import deepcopy
import math

import numpy as np
import pytest

from temsim.cad_catalog import is_reconstruction_region, part_solids
from temsim.cad_geometry import transformed
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.component_representation import non_material_role, representation_note
from temsim.module_manifest import read_document
from temsim.multipole_winding import (
    AREA_FIELD,BASIS_FIELD,FIELDS,INSULATION_FIELD,NOTE_FIELD,SOURCE_NOTE,
    STATUS_FIELD,STATUS_VALUE,TURN_FIELD,area_field,electrical_channel_count,
    estimate_multipole_winding,is_primary_multipole,multipole_winding_definition,
    multipole_winding_envelopes,supported_multipole_winding,turn_field,runtime_basis_delta_rad,
)


@pytest.fixture(autouse=True)
def no_physical_source(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Winding CAD tests must not emit or propagate a physical source')
    monkeypatch.setattr('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden)
    monkeypatch.setattr('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden)


def part(key='probe_qpol_quadrupole',module='C3_ProbeCorrector'):
    p=next(p for p in read_document(f'configs/instruments/column/{module}.toml')['parts'] if p['key']==key)
    for name in FIELDS:
        p.pop(name,None)
    return p


def configured(p,turns=2,area=.02,basis=0.):
    p=deepcopy(p)
    p.update({turn_field(p):turns,INSULATION_FIELD:.005,BASIS_FIELD:basis,
              STATUS_FIELD:STATUS_VALUE,NOTE_FIELD:SOURCE_NOTE})
    if area is None:p.pop(area_field(p),None)
    else:p[area_field(p)]=area
    if turn_field(p)!=TURN_FIELD:
        p['hexapole_winding_geometry_status']='engineering_reconstruction_not_oem'
        p['hexapole_winding_source_note']='Explicit test-only known physical HP winding'
    return p


@pytest.mark.parametrize('key,module,order,channels,coils',(
    ('probe_hp1_hexapole','C3_ProbeCorrector',3,1,6),
    ('probe_hp2_hexapole','C3_ProbeCorrector',3,1,6),
    ('probe_qph1_quadrupole','C3_ProbeCorrector',2,2,8),
    ('probe_qph2_quadrupole','C3_ProbeCorrector',2,2,8),
    ('probe_qpc_quadrupole','C3_ProbeCorrector',2,2,8),
    ('probe_qpol_quadrupole','C3_ProbeCorrector',2,2,8),
    ('probe_hpc_hexapole','C3_ProbeCorrector',3,2,12),
    ('probe_hpol_hexapole','C3_ProbeCorrector',3,2,12),
    ('image_hp1_hexapole','C3_ImageCorrector',3,1,6),
    ('image_hp2_hexapole','C3_ImageCorrector',3,1,6),
    ('image_qpol_quadrupole','C3_ImageCorrector',2,2,8),
    ('image_hpol_hexapole','C3_ImageCorrector',3,2,12),
))
def test_scope_topology_and_alternating_loop_senses(key,module,order,channels,coils):
    p=part(key,module)
    assert supported_multipole_winding(p)
    assert multipole_winding_definition(p) is None
    d=multipole_winding_definition(configured(p))
    assert d['order']==order and d['electrical_channel_count']==channels
    assert electrical_channel_count(p)==channels and d['coil_count']==coils
    assert d['total_turn_count']==2*coils
    for group in 'xy'[:channels]:
        selected=[c for c in d['coils'] if c['group']==group]
        assert len(selected)==2*order
        assert [c['current_sign'] for c in selected]==[(-1)**i for i in range(2*order)]
        assert np.diff([c['azimuth_deg'] for c in selected]).tolist()==[180/order]*(2*order-1)
    assert p==part(key,module)


def test_auxiliary_groups_have_independent_half_cell_phases():
    for key,offset in (('probe_qpol_quadrupole',45.),('probe_hpol_hexapole',30.)):
        coils=multipole_winding_envelopes(part(key))
        first={c['group']:c for c in reversed(coils)}
        assert first['y']['azimuth_deg']-first['x']['azimuth_deg']==offset


def test_known_legacy_HP_N_does_not_activate_a_new_source_or_CAD():
    p=part('probe_hp1_hexapole')
    assert p['hexapole_winding_turn_count_per_coil']==2
    assert multipole_winding_definition(p) is None
    before=deepcopy(p)
    estimate=estimate_multipole_winding(p,.02)
    assert estimate['turn_count_per_coil']==2
    assert estimate['physical_probe_HP_count_preserved']
    assert p==before


def test_unknown_wire_area_keeps_exact_N_bulk_packs():
    p=configured(part(),turns=7,area=None)
    d=multipole_winding_definition(p)
    assert not d['detailed_cad'] and d['turn_count_per_coil']==7
    assert d['packing']==() and d['wire_diameter_mm'] is None
    solids=part_solids(p,{p['key']:p})
    assert len(solids)==8 and all(s.region.startswith('multipole_winding_bulk_coil_') for s in solids)


@pytest.mark.parametrize('changes',({TURN_FIELD:True},{TURN_FIELD:0},{TURN_FIELD:1.5},
    {AREA_FIELD:0.},{AREA_FIELD:float('nan')},{AREA_FIELD:True},
    {INSULATION_FIELD:-.1},{INSULATION_FIELD:True},{BASIS_FIELD:float('inf')},
    {STATUS_FIELD:'measured'},{NOTE_FIELD:''},{'multipole_winding_unknown':1}))
def test_invalid_recipe_rejected_without_mutation(changes):
    p=configured(part());p.update(changes);before=deepcopy(p)
    with pytest.raises(ValueError):multipole_winding_definition(p)
    assert repr(p)==repr(before)


def test_impossible_area_rejects_without_resize_or_turn_loss():
    p=configured(part(),turns=384,area=10.)
    before=deepcopy(p)
    with pytest.raises(ValueError,match='fit the fixed winding pack'):
        multipole_winding_definition(p)
    assert p==before


def test_probe_HP_cannot_save_a_second_divergent_turn_or_area_record():
    p=configured(part('probe_hp1_hexapole'))
    p[TURN_FIELD]=3
    with pytest.raises(ValueError,match='existing canonical'):
        multipole_winding_definition(p)
    p.pop(TURN_FIELD);p[AREA_FIELD]=.03
    with pytest.raises(ValueError,match='existing canonical'):
        multipole_winding_definition(p)


def test_replaced_CAD_keeps_recipe_inactive_until_native_geometry_restored():
    p=configured(part());p['model_3d']={'base':{'kind':'step'}}
    assert multipole_winding_definition(p) is None
    p['model_3d']={'base':{'kind':'existing'}}
    assert multipole_winding_definition(p)['total_turn_count']==16


@pytest.mark.parametrize('key',('probe_qpol_quadrupole','probe_hpol_hexapole'))
def test_every_closed_wire_and_residual_insulation_has_valid_mesh(key):
    p=configured(part(key))
    d=multipole_winding_definition(p);solids=part_solids(p,{p['key']:p})
    mesh=meshes_from_solids(solids,12)
    assert len(mesh)==len(solids)==d['total_turn_count']+d['coil_count']
    assert sum(s.region.startswith('winding_turn_multipole_') for s in solids)==d['total_turn_count']
    assert all(is_reconstruction_region(p,s.region) for s in solids)
    assert not is_reconstruction_region(p,'winding_turn_multipole_coil_1_000003')
    assert not is_reconstruction_region(p,'winding_insulation_multipole_coil_13')
    assert all(len(m.faces)>0 for m in mesh)


def test_new_winding_only_CAD_preserves_nonmaterial_transport_identity():
    p=part();before_role=non_material_role(p)
    assert before_role=='control_channel' and part_solids(p,{p['key']:p})==()
    p=configured(p)
    assert non_material_role(p)==before_role
    solids=part_solids(p,{p['key']:p})
    assert solids and {s.material for s in solids}=={'copper_winding','insulator'}
    assert all(s.parameters['transport_boundaries_unchanged'] for s in solids)
    assert 'winding-only' in representation_note(p)
    from temsim.part_model_3d import _part_meshes
    meshes,notes=_part_meshes(p,{p['key']:p},8)
    assert len(meshes)==len(solids) and notes


def test_primary_HP_passive_CAD_and_windings_share_saved_angular_datum():
    old=part('probe_hp1_hexapole')
    p=configured(old,basis=27.5)
    original=part_solids(old,{old['key']:old})
    current=part_solids(p,{p['key']:p})
    known={s.region:s for s in current}
    angle=math.radians(27.5);c,s=math.cos(angle),math.sin(angle)
    rotation=((c,-s,0.),(s,c,0.),(0.,0.,1.))
    for spec in original:
        if spec.material=='copper_winding' or spec.region.startswith('winding_insulation_coil_'):continue
        assert known[spec.region].shape==transformed(spec.shape,rotation=rotation)
    d=multipole_winding_definition(p)
    assert [coil['azimuth_deg'] for coil in d['coils']]==[27.5+60*i for i in range(6)]
    assert d['turn_count_per_coil']==2


def test_packing_preserves_enamel_and_all_fixed_sections():
    p=configured(part(),turns=384)
    d=multipole_winding_definition(p)
    wire_radius=d['wire_diameter_mm']/2+d['insulation_thickness_mm']
    for coil,pack in zip(d['coils'],d['packing']):
        radial=coil['radial_end_mm']-coil['radial_start_mm']
        band=min((coil['outer_width_mm']-coil['window_width_mm'])/2,
                 (coil['outer_height_mm']-coil['window_height_mm'])/2)
        assert len(pack['positions'])==384 and pack['pitch_mm']>=2*wire_radius
        assert all(wire_radius<=a<=radial-wire_radius and wire_radius<b<=band-wire_radius
            for a,b in pack['positions'])


@pytest.mark.parametrize('key',('probe_dp11_deflector','image_hp3_hexapole','energy_filter'))
def test_unnamed_sources_do_not_acquire_inferred_windings(key):
    p={'key':key}
    assert not supported_multipole_winding(p)
    assert multipole_winding_envelopes(p)==()


def test_runtime_primary_orientation_is_a_transient_delta_not_a_new_saved_datum():
    from types import SimpleNamespace
    from temsim.multipole_current import REFERENCE_ORIENTATION_FIELD
    p=configured(part('probe_hp1_hexapole'),basis=27.5)
    p[REFERENCE_ORIENTATION_FIELD]=-.04
    before=deepcopy(p)
    for runtime in ({'orientation_rad':-.02},SimpleNamespace(orientation_rad=-.02),-.02):
        assert runtime_basis_delta_rad(p,runtime)==pytest.approx(.02)
    assert runtime_basis_delta_rad(p,None)==0.
    assert runtime_basis_delta_rad(p,{})==0.
    assert p==before
    changed={**p,BASIS_FIELD:32.5}
    assert runtime_basis_delta_rad(changed,{'orientation_rad':-.02})==pytest.approx(.02)
    assert runtime_basis_delta_rad(configured(part()),{'orientation_rad':-.02})==0.
    old=deepcopy(p);old.pop(REFERENCE_ORIENTATION_FIELD)
    assert runtime_basis_delta_rad(old,{'orientation_rad':-.02})==0.


@pytest.mark.parametrize('orientation',(True,None,float('nan'),float('inf'),'0.1'))
def test_invalid_explicit_runtime_HP_orientation_fails(orientation):
    from temsim.multipole_current import REFERENCE_ORIENTATION_FIELD
    p=configured(part('probe_hp1_hexapole'));p[REFERENCE_ORIENTATION_FIELD]=0.
    with pytest.raises(ValueError):
        runtime_basis_delta_rad(p,{'orientation_rad':orientation})


@pytest.mark.parametrize('key,module',(('probe_hp1_hexapole','C3_ProbeCorrector'),
                                     ('image_hp1_hexapole','C3_ImageCorrector')))
def test_complete_primary_CAD_rotates_with_runtime_native_orientation(key,module):
    from temsim.multipole_current import REFERENCE_ORIENTATION_FIELD
    p=configured(part(key,module),basis=27.5);p[REFERENCE_ORIENTATION_FIELD]=-.04
    before=deepcopy(p);rows={p['key']:p}
    original=part_solids(p,rows,runtime={'orientation_rad':-.04})
    assert original==part_solids(p,rows)
    delta=.017
    moved=part_solids(p,rows,runtime={'orientation_rad':-.04+delta})
    assert len(moved)==len(original)
    rotation=((math.cos(delta),-math.sin(delta),0.),(math.sin(delta),math.cos(delta),0.),(0.,0.,1.))
    for base,actual in zip(original,moved):
        assert actual.region==base.region and actual.material==base.material
        assert actual.shape==transformed(base.shape,rotation=rotation)
        assert actual.parameters['runtime_basis_delta_rad']==pytest.approx(delta)
    assert p==before


def test_part3D_runtime_values_apply_the_same_whole_HP_covariance():
    from scipy.spatial import cKDTree
    from temsim.part_model_3d import part_model_from_document
    from temsim.multipole_current import REFERENCE_ORIENTATION_FIELD
    p=configured(part('probe_hp1_hexapole'),basis=27.5);p[REFERENCE_ORIENTATION_FIELD]=-.04
    document={'module':{'key':'test'},'parts':[p]}
    baseline=part_model_from_document(document,p['key'],angular_segments=8,include_children=False,
        runtime_values={p['key']:{'orientation_rad':-.04}})
    delta=.021
    moved=part_model_from_document(document,p['key'],angular_segments=8,include_children=False,
        runtime_values={p['key']:{'orientation_rad':-.04+delta}})
    rotation=np.array(((math.cos(delta),-math.sin(delta),0.),(math.sin(delta),math.cos(delta),0.),(0.,0.,1.)))
    assert len(baseline.meshes)==len(moved.meshes)
    for original,actual in zip(baseline.meshes,moved.meshes):
        expected=original.vertices@rotation.T
        assert len(expected)==len(actual.vertices)
        assert np.max(cKDTree(expected).query(actual.vertices)[0])<1e-8


def test_winding_only_auxiliary_CAD_is_included_in_module_export_without_new_boundary_role():
    from temsim.cad_catalog import module_catalog, posed_solids
    p=configured(part())
    role=non_material_role(p)
    solids=posed_solids(p,{p['key']:p})
    assert len(solids)==24 and all(s.material in {'copper_winding','insulator'} for s in solids)
    assert non_material_role(p)==role=='control_channel'
    catalog=module_catalog({'module':{'key':'test','name':'Test'},'parts':[p]})
    assert len(catalog['solids'])==24


@pytest.mark.parametrize('module',('C3_ImageCorrector','C3_BCOR',
                                 'C3_ProbeCorrector_ImageCorrector','C3_ProbeCorrector_BCOR'))
def test_colocated_Image_QPol_HPol_have_disjoint_complete_radial_packs(module):
    q=configured(part('image_qpol_quadrupole',module),turns=63)
    h=configured(part('image_hpol_hexapole',module),turns=63)
    qc=multipole_winding_envelopes(q);hc=multipole_winding_envelopes(h)
    maximum_q_radius=max(math.hypot(c['radial_end_mm'],c['outer_width_mm']/2) for c in qc)
    minimum_h_radius=min(c['radial_start_mm'] for c in hc)
    assert minimum_h_radius-maximum_q_radius==pytest.approx(.8876930002072463)
    assert q['local_start_z_mm']==h['local_start_z_mm'] and q['local_end_z_mm']==h['local_end_z_mm']
    for p in (q,h):
        assert estimate_multipole_winding(p)['turn_count_per_coil']==63
        d=multipole_winding_definition(p)
        assert d['turn_count_per_coil']==63


def test_colocated_Image_first_metal_turns_no_longer_intersect():
    from temsim.cad_multipole_winding import multipole_winding_solids
    q=configured(part('image_qpol_quadrupole','C3_ImageCorrector'),turns=63)
    h=configured(part('image_hpol_hexapole','C3_ImageCorrector'),turns=63)
    qm=_manifold(multipole_winding_solids(q)[0].shape,16)
    hm=_manifold(multipole_winding_solids(h)[0].shape,16)
    assert qm.volume()>0 and hm.volume()>0
    assert (qm^hm).volume()==0.
