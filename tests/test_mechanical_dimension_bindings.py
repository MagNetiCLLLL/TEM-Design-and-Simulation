"""Public dimension edits keep native and CAD material relationships coherent."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.cad_catalog import posed_solids
from temsim.cad_mesh import _manifold
from temsim.part_model_3d import part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


COLUMN = INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml'
PROJECTION = INSTRUMENT_CONFIG_ROOT/'project_and_recording_system/NoEnergyFilter.toml'
GAPS = [(COLUMN, key) for key in (
    'condenser_lens_3', 'adapter_lens', 'probe_tl22_lens', 'probe_tl21_lens',
    'probe_tl12_lens', 'mini_condenser')]
GAPS += [(PROJECTION, key) for key in (
    'diffraction_lens', 'intermediate_lens', 'projector_lens_1', 'projector_lens_2')]


def _material(draft, key):
    rows = {p['key']: p for p in draft.document['parts']}
    solids = posed_solids(draft.part(key), rows)
    meshes = part_model_from_document(draft.document, key, include_children=False,
                                      angular_segments=32).meshes
    native = {mesh.region: mesh for mesh in meshes if not mesh.wireframe}
    assert len(native) == len(solids)
    for spec in solids:
        body = _manifold(spec.shape, 32)
        mesh = native[spec.region]
        np.testing.assert_allclose(np.array(body.bounding_box()).reshape(2, 3),
            [mesh.vertices.min(0), mesh.vertices.max(0)], atol=1e-7)
        assert body.volume() > 0
    return {s.region: s.shape for s in solids}


def _assert_changed_only(before, after, allowed):
    old = {p['key']: p for p in before['parts']}
    new = {p['key']: p for p in after['parts']}
    assert old.keys() == new.keys()
    for key in old:
        for field in old[key].keys() | new[key].keys():
            if old[key].get(field) != new[key].get(field):
                assert field in allowed.get(key, ()), (key, field)


def _assert_saved_material_document(reopened, edited):
    # Save copy deliberately embeds owned subassemblies as independent rows.
    assert reopened == {k: v for k, v in edited.items() if k != 'subassemblies'}


@pytest.mark.parametrize('path,key', GAPS)
def test_pole_gap_moves_facing_material_only_and_survives_undo_save(path, key, tmp_path):
    draft = PartModelDocument(path)
    before = deepcopy(draft.document)
    poles = sorted((p for p in draft.document['parts'] if p.get('parent_key') == key
                    and p.get('mechanical_profile') == 'magnetic_pole_piece'),
                   key=lambda p: p['local_start_z_mm'])
    upper, lower = [p['key'] for p in poles]
    old_shapes = {p: _material(draft, p) for p in (upper, lower)}
    gap = draft.part(key)['pole_gap_mm']*1.01
    center = draft.part(key)['local_center_z_mm']
    draft.set_dimension(('parts', key, 'pole_gap_mm'), gap)
    assert draft.part(upper)['local_end_z_mm'] == pytest.approx(center-gap/2)
    assert draft.part(lower)['local_start_z_mm'] == pytest.approx(center+gap/2)
    assert draft.part(upper)['local_start_z_mm'] == poles[0]['local_start_z_mm']
    assert draft.part(lower)['local_end_z_mm'] == poles[1]['local_end_z_mm']
    for pole in (upper, lower):
        assert _material(draft, pole) != old_shapes[pole]
    _assert_changed_only(before, draft.document, {
        key: {'pole_gap_mm'}, upper: {'local_end_z_mm', 'local_center_z_mm', 'length_mm'},
        lower: {'local_start_z_mm', 'local_center_z_mm', 'length_mm'}})
    draft.validate()
    edited = deepcopy(draft.document)
    draft.undo()
    assert draft.document == before
    draft.redo()
    assert draft.document == edited
    destination = tmp_path/(key+'.toml')
    draft.save_copy(destination)
    reopened = PartModelDocument(destination)
    reopened.validate()
    _assert_saved_material_document(reopened.document, edited)


@pytest.mark.parametrize('key', ('condenser_lens_1', 'condenser_lens_2'))
def test_c1_c2_single_pole_gap_is_explicit_reference_not_fabricated_material(key):
    draft = PartModelDocument(COLUMN)
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match='[Rr]eference|one pole|material gap'):
        draft.set_dimension(('parts', key, 'pole_gap_mm'), 20.2)
    assert draft.document == before and not draft.dirty and not draft.can_undo


@pytest.mark.parametrize('key', ('condenser_lens_1', 'condenser_lens_2',
    'condenser_lens_3', 'objective_lens', 'mini_condenser'))
def test_winding_thickness_changes_real_copper_and_shared_cooling_interfaces(key, tmp_path):
    draft = PartModelDocument(COLUMN)
    before = deepcopy(draft.document)
    shared = key in {'condenser_lens_1', 'condenser_lens_2'}
    owners = ('condenser_lens_1', 'condenser_lens_2') if shared else (key,)
    old_shapes = {owner: _material(draft, owner+'_excitation_coil') for owner in owners}
    new = draft.part(key)['mechanical_coil_radial_thickness_mm']*1.01
    draft.set_dimension(('parts', key, 'mechanical_coil_radial_thickness_mm'), new)
    allowed = {}
    for owner in owners:
        coil = draft.part(owner+'_excitation_coil')
        assert (coil['mechanical_outer_diameter_mm']-coil['mechanical_inner_diameter_mm'])/2 == pytest.approx(new)
        assert _material(draft, coil['key']) != old_shapes[owner]
        allowed[owner] = {'mechanical_coil_radial_thickness_mm'}
        allowed[coil['key']] = {'mechanical_outer_diameter_mm'}
    if key == 'condenser_lens_3':
        housing = draft.part(key+'_housing')
        assert housing['mechanical_c3_end_outer_diameter_mm'] == coil['mechanical_outer_diameter_mm']
        _material(draft, housing['key'])
        allowed[housing['key']] = {'mechanical_c3_end_outer_diameter_mm'}
    _assert_changed_only(before, draft.document, allowed)
    draft.validate()
    edited = deepcopy(draft.document)
    draft.undo()
    assert draft.document == before
    draft.redo()
    assert draft.document == edited
    destination = tmp_path/(key+'.toml')
    draft.save_copy(destination)
    _assert_saved_material_document(PartModelDocument(destination).document, edited)


@pytest.mark.parametrize('key,prefix', (
    ('condenser_lens_3', 'mechanical_c3_end_'),
    ('adapter_lens', 'mechanical_adapter_end_'),
    ('probe_tl22_lens', 'mechanical_tl22_end_')))
@pytest.mark.parametrize('field', ('mechanical_inner_diameter_mm', 'mechanical_outer_diameter_mm'))
def test_direct_coil_radial_edit_updates_parent_and_contact_hardware(key, prefix, field, tmp_path):
    draft = PartModelDocument(COLUMN)
    before = deepcopy(draft.document)
    coilkey, housingkey = key+'_excitation_coil', key+'_housing'
    shapes = [_material(draft, p) for p in (coilkey, housingkey)]
    draft.set_dimension(('parts', coilkey, field), draft.part(coilkey)[field]*1.01)
    coil, housing = draft.part(coilkey), draft.part(housingkey)
    for side in ('inner', 'outer'):
        assert housing[prefix+side+'_diameter_mm'] == coil['mechanical_'+side+'_diameter_mm']
    assert draft.part(key)['mechanical_coil_radial_thickness_mm'] == pytest.approx(
        (coil['mechanical_outer_diameter_mm']-coil['mechanical_inner_diameter_mm'])/2)
    for p, old in zip((coilkey, housingkey), shapes):
        assert _material(draft, p) != old
    _assert_changed_only(before, draft.document, {key: {'mechanical_coil_radial_thickness_mm'},
        coilkey: {field}, housingkey: {prefix+field.removeprefix('mechanical_')}})
    draft.validate()
    edited = deepcopy(draft.document)
    draft.undo()
    assert draft.document == before
    draft.redo()
    assert draft.document == edited
    destination = tmp_path/(key+'.toml')
    draft.save_copy(destination)
    _assert_saved_material_document(PartModelDocument(destination).document, edited)


@pytest.mark.parametrize('key,field,value', (
    ('condenser_lens_3', 'pole_gap_mm', 200.),
    ('probe_tl22_lens', 'pole_gap_mm', 26.),
    ('condenser_lens_3', 'mechanical_coil_radial_thickness_mm', 100.),
    ('condenser_lens_3_excitation_coil', 'mechanical_inner_diameter_mm', 160.),
    ('adapter_lens_excitation_coil', 'mechanical_outer_diameter_mm', 200.)))
def test_invalid_linked_material_edit_rolls_back_every_part_and_history(key, field, value):
    draft = PartModelDocument(COLUMN)
    before = deepcopy(draft.document)
    with pytest.raises(ValueError):
        draft.set_dimension(('parts', key, field), value)
    assert draft.document == before and not draft.dirty and not draft.can_undo and not draft.can_redo


@pytest.mark.parametrize('source,prefix', (
    ('condenser_lens_3', 'mechanical_c3_end_'),
    ('adapter_lens', 'mechanical_adapter_end_'),
    ('probe_tl22_lens', 'mechanical_tl22_end_')))
def test_copied_round_lens_bindings_remain_local_after_placement_and_save(tmp_path, source, prefix):
    draft = PartModelDocument(COLUMN)
    originals = deepcopy(draft.document['parts'])
    key = draft.copy_component_from(draft.document, source,
                                    'round_copy', 2500., include_children=True)
    draft.place_component(key, 2600.)
    gap = draft.part(key)['pole_gap_mm']*1.01
    draft.set_dimension(('parts', key, 'pole_gap_mm'), gap)
    coilkey = key+'_excitation_coil'
    diameter = draft.part(coilkey)['mechanical_outer_diameter_mm']*1.01
    draft.set_dimension(('parts', coilkey, 'mechanical_outer_diameter_mm'), diameter)
    assert draft.part(key)['local_center_z_mm'] == 2600.
    assert draft.part(key+'_upper_pole')['local_end_z_mm'] == pytest.approx(2600.-gap/2)
    assert draft.part(key+'_lower_pole')['local_start_z_mm'] == pytest.approx(2600.+gap/2)
    housing = draft.part(key+'_housing')
    assert housing[prefix+'coil_key'] == coilkey
    assert housing[prefix+'outer_diameter_mm'] == diameter
    for original in originals:
        assert draft.part(original['key']) == original
    for child in ('upper_pole', 'lower_pole', 'excitation_coil', 'housing'):
        _material(draft, key+'_'+child)
    draft.validate()
    expected = deepcopy(draft.document)
    destination = tmp_path/'copied-lens.toml'
    draft.save_copy(destination)
    _assert_saved_material_document(PartModelDocument(destination).document, expected)


@pytest.mark.parametrize('source,field', (
    ('condenser_lens_3_housing', 'mechanical_c3_end_coil_key'),
    ('adapter_lens_housing', 'mechanical_adapter_end_coil_key'),
    ('probe_tl22_lens_housing', 'mechanical_tl22_end_coil_key'),
    ('c1_c2_pole_piece_cartridge', 'mechanical_cartridge_pole_keys')))
def test_independent_passive_hardware_copy_retains_source_reference_provenance(source, field):
    draft = PartModelDocument(COLUMN)
    old = deepcopy(draft.part(source)[field])
    key = draft.copy_component_from(draft.document, source, 'passive_copy',
                                    2500., include_children=False)
    assert draft.part(key)[field] == old
    draft.validate()


@pytest.mark.parametrize('field', ('mechanical_inner_diameter_mm', 'mechanical_outer_diameter_mm'))
def test_direct_shared_coil_edit_keeps_other_channel_radial_geometry_matched(field):
    draft = PartModelDocument(COLUMN)
    first = 'condenser_lens_1_excitation_coil'
    second = 'condenser_lens_2_excitation_coil'
    before = deepcopy(draft.document)
    new = draft.part(first)[field]*1.01
    draft.set_dimension(('parts', first, field), new)
    assert draft.part(first)[field] == draft.part(second)[field] == new
    for key in ('condenser_lens_1', 'condenser_lens_2'):
        coil = draft.part(key+'_excitation_coil')
        assert draft.part(key)['mechanical_coil_radial_thickness_mm'] == pytest.approx(
            (coil['mechanical_outer_diameter_mm']-coil['mechanical_inner_diameter_mm'])/2)
        _material(draft, coil['key'])
    _assert_changed_only(before, draft.document, {
        first: {field}, second: {field}, 'condenser_lens_1': {'mechanical_coil_radial_thickness_mm'},
        'condenser_lens_2': {'mechanical_coil_radial_thickness_mm'}})
    draft.validate()
