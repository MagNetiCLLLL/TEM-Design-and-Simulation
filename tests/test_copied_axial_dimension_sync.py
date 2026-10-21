"""Copied passive assemblies obey the same axial law in both editors."""
from copy import deepcopy

import pytest

from temsim.cad_catalog import part_solids
from temsim.component_operations import resized_copied_axial_fields
from temsim.manifest_editor import ManifestEditor, ManifestTarget, _complete_part_length_updates
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


PATH = INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml'
HOUSINGS = (
    ('condenser_lens_3_housing', 'mechanical_c3_end_'),
    ('adapter_lens_housing', 'mechanical_adapter_end_'),
    ('probe_tl22_lens_housing', 'mechanical_tl22_end_'),
    ('probe_tl21_lens_housing', 'mechanical_transfer_end_'),
    ('probe_tl12_lens_housing', 'mechanical_transfer_end_'),
)


def complete(document, updates):
    before = deepcopy((document, updates))
    fields = _complete_part_length_updates(document, updates)
    candidate = deepcopy(document)
    rows = {p['key']: p for p in candidate['parts']}
    for (section, key, field), value in fields.items():
        assert section == 'parts'
        rows[key][field] = deepcopy(value)
    assert (document, updates) == before
    return candidate, rows, fields


@pytest.mark.parametrize('source,prefix', HOUSINGS)
@pytest.mark.parametrize('move', [False, True])
def test_copied_housing_length_matches_3d_editor_in_every_family(source, prefix, move):
    draft = PartModelDocument(PATH)
    key = draft.copy_component_from(draft.document, source, 'axial_copy', 2500., include_children=False)
    before = deepcopy(draft.document)
    value = draft.part(key)['length_mm']*1.1
    updates = {('parts', key, 'length_mm'): value}
    draft.set_dimension(('parts', key, 'length_mm'), value)
    if move:
        updates[('parts', key, 'local_center_z_mm')] = 2600.
        draft.place_component(key, 2600.)
    candidate, rows, _ = complete(before, updates)
    validate_document(candidate)
    p, actual = draft.part(key), rows[key]
    for field in ('local_start_z_mm', 'local_end_z_mm', prefix+'upper_contact_local_z_mm',
                  prefix+'lower_contact_local_z_mm'):
        assert actual[field] == pytest.approx(p[field], abs=1e-10)
    assert part_solids(actual, rows) == part_solids(p, {p['key']: p for p in draft.document['parts']})
    assert all(rows[row['key']] == row for row in before['parts'] if row['key'] != key)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('explicit_endpoints', [False, True])
def test_consistent_explicit_contacts_preserved_independent_of_batch_order(reverse, explicit_endpoints):
    draft = PartModelDocument(PATH)
    key = draft.copy_component_from(draft.document, 'probe_tl21_lens_housing', 'axial_copy', 2500., include_children=False)
    before = deepcopy(draft.document)
    draft.set_dimension(('parts', key, 'length_mm'), 28.)
    expected = draft.part(key)
    fields = ['mechanical_transfer_end_upper_contact_local_z_mm',
              'mechanical_transfer_end_lower_contact_local_z_mm']
    if explicit_endpoints:
        fields += ['local_start_z_mm', 'local_end_z_mm']
    updates = {('parts', key, 'length_mm'): 28.,
               **{('parts', key, name): expected[name] for name in fields}}
    if reverse:
        updates = dict(reversed(list(updates.items())))
    candidate, _, completed = complete(before, updates)
    assert all(completed[path] == value for path, value in updates.items())
    assert candidate == draft.document


@pytest.mark.parametrize('contact', ['upper', 'lower'])
def test_conflicting_contact_rejects_before_manifest_save_touches_any_file(contact, tmp_path):
    draft = PartModelDocument(PATH)
    key = draft.copy_component_from(draft.document, 'probe_tl21_lens_housing', 'axial_copy', 2500., include_children=False)
    path = tmp_path/'copied-column.toml'
    draft.save_copy(path)
    original = path.read_bytes()
    document = read_document(path)
    source = deepcopy(document)
    field = f'mechanical_transfer_end_{contact}_contact_local_z_mm'
    updates = {('parts', key, 'length_mm'): 28., ('parts', key, field): draft.part(key)[field]}
    with pytest.raises(ValueError, match='explicit value conflicts with copied axial scaling'):
        ManifestEditor(tmp_path).save(ManifestTarget(path.name, key), updates, None)
    assert path.read_bytes() == original and document == source
    assert list(tmp_path.iterdir()) == [path]


def test_inactive_recipe_scaling_matches_box_length_and_restores_without_stale_contacts():
    draft = PartModelDocument(PATH)
    key = draft.copy_component_from(draft.document, 'probe_tl21_lens_housing', 'axial_copy', 2500., include_children=False)
    draft.set_model_3d(key, {'schema_version': 1, 'base': {
        'kind': 'box', 'width_mm': 10., 'height_mm': 10., 'length_mm': 25.5}})
    before = deepcopy(draft.document)
    draft.set_dimension(('parts', key, 'length_mm'), 28.)
    candidate, rows, _ = complete(before, {('parts', key, 'length_mm'): 28.})
    assert candidate == draft.document
    assert rows[key]['model_3d']['base']['length_mm'] == 28.
    draft.set_model_3d(key, {'schema_version': 1, 'base': {'kind': 'existing'}})
    draft.validate()
    assert len(part_solids(draft.part(key), {})) == 5


def test_saved_material_intervals_and_relative_profile_share_the_copy_scaling_law():
    part = dict(key='copy', mechanical_part_role='custom_mechanical_copy',
        length_mm=10., local_center_z_mm=50., material_intervals_mm=[[45., 47.], [53., 55.]],
        magnetic_radial_profile_mm=[[-5., 8., 10.], [5., 8., 10.]])
    snapshot = deepcopy(part)
    result = resized_copied_axial_fields(part, 20., center_z_mm=100.)
    assert result['material_intervals_mm'] == [[90., 94.], [106., 110.]]
    assert result['magnetic_radial_profile_mm'] == [[-10., 8., 10.], [10., 8., 10.]]
    assert part == snapshot


def test_native_optical_assemblies_do_not_inherit_copy_scaling():
    draft = PartModelDocument(PATH)
    for key in ('probe_tl21_lens', 'probe_tl21_lens_housing'):
        part = draft.part(key)
        assert resized_copied_axial_fields(part, part['length_mm']*1.1) == {}
