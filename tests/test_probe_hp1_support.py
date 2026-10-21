"""HP1 winding support follows its own geometry without changing optical inputs."""
from collections import Counter
from copy import deepcopy
from itertools import combinations

import numpy as np
import pytest

from temsim.cad_catalog import posed_solids
from temsim.cad_geometry import annulus
from temsim.cad_mesh import _manifold
from temsim.cad_multipoles import multipole_solids
from temsim.cad_probe_hexapole_support import (
    PREFIX, DIMENSIONS, FIELDS, hexapole_support_dimensions,
    hexapole_support_solids,
)
from temsim.component_representation import non_material_role
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_3d import part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


KEY = 'probe_hp1_hexapole'
PATH = INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml'


def _legacy(part):
    return {key: value for key, value in part.items() if not key.startswith(PREFIX)}


@pytest.fixture(scope='module')
def host():
    document = read_document(PATH)
    rows = {part['key']: part for part in document['parts']}
    part = rows[KEY]
    specs = multipole_solids(part, rows)
    return document, rows, part, specs, {s.region: _manifold(s.shape, 96) for s in specs}


@pytest.mark.parametrize('filename', [
    'C3_ProbeCorrector.toml', 'C3_ProbeCorrector_BCOR.toml',
    'C3_ProbeCorrector_ImageCorrector.toml',
])
def test_hp1_support_is_saved_for_probe_variants_with_its_own_window(filename, host):
    _, _, main, _, _ = host
    document = read_document(INSTRUMENT_CONFIG_ROOT/'column'/filename)
    validate_document(document)
    rows = {p['key']: p for p in document['parts']}
    p = rows[KEY]
    assert {k: v for k, v in p.items() if k.startswith(PREFIX)} == {
        k: v for k, v in main.items() if k.startswith(PREFIX)}
    assert {k for k in p if k.startswith(PREFIX)} == FIELDS
    d = hexapole_support_dimensions(p)
    assert d['coil_window_height_mm'] == pytest.approx(8.688)
    assert d['stem_height_mm'] == pytest.approx(7.24)
    assert d['core_bedding_axial_thickness_mm'] == pytest.approx(.474)
    assert d['core_bedding_tangential_thickness_mm'] == pytest.approx(.7445)
    assert d['start_mm'] == pytest.approx(841.8)
    assert d['end_mm'] == pytest.approx(859.9)
    assert p['effective_length_mm'] == 10.
    hp2 = hexapole_support_dimensions(rows['probe_hp2_hexapole'])
    assert d['coil_window_height_mm'] != hp2['coil_window_height_mm']
    assert d['core_bedding_axial_thickness_mm'] != hp2['core_bedding_axial_thickness_mm']


def test_hp1_original_copper_poles_yoke_and_optical_fields_are_preserved(host):
    document, rows, part, specs, _ = host
    snapshot = deepcopy(document)
    assert len(specs) == 25 and len(hexapole_support_solids(part)) == 12
    assert specs[:13] == multipole_solids(_legacy(part), rows)
    assert hexapole_support_solids(_legacy(part)) == ()
    assert [s.shape for s in hexapole_support_solids({**part, 'effective_length_mm': 5.})] == [
        s.shape for s in hexapole_support_solids(part)]
    for key in ('probe_hpc_hexapole', 'probe_qpc_quadrupole', 'probe_qph1_quadrupole',
                'probe_hpol_hexapole', 'probe_qpol_quadrupole', 'probe_dp12_scan_deflector'):
        assert non_material_role(rows[key])
        assert multipole_solids(rows[key], rows) == ()
    assert document == snapshot


def test_hp1_support_materials_are_closed_connected_and_disjoint(host):
    document, _, part, specs, bodies = host
    model = part_model_from_document(document, KEY, include_children=False, angular_segments=96)
    assert len(model.meshes) == 25
    for spec, mesh in zip(specs, model.meshes, strict=True):
        body = bodies[spec.region]
        assert body.volume() > 0 and len(body.decompose()) == 1
        edges = Counter(tuple(sorted((int(a), int(b)))) for face in mesh.faces
                        for a, b in zip(face, np.roll(face, -1)))
        assert set(edges.values()) == {2}
        np.testing.assert_allclose([mesh.vertices.min(0), mesh.vertices.max(0)],
                                   np.asarray(body.bounding_box()).reshape(2, 3), atol=1e-8)
    for (ka, a), (kb, b) in combinations(bodies.items(), 2):
        assert (a ^ b).volume() == pytest.approx(0., abs=1e-7), (ka, kb)
    for family in ('former', 'core_bedding'):
        first = bodies[f'coil_1_{family}']
        for i in range(2, 7):
            rotated = first.rotate((0., 0., 60.*(i-1)))
            actual = bodies[f'coil_{i}_{family}']
            assert (rotated-actual).volume() == pytest.approx(0., abs=1e-7)
            assert (actual-rotated).volume() == pytest.approx(0., abs=1e-7)


def test_hp1_formers_and_bedding_join_original_winding_and_core(host):
    _, _, _, _, bodies = host
    for i in range(1, 7):
        for a, b in ((bodies[f'coil_{i}'], bodies[f'coil_{i}_former']),
                     (bodies[f'coil_{i}_former'], bodies[f'coil_{i}_core_bedding']),
                     (bodies[f'coil_{i}_core_bedding'], bodies[f'pole_pin_{i}'])):
            assert a.min_gap(b, 1.) == pytest.approx(0., abs=1e-7)
            assert (a.surface_area()+b.surface_area()-(a+b).surface_area())/2 > 50.


def test_hp1_open_bore_dph1_and_neighbor_clearances_are_preserved(host):
    _, rows, part, _, bodies = host
    # A full annular insert is conservative for every winding azimuth. The
    # optional end insulation may fit inside HP1, but never closes its bore.
    dph = rows['probe_dph1_coils']
    edge = float(dph.get('winding_support_end_cover_thickness_mm', 0.))
    insert = _manifold(annulus(dph['local_start_z_mm']-edge,
        dph['local_end_z_mm']+edge, 11.06, 11.94), 96)
    tube = _manifold(annulus(841.8, 859.9, 2.88, 9.6), 96)
    bore = _manifold(annulus(841.8, 859.9, 0., 12.), 96)
    for region, body in bodies.items():
        assert (body ^ insert).volume() == pytest.approx(0., abs=1e-7), region
        assert (body ^ tube).volume() == pytest.approx(0., abs=1e-7), region
        assert (body ^ bore).volume() == pytest.approx(0., abs=1e-7), region
    assert bodies['pole_pin_1'].min_gap(insert, .1) == pytest.approx(.06, abs=1e-6)
    assert bodies['coil_1_former'].min_gap(insert, 15.) > 10.
    assert part['local_start_z_mm']-rows['probe_tl21_lens']['local_end_z_mm'] == pytest.approx(29.1)
    assert rows['probe_dp11_coils']['local_start_z_mm']-part['local_end_z_mm'] == pytest.approx(29.1)


@pytest.mark.parametrize('field,value', [
    (DIMENSIONS[0], .3), (DIMENSIONS[1], .35),
    ('mechanical_hexapole_coil_window_height_mm', 8.8),
    ('mechanical_hexapole_coil_width_mm', 12.4),
    ('mechanical_hexapole_stem_height_mm', 7.3),
])
def test_hp1_support_follows_each_edit_and_does_not_change_optical_settings(field, value, host):
    _, _, original, specs, _ = host
    draft = PartModelDocument(PATH)
    draft.set_dimension(('parts', KEY, field), value)
    p = draft.part(KEY)
    new_specs = multipole_solids(p, {})
    assert any(a.shape != b.shape for a, b in zip(specs[13:], new_specs[13:], strict=True))
    for optical in ('effective_length_mm', 'optical_reference_local_z_mm',
                    'local_start_z_mm', 'local_center_z_mm', 'local_end_z_mm'):
        assert p[optical] == original[optical]
    if field in DIMENSIONS:
        assert specs[:13] == new_specs[:13]
    draft.undo()
    assert draft.part(KEY)[field] == original[field]
    draft.redo()
    assert draft.part(KEY)[field] == value


def test_hp1_support_copy_move_and_reopen_preserve_template_and_dimensions(tmp_path):
    draft = PartModelDocument(PATH)
    draft.set_dimension(('parts', KEY, DIMENSIONS[0]), .3)
    key = draft.copy_component_from(draft.document, KEY, 'independent_hp1', 2500., include_children=False)
    draft.place_component(key, 2600.)
    copied = draft.part(key)
    assert copied['geometry_template_key'] == KEY
    assert hexapole_support_dimensions(copied)['start_mm'] == pytest.approx(2590.95)
    assert copied['effective_length_mm'] == 10.
    expected = posed_solids(copied, {p['key']: p for p in draft.document['parts']})
    target = tmp_path/'hp1-copy.toml'
    draft.save_copy(target)
    reopened = PartModelDocument(target)
    assert reopened.part(key)[DIMENSIONS[0]] == .3
    assert posed_solids(reopened.part(key), {p['key']: p for p in reopened.document['parts']}) == expected
    reopened.validate()


def test_hp1_invalid_former_update_and_replacement_restore_are_atomic():
    draft = PartModelDocument(PATH)
    original = deepcopy(draft.document)
    with pytest.raises(ValueError, match='core bedding'):
        draft.set_dimension(('parts', KEY, DIMENSIONS[0]), .73)
    assert draft.document == original and not draft.can_undo
    key = draft.copy_component_from(draft.document, KEY, 'hp1_box', 2500., include_children=False)
    draft.set_model_3d(key, {'schema_version': 1, 'base': {
        'kind': 'box', 'width_mm': 10., 'height_mm': 10., 'length_mm': 18.1}})
    draft.set_dimension(('parts', key, DIMENSIONS[0]), .73)
    assert hexapole_support_solids(draft.part(key)) == ()
    before, history = deepcopy(draft.document), deepcopy(draft._history)
    with pytest.raises(ValueError, match='core bedding'):
        draft.set_model_3d(key, {'schema_version': 1, 'base': {'kind': 'existing'}})
    assert draft.document == before and draft._history == history
