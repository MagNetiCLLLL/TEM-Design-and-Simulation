"""HP2 passive support geometry, legacy compatibility and editor transactions."""
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
    PREFIX, FIELDS, DIMENSIONS, hexapole_support_dimensions,
    hexapole_support_solids, validate_hexapole_support,
)
from temsim.module_manifest import read_document
from temsim.part_model_3d import part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT

KEY = 'probe_hp2_hexapole'
PATH = INSTRUMENT_CONFIG_ROOT / 'column/C3_ProbeCorrector.toml'


@pytest.mark.parametrize('filename', [
    'C3_ProbeCorrector.toml', 'C3_ProbeCorrector_BCOR.toml',
    'C3_ProbeCorrector_ImageCorrector.toml',
])
def test_probe_corrector_variants_share_supports_and_preserve_copper_and_iron(filename):
    main = read_document(PATH)
    document = read_document(INSTRUMENT_CONFIG_ROOT/'column'/filename)
    before = deepcopy(document)
    by_key = {p['key']: p for p in document['parts']}
    for key, prefix, count, original_count in (
        (KEY, PREFIX, 25, 13), ('probe_dph2_coils', 'winding_support_', 13, 4),
    ):
        part = by_key[key]
        source = _part(main, key)
        assert {k: v for k, v in part.items() if k.startswith(prefix)} == {
            k: v for k, v in source.items() if k.startswith(prefix)}
        legacy = {k: v for k, v in part.items() if not k.startswith(prefix)}
        old = multipole_solids(legacy, {**by_key, key: legacy})
        current = multipole_solids(part, by_key)
        assert len(old) == original_count and len(current) == count
        current_by_region = {s.region: s for s in current}
        for original in old:
            actual = current_by_region[original.region]
            assert actual.shape == original.shape
            assert actual.material == original.material
        assert all(s.material in {'soft_magnetic', 'copper_winding'} for s in old)
    assert document == before


def _part(document, key=KEY):
    return next(p for p in document['parts'] if p['key'] == key)


def _legacy(part):
    return {k: v for k, v in part.items() if not k.startswith(PREFIX)}


def _body(spec):
    return _manifold(spec.shape, 96)


@pytest.fixture(scope='module')
def host():
    document = read_document(PATH)
    part = _part(document)
    specs = multipole_solids(part, {})
    return document, part, specs, {s.region: _body(s) for s in specs}


def test_default_support_preserves_all_original_thirteen_material_shapes_and_optics(host):
    document, p, specs, _ = host
    before = deepcopy(document)
    assert {k for k in p if k.startswith(PREFIX)} == FIELDS
    assert len(specs) == 25
    assert specs[:13] == multipole_solids(_legacy(p), {})
    assert len(hexapole_support_solids(p)) == 12
    assert (p['local_start_z_mm'], p['local_center_z_mm'], p['local_end_z_mm'],
            p['effective_length_mm'], p['mechanical_inner_diameter_mm'],
            p['mechanical_outer_diameter_mm']) == (685.4, 694.5, 703.6, 6., 24., 75.)
    assert all(s.parameters['passive_geometry_only'] and s.parameters['optical_field_unchanged']
               for s in specs[13:])
    assert len(hexapole_support_solids(_part(document, 'probe_hp1_hexapole'))) == 12
    assert hexapole_support_solids(_legacy(p)) == ()
    assert document == before


def test_supports_are_connected_closed_sixfold_and_materials_never_interpenetrate(host):
    document, p, specs, bodies = host
    model = part_model_from_document(document, KEY, include_children=False, angular_segments=96)
    assert len(model.meshes) == 25
    for spec, mesh in zip(specs, model.meshes, strict=True):
        body = bodies[spec.region]
        assert body.volume() > 0 and len(body.decompose()) == 1
        counts = Counter(tuple(sorted((int(a), int(b)))) for face in mesh.faces
                         for a, b in zip(face, np.roll(face, -1)))
        assert set(counts.values()) == {2}
        np.testing.assert_allclose([mesh.vertices.min(0), mesh.vertices.max(0)],
                                   np.asarray(body.bounding_box()).reshape(2, 3), atol=1e-9)
        radius = np.linalg.norm(mesh.vertices[:, :2], axis=1)
        assert radius.min() >= 12.-1e-8 and radius.max() <= 37.5+1e-8
    for (ka, a), (kb, b) in combinations(bodies.items(), 2):
        assert (a ^ b).volume() == pytest.approx(0., abs=1e-7), (ka, kb)
    for family in ('former', 'core_bedding'):
        original = bodies[f'coil_1_{family}']
        for i in range(2, 7):
            rotated = original.rotate((0., 0., 60.*(i-1)))
            actual = bodies[f'coil_{i}_{family}']
            assert (rotated-actual).volume() == pytest.approx(0., abs=1e-7)
            assert (actual-rotated).volume() == pytest.approx(0., abs=1e-7)


def test_former_contacts_copper_and_bedding_joins_core_without_blocking_axial_bore(host):
    _, p, _, bodies = host
    d = hexapole_support_dimensions(p)
    for i in range(1, 7):
        for a, b in ((bodies[f'coil_{i}'], bodies[f'coil_{i}_former']),
                     (bodies[f'coil_{i}_former'], bodies[f'coil_{i}_core_bedding']),
                     (bodies[f'coil_{i}_core_bedding'], bodies[f'pole_pin_{i}'])):
            assert a.min_gap(b, 1.) == pytest.approx(0., abs=1e-7)
            # Finite-area contact, not merely one coincident vertex. Coplanar
            # mesh booleans can split a contact; use a conservative lower bound.
            area = (a.surface_area()+b.surface_area()-(a+b).surface_area())/2
            assert area > 50.
    a, b = bodies['coil_1'], bodies['coil_1_former']
    expected = (2*(d['coil_window_width_mm']+d['coil_window_height_mm'])*d['coil_radial_length_mm']
        +2*(d['coil_width_mm']*d['coil_height_mm']-d['coil_window_width_mm']*d['coil_window_height_mm']))
    assert (a.surface_area()+b.surface_area()-(a+b).surface_area())/2 == pytest.approx(expected)
    bore = _manifold(annulus(d['start_mm'], d['end_mm'], 0., 12.), 96)
    for body in bodies.values():
        assert (body ^ bore).volume() == pytest.approx(0., abs=1e-7)


def test_added_supports_preserve_dph2_tube_clearance_and_existing_tl22_contact(host):
    _, p, _, bodies = host
    # Conservative filled envelopes include every possible DPH2 winding angle
    # and the new end seals, rather than relying on an empty copper window.
    dph = _manifold(annulus(687.21, 701.79, 11.06, 11.94), 96)
    tube = _manifold(annulus(685.4, 703.6, 2.88, 9.6), 96)
    support = _body(hexapole_support_solids(p)[0])
    for region, body in bodies.items():
        assert (body ^ dph).volume() == pytest.approx(0., abs=1e-7), region
        assert (body ^ tube).volume() == pytest.approx(0., abs=1e-7), region
    assert bodies['pole_pin_1'].min_gap(dph, .1) == pytest.approx(.06, abs=1e-6)
    assert support.min_gap(dph, 15.) > 10.
    # HP2's unchanged yoke reaches the TL22 start plane; do not claim a gap.
    assert bodies['return_yoke'].bounding_box()[5] == pytest.approx(703.6)
    assert support.bounding_box()[5] < 703.6


def test_saved_flange_dimension_changes_supports_without_changing_original_material(host):
    _, p, original, _ = host
    changed = multipole_solids({**p, PREFIX+'flange_thickness_mm': .35}, {})
    assert changed[:13] == original[:13]
    for before, after in zip(original[13:], changed[13:], strict=True):
        assert _body(after).volume() > _body(before).volume()
    assert hexapole_support_dimensions(p)['core_bedding_tangential_thickness_mm'] == pytest.approx(.7445)
    assert hexapole_support_dimensions(p)['core_bedding_axial_thickness_mm'] == pytest.approx(.478)


@pytest.mark.parametrize('field', DIMENSIONS)
@pytest.mark.parametrize('value', [False, 0., -1., float('nan'), float('inf'), 10**1000])
def test_invalid_support_dimensions_are_rejected(field, value, host):
    _, p, _, _ = host
    with pytest.raises(ValueError, match='finite and positive'):
        validate_hexapole_support({**p, field: value})


@pytest.mark.parametrize('changes', [
    {PREFIX+'former_wall_thickness_mm': .73},
    {PREFIX+'flange_thickness_mm': 2.},
    {PREFIX+'unknown_width_mm': .1}, {PREFIX+'profile': 'invented'},
    {PREFIX+'geometry_status': 'OEM'}, {PREFIX+'source_note': ''},
    {'geometry_template_key': 'probe_hpc_hexapole'},
])
def test_bad_profiles_or_support_collisions_fail_before_building(changes, host):
    _, p, _, _ = host
    with pytest.raises(ValueError):
        hexapole_support_solids({**p, **changes})


def test_parameters_edit_copy_move_transform_save_and_reopen(tmp_path, host):
    _, p, _, _ = host
    draft = PartModelDocument(PATH)
    before = multipole_solids(draft.part(KEY), {})
    draft.set_dimension(('parts', KEY, DIMENSIONS[0]), .3)
    after = multipole_solids(draft.part(KEY), {})
    assert before[:13] == after[:13]
    assert _body(after[13]).volume() > _body(before[13]).volume()
    assert _body(after[14]).volume() < _body(before[14]).volume()
    draft.undo()
    assert draft.part(KEY)[DIMENSIONS[0]] == .25
    draft.redo()
    key = draft.copy_component_from(draft.document, KEY, 'independent_hp2', 2500., include_children=False)
    draft.place_component(key, 2600.)
    copied = draft.part(key)
    assert copied['geometry_template_key'] == KEY
    assert hexapole_support_dimensions(copied)['start_mm'] == pytest.approx(2590.9)
    assert [s.shape for s in hexapole_support_solids({**copied, 'effective_length_mm': 4.})] == [
        s.shape for s in hexapole_support_solids(copied)]
    draft.update_model_3d(key, {'transform': {'scale_xy': [1.1, .9], 'rotation_deg': [5., 10., 25.]}})
    native = part_model_from_document(draft.document, key, include_children=False, angular_segments=96)
    posed = posed_solids(draft.part(key), {p['key']: p for p in draft.document['parts']})
    for spec, mesh in zip(posed, native.meshes, strict=True):
        np.testing.assert_allclose([mesh.vertices.min(0), mesh.vertices.max(0)],
                                   np.asarray(_body(spec).bounding_box()).reshape(2, 3), atol=1e-8)
    target = tmp_path/'hp2.toml'
    draft.save_copy(target)
    reopened = PartModelDocument(target)
    assert reopened.part(key)[DIMENSIONS[0]] == .3
    reopened.validate()


def test_invalid_editor_edit_is_atomic_and_replacement_keeps_inactive_recipe(host):
    draft = PartModelDocument(PATH)
    before = deepcopy(draft.document)
    history = (draft.dirty, draft.can_undo, draft.can_redo)
    with pytest.raises(ValueError, match='core bedding'):
        draft.set_dimension(('parts', KEY, DIMENSIONS[0]), .73)
    assert draft.document == before
    assert (draft.dirty, draft.can_undo, draft.can_redo) == history
    key = draft.copy_component_from(draft.document, KEY, 'hp2_replacement', 2500., include_children=False)
    draft.set_model_3d(key, {'schema_version': 1, 'base': {
        'kind': 'box', 'width_mm': 10., 'height_mm': 10., 'length_mm': 1.}})
    assert len(part_model_from_document(draft.document, key, include_children=False).meshes) == 1
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match='pole length'):
        draft.set_model_3d(key, {'schema_version': 1, 'base': {'kind': 'existing'}})
    assert draft.document == before
    draft.set_dimension(('parts', key, 'length_mm'), 18.2)
    draft.set_dimension(('parts', key, DIMENSIONS[0]), .73)
    assert hexapole_support_solids(draft.part(key)) == ()
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match='core bedding'):
        draft.set_model_3d(key, {'schema_version': 1, 'base': {'kind': 'existing'}})
    assert draft.document == before
