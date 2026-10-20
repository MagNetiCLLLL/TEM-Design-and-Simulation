"""C3 aperture attachment, real material clearance and editor contracts."""
from collections import Counter
from copy import deepcopy
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.cad_aperture_support import PREFIX, FIELDS, aperture_support_solids, validate_aperture_support
from temsim.cad_auxiliary import aperture_solids
from temsim.cad_catalog import is_reconstruction_region, posed_solids
from temsim.cad_geometry import annulus, box, transformed
from temsim.cad_mesh import _manifold
from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, decorate_physical_tube_row, tube_solids
from temsim.module_manifest import read_document
from temsim.part_model_3d import part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


KEY = 'condenser_aperture_3'
PATH = INSTRUMENT_CONFIG_ROOT / 'column/C3_ProbeCorrector.toml'


@pytest.fixture
def document():
    return read_document(PATH)


def _part(document):
    return next(p for p in document['parts'] if p['key'] == KEY)


def _mat(shape):
    return _manifold(shape, 64)


def _bounds(solid):
    return np.asarray(solid.bounding_box()).reshape(2, 3)


def _tube(document, runtime=None):
    g = document['geometry']
    parts = [SimpleNamespace(key=p['key'], module_key='column', data=p,
                             center_z_mm=float(p['local_center_z_mm'])+450.) for p in document['parts']]
    module = SimpleNamespace(key='column', geometry=g, parts=parts)
    row = dict(key=CONTINUOUS_TUBE_KEY, mechanical_profile='vacuum_liner',
               local_start_z_mm=g['c1_c2_objective_vacuum_tube_start_z_mm']+450.,
               local_end_z_mm=g['c1_c2_objective_vacuum_tube_end_z_mm']+450.,
               mechanical_inner_diameter_mm=g['c1_c2_objective_vacuum_tube_inner_diameter_mm'],
               mechanical_outer_diameter_mm=g['c1_c2_objective_vacuum_tube_outer_diameter_mm'])
    decorated = decorate_physical_tube_row(row, SimpleNamespace(modules=[module], parts=parts),
                                          runtime_values=runtime)
    return _mat(tube_solids(decorated)[0].shape), decorated


def test_c3_saved_optical_geometry_and_unmeasured_hole_count_remain_explicit(document):
    p = _part(document)
    assert {k for k in p if k.startswith(PREFIX)} == FIELDS
    assert (p['local_start_z_mm'], p['local_center_z_mm'], p['local_end_z_mm'],
            p['optical_reference_local_z_mm'], p['length_mm']) == (550., 560., 570., 560., 20.)
    assert (p['plate_thickness_mm'], p['mechanical_outer_diameter_mm'], p['mechanical_bore_diameter_mm'],
            p['vacuum_inner_diameter_mm'], p['maximum_radius_mm']) == (.2, 75., 4., 5., 2.)
    assert not any(k in p for k in ('aperture_hole_diameters_mm', 'aperture_hole_diameters_um',
                                  'hole_diameters_mm', 'hole_diameters_um'))
    original = deepcopy(document)
    specs = aperture_support_solids(p)
    assert [s.region for s in specs] == ['body', 'connection_rod', 'fixing_screw']
    assert all(s.status == 'engineering_reconstruction' and 'C3' in s.note for s in specs)
    assert all(s.parameters['working_diameter_mm'] is None and not s.parameters['field_source_added']
               and s.parameters['working_hole_count'] == 'selected_only_total_unknown' for s in specs)
    assert not is_reconstruction_region(p, 'body')
    assert all(is_reconstruction_region(p, s.region) for s in specs[1:])
    assert document == original


def test_c3_material_is_closed_positive_seated_and_nonoverlapping(document):
    p = _part(document)
    model = part_model_from_document(document, KEY, include_children=False, angular_segments=64)
    solids = [_mat(s.shape) for s in aperture_support_solids(p)]
    assert len(model.meshes) == 3
    for mesh, solid in zip(model.meshes, solids, strict=True):
        edges = Counter(tuple(sorted((int(a), int(b)))) for face in mesh.faces
                        for a, b in zip(face, np.roll(face, -1)))
        assert set(edges.values()) == {2} and not mesh.is_exact and solid.volume() > 0
        np.testing.assert_allclose([mesh.vertices.min(0), mesh.vertices.max(0)], _bounds(solid), atol=1e-10)
        assert np.linalg.norm(mesh.vertices[:, :2], axis=1).max() < p['mechanical_outer_diameter_mm']/2
    for a, b in combinations(solids, 2):
        assert (a ^ b).volume() == pytest.approx(0., abs=1e-8)
    bounds = [_bounds(s) for s in solids]
    assert bounds[0][1, 2] == pytest.approx(bounds[1][0, 2])
    assert min(b[0, 2] for b in bounds) == pytest.approx(559.3)
    assert max(b[1, 2] for b in bounds) == pytest.approx(562.1)


@pytest.mark.parametrize('radius,dx,dy', [(0., 0., 0.), (.05, .2, -.1), (2., 0., 0.)])
def test_c3_working_opening_does_not_move_fixing_bore_or_support(document, radius, dx, dy):
    p = _part(document)
    runtime = dict(radius_mm=radius, offset_x_mm=dx, offset_y_mm=dy)
    specs = aperture_support_solids(p, runtime)
    baseline = aperture_support_solids(p)
    assert [s.shape for s in specs[1:]] == [s.shape for s in baseline[1:]]
    expected_removed = 32*np.sin(2*np.pi/64)*radius**2*p['plate_thickness_mm']
    assert _mat(baseline[0].shape).volume()-_mat(specs[0].shape).volume() == pytest.approx(expected_removed, abs=1e-9)
    if radius:
        bore = _mat(transformed(annulus(559.9, 560.1, 0., radius), offset=(dx, dy, 0.)))
        assert (_mat(specs[0].shape) ^ bore).volume() == pytest.approx(0., abs=1e-9)
    model = part_model_from_document(document, KEY, include_children=False, angular_segments=64,
                                     runtime_values={KEY: runtime})
    plate = next(m for m in model.meshes if m.region == 'body')
    assert all(path[0] != 'runtime' for path in plate.surfaces['fixing_bore']['parameter_paths'])
    if radius:
        assert ('runtime', KEY, 'radius_mm') in plate.surfaces['working_opening']['parameter_paths']


def test_c3_tube_slot_insertion_and_neighbor_material_clearance(document):
    p = _part(document)
    rows = {p['key']: p for p in document['parts']}
    validate_aperture_support(p, rows, document['geometry'])
    tube, decorated = _tube(document)
    slot = next(s for s in decorated['mechanical_material_tube_aperture_slots'] if s['aperture_key'] == KEY)
    np.testing.assert_allclose(slot['local_bounds_mm'], [[-3.45, -3.45, 559.7], [3.45, 9.8, 560.3]], atol=1e-12)
    for travel in (0., 2., 8., 20.):
        for spec in aperture_support_solids(p):
            assert (tube ^ _mat(transformed(spec.shape, offset=(0., travel, 450.)))).volume() == pytest.approx(0., abs=1e-8)
    assert (tube ^ _mat(box((0., -9.4, 1010.), (.05, .05, .05)))).volume() > 0
    own = [_mat(s.shape) for s in posed_solids(p, rows)]
    neighbors = [s for k, row in rows.items() if k.startswith('condenser_lens_3') or k == 'beam_deflector'
                 for s in posed_solids(row, rows)]
    assert neighbors
    for neighbor in neighbors:
        material = _mat(neighbor.shape)
        for solid in own:
            assert (solid ^ material).volume() == pytest.approx(0., abs=1e-8)
    upper = max(_bounds(_mat(s.shape))[1, 2] for s in neighbors if s.key.startswith('condenser_lens_3'))
    lower = min(_bounds(_mat(s.shape))[0, 2] for s in neighbors if s.key == 'beam_deflector')
    assert min(_bounds(s)[0, 2] for s in own)-upper == pytest.approx(9.3)
    assert lower-max(_bounds(s)[1, 2] for s in own) == pytest.approx(7.9)


def test_c3_scaled_and_offset_native_cad_and_tube_share_one_placement():
    draft = PartModelDocument(PATH)
    draft.update_model_3d(KEY, {'transform': {'scale_xy': [1.4, 1.1], 'offset_mm': [.2, -.1, 0.],
                                           'rotation_deg': [0., 0., 17.]}})
    rows = {p['key']: p for p in draft.document['parts']}
    specs = posed_solids(rows[KEY], rows)
    model = part_model_from_document(draft.document, KEY, include_children=False, angular_segments=64)
    tube, decorated = _tube(draft.document)
    slot = next(s for s in decorated['mechanical_material_tube_aperture_slots'] if s['aperture_key'] == KEY)
    np.testing.assert_allclose(slot['rotation_local_to_global'], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(slot['local_bounds_mm'], [[-4.75, -3.775, 559.7], [4.75, 9.8, 560.3]], atol=1e-12)
    for spec, mesh in zip(specs, model.meshes, strict=True):
        np.testing.assert_allclose([mesh.vertices.min(0), mesh.vertices.max(0)], _bounds(_mat(spec.shape)), atol=1e-10)
        assert (tube ^ _mat(transformed(spec.shape, offset=(0., 0., 450.)))).volume() == pytest.approx(0., abs=1e-8)


@pytest.mark.parametrize('path,value', [
    ((PREFIX+'rod_end_y_mm',), 39.),
    ((PREFIX+'rod_thickness_mm',), 20.),
    (('model_3d', 'transform', 'scale_xy', 1), .5),
])
def test_c3_invalid_geometry_is_rejected_without_draft_or_history_changes(path, value):
    draft = PartModelDocument(PATH)
    if path[0] == 'model_3d':
        draft.update_model_3d(KEY, {'transform': {'scale_xy': [1., 1.]}})
    before = deepcopy(draft.document)
    dirty, can_undo, can_redo = draft.dirty, draft.can_undo, draft.can_redo
    with pytest.raises(ValueError):
        draft.set_dimension(('parts', KEY, *path), value)
    assert draft.document == before
    assert (draft.dirty, draft.can_undo, draft.can_redo) == (dirty, can_undo, can_redo)


def test_c3_edit_copy_save_reopen_and_old_files_preserve_contract(document, tmp_path):
    draft = PartModelDocument(PATH)
    before = aperture_support_solids(draft.part(KEY))
    draft.set_dimension(('parts', KEY, PREFIX+'rod_end_y_mm'), 35.)
    after = aperture_support_solids(draft.part(KEY))
    assert _mat(after[1].shape).volume() < _mat(before[1].shape).volume()
    assert after[0].shape == before[0].shape and after[2].shape == before[2].shape
    draft.undo()
    assert draft.part(KEY)[PREFIX+'rod_end_y_mm'] == 36.5
    draft.redo()
    key = draft.copy_component_from(draft.document, KEY, 'c3_aperture_copy', 2500., include_children=False)
    assert draft.part(key)['geometry_template_key'] == KEY
    draft.update_model_3d(key, {'transform': {'rotation_deg': [5., 10., 25.]}})
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match='outside the continuous tube'):
        draft.place_component(key, 560.)
    assert draft.document == before
    destination = tmp_path / 'independent-c3.toml'
    draft.save_copy(destination)
    reopened = PartModelDocument(destination)
    assert reopened.part(key)[PREFIX+'rod_end_y_mm'] == 35.
    reopened.validate()
    for path in (INSTRUMENT_CONFIG_ROOT / 'column').glob('*.toml'):
        if path == PATH:
            continue
        row = next((p for p in read_document(path)['parts'] if p['key'] == KEY), None)
        if row is not None:
            assert aperture_support_solids(row) is None
            assert len(aperture_solids(row)) == 1
