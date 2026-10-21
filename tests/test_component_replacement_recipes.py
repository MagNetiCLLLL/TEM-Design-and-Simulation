"""Explicit mechanical bases supersede aperture/ordinary-deflector recipes."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.cad_catalog import posed_solids
from temsim.cad_mesh import _manifold
from temsim.cad_aperture_support import validate_aperture_support
from temsim.ordinary_deflector_geometry import support_dimensions
from temsim.part_model_3d import part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT

PATH = INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml'
APERTURES = ('condenser_aperture_2', 'condenser_aperture_3')
DEFLECTORS = ('condenser_deflector', 'beam_deflector', 'ac_deflector', 'image_diffraction_deflector')


def _copy(key):
    draft = PartModelDocument(PATH)
    copy = draft.copy_component_from(draft.document, key, 'replacement_copy', 2500., include_children=False)
    return draft, copy


def _model(draft, key):
    return part_model_from_document(draft.document, key, include_children=False, angular_segments=64)


def _bounds(draft, key):
    mesh, = _model(draft, key).meshes
    bounds = np.array([mesh.vertices.min(0), mesh.vertices.max(0)])
    rows = {p['key']: p for p in draft.document['parts']}
    spec, = posed_solids(draft.part(key), rows)
    base = draft.part(key)['model_3d']['base']
    if base['kind'] == 'box':
        body = _manifold(spec.shape, 64)
        assert body.volume() > 0
        np.testing.assert_allclose(bounds, np.asarray(body.bounding_box()).reshape(2, 3), atol=1e-8)
    else:
        # The STEP recipe retains an analytic ellipse, whereas the internal
        # CAD mesh helper only handles the reconstruction primitive subset.
        shape = spec.shape
        while shape['kind'] in {'affine', 'transform'}:
            shape = shape['shape']
        assert shape['kind'] == 'elliptic_cylinder'
        assert [shape[n] for n in ('width_mm', 'height_mm', 'length_mm')] == [
            base[n] for n in ('width_mm', 'height_mm', 'length_mm')]
        import manifold3d
        body = manifold3d.Manifold(manifold3d.Mesh64(
            np.array(mesh.vertices, dtype=np.float64, order='C', copy=True),
            np.array(mesh.faces, dtype=np.uint64, order='C', copy=True)))
        scale = draft.part(key).get('model_3d', {}).get('transform', {}).get('scale_xy', [1., 1.])
        expected = np.pi/4*base['width_mm']*base['height_mm']*base['length_mm']*np.prod(scale)
        assert body.volume() == pytest.approx(expected, rel=.002)
    return bounds


@pytest.mark.parametrize('host', (*APERTURES, *DEFLECTORS))
@pytest.mark.parametrize('kind', ('box', 'elliptic_cylinder'))
def test_copy_replacement_resizes_scales_saves_and_restores_recipe_atomically(host, kind, tmp_path):
    draft, key = _copy(host)
    initial = deepcopy(draft.part(key))
    other_parts = {p['key']: deepcopy(p) for p in draft.document['parts'] if p['key'] != key}
    config = dict(schema_version=1, base=dict(kind=kind, width_mm=8., height_mm=6., length_mm=.1))
    draft.set_model_3d(key, config)
    np.testing.assert_allclose(np.ptp(_bounds(draft, key), axis=0), [8., 6., .1], atol=1e-8)
    draft.set_dimension(('parts', key, 'model_3d', 'base', 'length_mm'), 5.)
    assert draft.part(key)['length_mm'] == 5.
    draft.update_model_3d(key, {'transform': {'scale_xy': [1.25, .75]}})
    np.testing.assert_allclose(np.ptp(_bounds(draft, key), axis=0), [10., 4.5, 5.], atol=1e-8)
    draft.set_dimension(('parts', key, 'length_mm'), .1)
    assert draft.part(key)['model_3d']['base']['length_mm'] == .1
    np.testing.assert_allclose(np.ptp(_bounds(draft, key), axis=0), [10., 4.5, .1], atol=1e-8)
    before = deepcopy(draft.document)
    history = draft.dirty, draft.can_undo, draft.can_redo
    with pytest.raises(ValueError):
        draft.set_model_3d(key, {'schema_version': 1, 'base': {'kind': 'existing'}})
    assert draft.document == before
    assert (draft.dirty, draft.can_undo, draft.can_redo) == history
    # Saved native dimensions remain editable metadata while their material is
    # replaced. Invalid native dimensions must matter only when restoring it.
    field = ('aperture_support_rod_thickness_mm' if host in APERTURES
             else 'deflector_support_radial_skin_thickness_mm')
    original = initial[field]
    active_bounds = _bounds(draft, key)
    draft.set_dimension(('parts', key, field), 1000.)
    np.testing.assert_allclose(_bounds(draft, key), active_bounds, atol=1e-8)
    target = tmp_path/f'{host}-{kind}.toml'
    draft.save_copy(target)
    reopened = PartModelDocument(target)
    reopened.validate()
    assert reopened.part(key)[field] == 1000.
    np.testing.assert_allclose(np.ptp(_bounds(reopened, key), axis=0), [10., 4.5, .1], atol=1e-8)
    reopened.set_dimension(('parts', key, 'length_mm'), initial['length_mm'])
    before = deepcopy(reopened.document)
    with pytest.raises(ValueError):
        reopened.set_model_3d(key, {'schema_version': 1, 'base': {'kind': 'existing'}})
    assert reopened.document == before
    reopened.set_dimension(('parts', key, field), original)
    reopened.set_model_3d(key, {'schema_version': 1, 'base': {'kind': 'existing'}})
    assert len(_model(reopened, key).meshes) == (3 if host in APERTURES else 20)
    if host in APERTURES:
        assert validate_aperture_support(reopened.part(key)) is not None
    else:
        assert support_dimensions(reopened.part(key)) is not None
    assert {p['key']: p for p in draft.document['parts'] if p['key'] != key} == other_parts


@pytest.mark.parametrize('host', APERTURES)
@pytest.mark.parametrize('kind', ('box', 'elliptic_cylinder'))
def test_aperture_replacement_tube_guard_uses_active_dimensions_and_rejects_atomically(host, kind):
    draft, key = _copy(host)
    z = draft.part(host)['local_center_z_mm']
    draft.set_model_3d(key, {'schema_version': 1,
        'base': {'kind': kind, 'width_mm': 8., 'height_mm': 6., 'length_mm': .1},
        'transform': {'offset_mm': [30., 0., z-2500.]}})
    # This inactive rod value would fail native construction, but must neither
    # disable collision checks on the replacement nor constrain its dimensions.
    draft.set_dimension(('parts', key, 'aperture_support_rod_thickness_mm'), 1000.)
    before = deepcopy(draft.document)
    history = draft.dirty, draft.can_undo, draft.can_redo
    for update in ({'transform': {'offset_mm': [0., 0., z-2500.]}},
                   {'transform': {'scale_xy': [6., 1.]}}):
        with pytest.raises(ValueError, match='outside the continuous tube'):
            draft.update_model_3d(key, update)
        assert draft.document == before
        assert (draft.dirty, draft.can_undo, draft.can_redo) == history
    # Both the original installed host and a detached copy remain protected.
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match='outside the continuous tube'):
        draft.set_model_3d(host, {'schema_version': 1,
            'base': {'kind': kind, 'width_mm': 8., 'height_mm': 6., 'length_mm': .1}})
    assert draft.document == before
