"""Condenser support topology, open passages and unchanged optical ownership."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.cad_catalog import part_solids
from temsim.cad_geometry import annulus, box, transformed
from temsim.cad_mesh import _manifold
from temsim.cad_stigmator_support import (
    DIMENSIONS, PREFIX, stigmator_support_dimensions, stigmator_support_solids,
)
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = 'column/C3_ProbeCorrector.toml'
KEY = 'condenser_stigmator'


@pytest.fixture
def document():
    return read_document(INSTRUMENT_CONFIG_ROOT/MODULE)


def _row(document):
    return next(row for row in document['parts'] if row['key'] == KEY)


def test_condenser_support_preserves_eight_copper_leaves_and_one_effective_field(document):
    from temsim.cad_multipoles import _stigmator_windings

    part = _row(document)
    before = deepcopy(part)
    copper = _stigmator_windings(part)
    specs = stigmator_support_solids(part)
    assert specs[:8] == copper and len(specs) == 33
    assert {s.region for s in specs} == {f'coil_{i}' for i in range(1, 9)} | {'slotted_carrier'} | {
        f'coil_{i}_{suffix}' for i in range(1, 9) for suffix in ('insulation', 'former', 'adhesive_seat')}
    assert part == before
    assert part['effective_length_mm'] == 6. and part['local_center_z_mm'] == 1001.5
    assert {s.parameters['coil_group'] for s in copper} == {'x', 'y'}
    assert all(s.parameters['optical_field_unchanged'] for s in specs[8:])
    assert all('not OEM' in s.note for s in specs[8:])
    legacy = {name: value for name, value in part.items() if not name.startswith(PREFIX)}
    assert stigmator_support_dimensions(legacy) is None and stigmator_support_solids(legacy) is None
    assert _stigmator_windings(legacy) == copper


def test_material_is_closed_nonoverlapping_and_all_winding_windows_and_beam_bore_stay_open(document):
    part = _row(document)
    d = stigmator_support_dimensions(part)
    specs = stigmator_support_solids(part)
    solids = [_manifold(spec.shape, 48) for spec in specs]
    for index, solid in enumerate(solids):
        assert solid.volume() > 0
        # Enclosing insulation has an outward surface and an inward cavity
        # boundary. Manifold.decompose separates these oriented surfaces; it
        # does not count connected material solids.
        shells = solid.decompose()
        if specs[index].region.endswith('_insulation'):
            assert len(shells) == 2
            signed = sorted(shell.volume() for shell in shells)
            copper_index = int(specs[index].region.split('_')[1])-1
            assert signed[0] == pytest.approx(-solids[copper_index].volume())
            assert sum(signed) == pytest.approx(solid.volume())
        else:
            assert len(shells) == 1
        mesh = solid.to_mesh64()
        edges = np.sort(np.concatenate((mesh.tri_verts[:, [0, 1]],
            mesh.tri_verts[:, [1, 2]], mesh.tri_verts[:, [2, 0]])), axis=1)
        assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
        for other in solids[index+1:]:
            assert (solid ^ other).volume() < 1e-8
    beam = _manifold(annulus(d['start_mm']-1, d['end_mm']+1, 0., d['inner_radius_mm']), 48)
    assert all((solid ^ beam).volume() < 1e-8 for solid in solids)
    ri, ro = d['pocket_inner_mm'], d['pocket_outer_mm']
    t = d['insulation_thickness_mm']+d['former_wall_thickness_mm']
    window = box(((ri+ro)/2, 0., d['centre_mm']),
        (ro-ri, d['window_width_mm']-2*t, d['window_length_mm']-2*t))
    for winding in specs[:8]:
        passage = _manifold(transformed(window, rotation=winding.shape['rotation']), 48)
        assert all((solid ^ passage).volume() < 1e-8 for solid in solids)


def test_native_and_cad_share_regions_material_and_editable_dimensions(document):
    rows = {r['key']: r for r in document['parts']}
    specs = part_solids(rows[KEY], rows)
    model = part_model_from_document(document, KEY, include_children=False)
    assert len(specs) == len(model.meshes) == 33
    assert {s.region for s in specs} == {m.region for m in model.meshes}
    assert all(np.isfinite(mesh.vertices).all() and not mesh.is_exact for mesh in model.meshes)
    fields = {spec.path[-1]: spec for spec in part_dimension_specs(document, KEY)}
    for name in DIMENSIONS:
        assert fields[name].editable and fields[name].unit == 'mm'
    for mesh in model.meshes[8:]:
        paths = {tuple(path) for surface in mesh.surfaces.values() for path in surface['parameter_paths']}
        assert {('parts', KEY, name) for name in DIMENSIONS} <= paths


@pytest.mark.parametrize('changes', [
    {PREFIX+'insulation_thickness_mm': False}, {PREFIX+'insulation_thickness_mm': 0.},
    {PREFIX+'former_wall_thickness_mm': float('nan')}, {PREFIX+'adhesive_thickness_mm': float('inf')},
    {PREFIX+'former_wall_thickness_mm': 3.}, {PREFIX+'adhesive_thickness_mm': 5.},
    {PREFIX+'profile': 'unverified'}, {PREFIX+'geometry_status': 'OEM'},
    {PREFIX+'source': 'other'}, {PREFIX+'source_note': ''}, {PREFIX+'unknown_mm': 1.},
    {'channel_y_angle_deg': 1.}, {'coil_count': 12}, {'layout_role': 'control_channel'},
])
def test_invalid_support_cannot_hide_behind_legacy_rendering_or_manifest(document, changes):
    _row(document).update(changes)
    for validate in (stigmator_support_dimensions, stigmator_support_solids):
        with pytest.raises(ValueError):
            validate(_row(document))
    with pytest.raises(ValueError):
        validate_document(document)


def test_support_edit_is_atomic_undoable_and_does_not_change_copper_or_field_inputs(tmp_path):
    from temsim.shared_tip import copy_catalog_inputs

    root = tmp_path/'instruments'
    copy_catalog_inputs(INSTRUMENT_CONFIG_ROOT, root, (MODULE,))
    draft = PartModelDocument(root/MODULE)
    before, raw = deepcopy(draft.document), draft.path.read_bytes()
    old = stigmator_support_solids(draft.part(KEY))
    field = PREFIX+'former_wall_thickness_mm'
    with pytest.raises(ValueError):
        draft.set_dimension(('parts', KEY, field), 3.)
    assert draft.document == before and not draft.dirty and not draft.can_undo
    draft.set_dimension(('parts', KEY, field), .35)
    new = stigmator_support_solids(draft.part(KEY))
    assert old[:8] == new[:8]
    assert any(a.shape != b.shape for a, b in zip(old[8:], new[8:]))
    assert {k: v for k, v in draft.part(KEY).items() if not k.startswith(PREFIX)} == {
        k: v for k, v in _row(before).items() if not k.startswith(PREFIX)}
    assert draft.path.read_bytes() == raw
    draft.undo()
    assert draft.document == before
    draft.redo()
    destination = tmp_path/'independent-stigmator.toml'
    draft.save_copy(destination)
    assert PartModelDocument(destination).part(KEY)[field] == .35


def test_rotated_channel_bases_rotate_their_own_supports_at_the_same_station(document):
    part = _row(document)
    old = stigmator_support_solids(part)
    rotated = {**part, 'channel_x_angle_deg': 17., 'channel_y_angle_deg': 62.}
    new = stigmator_support_solids(rotated)
    assert [s.region for s in old] == [s.region for s in new]
    for a, b in zip(old, new):
        if a.region == 'slotted_carrier':
            assert a.shape['base'] == b.shape['base']
            continue
        assert a.shape['shape'] == b.shape['shape']
        assert a.parameters.get('coil_group') == b.parameters.get('coil_group')
    assert stigmator_support_dimensions(rotated)['centre_mm'] == 1001.5


def test_independent_copy_replacement_owns_geometry_and_restore_is_atomic():
    from temsim.cad_catalog import posed_solids

    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/MODULE)
    original = deepcopy(draft.part(KEY))
    copied = draft.copy_component_from(draft.document, KEY, 'replaced_condenser_stigmator',
                                       2500., include_children=False)
    copied_native = deepcopy(draft.document)
    draft.update_model_3d(copied, {'base': {'kind': 'box', 'width_mm': 10.,
                                          'height_mm': 10., 'length_mm': 1.}})
    validate_document(draft.document)
    assert stigmator_support_dimensions(draft.part(copied)) is None
    assert stigmator_support_solids(draft.part(copied)) is None
    model = part_model_from_document(draft.document, copied, include_children=False)
    assert len(model.meshes) == 1
    np.testing.assert_allclose(np.ptp(model.meshes[0].vertices, axis=0), [10., 10., 1.])
    solids = posed_solids(draft.part(copied), {p['key']: p for p in draft.document['parts']})
    assert len(solids) == 1 and _manifold(solids[0].shape, 32).volume() == pytest.approx(100.)
    assert draft.part(KEY) == original
    before, history = deepcopy(draft.document), deepcopy(draft._history)
    history_index = draft._history_index
    native = deepcopy(draft.part(copied)['model_3d'])
    native['base'] = {'kind': 'existing'}
    with pytest.raises(ValueError, match='winding/former windows'):
        draft.set_model_3d(copied, native)
    assert draft.document == before and draft._history == history
    assert draft._history_index == history_index
    draft.undo()
    assert draft.document == copied_native
    assert len(part_model_from_document(draft.document, copied, include_children=False).meshes) == 33
    draft.redo()
    assert draft.document == before


def test_condenser_deflector_physical_coil_depth_is_not_reset_by_effective_field_length(document):
    from temsim.cad_multipoles import _deflector_solids

    part = next(p for p in document['parts'] if p['key'] == 'condenser_deflector')
    assert part['mechanical_coil_length_mm'] == 10.
    old = _deflector_solids(part)
    changed = _deflector_solids({**part, 'effective_thickness_mm': 5.})
    assert len(old) == len(changed) == 20
    assert old == changed


def test_installed_material_uses_real_profiled_tube_and_retains_positive_clearance():
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.cad_catalog import posed_solids
    from temsim.cad_stage_holder import physical_liner_rows
    from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, PROFILE as TUBE_PROFILE, tube_solids
    from temsim.optics.column import default_state
    from temsim.ordinary_deflector_geometry import support_dimensions

    state = default_state()
    assembly = AssemblyCatalog().apply(state, AssemblySelection('FEG', 'C3 + Probe Corrector', 'No Energy Filter'))
    tube_row = next(row for row in physical_liner_rows(assembly) if row['key'] == CONTINUOUS_TUBE_KEY)
    tube, = tube_solids(tube_row)
    tube_material = _manifold(tube.shape, 72)
    profile = np.asarray(tube_row[TUBE_PROFILE])
    assert tube_row['mechanical_inner_diameter_mm'] == 5.76
    rows = {p.key: p.data for p in assembly.parts}
    for key, expected_gap, expected_z in (('condenser_deflector', 2.9, (790., 830.)),
                                         (KEY, .4, (1436.5, 1466.5))):
        part = assembly.part(key)
        before = deepcopy(dict(part.data))
        assert (part.start_z_mm, part.end_z_mm) == expected_z
        offset = part.center_z_mm-part.data['local_center_z_mm']
        d = support_dimensions(part.data) if key.endswith('deflector') else stigmator_support_dimensions(part.data)
        inner = d['material_inner_radius_mm'] if key.endswith('deflector') else d['inner_radius_mm']
        # Include all profile knots inside the complete installed material span,
        # not just the nominal mechanical bore or the mid-plane OD.
        sample_z = np.r_[part.start_z_mm, profile[(profile[:, 0] > part.start_z_mm)
            & (profile[:, 0] < part.end_z_mm), 0], part.end_z_mm]
        max_od = np.interp(sample_z, profile[:, 0], profile[:, 1]).max()
        assert inner-max_od/2 == pytest.approx(expected_gap)
        for spec in posed_solids(part.data, rows):
            placed = transformed(spec.shape, offset=(0., 0., offset))
            assert (_manifold(placed, 72) ^ tube_material).volume() < 1e-8
        assert dict(part.data) == before
    # Per-part solver constraints are distinct from the continuous material ID.
    # The condenser deflector's 5 mm limit remains active in the resolved domain.
    assert rows['condenser_deflector']['vacuum_inner_diameter_mm'] == 5.
    assert rows[KEY]['vacuum_inner_diameter_mm'] == 20.
    assert state.condenser_stigmator.length_mm == 6.
    assert state.descan_deflector._physical_host is state.image_diffraction_deflector
