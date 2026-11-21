"""Installed CM12-style reconstruction, circuit ownership and material response."""
from dataclasses import replace
import json

import numpy as np
import pytest
from matplotlib.path import Path as Polygon

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.module_assembly import _module_vacuum_segments
from temsim.cad_mesh import _manifold
from temsim.cad_round_lenses import round_lens_solids
from temsim.magnetic_circuits import circuit_inventory
from temsim.module_manifest import read_document, validate_document
from temsim.optics.column import default_state
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.axisymmetric_magnetostatics import _part_mask
from temsim.physics.column_wall import clip_column_wall, _partition_vacuum_segments
from temsim.physics.lens_field_provider import lens_geometry_binding, resolve_runtime_lens_field_provider
from temsim.projector_iron import _sections_intersect


LENSES = ('diffraction_lens', 'intermediate_lens', 'projector_lens_1', 'projector_lens_2')
PLATES = ('dl_il_return_yoke', 'il_p1_return_yoke', 'p1_p2_return_yoke', 'p2_exit_return_yoke')
BASE = INSTRUMENT_CONFIG_ROOT/'project_and_recording_system'


@pytest.mark.parametrize('recording', ['No Energy Filter', 'Energy Filter'])
def test_return_plates_never_clip_at_the_context_bore_when_lens_is_displaced(recording):
    state, catalog = default_state(), AssemblyCatalog()
    assembly = catalog.apply(state, replace(catalog.default_selection(), recording=recording))
    assert all(assembly.part(key).data['axial_vacuum_context_only'] for key in PLATES)
    assert not set(PLATES).intersection(s.key for s in assembly.vacuum_bore_segments)

    # A saved manifest may predate the explicit context flag. Its mechanical
    # role still describes an iron plate outside the stationary 20 mm beam tube.
    parts = []
    for part in assembly.parts:
        if part.key in PLATES:
            data = dict(part.data)
            del data['axial_vacuum_context_only']
            part = replace(part, data=data)
        elif part.key == LENSES[0]:
            part = replace(part, data={**part.data, 'offset_x_mm': .1})
        parts.append(part)
    segments = []
    for module in assembly.modules:
        ref = next(p for p in parts if p.module_key == module.key)
        origin = ref.center_z_mm - ref.data['local_center_z_mm']
        segments.extend(_module_vacuum_segments(module, origin, parts))
    state._resolved_assembly = replace(assembly, parts=tuple(parts), vacuum_bore_segments=tuple(segments))
    stationary, placed = _partition_vacuum_segments(state, state._resolved_assembly.vacuum_bore_segments)
    assert not set(PLATES).intersection(s.key for s in stationary)
    assert not set(PLATES).intersection(s.key for s, _ in placed)

    z = assembly.part(PLATES[0]).center_z_mm + np.array([.3, 1.3])
    x = np.tile(np.r_[np.linspace(-9.98, -9.92, 1024), -10.02, 10.02]*1e-3, (2, 1))
    alive, stops, keys = clip_column_wall(state, z, x, np.zeros_like(x))
    assert alive[:-2].all() and np.isnan(stops[:-2]).all()
    assert not alive[-2:].any() and keys[-2:] == ['column_wall', 'column_wall']
    np.testing.assert_allclose(stops[-2:], z[0])


@pytest.mark.parametrize('name,recording,turns', [
    ('NoEnergyFilter', 'No Energy Filter', 648), ('EnergyFilter', 'Energy Filter', 720),
])
def test_installed_iron_contacts_bore_and_field_material_match_native_cad(name, recording, turns):
    doc = read_document(BASE/f'{name}.toml')
    validate_document(doc)
    rows = {row['key']: row for row in doc['parts']}
    state, catalog = default_state(), AssemblyCatalog()
    assembly = catalog.apply(state, replace(catalog.default_selection(), recording=recording))
    circuit = next(c for c in circuit_inventory(assembly.parts) if c.key == 'projector_iron_stack')
    assert circuit.channels == LENSES
    assert circuit.topology == 'shared_pole_multi_gap'
    assert set(circuit.body_keys) == set(PLATES) | {
        f'{key}_{part}' for key in LENSES for part in ('yoke', 'upper_pole', 'lower_pole')}
    assert {rows[key]['field_source_key'] for key in circuit.coil_keys} == set(LENSES)
    assert len(circuit.coil_keys) == 4
    assert rows['intermediate_lens_excitation_coil']['winding_turn_count'] == turns
    origin = assembly.part(LENSES[0]).center_z_mm - rows[LENSES[0]]['local_center_z_mm']
    binding = json.loads(lens_geometry_binding(state, LENSES[0]).canonical_geometry_json)
    payload = {row['key']: row for row in binding['lens_assembly']['parts']}
    sections = {}
    for key in circuit.body_keys:
        solid, = round_lens_solids(rows[key], rows)
        assert solid.material == 'soft_magnetic'
        assert any(url.endswith('273.pdf#page=6') for url in solid.evidence)
        mesh = _manifold(solid.shape, 32)
        assert mesh.status().name == 'NoError' and mesh.volume() > 0
        section = np.array(solid.shape['section_zr_mm'])
        sections[key] = section
        z0, z1 = section[:, 0].min(), section[:, 0].max()
        assert section[:, 1].min() >= 10.75
        # Non-boundary samples straddle every CAD contour segment, including
        # the two flares. The field payload uses the global microscope datum.
        z, r = np.meshgrid(np.linspace(z0-.031, z1+.027, 73), np.linspace(9.93, 85.17, 109))
        polygon = Polygon(section)
        inside = polygon.contains_points(np.c_[z.ravel(), r.ravel()]).reshape(z.shape)
        np.testing.assert_array_equal(_part_mask(payload[key], r*1e-3, (z+origin)*1e-3), inside)

    def beam_side(key, side):
        return sections[f'{key}_{side}_pole']

    for a, b, plate_key in zip(LENSES, LENSES[1:], PLATES[:3]):
        first, second = beam_side(a, 'lower'), beam_side(b, 'upper')
        plate = sections[plate_key]
        joint = first[:, 0].max()
        assert joint == second[:, 0].min()
        np.testing.assert_array_equal(sorted(first[first[:, 0] == joint, 1]),
                                      sorted(second[second[:, 0] == joint, 1]))
        assert first[first[:, 0] == joint, 1].max() == plate[:, 1].min() == 26.
        assert plate[:, 0].min() < joint < plate[:, 0].max()
        assert sections[a+'_yoke'][:, 0].max() == plate[:, 0].min()
        assert sections[b+'_yoke'][:, 0].min() == plate[:, 0].max()
        assert plate[:, 1].max() == sections[b+'_yoke'][:, 1].max()
    assert beam_side(LENSES[0], 'upper')[:, 1].max() == sections[LENSES[0]+'_yoke'][:, 1].min()
    exit_plate = sections[PLATES[-1]]
    assert exit_plate[:, 1].min() == beam_side(LENSES[-1], 'lower')[:, 1].max()
    assert exit_plate[:, 1].max() == sections[LENSES[-1]+'_yoke'][:, 1].min()
    for center, width, key in ((292.5, 6., LENSES[1]), (438.5, 40., LENSES[2]), (562.5, 7., LENSES[3])):
        assert beam_side(key, 'upper')[:, 0].max() == center - width/2
        assert beam_side(key, 'lower')[:, 0].min() == center + width/2
        assert not any(Polygon(s).contains_point((center, 22.)) for s in sections.values())


def test_four_installed_linear_field_bases_respond_to_iron_at_fixed_current():
    state = default_state()
    state.simulation_mode = 'custom'
    assembly = state._resolved_assembly
    # Tag the four plates in this test state to vary only their permeability
    # in the last control. All three cases have identical material boundaries.
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(p, data={**p.data, 'material_class': 'return_plate_control'}) if p.key in PLATES else p
        for p in assembly.parts))
    lenses = {lens.key: lens for lens in state.lenses}
    reference = state._resolved_assembly.part(LENSES[0]).center_z_mm - 82.5
    z = (np.linspace(32.5, 772.5, 741) + reference)*1e-3
    points = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
    basis = []
    for permeability, plate_mu in ((1., 1.), (100., 100.), (100., 1.)):
        settings = dict(solver='axisymmetric_linear_fem', relative_permeability=permeability,
                        material_permeabilities={'return_plate_control': plate_mu},
                        ampere_turns=100., radial_nodes=24, axial_nodes=64, padding_factor=2.)
        state.lens_field_map_descriptors = {key: dict(settings) for key in LENSES}
        fields = []
        for key in LENSES:
            provider = resolve_runtime_lens_field_provider(state, key, lenses[key])
            fields.append(provider.field_map.field_at_global_positions_t(points)[:, 2])
        basis.append(np.array(fields))
    air, iron, plates_at_vacuum_permeability = basis
    assert np.isfinite(basis).all()
    assert np.linalg.matrix_rank(iron) == 4
    assert np.all(np.linalg.norm(iron-air, axis=1) > .2*np.linalg.norm(air, axis=1))
    assert np.all(np.linalg.norm(iron-plates_at_vacuum_permeability, axis=1) > .01*np.linalg.norm(iron, axis=1))


def test_material_edit_invalidates_shared_fem_dependencies_and_preserves_fixed_gaussian_support():
    from temsim.calculation_cache import incident_field_dependencies

    state = default_state()
    state.simulation_mode = 'custom'
    settings = dict(solver='axisymmetric_linear_fem', relative_permeability=100.,
                    ampere_turns=100., radial_nodes=24, axial_nodes=64, padding_factor=2.)
    def dependencies():
        state.lens_field_map_descriptors = {}
        gaussian = incident_field_dependencies(state)
        state.lens_field_map_descriptors = {key: dict(settings) for key in LENSES}
        fields = {row['key']: row['field_geometry'] for row in incident_field_dependencies(state)['rows']
                  if row['key'] in LENSES}
        return gaussian, fields

    before, old_fields = dependencies()
    assembly = state._resolved_assembly
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(p, data={**p.data, 'pole_back_bore_diameter_mm': 43.5})
        if p.key == 'projector_lens_2_lower_pole' else p for p in assembly.parts))
    after, new_fields = dependencies()
    assert before == after
    assert all(old_fields[key] != new_fields[key] for key in LENSES)


@pytest.mark.parametrize('key,field,value,error', [
    ('projector_lens_2', 'pole_gap_center_offset_mm', -70., 'declared gap'),
    ('intermediate_lens_upper_pole', 'pole_back_bore_diameter_mm', 52., 'Flared pole'),
    ('intermediate_lens_upper_pole', 'mechanical_bore_diameter_mm', 20.5, 'intrudes into the vacuum liner'),
    ('p1_p2_return_yoke', 'mechanical_inner_diameter_mm', 50., 'overlap'),
])
def test_inconsistent_installed_iron_is_rejected(key, field, value, error):
    doc = read_document(BASE/'NoEnergyFilter.toml')
    next(row for row in doc['parts'] if row['key'] == key)[field] = value
    with pytest.raises(ValueError, match=error):
        validate_document(doc)


@pytest.mark.parametrize('key,field,value,defect', [
    ('dl_il_return_yoke', 'mechanical_inner_diameter_mm', 53., 'air gap 0.5 mm'),
    ('p1_p2_return_yoke', 'mechanical_inner_diameter_mm', 50., 'material overlap'),
    ('p2_exit_return_yoke', 'mechanical_outer_diameter_mm', 139., 'air gap 0.5 mm'),
])
def test_projector_plate_fit_is_checked_during_dimension_edit(key, field, value, defect):
    from copy import deepcopy
    from temsim.part_model_document import PartModelDocument

    editor = PartModelDocument(BASE/'NoEnergyFilter.toml')
    before = deepcopy(editor.document)
    with pytest.raises(ValueError, match=defect):
        editor.set_dimension(('parts', key, field), value)
    assert editor.document == before and not editor.can_undo


def test_intersecting_tapers_are_detected_between_disjoint_end_sections():
    rising = np.array([[0., 12., 13.], [10., 30., 31.]])
    falling = np.array([[0., 20., 22.], [10., 16., 18.]])
    assert _sections_intersect(rising, falling, 1e-9)
    assert not _sections_intersect(rising, rising + [0., 1., 1.], 1e-9)
