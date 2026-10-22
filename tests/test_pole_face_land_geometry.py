"""Pole face lands are material dimensions shared by native and CAD routes."""
from copy import deepcopy
import math

import numpy as np
import pytest

from temsim.cad_catalog import posed_solids
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.cad_round_lenses import round_lens_solids
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_3d import _pole_section, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.pole_geometry import validate_pole_face_land_geometry


FIELD = "pole_face_land_axial_thickness_mm"
MODULE = "project_and_recording_system/NoEnergyFilter.toml"
LENSES = ("diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2")


def _volume(section):
    # Exact volume of the surface of revolution of this piecewise linear wire.
    return abs(sum(math.pi / 3 * (b[0]-a[0]) * (a[1]**2+a[1]*b[1]+b[1]**2)
                   for a, b in zip(section, section[1:]+section[:1])))


@pytest.mark.parametrize("module", (MODULE, "project_and_recording_system/EnergyFilter.toml"))
@pytest.mark.parametrize("lens", LENSES)
@pytest.mark.parametrize("side", ("upper", "lower"))
def test_saved_land_has_correct_span_and_matches_native_cad_without_moving_faces(module, lens, side):
    doc = read_document(INSTRUMENT_CONFIG_ROOT/module)
    rows = {part["key"]: part for part in doc["parts"]}
    key = lens+"_"+side+"_pole"
    row = rows[key]
    before = deepcopy(doc)
    start, end = row["local_start_z_mm"], row["local_end_z_mm"]
    land, tip = row[FIELD], row["mechanical_tip_diameter_mm"]/2
    section = _pole_section(row, start, end)
    face = end if side == "upper" else start
    land_begin = face-land if side == "upper" else face+land
    assert (face, tip) in section and (land_begin, tip) in section
    assert min(r for z, r in section) == row["mechanical_bore_diameter_mm"]/2
    assert (min(z for z, r in section), max(z for z, r in section)) == (start, end)
    assert max(r for z, r in section) == row["mechanical_outer_diameter_mm"]/2

    legacy = {name: value for name, value in row.items() if name != FIELD}
    old = _pole_section(legacy, start, end)
    radius = row["mechanical_outer_diameter_mm"]/2
    expected_removed = land*math.pi/3*(radius**2+radius*tip-2*tip**2)
    assert _volume(old)-_volume(section) == pytest.approx(expected_removed, abs=1e-8)
    assert expected_removed == pytest.approx(1244.070690821558, abs=1e-8)
    specs = round_lens_solids(row, rows)
    assert specs[0].shape["section_zr_mm"] == [list(x) for x in section]
    native, = part_model_from_document(doc, key, include_children=False, angular_segments=32).meshes
    cad, = meshes_from_solids(specs, 32)
    vertices = lambda mesh: set(map(tuple, np.round(mesh.vertices, 9)))
    assert vertices(native) == vertices(cad)
    material = _manifold(specs[0].shape, 32)
    assert material.status().name == "NoError"
    assert material.volume() == pytest.approx(_volume(section)*math.sin(2*math.pi/32)/(2*math.pi/32))
    assert doc == before


@pytest.mark.parametrize("lens", LENSES)
def test_paired_public_edit_changes_material_only_and_survives_save_reopen(tmp_path, lens):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/MODULE)
    original = deepcopy(draft.document)
    expected = deepcopy(original)
    raw = draft.path.read_bytes()
    snapshots = {}
    for side in ("upper", "lower"):
        key = lens+"_"+side+"_pole"
        row = draft.part(key)
        snapshots[key] = _pole_section(row, row["local_start_z_mm"], row["local_end_z_mm"])
        draft.set_dimension(("parts", key, FIELD), 4.)
        next(p for p in expected["parts"] if p["key"] == key)[FIELD] = 4.
    draft.validate()
    assert draft.document == expected  # Includes all optical parents/calibrations.
    for key, old in snapshots.items():
        row = draft.part(key)
        assert _pole_section(row, row["local_start_z_mm"], row["local_end_z_mm"]) != old
    draft.save_copy(tmp_path/(lens+".toml"))
    reopened = PartModelDocument(draft.path)
    assert reopened.document == expected
    assert (INSTRUMENT_CONFIG_ROOT/MODULE).read_bytes() == raw


@pytest.mark.parametrize("value", (0., -1., 12., 13., float("nan"), float("inf"), True, 1 << 4096))
def test_invalid_land_is_rejected_before_draft_or_history_changes(value):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/MODULE)
    before, history = deepcopy(draft.document), deepcopy(draft._history)
    key = "diffraction_lens_upper_pole"
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", key, FIELD), value)
    assert draft.document == before and draft._history == history and not draft.can_undo
    invalid = deepcopy(before)
    for side in ("upper", "lower"):
        next(p for p in invalid["parts"] if p["key"] == "diffraction_lens_"+side+"_pole")[FIELD] = value
    with pytest.raises(ValueError):
        validate_document(invalid)


def test_explicit_profile_supersedes_land_and_missing_land_keeps_legacy_shape():
    rows = read_document(INSTRUMENT_CONFIG_ROOT/MODULE)["parts"]
    row = deepcopy(next(p for p in rows if p["key"] == "diffraction_lens_upper_pole"))
    start, end = row["local_start_z_mm"], row["local_end_z_mm"]
    row.pop(FIELD)
    assert _pole_section(row, start, end) == [(start, 10.75), (start, 26.),
                                              (end-12., 26.), (end, 20.), (end, 10.75)]
    row[FIELD] = 100.
    row["magnetic_radial_profile_mm"] = [[0., 10.75, 26.], [end-start, 10.75, 26.]]
    assert validate_pole_face_land_geometry(row) is None
    spec, = round_lens_solids(row, {})
    assert len(spec.shape["section_zr_mm"]) == 4


@pytest.mark.parametrize("side", ("upper", "lower"))
def test_detached_copy_placement_and_replacement_restore_are_atomic(side):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/MODULE)
    source = "diffraction_lens_"+side+"_pole"
    key = draft.copy_component_from(draft.document, source, "copied_pole", 2500., include_children=False)
    row = draft.part(key)
    first = np.array(_pole_section(row, row["local_start_z_mm"], row["local_end_z_mm"]))
    draft.place_component(key, 2600.)
    row = draft.part(key)
    moved = np.array(_pole_section(row, row["local_start_z_mm"], row["local_end_z_mm"]))
    np.testing.assert_allclose(moved-first, np.tile([100., 0.], (len(first), 1)))
    draft.update_model_3d(key, {"base": {"kind": "box", "width_mm": 10., "height_mm": 10., "length_mm": 1.}})
    assert validate_pole_face_land_geometry(draft.part(key)) is None
    specs = posed_solids(draft.part(key), {p["key"]: p for p in draft.document["parts"]})
    assert len(specs) == 1 and _manifold(specs[0].shape, 16).volume() == pytest.approx(100.)
    native, = part_model_from_document(draft.document, key, include_children=False).meshes
    np.testing.assert_allclose(np.ptp(native.vertices, axis=0), [10., 10., 1.])
    snapshot, history, index = deepcopy(draft.document), deepcopy(draft._history), draft._history_index
    with pytest.raises(ValueError, match="pole nose"):
        draft.set_model_3d(key, {"schema_version": 1, "base": {"kind": "existing"}})
    assert draft.document == snapshot and draft._history == history and draft._history_index == index
    draft.set_dimension(("parts", key, "model_3d", "base", "length_mm"), 32.)
    draft.set_model_3d(key, {"schema_version": 1, "base": {"kind": "existing"}})
    assert validate_pole_face_land_geometry(draft.part(key)) is not None


@pytest.mark.parametrize("lens", LENSES)
@pytest.mark.parametrize("side", ("upper", "lower"))
def test_fem_material_mask_matches_saved_polygon_including_terminal_land(lens, side):
    from matplotlib.path import Path
    from temsim.physics.axisymmetric_magnetostatics import _part_mask

    row = next(p for p in read_document(INSTRUMENT_CONFIG_ROOT/MODULE)["parts"]
               if p["key"] == lens+"_"+side+"_pole")
    start, end = row["local_start_z_mm"], row["local_end_z_mm"]
    polygon = Path(_pole_section(row, start, end))
    # Non-boundary samples independently classify the saved native/CAD polygon.
    z, radius = np.meshgrid(np.linspace(start+.013, end-.017, 103), np.linspace(10.731, 26.021, 71))
    expected = polygon.contains_points(np.c_[z.ravel(), radius.ravel()]).reshape(z.shape)
    wrapped = dict(key=row["key"], start_z_mm=start+1800., end_z_mm=end+1800., data=row)
    np.testing.assert_array_equal(_part_mask(wrapped, radius*1e-3, (z+1800.)*1e-3), expected)
    face = end if side == "upper" else start
    inward = -1 if side == "upper" else 1
    near_face = np.array([face+inward*1.5, face+inward*1.5])*1e-3
    local = {**wrapped, "start_z_mm": start, "end_z_mm": end}
    np.testing.assert_array_equal(_part_mask(local, np.array([19.9, 20.1])*1e-3, near_face), [True, False])
    legacy = {**local, "data": {k: v for k, v in row.items() if k != FIELD}}
    assert _part_mask(legacy, np.array([20.1])*1e-3, near_face[:1])[0]


def test_land_stales_only_own_map_while_analytic_field_and_optical_providers_are_unchanged():
    from dataclasses import replace
    from temsim.optics.column import default_state
    from temsim.physics.lens_field_provider import lens_geometry_binding

    state = default_state()
    z = np.linspace(0., 3500., 251)
    fields = {lens.key: lens.magnetic_field_t(z).copy() for lens in state.lenses if lens.key in LENSES}
    providers = deepcopy(state.lenses)
    before = {key: lens_geometry_binding(state, key).geometry_fingerprint for key in LENSES}
    assembly = state._resolved_assembly
    changed = {"diffraction_lens_upper_pole", "diffraction_lens_lower_pole"}
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(part, data={**part.data, FIELD: 4.}) if part.key in changed else part for part in assembly.parts))
    after = {key: lens_geometry_binding(state, key).geometry_fingerprint for key in LENSES}
    assert after["diffraction_lens"] != before["diffraction_lens"]
    assert all(after[key] == before[key] for key in LENSES if key != "diffraction_lens")
    for lens, previous in zip(state.lenses, providers):
        assert vars(lens) == vars(previous)
        if lens.key in fields:
            np.testing.assert_array_equal(lens.magnetic_field_t(z), fields[lens.key])


def test_fem_land_boundaries_enter_grid_and_change_recomputed_field_at_fixed_excitation():
    import json
    from types import SimpleNamespace
    from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_map

    rows = [p for p in read_document(INSTRUMENT_CONFIG_ROOT/MODULE)["parts"]
            if p.get("parent_key") == "diffraction_lens"]
    settings = dict(solver="axisymmetric_linear_fem", relative_permeability=100.,
                    ampere_turns=100., radial_nodes=24, axial_nodes=48, padding_factor=2.)
    positions = np.c_[np.zeros(201), np.zeros(201), np.linspace(.065, .100, 201)]
    fields, hashes = [], []
    for land in (3., 4.):
        parts = deepcopy(rows)
        for part in parts:
            if FIELD in part:
                part[FIELD] = land
        encoded = dict(lens_key="diffraction_lens", lens_assembly=dict(parts=[
            dict(key=p["key"], start_z_mm=p["local_start_z_mm"], end_z_mm=p["local_end_z_mm"], data=p)
            for p in parts]))
        field = solve_geometry_field_map(SimpleNamespace(canonical_geometry_json=json.dumps(encoded, sort_keys=True)), settings)
        for boundary in (80.5-land, 84.5+land, 80.5-12., 84.5+12.):
            assert np.any(np.isclose(field.axes_m[1], boundary*1e-3, atol=1e-14, rtol=0))
        fields.append(field.field_at_global_positions_t(positions))
        hashes.append(field.provenance.source_sha256)
    assert np.isfinite(fields).all()
    assert np.linalg.norm(fields[1]-fields[0]) > 1e-6
    assert hashes[0] != hashes[1]
    # This is a response regression at one mesh, not a convergence or OEM claim.


def test_land_impact_distinguishes_native_analytic_fields_from_selected_fem():
    from temsim.parameter_impact import describe_parameter_impact

    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/MODULE)
    key = "diffraction_lens_upper_pole"
    part, path = draft.part(key), ("parts", key, FIELD)
    rows = {p["key"]: p for p in draft.document["parts"]}
    for mode in ("ideal", "analytical"):
        result = describe_parameter_impact(part, path, by_key=rows, simulation_mode=mode)
        assert "geometry" in result.effects and "magnetic_field" not in result.effects
    result = describe_parameter_impact(part, path, by_key=rows, simulation_mode="custom",
        descriptors={"diffraction_lens": dict(solver="axisymmetric_linear_fem", relative_permeability=100.,
            ampere_turns=100., radial_nodes=24, axial_nodes=48, padding_factor=2.)})
    assert "geometry" in result.effects and "magnetic_field" in result.effects


def test_new_contour_identity_cannot_reuse_the_old_payload_at_unchanged_saved_land(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.physics import lens_field_provider as fields

    state = default_state()
    keys = (*LENSES, "condenser_lens_3", "objective_lens")
    current = {key: fields.lens_geometry_binding(state, key).geometry_fingerprint for key in keys}
    original = fields._part_geometry_data

    def before_land_implementation(part):
        result = original(part)
        result.pop("resolved_pole_face_land_model", None)
        return result

    # This reconstructs the previous payload: it already stored land=3 mm,
    # but the former material builders ignored that number.
    monkeypatch.setattr(fields, "_part_geometry_data", before_land_implementation)
    previous = {key: fields.lens_geometry_binding(state, key).geometry_fingerprint for key in keys}
    assert all(previous[key] != current[key] for key in LENSES)
    assert all(previous[key] == current[key] for key in keys if key not in LENSES)


def test_invalid_land_in_field_payload_without_a_source_row_key_is_a_value_error():
    from temsim.physics.axisymmetric_magnetostatics import _part_mask

    part = dict(key="upper", start_z_mm=0., end_z_mm=10., data=dict(
        mechanical_profile="magnetic_pole_piece", mechanical_outer_diameter_mm=20.,
        mechanical_tip_diameter_mm=8., mechanical_bore_diameter_mm=4.,
        pole_nose_axial_length_mm=4., pole_face_land_axial_thickness_mm=4.))
    with pytest.raises(ValueError, match="pole face land"):
        _part_mask(part, np.array([.003]), np.array([.009]))


@pytest.mark.parametrize("key, at_end, expected", [
    ("condenser_lens_1_lower_pole", True, [True, False]),
    ("condenser_lens_2_upper_pole", False, [False, True]),
])
@pytest.mark.parametrize("land", [None, 2.])
def test_joint_c1_c2_faces_match_cad_in_global_fem_coordinates(key, at_end, expected, land):
    from matplotlib.path import Path
    from temsim.physics.axisymmetric_magnetostatics import _part_mask

    doc = read_document(INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml')
    rows = {p['key']: p for p in doc['parts']}
    carrier = rows['c1_c2_pole_piece_cartridge']
    assert set(carrier['magnetic_lens_keys']) == {'condenser_lens_1', 'condenser_lens_2'}
    assert set(carrier['mechanical_cartridge_pole_keys']) == {
        'condenser_lens_1_lower_pole', 'condenser_lens_2_upper_pole'}
    row = deepcopy(rows[key])
    assert row['mechanical_container_key'] == carrier['key']
    if land is not None:
        row[FIELD] = land
    start, end = row['local_start_z_mm'], row['local_end_z_mm']
    shift = 450.
    wrapped = dict(key=key, start_z_mm=start+shift, end_z_mm=end+shift, data=row)
    # Independent fixed observations of the shared cartridge's two faces:
    # C1 narrows downstream, C2 narrows upstream, despite their optical names.
    samples = np.array([[start+1., 25.], [end-1., 25.]])
    np.testing.assert_array_equal(Path(_pole_section(row, start, end)).contains_points(samples), expected)
    np.testing.assert_array_equal(_part_mask(wrapped, samples[:, 1]*1e-3,
                                           (samples[:, 0]+shift)*1e-3), expected)
    z, r = np.meshgrid(np.linspace(start+.013, end-.017, 67), np.linspace(9.613, 30.021, 53))
    polygon = Path(_pole_section(row, start, end))
    expected_grid = polygon.contains_points(np.c_[z.ravel(), r.ravel()]).reshape(z.shape)
    np.testing.assert_array_equal(_part_mask(wrapped, r*1e-3, (z+shift)*1e-3), expected_grid)
    # A passive copy must keep the same source face direction after renaming.
    copy = {**row, 'key': 'renamed_copy', 'mechanical_part_role': 'custom_mechanical_copy',
            'geometry_template_key': key}
    assert _pole_section(copy, start, end) == _pole_section(row, start, end)
    face = end if at_end else start
    assert (face, row['mechanical_tip_diameter_mm']/2) in _pole_section(row, start, end)


def test_c1_c2_land_mesh_uses_the_shared_cartridge_face_directions():
    """Small material-domain solve only; not a calibrated column calculation."""
    import json
    from types import SimpleNamespace
    from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_map

    doc = read_document(INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml')
    channels = ('condenser_lens_1', 'condenser_lens_2')
    parts = [deepcopy(p) for p in doc['parts'] if p.get('parent_key') in channels
             and p.get('mechanical_profile') in {'magnetic_pole_piece', 'magnetic_excitation_coil'}]
    assert len(parts) == 4  # The same two pole faces and two current channels.
    for part in parts:
        if part['mechanical_profile'] == 'magnetic_pole_piece':
            part[FIELD] = 2.
    settings = dict(solver='axisymmetric_linear_fem', relative_permeability=50.,
                    ampere_turns=100., radial_nodes=12, axial_nodes=16, padding_factor=2.)
    encoded_parts = [dict(key=p['key'], start_z_mm=p['local_start_z_mm'],
                          end_z_mm=p['local_end_z_mm'], data=p) for p in parts]
    for channel in channels:
        geometry = dict(lens_key=channel, lens_assembly=dict(parts=encoded_parts))
        binding = SimpleNamespace(canonical_geometry_json=json.dumps(geometry, sort_keys=True))
        field = solve_geometry_field_map(binding, settings)
        # Face positions are 90/110 mm within the shared assembly. Their lands
        # and nose shoulders must enter the mesh on the inward material side.
        for boundary in (88., 71., 112., 132.8):
            assert np.any(np.isclose(field.axes_m[1], boundary*1e-3, atol=1e-14, rtol=0))
        assert all(np.isfinite(values).all() for values in field.components_t)


@pytest.mark.parametrize('fem', [False, True])
def test_corrected_c1_c2_orientation_invalidates_old_maps_without_changing_optics(monkeypatch, fem):
    import json
    from temsim.optics.column import default_state
    from temsim.physics import lens_field_provider as fields

    state = default_state()
    channels = ('condenser_lens_1', 'condenser_lens_2')
    keys = (*channels, 'condenser_lens_3', 'objective_lens', 'projector_lens_1')
    if fem:
        state.lens_field_map_descriptors = {key: dict(solver='axisymmetric_linear_fem', padding_factor=2.)
                                            for key in channels}
    z = np.linspace(450., 750., 31)
    values = {key: state.condenser_system[key].magnetic_field_t(z).copy() for key in channels}
    before_assembly = state._resolved_assembly
    current = {key: fields.lens_geometry_binding(state, key) for key in keys}
    for key in channels:
        payload = json.loads(current[key].canonical_geometry_json)['lens_assembly']
        parts = payload['parts']
        assert sum(p['key'] == 'c1_c2_pole_piece_cartridge' for p in parts) == 1
        all_parts = parts + payload.get('magnetostatic_neighbours', [])
        faces = {p['key']: p['data']['resolved_pole_face_at_end']
                 for p in all_parts if 'resolved_pole_face_at_end' in p['data']}
        assert faces[key+('_lower_pole' if key.endswith('_1') else '_upper_pole')] is key.endswith('_1')
        if fem:
            assert faces == {'condenser_lens_1_lower_pole': True, 'condenser_lens_2_upper_pole': False}
    original = fields._part_geometry_data

    def old_orientation_identity(part):
        data = original(part)
        data.pop('resolved_pole_face_at_end', None)
        return data

    monkeypatch.setattr(fields, '_part_geometry_data', old_orientation_identity)
    previous = {key: fields.lens_geometry_binding(state, key) for key in keys}
    assert all(previous[key].geometry_fingerprint != current[key].geometry_fingerprint for key in channels)
    assert all(previous[key] == current[key] for key in keys if key not in channels)
    assert state._resolved_assembly is before_assembly
    for key in channels:
        np.testing.assert_array_equal(state.condenser_system[key].magnetic_field_t(z), values[key])


@pytest.mark.parametrize('replacement', [False, True])
def test_orientation_identity_does_not_override_explicit_geometry(replacement):
    from types import SimpleNamespace
    from temsim.physics.lens_field_provider import _part_geometry_data

    data = dict(mechanical_profile='magnetic_pole_piece')
    if replacement:
        data['model_3d'] = {'base': {'kind': 'step'}}
    else:
        data['magnetic_radial_profile_mm'] = [[0., 10., 30.], [50., 10., 30.]]
    part = SimpleNamespace(key='condenser_lens_1_lower_pole', data=data)
    assert 'resolved_pole_face_at_end' not in _part_geometry_data(part)
