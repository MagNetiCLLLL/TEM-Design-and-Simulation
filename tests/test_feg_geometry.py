"""FEG geometry, electrode boundaries and CAD contracts; no finite-source trial.

Small arrays are synthetic chords or static field probes, not source populations.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim import module_manifest
from temsim.extractor_geometry import ExtractorContour
from temsim.gun_rear_geometry import annulus_contact_fractions, validate_rear_gun_document
from temsim.optics.column import default_state
from temsim.optics.electron_gun.electrostatic import accelerator_potential_rises_v
from temsim.optics.electron_gun.tip_assembly import model_from_part
from temsim.optics.electron_gun.tracing import _clip_body_bores
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.closed_gun_field import closed_field_request, electrode_boundary_arrays
from temsim.physics.grounded_tip_field import field_request, GroundedTipField


@pytest.fixture
def state():
    return default_state()



def test_shared_recess_floor_web_and_field_electrodes(state):
    gun = state.electron_gun
    contour = ExtractorContour.from_component(gun.extractor)
    np.testing.assert_allclose([contour.start_mm, contour.aperture_plane_mm, contour.end_mm,
                               contour.aperture_web_thickness_mm], [.3, .75, 1.3, .55], atol=1e-14)
    expected = np.array([[.3, .75, 2., 10.], [.75, 1.3, .25, 10.]])
    np.testing.assert_allclose(contour.annuli_mm, expected, rtol=0, atol=1e-14)
    flat = closed_field_request(gun)
    rings = [r for r in flat["rings"] if r["key"] == "extractor"]
    np.testing.assert_allclose([[r[k]*1000 for k in ("start_m", "stop_m", "inner_m", "outer_m")]
                               for r in rings], expected, atol=1e-14)
    assert [r["potential_rise_v"] for r in rings] == [4000., 4000.]
    gun.emitter.surface_model = model_from_part(state._resolved_assembly.part("feg_tip").data)
    curved = [r for r in field_request(gun)["rings"] if r[0] == "feg_extractor"]
    np.testing.assert_allclose(np.array([r[1:5] for r in curved])*1000, expected, atol=1e-14)
    assert [r[-1] for r in curved] == [4000., 4000.]



def test_recess_is_vacuum_and_aperture_shoulder_is_conducting(state):
    request = closed_field_request(state.electron_gun)
    # Include all ring endpoints to resolve every conductor. Extra coordinates
    # probe the recess vacuum, inner wall, shoulder and aperture web directly.
    radial = np.unique([0., .1e-3, .25e-3, 1e-3, 2e-3,
                       *[r[k] for r in request["rings"] for k in ("inner_m", "outer_m")]])
    axial = np.unique([0., .5e-3, .75e-3, 1e-3, request["domain"]["exit_m"],
                      *[r[k] for r in request["rings"] for k in ("start_m", "stop_m")]])
    fixed, values = electrode_boundary_arrays(request, radial, axial)
    def node(r, z):
        return np.flatnonzero(radial == r*1e-3)[0], np.flatnonzero(axial == z*1e-3)[0]
    assert not fixed[node(1., .5)]  # wide open recess
    assert not fixed[node(.1, 1.)]  # narrow through hole
    for point in ((2., .5), (1., .75), (.25, 1.)):
        assert fixed[node(*point)]
        assert values[node(*point)] == 4000.



def test_shared_clearance_and_frozen_contact_preserve_the_recess(state):
    from temsim.physics.column_wall import _independent_bore_constraints
    from temsim.physics.potential_geometry import capture_potential_geometry
    assembly = state._resolved_assembly
    rows = [r for r in assembly.vacuum_bore_segments if r.key == "feg_extractor"]
    np.testing.assert_allclose([(r.start_z_mm, r.end_z_mm, r.inner_diameter_mm) for r in rows],
                               [[.3, .75, 4.], [.75, 1.3, .5]], atol=1e-14)
    unreduced = _independent_bore_constraints(assembly, (), [assembly.part("feg_extractor")])
    rows = [r for r in unreduced if r.key == "feg_extractor"]
    np.testing.assert_allclose([(r.start_z_mm, r.end_z_mm, r.inner_diameter_mm) for r in rows],
                               [[.3, .75, 4.], [.75, 1.3, .5]], atol=1e-14)
    geometry = capture_potential_geometry(state, 0., .002)
    host = next(h for h in geometry.boundaries if h.host_key == "feg_extractor")
    np.testing.assert_allclose([(p.local_start_z_mm, p.local_end_z_mm, p.radius_mm) for p in host.pieces],
                               [[.3, .75, 2.], [.75, 1.3, .25]], atol=1e-14)



def test_crossing_chords_stop_at_recess_wall_shoulder_and_bore(state):
    from temsim.physics.potential_geometry import capture_potential_geometry
    from temsim.physics.unified_particle_contacts import contact
    gun = SimpleNamespace(extractor=state.electron_gun.extractor,
                          bore_components=[state.electron_gun.extractor])
    # Endpoints may lie beyond the whole electrode. Larger transverse slopes
    # meet real surfaces; no source angle is read by the intersection routine.
    start = np.array([[0, 0, 0], [0, 0, 0], [1, 0, .4], [1.9, 0, .4], [.3, 0, 1.5]], float)*1e-3
    end = np.array([[.1, 0, 2], [.5, 0, 2], [1, 0, 1.5], [2.1, 0, .6], [.3, 0, .5]], float)*1e-3
    alive, completed = np.ones(5, bool), np.zeros(5, bool)
    stop, keys = np.full(5, np.nan), [""]*5
    _clip_body_bores(gun, start, end, alive, completed, stop, keys)
    np.testing.assert_array_equal(alive, [True, False, False, False, False])
    np.testing.assert_allclose(stop[1:], [1., .75, .5, 1.3], rtol=0, atol=2e-14)
    assert keys == ["", *["feg_extractor"]*4]
    geometry = capture_potential_geometry(state, 0., .002)
    host = next(h for h in geometry.boundaries if h.host_key == "feg_extractor")
    fractions, labels = contact([(host.host_key, p) for p in host.pieces], start, end)
    np.testing.assert_array_equal(np.isfinite(fractions), ~alive)
    np.testing.assert_allclose((start[1:, 2]+fractions[1:]*(end[1:, 2]-start[1:, 2]))*1000,
                               stop[1:], rtol=0, atol=2e-14)
    np.testing.assert_array_equal(labels, keys)



def test_electrical_gauge_divider_and_independent_extraction(state):
    gun = state.electron_gun
    def potentials():
        request = closed_field_request(gun)
        return {r["key"]: r["potential_rise_v"] for r in request["rings"]}
    first = potentials()
    assert first["extractor"] == 4000.
    assert first["electrostatic_lens"] == 5200.
    assert first["accelerator:0"] == 34680.
    assert first["accelerator:9"] == 300000.
    gun.extractor.voltage_kv = 6.
    changed = potentials()
    assert changed["extractor"] == 6000.
    assert all(changed[k] == v for k, v in first.items() if k != "extractor")
    gun.accelerator.high_tension_kv = 200.
    changed = potentials()
    assert changed["accelerator:0"] == 24680.
    assert changed["accelerator:9"] == 200000.
    # Same physical gun-lens voltage expressed in another gauge is identical.
    gun.electrostatic_lens.voltage_reference = "ground"
    gun.electrostatic_lens.voltage_kv = -194.8
    for k, v in changed.items():
        assert potentials()[k] == pytest.approx(v)



def test_old_straight_bore_and_extractor_connected_divider_remain_explicit(state):
    contour = replace(ExtractorContour.from_component(state.electron_gun.extractor),
                      counterbore_depth_mm=0., counterbore_diameter_mm=0.)
    assert len(contour.annuli_mm) == 1
    assert len(contour.section_zr_mm) == 4
    accelerator = SimpleNamespace(high_tension_kv=300., stages=state.electron_gun.accelerator.stages)
    np.testing.assert_allclose(accelerator_potential_rises_v(accelerator, 4., 5200.),
                               4000+np.arange(1, 11)/10*296000)



@pytest.mark.parametrize("field,value", [("counterbore_depth_mm", 1.), ("counterbore_depth_mm", -.1),
    ("counterbore_diameter_mm", .5), ("counterbore_diameter_mm", 20.),
    ("counterbore_depth_mm", float("nan")), ("counterbore_diameter_mm", float("inf")),
    ("counterbore_depth_mm", True), ("counterbore_depth_mm", "0.45")])
def test_invalid_recess_rejected(field, value, state):
    with pytest.raises(ValueError):
        replace(ExtractorContour.from_component(state.electron_gun.extractor), **{field: value})



@pytest.mark.parametrize("base", [-1., 300000., float("nan")])
def test_invalid_divider_start_rejected(state, base):
    with pytest.raises(ValueError, match="divider start"):
        state.electron_gun.accelerator.stage_potential_rises_v(4., base)



def test_recess_edit_persistence_cache_and_runtime_ownership(tmp_path, state):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.shared_tip import copy_catalog_tree
    from temsim.manifest_editor import ManifestEditor, ManifestTarget
    from temsim.column.state_layout import layout_configuration_from_state
    from temsim.runtime_parameters import runtime_targets, editable_parameters
    root = tmp_path/"instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    catalog = AssemblyCatalog(root)
    catalog.apply(state, catalog.default_selection(), preserve_operating_parameters=True)
    before = closed_field_request(state.electron_gun)
    ManifestEditor(root).save(ManifestTarget("gun/FEG.toml", "feg_extractor"), {
        ("parts", "feg_extractor", "counterbore_depth_mm"): .35,
        ("parts", "feg_extractor", "counterbore_diameter_mm"): 3.,
    }, layout_configuration_from_state(state))
    catalog.apply(state, catalog.default_selection(), preserve_operating_parameters=True)
    assert state.electron_gun.extractor.counterbore_depth_mm == .35
    assert state.electron_gun.extractor.optical_reference_from_tip_mm == pytest.approx(.65)
    from temsim.column.module_assembly import _absolute_part_value
    assert _absolute_part_value(state._resolved_assembly.part("feg_extractor"), "optical_reference_local_z_mm") == pytest.approx(.65)
    assert closed_field_request(state.electron_gun) != before
    rows = [s for s in state._resolved_assembly.vacuum_bore_segments if s.key == "feg_extractor"]
    np.testing.assert_allclose([(s.start_z_mm, s.end_z_mm, s.inner_diameter_mm) for s in rows],
                               [[.3, .65, 3.], [.65, 1.3, .5]], atol=1e-14)
    targets = runtime_targets(state)
    assert not {"counterbore_depth_mm", "counterbore_diameter_mm"} & {
        p.name for p in editable_parameters(targets["feg_extractor"])}
    assert "voltage_start_reference" not in {p.name for p in editable_parameters(targets["feg_accelerator"])}



def test_new_geometry_cannot_silently_use_the_analytic_field(state):
    from temsim.optics.electron_gun.electrostatic import FegElectrostaticField
    gun = state.electron_gun
    with pytest.raises(ValueError, match="joint geometry field"):
        FegElectrostaticField(gun.emitter, gun.extractor, gun.electrostatic_lens, gun.accelerator)



@pytest.mark.parametrize("updates", [{"voltage_start_reference": "floating"},
                                    {"default_high_tension_kv": 30., "voltage_start_reference": "gun_lens"}])
def test_invalid_electrical_connection_is_rejected_by_assembly(updates):
    from temsim.optics.electron_gun.tip_assembly import validate_electrical_defaults
    parts = [{"key": "feg_extractor", "default_voltage_kv": 4.},
             {"key": "feg_electrostatic_lens", "default_voltage_kv": 40., "default_voltage_reference": "tip"},
             {"key": "feg_accelerator", **updates}]
    with pytest.raises(ValueError, match="reference|divider start"):
        validate_electrical_defaults(parts)



def test_source_dialog_displays_lab_potentials_and_divider_connection(qtbot, state):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    dialog = GunSourceDialog(state.electron_gun)
    qtbot.addWidget(dialog)
    text = dialog.surface_voltage.text()
    for expected in ("Tip -300 kV", "Extractor -296 kV", "Gun lens -294.8 kV",
                     "starts at gun_lens", "first ring -265.32 kV", "final ring 0 kV"):
        assert expected in text



def test_extractor_reference_markers_identify_the_actual_contour_controls(state):
    from temsim.parameter_semantics import describe_parameter
    from temsim.parameter_impact import describe_parameter_impact
    part = state._resolved_assembly.part("feg_extractor").data
    plane = describe_parameter(part, ("parts", "feg_extractor", "optical_reference_local_z_mm"))
    bore = describe_parameter(part, ("parts", "feg_extractor", "vacuum_inner_diameter_mm"))
    assert plane.label == "Extraction plane reference" and plane.category == "placement"
    assert bore.category == "vacuum"
    assert "bore_diameter_mm" in bore.description
    for name in ("optical_reference_local_z_mm", "vacuum_inner_diameter_mm"):
        assert describe_parameter_impact(part, ("parts", "feg_extractor", name)).status == "inactive"



def test_layout_source_distance_uses_apex_instead_of_shank_center(state):
    from temsim.column.state_layout import layout_configuration_from_state
    from temsim.column.layout import build_optics_layout
    layout = build_optics_layout(layout_configuration_from_state(state))
    assert layout.source_to_sample_mm == pytest.approx(state._resolved_assembly.part("sample").center_z_mm)
    source = next(part for part in layout if part.key == "feg_tip")
    assert source.local_s_center_mm-layout.source_to_sample_mm == pytest.approx(3.)



@pytest.mark.parametrize("name", ["FEG", "FEG_Mono"])
def test_negative_rear_extent_preserves_source_reference_and_real_solids(name):
    from temsim.cad_catalog import part_solids
    from temsim.part_model_3d import part_model_from_document
    doc = module_manifest.read_document(INSTRUMENT_CONFIG_ROOT/f"gun/{name}.toml")
    module_manifest.validate_document(doc)
    assert doc["ports"]["entrance"]["local_z_mm"] == 0.
    assert doc["geometry"]["parts_start_z_mm"] == -6.
    tip = next(p for p in doc["parts"] if p["key"] == "feg_tip")
    assert tip["local_end_z_mm"] == tip["optical_reference_local_z_mm"] == 0.
    assert tip["default_tip_emission"] == "flat_tip"
    parts = {part["key"]: part for part in doc["parts"]}
    for part in doc["parts"]:
        if not part.get("rear_gun_role"):
            continue
        mesh = part_model_from_document(doc, part["key"]).meshes[0]
        assert mesh.vertices[:, 2].min() == pytest.approx(part["local_start_z_mm"])
        assert mesh.vertices[:, 2].max() == pytest.approx(part["local_end_z_mm"])
        radii = np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])
        assert radii.min() == pytest.approx(part["bore_diameter_mm"]/2)
        assert radii.max() == pytest.approx(part["outer_diameter_mm"]/2)
        assert mesh.surfaces
        material = "insulator" if part["rear_gun_role"] == "insulator" else "electrode"
        assert mesh.material_class == material
        assert {solid.material for solid in part_solids(part, parts)} == {material}
        for surface in mesh.surfaces.values():
            assert surface["kind"] == "cad_material"
            assert ("parts", part["key"], "bore_diameter_mm") in surface["parameter_paths"]



def test_flat_policy_and_curved_electrical_references(state):
    gun = state.electron_gun
    assert gun.emitter.surface_model is None
    flat = closed_field_request(gun)
    assert flat["domain"]["entrance_m"] == 0.
    assert flat["rear_gun"]["field_status"] == "omitted_by_infinite_planar_cathode"
    assert flat["rear_gun"]["rings"] == []
    gun.emitter.surface_model = model_from_part(state._resolved_assembly.part("feg_tip").data)
    request = field_request(gun)
    rear = request["rear_gun"]
    assert len(rear["rings"]) == 2
    assert all(r["potential_rise_v"] == 0. for r in rear["rings"])
    assert rear["tip_connected_keys"] == ["feg_cathode_holder"]
    # A different connection must enter the actual boundary/cache request.
    gun._gun_rear_parts[0]["voltage_reference"] = "extractor"
    gun._gun_rear_parts[0]["bias_kv"] = -.5
    changed = field_request(gun)
    assert changed != request
    assert changed["rear_gun"]["rings"][0]["potential_rise_v"] == 3500.



def test_continuous_curvature_uses_the_same_rear_electrodes(state):
    gun = state.electron_gun
    gun.emitter.curvature_nm_inv = .01
    request = closed_field_request(gun)
    assert request["domain"]["entrance_m"] == pytest.approx(-.006)
    assert request["rear_gun"]["field_status"] == "active_curved_conductors"
    assert any(r["key"] == "feg_emitter_enclosure" for r in request["rings"])



def test_static_eee_bias_changes_vacuum_field_with_other_electrodes_fixed(state):
    gun = state.electron_gun
    model = model_from_part(state._resolved_assembly.part("feg_tip").data)
    gun.emitter.surface_model = replace(model, field_numerics=replace(model.field_numerics,
        radial_nodes=64, axial_nodes=128, electrode_cells_per_bore=0))
    request = field_request(gun)
    base = GroundedTipField(request)
    gun._gun_rear_parts[0]["bias_kv"] = .25
    changed = GroundedTipField(field_request(gun))
    # Same mesh/geometry; linear Laplace maximum principle bounds the response.
    np.testing.assert_array_equal(base.r, changed.r)
    np.testing.assert_array_equal(base.z, changed.z)
    diff = changed.rise-base.rise
    assert diff.min() >= -1e-5 and diff.max() <= 250.+1e-5
    points = np.array([[.2, 0., -1.65], [0., 0., .001]])*1e-3
    delta = changed.potential_v_at_global_positions(points)-base.potential_v_at_global_positions(points)
    assert np.all(delta > 0.) and np.all(delta < 250.)
    assert not np.array_equal(changed.field_at_global_positions_v_per_m(points),
                              base.field_at_global_positions_v_per_m(points))
    for field, expected in ((base, 0.), (changed, 250.)):
        r = np.flatnonzero(field.r == request["rear_gun"]["rings"][0]["inner_m"])[0]
        z = np.flatnonzero(field.z == request["rear_gun"]["rings"][0]["stop_m"])[0]
        assert field.rise[r, z] == expected
        assert field.report["linear_residual"] < 1e-9



def test_rear_edits_save_reload_and_invalid_origin_preserves_draft(tmp_path):
    path = tmp_path/"gun.toml"
    PartModelDocument(INSTRUMENT_CONFIG_ROOT/"gun/FEG.toml").save_copy(path)
    draft = PartModelDocument(path)
    draft.part("feg_emitter_enclosure")["rear_bias_kv"] = .25
    draft.set_dimension(("parts", "feg_emitter_enclosure", "bore_diameter_mm"), .8)
    draft.save()
    loaded = PartModelDocument(path)
    assert loaded.part("feg_emitter_enclosure")["rear_bias_kv"] == .25
    before = deepcopy(loaded.document)
    with pytest.raises(ValueError, match="clear"):
        loaded.set_dimension(("parts", "feg_emitter_enclosure", "bore_diameter_mm"), .01)
    assert loaded.document == before
    before["ports"]["entrance"]["local_z_mm"] = -6.
    with pytest.raises(ValueError, match="tip Z=0"):
        validate_rear_gun_document(before)



def test_assembly_controls_update_live_rear_voltage_and_clearance(tmp_path, state):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.shared_tip import copy_catalog_tree
    from temsim.manifest_editor import ManifestEditor, ManifestTarget
    from temsim.column.state_layout import layout_configuration_from_state
    root = tmp_path/"instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    catalog = AssemblyCatalog(root)
    catalog.apply(state, catalog.default_selection(), preserve_operating_parameters=True)
    key = "feg_emitter_enclosure"
    ManifestEditor(root).save(ManifestTarget("gun/FEG.toml", key), {
        ("parts", key, "rear_voltage_reference"): "extractor",
        ("parts", key, "rear_bias_kv"): -.5,
        ("parts", key, "bore_diameter_mm"): .8,
    }, layout_configuration_from_state(state))
    catalog.apply(state, catalog.default_selection(), preserve_operating_parameters=True)
    gun = state.electron_gun
    gun.emitter.surface_model = model_from_part(state._resolved_assembly.part("feg_tip").data)
    rear = field_request(gun)["rear_gun"]["rings"][0]
    assert rear["potential_rise_v"] == 3500.
    assert rear["inner_m"] == .0004
    segment = next(s for s in state._resolved_assembly.vacuum_bore_segments if s.key == key)
    assert segment.inner_diameter_mm == .8



def test_rear_contact_finds_front_back_and_cylinder_without_infinite_outer_plate(state):
    row = state.electron_gun._gun_rear_parts[0]
    start = np.array([[.5, 0, -1.], [.5, 0, -2.], [.2, 0, -1.65],
                      [11., 0, -2.], [0, 0, -2.], [11, 0, -1.65]])
    end = np.array([[.5, 0, -2.], [.5, 0, -1.], [.4, 0, -1.65],
                    [11., 0, -1.], [0, 0, -1.], [9, 0, -1.65]])
    hit = annulus_contact_fractions(start, end, row)
    np.testing.assert_allclose(hit, [.5, .2, .5, np.inf, np.inf, .5], atol=1e-12)
    gun = SimpleNamespace(bore_components=[], _gun_rear_parts=(row,))
    alive, completed = np.ones(len(start), bool), np.zeros(len(start), bool)
    stops, keys = np.full(len(start), np.nan), [""]*len(start)
    _clip_body_bores(gun, start*.001, end*.001, alive, completed, stops, keys)
    np.testing.assert_array_equal(alive, ~np.isfinite(hit))
    assert all(keys[i] == row["key"] for i in (0, 1, 2, 5))



def test_whole_extractor_geometric_limit_is_back_face_not_entrance(state):
    contour = ExtractorContour.from_component(state.electron_gun.extractor)
    acceptance = contour.straight_ray_acceptance()
    assert acceptance["half_angle_deg"] == pytest.approx(10.885527054658736)
    assert acceptance["entrance_half_angle_deg"] == pytest.approx(18.43494882292201)
    angles = np.deg2rad([10.88, 10.89, 18.])
    start = np.zeros((3, 3)); end = np.c_[2*np.tan(angles), np.zeros(3), np.full(3, 2.)]*.001
    gun = SimpleNamespace(extractor=state.electron_gun.extractor, bore_components=[state.electron_gun.extractor])
    alive, completed = np.ones(3, bool), np.zeros(3, bool)
    _clip_body_bores(gun, start, end, alive, completed, np.full(3, np.nan), [""]*3)
    np.testing.assert_array_equal(alive, [True, False, False])

