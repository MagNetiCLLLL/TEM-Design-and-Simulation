"""Configured recording shapes remain observable without changing fields."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.field_display_regions import build_field_display_regions
from temsim.module_manifest import read_document
from temsim.part_model_3d import part_model_from_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from test_field_display_regions import _state
from test_field_vector_page import Scene, make_source, wait_sample


KEYS = ("camera", "flu_screen", "haadf", "df", "bf")


@pytest.fixture(scope="module", params=("NoEnergyFilter", "EnergyFilter"))
def recording_document(request):
    return read_document(INSTRUMENT_CONFIG_ROOT/f"project_and_recording_system/{request.param}.toml")


@pytest.mark.parametrize("key", KEYS)
def test_recording_shapes_use_saved_footprint_and_include_inner_bores(recording_document, key):
    row = next(deepcopy(p) for p in recording_document["parts"] if p["key"] == key)
    before = deepcopy(row)
    state = _state([row], shift=987.)
    index = build_field_display_regions(state, ())
    region, = index.regions_for_component(key)
    width, centre = row["outer_width_mm"], row["local_center_z_mm"]+987.
    np.testing.assert_allclose(region.bounds_m,
        [[-width/2*.001, -width/2*.001, (row["local_start_z_mm"]+987.)*.001],
         [width/2*.001, width/2*.001, (row["local_end_z_mm"]+987.)*.001]])
    points = np.array([[0.,0.,centre], [.49*width,.49*width,centre],
                       [.51*width,0.,centre], [0.,0.,centre+1.]])*.001
    expected = [True, key=="camera", False, False]
    assert region.shape == ("box" if key=="camera" else "elliptic_cylinder")
    assert index.mask(points,key).tolist() == expected
    assert index.mask(points).tolist() == expected
    assert index.source_keys_for_component(key) == ()
    assert row == before


@pytest.mark.parametrize("key", KEYS)
def test_recording_affine_pose_matches_native_geometry(recording_document,key):
    row = next(deepcopy(p) for p in recording_document["parts"] if p["key"]==key)
    row["model_3d"] = dict(schema_version=1, transform=dict(
        offset_mm=[3.,-2.,4.], rotation_deg=[23.,37.,17.], scale_xy=[1.3,.7]))
    state = _state([row],shift=987.)
    region, = build_field_display_regions(state,()).regions_for_component(key)
    model = part_model_from_document({"parts":[row]},key,include_children=False,angular_segments=144)
    vertices = np.concatenate([mesh.vertices for mesh in model.meshes])
    vertices[:,2] += 987.
    vertices *= .001
    # Recorded shapes use the same transformed physical coordinates. The
    # analytic circular AABB encloses its finite polygonal approximation.
    assert np.all(vertices >= region.bounds_m[0]-1e-12)
    assert np.all(vertices <= region.bounds_m[1]+1e-12)
    # Avoid asking an exact inclusive boundary predicate to recover sub-ULP
    # local coordinates after two metre-scale translation/rotation operations.
    assert region.contains(region.origin_m+(vertices-region.origin_m)*(1.-1e-9)).all()
    if key=="camera":
        np.testing.assert_allclose([vertices.min(axis=0),vertices.max(axis=0)],region.bounds_m,atol=1e-12)
    relative = np.array([[0.,0.,0.],[row["outer_width_mm"]*.6,0.,0.]])*.001
    points = relative @ np.linalg.inv(region.inverse_linear).T + region.origin_m
    assert region.contains(points).tolist() == [True,False]


def test_invalid_reference_and_bent_eels_do_not_get_invented_volumes(recording_document):
    camera = next(deepcopy(p) for p in recording_document["parts"] if p["key"]=="camera")
    for change in ({"outer_width_mm":-1.}, {"outer_width_mm":True},
                   {"mechanical_footprint_shape":"disk"}, {"inner_diameter_mm":1.},
                   {"local_start_z_mm":camera["local_center_z_mm"],
                    "local_end_z_mm":camera["local_center_z_mm"],"length_mm":0.}):
        index = build_field_display_regions(_state([{**camera,**change}]),())
        assert not index.regions_for_component("camera")
    filtered = read_document(INSTRUMENT_CONFIG_ROOT/"project_and_recording_system/EnergyFilter.toml")
    eels = next(p for p in filtered["parts"] if p.get("mechanical_profile")=="zebra_eels_detector")
    assert eels["mechanical_footprint_shape"] == "rectangle"
    assert eels["strip_active_width_mm"] != eels["strip_active_height_mm"]
    index = build_field_display_regions(_state([eels]),())
    assert not index.regions_for_component(eels["key"])
    assert "Bent-path" in index.component_notes[eels["key"]][0]


def test_explicit_replacement_keeps_its_own_shape(recording_document):
    camera = next(deepcopy(p) for p in recording_document["parts"] if p["key"]=="camera")
    camera["model_3d"] = dict(schema_version=1,base=dict(kind="box",width_mm=2.,height_mm=4.,length_mm=3.))
    region, = build_field_display_regions(_state([camera]),()).regions_for_component("camera")
    np.testing.assert_allclose(region.half_extents_m,[.001,.002,.0015])


def test_page_switches_recording_observation_and_restores_all(qtbot,monkeypatch):
    from temsim.gui.field_vector_page import FieldVectorPage
    document=read_document(INSTRUMENT_CONFIG_ROOT/"project_and_recording_system/NoEnergyFilter.toml")
    rows=[deepcopy(p) for p in document["parts"] if p["key"] in KEYS]
    state=_state(rows,shift=987.)
    state.beam_voltage_kv=300.
    scene=Scene()
    scene.sources=(make_source("magnet"),)
    scene.bounds_m=np.array([[-.1,-.1,0.],[.1,.1,4.]])
    monkeypatch.setattr("temsim.field_vector_scene.prepare_field_vector_scene",lambda *_a,**_k:scene)
    page=FieldVectorPage()
    qtbot.addWidget(page)
    page.auto_detail.setChecked(False)
    page.update_snapshot(state)
    page.focus_component("camera")
    page.set_active(True)
    wait_sample(qtbot,page)
    for key in KEYS:
        page.component_picker.setCurrentIndex(page.component_picker.findData(key))
        wait_sample(qtbot,page)
        assert page._selected_component==key
        assert page._display_regions.mask(page._sample.points_m,key).all()
        assert len(page._sample.points_m)>0 and np.all(page._sample.b_t[:,2]==1.)
        assert page.field_scope.currentData()=="all" and page.selected_source_keys==("magnet",)
        # A detector contributes no independent field but still observes zero
        # from the explicitly empty selection, rather than losing its volume.
        page.field_scope.setCurrentIndex(page.field_scope.findData("component"))
        wait_sample(qtbot,page)
        assert not page.selected_source_keys and np.all(page._sample.b_t==0.)
        page.field_scope.setCurrentIndex(page.field_scope.findData("all"))
        wait_sample(qtbot,page)
    page.component_picker.setCurrentIndex(0)
    wait_sample(qtbot,page)
    assert page._selected_component is None
    assert page._display_regions.mask(page._sample.points_m).all()
    assert len(page._sample.points_m)>0 and np.all(page._sample.b_t[:,2]==1.)
    page.set_active(False)
