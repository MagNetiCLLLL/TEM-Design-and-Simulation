"""Ray Diagram placement regressions using captured rays, without propagation.

Optical graphics follow field-registration coordinates; mechanical graphics
follow the translated, unrotated CAD. The rays are already-computed fixtures.
"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest

from temsim.lens_pose import lens_pose_registration
from temsim.part_model_3d import part_model_from_document

from test_incremental_ray_scene import _part, _result, make_workspace  # noqa: F401


def _captured_result(*, origin=100.0, **pose):
    result = _result(stops=False)
    parts = list(result.assembly.parts)
    lens = parts[0]
    lens.data.update(
        mechanical_profile="magnetic_lens_assembly",
        optical_reference_local_z_mm=3.8,
        mechanical_inner_diameter_mm=4.0,
        mechanical_outer_diameter_mm=8.0,
        **pose,
    )
    parts.insert(1, _part(
        "test_lens_coil", "Test Lens Coil", 3.4,
        profile="magnetic_excitation_coil", parent_key="test_lens",
        mechanical_only=True, mechanical_inner_diameter_mm=4.0,
        mechanical_outer_diameter_mm=8.0,
    ))
    parts.insert(2, _part(
        "ac_deflector", "AC Pair", 5.0,
        profile="deflector_coils", optical_reference_local_z_mm=5.0,
        interaction_centers_local_z_mm=[4.7, 5.3],
        mechanical_inner_diameter_mm=4.0,
        mechanical_outer_diameter_mm=8.0,
    ))
    for part in parts:
        part.module_key = "synthetic-column"
        part.parent_key = part.data.get("parent_key")
        part.data.update(
            key=part.key, name=part.name,
            local_start_z_mm=part.start_z_mm,
            local_end_z_mm=part.end_z_mm, length_mm=part.length_mm,
        )
        part.start_z_mm += origin
        part.center_z_mm += origin
        part.end_z_mm += origin
    result.assembly.parts = tuple(parts)
    result.assembly.modules = (SimpleNamespace(
        key="synthetic-column", parts=result.assembly.parts,
    ),)
    for segment in result.assembly.vacuum_bore_segments:
        segment.start_z_mm += origin
        segment.end_z_mm += origin
    result.assembly.vacuum_liner_segments = ()
    result.state_snapshot._resolved_assembly = result.assembly
    result.state_snapshot.lenses[0].z_mm += origin
    result.state_snapshot.sample.z_mm += origin
    result.state_snapshot.recording_planes[0].z_mm += origin
    for branch in [result.simulation.incident, *result.simulation.branches.values()]:
        branch.z += origin
        branch.blocked_z += origin
    result.aperture_stops = tuple(
        {**record, "z_mm": record["z_mm"] + origin}
        for record in result.aperture_stops
    )
    result.lens_crossovers = tuple(
        {**record, "z_mm": record["z_mm"] + origin}
        for record in result.lens_crossovers
    )
    return result


def _items(view, key, role=None):
    return [item for item in view.plot.plotItem.items
            if getattr(item, "component_key", None) == key
            and (role is None or getattr(item, "datum_role", None) == role)]


def _one(view, key, role):
    items = _items(view, key, role)
    assert len(items) == 1, (key, role, items)
    return items[0]


def _field_point(result, key, local_z):
    part = result.assembly.part(key)
    origin = part.center_z_mm - part.data["local_center_z_mm"]
    registration = lens_pose_registration(result.state_snapshot, key)
    point = registration.rotation_array @ np.array((0.0, 0.0, origin + local_z))
    point += np.asarray(registration.origin_global_m) * 1000.0
    return point, registration.rotation_array[:, 2]


def _assert_graphic(item, point, normal, angle_deg):
    point = np.asarray(point)
    np.testing.assert_allclose(item.datum_point_mm, point, rtol=0, atol=1e-12)
    transverse = np.array((np.cos(np.deg2rad(angle_deg)),
                           np.sin(np.deg2rad(angle_deg))))
    expected = np.array((point[2], transverse @ point[:2]))
    if isinstance(item, pg.InfiniteLine):
        np.testing.assert_allclose((item.pos().x(), item.pos().y()), expected,
                                   rtol=0, atol=1e-12)
        anchor = getattr(item, "datum_anchor", None)
        if anchor is not None:
            x, y = anchor.getData()
            np.testing.assert_allclose((x[0], y[0]), expected, rtol=0, atol=1e-12)
        if normal is not None:
            np.testing.assert_allclose(item.datum_normal, normal, rtol=0, atol=1e-14)
            # The drawn plane section is orthogonal to its projected normal.
            section_normal = np.array((normal[2], transverse @ normal[:2]))
            direction = np.array((np.cos(np.deg2rad(item.angle)),
                                  np.sin(np.deg2rad(item.angle))))
            assert section_normal @ direction == pytest.approx(0.0, abs=1e-12)
        else:
            assert item.datum_normal is None  # A Z guide is not an optical plane.
    else:
        x, y = item.getData()
        np.testing.assert_allclose((x[0], y[0]), expected, rtol=0, atol=1e-12)


def test_optical_tilt_changes_field_markers_without_rotating_child_cad(make_workspace):
    result = _captured_result(offset_x_mm=0.24, offset_y_mm=-0.31,
                              offset_z_mm=0.7, rotation_y_mrad=80.0)
    view = make_workspace()
    view.display_result(result, "Preview")
    optical = _one(view, "test_lens", "optical_reference")
    mechanical = _one(view, "test_lens", "mechanical_center")
    point, normal = _field_point(result, "test_lens", 3.8)
    centre, _ = _field_point(result, "test_lens", 3.0)
    _assert_graphic(optical, point, normal, 0.0)
    _assert_graphic(mechanical, centre, None, 0.0)
    assert not np.allclose(point, centre)
    assert "optical" in optical.toolTip().lower()
    assert "mechanical" in mechanical.toolTip().lower()

    # CAD uses the same placed centre, with the 100 mm module origin once.
    rows = [deepcopy(part.data) for part in result.assembly.parts[:2]]
    cad = part_model_from_document({"parts": rows}, "test_lens_coil", angular_segments=8)
    vertices = np.concatenate([mesh.vertices for mesh in cad.meshes])
    cad_centre = vertices.mean(axis=0) + (0.0, 0.0, 100.0)
    coil_centre = np.array((0.24, -0.31, 104.1))
    np.testing.assert_allclose(cad_centre, coil_centre, rtol=0, atol=1e-12)
    optical_coil_centre, _ = _field_point(result, "test_lens_coil", 3.4)
    assert not np.allclose(optical_coil_centre, coil_centre)
    _assert_graphic(_one(view, "test_lens_coil", "mechanical_center"),
                    cad_centre, None, 0.0)
    assert result.state_snapshot.lenses[0].percent == 50.0


def test_parent_pose_edit_refreshes_owned_markers_with_identical_rays(make_workspace):
    view = make_workspace()
    before = _captured_result()
    view.display_result(before, "Preview")
    old_lens = _one(view, "test_lens", "optical_reference")
    old_coil = _one(view, "test_lens_coil", "mechanical_center")
    camera = tuple(view.recording_surface_range_items)
    wall = tuple(view.column_wall_items)
    after = deepcopy(before)
    after.simulation = before.simulation  # Only placement changed, no new trace.
    after.assembly.part("test_lens").data.update(offset_x_mm=0.6, rotation_y_mrad=40.0)
    view.display_result(after, "Preview")
    assert _one(view, "test_lens", "optical_reference") is not old_lens
    assert _one(view, "test_lens_coil", "mechanical_center") is not old_coil
    assert tuple(view.recording_surface_range_items) == camera
    assert tuple(view.column_wall_items) == wall
    expected = (0.6, 0.0, 103.4)
    _assert_graphic(_one(view, "test_lens_coil", "mechanical_center"), expected, None, 0.0)
    assert before.state_snapshot.lenses[0].percent == after.state_snapshot.lenses[0].percent == 50.0
    assert before.simulation is after.simulation


def test_projection_change_updates_placed_plane_without_rebuilding(make_workspace):
    result = _captured_result(offset_x_mm=0.24, offset_y_mm=-0.31,
                              rotation_x_mrad=50.0, rotation_y_mrad=80.0)
    view = make_workspace()
    view.display_result(result, "Preview")
    optical = _one(view, "test_lens", "optical_reference")
    mechanical = _one(view, "test_lens", "mechanical_center")
    point, normal = _field_point(result, "test_lens", 3.8)
    centre, _ = _field_point(result, "test_lens", 3.0)
    for angle in (90.0, 37.0, 0.0):
        view._set_projection_angle(angle)
        assert _one(view, "test_lens", "optical_reference") is optical
        assert _one(view, "test_lens", "mechanical_center") is mechanical
        _assert_graphic(optical, point, normal, angle)
        _assert_graphic(mechanical, centre, None, angle)
    # Publishing the same retained result must not restore the old projection.
    view._set_projection_angle(37.0)
    view.display_result(result, "Preview")
    assert _one(view, "test_lens", "optical_reference") is optical
    _assert_graphic(optical, point, normal, 37.0)


def test_deflector_interaction_planes_follow_rigid_field_frame(make_workspace):
    result = _captured_result()
    result.assembly.part("ac_deflector").data.update(
        offset_x_mm=0.2, offset_y_mm=-0.1, offset_z_mm=0.4,
        rotation_x_mrad=30.0, rotation_y_mrad=-60.0,
    )
    view = make_workspace()
    view.display_result(result, "Preview")
    view._set_projection_angle(41.0)
    planes = sorted(_items(view, "ac_deflector", "interaction_plane"),
                    key=lambda item: item.datum_point_mm[2])
    assert len(planes) == 2
    for item, local_z in zip(planes, (4.7, 5.3), strict=True):
        point, normal = _field_point(result, "ac_deflector", local_z)
        _assert_graphic(item, point, normal, 41.0)
    centre, _ = _field_point(result, "ac_deflector", 5.0)
    _assert_graphic(_one(view, "ac_deflector", "mechanical_center"), centre, None, 41.0)


def test_reopening_cached_result_uses_its_captured_geometry(make_workspace):
    view = make_workspace()
    old = _captured_result(offset_x_mm=0.15, offset_z_mm=0.5)
    old_point, old_normal = _field_point(old, "test_lens", 3.8)
    view.display_result(old, "Preview")
    live = deepcopy(old)
    live.assembly.part("test_lens").data.update(
        offset_x_mm=0.9, offset_z_mm=3.0, rotation_y_mrad=70.0)
    view.display_result(live, "Preview")
    assert _one(view, "test_lens", "optical_reference").datum_point_mm != tuple(old_point)
    old.cache_hit = True
    view.display_result(old, "Preview")
    _assert_graphic(_one(view, "test_lens", "optical_reference"),
                    old_point, old_normal, 0.0)
    assert old.assembly.part("test_lens").data["offset_x_mm"] == 0.15
    assert old.state_snapshot.lenses[0].percent == live.state_snapshot.lenses[0].percent == 50.0
    # Tree selection can supply the live object: focus/highlight still refer to
    # the captured result being displayed, and to its mechanical centre.
    view._highlight_ray_component(live.assembly.part("test_lens"))
    centre, _ = _field_point(old, "test_lens", 3.0)
    assert np.mean(view._ray_component_highlight.getRegion()) == pytest.approx(centre[2])
    assert np.mean(view._component_x_range(live.assembly.part("test_lens"))) == pytest.approx(centre[2])
    # An uncaptured live addition and an installed curved-branch part have no
    # main-column datum. Neither may invent a focus position in this plot.
    added = _part("new_live_lens", "New live lens", 1000.0)
    branch = deepcopy(old.assembly.part("test_lens"))
    branch.key = "energy_filter_prism"
    branch.data.update(key=branch.key, branch_path_only=True)
    old.assembly.parts = (*old.assembly.parts, branch)
    view.plot.setRange(xRange=(101.0, 119.0), yRange=(-0.7, 0.9),
                       padding=0.0, disableAutoRange=True)
    saved_range = np.array(view.plot.viewRange())
    for unplaced in (added, branch):
        view._highlight_ray_component(old.assembly.part("test_lens"))
        view._highlight_ray_component(unplaced)
        assert view._ray_component_highlight is None
        assert view._component_x_range(unplaced) is None
        view._apply_component_zoom(unplaced)
        np.testing.assert_allclose(view.plot.viewRange(), saved_range, rtol=0, atol=1e-12)
    view._last_result = None
    view._highlight_ray_component(live.assembly.part("test_lens"))
    assert view._ray_component_highlight is None
    assert view._component_x_range(live.assembly.part("test_lens")) is None
    view._apply_component_zoom(live.assembly.part("test_lens"))
    np.testing.assert_allclose(view.plot.viewRange(), saved_range, rtol=0, atol=1e-12)


def test_tilted_aperture_contour_uses_local_hole_offset_and_captured_field_basis(make_workspace):
    result = _captured_result()
    aperture = result.assembly.part("test_aperture")
    aperture.key = "condenser_aperture_2"
    aperture.data.update(
        key=aperture.key, optical_reference_local_z_mm=7.1,
        offset_x_mm=0.2, offset_y_mm=-0.4, offset_z_mm=0.1,
        rotation_x_mrad=30.0, rotation_y_mrad=60.0,
    )
    assert result.aperture_stops[0]["z_mm"] == 107.0
    result.aperture_stops = ({**result.aperture_stops[0], "key": aperture.key,
                              "z_mm": 107.1},)
    source_record = deepcopy(result.aperture_stops[0])
    view = make_workspace()
    view.display_result(result, "Preview")
    point, normal = _field_point(result, aperture.key, 7.1)
    mechanical_point, _ = _field_point(result, aperture.key, 7.0)
    mechanical = _one(view, aperture.key, "mechanical_center")
    assert mechanical.datum_normal is None
    assert np.linalg.norm(point-mechanical_point) == pytest.approx(0.1)
    registration = lens_pose_registration(result.state_snapshot, aperture.key)
    basis = registration.rotation_array
    lower, upper, _name, _z, record = view._aperture_projection_records[0]
    np.testing.assert_allclose(record["_datum_point_mm"], point, atol=1e-12)
    np.testing.assert_allclose(record["_datum_basis"], basis, atol=1e-14)
    outline = record["_outline_item"]
    for angle in (0.0, 47.0, 90.0):
        view._set_projection_angle(angle)
        _assert_graphic(lower, point, normal, angle)
        _assert_graphic(upper, point, normal, angle)
        _assert_graphic(mechanical, mechanical_point, None, angle)
        z, u = outline.getData()
        direction = np.array((np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))))
        projected_basis = np.vstack((basis[2, :2], direction @ basis[:2, :2]))
        # Undo only the 2-D view projection: the shown contour must still be
        # the same circular hole in its own field/CAD frame at every angle.
        local = np.linalg.solve(projected_basis,
                                np.vstack((z-point[2], u-direction @ point[:2])))
        hole_offset = np.array((source_record["offset_x_mm"], source_record["offset_y_mm"]))
        radius = np.linalg.norm(local-hole_offset[:, None], axis=0)
        np.testing.assert_allclose(radius, source_record["diameter_mm"]/2, rtol=0, atol=3e-11)
        np.testing.assert_allclose(local[:, :-1].mean(axis=1), hole_offset, rtol=0, atol=3e-11)
    assert result.aperture_stops[0] == source_record
