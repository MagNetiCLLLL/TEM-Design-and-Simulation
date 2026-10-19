"""Material/field registration and real section checks, without desktop use."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from temsim.gui.field_vector_canvas import FieldVectorCanvas
from temsim.part_model_3d import TriangleMesh, revolve_section


def sample_at(points):
    points = np.asarray(points, dtype=float)
    values = np.tile((.2, .4, .8), (len(points), 1))
    valid = np.ones(len(points), dtype=bool)
    return SimpleNamespace(points_m=points, b_t=values, e_v_per_m=values * 2,
                           force_n=values * 3, magnetic_valid=valid,
                           electric_valid=valid, force_valid=valid)


def tube_mesh():
    # The existing mesh constructor is geometrically unit-neutral. Values
    # here are metres, matching the canvas's explicitly documented boundary.
    return revolve_section(((-.002, .0004), (-.002, .0008),
                            (.002, .0008), (.002, .0004)), key="tube",
                           angular_segments=64, material_class="copper_winding")


def canvas_with_sample(qtbot, section="3D"):
    canvas = FieldVectorCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(760, 540)
    bounds = np.array([[-.0002, -.0002, -.0003], [.0002, .0002, .0003]])
    canvas.set_sample(sample_at([[0, 0, 0], [.0001, .0001, .0002]]), bounds, section)
    canvas.set_model_visible(True)
    canvas.show()
    return canvas


def pixels(canvas):
    image = canvas.grab().toImage()
    return bytes(image.constBits())


def test_mesh_and_field_share_projection_under_fit_rotation_pan_and_zoom(qtbot):
    canvas = canvas_with_sample(qtbot)
    mesh = tube_mesh()
    points = mesh.vertices[::17].copy()
    canvas.set_sample(sample_at(points), canvas._bounds_m, "3D")
    canvas.set_physical_meshes((mesh,))
    canvas.fit_physical_models()
    for yaw, pitch, zoom, pan in ((.55, .45, 1., QPointF()),
                                  (-.9, .7, 2.3, QPointF(37, -23))):
        canvas._yaw, canvas._pitch, canvas._zoom, canvas._pan = yaw, pitch, zoom, pan
        canvas.grab()
        actual = canvas._project_positions(mesh.vertices[::17])[:, :2]
        np.testing.assert_allclose(actual, canvas._screen_points, atol=1e-10)
        centre, span, basis, size, origin = canvas._projection()
        expected = (((points - centre) / span) @ basis.T)[:, :2] * size + origin
        np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_fit_models_only_changes_display_bounds_and_reset_restores_sample(qtbot):
    canvas = canvas_with_sample(qtbot)
    mesh = tube_mesh()
    sample = canvas.sample
    points = sample.points_m.copy()
    bounds = canvas._bounds_m.copy()
    refs = canvas.references.copy()
    canvas.set_physical_meshes((mesh,))
    np.testing.assert_array_equal(canvas._view_bounds_m, bounds)
    canvas.fit_physical_models()
    assert np.all(canvas._view_bounds_m[0] <= mesh.vertices.min(axis=0))
    assert np.all(canvas._view_bounds_m[1] >= mesh.vertices.max(axis=0))
    assert canvas.sample is sample and canvas.references == refs
    np.testing.assert_array_equal(canvas._bounds_m, bounds)
    np.testing.assert_array_equal(sample.points_m, points)
    plot = canvas._plot_rect()
    screen = canvas._project_positions(mesh.vertices)
    assert all(plot.contains(QPointF(*point)) for point in screen[:, :2])
    canvas.reset_view()
    np.testing.assert_array_equal(canvas._view_bounds_m, bounds)
    assert canvas._zoom == 1. and not canvas._fit_models


@pytest.mark.parametrize("section", ["3D", "XY", "XZ", "YZ"])
def test_zero_opacity_is_identical_to_field_only_render(qtbot, section):
    canvas = canvas_with_sample(qtbot, section)
    canvas.set_physical_meshes((tube_mesh(),))
    canvas.fit_physical_models()
    canvas.set_model_visible(False)
    field_only = pixels(canvas)
    canvas.set_model_visible(True)
    canvas.set_model_opacity(0.)
    assert not canvas.model_overlay_visible
    assert pixels(canvas) == field_only
    canvas.set_model_opacity(.6)
    assert pixels(canvas) != field_only


@pytest.mark.parametrize("section, normal", [("XY", 2), ("XZ", 1), ("YZ", 0)])
def test_sections_are_true_intersections_and_preserve_the_tube_bore(qtbot, section, normal):
    canvas = canvas_with_sample(qtbot, section)
    mesh = tube_mesh()
    bounds = canvas._bounds_m.copy()
    bounds[:, normal] = 0.
    canvas.set_sample(canvas.sample, bounds, section)
    canvas.set_physical_meshes((mesh,))
    cuts = canvas._physical_sections()
    assert len(cuts) == 1
    segments = cuts[0][1]
    np.testing.assert_array_equal(segments[..., normal], 0.)
    if section == "XY":
        # A slice through the hollow tube contains two separate rings;
        # no false face spans its empty centre.
        radius = np.linalg.norm(segments[..., :2], axis=-1)
        assert radius.min() >= .0004 * np.cos(np.pi / 64) - 1e-12
        assert radius.max() <= .0008 + 1e-12
        mid_radius = np.linalg.norm(segments[..., :2].mean(axis=1), axis=-1)
        assert mid_radius.min() > .00039
    else:
        transverse = 0 if section == "XZ" else 1
        assert np.abs(segments[..., transverse]).min() >= .0004 - 1e-12
        assert not np.any(segments[:, 0, transverse] * segments[:, 1, transverse] < 0)
    # An offset plane missing the solid must have no projected silhouette.
    bounds[:, normal] = .003
    canvas.set_sample(canvas.sample, bounds, section)
    canvas.set_physical_meshes((mesh,))
    assert canvas._physical_sections() == ()


def test_section_follows_sample_plane_even_when_fit_changes_display_centre(qtbot):
    canvas = canvas_with_sample(qtbot, "XY")
    bounds = canvas._bounds_m.copy()
    bounds[:, 2] = .0011
    canvas.set_sample(canvas.sample, bounds, "XY")
    canvas.set_physical_meshes((tube_mesh(),))
    canvas.fit_physical_models()
    assert np.mean(canvas._view_bounds_m[:, 2]) != .0011
    segments = canvas._physical_sections()[0][1]
    np.testing.assert_array_equal(segments[..., 2], .0011)


def test_replacement_and_clear_cannot_keep_obsolete_geometry(qtbot):
    canvas = canvas_with_sample(qtbot)
    mesh = tube_mesh()
    canvas.set_physical_meshes((mesh,))
    canvas.fit_physical_models()
    canvas.set_sample(canvas.sample, canvas._bounds_m, "3D")
    assert not canvas._physical_meshes and not canvas.model_overlay_visible
    canvas.set_physical_meshes((mesh,))
    assert canvas._fit_models and canvas._view_bounds_m[1, 2] == .002
    canvas.clear("Stale inputs")
    assert canvas.sample is None and not canvas._physical_meshes
    assert canvas._section_cache is None


@pytest.mark.parametrize("opacity", [-.1, 1.1, float("nan"), float("inf")])
def test_opacity_rejects_invalid_values(qtbot, opacity):
    canvas = canvas_with_sample(qtbot)
    with pytest.raises(ValueError, match="opacity"):
        canvas.set_model_opacity(opacity)


def test_invalid_face_indices_rejected_and_nonmaterial_guides_not_shown(qtbot):
    canvas = canvas_with_sample(qtbot)
    mesh = tube_mesh()
    with pytest.raises(ValueError, match="outside"):
        canvas.set_physical_meshes((replace(mesh, faces=np.array([[0, 1, len(mesh.vertices)]])),))
    canvas.set_physical_meshes((replace(mesh, wireframe=True),))
    assert not canvas._physical_meshes


def test_opaque_triangles_are_sorted_back_to_front_across_materials(qtbot):
    canvas = canvas_with_sample(qtbot)
    canvas.set_sample(sample_at([[.0002, .0002, .0003]]), canvas._bounds_m, "3D")
    canvas._yaw = canvas._pitch = 0.
    # With this camera depth is -Y and vertical is Z. The near blue and far
    # red triangles cover the same screen location; submit near first.
    xz = np.array([[-.0001, -.0002], [.0001, -.0002], [0., .0002]])
    def triangle(y, colour):
        vertices = np.column_stack((xz[:, 0], np.full(3, y), xz[:, 1]))
        return TriangleMesh(vertices, np.array([[0, 1, 2]]), str(y), color=colour)
    canvas.set_physical_meshes((triangle(-.0001, (0., 0., 1., 1.)),
                                triangle(.0001, (1., 0., 0., 1.))))
    canvas.set_model_opacity(1.)
    image = canvas.grab().toImage()
    centre = canvas._project_positions(np.array([[0., 0., 0.]]))[0]
    colour = image.pixelColor(round(centre[0]), round(centre[1]))
    assert colour.blue() > 200 and colour.red() < 30
