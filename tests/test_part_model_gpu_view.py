"""Automatic CAD rendering fallback and real hardware rendering/picking checks.

Hardware checks skip when the active Qt platform cannot create a GPU context.
The constructor/runtime failure tests inject failures, never a fake successful
GPU renderer. Run with QT_QPA_PLATFORM=windows for native Windows GPU coverage.
"""

import numpy as np
import pytest
from PySide6.QtCore import QPoint

import temsim.gui.part_model_view as view_module


def _square(key="front", y=-20.0, radius=40.0):
    vertices = np.array([[-radius, y, -radius], [radius, y, -radius],
                         [radius, y, radius], [-radius, y, radius]])
    paths = (("parts", key, "length_mm"),)
    return dict(key=key, region="body", color=(0.3, 0.6, 0.9),
                vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]),
                face_groups=("front", "front"),
                surfaces={"front": dict(label="Front face", parameter_paths=paths)},
                edges=[dict(id="rim", vertices=vertices[[0, 1, 2, 3, 0]],
                            parameter_paths=paths)])


def _make_view(qtbot, records=None):
    view = view_module.PartModelView()
    qtbot.addWidget(view)
    view.resize(640, 520)
    view.set_meshes([_square()] if records is None else records)
    view.set_front_view()
    view._render()
    return view


def _point(view, world):
    point = view.project_points([world])[0]
    return QPoint(round(point[0]), round(point[1]))


@pytest.fixture
def gpu_view(qtbot):
    view = _make_view(qtbot)
    if view.renderer_backend != "GPU":
        pytest.skip(f"Real GPU rendering unavailable: {view.renderer_detail}")
    assert view._gpu_renderer is not None
    return view


def test_unavailable_gpu_falls_back_once_and_retains_visible_surface_picking(qtbot, monkeypatch):
    attempts = []

    def unavailable(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError("test context unavailable")

    monkeypatch.setattr(view_module, "_create_gpu_renderer", unavailable)
    view = _make_view(qtbot, [_square("back", 20), _square()])
    assert view.renderer_backend == "CPU"
    assert "test context unavailable" in view.renderer_detail
    assert not view._image.isNull()
    assert view.pick_at(_point(view, [0, -20, 0])) == ("front", "body")
    view.set_isometric_view()
    view._render()
    view.set_front_view()
    view._render()
    assert len(attempts) == 1  # No costly context retries during every gesture.


def test_empty_preview_defers_gpu_initialization_until_a_solid_is_shown(qtbot, monkeypatch):
    attempts = []

    def unavailable():
        attempts.append(True)
        raise RuntimeError("test context unavailable")

    monkeypatch.setattr(view_module, "_create_gpu_renderer", unavailable)
    view = _make_view(qtbot, [])
    assert attempts == []
    assert not view._image.isNull()
    view.set_meshes([_square()])
    view._render()
    assert len(attempts) == 1
    assert view.renderer_backend == "CPU"


def test_real_gpu_runtime_failure_falls_back_in_the_same_frame(gpu_view, monkeypatch):
    view = gpu_view

    def failed_frame(*args, **kwargs):
        raise RuntimeError("test GPU context lost")

    monkeypatch.setattr(view._gpu_renderer, "render", failed_frame)
    view.set_isometric_view()
    view._render()
    assert view.renderer_backend == "CPU"
    assert "test GPU context lost" in view.renderer_detail
    assert not view._image.isNull()
    assert not view._dirty
    view.set_front_view()
    assert view.pick_at(_point(view, [0, -20, 0])) == ("front", "body")


def test_real_gpu_keeps_cad_triangles_and_interior_lighting(gpu_view):
    view = gpu_view
    original_vertices = view._meshes[0].vertices.copy()
    original_faces = view._meshes[0].faces.copy()
    assert sum(len(item.triangles) for item in view._prepared) == len(original_faces)
    gpu_image = view._image.copy()
    assert not gpu_image.isNull()
    interior = _point(view, [10, -20, -10])
    gpu_color = gpu_image.pixelColor(interior)
    assert gpu_color != gpu_image.pixelColor(QPoint(2, 2))
    view._dirty = True
    view._render_cpu()
    cpu_color = view._image.pixelColor(interior)
    # GPU float arithmetic and byte conversion may round a channel by one.
    assert np.max(np.abs(np.array(gpu_color.getRgb(), dtype=int)
                         - np.array(cpu_color.getRgb(), dtype=int))) <= 1
    np.testing.assert_array_equal(view._meshes[0].vertices, original_vertices)
    np.testing.assert_array_equal(view._meshes[0].faces, original_faces)


def test_real_gpu_fitted_small_part_keeps_detail_at_column_length_coordinates(gpu_view):
    view = gpu_view
    record = _square(y=-0.02, radius=0.05)
    view.set_meshes([record])
    view._render()
    near_origin = view._image.copy()
    offset = np.array([30.0, 20.0, 3000.0])
    record["vertices"] += offset
    record["edges"][0]["vertices"] += offset
    view.set_meshes([record])
    view._render()
    # A 0.1 mm component has identical geometry wherever it is installed.
    # At z=3000 mm unrebased float32 vertices visibly shift its fine edges.
    assert view._image == near_origin
    assert view.pick_at(_point(view, offset + [0, -0.02, 0])) == ("front", "body")
    assert view.renderer_backend == "GPU"


def test_real_gpu_camera_changes_reuse_geometry_without_cpu_rasterization(gpu_view, monkeypatch):
    view = gpu_view
    renderer = view._gpu_renderer
    uploads = renderer.geometry_uploads
    assert uploads > 0
    prepared = tuple(id(item) for item in view._prepared)

    def unexpected_software_frame(*args, **kwargs):
        pytest.fail("Camera rendering unexpectedly ran the software triangle rasterizer")

    monkeypatch.setattr(view, "_render_cpu", unexpected_software_frame)
    view.set_isometric_view()
    view._render()
    view._zoom *= 1.3
    view._pan += [17, -8]
    view._invalidate(interactive=True)
    view._render()
    view._settle()
    view._render()
    assert view.renderer_backend == "GPU"
    assert view._gpu_renderer is renderer
    assert renderer.geometry_uploads == uploads
    assert tuple(id(item) for item in view._prepared) == prepared


def test_real_gpu_exact_face_picking_preserves_display_and_reuses_pick_buffer(gpu_view, monkeypatch):
    view = gpu_view
    view.set_selection_mode("face")
    view._render()
    image = view._image.copy()
    calls = []
    render_cpu = view._render_cpu

    def record_pick_frame(*args, **kwargs):
        calls.append(True)
        return render_cpu(*args, **kwargs)

    monkeypatch.setattr(view, "_render_cpu", record_pick_frame)
    for position in ((10, -20, -10), (-10, -20, 10)):
        picked = view.pick_topology_at(_point(view, position))
        assert (picked["key"], picked["kind"], picked["id"]) == ("front", "face", "front")
        assert picked["parameter_paths"] == (("parts", "front", "length_mm"),)
        assert picked["normal"] == pytest.approx([0, -1, 0])
        assert picked["point"][1] == pytest.approx(-20, abs=1e-10)
    assert len(calls) == 1
    assert view._image == image
    assert view.renderer_backend == "GPU"
    view.set_isometric_view()
    picked = view.pick_topology_at(_point(view, [0, -20, 0]))
    assert picked["point"][1] == pytest.approx(-20, abs=1e-10)
    assert len(calls) == 2


def test_real_gpu_section_invalidates_picking_and_preserves_source_faces(gpu_view):
    view = gpu_view
    records = [_square("removed", -20), _square("retained", 20)]
    original = [(record["vertices"].copy(), record["faces"].copy()) for record in records]
    view.set_meshes(records)
    view.set_selection_mode("face")
    before = view.pick_topology_at(_point(view, [0, -20, 0]))
    assert before["key"] == "removed"
    view.set_section_enabled(True)
    after = view.pick_topology_at(_point(view, [0, 20, 0]))
    assert after["key"] == "retained"
    assert after["id"] == "front"
    assert after["point"][1] == pytest.approx(20, abs=1e-10)
    assert view.renderer_backend == "GPU"
    view.set_section_enabled(False)
    assert view.pick_at(_point(view, [0, -20, 0])) == ("removed", "body")
    for mesh, (vertices, faces) in zip(view._meshes, original):
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)


def test_real_gpu_edge_pick_does_not_select_occluded_back_rim(gpu_view):
    view = gpu_view
    view.set_meshes([_square("back", 20, 25), _square()])
    view.set_selection_mode("edge")
    assert view.pick_topology_at(_point(view, [25, 20, 0])) is None
    edge = view.pick_topology_at(_point(view, [40, -20, 0]))
    assert (edge["key"], edge["kind"], edge["id"]) == ("front", "edge", "rim")
    assert edge["point"] == pytest.approx([40, -20, 0], abs=0.3)
    assert view.renderer_backend == "GPU"


def _axial_depth_pair():
    xy = np.array([[-0.05, -0.05], [0.05, -0.05], [0.0, 0.05]])
    return [dict(key=key, region="body", color=color,
                 vertices=np.column_stack((xy, np.full(3, depth))),
                 faces=np.array([[0, 1, 2]]))
            for key, depth, color in (("far", 0.0, (1.0, 0.0, 0.0)),
                                      ("near", 1e-5, (0.0, 0.0, 1.0)))]


def _assert_visible_near_triangle(view):
    # Inspect the rendered IDs/color directly: exact CPU picking must not mask
    # loss of GPU depth precision by independently returning the correct face.
    position = _point(view, [-0.02, -0.02, 1e-5])
    assert view._ids[position.y(), position.x()] == 1
    color = view._image.pixelColor(position)
    assert color.blue() > color.red()


def test_real_gpu_uses_visible_depth_range_when_fitting_a_tiny_component(gpu_view):
    view = gpu_view
    records = _axial_depth_pair()
    records.append(dict(key="offscreen", region="body", color=(0.0, 1.0, 0.0),
                        vertices=np.array([[99.95, -0.05, 3000.0],
                                           [100.05, -0.05, 3000.0],
                                           [100.0, 0.05, 3000.0]]),
                        faces=np.array([[0, 1, 2]])))
    view.set_meshes(records)
    view.set_axial_view()
    assert view.fit_region("near", "body")
    view._render()
    assert view.renderer_backend == "GPU"
    _assert_visible_near_triangle(view)


def test_real_gpu_uses_temporary_cpu_for_unresolvable_depth_then_resumes_gpu(gpu_view):
    view = gpu_view
    renderer = view._gpu_renderer
    records = _axial_depth_pair()
    # Its projected bounding box covers the fitted view, but its material lies
    # in x+y>=1 mm, away from the near pair. Keeping this conservative bound is
    # safe only if the precision guard prevents an unresolved GPU depth tie.
    records.append(dict(key="diagonal", region="body", color=(0.0, 1.0, 0.0),
                        vertices=np.array([[-100.0, 101.0, 3000.0],
                                           [101.0, -100.0, 3000.0],
                                           [101.0, 101.0, 3000.0]]),
                        faces=np.array([[0, 1, 2]])))
    originals = [(record["vertices"].copy(), record["faces"].copy()) for record in records]
    view.set_meshes(records)
    view.set_axial_view()
    assert view.fit_region("near", "body")
    view._render()
    assert view.renderer_backend == "CPU"
    assert "depth precision" in view.renderer_detail.lower()
    assert view._gpu_precision_fallback
    assert view._gpu_renderer is renderer
    _assert_visible_near_triangle(view)
    view.fit_all()
    view._render()
    assert view.renderer_backend == "GPU"
    assert not view._gpu_precision_fallback
    assert view._gpu_renderer is renderer
    for mesh, (vertices, faces) in zip(view._meshes, originals):
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)
