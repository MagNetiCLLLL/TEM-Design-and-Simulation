"""Retained geometry must preserve exact sections, edits and visible picking."""

import itertools

import numpy as np
import pytest

from temsim.gui.part_model_view import PartModelView, _section_triangles


def _record(key, y, *, x=0.0, color=(0.3, 0.6, 0.9)):
    return dict(key=key, region="body", color=color,
                vertices=np.array([[-20. + x, y, -20.], [20. + x, y, -20.], [x, y, 20.]]),
                faces=np.array([[0, 1, 2]]), face_groups=("face",),
                surfaces={"face": {"label": "surface", "color": color}})


@pytest.fixture
def view(qtbot):
    widget = PartModelView()
    qtbot.addWidget(widget)
    widget.resize(640, 520)
    widget.set_front_view()
    return widget


def _buffers(view):
    view._render()
    return (bytes(view._image.constBits()), view._ids.copy(), view._face_ids.copy(),
            view._depth.copy())


def _assert_buffers_equal(first, second):
    assert first[0] == second[0]
    for a, b in zip(first[1:], second[1:], strict=True):
        np.testing.assert_array_equal(a, b)


def _scalar_clip(triangles, positive):
    """Straight polygon clipping oracle, including coplanar vertices."""
    sign = 1 if positive else -1
    result, sources = [], []
    for source, triangle in enumerate(triangles):
        polygon = []
        for index, current in enumerate(triangle):
            previous = triangle[index - 1]
            a, b = sign * previous[1] >= 0, sign * current[1] >= 0
            if a != b:
                fraction = previous[1] / (previous[1] - current[1])
                polygon.append(previous + fraction * (current - previous))
            if b:
                polygon.append(current)
        for index in range(1, len(polygon) - 1):
            result.append((polygon[0], polygon[index], polygon[index + 1]))
            sources.append(source)
    return np.asarray(result).reshape(-1, 3, 3), np.asarray(sources, dtype=np.int64)


@pytest.mark.parametrize("positive", [False, True])
def test_batched_section_is_exact_and_preserves_source_order(positive):
    generator = np.random.default_rng(57311)
    triangles = generator.normal(size=(500, 3, 3))
    # Include all combinations of outside/on/inside, not just generic crossings.
    boundary = generator.normal(size=(27, 3, 3))
    boundary[..., 1] = tuple(itertools.product((-1.0, 0.0, 1.0), repeat=3))
    triangles = np.concatenate((triangles, boundary))
    original = triangles.copy()
    expected, sources = _scalar_clip(triangles, positive)
    actual, actual_sources = _section_triangles(triangles, with_sources=True, keep_positive_y=positive)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_sources, sources)
    np.testing.assert_array_equal(triangles, original)


def test_pool_hide_restore_retains_snapshot_and_exact_render(view, monkeypatch):
    import temsim.gui.part_model_view as module

    records = [_record("near", -10), _record("far", 10)]
    view.set_meshes(records)
    before = _buffers(view)
    snapshots, prepared = view._meshes, view._prepared
    # External records may be mutable, but installing a pool captures them.
    records[0]["vertices"] += 1000
    records[0]["surfaces"]["face"]["color"] = (1.0, 0.0, 0.0)
    monkeypatch.setattr(module, "_mesh", lambda *_: pytest.fail("Visibility must not reread source records"))
    view.set_visible_mesh_indices((1,), preserve_view=True)
    centre = view.project_points([[0, 10, 0]])[0, :2]
    assert view.pick_at(centre.tolist()) == ("far", "body")
    view.set_visible_mesh_indices((0, 1), preserve_view=True)
    assert all(a is b for a, b in zip(snapshots, view._meshes, strict=True))
    assert all(a is b for a, b in zip(prepared, view._prepared, strict=True))
    _assert_buffers_equal(before, _buffers(view))


def test_retained_pool_can_reorder_faces_without_changing_picking(view):
    records = [_record("left", -10, x=-30), _record("right", -10, x=30)]
    view.set_meshes(records)
    view.set_selection_mode("face")
    positions = [view.project_points([[x, -10, 0]])[0, :2].tolist() for x in (-30, 30)]
    expected = [view.pick_topology_at(position) for position in positions]
    view.set_visible_mesh_indices((1, 0), preserve_view=True)
    assert [view.pick_topology_at(position) for position in positions] == expected


def test_fresh_set_meshes_reads_same_mutable_object_again(view):
    record = _record("part", -10)
    view.set_meshes([record])
    before = _buffers(view)[0]
    old = view._meshes[0]
    record["surfaces"]["face"]["color"] = (1.0, 0.0, 0.0)
    view.set_meshes([record], preserve_view=True)
    assert view._meshes[0] is not old
    assert _buffers(view)[0] != before
    assert view.pick_at(view.project_points([[0, -10, 0]])[0, :2].tolist()) == ("part", "body")


def test_section_cache_is_reused_and_new_pool_discards_old_geometry(view, monkeypatch):
    import temsim.gui.part_model_view as module

    record = _record("part", 1)
    record["vertices"][:, 1] = (-2, 2, 2)
    view.set_meshes([record])
    view.set_section_enabled(True)
    before = _buffers(view)
    view.set_section_enabled(False)
    original_prepare = module._prepare_mesh
    monkeypatch.setattr(module, "_prepare_mesh", lambda *_: pytest.fail("Cached section should be reused"))
    view.set_section_enabled(True)
    _assert_buffers_equal(before, _buffers(view))
    monkeypatch.setattr(module, "_prepare_mesh", original_prepare)
    replacement = _record("part", -1)
    view.set_mesh_pool([replacement])
    view.set_visible_mesh_indices((0,), preserve_view=True)
    assert len(view._triangles[0]) == 0


@pytest.mark.parametrize("indices", [(0, 0), (-1,), (1,), (True,), (0.0,)])
def test_invalid_visibility_indices_do_not_change_display(view, indices):
    view.set_meshes([_record("part", -1)])
    before = _buffers(view)
    with pytest.raises(ValueError, match="indices"):
        view.set_visible_mesh_indices(indices, preserve_view=True)
    _assert_buffers_equal(before, _buffers(view))
