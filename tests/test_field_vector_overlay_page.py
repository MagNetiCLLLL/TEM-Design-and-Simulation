"""Captured CAD lifecycle must not alter or resample scientific field arrays."""
from threading import Event
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import Qt

from temsim.part_model_3d import TriangleMesh
from test_field_vector_page import activate, setup_page, wait_sample


def geometry(offset=0.):
    mesh = TriangleMesh(np.array([[0., 0., .002], [.01, 0., .002], [0., .01, .002]]) + offset,
                        np.array([[0, 1, 2]]), 'coil', material_class='copper')
    return SimpleNamespace(meshes=(mesh,), source_mesh_keys={'magnet': ('coil',)},
                           notes=('Captured test geometry.',), errors=(), source_notes={},
                           select=lambda keys: (mesh,) if 'magnet' in keys else ())


def wait_geometry(qtbot, page):
    qtbot.waitUntil(lambda: page._geometry is not None and page._geometry_worker is None, timeout=10000)


def test_overlay_is_lazy_and_opacity_camera_visibility_do_not_resample(qtbot, monkeypatch):
    page, scene, _ = setup_page(qtbot, monkeypatch)
    calls = []
    model = geometry()
    monkeypatch.setattr('temsim.field_vector_geometry.build_field_vector_geometry',
                        lambda state, sources: (calls.append((state, sources)), model)[1])
    activate(qtbot, page)
    assert calls == []
    samples = page._sample
    count = len(scene.calls)
    page.show_models.setChecked(True)
    wait_geometry(qtbot, page)
    assert calls[0][0] is page._state and calls[0][1] is scene.sources
    assert page.fit_models.isEnabled()
    assert 'no material model' in page.model_status.text()
    assert 'Captured test geometry' in page.details.toPlainText()
    page.model_opacity.setValue(72)
    assert page.model_opacity_label.text() == '72%'
    page.fit_models.click()
    page.fit_view.click()
    page.show_models.setChecked(False)
    page.show_models.setChecked(True)
    qtbot.wait(150)
    assert len(calls) == 1 and len(scene.calls) == count
    assert page._sample is samples


def test_selection_changes_reuse_mesh_capture_and_reapply_after_field_sampling(qtbot, monkeypatch):
    page, scene, _ = setup_page(qtbot, monkeypatch)
    calls = []
    model = geometry()
    monkeypatch.setattr('temsim.field_vector_geometry.build_field_vector_geometry',
                        lambda *args: (calls.append(args), model)[1])
    activate(qtbot, page)
    page.show_models.setChecked(True)
    wait_geometry(qtbot, page)
    page.sources.item(0).setCheckState(Qt.CheckState.Unchecked)
    wait_sample(qtbot, page)
    assert not page.fit_models.isEnabled()
    page.sources.item(0).setCheckState(Qt.CheckState.Checked)
    wait_sample(qtbot, page)
    assert page.fit_models.isEnabled()
    assert len(calls) == 1


def test_geometry_is_invalidated_with_inputs_and_new_snapshot_rebuilds(qtbot, monkeypatch):
    page, _, _ = setup_page(qtbot, monkeypatch)
    models = [geometry(), geometry(.003)]
    calls = []
    monkeypatch.setattr('temsim.field_vector_geometry.build_field_vector_geometry',
                        lambda *args: (calls.append(args), models[len(calls)-1])[1])
    activate(qtbot, page)
    page.show_models.setChecked(True)
    wait_geometry(qtbot, page)
    page.mark_inputs_stale()
    assert page._geometry is None and not page.fit_models.isEnabled()
    page._ensure_geometry()
    qtbot.wait(120)
    assert len(calls) == 1
    page.update_snapshot(SimpleNamespace(beam_voltage_kv=200.))
    wait_sample(qtbot, page)
    wait_geometry(qtbot, page)
    assert page._geometry is models[1] and len(calls) == 2
    page.invalidate()
    assert page._geometry is None and not page.fit_models.isEnabled()


def test_late_geometry_cannot_repopulate_newer_snapshot(qtbot, monkeypatch):
    page, _, _ = setup_page(qtbot, monkeypatch)
    entered, release = Event(), Event()
    old, new = geometry(), geometry(.004)
    first_state = SimpleNamespace(beam_voltage_kv=200.)

    def build(state, _sources):
        if state is first_state:
            entered.set()
            assert release.wait(10.)
            return old
        return new

    monkeypatch.setattr('temsim.field_vector_geometry.build_field_vector_geometry', build)
    page.update_snapshot(first_state)
    page.set_active(True)
    wait_sample(qtbot, page)
    page.show_models.setChecked(True)
    qtbot.waitUntil(entered.is_set, timeout=10000)
    try:
        page.update_snapshot(SimpleNamespace(beam_voltage_kv=300.))
        # Numerical jobs share a resource lease. Let the cancelled old build
        # return after replacement so its queued publication must be rejected.
        release.set()
        wait_sample(qtbot, page)
        wait_geometry(qtbot, page)
        assert page._geometry is new
    finally:
        release.set()
    qtbot.waitUntil(lambda: not page._geometry_jobs, timeout=10000)
    assert page._geometry is new


def test_geometry_failure_keeps_valid_field_and_can_retry(qtbot, monkeypatch):
    page, scene, _ = setup_page(qtbot, monkeypatch)
    def broken(*_args):
        raise ValueError('No captured assembly')
    monkeypatch.setattr('temsim.field_vector_geometry.build_field_vector_geometry', broken)
    activate(qtbot, page)
    samples = page._sample
    count = len(scene.calls)
    page.show_models.setChecked(True)
    qtbot.waitUntil(lambda: page._geometry_error is not None, timeout=10000)
    assert 'No captured assembly' in page.model_status.text()
    assert page._sample is samples and page.canvas.sample is samples
    monkeypatch.setattr('temsim.field_vector_geometry.build_field_vector_geometry', lambda *_args: geometry())
    page.show_models.setChecked(False)
    page.show_models.setChecked(True)
    wait_geometry(qtbot, page)
    assert len(scene.calls) == count and page._sample is samples


def test_hidden_geometry_worker_cannot_publish_until_reopened(qtbot, monkeypatch):
    page, _, _ = setup_page(qtbot, monkeypatch)
    entered, release = Event(), Event()
    def build(*_args):
        entered.set()
        assert release.wait(10.)
        return geometry()
    monkeypatch.setattr('temsim.field_vector_geometry.build_field_vector_geometry', build)
    activate(qtbot, page)
    page.show_models.setChecked(True)
    qtbot.waitUntil(entered.is_set, timeout=10000)
    page.set_active(False)
    release.set()
    qtbot.waitUntil(lambda: not page._geometry_jobs, timeout=10000)
    assert page._geometry is None
    page.set_active(True)
    wait_geometry(qtbot, page)
    assert page.fit_models.isEnabled()
