"""Exact-plane switching with recorded rays and a substituted wave solve.

The fixtures exercise the Qt entry point, job delivery and wave presentation.
They do not launch particle tracing or coherent propagation.
"""

from copy import deepcopy
from dataclasses import replace
from threading import Event

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QThread, Qt

import temsim.gui.coherent_beam as coherent_beam
from temsim.gui.visualization import VisualizationWorkspace
from test_coherent_beam_gui import panel, result as wave_result
from test_incremental_ray_scene import _result as ray_result
from test_wave_beam_analysis import checkpoint


@pytest.fixture
def workspace(panel, qtbot, monkeypatch):
    monkeypatch.setattr(coherent_beam, "CoherentBeamPage", lambda: panel)
    view = VisualizationWorkspace()
    qtbot.addWidget(view)
    monkeypatch.setattr(view, "_update_interaction_detail", lambda: None)
    monkeypatch.setattr(view.selected_plane_readout, "select_z", lambda _z: None)
    view.show()
    panel.source_enabled.setChecked(True)
    panel.apply_source()
    monkeypatch.setattr(coherent_beam, "prepare_coherent_state", deepcopy)
    yield view
    view.transverse_beam.coherent_wave_toggle.setChecked(False)
    assert view.selected_plane_readout.shutdown()


def completed_wave(z_mm):
    output = wave_result(z_mm)
    output.checkpoint = replace(checkpoint(), plane_z_mm=z_mm)
    return output


def recorded_particles(panel, **kwargs):
    rays = ray_result(**kwargs)
    rays.state_snapshot.electron_gun = deepcopy(panel._state.electron_gun)
    rays.state_snapshot.simulation_time_s = 1.25e-9
    return rays


def displayed_at(transverse, z_mm):
    wave = transverse.analysis.wave
    return wave is not None and wave.checkpoint.plane_z_mm == z_mm and wave.values is not None


@pytest.mark.parametrize(("quality", "count"), [("Preview", 49), ("High accuracy", 8192)])
def test_transverse_wave_follows_selected_plane_and_restores_particles(
        workspace, panel, qtbot, monkeypatch, quality, count):
    calls = []

    def solve(_state, request, **_kwargs):
        calls.append((request.observation_z_mm, request.tip_time_s,
                      request.wave_grid.compute_backend, QThread.currentThread() == panel.thread()))
        return completed_wave(request.observation_z_mm)

    monkeypatch.setattr(coherent_beam, "simulate_tip_wave", solve)
    rays = recorded_particles(panel, count=count, stops=False)
    rays.state_snapshot.acceleration_backend = "CPU"
    rays.state_snapshot.acceleration_enabled = False
    job_settings = (rays.state_snapshot.simulation_time_s, "Require GPU", False)
    workspace.display_result(rays, quality, calculation_scope="rays")
    transverse = workspace.transverse_beam
    first_z = 11.123456789123
    workspace.jump_to_ray_position(first_z, activate_tab=False)
    assert transverse.isVisible()
    assert transverse._scatter is not None
    assert transverse.analysis.display_combo.currentData() == "particles"
    assert not transverse.coherent_wave_toggle.isChecked()
    assert calls == []

    qtbot.mouseClick(transverse.coherent_wave_toggle, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: displayed_at(transverse, first_z), timeout=5000)
    assert calls == [(first_z, *job_settings)]
    assert any(isinstance(item, pg.ImageItem) for item in transverse.plot.items())

    second_z = 12.987654321987
    workspace.axial_cursor_item.setValue(second_z)
    qtbot.waitUntil(lambda: displayed_at(transverse, second_z), timeout=5000)
    assert calls == [(first_z, *job_settings), (second_z, *job_settings)]
    assert panel.result is None

    qtbot.mouseClick(transverse.coherent_wave_toggle, Qt.MouseButton.LeftButton)
    assert transverse.analysis.wave is None
    assert transverse.analysis.display_combo.currentData() == "particles"
    assert transverse._scatter is not None
    assert transverse._result is rays
    assert rays.simulation.incident.x.shape[1] == count
    assert rays.state_snapshot.acceleration_backend == "CPU"
    assert not rays.state_snapshot.acceleration_enabled
    workspace.axial_cursor_item.setValue(13.5)
    assert transverse.analysis.wave is None
    assert calls == [(first_z, *job_settings), (second_z, *job_settings)]


def test_transverse_wave_rejects_obsolete_planes_and_falls_back_on_failure(
        workspace, panel, qtbot, monkeypatch):
    release = Event()
    started = Event()
    first_z, second_z = 11.25, 12.25

    def solve(_state, request, **_kwargs):
        if request.observation_z_mm == first_z:
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test fixture did not release the first observation")
            return completed_wave(first_z)
        raise RuntimeError("Synthetic propagation failure")

    monkeypatch.setattr(coherent_beam, "simulate_tip_wave", solve)
    workspace.display_result(recorded_particles(panel), "Preview", calculation_scope="rays")
    workspace.jump_to_ray_position(first_z, activate_tab=False)
    transverse = workspace.transverse_beam
    transverse.coherent_wave_toggle.setChecked(True)
    try:
        qtbot.waitUntil(started.is_set, timeout=5000)
        workspace.axial_cursor_item.setValue(second_z)
        panel.transverse_wave_ready.emit(completed_wave(first_z).checkpoint, .125, None)
        assert transverse.analysis.wave is None
        assert transverse._scatter is not None
    finally:
        release.set()

    qtbot.waitUntil(lambda: "Synthetic propagation failure" in transverse.coherent_wave_status.text(),
                    timeout=5000)
    assert transverse.analysis.wave is None
    assert transverse._scatter is not None
    transverse.coherent_wave_toggle.setChecked(False)
    panel.transverse_wave_ready.emit(completed_wave(second_z).checkpoint, .125, None)
    assert transverse.analysis.wave is None
    assert transverse._scatter is not None


def test_transverse_wave_requires_usable_cuda_before_starting_job(workspace, panel, qtbot, monkeypatch):
    from temsim.physics import compute_backend

    monkeypatch.setattr(compute_backend, "coherent_cuda_capability", lambda:
        compute_backend.BackendCapability(False, "Synthetic device unavailable"))
    queued = []
    monkeypatch.setattr(panel._transverse_pool, "start", queued.append)
    workspace.display_result(recorded_particles(panel), "Preview", calculation_scope="rays")
    workspace.jump_to_ray_position(11.25, activate_tab=False)
    transverse = workspace.transverse_beam
    transverse.coherent_wave_toggle.setChecked(True)

    qtbot.waitUntil(lambda: "Synthetic device unavailable" in transverse.coherent_wave_status.text())
    assert "CUDA" in transverse.coherent_wave_status.text()
    assert transverse.analysis.wave is None
    assert transverse._scatter is not None
    assert queued == []


def test_transverse_wave_restarts_at_same_z_after_republication(workspace, panel, qtbot, monkeypatch):
    calls = []

    def solve(_state, request, **_kwargs):
        calls.append(request.tip_time_s)
        return completed_wave(request.observation_z_mm)

    monkeypatch.setattr(coherent_beam, "simulate_tip_wave", solve)
    rays = recorded_particles(panel)
    workspace.display_result(rays, "Preview", calculation_scope="rays")
    z_mm = 11.25
    workspace.jump_to_ray_position(z_mm, activate_tab=False)
    transverse = workspace.transverse_beam
    transverse.coherent_wave_toggle.setChecked(True)
    qtbot.waitUntil(lambda: displayed_at(transverse, z_mm), timeout=5000)
    assert calls == [1.25e-9]

    rays.state_snapshot.simulation_time_s = 2.5e-9
    workspace.display_result(rays, "Preview", calculation_scope="rays")
    assert transverse.analysis.wave is None
    assert transverse._scatter is not None
    qtbot.waitUntil(lambda: displayed_at(transverse, z_mm), timeout=5000)
    assert calls == [1.25e-9, 2.5e-9]


def test_transverse_wave_keeps_particles_when_intensity_readout_fails(
        workspace, panel, qtbot, monkeypatch):
    from temsim.gui import wave_beam_analysis

    monkeypatch.setattr(coherent_beam, "simulate_tip_wave",
                        lambda _state, request, **_kwargs: completed_wave(request.observation_z_mm))

    def fail_readout(_request, _cancel_event):
        raise MemoryError("Synthetic intensity readout failure")

    monkeypatch.setattr(wave_beam_analysis, "_derive_view", fail_readout)
    workspace.display_result(recorded_particles(panel), "High accuracy", calculation_scope="rays")
    workspace.jump_to_ray_position(11.25, activate_tab=False)
    transverse = workspace.transverse_beam
    transverse.coherent_wave_toggle.setChecked(True)

    qtbot.waitUntil(lambda: "Synthetic intensity readout failure" in transverse.coherent_wave_status.text(),
                    timeout=5000)
    assert transverse.analysis.wave is None
    assert transverse._scatter is not None
    assert transverse.analysis.display_combo.currentData() == "particles"
