"""Window routing preserves classical source editing and uses accepted particles."""
from types import SimpleNamespace
import pytest
from PySide6.QtWidgets import QPushButton
from temsim.instrument_snapshot import capture_instrument_snapshot
from test_working_point_restore_gui import window


def test_particle_toolbar_and_mixed_page_have_separate_entries(window):
    page = window.workspace.coherent_beam
    assert page.calculate_button.text() == "Calculate mixed wave"
    assert not hasattr(page, "tip_boundary") and not hasattr(page, "state_list")
    assert window.findChild(QPushButton, "highAccuracyButton") is not None
    assert window.classical_rays_action in window.simulation_menu.actions()


def test_particle_toolbar_preserves_optics_and_does_not_start_a_wave(window, monkeypatch):
    before = capture_instrument_snapshot(window.state)
    calls = []
    page = window.workspace.coherent_beam
    monkeypatch.setattr(page, "calculate", lambda: pytest.fail("Particle toolbar started a wave"))
    monkeypatch.setattr(window, "_submit_high_accuracy", lambda **kwargs: calls.append(kwargs))
    window.findChild(QPushButton, "highAccuracyButton").click()
    assert calls == [{"workflow": "rays"}]
    assert capture_instrument_snapshot(window.state).digest == before.digest


def test_mixed_page_requires_an_accepted_particle_result_without_retracing(window, monkeypatch):
    page = window.workspace.coherent_beam
    monkeypatch.setattr(window.calculations, "submit_background", lambda *a, **k:
                        pytest.fail("Mixed page must reuse accepted particles"))
    window.workspace._last_result = None
    page.calculate()
    assert page._worker is None
    assert "Calculate particles first" in page.status.text()


def test_window_provider_rejects_stale_particle_snapshot(window):
    sentinel = object()
    workspace = window.workspace
    workspace._last_result = sentinel
    workspace.transverse_beam._inputs_stale = False
    assert workspace.coherent_beam.particle_result_provider() is sentinel
    workspace.transverse_beam._inputs_stale = True
    assert workspace.coherent_beam.particle_result_provider() is None


def test_classical_menu_keeps_explicit_ray_workflow(window, monkeypatch):
    submitted = []
    monkeypatch.setattr(window, "_submit_high_accuracy", lambda **kwargs: submitted.append(kwargs))
    window.classical_rays_action.trigger()
    assert submitted == [{"workflow": "rays"}]


def test_shared_projection_updates_beam_observation_without_calculating(window, monkeypatch):
    page = window.workspace.coherent_beam
    projected = []
    before = capture_instrument_snapshot(window.state)
    monkeypatch.setattr(page, "set_projection_angle", projected.append)
    monkeypatch.setattr(page, "calculate", lambda: pytest.fail("Projection started a beam calculation"))

    window.workspace._set_projection_angle(37.5)

    assert projected == [37.5]
    assert capture_instrument_snapshot(window.state).digest == before.digest
