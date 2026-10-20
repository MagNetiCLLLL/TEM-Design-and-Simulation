"""Choosing a device policy must not initialise a native GPU driver."""
from PySide6.QtCore import QSettings

from temsim.physics import compute_backend as backend


def test_window_and_backend_selector_defer_gpu_discovery(qtbot, monkeypatch, tmp_path):
    # Record calls rather than raising: display wrappers used to catch errors,
    # hiding the fact that startup had attempted to initialise a GPU driver.
    probes = []

    def probe():
        probes.append(True)
        return backend.BackendCapability(False, "GPU must not be queried by UI")

    monkeypatch.setattr(backend, "cuda_capability", probe)
    monkeypatch.setattr(backend, "cupy_capability", probe)
    from temsim.gui import main_window as shell

    settings = QSettings(str(tmp_path / "discovery.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    # Cover old imported aliases as well, so this test detects the former eager
    # probe even if another test already imported the main-window module.
    for name in ("cuda_capability", "cupy_capability"):
        if hasattr(shell, name):
            monkeypatch.setattr(shell, name, probe)
    window = shell.MainWindow()
    window.preview_timer.stop()
    qtbot.addWidget(window)
    assert not probes

    for policy in backend.BACKEND_CHOICES:
        window.compute_backend.setCurrentIndex(window.compute_backend.findData(policy))
        assert window.state.acceleration_backend == policy
        assert window.state.acceleration_enabled is (policy != backend.BACKEND_CPU)
        assert not window.preview_timer.isActive()
        assert not probes
    assert window.calculations.pool.activeThreadCount() == 0
