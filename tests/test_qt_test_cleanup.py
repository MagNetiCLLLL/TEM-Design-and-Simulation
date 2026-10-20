"""The suite must release a plot's Python cycles before starting another test."""
import os
from pathlib import Path
import subprocess
import sys


def test_nested_plot_cycles_do_not_survive_into_next_test(tmp_path):
    # A child process exercises actual pytest-qt teardown order. Tiny curves
    # suffice; this does not instantiate a microscope or run a numerical job.
    (tmp_path / "conftest.py").write_text(
        Path(__file__).with_name("conftest.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "test_plot_cleanup.py").write_text(
        '''
import weakref
import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
from temsim.gui.garbage_collection import install_gui_gc

retained = []

def make_plot(qtbot, qapp):
    install_gui_gc(qapp)
    # Only teardown may collect: no periodic GUI timer or automatic Python GC.
    qapp._temsim_gui_gc.timer.stop()
    owner = QWidget()
    layout = QVBoxLayout(owner)
    plot = pg.PlotWidget()
    owner.plot = plot
    # Pages retain providers/closures as well as their child plots.
    owner.current_plot = lambda: owner.plot
    layout.addWidget(plot)
    curve = plot.plot([0., 1., 2., 3.], [0., 1., -1., 0.])
    view = plot.getViewBox()
    retained.extend(weakref.ref(obj) for obj in
                    (owner, plot, curve, view, plot.getPlotItem().ctrlMenu))
    qtbot.addWidget(owner)
    return owner

def test_create_nested_plot(qtbot, qapp):
    make_plot(qtbot, qapp)

def test_next_case_starts_without_previous_plot(qapp):
    assert retained and all(ref() is None for ref in retained)
    assert len(pg.ViewBox.AllViews) == 0
    assert len(qapp.topLevelWidgets()) == 0

@pytest.fixture
def nested_plot(qtbot, qapp):
    return make_plot(qtbot, qapp)

def test_plot_returned_by_fixture(nested_plot):
    assert nested_plot.layout().count() == 1

def test_fixture_plot_released_before_next_case():
    # pytest keeps item.funcargs until after all teardown hooks have finished.
    # Cleanup also runs when the next case does not request a Qt fixture.
    assert retained and all(ref() is None for ref in retained)
    assert len(pg.ViewBox.AllViews) == 0
    assert len(QApplication.instance().topLevelWidgets()) == 0
''',
        encoding="utf-8",
    )
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": "-1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-m", "pytest", "-p", "pytestqt.plugin",
         "-q", "-x", "--tb=short", "test_plot_cleanup.py"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout
