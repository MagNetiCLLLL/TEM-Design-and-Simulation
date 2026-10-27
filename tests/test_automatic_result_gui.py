"""Automatic startup reuses executed particle files without running physics."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from temsim.automatic_result import AutomaticResult
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.particle_section_io import save_section_result, checked_section_archive_info
from temsim.result_library import ResultLibrary
from test_result_files_gui import gui_environment
from test_particle_section_io import executed_section


def _save_entry(root, result):
    path = root / "completed.temresult"
    save_section_result(result, path)
    info = checked_section_archive_info(result, path, maximum_unpacked_bytes=8 * 1024**3)
    store = AutomaticResult(root / "saved_results")
    token = store.begin_save(info["identity"])
    assert store.commit_save(token, info)
    return store, path, info


def _forbid_physics(monkeypatch):
    from temsim.gui import main_window, calculation_controller
    from temsim.optics.electron_gun import source
    from temsim.physics import simulation, particle_sections

    def forbidden(*args, **kwargs):
        pytest.fail("Automatic startup submitted transport or reapplied a preset")

    monkeypatch.setattr(main_window.MainWindow, "_apply_state_operating_modes", forbidden)
    monkeypatch.setattr(calculation_controller.CalculationController, "submit_background", forbidden)
    monkeypatch.setattr(source, "trace_source_to_exit", forbidden)
    monkeypatch.setattr(simulation, "run", forbidden)
    monkeypatch.setattr(particle_sections, "run_particle_section", forbidden)


def test_completed_automatic_archive_survives_new_window_without_recalculation(
    gui_environment, executed_section, monkeypatch, qtbot,
):
    make, errors, root = gui_environment
    first = make()
    first.calculations.archive_completed_section(executed_section)
    # Publication must complete in the worker even before Qt delivers its result
    # signal; closing just after file IO must not lose the startup association.
    assert first.calculations.section_file_pool.waitForDone(30000)
    entry = AutomaticResult(root / "saved_results").entry()
    assert entry is not None
    first.close()
    _forbid_physics(monkeypatch)
    second = make()
    assert second.result_files.automatic_startup
    qtbot.waitUntil(lambda: not second.result_files.loading, timeout=30000)
    assert not errors
    restored = second.workspace._last_result
    assert restored is not None and restored is not executed_section
    np.testing.assert_array_equal(restored.simulation.incident.x, executed_section.simulation.incident.x)
    np.testing.assert_array_equal(restored.simulation.incident.ray_weight, executed_section.simulation.incident.ray_weight)
    assert capture_instrument_snapshot(second.state).digest == capture_instrument_snapshot(executed_section.state_snapshot).digest
    assert second.workspace._ray_bundle_records
    assert second.workspace.interactive_calculation._section_result is restored
    assert "Restored automatic result; no recalculation" in second.status_label.text()
    assert not second.preview_timer.isActive()
    assert second.result_files.hold_automatic_preview


@pytest.mark.parametrize("corrupt_automatic_index", [False, True])
def test_named_startup_choice_has_priority_over_automatic_result(
    gui_environment, executed_section, monkeypatch, qtbot, corrupt_automatic_index,
):
    make, errors, root = gui_environment
    store, path, info = _save_entry(root, executed_section)
    library = ResultLibrary(root / "saved_results")
    selected = library.remember("Explicit startup", info)
    library.set_startup(selected["id"])
    if corrupt_automatic_index:
        store.index_path.write_bytes(b"Invalid automatic index")
    _forbid_physics(monkeypatch)
    window = make()
    assert not window.result_files.automatic_startup
    assert window.result_files.startup_entry == selected
    qtbot.waitUntil(lambda: not window.result_files.loading, timeout=30000)
    assert window.workspace._last_result is not None and not errors
    if corrupt_automatic_index:
        assert store.index_path.read_bytes() == b"Invalid automatic index"
        assert window.result_files.startup_error is None
    else:
        assert store.entry()["path"] == str(path)


def test_automatic_archive_reopens_in_a_fresh_process(executed_section, tmp_path):
    store, path, info = _save_entry(tmp_path, executed_section)
    script = """
import hashlib
import json
import sys
from temsim.automatic_result import AutomaticResult
from temsim.particle_section_io import load_section_result
from temsim.optics.electron_gun import source
from temsim.physics import simulation, particle_sections

def forbidden(*args, **kwargs):
    raise AssertionError('Restoring an archive must not run propagation')

source.trace_source_to_exit = forbidden
simulation.run = forbidden
particle_sections.run_particle_section = forbidden
entry = AutomaticResult(sys.argv[1]).entry()
restored = load_section_result(entry['path'])
branch = restored.simulation.incident
print(json.dumps({
    'x': hashlib.sha256(branch.x.tobytes()).hexdigest(),
    'weight': hashlib.sha256(branch.ray_weight.tobytes()).hexdigest(),
    'identity': entry['result_identity'],
}))
"""
    child = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script, str(store.index_path.parent)],
        capture_output=True, text=True, encoding="utf-8", timeout=90, check=True,
    )
    restored = json.loads(child.stdout.strip().splitlines()[-1])
    branch = executed_section.simulation.incident
    assert restored == {
        "x": hashlib.sha256(branch.x.tobytes()).hexdigest(),
        "weight": hashlib.sha256(branch.ray_weight.tobytes()).hexdigest(),
        "identity": info["identity"],
    }


def test_corrupt_automatic_index_falls_back_and_is_preserved(gui_environment, monkeypatch, qtbot):
    from temsim.gui import main_window
    make, errors, root = gui_environment
    store = AutomaticResult(root / "saved_results")
    store.index_path.parent.mkdir(parents=True, exist_ok=True)
    store.index_path.write_bytes(b"Invalid automatic index")
    fallbacks = []
    monkeypatch.setattr(main_window.MainWindow, "_resume_default_startup",
                        lambda self, reason: fallbacks.append(str(reason)))
    window = make()
    qtbot.waitUntil(lambda: bool(fallbacks), timeout=30000)
    assert not errors
    assert not window.result_files.loading
    assert not window.result_files.hold_automatic_preview
    assert window.workspace._last_result is None
    assert store.index_path.read_bytes() == b"Invalid automatic index"


@pytest.mark.parametrize("failure", ["missing", "corrupt", "replaced", "implementation"])
def test_invalid_automatic_result_falls_back_without_installing_stale_data(
    gui_environment, executed_section, monkeypatch, qtbot, failure,
):
    from temsim.gui import main_window
    from temsim import calculation_manifest
    make, errors, root = gui_environment
    store, path, info = _save_entry(root, executed_section)
    original_entry = store.entry()
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_bytes(b"Invalid automatic archive")
    elif failure == "replaced":
        # A replaced package may be otherwise valid; the pointer must still
        # bind its exact saved digest before any settings become active.
        document = store.index_path.read_text(encoding="utf-8")
        store.index_path.write_text(document.replace(info["_package_digest"], "e" * 64), encoding="utf-8")
    else:
        original_identity = calculation_manifest.solver_source_identity()
        monkeypatch.setattr(calculation_manifest, "solver_source_identity", lambda: original_identity + "-changed")
    fallbacks = []
    monkeypatch.setattr(main_window.MainWindow, "_resume_default_startup",
                        lambda self, reason: fallbacks.append(str(reason)))
    window = make()
    qtbot.waitUntil(lambda: bool(fallbacks), timeout=30000)
    assert not errors
    assert not window.result_files.loading
    assert not window.result_files.hold_automatic_preview
    assert window.workspace._last_result is None
    assert store.entry()["result_identity"] == original_entry["result_identity"]


def test_automatic_and_standard_startup_menu_preserve_named_results(
    gui_environment, executed_section, monkeypatch,
):
    make, errors, root = gui_environment
    window = make()
    store, path, info = _save_entry(root, executed_section)
    named = window.result_files.library.remember("Keep this result", info)
    window.result_files.set_startup(named["id"])
    window.result_files.set_automatic_startup()
    assert window.result_files.library.startup_entry() is None
    assert store.enabled()
    window.result_files.set_startup(None)
    assert not store.enabled()
    assert store.entry() is not None
    assert window.result_files.library.entries() == (named,)
    assert path.is_file() and not errors


def test_standard_fallback_prepares_default_mode_and_schedules_one_initial_preview(
    gui_environment, monkeypatch,
):
    make, errors, root = gui_environment
    window = make()
    before = window.workspace._last_result
    window._resume_default_startup("Automatic package is unavailable")
    assert not errors
    assert window.preview_timer.isActive()
    assert window.workspace._last_result is before
    assert window.state.simulation_mode == "unified"
    assert window.state.step_mm == 0.05
    lenses = {lens.key: lens for lens in window.state.lenses}
    assert lenses["condenser_lens_2"].percent == 13.899040164772698
    assert lenses["objective_lens"].percent == 68.97810600833917
    assert "starting a fresh calculation" in window.status_label.text()
