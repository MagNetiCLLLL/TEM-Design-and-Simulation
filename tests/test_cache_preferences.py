"""Cache configuration persistence without physical models or calculations."""
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from temsim import cache_preferences as cache
from temsim.gui import cache_settings as gui


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "cache.ini"), QSettings.Format.IniFormat)


@pytest.mark.parametrize("helper, new_application, old_organization, old_application", [
    ("application_settings", "TEM Design & Simulation", "TEM Simulator", "TEM Simulator v2"),
    ("recorder_settings", "TEM Design & Simulation Instrument Recorder", "TEM Simulator", "Instrument Recorder"),
    ("application_settings", "TEM Design & Simulation", "TEMForge", "TEMForge"),
    ("recorder_settings", "TEM Design & Simulation Instrument Recorder", "TEMForge", "TEMForge Instrument Recorder"),
])
def test_branding_import_preserves_settings_without_overwriting_or_resurrecting(
    tmp_path, monkeypatch, helper, new_application, old_organization, old_application,
):
    from PySide6 import QtCore
    from temsim import branding

    def isolated_settings(organization, application):
        return QSettings(str(tmp_path / organization / f"{application}.ini"),
                         QSettings.Format.IniFormat)

    legacy = isolated_settings(old_organization, old_application)
    geometry = QtCore.QByteArray(b"\x00saved-window-geometry\xff")
    legacy.setValue("geometry", geometry)
    legacy.setValue("output", str(tmp_path / "recordings"))
    legacy.setValue("workspace_layouts/v1/active", "alignment")
    preferences = replace(cache.CachePreferences(), high_cache_budget_bytes=12 * cache.GIB)
    cache.save_cache_preferences(legacy, preferences, 64 * cache.GIB)
    current = isolated_settings("TEM Design & Simulation", new_application)
    current.setValue("workspace_layouts/v1/active", "my-new-layout")
    current.sync()
    monkeypatch.setattr(QtCore, "QSettings", isolated_settings)

    imported = getattr(branding, helper)()
    assert imported.value("geometry") == geometry
    assert imported.value("output") == str(tmp_path / "recordings")
    assert imported.value("workspace_layouts/v1/active") == "my-new-layout"
    assert cache.load_cache_preferences(imported, 64 * cache.GIB) == preferences
    assert legacy.value("workspace_layouts/v1/active") == "alignment"
    assert set(legacy.allKeys()) == {"geometry", "output", "workspace_layouts/v1/active"} | {
        f"{cache.SETTINGS_ROOT}/{field.name}" for field in cache.fields(preferences)
    }

    # Resetting a preference after import must not restore the previous copy.
    imported.remove("geometry")
    imported.sync()
    reopened = getattr(branding, helper)()
    assert not reopened.contains("geometry")
    assert reopened.value("workspace_layouts/v1/active") == "my-new-layout"


@pytest.mark.parametrize("helper, previous_app, oldest_app", [
    ("application_settings", "TEMForge", "TEM Simulator v2"),
    ("recorder_settings", "TEMForge Instrument Recorder", "Instrument Recorder"),
])
@pytest.mark.parametrize("reset_all", [False, True])
def test_latest_brand_import_does_not_resurrect_settings_from_an_older_brand(
    tmp_path, monkeypatch, helper, previous_app, oldest_app, reset_all,
):
    from PySide6 import QtCore
    from temsim import branding

    def isolated_settings(organization, application):
        return QSettings(str(tmp_path / organization / f"{application}.ini"),
                         QSettings.Format.IniFormat)

    oldest = isolated_settings("TEM Simulator", oldest_app)
    oldest.setValue("geometry", "deleted-in-TEMForge")
    oldest.setValue("output", "obsolete-output")
    oldest.sync()
    previous = isolated_settings("TEMForge", previous_app)
    previous.setValue("branding/preferences_imported_v1", True)
    if not reset_all:
        previous.setValue("output", "latest-output")
    previous.sync()
    before = {key: previous.value(key) for key in previous.allKeys()}
    monkeypatch.setattr(QtCore, "QSettings", isolated_settings)

    imported = getattr(branding, helper)()
    assert not imported.contains("geometry")
    assert imported.value("output") == (None if reset_all else "latest-output")
    assert not imported.contains("branding/preferences_imported_v1")
    assert imported.value(branding._MIGRATION_KEY, type=bool)
    assert {key: previous.value(key) for key in previous.allKeys()} == before


def _saved_library(root, name):
    from temsim.result_library import ResultLibrary

    library = ResultLibrary(root)
    package = library.allocate_path()
    package.write_bytes(b"Result-library path fixture; not a calculation payload.")
    entry = library.remember(name, {
        "path": str(package), "identity": "a" * 64, "_package_digest": "b" * 64,
        "quality": "Preview", "target_z_mm": 1000., "resumable_through_z_mm": 1000.,
        "saved_at_utc": "2026-10-22T00:00:00+00:00",
    })
    library.set_startup(entry["id"])
    return library, entry


@pytest.mark.parametrize("legacy_cache", [False, True])
@pytest.mark.parametrize("current_cache", [False, True])
@pytest.mark.parametrize("empty_current_library", [False, True])
@pytest.mark.parametrize("previous_name", ["TEMForge", "TEM Simulator v2"])
def test_saved_results_and_startup_survive_independent_cache_directories(
    tmp_path, monkeypatch, legacy_cache, current_cache, empty_current_library, previous_name,
):
    from temsim.branding import application_result_library_root
    from temsim.result_library import ResultLibrary

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    legacy, entry = _saved_library(tmp_path / previous_name / "saved_results", "Old result")
    index_before = legacy.index_path.read_bytes()
    package_before = Path(entry["path"]).read_bytes()
    current = tmp_path / "TEM Design & Simulation" / "saved_results"
    for application, enabled in ((previous_name, legacy_cache), ("TEM Design & Simulation", current_cache)):
        if enabled:
            (tmp_path / application / "high_accuracy_artifacts").mkdir(parents=True)
    if empty_current_library:
        current.mkdir(parents=True, exist_ok=True)

    selected = ResultLibrary(application_result_library_root())
    assert selected.root == legacy.root
    assert selected.entries() == (entry,)
    assert selected.startup_entry() == entry
    assert legacy.index_path.read_bytes() == index_before
    assert Path(entry["path"]).read_bytes() == package_before
    assert not (current / "index.json").exists()
    assert current.exists() == empty_current_library


@pytest.mark.parametrize("current_contents", ["saved", "empty", "corrupt"])
def test_current_saved_index_is_authoritative_even_when_empty_or_corrupt(
    tmp_path, monkeypatch, current_contents,
):
    from temsim.branding import application_result_library_root
    from temsim.result_library import ResultLibrary

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    legacy, _ = _saved_library(tmp_path / "TEM Simulator v2" / "saved_results", "Old result")
    current, entry = _saved_library(tmp_path / "TEM Design & Simulation" / "saved_results", "New result")
    if current_contents == "empty":
        current.index_path.write_text(
            '{"schema":"temsim-result-library-v1","entries":[],"startup_entry_id":null}',
            encoding="utf-8",
        )
    elif current_contents == "corrupt":
        current.index_path.write_text("broken", encoding="utf-8")
    before = {root.index_path: root.index_path.read_bytes() for root in (legacy, current)}

    assert application_result_library_root() == current.root
    if current_contents == "corrupt":
        with pytest.raises(ValueError, match="Malformed result library index"):
            ResultLibrary(application_result_library_root())
    else:
        selected = ResultLibrary(application_result_library_root())
        assert selected.entries() == ((entry,) if current_contents == "saved" else ())
        assert selected.startup_entry() == (entry if current_contents == "saved" else None)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("previous_contents", ["missing", "empty", "saved", "corrupt"])
def test_saved_library_checks_legacy_indexes_in_order_without_hiding_old_results(
    tmp_path, monkeypatch, previous_contents,
):
    from temsim.branding import application_result_library_root
    from temsim.result_library import ResultLibrary

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    oldest, _ = _saved_library(tmp_path / "TEM Simulator v2" / "saved_results", "Oldest")
    previous = tmp_path / "TEMForge" / "saved_results"
    previous.mkdir(parents=True)
    current = tmp_path / "TEM Design & Simulation" / "saved_results"
    current.mkdir(parents=True)
    if previous_contents != "missing":
        library, _ = _saved_library(previous, "Previous")
        if previous_contents == "empty":
            library.index_path.write_text(
                '{"schema":"temsim-result-library-v1","entries":[],"startup_entry_id":null}',
                encoding="utf-8",
            )
        elif previous_contents == "corrupt":
            library.index_path.write_text("broken", encoding="utf-8")
    assert application_result_library_root() == (oldest.root if previous_contents == "missing" else previous)
    if previous_contents == "corrupt":
        with pytest.raises(ValueError, match="Malformed result library index"):
            ResultLibrary(application_result_library_root())
    assert not (current / "index.json").exists()


@pytest.mark.parametrize("local_appdata", [True, False])
def test_fresh_result_library_resolution_does_not_create_storage(tmp_path, monkeypatch, local_appdata):
    from temsim.branding import application_result_library_root

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    if local_appdata:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        base = tmp_path
    else:
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        base = tmp_path / ".cache"
    assert application_result_library_root() == base / "TEM Design & Simulation" / "saved_results"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("ram_gib", [1, 2, 4, 8, 16, 32, 128])
def test_hardware_defaults_are_bounded(ram_gib):
    prefs = cache.default_cache_preferences(ram_gib * cache.GIB)
    cache.validate_cache_preferences(prefs, ram_gib * cache.GIB)
    assert prefs.managed_ram_budget_bytes <= ram_gib * cache.GIB // 2
    assert prefs.high_cache_budget_bytes <= 8 * cache.GIB
    assert prefs.tuning_cache_budget_bytes <= cache.GIB
    assert prefs.ray_display_cache_budget_bytes <= 512 * cache.MIB
    assert prefs.prepared_specimen_cache_budget_bytes <= 256 * cache.MIB
    assert prefs.sample_display_cache_budget_bytes <= 128 * cache.MIB


def test_unknown_hardware_defaults_and_controller_api():
    prefs = cache.default_cache_preferences()
    assert prefs == cache.CachePreferences()
    assert prefs.controller_kwargs() == {
        "high_cache_budget_bytes": 8 * cache.GIB,
        "tuning_cache_budget_bytes": cache.GIB,
        "disk_cache_budget_bytes": 16 * cache.GIB,
        "high_cache_limit": 32,
        "tuning_cache_limit": 128,
    }


@pytest.mark.parametrize("bad_value", [True, False, "nan", "inf", "1.2", -1, 0, str(2**65), None])
def test_corrupt_saved_preference_falls_back_safely(settings, bad_value):
    settings.setValue(f"{cache.SETTINGS_ROOT}/high_cache_budget_bytes", bad_value)
    assert cache.load_cache_preferences(settings, 4 * cache.GIB) == cache.default_cache_preferences(4 * cache.GIB)


def test_persistence_is_separate_and_validates_all_fields(settings):
    settings.setValue("workspace_layouts/v1/active", "alignment")
    prefs = replace(cache.CachePreferences(), high_cache_budget_bytes=12 * cache.GIB)
    cache.save_cache_preferences(settings, prefs, 64 * cache.GIB)
    reopened = QSettings(settings.fileName(), QSettings.Format.IniFormat)
    assert cache.load_cache_preferences(reopened, 64 * cache.GIB) == prefs
    assert reopened.value("workspace_layouts/v1/active") == "alignment"
    assert cache.load_cache_preferences(reopened, 8 * cache.GIB) == cache.default_cache_preferences(8 * cache.GIB)
    with pytest.raises(ValueError, match="half"):
        cache.save_cache_preferences(settings, prefs, 8 * cache.GIB)
    assert cache.load_cache_preferences(settings, 64 * cache.GIB) == prefs
    with pytest.raises(ValueError, match="entry"):
        cache.validate_cache_preferences(replace(prefs, tuning_cache_limit=0))


def test_invalid_limits_do_not_write(settings):
    with pytest.raises(ValueError):
        cache.save_cache_preferences(settings, replace(cache.CachePreferences(), disk_cache_budget_bytes=2048 * cache.GIB), None)
    assert settings.allKeys() == []


def test_memory_detection_is_read_only_and_optional():
    total = cache.detect_total_memory_bytes()
    assert total is None or isinstance(total, int) and total >= 128 * cache.MIB


def test_dialog_only_applies_explicit_changes_and_stats_on_request(qtbot, monkeypatch, settings):
    monkeypatch.setattr(gui, "detect_total_memory_bytes", lambda: 64 * cache.GIB)
    stats_calls, changes = [], []

    def stats():
        stats_calls.append(True)
        return {"calculation": {"high_used_bytes": 2 * cache.MIB, "high_entries": 3, "high_hits": 4},
                "ray_display": {"used_bytes": cache.MIB, "entries": 2, "hits": 7}}

    dialog = gui.CacheSettingsDialog(settings, stats)
    qtbot.addWidget(dialog)
    dialog.preferencesChanged.connect(changes.append)
    assert not dialog.isModal()
    assert stats_calls == []
    dialog.show()
    qtbot.wait(20)
    assert len(stats_calls) == 1
    assert "High accuracy: 2.0 MiB | 3 entries | 4 hits" in dialog.statistics.text()
    dialog.high_cache.setValue(12)
    qtbot.wait(30)
    assert changes == []
    assert len(stats_calls) == 1
    assert dialog.apply_preferences()
    assert len(changes) == 1
    assert changes[0].high_cache_budget_bytes == 12 * cache.GIB
    assert cache.load_cache_preferences(settings, 64 * cache.GIB) == changes[0]
    assert len(stats_calls) == 2
    dialog.refresh_button.click()
    assert len(stats_calls) == 3
    dialog.high_cache.setValue(40)
    assert not dialog.apply_preferences()
    assert "half" in dialog.feedback.text()
    assert len(changes) == 1
    dialog.close()
    assert dialog.high_cache.value() == 12


def test_dialog_defaults_require_apply_and_geometry_restores(qtbot, monkeypatch, settings):
    monkeypatch.setattr(gui, "detect_total_memory_bytes", lambda: 64 * cache.GIB)
    initial = replace(cache.CachePreferences(), high_cache_budget_bytes=12 * cache.GIB)
    cache.save_cache_preferences(settings, initial, 64 * cache.GIB)
    dialog = gui.CacheSettingsDialog(settings)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.resize(690, 580)
    qtbot.wait(20)
    geometry = dialog.saveGeometry()
    dialog.restore_defaults()
    assert dialog.high_cache.value() == 8
    assert cache.load_cache_preferences(settings, 64 * cache.GIB) == initial
    dialog.close()
    assert settings.value(f"{cache.SETTINGS_ROOT}/dialog_geometry") == geometry
    restored = gui.CacheSettingsDialog(settings)
    qtbot.addWidget(restored)
    assert restored.size() == dialog.size()
    assert restored.high_cache.value() == 12


def test_missing_statistics_provider_and_failed_provider_do_not_crash(qtbot, settings):
    dialog = gui.CacheSettingsDialog(settings)
    qtbot.addWidget(dialog)
    dialog.refresh_statistics()
    assert dialog.statistics.text() == "Cache statistics unavailable."
    dialog.statistics_provider = lambda: None
    dialog.refresh_statistics()
    assert dialog.statistics.text() == "Cache statistics unavailable."


def test_external_preferences_refresh_does_not_emit_or_save(qtbot, monkeypatch, settings):
    monkeypatch.setattr(gui, "detect_total_memory_bytes", lambda: 64 * cache.GIB)
    dialog = gui.CacheSettingsDialog(settings)
    qtbot.addWidget(dialog)
    changes = []
    dialog.preferencesChanged.connect(changes.append)
    prefs = replace(cache.CachePreferences(), high_cache_budget_bytes=12 * cache.GIB)
    dialog.set_preferences(prefs)
    assert dialog.high_cache.value() == 12
    assert dialog.preferences == prefs
    assert changes == []
    assert settings.allKeys() == []


def test_dialog_reports_disk_availability_without_scanning(qtbot, settings):
    data = {"calculation": {"disk_enabled": False, "disk_budget_bytes": 0}}
    dialog = gui.CacheSettingsDialog(settings, lambda: data)
    qtbot.addWidget(dialog)
    dialog.refresh_statistics()
    assert "Disk checkpoints: unavailable; memory caching remains active." in dialog.statistics.text()
    data["calculation"] = {"disk_enabled": True, "disk_budget_bytes": 16 * cache.GIB}
    dialog.refresh_statistics()
    assert "Disk checkpoints: enabled | 16 GiB retention limit" in dialog.statistics.text()


def test_new_cache_budgets_persist_without_changing_physics(settings):
    prefs = replace(cache.CachePreferences(), prepared_specimen_cache_budget_bytes=2 * cache.GIB,
                    sample_display_cache_budget_bytes=256 * cache.MIB)
    cache.save_cache_preferences(settings, prefs, 64 * cache.GIB)
    assert cache.load_cache_preferences(settings, 64 * cache.GIB) == prefs
    assert prefs.managed_ram_budget_bytes == (
        prefs.high_cache_budget_bytes + prefs.tuning_cache_budget_bytes
        + prefs.ray_display_cache_budget_bytes + 2 * cache.GIB + 256 * cache.MIB
        + prefs.input_asset_cache_budget_bytes)
    assert "prepared_specimen_cache_budget_bytes" not in prefs.controller_kwargs()
    assert "sample_display_cache_budget_bytes" not in prefs.controller_kwargs()
    assert "input_asset_cache_budget_bytes" not in prefs.controller_kwargs()


def test_dialog_displays_independent_specimen_cache_hit_rates(qtbot, monkeypatch, settings):
    monkeypatch.setattr(gui, "detect_total_memory_bytes", lambda: 64 * cache.GIB)
    data = {"prepared_specimen": {"used_bytes": cache.MIB, "entries": 1, "hits": 3, "misses": 1},
            "sample_display": {"used_bytes": 0, "entries": 0, "hits": 0, "misses": 0}}
    dialog = gui.CacheSettingsDialog(settings, lambda: data)
    qtbot.addWidget(dialog)
    dialog.refresh_statistics()
    assert "Specimen potentials: 1.0 MiB | 1 entries | 3 hits | 75.0% hit rate" in dialog.statistics.text()
    assert "Sample atom display: 0.0 MiB | 0 entries | 0 hits | not used" in dialog.statistics.text()
    dialog.prepared_specimen_cache.setValue(512)
    dialog.sample_display_cache.setValue(192)
    assert dialog.apply_preferences()
    prefs = cache.load_cache_preferences(settings, 64 * cache.GIB)
    assert prefs.prepared_specimen_cache_budget_bytes == 512 * cache.MIB
    assert prefs.sample_display_cache_budget_bytes == 192 * cache.MIB
