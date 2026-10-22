"""Public application identity and non-destructive preference compatibility."""

from __future__ import annotations

import os
from pathlib import Path
import warnings


APP_NAME = "TEM Design & Simulation"
APP_SUBTITLE = "An Interactive TEM/STEM Platform for Education and Research"
WINDOW_TITLE = f"{APP_NAME} — {APP_SUBTITLE}"
ORGANIZATION_NAME = APP_NAME
DISTRIBUTION_NAME = "tem-design-simulation"
RECORDER_NAME = f"{APP_NAME} Instrument Recorder"

# Compatibility identifiers are never used as product labels. Keep previous
# user settings and large caches intact instead of replacing or moving them.
_LEGACY_APPLICATIONS = ("TEMForge", "TEM Simulator v2")
_MIGRATION_KEY = "branding/preferences_imported_v2"


def _migrate_settings(destination, *previous_settings) -> None:
    """Import the newest existing settings once, preserving explicit resets.

    A previous migration marker counts as an existing store, even if every
    user preference was reset. Never fill its missing keys from an older brand.
    """
    if destination.value(_MIGRATION_KEY, False, type=bool):
        return
    for previous in previous_settings:
        previous.setFallbacksEnabled(False)
        keys = previous.allKeys()
        if not keys:
            continue
        for key in keys:
            if not key.startswith("branding/preferences_imported_") and not destination.contains(key):
                destination.setValue(key, previous.value(key))
        break
    destination.sync()
    if destination.status().value != 0:
        warnings.warn(f"{APP_NAME} preferences could not be imported; the original settings are retained.",
                      RuntimeWarning, stacklevel=2)
        return
    destination.setValue(_MIGRATION_KEY, True)
    destination.sync()


def application_settings():
    """Return the primary preferences, retaining saved layouts and cache limits."""
    from PySide6.QtCore import QSettings

    destination = QSettings(ORGANIZATION_NAME, APP_NAME)
    _migrate_settings(destination, QSettings("TEMForge", "TEMForge"),
                      QSettings("TEM Simulator", "TEM Simulator v2"))
    return destination


def recorder_settings():
    """Keep recorder output paths and geometry separate from workbench settings."""
    from PySide6.QtCore import QSettings

    destination = QSettings(ORGANIZATION_NAME, RECORDER_NAME)
    _migrate_settings(destination, QSettings("TEMForge", "TEMForge Instrument Recorder"),
                      QSettings("TEM Simulator", "Instrument Recorder"))
    return destination


def _application_storage_base() -> Path:
    local = str(os.environ.get("LOCALAPPDATA", "")).strip()
    return Path(local) if local else Path.home() / ".cache"


def application_cache_root(directory: str, *, reuse_legacy: bool = False) -> Path:
    """Use current storage; an existing legacy result store may stay in place.

    Reusing a cache never copies, renames or deletes saved scientific results.
    If both exist, the current branded store takes precedence.
    """
    base = _application_storage_base()
    current = base / APP_NAME / directory
    if reuse_legacy and not current.exists():
        for application in _LEGACY_APPLICATIONS:
            previous = base / application / directory
            if previous.is_dir():
                return previous
    return current


def application_result_library_root() -> Path:
    """Locate saved results independently of disposable calculation caches.

    A current index is authoritative, including an intentionally empty or
    invalid index. ResultLibrary reports invalid data rather than silently
    switching libraries. An unindexed new directory must not hide an older
    index. No directories or scientific result packages are moved or created.
    """
    base = _application_storage_base()
    current = base / APP_NAME / "saved_results"
    roots = (current, *(base / name / "saved_results" for name in _LEGACY_APPLICATIONS))

    def has_index(root):
        index = root / "index.json"
        # Broken symlinks must reach ResultLibrary's owned-index validation.
        return index.exists() or index.is_symlink()

    for root in roots:
        if has_index(root) or (root.exists() and not root.is_dir()):
            return root
    return next((root for root in roots if root.is_dir()), current)
