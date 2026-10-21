"""Public application identity and non-destructive preference compatibility."""

from __future__ import annotations

import os
from pathlib import Path
import warnings


APP_NAME = "TEMForge"
APP_SUBTITLE = "Interactive TEM/STEM Simulation and Design Workbench"
WINDOW_TITLE = f"{APP_NAME} — {APP_SUBTITLE}"
ORGANIZATION_NAME = APP_NAME
DISTRIBUTION_NAME = "temforge"
RECORDER_NAME = f"{APP_NAME} Instrument Recorder"

# Compatibility identifiers are never used as product labels. Keep previous
# user settings and large caches intact instead of replacing or moving them.
_LEGACY_ORGANIZATION = "TEM Simulator"
_LEGACY_APPLICATION = "TEM Simulator v2"
_MIGRATION_KEY = "branding/preferences_imported_v1"


def _migrate_settings(destination, legacy) -> None:
    """Copy missing preferences once; current choices and original data win."""
    if destination.value(_MIGRATION_KEY, False, type=bool):
        return
    legacy.setFallbacksEnabled(False)
    for key in legacy.allKeys():
        if key != _MIGRATION_KEY and not destination.contains(key):
            destination.setValue(key, legacy.value(key))
    destination.sync()
    if destination.status().value != 0:
        warnings.warn("TEMForge preferences could not be imported; the original settings are retained.",
                      RuntimeWarning, stacklevel=2)
        return
    destination.setValue(_MIGRATION_KEY, True)
    destination.sync()


def application_settings():
    """Return the primary preferences, retaining saved layouts and cache limits."""
    from PySide6.QtCore import QSettings

    destination = QSettings(ORGANIZATION_NAME, APP_NAME)
    _migrate_settings(destination, QSettings(_LEGACY_ORGANIZATION, _LEGACY_APPLICATION))
    return destination


def recorder_settings():
    """Keep recorder output paths and geometry separate from workbench settings."""
    from PySide6.QtCore import QSettings

    destination = QSettings(ORGANIZATION_NAME, RECORDER_NAME)
    _migrate_settings(destination, QSettings(_LEGACY_ORGANIZATION, "Instrument Recorder"))
    return destination


def application_cache_root(directory: str, *, reuse_legacy: bool = False) -> Path:
    """Use TEMForge storage; an existing legacy result store may stay in place.

    Reusing a cache never copies, renames or deletes saved scientific results.
    If both exist, the current branded store takes precedence.
    """
    local = str(os.environ.get("LOCALAPPDATA", "")).strip()
    base = Path(local) if local else Path.home() / ".cache"
    current = base / APP_NAME / directory
    if reuse_legacy and not current.exists():
        previous = base / _LEGACY_APPLICATION / directory
        if previous.is_dir():
            return previous
    return current
