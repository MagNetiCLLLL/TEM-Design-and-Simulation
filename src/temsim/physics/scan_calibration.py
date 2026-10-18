"""Persistent drive calibration, not an independent downstream beam state."""
from __future__ import annotations

import json
import numpy as np
from temsim.optics.shared_deflectors import shared_channel_enabled

SCAN_CALIBRATION_MODEL = "finite-coil-canonical-descan-phase-v3"
AUTOMATIC_AC_ANGLE_TARGET = "canonical"
AC_ANGLE_TARGETS = frozenset({"canonical", "mechanical", "legacy_unknown"})


def ac_angle_target(state):
    """Describe the accepted angle constraint, without relabelling old drives."""
    if held(state):
        return validate_record(state.ac_deflector.calibration_record_json).get(
            "ac_angle_target", "legacy_unknown")
    return AUTOMATIC_AC_ANGLE_TARGET


def held(state):
    mode = state.ac_deflector.calibration_mode
    if mode not in ("automatic", "held"):
        raise ValueError("Scan calibration mode must be automatic or held")
    return mode == "held"


def sample_reference_z_mm(state):
    reference = state.ac_deflector.scan_reference
    if reference == "sample_centre":
        return float(state.sample.z_mm)
    if reference == "sample_entrance":
        return float(state.sample.upper_surface_z_mm)
    raise ValueError("Unknown scan reference; choose specimen centre or entrance")


def validate_record(text):
    if not text:
        raise ValueError("No held scan calibration. Use Calibrate and hold first.")
    record = json.loads(text)
    if (not isinstance(record, dict) or type(record.get("version")) is not int
            or record["version"] != 1):
        raise ValueError("Unsupported scan calibration record")
    for key in ("ac_ratio", "descan_ratio", "command_mrad"):
        values = np.asarray(record.get(key), dtype=float)
        if values.shape != (2, 2) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid held scan matrix: {key}")
    fov = np.asarray(record.get("fov_nm"), dtype=float)
    if fov.shape != (2,) or not np.all(np.isfinite(fov)) or np.any(fov <= 0):
        raise ValueError("Held scan FOV must have two finite positive extents")
    if not np.isfinite(float(record.get("target_z_mm", float("nan")))):
        raise ValueError("Held descan target must be finite")
    if not isinstance(record.get("descan_calibrated"), bool):
        raise ValueError("Held scan record must declare descan calibration status")
    target_key = record.get("target_key")
    if not isinstance(target_key, str):
        raise ValueError("Held scan record must declare its physical descan target key")
    if record["descan_calibrated"] and (
        not target_key.strip() or target_key == "legacy_image_reference"
    ):
        raise ValueError("Held descan calibration requires an explicit physical target key")
    if record.get("reference") not in ("sample_centre", "sample_entrance"):
        raise ValueError("Invalid held scan reference")
    host_key = record.get("descan_physical_host_key")
    if "descan_physical_host_key" in record and (not isinstance(host_key, str) or not host_key.strip()):
        raise ValueError("Held descan calibration must name its physical deflector host")
    if record.get("ac_angle_target", "legacy_unknown") not in AC_ANGLE_TARGETS:
        raise ValueError("Invalid held AC angle target")
    constraint = record.get("descan_constraint", "position")
    if constraint not in ("position", "position_and_angle"):
        raise ValueError("Invalid held Descan constraint")
    if "descan_command_mrad" in record:
        values = np.asarray(record["descan_command_mrad"], dtype=float)
        if values.shape != (2, 2) or not np.all(np.isfinite(values)):
            raise ValueError("Invalid held Descan command matrix")
    elif constraint == "position_and_angle":
        raise ValueError("Position-and-angle calibration requires its explicit Descan command")
    return record


def capture_record(state, *, descan_calibrated: bool, ac_angle_target=None):
    """Capture only the calibration stages executed in the accepted solve."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    ac, descan = state.ac_deflector, state.descan_deflector
    if ac_angle_target is None:
        # Descan-only capture preserves AC provenance. Manually supplied
        # drives without explicit provenance must not be called canonical.
        ac_angle_target = (validate_record(ac.calibration_record_json).get(
            "ac_angle_target", "legacy_unknown") if held(state) else
            getattr(ac, "_pure_shift_angle_target", "legacy_unknown"))
    if ac_angle_target not in AC_ANGLE_TARGETS:
        raise ValueError("Invalid held AC angle target")
    record = dict(version=1, calibrated_snapshot_id=capture_instrument_snapshot(state).digest,
                  ac_angle_target=ac_angle_target,
                  ac_ratio=ac.pure_shift_lower_ratio_matrix,
                  descan_ratio=descan.image_plane_lower_ratio_matrix,
                  descan_constraint=getattr(descan, "_calibrated_constraint", "position"),
                  descan_command_mrad=descan.scan_command_matrix_mrad,
                  command_mrad=ac.scan_command_matrix_mrad,
                  fov_nm=(ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm),
                  target_key=descan.image_plane_target_key,
                  target_z_mm=descan.image_plane_target_z_mm or float(state.sample.z_mm),
                  descan_calibrated=descan_calibrated,
                  descan_physical_host_key=str(getattr(descan, "_physical_host_key", descan.key)),
                  reference=ac.scan_reference)
    encoded = json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))
    validate_record(encoded)
    ac.calibration_record_json = encoded
    return encoded


def validate_descan_shared_peaks(state):
    """Check every raster corner after summing the shared host alignment.

    The affine square-raster bound is exact per physical coil axis. It uses
    the accepted matrices, so held FOV changes and mixed X/Y coupling are
    checked without refitting or changing static alignment.
    """
    descan = state.descan_deflector
    host = getattr(descan, "_physical_host", None)
    if host is None or not shared_channel_enabled(descan):
        return
    host_static = np.array(((host.upper_x_mrad, host.upper_y_mrad),
                            (host.lower_x_mrad, host.lower_y_mrad)), dtype=float)
    bias = np.array((descan.kick_x_mrad, descan.kick_y_mrad), dtype=float)
    command = (np.asarray(descan.scan_command_matrix_mrad, dtype=float)
               if descan.scan_enabled else np.zeros((2, 2)))
    peaks = np.array([np.abs(static + np.asarray(mapping) @ bias)
                      + np.sum(np.abs(np.asarray(mapping) @ command), axis=1)
                      for static, mapping in zip(host_static, descan.coil_kick_matrices())])
    if not np.isfinite(peaks).all() or np.any(peaks > float(host.maximum_kick_mrad)):
        raise ValueError("Descan raster plus Image/Diffraction alignment exceeds the shared physical coil limit")


def restore_held(state, *, restore_descan=True):
    """Restore fixed drive ratios, scaling only the requested raster extent.

    The explicit Descan-only recalibration operation first restores just AC.
    That path may replace an absent or obsolete downstream calibration without
    requiring it to be executable. Normal held execution still restores both.
    """
    ac, descan = state.ac_deflector, state.descan_deflector
    record = validate_record(ac.calibration_record_json)
    if record["reference"] != ac.scan_reference:
        raise ValueError("Scan reference changed. Calibrate and hold at the new specimen plane.")
    if (restore_descan and shared_channel_enabled(descan) and descan.scan_enabled
            and not record["descan_calibrated"]):
        raise ValueError("Descan was not calibrated. Enable it and use Calibrate Descan (keep AC).")
    host = getattr(descan, "_physical_host", None)
    if (restore_descan and host is not None and shared_channel_enabled(descan) and descan.scan_enabled
            and record.get("descan_physical_host_key") != str(host.key)):
        raise ValueError("The held descan calibration belongs to an older or different physical deflector. Use Calibrate Descan (keep AC).")
    fov = np.array((ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm))
    command = np.asarray(record["command_mrad"]) * (fov / record["fov_nm"])[None, :]
    snapshots = [dict(c.__dict__) for c in (ac, descan)]
    try:
        ac.set_pure_shift_coupling(record["ac_ratio"])
        object.__setattr__(ac, "_pure_shift_angle_target", record.get(
            "ac_angle_target", "legacy_unknown"))
        ac.set_scan_command_matrix_mrad(command)
        ac.validate()
        if restore_descan:
            descan.set_image_plane_coupling(record["descan_ratio"],
                target_key=record["target_key"], target_z_mm=record["target_z_mm"])
            # Older records contain only the opposite-AC command. New records
            # retain independent Descan drives and scale the same raster axes.
            descan_command = np.asarray(record.get(
                "descan_command_mrad", -np.asarray(record["command_mrad"])))
            descan.set_scan_command_matrix_mrad(
                descan_command * (fov / record["fov_nm"])[None, :])
            object.__setattr__(descan, "_calibrated_constraint", record.get(
                "descan_constraint", "position"))
            descan.validate()
            validate_descan_shared_peaks(state)
    except Exception:
        for component, saved in zip((ac, descan), snapshots):
            component.__dict__.clear()
            component.__dict__.update(saved)
        raise
    return command
