"""Content and upstream-physics binding for saved Descan study prefixes.

This is a study protocol, not a new simulator cache.  Its caller supplies the
actual centre-frozen state: no calibration or scan refit is performed here.
Historical reconstruction must be identified and independently replayed; a
checksum calculated later is never represented as a contemporaneous capture.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SCHEMA = "descan-prefix-provenance-v1"
NUMERIC_KEYS = ("ray_id", "x_m", "tx_rad", "y_m", "ty_rad", "weights",
                "energy_offset_ev", "alive", "prefix_z_mm")
CAPTURE_KINDS = {"contemporaneous_prefix_inputs",
                 "reconstructed_prefix_inputs_verified_by_replay"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _read_prefix(source):
    if isinstance(source, (str, Path)):
        try:
            with np.load(source, allow_pickle=False) as value:
                return {key: np.array(value[key], copy=True) for key in
                        (*NUMERIC_KEYS, "exit_bundle_sha256")}
        except (KeyError, TypeError) as error:
            raise ValueError("Prefix is missing required numeric/source arrays") from error
    if not isinstance(source, Mapping) and not hasattr(source, "files"):
        raise ValueError("Prefix must be an NPZ path or array mapping")
    try:
        return {key: np.array(source[key], copy=True) for key in
                (*NUMERIC_KEYS, "exit_bundle_sha256")}
    except (KeyError, TypeError) as error:
        raise ValueError("Prefix is missing required numeric/source arrays") from error


def _validate_prefix(value):
    ids = value["ray_id"]
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu" or not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("Prefix ray IDs must retain exact original index 0..N-1")
    for key in NUMERIC_KEYS[1:]:
        array = value[key]
        expected = () if key == "prefix_z_mm" else ids.shape
        if array.shape != expected or array.dtype.hasobject or array.dtype.fields is not None:
            raise ValueError(f"Prefix {key} has an invalid shape or dtype")
        if key == "alive":
            if array.dtype.kind != "b":
                raise ValueError("Prefix alive must be boolean")
        elif array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"Prefix {key} must be finite real numeric data")
    if np.any(value["weights"] < 0.) or not np.any(value["weights"] > 0.):
        raise ValueError("Prefix weights must be nonnegative with positive total current")
    identity = value["exit_bundle_sha256"]
    if identity.shape != () or identity.dtype.kind not in "US" or not re.fullmatch("[0-9a-f]{64}", str(identity.item())):
        raise ValueError("Prefix requires the original source exit SHA256")


def prefix_numeric_digest(path_or_mapping):
    """Hash validated names, exact dtypes, shapes and bytes, including source ID."""
    value = _read_prefix(path_or_mapping)
    _validate_prefix(value)
    digest = hashlib.sha256(SCHEMA.encode("ascii"))
    for key in (*NUMERIC_KEYS, "exit_bundle_sha256"):
        array = np.ascontiguousarray(value[key])
        header = json.dumps([key, value[key].dtype.str, list(value[key].shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def _graph(value):
    from temsim.immutable_json import thaw_json
    from temsim.instrument_snapshot import encode_instrument
    return thaw_json(encode_instrument(value))


def physical_upstream_signature(state, prefix_z_mm, *, checkpoint_z_mm=()):
    """Bind actual finite field samples and clipping before the frozen prefix.

    Backend and plotting history are normalized because a tail CPU replay may
    legitimately reuse a GPU prefix.  All physical controls remain unchanged.
    Unknown model types and unidentified electric providers fail closed. Posed
    clipping objects are conservatively included even when nominally downstream.
    """
    from temsim.calculation_manifest import solver_source_identity
    from temsim.instrument_snapshot import decode_instrument, encode_instrument
    from temsim.physics.aperture_clipping import posed_aperture_registration
    from temsim.physics.column_wall import _vacuum_segments, _partition_vacuum_segments_uncached, _expanded_profile_axis
    from temsim.physics.core import build_propagation_plan
    from temsim.physics.instrument_magnetic import column_dipole_fields

    if isinstance(prefix_z_mm, (bool, np.bool_)) or not np.isfinite(prefix_z_mm):
        raise ValueError("Prefix Z must be finite and real")
    frozen = decode_instrument(encode_instrument(state))
    start, stop = float(frozen.electron_gun.exit_plane_z_mm), float(prefix_z_mm)
    if stop <= start:
        raise ValueError("Prefix must lie downstream of the gun exit")
    checkpoints = np.asarray(tuple(checkpoint_z_mm), dtype=float)
    if checkpoints.ndim != 1 or not np.isfinite(checkpoints).all() or np.any(checkpoints < start) or np.any(checkpoints > stop):
        raise ValueError("Prefix checkpoint planes must lie inside the upstream domain")
    checkpoints = np.unique(np.r_[checkpoints, stop])
    # These are comparison settings only; never modify the caller's run state.
    frozen.step_mm = .05
    frozen.history_step_mm = .05
    frozen.acceleration_enabled = True
    frozen.acceleration_backend = "Numba CPU"
    events = tuple((field.event_z_mm, field.event_dx_rad, field.event_dy_rad)
                   for field in column_dipole_fields(frozen)
                   if field.field_support_mm[0] <= stop and field.field_support_mm[1] >= start)
    plan = build_propagation_plan(frozen, start, stop, events,
        include_spherical_aberration=True, include_hexapole=True,
        include_intrinsic_aberrations=True, checkpoint_z_mm=checkpoints,
        save_z_mm=tuple(checkpoints)+tuple(float(a.z_mm) for a in frozen.apertures if start < float(a.z_mm) <= stop),
        maximum_step_mm=.05)
    if plan.electric_field is not None and plan.electric_field_identity is None:
        raise ValueError("Upstream electric field has no stable numerical identity")
    apertures = list(frozen.apertures)
    nanopulser = getattr(frozen, "nanopulser", None)
    if nanopulser is not None and nanopulser.installed:
        apertures.append(nanopulser.aperture)
    clipping = []
    parts = {part.key: part for part in getattr(getattr(frozen, "_resolved_assembly", None), "parts", ())}
    for aperture in apertures:
        registration = posed_aperture_registration(frozen, aperture)
        if registration is not None or start <= float(aperture.z_mm) <= stop:
            row = dict(component=_graph(aperture), registration=None)
            if registration is not None:
                row["registration"] = dict(origin_global_m=registration.origin_global_m,
                    rotation_local_to_global=registration.rotation_local_to_global)
                part = parts.get(aperture.key)
                row["finite_plate_geometry"] = None if part is None else _graph(part.data)
            clipping.append(row)
    axis = np.array([start, stop])
    stationary, placed = _partition_vacuum_segments_uncached(frozen, _vacuum_segments(frozen, axis))
    wall_axis, interval_radius, node_radius = _expanded_profile_axis(axis, stationary)
    # Explicit strings preserve the valid absence of a wall (infinite radius).
    wall = dict(axis_mm=wall_axis.tolist(), interval_radius_mm=[float(v) if np.isfinite(v) else "no_wall" for v in interval_radius],
                node_radius_mm=[float(v) if np.isfinite(v) else "no_wall" for v in node_radius], placed=[])
    for segment, registration in placed:
        wall["placed"].append(dict(start_z_mm=float(segment.start_z_mm), end_z_mm=float(segment.end_z_mm),
            inner_diameter_mm=float(segment.inner_diameter_mm), origin_global_m=registration.origin_global_m,
            rotation_local_to_global=registration.rotation_local_to_global))
    payload = dict(schema=SCHEMA, domain_mm=[start, stop], checkpoint_z_mm=checkpoints.tolist(),
        comparison_step_mm=.05, comparison_backend="Numba CPU", implementation=solver_source_identity(),
        plan_sha256=plan.signature, gun=_graph(frozen.electron_gun), apertures=clipping, wall=wall)
    return dict(sha256=_digest(payload), payload=payload)


def create_prefix_provenance(state, prefix_npz, *, capture_kind,
                             source_snapshot_digest, checkpoint_z_mm=(), evidence=None):
    """Create a content seal; legacy reconstruction additionally requires replay."""
    if capture_kind not in CAPTURE_KINDS:
        raise ValueError("Unknown prefix input capture kind")
    if not isinstance(source_snapshot_digest, str) or not re.fullmatch("[0-9a-f]{64}", source_snapshot_digest):
        raise ValueError("A verified source snapshot digest is required")
    if capture_kind == "reconstructed_prefix_inputs_verified_by_replay":
        if not isinstance(evidence, Mapping) or evidence.get("independent_saved_exit_replay_passed") is not True:
            raise ValueError("Historical reconstructed prefix requires independent saved-exit replay evidence")
    value = _read_prefix(prefix_npz)
    numeric_digest = prefix_numeric_digest(value)
    receipt = dict(schema=SCHEMA, capture_kind=capture_kind,
        source_snapshot_digest=source_snapshot_digest,
        prefix_numeric_sha256=numeric_digest, exit_bundle_sha256=str(value["exit_bundle_sha256"].item()),
        prefix_z_mm=float(value["prefix_z_mm"]),
        upstream=physical_upstream_signature(state, float(value["prefix_z_mm"]), checkpoint_z_mm=checkpoint_z_mm),
        evidence=dict(evidence or {}))
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def verify_prefix_provenance(receipt, state, prefix_npz):
    """Reject altered arrays or any changed upstream field/clipping dependency."""
    value = dict(receipt)
    claimed = value.pop("receipt_sha256", None)
    if value.get("schema") != SCHEMA or _digest(value) != claimed:
        raise ValueError("Prefix provenance receipt checksum mismatch")
    if value.get("capture_kind") not in CAPTURE_KINDS:
        raise ValueError("Unknown prefix input capture kind")
    if value["capture_kind"] == "reconstructed_prefix_inputs_verified_by_replay" and value.get("evidence", {}).get("independent_saved_exit_replay_passed") is not True:
        raise ValueError("Reconstructed prefix lacks independent replay evidence")
    if prefix_numeric_digest(prefix_npz) != value["prefix_numeric_sha256"]:
        raise ValueError("Saved prefix numeric arrays changed")
    current = physical_upstream_signature(state, value["prefix_z_mm"],
        checkpoint_z_mm=value["upstream"]["payload"]["checkpoint_z_mm"])
    if current["sha256"] != value["upstream"]["sha256"]:
        raise ValueError("Upstream propagation or physical clipping changed since prefix execution")


def reconstruct_neutral_prefix_snapshot(inputs_path, report):
    """Restore recorded pre-freeze inputs and apply only documented centre drives.

    The returned metadata is a reconstruction claim, not execution proof; the
    caller must compare a new saved-exit propagation with the historic prefix.
    """
    from temsim.instrument_snapshot import InstrumentSnapshot
    snapshot = InstrumentSnapshot.from_dict(json.loads(Path(inputs_path).read_text(encoding="utf-8")))
    if report.get("input_identity") != snapshot.digest:
        raise ValueError("Report does not bind the supplied pre-freeze snapshot")
    state = snapshot.restore()
    adjustments = {}
    for key, component in (("ac", state.ac_deflector), ("descan", state.descan_deflector)):
        drive = np.asarray(report["static_bias_mrad"][key], dtype=float)
        if drive.shape != (2,) or not np.isfinite(drive).all():
            raise ValueError("Recorded static centre drives must be finite 2-vectors")
        component.scan_enabled = False
        component.kick_x_mrad, component.kick_y_mrad = map(float, drive)
        adjustments[key] = dict(scan_enabled=False, kick_x_mrad=float(drive[0]), kick_y_mrad=float(drive[1]))
    state.ac_deflector.wobble_enabled = False
    adjustments["ac"]["wobble_enabled"] = False
    return state, dict(origin="reconstructed_from_pre_freeze_inputs_and_recorded_static_drives",
                       source_snapshot_digest=snapshot.digest, adjustments=adjustments,
                       contemporaneous_capture=False)
