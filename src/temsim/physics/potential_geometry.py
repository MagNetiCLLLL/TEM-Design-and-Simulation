"""Frozen finite absorbing boundaries for the opt-in potential-wave route.

Only the authoritative vacuum-bore inventory and declared circular aperture
plates are represented. This is a global-Z absorbing-boundary discretization,
not an atomic wall model, material diffraction solver or inferred CAD solid.
Excitation does not determine the presence of an installed mechanical bore.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_wave_hardware import PosedWaveBore, placed_wave_bores
from temsim.physics.posed_wave_aperture import PosedWaveAperture
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError, WaveGridBudgetError, apply_resolved_operator


_SCHEMA = "potential-absorbing-geometry-v1"
_SCOPE = "finite opaque slabs outside declared circular openings; swept global-Z absorbing boundary; no wall atomistics or inferred CAD solid"


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite"+(" and positive" if positive else ""))
    return value


def _key(value):
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("Geometry identities must be nonempty trimmed strings")
    return value


def _piece_dict(piece):
    return {"kind": "aperture" if type(piece) is PosedWaveAperture else "bore", "key": piece.key,
        "local_start_z_mm": piece.local_start_z_mm, "local_end_z_mm": piece.local_end_z_mm,
        "radius_mm": piece.radius_mm, "origin_global_m": piece.registration.origin_global_m,
        "rotation_local_to_global": piece.registration.rotation_local_to_global}


def _piece_from_dict(record):
    expected = {"kind", "key", "local_start_z_mm", "local_end_z_mm", "radius_mm",
                "origin_global_m", "rotation_local_to_global"}
    if not isinstance(record, Mapping) or set(record) != expected or record["kind"] not in {"bore", "aperture"}:
        raise ValueError("Unsupported absorbing geometry primitive")
    for key in ("local_start_z_mm", "local_end_z_mm", "radius_mm"):
        _number(record[key], key)
    registration = CoordinateRegistration(record["origin_global_m"], record["rotation_local_to_global"])
    cls = PosedWaveAperture if record["kind"] == "aperture" else PosedWaveBore
    return cls(_key(record["key"]), float(record["local_start_z_mm"]), float(record["local_end_z_mm"]),
               float(record["radius_mm"]), registration)


@dataclass(frozen=True, slots=True)
class PotentialBoundaryHost:
    """One declared material host, possibly with several bore intervals."""
    host_key: str
    source_keys: tuple
    pieces: tuple

    def __post_init__(self):
        _key(self.host_key)
        keys = tuple(_key(key) for key in self.source_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Boundary source identities must be unique")
        pieces, identities = [], set()
        for piece in self.pieces:
            if type(piece) not in (PosedWaveBore, PosedWaveAperture) or type(piece.registration) is not CoordinateRegistration:
                raise ValueError("Boundary hosts require exact frozen supported primitives")
            record = _piece_dict(piece)
            identity = json_digest({key: value for key, value in record.items() if key != "key"})
            if identity in identities:
                raise ValueError("Duplicate material geometry within one physical host")
            identities.add(identity)
            pieces.append(_piece_from_dict(record))
        if not pieces:
            raise ValueError("A boundary host must contain actual material geometry")
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "pieces", tuple(pieces))

    def to_dict(self):
        return {"host_key": self.host_key, "source_keys": list(self.source_keys),
                "pieces": [_piece_dict(piece) for piece in self.pieces]}


@dataclass(frozen=True, slots=True)
class PotentialGeometrySnapshot:
    start_z_m: float
    stop_z_m: float
    boundaries: tuple
    aperture_inventory: tuple = ()
    source_scene_geometry_digest: str | None = None
    digest: str = field(init=False)

    def __post_init__(self):
        start, stop = _number(self.start_z_m, "Geometry start"), _number(self.stop_z_m, "Geometry stop")
        if stop <= start:
            raise ValueError("Geometry capture must have a positive axial extent")
        authority = self.source_scene_geometry_digest
        if authority is not None and (type(authority) is not str or len(authority) != 64
                or any(value not in "0123456789abcdef" for value in authority)):
            raise ValueError("Invalid source-scene geometry identity")
        hosts = tuple(self.boundaries)
        if any(type(host) is not PotentialBoundaryHost for host in hosts):
            raise ValueError("Geometry requires exact frozen boundary hosts")
        if len({host.host_key for host in hosts}) != len(hosts):
            raise ValueError("Shared physical hosts must occur only once")
        inventory = tuple(freeze_json(thaw_json(row)) for row in self.aperture_inventory)
        for row in inventory:
            if set(row) != {"key", "installed", "enabled", "inserted"}:
                raise ValueError("Invalid captured aperture selection")
            _key(row["key"])
            if any(type(row[key]) is not bool for key in ("installed", "enabled", "inserted")):
                raise ValueError("Aperture selections must be booleans")
        if len({row["key"] for row in inventory}) != len(inventory):
            raise ValueError("Aperture selections must have unique identities")
        object.__setattr__(self, "start_z_m", start)
        object.__setattr__(self, "stop_z_m", stop)
        object.__setattr__(self, "boundaries", hosts)
        object.__setattr__(self, "aperture_inventory", inventory)
        object.__setattr__(self, "digest", json_digest(self._payload()))

    def _payload(self):
        return {"schema": _SCHEMA, "start_z_m": self.start_z_m, "stop_z_m": self.stop_z_m,
            "boundaries": [host.to_dict() for host in self.boundaries],
            "aperture_inventory": [thaw_json(row) for row in self.aperture_inventory],
            "source_scene_geometry_digest": self.source_scene_geometry_digest, "scope": _SCOPE}

    def to_dict(self):
        return thaw_json(freeze_json({**self._payload(), "digest": self.digest}))

    @classmethod
    def from_dict(cls, record):
        if (not isinstance(record, Mapping) or set(record) != {"schema", "start_z_m", "stop_z_m", "boundaries",
                "aperture_inventory", "source_scene_geometry_digest", "scope", "digest"} or record["schema"] != _SCHEMA or record["scope"] != _SCOPE):
            raise ValueError("Invalid absorbing geometry schema")
        hosts = []
        for row in record["boundaries"]:
            if set(row) != {"host_key", "source_keys", "pieces"}:
                raise ValueError("Invalid boundary-host schema")
            hosts.append(PotentialBoundaryHost(row["host_key"], tuple(row["source_keys"]),
                         tuple(_piece_from_dict(piece) for piece in row["pieces"])))
        result = cls(record["start_z_m"], record["stop_z_m"], tuple(hosts), tuple(record["aperture_inventory"]), record["source_scene_geometry_digest"])
        if result.digest != record["digest"]:
            raise ValueError("Absorbing geometry digest mismatch")
        return result

    @property
    def pieces(self):
        return tuple(piece for host in self.boundaries for piece in host.pieces)


def capture_potential_geometry(state, start_z_m, stop_z_m):
    """Capture authoritative mechanics, not excitation, CAD or inferred plates.

    The captured interval is fixed for later restart. All placed constraints
    are retained: an off-axis wave may meet a tilted slab beyond its opening's
    nominal axial extrema. Unknown material outside this inventory is not added.
    """
    from temsim.physics.physical_scene import capture_physical_scene
    from temsim.physics.aperture_clipping import posed_aperture_registration
    start, stop = _number(start_z_m, "Geometry start"), _number(stop_z_m, "Geometry stop")
    if stop <= start:
        raise ValueError("Geometry capture must have a positive axial extent")
    strategy = getattr(getattr(state, "sample", None), "wave_objective_aperture_strategy", "physical_plane")
    if strategy != "physical_plane":
        raise ValueError("Potential geometry requires objective aperture strategy physical_plane; an equivalent pupil would double-count or relocate material")
    scene = capture_physical_scene(state)
    stationary, placed = placed_wave_bores(state, start*1000, stop*1000)
    groups, inventory = {}, []
    def add(piece):
        owner = scene.owner_for(piece.key)
        if piece.key in scene.payload["ownership"] and owner is None:
            raise ValueError(f"Non-material control channel cannot supply a physical bore: {piece.key}")
        owner = piece.key if owner is None else owner
        group = groups.setdefault(owner, {"source_keys": [], "pieces": {}, "host_key": owner})
        if piece.key not in group["source_keys"]:
            group["source_keys"].append(piece.key)
        content = _piece_dict(piece)
        identity = json_digest({key: value for key, value in content.items() if key != "key"})
        group["pieces"].setdefault(identity, piece)
    for segment in stationary:
        add(PosedWaveBore.from_segment(segment, CoordinateRegistration()))
    for piece in placed:
        add(piece)
    for aperture in getattr(state, "apertures", ()):
        row = {"key": _key(aperture.key)}
        for key in ("installed", "enabled", "inserted"):
            value = getattr(aperture, key, True)
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError("Aperture installation selections must be booleans")
            row[key] = bool(value)
        inventory.append(row)
        if all(row[key] for key in ("installed", "enabled", "inserted")):
            registration = posed_aperture_registration(state, aperture) or CoordinateRegistration()
            add(PosedWaveAperture.from_component(state, aperture, registration))
    hosts = tuple(PotentialBoundaryHost(row["host_key"], tuple(row["source_keys"]), tuple(row["pieces"].values()))
                  for row in groups.values())
    return PotentialGeometrySnapshot(start, stop, hosts, tuple(inventory), scene.geometry_digest)


def apply_potential_geometry(wave, geometry, start_z_m, stop_z_m, *, grid_numerics=None,
                             minimum_radius_cells=16., cancelled=lambda: False):
    """Apply resolved swept material masks without restoring lost probability.

    Incident complex fields are Fourier-refined BEFORE a small opening is
    masked. The 16-cell default is an admission guard, not edge convergence.
    Direct masked sums retain losses far below total-probability roundoff.
    """
    if type(wave) is not PlaneWave or not isinstance(wave.amplitude, np.ndarray):
        raise ValueError("Potential mechanical boundaries require a CPU PlaneWave")
    if type(geometry) is not PotentialGeometrySnapshot:
        raise ValueError("A frozen potential geometry snapshot is required")
    start, stop = _number(start_z_m, "Boundary start"), _number(stop_z_m, "Boundary stop")
    if not geometry.start_z_m <= start <= stop <= geometry.stop_z_m:
        raise ValueError("Boundary interval lies outside the captured geometry")
    cells = _number(minimum_radius_cells, "Minimum radius cells", positive=True)
    if cells < 16:
        raise ValueError("Finite boundaries require at least 16 cells in opening radius")
    numerics = WaveGridNumerics() if grid_numerics is None else grid_numerics
    if type(numerics) is not WaveGridNumerics:
        raise ValueError("Boundary sampling requires explicit WaveGridNumerics")
    def mask(candidate):
        for piece in geometry.pieces:
            if piece.clears_grid(candidate.amplitude.shape, candidate.basis_m, candidate.origin_m,
                                 stop*1000, previous_z_mm=start*1000):
                continue
            if piece.radius_mm > 0:
                resolved = piece.radius_mm*.001/np.linalg.norm(candidate.basis_m, ord=2)
                if resolved < cells:
                    raise WaveSamplingError(f"Finite boundary {piece.key} requires resolved incident sampling before absorption", cells/resolved)
        current, records = candidate, []
        xy = candidate.coordinates_m()*1000
        for host in geometry.boundaries:
            transmission = None
            for piece in host.pieces:
                if piece.clears_grid(current.amplitude.shape, current.basis_m, current.origin_m,
                                     stop*1000, previous_z_mm=start*1000):
                    continue
                passed = piece.swept_transmission_mask(xy[0], xy[1], start*1000, stop*1000)
                transmission = passed if transmission is None else transmission & passed
            if transmission is None or np.all(transmission):
                continue
            before = current.probability
            lost = float(np.sum(abs(current.amplitude[~transmission])**2))
            current = replace(current, amplitude=np.where(transmission, current.amplitude, 0.))
            records.append({"physical_host_key": host.host_key, "source_keys": list(host.source_keys),
                "start_z_m": start, "stop_z_m": stop, "input_probability": before,
                "output_probability": current.probability, "lost_probability": lost,
                "physical_absorption": True, "normalization_applied": False})
        return current, records
    (result, records), refinements = apply_resolved_operator(wave, mask, numerics=numerics, cancelled=cancelled,
        operator_working_bytes=lambda shape: 128*math.prod(shape))
    return result, {"geometry_digest": geometry.digest, "physical_losses": records,
        "physical_loss_probability": math.fsum(row["lost_probability"] for row in records),
        "sampling_refinements": refinements, "minimum_radius_cells": cells,
        "scope": _SCOPE, "loss_attribution": "first opaque host in captured order; intersections are not added twice"}


def _geometry_step_limit(checkpoint, geometry, lower, upper, pixel_fraction, slab_steps, *, include_zero=False):
    limit = upper-lower
    for mode in checkpoint.beam.modes:
        if not include_zero and (mode.weight_per_reference_electron == 0 or mode.plane.probability == 0):
            continue
        wave = mode.plane
        for piece in geometry.pieces:
            if piece.clears_grid(wave.amplitude.shape, wave.basis_m, wave.origin_m,
                                 upper*1000, previous_z_mm=lower*1000):
                continue
            guide = piece.maximum_step_mm(wave.basis_m, pixel_fraction=pixel_fraction)*.001
            if guide <= 0:
                raise ValueError("A grazing boundary needs a different marching chart; a zero geometry step is not admitted")
            axial = abs(piece.registration.rotation_array[2, 2])
            if axial > 1e-14:
                guide = min(guide, (piece.local_end_z_mm-piece.local_start_z_mm)*.001/axial/slab_steps)
            limit = min(limit, guide)
    return limit


def _bounded_upper(lower, upper, distance_limit):
    """Choose an inward representable plane satisfying the exact step bound.

    Adding a small positive guide to a large absolute Z can round outward.
    Repeating the same rounded sum cannot repair that violation. Move one
    representable plane inward instead; never enlarge a guide by an epsilon
    or merge a nearby requested observation with an existing plane.
    """
    candidate = min(upper, lower+distance_limit)
    if candidate-lower > distance_limit:
        candidate = float(np.nextafter(candidate, lower))
    if candidate <= lower or candidate-lower > distance_limit:
        raise ValueError("Geometry step cannot be represented at the current axial scale")
    return candidate


def _mask_checkpoint(checkpoint, geometry, lower, upper, *, numerics, cells, cancelled):
    from temsim.physics.wave_flux import BeamState
    modes, records = [], []
    refinements = 0
    for mode in checkpoint.beam.modes:
        if mode.weight_per_reference_electron == 0 or mode.plane.probability == 0:
            modes.append(mode)
            continue
        plane, receipt = apply_potential_geometry(mode.plane, geometry, lower, upper,
            grid_numerics=numerics, minimum_radius_cells=cells, cancelled=cancelled)
        refinements += len(receipt["sampling_refinements"])
        norm = plane.probability
        # Existing WaveMode stores unit shape plus absolute reference weight.
        # This factorization does not restore any physically absorbed current.
        unit = replace(plane, amplitude=plane.amplitude/math.sqrt(norm)) if norm else plane
        modes.append(replace(mode, plane=unit,
            weight_per_reference_electron=mode.weight_per_reference_electron*norm))
        for row in receipt["physical_losses"]:
            if row["lost_probability"] > 0:
                records.append({**row, "mode_id": mode.mode_id,
                    "lost_probability_per_reference": mode.weight_per_reference_electron*row["lost_probability"],
                    "input_weight_per_reference": mode.weight_per_reference_electron*row["input_probability"],
                    "output_weight_per_reference": mode.weight_per_reference_electron*row["output_probability"]})
    return replace(checkpoint, beam=BeamState(tuple(modes), checkpoint.beam.reference_plane)), records, refinements


def observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, observation_z_m, *,
        maximum_step_m=.0001, geometry_pixel_fraction=.5, minimum_radius_cells=16.,
        minimum_slab_steps=4, maximum_geometry_steps=100000, grid_numerics=None,
        cancelled=lambda: False, progress_callback=None, **wave_options):
    """Continue a frozen potential checkpoint through finite material boundaries.

    Each accepted interval propagates the complete field, then applies resolved
    swept slab masks. Smaller axial steps converge this absorbing-boundary
    discretization; it is not exact continuous wall scattering. Geometry is
    checked again after optical grid changes. A retry starts from the unchanged
    checkpoint, so discarded field/support attempts never consume saved budget.

    Use one captured full interval for all slice/downstream continuations. Keep
    record['potential_geometry'] through external sample wrappers. Changed
    geometry or boundary numerics requires restart from the original entrance.
    Thin aperture options are rejected because this geometry owns the plates.
    """
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
    from temsim.physics.potential_wave_observation import observe_potential_mode
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.potential_material_phase_budget import (
        validate_material_kinetic_record, exact_material_kinetic_plane,
    )
    from temsim.physics.potential_support_lineage import validate_inherited_support
    from temsim.physics.wave_flux import BeamState
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    if type(snapshot) not in (ElectromagneticPotentialSnapshot, DynamicPotentialSnapshot) or type(geometry) is not PotentialGeometrySnapshot:
        raise ValueError("Potential propagation requires frozen field and geometry snapshots")
    if (geometry.source_scene_geometry_digest is not None
            and geometry.source_scene_geometry_digest != snapshot.scene.geometry_digest):
        raise ValueError("Field and mechanical geometry were captured from different scene geometry")
    if type(checkpoint) is not TipGunCheckpoint:
        raise ValueError("Finite geometry propagation requires a potential checkpoint")
    from temsim.physics.potential_wave_policy import potential_wave_operator_policy, potential_support_policy
    operator_policy = potential_wave_operator_policy(wave_options, checkpoint.record)
    validate_material_kinetic_record(checkpoint.record, wave_options)
    has_material_plane = "material_kinetic_native_plane" in checkpoint.record
    material_start = (exact_material_kinetic_plane(checkpoint.record, checkpoint.plane_z_mm)
                      if has_material_plane else None)
    validate_inherited_support(checkpoint.record, wave_options)
    stored_support = checkpoint.record.get("support_ledger")
    support_policy = potential_support_policy(wave_options, checkpoint.record)
    if stored_support is not None:
        saved = NumericalSupportLedger.from_dict(stored_support)
        if "support_ledger" in wave_options and wave_options["support_ledger"] != saved:
            raise ValueError("Saved numerical support history cannot be reset")
    from temsim.physics.potential_representation_estimates import RepresentationEstimateLedger
    stored_estimates = checkpoint.record.get("representation_estimates")
    if stored_estimates is not None and stored_support is not None:
        saved_estimates = RepresentationEstimateLedger.from_dict(stored_estimates)
        supplied_estimates = wave_options.get("representation_estimates")
        if isinstance(supplied_estimates, Mapping):
            supplied_estimates = RepresentationEstimateLedger.from_dict(supplied_estimates)
        if supplied_estimates is not None and supplied_estimates != saved_estimates:
            raise ValueError("Saved representation estimates cannot be reset")
    if wave_options.get("apertures"):
        raise ValueError("Finite geometry already owns physical apertures; duplicate thin aperture actions are forbidden")
    wave_options.pop("apertures", None)
    step = _number(maximum_step_m, "Maximum geometry step", positive=True)
    fraction = _number(geometry_pixel_fraction, "Geometry pixel fraction", positive=True)
    cells = _number(minimum_radius_cells, "Minimum radius cells", positive=True)
    if fraction > 1 or cells < 16:
        raise ValueError("Geometry requires pixel fraction <=1 and at least 16 cells per radius")
    if type(minimum_slab_steps) is not int or minimum_slab_steps < 1:
        raise ValueError("Minimum slab steps must be a positive integer")
    if type(maximum_geometry_steps) is not int or maximum_geometry_steps < 1:
        raise ValueError("Geometry step budget must be a positive integer")
    numerics = WaveGridNumerics() if grid_numerics is None else grid_numerics
    if type(numerics) is not WaveGridNumerics:
        raise ValueError("Geometry requires explicit WaveGridNumerics")
    raw_observations = np.asarray(observation_z_m)
    if (raw_observations.dtype.kind not in "iuf" or raw_observations.ndim != 1
            or not len(raw_observations) or not np.isfinite(raw_observations).all()
            or np.any(np.diff(raw_observations) <= 0)):
        raise ValueError("Geometry observation planes must be finite and strictly increasing")
    observations = tuple(map(float, raw_observations))
    previous = checkpoint.record.get("potential_geometry")
    start = float(checkpoint.plane_z_mm)*.001
    if previous is not None:
        if (checkpoint.plane_z_mm != previous["last_z_m"]*1000
                and checkpoint.plane_z_mm*.001 != previous["last_z_m"]):
            raise ValueError("Checkpoint native plane does not match its exact saved geometry plane")
        start = previous["last_z_m"]
    elif checkpoint.plane_z_mm == geometry.start_z_m*1000:
        start = geometry.start_z_m
    elif checkpoint.plane_z_mm == geometry.stop_z_m*1000:
        start = geometry.stop_z_m
    kinetic_ledger = wave_options.get("kinetic_phase_ledger")
    from temsim.physics.electric_wave_reference import ElectricWaveHistory, electric_rows
    electric_history = wave_options.get("electric_wave_history")
    stored_electric = checkpoint.record.get("electric_wave_history")
    if stored_electric is not None:
        restored_electric = ElectricWaveHistory.from_dict(stored_electric)
        if "electric_wave_history" in wave_options and electric_history != restored_electric:
            raise ValueError("Saved electric wave energy/error history cannot be reset")
        electric_history = restored_electric
    if (electric_history is not None or electric_rows(snapshot) or type(snapshot) is DynamicPotentialSnapshot) and len(checkpoint.beam.modes) != 1:
        raise ValueError("Electric geometry execution needs one independent energy history")
    if electric_history is not None:
        if type(electric_history) is not ElectricWaveHistory:
            raise ValueError("Electric geometry continuation requires an immutable history")
        if ((has_material_plane and material_start != electric_history.last_z_m)
                or (not has_material_plane and checkpoint.plane_z_mm != electric_history.last_z_m*1000.)):
            raise ValueError("Checkpoint plane differs from its electric wave energy history")
        start = electric_history.last_z_m
        electric_history.require(snapshot, start, checkpoint.beam.modes[0].energy_kev*1000.)
    stored_kinetic = checkpoint.record.get("kinetic_phase_ledger")
    if stored_kinetic is not None:
        restored = KineticPhaseLedger.from_dict(stored_kinetic)
        if "kinetic_phase_ledger" in wave_options and kinetic_ledger != restored:
            raise ValueError("Saved kinetic phase history and budgets cannot be reset or replaced")
        kinetic_ledger = restored
    if kinetic_ledger is not None and type(kinetic_ledger) is not KineticPhaseLedger:
        raise ValueError("Geometry requires an immutable KineticPhaseLedger or None")
    if (kinetic_ledger is not None and kinetic_ledger.events and electric_history is None
            and (electric_rows(snapshot) or type(snapshot) is DynamicPotentialSnapshot)):
        raise ValueError("An executed electric/dynamic phase history cannot lose its original energy and timing reference")
    if kinetic_ledger is not None:
        if type(kinetic_ledger) is not KineticPhaseLedger:
            raise ValueError("Geometry requires an immutable KineticPhaseLedger or None")
        if len(checkpoint.beam.modes) != 1:
            raise ValueError("Kinetic phase accounting currently requires one mode")
        if kinetic_ledger.binding is not None:
            # Recover the operator's exact metre plane from its authoritative
            # stored mm identity. Low-level ledger continuity stays strict;
            # genuine neighbouring observation planes are never merged.
            if has_material_plane:
                start = material_start
            else:
                if checkpoint.plane_z_mm != kinetic_ledger.last_z_m*1000.:
                    raise ValueError("Checkpoint plane does not match its exact kinetic ledger plane")
                start = kinetic_ledger.last_z_m
            if previous is not None and previous["last_z_m"] != start:
                raise ValueError("Saved geometry and kinetic ledger planes disagree")
        kinetic_ledger = kinetic_ledger.require(snapshot.numerical_digest,
            checkpoint.beam.modes[0].energy_kev*1000. if electric_history is None else electric_history.reference.entrance_energy_ev,
            wave_options.get("kinetic_order", 4), start)
    if not geometry.start_z_m <= start <= observations[0] <= observations[-1] <= geometry.stop_z_m:
        raise ValueError("Requested geometry propagation lies outside the fixed captured interval")
    policy = {"maximum_step_m": step, "geometry_pixel_fraction": fraction,
              "minimum_radius_cells": cells, "minimum_slab_steps": minimum_slab_steps}
    if previous is not None:
        if previous.get("geometry_digest") != geometry.digest:
            raise ValueError("Mechanical geometry changed; restart from the original entrance")
        if thaw_json(previous.get("policy")) != policy:
            raise ValueError("Boundary discretization changed; restart from the original entrance")
    if checkpoint.record.get("snapshot_digest", snapshot.numerical_digest) != snapshot.numerical_digest:
        raise ValueError("Potential snapshot changed; restart from the original entrance")
    old_losses = [] if previous is None else thaw_json(previous["physical_losses"])
    old_steps = 0 if previous is None else previous["accepted_steps"]
    old_refinements = 0 if previous is None else previous["sampling_refinements"]
    axial_retries = [] if previous is None else thaw_json(previous.get("axial_resource_retries", ()))
    if not isinstance(axial_retries, (tuple, list)) or any(not isinstance(row, Mapping) for row in axial_retries):
        raise ValueError("Invalid saved axial resource retry history")
    axial_retries = list(axial_retries)
    entry = start if previous is None else previous["entry_z_m"]
    entry_weight = checkpoint.beam.total_weight if previous is None else previous["entry_weight_per_reference"]
    losses, accepted_steps, refinement_count = old_losses, old_steps, old_refinements
    current, z, results = checkpoint, start, []
    if progress_callback is not None:
        progress_callback(0, 0, "Unified coherent boundary steps; adaptive total not yet known")
    total_distance = observations[-1]-start
    options = dict(wave_options)
    options.pop("kinetic_phase_ledger", None)
    options.pop("electric_wave_history", None)
    axial_resource_recovery = (numerics.automatic_refinement
        and options.get("support_projection_policy") in ("weighted_segment_and_retry", "preview_segment_and_retry", "preview_phase_segment_and_retry"))
    exponential = _number(options.pop("exponential_error", 1e-10), "Total exponential budget", positive=True)
    kinetic = _number(options.pop("kinetic_remainder_phase_limit", .01), "Total kinetic phase budget", positive=True)
    error_log = 0. if previous is None else previous["series_error_log_bound"]
    kinetic_bound = 0. if previous is None else previous["kinetic_phase_bound"]
    events = np.unique([value*.001 for piece in geometry.pieces for value in piece.event_z_mm()
                        if math.isfinite(value) and start < value*.001 <= observations[-1]])
    def stamp(value):
        metadata = thaw_json(value.record)
        metadata["potential_operator_policy"] = operator_policy
        metadata["potential_support_policy"] = support_policy
        metadata["snapshot_digest"] = snapshot.numerical_digest
        metadata["kinetic_phase_ledger"] = None if kinetic_ledger is None else kinetic_ledger.to_dict()
        metadata["electric_wave_history"] = None if electric_history is None else electric_history.to_dict()
        metadata["potential_geometry"] = {"schema": "executed-potential-geometry-v1",
            "geometry_digest": geometry.digest, "geometry_scope": _SCOPE,
            "source_scene_geometry_digest": geometry.source_scene_geometry_digest,
            "geometry_authority": "captured instrument scene" if geometry.source_scene_geometry_digest is not None
                else "explicit supplied primitives only; not an instrument capture",
            "capture_bounds_m": [geometry.start_z_m, geometry.stop_z_m], "entry_z_m": entry,
            "last_z_m": z, "policy": policy, "accepted_steps": accepted_steps,
            "sampling_refinements": refinement_count, "physical_losses": losses,
            "axial_resource_retries": axial_retries,
            "physical_loss_per_reference": math.fsum(row["lost_probability_per_reference"] for row in losses),
            "entry_weight_per_reference": entry_weight, "output_weight_per_reference": value.beam.total_weight,
            "series_error_log_bound": error_log, "kinetic_phase_bound": kinetic_bound,
            "normalization_repair": False, "edge_diffraction_qualified": False,
            "source_scope": "geometry execution begins at the supplied entrance checkpoint",
            "loss_scope": "physical absorption only; numerical support ledger remains separate",
            "production_admission": False}
        return replace(value, record=metadata)
    # A resumed output has already been masked at this plane. Idempotence is
    # mathematical, but skip the duplicate unit-shape factorization as well.
    if previous is None:
        current, initial_loss, count = _mask_checkpoint(current, geometry, z, z,
            numerics=numerics, cells=cells, cancelled=cancelled)
        losses.extend(initial_loss)
        refinement_count += count
        current = stamp(current)
    elif kinetic_ledger is not None and stored_kinetic is None:
        # Explicitly starting a new ledger at this supplied checkpoint binds
        # its entrance here; it does not certify earlier per-call propagation.
        current = stamp(current)
    for target in observations:
        while z < target:
            if cancelled():
                raise InterruptedError("Potential geometry propagation cancelled")
            if accepted_steps-old_steps >= maximum_geometry_steps:
                raise ValueError("Finite geometry execution exceeds its explicit step budget")
            end = target
            future = events[(events > z) & (events <= target)]
            if len(future):
                end = float(future[0])
            # Partition up to an actual face, instead of accumulating a fixed
            # step and inventing a one-ULP final interval. Genuine adjacent
            # user/event faces remain separate and are never epsilon-merged.
            count = max(1, math.ceil((end-z)/step))
            upper = end if count == 1 else z+(end-z)/count
            upper = _bounded_upper(z, upper, step)
            upper = _bounded_upper(z, upper,
                _geometry_step_limit(current, geometry, z, upper, fraction, minimum_slab_steps))
            pending_axial_retries = []
            for attempt in range(12):
                local_options = dict(options)
                if kinetic_ledger is not None:
                    local_options["kinetic_phase_ledger"] = kinetic_ledger
                if electric_history is not None:
                    local_options["electric_wave_history"] = electric_history
                if current.record.get("support_ledger") is not None:
                    local_options.pop("support_ledger", None)
                stored_support = current.record.get("support_ledger")
                if stored_support is not None:
                    saved = NumericalSupportLedger.from_dict(stored_support)
                    if "support_ledger" in local_options and local_options["support_ledger"] != saved:
                        raise ValueError("Saved numerical support history cannot be reset")
                    local_options["support_ledger"] = saved
                current_estimates = current.record.get("representation_estimates")
                if current_estimates is not None and stored_support is not None:
                    # The initial options describe the entrance only. Every
                    # accepted step advances both histories transactionally.
                    local_options["representation_estimates"] = RepresentationEstimateLedger.from_dict(current_estimates)
                elif stored_support is None and current_estimates is not None:
                    local_options.pop("representation_estimates", None)
                if local_options.get("support_ledger") is not None and len(current.beam.modes) != 1:
                    raise ValueError("Controlled support accounting currently requires one mode")
                # Use the explicit metre start interface: an intermediate
                # mm->m roundtrip must not merge adjacent observation planes.
                try:
                    outputs = [observe_potential_mode(snapshot, mode, z, (upper,),
                        maximum_step_m=upper-z, exponential_error=exponential*(upper-z)/total_distance,
                        kinetic_remainder_phase_limit=kinetic*(upper-z)/total_distance,
                        grid_numerics=numerics, cancelled=cancelled, **local_options)[0]
                        for mode in current.beam.modes]
                except WaveGridBudgetError as error:
                    from temsim.physics.potential_wave_carrier import _scalar_phase_resource_pressure
                    if not axial_resource_recovery or not _scalar_phase_resource_pressure(error):
                        raise
                    error.add_note(f"Whole optical step scalar resource pressure on [{z:.17g}, {upper:.17g}] m; "
                                   f"attempt {attempt+1}/12; original checkpoint and budgets retained")
                    if cancelled():
                        raise InterruptedError("Axial resource retry cancelled") from error
                    if attempt == 11:
                        error.add_note("Whole optical step exhausted its 12 bounded admission attempts")
                        raise
                    try:
                        smaller_upper = _bounded_upper(z, upper, (upper-z)/2.)
                    except ValueError:
                        smaller_upper = None
                    if smaller_upper is None:
                        error.add_note("Whole optical step cannot be bisected at this representable axial scale")
                        raise
                    if not z < smaller_upper < upper:
                        error.add_note("Whole optical step cannot produce a strictly smaller axial interval")
                        raise
                    diagnostic = {"lower_z_m": z, "failed_upper_z_m": upper,
                        "retry_upper_z_m": smaller_upper, "attempt": attempt+1,
                        "failure_kind": type(error).__name__, "reason": str(error),
                        "failure_notes": list(getattr(error, "__notes__", ())),
                        "failed_candidate_committed": False}
                    pending_axial_retries.append(diagnostic)
                    # The failed full step may already own a canonical output,
                    # clock solution and pending projection receipts. Release
                    # its private frames before retrying the same current mode.
                    import traceback
                    failures, seen = [error], set()
                    while failures:
                        failure = failures.pop()
                        if failure is None or id(failure) in seen:
                            continue
                        seen.add(id(failure))
                        traceback.clear_frames(failure.__traceback__)
                        failures.extend((failure.__cause__, failure.__context__))
                    del failures, failure
                    upper = smaller_upper
                    continue
                record = thaw_json(current.record)
                # This candidate writes a canonical mm plane from its executed
                # metre endpoint. The input receipt describes a different
                # native material face; retain it only on the unchanged input
                # if this candidate is discarded or no propagation occurs.
                record.pop("material_kinetic_native_plane", None)
                record.update(snapshot_digest=snapshot.numerical_digest, upstream_digest=current.digest,
                    modes=[thaw_json(output.metadata) for output in outputs],
                    support_ledger=outputs[0].metadata["support_ledger"] if len(outputs) == 1 else None)
                if len(outputs) == 1 and outputs[0].metadata.get("representation_estimates") is not None:
                    record["representation_estimates"] = thaw_json(outputs[0].metadata["representation_estimates"])
                    record["high_accuracy_eligible"] = False
                candidate_kinetic = None
                candidate_electric = None
                if len(outputs) == 1 and outputs[0].metadata["electric_wave_history"] is not None:
                    candidate_electric = ElectricWaveHistory.from_dict(outputs[0].metadata["electric_wave_history"])
                    record["electric_wave_history"] = candidate_electric.to_dict()
                if kinetic_ledger is not None:
                    candidate_kinetic = KineticPhaseLedger.from_dict(outputs[0].metadata["kinetic_phase_ledger"])
                    record["kinetic_phase_ledger"] = candidate_kinetic.to_dict()
                advanced = replace(current, beam=BeamState(tuple(output.mode for output in outputs),
                    current.beam.reference_plane), plane_z_mm=upper*1000., record=record)
                masked, new_losses, count = _mask_checkpoint(advanced, geometry, z, upper,
                    numerics=numerics, cells=cells, cancelled=cancelled)
                admitted = _geometry_step_limit(masked, geometry, z, upper, fraction, minimum_slab_steps, include_zero=True)
                if admitted < upper-z:
                    upper = _bounded_upper(z, upper, admitted)
                    del outputs, record, advanced, masked, new_losses, candidate_kinetic, candidate_electric
                    continue
                break
            else:
                raise ValueError("Optical/geometry grid changes failed to converge an admissible step")
            # Only a successfully admitted candidate consumes history/budgets.
            if pending_axial_retries and cancelled():
                raise InterruptedError("Axial resource retry cancelled before step commit")
            kinetic_ledger = candidate_kinetic
            electric_history = candidate_electric
            losses.extend(new_losses)
            refinement_count += count
            accepted_steps += 1
            axial_retries.extend(pending_axial_retries)
            stage_series, stage_kinetic = [], []
            for row in advanced.record.get("modes", ()):
                stage_series.append(math.fsum(math.log1p(stage["series_amplitude_error_bound"])
                    for stage in row.get("stages", ())))
                stage_kinetic.append(math.fsum(stage["kinetic_remainder_phase_bound"]
                    for stage in row.get("stages", ())))
            error_log += max(stage_series, default=0.)
            kinetic_bound += max(stage_kinetic, default=0.)
            z, current = upper, masked
            current = stamp(current)
            if progress_callback is not None:
                progress_callback(accepted_steps-old_steps, 0,
                    f"Unified coherent boundary steps at Z {z*1000.:.9g} mm; adaptive total not yet known")
        results.append(current)
    if progress_callback is not None:
        count = accepted_steps-old_steps
        progress_callback(count, count, "Unified coherent boundary interval complete")
    return tuple(results)


__all__ = ("PotentialBoundaryHost", "PotentialGeometrySnapshot", "capture_potential_geometry",
           "apply_potential_geometry", "observe_potential_geometry_checkpoint")
