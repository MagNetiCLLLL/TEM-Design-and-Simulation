"""Main-column placement and optical-axis misalignment.

Offsets use mm, right-handed Euler angles use mrad, in Rz @ Ry @ Rx order.
Translations place both hardware and optics. Rotations tilt the optical model
about its unposed mechanical centre, without rotating material CAD or bores.
Parent optical transforms are applied outside the child's optical transform.
Shape, mechanical interference and excitation stay separate from optical tilt.
"""
import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


PHYSICAL_POSE_FIELDS = (
    "offset_x_mm", "offset_y_mm", "offset_z_mm",
    "rotation_x_mrad", "rotation_y_mrad", "rotation_z_mrad",
)
_PROFILES = frozenset({"magnetic_lens_assembly", "magnetic_lens_housing"})


def column_pose_kind(part):
    """Return ``hardware``/``field`` for supported post-gun column entries.

    A control channel without a verified material host moves its local field
    coordinates only; it does not acquire a fictitious solid. Known channels
    on shared hardware have no second editable placement. Gun, receiver and
    energy-filter branches are deliberately outside this model.
    """
    from temsim.component_keys import PROBE_CORRECTOR_KEYS, IMAGE_CORRECTOR_ALL_KEYS
    from temsim.component_representation import non_material_role, SHARED_DEFLECTOR_HOSTS
    from temsim.optics.field_hardware_binding import COIL_INSERT_ROLE, field_hardware_host_key
    key = str(part.get("key", ""))
    if (key.startswith(("feg_", "gun_", "thermionic_", "energy_filter_"))
            or part.get("branch") in {"gun", "source", "detection", "energy_filter"}
            or part.get("mechanical_part_role") == "custom_mechanical_copy"
            or key in SHARED_DEFLECTOR_HOSTS
            or field_hardware_host_key(part)
            or key in {"image_sad_plane", "probe_dp12_scan_deflector"}
            or non_material_role(part) == "virtual_reference"):
        return None
    if part.get("mechanical_part_role") == COIL_INSERT_ROLE:
        return "hardware"
    profile = part.get("mechanical_profile")
    if (profile == "magnetic_lens_assembly"
            or (profile == "magnetic_lens_housing" and not part.get("parent_key"))):
        return "hardware"
    independent = {
        "condenser_deflector", "beam_deflector", "dc_deflector", "ac_deflector",
        "image_diffraction_deflector", "condenser_stigmator", "objective_stigmator",
        "diffraction_stigmator", "condenser_aperture_2", "condenser_aperture_3",
        "objective_aperture", "selected_area_aperture", "projection_chamber_dpa_aperture",
        *PROBE_CORRECTOR_KEYS, *IMAGE_CORRECTOR_ALL_KEYS,
    }
    if key not in independent:
        return None
    return "field" if (non_material_role(part) == "control_channel"
                       or profile in {"quadrupole_field_channel", "deflector_field_channel"}) else "hardware"


def supports_physical_column_pose(part):
    return column_pose_kind(part) is not None


def supports_physical_lens_pose(part):
    """Compatibility entry point for the shared main-column placement law."""
    return supports_physical_column_pose(part)


def physical_pose_description(part):
    if column_pose_kind(part) == "field":
        subject = ("Local field-coordinate placement. This channel has no verified separate material host; "
                   "the placement moves its field without creating a material body. ")
    elif "aperture" in str(part.get("key", "")):
        subject = ("Aperture position and effective optical-stop tilt. Position moves the plate and "
                   "optical stop; tilt changes the effective acceptance plane only. "
                   "This idealised optical tilt does not model a mechanically tilted hole. ")
    else:
        subject = "Component position moves both its field and owned hardware. "
    return subject + ("Optical tilts are right-handed X, then Y, then Z about the component centre, in mrad. "
                      "They represent optical-axis imperfections; material envelopes and mechanical bores "
                      "do not rotate. Excitation and material dimensions are unchanged.")


def inherits_parent_lens_pose(part):
    """Distinguish rigid lens ownership from navigation or packing parents.

    Specimen and steering devices can sit in the objective's envelope without
    belonging to that lens's rigid magnetic assembly. Their children must not
    inherit the objective pose through that independently mounted device.
    """
    from temsim.component_representation import non_material_role
    from temsim.optics.field_hardware_binding import COIL_INSERT_ROLE
    if part.get("mechanical_profile") == "objective_electronics_sleeve":
        # The objective is its navigation/enclosure parent. Its separately
        # mounted electronics do not move with objective-axis placement.
        return False
    if part.get("mechanical_part_role") == COIL_INSERT_ROLE:
        return True
    if supports_physical_lens_pose(part) and part.get("mechanical_profile") in _PROFILES:
        return True
    if supports_physical_column_pose(part):
        # Apertures and non-round optical devices are separately mounted even
        # when parent_key names an enclosing lens for packing/navigation.
        return False
    key = str(part.get("key", ""))
    return not (
        key in {"sample", "sample_stage"}
        or key.endswith(("_deflector", "_stigmator"))
        or part.get("mechanical_profile") == "transverse_goniometer"
        or non_material_role(part)
        or any(part.get(name) for name in (
            "stigmator_structure", "deflector_structure", "scan_structure", "field_basis"))
    )


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def physical_pose_values(part):
    """Read canonical values, accepting historical lens CAD rigid placement.

    Canonical scalar values override the corresponding legacy component; they
    are never added twice. Scaling and Boolean CAD features remain shape edits.
    """
    if not supports_physical_lens_pose(part):
        return dict.fromkeys(PHYSICAL_POSE_FIELDS, 0.0)
    transform = part.get("model_3d", {}).get("transform", {})
    offset = transform.get("offset_mm", (0.0, 0.0, 0.0))
    angles = transform.get("rotation_deg", (0.0, 0.0, 0.0))
    if len(offset) != 3 or len(angles) != 3:
        raise ValueError("Component placement requires three offsets and rotations")
    legacy = tuple(offset) + tuple(_finite(v, "rotation_deg") * math.pi / 180.0 * 1000.0 for v in angles)
    return {name: _finite(part.get(name, value), name)
            for name, value in zip(PHYSICAL_POSE_FIELDS, legacy)}


def validate_physical_lens_pose(part):
    if not supports_physical_lens_pose(part):
        # Apertures and detectors have their own independent offset controls.
        if any(name in part for name in PHYSICAL_POSE_FIELDS[2:]):
            raise ValueError("Physical pose is supported only for post-gun main-column components; edit the physical host for a shared channel")
        return
    physical_pose_values(part)


def migrate_legacy_lens_pose(part):
    """Canonicalise an editable document without losing existing placement."""
    if not supports_physical_lens_pose(part):
        return
    part.update(physical_pose_values(part))
    transform = part.get("model_3d", {}).get("transform", {})
    transform.pop("offset_mm", None)
    transform.pop("rotation_deg", None)


def rotation_matrix_mrad(angles):
    x, y, z = np.asarray(angles, dtype=float) * 1.0e-3
    cx, cy, cz = np.cos((x, y, z))
    sx, sy, sz = np.sin((x, y, z))
    return np.array(((cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx),
                     (sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx),
                     (-sy, cy*sx, cy*cx)))


def effective_part_transform_mm(part, by_key):
    """Return the optical (R,t), with points = points @ R.T + t.

    ``part`` and ``by_key`` are module-coordinate TOML dictionaries. A child
    inherits its optical parent's placement even if the child has no controls.
    Specimen/steering/channel entries stop inheritance: their parent_key can
    express shared space or navigation rather than a rigid mechanical mount.
    Material geometry must use ``mechanical_part_transform_mm`` instead.
    """
    return _part_transform_mm(part, by_key, optical_tilt=True)


def mechanical_part_transform_mm(part, by_key):
    """Shared assembly translations, without optical-axis rotation of solids.

    Explicit shape edits of CAD-only parts are handled by the CAD shape code;
    this helper describes canonical column placement and its inherited offsets.
    """
    return _part_transform_mm(part, by_key, optical_tilt=False)


def _part_transform_mm(part, by_key, *, optical_tilt):
    rotation, translation = np.eye(3), np.zeros(3)
    seen = set()
    current = part
    from temsim.component_representation import SHARED_DEFLECTOR_HOSTS
    from temsim.optics.field_hardware_binding import field_hardware_host_key
    host = (SHARED_DEFLECTOR_HOSTS.get(str(part.get("key", "")))
            or field_hardware_host_key(part))
    if host:
        current = by_key.get(host)
        if current is None:
            raise ValueError(f"Shared field channel requires its captured physical host: {host}")
    while current is not None:
        key = current.get("key", "")
        if key in seen:
            raise ValueError(f"Cyclic lens pose ownership: {key}")
        seen.add(key)
        if supports_physical_lens_pose(current):
            values = physical_pose_values(current)
            local_r = (rotation_matrix_mrad([values[name] for name in PHYSICAL_POSE_FIELDS[3:]])
                       if optical_tilt else np.eye(3))
            centre = np.array((0.0, 0.0, float(current.get("local_center_z_mm", 0.0))))
            local_t = centre - local_r @ centre + np.array([values[name] for name in PHYSICAL_POSE_FIELDS[:3]])
            translation = local_r @ translation + local_t
            rotation = local_r @ rotation
        if not inherits_parent_lens_pose(current):
            break
        parent = current.get("parent_key")
        current = by_key.get(parent) if parent else None
    return rotation, translation


@dataclass(frozen=True)
class PlaneDatum:
    """A declared reference plane in global column mm, with a unit normal.

    A reference plane is not necessarily a material surface or a thin field.
    In particular the objective's virtual reference is distinct from its two
    finite field centres. No focal or conjugate plane is inferred here.
    """

    point_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    kind: str
    is_virtual: bool = False


@dataclass(frozen=True)
class ComponentDatums:
    """Placed declared datums, independent of the edited mesh bounding box."""

    mechanical_center_mm: tuple[float, float, float]
    optical_reference: PlaneDatum | None
    interaction_planes: tuple[PlaneDatum, ...]
    representation_role: str
    is_virtual: bool
    # Rows of the local-to-global rotation; columns are the local X/Y/Z axes.
    basis: tuple[tuple[float, float, float], ...]


def component_datums_mm(part, assembly_or_state):
    """Read main-column datums from an AssemblyPart and its captured context.

    Mechanical centres use CAD placement (translations); optical references
    use field registration (translations and optical-axis tilt). Both read the
    same captured module origin and ownership. ``assembly_or_state`` is a
    resolved assembly or a state carrying it.
    This does not mutate geometry or excitation, inspect mesh bounds, or infer
    an optical role for mechanical-only/imported components. CAD-only shape
    offsets, Boolean features and scaling do not move these optical datums.

    Aperture operating-hole offsets and detector readout-centre offsets are
    not part of this component-axis datum; their existing acceptance models
    apply those transverse values in the returned plane's local frame.
    Curved energy-filter branch parts require their own mounted path frames.
    """
    from temsim.component_representation import non_material_role

    assembly = getattr(assembly_or_state, "_resolved_assembly", assembly_or_state)
    if assembly is None:
        raise ValueError("Component datums require the captured assembly")
    data = part.data
    if data.get("branch_path_only", False):
        raise ValueError("Energy-filter path datums require the mounted branch frame")
    parts = tuple(getattr(assembly, "parts", ()))
    matches = [item for item in parts
               if item.key == part.key and item.module_key == part.module_key]
    if len(matches) != 1 or matches[0] != part:
        raise ValueError("Component datum is not from this captured assembly")
    by_key = {item.key: item.data for module in getattr(assembly, "modules", ())
              if module.key == part.module_key for item in module.parts}
    by_key.update({item.key: item.data for item in parts
                   if item.module_key == part.module_key})
    local_center = _finite(data.get("local_center_z_mm", part.center_z_mm),
                           "local_center_z_mm")
    module_origin = _finite(part.center_z_mm, "center_z_mm") - local_center
    rotation, translation = effective_part_transform_mm(data, by_key)
    mechanical_r, mechanical_t = mechanical_part_transform_mm(data, by_key)
    origin = np.array((0., 0., module_origin))
    normal = tuple(float(value) for value in rotation[:, 2])
    basis = tuple(tuple(float(value) for value in row) for row in rotation)

    def point(local_z):
        result = rotation @ np.array((0., 0., _finite(local_z, "datum_z_mm")))
        result += translation + origin
        if not np.isfinite(result).all():
            raise ValueError("Component datum must be finite")
        return tuple(float(value) for value in result)

    non_material = non_material_role(data)
    virtual = non_material == "virtual_reference"
    mechanical_only = data.get("mechanical_only", False)
    if not isinstance(mechanical_only, bool):
        raise ValueError("mechanical_only must be a boolean")
    role = non_material or ("mechanical_only" if mechanical_only else
                            str(data.get("mechanical_part_role", "hardware")))
    custom = str(data.get("mechanical_part_role", "")).startswith("custom_")
    optical_reference = None
    interactions = ()
    if not mechanical_only and not custom:
        if "virtual_reference_local_z_mm" in data:
            optical_reference = PlaneDatum(point(data["virtual_reference_local_z_mm"]),
                                           normal, "virtual_reference", True)
        elif "optical_reference_local_z_mm" in data:
            kind = ("virtual_reference" if virtual else "control_reference"
                    if non_material == "control_channel" else "optical_reference")
            optical_reference = PlaneDatum(point(data["optical_reference_local_z_mm"]),
                                           normal, kind, virtual)
        interactions = tuple(PlaneDatum(point(value), normal, "interaction_plane", virtual)
                             for value in data.get("interaction_centers_local_z_mm", ()))
    mechanical = mechanical_r @ np.array((0., 0., local_center)) + mechanical_t + origin
    return ComponentDatums(tuple(float(v) for v in mechanical), optical_reference, interactions,
                           role, virtual, basis)


def lens_pose_registration(state, lens_key):
    """Global-SI optical transform, including axis tilt, from captured assembly."""
    return _pose_registration(state, lens_key, optical_tilt=True)


def mechanical_pose_registration(state, component_key):
    """Global-SI material placement; optical tilt does not tilt a physical bore."""
    return _pose_registration(state, component_key, optical_tilt=False)


def _pose_registration(state, lens_key, *, optical_tilt):
    from temsim.component_representation import SHARED_DEFLECTOR_HOSTS
    lens_key = SHARED_DEFLECTOR_HOSTS.get(str(lens_key), lens_key)
    cache_key = lens_key if optical_tilt else ("mechanical", lens_key)
    from temsim.physics.lens_field_provider import CoordinateRegistration
    assembly = getattr(state, "_resolved_assembly", None)
    from temsim.column.module_assembly import ResolvedAssembly
    # Real assembled inputs are immutable and replaced after a geometry edit.
    # Cache their registration for per-step clipping/field queries on CPU. Do
    # not memoise mutable ad-hoc fixtures or retain this memo in a snapshot.
    cache = None
    if isinstance(assembly, ResolvedAssembly):
        captured = getattr(state, "_lens_pose_registration_cache", None)
        if captured is None or captured[0] is not assembly:
            captured = (assembly, {})
            state._lens_pose_registration_cache = captured
        cache = captured[1]
        if cache_key in cache:
            return cache[cache_key]
    parts = getattr(assembly, "parts", ())
    part = next((p for p in parts if p.key == lens_key), None)
    if part is None:
        result = CoordinateRegistration()
        if cache is not None:
            cache[cache_key] = result
        return result
    # Omitted optical-parent envelopes can still own active physical children.
    # Match the renderer's captured source context, then let active rows win.
    by_key = {p.key: p.data for module in getattr(assembly, "modules", ())
              if module.key == part.module_key for p in module.parts}
    by_key.update({p.key: p.data for p in parts if p.module_key == part.module_key})
    transform = effective_part_transform_mm if optical_tilt else mechanical_part_transform_mm
    rotation, translation = transform(part.data, by_key)
    origin = np.array((0.0, 0.0, part.center_z_mm-float(part.data.get("local_center_z_mm", part.center_z_mm))))
    global_t = origin - rotation @ origin + translation
    result = CoordinateRegistration(tuple(global_t * 1e-3), tuple(map(tuple, rotation)))
    if cache is not None:
        cache[cache_key] = result
    return result


def has_lens_pose(state, lens_key):
    registration = lens_pose_registration(state, lens_key)
    return (any(v != 0.0 for v in registration.origin_global_m)
            or registration.rotation_local_to_global != ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))


def without_lens_rigid_cad(part):
    """Copy CAD settings with physical rigid placement removed from mesh edits."""
    if not supports_physical_lens_pose(part) or "model_3d" not in part:
        return part
    model = dict(part["model_3d"])
    transform = dict(model.get("transform", {}))
    transform.pop("offset_mm", None)
    transform.pop("rotation_deg", None)
    model["transform"] = transform
    return {**part, "model_3d": model}


def has_nonrigid_lens_cad(part):
    """Distinguish unsupported shape edits from a supported rigid placement."""
    model = without_lens_rigid_cad(part).get("model_3d", {})
    return bool(model and (
        model.get("base", {}).get("kind", "existing") != "existing"
        or model.get("features")
        or tuple(model.get("transform", {}).get("scale_xy", (1., 1.))) != (1., 1.)
        or any(model.get("transform", {}).get("offset_mm", ()))
        or any(model.get("transform", {}).get("rotation_deg", ()))
    ))
