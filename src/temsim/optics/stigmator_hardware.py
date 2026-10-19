"""Read-only column stigmator and shared-channel audit.

An operating-channel alias establishes which existing drive is changed, not
which OEM winding or cartridge contains it. This report retains that boundary:
geometry comes from the resolved assembly and fields from the runtime objects.
The electron-gun stigmator has a separate magnetic-gradient model and is outside
this column-quadrupole readout. No field is created, moved or recalibrated here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from temsim.component_representation import SHARED_DEFLECTOR_HOSTS, non_material_role


@dataclass(frozen=True)
class StigmatorHardware:
    key: str
    name: str
    field_reference_z_mm: float
    assembled_extent_z_mm: tuple[float, float]
    assembly_parent_key: str | None
    representation_role: str
    declared_physical_host_key: str | None
    physical_host_status: str
    geometry_status: str
    coil_topology: str | None
    coil_count: int | None
    enabled: bool
    field_model: str
    field_width_fwhm_mm: float
    maximum_strength_m2: float
    channel_commands_percent: tuple[float, float]
    channel_basis_angles_deg: tuple[float, float]
    tensor_at_reference_m2: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class StigmatorControlAlias:
    channel: str
    component_key: str
    drive_fields: tuple[str, ...]
    relationship: str = "same_runtime_drives_not_a_second_excitation"
    physical_host_evidence: str = "control_alias_does_not_establish_oem_host"


@dataclass(frozen=True)
class SharedDeflectorOwnership:
    channel_key: str
    host_key: str
    host_interaction_z_mm: tuple[float, float]
    enabled: bool
    relationship: str = "independent_command_summed_on_existing_host_coils"


@dataclass(frozen=True)
class StigmatorHardwareReport:
    stigmators: tuple[StigmatorHardware, ...]
    control_aliases: tuple[StigmatorControlAlias, ...]
    shared_deflector_channels: tuple[SharedDeflectorOwnership, ...]
    geometry_frame: str = "assembled_axial_coordinates_before_rigid_pose"
    scope: str = "column_stigmators_excluding_gun_and_corrector_multipoles"
    qualification: str = "model_structure_and_drive_identity_not_oem_geometry_validation"

    def to_dict(self):
        """Return detached JSON-compatible content for a GUI or saved report."""
        return asdict(self)


def _unique_components(state):
    """Reject a duplicated provider instead of reporting it as another channel.

    The propagation loops sum each stigmator/corrector provider. Repeating one
    object or one physical key in those lists would therefore count its field
    twice. Separate logical commands on a declared shared host remain valid.
    """
    result, identities = {}, set()
    for collection in ("stigmators", "corrector_elements", "deflectors"):
        for component in getattr(state, collection, ()):
            key = str(component.key)
            if key in result or id(component) in identities:
                raise ValueError(f"Duplicate runtime field component: {key}")
            result[key] = component
            identities.add(id(component))
    return result


def _finite(value, description):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def inspect_stigmator_hardware(state) -> StigmatorHardwareReport:
    """Describe installed column stigmators without mutating their state.

    Assembly-parent membership is reported separately from an explicitly
    declared physical host. In particular A1 fine's use of the condenser
    stigmator does not identify a corrector-internal coil or relocate its field.
    Disabled but installed hardware is retained; absent hardware is omitted.
    """
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is None:
        raise ValueError("Stigmator hardware audit requires a resolved assembly")
    parts = {str(part.key): part for part in assembly.parts}
    if len(parts) != len(assembly.parts):
        raise ValueError("Duplicate resolved assembly part key")
    components = _unique_components(state)
    records = []
    for component in getattr(state, "stigmators", ()):
        key = str(component.key)
        part = parts.get(key)
        if part is None or not bool(getattr(component, "_layout_installed", True)):
            continue
        data = part.data
        z = _finite(component.z_mm, f"{key} field reference")
        xx, yy, xy = (float(value) for value in component.quadrupole_tensor_m2(z))
        if not all(math.isfinite(value) for value in (xx, yy, xy)):
            raise ValueError(f"{key}: non-finite stigmator tensor")
        records.append(StigmatorHardware(
            key=key, name=str(component.name), field_reference_z_mm=z,
            assembled_extent_z_mm=(float(part.start_z_mm), float(part.end_z_mm)),
            assembly_parent_key=part.parent_key,
            representation_role=non_material_role(data) or str(data.get("mechanical_part_role", "configured_component")),
            declared_physical_host_key=data.get("physical_host_key"),
            physical_host_status=str(data.get("physical_host_status", "not_declared")),
            geometry_status=str(data.get("mechanical_geometry_status", data.get("coil_topology_evidence", "not_declared"))),
            coil_topology=data.get("coil_topology"), coil_count=data.get("coil_count"),
            enabled=bool(component.enabled), field_model=str(component.field_model),
            field_width_fwhm_mm=float(component.length_mm),
            maximum_strength_m2=float(component.max_strength_m2),
            channel_commands_percent=(float(component.strength_x_percent), float(component.strength_y_percent)),
            channel_basis_angles_deg=(float(component.channel_x_angle_deg), float(component.channel_y_angle_deg)),
            tensor_at_reference_m2=((xx, xy), (xy, yy)),
        ))
    records.sort(key=lambda record: (record.field_reference_z_mm, record.key))
    installed = {record.key for record in records}
    aliases = []
    # These are the existing corrector_channels.channel_control_specs bindings.
    # They are operating aliases; no inferred mechanical owner is attached.
    for system, key in (("probe", "condenser_stigmator"), ("image", "objective_stigmator")):
        if bool(getattr(state, f"{system}_corrector_installed", False)):
            if key not in installed:
                raise ValueError(f"{system} A1 fine requires installed {key}")
            aliases.append(StigmatorControlAlias(f"{system}.a1_fine", key,
                ("strength_x_percent", "strength_y_percent")))
    shared = []
    for channel_key, host_key in SHARED_DEFLECTOR_HOSTS.items():
        channel = components.get(channel_key)
        if channel_key not in parts or channel is None:
            continue
        host = components.get(host_key)
        if host_key not in parts or host is None:
            raise ValueError(f"{channel_key}: missing installed physical host {host_key}")
        if getattr(channel, "_physical_host", None) is not host:
            raise ValueError(f"{channel_key}: runtime channel is not bound to {host_key}")
        declared = parts[channel_key].data.get("physical_host_key")
        if declared != host_key:
            raise ValueError(f"{channel_key}: resolved host disagrees with runtime binding")
        shared.append(SharedDeflectorOwnership(channel_key, host_key,
            (float(host.upper_z_mm), float(host.lower_z_mm)),
            bool(channel.enabled and host.enabled)))
    return StigmatorHardwareReport(tuple(records), tuple(aliases), tuple(shared))
