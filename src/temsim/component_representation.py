"""Separate optical/control coordinates from independently modelled hardware.

Control-channel names do not necessarily identify separate mechanical
cartridges. Keep their calibrated coordinates in the optical model; consumers
of material geometry must not turn their legacy envelopes into solids.
"""

from temsim.component_keys import (
    IMAGE_CORRECTOR_DP11_DEFLECTOR, IMAGE_CORRECTOR_DP12_DEFLECTOR,
    IMAGE_CORRECTOR_DP21_DEFLECTOR, IMAGE_CORRECTOR_DP22_DEFLECTOR,
    IMAGE_CORRECTOR_DPH1_DEFLECTOR, IMAGE_CORRECTOR_DPH2_DEFLECTOR,
    IMAGE_CORRECTOR_DSH_DEFLECTOR, IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
    IMAGE_CORRECTOR_HPOL_HEXAPOLE, IMAGE_CORRECTOR_ISH_DEFLECTOR,
    IMAGE_CORRECTOR_QPOL_QUADRUPOLE,
    PROBE_DP11_DEFLECTOR, PROBE_DP21_DEFLECTOR, PROBE_DP22_DEFLECTOR,
    PROBE_DPH1_DEFLECTOR, PROBE_DPH2_DEFLECTOR,
    PROBE_HPC_HEXAPOLE, PROBE_HPOL_HEXAPOLE,
    PROBE_QPC_QUADRUPOLE, PROBE_QPH1_QUADRUPOLE,
    PROBE_QPH2_QUADRUPOLE, PROBE_QPOL_QUADRUPOLE,
)


PROBE_CONTROL_CHANNEL_KEYS = frozenset({
    PROBE_DP11_DEFLECTOR, PROBE_DP21_DEFLECTOR, PROBE_DP22_DEFLECTOR,
    PROBE_DPH1_DEFLECTOR, PROBE_DPH2_DEFLECTOR,
    PROBE_HPC_HEXAPOLE, PROBE_HPOL_HEXAPOLE,
    PROBE_QPC_QUADRUPOLE, PROBE_QPH1_QUADRUPOLE,
    PROBE_QPH2_QUADRUPOLE, PROBE_QPOL_QUADRUPOLE,
})


IMAGE_CONTROL_CHANNEL_KEYS = frozenset({
    IMAGE_CORRECTOR_DP11_DEFLECTOR, IMAGE_CORRECTOR_DP12_DEFLECTOR,
    IMAGE_CORRECTOR_DP21_DEFLECTOR, IMAGE_CORRECTOR_DP22_DEFLECTOR,
    IMAGE_CORRECTOR_DPH1_DEFLECTOR, IMAGE_CORRECTOR_DPH2_DEFLECTOR,
    IMAGE_CORRECTOR_DSH_DEFLECTOR, IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
    IMAGE_CORRECTOR_HPOL_HEXAPOLE, IMAGE_CORRECTOR_ISH_DEFLECTOR,
    IMAGE_CORRECTOR_QPOL_QUADRUPOLE,
})


SHARED_DEFLECTOR_HOSTS = {
    "descan_deflector": "image_diffraction_deflector",
}


SHARED_DEFLECTOR_GEOMETRY_FIELDS = frozenset({
    "local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "length_mm",
    "optical_reference_local_z_mm", "interaction_centers_local_z_mm",
    "vacuum_inner_diameter_mm", "mechanical_outer_diameter_mm",
    "mechanical_clear_bore_diameter_mm", "mechanical_coil_length_mm",
    "effective_thickness_mm", "mechanical_inter_coil_gap_mm", "parent_key",
    "mechanical_overlap_group", "mechanical_overlap_role", "mechanical_overlap_reason",
})

SHARED_DEFLECTOR_READ_ONLY_FIELDS = SHARED_DEFLECTOR_GEOMETRY_FIELDS | {
    "layout_role", "layout_owner", "physical_host_key", "physical_host_status",
    "mechanical_profile", "model_3d", "material_regions",
    "offset_x_mm", "offset_y_mm", "offset_z_mm",
    "rotation_x_mrad", "rotation_y_mrad", "rotation_z_mrad",
    "assembly_azimuth_deg", "field_azimuth_error_deg",
}


def shared_deflector_field_owner(part_key, field) -> str:
    """Identify fields that are supplied by shared hardware, not its channel."""
    if str(field) in SHARED_DEFLECTOR_READ_ONLY_FIELDS or str(field).startswith("deflector_support_"):
        return SHARED_DEFLECTOR_HOSTS.get(str(part_key), "")
    return ""


def belongs_to_material_selection(row, keys):
    """Include a physical child or a single return shared by selected windings."""
    return (row.get("parent_key") in keys or bool(keys.intersection(row.get("magnetic_lens_keys", ())))
            or bool(keys.intersection(row.get("mechanical_shared_winding_keys", ()))))


def vacuum_context_only(row):
    """Return plates surround a separate beam tube, including in older manifests."""
    return (bool(row.get("axial_vacuum_context_only"))
            or row.get("mechanical_part_role") in {"return_plate", "radial_return_plates", "objective_return_plates",
                                                  "objective_pole_sleeves"})


def non_material_role(part_data, *, profile=None) -> str:
    """Return the non-material role, including compatibility for older files.

An empty result leaves existing physical/reference-surface handling intact.
In particular, a detector plane is not virtual merely because it is thin.
"""
    key = str(part_data.get("key", ""))
    role = str(part_data.get("layout_role", ""))
    if key == "probe_dp12_scan_deflector":
        return "virtual_reference"
    if (key in PROBE_CONTROL_CHANNEL_KEYS or key in IMAGE_CONTROL_CHANNEL_KEYS
            or key in SHARED_DEFLECTOR_HOSTS):
        return "control_channel"
    if role in {"control_channel", "virtual_reference"}:
        return role
    shape = profile if profile is not None else part_data.get("mechanical_profile", "")
    if (shape in {"virtual_plane", "virtual_layout"}
            or part_data.get("virtual") or part_data.get("is_virtual")):
        return "virtual_reference"
    return ""


def is_reference_without_material(part):
    """A control channel can own an explicit winding even with no cartridge."""
    if not non_material_role(part):
        return False
    from temsim.multipole_winding import FIELDS
    native = part.get("model_3d", {}).get("base", {}).get("kind", "existing") == "existing"
    return not (native and FIELDS.intersection(part))


def representation_note(part_data, *, profile=None) -> str:
    from temsim.multipole_winding import FIELDS as MULTIPOLE_FIELDS
    if MULTIPOLE_FIELDS.intersection(part_data) and non_material_role(part_data,profile=profile):
        return ('Explicit engineering winding-only CAD at the existing control reference. '
            'It is not verified OEM winding geometry or an additional opaque cartridge. '
            'The original transport boundary ownership and clear bore remain unchanged. '
            'Saved turns and pack geometry define the winding. Where explicitly calibrated, '
            'prescribed currents define its ideal vacuum field source.')
    role = non_material_role(part_data, profile=profile)
    if role == "control_channel":
        reference_host = str(part_data.get("optical_reference_host_key", ""))
        if reference_host and not part_data.get("physical_host_key"):
            default_owner = "image_corrector" if str(part_data.get("key", "")) in IMAGE_CONTROL_CHANNEL_KEYS else "probe_corrector"
            owner = str(part_data.get("layout_owner", default_owner)).replace("_", " ")
            return (
                f"Control channel in {owner} at the optical reference of {reference_host}. "
                "Its Z follows that component; the finite field remains active. "
                "Winding shape and separate housing remain unverified; "
                "no independent material body is drawn."
            )
        declared_host = str(part_data.get("physical_host_key", ""))
        if declared_host and part_data.get("physical_host_status") == "modelled":
            return (
                f"Control channel driving {declared_host}. Geometry and field "
                "position are read from that winding insert; this channel has "
                "no independent material body. The insert is a documented "
                "model reconstruction, not verified OEM winding geometry."
            )
        host = SHARED_DEFLECTOR_HOSTS.get(str(part_data.get("key", "")))
        if host:
            owner = host.replace("_", " ")
            return (
                f"Control channel driving the same physical coils as {owner}. "
                "The marker shows their shared optical-model reference position; "
                "no independent material body is defined. The coil positions "
                "and dimensions are adjustable model values, not OEM measurements."
            )
        default_owner = (
            "image_corrector" if str(part_data.get("key", "")) in IMAGE_CONTROL_CHANNEL_KEYS
            else "probe_corrector"
        )
        owner = str(part_data.get("layout_owner", default_owner)).replace("_", " ")
        return (
            f"Control channel in {owner}. The marker shows its optical-model "
            "reference position. Its physical host and separate housing are "
            "unverified; no independent material body is defined."
        )
    if role == "virtual_reference":
        return "Virtual reference plane; no independent material body or thickness."
    return ""
