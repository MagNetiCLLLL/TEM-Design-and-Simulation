"""Assembly readout preserves drive identity without asserting an OEM host."""
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.corrector_channels import channel_control_specs
from temsim.optics.stigmator_hardware import inspect_stigmator_hardware
from temsim.physics.core import multipole_focusing_fields, skew_quadrupole_field


@pytest.fixture(scope="module")
def base_state():
    return default_state()


@pytest.mark.parametrize("column,systems", [
    ("C2", ()), ("C3", ()),
    ("C3 + Probe Corrector", ("probe",)),
    ("C3 + Image Corrector", ("image",)),
    ("C3 + Probe Corrector + Image Corrector", ("probe", "image")),
])
def test_assembly_geometry_and_existing_control_aliases_are_reported_once(base_state, column, systems):
    state = type(base_state).from_dict(base_state.to_dict())
    assembly = AssemblyCatalog().apply(state, AssemblySelection("FEG", column, "No Energy Filter"))
    before = deepcopy(state.to_dict())
    report = inspect_stigmator_hardware(state)
    assert state.to_dict() == before
    assert {r.key for r in report.stigmators} == {
        "condenser_stigmator", "objective_stigmator", "diffraction_stigmator"}
    assert [a.channel for a in report.control_aliases] == [f"{s}.a1_fine" for s in systems]
    by_key = {s.key: s for s in state.stigmators}
    for record in report.stigmators:
        part = assembly.part(record.key)
        assert record.assembled_extent_z_mm == (part.start_z_mm, part.end_z_mm)
        assert record.field_reference_z_mm == by_key[record.key].z_mm
        assert record.field_width_fwhm_mm == by_key[record.key].length_mm
        assert record.assembly_parent_key == part.parent_key
        assert record.declared_physical_host_key == part.data.get("physical_host_key")
    for system, alias in zip(systems, report.control_aliases, strict=True):
        specs = channel_control_specs(state, system, "a1_fine")
        assert {spec.key for spec in specs} == {alias.component_key}
        assert tuple(spec.field for spec in specs) == alias.drive_fields
        assert "does_not_establish_oem_host" in alias.physical_host_evidence
    ownership, = report.shared_deflector_channels
    host = state.image_diffraction_deflector
    assert ownership.host_key == "image_diffraction_deflector"
    assert ownership.channel_key == "descan_deflector"
    assert ownership.host_interaction_z_mm == (host.upper_z_mm, host.lower_z_mm)


@pytest.mark.parametrize("basis", [(0., 45.), (15., 60.)])
def test_readout_preserves_snapshot_and_reports_nonzero_tensor_used_by_propagation(base_state, basis):
    state = type(base_state).from_dict(base_state.to_dict())
    stigma = state.condenser_stigmator
    stigma.strength_x_percent, stigma.strength_y_percent = 4., -3.
    stigma.channel_x_angle_deg, stigma.channel_y_angle_deg = basis
    snapshot = capture_instrument_snapshot(state)
    report = inspect_stigmator_hardware(state)
    record = next(r for r in report.stigmators if r.key == stigma.key)
    xx, yy = multipole_focusing_fields(np.array([stigma.z_mm]), state)
    xy = skew_quadrupole_field(np.array([stigma.z_mm]), state)
    np.testing.assert_allclose(record.tensor_at_reference_m2, ((xx[0], xy[0]), (xy[0], yy[0])), atol=1e-12)
    assert record.channel_commands_percent == (4., -3.)
    assert record.channel_basis_angles_deg == basis
    angles = np.deg2rad(2 * np.asarray(basis))
    amplitudes = stigma.max_strength_m2 * np.array([4., -3.]) / 100.
    normal, skew = np.dot(amplitudes, np.cos(angles)), np.dot(amplitudes, np.sin(angles))
    np.testing.assert_allclose(record.tensor_at_reference_m2, ((normal, skew), (skew, -normal)))
    if basis != (0., 45.):
        assert not np.allclose([normal, skew], amplitudes)
    assert capture_instrument_snapshot(state).physical_digest == snapshot.physical_digest
    payload = report.to_dict()
    json.dumps(payload, allow_nan=False)
    payload["stigmators"][0]["name"] = "detached"
    assert record.name != "detached"
    with pytest.raises(FrozenInstanceError):
        record.enabled = False


def test_disabled_hardware_keeps_alias_but_never_reports_nonzero_active_tensor(base_state):
    state = type(base_state).from_dict(base_state.to_dict())
    state.condenser_stigmator.enabled = False
    state.condenser_stigmator.strength_x_percent = 10.
    state.image_diffraction_deflector.enabled = False
    report = inspect_stigmator_hardware(state)
    record = next(r for r in report.stigmators if r.key == "condenser_stigmator")
    assert not record.enabled
    assert record.channel_commands_percent == (10., 0.)
    assert record.tensor_at_reference_m2 == ((0., 0.), (0., 0.))
    assert report.control_aliases[0].component_key == record.key
    assert not report.shared_deflector_channels[0].enabled


@pytest.mark.parametrize("duplicate", ["same_object", "same_key", "cross_collection"])
def test_duplicate_field_provider_cannot_be_misreported_as_a_control_alias(base_state, duplicate):
    state = type(base_state).from_dict(base_state.to_dict())
    stigma = state.condenser_stigmator
    if duplicate == "same_object":
        state.stigmators.append(stigma)
    elif duplicate == "same_key":
        state.stigmators.append(copy(stigma))
    else:
        state.corrector_elements.append(stigma)
    with pytest.raises(ValueError, match="Duplicate runtime field component"):
        inspect_stigmator_hardware(state)


def test_broken_descan_binding_is_rejected_without_rebinding_or_mutating_state(base_state):
    state = type(base_state).from_dict(base_state.to_dict())
    channel = state.descan_deflector
    object.__setattr__(channel, "_physical_host", copy(state.image_diffraction_deflector))
    wrong_host = channel._physical_host
    with pytest.raises(ValueError, match="not bound"):
        inspect_stigmator_hardware(state)
    assert channel._physical_host is wrong_host


def test_no_resolved_assembly_cannot_claim_physical_hardware(base_state):
    state = type(base_state).from_dict(base_state.to_dict())
    del state._resolved_assembly
    with pytest.raises(ValueError, match="resolved assembly"):
        inspect_stigmator_hardware(state)
