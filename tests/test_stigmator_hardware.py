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


@pytest.fixture
def editable_catalog(tmp_path):
    from temsim.paths import INSTRUMENT_CONFIG_ROOT
    from temsim.shared_tip import copy_catalog_tree
    root = tmp_path / "configs" / "instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    return AssemblyCatalog(root)


def _set_manifest_widths(catalog, values):
    from temsim.part_model_document import PartModelDocument
    for module, entries in (
        ("column/C3_ProbeCorrector.toml", zip(("condenser_stigmator", "objective_stigmator"), values[:2])),
        ("project_and_recording_system/NoEnergyFilter.toml", [("diffraction_stigmator", values[2])]),
    ):
        draft = PartModelDocument(catalog.root / module)
        for key, width in entries:
            draft.set_dimension(("parts", key, "effective_length_mm"), width)
        draft.save()


def test_manifest_width_initialization_edit_and_operating_override(editable_catalog):
    from temsim.column.state_layout import apply_physical_layout_to_state
    selection = AssemblySelection("FEG", "C3 + Probe Corrector", "No Energy Filter")
    state = default_state()
    _set_manifest_widths(editable_catalog, (4.5, 4.5, 6.))
    editable_catalog.apply(state, selection)
    assert [s.length_mm for s in state.stigmators] == [4.5, 4.5, 6.]
    for s in state.stigmators:
        s.length_mm = 2.
        s.strength_x_percent, s.strength_y_percent = 7., -2.
    editable_catalog.apply(state, selection, preserve_operating_parameters=True)
    assert [s.length_mm for s in state.stigmators] == [2.] * 3
    _set_manifest_widths(editable_catalog, (4., 4., 5.))
    editable_catalog.apply(state, selection, preserve_operating_parameters=True)
    assert [s.length_mm for s in state.stigmators] == [4., 4., 5.]
    for s in state.stigmators:
        assert (s.strength_x_percent, s.strength_y_percent) == (7., -2.)
        s.length_mm = 3.
    apply_physical_layout_to_state(state)
    assert [s.length_mm for s in state.stigmators] == [3.] * 3


def test_live_editor_copy_retains_old_manifest_widths_until_reload(editable_catalog):
    from types import SimpleNamespace
    from temsim.gui.main_window import MainWindow
    selection = AssemblySelection("FEG", "C3 + Probe Corrector", "No Energy Filter")
    state = default_state()
    _set_manifest_widths(editable_catalog, (4.5, 4.5, 6.))
    editable_catalog.apply(state, selection)
    for s in state.stigmators:
        s.length_mm = 2.
    # Real editor order: save first, copy the old live state, then reload.
    _set_manifest_widths(editable_catalog, (4., 4., 5.))
    candidate = MainWindow._copy_state_for_assembly(SimpleNamespace(state=state))
    assert [s.length_mm for s in candidate.stigmators] == [2.] * 3
    editable_catalog.apply(candidate, selection, preserve_operating_parameters=True)
    assert [s.length_mm for s in candidate.stigmators] == [4., 4., 5.]
    assert [s.length_mm for s in state.stigmators] == [2.] * 3


@pytest.mark.parametrize("quality", ["Preview", "Medium", "High accuracy"])
def test_calculation_snapshot_keeps_runtime_width_and_field(editable_catalog, quality):
    from threading import Event
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.profile_io import apply_profile_values
    state = default_state()
    _set_manifest_widths(editable_catalog, (4.5, 4.5, 6.))
    editable_catalog.apply(state, AssemblySelection("FEG", "C3 + Probe Corrector", "No Energy Filter"))
    apply_profile_values(state, {s.key: {"length_mm": 3., "strength_x_percent": 1.,
                                       "strength_y_percent": -2.} for s in state.stigmators})
    z = np.array([s.z_mm + 3. for s in state.stigmators])
    expected = (*multipole_focusing_fields(z, state), skew_quadrupole_field(z, state))
    request = CapturedCalculationRequest.capture(state, quality, 49, 1.)
    try:
        snapshot = request.prepare(Event()).snapshot
        assert [s.length_mm for s in snapshot.stigmators] == [3.] * 3
        assert [s.length_mm for s in state.stigmators] == [3.] * 3
        actual = (*multipole_focusing_fields(z, snapshot), skew_quadrupole_field(z, snapshot))
        np.testing.assert_array_equal(actual, expected)
        # The selected catalog differs from process defaults; later refreshes
        # must also preserve this captured width rather than reinterpret it.
        from temsim.column.state_layout import apply_physical_layout_to_state
        apply_physical_layout_to_state(snapshot)
        assert [s.length_mm for s in snapshot.stigmators] == [3.] * 3
    finally:
        if request._input_assets is not None:
            request._input_assets.close()


def test_snapshot_validates_field_width_against_captured_body(editable_catalog):
    from threading import Event
    from temsim.part_model_document import PartModelDocument
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.gui.main_window import MainWindow
    from types import SimpleNamespace
    draft = PartModelDocument(editable_catalog.root / "column/C3_ProbeCorrector.toml")
    draft.set_dimension(("parts", "condenser_stigmator", "length_mm"), 40.)
    draft.set_dimension(("parts", "condenser_stigmator", "effective_length_mm"), 35.)
    draft.save()
    state = default_state()
    editable_catalog.apply(state, AssemblySelection("FEG", "C3 + Probe Corrector", "No Energy Filter"))
    assert state.condenser_stigmator.mechanical_length_mm == 40.
    state.condenser_stigmator.length_mm = 34.
    snapshot = CapturedCalculationRequest.capture(state, "Preview", 49, 1.).prepare(Event()).snapshot
    assert snapshot.condenser_stigmator.length_mm == 34.
    assert MainWindow._copy_state_for_assembly(SimpleNamespace(state=state)).condenser_stigmator.length_mm == 34.
    invalid = state.to_dict()
    next(row for row in invalid["stigmators"] if row["key"] == "condenser_stigmator")["length_mm"] = 41.
    with pytest.raises(ValueError, match="field length must fit"):
        type(state).from_dict(invalid, resolved_assembly=state._resolved_assembly)
