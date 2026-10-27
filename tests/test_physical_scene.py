"""Immutable P0 metadata and declared ownership, without changing transport."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.physical_scene import PhysicalSceneSnapshot, capture_physical_scene


def lightweight_state():
    lens = SimpleNamespace(key="lens", z_mm=12., percent=40., cs_mm=2.,
                           enabled=True, mechanical_clear_bore_diameter_mm=2.)
    part = SimpleNamespace(key="lens", module_key="column", source_file="column/test.toml",
        start_z_mm=11., center_z_mm=12., end_z_mm=13., length_mm=2., parent_key=None,
        data={"key": "lens", "mechanical_profile": "magnetic_lens_assembly", "length_mm": 2.,
              "local_center_z_mm": 4., "vacuum_inner_diameter_mm": 2.})
    return SimpleNamespace(lenses=[lens], apertures=[], corrector_elements=[], stigmators=[], deflectors=[],
        beam_voltage_kv=300., simulation_mode="analytical", _resolved_assembly=SimpleNamespace(parts=[part],
            vacuum_bore_segments=[{"start_z_mm": 11., "end_z_mm": 13., "inner_diameter_mm": 2.}]))


def test_detached_immutable_roundtrip_and_digest_validation():
    state = lightweight_state()
    snapshot = capture_physical_scene(state)
    restored = PhysicalSceneSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert restored.digest == snapshot.digest
    assert restored.geometry_digest == snapshot.geometry_digest
    assert restored.identity_complete
    assert restored.payload["geometry"]["parts"]["lens"]["resolved"]["center_z_mm"] == 12.
    assert restored.payload["geometry"]["parts"]["lens"]["definition"]["local_center_z_mm"] == 4.
    state.lenses[0].z_mm = 99.
    state._resolved_assembly.parts[0].data["length_mm"] = 100.
    assert snapshot.payload["geometry"]["runtime"]["lens"]["z_mm"] == 12.
    assert snapshot.payload["geometry"]["parts"]["lens"]["definition"]["length_mm"] == 2.
    with pytest.raises(TypeError):
        snapshot.payload["geometry"]["runtime"]["lens"]["z_mm"] = 88.
    detached = snapshot.to_dict()
    detached["payload"]["geometry"]["runtime"]["lens"]["z_mm"] = 88.
    with pytest.raises(ValueError, match="digest mismatch"):
        PhysicalSceneSnapshot.from_dict(detached)


def test_excitation_and_zero_field_keep_installed_geometry():
    state = lightweight_state()
    original = capture_physical_scene(state)
    state.lenses[0].percent = 0.
    state.lenses[0].enabled = False
    disabled = capture_physical_scene(state)
    assert disabled.digest != original.digest
    assert disabled.geometry_digest == original.geometry_digest
    assert disabled.payload["geometry"]["vacuum_bore_segments"]
    assert disabled.owner_for("lens") == "lens"
    state.lenses[0].cs_mm = 20.
    aberration = capture_physical_scene(state)
    assert aberration.digest != disabled.digest
    assert aberration.geometry_digest == disabled.geometry_digest
    state.lenses[0].mechanical_clear_bore_diameter_mm = 1.
    assert capture_physical_scene(state).geometry_digest != disabled.geometry_digest


def test_explicit_host_shared_host_and_unbound_control_are_distinct():
    state = lightweight_state()
    state.deflectors = [SimpleNamespace(key="image_diffraction_deflector", enabled=True),
                        SimpleNamespace(key="descan_deflector", enabled=True)]
    state.corrector_elements = [
        SimpleNamespace(key="probe_dph2_deflector", _field_hardware_host_key="coil_insert"),
        SimpleNamespace(key="probe_hpc_hexapole"),
        SimpleNamespace(key="probe_dp12_scan_deflector")]
    state._resolved_assembly.parts.append({"key": "coil_insert", "mechanical_only": True,
        "mechanical_part_role": "deflector_coil_insert"})
    scene = capture_physical_scene(state)
    assert scene.owner_for("descan_deflector") == "image_diffraction_deflector"
    assert scene.owner_for("probe_dph2_deflector") == "coil_insert"
    assert scene.owner_for("probe_hpc_hexapole") is None
    assert scene.owner_for("probe_dp12_scan_deflector") is None
    assert scene.owner_for("not_present") is None
    assert "descan_deflector" not in scene.payload["geometry"]["parts"]


def test_deflector_drive_and_physical_orientation_have_separate_geometry_identity():
    state = lightweight_state()
    state.deflectors = [SimpleNamespace(key="beam_deflector", upper_x_mrad=1., rotation_x_mrad=0.)]
    first = capture_physical_scene(state)
    state.deflectors[0].upper_x_mrad = 2.
    driven = capture_physical_scene(state)
    assert first.digest != driven.digest
    assert first.geometry_digest == driven.geometry_digest
    state.deflectors[0].rotation_x_mrad = .1
    assert capture_physical_scene(state).geometry_digest != driven.geometry_digest


def test_missing_host_and_unknown_provider_cannot_claim_complete_identity():
    state = lightweight_state()
    state.deflectors = [SimpleNamespace(key="descan_deflector")]
    state.lenses[0].extra_provider = object()
    scene = capture_physical_scene(state)
    assert not scene.identity_complete
    reasons = {row["reason"] for row in scene.payload["unresolved_inputs"]}
    assert "Opaque input has no declared stored parameters" in reasons
    assert "Declared physical host is absent from this snapshot" in reasons
    assert not PhysicalSceneSnapshot.from_dict(scene.to_dict()).identity_complete


def test_no_fields_properties_or_cached_results_are_evaluated():
    class Provider:
        def __init__(self):
            self.voltage = 2000.
            self._trace_cache = object()
            self._closed_gun_field = object()

        @property
        def electric_field(self):
            pytest.fail("Scene capture must not solve a field")

    state = lightweight_state()
    state.electron_gun = Provider()
    before = vars(state.electron_gun).copy()
    first = capture_physical_scene(state)
    state.electron_gun._trace_cache = object()
    state.electron_gun._closed_gun_field = object()
    second = capture_physical_scene(state)
    assert first.digest == second.digest
    assert state.electron_gun.voltage == before["voltage"]
    assert not second.identity_complete  # Custom behavior is never qualified by attributes alone.
    state.electron_gun.voltage = 2001.
    assert capture_physical_scene(state).digest != first.digest


def test_arrays_are_detached_content_identities_not_embedded_populations():
    state = lightweight_state()
    state.lenses[0].field_samples = np.array([1., 2., 3.])
    first = capture_physical_scene(state)
    row = first.payload["components"]["lens"]["inputs"]["attributes"]["field_samples"]
    assert set(row) == {"array_sha256", "dtype", "shape"}
    assert row["shape"] == (3,)
    state.lenses[0].field_samples[1] = 9.
    assert capture_physical_scene(state).digest != first.digest


def test_component_aliases_are_retained_and_conflicting_keys_rejected():
    state = lightweight_state()
    state.corrector_elements = [state.lenses[0]]
    scene = capture_physical_scene(state)
    assert scene.payload["components"]["lens"]["collections"] == ("lenses", "corrector_elements")
    state.corrector_elements = [SimpleNamespace(key="lens", z_mm=999.)]
    with pytest.raises(ValueError, match="Conflicting runtime components"):
        capture_physical_scene(state)


def test_declared_component_order_is_not_lost_in_keyed_inventory():
    state = lightweight_state()
    state.lenses.append(SimpleNamespace(key="second_lens", z_mm=15., percent=20.))
    scene = capture_physical_scene(state)
    state.lenses.reverse()
    reordered = capture_physical_scene(state)
    assert reordered.digest != scene.digest
    assert scene.payload["component_order"]["lenses"] == ("lens", "second_lens")
    assert reordered.payload["component_order"]["lenses"] == ("second_lens", "lens")


def test_default_instrument_capture_does_not_mutate_or_reconstruct(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import encode_instrument
    from temsim.immutable_json import json_digest
    state = default_state()
    before = json_digest(encode_instrument(state))

    def forbidden(*args, **kwargs):
        pytest.fail("Scene capture must not construct, resolve or solve")

    monkeypatch.setattr("temsim.optics.column.default_state", forbidden)
    monkeypatch.setattr("temsim.column.module_assembly.resolve_module_assembly", forbidden)
    monkeypatch.setattr("temsim.physics.lens_field_provider.resolve_runtime_lens_field_provider", forbidden)
    monkeypatch.setattr("temsim.physics.instrument_electric.capture_instrument_electric_field", forbidden)
    scene = capture_physical_scene(state)
    assert scene.identity_complete
    assert scene.owner_for("descan_deflector") == "image_diffraction_deflector"
    assert len(scene.payload["geometry"]["parts"]) == len(state._resolved_assembly.parts)
    assert scene.payload["geometry"]["vacuum_bore_segments"]
    assert scene.payload["circuits"]
    assert json_digest(encode_instrument(state)) == before


@pytest.mark.parametrize("variant", ["C2", "C3", "C3_BCOR", "C3_ImageCorrector", "C3_ProbeCorrector",
                                      "C3_ProbeCorrector_BCOR", "C3_ProbeCorrector_ImageCorrector"])
def test_all_column_definitions_preserve_control_and_circuit_inventory(variant):
    from temsim.module_manifest import read_document
    from temsim.component_representation import non_material_role
    from temsim.magnetic_circuits import circuit_inventory
    root = Path(__file__).resolve().parents[1] / "configs/instruments/column"
    parts = read_document(root / (variant+".toml"))["parts"]
    state = SimpleNamespace(_resolved_assembly=SimpleNamespace(parts=parts))
    original = deepcopy(parts)
    scene = capture_physical_scene(state)
    assert len(scene.payload["geometry"]["parts"]) == len(parts)
    assert len(scene.payload["circuits"]) == len(circuit_inventory(parts))
    for part in parts:
        if non_material_role(part) == "virtual_reference":
            assert scene.owner_for(part["key"]) is None
    assert parts == original


def test_invalid_record_schema_or_completeness_is_not_silently_accepted():
    record = capture_physical_scene(lightweight_state()).to_dict()
    record["schema"] = "future"
    with pytest.raises(ValueError, match="schema"):
        PhysicalSceneSnapshot.from_dict(record)
    record = capture_physical_scene(lightweight_state()).to_dict()
    record["payload"]["identity_complete"] = False
    with pytest.raises(ValueError, match="completeness"):
        PhysicalSceneSnapshot.from_dict(record)
