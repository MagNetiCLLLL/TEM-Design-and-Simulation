"""Declared coil inserts own field geometry; legacy channels stay independent."""
from copy import deepcopy
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from temsim import module_manifest
from temsim.lens_pose import (
    effective_part_transform_mm, lens_pose_registration,
    supports_physical_column_pose,
)
from temsim.optics.field_hardware_binding import resolve_field_hardware_parts
from temsim.paths import INSTRUMENT_CONFIG_ROOT


CHANNELS = ("probe_dph2_deflector", "probe_dph1_deflector", "probe_dp22_deflector",
            "probe_dp21_deflector", "probe_dp11_deflector")


def raw_document():
    return tomllib.loads((INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml").read_text(encoding="utf-8"))


def parts(document):
    return {p["key"]: p for p in document["parts"]}


def test_runtime_metadata_describes_finite_field_without_breaking_drive_events():
    from temsim.optics.single_plane_deflector import SinglePlaneDeflectorComponent
    from temsim.optics.probe_corrector import Dph2DeflectorDefinition, Dp22DeflectorDefinition
    for definition in (Dph2DeflectorDefinition, Dp22DeflectorDefinition):
        assert definition.interaction_kind == "finite_transverse_dipole_field"
    drive = SimpleNamespace(enabled=True, z_mm=1126.65, kick_x_mrad=.05, kick_y_mrad=-.02)
    assert SinglePlaneDeflectorComponent.INTERACTION_KIND == "finite_transverse_dipole_field"
    assert SinglePlaneDeflectorComponent.kick_events(drive) == ((1126.65, .00005, -.00002),)


@pytest.mark.parametrize("variant", ["C3_ProbeCorrector", "C3_ProbeCorrector_ImageCorrector"])
def test_read_validate_and_runtime_share_host_reference_without_changing_other_optics(variant):
    path = INSTRUMENT_CONFIG_ROOT / f"column/{variant}.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    document = module_manifest.read_document(path)
    resolved = parts(document)
    validated = parts(module_manifest.validate_document(document))
    for channel in CHANNELS:
        row, host = resolved[channel], resolved[resolved[channel]["physical_host_key"]]
        assert validated[channel] == row
        assert validated[host["key"]] == host
        assert row["local_center_z_mm"] == host["local_center_z_mm"]
        assert row["optical_reference_local_z_mm"] == host["optical_reference_local_z_mm"]
        assert row["effective_thickness_mm"] == host["effective_thickness_mm"] == 4.
        assert row["mechanical_clear_bore_diameter_mm"] == 20.
        if "dph" in channel:
            hp = resolved[host["optical_reference_host_key"]]
            assert row["optical_reference_local_z_mm"] == hp["optical_reference_local_z_mm"]
    before = parts(raw)
    for key in ("probe_hp2_hexapole", "probe_hp1_hexapole", "probe_tl22_lens",
                "probe_tl21_lens", "probe_tl12_lens", "sample"):
        assert resolved[key] == before[key]


def test_host_move_rebinds_channel_and_insert_once_without_mutating_document():
    original = raw_document()
    rows = parts(original)
    hp = rows["probe_hp2_hexapole"]
    for name in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "optical_reference_local_z_mm"):
        hp[name] += 3.
    rows["probe_dph2_coils"]["effective_thickness_mm"] = 3.
    rows["probe_dph2_deflector"]["optical_reference_local_z_mm"] = 10.
    before = deepcopy(original)
    resolved = resolve_field_hardware_parts(original)
    actual = parts(resolved)
    assert original == before
    assert resolve_field_hardware_parts(resolved) == resolved
    assert actual["probe_dph2_coils"]["local_center_z_mm"] == hp["optical_reference_local_z_mm"]
    assert actual["probe_dph2_deflector"]["optical_reference_local_z_mm"] == hp["optical_reference_local_z_mm"]
    assert actual["probe_dph2_deflector"]["effective_thickness_mm"] == 3.
    assert actual["probe_dph2_coils"]["length_mm"] == parts(before)["probe_dph2_coils"]["length_mm"]


def test_no_declared_binding_preserves_legacy_channel_positions_and_pose():
    document = raw_document()
    document["parts"] = [row for row in document["parts"] if row.get("mechanical_part_role") != "deflector_coil_insert"]
    for row in document["parts"]:
        if row["key"] in CHANNELS:
            row.pop("physical_host_key", None)
            row["physical_host_status"] = "unverified"
    row = parts(document)["probe_dph2_deflector"]
    row.update(local_center_z_mm=637.2, optical_reference_local_z_mm=637.2, rotation_y_mrad=5.)
    before = deepcopy(document)
    assert resolve_field_hardware_parts(document) is document
    assert document == before
    assert supports_physical_column_pose(row)


@pytest.mark.parametrize("case,error", [("missing", "Missing field hardware host"),
    ("cycle", "Cyclic field hardware binding"), ("thickness", "thickness"),
    ("pose", "physical host pose")])
def test_bad_explicit_bindings_fail_instead_of_silently_moving_fields(case, error):
    document = raw_document()
    rows = parts(document)
    if case == "missing":
        rows["probe_dph2_deflector"]["physical_host_key"] = "absent_coil"
    elif case == "cycle":
        rows["probe_dph2_coils"]["optical_reference_host_key"] = "probe_dph2_coils"
    elif case == "thickness":
        rows["probe_dph2_coils"]["effective_thickness_mm"] = 100.
    else:
        rows["probe_dph2_deflector"]["offset_x_mm"] = .1
    with pytest.raises(ValueError, match=error):
        resolve_field_hardware_parts(document)


def test_insert_and_channel_share_one_pose_and_dph_follows_hp_mount():
    rows = parts(resolve_field_hardware_parts(raw_document()))
    hp, coil, channel = (rows[key] for key in ("probe_hp2_hexapole", "probe_dph2_coils", "probe_dph2_deflector"))
    hp.update(rotation_y_mrad=3., offset_x_mm=.2)
    coil.update(rotation_x_mrad=2., offset_y_mm=.03)
    assert supports_physical_column_pose(coil)
    assert not supports_physical_column_pose(channel)
    rotation, shift = effective_part_transform_mm(coil, rows)
    host_rotation, host_shift = effective_part_transform_mm(hp, rows)
    centre = np.array((0., 0., coil["local_center_z_mm"]))
    np.testing.assert_allclose(rotation @ centre + shift,
        host_rotation @ (centre + (0., .03, 0.)) + host_shift, atol=1e-12)
    for actual, expected in zip(effective_part_transform_mm(channel, rows), (rotation, shift)):
        np.testing.assert_array_equal(actual, expected)
    assembled = tuple(SimpleNamespace(key=key, data=row, module_key="column", center_z_mm=row["local_center_z_mm"] + 450.)
                      for key, row in rows.items())
    state = SimpleNamespace(_resolved_assembly=SimpleNamespace(parts=assembled))
    assert lens_pose_registration(state, channel["key"]) == lens_pose_registration(state, coil["key"])
    # An independent DP insert is not accidentally attached to the HP mount.
    dp = rows["probe_dp11_coils"]
    np.testing.assert_array_equal(effective_part_transform_mm(dp, rows)[0], np.eye(3))


@pytest.fixture(scope="module")
def runtime():
    from temsim.optics.column import default_state
    return default_state()


@pytest.mark.parametrize("key", CHANNELS)
def test_runtime_editor_cannot_override_host_geometry_but_keeps_electrical_controls(runtime, key):
    from temsim.runtime_parameters import runtime_targets, editable_parameters, validate_runtime_assignment
    target = runtime_targets(runtime)[key]
    names = {p.name for p in editable_parameters(target)}
    assert {"kick_x_mrad", "kick_y_mrad", "enabled"} <= names
    assert "effective_thickness_mm" not in names
    with pytest.raises(ValueError, match="edit probe_.*_coils instead"):
        validate_runtime_assignment(target, "effective_thickness_mm", 9.)
    assert validate_runtime_assignment(target, "kick_x_mrad", .1) == .1
    assert validate_runtime_assignment(target, "enabled", False) is False


def test_legacy_runtime_without_binding_keeps_independent_geometry_edit():
    from temsim.optics.probe_corrector import create_dph2_deflector
    from temsim.runtime_parameters import RuntimeTarget, editable_parameters, validate_runtime_assignment
    component = create_dph2_deflector()
    target = RuntimeTarget(component.key, component.name, component)
    assert "effective_thickness_mm" in {p.name for p in editable_parameters(target)}
    assert validate_runtime_assignment(target, "effective_thickness_mm", 3.) == 3.


@pytest.mark.parametrize("key", CHANNELS)
def test_runtime_nonzero_deflector_is_one_finite_field_and_wave_uses_same_plan(runtime, key):
    from temsim.physics.core import build_propagation_plan
    from temsim.physics.column_wave import _prepare_column
    from temsim.physics.instrument_magnetic import active_column_events, column_dipole_fields
    component = deepcopy(next(c for c in runtime.corrector_elements if c.key == key))
    component.enabled = True
    component.kick_x_mrad, component.kick_y_mrad = .2, -.1
    row = runtime._resolved_assembly.part(key)
    host = runtime._resolved_assembly.part(row.data["physical_host_key"])
    assert component.z_mm == host.center_z_mm
    assert component.effective_thickness_mm == 4.
    state = SimpleNamespace(stigmators=[], corrector_elements=[component], deflectors=[], lenses=[],
        _resolved_assembly=runtime._resolved_assembly, condenser_system=runtime.condenser_system,
        sample=runtime.sample, simulation_mode="analytical", beam_voltage_kv=300.,
        step_mm=.2, history_step_mm=.2, acceleration_enabled=False, acceleration_backend="CPU",
        projector_mode="diffraction", equivalent_image_lenses_enabled=False,
        simulation_time_s=0., apertures=[], recording_planes=[])
    coil, = column_dipole_fields(state)
    assert (coil.lower_m * 1e3, coil.upper_m * 1e3) == pytest.approx((component.z_mm-2., component.z_mm+2.))
    assert (coil.event_dx_rad, coil.event_dy_rad) == pytest.approx((.2e-3, -.1e-3))
    events = active_column_events(state)
    assert len(events) == 1
    start, stop = component.z_mm-3., component.z_mm+3.
    ray = build_propagation_plan(state, start, stop, events)
    wave, _, _, owners = _prepare_column(state, start, stop, .2)
    assert len(owners) == 1 and owners[0]["finite_field"]
    assert owners[0]["effective_thickness_mm"] == pytest.approx(4.)
    assert owners[0]["kick_rad"] == pytest.approx((.2e-3, -.1e-3))
    for plan in (ray, wave):
        assert np.any(plan.dipole_by_t) and np.any(plan.dipole_bx_t)
        assert np.count_nonzero(plan.kick_x_rad) == np.count_nonzero(plan.kick_y_rad) == 0
        integrated = np.sum(plan.dipole_by_t[1::3] * plan.step_m)
        assert integrated == pytest.approx(coil.by_t * .004, rel=1e-12)
