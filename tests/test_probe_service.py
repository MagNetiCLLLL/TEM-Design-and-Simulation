"""Probe service readbacks, hardware limits, field superposition and persistence."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.optics.column import default_state
from temsim.optics.probe_service import (
    ROUTES, apply_probe_service, preview_probe_service, output_reading,
    service_settings, set_probe_output_percent,
)
from temsim.physics.core import electron, hexapole_field_components, skew_quadrupole_field
from temsim.physics.instrument_magnetic import column_dipole_fields
from temsim.physics.posed_column_fields import capture_posed_column_fields
from temsim.runtime_parameters import runtime_targets, validate_runtime_assignment


@pytest.fixture(scope="module")
def initial():
    return encode_instrument(default_state())


@pytest.fixture
def state(initial):
    return decode_instrument(initial)


def drives(changes):
    return {(c.component, c.field): c.after-c.before for c in changes}


def test_a2_uses_two_different_inputs_and_retains_installed_angles(state):
    t = runtime_targets(state)
    angles = [t[f"probe_hp{i}_hexapole"].obj.orientation_rad for i in (1, 2)]
    assert dict(ROUTES["AT_A2.upper"].intrinsic) == {"HP1": 1., "HP2": -1.}
    upper = preview_probe_service(state, "AT_A2.upper", delta_percent=.0009)
    original = t["probe_hp2_hexapole"].obj.strength_m3
    assert drives(upper) == pytest.approx({
        ("probe_hp1_hexapole", "strength_m3"): 9000., ("probe_hp2_hexapole", "strength_m3"): -9000.,
    })
    assert t["probe_hp2_hexapole"].obj.strength_m3 == original
    assert apply_probe_service(state, "AT_A2.upper", delta_percent=.0009, expected=upper) == upper
    before = encode_instrument(state)
    with pytest.raises(ValueError, match="displayed drives"):
        apply_probe_service(state, "AT_A2.upper", delta_percent=.0009, expected=upper)
    assert encode_instrument(state) == before
    lower = apply_probe_service(state, "AT_A2.lower", delta_percent=.09)
    assert drives(lower) == pytest.approx({
        ("probe_tl21_lens", "percent"): .09, ("probe_tl22_lens", "percent"): -.09,
    })
    assert angles == [t[f"probe_hp{i}_hexapole"].obj.orientation_rad for i in (1, 2)]
    assert drives(apply_probe_service(state, "AT_A2.lower", delta_percent=-.09)) == pytest.approx(
        {k: -v for k, v in drives(lower).items()})


def test_learning_is_zero_by_default_and_requires_explicit_enable(state):
    state.probe_service_gains = {"AT_A2.lower": {"DP22x": 25.}}
    primary = drives(preview_probe_service(state, "AT_A2.lower", delta_percent=.1,
                                          learning={"DP22x": .2}, learning_enabled=False))
    assert set(primary) == {("probe_tl21_lens", "percent"), ("probe_tl22_lens", "percent")}
    learned = apply_probe_service(state, "AT_A2.lower", delta_percent=.1,
                                  learning={"DP22x": .2}, learning_enabled=True)
    assert drives(learned)[("probe_dp22_deflector", "kick_x_mrad")] == pytest.approx(.02)
    disabled = preview_probe_service(state, "AT_A2.lower", delta_percent=.1, learning_enabled=False)
    assert len(disabled) == 2
    with pytest.raises(ValueError, match="outside the intrinsic"):
        apply_probe_service(state, "AT_A2.lower", delta_percent=.01, intrinsic={"DP22x": .2})
    with pytest.raises(ValueError, match="Boolean"):
        apply_probe_service(state, "AT_A2.lower", delta_percent=.01, learning_enabled="false")
    with pytest.raises(TypeError):
        apply_probe_service(state, "AT_A2.lower", 9)


def test_absolute_percent_uses_declared_maximum_and_preserves_orthogonal_drives(state):
    assert output_reading(state, "QPolx").maximum == 300.
    set_probe_output_percent(state, "QPolx", percent=60.)
    set_probe_output_percent(state, "QPoly", percent=80.)
    assert output_reading(state, "QPolx").native == 180.
    assert output_reading(state, "QPoly").native == 240.
    assert set_probe_output_percent(state, "QPoly", percent=80.) == ()
    before = encode_instrument(state)
    with pytest.raises(ValueError, match="strength limit"):
        set_probe_output_percent(state, "QPoly", percent=81.)
    with pytest.raises(ValueError):
        set_probe_output_percent(state, "TL21", percent=-1.)
    assert encode_instrument(state) == before
    set_probe_output_percent(state, "BShx", percent=2.)
    set_probe_output_percent(state, "BTltx", percent=3.)
    assert output_reading(state, "BShx").percent == pytest.approx(2.)
    assert output_reading(state, "BTltx").percent == pytest.approx(3.)


def test_probe_inverse_fit_selects_fixed_mount_intrinsic_outputs(state):
    from temsim.optics.corrector_channels import CHANNELS, channel_control_specs

    specs = {key: channel_control_specs(state, "probe", key) for key in CHANNELS}
    assert {s.key for s in specs["a2_coarse"]} == {
        "probe_hp1_hexapole", "probe_hp2_hexapole", "probe_tl21_lens", "probe_tl22_lens"}
    assert {s.key for s in specs["a1_coarse"]} == {"probe_dp21_deflector", "probe_dp22_deflector"}
    assert {s.key for s in specs["a1_fine"]} == {"probe_qpol_quadrupole"}
    assert {s.key for s in specs["a2_fine"]} == {"probe_hpol_hexapole"}
    assert all(s.field != "orientation_rad" and not s.field.startswith("hexapole_")
               for group in specs.values() for s in group)


@pytest.mark.parametrize("key,harmonic", [("probe_qpc_quadrupole", 2), ("probe_hpol_hexapole", 3)])
def test_local_mount_roll_has_harmonic_phase_without_rotating_sample_coordinates(state, key, harmonic):
    from copy import copy
    from dataclasses import replace
    from test_posed_column_fields import component_state

    local, element = component_state(state, key)
    angle, strength = .17, 25.
    if harmonic == 2:
        element.strength_m2, element.skew_strength_m2 = strength, 0.
    else:
        element.strength_m3, element.skew_strength_m3, element.orientation_rad = strength, 0., 0.
    rolled = copy(local)
    rolled._resolved_assembly = replace(local._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "rotation_z_mrad": angle*1000.}) if part.key == key else part
        for part in local._resolved_assembly.parts))
    physical, = capture_posed_column_fields(rolled, include_aligned=True)
    if harmonic == 2:
        element.strength_m2 = strength*np.cos(harmonic*angle)
        element.skew_strength_m2 = strength*np.sin(harmonic*angle)
    else:
        element.strength_m3 = strength*np.cos(harmonic*angle)
        element.skew_strength_m3 = strength*np.sin(harmonic*angle)
    electrical, = capture_posed_column_fields(local, include_aligned=True)
    points = np.array([[1e-5, 2e-5, element.z_mm*.001], [-2e-5, 1e-5, element.z_mm*.001]])
    np.testing.assert_allclose(physical.field_at_global_positions_t(points),
                               electrical.field_at_global_positions_t(points), rtol=1e-12, atol=1e-25)


def test_auxiliary_bases_reach_native_and_potential_fields_at_fixed_mount(state):
    t = runtime_targets(state)
    q = t["probe_qpol_quadrupole"].obj
    h = t["probe_hpol_hexapole"].obj
    h.orientation_rad = .19
    h.strength_m3 = q.strength_m2 = 0.
    apply_probe_service(state, "AT_A2fine.lower", delta_percent=.0001)
    assert q.strength_m2 == q.skew_strength_m2 == 0.
    set_probe_output_percent(state, "QPoly", percent=.4/3)
    assert h.orientation_rad == .19 and h.strength_m3 == 0.
    isolated = SimpleNamespace(corrector_elements=[q, h], stigmators=[], deflectors=[],
                               simulation_mode="ideal", beam_voltage_kv=300.)
    np.testing.assert_allclose(skew_quadrupole_field([q.z_mm, q.z_mm + q.effective_length_mm / 2], isolated),
                               [.4, .4*np.exp(-.5*(2.355/2)**2)])
    assert skew_quadrupole_field(q.z_mm, isolated, exclude_mapped_keys={q.key}) == 0.
    np.testing.assert_allclose(hexapole_field_components(h.z_mm, isolated),
                               [-1000*np.sin(.57), 1000*np.cos(.57)])
    fields = {f.component_key: f for f in capture_posed_column_fields(
        state, include_aligned=True, component_keys=(q.key, h.key))}
    charge, momentum, _ = electron(state)
    coeff = {(px, py): v for px, py, v in fields[q.key].polynomial_terms}
    assert coeff[(1, 1)] == pytest.approx(-momentum/charge*.4, rel=2e-15, abs=0.)
    assert fields[h.key].scale == 1.
    q.enabled = h.enabled = False
    assert not skew_quadrupole_field(q.z_mm, isolated)
    assert hexapole_field_components(h.z_mm, isolated) == (0., 0.)


@pytest.mark.parametrize("raster", [False, True], ids=["wobble", "raster"])
def test_b2_bias_uses_existing_scan_coils_once_at_all_times(state, raster):
    ac = state.ac_deflector
    ac.wobble_enabled, ac.scan_enabled = not raster, raster
    ac.kick_x_mrad, ac.kick_y_mrad = .002, -.003
    scan_before = ac.to_dict()
    times = (0., .125, .375)

    def read(t):
        return [c for c in column_dipole_fields(state, time_s=t) if c.key.startswith("ac_deflector:")]

    original = [read(t) for t in times]
    coarse = preview_probe_service(state, "AT_B2.x", delta_percent=.03)
    assert coarse == preview_probe_service(state, "AT_B2fine.x", delta_percent=.03)
    apply_probe_service(state, "AT_B2fine.x", delta_percent=.03)
    apply_probe_service(state, "AT_B2.y", delta_percent=-.02)
    command = np.array((.03, -.02))  # Model mrad
    for t, baseline in zip(times, original):
        coils = read(t)
        assert len(coils) == len(baseline) == 2
        assert all(c.drive_keys == ("ac_deflector", "probe_dp12_scan_deflector") for c in coils)
        kicks = 1000*np.asarray([(c.event_dx_rad-b.event_dx_rad, c.event_dy_rad-b.event_dy_rad)
                                 for c, b in zip(coils, baseline)])
        z = np.array([c.event_z_mm for c in coils])
        np.testing.assert_allclose(kicks.sum(axis=0), command, atol=1e-14)
        np.testing.assert_allclose(z @ kicks, state.dp12_scan_deflector.upper_z_mm * command, atol=1e-10)
    assert ac.to_dict() == scan_before
    assert state.dp12_scan_deflector.kick_events() == ()
    state.probe_corrector_installed = False
    assert all(c.drive_keys == ("ac_deflector",) for c in read(0.))
    with pytest.raises(ValueError, match="Install"):
        apply_probe_service(state, "AT_B2.x", delta_percent=.001)


def test_unmeasured_routes_are_explicit_and_invalid_moves_are_atomic(state):
    for name in ("TL2SymFoc.single", "AT_S3.lower", "AT_A3C.lower", "AT_A1.lower"):
        with pytest.raises(ValueError, match="nonzero output"):
            apply_probe_service(state, name, delta_percent=.001)
    apply_probe_service(state, "AT_C3.single", delta_percent=.000001)
    before = encode_instrument(state)
    for route, gains in (("AT_A2.lower", {"TL22": 1000.}), ("AT_A2.upper", {"HP1": float("nan")}),
                         ("AT_A2.upper", {"TL22": 1.}), ("AT_B2.x", {"DP12x": 99.})):
        with pytest.raises(ValueError):
            apply_probe_service(state, route, delta_percent=1., intrinsic=gains)
    assert encode_instrument(state) == before
    t = runtime_targets(state)
    with pytest.raises(ValueError, match="individual limit"):
        validate_runtime_assignment(t["probe_dp12_scan_deflector"], "alignment_x_mrad", 99.)
    t["probe_tl22_lens"].obj.enabled = False
    held = t["probe_tl21_lens"].obj.percent
    with pytest.raises(ValueError, match="enabled"):
        apply_probe_service(state, "AT_A2.lower", delta_percent=.001)
    assert t["probe_tl21_lens"].obj.percent == held


def test_default_snapshot_remains_sparse_and_profiles_restore_combined_drives(state, initial, tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import apply_profile_values, read_profile, save_profile

    assert not any("skew_strength_m2" in row.get("fields", ()) or "alignment_x_mrad" in row.get("fields", ())
                   for row in initial["nodes"] if row["type"].startswith("temsim.optics.probe_corrector:"))
    assert encode_instrument(state) == initial
    apply_probe_service(state, "AT_A2fine.lower", delta_percent=.0003,
                        learning={"QPolx": -.2}, learning_enabled=True)
    apply_probe_service(state, "AT_B2.y", delta_percent=.003)
    q = runtime_targets(state)["probe_qpc_quadrupole"].obj
    q.strength_m2, q.skew_strength_m2 = 0., q.maximum_strength_m2
    target = runtime_targets(decode_instrument(encode_instrument(state)))
    assert target[q.key].obj.skew_strength_m2 == q.maximum_strength_m2
    assert target["probe_hpol_hexapole"].obj.skew_strength_m3 == 3000.
    assert target["probe_dp12_scan_deflector"].obj.alignment_y_mrad == .003
    path = tmp_path / "service.toml"
    save_profile(path, state, AssemblyCatalog().selection_for_resolved(state._resolved_assembly))
    _, settings = read_profile(path)
    q.strength_m2, q.skew_strength_m2 = q.maximum_strength_m2, 0.
    state.dp12_scan_deflector.alignment_y_mrad = 0.
    del state.probe_service_percent
    apply_profile_values(state, settings)
    assert state.probe_service_percent["learning"]["AT_A2fine.lower"] == {"QPolx": -.2}
    assert state.probe_service_percent["learning_enabled"] is True
    assert (q.strength_m2, q.skew_strength_m2, state.dp12_scan_deflector.alignment_y_mrad) == (0., q.maximum_strength_m2, .003)
    with pytest.raises(ValueError, match="strength limit"):
        apply_profile_values(state, {q.key: {"strength_m2": 1., "skew_strength_m2": q.maximum_strength_m2}})
    assert q.strength_m2 == 0.
    # A full older profile predates these additional fields and service gains.
    import tomllib
    import tomli_w
    old = tomllib.loads(path.read_text(encoding="utf-8"))
    del old["devices"]["probe_dp12_scan_deflector"], old["probe_service_percent"]
    old["probe_service_gains"] = {"AT_A2.lower": {"DP22x": -.0000555}}
    for record in old["devices"].values():
        record.pop("skew_strength_m2", None)
        record.pop("skew_strength_m3", None)
    path.write_text(tomli_w.dumps(old), encoding="utf-8")
    _, historical = read_profile(path)
    apply_profile_values(state, historical)
    assert (q.skew_strength_m2, state.dp12_scan_deflector.alignment_y_mrad) == (0., 0.)
    assert state.probe_service_gains == old["probe_service_gains"]
    assert service_settings(state, "AT_A2.lower") == ({"TL21": 1., "TL22": -1.}, {"DP22x": 0., "DP22y": 0.}, False)


def test_cuda_fixed_mount_skew_winding_matches_rotated_hexapole_wave():
    from dataclasses import replace

    from temsim.optics.probe_corrector import create_hpol_hexapole
    from temsim.physics.column_wave import _propagate_column
    from temsim.physics.compute_backend import coherent_cuda_capability, require_coherent_cuda
    from temsim.physics.wave_grid import WaveGridNumerics
    from test_corrector_channel_wave_transport import _full_field, _source
    from test_hexapole_thickness_response import isolated_hexapole_state

    capability = coherent_cuda_capability()
    if not capability.available:
        pytest.skip(capability.detail)
    policy = require_coherent_cuda("Require GPU")
    numerics = WaveGridNumerics(compute_backend=policy, maximum_pixels=256, maximum_working_bytes=1 << 28,
                                maximum_support_loss_fraction=0., maximum_spectral_loss_fraction=0.)
    # Preserve the analytical 6 um Gaussian and 64 um field of view while
    # resolving the propagated phase carrier with two samples per old pixel.
    source = _source()
    mode, = source.beam.modes
    x = (np.arange(128)-64)*.5e-6
    xx, yy = np.meshgrid(x, x)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*(6e-6)**2)).astype(complex)
    plane = replace(mode.plane, amplitude=amplitude / np.linalg.norm(amplitude),
                    basis_m=np.eye(2)*.5e-6)
    source = replace(source, beam=replace(source.beam, modes=(replace(mode, plane=plane),)))

    def execute(normal, skew, angle, *, assembly=0., error=0.):
        instrument = isolated_hexapole_state(.1, 0.)
        h = create_hpol_hexapole()
        h.z_mm, h.effective_length_mm = 2., .1
        h.strength_m3, h.skew_strength_m3, h.orientation_rad = normal, skew, angle
        instrument.corrector_elements = [h.validate()]
        if assembly or error:
            row = dict(key=h.key, local_center_z_mm=h.z_mm,
                       assembly_azimuth_deg=assembly, field_azimuth_error_deg=error)
            instrument._resolved_assembly = SimpleNamespace(parts=(SimpleNamespace(
                key=h.key, data=row, module_key="column", center_z_mm=h.z_mm, parent_key=None),))
        instrument.acceleration_enabled, instrument.acceleration_backend = True, policy
        result = _propagate_column(instrument, source, 4., maximum_step_mm=.02, grid_numerics=numerics)
        assert result.beam.total_weight == pytest.approx(source.beam.total_weight, rel=1e-10)
        return _full_field(result)

    zero = execute(0., 0., 0.)
    electrical = execute(0., 1e6, 0.)
    rotated = execute(1e6, 0., np.pi/6)
    np.testing.assert_allclose(electrical, rotated, rtol=1e-9, atol=1e-12)
    assert np.linalg.norm(electrical-zero) > 1e-4
    for assembly, error in ((30., 0.), (0., 30.), (23., 7.)):
        placed = execute(1e6, 0., 0., assembly=assembly, error=error)
        np.testing.assert_allclose(placed, rotated, rtol=1e-9, atol=1e-12)
