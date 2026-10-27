"""Finite physical boundaries retain absolute flux and immutable scene identity."""
from dataclasses import replace
from functools import lru_cache
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.immutable_json import thaw_json
from temsim.lens_pose import rotation_matrix_mrad
from temsim.optics.column import default_state
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.posed_wave_hardware import PosedWaveBore
from temsim.physics.posed_wave_aperture import PosedWaveAperture
from temsim.physics.potential_geometry import (
    PotentialBoundaryHost, PotentialGeometrySnapshot, capture_potential_geometry,
    apply_potential_geometry, observe_potential_geometry_checkpoint,
)
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError
from temsim.physics.wave_reference import AxialWaveReference
from test_potential_wave_carrier import _snapshot, _wave


def _geometry(*pieces, start=1., stop=1.000004, authority=None):
    return PotentialGeometrySnapshot(start, stop, tuple(PotentialBoundaryHost(piece.key, (piece.key,), (piece,))
        for piece in pieces), source_scene_geometry_digest=authority)


def _plate(radius_m=1e-6, thickness_m=1e-6, z=1.0000015, x_m=0., angle_mrad=0., key="iris"):
    rotation = rotation_matrix_mrad((0., angle_mrad, 0.))
    centre = np.array((0., 0., z))
    origin = centre-rotation@centre+np.array((x_m, 0., 0.))
    return PosedWaveAperture(key, (z-thickness_m/2)*1000, (z+thickness_m/2)*1000,
        radius_m*1000, CoordinateRegistration(tuple(origin), tuple(map(tuple, rotation))))


def _checkpoint(wave=None):
    wave = _wave(n=80, pitch=5e-8, width=2e-7, curvature=-100., tilt=(1e-6, -2e-6)) if wave is None else wave
    mode = WaveMode(wave, .27, TIP_REFERENCE, "controlled:0", 300., AxialWaveReference(2e-12, 3e-25))
    return TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 1000., 1e-12,
        {"source_scope": "controlled test mode; no physical gun claim"})


@pytest.mark.parametrize("column", [row.name for row in AssemblyCatalog().columns])
def test_capture_all_seven_columns_preserves_unique_physical_hosts_and_real_plate_thickness(column):
    state, catalog = default_state(), AssemblyCatalog()
    catalog.apply(state, AssemblySelection("FEG", column, "No Energy Filter"))
    geometry = capture_potential_geometry(state, .6, 3.5)
    assert len({host.host_key for host in geometry.boundaries}) == len(geometry.boundaries)
    assert geometry.source_scene_geometry_digest is not None
    assert not any(host.host_key.endswith("dp12_scan_deflector") for host in geometry.boundaries)
    assert not any(host.host_key == "descan_deflector" for host in geometry.boundaries)
    # The reduced on-axis vacuum envelope can be narrower than the shared
    # deflector bore; in that case it supplies no additional wall constraint.
    assert sum(host.host_key == "image_diffraction_deflector" for host in geometry.boundaries) <= 1
    plates = {piece.key: piece for piece in geometry.pieces if type(piece) is PosedWaveAperture}
    for aperture in state.apertures:
        active = all(getattr(aperture, key, True) for key in ("installed", "enabled", "inserted"))
        assert (aperture.key in plates) == active
        if active:
            declared = PosedWaveAperture.from_component(state, aperture, CoordinateRegistration())
            assert plates[aperture.key].local_end_z_mm-plates[aperture.key].local_start_z_mm == pytest.approx(
                declared.local_end_z_mm-declared.local_start_z_mm)
    assert PotentialGeometrySnapshot.from_dict(json.loads(json.dumps(geometry.to_dict()))).digest == geometry.digest


def test_excitation_off_does_not_remove_hardware_but_dimensions_and_insertion_change_identity():
    state = default_state()
    original = capture_potential_geometry(state, .6, 3.5)
    for lens in state.lenses:
        lens.percent = 0.
        lens.enabled = False
    assert capture_potential_geometry(state, .6, 3.5).digest == original.digest
    aperture = next(item for item in state.apertures if item.enabled)
    aperture.inserted = False
    removed = capture_potential_geometry(state, .6, 3.5)
    assert removed.digest != original.digest
    assert not any(type(piece) is PosedWaveAperture and piece.key == aperture.key for piece in removed.pieces)
    aperture.inserted = True
    aperture.radius_mm *= .9
    assert capture_potential_geometry(state, .6, 3.5).digest != original.digest


def test_capture_rejects_equivalent_pupil_and_missing_real_thickness():
    state = default_state()
    state.sample.wave_objective_aperture_strategy = "equivalent_pupil"
    with pytest.raises(ValueError, match="physical_plane"):
        capture_potential_geometry(state, .6, 3.5)
    aperture = SimpleNamespace(key="objective_aperture", z_mm=10., radius_mm=.1, length_mm=1.)
    with pytest.raises(ValueError, match="plate_thickness_mm"):
        PosedWaveAperture.from_component(SimpleNamespace(), aperture, CoordinateRegistration())


def test_posed_shared_deflector_has_one_physical_host_and_no_logical_copy():
    from temsim.immutable_json import freeze_json
    state = default_state()
    assembly = state._resolved_assembly
    parts = tuple(replace(part, data=freeze_json({**part.data, "offset_x_mm": 1.}))
        if part.key == "image_diffraction_deflector" else part for part in assembly.parts)
    state._resolved_assembly = replace(assembly, parts=parts)
    geometry = capture_potential_geometry(state, .6, 3.5)
    hosts = [host for host in geometry.boundaries if host.host_key == "image_diffraction_deflector"]
    assert len(hosts) == 1 and len(hosts[0].pieces) == 1
    assert not any("descan_deflector" in host.source_keys for host in geometry.boundaries)
    assert hosts[0].pieces[0].registration.origin_global_m[0] == .001


def test_capture_retains_fixed_tube_hidden_behind_moved_narrower_lens():
    from temsim.column.module_assembly import _module_vacuum_segments
    from temsim.immutable_json import freeze_json
    state = default_state()
    assembly = state._resolved_assembly
    changed = tuple(replace(part, data=freeze_json({**part.data, **values}))
        if (values := ({"offset_x_mm": 1.} if part.key == "condenser_lens_1" else
                       {"vacuum_inner_diameter_mm": 5.} if part.key == "condenser_lens_1_lower_pole" else {}))
        else part for part in assembly.parts)
    reduced = []
    for module in assembly.modules:
        reference = next(part for part in changed if part.module_key == module.key)
        origin = reference.center_z_mm-float(reference.data["local_center_z_mm"])
        reduced.extend(_module_vacuum_segments(module, origin, changed))
    state._resolved_assembly = replace(assembly, parts=changed, vacuum_bore_segments=tuple(reduced))
    pole = next(part for part in changed if part.key == "condenser_lens_1_lower_pole")
    z = pole.start_z_mm+1.
    geometry = capture_potential_geometry(state, .45, 3.5)
    tube = next(piece for piece in geometry.pieces if piece.key == "@vacuum_tube:c1_c2_to_upper_objective"
        and piece.local_start_z_mm < z < piece.local_end_z_mm)
    assert tube.radius_mm == 2.88
    assert not tube.swept_transmission_mask(np.array([3.1]), np.array([0.]), z, z+1)[0]
    moved = next(piece for piece in geometry.pieces if piece.key == pole.key)
    assert moved.swept_transmission_mask(np.array([3.1]), np.array([0.]), z, z+1)[0]


def test_snapshot_is_detached_immutable_and_rejects_duplicate_hosts_and_tampering():
    geometry = _geometry(_plate())
    record = geometry.to_dict()
    restored = PotentialGeometrySnapshot.from_dict(record)
    assert restored.digest == geometry.digest
    record["boundaries"][0]["pieces"][0]["radius_mm"] *= 2
    assert restored.pieces[0].radius_mm == 1e-3
    with pytest.raises(ValueError, match="digest"):
        PotentialGeometrySnapshot.from_dict(record)
    with pytest.raises(ValueError, match="only once"):
        replace(geometry, boundaries=geometry.boundaries*2)
    with pytest.raises(ValueError, match="Duplicate material"):
        PotentialBoundaryHost("host", ("a", "b"), (_plate(key="a"), _plate(key="b")))
    with pytest.raises(ValueError, match="trimmed strings"):
        PotentialBoundaryHost(["host"], ("a",), (_plate(),))


def test_swept_tilted_bore_masks_hidden_shoulders_and_matches_analytic_cylinder_intersection():
    aperture = _plate(radius_m=1e-6, thickness_m=3e-6, z=1.000002, x_m=1.5e-7, angle_mrad=180.)
    bore = PosedWaveBore("bore", aperture.local_start_z_mm, aperture.local_end_z_mm,
                         aperture.radius_mm, aperture.registration)
    geometry = _geometry(bore)
    wave = _wave(n=120, pitch=4e-8, width=6e-7)
    xy = wave.coordinates_m()*1000
    assert np.all(bore.transmission_mask(xy[0], xy[1], 1000.))
    assert np.all(bore.transmission_mask(xy[0], xy[1], 1000.004))
    result, receipt = apply_potential_geometry(wave, geometry, 1., 1.000004)
    # Independent line/cylinder calculation in the material's local frame.
    rotation, origin = bore.registration.rotation_array, bore.registration.origin_array_m*1000
    p = np.moveaxis(np.concatenate((xy, np.full((1, *xy.shape[1:]), 1000.)), axis=0), 0, -1)
    local = (p-origin)@rotation
    direction = np.array((0., 0., 1.))@rotation
    t0 = (bore.local_start_z_mm-local[..., 2])/direction[2]
    t1 = (bore.local_end_z_mm-local[..., 2])/direction[2]
    a, b = local+t0[..., None]*direction, local+t1[..., None]*direction
    passed = (np.hypot(a[..., 0], a[..., 1]) < bore.radius_mm) & (np.hypot(b[..., 0], b[..., 1]) < bore.radius_mm)
    np.testing.assert_array_equal(result.amplitude, np.where(passed, wave.amplitude, 0.))
    assert receipt["physical_loss_probability"] == pytest.approx(np.sum(abs(wave.amplitude[~passed])**2), rel=1e-15)
    assert result.probability < wave.probability


def test_tiny_physical_loss_is_summed_directly_and_shared_geometry_is_not_added_twice():
    amplitude = np.zeros((64, 64), complex)
    amplitude[32, 32], amplitude[32, 60] = 1., 1e-11*(.6+.8j)
    wave = PlaneWave(amplitude, np.eye(2)*5e-8, np.zeros(2))
    first, second = _plate(key="first"), _plate(key="second")
    result, receipt = apply_potential_geometry(wave, _geometry(first, second), 1., 1.000004)
    assert wave.probability == result.probability == 1.
    assert receipt["physical_loss_probability"] == pytest.approx(1e-22, rel=1e-15, abs=0.)
    assert sum(row["lost_probability"] > 0 for row in receipt["physical_losses"]) == 1
    assert result.amplitude[32, 32] == wave.amplitude[32, 32]


def test_unresolved_opening_refines_incident_wave_before_absorption_and_respects_fixed_grid():
    wave = _wave(n=48, pitch=1e-7, width=3e-7)
    geometry = _geometry(_plate(radius_m=5e-7))
    with pytest.raises(WaveSamplingError, match="resolved incident"):
        apply_potential_geometry(wave, geometry, 1., 1.000004,
            grid_numerics=WaveGridNumerics(automatic_refinement=False))
    result, receipt = apply_potential_geometry(wave, geometry, 1., 1.000004,
        grid_numerics=WaveGridNumerics(maximum_pixels=256))
    assert receipt["sampling_refinements"]
    assert result.amplitude.shape[0] >= 48*3.2
    assert .2 < result.probability < .9


def test_scene_geometry_authority_mismatch_rejected_before_propagation():
    snapshot = _snapshot()
    geometry = _geometry(authority="f"*64)
    with pytest.raises(ValueError, match="different scene geometry"):
        observe_potential_geometry_checkpoint(snapshot, geometry, _checkpoint(), (1.000001,))
    valid = replace(geometry, source_scene_geometry_digest=snapshot.scene.geometry_digest)
    result, = observe_potential_geometry_checkpoint(snapshot, valid, _checkpoint(), (1.,))
    assert result.record["potential_geometry"]["geometry_authority"] == "captured instrument scene"


def test_absolute_weight_carriers_and_phase_survive_entrance_absorption_and_noop_resume():
    checkpoint = _checkpoint()
    geometry = _geometry(_plate(radius_m=1e-6, z=1., x_m=6e-7))
    result, = observe_potential_geometry_checkpoint(_snapshot(), geometry, checkpoint, (1.,))
    before, after = checkpoint.beam.modes[0], result.beam.modes[0]
    raw, receipt = apply_potential_geometry(before.plane, geometry, 1., 1.)
    np.testing.assert_allclose(after.plane.amplitude*np.sqrt(after.weight_per_reference_electron),
                               raw.amplitude*np.sqrt(before.weight_per_reference_electron), rtol=1e-15)
    for key in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(after.plane, key), getattr(before.plane, key))
    assert after.axial_reference == before.axial_reference
    assert result.record["potential_geometry"]["physical_loss_per_reference"] == pytest.approx(.27*receipt["physical_loss_probability"])
    resumed, = observe_potential_geometry_checkpoint(_snapshot(), geometry, result, (1.,))
    assert resumed.digest == result.digest


def test_resume_rejects_geometry_policy_change_and_duplicate_aperture_options():
    geometry, checkpoint, snapshot = _geometry(), _checkpoint(), _snapshot()
    result, = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (1.,))
    with pytest.raises(ValueError, match="geometry changed"):
        observe_potential_geometry_checkpoint(snapshot, _geometry(_plate()), result, (1.000001,))
    with pytest.raises(ValueError, match="discretization changed"):
        observe_potential_geometry_checkpoint(snapshot, geometry, result, (1.000001,), maximum_step_m=1e-5)
    with pytest.raises(ValueError, match="duplicate thin"):
        observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (1.,), apertures=(object(),))


def test_exact_neighbouring_planes_are_not_merged_and_store_resume_preserves_absolute_complex_field(tmp_path):
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    geometry, checkpoint, snapshot = _geometry(), _checkpoint(), _snapshot()
    middle, end = 1.000001, np.nextafter(1.000001, np.inf)
    first, last = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (middle, end), maximum_step_m=1e-6,
                                                       kinetic_order=2)
    assert last.record["potential_geometry"]["last_z_m"] == end
    assert last.record["potential_geometry"]["accepted_steps"] == first.record["potential_geometry"]["accepted_steps"]+1
    store = ExecutedWaveStore(tmp_path, snapshot.numerical_digest, 20*1024**2)
    key = store.key("finite-geometry", first.digest)
    store.put(key, first)
    loaded = store.get(key)
    resumed, = observe_potential_geometry_checkpoint(snapshot, geometry, loaded, (end,), maximum_step_m=1e-6,
                                                     kinetic_order=2)
    np.testing.assert_array_equal(resumed.beam.modes[0].plane.amplitude, last.beam.modes[0].plane.amplitude)
    assert resumed.record["potential_geometry"] == last.record["potential_geometry"]


def test_saved_support_budget_cannot_reset_and_inherited_branch_budget_cannot_restart():
    from temsim.physics.potential_support_lineage import capture_inherited_support
    checkpoint, snapshot, geometry = _checkpoint(), _snapshot(), _geometry()
    ledger = NumericalSupportLedger(1e-6, 1e-10)
    checkpoint = replace(checkpoint, record={**thaw_json(checkpoint.record), "snapshot_digest": snapshot.numerical_digest,
                                          "support_ledger": ledger.to_dict()})
    with pytest.raises(ValueError, match="cannot be reset"):
        observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (1.,), support_ledger=NumericalSupportLedger(1e-5, 1e-9))
    prefix = capture_inherited_support(checkpoint, snapshot.numerical_digest)
    branched = replace(checkpoint, record={"snapshot_digest": snapshot.numerical_digest,
        "inherited_potential_support": prefix, "support_prefix_digest": prefix["digest"]})
    with pytest.raises(ValueError, match="disabled after material branching"):
        observe_potential_geometry_checkpoint(snapshot, geometry, branched, (1.,), support_ledger=ledger)
    output, = observe_potential_geometry_checkpoint(snapshot, geometry, branched, (1.000001,), maximum_step_m=1e-6)
    assert output.record["inherited_potential_support"] == prefix
    assert output.record["support_prefix_digest"] == prefix["digest"]


@lru_cache(maxsize=8)
def _finite_plate_run(n, pitch, step):
    # An explicit controlled edge-resolution study. The 1% spectral admission
    # tail is a check only: no Fourier coefficient is removed. This relaxed
    # study policy is NOT the production/default 1e-10 spectral admission.
    plane = _wave(n=n, pitch=pitch, width=4e-7, tilt=(.002, 0.))
    geometry = _geometry(_plate(radius_m=1e-6, thickness_m=2e-6,
        z=1.000002, x_m=2e-7, angle_mrad=5.))
    output, = observe_potential_geometry_checkpoint(_snapshot(), geometry, _checkpoint(plane), (1.000004,),
        maximum_step_m=step, kinetic_order=2, envelope_tail_limit=.01,
        grid_numerics=WaveGridNumerics(automatic_refinement=False), boundary_fraction_limit=1e-6)
    return output


def test_finite_offset_tilted_plate_has_independent_pixel_step_and_window_convergence(record_property):
    grids = [_finite_plate_run(n, 8e-6/n, 2.5e-7) for n in (160, 320, 640)]
    weights = np.array([row.beam.total_weight for row in grids])
    assert abs(weights[2]-weights[1]) < 2e-5
    assert abs(weights[2]-weights[1]) < abs(weights[1]-weights[0])/10
    steps = [_finite_plate_run(320, 2.5e-8, step) for step in (1e-6, 2.5e-7, 1.25e-7)]
    reference = steps[-1].beam.modes[0]
    errors = []
    for result in steps[:-1]:
        mode = result.beam.modes[0]
        np.testing.assert_allclose(mode.plane.origin_m, reference.plane.origin_m, atol=1e-22, rtol=0)
        np.testing.assert_array_equal(mode.plane.basis_m, reference.plane.basis_m)
        np.testing.assert_array_equal(mode.plane.tilt_rad, reference.plane.tilt_rad)
        errors.append(float(np.linalg.norm(mode.plane.amplitude*np.sqrt(mode.weight_per_reference_electron)
            - reference.plane.amplitude*np.sqrt(reference.weight_per_reference_electron))))
    assert errors[1] < errors[0]/2 and errors[1] < 2e-6
    larger = _finite_plate_run(384, 2.5e-8, 2.5e-7)
    assert abs(larger.beam.total_weight-steps[1].beam.total_weight) < 1e-10
    for row in (*grids, *steps, larger):
        report = row.record["potential_geometry"]
        assert report["entry_weight_per_reference"] == pytest.approx(
            row.beam.total_weight+report["physical_loss_per_reference"], abs=5e-14)
        assert report["edge_diffraction_qualified"] is False
        assert not report["normalization_repair"]
    record_property("finite_plate_convergence", json.dumps({
        "pixel_counts": [160, 320, 640], "output_weights": weights.tolist(),
        "step_m": [1e-6, 2.5e-7, 1.25e-7], "absolute_complex_step_errors": errors,
        "window_weight_difference": larger.beam.total_weight-steps[1].beam.total_weight,
        "scope": "controlled finite absorbing plate; explicit 0.01 spectral admission, no spectral filtering; no atomic edge diffraction qualification"}))


def test_native_millimetre_authority_is_accepted_only_by_exact_forward_conversion():
    checkpoint, snapshot = _checkpoint(), _snapshot()
    native_mm = 1599.2000000000003
    exact_m = native_mm*.001
    geometry = _geometry(start=exact_m, stop=exact_m+1e-6)
    checkpoint = replace(checkpoint, plane_z_mm=native_mm)
    first, = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (exact_m,))
    # A material slice preserves its independently authoritative native mm.
    restored_native = replace(first, plane_z_mm=native_mm)
    again, = observe_potential_geometry_checkpoint(snapshot, geometry, restored_native, (exact_m,))
    assert again.record["potential_geometry"]["last_z_m"] == exact_m
    changed = replace(first, plane_z_mm=np.nextafter(native_mm, np.inf))
    if changed.plane_z_mm*.001 != exact_m and changed.plane_z_mm != exact_m*1000:
        with pytest.raises(ValueError, match="native plane"):
            observe_potential_geometry_checkpoint(snapshot, geometry, changed, (exact_m,))


@pytest.mark.parametrize("thickness", [1e-6, 2e-6])
def test_complete_entrance_absorption_continues_without_rounded_guide_retry_loop(thickness):
    geometry = _geometry(_plate(radius_m=0., z=1., thickness_m=thickness))
    checkpoint = _checkpoint()
    output, = observe_potential_geometry_checkpoint(_snapshot(), geometry, checkpoint, (1.000001,))
    assert output.beam.total_weight == 0.
    assert output.beam.modes[0].plane.probability == 0.
    report = output.record["potential_geometry"]
    assert report["last_z_m"] == 1.000001
    assert report["physical_loss_per_reference"] == pytest.approx(checkpoint.beam.total_weight)
    assert len(report["physical_losses"]) == 1
    assert report["accepted_steps"] < 20


def test_rounded_guide_uses_inward_plane_and_preserves_adjacent_requested_planes():
    from temsim.physics.potential_geometry import _bounded_upper
    lower, guide = 1., 2.500000000509317e-7
    # Select a reproducible positive guide whose naive addition rounds out.
    guide = np.nextafter((lower+guide)-lower, 0.)
    assert (lower+guide)-lower > guide
    upper = _bounded_upper(lower, lower+1e-6, guide)
    assert 0 < upper-lower <= guide
    assert upper == np.nextafter(lower+guide, lower)
    neighbour = np.nextafter(lower, np.inf)
    assert _bounded_upper(lower, neighbour, neighbour-lower) == neighbour
    with pytest.raises(ValueError, match="cannot be represented"):
        _bounded_upper(lower, neighbour, (neighbour-lower)/2)


def test_nonzero_wave_crosses_rounded_finite_slab_guides_with_flux_and_raw_phase_intact():
    wave = _wave(n=80, pitch=5e-8, width=1e-7)
    checkpoint = _checkpoint(wave)
    geometry = _geometry(_plate(radius_m=1.9e-6, thickness_m=1e-6, z=1.))
    output, = observe_potential_geometry_checkpoint(_snapshot(), geometry, checkpoint, (1.000001,), kinetic_order=2)
    vacuum, = observe_potential_geometry_checkpoint(_snapshot(), _geometry(), checkpoint, (1.000001,), kinetic_order=2)
    actual, expected = output.beam.modes[0], vacuum.beam.modes[0]
    assert actual.weight_per_reference_electron == pytest.approx(expected.weight_per_reference_electron, abs=2e-14)
    np.testing.assert_allclose(actual.plane.amplitude, expected.plane.amplitude, rtol=0., atol=2e-14)
    assert output.record["potential_geometry"]["last_z_m"] == 1.000001


@pytest.mark.parametrize("stored", [False, True])
def test_invalid_kinetic_history_cannot_enter_geometry_steps(stored):
    checkpoint, options = _checkpoint(), {}
    if stored:
        checkpoint = replace(checkpoint, record={**thaw_json(checkpoint.record),
            "kinetic_phase_ledger": {"scope": "nonempty inherited budget must be rejected"}})
    else:
        options["kinetic_phase_ledger"] = object()
    with pytest.raises(ValueError, match="kinetic phase ledger|KineticPhaseLedger"):
        observe_potential_geometry_checkpoint(_snapshot(), _geometry(), checkpoint, (1.,), **options)


def test_geometry_preserves_one_total_kinetic_budget_across_steps_noop_and_stored_resume(tmp_path):
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    snapshot, geometry, checkpoint = _snapshot(), _geometry(), _checkpoint()
    ledger = KineticPhaseLedger(.01)
    first, = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (1.000001,),
        maximum_step_m=5e-7, kinetic_phase_ledger=ledger)
    saved = KineticPhaseLedger.from_dict(first.record["kinetic_phase_ledger"])
    assert 0 < saved.used_phase_rad < .01
    assert saved.maximum_phase_rad == .01
    assert ledger.used_phase_rad == 0 and ledger.binding is None
    same, = observe_potential_geometry_checkpoint(snapshot, geometry, first, (1.000001,), maximum_step_m=5e-7)
    assert same.digest == first.digest
    store = ExecutedWaveStore(tmp_path, snapshot.numerical_digest, 20*1024**2)
    key = store.key("kinetic-geometry", first.digest)
    store.put(key, first)
    restored = store.get(key)
    continued, = observe_potential_geometry_checkpoint(snapshot, geometry, restored, (1.000002,), maximum_step_m=5e-7)
    direct, = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (1.000002,),
        maximum_step_m=5e-7, kinetic_phase_ledger=ledger)
    final = KineticPhaseLedger.from_dict(continued.record["kinetic_phase_ledger"])
    assert final.events[:len(saved.events)] == saved.events
    assert saved.used_phase_rad < final.used_phase_rad <= final.maximum_phase_rad == .01
    assert final.used_phase_rad == pytest.approx(continued.record["potential_geometry"]["kinetic_phase_bound"], rel=2e-15)
    a, b = continued.beam.modes[0], direct.beam.modes[0]
    np.testing.assert_allclose(a.plane.amplitude*np.sqrt(a.weight_per_reference_electron),
        b.plane.amplitude*np.sqrt(b.weight_per_reference_electron), atol=3e-13, rtol=0.)


def test_geometry_rejects_kinetic_reset_source_energy_order_and_multiple_modes():
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.potential_snapshot import StaticPolynomialGauge
    snapshot, geometry, checkpoint = _snapshot(), _geometry(), _checkpoint()
    first, = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (1.,),
        kinetic_phase_ledger=KineticPhaseLedger(.01))
    for ledger in (None, KineticPhaseLedger(.01), KineticPhaseLedger(.02)):
        with pytest.raises(ValueError, match="cannot be reset"):
            observe_potential_geometry_checkpoint(snapshot, geometry, first, (1.000001,), kinetic_phase_ledger=ledger)
    with pytest.raises(ValueError, match="source, energy or operator order changed"):
        observe_potential_geometry_checkpoint(_snapshot(gauge=StaticPolynomialGauge(1.)), geometry, first, (1.000001,))
    changed = replace(first, beam=BeamState((replace(first.beam.modes[0], energy_kev=299.),), TIP_REFERENCE))
    with pytest.raises(ValueError, match="source, energy or operator order changed"):
        observe_potential_geometry_checkpoint(snapshot, geometry, changed, (1.000001,))
    with pytest.raises(ValueError, match="operator policy changed"):
        observe_potential_geometry_checkpoint(snapshot, geometry, first, (1.000001,), kinetic_order=2)
    with pytest.raises(ValueError, match="geometry changed"):
        observe_potential_geometry_checkpoint(snapshot, _geometry(_plate()), first, (1.000001,))
    extra = replace(checkpoint.beam.modes[0], mode_id="other", weight_per_reference_electron=.1)
    multiple = replace(checkpoint, beam=BeamState((*checkpoint.beam.modes, extra), TIP_REFERENCE))
    with pytest.raises(ValueError, match="requires one mode"):
        observe_potential_geometry_checkpoint(snapshot, geometry, multiple, (1.,), kinetic_phase_ledger=KineticPhaseLedger(.01))


def test_discarded_geometry_step_does_not_spend_kinetic_budget(monkeypatch):
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    import temsim.physics.potential_geometry as module
    original, rejected = module._geometry_step_limit, []
    def once(checkpoint, geometry, lower, upper, *args, include_zero=False):
        if include_zero and not rejected:
            rejected.append((lower, upper))
            return (upper-lower)/2
        return original(checkpoint, geometry, lower, upper, *args, include_zero=include_zero)
    monkeypatch.setattr(module, "_geometry_step_limit", once)
    output, = observe_potential_geometry_checkpoint(_snapshot(), _geometry(), _checkpoint(), (1.000001,),
        maximum_step_m=1e-6, kinetic_phase_ledger=KineticPhaseLedger(.01))
    ledger = KineticPhaseLedger.from_dict(output.record["kinetic_phase_ledger"])
    assert rejected
    assert len(ledger.events) == output.record["potential_geometry"]["accepted_steps"]
    assert ledger.events[0]["upper_z_m"] < rejected[0][1]
    assert ledger.used_phase_rad == pytest.approx(output.record["potential_geometry"]["kinetic_phase_bound"], rel=2e-15)
    assert ledger.events[-1]["upper_z_m"] == 1.000001


def test_fully_absorbed_wave_continues_zero_kinetic_expenditure():
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    geometry = _geometry(_plate(radius_m=0., z=1.))
    output, = observe_potential_geometry_checkpoint(_snapshot(), geometry, _checkpoint(), (1.000001,),
        kinetic_phase_ledger=KineticPhaseLedger(.01))
    ledger = KineticPhaseLedger.from_dict(output.record["kinetic_phase_ledger"])
    assert output.beam.total_weight == 0
    assert ledger.used_phase_rad == 0
    resumed, = observe_potential_geometry_checkpoint(_snapshot(), geometry, output, (1.000002,))
    assert resumed.beam.total_weight == 0
    assert KineticPhaseLedger.from_dict(resumed.record["kinetic_phase_ledger"]).used_phase_rad == 0


def test_potential_prefix_kinetic_history_continues_through_finite_geometry_without_changing_wave():
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.potential_wave_observation import observe_potential_checkpoint
    snapshot, checkpoint = _snapshot(), _checkpoint()
    prefix, = observe_potential_checkpoint(snapshot, checkpoint, (1.000001,), maximum_step_m=5e-7,
        kinetic_phase_ledger=KineticPhaseLedger(.01))
    prior = KineticPhaseLedger.from_dict(prefix.record["kinetic_phase_ledger"])
    geometry = _geometry(_plate(radius_m=1.9e-6, z=1.000003), start=1.000001)
    output, = observe_potential_geometry_checkpoint(snapshot, geometry, prefix, (1.000002,), maximum_step_m=5e-7)
    final = KineticPhaseLedger.from_dict(output.record["kinetic_phase_ledger"])
    assert final.events[:len(prior.events)] == prior.events
    assert prior.used_phase_rad < final.used_phase_rad < .01
    assert final.used_phase_rad == pytest.approx(prior.used_phase_rad+
        output.record["potential_geometry"]["kinetic_phase_bound"], rel=2e-15)
    # This budget policy changes admission/accounting only, not the operator.
    plain_prefix = replace(prefix, record={key: value for key, value in thaw_json(prefix.record).items()
                                          if key != "kinetic_phase_ledger"})
    plain, = observe_potential_geometry_checkpoint(snapshot, geometry, plain_prefix, (1.000002,), maximum_step_m=5e-7)
    np.testing.assert_array_equal(output.beam.modes[0].plane.amplitude, plain.beam.modes[0].plane.amplitude)
    assert output.beam.total_weight == plain.beam.total_weight


def test_kinetic_total_exhaustion_rejects_without_spending_entrance_ledger():
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    checkpoint, ledger = _checkpoint(), KineticPhaseLedger(1e-30)
    identity = checkpoint.digest
    with pytest.raises(ValueError, match="kinetic truncation.*phase budget"):
        observe_potential_geometry_checkpoint(_snapshot(), _geometry(), checkpoint, (1.000001,),
            kinetic_phase_ledger=ledger)
    assert ledger.events == () and ledger.binding is None and ledger.used_phase_rad == 0
    assert checkpoint.digest == identity


def test_geometry_retains_and_validates_material_kinetic_lineage():
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.potential_material_phase_budget import capture_material_kinetic, validate_material_kinetic_record
    from temsim.physics.potential_wave_observation import observe_potential_checkpoint
    snapshot, checkpoint = _snapshot(), _checkpoint()
    prefix, = observe_potential_checkpoint(snapshot, checkpoint, (1.,), kinetic_phase_ledger=KineticPhaseLedger(.01))
    context = capture_material_kinetic(prefix, snapshot.numerical_digest)
    entered = replace(prefix, record={**thaw_json(prefix.record), **context})
    output, = observe_potential_geometry_checkpoint(snapshot, _geometry(), entered, (1.000001,))
    validate_material_kinetic_record(output.record)
    assert output.record["material_kinetic_history"] == entered.record["material_kinetic_history"]
    assert output.record["material_kinetic_prefix_digest"] == entered.record["material_kinetic_prefix_digest"]
    missing = thaw_json(entered.record)
    missing.pop("material_kinetic_history")
    with pytest.raises(ValueError, match="lost its common prefix"):
        observe_potential_geometry_checkpoint(snapshot, _geometry(), replace(entered, record=missing), (1.,))
    increased = thaw_json(entered.record)
    increased["kinetic_phase_ledger"] = KineticPhaseLedger(.02).require(snapshot.numerical_digest, 300000., 4, 1.).to_dict()
    with pytest.raises(ValueError, match="reset or increased"):
        observe_potential_geometry_checkpoint(snapshot, _geometry(), replace(entered, record=increased), (1.,))


def test_kinetic_entrance_recovers_exact_native_plane_without_merging_neighbour():
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    snapshot, checkpoint = _snapshot(), replace(_checkpoint(), plane_z_mm=700.)
    ledger = KineticPhaseLedger(.01).require(snapshot.numerical_digest, 300000., 4, .7)
    checkpoint = replace(checkpoint, record={**thaw_json(checkpoint.record),
        "snapshot_digest": snapshot.numerical_digest, "kinetic_phase_ledger": ledger.to_dict()})
    geometry = _geometry(start=.69, stop=.71)
    assert checkpoint.plane_z_mm*.001 != ledger.last_z_m
    same, following = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint,
        (.7, np.nextafter(.7, np.inf)))
    assert same.record["potential_geometry"]["last_z_m"] == .7
    continued = KineticPhaseLedger.from_dict(following.record["kinetic_phase_ledger"])
    assert len(continued.events) == 1
    assert continued.events[0]["lower_z_m"] == .7
    assert continued.last_z_m == np.nextafter(.7, np.inf)
    changed = replace(checkpoint, plane_z_mm=np.nextafter(700., np.inf))
    with pytest.raises(ValueError, match="exact kinetic ledger plane"):
        observe_potential_geometry_checkpoint(snapshot, geometry, changed, (.700001,))


def _material_native_checkpoint():
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.potential_material_phase_budget import capture_material_kinetic, stamp_material_kinetic_plane
    snapshot, native = _snapshot(), 1599.2
    exact = native*.001
    ledger = KineticPhaseLedger(.01).require(snapshot.numerical_digest, 300000., 4, exact)
    canonical = replace(_checkpoint(), plane_z_mm=exact*1000., record={
        "snapshot_digest": snapshot.numerical_digest, "kinetic_phase_ledger": ledger.to_dict()})
    context = capture_material_kinetic(canonical, snapshot.numerical_digest)
    context = stamp_material_kinetic_plane({**thaw_json(canonical.record), **context}, native)
    checkpoint = replace(canonical, plane_z_mm=native, record=context)
    return snapshot, _geometry(start=exact, stop=exact+4e-6), checkpoint, exact


def test_material_native_receipt_has_priority_over_canonical_roundtrip_and_rejects_none():
    snapshot, geometry, checkpoint, exact = _material_native_checkpoint()
    assert checkpoint.plane_z_mm != exact*1000.
    assert np.nextafter(checkpoint.plane_z_mm, np.inf) == exact*1000.
    same, = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (exact,))
    assert same.record["material_kinetic_native_plane"] == checkpoint.record["material_kinetic_native_plane"]
    for value in (checkpoint, same):
        changed = replace(value, plane_z_mm=exact*1000.)
        with pytest.raises(ValueError, match="exact kinetic ledger plane"):
            observe_potential_geometry_checkpoint(snapshot, geometry, changed, (exact,))
        null = replace(value, record={**thaw_json(value.record), "material_kinetic_native_plane": None})
        with pytest.raises(ValueError):
            observe_potential_geometry_checkpoint(snapshot, geometry, null, (exact,))


def test_material_to_geometry_removes_old_native_receipt_only_after_accepted_forward_step():
    snapshot, geometry, checkpoint, exact = _material_native_checkpoint()
    first, second = exact+1e-6, exact+2e-6
    options = {"maximum_step_m": 5e-7}
    before, middle, complete = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint,
        (exact, first, second), **options)
    assert "material_kinetic_native_plane" in before.record
    assert "material_kinetic_native_plane" not in middle.record
    assert "material_kinetic_native_plane" not in complete.record
    assert complete.record["material_kinetic_history"] == checkpoint.record["material_kinetic_history"]
    continued, = observe_potential_geometry_checkpoint(snapshot, geometry, middle, (second,), **options)
    direct, = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (second,), **options)
    for other in (continued, direct):
        np.testing.assert_allclose(other.beam.modes[0].plane.amplitude, complete.beam.modes[0].plane.amplitude,
                                   rtol=0., atol=3e-13)
        assert other.beam.total_weight == pytest.approx(complete.beam.total_weight, abs=3e-14)
    assert continued.record["kinetic_phase_ledger"] == complete.record["kinetic_phase_ledger"]


def test_existing_geometry_kinetic_checkpoint_rejects_other_native_mm_that_folds_to_same_si():
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    snapshot = _snapshot()
    native, exact = 1599.2, 1599.2*.001
    canonical = exact*1000.
    assert native != canonical and native*.001 == exact
    checkpoint = replace(_checkpoint(), plane_z_mm=canonical)
    geometry = _geometry(start=exact, stop=exact+2e-6)
    output, = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (exact,),
        kinetic_phase_ledger=KineticPhaseLedger(.01))
    assert "potential_geometry" in output.record and "material_kinetic_native_plane" not in output.record
    with pytest.raises(ValueError, match="exact kinetic ledger plane"):
        observe_potential_geometry_checkpoint(snapshot, geometry, replace(output, plane_z_mm=native), (exact,))
