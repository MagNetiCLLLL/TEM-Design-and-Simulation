"""An explicit monochromatic Tip preview is not a relabelled cached mode."""
from dataclasses import asdict, replace
import math

import numpy as np
import pytest

from temsim.instrument_snapshot import encode_instrument
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_coherence import (
    TipCoherence, TipWaveNumerics, generate_tip_emission, generate_tip_boundary_emission,
    tip_energy_samples,
)
from temsim.physics.tip_gun_wave import build_tip_gun_checkpoint


def _gun(*, driven=True, incoherent=0.):
    gun = FieldEmissionGun()
    gun.emitter.coherence = TipCoherence(
        boundary_model="driven_gaussian_schell" if driven else "forward_gaussian_schell",
        incoherent_angle_rms_mrad=incoherent, offset_x_nm=1., tilt_y_mrad=.2)
    if not driven:
        # Explicitly admissible forward/paraxial source fixture. Production
        # defaults are checked independently with their actual driven model.
        gun.emitter.emission_energy_ev = 4.
        gun.emitter.minimum_kinetic_energy_ev = 3.
        gun.emitter.energy_spread_fwhm_ev = .4
    return gun


@pytest.mark.parametrize("generate", [generate_tip_emission, generate_tip_boundary_emission])
def test_actual_nominal_point_three_ev_is_new_source_with_original_parameters(generate):
    gun = _gun()
    before = gun.to_dict()
    many = generate(gun, TipWaveNumerics(energy_samples=9))
    assert gun.emitter.emission_energy_ev == .3
    assert all(row["energy_ev"] != .3 for row in many.record["energy_modes"])
    preview = generate(gun, TipWaveNumerics(energy_samples=1), nominal_energy_preview=True)
    mode, = tuple(preview.modes())
    assert preview.record["energy_modes"][0]["energy_ev"] == .3
    assert mode.energy_kev == .3*1e-3
    assert mode.weight_per_reference_electron == 1.
    assert mode.axial_reference.flight_time_s == 0.
    assert preview.record["plane_z_mm"] == 0.
    assert preview.parameters == many.parameters
    assert preview.reference_current_a == gun.emitter.emitted_current_a
    assert preview.record["nominal_energy_preview"] is True
    assert preview.record["configured_energy_spread_integrated"] is False
    assert preview.record["high_accuracy_eligible"] is False
    assert preview.record["configured_energy_spread_fwhm_ev"] == gun.emitter.energy_spread_fwhm_ev
    assert preview.digest != many.digest
    assert mode.mode_id not in {m.mode_id for m in many.modes()}
    assert gun.to_dict() == before


@pytest.mark.parametrize("driven", [False, True])
def test_preview_retains_original_spatial_modes_weights_phase_and_omission(driven):
    gun = _gun(driven=driven, incoherent=18.)
    numerics = TipWaveNumerics(energy_samples=1, grid_pixels=256, mode_tail_tolerance=1e-9)
    preview = generate_tip_emission(gun, numerics, nominal_energy_preview=True)
    # Independent original-generator reference at exactly the same energy.
    # Only this test copy has zero spread; the production preview must retain
    # the actual nonzero configured spread and all its physical input hashes.
    from copy import deepcopy
    monochromatic_reference = deepcopy(gun)
    monochromatic_reference.emitter.energy_spread_fwhm_ev = 0.
    reference = generate_tip_emission(monochromatic_reference, numerics)
    modes = tuple(preview.modes())
    expected = tuple(reference.modes())
    assert len(modes) == len(expected)
    assert preview.record["energy_modes"] == reference.record["energy_modes"]
    for actual, original in zip(modes, expected, strict=True):
        assert actual.weight_per_reference_electron == original.weight_per_reference_electron
        assert actual.energy_kev == original.energy_kev
        for key in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            np.testing.assert_array_equal(getattr(actual.plane, key), getattr(original.plane, key))
    assert math.fsum(mode.weight_per_reference_electron for mode in modes) == pytest.approx(
        1-preview.record["omitted_probability"], abs=1e-13)
    assert preview.parameters["energy_spread_fwhm_ev"] == gun.emitter.energy_spread_fwhm_ev > 0.


@pytest.mark.parametrize("generate", [generate_tip_emission, generate_tip_boundary_emission])
def test_default_false_has_identical_record_digest_and_retains_one_sample_rejection(generate):
    gun = _gun()
    original = generate(gun)
    explicit = generate(gun, nominal_energy_preview=False)
    assert original.record == explicit.record
    assert original.digest == explicit.digest
    assert "nominal_energy_preview" not in original.record
    with pytest.raises(ValueError, match="discard.*energy spread"):
        generate(gun, TipWaveNumerics(energy_samples=1))
    with pytest.raises(ValueError, match="discard.*energy spread"):
        tip_energy_samples(gun.emitter, 1)
    assert asdict(TipWaveNumerics()) == {"grid_pixels": 128, "energy_samples": 9,
        "mode_tail_tolerance": 1e-6, "maximum_modes": 1024}


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
@pytest.mark.parametrize("generate", [generate_tip_emission, generate_tip_boundary_emission, build_tip_gun_checkpoint])
def test_preview_flag_requires_an_explicit_boolean(flag, generate):
    with pytest.raises(ValueError, match="explicit boolean"):
        generate(_gun(), nominal_energy_preview=flag)


@pytest.mark.parametrize("count", [2, 9])
@pytest.mark.parametrize("generate", [generate_tip_emission, generate_tip_boundary_emission, build_tip_gun_checkpoint])
def test_preview_requires_exactly_one_new_energy_sample(count, generate):
    kwargs = {"source_numerics" if generate is build_tip_gun_checkpoint else "numerics":
              TipWaveNumerics(energy_samples=count), "nominal_energy_preview": True}
    with pytest.raises(ValueError, match="exactly one source energy sample"):
        generate(_gun(), **kwargs)


def test_preview_does_not_waive_source_domain_or_boundary_gates():
    gun = _gun()
    gun.emitter.coherence = replace(gun.emitter.coherence, boundary_model="forward_gaussian_schell")
    from temsim.optics.electron_gun.tip_source_domain import TipSourceDomainError
    with pytest.raises(TipSourceDomainError):
        generate_tip_emission(gun, TipWaveNumerics(energy_samples=1), nominal_energy_preview=True)
    gun.emitter.curvature_nm_inv = .001
    with pytest.raises(ValueError, match="Continuous curvature"):
        generate_tip_boundary_emission(gun, TipWaveNumerics(energy_samples=1), nominal_energy_preview=True)


def test_gun_builder_passes_nominal_source_before_actual_boundary_solver_without_editing_fields(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.physics import accelerating_tip_boundary, tip_gun_wave
    state = default_state()
    state.electron_gun.emitter.surface_model = None
    state.electron_gun.emitter.coherence = TipCoherence(boundary_model="driven_gaussian_schell")
    before = encode_instrument(state)
    sources = []
    original = tip_gun_wave.generate_tip_boundary_emission
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        sources.append(result)
        return result
    class BoundaryReached(Exception):
        pass
    def stop_before_transport(**kwargs):
        raise BoundaryReached
    monkeypatch.setattr(tip_gun_wave, "generate_tip_boundary_emission", capture)
    monkeypatch.setattr(accelerating_tip_boundary, "AcceleratingTipNumerics", stop_before_transport)
    with pytest.raises(BoundaryReached):
        build_tip_gun_checkpoint(state.electron_gun, source_numerics=TipWaveNumerics(energy_samples=1),
                                 nominal_energy_preview=True, _column_state=state)
    assert len(sources) == 1
    assert sources[0].record["nominal_energy_ev"] == state.electron_gun.emitter.emission_energy_ev == .3
    assert sources[0].parameters["energy_spread_fwhm_ev"] == state.electron_gun.emitter.energy_spread_fwhm_ev
    assert encode_instrument(state) == before


def test_nominal_source_identity_changes_when_configured_spread_changes_without_replacing_parameters():
    gun = _gun()
    numerics = TipWaveNumerics(energy_samples=1)
    first = generate_tip_emission(gun, numerics, nominal_energy_preview=True)
    gun.emitter.energy_spread_fwhm_ev *= .9
    second = generate_tip_emission(gun, numerics, nominal_energy_preview=True)
    assert first.record["energy_modes"] == second.record["energy_modes"]
    assert first.digest != second.digest
    assert first.parameters["energy_spread_fwhm_ev"] != second.parameters["energy_spread_fwhm_ev"]


def test_nominal_energy_is_distinct_from_configured_energy_spread_fwhm():
    gun = _gun()
    gun.emitter.emission_energy_ev = 2.
    gun.emitter.energy_spread_fwhm_ev = .3
    before = gun.to_dict()
    emission = generate_tip_emission(gun, TipWaveNumerics(energy_samples=1),
                                    nominal_energy_preview=True)
    mode, = tuple(emission.modes())
    assert emission.record["nominal_energy_ev"] == 2.
    assert emission.record["energy_modes"][0]["energy_ev"] == 2.
    assert mode.energy_kev == 2.*1e-3
    assert emission.record["configured_energy_spread_fwhm_ev"] == .3
    assert emission.parameters["emission_energy_ev"] == 2.
    assert emission.parameters["energy_spread_fwhm_ev"] == .3
    assert emission.record["configured_energy_spread_integrated"] is False
    assert mode.weight_per_reference_electron == 1.
    assert gun.to_dict() == before


def test_actual_nominal_tip_executes_robin_acceleration_with_captured_electric_field():
    from temsim.optics.column import default_state
    from temsim.physics.accelerating_tip_boundary import execute_accelerating_tip
    from temsim.physics.instrument_electric import capture_instrument_electric_field
    state = default_state()
    state.electron_gun.emitter.surface_model = None
    state.electron_gun.emitter.coherence = TipCoherence(boundary_model="driven_gaussian_schell")
    before = encode_instrument(state)
    emission = generate_tip_boundary_emission(state.electron_gun,
        TipWaveNumerics(energy_samples=1), nominal_energy_preview=True)
    electric = capture_instrument_electric_field(state)
    result = execute_accelerating_tip(state, emission, electric)
    mode, = result.modes
    row, = result.record["modes"]
    assert result.plane_z_mm == pytest.approx(.004)
    assert row["tip_energy_ev"] == state.electron_gun.emitter.emission_energy_ev == .3
    assert row["endpoint_axial_energy_ev"] > row["tip_energy_ev"]
    assert row["balance_residual"] < 1e-8
    assert row["layers_with_evanescent_channels"] > 0
    assert mode.weight_per_reference_electron == pytest.approx(row["transmitted_fraction"], abs=1e-10)
    assert mode.axial_reference.flight_time_s > 0.
    assert result.record["tip_emission_id"] == emission.digest
    assert result.record["tip_emission"]["nominal_energy_preview"] is True
    assert result.record["tip_emission"]["high_accuracy_eligible"] is False
    assert result.record["electric_numerical_identity"] == electric.numerical_identity
    assert np.isfinite(mode.plane.amplitude).all()
    assert np.isfinite(result.spectral_derivatives_per_m[0]).all()
    assert not result.spectral_derivatives_per_m[0].flags.writeable
    assert encode_instrument(state) == before


@pytest.mark.parametrize("mutation", ["nominal", "parameters", "tag"])
def test_accelerating_tip_still_rejects_forged_nominal_source_before_field_capture(monkeypatch, mutation):
    from types import SimpleNamespace
    from temsim.physics.accelerating_tip_boundary import execute_accelerating_tip
    gun = _gun()
    emission = generate_tip_boundary_emission(gun, TipWaveNumerics(energy_samples=1),
                                             nominal_energy_preview=True)
    if mutation == "nominal":
        emission = replace(emission, record={**dict(emission.record), "nominal_energy_ev": 2.})
    elif mutation == "parameters":
        emission = replace(emission, parameters={**dict(emission.parameters), "energy_spread_fwhm_ev": .9})
    else:
        emission = replace(emission, record={**dict(emission.record), "nominal_energy_preview": "true"})
    monkeypatch.setattr("temsim.physics.instrument_electric.capture_instrument_electric_field",
        lambda *args: pytest.fail("Forged source reached the actual field capture"))
    with pytest.raises(ValueError, match="captured physical source|discard.*energy spread"):
        execute_accelerating_tip(SimpleNamespace(electron_gun=gun), emission, object())
