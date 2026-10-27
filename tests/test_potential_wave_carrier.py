"""Carrier benchmark retains complete phase and differential magnetic action."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, hbar, m_e

from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_snapshot import (
    ElectromagneticPotentialSnapshot, PotentialContribution, StaticPolynomialGauge,
)
from temsim.physics.potential_wave_carrier import (
    propagate_carrier_potential, _apply_residual, _coefficients, _reference,
)


ENERGY_EV = 200000.
MOMENTUM = np.sqrt(ENERGY_EV*e*(ENERGY_EV*e+2*m_e*c*c))/c
WAVELENGTH = 2*np.pi*hbar/MOMENTUM


def _wave(n=32, pitch=1e-7, width=3e-7, curvature=0., tilt=(0., 0.)):
    xy = (np.arange(n)-n//2)*pitch
    x, y = np.meshgrid(xy, xy)
    amplitude = np.exp(-(x*x+y*y)/(4*width**2))*(1+.1j*x/width)*np.exp(.37j)
    amplitude /= np.sqrt(np.sum(abs(amplitude)**2))
    return PlaneWave(amplitude, np.eye(2)*pitch, np.zeros(2), np.eye(2)*curvature, np.array(tilt))


def _snapshot(*providers, gauge=None):
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    return ElectromagneticPotentialSnapshot(scene, tuple(PotentialContribution(f"s{i}", (f"s{i}",), provider,
        "magnetic") for i, provider in enumerate(providers)), gauge or StaticPolynomialGauge())


def _round(strength=.2, order=3):
    native = FrozenAnalyticField("lens", ((strength, 0., .01),), CoordinateRegistration(), (-50., 50.), .003)
    return HighOrderRoundLensPotential(native, radial_order=order, axial_domain="gaussian_all_z")


def _multipole(*, dynamic=False):
    native = FrozenPosedMultipole("hex", "hex", CoordinateRegistration(),
        ((1, 0, -2e-4), (0, 1, 1e-4), (3, 0, 1e6), (1, 2, -3e6)),
        "uniform", 0., .01, (-50., 50.), .003, MOMENTUM, dynamic=dynamic)
    return AnalyticPotentialDomain(AnalyticMagneticPotential(native))


def _full_phase(wave):
    """Independent materialisation for deliberately well-resolved comparisons."""
    xy = wave.coordinates_m()-wave.origin_m[:, None, None]
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    phase = (.5*np.einsum("iyx,ij,jyx->yx", xy, curvature, xy)+np.einsum("i,iyx->yx", tilt, xy))*2*np.pi/WAVELENGTH
    return wave.amplitude*np.exp(1j*phase)


def test_entrance_preparation_preserves_every_non_gaussian_fourier_coefficient(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    wave = _wave(n=128, pitch=1e-7, width=8e-7, curvature=.2, tilt=(2e-7, -1e-7))
    indices = np.arange(128)-64
    wave = replace(wave, amplitude=wave.amplitude*(1+.17*np.exp(2j*np.pi*.421875*indices[None, :])
                                                    +.09j*np.exp(-2j*np.pi*.4140625*indices[:, None])))
    original = wave.amplitude.copy()
    captured = []
    residual = carrier._residual
    def inspect(snapshot, candidate, *args, **kwargs):
        captured.append(candidate)
        return residual(snapshot, candidate, *args, **kwargs)
    monkeypatch.setattr(carrier, "_residual", inspect)
    distance = 1e-9
    output, record = propagate_carrier_potential(_snapshot(), wave, (0., distance),
        energy_ev=ENERGY_EV, kinetic_order=2)
    prepared = captured[0]
    assert prepared.amplitude.shape[0] > wave.amplitude.shape[0]
    spectrum = np.fft.fftshift(np.fft.fft2(prepared.amplitude, norm="ortho"))
    old_spectrum = np.fft.fftshift(np.fft.fft2(wave.amplitude, norm="ortho"))
    first_y, first_x = np.array(prepared.amplitude.shape)//2-np.array(wave.amplitude.shape)//2
    np.testing.assert_allclose(spectrum[first_y:first_y+128, first_x:first_x+128],
                               old_spectrum, rtol=2e-13, atol=2e-16)
    outside = spectrum.copy(); outside[first_y:first_y+128, first_x:first_x+128] = 0.
    assert np.linalg.norm(outside) < 1e-14
    for name in ("origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(prepared, name), getattr(wave, name))
    np.testing.assert_allclose(prepared.basis_m@np.diag(prepared.amplitude.shape[::-1]),
                               wave.basis_m@np.diag(wave.amplitude.shape[::-1]), atol=1e-20)
    np.testing.assert_array_equal(wave.amplitude, original)
    assert prepared.probability == pytest.approx(wave.probability, rel=2e-14)
    assert output.probability == pytest.approx(wave.probability, rel=2e-13)
    matrix = np.eye(4); matrix[:2, 2:] = np.eye(2)*distance
    path = CanonicalPath(1e-3); path.append(matrix)
    expected = propagate_plane_wave(prepared, matrix, np.zeros(4), WAVELENGTH, **path.phase_kwargs())
    np.testing.assert_allclose(output.amplitude, expected.amplitude, rtol=2e-12, atol=2e-14)
    rows = [row for row in record["canonical_sampling"] if row.get("stage") == "entrance Fourier operator preparation"]
    assert rows and all("not established" in row["continuous_source_convergence"] for row in rows)
    assert record["normalization_applied"] is False


def _entrance_with_hidden_interpolant_edges():
    wave = _wave(n=64)
    # Every original edge is resolved, but the small retained Nyquist packet
    # exposes weak nonzero edge values between coarse samples after embedding.
    return replace(wave, amplitude=wave.amplitude*(1+.001*(-1.)**np.arange(64)[None, :]))


@pytest.mark.parametrize("limits", [None, (0., 0.), (.01, 1e-12)])
def test_fourier_entrance_edges_require_an_explicit_sufficient_projection_budget(limits):
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_wave_carrier import OccupiedCarrierBoundaryError
    source = _entrance_with_hidden_interpolant_edges()
    original = source.amplitude.copy()
    ledger = None if limits is None else NumericalSupportLedger(*limits)
    with pytest.raises(OccupiedCarrierBoundaryError):
        propagate_carrier_potential(_snapshot(), source, (0., 1e-12), energy_ev=ENERGY_EV,
                                    kinetic_order=2, support_ledger=ledger)
    np.testing.assert_array_equal(source.amplitude, original)
    assert ledger is None or ledger.events == ()


def test_budgeted_fourier_entrance_taper_records_loss_and_preserves_free_propagation():
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.wave_grid import refine_plane_wave, WaveGridNumerics
    source = _entrance_with_hidden_interpolant_edges()
    original = source.amplitude.copy()
    ledger = NumericalSupportLedger(.01, 5e-7)
    output, record = propagate_carrier_potential(_snapshot(), source, (0., 1e-12),
        energy_ev=ENERGY_EV, kinetic_order=2, support_ledger=ledger)
    spent = NumericalSupportLedger.from_dict(record["support_ledger"])
    assert len(spent.events) == 1 and spent.events[0]["kind"] == "numerical_boundary_taper"
    assert spent.used_probability > 0 and spent.used_l2 > 0
    assert source.probability-output.probability == pytest.approx(spent.used_probability, abs=1e-15)
    np.testing.assert_array_equal(source.amplitude, original)
    assert not record["normalization_applied"]
    prepared = refine_plane_wave(source, tuple(spent.events[0]["from_shape"]), numerics=WaveGridNumerics())
    prepared, _ = ledger.taper_boundary(prepared, z_m=0.)
    matrix = np.eye(4); matrix[:2, 2:] = np.eye(2)*1e-12
    path = CanonicalPath(1e-3); path.append(matrix)
    expected = propagate_plane_wave(prepared, matrix, np.zeros(4), WAVELENGTH,
                                   **path.phase_kwargs(), grid_numerics=WaveGridNumerics())
    np.testing.assert_allclose(output.basis_m, expected.basis_m, rtol=1e-13, atol=1e-21)
    np.testing.assert_allclose(output.amplitude, expected.amplitude, rtol=2e-12, atol=2e-14)


def test_budget_cannot_hide_occupied_original_entrance_or_unknown_field_domain():
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_wave_carrier import OccupiedCarrierBoundaryError
    source = _wave(n=64)
    amplitude = source.amplitude.copy(); amplitude[0, 0] = .1
    with pytest.raises(OccupiedCarrierBoundaryError):
        propagate_carrier_potential(_snapshot(), replace(source, amplitude=amplitude), (0., 1e-12),
            energy_ev=ENERGY_EV, kinetic_order=2, support_ledger=NumericalSupportLedger(.1, .1))
    narrow = FrozenAnalyticField("narrow", ((.2, 0., .01),), CoordinateRegistration(),
                                (-50., 50.), 1e-8)
    with pytest.raises(ValueError, match="domain|outside|radius|radial"):
        propagate_carrier_potential(_snapshot(HighOrderRoundLensPotential(narrow, radial_order=3)),
            _entrance_with_hidden_interpolant_edges(), (0., 1e-12), energy_ev=ENERGY_EV,
            kinetic_order=2, support_ledger=NumericalSupportLedger(.01, 5e-7))


def test_post_canonical_projection_uses_its_own_resource_budget_and_records_full_complex_loss():
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_wave_device import RESIDUAL_HOST_PREPARATION_RESERVE
    from temsim.physics.wave_grid import WaveGridNumerics, WaveMemoryBudgetError
    source = _wave(n=64)
    common = {"energy_ev": ENERGY_EV, "kinetic_order": 2,
              "support_ledger": NumericalSupportLedger(.01, 5e-7)}
    full, _ = propagate_carrier_potential(_snapshot(), source, (0., .05),
        grid_numerics=WaveGridNumerics(angular_domain_padding="minimum_full_band", maximum_pixels=512,
            maximum_working_bytes=RESIDUAL_HOST_PREPARATION_RESERVE+8*1024**2),
        support_projection_policy="step_start_and_retry", **common)
    # The minimal canonical domain is smaller than the old doubled domain.
    # Put the budget strictly between the actual entrance and unprojected
    # residual work spaces, without tying this ownership check to a shape.
    small = WaveGridNumerics(angular_domain_padding="minimum_full_band", maximum_pixels=512,
        maximum_working_bytes=RESIDUAL_HOST_PREPARATION_RESERVE
            +512*(source.amplitude.size+full.amplitude.size)//2)
    with pytest.raises(WaveMemoryBudgetError):
        propagate_carrier_potential(_snapshot(), source, (0., .05),
            grid_numerics=small, support_projection_policy="step_start_and_retry", **common)
    compact, record = propagate_carrier_potential(_snapshot(), source, (0., .05),
        grid_numerics=small, support_projection_policy="step_boundaries_and_retry", **common)
    ledger = NumericalSupportLedger.from_dict(record["support_ledger"])
    assert len(ledger.events) == 2 and ledger.events[-1]["z_m"] == .05
    assert full.amplitude.shape == tuple(ledger.events[-1]["from_shape"])
    assert compact.amplitude.shape == tuple(ledger.events[-1]["to_shape"])
    assert compact.amplitude.size < full.amplitude.size
    for key in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(full, key), getattr(compact, key))
    (y0, y1), (x0, x1) = ledger.events[-1]["crop_indices_yx"]
    embedded = np.zeros_like(full.amplitude)
    embedded[y0:y1, x0:x1] = compact.amplitude
    assert np.linalg.norm(full.amplitude-embedded) == pytest.approx(
        ledger.events[-1]["absolute_l2_error"], rel=2e-9)
    assert source.probability-compact.probability == pytest.approx(ledger.used_probability, abs=1e-15)
    assert not record["normalization_applied"]
    assert record["residual_steps"][0]["unprojected_canonical_output_radius_m"] > (
        np.max(np.linalg.norm(compact.coordinates_m(), axis=0)))


def test_post_canonical_projection_cannot_hide_the_full_output_field_domain(monkeypatch):
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_wave_device import RESIDUAL_HOST_PREPARATION_RESERVE
    from temsim.physics.wave_grid import WaveGridNumerics
    import temsim.physics.potential_wave_carrier as kernel
    source = _wave(n=64)
    full, _ = propagate_carrier_potential(_snapshot(), source, (0., .05), energy_ev=ENERGY_EV,
        kinetic_order=2, support_ledger=NumericalSupportLedger(.01, 5e-7),
        support_projection_policy="step_start_and_retry",
        grid_numerics=WaveGridNumerics(angular_domain_padding="minimum_full_band"))
    # Admit the original entrance but exclude part of the actually executed
    # canonical output. Numerical cropping must not conceal that unknown field.
    radius = .5*(np.max(np.linalg.norm(source.coordinates_m(), axis=0))
                  +np.max(np.linalg.norm(full.coordinates_m(), axis=0)))
    field = FrozenAnalyticField("zero-known-domain", ((0., 0., 1.),), CoordinateRegistration(),
                                (-1000., 1000.), radius)
    snapshot = _snapshot(HighOrderRoundLensPotential(field, radial_order=3, axial_domain="gaussian_all_z"))
    completed = []
    original = kernel.propagate_plane_wave
    def execute(*args, **kwargs):
        output = original(*args, **kwargs)
        completed.append(output.amplitude.shape)
        return output
    monkeypatch.setattr(kernel, "propagate_plane_wave", execute)
    with pytest.raises(ValueError, match="domain"):
        propagate_carrier_potential(snapshot, source, (0., .05), energy_ev=ENERGY_EV,
            kinetic_order=2, support_ledger=NumericalSupportLedger(.01, 5e-7),
            support_projection_policy="step_boundaries_and_retry",
            grid_numerics=WaveGridNumerics(angular_domain_padding="minimum_full_band", maximum_pixels=512,
                maximum_working_bytes=RESIDUAL_HOST_PREPARATION_RESERVE+4*1024**2))
    assert completed == [full.amplitude.shape]


def test_post_canonical_projection_cannot_create_a_budget_when_disabled(monkeypatch):
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_wave_device import residual_host_working_bytes
    from temsim.physics.wave_grid import WaveGridNumerics, WaveMemoryBudgetError
    import temsim.physics.potential_wave_carrier as kernel
    ledger = NumericalSupportLedger(0., 0.)
    source = _wave(n=64)
    completed = []
    canonical = kernel.propagate_plane_wave
    def observe(*args, **kwargs):
        result = canonical(*args, **kwargs)
        completed.append(result.amplitude.shape)
        return result
    monkeypatch.setattr(kernel, "propagate_plane_wave", observe)
    # The first residual fits. After canonical execution, the unchanged
    # source remains live alongside the next residual's input. A disabled
    # support ledger cannot crop that input to manufacture the missing space.
    budget = residual_host_working_bytes(source.amplitude.shape)+source.amplitude.nbytes//2
    with pytest.raises(WaveMemoryBudgetError):
        propagate_carrier_potential(_snapshot(), source, (0., .05), energy_ev=ENERGY_EV,
            kinetic_order=2, support_ledger=ledger, support_projection_policy="step_boundaries_and_retry",
            grid_numerics=WaveGridNumerics(angular_domain_padding="minimum_full_band", maximum_pixels=512,
                maximum_working_bytes=budget))
    assert ledger.events == ()
    assert completed == [source.amplitude.shape]


def _support_projection_with_new_spectral_tail():
    # This resolved finite wave has an optional 128 -> 72 support window.
    # The window's weak high-frequency tail needs lossless 72 -> 90 embedding.
    coordinate = np.arange(128)-64
    x, y = np.meshgrid(coordinate, coordinate)
    amplitude = np.exp(-(x*x+y*y)/100)*np.exp(2j*np.pi*.3*x+.37j)
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude, np.eye(2)*1e-7, np.zeros(2), np.zeros((2, 2)), np.zeros(2))


@pytest.mark.parametrize("stage", ["step-start support", "post-canonical support"])
def test_support_candidate_gets_lossless_spectral_preparation_before_ledger_commit(stage, monkeypatch):
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError, refine_plane_wave
    import temsim.physics.potential_wave_carrier as kernel
    source = _support_projection_with_new_spectral_tail()
    original = source.amplitude.copy()
    ledger = NumericalSupportLedger(.02, 1e-6)
    kernel._validate_wave(source, WaveGridNumerics(), 1e-10, 1e-10)
    proposal, proposed_ledger = ledger.compact(source, wavelength_m=WAVELENGTH, z_m=0.)
    with pytest.raises(WaveSamplingError, match="Nyquist"):
        kernel._validate_wave(proposal, WaveGridNumerics(), 1e-10, 1e-10)
    compact = NumericalSupportLedger.compact
    if stage == "post-canonical support":
        def only_after_map(self, wave, **options):
            return (wave, self) if options["z_m"] == 0. else compact(self, wave, **options)
        monkeypatch.setattr(NumericalSupportLedger, "compact", only_after_map)
    residual_inputs, canonical_outputs = [], []
    residual, canonical = kernel._residual, kernel.propagate_plane_wave
    def inspect_residual(snapshot, wave, *args, **kwargs):
        residual_inputs.append(wave)
        return residual(snapshot, wave, *args, **kwargs)
    def inspect_canonical(*args, **kwargs):
        output = canonical(*args, **kwargs)
        canonical_outputs.append(output)
        return output
    monkeypatch.setattr(kernel, "_residual", inspect_residual)
    monkeypatch.setattr(kernel, "propagate_plane_wave", inspect_canonical)
    output, record = propagate_carrier_potential(_snapshot(), source, (0., 1e-12),
        energy_ev=ENERGY_EV, kinetic_order=2, support_ledger=ledger,
        support_projection_policy="step_boundaries_and_retry" if stage.startswith("post") else "step_start_and_retry")
    selected = residual_inputs[0 if stage.startswith("step") else 1]
    before = source if stage.startswith("step") else canonical_outputs[-1]
    proposal, expected_ledger = compact(ledger, before, wavelength_m=WAVELENGTH,
                                      z_m=0. if stage.startswith("step") else 1e-12)
    assert selected.amplitude.shape[0] > proposal.amplitude.shape[0]
    expected = refine_plane_wave(proposal, selected.amplitude.shape, numerics=WaveGridNumerics())
    np.testing.assert_array_equal(selected.amplitude, expected.amplitude)
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(selected, name), getattr(expected, name))
    assert record["support_ledger"] == expected_ledger.to_dict()
    assert ledger.events == ()
    np.testing.assert_array_equal(source.amplitude, original)
    assert source.probability-output.probability == pytest.approx(expected_ledger.used_probability, abs=1e-15)
    assert any(row.get("stage") == stage+" Fourier operator preparation" for row in record["canonical_sampling"])


def test_unresolved_optional_support_candidate_keeps_valid_original_and_unspent_ledger():
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.wave_grid import WaveGridNumerics
    source = _support_projection_with_new_spectral_tail()
    numerics = WaveGridNumerics(automatic_refinement=False)
    reference, _ = propagate_carrier_potential(_snapshot(), source, (0., 1e-12),
        energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numerics)
    ledger = NumericalSupportLedger(.02, 1e-6)
    output, record = propagate_carrier_potential(_snapshot(), source, (0., 1e-12),
        energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numerics,
        support_ledger=ledger, support_projection_policy="step_start_and_retry")
    np.testing.assert_array_equal(output.amplitude, reference.amplitude)
    np.testing.assert_array_equal(output.basis_m, reference.basis_m)
    assert record["support_ledger"] == ledger.to_dict()
    declined = [row for row in record["canonical_sampling"] if row["method"] == "support projection proposal declined"]
    assert len(declined) == 1 and not declined[0]["projection_budget_spent"]
    assert "Nyquist" in declined[0]["reason"]


@pytest.mark.parametrize("failure", ["domain", "cancelled"])
def test_support_candidate_does_not_swallow_domain_errors_or_cancellation(failure, monkeypatch):
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    import temsim.physics.potential_wave_carrier as kernel
    source = _support_projection_with_new_spectral_tail()
    ledger = NumericalSupportLedger(.02, 1e-6)
    proposed = []
    compact, coverage = NumericalSupportLedger.compact, kernel._coverage
    def inspect_compact(self, wave, **options):
        result = compact(self, wave, **options)
        proposed.append(result[0])
        return result
    def inspect_coverage(snapshot, wave, *args):
        if proposed and wave.amplitude.shape != source.amplitude.shape:
            raise ValueError("Unknown field domain for projected lattice")
        return coverage(snapshot, wave, *args)
    monkeypatch.setattr(NumericalSupportLedger, "compact", inspect_compact)
    if failure == "domain":
        monkeypatch.setattr(kernel, "_coverage", inspect_coverage)
    with pytest.raises(ValueError if failure == "domain" else InterruptedError,
                       match="Unknown field domain" if failure == "domain" else "cancelled"):
        propagate_carrier_potential(_snapshot(), source, (0., 1e-12), energy_ev=ENERGY_EV,
            kinetic_order=2, support_ledger=ledger, support_projection_policy="step_start_and_retry",
            cancelled=lambda: failure == "cancelled" and bool(proposed))
    assert proposed and ledger.events == ()


def test_nested_support_retry_reserves_distinct_outer_and_kernel_entrances(monkeypatch):
    from temsim.cache_memory import retained_memory_inventory
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.wave_grid import WaveSamplingError, WaveGridNumerics
    import temsim.physics.potential_wave_carrier as kernel
    source = _wave(n=128, width=8e-7)
    residual, compact = kernel._residual, NumericalSupportLedger.compact
    additional_wave_bytes = kernel._additional_wave_bytes
    attempts, inventories, reservations = [], [], []
    def force_retry(snapshot, wave, z, *args, **kwargs):
        if z > 1e-12 and len(attempts) < 2:
            attempts.append(wave)
            raise WaveSamplingError("Independent forced operator sampling check", required_scale=1.25)
        return residual(snapshot, wave, z, *args, **kwargs)
    def postpone_first_crop(self, wave, **options):
        if len(attempts) == 1:
            return wave, self
        return compact(self, wave, **options)
    def inventory(candidate, *arrays):
        amount = additional_wave_bytes(candidate, *arrays)
        active = retained_memory_inventory(candidate.amplitude)
        outside = retained_memory_inventory(*arrays)
        assert amount == sum(size for identity, size in outside.items() if identity not in active)
        inventories.append((candidate, arrays, amount))
        return amount
    check = WaveGridNumerics.check_working_memory
    def reserve(self, shape, *, working_bytes, retained_bytes=0):
        reservations.append(retained_bytes)
        return check(self, shape, working_bytes=working_bytes, retained_bytes=retained_bytes)
    monkeypatch.setattr(kernel, "_residual", force_retry)
    monkeypatch.setattr(NumericalSupportLedger, "compact", postpone_first_crop)
    monkeypatch.setattr(kernel, "_additional_wave_bytes", inventory)
    monkeypatch.setattr(WaveGridNumerics, "check_working_memory", reserve)
    output, record = propagate_carrier_potential(_snapshot(), source, (0., 1e-12, 2e-12),
        energy_ev=ENERGY_EV, kinetic_order=2, support_ledger=NumericalSupportLedger(.02, 1e-6))
    assert len(attempts) == 2 and attempts[0].amplitude.shape != attempts[1].amplitude.shape
    # A raw pending window can alias the active candidate. Its explicit
    # reservation must not hide or double-count the three distinct entrances.
    candidate, arrays, reserved = next((candidate, outside, amount)
        for candidate, arrays, amount in inventories
        for outside in [tuple(array for array in arrays
                              if not np.shares_memory(candidate.amplitude, array))]
        if len({id(array) for array in outside}) >= 3)
    unique_arrays = list({id(array): array for array in arrays}.values())
    assert all(not np.shares_memory(left, right) for i, left in enumerate(unique_arrays)
        for right in unique_arrays[i+1:])
    assert reserved >= sum(array.nbytes for array in unique_arrays)
    assert reserved in reservations
    assert any(row.get("retained_input_bytes") == reserved for row in record["canonical_sampling"])
    assert output.probability > 0


@pytest.mark.parametrize("failure", ["disabled", "boundary", "invalid_basis"])
def test_entrance_preparation_does_not_bypass_non_sampling_failures(failure, monkeypatch):
    from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError
    import temsim.physics.wave_grid as grid
    wave = _wave(n=64, pitch=1e-7, width=4e-7)
    wave = replace(wave, amplitude=wave.amplitude*np.exp(2j*np.pi*.44*np.arange(64)[None, :]))
    numerics = WaveGridNumerics(automatic_refinement=failure != "disabled")
    if failure == "boundary":
        wave = replace(wave, amplitude=np.ones((64, 64), complex))
    if failure == "invalid_basis":
        wave = replace(wave, basis_m=np.diag((-1e-7, 1e-7)))
    monkeypatch.setattr(grid, "refine_plane_wave", lambda *a, **k: pytest.fail("Unadmitted input must not be refined"))
    expected = WaveSamplingError if failure == "disabled" else ValueError
    with pytest.raises(expected, match={"disabled": "Nyquist", "boundary": "boundary", "invalid_basis": "positive-orientation"}[failure]):
        propagate_carrier_potential(_snapshot(), wave, (0., 1e-9), energy_ev=ENERGY_EV, grid_numerics=numerics)


def test_remote_endpoints_do_not_admit_a_swept_box_across_the_near_lens():
    from temsim.physics.gaussian_remote_domain import POLICY
    from temsim.physics.potential_wave_carrier import _coverage_points
    native = FrozenAnalyticField("remote-sweep", ((.1, 0., .0001),),
        CoordinateRegistration(), (-100., 100.), 1e-6)
    provider = HighOrderRoundLensPotential(native, radial_order=3,
        axial_domain="gaussian_all_z", remote_domain=POLICY)
    points = np.array([(x, y, z) for x in (-2e-6, 2e-6)
                       for y in (-2e-6, 2e-6) for z in (-.1, .1)])
    np.testing.assert_array_equal(provider.sample(points)["B_t"], 0.)
    with pytest.raises(ValueError, match="box|domain"):
        _coverage_points(_snapshot(provider), points, -.1, .1)
    # An interval wholly in the proven remote tail remains admitted.
    points[:, 2] = np.where(points[:, 2] < 0., -.2, -.1)
    _coverage_points(_snapshot(provider), points, -.2, -.1)


def test_large_curvature_uses_exact_free_reference_without_expanding_or_erasing_phase(monkeypatch):
    wave = _wave(n=48, width=2e-7, curvature=-1000.)
    with pytest.raises(ValueError, match="undersampled"):
        wave.full_amplitude(WAVELENGTH)
    distance = 1e-4
    matrix = np.eye(4)
    matrix[:2, 2:] = np.eye(2)*distance
    path = CanonicalPath(1e-3)
    path.append(matrix)
    expected = propagate_plane_wave(wave, matrix, np.zeros(4), WAVELENGTH, **path.phase_kwargs())
    monkeypatch.setattr(PlaneWave, "full_amplitude", lambda *a, **k: pytest.fail("Carrier must remain analytic"))
    result, record = propagate_carrier_potential(_snapshot(), wave, (0., distance), energy_ev=ENERGY_EV,
        kinetic_order=2, kinetic_remainder_phase_limit=100.)
    np.testing.assert_allclose(result.amplitude, expected.amplitude, rtol=0., atol=1e-14)
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_allclose(getattr(result, name), getattr(expected, name), rtol=1e-13, atol=1e-15)
    assert record["phase_carrier_expanded"] is False and record["physical_phase_removed"] is False
    assert record["production_admission"] is False and record["normalization_applied"] is False
    assert result.probability == pytest.approx(wave.probability, abs=1e-12)


@pytest.mark.parametrize("shear", [0., .2])
def test_optional_exact_carrier_reselection_crosses_a_focus_with_analytic_complex_phase(shear):
    from temsim.physics.wave_grid import WaveGridNumerics
    width, curvature, distance = 1e-6, -10., .15
    wave = _wave(n=64, pitch=3e-7, width=width, curvature=curvature)
    wave = replace(wave, basis_m=3e-7*np.array(((1., shear), (0., 1.))))
    x, y = wave.coordinates_m()
    amplitude = np.exp(-(x*x+y*y)/(4*width**2)+.37j)
    amplitude /= np.linalg.norm(amplitude)
    wave = replace(wave, amplitude=amplitude)
    result, record = propagate_carrier_potential(_snapshot(), wave, np.linspace(0., distance, 151),
        energy_ev=ENERGY_EV, kinetic_order=2, reselect_carrier=True,
        grid_numerics=WaveGridNumerics(maximum_pixels=256))
    # Exact free Fresnel Gaussian, evaluated on the actual output coordinates.
    # The scalar complex denominator includes the focus/Gouy phase; no fitted
    # piston, intensity normalization or curvature subtraction is permitted.
    k = 2*np.pi/WAVELENGTH
    alpha = 1/(4*width**2)-.5j*k*curvature
    denominator = 1+2j*alpha*distance/k
    x, y = result.coordinates_m()
    expected = (wave.amplitude[32, 32]/denominator
        *np.exp(-alpha*(x*x+y*y)/denominator)
        *np.sqrt(np.linalg.det(result.basis_m)/np.linalg.det(wave.basis_m)))
    assert np.linalg.norm(_full_phase(result)-expected) < 2e-9
    assert result.probability == pytest.approx(wave.probability, abs=1e-12)
    assert record["exact_carrier_reselection"] is True
    assert any("exact envelope chirp compensation" in row.get("method", "")
               for row in record["canonical_sampling"])
    with pytest.raises(ValueError, match="explicit boolean"):
        propagate_carrier_potential(_snapshot(), wave, (0., 1e-6), energy_ev=ENERGY_EV,
                                    reselect_carrier="true")


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("shear", [0., .35])
def test_residual_matches_independent_full_operator_minus_quadratic_reference(order, shear):
    wave = _wave(n=12, pitch=2e-7, width=2e-7, curvature=12., tilt=(3e-5, -2e-5))
    wave = replace(wave, basis_m=2e-7*np.array(((1., shear), (0., 1.))))
    snapshot = _snapshot(_round(.2), _multipole())
    reference = _reference(snapshot, wave, .002, 1e-7, MOMENTUM)
    velocity, scalar, mechanical, frequency = _coefficients(snapshot, wave, .002, reference, WAVELENGTH)
    rng = np.random.default_rng(128)
    amplitude = rng.normal(size=wave.amplitude.shape)+1j*rng.normal(size=wave.amplitude.shape)
    actual = _apply_residual(amplitude, velocity, scalar, mechanical, frequency, order)
    # Dense Fourier differentiation is constructed independently of the
    # production FFT application and includes every cross-axis term in Pi^4.
    n = wave.amplitude.shape[0]
    fourier = np.exp(-2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/np.sqrt(n)
    symbol = WAVELENGTH*np.fft.fftfreq(n)
    one = fourier.conj().T@np.diag(symbol)@fourier
    index_derivatives = (np.kron(np.eye(n), one), np.kron(one, np.eye(n)))
    covectors = np.linalg.inv(wave.basis_m).T
    d = tuple(sum(covectors[i, j]*index_derivatives[j] for j in range(2)) for i in range(2))
    xy = wave.coordinates_m()
    carrier = wave.tilt_rad[:, None, None]+np.einsum("ij,jyx->iyx", wave.curvature_m1, xy)
    canonical = [d[i]+np.diag(carrier[i].ravel()) for i in range(2)]
    kinetic = [d[i]+np.diag(mechanical[i].ravel()) for i in range(2)]
    points = np.stack((xy[0], xy[1], np.full((n, n), .002)), axis=-1)
    a = np.moveaxis(snapshot.sample(points)["A_t_m"], -1, 0)
    total_k2 = sum(item@item for item in kinetic)
    full = .5*total_k2-reference.alpha*np.diag(a[2].ravel())
    if order == 4:
        full += total_k2@total_k2/8
    delta = xy-reference.center_xy[:, None, None]
    linear = np.einsum("ij,jyx->iyx", reference.jac[:, :2], delta)
    quadratic = .5*np.einsum("iyx,kij,jyx->kyx", delta, reference.hess[:, :2, :2], delta)
    alpha = reference.alpha
    vlinear = -alpha*(reference.a[:2, None, None]+linear[:2])
    vquadratic = (.5*alpha**2*(np.sum(reference.a[:2]**2)
        +2*np.einsum("i,iyx->yx", reference.a[:2], linear[:2]+quadratic[:2])
        +np.sum(linear[:2]**2, axis=0))-alpha*(reference.a[2]+linear[2]+quadratic[2])
        -alpha*np.einsum("i,iyx->yx", reference.center_u, quadratic[:2]))
    reference_operator = .5*sum(item@item for item in canonical)+np.diag(vquadratic.ravel())
    for i in range(2):
        coefficient = np.diag(vlinear[i].ravel())
        reference_operator += .5*(coefficient@canonical[i]+canonical[i]@coefficient)
    expected = ((full-reference_operator)@amplitude.ravel()).reshape(amplitude.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-20)


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("shear", [0., .35])
def test_residual_is_hermitian_for_spatially_varying_carrier_and_total_fields(order, shear):
    wave = _wave(n=16, pitch=2e-7, width=2e-7, curvature=35., tilt=(1e-4, -5e-5))
    wave = replace(wave, basis_m=2e-7*np.array(((1., shear), (0., 1.))))
    snapshot = _snapshot(_round(), _multipole())
    reference = _reference(snapshot, wave, .003, 1e-6, MOMENTUM)
    coefficients = _coefficients(snapshot, wave, .003, reference, WAVELENGTH)
    rng = np.random.default_rng(935)
    a = rng.normal(size=wave.amplitude.shape)+1j*rng.normal(size=wave.amplitude.shape)
    b = rng.normal(size=wave.amplitude.shape)+1j*rng.normal(size=wave.amplitude.shape)
    left = np.vdot(a, _apply_residual(b, *coefficients, order))
    right = np.vdot(_apply_residual(a, *coefficients, order), b)
    assert abs(left-right) <= 2e-13*max(abs(left), abs(right), 1e-30)


def test_fourth_order_vacuum_is_differential_and_preserves_nonzero_carrier():
    wave = _wave(n=48, pitch=1e-7, width=2e-7, tilt=(.001, -.0003))
    with pytest.raises(ValueError, match="undersampled"):
        wave.full_amplitude(WAVELENGTH)
    distance = 1e-6
    result, record = propagate_carrier_potential(_snapshot(), wave, (0., distance), energy_ev=ENERGY_EV,
        kinetic_order=4, kinetic_remainder_phase_limit=1.)
    fy, fx = np.meshgrid(np.fft.fftfreq(48), np.fft.fftfreq(48), indexing="ij")
    frequency = WAVELENGTH*np.stack((fx, fy))/1e-7
    kinetic = np.sum((frequency+wave.tilt_rad[:, None, None])**2, axis=0)
    # Moving with the exact quadratic reference removes only the affine
    # coordinate drift, not its scalar action or the quartic differential part.
    phase = -2*np.pi/WAVELENGTH*distance*(.5*np.sum(frequency**2, axis=0)+kinetic**2/8)
    expected = np.fft.ifft2(np.fft.fft2(wave.amplitude)*np.exp(1j*phase))
    expected *= np.exp(1j*2*np.pi/WAVELENGTH*distance*np.sum(wave.tilt_rad**2)/2)
    np.testing.assert_allclose(result.amplitude, expected, rtol=1e-8, atol=3e-11)
    np.testing.assert_allclose(result.origin_m, wave.tilt_rad*distance, atol=1e-18)
    assert record["series_amplitude_error_bound"] < 1e-10


def test_compose_sources_before_squaring_retains_magnetic_cross_terms():
    wave = _wave(n=32, width=2e-7)
    split = _snapshot(_round(.11), _round(.07))
    combined = _snapshot(_round(.18))
    nodes = np.linspace(-1e-4, 1e-4, 3)
    first, _ = propagate_carrier_potential(split, wave, nodes, energy_ev=ENERGY_EV)
    second, _ = propagate_carrier_potential(combined, wave, nodes, energy_ev=ENERGY_EV)
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad", "amplitude"):
        np.testing.assert_allclose(getattr(first, name), getattr(second, name), rtol=5e-12, atol=3e-13)


@pytest.mark.parametrize("dynamic", [True, False])
def test_hard_interfaces_and_frozen_dynamic_drives_are_rejected(dynamic):
    snapshot = _snapshot(_multipole(dynamic=dynamic))
    with pytest.raises(ValueError, match="dynamic|hard axial"):
        propagate_carrier_potential(snapshot, _wave(n=32, width=2e-7), (0., .06), energy_ev=ENERGY_EV)


def test_ill_conditioned_lattice_and_occupied_boundary_are_rejected_without_mutating_input():
    wave = _wave(n=32, width=2e-7)
    original = wave.amplitude.copy()
    sheared = replace(wave, basis_m=np.array(((1e-7, 0.), (0., 1e-16))))
    with pytest.raises(ValueError, match="resolved affine lattice"):
        propagate_carrier_potential(_snapshot(), sheared, (0., 1e-6), energy_ev=ENERGY_EV)
    occupied = replace(wave, amplitude=np.ones(wave.amplitude.shape, dtype=complex))
    with pytest.raises(ValueError, match="boundary"):
        propagate_carrier_potential(_snapshot(), occupied, (0., 1e-6), energy_ev=ENERGY_EV)
    np.testing.assert_array_equal(wave.amplitude, original)


def test_affine_corner_domain_admission_matches_every_grid_point_and_keeps_face_gate():
    from temsim.physics.potential_wave_carrier import _coverage, _coverage_points, _points, _corners
    wave = _wave(n=17, pitch=2e-5, width=4e-5)
    wave = replace(wave, basis_m=2e-5*np.array(((1., .4), (.2, 1.))))
    provider = HighOrderRoundLensPotential(_round().native,
        valid_radius_m=.0004, radial_order=3, axial_domain="gaussian_all_z")
    snapshot = _snapshot(provider)
    for x in (0., .0002, .0004):
        current = replace(wave, origin_m=np.array((x, 0.)))
        all_points = np.concatenate((_points(current, 0.).reshape(-1, 3),
                                    _points(current, .001).reshape(-1, 3)))
        outcomes = []
        for callback in (lambda: _coverage(snapshot, current, 0., .001),
                         lambda: _coverage_points(snapshot, all_points, 0., .001)):
            try:
                callback()
                outcomes.append(True)
            except ValueError:
                outcomes.append(False)
        assert outcomes[0] == outcomes[1]
        corners = _corners(current, 0.)
        assert np.allclose(corners.min(axis=0), all_points[:current.amplitude.size].min(axis=0))
        assert np.allclose(corners.max(axis=0), all_points[:current.amplitude.size].max(axis=0))
    compact = AnalyticPotentialDomain(AnalyticMagneticPotential(replace(_multipole().base.field,
        native_support_mm=(.2, .4))))
    # Both endpoint planes are model-zero, but the active slab is inside.
    with pytest.raises(ValueError, match="hard axial support face"):
        _coverage(_snapshot(compact), wave, 0., .001)


def test_unknown_fem_and_electric_provider_are_not_admitted_by_jet_like_attributes():
    from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
    provider = AxisymmetricMagneticPotential((0., .003), (-.01, .01), ((0., 0.), (.001, .001)))
    with pytest.raises(ValueError, match="native analytical static"):
        propagate_carrier_potential(_snapshot(provider), _wave(), (0., 1e-6), energy_ev=ENERGY_EV)


def test_large_carrier_kinetic_budget_and_cancellation_are_explicit():
    wave = _wave(n=32, width=2e-7, tilt=(.4, 0.))
    with pytest.raises(ValueError, match="small-angle"):
        propagate_carrier_potential(_snapshot(), wave, (0., 1e-6), energy_ev=ENERGY_EV)
    with pytest.raises(InterruptedError, match="cancelled"):
        propagate_carrier_potential(_snapshot(), _wave(n=32, width=2e-7), (0., 1e-6),
            energy_ev=ENERGY_EV, cancelled=lambda: True)


@pytest.mark.parametrize("nodes", [(0.,), (1., 0.), (0., np.inf), (0., 0.)])
def test_invalid_axial_nodes_are_rejected(nodes):
    with pytest.raises(ValueError, match="axial nodes"):
        propagate_carrier_potential(_snapshot(), _wave(), nodes, energy_ev=ENERGY_EV)


def _gauge_wave(wave, gauge, z):
    alpha = -e/MOMENTUM
    point = np.r_[wave.origin_m, z]
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    return replace(wave, amplitude=wave.amplitude*gauge.electron_phase_factor(point),
        tilt_rad=tilt+alpha*gauge.vector_potential_shift_t_m(point)[:2],
        curvature_m1=curvature+alpha*gauge.hessian_t[:2, :2])


@pytest.mark.parametrize("shear,reselect,scalar", [(0., False, False), (.35, False, False),
                                                (.35, True, False), (.35, True, True)])
def test_static_xy_xz_yz_gauge_retains_endpoint_complex_phase_and_flux(shear, reselect, scalar):
    wave = _wave(n=32, width=2e-7)
    wave = replace(wave, basis_m=1e-7*np.array(((1., shear), (0., 1.))))
    gauge = StaticPolynomialGauge(3e-16, (2e-11, -3e-11, 1e-11),
        ((2e-5, 3e-6, 1e-5), (3e-6, -1e-5, -7e-6), (1e-5, -7e-6, 0.)))
    nodes = np.linspace(-1e-5, 1e-5, 5)
    source = _round(.15)
    first, _ = propagate_carrier_potential(_snapshot(source), wave, nodes, energy_ev=ENERGY_EV,
                                          reselect_carrier=reselect, scalar_splitting=scalar)
    second, _ = propagate_carrier_potential(_snapshot(source, gauge=gauge), _gauge_wave(wave, gauge, nodes[0]),
        nodes, energy_ev=ENERGY_EV, reselect_carrier=reselect, scalar_splitting=scalar)
    expected = _gauge_wave(first, gauge, nodes[-1])
    np.testing.assert_allclose(second.basis_m, expected.basis_m, rtol=2e-10, atol=1e-18)
    np.testing.assert_allclose(second.origin_m, expected.origin_m, rtol=0., atol=1e-15)
    # Compare actual complex values; no best-fit phase, centroid alignment,
    # or intensity renormalization is permitted in the gauge test.
    np.testing.assert_allclose(_full_phase(second), _full_phase(expected), rtol=3e-8, atol=3e-10)
    assert second.probability == pytest.approx(first.probability, abs=3e-12)


def _pure_hex(strength=1e8):
    return AnalyticPotentialDomain(AnalyticMagneticPotential(replace(_multipole().base.field,
        polynomial_terms=((3, 0, strength), (1, 2, -3*strength)))))


@pytest.mark.parametrize("scalar_splitting", [False, True])
@pytest.mark.parametrize("reference_quadrature", ["midpoint", "cf4"])
def test_axial_splitting_complex_error_converges_without_phase_fitting(scalar_splitting, reference_quadrature):
    from scipy.integrate import solve_ivp
    wave = _wave(n=64, pitch=5e-8, width=1e-7)
    snapshot = _snapshot(_pure_hex())
    results = [propagate_carrier_potential(snapshot, wave, np.linspace(0., 1e-4, count+1),
        energy_ev=ENERGY_EV, exponential_error=1e-11, scalar_splitting=scalar_splitting,
        reference_quadrature=reference_quadrature)[0]
        for count in (32, 64, 128)]
    # Independently integrate the complete finite-grid Hamiltonian with an
    # adaptive high-order ODE method. It uses no carrier split, midpoint map,
    # fitted phase, centroid registration or probability normalization.
    fy, fx = np.meshgrid(np.fft.fftfreq(64), np.fft.fftfreq(64), indexing="ij")
    squared = (WAVELENGTH/5e-8)**2*(fx*fx+fy*fy)
    xy = wave.coordinates_m()
    points = np.stack((xy[0], xy[1], np.zeros((64, 64))), axis=-1)
    potential = e/MOMENTUM*snapshot.sample(points)["A_t_m"][..., 2]
    def rhs(_z, value):
        amplitude = value.reshape(64, 64)
        kinetic = np.fft.ifft2((.5*squared+.125*squared**2)*np.fft.fft2(amplitude))
        return (-2j*np.pi/WAVELENGTH*(kinetic+potential*amplitude)).ravel()
    solved = solve_ivp(rhs, (0., 1e-4), wave.amplitude.ravel(), method="DOP853", rtol=2e-13, atol=2e-15)
    assert solved.success
    exact_grid = solved.y[:, -1].reshape(64, 64)
    errors = [np.linalg.norm(result.amplitude-exact_grid) for result in results]
    assert errors[1] < .3*errors[0] and errors[2] < .3*errors[1]
    assert errors[2] < 1e-8
    for result in results:
        np.testing.assert_allclose(result.basis_m, wave.basis_m, rtol=1e-14, atol=0.)
        assert result.probability == pytest.approx(wave.probability, abs=3e-11)


def test_exact_real_scalar_split_converges_to_independent_dense_mixed_covariant_operator():
    from temsim.physics.potential_wave_carrier import _residual
    energy = 100.*e
    momentum = np.sqrt(energy*(energy+2*m_e*c*c))/c
    wavelength = 2*np.pi*hbar/momentum
    wave = _wave(n=24, pitch=1e-8, width=1.5e-8, tilt=(.01, -.008))
    snapshot = _snapshot(_round(1.), _pure_hex(2e11))
    distance = 1e-5
    reference = _reference(snapshot, wave, 0., distance, momentum)
    velocity, scalar, mechanical, _ = _coefficients(snapshot, wave, 0., reference, wavelength)
    n = len(wave.amplitude)
    transform = np.exp(-2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/np.sqrt(n)
    one = transform.conj().T@np.diag(wavelength*np.fft.fftfreq(n)/1e-8)@transform
    derivatives = (np.kron(np.eye(n), one), np.kron(one, np.eye(n)))
    dense = np.diag(scalar.ravel()).astype(complex)
    k_squared = np.zeros_like(dense)
    for axis in range(2):
        v = np.diag(velocity[axis].ravel())
        dense += .5*(v@derivatives[axis]+derivatives[axis]@v)
        covariant = derivatives[axis]+np.diag(mechanical[axis].ravel())
        k_squared += covariant@covariant
    dense += k_squared@k_squared/8
    np.testing.assert_allclose(dense, dense.conj().T, atol=1e-17, rtol=1e-13)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    exact = eigenvectors@(np.exp(-2j*np.pi/wavelength*distance*eigenvalues)
                          *(eigenvectors.conj().T@wave.amplitude.ravel()))
    errors = []
    for count in (16, 32, 64):
        result = wave
        for _ in range(count):
            result, evidence = _residual(snapshot, result, 0., distance/count, reference, wavelength,
                4, 1e-11/count, .01/count, 1e-9, 1e-10, lambda: False, scalar_splitting=True)
        errors.append(np.linalg.norm(result.amplitude.ravel()-exact))
        assert result.probability == pytest.approx(wave.probability, abs=2e-12)
        assert evidence["scalar_exact_strang"] is True
        assert evidence["scalar_splitting_error_in_series_bound"] is False
    assert errors[1] < .3*errors[0] and errors[2] < .3*errors[1]
    assert errors[2] < 1e-8


def test_residual_refinement_replays_unchanged_action_without_filtering_or_probability_repair():
    from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError, refine_plane_wave
    wave = _wave(n=48, pitch=1e-7, width=2e-7)
    original = wave.amplitude.copy()
    snapshot = _snapshot(_pure_hex(5e7))
    kwargs = {"energy_ev": ENERGY_EV, "scalar_splitting": True}
    with pytest.raises(WaveSamplingError, match="scalar phase"):
        propagate_carrier_potential(snapshot, wave, (0., 1e-5), **kwargs,
            grid_numerics=WaveGridNumerics(maximum_pixels=256, automatic_refinement=False))
    actual, record = propagate_carrier_potential(snapshot, wave, (0., 1e-5), **kwargs,
        grid_numerics=WaveGridNumerics(maximum_pixels=256))
    refined = refine_plane_wave(wave, actual.amplitude.shape)
    expected, _ = propagate_carrier_potential(snapshot, refined, (0., 1e-5), **kwargs,
        grid_numerics=WaveGridNumerics(maximum_pixels=256, automatic_refinement=False))
    assert actual.amplitude.shape[0] > wave.amplitude.shape[0]
    np.testing.assert_allclose(actual.amplitude, expected.amplitude, atol=2e-14, rtol=2e-12)
    np.testing.assert_allclose(actual.basis_m, expected.basis_m, atol=0., rtol=1e-14)
    np.testing.assert_array_equal(wave.amplitude, original)
    assert actual.probability == pytest.approx(wave.probability, abs=1e-12)
    assert any("unitary complex Fourier embedding" in row.get("method", "")
               for row in record["canonical_sampling"])


@pytest.mark.parametrize("backend", ["native_compiled", "native_plane"])
def test_strict_compiled_sampling_preserves_complex_carrier_phase_gauge_and_absolute_flux(backend):
    wave = _wave(n=32, width=2e-7, curvature=-20., tilt=(2e-5, -1e-5))
    wave = replace(wave, basis_m=1e-7*np.array(((1., .2), (-.1, 1.))))
    snapshot = _snapshot(_round(), _multipole(), gauge=StaticPolynomialGauge(
        2e-16, (1e-11, -2e-11, 3e-11), ((1e-5, 2e-6, 3e-6), (2e-6, -1e-5, -4e-6), (3e-6, -4e-6, 0.))))
    kwargs = {"energy_ev": ENERGY_EV, "scalar_splitting": True, "reselect_carrier": True}
    expected, _ = propagate_carrier_potential(snapshot, wave, np.linspace(0., 2e-6, 5), **kwargs)
    progress = []
    actual, metadata = propagate_carrier_potential(snapshot, wave, np.linspace(0., 2e-6, 5), **kwargs,
                                                   sampling_backend=backend, progress_callback=progress.append)
    np.testing.assert_allclose(actual.amplitude, expected.amplitude, atol=2e-13, rtol=2e-10)
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        if name == "curvature_m1":
            # Nearly-zero off-diagonal entries cannot carry a relative error
            # test independent of the full curvature's float64 roundoff.
            delta = actual.curvature_m1-expected.curvature_m1
            assert np.linalg.norm(delta) < 64*np.finfo(float).eps*max(1., np.linalg.norm(expected.curvature_m1))
        else:
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-15, rtol=2e-12)
    np.testing.assert_allclose(_full_phase(actual), _full_phase(expected), atol=2e-13, rtol=2e-10)
    assert actual.probability == pytest.approx(expected.probability, abs=2e-13)
    assert metadata["field_sampling_backend"] == ("numba-native-strict" if backend == "native_compiled"
                                                   else "numba-aligned-plane-strict")
    assert metadata["snapshot_digest"] == snapshot.numerical_digest
    assert progress[-1]["z_m"] == 2e-6
    assert progress[-1]["probability"] == actual.probability
    assert progress[-1]["completed_steps"] == progress[-1]["total_steps"] == 4
    with pytest.raises(ValueError, match="sampling_backend"):
        propagate_carrier_potential(snapshot, wave, (0., 1e-6), energy_ev=ENERGY_EV,
                                    sampling_backend="automatic")


def test_aligned_compact_faces_at_nodes_apply_finite_deflection_once_without_extra_kick():
    strength = 1e-4
    provider = AnalyticPotentialDomain(AnalyticMagneticPotential(replace(_multipole().base.field,
        polynomial_terms=((1, 0, -strength),), native_support_mm=(-.001, .001))))
    nodes = (-2e-6, -1e-6, 1e-6, 2e-6)
    result, _ = propagate_carrier_potential(_snapshot(provider), _wave(n=32, width=2e-7), nodes,
        energy_ev=ENERGY_EV, kinetic_order=2)
    acceleration = e/MOMENTUM*strength
    np.testing.assert_allclose(result.tilt_rad, (acceleration*2e-6, 0.), rtol=1e-13, atol=1e-20)
    np.testing.assert_allclose(result.origin_m, (acceleration*4e-12, 0.), rtol=1e-13, atol=1e-25)
    with pytest.raises(ValueError, match="hard axial"):
        propagate_carrier_potential(_snapshot(provider), _wave(n=32, width=2e-7), nodes[::3],
            energy_ev=ENERGY_EV, kinetic_order=2)
    with pytest.raises(ValueError, match="hard axial face"):
        provider.sample_jet((0., 0., 1e-6))


def test_tilted_compact_face_cutting_observation_grid_stays_rejected():
    from scipy.spatial.transform import Rotation
    rotation = tuple(map(tuple, Rotation.from_rotvec((.01, 0., 0.)).as_matrix()))
    provider = AnalyticPotentialDomain(AnalyticMagneticPotential(replace(_multipole().base.field,
        native_support_mm=(-.001, .001), registration=CoordinateRegistration((0., 0., 0.), rotation))))
    with pytest.raises(ValueError, match="hard axial"):
        propagate_carrier_potential(_snapshot(provider), _wave(n=32, width=2e-7),
            (-2e-6, -1e-6, 1e-6, 2e-6), energy_ev=ENERGY_EV)


def test_long_quartic_step_cannot_wrap_through_periodic_boundary_and_return_unnoticed():
    # The exact quartic group velocity adds a positive ~160 nm to the
    # reference shift, twice this period. An endpoint-only check accepted a
    # falsely wrapped negative shift. Every bounded substep now checks support.
    wave = _wave(n=32, pitch=2.5e-9, width=6.25e-9, tilt=(.004, 0.))
    with pytest.raises(ValueError, match="Occupied carrier envelope boundary"):
        propagate_carrier_potential(_snapshot(), wave, (0., 5.), energy_ev=300000.,
            boundary_fraction_limit=1e-8)


def test_closed_reference_orbit_cannot_hide_an_intermediate_excursion_outside_field_domain():
    energy = 300000.*e
    momentum = np.sqrt(energy*(energy+2*m_e*c*c))/c
    full_turn = 2*np.pi*momentum/e
    potential = AnalyticMagneticPotential(FrozenAnalyticField("uniform-limit", ((1., 0., 1e3),),
        CoordinateRegistration(), (-1e8, 1e8), 1.5e-6))
    wave = _wave(n=32, pitch=5e-8, width=8e-8, tilt=(.0005, 0.))
    # Both reference endpoints return almost to zero, but the exact uniform-B
    # orbit reaches 2*p*tilt/(e*B) = 2.10 um, outside this 1.5 um cylinder.
    with pytest.raises(ValueError, match="swept domain"):
        propagate_carrier_potential(_snapshot(potential), wave, (-full_turn/2, full_turn/2),
            energy_ev=300000., kinetic_order=2, boundary_fraction_limit=1e-8)


def test_unresolved_envelope_spectrum_is_rejected_when_preparation_is_disabled():
    from temsim.physics.wave_grid import WaveGridNumerics
    wave = _wave(n=32, width=2e-7)
    modulation = np.exp(2j*np.pi*.45*np.arange(32))
    wave = replace(wave, amplitude=wave.amplitude*modulation[None, :])
    before = wave.amplitude.copy()
    with pytest.raises(ValueError, match="Nyquist guard"):
        propagate_carrier_potential(_snapshot(), wave, (0., 1e-6), energy_ev=ENERGY_EV,
            grid_numerics=WaveGridNumerics(automatic_refinement=False))
    np.testing.assert_array_equal(wave.amplitude, before)
