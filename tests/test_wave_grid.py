"""Numerical complex-field checks, not tip-to-image source qualification."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.multipole_wave import apply_multipole_phase, multipole_action
from temsim.physics.wave_grid import (WaveGridNumerics, WaveSamplingError,
                                      apply_resolved_operator, refine_plane_wave)


@pytest.mark.parametrize("shape", [(32, 32), (33, 35), (32, 35)])
def test_refinement_retains_all_complex_fourier_coefficients_and_affine_carriers(shape):
    rng = np.random.default_rng(937)
    amplitude = rng.normal(size=shape)+1j*rng.normal(size=shape)
    amplitude *= np.sqrt(.37/np.sum(abs(amplitude)**2))
    wave = PlaneWave(amplitude, np.array(((2e-8, .3e-8), (-.2e-8, 3e-8))),
                     np.array((7e-6, -8e-6)), np.array(((.2, -.3), (-.3, .4))), np.array((.01, -.02)))
    result = refine_plane_wave(wave, tuple(2*n for n in shape))
    np.testing.assert_allclose(result.basis_m, wave.basis_m/2, atol=0, rtol=0)
    for field in ("origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(result, field), getattr(wave, field))
    starts = tuple(n % 2 for n in shape)
    selected = result.amplitude[starts[0]::2, starts[1]::2]
    np.testing.assert_allclose(selected*2, wave.amplitude, atol=1e-16, rtol=1e-13)
    assert result.probability == pytest.approx(.37, abs=3e-16)
    assert not result.amplitude.flags.writeable


def test_even_negative_nyquist_bin_keeps_its_complex_phase_and_norm():
    n, factor = 32, 4
    x = np.arange(n)-n//2
    a = np.tile(np.exp(-1j*np.pi*x), (n, 1))/n
    wave = PlaneWave(a, np.eye(2), np.zeros(2))
    result = refine_plane_wave(wave, (n*factor, n*factor))
    fine_x = (np.arange(n*factor)-n*factor//2)/factor
    expected = np.tile(np.exp(-1j*np.pi*fine_x), (n*factor, 1))/(n*factor)
    np.testing.assert_allclose(result.amplitude, expected, atol=5e-17)
    assert result.probability == pytest.approx(1., abs=2e-15)


def test_cuda_fourier_refinement_and_nonlinear_carriers_preserve_phase_and_rejections():
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        available = False
    if not available:
        pytest.skip("CUDA is unavailable")
    from temsim.physics.wave_device import device_scope
    source = gaussian(128)
    x, y = source.coordinates_m()
    source = replace(source, amplitude=source.amplitude*(1+.1*x/16e-9+.03j*y/16e-9),
                     curvature_m1=np.array(((300., 20.), (20., -100.))),
                     tilt_rad=np.array((1e-6, -2e-6)))
    strengths = dict(normal_m2=2e8, skew_m2=-3e8, spherical_m3=4e15)
    expected = apply_multipole_phase(refine_plane_wave(source, (257, 259)), 2e-12, **strengths)
    with device_scope("Require GPU", maximum_working_bytes=256*1024**2):
        device = replace(source, amplitude=cp.asarray(source.amplitude))
        actual = apply_multipole_phase(refine_plane_wave(device, (257, 259)), 2e-12, **strengths)
        assert isinstance(actual.amplitude, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(actual.full_amplitude(2e-12)), expected.full_amplitude(2e-12),
                                   rtol=2e-11, atol=2e-13)
        assert actual.probability == pytest.approx(expected.probability, abs=2e-14)
        with pytest.raises(WaveSamplingError, match="undersampled"):
            apply_multipole_phase(device, 2e-12, spherical_m3=1e26)


def test_wave_compute_policy_and_device_budget_are_validated_and_identified():
    for policy in ("cpu", "gpu", "auto", "CPU", "Numba CPU", "CUDA GPU", "Prefer GPU", "Require GPU"):
        numerics = WaveGridNumerics(compute_backend=policy, maximum_device_working_bytes=2*1024**3)
        numerics.validate()
        assert numerics.column_identity()["compute_backend"] == policy
        assert numerics.column_identity()["maximum_device_working_bytes"] == 2*1024**3
    for value in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="device working memory"):
            WaveGridNumerics(maximum_device_working_bytes=value).validate()


@pytest.mark.parametrize("field", ["maximum_posed_lens_phase_error_rad_per_m",
                                   "maximum_spherical_obliquity_phase_error_rad"])
def test_posed_lens_phase_budget_is_validated_and_invalidates_column_cache(field):
    original = WaveGridNumerics()
    assert original.column_identity()[field] == .01
    changed = replace(original, **{field: .001})
    assert changed.column_identity() != original.column_identity()
    for value in (True, 0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match="phase error budget"):
            replace(original, **{field: value}).validate()


@pytest.mark.parametrize("field", ["maximum_support_loss_fraction", "maximum_spectral_loss_fraction",
                                  "maximum_incoherent_mode_loss_probability",
                                  "maximum_accumulated_support_l2"])
def test_support_budget_is_numerical_validated_and_part_of_column_identity(field):
    original = WaveGridNumerics()
    for value in (0., 1e-7, .01):
        changed = replace(original, **{field: value})
        changed.validate()
        assert changed.column_identity()[field] == value
        if value != getattr(original, field):
            assert changed.column_identity() != original.column_identity()
    for value in (True, -1., np.nan, np.inf, .10001):
        with pytest.raises(ValueError, match=field):
            replace(original, **{field: value}).validate()


def test_multipole_tail_admission_is_separate_validated_numerics_and_invalidates_column_cache():
    original = WaveGridNumerics()
    assert original.multipole_spectral_tail_fraction == 1e-12
    for value in (0., 1e-12, 1e-8, 1e-4):
        changed = replace(original, multipole_spectral_tail_fraction=value)
        changed.validate()
        assert changed.column_identity()["multipole_spectral_tail_fraction"] == value
        assert changed.maximum_spectral_loss_fraction == original.maximum_spectral_loss_fraction
        assert changed.maximum_accumulated_support_l2 == original.maximum_accumulated_support_l2
        if value != original.multipole_spectral_tail_fraction:
            assert changed.column_identity() != original.column_identity()
    for value in (True, np.bool_(False), -1e-12, np.nan, np.inf, 1.00001e-4):
        with pytest.raises(ValueError, match="Multipole spectral tail fraction"):
            replace(original, multipole_spectral_tail_fraction=value).validate()


def gaussian(n):
    axis = (np.arange(n)-n//2)*(128/n)*4e-9
    xx, yy = np.meshgrid(axis, axis)
    a = np.exp(-(xx*xx+yy*yy)/(4*(16e-9)**2)).astype(complex)
    a /= np.linalg.norm(a)
    return PlaneWave(a, np.eye(2)*(128/n)*4e-9, np.zeros(2))


def test_adaptive_cs_complex_field_and_diffraction_match_independent_fine_grid():
    source, lam, cs = gaussian(128), 2e-12, 4e17
    operator = lambda w: apply_multipole_phase(w, lam, spherical_m3=cs)
    with pytest.raises(WaveSamplingError):
        operator(source)
    result, rows = apply_resolved_operator(source, operator)
    assert rows and result.amplitude.shape[0] > 128
    assert result.probability == pytest.approx(source.probability, abs=1e-14)
    reference = gaussian(result.amplitude.shape[0]*2)
    x, y = reference.coordinates_m()
    # Independent original quartic law, without factoring/fitting any phase.
    expected = reference.amplitude*np.exp(-.5j*np.pi*cs*(x*x+y*y)**2/lam)
    np.testing.assert_allclose(result.amplitude, expected[::2, ::2]*2, atol=2e-12, rtol=2e-9)
    def free_field(a, basis, length=1e-5):
        fy, fx = np.meshgrid(np.fft.fftfreq(len(a)), np.fft.fftfreq(len(a)), indexing="ij")
        f = np.einsum("ij,jyx->iyx", np.linalg.inv(basis).T, np.stack((fx, fy)))
        return np.fft.ifft2(np.fft.fft2(a)*np.exp(-1j*np.pi*lam*length*(f*f).sum(axis=0)))
    propagated = free_field(result.amplitude, result.basis_m)
    fine = free_field(expected, reference.basis_m)
    assert np.linalg.norm(propagated-2*fine[::2, ::2]) < 2e-8


def test_budget_failure_precedes_allocation_and_does_not_apply_or_ignore_an_operator(monkeypatch):
    source = gaussian(128)
    calls = []
    def operator(w):
        calls.append(w)
        raise WaveSamplingError("known unresolved optical operator", 16)
    def allocation(*a, **kw):
        raise AssertionError("large allocation must be checked first")
    monkeypatch.setattr(np, "zeros", allocation)
    with pytest.raises(ValueError, match="budget exceeded"):
        apply_resolved_operator(source, operator, numerics=WaveGridNumerics(maximum_pixels=256))
    assert calls == [source]
    with pytest.raises(WaveSamplingError, match="unresolved"):
        apply_resolved_operator(source, operator, numerics=WaveGridNumerics(automatic_refinement=False))
    assert calls == [source, source]
    with pytest.raises(InterruptedError):
        apply_resolved_operator(source, operator, cancelled=lambda: True)
    assert len(calls) == 2


def test_non_sampling_errors_are_not_retried_and_invalid_budgets_are_rejected():
    def wrong_operator(w):
        raise ValueError("physical field is unavailable")
    with pytest.raises(ValueError, match="field is unavailable"):
        apply_resolved_operator(gaussian(128), wrong_operator)
    for bad in (WaveGridNumerics(maximum_pixels=True), WaveGridNumerics(maximum_pixels=16),
                WaveGridNumerics(automatic_refinement=1), WaveGridNumerics(maximum_working_bytes=0)):
        with pytest.raises(ValueError):
            bad.validate()


def test_resolved_wave_zero_does_not_masquerade_as_aliasing():
    wave = gaussian(128)
    x, y = wave.coordinates_m()
    a = wave.amplitude*(x/16e-9+1j*y/16e-9)
    a /= np.linalg.norm(a)
    vortex = replace(wave, amplitude=a)
    result = apply_multipole_phase(vortex, 2e-12, spherical_m3=4e14)
    expected = a*np.exp(2j*np.pi*multipole_action(x, y, spherical_m3=4e14)/2e-12)
    np.testing.assert_allclose(result.amplitude, expected, atol=1e-15)


def test_operator_specific_memory_estimate_respects_existing_budget():
    numerics = WaveGridNumerics(maximum_working_bytes=400*64*64)
    assert numerics.check((64, 64)) == 256*64*64
    with pytest.raises(ValueError, match="budget exceeded"):
        numerics.check((64, 64), working_bytes_per_pixel=512)


def test_refinement_uses_required_resolution_without_power_of_two_overshoot():
    source = gaussian(128)
    visited = []
    def operator(wave):
        visited.append(wave.amplitude.shape)
        if wave.amplitude.shape[0] < 263:
            raise WaveSamplingError("needs slightly more than double resolution", 2.05)
        return wave
    # The former power-of-two factor would request 512 and fail this budget,
    # although the operator requires only 263 cells. It must still be retried.
    result, rows = apply_resolved_operator(source, operator,
        numerics=WaveGridNumerics(maximum_pixels=300))
    assert len(visited) == 2 and visited[0] == (128, 128)
    assert 263 <= result.amplitude.shape[0] <= 300
    assert result.amplitude.shape[0] % 2 == 0
    assert result.probability == pytest.approx(source.probability, abs=2e-14)
    assert rows[0]["requested_scale"] == 2.05


def test_small_required_scale_still_grows_and_rechecks_the_physical_operator():
    source = gaussian(128)
    visited = []
    def operator(wave):
        visited.append(wave.amplitude.shape)
        if len(visited) == 1:
            raise WaveSamplingError("resolution guard", 1.001)
        return wave
    result, _ = apply_resolved_operator(source, operator)
    assert visited == [(128, 128), (160, 160)]
    assert result.probability == pytest.approx(source.probability, abs=2e-14)
    assert "fft-friendly" in WaveGridNumerics().column_identity()["refinement_grid_selection"]


def test_sampling_preparation_rechecks_changed_input_before_using_resolution_scale():
    source = gaussian(128)
    original = source.amplitude.copy()
    events = []
    def operator(wave):
        events.append(("operator", wave.amplitude.shape, wave.curvature_m1 is None))
        if wave.curvature_m1 is None:
            raise WaveSamplingError("old representation requires a large grid", 16.)
        if wave.amplitude.shape[0] < 160:
            raise WaveSamplingError("prepared representation needs a small refinement", 1.01)
        return wave
    def prepare(wave, error):
        events.append(("prepare", error.required_scale))
        # Toy preparation deliberately alters the callback's representation;
        # this test checks retry ordering, not a physical gauge-change claim.
        return replace(wave, curvature_m1=np.eye(2)) if wave.curvature_m1 is None else wave
    result, rows = apply_resolved_operator(source, operator, prepare_retry=prepare,
                                          numerics=WaveGridNumerics(maximum_pixels=256))
    assert result.amplitude.shape == (160, 160)
    assert events == [("operator", (128, 128), True), ("prepare", 16.),
                      ("operator", (128, 128), False), ("prepare", 1.01),
                      ("operator", (160, 160), False)]
    assert "physical operator rechecked" in rows[0]["method"]
    np.testing.assert_array_equal(source.amplitude, original)
    assert source.curvature_m1 is None


def test_disabled_refinement_does_not_call_sampling_preparation():
    def operator(wave):
        raise WaveSamplingError("strict unresolved operator", 2.)
    def prepare(wave, error):
        raise AssertionError("disabled refinement must not invoke preparation")
    with pytest.raises(WaveSamplingError, match="strict unresolved"):
        apply_resolved_operator(gaussian(128), operator, prepare_retry=prepare,
                                numerics=WaveGridNumerics(automatic_refinement=False))


def test_mandatory_operator_resource_uses_smaller_resolved_candidate_before_allocation(monkeypatch):
    import temsim.physics.wave_grid as grid
    source = gaussian(128)
    original = source.amplitude.copy()
    visited, allocated = [], []
    def operator(wave):
        visited.append(wave.amplitude.shape)
        if len(visited) == 1:
            raise WaveSamplingError("same unresolved operator", 1.001)
        return wave
    def device_budget(shape):
        if max(shape) > 150:
            raise grid.WaveGridBudgetError("required device partition")
    refine = grid.refine_plane_wave
    def allocation(wave, shape, **kwargs):
        allocated.append(shape)
        return refine(wave, shape, **kwargs)
    monkeypatch.setattr(grid, "refine_plane_wave", allocation)
    output, rows = apply_resolved_operator(source, operator,
        operator_resource_check=device_budget,
        numerics=WaveGridNumerics(maximum_working_bytes=2**30))
    assert visited == [(128, 128), (140, 140)]
    assert allocated == [(140, 140)]
    assert rows[0]["budget_limited_growth"] is True
    assert rows[0]["resource_preferred_growth"] is False
    expected = refine(source, (140, 140))
    np.testing.assert_array_equal(output.amplitude, expected.amplitude)
    np.testing.assert_array_equal(source.amplitude, original)


@pytest.mark.parametrize("limit,expected", [(150, 140), (100, 160)])
def test_optional_resource_preference_keeps_valid_fallback(limit, expected):
    visited = []
    def operator(wave):
        visited.append(wave.amplitude.shape)
        if len(visited) == 1:
            raise WaveSamplingError("resolution requires a new candidate", 1.001)
        return wave
    output, rows = apply_resolved_operator(gaussian(128), operator,
        preferred_resource_check=lambda shape: max(shape) <= limit)
    assert visited == [(128, 128), (expected, expected)]
    assert rows[0]["resource_preferred_growth"] is (limit == 150)
    assert rows[0]["budget_limited_growth"] is False
    assert output.probability == pytest.approx(1., abs=2e-14)


def test_required_resource_and_cancellation_fail_before_any_operator_or_refinement(monkeypatch):
    import temsim.physics.wave_grid as grid
    source = gaussian(128)
    def unavailable(shape):
        raise grid.WaveGridBudgetError("required device partition")
    def never(*args, **kwargs):
        pytest.fail("no operator or grid allocation may follow rejected resources")
    monkeypatch.setattr(grid, "refine_plane_wave", never)
    with pytest.raises(grid.WaveGridBudgetError, match="device partition"):
        apply_resolved_operator(source, never, operator_resource_check=unavailable)
    with pytest.raises(InterruptedError):
        apply_resolved_operator(source, never, operator_resource_check=never, cancelled=lambda: True)


@pytest.mark.parametrize("error", [InterruptedError("stop"), ValueError("invalid device settings")])
def test_additional_resource_errors_do_not_become_sampling_retries(error):
    def check(shape):
        raise error
    with pytest.raises(type(error), match=str(error)):
        apply_resolved_operator(gaussian(128), lambda _: pytest.fail("must not execute"),
            operator_resource_check=check)


@pytest.mark.parametrize("policy,enabled,expected", [("Require GPU", True, 70),
    ("Auto", True, 70), ("Prefer GPU", True, 70), ("CUDA GPU", True, 70),
    ("CPU", True, 80), ("Numba CPU", True, 80), ("Auto", False, 80)])
def test_carrier_refinement_plans_host_and_residual_device_partitions_separately(monkeypatch, policy, enabled, expected):
    from temsim.physics import potential_wave_carrier as carrier
    import temsim.physics.wave_device as device
    from test_potential_wave_carrier import _wave, _snapshot, ENERGY_EV
    monkeypatch.setattr(device, "cupy_module", lambda: pytest.fail("numeric grid planning must not probe CUDA"))
    source = _wave(n=64)
    original = source.amplitude.copy()
    visited = []
    class ReachedAdmittedResidual(RuntimeError):
        pass
    def residual(snapshot, wave, *args, **kwargs):
        visited.append(wave.amplitude.shape)
        if len(visited) == 1:
            raise WaveSamplingError("resolved-operator retry fixture", 1.001)
        raise ReachedAdmittedResidual("candidate passed resource planning")
    monkeypatch.setattr(carrier, "_residual", residual)
    with pytest.raises(ReachedAdmittedResidual):
        carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-6), energy_ev=ENERGY_EV,
            residual_compute_backend=policy,
            grid_numerics=WaveGridNumerics(maximum_working_bytes=2**30,
                maximum_device_working_bytes=448*72*72, acceleration_enabled=enabled))
    assert visited == [(64, 64), (expected, expected)]
    np.testing.assert_array_equal(source.amplitude, original)


def test_required_residual_grid_cannot_borrow_unused_host_memory():
    from temsim.physics.potential_wave_carrier import _residual_grid_resource_checks
    from temsim.physics.wave_grid import WaveGridBudgetError
    numerics = WaveGridNumerics(maximum_working_bytes=2**30, maximum_device_working_bytes=448*70*70)
    required, preferred = _residual_grid_resource_checks(numerics, "Require GPU")
    assert preferred is None
    required((70, 70))
    with pytest.raises(WaveGridBudgetError, match="Required residual device budget exceeded"):
        required((71, 71))
    required, preferred = _residual_grid_resource_checks(numerics, "Auto")
    assert required is None
    assert preferred((70, 70)) and not preferred((71, 71))
