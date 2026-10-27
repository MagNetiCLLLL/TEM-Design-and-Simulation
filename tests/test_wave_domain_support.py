"""Nonperiodic physical-domain checks, independent of finite-FFT unitarity."""
import numpy as np
import pytest

from temsim.physics.cpu_angular_spectrum import conservative_angular_spectrum
from temsim.physics.multiplane_wave import (
    PlaneWave, _input_boundary_probability, _sampled_angular_spectrum,
    propagate_plane_wave,
)
from temsim.physics.wave_grid import WaveGridNumerics


def split_packet():
    """Two resolved angular peaks in a displaced Gaussian, not a fitted source."""
    n, pitch, sigma, centre, frequency = 256, 1e-9, 4e-9, 98e-9, .24e9
    axis = (np.arange(n)-n//2)*pitch
    y, x = np.meshgrid(axis, axis, indexing="ij")
    value = np.exp(-((x-centre)**2+y*y)/(4*sigma*sigma))*np.cos(2*np.pi*frequency*x)
    norm = np.linalg.norm(value)
    wave = PlaneWave((value/norm).astype(complex), np.eye(2)*pitch, np.zeros(2))
    return wave, sigma, centre, frequency, norm


def drift_map(distance):
    result = np.eye(4)
    result[:2, 2:] = np.eye(2)*distance
    return result


def test_conservative_drift_expands_displaced_support_and_matches_analytic_wave():
    wave, sigma, centre, frequency, norm = split_packet()
    wavelength, distance = 1.968748899648993e-12, 31.5e-6
    before = wave.amplitude.copy()
    assert _input_boundary_probability(before) < 1e-12
    # The old full-band criterion admits this step, despite its spatial exit.
    assert distance*wavelength/(wave.basis_m[0, 0]**2*wave.amplitude.shape[0]) < .25
    records = []
    output = propagate_plane_wave(wave, drift_map(distance), np.zeros(4), wavelength,
        grid_numerics=WaveGridNumerics(maximum_pixels=1024, maximum_working_bytes=512*1024**2),
        sampling_records=records)
    assert output.amplitude.shape[0] > wave.amplitude.shape[0]
    assert output.probability == pytest.approx(wave.probability, rel=3e-14)
    assert _input_boundary_probability(output.amplitude) < output.probability*1e-12
    np.testing.assert_array_equal(wave.amplitude, before)
    assert records[0]["previous_output_boundary_probability"] > 1e-4
    assert records[0]["previous_input_displacement_margin_probability"] > 0.

    # Independent continuous paraxial Gaussian solution with both angular
    # components, including their full complex phase and physical position.
    x, y = output.coordinates_m()
    width = 1+1j*wavelength*distance/(4*np.pi*sigma*sigma)
    expected = sum(np.exp(2j*np.pi*f*x-1j*np.pi*wavelength*distance*f*f)
        *np.exp(-((x-centre-wavelength*distance*f)**2+y*y)/(4*sigma*sigma*width))/width
        for f in (-frequency, frequency))/(2*norm)
    actual = output.full_amplitude(wavelength)
    assert np.linalg.norm(actual-expected) < 2e-6
    assert float(np.sum(x*abs(actual)**2)) == pytest.approx(centre, abs=1e-13)

    # Independently zero-extend the very same executed quadrature cells.
    enlarged = np.zeros(output.amplitude.shape, complex)
    start = np.array(enlarged.shape)//2-np.array(wave.amplitude.shape)//2
    enlarged[start[0]:start[0]+256, start[1]:start[1]+256] = before
    reference = conservative_angular_spectrum(
        PlaneWave(enlarged, wave.basis_m, wave.origin_m), np.eye(2)*distance, wavelength)
    np.testing.assert_allclose(actual, reference, atol=5e-15, rtol=2e-11)


@pytest.mark.parametrize("numerics", [
    WaveGridNumerics(maximum_pixels=256),
    # Below the minimum complete padded work space; 40 MiB now legitimately
    # admits the smaller full-band candidate and must not be called a failure.
    WaveGridNumerics(maximum_pixels=1024, maximum_working_bytes=24*1024**2),
    WaveGridNumerics(automatic_refinement=False),
])
def test_exhausted_domain_cannot_publish_wrapped_output_when_expansion_unavailable(numerics):
    wave, *_ = split_packet()
    before, records = wave.amplitude.copy(), [{"prior": "complete"}]
    with pytest.raises(ValueError, match="domain extension.*Collins alternate failed"):
        propagate_plane_wave(wave, drift_map(31.5e-6), np.zeros(4), 1.968748899648993e-12,
            grid_numerics=numerics, sampling_records=records)
    np.testing.assert_array_equal(wave.amplitude, before)
    assert records == [{"prior": "complete"}]


@pytest.mark.parametrize("conservative", [False, True])
def test_dense_route_checks_physical_support_even_when_phase_step_is_conservative(conservative):
    wave, *_ = split_packet()
    diagnostic = {}
    assert _sampled_angular_spectrum(wave, np.eye(2)*31.5e-6, 1.968748899648993e-12,
        conservative=conservative, diagnostic=diagnostic) is None
    assert diagnostic["failure"] == "output_boundary"
    assert diagnostic["output_boundary_probability"] > 1e-4


def test_zero_drift_keeps_a_nonzero_edge_without_spurious_domain_rejection():
    wave = PlaneWave(np.ones((32, 32), complex)/32, np.eye(2)*1e-9, np.zeros(2))
    output = propagate_plane_wave(wave, np.eye(4), np.zeros(4), 2e-12,
        grid_numerics=WaveGridNumerics(maximum_pixels=32))
    np.testing.assert_array_equal(output.amplitude, wave.amplitude)
    diagnostic = {}
    assert _sampled_angular_spectrum(wave, np.zeros((2, 2)), 2e-12,
        conservative=True, diagnostic=diagnostic) is wave.amplitude
    assert diagnostic == {}


def test_cuda_conservative_domain_expansion_matches_cpu_complex_field():
    from dataclasses import replace
    from test_wave_device import _cuda
    from temsim.physics.wave_device import device_scope
    cp = _cuda()
    wave, *_ = split_packet()
    wavelength, distance = 1.968748899648993e-12, 31.5e-6
    numerics = WaveGridNumerics(maximum_pixels=1024, maximum_working_bytes=512*1024**2)
    expected = propagate_plane_wave(wave, drift_map(distance), np.zeros(4), wavelength,
        grid_numerics=numerics)
    with device_scope("Require GPU", maximum_working_bytes=512*1024**2):
        device_wave = replace(wave, amplitude=cp.asarray(wave.amplitude))
        before = device_wave.amplitude.copy()
        result = propagate_plane_wave(device_wave, drift_map(distance), np.zeros(4), wavelength,
            grid_numerics=numerics)
        assert result.amplitude.shape == expected.amplitude.shape
        np.testing.assert_allclose(cp.asnumpy(result.amplitude), expected.amplitude, atol=2e-14, rtol=2e-10)
        cp.testing.assert_array_equal(device_wave.amplitude, before)
        assert _input_boundary_probability(result.amplitude) < 1e-12


def test_actual_local_column_expands_window_without_restoring_or_losing_weight():
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
    from temsim.physics.column_wave import _prepare_column, _propagate_column
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState, WaveMode
    state = default_state()
    for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
        for component in collection:
            component.enabled = False
    state.apertures = []
    wave, *_ = split_packet()
    captured = TipGunCheckpoint(BeamState((WaveMode(wave, .6, TIP_REFERENCE,
        "synthetic-resolved-two-direction", 300.),), TIP_REFERENCE), 2000., 1e-9,
        {"fixture": "actual local column; no executed gun qualification"})
    prepared = _prepare_column(state, 2000., 2000.0315, .0315)
    result = _propagate_column(state, captured, 2000.0315, maximum_step_mm=.0315,
        grid_numerics=WaveGridNumerics(maximum_pixels=1024, maximum_working_bytes=512*1024**2),
        _prepared=prepared)
    mode = result.beam.modes[0]
    assert mode.plane.amplitude.shape[0] > wave.amplitude.shape[0]
    assert mode.weight_per_reference_electron == pytest.approx(.6, rel=3e-14)
    assert _input_boundary_probability(mode.plane.amplitude) < 1e-12
    rows = result.record["modes"][0]["grid_refinements"]
    assert any(row["method"].startswith("spatial zero extension") for row in rows)


@pytest.mark.parametrize("shape,shear", [((65, 96), .3), ((80, 63), -.2), ((64, 96), .0)])
def test_minimum_domain_retains_odd_rectangular_affine_cells_and_matches_large_grid(shape, shear, monkeypatch):
    from temsim.physics import multiplane_wave as implementation
    ny, nx = shape
    iy, ix = np.meshgrid(np.arange(ny)-ny//2, np.arange(nx)-nx//2, indexing="ij")
    # A resolved Gaussian with negligible Nyquist content, displaced far
    # enough to require a full-band margin but not a truncated source tail.
    value = np.exp(-((ix-(nx//2-27))**2+(iy-(ny//2-27))**2)/(4*3.**2))
    value = value*np.exp(2j*np.pi*(.12*ix-.10*iy))
    value = value/np.linalg.norm(value)
    basis = 1e-9*np.array(((1.0, shear), (.15, 1.1)))
    origin, curvature, tilt = np.array((2e-8, -1e-8)), np.diag((5e3, -3e3)), np.array((1e-5, -2e-5))
    wave = PlaneWave(value, basis, origin, curvature, tilt)
    original = wave.amplitude.copy()
    wavelength, distance = 2e-12, 25e-6
    transform = drift_map(distance)
    transform[:2, 2:] = np.diag((distance, .45*distance))
    sampled, actual_sample = [], implementation._sampled_angular_spectrum

    def inspect_input(candidate, *args, **kwargs):
        if candidate.amplitude.shape != shape:
            py, px = np.array(candidate.amplitude.shape)//2-np.array(shape)//2
            np.testing.assert_array_equal(candidate.amplitude[py:py+ny, px:px+nx], original)
            np.testing.assert_array_equal(candidate.basis_m, basis)
            np.testing.assert_array_equal(candidate.origin_m, origin)
            np.testing.assert_array_equal(candidate.curvature_m1, curvature)
            np.testing.assert_array_equal(candidate.tilt_rad, tilt)
            np.testing.assert_array_equal(candidate.coordinates_m()[:, py:py+ny, px:px+nx], wave.coordinates_m())
            sampled.append(candidate.amplitude.shape)
        return actual_sample(candidate, *args, **kwargs)

    numerics = WaveGridNumerics(angular_domain_padding="minimum_full_band", maximum_pixels=1024, maximum_working_bytes=512*1024**2)
    records = []
    with monkeypatch.context() as patch:
        patch.setattr(implementation, "_sampled_angular_spectrum", inspect_input)
        output = propagate_plane_wave(wave, transform, np.zeros(4), wavelength,
            grid_numerics=numerics, sampling_records=records)
    assert records and sampled
    row = records[0]
    actual_shape = np.array(output.amplitude.shape)
    start_y, start_x = actual_shape//2-np.array(shape)//2
    sides = np.array(((start_x, actual_shape[1]-nx-start_x), (start_y, actual_shape[0]-ny-start_y)))
    np.testing.assert_array_equal(row["domain_side_margins_cells_xy"], sides)
    assert np.all(np.min(sides, axis=1) >= np.ceil(np.array(row["maximum_group_displacement_cells_xy"])+2))
    assert all(count % 2 == 0 for count in actual_shape)
    assert actual_shape[0] != actual_shape[1]
    assert actual_shape[0] < 2*ny and actual_shape[1] < 2*nx
    np.testing.assert_array_equal(wave.amplitude, original)

    # The independent, generously enlarged input retains exactly the same
    # executed cells. The public canonical map supplies the identical action
    # and continuous/principal lift; no fitted piston or norm correction.
    large_shape = tuple(4*count for count in shape)
    enlarged = np.zeros(large_shape, complex)
    by, bx = np.array(large_shape)//2-np.array(shape)//2
    enlarged[by:by+ny, bx:bx+nx] = original
    large_input = PlaneWave(enlarged, basis, origin, curvature, tilt)
    reference = propagate_plane_wave(large_input, transform, np.zeros(4), wavelength, grid_numerics=numerics)
    for field in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(output, field), getattr(reference, field))
    embedded = np.zeros(reference.amplitude.shape, complex)
    oy, ox = np.array(embedded.shape)//2-np.array(output.amplitude.shape)//2
    embedded[oy:oy+actual_shape[0], ox:ox+actual_shape[1]] = output.amplitude
    assert np.linalg.norm(embedded-reference.amplitude) < 3e-7
    assert output.probability == pytest.approx(wave.probability, rel=3e-14)
    assert _input_boundary_probability(output.amplitude) < output.probability*1e-12


def test_domain_candidate_retries_only_failed_output_admission_with_full_band(monkeypatch):
    from temsim.physics import multiplane_wave as implementation
    wave, *_ = split_packet()
    actual, shapes = implementation._sampled_angular_spectrum, []

    def reject_first(candidate, *args, diagnostic, **kwargs):
        shapes.append(candidate.amplitude.shape)
        assert kwargs["full_band"] is True and kwargs["conservative"] is False
        if len(shapes) == 1:
            diagnostic.update(failure="output_boundary", output_boundary_probability=1e-3)
            return None
        return actual(candidate, *args, diagnostic=diagnostic, **kwargs)

    monkeypatch.setattr(implementation, "_sampled_angular_spectrum", reject_first)
    records = []
    enlarged, value = implementation._extend_angular_spectrum_domain(wave, np.eye(2)*31.5e-6,
        1.968748899648993e-12, {"output_boundary_probability": 1e-3},
        numerics=WaveGridNumerics(angular_domain_padding="minimum_full_band", maximum_pixels=1024), sampling_records=records)
    assert len(shapes) == 2
    assert np.all(np.array(shapes[1]) >= 1.25*np.array(shapes[0]))
    assert records[0]["attempted_shapes"] == shapes
    assert _input_boundary_probability(value) < 1e-12
    assert enlarged.amplitude.shape == shapes[-1]


def test_minimum_domain_checks_budget_before_padding_allocation(monkeypatch):
    from temsim.physics import multiplane_wave as implementation
    wave, *_ = split_packet()
    before = wave.amplitude.copy()
    monkeypatch.setattr(implementation.np, "zeros", lambda *args, **kwargs: pytest.fail("Padding allocation preceded its budget check"))
    with pytest.raises(implementation._AngularSpectrumDomainUnavailable, match="budget exceeded"):
        implementation._extend_angular_spectrum_domain(wave, np.eye(2)*31.5e-6, 2e-12,
            {"output_boundary_probability": 1e-3}, numerics=WaveGridNumerics(angular_domain_padding="minimum_full_band", maximum_pixels=256))
    np.testing.assert_array_equal(wave.amplitude, before)


@pytest.mark.parametrize("cancel_after_allocation", [False, True])
def test_minimum_domain_cancellation_preserves_original_and_does_not_publish_records(cancel_after_allocation):
    from temsim.physics import multiplane_wave as implementation
    wave, *_ = split_packet()
    before, records, calls = wave.amplitude.copy(), [{"prior": "complete"}], []

    def cancelled():
        calls.append(None)
        return len(calls) >= (3 if cancel_after_allocation else 1)

    with pytest.raises(InterruptedError, match="domain extension cancelled"):
        implementation._extend_angular_spectrum_domain(wave, np.eye(2)*31.5e-6, 2e-12,
            {"output_boundary_probability": 1e-3}, numerics=WaveGridNumerics(angular_domain_padding="minimum_full_band", maximum_pixels=1024),
            cancelled=cancelled, sampling_records=records)
    np.testing.assert_array_equal(wave.amplitude, before)
    assert records == [{"prior": "complete"}]


@pytest.mark.parametrize("pitch,distance", [(1e-9, 1e8), (1e-150, 1.), (1e-150, 1e100)])
def test_extreme_domain_rejects_unrepresentable_or_over_budget_candidate_before_fft_planning(pitch, distance, monkeypatch):
    from scipy import fft
    from temsim.physics import multiplane_wave as implementation
    wave = PlaneWave(np.ones((8, 8), complex)/8, np.eye(2)*pitch, np.zeros(2))
    original = wave.amplitude.copy()
    monkeypatch.setattr(fft, "next_fast_len", lambda *args, **kwargs: pytest.fail("Oversized chart reached the native FFT planner"))
    with pytest.raises(implementation._AngularSpectrumDomainUnavailable, match="budget exceeded|unrepresentable"):
        implementation._extend_angular_spectrum_domain(wave, np.eye(2)*distance, 2e-12,
            {"output_boundary_probability": 1.}, numerics=WaveGridNumerics(angular_domain_padding="minimum_full_band", maximum_pixels=1024))
    np.testing.assert_array_equal(wave.amplitude, original)


def test_unavailable_extreme_domain_preserves_existing_collins_alternate_route(monkeypatch):
    from scipy import fft
    from temsim.physics import multiplane_wave as implementation
    wave, *_ = split_packet()
    calls = []

    # Isolate alternate-chart dispatch from the original chart's spectral
    # admission. A finite but enormous requested domain must be classified
    # as unavailable, rather than leaking OverflowError from native planning.
    def unavailable_support(*args, diagnostic, **kwargs):
        diagnostic.update(failure="output_boundary", output_boundary_probability=1.)
        return None

    def alternate(candidate, *args, **kwargs):
        calls.append(candidate)
        return candidate, 0.

    monkeypatch.setattr(implementation, "_sampled_angular_spectrum", unavailable_support)
    monkeypatch.setattr(implementation, "_propagate_fresnel_chart", alternate)
    monkeypatch.setattr(fft, "next_fast_len", lambda *args, **kwargs: pytest.fail("Oversized chart reached FFT planning"))
    rows = []
    output = propagate_plane_wave(wave, drift_map(1e8), np.zeros(4), 2e-12,
        grid_numerics=WaveGridNumerics(angular_domain_padding="minimum_full_band", maximum_pixels=1024), sampling_records=rows)
    assert calls == [wave]
    assert output.probability == pytest.approx(wave.probability, rel=3e-14)
    assert "Collins alternate" in rows[-1]["method"]


def test_failed_domain_releases_expanded_wave_before_collins_fallback(monkeypatch):
    """A failed optional chart must not pin its arrays through a traceback."""
    import weakref
    from temsim.physics import multiplane_wave as implementation
    wave, *_ = split_packet()
    abandoned = []

    def rejected_extension(candidate, *args, diagnostic, **kwargs):
        assert candidate.amplitude.shape == (512, 512)
        abandoned.append(weakref.ref(candidate.amplitude))
        diagnostic.update(failure="output_boundary", output_boundary_probability=1.)
        return None

    def alternate(candidate, *args, **kwargs):
        assert abandoned and all(reference() is None for reference in abandoned)
        assert candidate is wave
        return candidate, 0.

    monkeypatch.setattr(implementation, "_sampled_angular_spectrum", rejected_extension)
    monkeypatch.setattr(implementation, "_propagate_fresnel_chart", alternate)
    rows = []
    output = propagate_plane_wave(wave, drift_map(31.5e-6), np.zeros(4), 2e-12,
        grid_numerics=WaveGridNumerics(maximum_pixels=512), sampling_records=rows)
    assert output.probability == pytest.approx(wave.probability, rel=3e-14)
    assert "Collins alternate" in rows[-1]["method"]
    assert "budget exceeded" in rows[-1]["reason"]
    assert rows[-1]["angular_spectrum"]["attempted_shapes"] == [(512, 512)]


@pytest.mark.parametrize("value", [None, True, 1, "minimum", "Doubling", ["doubling"]])
def test_domain_padding_policy_rejects_unknown_or_non_string_values(value):
    with pytest.raises(ValueError, match="Angular domain padding"):
        WaveGridNumerics(angular_domain_padding=value).validate()


def test_domain_padding_policy_is_serialized_and_part_of_column_identity():
    from dataclasses import asdict, replace
    original = WaveGridNumerics()
    minimum = replace(original, angular_domain_padding="minimum_full_band")
    assert original.angular_domain_padding == "doubling"
    assert asdict(minimum)["angular_domain_padding"] == "minimum_full_band"
    assert WaveGridNumerics(**asdict(minimum)) == minimum
    assert original.column_identity() != minimum.column_identity()
    wave, *_ = split_packet()
    rows = []
    doubled = propagate_plane_wave(wave, drift_map(31.5e-6), np.zeros(4), 1.968748899648993e-12,
        grid_numerics=original, sampling_records=rows)
    assert doubled.amplitude.shape == (512, 512)
    assert rows[0]["angular_domain_padding"] == "doubling"
    rows = []
    smaller = propagate_plane_wave(wave, drift_map(31.5e-6), np.zeros(4), 1.968748899648993e-12,
        grid_numerics=minimum, sampling_records=rows)
    assert smaller.amplitude.size < doubled.amplitude.size
    assert rows[0]["angular_domain_padding"] == "minimum_full_band"
