"""Angular quadrature refinement retains every represented complex tip cell."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.accelerating_tip_boundary import (
    AcceleratingTipNumerics, _zero_extend_tip_source, scalar_robin_transport,
)
from temsim.physics.tip_gun_wave import GunWaveNumerics
from temsim.physics.wave_grid import WaveGridNumerics, WaveMemoryBudgetError


def source():
    axis = np.arange(32)-16
    y, x = np.meshgrid(axis, axis, indexing="ij")
    # An asymmetric complex fixture ensures padding cannot pass by fitting a
    # circular intensity or reconstructing a different zero-phase source.
    return np.exp(-(x*x+y*y)/12+1j*(.17*x-.12*y+.011*x*y))*(1+.03j*x)


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_padding_preserves_cells_and_matches_independent_finite_fourier_sum(factor):
    original = source()
    padded, record = _zero_extend_tip_source(original, factor, grid_numerics=WaveGridNumerics())
    n = 32*factor
    lower = n//2-16
    np.testing.assert_array_equal(padded[lower:lower+32, lower:lower+32], original)
    assert np.sum(abs(padded)**2) == pytest.approx(np.sum(abs(original)**2), rel=2e-15)
    index = np.arange(n)-n//2
    positions = np.arange(32)-16
    kernel = np.exp(-2j*np.pi*np.outer(index, positions)/n)
    expected = kernel@original@kernel.T/n
    actual = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded), norm="ortho"))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=1e-10)
    assert record["original_shape"] == (32, 32)
    assert record["padded_shape"] == (n, n)
    assert record["discarded_represented_probability"] == 0.
    assert record["normalisation_correction"] is False


def test_dense_angular_transfer_matches_independent_uniform_medium_boundary():
    # Every new angular sample solves the same physical Robin problem; the
    # transmitted coarse field is never interpolated to create these values.
    original = source()
    padded, _ = _zero_extend_tip_source(original, 2, grid_numerics=WaveGridNumerics())
    n = len(padded)
    k = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(n))
    transverse = k[:, None]**2+k[None, :]**2
    normal, distance = 4., .17
    logarithm, _, kz, _ = scalar_robin_transport(
        [normal**2], [distance], normal**2, transverse, normal)
    actual_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded), norm="ortho"))
    actual = actual_spectrum*np.exp(logarithm)/np.sqrt(normal)
    kernel = np.exp(-1j*np.outer(k, np.arange(32)-16))
    independent_spectrum = kernel@original@kernel.T/n
    expected = independent_spectrum*(2*normal/(normal+kz))*np.exp(1j*kz*distance)/np.sqrt(normal)
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=1e-10)


def test_padding_rejects_unresolved_source_edges_without_inventing_zero_exterior():
    original = np.ones((32, 32), complex)
    with pytest.raises(ValueError, match="resolved input domain"):
        _zero_extend_tip_source(original, 2, grid_numerics=WaveGridNumerics())
    np.testing.assert_array_equal(original, np.ones((32, 32), complex))


def test_padding_budget_includes_previous_modes_before_allocation(monkeypatch):
    original = source()
    budget = 512*64**2+original.nbytes
    def forbidden(*args, **kwargs):
        pytest.fail("A padded source was allocated before its retained-memory check")
    monkeypatch.setattr("temsim.physics.accelerating_tip_boundary.np.zeros", forbidden)
    with pytest.raises(WaveMemoryBudgetError):
        _zero_extend_tip_source(original, 2,
            grid_numerics=WaveGridNumerics(maximum_working_bytes=budget), retained_bytes=1)


def test_padding_cancellation_precedes_allocation():
    with pytest.raises(InterruptedError):
        _zero_extend_tip_source(source(), 2, grid_numerics=WaveGridNumerics(), cancelled=lambda: True)


@pytest.mark.parametrize("factor", [0, 17, True, 1.5])
def test_padding_numerics_are_explicit_and_bounded(factor):
    with pytest.raises(ValueError, match="source_padding_factor"):
        AcceleratingTipNumerics(source_padding_factor=factor).validate()
    with pytest.raises(ValueError, match="source_padding_factor"):
        GunWaveNumerics(near_tip_source_padding_factor=factor).validate()


def test_padding_is_a_numerical_control_not_an_emission_width():
    assert AcceleratingTipNumerics().source_padding_factor == 2
    assert GunWaveNumerics().near_tip_source_padding_factor == 2
    assert replace(AcceleratingTipNumerics(), source_padding_factor=4).validate().source_padding_factor == 4


def test_robin_budget_includes_caller_arrays_before_operator_invocation(monkeypatch):
    import temsim.physics.accelerating_tip_boundary as boundary
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.wave_flux import WaveMode
    from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, TipWaveNumerics
    amplitude = source()
    amplitude /= np.linalg.norm(amplitude)
    mode = WaveMode(PlaneWave(amplitude, np.eye(2)*1e-9, np.zeros(2)),
                    1., TIP_REFERENCE, "fixture", 300.)
    emission = SimpleNamespace(digest="fixture", numerics=TipWaveNumerics(),
        record={"energy_modes": [{"energy_ev": 300_000.}], "mode_count": 1},
        reference_current_a=1e-9, modes=lambda: iter((mode,)))
    electric = SimpleNamespace(physical_identity="p", numerical_identity="n", request_identity="r",
        base_field=SimpleNamespace(r=np.array([0., 1e-6]), z=np.array([0., 4e-6])),
        potential_rise_v_at_global_positions=lambda p: np.zeros(p.shape[:-1]))
    monkeypatch.setattr("temsim.optics.electron_gun.tip_coherence.generate_tip_boundary_emission",
                        lambda *args: emission)
    monkeypatch.setattr("temsim.physics.instrument_electric.capture_instrument_electric_field", lambda state: electric)
    monkeypatch.setattr("temsim.physics.instrument_magnetic.capture_instrument_magnetic_field",
                        lambda state: SimpleNamespace(numerical_identity="m"))
    monkeypatch.setattr("temsim.instrument_snapshot.encode_instrument", lambda state: {})
    monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: "fixture")
    monkeypatch.setattr("temsim.physics.tip_gun_wave._refine_energy_grid", lambda grid, *args: grid)
    monkeypatch.setattr(boundary, "_near_field_domain", lambda *args: ())
    monkeypatch.setattr(boundary, "scalar_robin_transport",
                        lambda *args, **kwargs: pytest.fail("Robin allocated before its budget check"))
    # Padding alone fits. Its full complex input, coordinate arrays and FFT
    # spectrum must still be resident when Robin asks for its 512 B/cell.
    budget = 550*64**2
    with pytest.raises(WaveMemoryBudgetError):
        boundary.execute_accelerating_tip(SimpleNamespace(electron_gun=object()), emission, electric,
            numerics=AcceleratingTipNumerics(maximum_working_bytes=budget))


def test_near_tip_publication_reserves_originals_copies_and_largest_temporary(monkeypatch):
    import temsim.physics.accelerating_tip_boundary as boundary
    arrays = tuple(np.arange(n, dtype=complex) for n in (4, 6, 8))
    modes = (SimpleNamespace(plane=SimpleNamespace(amplitude=np.zeros((4, 4), complex))),)
    retained = 123
    required = retained+modes[0].plane.amplitude.nbytes+2*sum(a.nbytes for a in arrays)+max(a.nbytes for a in arrays)
    real_constructor = boundary.ExecutedAcceleratingTip
    monkeypatch.setattr(boundary, "ExecutedAcceleratingTip",
                        lambda *args: pytest.fail("Near-tip immutable copies allocated before their budget check"))
    with pytest.raises(WaveMemoryBudgetError, match="publication"):
        boundary._publish_accelerating_tip(modes, (arrays[0],), (arrays[1],), (arrays[2],),
            (1.,), .004, 1e-9, {}, maximum_working_bytes=required-1, retained_bytes=retained)
    monkeypatch.setattr(boundary, "ExecutedAcceleratingTip", real_constructor)
    result = boundary._publish_accelerating_tip(modes, (arrays[0],), (arrays[1],), (arrays[2],),
        (1.,), .004, 1e-9, {}, maximum_working_bytes=required, retained_bytes=retained)
    assert not result.spectral_amplitudes[0].flags.writeable
    np.testing.assert_array_equal(result.spectral_amplitudes[0], arrays[0])


def test_near_tip_publication_cancelled_before_copy(monkeypatch):
    import temsim.physics.accelerating_tip_boundary as boundary
    monkeypatch.setattr(boundary, "ExecutedAcceleratingTip",
                        lambda *args: pytest.fail("Cancelled near-tip result was constructed"))
    with pytest.raises(InterruptedError, match="publication cancelled"):
        boundary._publish_accelerating_tip((), (), (), (), (), .004, 1e-9, {},
            maximum_working_bytes=1024, cancelled=lambda: True)
