"""Array-level hybrid handoff checks, not finite-source physical trials."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.constants import e, hbar

from temsim.physics.hybrid_exit import local_diffraction, sample_exit
from temsim.physics.hybrid_semiclassical import SemiclassicalNumerics
from temsim.physics.hybrid_pipeline import HybridGrid, HybridRequest, simulate_hybrid, _source_count
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState
from temsim.physics.potential_snapshot import StaticPolynomialGauge
from test_hybrid_semiclassical import gaussian_mode
from test_potential_ray import uniform_snapshot


def options():
    return SemiclassicalNumerics(frame_stride=12, window_sigma_pixels=12.)


def test_exit_partition_keeps_absolute_mode_weights_and_energy():
    first = gaussian_mode()
    second = replace(first, mode_id="independent:1", weight_per_reference_electron=.15, energy_kev=150.)
    checkpoint = TipGunCheckpoint(BeamState((first, second), first.reference_plane), 1., 1e-9, {"schema": "array-test"})
    particles, spectra = sample_exit(checkpoint, uniform_snapshot(), 256, options(), xp=np, seed=12)
    assert particles.weight.sum() == pytest.approx(.55, abs=1e-14)
    for index, mode in enumerate((first, second)):
        selected = particles.mode_index == index
        assert particles.weight[selected].sum() == pytest.approx(mode.weight_per_reference_electron)
        np.testing.assert_array_equal(particles.energy_ev[selected], mode.energy_kev*1000)
        assert spectra[index]["transport_diffraction"].sum() == pytest.approx(mode.weight_per_reference_electron)
    assert not particles.record["zero_loss_attenuation_applied_again"]
    assert not particles.record["original_source_particles"]
    with pytest.raises(ValueError):
        particles.weight[0] = 0


def test_magnetic_gauge_changes_raw_spectrum_not_local_kinetic_distribution():
    mode = gaussian_mode()
    baseline = local_diffraction(mode, uniform_snapshot(), .001, options(), xp=np)
    # Linear gauge, one reciprocal cell: the Wilson phase cancels it exactly.
    gradient = np.array((hbar/e*2*np.pi/(48e-9), 0., 0.))
    snapshot = uniform_snapshot(gauge=StaticPolynomialGauge(gradient_t_m=gradient))
    xy = mode.plane.coordinates_m()
    phase = -e/hbar*(gradient[0]*xy[0]+gradient[1]*xy[1])
    changed = replace(mode, plane=replace(mode.plane, amplitude=mode.plane.amplitude*np.exp(1j*phase)))
    result = local_diffraction(changed, snapshot, .001, options(), xp=np)
    np.testing.assert_allclose(result[2], baseline[2], rtol=1e-11, atol=1e-16)
    assert not np.allclose(result[3]["raw_canonical_diffraction"], baseline[3]["raw_canonical_diffraction"])


def test_cuda_guard_precedes_source_or_field_work(monkeypatch):
    from temsim.physics.compute_backend import GPUExecutionError
    def unavailable(*args, **kwargs):
        raise GPUExecutionError("unavailable", "test CUDA unavailable")
    monkeypatch.setattr("temsim.physics.compute_backend.require_coherent_cuda", unavailable)
    with pytest.raises(GPUExecutionError, match="CUDA unavailable"):
        simulate_hybrid(object(), HybridRequest("Camera", HybridGrid(500., 32, 20.)))


def test_original_count_cannot_be_satisfied_by_duplicates_or_exit_resampling():
    from types import SimpleNamespace
    def trace(ids):
        return SimpleNamespace(emission_reference={"ray_id": ids}, exit_bundle=SimpleNamespace(
            weight=np.ones(len(ids))/len(ids), alive=np.ones(len(ids), bool)))
    with pytest.raises(ValueError, match="1000"):
        _source_count(trace(np.arange(999)))
    with pytest.raises(ValueError, match="duplicate"):
        _source_count(trace(np.zeros(1024)))
    assert _source_count(trace(np.arange(1024)))["positive_current_count"] == 1024


def test_physical_exit_window_is_independent_of_lattice_refinement():
    from temsim.physics.multiplane_wave import PlaneWave
    first = gaussian_mode()
    x = (np.arange(96)-48)*.5e-9
    yy, xx = np.meshgrid(x, x, indexing="ij")
    a = np.exp(-(xx*xx+yy*yy)/(4*(3e-9)**2)).astype(complex)
    a /= np.linalg.norm(a)
    finer = replace(first, plane=PlaneWave(a, np.eye(2)*.5e-9, np.zeros(2)))
    results = []
    for mode in (first, finer):
        checkpoint = TipGunCheckpoint(BeamState((mode,), mode.reference_plane), 1., 1e-9, {"schema": "array-test"})
        results.append(sample_exit(checkpoint, uniform_snapshot(), 128, options(), xp=np,
                                   seed=8, window_sigma_m=6e-9))
    for key in ("window_sigma_m", "window_count", "window_nominal_angular_rms_rad"):
        assert results[0][1][0][key] == pytest.approx(results[1][1][0][key])
    np.testing.assert_allclose(results[0][0].position_m, results[1][0].position_m, rtol=0., atol=1e-20)
    np.testing.assert_allclose(results[0][0].direction, results[1][0].direction, rtol=0., atol=1e-14)


def test_separable_windows_match_full_gaussians_on_rotated_rectangular_grid():
    from temsim.physics.hybrid_exit import _partition_factors
    from temsim.physics.hybrid_semiclassical import frame_centres, gaussian_window
    from temsim.physics.multiplane_wave import PlaneWave
    rotation = np.array(((np.cos(.37), -np.sin(.37)), (np.sin(.37), np.cos(.37))))
    plane = PlaneWave(np.zeros((24, 32), complex), rotation@np.diag((1e-9, 1.2e-9)), np.array((3e-9, -2e-9)))
    numerics = SemiclassicalNumerics(frame_stride=6, window_sigma_pixels=6.)
    points, _, centres, sigma = frame_centres(plane, numerics)
    dense = np.array([gaussian_window(points, centre, sigma) for centre in centres])
    dense /= np.sqrt(np.sum(dense*dense, axis=0))
    wx, wy = _partition_factors(plane, centres, sigma, xp=np)
    factored = np.array([(wy[iy, :, None]*wx[ix, None, :]).ravel()
                        for iy in range(len(wy)) for ix in range(len(wx))])
    np.testing.assert_allclose(factored, dense, atol=3e-15, rtol=2e-13)
    np.testing.assert_allclose(np.sum(factored**2, axis=0), 1., atol=3e-15)
