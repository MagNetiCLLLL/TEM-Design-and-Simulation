"""Independent finite-depth atomic Fourier checks; no source or wave trial."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from temsim.physics import finite_lobato_fourier as atomic


def test_finite_depth_matches_independent_real_space_hankel_and_z_quadrature():
    from abtem.parametrizations import LobatoParametrization
    from abtem.parametrizations.functions.lobato import potential
    from scipy.integrate import quad_vec
    from scipy.special import j0
    a, mu = LobatoParametrization().scaled_parameters("Si", "potential")
    q = np.array([0., .5, 1.9, 7.6])
    parameters = np.stack((a, mu))
    radius = 40/min(mu)
    for lower, upper in ((-.0625, .0625), (0., .125), (.25, .375), (.7, .825)):
        def radial(rho):
            def real_space(z):
                return potential(float(np.hypot(rho, z)), parameters)
            integral, _ = quad(real_space, lower, upper, epsabs=2e-8, epsrel=2e-10,
                points=[0.] if lower < 0 < upper else None, limit=180)
            return 2*np.pi*rho*integral*j0(2*np.pi*q*rho)
        oracle, quadrature_error = quad_vec(radial, 0., radius, epsabs=2e-7, epsrel=2e-10, limit=1000)
        actual = atomic.finite_lobato_fourier_coefficients(q*q, lower, upper, a, mu)
        # The far-z/high-q integral is nearly zero. Use absolute/DC-scaled
        # error, rather than reporting a misleading tiny-denominator ratio.
        assert quadrature_error < 1e-6
        assert np.max(abs(actual-oracle))/abs(oracle[0]) < 2e-8


def test_full_depth_limit_has_the_original_lobato_normalization_and_units():
    a, mu = np.array([2.5, -.3]), np.array([3., 5.])
    q2 = np.array([0., .25, 1., 25.])
    beta2 = mu[:, None]**2+4*np.pi**2*q2
    oracle = np.sum(8*np.pi*a[:, None]*(1/(mu[:, None]*beta2)+mu[:, None]/beta2**2), axis=0)
    actual = atomic.finite_lobato_fourier_coefficients(q2, -100., 100., a, mu)
    np.testing.assert_allclose(actual, oracle, rtol=3e-15, atol=1e-15)
    assert actual[0] == pytest.approx(float(np.sum(16*np.pi*a/mu**3)), rel=3e-15)
    half = atomic.finite_lobato_fourier_coefficients(q2, 0., 100., a, mu)
    np.testing.assert_allclose(half*2, oracle, rtol=3e-15, atol=1e-15)


def test_broadcast_finite_bounds_are_even_additive_and_include_no_infinite_z_atom():
    a, mu = np.array([2.5]), np.array([3.])
    q2 = np.array([0., .25])[:, None]
    lower = np.array([-.2, 0., .4])[None]
    upper = np.array([.1, .125, .6])[None]
    result = atomic.finite_lobato_fourier_coefficients(q2, lower, upper, a, mu)
    mirrored = atomic.finite_lobato_fourier_coefficients(q2, -upper, -lower, a, mu)
    np.testing.assert_array_equal(result, mirrored)
    parts = sum(atomic.finite_lobato_fourier_coefficients(q2, lo, hi, a, mu)
        for lo, hi in ((-.2, 0.), (0., .05), (.05, .1)))
    np.testing.assert_allclose(parts[:, 0], result[:, 0], rtol=3e-15, atol=1e-15)
    np.testing.assert_array_equal(atomic.finite_lobato_fourier_coefficients(q2, .1, .1, a, mu), np.zeros((2, 1)))


@pytest.mark.parametrize("q2,lo,hi", [(-1., 0., 1.), (np.nan, 0., 1.), (0., 1., 0.), (0., 0., np.inf)])
def test_invalid_finite_integrals_are_rejected(q2, lo, hi):
    with pytest.raises(ValueError):
        atomic.finite_lobato_fourier_coefficients(q2, lo, hi, [1.], [2.])


@pytest.mark.parametrize("lower,upper", [(0., .0625), (.7, .825), (49.9, 50.), (-.1, .1)])
def test_finite_chain_closed_sum_matches_each_original_atom(lower, upper):
    a, mu = np.array([2.5, -.2]), np.array([3., 5.])
    q2 = np.array([0., .1, 1., 64.])
    z = .13+np.arange(14)*3.84927886
    oracle = sum(atomic.finite_lobato_fourier_coefficients(q2, lower-value, upper-value, a, mu) for value in z)
    actual = atomic._chain(q2, lower, upper, z, a, mu, xp=np)
    np.testing.assert_allclose(actual, oracle, rtol=8e-14, atol=1e-14)


def test_xy_image_certificate_bounds_explicit_images_and_scales_with_original_atoms():
    a, mu = np.array([2.5, -.2]), np.array([.7, 2.])
    params = {14: (a, mu)}
    # One actual atom at the box's +X face; the nearest image is one A from
    # the ROI point. No additional atom is inserted in the physical specimen.
    total = 0.
    for iy in range(-20, 21):
        for ix in range(-20, 21):
            if ix == iy == 0:continue
            distance = np.hypot(9.+ix*10., iy*10.)
            total += float(np.sum(abs(a)*(1+2/(mu*distance))*np.exp(-mu*distance)))
    bound = atomic.periodic_image_potential_bound(np.array([14]), params, 1., 10.)
    assert 0 < total <= bound
    assert atomic.periodic_image_potential_bound(np.array([14, 14]), params, 1., 10.) == 2*bound
    assert atomic.periodic_image_potential_bound(np.array([14]), params, 2., 10.) < bound


def test_finite_structure_factors_use_actual_selected_atoms_and_the_correct_fourier_origin(monkeypatch):
    from ase import Atoms
    from temsim.specimen import atomistic
    selected_xy = np.array([[-2., 0.], [0., -2.], [0., 0.], [0., 2.], [2., 0.]])
    origin = np.array([.3, -.2])
    seen = {}
    def sites(path, **kwargs):
        assert kwargs["specimen_envelope_shape"] == "disk"
        size = np.asarray(kwargs["field_of_view_angstrom"])
        positions = np.c_[selected_xy+origin+size/2, np.full(5, 5.)]
        seen["positions"] = positions
        return Atoms(numbers=np.full(5, 14), positions=positions, cell=(*size, 10.), pbc=False), (4., 4., 4.)
    monkeypatch.setattr(atomistic, "build_cif_equilibrium_atoms", sites)
    def build(groups, extent, shape, count, length, *, xp, output, **kwargs):
        fx, fy = np.meshgrid(np.fft.fftfreq(shape[1], extent[0]/shape[1]), np.fft.fftfreq(shape[0], extent[1]/shape[0]))
        expected = sum(np.exp(-2j*np.pi*(fx*row[0]+fy*row[1])) for row in seen["positions"])
        assert sum(len(group.z)*group.atoms_per_plane for group in groups) == 5
        # The independent direct exponent and count-FFT origin factor use
        # different argument reductions at the padded-box edge. This is an
        # absolute float64 roundoff check, not a physical tolerance.
        np.testing.assert_allclose(sum(group.structure for group in groups), expected, rtol=0, atol=2e-13)
        for index in range(count):output(index, np.zeros(shape, np.float32))
        return np.full(count, length/count), 0., 0., fx*fx+fy*fy
    monkeypatch.setattr(atomic, "_build", build)
    scene = SimpleNamespace(cif_path="synthetic.cif", thickness_nm=1., size_xy_nm=(1., 1.),
        centre_xy_nm=tuple(origin/10), envelope_shape="disk", orientation_quaternion_wxyz=(1.,0.,0.,0.))
    crystal = (np.array([4.,4.,4.]), np.zeros((1,3)), np.array([14]), {})
    array, _, record = atomic.build_finite_lobato_potential(scene, crystal, (8.,8.), (8,8), 2,
        maximum_working_bytes=4*1024**2, xp=np)
    assert array.shape == (2,8,8)
    assert record["atom_count"] == 5
    assert record["physical_atomic_site_selection_before_potential"] is True
    assert record["potential_pixel_envelope_mask_applied"] is False
    # The image remainder must hold on the whole Fourier cell before
    # truncation. Padding the observation ROI alone would not prove this.
    assert record["padding_effective_angstrom"] >= 20.
    extent = np.asarray(record["padded_fov_angstrom_xy"])
    actual_sites = seen["positions"][:, :2]-extent/2
    assert record["padding_effective_angstrom"] == pytest.approx(
        float(np.min(extent/2-abs(actual_sites))), abs=1e-14)
    assert record["fourier_projection_linf_stability_bound"] == np.prod(record["padded_grid_yx"])
