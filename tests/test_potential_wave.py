"""Independent benchmarks for the opt-in static scalar magnetic operator.

These tests do not admit an instrument wave product. They distinguish the
finite-lattice exponential, axial splitting, transverse discretisation and
small-angle kinetic expansion. Comparisons retain absolute complex phase,
centroid and flux; no phase fit, translation or intensity normalisation is
performed on propagated results.
"""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, hbar, m_e
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply
from scipy.special import roots_hermite

from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_snapshot import (
    ElectromagneticPotentialSnapshot, PotentialContribution, StaticPolynomialGauge,
)
from temsim.physics.potential_wave import propagate_plane_wave_potential, propagate_static_magnetic_wave


ENERGY_EV = 300_000.


def _wavenumber():
    kinetic = ENERGY_EV*e
    return np.sqrt(kinetic*(kinetic+2*m_e*c*c))/(c*hbar)


def _snapshot(*fields, gauge=None):
    scene = capture_physical_scene(SimpleNamespace(lenses=[
        SimpleNamespace(key=f"source:{index}", z_mm=0.) for index in range(len(fields))]))
    rows = tuple(PotentialContribution(f"magnetic:{index}", (f"source:{index}",), field, "magnetic")
                 for index, field in enumerate(fields))
    return ElectromagneticPotentialSnapshot(scene, rows,
        StaticPolynomialGauge() if gauge is None else gauge)


def _uniform_field(bz, *, half_length=1e-3):
    radial, axial = np.array((0., 1e-6)), np.array((-half_length, half_length))
    return AxisymmetricMagneticPotential(radial, axial,
        np.broadcast_to(.5*bz*radial[:, None], (2, 2)),
        provenance={"source": "manufactured uniform axial field"})


def _points(x, y, z):
    xx, yy = np.meshgrid(x, y)
    return np.stack((xx, yy, np.full_like(xx, z)), axis=-1)


def _packet(x, y, *, sigma=3e-9, flux=.37):
    xx, yy = np.meshgrid(x, y)
    values = np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma)+1j*(.37+.23*xx/sigma-.17*yy/sigma))
    # Set the source flux once. No propagated result is ever rescaled.
    return values*np.sqrt(flux/np.sum(abs(values)**2))


def _run(snapshot, initial, x, y, z, **kwargs):
    return propagate_static_magnetic_wave(snapshot, initial, x, y, z,
        energy_ev=ENERGY_EV, **kwargs)


def _dirichlet_matrix(x, y, bz=0.):
    """Independent sparse matrix from analytic straight Wilson integrals.

    Symmetric gauge A=(-B*y/2, B*x/2, 0). Unknowns are interior samples;
    zero ghost nodes sit one pitch beyond the first/last sample on each side.
    """
    nx, ny = len(x), len(y)
    dx, dy = x[1]-x[0], y[1]-y[0]
    rows, columns, values = [], [], []
    for j in range(ny):
        for i in range(nx):
            row = j*nx+i
            rows.append(row); columns.append(row); values.append(2/dx**2+2/dy**2)
            if i+1 < nx:
                link = np.exp(-1j*e*bz*y[j]*dx/(2*hbar))
                rows.extend((row, row+1)); columns.extend((row+1, row))
                values.extend((-link/dx**2, -link.conjugate()/dx**2))
            if j+1 < ny:
                link = np.exp(1j*e*bz*x[i]*dy/(2*hbar))
                rows.extend((row, row+nx)); columns.extend((row+nx, row))
                values.extend((-link/dy**2, -link.conjugate()/dy**2))
    from scipy.sparse import coo_matrix
    return coo_matrix((values, (rows, columns)), shape=(nx*ny, nx*ny)).tocsr()


def _hamiltonian(laplacian, kinetic_order=4):
    k = _wavenumber()
    result = laplacian/(2*k)
    return result if kinetic_order == 2 else result+(laplacian@laplacian)/(8*k**3)


@pytest.mark.parametrize("kinetic_order", (2, 4))
def test_vacuum_dirichlet_eigenphase_and_nonunit_flux_are_preserved(kinetic_order):
    nx, ny, pitch, distance = 31, 29, 1e-11, 1e-9
    x, y = (np.arange(nx)-(nx-1)/2)*pitch, (np.arange(ny)-(ny-1)/2)*pitch
    mx, my = 5, 4
    initial = np.outer(np.sin(np.pi*my*(np.arange(ny)+1)/(ny+1)),
                       np.sin(np.pi*mx*(np.arange(nx)+1)/(nx+1))).astype(complex)
    initial *= np.exp(.41j)*np.sqrt(3.7/np.sum(abs(initial)**2))
    eigenvalue = (4*np.sin(np.pi*mx/(2*(nx+1)))**2/pitch**2
                  +4*np.sin(np.pi*my/(2*(ny+1)))**2/pitch**2)
    k = _wavenumber()
    phase = distance*(eigenvalue/(2*k)+(eigenvalue**2/(8*k**3) if kinetic_order == 4 else 0.))
    result = _run(_snapshot(), initial, x, y, (0., distance), kinetic_order=kinetic_order,
                  boundary_fraction_limit=.95, kinetic_remainder_phase_limit=.05)
    expected = initial*np.exp(-1j*phase)
    error = np.linalg.norm(result.amplitude-expected)/np.linalg.norm(initial)
    assert error < 2e-11
    assert error <= (result.metadata["series_amplitude_error_bound"]
                     +result.metadata["roundoff_relative_allowance"])
    assert result.metadata["input_flux"] == pytest.approx(3.7, abs=2e-15)
    assert result.metadata["output_flux"] == pytest.approx(3.7, abs=2e-11)
    assert result.metadata["normalization_applied"] is False
    assert result.metadata["production_admission"] is False
    for values in (result.amplitude, result.x_m, result.y_m):
        assert values.flags.writeable is False
        with pytest.raises(ValueError):
            values.setflags(write=True)
    with pytest.raises(TypeError):
        result.metadata["production_admission"] = True


@pytest.mark.parametrize("kinetic_order", (2, 4))
def test_kinetic_remainder_bounds_the_unexpanded_forward_eigenphase(kinetic_order):
    count, pitch, distance = 15, 1e-11, 1e-9
    axis = (np.arange(count)-(count-1)/2)*pitch
    mode = np.sin(np.pi*count*(np.arange(count)+1)/(count+1))
    initial = np.outer(mode, mode).astype(complex)
    initial *= np.sqrt(1.3/np.sum(abs(initial)**2))*np.exp(.13j)
    eigenvalue = 8*np.sin(np.pi*count/(2*(count+1)))**2/pitch**2
    k = _wavenumber()
    ratio = eigenvalue/k**2
    # Stable evaluation of k-sqrt(k^2-lambda), with the same removed carrier.
    exact_phase = distance*k*ratio/(1+np.sqrt(1-ratio))
    result = _run(_snapshot(), initial, axis, axis, (0., distance), kinetic_order=kinetic_order,
                  boundary_fraction_limit=.95, kinetic_remainder_phase_limit=.05)
    expected = initial*np.exp(-1j*exact_phase)
    observed = np.linalg.norm(result.amplitude-expected)/np.linalg.norm(initial)
    budget = (result.metadata["kinetic_remainder_phase_bound"]
              +result.metadata["series_amplitude_error_bound"]
              +result.metadata["roundoff_relative_allowance"])
    assert 1e-6 < observed <= budget


def test_uniform_magnetic_field_matches_independent_sparse_exponential():
    x, y = (np.arange(19)-9)*1e-9, (np.arange(17)-8)*1e-9
    initial, bz, distance = _packet(x, y), 40., 2e-6
    matrix = _hamiltonian(_dirichlet_matrix(x, y, bz))
    expected = expm_multiply(-1j*distance*matrix, initial.ravel()).reshape(initial.shape)
    result = _run(_snapshot(_uniform_field(bz)), initial, x, y, (0., distance),
                  boundary_fraction_limit=.1)
    np.testing.assert_allclose(result.amplitude, expected, rtol=2e-10, atol=2e-13)
    assert result.metadata["output_flux"] == pytest.approx(.37, abs=2e-12)


def test_overlapping_fields_are_summed_before_forming_kinetic_cross_terms():
    x, y = (np.arange(19)-9)*1e-9, (np.arange(17)-8)*1e-9
    initial, distance = _packet(x, y), 2e-6
    options = dict(boundary_fraction_limit=.1)
    combined = _run(_snapshot(_uniform_field(24.), _uniform_field(16.)), initial, x, y,
                    (0., distance), **options)
    whole = _run(_snapshot(_uniform_field(40.)), initial, x, y, (0., distance), **options)
    np.testing.assert_allclose(combined.amplitude, whole.amplitude, rtol=2e-12, atol=2e-14)
    # A tempting but wrong H(A1)+H(A2)-H(0) construction loses A1.A2.
    wrong = (_hamiltonian(_dirichlet_matrix(x, y, 24.))
             +_hamiltonian(_dirichlet_matrix(x, y, 16.))
             -_hamiltonian(_dirichlet_matrix(x, y)))
    wrong_result = expm_multiply(-1j*distance*wrong, initial.ravel()).reshape(initial.shape)
    assert np.linalg.norm(whole.amplitude-wrong_result)/np.linalg.norm(initial) > 1e-4
    cancelled = _run(_snapshot(_uniform_field(40.), _uniform_field(-40.)), initial, x, y,
                     (0., distance), **options)
    vacuum = _run(_snapshot(), initial, x, y, (0., distance), **options)
    np.testing.assert_allclose(cancelled.amplitude, vacuum.amplitude, rtol=0., atol=2e-14)


@pytest.mark.parametrize("bz", (0., 15.))
def test_polynomial_gauge_with_xz_yz_and_constant_phase_is_preserved(bz):
    x = (np.arange(19)-9)*2e-9
    y = (np.arange(17)-8)*2e-9
    z = np.linspace(0., 4e-6, 5)
    initial = _packet(x, y, sigma=4e-9, flux=2.3)
    original = _snapshot(_uniform_field(bz))
    scale = hbar/e
    gauge = StaticPolynomialGauge(.71*scale, scale*np.array((2e7, -3e7, 2e5)),
        scale*np.array(((5e14, 2e14, 3e12), (2e14, -4e14, -2e12), (3e12, -2e12, 2e10))))
    changed = replace(original, gauge=gauge)
    transformed = initial*gauge.electron_phase_factor(_points(x, y, z[0]))
    baseline = _run(original, initial, x, y, z, boundary_fraction_limit=.01)
    output = _run(changed, transformed, x, y, z, boundary_fraction_limit=.01)
    expected = baseline.amplitude*gauge.electron_phase_factor(_points(x, y, z[-1]))
    np.testing.assert_allclose(output.amplitude, expected, rtol=2e-11, atol=3e-13)
    np.testing.assert_allclose(abs(output.amplitude)**2, abs(baseline.amplitude)**2, rtol=2e-11, atol=2e-14)


@pytest.mark.parametrize("blocked", (False, True))
def test_aperture_loss_is_absolute_and_zero_transmission_is_retained(blocked):
    x = y = (np.arange(25)-12)*2e-9
    initial = _packet(x, y, sigma=4e-9, flux=2.7)
    z = (0., 1e-6, 2e-6)
    baseline = _run(_snapshot(), initial, x, y, z, boundary_fraction_limit=1e-5)
    mask = np.zeros(initial.shape, dtype=bool) if blocked else np.broadcast_to(x > 0., initial.shape)
    result = _run(_snapshot(), initial, x, y, z, boundary_fraction_limit=1e-5,
                  aperture_masks={2: mask})
    np.testing.assert_array_equal(result.amplitude, np.where(mask, baseline.amplitude, 0.))
    lost = float(np.sum(abs(baseline.amplitude[~mask])**2))
    assert result.metadata["aperture_loss"] == pytest.approx(lost, abs=3e-12)
    assert result.metadata["output_flux"]+result.metadata["aperture_loss"] == pytest.approx(2.7, abs=3e-12)
    assert result.metadata["normalization_applied"] is False
    if blocked:
        assert result.metadata["output_flux"] == 0.


def test_occupied_edge_and_outside_endpoint_are_rejected():
    x = y = (np.arange(9)-4)*1e-9
    initial = _packet(x, y, sigma=1e-9)
    initial[0, 0] = 1.
    with pytest.raises(ValueError, match="boundary"):
        _run(_snapshot(), initial, x, y, (0., 1e-8), boundary_fraction_limit=1e-5)
    initial = _packet(x, y, sigma=1e-9)
    # Four-point quadrature samples alone lie inside this field even though
    # both physical endpoint planes are outside its represented domain.
    with pytest.raises(ValueError, match="outside"):
        _run(_snapshot(_uniform_field(0., half_length=1e-6)), initial, x, y,
             (-1.01e-6, 1.01e-6), boundary_fraction_limit=.01)


def test_dirichlet_revival_cannot_hide_intermediate_boundary_occupation():
    # Two exact finite-box eigenmodes beat once and return to the same shape.
    # Choose their mixture by minimizing boundary probability at the input;
    # an independent sparse exponential proves that the middle is occupied.
    # The synthetic 4% guard separates these states; it is not an accepted
    # physical boundary-loss tolerance for a microscope wave calculation.
    count, pitch = 23, 1e-9
    axis = (np.arange(count)-count//2)*pitch
    modes = np.array([np.sqrt(2/(count+1))*np.sin(
        mode*np.pi*(np.arange(count)+1)/(count+1)) for mode in (1, 3)])
    edges = np.r_[0, 1, count-2, count-1]
    _, mixtures = np.linalg.eigh(modes[:, edges]@modes[:, edges].T)
    initial = np.outer(modes[0], mixtures[:, 0]@modes).astype(complex)
    initial *= np.exp(.37j)
    eigenvalues = [4*np.sin(mode*np.pi/(2*(count+1)))**2/pitch**2 for mode in (1, 3)]
    distance = 4*np.pi*_wavenumber()/(eigenvalues[1]-eigenvalues[0])
    matrix = _dirichlet_matrix(axis, axis)/(2*_wavenumber())
    middle = expm_multiply(-.5j*distance*matrix, initial.ravel()).reshape(initial.shape)
    final = expm_multiply(-1j*distance*matrix, initial.ravel()).reshape(initial.shape)
    def edge_fraction(values):
        probability = abs(values)**2
        return 1-float(probability[2:-2, 2:-2].sum()/probability.sum())
    limit = .04
    assert edge_fraction(initial) < .015 < limit
    assert edge_fraction(final) < .015 < limit
    assert edge_fraction(middle) > .06 > limit
    # Revival includes its absolute eigenphase; no fitted phase or shift.
    expected = initial*np.exp(-1j*distance*eigenvalues[0]/_wavenumber())
    np.testing.assert_allclose(final, expected, atol=3e-12, rtol=0)
    with pytest.raises(ValueError, match="boundary"):
        _run(_snapshot(), initial, axis, axis, (0., distance),
             kinetic_order=2, boundary_fraction_limit=limit)


def _dipole_potential(*, dynamic=False):
    field = FrozenPosedMultipole("source:0", "source:0", CoordinateRegistration(),
        ((1, 0, .02),), "gaussian", 2e-6, 1e-6, (-.01, .02), 1e-6,
        hbar*_wavenumber(), captured_time_s=0. if dynamic else None, dynamic=dynamic)
    return AnalyticMagneticPotential(field)


def test_dynamic_provider_cannot_enter_static_operator_without_provenance_hint():
    x = y = (np.arange(9)-4)*1e-9
    with pytest.raises(ValueError, match="dynamic|static"):
        _run(_snapshot(_dipole_potential(dynamic=True)), _packet(x, y, sigma=1e-9),
             x, y, (0., 1e-8), boundary_fraction_limit=.01)


def test_axial_splitting_converges_to_independent_complex_ode(record_testsuite_property):
    x = y = (np.arange(9)-4)*4e-9
    initial = _packet(x, y, sigma=2.5e-9)
    length = 4e-6
    matrix = _hamiltonian(_dirichlet_matrix(x, y))
    xx, _ = np.meshgrid(x, y)
    def derivative(fraction, psi):
        axial = length*fraction
        az = .02*xx.ravel()*np.exp(-.5*((axial-2e-6)/1e-6)**2)
        return -1j*length*(matrix@psi+(e/hbar)*az*psi)
    reference = solve_ivp(derivative, (0., 1.), initial.ravel(), method="DOP853",
                          rtol=2e-12, atol=2e-14)
    assert reference.success
    expected = reference.y[:, -1].reshape(initial.shape)
    errors = []
    snapshot = _snapshot(_dipole_potential())
    for steps in (4, 8, 16):
        result = _run(snapshot, initial, x, y, np.linspace(0., length, steps+1),
                      boundary_fraction_limit=.01)
        errors.append(float(np.linalg.norm(result.amplitude-expected)/np.linalg.norm(initial)))
    record_testsuite_property("potential_wave_axial_errors", json.dumps(errors))
    assert errors[1] < .35*errors[0] and errors[2] < .35*errors[1], errors
    assert errors[-1] < 5e-4, errors


def test_complex_grid_convergence_keeps_the_same_dirichlet_box(record_testsuite_property):
    width, distance, mx, my = 1e-8, 2e-6, 2, 3
    k = _wavenumber()
    continuum_eigenvalue = np.pi**2*(mx*mx+my*my)/width**2
    phase = distance*(continuum_eigenvalue/(2*k)+continuum_eigenvalue**2/(8*k**3))
    errors = []
    for count in (15, 31, 63):
        pitch = width/(count+1)
        axis = (np.arange(count)+1)*pitch-width/2
        initial = np.outer(np.sin(np.pi*my*(np.arange(count)+1)/(count+1)),
                           np.sin(np.pi*mx*(np.arange(count)+1)/(count+1))).astype(complex)
        initial *= (2*pitch/width)*np.exp(.29j)
        result = _run(_snapshot(), initial, axis, axis, (0., distance), boundary_fraction_limit=.95)
        expected = initial*np.exp(-1j*phase)
        errors.append(float(np.linalg.norm(result.amplitude-expected)/np.linalg.norm(initial)))
    record_testsuite_property("potential_wave_transverse_errors", json.dumps(errors))
    assert errors[1] < .3*errors[0] and errors[2] < .3*errors[1], errors
    assert errors[-1] < .002, errors


@pytest.mark.parametrize("options, message", [
    ({"kinetic_order": True}, "kinetic_order"),
    ({"exponential_error": 1e-20}, "precision"),
    ({"boundary_fraction_limit": 1.}, "less than one"),
    ({"kinetic_remainder_phase_limit": 1e-30}, "truncation"),
    ({"aperture_masks": {7: np.ones((9, 9), dtype=bool)}}, "axial node"),
    ({"aperture_masks": {0: np.ones((9, 9), dtype=float)}}, "boolean"),
])
def test_invalid_or_unattainable_numeric_requests_fail(options, message):
    x = y = (np.arange(9)-4)*1e-9
    options = {"boundary_fraction_limit": .01, **options}
    with pytest.raises(ValueError, match=message):
        _run(_snapshot(), _packet(x, y, sigma=1e-9), x, y, (0., 1e-8), **options)


def test_plane_wave_bridge_preserves_carrier_origin_scalar_phase_and_input():
    from temsim.physics.multiplane_wave import PlaneWave
    x, y = (np.arange(53)-26)*1e-9, (np.arange(49)-24)*1e-9
    basis, origin = np.eye(2)*1e-9, np.array((3e-9, -4e-9))
    curvature = np.array(((1500., 300.), (300., -700.)))
    tilt = np.array((4e-6, -3e-6))
    wave = PlaneWave(_packet(x, y, sigma=4e-9, flux=2.4), basis, origin, curvature, tilt)
    saved = {name: getattr(wave, name).copy() for name in
             ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad")}
    wavelength = 2*np.pi/_wavenumber()
    expanded = wave.full_amplitude(wavelength)
    assert np.linalg.norm(expanded-wave.amplitude)/np.linalg.norm(expanded) > .01
    snapshot, z = _snapshot(_uniform_field(12.)), (0., 1e-6, 2e-6)
    expected = _run(snapshot, expanded, x+origin[0], y+origin[1], z, boundary_fraction_limit=1e-5)
    output, metadata = propagate_plane_wave_potential(snapshot, wave, z, energy_ev=ENERGY_EV,
                                                      boundary_fraction_limit=1e-5)
    np.testing.assert_allclose(output.full_amplitude(wavelength), expected.amplitude, rtol=2e-12, atol=2e-14)
    np.testing.assert_array_equal(output.basis_m, wave.basis_m)
    np.testing.assert_array_equal(output.origin_m, wave.origin_m)
    assert output.curvature_m1 is None and output.tilt_rad is None
    assert metadata["output_flux"] == pytest.approx(wave.probability, abs=3e-12)
    for name, value in saved.items():
        np.testing.assert_array_equal(getattr(wave, name), value)
    assert not np.shares_memory(output.amplitude, wave.amplitude)


@pytest.mark.parametrize("invalid", ("undersampled_carrier", "sheared_lattice"))
def test_plane_wave_bridge_rejects_unsupported_representation(invalid):
    from temsim.physics.multiplane_wave import PlaneWave
    axis = (np.arange(25)-12)*2e-9
    basis = np.array(((2e-9, 1e-10 if invalid == "sheared_lattice" else 0.), (0., 2e-9)))
    tilt = np.array((.01 if invalid == "undersampled_carrier" else 0., 0.))
    wave = PlaneWave(_packet(axis, axis, sigma=4e-9), basis, np.zeros(2), tilt_rad=tilt)
    pattern = "undersampled" if invalid == "undersampled_carrier" else "axis-aligned"
    with pytest.raises(ValueError, match=pattern):
        propagate_plane_wave_potential(_snapshot(), wave, (0., 1e-6), energy_ev=ENERGY_EV)


def _lorentz_packet_reference(sigma, length, final_transverse_fraction, quadrature_order):
    """Full constant-energy Lorentz circles, averaged over a Gaussian source.

    For B=(0,B_y,0), q=-e, and +z motion, dp_x/dz=e*B_y exactly.
    With u=p_x/p, v=p_y/p, w=p_z/p, u_1=u_0+alpha and v is constant.
    The exact position change is (p/eB_y)*(w_0-w_1), not a paraxial
    Hamiltonian trajectory. A rationalized difference avoids cancellation.

    The initial amplitude exp(-r^2/(2*sigma^2)) has Var(u)=Var(v)=
    1/(2*k^2*sigma^2). These are flux weights. The finite Gaussian quadrature
    remains wholly inside the forward shell; its independent order check is
    below. This is a semiclassical Gaussian source, not a spin benchmark.
    """
    nodes, weights = roots_hermite(quadrature_order)
    u, v = np.meshgrid(nodes/(_wavenumber()*sigma), nodes/(_wavenumber()*sigma))
    weights = np.outer(weights, weights)/np.pi
    alpha = final_transverse_fraction
    assert np.max((u+alpha)**2+v*v) < 1.
    w0, w1 = np.sqrt(1-u*u-v*v), np.sqrt(1-(u+alpha)**2-v*v)
    x = length/alpha*np.sum(weights*(2*alpha*u+alpha*alpha)/(w0+w1))
    slope = np.sum(weights*(u+alpha)/w1)
    return float(x), float(slope)


@pytest.mark.parametrize("sigma", (3e-10, 6e-10))
def test_narrow_packet_centroid_and_mechanical_angle_approach_full_lorentz(
        sigma, record_testsuite_property):
    # Both 300/600 pm packets are narrow relative to the 1 um represented
    # magnetic region. Increasing k*sigma narrows momentum spread, rather than
    # incorrectly taking a zero-position-width wave to mean a classical ray.
    # The deliberately short synthetic field gives 2 mrad deflection while
    # keeping this fixed-grid benchmark inexpensive; it is not an OEM magnet.
    k, length, alpha, axial_steps = _wavenumber(), 5e-8, .002, 4
    magnetic_y = alpha*hbar*k/(e*length)
    registration = CoordinateRegistration(rotation_local_to_global=(
        (1., 0., 0.), (0., 0., 1.), (0., -1., 0.)))
    rotated = replace(_uniform_field(magnetic_y, half_length=1e-6), registration=registration)
    # The rotated symmetric gauge has Ax=B_y*z/2, Az=-B_y*x/2.
    # chi=-B_y*x*z/2 gives the same field with Ax=Ay=0, Az=-B_y*x.
    hessian = np.zeros((3, 3))
    hessian[0, 2] = hessian[2, 0] = -magnetic_y/2
    snapshot = _snapshot(rotated, gauge=StaticPolynomialGauge(hessian_t=hessian))
    sample = snapshot.sample(((1e-9, 0., length/2),))
    np.testing.assert_allclose(sample["B_t"], ((0., magnetic_y, 0.),), rtol=2e-15, atol=1e-14)
    np.testing.assert_allclose(sample["A_t_m"], ((0., 0., -magnetic_y*1e-9),), rtol=2e-15, atol=1e-22)

    central_x = length*alpha/(1+np.sqrt(1-alpha*alpha))
    central_slope = alpha/np.sqrt(1-alpha*alpha)
    ensemble_x, ensemble_slope = _lorentz_packet_reference(sigma, length, alpha, 20)
    check_x, check_slope = _lorentz_packet_reference(sigma, length, alpha, 28)
    assert abs(check_x-ensemble_x) < 1e-24
    assert abs(check_slope-ensemble_slope) < 2e-15

    variance = 1/(2*k*k*sigma*sigma)
    continuum_quartic_x = length*(alpha/2+alpha**3/8+alpha*variance)
    # In this linear force the exact continuous-quartic mean velocity is
    # u+u^3/2+2*u*variance. Composite midpoint sampling of u^3 has the known
    # bias below. Keep it separate from the transverse lattice dispersion.
    axial_bias = -length*alpha**3/(16*axial_steps**2)
    split_quartic_x = continuum_quartic_x+axial_bias
    finite_width_x = ensemble_x-central_x
    kinetic_truncation_x = continuum_quartic_x-ensemble_x
    assert 0 < finite_width_x < 1e-15
    assert abs(kinetic_truncation_x) < 1e-20
    assert abs(axial_bias) < 1e-17
    errors, observations = [], []
    for cells_per_sigma in (4, 8, 16):
        pitch = sigma/cells_per_sigma
        axis = np.arange(-5*cells_per_sigma, 5*cells_per_sigma+1)*pitch
        xx, yy = np.meshgrid(axis, axis)
        initial = np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma)+.37j)
        initial *= np.sqrt(.83/np.sum(abs(initial)**2))
        result = _run(snapshot, initial, axis, axis, np.linspace(0., length, axial_steps+1),
                      boundary_fraction_limit=1e-8)
        intensity = abs(result.amplitude)**2
        flux = float(intensity.sum())
        centroid = float(np.sum(xx*intensity)/flux)
        # This Fourier transform is only a mechanical-momentum observable on
        # a negligibly occupied boundary; propagation is still Dirichlet.
        # A_perp=0 in this gauge, so these momenta are mechanical, not a
        # gauge-dependent canonical-angle proxy. Divide by flux only for an
        # expectation value, without changing any propagated amplitude.
        spectrum = abs(np.fft.fft2(result.amplitude, norm="ortho"))**2/flux
        fractions = 2*np.pi*np.fft.fftfreq(len(axis), pitch)/k
        u, v = np.meshgrid(fractions, fractions)
        assert np.max(u*u+v*v) < 1.
        slope = float(np.sum(spectrum*u/np.sqrt(1-u*u-v*v)))
        angle = float(np.arctan(slope))
        grid_error = centroid-split_quartic_x
        errors.append(abs(grid_error))
        observations.append({"cells_per_sigma": cells_per_sigma,
                             "centroid_m": centroid, "mechanical_incidence_angle_rad": angle,
                             "transverse_grid_centroid_error_m": grid_error,
                             "mechanical_slope_error": slope-ensemble_slope})
        assert abs(slope-ensemble_slope) < 2e-12, observations
        assert angle == pytest.approx(np.arctan(ensemble_slope), abs=2e-12)
        assert flux == pytest.approx(.83, abs=3e-11)
    # The derivative stencil is second order. A 1% centroid budget is a
    # bounded continuum-limit demonstration, not nanometre instrument
    # qualification. Refinement must improve the actual complex evolution.
    assert errors[1] < .3*errors[0] and errors[2] < .3*errors[1], observations
    assert errors[-1] < .01*central_x, observations
    # At this small angle the physical quartic truncation is below resolvable
    # lattice error; report the analytic decomposition, never claim to have
    # numerically resolved that tiny effect. The eigenphase test above
    # separately exercises a resolvable quartic kinetic contribution.
    evidence = {"sigma_m": sigma, "length_m": length, "B_y_t": magnetic_y,
                "central_full_lorentz_x_m": central_x, "central_full_lorentz_angle_rad": float(np.arcsin(alpha)),
                "finite_packet_centroid_shift_m": finite_width_x,
                "finite_packet_mechanical_slope_shift": ensemble_slope-central_slope,
                "quartic_dynamics_centroid_error_m": kinetic_truncation_x,
                "axial_midpoint_centroid_error_m": axial_bias,
                "observations": observations}
    record_testsuite_property(f"potential_wave_lorentz_{round(sigma*1e12)}pm", json.dumps(evidence, sort_keys=True))


def test_actual_cuda_operator_matches_cpu_complex_field_and_flux(record_testsuite_property):
    cp = pytest.importorskip("cupy", reason="Actual CUDA verification requires optional CuPy")
    from temsim.physics.compute_backend import cupy_capability
    capability = cupy_capability()
    if not capability.available:
        pytest.skip(capability.detail)
    x = y = (np.arange(25)-12)*2e-9
    initial = _packet(x, y, sigma=4e-9, flux=1.7)
    z = np.linspace(0., 2e-6, 5)
    scale = hbar/e
    gauge = StaticPolynomialGauge(.31*scale, scale*np.array((1e7, -2e7, 3e5)),
        scale*np.array(((2e14, 0., 2e12), (0., -1e14, -1e12), (2e12, -1e12, 0.))))
    snapshot = _snapshot(_uniform_field(11.), _uniform_field(9.), gauge=gauge)
    initial *= gauge.electron_phase_factor(_points(x, y, z[0]))
    mask = np.broadcast_to(x < 8e-9, initial.shape)
    options = dict(boundary_fraction_limit=1e-4, aperture_masks={2: mask})
    cpu = _run(snapshot, initial, x, y, z, backend="numpy", **options)
    gpu = _run(snapshot, cp.asarray(initial), x, y, z, backend="cupy", **options)
    cp.cuda.get_current_stream().synchronize()
    assert isinstance(gpu.amplitude, cp.ndarray)
    assert gpu.metadata["backend"] == "cupy"
    np.testing.assert_allclose(cp.asnumpy(gpu.amplitude), cpu.amplitude, rtol=3e-10, atol=3e-13)
    for key in ("input_flux", "output_flux", "aperture_loss"):
        assert gpu.metadata[key] == pytest.approx(cpu.metadata[key], rel=2e-12, abs=2e-13)
    record_testsuite_property("potential_wave_actual_cuda", capability.detail)
