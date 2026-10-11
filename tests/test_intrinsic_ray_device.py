"""Real CUDA executes the same lens action and native fields as CPU rays."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import e
from scipy.spatial.transform import Rotation

from temsim.optics.intrinsic_aberrations import INTRINSIC_TERMS
from temsim.physics.intrinsic_aberrations import FrozenIntrinsicAberrationKick
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.posed_aberrations import FrozenLensAberrationKick
from temsim.physics.posed_aberration_wave import _potential
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.tip_gun_wave import _momentum_velocity
from test_wave_device import _cuda


P = float(_momentum_velocity(300000.)[0])


def registration(tilt=.001):
    matrix = Rotation.from_rotvec((tilt/2, -tilt, .07)).as_matrix()
    return CoordinateRegistration((1e-6, -2e-6, .5), tuple(map(tuple, matrix)))


def rays():
    rng = np.random.default_rng(842)
    data = (rng.uniform(-70e-6, 70e-6, 31), rng.uniform(-.002, .002, 31),
            rng.uniform(-70e-6, 70e-6, 31), rng.uniform(-.002, .002, 31))
    data[0][-1] = np.nan
    return data


def fields():
    pose = registration(.002)
    return (FrozenAnalyticField("lens", ((.15, 0., .01), (-.025, .005, .008)), pose, (-100., 100.), .01),
        FrozenPosedMultipole("multipole", "multipole", pose,
            ((1, 0, .0002), (0, 1, -.0001), (2, 0, .5), (1, 1, -.3),
             (3, 0, 15.), (1, 2, -45.)), "gaussian", 0., .01, (-100., 100.), .01, P))


def assert_pair(actual, expected, cp, *, rtol=3e-12, atol=3e-17):
    for device, host in zip(actual, expected):
        assert isinstance(device, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(device), host, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("tilt", (0., .001), ids=("aligned", "tilted"))
@pytest.mark.parametrize("term", INTRINSIC_TERMS, ids=lambda term: term.name)
def test_real_cuda_each_intrinsic_term_keeps_cpu_action(term, tilt):
    cp = _cuda()
    coefficients = {term.field: .01*100**(term.power-2)}
    if term.azimuth_field:
        coefficients[term.azimuth_field] = 23.
    kick = FrozenIntrinsicAberrationKick("screen", 0., .01,
        tuple(coefficients.items()), registration(tilt))
    source = rays()
    before = tuple(value.copy() for value in source)
    momenta = P*np.linspace(.8, 1.2, len(source[0]))
    cpu = kick.apply_canonical(*source, momentum_kg_m_s=momenta,
        reference_momentum_kg_m_s=P, aligned_bz_t=.02)
    gpu_source = tuple(cp.asarray(value) for value in source)
    gpu = kick.apply_canonical(*gpu_source, momentum_kg_m_s=cp.asarray(momenta),
        reference_momentum_kg_m_s=P, aligned_bz_t=.02)
    assert_pair(gpu, cpu, cp)
    assert np.nanmax(abs(cpu[1]-source[1]))+np.nanmax(abs(cpu[3]-source[3])) > 1e-9
    for unchanged, original, device in zip(source, before, gpu_source):
        np.testing.assert_array_equal(unchanged, original)
        np.testing.assert_array_equal(cp.asnumpy(device), original)


def test_real_cuda_combined_intrinsic_action_with_analytic_and_multipole_gauge():
    cp = _cuda()
    coefficients = {term.field: .003*100**(term.power-2) for term in INTRINSIC_TERMS}
    coefficients.update({term.azimuth_field: 17. for term in INTRINSIC_TERMS if term.azimuth_field})
    kick = FrozenIntrinsicAberrationKick("mixed", 0., .01,
        tuple(coefficients.items()), registration(.002))
    source = rays()
    kwargs = dict(fields=fields(), aligned_bz_t=.015,
        electrostatic_gradient_v_m=(.02, -.03, .001),
        electrostatic_hessian_v_m2=np.diag((1., 2., -3.)))
    momenta = P*np.linspace(.8, 1.2, len(source[0]))
    cpu = kick.apply_canonical(*source, momentum_kg_m_s=momenta,
        reference_momentum_kg_m_s=P, **kwargs)
    gpu = kick.apply_canonical(*(cp.asarray(value) for value in source),
        momentum_kg_m_s=cp.asarray(momenta), reference_momentum_kg_m_s=P, **kwargs)
    assert_pair(gpu, cpu, cp)
    assert np.nanmax(abs(cpu[0]-source[0])) > 1e-13


def test_real_cuda_native_field_and_potential_derivatives_match_cpu():
    cp = _cuda()
    source = rays()
    points = np.column_stack((source[0][:-1], source[2][:-1], np.linspace(.496, .504, 30)))
    slopes = np.column_stack((source[1][:-1], source[3][:-1]))
    charge_over_p = -e/(P*np.linspace(.8, 1.2, len(points)))
    gpu_points, gpu_slopes = cp.asarray(points), cp.asarray(slopes)
    for field in fields():
        assert_pair((field.field_at_global_positions_t(gpu_points),), (field.field_at_global_positions_t(points),), cp)
        assert_pair((field.slope_derivative(gpu_points, gpu_slopes, cp.asarray(charge_over_p)),),
                    (field.slope_derivative(points, slopes, charge_over_p),), cp, atol=1e-13)
        if isinstance(field, FrozenPosedMultipole):
            assert_pair(field.vector_potential_jet(gpu_points), field.vector_potential_jet(points), cp)
    assert_pair(_potential(gpu_points, fields(), .015, gradient=True, xp=cp),
                _potential(points, fields(), .015, gradient=True), cp)


def test_real_cuda_legacy_cs_geometric_screen_remains_identical():
    cp = _cuda()
    kick = FrozenLensAberrationKick("cs", 0., 1e7, registration(.002))
    source = rays()
    cpu = kick.apply(*source)
    gpu = kick.apply(*(cp.asarray(value) for value in source))
    assert_pair(gpu, cpu, cp)


def test_real_cuda_zero_and_invalid_input_have_cpu_semantics():
    cp = _cuda()
    source = tuple(cp.asarray(value) for value in rays())
    zero = FrozenIntrinsicAberrationKick("zero", 0., .01, (), registration(.2))
    assert all(x is y for x, y in zip(source, zero.apply_canonical(*source, momentum_kg_m_s=P)))
    active = replace(zero, coefficients=(("a1_mm", .001),))
    for data in (source, tuple(cp.asnumpy(value) for value in source)):
        with pytest.raises(ValueError, match="10 mrad"):
            active.apply_canonical(*data, momentum_kg_m_s=P)
    active = replace(active, registration=registration(.001))
    for data in (source, tuple(cp.asnumpy(value) for value in source)):
        with pytest.raises(ValueError, match="phase remainder"):
            active.apply_canonical(*data, momentum_kg_m_s=P, maximum_obliquity_phase_error_rad=1e-30)
        with pytest.raises(ValueError, match="reference momentum"):
            active.apply_canonical(*data, momentum_kg_m_s=P, reference_momentum_kg_m_s=-P)
    stopped = tuple(cp.full(4, cp.nan) for _ in range(4))
    output = active.apply_canonical(*stopped, momentum_kg_m_s=P)
    assert all(isinstance(value, cp.ndarray) and bool(cp.isnan(value).all()) for value in output)
