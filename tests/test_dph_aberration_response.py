"""Local DPH/HP physics checks, independent of full-column calibration."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.multipole_wave import apply_multipole_phase
from temsim.physics.ray_integrator import canonical_rk4_step


def _force(x, y, normal, skew):
    return np.array((-normal*(x*x-y*y)-2*skew*x*y,
                     2*normal*x*y-skew*(x*x-y*y)))


def _feed_down(displacement, normal, skew):
    a, b = displacement
    return np.array(((-2*(normal*a+skew*b), 2*(normal*b-skew*a)),
                     (2*(normal*b-skew*a), 2*(normal*a+skew*b))))


@pytest.mark.parametrize("displacement", [(4e-8, -3e-8), (-4e-8, 3e-8), (0., 0.)])
def test_displaced_hexapole_wave_keeps_exact_constant_affine_and_cubic_terms(displacement):
    normal, skew, wavelength = 2e8, -3e8, 2e-12
    axis = (np.arange(64)-32)*2e-9
    u, v = np.meshgrid(axis, axis)
    amplitude = np.exp(-(u*u+v*v)/(2*(12e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    source = PlaneWave(amplitude, np.eye(2)*2e-9, np.asarray(displacement))
    output = apply_multipole_phase(source, wavelength, normal_m2=normal, skew_m2=skew)
    jacobian = _feed_down(displacement, normal, skew)
    np.testing.assert_allclose(output.curvature_m1, jacobian, rtol=2e-15, atol=1e-14)
    np.testing.assert_allclose(output.tilt_rad, _force(*displacement, normal, skew), atol=1e-20)
    assert np.trace(jacobian) == 0.
    np.testing.assert_array_equal(jacobian, jacobian.T)
    x, y = source.coordinates_m()
    # Independent full cubic polynomial; no fitted phase or renormalisation.
    action = -normal*(x**3/3-x*y*y)-skew*(x*x*y-y**3/3)
    expected = amplitude*np.exp(2j*np.pi*action/wavelength)
    np.testing.assert_allclose(output.full_amplitude(wavelength), expected, atol=2e-15, rtol=2e-13)
    assert output.probability == pytest.approx(source.probability, rel=1e-14)
    # Fixed transverse displacement changes a pure quadratic force only by
    # constant + affine terms. A non-affine remainder is NOT required here.
    translated = _force(x, y, normal, skew)-_force(u, v, normal, skew)
    exact_delta = (_force(*displacement, normal, skew)[:, None, None]
                   + np.einsum("ij,jyx->iyx", jacobian, np.array((u, v))))
    np.testing.assert_allclose(translated, exact_delta, atol=3e-21, rtol=2e-13)


def test_finite_dipole_moves_rays_inside_hp_before_later_nonlinear_force_evaluation():
    # Deliberately measurable local fixture, not a proposed instrument setting.
    length, normal, skew, drive = .004, 5e7, -2e7, .002
    x, y = np.meshgrid(np.array((-1., 0., 1.))*1e-4,
                       np.array((-1., 0., 1.))*1e-4)
    initial = np.array((x.ravel(), np.zeros(9), y.ravel(), np.zeros(9)))

    def rk_trace(hn, hs, angle, steps=64):
        state = initial.copy()
        for _ in range(steps):
            state = np.array(canonical_rk4_step(*state, length/steps,
                *([0.]*9), hn, hn, hn, hs, hs, hs,
                fx0=angle/length, fxm=angle/length, fx1=angle/length))
        return state

    def derivative(_z, flat):
        xx, tx, yy, ty = flat.reshape(4, -1)
        fx, fy = _force(xx, yy, normal, skew)
        return np.array((tx, fx+drive/length, ty, fy)).ravel()

    reference = solve_ivp(derivative, (0., length), initial.ravel(),
        method="DOP853", rtol=2e-13, atol=1e-16).y[:, -1].reshape(4, -1)
    actual = rk_trace(normal, skew, drive)
    coarse = rk_trace(normal, skew, drive, steps=32)
    assert np.linalg.norm(actual-reference) < np.linalg.norm(coarse-reference)/12
    np.testing.assert_allclose(actual, reference, rtol=3e-10, atol=2e-13)
    centred = lambda array: array-array.mean(axis=1, keepdims=True)
    no_dipole = rk_trace(normal, skew, 0.)
    assert np.linalg.norm(centred(actual)-centred(no_dipole)) > 1e-7
    # A deferred centroid kick would leave the centred HP response unchanged.
    # Removing HP explicitly, however, must reduce DPH to a rigid translation.
    pure_dipole = rk_trace(0., 0., drive)
    np.testing.assert_allclose(centred(pure_dipole), centred(initial), atol=1e-18)
    np.testing.assert_allclose(pure_dipole[0]-initial[0], drive*length/2, atol=1e-18)
    np.testing.assert_allclose(pure_dipole[1], drive, atol=1e-18)
def _diagnostic_metrics():
    import importlib.util
    from pathlib import Path
    import sys
    scripts=Path(__file__).resolve().parents[1]/"scripts"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location("dph_metrics",scripts/"validate_dph_aberrations.py")
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_dph_metrics_remove_only_centroid_translation():
    metrics=_diagnostic_metrics()
    rng=np.random.default_rng(18)
    phase=rng.normal(size=(4,30))*np.array([1e-6,1e-6,1e-3,1e-3])[:,None]
    shifted=phase+np.array([4e-7,-3e-7,2e-4,-1e-4])[:,None]
    weights=np.arange(1.,31.)
    result=metrics.compare_shape(phase,shifted,weights,phase)
    np.testing.assert_allclose(result["centroid_change_m"],[4e-7,-3e-7],atol=1e-21)
    assert result["centered_same_ray_change_rms_m"]<1e-21
    np.testing.assert_allclose(result["spatial_affine"]["matrix"],np.eye(2),atol=1e-14)


def test_dph_metrics_keep_affine_shape_and_identify_nonaffine_remainder():
    metrics=_diagnostic_metrics()
    rng=np.random.default_rng(27)
    phase=rng.normal(size=(4,80))*np.array([1e-6,1e-6,1e-3,1e-3])[:,None]
    weights=np.ones(80)
    changed=phase.copy()
    matrix=np.array([[1.03,.02],[.02,.97]])
    changed[:2]=matrix@phase[:2]+np.array([3e-7,-1e-7])[:,None]
    result=metrics.compare_shape(phase,changed,weights,phase)
    np.testing.assert_allclose(result["spatial_affine"]["matrix"],matrix,atol=1e-14)
    assert result["centered_same_ray_change_rms_m"]>1e-8
    assert result["entrance_phase_space_affine_response"]["residual_rms_m"]<1e-21
    changed[0]+=1e5*phase[0]**2
    nonlinear=metrics.compare_shape(phase,changed,weights,phase)
    assert nonlinear["entrance_phase_space_affine_response"]["residual_rms_m"]>1e-8


def test_dph_metrics_common_cohort_includes_refinement_and_modes():
    metrics=_diagnostic_metrics()
    alive=np.ones(10,dtype=bool);mask=np.ones((3,10),dtype=bool)
    altered=mask.copy();altered[1,2]=False
    final=alive.copy();final[3]=False
    results=[(None,mask,alive,None,None),(None,altered,final,None,None)]
    selected,fraction=metrics.common_cohort(results,{"alive":alive},np.arange(1.,11.))
    assert not selected[2] and not selected[3]
    assert selected.sum()==8
    np.testing.assert_allclose(fraction,48/55)
