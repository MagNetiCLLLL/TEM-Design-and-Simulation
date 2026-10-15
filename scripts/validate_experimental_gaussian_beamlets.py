"""Small CPU checks for experimental coherent Gaussian packet operators.

This diagnostic validates the stated local mathematical operations only. It
creates no source checkpoint and does not establish source-to-specimen accuracy.
Run with --output PATH to retain the complex-field errors and residual bounds.
"""
import json
from pathlib import Path
import sys

import numpy as np

if not __package__:
    # Use one module identity for Numba's cache in scripts and pytest alike.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experimental_gaussian_beamlets import GaussianBeamlets, propagate_affine, apply_taylor_screen
from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.multipole_wave import multipole_action


def packet(sigma=1e-7, centre=(0., 0.), weight=1.+0j, real=None):
    wavelength = 2e-12
    k = 2*np.pi/wavelength
    c = (np.zeros((2, 2)) if real is None else np.asarray(real))+1j*np.eye(2)/(2*k*sigma*sigma)
    return GaussianBeamlets(np.array([centre]), np.array([[2e-6, -3e-6]]), c[None],
                            np.array([weight]), wavelength)


def points(shape, basis, origin):
    ny, nx = shape
    y, x = np.meshgrid(np.arange(ny)-ny//2, np.arange(nx)-nx//2, indexing="ij")
    return origin+np.stack((x.ravel(), y.ravel()), axis=1)@basis.T


def complete(wave, wavelength):
    p = points(wave.amplitude.shape, wave.basis_m, wave.origin_m)-wave.origin_m
    c = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    return wave.amplitude*np.exp(2j*np.pi/wavelength*(.5*np.einsum("pi,ij,pj->p", p, c, p)+p@tilt)).reshape(wave.amplitude.shape)


def canonical_checks():
    source = packet(centre=(2e-8, -3e-8), real=[[20., 3.], [3., -10.]])
    basis = np.eye(2)*1e-8
    coordinates = points((256, 256), basis, source.centres_m[0])
    c, p = source.precision_m1[0].real, source.momenta[0]
    relative = coordinates-source.centres_m[0]
    carrier = 2*np.pi/source.wavelength_m*(.5*np.einsum("pi,ij,pj->p", relative, c, relative)+relative@p)
    full = source.evaluate(coordinates).reshape((256, 256))*1e-8
    wave = PlaneWave(full*np.exp(-1j*carrier).reshape(full.shape), basis, source.centres_m[0], c, p)
    matrices = []
    drift = np.eye(4); drift[:2, 2:] = np.eye(2)*1e-3
    matrices.append(drift)
    matrices.append(np.block([[np.zeros((2, 2)), np.eye(2)*1e-3], [-np.eye(2)*1e3, np.zeros((2, 2))]]))
    lens = np.eye(4); lens[2:, :2] = np.array([[-500., 37.], [37., 200.]])
    matrices.append(lens@drift)
    errors = []
    for matrix in matrices:
        shift = np.array([1e-8, -2e-8, 3e-6, -1e-6])
        result = propagate_affine(source, matrix, shift, affine_action_m=3e-12)
        reference = propagate_plane_wave(wave, matrix, shift, source.wavelength_m, affine_action_m=3e-12)
        sampled = result.evaluate(points(reference.amplitude.shape, reference.basis_m, reference.origin_m))
        sampled = sampled.reshape(reference.amplitude.shape)*np.sqrt(abs(np.linalg.det(reference.basis_m)))
        actual_reference = complete(reference, source.wavelength_m)
        error = np.linalg.norm(sampled-actual_reference)/np.linalg.norm(actual_reference)
        assert error < 2e-10, error
        errors.append(float(error))
    path = CanonicalPath(1e-3)
    theta = 2*np.pi/64
    step = np.eye(4)
    step[np.ix_([0, 2], [0, 2])] = [[np.cos(theta), 1e-3*np.sin(theta)], [-1e3*np.sin(theta), np.cos(theta)]]
    evolved = source
    for _ in range(64):
        path.append(step)
        evolved = propagate_affine(evolved, step, reference_length_m=1e-3)
    np.testing.assert_allclose(evolved.coefficients, -source.coefficients, rtol=1e-11, atol=1e-11)
    direct = propagate_affine(source, path.matrix, path.offset, **path.phase_kwargs())
    np.testing.assert_allclose(evolved.coefficients, direct.coefficients, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(evolved.precision_m1, source.precision_m1, rtol=1e-10, atol=1e-10)
    return {"canonical_complex_relative_errors": errors, "single_axis_full_cycle_phase": float(np.angle(evolved.coefficients[0]))}


def screen_checks():
    errors, bounds = [], []
    for sigma in (2e-7, 1e-7, .5e-7):
        source = packet(sigma, (7e-7, -4e-7), .8+.3j, [[10., 2.], [2., 30.]])
        axis = (np.arange(192)-96)*sigma/12
        y, x = np.meshgrid(axis, axis, indexing="ij")
        coordinates = source.centres_m[0]+np.stack((x.ravel(), y.ravel()), axis=1)
        hn, hs, cs = 2e6, -1e6, 2e12
        expected = source.evaluate(coordinates)*np.exp(2j*np.pi/source.wavelength_m*multipole_action(
            coordinates[:, 0], coordinates[:, 1], normal_m2=hn, skew_m2=hs, spherical_m3=cs))
        result, record = apply_taylor_screen(source, normal_m2=hn, skew_m2=hs, spherical_m3=cs)
        actual = result.evaluate(coordinates)
        error = float(np.linalg.norm(actual-expected)*sigma/12)
        bound = record["coherent_absolute_l2_error_upper_bound"]
        assert error <= bound*(1+1e-9), (error, bound)
        restored, _ = apply_taylor_screen(result, normal_m2=-hn, skew_m2=-hs, spherical_m3=-cs)
        np.testing.assert_allclose(restored.evaluate(coordinates), source.evaluate(coordinates), rtol=3e-11, atol=1e-6)
        errors.append(error); bounds.append(bound)
    assert errors[1] < errors[0]/6 and errors[2] < errors[1]/6, errors
    return {"screen_complex_l2_errors": errors, "screen_l2_upper_bounds": bounds}


def noncommuting_lift_checks():
    from scipy.linalg import expm
    source = packet(real=[[50., 35.], [35., -20.]])
    precision = source.precision_m1.copy()
    precision[0] = precision[0].real+1j*np.array([[20., 3.], [3., 30.]])
    source = GaussianBeamlets(source.centres_m, source.momenta, precision,
                             np.array([.7+.2j]), source.wavelength_m)
    evolved, path = source, CanonicalPath(1e-3)
    for index in range(40):
        rotation = np.array([[0., 200.*(-1)**index], [-200.*(-1)**index, 0.]])
        stiffness = np.array([[1e6+index*1000, 2e5], [2e5, 5e5-index*1000]])
        generator = np.block([[rotation, np.eye(2)], [-stiffness, rotation]])
        step = expm(generator*1e-4)
        displacement = np.array([2e-9, -3e-9, 1e-6, -2e-6])*(-1)**index
        action = (index+1)*1e-15
        path.append(step, displacement, action_m=action)
        evolved = propagate_affine(evolved, step, displacement,
            affine_action_m=action, reference_length_m=1e-3)
    direct = propagate_affine(source, path.matrix, path.offset, **path.phase_kwargs())
    points_here = points((64, 64), np.eye(2)*2e-8, evolved.centres_m[0])
    relative_error = np.linalg.norm(evolved.evaluate(points_here)-direct.evaluate(points_here))/np.linalg.norm(direct.evaluate(points_here))
    assert relative_error < 5e-11, relative_error
    extra_lift = dict(path.phase_kwargs(), reference_phase_rad=path.reference_phase_rad+np.pi)
    flipped = propagate_affine(source, path.matrix, path.offset, **extra_lift)
    np.testing.assert_allclose(flipped.evaluate(points_here), -direct.evaluate(points_here), rtol=5e-13, atol=1e-8)
    two_axis = CanonicalPath(1e-3)
    theta = np.pi/32
    step = np.block([[np.eye(2)*np.cos(theta), np.eye(2)*1e-3*np.sin(theta)],
                     [-np.eye(2)*1e3*np.sin(theta), np.eye(2)*np.cos(theta)]])
    for _ in range(32):
        two_axis.append(step)
    transformed = propagate_affine(source, two_axis.matrix, two_axis.offset, **two_axis.phase_kwargs())
    np.testing.assert_allclose(transformed.coefficients, -source.coefficients, rtol=5e-12, atol=5e-12)
    return {"noncommuting_path_complex_relative_error": float(relative_error),
            "two_axis_half_cycle_phase_rad": two_axis.reference_phase_rad}


def readout_checks():
    source = packet()
    basis, origin, shape = np.array([[1e-8, .4e-8], [-.2e-8, 1.1e-8]]), np.array([1e-8, -3e-8]), (192, 128)
    reference = source.evaluate(points(shape, basis, origin)).reshape(shape)*np.sqrt(abs(np.linalg.det(basis)))
    actual, evidence = source.sample_plane(shape, basis, origin, amplitude_tail=1e-8, return_record=True)
    error = float(np.max(abs(reference-actual)))
    assert error <= evidence["pointwise_cell_amplitude_error_bound"]+1e-15
    axis = (np.arange(128)-64)*2e-8
    actual_grid, evidence_grid = source.evaluate_grid(axis, axis, amplitude_tail=1e-8)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    expected_grid = source.evaluate(np.stack((x.ravel(), y.ravel()), axis=1)).reshape(actual_grid.shape)
    assert np.max(abs(actual_grid-expected_grid)) <= evidence_grid["pointwise_amplitude_density_error_bound"]+1e-7
    try:
        source.sample_plane(shape, basis, origin, maximum_working_bytes=1)
    except MemoryError:
        pass
    else:
        raise AssertionError("Readout budget did not reject")
    return {"affine_readout_max_cell_error": error, "affine_readout_bound": evidence["pointwise_cell_amplitude_error_bound"]}


def run_checks():
    return {"schema": "experimental-gaussian-beamlet-validation-v1",
            "scope": "Local canonical operators, Taylor-screen bounds and readout; not source-to-specimen validation",
            "production_solver": False,
            **canonical_checks(), **noncommuting_lift_checks(), **screen_checks(), **readout_checks()}


def main():
    import argparse
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        report = run_checks()
    rendered = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
