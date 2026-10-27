"""Bounded natural-aberration qualification of an actual solved air-core lens.

The source is a 3000 ampere-turn uniform annular winding, r=4--6 mm and
z=-5--5 mm, in vacuum. The existing FEM actually solves for A_phi. Its raw
P1 curl and an explicitly different axis-regular A reconstruction are both
reported. Neither uses a fitted Cs/Cc kick, a changed field amplitude, or
an inferred field map. This is not a pole-piece/material/OEM calibration.

The independent reference integrates Biot--Savart over that same winding.
First/third-order Lorentz perturbation equations supply its infinitesimal
coefficients. The finite-pupil FEM traces keep the input, energy-independent
hardware and nominal image plane fixed. Their raw intercepts retain defocus
and tangential residuals. Pupil fitting does not modify propagated rays.

Qualification budgets for this synthetic reference are 2% Cs, 0.5% Cc,
1% field, 20 nm held-out intercept, and separately small numerical changes.
These deliberately bounded gates do not grant production or wave admission.
"""
from functools import lru_cache
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, m_e
from scipy.integrate import solve_ivp
from scipy.special import ellipe, ellipk

from temsim.physics.axisymmetric_magnetostatics import MU0, solve_axisymmetric
from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_ray import trace_potential_rays
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
from temsim.physics.regular_axisymmetric_magnetic_potential import RegularAxisymmetricMagneticPotential
from temsim.physics.round_lens_aberration_benchmark import _linear_cubic_reference


ENERGY_EV = 300000.
START_M = -.035
SEMIANGLE_RAD = .03
SOURCE_NOTE = "Uniform 3000 AT winding at r=4--6 mm, |z|<5 mm; mu_r=1 everywhere."
VACUUM_NOTE = "Entire declared cylinder lies at r<4 mm, strictly inside the annular current source; mu_r=1."


def _axis_derivatives(z, order=3):
    """Analytic axial derivatives of the finite-solenoid Biot--Savart law."""
    points, weights = _quadrature(48)
    radius, weights = .005+.001*points, .001*weights
    values = []
    for derivative in range(order+1):
        def primitive(t):
            d = radius*radius+t*t
            return (t/d**.5, radius*radius/d**1.5,
                    -3*radius*radius*t/d**2.5,
                    3*radius*radius*(4*t*t-radius*radius)/d**3.5)[derivative]
        values.append(MU0*(3000/(.002*.01))/2
                      *np.dot(weights, primitive(z+.005)-primitive(z-.005)))
    return np.asarray(values)


@lru_cache(maxsize=2)
def _quadrature(count):
    return np.polynomial.legendre.leggauss(count)


def _biot_savart_field(points):
    """Independent off-axis circular-loop elliptic integral, then coil integral."""
    nodes, weights = _quadrature(48)
    a, coil_z = np.meshgrid(.005+.001*nodes, .005*nodes, indexing="ij")
    currents = (3000/(.002*.01))*.001*.005*weights[:, None]*weights[None, :]
    a, coil_z, currents = a.ravel(), coil_z.ravel(), currents.ravel()
    x, y, z = np.asarray(points).T
    r = np.hypot(x, y)
    assert np.all(r > 0.)
    rr, zz = r[:, None], z[:, None]-coil_z
    d = (a-rr)**2+zz*zz
    denominator = np.sqrt((a+rr)**2+zz*zz)
    parameter = 4*a*rr/denominator**2
    first, second = ellipk(parameter), ellipe(parameter)
    factor = MU0*currents/(2*np.pi*denominator)
    br = np.sum(factor*zz/rr*(-first+(a*a+rr*rr+zz*zz)/d*second), axis=1)
    bz = np.sum(factor*(first+(a*a-rr*rr-zz*zz)/d*second), axis=1)
    return np.c_[br*x/r, br*y/r, bz]


@lru_cache(maxsize=1)
def _reference():
    kinetic = ENERGY_EV*e
    k = -e/(np.sqrt(kinetic*(kinetic+2*m_e*c*c))/c)
    def equation(z, state):
        field = _axis_derivatives(z, 0)[0]
        return state[1], -(k*field/2)**2*state[0], -k*field/2
    def focus(z, state):
        return state[0]
    focus.terminal, focus.direction = True, -1
    result = solve_ivp(equation, (START_M, .06), (1., 0., 0.), events=focus,
        method="DOP853", rtol=2e-12, atol=1e-13, max_step=.0002)
    assert result.success and len(result.t_events[0]) == 1
    image = float(result.t_events[0][0])
    _, slope, angle = result.y_events[0][0]
    initial_slopes = np.array([[0., .5*k*_axis_derivatives(START_M, 0)[0]],
                              [-.5*k*_axis_derivatives(START_M, 0)[0], 0.]])
    # This order-by-order ODE has no ray-fit or reconstructed-FEM call. The
    # existing perturbation solver is reused with independent analytic B0.
    field = SimpleNamespace(axial_derivatives=_axis_derivatives,
                            registration=CoordinateRegistration(), radial_order=3)
    delta = .0005
    inputs = (START_M, image, .0002, initial_slopes)
    central = _linear_cubic_reference(field, ENERGY_EV, *inputs)
    minus = _linear_cubic_reference(field, ENERGY_EV*(1-delta), *inputs)
    plus = _linear_cubic_reference(field, ENERGY_EV*(1+delta), *inputs)
    direction = np.array([np.cos(angle), np.sin(angle)])
    cs = -float(central[4:6]@direction)/(-slope)**3*1000
    cc = float((plus[:2]-minus[:2])@direction)/(2*delta)/(-slope)*1000
    return {"image_z_m": image, "radial_slope_m1": float(slope), "angle_rad": float(angle),
            "initial_slope_matrix_m1": initial_slopes, "cs_mm": cs, "cc_mm": cc}


def _extended_axis(inner, spacing, outer):
    points = list(np.linspace(0., inner, round(inner/spacing)+1))
    step = spacing
    while True:
        step *= 1.12
        if points[-1]+step >= outer:
            break
        points.append(points[-1]+step)
    return np.r_[points, outer]


@lru_cache(maxsize=4)
def _solved(inner_cells, boundary=.08):
    spacing = .008/inner_cells
    r = _extended_axis(.008, spacing, boundary)
    positive_z = _extended_axis(.02, spacing, boundary)
    z = np.r_[-positive_z[:0:-1], positive_z]
    def current(r, z):
        return np.where((r > .004) & (r < .006) & (np.abs(z) < .005), 3000/(.002*.01), 0.)
    solution = solve_axisymmetric(r, z, lambda r, z: np.ones_like(r), current)
    assert solution.relative_residual < 1e-9
    return AxisymmetricMagneticPotential.from_solution(solution, source_id="solved_air_core_lens",
        provenance={"source": SOURCE_NOTE, "radial_spacing_in_source_m": spacing,
                    "remote_boundary_m": boundary, "solver_relative_residual": solution.relative_residual})


def _regular(inner_cells=128, boundary=.08, window=.003, radial_degree=3, axial_degree=5):
    return RegularAxisymmetricMagneticPotential(_solved(inner_cells, boundary), window,
        radial_degree, axial_degree, valid_radius_m=.0015, vacuum_domain_note=VACUUM_NOTE)


def _measure(potential, step=.0001):
    reference = _reference()
    train = np.linspace(.004, SEMIANGLE_RAD, 12)
    holdout = (train[:-1]+train[1:])/2
    pupil = np.r_[train, holdout, 0.]
    positions = np.c_[pupil/(-reference["radial_slope_m1"]), pupil*0.]
    directions = positions@reference["initial_slope_matrix_m1"].T
    inputs = np.c_[positions, directions]
    deltas = np.array([0., -.002, .002, -.001, .001, -.0005, .0005])
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    snapshot = ElectromagneticPotentialSnapshot(scene,
        (PotentialContribution("solved_air_core_lens", ("solved_air_core_lens",), potential, "magnetic"),))
    digest = snapshot.numerical_digest
    result = trace_potential_rays(snapshot, np.tile(inputs, (len(deltas), 1)),
        (START_M, reference["image_z_m"]), energy_ev=np.repeat(ENERGY_EV*(1+deltas), len(pupil)),
        maximum_step_m=step, method="RK4")
    assert snapshot.numerical_digest == digest
    assert result.metadata["empirical_aberrations_applied"] is False
    assert result.metadata["production_admission"] is False
    final = result.xy_slopes[-1, :, :2].reshape(len(deltas), len(pupil), 2)
    values = (final[:, :, 0]+1j*final[:, :, 1])*np.exp(-1j*reference["angle_rad"])
    basis = np.array([(train/SEMIANGLE_RAD)**n for n in (1, 3, 5)]).T
    coefficients = np.linalg.lstsq(basis, values[:, :len(train)].T, rcond=None)[0].T
    prediction = coefficients@np.array([(holdout/SEMIANGLE_RAD)**n for n in (1, 3, 5)])
    cs = -coefficients[0, 1]/SEMIANGLE_RAD**3*1000
    cc = [(coefficients[b, 0]-coefficients[a, 0])/(2*abs(deltas[a]))/SEMIANGLE_RAD*1000
          for a, b in ((1, 2), (3, 4), (5, 6))]
    return {"cs_mm": float(cs.real), "cs_tangential_mm": float(cs.imag),
        "cc_mm": float(cc[-1].real), "cc_tangential_mm": float(cc[-1].imag),
        "cc_energy_steps_mm": [float(item.real) for item in cc],
        "holdout_rms_m": float(np.sqrt(np.mean(np.abs(prediction-values[:, len(train):-1])**2))),
        "chief_max_m": float(np.max(np.abs(values[:, -1]))), "snapshot_digest": digest,
        "raw_intercepts_m": final, "maximum_step_m": step}


def _record(record_testsuite_property, name, values):
    clean = [{k: v for k, v in row.items() if k != "raw_intercepts_m"} for row in values]
    record_testsuite_property(name, json.dumps(clean, sort_keys=True))


def test_resolved_fem_natural_cs_cc_converge_against_independent_source_reference(record_testsuite_property):
    reference = _reference()
    results = [_measure(_regular(count)) for count in (32, 64, 128)]
    _record(record_testsuite_property, "resolved_fem_mesh_aberrations", results)
    record_testsuite_property("independent_solenoid_reference", json.dumps({
        k: v for k, v in reference.items() if k != "initial_slope_matrix_m1"}, sort_keys=True))
    cs_error = np.abs(np.array([row["cs_mm"] for row in results])-reference["cs_mm"])/reference["cs_mm"]
    assert np.all(cs_error[1:] < .7*cs_error[:-1]), results
    assert cs_error[-1] < .02, results
    assert abs(results[-1]["cc_mm"]/reference["cc_mm"]-1) < .005, results
    assert abs(results[-1]["cs_tangential_mm"]) < .005*reference["cs_mm"]
    assert abs(results[-1]["cc_tangential_mm"]) < .005*reference["cc_mm"]
    assert all(row["holdout_rms_m"] < 2e-8 and row["chief_max_m"] == 0. for row in results)
    # Energy differences are recomputed on the same frozen source and plane.
    differences = np.abs(np.diff(results[-1]["cc_energy_steps_mm"]))
    assert differences[1] < .4*differences[0] and differences[1] < .0001


def test_reconstruction_orders_windows_and_ray_steps_are_independent_axes(record_testsuite_property):
    choices = [(3, .003, 3), (3, .003, 5), (2, .003, 5), (4, .003, 5), (3, .0025, 5)]
    reconstructed = [_measure(_regular(radial_degree=degree, window=window, axial_degree=axial))
                     for degree, window, axial in choices]
    _record(record_testsuite_property, "resolved_fem_reconstruction_choices", reconstructed)
    nominal = reconstructed[1]
    # The independent window/degree variation is an error estimate, never
    # an optimizer that selects the coefficient closest to the reference.
    assert max(abs(row["cs_mm"]-nominal["cs_mm"]) for row in reconstructed) < .02*nominal["cs_mm"]
    assert max(abs(row["cc_mm"]-nominal["cc_mm"]) for row in reconstructed) < .001*nominal["cc_mm"]
    assert all(row["holdout_rms_m"] < 2e-8 for row in reconstructed)
    steps = [_measure(_regular(), step) for step in (.0002, .0001, .00005)]
    _record(record_testsuite_property, "resolved_fem_ray_steps", steps)
    errors = [np.max(np.abs(a["raw_intercepts_m"]-b["raw_intercepts_m"])) for a, b in zip(steps, steps[1:])]
    assert errors[1] < .2*errors[0] and errors[1] < 1e-10, errors
    assert abs(steps[-1]["cs_mm"]-steps[-2]["cs_mm"]) < .001


def test_same_current_off_axis_field_and_remote_boundary_are_checked_separately(record_testsuite_property):
    z = np.linspace(-.025, .025, 51)+.000017
    probes = np.c_[np.linspace(.0001, .0013, len(z)), np.zeros_like(z), z]
    exact = _biot_savart_field(probes)
    grids = [_regular(count).sample(probes)["B_t"] for count in (32, 64, 128)]
    errors = [float(np.linalg.norm(value-exact)/np.linalg.norm(exact)) for value in grids]
    assert errors[1] < errors[0] and errors[2] < errors[1] and errors[-1] < .01, errors
    # Mesh and reconstructed window stay fixed while only remote faces move.
    fields = [_regular(64, boundary).sample(probes)["B_t"] for boundary in (.04, .06, .08)]
    boundary_errors = [float(np.linalg.norm(value-exact)/np.linalg.norm(exact)) for value in fields]
    assert boundary_errors[-1] < boundary_errors[0] and boundary_errors[-1] < .01, boundary_errors
    record_testsuite_property("resolved_fem_independent_field_errors", json.dumps({
        "mesh_relative_errors": errors, "boundary_relative_errors": boundary_errors,
        "mesh_inner_cells": [32, 64, 128], "remote_boundary_m": [.04, .06, .08]}))
    boundary_rays = [_measure(_regular(64, boundary)) for boundary in (.04, .06, .08)]
    _record(record_testsuite_property, "resolved_fem_natural_aberration_boundaries", boundary_rays)
    for name in ("cs_mm", "cc_mm"):
        changes = np.abs(np.diff([row[name] for row in boundary_rays]))
        assert changes[-1] < .4*changes[0]
        assert changes[-1] < .001*abs(boundary_rays[-1][name])


def test_source_free_maxwell_residual_decreases_without_being_forced_to_zero(record_testsuite_property):
    z = np.linspace(-.02, .02, 31)+.000017
    points = np.c_[np.linspace(.0001, .0013, len(z)), np.full(len(z), .0001), z]
    reference_norm = np.linalg.norm(_biot_savart_field(points))
    epsilon, length_scale = 1e-6, .005
    residuals = []
    for count in (32, 64, 128):
        potential = _regular(count)
        derivative = np.stack([(potential.sample(points+epsilon*np.eye(3)[axis])["B_t"]
                              -potential.sample(points-epsilon*np.eye(3)[axis])["B_t"])/(2*epsilon)
                              for axis in range(3)], axis=-1)
        curl = np.c_[derivative[:, 2, 1]-derivative[:, 1, 2],
                     derivative[:, 0, 2]-derivative[:, 2, 0],
                     derivative[:, 1, 0]-derivative[:, 0, 1]]
        divergence = np.trace(derivative, axis1=1, axis2=2)
        residuals.append((float(length_scale*np.linalg.norm(curl)/reference_norm),
                          float(length_scale*np.linalg.norm(divergence)/reference_norm)))
    # Curl B is not overridden to zero: it measures the represented potential's
    # source-free error. Divergence is a finite-difference check of curl A.
    residuals = np.asarray(residuals)
    assert np.all(residuals[1:, 0] < .5*residuals[:-1, 0])
    assert residuals[-1, 0] < .005 and np.all(residuals[:, 1] < 1e-6)
    record_testsuite_property("resolved_fem_vacuum_maxwell_residuals", json.dumps({
        "inner_cells": [32, 64, 128], "curl_divergence_relative": residuals.tolist(),
        "normalizing_length_m": length_scale, "difference_step_m": epsilon}))


def test_preserved_p1_basis_is_reported_without_claiming_natural_aberration_admission(record_testsuite_property):
    # This diagnostic guards the distinction between preserving a numerical
    # basis and qualifying its third-order electron optics. It does not alter
    # the original FEM representation or make it secretly use the smooth one.
    raw = _solved(32)
    results = [_measure(raw, step) for step in (.000025, .0000125)]
    _record(record_testsuite_property, "unqualified_raw_p1_fem_aberrations", results)
    assert raw.to_metadata()["wave_admission"] is False
    assert all(row["holdout_rms_m"] > 2e-8 for row in results), results
    assert abs(results[-1]["cs_tangential_mm"]) > .02*_reference()["cs_mm"], results
