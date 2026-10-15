"""Small-matrix integration of experimental Gaussian replay stages."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.constants import e

from scripts.experimental_gaussian_beamlets import append_physical_column_step, propagate_affine
from scripts import gaussian_quadratic_paths as quadratic
from scripts.validate_experimental_gaussian_beamlets import packet
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.column_wave import _linear_factor
from temsim.physics.tip_gun_wave import _momentum_velocity


def test_integrated_lift_composition_matches_separate_gaussian_actions():
    rng = np.random.default_rng(19)
    source = packet(centre=(1e-7, -2e-7), real=[[10., 20.], [20., -50.]])
    for _ in range(80):
        combined, sequential = CanonicalPath(1e-3), source
        for index in range(3):
            angle = rng.uniform(-np.pi, np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            stiffness = rotation@np.diag(rng.uniform(2e5, 1e7, 2))@rotation.T
            generator = np.block([[np.zeros((2, 2)), np.eye(2)], [-stiffness, np.zeros((2, 2))]])
            force = np.array([0., 0., .001, -.002])*(index+1)
            step = CanonicalPath((index+1)*.0007)
            _linear_factor(step, generator, rng.uniform(1e-5, .002), force=force)
            append_physical_column_step(combined, step)
            sequential = propagate_affine(sequential, step.matrix, step.offset, **step.phase_kwargs())
        actual = propagate_affine(source, combined.matrix, combined.offset, **combined.phase_kwargs())
        np.testing.assert_allclose(actual.coefficients, sequential.coefficients, rtol=3e-10, atol=3e-10)
        np.testing.assert_allclose(actual.precision_m1, sequential.precision_m1, rtol=2e-9, atol=2e-6)


@dataclass
class Plan:
    z_mm: np.ndarray
    step_m: np.ndarray
    midpoint_magnetic_t: np.ndarray
    midpoint_sx_m2: np.ndarray
    midpoint_sy_m2: np.ndarray
    midpoint_sxy_m2: np.ndarray
    midpoint_hex_normal_m3: np.ndarray
    midpoint_hex_skew_m3: np.ndarray
    cs_kick_m3: np.ndarray
    kick_x_rad: np.ndarray
    kick_y_rad: np.ndarray
    dipole_bx_t: np.ndarray
    dipole_by_t: np.ndarray
    electric_field: object = None


def fixture_plan(z):
    z = np.asarray(z, float)
    n = len(z)-1
    return Plan(z, np.diff(z)*1e-3, *[np.zeros(n) for _ in range(6)],
                *[np.zeros(n+1) for _ in range(3)], *[np.zeros(3*n) for _ in range(2)])


def test_quadratic_microsteps_are_fourth_order_and_reversible(monkeypatch):
    momentum, _ = _momentum_velocity(200_000.)
    def provider(z_mm, state):
        z = np.asarray(z_mm)*1e-3
        return .018+.012*np.sin(400*z), 2e5*(1+.2*np.sin(700*z)), -1e5*(1+.3*np.cos(500*z))
    monkeypatch.setattr(quadratic, 'fields', provider)
    monkeypatch.setattr(quadratic, 'skew_quadrupole_field', lambda z, state: 3e4*np.cos(300*np.asarray(z)*1e-3))
    def rhs(z, values):
        magnetic, sx, sy = provider(z*1e3, None)
        b = -e*magnetic/2
        g = b/momentum
        rotation = np.array([[0., g], [-g, 0.]])
        stiff = np.array([[sx+(b/momentum)**2, 3e4*np.cos(300*z)],
                          [3e4*np.cos(300*z), sy+(b/momentum)**2]])
        generator = np.block([[rotation, np.eye(2)], [-stiff, rotation]])
        return (generator@values.reshape(4, 4)).ravel()
    reference = solve_ivp(rhs, (0., .003), np.eye(4).ravel(), method='DOP853',
                          rtol=2e-13, atol=1e-14).y[:, -1].reshape(4, 4)
    scale = np.diag([1e-3, 1e-3, 1., 1.])
    source = packet()
    errors = []
    for maximum in (.3, .15, .075):
        paths, record = quadratic.integrated_quadratic_stage_paths(None, fixture_plan([0., 1.2, 3.]),
            200., maximum_step_mm=maximum)
        combined = CanonicalPath(1e-3)
        for path in paths:
            append_physical_column_step(combined, path)
        errors.append(np.linalg.norm(np.linalg.solve(scale, (combined.matrix-reference)@scale)))
        assert record['outer_stage_count'] == 2
        backward, _ = quadratic.integrated_quadratic_stage_paths(None, fixture_plan([3., 1.2, 0.]),
            200., maximum_step_mm=maximum)
        evolved = propagate_affine(source, combined.matrix, combined.offset, **combined.phase_kwargs())
        for path in backward:
            evolved = propagate_affine(evolved, path.matrix, path.offset, **path.phase_kwargs())
        np.testing.assert_allclose(evolved.coefficients, source.coefficients, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(evolved.precision_m1, source.precision_m1, rtol=2e-10, atol=1e-6)
    assert errors[0]/errors[1] > 13 and errors[1]/errors[2] > 13, errors


def test_quadratic_budget_cancel_and_finite_drive_reject_before_provider(monkeypatch):
    monkeypatch.setattr(quadratic, 'fields', lambda *args: pytest.fail('Provider called before admission'))
    plan = fixture_plan([0., 1.])
    with pytest.raises(MemoryError):
        quadratic.integrated_quadratic_stage_paths(None, plan, 200., maximum_working_bytes=1)
    with pytest.raises(InterruptedError):
        quadratic.integrated_quadratic_stage_paths(None, plan, 200., cancelled=lambda: True)
    plan.dipole_bx_t[1] = .001
    with pytest.raises(ValueError, match='zero finite dipole'):
        quadratic.integrated_quadratic_stage_paths(None, plan, 200.)
