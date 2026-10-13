"""Focused empirical-estimator checks, not full-column qualification."""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_aperture_edge import (
    _density_rings, aggregate_aperture_edge_estimates, estimate_aperture_wall_edges,
)


def _wave(order=0):
    y, x = np.mgrid[-80:80, -80:80]*8e-6
    value = np.exp(-(x*x+y*y)/(2*(200e-6)**2)).astype(complex)
    if order:
        value *= (x+1j*y)/200e-6
    value /= np.linalg.norm(value)
    return PlaneWave(value, np.eye(2)*8e-6, np.zeros(2), np.eye(2)*-100., np.zeros(2))


def _arguments():
    step = np.eye(4)
    step[:2, 2:] = np.eye(2)*.005
    return dict(wavelength_m=2e-12, spherical_m3=1e5,
                paths=[SimpleNamespace(matrix=step, offset=np.zeros(4)) for _ in range(20)],
                plane_z_mm=np.linspace(0., 100., 21), bore_radii_mm=np.full(21, 2.),
                aperture_radius_m=50e-6, accepted_probability=1e-4)


def test_ring_sampler_preserves_shifted_carrier_and_complex_angular_channel():
    wavelength, sigma, q = 2e-12, 150e-6, 3.
    angle = .31
    basis = 4e-6*np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    origin = np.array((12e-6, -9e-6))
    y, x = np.mgrid[-80:80, -80:80]
    xy = origin[:, None, None]+np.einsum('ij,jyx->iyx', basis, np.stack((x, y)))
    physical_envelope = (xy[0]+1j*xy[1])/sigma**2*np.exp(-np.sum(xy*xy, axis=0)/(2*sigma*sigma))
    # old envelope carries the precise constant caused by the shifted chart.
    amplitude = physical_envelope*np.sqrt(abs(np.linalg.det(basis)))*np.exp(1j*np.pi/wavelength*q*(origin@origin))
    wave = PlaneWave(amplitude, basis, origin, np.eye(2)*q, q*origin)
    rings = _density_rings(wave, np.array((50e-6, 100e-6)), 64, wavelength)
    theta = np.arange(64)*2*np.pi/64
    radius = np.array((50e-6, 100e-6))[:, None]
    expected = radius/sigma**2*np.exp(-radius*radius/(2*sigma*sigma))*np.exp(1j*theta)
    np.testing.assert_allclose(rings, expected, rtol=2e-7, atol=3e-4)
    harmonics = np.fft.fft(rings, axis=1)/64
    assert np.sum(abs(harmonics[:, 1])**2)/np.sum(abs(harmonics)**2) > 1-1e-10


def test_actual_angular_modes_get_distinct_estimates_without_changing_input():
    records = []
    for order in (0, 1):
        wave = _wave(order)
        original = wave.amplitude.copy()
        record = estimate_aperture_wall_edges(wave, **_arguments())
        assert record['physical_wall_samples'] > 0
        assert record['empirical_edge_probability'] > 0
        assert record['feedback_absolute_row_sum'] < .2
        assert record['angular_power_relative_change'] < .005
        assert record['high_angular_power_fraction'] < .001
        assert not record['phase_fitted'] and not record['normalized']
        np.testing.assert_array_equal(wave.amplitude, original)
        json.dumps(record, allow_nan=False)
        records.append(record)
    assert records[1]['diagonal_edge_probability'] > records[0]['diagonal_edge_probability']


def test_mode_weighted_intensity_gate_retains_weak_dark_states():
    schema = 'empirical-aperture-wall-edge-v1'
    bright = dict(schema=schema, accepted_probability=.001, empirical_intensity_error_probability=2e-6)
    dark = dict(schema=schema, accepted_probability=1e-12, empirical_intensity_error_probability=1e-8)
    result = aggregate_aperture_edge_estimates([(1., bright), (1e-5, dark)])
    assert result['within_declared_accuracy']
    assert result['empirical_relative_intensity_error'] == pytest.approx((2e-6+1e-13)/(.001+1e-17))
    assert not aggregate_aperture_edge_estimates([(1., bright)], maximum_relative_error=.001)['within_declared_accuracy']
    with pytest.raises(ValueError, match='own empirical'):
        aggregate_aperture_edge_estimates([(1., {})])


@pytest.mark.parametrize('invalid', ['offset', 'nonround', 'curvature', 'nodes'])
def test_uncovered_geometry_is_refused(invalid):
    wave = _wave()
    arguments = _arguments()
    if invalid == 'offset':
        arguments['paths'][0] = SimpleNamespace(matrix=arguments['paths'][0].matrix, offset=np.array((1e-6, 0., 0., 0.)))
    elif invalid == 'nonround':
        step = arguments['paths'][0].matrix.copy()
        step[0, 2] *= 2
        arguments['paths'][0] = SimpleNamespace(matrix=step, offset=np.zeros(4))
    elif invalid == 'curvature':
        wave = replace(wave, curvature_m1=np.diag((-100., -90.)))
    else:
        arguments['plane_z_mm'] = arguments['plane_z_mm'][:-1]
    with pytest.raises(ValueError):
        estimate_aperture_wall_edges(wave, **arguments)


def test_device_field_estimate_matches_cpu():
    from test_wave_device import _cuda
    cp = _cuda()
    wave = _wave(1)
    cpu = estimate_aperture_wall_edges(wave, **_arguments())
    gpu = estimate_aperture_wall_edges(replace(wave, amplitude=cp.asarray(wave.amplitude)), **_arguments())
    assert gpu['empirical_edge_probability'] == pytest.approx(cpu['empirical_edge_probability'], rel=1e-13)
