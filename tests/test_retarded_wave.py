"""Retarded native scan waves preserve actual clocks and reject excess error."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.constants import c, e, hbar, m_e

from temsim.physics.dynamic_potential import DynamicPotentialSnapshot, DynamicDipolePotential, NativeTransverseDrive
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.potential_wave_observation import observe_potential_mode
from temsim.physics.potential_geometry import PotentialGeometrySnapshot, observe_potential_geometry_checkpoint
from temsim.physics.potential_phase_budget import KineticPhaseLedger
from temsim.physics.electric_wave_reference import ElectricWaveHistory, momentum_velocity
from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_reference import AxialWaveReference
from temsim.physics.multiplane_wave import PlaneWave
from test_dynamic_potential import coil
from test_potential_wave_carrier import _snapshot, _wave


def _source(*, period=1., offset=0.):
    original = coil()
    native = replace(original.base.base.field, polynomial_terms=((1, 0, 0.),),
        registration=replace(original.base.base.field.registration, origin_global_m=(offset, 0., 0.)))
    provider = DynamicDipolePotential(AnalyticPotentialDomain(AnalyticMagneticPotential(native)),
        original.owner_ids, (NativeTransverseDrive("scan", "raster", ((.01, 0.), (0., 0.)), period, 1),), 0.)
    return DynamicPotentialSnapshot(_snapshot(), (provider,))


def _mode():
    return WaveMode(_wave(n=48, width=2e-7, curvature=.1), .27, TIP_REFERENCE, "controlled:0", 300.,
                    AxialWaveReference(3e-9, 1e-22))


def test_retarded_scan_changes_raw_wave_and_uses_actual_elapsed_clock():
    source, mode = _source(), _mode()
    first, = observe_potential_mode(source, mode, .45, (.452,), maximum_step_m=2e-4, emission_time_s=.1)
    second, = observe_potential_mode(source, mode, .45, (.452,), maximum_step_m=2e-4, emission_time_s=.2)
    assert first.metadata["dynamic_clock"]["emission_time_s"] == .1
    assert first.metadata["dynamic_clock"]["initial_flight_time_s"] == 3e-9
    assert first.metadata["dynamic_clock"]["flight_time_s"] > 3e-9
    assert first.mode.axial_reference.flight_time_s == first.metadata["dynamic_clock"]["flight_time_s"]
    assert first.metadata["dynamic_clock"]["energy_spread_bound_ev"] > 0.
    assert 0. < first.metadata["dynamic_clock"]["phase_error_bound_rad"] < .01
    assert first.mode.weight_per_reference_electron == pytest.approx(.27, abs=1e-11)
    assert np.linalg.norm(first.mode.plane.tilt_rad-second.mode.plane.tilt_rad) > 1e-7


def test_dynamic_geometry_resume_preserves_clock_field_and_error_receipts():
    source, mode = _source(), _mode()
    initial = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 450., 1e-12, {})
    geometry = PotentialGeometrySnapshot(.45, .452, ())
    options = dict(maximum_step_m=2e-4, emission_time_s=.1)
    whole, = observe_potential_geometry_checkpoint(source, geometry, initial, (.452,),
        kinetic_phase_ledger=KineticPhaseLedger(.01), **options)
    partial, = observe_potential_geometry_checkpoint(source, geometry, initial, (.451,),
        kinetic_phase_ledger=KineticPhaseLedger(.01), **options)
    final, = observe_potential_geometry_checkpoint(source, geometry, partial, (.452,), **options)
    history = ElectricWaveHistory.from_dict(final.record["electric_wave_history"])
    assert history.reference.snapshot_digest == source.numerical_digest
    assert history.dynamic_state["flight_time_s"] == final.beam.modes[0].axial_reference.flight_time_s
    np.testing.assert_allclose(final.beam.modes[0].plane.origin_m, whole.beam.modes[0].plane.origin_m, rtol=1e-8, atol=1e-15)
    np.testing.assert_allclose(final.beam.modes[0].plane.tilt_rad, whole.beam.modes[0].plane.tilt_rad, rtol=1e-9, atol=1e-15)
    assert final.beam.total_weight == pytest.approx(whole.beam.total_weight, abs=1e-11)
    with pytest.raises(ValueError, match="emission clock"):
        observe_potential_geometry_checkpoint(source, geometry, partial, (.452,), maximum_step_m=2e-4, emission_time_s=.2)


def test_offset_native_scan_retains_nonzero_induced_axial_work():
    source, mode = _source(offset=1e-4), _mode()
    output, = observe_potential_mode(source, mode, .45, (.452,), maximum_step_m=2e-4, emission_time_s=.1)
    native = source.dynamic_coils[0]
    expected = native.coefficients(.1)[1][0]*(-1e-4)*.002
    assert output.metadata["dynamic_clock"]["axial_work_ev"] == pytest.approx(expected, rel=1e-10, abs=1e-20)
    assert output.metadata["electric_axial_integrals"]["electric_phase_correction_rad"] != 0.


def test_unresolved_scan_timing_and_flyback_are_explicit_failures():
    mode = _mode()
    with pytest.raises(ValueError, match="explicit emission"):
        observe_potential_mode(_source(), mode, .45, (.452,), maximum_step_m=2e-4)
    with pytest.raises(ValueError, match="raw-phase budget|flyback"):
        observe_potential_mode(_source(period=1e-8), mode, .45, (.452,), maximum_step_m=2e-4, emission_time_s=.1)


@pytest.mark.parametrize("offset", [0., 1e-4])
@pytest.mark.parametrize("method", ["taylor", "chebyshev"])
def test_actual_retarded_complex_wave_matches_independent_affine_force_gaussian(offset, method):
    source = _source(period=1e-4, offset=offset)
    sigma, pitch, count = 3e-7, 1e-7, 64
    axis = (np.arange(count)-count//2)*pitch
    x, y = np.meshgrid(axis, axis)
    initial = np.exp(-(x*x+y*y)/(4*sigma*sigma)+.37j)
    norm = np.linalg.norm(initial)
    mode = WaveMode(PlaneWave(initial/norm, np.eye(2)*pitch, np.zeros(2)), .27,
        TIP_REFERENCE, "gaussian:0", 300., AxialWaveReference(3e-9, 1e-22))
    epoch, length = 1e-5, .002
    answer, = observe_potential_mode(source, mode, .45, (.452,), maximum_step_m=1e-4,
        emission_time_s=epoch, reference_quadrature="cf4", residual_exponential_method=method)
    momentum, velocity = momentum_velocity(300000.)
    k = momentum/hbar
    coefficient, derivative, _ = source.dynamic_coils[0].coefficients(epoch+3e-9)
    force0, force1 = -e*coefficient[0]/momentum, -e*derivative[0]/(momentum*velocity)
    b = force0*length+force1*length**2/2
    shift = force0*length**2/2+force1*length**3/6
    scalar = -(force0**2*length**3/6+force0*force1*length**4/8+force1**2*length**5/40)-offset*b
    output = answer.mode.plane
    xx, yy = output.coordinates_m()
    denominator = 1+1j*length/(2*k*sigma*sigma)
    expected = np.exp(.37j)/(norm*denominator)*np.exp(-((xx-shift)**2+yy**2)/(4*sigma*sigma*denominator))
    expected *= np.sqrt(abs(np.linalg.det(output.basis_m))/pitch**2)*np.exp(1j*k*(b*xx+scalar))
    wavelength = 2*np.pi*hbar/momentum_velocity(answer.mode.energy_kev*1000.)[0]
    actual = output.full_amplitude(wavelength)
    error = np.linalg.norm(actual-expected)
    assert error < 3e-7
    if offset:
        assert abs(answer.metadata["electric_axial_integrals"]["dynamic_temporal_phase_rad"]) > 1e-7
        # Omitting the temporal term while claiming raw phase must fail this
        # independent principal-action comparison, even at weak acceleration.
        omitted = actual*np.exp(-1j*answer.metadata["electric_axial_integrals"]["dynamic_temporal_phase_rad"])
        assert np.linalg.norm(omitted-expected) > error*5
