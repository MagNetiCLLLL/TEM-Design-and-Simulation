"""Compact-ring steering regression; the local fixture is not a microscope fit."""
from types import SimpleNamespace

import numpy as np

from temsim.optics.aberrations import _corrector_trace_ratio
from temsim.optics.model import DeflectorPair
from temsim.physics import core
from temsim.physics.instrument_magnetic import active_column_events, events_overlapping_interval


def _state():
    hp = SimpleNamespace(
        key="probe_hp2_hexapole", z_mm=5., effective_length_mm=1., enabled=True,
        hexapole_strength_m3=lambda z: 5e7*np.exp(-.5*((np.asarray(z)-5.)/(1./2.355))**2),
    )
    dph = DeflectorPair("Local DPH fixture", "dph_fixture", 5., 30.,
                       2., 0., 0., 0., thickness_mm=2.)
    return SimpleNamespace(
        lenses=[], stigmators=[], corrector_elements=[hp], deflectors=[dph],
        beam_voltage_kv=300., step_mm=.05, history_step_mm=.05,
        acceleration_enabled=False, acceleration_backend="CPU", simulation_mode="ideal",
        projector_mode="diffraction", equivalent_image_lenses_enabled=False,
        sample=SimpleNamespace(z_mm=8.), simulation_time_s=.125,
        probe_corrector_installed=True,
    )


def test_corrector_reference_traces_share_finite_dph_events_and_respond(monkeypatch):
    state = _state()
    # A specified thin cubic impulse isolates the C3 diagnostic from fitted
    # lens calibration. The finite DPH field and distributed HP are real code.
    def spherical(z, _state):
        result = np.zeros_like(z)
        result[int(np.argmin(np.abs(z-6.)))] = 1e8
        return result
    monkeypatch.setattr(core, "spherical_aberration_kick_m3", spherical)
    original = core.propagate
    calls = []
    def traced(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(core, "propagate", traced)
    driven = _corrector_trace_ratio(state, "probe")
    hp = state.corrector_elements[0]
    expected = events_overlapping_interval(state, active_column_events(state),
        hp.z_mm-5*hp.effective_length_mm/2.355, state.sample.z_mm)
    assert expected == ((5., .002, 0.),)
    assert len(calls) == 3
    assert all(row["events"] is calls[0]["events"] for row in calls)
    assert calls[0]["events"] == expected
    assert [(row["include_spherical_aberration"], row["include_hexapole"]) for row in calls] == [
        (False, False), (True, False), (True, True)]
    assert np.isfinite(driven[:3]).all() and driven[1] > 0.
    state.deflectors[0].upper_x_mrad = 0.
    undriven = _corrector_trace_ratio(state, "probe")
    # The diagnostic's nonlinear error changes, not just its baseline centroid.
    assert abs(driven[2]-undriven[2]) > 1e-5*undriven[2]
    assert state.simulation_time_s == .125
