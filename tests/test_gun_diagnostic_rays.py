"""Zero-current gun diagnostics share transport without replacing emission."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.base import EmissionBundle, GunExitBundle, GunTraceResult
from temsim.optics.electron_gun import tracing


def _bundle():
    return EmissionBundle(np.array([0., 1e-9]), np.zeros(2), np.array([0., .01]),
        np.zeros(2), np.array([0., .1]), np.zeros(2), np.array([10, 20]))


def _gun():
    return SimpleNamespace(emitter=SimpleNamespace(emission_energy_ev=.3,
        surface_model=None, curvature_nm_inv=0.), validate=lambda: None)


def test_invalid_diagnostics_fail_before_transport_and_do_not_mutate_inputs(monkeypatch):
    gun, bundle = _gun(), _bundle()
    monkeypatch.setattr(tracing, "_trace_feg_to_exit",
        lambda *args, **kwargs: pytest.fail("Invalid probes reached transport"))
    invalid = (
        replace(bundle, weight=np.array([0., 1.])),
        replace(bundle, tx_rad=np.array([0., np.nan])),
        replace(bundle, y_m=np.zeros(3)),
        replace(bundle, ray_id=np.array([10, 10])),
        replace(bundle, ray_id=np.array([10., 20.])),
        replace(bundle, energy_offset_ev=np.array([0., -.3])),
    )
    for candidate in invalid:
        with pytest.raises(ValueError):
            tracing.trace_feg_diagnostic_to_exit(gun, candidate)
    np.testing.assert_array_equal(bundle.weight, [0., 0.])
    gun.emitter.surface_model = object()
    with pytest.raises(ValueError, match="surface source"):
        tracing.trace_feg_diagnostic_to_exit(gun, bundle)


def test_diagnostic_and_ordinary_dispatch_share_kernel_without_source_replacement(monkeypatch):
    gun, bundle, calls = _gun(), _bundle(), []
    gun.emit = lambda *args: pytest.fail("Diagnostic route sampled the real source")
    exiting = GunExitBundle(bundle.x_m, bundle.y_m, bundle.tx_rad, bundle.ty_rad,
        bundle.energy_offset_ev, bundle.weight, bundle.ray_id, np.ones(2, bool), np.ones(2))
    result = GunTraceResult(np.array([0., 1.]), np.zeros((2, 2)), np.zeros((2, 2)),
        np.zeros((2, 2)), np.zeros((2, 2)), exiting, np.full(2, np.nan), ("", ""),
        1e-5, 1e-5, 1e-5, monochromator_transmitted_current_a=1e-6)

    def kernel(current, count=None, **kwargs):
        calls.append((current, count, kwargs))
        return result

    monkeypatch.setattr(tracing, "_trace_feg_to_exit", kernel)
    ordinary = tracing.trace_feg_to_exit(gun, 193)
    diagnostic = tracing.trace_feg_diagnostic_to_exit(gun, bundle)
    assert ordinary is result and ordinary.emitted_current_a == 1e-5
    assert calls[0][1:] == (193, {"cancelled": None})
    copied = calls[1][2]["emitted_bundle"]
    assert copied is not bundle and not np.shares_memory(copied.x_m, bundle.x_m)
    np.testing.assert_array_equal(copied.tx_rad, bundle.tx_rad)
    assert diagnostic.exit_bundle is result.exit_bundle
    assert (diagnostic.emitted_current_a, diagnostic.dpa_transmitted_current_a,
            diagnostic.c1_transmitted_current_a, diagnostic.monochromator_transmitted_current_a) == (0., 0., 0., 0.)
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="Superseded"):
        tracing.trace_feg_diagnostic_to_exit(gun, bundle, cancelled=lambda: True)
    assert len(calls) == 2


def test_curved_launch_frame_is_retained_and_inconsistent_projection_is_rejected():
    from temsim.optics.electron_gun.tip_curvature import curve_bundle, MODEL
    gun, bundle = _gun(), _bundle()
    gun.emitter.curvature_nm_inv = .01
    gun.emitter.curvature_model = MODEL
    gun.emitter.virtual_source_fwhm_nm = 5.
    gun.emitter.angular_rms_mrad = 1.
    gun.emitter.angular_cutoff_mrad = 3.
    gun.emitter.coherence = None
    with pytest.raises(ValueError, match="actual launch frame"):
        tracing._diagnostic_emission(gun, bundle)
    curved = curve_bundle(bundle, gun.emitter)
    copied = tracing._diagnostic_emission(gun, curved)
    np.testing.assert_array_equal(copied.surface_position_m, curved.surface_position_m)
    np.testing.assert_array_equal(copied.surface_direction, curved.surface_direction)
    assert not np.shares_memory(copied.surface_direction, curved.surface_direction)
    object.__setattr__(curved, "tx_rad", np.zeros(2))
    with pytest.raises(ValueError, match="inconsistent"):
        tracing._diagnostic_emission(gun, curved)
