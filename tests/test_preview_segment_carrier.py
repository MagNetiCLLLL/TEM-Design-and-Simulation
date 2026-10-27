"""Actual CPU carrier propagation after optional private segment preparation."""
import math
import weakref
from dataclasses import replace

import numpy as np
import pytest

import temsim.physics.potential_wave_carrier as carrier
import temsim.physics.potential_segment_resampling as preparation
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError
from test_potential_wave_carrier import _wave, _snapshot, _round, _multipole, ENERGY_EV, WAVELENGTH


def _case(*, steps=1, magnetic=False):
    wave = _wave(n=128, pitch=1e-7, width=8e-7, curvature=.2, tilt=(2e-7, -1e-7))
    snapshot = _snapshot(_round(.02), _multipole()) if magnetic else _snapshot()
    nodes = .001+np.arange(steps+1)*1e-6
    ledger = NumericalSupportLedger(.02, 1e-6)
    numerics = WaveGridNumerics(maximum_pixels=512, maximum_working_bytes=128*1024**2)
    options = dict(energy_ev=ENERGY_EV, kinetic_order=8, scalar_splitting=True,
        higher_order_scalar_splitting=True, residual_compute_backend="CPU",
        support_ledger=ledger, support_mode_weight=.27, grid_numerics=numerics,
        support_projection_policy="preview_segment_and_retry", segment_start_m=float(nodes[0]))
    return snapshot, wave, nodes, options


def _physical_values(wave, origin, common_basis, px, py):
    """Evaluate the finite Fourier envelope and full analytical phase at SI points.

    These fixtures have parallel lattice axes, but may have different periods,
    pitches and sizes. Each represented period is zero-extended outside its
    physical domain; no periodic copies, phase alignment or norm repair are
    used in the common physical integration window.
    """
    transform = np.linalg.solve(wave.basis_m, common_basis)
    diagonal = np.diag(np.diag(transform))
    np.testing.assert_allclose(transform, diagonal, rtol=0., atol=2e-13)
    offset = np.linalg.solve(wave.basis_m, origin-wave.origin_m)
    qx, qy = offset[0]+transform[0, 0]*px, offset[1]+transform[1, 1]*py
    ny, nx = wave.amplitude.shape
    fy, fx = np.fft.fftshift(np.fft.fftfreq(ny)), np.fft.fftshift(np.fft.fftfreq(nx))
    coefficients = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave.amplitude)))/(nx*ny)
    envelope = (np.exp(2j*np.pi*np.outer(qy, fy))@coefficients
                @np.exp(2j*np.pi*np.outer(fx, qx)))/math.sqrt(abs(np.linalg.det(wave.basis_m)))
    inside = ((qy[:, None] >= -ny//2) & (qy[:, None] < ny-ny//2)
              & (qx[None, :] >= -nx//2) & (qx[None, :] < nx-nx//2))
    x, y = np.meshgrid(px, py)
    xy = np.einsum("ij,jyx->iyx", common_basis, np.stack((x, y)))
    xy += (origin-wave.origin_m)[:, None, None]
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    action = .5*np.einsum("iyx,ij,jyx->yx", xy, q, xy)+np.einsum("i,iyx->yx", tilt, xy)
    return np.where(inside, envelope*np.exp(2j*np.pi*action/WAVELENGTH), 0.)


def _common_comparison(first, second, samples):
    basis, origin = first.basis_m, first.origin_m
    lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
    for wave in (first, second):
        transform = np.linalg.solve(basis, wave.basis_m)
        offset = np.linalg.solve(basis, wave.origin_m-origin)
        n = np.array(wave.amplitude.shape[::-1])
        corners = np.array([[x, y] for x in (-n[0]/2, n[0]/2)
                            for y in (-n[1]/2, n[1]/2)])
        corners = corners@transform.T+offset
        lower = np.minimum(lower, corners.min(axis=0))
        upper = np.maximum(upper, corners.max(axis=0))
    pitch = (upper-lower)/samples
    px, py = [lower[i]+(np.arange(samples)+.5)*pitch[i] for i in range(2)]
    a, b = [_physical_values(wave, origin, basis, px, py) for wave in (first, second)]
    area = abs(np.linalg.det(basis))*math.prod(pitch)
    return dict(raw_complex_l2=math.sqrt(float(np.sum(abs(a-b)**2))*area),
        intensity_l1=float(np.sum(abs(abs(a)**2-abs(b)**2)))*area,
        reference_probability=float(np.sum(abs(a)**2))*area,
        candidate_probability=float(np.sum(abs(b)**2))*area)


@pytest.mark.parametrize("steps,magnetic,broad", [(1, False, False), (2, False, False),
    (1, True, False), (2, True, False), (1, False, True)])
def test_actual_preview_is_compared_on_common_physical_coordinates(steps, magnetic, broad, record_property):
    snapshot, wave, nodes, options = _case(steps=steps, magnetic=magnetic)
    if broad:
        options["support_ledger"] = NumericalSupportLedger(.5, .01)
        options["grid_numerics"] = replace(options["grid_numerics"],
            maximum_accumulated_support_l2=.5, maximum_support_loss_fraction=.01,
            exploratory_support_budget=True)
    before_bytes, ledger = wave.amplitude.tobytes(), options["support_ledger"]
    old_options = {**options, "support_projection_policy": "weighted_segment_and_retry", "segment_start_m": None}
    reference, old_record = carrier.propagate_carrier_potential(snapshot, wave, nodes, **old_options)
    candidate, record = carrier.propagate_carrier_potential(snapshot, wave, nodes, **options)
    accepted = [row for row in record["canonical_sampling"]
                if row["method"] == "segment preview projection accepted after physical residual"]
    assert len(accepted) == 1 and accepted[0]["z_m"] == nodes[0]
    assert record["new_support_event_count"] > 0
    assert old_record["new_support_event_count"] == 0
    assert not record["normalization_applied"]
    assert wave.amplitude.tobytes() == before_bytes and ledger.events == ()
    for event in record["support_ledger"]["events"]:
        assert event["mode_weight"] == .27
    coarse, fine = _common_comparison(reference, candidate, 256), _common_comparison(reference, candidate, 512)
    # This is an observed local comparison, not a propagated-error certificate
    # inferred from the smaller input-projection ledger.
    assert fine["raw_complex_l2"] < (1e-2 if broad else 1e-3)
    assert fine["intensity_l1"] < (1e-2 if broad else 1e-3)
    assert abs(coarse["raw_complex_l2"]-fine["raw_complex_l2"]) < 1e-7
    assert fine["reference_probability"] == pytest.approx(reference.probability, abs=2e-8)
    assert fine["candidate_probability"] == pytest.approx(candidate.probability, abs=2e-8)
    record_property("physical_common_grid_256", str(coarse))
    record_property("physical_common_grid_512", str(fine))
    record_property("reference_shape", str(reference.amplitude.shape))
    record_property("preview_shape", str(candidate.amplitude.shape))
    reference_period = reference.basis_m*np.array(reference.amplitude.shape)[::-1]
    candidate_period = candidate.basis_m*np.array(candidate.amplitude.shape)[::-1]
    if broad:
        assert not np.allclose(reference_period, candidate_period, rtol=1e-5, atol=0.)
    record_property("reference_period_m", str(reference_period.tolist()))
    record_property("candidate_period_m", str(candidate_period.tolist()))
    record_property("propagated_error_scope", "Observed local raw complex comparison, not certified by input projection cost")


def test_preview_offer_occurs_only_at_exact_segment_first_half(monkeypatch):
    snapshot, wave, nodes, options = _case(steps=3)
    seen = []
    original = preparation.prepare_segment_resampling
    def inspect(candidate, ledger, **kwargs):
        seen.append((candidate, ledger, kwargs["z_m"], kwargs["mode_weight"]))
        return original(candidate, ledger, **kwargs)
    monkeypatch.setattr(preparation, "prepare_segment_resampling", inspect)
    _, record = carrier.propagate_carrier_potential(snapshot, wave, nodes, **options)
    assert len(seen) == 1 and seen[0] == (wave, options["support_ledger"], nodes[0], .27)
    assert len(record["residual_steps"]) == 6


@pytest.mark.parametrize("policy,start", [("weighted_segment_and_retry", None),
    ("preview_segment_and_retry", None), ("preview_segment_and_retry", .0009)])
def test_old_policy_and_already_started_segment_do_not_offer_projection(monkeypatch, policy, start):
    snapshot, wave, nodes, options = _case(steps=2)
    monkeypatch.setattr(preparation, "prepare_segment_resampling",
        lambda *_a, **_k: pytest.fail("This entrance is not a new preview segment"))
    _, record = carrier.propagate_carrier_potential(snapshot, wave, nodes,
        **{**options, "support_projection_policy": policy, "segment_start_m": start})
    assert record["support_ledger"] == options["support_ledger"].to_dict()
    assert record["new_support_event_count"] == 0


def test_rejected_physical_candidate_retries_original_wave_without_spending(monkeypatch):
    snapshot, wave, nodes, options = _case()
    reference, _ = carrier.propagate_carrier_potential(snapshot, wave, nodes,
        **{**options, "support_projection_policy": "weighted_segment_and_retry", "segment_start_m": None})
    original = carrier._residual
    calls, rejected = [], []
    def inspect(snapshot, candidate, *args, **kwargs):
        calls.append(candidate.amplitude.shape)
        if len(calls) == 1:
            scratch = np.ones((31, 31))
            rejected.append(weakref.ref(scratch))
            rejected.append(weakref.ref(candidate.amplitude))
            raise carrier.OccupiedCarrierBoundaryError("Private candidate physical boundary rejection")
        assert all(reference() is None for reference in rejected)
        return original(snapshot, candidate, *args, **kwargs)
    monkeypatch.setattr(carrier, "_residual", inspect)
    output, record = carrier.propagate_carrier_potential(snapshot, wave, nodes, **options)
    assert calls[0] != wave.amplitude.shape and calls[1] == wave.amplitude.shape
    assert output.amplitude.tobytes() == reference.amplitude.tobytes()
    assert record["support_ledger"] == options["support_ledger"].to_dict()
    assert record["new_support_event_count"] == 0


@pytest.mark.parametrize("kind", ["domain", "cancel", "later_canonical"])
def test_failed_segment_never_publishes_a_wave_or_spends_external_history(monkeypatch, kind):
    snapshot, wave, nodes, options = _case()
    original, cancelled, progress, calls = carrier._residual, [False], [], []
    def inspect(snapshot, candidate, *args, **kwargs):
        calls.append(candidate.amplitude.shape)
        if kind == "domain":
            raise ValueError("Unsupported physical field domain")
        result = original(snapshot, candidate, *args, **kwargs)
        if kind == "cancel":
            cancelled[0] = True
        return result
    monkeypatch.setattr(carrier, "_residual", inspect)
    if kind == "later_canonical":
        def reject(*_a, **_k):
            raise ValueError("Later physical canonical failure")
        monkeypatch.setattr(carrier, "propagate_plane_wave", reject)
    expected = InterruptedError if kind == "cancel" else ValueError
    with pytest.raises(expected):
        carrier.propagate_carrier_potential(snapshot, wave, nodes,
            cancelled=lambda: cancelled[0], progress_callback=progress.append, **options)
    assert len(calls) == 1 and calls[0] != wave.amplitude.shape
    assert options["support_ledger"].events == () and progress == []


def test_refined_private_proposal_is_counted_in_inner_residual_host_allowance(monkeypatch):
    from temsim.cache_memory import retained_memory_inventory
    snapshot, wave, nodes, options = _case()
    real_prepare, real_residual = preparation.prepare_segment_resampling, carrier._residual
    captured, calls = {}, []
    def prepare(*args, **kwargs):
        proposal = real_prepare(*args, **kwargs)
        captured["proposal"] = weakref.ref(proposal.wave.amplitude)
        captured["proposal_bytes"] = proposal.wave.amplitude.nbytes
        return proposal
    def residual(snapshot, candidate, *args, **kwargs):
        calls.append((candidate.amplitude.shape, kwargs["maximum_host_working_bytes"]))
        if len(calls) == 1:
            raise WaveSamplingError("Controlled candidate needs one lossless refinement", required_scale=1.01)
        if len(calls) == 2:
            assert captured["proposal"]() is not None
            retained = sum(retained_memory_inventory(wave.amplitude, captured["proposal"]()).values())
            expected = options["grid_numerics"].maximum_working_bytes-retained
            assert kwargs["maximum_host_working_bytes"] == expected
        return real_residual(snapshot, candidate, *args, **kwargs)
    monkeypatch.setattr(preparation, "prepare_segment_resampling", prepare)
    monkeypatch.setattr(carrier, "_residual", residual)
    _, record = carrier.propagate_carrier_potential(snapshot, wave, nodes, **options)
    assert calls[1][0] != calls[0][0]
    assert record["new_support_event_count"] > 0
    assert captured["proposal"]() is None


def test_real_gpu_preview_segment_h8_matches_cpu_fields_weights_and_ledger():
    from temsim.physics.compute_backend import cupy_capability
    if not cupy_capability().available:
        pytest.skip("Optional real CUDA device unavailable")
    snapshot, wave, nodes, options = _case(steps=2, magnetic=True)
    original = wave.amplitude.tobytes()
    options["grid_numerics"] = replace(options["grid_numerics"],
        maximum_device_working_bytes=256*1024**2)
    cpu, a = carrier.propagate_carrier_potential(snapshot, wave, nodes, **options)
    gpu, b = carrier.propagate_carrier_potential(snapshot, wave, nodes,
        **{**options, "residual_compute_backend": "Require GPU"})
    assert type(gpu.amplitude) is np.ndarray
    assert a["residual_backends"] == ["numpy"] and b["residual_backends"] == ["cupy"]
    assert cpu.amplitude.shape == gpu.amplitude.shape
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(cpu, name), getattr(gpu, name))
    # Equal physical lattices and exactly equal retained analytic carriers make
    # this envelope difference also the full-complex physical L2 difference.
    assert np.linalg.norm(gpu.amplitude-cpu.amplitude) < 2e-13
    assert gpu.probability == pytest.approx(cpu.probability, rel=2e-13)
    assert a["input_flux"] == b["input_flux"] == wave.probability
    assert not a["normalization_applied"] and not b["normalization_applied"]
    assert a["support_ledger"] == b["support_ledger"]
    assert a["new_support_event_count"] == b["new_support_event_count"] > 0
    assert all(row["mode_weight"] == .27 for row in b["support_ledger"]["events"])
    for record in (a, b):
        accepted = [row for row in record["canonical_sampling"]
                    if row["method"] == "segment preview projection accepted after physical residual"]
        assert len(accepted) == 1 and accepted[0]["z_m"] == nodes[0]
        assert len(record["residual_steps"]) == 4
        assert all(row["boundary_check"] == "each bounded exponential substep"
                   for row in record["residual_steps"])
    assert wave.amplitude.tobytes() == original and options["support_ledger"].events == ()
