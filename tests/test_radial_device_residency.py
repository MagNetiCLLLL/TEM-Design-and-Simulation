"""CUDA radial chains preserve the full complex field and host handoff."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import radial_column_wave as radial
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.radial_fftlog import complex_hankel, hankel_backend
from temsim.physics.wave_grid import WaveSamplingError


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable: no accessible CUDA runtime")
    return cp


def _source(order=0):
    radius = np.geomspace(1e-10, 100., 32768, endpoint=False)
    import math
    amplitude = (radius**abs(order)*np.exp(-radius**2/2+.37j)
                 /np.sqrt(np.pi*math.factorial(abs(order))))
    return radial.RadialWave(radius, amplitude, angular_order=order)


def _field_error(expected, actual):
    np.testing.assert_allclose(actual.radius_m, expected.radius_m, rtol=3e-14, atol=0.)
    assert actual.curvature_m1 == pytest.approx(expected.curvature_m1, rel=2e-14, abs=2e-14)
    assert actual.angular_order == expected.angular_order
    # No fitted scalar phase, shift, amplitude rescaling or renormalisation.
    return np.sqrt(2*np.pi*expected.dln*np.sum(abs(expected.radius_m*(actual.amplitude-expected.amplitude))**2))


@pytest.mark.parametrize("order", (0, -2, 1))
@pytest.mark.parametrize("chart", ("drift", "fourier"))
def test_full_complex_phase_cs_stop_stay_on_device_until_handoff(cupy, monkeypatch, order, chart):
    source = _source(order)
    path = CanonicalPath(1.)
    if chart == "drift":
        matrix = np.eye(4); matrix[:2, 2:] = .2*np.eye(2)
        path.append(matrix)
    else:
        angle = np.pi/32
        step = np.block([[np.cos(angle)*np.eye(2), np.sin(angle)*np.eye(2)],
                         [-np.sin(angle)*np.eye(2), np.cos(angle)*np.eye(2)]])
        for _ in range(16):
            path.append(step)
    def execute(wave):
        wave = radial.propagate_radial(wave, path.matrix, 2*np.pi,
            reference_phase_rad=path.reference_phase_rad, reference_length_m=1.)
        wave = radial.radial_cs(wave, 2*np.pi, .1)
        return radial._radial_stop(wave, 1.7)
    with hankel_backend("cpu"):
        expected = execute(source)
    with hankel_backend("cuda"):
        device = radial.to_device(source)
        def forbidden(*args, **kwargs):
            raise AssertionError("Intermediate full field must not be downloaded")
        with monkeypatch.context() as local:
            local.setattr(cupy, "asnumpy", forbidden)
            result = execute(device)
            assert isinstance(result.amplitude, cupy.ndarray)
            assert isinstance(result.radius_m, cupy.ndarray)
            assert result.amplitude.dtype == cupy.complex128
            assert result.radius_m.dtype == cupy.float64
        actual = radial.to_host(result)
    assert _field_error(expected, actual) < 2e-10
    assert actual.probability == pytest.approx(expected.probability, rel=2e-11, abs=1e-14)


def test_existing_hankel_host_contract_and_explicit_native_return(cupy):
    source = _source()
    with hankel_backend("cuda"):
        host = complex_hankel(source.radius_m*source.amplitude, source.dln, -.3)
        native = complex_hankel(cupy.asarray(source.radius_m*source.amplitude),
                               source.dln, -.3, keep_on_device=True)
        assert isinstance(host, np.ndarray)
        assert isinstance(native, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(native), host)


def test_device_expansion_refinement_and_angular_order_preserve_field(cupy):
    source = _source(-2)
    with hankel_backend("cuda"):
        device = radial.to_device(source)
        expanded = radial.expand_radial_domain(device)
        refined = radial.refine_radial(device, 1.5)
        assert isinstance(expanded.amplitude, cupy.ndarray)
        assert isinstance(refined.amplitude, cupy.ndarray)
        expanded = radial.to_host(expanded); refined = radial.to_host(refined)
    cpu_expanded = radial.expand_radial_domain(source)
    cpu_refined = radial.refine_radial(source, 1.5)
    assert _field_error(cpu_expanded, expanded) < 2e-12
    assert _field_error(cpu_refined, refined) < 2e-12
    assert expanded.probability == pytest.approx(source.probability, rel=2e-12)
    assert refined.probability == pytest.approx(source.probability, rel=2e-12)


def test_device_sampling_and_domain_guards_remain_active(cupy):
    with hankel_backend("cuda"):
        with pytest.raises(WaveSamplingError, match="phase needs refinement"):
            radial._phase_sampling(cupy.ones(8, dtype=cupy.complex128), cupy.arange(8)*2., "fixture")
        with pytest.raises(radial.RadialDomainError, match="boundary amplitude"):
            radial._check_hankel_edges(cupy.ones(100, dtype=cupy.complex128))


def test_production_round_prefix_uploads_once_returns_host_and_keeps_losses(cupy, monkeypatch):
    """Exercise the production dispatcher and actual column transport chain."""
    from temsim.physics import column_wave
    nodes = 5
    plan = SimpleNamespace(z_mm=np.linspace(0., 1., nodes), step_m=np.full(nodes-1, .00025),
        mapped_fields=(), signature="fixed-round-C3-stop",
        midpoint_magnetic_t=np.full(nodes-1, .02), cs_kick_m3=np.array((0., 2e9, 2e9, 2e9, 0.)),
        kick_x_rad=np.zeros(nodes), kick_y_rad=np.zeros(nodes))
    for name in ("midpoint_sx_m2", "midpoint_sy_m2", "midpoint_sxy_m2",
                 "midpoint_hex_normal_m3", "midpoint_hex_skew_m3"):
        setattr(plan, name, np.zeros(nodes-1))
    stops = {nodes-1: (SimpleNamespace(radius_mm=.0013, key="fixture-aperture"),)}
    monkeypatch.setattr(column_wave, "_prepare_column", lambda *args: (plan, np.full(nodes, np.inf), stops, ()))
    radius = np.geomspace(1e-12, 1e-4, 32768, endpoint=False)
    source = radial.RadialWave(radius, np.exp(-radius**2/(2e-12)+.37j)/(np.sqrt(np.pi)*1e-6))
    monkeypatch.setattr(radial, "from_executed_gun", lambda *args: source)
    mode = SimpleNamespace(mode_id="same-executed-mode", energy_kev=300., weight_per_reference_electron=.7)
    checkpoint = SimpleNamespace(plane_z_mm=0., digest="fixture-only", beam=SimpleNamespace(modes=(mode,)),
        record={"radial_output_modes": ({"mode_id": mode.mode_id},)})
    state = SimpleNamespace(simulation_mode="ideal", beam_voltage_kv=300.)
    cpu = radial.round_column_prefix(state, checkpoint, 1., backend="cpu", initial_samples=len(radius))
    uploaded, downloaded = [], []
    upload = radial.to_device
    download = radial.to_host
    def counted_upload(wave):
        uploaded.append(len(wave.amplitude)); return upload(wave)
    def counted_download(wave):
        downloaded.append(len(wave.amplitude)); return download(wave)
    monkeypatch.setattr(radial, "to_device", counted_upload)
    monkeypatch.setattr(radial, "to_host", counted_download)
    gpu = radial.round_column_prefix(state, checkpoint, 1., backend="cuda", initial_samples=len(radius))
    assert uploaded == [len(radius)]
    assert downloaded == [len(radius)]
    assert gpu[0] == cpu[0] == 1.
    assert gpu[1][0][0] is mode
    actual, expected = gpu[1][0][1], cpu[1][0][1]
    assert isinstance(actual.amplitude, np.ndarray)
    assert isinstance(actual.radius_m, np.ndarray)
    assert _field_error(expected, actual) < 2e-10
    assert actual.probability == pytest.approx(expected.probability, rel=2e-11)
    assert .7 < actual.probability < .9  # The final physical stop still absorbs.
    for cpu_loss, gpu_loss in zip(cpu[2]["modes"][0]["losses"], gpu[2]["modes"][0]["losses"]):
        assert gpu_loss["component"] == cpu_loss["component"]
        assert gpu_loss["outgoing"] == pytest.approx(cpu_loss["outgoing"], rel=2e-11)
    assert gpu[2]["radial_fft_backend"] == "cuda"
    assert gpu[2]["precision"] == cpu[2]["precision"]


def test_device_failures_propagate_without_cpu_retry(cupy, monkeypatch):
    import temsim.physics.radial_fftlog as fftlog
    with hankel_backend("cuda"):
        wave = radial.to_device(_source())
        def fail(*args, **kwargs):
            raise RuntimeError("device execution failure")
        monkeypatch.setattr(fftlog.Hankel, "apply", fail)
        path = CanonicalPath(1.)
        matrix = np.eye(4); matrix[:2, 2:] = .2*np.eye(2); path.append(matrix)
        with pytest.raises(RuntimeError, match="device execution failure"):
            radial.propagate_radial(wave, matrix, 2*np.pi,
                reference_phase_rad=path.reference_phase_rad, reference_length_m=1.)
