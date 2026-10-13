"""B-COR hardware executes through existing ray/wave paths, not a correction certificate."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.physics import column_wave
from temsim.physics.core import propagate
from temsim.physics.instrument_magnetic import active_column_events
from temsim.physics.wave_grid import WaveGridNumerics
from test_wave_detector_readout import checkpoint


def _state():
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", "C3 + B-COR", "No Energy Filter"))
    state.acceleration_enabled = True
    state.acceleration_backend = "CPU"
    state.step_mm = .05
    return state


def _local_wave(state):
    hp = next(p for p in state.corrector_elements if p.key == "image_hp3_hexapole")
    source = replace(checkpoint(), plane_z_mm=hp.z_mm-.001)
    target = hp.z_mm+.001
    prepared = column_wave._prepare_column(state, source.plane_z_mm, target, .001)
    return source, target, prepared


def test_third_hexapole_changes_executed_complex_wave_with_same_source():
    state = _state()
    source, target, prepared = _local_wave(state)
    result = column_wave._propagate_column(state, source, target, _prepared=prepared,
        grid_numerics=WaveGridNumerics(compute_backend="CPU"))
    hp = next(p for p in state.corrector_elements if p.key == "image_hp3_hexapole")
    hp.strength_m3 = 0.
    without = column_wave._propagate_column(state, source, target, maximum_step_mm=.001,
        grid_numerics=WaveGridNumerics(compute_backend="CPU"))
    a = result.beam.modes[0].plane.amplitude
    b = without.beam.modes[0].plane.amplitude
    assert np.linalg.norm(a-b) > 1e-8
    assert result.beam.total_weight == pytest.approx(source.beam.total_weight, rel=1e-10)
    assert result.record["upstream_digest"] == source.digest


def test_real_cuda_bcor_rays_and_third_hexapole_wave_match_cpu(record_property):
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No accessible CUDA runtime")
    if not count:
        pytest.skip("No CUDA device")
    record_property("gpu_device", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    from temsim.physics.backend_execution import backend_receipts
    state = _state()
    parts = {p.key: p for p in state.image_corrector_system.components}
    z0 = parts["image_hp1_hexapole"].z_mm-22.
    z1 = parts["image_sad_plane"].z_mm
    x = np.array([0., 1e-5, -1e-5, 0., 0.])
    y = np.array([0., 0., 0., 1e-5, -1e-5])
    t = np.zeros_like(x)
    values, receipts = [], []
    with backend_receipts(receipts):
        for backend in ("CPU", "Require GPU"):
            state.acceleration_backend = backend
            result = propagate(state, z0, z1, x, t, y, t,
                events=active_column_events(state), maximum_step_mm=.05,
                save_z_mm=(z1,), checkpoint_z_mm=(z1,), return_checkpoints=True)
            end = result[-1]
            values.append(np.array([end.x_m[-1], end.y_m[-1], end.tx_rad[-1], end.ty_rad[-1]]))
    assert any(r.get("actual") == "CUDA GPU" and r.get("outcome") == "returned" for r in receipts)
    np.testing.assert_allclose(values[1][:2], values[0][:2], atol=1e-11, rtol=1e-8)
    np.testing.assert_allclose(values[1][2:], values[0][2:], atol=1e-10, rtol=1e-8)
    record_property("ray_max_position_difference_m", float(np.max(np.abs(values[1][:2]-values[0][:2]))))
    record_property("ray_max_angle_difference_rad", float(np.max(np.abs(values[1][2:]-values[0][2:]))))
    source, target, prepared = _local_wave(state)
    results = [column_wave._propagate_column(state, source, target, _prepared=prepared,
        grid_numerics=WaveGridNumerics(compute_backend=backend)) for backend in ("CPU", "Require GPU")]
    assert all(r["compute_backend"] == "cupy" for r in results[1].record["modes"])
    a, b = [r.beam.modes[0].plane for r in results]
    np.testing.assert_allclose(b.amplitude, a.amplitude, atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(b.curvature_m1, a.curvature_m1, atol=1e-10, rtol=2e-10)
    np.testing.assert_allclose(b.basis_m, a.basis_m, atol=1e-18, rtol=2e-10)
    record_property("hp3_local_wave_relative_complex_error", float(np.linalg.norm(b.amplitude-a.amplitude)/np.linalg.norm(a.amplitude)))
