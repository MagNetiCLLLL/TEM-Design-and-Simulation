"""Residual diagnostics use production fields; synthetic inputs are not OEM data."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.system_residual_aberrations import measure_system_residual


class _Lens:
    key, name = "test_round_lens", "Analytic test lens"
    z_mm, a_mm, percent, max_percent, polarity = 0., 1., 100., 100., 1
    gaussian, enabled, b0_t, cc_mm = (), True, 0., 0.

    def __init__(self, cs=1.):
        self.cs_mm = cs

    def magnetic_field_t(self, z):
        return np.zeros_like(np.asarray(z, float))

    def focal_length_for_voltage_mm(self, _energy):
        return 10.


def _state():
    return SimpleNamespace(lenses=[_Lens()], stigmators=[], corrector_elements=[], deflectors=[],
        apertures=[], recording_planes=[], sample=SimpleNamespace(z_mm=-1.),
        objective_image_plane_z_mm=1., beam_voltage_kv=300., step_mm=.1,
        history_step_mm=.1, acceleration_enabled=False, acceleration_backend="CPU",
        equivalent_image_lenses_enabled=False, lens_intrinsic_aberrations={})


def test_production_cs_residual_without_fieldmap_and_no_optical_mutation():
    state = _state()
    before = deepcopy(vars(state.lenses[0]))
    report = measure_system_residual(state, "image", semiangle_mrad=20, maximum_step_mm=.1)
    # r_lens = L1*alpha; delta r_screen = -L2*(Cs/f^4)*(L1*alpha)^3.
    expected = state.lenses[0].cs_mm*(.001**3*.001/.010**4)
    assert report.coefficients.c3_mm == pytest.approx(expected, rel=1e-7)
    assert abs(report.coefficients.c1_mm) < 1e-10
    assert report.diagnostics["holdout_rms_m"] < 1e-14
    assert report.diagnostics["holdout_independent"]
    assert report.diagnostics["unmeasured_coefficients"] == ("Cc",)
    assert not report.diagnostics["coefficients_applied"]
    assert report.diagnostics["convergence_status"] == "NOT_ASSESSED"
    assert vars(state.lenses[0]) == before
    assert not hasattr(state, "_tuning_cancelled")
    assert state.step_mm == .1 and state.lens_intrinsic_aberrations == {}


def test_real_intrinsic_a1_and_c1_remain_in_residual_not_absorbed_by_reference():
    state = _state()
    state.lenses[0].cs_mm = 0.
    state.lens_intrinsic_aberrations = {state.lenses[0].key: {
        "a1_mm": .1, "a1_azimuth_deg": 17., "c1_mm": .2}}
    before = deepcopy(state.lens_intrinsic_aberrations)
    report = measure_system_residual(state, "image", semiangle_mrad=10., maximum_step_mm=.1)
    # A local quadratic screen is -[C1*r^2 + A1*(x'^2-y'^2)]/(2*f^2).
    # With L1=L2=1 mm dr=-L1*L2/f^2 * grad_alpha W.
    factor = .001*.001/.01**2
    assert report.coefficients.c1_mm == pytest.approx(.2*factor, rel=2e-8)
    assert report.coefficients.a1_mm == pytest.approx(.1*factor, rel=2e-8)
    assert report.coefficients.a1_azimuth_deg == pytest.approx(17., abs=1e-7)
    assert report.diagnostics["holdout_rms_m"] < 1e-13
    assert state.lens_intrinsic_aberrations == before
    assert state.lenses[0].percent == 100.


@pytest.mark.parametrize("design", ["cetcor", "bcor"])
def test_image_residual_includes_lens_after_objective_image(design):
    state = _state()
    state.image_corrector_installed = True
    state.image_corrector_design = design
    state.objective_image_plane_z_mm = -.5  # Upstream of the aberrating lens.
    state.corrector_elements = [SimpleNamespace(
        key="image_sad_plane", z_mm=1., enabled=True, _layout_installed=True,
        kind="reference_plane", optical_active=False)]
    report = measure_system_residual(state, "image", semiangle_mrad=20., maximum_step_mm=.1)
    expected = state.lenses[0].cs_mm*(.001**3*.001/.010**4)
    assert report.coefficients.c3_mm == pytest.approx(expected, rel=1e-7)
    assert report.diagnostics["reference_z_mm"] == 1.
    assert report.diagnostics["coefficient_reference_z_mm"] == -1.
    assert "SAD" in report.coefficients.reference_plane
    assert "off-axial" in report.diagnostics["scope"]


def test_missing_or_upstream_corrector_output_cannot_fall_back_to_objective_image():
    from temsim.optics.system_residual_aberrations import image_residual_reference
    state = _state()
    state.image_corrector_installed = True
    with pytest.raises(ValueError, match="one active SAD"):
        image_residual_reference(state)
    state.corrector_elements = [SimpleNamespace(key="image_sad_plane", z_mm=1., enabled=True),
        SimpleNamespace(key="image_hp3_hexapole", z_mm=2., enabled=True)]
    with pytest.raises(ValueError, match="downstream"):
        image_residual_reference(state)


def _mock_propagation(monkeypatch, *, singular=False, exact=True):
    from temsim.physics import core, first_order
    matrix = np.eye(4)
    if singular:
        matrix[0] = 0
    monkeypatch.setattr(first_order, "trace_transverse_transfer", lambda *a, **k:
        SimpleNamespace(matrix=matrix, j_img=matrix[:2, :2]))
    def plan(*args, **kwargs):
        assert kwargs["include_spherical_aberration"] and kwargs["include_hexapole"]
        assert kwargs["checkpoint_z_mm"] == (1.,)
        return SimpleNamespace(step_m=np.array([.001]), z_mm=np.array([-1., 1.]))
    monkeypatch.setattr(core, "build_propagation_plan", plan)
    def execute(state, plan, x, tx, y, ty):
        a = np.column_stack((tx, ty))
        radius2 = np.sum(a*a, axis=1)
        # Independent analytical residual: nonzero C1, A1 and C3.
        displacement = -1e-3*(a*(.004+1.3*radius2[:, None]) + a*np.array([.002, -.002]))
        positions = np.column_stack((x, y))+displacement+[7e-6, -4e-6]
        cp = SimpleNamespace(z_mm=np.array([1. if exact else .99]),
                             x_m=positions[:, 0][None], y_m=positions[:, 1][None])
        # Plot arrays are deliberately unusable: diagnostics must use checkpoints.
        return None, None, None, None, None, cp
    monkeypatch.setattr(core, "execute_propagation_plan", execute)


def test_probe_fit_keeps_c1_uses_exact_checkpoint_and_removes_only_affine_offset(monkeypatch):
    state = _state()
    state.sample.z_mm = 1.
    state.electron_gun = SimpleNamespace(exit_plane_z_mm=-1.)
    _mock_propagation(monkeypatch)
    report = measure_system_residual(state, "probe", semiangle_mrad=20.)
    assert report.coefficients.c1_mm == pytest.approx(.004, abs=1e-12)
    assert report.coefficients.a1_mm == pytest.approx(.002, abs=1e-12)
    assert report.coefficients.c3_mm == pytest.approx(1.3, abs=1e-7)
    np.testing.assert_allclose(report.diagnostics["reference_centroid_xy_m"], [7e-6, -4e-6])
    assert report.diagnostics["reference_z_mm"] == 1.
    assert report.diagnostics["coefficient_reference_z_mm"] == 1.
    assert report.diagnostics["training_ray_count"] == 192
    assert report.diagnostics["holdout_ray_count"] == 161


@pytest.mark.parametrize("case, message", [("singular", "singular"), ("checkpoint", "exact target")])
def test_invalid_reference_fails_instead_of_reporting_coefficients(monkeypatch, case, message):
    _mock_propagation(monkeypatch, singular=case == "singular", exact=case != "checkpoint")
    with pytest.raises(ValueError, match=message):
        measure_system_residual(_state(), "image")


@pytest.mark.parametrize("options", [{"semiangle_mrad": 0}, {"maximum_step_mm": float("nan")},
                                    {"semiangle_mrad": 101}])
def test_invalid_budget_rejected(options):
    with pytest.raises(ValueError):
        measure_system_residual(_state(), "image", **options)


def test_cancel_before_snapshot_or_trace():
    with pytest.raises(RuntimeError, match="cancelled"):
        measure_system_residual(_state(), "image", cancelled=lambda: True)


def test_gui_never_measures_on_open_and_measure_button_is_explicit(qtbot, monkeypatch):
    import temsim.gui.aberration_view as view_module
    calls = []
    monkeypatch.setattr(view_module, "effective_aberration_comparison", lambda *a: calls.append(a))
    view = view_module.AberrationComparisonView(fixed_system="image")
    qtbot.addWidget(view)
    view.display_result(SimpleNamespace(state_snapshot=_state()))
    # Avoid requiring a complete instrument for legacy cache identity in this GUI fixture.
    monkeypatch.setattr(view_module, "_aberration_cache_signature", lambda *a: "current")
    view.show()
    view._refresh()
    assert calls == []
    assert view.residual_button.isEnabled()
    assert view.system_residual_table.rowCount() == 0
    assert "System residual" in [view.pages.tabText(i) for i in range(view.pages.count())]
    # Execute the worker synchronously through a recording test pool; production
    # uses its existing background/resource coordinator.
    starts = []
    monkeypatch.setattr(view.residual_pool, "start", lambda worker: starts.append(worker))
    view._start_residual()
    assert len(starts) == 1 and view._residual_worker is starts[0]
    view._cancel_residual()
    assert starts[0].event.is_set()
    view._residual_finished(starts[0].generation)
    assert view.residual_button.isEnabled()
    view.shutdown()


def test_background_gui_measurement_renders_same_plane_and_invalidates_old_data(qtbot):
    from temsim.gui.aberration_view import AberrationComparisonView
    view = AberrationComparisonView(fixed_system="image")
    qtbot.addWidget(view)
    result = SimpleNamespace(state_snapshot=_state())
    view.display_result(result)
    view.residual_button.click()
    qtbot.waitUntil(lambda: view._residual_worker is None, timeout=15000)
    assert view._residual_report is not None, view.residual_summary.text()
    assert view.system_residual_table.rowCount() == 10
    assert "exact Z 1 mm" in view.residual_summary.text()
    assert "Cc and step convergence not measured" in view.residual_summary.text()
    assert view.system_residual_table.item(9, 2).text() == "Not measured"
    view.display_result(result)
    assert view._residual_report is None
    assert view.system_residual_table.rowCount() == 0
    assert view.shutdown()
