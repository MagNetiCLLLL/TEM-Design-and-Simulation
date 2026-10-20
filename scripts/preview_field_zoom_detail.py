"""Native zoom/refinement QA using one production magnetic-field provider.

The existing C3 + Probe Corrector assembly is unchanged. A temporary +15/-8%
condenser-stigmator diagnostic drive produces the B field. This is not a full
column calculation: no electron trajectory, gun E solver or FEM solve runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, sleep
from types import MappingProxyType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/field-lod-20261020"))
    parser.add_argument("--angle", type=float, default=0.,
                        help="Shared longitudinal Z–U section angle in degrees.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    implementation = ["scripts/preview_field_zoom_detail.py", "src/temsim/gui/field_vector_canvas.py",
                      "src/temsim/gui/field_vector_page.py", "src/temsim/field_vector_scene.py"]
    implementation_hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                             for name in implementation}
    sys.path.insert(0, str(root/"src"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "TEMSIM_CPU_THREADS"):
        os.environ[name] = "1"
    started = perf_counter()
    import numpy as np
    if not np.isfinite(args.angle):
        parser.error("--angle must be finite")
    angle = args.angle % 360.
    from PySide6.QtCore import QEvent, QPoint, QPointF, QSignalBlocker, QThreadPool, Qt
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QMouseEvent, QPalette, QWheelEvent
    from PySide6.QtWidgets import QApplication, QLabel, QListWidgetItem, QVBoxLayout, QWidget
    from temsim.assembly_model_3d import _contexts, _global_mesh
    from temsim.field_display_regions import build_field_display_regions
    from temsim.field_density_texture import section_grid_axes
    from temsim.field_vector_geometry import FieldVectorGeometry, _mesh_metres
    from temsim.field_vector_scene import FieldVectorScene, _source_groups
    from temsim.gui.field_vector_page import FieldVectorPage, _Samples
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.optics.column import default_state
    from temsim.part_model_3d import part_model_from_document

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = root/"configs/instruments/column/C3_ProbeCorrector.toml"
    source_bytes = source_path.read_bytes()
    state = default_state()
    key = "condenser_stigmator"
    component = next(item for item in state.stigmators if item.key == key)
    component.strength_x_percent, component.strength_y_percent = 15., -8.
    snapshot = json.dumps(state.to_dict(), sort_keys=True, indent=2, default=str)
    (output/"diagnostic_state.json").write_text(snapshot, encoding="utf-8")
    magnetic = prepare_magnetic_scene(state)
    source, = (item for item in _source_groups(state, magnetic) if item.key == key)
    regions = build_field_display_regions(state, (source,))
    part = next(item for item in state._resolved_assembly.parts if item.key == key)
    physical_bounds = np.asarray(regions.bounds_for_component(key))
    scene_bounds = np.stack((np.minimum(physical_bounds[0], source.bounds_m[0]),
                             np.maximum(physical_bounds[1], source.bounds_m[1])))
    note = "One production B provider with diagnostic drives; no complete column, gun E, FEM or trajectory calculation."
    real_scene = FieldVectorScene((source,), scene_bounds, (note,))

    class CountedScene:
        """Transparent instrumentation: every value comes from real_scene."""
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return getattr(real_scene, name)

        def sample(self, points, *arguments, **keywords):
            begin = perf_counter()
            result = real_scene.sample(points, *arguments, **keywords)
            self.calls.append({"points": len(points), "elapsed_seconds": perf_counter()-begin,
                               "valid_B": int(np.count_nonzero(result.magnetic_valid))})
            return result

    scene = CountedScene()
    radial = min(item.radius_m for item in source._magnetic)
    half = radial*.75/np.sqrt(2.)
    centre = np.mean(source.bounds_m, axis=0)
    z_low = max(physical_bounds[0, 2], source.bounds_m[0, 2])
    z_high = min(physical_bounds[1, 2], source.bounds_m[1, 2])
    centre[2] = (z_low+z_high)/2.
    bounds = np.array([[centre[0]-half, centre[1]-half, z_low],
                       [centre[0]+half, centre[1]+half, z_high]])
    theta = np.deg2rad(angle)
    cosine, sine = np.cos(theta), np.sin(theta)
    u_centre = centre[0]*cosine + centre[1]*sine
    uu, zz = np.meshgrid(np.linspace(u_centre-half, u_centre+half, 15),
                         np.linspace(z_low, z_high, 15), indexing="ij")
    points = np.column_stack((uu.ravel()*cosine, uu.ravel()*sine, zz.ravel()))
    grid_axes = section_grid_axes(points, "UZ", angle_deg=angle)
    points = points[regions.mask(points, key)]
    raw = scene.sample(points, (source.key,), kinetic_energy_ev=state.beam_voltage_kv*1000.)
    sample_names = ("points_m", "b_t", "e_v_per_m", "force_n",
                    "magnetic_valid", "electric_valid", "force_valid")
    sample = _Samples(**{name: getattr(raw, name) for name in sample_names}, notes=raw.notes,
                      grid_axes_m=grid_axes, section_angle_deg=angle)
    if not np.all(sample.magnetic_valid) or not np.any(sample.b_t):
        raise RuntimeError("The diagnostic requires a supported nonzero production B field")
    _parts, documents = _contexts(state._resolved_assembly)
    model = part_model_from_document(documents[part.module_key], key,
                                     angular_segments=24, include_children=False)
    shift = part.center_z_mm-float(part.data["local_center_z_mm"])
    meshes = tuple(_mesh_metres(_global_mesh(mesh, shift)) for mesh in model.meshes)
    mapping = MappingProxyType({key: (key,)})
    geometry = FieldVectorGeometry(meshes, mapping, tuple(model.notes), (),
                                   MappingProxyType({key: tuple(model.notes)}), meshes, mapping)

    app = QApplication.instance() or QApplication([])
    QThreadPool.globalInstance().setMaxThreadCount(1)
    for filename in ("segoeui.ttf", "segoeuib.ttf"):
        font = Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"/filename
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))
    app.setFont(QFont("Segoe UI", 9))
    palette = app.palette()
    for role, colour in ((QPalette.ColorRole.Window, "#101823"), (QPalette.ColorRole.Base, "#080e19"),
                         (QPalette.ColorRole.Text, "#e3edf8"), (QPalette.ColorRole.WindowText, "#e3edf8"),
                         (QPalette.ColorRole.Button, "#223043"), (QPalette.ColorRole.ButtonText, "#e3edf8")):
        palette.setColor(role, QColor(colour))
    app.setPalette(palette)
    window = QWidget()
    layout = QVBoxLayout(window)
    heading = QLabel("C3 + Probe Corrector | condenser stigmator | native zoom detail")
    heading.setStyleSheet("font-size:18px;font-weight:600;")
    layout.addWidget(heading)
    qualification = QLabel(
        f"Production B diagnostic: temporary X/Y +15 / -8%. Z–U section at {angle:g}° "
        "in the provider-supported near-axis region. No beam / gun E solve.")
    qualification.setWordWrap(True)
    layout.addWidget(qualification)
    page = FieldVectorPage()
    layout.addWidget(page, 1)
    page.auto_detail.setChecked(False)
    page._state, page._scene, page._display_regions = state, scene, regions
    page._geometry = geometry
    page._populate_component_picker()
    with QSignalBlocker(page.sources):
        item = QListWidgetItem(source.label)
        item.setData(Qt.ItemDataRole.UserRole, source.key)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        page.sources.addItem(item)
    page.section.setCurrentIndex(page.section.findData("UZ"))
    page.quantity.setCurrentIndex(page.quantity.findData("B"))
    page.field_scope.setCurrentIndex(page.field_scope.findData("component"))
    page.focus_component(key)
    for control, value in ((page.centre_x, centre[0]*1000), (page.centre_y, centre[1]*1000),
                           (page.centre_z, centre[2]*1000), (page.radius, half*1000),
                           (page.axial_half_span, (z_high-z_low)*500.)):
        with QSignalBlocker(control):
            control.setValue(float(value))
    page.set_projection_angle(angle)
    page._base_sample_data = (sample, bounds, "UZ")
    page._publish_sample(page._base_sample_data)
    page.canvas.reset_view()
    page._fit_initial = page._selection_needs_fit = False
    page.set_active(True)
    window.resize(1180, 760)
    window.show()
    app.processEvents()
    initial_references = page.canvas.references.copy()
    frames = []

    def capture(name):
        app.processEvents()
        if not window.grab().save(str(output/name)):
            raise RuntimeError("Native QWidget screenshot save failed")
        frame = dict(image=name, samples=len(page.canvas.sample.points_m),
                     visible_glyphs=len(page.canvas._visible_sample_indices),
                     provider_calls=len(scene.calls), zoom=page.canvas._zoom,
                     pan=[page.canvas._pan.x(), page.canvas._pan.y()],
                     canvas=[page.canvas.width(), page.canvas.height()],
                     window=[window.width(), window.height()], references=page.canvas.references.copy(),
                     all_samples_inside_physical_envelope=bool(
                         np.all(regions.mask(page.canvas.sample.points_m, key))),
                     detail_status=page.detail_status.text())
        frames.append(frame)
        return frame

    def wait_detail():
        begin = perf_counter()
        while (page._detail_timer.isActive() or page._detail_worker is not None
               or page._detail_dirty):
            if perf_counter()-begin > 20.:
                raise RuntimeError("Visible refinement did not finish: "+page.detail_status.text())
            app.processEvents()
            sleep(.005)
        app.processEvents()
        return perf_counter()-begin

    def drag(delta):
        start = page.canvas._plot_rect().center()
        end = start+delta
        for kind, position, button, buttons in (
                (QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton),
                (QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton),
                (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)):
            event = QMouseEvent(kind, position, QPointF(page.canvas.mapToGlobal(position.toPoint())),
                                button, buttons, Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(page.canvas, event)

    capture("01_overview_fixed_1180x760.png")
    cursor = page.canvas._plot_rect().center()+QPointF(70., -25.)
    event = QWheelEvent(cursor, QPointF(page.canvas.mapToGlobal(cursor.toPoint())), QPoint(), QPoint(0, 8*120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(page.canvas, event)
    fixed = capture("02_zoom_fixed_1180x760.png")
    refinement_begin = perf_counter()
    page.auto_detail.setChecked(True)
    first_wait = wait_detail()
    refined = capture("03_zoom_refined_1180x760.png")
    refinement_seconds = perf_counter()-refinement_begin
    assert fixed["zoom"] == refined["zoom"] and fixed["pan"] == refined["pan"]
    assert refined["visible_glyphs"] > fixed["visible_glyphs"]
    assert initial_references == page.canvas.references
    before_pan = len(scene.calls)
    drag(QPointF(48., 22.))
    wait_detail()
    after_pan = len(scene.calls)
    assert after_pan > before_pan
    drag(QPointF(-48., -22.))
    cache_wait = wait_detail()
    after_return = len(scene.calls)
    capture("04_zoom_cached_return_1180x760.png")
    assert after_return == after_pan and "cache" in page.detail_status.text().lower()
    window.resize(1000, 650)
    app.processEvents()
    wait_detail()
    capture("05_zoom_refined_1000x650.png")
    page.controls_toggle.setChecked(True)
    app.processEvents()
    wait_detail()
    capture("06_controls_1000x650.png")
    assert initial_references == page.canvas.references
    assert source_path.read_bytes() == source_bytes
    page.set_active(False)
    QThreadPool.globalInstance().waitForDone(5000)
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest() != value
           for name, value in implementation_hashes.items()):
        raise RuntimeError("Preview source changed during capture; rerun against stable source")
    manifest = dict(method="Unedited native FieldVectorPage QWidget.grab with actual asynchronous viewport workers",
                    note=note, configured_geometry="C3 + Probe Corrector",
                    section="UZ", section_angle_deg=angle,
                    source_key=source.key, source_model=source.model, source_notes=list(source.notes),
                    temporary_overrides={key: dict(strength_x_percent=15., strength_y_percent=-8.)},
                    physical_bounds_m=physical_bounds.tolist(), sampling_bounds_m=bounds.tolist(),
                    field_valid_radius_mm=radial*1000., initial_samples=len(points),
                    calls=scene.calls, frames=frames, references_unchanged=True,
                    cache_return_added_provider_calls=after_return-after_pan,
                    first_refinement_including_debounce_seconds=first_wait,
                    refinement_and_capture_seconds=refinement_seconds,
                    cached_return_including_debounce_seconds=cache_wait,
                    source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                    diagnostic_state_sha256=hashlib.sha256(snapshot.encode()).hexdigest(),
                    implementation_sha256=implementation_hashes,
                    saved_geometry_unchanged=True, particle_trace=False, gun_electrostatic_solve=False,
                    finite_element_solve=False, cpu_worker_threads=1, elapsed_seconds=perf_counter()-started)
    (output/"preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": len(frames),
                      "provider_calls": len(scene.calls), "cache_return_added_calls": after_return-after_pan,
                      "fixed_visible_glyphs": fixed["visible_glyphs"], "refined_visible_glyphs": refined["visible_glyphs"],
                      "elapsed_seconds": manifest["elapsed_seconds"]}))
    window.close()


if __name__ == "__main__":
    main()
