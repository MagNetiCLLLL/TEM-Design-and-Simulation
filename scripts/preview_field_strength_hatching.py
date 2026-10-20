"""Capture native arrow / strength-hatching comparisons on one CPU.

The production condenser-stigmator B provider is sampled once per plane in the
existing C3 + Probe Corrector assembly, using temporary +15/-8% diagnostic drives.
No particle trace, gun electrostatic solve, finite-element solve or CUDA is used.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import MappingProxyType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/field-longitudinal-20261020"))
    parser.add_argument("--angles", default="0,45,90",
                        help="Comma-separated shared longitudinal-section angles in degrees.")
    parser.add_argument("--zoom", action="store_true",
                        help="Also capture a fixed-sample zoom comparison at each angle.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "TEMSIM_CPU_THREADS"):
        os.environ[name] = "1"
    started = perf_counter()

    import numpy as np
    try:
        angles = tuple(dict.fromkeys(float(value.strip()) % 360. for value in args.angles.split(",")))
    except ValueError:
        parser.error("--angles must be a comma-separated list of numbers")
    if not angles or not np.all(np.isfinite(angles)):
        parser.error("--angles must contain finite values")
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QSignalBlocker, QThreadPool, Qt
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette, QWheelEvent
    from PySide6.QtWidgets import QApplication, QLabel, QListWidgetItem, QVBoxLayout, QWidget
    from temsim.assembly_model_3d import _contexts, _global_mesh
    from temsim.cpu_resources import numerical_job
    from temsim.field_density_texture import section_grid_axes
    from temsim.field_display_regions import build_field_display_regions
    from temsim.field_vector_geometry import FieldVectorGeometry, _mesh_metres
    from temsim.field_vector_scene import FieldVectorScene, _source_groups
    from temsim.gui.field_vector_page import FieldVectorPage, _Samples
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.optics.column import default_state
    from temsim.part_model_3d import part_model_from_document

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    implementation = (
        "scripts/preview_field_strength_hatching.py", "src/temsim/gui/field_vector_canvas.py",
        "src/temsim/gui/field_vector_page.py", "src/temsim/field_density_texture.py",
        "src/temsim/field_vector_scene.py", "src/temsim/field_display_regions.py",
    )
    implementation_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in implementation
    }
    source_path = root / "configs/instruments/column/C3_ProbeCorrector.toml"
    source_bytes = source_path.read_bytes()
    state = default_state()
    key = "condenser_stigmator"
    component = next(item for item in state.stigmators if item.key == key)
    component.strength_x_percent, component.strength_y_percent = 15., -8.
    snapshot = json.dumps(state.to_dict(), sort_keys=True, indent=2, default=str)
    (output / "diagnostic_state.json").write_text(snapshot, encoding="utf-8")
    with numerical_job(1):
        magnetic = prepare_magnetic_scene(state)
        source, = (item for item in _source_groups(state, magnetic) if item.key == key)
        regions = build_field_display_regions(state, (source,))
    part = next(item for item in state._resolved_assembly.parts if item.key == key)
    physical_bounds = np.asarray(regions.bounds_for_component(key))
    scene_bounds = np.stack((np.minimum(physical_bounds[0], source.bounds_m[0]),
                             np.maximum(physical_bounds[1], source.bounds_m[1])))
    note = ("One production B provider with temporary diagnostic drives; no complete column, "
            "gun electrostatic, finite-element or particle calculation.")
    real_scene = FieldVectorScene((source,), scene_bounds, (note,))

    class CountedScene:
        """Instrumentation only: field values always come from the real provider."""

        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return getattr(real_scene, name)

        def sample(self, points, *arguments, **keywords):
            begin = perf_counter()
            with numerical_job(1):
                result = real_scene.sample(points, *arguments, **keywords)
            self.calls.append({
                "points": len(points), "elapsed_ms": (perf_counter() - begin) * 1000.,
                "valid_B": int(np.count_nonzero(result.magnetic_valid)),
            })
            return result

    scene = CountedScene()
    radial = min(item.radius_m for item in source._magnetic)
    half = radial * .75 / np.sqrt(2.)
    centre = np.mean(source.bounds_m, axis=0)
    z_low = max(physical_bounds[0, 2], source.bounds_m[0, 2])
    z_high = min(physical_bounds[1, 2], source.bounds_m[1, 2])
    centre[2] = (z_low + z_high) / 2.
    bounds = np.array([[centre[0] - half, centre[1] - half, z_low],
                       [centre[0] + half, centre[1] + half, z_high]])
    # Build only the selected body, not the entire column's physical geometry.
    with numerical_job(1):
        _parts, documents = _contexts(state._resolved_assembly)
        model = part_model_from_document(documents[part.module_key], key,
                                         angular_segments=24, include_children=False)
        shift = part.center_z_mm - float(part.data["local_center_z_mm"])
        meshes = tuple(_mesh_metres(_global_mesh(mesh, shift)) for mesh in model.meshes)
    mapping = MappingProxyType({key: (key,)})
    geometry = FieldVectorGeometry(meshes, mapping, tuple(model.notes), (),
                                   MappingProxyType({key: tuple(model.notes)}), meshes, mapping)

    app = QApplication.instance() or QApplication([])
    QThreadPool.globalInstance().setMaxThreadCount(1)
    for filename in ("segoeui.ttf", "segoeuib.ttf"):
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename
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
    heading = QLabel("C3 + Probe Corrector | condenser stigmator | field-strength hatching")
    heading.setStyleSheet("font-size:18px;font-weight:600;")
    layout.addWidget(heading)
    qualification = QLabel()
    qualification.setWordWrap(True)
    layout.addWidget(qualification)
    page = FieldVectorPage()
    layout.addWidget(page, 1)
    page.auto_detail.setChecked(False)
    # Accepted production samples are installed explicitly. Keeping the page
    # inactive prevents any background capture or refinement during this QA.
    page._state, page._scene, page._display_regions = state, scene, regions
    page._geometry = geometry
    page._populate_component_picker()
    with QSignalBlocker(page.sources):
        item = QListWidgetItem(source.label)
        item.setData(Qt.ItemDataRole.UserRole, source.key)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        page.sources.addItem(item)
    page.quantity.setCurrentIndex(page.quantity.findData("B"))
    page.field_scope.setCurrentIndex(page.field_scope.findData("component"))
    page.focus_component(key)
    for control, value in ((page.centre_x, centre[0] * 1000), (page.centre_y, centre[1] * 1000),
                           (page.centre_z, centre[2] * 1000), (page.radius, half * 1000),
                           (page.axial_half_span, (z_high - z_low) * 500.)):
        with QSignalBlocker(control):
            control.setValue(float(value))
    page._fit_initial = page._selection_needs_fit = False
    window.resize(1100, 760)
    frames = []
    plane_records = []

    def capture(angle, framing, style):
        """Time synchronous native paints separately from PNG compression."""
        page.render_style.setCurrentIndex(page.render_style.findData(style))
        qualification.setText(
            f"Production B only | temporary X/Y +15 / -8% | Z–U longitudinal section at {angle:g}° | "
            "same cached samples and fixed B reference for arrows / hatching; CPU only.")
        app.processEvents()
        before_calls = len(scene.calls)
        # Force a cold texture paint after layout has reached its final size.
        if style == "density":
            page.canvas._density_image_cache = None
        begin = perf_counter()
        first_pixmap = page.canvas.grab()
        first_ms = (perf_counter() - begin) * 1000.
        del first_pixmap
        first_cache = page.canvas._density_image_cache if style == "density" else None
        begin = perf_counter()
        cached_pixmap = page.canvas.grab()
        cached_ms = (perf_counter() - begin) * 1000.
        del cached_pixmap
        cache_reused = (style == "density" and first_cache is not None
                        and page.canvas._density_image_cache is first_cache)
        angle_label = f"{angle:06.2f}".replace(".", "p")
        name = f"uz_angle_{angle_label}_{framing}_{style}_1100x760.png"
        if not window.grab().save(str(output / name)):
            raise RuntimeError("Native QWidget screenshot save failed")
        plot = page.canvas._plot_rect()
        corners = np.array([[plot.left(), plot.top()], [plot.right(), plot.top()],
                            [plot.right(), plot.bottom()], [plot.left(), plot.bottom()]])
        inverse = page.canvas._section_inverse(plot)
        viewport_xyz = page.canvas._section_positions(corners, inverse)
        texture = page.canvas._density_image_cache[1] if style == "density" else None
        record = dict(
            image=name, section="UZ", section_angle_deg=angle, framing=framing, render_style=style,
            samples=len(page._sample.points_m), first_redraw_ms=first_ms,
            cached_redraw_ms=cached_ms, texture_cache_reused=cache_reused,
            raster_bytes=int(texture.sizeInBytes()) if texture is not None else 0,
            raster_size=[texture.width(), texture.height()] if texture is not None else None,
            provider_calls=len(scene.calls), redraw_added_provider_calls=len(scene.calls) - before_calls,
            zoom=page.canvas._zoom.tolist(), pan=[page.canvas._pan.x(), page.canvas._pan.y()],
            canvas_size=[page.canvas.width(), page.canvas.height()],
            window_size=[window.width(), window.height()],
            plot_rect_pixels=[plot.x(), plot.y(), plot.width(), plot.height()],
            viewport_corners_xyz_m=viewport_xyz.tolist(),
            viewport_bounds_xyz_m=[np.min(viewport_xyz, axis=0).tolist(), np.max(viewport_xyz, axis=0).tolist()],
            sample_bounds_xyz_m=[np.min(page._sample.points_m, axis=0).tolist(),
                                 np.max(page._sample.points_m, axis=0).tolist()],
            references=page.canvas.references.copy(), legend=page.legend.text(),
            density_notice=page.canvas._density_notice if style == "density" else None,
            all_samples_inside_physical_envelope=bool(np.all(regions.mask(page._sample.points_m, key))),
            grid_axes_lengths=[len(axis) for axis in page._sample.grid_axes_m],
            sample_sha256=hashlib.sha256(page._sample.points_m.tobytes() + page._sample.b_t.tobytes()).hexdigest(),
        )
        if len(scene.calls) != before_calls:
            raise RuntimeError("A redraw unexpectedly queried the production field provider")
        if style == "density" and (not cache_reused or record["raster_bytes"] > 2 * 1024 * 1024):
            raise RuntimeError("Density texture cache or 2 MiB retained-raster budget failed")
        frames.append(record)
        return record

    try:
        for angle in angles:
            page.section.setCurrentIndex(page.section.findData("UZ"))
            # Rotation invalidates any old plane. Set it before publishing the
            # new production samples, not after, even while the page is inactive.
            page.set_projection_angle(angle)
            # Save the complete regular grid before clipping its physical region.
            theta = np.deg2rad(angle)
            cosine, sine = np.cos(theta), np.sin(theta)
            u_centre = centre[0] * cosine + centre[1] * sine
            uu, zz = np.meshgrid(np.linspace(u_centre-half, u_centre+half, 15),
                                 np.linspace(z_low, z_high, 15), indexing="ij")
            full_points = np.column_stack((uu.ravel()*cosine, uu.ravel()*sine, zz.ravel()))
            grid_axes = section_grid_axes(full_points, "UZ", angle_deg=angle)
            points = full_points[regions.mask(full_points, key)]
            raw = scene.sample(points, (source.key,), kinetic_energy_ev=state.beam_voltage_kv * 1000.)
            if not np.all(raw.magnetic_valid) or not np.any(raw.b_t):
                raise RuntimeError("Diagnostic requires a supported nonzero production B field")
            names = ("points_m", "b_t", "e_v_per_m", "force_n",
                     "magnetic_valid", "electric_valid", "force_valid")
            sample = _Samples(**{name: getattr(raw, name) for name in names},
                              notes=raw.notes, grid_axes_m=grid_axes, section_angle_deg=angle)
            page._base_sample_data = (sample, bounds, "UZ")
            page._publish_sample(page._base_sample_data)
            page.canvas.reset_view()
            page.controls_toggle.setChecked(False)
            window.show()
            app.processEvents()
            reference = page.canvas.references.copy()
            calls_before_switches = len(scene.calls)
            capture(angle, "overview", "arrows")
            capture(angle, "overview", "density")
            if args.zoom:
                cursor = page.canvas._plot_rect().center()
                event = QWheelEvent(
                    cursor, QPointF(page.canvas.mapToGlobal(cursor.toPoint())),
                    QPoint(), QPoint(0, 4 * 120), Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
                QApplication.sendEvent(page.canvas, event)
                capture(angle, "zoom_fixed_samples", "arrows")
                capture(angle, "zoom_fixed_samples", "density")
            if reference != page.canvas.references or len(scene.calls) != calls_before_switches:
                raise RuntimeError("Style switches or fixed-sample navigation changed field data or references")
            plane_records.append(dict(
                section="UZ", section_angle_deg=angle, initial_samples=len(points),
                provider_calls_after_sample=calls_before_switches,
                switches_added_provider_calls=len(scene.calls) - calls_before_switches,
                references_unchanged=True,
            ))
        if source_path.read_bytes() != source_bytes:
            raise RuntimeError("Source geometry changed during capture; rerun against stable source")
        for name, digest in implementation_hashes.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Preview implementation changed during capture; rerun against stable source")
        manifest = dict(
            method="Unedited native FieldVectorPage QWidget.grab; fixed production samples",
            note=note, configured_geometry="C3 + Probe Corrector", source_key=source.key,
            source_model=source.model, source_notes=list(source.notes),
            temporary_overrides={key: dict(strength_x_percent=15., strength_y_percent=-8.)},
            physical_bounds_m=physical_bounds.tolist(), sampling_bounds_m=bounds.tolist(),
            field_valid_radius_mm=radial * 1000., calls=scene.calls, planes=plane_records, frames=frames,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            diagnostic_state_sha256=hashlib.sha256(snapshot.encode()).hexdigest(),
            implementation_sha256=implementation_hashes, saved_geometry_unchanged=True,
            particle_trace=False, gun_electrostatic_solve=False, finite_element_solve=False,
            cuda_enabled=False, cpu_worker_threads=1, auto_detail=False,
            elapsed_seconds=perf_counter() - started,
        )
        (output / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({
            "output": str(output), "frames": len(frames), "provider_calls": len(scene.calls),
            "switches_added_provider_calls": sum(p["switches_added_provider_calls"] for p in plane_records),
            "max_raster_bytes": max(frame["raster_bytes"] for frame in frames),
            "elapsed_seconds": manifest["elapsed_seconds"],
        }))
    finally:
        page.set_active(False)
        QThreadPool.globalInstance().waitForDone(5000)
        app.processEvents()
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        # Drop wrappers after all children have received DeferredDelete.
        del item, qualification, heading, layout, page, window
        gc.collect()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


if __name__ == "__main__":
    main()
