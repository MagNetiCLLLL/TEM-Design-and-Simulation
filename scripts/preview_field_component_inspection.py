"""Native CPU field-component previews from an explicit diagnostic snapshot.

The saved C3 + Probe Corrector geometry is unchanged. The condenser stigmator
gets temporary X/Y drives of 15/-8 percent, and its production magnetic field
is sampled once. No beam trajectory, gun electrostatic or FEM solve is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import MappingProxyType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/field-component-inspection-20261020"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"src"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "TEMSIM_CPU_THREADS"):
        os.environ[name] = "1"
    started = perf_counter()
    import numpy as np
    from PySide6.QtCore import QSignalBlocker, Qt
    from PySide6.QtGui import QFont, QFontDatabase, QPalette, QColor
    from PySide6.QtWidgets import QApplication, QLabel, QListWidgetItem, QVBoxLayout, QWidget
    from temsim.optics.column import default_state
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.field_vector_scene import FieldVectorScene, _source_groups
    from temsim.field_display_regions import build_field_display_regions
    from temsim.field_density_texture import section_grid_axes
    from temsim.field_vector_geometry import FieldVectorGeometry, _mesh_metres
    from temsim.assembly_model_3d import _contexts, _global_mesh
    from temsim.part_model_3d import part_model_from_document
    from temsim.gui.field_vector_page import FieldVectorPage, _Samples

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
    note = "Diagnostic B-only subset of a captured C3 + Probe Corrector state; no complete beam or gun E calculation."
    scene = FieldVectorScene((source,), scene_bounds, (note,))
    radial = min(item.radius_m for item in source._magnetic)
    half = radial * .75 / np.sqrt(2.)
    centre = np.mean(source.bounds_m, axis=0)
    z_low = max(physical_bounds[0, 2], source.bounds_m[0, 2])
    z_high = min(physical_bounds[1, 2], source.bounds_m[1, 2])
    grid = np.meshgrid(np.linspace(centre[0]-half, centre[0]+half, 9),
                       np.linspace(centre[1]-half, centre[1]+half, 9),
                       np.linspace(z_low, z_high, 9), indexing="ij")
    full_points = np.column_stack([axis.ravel() for axis in grid])
    points = full_points[regions.mask(full_points, key)]
    if not 0 < len(points) <= 729:
        raise RuntimeError("Unexpected bounded sampling count")
    sample = scene.sample(points, (source.key,), kinetic_energy_ev=state.beam_voltage_kv*1000.)
    if not np.all(sample.magnetic_valid) or not np.any(sample.b_t):
        raise RuntimeError("Preview requires a supported nonzero production magnetic field")
    bounds = np.array([[centre[0]-half, centre[1]-half, z_low],
                       [centre[0]+half, centre[1]+half, z_high]])
    # Reuse the same captured native model builder, but build just this body to
    # avoid meshing the entire column on low-memory machines.
    _parts, documents = _contexts(state._resolved_assembly)
    model = part_model_from_document(documents[part.module_key], key,
                                     angular_segments=24, include_children=False)
    shift = part.center_z_mm-float(part.data["local_center_z_mm"])
    meshes = tuple(_mesh_metres(_global_mesh(mesh, shift)) for mesh in model.meshes)
    mapping = MappingProxyType({key: (key,)})
    geometry = FieldVectorGeometry(meshes, mapping, tuple(model.notes), (),
                                   MappingProxyType({key: tuple(model.notes)}), meshes, mapping)

    app = QApplication.instance() or QApplication([])
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
    heading = QLabel("C3 + Probe Corrector | condenser stigmator | diagnostic snapshot")
    heading.setStyleSheet("font-size:18px;font-weight:600;")
    layout.addWidget(heading)
    qualification = QLabel("Production B only; temporary X/Y drives +15 / -8%. Native generic coil geometry. No beam/gun E solve.")
    qualification.setWordWrap(True)
    layout.addWidget(qualification)
    page = FieldVectorPage()
    layout.addWidget(page, 1)
    # Install already accepted, real production data into the native page. No
    # fake calculation result or background provider replacement is created.
    page._state, page._scene, page._display_regions = state, scene, regions
    page._geometry, page._sample = geometry, sample
    page._populate_component_picker()
    with QSignalBlocker(page.sources):
        item = QListWidgetItem(source.label)
        item.setData(Qt.ItemDataRole.UserRole, source.key)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        page.sources.addItem(item)
    page.section.setCurrentIndex(page.section.findData("3D"))
    page.density.setCurrentIndex(page.density.findData(9))
    page.quantity.setCurrentIndex(page.quantity.findData("B"))
    page.field_scope.setCurrentIndex(page.field_scope.findData("component"))
    page.focus_component(key)
    for control, value in ((page.centre_x, centre[0]*1000), (page.centre_y, centre[1]*1000),
                           (page.centre_z, (z_low+z_high)*500), (page.radius, half*1000),
                           (page.axial_half_span, (z_high-z_low)*500)):
        with QSignalBlocker(control):
            control.setValue(float(value))
    page._sample = sample
    page.canvas.set_sample(sample, bounds, "3D")
    page.canvas.set_mode("B")
    page._refresh_geometry()
    page._refresh_references()
    page._update_details()
    page.status.setText(f"Diagnostic snapshot only | {len(points)} production samples, all valid | same samples at every view angle")
    page.controls_toggle.setChecked(False)
    window.resize(1180, 760)
    window.show()
    app.processEvents()
    records = []
    sample_hash = hashlib.sha256(points.tobytes()+sample.b_t.tobytes()).hexdigest()
    for angle in (0., 45., 90.):
        page.set_projection_angle(angle)
        for framing in ("models", "field_core"):
            if framing == "models":
                page.canvas.fit_physical_models()
            else:
                page.canvas.reset_view()
            qualification.setText(
                f"Production B only; temporary X/Y +15 / -8%. Native coils; field valid within r ≤ {radial*1000:.4g} mm. "
                + ("Full model view." if framing == "models" else "Near-axis field view; projected coils may overlap."))
            app.processEvents()
            name = f"stigmator_{angle:03.0f}_{framing}.png"
            if not window.grab().save(str(output/name)):
                raise RuntimeError("Native screenshot save failed")
            records.append(dict(image=name, angle_deg=angle, section="3D", framing=framing,
                                sample_sha256=sample_hash, canvas_size=[page.canvas.width(), page.canvas.height()]))
    page.controls_toggle.setChecked(True)
    window.resize(1000, 650)
    app.processEvents()
    low_resolution = dict(window=[window.width(), window.height()],
                          canvas=[page.canvas.width(), page.canvas.height()],
                          controls_scroll=[page.control_scroll.width(), page.control_scroll.height()])
    window.grab().save(str(output/"controls_1000x650.png"))
    page.controls_toggle.setChecked(False)
    app.processEvents()
    low_resolution["collapsed_canvas"] = [page.canvas.width(), page.canvas.height()]
    window.grab().save(str(output/"collapsed_1000x650.png"))
    # The 0° longitudinal view is an exact subset of the same volume data. Do not
    # sample providers again or mislabel a rotated volume as a physical cut.
    xz_mask = np.isclose(points[:, 1], 0., rtol=0., atol=1e-15)
    full_xz_points = full_points[np.isclose(full_points[:, 1], 0., rtol=0., atol=1e-15)]
    if not len(full_xz_points):
        raise RuntimeError("The captured volume does not contain the V=0 longitudinal plane")
    names = ("points_m", "b_t", "e_v_per_m", "force_n", "magnetic_valid", "electric_valid", "force_valid")
    xz_sample = _Samples(**{name: getattr(sample, name)[xz_mask] for name in names},
                         notes=sample.notes, grid_axes_m=section_grid_axes(full_xz_points, "UZ", angle_deg=0.),
                         section_angle_deg=0.)
    with QSignalBlocker(page.section):
        page.section.setCurrentIndex(page.section.findData("UZ"))
    page.set_projection_angle(0.)
    page._base_sample_data = (xz_sample, bounds, "UZ")
    page._publish_sample(page._base_sample_data)
    page.status.setText(f"Diagnostic snapshot only | {int(xz_mask.sum())} cached 0° longitudinal samples extracted from the same volume; no extra sampling")
    for framing in ("models", "field_core"):
        if framing == "models":
            page.canvas.fit_physical_models()
        else:
            page.canvas.reset_view()
        qualification.setText(
            f"Production B only; temporary X/Y +15 / -8%. Z–U cut at 0°; valid field radius {radial*1000:.4g} mm. "
            + ("Full model section." if framing == "models" else "Near-axis field view; material cuts outside this window."))
        app.processEvents()
        name = f"xz_{framing}_1000x650.png"
        window.grab().save(str(output/name))
        records.append(dict(image=name, angle_deg=0., section="UZ", framing=framing,
                            sample_sha256=hashlib.sha256(xz_sample.points_m.tobytes()+xz_sample.b_t.tobytes()).hexdigest(),
                            canvas_size=[page.canvas.width(), page.canvas.height()]))
    if source_path.read_bytes() != source_bytes:
        raise RuntimeError("Source geometry changed while capturing; repeat against stable input")
    implementation = ["scripts/preview_field_component_inspection.py", "src/temsim/gui/field_vector_canvas.py",
                      "src/temsim/gui/field_vector_page.py", "src/temsim/field_vector_scene.py",
                      "src/temsim/field_display_regions.py", "src/temsim/field_vector_geometry.py"]
    manifest = dict(method="Native FieldVectorPage QWidget.grab; no image editing", note=note,
        configured_geometry="C3 + Probe Corrector", source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        snapshot_sha256=hashlib.sha256(snapshot.encode()).hexdigest(),
        temporary_overrides={key: dict(strength_x_percent=15., strength_y_percent=-8.)},
        source_key=source.key, source_model=source.model, source_bounds_m=np.asarray(source.bounds_m).tolist(),
        field_valid_radius_mm=radial*1000, sample_count=len(points), sample_evaluations=1,
        magnetic_peak_t=float(np.linalg.norm(sample.b_t, axis=1).max()), source_notes=list(source.notes),
        meshes=[dict(key=mesh.key, region=mesh.region, vertices=len(mesh.vertices), triangles=len(mesh.faces)) for mesh in meshes],
        selected_alpha_cap=.35, selected_alpha_composition="One viewport layer for the complete selected assembly",
        derived_xz_sample_count=int(xz_mask.sum()), frames=records, low_resolution=low_resolution,
        saved_geometry_unchanged=True, beam_trace_performed=False, gun_electrostatic_solve_performed=False,
        elapsed_seconds=perf_counter()-started,
        implementation_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in implementation})
    (output/"preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(points), "frames": len(records),
                      "low_resolution": low_resolution, "elapsed_seconds": manifest["elapsed_seconds"]}))
    window.close()


if __name__ == "__main__":
    main()
