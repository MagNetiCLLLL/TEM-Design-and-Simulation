"""Capture actual Qt PartModelView stage/holder models and saved TOML baseline.

No meshes are invented, scaled, recoloured or edited for these captures. Camera
framing and material visibility alone change. The historical document is loaded
from its archived path, without the current sample-stage subassembly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("outputs/stage_holder_20261011/preview"))
    parser.add_argument("--original", type=Path, default=Path(
        "outputs/stage_holder_20261011/originals/configs/instruments/column/C3_ProbeCorrector.toml"))
    parser.add_argument("--angular-segments", type=int, default=96)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    original = (root / args.original).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import (
        QApplication, QGridLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget,
    )
    from temsim.gui.part_model_view import PartModelView
    from temsim.part_model_3d import part_model_from_document
    from temsim.part_model_document import PartModelDocument
    from temsim.shared_tip import dependencies

    app = QApplication.instance() or QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
    if font_path.exists():
        ident = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(ident)
        if families:
            app.setFont(QFont(families[0], 10))
    current = root / "configs/instruments/column/C3_ProbeCorrector.toml"
    sources = {"original": original, "current": current}
    documents = {key: PartModelDocument(path).document for key, path in sources.items()}
    rows = {key: {part["key"]: part for part in doc["parts"]}
            for key, doc in documents.items()}
    if "sample_holder" in rows["original"] or documents["original"].get("subassemblies"):
        raise RuntimeError("The comparison source must be the archived pre-holder document")
    if "housing_side_port_diameter_mm" in rows["original"]["sample_stage"]:
        raise RuntimeError("Historical stage already has a side port; choose the actual old input")
    if "sample_holder" not in rows["current"]:
        raise RuntimeError("Current subassembly did not resolve")
    z = float(rows["current"]["sample"]["local_center_z_mm"])
    if float(rows["original"]["sample"]["local_center_z_mm"]) != z:
        raise RuntimeError("Comparison sample coordinates differ")
    holder, stage = rows["current"]["sample_holder"], rows["current"]["sample_stage"]
    implementation_paths = ["scripts/preview_stage_holder.py", "src/temsim/gui/part_model_view.py",
        "src/temsim/part_model_3d.py", "src/temsim/cad_catalog.py", "src/temsim/cad_mesh.py",
        "src/temsim/cad_geometry.py", "src/temsim/cad_round_lenses.py", "src/temsim/cad_stage_holder.py"]
    all_files = {str(path.relative_to(root)): path.read_bytes() for path in sources.values()}
    for path in sources.values():
        all_files.update({str(p.relative_to(root)): blob for p, blob in dependencies(path).items()})
    all_files.update({p: (root / p).read_bytes() for p in implementation_paths})
    hashes = {path: hashlib.sha256(blob).hexdigest() for path, blob in all_files.items()}

    def meshes(version, keys):
        result = []
        for key in keys:
            result.extend(part_model_from_document(documents[version], key,
                angular_segments=args.angular_segments, include_children=False).meshes)
        return tuple(result)

    context_keys = ("objective_upper_pole", "objective_lower_pole", "objective_lens_housing",
                    "objective_lens_yoke", "objective_lens_excitation_coil")
    original_meshes = meshes("original", context_keys)
    current_meshes = meshes("current", (*context_keys, "sample_stage", "sample_holder"))
    holder_meshes = meshes("current", ("sample_holder",))
    stage_meshes = meshes("current", ("sample_stage",))
    near_meshes = meshes("current", ("objective_upper_pole", "objective_lower_pole", "sample_holder"))
    tip_meshes = tuple(mesh for mesh in holder_meshes if mesh.region == "tip")
    if not all((original_meshes, current_meshes, holder_meshes, stage_meshes, tip_meshes)):
        raise RuntimeError("A preview contains no native geometry")

    records = []
    windows = []
    style = "QWidget{background:#101823;color:#e5edf7;} QLabel{background:transparent;}"

    def board(title, subtitle, size):
        window = QWidget()
        window.setStyleSheet(style)
        layout = QVBoxLayout(window)
        heading = QLabel(title)
        heading.setStyleSheet("font-size:22px;font-weight:600;")
        layout.addWidget(heading)
        caption = QLabel(subtitle)
        caption.setWordWrap(True)
        caption.setStyleSheet("font-size:13px;color:#b7cadc;")
        layout.addWidget(caption)
        grid = QGridLayout()
        layout.addLayout(grid, 1)
        window.resize(*size)
        windows.append(window)
        return window, grid

    def panel(slug, title, selected, version, *, section=False, camera="column", bounds=None, zoom=1., note=""):
        container = QWidget()
        layout = QVBoxLayout(container)
        label = QLabel(title)
        label.setStyleSheet("font-size:18px;font-weight:600;")
        layout.addWidget(label)
        view = PartModelView()
        view.setMinimumSize(680, 350)
        view.set_view_labels(title=title)
        view.set_meshes(selected)
        # Column view looks from +Y: retain the far half to expose the cut.
        view.set_section_enabled(section, keep_positive_y=False)
        getattr(view, {"column": "set_column_view", "isometric": "set_column_isometric_view",
                       "axial": "set_axial_view"}[camera])()
        if bounds is not None:
            view._fit_points([np.asarray(bounds, dtype=float)])
        else:
            view.fit_all()
        view._zoom *= zoom
        view._invalidate()
        layout.addWidget(view, 1)
        caption = QLabel(note)
        caption.setWordWrap(True)
        caption.setStyleSheet("font-size:13px;color:#b7cadc;")
        layout.addWidget(caption)
        records.append({"id": slug, "source_version": version, "title": title,
            "section_enabled": section, "section_retains": "y <= 0" if section else None,
            "camera_rotation": view.camera_rotation.tolist(), "framing_bounds_mm": bounds,
            "zoom_factor": view.zoom_factor,
            "scope": note, "material_counts": dict(Counter(m.material_class for m in selected)),
            "meshes": [{"key": m.key, "region": m.region, "material_class": m.material_class,
                "vertices": len(m.vertices), "triangles": len(m.faces), "is_exact": m.is_exact,
                "bounds_mm": [m.vertices.min(axis=0).tolist(), m.vertices.max(axis=0).tolist()],
                "geometry_sha256": hashlib.sha256(m.vertices.tobytes()+m.faces.tobytes()).hexdigest()}
                for m in selected]})
        return container, view

    bounds = [[-95., -40., z-215.], [320., 40., z+215.]]
    comparison, grid = board("Side-entry holder: original TOML and current assembly",
        "Actual Qt PartModelView | identical camera framing | original file is loaded independently, without the new subassembly",
        (1680, 1120))
    for i, (version, selected, title, note) in enumerate([
        ("original", original_meshes, "Original: no holder, closed column housing",
         "Historical 650 x 500 mm stage allocation had no defined holder body. Its undefined envelope is omitted, not replaced with the new stage."),
        ("current", current_meshes, "Current: lateral port, stage and holder",
         "7 mm shaft; separate stage guide, flange and body. All non-reference dimensions are engineering reconstruction, not OEM CAD."),
    ]):
        widget, view = panel(version+"_external", title, selected, version,
                             bounds=bounds, zoom=1.2, note=note)
        grid.addWidget(widget, 0, i)
    old = rows["original"]["sample_stage"]
    text_old = (
        f"ARCHIVED C3_ProbeCorrector.toml\nSHA-256 {hashes[str(original.relative_to(root))][:20]}...\n\n"
        f"sample_stage: transverse_goniometer\n"
        f"transverse_envelope_x_mm = {old['transverse_envelope_x_mm']}\n"
        f"transverse_envelope_y_mm = {old['transverse_envelope_y_mm']}\n"
        f"length_mm = {old['length_mm']}  (beam gap allocation)\n"
        f"holder_insertion_axis = {old['holder_insertion_axis']!r}\n"
        "sample_holder: absent\nhousing side port: absent\n"
        "No new stage parameters are injected into this baseline.")
    text_new = (
        "CURRENT configs/subassemblies/sample_stage.toml\n"
        "Anchored to the existing sample centre; +X points outward\n\n"
        f"shaft: diameter {holder['holder_shaft_diameter_mm']:g}; X {holder['holder_shaft_start_x_mm']:g}..{holder['holder_shaft_end_x_mm']:g} mm\n"
        f"tip: width {holder['holder_tip_width_mm']:g}; thickness {holder['holder_tip_thickness_mm']:g} mm\n"
        f"seat: diameter {holder['holder_tip_seat_diameter_mm']:g}; depth {holder['holder_tip_seat_depth_mm']:g} mm\n"
        f"beam hole: diameter {holder['holder_tip_opening_diameter_mm']:g} mm\n"
        f"guide: OD {stage['stage_port_outer_diameter_mm']:g} / ID {stage['stage_port_inner_diameter_mm']:g} mm\n"
        f"housing side port: diameter {stage['housing_side_port_diameter_mm']:g} mm\n"
        "5.4 mm pole gap and sample optical position remain unchanged.")
    for i, text in enumerate((text_old, text_new)):
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setPlainText(text)
        box.setMinimumHeight(290)
        box.setStyleSheet("font-family:Consolas;font-size:13px;background:#0b111c;")
        grid.addWidget(box, 1, i)
    grid.setRowStretch(0, 3)
    grid.setRowStretch(1, 1)

    detail, detail_grid = board("Holder and stage: native material details",
        "All dimensions in mm | viewing selections only | the specimen / TEM grid is not added as an invented support solid",
        (1680, 1150))
    detail_cases = [
        ("pole_gap", "Pole gap and flat tip — section", near_meshes,
         dict(section=True, bounds=[[-8., -5., z-12.], [24., 5., z+12.]],
              note="5.4 mm pole gap; 1.2 mm tip. The 7 mm rod begins at X = 18 mm, beyond the central pole noses.")),
        ("tip_seat", "Sample seat — oblique section", tip_meshes,
         dict(camera="isometric", section=True, bounds=[[-3., -3., z-.7], [6., 3., z+.7]],
              note="Seat diameter 3.1, depth 0.65; through-hole diameter 2.4. The seat ledge is explicit; no grid thickness is assumed.")),
        ("holder_only", "Independent holder", holder_meshes,
         dict(section=True, zoom=1.65, note="Flat tip + 7 mm shaft + 40 mm handle. Length and handle dimensions are simulator engineering values.")),
        ("stage_only", "Independent stage — section", stage_meshes,
         dict(section=True, camera="isometric", zoom=1.5, note="9 mm guide bore; 12 mm guide OD; 70 mm flange; 50 mm body. Internal actuator/bearing details are unspecified.")),
    ]
    for i, (slug, title, selected, options) in enumerate(detail_cases):
        widget, view = panel(slug, title, selected, "current", **options)
        detail_grid.addWidget(widget, i // 2, i % 2)

    assembly, assembly_grid = board("Complete holder / stage / objective context — section",
        "Material opening through the +X housing wall; insertion along -X | generic engineering reconstruction, not OEM manufacturing geometry",
        (1450, 1080))
    widget, view = panel("assembly_section", "Side-entry assembly", current_meshes,
        "current", section=True, bounds=bounds, zoom=1.3,
        note="The model retains the objective housing, poles, yoke and excitation coils. Other optical units and EDS hardware are hidden for this mechanical access view.")
    assembly_grid.addWidget(widget, 0, 0)

    for window, filename in ((comparison, "stage_holder_comparison.png"),
                             (detail, "stage_holder_detail.png"),
                             (assembly, "stage_holder_assembly.png")):
        window.show()
        app.processEvents()
        window.repaint()
        app.processEvents()
        if not window.grab().save(str(output / filename)):
            raise RuntimeError(f"Failed to save {filename}")
    # Reject concurrent source changes rather than mislabelling the capture.
    for path, digest in hashes.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during rendering: {path}; rerun the preview")
    manifest = {"schema": "native-stage-holder-preview-v1", "renderer": "Qt PartModelView",
        "platform": app.platformName(), "desktop_automation": False,
        "method": "Actual QWidget.grab; no mesh edits, no screenshot compositing or external rendering",
        "source_sha256": hashes, "angular_segments": args.angular_segments,
        "baseline": {"path": str(original.relative_to(root)), "sample_holder_present": False,
            "new_subassembly_used": False, "legacy_stage_allocation_omitted": old},
        "current": {"path": str(current.relative_to(root)), "sample_local_z_mm": z,
            "sample_holder": holder, "sample_stage": stage}, "panels": records}
    (output / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for window in windows:
        window.close()
    print(f"Saved 3 native stage/holder preview boards to {output}")


if __name__ == "__main__":
    main()
