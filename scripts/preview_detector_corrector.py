"""Capture actual Qt detector/corrector material views with source receipts.

No optical or mechanical parameter is edited. Sectioning and context selection
are presentation choices, listed in each image and its manifest. This uses the
same PartModelView and native meshes as Physical Layout / 3D Parts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/detector_corrector_cad_20261011/preview"))
    parser.add_argument("--angular-segments", type=int, default=64)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QVBoxLayout, QWidget
    from temsim.gui.part_model_view import PartModelView, _view_rotation
    from temsim.cad_detectors import eds_head_geometry
    from temsim.part_model_3d import module_model_from_document, part_model_from_document
    from temsim.part_model_document import PartModelDocument
    from temsim.shared_tip import dependencies

    column_path = root / "configs/instruments/column/C3_ImageCorrector.toml"
    filter_path = root / "configs/instruments/project_and_recording_system/EnergyFilter.toml"
    inputs = {column_path, filter_path}
    for path in tuple(inputs):
        inputs.update(dependencies(path))
    from temsim.paths import EDS_DETECTOR_CONFIG_ROOT
    inputs.update(Path(EDS_DETECTOR_CONFIG_ROOT).glob("*.toml"))
    implementation = [Path(__file__).resolve(), *(root / "src/temsim" / name for name in (
        "part_model_3d.py", "cad_catalog.py", "cad_detectors.py", "cad_geometry.py", "cad_mesh.py",
        "cad_multipoles.py", "cad_round_lenses.py", "corrector_model_3d.py",
        "energy_filter_model_3d.py", "part_materials.py", "gui/part_model_view.py"))]
    files = sorted(inputs | set(implementation))
    hashes = {str(p.relative_to(root)): _sha(p) for p in files}
    documents = {p: PartModelDocument(p).document for p in (column_path, filter_path)}

    def model(path, key):
        return part_model_from_document(documents[path], key, angular_segments=args.angular_segments)

    eds = model(column_path, "eds_detector_system")
    objective = model(column_path, "objective_lens")
    context_keys = {"objective_upper_pole", "objective_lower_pole"}
    context = tuple(m for m in objective.meshes if m.key in context_keys)
    eels = model(filter_path, "energy_filter_zebra")
    hp = model(column_path, "image_hp1_hexapole")
    column = module_model_from_document(documents[column_path], angular_segments=args.angular_segments)
    rows = {p["key"]: p for p in documents[column_path]["parts"]}
    eds_geometry = eds_head_geometry(rows["eds_detector_system"], rows)
    head_frame = np.asarray(eds_geometry["heads"][0]["rotation"])
    corrector_start = rows["image_hpol_hexapole"]["local_start_z_mm"]
    corrector_keys = {key for key, part in rows.items() if part.get("branch") == "image"
                      and float(part["local_start_z_mm"]) >= corrector_start}
    corrector = tuple(m for m in column.meshes if m.key in corrector_keys)
    head = tuple(m for m in eds.meshes if m.region.startswith("segment_1_"))
    # Existing native material colours are retained in every view.
    cases = [
        dict(id="eds_objective", title="EDS array + objective context", path=column_path,
             meshes=(*context, *eds.meshes), source=(*objective.meshes, *eds.meshes), section=True,
             view="isometric", note="Section y >= 0. Objective poles retained; objective housing, coil, yoke and other context hidden. "
             "Six windowless heads are engineering material illustrations; angular acceptance is unchanged."),
        dict(id="eels_camera", title="EELS camera: material + readout guide", path=filter_path,
             meshes=eels.meshes, source=eels.meshes, section=False, view="isometric",
             note="All five conversion strips, optical backing and housing shown. Orange alignment outline "
             "remains a logical guide at the original readout plane, not a second material sensor."),
        dict(id="image_hp1_dph1", title="Image HP1 + DPH1 insert", path=column_path,
             meshes=hp.meshes, source=hp.meshes, section=True, view="isometric",
             note="Section y >= 0. Six poles, whole coil packs, return yoke and DPH saddle insert. "
             "Simplified material topology; no microscopic windings or unverified hardware added."),
        dict(id="image_corrector", title="Image corrector: complete internal sequence", path=column_path,
             meshes=corrector, source=column.meshes, section=True, keep_positive_y=False, view="column",
             note="Section y <= 0. TL11 - DP12 - TL12 - HP1 - TL21/TL22 - HP2 - ADL. "
             "All material belonging to this branch retained; other column modules hidden. Uniform spatial scale."),
        dict(id="eds_head_detail", title="EDS head: separate sensor, support and shield", path=column_path,
             meshes=head, source=eds.meshes, section=False, view="detector-front",
             note="One of the six heads, enlarged view. Other heads hidden. Silicon die illustration, "
             "insulating support and open-front cold shield; no inferred active area, cooler or electronics."),
    ]
    app = QApplication.instance() or QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
    if font_path.exists():
        families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(font_path)))
        if families:
            app.setFont(QFont(families[0], 10))
    window = QWidget()
    window.setWindowTitle("Detector and image-corrector native CAD views")
    window.setStyleSheet("QWidget {background:#101823;color:#e5edf7;font-family:Segoe UI;} QLabel {background:transparent;}")
    outer = QVBoxLayout(window)
    heading = QLabel("Native detector and image-corrector material views")
    heading.setStyleSheet("font-size:21px;font-weight:600;")
    outer.addWidget(heading)
    scope = QLabel("Actual PartModelView captures | Saved dimensions | Engineering reconstruction, not OEM CAD | No geometry edited")
    scope.setWordWrap(True)
    scope.setStyleSheet("font-size:13px;color:#b8c9da;")
    outer.addWidget(scope)
    grid = QGridLayout()
    outer.addLayout(grid)
    panels, records = [], []
    for index, case in enumerate(cases):
        meshes = case["meshes"]
        if not meshes:
            raise RuntimeError(f"No native meshes for {case['id']}")
        panel = QWidget()
        panel.setStyleSheet(window.styleSheet())
        layout = QVBoxLayout(panel)
        title = QLabel(case["title"])
        title.setStyleSheet("font-size:18px;font-weight:600;")
        layout.addWidget(title)
        view = PartModelView()
        view.setMinimumSize(770, 440)
        view.set_view_labels(title=case["title"])
        view.set_meshes(meshes)
        view.set_section_enabled(case["section"], keep_positive_y=case.get("keep_positive_y", True))
        if case["view"] == "detector-front":
            # The ordinary trackball supports this same proper rotation. A
            # front oblique camera reveals the open shield without moving it.
            view._rotation = _view_rotation(-head_frame[:, 2] + .7*head_frame[:, 0] + .4*head_frame[:, 1])
            view._invalidate()
        else:
            (view.set_column_view if case["view"] == "column" else view.set_isometric_view)()
        view.fit_all()
        layout.addWidget(view, 1)
        colors = {m.material_class: m.color for m in meshes if not m.wireframe}
        swatches = []
        for material, color in colors.items():
            rgb = "#" + "".join(f"{round(255*float(c)):02x}" for c in color[:3])
            swatches.append(f'<span style="color:{rgb}">■</span> {material}')
        legend = QLabel(" &nbsp; ".join(swatches))
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setWordWrap(True)
        legend.setStyleSheet("font-size:12px;")
        layout.addWidget(legend)
        note = QLabel(case["note"])
        note.setWordWrap(True)
        note.setStyleSheet("font-size:12px;color:#b8c9da;")
        layout.addWidget(note)
        if index < 4:
            grid.addWidget(panel, index // 2, index % 2)
        else:
            panel.resize(960, 660)
        shown = {(m.key, m.region) for m in meshes}
        vertices = np.concatenate([m.vertices for m in meshes])
        records.append(dict(id=case["id"], title=case["title"], source=str(case["path"].relative_to(root)),
            display_note=case["note"], section_enabled=case["section"],
            section_retains=("y >= 0" if case.get("keep_positive_y", True) else "y <= 0") if case["section"] else "all",
            camera_rotation=view.camera_rotation.tolist(),
            bounds_mm=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            native_meshes=[dict(key=m.key, name=rows.get(m.key, {}).get("name", m.key), region=m.region,
                material=m.material_class, vertices=len(m.vertices), triangles=len(m.faces),
                wireframe=m.wireframe, description=m.description) for m in meshes],
            material_counts=dict(Counter(m.material_class for m in meshes)),
            hidden_context=[dict(key=m.key, region=m.region) for m in case["source"] if (m.key, m.region) not in shown]))
        panels.append((case["id"], panel, view))
    window.resize(1650, 1300)
    window.show()
    app.processEvents()
    for slug, panel, view in panels:
        if slug == "eds_head_detail":
            panel.show()
        view.repaint()
        app.processEvents()
        if not panel.grab().save(str(output / f"{slug}.png")):
            raise RuntimeError(f"Could not save {slug}")
    if not window.grab().save(str(output / "detector_corrector_native.png")):
        raise RuntimeError("Could not save native overview")
    after = {str(p.relative_to(root)): _sha(p) for p in files}
    changed = [p for p in hashes if hashes[p] != after[p]]
    report = dict(schema="native-detector-corrector-preview-v1", renderer="PartModelView",
                  platform=app.platformName(), desktop_automation=False,
                  method="Actual QWidget.grab; no external rendering, image editing, or geometry changes",
                  angular_segments=args.angular_segments, panels=records,
                  source_sha256=hashes, source_sha256_after=after,
                  source_unchanged=not changed, changed_sources=changed,
                  images={p.name: _sha(p) for p in sorted(output.glob("*.png"))})
    (output / "preview_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    window.close()
    for _, panel, _ in panels:
        panel.close()
    if changed:
        raise RuntimeError("Sources changed during capture: " + ", ".join(changed))
    print(f"Captured {len(cases)} real Qt panels and overview: {output}")


if __name__ == "__main__":
    main()
