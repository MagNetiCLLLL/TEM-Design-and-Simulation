"""Capture the application's real PartModelView, without a desktop session.

Only presentation selection is changed: the saved part dimensions and native
meshes are used unchanged. The output manifest records omitted context meshes,
source hashes, camera/section state and the actual rendered mesh inventory.
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
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/cad_models_20261011/preview"))
    parser.add_argument("--angular-segments", type=int, default=64)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QVBoxLayout, QWidget
    from temsim.gui.part_model_view import PartModelView
    from temsim.part_model_3d import part_model_from_document
    from temsim.part_model_document import PartModelDocument

    app = QApplication.instance() or QApplication([])
    # Windows' offscreen platform does not reliably enumerate system fonts.
    # Register the existing local font explicitly, without editing screenshots.
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
    if font_path.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 10))
    column_path = root / "configs/instruments/column/C3_ProbeCorrector.toml"
    gun_path = root / "configs/instruments/gun/FEG.toml"
    documents = {p: PartModelDocument(p).document for p in (column_path, gun_path)}

    def model(path, key):
        return part_model_from_document(documents[path], key,
                                        angular_segments=args.angular_segments)

    hp = model(column_path, "probe_hp2_hexapole")
    ac = model(column_path, "ac_deflector")
    objective = model(column_path, "objective_lens")
    accelerator = model(gun_path, "feg_accelerator")
    objective_keys = {
        "objective_upper_pole", "objective_lower_pole", "objective_lens_housing",
        "objective_lens_yoke", "objective_lens_excitation_coil",
        "mini_condenser_upper_pole", "mini_condenser_lower_pole",
        "mini_condenser_housing", "mini_condenser_yoke",
        "mini_condenser_excitation_coil",
    }
    objective_meshes = tuple(m for m in objective.meshes if m.key in objective_keys)
    cases = [
        ("hp2_dph2", "HP2 + DPH2 insert", hp, hp.meshes, column_path,
         "Six-pole reconstruction; linked DPH2 saddle windings remain inside the bore."),
        ("ac_pair", "AC scan coil pair", ac, ac.meshes, column_path,
         "Two saved axial coil centres; each contains separate X/Y saddle pairs."),
        ("objective_mini", "Objective + mini condenser", objective, objective_meshes, column_path,
         "Physical pole / coil / yoke / housing children. Parent envelopes and other units hidden."),
        ("accelerator", "Accelerator electrode stack", accelerator, accelerator.meshes, gun_path,
         "Configured electrode centres and thickness; separate insulating spacers."),
    ]
    window = QWidget()
    window.setWindowTitle("Native CAD material previews")
    window.setStyleSheet("QWidget {background:#101823;color:#e5edf7;font-family:Segoe UI;}"
                         "QLabel {background:transparent;} ")
    outer = QVBoxLayout(window)
    heading = QLabel("Native 3D views — saved dimensions and generic internal reconstruction")
    heading.setStyleSheet("font-size:21px;font-weight:600;")
    outer.addWidget(heading)
    scope = QLabel("Actual PartModelView captures • Orthographic section, y ≥ 0 retained • "
                   "Not OEM CAD or a field solution • No geometry was edited")
    scope.setStyleSheet("font-size:13px;color:#b8c9da;")
    outer.addWidget(scope)
    grid = QGridLayout()
    grid.setSpacing(12)
    outer.addLayout(grid)
    records = []
    panels = []
    for i, (slug, title, source_model, meshes, path, note) in enumerate(cases):
        if not meshes:
            raise RuntimeError(f"No native meshes for {slug}")
        panel = QWidget()
        layout = QVBoxLayout(panel)
        label = QLabel(title)
        label.setStyleSheet("font-size:18px;font-weight:600;")
        layout.addWidget(label)
        view = PartModelView()
        view.setMinimumSize(790, 460)
        view.set_view_labels(title=title)
        view.set_meshes(meshes)
        view.set_section_enabled(True)
        view.set_isometric_view()
        view.fit_all()
        layout.addWidget(view, 1)
        colors = {}
        for mesh in meshes:
            # Preserve native colors; the legend describes actual material
            # category metadata rather than inventing a new color convention.
            colors.setdefault(mesh.material_class, mesh.color)
        legend = []
        for material, color in colors.items():
            rgb = "#" + "".join(f"{round(255*float(c)):02x}" for c in color[:3])
            legend.append(f'<span style="color:{rgb}">■</span> {material}')
        material_label = QLabel(" &nbsp; ".join(legend))
        material_label.setTextFormat(Qt.TextFormat.RichText)
        material_label.setWordWrap(True)
        material_label.setStyleSheet("font-size:12px;")
        layout.addWidget(material_label)
        description = QLabel(note)
        description.setWordWrap(True)
        description.setStyleSheet("font-size:12px;color:#b8c9da;")
        layout.addWidget(description)
        grid.addWidget(panel, i // 2, i % 2)
        panels.append((slug, panel, view))
        shown = {(m.key, m.region) for m in meshes}
        records.append({
            "id": slug, "title": title, "source": str(path.relative_to(root)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "part_key": source_model.part_key, "selection_note": note,
            "section_enabled": True, "section_retains": "y >= 0 mm",
            "camera_rotation": view.camera_rotation.tolist(),
            "native_meshes": [{"key": m.key, "region": m.region,
                               "material_class": m.material_class,
                               "vertices": len(m.vertices), "triangles": len(m.faces),
                               "is_exact": m.is_exact, "description": m.description}
                              for m in meshes],
            "omitted_context_meshes": [{"key": m.key, "region": m.region}
                                       for m in source_model.meshes
                                       if (m.key, m.region) not in shown],
            "material_counts": dict(Counter(m.material_class for m in meshes)),
            "model_notes": list(source_model.notes),
        })
    window.resize(1680, 1320)
    window.show()
    app.processEvents()
    for slug, panel, view in panels:
        view.repaint()
        app.processEvents()
        if not panel.grab().save(str(output / f"{slug}.png")):
            raise RuntimeError(f"Could not save {slug}")
    if not window.grab().save(str(output / "native_material_previews.png")):
        raise RuntimeError("Could not save the four-panel native preview")
    source_files = ["scripts/preview_cad_models.py", "src/temsim/gui/part_model_view.py",
                    "src/temsim/part_model_3d.py", "src/temsim/cad_catalog.py",
                    "src/temsim/cad_mesh.py", "src/temsim/cad_multipoles.py",
                    "src/temsim/cad_auxiliary.py", "src/temsim/corrector_model_3d.py"]
    manifest = {"schema": "native-part-model-preview-v1", "renderer": "PartModelView",
                "platform": app.platformName(), "desktop_automation": False,
                "method": "QWidget.grab of actual native 3D views; no external image rendering or editing",
                "scope": "Configured dimensions; generic reconstructed internals, not OEM CAD",
                "angular_segments": args.angular_segments, "panels": records,
                "implementation_sha256": {p: hashlib.sha256((root/p).read_bytes()).hexdigest()
                                          for p in source_files}}
    (output / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    window.close()
    print(f"Saved {len(cases)} native section previews and collage to {output}")


if __name__ == "__main__":
    main()
