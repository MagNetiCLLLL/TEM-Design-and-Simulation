"""Reproduce lightweight, native stigmator and objective-package views.

Saved TOML geometry is read without modification. A separate in-memory copy
rotates the two condenser-stigmator bases by 15 degrees to demonstrate their
CAD binding. Rendering uses the application's actual CPU PartModelView.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/stigmator-research-20261019/preview"))
    parser.add_argument("--angular-segments", type=int, default=32)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    started = perf_counter()
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget
    from temsim.gui.part_model_view import PartModelView
    from temsim.lens_pose import effective_part_transform_mm, mechanical_part_transform_mm
    from temsim.part_model_3d import part_model_from_document
    from temsim.part_model_document import PartModelDocument

    app = QApplication.instance() or QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
    if font_path.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 11))
    source = root / "configs/instruments/column/C3_ProbeCorrector.toml"
    source_bytes = source.read_bytes()
    saved = PartModelDocument(source).document
    changed = deepcopy(saved)
    for row in changed["parts"]:
        if row["key"] == "condenser_stigmator":
            row.update(channel_x_angle_deg=15., channel_y_angle_deg=60.)
    documents = {"saved": saved, "basis_rotated": changed}
    indices = {name: {r["key"]: r for r in doc["parts"]} for name, doc in documents.items()}
    cache = {}

    def model(version, key):
        identity = (version, key)
        if identity not in cache:
            cache[identity] = part_model_from_document(documents[version], key,
                angular_segments=args.angular_segments, include_children=False)
        return cache[identity]

    mini = ["mini_condenser_upper_pole", "mini_condenser_lower_pole",
            "mini_condenser_yoke", "mini_condenser_excitation_coil"]
    upper = ["ac_deflector", "condenser_stigmator", *mini, "objective_upper_pole"]
    lower = ["objective_lower_pole", "image_diffraction_deflector", "objective_stigmator"]
    sleeves = ["objective_upper_electronics_sleeve", "objective_lower_electronics_sleeve"]
    cases = [
        dict(id="condenser_stigmator_default", title="Condenser stigmator | saved 0 / 45 degree bases",
             version="saved", keys=["condenser_stigmator"], section=False, camera="isometric",
             note="Eight bulk copper windings at one axial station. X: coils 1, 3, 5, 7; Y: coils 2, 4, 6, 8."),
        dict(id="condenser_stigmator_rotated", title="Condenser stigmator | temporary 15 / 60 degree bases",
             version="basis_rotated", keys=["condenser_stigmator"], section=False, camera="isometric",
             note="In-memory structural-angle demonstration only. Saved TOML, optical excitation and axial positions are unchanged."),
        dict(id="objective_internal_package", title="C3 + Probe Corrector | objective internal package",
             version="saved", keys=upper+lower, section=False, camera="column_isometric",
             note="Selected internals at their saved coordinates. Outer objective coil, yoke, housing, tube, holder and specimen omitted for visibility."),
        dict(id="upper_objective_package", title="Upper package | AC, condenser stigmator and mini condenser",
             version="saved", keys=upper, section=True, camera="column_isometric",
             note="Native half-section: y <= 0 retained. AC is upstream of the condenser stigmator; mini condenser precedes the upper pole."),
        dict(id="lower_objective_package", title="Lower package | image/descan pair and objective stigmator",
             version="saved", keys=lower, section=True, camera="column_isometric",
             note="Native half-section: y <= 0 retained. This selected configuration places image/descan upstream of the objective stigmator."),
        dict(id="objective_sleeves_exterior", title="Objective electronics | opaque polymer sleeve exterior",
             version="saved", keys=upper+lower+sleeves, section=False, camera="column_isometric",
             note="Two open-ended polymer carriers enclose the electronics. They stop outside the objective poles. Outer objective housing is omitted."),
        dict(id="objective_sleeves_section", title="Objective electronics | polymer sleeves in half-section",
             version="saved", keys=upper+lower+sleeves, section=True, camera="column_isometric",
             note="Native half-section exposes the contained coils. Sleeve ID / OD: 59.2 / 59.8 mm; 0.3 mm wall. Engineering dimensions, polymer grade unspecified."),
        dict(id="upper_sleeve_section", title="Upper electronics | open polymer sleeve half-section",
             version="saved", keys=upper+[sleeves[0]], section=True, camera="column_isometric",
             note="Passive sleeve around AC, condenser stigmator and mini condenser. Nominal radial clearance to mini housing: 0.1 mm. No added field or vacuum boundary."),
        dict(id="lower_sleeve_section", title="Lower electronics | open polymer sleeve half-section",
             version="saved", keys=lower+[sleeves[1]], section=True, camera="column_isometric",
             note="Passive sleeve around image/descan and objective stigmator. Separate from the lower objective pole. No extra descan winding is introduced."),
    ]
    records = []
    for case in cases:
        by_key = indices[case["version"]]
        models = [model(case["version"], key) for key in case["keys"]]
        meshes = tuple(mesh for item in models for mesh in item.meshes)
        if not meshes:
            raise RuntimeError(f"No native material meshes for {case['id']}")
        window = QWidget()
        window.setStyleSheet("QWidget {background:#101823;color:#e5edf7;font-family:Segoe UI;}"
                             "QLabel {background:transparent;}")
        outer = QVBoxLayout(window)
        heading = QLabel(case["title"])
        heading.setStyleSheet("font-size:24px;font-weight:600;")
        outer.addWidget(heading)
        caveat = QLabel("Configured simulator geometry | generic reconstruction, not OEM CAD or a field solution")
        caveat.setStyleSheet("font-size:14px;color:#f6cb83;")
        outer.addWidget(caveat)
        body = QHBoxLayout()
        outer.addLayout(body, 1)
        view = PartModelView()
        view.setMinimumSize(820, 650)
        view.set_meshes(meshes)
        view.set_section_enabled(case["section"], keep_positive_y=False)
        view.set_view_labels(title="Native geometry / module-local mm")
        getattr(view, "set_"+case["camera"]+"_view")()
        view.fit_all()
        body.addWidget(view, 1)
        side = QWidget()
        side.setFixedWidth(350)
        sidebar = QVBoxLayout(side)
        legend = QLabel("<b>Native materials</b><br><span style='color:#d46e33'>Copper windings</span><br>"
                        "<span style='color:#5c7da3'>Magnetic poles / mini yoke</span><br>"
                        "<span style='color:#ccc4a8'>Unspecified insulating polymer</span>")
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setStyleSheet("font-size:15px;")
        sidebar.addWidget(legend)
        description = QLabel(case["note"])
        description.setWordWrap(True)
        description.setStyleSheet("font-size:15px;color:#b8c9da;")
        sidebar.addWidget(description)
        sidebar.addWidget(QLabel("<b>Saved axial stations, local Z (mm)</b>"))
        station_keys = [key for key in case["keys"] if not key.startswith("mini_condenser_")]
        if any(key.startswith("mini_condenser_") for key in case["keys"]):
            station_keys.append("mini_condenser")
        for key in sorted(station_keys, key=lambda k: by_key[k]["local_center_z_mm"]):
            row = by_key[key]
            centres = row.get("interaction_centers_local_z_mm")
            detail = ("coil centres "+" / ".join(f"{z:.1f}" for z in centres) if centres else
                      f"centre {row['local_center_z_mm']:.1f}")
            label = QLabel(f"{row.get('name', key)}\n{detail}")
            label.setWordWrap(True)
            label.setStyleSheet("font-size:14px;padding-top:5px;")
            sidebar.addWidget(label)
        sidebar.addStretch()
        body.addWidget(side)
        footer = QLabel("Bulk windings and user-requested polymer carriers are generic engineering geometry. "
                        "Image shift and descan share one physical pair; no separate descan body is drawn.")
        footer.setWordWrap(True)
        footer.setStyleSheet("font-size:14px;color:#b8c9da;")
        outer.addWidget(footer)
        window.resize(1240, 920 if len(meshes) > 8 else 800)
        # Exercise the normal native software fallback, avoiding a GPU-context
        # allocation during this low-resource, reproducible read-only preview.
        with patch("temsim.gui.part_model_view._create_gpu_renderer",
                   side_effect=RuntimeError("CPU requested for research preview")):
            window.show()
            app.processEvents()
            view.repaint()
            app.processEvents()
            if not window.grab().save(str(output / (case["id"]+".png"))):
                raise RuntimeError(f"Failed to capture {case['id']}")
        dimensions = []
        for key in case["keys"]:
            row = by_key[key]
            material_r, material_t = mechanical_part_transform_mm(row, by_key)
            optical_r, optical_t = effective_part_transform_mm(row, by_key)
            centre = np.array([0., 0., row["local_center_z_mm"]])
            dimensions.append(dict(key=key, parent_key=row.get("parent_key"),
                configured={k: value for k, value in row.items()
                            if k.endswith(("_mm", "_deg", "_mrad")) or k in
                            {"coil_count", "coil_topology", "field_geometry_status", "physical_host_key"}},
                material_transform=dict(rotation=material_r.tolist(), offset_mm=material_t.tolist()),
                optical_transform=dict(rotation=optical_r.tolist(), offset_mm=optical_t.tolist()),
                material_center_module_mm=(material_r@centre+material_t).tolist()))
        records.append({**case, "image": case["id"]+".png", "renderer_backend": view.renderer_backend,
            "renderer_detail": view.renderer_detail, "camera_rotation": view.camera_rotation.tolist(),
            "section_retains": "y <= 0" if case["section"] else "full geometry",
            "temporary_overrides": ({"condenser_stigmator": {"channel_x_angle_deg": 15., "channel_y_angle_deg": 60.}}
                                    if case["version"] == "basis_rotated" else {}),
            "part_dimensions_and_poses": dimensions,
            "native_meshes": [{"key": mesh.key, "region": mesh.region,
                "material_class": mesh.material_class, "vertices": len(mesh.vertices), "triangles": len(mesh.faces),
                "bounds_module_mm": [mesh.vertices.min(axis=0).tolist(), mesh.vertices.max(axis=0).tolist()],
                "description": mesh.description} for mesh in meshes],
            "omitted_parts": [key for key in by_key if key not in case["keys"]],
            "notes": list(dict.fromkeys(note for item in models for note in item.notes))})
        window.close()
        window.deleteLater()
        app.processEvents()
        print(f"Saved {case['id']}: {len(meshes)} meshes, {view.renderer_backend}", flush=True)
    if source.read_bytes() != source_bytes:
        raise RuntimeError("The source TOML changed during capture; rerun against a stable input")
    implementation = ["scripts/preview_stigmator_hardware.py", "src/temsim/cad_multipoles.py",
                      "src/temsim/cad_objective_sleeves.py", "src/temsim/cad_catalog.py", "src/temsim/cad_geometry.py",
                      "src/temsim/part_model_3d.py", "src/temsim/gui/part_model_view.py",
                      "src/temsim/cad_mesh.py", "src/temsim/lens_pose.py"]
    manifest = dict(schema="native-stigmator-hardware-preview-v1", renderer="PartModelView",
        method="Native QWidget.grab; no external image editing or scene replacement",
        platform=app.platformName(), angular_segments=args.angular_segments,
        source=str(source.relative_to(root)), source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        saved_geometry_unchanged=True, physics_calculation_performed=False,
        coordinates="module-local mm; no assumed gun or accessory offset added",
        scope="Generic simulator reconstruction; no OEM dimensions, winding currents or magnetic material solve inferred",
        elapsed_seconds=perf_counter()-started, cases=records,
        implementation_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in implementation})
    (output/"preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Finished {len(records)} native views in {manifest['elapsed_seconds']:.1f}s: {output}", flush=True)


if __name__ == "__main__":
    main()
