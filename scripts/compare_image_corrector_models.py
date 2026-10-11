"""Render the saved pre-repair image corrector against its current native CAD.

This is a mechanical comparison, not an optical simulation. The legacy input
is read as a frozen document without migrating it into the new runtime model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib
from unittest.mock import patch

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meshes(document, *, legacy=False, keys=None):
    from temsim.part_model_3d import _part_meshes
    by_key = {p['key']: p for p in document['parts']}
    def build():
        result = []
        for key, part in by_key.items():
            if key.startswith('image_') and (keys is None or key in keys):
                values, _ = _part_meshes(part, by_key, 64)
                result.extend(values)
        return tuple(result)
    if legacy:
        # Exact old TOML geometry through the pre-CAD native envelope route.
        # No source files or current runtime instrument are changed.
        with patch('temsim.cad_catalog.native_reconstruction', return_value=None):
            return build()
    return build()


def render_pair(output, name, title, old, new, *, column=True, captions=None, explanation=None):
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
    from temsim.gui.part_model_view import PartModelView
    from compare_cad_assembly import normalized_palette
    app = QApplication.instance()
    window = QWidget()
    window.setStyleSheet('QWidget {background:#101823;color:#e5edf7;} QLabel {background:transparent;}')
    layout = QVBoxLayout(window)
    heading = QLabel(title)
    heading.setStyleSheet('font-size:20px;font-weight:600;')
    layout.addWidget(heading)
    layout.addWidget(QLabel('Same module-local coordinates, camera and scale | Native PartModelView | Engineering reconstruction, not OEM CAD'))
    panels = QHBoxLayout()
    layout.addLayout(panels)
    views = []
    groups = [tuple(normalized_palette(m) for m in items) for items in (old, new)]
    captions = captions or ('Saved original: OLpost body + integrated TL12', 'Revised: five round lenses; TL11 / DP12 / TL12 separated')
    for caption, items in zip(captions, groups):
        panel = QWidget()
        inner = QVBoxLayout(panel)
        label = QLabel(caption)
        label.setWordWrap(True)
        inner.addWidget(label)
        view = PartModelView()
        view.setFixedSize(710, 810 if column else 540)
        view.set_meshes(items)
        view.set_section_enabled(True, keep_positive_y=not column)
        if column:
            view.set_column_view()
        else:
            view.set_isometric_view()
        view.set_view_labels(title='Material section')
        inner.addWidget(view)
        panels.addWidget(panel)
        views.append(view)
    points = [m.vertices for group in groups for m in group]
    for view in views:
        view._fit_points(points)
    note = QLabel('Copper: bulk winding; blue: magnetic material; grey: housing or unspecified envelope. '
                  + (explanation or 'DPH and other unresolved control channels have no fabricated material. '
                      'The legacy OLpost allocation now houses TL11; its objective interpretation is removed.'))
    note.setWordWrap(True)
    layout.addWidget(note)
    window.show()
    app.processEvents()
    for view in views:
        view.repaint()
    app.processEvents()
    path = output / (name + '.png')
    if not window.grab().save(str(path)):
        raise RuntimeError(f'Cannot save {path}')
    same = np.array_equal(views[0].camera_rotation, views[1].camera_rotation) and views[0]._scale() == views[1]._scale()
    record = dict(file=path.name, sha256=sha(path), renderer='PartModelView', platform=app.platformName(),
                  desktop_automation=False, same_camera_and_scale=bool(same), pixels_per_mm=views[0]._scale(),
                  camera_rotation=views[0].camera_rotation.tolist(), centre_mm=views[0]._center.tolist(),
                  section_retains='y <= 0' if column else 'y >= 0',
                  old_mesh_count=len(old), new_mesh_count=len(new))
    window.close()
    if not same:
        raise RuntimeError('Comparison cameras differ')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--originals', type=Path, default=Path('outputs/image_corrector_20261011/mechanical-originals'))
    parser.add_argument('--output', type=Path, default=Path('outputs/image_corrector_20261011/mechanical-comparison'))
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'src'))
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QFontDatabase
    from temsim.part_model_document import PartModelDocument
    from compare_cad_assembly import mesh_summary, dimensional_checks, bounds
    app = QApplication.instance() or QApplication([])
    font = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/segoeui.ttf'
    if font.exists():
        families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(font)))
        if families:
            app.setFont(QFont(families[0], 10))
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(schema='image-corrector-mechanical-before-after-v1', scope='Mechanical topology and native rendering only; no ray/wave correctness claim.',
                  evidence_scope='CETCOR pin/coil/yoke family inferred from author text and the CEOS six-pin family photograph; original CETCOR mechanical figure unavailable.',
                  modules=[], images=[])
    for filename in ('C3_ImageCorrector.toml', 'C3_ProbeCorrector_ImageCorrector.toml'):
        old_path = args.originals / filename
        new_path = root / 'configs/instruments/column' / filename
        old = tomllib.loads(old_path.read_text(encoding='utf-8'))
        new = PartModelDocument(new_path).document
        source_hash = sha(new_path)
        a, b = meshes(old, legacy=True), meshes(new)
        old_parts = {p['key']: p for p in old['parts'] if p['key'].startswith('image_')}
        new_parts = {p['key']: p for p in new['parts'] if p['key'].startswith('image_')}
        fields = ('local_start_z_mm', 'local_center_z_mm', 'local_end_z_mm', 'mechanical_profile',
                  'mechanical_outer_diameter_mm', 'effective_length_mm', 'parent_key', 'optical_reference_host_key')
        rows = []
        for key in sorted(old_parts.keys() | new_parts.keys()):
            before, after = old_parts.get(key, {}), new_parts.get(key, {})
            old_mesh = [m for m in a if m.key == key]
            new_mesh = [m for m in b if m.key == key]
            rows.append(dict(key=key, before={k: before[k] for k in fields if k in before},
                             after={k: after[k] for k in fields if k in after},
                             old_meshes=[mesh_summary(m) for m in old_mesh], new_meshes=[mesh_summary(m) for m in new_mesh],
                             new_envelope_check=dimensional_checks(after, bounds(new_mesh, after, new_parts), new_parts) if after else None))
        module = dict(key=new['module']['key'], original=str(old_path), current=str(new_path), original_sha256=sha(old_path), current_sha256=source_hash,
                      old_display_meshes=len(a), new_display_meshes=len(b), parts=rows,
                      old_wireframe_meshes=sum(bool(m.wireframe) for m in a),
                      new_wireframe_meshes=sum(bool(m.wireframe) for m in b),
                      removed_keys=sorted(old_parts.keys()-new_parts.keys()), added_keys=sorted(new_parts.keys()-old_parts.keys()),
                      current_material_counts=dict(Counter(m.material_class for m in b)))
        report['modules'].append(module)
        report['images'].append(render_pair(args.output, filename[:-5], new['module']['name'] + ' — image section', a, b))
        if filename == 'C3_ImageCorrector.toml':
            hp = 'image_hp1_hexapole'
            report['images'].append(render_pair(args.output, 'HP1_detail', 'Image HP1 — envelope versus documented structure family',
                tuple(m for m in a if m.key == hp), tuple(m for m in b if m.key == hp), column=False,
                captions=('Original: unresolved HP envelope', 'Revised: six pins + six bulk coils + one yoke'),
                explanation='Same 180 mm OD / 8 mm bore / 44 mm envelope; 30 mm configured pole allocation. '
                            'CETCOR family inference, not a verified OEM cross-section. DPH remains an optical channel.'))
            old_keys = {k for k in old_parts if k.startswith(('image_ol_post_lens', 'image_tl11_lens'))}
            new_keys = {k for k in new_parts if k.startswith(('image_tl11_lens', 'image_tl12_lens'))}
            report['images'].append(render_pair(args.output, 'transfer_pair_detail', 'Reallocated transfer-lens material — same physical scale',
                tuple(m for m in a if m.key in old_keys), tuple(m for m in b if m.key in new_keys), column=False,
                captions=('Original: independent OLpost (180 mm OD) + TL11 (180 mm OD)',
                          'Revised: TL11 (180 mm OD) + TL12 (160 mm OD)'),
                explanation='Retained axial blocks are 45 mm and 46 mm long, with a 28.5 mm gap containing the DP12 field. '
                            'Five material children per lens; millimetre geometry is an engineering allocation.'))
        if sha(new_path) != source_hash:
            raise RuntimeError('Input changed during comparison; regenerate against a frozen input')
    report['status'] = 'COMPLETE_MECHANICAL_COMPARISON'
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Image corrector mechanical comparison', '', report['scope'], '', report['evidence_scope'], '',
             'The same coordinate system, camera and scale are used in every paired native view. The old inputs are preserved without migration.', '',
             'Removed OLpost identities are replaced by TL11 hardware; the old TL11 material becomes a 160 mm-OD TL12. HP1 and downstream centres and module ports remain fixed.', '',
             'DP12 is now in the TL11/TL12 gap. DPH1/2 follow HP1/2 as optical channels; no independent material winding is invented.', '']
    lines.extend(f'- [{image["file"]}]({image["file"]})' for image in report['images'])
    (args.output / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], images=len(report['images']), modules=len(report['modules']))), flush=True)


if __name__ == '__main__':
    main()
