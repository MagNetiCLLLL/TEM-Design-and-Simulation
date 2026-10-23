"""Export the native probe-corrector material catalog and optional offscreen previews.

The saved baseline, if supplied, must be a catalog captured before modification;
it is never regenerated from the new builder and presented as an old model.
Run export_cad_models.py with the separate CAD Python to produce verified STEP.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys


def corrector_key(key):
    return key.startswith('probe_') or key == 'adapter_lens' or key.startswith('adapter_lens_')


def catalog_from_document(document, source):
    from temsim.cad_catalog import module_catalog
    module = module_catalog(document, source=source)
    module['solids'] = [s for s in module['solids'] if corrector_key(s['key'])]
    ids = {s['key']+'/'+s['region'] for s in module['solids']}
    module['units'] = [dict(u, solid_ids=[s for s in u['solid_ids'] if s in ids])
                       for u in module['units'] if any(s in ids for s in u['solid_ids'])]
    module['omitted'] = [u for u in module['omitted'] if corrector_key(u['key'])]
    return dict(schema='temsim-analytic-material-cad-v1', units='mm', modules=[module],
        scope='Probe corrector material units only; engineering reconstruction, not OEM CAD or field calibration')


def compare(before, after):
    if (before.get('schema') != after.get('schema') or before.get('units') != 'mm'
            or len(before.get('modules', [])) != 1
            or before['modules'][0]['key'] != after['modules'][0]['key']):
        raise ValueError('Baseline must be a millimetre SolidSpec catalog of this probe corrector')
    rows = lambda c: {s['key']+'/'+s['region']: s for s in c['modules'][0]['solids']}
    a, b = rows(before), rows(after)
    common = sorted(a.keys() & b.keys())
    return dict(added=sorted(b.keys()-a.keys()), removed=sorted(a.keys()-b.keys()),
        changed_shapes=[k for k in common if a[k]['shape'] != b[k]['shape']],
        unchanged_shapes=[k for k in common if a[k]['shape'] == b[k]['shape']],
        changed_materials=[k for k in common if a[k]['material'] != b[k]['material']],
        baseline='Supplied pre-change SolidSpec snapshot; no legacy-envelope substitution')


def preview(catalog, baseline, output):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication, QLabel, QGridLayout, QVBoxLayout, QWidget
    from temsim.cad_geometry import SolidSpec
    from temsim.cad_mesh import meshes_from_solids
    from temsim.gui.part_model_view import PartModelView
    app = QApplication.instance() or QApplication([])
    font = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/segoeui.ttf'
    if font.exists():
        families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(font)))
        if families:
            app.setFont(QFont(families[0], 10))
    names = {f.name for f in fields(SolidSpec)}

    def meshes(source, keys=None):
        specs = [SolidSpec(**{k: v for k, v in s.items() if k in names})
                 for s in source['modules'][0]['solids'] if keys is None or s['key'] in keys]
        return meshes_from_solids(specs, angular_segments=96)

    hp_keys = {'probe_hp2_hexapole', 'probe_dph2_coils'}
    current = meshes(catalog, hp_keys)
    old = meshes(baseline, hp_keys) if baseline else None
    records = []

    def panel(layout, row, col, label, items, view_kind, section):
        wrapper = QWidget()
        box = QVBoxLayout(wrapper)
        box.addWidget(QLabel(label))
        view = PartModelView()
        view.setMinimumSize(630, 420)
        view.set_meshes(items)
        view.set_section_enabled(section)
        getattr(view, 'set_'+view_kind+'_view')()
        view.fit_all()
        box.addWidget(view, 1)
        layout.addWidget(wrapper, row, col)
        records.append(dict(label=label, view=view_kind, section_y_positive=section,
            camera_rotation=view.camera_rotation.tolist(),
            solids=[dict(key=m.key, region=m.region, material=m.material_class) for m in items]))
        return view

    window = QWidget()
    window.setStyleSheet('QWidget {background:#101823;color:#e5edf7;} QLabel {font-size:16px;}')
    outer = QVBoxLayout(window)
    outer.addWidget(QLabel('Probe hexapole structure | native material model | dimensions in mm'))
    outer.addWidget(QLabel('Blue: magnetic material   Copper: bulk windings   Same camera and envelope'))
    grid = QGridLayout()
    outer.addLayout(grid)
    if old:
        panel(grid, 0, 0, 'Before | HP2 axial view', old, 'axial', False)
        panel(grid, 1, 0, 'Before | HP2 section', old, 'isometric', True)
    col = 1 if old else 0
    panel(grid, 0, col, 'After | tapered radial poles and winding windows', current, 'axial', False)
    panel(grid, 1, col, 'After | HP2 section with existing DPH2 insert', current, 'isometric', True)
    window.resize(1400 if old else 760, 1050)
    window.show()
    app.processEvents()
    if not window.grab().save(str(output/'probe-structure-comparison.png')):
        raise RuntimeError('Could not save native comparison preview')
    window.close()

    assembly = QWidget()
    assembly.setStyleSheet('QWidget {background:#101823;color:#e5edf7;} QLabel {font-size:16px;}')
    layout = QVBoxLayout(assembly)
    layout.addWidget(QLabel('Probe Corrector | ADL, HP2, TL22, TL21, HP1, TL12 and alignment inserts'))
    layout.addWidget(QLabel('Native section view | configured engineering dimensions | unresolved channels have no material'))
    grid = QGridLayout()
    layout.addLayout(grid)
    panel(grid, 0, 0, 'Same SolidSpec geometry as the STEP assembly', meshes(catalog), 'column_isometric', True)
    assembly.resize(1250, 950)
    assembly.show()
    app.processEvents()
    if not assembly.grab().save(str(output/'probe-corrector-assembly.png')):
        raise RuntimeError('Could not save native corrector assembly preview')
    assembly.close()

    # The narrow steering inserts disappear at whole-column scale. Show the
    # same material recipes at their own scale and in the real saved placement.
    steering = QWidget()
    steering.setStyleSheet('QWidget {background:#101823;color:#e5edf7;} QLabel {font-size:16px;}')
    layout = QVBoxLayout(steering)
    layout.addWidget(QLabel('Corrector steering inserts | four-loop X/Y construction and declared laminate'))
    layout.addWidget(QLabel('Engineering geometry | beige: dielectric | brown: interlayer bond | host fastening unverified'))
    grid = QGridLayout()
    layout.addLayout(grid)
    dp11 = meshes(catalog, {'probe_dp11_coils'})
    panel(grid, 0, 0, 'DP11 | two orthogonal saddle pairs', dp11, 'isometric', False)
    panel(grid, 0, 1, 'DP11 | axial view of concentric X/Y layers', dp11, 'axial', False)
    panel(grid, 1, 0, 'DPH1 | steering insert inside HP1 (section)',
          meshes(catalog, {'probe_hp1_hexapole', 'probe_dph1_coils'}), 'isometric', True)
    relay_keys = {s['key'] for s in catalog['modules'][0]['solids']
                  if s['key'].startswith('probe_tl12_lens')}
    relay_keys.update({'probe_hp1_hexapole', 'probe_dph1_coils', 'probe_dp11_coils'})
    panel(grid, 1, 1, 'HP1 > DP11 > TL12 | saved axial placement (section)',
          meshes(catalog, relay_keys), 'column_isometric', True)
    steering.resize(1500, 1100)
    steering.show()
    app.processEvents()
    if not steering.grab().save(str(output/'probe-deflectors.png')):
        raise RuntimeError('Could not save native corrector steering preview')
    steering.close()

    if baseline:
        support = QWidget()
        support.setStyleSheet('QWidget {background:#101823;color:#e5edf7;} QLabel {font-size:16px;}')
        layout = QVBoxLayout(support)
        layout.addWidget(QLabel('DP11 support comparison | same dimensions and native renderer'))
        layout.addWidget(QLabel('Added: two carriers, two radial covers, one bond layer | bulk reconstruction, not OEM winding detail'))
        grid = QGridLayout()
        layout.addLayout(grid)
        old_dp = meshes(baseline, {'probe_dp11_coils'})
        before_view = panel(grid, 0, 0, 'Before | four copper loops', old_dp, 'isometric', True)
        after_view = panel(grid, 0, 1, 'After | bonded laminate (section)', dp11, 'isometric', True)
        # Use one full-model fit for both sides, even when the supports grow
        # the radial bounds slightly. This makes scale comparisons honest.
        before_view.set_meshes(dp11)
        before_view.fit_all()
        before_view.set_meshes(old_dp, preserve_view=True)
        panel(grid, 1, 0, 'After | carriers and radial covers (copper hidden)',
              tuple(m for m in dp11 if m.material_class == 'insulator'), 'isometric', True)
        panel(grid, 1, 1, 'After | interlayer bond (other materials hidden)',
              tuple(m for m in dp11 if m.region == 'interlayer_bond'), 'isometric', True)
        support.resize(1500, 1100)
        support.show()
        app.processEvents()
        if not support.grab().save(str(output/'deflector-support-comparison.png')):
            raise RuntimeError('Could not save native deflector support comparison')
        support.close()
    return dict(platform=app.platformName(), desktop_automation=False, renderer='PartModelView',
                angular_segments=96, panels=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-catalog', type=Path)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root/'src'))
    from temsim.part_model_document import PartModelDocument
    source = root/'configs/instruments/column/C3_ProbeCorrector.toml'
    catalog = catalog_from_document(PartModelDocument(source).document, str(source.relative_to(root)))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'catalog.json').write_text(json.dumps(catalog, indent=2, allow_nan=False), encoding='utf8')
    baseline = json.loads(args.baseline_catalog.read_text(encoding='utf8')) if args.baseline_catalog else None
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        material_solids=len(catalog['modules'][0]['solids']), units=len(catalog['modules'][0]['units']))
    if baseline:
        report['comparison'] = compare(baseline, catalog)
    if args.preview:
        report['preview'] = preview(catalog, baseline, output)
    (output/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(dict(output=str(output), solids=report['material_solids'], units=report['units'],
                         preview=args.preview)))


if __name__ == '__main__':
    main()
