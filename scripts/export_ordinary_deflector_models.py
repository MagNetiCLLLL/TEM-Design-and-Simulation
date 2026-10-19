"""Freeze ordinary deflector materials and compare against a saved baseline.

Uses the normal SolidSpec catalog, native offscreen viewer and STEP exporter.
No desktop interaction, optical solving or configuration mutation occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

KEYS = ('condenser_deflector', 'beam_deflector', 'ac_deflector', 'image_diffraction_deflector')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def selected_catalog(module):
    module = deepcopy(module)
    module['solids'] = [s for s in module['solids'] if s['key'] in KEYS]
    module['units'] = [dict(key=k, name=k, solid_ids=[s['key']+'/'+s['region']
                       for s in module['solids'] if s['key'] == k]) for k in KEYS]
    return dict(schema='temsim-analytic-material-cad-v1', units='mm', modules=[module],
                scope='Four ordinary double-stage deflectors in C3 + Probe Corrector; non-OEM engineering geometry')


def collision_input(module, document):
    """Include nearby saved material and the actual continuous tube profile.

    The tube is clipped to each material station's Z span; declared aperture
    slots use the normal material-tube adapter. No tube is inferred beyond the
    saved continuous material tube's end.
    """
    from temsim.cad_geometry import SolidSpec, difference, revolve
    from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, SLOTS, decorate_physical_tube_row
    from temsim.ordinary_deflector_geometry import support_dimensions
    rows = {p['key']: p for p in document['parts']}
    spans = [(rows[k]['local_start_z_mm'], rows[k]['local_end_z_mm']) for k in KEYS]
    adjacent = {k for k, p in rows.items() if any(p['local_start_z_mm'] < b and
                p['local_end_z_mm'] > a for a, b in spans)}
    module = deepcopy(module)
    module['solids'] = [s for s in module['solids'] if s['key'] in adjacent]
    geometry = document['geometry']
    profile = geometry['mechanical_material_tube_outer_profile_z_od_mm']
    bore = geometry['c1_c2_objective_vacuum_tube_inner_diameter_mm']/2
    # Local placement of the same document; let the production tube adapter
    # derive the physical aperture slots, including mechanical pose.
    local_parts = [SimpleNamespace(key=p['key'], data=p, module_key=module['key'],
                                  center_z_mm=p['local_center_z_mm']) for p in rows.values()]
    assembly = SimpleNamespace(parts=local_parts, modules=[SimpleNamespace(
        key=module['key'], geometry=geometry, parts=local_parts)])
    tube_row = decorate_physical_tube_row(dict(key=CONTINUOUS_TUBE_KEY,
        local_start_z_mm=profile[0][0], local_end_z_mm=profile[-1][0],
        mechanical_outer_diameter_mm=geometry['c1_c2_objective_vacuum_tube_outer_diameter_mm']), assembly)
    def radius(z):
        for (a, x), (b, y) in zip(profile, profile[1:]):
            if a <= z <= b:
                return (x+(y-x)*(z-a)/(b-a))/2
        raise ValueError('Tube sample outside declared material profile')
    station_spans = []
    for key in KEYS:
        dimensions = support_dimensions(rows[key])
        half_length = dimensions['mechanical_coil_length_mm']/2
        station_spans.extend((key+'_'+str(index), centre-half_length, centre+half_length)
            for index, centre in enumerate(dimensions['finite_coil_centres_mm'], 1))
    # Host allocation spans include empty gaps. Audit the real tube wherever
    # there is deflector material, retaining its actual side-entry openings.
    for key, a, b in station_spans:
        a, b = max(a, profile[0][0]), min(b, profile[-1][0])
        if a >= b:
            continue
        section = [(a, radius(a))]+[(z, od/2) for z, od in profile if a < z < b]
        section += [(b, radius(b)), (b, bore), (a, bore)]
        shape = difference(revolve(section), *(slot['shape'] for slot in tube_row[SLOTS]))
        tube = SolidSpec('@scope:tube_'+key, 'body', 'nonmagnetic_housing', shape,
                         note='Actual configured continuous tube material, clipped to this audited material station.',
                         parameters=dict(source='geometry.mechanical_material_tube_outer_profile_z_od_mm'))
        module['solids'].append(asdict(tube))
    return dict(schema='temsim-material-collision-input-v1', units='mm', modules=[module],
        material_liner_scope='Saved continuous material tube profile clipped to selected material station spans',
        assemblies=[dict(key='ordinary_deflectors_with_neighbours',
            placements=[dict(module_key=module['key'], offset_mm=[0., 0., 0.])],
            clearance_focus_keys=list(KEYS),
            material_liner_scope='Saved tube profile; no tube inferred outside its declared extent')])


def preview(catalog, before, document, output):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel
    from PySide6.QtGui import QFont, QFontDatabase
    from temsim.cad_geometry import SolidSpec
    from temsim.cad_mesh import meshes_from_solids
    from temsim.gui.part_model_view import PartModelView
    from temsim.part_model_3d import part_model_from_document
    app = QApplication.instance() or QApplication([])
    font = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/segoeui.ttf'
    if font.exists():
        families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(font)))
        if families:
            app.setFont(QFont(families[0], 10))
    names = {f.name for f in fields(SolidSpec)}
    records = []
    current = {k: part_model_from_document(document, k, angular_segments=96).meshes for k in KEYS}
    def old_meshes(key):
        return meshes_from_solids([SolidSpec(**{k: v for k, v in s.items() if k in names})
            for s in before['modules'][0]['solids'] if s['key'] == key], angular_segments=96)
    def window(title):
        w = QWidget()
        w.setStyleSheet('QWidget {background:#101823;color:#e5edf7;} QLabel {font-size:15px;}')
        layout = QVBoxLayout(w); layout.addWidget(QLabel(title))
        grid = QGridLayout(); layout.addLayout(grid)
        return w, grid
    def panel(grid, row, col, title, meshes, fit=None, *, section=True, view_kind='isometric'):
        wrapper = QWidget(); layout = QVBoxLayout(wrapper); layout.addWidget(QLabel(title))
        view = PartModelView(); view.setMinimumSize(580, 310)
        view.set_meshes(fit or meshes)
        getattr(view, 'set_'+view_kind+'_view')(); view.set_section_enabled(section)
        view.fit_all(); view.set_meshes(meshes, preserve_view=True)
        layout.addWidget(view); grid.addWidget(wrapper, row, col)
        records.append(dict(title=title, camera_rotation=view.camera_rotation.tolist(),
            section_y_positive=section, view_kind=view_kind, solids=[m.key+'/'+m.region for m in meshes]))
    w, grid = window('Ordinary TEM deflectors | copper retained | insulating carrier, spacer, casing and registration pins')
    for index, key in enumerate(KEYS):
        if before:
            panel(grid, index, 0, key+' | before', old_meshes(key), current[key])
        panel(grid, index, 1 if before else 0, key+' | after (section)', current[key])
    w.resize(1380 if before else 720, 1550); w.show(); app.processEvents()
    if not w.grab().save(str(output/'ordinary-deflector-comparison.png')):
        raise RuntimeError('Cannot save native comparison')
    w.close()
    station = tuple(m for m in current['ac_deflector'] if m.region.startswith('coil_1_'))
    w, grid = window('AC upper station | actual component meshes | selected materials hidden for inspection')
    panel(grid, 0, 0, 'Complete station (section)', station)
    panel(grid, 0, 1, 'X carrier and original copper', tuple(m for m in station if '_x_' in m.region), station)
    panel(grid, 1, 0, 'Y carrier and original copper', tuple(m for m in station if '_y_' in m.region), station)
    panel(grid, 1, 1, 'Spacer and two insulating registration pins (axial view)',
          tuple(m for m in station if 'spacer' in m.region or 'pin_' in m.region), station,
          section=False, view_kind='axial')
    w.resize(1380, 1000); w.show(); app.processEvents()
    if not w.grab().save(str(output/'ordinary-deflector-details.png')):
        raise RuntimeError('Cannot save native detail preview')
    w.close()
    return dict(platform=app.platformName(), desktop_automation=False, panels=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-catalog', type=Path)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    from temsim.part_model_document import PartModelDocument
    from temsim.cad_catalog import module_catalog
    from export_probe_corrector_model import compare
    source = root/'configs/instruments/column/C3_ProbeCorrector.toml'
    document = PartModelDocument(source).document
    module = module_catalog(document, source=str(source.relative_to(root)))
    catalog = selected_catalog(module)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    write_json(output/'catalog.json', catalog)
    write_json(output/'collision-input.json', collision_input(module, document))
    before = json.loads(args.baseline_catalog.read_text(encoding='utf8')) if args.baseline_catalog else None
    report = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  solids=len(catalog['modules'][0]['solids']), units=len(KEYS))
    if before:
        report['comparison'] = compare(before, catalog)
    if args.preview:
        report['preview'] = preview(catalog, before, document, output)
    write_json(output/'comparison.json', report)
    print(json.dumps(dict(output=str(output), solids=report['solids'], units=len(KEYS))))


if __name__ == '__main__':
    main()
