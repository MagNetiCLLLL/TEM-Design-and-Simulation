"""Optional exact-BRep material collision audit; no simulator/OCC import on load.

Inputs are frozen JSON SolidSpecs in module-local millimetres and resolved
module translations. Unknown envelopes are reported separately from materials.
No parent, owner, component name or overlap group exempts a material pair.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from time import perf_counter


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _backend():
    definition = importlib.util.spec_from_file_location('_temsim_cad_collision_backend', Path(__file__).with_name('cad_step.py'))
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def _vector(value):
    if len(value) != 3 or any(isinstance(x, bool) or not math.isfinite(float(x)) for x in value):
        raise ValueError('Placement requires three finite millimetre coordinates')
    return tuple(float(x) for x in value)


def _bounds(shape):
    box = shape.BoundingBox()
    return (box.xmin, box.ymin, box.zmin, box.xmax, box.ymax, box.zmax)


def _shift_bounds(bounds, shift):
    return tuple(value + shift[index % 3] for index, value in enumerate(bounds))


def bbox_distance(a, b):
    """Rigorous lower bound on the distance between two enclosed shapes."""
    return math.sqrt(sum(max(a[k] - b[k + 3], b[k] - a[k + 3], 0.) ** 2 for k in range(3)))


def occupancy_kind(spec):
    """An unknown occupied volume cannot establish a material collision."""
    if spec.get('status') == 'configured_envelope' or spec.get('parameters', {}).get('geometry_semantics') == 'envelope_only_not_material_distribution':
        return 'unresolved_envelope'
    if spec.get('status') == 'mixed_material_cartridge_unresolved':
        return 'unresolved_material_cartridge'
    return 'material'


class AssemblyCollisionAuditor:
    def __init__(self, *, linear_tolerance_mm=1e-7, volume_tolerance_mm3=1e-10,
                 relative_volume_tolerance=1e-12):
        self.linear_tolerance_mm = float(linear_tolerance_mm)
        self.volume_tolerance_mm3 = float(volume_tolerance_mm3)
        self.relative_volume_tolerance = float(relative_volume_tolerance)
        if any(not math.isfinite(x) or x <= 0 for x in (self.linear_tolerance_mm, self.volume_tolerance_mm3, self.relative_volume_tolerance)):
            raise ValueError('Collision tolerances must be finite and positive')
        self.backend = _backend()
        self.shapes = {}
        self.pairs = {}
        self.statistics = dict(shapes_built=0, pair_evaluations=0, pair_cache_hits=0,
                               boolean_intersections=0, exact_distances=0)

    def _shape(self, spec):
        key = digest(spec['shape'])
        if key not in self.shapes:
            shape = self.backend.build_shape(spec['shape'])
            metrics = self.backend._shape_metrics(shape)
            self.shapes[key] = dict(shape=shape, bounds=_bounds(shape), volume=metrics['volume_mm3'])
            self.statistics['shapes_built'] += 1
        return key, self.shapes[key]

    def instances(self, assembly, modules):
        result, identifiers = [], set()
        for placement in assembly['placements']:
            module = modules[placement['module_key']]
            offset = _vector(placement.get('offset_mm', (0., 0., 0.)))
            for spec in module['solids']:
                leaf = spec.get('instance_key', f"{spec['key']}/{spec['region']}")
                identity = f"{placement.get('instance_key', placement['module_key'])}/{leaf}"
                if identity in identifiers:
                    raise ValueError(f'Duplicate solid instance {identity}')
                identifiers.add(identity)
                geometry, cached = self._shape(spec)
                result.append(dict(id=identity, key=spec['key'], region=spec['region'], module_key=module['key'],
                    parent_key=spec.get('parent_key'), material=spec.get('material'), status=spec.get('status'),
                    occupancy=occupancy_kind(spec), geometry=geometry, offset=offset,
                    bounds=_shift_bounds(cached['bounds'], offset), cached=cached))
        return result

    def _pair(self, a, b):
        delta = tuple(y - x for x, y in zip(a['offset'], b['offset']))
        cache_key = (a['geometry'], b['geometry'], delta)
        if cache_key in self.pairs:
            self.statistics['pair_cache_hits'] += 1
            return self.pairs[cache_key]
        self.statistics['pair_evaluations'] += 1
        first = a['cached']['shape']
        second = b['cached']['shape'].translate(delta)
        lower = bbox_distance(a['cached']['bounds'], _shift_bounds(b['cached']['bounds'], delta))
        volume, box = 0., None
        if lower <= self.linear_tolerance_mm:
            common = first.intersect(second)
            self.statistics['boolean_intersections'] += 1
            if not common.isValid():
                raise RuntimeError(f'Invalid BRep intersection: {a["id"]} / {b["id"]}')
            volume = sum(s.Volume() for s in common.Solids())
            if not math.isfinite(volume) or volume < 0:
                raise RuntimeError('Nonphysical intersection volume')
            if volume:
                box = _bounds(common)
        threshold = max(self.volume_tolerance_mm3,
                        self.relative_volume_tolerance * min(a['cached']['volume'], b['cached']['volume']))
        nearest = None
        if volume > threshold:
            kind, distance = 'intersection', 0.
        else:
            from OCP.BRepExtrema import BRepExtrema_DistShapeShape
            extrema = BRepExtrema_DistShapeShape(first.wrapped, second.wrapped)
            if not extrema.IsDone() or extrema.NbSolution() < 1:
                raise RuntimeError(f'BRep distance did not converge: {a["id"]} / {b["id"]}')
            distance = extrema.Value()
            self.statistics['exact_distances'] += 1
            if not math.isfinite(distance) or distance < 0:
                raise RuntimeError('Invalid exact BRep distance')
            points = (extrema.PointOnShape1(1), extrema.PointOnShape2(1))
            nearest = [[point.X(), point.Y(), point.Z()] for point in points]
            kind = ('sub_tolerance_intersection' if volume else 'contact'
                    if distance <= self.linear_tolerance_mm else 'separated')
        result = dict(kind=kind, distance_mm=distance, intersection_volume_mm3=volume,
                      volume_tolerance_mm3=threshold, intersection_bounds_in_first_frame_mm=box,
                      nearest_points_in_first_frame_mm=nearest)
        self.pairs[cache_key] = result
        return result

    @staticmethod
    def _record(a, b, result):
        identity = lambda item: {k: item[k] for k in ('id', 'key', 'region', 'module_key', 'parent_key', 'material', 'status', 'occupancy')}
        data = {k: v for k, v in result.items() if k not in (
            'intersection_bounds_in_first_frame_mm', 'nearest_points_in_first_frame_mm')}
        box = result['intersection_bounds_in_first_frame_mm']
        data['intersection_bounds_column_mm'] = list(_shift_bounds(box, a['offset'])) if box else None
        nearest = result['nearest_points_in_first_frame_mm']
        data['nearest_points_column_mm'] = [[v + d for v, d in zip(point, a['offset'])]
                                             for point in nearest] if nearest else None
        data.update(first=identity(a), second=identity(b))
        if a['occupancy'] != 'material' or b['occupancy'] != 'material':
            data['qualification'] = 'uncertain_occupied_volume'
        else:
            data['qualification'] = 'declared_material_geometry'
        return data

    def audit_assembly(self, assembly, modules):
        started = perf_counter()
        instances = self.instances(assembly, modules)
        pairs = [(bbox_distance(a['bounds'], b['bounds']), i, j)
                 for i, a in enumerate(instances) for j, b in enumerate(instances[i + 1:], i + 1)]
        pairs.sort()
        collisions, contacts, unresolved, tiny = [], [], [], []
        closest = None
        minimum_positive = math.inf
        focus_keys = set(assembly.get('clearance_focus_keys', []))
        focus = dict(keys=sorted(focus_keys), scope='Requested parts versus other component keys; internal same-key pairs remain in the whole-assembly collision audit.',
                     minimum_positive_material_clearance_mm=None, minimum_material_separation_mm=None,
                     closest_separated_material_pair=None, collisions=[], contacts=[], sub_tolerance_intersections=[])
        focus_positive = math.inf
        candidate_count = 0
        for lower, i, j in pairs:
            a, b = instances[i], instances[j]
            known = a['occupancy'] == b['occupancy'] == 'material'
            focused = a['key'] != b['key'] and bool(focus_keys.intersection((a['key'], b['key'])))
            # Envelope proximity outside their bounding boxes says nothing
            # about missing material. For known material, use the lower bound
            # to certify the minimum strictly positive separation as well.
            if lower > self.linear_tolerance_mm and (not known or (lower > minimum_positive
                    and (not focused or lower > focus_positive))):
                continue
            candidate_count += 1
            record = self._record(a, b, self._pair(a, b))
            if known and focused:
                if record['kind'] in ('intersection', 'contact', 'sub_tolerance_intersection'):
                    name = dict(intersection='collisions', contact='contacts',
                                sub_tolerance_intersection='sub_tolerance_intersections')[record['kind']]
                    focus[name].append(record)
                elif record['distance_mm'] < focus_positive:
                    focus_positive = record['distance_mm']
                    focus['closest_separated_material_pair'] = record
            if not known:
                if record['kind'] != 'separated':
                    unresolved.append(record)
            elif record['kind'] == 'intersection':
                collisions.append(record)
            elif record['kind'] == 'contact':
                contacts.append(record)
            elif record['kind'] == 'sub_tolerance_intersection':
                tiny.append(record)
            elif record['distance_mm'] < minimum_positive:
                minimum_positive = record['distance_mm']
                closest = record
        unknown = [{k: row[k] for k in ('id', 'key', 'region', 'occupancy', 'status')}
                   for row in instances if row['occupancy'] != 'material']
        status = ('material_collisions_found' if collisions else
                  'no_resolved_collision_incomplete_material_coverage' if unknown or tiny else
                  'no_material_collision_above_tolerance')
        if math.isfinite(focus_positive):
            focus['minimum_positive_material_clearance_mm'] = focus_positive
            focus['minimum_material_separation_mm'] = focus_positive
        if focus['collisions'] or focus['contacts'] or focus['sub_tolerance_intersections']:
            focus['minimum_material_separation_mm'] = 0.
        return dict(key=assembly['key'], selection=assembly.get('selection'), status=status,
                    material_liner_scope=assembly.get('material_liner_scope', 'not_declared'),
                    liner_material_decisions=assembly.get('liner_material_decisions', []),
                    focus_clearance=focus if focus_keys else None,
                    solid_instances=len(instances), potential_pairs=len(pairs), evaluated_candidates=candidate_count,
                    collisions=collisions, contacts=contacts, sub_tolerance_intersections=tiny,
                    uncertain_occupancy_intersections=unresolved, unresolved_solids=unknown,
                    closest_separated_material_pair=closest,
                    minimum_material_separation_mm=0. if collisions or contacts or tiny else (minimum_positive if closest else None),
                    minimum_positive_material_clearance_mm=minimum_positive if closest else None,
                    elapsed_s=perf_counter() - started)


def audit_document(document, *, progress=None):
    if document.get('units') != 'mm':
        raise ValueError('Collision inputs must explicitly declare millimetres')
    started = perf_counter()
    auditor = AssemblyCollisionAuditor(**document.get('tolerances', {}))
    modules = {module['key']: module for module in document['modules']}
    if len(modules) != len(document['modules']):
        raise ValueError('Duplicate audit module keys')
    report = dict(schema='temsim-material-collision-audit-v1', units='mm', input_sha256=digest(document),
                  scope='Static configured material BRep overlap and clearance, not manufacturing tolerance or dynamic insertion clearance. Parent and overlap-group labels are not exemptions.',
                  cadquery_version=auditor.backend._cadquery().__version__,
                  audit_implementation_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__), Path(__file__).with_name('cad_step.py'))},
                  omitted_definitions_by_module={key: module.get('omitted', []) for key, module in modules.items()
                      if module.get('omitted')},
                  tolerances=dict(linear_tolerance_mm=auditor.linear_tolerance_mm,
                      volume_tolerance_mm3=auditor.volume_tolerance_mm3,
                      relative_volume_tolerance=auditor.relative_volume_tolerance), assemblies=[])
    for assembly in document['assemblies']:
        row = auditor.audit_assembly(assembly, modules)
        report['assemblies'].append(row)
        if progress:
            progress(row)
    report.update(statistics=auditor.statistics, elapsed_s=perf_counter()-started,
                  status='completed', collision_occurrences=sum(len(a['collisions']) for a in report['assemblies']),
                  assemblies_with_collisions=sum(bool(a['collisions']) for a in report['assemblies']))
    report['uncertain_occupancy_volume_intersection_occurrences'] = sum(
        row['kind'] == 'intersection' for a in report['assemblies'] for row in a['uncertain_occupancy_intersections'])
    report['material_geometry_conclusion'] = ('resolved_material_collisions_found' if report['collision_occurrences'] else
        'no_resolved_overlap_but_incomplete_material_coverage' if any(a['unresolved_solids'] or a['sub_tolerance_intersections']
            for a in report['assemblies']) else 'no_resolved_material_overlap_above_tolerance')
    return report
