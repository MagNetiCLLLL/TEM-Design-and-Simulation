"""Real analytic-solid collision fixtures for the optional isolated CAD env."""
import importlib.util
import math
from pathlib import Path
import unittest


definition = importlib.util.spec_from_file_location('cad_assembly_validation_test',
    Path(__file__).resolve().parents[1] / 'src/temsim/cad_assembly_validation.py')
audit = importlib.util.module_from_spec(definition)
definition.loader.exec_module(audit)


def solid(key, x=0., *, size=1., status='configured_geometry'):
    return dict(key=key, region='body', parent_key='same_parent', material='steel', status=status,
                shape=dict(kind='box', center_mm=[x, 0., 0.], size_mm=[size]*3))


def assembly(offset=(0., 0., 0.)):
    return dict(key='fixture', placements=[dict(module_key='module', offset_mm=list(offset))])


@unittest.skipUnless(importlib.util.find_spec('cadquery'), 'Optional CadQuery CAD environment is not installed')
class MaterialCollisionTests(unittest.TestCase):
    def test_same_parent_never_exempts_true_overlap(self):
        engine = audit.AssemblyCollisionAuditor()
        modules = {'module': dict(key='module', solids=[solid('a'), solid('b', .75)])}
        result = engine.audit_assembly(assembly(), modules)
        self.assertEqual(result['status'], 'material_collisions_found')
        self.assertEqual(len(result['collisions']), 1)
        self.assertAlmostEqual(result['collisions'][0]['intersection_volume_mm3'], .25, places=11)
        self.assertEqual(result['minimum_material_separation_mm'], 0.)

    def test_contact_and_actual_positive_clearance_are_separate(self):
        engine = audit.AssemblyCollisionAuditor()
        modules = {'module': dict(key='module', solids=[solid('a'), solid('b', 1.), solid('c', 3.)])}
        result = engine.audit_assembly(assembly(), modules)
        self.assertFalse(result['collisions'])
        self.assertEqual(len(result['contacts']), 1)
        self.assertAlmostEqual(result['minimum_positive_material_clearance_mm'], 1., places=11)
        points = result['closest_separated_material_pair']['nearest_points_column_mm']
        self.assertAlmostEqual(math.dist(*points), 1., places=11)
        shifted = engine.audit_assembly(assembly((0., 2., 17.)), modules)
        moved = shifted['closest_separated_material_pair']['nearest_points_column_mm']
        for before, after in zip(points, moved):
            self.assertAlmostEqual(after[1] - before[1], 2.)
            self.assertAlmostEqual(after[2] - before[2], 17.)

    def test_overlapping_bounds_do_not_imply_material_intersection(self):
        ring = dict(key='tube', region='wall', material='steel', status='configured_geometry',
                    shape=dict(kind='revolve', section_zr_mm=[[-1., 2.], [-1., 3.], [1., 3.], [1., 2.]]))
        engine = audit.AssemblyCollisionAuditor()
        result = engine.audit_assembly(assembly(), {'module': dict(key='module', solids=[ring, solid('inside', size=2.)])})
        self.assertFalse(result['collisions'])
        self.assertFalse(result['contacts'])
        self.assertAlmostEqual(result['minimum_positive_material_clearance_mm'], 2-math.sqrt(2), places=9)

    def test_unknown_envelope_overlap_cannot_certify_a_material_collision(self):
        engine = audit.AssemblyCollisionAuditor()
        modules = {'module': dict(key='module', solids=[solid('a'), solid('unknown', .75, status='configured_envelope')])}
        result = engine.audit_assembly(assembly(), modules)
        self.assertFalse(result['collisions'])
        self.assertEqual(len(result['uncertain_occupancy_intersections']), 1)
        self.assertIn('incomplete_material_coverage', result['status'])

    def test_cache_is_translation_invariant_but_reports_current_global_location(self):
        engine = audit.AssemblyCollisionAuditor()
        modules = {'module': dict(key='module', solids=[solid('a'), solid('b', .75)])}
        first = engine.audit_assembly(assembly(), modules)
        second = engine.audit_assembly(assembly((0., 0., 17.)), modules)
        self.assertEqual(engine.statistics['pair_evaluations'], 1)
        self.assertEqual(engine.statistics['pair_cache_hits'], 1)
        before = first['collisions'][0]['intersection_bounds_column_mm']
        after = second['collisions'][0]['intersection_bounds_column_mm']
        self.assertAlmostEqual(after[2]-before[2], 17.)
        self.assertAlmostEqual(after[5]-before[5], 17.)

    def test_resolved_module_offset_changes_collision_result(self):
        engine = audit.AssemblyCollisionAuditor()
        modules = {'a': dict(key='a', solids=[solid('one')]), 'b': dict(key='b', solids=[solid('two')])}
        item = dict(key='fixture', placements=[dict(module_key='a'), dict(module_key='b', offset_mm=[2., 0., 0.])])
        result = engine.audit_assembly(item, modules)
        self.assertFalse(result['collisions'])
        self.assertAlmostEqual(result['minimum_positive_material_clearance_mm'], 1.)
        item['placements'][1]['offset_mm'][0] = .5
        result = engine.audit_assembly(item, modules)
        self.assertEqual(len(result['collisions']), 1)
        self.assertAlmostEqual(result['collisions'][0]['intersection_volume_mm3'], .5)

    def test_split_liner_instances_keep_owner_without_exemption(self):
        engine = audit.AssemblyCollisionAuditor()
        first, second = solid('liner'), solid('liner', .5)
        first['instance_key'], second['instance_key'] = 'liner/body/segment-0', 'liner/body/segment-1'
        result = engine.audit_assembly(assembly(), {'module': dict(key='module', solids=[first, second])})
        self.assertEqual(len(result['collisions']), 1)
        pair = result['collisions'][0]
        self.assertEqual(pair['first']['key'], pair['second']['key'])
        self.assertNotEqual(pair['first']['id'], pair['second']['id'])

    def test_focus_clearance_is_not_pruned_by_an_unrelated_smaller_gap(self):
        engine = audit.AssemblyCollisionAuditor()
        item = assembly()
        item['clearance_focus_keys'] = ['holder']
        modules = {'module': dict(key='module', solids=[solid('a'), solid('b', 1.1),
                                                       solid('holder', 8.), solid('stage', 10.)])}
        result = engine.audit_assembly(item, modules)
        self.assertAlmostEqual(result['minimum_positive_material_clearance_mm'], .1)
        self.assertAlmostEqual(result['focus_clearance']['minimum_positive_material_clearance_mm'], 1.)
        self.assertFalse(result['focus_clearance']['collisions'])


if __name__ == '__main__':
    unittest.main()
