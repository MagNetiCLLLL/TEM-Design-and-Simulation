"""Exact-CAD regressions executed in the optional isolated CAD environment."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cad_kernel_under_test", ROOT / "src/temsim/cad_kernel.py")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


def box(center=(0, 0, 0), size=(4, 4, 4)):
    return dict(kind="box", center_mm=list(center), size_mm=list(size))


@unittest.skipUnless(importlib.util.find_spec("cadquery"), "Optional CadQuery CAD environment is not installed")
class ExactCadKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = kernel._module("cad_step")
        cls.cq = cls.backend._cadquery()

    def test_brep_integrity_and_exact_primitives(self):
        b = self.backend
        ellipse = dict(kind="elliptic_cylinder", width_mm=4., height_mm=2., length_mm=3., center_mm=[0., 0., 5.])
        shape = b.build_shape(ellipse)
        self.assertAlmostEqual(shape.Volume(), 6*math.pi, places=9)
        source = b.shape_brep_spec(shape)
        self.assertAlmostEqual(b.build_shape(source).Volume(), shape.Volume(), places=10)
        source["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            b.build_shape(source)
        transformed = b.build_shape(dict(kind="affine", shape=ellipse,
            matrix=[[0., -3., 0.], [2., 0., 0.], [0., 0., 1.]], offset_mm=[10., 20., 30.]))
        self.assertAlmostEqual(b._shape_metrics(transformed)["volume_mm3"], 36*math.pi, places=7)

    def test_segment_instances_keep_material_in_catalog_and_clearance_probes(self):
        segments = [dict(key="liner", region="body", instance_key=f"liner/body/segment-{i}",
                         status="configured_geometry", material="steel",
                         shape=box(center=(0, 0, z), size=(2, 2, 2)))
                    for i, z in enumerate((0, 4))]
        block = dict(key="block", region="body", status="configured_geometry", material="steel",
                     shape=box(center=(0, 0, 4.5), size=(1, 1, 1)))
        probe = dict(id="passage", kind="clearance", shape=box(size=(.5, .5, 12)))
        result = kernel.validate_catalog(dict(key="fixture", solids=[*segments, block]), optical_probes=[probe])
        pair = result["geometry_report"]["assemblies"][0]["collisions"]
        self.assertEqual(len(pair), 1)
        self.assertAlmostEqual(pair[0]["intersection_volume_mm3"], 1.)
        blocked = result["probe_results"][0]["material_blockers"]
        self.assertEqual({row["solid_id"] for row in blocked},
                         {"liner/body/segment-0", "liner/body/segment-1", "block/body"})
        covered = kernel.validate_catalog(dict(key="fixture", solids=segments), optical_probes=[
            dict(id="surface", kind="coverage", expected_solid_ids=["liner/body"],
                 shape=dict(kind="union", shapes=[row["shape"] for row in segments]))])
        self.assertEqual(covered["probe_results"][0]["status"], "passed")
        evaluated = kernel.evaluate(segments, include_mesh=False)
        self.assertEqual({row["id"] for row in evaluated["leaves"]},
                         {row["instance_key"] for row in segments})

    def test_duplicate_explicit_solid_instance_is_rejected(self):
        first = dict(key="liner", region="body", instance_key="segment-0", shape=box())
        with self.assertRaisesRegex(ValueError, "Duplicate solid ID"):
            self.backend.build_solid_rows([first, dict(first, key="different_owner")])

    def test_export_preserves_segment_instances_with_the_same_component_region(self):
        segments = [dict(key="liner", region="body", instance_key=f"segment-{i}",
                         shape=box(center=(0, 0, 4*i), size=(1, 2, 3))) for i in range(2)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "segments.step"
            kernel.export_step(segments, path)
            restored = kernel.import_step(path)
        self.assertEqual(len(restored["leaves"]), 2)
        self.assertAlmostEqual(restored["metrics"]["volume_mm3"], 12.)

    def test_hole_slot_and_strict_edit_noop_or_empty(self):
        b = self.backend
        hole = dict(kind="cylinder", radius_mm=.5, length_mm=8., center_mm=[0., 0., 0.])
        cut = b.build_shape(dict(kind="strict_difference", base=box(), tools=[hole]))
        self.assertAlmostEqual(cut.Volume(), 64.-math.pi, places=9)
        slot = dict(kind="union", shapes=[box(size=[1., 1., 8.]),
            dict(hole, center_mm=[-.5, 0., 0.]), dict(hole, center_mm=[.5, 0., 0.])])
        cut = b.build_shape(dict(kind="strict_difference", base=box(), tools=[slot]))
        self.assertAlmostEqual(cut.Volume(), 64.-4.*(1.+math.pi*.25), places=8)
        for tools in ([box(center=[10, 0, 0])], [box(size=[8, 8, 8])]):
            with self.assertRaises(ValueError):
                b.build_shape(dict(kind="strict_difference", base=box(), tools=tools))

    def test_named_colored_step_instances_units_and_source_immutable(self):
        b, cq = self.backend, self.cq
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "source.step"
            child = cq.Assembly(name="holder")
            child.add(b.build_shape(box(size=[1, 2, 3])), name="copper", color=cq.Color(.8, .4, .2, 1.))
            assembly = cq.Assembly(name="instrument")
            assembly.add(child, loc=cq.Location((10, 20, 30)))
            b.export_assembly(assembly, source, {"copper": "copper"})
            raw = source.read_bytes()
            asset = kernel.import_step(source)
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(asset["source"]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(asset["units"], "mm")
            self.assertEqual(len(asset["leaves"]), 1)
            leaf = asset["leaves"][0]
            self.assertEqual(leaf["path"], "instrument/holder/copper")
            self.assertIn("material:copper", leaf["material_layers"])
            for actual, expected in zip(leaf["color_rgba"], [.8, .4, .2, 1.]):
                self.assertAlmostEqual(actual, expected, places=6)
            for actual, expected in zip(asset["metrics"]["bounds_mm"], [9.5, 19, 28.5, 10.5, 21, 31.5]):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(len(leaf["mesh"]["faces"]), len(leaf["mesh"]["face_ids"]))
            # Oriented triangles have the same signed volume as the exact BRep.
            points = leaf["mesh"]["vertices_mm"]
            signed = 0.
            for i, j, k in leaf["mesh"]["faces"]:
                a, c, d = points[i], points[j], points[k]
                signed += sum(a[n] * (c[(n+1)%3]*d[(n+2)%3]-c[(n+2)%3]*d[(n+1)%3]) for n in range(3)) / 6.
            self.assertAlmostEqual(signed, 6., places=7)
            exported = kernel.export_step(asset["leaves"], folder / "edited.step")
            self.assertAlmostEqual(exported["volume_mm3"], 6., places=8)
            self.assertEqual(exported["source_names"][0]["source_path"], leaf["path"])
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                kernel.export_step(asset["leaves"], source)

    def test_inch_step_is_converted_by_reader_to_mm(self):
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.Interface import Interface_Static
        from OCP.IFSelect import IFSelect_RetDone
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inch.step"
            writer = STEPControl_Writer()
            Interface_Static.SetCVal_s("write.step.unit", "INCH")
            Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
            # This low-level writer emits numeric model coordinates in inches.
            self.assertEqual(writer.Transfer(self.backend.build_shape(box(size=[1., 2., 3.])).wrapped,
                                             STEPControl_AsIs), IFSelect_RetDone)
            self.assertEqual(writer.Write(str(path)), IFSelect_RetDone)
            try:
                self.assertIn("CONVERSION_BASED_UNIT", path.read_text())
                asset = kernel.import_step(path)
                self.assertAlmostEqual(asset["metrics"]["volume_mm3"], 25.4*50.8*76.2, places=5)
                self.assertAlmostEqual(asset["metrics"]["bounds_mm"][3], 12.7, places=8)
            finally:
                Interface_Static.SetCVal_s("write.step.unit", "MM")

    def test_exact_overlap_contact_and_gap(self):
        for offset, relation, volume, gap in [(1., "overlap", 48., 0.), (4., "contact", 0., 0.), (5., "separate", 0., 1.)]:
            result = kernel.collision(dict(id="a", shape=box()), dict(id="b", shape=box(center=[offset, 0, 0])))
            self.assertEqual(result["relation"], relation)
            self.assertAlmostEqual(result["intersection_volume_mm3"], volume, places=8)
            self.assertAlmostEqual(result["distance_mm"], gap, places=8)

    def test_whole_part_edit_allows_removal_of_one_leaf_and_rejects_noop(self):
        tool = box(center=[0, 0, 0], size=[3, 3, 3])
        rows = []
        for region, original in [("first", box(size=[2, 2, 2])), ("second", box(center=[5, 0, 0], size=[2, 2, 2]))]:
            rows.append(dict(key="part", region=region, material="copper", color=[.8, .4, .2, 1.],
                shape=dict(kind="difference", base=original, tools=[tool]),
                parameters=dict(editor_tools=[tool], editor_original_shape=original, editor_tools_global=[tool])))
        result = kernel.evaluate(rows)
        self.assertEqual(result["edit_report"]["removed_leaf_ids"], ["part/first"])
        self.assertEqual(len(result["leaves"]), 1)
        self.assertAlmostEqual(result["metrics"]["volume_mm3"], 8.)
        with tempfile.TemporaryDirectory() as folder:
            report = self.backend.export_catalog(dict(units="mm", modules=[dict(key="m", solids=rows,
                units=[dict(key="part", solid_ids=["part/first", "part/second"])])]), folder)
            self.assertEqual(report["modules"][0]["solid_count"], 1)
        for bad in [box(center=[20, 0, 0]), box(size=[50, 50, 50])]:
            broken = copy.deepcopy(rows)
            for row in broken:
                row["shape"]["tools"] = [bad]
                row["parameters"].update(editor_tools=[bad], editor_tools_global=[bad])
            with self.assertRaises(ValueError):
                kernel.evaluate(broken)

    def test_catalog_probes_do_not_exempt_same_parent_or_unknown_geometry(self):
        rows = [dict(key="sensor", region="active", material="silicon", shape=box(), parent_key="p"),
                dict(key="obstruction", region="body", material="steel", shape=box(center=[1, 0, 0]), parent_key="p"),
                dict(key="unknown", region="envelope", material="unspecified", status="configured_envelope",
                     shape=box(center=[1, 0, 0]))]
        catalog = dict(units="mm", modules=[dict(key="m", solids=rows)])
        probes = [dict(id="clear", module_key="m", kind="clearance", shape=box(size=[1, 1, 1])),
                  dict(id="cover", module_key="m", kind="coverage", shape=box(size=[1, 1, 1]),
                       expected_solid_ids=["sensor/active"])]
        report = kernel.validate_catalog(catalog, optical_probes=probes, binding_report_sha256="abc")
        self.assertEqual(report["geometry_report"]["collision_occurrences"], 1)
        self.assertTrue(report["probe_results"][0]["material_blockers"])
        self.assertTrue(report["probe_results"][1]["unresolved_intersections"])
        self.assertAlmostEqual(report["probe_results"][1]["uncovered_volume_mm3"], 0.)
        self.assertEqual(report["report_sha256"], kernel.digest({k: v for k, v in report.items() if k != "report_sha256"}))

    def test_worker_protocol_digest_unicode_failure_and_no_simulator_import(self):
        self.assertNotIn("temsim", sys.modules)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            request = dict(operation="evaluate", request_id="test-φ", draft_digest="draft1", include_mesh=False,
                           solids=[dict(shape=box())])
            request_path = folder / "request.json"
            request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/cad_geometry_worker.py"), str(request_path), "--output", str(folder)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((folder / "result.json").read_text())
            self.assertEqual(result["request_sha256"], kernel.digest(request))
            self.assertEqual(result["request_id"], "test-φ")
            self.assertEqual(result["status"], "complete")
            request["operation"] = "unrecognized"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 1)
            result = json.loads((folder / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("result", result)
            self.assertFalse(list(folder.glob("*.pending")))

    def test_assembly_probe_rejects_ambiguous_owner_and_limits_coverage_to_that_instance(self):
        row = dict(key="sensor", region="active", material="silicon", shape=box())
        catalog = dict(units="mm", modules=[dict(key="module", solids=[row])], assemblies=[dict(key="installed",
            placements=[dict(module_key="module", instance_key="left", offset_mm=[0, 0, 20]),
                        dict(module_key="module", instance_key="right", offset_mm=[0, 0, 40])])])
        probe = dict(id="coverage", module_key="module", assembly_key="installed", kind="coverage",
                     shape=box(size=[1, 1, 1]), expected_solid_ids=["sensor/active"])
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            kernel.validate_catalog(catalog, optical_probes=[probe])
        probe["owner_instance_key"] = "left"
        result = kernel.validate_catalog(catalog, optical_probes=[probe])["probe_results"][0]
        self.assertEqual(result["module_offset_mm"], [0., 0., 20.])
        self.assertEqual(result["coordinate_frame"], "global_column_mm")
        self.assertEqual(result["assembly_key"], "installed")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["material_blockers"], [])


if __name__ == "__main__":
    unittest.main()
