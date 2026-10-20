"""Optional isolated-CAD tests; no simulator or scientific dependency import."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

definition = importlib.util.spec_from_file_location("temsim_cad_step_test",
    Path(__file__).resolve().parents[1]/"src"/"temsim"/"cad_step.py")
cad = importlib.util.module_from_spec(definition)
definition.loader.exec_module(cad)


def box(centre, size):
    return dict(kind="box", center_mm=centre, size_mm=size)


@unittest.skipUnless(importlib.util.find_spec("cadquery"), "Optional CadQuery CAD environment is not installed")
class AnalyticStepTests(unittest.TestCase):
    def test_primitives_have_analytic_volume_and_rigid_pose(self):
        annulus = cad.build_shape(dict(kind="revolve", section_zr_mm=[[0, 1], [0, 2], [3, 2], [3, 1]]))
        self.assertAlmostEqual(annulus.Volume(), 9*math.pi, places=11)
        quarter = cad.build_shape(dict(kind="annular_sector", start_mm=0, end_mm=3,
            inner_mm=1, outer_mm=2, angle_deg=45, span_deg=90))
        self.assertAlmostEqual(quarter.Volume(), annulus.Volume()/4, places=11)
        self.assertGreater(quarter.Center().x, 0)
        self.assertGreater(quarter.Center().y, 0)
        cut = cad.build_shape(dict(kind="difference", base=box([0, 0, 0], [4, 4, 4]),
                                  tools=[box([0, 0, 0], [2, 2, 6])]))
        self.assertAlmostEqual(cut.Volume(), 48., places=11)
        union = cad.build_shape(dict(kind="union", shapes=[box([0, 0, 0], [1, 1, 1]), box([3, 0, 0], [1, 1, 1])]))
        self.assertEqual(len(union.Solids()), 2)
        self.assertAlmostEqual(union.Volume(), 2., places=11)
        moved = cad.build_shape(dict(kind="transform", shape=box([2, 0, 0], [1, 2, 3]),
            rotation=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], offset_mm=[10, 20, 30]))
        self.assertAlmostEqual(moved.Center().x, 10.)
        self.assertAlmostEqual(moved.Center().y, 22.)
        self.assertAlmostEqual(moved.Center().z, 30.)
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            cad.build_shape(dict(kind="transform", shape=box([0, 0, 0], [1, 1, 1]),
                rotation=[[2, 0, 0], [0, 1, 0], [0, 0, 1]], offset_mm=[0, 0, 0]))

    def test_round_tip_retains_spherical_cap_and_tangent_cone(self):
        r, length, angle = 1e-4, .05, math.radians(10)
        shape = cad.build_shape(dict(kind="round_tip", radius_mm=r, length_mm=length,
            cone_half_angle_deg=10., apex_z_mm=1.))
        h = r*(1-math.sin(angle))
        base = r*math.cos(angle)+(length-h)*math.tan(angle)
        join = r*math.cos(angle)
        expected = math.pi*h*h*(r-h/3)+math.pi*(length-h)*(join*join+join*base+base*base)/3
        self.assertAlmostEqual(shape.Volume()/expected, 1., places=9)
        self.assertIn("SPHERE", [face.geomType() for face in shape.Faces()])
        self.assertIn("CONE", [face.geomType() for face in shape.Faces()])

    def test_step_roundtrip_preserves_hierarchy_material_layers_colors_and_solids(self):
        catalog = dict(units="mm", modules=[dict(key="test_module", name="Test module", solids=[
            dict(key="lens", region="yoke", material="soft_magnetic", color=[.3, .4, .6, 1.],
                shape=dict(kind="revolve", section_zr_mm=[[0, 1], [0, 2], [3, 2], [3, 1]])),
            dict(key="lens", region="coil", material="copper_winding", color=[.8, .4, .2, 1.],
                shape=dict(kind="transform", shape=dict(kind="difference", base=box([4, 0, 1], [1, 2, 3]),
                    tools=[box([4, 0, 1], [.5, 1, 4])]),
                    rotation=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], offset_mm=[10, 20, 30]))],
            units=[dict(key="lens", name="Lens", solid_ids=["lens/yoke", "lens/coil"])])])
        with tempfile.TemporaryDirectory() as path:
            report = cad.export_catalog(catalog, path)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["exports"]), 2)
            self.assertEqual(report["exports"][-1]["leaf_count"], 2)
            self.assertEqual({row["material"] for row in report["exports"][-1]["records"]},
                             {"soft_magnetic", "copper_winding"})
            self.assertEqual({row["path"] for row in report["exports"][-1]["records"]},
                             {"test_module/lens/lens__yoke", "test_module/lens/lens__coil"})
            coil = next(row for row in report["exports"][-1]["records"] if row["material"] == "copper_winding")
            for expected, actual in zip([9, 23.5, 29.5, 11, 24.5, 32.5], coil["bounds_mm"]):
                self.assertAlmostEqual(actual, expected, places=5)
            self.assertEqual(json.loads((Path(path)/"step-manifest.json").read_text())["status"], "passed")

    def test_disconnected_insulation_region_keeps_its_colour_and_material(self):
        # Conductor-layer insulation has disconnected windows and spaces
        # between windings, all belonging to the same named material region.
        shape = dict(kind="difference", base=box([0, 0, 0], [8, 8, 2]), tools=[
            box([0, 0, 0], [2, 10, 4]), box([0, 0, 0], [10, 2, 4])])
        color = [.87, .83, .68, .75]
        catalog = dict(units="mm", modules=[dict(key="laminate", name="Laminate",
            solids=[dict(key="insert", region="insulation", material="insulator",
                         color=color, shape=shape)],
            units=[dict(key="insert", name="Insert", solid_ids=["insert/insulation"])])])
        with tempfile.TemporaryDirectory() as path:
            report = cad.export_catalog(catalog, path)
            self.assertEqual(report["status"], "passed")
            for export in report["exports"]:
                record, = export["records"]
                self.assertEqual(record["solid_count"], 4)
                self.assertEqual(record["material"], "insulator")
                self.assertAlmostEqual(record["volume_mm3"], 72., places=9)
                for expected, actual in zip(color, record["color_rgba"]):
                    self.assertAlmostEqual(actual, expected, places=6)

    def test_compound_readback_does_not_hide_mixed_or_missing_solid_colours(self):
        cq = cad._cadquery()
        shape = cad.build_shape(dict(kind="union", shapes=[
            box([0, 0, 0], [1, 1, 1]), box([3, 0, 0], [1, 1, 1])]))
        for second in (None, cq.Color(.2, .4, .6)):
            with self.subTest(second=second):
                assembly = cq.Assembly(name="insert")
                assembly.add(shape, name="gap")
                first, other = shape.Solids()
                assembly["gap"].addSubshape(first, color=cq.Color(.87, .83, .68))
                if second is not None:
                    assembly["gap"].addSubshape(other, color=second)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"mixed.step"
                    cad._write_step(assembly, path, {"gap": "insulator"})
                    restored, _ = cad._read_step(path)
                    self.assertIsNone(restored["gap"].color)

    def test_compound_default_colour_preserves_explicit_solid_override(self):
        cq = cad._cadquery()
        shape = cad.build_shape(dict(kind="union", shapes=[
            box([0, 0, 0], [1, 1, 1]), box([3, 0, 0], [1, 1, 1])]))
        assembly = cq.Assembly(name="insert")
        assembly.add(shape, name="gap", color=cq.Color(1., 0., 0.))
        first, second = shape.Solids()
        assembly["gap"].addSubshape(second, color=cq.Color(0., 0., 1.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"override.step"
            cad._write_step(assembly, path, {"gap": "insulator"})
            restored, layers = cad._read_step(path)
            leaf = restored["gap"]
            self.assertIsNone(leaf.color)
            actual = sorted((solid.Center().x, color.toTuple())
                            for solid, color in leaf._subshape_colors.items())
            self.assertEqual(len(actual), 2)
            for (position, color), expected_position, expected_color in zip(
                    actual, (0., 3.), ((1., 0., 0., 1.), (0., 0., 1., 1.))):
                self.assertAlmostEqual(position, expected_position, places=9)
                for value, expected in zip(color, expected_color):
                    self.assertAlmostEqual(value, expected, places=6)
            self.assertIn("material:insulator", layers["gap"])
            # The uniform-leaf verification contract must reject a mixed leaf,
            # rather than reporting success after erasing its explicit override.
            with self.assertRaisesRegex(ValueError, "RGBA colour"):
                cad.export_assembly(assembly, Path(directory)/"verified.step", {"gap": "insulator"})


if __name__ == "__main__":
    unittest.main()
