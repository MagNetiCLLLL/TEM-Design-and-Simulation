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


if __name__ == "__main__":
    unittest.main()
