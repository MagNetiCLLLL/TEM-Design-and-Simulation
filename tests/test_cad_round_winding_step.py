"""Exact circular turns and insulation round-trip in the isolated CAD runtime."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


definition = importlib.util.spec_from_file_location("temsim_winding_cad_step_test",
    Path(__file__).resolve().parents[1]/"src"/"temsim"/"cad_step.py")
cad = importlib.util.module_from_spec(definition)
definition.loader.exec_module(cad)


def wire(radius=3., z=1.):
    return dict(kind="torus", major_radius_mm=radius, minor_radius_mm=.2, center_z_mm=z)


def insulation():
    return dict(kind="round_winding_insulation", start_mm=0., end_mm=4., inner_mm=2., outer_mm=5.,
                wire_radius_mm=.2, turns_rz_mm=[[3., 1.], [4., 3.]])


@unittest.skipUnless(importlib.util.find_spec("cadquery"), "Optional CadQuery CAD environment is not installed")
class RoundWindingStepTests(unittest.TestCase):
    def test_exact_tori_and_insulation_partition_have_analytic_volume(self):
        a, b = cad.build_shape(wire()), cad.build_shape(wire(4., 3.))
        remaining = cad.build_shape(insulation())
        self.assertTrue(all(shape.isValid() for shape in (a, b, remaining)))
        self.assertEqual([face.geomType() for face in a.Faces()], ["TORUS"])
        self.assertAlmostEqual(cad.shape_volume(a), 2*math.pi**2*3.*.2**2, places=10)
        total = sum(cad.shape_volume(shape) for shape in (a, b, remaining))
        self.assertAlmostEqual(total, math.pi*(5**2-2**2)*4., places=8)
        self.assertLess(cad.shape_volume(a.intersect(remaining)), 1e-10)
        self.assertLess(cad.shape_volume(b.intersect(remaining)), 1e-10)

    def test_step_roundtrip_retains_every_wire_leaf_and_insulation_material(self):
        solids = [dict(key="coil", region=f"turn_{i}", material="copper_winding", shape=shape)
                  for i, shape in enumerate((wire(), wire(4., 3.)))]
        solids.append(dict(key="coil", region="insulation", material="insulator", shape=insulation()))
        catalog = dict(units="mm", modules=[dict(key="test_winding", name="Test winding", solids=solids,
            units=[dict(key="coil", name="Coil", solid_ids=["coil/turn_0", "coil/turn_1", "coil/insulation"])])])
        with tempfile.TemporaryDirectory() as directory:
            report = cad.export_catalog(catalog, directory)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["exports"][-1]["leaf_count"], 3)
        records = report["exports"][-1]["records"]
        self.assertEqual(sum(item["material"] == "copper_winding" for item in records), 2)
        self.assertEqual(sum(item["material"] == "insulator" for item in records), 1)
        self.assertAlmostEqual(sum(item["volume_mm3"] for item in records), math.pi*84., places=6)

    def test_insulation_rejects_a_wire_outside_the_original_pack(self):
        spec = insulation()
        spec["turns_rz_mm"][0][0] = 2.
        with self.assertRaisesRegex(ValueError, "outside"):
            cad.build_shape(spec)

    def test_boundary_tangent_and_touching_turns_keep_valid_analytic_insulation(self):
        spec = insulation()
        spec["turns_rz_mm"] = [[3., .2], [3., .6], [4., 3.8+2e-15]]
        shape = cad.build_shape(spec)
        self.assertTrue(shape.isValid())
        expected = math.pi*84.-2*math.pi**2*.2**2*(3.+3.+4.)
        self.assertAlmostEqual(cad.shape_volume(shape), expected, places=8)

    def test_fully_occupied_orthogonal_pack_keeps_disconnected_insulation_solids(self):
        spec = dict(kind="round_winding_insulation", start_mm=0., end_mm=2., inner_mm=2., outer_mm=4.,
                    wire_radius_mm=.5, turns_rz_mm=[[2.5, .5], [3.5, .5], [2.5, 1.5], [3.5, 1.5]])
        shape = cad.build_shape(spec)
        self.assertTrue(shape.isValid())
        self.assertEqual(len(shape.Solids()), 9)
        expected = math.pi*(4**2-2**2)*2.-2*math.pi**2*.5**2*(2.5+3.5+2.5+3.5)
        self.assertAlmostEqual(cad.shape_volume(shape), expected, places=8)

    def test_step_resource_guard_runs_before_any_brep_construction(self):
        spec = insulation()
        spec["turns_rz_mm"] = [[3., 1.]]*8193
        from unittest.mock import patch
        with patch.object(cad, "build_shape", side_effect=AssertionError("Must reject before BRep work")):
            with self.assertRaisesRegex(ValueError, "No turns were omitted"):
                cad.build_solid_rows([dict(key="coil", region="insulation", shape=spec)])


if __name__ == "__main__":
    unittest.main()
