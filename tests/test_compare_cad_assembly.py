"""The comparison must expose changed geometry without rewriting its baseline."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


PATH=Path(__file__).resolve().parents[1]/"scripts/compare_cad_assembly.py"
SPEC=importlib.util.spec_from_file_location("cad_assembly_comparison",PATH)
comparison=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def test_geometry_override_restores_on_error(monkeypatch):
    from temsim import cad_catalog
    original=lambda *args,**kwargs: "original hook"
    monkeypatch.setattr(cad_catalog,"native_reconstruction",original)
    with pytest.raises(RuntimeError):
        with comparison.original_geometry():
            assert cad_catalog.native_reconstruction({},{}) is None
            raise RuntimeError("stop")
    assert cad_catalog.native_reconstruction is original


def test_envelope_diagnostic_exposes_all_three_material_violations():
    part=dict(key="test",local_start_z_mm=10.,local_center_z_mm=15.,local_end_z_mm=20.,
              mechanical_outer_diameter_mm=20.,mechanical_inner_diameter_mm=4.)
    measured=dict(minimum_mm=[-11.,-11.,9.],maximum_mm=[11.,11.,21.],
                  radial_vertex_min_mm=1.,radial_vertex_max_mm=11.)
    result=comparison.dimensional_checks(part,measured)
    assert set(result["negative_margins"])=={"material_bore","outer_envelope","axial_envelope"}
    assert result["checks"]["material_bore"]["margin_mm"]==-1.
    assert result["checks"]["axial_envelope"]["end_margin_mm"]==-1.


def test_reconstruction_is_reported_as_geometry_change_not_new_optical_part():
    from temsim.part_model_3d import revolve_section
    from copy import deepcopy
    part=dict(key="test",order=2,local_start_z_mm=10.,local_center_z_mm=15.,local_end_z_mm=20.,
              mechanical_outer_diameter_mm=20.,mechanical_inner_diameter_mm=4.)
    document=dict(module=dict(key="test_module"),parts=[part])
    before=deepcopy(document)
    old=revolve_section([(10.,2.),(10.,10.),(20.,10.),(20.,2.)],key="test",angular_segments=8)
    new=revolve_section([(10.,3.),(10.,10.),(20.,10.),(20.,3.)],key="test",angular_segments=8)
    from types import SimpleNamespace
    result=comparison.compare_module(document,SimpleNamespace(meshes=(old,)),SimpleNamespace(meshes=(new,)))
    assert document==before
    row,=result["parts"]
    assert row["key"]=="test" and row["saved_order"]==2
    assert row["bounds_delta"]["radial_vertex_min_mm"]==pytest.approx(1.)
    assert row["classification"]=="display_geometry_or_representation_changed"
    assert row["newly_negative_margins"]==[]


def test_split_objective_material_uses_authoritative_intervals_and_retains_packing_difference():
    parent=dict(key="objective",local_start_z_mm=10.,upper_yoke_start_local_z_mm=10.,
                upper_yoke_end_local_z_mm=15.,lower_yoke_start_local_z_mm=20.,lower_yoke_end_local_z_mm=25.,
                mechanical_coil_axial_inset_mm=0.)
    part=dict(key="coil",parent_key="objective",mechanical_profile="magnetic_excitation_coil",
              local_start_z_mm=15.,local_center_z_mm=17.5,local_end_z_mm=20.)
    measured=dict(minimum_mm=[-5.,-5.,10.],maximum_mm=[5.,5.,25.],
                  radial_vertex_min_mm=2.,radial_vertex_max_mm=5.)
    result=comparison.dimensional_checks(part,measured,{"objective":parent})
    assert result["negative_margins"]==[]
    assert result["checks"]["physical_intervals"]["intervals_mm"]==[[10.,15.],[20.,25.]]
    assert result["checks"]["axial_packing_reference"]["rendered_minimum_minus_packing_minimum_mm"]==-5.
    assert result["checks"]["axial_envelope"]["start_margin_mm"]==0.
