"""Compact checks for unchanged-source downstream optical-reference audits."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_spec = spec_from_file_location("downstream_crossover_audit",
    Path(__file__).resolve().parents[1]/"scripts"/"validate_downstream_crossovers.py")
audit = module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_capture_and_refinement_keep_endpoints_and_exact_receiver_planes():
    state = SimpleNamespace(sample=SimpleNamespace(upper_surface_z_mm=1.),
        fluorescent_screen=SimpleNamespace(z_mm=4.), lenses=[], apertures=[],
        recording_planes=[SimpleNamespace(z_mm=2.123, installed=True)])
    planes = audit.requested_planes(state)
    assert planes[0] == 1. and planes[-1] == 4.
    assert 2.123 in planes and np.diff(planes).max() <= .5
    report = {"interpolated_candidates_diagnostic_only": [
        {"z_mm": 1.000004, "bracket_mm": [1., 1.5]},
        {"z_mm": 3.99999, "bracket_mm": [3.5, 4.]}]}
    refined = audit.refined_planes(planes, report)
    assert refined[0] == 1. and refined[-1] == 4.
    assert np.all(np.diff(refined) > 0)
    assert 1.000004 in refined and 3.99999 in refined


def test_archived_source_identity_tracks_energy_clock_weight_and_id():
    arrays = {key: np.array([[0., 1.]]) for key in
              ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s")}
    arrays.update(source_energy_offset_ev=np.array([.1, -.2]), weight=np.array([.3, .7]),
                  alive=np.array([[True, True]]), source_id=np.array([17, 29]))
    reference = audit.source_identity(arrays)
    for key in ("flight_time_s", "source_energy_offset_ev", "weight", "source_id"):
        changed = {name: value.copy() for name, value in arrays.items()}
        changed[key].flat[0] += 1
        assert audit.source_identity(changed) != reference


def test_step_comparison_retains_real_centroid_and_current_changes():
    coarse = dict(z_mm=np.array([1., 2.]), source_id=np.array([3, 8]), weight=np.array([.25, .75]),
        alive=np.ones((2, 2), bool), x_m=np.array([[0., 1e-9], [0., 2e-9]]),
        y_m=np.zeros((2, 2)), tx_rad=np.array([[0., .002], [0., .003]]),
        ty_rad=np.zeros((2, 2)), kinetic_energy_ev=np.full((2, 2), 300000.))
    fine = {key: value.copy() for key, value in coarse.items()}
    fine["x_m"] += 2e-9
    fine["alive"][-1, 0] = False
    compared = audit.compare_steps(coarse, fine)
    assert compared["changed_particle_plane_survival_count"] == 1
    assert np.isclose(compared["maximum_same_particle_position_coordinate_change_nm"], 2.)
    assert compared["maximum_actual_source_current_fraction_change"] == .25
    assert compared["maximum_actual_alpha95_change_mrad"] > 0.
