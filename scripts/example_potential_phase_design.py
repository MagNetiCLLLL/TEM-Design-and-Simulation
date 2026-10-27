"""Bounded native-field phase matching on private instrument snapshots.

The target is synthetic: one isolated condenser stigmator, a controlled complex
entrance and a short static magnetic interval. This exercises the potential
wave executor as the design observable, not an empirical aberration screen or
a thin-ray response. It neither qualifies a whole column nor applies settings.
"""
import json

import numpy as np

from temsim.immutable_json import thaw_json
from temsim.optics.column import default_state
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.bounded_optical_design import DesignParameter, optimize_captured_design
from temsim.physics.potential_capture import capture_selected_column_potentials
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
from temsim.physics.potential_wave import propagate_static_magnetic_wave


def native_stigmator_wave(state, *, pixels=33, axial_steps=10):
    """Return the full complex field on the same physical lattice for every drive."""
    component = state.condenser_stigmator
    native = capture_selected_column_potentials(state, auxiliary_keys=(component.key,))
    snapshot = ElectromagneticPotentialSnapshot(native.scene, tuple(
        PotentialContribution(row.source_id, row.owner_ids,
                              AnalyticPotentialDomain(row.potential), row.kind)
        for row in native.contributions))
    axis = np.linspace(-400e-9, 400e-9, pixels)
    x, y = np.meshgrid(axis, axis)
    width = 60e-9
    amplitude = np.exp(-(x*x+y*y)/(2*width*width)
                       + 1j*(.37+.18*x/width-.13*y/width+.09*(x*x+y*y)/(width*width)))
    amplitude *= np.sqrt(.37/np.sum(abs(amplitude)**2))
    center = component.z_mm*1e-3
    result = propagate_static_magnetic_wave(snapshot, amplitude, axis, axis,
        np.linspace(center-50e-6, center+50e-6, axial_steps+1), energy_ev=300000.,
        kinetic_order=4, exponential_error=1e-11)
    return result.amplitude, snapshot.numerical_digest


def run_example():
    state = default_state()
    state.simulation_mode = "ideal"
    component = state.condenser_stigmator
    component.strength_x_percent, component.strength_y_percent = .18, -.11
    target, target_digest = native_stigmator_wave(state)
    component.strength_x_percent, component.strength_y_percent = .05, .03
    initial, initial_digest = native_stigmator_wave(state)
    scale = max(float(np.linalg.norm(initial-target)), 1e-12)

    def residual(trial):
        actual, _ = native_stigmator_wave(trial)
        difference = (actual-target).ravel()/scale
        # Keep both quadratures, the original absolute flux and all phase.
        # Do not align, recenter, normalize or fit away piston/tilt/defocus.
        return np.r_[difference.real, difference.imag]

    study = optimize_captured_design(state, (
        DesignParameter(("condenser_stigmator", "strength_x_percent"), -.3, .3, 1e-4),
        DesignParameter(("condenser_stigmator", "strength_y_percent"), -.3, .3, 1e-4)),
        residual, maximum_evaluations=48)
    return {"scope": __doc__, "target_drive_percent": [.18, -.11],
        "initial_drive_percent": [.05, .03], "target_snapshot_digest": target_digest,
        "initial_snapshot_digest": initial_digest, "input_probability": .37,
        "target_probability": float(np.sum(abs(target)**2)),
        "initial_complex_l2_difference": scale,
        "objective": "full complex field on a fixed laboratory lattice; both quadratures; no phase fitting",
        "design": thaw_json(study),
        "live_drive_percent_after_study": [component.strength_x_percent, component.strength_y_percent]}


def compact_summary(result):
    """Keep the CLI readable; the callable above retains every residual sample."""
    summary = {key: value for key, value in result.items() if key != "design"}
    study = result["design"]
    summary["design"] = {key: value for key, value in study.items() if key not in ("best", "evaluations")}
    summary["design"]["best"] = {key: value for key, value in study["best"].items() if key != "residual"}
    summary["design"]["evaluation_count"] = len(study["evaluations"])
    return summary


if __name__ == "__main__":
    print(json.dumps(compact_summary(run_example()), indent=2))
