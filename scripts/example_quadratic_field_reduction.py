"""Run bounded examples without applying a setting or writing a result file.

Run from the repository with PYTHONPATH=src and the project Python environment.
The uniform-field example qualifies a represented analytical limit. The design
example uses existing thin particle optics as its explicitly declared forward
model; it does not qualify that model as a finite-field surrogate.
"""
import json
from types import SimpleNamespace

import numpy as np

from temsim.immutable_json import thaw_json
from temsim.optics.column import default_state
from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.bounded_optical_design import DesignParameter, optimize_captured_design
from temsim.physics.core import build_propagation_plan, execute_propagation_plan
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
from temsim.physics.qualified_field_reduction import qualify_quadratic_interval


def thin_projector_position(state):
    """Actual existing thin-column execution, not a fitted response formula."""
    lens = state.projector_lens_p1
    stop = lens.z_mm+50.
    plan = build_propagation_plan(state, lens.z_mm-5., stop, checkpoint_z_mm=(stop,))
    values = execute_propagation_plan(state, plan, np.array([1e-7]), np.array([0.]),
                                     np.array([0.]), np.array([0.]))
    return float(values[-1].x_m[-1, 0])


def run_example():
    radial, axial = np.array([0., 1e-4]), np.array([-.1, .1])
    field = AxisymmetricMagneticPotential(radial, axial,
        np.broadcast_to(.5*.125*radial[:, None], (2, 2)))
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    snapshot = ElectromagneticPotentialSnapshot(scene,
        (PotentialContribution("uniform_B", ("example_coil",), field, "magnetic"),))
    reduced = qualify_quadratic_interval(snapshot, interval_m=(0., 1e-5),
        position_bounds_m=((-1e-5, 1e-5), (-1e-5, 1e-5)),
        energy_range_ev=(290000., 310000.), maximum_canonical_slope=.001)
    position, slope, evidence = reduced.propagate_rays(snapshot, [[1e-7, 0.]],
        [[1e-4, 0.]], energy_ev=300000.)

    state = default_state()
    state.simulation_mode = "ideal_thin"
    state.acceleration_backend, state.acceleration_enabled = "CPU", False
    state.history_step_mm = 20.
    # A synthetic target from the same declared forward model demonstrates
    # execution and recovery, not an experimentally calibrated optical design.
    state.projector_lens_p1.percent = 68.
    target = thin_projector_position(state)
    state.projector_lens_p1.percent = 50.
    design = optimize_captured_design(state,
        (DesignParameter(("projector_lens_p1", "percent"), 30., 90.),),
        lambda trial: np.array([(thin_projector_position(trial)-target)/1e-7]),
        maximum_evaluations=24)
    return {"uniform_field_position_m": position.tolist(), "uniform_field_slope_rad": slope.tolist(),
        "reduction": thaw_json(evidence), "design": thaw_json(design),
        "live_projector_percent_after_study": state.projector_lens_p1.percent}


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2))
