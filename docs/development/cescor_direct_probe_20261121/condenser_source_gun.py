"""Relaunch the same original tip quadrature and zero-weight transfer probes in the current field."""

import argparse
import json
from time import monotonic

import numpy as np

from condenser_source_fields import Fields, OUT
from temsim.optics.electron_gun.base import EmissionBundle
from temsim.optics.electron_gun.tracing import _trace_feg_to_exit
from temsim.physics.instrument_electric import (
    configure_instrument_electric_domain,
    instrument_electric_end_mm,
)
from temsim.vacuum import bind_gun_environment


def run(case, n=4096):
    if n < 1000:
        raise ValueError("At least 1000 original positive-current emission records are required")
    f = Fields()
    state = f.state
    gun = state.electron_gun
    if case == "after":
        q = f.current(*json.loads((OUT / "proposal.json").read_text())["series_A"])
    else:
        q = f.before.copy()
        if case in ("c1_off", "both_off"):
            q[:4] = 0
        if case in ("c2_off", "both_off"):
            q[4:8] = 0
    bind_gun_environment(state)
    configure_instrument_electric_domain(gun, instrument_electric_end_mm(state))
    gun._instrument_magnetic_field = f.gun_field(q)
    gun._instrument_magnetic_identity = None
    identity = f.gun_key(q, n)
    gun.emitter.ray_count = n
    physical = gun.emit()
    for attr in ("ray_id", "x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev"):
        np.testing.assert_array_equal(getattr(physical, attr), getattr(f.emission, attr)[:n])
    assert (physical.weight > 0).all() and np.isclose(physical.weight.sum(), 1.0)
    # Two symmetric perturbation amplitudes at fixed launch energy. They carry zero current.
    scale = np.array([1.0e-9, 1.0e-9, 1.0e-6, 1.0e-6])
    diag = np.r_[
        np.zeros((1, 4)), np.diag(scale), -np.diag(scale), np.diag(scale / 2), -np.diag(scale / 2)
    ]
    names = ("x_m", "y_m", "tx_rad", "ty_rad")
    args = {k: np.r_[getattr(physical, k), diag[:, j]] for j, k in enumerate(names)}
    launch = EmissionBundle(
        **args,
        energy_offset_ev=np.r_[physical.energy_offset_ev, np.zeros(len(diag))],
        weight=np.r_[physical.weight, np.zeros(len(diag))],
        ray_id=np.r_[physical.ray_id, np.arange(n, n + len(diag)) + 1000000],
    )
    start = monotonic()
    print("Tracing original tip:", case, n, flush=True)
    result = _trace_feg_to_exit(gun, emitted_bundle=launch)
    exit_bundle = result.exit_bundle
    np.testing.assert_array_equal(exit_bundle.ray_id, launch.ray_id)
    np.testing.assert_array_equal(exit_bundle.weight, launch.weight)
    if not exit_bundle.alive.all():
        raise ValueError("Gun contact: trace and retain intercepts before fitting the column")
    phase = np.column_stack([getattr(exit_bundle, k) for k in (*names, "flight_time_s")])
    zhist = np.asarray(result.z_mm)
    selected = np.linspace(0, n - 1, 64, dtype=int)
    np.savez_compressed(
        OUT / f"gun-{case}-{n}.npz",
        phase=phase,
        ids=launch.ray_id,
        weight=launch.weight,
        gun_key=identity,
        energy=exit_bundle.energy_offset_ev + 300000.0,
        launch=np.column_stack([getattr(launch, k) for k in names]),
        launch_energy=launch.energy_offset_ev + gun.emitter.emission_energy_ev,
        current=q,
        emitted_A=result.emitted_current_a,
        gun_max_radius_m=np.nanmax(np.hypot(result.x_m, result.y_m)),
        gun_z_mm=zhist,
        gun_paths_m=np.stack([result.x_m[:, selected], result.y_m[:, selected]], axis=2),
        gun_seconds=monotonic() - start,
    )
    print("Completed gun:", case, n, round(monotonic() - start, 1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("before", "after", "c1_off", "c2_off", "both_off"))
    parser.add_argument("--count", type=int, default=4096)
    args = parser.parse_args()
    run(args.case, args.count)
