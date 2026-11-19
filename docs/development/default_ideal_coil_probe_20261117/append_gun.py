"""Append a verified nested source to an existing stationary magnetic-table gun."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gun_table import GunField
from temsim import input_io
from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot
from temsim.optics.electron_gun.tracing import _trace_feg_to_exit
from temsim.physics.beam_statistics import original_source_population
from temsim.physics.grounded_particle_step import gun_step_workers
from temsim.physics.gun_field_environment import instrument_gun_field_context
from temsim.physics.incremental_particles import _slice_bundle, capture_emission_phase, prove_emission_prefix
from temsim.physics.ray_identity import emitted_source_identity
from temsim.vacuum import bind_gun_environment
from trial import write


@input_io.using_state_inputs
def append(state, prior, receipt, previous, new_count, out):
    gun = state.electron_gun
    old_count = gun.emitter.ray_count
    if new_count - old_count < 1000 or receipt["population"]["positive_current_count"] != old_count:
        raise ValueError("Require a complete physical prefix and at least 1000 additional positive originals")
    if capture_instrument_snapshot(state).digest != prior.digest:
        raise ValueError("Restored physical inputs or solver implementation changed")
    old_emit = capture_emission_phase(gun, gun.emit())
    for name in ("ray_id", "position_m", "direction", "normal"):
        if not np.array_equal(old_emit[name], previous["emission_" + name], equal_nan=True):
            raise ValueError("Regenerated source differs from recorded original emission")
    if not np.array_equal(old_emit["weight"], previous["weight"]):
        raise ValueError("Executed source weights changed")
    gun.emitter.ray_count = new_count
    snap = capture_instrument_snapshot(state)
    write(out / "inputs.json", snap.to_dict())
    emitted = gun.emit()
    new_emit = capture_emission_phase(gun, emitted)
    prove_emission_prefix(old_emit, new_emit, old_count)
    bind_gun_environment(state)
    with instrument_gun_field_context(state) as native:
        cached = receipt["magnetic_table"]
        table = GunField(native, (gun.exit_plane_z_mm + 3) * 0.001, round(cached["spacing_mm"], 9))
        if table.receipt()["coefficient_sha256"] != cached["coefficient_sha256"]:
            raise ValueError("Changed gun field table invalidates prefix reuse")
        gun._instrument_magnetic_field = table
        gun._instrument_magnetic_identity = None
        with gun_step_workers():
            result = _trace_feg_to_exit(gun, emitted_bundle=_slice_bundle(emitted, old_count))
    if result.emitted_current_a != float(previous["emitted_current_a"]):
        raise ValueError("Reference current changed")
    e = result.exit_bundle
    if not np.array_equal(e.ray_id, new_emit["ray_id"][old_count:]) or not np.array_equal(
        e.weight, new_emit["weight"][old_count:]
    ):
        raise ValueError("Appended IDs or source weights differ from the declared launch")
    arrays = {
        name: np.r_[previous[name], getattr(e, name)]
        for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "flight_time_s", "alive", "ray_id")
    }
    arrays.update(
        weight=new_emit["weight"],
        blocked_z_mm=np.r_[previous["blocked_z_mm"], result.blocked_z_mm],
        blocked_key=np.concatenate([previous["blocked_key"], np.array(result.blocked_key, dtype=str)]),
        source_azimuth_rad=np.r_[previous["source_azimuth_rad"], emitted_source_identity(result)[1]],
        emitted_current_a=result.emitted_current_a,
    )
    reference = {}
    for name in ("ray_id", "position_m", "direction", "normal"):
        value = np.concatenate([previous["emission_" + name], result.emission_reference[name]])
        if not np.array_equal(value, new_emit[name], equal_nan=True):
            raise ValueError("Executed concatenated source differs from its declared launch")
        reference[name] = arrays["emission_" + name] = value
    population = original_source_population(
        SimpleNamespace(
            emission_reference=reference,
            exit_bundle=SimpleNamespace(**arrays),
            emitted_current_a=result.emitted_current_a,
        )
    )
    np.savez_compressed(out / "gun.npz", **arrays)
    write(
        out / "gun.json",
        dict(
            population=population,
            input_digest=snap.digest,
            sha256=sha256((out / "gun.npz").read_bytes()).hexdigest(),
            magnetic_table=table.receipt(),
            parent_gun_sha256=receipt["sha256"],
            reused_originals=old_count,
            appended_originals=new_count - old_count,
            prefix_weight_scale=old_count / new_count,
            verified_emission_fields=sorted(new_emit),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--particles", type=int, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.parent / "gun.json").read_text())
    snap = InstrumentSnapshot.from_dict(json.loads((args.parent / "inputs.json").read_text()))
    path = args.parent / "gun.npz"
    if receipt["input_digest"] != snap.digest or sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("Gun checkpoint or original physical inputs changed")
    args.output.mkdir(exist_ok=False)
    with np.load(path) as g:
        append(snap.restore(), snap, receipt, g, args.particles, args.output)
