"""Propose an Objective trim from the measured, field-specific surface population."""

import argparse
import json
from pathlib import Path

import numpy as np

from fit_stationary import scaled_recipe, write
from stationary_pupil import (
    CHANNELS,
    Column,
    current_vector,
    half_nodes,
    hex_vector,
    kinematics,
    read_gun,
    transport,
)
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.beam_statistics import _weighted_quantile


def centred(values, weights):
    return values - np.average(values, weights=weights, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("basis", "recipe", "gun", "observation", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    base = json.loads(args.recipe.read_text())
    measured = json.loads((args.observation / "optics.json").read_text())
    receipt = json.loads((args.gun / "gun.json").read_text())
    if measured["input_digest"] != receipt["input_digest"] or measured["gun_sha256"] != receipt["sha256"]:
        raise ValueError("The observation and physical gun must have identical inputs")
    state = InstrumentSnapshot.from_dict(json.loads((args.gun / "inputs.json").read_text())).restore()
    if any(
        state.lens_field_map_descriptors[group]["ideal_current_segments"] != spec
        for group, spec in base["recipes"].items()
    ):
        raise ValueError("The measured gun and the proposed winding reference differ")
    components = {element.key: element for element in state.corrector_elements}
    if any(getattr(components[key], attr) != base["multipole_commands"][key][attr] for key, attr in CHANNELS):
        raise ValueError("The measured gun and the proposed corrector reference differ")
    phase, weights, ids, energy, _ = read_gun(args.gun)
    with np.load(args.observation / "surface.npz") as samples:
        if not np.array_equal(ids, samples["ray_id"]) or not np.array_equal(weights, samples["weight"]):
            raise ValueError("Surface membership or absolute current changed")
        surface_phase = samples["phase"]
    with np.load(args.basis) as fields:
        start, stop = fields["z"][0], float(fields["surface"])
    z = np.linspace(start, stop, int(np.ceil((stop - start) / 0.00018)) + 1)
    coil = Column(args.basis, base, half_nodes(z))
    current, hp = current_vector(base), hex_vector(base)
    objective = current * (np.array(base["owner_per_group"]) == "objective_lens")
    field = coil.field(current, hp)
    dfield = coil.field(objective, np.zeros_like(hp))
    scale = 1.0e-6
    end = np.array([len(z) - 1])
    momentum = kinematics(state.electron_gun.nominal_exit_energy_ev + energy)
    plus, minus = [
        transport(phase, field + sign * scale * dfield, np.diff(z), *momentum, end)[0] for sign in (1, -1)
    ]
    # Anchor the intercept at the measured full-field result. Only the
    # differential response is taken from the cheaper magnetic design operator.
    response = centred((plus[:, :2] - minus[:, :2]) * 0.5e9, weights)
    position = centred(surface_phase[:, :2] * 1e9, weights)
    slope = centred(surface_phase[:, 2:4], weights)
    quantile_index = int(np.ceil(0.95 * len(weights))) - 1
    uniform = np.all(weights == weights[0])

    def diameter(offset, dz):
        xy = position + offset * response + dz * slope
        squared = np.einsum("ij,ij->i", xy, xy)
        q = (
            np.partition(squared, quantile_index)[quantile_index]
            if uniform
            else _weighted_quantile(squared, weights, 0.95)
        )
        return float(2 * np.sqrt(q))

    axial_response = -np.sum(weights[:, None] * response * slope) / np.sum(weights[:, None] * slope**2)
    old_focus = measured["full"]["D95"]["representative_offset_nm"]
    expected = -old_focus / axial_response
    # Search the actual order statistic near its anticipated shift. Sampling
    # z independently allows the current to place the smallest D95 on target.
    dz = np.arange(-max(32.0, abs(old_focus) + 12.0), max(32.0, abs(old_focus) + 12.0) + 0.125, 0.25)
    centre, spread = expected, max(6.0, abs(old_focus) / 2) / abs(axial_response)
    for _ in range(3):
        grid = np.linspace(centre - spread, centre + spread, 17)
        candidates = []
        for trim in grid:
            widths = np.array([diameter(trim, v) for v in dz])
            offset = dz[np.argmin(widths)]
            candidates.append((abs(offset), diameter(trim, 0.0), trim, offset))
        _, width, centre, offset = min(candidates)
        spread /= 5.0
    proposal = scaled_recipe(base, current + centre * scale * objective, hp)
    write(args.output / "currents.json", proposal)
    write(
        args.output / "adjustment.json",
        dict(
            parent_input=receipt["input_digest"],
            original_particles=len(phase),
            original_D95_focus_nm=old_focus,
            predicted_D95_focus_nm=offset,
            predicted_surface_D95_nm=width,
            axial_response_nm_per_ppm=axial_response,
            objective_fractional_change=centre * scale,
            scope="Differential design estimate requiring a new field-specific source and transported through-focus check",
        ),
    )
    print("Objective fractional trim", centre * scale, "predicted waist offset (nm)", offset, flush=True)


if __name__ == "__main__":
    main()
