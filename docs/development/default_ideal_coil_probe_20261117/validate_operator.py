"""Benchmark the near-axis design operator on an authenticated same-current gun.

The operator has a truncated vacuum field and omits all mechanical stops. Its
conditional output is compared only to the same IDs at a native entrance.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.constants import c, e, m_e

from aberrations import trace_spectral
from verify_pencil import field_for_recipe
from temsim.physics.beam_statistics import transverse_beam_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("gun", type=Path)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("auxiliary_basis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inputs = json.loads((args.trial / "inputs.json").read_text())
    meta = json.loads((args.gun / "gun.json").read_text())
    if inputs["digest"] != meta["input_digest"]:
        raise ValueError("Gun inputs differ from the column trial")
    path = args.gun / "gun.npz"
    checksum = sha256(path.read_bytes()).hexdigest()
    if checksum != meta.get(
        "sha256", "8fa665031e0169882ccda4c0495f2061ed1252818e6760b7683960fd83aa7885"
    ):
        raise ValueError("Original gun checkpoint changed")
    from temsim.instrument_snapshot import InstrumentSnapshot

    state = InstrumentSnapshot.from_dict(inputs).restore()
    recipe = json.loads(args.recipe.read_text())
    for owner, group in recipe["recipes"].items():
        if group != state.lens_field_map_descriptors[owner]["ideal_current_segments"]:
            raise ValueError(f"Winding recipe differs from the executed gun for {owner}")
    for owner, commands in recipe.get("multipole_commands", {}).items():
        component = next(c for c in state.corrector_elements if c.key == owner)
        if any(getattr(component, k) != v for k, v in commands.items()):
            raise ValueError(f"Multipole commands differ for {owner}")
    with np.load(path) as gun:
        live = gun["alive"] & (gun["weight"] > 0)
        ids = gun["ray_id"]
        weights = gun["weight"][live]
        initial = np.column_stack([gun[k][live] for k in ("x_m", "y_m", "tx_rad", "ty_rad")])
        offset = gun["energy_offset_ev"][live]
        emitted_current = float(gun["emitted_current_a"])
        if len(np.unique(ids)) != len(ids) or np.sum(gun["weight"] > 0) < 1000:
            raise ValueError("Insufficient original source support")
    field, h, start, stop = field_for_recipe(recipe, args.auxiliary_basis, 0.075)
    nominal = state.electron_gun.nominal_exit_energy_ev
    rest = m_e * c * c / e
    energy = nominal + offset
    rigidity = np.sqrt(nominal * (nominal + 2 * rest) / (energy * (energy + 2 * rest)))
    result = trace_spectral(initial, field, h, rigidity)
    obs = transverse_beam_statistics(*result.T, weights=weights)
    report = dict(
        status="Diagnostic approximation; native column interception and focus require independent transport",
        original_positive=meta["population"]["positive_current_count"],
        live_gun_exit=int(live.sum()),
        input_digest=inputs["digest"],
        gun_sha256=checksum,
        step_mm=h * 1000,
        recipe_sha256=sha256(args.recipe.read_bytes()).hexdigest(),
        z_interval_m=[start, stop],
        predicted_rms_radius_nm=obs.radius_rms_m * 1e9,
        predicted_D95_nm=obs.radius_95_m * 2e9,
        predicted_alpha95_mrad=obs.convergence_95_mrad,
        predicted_50pm_current_a=float(
            emitted_current * weights[np.hypot(*result[:, :2].T) <= 50e-12].sum()
        ),
    )
    if (args.trial / "metrics.json").is_file():
        with np.load(args.trial / "surface.npz") as native:
            if str(native["input_digest"]) != inputs["digest"] or not np.array_equal(
                native["ray_id"], ids
            ):
                raise ValueError("Native and design samples differ")
            mask = native["alive"][live]
            native_phase = np.column_stack(
                [native[k][live][mask] for k in ("x_m", "y_m", "tx_rad", "ty_rad")]
            )
        difference = result[mask] - native_phase
        report["native_comparison"] = dict(
            survivors=int(mask.sum()),
            position_vector_rms_error_nm=float(
                np.sqrt(np.average(np.sum(difference[:, :2] ** 2, axis=1), weights=weights[mask])) * 1e9
            ),
            slope_vector_rms_error_mrad=float(
                np.sqrt(np.average(np.sum(difference[:, 2:] ** 2, axis=1), weights=weights[mask])) * 1e3
            ),
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
