"""Inspect binary64 excitation grouping without changing model tolerances."""
from copy import copy, deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.simulation_modes import switch_mode


def main():
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    switch_mode(state, "unified")
    apply_operating_mode_pair(
        state, "nano_probe", "diffraction", column_name=selection.column,
        recording_name=selection.recording,
    )
    apply_physical_layout_to_state(state)
    state.round_lens_field_model = "copper_coil"
    state._runtime_lens_field_provider_cache = {}
    state._runtime_copper_coil_provider_cache = {}
    rows = []
    for lens in state.lenses:
        if not lens.enabled:
            continue
        key = lens.key
        legacy = capture_round_lens_potential(state, key).potential
        points = []
        for coil in legacy.coils:
            for offset in (-2., -.5, 0., .5, 2.):
                for radial_fraction in (0., .005, .2, .7):
                    for azimuth in (0., .7):
                        radius = coil.inner_radius_m * radial_fraction
                        points.append((radius*np.cos(azimuth), radius*np.sin(azimuth),
                                       coil.center_z_m + coil.length_m*offset))
        registration = legacy.coils[0].registration
        points = np.asarray(points) @ registration.rotation_array.T + registration.origin_array_m
        baseline = legacy.sample(points)
        coil_keys = [part.key for part in state._resolved_assembly.parts
                     if part.data.get("mechanical_profile") == "magnetic_excitation_coil"
                     and part.data.get("field_source_key") == key]
        results = []
        for turns, current in ((100, 30.), (128, 23.4375)):
            candidate = copy(state)
            candidate.lenses = deepcopy(state.lenses)
            candidate._runtime_lens_field_provider_cache = {}
            candidate._runtime_copper_coil_provider_cache = {}
            def edited(part):
                if part.key == key:
                    return replace(part, data={**part.data, "ideal_coil_current_a_at_100_percent": current})
                if part.key in coil_keys:
                    return replace(part, data={**part.data, "winding_turn_count": turns,
                                               "winding_wire_diameter_mm": .125})
                return part
            candidate._resolved_assembly = replace(
                state._resolved_assembly,
                parts=tuple(edited(part) for part in state._resolved_assembly.parts),
            )
            explicit = capture_round_lens_potential(candidate, key).potential
            sampled = explicit.sample(points)
            difference = sampled["B_t"] - baseline["B_t"]
            limits = 1e-20 + 4e-14*np.abs(baseline["B_t"])
            fails = np.abs(difference) > limits
            worst = np.unravel_index(np.argmax(np.abs(difference)/limits), difference.shape)
            result = {
                "N": turns, "I100_A": current, "nominal_NI100": turns*current,
                "legacy_per_interval_NI": [coil.ampere_turns for coil in legacy.coils],
                "explicit_per_interval_NI": [coil.ampere_turns for coil in explicit.coils],
                "NI_array_equal": all(a.ampere_turns == b.ampere_turns
                                      for a, b in zip(legacy.coils, explicit.coils)),
                "A_array_equal": bool(np.array_equal(sampled["A_t_m"], baseline["A_t_m"])),
                "B_array_equal": bool(np.array_equal(sampled["B_t"], baseline["B_t"])),
                "max_abs_B_difference_T": float(np.max(np.abs(difference))),
                "max_abs_A_difference_T_m": float(np.max(np.abs(sampled["A_t_m"]-baseline["A_t_m"]))),
                "B_existing_tolerance_passed": bool(not fails.any()),
                "B_failing_components": int(fails.sum()),
                "worst_tolerance_ratio": float(abs(difference[worst])/limits[worst]),
                "worst_legacy_B_T": float(baseline["B_t"][worst]),
                "worst_explicit_B_T": float(sampled["B_t"][worst]),
                "worst_point_m": points[worst[0]].tolist(),
                "worst_component": int(worst[1]),
            }
            results.append(result)
        rows.append({"key": key, "point_count": len(points), "decompositions": results})
        print(key, [(result["N"], result["NI_array_equal"], result["B_existing_tolerance_passed"],
                     result["max_abs_B_difference_T"]) for result in results])
    receipt = {
        "scope": "Detached component arrays only. No beam, source emission, propagation, solver, configuration, physical parameter or tolerance edits.",
        "construction": "Standard no-compatible-result GUI startup stored preset; ideal-current model selected only on the private inspection state.",
        "sampling": "40 points per current-bearing interval: bore radii 0, 0.005, 0.2, 0.7 times inner radius, azimuths 0 and 0.7 rad, z offsets -2, -0.5, 0, 0.5, 2 times interval length.",
        "B_comparison_tolerance": {"rtol": 4e-14, "atol_T": 1e-20},
        "diagnosis": "The legacy grouping (NI100*percent)*0.01*polarity differs from N*(I100*(percent*0.01*polarity)) in binary64. Nominal NI100 equality does not always yield identical captured per-interval coefficients. Both 100/30 and power-of-two 128/23.4375 can differ. Near-zero cancellation components may fail the unchanged strict absolute tolerance even when absolute field differences are tiny. Results are raw diagnostics, not an end-to-end pass claim.",
        "rows": rows,
    }
    path = Path(__file__).with_name("nominal-ni-roundoff-review.json")
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("RECEIPT", path.resolve())


if __name__ == "__main__":
    main()
