"""Independent ideal Objective audit, with exact axial integration and no transport."""
from copy import copy, deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.constants import mu_0
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from temsim.assembly_catalog import AssemblyCatalog
from temsim.coil_winding import OWNER_CURRENT_FIELD
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.lens_field_provider import resolve_runtime_lens_field_provider
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.simulation_modes import switch_mode


def forbidden(*args, **kwargs):
    raise AssertionError("No source or particle transport in a field-only audit")


def run():
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    switch_mode(state, "unified")
    apply_operating_mode_pair(state, "nano_probe", "diffraction",
                             column_name=selection.column, recording_name=selection.recording)
    apply_physical_layout_to_state(state)
    state.step_mm = .05
    original = deepcopy(state.to_dict())
    lens = next(item for item in state.lenses if item.key == "objective_lens")
    plane = float(state.sample.upper_surface_z_mm)
    gaussian = resolve_runtime_lens_field_provider(state, lens.key, lens)
    target = float(gaussian.magnetic_field_t(plane))
    reference_current = 1575.2330285262728
    detached = copy(state)
    detached.lenses = deepcopy(state.lenses)
    detached.round_lens_field_model = "copper_coil"
    detached._runtime_copper_coil_provider_cache = {}
    detached._runtime_lens_field_provider_cache = {}
    detached._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(item, data={**item.data, OWNER_CURRENT_FIELD: reference_current})
        if item.key == lens.key else item for item in state._resolved_assembly.parts))
    potential = capture_round_lens_potential(detached, lens.key).potential
    current = potential.provenance["winding_current_definition"]["current_a"]

    def axis(z_mm):
        return sum(float(coil.axis_bz_t(z_mm*.001)) for coil in potential.coils)

    def independent_axis(z_mm):
        # Integrate each ideal circular loop analytically along the occupied
        # axial interval, then integrate only the radius with adaptive QUADPACK.
        # This does not call the production tensor Gauss-Legendre loop sampler.
        z_m, total = z_mm*.001, 0.
        for coil in potential.coils:
            before = z_m-coil.center_z_m+coil.length_m/2
            after = z_m-coil.center_z_m-coil.length_m/2
            radial = quad(lambda radius: (
                before/np.sqrt(radius*radius+before*before)
                - after/np.sqrt(radius*radius+after*after)),
                coil.inner_radius_m, coil.outer_radius_m,
                epsabs=1e-14, epsrel=1e-14)[0]
            total += (mu_0*coil.ampere_turns
                      /(2*(coil.outer_radius_m-coil.inner_radius_m)*coil.length_m)*radial)
        return total

    fit = minimize_scalar(lambda z_mm: -axis(z_mm), bounds=(1650, 1800),
                          method="bounded", options={"xatol": 1e-9})
    points = np.linspace(1300, 1900, 121)
    actual = np.array([axis(z_mm) for z_mm in points])
    reference = np.array([independent_axis(z_mm) for z_mm in points])
    assert state.to_dict() == original
    np.testing.assert_allclose(axis(plane), target, rtol=2e-14, atol=0)
    assert all(coil.ampere_turns == 586*current for coil in potential.coils)
    result = dict(
        scope="Ideal Objective field only; no emission, particle transport or coherent calculation",
        surface_z_mm=plane, unchanged_excitation_percent=lens.percent,
        current_at_100_percent_a=reference_current, current_a=current,
        target_gaussian_surface_t=target, selected_q12_surface_t=axis(plane),
        surface_match_relative=axis(plane)/target-1,
        independent_exact_surface_t=independent_axis(plane),
        independently_calculated_exact_matching_current_a=current*target/independent_axis(plane),
        q12_surface_vs_independent_relative=axis(plane)/independent_axis(plane)-1,
        global_peak_q12_t=float(-fit.fun), lower_peak_z_mm=float(fit.x),
        sampled_curve_z_mm=[1300, 1900], sampled_curve_count=121,
        q12_curve_peak_normalized_error=float(np.max(np.abs(actual-reference))/np.max(reference)),
        per_interval_turns=[586, 586],
        per_interval_ampere_turns=[coil.ampere_turns for coil in potential.coils],
        geometry=[dict(inner_diameter_mm=coil.inner_radius_m*2000,
                       outer_diameter_mm=coil.outer_radius_m*2000,
                       z_start_mm=(coil.center_z_m-coil.length_m/2)*1000,
                       z_end_mm=(coil.center_z_m+coil.length_m/2)*1000)
                  for coil in potential.coils],
        original_field_model=state.round_lens_field_model,
        detached_selected_field_model=detached.round_lens_field_model,
        original_controls_unchanged=True, potential_digest=potential.digest)
    path = Path(__file__).with_name("receipt.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    with patch("temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit", forbidden), \
         patch("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden):
        run()
