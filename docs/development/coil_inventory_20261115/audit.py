"""Read-only installed-winding inventory and matched-NI field comparison.

Run with the project Python and --output pointing to a new evidence directory.
No particles are emitted, no waves are propagated, and no inputs are saved.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.cad_round_winding import resolve_round_winding_geometry
from temsim.coil_reference_metadata import OWNER_CURRENT_FIELD, current_intervals_mm
from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair, condenser_mode_key
from temsim.physics.lens_field_provider import _runtime_excitation
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.simulation_modes import switch_mode


ROOT = Path(__file__).resolve().parents[3]
RTOL, ATOL_T = 4e-14, 1e-20  # Existing explicit-winding B-equivalence criterion.
LABELS = {
    "condenser_lens_1": "C1 lens", "condenser_lens_2": "C2 lens",
    "condenser_lens_3": "C3 lens", "adapter_lens": "Probe Adapter lens",
    "probe_tl22_lens": "Probe TL22 lens", "probe_tl21_lens": "Probe TL21 lens",
    "probe_tl12_lens": "Probe TL12 lens", "mini_condenser": "Mini Condenser lens",
    "objective_lens": "Objective lens", "diffraction_lens": "Diffraction lens",
    "intermediate_lens": "Intermediate lens", "projector_lens_1": "P1 lens",
    "projector_lens_2": "P2 lens",
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+"\n", encoding="utf-8")


def protected_hashes():
    files = [p for tree in (ROOT/"configs", ROOT/"src") for p in tree.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def detached(state):
    result = copy(state)
    result.lenses = deepcopy(state.lenses)
    result.simulation_mode = "unified"
    result.round_lens_field_model = "copper_coil"  # Compatibility identifier.
    result._runtime_copper_coil_provider_cache = {}
    result._runtime_lens_field_provider_cache = {}
    return result


def edit(state, changes):
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(p, data={**p.data, **changes[p.key]}) if p.key in changes else p
        for p in state._resolved_assembly.parts))


def sample_positions(potential):
    """Vacuum points spanning axis, branch transition, bore, exterior and tails."""
    coils = potential.coils
    inner = min(c.inner_radius_m for c in coils)
    outer = max(c.outer_radius_m for c in coils)
    lower = min(c.center_z_m-c.length_m/2 for c in coils)
    upper = max(c.center_z_m+c.length_m/2 for c in coils)
    scale = max(outer, upper-lower)
    axial = {lower, upper, (lower+upper)/2}
    for coil in coils:
        axial.update(coil.center_z_m+t*coil.length_m for t in (-.501, -.5, 0., .5, .501))
    axial.update(lower-t*scale for t in (.1, 1., 3., 10.))
    axial.update(upper+t*scale for t in (.1, 1., 3., 10.))
    rows = []
    for z in sorted(axial):
        rows.append((0., 0., z))
        for r in (1e-7*inner, .009*inner, .011*inner, .2*inner, .8*inner, .99*inner, 1.5*outer):
            for theta in np.arange(8)*np.pi/4:
                rows.append((r*math.cos(theta), r*math.sin(theta), z))
    local = np.array(rows)
    registration = coils[0].registration
    assert all(c.registration == registration for c in coils)
    global_points = local@registration.rotation_array.T+registration.origin_array_m
    branches = [c._near_axis(c._local(global_points)) for c in coils]
    return global_points, dict(points=len(rows),
        near_axis_point_evaluations=sum(int(np.sum(b)) for b in branches),
        elliptic_point_evaluations=sum(int(np.sum(~b)) for b in branches),
        maximum_tail_distance_in_envelope_scales=10.)


def compare(old, new, points, *, require_pass):
    old_geometry = [{k:v for k,v in c.to_metadata().items() if k != "ampere_turns"} for c in old.coils]
    new_geometry = [{k:v for k,v in c.to_metadata().items() if k != "ampere_turns"} for c in new.coils]
    assert old_geometry == new_geometry, "Source envelope, pose or quadrature changed"
    before, after = old.sample(points), new.sample(points)
    delta = after["B_t"]-before["B_t"]
    accepted = np.abs(delta) <= ATOL_T+RTOL*np.abs(before["B_t"])
    if require_pass:
        np.testing.assert_allclose(after["B_t"], before["B_t"], rtol=RTOL, atol=ATOL_T)
    # Public fast B-only sampling must satisfy the same unchanged criterion.
    fast_before = old.field_at_global_positions_t(points)
    fast_after = new.field_at_global_positions_t(points)
    fast_accepted = np.abs(fast_after-fast_before) <= ATOL_T+RTOL*np.abs(fast_before)
    if require_pass:
        np.testing.assert_allclose(fast_after, fast_before, rtol=RTOL, atol=ATOL_T)
    norm = np.linalg.norm(before["B_t"], axis=1)
    relative = np.linalg.norm(delta, axis=1)/norm
    coefficients_equal = all(a.to_metadata() == b.to_metadata() for a,b in zip(old.coils, new.coils))
    return dict(passed=bool(accepted.all() and fast_accepted.all()),
        failed_b_components=int(np.sum(~accepted)), failed_fast_b_components=int(np.sum(~fast_accepted)),
        geometry_pose_quadrature_identical=True,
        coefficients_bitwise_equal=coefficients_equal,
        max_abs_delta_ni_a_turn=max(abs(a.ampere_turns-b.ampere_turns) for a,b in zip(old.coils,new.coils)),
        max_abs_delta_b_t=float(np.max(np.abs(delta))),
        max_vector_relative_delta_b=float(np.max(relative)),
        max_abs_delta_b_fast_t=float(np.max(np.abs(fast_after-fast_before))),
        max_abs_delta_a_t_m=float(np.max(np.abs(after["A_t_m"]-before["A_t_m"]))),
        peak_sampled_b_t=float(np.max(norm)),
        provenance_identity_distinct=old.digest != new.digest), before, after, fast_before, fast_after


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    protected_before = protected_hashes()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    switch_mode(state, "unified")
    condenser = condenser_mode_key(state)
    projector = "imaging" if state.projector_mode == "image" else "diffraction"
    assert condenser == "nano_probe" and projector == "diffraction"
    apply_operating_mode_pair(state, condenser, projector,
        column_name=selection.column, recording_name=selection.recording)
    from temsim.column.state_layout import apply_physical_layout_to_state
    apply_physical_layout_to_state(state)
    state.step_mm = .05
    baseline = state.to_dict()
    dump(output/"standard-startup-state.json", baseline)
    assembly = state._resolved_assembly
    parts = {p.key:p for p in assembly.parts}
    inventory, equivalence, arrays = [], [], {}
    owners = {p.key for p in assembly.parts if p.data.get("mechanical_profile")
              in ("magnetic_lens_assembly", "integrated_magnetic_lens_channel")}
    enabled = {lens.key for lens in state.lenses if lens.enabled and lens.key in owners}
    assert len(enabled) == len(owners) == 13
    for lens in state.lenses:
        if lens.key not in enabled:
            continue
        key, owner = lens.key, parts[lens.key]
        owned = [p for p in assembly.parts if p.data.get("mechanical_profile") == "magnetic_excitation_coil"
                 and p.data.get("field_source_key") == key]
        assert len(owned) == 1 and owned[0].data["coil_ampere_turn_fraction"] == 1.
        winding = owned[0]
        native = state.condenser_system[key] if key in CONDENSER_LENS_KEYS else lens
        on, percent, polarity = _runtime_excitation(native)
        legacy = capture_round_lens_potential(detached(state), key).potential
        intervals = current_intervals_mm(winding.data, owner.data)
        occupied_length = math.fsum(b-a for a,b in intervals)
        captured_length = math.fsum(c.length_m*1000 for c in legacy.coils)
        assert math.isclose(captured_length, occupied_length, rel_tol=0, abs_tol=1e-10)
        missing = [name for name,data in (("winding_turn_count",winding.data),
            ("winding_wire_diameter_mm",winding.data), (OWNER_CURRENT_FIELD,owner.data)) if name not in data]
        inventory.append(dict(owner=key, label=LABELS[key], winding=winding.key,
            source_module=owner.module_key, enabled=on, excitation_percent=percent, polarity=polarity,
            inner_diameter_mm=winding.data["mechanical_inner_diameter_mm"],
            outer_diameter_mm=winding.data["mechanical_outer_diameter_mm"],
            occupied_axial_length_mm=occupied_length,
            local_current_intervals_mm=intervals,
            global_current_intervals_mm=[[(c.center_z_m-c.length_m/2)*1000,
                                         (c.center_z_m+c.length_m/2)*1000] for c in legacy.coils],
            child_display_length_mm=winding.end_z_mm-winding.start_z_mm,
            turn_count=winding.data.get("winding_turn_count"),
            wire_diameter_mm=winding.data.get("winding_wire_diameter_mm"),
            current_at_100_percent_a=owner.data.get(OWNER_CURRENT_FIELD),
            current_a=None, missing_fields=missing,
            reference_ni_a_turn=owner.data["air_core_reference_ampere_turns"],
            reference_status=owner.data["air_core_reference_status"],
            actual_ideal_ni_a_turn=math.fsum(c.ampere_turns for c in legacy.coils),
            captured_ideal_source=legacy.to_dict()))
        points, domain = sample_positions(legacy)
        comparisons = []
        for name,turns,wire in (("explicit",100,.25),("same_nominal_ni_different_n_i",200,.125),
                                ("same_nominal_ni_different_wire",100,.125)):
            # Synthetic test-only decomposition of the same reference NI.
            current = owner.data["air_core_reference_ampere_turns"]/turns
            candidate = detached(state)
            edit(candidate, {key:{OWNER_CURRENT_FIELD:current}, winding.key:{
                "winding_turn_count":turns, "winding_wire_diameter_mm":wire}})
            candidate_parts = {p.key:p.data for p in candidate._resolved_assembly.parts}
            layout = resolve_round_winding_geometry(candidate_parts[winding.key], candidate_parts)
            assert len(layout.layout.turns) == turns
            new = capture_round_lens_potential(candidate,key).potential
            metrics,before,after,fast_before,fast_after = compare(legacy,new,points,require_pass=False)
            # Explicit controlled operator check at *exactly* equal segment NI.
            # Retain and report the unmodified adapter result separately above.
            matched = replace(new, coils=tuple(replace(c,ampere_turns=reference.ampere_turns)
                for c,reference in zip(new.coils,legacy.coils)))
            aligned,_,aligned_after,_,_ = compare(legacy,matched,points,require_pass=True)
            assert aligned["coefficients_bitwise_equal"]
            comparisons.append(dict(case=name, synthetic_turns=turns,
                synthetic_wire_diameter_mm=wire, synthetic_i100_a=current,
                actual_synthetic_current_a=current*percent*.01*polarity,
                layout_valid=True, raw_adapter_metrics=metrics,
                controlled_equal_segment_ni_metrics=aligned,
                captured_source=new.to_dict()))
            arrays[f"{key}__{name}__B_t"] = after["B_t"]
            arrays[f"{key}__{name}__fast_B_t"] = fast_after
            arrays[f"{key}__{name}__A_t_m"] = after["A_t_m"]
            arrays[f"{key}__{name}__controlled_equal_NI_B_t"] = aligned_after["B_t"]
        arrays[f"{key}__points_m"] = points
        arrays[f"{key}__legacy_B_t"] = before["B_t"]
        arrays[f"{key}__legacy_fast_B_t"] = fast_before
        arrays[f"{key}__legacy_A_t_m"] = before["A_t_m"]
        equivalence.append(dict(owner=key, sample_domain=domain, comparisons=comparisons))
        print(key, len(points), "exact NI passed", "raw adapter", comparisons[0]["raw_adapter_metrics"]["passed"], flush=True)
    assert baseline == state.to_dict(), "Default operating state was mutated"
    protected_after = protected_hashes()
    assert protected_before == protected_after, "Source or input definition changed during audit"
    np.savez_compressed(output/"field-samples.npz", **arrays)
    dump(output/"inventory.json", inventory)
    dump(output/"equivalence.json", equivalence)
    dump(output/"protected-files.json", protected_before)
    metrics = [c["controlled_equal_segment_ni_metrics"] for e in equivalence for c in e["comparisons"]]
    nominal = [dict(owner=e["owner"], case=c["case"], **c["raw_adapter_metrics"])
               for e in equivalence for c in e["comparisons"]]
    summary = dict(status="equal_ni_operator_passed_raw_adapter_has_roundoff_failures", scope="Component field arrays and CAD packing only; no beam or wave propagation",
        baseline="Standard GUI startup after stored Nanoprobe/Diffraction presets; no saved-result restore",
        selected_modules=dict(assembly.selected_module_paths),
        default_simulation_mode=state.simulation_mode, default_round_lens_field_model=state.round_lens_field_model,
        tested_round_lenses=len(equivalence), cases=len(metrics),
        unique_sample_positions=sum(e["sample_domain"]["points"] for e in equivalence),
        exact_ni_point_case_evaluations=sum(e["sample_domain"]["points"]*3 for e in equivalence),
        raw_adapter_diagnostics=nominal,
        raw_adapter_failed_lenses=sorted({m["owner"] for m in nominal if not m["passed"]}),
        raw_adapter_failed_cases=sum(not m["passed"] for m in nominal),
        raw_adapter_max_abs_delta_b_t=max(m["max_abs_delta_b_t"] for m in nominal),
        raw_adapter_max_vector_relative_delta_b=max(m["max_vector_relative_delta_b"] for m in nominal),
        raw_adapter_max_abs_delta_ni_a_turn=max(m["max_abs_delta_ni_a_turn"] for m in nominal),
        criterion=dict(rtol=RTOL, atol_b_t=ATOL_T, source="tests/test_explicit_round_winding_field.py"),
        max_abs_delta_b_t=max(m["max_abs_delta_b_t"] for m in metrics),
        max_vector_relative_delta_b=max(m["max_vector_relative_delta_b"] for m in metrics),
        max_abs_delta_b_fast_t=max(m["max_abs_delta_b_fast_t"] for m in metrics),
        max_abs_delta_ni_a_turn=max(m["max_abs_delta_ni_a_turn"] for m in metrics),
        baseline_state_unchanged=True, protected_input_source_files_unchanged=True,
        protected_files=len(protected_before),
        synthetic_inputs_persisted_to_default=False)
    dump(output/"summary.json", summary)
    dump(output/"manifest.json", {p.name:dict(bytes=p.stat().st_size, sha256=sha256(p.read_bytes()).hexdigest())
        for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("This component audit must not emit or transport a source")
    with patch("temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit", forbidden), \
            patch("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden):
        run(args.output.resolve())
