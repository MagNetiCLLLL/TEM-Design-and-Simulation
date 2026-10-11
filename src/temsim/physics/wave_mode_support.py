"""Bounded omission of independent positive-weight modes, never amplitudes.

For a mixture rho=sum(w_i |psi_i><psi_i|), the omitted positive operator has
trace sum(w_i). A common passive, linear quantum channel cannot increase this
trace; any downstream intensity measurement therefore has an integrated L1
error no larger than that omitted probability per tip electron. This is not a
pointwise relative error bound, nor a bound for coherently adding amplitudes.
"""
from collections.abc import Mapping
from dataclasses import replace
import math

from temsim.immutable_json import json_digest
from temsim.physics.wave_flux import BeamState


MODE_SUPPORT_SCHEMA = "bounded-independent-mode-omission-v1"


def incoherent_mode_approximation(record):
    """Find the single admission record through column/material wrappers."""
    while isinstance(record, Mapping):
        evidence = record.get("incoherent_mode_approximation")
        if evidence is not None:
            if evidence.get("schema") != MODE_SUPPORT_SCHEMA:
                raise ValueError("Unknown independent-mode approximation record")
            return evidence
        record = record.get("upstream")
    return None


def select_incoherent_modes(checkpoint, *, maximum_loss_probability):
    """Select weakest complete modes once under one total probability budget.

    Source settings and the complete gun result are untouched. All retained
    mode objects, absolute weights, reference current, phases and coordinates
    are unchanged. At least the largest-weight mode always survives. A zero
    budget retains every mode, including zero-weight entries.
    """
    budget = maximum_loss_probability
    if isinstance(budget, bool) or not math.isfinite(budget) or not 0 <= budget <= .1:
        raise ValueError("Independent-mode loss budget must be finite and between zero and 0.1")
    budget = float(budget)
    prior = incoherent_mode_approximation(checkpoint.record)
    if prior is not None:
        if prior["budget_probability_per_tip_electron"] != budget:
            raise ValueError("Independent-mode budget changed after selection; restart from the complete gun checkpoint")
        return checkpoint

    beam = checkpoint.beam
    modes = beam.modes
    # The executed store already validates its manifest and BeamState contract.
    # Reading metadata avoids loading all large arrays just to choose indices.
    stored = hasattr(beam, "content_identity") and hasattr(modes, "rows")
    rows = ([{"mode_id": row["mode_id"], "weight_per_reference_electron": float(row["weight"]),
              "energy_kev": float(row["energy_kev"])} for row in modes.rows]
            if stored else [{"mode_id": mode.mode_id,
                "weight_per_reference_electron": float(mode.weight_per_reference_electron),
                "energy_kev": float(mode.energy_kev)} for mode in modes])
    if not rows or any(not math.isfinite(row["weight_per_reference_electron"])
                       or row["weight_per_reference_electron"] < 0 for row in rows):
        raise ValueError("Independent modes require finite nonnegative probability weights")
    if len({row["mode_id"] for row in rows}) != len(rows):
        raise ValueError("Independent modes require unique identities")
    largest = max(range(len(rows)), key=lambda i: rows[i]["weight_per_reference_electron"])
    omitted = []
    weights = []
    if budget:
        for index in sorted(range(len(rows)), key=lambda i: (rows[i]["weight_per_reference_electron"], rows[i]["mode_id"])):
            if index == largest:
                continue
            proposed = math.fsum((*weights, rows[index]["weight_per_reference_electron"]))
            # Round the reported positive trace bound upward, rather than
            # spending a fraction more than the caller's total budget.
            bound = math.nextafter(proposed, math.inf) if proposed else 0.
            if bound > budget:
                break
            omitted.append(index)
            weights.append(rows[index]["weight_per_reference_electron"])
    omitted_indices = frozenset(omitted)
    kept = tuple(i for i in range(len(rows)) if i not in omitted_indices)
    probability = math.fsum(weights)
    evidence = {"schema": MODE_SUPPORT_SCHEMA,
        "budget_probability_per_tip_electron": budget,
        "input_mode_count": len(rows), "retained_mode_count": len(kept),
        "omitted_modes": [rows[i] for i in omitted],
        "omitted_probability_per_tip_electron": probability,
        "intensity_l1_error_bound_per_tip_electron": math.nextafter(probability, math.inf) if probability else 0.,
        "input_probability_per_tip_electron": math.fsum(row["weight_per_reference_electron"] for row in rows),
        "retained_probability_per_tip_electron": math.fsum(rows[i]["weight_per_reference_electron"] for i in kept),
        "renormalised": False,
        "scope": "Independent positive-weight mixture; integrated absolute intensity error after passive linear propagation, not local relative error or coherent amplitude error"}
    if stored:
        selected_beam = replace(beam, modes=replace(modes, rows=tuple(modes.rows[i] for i in kept)),
            content_identity=json_digest({"upstream": beam.content_identity, "mode_selection": evidence}))
    else:
        selected_beam = BeamState(tuple(modes[i] for i in kept), beam.reference_plane)
    record = {"schema": "executed-incoherent-mode-support-v1", "upstream": checkpoint.record,
              "upstream_digest": checkpoint.digest, "incoherent_mode_approximation": evidence}
    if "radial_output_modes" in checkpoint.record:
        kept_ids = {rows[i]["mode_id"] for i in kept}
        record["radial_output_modes"] = tuple(row for row in checkpoint.record["radial_output_modes"]
                                             if row["mode_id"] in kept_ids)
    return replace(checkpoint, beam=selected_beam, record=record)
