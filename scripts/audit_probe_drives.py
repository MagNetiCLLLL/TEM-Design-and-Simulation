"""Audit effective field coordinates and their physical drive evidence.

Perturbations are detached field samples only. They do not propagate particles
or apply a tuning command to an accepted working point.
"""
from argparse import ArgumentParser
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reproduce_probe_baseline import load_baseline
from measure_crossover_cs import component_plane
from measure_probe_waist import _json, _npz
from temsim.instrument_snapshot import capture_instrument_snapshot, decode_instrument
from temsim.physics.unified_field_runtime import get_unified_field_runtime
from temsim.runtime_parameters import editable_parameters, runtime_targets


def channel_specs():
    rows = []
    for label, key in (("HP1", "probe_hp1_hexapole"), ("HP2", "probe_hp2_hexapole"),
                       ("HPC", "probe_hpc_hexapole"), ("HPol", "probe_hpol_hexapole")):
        for axis in ("normal", "skew"):
            rows.append((label+"_"+axis, key, axis, 1000., "m^-3"))
    for label, key in (("TL22", "probe_tl22_lens"), ("TL21", "probe_tl21_lens"),
                       ("TL12", "probe_tl12_lens"), ("ADL", "adapter_lens")):
        rows.append((label, key, "percent", .001, "percentage point"))
    rows.append(("UpperOL", "objective_lens", "upper_field_equivalent", .00002, "equivalent percentage point"))
    for label in ("QPH1", "QPH2", "QPC", "QPol"):
        rows.append((label, "probe_"+label.lower()+"_quadrupole", "strength_m2", .25, "m^-2"))
    for label in ("DPH1", "DPH2", "DP11", "DP21", "DP22"):
        for axis in ("x", "y"):
            rows.append((label+"_"+axis, "probe_"+label.lower()+"_deflector",
                         "kick_"+axis+"_mrad", .001, "mrad"))
    for axis in ("x", "y"):
        rows.append(("CStig_"+axis, "condenser_stigmator", "strength_"+axis+"_percent", .01, "percentage point"))
    return rows


def objects(state):
    return {v.key: v for v in (*state.corrector_elements, *state.lenses,
                               state.condenser_stigmator, state.objective_lens)}


def perturb(state, specification, sign):
    _, key, axis, step, _ = specification
    part = objects(state)[key]
    change = sign*step
    if axis in ("normal", "skew"):
        old = part.strength_m3*np.exp(3j*part.orientation_rad)
        new = old + change*(1 if axis == "normal" else 1j)
        part.strength_m3 = float(abs(new))
        part.orientation_rad = float(np.angle(new)/3)
    elif axis == "upper_field_equivalent":
        part.upper_b0_t *= 1+change/part.percent
    else:
        setattr(part, axis, getattr(part, axis)+change)


def normalized_rank(columns):
    matrix = np.column_stack(columns)
    lengths = np.linalg.norm(matrix, axis=0)
    if np.any(lengths == 0):
        raise ValueError("A declared effective drive produced no field response")
    singular = np.linalg.svd(matrix/lengths, compute_uv=False)
    return dict(rank=int(np.count_nonzero(singular > singular[0]*1e-8)),
                relative_threshold=1e-8, column_norms=lengths.tolist(),
                normalized_singular_values=singular.tolist())


def unchanged_interception_geometry(before, after):
    # The scene binding also hashes source metadata such as strength and
    # orientation. Compare the retained physical opening and boundary records.
    fields = ("schema", "scope", "start_z_m", "stop_z_m", "boundaries", "aperture_inventory")
    return all(before.to_dict()[key] == after.to_dict()[key] for key in fields)


def engineering_evidence(key, axis, part_data):
    host = part_data.get("physical_host_key")
    if axis == "upper_field_equivalent":
        return "Model calibration amplitude; separate upper-objective supply unverified"
    if axis in ("normal", "skew"):
        return "Two effective harmonics from strength and orientation; independent winding quadratures unverified"
    if "quadrupole" in key:
        return "One fixed-axis scalar field; independent skew drive absent from this model"
    if host:
        return "Modelled X/Y coil insert; electrical connectivity and ampere calibration unverified"
    if key == "condenser_stigmator":
        return "Two declared field axes; winding and supply map unavailable"
    return "Single effective excitation scale; ampere-to-field transfer unavailable"


def audit(input_folder, output):
    state, identity, metadata, _ = load_baseline(input_folder)
    output.mkdir(parents=True, exist_ok=False)
    original_runtime = get_unified_field_runtime(state)
    base = original_runtime.snapshot
    specs = channel_specs()
    parts = {p.key: p for p in state._resolved_assembly.parts}
    controlled = {row[1] for row in specs}
    source_ids = [c.source_id for c in base.contributions
                  if any(c.source_id == key or c.source_id.startswith(key+":") for key in controlled)]
    coords = objects(state)
    # Both radial and axial support are sampled. Column normalization makes
    # the rank independent of the arbitrary units of different drive families.
    z_values = [component_plane(state, coords[key]) for key in controlled]
    z_values.extend((state.objective_lens.upper_field_center_z_mm,
                     state.objective_lens.lower_field_center_z_mm))
    angles = np.arange(12)*2*np.pi/12
    xy = np.vstack([np.zeros((1, 2)), *(radius*np.c_[np.cos(angles), np.sin(angles)]
                                       for radius in (10e-6, 40e-6))])
    z_samples = np.unique(np.ravel(np.array(z_values)[:, None]+np.array([-3., 0., 3.])))
    points = np.vstack([np.c_[xy, np.full(len(xy), z*.001)] for z in z_samples])
    base_field = base.sample(points, source_ids=source_ids)["B_t"]
    inventory, columns, field_changes = [], [], {}
    for spec in specs:
        label, key, axis, step, unit = spec
        results = []
        changed_ids = []
        for sign in (-1, 1):
            candidate = decode_instrument(identity.graph)
            perturb(candidate, spec, sign)
            runtime = get_unified_field_runtime(candidate)
            if not unchanged_interception_geometry(original_runtime.geometry, runtime.geometry):
                raise ValueError("An effective drive changed the physical geometry")
            fields = {c.source_id: c.potential.digest for c in runtime.snapshot.contributions}
            before = {c.source_id: c.potential.digest for c in base.contributions}
            if fields.keys() != before.keys():
                raise ValueError("A drive introduced or removed a field provider")
            changes = sorted(k for k in fields if fields[k] != before[k])
            if not changes or any(not (k == key or k.startswith(key+":")) for k in changes):
                raise ValueError(f"{label} changed an undeclared field owner")
            changed_ids.append(changes)
            results.append(runtime.snapshot.sample(points, source_ids=source_ids)["B_t"])
            if axis == "upper_field_equivalent":
                old_terms = next(c.potential.native.terms_t_m for c in base.contributions if c.source_id == key)
                new_terms = next(c.potential.native.terms_t_m for c in runtime.snapshot.contributions if c.source_id == key)
                split = len(state.objective_lens.upper_gaussian)
                np.testing.assert_array_equal(old_terms[split:], new_terms[split:])
        delta = (results[1]-results[0])/2
        columns.append(delta.ravel())
        curvature = float(np.linalg.norm((results[0]+results[1])/2-base_field)/np.linalg.norm(delta))
        record = parts.get(key)
        data = {} if record is None else record.data
        target = runtime_targets(state).get(key)
        edit = [p.name for p in editable_parameters(target)] if target else []
        part = coords[key]
        inventory.append(dict(channel=label, component=key, coordinate=axis, step=step, unit=unit,
            field_sources=changed_ids[1], modelled_host=data.get("physical_host_key", ""),
            host_status=data.get("physical_host_status", "module component"),
            optical_reference_host=data.get("optical_reference_host_key", ""),
            evidence=engineering_evidence(key, axis, data), electrical_rank=None,
            winding_to_ampere_calibration="NOT_AVAILABLE", resolution="NOT_DECLARED",
            runtime_editable_fields=edit, limits={k:v for k,v in vars(part).items()
                if k in ("maximum_strength_m2", "maximum_strength_m3", "maximum_kick_mrad",
                         "max_percent", "max_strength_m2")},
            plus_minus_even_to_odd_ratio=curvature))
        field_changes[label] = changed_ids
    rank = normalized_rank(columns)
    h = np.arange(6)*2*np.pi/6
    ideal_rank = int(np.linalg.matrix_rank(np.c_[np.cos(3*h), np.sin(3*h)], tol=1e-12))
    source_list = [c.source_id for c in base.contributions]
    owner_list = [owner for c in base.contributions for owner in c.owner_ids if c.kind == "magnetic"]
    if len(source_list) != len(set(source_list)) or len(owner_list) != len(set(owner_list)):
        raise ValueError("Duplicate field source or magnetic owner")
    if capture_instrument_snapshot(state).digest != identity.digest:
        raise ValueError("Detached audit changed the accepted state")
    result = dict(schema="probe-drive-audit-v1", input_digest=identity.digest,
        original_input_digest=metadata["original_input_digest"], field_digest=base.numerical_digest,
        geometry_digest=original_runtime.geometry.digest, production_channels=inventory,
        model_field_rank=rank, physical_electrical_rank="NOT_IDENTIFIABLE_FROM_AVAILABLE_RECORDS",
        ideal_six_equal_poles_m3_rank=ideal_rank,
        ideal_example_scope="Demonstrates that six equally spaced single-family poles do not prove two quadratures",
        duplicate_magnetic_sources=False, physical_supply_count=None, controls_applied_to_baseline=False,
        particle_propagation=False, new_aberrations_tuned=False,
        mini_condenser=dict(installed="mini_condenser" in parts, separate_probe_tl11="probe_tl11_lens" in parts,
                            optical_equivalence_to_tl11="NOT_VERIFIED_BY_STAGE_2"),
        dp12=dict(part_present="probe_dp12_scan_deflector" in parts,
                  runtime_field_sources=[s for s in source_list if "dp12" in s or s.startswith("ac_deflector:")],
                  role="virtual reference", independent_supply=False,
                  command_mapping="AC X/Y command is distributed over the existing upper and lower scan coils"))
    _json(output / "audit.json", result)
    _npz(output / "field-response.npz", dict(points_m=points, delta_B_t=np.column_stack(columns),
                                          channel_names=np.asarray([s[0] for s in specs])))
    with (output / "drive-inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    print("Effective field rank", rank["rank"], "of", len(specs), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "docs/development/probe_stage12_20261101/inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.inputs.resolve(), args.output.resolve())
