"""Measure and retune actual gun-exit particles, including mechanical interception."""

import argparse
import json

import numpy as np
from column_current import (
    ApertureTracker,
    C2_FOIL,
    OBJECTIVE_NI,
    OUT,
    SURFACE,
    baseline,
    current_ledger,
    field_array,
    fields,
    metrics,
    source_digest,
    source_rate,
)
from observe import axis_crossings
from tune import waist_proposal


def bundle(step, obs, ni, n, current, table_file=None, *, transverse_interpolation="linear"):
    mesh, table, bg, coils = fields(step, obs, ni, table_file)
    tracker = ApertureTracker(mesh, table, transverse_interpolation)
    size = len(tracker.ids)
    if not (1000 <= n <= size) or not np.all(tracker.weights == 1 / size):
        raise ValueError("Expected a nested, equal-probability original FEG population")
    # Prefix normalisation is the declared emission quadrature before any contact.
    tracker.input, tracker.de, tracker.ids = (a[:n] for a in (tracker.input, tracker.de, tracker.ids))
    tracker.weights = np.full(n, 1 / n)
    tracker.current = current
    return mesh, tracker, bg, coils


def survey(bundle, q):
    mesh, t, bg, coils = bundle
    data, loss, _, _ = t.run(field_array(mesh, bg, coils, q), keep_history=False)
    live = loss == -1
    row = dict(
        emitted=len(t.ids),
        survived=int(live.sum()),
        effective=int(live.sum()),
        ledger=current_ledger(mesh, t, loss),
    )
    if live.sum() < 2:
        return row
    p, w = data[-1, live], t.weights[live]
    d, r = waist_proposal(p, w)
    row.update(metrics(p, w, t.current), drift_D95_m=d, drift_RMS_m=r)
    return row


def fit(b, q):
    directions = np.zeros((len(q), 4))
    directions[4:8, 0] = np.maximum(abs(q[4:8]), 20) * np.sign(q[4:8])
    directions[8:12, 1] = np.maximum(abs(q[8:12]), 4) * np.sign(q[8:12])
    directions[18:21, 2] = np.maximum(abs(q[18:21]), 150) * np.sign(q[18:21])
    directions[15:18, 3] = np.maximum(abs(q[15:18]), 100) * np.sign(q[15:18])

    def error(row):
        return np.array([(row["alpha95_mrad"] - 30) / 30, row["drift_D95_m"] / 0.003])

    row = survey(b, q)
    for _ in range(14):
        err = error(row)
        if abs(err[0]) < 2e-6 and abs(row["drift_D95_m"]) < 0.5e-9:
            break
        sj = max(4e-8, min(0.00012, np.linalg.norm(err) * 0.0001))
        jac = []
        for d in directions.T:
            trial = survey(b, q + d * sj)
            jac.append((error(trial) - err) / sj)
        dx = np.linalg.lstsq(np.array(jac).T, -err, rcond=1e-8)[0]
        dx *= min(1.0, 0.055 / max(abs(dx).max(), 1e-12))
        for scale in (1.0, 0.5, 0.2, 0.06, 0.015):
            proposed = q + directions @ (dx * scale)
            trial = survey(b, proposed)
            if np.linalg.norm(error(trial)) < np.linalg.norm(err):
                q, row = proposed, trial
                break
        else:
            break
    return q, row


def axial(q, ni, n, step, rate, name, lens_centres, table_file=None, output_dir=OUT, extra_offsets_nm=(),
          *, transverse_interpolation="linear"):
    offsets = np.unique(
        np.r_[np.arange(-300000.0, 300001.0, 1000.0), np.arange(-100.0, 101.0, 2.0),
              -0.5, 0.0, 0.5, extra_offsets_nm]
    )
    near = SURFACE + offsets * 1e-9
    boundary_nodes = (lens_centres[:, None] + np.arange(-40, 41) * 0.000025).ravel()
    obs = np.unique(
        np.r_[
            np.arange(0.645, SURFACE - 0.00031, 0.00075),
            near,
            C2_FOIL,
            boundary_nodes,
            lens_centres,
            SURFACE - 0.00025,
        ]
    )
    m, t, bg, cs = bundle(step, obs, ni, n, rate, table_file, transverse_interpolation=transverse_interpolation)
    f = field_array(m, bg, cs, q)
    t.run(f, keep_history=False, surface_index=np.searchsorted(m.z, SURFACE), record_probe=True)
    live = t.surface_loss == -1
    sample = dict(metrics(t.surface_phase[live], t.weights[live], t.current))
    row = dict(
        surface=sample,
        current_ledger=current_ledger(m, t, t.surface_loss),
        currents_A=q.tolist(),
        objective_At=ni,
        source_digest=source_digest(t),
        emitted=n,
        survived=int(live.sum()),
        effective=int(live.sum()),
        max_step_mm=step,
        phase_file=name + ".npz",
    )
    save = dict(
        z_m=obs,
        offsets_nm=offsets,
        moments=t.moments,
        profiles=t.probes,
        phase=t.surface_phase,
        loss=t.surface_loss,
        ids=t.ids,
        weights=t.weights,
        energy_offset_eV=t.de,
    )
    loss_owner = np.full(len(t.surface_loss), -1, dtype=np.int32)
    loss_owner[~live] = m.owner[t.surface_loss[~live]]
    save.update(loss_owner=loss_owner, component_names=m.keys)
    row["C2_foil"] = {}
    for label, face in zip(("entrance", "exit"), C2_FOIL):
        i = np.searchsorted(obs, face)
        row["C2_foil"][label] = dict(
            z_m=face,
            current_A=float(t.moments[i, 0] * t.current),
            D95_diameter_um=float(t.probes[i, 0] * 1e-3),
        )
    near_ix = np.searchsorted(obs, near)
    for key, data in [("D95", t.probes[:, 0]), ("RMS", t.probes[:, 1])]:
        i = data[near_ix].argmin()
        row[key + "_minimum_bracket_nm"] = offsets[max(0, i - 1) : i + 2].tolist()
    c2, c3, adl, mini = m.intervals
    bands = [
        ("C2_centre_window", c2 - 0.001, c2 + 0.001),
        ("C2_to_C3", c2, c3),
        ("C3_to_ADL", c3, adl),
        ("Mini_to_pre_sample", mini, SURFACE - 0.00025),
        ("surface_window", SURFACE - 0.00025, SURFACE + 0.00025),
    ]

    def waists(moments):
        return {k: axis_crossings(obs, moments, lo, hi) for k, lo, hi in bands}

    row["waists_mm"] = waists(t.moments)
    if not np.all(live):
        t.input, t.de, t.ids, t.weights = (a[live] for a in (t.input, t.de, t.ids, t.weights))
        if len(t.ids) < 1000:
            raise ValueError("Insufficient common cohort for an independently replayed waist")
        t.run(f, keep_history=False, record_probe=True)
        row["common_waists_mm"] = waists(t.moments)
        save["common_moments"] = t.moments
    else:
        row["common_waists_mm"] = row["waists_mm"]
    np.savez_compressed(output_dir / (name + ".npz"), **save)
    return row


def main(args):
    rate = source_rate()
    if args.seed:
        item = json.loads(args.seed.read_text())
        candidates = item if isinstance(item, list) else [item]
    else:
        candidates = [dict(currents_A=baseline().tolist())]
    if args.index is not None:
        candidates = [candidates[args.index]]
    if args.observe and len(candidates) != 1:
        raise ValueError("Select one current setting with --index for a named axial archive")
    bundles = {}
    results = []
    path = OUT / (args.name + ".json")
    for j, seed in enumerate(candidates):
        q = np.array(seed["currents_A"])
        if q.shape != (21,) or not np.isfinite(q).all():
            raise ValueError("Expected signed amperes for the 21 registered winding groups")
        ni = seed.get("objective_At", OBJECTIVE_NI) if args.ni is None else args.ni
        if args.ni is not None and seed.get("objective_At", ni) != ni:
            raise ValueError("The requested objective ampere-turns differ from the stored setting")
        if ni not in bundles:
            bundles[ni] = bundle(args.step, [SURFACE], ni, args.count, rate)
        b = bundles[ni]
        try:
            if args.fit:
                q, row = fit(b, q)
            else:
                row = survey(b, q)
        except ValueError as exc:
            row = dict(failed=str(exc))
        row.update(
            currents_A=q.tolist(),
            objective_At=ni,
            seed=(args.index or 0) + j,
            max_step_mm=args.step,
            source_digest=source_digest(b[1]),
        )
        if args.observe and "failed" not in row:
            row.update(axial(q, ni, args.count, args.step, rate, args.name, b[0].intervals))
        results.append(row)
        path.write_text(json.dumps(results, indent=2))
        measured = row.get("surface", row)
        display = {
            key: measured[key] for key in ("transmitted_uA", "alpha95_mrad", "D95_nm") if key in measured
        }
        for key in ("survived", "D95_minimum_bracket_nm", "RMS_minimum_bracket_nm", "failed"):
            if key in row:
                display[key] = row[key]
        print(row["seed"], display, flush=True)


if __name__ == "__main__":
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--seed", type=Path)
    p.add_argument("--index", type=int)
    p.add_argument("--count", type=int, default=32768)
    p.add_argument("--step", type=float, default=0.2)
    p.add_argument("--ni", type=int, help="Signed objective ampere-turns; defaults to the stored setting")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--observe", action="store_true")
    main(p.parse_args())
