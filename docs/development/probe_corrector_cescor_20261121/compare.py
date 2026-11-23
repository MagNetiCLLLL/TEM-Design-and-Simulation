"""Matched finite-source test of a fixed four-winding FEM field in the reference column."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
import sys
from tempfile import NamedTemporaryFile

import cupy as cp
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "objective_pole_study_20261119"
sys.path.insert(0, str(PREVIOUS))
from column import GROUPS, Mesh, POINT, START, SURFACE, metrics
from particles import Tracker
from branches import inputs, one
from tune import waist_proposal
from observe import axis_crossings

ARCHIVE = HERE.parent / "material_objective_si110_20261120/evidence"
WORK = ROOT / "outputs/cescor-20261121"
NI = -4500.0


def shared_inputs(ni=NI):
    from column import prepare
    from measure import desktop_state
    from importlib.util import module_from_spec, spec_from_file_location
    from temsim.physics.potential_geometry import capture_potential_geometry
    from temsim.physics.unified_particle_contacts import compile_contact_geometry

    target = WORK / "column"
    target.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs/material_objective_si110_20261120"
    for name in ("columns.npz", "electric.npz"):
        if not (target / name).is_file() and (source / name).is_file():
            copyfile(source / name, target / name)
    if not all((target / name).is_file() for name in ("columns.npz", "electric.npz")):
        prepare(target, target / "replay-basis.npz")
    spec = spec_from_file_location("reference_objective_fields", ARCHIVE.parent / "solve_fields.py")
    objective = module_from_spec(spec)
    spec.loader.exec_module(objective)
    study = json.loads((ARCHIVE / "study.json").read_text())
    if objective.electrostatic_grid_digest(target / "electric.npz") != study["electrostatic_grid_sha256"]:
        raise ValueError("The reference electrostatic potential values or grid have changed")
    pilot_field = target / "armco-4500-gap-fine-p8.npz"
    if not pilot_field.is_file():
        copyfile(ARCHIVE / "after/objective-field.npz", pilot_field)
    if not (target / f"armco-{abs(ni):.12g}-gap-fine-p8.npz").is_file():
        objective.CACHE = target
        objective.solve(abs(ni), "gap-fine", 8)
    state = desktop_state(32768)[1]
    geom = compile_contact_geometry(capture_potential_geometry(state, START, SURFACE + 0.0004))
    keys = np.array([r[0] for r in geom.rows])
    positions = {p.key: p.z_mm for p in state.lenses}
    if not (
        geom.native.all()
        and not geom.origins_mm[:, :2].any()
        and np.isclose(geom.radii_mm[keys == "condenser_aperture_2"].min(), 0.05)
    ):
        raise ValueError("Expected centered native interception and a 100 micrometre C2 aperture")
    with NamedTemporaryFile(dir=target, suffix=".npz", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        np.savez(
            temporary,
            lo=geom.axial_lower_m,
            hi=geom.axial_upper_m,
            r=0.001 * geom.radii_mm,
            names=keys,
            intervals=0.001
            * np.array([positions[k] for k in (GROUPS[0], GROUPS[1], "adapter_lens", GROUPS[2])]),
        )
        temporary.replace(target / "mechanical.npz")
    finally:
        temporary.unlink(missing_ok=True)
    return target


class CircuitTracker(Tracker):
    """Add all three FEM components at every Runge-Kutta stage of the archived tracker."""

    def __init__(self, mesh, table, *, source=None):
        super().__init__(mesh, source=source)
        xy, z, field = (table[k] for k in ("xy_m", "z_local_m", "B_zyx_T"))
        zv = mesh.nodes - START
        if zv.min() < z[0] - 1e-12 or zv.max() > z[-1] + 1e-12:
            raise ValueError("Particle path exceeds corrector field registration")
        iz = np.clip(np.searchsorted(z, zv, side="right") - 1, 0, len(z) - 2)
        frac = (zv - z[iz]) / (z[iz + 1] - z[iz])
        self.tables = [cp.asarray(a) for a in (field, iz.astype(np.int32), frac)]
        helper = (HERE / "field_lookup.cu").read_text()
        lookup = f"add_corrector(node,x,y,{len(xy)},{xy[0]:.17g},{1/(xy[1]-xy[0]):.17g},{float(table['valid_radius_m']):.17g},bx,by,bz CORRECTOR_FIELD_ARGUMENTS);"
        original = (PREVIOUS / "lorentz.cu").read_text()
        anchor = "int lo=0,hi=nr-1;"
        if original.count(anchor) != 1:
            raise RuntimeError("Reference integrator no longer matches the field hook")
        self.install_field_kernel(helper + original.replace(anchor, lookup + anchor))

    def install_field_kernel(self, source):
        self.module = cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"))
        self._trace_kernel = self.module.get_function("trace")


def setup(table, step, obs, cache, ni=NI):
    m = Mesh(step, obs, cache)
    m.reference_pencil = inputs(m, "armco", NI, "gap-fine", 8)
    # Remove the archived two TL current patterns and both prescribed HP fields.
    m.group[0] -= m.transfers[1] + m.transfers[2]
    m.transfers[1:3] = 0
    m.hex.fill(0.0)
    return m, CircuitTracker(m, table), m.base("armco", ni, "gap-fine", 8)


def pencil(mesh, table, q0, ni=NI):
    # Continue the old first-order map onto the new circuit to seed a physical
    # search. Current, angle and focus acceptance are calculated with particles.
    old_par = mesh.reference_pencil
    par = inputs(mesh, "armco", ni, "gap-fine", 8)
    field = par[0]
    idx = table["xy_m"].size // 2
    bz = table["B_zyx_T"][:, idx, idx, 2]
    field[:, 0] += np.interp(mesh.nodes[1::2] - START, table["z_local_m"], bz)
    target = np.array(one(q0, *old_par)[:2])
    q = q0.copy()
    for t in np.linspace(0, 1, 21)[1:]:
        # A different axial node count is possible for through-focus observations.
        if len(field) != len(old_par[0]):
            raise ValueError("Use the specimen-only mesh for the pilot transfer pencil")
        evolving = (1 - t) * old_par[0] + t * field

        def cost(pair):
            trial = q.copy()
            trial[[0, 2]] = pair
            return (np.array(one(trial, evolving, *par[1:])[:2]) - target) / [350.0, 30.0]

        q[[0, 2]] = least_squares(cost, q[[0, 2]], bounds=(0.02, 6.0), ftol=2e-12, xtol=2e-12, gtol=2e-12).x
    return q, par


def measure(t, m, base, q):
    data, loss, peak, margin = t.run(m.field(base, q), keep_history=False)
    live = loss == -1
    if not live.any():
        raise ValueError("No source particles reached the specimen")
    p, w = data[-1, live], t.weights[live]
    result = metrics(p, w, t.current)
    d95, rms = waist_proposal(p, w)
    result.update(
        D95_drift_nm=d95 * 1e9,
        RMS_drift_nm=rms * 1e9,
        survivors=int(live.sum()),
        source_count=len(live),
        peak_mm=float(peak[live].max() * 1000),
        clearance_mm=float(margin[live].min() * 1000),
    )
    return result


def tune(t, m, base, q, out, max_iter):
    best = None
    trace = []

    def trial(x, journal=False):
        nonlocal best
        row = measure(t, m, base, x)
        err = np.array(
            [
                (row["alpha95_mrad"] - 30) / 30,
                row["D95_drift_nm"] / 1e7,
                3 * max(0, 9.0 - row["transmitted_uA"]) / 9,
            ]
        )
        score = float(err @ err)
        if best is None or score < best[0]:
            best = (score, x.copy(), row)
        if journal:
            trace.append(dict(controls=x.tolist(), **row))
            out.write_text(
                json.dumps(dict(history=trace, best=best[1].tolist(), metrics=best[2]), indent=2)
            )
            print(
                "alpha, D95 (nm), waist offset (nm), current (uA):",
                *[row[k] for k in ("alpha95_mrad", "D95_nm", "D95_drift_nm", "transmitted_uA")],
                flush=True,
            )
        return err

    for _ in range(max_iter):
        err = trial(q, True)
        if np.linalg.norm(err) < 1e-7:
            break
        knobs = (0, 2) if err[2] == 0 else (0, 1, 2, 3, 6)
        sj = max(3e-7, min(0.0004, np.linalg.norm(err) * 0.002))
        deriv = []
        for k in knobs:
            b = q.copy()
            b[k] += sj
            deriv.append((trial(b) - err) / sj)
        delta = np.linalg.lstsq(np.array(deriv).T, -err, rcond=1e-8)[0]
        delta *= min(1.0, 0.16 / max(1e-12, abs(delta).max()))
        for shrink in (1.0, 0.5, 0.2, 0.07, 0.025):
            b = q.copy()
            b[list(knobs)] += delta * shrink
            if min(b) > 0.01 and max(b) < 40 and np.linalg.norm(trial(b)) < np.linalg.norm(err):
                q = b
                break
        else:
            break
    trial(best[1], True)
    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step-mm", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--seed-case", help="Case name inside a published comparison.json")
    parser.add_argument("--objective-ni", type=float, default=NI)
    parser.add_argument("--observe", action="store_true")
    args = parser.parse_args()
    if args.seed_case and not args.seed:
        parser.error("--seed-case requires a comparison file passed with --seed")
    cache = shared_inputs(args.objective_ni)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.table) as data:
        table = dict(data)
    m, t, base = setup(table, args.step_mm, [SURFACE], cache, args.objective_ni)
    q0 = np.array(json.loads((ARCHIVE / "after/focus.json").read_text())["controls"])
    if args.seed:
        seed = json.loads(args.seed.read_text())
        if isinstance(seed, list):
            seed = seed[args.seed_index]
        elif args.seed_case:
            seed = seed["probes"][args.seed_case]
        q = np.array(seed.get("controls", seed.get("best")))
    else:
        q, _ = pencil(m, table, q0, args.objective_ni)
    print("Physical seed", q.tolist(), flush=True)
    q, result = tune(t, m, base, q, args.output.with_suffix(".trials.json"), args.iterations)
    row = dict(
        controls=q.tolist(),
        surface=result,
        table_sha256=sha256(args.table.read_bytes()).hexdigest(),
        current_floor_uA=9.0,
        C2_aperture_um=100.0,
        objective_A=args.objective_ni / 1160.0,
        physical_corrector_A=json.loads(str(table["report"]))["currents_A"],
        max_step_mm=args.step_mm,
        specimen_z_m=SURFACE,
    )
    args.output.write_text(json.dumps(row, indent=2))
    if args.observe:
        offsets = np.unique(
            np.r_[
                np.arange(-300000.0, 300001.0, 500.0),
                np.arange(-100.0, 100.1, 2.0),
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
            ]
        )
        fo = SURFACE + offsets * 1e-9
        obs = np.unique(np.r_[np.arange(0.645, SURFACE - 0.00031, 0.00075), fo])
        m, t, base = setup(table, args.step_mm, obs, cache, args.objective_ni)
        t.run(
            m.field(base, q),
            keep_history=False,
            record_probe=True,
            surface_index=np.searchsorted(m.z, SURFACE),
        )
        j = np.searchsorted(obs, fo)
        profiles = t.probes[j]
        live = t.surface_loss == -1
        row["surface"].update(metrics(t.surface_phase[live], t.weights[live], t.current))
        k = profiles[:, 0].argmin()
        row.update(
            D95_min_nm=float(profiles[k, 0]),
            D95_waist_bracket_nm=offsets[max(0, k - 1) : k + 2].tolist(),
            RMS_waist_offset_nm=float(offsets[profiles[:, 1].argmin()]),
            first_max_core_offset_nm=(
                float(offsets[profiles[:, 2].argmax()]) if profiles[:, 2].max() > 0 else None
            ),
            peak_core_nA=float(profiles[:, 2].max()),
            losses_near_focus=bool(np.ptp(t.moments[j, 0]) > 1e-13),
        )
        c2, c3, adl, mini = m.intervals
        row["xy_waists_mm"] = {
            name: axis_crossings(obs, t.moments, l, r)
            for name, l, r in (
                ("C2_to_C3", c2, c3),
                ("C3_to_ADL", c3, adl),
                ("Mini_to_sample", mini, SURFACE + 0.0003),
            )
        }
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            z_m=obs,
            focus_z_m=fo,
            profiles=profiles,
            moments=t.moments,
            phase=t.surface_phase,
            loss=t.surface_loss,
            ids=t.ids,
            weights=t.weights,
            energy_offset_eV=t.de,
        )
        args.output.write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
