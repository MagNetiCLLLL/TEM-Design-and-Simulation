"""Measure differential relay imaging and the physical C2 foil edge at the specimen."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tomllib

import cupy as cp
import numpy as np

from adjust_a2 import OUT
from column_current import (
    C2_FOIL,
    FACTOR,
    START,
    STUDY,
    SURFACE,
    current_ledger,
    field_array,
    metrics,
    source_digest,
)
from finite_probe import bundle
from temsim.physics.beam_statistics import transverse_beam_statistics
from tune import waist_proposal


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORK = ROOT / "outputs/cescor-relay-20261121"
WORK.mkdir(parents=True, exist_ok=True)
REF = json.loads((HERE / "evidence/focus-and-pupil.json").read_text(encoding="utf-8"))["model_runs"][
    "focus-audit-native"
]
REF["field_sha256"] = REF["corrector_sha256"]
assembly = json.loads((STUDY / "evidence/comparison.json").read_text(encoding="utf-8"))["fields"]["material"][
    "source_file"
]
PARTS = {
    p["key"]: p
    for p in tomllib.loads((ROOT / "configs/instruments/column" / assembly).read_text(encoding="utf-8"))[
        "parts"
    ]
}
HP2, TL22, TL21, HP1 = [
    START + PARTS[key]["local_center_z_mm"] * 0.001
    for key in ("probe_hp2_hexapole", "probe_tl22_lens", "probe_tl21_lens", "probe_hp1_hexapole")
]
FOIL = np.array(C2_FOIL)
RADIUS = 50e-6


def mechanical_span_mm(key):
    part = PARTS[key]
    center = START * 1000 + part['local_center_z_mm']
    half_length = part['length_mm'] / 2
    return center - half_length, center + half_length


def save(name, record):
    (WORK / (name + ".json")).write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def advance(t, field, phase, energy, phi0, loss, first, stops):
    """Propagate each input from its given plane with the native contact and energy rules."""
    p = cp.array(phase, dtype=float, order="C")
    en, ref, lost = cp.asarray(energy), cp.asarray(phi0), cp.array(loss, dtype=cp.int32)
    n = len(p)
    peak, margin = cp.zeros(n), cp.full(n, cp.inf)
    result = []
    for end in stops:
        for j in range(first, end, 160):
            t.kernel(
                ((n + 127) // 128,),
                (128,),
                (
                    p,
                    en,
                    ref,
                    field,
                    t.ep,
                    t.er,
                    np.int32(len(t.r2)),
                    t.h,
                    t.rad,
                    lost,
                    peak,
                    margin,
                    np.int32(j),
                    np.int32(min(j + 160, end)),
                    np.int32(n),
                    t.rest,
                    FACTOR,
                    t.pc,
                ),
            )
        first = end
        result.append((p.get(), lost.get()))
    if np.any(result[-1][1] < -1):
        raise ValueError("Field interpolation boundary reached before a physical contact")
    return result


def electric_reference(t):
    r2 = (t.input[:, :2] ** 2).sum(axis=1)
    ix = np.clip(np.searchsorted(t.r2, r2, side="right") - 1, 0, len(t.r2) - 2)
    return t.e0[ix, 0] + (r2 - t.r2[ix]) * t.e0[ix, 1]


class Native:
    def __init__(self, table, step=0.075, count=32768, ni=None, extra=()):
        self.path = Path(table)
        self.ni = REF["objective_At"] if ni is None else ni
        self.step = step
        planes = np.unique(np.r_[FOIL, HP2, HP1, SURFACE, extra])
        self.m, self.t, self.bg, self.coils = bundle(
            step, planes, self.ni, count, 15e-6, table, transverse_interpolation="cubic"
        )
        self.en = self.t.de + 300000.0
        self.phi0 = electric_reference(self.t)
        self.loss = np.full(count, -1, dtype=np.int32)
        self.rim_data = None

    def indices(self, zs):
        idx = np.searchsorted(self.m.z, zs)
        np.testing.assert_array_equal(self.m.z[idx], zs)
        return idx

    def field(self, q):
        return cp.asarray(field_array(self.m, self.bg, self.coils, q))

    def entrance(self, f, incoming=None):
        if incoming is None:
            index = self.indices(FOIL[:1])[0]
            (incoming,) = advance(self.t, f, self.t.input, self.en, self.phi0, self.loss, 0, [index])
        phase, lost = incoming
        r = np.linalg.norm(phase[:, :2], axis=1)
        # The original gun distribution is axisymmetric. Fit the conditional marginal
        # direction in the laboratory frame, retaining both the radial and Larmor terms.
        band = (lost == -1) & (r > 0.42 * RADIUS) & (r < RADIUS * 1.02)
        x = (phase[band, 0] + 1j * phase[band, 1]) / RADIUS
        t = phase[band, 2] + 1j * phase[band, 3]
        basis = np.column_stack((x, x * abs(x) ** 2))
        weights = self.t.weights[band]
        amplitude, _, rank, _ = np.linalg.lstsq(
            np.sqrt(weights[:, None]) * basis, np.sqrt(weights) * t, rcond=None
        )
        if rank != 2 or len(weights) < 6:
            raise ValueError("Too few illuminated foil radii to determine the marginal incident direction")
        residual = np.abs(t - basis @ amplitude)
        info = dict(
            fit_count=int(band.sum()),
            beyond_90pct_count=int(np.sum(band & (r > 0.9 * RADIUS))),
            largest_fitted_radius_um=float(abs(x).max() * RADIUS * 1e6),
            fit_rms_urad=float(1e6 * np.sqrt(np.average(residual**2, weights=weights))),
            fit_median_energy_eV=float(np.median(self.en[band])),
            fit_median_reference_V=float(np.median(self.phi0[band])),
        )
        return amplitude, info

    def edge(self, f, data, fraction=0.99999, azimuths=64):
        amp, info = data
        circle = np.exp(2j * np.pi * np.arange(azimuths) / azimuths)
        en = np.full(azimuths, info["fit_median_energy_eV"])
        phi = np.full(azimuths, info["fit_median_reference_V"])
        lost = np.full(azimuths, -1, dtype=np.int32)
        front, rear, surface = self.indices([*FOIL, SURFACE])

        def start(r):
            angle = (amp[0] + amp[1] * r * r) * r * circle
            x = RADIUS * r * circle
            return np.column_stack((x.real, x.imag, angle.real, angle.imag, np.zeros(azimuths)))

        # Find the entrance radius whose largest radius on the two faces approaches R
        # from inside. Approach from smaller radii so contacts on the foil stay active.
        radii = np.full(azimuths, 0.97)
        for _ in range(3):
            p, contact = advance(self.t, f, start(radii), en, phi, lost, front, [rear])[0]
            if np.any(contact != -1):
                raise ValueError("Unexpected foil contact while approaching the pupil from inside")
            envelope = np.maximum(radii, np.linalg.norm(p[:, :2], axis=1) / RADIUS)
            radii *= fraction / envelope
        (p, contact), (at_sample, lost) = advance(
            self.t, f, start(radii), en, phi, lost, front, [rear, surface]
        )
        live = lost == -1
        angle = 1e3 * np.arctan(np.linalg.norm(at_sample[live, 2:4], axis=1))
        names, numbers = np.unique(self.m.keys[self.m.owner[lost[~live]]], return_counts=True)
        return dict(
            rim_fraction=fraction,
            azimuths=azimuths,
            reached_sample=int(live.sum()),
            foil_contacts=int(np.count_nonzero(contact != -1)),
            intercepted=dict(zip(names.tolist(), numbers.tolist())),
            max_face_radius_um=float(
                np.maximum(radii * RADIUS, np.linalg.norm(p[:, :2], axis=1)).max() * 1e6
            ),
            angle_mrad_min_median_max=np.percentile(angle, [0, 50, 100]).tolist() if len(angle) else [],
            azimuth_deg=(360 * np.arange(azimuths) / azimuths).tolist(),
            angle_mrad=[
                float(1e3 * np.arctan(np.hypot(*p[2:4]))) if ok else None for p, ok in zip(at_sample, live)
            ],
        )

    def replay(self, q, zs):
        f = self.field(q)
        return f, advance(self.t, f, self.t.input, self.en, self.phi0, self.loss, 0, self.indices(zs))

    def measure(self, q, *, details=False):
        # Only controls downstream of the C2 foil are fixed between calls to entrance;
        # recompute it on every distinct setting to include winding fringe fields.
        f, (front, rear, (phase, loss)) = self.replay(q, [*FOIL, SURFACE])
        live = loss == -1
        p, w = phase[live], self.t.weights[live]
        m = metrics(p, w, self.t.current)
        m["axis_50nm_uA"] = float(1e6 * self.t.current * w[np.linalg.norm(p[:, :2], axis=1) <= 50e-9].sum())
        entering = front[1] == -1
        foil_stats = transverse_beam_statistics(*front[0][entering, :4].T, weights=self.t.weights[entering])
        dz, rms_dz = waist_proposal(p, w)
        rim_data = self.entrance(f, front)
        rim = self.edge(f, rim_data)
        row = dict(
            specimen=m,
            drift_D95_um=dz * 1e6,
            drift_rms_um=rms_dz * 1e6,
            rim=rim,
            entrance=rim_data[1],
            max_step_mm=self.step,
            foil_entrance_D95_um=2e6 * foil_stats.radius_95_m,
            survived=int(live.sum()),
            emitted=len(live),
            currents_A=np.asarray(q).tolist(),
            objective_At=self.ni,
            source_digest=source_digest(self.t),
            current_ledger=current_ledger(self.m, self.t, loss),
        )
        if details:
            footprint = np.maximum(
                np.linalg.norm(front[0][:, :2], axis=1), np.linalg.norm(rear[0][:, :2], axis=1)
            )
            row["physical_edge_cohorts"] = []
            for lower in (0.90, 0.95, 0.98):
                select = live & (footprint >= lower * RADIUS)
                angles = np.arctan(np.linalg.norm(phase[select, 2:4], axis=1)) * 1000
                row["physical_edge_cohorts"].append(
                    dict(
                        minimum_radius_fraction=lower,
                        packets=int(select.sum()),
                        current_nA=float(1e9 * self.t.current * self.t.weights[select].sum()),
                        mrad_min_median_max=(
                            np.percentile(angles, [0, 50, 100]).tolist() if len(angles) else []
                        ),
                    )
                )
            row["rim_sensitivity"] = self.edge(f, rim_data, 0.999, 128)
            row["rim_azimuth_check"] = self.edge(f, rim_data, 0.99999, 128)
            row["field_sha256"] = sha256(self.path.read_bytes()).hexdigest()
        return row, phase, loss

    def relay(self, q, planes):
        f, states = self.replay(q, planes)
        idx = self.indices(planes)
        result = {}
        for a, b in [(i, j) for i in range(len(idx)) for j in range(i + 1, len(idx))]:
            start, contact = states[a]
            matrices = []
            common = states[b][1] == -1
            replays = []
            for delta in (2e-6, 1e-6):
                slopes = []
                for col in (2, 3):
                    pair = []
                    for sign in (1, -1):
                        probe = start.copy()
                        probe[:, col] += sign * delta
                        ((p, lost),) = advance(
                            self.t, f, probe, self.en, self.phi0, contact, idx[a], [idx[b]]
                        )
                        common &= lost == -1
                        pair.append(p[:, :2])
                    slopes.append((pair[0] - pair[1]) / (2 * delta))
                replays.append(1e3 * np.stack(slopes, axis=2))
            original, _ = advance(self.t, f, start, self.en, self.phi0, contact, idx[a], [idx[b]])[0]
            np.testing.assert_allclose(original[common], states[b][0][common], rtol=0, atol=1e-14)
            large, small = [r[common] for r in replays]
            singular = np.linalg.svd(small, compute_uv=False)
            result[f"{a}-{b}"] = dict(
                z_mm=[1000 * planes[a], 1000 * planes[b]],
                common=int(common.sum()),
                matrix_mm_per_rad=small.mean(axis=0).tolist(),
                singular_values_p05_p50_p95_mm_per_rad=np.percentile(singular, [5, 50, 95], axis=0).tolist(),
                maximum_1_vs_2_urad_change_mm_per_rad=float(np.max(abs(small - large))),
            )
        return result

    def central_relay(self, f, planes):
        # Unrotate by the Larmor angle. The position block can rotate through
        # 90 degrees with respect to the slope block away from an image plane.
        indices = self.indices(planes)
        grid, iz, frac = self.t.tables
        mid = grid.shape[1] // 2
        bz = grid[:, mid, mid, 2]
        axial_b = f[:, 0] + (1 - frac) * bz[iz] + frac * bz[iz + 1]
        kin = float(np.median(self.en) - np.median(self.phi0)) + self.t.ep[:, 0, 0]
        k = (FACTOR * self.t.pc * axial_b / cp.sqrt(kin * (kin + 2 * self.t.rest))).get()
        larmor = np.r_[0.0, np.cumsum(np.diff(self.m.z) * (k[:-2:2] + 4 * k[1::2] + k[2::2]) / 6)]
        output = []
        for start, end in zip(indices, indices[1:]):
            p = np.zeros((9, 5))
            d = np.array([1e-7, 1e-7, 1e-6, 1e-6])
            p[1:5, :4] = np.diag(d)
            p[5:9, :4] = -np.diag(d)
            ((final, lost),) = advance(
                self.t,
                f,
                p,
                np.full(9, np.median(self.en)),
                np.full(9, np.median(self.phi0)),
                np.full(9, -1, dtype=np.int32),
                start,
                [end],
            )
            if np.any(lost != -1):
                raise ValueError("Lost a paraxial calibration ray")
            jac = ((final[1:5, :2] - final[5:9, :2]) / (2 * d[:, None])).T
            a, b = jac[:, :2], jac[:, 2:]
            theta = larmor[end] - larmor[start]
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            projected = np.trace(rotation.T @ b) / 2
            output.append(
                dict(
                    signed_B_mm=1000 * projected,
                    larmor_rotation_rad=theta,
                    B_mm_per_rad=(1000 * b).tolist(),
                    singular_values_mm_per_rad=(1000 * np.linalg.svd(b, compute_uv=False)).tolist(),
                    transverse_magnifications=np.linalg.svd(a, compute_uv=False).tolist(),
                )
            )
        return output


def match(model, q, start):
    # Two optical controls: C3 sets convergence and Mini sets focus. Recompute both
    # with actual fields and a fixed foil. No ray is refocused individually.
    def run(gains):
        proposal = q.copy()
        proposal[8:12] *= gains[0]
        proposal[18:21] *= gains[1]
        row, _, _ = model.measure(proposal)
        angles = row["rim"]["angle_mrad_min_median_max"]
        if len(angles) != 3 or row["rim"]["reached_sample"] != row["rim"]["azimuths"]:
            raise ValueError("Aperture rim is vignetted at the proposed setting")
        print(gains, angles[2], row["drift_D95_um"], row["specimen"]["D95_nm"], flush=True)
        return np.array([(angles[2] - 30) / 30, row["drift_D95_um"] / 3000])

    x = np.asarray(start, dtype=float)
    err = run(x)
    jac = None
    for i in range(14):
        if abs(err[0]) * 30 < 0.0008 and abs(err[1]) * 3000 < 0.003:
            return x
        if jac is None or i % 4 == 0:
            step = 6e-6
            jac = np.column_stack([(run(x + d) - err) / step for d in np.eye(2) * step])
        delta = np.linalg.solve(jac, -err)
        delta *= min(1.0, 0.04 / max(np.max(abs(delta)), 1e-15))
        new = run(x + delta)
        jac += np.outer(new - err - jac @ delta, delta) / np.dot(delta, delta)
        x, err = x + delta, new
    raise ValueError("Rim-angle and finite-source D95 focus did not converge")


def align_and_focus(model, q, planes, name, foil_d95_um=None):
    """Use a common native differential map, then check the unprojected imaging blocks."""
    groups = [slice(4, 8), slice(8, 12), slice(12, 15), slice(15, 18), slice(18, 21)]
    if foil_d95_um is not None:
        groups.insert(0, slice(0, 4))

    def measure(x):
        candidate = q.copy()
        for gain, group in zip(x, groups):
            candidate[group] *= 1 + gain
        central = model.central_relay(model.field(candidate), planes)
        row, _, _ = model.measure(candidate)
        rim = row["rim"]
        if rim["reached_sample"] != rim["azimuths"]:
            raise ValueError("Downstream contact obscured the marginal angle")
        errors = np.array(
            [
                central[0]["signed_B_mm"],
                central[1]["signed_B_mm"],
                rim["angle_mrad_min_median_max"][2] - 30.0,
                row["drift_D95_um"],
            ]
        )
        if foil_d95_um is not None:
            errors = np.r_[errors, row["foil_entrance_D95_um"] - foil_d95_um]
        return errors, candidate, central

    x = np.zeros(len(groups))
    err, cur, blocks = measure(x)
    scales = np.array([0.01, 0.01, 0.003, 0.15])
    if foil_d95_um is not None:
        scales = np.r_[scales, 0.02]
    # Modest steps at high original current prevent changing crossover branch.
    for k in range(22):
        score = np.linalg.norm(err / scales)
        print(k, "native residuals (mm, mm, mrad, um)", err, flush=True)
        save(name + "-input", dict(currents_A=cur.tolist(), objective_At=model.ni, errors=err.tolist()))
        if (
            max(v["singular_values_mm_per_rad"][0] for v in blocks) < 0.001
            and abs(err[2]) < 0.0005
            and abs(err[3]) < 0.004
            and (foil_d95_um is None or abs(err[4]) < 0.005)
        ):
            return cur
        d = 2e-6
        jac = np.column_stack([(measure(x + v * d)[0] - err) / d for v in np.eye(len(x))])
        step = np.linalg.lstsq(jac / scales[:, None], -err / scales, rcond=1e-10)[0]
        step *= min(1.0, 0.035 / max(1e-15, np.max(abs(step))))
        for fraction in (1.0, 0.5, 0.25, 0.1, 0.03):
            e, c, b = measure(x + fraction * step)
            if np.linalg.norm(e / scales) < score:
                x += fraction * step
                err, cur, blocks = e, c, b
                break
        else:
            break
    raise ValueError(f"Relay, edge and focus constraints did not converge. Residuals: {err.tolist()}")


def core_through_focus(path, q, ni, step, count):
    """Measure fixed-axis bright-core current through the real specimen focus."""
    offsets = np.arange(-250.0, 250.01, 0.5)
    z = SURFACE + offsets * 1e-6
    model = Native(path, step, count, ni, z)
    f = model.field(q)
    p = cp.asarray(model.t.input).copy()
    en, phi = cp.asarray(model.en), cp.asarray(model.phi0)
    loss = cp.asarray(model.loss)
    weights = cp.asarray(model.t.weights) * model.t.current * 1e9
    peak = cp.zeros(count)
    margin = cp.full(count, cp.inf)
    current = cp.empty(len(z))
    first = 0
    for j, end in enumerate(model.indices(z)):
        for i in range(first, end, 160):
            model.t.kernel(
                ((count + 127) // 128,),
                (128,),
                (
                    p,
                    en,
                    phi,
                    f,
                    model.t.ep,
                    model.t.er,
                    np.int32(len(model.t.r2)),
                    model.t.h,
                    model.t.rad,
                    loss,
                    peak,
                    margin,
                    np.int32(i),
                    np.int32(min(i + 160, end)),
                    np.int32(count),
                    model.t.rest,
                    FACTOR,
                    model.t.pc,
                ),
            )
        current[j] = cp.sum(weights * ((loss == -1) & (p[:, 0] ** 2 + p[:, 1] ** 2 <= (50e-9) ** 2)))
        first = end
    if bool(cp.any(loss < -1)):
        raise ValueError("Unresolved mechanical contact in through-focus trace")
    values = current.get()
    return dict(
        radius_nm=50.0,
        offset_um=offsets.tolist(),
        current_nA=values.tolist(),
        maximum_nA=float(values.max()),
        offsets_at_max_um=offsets[values == values.max()].tolist(),
        surface_nA=float(values[offsets == 0].item()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--configuration", choices=("before", "after"))
    parser.add_argument("--count", type=int, default=32768)
    parser.add_argument("--step", type=float, default=0.075)
    parser.add_argument("--coma-z-mm", type=float)
    adjustment = parser.add_mutually_exclusive_group()
    adjustment.add_argument("--match", nargs=2, type=float)
    parser.add_argument("--relay", action="store_true")
    adjustment.add_argument("--align-focus", action="store_true")
    parser.add_argument(
        "--foil-d95-um",
        type=float,
        help="Match the physical incoming foil D95 during the relay and focus fit",
    )
    parser.add_argument("--axial", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.settings.read_text()) if args.settings else REF
    if args.configuration:
        if not args.settings:
            parser.error("--configuration requires an archived --settings comparison")
        recipe = recipe[args.configuration]
        if args.coma_z_mm is None:
            args.coma_z_mm = recipe["relay"]["1-2"]["z_mm"][1]
    if args.table is None:
        if "field_file" in recipe:
            name = recipe["field_file"]
            args.table = next((d / name for d in (WORK, OUT) if (d / name).is_file()), WORK / name)
            if sha256(args.table.read_bytes()).hexdigest() != recipe["field_sha256"]:
                raise ValueError("Cached corrector field differs from the selected archived model")
        else:
            args.table = OUT / "a2-cubic-both-fine-field.npz"
    q = np.array(recipe["currents_A"])
    if args.core_only:
        save(args.case, core_through_focus(args.table, q, recipe["objective_At"], args.step, args.count))
        raise SystemExit(0)
    target = [HP2, HP1] + ([args.coma_z_mm * 0.001] if args.coma_z_mm else [])
    model = Native(args.table, args.step, args.count, recipe["objective_At"], target)
    if args.align_focus:
        if len(target) != 3:
            raise ValueError("Supply the independently located objective pupil")
        q = align_and_focus(model, q, target, args.case, args.foil_d95_um)
    if args.match:
        gains = match(model, q, args.match)
        q[8:12] *= gains[0]
        q[18:21] *= gains[1]
    report, phase, loss = model.measure(q, details=True)
    if args.relay:
        report["relay"] = model.relay(q, target)
        report["central_relay"] = model.central_relay(model.field(q), target)
    np.savez_compressed(
        WORK / (args.case + ".npz"),
        phase=phase,
        loss=loss,
        ids=model.t.ids,
        weights=model.t.weights,
        energy_offset_eV=model.t.de,
    )
    save(args.case, report)
    if args.axial:
        from finite_probe import axial

        record = axial(
            q,
            model.ni,
            args.count,
            args.step,
            15e-6,
            args.case + "-axial",
            model.m.intervals,
            args.table,
            WORK,
            extra_offsets_nm=np.arange(-500, 501, 10),
            transverse_interpolation="cubic",
        )
        save(args.case + "-axial", record)
    print(report["specimen"], report["rim"]["angle_mrad_min_median_max"], flush=True)
