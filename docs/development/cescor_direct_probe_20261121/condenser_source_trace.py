"""Transport the measured gun exit to ADL using fresh winding fields and registered material fields."""

import argparse
import json
from types import SimpleNamespace

import cupy as cp
import numpy as np
from scipy.interpolate import CubicHermiteSpline, make_interp_spline
from scipy.optimize import minimize_scalar

from column_current import ApertureTracker, C2_FOIL, POINT
from condenser_source_fields import EV, FEM, OUT, Fields, check_file, write
from condenser_material_binding import check_materials
from joint_relay_search import mono_roots
from relay_transfer import advance, electric_reference
from stationary_pupil import half_nodes
from temsim.physics.potential_geometry import capture_potential_geometry
from temsim.physics.unified_particle_contacts import compile_contact_geometry
from trace_normal_joint_field import beam_moments


def section(fields, step_mm):
    check_materials(fields.state)
    lo, hi = 0.45, 1.07
    hardware = compile_contact_geometry(capture_potential_geometry(fields.state, lo, hi))
    if not hardware.native.all() or np.any(hardware.origins_mm[:, :2]):
        raise ValueError("Expected centered, native interception along the condenser section")
    left, right = hardware.axial_lower_m, hardware.axial_upper_m
    names = np.array([r[0] for r in hardware.rows])
    radii = 0.001 * hardware.radii_mm
    foil = (names == "condenser_aperture_2") & np.isclose(radii, 50e-6)
    np.testing.assert_allclose([left[foil].item(), right[foil].item()], C2_FOIL, rtol=0, atol=1e-12)
    faces = np.r_[left, right]
    observe = (
        np.unique(
            np.round(
                np.r_[
                    np.arange(450.0, 750.0, 0.125),
                    np.arange(750.0, 1070.01, 0.25),
                    *np.array(C2_FOIL) * 1000,
                ],
                7,
            )
        )
        * 0.001
    )
    z = np.unique(
        np.round(
            np.r_[
                np.linspace(lo, hi, round((hi - lo) * 1000 / step_mm) + 1),
                faces[(faces > lo) & (faces < hi)],
                observe,
            ],
            13,
        )
    )
    mid = (z[1:] + z[:-1]) / 2
    radius = np.full(len(mid), np.inf)
    for a, b, r in zip(left, right, radii):
        ok = (mid >= a) & (mid <= b)
        radius[ok] = np.minimum(radius[ok], r)
    nodes = half_nodes(z)
    efield = fields.electric.base_field
    zz, rr, v = efield.z, efield.r, efield.voltage
    j = np.clip(np.searchsorted(zz, nodes, side="right") - 1, 0, len(zz) - 2)
    dz = np.diff(zz)[j]
    mesh = SimpleNamespace(
        z=z, nodes=nodes, radius=radius, electric=(rr, v, j, (nodes - zz[j]) / dz, dz)
    )
    return mesh, np.searchsorted(z, np.round(observe, 13))


def packet_input(f, case, n):
    if n < 1000 or n > len(f.emission.weight):
        raise ValueError("Select a supported original positive-current source population")
    q = f.before.copy()
    archived = case.startswith("archived_")
    short = case.removeprefix("archived_")
    if short == "after":
        proposal = OUT / "proposal.json"
        if proposal.is_file():
            q = f.current(*json.loads(proposal.read_text())["series_A"])
        else:
            q = np.asarray(
                json.loads((EV / "condenser-source-working-point.json").read_text())[
                    "after_group_A"
                ]
            )
    if short in ("c1_off", "both_off"):
        q[:4] = 0
    if short in ("c2_off", "both_off"):
        q[4:8] = 0
    if archived:
        with np.load(POINT / "gun.npz") as g:
            p = np.column_stack(
                [g[k][:n] for k in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s")]
            )
            ids = g["ray_id"][:n]
            energy = 300000.0 + g["energy_offset_ev"][:n]
        return q, (p, np.full(n, 1.0 / n), ids, energy - 300000.0, 15e-6)
    path = OUT / f"gun-{case}-{n}.npz"
    if not path.is_file() and case == "after" and n == 32768:
        saved = json.loads((EV / "condenser-source-working-point.json").read_text())
        path = EV / saved["fresh_gun"]
        check_file(path, saved["evidence_sha256"][path.name])
    with np.load(path) as g:
        if "gun_key" not in g or str(g["gun_key"]) != f.gun_key(q, n):
            raise ValueError(
                "The gun exit was produced by different, or unidentified, physical inputs. Relaunch the tip."
            )
        np.testing.assert_array_equal(g["current"], q)
        np.testing.assert_array_equal(g["ids"][:n], f.emission.ray_id[:n])
        if (g["weight"][:n] <= 0).any() or g["weight"][n:].any() or n < 1000:
            raise ValueError("Invalid original source or differential probe weights")
        return q, (g["phase"], g["weight"], g["ids"], g["energy"] - 300000.0, float(g["emitted_A"]))


def fresh_round(fields, z, q):
    # Only live owners are accumulated. Zero drive does not subtract any historical coefficient.
    grid = np.linspace(0.449, 1.071, 4977)
    total = np.zeros((len(grid), 12))
    for coil, amp in zip(fields.units, q, strict=True):
        if amp:
            total += amp * coil.axial_derivatives(grid, 11).T
    return make_interp_spline(grid, total, axis=0, k=5)(z) + fields.objective(z)


def images(z, jac):
    result = []
    for J in np.moveaxis(jac, 1, 0):
        b = J[:, :2, 2:]
        cost = np.linalg.norm(b, axis=(1, 2))
        local = np.flatnonzero((cost[1:-1] < cost[:-2]) & (cost[1:-1] < cost[2:])) + 1
        records = []
        for i in local:
            if not 0.45 < z[i] < 0.55:
                continue
            s = CubicHermiteSpline(
                z[i - 1 : i + 2], J[i - 1 : i + 2, :2], J[i - 1 : i + 2, 2:], axis=0
            )
            opt = minimize_scalar(
                lambda at: np.linalg.norm(s(at)[:, 2:]),
                bounds=(z[i - 1], z[i + 1]),
                method="bounded",
                options=dict(xatol=1.0e-15),
            )
            mat = s(opt.x)
            records.append(
                dict(
                    z_mm=opt.x * 1000,
                    magnifications=np.linalg.svd(mat[:, :2], compute_uv=False).tolist(),
                    angle_singular_nm_per_urad=(
                        np.linalg.svd(mat[:, 2:], compute_uv=False) * 1000
                    ).tolist(),
                )
            )
        result.append(records)
    return result


def trace(f, mesh, stops, table, case, n, step):
    q, source = packet_input(f, case, n)
    t = ApertureTracker(mesh, table, "cubic", source=source)
    p, w, ids, _, current = source
    en = t.de + 300000.0
    phi = electric_reference(t)
    loss = np.full(len(p), -1, np.int32)
    field = cp.asarray(
        np.column_stack([fresh_round(f, mesh.nodes, q), np.zeros((len(mesh.nodes), 8))])
    )
    selected = np.linspace(0, n - 1, 64, dtype=int)
    paths = []
    stats = []
    counts = []
    jac = []
    j = 0
    for group in np.array_split(stops, (len(stops) + 15) // 16):
        for p, loss in advance(t, field, p, en, phi, loss, j, group):
            live = loss[:n] == -1
            counts.append(int(live.sum()))
            stats.append(beam_moments(p[:n], live, w[:n]))
            paths.append(np.where(live[selected, None], p[selected, :2], np.nan))
            if len(p) > n:
                if (loss[n:] != -1).any():
                    raise ValueError("A differential probe hit a stop")
                diag = p[n:, :4]
                delta = np.array([1.0e-9, 1.0e-9, 1.0e-6, 1.0e-6])
                jac.append(
                    [
                        (diag[1 + k : 5 + k] - diag[5 + k : 9 + k]).T / (2 * v * delta)
                        for k, v in ((0, 1.0), (8, 0.5))
                    ]
                )
        j = group[-1]
    stats = np.array(stats)
    zs = mesh.z[stops]
    radial = stats[:, 1] + stats[:, 2]
    roots = mono_roots(zs * 1000, radial)
    splits = [450, 550, 750, 830, 1000, 1070]
    using = tuple(zip(splits[:-1], splits[1:]))
    rootlist = lambda x: [[float(v) for v in x if a < v < b] for a, b in using]
    rec = dict(
        step_mm=step,
        count=n,
        emitted_A=current,
        intervals_mm=using,
        waists_mm=rootlist(roots),
        principal_counts=[
            [len(v) for v in rootlist(mono_roots(zs * 1000, stats[:, k]))] for k in (3, 4)
        ],
        foil_survivors=int(counts[np.argmin(abs(zs - C2_FOIL[1]))]),
        at_ADL=int(counts[-1]),
        images=images(zs, np.array(jac)) if jac else [],
        current_A=q.tolist(),
    )
    stem = OUT / f"section-{case}-{n}-{step:g}"
    write(stem.parent / (stem.name + ".json"), rec)
    np.savez_compressed(
        stem.parent / (stem.name + ".npz"),
        z=zs,
        stats=stats,
        paths=np.array(paths),
        ids=ids[selected],
        count=counts,
        jacobian=np.asarray(jac),
        phase=p,
        source_ids=ids,
        weight=w,
        loss=loss,
    )
    print(
        case,
        n,
        step,
        "waists",
        [len(v) for v in rec["waists_mm"]],
        rec["waists_mm"][2:],
        "first image",
        rec["images"][-1][:1] if jac else [],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    fields = Fields()
    mesh, stops = section(fields, args.step)
    with np.load(FEM) as data:
        table = dict(data)
    for case in args.cases:
        trace(fields, mesh, stops, table, case, args.count, args.step)
