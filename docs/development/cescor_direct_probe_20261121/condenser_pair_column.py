"""Transport the same freshly emitted, weighted packet to the actual specimen entrance."""

import argparse
from types import SimpleNamespace

import cupy as cp
import numpy as np
from scipy.interpolate import make_interp_spline

from column_current import ApertureTracker, C2_FOIL, SURFACE, current_ledger
from condenser_material_binding import check_materials
from condenser_pair_gun import identity, selected_field
from condenser_pair_study import OUT, Pair
from condenser_source_fields import FEM, write
from condenser_source_trace import images
from joint_relay_search import mono_roots
from relay_transfer import advance, electric_reference
from stationary_pupil import half_nodes
from temsim.physics.potential_geometry import capture_potential_geometry
from temsim.physics.unified_particle_contacts import compile_contact_geometry
from trace_normal_joint_field import beam_moments


def column_mesh(f, step, extra_observation_mm=()):
    lo = 0.45
    top = f.state.sample.upper_surface_z_mm * 0.001
    assert abs(top - SURFACE) < 1e-12
    extra = np.asarray(extra_observation_mm, dtype=float) * 0.001
    if extra.ndim != 1 or not np.all(np.isfinite(extra)) or np.any((extra < lo) | (extra > top)):
        raise ValueError("Additional observation planes must lie in the traced column")
    hw = compile_contact_geometry(capture_potential_geometry(f.state, lo, top))
    if not hw.native.all() or np.any(hw.origins_mm[:, :2]):
        raise ValueError("A hardware body requires a non-centred contact calculation")
    keys = np.array([r[0] for r in hw.rows])
    left, right = hw.axial_lower_m, hw.axial_upper_m
    radii = 0.001 * hw.radii_mm
    foil = (keys == "condenser_aperture_2") & np.isclose(radii, 50e-6)
    np.testing.assert_allclose([left[foil].item(), right[foil].item()], C2_FOIL, rtol=0, atol=1e-12)
    observe = np.unique(
        np.round(
            np.r_[
                np.arange(450, 1599.19, 0.5) * 0.001,
                np.arange(450, 551, 0.125) * 0.001,
                np.arange(775, 817, 0.125) * 0.001,
                np.arange(1340, 1425, 0.25) * 0.001,
                top - np.arange(0, 0.4001, 0.01) * 0.001,
                lo,
                0.55,
                0.75,
                0.83,
                1.0,
                1.07,
                C2_FOIL,
                extra,
                top,
            ],
            13,
        )
    )
    faces = np.r_[left, right]
    z = np.unique(
        np.round(
            np.r_[
                np.linspace(lo, top, 1 + int(np.ceil((top - lo) * 1000 / step))),
                faces[(faces > lo) & (faces < top)],
                observe,
            ],
            13,
        )
    )
    mid = (z[1:] + z[:-1]) / 2
    radius = np.full(len(mid), np.inf)
    owner = np.full(len(mid), -1, int)
    for i, (a, b, r) in enumerate(zip(left, right, radii, strict=True)):
        take = (mid >= a) & (mid <= b) & (r < radius)
        radius[take] = r
        owner[take] = i
    nodes = half_nodes(z)
    e = f.electric.base_field
    j = np.clip(np.searchsorted(e.z, nodes, side="right") - 1, 0, len(e.z) - 2)
    dz = np.diff(e.z)[j]
    mesh = SimpleNamespace(
        z=z,
        nodes=nodes,
        radius=radius,
        owner=owner,
        keys=keys,
        electric=(e.r, e.voltage, j, (nodes - e.z[j]) / dz, dz),
    )
    return mesh, np.searchsorted(z, observe)


def round_field(f, nodes, q, d, order):
    grid = np.linspace(0.449, 1.601, 4609)
    out = np.zeros((len(grid), 12))
    for coil, current in zip(f.units, q, strict=True):
        if current:
            out += current * coil.axial_derivatives(grid, 11).T
    field = make_interp_spline(grid, out, axis=0, k=5)(nodes) + f.objective(nodes)
    induced = make_interp_spline(d["z"], d["induced_T"], k=order)
    for j in range(order + 1):
        field[:, j] += induced(nodes, nu=j)
    return cp.asarray(np.ascontiguousarray(np.column_stack([field, np.zeros((len(field), 8))])))


def specimen(p, w, rate):
    center = np.average(p[:, :2], axis=0, weights=w)
    r = np.linalg.norm(p[:, :2] - center, axis=1)
    axis = np.linalg.norm(p[:, :2], axis=1)
    angle = np.arctan(np.linalg.norm(p[:, 2:4], axis=1))

    def quant(values):
        order = np.argsort(values)
        return values[order][np.searchsorted(np.cumsum(w[order]), 0.95 * w.sum())]

    return dict(
        D95_um=float(2e6 * quant(r)),
        RMS_radius_um=float(1e6 * np.sqrt(np.average(r * r, weights=w))),
        axis_50nm_nA=float(rate * 1e9 * w[axis <= 50e-9].sum()),
        axis_50pm_pA=float(rate * 1e12 * w[axis <= 50e-12].sum()),
        centroid_um=(center * 1e6).tolist(),
        alpha95_mrad=float(1e3 * quant(angle)),
        largest_sampled_angle_mrad=float(1e3 * angle.max()),
        specimen_uA=float(rate * 1e6 * w.sum()),
        count=len(p),
    )


def run(pair, count, step, order, *, extra_observation_mm=()):
    f = pair.f
    q, d = selected_field(pair)
    check_materials(f.state)
    np.testing.assert_array_equal(q[8:], f.before[8:])
    with np.load(OUT / f"tip-{count}.npz") as saved:
        g = dict(saved)
    if count < 1000 or str(g["signature"]) != identity(pair, d, q, count):
        raise ValueError("Gun handoff differs from the selected physical field")
    np.testing.assert_array_equal(g["current"], q)
    np.testing.assert_array_equal(g["ids"][:count], f.emission.ray_id[:count])
    if not (g["weight"][:count] > 0).all() or g["weight"][count:].any():
        raise ValueError("Unexpected launch weights")
    m, stops = column_mesh(f, step, extra_observation_mm)
    with np.load(FEM) as table:
        t = ApertureTracker(
            m,
            dict(table),
            "cubic",
            source=(g["phase"], g["weight"], g["ids"], g["energy_offset_ev"], float(g["emitted_A"])),
        )
    field = round_field(f, m.nodes, q, d, order)
    p, w = t.input, t.weights
    loss = np.full(len(p), -1, np.int32)
    phi = electric_reference(t)
    en = t.de + 300000
    selected = np.linspace(0, count - 1, 64, dtype=int)
    np.testing.assert_array_equal(t.ids[selected], g["selected_ids"])
    paths, stats, jac, counts, near = [], [], [], [], []
    foil_profiles = {}
    j = 0
    delta = np.array([1e-9, 1e-9, 1e-6, 1e-6])
    print("Full-column finite source", count, "step", step, "mm", len(m.z), "nodes", flush=True)
    for group in np.array_split(stops, (len(stops) + 15) // 16):
        for idx, (p, loss) in zip(group, advance(t, field, p, en, phi, loss, j, group), strict=True):
            live = loss[:count] == -1
            if not live.any() or (loss[count:] != -1).any():
                raise ValueError("No sample beam or intercepted differential probe")
            counts.append(int(live.sum()))
            stats.append(beam_moments(p[:count], live, w[:count]))
            paths.append(np.where(live[selected, None], p[selected, :2], np.nan))
            diag = p[count:, :4]
            jac.append(
                [
                    (diag[1 + k : 5 + k] - diag[5 + k : 9 + k]).T / (2 * v * delta)
                    for k, v in ((0, 1.0), (8, 0.5))
                ]
            )
            if m.z[idx] in np.round(C2_FOIL, 13):
                radii_um = np.linalg.norm(p[:count][live, :2], axis=1) * 1e6
                order_r = np.argsort(radii_um)
                w_sorted = w[:count][live][order_r]
                i95 = np.searchsorted(np.cumsum(w_sorted), 0.95 * w_sorted.sum())
                foil_profiles[str(round(m.z[idx] * 1000, 3))] = dict(
                    radius95_um=float(radii_um[order_r[i95]]),
                    maximum_radius_um=float(radii_um.max()),
                    aperture_radius_um=50.0,
                    survivors=int(live.sum()),
                )
            if m.z[idx] >= SURFACE - 0.000401:
                near.append(
                    dict(z_mm=m.z[idx] * 1000, **specimen(p[:count][live], w[:count][live], t.current))
                )
        j = group[-1]
    zs = m.z[stops]
    mm = zs * 1000
    stats = np.array(stats)
    extrema = mono_roots(mm, stats[:, 1] + stats[:, 2])
    bucket = lambda x: [
        [float(a) for a in x if lo < a < hi]
        for lo, hi in [(450, 550), (550, 750), (750, 830), (830, 1000), (1000, 1070)]
    ]
    ledger = current_ledger(m, t, loss)
    record = dict(
        count=count,
        series_A=[q[0], q[4]],
        source_A=t.current,
        step_mm=step,
        surface_z_mm=mm[-1],
        source_images=images(zs, np.array(jac)),
        radial_waists_mm=bucket(extrema),
        all_column_waists_mm=extrema.tolist(),
        principal_waists_mm=[bucket(mono_roots(mm, stats[:, k])) for k in (3, 4)],
        surface=near[-1],
        near_surface=near,
        C2_aperture=foil_profiles,
        field_key=str(d["key"]),
        gun_signature=str(g["signature"]),
        ledger=ledger,
    )
    stem = OUT / f"column-{count}-{step:g}-{order}"
    write(stem.with_name(stem.name + ".json"), record)
    np.savez_compressed(
        stem.with_name(stem.name + ".npz"),
        z_mm=mm,
        stats=stats,
        paths_m=np.array(paths),
        selected_ids=t.ids[selected],
        counts=counts,
        source_ids=t.ids,
        weights=w,
        source_A=t.current,
        energy_offset_eV=t.de,
        surface_phase=p,
        loss=loss,
        loss_z_mm=np.where(loss >= 0, m.z[np.maximum(loss, 0)] * 1000, np.nan),
        series_A=q,
        material_key=d["key"],
        signature=g["signature"],
    )
    print("Completed", record["radial_waists_mm"], record["source_images"], near[-1], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=32768)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--order", type=int, choices=[5, 7], default=7)
    args = ap.parse_args()
    run(Pair(), args.count, args.step, args.order)
