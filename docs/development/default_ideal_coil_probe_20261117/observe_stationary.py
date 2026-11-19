"""Observe a fixed, authenticated source in the high-order stationary design field."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fit_stationary import metrics, write
from stationary_pupil import (
    CHANNELS,
    Column,
    _transport,
    current_vector,
    half_nodes,
    hex_vector,
    kinematics,
    magnetic,
    read_gun,
    electric_mesh,
    electric_at,
)
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.optics import direct_focus_topology
from temsim.optics.crossover_acceptance import regions_for_state
from temsim.optics.waist_evidence import (
    corrector_evidence_for_state,
    finite_interval,
    requested_corrector_planes,
)
from temsim.physics.potential_geometry import capture_potential_geometry
from temsim.physics.beam_statistics import _weighted_quantile
from temsim.physics.unified_particle_contacts import compile_contact_geometry
from trial import observed_minimum, plane_metrics
from stationary_cache import column_identity, load_prefix, save_column


def spatial_widths(positions, weights):
    xy = positions - np.average(positions, axis=0, weights=weights)
    squared = np.einsum("ij,ij->i", xy, xy)
    if np.all(weights == weights[0]):
        # Smallest whole number of equal-current packets containing 19/20.
        index = (19 * len(weights) + 19) // 20 - 1
        q95 = np.partition(squared, index)[index]
    else:
        q95 = _weighted_quantile(squared, weights, 0.95)
    return 2e9 * np.sqrt(q95), 1e9 * np.sqrt(np.average(squared, weights=weights))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("basis", "recipe", "trial", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--step-mm", type=float, default=0.05)
    parser.add_argument(
        "--focus-spacing-nm",
        type=float,
        default=0.25,
        help="Spacing in the inner 240 nm through-focus window",
    )
    parser.add_argument("--witness", action="store_true")
    parser.add_argument("--gun-directory", type=Path, help="Separate matching original gun checkpoint")
    parser.add_argument(
        "--native-focus",
        action="store_true",
        help="Repeat the final 15 micrometres with the production full-field Lorentz solver",
    )
    parser.add_argument(
        "--save-column",
        action="store_true",
        help="Keep verified trajectories for subsequent nested source growth",
    )
    parser.add_argument(
        "--parent-column",
        type=Path,
        help="Reuse a completed parent population at the identical integration mesh and fixed fields",
    )
    args = parser.parse_args()
    if not (
        np.isfinite(args.step_mm)
        and args.step_mm > 0
        and np.isfinite(args.focus_spacing_nm)
        and 0 < args.focus_spacing_nm <= 1.0
    ):
        parser.error("Require a positive transport step and a through-focus spacing in (0, 1] nm")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = InstrumentSnapshot.from_dict(json.loads((args.trial / "inputs.json").read_text()))
    state = snapshot.restore()
    recipe = json.loads(args.recipe.read_text())
    gun_directory = args.gun_directory or args.trial
    gun = json.loads((gun_directory / "gun.json").read_text())
    if gun["input_digest"] != snapshot.digest:
        raise ValueError("The executed gun belongs to a different input snapshot")
    for key, spec in recipe["recipes"].items():
        if state.lens_field_map_descriptors[key]["ideal_current_segments"] != spec:
            raise ValueError("Incompatible gun winding currents")
    for key, attr in CHANNELS:
        value = getattr(next(c for c in state.corrector_elements if c.key == key), attr)
        if value != recipe["multipole_commands"][key][attr]:
            raise ValueError("Incompatible gun multipole currents")
    phase, w, ids, de, current = read_gun(gun_directory)
    start = state.electron_gun.exit_plane_z_mm
    surface = state.sample.upper_surface_z_mm
    stop = surface + 0.12
    regions = regions_for_state(state)
    topology_planes = [z for region in regions.values() for z in region["bounds_mm"]]
    fine_focus = np.linspace(-120.0, 120.0, 2 * int(np.ceil(120.0 / args.focus_spacing_nm)) + 1)
    focus_nm = np.unique(
        np.r_[fine_focus, -np.geomspace(120.0, 120000.0, 81), np.geomspace(120.0, 120000.0, 81)]
    )
    observe = np.unique(
        np.r_[
            np.arange(start, surface, 0.5),
            requested_corrector_planes(state),
            direct_focus_topology.requested_capture_planes(state),
            topology_planes,
            surface + focus_nm * 1e-6,
            surface - 0.010,
            surface,
            start,
            stop,
        ]
    )
    observe = observe[(observe >= start) & (observe <= stop)]
    geometry = compile_contact_geometry(capture_potential_geometry(state, start * 0.001, stop * 0.001))
    faces = np.r_[geometry.axial_lower_m, geometry.axial_upper_m] * 1000
    z = np.unique(
        np.r_[
            observe,
            faces[(faces > start) & (faces < stop)],
            np.linspace(start, stop, int(np.ceil((stop - start) / args.step_mm)) + 1),
        ]
    )
    nodes = half_nodes(z * 0.001)
    col = Column(args.basis, recipe, nodes)
    field = col.field(current_vector(recipe), hex_vector(recipe))
    del col
    saved = np.searchsorted(z, observe)
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot

    runtime = get_unified_field_runtime(state)
    (potential,) = (
        row.potential
        for row in runtime.snapshot.contributions
        if type(row.potential) is ElectricPotentialSnapshot
    )
    electric = electric_mesh(potential, nodes)
    entrance_energy = state.electron_gun.nominal_exit_energy_ev + de
    signature = column_identity(z, saved, field, potential.digest, geometry)
    prefix_count = 0
    if args.parent_column is not None:
        old, old_radii = load_prefix(args.parent_column, gun, signature, ids, phase, entrance_energy, w)
        prefix_count = old.shape[1]
    added = slice(prefix_count, None)
    data, extent = _transport(
        phase[added],
        field,
        np.diff(z * 0.001),
        *kinematics(entrance_energy[added]),
        saved,
        True,
        electric,
        entrance_energy[added],
    )
    if not np.isfinite(data).all() or not np.isfinite(extent).all():
        raise ValueError("Nonfinite propagated phase or swept radius")
    max_r = extent.max(axis=0) * 1000
    if prefix_count:
        data = np.concatenate((old, data), axis=1)
        max_r = np.maximum(max_r, old_radii)
    margins = []
    for i, (key, _) in enumerate(geometry.rows):
        ix = (z[:-1] * 0.001 <= geometry.axial_upper_m[i]) & (z[1:] * 0.001 >= geometry.axial_lower_m[i])
        if not np.any(ix):
            continue
        if not geometry.native[i]:
            raise ValueError(f"Direct native contact is required for the posed boundary {key}")
        margin = geometry.radii_mm[i] - np.linalg.norm(geometry.origins_mm[i, :2]) - max_r[ix].max()
        margins.append(dict(key=key, radius_mm=float(geometry.radii_mm[i]), clearance_mm=float(margin)))
    if not margins or min(m["clearance_mm"] for m in margins) <= 0:
        raise ValueError(f"Physical aperture transmission requires native contact evaluation: {margins}")
    if args.save_column:
        save_column(args.output, gun, signature, ids, phase, entrance_energy, w, data, max_r)
    idx = int(np.flatnonzero(observe == surface).item())
    native_window = np.isin(observe, np.unique(np.r_[surface + focus_nm * 1e-6, surface]))
    fz = observe[native_window]
    print(
        "Complete gun and clear mechanical envelopes",
        len(w),
        "surface",
        metrics(data[idx], w, current),
        flush=True,
    )
    focus_indices = np.flatnonzero(native_window)

    def survey(mask):
        weights = w[mask]
        widths, core = [], []
        for j in focus_indices:
            xy = data[j, mask, :2]
            widths.append(spatial_widths(xy, weights))
            core.append(current * weights[np.hypot(xy[:, 0], xy[:, 1]) <= 50e-12].sum())
        d95, rms = np.array(widths).T
        core = np.asarray(core)
        peak = float(core.max())
        peak_planes = fz[core == peak]
        surface_core = float(core[fz == surface].item())
        summary = dict(
            surface=metrics(data[idx, mask], weights, current),
            original_cohort=len(weights),
            D95=observed_minimum(fz, d95, surface),
            RMS=observed_minimum(fz, rms, surface),
            core_50pm=dict(
                maximum_current_a=peak,
                observed_maximum_planes_mm=peak_planes.tolist(),
                surface_attains_maximum=bool(np.any(peak_planes == surface)),
                surface_fractional_loss=(1 - surface_core / peak) if peak else None,
            ),
        )
        return summary, d95, core

    complete, focus_widths, focus_core = survey(slice(None))
    report = dict(
        status="Independent stationary high-order winding-field calculation",
        input_digest=snapshot.digest,
        gun_sha256=gun["sha256"],
        native_gun_field="magnetic_table" not in gun,
        original_emitted=gun["population"],
        positive_surviving=len(w),
        effective=w.sum() ** 2 / (w @ w),
        current_a=current * w.sum(),
        upper_surface_mm=surface,
        step_mm=args.step_mm,
        mechanical_clearances=margins,
        focus_spacing_nm=float(fine_focus[1] - fine_focus[0]),
        swept_radius_max_mm=float(max_r.max()),
        full=complete,
        prefixes={},
        reused_column_survivors=prefix_count,
        propagated_column_survivors=len(w) - prefix_count,
        electric_potential_digest=potential.digest,
        electric_basis=potential.metadata["basis"],
    )
    prefix_widths = {}
    for n in (1024, 2048, 4096, 8192, 16384):
        if n < gun["population"]["positive_current_count"]:
            sub = ids < n
            item, prefix_widths[f"focus_D95_nm_{n}"], _ = survey(sub)
            item["absolute_current_in_full_run_a"] = current * w[sub].sum()
            report["prefixes"][str(n)] = item
    cp = SimpleNamespace(
        z_mm=observe, x_m=data[:, :, 0], y_m=data[:, :, 1], tx_rad=data[:, :, 2], ty_rad=data[:, :, 3]
    )
    active = np.ones(cp.x_m.shape, dtype=bool)
    report["condensers"] = {}
    for key, required in (("c2_to_c3", 1), ("c3_to_adl", 0)):
        r = finite_interval(cp, active, w, *regions[key]["bounds_mm"], maximum_spacing_mm=0.5, source_ids=ids)
        report["condensers"][key] = dict(
            required=required,
            observed=len(r["regions"]),
            passed=r["status"] == "RESOLVED_AT_BOUNDED_SAMPLING" and len(r["regions"]) == required,
            evidence=r,
        )
    report["corrector"] = corrector_evidence_for_state(state, cp, active, w, ids)
    lower, target = direct_focus_topology.bounds_for_state(state)
    report["direct"] = direct_focus_topology.evaluate(
        cp,
        active,
        w,
        lower_z_mm=lower,
        surface_z_mm=target,
        maximum_spacing_mm=0.5,
        surface_focus_tolerance_nm=1.0,
        waist_detector="finite_region",
        source_ids=ids,
    )
    if args.witness:
        from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler

        rng = np.random.default_rng(7341)
        plane = np.r_[rng.integers(0, len(observe), 3500), np.linspace(0, len(observe) - 1, 1000, dtype=int)]
        ray = rng.integers(0, len(w), len(plane))
        # Include the outermost transmitted particle at uniformly spaced
        # axial positions, where the off-axis expansion is most demanding.
        rim_planes = np.searchsorted(observe, np.linspace(observe[0], observe[-1], 1000))
        rim_rays = [np.argmax(np.einsum("ij,ij->i", data[j, :, :2], data[j, :, :2])) for j in rim_planes]
        plane, ray = np.r_[plane, rim_planes], np.r_[ray, rim_rays]
        points = np.column_stack((data[plane, ray, :2], observe[plane] * 0.001))
        sample = compile_unified_potential_sampler(
            get_unified_field_runtime(state).snapshot, execution_backend="Require GPU"
        )
        reference = [sample.sample_fields(part) for part in np.array_split(points, 24)]
        b = np.vstack([r["B_t"] for r in reference])
        estimate = np.array([magnetic(x, y, field[2 * saved[i]]) for (x, y, _), i in zip(points, plane)])
        error = np.linalg.norm(estimate - b, axis=1)
        sampled_e = np.vstack([r["E_v_per_m"] for r in reference])
        verified_e = np.array(
            [electric_at(x, y, 2 * saved[i], electric)[1:] for (x, y, _), i in zip(points, plane)]
        )
        report["field_witness"] = dict(
            sample_count=len(points),
            outermost_particles=len(rim_planes),
            max_outermost_B_error_t=float(error[-len(rim_planes) :].max()),
            max_B_error_t=error.max(),
            rms_B_error_t=np.sqrt(np.mean(error**2)),
            max_E_v_per_m=np.max(abs(sampled_e)),
            max_E_error_v_per_m=np.max(abs(verified_e - sampled_e)),
        )
    source_phi = np.array([electric_at(v[0], v[1], 0, electric)[0] for v in phase])
    target_phi = np.array([electric_at(v[0], v[1], 2 * saved[idx], electric)[0] for v in data[idx]])
    if args.native_focus:
        from temsim.physics.particle_sections import _events, _trace_segment

        entry = int(np.flatnonzero(observe == surface - 0.010).item())
        block = data[entry]
        e0 = np.array([electric_at(v[0], v[1], 2 * saved[entry], electric)[0] for v in block])
        near = [
            -5000,
            -1000,
            -200,
            -80,
            -40,
            -20,
            -8,
            -4,
            -2,
            -1,
            -0.75,
            -0.5,
            -0.25,
            0.0,
            0.25,
            0.5,
            0.75,
            1,
            2,
            4,
            8,
            20,
            40,
            80,
            200,
            1000,
            5000,
        ]
        for key in ("D95", "RMS"):
            centre = complete[key]["representative_offset_nm"]
            near.extend(centre + fine_focus[abs(fine_focus) <= 1.0])
        near = np.unique(surface + np.asarray(near) * 1e-6)
        with np.load(gun_directory / "gun.npz") as g:
            a = g["alive"] & (g["weight"] > 0)
            identity = (ids, g["source_azimuth_rad"][a])
            survival = (g["alive"][a], g["blocked_z_mm"][a], g["blocked_key"][a])
        segment, _, _ = _trace_segment(
            state,
            "incident",
            observe[entry],
            surface + 0.005,
            tuple(block[:, k] for k in (0, 2, 1, 3, 4)),
            de,
            w,
            identity,
            survival,
            _events(state),
            None,
            np.inf,
            initial_kicks=False,
            save_z_mm=near,
            initial_kinetic_energy_ev=entrance_energy + (e0 - source_phi),
        )
        nc = segment.checkpoints
        if not np.all(nc.unified_state.active) or not np.array_equal(segment.branch.source_ray_id, ids):
            raise ValueError("Native focus replay changed source membership")
        rows = [plane_metrics(nc, j, w, current) for j in range(len(nc.z_mm))]
        ii = int(np.flatnonzero(nc.z_mm == surface).item())
        replay = np.stack(
            [getattr(nc, k)[ii] for k in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s")], axis=1
        )
        report["native_focus"] = dict(
            start_mm=observe[entry],
            end_mm=surface + 0.005,
            source="Same field-specific high-order column checkpoint 10 um before the surface",
            surface=rows[ii],
            full_profile=rows,
            D95=observed_minimum(nc.z_mm, [v["D95_nm"] for v in rows], surface),
            RMS=observed_minimum(nc.z_mm, [v["rms_radius_nm"] for v in rows], surface),
            max_surface_difference_xyz_slopes_time=np.max(np.abs(data[idx] - replay), axis=0).tolist(),
        )
    write(args.output / "optics.json", report)
    np.savez_compressed(
        args.output / "surface.npz",
        ray_id=ids,
        weight=w,
        energy_ev=entrance_energy + (target_phi - source_phi),
        phase=data[idx],
        input_digest=snapshot.digest,
        z_mm=surface,
        emitted_current_a=current,
        focus_offset_nm=(fz - surface) * 1e6,
        focus_D95_nm=focus_widths,
        focus_core_50pm_a=focus_core,
        **prefix_widths,
    )
    print(
        "Bounded D95 focus (nm)",
        complete["D95"]["neighbour_bracket_offset_nm"],
        "condenser/corrector/direct",
        [a["passed"] for a in report["condensers"].values()],
        report["corrector"]["status"],
        report["direct"]["status"],
        flush=True,
    )


if __name__ == "__main__":
    main()
