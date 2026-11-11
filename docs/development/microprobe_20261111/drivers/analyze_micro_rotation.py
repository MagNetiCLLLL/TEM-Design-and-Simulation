"""Post-analyze saved Microprobe particles; never transport or reweight a source.

Only existing phase arrays and static magnetic-potential samples are evaluated.
The symmetric-gauge diagnostic follows the signed electron convention in
ray_integrator.py: g=q*Bz/(2*p), ux=tx-g*y, uy=ty+g*x.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.beam_statistics import transverse_beam_statistics, _weighted_quantile
from temsim.physics.relativistic_lorentz import ELECTRON, ELECTRON_MASS_KG, SPEED_OF_LIGHT_M_PER_S
from validate_joint_probe import _restore_exact


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def magnetic_snapshot(state):
    """Reproduce production static magnetic adapters without electric solving."""
    from temsim.physics.unified_field_runtime import unified_source_selection, time_dependent_drive
    from temsim.physics.potential_capture import capture_selected_column_potentials
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
    from temsim.physics.analytic_potential import AnalyticMagneticPotential
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.lens_field_provider import FrozenAnalyticField
    from temsim.physics.gaussian_remote_domain import POLICY
    from temsim.physics.emission_timing import resolve_emission_time_s
    from temsim.immutable_json import thaw_json

    if any(time_dependent_drive(item) for item in (*state.deflectors, *state.stigmators,
                                                  *state.corrector_elements)):
        raise ValueError("This postanalysis requires static magnetic drives")
    lenses, auxiliary = unified_source_selection(state)
    native = capture_selected_column_potentials(state, round_lens_keys=lenses,
        auxiliary_keys=auxiliary, include_electric=False, include_gun_alignment=True,
        time_s=resolve_emission_time_s(state))
    rows = []
    for row in native.contributions:
        potential = row.potential
        if type(potential) is AnalyticMagneticPotential:
            field = potential.field
            if getattr(field, "dynamic", False):
                potential = AnalyticMagneticPotential(replace(field, dynamic=False), {
                    **thaw_json(potential.provenance), "dynamic_drive_frozen": False,
                    "static_drive_proof": "native physical coil has identically zero varying coefficients"})
            potential = (HighOrderRoundLensPotential(potential.field, radial_order=3,
                axial_domain="gaussian_all_z", remote_domain=POLICY)
                if type(potential.field) is FrozenAnalyticField else AnalyticPotentialDomain(potential))
        rows.append(PotentialContribution(row.source_id, row.owner_ids, potential, row.kind))
    return ElectromagneticPotentialSnapshot(native.scene, tuple(rows), native.gauge)


def formula_checks():
    """Pure-array checks against the repository's canonical basis and charge."""
    from temsim.optics.direct_alignment import _canonical_source_basis
    from temsim.physics.ray_integrator import ELECTRON_CHARGE_C
    if ELECTRON.charge_c != ELECTRON_CHARGE_C or ELECTRON.charge_c >= 0:
        raise ValueError("Electron charge conventions disagree")
    phase = np.array([[2e-6, -3e-6, 4e-4, -2e-4], [-1e-6, 5e-6, 2e-4, 3e-4]])
    g = -112.7
    canonical = phase.copy()
    canonical[:, 2] -= g * phase[:, 1]
    canonical[:, 3] += g * phase[:, 0]
    np.testing.assert_allclose(canonical @ _canonical_source_basis(g).T, phase,
                               rtol=1e-14, atol=1e-18)
    rotation_only = np.c_[phase[:, :2], g*phase[:, 1], -g*phase[:, 0]]
    removed = rotation_only[:, 2:] - np.c_[g*phase[:, 1], -g*phase[:, 0]]
    np.testing.assert_array_equal(removed, np.zeros_like(removed))
    return dict(electron_charge_c=ELECTRON.charge_c, basis_round_trip="PASS",
                pure_larmor_removal="PASS", formula="ux=tx-g*y; uy=ty+g*x; g=q*Bz/(2*p)")


def moments(xy, slopes, weights):
    """Centered current-weighted covariance, vergence and projected emittances."""
    w = weights / weights.sum()
    phase = np.c_[xy, slopes]
    mean = np.sum(w[:, None] * phase, axis=0)
    centered = phase - mean
    r, t = centered[:, :2], centered[:, 2:]
    covariance = (centered * w[:, None]).T @ centered
    rr, tr = covariance[:2, :2], covariance[2:, :2]
    k = tr @ np.linalg.pinv(rr, rcond=1e-12)
    radius = np.linalg.norm(r, axis=1)
    nonzero = radius > 0.
    radial = np.zeros(len(radius))
    tangent = np.zeros(len(radius))
    radial[nonzero] = np.sum(r[nonzero]*t[nonzero], axis=1) / radius[nonzero]
    tangent[nonzero] = (r[nonzero, 0]*t[nonzero, 1] - r[nonzero, 1]*t[nonzero, 0]) / radius[nonzero]
    onaxis_residual = float(np.sum(w[~nonzero] * np.sum(t[~nonzero]**2, axis=1)))
    radial_second = float(np.sum(w*radial**2))
    tangent_second = float(np.sum(w*tangent**2))
    np.testing.assert_allclose(radial_second+tangent_second+onaxis_residual,
                               np.trace(covariance[2:, 2:]), rtol=1e-12, atol=1e-25)
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-300))
    scaled = covariance / np.outer(scale, scale)
    determinant = float(np.linalg.det(scaled))
    if determinant < -1e-10:
        raise ValueError("Covariance is not positive semidefinite")
    volume4 = float(np.sqrt(max(determinant, 0.)) * np.prod(scale))
    emittances = []
    for index in (0, 1):
        indices = [index, index+2]
        sub = covariance[np.ix_(indices, indices)]
        emittances.append(float(np.sqrt(max(0., np.linalg.det(sub)))))
    r2 = float(np.trace(rr))
    def second_quantities(values):
        return dict(mean_mrad=float(np.sum(w*values)*1e3),
                    rms_mrad=float(np.sqrt(np.sum(w*values**2))*1e3),
                    absolute95_mrad=float(_weighted_quantile(np.abs(values), w, .95)*1e3))
    return dict(mean_xy_m=mean[:2].tolist(), mean_slopes=mean[2:].tolist(),
        covariance_xy_tx_ty=covariance.tolist(),
        per_axis_position_slope_covariance_m_rad=[float(tr[0, 0]), float(tr[1, 1])],
        per_axis_scalar_vergence_per_m=[float(tr[i, i]/rr[i, i]) if rr[i, i] > 0 else None for i in (0, 1)],
        fitted_vergence_matrix_per_m=k.tolist(),
        symmetric_vergence_eigenvalues_per_m=np.linalg.eigvalsh((k+k.T)/2).tolist(),
        radial_vergence_per_m=float(np.trace(tr)/r2) if r2 > 0 else None,
        tangential_rotation_per_m=float((tr[1, 0]-tr[0, 1])/r2) if r2 > 0 else None,
        radial_angle=second_quantities(radial), tangential_angle=second_quantities(tangent),
        zero_radius_residual_angle_second_moment_rad2=onaxis_residual,
        geometric_projected_rms_emittance_xy_m_rad=emittances,
        geometric_rms_4d_volume_m2_rad2=volume4,
        geometric_rms_4d_equivalent_emittance_m_rad=float(np.sqrt(volume4)),
        rms_slope_mrad=float(np.sqrt(np.trace(covariance[2:, 2:]))*1e3),
        radius_rms_um=float(np.sqrt(r2)*1e6),
        diameter95_um=float(2e6*_weighted_quantile(radius, w, .95)),
        decomposition_check="PASS")


def analyze_plane(arrays, index, current, potential):
    weights = arrays["weights"]
    keep = np.asarray(arrays["alive"][index], bool) & (weights > 0.)
    xy = np.c_[arrays["x_m"][index, keep], arrays["y_m"][index, keep]]
    slopes = np.c_[arrays["tx_rad"][index, keep], arrays["ty_rad"][index, keep]]
    energy = arrays["kinetic_energy_ev"][index, keep]
    w = weights[keep]
    if not len(w) or not np.isfinite(np.c_[xy, slopes, energy]).all() or np.any(energy <= 0):
        raise ValueError("The saved physical observation has no finite positive-current population")
    center = np.average(xy, axis=0, weights=w)
    z = float(arrays["z_mm"][index])
    positions = np.c_[xy, np.full(len(xy), z*.001)]
    center_position = np.r_[center, z*.001][None, :]
    center_field = potential.sample(center_position)
    sampled = potential.sample(positions)
    axis_field = potential.sample(np.array([[0., 0., z*.001]]))
    e_joule = energy * abs(ELECTRON.charge_c)
    momentum = np.sqrt(e_joule * (e_joule + 2*ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S**2)) / SPEED_OF_LIGHT_M_PER_S
    bz = float(center_field["B_t"][0, 2])
    g = ELECTRON.charge_c*bz/(2*momentum)
    centered_xy = xy-center
    local = slopes - np.c_[g*centered_xy[:, 1], -g*centered_xy[:, 0]]
    # P_can=p_mechanical+q*A. This is a local paraxial increment convention,
    # normalized by each particle's total mechanical momentum, not P_can,z.
    native = slopes + (ELECTRON.charge_c/momentum)[:, None] * (sampled["A_t_m"][:, :2]-center_field["A_t_m"][0, :2])
    statistics = transverse_beam_statistics(xy[:, 0], xy[:, 1], slopes[:, 0], slopes[:, 1], weights=w)
    local_statistics = transverse_beam_statistics(xy[:, 0], xy[:, 1], local[:, 0], local[:, 1], weights=w)
    native_statistics = transverse_beam_statistics(xy[:, 0], xy[:, 1], native[:, 0], native[:, 1], weights=w)
    return dict(z_mm=z, positive_current_count=int(keep.sum()), effective_count=float(w.sum()**2/np.dot(w,w)),
        original_current_weight=float(w.sum()), current_a=float(current*w.sum()),
        field=dict(chief_B_t=center_field["B_t"][0].tolist(), axis_B_t=axis_field["B_t"][0].tolist(),
            sampled_Bz_range_t=[float(sampled["B_t"][:, 2].min()),float(sampled["B_t"][:, 2].max())],
            chief_A_t_m=center_field["A_t_m"][0].tolist(),
            signed_g_per_m_range=[float(g.min()),float(g.max())], mean_g_per_m=float(np.average(g,weights=w))),
        mechanical=dict(**moments(xy, slopes, w), physical_direction_statistics=asdict(statistics)),
        local_symmetric_gauge=dict(**moments(xy, local, w),
            alpha95_paraxial_surrogate_mrad=local_statistics.convergence_95_mrad,
            alpha_rms_paraxial_surrogate_mrad=local_statistics.convergence_rms_rad*1e3),
        native_gauge_increment=dict(**moments(xy, native, w),
            alpha95_paraxial_surrogate_mrad=native_statistics.convergence_95_mrad,
            alpha_rms_paraxial_surrogate_mrad=native_statistics.convergence_rms_rad*1e3,
            maximum_local_vs_native_slope_difference_rad=float(np.linalg.norm(native-local,axis=1).max())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-um", type=float, default=100.001)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new postanalysis output; execution receipts are never overwritten")
    if not np.isfinite(args.window_um) or args.window_um < 0:
        parser.error("The observation window must be finite and nonnegative")
    trial = args.trial.resolve()
    paths = {name:trial/name for name in ("observations.npz", "inputs.json", "input.toml", "report.json")}
    checks = formula_checks()
    receipt = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    if sha256(paths["observations.npz"]) != receipt["observations_sha256"]:
        raise ValueError("Saved observation hash differs from its completed trial receipt")
    snapshot = InstrumentSnapshot.from_dict(json.loads(paths["inputs.json"].read_text(encoding="utf-8")))
    if snapshot.digest != receipt["execution"]["input_digest"]:
        raise ValueError("Input snapshot differs from the executed particle input")
    state = _restore_exact(snapshot)
    from temsim.profile_io import read_profile
    read_profile(paths["input.toml"])
    potential = magnetic_snapshot(state)
    with np.load(paths["observations.npz"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    weights = arrays["weights"]
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0) or np.count_nonzero(weights > 0) < 1000:
        raise ValueError("Postanalysis requires at least 1000 original positive-current particles")
    if len(np.unique(arrays["source_id"])) != len(weights):
        raise ValueError("Original source identities must remain unique")
    surface = float(state.sample.upper_surface_z_mm)
    chosen = np.flatnonzero(np.abs(arrays["z_mm"]-surface)*1e3 <= args.window_um)
    if surface not in arrays["z_mm"][chosen]:
        raise ValueError("The actual specimen upper-surface observation is missing")
    rows = [analyze_plane(arrays, int(index), receipt["execution"]["source_current_a"], potential) for index in chosen]
    for row in rows:
        row["offset_from_surface_um"] = (row["z_mm"]-surface)*1e3
    lost = ~np.asarray(arrays["terminal_alive"],bool)
    losses = {str(key):dict(count=int(np.count_nonzero(lost & (arrays["blocked_key"]==key))),
        original_weight=float(weights[lost & (arrays["blocked_key"]==key)].sum()))
        for key in np.unique(arrays["blocked_key"][lost])}
    report = dict(schema="microprobe-saved-phase-rotation-analysis-v1", status="COMPLETE_POSTANALYSIS",
        physical_qualification="NOT_ADDED", formula_checks=checks,
        inputs={name:dict(path=str(path),sha256=sha256(path)) for name,path in paths.items()},
        script_sha256=sha256(__file__), executed_input_digest=snapshot.digest,
        source_count=len(weights), original_positive_current_count=int(np.count_nonzero(weights > 0)),
        source_current_a=receipt["execution"]["source_current_a"], original_weight_sum=float(weights.sum()),
        magnetic_snapshot_digest=potential.numerical_digest, magnetic_sources=[row.source_id for row in potential.contributions],
        magnetic_capture="Production static magnetic adapters only; no electric-field solve or particle propagation",
        reference_surface_mm=surface, surface=next(row for row in rows if row["z_mm"]==surface), axial_observations=rows,
        original_terminal_stops=losses,
        limitations=["Mechanical angles use actual saved directions and the current-weighted chief ray.",
            "Canonical increments use a specified gauge and paraxial normalization; they are not mechanical electron directions or a universal zero-angle criterion.",
            "The local symmetric-gauge transform uses chief Bz and centroid-relative positions; the native-potential variant retains transverse and nonlinear A variation.",
            "Slopes approximate transverse momentum divided by total momentum. Exact finite-angle canonical momenta and a shared reference-momentum symplectic chart require a separate analysis.",
            "Per-axis projected RMS emittance changes with coupling, gauge and aperture interception. The 4D covariance volume is a descriptive moment, not a D95 acceptance bound.",
            "Radial vergence can vanish while opposite axis vergences, tangential rotation, emittance or tails leave substantial angular spread.",
            "These saved observations do not refine source count, integration step, field calibration or the native D95 axial-minimum bracket.",
            "No source particle, original weight, physical stop or executed receipt was changed."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    surface_row = report["surface"]
    print(json.dumps(dict(output=str(args.output.resolve()), count=surface_row["positive_current_count"],
        d95_um=surface_row["mechanical"]["diameter95_um"],
        mechanical_alpha95_mrad=surface_row["mechanical"]["physical_direction_statistics"]["convergence_95_rad"]*1e3,
        mechanical_alpha_rms_mrad=surface_row["mechanical"]["physical_direction_statistics"]["convergence_rms_rad"]*1e3,
        native_alpha95_paraxial_surrogate_mrad=surface_row["native_gauge_increment"]["alpha95_paraxial_surrogate_mrad"],
        per_axis_vergence_per_m=surface_row["mechanical"]["per_axis_scalar_vergence_per_m"],
        radial_rms_mrad=surface_row["mechanical"]["radial_angle"]["rms_mrad"],
        tangential_rms_mrad=surface_row["mechanical"]["tangential_angle"]["rms_mrad"],
        native_tangential_rms_mrad=surface_row["native_gauge_increment"]["tangential_angle"]["rms_mrad"]),
        indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
