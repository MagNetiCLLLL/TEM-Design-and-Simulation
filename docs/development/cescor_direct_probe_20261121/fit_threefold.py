"""Identify threefold ray aberrations in the archived D95-focus specimen phase space."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from temsim.optics.aberration_basis import CARTESIAN_NAMES, polynomial_basis
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.particle_wave_source import threefold_basis
from plot_specimen_probe import ANNULUS_UM, HERE


EVIDENCE = HERE / "evidence"
REFERENCE_RAD = 0.030
ANGLE_EDGES_MRAD = (0, 10, 20, 30, np.inf)
FOLD_SEED = 20261121
MODELS = {"background": (), "A2": ("A2",), "D4": ("D4",), "both": ("A2", "D4")}


@dataclass
class Sample:
    xy_um: np.ndarray
    angles_rad: np.ndarray
    current_na: np.ndarray
    ids: np.ndarray
    chief: np.ndarray
    frame_xy: np.ndarray

    @property
    def alpha_mrad(self):
        return np.linalg.norm(self.angles_rad, axis=1) * 1000


def read_sample(path, run, summary):
    with np.load(path, allow_pickle=False) as data:
        live = data["loss"] == -1
        ids, weights = data["ids"], data["weights"]
        phase = data["phase"][live, :4]
        np.testing.assert_equal(len(ids), run["emitted"])
        np.testing.assert_equal(len(np.unique(ids)), len(ids))
        np.testing.assert_equal(live.sum(), run["survived"])
        np.testing.assert_array_equal(live, data["loss_owner"] == -1)
        if not np.isfinite(phase).all() or not np.all(weights > 0):
            raise ValueError("Invalid coordinates or source weights")
        row = np.flatnonzero(data["z_m"] == summary["specimen_upper_surface_z_m"]).item()
        np.testing.assert_allclose(data["profiles"][row, 0], run["surface"]["D95_nm"], rtol=1e-9)
        ids, weights = ids[live], weights[live]

    stats = transverse_beam_statistics(*phase.T, weights=weights)
    np.testing.assert_allclose(2e9 * stats.radius_95_m, run["surface"]["D95_nm"], rtol=1e-10)
    np.testing.assert_allclose(stats.convergence_95_mrad, run["surface"]["alpha95_mrad"], rtol=1e-10)
    current = summary["gun"]["gun_exit_A"] * 1e9 * weights
    np.testing.assert_allclose(current.sum() / 1000, run["surface"]["transmitted_uA"], rtol=1e-12)

    return angular_sample(phase, ids, current, stats)


def angular_sample(phase, ids, current, stats):
    """Express measured mechanical angles in the current-weighted chief-ray frame."""
    direction = np.column_stack((phase[:, 2:4], np.ones(len(phase))))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    chief = np.array((stats.mean_tx_rad, stats.mean_ty_rad, 1.0))
    chief /= np.linalg.norm(chief)
    ex = np.array((1., 0., 0.)) - chief[0] * chief
    ex /= np.linalg.norm(ex)
    frame = np.stack((ex, np.cross(chief, ex)))
    transverse = direction @ frame.T
    tangent = np.linalg.norm(transverse, axis=1)
    alpha = np.arctan2(tangent, direction @ chief)
    angles = transverse * np.divide(alpha, tangent, out=np.ones_like(alpha), where=tangent > 0)[:, None]
    return Sample(phase[:, :2] * 1e6, angles, current, ids, chief, frame[:, :2])


def additional_gradient(u, power, harmonic):
    """Cosine and sine gradients of |u|^(power-m) (ux+i uy)^m / power."""
    x, y = u.T
    q = x + 1j*y
    r2 = x*x + y*y
    k = (power - harmonic) // 2
    radial = k * r2**(k-1) * q**harmonic if k else np.zeros_like(q)
    azimuthal = harmonic * r2**k * q**(harmonic-1)
    derivative = np.stack((2*x*radial + azimuthal, 2*y*radial + 1j*azimuthal), axis=1) / power
    return np.stack((derivative.real, derivative.imag), axis=2)


def design(sample, sixth_order=False):
    u = sample.angles_rad / REFERENCE_RAD
    names = ["offset_x", "offset_y", *CARTESIAN_NAMES, "D4_x", "D4_y"]
    # Keep the project's ray convention, displacement = -grad W. The extra
    # columns complete the nuisance basis through geometric order five.
    gradients = [polynomial_basis(u, gradient=True), threefold_basis(u, 5)[1]]
    extra = [(5, 1), (5, 5), (6, 2), (6, 4)]
    if sixth_order:
        extra.append((7, 3))
    for power, harmonic in extra:
        gradients.append(additional_gradient(u, power, harmonic))
        names.extend(f"C{power-1}{harmonic}_{part}" for part in ("x", "y"))
    # Map tangent-plane displacements back to the fixed laboratory z plane.
    projection = np.linalg.inv(sample.frame_xy)
    lab_gradient = projection @ np.concatenate(gradients, axis=2)
    intercept = np.broadcast_to(np.eye(2), (len(u), 2, 2))
    return np.concatenate((intercept, -lab_gradient), axis=2), np.array(names)


def fit(basis, xy, weights):
    scale = np.sqrt(weights)
    matrix = (basis * scale[:, None, None]).reshape(-1, basis.shape[-1])
    result, _, rank, singular = np.linalg.lstsq(matrix, (xy*scale[:, None]).ravel(), rcond=1e-11)
    if rank != basis.shape[-1]:
        raise ValueError("Angular data do not resolve all requested fit columns")
    return result, float(singular[0] / singular[-1])


def terms(names, label):
    return np.isin(names, (label + "_x", label + "_y"))


def physical_coefficients(coeff, names):
    result = {}
    for name, order in (("A2", 2), ("D4", 4)):
        pair_mm = coeff[terms(names, name)] * 1e-3 / REFERENCE_RAD**order
        c = complex(*pair_mm)
        # For a uniform azimuth the D4 vector has harmonics -2 and +4
        # with amplitudes 4/5 and 1/5. A2 has only the -2 harmonic.
        azimuth_factor = 1. if name == "A2" else np.sqrt(17.) / 5.
        result[name] = {
            "cos_sin_mm": pair_mm.tolist(),
            "magnitude_mm": float(abs(c)),
            "azimuth_deg_mod120": float((np.degrees(np.angle(c)) / 3) % 120),
            "rms_displacement_at_30mrad_um": float(abs(c) * 1e3 * REFERENCE_RAD**order * azimuth_factor),
        }
    return result


def ring_measure(xy, current):
    radius = np.hypot(*xy.T)
    chosen = (radius >= ANNULUS_UM[0]) & (radius < ANNULUS_UM[1])
    count = int(chosen.sum())
    strength = azimuth = None
    if count:
        z = (xy[chosen, 0] + 1j*xy[chosen, 1]) / radius[chosen]
        phasor = np.average(z**3, weights=current[chosen])
        strength = float(abs(phasor))
        azimuth = float((np.degrees(np.angle(phasor)) / 3) % 120)
    return {"count": count, "current_nA": float(current[chosen].sum()), "H3": strength,
            "orientation_deg_mod120": azimuth}


def angle_groups(sample):
    alpha = sample.alpha_mrad
    return [(alpha >= low) & (alpha < high) for low, high in zip(ANGLE_EDGES_MRAD, ANGLE_EDGES_MRAD[1:])]


def grouped_current(sample):
    groups = angle_groups(sample)
    np.testing.assert_array_equal(np.sum(groups, axis=0), 1)
    result = []
    for low, high, group in zip(ANGLE_EDGES_MRAD, ANGLE_EDGES_MRAD[1:], groups):
        angular_xy = sample.angles_rad[group]
        phi = np.arctan2(angular_xy[:, 1], angular_xy[:, 0])
        incident_h3 = np.average(np.exp(3j*phi), weights=sample.current_na[group])
        radii = np.hypot(*sample.xy_um[group].T)
        result.append({
            "lower_mrad": low, "upper_mrad_exclusive": high if np.isfinite(high) else None,
            "count": int(group.sum()), "current_uA": float(sample.current_na[group].sum() * .001),
            "arrival_direction_H3": float(abs(incident_h3)),
            "minimum_landing_radius_um": float(radii.min()), "maximum_landing_radius_um": float(radii.max()),
            "ring": ring_measure(sample.xy_um[group], sample.current_na[group]),
        })
    np.testing.assert_allclose(sum(b["current_uA"] for b in result), .001 * sample.current_na.sum())
    return result


def validation_folds(ids):
    order = np.random.default_rng(FOLD_SEED).permutation(np.argsort(ids))
    folds = np.empty(len(ids), dtype=int)
    folds[order] = np.arange(len(ids)) % 5
    return folds


def vector_rms(residual, weights):
    return float(np.sqrt(np.average(np.sum(residual**2, axis=1), weights=weights)))


def held_out(sample, basis, names):
    xy, current = sample.xy_um, sample.current_na
    folds = validation_folds(sample.ids)
    a2, d4 = terms(names, "A2"), terms(names, "D4")
    ring = (np.hypot(*xy.T) >= ANNULUS_UM[0]) & (np.hypot(*xy.T) < ANNULUS_UM[1])
    removed, checks = {}, {}
    fold_coefficients = []
    for term in ("A2", "D4"):
        removed[term] = np.empty_like(xy)
    for name, selected in MODELS.items():
        use = ~(a2 | d4)
        for term in selected:
            use |= terms(names, term)
        local = basis[:, :, use]
        predicted = np.empty_like(xy)
        for fold in range(5):
            test = folds == fold
            b, _ = fit(local[~test], xy[~test], current[~test])
            predicted[test] = local[test] @ b
            if name == "both":
                fold_coefficients.append(physical_coefficients(b, names))
                for term in removed:
                    take = terms(names, term)
                    removed[term][test] = xy[test] - basis[test][:, :, take] @ b[take]
        residual = xy - predicted
        checks[name] = {
            "rms_um": vector_rms(residual, current),
            "rms_on_observed_2_to_6um_ring_um": vector_rms(residual[ring], current[ring]),
            "rms_by_angle_group_um": [vector_rms(residual[g], current[g]) for g in angle_groups(sample)],
            "predicted_ring": ring_measure(predicted, current),
        }
    return checks, removed, fold_coefficients


def fit_record(sample, basis, names, mask):
    coeff, condition = fit(basis[mask], sample.xy_um[mask], sample.current_na[mask])
    return {
        "particles": int(mask.sum()), "condition_number_scaled_design": condition,
        "coefficients": physical_coefficients(coeff, names),
        "in_sample_vector_rms_um": vector_rms(sample.xy_um[mask] - basis[mask] @ coeff, sample.current_na[mask]),
    }


def input_controls(directory, summary, primary):
    excitation = json.loads((EVIDENCE / summary["runs"]["d95"]["winding_currents_file"]).read_text())
    checked = {}
    for tag in ("half-step", "prefix"):
        receipt = json.loads((directory / f"anchored-d95-{tag}.json").read_text())[0]
        path = directory / receipt["phase_file"]
        np.testing.assert_array_equal(receipt["currents_A"], excitation["currents_A"])
        np.testing.assert_equal(receipt["objective_At"], excitation["objective_At"])
        sample = read_sample(path, receipt, summary)
        with np.load(primary, allow_pickle=False) as orig, np.load(path, allow_pickle=False) as control:
            n = len(control["ids"])
            for key in ("ids", "energy_offset_eV", "loss_owner"):
                np.testing.assert_array_equal(control[key], orig[key][:n])
            np.testing.assert_allclose(control["weights"], orig["weights"][:n]*len(orig["ids"])/n,
                                       rtol=1e-13, atol=0)
        basis, names = design(sample)
        record = fit_record(sample, basis, names, np.ones(len(sample.ids), dtype=bool))
        record.update(max_step_mm=receipt["max_step_mm"], original_particles=n,
                      sha256=sha256(path.read_bytes()).hexdigest(), ring=ring_measure(sample.xy_um, sample.current_na))
        checked[tag] = record
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-dir", type=Path, help="Directory with existing half-step and half-count runs")
    args = parser.parse_args()
    summary = json.loads((EVIDENCE / "comparison.json").read_text())
    run = summary["runs"]["d95"]
    primary = EVIDENCE / run["trajectory_file"]
    digest = sha256(primary.read_bytes()).hexdigest()
    if digest != run["trajectory_sha256"] or summary["C2_diameter_um"] != 100:
        raise ValueError("Expected the archived 100 µm C2 aperture D95-focus checkpoint")
    sample = read_sample(primary, run, summary)
    basis, names = design(sample)
    all_rows = np.ones(len(sample.ids), dtype=bool)
    primary_fit = fit_record(sample, basis, names, all_rows)
    checks, removed, fold_coefficients = held_out(sample, basis, names)
    extended_basis, extended_names = design(sample, sixth_order=True)
    results = {
        "trajectory_file": run["trajectory_file"], "trajectory_sha256": digest,
        "source_digest": run["source_digest"], "corrector_circuit_sha256": summary["corrector_circuit_sha256"],
        "corrector_table_sha256": summary["corrector_table_sha256"],
        "winding_currents_file": run["winding_currents_file"],
        "winding_currents_sha256": sha256((EVIDENCE / run["winding_currents_file"]).read_bytes()).hexdigest(),
        "energy_keV": summary["energy_keV"], "C2_diameter_um": summary["C2_diameter_um"],
        "specimen_z_m": summary["specimen_upper_surface_z_m"], "source_A": summary["gun"]["gun_exit_A"],
        "emitted": run["emitted"], "survived": run["survived"], "effective": run["effective"],
        "surface": run["surface"], "chief_direction": sample.chief.tolist(),
        "max_angle_mrad": float(sample.alpha_mrad.max()), "angle_groups": grouped_current(sample),
        "angle_reference": "Actual incident mechanical direction relative to the current-weighted chief at the specimen",
        "convention": "W=C_nm alpha^(n+1) cos(m(phi-theta))/(n+1); r=offset-grad(W); theta from +x toward +y",
        "full_fit": primary_fit, "baseline_columns": names.tolist(),
        "fold_seed": FOLD_SEED, "held_out_models": checks, "training_fold_estimates": fold_coefficients,
        "ring_um": list(ANNULUS_UM), "observed_ring": ring_measure(sample.xy_um, sample.current_na),
        "ring_after_subtracting_held_out_term": {k: ring_measure(v, sample.current_na) for k, v in removed.items()},
        "cone_limit_sensitivity": {
            str(limit): fit_record(sample, basis, names, sample.alpha_mrad < limit) for limit in (20, 25, 30, 35)
        },
        "including_sixth_order_three_lobe_C63": fit_record(sample, extended_basis, extended_names, all_rows),
    }
    if args.controls_dir:
        results["native_controls"] = input_controls(args.controls_dir, summary, primary)
    from plot_aperture_angle_spots import plot_ablation, plot_groups

    plot_groups(sample, angle_groups(sample), results)
    plot_ablation(sample, removed, results)
    filename = "threefold-fit.json" if args.controls_dir else "threefold-fit-primary.json"
    (EVIDENCE / filename).write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
